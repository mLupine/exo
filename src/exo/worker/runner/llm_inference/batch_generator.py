import itertools
import os
import socket
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Generator, Iterable
from dataclasses import dataclass, field

import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.constants import EXO_MAX_CONCURRENT_REQUESTS
from exo.shared.types.chunks import ErrorChunk, PrefillProgressChunk
from exo.shared.types.common import ModelId
from exo.shared.types.events import ChunkGenerated, Event
from exo.shared.types.mlx import Model
from exo.shared.types.tasks import CANCEL_ALL_TASKS, TaskId, TextGeneration
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.runner_response import GenerationResponse, ToolCallResponse
from exo.utils.channels import MpReceiver, MpSender
from exo.worker.engines.mlx.cache import KVPrefixCache
from exo.worker.engines.mlx.generator.batch_generate import ExoBatchGenerator
from exo.worker.engines.mlx.generator.generate import (
    PrefillCancelled,
    mlx_generate,
    warmup_inference,
)
from exo.worker.engines.mlx.utils_mlx import (
    apply_chat_template,
    mx_all_gather_ints,
    mx_all_gather_tasks,
    mx_any,
    mx_count_true,
)
from exo.worker.engines.mlx.vision import VisionProcessor
from exo.worker.runner.bootstrap import logger

from .model_output_parsers import apply_all_parsers
from .tool_parsers import ToolParser


def _gemma4_seq_debug(model_id: ModelId, message: str) -> None:
    model_id_str = str(model_id).lower()
    if "gemma-4" not in model_id_str and "gemma4" not in model_id_str:
        return
    path = os.environ.get("EXO_GEMMA4_DEBUG_LOG", "/tmp/exo-gemma4-debug.log")
    try:
        with open(path, "a", encoding="utf-8") as f:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            f.write(
                f"{ts} host={socket.gethostname()} pid={os.getpid()} seq {message}\n"
            )
    except Exception:
        pass


class Cancelled:
    pass


class Finished:
    pass


class GeneratorQueue[T]:
    def __init__(self):
        self._q = deque[T]()

    def push(self, t: T):
        self._q.append(t)

    def gen(self) -> Generator[T | None]:
        while True:
            if len(self._q) == 0:
                yield None
            else:
                yield self._q.popleft()


class InferenceGenerator(ABC):
    _cancelled_tasks: set[TaskId]

    def should_cancel(self, task_id: TaskId) -> bool:
        return (
            task_id in self._cancelled_tasks
            or CANCEL_ALL_TASKS in self._cancelled_tasks
        )

    def take_cancellations(self) -> list[tuple[TaskId, "Cancelled"]]:
        cancellations = [(task_id, Cancelled()) for task_id in self._cancelled_tasks]
        self._cancelled_tasks.clear()
        return cancellations

    @abstractmethod
    def warmup(self) -> None: ...

    @abstractmethod
    def submit(
        self,
        task: TextGeneration,
    ) -> None: ...

    @abstractmethod
    def step(
        self,
    ) -> Iterable[
        tuple[TaskId, ToolCallResponse | GenerationResponse | Cancelled | Finished]
    ]: ...

    @abstractmethod
    def close(self) -> None: ...


EXO_RUNNER_MUST_FAIL = "EXO RUNNER MUST FAIL"
EXO_RUNNER_MUST_OOM = "EXO RUNNER MUST OOM"
EXO_RUNNER_MUST_TIMEOUT = "EXO RUNNER MUST TIMEOUT"


def _check_for_debug_prompts(task_params: TextGenerationTaskParams) -> None:
    """Check for debug prompt triggers in the input."""
    from exo.worker.engines.mlx.utils_mlx import mlx_force_oom

    if len(task_params.input) == 0:
        return
    prompt = task_params.input[0].content
    if not prompt:
        return
    if EXO_RUNNER_MUST_FAIL in prompt:
        raise Exception("Artificial runner exception - for testing purposes only.")
    if EXO_RUNNER_MUST_OOM in prompt:
        mlx_force_oom()
    if EXO_RUNNER_MUST_TIMEOUT in prompt:
        time.sleep(100)


@dataclass(eq=False)
class SequentialGenerator(InferenceGenerator):
    model: Model
    tokenizer: TokenizerWrapper
    group: mx.distributed.Group | None
    kv_prefix_cache: KVPrefixCache | None
    tool_parser: ToolParser | None
    model_id: ModelId
    device_rank: int
    cancel_receiver: MpReceiver[TaskId]
    event_sender: MpSender[Event]
    vision_processor: VisionProcessor | None = None
    check_for_cancel_every: int = 50

    _cancelled_tasks: set[TaskId] = field(default_factory=set, init=False)
    _maybe_queue: list[TextGeneration] = field(default_factory=list, init=False)
    _maybe_cancel: list[TextGeneration] = field(default_factory=list, init=False)
    _all_tasks: dict[TaskId, TextGeneration] = field(default_factory=dict, init=False)
    _queue: deque[TextGeneration] = field(default_factory=deque, init=False)
    _active: (
        tuple[
            TextGeneration,
            # mlx generator that does work
            Generator[GenerationResponse],
            # queue that the 1st generator should push to and 3rd generator should pull from
            GeneratorQueue[GenerationResponse],
            # generator to get parsed outputs
            Generator[GenerationResponse | ToolCallResponse | None],
        ]
        | None
    ) = field(default=None, init=False)

    def warmup(self):
        self.check_for_cancel_every = warmup_inference(
            model=self.model,
            tokenizer=self.tokenizer,
            group=self.group,
            model_id=self.model_id,
        )

    def _drain_stale_startup_cancellations(self) -> None:
        if self._active is not None or self._all_tasks or self._queue or self._maybe_queue:
            return

        stale = self.cancel_receiver.collect()
        if stale:
            logger.warning(
                f"Dropping stale startup cancellations before first task: {stale!r}"
            )

    def submit(
        self,
        task: TextGeneration,
    ) -> None:
        self._drain_stale_startup_cancellations()
        self._cancelled_tasks.discard(CANCEL_ALL_TASKS)
        self._all_tasks[task.task_id] = task
        self._maybe_queue.append(task)

    def agree_on_tasks(self) -> None:
        """Agree between all ranks about the task ordering (some may have received in different order or not at all)."""
        agreed, different = mx_all_gather_tasks(self._maybe_queue, self.group)
        self._queue.extend(task for task in self._maybe_queue if task in agreed)
        self._maybe_queue = [task for task in self._maybe_queue if task in different]

    def agree_on_cancellations(self) -> None:
        """Agree between all ranks about which tasks to cancel."""
        has_cancel_all = False
        collected = self.cancel_receiver.collect()
        _gemma4_seq_debug(
            self.model_id,
            f"agree_cancel_enter task_id={self._active[0].task_id if self._active else None} collected={collected!r} maybe_cancel={[task.task_id for task in self._maybe_cancel]!r} cancelled={list(self._cancelled_tasks)!r}",
        )
        if collected:
            logger.warning(
                f"SequentialGenerator collected cancellations active={self._active[0].task_id if self._active else None} known={list(self._all_tasks.keys())} ids={collected!r}"
            )
        for task_id in collected:
            if task_id == CANCEL_ALL_TASKS:
                has_cancel_all = True
                continue
            if task_id in self._all_tasks:
                self._maybe_cancel.append(self._all_tasks[task_id])

        gathered_cancel_flags = mx_all_gather_ints(1 if has_cancel_all else 0, self.group)
        cancel_all_count = mx_count_true(has_cancel_all, self.group)
        cancel_all_agreed = cancel_all_count > 0
        _gemma4_seq_debug(
            self.model_id,
            f"agree_cancel_after_all_sum task_id={self._active[0].task_id if self._active else None} local_has_cancel_all={has_cancel_all} gathered_cancel_flags={gathered_cancel_flags!r} cancel_all_count={cancel_all_count} group_size={self.group.size() if self.group is not None else 1}",
        )
        if cancel_all_agreed and not has_cancel_all:
            logger.warning(
                "CANCEL_ALL_TASKS observed from another rank during SequentialGenerator.agree_on_cancellations"
            )
        if cancel_all_agreed:
            self._cancelled_tasks.add(CANCEL_ALL_TASKS)

        agreed, different = mx_all_gather_tasks(self._maybe_cancel, self.group)
        self._cancelled_tasks.update(task.task_id for task in agreed)
        self._maybe_cancel = list(different)
        _gemma4_seq_debug(
            self.model_id,
            f"agree_cancel_exit task_id={self._active[0].task_id if self._active else None} agreed_cancel={[task.task_id for task in agreed]!r} remaining_maybe_cancel={[task.task_id for task in self._maybe_cancel]!r} cancelled={list(self._cancelled_tasks)!r}",
        )

    def step(
        self,
    ) -> Iterable[
        tuple[TaskId, GenerationResponse | ToolCallResponse | Cancelled | Finished]
    ]:
        if self._active is None:
            self.agree_on_tasks()

            if self._queue:
                self._start_next()
            else:
                return self.take_cancellations()

        assert self._active is not None

        task, mlx_gen, queue, output_generator = self._active
        _gemma4_seq_debug(
            self.model_id,
            f"step_enter task_id={task.task_id} command_id={task.command_id}",
        )
        output: list[
            tuple[TaskId, GenerationResponse | ToolCallResponse | Cancelled | Finished]
        ] = []
        try:
            _gemma4_seq_debug(
                self.model_id,
                f"next_start task_id={task.task_id} command_id={task.command_id}",
            )
            response = next(mlx_gen)
            _gemma4_seq_debug(
                self.model_id,
                f"next_return finish_reason={response.finish_reason!r} text_len={len(response.text)} task_id={task.task_id}",
            )
            queue.push(response)
            # drain potentially many responses every time
            while (parsed := next(output_generator, None)) is not None:
                output.append((task.task_id, parsed))

        except PrefillCancelled:
            _gemma4_seq_debug(
                self.model_id,
                f"prefill_cancelled task_id={task.task_id}",
            )
            output.append((task.task_id, Cancelled()))
            self._active = None
            if self._queue:
                self._start_next()

        except StopIteration:
            _gemma4_seq_debug(
                self.model_id,
                f"terminal_without_response task_id={task.task_id}",
            )
            while (parsed := next(output_generator, None)) is not None:
                output.append((task.task_id, parsed))

            if not output:
                output.append(
                    (
                        task.task_id,
                        GenerationResponse(
                            text="",
                            token=0,
                            finish_reason="stop",
                            usage=None,
                        ),
                    )
                )

            output.append((task.task_id, Finished()))
            self._active = None
            if self._queue:
                self._start_next()

        except Exception as e:
            self._send_error(task, e)
            self._active = None
            raise

        return itertools.chain(output, self.take_cancellations())

    def _start_next(self) -> None:
        task = self._queue.popleft()
        try:
            mlx_gen = self._build_generator(task)
        except Exception as e:
            self._send_error(task, e)
            raise
        queue = GeneratorQueue[GenerationResponse]()

        if task.task_params.bench:
            output_generator = queue.gen()
        else:
            output_generator = apply_all_parsers(
                queue.gen(),
                apply_chat_template(self.tokenizer, task.task_params),
                self.tool_parser,
                self.tokenizer,
                type(self.model),
                self.model_id,
                task.task_params.tools,
            )
        self._active = (task, mlx_gen, queue, output_generator)

    def _send_error(self, task: TextGeneration, e: Exception) -> None:
        if self.device_rank == 0:
            self.event_sender.send(
                ChunkGenerated(
                    command_id=task.command_id,
                    chunk=ErrorChunk(
                        model=self.model_id,
                        finish_reason="error",
                        error_message=str(e),
                    ),
                )
            )

    def _build_generator(self, task: TextGeneration) -> Generator[GenerationResponse]:
        _check_for_debug_prompts(task.task_params)
        _gemma4_seq_debug(
            self.model_id,
            f"build_generator task_id={task.task_id} command_id={task.command_id}",
        )
        prompt = apply_chat_template(self.tokenizer, task.task_params)

        model_id_str = str(self.model_id).lower()
        disable_prefill_progress = "gemma-4" in model_id_str or "gemma4" in model_id_str

        def on_prefill_progress(processed: int, total: int) -> None:
            if disable_prefill_progress:
                return
            if self.device_rank == 0:
                self.event_sender.send(
                    ChunkGenerated(
                        command_id=task.command_id,
                        chunk=PrefillProgressChunk(
                            model=self.model_id,
                            processed_tokens=processed,
                            total_tokens=total,
                        ),
                    )
                )

        def distributed_prompt_progress_callback() -> None:
            self.agree_on_cancellations()
            if self.should_cancel(task.task_id):
                raise PrefillCancelled()

            self.agree_on_tasks()

        tokens_since_cancel_check = self.check_for_cancel_every

        def on_generation_token() -> None:
            nonlocal tokens_since_cancel_check
            tokens_since_cancel_check += 1
            if tokens_since_cancel_check >= self.check_for_cancel_every:
                tokens_since_cancel_check = 0
                self.agree_on_cancellations()
                if self.should_cancel(task.task_id):
                    raise PrefillCancelled()

                self.agree_on_tasks()

        return mlx_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            task=task.task_params,
            prompt=prompt,
            kv_prefix_cache=self.kv_prefix_cache,
            on_prefill_progress=on_prefill_progress,
            distributed_prompt_progress_callback=distributed_prompt_progress_callback,
            on_generation_token=on_generation_token,
            group=self.group,
            vision_processor=self.vision_processor,
        )

    def close(self) -> None:
        del self.model, self.tokenizer, self.group


@dataclass(eq=False)
class BatchGenerator(InferenceGenerator):
    model: Model
    tokenizer: TokenizerWrapper
    group: mx.distributed.Group | None
    kv_prefix_cache: KVPrefixCache | None
    tool_parser: ToolParser | None
    model_id: ModelId
    device_rank: int
    cancel_receiver: MpReceiver[TaskId]
    event_sender: MpSender[Event]
    check_for_cancel_every: int = 50
    vision_processor: VisionProcessor | None = None

    _cancelled_tasks: set[TaskId] = field(default_factory=set, init=False)
    _maybe_queue: list[TextGeneration] = field(default_factory=list, init=False)
    _maybe_cancel: list[TextGeneration] = field(default_factory=list, init=False)
    _all_tasks: dict[TaskId, TextGeneration] = field(default_factory=dict, init=False)
    _queue: deque[TextGeneration] = field(default_factory=deque, init=False)
    _mlx_gen: ExoBatchGenerator = field(init=False)
    _active_tasks: dict[
        int,
        tuple[
            TextGeneration,
            GeneratorQueue[GenerationResponse],
            Generator[GenerationResponse | ToolCallResponse | None],
            int,
        ],
    ] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._mlx_gen = ExoBatchGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            group=self.group,
            kv_prefix_cache=self.kv_prefix_cache,
            vision_processor=self.vision_processor,
        )

    def warmup(self):
        self.check_for_cancel_every = warmup_inference(
            model=self.model,
            tokenizer=self.tokenizer,
            group=self.group,
            model_id=self.model_id,
        )

    def submit(
        self,
        task: TextGeneration,
    ) -> None:
        self._cancelled_tasks.discard(CANCEL_ALL_TASKS)
        self._all_tasks[task.task_id] = task
        self._maybe_queue.append(task)

    def agree_on_tasks(self) -> None:
        """Agree between all ranks about the task ordering (some may have received in different order or not at all)."""
        agreed, different = mx_all_gather_tasks(self._maybe_queue, self.group)
        self._queue.extend(task for task in self._maybe_queue if task in agreed)
        self._maybe_queue = [task for task in self._maybe_queue if task in different]

    def agree_on_cancellations(self) -> None:
        """Agree between all ranks about which tasks to cancel."""
        has_cancel_all = False
        collected = self.cancel_receiver.collect()
        if collected:
            logger.warning(
                f"BatchGenerator collected cancellations active_uids={list(self._active_tasks.keys())} known={list(self._all_tasks.keys())} ids={collected!r}"
            )
        for task_id in collected:
            if task_id == CANCEL_ALL_TASKS:
                has_cancel_all = True
                continue
            if task_id in self._all_tasks:
                self._maybe_cancel.append(self._all_tasks[task_id])

        cancel_all_agreed = mx_any(has_cancel_all, self.group)
        if cancel_all_agreed and not has_cancel_all:
            logger.warning(
                "CANCEL_ALL_TASKS observed from another rank during BatchGenerator.agree_on_cancellations"
            )
        if cancel_all_agreed:
            self._cancelled_tasks.add(CANCEL_ALL_TASKS)

        agreed, different = mx_all_gather_tasks(self._maybe_cancel, self.group)
        self._cancelled_tasks.update(task.task_id for task in agreed)
        self._maybe_cancel = list(different)

    def step(
        self,
    ) -> Iterable[
        tuple[TaskId, GenerationResponse | ToolCallResponse | Cancelled | Finished]
    ]:
        if not self._queue:
            self.agree_on_tasks()

        # Submit any queued tasks to the engine
        while self._queue and len(self._active_tasks) < EXO_MAX_CONCURRENT_REQUESTS:
            task = self._queue.popleft()
            try:
                uid = self._start_task(task)
            except PrefillCancelled:
                continue
            except Exception as e:
                self._send_error(task, e)
                raise

            queue = GeneratorQueue[GenerationResponse]()
            if task.task_params.bench:
                output_generator = queue.gen()
            else:
                output_generator = apply_all_parsers(
                    queue.gen(),
                    apply_chat_template(self.tokenizer, task.task_params),
                    self.tool_parser,
                    self.tokenizer,
                    type(self.model),
                    self.model_id,
                    task.task_params.tools,
                )
            self._active_tasks[uid] = (task, queue, output_generator, 0)

        if not self._mlx_gen.has_work:
            return self._apply_cancellations()

        results = self._mlx_gen.step()

        output: list[
            tuple[TaskId, GenerationResponse | ToolCallResponse | Cancelled | Finished]
        ] = []
        for uid, response in results:
            if uid not in self._active_tasks:
                # should we error here?
                logger.warning(f"{uid=} not found in active tasks")
                continue

            task, queue, output_generator, emitted_count = self._active_tasks[uid]
            queue.push(response)
            emitted_this_step = 0
            # If a generator fails to parse for some reason and returns early, we should not crash
            while (parsed := next(output_generator, None)) is not None:
                output.append((task.task_id, parsed))
                emitted_this_step += 1

            total_emitted = emitted_count + emitted_this_step

            # If parsers swallowed the whole stream, fall back to the raw terminal response
            # so the API never completes with an empty 200/bodyless success.
            if response.finish_reason is not None and total_emitted == 0:
                logger.warning(
                    f"No parsed chunks emitted for terminal response on {task.task_id=}; "
                    "falling back to raw response"
                )
                output.append((task.task_id, response))
                total_emitted += 1

            self._active_tasks[uid] = (task, queue, output_generator, total_emitted)

            # check if original response was terminal and append a Finished()
            if response.finish_reason is not None:
                output.append((task.task_id, Finished()))
                del self._active_tasks[uid]

        return itertools.chain(output, self._apply_cancellations())

    def _apply_cancellations(
        self,
    ) -> list[tuple[TaskId, Cancelled]]:
        if not self._cancelled_tasks:
            return []

        cancel_all = CANCEL_ALL_TASKS in self._cancelled_tasks

        uids_to_cancel: list[int] = []
        results: list[tuple[TaskId, Cancelled]] = []

        for uid, (task, _, _, _) in list(self._active_tasks.items()):
            if task.task_id in self._cancelled_tasks or cancel_all:
                uids_to_cancel.append(uid)
                results.append((task.task_id, Cancelled()))
                del self._active_tasks[uid]

        if uids_to_cancel:
            self._mlx_gen.cancel(uids_to_cancel)

        already_cancelled = {tid for tid, _ in results}
        for tid in self._cancelled_tasks:
            if tid != CANCEL_ALL_TASKS and tid not in already_cancelled:
                results.append((tid, Cancelled()))

        self._cancelled_tasks.clear()
        return results

    def _send_error(self, task: TextGeneration, e: Exception) -> None:
        if self.device_rank == 0:
            self.event_sender.send(
                ChunkGenerated(
                    command_id=task.command_id,
                    chunk=ErrorChunk(
                        model=self.model_id,
                        finish_reason="error",
                        error_message=str(e),
                    ),
                )
            )

    def _start_task(self, task: TextGeneration) -> int:
        _check_for_debug_prompts(task.task_params)
        prompt = apply_chat_template(self.tokenizer, task.task_params)

        model_id_str = str(self.model_id).lower()
        disable_prefill_progress = "gemma-4" in model_id_str or "gemma4" in model_id_str

        def on_prefill_progress(processed: int, total: int) -> None:
            if disable_prefill_progress:
                return
            if self.device_rank == 0:
                self.event_sender.send(
                    ChunkGenerated(
                        command_id=task.command_id,
                        chunk=PrefillProgressChunk(
                            model=self.model_id,
                            processed_tokens=processed,
                            total_tokens=total,
                        ),
                    )
                )

        def distributed_prompt_progress_callback() -> None:
            self.agree_on_cancellations()
            if self.should_cancel(task.task_id):
                raise PrefillCancelled()

            self.agree_on_tasks()

        tokens_since_cancel_check = self.check_for_cancel_every

        def on_generation_token() -> None:
            nonlocal tokens_since_cancel_check
            tokens_since_cancel_check += 1
            if tokens_since_cancel_check >= self.check_for_cancel_every:
                tokens_since_cancel_check = 0
                self.agree_on_cancellations()
                if self.should_cancel(task.task_id):
                    self._cancelled_tasks.add(task.task_id)

                self.agree_on_tasks()

        return self._mlx_gen.submit(
            task_params=task.task_params,
            prompt=prompt,
            on_prefill_progress=on_prefill_progress,
            distributed_prompt_progress_callback=distributed_prompt_progress_callback,
            on_generation_token=on_generation_token,
        )

    def close(self) -> None:
        self._mlx_gen.close()
        del self.model, self.tokenizer, self.group
