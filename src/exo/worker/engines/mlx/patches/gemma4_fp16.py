from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import gemma4_text

_patched = False


def _cast_fp16(value: Any) -> Any:
    if isinstance(value, mx.array) and value.dtype == mx.bfloat16:
        return value.astype(mx.float16)
    if isinstance(value, tuple):
        return tuple(_cast_fp16(v) for v in value)
    if isinstance(value, list):
        return [_cast_fp16(v) for v in value]
    return value


def patch_gemma4_fp16() -> None:
    global _patched
    if _patched:
        return
    _patched = True

    original_attention_call = gemma4_text.Attention.__call__
    original_decoder_call = gemma4_text.DecoderLayer.__call__
    original_text_model_call = gemma4_text.Gemma4TextModel.__call__

    def safe_logit_softcap(softcap, x):
        x = _cast_fp16(x)
        return mx.tanh(x / softcap) * softcap

    def safe_geglu(gate, x):
        gate = _cast_fp16(gate)
        x = _cast_fp16(x)
        return nn.gelu_approx(gate) * x

    def safe_complete_square(x2, y2, xy):
        x2 = _cast_fp16(x2)
        y2 = _cast_fp16(y2)
        xy = _cast_fp16(xy)
        return x2 + mx.expand_dims(y2, -1) - 2 * xy

    gemma4_text.logit_softcap = safe_logit_softcap
    gemma4_text.geglu = safe_geglu
    gemma4_text._complete_square = safe_complete_square

    def patched_attention_call(self, x, mask=None, cache=None, shared_kv=None, offset=None):
        output = original_attention_call(
            self,
            _cast_fp16(x),
            mask=_cast_fp16(mask),
            cache=cache,
            shared_kv=_cast_fp16(shared_kv),
            offset=offset,
        )
        return _cast_fp16(output)

    def patched_decoder_call(
        self,
        x,
        mask=None,
        cache=None,
        per_layer_input=None,
        shared_kv=None,
        offset=None,
    ):
        output = original_decoder_call(
            self,
            _cast_fp16(x),
            mask=_cast_fp16(mask),
            cache=cache,
            per_layer_input=_cast_fp16(per_layer_input),
            shared_kv=_cast_fp16(shared_kv),
            offset=offset,
        )
        return _cast_fp16(output)

    def patched_text_model_call(
        self,
        inputs=None,
        cache=None,
        input_embeddings=None,
        per_layer_inputs=None,
    ):
        output = original_text_model_call(
            self,
            inputs=inputs,
            cache=cache,
            input_embeddings=_cast_fp16(input_embeddings),
            per_layer_inputs=_cast_fp16(per_layer_inputs),
        )
        return _cast_fp16(output)

    def patched_model_call(
        self,
        inputs,
        cache=None,
        input_embeddings=None,
        per_layer_inputs=None,
    ):
        out = self.model(
            inputs,
            cache=cache,
            input_embeddings=_cast_fp16(input_embeddings),
            per_layer_inputs=_cast_fp16(per_layer_inputs),
        )
        out = _cast_fp16(out)
        if self.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        out = _cast_fp16(out)
        if self.final_logit_softcapping is not None:
            out = safe_logit_softcap(self.final_logit_softcapping, out)
        return _cast_fp16(out)

    gemma4_text.Attention.__call__ = patched_attention_call
    gemma4_text.DecoderLayer.__call__ = patched_decoder_call
    gemma4_text.Gemma4TextModel.__call__ = patched_text_model_call
    gemma4_text.Model.__call__ = patched_model_call
