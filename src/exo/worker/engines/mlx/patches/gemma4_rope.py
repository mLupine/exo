from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import gemma4_text, rope_utils

_patched = False


class ProportionalRoPE(nn.Module):
    def __init__(
        self,
        dims: int,
        rotated_dims: int,
        traditional: bool = False,
        base: float = 10000.0,
        factor: float = 1.0,
    ):
        super().__init__()
        self.dims = dims
        self.traditional = traditional

        if rotated_dims > dims:
            raise ValueError("rotated_dims should be smaller than dims")

        exponents = mx.arange(0, rotated_dims, 2, dtype=mx.float32) / dims
        self._freqs = mx.concatenate(
            [
                factor * (base**exponents),
                mx.full(((dims - rotated_dims) // 2,), mx.inf),
            ]
        )

    def __call__(self, x, offset=0):
        return mx.fast.rope(
            x,
            self.dims,
            traditional=self.traditional,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self._freqs,
        )


def patch_gemma4_rope() -> None:
    global _patched
    if _patched:
        return
    _patched = True

    original_initialize_rope = rope_utils.initialize_rope

    def patched_initialize_rope(
        dims,
        base,
        traditional,
        scaling_config=None,
        max_position_embeddings=None,
    ):
        rope_type = (
            (scaling_config or {}).get("type")
            or (scaling_config or {}).get("rope_type")
            or "default"
        )
        if rope_type == "proportional":
            return ProportionalRoPE(
                dims=dims,
                rotated_dims=int(dims * (scaling_config or {}).get("partial_rotary_factor", 1.0)),
                traditional=traditional,
                base=base,
                factor=(scaling_config or {}).get("factor", 1.0),
            )
        return original_initialize_rope(
            dims=dims,
            base=base,
            traditional=traditional,
            scaling_config=scaling_config,
            max_position_embeddings=max_position_embeddings,
        )

    rope_utils.initialize_rope = patched_initialize_rope
    gemma4_text.initialize_rope = patched_initialize_rope
