import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, RotatingKVCache

from exo.worker.engines.mlx.cache import materialized_cache_states


def test_materialized_cache_states_skips_uninitialized_entries() -> None:
    cache = [RotatingKVCache(max_size=16), ArraysCache(size=1)]

    assert materialized_cache_states(cache) == []


def test_materialized_cache_states_keeps_initialized_entries() -> None:
    cache = [ArraysCache(size=1)]
    cache[0].state = [mx.array([1, 2, 3], dtype=mx.float32)]

    states = materialized_cache_states(cache)

    assert len(states) == 1
    assert states[0] is cache[0].state
