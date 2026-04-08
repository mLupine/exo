from exo.worker.engines.mlx.utils_mlx import normalize_encoded_tokens


def test_normalize_encoded_tokens_keeps_plain_ints() -> None:
    assert normalize_encoded_tokens([1, 2, 3]) == [1, 2, 3]


def test_normalize_encoded_tokens_flattens_tuple_items() -> None:
    assert normalize_encoded_tokens([(1, 0), (2, 0), (3, 0)]) == [1, 2, 3]
