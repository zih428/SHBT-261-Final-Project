from textvqa_proj.data.normalization import normalize_answer, normalize_tokens


def test_normalize_answer_collapses_case_and_space() -> None:
    assert normalize_answer("  HELLO   World ") == "hello world"


def test_normalize_tokens_drops_empty_values() -> None:
    assert normalize_tokens([" A ", " ", "\nB"]) == ["a", "b"]
