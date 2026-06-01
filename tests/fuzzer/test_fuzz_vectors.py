from src.fuzzer.fuzz_vectors import BASE_VECTORS


def test_base_vectors_count():
    assert len(BASE_VECTORS) == 10


def test_base_vectors_type():
    assert all(isinstance(v, str) for v in BASE_VECTORS)


def test_base_vectors_has_xss():
    assert any("script" in v for v in BASE_VECTORS)


def test_base_vectors_has_sql():
    assert any("OR" in v for v in BASE_VECTORS)


def test_base_vectors_has_empty():
    assert "" in BASE_VECTORS
