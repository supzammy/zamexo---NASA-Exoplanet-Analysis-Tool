import pytest

features = pytest.importorskip("utils.features")
_nasa = pytest.importorskip("utils.nasa")


def _norm_local(s: str) -> str:
    return s.lower().replace(" ", "").replace("-", "")


def test_norm_name_consistency_when_available():
    names = ["Kepler-10", "Kepler 10", "TRAPPIST-1", "HD 209458", "Kepler-186"]
    f_norm = getattr(features, "norm_name", None)
    n_norm = getattr(_nasa, "norm_name", None)
    if f_norm is None and n_norm is None:
        pytest.skip("No norm_name in either module")
    for raw in names:
        local = _norm_local(raw)
        if f_norm:
            assert f_norm(raw) == local
        if n_norm:
            assert n_norm(raw) == local
