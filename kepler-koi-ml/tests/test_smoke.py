def test_utils_nasa_import():
    import utils.nasa as n

    assert hasattr(n, "clean_koi")
    assert isinstance(getattr(n, "FEATURE_COLS", []), list)
