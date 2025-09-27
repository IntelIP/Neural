from importlib import import_module


def test_can_import_top_level() -> None:
    mod = import_module("neural")
    assert hasattr(mod, "__version__") or hasattr(mod, "__all__")
