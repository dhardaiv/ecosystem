import pytest

_wm3_available = True
_losses_available = True

try:
    import wm3
except ImportError:
    _wm3_available = False

try:
    import losses
except ImportError:
    _losses_available = False


def pytest_collection_modifyitems(config, items):
    if not _wm3_available or not _losses_available:
        skip = pytest.mark.skip(
            reason="wm3 and/or losses modules not yet available"
        )
        for item in items:
            item.add_marker(skip)
