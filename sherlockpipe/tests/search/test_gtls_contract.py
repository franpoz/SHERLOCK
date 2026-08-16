import importlib.metadata

import pytest


def test_supported_pypi_version():
    try:
        version = importlib.metadata.version("gputls")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("gputls is only installed in the GPU test environment")
    assert version == "0.4.4"


def test_pypi_public_api():
    pytest.importorskip("gputls")
    from gputls import gtls

    assert callable(gtls)
    assert hasattr(gtls, "power")
