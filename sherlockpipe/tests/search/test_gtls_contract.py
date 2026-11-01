import subprocess
import sys

import pytest


def test_pypi_gtls_public_api_when_cuda_dependency_is_available():
    pytest.importorskip("gputls")
    from gputls import gtls

    assert callable(gtls)
    assert callable(gtls.power)


def test_importing_gpu_search_adapter_does_not_import_gtls_or_cupy():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import sherlockpipe.search.GtlsSearcher; "
            "assert 'gputls' not in sys.modules; "
            "assert 'cupy' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
