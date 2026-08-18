from sherlockpipe.search.GtlsSearcher import GtlsSearcher
from sherlockpipe.search.TlsSearcher import TlsSearcher
from sherlockpipe.search.sherlock_target import SherlockTarget
import numpy as np


def test_cpu_uses_foldedleastsquares_searcher():
    target = SherlockTarget(object_info=None, search_engine="cpu")
    assert isinstance(target.searchers["default"], TlsSearcher)


def test_gpu_uses_gtls_searcher_without_importing_gtls():
    target = SherlockTarget(object_info=None, search_engine="gpu")
    assert isinstance(target.searchers["default"], GtlsSearcher)


def test_gpu_approximate_is_rejected():
    try:
        SherlockTarget(object_info=None, search_engine="gpu_approximate")
    except ValueError as error:
        assert "not supported by GTLS" in str(error)
    else:
        raise AssertionError("gpu_approximate must not select an incomplete backend")


def test_gtls_depth_error_is_reported_in_ppt():
    errors = np.array([0.0001, 0.0002])
    assert GtlsSearcher._depth_error_ppt(errors) == np.sqrt(0.0001 ** 2 + 0.0002 ** 2) * 1000 / 2
