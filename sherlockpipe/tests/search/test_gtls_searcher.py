from types import SimpleNamespace
import sys
from types import ModuleType

import numpy as np
import pytest

from sherlockpipe.search.GtlsSearcher import GtlsSearcher


def _source_result():
    return SimpleNamespace(
        periods=np.array([1.0, 2.0, 3.0]),
        power=np.array([1.0, 2.0, 1.5]),
        chi2=np.array([3.0, 2.0, 2.5]),
        period=2.0,
        duration=0.2,
        T0=0.5,
        transitTimes=np.array([0.5, 2.5, 4.5]),
        depth=0.01,
        snr=10.0,
        SDE=8.0,
    )


def test_normalize_gtls_result_builds_sherlock_contract():
    time = np.arange(0.0, 6.0, 0.05)
    flux = np.ones_like(time)
    flux[np.abs((time - 0.5 + 1) % 2 - 1) < 0.1] -= 0.01

    results = GtlsSearcher()._normalize_results(_source_result(), time, flux, None)

    assert results.period == 2.0
    assert results.duration == 0.2
    assert results.transit_times.shape == (3,)
    assert results.transit_depths.shape == (3,)
    assert results.model_lightcurve_model.shape == time.shape
    assert results.folded_phase.shape == time.shape
    assert results.model_folded_phase.shape == results.model_folded_model.shape
    assert results.rp_rs == pytest.approx(0.1)


@pytest.mark.parametrize("fit_method, expected", [("default", "default"), ("box", "box"), ("grazing", "grazing")])
def test_supported_templates(fit_method, expected):
    assert GtlsSearcher._template(fit_method) == expected


def test_unsupported_template_is_explicit():
    with pytest.raises(ValueError, match="only the default, box, and grazing"):
        GtlsSearcher._template("tailed")


def test_search_translates_period_grid_and_gpu_options(monkeypatch):
    calls = {}

    class FakeModel:
        def __init__(self, **kwargs):
            calls["constructor"] = kwargs

        def power(self, **kwargs):
            calls["power"] = kwargs
            return _source_result()

    fake_module = ModuleType("gputls")
    fake_module.gtls = FakeModel
    monkeypatch.setitem(sys.modules, "gputls", fake_module)

    target = SimpleNamespace(
        search_engine="gpu", custom_transit_template=None, fit_method="default",
        period_min=1, period_max=3, t0_fit_margin=0.05, oversampling=3,
        duration_grid_step=1.1, gpu_device_id=2,
    )
    star = SimpleNamespace(
        radius=1.0, radius_min=0.9, radius_max=1.1, mass=1.0,
        mass_min=0.9, mass_max=1.1, ld_coefficients=None,
    )
    time = np.arange(0.0, 6.0, 0.05)
    flux = np.ones_like(time)
    result = GtlsSearcher().search(target, time, flux, star, 2, None, None, 120, [1, 2, 3], None)

    assert len(calls["power"]["periods"]) == 100
    assert calls["power"]["periods"][0] == 1
    assert calls["power"]["periods"][-1] == 3
    assert calls["power"]["GPUDeviceID"] == 2
    assert calls["power"]["transit_template"] == "default"
    assert result.mode == "tls"
