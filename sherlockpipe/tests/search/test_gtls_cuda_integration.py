"""Opt-in real GTLS searcher smoke test for CUDA-enabled runners."""

import os
import subprocess
import sys
import tempfile

import pytest


def test_gtls_searcher_recovers_synthetic_transit_on_cuda():
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("GTLS integration smoke requires an available CUDA device")
    except cupy.cuda.runtime.CUDARuntimeError as error:
        pytest.skip(f"GTLS integration smoke requires a working CUDA runtime: {error}")

    # A clean CuPy kernel cache avoids stale cubins from another CUDA/GPU stack.
    # Keep the actual search in a subprocess so CuPy reads this cache setting
    # before its first module import.
    smoke = r'''
from types import SimpleNamespace
import numpy as np
from sherlockpipe.search.GtlsSearcher import GtlsSearcher

true_period = 3.4
true_epoch = 0.73
duration = 0.16
time = np.arange(0.0, 30.0, 0.02)
phase = (time - true_epoch + true_period / 2) % true_period - true_period / 2
flux = np.ones_like(time)
flux[np.abs(phase) < duration / 2] -= 0.015
flux += np.random.default_rng(118).normal(0.0, 0.0005, len(time))
target = SimpleNamespace(
    search_engine="gpu", custom_transit_template=None, fit_method="box",
    period_min=2.0, period_max=5.0, t0_fit_margin=0.125,
    oversampling=3, duration_grid_step=1.1, gpu_device_id=0,
)
star = SimpleNamespace(
    radius=1.0, radius_min=0.9, radius_max=1.1,
    mass=1.0, mass_min=0.9, mass_max=1.1, ld_coefficients=None,
)
result = GtlsSearcher().search(
    target, time, flux, star, 2, None, None, 0.02,
    np.linspace(3.0, 3.8, 100), None)
assert abs(result.period - true_period) <= 0.02, result.period
assert np.isfinite(result.sde) and result.sde > 0, result.sde
assert result.count >= 5, result.count
'''
    environment = os.environ.copy()
    with tempfile.TemporaryDirectory(prefix="sherlock-gtls-cupy-") as cache:
        environment["CUPY_CACHE_DIR"] = cache
        completed = subprocess.run(
            [sys.executable, "-c", smoke], env=environment,
            capture_output=True, text=True, check=False,
        )
    assert completed.returncode == 0, completed.stdout + completed.stderr
