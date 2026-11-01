import subprocess
import sys
from pathlib import Path

import numpy as np

from sherlockpipe.exomoon import ExoMoonLeastSquares


def test_identical_observation_times_receive_identical_template_flux():
    """The model prediction must depend on time, not sample rank."""
    times = np.array([0.0, 0.0, 0.0])
    flux = np.array([0.9, 0.9, 0.9])
    flux_err = np.array([0.01, 0.01, 0.01])
    template = np.array([1.0, 0.8, 1.0])

    _, _, _, model = ExoMoonLeastSquares.calculate_residuals(
        it_time=times,
        it_flux=flux,
        it_flux_err=flux_err,
        oot_time=np.array([]),
        oot_flux=np.array([]),
        oot_flux_err=np.array([]),
        t0_flux_mean_value=0.9,
        datapoints=len(times),
        models=[(1.0, template)],
        min_radius=1.0,
        max_radius=100.0,
        star_radius=1.0,
        radius_grid_size=1,
    )

    np.testing.assert_allclose(model, np.full(len(times), model[0]))


def test_distribution_metadata_pins_pandoramoon_13(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    subprocess.run(
        [sys.executable, "setup.py", "egg_info", "--egg-base", str(tmp_path)],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    metadata = tmp_path / "sherlockpipe.egg-info" / "PKG-INFO"
    assert "Requires-Dist: pandoramoon==1.3" in metadata.read_text()
