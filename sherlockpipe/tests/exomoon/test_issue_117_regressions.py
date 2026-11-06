import logging
import subprocess
import sys
from pathlib import Path

import batman
import numpy as np
import pytest

from sherlockpipe.exomoon import AU_TO_RSUN, ExoMoonLeastSquares, R_earth_to_R_sun, SearchInput

V_TEMPLATE_TIME = np.array([-1.0, 0.0, 1.0])
V_TEMPLATE_FLUX = np.array([1.0, 0.8, 1.0])
PERIOD = 10.0
A_OVER_RSTAR = 20.0


def evaluate_model(times, t0_flux_mean_value=0.9, template_time=V_TEMPLATE_TIME, template_flux=V_TEMPLATE_FLUX):
    """Return the template fitted to flat data at the given timestamps."""
    times = np.asarray(times, dtype=float)
    return ExoMoonLeastSquares.calculate_residuals(
        it_time=times,
        it_flux=np.full(len(times), t0_flux_mean_value),
        it_flux_err=np.full(len(times), 0.01),
        oot_time=np.array([]),
        oot_flux=np.array([]),
        oot_flux_err=np.array([]),
        t0_flux_mean_value=t0_flux_mean_value,
        datapoints=len(times),
        models=[(1.0, template_time, template_flux)],
        min_radius=1.0,
        max_radius=100.0,
        star_radius=1.0,
        radius_grid_size=1,
    )


def v_shape(times):
    """Closed form of the V template scaled so that its trough is 0.9."""
    return 1 - 0.1 * np.clip(1 - np.abs(times), 0, None)


def transit_params(inc=90.0):
    params = batman.TransitParams()
    params.t0 = 10
    params.per = PERIOD
    params.a = A_OVER_RSTAR
    params.inc = inc
    params.ecc = 0.0
    params.w = 0
    params.u = [0.3, 0.2]
    params.limb_dark = "quadratic"
    return params


def total_duration(radius_earth, impact_param):
    """Circular-orbit contact-to-contact duration in days."""
    k = radius_earth * R_earth_to_R_sun
    inc = np.arccos(impact_param / A_OVER_RSTAR)
    return PERIOD / np.pi * np.arcsin(np.sqrt((1 + k) ** 2 - impact_param ** 2) / (A_OVER_RSTAR * np.sin(inc)))


def test_identical_observation_times_receive_identical_template_flux():
    """The model prediction must depend on time, not sample rank."""
    model = evaluate_model([0.0, 0.0, 0.0])[3]

    np.testing.assert_allclose(model, [0.9, 0.9, 0.9])


def test_template_is_evaluated_at_physical_timestamps():
    model = evaluate_model([-2.0, -0.5, 0.0, 0.5, 2.0])[3]

    np.testing.assert_allclose(model, [1.0, 0.95, 0.9, 0.95, 1.0])


def test_flux_is_unity_outside_template_support():
    model = evaluate_model([-2.0, 0.0, 2.0], template_flux=np.array([0.9, 0.8, 0.9]))[3]

    np.testing.assert_allclose(model, [1.0, 0.9, 1.0])


def test_gaps_do_not_compress_templates():
    times = np.linspace(-1.5, 1.5, 61)
    retained = (times < 0.2) | (times > 0.9)

    full_model = evaluate_model(times)[3]
    gapped_model = evaluate_model(times[retained])[3]

    np.testing.assert_allclose(full_model, v_shape(times), atol=1e-12)
    np.testing.assert_allclose(gapped_model, v_shape(times[retained]), atol=1e-12)


def test_dip_without_template_depth_is_reported_as_unusable():
    residual, _, _, model, template_unusable = evaluate_model([-3.0, 2.0, 3.0])

    assert template_unusable
    assert residual == ExoMoonLeastSquares.MAX_RESIDUAL
    np.testing.assert_allclose(model, 1.0)


def test_scenario_without_dip_is_not_reported_as_unusable():
    residual, _, _, _, template_unusable = evaluate_model([-0.5, 0.0, 0.5], t0_flux_mean_value=1.0)

    assert not template_unusable
    assert residual == ExoMoonLeastSquares.MAX_RESIDUAL


def test_fitted_scenario_is_not_reported_as_unusable():
    assert not evaluate_model([-0.5, 0.0, 0.5])[4]


@pytest.mark.parametrize("template_time, template_flux", [
    (np.array([-1.0, 1.0]), np.array([0.9, 0.8, 0.9])),
    (np.array([0.0]), np.array([0.9])),
])
def test_malformed_template_raises(template_time, template_flux):
    with pytest.raises(ValueError, match="aligned"):
        evaluate_model([0.0], template_time=template_time, template_flux=template_flux)


def test_templates_are_centred_on_zero_in_days():
    radius = 8.0
    radii = np.array([radius])

    _, template_time, template_flux = ExoMoonLeastSquares.build_moon_templates(transit_params(), radii, 1.0, 0.5)[0]

    contact_time = np.abs(template_time[template_flux < 1])
    grid_step = template_time[1] - template_time[0]
    assert abs(template_time[np.argmin(template_flux)]) <= grid_step
    np.testing.assert_allclose(template_time[0], -template_time[-1])
    assert contact_time.max() == pytest.approx(total_duration(radius, 0.0) / 2, abs=2 * grid_step)


@pytest.mark.parametrize("impact_param", [0.0, 0.8, 0.9])
def test_templates_reach_unity_inside_their_support(impact_param):
    radii = np.array([0.5, 4.0])
    inc = np.rad2deg(np.arccos(impact_param / A_OVER_RSTAR))
    # A half-window as short as 0.7 of a duration measured for b=0.99.
    min_half_window = 0.7 * total_duration(0.5, 0.99)

    models = ExoMoonLeastSquares.build_moon_templates(transit_params(inc), radii, 1.0, min_half_window)

    assert len(models) == 2
    for _, template_time, template_flux in models:
        assert template_flux[0] == 1.0
        assert template_flux[-1] == 1.0
        assert template_time[-1] > total_duration(4.0, impact_param) / 2
        assert template_time[1] - template_time[0] <= 2 * min_half_window / 999 * (1 + 1e-9)


def test_template_window_is_not_narrowed_when_it_already_covers_the_transit():
    models = ExoMoonLeastSquares.build_moon_templates(transit_params(), np.array([1.0]), 1.0, 1.5)

    assert models[0][1][-1] == pytest.approx(1.5)
    assert len(models[0][1]) == ExoMoonLeastSquares.TEMPLATE_MIN_SAMPLES


def test_template_window_stops_at_half_the_planet_period():
    models = ExoMoonLeastSquares.build_moon_templates(transit_params(), np.array([1.0]), 1.0, 2 * PERIOD)

    assert models[0][1][-1] == pytest.approx(PERIOD / 2)


def build_search(tmp_path, moon_depth=0.0, flux_time=None):
    """Return a small search over a flat light curve with a centred dip."""
    t0s = np.arange(4) * PERIOD
    time = np.concatenate([t0 + np.arange(-1.0, 1.0, 10 / 1440) for t0 in t0s]) if flux_time is None else flux_time
    rng = np.random.default_rng(7)
    offset = (time + PERIOD / 2) % PERIOD - PERIOD / 2
    flux = 1 + rng.normal(0, 1e-5, len(time)) - moon_depth * (np.abs(offset) < 0.08)
    flux_err = np.full(len(time), 1e-5)
    return ExoMoonLeastSquares(
        str(tmp_path), 1, 1.0, 1.0, [0.3, 0.2], 4.0, 0.4, 20.0, PERIOD, 0.0, 0.2,
        A_OVER_RSTAR / AU_TO_RSUN, 90.0, 1.0, 0.0, 0.0, 90.0, 0.0, 0.3, 0.6, t0s, time, flux, flux_err, 5.5,
        period_grid_size=40, radius_grid_size=3, skip_fold_fit=True)


def scenario_input(half_transit=0.05):
    return SearchInput(1.0, 0.0, 0, 90, 0, 0.3, 0.0, 0.6, 2 * half_transit, 0.1, half_transit, 0.05, 0.3)


def test_search_reports_template_outcome(tmp_path):
    emls = build_search(tmp_path, moon_depth=1e-3)
    template_flux = 1 - (1 - V_TEMPLATE_FLUX) * 1e-3
    emls.models = [(0.5, V_TEMPLATE_TIME + 5.0, template_flux)]
    out_of_support = emls.search(scenario_input())
    emls.models = [(0.5, V_TEMPLATE_TIME, template_flux)]
    usable = emls.search(scenario_input())

    assert out_of_support[0] == ExoMoonLeastSquares.MAX_RESIDUAL
    assert out_of_support[4]
    assert usable[0] < ExoMoonLeastSquares.MAX_RESIDUAL
    assert not usable[4]


def run_without_planet_subtraction(emls, monkeypatch, caplog):
    monkeypatch.setattr(ExoMoonLeastSquares, "subtract_planet_transit",
                        lambda self, time, flux, flux_err, mode, plot: (time, flux, flux_err))
    with caplog.at_level(logging.WARNING):
        emls.run()


def test_run_builds_day_centred_templates(tmp_path, caplog, monkeypatch):
    emls = build_search(tmp_path, moon_depth=1e-3)

    run_without_planet_subtraction(emls, monkeypatch, caplog)

    template_time = emls.models[0][1]
    np.testing.assert_allclose(template_time[0], -template_time[-1])
    assert abs(template_time[np.argmin(emls.models[-1][2])]) <= template_time[1] - template_time[0]
    assert all(model[2][0] == 1.0 and model[2][-1] == 1.0 for model in emls.models)
    assert "no usable template" not in caplog.text


def test_run_counts_scenarios_without_a_usable_template(tmp_path, caplog, monkeypatch):
    emls = build_search(tmp_path, moon_depth=1e-3)
    template_flux = 1 - (1 - V_TEMPLATE_FLUX) * 1e-3
    monkeypatch.setattr(ExoMoonLeastSquares, "build_moon_templates",
                        lambda *args: [(0.5, V_TEMPLATE_TIME + 5.0, template_flux)])

    run_without_planet_subtraction(emls, monkeypatch, caplog)

    assert "scenarios had no usable template" in caplog.text
    count, total = caplog.text.split(" scenarios had no usable template")[0].split()[-1].split("/")
    assert 0 < int(count) <= int(total) == 40 * 25


def test_run_warns_when_falling_back_to_flat_template(tmp_path, caplog, monkeypatch):
    emls = build_search(tmp_path)
    monkeypatch.setattr(ExoMoonLeastSquares, "build_moon_templates", lambda *args: [])

    run_without_planet_subtraction(emls, monkeypatch, caplog)

    assert "flat fallback template" in caplog.text


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
