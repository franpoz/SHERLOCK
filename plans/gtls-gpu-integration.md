# GTLS GPU Integration Plan

> Historical design notes for the SHERLOCK 1.6.0 GPU integration. Issue
> [#118](https://github.com/franpoz/SHERLOCK/issues/118) supersedes its
> vendoring recommendation: SHERLOCK uses the published `gputls==0.6.1`
> distribution. API observations originating from the 0.5.1 vendored snapshot
> should be checked against the installed 0.6.1 package before use.

## Goal

When SHERLOCK is run in GPU mode (`SEARCH_ENGINE: gpu`), use GTLS (`gputls`) as
the transit-search engine instead of foldedleastsquares' incomplete GPU path.
Keep foldedleastsquares for CPU TLS, BLS, shared helper functions, and any
non-search code until the GTLS path is proven equivalent.

## Findings

### SHERLOCK execution path

1. `sherlockpipe/search/run.py:184` reads `SEARCH_ENGINE` and passes it into
   every `SherlockTarget`.
2. `sherlockpipe/search/sherlock_target.py:173-198` always registers one
   default `TlsSearcher`, and stores `cpu`, `gpu`, or `gpu_approximate` as a
   target property.
3. `sherlockpipe/search/sherlock.py:840-847` selects the searcher and calls its
   common `search(...)` interface for the original curve and every detrended
   curve. This is the narrowest backend boundary.
4. `sherlockpipe/search/TlsSearcher.py:51-71` currently creates
   `foldedleastsquares.transitleastsquares` and turns both GPU modes into
   `use_gpu=True`; `gpu_approximate` is passed directly to foldedleastsquares.
5. The result is used beyond the adapter:
   - `sherlock.py:348-355, 774, 802` reads `rp_rs`, transit depths, and transit
     times.
   - `sherlock.py:863-881` uses `in_transit` and the full model to mask or
     subtract a detected signal on later iterative runs.
   - `plot/plotting.py:35-90` requires unfolded and folded model arrays,
     folded phases, folded flux, period grid, and power.
   - `scoring/AverageSpectrumSignalSelector.py:49-77` averages `chi2`, reuses
     the period grid and power, and reruns the selected searcher at one period.
   - `Searcher._is_harmonic` and `scoring/helper.py` require periods and power.
6. There is no separate `--gpu` parser path in the inspected SHERLOCK code;
   the existing user-facing switch is the YAML `SEARCH_ENGINE` value. Do not
   add a second flag unless a separate CLI requirement is confirmed.

### GTLS API and behavior

SHERLOCK 1.6.0 originally vendored GTLS from `sherlockpipe/gtls`, pinned at
commit `74e449c325792a763dde4fbffab98039c5e8c111` (version `0.5.1`) from
`https://github.com/martindevora/GTLS`. Issue #118 replaces that packaging with
the PyPI dependency `gputls==0.6.1`.

The API and result-schema notes below were collected against the old submodule
snapshot; verify them against the installed PyPI distribution when making
future compatibility changes.

1. Public usage is `from gputls import gtls`, then
   `gtls(t=time, y=flux).power(...)`.
2. GTLS always executes its CUDA/CuPy implementation; there is no
   `use_gpu`/CPU mode. `GPUDeviceID`, `GPUDeviceIDs`, `legacy`, `fast`, and
   `bar_location` are its relevant execution controls.
3. GTLS accepts `periods` as the explicit trial grid, not
   foldedleastsquares' `period_grid`. Its accepted options include stellar
   radius/mass bounds, `period_min/max`, `n_transits_min`,
   `oversampling_factor`, `duration_grid_step`, `T0_fit_margin`, limb
   darkening/template parameters, and `transit_depth_min`.
4. GTLS supports only `default`, `grazing`, and `box` transit templates. It
   does not expose foldedleastsquares' custom transit-template generator or
   `tailed` template.
5. `gputls.results.gtlsResult` supplies period grid, best period, raw and
   fitted duration, depth, T0, transit times, SDE, chi2, power, and several
   SNR variants. It does not supply SHERLOCK's expected `rp_rs`, FAP,
   odd/even values, per-transit depth arrays, folded arrays, or model
   lightcurve arrays.
6. GTLS depth is returned as a flux-depth value (`1 - transitMean`), while
   its transit-time/depth statistics use GTLS-specific duration and epoch
   conventions. These must be normalized and verified, not copied blindly.
7. GTLS 0.6.1 declares NumPy, Numba, tqdm, and batman-package. CuPy is
   CUDA-specific and remains a separately selected optional installation;
   CUDA version, CuPy wheel, NVIDIA driver, and VRAM remain deployment
   requirements.
8. GTLS uses compiled CUDA kernels, NVML memory inspection, and explicit GPU
   device selection. It can use multiple GPUs through subprocesses, but
   SHERLOCK should initially target one GPU per SHERLOCK process.

## Recommended design

### 1. Add a dedicated GTLS adapter

Create `sherlockpipe/search/GtlsSearcher.py` implementing the existing
`Searcher.search(...)` contract. Do not replace foldedleastsquares imports
globally and do not make GTLS pretend to be the foldedleastsquares module.

The adapter should:

1. Import `gputls` lazily inside `search()` or behind a GPU-only factory so
   CPU runs do not require CuPy or a CUDA driver.
2. Construct `gputls(time, lc, dy=...)` using SHERLOCK's flux and
   available flux errors. If no usable errors exist, allow GTLS to create its
   standard deviation weights.
3. Build one explicit `periods` array from SHERLOCK's `period_grid`. This
   preserves the already-selected search grid and avoids GTLS silently making
   a different grid. Pass the same stellar radius/mass values and bounds,
   `n_transits_min`, oversampling, duration step, T0 margin, and supported
   template choice.
4. Map SHERLOCK's `default`, `box`, and `grazing` fit methods to GTLS's
   supported template names. Reject or explicitly fall back for `tailed` and
   custom templates in GPU mode rather than silently changing morphology.
5. Pass a single configured GPU id, preferably from a new optional
   `GPU_DEVICE_ID` property. Defer multi-GPU configuration until single-GPU
   correctness and process interaction are established.
6. Catch missing-CUDA/CuPy/NVML and GTLS runtime errors with an actionable
   message. Do not silently fall back to CPU because `SEARCH_ENGINE: gpu`
   must be an explicit execution choice; if fallback is desired, add a
   separately documented opt-in later.

### 2. Normalize GTLS output into SHERLOCK's existing contract

Keep `TransitResult` as the boundary object, but create a small internal
result namespace/dictionary in the adapter containing every field consumed by
SHERLOCK. At minimum populate:

- `periods`, `power`, `chi2`, `period`, `duration`, `T0`, `transit_times`,
  `SDE`, `snr`, and `FAP` (use the GTLS FAP policy only if it can be mapped;
  otherwise document and use a clearly defined `NaN`/sentinel policy).
- `transit_depths` and uncertainties in normalized flux units expected by
  SHERLOCK. Convert them once to ppt in `TransitResult`, matching the current
  TLS adapter's `(1 - flux) * 1000` convention.
- `depth_mean_even`, `depth_mean_odd`, and `odd_even_mismatch`. Derive these
  from GTLS transit windows or provide a deliberate neutral value, with tests
  documenting the limitation. Do not use an absent attribute.
- `distinct_transit_count` and `rp_rs`. Compute `rp_rs` consistently from
  the normalized depth and limb-darkening law, or use the same radius/depth
  convention SHERLOCK already uses and record the approximation.
- `model_lightcurve_time` and `model_lightcurve_model` for iterative
  subtraction. Generate the model from GTLS's best duration/depth/template
  and transit times, or initially disable `MASK_MODE: subtract` for GTLS with
  a validation error. The default `mask` path must work.
- `model_folded_phase`, `model_folded_model`, `folded_phase`, and `folded_y`
  for plots. Fold and sort the original input arrays using one shared helper
  with the same phase convention as `transit_mask`; build a sampled box or
  GTLS-compatible model on that phase grid.

Prefer adding reusable phase/model construction helpers in the search package
over adding GTLS-specific assumptions to plotting or the orchestrator. The
rest of SHERLOCK should continue to consume the normalized result.

### 3. Route only GPU TLS through GTLS

Change target/searcher construction so:

- `cpu` continues to use the current `TlsSearcher` and foldedleastsquares.
- `gpu` uses `GtlsSearcher` and does not pass foldedleastsquares GPU flags.
- `gpu_approximate` is either rejected as unsupported by GTLS or mapped to an
  explicitly documented GTLS approximate/fast mode only after its semantics
  are verified. Do not map it to GTLS `fast` without checking: GTLS `fast`
  returns only periods and power and cannot satisfy SHERLOCK's result contract.
- `bls-periodogram` continues to use `BlsSearcher`; its helper use of
  foldedleastsquares for `transit_mask`/`fold` should remain independent of
  the TLS backend migration.

A factory selected by `SherlockTarget.search_engine` is preferable to making
`TlsSearcher` branch on GPU state. This keeps the current CPU implementation
stable and makes backend-specific tests isolated.

### 4. Dependency and configuration changes

1. Remove the GTLS gitlink and vendored package configuration. Pin
   `gputls==0.6.1` in SHERLOCK metadata and the maintained environment lock.
   CuPy and CUDA runtime packages are not base requirements; document a
   CUDA-matched CuPy installation separately for GPU deployments.
2. Validate the installed distribution version and location from site-packages.
   The external distribution owns the `gputls` import namespace.
3. Add optional `GPU_DEVICE_ID` configuration, defaulting to 0, and validate
   it only when GPU mode is selected.
4. Update `requirements.txt`, installation documentation, and
   `properties.yaml` comments without making CuPy a CPU requirement.
5. Log the selected backend, GTLS version, CUDA device, and relevant GTLS
   mode in the object report so result provenance is visible.

## Testing Plan

### Testing deliverables

The implementation is not complete until CPU-safe contract tests pass against
the installed PyPI distribution and GPU runtime tests are run where compatible
CUDA hardware is available:

1. `sherlockpipe/tests/search/test_gtls_searcher.py`: unit-test the adapter
   with a fake GTLS result. Verify argument translation, template mapping,
   depth/unit conversion, epoch normalization, transit counting, phase/model
   construction, `TransitResult` population, and unsupported-mode errors.
2. `sherlockpipe/tests/search/test_search_backend_selection.py`: verify that
   `cpu` selects foldedleastsquares, `gpu` selects GTLS, and neither GPU mode
   sends foldedleastsquares GPU flags. Verify CPU imports do not load GTLS,
   CuPy, or NVML.
3. `tests/test_gtls_distribution.py` and
   `sherlockpipe/tests/search/test_gtls_contract.py`: verify the installed
   distribution version/location, public API when CUDA dependencies are
   available, and lazy CPU-side import behavior.
4. `sherlockpipe/tests/search/test_gtls_synthetic.py`: run a deterministic
   injected transit on a CUDA runner and assert finite results, recovered
   period/epoch/duration/depth within tolerances, valid masks, and model
   arrays. Include gaps, shallow signals, and one/two-transit cases.
5. `sherlockpipe/tests/search/test_gtls_repeated_runs.py`: run the adapter over
   multiple detrended curves and iterative runs, checking that CUDA memory is
   released and no stale device state changes later results.
6. Extend the existing SHERLOCK regression coverage for plotting, candidate
   CSV/report output, pickling, `MASK_MODE: mask`, `MAX_RUNS > 1`, and the
   `average-spectrum` selector. Keep subtraction, custom templates, and
   `gpu_approximate` as explicit unsupported/feature tests until implemented.

Separate tests into CPU-safe and CUDA-required groups so normal CI remains
usable without NVIDIA hardware. Mark CUDA tests clearly and run them in a
dedicated environment with the supported CUDA/CuPy combination.

### Phase A: adapter contract tests without CUDA

Use a fake GTLS result or monkeypatch the lazy import to test the adapter's
argument mapping and normalization. Cover:

- explicit `period_grid` becomes GTLS `periods` unchanged;
- stellar priors, transit count, oversampling, duration step, and T0 margin
  are passed correctly;
- template mapping and unsupported custom/tailed behavior are explicit;
- depth, epoch, transit count, mask, folded arrays, model arrays, and
  `TransitResult` units are populated;
- CPU mode never imports GTLS/CuPy;
- GPU mode never passes `use_gpu` or `gpu_approximate` to foldedleastsquares.

### Phase B: real GTLS smoke tests

On a CUDA machine, run a small synthetic light curve and one representative
SHERLOCK light curve with a narrow period grid. Assert that GTLS returns finite
period, duration, T0, SDE, power, transit times, and model arrays, and that
memory is released between repeated detrend searches and iterative runs.

### Phase C: CPU/GPU scientific parity

Run the same prepared/detrended curves through CPU foldedleastsquares and GPU
GTLS. Compare recovered period, epoch modulo period, duration, depth, transit
count, SNR/SDE ordering, and candidate selection. Use tolerances appropriate
to the different kernels, not exact equality. Include gaps, NaNs, one/two
transit edge cases, long-period signals, shallow signals, and non-solar
stellar priors.

### Phase D: end-to-end SHERLOCK behavior

Exercise original plus multiple detrended curves, each signal selector
(especially `average-spectrum`), plotting, `MASK_MODE: mask`, iterative
`MAX_RUNS > 1`, pickling, candidate CSV/report generation, and the existing
regression fixtures. Treat `MASK_MODE: subtract`, custom templates, and
`gpu_approximate` as explicit acceptance items rather than allowing silent
degradation.

### Phase E: deployment checks

Validate the documented Python/CUDA/CuPy combination in the supported Docker
or Singularity environment, including a clean CPU install where importing
SHERLOCK and running `SEARCH_ENGINE: cpu` does not require a GPU package.

## Suggested implementation order

1. Add the adapter and pure NumPy normalization/model helpers behind tests.
2. Add target backend selection and lazy GPU dependency loading.
3. Add GTLS dependency/configuration/documentation and backend logging.
4. Make default GPU mode work with `mask`, standard templates, and the normal
   signal selectors.
5. Validate parity and resource cleanup on real CUDA hardware.
6. Only then decide whether to support `gpu_approximate`, subtraction,
   custom templates, GTLS multi-GPU, or a new CLI alias.

## Main risks and decisions

- GTLS's result schema is substantially smaller than foldedleastsquares'; the
  adapter must own all compatibility work.
- GTLS and foldedleastsquares have different transit-template and epoch/depth
  conventions; parity must be measured scientifically.
- GTLS's `fast` mode cannot support the current SHERLOCK pipeline because it
  omits the detailed fit result.
- GTLS's package metadata does not install CuPy, and its kernels require a
  compatible CUDA runtime. On the supported RTX 5060 Ti/driver combination,
  CUDA 13 NVRTC/runtime libraries are required; otherwise GTLS fails with
  `CUDA_ERROR_NO_BINARY_FOR_GPU` while compiling its kernels.
- The old GTLS 0.5.1 single-GPU search could compute a zero batch size for
  small period grids. The adapter pads grids below 100 periods for its
  internal search while preserving the requested range, which is required by
  SHERLOCK's average-spectrum rerun.
- SHERLOCK currently has one GPU search invocation per detrended curve. A
  single-GPU first implementation avoids GTLS subprocess/multi-GPU conflicts
  and makes VRAM cleanup measurable.
