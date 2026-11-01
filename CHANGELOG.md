# Changelog

## Unreleased

### Changed

- Replace the vendored GTLS submodule with the pinned `gputls==0.6.1` PyPI
  dependency. Keep CuPy as an optional, CUDA-specific installation.

## 1.6.0 — 2026-10-18

### Added

- Expose GTLS as SHERLOCK's GPU TLS backend, with GPU device selection and
  backend contract tests.
- Add ExoMoon analysis and calibrated EMLS-IR resources, alongside improved
  candidate vetting, scoring, and light-curve handling.

### Changed

- Update WATSON integration to `dearwatson==2.0.0`, with compatible
  `exoml==1.4.0`, `lcbuilder==0.27.0`, and `alexfitter==1.2.20` dependencies.
- Document the CUDA 13/CuPy installation profile and Poppler requirements for
  WATSON report generation.
