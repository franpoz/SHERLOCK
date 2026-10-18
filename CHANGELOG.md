# Changelog

## 1.6.0 — 2026-10-18

### Added

- Bundle the reviewed GTLS 0.5.1 submodule and expose it as SHERLOCK's GPU TLS
  backend, with GPU device selection and packaged backend contract tests.
- Add ExoMoon analysis and calibrated EMLS-IR resources, alongside improved
  candidate vetting, scoring, and light-curve handling.

### Changed

- Update WATSON integration to `dearwatson==2.0.0`, with compatible
  `exoml==1.4.0`, `lcbuilder==0.27.0`, and `alexfitter==1.2.20` dependencies.
- Document the CUDA 13/CuPy installation profile and Poppler requirements for
  WATSON report generation.
