# AGENTS.md — SHERLOCK (sherlockpipe)

> Agent-facing guide for the `SHERLOCK` (sherlockpipe) project.

## CRITICAL: Never Timeout Shell Commands

**WARNING: The `bash` tool defaults to a 120-second (120000ms) timeout if you omit the `timeout` parameter.** This means simply omitting `timeout` will STILL kill long-running SHERLOCK processes. You MUST explicitly pass a timeout value.

SHERLOCK executions (searches, fits, vetting, validation, stability checks, etc.) routinely take hours to days. Any timeout will kill the process mid-execution, producing truncated or useless results.

**Do NOT do this:**
```
bash(command="...")                       # WRONG — defaults to 120s timeout, will kill SHERLOCK
bash(command="...", timeout=120000)       # WRONG — same result
```

**Always pass an explicit, very large timeout:**
```
bash(command="...", timeout=0)            # 0 means no timeout
```

If `timeout=0` is rejected, use the maximum allowed value:
```
bash(command="...", timeout=9007199254740991)
```

Long-running commands should be delegated to a subagent via `task(subagent_type="general")`. The subagent prompt must include the full conda activation command, e.g.: `conda run -n sherlock311 python3 -m sherlockpipe --properties my_properties.yaml`. The subagent must also be instructed to use the timeout pattern above.

## Git Workflow

**CRITICAL: Always push to `origin` (PlanetHunters/SHERLOCK), never to `franpoz`.**

| Remote | URL | Purpose |
|--------|-----|---------|
| `origin` | `PlanetHunters/SHERLOCK` | **Daily work.** All commits, PRs, and pushes go here. |
| `franpoz` | `franpoz/SHERLOCK` | **Releases only.** Only used when syncing forks for version releases. Do NOT push daily work here. |

The `master` branch tracks `origin/master`. When committing changes, always verify the target with `git branch -vv` before pushing.

## Project Intent

**SHERLOCK** (Search for Hints of Exoplanets fRom Lightcurves Of spaCe-based seeKers) is an end-to-end Python pipeline for exploring TESS, Kepler, and K2 data to search for exoplanetary transit candidates. It can recover known candidates (TOIs, KOIs, EPICs), search for new ones, vet signals, perform Bayesian transit fitting, statistical validation, dynamical stability checks, and compute ground-based observational windows.

- **Package:** `sherlockpipe`
- **Version:** `1.4.0`
- **Authors:** M. Dévora-Pajares & F.J. Pozuelos
- **License:** MIT
- **Repository:** https://github.com/franpoz/SHERLOCK

## Architecture

```
sherlockpipe/
├── __init__.py              # Builds ellc shared lib, patches SSL/requests
├── __main__.py              # Main CLI entrypoint
├── properties.yaml          # Default configuration template
├── search/
│   ├── sherlock.py          # Sherlock class — central orchestrator
│   ├── sherlock_target.py   # SherlockTarget configuration DTO
│   ├── Searcher.py          # ABC for search algorithms
│   ├── TlsSearcher.py       # TLS via foldedleastsquares
│   ├── BlsSearcher.py       # BLS search
│   ├── transitresult.py     # Transit search result container
│   └── run.py               # Orchestrates search from YAML
├── scoring/                 # Signal selection algorithms
├── loading/                 # YAML/properties loading helpers
├── ois/                     # Objects of Interest manager
├── bayesian_fit/            # Transit fitting (alexfitter)
├── vetting/                 # Vetting (dearwatson)
├── system_stability/        # Rebound/SPOCK stability checks
├── observation_plan/        # Ground-based observation planning
├── single_transits/         # Single transit search tools
├── search_zones/            # Habitable zone definitions
├── plot/                    # Plotting utilities
├── catalog/                 # Catalog helpers (MAST)
├── ellc/                    # Bundled Fortran library (compiled to libellc.so)
└── tests/ & regression_tests/
```

### Key Classes

| Class | Role |
|-------|------|
| `Sherlock` | Central orchestrator. Prepares, detrends, identifies signals iteratively. |
| `SherlockTarget` | Configuration container for a single target (detrend params, search zone, etc.). |
| `Searcher` (ABC) | Interface for transit search engines. |
| `TlsSearcher` | TLS search via `foldedleastsquares`. |
| `BlsSearcher` | BLS search. |
| `SignalSelector` (ABC) | Picks the best signal among detrended curves. |
| `OisManager` | Downloads/loads TOI/KOI/EPIC metadata. |

### Execution Flow

1. `__main__.py` or `search/run.py` parses CLI / properties YAML.
2. `run_search()` loads default + user YAML, constructs `SherlockTarget`s.
3. `Sherlock.run()` iterates over targets:
   - **Prepare**: fetch light curve via `lcbuilder`, compute star info, FOV plots, mask known OIs.
   - **Detrend**: apply biweight or GP detrending with multiple window lengths.
   - **Identify signals**: run TLS/BLS on original + detrended curves.
   - **Select best signal** using configured `SignalSelector`.
   - **Mask / subtract** the best signal and repeat up to `max_runs`.
4. Results written to per-object directories (`candidates.csv`, plots, logs).

## Entrypoints

No `console_scripts` in `setup.py`. Entry is via `python -m` module execution or library import.

### CLI Entrypoints (`python -m <module>`)

| Command | Purpose |
|---------|---------|
| `python3 -m sherlockpipe --properties my_properties.yaml` | Run full search pipeline. |
| `python3 -m sherlockpipe --properties my_properties.yaml --explore` | Run only preparation/detrending (no search). |
| `python3 -m sherlockpipe.update` | Download/update TOI/KOI/EPIC metadata. |
| `python3 -m sherlockpipe.update --force` | Force update metadata. |
| `python3 -m sherlockpipe.update --clean` | Wipe and rebuild metadata. |
| `python3 -m sherlockpipe.vet --properties my_properties.yaml` | Run vetting (WATSON) on candidates. |
| `python3 -m sherlockpipe.vet --candidate N` | Simplified vet from results directory. |
| `python3 -m sherlockpipe.fit --properties my_properties.yaml` | Run Bayesian fitting (alexfitter). |
| `python3 -m sherlockpipe.fit --candidate N` | Simplified fit from results directory. |
| `python3 -m sherlockpipe.validate --candidate N` | Statistical validation with TRICERATOPS. |
| `python3 -m sherlockpipe.stability --bodies 1,2,4` | System stability with Rebound/SPOCK. |
| `python3 -m sherlockpipe.plan --candidate N --observatories obs.csv` | Observation planning. |

### Library Entrypoint

```python
from sherlockpipe.search.sherlock import Sherlock
from sherlockpipe.search.sherlock_target import SherlockTarget
from lcbuilder.objectinfo.MissionObjectInfo import MissionObjectInfo

info = MissionObjectInfo([9], "TIC 181804752", cadence=[1800])
target = SherlockTarget(object_info=info, detrends_number=2, max_runs=1)
Sherlock([target]).run()
```

## Environment Setup

### Prerequisites

- Python `>= 3.11`
- System build tools: `make`, `gfortran`, `gcc` (to compile the bundled `ellc` Fortran library)
- `conda` (tox-conda is used for tests)

### Conda Environment Setup

```bash
cd /path/to/SHERLOCK
conda create -n sherlock311 python=3.11
conda activate sherlock311

# Install the package (triggers make for libellc.so)
pip install -e .
# or install pinned deps first
pip install -r requirements.txt
```

### Docker (alternative)

```bash
docker build -t sherlock .
docker run -it sherlock
```

### Key Dependencies

| Package | Purpose |
|---------|---------|
| `lcbuilder==0.25.4` | Light curve building/preparation |
| `foldedleastsquares==1.1.11` | TLS search engine |
| `wotan==1.9` | Detrending |
| `alexfitter==1.2.19` | Bayesian transit fitting |
| `dearwatson==1.0.1` | Vetting (WATSON) |
| `rebound==4.4.1` | N-body stability |
| `triceratops==1.0.19` | Statistical false-positive validation |
| `astroplan==0.10.1` | Observation planning |
| `numpy==2.1.1` | Core numerics |
| `lightkurve==2.5.0` | Mission data download |
| `tensorflow==2.19.0`, `torch==2.7.0` | ML backends |

## Build & Test

### Run Tests (tox — recommended)

```bash
tox
```

Runs two environments:
- `py311-gha`: `pytest --forked -n 1 -v -x sherlockpipe/tests/`
- `py311-local`: `pytest --forked -n 1 -v -x sherlockpipe/regression_tests/`

### Run Tests (pytest directly)

```bash
# Unit tests
pytest --forked -n 1 -v -x sherlockpipe/tests/

# Regression tests
pytest --forked -n 1 -v -x sherlockpipe/regression_tests/
```

## TOI / KOI Target Resolution

When a user asks to search, vet, fit, or explore a **TOI** or **KOI** target by its catalog name (e.g. "TOI-2079", "KOI-1234"), you MUST first resolve it to its underlying mission ID:

1. **Look up the TIC or KIC ID** at [https://exofop.ipac.caltech.edu/](https://exofop.ipac.caltech.edu/). Search by TOI/KOI name; the target page will show the corresponding TIC (TESS) or KIC (Kepler) identifier.
2. **Retrieve the known candidates/planets** from the same ExoFOP page. Record each confirmed or candidate planet's period, epoch (t0), duration, and depth. These are needed for masking (`INITIAL_TRANSIT_MASK`) or fitting/vetting configuration.
3. **Use the resolved TIC/KIC ID** when constructing `MissionObjectInfo` or the properties YAML — SHERLOCK downloads data by mission ID, not by TOI/KOI alias.
4. **Populate `INITIAL_TRANSIT_MASK`** with the period, t0, and duration from ExoFOP to mask known signals during the search, or set `INITIAL_OIS_MASK: True` to auto-mask from the built-in OI catalog.

Example: "TOI-2079" → TIC 27989529 (from ExoFOP) → uses "TIC 27989529" as the target ID in the YAML.

## Execution Recipes

> **CRITICAL**: The `bash` tool defaults to a 120s timeout when `timeout` is omitted. You MUST explicitly pass `timeout=0` (or a huge value). See the [CRITICAL section](#critical-never-timeout-shell-commands) at the top. SHERLOCK executions can take hours to days.
>
> Long-running commands MUST be delegated to a subagent via `task(subagent_type="general")`. The orchestrator should never run these inline. The subagent prompt must include the full conda activation command, e.g.: `conda run -n sherlock311 python3 -m sherlockpipe --properties my_properties.yaml`.

### Use Case A: First-Time Setup

```bash
cd /path/to/SHERLOCK
python3 -m pip install -e .
python3 -m sherlockpipe.update --force
```

### Use Case B: Run a Search from YAML

1. Create a properties YAML based on `sherlockpipe/properties.yaml`.
2. Launch:
   ```bash
   python3 -m sherlockpipe --properties my_properties.yaml
   ```
3. Results appear in a directory named after the object ID (e.g., `TIC12345678_[sectors]/`).

### Use Case C: Explore-Only Mode

```bash
python3 -m sherlockpipe --properties my_properties.yaml --explore
```

### Use Case D: Vetting a Candidate

```bash
python3 -m sherlockpipe.vet --candidate 1
```

### Use Case E: Fitting a Candidate

```bash
python3 -m sherlockpipe.fit --candidate 1
```

### Use Case F: Validation

```bash
python3 -m sherlockpipe.validate --candidate 1
```

### Use Case G: Stability Analysis

```bash
python3 -m sherlockpipe.stability --bodies 1,2,4
```

### Use Case H: Observation Planning

```bash
python3 -m sherlockpipe.plan --candidate 1 --observatories observatories.csv
```

### Use Case I: Programmatic Usage

```python
from sherlockpipe.search.sherlock import Sherlock
from sherlockpipe.search.sherlock_target import SherlockTarget
from lcbuilder.objectinfo.MissionObjectInfo import MissionObjectInfo

info = MissionObjectInfo([9], "TIC 181804752", cadence=[1800])
target = SherlockTarget(object_info=info, detrends_number=2, max_runs=1)
Sherlock([target]).run()
```
