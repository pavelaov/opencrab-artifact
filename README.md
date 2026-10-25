# OpenCrab v2

OpenCrab v2 is a two-party private inference system for the Web: client and server hold additive
shares over Z_2^64 (fixed point with 16 fractional bits), an isolated offline dealer supplies the
comparison keys, and a session controller keeps or switches the protocol plan at every decoding
step. This artifact holds the Python implementation (package `opencrab2` with the experiment
drivers), a C++17 mirror of its protocol layers, run scripts for Ubuntu 22.04 and Windows, and the
records behind the evaluation tables of the paper. Every table is regenerated from
the records in `results/` by one command.

```
python/opencrab2/   lowlevel, highlevel, platform, experiments (drivers and make_tables)
python/tests/       pytest suite (configuration in python/pyproject.toml)
cpp/                C++17 mirror: include/, src/, apps/, tests/ with vectors/, CMakeLists.txt, build.py
scripts/            ubuntu.sh, windows.ps1, run_campaign.py (all stages), run_cpp_bench.py, smoke_session.py
results/            raw records, calibration, tables, PREREGISTRATION.md
requirements.txt    Python packages pinned to the measured environment
```

## Requirements

* CPython 3.12 with `requirements.txt` (measured: 3.12.10). Ubuntu 22.04 ships 3.10, so `setup`
  installs 3.12.10 with uv into `.venv`.
* C++: GCC 11.4 and CMake 3.22 on Ubuntu 22.04; on Windows the clang of the ziglang package
  (Zig 0.16) through `cpp/build.py`. Both compile with `-O2 -ffp-contract=off` into `cpp/build/bin`.
* GPT-2: `openai-community/gpt2` at revision `607a30d783dfa663caf39e06633721c8d4cfcd7e` in the
  Hugging Face cache, downloaded by `setup`. The code reads only that commit and never goes
  online. The quick run uses a tiny model.
* Windows without long paths enabled: keep the artifact path under 100 characters (the deepest
  file below `.venv` is 151 characters long, the limit is 260).

## Commands

`bash scripts/ubuntu.sh <command>` on Ubuntu 22.04, `powershell -ExecutionPolicy Bypass -File
scripts\windows.ps1 <command>` on Windows, from any directory. Only the apt step of `setup` needs
sudo; `OPENCRAB2_PYTHON=<interpreter>` replaces `.venv`; the run steps add 127.0.0.1 and localhost
to `NO_PROXY`, as the processes talk over loopback.

| command | what it does | Windows host |
|---|---|---|
| `setup` | `.venv`, `requirements.txt`, GPT-2 snapshot | 18 min, network bound: 93 MB (Ubuntu) or 180 MB (Windows) of wheels and 548 MB of GPT-2 |
| `build` | C++ executables (CMake on Ubuntu, `cpp/build.py` on Windows) | 30 s |
| `test` | C++ tests with hardware and with portable AES, then `python -m pytest python/tests` | about 13 min (C++ 3 min, Python 10 min) |
| `quick` | every stage in its quick form with the tiny model, into `runs/quick` | 2.5 min |
| `full` | the paper campaign: every stage but the calibration, into `runs/full` | 6.9 h on the authors' host (see below) |
| `tables` | tables regenerated from `results/` into `runs/tables_check`, compared with `results/tables` (after the final campaign) | 10 s |

Nothing was executed on Ubuntu. `test` skips the tests that need the GPT-2 snapshot or the C++
binaries when these are missing (and 3 by design). `quick` and `full` pass further options to
`scripts/run_campaign.py` (`--stage`, `--dry-run`, `--force`, `--help`). Its stages are primitives,
cpp_bench, calibration, simulation, diag4, full12, web and tables; without `--stage` the paper
campaign runs all of them except the calibration (the shipped `results/calibration` prices every
stage), `--stage all` adds a fresh calibration (4.6 h here), and `--quick` runs every stage. A
finished stage is skipped and an interrupted one resumes. Reruns write to `runs/<name>` or
`--out-root DIR` (passed to every child as `OPENCRAB2_RESULTS_DIR`), never into `results/` unless
`--out-root results` is given. The stages run the modules `run_primitives`, `run_calibration`,
`run_simulation`, `run_sessions`, `run_web` and `make_tables` of `opencrab2.experiments`
(`python -m opencrab2.experiments.<module>` with `python/` on `PYTHONPATH`); run directly, these
modules write into `results/` unless `OPENCRAB2_RESULTS_DIR` or `--out-root` is set, so reruns go
through `run_campaign.py`. Stage times of the final campaign are in the block below.
`cpp/tests/smoke_three_process.py --bin cpp/build/bin` runs the C++ apps (dealer, model server,
sidecar, parties, bench) over loopback and validates their records with the Python schema.

## What results/ holds

Files marked * were written by the final campaign; the calibration files ship from the earlier run.

| file | paper table |
|---|---|
| `raw/primitives.jsonl`*, `raw/cpp_bench.jsonl`* | 3, building blocks (`tables/blocks.tex`) |
| `calibration/calibration_metrics.json` | 4, cost model (`costmodel.tex`) |
| `raw/sessions.jsonl`*, full12 and the diag4 LAN anchors | 5, end to end (`e2e.tex`) |
| `raw/web.jsonl`* | 6, Web service (`web.tex`) |
| `raw/sessions.jsonl`*, diag4 and full12 | 7, dynamic conditions (`dynamic.tex`) |
| `raw/simulation.jsonl.gz`* | 8, modeled campaign (`dynamic_modeled.tex`) |
| `raw/factorial.jsonl.gz`* | 9, planner factorial (`factorial.tex`) |
| none: the registry in `python/opencrab2/highlevel/registry.py` | 10, catalog (`catalog.tex`) |

`calibration/fitted_model.json` is the cost model the drivers price with and the source of the
expected values the tables print in red. `calibration/formulas.json` holds the declared cost
formulas; its sha256 is recorded in the metrics, and a changed byte marks Table 4 stale.
`raw/calibration.jsonl` holds the measurements behind the fit, `raw/pilot/` the first measured
campaign and the block benchmarks of the earlier tree (they enter no table), `tables/manifest.json`* the inputs and cell counts. A raw file may
also be stored as `.jsonl.gz`. The records of the first campaign and of the calibration predate the
relative path convention: their `raw_measurement_path` (and in `raw/pilot/` the detail and
launcher paths) name the authors' working directory; no table reads these fields, and records
written by the current code store paths relative to the data root.

## Protocol declared before the runs

Details, deviations and the findings of the first campaign: `results/PREREGISTRATION.md`.

* Cost model hold-out, declared before fitting: shapes 1024x1024 and 3072x768 on WAN-mid, batch 32.
* diag4: GPT-2 4-layer prefix, 8 tokens; cells S0 stable LAN and WAN, S1 ramp, S2 RTT steps,
  S3 two changes, S8 cold and warm; policies B0 (fixed all-wmask and all-generic), B1, B2, B4, and
  on S1 and S3 also B4 without installation pricing and delayed(2). full12: 12 layers, 8 tokens,
  S0 stable LAN and S1 ramp, B0 all-wmask, B2, B4. Seeds 1 to 3, policy order randomized.
* 3 sessions side by side in both matrices (the first campaign ran diag4 at 6), 2 BLAS threads
  per process, 3 processes per session, 7200 s session timeout.
* Declared reductions: no all-generic plan on the S0 WAN, S1, S2 and S3 cells of diag4, and only
  B0 and B4 on the S1 cell of full12; 105 and 15 sessions.
* Fixed trace durations, shared by all policies and seeds of a cell: diag4 S0 LAN 450 s, S0 WAN
  650 s, S1 770 s, S2 440 s, S3 440 s, S8 cold 450 s, S8 warm 430 s; full12 S0 LAN 410 s,
  S1 1550 s (S2 and S3 amended on 2026-10-14 so that every delay step and regime change falls
  inside the session; see `results/PREREGISTRATION.md`).
* Modeled campaign: nine trace families, seeds 1 to 10, 8 and 32 tokens. Differences are paired
  within a seed and a concurrency level; 95 % percentile bootstrap intervals (10 000 resamples).

**Final campaign (2026-10-14, this tree).** `python scripts/run_campaign.py --out-root results`
wrote every record in `results/raw/` except `calibration.jsonl`. Stage times: primitives 52 s,
cpp_bench 130 s, simulation 6.3 min (16 workers), diag4 5.2 h (3 workers), full12 1.1 h
(3 workers), web 31 min, tables 2 s; 6.9 h in total. diag4 completed 104 of 105 sessions: one
session (S0 stable LAN, fixed all-generic plan, seed 1) finished in 393 s with the reference tokens,
but the launcher did not receive one party's result and waited out the 7200 s timeout, so its
record is an abort and the tables exclude it as declared; diag4 then exited 1 and full12, web and
tables ran in a second invocation with `--stage full12,web,tables`. full12 completed 15 of 15.
All 119 completed sessions release the float reference's tokens. Records: `sessions.jsonl` 239
lines (two parties per session and the abort), `web.jsonl` 36, `simulation.jsonl.gz` 16 500,
`factorial.jsonl.gz` 3 960, `primitives.jsonl` 27, `cpp_bench.jsonl` 79.
