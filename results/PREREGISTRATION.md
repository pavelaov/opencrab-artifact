# Preregistration of the WWW 2027 evaluation (written before any new run)

Date: 2026-10-10. Host: one Windows 11 build 26200 machine, Intel Core Ultra 9 285K (24 logical
processors), RTX 4090 present but NOT used by any secure path; CPython 3.12.10; the C++ build uses
Zig 0.16 (clang). Two online party processes and one dealer process on loopback TCP with
wall-clock exogenous shaping. Nothing below is a measured result.

## Hypotheses

* H1: candidate-specific incremental state costs improve on rerunning static assignment alone.
  Test: B4 versus B1 and versus B3 (planner adapter plus B2) on the same engine, catalog, calibration.
* H2: one-time material and safe recovery constraints matter to session feasibility.
  Test: S5 and S7 families; completion rate, safe waste, recovery cost.
* H3: value extends beyond embedding installation and to full models.
  Test: S2 (RTT only, k-ary selection crossover), S8 (installation state), 12-layer anchor.
* Null results are reported as such. If B2 and B4 tie, the paper claims a session runtime, not a
  superior assignment algorithm.

## Fixed design

* Profile: SH2P-TrustedOffline-v1; numeric contract NC-v2-f16-l40; model openai-community/gpt2
  (revision recorded at run time); prompt: the 16-token public prompt of the previous campaign;
  generation lengths T in {8, 32}; input lengths {16, 64}; the 4-layer prefix is diagnostic only.
* Trace families S0 to S8 with three variants each, seeds 1..10 (odd = tune, even = test);
  MODELED campaign uses all; MEASURED campaign uses a declared subset (below) because of wall-clock
  limits: S0 (stable LAN, stable WAN-c), S1 (ramp WAN-c to LAN), S2 (RTT 0.25 ms to 100 ms at
  100 Mbit/s), S3 (two changes), S8 (installed versus cold on the same link). At least 3 paired
  repeats per measured cell; 10 per modeled cell.
* Policies: B0 (best fixed in hindsight among fixed plans run), B1, B2, B4, ablations B4-no-install,
  B4-no-hyst, B4-no-unc, delayed(2); B3, B5, B6 in the modeled campaign.
* Statistics: paired differences with 95 % bootstrap intervals (10 000 resamples, seed 0), ratios
  with intervals, zero-failure one-sided 95 % upper bound 1 - 0.05^(1/n) with n = independent
  sessions; medians and p95 with n stated; policy order randomized per seed.
* Cost model: NNLS affine per component; hold-out declared before fitting: shapes 1024x1024 and
  3072x768, network WAN-mid, batch n = 32; crossover-near cases included; metrics MAPE, median,
  p95, Spearman, lane accuracy, full-plan accuracy, realized regret, and the informative pair.
* Timing boundaries: T_cold, T_warm, T_online as in INTERFACE_SPEC section 8.
* Any cell that does not execute (timeout 2 h per session, OOM, unsupported) is reported by reason
  and never filled with a prediction presented as a measurement.

## Stopping rule

Measured campaign budget: 6 wall-clock hours. Modeled campaign: full matrix. Expansion beyond the
budget only by a fixed precision rule (halve the widest interval), never until significance.

## Deviations and concrete campaign (appended 2026-10-10, before any measured session of the campaign below)

The text above is unchanged. This section records what differs from it and fixes the concrete
campaign. Nothing in this section is a measured result; the only runs made while writing it were
pipeline checks (primitives --quick, the two-cell smoke matrix, the tiny-model web run), which are
not evidence for any hypothesis.

### Why the plan changed

* Security errata S1 and S2 (INTERFACE_SPEC section 12) were found and repaired on 2026-10-10. Every
  record produced before the repair with a payload DReLU gate or a Z_t key is not security evidence.
  Consequences: the earlier `raw/primitives.jsonl` was moved to `raw/pre_s1/` and is not read by any
  table; the calibration and the modeled campaign must be re-run (their current files predate the
  repair, the fused k-ary formulas and `calibration/formulas.json`); until then the cost-model table
  marks every cell as an estimate and says so in its header.
* Four old-shape C++ lines in `raw/cpp_web_session.jsonl` do not validate against the schema and
  were produced with a public session seed; the file was moved to `raw/invalid/`.
* The party masks of a session now come from the OS CSPRNG (`SessionConfig.party_rng_from_os`); a
  session run with the public seed is never security eligible.

### Concrete measured campaign (replaces the "main" and "anchor" matrices of the driver)

* `diag4` (diagnostic, `workload_scope = prefix-4L`): GPT-2 4-layer prefix, the pinned 16-token
  prompt, T = 8. Trace cells: S0 stable LAN; S0 stable WAN (100 Mbit/s, 20 ms one way); S1 ramp
  from 20 Mbit/s to 1 Gbit/s over the session (variant ramp-WAN-c-to-LAN, the delay stays at its
  initial 40 ms one way); S2 RTT only (0.25, 10, 40, 100 ms one way at 100 Mbit/s, variant
  rtt-step-up); S3 two regime changes; S8 cold and S8 warm-installed on the same LAN link. Policies
  on every cell: B0 fixed all-wmask plan, B0 fixed all-generic plan, B1, B2, B4; on the S1 and S3
  cells additionally B4-no-install and delayed(2). Seeds 1, 2, 3. Policy order randomized per
  (seed, cell) with a seeded permutation. 117 sessions.
* `full12` (`workload_scope = full-model`): full 12-layer GPT-2, T = 8; cells S0 stable LAN and S1
  ramp; policies B0 fixed all-wmask, B2, B4; seeds 1, 2, 3. 18 sessions.
* Deviations from the fixed design above: (1) the 12-layer measured matrix is reduced to two cells
  and three policies, and the full policy and trace matrix is measured on the 4-layer prefix, which
  the fixed design names as diagnostic only: claims about policies on the full model rest on
  `full12` alone, and `diag4` results are reported as prefix-4L diagnostics; (2) T = 32 and input
  length 64 are not measured (modeled campaign only); (3) B4-no-hyst and B4-no-unc are not measured
  (modeled campaign only), B4-no-install and delayed(2) only on S1 and S3; (4) S0 stable WAN-c is
  replaced by S0 stable WAN; (5) the S8 pair runs on the LAN anchor: two variants `cold-LAN` and
  `warm-all-LAN` were added to family S8 (the three original S8 variants on the WAN anchor are
  unchanged), so S8 now has five variants in the modeled campaign; (6) B0 in the measured tables is
  the better of the TWO fixed plans that were run in the cell (lowest mean session time), not the
  best over the whole catalog.
* Warm state: the S8 warm cell is a real warm request. Both party processes first run an unshaped
  priming session (fixed all-wmask plan, one token) and then the measured session on the same
  stack; installed objects, shared server vectors and the ledger are reused per (model_version,
  client_key_domain). T_warm of that cell is the measured wall time of the warm request. For cells
  without a warm request, T_warm is cold minus setup and installation and is printed as an
  estimate. The priming session is recorded in a separate file and enters no table.
* Concurrency: sessions may run side by side (`--workers N`), each launched process capped at
  `--threads-per-process` BLAS threads (default 2). Every record carries the concurrency level and
  the thread cap; paired differences are formed only within one seed and one concurrency level,
  and levels are never pooled. Running concurrently changes absolute times; the hypotheses are
  about paired differences. If a paired difference is within its interval of zero at the campaign
  concurrency, the affected cells are repeated with one worker before any claim is made (fixed
  precision rule, not a significance hunt).
* Output match: the launcher computes the greedy continuation of the float reference (erf-GELU,
  exact softmax and LayerNorm) for the public test prompt and gives it to P1 only; each P1 record
  carries output_match and the reference top-1 minus top-2 margins; token ids stay in P1's own
  detail file. A mismatch is reported as such with its margin, never dropped.
* Aggregation rules (fixed in `make_tables.py` before the campaign): client (P1) records; aborted
  sessions are excluded from means and counted in the table header; session time is the measured
  wall time; intervals are percentile bootstrap intervals of the mean paired percent difference
  (10 000 resamples, seed 0); with three seeds the intervals are wide and are printed as they are.
* Budget: the dry run (`run_sessions --dry-run`) predicts each cell from the fitted cost model and
  scales by the measured/predicted ratio of completed sessions on this host. The 6 hour budget of
  the stopping rule above was set for another matrix and does not cover `diag4` plus `full12`; the
  operator states the budget actually spent in the paper. If the budget is cut, whole cells are
  dropped in this order: S0 stable WAN, S2, S8 cold and warm together, never single policies or
  seeds of a cell.

### Calibration and modeled campaign

* Calibration: the full four-network grid (LAN, WAN, WAN-c, WAN-mid), hold-out unchanged (shapes
  1024x1024 and 3072x768, network WAN-mid, n = 32). LayerNorm is now measured with the shared gain
  (18 rounds, as the formula prices it). A third evaluated model, "uncertainty-aware", reads the
  fitted model as B4 does: lanes are compared by mean plus one sigma (z = 1, declared here);
  its regression metrics equal the fitted model's by construction, only the lane, plan, regret and
  informative-pair columns can differ.
* Modeled campaign: all nine families, seeds 1..10, T in {8, 32}, as above, re-run after the
  calibration; per-epoch bytes and rounds use the exact batched formula
  (`candidates.plan_bytes_rounds`).
* Web (E6): cold TTFT, warm TTFT (second generation on one stack, reported only when the runner
  verified a warm start), inter-token latency, completion, cancel waste, resume recovery and the
  rejection counters; on the tiny byte-vocabulary model always (pipeline scale, not a model-scale
  claim) and on GPT-2 1-layer and 4-layer prefixes; 3 repeats, 10 rejection probes per kind.

### Entry point

`python/scripts/run_campaign.py --stage primitives|calibration|simulation|diag4|full12|web|tables`
(each stage resumable, logs under `experiments/logs/`).

### Operator decisions fixed before the first measured session (2026-10-11)

Written after the calibration stage was started and before any `diag4` or `full12` session ran.

* Concurrency: `diag4` runs with 6 workers, `full12` with 3 workers, every launched process capped
  at 2 BLAS threads. One level per matrix; levels are never pooled.
* Matrix reductions, made for wall-clock budget only (driver option `--skip`):
  the fixed all-generic plan is not run on the slow-link cells of `diag4` (S0 stable WAN, S1, S2,
  S3), where the dry run predicts 860 to 4140 s per session for it; it is run on the LAN cells
  (S0 stable LAN, S8 cold, S8 warm), 105 sessions in total. On those four cells "B0" in the
  measured table is therefore the fixed all-wmask plan, and the table says so. In `full12` the S1
  ramp cell is run for B0 fixed all-wmask and B4 only (B2 is measured against B4 on `diag4`),
  15 sessions in total.
* Order: C++ building-block benchmark, one smoke cell at 6 workers, `diag4`, `full12`, Web runs on
  the tiny model and on the GPT-2 1-layer and 4-layer prefixes, tables.
* Budget actually available: the campaign is left to run to completion; the wall-clock time spent
  is reported in the paper. Sessions that hit the 2 h per-session timeout are reported as aborted
  with the reason.
* Nothing in the analysis rules above is changed by these decisions.

### Erratum to the header of this file (2026-10-11)

The host line at the top names an Intel Core Ultra 9 285K with 24 logical processors. That line
was copied from the earlier artifact. The machine that runs every campaign of this revision is an
Intel Core i9-14900K (24 cores, 32 logical processors) with 192 GB RAM, Windows 11 Pro build
26200, with an NVIDIA RTX 4090 and an NVIDIA RTX 6000 Ada present and used by no secure path.
The paper reports this host.

### Operator decisions for the final campaign (2026-10-13)

Written after the first measured campaign (105 + 15 sessions, 2026-10-11) had been analysed and
before any session of the final campaign ran. The first campaign is kept as `raw/pilot/`; it is
evidence for no hypothesis and enters no table. It was used for the decisions below.

* Pilot findings that changed the code, in both implementations, each with a regression test:
  (1) the controller priced the remaining steps with the shapes of the current step instead of step
  by step as Eq. (1) of the paper is written, so a high bandwidth reading at the prefill deferred the
  installation of the weight masks by one step (3 of 57 adaptive sessions, every cold GPT-2 Web
  session); (2) the per-sample bandwidth estimator reached its 100 Gbit/s cap when one sleep tick of
  the shaper fell into the ping leg (15 of 432 LAN frames at or above 2.8 Gbit/s on a 1 Gbit/s link);
  it is replaced by one pooled estimate per probe, burst size over (median burst time minus minimum
  ping time), same traffic, same cap; (3) the Web stage ran without the BLAS thread cap, and two
  24-thread pools stalled each other on the warm prefill (warm first-token times of 24 to 218 s where
  the product takes 1 s); the Web driver now caps every process at 2 threads, as the session matrices
  do. The pilot's session, Web and modeled records are therefore not comparable with the final
  campaign.
* Concurrency: both matrices run with 3 workers. The pilot ran `diag4` at 6, where sessions with the
  identical plan and bytes differed by up to 33 % and the probe's 5th to 95th percentile spanned 0.61
  to 2.37 of the true bandwidth; at 3 workers the pilot `full12` spanned 0.90 to 1.58. One level per
  matrix, 2 BLAS threads per process, 3 processes per session.
* Trace durations: fixed per cell and shared by all policies and seeds of the cell, 1.2 times the
  median session length of the cell in the pilot, rounded to 10 s: `diag4` S0 stable LAN 450 s, S0
  stable WAN 650 s, S1 ramp 770 s, S2 RTT steps 1200 s, S3 two changes 590 s, S8 cold 450 s, S8 warm
  430 s; `full12` S0 stable LAN 410 s, S1 ramp 1550 s. In the pilot the duration followed each
  policy's own prediction, so the policies of one seed ran on different trace points and the S2 delay
  steps were over before the first token.
* One-worker repeat: the rule "repeated with one worker before any claim is made" is not executed;
  for 105 sessions it exceeds the budget. Instead the header of the dynamic table carries, for every
  printed difference, the Student t interval, whether the two policies executed the identical plan
  sequence in all seeds, and the largest difference among identical-plan pairs (the noise floor). A
  claim is made only where the bootstrap and the Student intervals exclude zero and the plans differ.
  Ties are reported as ties.
* Output match: every session record now carries the generated and the reference tokens, the first
  differing step and its margin, so a mismatch is reported from the record alone. Two of the 120
  pilot sessions mismatched. The cause consistent with all records is a wrap of the local
  truncation, which hits a truncated value x with probability |x| / 2^64; summed over a session of
  16 prompt and 8 generated tokens this gives a wrong token sequence in about 0.8 % of 4-layer and
  2.4 % of 12-layer sessions. The paper states this as a correctness limit; no protocol change.
* Unchanged: matrix reductions, policies, seeds, hypotheses, aggregation rules, Web scenarios. The
  tiny Web workload runs again in the final chain. Stages in order: primitives, C++ benchmark,
  modeled campaign (regenerated with the corrected pricing), `diag4`, `full12`, Web (tiny, GPT-2
  1-layer and 4-layer), tables. The calibration is not repeated: the controller fix does not change
  the fitted cost model.
* Entry point of the artifact: `scripts/run_campaign.py`; reruns write to `runs/<name>`, the shipped
  outputs live in `results/`. The wall-clock time spent is reported in the README and the paper.

### Amendment to the trace durations (2026-10-14)

Written before any session of the final campaign ran. The text above is unchanged.

* S2 and S3 are re-declared to 440 s (were 1200 s and 590 s). The rule "1.2 times the median pilot
  session" is circular for these two families, whose change points scale with the duration: the
  pilot S2 sessions were long because their trace had reached 100 ms by 215 s, and on the 1200 s
  trace the delay steps fall at 300, 600 and 900 s; a one-worker B4 session on that trace ended at
  333 s and sensed only the first step, at its last decode step. With 440 s the S2 steps fall at
  110, 220 and 330 s, and the S3 regime changes of seeds 1 to 3 between 61 and 290 s (seed 1 at
  288 and 290 s, seed 2 at 61 and 193 s, seed 3 at 71 and 182 s).
* Checked with the current code at 3 workers (pipeline checks, evidence for no hypothesis, kept out
  of `results/`): S2 seed 1 with B0 all-wmask, B1 and B4 took 827 to 847 s; in every session the
  decision of epoch 0 sensed the 0.25 ms delay, epoch 1 the 10 ms, epoch 2 the 40 ms and epoch 3 or
  4 onwards the 100 ms one-way delay. S3 seed 2 (the shortest pilot sessions) with B2, B4 and
  B4-no-install took 342 to 366 s; the change to LAN was sensed at epoch 1 and the change back to
  WAN-mid at epoch 4. The pilot S3 traces had nearly this duration (440.4 s, change points 0.1 %
  later), and every pilot S3 session outlasted its last change: at least 422 s (seed 1), 269 s
  (seed 2) and 590 s (seed 3) at 6 workers. S1 keeps 770 s and 1550 s: a ramp has no change point
  a session could miss.
* Unchanged: everything else, including the stage list (`scripts/run_campaign.py` runs exactly
  this list when no stage is named; a fresh calibration only with `--stage all`) and 3 workers in
  both matrices (now the script's default).

### Pointers in the text above (2026-10-14)

Some sections above name files that are not part of this artifact. The timing boundaries of
"INTERFACE_SPEC section 8" are those of the paper, Section 7.4; the security errata S1 and S2 of
"INTERFACE_SPEC section 12" are described in the paper, Section 5.3 and Appendix D (Disclosure);
`raw/pre_s1/` and `raw/invalid/` held records produced before the repair and are not shipped; "Web
(E6)" is the Web stage; `python/scripts/run_campaign.py` and `experiments/logs/` are now
`scripts/run_campaign.py` and `runs/<name>/logs/`.

### Execution note of the final campaign (2026-10-14)

Written after the final campaign ended; it changes no rule.

* Command: `python scripts/run_campaign.py --out-root results` (default stages, no calibration).
  Stage times: primitives 52 s, cpp_bench 130 s, simulation 377 s, diag4 18,576 s, full12 3,990 s,
  web 1,836 s, tables 2 s; 24,963 s (6.93 h) in total. The first campaign's stages took 35,363 s
  including the 16,437 s calibration.
* diag4 completed 104 of 105 sessions. The session S0 stable LAN, fixed all-generic plan, seed 1,
  wrote both party records (393 s, reference tokens), but the launcher did not receive one party's
  result and waited out the 7200 s timeout; its record is an abort, and the tables exclude it as
  the aggregation rules say. Because diag4 then exited 1, full12, web and tables ran in a second
  invocation (`--stage full12,web,tables`) 256 s later, with the same code and flags.
* All 119 completed sessions released the float reference's tokens.
* During part of the campaign an unrelated job of another project ran on the host (two
  single-threaded processes). It was not stopped.
* After the campaign the modeled records were stored as `.jsonl.gz` (byte-identical content), the
  per-session working directories (ledgers, dealer logs, detail files, keys of the finished test
  dealers) were moved out of the artifact, and the table generator was extended to print the
  full12 cells in the dynamic table; no record changed.

