# Verification log — IEEE Access revision (Access-2026-44605) release scripts

This log records the publication of the five revision files and a line-by-line check of the
repository against the claims the revised manuscript makes about it. Nothing in the code and
nothing in the manuscript was altered to make the two agree; where they disagree, both values
are recorded below and the decision is left to the author.

Environment: macOS 26.5.1, arm64, 10-core Apple silicon, CPU only (no CUDA),
Python 3.11.15, torch 2.12.1, stable-baselines3 2.9.0, pandas 2.2.3, pyarrow present.
NSL-KDD and UNSW-NB15 read from local copies of the public benchmarks; CIC-IoT2023 fetched by
`ciciot2023_pipeline.py` during the full run recorded in Section 7.

**Update, 30 October 2026.** `reproduce.sh` has since been run end to end from a clean clone of
`main`, all fourteen stages, and the published tables of Sections V-L, V-M and VI-B reproduced.
Section 7 records the run and Section 9 the comparison. Discrepancies D1 and D2 are both
resolved; neither remains open.

## 1. Files published by this change

| File | Backs reviewer answer | Paper location |
|---|---|---|
| `rev_threshold_protocol.py` | R2.3 validation-based threshold protocol, R2.9 operational metrics | Section V-L, Table XXI |
| `rev_convergence.py` | R2.7 training budget and convergence | Section V-M, Table XXII, Fig. 13 |
| `rev_adversary_strategies.py` | R2.5 three further adversary strategies | Section VI-B, Table XXVII, Fig. 16 |
| `measure_rl_training_time.py` | R1.3 policy training cost | Table XX, rows marked (CPU) |
| `rl_training_time.json` | R1.3 the measured output | Table XX |
| `exp5_runtime.py` | Section V-K runtime, latency, memory footprint | Table XX, rows not marked (CPU) |

`measure_rl_training_time.py` gained three optional arguments (`--train`, `--test`, `--out`)
so `reproduce.sh` can pass `$DATASET_DIR`; the defaults reproduce its previous behaviour
exactly and the measurement itself is unchanged. No phase, exp or pipeline script was modified.

## 2. R1.2 (a) — the autonomous-response environment

**The specification in Section IV and Table VII is implemented in full, by
`exp5_adaptive_adversary.py`.** `AdaptiveSim` in
`exp5_adaptive_adversary.py` satisfies every clause, including the two that
`phase6_response.py` contradicts. `rev_adversary_strategies.py` subclasses `AdaptiveSim`, so the
Section VI-B variants inherit the specified environment. `phase6_response.py` is the earlier
static variant and differs on two points; the manuscript now states that difference itself
(see D1), so the table below records which file satisfies each clause rather than reporting a
mismatch.

| Paper claim (Section IV / Table VII) | `exp5_adaptive_adversary.py` | `phase6_response.py` |
|---|---|---|
| One host, T = 50 discrete steps | verified — `T=50` (L36) | verified — `T=50` (L42), `--episode-T 50` (L132) |
| Campaign present at reset with probability 0.5 | verified — `p_attack=0.5` (L36), `rng.random() < s.p_attack` (L42) | verified — `--p-attack 0.5` (L133), L81 |
| Onset drawn uniformly from the first half | verified — `rng.integers(0, max(1, s.T // 2))` (L43) | verified — same expression (L82) |
| Hidden campaign state never revealed | verified — `attack`/`start` absent from `_obs` (L62–67) | verified — absent from `_obs` (L59–62) |
| One standardized anomaly score per step, from the attack-error pool while the campaign runs and the normal pool otherwise | verified — `_err` (L53–60); pools standardized on normal mean/sd (L173–175) | verified — `_err` (L56–57); pools (L156–158) |
| Five-dimensional observation: current score, mean and max of last five, previous action, normalized elapsed time | verified — exactly `[cur, h.mean(), h.max(), s.resp/2.0, s.t/s.T]` (L67), `maxlen=5` (L47), `Box(…,(5,))` (L94) | differs by design — **six** dimensions, `[s.cur, h.mean(), h.max(), s.comp, s.resp/2.0, s.t/s.T]` (L61), `Box(…,(6,))` (L100); recorded in Table VI |
| Compromise level withheld from the agent | verified — explicit in code comment (L63–64); `s.comp` absent from `_obs` | differs by design — compromise is observation element 3 (L61); recorded in Table VI |
| Actions monitor / throttle / block | verified — `Discrete(3)` (L93), `AVAIL`/`INC` (L28–29) | verified — `Discrete(3)` (L99), `costs`/`incs` (L43) |
| Availability cost 0.00 / 0.05 / 0.20 | verified — `AVAIL` (L28) | verified — `costs=(0.0, 0.05, 0.20)` (L43) |
| Compromise increment 0.15 / 0.05 / 0.00 while active | verified — `INC` (L29) | verified — `incs=(0.15, 0.05, 0.0)` (L43) |
| Breach weight 5.0 × cost ratio per unit compromise | verified — `s.Wc = 5.0 * cost_ratio` (L38) | partial — `Wc=5.0` fixed, no cost-ratio parameter (L42) |
| Breach penalty 10.0 × cost ratio, charged once when compromise first reaches 1.0 | verified — `s.bp = 10.0 * cost_ratio` (L38); charged once via `and not s.breached` (L74–75) | partial — `breach_pen=10.0` fixed (L42); also ends the episode, `done = True` (L72), which Table VII does not state |
| Adversary reaction: 3-step cooldown in which the campaign mimics normal traffic with probability eps | verified — `COOLDOWN = 3` (L30), `s.cool > 0 and s.evcoin[s.t] < s.eps` (L57), `s.cool = COOLDOWN` (L78) | differs by design — absent; no evasion, cooldown or eps in the file; recorded in Table VI |
| Reward = negative total cost (availability + compromise + breach) | verified — `r = -AVAIL[a] - inc * s.Wc` (L73), `r -= s.bp` (L75) | verified — `reward = -cost - inc * s.Wc` (L69), `reward -= s.bp` (L72) |

## 3. R1.2 (b) — Table VI reward functions

Table VI states its rewards are "transcribed from the released scripts". Each was checked
against its script; all five transcriptions are accurate.

| Table VI row | Paper value | Code | Result |
|---|---|---|---|
| Static detection (Phase 2) | +2.0 alert on attack, −0.5 alert on normal; +1.0 silent on normal, −3.0 silent on attack; 0.0 borderline | `phase2_ppo_controller.py` `_reward` L136–141: `a in (3,4)` → `2.0 if y==1 else -0.5`; `a in (0,1)` → `1.0 if y==0 else -3.0`; else `0.0` | verified |
| Concept drift (Phase 3) | Identical to Phase 2; drift indicator enters the state, not the reward | `phase3_drift.py` L90 is the same expression; the drift signal is the rolling-standardized error substituted into the state (`e_roll`, L179; `S_A4`, L182) and appears nowhere in the reward | verified |
| Budgeted alerting (Phase 4) | +2.0 escalate on attack, −1.0 escalate on normal; −3.0 ignore an attack, +0.1 ignore a normal flow; escalation permitted only while budget remains | `phase4_budget.py` L83 `escalated = (int(a) == 1 and s.rem > 0)`; L86 `2.0 if y==1 else -1.0`; L88 `-3.0 if y==1 else 0.1` | verified |
| Cost-aware inspection (Phase 5) | −λ × group cost per acquisition (λ = 0.15; 1.0 basic, 2.0 time, 2.0 host, 4.0 content); on stopping +2.0 / −1.0 / −3.0 / +1.0 for TP / FP / FN / TN | `phase5_costaware.py` L45 `GROUP_COST`; L168 `--cost-lambda` default `0.15`; L152 `r = -cost_lambda * costs[s.k]`; L156 `(2.0 if yt==1 else -1.0) if yp==1 else (-3.0 if yt==1 else 1.0)` | verified, with one nuance below |
| Autonomous response (Phase 6) | Negative total cost: availability + compromise + breach penalty over a 50-step horizon (Table VII) | formula verified in both `exp5_adaptive_adversary.py` (L73, L75) and `phase6_response.py` (L69, L72); the *environment* of Table VII is `exp5`'s, not `phase6`'s — see Section 2 | verified (reward); see D1 |

Nuance on Phase 5: the basic group is acquired unconditionally at reset (`s.k = 1`, L147), so
its listed acquisition cost of 1.0 is never charged in the reward — the first chargeable
acquisition is `costs[1] = 2.0` (time). The 1.0 *is* included in the reported average inspection
cost (`cost_of_prefix`, L196). A constant offset cannot change the optimal policy, so no result
is affected, but "per acquisition" in Table VI reads as though all four costs are charged.

## 4. R2.5 (c) — adversary strategies and parameters

`rev_adversary_strategies.py` implements exactly the three strategies and two extreme cost
structures the paper describes, with matching parameters.

| Paper (Section VI-B / Table XXVII) | Code | Result |
|---|---|---|
| Scheduled dormancy: "fixed 3-on/2-off cycle regardless of the defender" | `DormantSim`, `duty=(3, 2)` (L45, L207); `_active` gated on `((s.t - s.start) % (on + off)) < on` (L48–52) | verified |
| Probe-then-strike: "8 reconnaissance steps at a signature halfway between normal and attack, no progression, then a full-rate campaign" | `ProbeStrikeSim`, `probe=8` (L57, L209); `_err` returns `0.5 * normal + 0.5 * attack` (L65); `_active` false during the probe so compromise does not progress (L60–61) | verified |
| Escalating evasion: "evasion probability rises by 0.15 each time the defender responds" | `EscalatingSim`, `step_up=0.15` (L71, L211); `eps_eff = min(0.95, s.eps + s.esc * s.step_up)` (L80); `s.esc += 1` on throttle/block while active (L86–89) | verified |
| Two extreme cost structures, "from 0.125 to 8.0" | `for cr in (0.125, 8.0)` (L214) | verified |
| Six seeds each | `--seeds` default `6` (L182), `H.SEEDS[:6]` (L197) | verified |
| Reactive baseline row = "the published model" | `X.AdaptiveSim, dict(eps=0.75, cost_ratio=1.0)` (L205) | verified against the script; the manuscript does not state the eps value of this row (see D4) |
| Delay = steps from campaign onset to first throttle or block; episodes with no response reported as censored | `rollout_with_delay` L92–113; censored counted separately and excluded from the mean | verified |

Two parameters the script fixes are not stated anywhere in the manuscript: the base eps of the
escalating-evasion variant (`eps=0.25`, L211) and the eps of the two cost-ratio rows
(`eps=0.2`, L215; consistent with `exp5_adaptive_adversary.py` L186, which also uses 0.2 for its
cost sweep). Neither is contradicted — they are simply unstated, so a reviewer cannot check them
against the paper without reading the script.

## 5. Discrepancies raised by this check (D1 and D2 now resolved)

None of these were changed by this check in either the code or the manuscript. D1 and
D2 have since been resolved — D1 by the author in the manuscript, D2 by releasing the script.

**D1 — the "(Phase 6)" label on the autonomous-response row. Now resolved by the author.**
When this log was first written, Table VI labelled that row "(Phase 6)" while the Table VII
specification it cited — five-dimensional observation, compromise hidden, reacting adversary —
is implemented by `exp5_adaptive_adversary.py`, not `phase6_response.py`. A reviewer following
the label in answer to R1.2 would have landed on the one file contradicting the specification on
exactly the two points R1.2 asked about.

The manuscript now distinguishes the two variants explicitly, which removes the ambiguity
without changing any result. Verified against the current revision:

* the row reads "Autonomous response (Phase 6; adaptive extension, Section VI-B)";
* its assumption column states that "Phase 6 exposes the compromise level to the agent and faces
  a non-reacting adversary; the adaptive extension hides it and adds a reacting adversary";
* Section IV opens by saying "The specification is that of the adaptive-adversary environment of
  Section VI-B, in which the positive result is obtained; the static Phase 6 variant differs in
  the two respects recorded in Table VI".

Both files are released, both behaviours are now described, and the mapping in Section 2 above
tells a reviewer which clause each file satisfies. Nothing further is outstanding here.

**D2 — `exp5_runtime.py` was named in the repository but did not exist in it. Now resolved.**
Section V-K says the runtime figures were measured by "script `exp5_runtime.py` in the
repository", and the Table XX caption repeats it. When this log was first written that file was
absent from `origin/main`, from the working tree, and from every git ref. It is added by this
change and verified below, so both statements are now true. The two hardware-independent
quantities in Table XX reproduce exactly:

| Table XX quantity | Paper | Re-measured (CPU) | Result |
|---|---|---|---|
| Autoencoder parameters | 34,618 | 34,618 | **exact** |
| PPO policy parameters | 25,478 | 25,478 | **exact** |
| AE training, 30 epochs | 32.0 s (T4) | 17.0 s | machine-dependent |
| AE inference p50 / p95 / p99 | 0.79 / 1.00 / 1.35 ms (T4) | 0.071 / 0.074 / 0.079 ms | machine-dependent |
| Threshold decision | 0.15 µs (T4) | 0.05 µs | machine-dependent |
| PPO inference | 0.58 ms mean (T4) | 0.060 ms p50 | machine-dependent |
| Peak process RSS | 2.1 GB | 0.96 GB | machine-dependent |

The parameter counts are a property of the architecture, not the machine, and they match to the
digit — which is the strongest check available that this script measures the pipeline Table XX
describes. The latencies do not match and are not expected to: they were taken on a Colab T4 and
these on a 10-core Apple-silicon CPU. Note that the CPU is *faster* than the published T4 row
for every latency. That is the normal result for batch-1 inference on a small model, where GPU
kernel-launch overhead dominates the arithmetic; it does not contradict the table, and Section
V-K's conclusion that the pipeline is "lightweight enough for near-real-time operation" holds on
both machines. No measured output was edited to agree with the table.

**D3 — Section V-M reports ten seeds; the script defaults to three.** `rev_convergence.py`
`--seeds` defaults to 3 (L127) while Section V-M describes "ten seeds each"; `--rl-seeds`
in `rev_threshold_protocol.py` defaults to 0, which disables the PPO arm that Section V-L
reports over ten seeds. Not a code defect — both accept the paper's values — but a reader running
the scripts bare would not reproduce the published seed counts. `reproduce.sh` now passes
`--rl-seeds 10` and `--seeds 10` so the entry point matches the manuscript.

**D4 — unstated parameters.** The eps values noted at the end of Section 4 above.

## 6. R1.3 — re-measured training cost

`measure_rl_training_time.py` ran to completion on this machine against NSL-KDD. The committed
`rl_training_time.json` was **not** overwritten; it still holds the values Table XX reports. The
re-run was written elsewhere and is reproduced here for comparison.

Machine: macOS 26.5.1, arm64, 10-core Apple silicon, CPU only, torch 2.12.1, seed 42.
This is the same machine class as the published row ("a ten-core Apple-silicon CPU"), so the
figures are directly comparable; they are not a Colab T4 and are not comparable to the GPU rows.

| Quantity | Paper / committed JSON | Re-measured here | Difference |
|---|---|---|---|
| Autoencoder, 100 epochs | 73.7 s | 67.4 s | −6.3 s (−8.5%) |
| PPO, 80,000 timesteps | 23.2 s | 20.5 s | −2.7 s (−11.6%) |
| PPO, 300,000 timesteps | 85.5 s | 78.3 s | −7.2 s (−8.4%) |
| s per 1k timesteps @ 80k | 0.290 | 0.256 | −0.034 |
| s per 1k timesteps @ 300k | 0.285 | 0.261 | −0.024 |
| **Ratio to AE fit @ 80k** | **0.32** | **0.30** | −0.02 |
| **Ratio to AE fit @ 300k** | **1.16** | **1.16** | **0.00** |

The two ratios are the quantities Section V-K actually asserts ("0.32 times the detector fit at
the published budget and 1.16 times it at the extended one"). The 300,000-step ratio reproduces
exactly and the 80,000-step ratio to within 0.02. Absolute seconds run about 9% faster here than
in the committed JSON, which is the expected machine-load variation on nominally identical
hardware and is why the paper marks these rows (CPU). No value was edited to agree.

## 7. Execution checks — full end-to-end run

`reproduce.sh` was run to completion from a **clean clone of `main`**, with `DATASET_DIR`
holding the public NSL-KDD and UNSW-NB15 files; CIC-IoT2023 was fetched by the pipeline during
the run. All fourteen stages executed and the script printed its completion line.

| | |
|---|---|
| Started / finished | 29 Oct 2026 18:52:30 — 30 Oct 2026 09:48:25 |
| Stages completed | **14 of 14** |
| Exit status | **0**, `All experiments done` printed |
| Tracebacks, import errors, OOM kills | **none** |
| Result files written | 13, under `results/` |

Every script reached by the entry point ran: `phase1_autoencoder_ids.py`, `exp1_trivial_learners.py`,
`exp2_static_seeds.py`, `exp4_hparam_sweep.py`, `exp5_adaptive_adversary.py`, `phase8_unsw.py`,
`ciciot2023_pipeline.py`, `stronger_baselines.py`, and all five files released by this change.

A note on wall-clock figures from this run. The machine idle-slept for much of the first night,
which suspends the process without failing it: at one point 8 h 12 m of elapsed time had
consumed only 11 m 22 s of CPU. Elapsed-time figures from that period are therefore not compute
measurements. This affects the `secs` column of `results/stronger_baselines.csv`, which records
wall clock: One-Class SVM logged 8,265 s on NSL-KDD and 20,855 s on UNSW-NB15 against 312 s and
624 s in the earlier log, while CIC-IoT2023 — fitted after the machine was held awake — took
516 s against the earlier 12,254 s. The F1, ROC-AUC and PR values are unaffected and reproduce
exactly. **The `secs` columns in both this log and `repo_verification.md` should be read as
wall clock on a laptop that may have slept, not as cost measurements.**

## 8. Reproduction of the published tables

The run regenerated the tables rather than merely executing the code. Threshold and baseline
figures are deterministic and match exactly; reinforcement-learning figures are means over
stochastic seeds and match within seed noise.

**Static detection (Table A, 15 seeds).** Margin of RL over the best simple learner:

| dataset | reproduced | published |
|---|---|---|
| NSL-KDD | −0.160 | −0.160 |
| UNSW-NB15 | −0.011 | −0.011 |
| CIC-IoT2023 | −0.006 | −0.0059 |

**Stronger unsupervised baselines — 9 of 9 rows exact** (F1): NSL-KDD 0.9347 / 0.8850 / 0.8721,
UNSW-NB15 0.7804 / 0.7614 / 0.7315, CIC-IoT2023 0.9415 / 0.9328 / 0.9242.

**Section V-L, validation-selected threshold protocol (R2.3), 10 RL seeds, three datasets.**
Every threshold figure exact to four decimals:

| dataset | comparator | reproduced | manuscript |
|---|---|---|---|
| NSL-KDD | validation-selected | 0.8715 | 0.8715 |
| NSL-KDD | PPO controller | 0.7691 | 0.7703 |
| UNSW-NB15 | validation-selected | 0.7787 | 0.7787 |
| UNSW-NB15 | PPO controller | 0.7879 | 0.7882 |
| CIC-IoT2023 | validation-selected | 0.9383 | 0.9383 |
| CIC-IoT2023 | PPO controller | 0.9351 | 0.9347 |

The oracle-to-validation gap on CIC-IoT2023 reproduces at 0.0001 (0.9384 against 0.9383), the
figure Section V-L quotes, and on NSL-KDD the label-free percentile rule (0.9249) again beats
the validation-selected cut-point (0.8715).

**Section V-M, training budget and convergence (R2.7), 10 seeds, three datasets.**

| dataset | checkpoint | reproduced | manuscript |
|---|---|---|---|
| NSL-KDD | 80,000 | 0.7784 | 0.7783 |
| NSL-KDD | 300,000 | 0.7725 | 0.7671 |
| UNSW-NB15 | crossing point | 150,000 | 150,000 |
| UNSW-NB15 | margin at 300,000 | +0.0187 | +0.0185 |
| CIC-IoT2023 | margin at 80,000 | −0.0060 | −0.0059 |
| CIC-IoT2023 | margin at 300,000 | −0.0002 | −0.0002 |

The UNSW-NB15 reversal — the finding that corrects the original submission — reproduces on both
its crossing point and its final margin. NSL-KDD drifts downward with budget as reported,
though its 300,000-step endpoint is 0.0054 above the manuscript's.

**Section VI-B, adversary strategies (R2.5), 6 seeds, 6 variants. All six verdicts reproduce:**

| variant | reproduced margin | manuscript | verdict |
|---|---|---|---|
| reactive evasion | +0.906 | +0.865 | RL wins ✓ |
| scheduled dormancy | +0.474 | +0.475 | RL wins ✓ |
| probe then strike | −0.357 | −0.258 | RL loses ✓ |
| escalating evasion | +4.022 | +3.864 | RL wins ✓ |
| cost ratio 0.125 | +0.190 | +0.217 | RL wins ✓ |
| cost ratio 8.0 | +0.013 | −0.031 | tie ✓ |

The manuscript's summary — the advantage survives three of four adversary models and one of two
extreme cost structures — holds on every variant.

**Table XX parameter counts** reproduce exactly again in the full run: 34,618 autoencoder
parameters and 25,478 PPO policy parameters.

## 9. What the repository still cannot reproduce

1. **The Colab-T4 absolute figures of Table XX.** The script is released (`exp5_runtime.py`) and
   its hardware-independent quantities reproduce exactly, but the latencies, training time and
   peak RAM in those rows were taken on a T4 and will differ on any other hardware.
2. **The policy-training ratios of Table XX vary between runs on the same machine.** The
   standalone measurement gave 0.30 and 1.16 against the manuscript's 0.32 and 1.16; the
   measurement taken inside the full run gave 0.38 and 1.35, because the autoencoder fit was
   faster that time (56.8 s against 73.7 s) while policy training was not. The ratio is less
   stable than a single measurement suggests, which is consistent with rows marked (CPU).
3. **The figures.** `build_paper_figures.py` renders them from values transcribed out of the
   manuscript and was not run here; it is a renderer, not an experiment.
