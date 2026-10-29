# Verification log — IEEE Access revision (Access-2026-44605) release scripts

This log records the publication of the five revision files and a line-by-line check of the
repository against the claims the revised manuscript makes about it. Nothing in the code and
nothing in the manuscript was altered to make the two agree; where they disagree, both values
are recorded below and the decision is left to the author.

Environment: macOS 26.5.1, arm64, 10-core Apple silicon, CPU only (no CUDA),
Python 3.11.15, torch 2.12.1, stable-baselines3 2.9.0, pandas 2.2.3, pyarrow present.
NSL-KDD and UNSW-NB15 read from local copies of the public benchmarks; CIC-IoT2023 not fetched
in this session.

## 1. Files published by this change

| File | Backs reviewer answer | Paper location |
|---|---|---|
| `rev_threshold_protocol.py` | R2.3 validation-based threshold protocol, R2.9 operational metrics | Section V-L, Table XXI |
| `rev_convergence.py` | R2.7 training budget and convergence | Section V-M, Table XXII, Fig. 13 |
| `rev_adversary_strategies.py` | R2.5 three further adversary strategies | Section VI-B, Table XXVII, Fig. 16 |
| `measure_rl_training_time.py` | R1.3 policy training cost | Table XX, rows marked (CPU) |
| `rl_training_time.json` | R1.3 the measured output | Table XX |

`measure_rl_training_time.py` gained three optional arguments (`--train`, `--test`, `--out`)
so `reproduce.sh` can pass `$DATASET_DIR`; the defaults reproduce its previous behaviour
exactly and the measurement itself is unchanged. No phase, exp or pipeline script was modified.

## 2. R1.2 (a) — the autonomous-response environment

**The specification in Section IV and Table VII is implemented in full — but by
`exp5_adaptive_adversary.py`, not by `phase6_response.py`.** `AdaptiveSim` in
`exp5_adaptive_adversary.py` satisfies every clause, including the two that
`phase6_response.py` contradicts. `rev_adversary_strategies.py` subclasses `AdaptiveSim`, so the
Section VI-B variants inherit the specified environment. The manuscript names no script for
Section IV, and Table VI labels the row "(Phase 6)", which points a reviewer at the one file
that does not match.

| Paper claim (Section IV / Table VII) | `exp5_adaptive_adversary.py` | `phase6_response.py` |
|---|---|---|
| One host, T = 50 discrete steps | verified — `T=50` (L36) | verified — `T=50` (L42), `--episode-T 50` (L132) |
| Campaign present at reset with probability 0.5 | verified — `p_attack=0.5` (L36), `rng.random() < s.p_attack` (L42) | verified — `--p-attack 0.5` (L133), L81 |
| Onset drawn uniformly from the first half | verified — `rng.integers(0, max(1, s.T // 2))` (L43) | verified — same expression (L82) |
| Hidden campaign state never revealed | verified — `attack`/`start` absent from `_obs` (L62–67) | verified — absent from `_obs` (L59–62) |
| One standardized anomaly score per step, from the attack-error pool while the campaign runs and the normal pool otherwise | verified — `_err` (L53–60); pools standardized on normal mean/sd (L173–175) | verified — `_err` (L56–57); pools (L156–158) |
| Five-dimensional observation: current score, mean and max of last five, previous action, normalized elapsed time | verified — exactly `[cur, h.mean(), h.max(), s.resp/2.0, s.t/s.T]` (L67), `maxlen=5` (L47), `Box(…,(5,))` (L94) | **discrepancy — six dimensions** `[s.cur, h.mean(), h.max(), s.comp, s.resp/2.0, s.t/s.T]` (L61), `Box(…,(6,))` (L100) |
| Compromise level withheld from the agent | verified — explicit in code comment (L63–64); `s.comp` absent from `_obs` | **discrepancy — compromise is observation element 3** (L61) |
| Actions monitor / throttle / block | verified — `Discrete(3)` (L93), `AVAIL`/`INC` (L28–29) | verified — `Discrete(3)` (L99), `costs`/`incs` (L43) |
| Availability cost 0.00 / 0.05 / 0.20 | verified — `AVAIL` (L28) | verified — `costs=(0.0, 0.05, 0.20)` (L43) |
| Compromise increment 0.15 / 0.05 / 0.00 while active | verified — `INC` (L29) | verified — `incs=(0.15, 0.05, 0.0)` (L43) |
| Breach weight 5.0 × cost ratio per unit compromise | verified — `s.Wc = 5.0 * cost_ratio` (L38) | partial — `Wc=5.0` fixed, no cost-ratio parameter (L42) |
| Breach penalty 10.0 × cost ratio, charged once when compromise first reaches 1.0 | verified — `s.bp = 10.0 * cost_ratio` (L38); charged once via `and not s.breached` (L74–75) | partial — `breach_pen=10.0` fixed (L42); also ends the episode, `done = True` (L72), which Table VII does not state |
| Adversary reaction: 3-step cooldown in which the campaign mimics normal traffic with probability eps | verified — `COOLDOWN = 3` (L30), `s.cool > 0 and s.evcoin[s.t] < s.eps` (L57), `s.cool = COOLDOWN` (L78) | **discrepancy — absent**; no evasion, cooldown or eps anywhere in the file |
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

## 5. Discrepancies for the author to decide

None of these were changed in either the code or the manuscript.

**D1 — Table VI labels the autonomous-response row "(Phase 6)", but the Table VII specification
it refers to is implemented by `exp5_adaptive_adversary.py`.** `phase6_response.py` differs on
three counts: a six-dimensional observation rather than five, the compromise level visible to
the agent rather than withheld, and no reacting adversary at all. A reviewer following the
"Phase 6" label to `phase6_response.py` in answer to R1.2 will find a file that contradicts the
specification on exactly the two points R1.2 asked about (what the agent observes, and how the
adversary behaves). The published result is `exp5`'s. The README added in this change states
which file implements the specification, which addresses the repository side; the Table VI row
label is the author's to resolve.

**D2 — `exp5_runtime.py` is named in the repository but does not exist in it.** Section V-K says
the runtime figures were measured by "script `exp5_runtime.py` in the repository", and the
Table XX caption repeats it: "Rows not marked (CPU) were measured on a single Colab T4 GPU by
`exp5_runtime.py`". That file is absent from `origin/main`, absent from the working tree, and
absent from every git ref and every directory under `~/Documents/Documents/PHD` (searched). It is
not one of the five files in this change. Consequently the Colab-T4 rows of Table XX — 32.0 s
autoencoder training, the p50/p95/p99 inference latencies, the 0.15 µs threshold cost, the
0.58 ms PPO cost, the 34,618 and 25,478 parameter counts, the 56 MB peak GPU and 2.1 GB peak RAM
— have no released script behind them, and the same table's R1.3 rows now do. This is the same
class of problem this change was opened to fix, in the same table.

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

## 7. Execution checks

Each script was run before being committed. The two multi-seed experiments were run at reduced
epoch, seed and timestep budgets — enough to exercise every code path end to end, not enough to
reproduce the published tables. Full reproduction was not attempted in this session; the
estimated cost is noted below.

| Script | Command run | Result |
|---|---|---|
| `measure_rl_training_time.py` | full published configuration (100 AE epochs, 80k and 300k budgets, seed 42) | **PASS** — completed, wrote valid JSON, figures in Section 6 |
| `rev_threshold_protocol.py` | `--epochs 3 --rl-seeds 1 --rl-timesteps 4096`, NSL-KDD + UNSW-NB15 | **PASS** — all four comparators plus the five-point prevalence sweep written to `results/rev_threshold_protocol.csv`; CIC-IoT2023 absent and correctly reported as "inputs missing -> skipped" |
| `rev_convergence.py` | `--epochs 3 --timesteps 20000 --seeds 1`, NSL-KDD + UNSW-NB15 | **PASS** — checkpoint probe fired at both 10,000 and 20,000 timesteps on both datasets, full operating point logged per checkpoint to `results/rev_convergence.csv`, tuned-threshold and logistic-regression baselines computed; CIC-IoT2023 correctly skipped |
| `rev_adversary_strategies.py` | `--ae-epochs 3 --seeds 1 --timesteps 4096 --eval-episodes 150`, NSL-KDD | **PASS** — all six variants (four adversary strategies, two cost ratios) trained, evaluated and written to `results/rev_adversary_strategies.csv`, with the delay and censoring columns populated; see the 1-seed caveat below |

Caveat on the `rev_adversary_strategies.py` check: it was run with one seed, which makes the
variance-based statistics degenerate by construction — scipy and numpy emit
`Degrees of freedom <= 0` and `invalid value` warnings, the confidence interval collapses onto the
point estimate, and one variant reports `rl_delay = nan` because at 4,096 timesteps the policy
never responded in any attack episode, so every episode was censored. These are artefacts of the
reduced budget, not defects: the script's own default is six seeds (L182) and the paper's budget
is 100,000 timesteps, roughly 24 times what was used here. The smoke-run margins therefore do not
resemble Table XXVII and are not offered as a reproduction of it — two variants even invert
(scheduled dormancy loses here, wins in the paper), which is what a 4% training budget should be
expected to do.

At the smoke-test budget `rev_threshold_protocol.py` already shows the qualitative pattern
Section V-L reports: on UNSW-NB15 the PPO controller's F1 (0.7588) sits beside the
validation-selected threshold's (0.7643) while running at a materially worse false-positive rate
(0.698 against 0.572) and issuing more alerts per thousand flows (842 against 756) — the paper's
"buys F1 with false positives" finding. These are 3-epoch numbers and are not the published
values.

## 8. What the repository still cannot reproduce

1. **The Colab-T4 rows of Table XX** — no script exists for them (D2). Everything else in
   Table XX, the four rows marked (CPU), is reproducible with `measure_rl_training_time.py` as
   of this change.
2. **CIC-IoT2023 results were not re-run in this session.** `ciciot2023_pipeline.py` fetches the
   dataset from the Hugging Face Hub and was not exercised here; the three scripts that take
   `--ciciot-*` flags were verified to skip cleanly when the files are absent, not to run on
   them. The existing log in `repo_verification.md` records a prior exact reproduction of that
   pipeline.
3. **The published tables of Sections V-L, V-M and VI-B were not reproduced**, only executed.
   Full runs are expensive on CPU: Section V-M alone is 3 datasets × 10 seeds × 300,000
   timesteps, which at the 0.26 s per 1,000 timesteps measured in Section 6 is roughly 6.5 hours
   of policy training before the autoencoder fits and the per-checkpoint test inference. Section
   V-L adds 30 PPO trainings at 80,000 steps and Section VI-B 36 at 100,000.
4. **The figures** are produced by `build_paper_figures.py`, which was not run here.
