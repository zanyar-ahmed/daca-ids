# Experiments — script ↔ result map (for the high-journal protocol)

Every number is produced by a script here on a fixed seed (reproducible). Run on Colab.
See `V1/experiment_protocol.md` for the full design and where each result lands in the paper.

## Shared
- `exp_harness.py` — determinism (`set_all_seeds`), `metrics()`, CIs (`t_ci`,`boot_ci`),
  significance (`compare_to_constant`, `holm`), vector-figure style (`set_pub_style`). (Protocol #0/#3/#7)

## Built and runnable now
| Protocol | Script | Produces | Status |
|----------|--------|----------|--------|
| #1 Trivial-learner battery | `exp1_trivial_learners.py` | Table B (LR-on-e, best-τ-on-e vs PPO, 2 datasets) | ✅ ready — run on Colab |
| #3 Statistical tests | `exp_harness.compare_to_constant` | p-values + Cohen's d for Table A | ✅ done for Phase 6/7 |
| #8 Repro (partial) | `requirements.txt`, this README | reproducibility statement | ✅ |

Phase scripts `phase1..phase9` produce Phases 1–9 (already run).

## Phase 6/7 statistics (already computed, real)
RL margin over tuned baseline: **−0.062**, 95% CI **[−0.097, −0.026]**,
Wilcoxon **p=0.006**, t-test p=0.003, **Cohen's d=−1.25**, 1/10 seeds won → RL significantly worse.

## How to run Experiment 1 (the flagship)
Needs in Drive: `Train_data.csv`, `Test_data.csv` (NSL-KDD) and the two UNSW-NB15 parquet files.
```python
!python exp1_trivial_learners.py --epochs 40
```
→ prints Table B and saves `results/exp1_trivial.csv`.

## Table ↔ script map (all built)
| Result | Script | Run on Colab |
|--------|--------|--------------|
| Table B (trivial learners, +LR on z,e) | `exp1_trivial_learners.py` | `!python exp1_trivial_learners.py --epochs 40` |
| Table A (15-seed RL vs best simple learner) | `exp2_static_seeds.py` | `!python exp2_static_seeds.py --epochs 40 --timesteps 80000` |
| Table C (hyperparameter sweep) | `exp4_hparam_sweep.py` | `!python exp4_hparam_sweep.py --epochs 40 --timesteps 80000` |
| Table D (adaptive adversary + cost ratio) | `exp5_adaptive_adversary.py` | `!python exp5_adaptive_adversary.py --seeds 8 --timesteps 100000` |
| Phase 6/7 stats | `exp_harness.compare_to_constant` | (done: p=0.006, d=−1.25) |

## Reproducibility (#8)
- `requirements.txt`, `Dockerfile`, `reproduce.sh` (regenerates every table; set `DATASET_DIR`).
- Vector figures (#7): `exp_harness.set_pub_style()` + save figures as `.pdf` (TrueType, 300 dpi).

## Deferred (future work)
- #6 CICIDS2017 third dataset — needs the cleaned (WTMC-2021) release downloaded + preprocessed;
  listed as future work (days of cleaning for modest payoff).
