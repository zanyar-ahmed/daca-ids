# Experiments — script ↔ result map (for the high-journal protocol)

Every number is produced by a script here on a fixed seed (reproducible). Run on Colab.
See `V1/experiment_protocol.md` for the full design and where each result lands in the paper.

## Shared
- `exp_harness.py` — determinism (`set_all_seeds`), `metrics()`, CIs (`t_ci`,`boot_ci`),
  significance (`compare_to_constant`, `holm`), vector-figure style (`set_pub_style`). (Protocol #0/#3/#7)

## Built and runnable now
| Protocol | Script | Produces | Status |
|----------|--------|----------|--------|
| #1 Trivial-learner battery | `exp1_trivial_learners.py` | Table B (LR-on-e, best-τ-on-e vs PPO, up to 3 datasets) | ✅ ready — run on Colab |
| #6 Third dataset (CIC-IoT2023) | `ciciot2023_pipeline.py` | Section V-I, Table XV, Fig. 11 | ✅ ready — fetches + converts, then runs phase 1 |
| #6 Stronger unsupervised baselines | `stronger_baselines.py` | IsolationForest / OneClassSVM vs the AE score, 3 datasets | ✅ ready |
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

## The third dataset — CIC-IoT2023
`--ciciot-train` / `--ciciot-test` are **optional** on `exp1_trivial_learners.py` and
`exp2_static_seeds.py`. Omit them and both scripts behave exactly as they did before the third
dataset existed, so the published NSL-KDD and UNSW-NB15 numbers are untouched. Supply them and a
third block is appended to the same loop.

```python
!python ciciot2023_pipeline.py --dataset-dir /content/drive/MyDrive/dataset --epochs 100
!python exp1_trivial_learners.py --epochs 40 \
    --ciciot-train .../CICIoT2023_Train.csv --ciciot-test .../CICIoT2023_Test.csv
!python exp2_static_seeds.py --epochs 40 --timesteps 80000 \
    --ciciot-train .../CICIoT2023_Train.csv --ciciot-test .../CICIoT2023_Test.csv
```
CIC-IoT2023 has 46 numeric flow features and no categoricals, so the phase-1 preprocessor
produces a 46-dim input with no one-hot expansion. There is no PPO reference row for it in
Table B — DQN/A2C/PPO reference F1s were only measured on NSL-KDD and UNSW-NB15.

## Stronger unsupervised baselines
```python
!python stronger_baselines.py --epochs 40 \
    --ciciot-train .../CICIoT2023_Train.csv --ciciot-test .../CICIoT2023_Test.csv
```
→ `results/stronger_baselines.csv`, one row per (dataset, method) for IsolationForest,
OneClassSVM(RBF) and the AE reconstruction error, all scored at the same F1-optimal cut.
**Disclosed asymmetry:** IF and OCSVM are fitted on at most 60,000 Normal rows (OCSVM does not
scale) while the AE uses the full Normal training set; the `n_train` column records what each
method actually saw.

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

## Table ↔ script map — third dataset
| Result | Script |
|--------|--------|
| Section V-I / Fig. 11 (CIC-IoT2023 detection) | `ciciot2023_pipeline.py --epochs 100` |
| Table XV (CIC-IoT2023 15-seed RL vs best simple learner) | `exp2_static_seeds.py --ciciot-train ... --ciciot-test ...` |
| Table B, CIC-IoT2023 rows | `exp1_trivial_learners.py --ciciot-train ... --ciciot-test ...` |
| IF / OCSVM comparison | `stronger_baselines.py` |

Phases 3–7 (drift, budget, cost, response, robustness) were run on NSL-KDD only; CIC-IoT2023
covers static detection (phase 1) plus Tables A and B.

## Deferred (future work)
- CICIDS2017 as a fourth dataset — needs the cleaned (WTMC-2021) release downloaded +
  preprocessed; listed as future work (days of cleaning for modest payoff).
