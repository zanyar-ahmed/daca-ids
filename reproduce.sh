#!/usr/bin/env bash
# Regenerate every table/figure from raw scripts with the fixed seeds (Protocol #8).
# Datasets (NSL-KDD CSVs + UNSW-NB15 parquet) must live in $DATASET_DIR.
set -euo pipefail
DATA="${DATASET_DIR:-/content/drive/MyDrive/dataset}"
NSL_TR="$DATA/Train_data.csv"; NSL_TE="$DATA/Test_data.csv"
UN_TR="$DATA/UNSW_NB15_training-set.parquet"; UN_TE="$DATA/UNSW_NB15_testing-set.parquet"
mkdir -p results

echo "== Phase 1: autoencoder detection baseline (NSL-KDD) =="
python phase1_autoencoder_ids.py --train "$NSL_TR" --test "$NSL_TE"

echo "== Table B: trivial-learner battery (NSL + UNSW) =="
python exp1_trivial_learners.py --nsl-train "$NSL_TR" --nsl-test "$NSL_TE" \
       --unsw-train "$UN_TR" --unsw-test "$UN_TE" --epochs 40

echo "== Table A: per-seed RL vs best simple learner (15 seeds, NSL + UNSW) =="
python exp2_static_seeds.py --nsl-train "$NSL_TR" --nsl-test "$NSL_TE" \
       --unsw-train "$UN_TR" --unsw-test "$UN_TE" --epochs 40 --timesteps 80000

echo "== Table C: RL hyperparameter sweep (NSL-KDD) =="
python exp4_hparam_sweep.py --train "$NSL_TR" --test "$NSL_TE" --epochs 40 --timesteps 80000

echo "== Table D: adaptive adversary + cost-ratio (response game) =="
python exp5_adaptive_adversary.py --train "$NSL_TR" --ae-epochs 40 --seeds 8 --timesteps 100000

echo "== UNSW-NB15 second-dataset detection + RL =="
python phase8_unsw.py --train "$UN_TR" --test "$UN_TE" --epochs 40 --timesteps 120000

echo "All experiments done. Results in ./results/ ; figures saved next to each script."
