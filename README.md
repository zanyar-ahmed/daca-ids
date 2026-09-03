# Hybrid IDS — Reproducible Code (DACA-IDS)

Code for a self-supervised + reinforcement-learning intrusion detection system, evaluated on
three benchmarks — **NSL-KDD**, **UNSW-NB15** and **CIC-IoT2023**. **Every reported number is
produced by running this code with a fixed seed** — nothing is hand-typed.

Licensed under the MIT License (see `LICENSE`).

## One-command reproduction
```bash
pip install -r requirements.txt
DATASET_DIR=/path/to/datasets bash reproduce.sh
```
`reproduce.sh` regenerates every table and figure. NSL-KDD and UNSW-NB15 must be placed in
`$DATASET_DIR` (public benchmarks); CIC-IoT2023 is fetched and converted automatically by
`ciciot2023_pipeline.py`. `Dockerfile` builds the same environment.

> `requirements.txt` pins `pandas<3` deliberately: pandas 3.x reads the NSL-KDD categorical
> columns as `StringDtype` rather than `object`, so the preprocessor's dtype check misses them
> and the scaler fails on values like `'tcp'`.

## Phase 1 — Self-Supervised Autoencoder baseline
`phase1_autoencoder_ids.py` trains an autoencoder on *Normal* traffic only, then scores anomalies
by reconstruction error and reports real Accuracy / Precision / Recall / F1 / FPR + ROC-AUC.

## Datasets
| Dataset | Files in `$DATASET_DIR` | How to obtain |
|---|---|---|
| NSL-KDD | `Train_data.csv`, `Test_data.csv` | public benchmark |
| UNSW-NB15 | `UNSW_NB15_training-set.parquet`, `UNSW_NB15_testing-set.parquet` | public benchmark |
| CIC-IoT2023 | `CICIoT2023_Train.csv`, `CICIoT2023_Test.csv` | **generated** by `ciciot2023_pipeline.py` |

`ciciot2023_pipeline.py` downloads the canonical 46-flow-feature CIC-IoT2023 subsample
(HuggingFace `lacg030175/CIC-IoT-2023-neto-subsample`, config `random`), asserts the exact
snapshot the paper was run on (train 1,143,802 / test 285,951 rows), converts it into the CSV
shape the existing phase-1 loader already reads, then runs `phase1_autoencoder_ids.py`
**unmodified** on it. Datasets themselves are never committed to this repo.

## Third dataset + extra baselines
| Script | What it does |
|---|---|
| `ciciot2023_pipeline.py` | fetch + convert CIC-IoT2023, then run phase 1 on it (Section V-I, Table XV, Fig. 11) |
| `exp1_trivial_learners.py --ciciot-train/--ciciot-test` | adds the CIC-IoT2023 rows to Table B (optional flags; omit them and behaviour is unchanged) |
| `exp2_static_seeds.py --ciciot-train/--ciciot-test` | adds the CIC-IoT2023 rows to Table A (same) |
| `stronger_baselines.py` | IsolationForest / OneClassSVM(RBF) vs the AE score on all three datasets |

### Run on Google Colab (GPU) — private repo
1. Create a GitHub token: GitHub → Settings → Developer settings → **Fine-grained tokens** →
   give it *read-only* access to the `daca-ids` repo.
2. In Colab, click the **🔑 (Secrets)** icon in the left sidebar and add a secret named
   `GH_TOKEN` with the token value (toggle "Notebook access" on).
3. Runtime → Change runtime type → **GPU**, then in a cell:
   ```python
   from google.colab import userdata
   tok = userdata.get('GH_TOKEN')
   !git clone https://{tok}@github.com/zanyar-ahmed/daca-ids.git
   %cd daca-ids
   !python phase1_autoencoder_ids.py --epochs 30      # quick test
   # full run:
   # !python phase1_autoencoder_ids.py
   ```
Data is read from `/content/drive/MyDrive/dataset/Train_data.csv` and `Test_data.csv`
(the script auto-mounts Google Drive).

## Edit locally → run on Colab (the workflow)
1. Edit the code in VSCode on your Mac.
2. Push the change:
   ```bash
   git add -A && git commit -m "describe change" && git push
   ```
3. In Colab, pull and re-run (the token is already saved in the cloned repo's remote):
   ```python
   !git pull
   !python phase1_autoencoder_ids.py
   ```

## Seeds & reproducibility
Default seed = 42 (the experiment scripts use the fixed 15-seed list in `exp_harness.SEEDS`).
Same seed + same data → same numbers. Outputs (`phase1_metrics.json`, `phase1_recon_hist.png`,
everything under `results/`) are regenerated each run and are not version-controlled.
