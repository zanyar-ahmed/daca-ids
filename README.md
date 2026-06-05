# Hybrid IDS — Reproducible Code (DACA-IDS)

Code for a self-supervised + reinforcement-learning intrusion detection system, evaluated on
NSL-KDD. **Every reported number is produced by running this code with a fixed seed** — nothing
is hand-typed.

## Phase 1 — Self-Supervised Autoencoder baseline
`phase1_autoencoder_ids.py` trains an autoencoder on *Normal* traffic only, then scores anomalies
by reconstruction error and reports real Accuracy / Precision / Recall / F1 / FPR + ROC-AUC.

### Run on Google Colab (GPU)
1. Runtime → Change runtime type → **GPU**.
2. In a cell:
   ```python
   !git clone https://github.com/zanyar-ahmed/daca-ids.git
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
3. In Colab, pull and re-run:
   ```python
   !git pull
   !python phase1_autoencoder_ids.py
   ```

## Seeds & reproducibility
Default seed = 42. Same seed + same data → same numbers. Outputs (`phase1_metrics.json`,
`phase1_recon_hist.png`) are regenerated each run and are not version-controlled.
