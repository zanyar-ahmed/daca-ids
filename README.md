# Hybrid IDS — Reproducible Code (DACA-IDS)

Code for a self-supervised + reinforcement-learning intrusion detection system, evaluated on
NSL-KDD. **Every reported number is produced by running this code with a fixed seed** — nothing
is hand-typed.

## Phase 1 — Self-Supervised Autoencoder baseline
`phase1_autoencoder_ids.py` trains an autoencoder on *Normal* traffic only, then scores anomalies
by reconstruction error and reports real Accuracy / Precision / Recall / F1 / FPR + ROC-AUC.

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
Default seed = 42. Same seed + same data → same numbers. Outputs (`phase1_metrics.json`,
`phase1_recon_hist.png`) are regenerated each run and are not version-controlled.
