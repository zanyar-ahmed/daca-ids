"""
Phase 1 — Self-Supervised Autoencoder IDS  (DACA-IDS, step 1)
=============================================================
Goal: produce the FIRST genuine, reproducible detection metrics for the thesis.

What this does (matches Section V-B of the v3 paper):
  1. Load NSL-KDD, one-hot encode the categorical features  -> ~118-dim input
  2. Train a self-supervised autoencoder on NORMAL traffic only (MSE loss)
  3. Reconstruction error  e(x) = ||x - x_hat||^2   is the anomaly score
  4. Percentile thresholds (P60/P85/P95/P99) computed from the NORMAL-TRAIN errors
  5. Evaluate against the real labels: Accuracy / Precision / Recall / F1 / FPR
     + threshold-free ROC-AUC and PR-AUC, + confusion matrix

Principle: every number printed here comes from THIS code running on the real data,
with a fixed seed, so it can be re-run live. No hand-typed results.

HOW TO RUN
----------
Google Colab (recommended, uses GPU + your Drive CSVs):
    !python "phase1_autoencoder_ids.py"
  (defaults read /content/drive/MyDrive/dataset/Train_data.csv and Test_data.csv)

Custom paths / fewer epochs:
    !python phase1_autoencoder_ids.py --train /path/Train_data.csv --test /path/Test_data.csv --epochs 100

Outputs (written next to the script):
    phase1_metrics.json        all metrics, dims, seed, config
    phase1_recon_hist.png      reconstruction-error histograms (Normal vs Attack)
"""

import argparse
import json
import os
import random

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


# ----------------------------------------------------------------------------
# Data loading + labels
# ----------------------------------------------------------------------------
# Columns that are labels / metadata, never features.
NON_FEATURE_COLS = {"class", "label", "labels", "attack", "attack_cat",
                    "difficulty", "level", "score", "outcome"}


def _find_label_col(df: pd.DataFrame):
    """Return the name of the label column, or None if the file is unlabelled."""
    for c in df.columns:
        if str(c).strip().lower() in {"class", "label", "labels", "attack"}:
            return c
    # Fallback: a last column that is non-numeric with few unique values
    last = df.columns[-1]
    if df[last].dtype == object and df[last].nunique() < 50:
        return last
    return None


def _to_binary_labels(series: pd.Series) -> np.ndarray:
    """normal -> 0, anything else -> 1 (attack)."""
    return (series.astype(str).str.strip().str.lower() != "normal").astype(int).values


def load_csv(path: str):
    """Load a CSV, return (features_df, labels or None)."""
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    label_col = _find_label_col(df)
    labels = _to_binary_labels(df[label_col]) if label_col is not None else None
    drop = [c for c in df.columns if str(c).strip().lower() in NON_FEATURE_COLS]
    features = df.drop(columns=drop, errors="ignore")
    return features, labels


# ----------------------------------------------------------------------------
# Preprocessing: one-hot categoricals + standardise numerics (fit on TRAIN only)
# ----------------------------------------------------------------------------
def build_preprocessor(train_features: pd.DataFrame):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    cat_cols = [c for c in train_features.columns if train_features[c].dtype == object]
    num_cols = [c for c in train_features.columns if c not in cat_cols]

    try:                              # sklearn >= 1.2
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:                 # older sklearn
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    pre = ColumnTransformer(
        [("num", StandardScaler(), num_cols),
         ("cat", ohe, cat_cols)],
        remainder="drop",
    )
    return pre, cat_cols, num_cols


# ----------------------------------------------------------------------------
# Autoencoder (encoder 118->64->32->16, decoder 16->32->64->118; LayerNorm+ELU,
# residual projections, dropout). Section V-B-1.
# ----------------------------------------------------------------------------
def build_model(input_dim: int, dropout: float = 0.2):
    import torch.nn as nn

    class ResidualBlock(nn.Module):
        def __init__(self, in_dim, out_dim, p):
            super().__init__()
            self.lin = nn.Linear(in_dim, out_dim)
            self.norm = nn.LayerNorm(out_dim)
            self.act = nn.ELU()
            self.drop = nn.Dropout(p)
            self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

        def forward(self, x):
            h = self.drop(self.act(self.norm(self.lin(x))))
            return h + self.proj(x)

    class AE(nn.Module):
        def __init__(self, d, p):
            super().__init__()
            self.encoder = nn.Sequential(
                ResidualBlock(d, 64, p), ResidualBlock(64, 32, p), ResidualBlock(32, 16, p))
            self.decoder = nn.Sequential(
                ResidualBlock(16, 32, p), ResidualBlock(32, 64, p))
            self.out = nn.Linear(64, d)        # plain linear reconstruction

        def forward(self, x):
            z = self.encoder(x)
            return self.out(self.decoder(z)), z

    return AE(input_dim, dropout)


def recon_error(model, X, device, batch=4096):
    """Per-sample reconstruction error ||x - x_hat||^2 (mean over features)."""
    import torch
    model.eval()
    errs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.tensor(X[i:i + batch], dtype=torch.float32, device=device)
            xh, _ = model(xb)
            errs.append(((xb - xh) ** 2).mean(dim=1).cpu().numpy())
    return np.concatenate(errs)


# ----------------------------------------------------------------------------
# Metrics (against real labels)
# ----------------------------------------------------------------------------
def evaluate(scores, labels, threshold):
    from sklearn.metrics import confusion_matrix
    pred = (scores > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    fpr = fp / max(fp + tn, 1)
    return dict(threshold=float(threshold), accuracy=acc, precision=prec, recall=rec,
                f1=f1, fpr=fpr, tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="/content/drive/MyDrive/dataset/Train_data.csv")
    ap.add_argument("--test", default="/content/drive/MyDrive/dataset/Test_data.csv")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    set_seed(args.seed)

    # Optional: mount Google Drive if running in Colab and paths are under /content/drive
    if args.train.startswith("/content/drive") and not os.path.exists(args.train):
        try:
            from google.colab import drive
            drive.mount("/content/drive")
        except Exception:
            pass

    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score, average_precision_score
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | seed: {args.seed}")

    # ---- 1. Load ----
    print("\n[1] Loading data ...")
    train_feats, train_labels = load_csv(args.train)
    if train_labels is None:
        raise SystemExit("Training file has no label column — cannot proceed.")
    test_feats, test_labels = load_csv(args.test)
    print(f"    Train: {train_feats.shape}  Normal={int((train_labels==0).sum())} "
          f"Attack={int((train_labels==1).sum())}")
    print(f"    Test : {test_feats.shape}  labelled={test_labels is not None}")

    # ---- 2. Preprocess (fit on TRAIN features only) ----
    print("\n[2] Preprocessing (one-hot + standardise) ...")
    pre, cat_cols, num_cols = build_preprocessor(train_feats)
    Xtr_all = pre.fit_transform(train_feats).astype(np.float32)
    input_dim = Xtr_all.shape[1]
    print(f"    {len(num_cols)} numeric + {len(cat_cols)} categorical -> {input_dim}-dim input")

    # AE trains on NORMAL rows only
    Xtr_normal = Xtr_all[train_labels == 0]
    # 90/10 split of normal for early-stopping on validation reconstruction loss
    idx = np.random.permutation(len(Xtr_normal))
    n_val = max(1, int(0.1 * len(idx)))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    Xtr, Xval = Xtr_normal[tr_idx], Xtr_normal[val_idx]
    print(f"    Normal-only AE training set: {Xtr.shape}  (val {Xval.shape})")

    # ---- 3. Train autoencoder ----
    print(f"\n[3] Training autoencoder ({args.epochs} epochs) ...")
    model = build_model(input_dim, args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.MSELoss()

    Xtr_t = torch.tensor(Xtr, device=device)
    Xval_t = torch.tensor(Xval, device=device)
    best_val, best_state = float("inf"), None
    n = len(Xtr_t)
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, args.batch):
            b = perm[i:i + args.batch]
            xb = Xtr_t[b]
            opt.zero_grad()
            xh, _ = model(xb)
            loss = loss_fn(xh, xb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        sched.step()
        model.eval()
        with torch.no_grad():
            vh, _ = model(Xval_t)
            vloss = loss_fn(vh, Xval_t).item()
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep == 1 or ep % 10 == 0 or ep == args.epochs:
            print(f"    epoch {ep:3d}/{args.epochs}  train {tot/n:.5f}  val {vloss:.5f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"    best val reconstruction MSE: {best_val:.5f}")

    # ---- 4. Thresholds from NORMAL-TRAIN reconstruction errors ----
    print("\n[4] Percentile thresholds from Normal-train errors ...")
    normal_err = recon_error(model, Xtr_normal, device)
    pct = {f"P{p}": float(np.percentile(normal_err, p)) for p in (60, 85, 95, 99)}
    for k, v in pct.items():
        print(f"    {k} = {v:.6f}")

    # ---- 5. Evaluate against real labels ----
    def eval_split(name, Xfeat, y):
        if y is None:
            print(f"\n[5] {name}: no labels -> skipped (cannot score).")
            return None
        X = pre.transform(Xfeat).astype(np.float32)
        scores = recon_error(model, X, device)
        roc = roc_auc_score(y, scores)
        prauc = average_precision_score(y, scores)
        print(f"\n[5] {name}  (n={len(y)})   ROC-AUC={roc:.4f}  PR-AUC={prauc:.4f}")
        print(f"    {'tier':>5} {'thr':>9} {'acc':>7} {'prec':>7} {'rec':>7} {'f1':>7} {'fpr':>7}")
        rows = {}
        for tier, thr in pct.items():
            m = evaluate(scores, y, thr)
            rows[tier] = m
            print(f"    {tier:>5} {thr:9.4f} {m['accuracy']:7.4f} {m['precision']:7.4f} "
                  f"{m['recall']:7.4f} {m['f1']:7.4f} {m['fpr']:7.4f}")
        return dict(roc_auc=roc, pr_auc=prauc, by_tier=rows)

    # Always evaluate on a held-out LABELLED slice of training (guaranteed labels),
    # plus the official test set if it carries labels.
    from sklearn.model_selection import train_test_split
    _, Xho, _, yho = train_test_split(train_feats, train_labels,
                                      test_size=0.2, random_state=args.seed,
                                      stratify=train_labels)
    results = {}
    results["holdout_from_train"] = eval_split("Hold-out from TRAIN (labelled)", Xho, yho)
    results["official_test"] = eval_split("Official TEST set", test_feats, test_labels)

    # ---- 6. Save artifacts ----
    out_json = os.path.join(args.outdir, "phase1_metrics.json")
    with open(out_json, "w") as f:
        json.dump(dict(seed=args.seed, device=str(device), input_dim=input_dim,
                       epochs=args.epochs, best_val_mse=best_val,
                       thresholds=pct, results=results), f, indent=2)
    print(f"\nSaved metrics -> {out_json}")

    # Reconstruction-error histogram (Normal vs Attack) on the labelled hold-out
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        Xho_p = pre.transform(Xho).astype(np.float32)
        e = recon_error(model, Xho_p, device)
        plt.figure(figsize=(8, 5))
        bins = np.linspace(0, np.percentile(e, 99), 80)
        plt.hist(e[yho == 0], bins=bins, alpha=0.6, label="Normal", density=True)
        plt.hist(e[yho == 1], bins=bins, alpha=0.6, label="Attack", density=True)
        for k, v in pct.items():
            plt.axvline(v, ls="--", lw=1, label=k)
        plt.xlabel("Reconstruction error"); plt.ylabel("Density")
        plt.title("Autoencoder reconstruction error — Normal vs Attack")
        plt.legend(); plt.tight_layout()
        png = os.path.join(args.outdir, "phase1_recon_hist.png")
        plt.savefig(png, dpi=130)
        print(f"Saved plot    -> {png}")
    except Exception as ex:
        print(f"(plot skipped: {ex})")

    print("\nDone. These numbers are produced by this script and re-runnable with the same seed.")


if __name__ == "__main__":
    main()
