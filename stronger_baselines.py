"""
Stronger unsupervised baselines — Isolation Forest and One-Class SVM vs the autoencoder score
==============================================================================================
Reviewer question: "your simple learners are thresholds on YOUR OWN detector — how does the
autoencoder compare against established unsupervised anomaly detectors?" This runs
IsolationForest and OneClassSVM(RBF) on the same preprocessed features, on all three datasets,
against the AE reconstruction-error score.

Everything is reused from the project's own code, nothing is re-implemented:
  * loaders      phase1_autoencoder_ids.load_csv (NSL-KDD, CIC-IoT2023 CSVs)
                 phase8_unsw.load_unsw           (UNSW-NB15 parquet — load_csv cannot read parquet)
  * preprocessor phase1/phase8 build_preprocessor (fit on TRAIN only)
  * autoencoder  phase3_drift.train_ae           (deterministic, Normal-only, CPU, fixed seed)
  * threshold    phase8_unsw.best_threshold      (the F1-optimal cut — same rule for all methods,
                                                  so no method is handicapped by its offset)

*** DISCLOSED ASYMMETRY (also stated in the paper) ***
IsolationForest and OneClassSVM are fitted on a SUBSAMPLE of at most 60,000 Normal training rows,
because OneClassSVM is O(n^2)-ish and does not finish on 160k+ rows in reasonable time. The
autoencoder is fitted on the FULL Normal training set. This favours the autoencoder on the two
larger datasets and must be reported alongside the numbers, not quietly dropped: the `n_train`
column in the output CSV records exactly how many rows each method saw.

All three methods are unsupervised (Normal-only fitting); labels are used only for scoring and
for the shared F1-optimal threshold.

RUN:
  python stronger_baselines.py --epochs 40 \
      --nsl-train .../Train_data.csv --nsl-test .../Test_data.csv \
      --unsw-train .../UNSW_NB15_training-set.parquet --unsw-test .../UNSW_NB15_testing-set.parquet \
      --ciciot-train .../CICIoT2023_Train.csv --ciciot-test .../CICIoT2023_Test.csv

Outputs: results/stronger_baselines.csv — one row per (dataset, method), written incrementally
so an interrupted run still leaves usable rows.
"""
import argparse, csv, importlib, os, subprocess, sys, time
import numpy as np


def _ensure(mod, pip_name=None):
    try: importlib.import_module(mod)
    except Exception:
        print(f"installing {pip_name or mod} ..."); subprocess.run([sys.executable,"-m","pip","install","-q",pip_name or mod], check=True)


import phase1_autoencoder_ids as p1
import phase3_drift as p3
import phase8_unsw as p8
import exp_harness as H

FIELDS = ["dataset", "method", "family", "f1", "roc", "pr", "n_train", "secs"]
SUBSAMPLE = 60000          # cap on the Normal rows given to IsolationForest / OneClassSVM


def score_row(dataset, method, family, y_test, scores, n_train, secs):
    """F1 at the shared F1-optimal cut, plus threshold-free ROC-AUC / PR-AUC."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    best = p8.best_threshold(scores, y_test)
    return dict(dataset=dataset, method=method, family=family,
                f1=float(best["f1"]), roc=float(roc_auc_score(y_test, scores)),
                pr=float(average_precision_score(y_test, scores)),
                n_train=int(n_train), secs=round(float(secs), 1))


def run_dataset(name, load, preproc, epochs, seed, writer, fh):
    import torch
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    dev = torch.device("cpu")

    print(f"\n===== {name} =====")
    (trf, trl), (tef, tel) = load()
    pre, _, _ = preproc(trf)
    Xtr = pre.fit_transform(trf).astype(np.float32)
    Xte = pre.transform(tef).astype(np.float32)
    Xn = Xtr[trl == 0]                                    # Normal-only fitting set
    print(f"  train {Xtr.shape}  normal={len(Xn)}  test {Xte.shape}")

    # ---- 1. Autoencoder reconstruction error (this thesis) — FULL normal training set ----
    t0 = time.time()
    H.set_all_seeds(seed)
    ae = p3.train_ae(Xn, Xtr.shape[1], dev, epochs, seed)
    _, ete = p3.latent_err(ae, Xte, dev)
    row = score_row(name, "Autoencoder error + tuned threshold", "this thesis",
                    tel, ete, len(Xn), time.time() - t0)
    writer.writerow(row); fh.flush()
    print(f"  {row['method']:<38} F1 {row['f1']:.4f}  ROC {row['roc']:.4f}  PR {row['pr']:.4f}  ({row['secs']}s)")

    # ---- 2/3. Established unsupervised detectors — SUBSAMPLED normal fitting set ----
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(Xn))[:SUBSAMPLE]
    Xsub = Xn[idx]
    if len(Xsub) < len(Xn):
        print(f"  (IF / OCSVM fitted on a {len(Xsub)}-row subsample of {len(Xn)} normals — "
              f"OCSVM does not scale; disclosed in the paper)")

    for method, est in (("Isolation Forest", IsolationForest(random_state=seed, n_jobs=-1)),
                        ("One-Class SVM (RBF)", OneClassSVM(kernel="rbf"))):
        t0 = time.time()
        est.fit(Xsub)
        s = -est.decision_function(Xte)                   # higher = more anomalous, like e(x)
        row = score_row(name, method, "unsupervised baseline", tel, s, len(Xsub), time.time() - t0)
        writer.writerow(row); fh.flush()
        print(f"  {row['method']:<38} F1 {row['f1']:.4f}  ROC {row['roc']:.4f}  PR {row['pr']:.4f}  ({row['secs']}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsl-train", default="/content/drive/MyDrive/dataset/Train_data.csv")
    ap.add_argument("--nsl-test", default="/content/drive/MyDrive/dataset/Test_data.csv")
    ap.add_argument("--unsw-train", default="/content/drive/MyDrive/dataset/UNSW_NB15_training-set.parquet")
    ap.add_argument("--unsw-test", default="/content/drive/MyDrive/dataset/UNSW_NB15_testing-set.parquet")
    ap.add_argument("--ciciot-train", default="/content/drive/MyDrive/dataset/CICIoT2023_Train.csv")
    ap.add_argument("--ciciot-test", default="/content/drive/MyDrive/dataset/CICIoT2023_Test.csv")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args()
    _ensure("pyarrow")

    DS = [("NSL-KDD",   lambda: (p1.load_csv(a.nsl_train), p1.load_csv(a.nsl_test)), p1.build_preprocessor),
          ("UNSW-NB15", lambda: (p8.load_unsw(a.unsw_train), p8.load_unsw(a.unsw_test)), p8.build_preprocessor),
          ("CIC-IoT2023", lambda: (p1.load_csv(a.ciciot_train), p1.load_csv(a.ciciot_test)), p1.build_preprocessor)]

    os.makedirs(os.path.join(a.outdir, "results"), exist_ok=True)
    out = os.path.join(a.outdir, "results", "stronger_baselines.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); fh.flush()
        for name, load, pre in DS:
            if not all(os.path.exists(p) for p in _paths(a, name)):
                print(f"\n===== {name}: input files missing -> skipped =====")
                continue
            run_dataset(name, load, pre, a.epochs, a.seed, w, fh)
    print(f"\nSaved -> {out}")
    print("Reminder: n_train differs by method on purpose (see the header of this file).")


def _paths(a, name):
    return {"NSL-KDD": (a.nsl_train, a.nsl_test),
            "UNSW-NB15": (a.unsw_train, a.unsw_test),
            "CIC-IoT2023": (a.ciciot_train, a.ciciot_test)}[name]


if __name__ == "__main__":
    main()
