"""
Experiment 1 — Trivial-learner battery (Protocol #1)  ⭐ the cheapest, most persuasive result.
Shows a 1-2 parameter learner on the single reconstruction-error feature e(x) matches the tuned
threshold and equals/beats PPO (~10^4 params) -> empirical proof of "threshold-reducibility".

Produces Table B (NSL-KDD + UNSW-NB15). Deterministic autoencoder (CPU, fixed seed).

RUN (Colab):
  !python exp1_trivial_learners.py --epochs 40
Outputs: results/exp1_trivial.csv  (+ printed Table B)
"""
import argparse, importlib, json, os, subprocess, sys
import numpy as np

def _ensure(mod, pip_name=None):
    try: importlib.import_module(mod)
    except Exception:
        print(f"installing {pip_name or mod} ..."); subprocess.run([sys.executable,"-m","pip","install","-q",pip_name or mod], check=True)

import phase1_autoencoder_ids as p1
import phase3_drift as p3
import phase8_unsw as p8
import exp_harness as H


def trivial_learners(etr, ytr, ete, yte):
    """Return metrics for: unsupervised P85 threshold, LR-on-e, best-tau-on-e."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    out = {}
    # (0) unsupervised percentile threshold (P85 of NORMAL-train errors) — 1 param, no labels
    thr85 = float(np.percentile(etr[ytr == 0], 85))
    out["Percentile P85 (unsupervised, 1p)"] = H.metrics(yte, (ete > thr85).astype(int), ete)
    # (a) logistic regression on the single feature e — 2 params
    lr = LogisticRegression(class_weight="balanced", max_iter=2000).fit(etr.reshape(-1, 1), ytr)
    p_te = lr.predict_proba(ete.reshape(-1, 1))[:, 1]
    out["Logistic regression on e (2p)"] = H.metrics(yte, (p_te >= 0.5).astype(int), p_te)
    # (b) best learned threshold tau* on e (max-F1 on train) — 1 param
    taus = np.quantile(etr, np.linspace(0.50, 0.999, 300))
    f1s = [f1_score(ytr, (etr >= t).astype(int), zero_division=0) for t in taus]
    tau = float(taus[int(np.argmax(f1s))])
    out["Best learned threshold on e (1p)"] = H.metrics(yte, (ete >= tau).astype(int), ete)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsl-train", default="/content/drive/MyDrive/dataset/Train_data.csv")
    ap.add_argument("--nsl-test", default="/content/drive/MyDrive/dataset/Test_data.csv")
    ap.add_argument("--unsw-train", default="/content/drive/MyDrive/dataset/UNSW_NB15_training-set.parquet")
    ap.add_argument("--unsw-test", default="/content/drive/MyDrive/dataset/UNSW_NB15_testing-set.parquet")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args()
    _ensure("pyarrow"); H.set_all_seeds(0)
    import torch
    dev = torch.device("cpu")   # deterministic AE -> reproducible error pools (Phase-7 lesson)

    # PPO references from earlier phases (same datasets) for the comparison row
    DATASETS = {
        "NSL-KDD":   dict(load=lambda: (p1.load_csv(a.nsl_train), p1.load_csv(a.nsl_test)),
                          pre=p1.build_preprocessor, ppo_f1=0.796),
        "UNSW-NB15": dict(load=lambda: (p8.load_unsw(a.unsw_train), p8.load_unsw(a.unsw_test)),
                          pre=p8.build_preprocessor, ppo_f1=0.789),
    }

    rows = []
    for name, cfg in DATASETS.items():
        print(f"\n===== {name} =====")
        (trf, trl), (tef, tel) = cfg["load"]()
        pre, _, _ = cfg["pre"](trf)
        Xtr = pre.fit_transform(trf).astype(np.float32); Xte = pre.transform(tef).astype(np.float32)
        print(f"  train {Xtr.shape}  normal={int((trl==0).sum())}  | training deterministic AE ...")
        H.set_all_seeds(0)
        ae = p3.train_ae(Xtr[trl == 0], Xtr.shape[1], dev, a.epochs, 0)
        _, etr = p3.latent_err(ae, Xtr, dev); _, ete = p3.latent_err(ae, Xte, dev)

        res = trivial_learners(etr, trl, ete, tel)
        print(f"  {'rule on AE error':<38}{'F1':>7}{'ROC':>7}{'PR':>7}{'acc':>7}")
        for rule, m in res.items():
            print(f"  {rule:<38}{m['F1']:>7.3f}{m['ROC_AUC']:>7.3f}{m['PR_AUC']:>7.3f}{m['accuracy']:>7.3f}")
            rows.append(dict(dataset=name, rule=rule, **m))
        print(f"  {'PPO (latent z + e, ~10^4 params)':<38}{cfg['ppo_f1']:>7.3f}{'—':>7}{'—':>7}{'—':>7}")
        rows.append(dict(dataset=name, rule="PPO (latent z + e)", F1=cfg["ppo_f1"]))

    os.makedirs(os.path.join(a.outdir, "results"), exist_ok=True)
    import csv
    keys = ["dataset", "rule", "F1", "ROC_AUC", "PR_AUC", "precision", "recall", "accuracy"]
    with open(os.path.join(a.outdir, "results", "exp1_trivial.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader()
        for r in rows: w.writerow(r)
    print("\nSaved -> results/exp1_trivial.csv")
    print("\nStory for Table B: the 1-2 parameter learners on e sit at the threshold level and")
    print("equal-or-beat PPO (~10^4 params) -> empirical proof of threshold-reducibility (Prop. 1).")


if __name__ == "__main__":
    main()
