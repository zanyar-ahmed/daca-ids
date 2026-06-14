"""
Phase 8 — Second-dataset validation on UNSW-NB15
=================================================
Confirms the two headline findings generalise beyond NSL-KDD:
  (1) the self-supervised autoencoder detects attacks (F1 / ROC-AUC), and
  (2) the RL tier-controller does NOT beat a tuned threshold (as on NSL-KDD).

UNSW-NB15 (dhoogla parquet): 175,341 train / 82,332 test; 34 features + attack_cat + label
(1=attack, 0=normal); categoricals: proto, service, state. AE trains on Normal only.
Reproducible (fixed seed).

RUN (Colab):
  !python phase8_unsw.py --epochs 40 --timesteps 120000
Outputs: phase8_metrics.json
"""
import argparse, importlib, json, os, subprocess, sys
import numpy as np

def _ensure(mod, pip_name=None):
    try: importlib.import_module(mod)
    except Exception:
        print(f"installing {pip_name or mod} ..."); subprocess.run([sys.executable,"-m","pip","install","-q",pip_name or mod], check=True)

import phase1_autoencoder_ids as p1
import phase2_ppo_controller as p2
import phase3_drift as p3

CAT_COLS = ["proto", "service", "state"]
DROP_COLS = ["attack_cat", "label", "id"]    # non-feature columns


def load_unsw(path):
    import pandas as pd
    df = pd.read_parquet(path)
    df.columns = [str(c).strip() for c in df.columns]
    labels = df["label"].astype(int).values          # 1=attack, 0=normal
    feats = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    for c in feats.columns:                            # category -> str for the encoder
        if str(feats[c].dtype) in ("object", "category"):
            feats[c] = feats[c].astype(str)
    return feats, labels


def build_preprocessor(train_feats):
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    num = [c for c in train_feats.columns if pd.api.types.is_numeric_dtype(train_feats[c])]
    cat = [c for c in train_feats.columns if c not in num]
    try: ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError: ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer([("n", StandardScaler(), num), ("c", ohe, cat)], remainder="drop"), cat, num


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="/content/drive/MyDrive/dataset/UNSW_NB15_training-set.parquet")
    ap.add_argument("--test", default="/content/drive/MyDrive/dataset/UNSW_NB15_testing-set.parquet")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--timesteps", type=int, default=120000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args()
    _ensure("pyarrow"); _ensure("gymnasium"); _ensure("stable_baselines3", "stable-baselines3")
    p1.set_seed(a.seed)
    if a.train.startswith("/content/drive") and not os.path.exists(a.train):
        try:
            from google.colab import drive; drive.mount("/content/drive")
        except Exception: pass

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from sklearn.metrics import roc_auc_score
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"Device: {dev} | seed {a.seed}")

    # --- load + preprocess ---
    print("\n[1] Loading UNSW-NB15 ...")
    trf, trl = load_unsw(a.train); tef, tel = load_unsw(a.test)
    print(f"    train {trf.shape} (normal {int((trl==0).sum())}, attack {int((trl==1).sum())}) | test {tef.shape}")
    pre, cat, num = build_preprocessor(trf)
    Xtr = pre.fit_transform(trf).astype(np.float32); Xte = pre.transform(tef).astype(np.float32)
    dim = Xtr.shape[1]; print(f"    {len(num)} numeric + {len(cat)} categorical -> {dim}-dim input")

    # --- frozen autoencoder on Normal ---
    print("\n[2] Training autoencoder on Normal only ...")
    ae = p3.train_ae(Xtr[trl == 0], dim, dev, a.epochs, a.seed)

    # --- Phase-1-style detection ---
    print("\n[3] Detection (thresholds from Normal-train errors) ...")
    _, etr = p3.latent_err(ae, Xtr, dev)
    _, ete = p3.latent_err(ae, Xte, dev)
    normal_err = etr[trl == 0]
    pct = {f"P{p}": float(np.percentile(normal_err, p)) for p in (60, 85, 95, 99)}
    roc = roc_auc_score(tel, ete)
    print(f"    ROC-AUC = {roc:.4f}")
    print(f"    {'tier':>5}{'acc':>8}{'prec':>8}{'rec':>8}{'f1':>8}{'fpr':>8}")
    det = {}
    for t, thr in pct.items():
        m = p1.evaluate(ete, tel, thr); det[t] = m
        print(f"    {t:>5}{m['accuracy']:>8.3f}{m['precision']:>8.3f}{m['recall']:>8.3f}{m['f1']:>8.3f}{m['fpr']:>8.3f}")
    best_f1 = max(m["f1"] for m in det.values())

    # --- Phase-2-style RL test: does RL beat the threshold? ---
    print(f"\n[4] RL tier-controller vs threshold ({a.timesteps:,} steps) ...")
    Ztr, Etr = p3.latent_err(ae, Xtr, dev); Zte, Ete = p3.latent_err(ae, Xte, dev)
    zmu, zsd = Ztr.mean(0), Ztr.std(0) + 1e-8
    emu, esd = float(Etr.mean()), float(Etr.std()) + 1e-8
    Str = np.concatenate([np.clip((Ztr - zmu) / zsd, -10, 10),
                          np.clip(((Etr - emu) / esd)[:, None], -10, 10)], 1).astype(np.float32)
    Ste = np.concatenate([np.clip((Zte - zmu) / zsd, -10, 10),
                          np.clip(((Ete - emu) / esd)[:, None], -10, 10)], 1).astype(np.float32)
    TierEnv = p2.make_env_class()
    env = DummyVecEnv([lambda: TierEnv(Str, trl, 2048, a.seed)])
    model = PPO("MlpPolicy", env, seed=a.seed, verbose=0, n_steps=2048, batch_size=256,
                device="cpu", policy_kwargs=dict(net_arch=[128, 64, 32]))
    model.learn(total_timesteps=a.timesteps)
    actions = np.asarray(model.predict(Ste, deterministic=True)[0]).reshape(-1)
    rl_m = p2.metrics_from_pred(p2.tiers_to_binary(actions), tel)
    base85 = p1.evaluate(ete, tel, pct["P85"]); base95 = p1.evaluate(ete, tel, pct["P95"])

    print("\n==============  UNSW-NB15 RESULTS  ==============")
    print(f"AE detection ROC-AUC: {roc:.4f} | best threshold F1: {best_f1:.3f}")
    print(f"{'method':<24}{'acc':>8}{'prec':>8}{'rec':>8}{'f1':>8}")
    for n, m in [("Fixed threshold P85", base85), ("Fixed threshold P95", base95), ("RL tier-controller", rl_m)]:
        print(f"{n:<24}{m['accuracy']:>8.3f}{m['precision']:>8.3f}{m['recall']:>8.3f}{m['f1']:>8.3f}")
    bestfix = max(base85["f1"], base95["f1"])
    print(f"\nVerdict: {'RL beats threshold' if rl_m['f1'] > bestfix + 1e-3 else 'RL does NOT beat threshold (same finding as NSL-KDD) ✅'}"
          f"  (RL {rl_m['f1']:.3f} vs threshold {bestfix:.3f})")
    print("================================================")

    json.dump(dict(seed=a.seed, dataset="UNSW-NB15", roc_auc=roc, detection=det,
                   rl=rl_m, fixed_P85=base85, fixed_P95=base95), open(os.path.join(a.outdir, "phase8_metrics.json"), "w"), indent=2)
    print("Saved -> phase8_metrics.json")


if __name__ == "__main__":
    main()
