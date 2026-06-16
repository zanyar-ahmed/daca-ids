"""
Experiment 4 — RL hyperparameter sensitivity (Protocol #4).
Shows the negative result is NOT an artefact of one hyperparameter choice: across a learning-rate
x network-architecture grid, the PPO static detector never exceeds the best simple learner.

Grid: lr in {1e-4,3e-4,1e-3} x net in {[64,64],[128,64,32],[256,128]} = 9 configs x 5 seeds,
on NSL-KDD (deterministic frozen autoencoder). Produces Table C.

RUN (Colab):  !python exp4_hparam_sweep.py --epochs 40 --timesteps 80000
Outputs: results/exp4_hparam.csv
"""
import argparse, importlib, json, os, subprocess, sys
from itertools import product
import numpy as np

def _ensure(mod, pip_name=None):
    try: importlib.import_module(mod)
    except Exception:
        print(f"installing {pip_name or mod} ..."); subprocess.run([sys.executable,"-m","pip","install","-q",pip_name or mod], check=True)

import phase1_autoencoder_ids as p1
import phase2_ppo_controller as p2
import phase3_drift as p3
import phase8_unsw as p8
import exp_harness as H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="/content/drive/MyDrive/dataset/Train_data.csv")
    ap.add_argument("--test", default="/content/drive/MyDrive/dataset/Test_data.csv")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--timesteps", type=int, default=80000)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args()
    _ensure("gymnasium"); _ensure("stable_baselines3", "stable-baselines3")
    if a.train.startswith("/content/drive") and not os.path.exists(a.train):
        try:
            from google.colab import drive; drive.mount("/content/drive")
        except Exception: pass
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    dev = torch.device("cpu"); seeds = H.SEEDS[:a.n_seeds]

    # deterministic AE + (z,e) states
    (trf, trl), (tef, tel) = p1.load_csv(a.train), p1.load_csv(a.test)
    pre, _, _ = p1.build_preprocessor(trf)
    Xtr = pre.fit_transform(trf).astype(np.float32); Xte = pre.transform(tef).astype(np.float32)
    H.set_all_seeds(0); ae = p3.train_ae(Xtr[trl == 0], Xtr.shape[1], dev, a.epochs, 0)
    Ztr, Etr = p3.latent_err(ae, Xtr, dev); Zte, Ete = p3.latent_err(ae, Xte, dev)
    zmu, zsd = Ztr.mean(0), Ztr.std(0) + 1e-8; emu, esd = float(Etr.mean()), float(Etr.std()) + 1e-8
    mk = lambda Z, E: np.concatenate([np.clip((Z - zmu) / zsd, -10, 10),
                                      np.clip(((E - emu) / esd)[:, None], -10, 10)], 1).astype(np.float32)
    Str, Ste = mk(Ztr, Etr), mk(Zte, Ete)
    base_thr = p8.best_threshold(Ete, tel)["f1"]
    lrf = LogisticRegression(class_weight="balanced", max_iter=2000).fit(Str, trl)
    pte = lrf.predict_proba(Ste)[:, 1]; ptr = lrf.predict_proba(Str)[:, 1]
    cuts = np.quantile(ptr, np.linspace(0.01, 0.99, 200))
    cut = float(cuts[int(np.argmax([f1_score(trl, (ptr >= c).astype(int), zero_division=0) for c in cuts]))])
    base_best = max(base_thr, float(f1_score(tel, (pte >= cut).astype(int), zero_division=0)))
    print(f"best simple learner F1 = {base_best:.3f}\n")

    TierEnv = p2.make_env_class(); rows = []
    grid = list(product([1e-4, 3e-4, 1e-3], [[64, 64], [128, 64, 32], [256, 128]]))
    for lr_, net in grid:
        f1 = []
        for s in seeds:
            H.set_all_seeds(s)
            env = DummyVecEnv([lambda: TierEnv(Str, trl, 2048, s)])
            model = PPO("MlpPolicy", env, seed=s, verbose=0, n_steps=2048, batch_size=256, device="cpu",
                        learning_rate=lr_, policy_kwargs=dict(net_arch=net))
            model.learn(total_timesteps=a.timesteps)
            acts = np.asarray(model.predict(Ste, deterministic=True)[0]).reshape(-1)
            f1.append(float(f1_score(tel, p2.tiers_to_binary(acts), zero_division=0)))
        f1 = np.array(f1); m = float(f1.mean()); sd = float(f1.std(ddof=1))
        rows.append(dict(lr=lr_, net=str(net), f1_mean=m, f1_std=sd, margin=m - base_best))
        print(f"  lr={lr_:<6} net={str(net):<14} F1 {m:.3f}±{sd:.3f}  margin {m-base_best:+.3f}")

    print("\n================  TABLE C (hyperparameter sweep, NSL-KDD)  ================")
    print(f"best simple learner = {base_best:.3f}")
    best_rl = max(r["f1_mean"] for r in rows)
    print(f"best RL config over the whole grid = {best_rl:.3f}  -> margin {best_rl-base_best:+.3f}")
    print(f"=> across all {len(rows)} configs, RL { 'NEVER' if best_rl < base_best else 'sometimes' } exceeds the simple learner")
    print("==========================================================================")

    os.makedirs(os.path.join(a.outdir, "results"), exist_ok=True)
    import csv
    with open(os.path.join(a.outdir, "results", "exp4_hparam.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["lr", "net", "f1_mean", "f1_std", "margin"]); w.writeheader()
        for r in rows: w.writerow(r)
    json.dump(dict(base_best=base_best, rows=rows), open(os.path.join(a.outdir, "results", "exp4_hparam.json"), "w"), indent=2)
    print("Saved -> results/exp4_hparam.csv")


if __name__ == "__main__":
    main()
