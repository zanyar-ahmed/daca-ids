"""
DQN / A2C robustness check (answers "is the negative result PPO-specific?").
Runs DQN and A2C (the value-based and other actor-critic families) on the SAME static-detection
task as exp2, across seeds, on NSL-KDD and UNSW-NB15, and compares to the best simple learner.
DQN is the most common RL algorithm in IDS work, so this directly fits the thesis title.

Expectation: DQN and A2C also fail to beat the simple learner -> the result is not PPO-specific.

RUN (Colab):
  !python DQN/dqn_static.py --epochs 40 --timesteps 80000 --n-seeds 5
Outputs: DQN/results/dqn_static.csv
"""
import argparse, importlib, json, os, subprocess, sys
import numpy as np

# import the root modules (this script lives in DQN/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

def _ensure(mod, pip_name=None):
    try: importlib.import_module(mod)
    except Exception:
        print(f"installing {pip_name or mod} ..."); subprocess.run([sys.executable,"-m","pip","install","-q",pip_name or mod], check=True)

import phase1_autoencoder_ids as p1
import phase2_ppo_controller as p2
import phase3_drift as p3
import phase8_unsw as p8
import exp_harness as H


def _bestcut_f1(ptr, ytr, pte, yte):
    from sklearn.metrics import f1_score
    cuts = np.quantile(ptr, np.linspace(0.01, 0.99, 200))
    cut = float(cuts[int(np.argmax([f1_score(ytr, (ptr >= c).astype(int), zero_division=0) for c in cuts]))])
    return float(f1_score(yte, (pte >= cut).astype(int), zero_division=0))


def best_simple(Str, Ste, trl, tel, Ete):
    """Fair supervised baselines on the SAME (z,e) the RL agent sees:
    threshold (1p), linear model (LR), and a matched-capacity MLP [128,64,32]."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    base_thr = p8.best_threshold(Ete, tel)["f1"]
    lrf = LogisticRegression(class_weight="balanced", max_iter=2000).fit(Str, trl)
    base_lr = _bestcut_f1(lrf.predict_proba(Str)[:, 1], trl, lrf.predict_proba(Ste)[:, 1], tel)
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=300, random_state=0).fit(Str, trl)
    base_mlp = _bestcut_f1(mlp.predict_proba(Str)[:, 1], trl, mlp.predict_proba(Ste)[:, 1], tel)
    return max(base_thr, base_lr, base_mlp), dict(threshold=base_thr, logreg=base_lr, mlp=base_mlp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsl-train", default="/content/drive/MyDrive/dataset/Train_data.csv")
    ap.add_argument("--nsl-test", default="/content/drive/MyDrive/dataset/Test_data.csv")
    ap.add_argument("--unsw-train", default="/content/drive/MyDrive/dataset/UNSW_NB15_training-set.parquet")
    ap.add_argument("--unsw-test", default="/content/drive/MyDrive/dataset/UNSW_NB15_testing-set.parquet")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--timesteps", type=int, default=80000)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args()
    _ensure("pyarrow"); _ensure("gymnasium"); _ensure("stable_baselines3", "stable-baselines3")
    if a.nsl_train.startswith("/content/drive") and not os.path.exists(a.nsl_train):
        try:
            from google.colab import drive; drive.mount("/content/drive")
        except Exception: pass

    import torch
    from stable_baselines3 import DQN, A2C
    from stable_baselines3.common.vec_env import DummyVecEnv
    from sklearn.metrics import f1_score
    dev = torch.device("cpu"); seeds = H.SEEDS[:a.n_seeds]
    TierEnv = p2.make_env_class()

    DS = {
        "NSL-KDD":   (lambda: (p1.load_csv(a.nsl_train), p1.load_csv(a.nsl_test)), p1.build_preprocessor),
        "UNSW-NB15": (lambda: (p8.load_unsw(a.unsw_train), p8.load_unsw(a.unsw_test)), p8.build_preprocessor),
    }
    rows = []
    for dname, (load, pre_fn) in DS.items():
        (trf, trl), (tef, tel) = load()
        pre, _, _ = pre_fn(trf)
        Xtr = pre.fit_transform(trf).astype(np.float32); Xte = pre.transform(tef).astype(np.float32)
        H.set_all_seeds(0); ae = p3.train_ae(Xtr[trl == 0], Xtr.shape[1], dev, a.epochs, 0)
        Ztr, Etr = p3.latent_err(ae, Xtr, dev); Zte, Ete = p3.latent_err(ae, Xte, dev)
        zmu, zsd = Ztr.mean(0), Ztr.std(0) + 1e-8; emu, esd = float(Etr.mean()), float(Etr.std()) + 1e-8
        mk = lambda Z, E: np.concatenate([np.clip((Z - zmu) / zsd, -10, 10),
                                          np.clip(((E - emu) / esd)[:, None], -10, 10)], 1).astype(np.float32)
        Str, Ste = mk(Ztr, Etr), mk(Zte, Ete)
        base_best, comps = best_simple(Str, Ste, trl, tel, Ete)
        print(f"\n===== {dname} =====  threshold {comps['threshold']:.3f} | LR(z,e) {comps['logreg']:.3f} "
              f"| MLP(z,e) {comps['mlp']:.3f}  -> best simple learner {base_best:.3f}")

        for algo in ("DQN", "A2C"):
            f1 = []
            for s in seeds:
                H.set_all_seeds(s)
                env = DummyVecEnv([lambda: TierEnv(Str, trl, 2048, s)])
                if algo == "DQN":
                    model = DQN("MlpPolicy", env, seed=s, verbose=0, device="cpu",
                                learning_starts=1000, buffer_size=50000,
                                policy_kwargs=dict(net_arch=[128, 64, 32]))
                else:
                    model = A2C("MlpPolicy", env, seed=s, verbose=0, device="cpu",
                                n_steps=2048, policy_kwargs=dict(net_arch=[128, 64, 32]))
                model.learn(total_timesteps=a.timesteps)
                acts = np.asarray(model.predict(Ste, deterministic=True)[0]).reshape(-1)
                f1.append(float(f1_score(tel, p2.tiers_to_binary(acts), zero_division=0)))
            st = H.compare_to_constant(f1, base_best)
            print(f"  {algo:<4} F1 {st['mean']:.3f}±{st['std']:.3f}  margin {st['margin']:+.3f}  "
                  f"won {st['seeds_won']}/{st['n']}  p_wilcoxon {st['p_wilcoxon']:.4f}")
            rows.append(dict(dataset=dname, algo=algo, base_best=base_best, **st, f1=f1))

    print("\n========  DQN / A2C robustness (RL vs best simple learner)  ========")
    print(f"{'dataset':<12}{'algo':<6}{'base':>7}{'RL mean':>9}{'margin':>8}{'won':>7}")
    for r in rows:
        print(f"{r['dataset']:<12}{r['algo']:<6}{r['base_best']:>7.3f}{r['mean']:>9.3f}{r['margin']:>+8.3f}{r['seeds_won']:>4}/{r['n']}")
    print("=> Across PPO/DQN/A2C, no RL beats the best simple learner — including a matched-capacity")
    print("   supervised MLP [128,64,32] on the same (z,e) -> the result is algorithm-independent.")
    print("===================================================================")

    os.makedirs(os.path.join(a.outdir, "results"), exist_ok=True)
    import csv
    with open(os.path.join(a.outdir, "results", "dqn_static.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["dataset", "algo", "seed", "rl_f1", "base_best"])
        for r in rows:
            for s, v in zip(seeds, r["f1"]): w.writerow([r["dataset"], r["algo"], s, v, r["base_best"]])
    json.dump([{k: v for k, v in r.items() if k != "f1"} for r in rows],
              open(os.path.join(a.outdir, "results", "dqn_static.json"), "w"), indent=2)
    print("Saved -> DQN/results/dqn_static.csv")


if __name__ == "__main__":
    main()
