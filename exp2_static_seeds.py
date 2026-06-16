"""
Experiment 2 — Per-seed variance + 95% CI for the static RL detector (Protocol #2/#3)  ⭐
The backbone rigor result: PPO tier-controller vs the F1-OPTIMAL threshold, across 15 seeds,
on NSL-KDD AND UNSW-NB15, with mean±std, 95% CI, Wilcoxon/t-test, Cohen's d, seeds-won.
Deterministic frozen autoencoder (CPU) so the only randomness is the RL seed.

Produces Table A (static-detection rows). RUN (Colab):
  !python exp2_static_seeds.py --epochs 40 --timesteps 80000
Outputs: results/exp2_static_seeds.csv
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
import phase8_unsw as p8
import exp_harness as H


def make_states(ae, X, dev, zmu, zsd, emu, esd):
    Z, E = p3.latent_err(ae, X, dev)
    S = np.concatenate([np.clip((Z - zmu) / zsd, -10, 10),
                        np.clip(((E - emu) / esd)[:, None], -10, 10)], 1).astype(np.float32)
    return S, E


def run_dataset(name, load, preproc, epochs, seeds, timesteps):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from sklearn.metrics import f1_score
    dev = torch.device("cpu")
    (trf, trl), (tef, tel) = load()
    pre, _, _ = preproc(trf)
    Xtr = pre.fit_transform(trf).astype(np.float32); Xte = pre.transform(tef).astype(np.float32)
    H.set_all_seeds(0)
    ae = p3.train_ae(Xtr[trl == 0], Xtr.shape[1], dev, epochs, 0)        # deterministic, frozen
    Ztr, Etr = p3.latent_err(ae, Xtr, dev)
    zmu, zsd = Ztr.mean(0), Ztr.std(0) + 1e-8; emu, esd = float(Etr.mean()), float(Etr.std()) + 1e-8
    Str = np.concatenate([np.clip((Ztr - zmu) / zsd, -10, 10),
                          np.clip(((Etr - emu) / esd)[:, None], -10, 10)], 1).astype(np.float32)
    Ste, Ete = make_states(ae, Xte, dev, zmu, zsd, emu, esd)
    base = p8.best_threshold(Ete, tel)["f1"]                              # F1-optimal threshold

    TierEnv = p2.make_env_class(); rl_f1 = []
    for s in seeds:
        H.set_all_seeds(s)
        env = DummyVecEnv([lambda: TierEnv(Str, trl, 2048, s)])
        model = PPO("MlpPolicy", env, seed=s, verbose=0, n_steps=2048, batch_size=256,
                    device="cpu", policy_kwargs=dict(net_arch=[128, 64, 32]))
        model.learn(total_timesteps=timesteps)
        acts = np.asarray(model.predict(Ste, deterministic=True)[0]).reshape(-1)
        f1 = float(f1_score(tel, p2.tiers_to_binary(acts), zero_division=0))
        rl_f1.append(f1); print(f"  {name} seed {s:<3} RL F1 {f1:.3f}")
    st = H.compare_to_constant(rl_f1, base)
    print(f"  -> {name}: baseline(F1-opt threshold) {base:.3f} | RL {st['mean']:.3f}±{st['std']:.3f} "
          f"95% CI [{st['ci_lo']:.3f},{st['ci_hi']:.3f}] | margin {st['margin']:+.3f} | "
          f"won {st['seeds_won']}/{st['n']} | p_wilcoxon {st['p_wilcoxon']:.4f} | d {st['cohen_d']:.2f}")
    return dict(dataset=name, baseline=base, **st, rl_f1=rl_f1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsl-train", default="/content/drive/MyDrive/dataset/Train_data.csv")
    ap.add_argument("--nsl-test", default="/content/drive/MyDrive/dataset/Test_data.csv")
    ap.add_argument("--unsw-train", default="/content/drive/MyDrive/dataset/UNSW_NB15_training-set.parquet")
    ap.add_argument("--unsw-test", default="/content/drive/MyDrive/dataset/UNSW_NB15_testing-set.parquet")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--timesteps", type=int, default=80000)
    ap.add_argument("--n-seeds", type=int, default=15)
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args()
    _ensure("pyarrow"); _ensure("gymnasium"); _ensure("stable_baselines3", "stable-baselines3")
    seeds = H.SEEDS[:a.n_seeds]
    DS = {
        "NSL-KDD":   (lambda: (p1.load_csv(a.nsl_train), p1.load_csv(a.nsl_test)), p1.build_preprocessor),
        "UNSW-NB15": (lambda: (p8.load_unsw(a.unsw_train), p8.load_unsw(a.unsw_test)), p8.build_preprocessor),
    }
    rows = []
    for name, (load, pre) in DS.items():
        print(f"\n===== {name} ({len(seeds)} seeds) =====")
        rows.append(run_dataset(name, load, pre, a.epochs, seeds, a.timesteps))

    # Holm correction across the datasets' p-values
    p_w = [r["p_wilcoxon"] for r in rows]
    adj, rej = H.holm(p_w)
    print("\n================  TABLE A (static detection: RL vs F1-optimal threshold)  ================")
    print(f"{'dataset':<12}{'base':>7}{'RL mean':>9}{'±std':>7}{'margin':>8}{'won':>6}{'p_holm':>9}{'d':>7}")
    for r, pa in zip(rows, adj):
        print(f"{r['dataset']:<12}{r['baseline']:>7.3f}{r['mean']:>9.3f}{r['std']:>7.3f}"
              f"{r['margin']:>+8.3f}{r['seeds_won']:>4}/{r['n']}{pa:>9.4f}{r['cohen_d']:>7.2f}")
    print("=========================================================================================")

    os.makedirs(os.path.join(a.outdir, "results"), exist_ok=True)
    import csv
    with open(os.path.join(a.outdir, "results", "exp2_static_seeds.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["dataset", "seed", "rl_f1", "baseline_f1"])
        for r in rows:
            for s, f1 in zip(seeds, r["rl_f1"]): w.writerow([r["dataset"], s, f1, r["baseline"]])
    json.dump([{k: v for k, v in r.items() if k != "rl_f1"} for r in rows],
              open(os.path.join(a.outdir, "results", "exp2_summary.json"), "w"), indent=2)
    print("Saved -> results/exp2_static_seeds.csv (+ exp2_summary.json)")


if __name__ == "__main__":
    main()
