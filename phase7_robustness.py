"""
Phase 7 — Robustness sweep + ablation for the autonomous-response RL win (Phase 6)
===================================================================================
Goal: prove the Phase-6 result is REAL, not a lucky setting. Two checks:

  1) ROBUSTNESS — train RL and tuned baselines across several SETTINGS x SEEDS and report how
     often (and by how much) RL beats the best tuned baseline. A genuine result wins consistently.

  2) ABLATION — retrain the RL agent with the temporal history REMOVED from its state (only the
     current error, no recent-mean / recent-max). If the advantage shrinks, that proves the win
     comes from integrating evidence over time (the mechanism), not from luck.

Reuses the Phase-6 simulator and grounds error signals in real NSL-KDD autoencoder errors.
Reproducible (fixed seeds). Heavier run (many PPO trainings) — expect ~15-25 min.

RUN (Colab):
  !python phase7_robustness.py --timesteps 150000 --seeds 42 7 2024
Outputs: phase7_metrics.json , phase7_robustness.png
"""
import argparse, importlib, json, os, subprocess, sys
import numpy as np

def _ensure(mod, pip_name=None):
    try: importlib.import_module(mod)
    except Exception:
        print(f"installing {pip_name or mod} ..."); subprocess.run([sys.executable,"-m","pip","install","-q",pip_name or mod], check=True)

import phase1_autoencoder_ids as p1
import phase3_drift as p3
import phase6_response as p6


SETTINGS = {
    "base":          dict(),
    "fast_attack":   dict(incs=(0.25, 0.08, 0.0)),     # compromise rises faster
    "costly_block":  dict(costs=(0.0, 0.05, 0.40)),    # blocking hurts availability more
}


def mask_obs(o, use_history):
    if use_history:
        return o
    o = np.array(o, np.float32); o[1] = 0.0; o[2] = 0.0   # zero recent-mean & recent-max
    return o


def make_env(ne, ae, setting, p_attack, seed, use_history, T):
    import gymnasium as gym
    from gymnasium import spaces

    class Env(gym.Env):
        def __init__(e):
            super().__init__()
            e.sim = p6.Simulator(ne, ae, T=T, **setting)
            e.action_space = spaces.Discrete(3)
            e.observation_space = spaces.Box(-10, 10, (6,), np.float32)
            e.rng = np.random.default_rng(seed)

        def reset(e, seed=None, options=None):
            super().reset(seed=seed)
            sc = p6.fill_scenario(p6.gen_scenario(e.rng, T, p_attack), e.rng, ne, ae, T)
            return mask_obs(e.sim.reset(sc), use_history), {}

        def step(e, a):
            o, r, d = e.sim.step(a)
            return mask_obs(o, use_history), r, d, False, {}
    return Env


def tune_baselines(sim, tune_scen):
    def block_cur(thr):   return lambda o: 2 if o[0] > thr else 0
    def two_cur(lo, hi):  return lambda o: 2 if o[0] > hi else (1 if o[0] > lo else 0)
    def two_mean(lo, hi): return lambda o: 2 if o[1] > hi else (1 if o[1] > lo else 0)
    cands = [block_cur(t) for t in np.linspace(0, 4, 9)]
    cands += [two_cur(lo, hi) for lo in np.linspace(0, 2, 5) for hi in np.linspace(1, 4, 5) if hi > lo]
    cands += [two_mean(lo, hi) for lo in np.linspace(0, 2, 5) for hi in np.linspace(1, 4, 5) if hi > lo]
    return max(p6.rollout(c, tune_scen, sim)["reward"] for c in cands)


def train_rl(ne, ae, setting, p_attack, seed, use_history, T, timesteps):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    Env = make_env(ne, ae, setting, p_attack, seed, use_history, T)
    env = DummyVecEnv([lambda: Env()])
    model = PPO("MlpPolicy", env, seed=seed, verbose=0, n_steps=2048, batch_size=256,
                device="cpu", policy_kwargs=dict(net_arch=[128, 64, 32]))
    model.learn(total_timesteps=timesteps)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="/content/drive/MyDrive/dataset/Train_data.csv")
    ap.add_argument("--ae-epochs", type=int, default=30)
    ap.add_argument("--timesteps", type=int, default=150000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 2024])
    ap.add_argument("--episode-T", type=int, default=50)
    ap.add_argument("--p-attack", type=float, default=0.5)
    ap.add_argument("--eval-episodes", type=int, default=2000)
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args()
    _ensure("gymnasium"); _ensure("stable_baselines3", "stable-baselines3"); p1.set_seed(a.seeds[0])
    if a.train.startswith("/content/drive") and not os.path.exists(a.train):
        try:
            from google.colab import drive; drive.mount("/content/drive")
        except Exception: pass

    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"Device: {dev}")

    # --- AE error pools (once) ---
    trf, trl = p1.load_csv(a.train)
    pre, _, _ = p1.build_preprocessor(trf)
    Xtr = pre.fit_transform(trf).astype(np.float32); dim = Xtr.shape[1]
    print("Training frozen autoencoder (once) ...")
    ae_model = p3.train_ae(Xtr[trl == 0], dim, dev, a.ae_epochs, a.seeds[0])
    _, err = p3.latent_err(ae_model, Xtr, dev)
    emu, esd = float(err[trl == 0].mean()), float(err[trl == 0].std()) + 1e-8
    ne = ((err[trl == 0] - emu) / esd).astype(np.float32)
    ae = ((err[trl == 1] - emu) / esd).astype(np.float32)
    T = a.episode_T

    # fixed eval / tune scenarios (common random numbers across ALL runs)
    er = np.random.default_rng(12345)
    eval_scen = [p6.fill_scenario(p6.gen_scenario(er, T, a.p_attack), er, ne, ae, T) for _ in range(a.eval_episodes)]
    tr = np.random.default_rng(54321)
    tune_scen = [p6.fill_scenario(p6.gen_scenario(tr, T, a.p_attack), tr, ne, ae, T) for _ in range(600)]

    # ---------- 1) ROBUSTNESS SWEEP ----------
    print("\n========== ROBUSTNESS SWEEP ==========")
    print(f"{'setting':<14}{'seed':>6}{'RL':>9}{'baseline':>10}{'margin':>9}{'win':>5}")
    sweep = {}; wins = 0; total = 0
    for sname, sval in SETTINGS.items():
        sim = p6.Simulator(ne, ae, T=T, **sval)
        base = tune_baselines(sim, tune_scen)
        margins = []
        for seed in a.seeds:
            model = train_rl(ne, ae, sval, a.p_attack, seed, True, T, a.timesteps)
            rlr = p6.rollout(lambda o: int(model.predict(o, deterministic=True)[0]), eval_scen, sim)["reward"]
            m = rlr - base; margins.append(m); total += 1; w = m > 0; wins += int(w)
            print(f"{sname:<14}{seed:>6}{rlr:>9.3f}{base:>10.3f}{m:>9.3f}{('Y' if w else 'n'):>5}")
        sweep[sname] = dict(baseline=base, rl_margins=margins,
                            mean_margin=float(np.mean(margins)), std_margin=float(np.std(margins)))
    print(f"\nRL beat the best tuned baseline in {wins}/{total} runs.")
    for s, v in sweep.items():
        print(f"  {s:<14} mean margin {v['mean_margin']:+.3f} ± {v['std_margin']:.3f}")

    # ---------- 2) ABLATION (base setting): full state vs no-history ----------
    print("\n========== ABLATION: remove temporal history ==========")
    sim = p6.Simulator(ne, ae, T=T)
    base = tune_baselines(sim, tune_scen)
    full_m, nohist_m = [], []
    for seed in a.seeds:
        mf = train_rl(ne, ae, {}, a.p_attack, seed, True, T, a.timesteps)
        rf = p6.rollout(lambda o: int(mf.predict(o, deterministic=True)[0]), eval_scen, sim)["reward"]
        mn = train_rl(ne, ae, {}, a.p_attack, seed, False, T, a.timesteps)
        rn = p6.rollout(lambda o: int(mn.predict(mask_obs(o, False), deterministic=True)[0]), eval_scen, sim)["reward"]
        full_m.append(rf - base); nohist_m.append(rn - base)
        print(f"  seed {seed:<6} full-RL margin {rf-base:+.3f} | no-history margin {rn-base:+.3f}")
    fm, nm = float(np.mean(full_m)), float(np.mean(nohist_m))
    print(f"\nFull-state RL margin {fm:+.3f}  vs  no-history RL margin {nm:+.3f}")
    print(f"-> history contributes {fm - nm:+.3f} reward "
          f"({'temporal integration IS the mechanism ✅' if fm > nm + 1e-3 else 'history not decisive — investigate'})")

    verdict = "ROBUST WIN ✅" if wins == total else (f"mostly wins ({wins}/{total})" if wins > total/2 else "NOT robust")
    print(f"\n=========== VERDICT: {verdict} ===========")

    json.dump(dict(seeds=a.seeds, timesteps=a.timesteps, sweep=sweep, wins=wins, total=total,
                   ablation=dict(full_mean=fm, nohist_mean=nm, history_value=fm - nm)),
              open(os.path.join(a.outdir, "phase7_metrics.json"), "w"), indent=2)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        xs = list(SETTINGS); ms = [sweep[s]["mean_margin"] for s in xs]; es = [sweep[s]["std_margin"] for s in xs]
        ax[0].bar(xs, ms, yerr=es, color="tab:green", capsize=5)
        ax[0].axhline(0, color="k", lw=1); ax[0].set_ylabel("RL margin over best baseline")
        ax[0].set_title(f"Robustness ({len(a.seeds)} seeds/setting)")
        ax[1].bar(["full state", "no history"], [fm, nm], color=["tab:green", "tab:gray"])
        ax[1].axhline(0, color="k", lw=1); ax[1].set_ylabel("RL margin over baseline")
        ax[1].set_title("Ablation: value of temporal history")
        plt.tight_layout(); png = os.path.join(a.outdir, "phase7_robustness.png"); plt.savefig(png, dpi=130)
        print(f"Saved plot -> {png}")
    except Exception as ex:
        print(f"(plot skipped: {ex})")
    print("Saved -> phase7_metrics.json")


if __name__ == "__main__":
    main()
