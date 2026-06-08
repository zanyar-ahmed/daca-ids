"""
Phase 7 — Robustness sweep + ablation (RIGOROUS) for the autonomous-response RL win
====================================================================================
Hardened version to remove the run-to-run wobble and give defensible statistics:
  * DETERMINISTIC autoencoder: trained on CPU with a fixed seed and cuDNN determinism, so the
    error pools (and therefore the whole experiment) are identical on every run/session.
  * MANY SEEDS (default 10) for the RL agent, so margins have meaningful spread.
  * 95% CONFIDENCE INTERVALS + a significance flag (CI lower bound > 0 => a real win).
  * ABLATION: full-state RL vs no-temporal-history RL (reuses the base-setting full runs).

A genuine result = RL's margin over the best tuned baseline is positive with a 95% CI that
EXCLUDES ZERO, consistently across settings; and full-state beats no-history.
Reuses the Phase-6 simulator; signals grounded in real NSL-KDD autoencoder errors. Reproducible.

RUN (Colab) — long (~45-75 min, many PPO trainings):
  !python phase7_robustness.py --timesteps 150000 --n-seeds 10
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
    "fast_attack":   dict(incs=(0.25, 0.08, 0.0)),
    "costly_block":  dict(costs=(0.0, 0.05, 0.40)),
}
SEED_POOL = [42, 7, 2024, 1, 99, 123, 2025, 5, 77, 314, 808, 1234, 2718, 161, 50]


def mean_ci(xs):
    """Return (mean, half_width_95, lo, hi) using a t-interval."""
    xs = np.asarray(xs, float); n = len(xs); m = float(xs.mean())
    if n < 2:
        return m, 0.0, m, m
    s = float(xs.std(ddof=1))
    try:
        from scipy import stats; tcrit = float(stats.t.ppf(0.975, n - 1))
    except Exception:
        tcrit = 1.96
    half = tcrit * s / np.sqrt(n)
    return m, half, m - half, m + half


def mask_obs(o, use_history):
    if use_history:
        return o
    o = np.array(o, np.float32); o[1] = 0.0; o[2] = 0.0
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


def eval_rl(model, sim, eval_scen, use_history):
    return p6.rollout(lambda o: int(model.predict(mask_obs(o, use_history), deterministic=True)[0]),
                      eval_scen, sim)["reward"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="/content/drive/MyDrive/dataset/Train_data.csv")
    ap.add_argument("--ae-epochs", type=int, default=40)
    ap.add_argument("--timesteps", type=int, default=150000)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--episode-T", type=int, default=50)
    ap.add_argument("--p-attack", type=float, default=0.5)
    ap.add_argument("--eval-episodes", type=int, default=3000)
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args()
    seeds = SEED_POOL[:a.n_seeds]
    _ensure("gymnasium"); _ensure("stable_baselines3", "stable-baselines3")
    if a.train.startswith("/content/drive") and not os.path.exists(a.train):
        try:
            from google.colab import drive; drive.mount("/content/drive")
        except Exception: pass

    import torch
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    p1.set_seed(1234)
    dev_ae = torch.device("cpu")     # AE on CPU -> identical error pools every run
    print(f"AE device: {dev_ae} (deterministic) | seeds: {seeds}")

    # --- DETERMINISTIC AE -> fixed error pools ---
    trf, trl = p1.load_csv(a.train)
    pre, _, _ = p1.build_preprocessor(trf)
    Xtr = pre.fit_transform(trf).astype(np.float32); dim = Xtr.shape[1]
    print("Training deterministic frozen autoencoder ...")
    ae_model = p3.train_ae(Xtr[trl == 0], dim, dev_ae, a.ae_epochs, 1234)
    _, err = p3.latent_err(ae_model, Xtr, dev_ae)
    emu, esd = float(err[trl == 0].mean()), float(err[trl == 0].std()) + 1e-8
    ne = ((err[trl == 0] - emu) / esd).astype(np.float32)
    ae = ((err[trl == 1] - emu) / esd).astype(np.float32)
    T = a.episode_T
    print(f"error pools: normal mean {ne.mean():.3f} | attack mean {ae.mean():.3f}")

    er = np.random.default_rng(12345)
    eval_scen = [p6.fill_scenario(p6.gen_scenario(er, T, a.p_attack), er, ne, ae, T) for _ in range(a.eval_episodes)]
    tr = np.random.default_rng(54321)
    tune_scen = [p6.fill_scenario(p6.gen_scenario(tr, T, a.p_attack), tr, ne, ae, T) for _ in range(600)]

    # ---------- ROBUSTNESS SWEEP ----------
    print("\n========== ROBUSTNESS SWEEP (margin over best tuned baseline) ==========")
    sweep = {}; base_full_margins = []
    for sname, sval in SETTINGS.items():
        sim = p6.Simulator(ne, ae, T=T, **sval)
        base = tune_baselines(sim, tune_scen)
        margins = []
        for i, seed in enumerate(seeds):
            model = train_rl(ne, ae, sval, a.p_attack, seed, True, T, a.timesteps)
            margins.append(eval_rl(model, sim, eval_scen, True) - base)
            print(f"  {sname:<13} seed {seed:<5} ({i+1}/{len(seeds)})  margin {margins[-1]:+.3f}")
            if sname == "base":
                base_full_margins.append(margins[-1])
        m, half, lo, hi = mean_ci(margins)
        sig = lo > 0
        sweep[sname] = dict(baseline=base, margins=margins, mean=m, ci_half=half, lo=lo, hi=hi,
                            wins=int(sum(x > 0 for x in margins)), significant=bool(sig))
        print(f"  -> {sname}: margin {m:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
              f"wins {sweep[sname]['wins']}/{len(seeds)}  {'SIGNIFICANT WIN ✅' if sig else 'not significant'}")

    # ---------- ABLATION (base): no-history vs full ----------
    print("\n========== ABLATION: remove temporal history (base setting) ==========")
    sim = p6.Simulator(ne, ae, T=T)
    base = sweep["base"]["baseline"]
    nohist_margins = []
    for i, seed in enumerate(seeds):
        mn = train_rl(ne, ae, {}, a.p_attack, seed, False, T, a.timesteps)
        nohist_margins.append(eval_rl(mn, sim, eval_scen, False) - base)
        print(f"  no-history seed {seed:<5} ({i+1}/{len(seeds)})  margin {nohist_margins[-1]:+.3f}")
    fm, fh, flo, fhi = mean_ci(base_full_margins)
    nm, nh, nlo, nhi = mean_ci(nohist_margins)
    diff, dh, dlo, dhi = mean_ci(np.array(base_full_margins) - np.array(nohist_margins))
    print(f"\n  full-state margin : {fm:+.3f}  95% CI [{flo:+.3f}, {fhi:+.3f}]")
    print(f"  no-history margin : {nm:+.3f}  95% CI [{nlo:+.3f}, {nhi:+.3f}]")
    print(f"  history value (full - no-history): {diff:+.3f}  95% CI [{dlo:+.3f}, {dhi:+.3f}]  "
          f"{'history MATTERS ✅' if dlo > 0 else 'history not significant'}")

    # ---------- verdict ----------
    sig_settings = [s for s, v in sweep.items() if v["significant"]]
    print("\n=========== VERDICT ===========")
    print(f"RL significantly beats the best tuned baseline in: {sig_settings if sig_settings else 'NO settings'}")
    print(f"(base is the main regime; fast_attack/costly_block test robustness)")

    json.dump(dict(seeds=seeds, timesteps=a.timesteps, sweep=sweep,
                   ablation=dict(full_mean=fm, full_ci=[flo, fhi], nohist_mean=nm, nohist_ci=[nlo, nhi],
                                 history_value=diff, history_ci=[dlo, dhi])),
              open(os.path.join(a.outdir, "phase7_metrics.json"), "w"), indent=2)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        xs = list(SETTINGS); ms = [sweep[s]["mean"] for s in xs]; es = [sweep[s]["ci_half"] for s in xs]
        cols = ["tab:green" if sweep[s]["significant"] else "tab:gray" for s in xs]
        ax[0].bar(xs, ms, yerr=es, color=cols, capsize=6)
        ax[0].axhline(0, color="k", lw=1); ax[0].set_ylabel("RL margin over best baseline (95% CI)")
        ax[0].set_title(f"Robustness ({len(seeds)} seeds; green = significant)")
        ax[1].bar(["full state", "no history"], [fm, nm], yerr=[fh, nh],
                  color=["tab:green", "tab:gray"], capsize=6)
        ax[1].axhline(0, color="k", lw=1); ax[1].set_ylabel("RL margin over baseline (95% CI)")
        ax[1].set_title("Ablation: value of temporal history")
        plt.tight_layout(); png = os.path.join(a.outdir, "phase7_robustness.png"); plt.savefig(png, dpi=130)
        print(f"Saved plot -> {png}")
    except Exception as ex:
        print(f"(plot skipped: {ex})")
    print("Saved -> phase7_metrics.json")


if __name__ == "__main__":
    main()
