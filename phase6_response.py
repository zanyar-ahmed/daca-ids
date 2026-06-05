"""
Phase 6 — Autonomous Response IDS via RL  (the genuinely RL-shaped problem)
===========================================================================
Earlier phases showed "flag or not" is threshold-reducible, so RL can't win there. This phase
gives RL a problem with the structure it is actually built for: a sequential MDP with delayed
consequences and graded actions.

A host is monitored for T steps. An attack campaign may start mid-episode (hidden). Each step the
agent sees a noisy detection signal (autoencoder reconstruction error from REAL NSL-KDD flows,
plus its recent history and the current compromise level) and picks a RESPONSE:
    0 monitor   (no cost, no effect)
    1 throttle  (small availability cost, slows the attack)
    2 block     (large availability cost, stops the attack)
Dynamics: while under active attack, compromise rises unless throttled/blocked; if it reaches 1.0
the host is BREACHED (large penalty). Responding when there is no attack wastes availability.
The agent must integrate evidence over time, escalate before a breach, and not over-react to
benign noise — a planning problem a per-flow threshold cannot solve.

Baselines (same episodes, common random numbers, tuned): no-response, block-on-error,
two-level thresholds on the current error AND on the recent-mean error. RL must beat them on
total cost (reward), breach rate, and availability cost. Reproducible (fixed seed).

RUN (Colab):
  !python phase6_response.py --ae-epochs 30 --timesteps 300000
Outputs: phase6_metrics.json , phase6_response.png
"""
import argparse, importlib, json, os, subprocess, sys
from collections import deque
import numpy as np

def _ensure(mod, pip_name=None):
    try: importlib.import_module(mod)
    except Exception:
        print(f"installing {pip_name or mod} ..."); subprocess.run([sys.executable,"-m","pip","install","-q",pip_name or mod], check=True)

import phase1_autoencoder_ids as p1
import phase3_drift as p3


# ---------------- simulator (pure numpy; shared by training env and evaluation) ----------------
class Simulator:
    def __init__(s, ne, ae, T=50, Wc=5.0, breach_pen=10.0,
                 costs=(0.0, 0.05, 0.20), incs=(0.15, 0.05, 0.0)):
        s.ne, s.ae, s.T = ne, ae, T
        s.Wc, s.bp, s.costs, s.incs = Wc, breach_pen, costs, incs

    def reset(s, sc):
        s.sc = sc; s.t = 0; s.comp = 0.0; s.resp = 0
        s.hist = deque([sc["nseq"][0]], maxlen=5); s.cur = sc["nseq"][0]
        s.avail = 0.0; s.breached = False
        return s._obs()

    def _active(s):
        return s.sc["attack"] and s.t >= s.sc["start"]

    def _err(s):
        return s.sc["aseq"][s.t] if s._active() else s.sc["nseq"][s.t]

    def _obs(s):
        h = np.array(s.hist, np.float32)
        return np.clip(np.array([s.cur, h.mean(), h.max(), s.comp, s.resp / 2.0, s.t / s.T],
                                np.float32), -10, 10)

    def step(s, a):
        a = int(a); s.resp = a
        inc = s.incs[a] if s._active() else 0.0
        s.comp = min(1.0, s.comp + inc)
        cost = s.costs[a]; s.avail += cost
        reward = -cost - inc * s.Wc
        done = False
        if s.comp >= 1.0:
            reward -= s.bp; s.breached = True; done = True
        s.t += 1
        if s.t >= s.T: done = True
        if not done:
            s.cur = s._err(); s.hist.append(s.cur)
        return s._obs(), reward, done


def gen_scenario(rng, T, p_attack):
    return dict(attack=bool(rng.random() < p_attack),
                start=int(rng.integers(0, max(1, T // 2))),
                aseq=None, nseq=None, _rng=None)  # filled below


def fill_scenario(sc, rng, ne, ae, T):
    sc["aseq"] = ae[rng.integers(0, len(ae), T)]
    sc["nseq"] = ne[rng.integers(0, len(ne), T)]
    return sc


def make_train_env(sim, ne, ae, p_attack, seed):
    import gymnasium as gym
    from gymnasium import spaces

    class Env(gym.Env):
        def __init__(e):
            super().__init__()
            e.action_space = spaces.Discrete(3)
            e.observation_space = spaces.Box(-10, 10, (6,), np.float32)
            e.rng = np.random.default_rng(seed)

        def reset(e, seed=None, options=None):
            super().reset(seed=seed)
            sc = fill_scenario(gen_scenario(e.rng, sim.T, p_attack), e.rng, ne, ae, sim.T)
            return sim.reset(sc), {}

        def step(e, a):
            o, r, d = sim.step(a)
            return o, r, d, False, {}
    return Env


def rollout(policy, scenarios, sim):
    """Run a policy over fixed scenarios (common random numbers). policy(obs)->action."""
    R, breaches, avail = [], 0, []
    for sc in scenarios:
        o = sim.reset(sc); done = False; tot = 0.0
        while not done:
            o, r, done = sim.step(policy(o)); tot += r
        R.append(tot); avail.append(sim.avail); breaches += int(sim.breached)
    return dict(reward=float(np.mean(R)), breach_rate=breaches / len(scenarios),
                avail_cost=float(np.mean(avail)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="/content/drive/MyDrive/dataset/Train_data.csv")
    ap.add_argument("--test", default="/content/drive/MyDrive/dataset/Test_data.csv")
    ap.add_argument("--ae-epochs", type=int, default=30)
    ap.add_argument("--timesteps", type=int, default=300000)
    ap.add_argument("--episode-T", type=int, default=50)
    ap.add_argument("--p-attack", type=float, default=0.5)
    ap.add_argument("--eval-episodes", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args()
    _ensure("gymnasium"); _ensure("stable_baselines3", "stable-baselines3"); p1.set_seed(a.seed)
    if a.train.startswith("/content/drive") and not os.path.exists(a.train):
        try:
            from google.colab import drive; drive.mount("/content/drive")
        except Exception: pass

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"Device: {dev} | seed {a.seed}")

    # --- frozen AE -> error pools (standardised) ---
    trf, trl = p1.load_csv(a.train)
    pre, _, _ = p1.build_preprocessor(trf)
    Xtr = pre.fit_transform(trf).astype(np.float32)
    dim = Xtr.shape[1]; print(f"input_dim={dim}\nTraining frozen autoencoder ...")
    ae_model = p3.train_ae(Xtr[trl == 0], dim, dev, a.ae_epochs, a.seed)
    _, err = p3.latent_err(ae_model, Xtr, dev)
    emu, esd = float(err[trl == 0].mean()), float(err[trl == 0].std()) + 1e-8
    ne = ((err[trl == 0] - emu) / esd).astype(np.float32)   # normal error pool
    ae = ((err[trl == 1] - emu) / esd).astype(np.float32)   # attack error pool
    print(f"error pools: normal {len(ne)} (mean {ne.mean():.2f}) | attack {len(ae)} (mean {ae.mean():.2f})")

    sim = Simulator(ne, ae, T=a.episode_T)

    # --- train PPO ---
    print(f"\nTraining response PPO ({a.timesteps:,} steps) ...")
    Env = make_train_env(sim, ne, ae, a.p_attack, a.seed)
    env = DummyVecEnv([lambda: Env()])
    model = PPO("MlpPolicy", env, seed=a.seed, verbose=0, n_steps=2048, batch_size=256,
                device="cpu", policy_kwargs=dict(net_arch=[128, 64, 32]))
    model.learn(total_timesteps=a.timesteps); print("PPO done.")

    # --- fixed evaluation scenarios (common random numbers for all policies) ---
    ev_rng = np.random.default_rng(a.seed + 7)
    scen = [fill_scenario(gen_scenario(ev_rng, a.episode_T, a.p_attack), ev_rng, ne, ae, a.episode_T)
            for _ in range(a.eval_episodes)]
    # tuning scenarios (separate) for picking baseline thresholds fairly
    tu_rng = np.random.default_rng(a.seed + 99)
    tune = [fill_scenario(gen_scenario(tu_rng, a.episode_T, a.p_attack), tu_rng, ne, ae, a.episode_T)
            for _ in range(800)]

    def block_cur(thr):  return lambda o: 2 if o[0] > thr else 0
    def two_cur(lo, hi): return lambda o: 2 if o[0] > hi else (1 if o[0] > lo else 0)
    def two_mean(lo, hi):return lambda o: 2 if o[1] > hi else (1 if o[1] > lo else 0)

    def best(make, grid):
        b = None
        for params in grid:
            m = rollout(make(*params), tune, sim)["reward"]
            if b is None or m > b[0]: b = (m, params)
        return make(*b[1]), b[1]

    grid1 = [(t,) for t in np.linspace(0, 4, 9)]
    grid2 = [(lo, hi) for lo in np.linspace(0, 2, 5) for hi in np.linspace(1, 4, 5) if hi > lo]
    pol_block, p_b = best(block_cur, grid1)
    pol_tc, p_tc = best(two_cur, grid2)
    pol_tm, p_tm = best(two_mean, grid2)

    policies = {
        "No response": lambda o: 0,
        f"Block-on-error (thr={p_b[0]:.1f})": pol_block,
        f"Two-level on current": pol_tc,
        f"Two-level on recent-mean": pol_tm,
        "RL (learned)": lambda o: int(model.predict(o, deterministic=True)[0]),
    }
    print("\n==============  AUTONOMOUS RESPONSE (test episodes)  ==============")
    print(f"{'policy':<30}{'reward':>9}{'breach%':>9}{'avail':>8}")
    res = {}
    for name, pol in policies.items():
        m = rollout(pol, scen, sim); res[name] = m
        print(f"{name:<30}{m['reward']:>9.3f}{100*m['breach_rate']:>9.1f}{m['avail_cost']:>8.2f}")
    rl = res["RL (learned)"]["reward"]
    best_base = max(v["reward"] for k, v in res.items() if k != "RL (learned)")
    print(f"\nVerdict: {'RL WINS (highest reward = lowest total cost) ✅' if rl > best_base + 1e-3 else 'RL does not beat best baseline — tune timesteps/dynamics'}"
          f"  (RL {rl:.3f} vs best baseline {best_base:.3f})")
    print("==================================================================")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        names = list(res.keys()); rew = [res[n]["reward"] for n in names]
        plt.figure(figsize=(9, 5))
        plt.barh(names, rew, color=["gray"] * (len(names) - 1) + ["tab:green"])
        plt.xlabel("mean episode reward  (higher = better: fewer breaches + less disruption)")
        plt.title("Autonomous response: RL vs tuned baselines"); plt.tight_layout()
        png = os.path.join(a.outdir, "phase6_response.png"); plt.savefig(png, dpi=130); print(f"Saved plot -> {png}")
    except Exception as ex:
        print(f"(plot skipped: {ex})")

    json.dump({"seed": a.seed, "results": res, "rl_reward": rl, "best_baseline_reward": best_base},
              open(os.path.join(a.outdir, "phase6_metrics.json"), "w"), indent=2)
    print("Saved -> phase6_metrics.json")


if __name__ == "__main__":
    main()
