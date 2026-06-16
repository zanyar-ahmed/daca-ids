"""
Experiment 5 — Adaptive adversary + cost-ratio variants of the response game (Protocol #5)  ⭐⭐
Hardens the Phase-6/7 conclusion against the #1 reviewer concern (the static simulator). The
attacker now EVADES (lowers its anomaly signature by eps for a cooldown) whenever it was
throttled/blocked, and we sweep the breach-vs-availability cost ratio. For each variant we train
the RL defender across seeds and compare to the best tuned heuristic, with 95% CI + significance.

Outcome is win-win: RL still loses -> conclusion hardens; RL wins once the adversary adapts ->
that is the positive half of the story ("RL earns its place when the environment is adaptive").

Produces Table D. RUN (Colab):
  !python exp5_adaptive_adversary.py --seeds 8 --timesteps 100000
Outputs: results/exp5_adaptive.csv
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
import exp_harness as H

AVAIL = np.array([0.0, 0.05, 0.20])   # monitor / throttle / block availability cost
INC   = np.array([0.15, 0.05, 0.0])   # compromise increment per step if attack active & unblocked
COOLDOWN = 3


class AdaptiveSim:
    """Host-response game with an EVADING adversary and a tunable cost ratio.
    cost_ratio = breach_cost / availability_cost  (scales the security side)."""
    def __init__(s, ne, ae, T=50, eps=0.0, cost_ratio=1.0, p_attack=0.5):
        s.ne, s.ae, s.T, s.eps = ne, ae, T, eps
        s.Wc = 5.0 * cost_ratio; s.bp = 10.0 * cost_ratio; s.p_attack = p_attack

    def reset(s, seed):
        rng = np.random.default_rng(seed)
        s.attack = bool(rng.random() < s.p_attack)
        s.start = int(rng.integers(0, max(1, s.T // 2)))
        s.aidx = rng.integers(0, len(s.ae), s.T); s.nidx = rng.integers(0, len(s.ne), s.T)
        s.evcoin = rng.random(s.T)               # evasion coin per step (deterministic given seed)
        s.t = 0; s.comp = 0.0; s.resp = 0; s.cool = 0; s.breached = False; s.avail = 0.0
        s.hist = deque([s._err()], maxlen=5)
        return s._obs()

    def _active(s):
        return s.attack and s.t >= s.start

    def _err(s):
        if s._active():
            # evasion: while in cooldown (recently throttled/blocked), the attack MIMICS normal
            # traffic with probability eps -> its anomaly signature looks benign.
            if s.cool > 0 and s.evcoin[s.t] < s.eps:
                return float(s.ne[s.nidx[s.t]])
            return float(s.ae[s.aidx[s.t]])
        return float(s.ne[s.nidx[s.t]])

    def _obs(s):
        # compromise is HIDDEN (no ground-truth meter in reality); the defender sees only the
        # detection signal (current/recent error), its own last action, and time.
        h = np.array(s.hist, np.float32)
        cur = s.cur if hasattr(s, "cur") else s.hist[-1]
        return np.clip(np.array([cur, h.mean(), h.max(), s.resp / 2.0, s.t / s.T], np.float32), -10, 10)

    def step(s, a):
        a = int(a); s.resp = a
        inc = INC[a] if s._active() else 0.0
        s.comp = min(1.0, s.comp + inc); s.avail += AVAIL[a]
        r = -AVAIL[a] - inc * s.Wc
        if s.comp >= 1.0 and not s.breached:
            r -= s.bp; s.breached = True
        # adversary reaction: if it was throttled/blocked during an active attack -> evade next steps
        if s._active() and a in (1, 2):
            s.cool = COOLDOWN
        else:
            s.cool = max(0, s.cool - 1)
        s.t += 1
        if s.t < s.T:
            s.cur = s._err(); s.hist.append(s.cur)
        return s._obs(), r, s.t >= s.T


def make_env(sim_kwargs, ne, ae, seed):
    import gymnasium as gym
    from gymnasium import spaces
    class Env(gym.Env):
        def __init__(e):
            super().__init__(); e.sim = AdaptiveSim(ne, ae, **sim_kwargs)
            e.action_space = spaces.Discrete(3)
            e.observation_space = spaces.Box(-10, 10, (5,), np.float32)
            e.rng = np.random.default_rng(seed); e.ep = 0
        def reset(e, seed=None, options=None):
            super().reset(seed=seed); e.ep += 1
            return e.sim.reset(int(e.rng.integers(1 << 30)) + e.ep), {}
        def step(e, a):
            o, r, d = e.sim.step(a); return o, r, d, False, {}
    return Env


def rollout(policy, sim, seeds):
    R = []
    for sd in seeds:
        o = sim.reset(sd); tot = 0.0; done = False
        while not done:
            o, r, done = sim.step(policy(o)); tot += r
        R.append(tot)
    return np.array(R)


def tune_baseline(sim, tune_seeds):
    """Strong, FAIR tuned heuristics on the observable detection signal: thresholds on the
    current error AND on the recent-mean / recent-max (memory, to resist evasion)."""
    cands = []
    for lo in np.linspace(0, 2, 5):
        for hi in np.linspace(1, 4, 5):
            if hi > lo:
                cands.append(lambda o, lo=lo, hi=hi: 2 if o[1] > hi else (1 if o[1] > lo else 0))
                cands.append(lambda o, lo=lo, hi=hi: 2 if o[2] > hi else (1 if o[2] > lo else 0))  # on recent-max
    for thr in np.linspace(0, 4, 9):
        cands.append(lambda o, thr=thr: 2 if o[0] > thr else 0)
    return max(cands, key=lambda p: rollout(p, sim, tune_seeds).mean())


def run_variant(tag, sim_kwargs, ne, ae, seeds, timesteps, eval_seeds, tune_seeds):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    sim = AdaptiveSim(ne, ae, **sim_kwargs)
    base_pol = tune_baseline(sim, tune_seeds)
    base = rollout(base_pol, sim, eval_seeds).mean()
    margins = []
    for s in seeds:
        H.set_all_seeds(s)
        env = DummyVecEnv([lambda: make_env(sim_kwargs, ne, ae, s)()])
        model = PPO("MlpPolicy", env, seed=s, verbose=0, n_steps=2048, batch_size=256,
                    device="cpu", policy_kwargs=dict(net_arch=[128, 64, 32]))
        model.learn(total_timesteps=timesteps)
        rl = rollout(lambda o: int(model.predict(o, deterministic=True)[0]), sim, eval_seeds).mean()
        margins.append(rl - base)
    st = H.compare_to_constant(np.array(margins) + base, base)   # test RL distribution vs base
    print(f"  {tag:<22} base {base:+.2f} | RL margin {st['margin']:+.3f} "
          f"CI[{st['ci_lo']-base:+.3f},{st['ci_hi']-base:+.3f}] won {st['seeds_won']}/{st['n']} p {st['p_wilcoxon']:.3f}")
    return dict(variant=tag, baseline=base, margin=st["margin"], ci_lo=st["ci_lo"] - base,
                ci_hi=st["ci_hi"] - base, seeds_won=st["seeds_won"], n=st["n"],
                p_wilcoxon=st["p_wilcoxon"], cohen_d=st["cohen_d"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="/content/drive/MyDrive/dataset/Train_data.csv")
    ap.add_argument("--ae-epochs", type=int, default=40)
    ap.add_argument("--timesteps", type=int, default=100000)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--eval-episodes", type=int, default=1500)
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args()
    _ensure("gymnasium"); _ensure("stable_baselines3", "stable-baselines3"); H.set_all_seeds(0)
    if a.train.startswith("/content/drive") and not os.path.exists(a.train):
        try:
            from google.colab import drive; drive.mount("/content/drive")
        except Exception: pass
    import torch
    dev = torch.device("cpu")

    # deterministic AE -> error pools (standardised on Normal)
    trf, trl = p1.load_csv(a.train); pre, _, _ = p1.build_preprocessor(trf)
    Xtr = pre.fit_transform(trf).astype(np.float32)
    ae_m = p3.train_ae(Xtr[trl == 0], Xtr.shape[1], dev, a.ae_epochs, 0)
    _, err = p3.latent_err(ae_m, Xtr, dev)
    emu, esd = float(err[trl == 0].mean()), float(err[trl == 0].std()) + 1e-8
    ne = ((err[trl == 0] - emu) / esd).astype(np.float32)
    ae = ((err[trl == 1] - emu) / esd).astype(np.float32)

    seeds = H.SEEDS[:a.seeds]
    ev = list(range(20000, 20000 + a.eval_episodes)); tu = list(range(30000, 30000 + 600))
    variants = []
    print("Adversary-evasion sweep (cost_ratio=1; eps = mimic-normal probability while evading):")
    for eps in (0.0, 0.25, 0.5, 0.75):
        variants.append(run_variant(f"evasion eps={eps}", dict(eps=eps, cost_ratio=1.0),
                                     ne, ae, seeds, a.timesteps, ev, tu))
    print("Cost-ratio sweep (eps=0.2):")
    for cr in (0.25, 0.5, 1.0, 2.0, 4.0):
        variants.append(run_variant(f"cost_ratio={cr}", dict(eps=0.2, cost_ratio=cr),
                                     ne, ae, seeds, a.timesteps, ev, tu))

    print("\n================  TABLE D (adaptive adversary / cost ratio)  ================")
    print(f"{'variant':<22}{'RL margin':>11}{'95% CI':>22}{'won':>7}{'verdict':>16}")
    for v in variants:
        verdict = "RL wins" if v["ci_lo"] > 0 else ("RL loses" if v["ci_hi"] < 0 else "tie")
        print(f"{v['variant']:<22}{v['margin']:>+11.3f}  [{v['ci_lo']:+.3f},{v['ci_hi']:+.3f}]"
              f"{v['seeds_won']:>5}/{v['n']}{verdict:>16}")
    print("============================================================================")
    os.makedirs(os.path.join(a.outdir, "results"), exist_ok=True)
    json.dump(variants, open(os.path.join(a.outdir, "results", "exp5_adaptive.json"), "w"), indent=2)
    print("Saved -> results/exp5_adaptive.json")


if __name__ == "__main__":
    main()
