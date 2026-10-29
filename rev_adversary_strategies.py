"""
Revision experiments for IEEE Access reviewer comment R2.5 (and the detection-delay part of R2.9).

Reviewer 2 observes that the positive reinforcement-learning result rests on a single
synthetic adversary: a reactive evader that mimics normal traffic after being throttled or
blocked. A result that holds for one adversary is weaker than one that survives several, so
this script re-runs the response game against three further strategies and across a wider
range of response-cost structures.

The three additional strategies are genuine variations, not cosmetic ones. Each overrides when
the campaign is ACTIVE, which in the base simulator controls both the anomaly signature and the
progression of compromise, so a quiet period means the attack is genuinely paused rather than
merely invisible:

  scheduled dormancy  - the campaign goes quiet on a fixed duty cycle regardless of what the
                        defender does, breaking the reactive coupling the base adversary relies
                        on. A memoryless threshold can no longer be provoked into revealing it.
  probe then strike   - a reconnaissance phase with a signature halfway between normal and
                        attack and no progression, followed by a full-rate campaign. This tests
                        whether a policy learns to act on a weak early signal.
  escalating evasion  - the evasion probability rises each time the defender responds, so the
                        adversary adapts to the defender rather than reacting identically.

Detection delay (R2.9) is measured here rather than on the static detectors because delay needs
a time ordering, which only the sequential task has: for each attack episode it is the number of
steps from campaign onset to the defender's first throttle or block. Episodes where the defender
never responds are reported separately as censored rather than averaged in, which would
understate the delay.

Reuses exp5_adaptive_adversary unchanged for the simulator, baseline tuning, rollout and
statistics, so the reactive row here is directly comparable to Table XX of the manuscript.
"""
import os, sys, json, csv, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase1_autoencoder_ids as p1
import phase3_drift as p3
import exp_harness as H
import exp5_adaptive_adversary as X


class DormantSim(X.AdaptiveSim):
    """Quiet on a schedule, not in response to the defender."""
    def __init__(s, ne, ae, duty=(3, 2), **kw):
        super().__init__(ne, ae, **kw); s.duty = duty

    def _active(s):
        if not (s.attack and s.t >= s.start):
            return False
        on, off = s.duty
        return ((s.t - s.start) % (on + off)) < on


class ProbeStrikeSim(X.AdaptiveSim):
    """Weak-signature reconnaissance with no progression, then a full-rate campaign."""
    def __init__(s, ne, ae, probe=8, **kw):
        super().__init__(ne, ae, **kw); s.probe = probe

    def _active(s):
        return s.attack and s.t >= s.start + s.probe

    def _err(s):
        if s.attack and s.start <= s.t < s.start + s.probe:
            return 0.5 * float(s.ne[s.nidx[s.t]]) + 0.5 * float(s.ae[s.aidx[s.t]])
        return super()._err()


class EscalatingSim(X.AdaptiveSim):
    """Evasion probability rises with every defender response."""
    def __init__(s, ne, ae, step_up=0.15, **kw):
        super().__init__(ne, ae, **kw); s.step_up = step_up; s.esc = 0

    def reset(s, seed):
        s.esc = 0                      # must precede super().reset, which calls _err
        return super().reset(seed)

    def _err(s):
        if s._active():
            eps_eff = min(0.95, s.eps + s.esc * s.step_up)
            if s.cool > 0 and s.evcoin[s.t] < eps_eff:
                return float(s.ne[s.nidx[s.t]])
            return float(s.ae[s.aidx[s.t]])
        return float(s.ne[s.nidx[s.t]])

    def step(s, a):
        if s._active() and int(a) in (1, 2):
            s.esc += 1
        return super().step(a)


def rollout_with_delay(policy, sim, seeds):
    """Mean episode reward, plus detection delay on attack episodes.
    Delay = steps from campaign onset to the first throttle or block. Episodes with no
    response at all are counted as censored and excluded from the mean."""
    R, delays, censored, n_attack = [], [], 0, 0
    for sd in seeds:
        o = sim.reset(sd); tot = 0.0; done = False; first = None; t = 0
        is_attack = sim.attack; start = sim.start
        while not done:
            a = int(policy(o))
            if is_attack and first is None and t >= start and a in (1, 2):
                first = t - start
            o, r, done = sim.step(a); tot += r; t += 1
        R.append(tot)
        if is_attack:
            n_attack += 1
            if first is None:
                censored += 1
            else:
                delays.append(first)
    return (np.array(R), float(np.mean(delays)) if delays else float("nan"),
            censored, n_attack)


def run_variant(tag, sim_cls, sim_kwargs, ne, ae, seeds, timesteps, eval_seeds, tune_seeds, w, fh):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    sim = sim_cls(ne, ae, **sim_kwargs)
    base_pol = X.tune_baseline(sim, tune_seeds)
    bR, bdelay, bcens, natt = rollout_with_delay(base_pol, sim, eval_seeds)
    base = bR.mean()

    margins, rl_delays, rl_cens = [], [], []
    for s in seeds:
        H.set_all_seeds(s)

        def _mk():
            import gymnasium as gym
            from gymnasium import spaces

            class Env(gym.Env):
                def __init__(e):
                    super().__init__(); e.sim = sim_cls(ne, ae, **sim_kwargs)
                    e.action_space = spaces.Discrete(3)
                    e.observation_space = spaces.Box(-10, 10, (5,), np.float32)
                    e.rng = np.random.default_rng(s); e.ep = 0

                def reset(e, seed=None, options=None):
                    super().reset(seed=seed); e.ep += 1
                    return e.sim.reset(int(e.rng.integers(1 << 30)) + e.ep), {}

                def step(e, a):
                    o, r, d = e.sim.step(a); return o, r, d, False, {}
            return Env()

        env = DummyVecEnv([_mk])
        model = PPO("MlpPolicy", env, seed=s, verbose=0, n_steps=2048, batch_size=256,
                    device="cpu", policy_kwargs=dict(net_arch=[128, 64, 32]))
        model.learn(total_timesteps=timesteps)
        pol = lambda o: int(model.predict(o, deterministic=True)[0])
        rR, rdelay, rcens, _ = rollout_with_delay(pol, sim, eval_seeds)
        margins.append(rR.mean() - base); rl_delays.append(rdelay); rl_cens.append(rcens)

    st = H.compare_to_constant(np.array(margins) + base, base)
    verdict = "RL wins" if st["ci_lo"] - base > 0 else ("RL loses" if st["ci_hi"] - base < 0 else "tie")
    row = dict(variant=tag, baseline=round(base, 4), margin=round(st["margin"], 4),
               ci_lo=round(st["ci_lo"] - base, 4), ci_hi=round(st["ci_hi"] - base, 4),
               seeds_won=st["seeds_won"], n=st["n"], p_wilcoxon=round(st["p_wilcoxon"], 5),
               cohen_d=round(st["cohen_d"], 3), verdict=verdict,
               baseline_delay=round(bdelay, 3), rl_delay=round(float(np.nanmean(rl_delays)), 3),
               baseline_censored=bcens, rl_censored=int(np.mean(rl_cens)), n_attack_episodes=natt,
               timesteps=timesteps)
    w.writerow(row); fh.flush()
    print(f"  {tag:<26} base {base:+.2f} | margin {row['margin']:+.3f} "
          f"CI[{row['ci_lo']:+.3f},{row['ci_hi']:+.3f}] won {row['seeds_won']}/{row['n']} "
          f"| delay base {row['baseline_delay']:.2f} RL {row['rl_delay']:.2f} | {verdict}")
    return row


FIELDS = ["variant", "baseline", "margin", "ci_lo", "ci_hi", "seeds_won", "n", "p_wilcoxon",
          "cohen_d", "verdict", "baseline_delay", "rl_delay", "baseline_censored",
          "rl_censored", "n_attack_episodes", "timesteps"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--ae-epochs", type=int, default=40)
    ap.add_argument("--timesteps", type=int, default=100000)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--eval-episodes", type=int, default=1500)
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args()

    import torch
    H.set_all_seeds(0); dev = torch.device("cpu")
    trf, trl = p1.load_csv(a.train); pre, _, _ = p1.build_preprocessor(trf)
    Xtr = pre.fit_transform(trf).astype(np.float32)
    ae_m = p3.train_ae(Xtr[trl == 0], Xtr.shape[1], dev, a.ae_epochs, 0)
    _, err = p3.latent_err(ae_m, Xtr, dev)
    emu, esd = float(err[trl == 0].mean()), float(err[trl == 0].std()) + 1e-8
    ne = ((err[trl == 0] - emu) / esd).astype(np.float32)
    ae = ((err[trl == 1] - emu) / esd).astype(np.float32)

    seeds = H.SEEDS[:a.seeds]
    ev = list(range(20000, 20000 + a.eval_episodes)); tu = list(range(30000, 30000 + 600))
    os.makedirs(os.path.join(a.outdir, "results"), exist_ok=True)
    out = os.path.join(a.outdir, "results", "rev_adversary_strategies.csv")

    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); fh.flush()
        print("Adversary strategies (cost_ratio=1, eps=0.75 where evasion applies):")
        run_variant("reactive evasion (paper)", X.AdaptiveSim, dict(eps=0.75, cost_ratio=1.0),
                    ne, ae, seeds, a.timesteps, ev, tu, w, fh)
        run_variant("scheduled dormancy", DormantSim, dict(eps=0.0, cost_ratio=1.0, duty=(3, 2)),
                    ne, ae, seeds, a.timesteps, ev, tu, w, fh)
        run_variant("probe then strike", ProbeStrikeSim, dict(eps=0.0, cost_ratio=1.0, probe=8),
                    ne, ae, seeds, a.timesteps, ev, tu, w, fh)
        run_variant("escalating evasion", EscalatingSim, dict(eps=0.25, cost_ratio=1.0, step_up=0.15),
                    ne, ae, seeds, a.timesteps, ev, tu, w, fh)
        print("Response-cost structure, extending the published range (reactive evader, eps=0.2):")
        for cr in (0.125, 8.0):
            run_variant(f"cost_ratio={cr}", X.AdaptiveSim, dict(eps=0.2, cost_ratio=cr),
                        ne, ae, seeds, a.timesteps, ev, tu, w, fh)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
