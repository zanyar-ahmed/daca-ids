"""
Phase 9 — Adaptive Intrusion-Defense Game (where RL is theoretically favoured)
==============================================================================
A repeated security game with an ADAPTIVE attacker. K channels (attack families) of different
VALUE. Each step the defender protects ONE channel; a learning attacker (no-regret / reinforce)
shifts toward whatever is least defended. A FIXED defence is exploitable (the attacker finds the
gap); a naive random defence ignores channel value; a myopic "defend-most-attacked" defence is
one step behind. An RL defender can be value-aware AND anticipate the attacker's shifts.

This is the principled RL niche: against an adaptive adversary, a static/myopic policy is
exploitable, an adaptive policy is not. We test whether RL actually realises that edge — with
multiple seeds and 95% confidence intervals (rigorous from the start).

Baselines (same adaptive attacker, common seeds): fixed-highest-value, uniform-random,
value-weighted-random, and defend-most-recently-attacked (myopic adaptive). RL must beat the best.
Metric: mean defender reward per episode (= −total damage); higher is better.

RUN (Colab):
  !python phase9_advgame.py --timesteps 300000 --seeds 10
Outputs: phase9_metrics.json , phase9_advgame.png
"""
import argparse, importlib, json, os, subprocess, sys
import numpy as np

def _ensure(mod, pip_name=None):
    try: importlib.import_module(mod)
    except Exception:
        print(f"installing {pip_name or mod} ..."); subprocess.run([sys.executable,"-m","pip","install","-q",pip_name or mod], check=True)

import phase1_autoencoder_ids as p1   # only for set_seed

# channel = attack family; higher value = more damaging if breached (e.g. privilege escalation)
VALUES   = np.array([1.0, 1.0, 2.0, 3.0])      # DoS, Probe, R2L, U2R (illustrative severities)
P_DEF    = np.array([0.95, 0.95, 0.95, 0.95])  # catch prob if this channel IS defended & attacked
P_UNDEF  = np.array([0.35, 0.35, 0.20, 0.10])  # catch prob if attacked while NOT defended
K = len(VALUES)


def softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()


def mean_ci(xs):
    xs = np.asarray(xs, float); n = len(xs); m = float(xs.mean())
    if n < 2: return m, 0.0
    s = float(xs.std(ddof=1))
    try:
        from scipy import stats; t = float(stats.t.ppf(0.975, n - 1))
    except Exception:
        t = 1.96
    return m, t * s / np.sqrt(n)


class AdaptiveAttacker:
    """No-regret style learner: reinforces channels that yielded damage; concentrates on the
    defender's weak, high-value spots. Slow lr -> exploitable structure the defender can learn."""
    def __init__(self, lr, beta, seed):
        self.lr, self.beta = lr, beta
        self.w = np.zeros(K); self.rng = np.random.default_rng(seed); self.a = 0
    def act(self):
        self.a = int(self.rng.choice(K, p=softmax(self.beta * self.w))); return self.a
    def update(self, damage):
        self.w[self.a] += self.lr * damage


class Game:
    def __init__(self, T, atk_lr, atk_beta):
        self.T, self.atk_lr, self.atk_beta = T, atk_lr, atk_beta
    def reset(self, seed):
        self.rng = np.random.default_rng(seed)
        self.att = AdaptiveAttacker(self.atk_lr, self.atk_beta, seed + 1)
        self.t = 0; self.hist = np.zeros(K); self.lastdef = np.zeros(K); self.dmg = 0.0
        return self._state()
    def _state(self):
        h = self.hist / max(self.hist.sum(), 1.0)
        return np.concatenate([h, self.lastdef]).astype(np.float32)
    def step(self, d):
        d = int(d); a = self.att.act()
        pc = P_DEF[a] if a == d else P_UNDEF[a]
        caught = self.rng.random() < pc
        dmg = 0.0 if caught else float(VALUES[a])
        self.att.update(dmg)
        self.hist[a] += 1; self.lastdef = np.eye(K)[d]; self.dmg += dmg
        self.t += 1
        return self._state(), -dmg, self.t >= self.T


def make_env(T, atk_lr, atk_beta, seed):
    import gymnasium as gym
    from gymnasium import spaces
    class Env(gym.Env):
        def __init__(e):
            super().__init__(); e.g = Game(T, atk_lr, atk_beta)
            e.action_space = spaces.Discrete(K)
            e.observation_space = spaces.Box(0, 1, (2 * K,), np.float32)
            e.rng = np.random.default_rng(seed); e.ep = 0
        def reset(e, seed=None, options=None):
            super().reset(seed=seed); e.ep += 1
            return e.g.reset(int(e.rng.integers(1 << 30)) + e.ep), {}
        def step(e, a):
            s, r, d = e.g.step(a); return s, r, d, False, {}
    return Env


def rollout(policy, T, atk_lr, atk_beta, seeds):
    """Mean episode reward for a defender policy(state)->action over fixed episode seeds."""
    g = Game(T, atk_lr, atk_beta); rew = []
    for sd in seeds:
        s = g.reset(sd); tot = 0.0; done = False
        while not done:
            s, r, done = g.step(policy(s)); tot += r
        rew.append(tot)
    return np.array(rew)


# ---- baseline defender policies (state = [hist(K), lastdef(K)]) ----
def pol_fixed_value():            return lambda s: int(np.argmax(VALUES))
def pol_uniform(seed):
    rng = np.random.default_rng(seed); return lambda s: int(rng.integers(K))
def pol_value_random(seed):
    rng = np.random.default_rng(seed); p = VALUES / VALUES.sum()
    return lambda s: int(rng.choice(K, p=p))
def pol_freq_follow():            return lambda s: int(np.argmax(s[:K]))           # most-attacked recently
def pol_value_freq():             return lambda s: int(np.argmax(s[:K] * VALUES))  # value-weighted frequency


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=300000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--episode-T", type=int, default=100)
    ap.add_argument("--atk-lr", type=float, default=0.15)
    ap.add_argument("--atk-beta", type=float, default=1.0)
    ap.add_argument("--eval-episodes", type=int, default=400)
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args()
    _ensure("gymnasium"); _ensure("stable_baselines3", "stable-baselines3")
    import torch
    torch.backends.cudnn.deterministic = True
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    T = a.episode_T
    ev_seeds = list(range(10_000, 10_000 + a.eval_episodes))   # fixed eval episodes (common RNG)

    # baselines (deterministic given seeds)
    base = {
        "Fixed (highest value)": pol_fixed_value(),
        "Uniform random": pol_uniform(1),
        "Value-weighted random": pol_value_random(2),
        "Defend most-attacked": pol_freq_follow(),
        "Value x frequency": pol_value_freq(),
    }
    base_scores = {n: rollout(p, T, a.atk_lr, a.atk_beta, ev_seeds) for n, p in base.items()}

    # RL defender over several training seeds (rigorous)
    rl_means = []
    train_seeds = [42, 7, 2024, 1, 99, 123, 2025, 5, 77, 314][:a.seeds]
    print(f"Training {len(train_seeds)} RL defenders ({a.timesteps:,} steps each) ...")
    rl_curves = []
    for i, sd in enumerate(train_seeds):
        p1.set_seed(sd)
        Env = make_env(T, a.atk_lr, a.atk_beta, sd)
        env = DummyVecEnv([lambda: Env()])
        model = PPO("MlpPolicy", env, seed=sd, verbose=0, n_steps=2048, batch_size=256,
                    device="cpu", policy_kwargs=dict(net_arch=[128, 64, 32]))
        model.learn(total_timesteps=a.timesteps)
        sc = rollout(lambda s: int(model.predict(s, deterministic=True)[0]), T, a.atk_lr, a.atk_beta, ev_seeds)
        rl_means.append(float(sc.mean())); rl_curves.append(sc)
        print(f"  RL seed {sd:<5} ({i+1}/{len(train_seeds)})  mean reward {sc.mean():+.3f}")

    rl_m, rl_h = mean_ci(rl_means)
    print("\n==============  ADVERSARIAL GAME RESULTS (mean episode reward, higher=better)  ==============")
    rows = {}
    for n, sc in base_scores.items():
        m, h = mean_ci(sc); rows[n] = (m, h)
        print(f"  {n:<26} {m:+.3f}")
    print(f"  {'RL defender (learned)':<26} {rl_m:+.3f}  95% CI [{rl_m-rl_h:+.3f}, {rl_m+rl_h:+.3f}]  (over {len(train_seeds)} seeds)")
    best_base_name = max(rows, key=lambda k: rows[k][0]); best_base = rows[best_base_name][0]
    win = (rl_m - rl_h) > best_base                      # RL CI lower bound beats best baseline mean
    print(f"\n  Best baseline: {best_base_name} ({best_base:+.3f})")
    print(f"  Verdict: {'RL SIGNIFICANTLY WINS the adversarial game ✅' if win else 'RL does not significantly beat the best baseline'}"
          f"  (RL {rl_m:+.3f} vs {best_base:+.3f})")
    print("===========================================================================================")

    json.dump(dict(seeds=train_seeds, timesteps=a.timesteps,
                   baselines={n: dict(mean=rows[n][0], ci=rows[n][1]) for n in rows},
                   rl=dict(mean=rl_m, ci=rl_h), best_baseline=best_base_name, win=bool(win)),
              open(os.path.join(a.outdir, "phase9_metrics.json"), "w"), indent=2)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        names = list(rows) + ["RL defender"]; means = [rows[n][0] for n in rows] + [rl_m]
        errs = [rows[n][1] for n in rows] + [rl_h]
        cols = ["tab:gray"] * len(rows) + ["tab:green"]
        plt.figure(figsize=(9, 5)); plt.barh(names, means, xerr=errs, color=cols, capsize=5)
        plt.xlabel("mean episode reward (higher = less damage)"); plt.title("Adaptive intrusion-defense game: RL vs baselines")
        plt.tight_layout(); png = os.path.join(a.outdir, "phase9_advgame.png"); plt.savefig(png, dpi=130)
        print(f"Saved plot -> {png}")
    except Exception as ex:
        print(f"(plot skipped: {ex})")
    print("Saved -> phase9_metrics.json")


if __name__ == "__main__":
    main()
