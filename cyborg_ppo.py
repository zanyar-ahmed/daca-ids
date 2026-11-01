"""Train PPO on CAGE Challenge 2 and evaluate it under the challenge's own protocol.

The paper's criterion predicts reinforcement learning should win here: the observation is
52-dimensional, the action set has 145 members, the Red agent adapts, and cost is delayed,
so assumptions (A1) scalar sufficiency and (A4) stationarity both fail.

Writes cyborg_ppo.json.
"""
import argparse, inspect, json, os, sys, time
import numpy as np
import gymnasium as gymn
from gymnasium import spaces

from CybORG import CybORG
from CybORG.Agents import B_lineAgent, RedMeanderAgent
from CybORG.Agents.Wrappers import ChallengeWrapper

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

PATH = str(inspect.getfile(CybORG))[:-10] + "/Shared/Scenarios/Scenario2.yaml"
REDS = {"B_line": B_lineAgent, "Meander": RedMeanderAgent}


class Gymnasium(gymn.Env):
    """Old-gym CybORG wrapper presented through the gymnasium API stable-baselines3 expects."""

    def __init__(self, Red, max_steps=30):
        cyborg = CybORG(PATH, "sim", agents={"Red": Red})
        self.env = ChallengeWrapper(env=cyborg, agent_name="Blue")
        self.max_steps = max_steps
        self.observation_space = spaces.Box(-1.0, 3.0, (52,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.env.action_space.n)
        self._t = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        return np.asarray(self.env.reset(), dtype=np.float32), {}

    def step(self, action):
        obs, r, done, info = self.env.step(int(action))
        self._t += 1
        truncated = self._t >= self.max_steps
        return (np.asarray(obs, dtype=np.float32), float(r),
                bool(done), bool(truncated), info if isinstance(info, dict) else {})


def evaluate(model, Red, steps, n_eps):
    e = Gymnasium(Red, max_steps=steps)
    totals = []
    for _ in range(n_eps):
        obs, _ = e.reset()
        tot = 0.0
        for _ in range(steps):
            a, _ = model.predict(obs, deterministic=True)
            obs, r, done, trunc, _ = e.step(a)
            tot += r
            if done or trunc:
                break
        totals.append(tot)
    a = np.asarray(totals, dtype=float)
    return float(a.mean()), float(a.std(ddof=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=300000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--train-steps", type=int, default=30, help="episode length during training")
    ap.add_argument("--eval-eps", type=int, default=100)
    ap.add_argument("--out", default="cyborg_ppo.json")
    a = ap.parse_args()

    out = {"config": {"timesteps": a.timesteps, "seeds": a.seeds,
                      "train_episode_len": a.train_steps, "eval_episodes": a.eval_eps,
                      "scenario": "Scenario2"}, "results": {}}

    for red_name, Red in REDS.items():
        for seed in range(a.seeds):
            t0 = time.perf_counter()
            env = DummyVecEnv([lambda R=Red: Gymnasium(R, max_steps=a.train_steps)])
            model = PPO("MlpPolicy", env, seed=seed, verbose=0, n_steps=2048,
                        batch_size=256, device="cpu",
                        policy_kwargs=dict(net_arch=[128, 64, 32]))
            model.learn(total_timesteps=a.timesteps)
            train_s = time.perf_counter() - t0
            for steps in [30, 50, 100]:
                m, s = evaluate(model, Red, steps, a.eval_eps)
                out["results"].setdefault(f"{red_name}_{steps}", []).append(round(m, 2))
                print(f"  PPO seed {seed}  {red_name:<8} {steps:>3} steps  {m:8.2f} ± {s:5.2f}"
                      f"   (train {train_s:.0f}s)", flush=True)
            json.dump(out, open(a.out, "w"), indent=1)

    print("\nwrote", a.out)
