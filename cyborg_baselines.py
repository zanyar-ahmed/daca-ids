"""CAGE Challenge 2 baselines, under the challenge's own evaluation protocol.

Evaluates the shipped heuristic Blue agents and a tuned scalar-threshold rule against
both Red agents at episode lengths 30, 50 and 100. The scalar-threshold rule is the
comparator this paper's criterion is about: it collapses the 52-dimensional observation
to one number and acts when it crosses a cut-point, with the cut-point tuned by sweep.

Writes cyborg_baselines.json.
"""
import inspect, json, os, sys
import numpy as np

from CybORG import CybORG
from CybORG.Agents import (B_lineAgent, RedMeanderAgent, SleepAgent,
                           BlueMonitorAgent, BlueReactRemoveAgent, BlueReactRestoreAgent)
from CybORG.Agents.Wrappers import ChallengeWrapper

PATH = str(inspect.getfile(CybORG))[:-10] + "/Shared/Scenarios/Scenario2.yaml"
REDS = {"B_line": B_lineAgent, "Meander": RedMeanderAgent}
LENGTHS = [30, 50, 100]
N_EPISODES = 100


def eval_agent(act_fn, Red, steps, n_eps, seed=0):
    """Mean episode reward for a callable act_fn(obs) -> action."""
    rng = np.random.default_rng(seed)
    cyborg = CybORG(PATH, "sim", agents={"Red": Red})
    env = ChallengeWrapper(env=cyborg, agent_name="Blue")
    totals = []
    for _ in range(n_eps):
        obs = env.reset()
        tot = 0.0
        for _ in range(steps):
            obs, r, done, _ = env.step(act_fn(obs, rng))
            tot += r
            if done:
                break
        totals.append(tot)
    a = np.asarray(totals, dtype=float)
    return float(a.mean()), float(a.std(ddof=1))


def wrapped_agent_fn(AgentCls, Red, steps, n_eps, seed=0):
    """Shipped Blue agents act on the raw CybORG observation, so they get their own loop."""
    cyborg = CybORG(PATH, "sim", agents={"Red": Red, "Blue": AgentCls})
    totals = []
    for _ in range(n_eps):
        cyborg.reset()
        tot = 0.0
        for _ in range(steps):
            cyborg.step()
            tot += cyborg.get_rewards()["Blue"]
        totals.append(tot)
    a = np.asarray(totals, dtype=float)
    return float(a.mean()), float(a.std(ddof=1))


def scalar_score(obs):
    """Collapse the 52-dim observation to one 'alert intensity' number."""
    return float(np.asarray(obs, dtype=float).sum())


def make_threshold_fn(tau, hot_action, cold_action=0):
    def f(obs, rng):
        return hot_action if scalar_score(obs) > tau else cold_action
    return f


if __name__ == "__main__":
    out = {"protocol": {"episodes": N_EPISODES, "lengths": LENGTHS,
                        "scenario": "Scenario2", "reds": list(REDS)},
           "heuristics": {}, "threshold_sweep": {}, "random": {}}

    for red_name, Red in REDS.items():
        for AgentCls in [SleepAgent, BlueMonitorAgent, BlueReactRemoveAgent, BlueReactRestoreAgent]:
            for steps in LENGTHS:
                m, s = wrapped_agent_fn(AgentCls, Red, steps, N_EPISODES)
                out["heuristics"].setdefault(AgentCls.__name__, {})[f"{red_name}_{steps}"] = [round(m, 2), round(s, 2)]
                print(f"  {AgentCls.__name__:<24} {red_name:<8} {steps:>3} steps  {m:8.2f} ± {s:5.2f}", flush=True)

    # Tuned scalar-threshold rule: sweep the cut-point, and sweep which action to fire.
    cyborg = CybORG(PATH, "sim", agents={"Red": B_lineAgent})
    env = ChallengeWrapper(env=cyborg, agent_name="Blue")
    n_actions = env.action_space.n
    for red_name, Red in REDS.items():
        best = None
        for tau in [0.0, 1.0, 2.0, 4.0, 8.0, 16.0]:
            for hot in [1, 2, 3, 4]:
                m, s = eval_agent(make_threshold_fn(tau, hot), Red, 30, 20)
                if best is None or m > best[0]:
                    best = (m, s, tau, hot)
        m30, s30, tau, hot = best
        out["threshold_sweep"][red_name] = {"best_tau": tau, "best_action": hot, "tuned_on": "30 steps, 20 eps"}
        for steps in LENGTHS:
            m, s = eval_agent(make_threshold_fn(tau, hot), Red, steps, N_EPISODES)
            out["threshold_sweep"][red_name][f"{steps}"] = [round(m, 2), round(s, 2)]
            print(f"  tuned threshold (tau={tau}, a={hot})  {red_name:<8} {steps:>3} steps  {m:8.2f} ± {s:5.2f}", flush=True)

    for red_name, Red in REDS.items():
        for steps in LENGTHS:
            m, s = eval_agent(lambda o, rng, n=n_actions: int(rng.integers(n)), Red, steps, N_EPISODES)
            out["random"][f"{red_name}_{steps}"] = [round(m, 2), round(s, 2)]

    json.dump(out, open("cyborg_baselines.json", "w"), indent=1)
    print("\nwrote cyborg_baselines.json")
