"""
Phase 4 — Budget experiment (DACA-IDS: where RL is genuinely irreplaceable)
============================================================================
A finite analyst alert budget over a one-pass stream whose ATTACK DENSITY RISES over time
(an attack campaign ramping up). You may raise at most K alerts total; K < number of attacks,
so you MUST prioritise. The decision is online (no peeking ahead): "spend an alert now, or save
budget for the attack-dense part later?" A fixed threshold cannot reason about remaining budget
or where it is in the stream — an RL agent that sees [remaining budget, stream progress] can.

Compared (same stream, same budget K):
  A0  Fixed threshold + budget (online, greedy until budget runs out) -- best of several cuts
  A_RL  PPO with state [z, e, remaining-budget, stream-progress]      -- budget/context aware

Headline metric: ATTACKS CAUGHT for the same budget (also precision/recall/F1 of the alerts).
If the agent catches more attacks per budget, RL is doing something a threshold cannot.
Reproducible (fixed seed); nothing hand-typed.

RUN (Colab):
  !python phase4_budget.py --ae-epochs 30 --timesteps 150000 --budget-frac 0.15
Outputs: phase4_budget.png , phase4_metrics.json
"""
import argparse, importlib, json, os, subprocess, sys
import numpy as np


def _ensure(mod, pip_name=None):
    try:
        importlib.import_module(mod)
    except Exception:
        print(f"installing {pip_name or mod} ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name or mod], check=True)


import phase1_autoencoder_ids as p1
import phase3_drift as p3      # reuse train_ae(), latent_err()


def attack_prob(t, L, base=0.08, peak=0.65):
    """Rising attack density along the stream (campaign ramp-up)."""
    return base + (peak - base) * (t / max(L - 1, 1))


def build_stream(z, e, y, L, seed, base=0.08, peak=0.65):
    """Construct a length-L stream whose attack probability ramps from `base` to `peak`."""
    rng = np.random.default_rng(seed)
    atk = np.where(y == 1)[0]
    nrm = np.where(y == 0)[0]
    zs, es, ys = [], [], []
    for t in range(L):
        is_atk = rng.random() < attack_prob(t, L, base, peak)
        pool = atk if (is_atk and len(atk)) else nrm
        i = pool[rng.integers(len(pool))]
        zs.append(z[i]); es.append(e[i]); ys.append(int(y[i]))
    return np.array(zs, np.float32), np.array(es, np.float32), np.array(ys, np.int64)


def make_env(zmu, zsd, emu, esd):
    import gymnasium as gym
    from gymnasium import spaces

    class BudgetEnv(gym.Env):
        def __init__(s, z, e, y, L, budget_frac, seed):
            super().__init__()
            s.z, s.e, s.y = z, e, y          # full train pools
            s.L = L; s.bf = budget_frac
            s.rng_seed = seed; s.ep = 0
            s.action_space = spaces.Discrete(2)        # 0 ignore, 1 escalate(=alert)
            s.observation_space = spaces.Box(-10, 10, (z.shape[1] + 3,), np.float32)

        def _obs(s, t):
            zt = np.clip((s.sz[t] - zmu) / zsd, -10, 10)
            et = np.clip((s.se[t] - emu) / esd, -10, 10)
            return np.concatenate([zt, [et, s.rem / max(s.budget, 1), t / s.L]]).astype(np.float32)

        def reset(s, seed=None, options=None):
            super().reset(seed=seed)
            s.ep += 1
            s.sz, s.se, s.sy = build_stream(s.z, s.e, s.y, s.L, s.rng_seed + s.ep)
            s.budget = max(1, int(s.bf * s.L)); s.rem = s.budget; s.t = 0
            return s._obs(0), {}

        def step(s, a):
            y = s.sy[s.t]; escalated = (int(a) == 1 and s.rem > 0)
            if escalated:
                s.rem -= 1
                r = 2.0 if y == 1 else -1.0           # caught attack vs wasted alert
            else:
                r = -3.0 if y == 1 else 0.1           # missed attack vs correctly ignored
            s.t += 1; done = s.t >= s.L
            return s._obs(min(s.t, s.L - 1)), r, done, False, {}
    return BudgetEnv


def run_fixed(e_stream, y_stream, thr, budget):
    """Online fixed-threshold policy under a global budget."""
    rem = budget; pred = np.zeros(len(y_stream), int)
    for t in range(len(y_stream)):
        if rem > 0 and e_stream[t] > thr:
            pred[t] = 1; rem -= 1
    return pred


def run_rl(model, sz, se, sy, zmu, zsd, emu, esd, budget):
    """Online RL policy under a global budget (deterministic)."""
    rem = budget; pred = np.zeros(len(sy), int); L = len(sy)
    for t in range(L):
        zt = np.clip((sz[t] - zmu) / zsd, -10, 10)
        et = np.clip((se[t] - emu) / esd, -10, 10)
        obs = np.concatenate([zt, [et, rem / max(budget, 1), t / L]]).astype(np.float32)
        a, _ = model.predict(obs, deterministic=True)
        if int(a) == 1 and rem > 0:
            pred[t] = 1; rem -= 1
    return pred


def score(pred, y):
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    alerts = tp + fp; attacks = int((y == 1).sum())
    prec = tp / max(alerts, 1); rec = tp / max(attacks, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return dict(attacks_caught=tp, alerts_spent=alerts, total_attacks=attacks,
                precision=prec, recall=rec, f1=f1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="/content/drive/MyDrive/dataset/Train_data.csv")
    ap.add_argument("--test", default="/content/drive/MyDrive/dataset/Test_data.csv")
    ap.add_argument("--ae-epochs", type=int, default=100)
    ap.add_argument("--timesteps", type=int, default=200000)
    ap.add_argument("--episode-len", type=int, default=2000)
    ap.add_argument("--stream-len", type=int, default=20000)
    ap.add_argument("--budget-frac", type=float, default=0.15)
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

    # data + frozen AE
    trf, trl = p1.load_csv(a.train); tef, tel = p1.load_csv(a.test)
    pre, _, _ = p1.build_preprocessor(trf)
    Xtr = pre.fit_transform(trf).astype(np.float32); Xte = pre.transform(tef).astype(np.float32)
    dim = Xtr.shape[1]; print(f"input_dim={dim}\nTraining frozen autoencoder ...")
    ae = p3.train_ae(Xtr[trl == 0], dim, dev, a.ae_epochs, a.seed)

    ztr, etr = p3.latent_err(ae, Xtr, dev)
    zte, ete = p3.latent_err(ae, Xte, dev)
    zmu, zsd = ztr.mean(0), ztr.std(0) + 1e-8
    emu, esd = float(etr.mean()), float(etr.std()) + 1e-8

    # ---- train budget-aware PPO ----
    print(f"\nTraining budget-aware PPO ({a.timesteps:,} steps) ...")
    Env = make_env(zmu, zsd, emu, esd)
    env = DummyVecEnv([lambda: Env(ztr, etr, trl, a.episode_len, a.budget_frac, a.seed)])
    model = PPO("MlpPolicy", env, seed=a.seed, verbose=0, n_steps=a.episode_len,
                device="cpu", policy_kwargs=dict(net_arch=[128, 64, 32]))
    model.learn(total_timesteps=a.timesteps); print("PPO done.")

    # ---- one fixed TEST stream (rising attack density), shared by all policies ----
    sz, se, sy = build_stream(zte, ete, tel, a.stream_len, a.seed + 999)
    budget = max(1, int(a.budget_frac * a.stream_len))
    print(f"\nTest stream: {a.stream_len} flows, {int((sy==1).sum())} attacks, budget {budget} alerts")

    # A0: best fixed threshold (try several, report the best by attacks caught)
    best = None
    for p in (80, 85, 90, 95, 98):
        thr = float(np.percentile(etr[trl == 0], p))
        m = score(run_fixed(se, sy, thr, budget), sy); m["pctile"] = p
        if best is None or m["attacks_caught"] > best["attacks_caught"]:
            best = m
    # A_RL
    rl = score(run_rl(model, sz, se, sy, zmu, zsd, emu, esd, budget), sy)

    print("\n========= BUDGET-CONSTRAINED RESULTS (same budget) =========")
    print(f"{'policy':<26}{'caught':>8}{'spent':>8}{'recall':>8}{'prec':>8}{'f1':>8}")
    print(f"{'A0 fixed thr (P%d)'%best['pctile']:<26}{best['attacks_caught']:>8}{best['alerts_spent']:>8}"
          f"{best['recall']:>8.3f}{best['precision']:>8.3f}{best['f1']:>8.3f}")
    print(f"{'A_RL budget-aware':<26}{rl['attacks_caught']:>8}{rl['alerts_spent']:>8}"
          f"{rl['recall']:>8.3f}{rl['precision']:>8.3f}{rl['f1']:>8.3f}")
    gain = rl["attacks_caught"] - best["attacks_caught"]
    print(f"\nVerdict: {'RL CATCHES MORE attacks per budget ✅ (+%d)' % gain if gain > 0 else 'RL does NOT beat fixed threshold (%+d) — tune budget/timesteps or reframe' % gain}")
    print("============================================================")

    # ---- plot: cumulative attacks caught vs alerts spent ----
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        thr_best = float(np.percentile(etr[trl == 0], best["pctile"]))
        pf = run_fixed(se, sy, thr_best, budget)
        pr = run_rl(model, sz, se, sy, zmu, zsd, emu, esd, budget)
        plt.figure(figsize=(8, 5))
        plt.plot(np.cumsum((pf == 1) & (sy == 0)), np.cumsum((pf == 1) & (sy == 1)), label=f"A0 fixed thr (P{best['pctile']})")
        plt.plot(np.cumsum((pr == 1) & (sy == 0)), np.cumsum((pr == 1) & (sy == 1)), label="A_RL budget-aware")
        plt.xlabel("false alarms spent"); plt.ylabel("attacks caught")
        plt.title(f"Attacks caught vs budget ({budget} alerts, rising attack density)")
        plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
        png = os.path.join(a.outdir, "phase4_budget.png"); plt.savefig(png, dpi=130)
        print(f"Saved plot -> {png}")
    except Exception as ex:
        print(f"(plot skipped: {ex})")

    json.dump({"seed": a.seed, "budget": budget, "stream_len": a.stream_len,
               "budget_frac": a.budget_frac, "A0_fixed": best, "A_RL": rl, "rl_gain_attacks": gain},
              open(os.path.join(a.outdir, "phase4_metrics.json"), "w"), indent=2)
    print("Saved -> phase4_metrics.json")


if __name__ == "__main__":
    main()
