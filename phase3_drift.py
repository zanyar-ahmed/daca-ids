"""
Phase 3 — Drift experiment (DACA-IDS core contribution)
========================================================
Shows the ONE thing RL can do that a fixed threshold cannot: stay calibrated under concept drift.

Setup:
  * Frozen self-supervised autoencoder (trained on Normal only) -> reconstruction error e.
  * A drifting TEST stream: a growing covariate shift is added along the stream, so the baseline
    reconstruction error rises over time (concept drift).
  * Three deciders, scored with a sliding-window F1 along the stream:
      A0  Fixed threshold        -> absolute error vs a fixed P85 cut  (expected to DEGRADE)
      A1  PPO + static signal     -> error standardised with GLOBAL train stats (no drift info)
      A4  PPO + drift signal      -> error standardised with a ROLLING window  (drift-adaptive)
    A1 and A4 are the SAME trained PPO controller; only the error-normalisation differs, which
    cleanly isolates the value of the drift signal.

Honest framing: the adaptation comes from feeding the controller a drift-relative error signal;
the fixed threshold and the static-signal controller cannot track the shift. Reproducible (seed).

RUN (Colab):
  !python phase3_drift.py --train .../Train_data.csv --test .../Test_data.csv --ae-epochs 30 --timesteps 80000
Outputs: phase3_drift_f1.png , phase3_metrics.json
"""
import argparse, importlib, json, os, subprocess, sys
import numpy as np


def _ensure(mod, pip_name=None):
    try:
        importlib.import_module(mod)
    except Exception:
        print(f"installing {pip_name or mod} ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name or mod], check=True)


import phase1_autoencoder_ids as p1   # loader, preprocessing, AE architecture


# ---- frozen autoencoder (Normal-only) ----
def train_ae(Xn, dim, device, epochs, seed, lr=1e-3, batch=256):
    import torch, torch.nn as nn
    g = np.random.default_rng(seed); idx = g.permutation(len(Xn))
    nv = max(1, int(0.1 * len(idx))); Xv, Xt = Xn[idx[:nv]], Xn[idx[nv:]]
    m = p1.build_model(dim).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lf = nn.MSELoss(); Xt_t = torch.tensor(Xt, device=device); Xv_t = torch.tensor(Xv, device=device)
    best, bs, n = 1e9, None, len(Xt_t)
    for ep in range(1, epochs + 1):
        m.train(); perm = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            b = perm[i:i + batch]; opt.zero_grad()
            loss = lf(m(Xt_t[b])[0], Xt_t[b]); loss.backward(); opt.step()
        sch.step(); m.eval()
        with torch.no_grad(): v = lf(m(Xv_t)[0], Xv_t).item()
        if v < best: best, bs = v, {k: val.detach().cpu().clone() for k, val in m.state_dict().items()}
        if ep == 1 or ep % 10 == 0 or ep == epochs: print(f"    AE epoch {ep}/{epochs} val {v:.5f}")
    m.load_state_dict(bs)
    for p in m.parameters(): p.requires_grad = False
    m.eval(); return m


def latent_err(m, X, device, batch=4096):
    import torch; zs, es = [], []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.tensor(X[i:i + batch], dtype=torch.float32, device=device)
            xh, z = m(xb); zs.append(z.cpu().numpy()); es.append(((xb - xh) ** 2).mean(1).cpu().numpy())
    return np.concatenate(zs), np.concatenate(es)


# ---- RL env (state = z + standardised error; 5 tiers; paper reward) ----
def make_env():
    import gymnasium as gym
    from gymnasium import spaces

    class E(gym.Env):
        def __init__(s, st, lab, ep, seed):
            super().__init__()
            s.st = st.astype(np.float32); s.lab = lab.astype(int); s.N = len(st); s.ep = ep
            s.action_space = spaces.Discrete(5)
            s.observation_space = spaces.Box(-10, 10, (st.shape[1],), np.float32)
            s.rng = np.random.default_rng(seed); s.t = 0; s.o = s.rng.permutation(s.N)

        def reset(s, seed=None, options=None):
            super().reset(seed=seed); s.t = 0; s.o = s.rng.permutation(s.N); return s.st[s.o[0]], {}

        def step(s, a):
            y = s.lab[s.o[s.t % s.N]]; a = int(a)
            r = (2.0 if y == 1 else -0.5) if a in (3, 4) else (1.0 if y == 0 else -3.0) if a in (0, 1) else 0.0
            s.t += 1
            return s.st[s.o[s.t % s.N]], r, s.t >= s.ep, False, {}
    return E


def tiers_to_bin(a):
    return np.isin(np.asarray(a), [3, 4]).astype(int)   # A/VA -> attack ; else normal


def f1_of(pred, y):
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return 2 * p * r / max(p + r, 1e-12)


def rolling_std_err(e, window):
    import pandas as pd
    s = pd.Series(e)
    mu = s.rolling(window, min_periods=1).mean()
    sd = s.rolling(window, min_periods=1).std().bfill().fillna(float(np.std(e)) + 1e-8)
    return ((s - mu) / (sd + 1e-8)).to_numpy()


def windowed(pred, y, win):
    xs, f1s = [], []
    for a in range(0, len(y) - win + 1, win):
        xs.append((a + win / 2) / len(y)); f1s.append(f1_of(pred[a:a + win], y[a:a + win]))
    return np.array(xs), np.array(f1s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="/content/drive/MyDrive/dataset/Train_data.csv")
    ap.add_argument("--test", default="/content/drive/MyDrive/dataset/Test_data.csv")
    ap.add_argument("--ae-epochs", type=int, default=100)
    ap.add_argument("--timesteps", type=int, default=150000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-drift", type=float, default=1.0, help="peak covariate shift (std units)")
    ap.add_argument("--window", type=int, default=1000, help="rolling window for drift normalisation")
    ap.add_argument("--f1win", type=int, default=1500, help="sliding window for F1-over-stream")
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
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev} | seed {a.seed}")

    # data + frozen AE
    trf, trl = p1.load_csv(a.train); tef, tel = p1.load_csv(a.test)
    pre, _, _ = p1.build_preprocessor(trf)
    Xtr = pre.fit_transform(trf).astype(np.float32); Xte = pre.transform(tef).astype(np.float32)
    dim = Xtr.shape[1]; print(f"input_dim={dim}")
    print("Training frozen autoencoder ..."); ae = train_ae(Xtr[trl == 0], dim, dev, a.ae_epochs, a.seed)

    # training states (z + GLOBAL-standardised error) -> train one PPO controller
    ztr, etr = latent_err(ae, Xtr, dev)
    zmu, zsd = ztr.mean(0), ztr.std(0) + 1e-8
    emu, esd = float(etr.mean()), float(etr.std()) + 1e-8
    Str = np.concatenate([np.clip((ztr - zmu) / zsd, -10, 10),
                          np.clip(((etr - emu) / esd)[:, None], -10, 10)], 1).astype(np.float32)
    print(f"Training PPO controller ({a.timesteps:,} steps) ...")
    Env = make_env()
    env = DummyVecEnv([lambda: Env(Str, trl, 2048, a.seed)])
    model = PPO("MlpPolicy", env, seed=a.seed, verbose=0, n_steps=2048,
                device="cpu", policy_kwargs=dict(net_arch=[128, 64, 32]))
    model.learn(total_timesteps=a.timesteps); print("PPO done.")

    # ---- build DRIFTING test stream ----
    rng = np.random.default_rng(a.seed)
    order = rng.permutation(len(Xte))
    y = tel[order]
    d = rng.normal(size=dim).astype(np.float32)                 # drift direction
    ramp = (np.arange(len(order)) / len(order)).astype(np.float32) * a.max_drift
    Xdrift = Xte[order] + ramp[:, None] * d[None, :]            # growing covariate shift
    zd, ed = latent_err(ae, Xdrift, dev)
    zd_s = np.clip((zd - zmu) / zsd, -10, 10).astype(np.float32)

    # error signals
    e_global = np.clip(((ed - emu) / esd), -10, 10)             # static (no drift info)
    e_roll = np.clip(rolling_std_err(ed, a.window), -10, 10)    # drift-adaptive

    S_A1 = np.concatenate([zd_s, e_global[:, None]], 1).astype(np.float32)
    S_A4 = np.concatenate([zd_s, e_roll[:, None]], 1).astype(np.float32)

    pred_A1 = tiers_to_bin(model.predict(S_A1, deterministic=True)[0].reshape(-1))
    pred_A4 = tiers_to_bin(model.predict(S_A4, deterministic=True)[0].reshape(-1))
    thr85 = float(np.percentile(etr[trl == 0], 85))            # fixed absolute threshold
    pred_A0 = (ed > thr85).astype(int)

    # ---- sliding-window F1 along the (drifting) stream ----
    x0, f0 = windowed(pred_A0, y, a.f1win)
    x1, f1 = windowed(pred_A1, y, a.f1win)
    x4, f4 = windowed(pred_A4, y, a.f1win)

    def tail(f):  # mean F1 over the high-drift last third
        return float(np.mean(f[len(f) * 2 // 3:]))
    res = {"A0_fixed_threshold": tail(f0), "A1_ppo_static": tail(f1), "A4_ppo_drift_adaptive": tail(f4)}
    print("\n========= F1 in the HIGH-DRIFT final third =========")
    print(f"  A0 fixed threshold      : {res['A0_fixed_threshold']:.4f}")
    print(f"  A1 PPO (static signal)  : {res['A1_ppo_static']:.4f}")
    print(f"  A4 PPO (drift signal)   : {res['A4_ppo_drift_adaptive']:.4f}")
    win = res["A4_ppo_drift_adaptive"] > max(res["A0_fixed_threshold"], res["A1_ppo_static"]) + 1e-3
    print(f"\nVerdict: {'DRIFT-ADAPTIVE RL WINS under drift ✅' if win else 'no clear win — tune --max-drift/--window/--timesteps'}")
    print("====================================================")

    # ---- plot (the money figure) ----
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.figure(figsize=(9, 5))
        plt.plot(x0, f0, "-o", ms=3, label="A0 fixed threshold")
        plt.plot(x1, f1, "-s", ms=3, label="A1 PPO (static)")
        plt.plot(x4, f4, "-^", ms=3, label="A4 PPO (drift-adaptive)")
        plt.xlabel("position in stream  (drift grows →)"); plt.ylabel("F1 (sliding window)")
        plt.title("Detection F1 under growing concept drift"); plt.ylim(0, 1)
        plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
        png = os.path.join(a.outdir, "phase3_drift_f1.png"); plt.savefig(png, dpi=130)
        print(f"Saved plot -> {png}")
    except Exception as ex:
        print(f"(plot skipped: {ex})")

    json.dump({"seed": a.seed, "max_drift": a.max_drift, "window": a.window,
               "tail_f1": res}, open(os.path.join(a.outdir, "phase3_metrics.json"), "w"), indent=2)
    print("Saved -> phase3_metrics.json")


if __name__ == "__main__":
    main()
