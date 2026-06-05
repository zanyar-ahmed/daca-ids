"""
Phase 5 — Cost-aware intrusion detection via RL  (the genuine RL contribution)
==============================================================================
RL's job here is NOT "flag or not" (a threshold wins that). It is a SEQUENTIAL decision a
threshold cannot make: per connection, decide HOW MUCH to inspect before classifying.

NSL-KDD features split into 4 real cost groups (standard taxonomy), acquired cheapest-first:
    basic   (header-level)        cost 1
    time    (2-second window)     cost 2
    host    (100-connection win)  cost 2
    content (payload inspection)  cost 4
A classifier (trained to work on ANY acquired prefix) makes the final decision. The PPO agent
chooses, after each group, ACQUIRE-next vs STOP-and-classify, paying the acquisition cost.

Compared on the test set (F1 vs average inspection cost):
    Cheap   : basic group only        (low cost, lower F1)
    Full    : all groups always       (high cost, high F1)
    Cascade : acquire until classifier confidence >= tau  (hand-tuned heuristic, several taus)
    RL      : learned adaptive policy
A genuine win = RL reaches Full-level F1 at much lower average cost (Pareto), beating the
hand-tuned cascade. Reproducible (fixed seed). Nothing hand-typed.

RUN (Colab):
  !python phase5_costaware.py --ae-epochs 0 --clf-epochs 25 --timesteps 200000 --cost-lambda 0.15
Outputs: phase5_pareto.png , phase5_metrics.json
"""
import argparse, importlib, json, os, subprocess, sys
import numpy as np

def _ensure(mod, pip_name=None):
    try: importlib.import_module(mod)
    except Exception:
        print(f"installing {pip_name or mod} ..."); subprocess.run([sys.executable,"-m","pip","install","-q",pip_name or mod], check=True)

import phase1_autoencoder_ids as p1

F = p1.NSL_KDD_FEATURES
GROUP_FEATURES = {                         # acquired cheapest-first
    "basic":   F[0:9],
    "time":    F[22:31],
    "host":    F[31:41],
    "content": F[9:22],
}
GROUP_ORDER = ["basic", "time", "host", "content"]
GROUP_COST  = {"basic": 1.0, "time": 2.0, "host": 2.0, "content": 4.0}


def build_group_blocks(trf, tef):
    """Preprocess each cost group separately so we know which columns belong to which group."""
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    tr_blocks, te_blocks, sizes = [], [], []
    for g in GROUP_ORDER:
        cols = [c for c in GROUP_FEATURES[g] if c in trf.columns]
        cat = [c for c in cols if trf[c].dtype == object]
        num = [c for c in cols if c not in cat]
        try: ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError: ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        ct = ColumnTransformer([("n", StandardScaler(), num), ("c", ohe, cat)], remainder="drop")
        b_tr = ct.fit_transform(trf[cols]).astype(np.float32)
        b_te = ct.transform(tef[cols]).astype(np.float32)
        tr_blocks.append(b_tr); te_blocks.append(b_te); sizes.append(b_tr.shape[1])
    bounds = np.cumsum([0] + sizes)
    slices = [slice(bounds[i], bounds[i + 1]) for i in range(len(sizes))]
    return np.hstack(tr_blocks), np.hstack(te_blocks), slices


def mask_prefix(X, slices, k):
    """Keep first k groups (in cost order); zero the rest."""
    Xm = np.zeros_like(X)
    if k > 0:
        Xm[:, :slices[k - 1].stop] = X[:, :slices[k - 1].stop]
    return Xm


def build_classifier(dim, device):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(dim, 128), nn.ReLU(), nn.Dropout(0.3),
                         nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2)).to(device)


def train_classifier(clf, X, y, slices, device, epochs, seed, batch=512):
    """Train so the classifier works on ANY acquired prefix (random prefix masking)."""
    import torch, torch.nn as nn
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    lf = nn.CrossEntropyLoss()
    Xy = torch.tensor(X, device=device); yt = torch.tensor(y, device=device, dtype=torch.long)
    n = len(X)
    for ep in range(1, epochs + 1):
        clf.train(); perm = rng.permutation(n); tot = 0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            k = int(rng.integers(1, len(slices) + 1))               # random prefix per batch
            xb = torch.tensor(mask_prefix(X[idx], slices, k), device=device)
            opt.zero_grad(); loss = lf(clf(xb), yt[idx]); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        if ep == 1 or ep % 5 == 0 or ep == epochs: print(f"    clf epoch {ep}/{epochs} loss {tot/n:.4f}")
    clf.eval(); return clf


def clf_probs(clf, Xm, device, batch=8192):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(Xm), batch):
            out.append(torch.softmax(clf(torch.tensor(Xm[i:i+batch], device=device)), 1).cpu().numpy())
    return np.concatenate(out)


def f1_acc(pred, y):
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return 2 * p * r / max(p + r, 1e-12), acc


# ---- precompute per-prefix predictions/confidences (for fast env + baselines) ----
def prefix_tables(clf, X, slices, device):
    K = len(slices)
    probs = np.stack([clf_probs(clf, mask_prefix(X, slices, k), device) for k in range(1, K + 1)], 0)  # (K,N,2)
    pred = probs.argmax(2)                 # (K,N)
    conf = probs.max(2)                    # (K,N)
    return pred, conf


def make_env(X, y, slices, pred_tab, cost_lambda, seed):
    import gymnasium as gym
    from gymnasium import spaces
    K = len(slices); costs = np.array([GROUP_COST[g] for g in GROUP_ORDER], np.float32)

    class CostEnv(gym.Env):
        def __init__(s):
            super().__init__()
            s.N = len(X); s.action_space = spaces.Discrete(2)      # 0 acquire, 1 stop
            s.observation_space = spaces.Box(-10, 10, (X.shape[1] + K,), np.float32)
            s.rng = np.random.default_rng(seed); s.order = s.rng.permutation(s.N); s.t = 0

        def _obs(s):
            xm = np.clip(mask_prefix(X[s.i:s.i + 1], slices, s.k)[0], -10, 10)
            bits = np.zeros(K, np.float32); bits[:s.k] = 1.0
            return np.concatenate([xm, bits]).astype(np.float32)

        def reset(s, seed=None, options=None):
            super().reset(seed=seed)
            s.i = s.order[s.t % s.N]; s.t += 1; s.k = 1                # always start with basic
            return s._obs(), {}

        def step(s, a):
            if int(a) == 0 and s.k < K:                                # ACQUIRE next group
                r = -cost_lambda * costs[s.k]; s.k += 1
                return s._obs(), r, False, False, {}
            # STOP (or no more groups) -> classify with current prefix
            yp = pred_tab[s.k - 1, s.i]; yt = y[s.i]
            r = (2.0 if yt == 1 else -1.0) if yp == 1 else (-3.0 if yt == 1 else 1.0)
            return s._obs(), r, True, False, {}
    return CostEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="/content/drive/MyDrive/dataset/Train_data.csv")
    ap.add_argument("--test", default="/content/drive/MyDrive/dataset/Test_data.csv")
    ap.add_argument("--ae-epochs", type=int, default=0)   # unused here; kept for arg-compat
    ap.add_argument("--clf-epochs", type=int, default=25)
    ap.add_argument("--timesteps", type=int, default=200000)
    ap.add_argument("--cost-lambda", type=float, default=0.15)
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

    trf, trl = p1.load_csv(a.train); tef, tel = p1.load_csv(a.test)
    Xtr, Xte, slices = build_group_blocks(trf, tef)
    K = len(slices); costs = np.array([GROUP_COST[g] for g in GROUP_ORDER])
    print(f"groups={GROUP_ORDER} sizes={[s.stop-s.start for s in slices]} costs={list(costs)} dim={Xtr.shape[1]}")

    print("\nTraining prefix-robust classifier ...")
    clf = build_classifier(Xtr.shape[1], dev)
    clf = train_classifier(clf, Xtr, trl, slices, dev, a.clf_epochs, a.seed)

    pred_tr, _ = prefix_tables(clf, Xtr, slices, dev)
    pred_te, conf_te = prefix_tables(clf, Xte, slices, dev)

    # ---- baselines on TEST ----
    def cost_of_prefix(k): return float(costs[:k].sum())
    rows = {}
    f1c, acc = f1_acc(pred_te[0], tel); rows["Cheap (basic only)"] = (cost_of_prefix(1), f1c)
    f1f, _   = f1_acc(pred_te[K - 1], tel); rows["Full (all groups)"] = (cost_of_prefix(K), f1f)
    casc = []
    for tau in (0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
        ks = np.ones(len(tel), int)
        for k in range(1, K):                       # acquire next while not confident
            need = (conf_te[k - 1] < tau) & (ks == k)
            ks[need] = k + 1
        pr = pred_te[ks - 1, np.arange(len(tel))]
        avg_cost = float(np.mean([costs[:k].sum() for k in ks]))
        f1t, _ = f1_acc(pr, tel); casc.append((tau, avg_cost, f1t))

    # ---- train RL + evaluate on TEST ----
    print(f"\nTraining cost-aware PPO ({a.timesteps:,} steps, lambda={a.cost_lambda}) ...")
    Env = make_env(Xtr, trl, slices, pred_tr, a.cost_lambda, a.seed)
    env = DummyVecEnv([lambda: Env()])
    model = PPO("MlpPolicy", env, seed=a.seed, verbose=0, n_steps=2048, batch_size=256,
                device="cpu", policy_kwargs=dict(net_arch=[128, 64, 32]))
    model.learn(total_timesteps=a.timesteps); print("PPO done.")

    # roll the policy over each TEST flow to get its stop-depth k
    ks = np.ones(len(tel), int)
    for i in range(len(tel)):
        k = 1
        while k < K:
            xm = np.clip(mask_prefix(Xte[i:i + 1], slices, k)[0], -10, 10)
            bits = np.zeros(K, np.float32); bits[:k] = 1.0
            a_i, _ = model.predict(np.concatenate([xm, bits]).astype(np.float32), deterministic=True)
            if int(a_i) == 1: break
            k += 1
        ks[i] = k
    pr_rl = pred_te[ks - 1, np.arange(len(tel))]
    rl_cost = float(np.mean([costs[:k].sum() for k in ks]))
    rl_f1, _ = f1_acc(pr_rl, tel)
    rows["RL (adaptive)"] = (rl_cost, rl_f1)

    print("\n================  COST vs F1 (TEST)  ================")
    print(f"{'method':<24}{'avg_cost':>10}{'F1':>8}")
    for n, (c, f) in rows.items(): print(f"{n:<24}{c:>10.2f}{f:>8.3f}")
    print("  cascade frontier (tau, cost, F1):")
    for tau, c, f in casc: print(f"    tau={tau:<5} cost={c:6.2f}  F1={f:.3f}")
    # is RL on/above the cascade frontier? (>= cascade F1 at <= its cost)
    better = all(not (c <= rl_cost and f > rl_f1 + 1e-3) for _, c, f in casc)
    print(f"\nRL: F1={rl_f1:.3f} at avg cost {rl_cost:.2f} (Full F1={f1f:.3f} at cost {cost_of_prefix(K):.0f})")
    print(f"Verdict: {'RL is ON/ABOVE the cost-accuracy frontier ✅' if better else 'cascade matches/beats RL — tune --cost-lambda/--timesteps'}")
    print("====================================================")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        cf = sorted([(c, f) for _, c, f in casc])
        plt.plot([c for c, _ in cf], [f for _, f in cf], "-o", color="gray", label="cascade (tuned)")
        for n, (c, f) in rows.items():
            plt.scatter([c], [f], s=90, label=n, zorder=5)
        plt.xlabel("average inspection cost / flow"); plt.ylabel("F1")
        plt.title("Cost-efficiency frontier: RL vs baselines"); plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
        png = os.path.join(a.outdir, "phase5_pareto.png"); plt.savefig(png, dpi=130); print(f"Saved plot -> {png}")
    except Exception as ex:
        print(f"(plot skipped: {ex})")

    json.dump({"seed": a.seed, "cost_lambda": a.cost_lambda,
               "rows": {k: {"cost": v[0], "f1": v[1]} for k, v in rows.items()},
               "cascade": [{"tau": t, "cost": c, "f1": f} for t, c, f in casc]},
              open(os.path.join(a.outdir, "phase5_metrics.json"), "w"), indent=2)
    print("Saved -> phase5_metrics.json")


if __name__ == "__main__":
    main()
