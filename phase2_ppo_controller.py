"""
Phase 2 — PPO Tier-Controller on a frozen Self-Supervised Autoencoder  (DACA-IDS, step 2)
==========================================================================================
Builds on Phase 1. The autoencoder (trained on Normal only) is FROZEN and provides, per flow:
    state = [ latent z (16-d) , reconstruction error e (1-d) ]   -> 17-d
A PPO agent chooses one of 5 alert tiers {VC, C, B, A, VA}. Reward (paper Section V-C-2):
    A/VA + attack  -> +2.0      VC/C + normal -> +1.0     B -> 0.0
    A/VA + normal  -> -0.5      VC/C + attack -> -3.0   (6:1 miss:false-alarm cost)
For scoring, tiers map to binary:  VC,C -> Normal ;  A,VA -> Attack ;  B -> Normal (conservative).

THE KEY EXPERIMENT (A0 vs A1): we print the PPO controller's test metrics RIGHT NEXT TO the
fixed-percentile-threshold baseline. If PPO does not beat the fixed threshold here, that is an
honest result and motivates adding drift/budget signals in Phase 3. Numbers are reproducible
(fixed seed); nothing is hand-typed.

RUN (Colab, after `!git pull`):
    !python phase2_ppo_controller.py                      # full
    !python phase2_ppo_controller.py --ae-epochs 30 --timesteps 80000   # quick test
Outputs: phase2_metrics.json
"""

import argparse
import importlib
import json
import os
import subprocess
import sys

import numpy as np


# ----------------------------------------------------------------------------
# Make sure RL deps exist (Colab usually lacks stable-baselines3 by default)
# ----------------------------------------------------------------------------
def _ensure(mod, pip_name=None):
    try:
        importlib.import_module(mod)
    except Exception:
        print(f"installing {pip_name or mod} ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name or mod], check=True)


# Reuse Phase 1 helpers (loader, preprocessing, autoencoder, metrics)
import phase1_autoencoder_ids as p1


# ----------------------------------------------------------------------------
# Compact autoencoder training (Normal-only) — same architecture as Phase 1
# ----------------------------------------------------------------------------
def train_autoencoder(X_normal, input_dim, device, epochs, lr, batch, seed):
    import torch
    import torch.nn as nn
    g = np.random.default_rng(seed)
    idx = g.permutation(len(X_normal))
    n_val = max(1, int(0.1 * len(idx)))
    Xval, Xtr = X_normal[idx[:n_val]], X_normal[idx[n_val:]]

    model = p1.build_model(input_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()
    Xtr_t = torch.tensor(Xtr, device=device)
    Xval_t = torch.tensor(Xval, device=device)
    best, best_state, n = float("inf"), None, len(Xtr_t)
    import torch as _t
    for ep in range(1, epochs + 1):
        model.train()
        perm = _t.randperm(n, device=device)
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            xh, _ = model(Xtr_t[b])
            loss = loss_fn(xh, Xtr_t[b])
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with _t.no_grad():
            vh, _ = model(Xval_t)
            v = loss_fn(vh, Xval_t).item()
        if v < best:
            best, best_state = v, {k: val.detach().cpu().clone() for k, val in model.state_dict().items()}
        if ep == 1 or ep % 10 == 0 or ep == epochs:
            print(f"    AE epoch {ep:3d}/{epochs}  val {v:.5f}")
    model.load_state_dict(best_state)
    for prm in model.parameters():            # FREEZE the detector
        prm.requires_grad = False
    model.eval()
    print(f"    frozen AE best val MSE: {best:.5f}")
    return model


def latent_and_error(model, X, device, batch=4096):
    """Return (z, e): 16-d latent and scalar reconstruction error per sample."""
    import torch
    zs, es = [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.tensor(X[i:i + batch], dtype=torch.float32, device=device)
            xh, z = model(xb)
            zs.append(z.cpu().numpy())
            es.append(((xb - xh) ** 2).mean(dim=1).cpu().numpy())
    return np.concatenate(zs), np.concatenate(es)


# ----------------------------------------------------------------------------
# RL environment: present one flow's (z,e) state, agent picks a tier, get reward
# ----------------------------------------------------------------------------
def make_env_class():
    import gymnasium as gym
    from gymnasium import spaces

    class TierEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self, states, labels, episode_len, seed):
            super().__init__()
            self.states = states.astype(np.float32)
            self.labels = labels.astype(int)
            self.N = len(states)
            self.episode_len = episode_len
            self.action_space = spaces.Discrete(5)          # 0VC 1C 2B 3A 4VA
            self.observation_space = spaces.Box(-10.0, 10.0, shape=(states.shape[1],), dtype=np.float32)
            self._rng = np.random.default_rng(seed)
            self.t = 0
            self.order = self._rng.permutation(self.N)

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.t = 0
            self.order = self._rng.permutation(self.N)
            return self.states[self.order[0]], {}

        @staticmethod
        def _reward(a, y):
            if a in (3, 4):                 # predict ATTACK
                return 2.0 if y == 1 else -0.5
            if a in (0, 1):                 # predict NORMAL
                return 1.0 if y == 0 else -3.0
            return 0.0                      # B: borderline / abstain

        def step(self, action):
            idx = self.order[self.t % self.N]
            r = self._reward(int(action), self.labels[idx])
            self.t += 1
            done = self.t >= self.episode_len
            obs = self.states[self.order[self.t % self.N]]
            return obs, r, done, False, {}

    return TierEnv


def tiers_to_binary(actions, b_as_attack=False):
    actions = np.asarray(actions)
    pred = np.isin(actions, [3, 4]).astype(int)          # A/VA -> attack
    if b_as_attack:
        pred[actions == 2] = 1
    return pred


def metrics_from_pred(pred, y):
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    fpr = fp / max(fp + tn, 1)
    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1, fpr=fpr,
                tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn))


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="/content/drive/MyDrive/dataset/Train_data.csv")
    ap.add_argument("--test", default="/content/drive/MyDrive/dataset/Test_data.csv")
    ap.add_argument("--ae-epochs", type=int, default=100)
    ap.add_argument("--timesteps", type=int, default=300000)
    ap.add_argument("--episode-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--b-as-attack", action="store_true",
                    help="map the Balanced tier to Attack instead of Normal")
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    _ensure("gymnasium")
    _ensure("stable_baselines3", "stable-baselines3")
    p1.set_seed(args.seed)

    if args.train.startswith("/content/drive") and not os.path.exists(args.train):
        try:
            from google.colab import drive
            drive.mount("/content/drive")
        except Exception:
            pass

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from sklearn.metrics import roc_auc_score
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | seed: {args.seed}")

    # ---- 1. Data + preprocessing (reuse Phase 1) ----
    print("\n[1] Loading + preprocessing ...")
    tr_feats, tr_labels = p1.load_csv(args.train)
    te_feats, te_labels = p1.load_csv(args.test)
    if tr_labels is None or te_labels is None:
        raise SystemExit("Need labelled train AND test for Phase 2.")
    pre, cat, num = p1.build_preprocessor(tr_feats)
    Xtr = pre.fit_transform(tr_feats).astype(np.float32)
    Xte = pre.transform(te_feats).astype(np.float32)
    input_dim = Xtr.shape[1]
    print(f"    input_dim={input_dim} | train={Xtr.shape} test={Xte.shape}")

    # ---- 2. Frozen autoencoder ----
    print("\n[2] Training + freezing autoencoder (Normal only) ...")
    ae = train_autoencoder(Xtr[tr_labels == 0], input_dim, device,
                           args.ae_epochs, 1e-3, 256, args.seed)

    # ---- 3. Build (z,e) states; standardise with train stats ----
    print("\n[3] Building (z,e) states ...")
    ztr, etr = latent_and_error(ae, Xtr, device)
    zte, ete = latent_and_error(ae, Xte, device)
    Str = np.concatenate([ztr, etr[:, None]], axis=1)
    Ste = np.concatenate([zte, ete[:, None]], axis=1)
    mu, sd = Str.mean(0), Str.std(0) + 1e-8
    Str = np.clip((Str - mu) / sd, -10, 10).astype(np.float32)
    Ste = np.clip((Ste - mu) / sd, -10, 10).astype(np.float32)
    print(f"    state dim = {Str.shape[1]} (16 latent + 1 error)")

    # ---- 4. Train PPO tier-controller ----
    print(f"\n[4] Training PPO ({args.timesteps:,} steps) ...")
    TierEnv = make_env_class()
    env = DummyVecEnv([lambda: TierEnv(Str, tr_labels, args.episode_len, args.seed)])
    model = PPO("MlpPolicy", env, seed=args.seed, verbose=0, n_steps=args.episode_len,
                policy_kwargs=dict(net_arch=[128, 64, 32]))
    model.learn(total_timesteps=args.timesteps)
    print("    PPO training done.")

    # ---- 5. Evaluate PPO on TEST ----
    print("\n[5] Evaluating PPO controller on official TEST set ...")
    actions, _ = model.predict(Ste, deterministic=True)
    actions = np.asarray(actions).reshape(-1)
    dist = {t: int((actions == i).sum()) for i, t in enumerate(["VC", "C", "B", "A", "VA"])}
    pred = tiers_to_binary(actions, args.b_as_attack)
    ppo_m = metrics_from_pred(pred, te_labels)
    print(f"    tier usage: {dist}")

    # ---- 6. Fixed-threshold baseline (A0) on TEST, for direct comparison ----
    normal_err = etr[tr_labels == 0]
    thr85 = float(np.percentile(normal_err, 85))
    thr95 = float(np.percentile(normal_err, 95))
    base85 = p1.evaluate(ete, te_labels, thr85)
    base95 = p1.evaluate(ete, te_labels, thr95)
    roc = roc_auc_score(te_labels, ete)

    # ---- 7. Report: A0 vs A1(PPO) ----
    print("\n================  TEST RESULTS  ================")
    print(f"AE reconstruction-error ROC-AUC: {roc:.4f}")
    print(f"{'method':<28}{'acc':>8}{'prec':>8}{'rec':>8}{'f1':>8}{'fpr':>8}")
    def row(name, m):
        print(f"{name:<28}{m['accuracy']:8.4f}{m['precision']:8.4f}{m['recall']:8.4f}{m['f1']:8.4f}{m['fpr']:8.4f}")
    row("A0 fixed threshold P85", base85)
    row("A0 fixed threshold P95", base95)
    row("A1 PPO tier-controller", ppo_m)
    best_fixed = max(base85["f1"], base95["f1"])
    verdict = ("PPO BEATS fixed threshold" if ppo_m["f1"] > best_fixed + 1e-4
               else "PPO does NOT beat fixed threshold (honest result -> motivates drift/budget, Phase 3)")
    print(f"\nVerdict: {verdict}  (PPO F1 {ppo_m['f1']:.4f} vs best fixed F1 {best_fixed:.4f})")
    print("================================================")

    with open(os.path.join(args.outdir, "phase2_metrics.json"), "w") as f:
        json.dump(dict(seed=args.seed, input_dim=input_dim, timesteps=args.timesteps,
                       ae_roc_auc=roc, tier_usage=dist,
                       ppo=ppo_m, fixed_P85=base85, fixed_P95=base95,
                       b_as_attack=args.b_as_attack), f, indent=2)
    print("Saved -> phase2_metrics.json")
    print("\nReproducible with the same seed. PPO vs fixed threshold is the core Phase-2 ablation.")


if __name__ == "__main__":
    main()
