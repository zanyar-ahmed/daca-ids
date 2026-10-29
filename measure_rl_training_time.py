"""Measure wall-clock training cost of the PPO controller (Reviewer 1, Comment 3).

Reuses phase2_ppo_controller.py's own pipeline so the measurement is of the code
actually released with the paper. Autoencoder training and PPO training are timed
separately, at both budgets the paper uses: 80,000 timesteps (detection phases) and
300,000 (the convergence experiment of Section V-M).

Hardware is recorded in the output; it is NOT the Colab T4 of Table XX, so the
absolute seconds are not comparable to that table's AE figure. The AE time is
re-measured here on the same machine so the AE:PPO ratio is internally consistent.
"""
import argparse, json, os, platform, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase2_ppo_controller as p2
import phase1_autoencoder_ids as p1

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

TRAIN = "dataset/Train_data.csv"
TEST = "dataset/Test_data.csv"
SEED = 42
AE_EPOCHS = 100
EPISODE_LEN = 2048
BUDGETS = [80_000, 300_000]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=TRAIN)
    ap.add_argument("--test", default=TEST)   # unused here; kept for arg-compat with reproduce.sh
    ap.add_argument("--out", default="rl_training_time.json")
    a = ap.parse_args()
    p1.set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Same preprocessing path as the released phase-2 script
    tr_feats, tr_labels = p1.load_csv(a.train)
    pre, cat, num = p1.build_preprocessor(tr_feats)
    Xtr = pre.fit_transform(tr_feats).astype(np.float32)
    input_dim = Xtr.shape[1]

    t0 = time.perf_counter()
    ae = p2.train_autoencoder(Xtr[tr_labels == 0], input_dim, device,
                              AE_EPOCHS, 1e-3, 256, SEED)
    ae_seconds = time.perf_counter() - t0

    ztr, etr = p2.latent_and_error(ae, Xtr, device)
    Str = np.concatenate([ztr, etr[:, None]], axis=1)
    mu, sd = Str.mean(0), Str.std(0) + 1e-8
    Str = np.clip((Str - mu) / sd, -10, 10).astype(np.float32)
    TierEnv = p2.make_env_class()

    out = {
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "cpu_count": os.cpu_count(),
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "torch": torch.__version__,
        },
        "seed": SEED,
        "ae_epochs": AE_EPOCHS,
        "ae_train_seconds": round(ae_seconds, 1),
        "episode_len": EPISODE_LEN,
        "ppo": {},
    }

    for budget in BUDGETS:
        p1.set_seed(SEED)
        env = DummyVecEnv([lambda: TierEnv(Str, tr_labels, EPISODE_LEN, SEED)])
        model = PPO("MlpPolicy", env, seed=SEED, verbose=0, n_steps=EPISODE_LEN,
                    policy_kwargs=dict(net_arch=[128, 64, 32]))
        t1 = time.perf_counter()
        model.learn(total_timesteps=budget)
        secs = time.perf_counter() - t1
        out["ppo"][str(budget)] = {
            "train_seconds": round(secs, 1),
            "seconds_per_1k_timesteps": round(secs / (budget / 1000.0), 3),
            "ratio_to_ae_training": round(secs / ae_seconds, 2),
        }
        print("budget %d -> %.1f s" % (budget, secs), flush=True)

    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
