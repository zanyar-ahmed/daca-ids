"""Runtime, latency and memory footprint of the detection pipeline (Table XX).

Measures every quantity Table XX reports, reusing the released phase-1 and phase-2 code so
the numbers come from the same pipeline the paper evaluates:

  * autoencoder training, 30 epochs on the 90% split of the normal training flows
  * autoencoder inference latency per single flow (batch = 1), p50 / p95 / p99
  * percentile-threshold decision time
  * PPO policy inference latency per single flow, p50 / p95 / p99
  * autoencoder and PPO parameter counts
  * peak GPU allocation (when CUDA is present) and peak process RSS

NOTE ON THE PUBLISHED FIGURES. The Colab T4 values printed in Table XX were measured in the
original Colab session; this script performs the same measurements and will reproduce them
only up to hardware. Absolute latencies are machine-dependent by construction, which is why
Table XX labels its rows by the machine they were taken on. Do not edit measured output to
agree with the table.

Usage:
    python exp5_runtime.py --train dataset/Train_data.csv [--reps 2000] [--out runtime.json]
"""
import argparse, json, os, platform, resource, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase1_autoencoder_ids as p1
import phase2_ppo_controller as p2


def _pct(xs):
    a = np.asarray(xs, dtype=float) * 1e3           # seconds -> ms
    return {"p50_ms": round(float(np.percentile(a, 50)), 3),
            "p95_ms": round(float(np.percentile(a, 95)), 3),
            "p99_ms": round(float(np.percentile(a, 99)), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="dataset/Train_data.csv")
    ap.add_argument("--ae-epochs", type=int, default=30)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runtime.json")
    a = ap.parse_args()

    p1.set_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feats, labels = p1.load_csv(a.train)
    pre, _, _ = p1.build_preprocessor(feats)
    X = pre.fit_transform(feats).astype(np.float32)
    dim = X.shape[1]

    t0 = time.perf_counter()
    ae = p2.train_autoencoder(X[labels == 0], dim, dev, a.ae_epochs, 1e-3, 256, a.seed)
    ae_train_s = time.perf_counter() - t0

    # Per-flow autoencoder forward pass, batch = 1
    ae.eval()
    rows = [torch.tensor(X[i:i + 1], dtype=torch.float32, device=dev)
            for i in range(min(a.reps, len(X)))]
    with torch.no_grad():
        for r in rows[:50]:
            ae(r)                                    # warm up
        ae_lat = []
        for r in rows:
            s = time.perf_counter(); ae(r); ae_lat.append(time.perf_counter() - s)

    # Percentile-threshold decision: one comparison against a precomputed cut-point
    _, errs = p2.latent_and_error(ae, X, dev)
    tau = float(np.percentile(errs[labels == 0], 85))
    thr_lat = []
    for e in errs[:a.reps]:
        s = time.perf_counter(); _ = float(e) >= tau; thr_lat.append(time.perf_counter() - s)

    # PPO policy forward pass, batch = 1, on the (z, e) state the controller consumes
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    ztr, etr = p2.latent_and_error(ae, X, dev)
    St = np.concatenate([ztr, etr[:, None]], axis=1)
    mu, sd = St.mean(0), St.std(0) + 1e-8
    St = np.clip((St - mu) / sd, -10, 10).astype(np.float32)
    TierEnv = p2.make_env_class()
    env = DummyVecEnv([lambda: TierEnv(St, labels, 2048, a.seed)])
    model = PPO("MlpPolicy", env, seed=a.seed, verbose=0, n_steps=2048,
                policy_kwargs=dict(net_arch=[128, 64, 32]))
    for s_ in St[:50]:
        model.predict(s_, deterministic=True)        # warm up
    rl_lat = []
    for s_ in St[:a.reps]:
        s = time.perf_counter(); model.predict(s_, deterministic=True)
        rl_lat.append(time.perf_counter() - s)

    out = {
        "hardware": {"platform": platform.platform(), "cpu_count": os.cpu_count(),
                     "device": str(dev), "cuda_available": torch.cuda.is_available(),
                     "torch": torch.__version__},
        "seed": a.seed, "input_dim": dim, "reps": len(ae_lat),
        "ae_train_seconds": round(ae_train_s, 1),
        "ae_epochs": a.ae_epochs,
        "ae_inference": _pct(ae_lat),
        "threshold_decision_mean_us": round(float(np.mean(thr_lat)) * 1e6, 3),
        "ppo_inference": _pct(rl_lat),
        "ae_parameters": int(sum(p.numel() for p in ae.parameters())),
        "ppo_parameters": int(sum(p.numel() for p in model.policy.parameters())),
        "peak_gpu_mb": (round(torch.cuda.max_memory_allocated() / 1e6, 1)
                        if torch.cuda.is_available() else None),
        "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                             / (1e9 if sys.platform == "darwin" else 1e6), 2),
    }
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
