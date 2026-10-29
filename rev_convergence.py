"""
Revision experiment for IEEE Access reviewer comment R2.7 (convergence and training budget).

Reviewer 2 asks whether the negative reinforcement-learning result could be an artefact of
insufficient training rather than a property of the problem. The manuscript reports results at
80,000 timesteps. This script answers the question directly by training the same controller for
300,000 timesteps - 3.75 times the published budget - and recording test F1 at checkpoints
throughout, so the shape of the curve is visible rather than inferred from its endpoint.

Two things are logged at every checkpoint: test F1 of the deterministic policy, and the mean
training episode reward from the agent's own buffer. A policy that is still improving shows a
rising curve in both; a policy that has converged to a decision the task does not reward shows a
plateau. The tuned-threshold and matched-capacity linear baselines are computed exactly as in
exp2_static_seeds so the gap can be read against the same reference the manuscript uses.

Run on NSL-KDD, where the published margin is widest, and UNSW-NB15, where it is narrowest: if
extra budget were going to change a verdict anywhere, UNSW-NB15 is where it would show.

Reuses exp2_static_seeds, phase2_ppo_controller, phase3_drift and phase8_unsw unchanged.
"""
import os, sys, json, csv, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase1_autoencoder_ids as p1
import phase2_ppo_controller as p2
import phase3_drift as p3
import phase8_unsw as p8
import exp_harness as H
import exp2_static_seeds as E2

CHECKPOINTS = (10_000, 20_000, 40_000, 80_000, 150_000, 225_000, 300_000)


def make_callback(Ste, tel, seed, dataset, rows, writer, fh):
    """Logs the FULL operating point at each checkpoint, not F1 alone. An agent can raise F1 on
    an attack-heavy benchmark simply by flagging more traffic, so precision, false-positive rate
    and alert volume are needed to tell a genuine improvement from a shift along the same curve."""
    from stable_baselines3.common.callbacks import BaseCallback
    from sklearn.metrics import confusion_matrix

    class Probe(BaseCallback):
        def __init__(cb):
            super().__init__(); cb.next_i = 0

        def _on_step(cb):
            if cb.next_i < len(CHECKPOINTS) and cb.num_timesteps >= CHECKPOINTS[cb.next_i]:
                ck = CHECKPOINTS[cb.next_i]; cb.next_i += 1
                acts = np.asarray(cb.model.predict(Ste, deterministic=True)[0]).reshape(-1)
                pred = p2.tiers_to_binary(acts)
                tn, fp, fn, tp = confusion_matrix(tel, pred, labels=[0, 1]).ravel()
                prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
                f1 = 2 * prec * rec / max(prec + rec, 1e-12)
                buf = list(cb.model.ep_info_buffer or [])
                ep_rew = float(np.mean([e["r"] for e in buf])) if buf else float("nan")
                row = dict(dataset=dataset, seed=seed, timesteps=ck, test_f1=round(f1, 4),
                           precision=round(prec, 4), recall=round(rec, 4),
                           fpr=round(fp / max(fp + tn, 1), 4),
                           fnr=round(fn / max(fn + tp, 1), 4),
                           alerts_per_1000=round(1000.0 * (tp + fp) / max(len(tel), 1), 1),
                           frac_flagged=round(float(np.mean(pred)), 4),
                           train_ep_reward=round(ep_rew, 4))
                rows.append(row); writer.writerow(row); fh.flush()
                print(f"    [{dataset} seed {seed}] {ck:>7,}  F1 {f1:.4f}  prec {prec:.4f}  "
                      f"rec {rec:.4f}  FPR {row['fpr']:.4f}  alerts/1k {row['alerts_per_1000']:.0f}")
            return True
    return Probe()


def run_dataset(name, load, preproc, epochs, seeds, timesteps, writer, fh):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from sklearn.metrics import f1_score
    from sklearn.linear_model import LogisticRegression

    dev = torch.device("cpu")
    (trf, trl), (tef, tel) = load()
    pre, _, _ = preproc(trf)
    Xtr = pre.fit_transform(trf).astype(np.float32); Xte = pre.transform(tef).astype(np.float32)
    H.set_all_seeds(0)
    ae = p3.train_ae(Xtr[trl == 0], Xtr.shape[1], dev, epochs, 0)
    Ztr, Etr = p3.latent_err(ae, Xtr, dev)
    zmu, zsd = Ztr.mean(0), Ztr.std(0) + 1e-8
    emu, esd = float(Etr.mean()), float(Etr.std()) + 1e-8
    Str = np.concatenate([np.clip((Ztr - zmu) / zsd, -10, 10),
                          np.clip(((Etr - emu) / esd)[:, None], -10, 10)], 1).astype(np.float32)
    Ste, Ete = E2.make_states(ae, Xte, dev, zmu, zsd, emu, esd)

    base_thr = p8.best_threshold(Ete, tel)["f1"]
    lrf = LogisticRegression(class_weight="balanced", max_iter=2000).fit(Str, trl)
    ptr = lrf.predict_proba(Str)[:, 1]; pte = lrf.predict_proba(Ste)[:, 1]
    cuts = np.quantile(ptr, np.linspace(0.01, 0.99, 200))
    cut = float(cuts[int(np.argmax([f1_score(trl, (ptr >= c).astype(int), zero_division=0)
                                    for c in cuts]))])
    base_lr = float(f1_score(tel, (pte >= cut).astype(int), zero_division=0))
    base_best = max(base_thr, base_lr)
    print(f"  {name}: threshold(e) {base_thr:.4f} | LR(z,e) {base_lr:.4f} | best-simple {base_best:.4f}")

    TierEnv = p2.make_env_class(); rows = []
    for s in seeds:
        H.set_all_seeds(s)
        from stable_baselines3.common.monitor import Monitor
        # Monitor populates ep_info_buffer; without it the training-reward column is all NaN.
        env = DummyVecEnv([lambda: Monitor(TierEnv(Str, trl, 2048, s))])
        model = PPO("MlpPolicy", env, seed=s, verbose=0, n_steps=2048, batch_size=256,
                    device="cpu", policy_kwargs=dict(net_arch=[128, 64, 32]))
        model.learn(total_timesteps=timesteps,
                    callback=make_callback(Ste, tel, s, name, rows, writer, fh))
    for r in rows:
        r["baseline_best_simple"] = round(base_best, 4)
    return dict(dataset=name, base_thr=base_thr, base_lr=base_lr, baseline=base_best,
                curve=rows, timesteps=timesteps, seeds=list(seeds))


FIELDS = ["dataset", "seed", "timesteps", "test_f1", "precision", "recall", "fpr", "fnr",
          "alerts_per_1000", "frac_flagged", "train_ep_reward"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsl-train"); ap.add_argument("--nsl-test")
    ap.add_argument("--unsw-train"); ap.add_argument("--unsw-test")
    ap.add_argument("--ciciot-train"); ap.add_argument("--ciciot-test")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--timesteps", type=int, default=300000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args()

    DS = [("NSL-KDD", (a.nsl_train, a.nsl_test),
           lambda: (p1.load_csv(a.nsl_train), p1.load_csv(a.nsl_test)), p1.build_preprocessor),
          ("UNSW-NB15", (a.unsw_train, a.unsw_test),
           lambda: (p8.load_unsw(a.unsw_train), p8.load_unsw(a.unsw_test)), p8.build_preprocessor),
          ("CIC-IoT2023", (a.ciciot_train, a.ciciot_test),
           lambda: (p1.load_csv(a.ciciot_train), p1.load_csv(a.ciciot_test)), p1.build_preprocessor)]

    seeds = H.SEEDS[:a.seeds]
    os.makedirs(os.path.join(a.outdir, "results"), exist_ok=True)
    out = os.path.join(a.outdir, "results", "rev_convergence.csv")
    summary = []
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); fh.flush()
        for name, paths, load, pre in DS:
            if not all(p and os.path.exists(p) for p in paths):
                print(f"  {name}: inputs missing -> skipped"); continue
            print(f"\n===== {name} ({a.timesteps:,} timesteps, {a.seeds} seeds) =====")
            summary.append(run_dataset(name, load, pre, a.epochs, seeds, a.timesteps, w, fh))
    json.dump(summary, open(os.path.join(a.outdir, "results", "rev_convergence.json"), "w"), indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
