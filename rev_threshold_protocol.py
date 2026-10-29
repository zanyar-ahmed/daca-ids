"""
Revision experiments for IEEE Access reviewer comments R2.3 and R2.9.

R2.3 (validation-based threshold selection). The manuscript reports an ORACLE F1-optimal
threshold, whose cut-point is chosen on the evaluation partition, alongside a deployable
percentile rule. Reviewer 2 asks for the protocol in between: choose the F1-optimal cut-point
on a held-out VALIDATION split, then evaluate it once on the untouched test partition. That
is what this script adds.

To keep the protocol airtight the validation split is carved out FIRST and the autoencoder is
trained only on the normal rows of the remainder, so no comparator is ever tuned on data the
detector saw. All three comparators are then computed from that one detector, which makes them
directly comparable to each other. The detector therefore sees 80 per cent of the normal
training rows rather than all of them; it is a slightly weaker detector than the one in the
manuscript, which means any threshold result it produces is a conservative bound.

R2.9 (operational metrics). For every comparator this reports precision, recall, false-positive
rate, false-negative rate, accuracy, F1 and alert volume per 1000 flows, and sweeps the attack
prevalence of the test partition so each operating point can be read at realistic base rates.
Thresholds are NOT re-tuned per prevalence: each keeps the cut-point its own protocol produced,
which is what an operator would actually deploy.

Reuses the project's loaders, preprocessor, autoencoder and metric function unchanged.
"""
import os, sys, json, csv, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase1_autoencoder_ids as p1
import phase3_drift as p3
import phase8_unsw as p8
import phase2_ppo_controller as p2
import exp_harness as H

TIERS = (60, 85, 95, 99)
PREVALENCES = (0.05, 0.10, 0.20, 0.30, 0.50)
VAL_FRAC = 0.20


def op_metrics(scores, y, thr, n_ref=None):
    """phase1.evaluate plus the operational quantities the reviewer asks for."""
    m = p1.evaluate(scores, y, float(thr))
    n = len(y) if n_ref is None else n_ref
    alerts = m["tp"] + m["fp"]
    m["fnr"] = m["fn"] / max(m["fn"] + m["tp"], 1)
    m["alerts_per_1000"] = 1000.0 * alerts / max(n, 1)
    m["alert_count"] = int(alerts)
    m["n"] = int(len(y))
    m["prevalence"] = float(np.mean(y))
    return m


def f1_optimal(scores, y, n_grid=400):
    """Cut-point maximising F1 on whatever partition it is handed."""
    qs = np.unique(np.quantile(scores, np.linspace(0.001, 0.999, n_grid)))
    best_thr, best_f1 = None, -1.0
    for t in qs:
        f = p1.evaluate(scores, y, float(t))["f1"]
        if f > best_f1:
            best_thr, best_f1 = float(t), f
    return best_thr


def stratified_split(y, frac, seed):
    rng = np.random.default_rng(seed)
    val = np.zeros(len(y), bool)
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        val[idx[: int(round(frac * len(idx)))]] = True
    return val


def resample_to_prevalence(y, target, seed):
    """Index set whose attack prevalence equals `target`, keeping as many rows as possible.

    Exactly one class is kept whole and the other subsampled. Only options that can actually
    reach the target are considered: keeping all attacks needs len(a)*(1-p)/p normals, which on
    an attack-heavy benchmark often exceeds the normals available. Comparing the two options
    without first testing feasibility silently returns the whole set at its native prevalence.
    """
    rng = np.random.default_rng(seed)
    a = np.flatnonzero(y == 1)
    n = np.flatnonzero(y == 0)
    opts = []
    need_n = int(round(len(a) * (1 - target) / max(1e-9, target)))      # keep all attacks
    if need_n <= len(n):
        opts.append((len(a) + need_n, a, rng.choice(n, need_n, replace=False)))
    need_a = int(round(len(n) * target / max(1e-9, 1 - target)))        # keep all normals
    if need_a <= len(a):
        opts.append((len(n) + need_a, rng.choice(a, need_a, replace=False), n))
    if not opts:
        raise ValueError(f"prevalence {target} unreachable from {len(a)} attack / {len(n)} normal rows")
    _, ai, ni = max(opts, key=lambda o: o[0])
    keep = np.concatenate([ai, ni])
    rng.shuffle(keep)
    return keep


def run_dataset(name, load, pre, epochs, seed, writer, fh, rl_seeds=(), rl_timesteps=80000):
    import torch
    t0 = time.time()
    (trf, trl), (tef, tel) = load()
    prep, _, _ = pre(trf)
    # build_preprocessor returns an UNFITTED transformer; fit on the training partition only.
    # It is fitted on features alone, so this leaks no label and nothing from the test split.
    Xtr = prep.fit_transform(trf).astype(np.float32)
    Xte = prep.transform(tef).astype(np.float32)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. carve the validation split BEFORE the detector is trained
    is_val = stratified_split(trl, VAL_FRAC, seed)
    Xfit, yfit = Xtr[~is_val], trl[~is_val]
    Xval, yval = Xtr[is_val], trl[is_val]

    # 2. detector: normal rows of the fitting remainder only.
    # train_ae seeds only its own train/validation split; torch weight initialisation reads the
    # GLOBAL seed, so without this the detector drifts by ~0.005 F1 between identical runs.
    H.set_all_seeds(seed)
    Xn = Xfit[yfit == 0]
    model = p3.train_ae(Xn, Xtr.shape[1], dev, epochs, seed)
    _, e_fit_n = p3.latent_err(model, Xn, dev)
    _, e_val = p3.latent_err(model, Xval, dev)
    _, e_te = p3.latent_err(model, Xte, dev)

    # 3. three comparators, all from this one detector
    thr_val = f1_optimal(e_val, yval)                 # NEW: chosen on validation
    thr_oracle = f1_optimal(e_te, tel)                # the manuscript's oracle comparator
    tiers = {f"P{p}": float(np.percentile(e_fit_n, p)) for p in TIERS}
    best_tier = max(tiers, key=lambda k: p1.evaluate(e_te, tel, tiers[k])["f1"])

    comparators = [
        ("validation-selected F1-optimal", thr_val, "labels from a held-out validation split; test untouched"),
        ("oracle F1-optimal (test-selected)", thr_oracle, "upper bound; selected on the evaluation partition"),
        (f"deployable percentile {best_tier}", tiers[best_tier], "cut-point from normal training errors only"),
    ]

    # 3b. the reinforcement-learning arm, trained on the SAME detector so that the comparison
    # against each threshold protocol is like-for-like. Without this the RL figures quoted in the
    # manuscript would be coming from a detector trained on all the normal rows, not 80 per cent
    # of them, and the margin against the validation-selected threshold would not be meaningful.
    rl_rows = []
    if rl_seeds:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.monitor import Monitor
        from sklearn.metrics import confusion_matrix
        Zfit, Efit = p3.latent_err(model, Xfit, dev)
        zmu, zsd = Zfit.mean(0), Zfit.std(0) + 1e-8
        emu, esd = float(Efit.mean()), float(Efit.std()) + 1e-8

        def states(Z, E):
            return np.concatenate([np.clip((Z - zmu) / zsd, -10, 10),
                                   np.clip(((E - emu) / esd)[:, None], -10, 10)], 1).astype(np.float32)
        Sfit = states(Zfit, Efit)
        Zte, Ete2 = p3.latent_err(model, Xte, dev)
        Ste = states(Zte, Ete2)
        TierEnv = p2.make_env_class()
        for sd in rl_seeds:
            H.set_all_seeds(sd)
            env = DummyVecEnv([lambda: Monitor(TierEnv(Sfit, yfit, 2048, sd))])
            agent = PPO("MlpPolicy", env, seed=sd, verbose=0, n_steps=2048, batch_size=256,
                        device="cpu", policy_kwargs=dict(net_arch=[128, 64, 32]))
            agent.learn(total_timesteps=rl_timesteps)
            pred = p2.tiers_to_binary(np.asarray(agent.predict(Ste, deterministic=True)[0]).reshape(-1))
            tn, fp, fn, tp = confusion_matrix(tel, pred, labels=[0, 1]).ravel()
            prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
            rl_rows.append(dict(dataset=name, comparator=f"PPO controller (seed {sd})",
                                protocol=f"same detector; {rl_timesteps} timesteps",
                                prevalence_setting="native", prevalence=float(np.mean(tel)),
                                threshold="", f1=2 * prec * rec / max(prec + rec, 1e-12),
                                precision=prec, recall=rec, fpr=fp / max(fp + tn, 1),
                                fnr=fn / max(fn + tp, 1),
                                alerts_per_1000=1000.0 * (tp + fp) / max(len(tel), 1),
                                alert_count=int(tp + fp), tp=int(tp), tn=int(tn), fp=int(fp),
                                fn=int(fn), n=int(len(tel)), accuracy=(tp + tn) / max(len(tel), 1),
                                input_dim=int(Xtr.shape[1]), n_ae_train=int(len(Xn)),
                                epochs=epochs, seed=sd, secs=""))
        if rl_rows:
            mf = float(np.mean([r["f1"] for r in rl_rows]))
            print(f"  {name:<12} {'PPO controller (mean of ' + str(len(rl_rows)) + ' seeds)':<34} "
                  f"           F1 {mf:.4f}  "
                  f"prec {np.mean([r['precision'] for r in rl_rows]):.4f}  "
                  f"rec {np.mean([r['recall'] for r in rl_rows]):.4f}  "
                  f"FPR {np.mean([r['fpr'] for r in rl_rows]):.4f}  "
                  f"alerts/1k {np.mean([r['alerts_per_1000'] for r in rl_rows]):.1f}")

    rows = []
    for cname, thr, note in comparators:
        m = op_metrics(e_te, tel, thr)
        m.update(dataset=name, comparator=cname, protocol=note, prevalence_setting="native",
                 input_dim=int(Xtr.shape[1]), n_ae_train=int(len(Xn)), epochs=epochs, seed=seed,
                 secs=round(time.time() - t0, 1))
        rows.append(m)
        # 4. prevalence sweep, cut-point held fixed
        for pv in PREVALENCES:
            keep = resample_to_prevalence(tel, pv, seed)
            mp = op_metrics(e_te[keep], tel[keep], thr)
            mp.update(dataset=name, comparator=cname, protocol=note,
                      prevalence_setting=f"{pv:.2f}", input_dim=int(Xtr.shape[1]),
                      n_ae_train=int(len(Xn)), epochs=epochs, seed=seed, secs="")
            rows.append(mp)

    rows.extend(rl_rows)
    for r in rows:
        writer.writerow({k: r.get(k, "") for k in FIELDS})
    fh.flush()
    # PPO rows carry no threshold and are printed above; formatting "" as a float aborts the run
    native = [r for r in rows if r["prevalence_setting"] == "native"
              and isinstance(r.get("threshold"), (int, float))]
    for r in native:
        print(f"  {name:<12} {r['comparator']:<34} thr {r['threshold']:.4f}  F1 {r['f1']:.4f}  "
              f"prec {r['precision']:.4f}  rec {r['recall']:.4f}  FPR {r['fpr']:.4f}  "
              f"FNR {r['fnr']:.4f}  alerts/1k {r['alerts_per_1000']:.1f}")
    return rows


FIELDS = ["dataset", "comparator", "protocol", "prevalence_setting", "prevalence", "threshold",
          "f1", "precision", "recall", "fpr", "fnr", "accuracy", "alerts_per_1000", "alert_count",
          "tp", "tn", "fp", "fn", "n", "input_dim", "n_ae_train", "epochs", "seed", "secs"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsl-train"); ap.add_argument("--nsl-test")
    ap.add_argument("--unsw-train"); ap.add_argument("--unsw-test")
    ap.add_argument("--ciciot-train"); ap.add_argument("--ciciot-test")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rl-seeds", type=int, default=0, help="0 disables the RL arm")
    ap.add_argument("--rl-timesteps", type=int, default=80000)
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args()

    DS = [("NSL-KDD", (a.nsl_train, a.nsl_test),
           lambda: (p1.load_csv(a.nsl_train), p1.load_csv(a.nsl_test)), p1.build_preprocessor),
          ("UNSW-NB15", (a.unsw_train, a.unsw_test),
           lambda: (p8.load_unsw(a.unsw_train), p8.load_unsw(a.unsw_test)), p8.build_preprocessor),
          ("CIC-IoT2023", (a.ciciot_train, a.ciciot_test),
           lambda: (p1.load_csv(a.ciciot_train), p1.load_csv(a.ciciot_test)), p1.build_preprocessor)]

    os.makedirs(os.path.join(a.outdir, "results"), exist_ok=True)
    out = os.path.join(a.outdir, "results", "rev_threshold_protocol.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); fh.flush()
        for name, paths, load, pre in DS:
            if not all(p and os.path.exists(p) for p in paths):
                print(f"  {name}: inputs missing -> skipped"); continue
            print(f"\n===== {name} =====")
            run_dataset(name, load, pre, a.epochs, a.seed, w, fh,
                        rl_seeds=H.SEEDS[:a.rl_seeds], rl_timesteps=a.rl_timesteps)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
