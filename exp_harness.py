"""
Shared evaluation harness (Protocol #0, #3, #7) — used by the experiment scripts.
Determinism + metrics + confidence intervals + significance tests + publication figure style.
"""
import os, random
import numpy as np


SEEDS = list(range(15))   # fixed 15-seed list; reuse for EVERY phase (journals like >=10-15)


def set_all_seeds(s):
    os.environ["PYTHONHASHSEED"] = str(s)
    random.seed(s); np.random.seed(s)
    try:
        import torch
        torch.manual_seed(s); torch.cuda.manual_seed_all(s)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def metrics(y_true, y_pred, y_score):
    from sklearn.metrics import (f1_score, roc_auc_score, average_precision_score,
                                 precision_score, recall_score, accuracy_score)
    return dict(
        F1=float(f1_score(y_true, y_pred, zero_division=0)),
        ROC_AUC=float(roc_auc_score(y_true, y_score)),
        PR_AUC=float(average_precision_score(y_true, y_score)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        accuracy=float(accuracy_score(y_true, y_pred)))


def t_ci(x, conf=0.95):
    from scipy import stats
    x = np.asarray(x, float)
    if len(x) < 2:
        return (float(x.mean()), float(x.mean()))
    lo, hi = stats.t.interval(conf, len(x) - 1, loc=x.mean(), scale=stats.sem(x))
    return float(lo), float(hi)


def boot_ci(x, n=10000, conf=0.95, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x, float)
    bs = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    a = (1 - conf) / 2
    return float(np.quantile(bs, a)), float(np.quantile(bs, 1 - a))


def compare_to_constant(rl, baseline):
    """RL per-seed scores vs a deterministic baseline constant: t-test, Wilcoxon, Cohen's d, CI."""
    from scipy import stats
    rl = np.asarray(rl, float); diff = rl - baseline
    t, p_t = stats.ttest_1samp(diff, 0.0)
    try:
        w, p_w = stats.wilcoxon(diff)
    except ValueError:
        w, p_w = np.nan, np.nan
    sd = rl.std(ddof=1)
    d = float(diff.mean() / sd) if sd > 0 else 0.0
    lo, hi = t_ci(rl)
    return dict(mean=float(rl.mean()), std=float(sd), margin=float(diff.mean()),
                ci_lo=lo, ci_hi=hi, p_ttest=float(p_t), p_wilcoxon=float(p_w),
                cohen_d=d, n=len(rl), seeds_won=int((rl > baseline).sum()))


def holm(pvals):
    """Holm-Bonferroni correction for a family of p-values."""
    try:
        from statsmodels.stats.multitest import multipletests
        rej, p_adj, *_ = multipletests(pvals, alpha=0.05, method="holm")
        return list(p_adj), list(rej)
    except Exception:                       # manual fallback
        p = np.asarray(pvals, float); order = np.argsort(p); m = len(p)
        adj = np.empty(m); run = 0.0
        for i, idx in enumerate(order):
            run = max(run, (m - i) * p[idx]); adj[idx] = min(run, 1.0)
        return list(adj), list(adj < 0.05)


def set_pub_style():
    """Vector-friendly publication figure style (Protocol #7): selectable text in PDFs."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "serif", "font.size": 9,
        "axes.linewidth": 0.6, "savefig.bbox": "tight", "savefig.dpi": 300})
