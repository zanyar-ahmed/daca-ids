# Experimental Protocol — RL-IDS Paper (high-impact journal submission)

Target venues: IEEE TDSC / IEEE TNSM / ACM TOPS. These review negative-results papers
hardest exactly where your synthetic simulator and your seed counts sit, so the protocol
below is built to remove those objections. Work top-to-bottom; the cheap ones (1, 3, 7)
can be done in an afternoon, the rest over a few days.

| # | Experiment | Why it matters | Effort | Feeds (manuscript) |
|---|-----------|----------------|--------|--------------------|
| 0 | Shared evaluation harness | Makes everything comparable & deterministic | 0.5 d | Methods + Reproducibility |
| 1 | Trivial-learner battery (LR + optimal-τ on e) | Direct empirical proof of Proposition 1 | 2–3 h | New Table B, next to Prop. 1 |
| 2 | Per-seed variance + 95% CI for ALL RL phases | The rigor reviewers demand | 1–2 d | New Table A + boxplot figure |
| 3 | Statistical tests + effect sizes | Closes "add stats" comment | 1 h | Table A columns |
| 4 | RL hyperparameter sensitivity | Shows negative result is robust | 0.5–1 d | New Table C |
| 5 | Adaptive-adversary + cost-ratio variants (Phase 6) | #1 reviewer vulnerability | 2–4 d | New Table D + revised §VI |
| 6 | CICIDS2017 third dataset | Third generalization point | 1–2 d | Extends Table A / Phase 8 |
| 7 | Vector figures | Free polish journals like | 0.5–2 h | All result figures |
| 8 | Reproducibility package | Required at this tier | 0.5 d | Reproducibility statement |

---

## 0. Shared evaluation harness (do this first)

Standardize so every number is comparable and the **detector is deterministic** (the Phase 7
lesson: the only randomness should be the RL seed).

**Seeds.** Use a fixed list of **15 seeds** (more than the 10 you have; journals like ≥10–15):
```python
SEEDS = list(range(15))   # [0..14]; reuse the SAME list for every phase
```

**Full determinism** (call at the start of every run):
```python
import os, random, numpy as np, torch
def set_all_seeds(s):
    os.environ["PYTHONHASHSEED"] = str(s)
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# also pass seed= to PPO(...) and to your gym env reset(seed=s)
```
Train the autoencoder **once with a fixed seed** and reuse that frozen detector everywhere,
so AE randomness can never leak into the RL comparison again.

**Metrics** — report all of these every time (PR-AUC matters under class imbalance):
```python
from sklearn.metrics import (f1_score, roc_auc_score, average_precision_score,
                             precision_score, recall_score)
def metrics(y_true, y_pred, y_score):
    return dict(
        F1=f1_score(y_true, y_pred),
        ROC_AUC=roc_auc_score(y_true, y_score),
        PR_AUC=average_precision_score(y_true, y_score),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0))
```

**CI helpers** (use the t-interval for seed means; bootstrap for single-run metrics):
```python
from scipy import stats
def t_ci(x, conf=0.95):
    x = np.asarray(x, float)
    return stats.t.interval(conf, len(x)-1, loc=x.mean(), scale=stats.sem(x))

def boot_ci(x, n=10000, conf=0.95, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x, float)
    bs = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    a = (1-conf)/2
    return np.quantile(bs, [a, 1-a])
```

Save **one CSV per experiment** (`results/<exp>.csv`) with raw per-seed rows — send me those.

---

## 1. Trivial-learner battery  ⭐ (cheapest high-value experiment)

**Goal.** Show a one-parameter learner on the single reconstruction-error feature `e(x)`
matches the tuned threshold and equals or beats PPO — i.e., empirical Proposition 1.

**Design.** Use the frozen AE to get `e` for every flow. Give the trivial learners the **same
supervision the RL agent receives** (the labelled split used for rewards). Two learners:

```python
from sklearn.linear_model import LogisticRegression

# e_tr,y_tr : error + labels on the labelled (reward) split
# e_va,y_va : validation split   |   e_te,y_te : test split
# (a) Logistic regression on the single feature e  -> a smooth threshold
lr = LogisticRegression(class_weight="balanced", max_iter=2000)
lr.fit(e_tr.reshape(-1,1), y_tr)
p_te = lr.predict_proba(e_te.reshape(-1,1))[:,1]
print("LR-on-e :", metrics(y_te, (p_te>=0.5).astype(int), p_te))

# (b) Single learned threshold tau* (pick on validation to max F1) -> 1 parameter
taus = np.quantile(e_va, np.linspace(0.50, 0.999, 300))
f1s  = [f1_score(y_va, (e_va>=t).astype(int)) for t in taus]
tau_star = taus[int(np.argmax(f1s))]
print("Best-tau-on-e :", metrics(y_te, (e_te>=tau_star).astype(int), e_te))
```

Run on **NSL-KDD and UNSW-NB15**.

**Report → new Table B** (place right after Proposition 1):

| Decision rule on the AE error | params | NSL-KDD F1 / ROC-AUC | UNSW F1 / ROC-AUC |
|---|---|---|---|
| Percentile threshold (unsupervised) | 1 | ~0.92 / 0.952 | ~0.78 / 0.859 |
| **Logistic regression on e** | 2 | … | … |
| **Best learned threshold on e** | 1 | … | … |
| PPO (latent z + e) | ~10⁴ | 0.796 / … | … |

Expected story: the 1–2 parameter learners sit at the threshold level and **≥ PPO**, despite
PPO having orders of magnitude more parameters. That single table is the most persuasive
evidence in the paper for the threshold-reducibility claim.

---

## 2. Per-seed variance + 95% CI for ALL RL phases  ⭐

**Goal.** Extend the deterministic 10-seed treatment (currently only Phase 6→7) to **every**
phase that reports an RL number (Phases 2, 3, 4, 5, 6, and the RL parts of Phase 8).

**Harness skeleton** (you already have the Phase 7 version — generalize it):
```python
rows = []
for phase in PHASES:                       # e.g. ["P2_static","P3_drift","P4_budget","P5_cost","P6_resp"]
    base = run_baseline(phase)             # deterministic -> single float (F1)
    rl = []
    for s in SEEDS:
        set_all_seeds(s)
        rl.append(run_rl(phase, seed=s))   # returns test F1 for that seed
    rl = np.array(rl)
    lo, hi = t_ci(rl)
    rows.append(dict(phase=phase, baseline=base,
                     rl_mean=rl.mean(), rl_std=rl.std(ddof=1),
                     ci_lo=lo, ci_hi=hi, margin=rl.mean()-base,
                     n_wins=int((rl>base).sum()), n=len(SEEDS)))
import pandas as pd; pd.DataFrame(rows).to_csv("results/per_phase_seeds.csv", index=False)
```

**Report → new Table A** (the backbone results table):

| Phase | Baseline F1 | RL mean ± std | RL 95% CI | Margin (RL−base) | Seeds won | p | Cohen's d |
|------|-------------|---------------|-----------|------------------|-----------|---|-----------|
| 2 static | 0.922 | … | […, …] | … | k/15 | … | … |
| 3 drift | … | … | … | … | … | … | … |
| 4 budget | … | … | … | … | … | … | … |
| 5 cost | … | … | … | … | … | … | … |
| 6 response | … | … | … | … | … | … | … |

**Report → new boxplot figure (supplement):** one box per phase of the per-seed RL margin,
with a line at 0. Save as **vector** (see §7).

---

## 3. Statistical tests + effect sizes  ⭐ (cheap)

For each phase, the baseline is deterministic (a constant), so test the **RL distribution vs
that constant** (one-sample), and compute Cohen's d:
```python
def compare_to_constant(rl, baseline):
    rl = np.asarray(rl, float); diff = rl - baseline
    t,  p_t = stats.ttest_1samp(diff, 0.0)
    try:    w, p_w = stats.wilcoxon(diff)        # nonparametric
    except ValueError: w, p_w = np.nan, np.nan   # all-zero diffs
    d = diff.mean() / rl.std(ddof=1)             # one-sample Cohen's d
    return dict(p_ttest=p_t, p_wilcoxon=p_w, cohen_d=d,
                ci=t_ci(rl), mean=rl.mean(), margin=diff.mean())
```
If you ever compare **two stochastic methods**, use the paired versions instead:
`stats.ttest_rel(a, b)` and `stats.wilcoxon(a, b)`.

**Multiple comparisons.** With ~6 phases, correct the family of p-values:
```python
from statsmodels.stats.multitest import multipletests
reject, p_adj, *_ = multipletests(p_values, alpha=0.05, method="holm")
```
State in the paper that significance is reported after **Holm correction**. These p/d values
fill the last two columns of Table A.

---

## 4. RL hyperparameter sensitivity

**Goal.** Show the negative result is not an artefact of one unlucky hyperparameter choice.

**Grid** (run on the cheap static framing **Phase 2** and the interesting **Phase 6**):
- learning rate ∈ {1e-4, 3e-4, 1e-3}
- network (pi/vf) ∈ {[64,64], [128,64,32], [256,128]}
- (optional) clip range ∈ {0.1, 0.2, 0.3}

3×3 = 9 configs × **5 seeds** each (5 is enough for a sweep). For each config report mean ± std F1
and the margin to the threshold.

```python
from itertools import product
grid = []
for lr_, net in product([1e-4,3e-4,1e-3], [[64,64],[128,64,32],[256,128]]):
    f1 = []
    for s in SEEDS[:5]:
        set_all_seeds(s)
        f1.append(run_rl("P2_static", seed=s, learning_rate=lr_,
                         policy_kwargs=dict(net_arch=net)))
    f1 = np.array(f1)
    grid.append(dict(lr=lr_, net=str(net), f1_mean=f1.mean(),
                     f1_std=f1.std(ddof=1), margin=f1.mean()-BASELINE_P2))
pd.DataFrame(grid).to_csv("results/hparam_P2.csv", index=False)
```

**Report → new Table C:** config × (F1 mean ± std, margin). Message: across the entire grid,
RL never exceeds the threshold → robustness of the negative result.

---

## 5. Adaptive-adversary + cost-ratio variants (Phase 6)  ⭐⭐ (highest scientific value)

This is the experiment that decides a top-tier outcome, because the synthetic, **static**
simulator is the single thing a reviewer will attack. You need to show the Phase-6/7 conclusion
holds (or sharpens) when the environment is harder and not hand-tuned.

**Variant A — Adaptive adversary.** Give the attacker a reactive policy instead of a fixed
campaign. Minimal, defensible design inside your existing gym env:
- Attacker hidden state = `progress ∈ [0,1]` (foothold toward compromise).
- Each step, the attacker observes the defender's recent action (monitor/throttle/block) and:
  - if throttled/blocked in the last *k* steps → **evade**: lower its anomaly signature
    (shift its reconstruction-error draw down by ε) for a cooldown window;
  - if unmonitored/allowed → **escalate**: increase `progress` faster.
- Expose ε (evasion strength) and the escalation rate as parameters; sweep ε ∈ {0, 0.1, 0.2, 0.3}.

**Variant B — Cost-ratio sweep.** Vary the breach-vs-availability tradeoff in the reward:
`cost_ratio = breach_cost / availability_cost ∈ {0.25, 0.5, 1, 2, 4}`. RL's apparent edge in
sequential settings is usually sensitive to this; showing the margin across the sweep is exactly
the parameter-sensitivity the reviewer asked for.

For **each (variant, parameter)** run RL **and** the best non-RL policy (your tuned threshold/
heuristic responder) across the 15 seeds; report margin + 95% CI + seeds-won, same columns as
Table A.

**Report → new Table D** and a short rewrite of the Limitations/Discussion:

| Variant | Param | RL margin (mean ± CI) | Seeds won | Verdict |
|---|---|---|---|---|
| Static (current) | — | −0.062 [−0.097,−0.027] | 1/10 | RL loses |
| Adaptive adversary | ε=0.1 | … | … | … |
| Adaptive adversary | ε=0.3 | … | … | … |
| Cost ratio | 0.25 | … | … | … |
| Cost ratio | 4 | … | … | … |

Both outcomes are publishable and you cannot lose:
- **RL still loses** → your thesis now holds even under an adaptive adversary — much stronger.
- **RL wins once the adversary adapts** → that is the *positive* half of your story ("RL earns
  its place precisely when the environment is genuinely adaptive/sequential"), which completes
  the paper rather than weakening it.

**Optional gold standard** (if you have time): port the response setting onto a published
benchmark such as **CybORG / CAGE Challenge** so the dynamics are not your own. This essentially
eliminates the "synthetic simulator" criticism. Bigger lift; only if you're chasing TDSC.

---

## 6. CICIDS2017 third dataset

**Goal.** A third, modern generalization point (you already list it as future work — doing it
upgrades the paper).

- Train the AE on **benign flows only**; evaluate detection + the static RL-vs-threshold test,
  same protocol as NSL-KDD/UNSW.
- **Preprocessing caution:** the original CICIDS2017 has known label/duplicate problems — cite
  and use a cleaned release (e.g., the "Improved CICIDS2017" / WTMC-2021 corrected version) and
  say so explicitly; reviewers know this dataset's issues. Drop socket/IP/timestamp identifier
  columns, handle Inf/NaN, standardize on the train split only.

**Report:** add a CICIDS2017 row to Table A (AE F1/ROC-AUC + RL margin). Expectation: AE in the
high-0.8s/low-0.9s, RL margin ≤ 0 again → "the pattern holds on a third dataset."

---

## 7. Vector figures  (cheap)

Re-save every result figure as PDF (and/or SVG) with embedded editable text:
```python
import matplotlib as mpl
mpl.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,     # TrueType -> selectable text in the PDF
    "font.family": "serif", "font.size": 9,
    "axes.linewidth": 0.6, "savefig.bbox": "tight", "savefig.dpi": 300})
# fig.savefig("figs/phaseX.pdf")   # and .svg if the journal wants it
```
Ensure each caption states **n seeds** and that error bars are **95% CI** (add error bars to any
bar/line plot that currently shows a single run).

---

## 8. Reproducibility package (required at this tier)

- Public repo (GitHub + an archived **Zenodo DOI** — journals increasingly want the DOI).
- `requirements.txt` *and* a `Dockerfile` (or `environment.yml`) pinning versions
  (python, torch, stable-baselines3, gymnasium, scikit-learn, numpy).
- `reproduce.sh` that regenerates **each table and figure** from raw scripts with the fixed seeds.
- A top-level `README` mapping every paper figure/table → the script that makes it.
- **Then tell me the repo URL / DOI** and I'll insert it into the Reproducibility paragraph
  (currently it says the scripts "are released" — it needs your actual link/footnote).

---

## 9. What to send back to me (so I can fold results into the manuscript)

For each experiment, the raw CSV plus the summary number(s). Concretely I need:

1. **Table B** (trivial learners) — LR-on-e and best-τ-on-e F1/ROC-AUC on NSL-KDD + UNSW.
2. **Table A** (per-phase seeds) — the `results/per_phase_seeds.csv` (raw per-seed F1) so I can
   compute means/CIs/tests and write the prose.
3. **Table C** (hparam sweep) — `results/hparam_*.csv`.
4. **Table D** (adaptive adversary + cost ratio) — per-seed CSV per variant/param.
5. **CICIDS2017** — AE metrics + RL margin (+ which cleaned version you used).
6. **Figures** — the regenerated vector PDFs (boxplots + updated result plots).
7. **Repo URL / Zenodo DOI.**

Once those arrive I will: add Tables A–D and the boxplot figure, rewrite the Phase-6 limitation
around the adaptive-adversary result, update the abstract's claim to "three datasets," wire the
new statistics into the text, and re-render the full IEEE PDF.

---

### Minimum viable set (if time gets short)
Do **1, 2, 3, 7** — trivial-learner battery, all-phase seeds+CIs, stats, vector figures. That
alone answers most reviewer objections. Items **5** (adaptive adversary) and **6** (CICIDS2017)
are what separate a strong submission from a borderline one at TDSC/TNSM, so prioritize **5**
next if you can.
