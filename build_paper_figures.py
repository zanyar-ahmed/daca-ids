"""Rebuild all 13 paper figures at 1:1 document scale.

Every figure is rendered at exactly the inch size it will occupy in the .docx,
so a point size specified here is the same physical size in every figure.
All values come from the paper's own tables (numbers hard-coded from Tables
VII-XIV as read out of the document; no re-computation, no substitution).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import numpy as np

CM = 2.54
W_FULL = 15.5 / CM          # 6.10 in  == full text width; NEVER exceed (Word would
                            #             rescale the image and shrink its fonts with it)
H_CAP  = 3.20               # in       == max height, keeps caption on the same page

BASE, TICK, LEG, TITLE, VAL = 10.0, 9.5, 9.5, 10.5, 8.5
NAVY, MID, RED, GOLD, GREEN, GREY = "#1f4e79", "#4a7fb5", "#c00000", "#bf8f00", "#4a7a3a", "#8a8a8a"
LBLUE, LRED = "#dbe6f3", "#f6dede"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": BASE, "axes.labelsize": BASE, "axes.titlesize": TITLE,
    "xtick.labelsize": TICK, "ytick.labelsize": TICK, "legend.fontsize": LEG,
    "figure.dpi": 400, "savefig.dpi": 400,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.linestyle": ":", "grid.linewidth": 0.6, "grid.color": "#c8c8c8",
    "axes.axisbelow": True, "legend.frameon": False,
    "lines.linewidth": 1.8, "lines.markersize": 6,
})

# ---- source values, read from the paper's tables -------------------------
T7  = [("P60", 0.900, 0.82, 1.00, 0.29), ("P85", 0.923, 0.893, 0.956, 0.152),
       ("P95", 0.830, 0.931, 0.748, 0.073), ("P99", 0.441, 0.972, 0.285, 0.011)]
T8  = [("Threshold\nP85", 0.922, NAVY), ("Threshold\nP95", 0.837, MID),
       ("PPO\ncontroller", 0.796, RED)]
T9  = [("Fixed threshold", 0.721, 0.721, RED), ("PPO, static error norm", 0.804, 0.820, NAVY),
       ("PPO, drift-adaptive norm", 0.808, 0.822, GREEN)]
T10 = [("Fixed threshold\n(P98)", 2789, 0.539, NAVY), ("RL\nbudget-aware", 2338, 0.452, RED)]
T11 = [("Cheap (basic only)", 1.0, 0.786, NAVY, "o"), ("Full (all groups)", 9.0, 0.819, GOLD, "s"),
       ("Tuned cascade (approx.)", 2.0, 0.811, GREEN, "D"), ("RL (degenerates to cheapest)", 1.0, 0.786, RED, "o")]
T12 = [("No response", -7.56, GREY), ("Block-on-error\n(tuned)", -5.09, NAVY),
       ("Two-level\n(current)", -4.81, NAVY), ("Two-level\n(recent-mean)", -4.54, NAVY),
       ("RL\n(learned)", -3.98, RED)]
T13 = [("Single run\n(Phase 6)", 0.56, None, None, "1/1", GREEN),
       ("Deterministic,\n10 seeds", -0.062, -0.097, -0.026, "1/10", RED)]
T14 = ([0.796, 0.775, np.nan], [0.789, 0.706, 0.995])
GEN = {"NSL-KDD": (0.952, 0.923), "UNSW-NB15": (0.859, 0.779), "CIC-IoT2023": (0.944, 0.938)}

PIPE = [("Normal traffic\nHTTP/S · DNS\nSSH · IoT", LBLUE, NAVY, False),
        ("Self-supervised\nautoencoder\n(normal only)", NAVY, "white", True),
        ("Reconstruction\nerror  e(x)", LBLUE, NAVY, False),
        ("Percentile tiers\nP60 · P85\nP95 · P99", LBLUE, NAVY, False),
        ("Alert tier\n(threshold\nor RL policy)", LRED, RED, False)]


def report(fig, name, outdir):
    """Save, then check text-text overlaps and record the doc size."""
    path = os.path.join(outdir, name + ".png")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    fig.canvas.draw()
    r = fig.canvas.renderer
    tx = [(t, t.get_window_extent(r)) for t in fig.findobj(matplotlib.text.Text)
          if t.get_text().strip() and t.get_visible()]
    ticks = set()
    for a in fig.axes:
        ticks |= set(a.get_xticklabels() + a.get_yticklabels())
    ov = [(a.get_text()[:18].replace("\n", "|"), b.get_text()[:18].replace("\n", "|"))
          for i, (a, ba) in enumerate(tx) for b, bb in tx[i + 1:]
          if ba.overlaps(bb) and not (a in ticks or b in ticks)]
    w, h = fig.get_size_inches()
    print(f"  {name:<8} {w*CM:5.1f} x {h*CM:5.1f} cm in doc | overlaps: {ov if ov else 'none'}")
    plt.close(fig)
    return w * CM


def fits_margin(fig, ax, boxes, margin_px=12):
    fig.canvas.draw()
    r = fig.canvas.renderer
    bad = []
    for t, (bx, by, bw, bh) in boxes:
        tb = t.get_window_extent(r)
        p0 = ax.transData.transform((bx, by))
        p1 = ax.transData.transform((bx + bw, by + bh))
        if not (tb.x0 >= min(p0[0], p1[0]) + margin_px and tb.x1 <= max(p0[0], p1[0]) - margin_px
                and tb.y0 >= min(p0[1], p1[1]) + margin_px and tb.y1 <= max(p0[1], p1[1]) - margin_px):
            bad.append(t.get_text()[:22])
    return bad


def build(outdir):
    os.makedirs(outdir, exist_ok=True)
    widths = {}

    # ---- Fig 1: pipeline (iterate figure width until labels clear the boxes)
    def pipeline(fs):
        BW, GAP, BH = 4.2, 0.60, 1.0
        fig, ax = plt.subplots(figsize=(W_FULL, 1.30))
        tot = len(PIPE) * BW + (len(PIPE) - 1) * GAP
        ax.set_xlim(-0.05, tot + 0.05); ax.set_ylim(0, BH + 0.16)
        ax.axis("off"); ax.grid(False)
        bx, x = [], 0.0
        for lbl, fc, tc, bd in PIPE:
            ax.add_patch(mp.FancyBboxPatch((x, 0.08), BW, BH,
                         boxstyle="round,pad=0.03,rounding_size=0.09", fc=fc, ec="#333", lw=1.0))
            t = ax.text(x + BW / 2, 0.08 + BH / 2, lbl, ha="center", va="center",
                        fontsize=fs, color=tc, fontweight="bold" if bd else "normal",
                        linespacing=1.42, zorder=5)
            bx.append((t, (x, 0.08, BW, BH)))
            if x + BW < tot - 0.1:
                ax.annotate("", xy=(x + BW + GAP - 0.07, 0.08 + BH / 2),
                            xytext=(x + BW + 0.07, 0.08 + BH / 2),
                            arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#333"))
            x += BW + GAP
        fig.tight_layout(pad=0.15); fig.canvas.draw()
        return fig, ax, bx
    fs = BASE
    for _ in range(14):
        fig, ax, bx = pipeline(fs)
        if not fits_margin(fig, ax, bx, margin_px=8):
            break
        plt.close(fig); fs -= 0.3
    print(f"  fig01 label font {fs:.1f} pt (fits at page width)")
    widths[1] = report(fig, "fig01", outdir)

    # ---- Fig 2: threshold-reducibility concept
    s = np.linspace(0, 1, 500)
    dh = 2.4 * s - 1.05
    df = 0.85 * np.sin(9.5 * s) + 0.5 * np.sin(3.1 * s + 1.0)
    tau = s[np.argmin(np.abs(dh))]
    cross = s[1:][np.diff(np.sign(df)) != 0]
    lo = min(dh.min(), df.min()) - 0.34
    hi = max(dh.max(), df.max()) + 0.22
    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 2.35))
    for ax, (dd, col, ttl) in zip(axes, [(dh, NAVY, "(a) Assumptions hold"),
                                         (df, GOLD, "(b) Assumptions fail")]):
        ax.axhline(0, color="#555", lw=0.9, ls=":")
        ax.plot(s, dd, color=col, lw=2.0)
        ax.fill_between(s, 0, dd, where=(dd >= 0), color=col, alpha=0.14, lw=0)
        ax.set_xlabel("Anomaly score  $s(x)$"); ax.set_ylim(lo, hi); ax.set_xlim(0, 1)
        ax.set_yticks([]); ax.set_xticks([0, 0.5, 1])
        ax.set_title(ttl, loc="left", fontsize=TITLE); ax.grid(False)
    axes[0].axvline(tau, color=RED, ls="--", lw=1.3)
    axes[0].annotate(r"$\tau^*$", xy=(tau, 0), xytext=(tau + 0.12, -0.72), fontsize=BASE + 1,
                     color=RED, arrowprops=dict(arrowstyle="-", lw=0.9, color=RED))
    axes[0].text(0.03, hi - 0.08, "alert preferred", fontsize=VAL, color=NAVY, va="top")
    axes[0].text(0.03, lo + 0.20, "silence preferred", fontsize=VAL, color="#555", va="center")
    axes[0].set_ylabel(r"$\Delta(s)=r(1,s)-r(0,s)$", fontsize=BASE - 0.5)
    for c_ in cross:
        axes[1].axvline(c_, color=RED, ls="--", lw=0.9, alpha=0.7)
    axes[1].text(0.5, lo + 0.20, f"{len(cross)} sign changes — no single $\\tau$ optimal",
                 fontsize=VAL, color=RED, ha="center", va="center")
    fig.tight_layout()
    widths[2] = report(fig, "fig02", outdir)

    # ---- Fig 3: Table VII tiers
    fig, ax = plt.subplots(figsize=(W_FULL, H_CAP))
    xg = np.arange(4); w = 0.20
    for k, (lab, idx, c) in enumerate([("F1", 1, NAVY), ("Precision", 2, MID),
                                       ("Recall", 3, GREEN), ("FPR", 4, RED)]):
        vals = [r[idx] for r in T7]
        b = ax.bar(xg + (k - 1.5) * w, vals, width=w, color=c, label=lab)
        # Only the F1 series carries value labels: 16 labels cannot be set legibly at
        # page width, and Table VII on the same page gives every value exactly.
        if lab == "F1":
            for bb, vv in zip(b, vals):
                ax.text(bb.get_x() + bb.get_width() / 2, vv + 0.028, f"{vv:.3f}",
                        ha="center", fontsize=VAL, color=c)
    ax.set_xticks(xg); ax.set_xticklabels([r[0] for r in T7])
    ax.set_xlabel("Percentile threshold tier"); ax.set_ylabel("Metric value")
    ax.set_ylim(0, 1.22); ax.grid(axis="x", visible=False)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.14),
              handlelength=1.3, columnspacing=1.6)
    fig.tight_layout()
    widths[3] = report(fig, "fig03", outdir)

    # ---- Fig 4: empirical ROC through the Table VII operating points (square)
    pts = sorted([(r[4], r[3], r[0]) for r in T7])
    fx = [0.0] + [p[0] for p in pts] + [1.0]
    fy = [0.0] + [p[1] for p in pts] + [1.0]
    fig, ax = plt.subplots(figsize=(H_CAP, H_CAP))
    ax.plot([0, 1], [0, 1], ls=":", lw=0.9, color=GREY, label="chance")
    ax.plot(fx, fy, "-", color=NAVY, marker="o", markerfacecolor="white",
            markeredgecolor=NAVY, markeredgewidth=1.5, label="AE (ROC-AUC = 0.952)")
    for fpr, tpr, nm in pts:
        ax.plot([fpr], [tpr], "o", color=RED, ms=5.5, zorder=5)
        ax.annotate(nm, xy=(fpr, tpr), xytext=(8, -10), textcoords="offset points",
                    fontsize=VAL, color=RED, fontweight="bold")
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.05)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate (recall)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    widths[4] = report(fig, "fig04", outdir)

    # ---- Fig 5: Table VIII static detection
    fig, ax = plt.subplots(figsize=(3.10, H_CAP))
    b = ax.bar([x[0] for x in T8], [x[1] for x in T8], color=[x[2] for x in T8], width=0.55)
    for bb, x in zip(b, T8):
        ax.text(bb.get_x() + bb.get_width() / 2, x[1] + 0.006, f"{x[1]:.3f}",
                ha="center", fontsize=VAL)
    ax.set_ylabel("F1-score"); ax.set_ylim(0.60, 0.97); ax.grid(axis="x", visible=False)
    fig.tight_layout()
    widths[5] = report(fig, "fig05", outdir)

    # ---- Fig 6: Table IX drift
    fig, ax = plt.subplots(figsize=(W_FULL, H_CAP))
    xg2 = np.arange(2); w2 = 0.24
    for k, (lab, a, bv, c) in enumerate(T9):
        vals = [a, bv]
        bb = ax.bar(xg2 + (k - 1) * w2, vals, width=w2, color=c, label=lab)
        for r_, vv in zip(bb, vals):
            ax.text(r_.get_x() + r_.get_width() / 2, vv + 0.004, f"{vv:.3f}",
                    ha="center", fontsize=VAL, color=c)
    ax.set_xticks(xg2); ax.set_xticklabels(["drift severity 1.0", "drift severity 3.0"])
    ax.set_ylabel("F1 (high-drift portion)"); ax.set_ylim(0.68, 0.875)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", handlelength=1.3, ncol=1)
    fig.tight_layout()
    widths[6] = report(fig, "fig06", outdir)

    # ---- Fig 7: Table X budget
    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, H_CAP))
    ax = axes[0]
    b = ax.bar([x[0] for x in T10], [x[1] for x in T10], color=[x[3] for x in T10], width=0.5)
    for bb, x in zip(b, T10):
        ax.text(bb.get_x() + bb.get_width() / 2, x[1] + 45, f"{x[1]:,}", ha="center", fontsize=VAL)
    ax.set_ylabel("Attacks caught (budget 3,000)"); ax.set_ylim(0, 3250)
    ax.set_title("(a) Attacks caught", loc="left"); ax.grid(axis="x", visible=False)
    ax = axes[1]
    b = ax.bar([x[0] for x in T10], [x[2] for x in T10], color=[x[3] for x in T10], width=0.5)
    for bb, x in zip(b, T10):
        ax.text(bb.get_x() + bb.get_width() / 2, x[2] + 0.008, f"{x[2]:.3f}", ha="center", fontsize=VAL)
    ax.set_ylabel("F1-score"); ax.set_ylim(0, 0.63)
    ax.set_title("(b) F1-score", loc="left"); ax.grid(axis="x", visible=False)
    fig.tight_layout()
    widths[7] = report(fig, "fig07", outdir)

    # ---- Fig 8: Table XI cost-aware
    fig, ax = plt.subplots(figsize=(W_FULL, H_CAP))
    for lab, c_, f_, col, mk in T11:
        rl = lab.startswith("RL")
        ax.plot([c_], [f_], mk, color=col, ms=11 if rl else 8,
                markerfacecolor="none" if rl else col,
                markeredgewidth=2.2 if rl else 1.0, label=lab, zorder=4)
    ax.annotate("RL lands on the cheapest option", xy=(1.0, 0.786), xytext=(2.5, 0.7805),
                fontsize=VAL, color=RED, va="center", ha="left",
                arrowprops=dict(arrowstyle="-", lw=0.9, color=RED, shrinkA=0, shrinkB=8))
    ax.set_xlabel("Average inspection cost"); ax.set_ylabel("F1-score")
    ax.set_xlim(0, 10.6); ax.set_ylim(0.775, 0.833)
    ax.legend(loc="upper left", handlelength=1.1, labelspacing=0.3)
    fig.tight_layout()
    widths[8] = report(fig, "fig08", outdir)

    # ---- Fig 9: Table XII response
    fig, ax = plt.subplots(figsize=(W_FULL, H_CAP))
    b = ax.bar([x[0] for x in T12], [x[1] for x in T12], color=[x[2] for x in T12], width=0.58)
    for bb, x in zip(b, T12):
        ax.text(bb.get_x() + bb.get_width() / 2, x[1] - 0.30, f"{x[1]:.2f}", ha="center", fontsize=VAL)
    ax.set_ylabel("Mean episode reward"); ax.set_ylim(-8.9, 0.4)
    ax.axhline(0, color="#333", lw=0.9); ax.grid(axis="x", visible=False)
    fig.tight_layout()
    widths[9] = report(fig, "fig09", outdir)

    # ---- Fig 10: Table XIII robustness
    fig, ax = plt.subplots(figsize=(3.55, H_CAP))
    for i, (lab, m_, lo_, hi_, sw, col) in enumerate(T13):
        ax.bar([i], [m_], width=0.42, color=col)
        if lo_ is not None:
            ax.errorbar([i], [m_], yerr=[[m_ - lo_], [hi_ - m_]], fmt="none",
                        ecolor="#222", lw=1.2, capsize=4, zorder=5)
        if m_ > 0:
            ax.text(i, m_ + 0.030, f"{m_:+.2f}", ha="center", va="bottom", fontsize=VAL, color=col)
            ax.text(i, -0.055, f"won {sw}", ha="center", va="top", fontsize=VAL - 0.5, color="#444")
        else:
            ax.text(i - 0.30, m_ - 0.012, f"{m_:+.3f}", ha="right", va="top", fontsize=VAL, color=col)
            ax.text(i, -0.165, f"won {sw}", ha="center", va="top", fontsize=VAL - 0.5, color="#444")
    ax.axhline(0, color="#333", lw=0.9)
    ax.set_xticks(np.arange(2)); ax.set_xticklabels([x[0] for x in T13])
    ax.set_ylabel("RL margin vs best baseline")
    ax.set_ylim(-0.24, 0.72); ax.set_xlim(-0.6, 1.6); ax.grid(axis="x", visible=False)
    fig.tight_layout()
    widths[10] = report(fig, "fig10", outdir)

    # ---- Fig 11: three-dataset generalisation
    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, H_CAP))
    names = list(GEN); cols = [NAVY, GOLD, GREEN]; xg4 = np.arange(3)
    for ax, (ttl, idx, lab) in zip(axes, [("(a) Detector ROC-AUC", 0, "ROC-AUC"),
                                          ("(b) Detector best-tier F1", 1, "Best-tier F1")]):
        vals = [GEN[n][idx] for n in names]
        b = ax.bar(xg4, vals, color=cols, width=0.58)
        for bb, vv in zip(b, vals):
            ax.text(bb.get_x() + bb.get_width() / 2, vv + 0.016, f"{vv:.3f}", ha="center", fontsize=VAL)
        ax.set_xticks(xg4)
        ax.set_xticklabels([n.replace("-", "-\n") for n in names], fontsize=TICK - 1.0)
        ax.set_ylim(0, 1.13); ax.set_ylabel(lab); ax.set_title(ttl, loc="left")
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    widths[11] = report(fig, "fig11", outdir)

    # ---- Fig 12: Table XIV UNSW
    thr, rl = T14
    fig, ax = plt.subplots(figsize=(W_FULL, H_CAP))
    xg3 = np.arange(3); w3 = 0.32
    b1 = ax.bar(xg3 - w3 / 2, thr, width=w3, color=NAVY, label="F1-optimal threshold")
    b2 = ax.bar(xg3 + w3 / 2, rl, width=w3, color=RED, label="RL controller")
    for bb, vv in zip(b1, thr):
        if not np.isnan(vv):
            ax.text(bb.get_x() + bb.get_width() / 2, vv + 0.012, f"{vv:.3f}",
                    ha="center", fontsize=VAL, color=NAVY)
    for bb, vv in zip(b2, rl):
        ax.text(bb.get_x() + bb.get_width() / 2, vv + 0.012, f"{vv:.3f}",
                ha="center", fontsize=VAL, color=RED)
    ax.text(xg3[2] - w3 / 2, 0.02, "not\nreported", ha="center", va="bottom",
            fontsize=VAL - 1, color=GREY, style="italic")
    ax.set_xticks(xg3); ax.set_xticklabels(["F1", "Accuracy", "Recall"])
    ax.set_ylabel("Metric value"); ax.set_ylim(0, 1.16); ax.grid(axis="x", visible=False)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.14),
              handlelength=1.3, columnspacing=1.8)
    fig.tight_layout()
    widths[12] = report(fig, "fig12", outdir)

    # ---- Fig 13: decision logic
    def decision(fs):
        fig, ax = plt.subplots(figsize=(W_FULL, W_FULL * 0.56))
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off"); ax.grid(False)
        bx = []
        def box(cx, cy, w_, h_, txt, fc, tc, bd):
            ax.add_patch(mp.FancyBboxPatch((cx - w_ / 2, cy - h_ / 2), w_, h_,
                         boxstyle="round,pad=0.05,rounding_size=0.16", fc=fc, ec="#333", lw=1.0))
            t = ax.text(cx, cy, txt, ha="center", va="center", fontsize=fs, color=tc,
                        fontweight="bold" if bd else "normal", linespacing=1.42, zorder=5)
            bx.append((t, (cx - w_ / 2, cy - h_ / 2, w_, h_)))
        box(5.0, 8.75, 6.1, 1.35, "Strong scalar anomaly score s(x)\navailable for the decision?",
            LBLUE, NAVY, False)
        box(5.0, 5.70, 6.9, 1.85,
            "Is the decision genuinely sequential?\n(delayed consequences, graded actions,\nadaptive adversary)",
            LBLUE, NAVY, False)
        box(2.30, 1.45, 4.35, 1.35, "Percentile threshold on s(x)\n(RL not warranted)", NAVY, "white", True)
        box(7.70, 1.45, 4.35, 1.35, "Reinforcement learning\n(adaptive sequential response)", LRED, RED, True)
        ax.annotate("", xy=(5.0, 6.70), xytext=(5.0, 8.00),
                    arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#333"))
        ax.text(5.22, 7.30, "yes", fontsize=VAL, style="italic", color="#333", ha="left", va="center")
        ax.annotate("", xy=(2.30, 2.20), xytext=(4.20, 4.71),
                    arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#333"))
        ax.text(2.55, 3.55, "no", fontsize=VAL, style="italic", color="#333", ha="right", va="center")
        ax.annotate("", xy=(7.70, 2.20), xytext=(5.80, 4.71),
                    arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#333"))
        ax.text(7.45, 3.55, "yes", fontsize=VAL, style="italic", color="#333", ha="left", va="center")
        fig.tight_layout(pad=0.15); fig.canvas.draw()
        return fig, ax, bx
    fs13 = BASE
    for _ in range(14):
        fig, ax, bx = decision(fs13)
        if not fits_margin(fig, ax, bx, margin_px=8):
            break
        plt.close(fig); fs13 -= 0.3
    print(f"  fig13 label font {fs13:.1f} pt (fits at page width)")
    widths[13] = report(fig, "fig13", outdir)
    return widths


if __name__ == "__main__":
    import json, sys
    out = sys.argv[1]
    w = build(out)
    json.dump(w, open(os.path.join(out, "widths_cm.json"), "w"))
    print("\nmax height:", f"{H_CAP*CM:.1f} cm (cap) | all 13 built")
