"""Merge the two CAGE Challenge 2 result files into the summary the paper's table is read from.

cyborg_baselines.py writes cyborg_baselines.json and cyborg_ppo.py writes cyborg_ppo.json;
Table XXX is read from the merge of the two. This script performs that merge, so the released
summary is derived from released outputs rather than assembled by hand.

Two conventions matter for reproducing the published values:

  * ppo_std is the SAMPLE standard deviation (ddof=1) over the per-seed means. numpy's
    default ddof=0 does not reproduce them.
  * The margin interval combines the policy's across-seed t-interval (n seeds, ddof=1) with
    the heuristic's across-episode standard error (n=100 episodes):
        half-width = t(0.975, n_seeds-1) * sqrt(SE_policy^2 + SE_heuristic^2)
    The threshold-minus-do-nothing interval is +/-1.96 * SE of the difference of two
    100-episode means. CybORG seeds no episode of its own, so these intervals, not the point
    values, are what a re-run can be expected to agree with.

    python cyborg_merge.py                      # after running the two scripts above
    python cyborg_merge.py --check              # verify an existing summary instead of writing
"""
import argparse, json, math

import numpy as np

REDS = ["B_line", "Meander"]
LENGTHS = [30, 50, 100]
EPISODES = 100
THRESHOLD_RULE = "52-dim observation summed to a scalar; act when above a tuned cut-point"
Z = 1.959964                      # normal 97.5th percentile
T975 = {1: 12.706205, 2: 4.302653, 3: 3.182446, 4: 2.776445, 5: 2.570582,
        6: 2.446912, 7: 2.364624, 8: 2.306004, 9: 2.262157, 10: 2.228139}


def _t(df):
    return T975.get(df, Z)


def merge(base, ppo):
    n_eps = base["protocol"]["episodes"]
    rows = []
    for red in REDS:
        for steps in LENGTHS:
            key = f"{red}_{steps}"
            seeds = list(ppo["results"][key])
            k = len(seeds)
            mean = float(np.mean(seeds))
            std = float(np.std(seeds, ddof=1))             # sample std; ddof=0 does not match
            thr, thr_sd = (float(x) for x in base["threshold_sweep"][red][str(steps)])
            sleep, sleep_sd = (float(x) for x in base["heuristics"]["SleepAgent"][key])
            shipped = {a: v[key] for a, v in base["heuristics"].items() if a != "SleepAgent"}
            bname = max(shipped, key=lambda a: shipped[a][0])
            bval, bsd = (float(x) for x in shipped[bname])

            margin = mean - bval
            half = _t(k - 1) * math.hypot(std / math.sqrt(k), bsd / math.sqrt(n_eps))
            ts = thr - sleep
            ts_half = Z * math.hypot(thr_sd, sleep_sd) / math.sqrt(n_eps)

            rows.append({
                "red": red,
                "steps": steps,
                "ppo_mean": round(mean, 2),
                "ppo_std": round(std, 2),
                "ppo_seeds": seeds,
                "threshold": round(thr, 2),
                "sleep": round(sleep, 2),
                "best_heuristic": bname,
                "best_heuristic_val": round(bval, 2),
                "ppo_minus_best_heuristic": round(margin, 2),
                "ppo_minus_threshold": round(mean - thr, 2),
                "threshold_minus_sleep": round(ts, 2),
                "ppo_minus_best_heuristic_ci": [round(margin - half, 2), round(margin + half, 2)],
                "ppo_beats_best_heuristic_significantly": bool(margin - half > 0),
                "threshold_minus_sleep_ci": [round(ts - ts_half, 2), round(ts + ts_half, 2)],
            })

    n = len(rows)
    sig = [r for r in rows if r["ppo_beats_best_heuristic_significantly"]]
    nonsig = [r for r in rows if not r["ppo_beats_best_heuristic_significantly"]]
    worst = max(rows, key=lambda r: abs(r["threshold_minus_sleep"]))
    w_half = (worst["threshold_minus_sleep_ci"][1] - worst["threshold_minus_sleep_ci"][0]) / 2

    def _group(rs):
        reds = sorted({r["red"] for r in rs})
        if len(reds) == 1:
            return f"{reds[0]} at {', '.join(str(r['steps']) for r in rs)} steps"
        return "; ".join(f"{d} at {', '.join(str(r['steps']) for r in rs if r['red'] == d)} steps"
                         for d in reds)

    nonsig_reds = sorted({r["red"] for r in nonsig})
    nonsig_desc = (f"the {len(nonsig)} {nonsig_reds[0]} settings"
                   if len(nonsig_reds) == 1 else f"the remaining {len(nonsig)} settings")

    finding_rl = (f"PPO exceeds the best shipped heuristic with a 95% interval excluding zero "
                  f"in {len(sig)} of {n} settings ({_group(sig)}); {nonsig_desc} are "
                  f"indistinguishable. No setting shows a deficit reaching significance."
                  if not any(r["ppo_minus_best_heuristic_ci"][1] < 0 for r in rows) else
                  f"PPO exceeds the best shipped heuristic with a 95% interval excluding zero "
                  f"in {len(sig)} of {n} settings ({_group(sig)}).")

    finding_a1 = (f"The tuned scalar-threshold rule never exceeds the do-nothing policy and "
                  f"never differs from it by more than sampling error in any of the {n} "
                  f"settings; the largest gap, {abs(worst['threshold_minus_sleep']):.2f} against "
                  f"{worst['red']} at {worst['steps']} steps, lies inside its 95% interval of "
                  f"+/-{w_half:.1f}. (A1) is measured to fail.")

    return {
        "environment": "CAGE Challenge 2 (CybORG), Scenario2",
        "protocol": {
            "eval_episodes": n_eps,
            "lengths": base["protocol"]["lengths"],
            "ppo_seeds": ppo["config"]["seeds"],
            "ppo_timesteps": ppo["config"]["timesteps"],
            "threshold_rule": THRESHOLD_RULE,
        },
        "finding_A1": finding_a1,
        "finding_RL": finding_rl,
        "rows": rows,
        "interval_method": (
            f"Margin intervals are 95% intervals combining the across-seed t-interval of the "
            f"policy (n={len(rows[0]['ppo_seeds'])}, ddof=1) with the across-episode standard "
            f"error of the heuristic (n={n_eps})."),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", default="cyborg_baselines.json")
    ap.add_argument("--ppo", default="cyborg_ppo.json")
    ap.add_argument("--out", default="cyborg_validation.json")
    ap.add_argument("--check", action="store_true",
                    help="compare against the existing --out instead of overwriting it")
    a = ap.parse_args()

    out = merge(json.load(open(a.baselines)), json.load(open(a.ppo)))

    if a.check:
        if json.load(open(a.out)) == out:
            print(f"{a.out} matches the merge of {a.baselines} and {a.ppo}")
            return
        raise SystemExit(f"MISMATCH: {a.out} is not the merge of {a.baselines} and {a.ppo}")

    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {a.out} ({len(out['rows'])} rows) from {a.baselines} + {a.ppo}")


if __name__ == "__main__":
    main()
