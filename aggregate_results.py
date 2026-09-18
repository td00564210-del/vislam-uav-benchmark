"""Aggregate repeated runs into summary tables, paired statistics and plots.

Expects metrics files at results/<method>/<sequence>/run_<k>/metrics.json.

    python aggregate_results.py --results results --baseline orbslam3 --out summary
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


def collect(results_dir: Path) -> dict:
    runs = defaultdict(list)  # (method, sequence) -> list of metrics dicts
    for f in sorted(results_dir.glob("*/*/run_*/metrics.json")):
        method, sequence = f.parts[-4], f.parts[-3]
        runs[(method, sequence)].append(json.loads(f.read_text()))
    return runs


def summarise(runs: dict) -> list[dict]:
    rows = []
    for (method, sequence), items in sorted(runs.items()):
        ok = [r for r in items if r.get("status") == "ok"]
        ate = np.array([r["ate"]["rmse"] for r in ok])
        rpe = np.array([r["rpe"]["trans_rmse"] for r in ok])
        fps = np.array([r["runtime"]["fps"] for r in ok if r.get("runtime", {}).get("fps")])
        rows.append({
            "method": method, "sequence": sequence, "runs": len(items), "failures": len(items) - len(ok),
            "ate_median_m": float(np.median(ate)) if len(ate) else np.nan,
            "ate_mean_m": float(ate.mean()) if len(ate) else np.nan,
            "ate_std_m": float(ate.std(ddof=1)) if len(ate) > 1 else np.nan,
            "rpe_median_m": float(np.median(rpe)) if len(rpe) else np.nan,
            "fps_median": float(np.median(fps)) if len(fps) else np.nan,
        })
    return rows


def paired_tests(rows: list[dict], baseline: str) -> list[dict]:
    """Wilcoxon signed-rank test on per-sequence median ATE, each method vs. baseline.

    Also reports the matched-pairs rank-biserial correlation as an effect size and a
    bootstrap 95% CI of the median ATE difference.
    """
    table = defaultdict(dict)
    for r in rows:
        table[r["method"]][r["sequence"]] = r["ate_median_m"]
    out = []
    rng = np.random.default_rng(0)
    for method in sorted(table):
        if method == baseline or baseline not in table:
            continue
        seqs = [s for s in table[baseline] if s in table[method]
                and np.isfinite(table[baseline][s]) and np.isfinite(table[method][s])]
        if len(seqs) < 5:
            out.append({"method": method, "baseline": baseline, "n_sequences": len(seqs),
                        "note": "fewer than 5 paired sequences; test not meaningful"})
            continue
        diff = np.array([table[method][s] - table[baseline][s] for s in seqs])
        res = stats.wilcoxon(diff)
        ranks = stats.rankdata(np.abs(diff))
        effect = (ranks[diff > 0].sum() - ranks[diff < 0].sum()) / ranks.sum()
        boot = [np.median(rng.choice(diff, len(diff))) for _ in range(5000)]
        out.append({"method": method, "baseline": baseline, "n_sequences": len(seqs),
                    "median_diff_m": float(np.median(diff)), "ci95_low": float(np.percentile(boot, 2.5)),
                    "ci95_high": float(np.percentile(boot, 97.5)), "wilcoxon_p": float(res.pvalue),
                    "rank_biserial": float(effect)})
    return out


def plot(rows: list[dict], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = sorted({r["method"] for r in rows})
    sequences = sorted({r["sequence"] for r in rows})
    width = 0.8 / max(len(methods), 1)
    fig, ax = plt.subplots(figsize=(max(6, len(sequences) * 1.2), 4))
    for k, m in enumerate(methods):
        vals = [next((r["ate_median_m"] for r in rows if r["method"] == m and r["sequence"] == s), np.nan)
                for s in sequences]
        ax.bar(np.arange(len(sequences)) + k * width, vals, width, label=m)
    ax.set_xticks(np.arange(len(sequences)) + width * (len(methods) - 1) / 2, sequences, rotation=30)
    ax.set_ylabel("Median ATE RMSE (m)")
    ax.set_title("Measured ATE per sequence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "ate_per_sequence.png", dpi=200)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--baseline", default="orbslam3")
    ap.add_argument("--out", default="summary")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = summarise(collect(Path(args.results)))
    if not rows:
        raise SystemExit(f"no metrics.json files found under {args.results}")
    with open(out_dir / "summary.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tests = paired_tests(rows, args.baseline)
    (out_dir / "paired_tests.json").write_text(json.dumps(tests, indent=2))
    plot(rows, out_dir)
    print(f"wrote {out_dir/'summary.csv'}, {out_dir/'paired_tests.json'}, {out_dir/'ate_per_sequence.png'}")


if __name__ == "__main__":
    main()
