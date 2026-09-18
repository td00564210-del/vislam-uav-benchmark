"""Evaluate one SLAM run against EuRoC ground truth and save the metrics as JSON.

Example:
    python evaluate_run.py \\
        --estimate results/orbslam3/MH_01_easy/run_0/trajectory.txt \\
        --groundtruth data/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv \\
        --runtime results/orbslam3/MH_01_easy/run_0/runtime.json \\
        --out results/orbslam3/MH_01_easy/run_0/metrics.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import trajectory_metrics as tm


def evaluate(estimate: str, groundtruth: str, with_scale: bool, max_diff: float, rpe_delta: float,
             min_coverage: float) -> dict:
    t_gt, p_gt, q_gt = (tm.load_euroc_groundtruth(groundtruth) if groundtruth.endswith(".csv")
                        else tm.load_tum(groundtruth))
    try:
        t_est, p_est, q_est = tm.load_tum(estimate)
    except (OSError, ValueError) as exc:
        return {"status": "failed", "reason": f"cannot read trajectory: {exc}"}

    i_est, i_gt = tm.associate(t_est, t_gt, max_diff)
    duration_gt = t_gt.max() - t_gt.min()
    coverage = (t_est[i_est].max() - t_est[i_est].min()) / duration_gt if len(i_est) > 1 else 0.0
    # Tracking loss is a benchmark outcome, not something to silently drop.
    if len(i_est) < 10 or coverage < min_coverage:
        return {"status": "failed", "reason": "tracking lost / insufficient coverage",
                "matched_poses": int(len(i_est)), "coverage": float(coverage)}

    ate_res = tm.ate(p_est[i_est], p_gt[i_gt], with_scale)
    ate_res.pop("aligned_positions")
    rpe_res = tm.rpe(t_est[i_est], p_est[i_est], q_est[i_est], p_gt[i_gt], q_gt[i_gt], rpe_delta)
    return {"status": "ok", "coverage": float(coverage),
            "alignment": "sim3" if with_scale else "se3", "ate": ate_res, "rpe": rpe_res}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--estimate", required=True, help="estimated trajectory (TUM format)")
    ap.add_argument("--groundtruth", required=True, help="EuRoC data.csv or TUM-format ground truth")
    ap.add_argument("--out", required=True, help="output metrics JSON")
    ap.add_argument("--runtime", help="optional JSON with wall_clock_s and n_frames from the run script")
    ap.add_argument("--sim3", action="store_true", help="use Sim(3) alignment (monocular only)")
    ap.add_argument("--max-time-diff", type=float, default=0.02)
    ap.add_argument("--rpe-delta", type=float, default=1.0)
    ap.add_argument("--min-coverage", type=float, default=0.9)
    args = ap.parse_args()

    result = evaluate(args.estimate, args.groundtruth, args.sim3, args.max_time_diff,
                      args.rpe_delta, args.min_coverage)
    if args.runtime and Path(args.runtime).exists():
        rt = json.loads(Path(args.runtime).read_text())
        result["runtime"] = rt
        if rt.get("wall_clock_s") and rt.get("n_frames"):
            result["runtime"]["fps"] = rt["n_frames"] / rt["wall_clock_s"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
