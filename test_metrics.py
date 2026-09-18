"""Sanity tests on synthetic trajectories (no dataset needed): python -m pytest -q"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

import trajectory_metrics as tm


def _random_rotation(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def _synthetic(n=500, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 50, n)
    pos = np.column_stack([np.sin(t / 5) * 3, np.cos(t / 7) * 2, 0.1 * t])
    quat = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))
    return rng, t, pos, quat


def test_umeyama_recovers_se3():
    rng, _, pos, _ = _synthetic()
    rot, trans = _random_rotation(rng), rng.normal(size=3)
    moved = (rot @ pos.T).T + trans
    r_hat, t_hat, s_hat = tm.umeyama_alignment(pos, moved)
    assert np.allclose(r_hat, rot, atol=1e-8) and np.allclose(t_hat, trans, atol=1e-8) and s_hat == 1.0


def test_umeyama_recovers_sim3_scale():
    rng, _, pos, _ = _synthetic()
    moved = 2.5 * (_random_rotation(rng) @ pos.T).T + 1.0
    assert abs(tm.umeyama_alignment(pos, moved, with_scale=True)[2] - 2.5) < 1e-8


def test_ate_zero_for_rigidly_moved_trajectory():
    rng, _, pos, _ = _synthetic()
    est = (_random_rotation(rng) @ pos.T).T + rng.normal(size=3)
    assert tm.ate(est, pos)["rmse"] < 1e-9


def test_ate_matches_injected_noise():
    rng, _, pos, _ = _synthetic(n=20000)
    sigma = 0.05
    est = pos + rng.normal(scale=sigma, size=pos.shape)
    # RMSE of isotropic 3D Gaussian noise is sigma * sqrt(3).
    assert abs(tm.ate(est, pos)["rmse"] - sigma * np.sqrt(3)) < 0.005


def test_rpe_zero_for_identical_and_positive_for_drift():
    _, t, pos, quat = _synthetic()
    assert tm.rpe(t, pos, quat, pos, quat)["trans_rmse"] < 1e-12
    drift = pos + np.column_stack([0.01 * t, np.zeros_like(t), np.zeros_like(t)])
    assert abs(tm.rpe(t, drift, quat, pos, quat, delta=1.0)["trans_rmse"] - 0.01) < 2e-3


def test_associate_handles_offsets_and_gaps():
    t_gt = np.arange(0, 10, 0.005)
    t_est = np.arange(0, 10, 0.05) + 0.001
    i_est, i_gt = tm.associate(t_est, t_gt, 0.02)
    assert len(i_est) == len(t_est) and np.all(np.abs(t_est[i_est] - t_gt[i_gt]) <= 0.02)


def test_cli_reports_tracking_failure(tmp_path: Path):
    _, t, pos, quat = _synthetic()
    gt = tmp_path / "gt.txt"
    np.savetxt(gt, np.column_stack([t, pos, quat]))
    est = tmp_path / "est.txt"
    np.savetxt(est, np.column_stack([t, pos, quat])[:100])  # tracking lost after 20% of the sequence
    out = tmp_path / "metrics.json"
    subprocess.run([sys.executable, "evaluate_run.py", "--estimate", str(est), "--groundtruth", str(gt),
                    "--out", str(out)], check=True, capture_output=True, cwd=Path(__file__).parent)
    assert json.loads(out.read_text())["status"] == "failed"
