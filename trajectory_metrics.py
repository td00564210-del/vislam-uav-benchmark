"""Trajectory accuracy metrics for visual-inertial SLAM benchmarking.

Implements timestamp association, rigid-body (SE(3)) or similarity (Sim(3))
alignment via the Umeyama method, Absolute Trajectory Error (ATE) and
Relative Pose Error (RPE).

Trajectory files use the TUM format, one pose per line:
    timestamp tx ty tz qx qy qz qw
EuRoC ground truth (state_groundtruth_estimate0/data.csv) can be converted with
`load_euroc_groundtruth`.
"""
from __future__ import annotations

import numpy as np


def load_tum(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a TUM-format trajectory. Returns (timestamps [s], positions Nx3, quats Nx4 xyzw)."""
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < 8:
        raise ValueError(f"{path}: expected 8 columns (t tx ty tz qx qy qz qw), got {data.shape[1]}")
    t = data[:, 0].copy()
    # ORB-SLAM3 EuRoC outputs use nanosecond timestamps; normalise to seconds.
    if np.median(t) > 1e14:  # e.g. 1403636579763555584 ns vs 1403636579.76 s
        t = t / 1e9
    return t, data[:, 1:4], data[:, 4:8]


def load_euroc_groundtruth(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load EuRoC state_groundtruth_estimate0/data.csv (t[ns], p xyz, q wxyz, ...)."""
    data = np.loadtxt(path, delimiter=",", comments="#")
    t = data[:, 0] / 1e9
    pos = data[:, 1:4]
    q_wxyz = data[:, 4:8]
    quat_xyzw = np.column_stack([q_wxyz[:, 1:4], q_wxyz[:, 0]])
    return t, pos, quat_xyzw


def associate(t_est: np.ndarray, t_gt: np.ndarray, max_diff: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    """Match each estimated timestamp to the nearest ground-truth timestamp within max_diff seconds.

    Returns index arrays (i_est, i_gt). Each ground-truth sample is used at most once.
    """
    order = np.argsort(t_gt)
    t_sorted = t_gt[order]
    pos = np.clip(np.searchsorted(t_sorted, t_est), 1, len(t_sorted) - 1)
    left, right = t_sorted[pos - 1], t_sorted[pos]
    nearest = np.where(np.abs(t_est - left) <= np.abs(t_est - right), pos - 1, pos)
    diff = np.abs(t_sorted[nearest] - t_est)
    i_est = np.nonzero(diff <= max_diff)[0]
    i_gt = order[nearest[i_est]]
    _, unique_idx = np.unique(i_gt, return_index=True)
    keep = np.sort(unique_idx)
    return i_est[keep], i_gt[keep]


def umeyama_alignment(src: np.ndarray, dst: np.ndarray, with_scale: bool = False) -> tuple[np.ndarray, np.ndarray, float]:
    """Least-squares transform (R, t, s) minimising ||dst - (s R src + t)||.

    Use with_scale=False (SE(3)) for stereo / visual-inertial systems whose scale is
    observable, and with_scale=True (Sim(3)) for monocular-only systems.
    """
    if src.shape != dst.shape or src.shape[0] < 3:
        raise ValueError("need matching point sets with at least 3 points")
    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - mu_src, dst - mu_dst
    cov = dst_c.T @ src_c / src.shape[0]
    u, d, vt = np.linalg.svd(cov)
    s_mat = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s_mat[2, 2] = -1
    rot = u @ s_mat @ vt
    scale = 1.0
    if with_scale:
        var_src = (src_c ** 2).sum() / src.shape[0]
        scale = float(np.trace(np.diag(d) @ s_mat) / var_src)
    trans = mu_dst - scale * rot @ mu_src
    return rot, trans, scale


def quat_to_rot(q_xyzw: np.ndarray) -> np.ndarray:
    """Convert quaternions (N x 4, xyzw) to rotation matrices (N x 3 x 3)."""
    q = q_xyzw / np.linalg.norm(q_xyzw, axis=1, keepdims=True)
    x, y, z, w = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=-1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=-1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=-1),
    ], axis=1)


def ate(pos_est: np.ndarray, pos_gt: np.ndarray, with_scale: bool = False) -> dict:
    """Absolute Trajectory Error after alignment.

    RMSE = sqrt((1/N) * sum ||e_i||^2), where e_i is the translation error of pose i.
    """
    rot, trans, scale = umeyama_alignment(pos_est, pos_gt, with_scale)
    aligned = (scale * (rot @ pos_est.T)).T + trans
    err = np.linalg.norm(aligned - pos_gt, axis=1)
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mean": float(err.mean()),
        "median": float(np.median(err)),
        "max": float(err.max()),
        "n_poses": int(len(err)),
        "scale": float(scale),
        "aligned_positions": aligned,
    }


def rpe(t: np.ndarray, pos_est: np.ndarray, q_est: np.ndarray, pos_gt: np.ndarray, q_gt: np.ndarray,
        delta: float = 1.0) -> dict:
    """Translational Relative Pose Error over a fixed time interval `delta` (seconds).

    For each pair (i, j) with t_j - t_i ~= delta, compares the relative motion of the
    estimate with that of the ground truth, expressed in the frame of pose i.
    Independent of global alignment, so it measures local drift.
    """
    r_est, r_gt = quat_to_rot(q_est), quat_to_rot(q_gt)
    j_idx = np.searchsorted(t, t + delta)
    valid = np.nonzero(j_idx < len(t))[0]
    if len(valid) == 0:
        raise ValueError("trajectory shorter than RPE interval")
    errors = []
    for i in valid:
        j = j_idx[i]
        d_est = r_est[i].T @ (pos_est[j] - pos_est[i])
        d_gt = r_gt[i].T @ (pos_gt[j] - pos_gt[i])
        rel_rot = r_gt[i].T @ r_gt[j]
        rel_rot_est = r_est[i].T @ r_est[j]
        err_rot = rel_rot.T @ rel_rot_est
        errors.append((np.linalg.norm(d_est - d_gt),
                       np.degrees(np.arccos(np.clip((np.trace(err_rot) - 1) / 2, -1.0, 1.0)))))
    errors = np.asarray(errors)
    return {
        "trans_rmse": float(np.sqrt(np.mean(errors[:, 0] ** 2))),
        "rot_rmse_deg": float(np.sqrt(np.mean(errors[:, 1] ** 2))),
        "n_pairs": int(len(errors)),
        "delta_s": delta,
    }
