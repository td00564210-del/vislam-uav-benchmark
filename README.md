# test# Visual-Inertial SLAM Benchmark for GNSS-Denied UAV Navigation

Code accompanying the HW#1 research proposal *"Benchmarking Visual-Inertial SLAM for UAV Navigation in GNSS-Denied Environments"* (Student ID 11576607, YEN-TING CHEN).

This repository implements the reproducible, apples-to-apples evaluation protocol described in Sections 3-5 of the proposal: every method is run on the same EuRoC MAV sequences, with the same sensor mode, alignment rule and error implementation, repeated several times, and summarised with robust statistics.

> **Status:** the evaluation pipeline is implemented and unit-tested on synthetic trajectories. The EuRoC experiments themselves have **not been run yet**, so this repository contains no measured results. The "To measure" cells in the proposal will be filled from `summary/` once the runs are executed.

## Files

| File | Purpose |
|---|---|
| `trajectory_metrics.py` | Timestamp association, Umeyama SE(3)/Sim(3) alignment, ATE and RPE |
| `evaluate_run.py` | Evaluates one run against EuRoC ground truth to `metrics.json` (tracking loss is recorded as a failure, never dropped) |
| `aggregate_results.py` | Median / mean / std ATE, failure counts, FPS; Wilcoxon signed-rank test with effect size and bootstrap CI; plot |
| `run_benchmark.sh` | Runs ORB-SLAM3 stereo-inertial on all 11 EuRoC sequences `N_RUNS` times and logs the environment |
| `test_metrics.py` | Sanity tests on synthetic data |

## Setup

```bash
pip install -r requirements.txt
python -m pytest -q          # runs the synthetic tests, no dataset needed
```

1. Build [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) following its README and note the commit hash.
2. Download the [EuRoC MAV dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) (ASL format) so that each sequence sits at `$EUROC_DIR/MH_01_easy/mav0/...`.

## Running the benchmark

```bash
ORB_SLAM3_DIR=~/ORB_SLAM3 EUROC_DIR=~/datasets/euroc N_RUNS=5 ./run_benchmark.sh
```

Outputs:

```
results/environment.txt                       # OS, CPU, threads, RAM, GPU, compiler, commit
results/orbslam3/<sequence>/run_<k>/          # trajectory, log, runtime.json, metrics.json
summary/summary.csv                           # per method x sequence statistics
summary/paired_tests.json                     # Wilcoxon tests vs. baseline
summary/ate_per_sequence.png
```

To add a comparator (e.g. VINS-Fusion, BASALT), write its TUM-format trajectories to `results/<method>/<sequence>/run_<k>/` and run `evaluate_run.py` on each, then `aggregate_results.py`.

## Protocol choices

- **Alignment:** SE(3) (no scale) for stereo-inertial runs; Sim(3) only via `--sim3` for monocular-only systems.
- **Association:** nearest ground-truth timestamp within 20 ms.
- **Failure rule:** a run with fewer than 10 matched poses or less than 90 % time coverage is a failure and counts in the failure rate.
- **Statistics:** median ATE is the headline number; mean +/- std shows dispersion; paired Wilcoxon signed-rank test across sequences (>= 5) with rank-biserial effect size and bootstrap 95 % CI.

## References

- C. Campos et al., "ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial, and Multimap SLAM," *IEEE T-RO*, 2021.
- M. Burri et al., "The EuRoC micro aerial vehicle datasets," *IJRR*, 2016.
- S. Umeyama, "Least-squares estimation of transformation parameters between two point patterns," *IEEE TPAMI*, 1991.
