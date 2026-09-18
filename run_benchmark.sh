#!/usr/bin/env bash
# Run ORB-SLAM3 (stereo-inertial) on EuRoC sequences N times, record runtime and
# environment, then evaluate every run.
#
# Usage: ORB_SLAM3_DIR=~/ORB_SLAM3 EUROC_DIR=~/datasets/euroc N_RUNS=5 ./run_benchmark.sh
set -euo pipefail

ORB_SLAM3_DIR="${ORB_SLAM3_DIR:?set ORB_SLAM3_DIR to the ORB-SLAM3 checkout}"
EUROC_DIR="${EUROC_DIR:?set EUROC_DIR to the folder containing MH_01_easy, ...}"
N_RUNS="${N_RUNS:-5}"
RESULTS_DIR="${RESULTS_DIR:-results}"
ORB_SLAM3_DIR="$(realpath "$ORB_SLAM3_DIR")"
EUROC_DIR="$(realpath "$EUROC_DIR")"
METHOD="orbslam3"
SEQUENCES=(MH_01_easy MH_02_easy MH_03_medium MH_04_difficult MH_05_difficult
           V1_01_easy V1_02_medium V1_03_difficult V2_01_easy V2_02_medium V2_03_difficult)

# Record the environment once so results are reproducible and comparable.
mkdir -p "$RESULTS_DIR"
{
  echo "date: $(date -Iseconds)"
  echo "os: $(uname -srm)"
  echo "cpu: $(lscpu 2>/dev/null | grep 'Model name' | sed 's/Model name:\s*//' || true)"
  echo "threads: $(nproc)"
  echo "ram: $(free -h 2>/dev/null | awk '/Mem:/ {print $2}' || true)"
  echo "gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo none)"
  echo "compiler: $(g++ --version 2>/dev/null | head -n1 || true)"
  echo "orbslam3_commit: $(git -C "$ORB_SLAM3_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "n_runs: $N_RUNS"
} > "$RESULTS_DIR/environment.txt"

EXE="$ORB_SLAM3_DIR/Examples/Stereo-Inertial/stereo_inertial_euroc"
VOCAB="$ORB_SLAM3_DIR/Vocabulary/ORBvoc.txt"
SETTINGS="$ORB_SLAM3_DIR/Examples/Stereo-Inertial/EuRoC.yaml"
TIMES_DIR="$ORB_SLAM3_DIR/Examples/Stereo-Inertial/EuRoC_TimeStamps"

for seq in "${SEQUENCES[@]}"; do
  short="${seq%%_[a-z]*}"   # MH_01_easy -> MH_01
  ts_file="$TIMES_DIR/${short//_/}.txt"  # MH01.txt
  n_frames=$(wc -l < "$ts_file")
  for ((k = 0; k < N_RUNS; k++)); do
    run_dir="$RESULTS_DIR/$METHOD/$seq/run_$k"
    mkdir -p "$run_dir"
    echo "== $seq run $k =="
    start=$(date +%s.%N)
    ( cd "$run_dir" && "$EXE" "$VOCAB" "$SETTINGS" "$EUROC_DIR/$seq" "$ts_file" traj ) \\
      > "$run_dir/log.txt" 2>&1 || echo "run exited with error (kept as failure)"
    end=$(date +%s.%N)
    printf '{"wall_clock_s": %s, "n_frames": %s}\\n' "$(echo "$end - $start" | bc)" "$n_frames" \\
      > "$run_dir/runtime.json"
    # ORB-SLAM3 writes f_traj.txt (all frames) in TUM format with ns timestamps.
    traj="$run_dir/f_traj.txt"
    python3 evaluate_run.py --estimate "$traj" \\
      --groundtruth "$EUROC_DIR/$seq/mav0/state_groundtruth_estimate0/data.csv" \\
      --runtime "$run_dir/runtime.json" --out "$run_dir/metrics.json" > /dev/null
  done
done

python3 aggregate_results.py --results "$RESULTS_DIR" --baseline "$METHOD" --out summary
