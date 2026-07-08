#!/usr/bin/env bash
# Build + run (or benchmark) a cuda-oxide kernel on a Modal GPU.
#
#   ./run.sh                 # vecadd correctness check
#   ./run.sh vecadd          # same
#   ./run.sh vecadd bench    # vecadd throughput benchmark
#   ./run.sh vecsum thrust   # Thrust/CUB reduce baseline (kernels/<k>/baselines)
#   GPU=A100 ./run.sh vecadd bench   # pick a GPU
#   SANITIZE=synccheck ./run.sh gemm # compute-sanitizer (memcheck/racecheck/synccheck/initcheck)
#   BASELINE=gemm_baseline ./run.sh gemm    # CUDA C++ baseline (kernels/<k>/baselines/<name>.cu)
set -euo pipefail
cd "$(dirname "$0")"

kernel="${1:-vecadd}"
bin="${2:-}"

args=(--kernel "$kernel")
if [[ "$bin" == "thrust" ]]; then
  args+=(--thrust)
elif [[ -n "$bin" ]]; then
  args+=(--bin "$bin")
fi
[[ -n "${GPU:-}" ]] && args+=(--gpu "$GPU")
[[ -n "${SANITIZE:-}" ]] && args+=(--sanitize "$SANITIZE")
[[ -n "${BASELINE:-}" ]] && args+=(--baseline "$BASELINE")

exec modal run modal_app.py "${args[@]}"
