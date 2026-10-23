#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FEDERATED_PYTHON="${FEDERATED_PYTHON:-python}"
FEDSCGEN_PYTHON="${FEDSCGEN_PYTHON:-python}"
PIPELINE_LOG="$ROOT/results/final_figure3_canonical7/pipeline.log"

cd "$ROOT"
mkdir -p results/final_figure3_canonical7 results/final_figure4_canonical7

{
  echo "[$(date --iso-8601=seconds)] preflight"
  "$FEDERATED_PYTHON" experiments/run_final_figure3_4_canonical7.py --stage preflight

  echo "[$(date --iso-8601=seconds)] native Figure 3/4 training"
  "$FEDERATED_PYTHON" experiments/run_final_figure3_4_canonical7.py --stage native

  echo "[$(date --iso-8601=seconds)] official FedscGen Figure 3/4 training"
  if [[ -n "${FEDSCGEN_LIBRARY_PATH:-}" ]]; then
    export LD_LIBRARY_PATH="${FEDSCGEN_LIBRARY_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
  "$FEDSCGEN_PYTHON" experiments/run_final_figure3_4_canonical7.py --stage fedscgen

  echo "[$(date --iso-8601=seconds)] common evaluation, statistics and figures"
  "$FEDERATED_PYTHON" experiments/run_final_figure3_4_canonical7.py --stage finalize
  echo "[$(date --iso-8601=seconds)] COMPLETE"
} >>"$PIPELINE_LOG" 2>&1
