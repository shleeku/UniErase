#!/usr/bin/env bash
set -euo pipefail

PY=python
SCRIPT1=train_UNL_github.py
SCRIPT2=run_edit_seq_github.py
SCRIPT3=eval_github.py

MODEL_SIZE=${1:-"1.7B"}
TASK=${2:-"RETURN"}
ALG_NAME=${3:-"AlphaEdit"}

START_STAGE=${4:-2}
END_STAGE=${5:-10}

BATCH_SIZE=${BATCH_SIZE:-4}
OUTDIR=${OUTDIR:-"eval_results"}

for stage in $(seq "$START_STAGE" "$END_STAGE"); do
  echo "=== stage=${stage} model_size=${MODEL_SIZE} task=${TASK} alg=${ALG_NAME} ==="
  $PY $SCRIPT1 \
  --model_size "$MODEL_SIZE" \
  --task "$TASK" \
  --stage "$stage"
  $PY $SCRIPT2 \
  --model_size "$MODEL_SIZE" \
  --task "$TASK" \
  --stage "$stage" \
  --alg_name "$ALG_NAME"
  $PY $SCRIPT3 \
    --model_size "$MODEL_SIZE" \
    --task "$TASK" \
    --alg_name "$ALG_NAME" \
    --stage "$stage" \
    --output_dir "$OUTDIR"
done