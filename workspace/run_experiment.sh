#!/bin/bash
# Usage: bash run_experiment.sh "description" [screen|full] [block]
# Example: bash run_experiment.sh "leaky_relu_0.5" full A
set -e

NAME=${1:-"unnamed"}
MODE=${2:-"full"}
BLOCK=${3:-"X"}
TS=$(date +%Y%m%d_%H%M%S)
EID="${TS}_${NAME// /_}"
LOG="logs/exp_${EID}.log"
JSON="experiments/exp_${EID}.json"
COST_PER_HOUR=2.39

mkdir -p logs experiments

if [ "$MODE" = "screen" ]; then
    ITERS=500; TLIMIT=0
    echo ">>> SCREEN [Block $BLOCK]: $NAME (500 steps)"
else
    ITERS=20000; TLIMIT=300
    echo ">>> FULL [Block $BLOCK]: $NAME (5 min)"
fi

# Capture current best before this experiment
PREV_BEST="null"
if [ -f experiments/results.csv ] && [ -s experiments/results.csv ]; then
    PREV_BEST=$(awk -F',' 'NR>1 && $4!="NA" {print $4}' experiments/results.csv | sort -n | head -1)
    [ -z "$PREV_BEST" ] && PREV_BEST="null"
fi

# Capture full diff (not truncated)
FULL_DIFF_FILE="experiments/diff_${EID}.patch"
git diff train_gpt.py > "$FULL_DIFF_FILE" 2>/dev/null || echo "" > "$FULL_DIFF_FILE"
DIFF_SUMMARY=$(head -50 "$FULL_DIFF_FILE")

# Capture code hash
CODE_HASH=$(md5sum train_gpt.py | cut -d' ' -f1)

# Count experiment number
EXP_NUM=$(( $(wc -l < experiments/results.csv 2>/dev/null || echo 1) ))

START=$(date +%s)

# Run training
SEED=${SEED:-1337} RUN_ID="exp_${EID}" \
ITERATIONS=$ITERS \
MAX_WALLCLOCK_SECONDS=$TLIMIT \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=50 \
torchrun --standalone --nproc_per_node=${NGPU:-1} \
    train_gpt.py 2>&1 | tee "$LOG"

EC=${PIPESTATUS[0]}
END=$(date +%s)
DUR=$((END - START))
COST=$(echo "scale=4; $DUR / 3600 * $COST_PER_HOUR" | bc 2>/dev/null || echo "0")

# ── Extract metrics ──

BPB=$(grep -oP 'val_bpb:\K[0-9.]+' "$LOG" | tail -1 || echo "")
VLOSS=$(grep -oP 'val_loss:\K[0-9.]+' "$LOG" | tail -1 || echo "")
TLOSS=$(grep -oP 'train_loss:\K[0-9.]+' "$LOG" | tail -1 || echo "")
PARAMS=$(grep -oP 'model_params:\K[0-9]+' "$LOG" | head -1 || echo "")
STEPS=$(grep -oP 'step:\K[0-9]+' "$LOG" | tail -1 || echo "")

# Artifact size — try multiple patterns
ART=$(grep -oP 'int6\+zstd: \K[0-9]+' "$LOG" | tail -1 || echo "")
[ -z "$ART" ] && ART=$(grep -oP 'int8\+zlib: \K[0-9]+' "$LOG" | tail -1 || echo "")
[ -z "$ART" ] && ART=$(grep -oP 'int6\+zlib: \K[0-9]+' "$LOG" | tail -1 || echo "")
[ -z "$ART" ] && ART=$(grep -oP 'Total submission size.*?: \K[0-9]+' "$LOG" | tail -1 || echo "")

# Extract loss curve (step:loss pairs at every logged step)
LOSS_CURVE=$(grep -oP 'step:\K[0-9]+ train_loss:[0-9.]+' "$LOG" | \
    sed 's/ train_loss/,/' | tr '\n' ';' | sed 's/;$//' || echo "")

# Extract hyperparameters from log header
MODEL_DIM=$(grep -oP 'model_dim:\K[0-9]+' "$LOG" | head -1 || echo "")
[ -z "$MODEL_DIM" ] && MODEL_DIM=$(grep -oP 'dim:\K[0-9]+' "$LOG" | head -1 || echo "")
NUM_LAYERS=$(grep -oP 'num_layers:\K[0-9]+' "$LOG" | head -1 || echo "")
SEQ_LEN=$(grep -oP 'train_seq_len:\K[0-9]+' "$LOG" | head -1 || echo "")
LR=$(grep -oP 'matrix_lr:\K[0-9.]+' "$LOG" | head -1 || echo "")
WD=$(grep -oP 'weight_decay:\K[0-9.]+' "$LOG" | head -1 || echo "")
BATCH=$(grep -oP 'train_batch_tokens:\K[0-9]+' "$LOG" | head -1 || echo "")
STEP_AVG=$(grep -oP 'step_avg:\K[0-9.]+' "$LOG" | tail -1 || echo "")

# ── Determine status ──

ST="success"
[ $EC -ne 0 ] && ST="crashed"
[ -z "$BPB" ] && [ "$ST" = "success" ] && ST="no_metrics"

if [ -n "$BPB" ]; then
    if (( $(echo "$BPB > 5.0" | bc -l 2>/dev/null || echo 0) )); then
        ST="diverged"
    fi
fi
if [ -n "$ART" ] && [ "$ART" -gt 16000000 ] 2>/dev/null; then
    ST="over_budget"
fi

# Compute delta from previous best
DELTA="null"
if [ -n "$BPB" ] && [ "$PREV_BEST" != "null" ]; then
    DELTA=$(echo "scale=6; $BPB - $PREV_BEST" | bc 2>/dev/null || echo "null")
fi

# ── Write JSON ──

cat > "$JSON" << ENDJSON
{
  "id": "${EID}",
  "experiment_number": ${EXP_NUM},
  "name": "${NAME}",
  "block": "${BLOCK}",
  "timestamp": "$(date -Iseconds)",
  "mode": "${MODE}",
  "status": "${ST}",
  "duration_seconds": ${DUR},
  "cost_usd": ${COST:-0},
  "seed": ${SEED:-1337},
  "code_hash": "${CODE_HASH}",
  "previous_best_bpb": ${PREV_BEST},
  "metrics": {
    "val_bpb": ${BPB:-null},
    "val_loss": ${VLOSS:-null},
    "final_train_loss": ${TLOSS:-null},
    "artifact_bytes": ${ART:-null},
    "total_params": ${PARAMS:-null},
    "total_steps": ${STEPS:-null},
    "step_avg_ms": ${STEP_AVG:-null},
    "delta_vs_best": ${DELTA}
  },
  "config": {
    "model_dim": ${MODEL_DIM:-null},
    "num_layers": ${NUM_LAYERS:-null},
    "seq_len": ${SEQ_LEN:-null},
    "matrix_lr": ${LR:-null},
    "weight_decay": ${WD:-null},
    "batch_tokens": ${BATCH:-null}
  },
  "loss_curve": "${LOSS_CURVE}",
  "diff_file": "${FULL_DIFF_FILE}",
  "diff_summary": $(echo "$DIFF_SUMMARY" | python3 -c "import sys,json; print(json.dumps(sys.stdin.read()))" 2>/dev/null || echo '""'),
  "log_file": "${LOG}"
}
ENDJSON

# ── Append to CSV ──

if [ ! -f experiments/results.csv ] || [ ! -s experiments/results.csv ]; then
    echo "exp_num,id,name,block,status,val_bpb,delta,artifact_bytes,duration,cost_usd,steps" > experiments/results.csv
fi
echo "${EXP_NUM},${EID},${NAME},${BLOCK},${ST},${BPB:-NA},${DELTA},${ART:-NA},${DUR},${COST},${STEPS:-NA}" >> experiments/results.csv

# ── Print summary ──

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  Experiment #${EXP_NUM} [Block ${BLOCK}]"
echo "║  Name:       ${NAME}"
echo "║  Status:     ${ST}"
echo "║  val_bpb:    ${BPB:-N/A}"
echo "║  Prev best:  ${PREV_BEST}"
echo "║  Delta:      ${DELTA}"
echo "║  Artifact:   ${ART:-N/A} bytes"
echo "║  Duration:   ${DUR}s | Cost: \$${COST}"
echo "║  Steps:      ${STEPS:-N/A} | Step avg: ${STEP_AVG:-N/A}ms"
echo "║  JSON:       ${JSON}"
echo "║  Diff:       ${FULL_DIFF_FILE}"
echo "╚══════════════════════════════════════════════╝"
exit $EC