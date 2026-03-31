#!/bin/bash
# Usage: bash run_experiment.sh "description" [screen|full]
set -e
NAME=${1:-"unnamed"}
MODE=${2:-"full"}
TS=$(date +%Y%m%d_%H%M%S)
EID="${TS}_${NAME// /_}"
LOG="logs/exp_${EID}.log"
JSON="experiments/exp_${EID}.json"
mkdir -p logs experiments

if [ "$MODE" = "screen" ]; then
    ITERS=500; TLIMIT=0
    echo ">>> SCREEN: $NAME (500 steps)"
else
    ITERS=20000; TLIMIT=300
    echo ">>> FULL: $NAME (5 min)"
fi

CODE_HASH=$(md5sum train_gpt.py | cut -d' ' -f1)
DIFF=$(git diff train_gpt.py 2>/dev/null | head -200 || echo "")
START=$(date +%s)

SEED=${SEED:-1337} RUN_ID="exp_${EID}" \
ITERATIONS=$ITERS \
MAX_WALLCLOCK_SECONDS=$TLIMIT \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=100 \
torchrun --standalone --nproc_per_node=${NGPU:-1} \
    train_gpt.py 2>&1 | tee "$LOG"

EC=${PIPESTATUS[0]}
END=$(date +%s)
DUR=$((END - START))

# Extract metrics
BPB=$(grep -oP 'val_bpb:\K[0-9.]+' "$LOG" | tail -1 || echo "")
VLOSS=$(grep -oP 'val_loss:\K[0-9.]+' "$LOG" | tail -1 || echo "")
ART=$(grep -oP 'int8\+zlib: \K[0-9]+|int6\+zstd: \K[0-9]+|int6\+zlib: \K[0-9]+' "$LOG" | tail -1 || echo "")
if [ -z "$ART" ]; then
    ART=$(grep -oP 'Total submission size.*?: \K[0-9]+' "$LOG" | tail -1 || echo "")
fi
TLOSS=$(grep -oP 'train_loss:\K[0-9.]+' "$LOG" | tail -1 || echo "")
PARAMS=$(grep -oP 'model_params:\K[0-9]+' "$LOG" | head -1 || echo "")
STEPS=$(grep -oP 'step:\K[0-9]+' "$LOG" | tail -1 || echo "")

# Determine status
ST="success"
[ $EC -ne 0 ] && ST="crashed"
[ -z "$BPB" ] && [ "$ST" = "success" ] && ST="no_metrics"

# Sanity checks
if [ -n "$BPB" ]; then
    if (( $(echo "$BPB > 5.0" | bc -l 2>/dev/null || echo 0) )); then
        ST="diverged"
    fi
fi
if [ -n "$ART" ] && [ "$ART" -gt 16000000 ] 2>/dev/null; then
    ST="over_budget"
fi

# Write JSON
cat > "$JSON" << ENDJSON
{
  "id": "${EID}",
  "name": "${NAME}",
  "timestamp": "$(date -Iseconds)",
  "mode": "${MODE}",
  "status": "${ST}",
  "duration_seconds": ${DUR},
  "seed": ${SEED:-1337},
  "code_hash": "${CODE_HASH}",
  "metrics": {
    "val_bpb": ${BPB:-null},
    "val_loss": ${VLOSS:-null},
    "final_train_loss": ${TLOSS:-null},
    "artifact_bytes": ${ART:-null},
    "total_params": ${PARAMS:-null},
    "total_steps": ${STEPS:-null}
  },
  "diff": $(echo "$DIFF" | python3 -c "import sys,json; print(json.dumps(sys.stdin.read()[:3000]))" 2>/dev/null || echo '""')
}
ENDJSON

# Append to CSV
if [ ! -f experiments/results.csv ] || [ ! -s experiments/results.csv ]; then
    echo "id,name,status,val_bpb,artifact_bytes,duration,steps" > experiments/results.csv
fi
echo "${EID},${NAME},${ST},${BPB:-NA},${ART:-NA},${DUR},${STEPS:-NA}" >> experiments/results.csv

# Print summary
echo ""
echo "============================================"
echo "  Experiment: $NAME"
echo "  Status:     $ST"
echo "  val_bpb:    ${BPB:-N/A}"
echo "  Artifact:   ${ART:-N/A} bytes"
echo "  Duration:   ${DUR}s | Steps: ${STEPS:-N/A}"
echo "  JSON:       $JSON"
echo "============================================"
exit $EC