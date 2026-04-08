#!/bin/bash
# Day 2 Experiments: Confirm the 1-GPU Dev Gap
# Run on 1×H100. Total: ~25 minutes. Cost: ~$1.
#
# Usage:
#   bash day2_experiments.sh setup   # Prepare variants
#   bash day2_experiments.sh all     # Run all 4 experiments

set -e
CMD=${1:?"Usage: bash day2_experiments.sh [setup|all]"}
cd /workspace/parameter-golf/parameter-golf/workspace

pkill -f train_gpt.py 2>/dev/null || true
sleep 2
mkdir -p day2_results logs

run() {
    local NAME="$1"
    local SCRIPT="$2"
    local EXTRA_ENV="$3"
    local TS=$(date +%Y%m%d_%H%M%S)
    local LOG="logs/day2_${NAME}_${TS}.log"

    echo ""
    echo "╔════════════════════════════════════════╗"
    echo "║  ${NAME}"
    echo "║  $(date)"
    echo "╚════════════════════════════════════════╝"

    cp "$SCRIPT" train_gpt.py
    local START=$(date +%s)

    eval $EXTRA_ENV \
    SEED=1337 \
    MAX_WALLCLOCK_SECONDS=300 \
    VAL_LOSS_EVERY=0 \
    TRAIN_LOG_EVERY=100 \
    EVAL_STRIDE=64 \
    torchrun --standalone --nproc_per_node=1 \
        train_gpt.py 2>&1 | tee "$LOG"

    local END=$(date +%s)
    local DUR=$((END - START))

    local MID_BPB=$(grep -oP 'val_bpb:\K[0-9.]+' "$LOG" | head -1 || echo "N/A")
    local POST_EMA=$(grep -oP 'DIAGNOSTIC.*val_bpb:\K[0-9.]+' "$LOG" | head -1 || echo "N/A")
    # If no DIAGNOSTIC line, post_ema = mid_train (EMA was disabled)
    [ "$POST_EMA" = "N/A" ] && POST_EMA="(no EMA)"
    local ROUNDTRIP=$(grep -oP 'final_int6_roundtrip_exact val_bpb:\K[0-9.]+' "$LOG" | tail -1 || echo "")
    [ -z "$ROUNDTRIP" ] && ROUNDTRIP=$(grep -oP 'final_int6_roundtrip.*val_bpb:\K[0-9.]+' "$LOG" | tail -1 || echo "")
    [ -z "$ROUNDTRIP" ] && ROUNDTRIP=$(grep -oP 'int6.*roundtrip.*val_bpb:\K[0-9.]+' "$LOG" | tail -1 || echo "N/A")
    local ART=$(grep -oP 'Total submission size.*?: \K[0-9]+' "$LOG" | tail -1 || echo "N/A")
    local STEPS=$(grep -oP 'step:\K[0-9]+' "$LOG" | tail -1 || echo "N/A")
    local STEP_AVG=$(grep -oP 'step_avg:\K[0-9.]+' "$LOG" | tail -1 || echo "N/A")
    local TLOSS=$(grep -oP 'train_loss:\K[0-9.]+' "$LOG" | tail -1 || echo "N/A")

    cat > "day2_results/${NAME}.json" << ENDJSON
{
    "experiment": "${NAME}",
    "timestamp": "$(date -Iseconds)",
    "duration_seconds": ${DUR},
    "mid_train_bpb": "${MID_BPB}",
    "post_ema_bpb": "${POST_EMA}",
    "roundtrip_bpb": "${ROUNDTRIP}",
    "artifact_bytes": "${ART}",
    "steps": "${STEPS}",
    "step_avg_ms": "${STEP_AVG}",
    "final_train_loss": "${TLOSS}",
    "log_file": "${LOG}"
}
ENDJSON

    echo ""
    echo "╔════════════════════════════════════════╗"
    echo "║  RESULT: ${NAME}"
    echo "║  Mid-train BPB: ${MID_BPB}"
    echo "║  Post-EMA BPB:  ${POST_EMA}"
    echo "║  Roundtrip BPB: ${ROUNDTRIP}"
    echo "║  Artifact:      ${ART} bytes"
    echo "║  Steps:         ${STEPS} | ${STEP_AVG}ms"
    echo "║  Final loss:    ${TLOSS}"
    echo "║  Duration:      ${DUR}s"
    echo "╚════════════════════════════════════════╝"

    pkill -f train_gpt.py 2>/dev/null || true
    sleep 3
}

case "$CMD" in
    setup)
        echo "=== Preparing Day 2 experiment variants ==="

        # Save baseline if not already saved
        if [ ! -f train_gpt_day1_baseline.py ]; then
            cp train_gpt.py train_gpt_day1_baseline.py
            echo "✓ Saved Day 1 baseline (LeakyReLU + INT8 MLP)"
        else
            echo "✓ Day 1 baseline already exists"
        fi

        # Variant: relu² (revert LeakyReLU back to relu)
        cp train_gpt_day1_baseline.py train_gpt_relu2.py
        sed -i 's/F.leaky_relu(self.fc(x), negative_slope=0.5)/torch.relu(self.fc(x))/' train_gpt_relu2.py
        if grep -q "torch.relu(self.fc(x))" train_gpt_relu2.py; then
            echo "✓ relu² variant created"
        else
            echo "✗ sed failed — manually revert LeakyReLU to torch.relu in train_gpt_relu2.py"
            echo "  Current activation line:"
            grep -n "leaky_relu\|torch.relu" train_gpt_relu2.py | head -3
        fi

        echo ""
        echo "=== Ready ==="
        echo "Variants:"
        echo "  train_gpt_day1_baseline.py  = LeakyReLU²(0.5) + INT8 MLP (Day 1 winner)"
        echo "  train_gpt_relu2.py          = relu² + INT8 MLP (activation reverted)"
        echo ""
        echo "Experiments:"
        echo "  1. LeakyReLU + EMA ON   → baseline reference"
        echo "  2. LeakyReLU + EMA OFF  → THE key experiment"
        echo "  3. relu² + EMA OFF      → clean activation comparison"
        echo "  4. LeakyReLU + EMA OFF  → (repeat of 2 for reproducibility check)"
        echo ""
        echo "Run: bash day2_experiments.sh all"
        ;;

    all)
        echo "=== Day 2: Confirming the 1-GPU Dev Gap ==="
        echo "4 experiments. ~28 minutes. ~$1.10."
        echo ""

        # Exp 1: LeakyReLU + EMA ON (reference)
        run "leaky_ema_on" "train_gpt_day1_baseline.py" ""

        # Exp 2: LeakyReLU + EMA OFF (the key experiment)
        run "leaky_ema_off" "train_gpt_day1_baseline.py" "EMA_ENABLED=0"

        # Exp 3: relu² + EMA OFF (clean activation comparison)
        run "relu2_ema_off" "train_gpt_relu2.py" "EMA_ENABLED=0"

        # Exp 4: LeakyReLU + EMA OFF again (reproducibility)
        run "leaky_ema_off_v2" "train_gpt_day1_baseline.py" "EMA_ENABLED=0"

        # ── Summary table ──
        echo ""
        echo "╔═══════════════════════════════════════════════════════════════════╗"
        echo "║                    DAY 2 RESULTS SUMMARY                         ║"
        echo "╠═══════════════════════════════════════════════════════════════════╣"
        printf "║ %-22s %-12s %-12s %-12s %-5s ║\n" "Experiment" "Mid-train" "Post-EMA" "Roundtrip" "Steps"
        echo "╠═══════════════════════════════════════════════════════════════════╣"
        for f in day2_results/*.json; do
            NAME=$(python3 -c "import json; print(json.load(open('$f'))['experiment'])")
            MID=$(python3 -c "import json; print(json.load(open('$f'))['mid_train_bpb'])")
            EMA=$(python3 -c "import json; print(json.load(open('$f'))['post_ema_bpb'])")
            RT=$(python3 -c "import json; print(json.load(open('$f'))['roundtrip_bpb'])")
            ST=$(python3 -c "import json; print(json.load(open('$f'))['steps'])")
            printf "║ %-22s %-12s %-12s %-12s %-5s ║\n" "$NAME" "$MID" "$EMA" "$RT" "$ST"
        done
        echo "╚═══════════════════════════════════════════════════════════════════╝"

        # ── Key comparisons ──
        echo ""
        echo "=== KEY COMPARISONS ==="
        echo ""

        # EMA impact
        EMA_ON_RT=$(python3 -c "import json; print(json.load(open('day2_results/leaky_ema_on.json'))['roundtrip_bpb'])" 2>/dev/null || echo "N/A")
        EMA_OFF_RT=$(python3 -c "import json; print(json.load(open('day2_results/leaky_ema_off.json'))['roundtrip_bpb'])" 2>/dev/null || echo "N/A")

        echo "1. EMA IMPACT ON ROUNDTRIP:"
        echo "   EMA ON:  roundtrip = $EMA_ON_RT"
        echo "   EMA OFF: roundtrip = $EMA_OFF_RT"
        if [ "$EMA_ON_RT" != "N/A" ] && [ "$EMA_OFF_RT" != "N/A" ]; then
            DELTA=$(python3 -c "print(f'{float(\"$EMA_ON_RT\") - float(\"$EMA_OFF_RT\"):.4f}')")
            echo "   Delta: $DELTA BPB (positive = EMA was hurting)"
            PCT=$(python3 -c "print(f'{(float(\"$EMA_ON_RT\") - float(\"$EMA_OFF_RT\")) / float(\"$EMA_ON_RT\") * 100:.1f}')")
            echo "   That's ${PCT}% of the roundtrip degradation caused by EMA"
        fi

        echo ""

        # Activation comparison (EMA off)
        LEAKY_RT=$(python3 -c "import json; print(json.load(open('day2_results/leaky_ema_off.json'))['roundtrip_bpb'])" 2>/dev/null || echo "N/A")
        RELU_RT=$(python3 -c "import json; print(json.load(open('day2_results/relu2_ema_off.json'))['roundtrip_bpb'])" 2>/dev/null || echo "N/A")
        LEAKY_MID=$(python3 -c "import json; print(json.load(open('day2_results/leaky_ema_off.json'))['mid_train_bpb'])" 2>/dev/null || echo "N/A")
        RELU_MID=$(python3 -c "import json; print(json.load(open('day2_results/relu2_ema_off.json'))['mid_train_bpb'])" 2>/dev/null || echo "N/A")

        echo "2. ACTIVATION COMPARISON (EMA OFF):"
        echo "   LeakyReLU²(0.5): mid=$LEAKY_MID, roundtrip=$LEAKY_RT"
        echo "   relu²:           mid=$RELU_MID, roundtrip=$RELU_RT"

        echo ""

        # Reproducibility
        V1=$(python3 -c "import json; print(json.load(open('day2_results/leaky_ema_off.json'))['roundtrip_bpb'])" 2>/dev/null || echo "N/A")
        V2=$(python3 -c "import json; print(json.load(open('day2_results/leaky_ema_off_v2.json'))['roundtrip_bpb'])" 2>/dev/null || echo "N/A")
        echo "3. REPRODUCIBILITY (same config, two runs):"
        echo "   Run 1: roundtrip = $V1"
        echo "   Run 2: roundtrip = $V2"

        echo ""
        echo "=== Save results: git add -A && git commit -m 'day2: EMA dev gap confirmed' ==="
        ;;

    *)
        echo "Usage: bash day2_experiments.sh [setup|all]"
        exit 1
        ;;
esac
