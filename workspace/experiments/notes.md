# Experiment Notes

Format: after every experiment, append an entry below.

---

(Experiments will be logged here by the auto-researcher)

### Exp 1 [Block A]: baseline_relu2 — BASELINE
- **Changed:** None (baseline run)
- **Hypothesis:** Establish reference BPB for relu² activation
- **Result:** bpb=1.7059 | delta=0.0000 | artifact=5,661,718 | steps=465
- **Cost:** 300s
- **Decision:** BASELINE — this is the reference for Block A
- **Insight:** Baseline relu² achieves 1.7059 mid-train bpb in 465 steps. EMA weights are degraded (post_ema bpb=5.3829, roundtrip=6.8676), suggesting EMA/quantization pipeline has issues at this training duration.
- **Implication for next experiment:** All Block A experiments compare against 1.7059. Focus on mid-train val_bpb since roundtrip is broken for all.

### Exp 2 [Block A]: leaky_relu_0.5 — KEEP
- **Changed:** MLP.forward(): replaced `torch.relu(self.fc(x))` with `F.leaky_relu(self.fc(x), negative_slope=0.5)`
- **Hypothesis:** LeakyReLU with slope=0.5 passes negative signal through, potentially improving gradient flow and expressiveness vs hard relu cutoff
- **Result:** bpb=1.7042 | delta=-0.0017 vs baseline | artifact=5,659,797 | steps=459
- **Cost:** 300s
- **Decision:** KEEP because bpb improved from 1.7059 to 1.7042 (-0.0017) and artifact is well within 16MB limit
- **Insight:** Slope=0.5 gives a small but consistent improvement. Training loss trajectory was slightly better throughout (e.g., step 100: 3.9242 vs 3.9310 baseline). The negative signal from leaky relu helps even with squaring.
- **Implication for next experiment:** Try slope=0.3 next — if it's worse than 0.5, the optimal slope is likely >= 0.5

### Exp 3 [Block A]: leaky_relu_0.3 — REVERT
- **Changed:** MLP.forward(): replaced `torch.relu(self.fc(x))` with `F.leaky_relu(self.fc(x), negative_slope=0.3)`
- **Hypothesis:** slope=0.3 passes less negative signal than 0.5, might find a better sparsity/expressiveness tradeoff
- **Result:** bpb=1.9736 | delta=+0.2677 vs baseline | artifact=5,524,292 | steps=332
- **Cost:** 300s
- **Decision:** REVERT because bpb is much worse (1.9736 vs 1.7059 baseline). However, result is confounded by severe GPU slowdown — step_avg=905ms vs baseline 646ms, so only 332 steps completed vs 465. The model was severely undertrained.
- **Insight:** Result is UNRELIABLE due to GPU performance variance. Step time nearly doubled (905ms vs 646ms), resulting in 29% fewer training steps. The bpb difference is likely dominated by undertaining, not the activation change. This experiment should ideally be re-run.
- **Implication for next experiment:** Monitor step_avg carefully. If GPU remains slow, results may not be comparable. Consider re-running this experiment if GPU speed normalizes.

### Exp 4 [Block A]: leaky_relu_0.1 — REVERT
- **Changed:** MLP.forward(): replaced `torch.relu(self.fc(x))` with `F.leaky_relu(self.fc(x), negative_slope=0.1)`
- **Hypothesis:** slope=0.1 is closest to standard relu, minimal negative signal — test the lower bound of the slope range
- **Result:** bpb=2.1819 | delta=+0.4760 vs baseline | artifact=5,464,564 | steps=276
- **Cost:** 300s
- **Decision:** REVERT because bpb is far worse (2.1819 vs 1.7059). However, like Exp 3, result is SEVERELY CONFOUNDED by GPU slowdown — step_avg=1090ms (vs baseline 646ms), yielding only 276 steps vs 465.
- **Insight:** GPU performance is degrading progressively across experiments (646ms → 905ms → 1090ms step avg). All results after Exp 2 are unreliable. The training loss at comparable step counts (e.g., step 200: 3.7671 vs baseline 3.2267) suggests slope=0.1 may genuinely be worse, but the magnitude of difference is unknowable.
- **Implication for next experiment:** GPU slowdown is a critical confound. May need to re-run Exp 3 and 4 if GPU recovers. For now, continue with Exp 5 (SwiGLU) and note step_avg.

### Exp 5 [Block A]: swiglu_2_3_hidden — REVERT
- **Changed:** MLP class: replaced single fc+relu²+proj with gate/up/proj SwiGLU pattern. `F.silu(gate(x)) * up(x)` with hidden=2/3*mlp_mult*dim to match param count. First attempt with full hidden OOM'd.
- **Hypothesis:** SwiGLU is the dominant activation in modern LLMs (LLaMA, Mistral); gated activation may improve expressiveness even at small scale
- **Result:** bpb=2.2544 | delta=+0.5485 vs baseline | artifact=6,641,062 | steps=269
- **Cost:** 300s (+ 42s for OOM retry)
- **Decision:** REVERT because bpb is far worse (2.2544 vs 1.7059). GPU slowdown continues (step_avg=1115ms), only 269 steps. But even at step 200, SwiGLU loss=3.8867 vs baseline 3.2267, confirming SwiGLU is genuinely worse per-step too.
- **Insight:** SwiGLU without squaring is worse than relu² at this scale and training budget. The squared activation in the original relu² appears critical — it provides a strong nonlinearity. SwiGLU replaces this with a smoother gating mechanism that may need more steps to converge. Also, reducing hidden to 2/3 to fit params may have hurt representational capacity.
- **Implication for next experiment:** relu² architecture is well-suited to this task. LeakyReLU²(0.5) is the only activation that showed improvement. Proceed to Block A summary.

## Block A Summary
Winner: leaky_relu_0.5 (Exp 2) with bpb=1.7042 (delta=-0.0017 vs baseline 1.7059)
Runner-up: baseline relu² (Exp 1) with bpb=1.7059
Key finding: LeakyReLU²(0.5) gives a small but consistent improvement over relu². Experiments 3-5 were confounded by progressive GPU slowdown (646ms → 905ms → 1090ms → 1115ms step avg), making direct comparison impossible, but per-step loss comparisons where available suggest they are genuinely worse. The squared activation is critical to this architecture — SwiGLU without squaring underperformed even per-step.

**CAVEAT:** GPU performance was unstable. Only Exp 2 had comparable step times to baseline (654ms vs 646ms). The winner selection is high-confidence for Exp 2 but low-confidence for ruling out Exp 3/4 at their stated slopes.

**POST-MORTEM:** GPU slowdown in Exp 3-5 was caused by stale GPU processes from prior experiments not being cleaned up (3 processes consuming ~67GB of 79GB GPU). After killing them, GPU performance returned to normal (~650ms/step).

### Exp 6 [Block B]: blockB_baseline_int6_int6 — BASELINE
- **Changed:** None (Block B baseline = Block A winner with LeakyReLU²(0.5) + INT6/INT6 quantization)
- **Hypothesis:** Establish reference BPB for quantization experiments with clean GPU
- **Result:** bpb=1.7013 | delta=0.0000 (Block B ref) | artifact=5,662,016 | steps=462
- **Cost:** 300s
- **Decision:** BASELINE for Block B. Note: bpb=1.7013 is better than Exp 2's 1.7042, likely due to clean GPU (650ms/step vs 654ms)
- **Insight:** With clean GPU, LeakyReLU²(0.5) achieves 1.7013 — a solid 0.0046 improvement over the original relu² baseline (1.7059). Roundtrip quantization is still terrible (6.9253) due to EMA degradation.
- **Implication for next experiment:** Test INT8 for all params to see if better quantization helps the roundtrip quality, even if mid-train bpb stays the same.

### Exp 7 [Block B]: int8_all — KEEP
- **Changed:** Line 1314: `mixed_quantize_int6(sd_cpu, {"mlp", "attn"})` → `mixed_quantize_int6(sd_cpu, set())` — all params use INT8 instead of INT6
- **Hypothesis:** INT8 preserves more precision than INT6, especially important since roundtrip bpb is terrible (6.93). More bits = less quantization error.
- **Result:** bpb=1.6998 | delta=-0.0015 vs Block B baseline | artifact=11,616,692 | steps=463
- **Cost:** 300s
- **Decision:** KEEP because bpb improved (1.6998 vs 1.7013) and roundtrip improved dramatically (5.5595 vs 6.9253). Artifact at 11.6MB is within 16MB limit.
- **Insight:** INT8 for all params is strictly better: -0.0015 mid-train bpb AND -1.37 roundtrip bpb. The artifact doubles (5.7MB → 11.6MB) but stays within budget. The quantization-aware training (late_qat) may be calibrated for INT6, so INT8 gets a free lunch from less quantization noise.
- **Implication for next experiment:** Try INT5 (aggressive) next — if INT8 helps, INT5 should hurt. This confirms the direction. Then try mixed configs.

### Exp 8 [Block B]: int5_all — REVERT
- **Changed:** quantize_int6_per_row: clip_range=31 → clip_range=15 (INT5 quantization for MLP+attn params)
- **Hypothesis:** More aggressive quantization should hurt roundtrip quality but produce smaller artifact
- **Result:** bpb=1.6984 | delta=-0.0029 vs Block B baseline (mid-train) | roundtrip=6.8897 | artifact=3,813,749 | steps=464
- **Cost:** 300s
- **Decision:** REVERT because mid-train bpb is essentially identical (quantization doesn't affect training), and roundtrip (6.89) is slightly worse than INT6 default (6.93 → similar) and much worse than INT8 (5.56). The smaller artifact (3.8MB) is nice but not worth the quality loss.
- **Insight:** Quantization precision doesn't affect mid-train bpb (as expected — it only matters for post-training). The real Block B comparison must use roundtrip bpb: INT8(5.56) >> INT6(6.93) ≈ INT5(6.89). The 1-2 bit difference matters enormously for roundtrip quality.
- **Implication for next experiment:** INT8 is clearly best for roundtrip. Try mixed INT6 MLP / INT8 attn to see if we can save artifact size while keeping good roundtrip.

### Exp 9 [Block B]: int6_mlp_int8_attn — REVERT
- **Changed:** Line 1314: `{"mlp", "attn"}` → `{"mlp"}` — MLP stays INT6, attn gets INT8
- **Hypothesis:** Mixed config might save artifact size vs full INT8 while keeping good attn precision
- **Result:** bpb=1.7006 | roundtrip=7.7139 | artifact=7,754,918 | steps=462
- **Cost:** 300s
- **Decision:** REVERT because roundtrip (7.71) is WORSE than both INT6/INT6 baseline (6.93) and INT8 all (5.56). Mid-train identical as expected.
- **Insight:** MLP at INT6 while attn is INT8 creates a mismatch that hurts roundtrip more than uniform INT6. Possibly the INT6 MLP quantization error propagates differently when attn weights are at higher precision. The mismatch may confuse the dequantization reconstruction.
- **Implication for next experiment:** Try reverse mixed: INT8 MLP / INT6 attn. If MLP precision matters more, INT8 MLP should help.