# Parameter Golf: Ablation Study Results

**Author:** [Abishek Satnur](https://github.com/abi2024) | **Blog:** [Day 0](https://substack.com) · [Day 1](https://substack.com) · [Day 2](https://substack.com)
**Hardware:** 1×H100 SXM 80GB on RunPod | **Base config:** [1.1233 merged record](https://github.com/openai/parameter-golf/blob/main/records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md)

> **Important:** All numbers below are from 1-GPU development runs (~450 steps in 5 minutes). Absolute BPB values are not comparable to the 8-GPU leaderboard (~3,700 steps in 10 minutes). Relative rankings between experiments are valid — all ran under identical conditions.

---

## Key Finding: The 1-GPU Dev Gap

EMA weight averaging contaminates 1-GPU roundtrip BPB. Disabling it reveals the actual quantization quality.

| Config | Mid-train BPB | Post-EMA BPB | Roundtrip BPB | Steps |
|--------|:---:|:---:|:---:|:---:|
| LeakyReLU + INT8 MLP + XSA-4 · **EMA ON** | 1.7153 | 5.5975 | **5.3290** | 452 |
| LeakyReLU + INT8 MLP + XSA-4 · **EMA OFF** | 1.7157 | 1.7157 | **2.3151** | 451 |

**EMA adds 3.01 BPB of noise** to 1-GPU roundtrip (5.33 → 2.32). That's 56% of the total score.

**Why:** EMA decay of 0.997 over 451 steps means step 1 contributes 0.997⁴⁵¹ ≈ 25% of the final average. Step 1 had loss 6.93 (random initialization). The EMA is averaging garbage with gold.

**Fix:** Disable EMA for 1-GPU development. Use mid-train BPB for architectural comparisons. Re-enable EMA only for 8-GPU submission runs where 0.997³⁷⁰⁰ ≈ 0.001% makes early steps negligible.

---

## All Day 2 Experiments (EMA disabled, clean metrics)

All experiments: seed=1337, wall-clock=300s, 1×H100, EMA off.

| # | Experiment | Activation | Quantization | XSA | Mid-train BPB | Roundtrip BPB | Steps |
|---|-----------|:---:|:---:|:---:|:---:|:---:|:---:|
| ref | Day 1 default (EMA ON) | LeakyReLU²(0.5) | INT8 MLP / INT6 attn | last 4 | 1.7153 | 5.3290 | 452 |
| 1 | EMA disabled baseline | LeakyReLU²(0.5) | INT8 MLP / INT6 attn | last 4 | 1.7157 | 2.3151 | 451 |
| 2 | INT6 uniform | LeakyReLU²(0.5) | INT6 / INT6 | last 4 | 1.7153 | 3.1815 | 452 |
| 3 | relu² activation | relu² | INT8 MLP / INT6 attn | last 4 | 1.7244 | 2.2781 | 451 |
| 4 | XSA-all | LeakyReLU²(0.5) | INT8 MLP / INT6 attn | all 11 | 1.7296 | **2.1366** | 436 |

---

## Finding 1: INT8 MLP quantization (confirmed)

| Config | Roundtrip BPB | Delta |
|--------|:---:|:---:|
| **INT8 MLP / INT6 attn** | **2.3151** | — |
| INT6 / INT6 | 3.1815 | +0.87 (worse) |

INT8 MLP beats INT6 MLP by **0.87 BPB** on clean metrics. MLP weights are more sensitive to quantization than attention weights.

## Finding 2: Activation × compression tradeoff

| Activation | Mid-train BPB | Roundtrip BPB | Verdict |
|-----------|:---:|:---:|---|
| LeakyReLU²(0.5) | **1.7157** (better) | 2.3151 | Trains better, compresses worse |
| relu² | 1.7244 | **2.2781** (better) | Trains worse, compresses better |

**Better training ≠ better compression.** LeakyReLU² produces denser weights that are harder to quantize. For the compressed model that gets submitted, relu² wins by 0.037 BPB.

## Finding 3: XSA-all

| Config | Roundtrip BPB | Delta |
|--------|:---:|:---:|
| XSA on last 4 layers | 2.3151 | — |
| **XSA on all 11 layers** | **2.1366** | **-0.1785** |

One env var change: `XSA_LAST_N=11`. Validates [PR #609](https://github.com/openai/parameter-golf/pull/609) on this config.

Note: XSA-all used 689ms/step vs 665ms for XSA-4, resulting in 436 steps vs 451. The roundtrip improvement (-0.18) is large enough to be real despite 15 fewer training steps.

---

## File Reference

| File | Activation | Quantization | XSA | EMA | Purpose |
|------|:---:|:---:|:---:|:---:|---------|
| `train_gpt.py` | LeakyReLU²(0.5) | INT8 MLP / INT6 attn | last 4 | ON | Day 1 config |
| `train_gpt_no_ema.py` | LeakyReLU²(0.5) | INT8 MLP / INT6 attn | last 4 | **OFF** | Day 2 methodology fix |
| `train_gpt_day1_baseline.py` | LeakyReLU²(0.5) | INT8 MLP / INT6 attn | last 4 | ON | Saved Day 1 copy |
| `train_gpt_relu2.py` | relu² | INT8 MLP / INT6 attn | last 4 | ON | Activation comparison |
| `train_gpt_relu2_no_ema.py` | relu² | INT8 MLP / INT6 attn | last 4 | **OFF** | Clean activation test |
| `train_gpt_int6_no_ema.py` | LeakyReLU²(0.5) | INT6 / INT6 | last 4 | **OFF** | Clean quantization test |
| `train_gpt_xsa_all_no_ema.py` | LeakyReLU²(0.5) | INT8 MLP / INT6 attn | all 11 | **OFF** | Frontier technique test |
| `train_gpt_block_baseline.py` | LeakyReLU²(0.5) | INT8 MLP / INT6 attn | last 4 | ON | Day 1 block baseline snapshot |

### How to disable EMA

Find this block in `main()`:

```python
# Apply EMA weights (better than SWA alone per PR#401)
log0("ema:applying EMA weights")
current_state = base_model.state_dict()
avg_state = {name: t.to(dtype=current_state[name].dtype) for name, t in ema_state.items()}
base_model.load_state_dict(avg_state, strict=True)
```

Replace with:

```python
# EMA DISABLED for 1-GPU development
log0("ema:SKIPPED (disabled for dev gap test)")
current_state = base_model.state_dict()
```

---

## Day 1 Experiments (19 total, EMA on)

See `experiments/notes.md` for detailed per-experiment notes including hypotheses, results, and insights.

See `logs/` for full training logs.

See `day2_results/` for structured JSON results from Day 2.

---

## Cost

| Phase | Experiments | GPU time | Cost |
|-------|:---:|:---:|:---:|
| Day 1 (19 experiments, autoresearch) | 19 | ~3 hours | ~$7.20 |
| Day 2 (EMA confirmation + clean metrics) | 5 | ~1 hour | ~$2.40 |
| Day 2 (failed EMA_ENABLED=0 attempts) | 4 | ~30 min | ~$1.20 |
| **Total** | **28** | **~4.5 hours** | **~$10.80** |
