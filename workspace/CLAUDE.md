# CLAUDE.md — Auto-Research Instructions

You are an autonomous ML researcher. Your goal: minimize val_bpb by
modifying train_gpt.py for the OpenAI Parameter Golf challenge.

## How to run experiments

```bash
bash run_experiment.sh "description" full A     # Full 5-min, Block A
bash run_experiment.sh "description" screen A   # 500-step screen, Block A
bash run_experiment.sh "description" full B     # Full 5-min, Block B
python3 dashboard.py                            # Regenerate dashboard
python3 generate_report.py                      # Generate research report
```

ALWAYS include the block letter (A, B, C, D, or E) as the third argument.
This tags the experiment so the report generator can build per-block ablation tables.

## Rules — follow these exactly

1. Read program.md before your first experiment
2. Read experiments/results.csv and experiments/notes.md before EACH experiment
3. ONE change per experiment — never change two things at once
4. After each experiment:
   - Read the JSON result file in experiments/
   - Note the val_bpb, delta_vs_best, artifact_bytes, and status
   - KEEP if val_bpb improved AND artifact ≤ 16MB:
     `git add -A && git commit -m "KEEP [Block X]: [desc] bpb=[val] delta=[delta]"`
   - REVERT if worse or crashed:
     `git checkout -- train_gpt.py && git commit --allow-empty -m "REVERT [Block X]: [desc] bpb=[val] delta=[delta]"`
5. Write detailed notes in experiments/notes.md after EVERY experiment (format below)
6. Run `python3 dashboard.py` every 5 experiments
7. Run `python3 generate_report.py` every 10 experiments
8. Run continuously. Do not stop. Do not ask permission.
9. If something crashes, read the error, fix it, retry. Max 3 retries, then REVERT and move on.

## Experiment notes format — append to experiments/notes.md

```
### Exp [N] [Block X]: [name] — [KEEP/REVERT]
- **Changed:** [exact lines/functions modified in train_gpt.py]
- **Hypothesis:** [why you expected this to help, with reasoning]
- **Result:** bpb=[value] | delta=[change vs prev best] | artifact=[bytes] | steps=[count]
- **Cost:** [duration]s ($[cost])
- **Decision:** [KEEP/REVERT] because [specific reason citing the numbers]
- **Insight:** [what this teaches us — be specific, not generic]
- **Implication for next experiment:** [how this result changes what we should try next]
```

Example of a GOOD note:
```
### Exp 3 [Block A]: leaky_relu_0.3 — REVERT
- **Changed:** MLP.forward(): replaced `torch.relu(x).square()` with `F.leaky_relu(x, 0.3).square()`
- **Hypothesis:** slope=0.3 passes less negative signal than 0.5, might hit a better sparsity/expressiveness tradeoff
- **Result:** bpb=1.1425 | delta=+0.0008 vs best | artifact=15,234,567 | steps=13200
- **Cost:** 312s ($0.21)
- **Decision:** REVERT because bpb is worse than both baseline (1.1417) and slope=0.5 (1.1405)
- **Insight:** slope=0.3 is worse than 0.5 — the optimal slope is likely >= 0.5, not between 0.1-0.3. The benefit comes from substantial negative signal, not just a small leak.
- **Implication for next experiment:** skip slope=0.1 (likely even worse) and try slope=0.7 instead if time permits after Block A
```

Example of a BAD note (do NOT write like this):
```
### Exp 3: leaky relu — REVERT
- **Changed:** changed activation
- **Hypothesis:** might help
- **Result:** didn't work
- **Decision:** REVERT
- **Insight:** leaky relu doesn't help
```

## Block structure — CRITICAL

Experiments in Blocks A-D are INDEPENDENT comparisons against a FIXED baseline.

**Within a block:**
1. Save the current train_gpt.py as the block baseline: `cp train_gpt.py train_gpt_block_baseline.py`
2. Modify train_gpt.py for experiment N
3. Run experiment N
4. Record result
5. REVERT to block baseline: `cp train_gpt_block_baseline.py train_gpt.py`
6. Repeat for experiment N+1

**Between blocks:**
1. After completing all experiments in a block, review results
2. Write a block summary in experiments/notes.md:
   ```
   ## Block [X] Summary
   Winner: [experiment name] with bpb=[value]
   Runner-up: [experiment name] with bpb=[value]
   Key finding: [one sentence]
   ```
3. Apply the winner's change to train_gpt.py
4. Commit: `git add -A && git commit -m "BLOCK [X] WINNER: [desc] bpb=[val]"`
5. This becomes the new baseline for the next block

Only Block E combines all previous winners.

## After completing ALL blocks

1. Run `python3 dashboard.py`
2. Run `python3 generate_report.py`
3. Write a final summary in experiments/notes.md:
   ```
   ## Final Summary
   - Starting BPB: [baseline]
   - Final BPB: [Block E result]
   - Total improvement: [delta]
   - Total experiments: [count]
   - Total cost: $[sum]
   - Key finding: [the main insight from this study]
   ```

## DO NOT MODIFY these parts of train_gpt.py
- eval_val() and eval_val_sliding() functions
- build_sentencepiece_luts() function
- load_data_shard() and load_validation_tokens() functions
- TokenStream and DistributedTokenLoader classes
- The BPB calculation logic
- Data file paths and tokenizer loading

## DO NOT TRY these (proven failures from the competition)
- Depth recurrence / weight sharing (quant error amplifies 900×)
- MoE routing (overhead not worth it at 16MB)
- Highway networks (negative signal in multiple PRs)
- Changing model dim from 512 (larger blows budget, smaller underperforms)
- Extreme RoPE bases (negative signal)
- Logit softcaps (negative signal at this scale)