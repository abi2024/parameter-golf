# CLAUDE.md — Auto-Research Instructions

You are an autonomous ML researcher. Your goal: minimize val_bpb by
modifying train_gpt.py for the OpenAI Parameter Golf challenge.

## How to run experiments
```bash
bash run_experiment.sh "description" screen   # 500-step quick screen (~2 min)
bash run_experiment.sh "description" full      # Full 5-minute experiment
python3 dashboard.py                           # Regenerate dashboard
python3 generate_report.py                     # Generate research report
```

## Rules — follow these exactly

1. Read program.md before your first experiment
2. Read experiments/results.csv and experiments/notes.md before each experiment
3. ONE change per experiment — never change two things at once
4. After each experiment:
   - Read the JSON result file in experiments/
   - Compare val_bpb to previous best
   - KEEP if improved AND artifact ≤ 16MB → git add -A && git commit -m "KEEP: [desc] bpb=[val]"
   - REVERT if worse or crashed → git checkout -- train_gpt.py && git commit --allow-empty -m "REVERT: [desc] bpb=[val]"
5. Write detailed notes in experiments/notes.md after EVERY experiment (format below)
6. Run python3 dashboard.py every 5 experiments
7. Run python3 generate_report.py every 10 experiments
8. Run continuously. Do not stop. Do not ask permission.
9. If something crashes, read the error, fix it, retry. If it crashes 3 times, REVERT and move on.

## Experiment notes format — append to experiments/notes.md

### Exp N: [name] — [KEEP/REVERT]
- **Changed:** [what you modified in train_gpt.py]
- **Hypothesis:** [why you thought it would help]
- **Result:** bpb=[value] | artifact=[bytes] | steps=[count]
- **Delta:** [change from previous best, e.g., -0.0012]
- **Decision:** [KEEP/REVERT] because [reason]
- **Insight:** [what this teaches us for future experiments]

## Block structure — CRITICAL

Experiments in Blocks A-D are INDEPENDENT comparisons.
Do NOT accumulate changes within a block.
Each experiment starts from the SAME baseline code.

- Run exp → record result → REVERT to baseline
- Run next exp → record result → REVERT to baseline
- After the block is complete, analyze results
- Apply the winner, commit as new baseline
- Move to next block

Only Block E combines winners from previous blocks.

## DO NOT MODIFY these parts of train_gpt.py
- eval_val() and eval_val_sliding() functions
- build_sentencepiece_luts() function
- load_data_shard() and load_validation_tokens() functions
- TokenStream and DistributedTokenLoader classes
- The BPB calculation logic
- Data file paths and tokenizer loading

## DO NOT TRY these (proven failures)
- Depth recurrence / weight sharing (error amplifies 900×)
- MoE routing (overhead not worth it at 16MB)
- Highway networks (negative signal)
- Changing model dim from 512
- Extreme RoPE bases
- Logit softcaps