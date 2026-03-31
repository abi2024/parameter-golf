# Research Program: Activation × Quantization Interaction

## Research Question
How does activation function choice interact with quantization quality
in a 16MB compressed transformer?

## Starting Point
Best available merged record from the Parameter Golf leaderboard.
Current baseline val_bpb will be established in Experiment 1.

## Experiment Plan — run in EXACT order

### Block A: Activation function screen (5 experiments)

Each experiment: change ONLY the activation function.
Revert to baseline between experiments.

1. Baseline relu² — full 5-min run (establish reference BPB)
2. LeakyReLU²(negative_slope=0.5) — full run
3. LeakyReLU²(negative_slope=0.3) — full run
4. LeakyReLU²(negative_slope=0.1) — full run
5. SwiGLU (F.silu(gate) * up) — full run

After Block A: identify the best activation. Apply it. Commit as new baseline.

### Block B: Best activation × quantization precision (6 experiments)

Each experiment: change ONLY the quantization settings.
Revert to Block B baseline between experiments.

6.  Best activation + INT5 MLP / INT6 attn (current default)
7.  Best activation + INT6 MLP / INT6 attn (uniform INT6)
8.  Best activation + INT5 MLP / INT5 attn (aggressive)
9.  Best activation + INT8 MLP / INT8 attn (conservative)
10. Best activation + INT5 MLP / INT8 attn (mixed)
11. Best activation + INT6 MLP / INT8 attn (mixed moderate)

After Block B: identify the best quantization combo. Apply it. Commit.

### Block C: Pruning interaction (4 experiments)

12. Best combo + 0% magnitude pruning (disable pruning entirely)
13. Best combo + 3% pruning (current default)
14. Best combo + 5% pruning
15. Best combo + 10% pruning

After Block C: identify best pruning level. Apply it. Commit.

### Block D: GPTQ-lite clip search (4 experiments)

16. Best combo + 3 clip percentiles
17. Best combo + 5 clip percentiles (current default)
18. Best combo + 7 clip percentiles
19. Best combo + 10 clip percentiles

After Block D: identify best GPTQ-lite setting. Apply it. Commit.

### Block E: Final combination (1 experiment)

20. Stack ALL winners from Blocks A+B+C+D
    This is the final configuration. Run as full 5-min experiment.
    If it beats all individual experiments, this is our submission candidate.

## Success Criteria
- val_bpb < 1.12 = submit as non-record with ablation study
- val_bpb < 1.11 = submit as record candidate
- Any result with a clean ablation table = submit regardless of number

## Kill Criteria — REVERT immediately if
- Loss > 6.0 at step 100
- Loss is NaN or Inf
- Artifact > 16,000,000 bytes
- No val_bpb in output (script crashed)
- val_bpb worse than baseline by > 0.02