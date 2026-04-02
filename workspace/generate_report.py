#!/usr/bin/env python3
"""Generate research report with per-block ablation tables. Usage: python3 generate_report.py"""
import json, glob, os
from datetime import datetime
from collections import defaultdict

def main():
    exps = []
    for f in sorted(glob.glob("experiments/exp_*.json")):
        try:
            with open(f) as fh: exps.append(json.load(fh))
        except: pass

    if not exps:
        print("No experiments found.")
        return

    succeeded = [e for e in exps if e.get("status") == "success" and e.get("metrics",{}).get("val_bpb")]
    failed = [e for e in exps if e.get("status") != "success"]
    succeeded.sort(key=lambda e: e["metrics"]["val_bpb"])
    best = succeeded[0] if succeeded else None
    baseline = 1.2244
    total_cost = sum(e.get("cost_usd", 0) for e in exps)
    total_time = sum(e.get("duration_seconds", 0) for e in exps)

    # Group by block
    blocks = defaultdict(list)
    for e in exps:
        b = e.get("block", "X")
        blocks[b].append(e)

    os.makedirs("reports", exist_ok=True)

    # ── Build report ──

    report = f"""# Parameter Golf: Activation × Quantization Ablation Study
## Research Report — {datetime.now().strftime('%Y-%m-%d')}

---

### Executive Summary

**{len(exps)} experiments** across {len(blocks)} blocks over **{total_time/3600:.1f} GPU-hours** (${total_cost:.2f}).

"""
    if best:
        report += f"""Best val_bpb: **{best['metrics']['val_bpb']:.4f}** — experiment "{best['name']}" (Block {best.get('block','?')}).
Improvement over 1.2244 baseline: **-{baseline - best['metrics']['val_bpb']:.4f} BPB** ({(baseline - best['metrics']['val_bpb'])/baseline*100:.1f}%).

"""

    report += f"""| Metric | Value |
|--------|-------|
| Total experiments | {len(exps)} |
| Successful | {len(succeeded)} ({len(succeeded)/max(len(exps),1)*100:.0f}%) |
| Crashed / failed | {len(failed)} |
| Best val_bpb | {f"{best['metrics']['val_bpb']:.4f}" if best else '—'} |
| Total GPU time | {total_time/3600:.1f} hours |
| Total cost | ${total_cost:.2f} |

---

"""

    # ── Per-block ablation tables ──

    block_order = sorted(blocks.keys())
    block_names = {
        "A": "Activation Function Screen",
        "B": "Quantization Precision",
        "C": "Magnitude Pruning",
        "D": "GPTQ-lite Clip Search",
        "E": "Final Combination",
    }

    for b in block_order:
        block_exps = blocks[b]
        block_succeeded = [e for e in block_exps if e.get("status") == "success" and e.get("metrics",{}).get("val_bpb")]
        block_succeeded.sort(key=lambda e: e["metrics"]["val_bpb"])
        block_best = block_succeeded[0] if block_succeeded else None

        title = block_names.get(b, f"Block {b}")
        report += f"### Block {b}: {title}\n\n"

        if not block_succeeded:
            report += "No successful experiments in this block.\n\n"
            continue

        report += f"**Winner: {block_best['name']}** (val_bpb = {block_best['metrics']['val_bpb']:.4f})\n\n"
        report += "| # | Experiment | val_bpb | Δ vs Block Best | Artifact | Status |\n"
        report += "|---|-----------|---------|----------------|----------|--------|\n"

        for e in block_exps:
            m = e.get("metrics", {})
            st = e.get("status", "?")
            bpb = f"{m['val_bpb']:.4f}" if m.get("val_bpb") else "—"
            art = f"{m['artifact_bytes']/1e6:.2f}MB" if m.get("artifact_bytes") else "—"

            if m.get("val_bpb") and block_best:
                delta = m["val_bpb"] - block_best["metrics"]["val_bpb"]
                delta_str = f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}"
                if delta == 0:
                    delta_str = "**BEST**"
            else:
                delta_str = "—"

            marker = " ✅" if e == block_best else ""
            report += f"| {e.get('experiment_number','?')} | {e.get('name','?')}{marker} | {bpb} | {delta_str} | {art} | {st} |\n"

        report += "\n"

        # Block insight
        if len(block_succeeded) >= 2:
            worst_in_block = block_succeeded[-1]
            spread = worst_in_block["metrics"]["val_bpb"] - block_best["metrics"]["val_bpb"]
            report += f"**Block spread:** {spread:.4f} BPB between best and worst successful experiment.\n\n"

        report += "---\n\n"

    # ── Improvement timeline ──

    report += "### Improvement Timeline\n\n"
    report += "Each row is a new all-time best:\n\n"
    report += "| # | Experiment | Block | val_bpb | Improvement vs Baseline |\n"
    report += "|---|-----------|-------|---------|------------------------|\n"

    best_so_far = 999
    timeline_count = 0
    for e in exps:
        b = e.get("metrics",{}).get("val_bpb")
        if b and e.get("status") == "success" and b < best_so_far:
            best_so_far = b
            timeline_count += 1
            report += f"| {timeline_count} | {e.get('name','')} | {e.get('block','?')} | {b:.4f} | -{baseline - b:.4f} |\n"

    report += f"""

---

### Methodology

- **Research question:** How does activation function choice interact with quantization quality?
- **Starting point:** Best merged record from the Parameter Golf leaderboard
- **Protocol:** Single-variable experiments within blocks; winners carried forward between blocks
- **Time budget:** 5 minutes per experiment (fixed, comparable)
- **Decision rule:** KEEP if val_bpb improved AND artifact ≤ 16MB; REVERT otherwise
- **Automation:** Claude Code agent following Karpathy's autoresearch pattern
- **Validation:** Fixed 50,000-document FineWeb validation set, sliding window eval (stride=64)

### Cost Breakdown

| Block | Experiments | GPU Time | Cost |
|-------|------------|----------|------|
"""
    for b in block_order:
        block_exps = blocks[b]
        bt = sum(e.get("duration_seconds", 0) for e in block_exps)
        bc = sum(e.get("cost_usd", 0) for e in block_exps)
        report += f"| {b} | {len(block_exps)} | {bt/60:.0f} min | ${bc:.2f} |\n"

    report += f"| **Total** | **{len(exps)}** | **{total_time/60:.0f} min** | **${total_cost:.2f}** |\n"

    # ── Best configuration details ──

    if best:
        report += f"""

---

### Best Configuration

**Experiment:** {best['name']} (Block {best.get('block','?')})
**val_bpb:** {best['metrics']['val_bpb']:.4f}
**Artifact:** {best['metrics'].get('artifact_bytes','?')} bytes
**Parameters:** {best['metrics'].get('total_params','?')}
**Training steps:** {best['metrics'].get('total_steps','?')}

**Config snapshot:**
"""
        config = best.get("config", {})
        for k, v in config.items():
            if v is not None:
                report += f"- {k}: {v}\n"

        diff_file = best.get("diff_file", "")
        if diff_file and os.path.exists(diff_file):
            with open(diff_file) as f:
                diff_content = f.read()[:5000]
            report += f"\n**Code diff from starting baseline:**\n```diff\n{diff_content}\n```\n"

    report += f"""

---

### Reproducing Results

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/parameter-golf.git
cd parameter-golf/workspace

# Checkout the best configuration
git log --oneline | grep "BLOCK\\|KEEP"  # Find the relevant commit
git checkout <commit_hash>

# Run with the same seed
SEED={best.get('seed', 1337) if best else 1337} torchrun --standalone --nproc_per_node=8 train_gpt.py
```

---

*Generated by generate_report.py — {datetime.now().strftime('%Y-%m-%d %H:%M')}*
*{len(exps)} experiments, {total_time/3600:.1f} GPU-hours, ${total_cost:.2f} total cost*
"""

    with open("reports/research_report.md", "w") as f:
        f.write(report)

    # ── PR summary ──

    summary = f"## Activation × Quantization Ablation Study\n\n"
    summary += f"**{len(exps)} experiments** → best val_bpb **{best['metrics']['val_bpb']:.4f}**\n\n" if best else ""
    summary += "### Per-block winners:\n"
    for b in block_order:
        block_succeeded = [e for e in blocks[b] if e.get("status") == "success" and e.get("metrics",{}).get("val_bpb")]
        if block_succeeded:
            block_succeeded.sort(key=lambda e: e["metrics"]["val_bpb"])
            bb = block_succeeded[0]
            summary += f"- **Block {b}** ({block_names.get(b,'')}): {bb['name']} → {bb['metrics']['val_bpb']:.4f}\n"

    with open("reports/pr_summary.md", "w") as f:
        f.write(summary)

    print(f"Report:     reports/research_report.md")
    print(f"PR summary: reports/pr_summary.md")
    if best:
        print(f"Best:       {best['name']} (Block {best.get('block','?')}) at {best['metrics']['val_bpb']:.4f} BPB")
    print(f"Cost:       ${total_cost:.2f} across {len(exps)} experiments")

if __name__ == "__main__":
    main()