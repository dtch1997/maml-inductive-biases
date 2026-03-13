"""Parse 1000-step MAML output and save CSV for plotting."""

import csv
import re

OUTPUT_FILE = "/private/tmp/claude-501/-Users-daniel-github-workspace-autoresearch/488cb43c-cfcb-429c-b703-2163df7f85d9/tasks/bpd6u4e1w.output"

rows = []
pattern = re.compile(
    r"outer\s+(\d+)\s+\|\s+pre:\s+train=([\d.]+)\s+rel=([\d.]+)\s+\|\s+post:\s+train=([\d.]+)\s+rel=([\d.]+)\s+\|\s+outer_loss=([-\d.]+)"
)

# Also grab base losses from the "Initial" line
base_pattern = re.compile(r"Initial.*train loss:\s+([\d.]+),\s+related loss:\s+([\d.]+)")

base_train = base_related = None

with open(OUTPUT_FILE) as f:
    for line in f:
        m = base_pattern.search(line)
        if m:
            base_train, base_related = float(m.group(1)), float(m.group(2))
        m = pattern.search(line)
        if m:
            rows.append({
                "outer_step": int(m.group(1)),
                "pre/loss_train": float(m.group(2)),
                "pre/loss_related": float(m.group(3)),
                "post/loss_train": float(m.group(4)),
                "post/loss_related": float(m.group(5)),
                "outer_loss": float(m.group(6)),
                "base/loss_train": base_train,
                "base/loss_related": base_related,
            })

out = "results/maml_metrics_1000.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

print(f"Saved {len(rows)} rows to {out}")
