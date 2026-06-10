#!/usr/bin/env python3
"""Plot the 8-metric grid (s05) as heatmaps: depth-robustness and within-EN age
trend, for {C3+,signed} x {linear,log} x {AGG,CELL} aggregation x 3 cohorts."""
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import _lib_c3 as L

df = pd.read_csv(L.OUT_DIR / "s05_metric_grid_summary.csv")
order = ["AGG_pos_lin", "AGG_pos_log", "AGG_sign_lin", "AGG_sign_log",
         "CELL_pos_lin", "CELL_pos_log", "CELL_sign_lin", "CELL_sign_log"]
cohorts = ["Herring-V3", "PsychAD-V3", "U01-V2"]

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
for ax, (col, title, cmap, vlim) in zip(axes, [
        ("rho_depth_ctrl_age", "Depth sensitivity  rho(score, depth | age)\n(closer to 0 = better)", "Reds", (0, 0.75)),
        ("rho_age_ctrl_depth", "Within-EN age trend  rho(score, age | depth)\n(neg = decline)", "RdBu_r", (-0.5, 0.5))]):
    M = df.pivot(index="metric", columns="cohort", values=col).reindex(order)[cohorts]
    im = ax.imshow(M.values, cmap=cmap, aspect="auto",
                   vmin=vlim[0], vmax=vlim[1])
    ax.set_xticks(range(len(cohorts))); ax.set_xticklabels(cohorts, rotation=20)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order)
    for i in range(len(order)):
        for j in range(len(cohorts)):
            ax.text(j, i, f"{M.values[i,j]:+.2f}", ha="center", va="center", fontsize=9)
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=.046)
    # separate AGG (top 4) from CELL (bottom 4)
    ax.axhline(3.5, color="k", lw=1.5)
fig.suptitle("8-metric grid: pole(C3+/signed) x transform(lin/log) x aggregation(AGG=sum-then-CPM / CELL=per-cell-CPM-then-mean)\n"
             "within mature EN, postnatal — note the age trend FLIPS sign under CELL-log in the V3 cohorts",
             y=1.02, fontsize=10)
fig.tight_layout()
fig.savefig(L.OUT_DIR / "s05_metric_grid.png", dpi=140, bbox_inches="tight")
print("saved s05_metric_grid.png")
