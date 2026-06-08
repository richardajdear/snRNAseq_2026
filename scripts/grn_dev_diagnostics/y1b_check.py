import anndata as ad
import pandas as pd
import numpy as np
from pathlib import Path
OUT = Path("scripts/grn_dev_diagnostics/outputs")
p = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/VelPsychAD_V3_allcell_dev30/scvi_output/integrated.h5ad"
a = ad.read_h5ad(p, backed="r")
obs = a.obs[["source", "cell_class", "age_years", "cell_type_raw"]].copy()
exn = set(pd.read_parquet(OUT / "y1_exn_cellids_ALL.parquet").index.astype(str))
obs["is_exn"] = obs.index.astype(str).isin(exn)

print("=== native cell_class composition per source (%) ===")
print((obs.groupby("source")["cell_class"].value_counts(normalize=True).unstack().fillna(0) * 100).round(1).to_string())

print("\n=== y1 ExN selection RECALL on native cell_class (fraction selected as ExN) ===")
rec = obs.groupby(["source", "cell_class"])["is_exn"].mean().unstack().fillna(0).round(3)
print((rec * 100).round(1).to_string())

print("\n=== ExN fraction by source x age-bin ===")
obs["agebin"] = pd.cut(obs["age_years"], [-1, 0, 1, 10, 18, 31],
                       labels=["prenatal", "0-1", "1-10", "10-18", "18-30"])
print((obs.groupby(["source", "agebin"])["is_exn"].agg(["mean", "size"])).round(3).to_string())

# Of native-Excitatory cells NOT selected, where did they go (age)?
miss = obs[(obs["cell_class"] == "Excitatory") & (~obs["is_exn"])]
print(f"\nnative-Excitatory NOT selected as ExN: {len(miss):,}")
print("  by source:", miss["source"].value_counts().to_dict())
print("  PsychAD missed by agebin:", miss[miss.source=="PSYCHAD"]["agebin"].value_counts().to_dict())
