#!/usr/bin/env python3
"""Driver: build neuron+progenitor trajectory manifolds for the c3_maturation
investigation. The actual prep is the STANDALONE module `code/trajectory/manifold.py`
(no scripts/ dependency); this file only selects the c3_maturation datasets, points
outputs at output/c3_maturation/, and runs all-cells when asked.

SUBMIT (all V3 cells incl. prenatal):
  sbatch --time=02:00:00 --mem=200G --partition=icelake scripts/run_script.sh \
      scripts/c3_maturation/s11_neuron_manifold.py --only Velmeshev-V3 --all-cells
"""
import sys
import argparse
from pathlib import Path

ROOT = Path("/home/rajd2/rds/hpc-work/snRNAseq_2026")
sys.path.insert(0, str(ROOT / "code/trajectory"))
import manifold as M   # standalone prep (no scripts/ deps)

B = Path("/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated")
OUT_DIR = ROOT / "output/c3_maturation"

DATASETS = {
    "PsychAD": dict(integrated=B / "PsychAD_noage_tuning5/scvi_output/integrated.h5ad", chem=None),
    "Velmeshev-V3": dict(integrated=B / "Vel_prepost_noage_tuning5/scvi_output/integrated.h5ad", chem="V3"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="run only this dataset")
    ap.add_argument("--all-cells", action="store_true",
                    help="keep all cells incl. prenatal/NaN-age, no subsample")
    ap.add_argument("--no-c3", action="store_true")
    a = ap.parse_args()
    for name, cfg in DATASETS.items():
        if a.only and name != a.only:
            continue
        out = cfg["integrated"].parent / "trajectory" / "neuron_manifold.h5ad"
        M.build_manifold(cfg["integrated"], out, dataset=name, chem=cfg["chem"],
                         all_cells=a.all_cells, compute_c3=not a.no_c3,
                         fig_dir=OUT_DIR, fig_prefix="s11")
    print(f"\nFigures -> {OUT_DIR}")


if __name__ == "__main__":
    main()
