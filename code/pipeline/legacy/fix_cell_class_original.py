"""Retroactive fix: add cell_class_original and recompute cell_class in integrated.h5ad.

For pipeline runs produced before the cell_class update fix was introduced, this
script applies the same correction that the updated pipeline now performs
automatically after each scANVI label transfer:

  1. Reads integrated.h5ad (fully into memory so obs can be modified).
  2. Saves the pre-transfer cell_class as cell_class_original (skipped if the
     column already exists, unless --force is given).
  3. Recomputes cell_class from cell_type_aligned via aligned_to_class.
  4. Overwrites integrated.h5ad in-place with the corrected obs columns.
  5. Re-runs scanvi_diagnostics to regenerate all plots (including UMAPs)
     with the corrected cell_class so they overwrite the stale versions.

Usage
-----
  # Fix a single integrated.h5ad directly:
  PYTHONPATH=code python -m pipeline.fix_cell_class_original \\
      --input /path/to/scvi_output/integrated.h5ad \\
      --output_dir /path/to/scanvi_diagnostics

  # Derive paths from a pipeline config YAML (most convenient):
  PYTHONPATH=code python -m pipeline.fix_cell_class_original \\
      --config code/pipeline/hpc_config.yaml

  # Skip re-running diagnostics (fix the h5ad only):
  PYTHONPATH=code python -m pipeline.fix_cell_class_original \\
      --config code/pipeline/hpc_config.yaml --no_diagnostics

  # Force re-apply even if cell_class_original already exists:
  PYTHONPATH=code python -m pipeline.fix_cell_class_original \\
      --config code/pipeline/hpc_config.yaml --force
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import scanpy as sc
import yaml

from pipeline.label_transfer.transfer import aligned_to_class


# ── helpers ─────────────────────────────────────────────────────────────────


def _apply_fix(h5ad_path: Path, force: bool = False) -> bool:
    """Read, fix, and overwrite integrated.h5ad.

    Returns True if the file was modified, False if it was already up-to-date
    (i.e. cell_class_original was present and --force was not given).
    """
    print(f"Loading {h5ad_path} …")
    adata = sc.read_h5ad(str(h5ad_path))
    print(f"  {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")

    # Validate required columns
    for col in ('cell_type_aligned', 'cell_class'):
        if col not in adata.obs.columns:
            print(f"ERROR: required column '{col}' not found in obs.")
            print(f"  Available: {sorted(adata.obs.columns.tolist())}")
            sys.exit(1)

    already_fixed = 'cell_class_original' in adata.obs.columns
    if already_fixed and not force:
        print(
            "  cell_class_original already present — file is up-to-date.\n"
            "  Pass --force to overwrite."
        )
        return False

    # Compute new cell_class from cell_type_aligned
    new_class = adata.obs['cell_type_aligned'].map(aligned_to_class)
    old_class = adata.obs['cell_class'].astype(str)
    n_changed = (old_class != new_class.astype(str)).sum()

    if already_fixed and force:
        print(
            f"  --force set: overwriting cell_class_original "
            f"(was already present)"
        )

    # Preserve pre-transfer cell_class
    adata.obs['cell_class_original'] = adata.obs['cell_class'].copy()
    # Overwrite cell_class with corrected values
    adata.obs['cell_class'] = pd.Categorical(new_class)

    print(
        f"  cell_class updated for {n_changed:,} / {len(adata):,} cells "
        f"({n_changed / len(adata):.1%})"
    )
    print(f"  Writing fixed h5ad back to {h5ad_path} …")
    adata.write_h5ad(str(h5ad_path))
    print("  Done.")
    return True


def _run_diagnostics(h5ad_path: Path, output_dir: Path,
                     confidence_threshold: float = 0.5):
    """Re-run scanvi_diagnostics to regenerate all plots (overwrites stale PNGs)."""
    import subprocess
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, '-m', 'pipeline.scanvi_diagnostics',
        '--input', str(h5ad_path),
        '--output_dir', str(output_dir),
        '--confidence_threshold', str(confidence_threshold),
    ]
    print(f"\nRunning diagnostics: {' '.join(cmd)}")
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        print(
            f"WARNING: scanvi_diagnostics exited with code {result.returncode}. "
            "Plots may be incomplete."
        )


# ── main ────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(
        description='Retroactively fix cell_class_original in integrated.h5ad',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Input: either a direct path or a pipeline config that resolves the paths
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--input', metavar='H5AD',
                     help='Path to integrated.h5ad to fix in-place')
    src.add_argument('--config', metavar='YAML',
                     help='Pipeline config YAML; output_dir/scvi_output/integrated.h5ad '
                          'is fixed and output_dir/scanvi_diagnostics is used for plots')

    p.add_argument('--output_dir', metavar='DIR',
                   help='Directory for diagnostic plots (default: <h5ad_dir>/../scanvi_diagnostics). '
                        'Ignored when --config is used.')
    p.add_argument('--confidence_threshold', type=float, default=0.5,
                   help='Confidence threshold forwarded to scanvi_diagnostics (default: 0.5)')
    p.add_argument('--no_diagnostics', action='store_true',
                   help='Skip re-running scanvi_diagnostics after fixing the h5ad')
    p.add_argument('--force', action='store_true',
                   help='Re-apply fix even if cell_class_original already exists')
    args = p.parse_args()

    # Resolve paths
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        pipeline_output_dir = Path(cfg['output_dir'])
        h5ad_path = pipeline_output_dir / 'scvi_output' / 'integrated.h5ad'
        diag_dir = pipeline_output_dir / 'scanvi_diagnostics'
        confidence_threshold = cfg.get('scanvi_label_transfer', {}).get(
            'confidence_threshold', args.confidence_threshold
        )
    else:
        h5ad_path = Path(args.input)
        diag_dir = (
            Path(args.output_dir) if args.output_dir
            else h5ad_path.parent.parent / 'scanvi_diagnostics'
        )
        confidence_threshold = args.confidence_threshold

    if not h5ad_path.exists():
        print(f"ERROR: File not found: {h5ad_path}")
        sys.exit(1)

    # Apply the fix
    modified = _apply_fix(h5ad_path, force=args.force)

    # Re-run diagnostics if the file was modified (or --force was set)
    if not args.no_diagnostics and (modified or args.force):
        _run_diagnostics(h5ad_path, diag_dir, confidence_threshold)
    elif args.no_diagnostics:
        print("Skipping diagnostics (--no_diagnostics).")
    else:
        print("File unchanged; skipping diagnostics.")


if __name__ == '__main__':
    main()
