"""
Test script: CellRank 2 pipeline on a downsampled dataset.

If the real integrated.h5ad is available (from the scVI/scANVI pipeline),
it is subsampled to ``n_cells`` cells before running CellRank.  Otherwise
a synthetic AnnData is generated so the pipeline logic can be verified
without access to the HPC data.

Usage (from project root):
    PYTHONPATH=code python -m CellRank2.test_pipeline

    # Point at the real data (subsample to 2000 cells):
    PYTHONPATH=code python -m CellRank2.test_pipeline \\
        --input rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/combined/pipeline_test/scvi_output/integrated.h5ad \\
        --n_cells 2000 \\
        --output_dir /tmp/cellrank_test
"""

import argparse
import logging
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

# Add code/ to path when run directly
_THIS_DIR = Path(__file__).resolve().parent
_CODE_DIR = _THIS_DIR.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from CellRank2.config import CellRankConfig
from CellRank2.run_pipeline import run
from CellRank2.utils import setup_logger


# ── Synthetic data generator ──────────────────────────────────────────────────

def make_synthetic_adata(
    n_cells: int = 1000,
    n_genes: int = 200,
    n_latent: int = 30,
    random_seed: int = 42,
) -> ad.AnnData:
    """Generate a minimal synthetic AnnData for pipeline testing.

    Cells are spread across 6 age bins and 4 broad cell types.  A noisy
    latent space (X_scANVI) and UMAP (X_umap_scanvi) are generated so that
    all downstream CellRank steps run without needing real data.
    """
    rng = np.random.RandomState(random_seed)

    # Gene expression (sparse counts)
    rng_sparse = np.random.default_rng(random_seed)
    nnz = int(n_cells * n_genes * 0.1)
    rows = rng_sparse.integers(0, n_cells, nnz)
    cols = rng_sparse.integers(0, n_genes, nnz)
    data = rng_sparse.exponential(2, nnz).astype(np.float32)
    X = sp.csr_matrix((data, (rows, cols)), shape=(n_cells, n_genes))

    # Age spanning prenatal → adult (years)
    age_values = np.concatenate([
        rng.uniform(0.0, 0.4, n_cells // 6),    # prenatal (GW as fraction)
        rng.uniform(0.4, 2.0, n_cells // 6),    # early infancy
        rng.uniform(2.0, 10.0, n_cells // 6),   # childhood
        rng.uniform(10.0, 20.0, n_cells // 6),  # adolescence
        rng.uniform(20.0, 40.0, n_cells // 6),  # adult
        rng.uniform(40.0, 80.0, n_cells - 5 * (n_cells // 6)),  # older adult
    ])
    rng.shuffle(age_values)

    # Cell types with realistic proportions including layer-specific excitatory subtypes
    cell_types_pool = [
        "ExcitatoryNeuron_L2-3", "ExcitatoryNeuron_L4-6",
        "InhibitoryNeuron", "Astrocyte", "Oligodendrocyte",
    ]
    cell_types = rng.choice(cell_types_pool, size=n_cells,
                            p=[0.25, 0.20, 0.20, 0.20, 0.15])

    # Batch (source)
    sources = rng.choice(["WANG", "VELMESHEV", "AGING"], size=n_cells,
                         p=[0.4, 0.3, 0.3])

    obs = pd.DataFrame(
        {
            "age_years": age_values.astype(np.float32),
            "cell_type_aligned": pd.Categorical(cell_types),
            "source": pd.Categorical(sources),
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )

    var = pd.DataFrame(
        {"gene_name": [f"gene_{i}" for i in range(n_genes)]},
        index=[f"gene_{i}" for i in range(n_genes)],
    )

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.layers["counts"] = X.copy()

    # Synthetic scANVI latent: cell-type mean + age trend + noise
    type_means = {
        "ExcitatoryNeuron_L2-3": rng.randn(n_latent) * 2,
        "ExcitatoryNeuron_L4-6": rng.randn(n_latent) * 2,
        "InhibitoryNeuron": rng.randn(n_latent) * 2,
        "Astrocyte": rng.randn(n_latent) * 2,
        "Oligodendrocyte": rng.randn(n_latent) * 2,
    }
    latent = np.zeros((n_cells, n_latent), dtype=np.float32)
    for i, ct in enumerate(cell_types):
        age_effect = (age_values[i] / 80.0) * rng.randn(n_latent) * 0.5
        latent[i] = type_means[ct] + age_effect + rng.randn(n_latent) * 0.3
    adata.obsm["X_scANVI"] = latent

    # Compute kNN + UMAP from the synthetic latent space
    sc.pp.neighbors(adata, use_rep="X_scANVI", n_neighbors=15,
                    key_added="neighbors_scanvi", random_state=random_seed)
    sc.tl.umap(adata, neighbors_key="neighbors_scanvi", random_state=random_seed)
    adata.obsm["X_umap_scanvi"] = adata.obsm["X_umap"].copy()

    return adata


# ── Real data loader with subsampling ─────────────────────────────────────────

def load_and_downsample(
    path: str,
    n_cells: int,
    random_seed: int,
    logger: logging.Logger,
) -> ad.AnnData:
    """Load the integrated h5ad and subsample to n_cells."""
    logger.info(f"Loading: {path}")
    adata = ad.read_h5ad(path)
    logger.info(f"  Loaded: {adata.n_obs} cells × {adata.n_vars} genes")

    if adata.n_obs > n_cells:
        rng = np.random.RandomState(random_seed)
        idx = rng.choice(adata.n_obs, size=n_cells, replace=False)
        adata = adata[idx].copy()
        logger.info(f"  Subsampled to: {adata.n_obs} cells")

    return adata


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CellRank 2 test pipeline on a downsampled dataset"
    )
    parser.add_argument(
        "--input",
        default="rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/combined/"
                "pipeline_test/scvi_output/integrated.h5ad",
        help="Path to integrated.h5ad (real data). Falls back to synthetic if missing.",
    )
    parser.add_argument(
        "--n_cells", type=int, default=2000,
        help="Number of cells to subsample (default: 2000).",
    )
    parser.add_argument(
        "--output_dir",
        default=str(
            Path(__file__).resolve().parent / "test_results"
        ),
        help="Output directory for test results (default: code/CellRank2/test_results/).",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Force use of synthetic data even if real data is available.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    args = parser.parse_args()

    logger = setup_logger("cellrank_test")
    logger.info("=" * 60)
    logger.info("CellRank 2 — Test Pipeline")
    logger.info("=" * 60)

    # Load or generate data
    real_path = Path(args.input)
    if not args.synthetic and real_path.exists():
        adata = load_and_downsample(
            str(real_path), args.n_cells, args.seed, logger
        )
        using_synthetic = False
    else:
        if not args.synthetic:
            logger.warning(
                f"Real data not found at '{args.input}'; "
                "generating synthetic data for testing."
            )
        else:
            logger.info("Generating synthetic data (--synthetic flag set).")
        adata = make_synthetic_adata(n_cells=args.n_cells, random_seed=args.seed)
        using_synthetic = True

    logger.info(f"  Using {'synthetic' if using_synthetic else 'real'} data: "
                f"{adata.n_obs} cells × {adata.n_vars} genes")

    # Save the (down)sampled input so the pipeline can load it
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_input = out_dir / "test_input.h5ad"
    adata.write_h5ad(str(tmp_input))
    logger.info(f"  Test input saved: {tmp_input}")

    # Build config: skip neighbors if already computed (synthetic path computes them)
    steps = ["ot", "kernels", "gpcca", "fate_probs", "save"]
    if "neighbors_scanvi" not in adata.uns:
        steps = ["neighbors"] + steps

    config = CellRankConfig(
        input_h5ad=str(tmp_input),
        output_dir=str(out_dir),
        latent_key="X_scANVI",
        time_key="age_years",
        cell_type_key="cell_type_aligned",
        batch_key="source",
        n_neighbors=15,
        neighbors_key="neighbors_scanvi",
        age_bin_edges=[0.0, 0.5, 2.0, 10.0, 20.0, 40.0, 100.0],
        ot_epsilon=0.05,
        ot_max_iterations=500,
        rtk_weight=0.8,
        n_macrostates=5,   # 5 states to match the 5 distinct cell types in synthetic data
        cluster_key="cell_type_aligned",
        terminal_states=[],
        initial_states=[],
        compute_drivers=False,
        lineage_targets=[],
        save_plots=True,
        plot_color_vars=["cell_type_aligned", "age_years", "source"],
        umap_key="X_umap_scanvi",
        point_size=3.0,
        steps=steps,
        random_seed=args.seed,
        overwrite=True,
    )

    logger.info(f"Pipeline steps: {config.steps}")

    adata_out = run(config)

    logger.info("=" * 60)
    logger.info("Test pipeline finished successfully.")
    logger.info(f"Results written to: {out_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
