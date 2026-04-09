"""
Top-level orchestration script for the full snRNAseq pipeline.

Steps (in order):
    1. downsample  — per-dataset: read + filter → individual h5ads
    2. combine     — concatenate individual h5ads → combined.h5ad
    3. scvi        — batch correction (scVI) + label transfer (scANVI) via scVI/run_pipeline.py
                     When scanvi_label_transfer.enabled=true in config, scANVI is trained with
                     WANG's fine-grained labels and model.predict() assigns cell_type_aligned
                     to all cells. Diagnostics are written to scanvi_diagnostics/.
    4. scanvi      — scANVI-only rerun using existing scVI model (no scVI retraining)

Usage (from project root):
    PYTHONPATH=code python -m pipeline.run_pipeline --config code/pipeline/default_config.yaml
    PYTHONPATH=code python -m pipeline.run_pipeline --config code/pipeline/default_config.yaml \\
        --steps downsample combine scvi
    PYTHONPATH=code python -m pipeline.run_pipeline --config code/pipeline/default_config.yaml \
        --steps scanvi
    PYTHONPATH=code python -m pipeline.run_pipeline --config code/pipeline/default_config.yaml \\
        --overwrite
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import yaml


def _setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger('pipeline')
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s  %(levelname)s  %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_path, mode='w')
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _run(cmd: list, logger: logging.Logger, required: bool = True):
    """Run a subprocess command, optionally allowing non-zero exits."""
    logger.info(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        level = logger.error if required else logger.warning
        level(f"Command failed (exit {result.returncode})")
        if required:
            sys.exit(result.returncode)


def step_downsample(cfg: dict, output_dir: Path, overwrite: bool,
                    logger: logging.Logger):
    """Run downsample.py for each source dataset."""
    logger.info("=" * 60)
    logger.info("STEP 1: DOWNSAMPLE")
    logger.info("=" * 60)

    per_dataset_dir = output_dir / 'per_dataset'
    per_dataset_dir.mkdir(parents=True, exist_ok=True)

    output_paths = []
    for src in cfg['sources']:
        name = src['name']
        out_path = per_dataset_dir / f"{name}.h5ad"
        output_paths.append(out_path)

        if out_path.exists() and not overwrite:
            logger.info(f"  {name}: already exists, skipping ({out_path})")
            continue

        logger.info(f"  Processing {name} ...")
        cmd = [
            sys.executable, '-m', 'pipeline.downsample',
            '--input',        str(src['path']),
            '--output',       str(out_path),
            '--dataset_type', src['dataset_type'],
        ]
        if src.get('cell_type_field'):
            cmd += ['--cell_type_field', src['cell_type_field']]
        if src.get('pfc_only', False):
            cmd += ['--pfc_only']
        if src.get('n_cells'):
            cmd += ['--n_cells', str(src['n_cells'])]
        if cfg.get('age_downsample', False):
            cmd += ['--age_downsample']
        if cfg.get('seed'):
            cmd += ['--seed', str(cfg['seed'])]

        _run(cmd, logger)

    return output_paths


def step_combine(cfg: dict, output_dir: Path, input_paths: list,
                 overwrite: bool, logger: logging.Logger) -> Path:
    """Combine per-dataset h5ads into one file."""
    logger.info("=" * 60)
    logger.info("STEP 2: COMBINE")
    logger.info("=" * 60)

    combined_path = output_dir / 'combined.h5ad'
    if combined_path.exists() and not overwrite:
        logger.info(f"  Combined file already exists, skipping ({combined_path})")
        return combined_path

    cmd = [
        sys.executable, '-m', 'pipeline.combine_data',
        '--output', str(combined_path),
    ] + [str(p) for p in input_paths]

    _run(cmd, logger)
    return combined_path


def step_scvi(cfg: dict, output_dir: Path, combined_path: Path,
              overwrite: bool, logger: logging.Logger) -> Path:
    """Run scVI batch correction + optional scANVI label transfer pipeline."""
    logger.info("=" * 60)
    logger.info("STEP 3: scVI")
    logger.info("=" * 60)

    scvi_output_dir = output_dir / 'scvi_output'
    integrated_path = scvi_output_dir / 'integrated.h5ad'

    if integrated_path.exists() and not overwrite:
        logger.info(f"  scVI output already exists, skipping ({integrated_path})")
        return integrated_path

    # Build a temporary scVI config that points to combined_path
    scvi_cfg = {
        'input_h5ad': str(combined_path),
        'output_dir': str(scvi_output_dir),
        'batch_key': 'source',
        'cell_type_key': 'cell_class',
        'counts_layer': 'counts',
    }
    scvi_cfg.update(cfg.get('scvi', {}))

    # scANVI label transfer: use fine-grained WANG labels (cell_type_for_scanvi)
    # instead of broad cell_class, and run model.predict() for label transfer
    slt = cfg.get('scanvi_label_transfer', {})
    if slt.get('enabled', False):
        scvi_cfg['cell_type_key'] = slt.get('label_column', 'cell_type_for_scanvi')
        scvi_cfg['run_scanvi'] = True
        scvi_cfg['predict_cell_types'] = True
        if 'max_epochs_scanvi' in slt:
            scvi_cfg['max_epochs_scanvi'] = slt['max_epochs_scanvi']
        # scANVI is the primary inference output: it conditions expression on both
        # batch AND cell type, giving better cell-type-aware batch correction.
        # scVI inference is redundant when scANVI is available.
        scvi_cfg['run_scanvi_inference'] = True
        scvi_cfg['run_scvi_inference'] = False
        # Default transform_batch to WANG (reference dataset) so all cells are
        # normalized as if WANG cells of their type — ideal for cross-dataset GRN scoring.
        scvi_cfg.setdefault('transform_batch', 'WANG')
        # Ensure train_scanvi is included so scANVI is trained end-to-end
        scvi_cfg['steps'] = ['prep', 'train_scvi', 'train_scanvi', 'infer', 'umap', 'plot', 'save']
        logger.info(
            f"  scANVI label transfer enabled: "
            f"cell_type_key={scvi_cfg['cell_type_key']}, "
            f"max_epochs_scanvi={scvi_cfg.get('max_epochs_scanvi', 20)}, "
            f"transform_batch={scvi_cfg['transform_batch']}"
        )

    scvi_config_path = output_dir / 'scvi_config.yaml'
    with open(scvi_config_path, 'w') as f:
        yaml.dump(scvi_cfg, f, default_flow_style=False)
    logger.info(f"  scVI config written to {scvi_config_path}")

    cmd = [
        sys.executable, '-m', 'scVI.run_pipeline',
        '--config', str(scvi_config_path),
    ]
    if overwrite:
        cmd += ['--overwrite_scvi', 'true']

    _run(cmd, logger)

    # Verify scVI.run_pipeline actually produced its output.  If it exited 0
    # but produced nothing (e.g. a crash before writing any files), fail loudly.
    if not integrated_path.exists():
        logger.error(
            f"  scVI.run_pipeline returned 0 but integrated.h5ad was not created: "
            f"{integrated_path}"
        )
        sys.exit(1)

    # Diagnostics for scANVI label transfer
    if slt.get('enabled', False) and integrated_path.exists():
        diag_dir = output_dir / 'scanvi_diagnostics'
        logger.info(f"  Running scANVI diagnostics → {diag_dir}")
        diag_cmd = [
            sys.executable, '-m', 'pipeline.scanvi_diagnostics',
            '--input', str(integrated_path),
            '--output_dir', str(diag_dir),
            '--confidence_threshold', str(slt.get('confidence_threshold', 0.5)),
        ]
        _run(diag_cmd, logger, required=False)

    return integrated_path


def step_scanvi(cfg: dict, output_dir: Path, combined_path: Path,
                overwrite: bool, logger: logging.Logger) -> Path:
    """Run scANVI-only label transfer using an existing scVI model."""
    logger.info("=" * 60)
    logger.info("STEP 4: scANVI (SCANVI-ONLY)")
    logger.info("=" * 60)

    scvi_output_dir = output_dir / 'scvi_output'
    integrated_path = scvi_output_dir / 'integrated.h5ad'
    scvi_model_dir = scvi_output_dir / 'scvi_model'

    if not combined_path.exists():
        logger.error(
            f"Combined input missing: {combined_path}. "
            f"Run --steps combine (or downsample+combine) first."
        )
        sys.exit(1)
    if not scvi_model_dir.exists():
        logger.error(
            f"scVI model missing: {scvi_model_dir}. "
            f"Run --steps scvi first to train/load the base scVI model."
        )
        sys.exit(1)

    if integrated_path.exists() and not overwrite:
        logger.info(f"  integrated.h5ad already exists, skipping ({integrated_path})")
        logger.info("  Use --overwrite to force a fresh scANVI-only rerun.")
        return integrated_path

    scvi_cfg = {
        'input_h5ad': str(combined_path),
        'output_dir': str(scvi_output_dir),
        'batch_key': 'source',
        'cell_type_key': 'cell_class',
        'counts_layer': 'counts',
    }
    scvi_cfg.update(cfg.get('scvi', {}))
    # Critical flags set after user config so they cannot be accidentally overridden
    scvi_cfg['steps'] = ['prep', 'train_scanvi', 'infer', 'umap', 'plot', 'save']
    scvi_cfg['run_scanvi'] = True
    scvi_cfg['run_scanvi_inference'] = True
    scvi_cfg['run_scvi_inference'] = False
    scvi_cfg['predict_cell_types'] = True

    slt = cfg.get('scanvi_label_transfer', {})
    if slt.get('enabled', True):
        scvi_cfg['cell_type_key'] = slt.get('label_column', 'cell_type_for_scanvi')
        if 'max_epochs_scanvi' in slt:
            scvi_cfg['max_epochs_scanvi'] = slt['max_epochs_scanvi']
    else:
        logger.warning(
            "scanvi_label_transfer.enabled=false in config, "
            "but scanvi step was requested. Forcing scANVI on for this run."
        )

    scvi_config_path = output_dir / 'scvi_config_scanvi_only.yaml'
    with open(scvi_config_path, 'w') as f:
        yaml.dump(scvi_cfg, f, default_flow_style=False)
    logger.info(f"  scANVI-only config written to {scvi_config_path}")

    cmd = [
        sys.executable, '-m', 'scVI.run_pipeline',
        '--config', str(scvi_config_path),
    ]
    if overwrite:
        cmd += ['--overwrite_scanvi', 'true']

    _run(cmd, logger)

    # Diagnostics for scANVI label transfer
    if integrated_path.exists():
        diag_dir = output_dir / 'scanvi_diagnostics'
        logger.info(f"  Running scANVI diagnostics → {diag_dir}")
        diag_cmd = [
            sys.executable, '-m', 'pipeline.scanvi_diagnostics',
            '--input', str(integrated_path),
            '--output_dir', str(diag_dir),
            '--confidence_threshold', str(slt.get('confidence_threshold', 0.5)),
        ]
        _run(diag_cmd, logger, required=False)

    return integrated_path


def step_pseudobulk(cfg: dict, output_dir: Path, overwrite: bool,
                    logger: logging.Logger) -> None:
    """Aggregate integrated.h5ad per donor (× cell type) → pseudobulk_output/."""
    logger.info("=" * 60)
    logger.info("STEP: PSEUDOBULK")
    logger.info("=" * 60)

    integrated_path = output_dir / 'scvi_output' / 'integrated.h5ad'
    if not integrated_path.exists():
        logger.error(
            f"integrated.h5ad not found: {integrated_path}. "
            "Run the scvi or scanvi step first."
        )
        sys.exit(1)

    pb_cfg = cfg.get('pseudobulk', {})
    pb_output_dir = pb_cfg.get('output_dir') or str(output_dir / 'pseudobulk_output')

    cmd = [
        sys.executable, '-m', 'pipeline.pseudobulk',
        '--input',  str(integrated_path),
        '--output', pb_output_dir,
        '--config', str(output_dir / 'pipeline_config.yaml'),
    ]
    if overwrite:
        cmd += ['--overwrite']

    _run(cmd, logger)


def step_diagnostics(cfg: dict, output_dir: Path, logger: logging.Logger) -> None:
    """Re-run scANVI diagnostics on an existing integrated.h5ad (no model re-run needed)."""
    logger.info("=" * 60)
    logger.info("STEP: DIAGNOSTICS")
    logger.info("=" * 60)

    integrated_path = output_dir / 'scvi_output' / 'integrated.h5ad'
    if not integrated_path.exists():
        logger.error(
            f"integrated.h5ad not found: {integrated_path}. "
            "Run the scvi or scanvi step first."
        )
        sys.exit(1)

    slt = cfg.get('scanvi_label_transfer', {})
    diag_dir = output_dir / 'scanvi_diagnostics'
    logger.info(f"  Running scANVI diagnostics → {diag_dir}")
    diag_cmd = [
        sys.executable, '-m', 'pipeline.scanvi_diagnostics',
        '--input', str(integrated_path),
        '--output_dir', str(diag_dir),
        '--confidence_threshold', str(slt.get('confidence_threshold', 0.5)),
    ]
    _run(diag_cmd, logger, required=False)


def main():
    parser = argparse.ArgumentParser(
        description='Full snRNAseq pipeline: downsample → combine → scVI (+ scANVI label transfer)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--config', required=True,
                        help='Path to pipeline_config.yaml')
    parser.add_argument('--steps', nargs='+',
                        choices=['downsample', 'combine', 'scvi', 'scanvi', 'label_transfer',
                                 'diagnostics', 'pseudobulk', 'all'],
                        default=['all'],
                        help='Which steps to run (default: all)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing outputs')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    steps = cfg.get('steps', ['downsample', 'combine', 'scvi'])
    if args.steps != ['all']:
        steps = args.steps  # CLI overrides config

    # Backward-compat: legacy label_transfer step now maps to scANVI rerun step.
    steps = ['scanvi' if s == 'label_transfer' else s for s in steps]
    overwrite = args.overwrite or cfg.get('overwrite', False)

    output_dir = Path(cfg['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger(str(output_dir / 'pipeline.log'))

    logger.info("snRNAseq Pipeline")
    logger.info(f"Config:     {args.config}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Steps:      {steps}")
    logger.info(f"Overwrite:  {overwrite}")

    # Save copy of config alongside log
    import shutil
    shutil.copy(args.config, output_dir / 'pipeline_config.yaml')

    # Validate source paths
    for src in cfg.get('sources', []):
        if not Path(src['path']).exists():
            logger.error(f"Source path not found: {src['path']} (source: {src['name']})")
            sys.exit(1)

    # Track intermediate paths across steps
    per_dataset_paths = [
        output_dir / 'per_dataset' / f"{src['name']}.h5ad"
        for src in cfg['sources']
    ]
    combined_path = output_dir / 'combined.h5ad'

    # Execute steps
    if 'downsample' in steps:
        per_dataset_paths = step_downsample(cfg, output_dir, overwrite, logger)

    if 'combine' in steps:
        combined_path = step_combine(cfg, output_dir, per_dataset_paths, overwrite, logger)

    if 'scvi' in steps:
        step_scvi(cfg, output_dir, combined_path, overwrite, logger)

    if 'scanvi' in steps:
        step_scanvi(cfg, output_dir, combined_path, overwrite, logger)

    if 'diagnostics' in steps:
        step_diagnostics(cfg, output_dir, logger)

    if 'pseudobulk' in steps:
        step_pseudobulk(cfg, output_dir, overwrite, logger)

    logger.info("Pipeline complete.")


if __name__ == '__main__':
    main()
