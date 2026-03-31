"""
Top-level orchestration script for the full snRNAseq pipeline.

Steps (in order):
    1. downsample      — per-dataset: read + filter → individual h5ads
    2. combine         — concatenate individual h5ads → combined.h5ad
    3. scvi            — batch correction via scVI (delegates to scVI/run_pipeline.py)
    4. label_transfer  — kNN transfer of cell_type labels → cell_type_aligned
    5. scanvi_aligned  — (optional) re-run scANVI with cell_type_aligned labels

Usage (from project root):
    PYTHONPATH=code python -m run_pipeline --config code/pipeline_config.yaml
    PYTHONPATH=code python -m run_pipeline --config code/pipeline_config.yaml \\
        --steps downsample combine scvi
    PYTHONPATH=code python -m run_pipeline --config code/pipeline_config.yaml \\
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


def _run(cmd: list, logger: logging.Logger):
    """Run a subprocess command, streaming output and raising on failure."""
    logger.info(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        logger.error(f"Command failed (exit {result.returncode})")
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
    """Run scVI batch correction pipeline."""
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
    return integrated_path


def step_label_transfer(cfg: dict, output_dir: Path, integrated_path: Path,
                        overwrite: bool, logger: logging.Logger) -> Path:
    """Run kNN label transfer to produce cell_type_aligned."""
    logger.info("=" * 60)
    logger.info("STEP 4: LABEL TRANSFER")
    logger.info("=" * 60)

    lt_cfg = cfg.get('label_transfer', {})
    lt_output_dir = output_dir / 'label_transfer'
    all_labels_path = lt_output_dir / 'all_cell_labels.csv'

    if all_labels_path.exists() and not overwrite:
        logger.info(f"  Label transfer output already exists, skipping ({lt_output_dir})")
        return lt_output_dir

    cmd = [
        sys.executable, '-m', 'pipeline.label_transfer.run_transfer',
        '--input',      str(integrated_path),
        '--output_dir', str(lt_output_dir),
    ]
    if lt_cfg.get('reference_source'):
        cmd += ['--reference_source', lt_cfg['reference_source']]
    if lt_cfg.get('target_sources'):
        cmd += ['--target_sources'] + lt_cfg['target_sources']
    if lt_cfg.get('k'):
        cmd += ['--k', str(lt_cfg['k'])]
    if lt_cfg.get('confidence_threshold'):
        cmd += ['--confidence_threshold', str(lt_cfg['confidence_threshold'])]
    if lt_cfg.get('embedding_key'):
        cmd += ['--embedding_key', lt_cfg['embedding_key']]
    if lt_cfg.get('umap_key'):
        cmd += ['--umap_key', lt_cfg['umap_key']]

    _run(cmd, logger)
    return lt_output_dir


def main():
    parser = argparse.ArgumentParser(
        description='Full snRNAseq pipeline: downsample → combine → scVI → label_transfer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--config', required=True,
                        help='Path to pipeline_config.yaml')
    parser.add_argument('--steps', nargs='+',
                        choices=['downsample', 'combine', 'scvi', 'label_transfer',
                                 'scanvi_aligned', 'all'],
                        default=['all'],
                        help='Which steps to run (default: all)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing outputs')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    steps = cfg.get('steps', ['downsample', 'combine', 'scvi', 'label_transfer'])
    if args.steps != ['all']:
        steps = args.steps  # CLI overrides config
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
    combined_path   = output_dir / 'combined.h5ad'
    integrated_path = output_dir / 'scvi_output' / 'integrated.h5ad'

    # Execute steps
    if 'downsample' in steps:
        per_dataset_paths = step_downsample(cfg, output_dir, overwrite, logger)

    if 'combine' in steps:
        combined_path = step_combine(cfg, output_dir, per_dataset_paths, overwrite, logger)

    if 'scvi' in steps:
        integrated_path = step_scvi(cfg, output_dir, combined_path, overwrite, logger)

    if 'label_transfer' in steps:
        step_label_transfer(cfg, output_dir, integrated_path, overwrite, logger)

    if 'scanvi_aligned' in steps:
        logger.info("=" * 60)
        logger.info("STEP 5: scANVI WITH ALIGNED LABELS")
        logger.info("Not yet implemented — run scVI/run_pipeline.py manually with "
                    "cell_type_key=cell_type_aligned after label transfer.")
        logger.info("=" * 60)

    logger.info("Pipeline complete.")


if __name__ == '__main__':
    main()
