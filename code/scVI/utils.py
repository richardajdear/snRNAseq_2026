"""Logging, timing, device detection, and memory utilities."""

import logging
import os
import time
import warnings
from typing import Optional


def setup_logger(
    name: str = "scvi_pipeline", log_file: Optional[str] = None
) -> logging.Logger:
    """Configure logger with timestamps and optional file output.

    Also captures Python warnings and logs them.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file:
        from pathlib import Path

        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="w")  # overwrite per run
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    # Capture Python warnings and log them
    logging.captureWarnings(True)
    warnings_logger = logging.getLogger("py.warnings")
    warnings_logger.setLevel(logging.WARNING)
    for handler in logger.handlers:
        warnings_logger.addHandler(handler)

    return logger


class Timer:
    """Context manager for timing pipeline steps."""

    def __init__(self, label: str, logger: logging.Logger):
        self.label = label
        self.logger = logger

    def __enter__(self):
        self.start = time.time()
        self.logger.info(f"Starting: {self.label}")
        return self

    def __exit__(self, *args):
        elapsed = time.time() - self.start
        if elapsed < 1.0:
            duration = f"{elapsed * 1000:.0f}ms"
        else:
            h, rem = divmod(elapsed, 3600)
            m, s = divmod(rem, 60)
            duration = f"{int(h)}h {int(m)}m {s:.1f}s"
        self.logger.info(f"Completed: {self.label} [{duration}]")


def get_device_info(logger: logging.Logger) -> dict:
    """Detect GPU, log info, return device metadata."""
    import torch

    info = {"device": "cpu", "gpu_name": None, "vram_bytes": 0}

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["device"] = "cuda"
        info["gpu_name"] = props.name
        info["vram_bytes"] = props.total_mem
        torch.set_float32_matmul_precision("high")
        logger.info(
            f"GPU detected: {props.name} ({props.total_mem / 1e9:.1f} GB VRAM)"
        )
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info["device"] = "mps"
        info["gpu_name"] = "Apple Silicon MPS"
        info["vram_bytes"] = 0  # MPS shares system RAM
        logger.info("Apple MPS backend detected (shared memory with system RAM)")
    else:
        logger.info("No GPU detected. Using CPU.")

    return info


def estimate_chunk_size(
    n_genes: int,
    n_mc_samples: int,
    vram_bytes: int,
    target_fraction: float = 0.25,
    max_chunk: int = 50000,
) -> int:
    """Estimate how many cells fit in one inference chunk given VRAM."""
    bytes_per_cell = n_mc_samples * n_genes * 4  # float32
    if bytes_per_cell == 0:
        return max_chunk
    # Default to 20GB target if VRAM unknown (CPU or MPS)
    target_bytes = (
        vram_bytes * target_fraction if vram_bytes > 0 else 20 * (1024**3)
    )
    chunk = int(target_bytes / bytes_per_cell)
    return max(1, min(chunk, max_chunk))


def log_memory(label: str, logger: logging.Logger):
    """Log current process memory usage (requires psutil)."""
    try:
        import psutil

        mem_gb = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
        logger.info(f"[Memory] {label}: {mem_gb:.2f} GB RSS")
    except ImportError:
        pass
