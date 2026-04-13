"""Logging, timing, and memory utilities for the CellRank 2 pipeline.

Mirrors code/scVI/utils.py style so both pipelines behave consistently.
"""

import logging
import os
import time
import warnings
from typing import Optional


def setup_logger(
    name: str = "cellrank_pipeline", log_file: Optional[str] = None
) -> logging.Logger:
    """Configure logger with timestamps and optional file output."""
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
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

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


def log_memory(label: str, logger: logging.Logger):
    """Log current process memory usage (requires psutil)."""
    try:
        import psutil

        mem_gb = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
        logger.info(f"[Memory] {label}: {mem_gb:.2f} GB RSS")
    except ImportError:
        pass


def get_device_info(logger: logging.Logger) -> dict:
    """Detect GPU availability, log info, return device metadata."""
    try:
        import torch

        info: dict = {"device": "cpu", "gpu_name": None, "has_gpu": False}
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info = {
                "device": "cuda",
                "gpu_name": props.name,
                "vram_bytes": props.total_memory,
                "has_gpu": True,
            }
            logger.info(
                f"GPU detected: {props.name} "
                f"({props.total_memory / 1e9:.1f} GB VRAM)"
            )
        else:
            logger.info("No GPU detected. Using CPU.")
        return info
    except ImportError:
        logger.info("torch not available; device detection skipped.")
        return {"device": "cpu", "has_gpu": False}
