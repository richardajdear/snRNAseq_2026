"""CellRank 2 pseudotime / lineage tracing pipeline for snRNA-seq data.

Recommended workflow:
  1. Build a ConnectivityKernel on the scANVI-corrected latent space (X_scANVI).
  2. Build a RealTimeKernel using donor chronological age (age_years) via moscot
     optimal transport between consecutive age groups.
  3. Combine the two kernels (RealTimeKernel for directionality,
     ConnectivityKernel for self-transitions within the same time point).
  4. Run GPCCA to identify macrostates, set terminal/initial states, and
     compute fate probabilities for each cell.
  5. Subset cells to a lineage of interest using the fate probability scores.
"""

from .config import CellRankConfig
from .run_pipeline import run

__all__ = ["CellRankConfig", "run"]
