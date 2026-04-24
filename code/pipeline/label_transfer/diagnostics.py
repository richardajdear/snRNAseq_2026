"""Diagnostics for label transfer: public API module.

All diagnostic functions live here. The ``legacy/`` subdirectory contains
an older standalone entry-point (``legacy/diagnostics.py``) that is kept for
reference only; new code should import from this module.

Usage (import)
--------------
    from pipeline.label_transfer import diagnostics as diag
    diag.make_tables(tf_df, out_dir)
    diag.make_confidence_histogram(tf_df, out_dir)
    ...
"""

from pipeline.label_transfer.legacy.diagnostics import (  # noqa: F401
    # Private helpers accessed by scanvi_diagnostics.py
    _FALLBACK,
    _en_sort_key,
    _in_sort_key,
    # Palette helpers
    get_en_palette,
    get_in_palette,
    # Table outputs
    make_tables,
    make_class_remapping_tables,
    # Confidence histograms
    make_confidence_histogram,
    make_remapped_confidence_histogram,
    # Age-based plots
    make_age_histogram_remapped,
    make_age_confidence_density,
    # UMAP plots
    make_umap_all,
    make_umap_excitatory,
    make_umap_inhibitory,
    # Sankey diagram
    make_sankey,
    # Marker gene validation
    make_marker_validation,
    make_marker_scatter_validation,
)
