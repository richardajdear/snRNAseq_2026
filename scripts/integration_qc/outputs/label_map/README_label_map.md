# shared_fine_labels.csv — usage

Columns:
- `shared_label`: the target vocabulary used to supervise scANVI across all three datasets.
- `broad_class`: broad cell class (Excitatory, Inhibitory, Astrocytes, Oligos, OPC, Microglia, Endothelial, Other).
- `vel_cell_type`: pipe-separated values matched against Velmeshev raw `cell_type`.
- `wang_type_updated`: pipe-separated values matched against Wang raw `Type-updated` (verify column name on HPC).
- `psychad_subclass`: pipe-separated values matched against PsychAD raw `subclass`.

Pipe (`|`) acts as OR. Empty string ⇒ no mapping for that dataset.

## How to consume in `code/pipeline/downsample.py`

Replace the block at lines ~288–306 that currently sets:

```python
adata.obs["cell_type_for_scanvi"] = np.where(
    adata.obs["dataset"] == "WANG",
    adata.obs["cell_type_raw"].astype(str),
    "Unknown",
)
```

with logic that consults this CSV per dataset, e.g.:

```python
mapping = pd.read_csv(SHARED_LABELS_CSV)

def _map_for(dataset, raw_label, col):
    for _, row in mapping.iterrows():
        if raw_label in str(row[col]).split("|"):
            return row["shared_label"]
    return "Unknown"

mask_v = adata.obs["dataset"] == "VELMESHEV"
mask_w = adata.obs["dataset"] == "WANG"
mask_p = adata.obs["dataset"] == "PSYCHAD"
adata.obs.loc[mask_v, "cell_type_for_scanvi"] = (
    adata.obs.loc[mask_v, "cell_type_raw"]
    .map(lambda x: _map_for("VELMESHEV", x, "vel_cell_type")))
... similar for WANG and PSYCHAD ...
```

## Coverage check

See `fine_label_coverage.csv`. Aim for ≥ 80% of cells in each dataset to map cleanly.
The unmapped_top10 column lists the most common unmapped labels — extend `MAPPING` in this script to absorb them.

## Why this addresses the divergence

Script 02 showed that the joint scANVI relabels PsychAD young-donor cells away from Excitatory entirely (10 donors with 0% Excitatory in joint vs 2–9% in PsychAD-only scANVI). The root cause is that the model is supervised only by Wang fine labels, and Wang's perinatal Excitatory phenotypes do not transfer cleanly to PsychAD young donors. Supervising scANVI with all three datasets' biological labels (via this CSV) anchors the latent space to each dataset's structure and should preserve PsychAD's Excitatory neurons.