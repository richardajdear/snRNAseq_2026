"""
Cross-assignment diagnostic: Velmeshev native labels vs PsychAD-supervised scANVI predictions.

Reads VelPsychAD_psychad_labels_age5/scvi_output/integrated.h5ad and produces:

  1. Cross-tab matrix — Vel cell_type_raw (rows) × cell_type_aligned/PsychAD labels (cols)
     Shows what each Vel native type maps to under PsychAD supervision.
     Key question: do Vel "Interneurons" land on EN or IN PsychAD subtypes?

  2. Side-by-side composition — per age bin (<1y, 1-5y):
     Left col:  Vel cell_type_aligned (PsychAD label assigned by scANVI)
     Right col: PsychAD cell_type_raw  (native subclass — gold standard)
     Lets you directly compare what scANVI puts Vel cells into vs what PsychAD
     cells are natively labelled as, at the same age.

  3. EN / IN summary table across sources and age bins.

Usage:
    python -m pipeline.diagnostics.vel_psychad_cross_assignment \\
        --h5ad <path/to/integrated.h5ad> \\
        [--out_dir <output_directory>]

Output: text report printed to stdout + optionally written to out_dir/cross_assignment.txt
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


# ── Helpers ───────────────────────────────────────────────────────────────────

def decode_cats(f, col):
    cats = [x.decode() if isinstance(x, bytes) else x
            for x in f['obs'][col]['categories'][:]]
    codes = f['obs'][col]['codes'][:]
    return pd.Categorical.from_codes(codes, categories=cats)


def classify_label(label: str) -> str:
    """Map a label string to broad class (EN / IN / Glia / Other)."""
    s = str(label)
    if s.startswith('EN') or s.startswith('EN_') or s in {'L2-3', 'L4', 'L5', 'L6',
                                                           'L5-6-IT', 'SP', 'Progenitors',
                                                           'Interneurons'}:
        return 'EN-like'
    if (s.startswith('IN') or s.startswith('IN_')
            or s in {'SST', 'PV', 'VIP', 'CALB2', 'SST_RELN', 'RELN', 'NOS',
                     'SV2C', 'CCK', 'PV_MP', 'INT', 'Micro'}):
        return 'IN-like'
    if s in {'OPC', 'Oligo', 'Astro', 'Fibrous_astrocytes',
             'Protoplasmic_astrocytes', 'Glial_progenitors', 'Endo',
             'PC', 'VLMC', 'PVM', 'SMC', 'Adaptive', 'Microglia'}:
        return 'Glia'
    return 'Other'


def section(title: str, width: int = 78) -> str:
    return "\n" + "=" * width + f"\n{title}\n" + "=" * width


# ── Main ──────────────────────────────────────────────────────────────────────

def run(h5ad_path: str, out_dir: str | None = None):
    lines = []

    def p(s=""):
        lines.append(s)
        print(s)

    # ── Load obs metadata only ────────────────────────────────────────────────
    with h5py.File(h5ad_path, 'r') as f:
        source   = decode_cats(f, 'source')
        age      = f['obs']['age_years'][:]
        ctr      = decode_cats(f, 'cell_type_raw')
        cta      = decode_cats(f, 'cell_type_aligned')
        ctf      = decode_cats(f, 'cell_type_for_scanvi')
        conf     = f['obs']['cell_type_aligned_confidence'][:]

    obs = pd.DataFrame({
        'source':              pd.Categorical(source),
        'age_years':           age,
        'cell_type_raw':       pd.Categorical(ctr),
        'cell_type_for_scanvi':pd.Categorical(ctf),
        'cell_type_aligned':   pd.Categorical(cta),
        'confidence':          conf,
    })

    vel    = obs[obs['source'] == 'VELMESHEV'].copy()
    psych  = obs[obs['source'] == 'PSYCHAD'].copy()

    age_bins = [('<1y',  -np.inf, 1),
                ('1-5y',  1,      5)]

    # ── Header ────────────────────────────────────────────────────────────────
    p(section("VelPsychAD cross-assignment diagnostic"))
    p(f"H5AD: {h5ad_path}")
    p(f"Total cells: {len(obs):,}  |  Vel: {len(vel):,}  |  PsychAD: {len(psych):,}")
    p(f"Vel labeled (non-Unknown): {(vel['cell_type_for_scanvi'] != 'Unknown').sum():,}  "
      f"(should be 0 — all Vel are Unknown in this run)")
    p(f"PsychAD labeled: {(psych['cell_type_for_scanvi'] != 'Unknown').sum():,}  "
      f"(all PsychAD are supervised with native subclass)")

    # ── 1.  Cross-tab: Vel native × cell_type_aligned ─────────────────────────
    p(section("1. Vel cell_type_raw  ×  cell_type_aligned  (PsychAD-label scANVI)"))
    p("Rows = Vel original paper labels.  Cols = PsychAD subclass assigned by scANVI.")
    p("EN_ prefixed cols are excitatory, IN_ are inhibitory.  % = row fraction.")
    p()

    ct = pd.crosstab(vel['cell_type_raw'], vel['cell_type_aligned'])
    ct['TOTAL'] = ct.sum(axis=1)
    ct = ct[ct['TOTAL'] > 0].sort_values('TOTAL', ascending=False)

    # Add broad-class summary columns
    en_cols  = [c for c in ct.columns if c.startswith('EN')]
    in_cols  = [c for c in ct.columns if c.startswith('IN') or c == 'Micro']
    glia_cols = [c for c in ct.columns if c not in en_cols + in_cols + ['TOTAL']]

    ct['EN%']   = (ct[en_cols].sum(axis=1)   / ct['TOTAL'] * 100).round(1)
    ct['IN%']   = (ct[in_cols].sum(axis=1)   / ct['TOTAL'] * 100).round(1)
    ct['GLIA%'] = (ct[glia_cols].sum(axis=1) / ct['TOTAL'] * 100).round(1)

    # Print compact summary first (just totals + %)
    p("─── Compact (TOTAL + EN% / IN% / GLIA%) ───")
    summary = ct[['TOTAL', 'EN%', 'IN%', 'GLIA%']].copy()
    p(summary.to_string())

    # Print full cross-tab (sorted cols: EN first, then IN, then glia)
    ordered_cols = sorted(en_cols) + sorted(in_cols) + sorted(glia_cols) + ['TOTAL', 'EN%', 'IN%']
    ordered_cols = [c for c in ordered_cols if c in ct.columns]
    p()
    p("─── Full matrix ───")
    p(ct[ordered_cols].to_string())

    # ── 2.  Focus rows: Interneurons and INT ──────────────────────────────────
    p(section("2. Focus: Vel 'Interneurons' and 'INT'  →  PsychAD label"))
    for raw_lbl in ['Interneurons', 'INT', 'Glial_progenitors', 'Progenitors']:
        sub = vel[vel['cell_type_raw'] == raw_lbl]
        if len(sub) == 0:
            continue
        p(f"\n── {raw_lbl}  (n={len(sub):,}) ──")
        cts2  = sub['cell_type_aligned'].value_counts()
        pct2  = (cts2 / len(sub) * 100).round(1)
        conf2 = sub.groupby('cell_type_aligned')['confidence'].median().round(2)
        df2   = pd.concat([cts2.rename('n'), pct2.rename('%'), conf2.rename('med_conf')], axis=1)
        df2   = df2[df2['n'] > 0]
        p(df2.head(15).to_string())
        p(f"  → EN%={df2.loc[df2.index.str.startswith('EN'), 'n'].sum() / len(sub)*100:.1f}%  "
          f"IN%={df2.loc[df2.index.str.startswith('IN'), 'n'].sum() / len(sub)*100:.1f}%")

    # ── 3.  Side-by-side composition by age bin ────────────────────────────────
    p(section("3. Side-by-side cell composition by age bin"))
    p("Left = Vel cell_type_aligned (what scANVI calls Vel cells using PsychAD labels)")
    p("Right = PsychAD cell_type_raw (native subclass labels — ground truth)")
    p("Both sorted by % descending.  Directly comparable since same label vocabulary.")

    for bin_label, lo, hi in age_bins:
        vel_bin   = vel[(vel['age_years'] >= lo)   & (vel['age_years'] < hi)]
        psych_bin = psych[(psych['age_years'] >= lo) & (psych['age_years'] < hi)]
        p(f"\n── {bin_label}  |  Vel n={len(vel_bin):,}  |  PsychAD n={len(psych_bin):,} ──")

        v_cts = (vel_bin['cell_type_aligned'].value_counts() / len(vel_bin) * 100).round(1)
        p_cts = (psych_bin['cell_type_raw'].value_counts()   / len(psych_bin) * 100).round(1)

        # Align to shared index (union of labels)
        all_labels = sorted(set(v_cts.index) | set(p_cts.index))
        df_side = pd.DataFrame({
            'Vel %':    v_cts.reindex(all_labels).fillna(0.0),
            'PsychAD %':p_cts.reindex(all_labels).fillna(0.0),
        })
        df_side['class'] = df_side.index.map(classify_label)
        df_side = df_side.sort_values('PsychAD %', ascending=False)

        # Add separating blank rows between broad classes
        current_class = None
        for lbl, row in df_side.iterrows():
            if row['class'] != current_class:
                if current_class is not None:
                    p()
                current_class = row['class']
                p(f"  [{current_class}]")
            v_str = f"{row['Vel %']:5.1f}%" if row['Vel %'] > 0 else "   — "
            p_str = f"{row['PsychAD %']:5.1f}%" if row['PsychAD %'] > 0 else "   — "
            p(f"    {lbl:30s}  Vel: {v_str}   PsychAD: {p_str}")

    # ── 4.  EN / IN summary table ─────────────────────────────────────────────
    p(section("4. EN / IN / OPC+Glia summary by source × age bin"))
    rows = []
    for bin_label, lo, hi in age_bins:
        for src_name, src_df in [('VELMESHEV', vel), ('PSYCHAD', psych)]:
            sub = src_df[(src_df['age_years'] >= lo) & (src_df['age_years'] < hi)]
            if len(sub) == 0:
                continue
            # For Vel: use cell_type_aligned (PsychAD labels assigned by scANVI)
            # For PsychAD: use cell_type_raw (native labels)
            if src_name == 'VELMESHEV':
                labels = sub['cell_type_aligned'].astype(str)
            else:
                labels = sub['cell_type_raw'].astype(str)

            n = len(sub)
            en_n    = labels.str.startswith('EN').sum()
            in_n    = labels.str.startswith('IN').sum()
            opc_n   = (labels == 'OPC').sum()
            micro_n = labels.isin({'Micro', 'Microglia'}).sum()
            rows.append({
                'age': bin_label, 'source': src_name, 'n': n,
                'EN%':    round(en_n    / n * 100, 1),
                'IN%':    round(in_n    / n * 100, 1),
                'EN/IN':  round(en_n    / max(in_n, 1), 2),
                'OPC%':   round(opc_n   / n * 100, 1),
                'Micro%': round(micro_n / n * 100, 1),
            })

    df_summ = pd.DataFrame(rows).set_index(['age', 'source'])
    p()
    p("(Vel labels = PsychAD subclass assigned by scANVI;  PsychAD labels = native subclass)")
    p(df_summ.to_string())

    # ── 5.  Vel "Interneurons" deep dive ──────────────────────────────────────
    p(section("5. Vel 'Interneurons' top assignments — confidence breakdown"))
    interneurons = vel[vel['cell_type_raw'] == 'Interneurons']
    p(f"n={len(interneurons):,}  |  mean confidence: {interneurons['confidence'].mean():.3f}")
    p(f"Low-confidence (<0.5): {(interneurons['confidence'] < 0.5).sum():,} "
      f"({(interneurons['confidence'] < 0.5).mean()*100:.1f}%)")
    p()
    top = interneurons['cell_type_aligned'].value_counts().head(15)
    pct = (top / len(interneurons) * 100).round(1)
    conf_med = interneurons.groupby('cell_type_aligned')['confidence'].median().round(3)
    df_top = pd.concat([top.rename('n'), pct.rename('%'), conf_med.rename('med_conf')], axis=1)
    p(df_top.to_string())

    # ── Save ─────────────────────────────────────────────────────────────────
    if out_dir:
        out_path = Path(out_dir) / 'cross_assignment.txt'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(lines))
        print(f"\nReport written to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--h5ad', required=True,
                        help='Path to integrated.h5ad from psychad_labels run')
    parser.add_argument('--out_dir', default=None,
                        help='Directory to write cross_assignment.txt')
    args = parser.parse_args()
    run(args.h5ad, args.out_dir)


if __name__ == '__main__':
    main()
