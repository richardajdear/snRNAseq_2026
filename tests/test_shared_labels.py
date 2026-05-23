"""Unit tests for code/pipeline/shared_labels.py.

Runs under pytest if available; otherwise execute directly:
    conda run -n scvi python tests/test_shared_labels.py
"""
import os
import sys
import textwrap

import numpy as np
import pandas as pd

import anndata as ad

try:
    import pytest  # type: ignore
except ImportError:  # pragma: no cover — provide a tiny shim
    class _PytestShim:
        class _Raises:
            def __init__(self, exc_type, match=None):
                self.exc_type = exc_type
                self.match = match

            def __enter__(self):
                return self

            def __exit__(self, et, ev, tb):
                if et is None:
                    raise AssertionError(
                        f'expected {self.exc_type.__name__} but no exception raised')
                if not issubclass(et, self.exc_type):
                    return False
                if self.match is not None:
                    import re
                    if not re.search(self.match, str(ev)):
                        raise AssertionError(
                            f'expected {self.exc_type.__name__} matching '
                            f'{self.match!r} but got {ev!r}')
                return True

        def raises(self, exc_type, match=None):
            return self._Raises(exc_type, match=match)

        def approx(self, value, rel=None, abs=None):
            tol = abs if abs is not None else 1e-6
            return _Approx(value, tol)

        def skip(self, reason):
            raise _Skip(reason)

    class _Approx:
        def __init__(self, value, tol):
            self.value, self.tol = value, tol

        def __eq__(self, other):
            return abs(other - self.value) <= self.tol

    class _Skip(Exception):
        pass

    pytest = _PytestShim()

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'code'))

from pipeline.shared_labels import (
    DATASET_COL,
    VALID_BROAD_CLASSES,
    apply_shared_labels,
    load_shared_label_map,
)


VALID_CSV = """\
shared_label,broad_class,vel_cell_type,wang_type_updated,psychad_subclass
EN_L2_3_IT,Excitatory,L2-3,EN-L2_3-IT,EN_L2_3_IT
IN_PV,Inhibitory,PV|PV_MP,IN-MGE-PV,IN_PVALB|IN_PVALB_CHC
Astro,Astrocytes,Fibrous_astrocytes|Protoplasmic_astrocytes,Astrocyte,Astro
Oligo,Oligos,Oligos,Oligodendrocyte,Oligo
"""


def _write(tmp_path, body):
    p = tmp_path / 'mapping.csv'
    p.write_text(textwrap.dedent(body))
    return str(p)


def _adata(fine_label_col, values):
    return ad.AnnData(
        X=np.zeros((len(values), 3), dtype=np.float32),
        obs=pd.DataFrame({fine_label_col: values},
                         index=[f'c{i}' for i in range(len(values))]),
    )


# ── load_shared_label_map ───────────────────────────────────────────────────
def test_load_valid_csv(tmp_path):
    df = load_shared_label_map(_write(tmp_path, VALID_CSV))
    assert list(df.columns) == ['shared_label', 'broad_class',
                                'vel_cell_type', 'wang_type_updated',
                                'psychad_subclass']
    assert len(df) == 4
    assert set(df['broad_class']).issubset(VALID_BROAD_CLASSES)


def test_load_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_shared_label_map('/no/such/file.csv')


def test_load_missing_column_raises(tmp_path):
    body = "shared_label,broad_class,vel_cell_type\nA,Excitatory,X\n"
    with pytest.raises(ValueError, match='missing required columns'):
        load_shared_label_map(_write(tmp_path, body))


def test_load_duplicate_shared_label_raises(tmp_path):
    body = VALID_CSV + "EN_L2_3_IT,Excitatory,L4,EN-L4-IT,EN_L4\n"
    with pytest.raises(ValueError, match='duplicate shared_label'):
        load_shared_label_map(_write(tmp_path, body))


def test_load_bad_broad_class_raises(tmp_path):
    body = VALID_CSV + "Mystery,Aliens,X,Y,Z\n"
    with pytest.raises(ValueError, match='invalid broad_class'):
        load_shared_label_map(_write(tmp_path, body))


def test_load_whitespace_in_pipe_fragment_raises(tmp_path):
    body = (
        "shared_label,broad_class,vel_cell_type,wang_type_updated,psychad_subclass\n"
        "IN_PV,Inhibitory,PV| PV_MP,IN-MGE-PV,IN_PVALB\n"
    )
    with pytest.raises(ValueError, match='whitespace'):
        load_shared_label_map(_write(tmp_path, body))


# ── apply_shared_labels ─────────────────────────────────────────────────────
def test_apply_velmeshev_basic(tmp_path):
    mapping = load_shared_label_map(_write(tmp_path, VALID_CSV))
    a = _adata('cell_type_raw',
               ['L2-3', 'PV', 'PV_MP', 'Oligos', 'CALB2'])
    labels, summary = apply_shared_labels(
        a, 'Velmeshev', 'cell_type_raw', mapping)

    assert list(labels) == ['EN_L2_3_IT', 'IN_PV', 'IN_PV', 'Oligo', 'Unknown']
    assert summary['n_cells'] == 5
    assert summary['n_mapped'] == 4
    assert summary['n_unmapped'] == 1
    assert summary['coverage_fraction'] == pytest.approx(0.8)
    assert summary['n_shared_labels'] == 3
    assert summary['unmapped_top10'] == {'CALB2': 1}


def test_apply_psychad_pipe_expansion(tmp_path):
    mapping = load_shared_label_map(_write(tmp_path, VALID_CSV))
    a = _adata('cell_type_raw',
               ['IN_PVALB', 'IN_PVALB_CHC', 'IN_PVALB', 'Astro'])
    labels, _ = apply_shared_labels(
        a, 'PsychAD', 'cell_type_raw', mapping)
    assert list(labels) == ['IN_PV', 'IN_PV', 'IN_PV', 'Astro']


def test_apply_wang_uses_right_column(tmp_path):
    mapping = load_shared_label_map(_write(tmp_path, VALID_CSV))
    # A Wang fine label that has no Velmeshev/PsychAD equivalent must still map
    a = _adata('cell_type_raw', ['EN-L2_3-IT', 'IN-MGE-PV', 'EN-L4-IT'])
    labels, summary = apply_shared_labels(
        a, 'Wang', 'cell_type_raw', mapping)
    # EN-L4-IT is not in the test CSV → Unknown
    assert list(labels) == ['EN_L2_3_IT', 'IN_PV', 'Unknown']
    assert summary['n_mapped'] == 2


def test_apply_unknown_dataset_raises(tmp_path):
    mapping = load_shared_label_map(_write(tmp_path, VALID_CSV))
    a = _adata('cell_type_raw', ['L2-3'])
    with pytest.raises(ValueError, match='dataset_type'):
        apply_shared_labels(a, 'Aliens', 'cell_type_raw', mapping)


def test_apply_missing_obs_column_raises(tmp_path):
    mapping = load_shared_label_map(_write(tmp_path, VALID_CSV))
    a = _adata('something_else', ['L2-3'])
    with pytest.raises(KeyError, match='cell_type_raw'):
        apply_shared_labels(a, 'Velmeshev', 'cell_type_raw', mapping)


def test_apply_preserves_obs_index(tmp_path):
    mapping = load_shared_label_map(_write(tmp_path, VALID_CSV))
    a = _adata('cell_type_raw', ['L2-3', 'Oligos'])
    labels, _ = apply_shared_labels(
        a, 'Velmeshev', 'cell_type_raw', mapping)
    assert list(labels.index) == list(a.obs.index)


def test_ambiguous_mapping_raises(tmp_path):
    """Two shared labels claim the same native label — must raise."""
    body = (
        "shared_label,broad_class,vel_cell_type,wang_type_updated,psychad_subclass\n"
        "EN_L5_ET,Excitatory,L5,EN-L5-ET,EN_L5_ET\n"
        "EN_L5_IT,Excitatory,L5,EN-L5-IT,EN_L5_IT\n"
    )
    mapping = load_shared_label_map(_write(tmp_path, body))
    a = _adata('cell_type_raw', ['L5'])
    with pytest.raises(ValueError, match='ambiguous mapping'):
        apply_shared_labels(a, 'Velmeshev', 'cell_type_raw', mapping)


def test_coverage_floor_raises_on_low_mapping(tmp_path):
    """When most cells don't map, apply_shared_labels must hard-fail (no
    silent fallback to near-empty supervision)."""
    mapping = load_shared_label_map(_write(tmp_path, VALID_CSV))
    # 10 cells: only 1 matches (L2-3); rest are unknown to the mapping
    a = _adata('cell_type_raw',
               ['L2-3'] + ['MysteryType'] * 9)
    with pytest.raises(ValueError, match='coverage'):
        apply_shared_labels(
            a, 'Velmeshev', 'cell_type_raw', mapping, min_coverage=0.5)


def test_coverage_floor_can_be_relaxed(tmp_path):
    """An advanced caller can opt into lower coverage explicitly."""
    mapping = load_shared_label_map(_write(tmp_path, VALID_CSV))
    a = _adata('cell_type_raw',
               ['L2-3'] + ['MysteryType'] * 9)
    labels, summary = apply_shared_labels(
        a, 'Velmeshev', 'cell_type_raw', mapping, min_coverage=0.0)
    assert summary['coverage_fraction'] == pytest.approx(0.1)
    assert (labels == 'Unknown').sum() == 9


def test_summary_includes_full_value_counts(tmp_path):
    mapping = load_shared_label_map(_write(tmp_path, VALID_CSV))
    a = _adata('cell_type_raw',
               ['L2-3', 'L2-3', 'PV', 'Oligos', 'Oligos', 'Oligos'])
    _, summary = apply_shared_labels(
        a, 'Velmeshev', 'cell_type_raw', mapping, min_coverage=0.0)
    assert summary['value_counts'] == {
        'Oligo': 3, 'EN_L2_3_IT': 2, 'IN_PV': 1,
    }


# ── Production CSV sanity check ─────────────────────────────────────────────
def test_production_csv_loads_and_resolves_all_datasets():
    """The shipped reference/shared_fine_labels.csv must validate AND resolve
    cleanly for all three dataset types (no ambiguous mappings).

    v3 changes:
    - EN subtypes collapsed: EN_L2_3_IT → EN_L2_3, EN_L4_IT → EN_L4,
      EN_L5_{ET,IT,NP} → EN_L5, EN_L6_{CT,IT,B} → EN_L6
    - Glial_progenitor row removed
    - IN_CALB2 Vel coverage stripped down to CALB2 only
    """
    prod_csv = os.path.join(_REPO_ROOT, 'reference', 'shared_fine_labels.csv')
    if not os.path.exists(prod_csv):
        pytest.skip(f'production CSV not present: {prod_csv}')
    mapping = load_shared_label_map(prod_csv)

    # Collapsed EN labels resolve from native inputs
    for ds, native, expected in [
        ('Velmeshev', 'L2-3',   'EN_L2_3'),
        ('Velmeshev', 'L4',     'EN_L4'),
        ('Velmeshev', 'L5',     'EN_L5'),
        ('Velmeshev', 'L5-6-IT', 'EN_L6'),
        ('Wang',      'EN-L2_3-IT', 'EN_L2_3'),
        ('Wang',      'EN-L5-ET',   'EN_L5'),
        ('Wang',      'EN-L6-CT',   'EN_L6'),
        ('PsychAD',   'EN_L2_3_IT', 'EN_L2_3'),
        ('PsychAD',   'EN_L5_ET',   'EN_L5'),
        ('PsychAD',   'EN_L6_CT',   'EN_L6'),
    ]:
        a = _adata('cell_type_raw', [native])
        labels, _ = apply_shared_labels(a, ds, 'cell_type_raw', mapping,
                                        min_coverage=0.0)
        assert labels.iloc[0] == expected, (
            f'{ds}: {native!r} → {labels.iloc[0]!r}, expected {expected!r}')

    # Velmeshev "Interneurons" must fall to Unknown (not IN_CALB2)
    a = _adata('cell_type_raw', ['Interneurons'])
    labels, _ = apply_shared_labels(a, 'Velmeshev', 'cell_type_raw', mapping,
                                    min_coverage=0.0)
    assert labels.iloc[0] == 'Unknown', (
        f'Velmeshev Interneurons → {labels.iloc[0]!r}, expected Unknown')

    # Glial_progenitors must fall to Unknown (row deleted)
    a = _adata('cell_type_raw', ['Glial_progenitors'])
    labels, _ = apply_shared_labels(a, 'Velmeshev', 'cell_type_raw', mapping,
                                    min_coverage=0.0)
    assert labels.iloc[0] == 'Unknown', (
        f'Velmeshev Glial_progenitors → {labels.iloc[0]!r}, expected Unknown')

    # No 'Glial_progenitor' shared_label in the CSV
    assert 'Glial_progenitor' not in mapping['shared_label'].values, (
        'Glial_progenitor row should have been deleted from the CSV')


# ── unlabel_below_age logic ─────────────────────────────────────────────────
def test_unlabel_below_age_overrides_shared_labels(tmp_path):
    """Simulate the Phase B withhold logic: after apply_shared_labels assigns
    a shared label, cells with age_years < threshold must be reset to Unknown."""
    mapping = load_shared_label_map(_write(tmp_path, VALID_CSV))

    # Mimic what downsample.py does: apply labels, then withhold young cells
    a = _adata('cell_type_raw', ['L2-3', 'PV', 'L2-3', 'PV', 'Oligos'])
    a.obs['age_years'] = [0.2, 0.5, 7.0, 15.0, 3.0]  # two young, two adult, one child

    labels, _ = apply_shared_labels(a, 'Velmeshev', 'cell_type_raw', mapping,
                                    min_coverage=0.0)
    a.obs['cell_type_for_scanvi'] = labels.values

    # Phase B: withhold labels for age < 5
    unlabel_age = 5.0
    young_mask = a.obs['age_years'] < unlabel_age
    a.obs.loc[young_mask, 'cell_type_for_scanvi'] = 'Unknown'

    result = a.obs['cell_type_for_scanvi'].tolist()
    # cells 0 (age 0.2), 1 (age 0.5), 4 (age 3.0) should be Unknown
    assert result[0] == 'Unknown', f'cell0 (age 0.2) should be Unknown, got {result[0]}'
    assert result[1] == 'Unknown', f'cell1 (age 0.5) should be Unknown, got {result[1]}'
    assert result[4] == 'Unknown', f'cell4 (age 3.0) should be Unknown, got {result[4]}'
    # cells 2 (age 7.0) and 3 (age 15.0) should keep their label
    assert result[2] == 'EN_L2_3_IT', f'cell2 (age 7.0) should be EN_L2_3_IT, got {result[2]}'
    assert result[3] == 'IN_PV',      f'cell3 (age 15.0) should be IN_PV, got {result[3]}'


def test_collapsed_en_labels_resolve_from_test_csv(tmp_path):
    """Collapsed EN subtypes must resolve if the test CSV uses new labels."""
    body = (
        "shared_label,broad_class,vel_cell_type,wang_type_updated,psychad_subclass\n"
        "EN_L2_3,Excitatory,L2-3,EN-L2_3-IT,EN_L2_3_IT\n"
        "EN_L5,Excitatory,L5,EN-L5-ET|EN-L5-IT,EN_L5_ET|EN_L3_5_IT_1\n"
        "IN_PV,Inhibitory,PV,IN-MGE-PV,IN_PVALB\n"
    )
    mapping = load_shared_label_map(_write(tmp_path, body))
    # Velmeshev L2-3 → EN_L2_3
    a = _adata('cell_type_raw', ['L2-3'])
    labels, _ = apply_shared_labels(a, 'Velmeshev', 'cell_type_raw', mapping)
    assert labels.iloc[0] == 'EN_L2_3'
    # Wang EN-L5-IT → EN_L5
    a = _adata('cell_type_raw', ['EN-L5-IT'])
    labels, _ = apply_shared_labels(a, 'Wang', 'cell_type_raw', mapping)
    assert labels.iloc[0] == 'EN_L5'
    # PsychAD EN_L3_5_IT_1 → EN_L5
    a = _adata('cell_type_raw', ['EN_L3_5_IT_1'])
    labels, _ = apply_shared_labels(a, 'PsychAD', 'cell_type_raw', mapping)
    assert labels.iloc[0] == 'EN_L5'


if __name__ == '__main__':
    import tempfile
    import traceback

    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith('test_') and callable(fn)]
    passed = failed = skipped = 0
    for name, fn in tests:
        params = fn.__code__.co_varnames[:fn.__code__.co_argcount]
        try:
            if 'tmp_path' in params:
                with tempfile.TemporaryDirectory() as d:
                    from pathlib import Path
                    fn(Path(d))
            else:
                fn()
        except Exception as e:
            if type(e).__name__ == '_Skip':
                print(f'  SKIP {name}: {e}')
                skipped += 1
            else:
                print(f'  FAIL {name}: {type(e).__name__}: {e}')
                traceback.print_exc()
                failed += 1
        else:
            print(f'  PASS {name}')
            passed += 1
    print(f'\n{passed} passed, {failed} failed, {skipped} skipped')
    sys.exit(0 if failed == 0 else 1)
