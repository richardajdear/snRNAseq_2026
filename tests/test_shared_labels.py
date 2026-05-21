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


# ── Production CSV sanity check ─────────────────────────────────────────────
def test_production_csv_loads_and_resolves_all_datasets():
    """The shipped reference/shared_fine_labels.csv must validate AND resolve
    cleanly for all three dataset types (no ambiguous mappings)."""
    prod_csv = os.path.join(_REPO_ROOT, 'reference', 'shared_fine_labels.csv')
    if not os.path.exists(prod_csv):
        pytest.skip(f'production CSV not present: {prod_csv}')
    mapping = load_shared_label_map(prod_csv)
    a = _adata('cell_type_raw', ['L2-3'])  # any native label that exists
    for ds, native in [('Velmeshev', 'L2-3'),
                       ('Wang', 'EN-L2_3-IT'),
                       ('PsychAD', 'EN_L2_3_IT')]:
        a = _adata('cell_type_raw', [native])
        labels, _ = apply_shared_labels(a, ds, 'cell_type_raw', mapping)
        assert labels.iloc[0] == 'EN_L2_3_IT', f'{ds}: {native} → {labels.iloc[0]}'


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
