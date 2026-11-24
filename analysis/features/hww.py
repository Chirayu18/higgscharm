"""Identical HWW event features for TRAIN ntuples and inline RUN inference.

Expressions remain in the workflow, evaluated once by BaseProcessor. This module
owns scalarization, the one-hot definition, ordering and nonfinite policy.
"""
import hashlib
import json
from pathlib import Path
import awkward as ak
import numpy as np
from analysis.corrections.ctag2d_categories import (
    ONEHOT_COLUMNS, onehot_columns, X_HFVLF_EDGES, Y_BVC_EDGES,
)

FEATURES = [
    'dilepton_pt', 'lepton1_pt', 'lepton2_pt', 'cjet_cand_pt', 'met_pt',
    'mtl1', 'mtl2', 'dilepton_mass', 'delta_R_ll_l1', 'delta_R_ll_l2',
    'delta_R_ll_c', 'delta_phi_l1PlusMET_c', 'delta_phi_l1_MET', 'delta_phi_l2_MET',
] + ONEHOT_COLUMNS + ['nSV']


def scalar(array, dtype=np.float32):
    if isinstance(array, ak.Array):
        if array.ndim == 2:
            array = ak.firsts(array)
        array = ak.to_numpy(ak.fill_none(array, np.nan))
    out = np.asarray(array, dtype=dtype)
    if out.ndim != 1:
        raise ValueError(f'Expected one scalar per event, got {out.shape}')
    return out


def signature(config):
    expressions = {}
    for name in FEATURES + ['cjet_cand_cvsl_pnet', 'cjet_cand_cvsb_pnet']:
        if name not in ONEHOT_COLUMNS:
            if name not in config.histogram_config.axes:
                raise ValueError(f'Missing HWW feature expression: {name}')
            expressions[name] = config.histogram_config.axes[name].expression
    spec = dict(features=FEATURES, dtype='float32', nonfinite=0.,
                implementation=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                onehot_x=X_HFVLF_EDGES, onehot_y=Y_BVC_EDGES, expressions=expressions)
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def selection_signature(config):
    root = Path(__file__).resolve().parents[1]
    sources = {}
    for directory in ('corrections', 'selections', 'working_points', 'leptonmva'):
        for path in sorted((root / directory).glob('*.py')):
            sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for directory in ('data/muon_HWW', 'data/electron_HWW', 'data/tthMVA_2022-2023_retrained'):
        for path in sorted((root / directory).iterdir()):
            if path.is_file():
                sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    spec = dict(objects=config.object_selection, events=config.event_selection,
                corrections=config.corrections_config, implementation=sources,
                external_inputs=external_inputs(config)['selection'])
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


# F14 (2026-11-10). External files, grouped by what they can change.
# 'selection': can change which events are selected or their features, so they
# are part of selection_signature and a change invalidates trained models.
# 'weights': change only event weights, which b-hive 2dcats does not read
# (class weights only, Chirayu 2026-11-10); recorded as provenance, not hashed,
# so a pure weight/normalisation update does not force retraining.
MODEL_ERAS = ('2022preEE', '2022postEE', '2023preBPix', '2023postBPix')
SELECTION_CORRECTIONS = ('jerc', 'jetvetomaps', 'muon_ss', 'electron_ss', 'met_phi')
SELECTION_DATA_DIRS = ('data/jec', 'data/lumimask', 'data/muon_ss', 'data/electron_ss', 'data/lhe_yukawa')
WEIGHT_CORRECTIONS = ('pileup', 'muon', 'electron_id', 'electron_reco', 'ctagging',
                      'ctagging_2d', 'muon_HWW', 'electron_HWW')
WEIGHT_DATA_DIRS = ('data/kfactors',)
_HASH_CACHE = {}


def _sha(path):
    path = Path(path)
    st = path.stat()
    key = (str(path), st.st_mtime_ns, st.st_size)
    if key not in _HASH_CACHE:
        _HASH_CACHE[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    return _HASH_CACHE[key]


def _label(path, root):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _entries(entry):
    if isinstance(entry, str):
        return [entry]
    if isinstance(entry, dict):
        return [p for p in entry.values() if isinstance(p, str)]
    return []


def _files_of(names, dirs, root, strict=True):
    from analysis.corrections.correctionlib_files import correction_files
    out = {}
    for name in names:
        for era in MODEL_ERAS:
            for p in _entries(correction_files.get(name, {}).get(era)):
                try:
                    if Path(p).is_file():
                        out[f'{name}/{era}:{_label(p, root)}'] = _sha(p)
                except PermissionError:
                    # weight-only provenance may sit on group EOS that a worker
                    # container cannot read; selection inputs must be readable.
                    if strict:
                        raise
                    out[f'{name}/{era}:{_label(p, root)}'] = 'unreadable-here'
    for d in dirs:
        base = root / d
        if base.is_dir():
            for p in sorted(base.rglob('*')):
                if p.is_file() and not p.name.endswith(('.pyc', '.swp')) and '.bak' not in p.name:
                    out[_label(p, root)] = _sha(p)
    return out


def external_inputs(config):
    root = Path(__file__).resolve().parents[1]
    selection = _files_of(SELECTION_CORRECTIONS, SELECTION_DATA_DIRS, root)
    weights = _files_of(WEIGHT_CORRECTIONS, WEIGHT_DATA_DIRS, root, strict=False)
    # 2026-11-12: the negrw model only changes event weights (weight_negrw), which b-hive
    # 2dcats does not read, so it is recorded, not hashed: retraining negrw must not
    # invalidate TRAIN outputs or fold models (Chirayu asked for negrw retraining).
    negrw_cfg = getattr(config, 'negrw', None)
    negrw = negrw_cfg.get('model') if isinstance(negrw_cfg, dict) else None
    if negrw:
        if not Path(negrw).is_file():
            raise FileNotFoundError(f'negrw model not found: {negrw}')
        weights[f'negrw:{negrw}'] = _sha(negrw)
    return dict(selection=selection, weights=weights)


def compute(variables):
    missing = [k for k in FEATURES if k not in ONEHOT_COLUMNS and k not in variables]
    if missing:
        raise KeyError(f'Missing HWW event-model features: {missing}')
    derived = onehot_columns(scalar(variables['cjet_cand_cvsl_pnet']),
                             scalar(variables['cjet_cand_cvsb_pnet']))
    cols = {name: scalar(derived[name] if name in derived else variables[name]) for name in FEATURES}
    lengths = {len(v) for v in cols.values()}
    if len(lengths) != 1:
        raise ValueError('HWW feature columns have different lengths')
    # Same documented nonfinite policy as the existing b-hive inference path.
    return {k: np.where(np.isfinite(v), v, 0.).astype(np.float32) for k, v in cols.items()}


def matrix(columns):
    return np.column_stack([columns[k] for k in FEATURES]).astype(np.float32)


def event_folds(events, k=5):
    if not isinstance(k, int) or isinstance(k, bool) or k < 2:
        raise ValueError('At least two integer folds are required')
    a = np.asarray(ak.to_numpy(events) if isinstance(events, ak.Array) else events)
    if a.dtype.kind not in 'iu' or a.ndim != 1 or np.any(a < 0):
        raise ValueError('Fold assignment requires nonnegative integer event IDs')
    return (a.astype(np.uint64) % np.uint64(k)).astype(np.int16)
