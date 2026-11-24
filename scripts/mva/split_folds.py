#!/usr/bin/env python3
"""Split TRAIN shards into event-modulo-k held-out folds for b-hive.

One validation owner (2026-11-05). For model i the outer fold i (event % k == i)
is the held-out TEST/scoring fold: it never enters model i's training or
validation, and RUN scores fold i with model i only. The other k-1 folds (80% for
k=5) form model_i/train.txt. b-hive owns validation: its TrainingTask splits the
constructed files of that list internally (train_val_split, default 0.85), so
the effective fractions are about 68/12/20 by constructed-file fraction, not an
exact event count. No external validation fold is written.

No data or unlabelled process is added to a supervised filelist. The appended
`weight` column (weight_nominal*lumi*xsec/sumw) is provided for weighted
studies; b-hive config HPlusCHToWW_2dcats does not read it (class re-weighting
from unweighted counts), which split_manifest.json records.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from analysis.features.hww import event_folds, FEATURES

ROLES = ('train', 'test')
VALIDATION = {
    'owner': 'b-hive TrainingTask',
    'mechanism': 'shuffles the constructed LZ4 files of model_<i>/train.txt and keeps '
                 'int(n*train_val_split) for training, the rest for validation',
    'train_val_split_default': 0.85,
    'reference': 'b-hive tasks/training.py TrainingTask.train_val_split and run()',
    'approx_fractions_k5': {'train': 0.68, 'validation': 0.12, 'test': 0.20},
    'fraction_basis': 'constructed-file fraction inside the 80% training list; not an event count',
}


def split_files(inputs, output, k=5, scales=None):
    if scales is None:
        raise ValueError("Provide preselection sumw normalization from all TRAIN coffea outputs")
    if k < 3:
        # k=2 would give b-hive only half the events; keep the established minimum.
        raise ValueError('Use at least 3 folds')
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite split directory: {output}')
    tables, metadata = [], []
    for filename in sorted(set(map(str, inputs))):
        table = pq.read_table(filename)
        if any(c not in table.column_names for c in FEATURES + ['event', 'fold', 'label', 'weight_nominal']):
            raise ValueError(f'{filename}: not a complete TRAIN shard')
        md = table.schema.metadata or {}
        if md.get(b'processing_mode') != b'train':
            raise ValueError(f'{filename}: not TRAIN-mode output')
        metadata.append((md.get(b'features_hash'), md.get(b'selection_hash'), md.get(b'folds')))
        if md.get(b'folds') != str(k).encode():
            raise ValueError('Fold count differs from TRAIN metadata')
        values = table['event'].to_numpy()
        if not np.array_equal(event_folds(values, k), table['fold'].to_numpy()):
            raise ValueError('Stored folds disagree with event IDs')
        if np.any(table['label'].to_numpy() >= 0):
            key = (md[b'year'].decode(), md[b'sample'].decode())
            if key not in scales:
                raise ValueError(f'Missing full-sample normalization for {key}')
            table = table.append_column('weight', pa.array(table['weight_nominal'].to_numpy() * scales[key]))
            tables.append((filename, table))
    if not metadata or len(set(metadata)) != 1 or any(x is None for x in metadata[0]):
        raise ValueError('TRAIN inputs have missing or inconsistent feature/selection metadata')
    output.mkdir(parents=True)
    lists = {i: {s: [] for s in ROLES} for i in range(k)}
    counts = {i: {s: 0 for s in ROLES} for i in range(k)}
    for index, (filename, table) in enumerate(tables):
        fold, label = table['fold'].to_numpy(), table['label'].to_numpy()
        # Store each event only once. Filelists, not data copies, define model i's roles.
        for f in range(k):
            mask = (fold == f) & (label >= 0)
            if not mask.any():
                continue
            dest = output / f'shard_{index:06d}_fold_{f}.parquet'
            pq.write_table(table.filter(pa.array(mask)), dest)
            for heldout in range(k):
                role = 'test' if f == heldout else 'train'
                lists[heldout][role].append(str(dest.resolve()))
                counts[heldout][role] += int(mask.sum())
    for f, roles in lists.items():
        if not roles['train'] or not roles['test']:
            raise ValueError(f'Model {f} has an empty train or test list')
        d = output / f'model_{f}'; d.mkdir()
        for role, paths in roles.items():
            # b-hive stops reading a filelist at the first empty line: no blank lines.
            (d / (role + '.txt')).write_text('\n'.join(paths) + '\n')
    info = {'schema_version': 2, 'folds': k, 'fold_field': 'event',
            'features_hash': metadata[0][0].decode(),
            'selection_hash': metadata[0][1].decode(),
            'inputs': [{'path': str(Path(p).resolve()), 'sha256': hashlib.sha256(Path(p).read_bytes()).hexdigest()}
                       for p in sorted(set(map(str, inputs)))],
            'validation': VALIDATION,
            'weight_column': {'name': 'weight',
                              'definition': 'weight_nominal*lumi*xsec/sumw(all TRAIN chunks, pre-selection)',
                              'consumed_by_bhive_2dcats': False,
                              'note': 'HPlusCHToWW_2dcats uses reweighting_mode=class from unweighted class counts'},
            'models': {i: {'test_fold': i, 'excluded_fold': i,
                           'train_folds': [f for f in range(k) if f != i],
                           'rows': counts[i],
                           'train_list': f'model_{i}/train.txt', 'test_list': f'model_{i}/test.txt'}
                       for i in range(k)}}
    info['filelist_sha256'] = hashlib.sha256(json.dumps(info['inputs'], sort_keys=True).encode()).hexdigest()
    for i in range(k):
        for role in ROLES:
            info['models'][i][role + '_list_sha256'] = hashlib.sha256(
                (output / f'model_{i}' / f'{role}.txt').read_bytes()).hexdigest()
    (output / 'split_manifest.json').write_text(json.dumps(info, indent=2) + '\n')
    return info


def check_label_order(cfg):
    groups = list(cfg.mva['labels']['process_groups'])
    if groups != list(cfg.combine['classes']):
        raise ValueError(f'mva.labels.process_groups order {groups} != combine.classes {cfg.combine["classes"]}')


def normalization(accumulators, workflow, exclusions=None, fileset_json=None):
    """(year, sample) -> lumi*xsec/sumw for MC; enforces dataset/file coverage.

    exclusions: {year: {sample: reason}} for deliberately absent MC samples.
    """
    import yaml
    from analysis.filesets.utils import get_dataset_config
    from analysis.processing_checks import sample_kind, check_dataset_coverage
    samples, chunks, files = {}, {}, {}
    for acc in accumulators:
        if acc['provenance']['mode'] != {'train'} or len(acc['provenance']['year']) != 1:
            raise ValueError('Supply TRAIN job accumulators')
        year = next(iter(acc['provenance']['year']))
        if 'files' not in acc.get('execution_checks', {}):
            raise ValueError('TRAIN accumulator lacks execution_checks file records')
        for sample, paths in acc['execution_checks']['files'].items():
            files.setdefault(year, {}).setdefault(sample, set()).update(paths)
        for sample, entry in acc['samples'].items():
            key = (year, sample)
            if key in chunks and chunks[key] & entry['chunks']:
                raise ValueError(f'Duplicate TRAIN chunks for {key}')
            chunks.setdefault(key, set()).update(entry['chunks'])
            samples[key] = samples.get(key, 0.) + entry['sumw']
    with open(Path(__file__).resolve().parents[2] / 'analysis/postprocess/luminosity.yaml') as f:
        lumi = yaml.safe_load(f)
    scales, present = {}, {}
    for (year, sample), sumw in samples.items():
        info = get_dataset_config(year)[sample]
        if sample_kind(sample, info) == 'data':
            continue  # data never enters supervised lists; its xsec is the string 'None'
        present.setdefault(year, set()).add(sample)
        xsec = float(info['xsec'])
        if not np.isfinite(sumw) or sumw == 0 or not xsec > 0:
            raise ValueError(f'Invalid normalization {year}/{sample}')
        scales[(year, sample)] = lumi[year] * xsec / sumw
    exclusions = exclusions or {}
    for year in sorted(present):
        check_dataset_coverage(workflow, year, present[year],
                               {s: v for s, v in files.get(year, {}).items() if s in present[year]},
                               exclusions.get(year, {}), fileset_json)
    return scales


def load_year_exclusions(path):
    import yaml
    if not path:
        return {}
    raw = yaml.safe_load(Path(path).read_text()) or {}
    for year, entries in raw.items():
        for sample, reason in (entries or {}).items():
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(f'exclusion {year}/{sample} needs a provenance reason')
    return {year: dict(entries or {}) for year, entries in raw.items()}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('-w', '--workflow', required=True, help='workflow whose datasets define completeness')
    p.add_argument('--inputs', nargs='+', required=True, help='TRAIN parquet shards (shell globs allowed)')
    p.add_argument('--output', required=True)
    p.add_argument('--coffea', nargs='+', required=True, help='all TRAIN job accumulators, including zero-selection chunks')
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--exclusions', help='YAML {year: {sample: reason}} of deliberately absent MC samples')
    a = p.parse_args()
    import glob
    from coffea.util import load
    from analysis.workflows.config import WorkflowConfigBuilder
    from analysis.features.hww import signature, selection_signature
    from analysis.processing_checks import check_train_rows
    files = [f for pattern in a.inputs for f in glob.glob(pattern, recursive=True)]
    if len(set(map(str, a.coffea))) != len(a.coffea):
        raise ValueError('Duplicate TRAIN coffea inputs')
    cfg = WorkflowConfigBuilder(a.workflow).build_workflow_config()
    check_label_order(cfg)
    exclusions = load_year_exclusions(a.exclusions)
    loaded = [load(f) for f in a.coffea]
    for acc in loaded:
        prov = acc['provenance']
        if prov.get('features_hash') != {signature(cfg)} or prov.get('selection_hash') != {selection_signature(cfg)}:
            raise ValueError('TRAIN accumulator features/selection differ from the current workflow')
    scales = normalization(loaded, a.workflow, exclusions)
    check_train_rows(loaded, files)
    info = split_files(files, a.output, a.folds, scales)
    info['workflow'] = a.workflow
    info['exclusions'] = exclusions
    (Path(a.output) / 'split_manifest.json').write_text(json.dumps(info, indent=2) + '\n')
    print(Path(a.output) / 'split_manifest.json')
