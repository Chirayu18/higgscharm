#!/usr/bin/env python3
"""Held-out scoring of complete nominal TRAIN outputs for Asimov iteration cards.

No object/weight shape variations are invented. Explicitly choose stat_only or
rate_systematics. Both keep configured rateParams and autoMCStats. The old TRAIN
schema needs --merge_channel_splits. Current negrw is rescored from generator
columns, never taken from the historical weight_negrw column.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import awkward as ak
import numpy as np
import pyarrow.parquet as pq
from coffea import processor
from coffea.util import load, save
from analysis.features.hww import FEATURES, event_folds, matrix, signature, selection_signature
from analysis.processing_checks import check_dataset_coverage, sample_kind, store_path
from analysis.filesets.utils import get_dataset_config
from analysis.models.registry import FoldModels
from analysis.processors.base import BaseProcessor, _dataset_matches, _routeb_provenance
from analysis.sinks.hww import make_templates, channel_layout, split_variables, final_channels, card_combine
from analysis.stitching.hplusc_routeb import RouteBConfig
from analysis.workflows.config import WorkflowConfigBuilder
from scripts.combine.build_cards_from_templates import (
    configure_era, make_process_hists, finalize_templates, write_root,
    write_data_obs, write_datacard, negrw_identity,
)


def canonical(path):
    # CERN EOS aliases identify the same file; avoid a remote stat per resolve().
    return str(Path(str(path).replace('/eos/home-c/cgupta/', '/eos/user/c/cgupta/')).absolute())


def unique_paths(paths, what):
    out = [canonical(p) for p in paths]
    if not out or len(out) != len(set(out)):
        raise ValueError(f'Empty or duplicate {what}')
    return out


def card_config(config, merge_splits, stat_only):
    cfg = deepcopy(config)
    if merge_splits:
        cfg.combine.pop('channel_splits', None)
    cfg.combine.pop('shape_only_systematics', None)
    cfg.combine.pop('alternative_samples', None)  # nominal-only iteration cards
    cfg.combine['shape_systematics'] = []
    cfg.combine['negrw_uncertainty'] = False
    if stat_only:
        cfg.combine['lnN'] = {}
    return cfg


def write_cards(output, config, year, dest, policy, routeb=None):
    """Write an independently configured card without rescoring the nominal rows."""
    dest = Path(dest)
    if dest.exists():
        raise FileExistsError(f'Refusing to overwrite {dest}')
    cfg = configure_era(config, year)
    policy = deepcopy(policy)
    hists, variations, object_shapes, edges = make_process_hists(output, cfg, year, policy=policy, routeb=routeb)
    finalize_templates(hists, cfg.combine, variations, object_shapes, policy)
    dest.mkdir(parents=True)
    root, card = dest/'templates.root', dest/'datacard.txt'
    channels = final_channels(cfg.combine)
    write_root(root, hists, channels, cfg.combine['classes'], [('nominal', '')], edges)
    write_data_obs(root, hists, channels, [c for c in cfg.combine['classes'] if c != cfg.combine['signal']], edges)
    policy['yields'] = {ch: {cp: float(hists[ch][cp]['nominal'][0].sum())
                            for cp in cfg.combine['classes']} for ch in channels}
    write_datacard(card, root.name, card_combine(cfg.combine), hists, edges, [])
    import uproot
    with uproot.open(root) as saved:
        for name in saved.keys(cycle=False):
            h = saved[name]
            if not np.all(np.isfinite(h.values())) or not np.all(np.isfinite(h.variances())):
                raise ValueError(f'Nonfinite written histogram {name}')
    (dest/'template_policy.json').write_text(json.dumps(policy, indent=2, sort_keys=True)+'\n')
    return card


def accumulate_train(manifest, year, cfg, models, datasets, feature_hash,
                     selection_hash, routeb_provenance=None, negrw_processor=None):
    """Validate all marker inputs and count real scored rows in one parquet read.

    Does not trust a subset merely because a readiness JSON says PASS. Coverage
    against current workflow/fileset is checked by the caller after this returns.
    """
    marker_paths = unique_paths(manifest['coffea'], 'TRAIN markers')
    all_expected = {canonical(x['marker']): x for x in manifest['expected_partitions']}
    if len(all_expected) != len(manifest['expected_partitions']) or set(marker_paths) != set(all_expected):
        raise ValueError('Marker list differs from expected partition inventory')
    if year not in {x['era'] for x in all_expected.values()}:
        raise ValueError(f'No expected partitions for {year}')
    all_parquets = unique_paths(manifest['parquets'], 'TRAIN parquet inputs')
    output = {'samples': {}, 'execution_checks': {'files': {}}}
    expected_rows, actual_rows = Counter(), Counter()
    chunks, source_files = {}, {}
    nmarkers = 0
    for path in marker_paths:
        spec = all_expected[path]
        if spec['era'] != year:
            continue
        acc = load(path)
        prov = acc.get('provenance', {})
        for key, want in [('mode', 'train'), ('year', year), ('features_hash', feature_hash),
                          ('selection_hash', selection_hash), ('folds', 5)]:
            if prov.get(key) != {want}:
                raise ValueError(f'{path}: invalid TRAIN {key}')
        if routeb_provenance is not None and prov.get('routeb_stitching') != {routeb_provenance}:
            raise ValueError(f'{path}: Route B provenance differs')
        ec = acc.get('execution_checks', {})
        if ('expected_entries' not in ec or ec['expected_entries'] != ec.get('processed_entries')):
            raise ValueError(f'{path}: incomplete entry counts')
        sample = spec['sample']
        if set(acc['samples']) != {sample} or set(ec.get('files', {})) != {sample}:
            raise ValueError(f'{path}: expected exactly sample {sample}')
        lfns = {store_path(x) for x in spec['lfns']}
        if len(lfns) != len(spec['lfns']) or set(ec['files'][sample]) != lfns:
            raise ValueError(f'{path}: partition input LFNs differ or repeat')
        if sample_kind(sample, datasets[sample]) != 'mc':
            raise ValueError(f'{path}: supplied inventory must be MC-only')
        entry = acc['samples'][sample]
        c = entry.get('chunks', set())
        if not c or entry.get('chunks_read') != len(c) or chunks.get(sample, set()) & c:
            raise ValueError(f'{path}: missing or duplicate TRAIN chunks')
        if source_files.get(sample, set()) & lfns:
            raise ValueError(f'{path}: file appears in multiple partitions')
        chunks.setdefault(sample, set()).update(c)
        source_files.setdefault(sample, set()).update(lfns)
        expected_rows[sample] += entry['selected_events']
        # Merge only the additive sample bookkeeping; no large legacy control objects.
        fields = {k: deepcopy(entry[k]) for k in
                  ('sumw', 'sumw2', 'events_read', 'chunks_read', 'chunks', 'selected_events', 'routeb_stitch')
                  if k in entry}
        old = output['samples'].get(sample)
        output['samples'][sample] = fields if old is None else processor.accumulate([old, fields])
        nmarkers += 1
    output['execution_checks']['files'] = source_files
    for entry in output['samples'].values():
        entry['evaluated_shifts'] = {}
    categories = list(cfg.event_selection['categories'])
    if len(categories) != 1:
        raise ValueError('TRAIN card path requires one stored event category')
    combine = cfg.combine
    splits = split_variables(combine)
    nparquets = 0
    scored_rows = 0
    for path in all_parquets:
        pf = pq.ParquetFile(path)
        md = pf.schema_arrow.metadata or {}
        if md.get(b'year', b'').decode() != year:
            continue
        for key, want in [('processing_mode', 'train'), ('features_hash', feature_hash),
                          ('selection_hash', selection_hash), ('folds', '5')]:
            if md.get(key.encode()) != want.encode():
                raise ValueError(f'{path}: invalid parquet {key}')
        if routeb_provenance is not None and md.get(b'routeb_stitching') != routeb_provenance.encode():
            raise ValueError(f'{path}: parquet Route B provenance differs')
        sample = md.get(b'sample', b'').decode()
        if sample not in output['samples']:
            raise ValueError(f'{path}: orphan parquet sample {sample}')
        if store_path(md.get(b'source_file', b'').decode()) not in source_files[sample]:
            raise ValueError(f'{path}: unprocessed source_file')
        if any(v not in pf.schema_arrow.names for v in splits):
            raise ValueError(f'{path}: missing channel split variable; explicitly use --merge_channel_splits')
        table = pf.read()
        cols = {k: table[k].to_numpy() for k in FEATURES}
        X = matrix(cols)
        if not np.all(np.isfinite(X)):
            raise ValueError(f'{path}: nonfinite stored MVA features')
        events = table['event'].to_numpy()
        if not np.array_equal(table['fold'].to_numpy(), event_folds(events, 5)):
            raise ValueError(f'{path}: stored fold disagrees with event modulo five')
        if set(table['sample'].to_pylist()) != {sample}:
            raise ValueError(f'{path}: sample column differs from metadata')
        if set(table['process'].to_pylist()) != {datasets[sample]['process']}:
            raise ValueError(f'{path}: process differs from current dataset configuration')
        weights = table['weight_nominal'].to_numpy().astype(np.float64)
        g = None
        negrw = getattr(cfg, 'negrw', None)
        if negrw and _dataset_matches(sample, negrw.get('datasets')):
            if negrw_processor is None:
                raise ValueError('Current negrw predictor is required')
            # Intentionally ignore historical weight_negrw, even if present.
            variables = {k: ak.from_arrow(table[k]) for k in table.column_names
                         if k.startswith(('gen', 'lhe_'))}
            g, _ = negrw_processor._score_negrw(negrw, variables)
        scores = models.score(X, events)
        record = make_templates(scores, combine['classes'], combine['channels'],
                                combine['binning']['edges'], {'nominal': weights}, g,
                                layout=channel_layout(combine),
                                split_values={v: table[v].to_numpy() for v in splits})
        for value in record.values():
            value.update(kind={'weight'}, chunks={path})
        entry = output['samples'][sample]
        target = entry.setdefault('templates', {}).get(categories[0])
        entry['templates'][categories[0]] = record if target is None else processor.accumulate([target, record])
        actual_rows[sample] += len(table)
        nparquets += 1
        scored_rows += len(table)
        if nparquets % 1000 == 0:
            print(f'{year}: scored {nparquets} shards, {scored_rows} rows', flush=True)
    if dict(actual_rows + Counter()) != dict(expected_rows + Counter()):
        raise ValueError(f'Selected row mismatch: expected={dict(expected_rows)}, actual={dict(actual_rows)}')
    if nparquets == 0:
        raise ValueError('No selected MC parquet inputs')
    return output, {'markers': nmarkers, 'parquets': nparquets, 'rows': scored_rows,
                    'selected_rows_by_sample': dict(expected_rows)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('-w', '--workflow', required=True)
    p.add_argument('-y', '--year', required=True)
    p.add_argument('--input_manifest', required=True, help='MC inputs.json with expected_partitions')
    p.add_argument('--model_manifest', required=True)
    p.add_argument('--output', required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--stat_only', action='store_true')
    mode.add_argument('--rate_systematics', action='store_true')
    p.add_argument('--also_rate_systematics', action='store_true',
                   help='with --stat_only, write rate-systematics cards from the same scored accumulator')
    p.add_argument('--merge_channel_splits', action='store_true')
    a = p.parse_args()
    if a.also_rate_systematics and not a.stat_only:
        p.error('--also_rate_systematics requires --stat_only')
    dest = Path(a.output)
    if dest.exists():
        raise FileExistsError(f'Refusing to overwrite {dest}')
    raw = Path(a.input_manifest).read_bytes()
    inventory = json.loads(raw)
    if inventory.get('workflow') != a.workflow:
        raise ValueError('Input inventory is for a different workflow')
    cfg = WorkflowConfigBuilder(a.workflow).build_workflow_config()
    f, s = signature(cfg), selection_signature(cfg)
    models = FoldModels(a.model_manifest, f, s, cfg.combine['classes'], year=a.year)
    # Do not initialize the full event processor or recompute any event corrections.
    neg = BaseProcessor.__new__(BaseProcessor)
    neg.mode, neg.workflow_config = 'run', cfg
    neg._negrw_bundle = neg._negrw_sha256 = None
    nhash = neg._negrw_model_identity()
    full_cfg = cfg
    cfg = card_config(full_cfg, a.merge_channel_splits, a.stat_only)
    routeb = RouteBConfig.from_workflow(a.workflow)
    output, counts = accumulate_train(inventory, a.year, cfg, models,
                                     get_dataset_config(a.year), f, s,
                                     _routeb_provenance(a.workflow) if routeb else None, neg)
    coverage = check_dataset_coverage(a.workflow, a.year, set(output['samples']),
                                      output['execution_checks']['files'])
    if negrw_identity(cfg) != nhash:
        raise ValueError('negrw model path changed during scoring')
    policy = {'source_mode': 'train', 'card_mode': 'stat_only' if a.stat_only else 'rate_systematics',
              'merge_channel_splits': a.merge_channel_splits, 'rate_params_kept': cfg.combine.get('rate_params', []),
              'coverage': coverage, 'counts': counts, 'negrw_model_sha256': nhash,
              'model_manifest_sha256': models.manifest_hash,
              'input_manifest_sha256': hashlib.sha256(raw).hexdigest(),
              'features_hash': f, 'selection_hash': s,
              'negrw_policy': 'rescore current model from stored generator features; per-sample/era yield preserving'}
    card = write_cards(output, cfg, a.year, dest, policy, routeb)
    save(output, str(dest/'nominal_accumulator.coffea'))
    if a.also_rate_systematics:
        rates = card_config(full_cfg, a.merge_channel_splits, False)
        rate_policy = dict(policy, card_mode='rate_systematics')
        write_cards(output, rates, a.year, dest/'rate_systematics', rate_policy, routeb)
    print(card)


if __name__ == '__main__':
    main()
