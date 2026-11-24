#!/usr/bin/env python3
"""Build per-era ROOT/datacard inputs from additive inline RUN statistics.

Normalization happens after all files/chunks are merged, including zero-selection
chunks. Missing shapes require an explicit per-sample not-applicable declaration.
Every MC sample the workflow runs over must be present and fully processed, or
be excluded explicitly with a reason. Signed per-chunk records stay signed; the
approved legacy final-template policy (make_combine_inputs.main: empty
object-shift fallback, coordinated non-positive-bin floor, optional smoothing and
re-floor) is applied once, to the final summed templates, in the legacy order.
This command never runs Combine or submits jobs.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import yaml
from coffea import processor
from coffea.util import load
from analysis.filesets.utils import get_dataset_config
from analysis.features.hww import signature, selection_signature
from analysis.sinks.hww import final_channels, card_combine
from analysis.processing_checks import sample_kind, check_dataset_coverage, load_exclusions
from analysis.workflows.config import WorkflowConfigBuilder
from make_combine_inputs import (write_root, write_data_obs, write_datacard, load_lumi,
                                 clip_negative_bins, smooth_shape_variations)


from alternative_templates import alternative_spec, alternative_sample_names, apply_alternative_templates
from template_prescriptions import load_card_policy, normalize_shape_only


def configure_era(config, year):
    # Same explicit era translation as the existing full-production wrapper.
    from copy import deepcopy
    config = deepcopy(config)
    yy = '2023' if year.startswith('2023') else '2022'
    combine = config.combine
    combine['shape_systematics'] = [s[:-4] + yy if s.startswith('CMS_') and s.endswith('_2022') else s
                                    for s in combine['shape_systematics']]
    ln = combine.get('lnN', {})
    if 'lumi_13p6TeV' in ln:
        ln['lumi_13p6TeV_' + yy] = {p: {'2022':1.014, '2023':1.013}[yy]
                                  for p in ln.pop('lumi_13p6TeV')}
    return config


def normalize_record(record, scale=1.):
    flags = record['negrw']
    if len(flags) != 1:
        raise ValueError('Mixed negrw configuration in accumulated template')
    renorm = 1.
    if True in flags:
        if record['norm_rw'] == 0:
            raise ValueError('negrw normalization denominator is zero')
        renorm = record['norm_raw'] / record['norm_rw']
    factor = scale * renorm
    return {ch: (v * factor, record['sumw2'][ch] * factor**2)
            for ch, v in record['counts'].items()}


NEGRW_DISABLED = 'disabled'


def negrw_identity(config):
    """Expected RUN `negrw_model_sha256` for a workflow: sha256 of the negrw.model file, or 'disabled'."""
    negrw = getattr(config, 'negrw', None)
    if not negrw:
        return NEGRW_DISABLED
    return hashlib.sha256(Path(negrw['model']).read_bytes()).hexdigest()


def _check_negrw_identity(path, out):
    # G2 (2026-11-12): checked per file, because accumulate() unions provenance sets and would let a file
    # without the key merge silently with one that has it.
    values = out.get('provenance', {}).get('negrw_model_sha256')
    if not values:
        raise ValueError(f'{path}: RUN provenance has no negrw_model_sha256 (pre-2026-11-12 output); '
                         'reprocess with the current code')
    for v in values:
        if v != NEGRW_DISABLED and not (isinstance(v, str) and re.fullmatch(r'[0-9a-f]{64}', v)):
            raise ValueError(f'{path}: malformed negrw_model_sha256 {v!r}')


def merge_outputs(files):
    files = [Path(p).resolve() for p in files]
    if len(set(files)) != len(files) or not files:
        raise ValueError('Supply nonempty, distinct RUN output files')

    def checked():
        for p in files:
            out = load(str(p))
            # Written only when the job's entry-count check passed (submit.py).
            if 'files' not in out.get('execution_checks', {}):
                raise ValueError(f'{p}: no execution_checks file record; incomplete or pre-fix job output')
            if out.get('provenance', {}).get('mode') == {'run'}:
                _check_negrw_identity(p, out)
            yield out
    result = processor.accumulate(checked())
    for name, values in result.get('provenance', {}).items():
        if len(values) != 1:
            raise ValueError(f'Incompatible RUN provenance: {name}={values}')
    if result.get('provenance', {}).get('mode') != {'run'}:
        raise ValueError('Only RUN outputs are accepted')
    for sample, entry in result['samples'].items():
        if entry['chunks_read'] != len(entry['chunks']):
            raise ValueError(f'Duplicate input chunks for {sample}; exclude retries/overlapping jobs')
    return result


def _syst(var):
    return var[:-2] if var.endswith('Up') else var[:-4] if var.endswith('Down') else ''


def make_process_hists(output, config, year, not_applicable=None, policy=None, routeb=None, alternative_output=None):
    """Sum normalized templates per (channel, class, variation).

    policy (optional dict) receives an audit log of every explicit decision.
    routeb (RouteBConfig or None): Route B H+c 3FS/4FS stitching. Its samples are scaled by
    L*X*c_s/F_s (analysis/stitching/routeb_io.sink_scale_factors) instead of L*X/sumw.
    """
    not_applicable = not_applicable or {}
    policy = {} if policy is None else policy
    log = policy.setdefault('declared_not_applicable_used', [])
    combine = config.combine
    category = next(iter(config.event_selection['categories']))
    classes, channels = combine['classes'], final_channels(combine)
    edges = np.asarray(combine['binning']['edges'], dtype=float)
    datasets, lumi = get_dataset_config(year), load_lumi(year)
    mapped = {}
    evaluated = set()
    data_skipped = []
    for sample, entry in output['samples'].items():
        info = datasets[sample]
        if sample_kind(sample, info) == 'data':
            data_skipped.append(sample)  # cards use Asimov data_obs; data never enters templates
            continue
        names = [p for p, procs in combine['process_map'].items() if info['process'].strip() in procs and p in classes]
        if len(names) != 1:
            raise ValueError(f'{sample}: process {info["process"]!r} has no unique card mapping')
        records = entry.get('templates', {}).get(category, {})
        mapped[sample] = (names[0], records)
        if 'evaluated_shifts' not in entry:
            raise ValueError(f'{sample}: RUN output lacks evaluated_shifts; reprocess with the fixed code')
        evaluated.update(entry['evaluated_shifts'])
    policy['data_samples_skipped'] = sorted(data_skipped)
    weight_shapes = list(combine['shape_systematics'])
    negrw = combine.get('negrw_shape_name', 'CMS_negrw_vjets')
    if combine.get('negrw_uncertainty', True) and 'vjets' in classes:
        weight_shapes += [negrw]
    object_shapes = sorted({v[:-2] for v in evaluated if v.endswith('Up')
                            and v[:-2] not in weight_shapes and v[:-2] + 'Down' in evaluated})
    variations = ['nominal'] + [s + d for s in weight_shapes + object_shapes for d in ('Up', 'Down')]
    result = {ch: {p: {v: (np.zeros(len(edges)-1), np.zeros(len(edges)-1)) for v in variations}
                   for p in classes} for ch in channels}
    routeb_scale = {}
    if routeb is not None:
        from analysis.stitching.routeb_io import sink_scale_factors
        routeb_scale, norm = sink_scale_factors(
            routeb, {s: output['samples'][s] for s in mapped}, lumi,
            {s: float(datasets[s]['xsec']) for s in mapped}, era_samples=datasets)
        if norm is not None:
            policy['routeb_stitching'] = {'samples': routeb_scale, 'c': norm.c, 'f': norm.f,
                                          'f_stat': norm.f_stat, 'closure': norm.closure,
                                          'neg_fraction': norm.neg_fraction}
    seen = set()
    for sample, (process, records) in mapped.items():
        entry, info = output['samples'][sample], datasets[sample]
        xsec, sumw = float(info['xsec']), float(entry['sumw'])
        if not xsec > 0 or not np.isfinite(xsec) or not np.isfinite(sumw) or sumw == 0:
            raise ValueError(f'{sample}: invalid xsec/sumw {xsec}/{sumw}')
        if not records:
            # Full preselection sumw still contributes. No selected events means zero templates.
            continue
        if 'nominal' not in records:
            raise ValueError(f'{sample}: variations exist without nominal')
        nominal_chunks = records['nominal']['chunks']
        seen.add(process)
        for var in variations:
            syst = _syst(var)
            declared = syst in not_applicable.get(sample, [])
            excluded = (declared
                        or (syst.startswith('scalevar') and process in combine.get('no_scalevar', []))
                        or (syst.startswith(('scalevar', 'ps_')) and process in combine.get('no_theory', []))
                        or (syst == negrw and process != 'vjets'))
            record = records.get(var)
            if syst in object_shapes:
                done = entry['evaluated_shifts'].get(var, set())
                if done == entry['chunks']:
                    # Evaluated on every chunk: an absent record is a real zero yield
                    # (all shifted events left the selection), never nominal.
                    if record is None:
                        continue
                elif declared and not done:
                    record = records['nominal']
                    log.append([sample, var])
                else:
                    raise ValueError(f'{sample}: object shift {var} evaluated on {len(done)} of '
                                     f'{len(entry["chunks"])} chunks; declare not-applicable explicitly')
            elif record is None:
                if not excluded:
                    raise ValueError(f'{sample}: missing required variation {var}; declare not-applicable explicitly')
                if declared:
                    log.append([sample, var])
                record = records['nominal']
            elif var != 'nominal' and record['chunks'] != nominal_chunks:
                # A weight variation present in only some chunks is an incomplete template.
                raise ValueError(f'{sample}: variation {var} covers {len(record["chunks"])} of '
                                 f'{len(nominal_chunks)} nominal chunks')
            scale = routeb_scale[sample]['scale'] if sample in routeb_scale else lumi * xsec / sumw
            normalized = normalize_record(record, scale)
            for ch, (counts, sumw2) in normalized.items():
                result[ch][process][var][0][:] += counts
                result[ch][process][var][1][:] += sumw2
    if set(classes) - seen:
        raise ValueError(f'Missing selected processes: {sorted(set(classes)-seen)}')
    apply_alternative_templates(result, variations, output, alternative_output, config,
                                datasets, lumi, normalize_record, policy)
    return result, variations, object_shapes, edges


def finalize_templates(hists, combine, variations, object_shapes, policy=None):
    """Approved legacy final-template policy, same order as make_combine_inputs.main.

    1. empty object shift for a (channel, process) with nominal > 0 -> nominal
       (legacy rule; each application is logged; combine.object_shift_empty_fallback:
       'error' refuses instead);
    2. coordinated floor of non-positive bins (clip_negative_bins, floor 1e-6);
    3. if combine.smooth_shapes: smooth_shape_variations, then re-floor.
    Applies to final summed templates only; per-chunk records stay signed.
    """
    policy = {} if policy is None else policy
    channels, processes = list(hists), list(next(iter(hists.values())))
    mode = combine.get('object_shift_empty_fallback', 'nominal')
    if mode not in ('nominal', 'error'):
        raise ValueError(f'object_shift_empty_fallback must be nominal or error, not {mode!r}')
    shift_vars = [s + d for s in object_shapes for d in ('Up', 'Down')]
    applied = policy.setdefault('object_shift_empty_fallback', [])
    for ch in channels:
        for cp in processes:
            nom_c, nom_s2 = hists[ch][cp]['nominal']
            for vn in shift_vars:
                c, _ = hists[ch][cp][vn]
                if c.sum() <= 0 < nom_c.sum():
                    if mode == 'error':
                        raise ValueError(f'{ch}/{cp}: object shift {vn} is empty with nominal {nom_c.sum():.4g}')
                    applied.append([ch, cp, vn, float(nom_c.sum())])
                    hists[ch][cp][vn] = (nom_c.copy(), nom_s2.copy())
    pairs = [(v, '') for v in variations]
    policy['nominal_bins_floored'] = clip_negative_bins(hists, channels, processes, pairs)
    if combine.get('smooth_shapes'):
        policy['smoothed'] = smooth_shape_variations(hists, channels, processes, pairs)
        clip_negative_bins(hists, channels, processes, pairs)
    normalize_shape_only(hists, combine, variations, policy)
    return hists


def validate_alternative_inputs(nominal, alternative, config, year, fileset_json=None):
    """Strict sample/file coverage and identical model/selection provenance."""
    expected = alternative_sample_names(config.combine)
    if not expected or alternative['provenance'] != nominal['provenance']:
        raise ValueError('Alternative and nominal RUN provenance differ (selection/features/models/era)')
    return check_dataset_coverage('', year, set(alternative['samples']),
                                  alternative['execution_checks']['files'],
                                  fileset_json=fileset_json, expected=expected)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('-w', '--workflow', required=True)
    p.add_argument('-y', '--year', required=True)
    p.add_argument('--inputs', nargs='+', required=True)
    p.add_argument('--card_policy', help='explicit card-only alternative/shape-normalization YAML')
    p.add_argument('--alternative_inputs', nargs='+', help='separate RUN outputs for configured alternative samples')
    p.add_argument('--output', required=True, help='new output directory')
    p.add_argument('--not_applicable', help='YAML sample -> list of inapplicable nuisance names')
    p.add_argument('--exclusions', help='YAML sample -> reason for MC samples deliberately absent')
    p.add_argument('--fileset_json', help='override the era fileset JSON used for file coverage')
    a = p.parse_args()
    dest = Path(a.output)
    if dest.exists():
        raise FileExistsError(f'Refusing to overwrite {dest}')
    cfg = load_card_policy(WorkflowConfigBuilder(a.workflow).build_workflow_config(), a.card_policy)
    output = merge_outputs(a.inputs)
    prov = output['provenance']
    if (prov['year'] != {a.year} or prov['features_hash'] != {signature(cfg)}
            or prov['selection_hash'] != {selection_signature(cfg)}):
        raise ValueError('Workflow/year differs from RUN output provenance')
    if prov['negrw_model_sha256'] != {negrw_identity(cfg)}:
        raise ValueError(f'negrw model of the RUN outputs {prov["negrw_model_sha256"]} differs from the '
                         f'workflow negrw.model ({negrw_identity(cfg)})')
    datasets = get_dataset_config(a.year)
    mc = {s for s in output['samples'] if sample_kind(s, datasets[s]) == 'mc'}
    exclusions = load_exclusions(a.exclusions)
    coverage = check_dataset_coverage(a.workflow, a.year, mc,
                                      {s: v for s, v in output['execution_checks']['files'].items() if s in mc},
                                      exclusions, a.fileset_json)
    na = yaml.safe_load(Path(a.not_applicable).read_text()) if a.not_applicable else {}
    alternative = None
    alt_coverage = None
    expected_alt = alternative_sample_names(cfg.combine)
    if bool(a.alternative_inputs) != bool(expected_alt):
        raise ValueError('alternative_samples configuration and --alternative_inputs must be supplied together')
    if expected_alt:
        alternative = merge_outputs(a.alternative_inputs)
        alt_coverage = validate_alternative_inputs(output, alternative, cfg, a.year, a.fileset_json)
    cfg = configure_era(cfg, a.year)
    policy = {'coverage': coverage, 'negrw_model_sha256': next(iter(prov['negrw_model_sha256']))}
    if cfg.combine.get('rate_constraints_provenance'):
        policy['rate_constraints'] = cfg.combine['rate_constraints_provenance']
    if a.card_policy:
        policy['card_policy'] = {'path': str(Path(a.card_policy).resolve()),
                               'sha256': hashlib.sha256(Path(a.card_policy).read_bytes()).hexdigest()}
    if alt_coverage is not None:
        policy['alternative_coverage'] = alt_coverage
        policy['alternative_inputs'] = [str(Path(p).resolve()) for p in a.alternative_inputs]
    from analysis.stitching.hplusc_routeb import RouteBConfig
    routeb = RouteBConfig.from_workflow(a.workflow)  # None unless the workflow has a stitching block
    hists, variations, object_shapes, edges = make_process_hists(output, cfg, a.year, na, policy, routeb, alternative)
    finalize_templates(hists, cfg.combine, variations, object_shapes, policy)
    dest.mkdir(parents=True)
    root, card = dest/'templates.root', dest/'datacard.txt'
    write_root(root, hists, final_channels(cfg.combine), cfg.combine['classes'],
               [(v, '') for v in variations], edges)
    write_data_obs(root, hists, final_channels(cfg.combine),
                   [p for p in cfg.combine['classes'] if p != cfg.combine['signal']], edges)
    write_datacard(card, root.name, card_combine(cfg.combine), hists, edges, object_shapes)
    (dest/'template_policy.json').write_text(json.dumps(policy, indent=2, sort_keys=True) + '\n')
    # Read every written object back; malformed histograms are fatal.
    import uproot
    with uproot.open(root) as f:
        for name in f.keys(cycle=False):
            if not np.all(np.isfinite(f[name].values())) or not np.all(np.isfinite(f[name].variances())):
                raise ValueError(f'Nonfinite written histogram: {name}')
    print(card)


if __name__ == '__main__':
    main()
