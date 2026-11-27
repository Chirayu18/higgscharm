"""Explicit card-only prescriptions; selection/model signatures stay separate."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import yaml


def load_card_policy(config, path):
    if path is None:
        return config
    policy = yaml.safe_load(Path(path).read_text())
    allowed = {'alternative_samples', 'shape_only_systematics', 'rate_constraints'}
    if not isinstance(policy, dict) or not policy or set(policy) - allowed:
        raise ValueError('Card policy supports only alternative_samples, shape_only_systematics and rate_constraints')
    out = deepcopy(config)
    out.combine.update({k:v for k,v in policy.items() if k != 'rate_constraints'})
    constraints = policy.get('rate_constraints', {})
    if not isinstance(constraints, dict):
        raise ValueError('rate_constraints must map process -> sourced lnN constraint')
    for process, spec in constraints.items():
        if process not in out.combine['classes'] or not isinstance(spec,dict) or set(spec) != {'name', 'value', 'source'}:
            raise ValueError(f'{process}: rate constraint needs name, value and source')
        name, value, source = spec['name'], spec['value'], spec['source']
        if not isinstance(name,str) or not name.strip() or any(c.isspace() for c in name):
            raise ValueError(f'{process}: invalid rate constraint name')
        if not isinstance(source,str) or not source.strip():
            raise ValueError(f'{process}: rate constraint needs a nonempty source')
        try:
            terms = [float(x) for x in str(value).split('/')]
        except (ValueError, TypeError) as e:
            raise ValueError(f'{process}: invalid lnN constraint') from e
        if len(terms) not in (1,2) or not all(np.isfinite(v) and v > 0 for v in terms) or all(v == 1 for v in terms):
            raise ValueError(f'{process}: invalid or trivial lnN constraint')
        if len(terms)==1 and terms[0] < 1:
            raise ValueError(f'{process}: a symmetric lnN factor must exceed 1')
        if name in out.combine.get('shape_systematics', []) or name in out.combine.get('alternative_samples', {}):
            raise ValueError(f'{name}: constraint name collides with a shape nuisance')
        if process in out.combine.get('lnN', {}).get(name, {}):
            raise ValueError(f'{name}: constraint already applies to {process}')
        out.combine.setdefault('lnN', {}).setdefault(name, {})[process] = value
        out.combine['rate_params'] = [p for p in out.combine.get('rate_params', []) if p != process]
    if constraints:
        out.combine['rate_constraints_provenance'] = deepcopy(constraints)
    return out


def normalize_shape_only(hists, combine, variations, policy):
    """Remove each process's varied rate across all final channels of one era.

    Matches htoww_submit_logs/higgs_scale_variants.py (12 Nov): normalize the
    total over channels, retaining SR/CR and Ncjet migrations. Applied after the
    existing floor/smoothing so the written integrals equal the written nominal.
    Variances scale by the square of the same factor. No nuisance/correlation or
    inclusive rate value is invented.
    """
    specs = combine.get('shape_only_systematics', {})
    if not isinstance(specs, dict):
        raise ValueError('shape_only_systematics must map nuisance -> process list')
    log = policy.setdefault('shape_only_systematics', [])
    for name, processes in specs.items():
        if not isinstance(processes, list) or not processes or len(processes) != len(set(processes)):
            raise ValueError(f'{name}: require a nonempty distinct process list')
        if set(processes) - set(combine['classes']):
            raise ValueError(f'{name}: unknown shape-only process')
        if any(name + d not in variations for d in ('Up', 'Down')):
            raise ValueError(f'{name}: missing paired shape-only templates')
        for p in processes:
            nom = sum(float(hists[ch][p]['nominal'][0].sum()) for ch in hists)
            for d in ('Up', 'Down'):
                var = name + d
                total = sum(float(hists[ch][p][var][0].sum()) for ch in hists)
                if not np.isfinite(nom) or not np.isfinite(total) or min(nom, total) <= 0:
                    raise ValueError(f'{p}/{var}: shape-only requires positive finite total yields')
                factor = nom / total
                for ch in hists:
                    c, v = hists[ch][p][var]
                    hists[ch][p][var] = (c * factor, v * factor**2)
                log.append(dict(nuisance=name, process=p, direction=d,
                                scope='all_final_channels_per_era', nominal=nom,
                                varied_before=total, factor=factor))
