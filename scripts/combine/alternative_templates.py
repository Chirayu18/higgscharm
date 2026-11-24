"""Explicit sample replacement for RUN templates; never add alternatives to nominal.

For hDamp only TTto2L2Nu has supplied alternatives. Preserve nominal hadronic and
semileptonic tt, replacing just the named dilepton component with each dedicated
sample at its own L*xsec/full-generated-sumw. No whole-tt ratio or guessed shape
normalization is applied.
"""
import numpy as np


def alternative_spec(combine):
    specs = combine.get('alternative_samples', {})
    if not isinstance(specs, dict):
        raise ValueError('alternative_samples must map nuisance names to replacements')
    used = set()
    for name, spec in specs.items():
        if name in combine.get('shape_systematics', []) or not name or name.endswith(('Up', 'Down')):
            raise ValueError(f'{name}: duplicate or invalid alternative nuisance')
        if set(spec) != {'process', 'nominal_samples', 'samples'}:
            raise ValueError(f'{name}: require process, nominal_samples and samples')
        if spec['process'] not in combine['classes']:
            raise ValueError(f'{name}: unknown target process')
        if set(spec['samples']) != {'Up', 'Down'}:
            raise ValueError(f'{name}: require explicit Up and Down sample lists')
        for samples in [spec['nominal_samples'], *spec['samples'].values()]:
            if not isinstance(samples, list) or not samples or len(samples) != len(set(samples)):
                raise ValueError(f'{name}: require nonempty distinct sample lists')
        alt = spec['samples']['Up'] + spec['samples']['Down']
        if len(alt) != len(set(alt)) or set(alt) & set(spec['nominal_samples']) or used & set(alt):
            raise ValueError(f'{name}: overlapping alternative sample definitions')
        used.update(alt)
    return specs


def alternative_sample_names(combine):
    return {s for spec in alternative_spec(combine).values() for ss in spec['samples'].values() for s in ss}


def apply_alternative_templates(hists, variations, nominal, alternative, config, datasets, lumi,
                                normalize_record, policy):
    specs = alternative_spec(config.combine)
    if not specs:
        if alternative is not None:
            raise ValueError('Alternative inputs supplied without alternative_samples configuration')
        return
    if alternative is None:
        raise ValueError('Configured alternative_samples require --alternative_inputs')
    expected = alternative_sample_names(config.combine)
    if set(alternative['samples']) != expected:
        raise ValueError('Alternative sample set differs from explicit configuration')
    if set(nominal['samples']) & expected:
        raise ValueError('Alternative samples must not enter nominal templates')
    cat = next(iter(config.event_selection['categories']))
    channels = list(hists)
    nbins = len(next(iter(next(iter(hists.values())).values()))['nominal'][0])

    normalization_inputs = policy.setdefault('alternative_normalization_inputs', {})

    def summed(source, samples):
        out = {ch: (np.zeros(nbins), np.zeros(nbins)) for ch in channels}
        for sample in samples:
            if sample not in source['samples']:
                raise ValueError(f'{sample}: missing replacement input')
            info, entry = datasets[sample], source['samples'][sample]
            xsec, sumw = float(info['xsec']), float(entry['sumw'])
            if not np.isfinite(xsec) or xsec <= 0 or not np.isfinite(sumw) or sumw == 0:
                raise ValueError(f'{sample}: invalid alternative xsec/sumw')
            normalization_inputs[sample] = dict(xsec=xsec, full_generated_sumw=sumw, scale=lumi*xsec/sumw)
            records = entry.get('templates', {}).get(cat, {})
            record = records.get('nominal')
            if record is None:
                if entry.get('selected_events', -1) != 0:
                    raise ValueError(f'{sample}: missing nominal record without zero-selection evidence')
                continue
            normalized = normalize_record(record, lumi * xsec / sumw)
            if set(normalized) != set(channels):
                raise ValueError(f'{sample}: inconsistent alternative channel/binning')
            for ch, (c, v) in normalized.items():
                if ch not in out or c.shape != (nbins,) or v.shape != (nbins,):
                    raise ValueError(f'{sample}: inconsistent alternative channel/binning')
                if not np.all(np.isfinite(c)) or not np.all(np.isfinite(v)) or np.any(v < 0):
                    raise ValueError(f'{sample}: invalid alternative counts/variance')
                out[ch][0][:] += c
                out[ch][1][:] += v
        return out

    log = policy.setdefault('alternative_samples', {})
    for name, spec in specs.items():
        process = spec['process']
        for s in spec['nominal_samples']:
            if datasets[s]['process'].strip() not in config.combine['process_map'][process]:
                raise ValueError(f'{s}: replacement nominal sample does not map to {process}')
        removed = summed(nominal, spec['nominal_samples'])
        log[name] = dict(process=process, nominal_samples=spec['nominal_samples'],
                         samples=spec['samples'], normalization='L*xsec/full_generated_sumw', yields={})
        for direction, samples in spec['samples'].items():
            for s in samples:
                if datasets[s]['process'].strip() != name + direction:
                    raise ValueError(f'{s}: alternative process does not match {name + direction}')
            added = summed(alternative, samples)
            var = name + direction
            if var in variations:
                raise ValueError(f'Duplicate alternative variation {var}')
            variations.append(var)
            log[name]['yields'][direction] = {}
            for ch in channels:
                for p in config.combine['classes']:
                    c, v = hists[ch][p]['nominal']
                    if p == process:
                        rest_v = v - removed[ch][1]
                        if np.any(rest_v < -1e-10 * np.maximum(1., v)):
                            raise ValueError(f'{ch}/{p}: replacement variance exceeds nominal')
                        hists[ch][p][var] = (c - removed[ch][0] + added[ch][0],
                                            np.maximum(rest_v, 0.) + added[ch][1])
                    else:
                        hists[ch][p][var] = (c.copy(), v.copy())
                log[name]['yields'][direction][ch] = float(hists[ch][process][var][0].sum())
