"""Additive template statistics; normalize only after summing every chunk.

Each variation stores (counts, sumw2) for all argmax channels, and additive
normalization sums for the yield-preserving negrw correction. Squared weights
are accumulated before any normalization, preserving variance exactly.
"""
import re
import numpy as np
from analysis.features.hww import event_folds


def dataset_info(dataset, config):
    if dataset in config:
        return dataset, config[dataset]
    matches = [(k, v) for k, v in config.items()
               if dataset.startswith(k + '_') and re.fullmatch(r'(\d+|test)', dataset[len(k)+1:])]
    if len(matches) != 1:
        raise ValueError(f'Dataset {dataset!r} does not resolve uniquely in the era fileset')
    return matches[0]


def labels(process, groups, n):
    matches = [i for i, (key, values) in enumerate(groups.items()) if process in values]
    if len(matches) > 1:
        raise ValueError(f'Process {process} belongs to multiple training classes')
    # Data and deliberately unclassified backgrounds remain available for closure,
    # but cannot silently become supervised training examples.
    label = matches[0] if matches else -1
    out = {'label': np.full(n, label, np.int16)}
    out.update({'is_' + k: np.full(n, int(i == label), np.int8) for i, k in enumerate(groups)})
    return out


def channel_layout(combine):
    """Final channel -> (class, split variable, lo, hi), lo <= value < hi (hi None = open).

    combine.channels keeps one argmax channel per class (legacy builders read it
    unchanged). combine.channel_splits optionally splits a channel on a per-event
    variable, e.g. the AN-23-102 SR split Nc-j = 1 / Nc-j > 1 (2026-11-11).
    """
    channels = combine['channels']
    splits = combine.get('channel_splits') or {}
    unknown = set(splits) - set(channels)
    if unknown:
        raise ValueError(f'channel_splits for unknown channels: {sorted(unknown)}')
    layout = {}
    for ch, cls in channels.items():
        if ch not in splits:
            layout[ch] = (cls, None, None, None)
            continue
        spec = splits[ch]
        var, bins = spec['variable'], spec['bins']
        if len(bins) < 2:
            raise ValueError(f'{ch}: a split needs at least two sub-channels')
        prev = None
        items = list(bins.items())
        for i, (sub, edge) in enumerate(items):
            lo, hi = edge
            if sub in channels or sub in layout:
                raise ValueError(f'{ch}: duplicate channel name {sub}')
            if lo is None or (hi is not None and hi <= lo):
                raise ValueError(f'{ch}/{sub}: bad split edges {edge}')
            if prev is not None and lo != prev:
                raise ValueError(f'{ch}: split bins must be contiguous')
            if hi is None and i != len(items) - 1:
                raise ValueError(f'{ch}: only the last split bin may be open')
            layout[sub] = (cls, var, lo, hi)
            prev = hi
    return layout


def final_channels(combine):
    return list(channel_layout(combine))


def card_combine(combine):
    """combine block with channels expanded to the final (split) channels."""
    out = dict(combine)
    out['channels'] = {ch: v[0] for ch, v in channel_layout(combine).items()}
    return out


def split_variables(combine):
    return sorted({v[1] for v in channel_layout(combine).values() if v[1]})


def make_templates(scores, classes, channels, edges, weight_map, negrw=None,
                   negrw_std=None, negrw_name='CMS_negrw_vjets', nominal_weight=None,
                   layout=None, split_values=None):
    scores = np.asarray(scores)
    winner = np.argmax(scores, axis=1)
    D = np.max(scores, axis=1)
    edges = np.asarray(edges, dtype=np.float64)
    if edges.ndim != 1 or len(edges) < 2 or not np.all(np.diff(edges) > 0):
        raise ValueError('Discriminant edges must be strictly increasing')
    D = np.clip(D, edges[0], np.nextafter(edges[-1], edges[0]))
    if set(channels.values()) != set(classes) or len(channels) != len(classes):
        raise ValueError('RUN needs one exclusive argmax channel per class')
    if layout is None:
        layout = {ch: (cls, None, None, None) for ch, cls in channels.items()}
    masks = {}
    for ch, (cls, var, lo, hi) in layout.items():
        mask = winner == classes.index(cls)
        if var is not None:
            if split_values is None or var not in split_values:
                raise ValueError(f'RUN channel split needs per-event {var!r}')
            v = np.asarray(split_values[var], dtype=np.float64)
            if v.shape != winner.shape:
                raise ValueError(f'Split variable {var!r} misaligned with scores')
            mask = mask & (v >= lo)
            if hi is not None:
                mask = mask & (v < hi)
        masks[ch] = mask
    for i, cls in enumerate(classes):
        covered = sum(masks[ch].astype(np.int64) for ch, v in layout.items() if v[0] == cls)
        if not np.array_equal(covered, (winner == i).astype(np.int64)):
            raise ValueError(f'Channel split does not cover every {cls} event exactly once')
    nominal = np.asarray(weight_map['nominal'] if nominal_weight is None else nominal_weight, dtype=np.float64)
    if negrw is not None:
        negrw = np.asarray(negrw, dtype=np.float64)
        if negrw.shape != nominal.shape or not np.all(np.isfinite(negrw)):
            raise ValueError('Invalid negrw scores')
    variations = {k: (np.asarray(w, dtype=np.float64), negrw) for k, w in weight_map.items()}
    if negrw is not None and negrw_std is not None and negrw_name:
        variations[negrw_name + 'Up'] = (nominal, np.clip(negrw + negrw_std, -1, 1))
        variations[negrw_name + 'Down'] = (nominal, np.clip(negrw - negrw_std, -1, 1))
    out = {}
    for name, (weight, g) in variations.items():
        if weight.shape != nominal.shape or not np.all(np.isfinite(weight)):
            raise ValueError(f'Nonfinite or malformed RUN weights: {name}')
        actual = weight if g is None else np.abs(weight) * g
        record = {'counts': {}, 'sumw2': {}, 'norm_raw': float(nominal.sum()),
                  'norm_rw': float(nominal.sum() if g is None else (np.abs(nominal) * g).sum()),
                  'negrw': {g is not None}}
        for ch, mask in masks.items():
            record['counts'][ch] = np.histogram(D[mask], bins=edges, weights=actual[mask])[0]
            record['sumw2'][ch] = np.histogram(D[mask], bins=edges, weights=actual[mask]**2)[0]
        out[name] = record
    return out
