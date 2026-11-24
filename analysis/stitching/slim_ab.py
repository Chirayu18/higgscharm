"""Explicit offline arm selection for inclusive legacy-slim signal trees.

A: all central 4FS, no 3FS. B: 4FS N_GEN_cjet==0 + 3FS N_GEN_cjet>=1.
This selects rows only; normalization must use full preselection marker sums,
including all chunks and main/ext datasets. Never normalize from selected rows.
"""
import numpy as np
from analysis.stitching.hplusc_routeb import keep_mask


def arm_mask(sample, columns, config, arm):
    if arm not in ('A', 'B'):
        raise ValueError('Choose arm A or B explicitly')
    if not config.defer_selection:
        raise ValueError('Inclusive A/B selection requires defer_selection provenance')
    nrows = len(columns['event'])
    scheme = config.scheme_of(sample)
    if scheme is None:
        return np.ones(nrows, dtype=bool)
    n = np.asarray(columns['gen_routeb_cjet_multiplicity'])
    saved = np.asarray(columns['routeb_keep'])
    if n.shape != (nrows,) or n.dtype.kind not in 'iu' or (n < 0).any():
        raise ValueError('Malformed GEN c-jet counts')
    if saved.shape != (nrows,) or saved.dtype.kind != 'b':
        raise ValueError('Malformed stored B keep mask')
    keep = keep_mask(n, scheme)
    if not np.array_equal(keep, saved):
        raise ValueError('Stored B keep mask disagrees with the GEN split prescription')
    return np.full(nrows, scheme == '4FS', dtype=bool) if arm == 'A' else keep
