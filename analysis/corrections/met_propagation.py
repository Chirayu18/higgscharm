"""Propagate calibrated object momenta into each independent MET shift.

MET is the negative visible transverse momentum sum. Use original NanoAOD
leptons as the common reference, not a previously corrected shift. This is the
minus-sign prescription of ua-cms/higgscharm coffea-2026 met.corrected_polar_met
at dfe3613daf681e29daaef039c82848bbe3249e83, applied to shift collections without
mutating shared awkward records.
"""
import awkward as ak
import numpy as np


def propagate_object_met(met, original, corrected):
    """Return a new MET record after replacing one visible-object collection."""
    if not np.array_equal(ak.to_numpy(ak.num(original)), ak.to_numpy(ak.num(corrected))):
        raise ValueError('MET propagation needs aligned original/corrected objects')
    dx = ak.sum(corrected.pt * np.cos(corrected.phi)
                - original.pt * np.cos(original.phi), axis=1)
    dy = ak.sum(corrected.pt * np.sin(corrected.phi)
                - original.pt * np.sin(original.phi), axis=1)
    px, py = met.pt * np.cos(met.phi) - dx, met.pt * np.sin(met.phi) - dy
    return ak.with_field(ak.with_field(met, np.hypot(px, py), 'pt'), np.arctan2(py, px), 'phi')


def propagate_lepton_shifts(shifts, original_leptons):
    """Propagate nominal and varied leptons once into every existing MET shift.

    A JES/JER/unclustered shift keeps its own MET and nominal leptons. A lepton
    shift starts with nominal jet MET and its own lepton momenta. Independent
    returned records prevent cross-shift aliasing and double application.
    """
    result = []
    for collections, name in shifts:
        updated = dict(collections)
        met = collections['MET']
        for key, original in original_leptons.items():
            if key not in collections:
                raise ValueError(f'{name}: missing {key} for MET propagation')
            met = propagate_object_met(met, original, collections[key])
        updated['MET'] = met
        result.append((updated, name))
    return result
