"""Run 3 PUPPI XY correction adapted from Daniel's pinned coffea-2026 source.

Source: ua-cms/higgscharm dfe3613daf681e29daaef039c82848bbe3249e83,
analysis/corrections/met.py. Keep every pre-existing JEC/JER/unclustered shift,
use independent records, and fail on unsupported eras/calibration errors.
Statistical x/y terms follow the source's quadrature in polar coordinates.
"""
import awkward as ak
import numpy as np
from analysis.corrections.cset_cache import load_correction_set
from analysis.corrections.correctionlib_files import correction_files

EPOCHS = {'2022preEE': '2022', '2022postEE': '2022EE',
          '2023preBPix': '2023', '2023postBPix': '2023BPix'}


def apply_run3_met_xy(events, year, shifts, corrections_config):
    if year not in EPOCHS or not shifts or shifts[0][1] is not None:
        raise ValueError('MET XY requires a supported Run 3 era and nominal first shift')
    cset = load_correction_set(correction_files['met_phi'][year])
    corr = cset['met_xy_corrections']
    kind = 'MC' if hasattr(events, 'genWeight') else 'DATA'
    npv = np.asarray(ak.to_numpy(events.PV.npvsGood), dtype=np.float64)

    def evaluate(met, quantity):
        return corr.evaluate(quantity, 'PuppiMET', EPOCHS[year], kind, 'nom',
                             np.asarray(ak.to_numpy(np.clip(met.pt, 0., 6499.)), dtype=np.float64),
                             np.asarray(ak.to_numpy(np.clip(met.phi, -3.15, 3.15)), dtype=np.float64), npv)

    def replace(met, pt, phi, signed_variation=False):
        # Upstream JEC/MET can already contain nonfinite coordinates. Preserve
        # those rows for the existing finite MET selection; do not fail an
        # entire otherwise valid chunk or turn invalid inputs into finite MET.
        raw_pt = np.asarray(ak.to_numpy(met.pt), dtype=np.float64)
        raw_phi = np.asarray(ak.to_numpy(met.phi), dtype=np.float64)
        invalid_input = ~np.isfinite(raw_pt) | ~np.isfinite(raw_phi)
        pt, phi = np.asarray(pt, dtype=np.float64), np.asarray(phi, dtype=np.float64)
        bad = ~np.isfinite(pt) | ~np.isfinite(phi) | ((pt < 0) & ~np.asarray(signed_variation))
        if np.any(bad & ~invalid_input):
            raise ValueError('MET XY produced invalid coordinates from finite input; inspect calibration/input')
        pt = np.where(invalid_input, raw_pt, pt)
        phi = np.where(invalid_input, raw_phi, phi)
        if signed_variation:
            # pT - sigma can cross zero. Convert its signed polar vector to
            # canonical coordinates without clipping or changing x/y.
            phi = np.where(pt < 0, phi + np.pi, phi)
            pt = np.abs(pt)
        # Canonicalize angles after adding a variation, including at the +/-pi boundary.
        phi = np.arctan2(np.sin(phi), np.cos(phi))
        return ak.with_field(ak.with_field(met, pt, 'pt'), phi, 'phi')

    result = []
    for collections, name in shifts:
        updated = dict(collections)
        raw = collections['MET']
        updated['MET'] = replace(raw, evaluate(raw, 'pt'), evaluate(raw, 'phi'))
        result.append((updated, name))
    if kind == 'MC' and corrections_config['object_shifts']:
        raw = shifts[0][0]['MET']
        nominal = result[0][0]['MET']
        for direction, suffix, sign in [('Up', 'up', 1.), ('Down', 'dn', -1.)]:
            dp = np.hypot(evaluate(raw, 'pt_stat_x'+suffix) - nominal.pt,
                          evaluate(raw, 'pt_stat_y'+suffix) - nominal.pt)
            # Differences of angles must be wrapped before quadrature.
            deltas = [evaluate(raw, 'phi_stat_'+axis+suffix) - nominal.phi for axis in ('x','y')]
            df = np.hypot(*[np.arctan2(np.sin(d), np.cos(d)) for d in deltas])
            shifted = dict(result[0][0])
            shifted['MET'] = replace(nominal, nominal.pt + sign*dp, nominal.phi + sign*df, signed_variation=True)
            result.append((shifted, 'met_xy_'+year[:4]+direction))
    return result
