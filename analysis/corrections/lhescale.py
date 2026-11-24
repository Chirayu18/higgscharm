import warnings
import awkward as ak
import numpy as np

from analysis.corrections.lhe_yukawa_fix import corrected_scale_weights


def add_scalevar_weight(events, weights_container, shift, scale_norm=None, yukawa_table=None):
    """
    Twiki: https://twiki.cern.ch/twiki/bin/viewauth/CMS/TopSystematics#Factorization_and_renormalizatio

    __doc__:
    ['LHE scale variation weights (w_var / w_nominal)',
    ' [0] is renscfact=0.5d0 facscfact=0.5d0 ',
    ' [1] is renscfact=0.5d0 facscfact=1d0 ',
    ' [2] is renscfact=0.5d0 facscfact=2d0 ',
    ' [3] is renscfact=1d0 facscfact=0.5d0 ',
    ' [4] is renscfact=1d0 facscfact=1d0 ',
    ' [5] is renscfact=1d0 facscfact=2d0 ',
    ' [6] is renscfact=2d0 facscfact=0.5d0 ',
    ' [7] is renscfact=2d0 facscfact=1d0 ',
    ' [8] is renscfact=2d0 facscfact=2d0 ']
    """
    lhe_weights = events.LHEScaleWeight
    nom = np.ones(len(weights_container.weight()))
    if shift is None:
        if len(lhe_weights) > 0:
            if len(lhe_weights[0]) == 9:
                # FIX 2026-10-01: the nominal factor must be 1, NOT LHEScaleWeight[4].
                # Before, all three nuisances used nom = w[4], so every event weight was
                # multiplied by w[4]**3 (invisible when w[4] ~ 1, but up to ~1e13 in the
                # WHTo2Tau powheg samples where w[4] reaches ~2e4). coffea Weights.add
                # divides weightUp/Down by the nominal factor, so pass w_var / w[4] with
                # nominal 1 -> the modifiers are exactly the ratios to the mu=1 entry.
                w4 = np.asarray(lhe_weights[:, 4], dtype=np.float64)
                safe = np.where(w4 != 0, w4, 1.0)

                # FIX 2026-10-13 (central H+c / H+b, opt-in): per-event running-Yukawa correction (lhe_yukawa_fix.py).
                # Replaces the global generator-mean division, which left a -6..-10% one-sided bias in the SR.
                if yukawa_table is not None:
                    if "LHEPdfWeight" not in events.fields:
                        raise RuntimeError("yukawa fix needs LHEPdfWeight to measure the per-event factor")
                    _corr, _info = corrected_scale_weights(
                        ak.to_numpy(ak.to_regular(lhe_weights)), ak.to_numpy(ak.to_regular(events.LHEPdfWeight)), yukawa_table)
                    if _info["n_fallback"] > max(10, 0.001 * _info["n"]):
                        raise RuntimeError(f"yukawa fix ({yukawa_table}): too many fallback events {_info}")

                def r(i):
                    if yukawa_table is not None:
                        return _corr[:, i]
                    rr = np.where(w4 != 0, np.asarray(lhe_weights[:, i], dtype=np.float64) / safe, 1.0)
                    # optional per-file normalisation (lhe_norm.py): divide by the generator-level mean
                    if scale_norm is not None and scale_norm[i] > 0:
                        rr = rr / scale_norm[i]
                    return rr

                # FIX 2026-10-29 (ported from the HcZZ branch, her 2026-08-28 fix): Up/Down were swapped.
                # Weights.add takes (name, weight, weightUp, weightDown); Up must be the scale x2 entry
                # ([7] muR=2, [5] muF=2, [8] both=2), Down the x0.5 entry ([1], [3], [0]).
                one = np.ones(len(w4))
                weights_container.add("scalevar_muR", one, r(7), r(1))
                weights_container.add("scalevar_muF", one, r(5), r(3))
                weights_container.add("scalevar_muR_muF", one, r(8), r(0))
            elif len(lhe_weights[0]) > 1:
                print("Scale variation vector has length ", len(lhe_weights[0]))
        else:
            warnings.warn(
                "LHE scale variation weights are not available, put nominal weights"
            )
            weights_container.add("scalevar_muR", nom, nom, nom)
            weights_container.add("scalevar_muF", nom, nom, nom)
            weights_container.add("scalevar_muR_muF", nom, nom, nom)

    else:
        weights_container.add("scalevar_3pt", nom)
