# Top-pT reweighting, applied to tt only.
#
# SOURCE (verified 2026-08-11): uhh-cms/hh2bbww, the analysis framework behind
# AN-24-091 (the Run 3 HH->bbWW note that is our closest same-era reference).
#   - hbw/production/top_pt_theory.py  -> top_pt_theory_weight
#   - hbw/config/config_run2.py L472   -> cfg.x.top_pt_theory_weight params
# which in turn cites
#   https://twiki.cern.ch/twiki/bin/viewauth/CMS/TopPtReweighting#TOP_PAG_corrections_based_on_the
#
# THEORY-based (NNLO/NLO) parameterisation, per top quark:
#     sf_run2 = 0.103 * exp(-0.0118 * pT) - 0.000134 * pT + 0.973
#     sf      = (0.991 + 0.000075 * pT) * sf_run2      <- Run 3 (13.6 TeV) rescaling
#     w_event = sqrt(sf(t) * sf(tbar))
#
# NOTE this is NOT the older data-based SF(pT) = exp(a + b*pT) with a=0.0615,
# b=-0.0005 (that is `cfg.x.top_pt_weight` in the same config, kept for the
# data-driven variant). The theory weight is what hh2bbww actually applies and
# what carries the `top_pt_up/down` shift.
#
# VARIATION CONVENTION (hh2bbww top_pt_theory.py):
#     nominal = the corrected weight
#     down    = 1.0                      i.e. "no correction applied"
#     up      = 2*(w - 1) + 1            i.e. symmetric about nominal
# So the FULL size of the correction is the +-1 sigma interval, matching what
# AN-24-091 states in prose. Unlike our first implementation, the correction IS
# applied to the nominal.
import awkward as ak
import numpy as np


def _top_pt_sf(pt):
    """Theory-based top-pT scale factor per top quark (hh2bbww parameterisation)."""
    sf_run2 = 0.103 * np.exp(-0.0118 * pt) - 0.000134 * pt + 0.973
    return (0.991 + 0.000075 * pt) * sf_run2


def add_toppt_weight(events, weights_container, shift, dataset):
    """Add top-pT reweighting for tt samples (nominal correction + up/down shift).

    Guarded to tt datasets; no-ops elsewhere. Events without exactly two last-copy
    generator tops get weight 1.0 rather than raising, since our tt samples are
    inclusive and a handful of events can fail the gen-level match.
    """
    if shift is not None:
        return
    if not dataset.startswith("TT"):
        return
    if "GenPart" not in events.fields:
        print("No GenPart in dataset, skip systematic: top pT reweight")
        return
    try:
        gp = events.GenPart
        # last-copy top quarks, matching hh2bbww's gen_parton_top producer
        # (abs(pdgId) == 6, hasFlags("isLastCopy") -> statusFlags bit 13)
        tops = gp[(abs(gp.pdgId) == 6) & ((gp.statusFlags & (1 << 13)) != 0)]

        sf = _top_pt_sf(tops.pt)
        weight = np.sqrt(ak.prod(sf, axis=1))
        # only events with exactly two gen tops get the correction
        weight = ak.where(ak.num(tops, axis=1) == 2, weight, 1.0)
        weight = ak.to_numpy(ak.values_astype(ak.fill_none(weight, 1.0), np.float64))

        weights_container.add(
            name="top_pt",
            weight=weight,
            weightUp=2.0 * (weight - 1.0) + 1.0,
            weightDown=np.ones_like(weight),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to build top pT reweight, skipping systematic: {exc}")
