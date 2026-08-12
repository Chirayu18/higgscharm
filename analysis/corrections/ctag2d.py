"""PNet 2D pseudo-continuous HF-tagging scale factor (ParticleNetAK4_pseudocontinuous).

This is the *2D-category* c-tag SF (cmshgg ingredients flavTaggingSF_<campaign>.json.gz),
distinct from the 1D fixed-WP CTagCorrector (BTV particleNet_wc/tnp). It is applied to the
event candidate c-jet (objects["candidate_cjet"], the max-CvsL jet), which is the jet whose
2D category feeds the MVA discriminant.

Per event:
  * recompute the 2D category from btagPNetCvL (=CvsL) and btagPNetCvB (=CvsB),
  * map it to the SF wp id (L0=0, C0..C4=40..44, B0..B4=50..54),
  * evaluate the SF by hadronFlavour (0/4/5), wp, abseta (inclusive->0), pt,
  * FOLD central into the weight (correction),
  * add CMS_ctag2d_<year> Up/Down from up_Total / down_Total (single nuisance).

NOTE (2026-07-23 decision): uses up_Total/down_Total as ONE nuisance for now. Full
decorrelation into the SF-file sources (Stat per-bin + JES/PU/lepton/theory mapped onto the
analysis's own shared nuisance names, HiggsDNA bTagShapeSF style) is deferred to a
whole-card decorrelation pass. To switch: change SYST_UP/SYST_DN below (e.g. up_Stat) or
loop over multiple source keys and .add() one nuisance each.
"""
import numpy as np
import awkward as ak
import correctionlib
from typing import Type
from coffea.analysis_tools import Weights
from analysis.corrections.correctionlib_files import correction_files

# 2D category edges (mirror the analysis MVA one-hot binning)
X_HFVLF_EDGES = [0.0, 0.250, 0.452, 0.808, 1.000]
Y_BVC_EDGES = [0.0, 0.006, 0.017, 0.055, 0.761, 0.944, 0.985, 0.995, 1.0]
CATS = ["L0", "C0", "C1", "C2", "C3", "C4", "B0", "B1", "B2", "B3", "B4"]
CID = {n: i for i, n in enumerate(CATS)}
WP_ID = {"L0": 0, "C0": 40, "C1": 41, "C2": 42, "C3": 43, "C4": 44,
         "B0": 50, "B1": 51, "B2": 52, "B3": 53, "B4": 54}
CAT_TO_WP = np.array([WP_ID[c] for c in CATS], dtype=np.int64)

NUIS_YEAR = {"2022preEE": "2022", "2022postEE": "2022",
             "2023preBPix": "2023", "2023postBPix": "2023"}
# single-nuisance source keys (decision 2026-07-23: Total; decorrelate later)
SYST_UP = "up_Total"
SYST_DN = "down_Total"
PT_BINS = (20.0001, 9999.0)


def _category_np(cvsl, cvsb):
    den = cvsl + cvsb * (1.0 - cvsl)
    with np.errstate(invalid="ignore", divide="ignore"):
        x = np.where(den != 0, cvsl / den, np.nan)
    y = 1.0 - cvsb
    good = np.isfinite(x) & np.isfinite(y)
    cat = np.full(x.shape, -1, dtype=np.int64)
    xe, ye = X_HFVLF_EDGES, Y_BVC_EDGES
    left = good & (x < xe[3]); right = good & (x >= xe[3])
    def put(m, n): cat[m] = CID[n]
    put(left & (x < xe[1]), "L0")
    put(left & (x >= xe[1]) & (x < xe[2]), "C0")
    put(left & (x >= xe[2]), "C1")
    put(right & (y < ye[1]), "C4")
    put(right & (y >= ye[1]) & (y < ye[2]), "C3")
    put(right & (y >= ye[2]) & (y < ye[3]), "C2")
    put(right & (y >= ye[3]) & (y < ye[4]), "B0")
    put(right & (y >= ye[4]) & (y < ye[5]), "B1")
    put(right & (y >= ye[5]) & (y < ye[6]), "B2")
    put(right & (y >= ye[6]) & (y < ye[7]), "B3")
    put(right & (y >= ye[7]), "B4")
    return cat


class CTag2DCorrector:
    def __init__(self, events, weights: Type[Weights], year: str, shift: str) -> None:
        self._year = year
        self._weights = weights
        self._shift = shift
        self._corr = correctionlib.CorrectionSet.from_file(
            correction_files["ctagging_2d"][year]
        )["ParticleNetAK4_pseudocontinuous"]
        self._nuis = f"CMS_ctag2d_{NUIS_YEAR[year]}"
        self._n = len(events)
        # event candidate c-jet, attached by base.py as 'selected_candidate_cjet'.
        # If absent (category without it), no-op: SF=1 everywhere.
        if "selected_candidate_cjet" not in events.fields:
            self._cvsl = np.full(self._n, np.nan)
            self._cvsb = np.full(self._n, np.nan)
            self._pt = np.full(self._n, np.nan)
            self._flav = np.zeros(self._n, dtype=np.int64)
            return
        # ak.firsts -> one per event, may be None
        cj = ak.firsts(events.selected_candidate_cjet)
        self._cvsl = ak.to_numpy(ak.fill_none(cj.btagPNetCvL, np.nan)).astype(np.float64)
        self._cvsb = ak.to_numpy(ak.fill_none(cj.btagPNetCvB, np.nan)).astype(np.float64)
        self._pt = ak.to_numpy(ak.fill_none(cj.pt, np.nan)).astype(np.float64)
        flav = ak.fill_none(getattr(cj, "hadronFlavour", ak.zeros_like(cj.pt)), 0)
        self._flav = ak.to_numpy(flav).astype(np.int64)

    def _eval(self, syst):
        n = len(self._flav)
        sf = np.ones(n, dtype=np.float64)
        cat = _category_np(self._cvsl, self._cvsb)
        wp = np.where(cat >= 0, CAT_TO_WP[np.clip(cat, 0, 10)], -1)
        ok = (wp >= 0) & np.isfinite(self._pt)
        if ok.any():
            sf[ok] = self._corr.evaluate(
                syst,
                self._flav[ok],
                wp[ok],
                np.zeros(ok.sum(), dtype=np.float64),
                np.clip(self._pt[ok], *PT_BINS),
            )
        return sf

    def add_weights(self) -> None:
        sf_c = self._eval("central")
        if self._shift is None:
            sf_up = self._eval(SYST_UP)
            sf_dn = self._eval(SYST_DN)
            r_up = np.where(sf_c != 0, sf_up / sf_c, 1.0)
            r_dn = np.where(sf_c != 0, sf_dn / sf_c, 1.0)
            self._weights.add(
                name=self._nuis,
                weight=sf_c,
                weightUp=sf_c * r_up,
                weightDown=sf_c * r_dn,
            )
        else:
            self._weights.add(name=self._nuis, weight=sf_c)
