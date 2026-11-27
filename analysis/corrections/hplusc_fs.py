"""H+c flavour-scheme (3FS vs 4FS FxFx) SHAPE systematic, 2026-11-04.

Per-event weight on the CENTRAL 4FS FxFx H+c signal, binned in the leading GEN c-hadron pT (T = 5 GeV, Run 3 combined),
delivered as correctionlib `hplusc_fs_shape(syst, gen_chad1_pt)` (copy of
/eos/user/c/cgupta/HToWW/fs_unc/fs_shape/hplusc_fs_shape_correctionlib.json, see analysis/data/fs_shape/README_weights.md):
    nominal = 1,  up = sigma3(bin)/sigma3,incl / (sigma4(bin)/sigma4,incl)  ("3FS-like"),  down = 2 - up (mirror).
The inclusive cross section is unchanged by construction; the weights are a SHAPE-only scheme difference
(NOT the NLO-vs-NNLO normalisation, LHCHWG-2026-007 Table 3). Events with no c-hadron above 5 GeV (gen_chad1_pt = -1 or < 5)
fall in the first bin (w0 = 0.776 up / 1.224 down).

gen_chad1_pt reproduces fs_unc/gen/derive_chad.py exactly: GenPart status 2, open charm (|pdgId| 4xx, 4xxx, 104xx, 204xx;
charmonium 44x excluded), WEAKLY decaying (no c-hadron daughter, so D* -> D counts once), not from a b-hadron
(immediate mother), |eta| < 2.5, leading pT; -1 if none. This is NOT the GenJet c-jet pT of select_gen_fs / gen_cjet1_pt.

Output Weights name: `xsec_hplusc_3FS_4FS` (parquet columns weight_xsec_hplusc_3FS_4FSUp/Down); the card builder falls back to
the nominal weight for processes without the column.
"""
from pathlib import Path

import awkward as ak
import numpy as np

SYST_NAME = "xsec_hplusc_3FS_4FS"
CORRECTION_NAME = "hplusc_fs_shape"
CORRECTION_FILE = Path(__file__).resolve().parent.parent / "data" / "fs_shape" / "hplusc_fs_shape_correctionlib.json"
CHAD_ETA_MAX = 2.5


def _is_chad(apid):
    openc = (
        ((apid > 400) & (apid < 500))
        | ((apid > 4000) & (apid < 5000))
        | ((apid > 10400) & (apid < 10500))
        | ((apid > 20400) & (apid < 20500))
    )
    return openc & (((apid // 10) % 100) != 44)  # exclude charmonium (44x)


def _is_bhad(apid):
    return ((apid > 500) & (apid < 600)) | ((apid > 5000) & (apid < 6000))


def gen_chad1_pt(events):
    """Leading weakly-decaying open-charm GEN hadron pT (not from b, |eta|<2.5), -1 if none. numpy float64 (n_events,)."""
    gp = events.GenPart
    apid = abs(gp.pdgId)
    mom = gp.genPartIdxMother
    is_c = _is_chad(apid)
    cand = is_c & (gp.status == 2) & (abs(gp.eta) < CHAD_ETA_MAX)
    # weakly decaying: no c-hadron (of any status) has this particle as its mother
    idx = ak.local_index(apid)
    c_moms = mom[is_c]
    has_c_daughter = ak.any(idx[cand][:, :, None] == c_moms[:, None, :], axis=-1)
    mom_pid = ak.fill_none(apid[ak.mask(mom, mom >= 0)], 0)
    from_b = _is_bhad(mom_pid)[cand]
    pts = gp.pt[cand][(~has_c_daughter) & (~from_b)]
    return ak.to_numpy(ak.fill_none(ak.max(pts, axis=1), -1.0)).astype(np.float64)


_cset = None


def _correction():
    global _cset
    if _cset is None:
        import correctionlib

        _cset = correctionlib.CorrectionSet.from_file(str(CORRECTION_FILE))[CORRECTION_NAME]
    return _cset


def fs_shape_weights(chad1_pt):
    """(up, down) per-event weights for an array of leading c-hadron pT."""
    corr = _correction()
    pt = np.asarray(chad1_pt, dtype=np.float64)
    return corr.evaluate("up", pt), corr.evaluate("down", pt)


def add_hplusc_fs_weight(events, weights_container, shift, dataset, datasets=None):
    """Add the 3FS/4FS shape nuisance to the datasets listed in `datasets` (prefix match on <sample>_<partition>).
    MC only; nominal-tree only (nominal weight is 1, so object-shift trees need nothing)."""
    if shift is not None or not datasets:
        return
    if not any(dataset == d or dataset.startswith(d + "_") for d in datasets):
        return
    if "GenPart" not in events.fields:
        raise RuntimeError(f"{dataset}: {SYST_NAME} needs GenPart")
    up, down = fs_shape_weights(gen_chad1_pt(events))
    if not (np.all(np.isfinite(up)) and np.all(np.isfinite(down))):
        raise RuntimeError(f"{dataset}: non-finite {SYST_NAME} weights")
    weights_container.add(name=SYST_NAME, weight=np.ones(len(events)), weightUp=up, weightDown=down)
