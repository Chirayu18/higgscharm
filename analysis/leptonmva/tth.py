"""
Retrained Run-3 ttH lepton MVA (mvaTTH, TMVA BDTG, 500 trees), evaluated with ONNX Runtime.

Port of Thomas Van Laer's tvl/MVA lepton-MVA working points (2026-08/09) onto hww-analysis, following the
2026-10-08 framework plan section 4.4 (Projects/HToWW/framework/optimization/2026-10-08-framework-optimization-plan.md):

* Features are built fully vectorised for all leptons of a chunk at once (no per-lepton python loop).
* The BDT is evaluated with ONNX graphs converted from the TMVA XMLs with the TMVA split convention
  (TMVA DecisionTreeNode::GoesRight is `x >= cut`; graph mode BRANCH_LT). Thomas's dict evaluator and the
  August-2026 ONNX files used `x > cut` and disagree with TMVA for ~3.3% of electrons (NanoAOD mvaIso sits
  exactly on split values), by up to 0.032. Validated 2026-11-05: max |ONNX - TMVA-exact| = 3.3e-7 on 50k real
  2022postEE tt leptons, all tie-sensitive leptons included (tests/test_allfix.py::TestTthMVA).
* No cache: scoring costs ~9 us/lepton, far cheaper than Thomas's parquet cache (plan section 4.2).
* Scores are computed ONCE per chunk on NanoAOD-level inputs (uncorrected pt), as TMVA was trained, BEFORE the
  object corrections, and attached as Muon.tthMVA / Electron.tthMVA so every object-shift pass reuses them.
  (Thomas computed them after muon scale/smearing; the training used NanoAOD pt.)
* log|dxy| / log|dz| = -inf for an exactly-zero impact parameter is passed through unchanged, as TMVA::Reader
  would see it (-inf >= cut is False -> left branch, identical in ONNX).

The same 2022EE-trained model is applied to all of 2022preEE, 2022postEE, 2023preBPix and 2023postBPix
(Thomas's choice); other years get no field and the working points raise.
"""
import pyarrow  # noqa: F401  -- must be imported BEFORE onnxruntime in the coffea image (GLIBCXX clash otherwise)
from pathlib import Path

import numpy as np
import awkward as ak

TTHMVA_YEARS = ("2022preEE", "2022postEE", "2023preBPix", "2023postBPix")
MODEL_DIR = Path(__file__).resolve().parents[1] / "data" / "tthMVA_2022-2023_retrained"
MODEL_FILES = {
    "Muon": MODEL_DIR / "muon_mvaTTH_2022EE.onnx",
    "Electron": MODEL_DIR / "electron_mvaTTH_2022EE.onnx",
}
# input order is fixed by the TMVA XML (swapping two columns gives plausible garbage, silently)
FEATURE_NAMES = [
    "pt", "eta", "pfRelIso03_all", "miniPFRelIso_chg", "miniPFRelIso_all-miniPFRelIso_chg",
    "jetNDauCharged", "jetPtRelv2", "jetIdx>-1?Jet_btagDeepFlavB[jetIdx]:0", "min(1/(1+jetRelIso),1.5)",
    "sip3d", "log(abs(dxy))", "log(abs(dz))", "segmentComp|mvaIso",
]

_SESSIONS = {}


def _session(coll):
    if coll not in _SESSIONS:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        _SESSIONS[coll] = ort.InferenceSession(
            str(MODEL_FILES[coll]), so, providers=["CPUExecutionProvider"]
        )
    return _SESSIONS[coll]


def lepton_features(events, coll):
    """All leptons of `coll` ("Muon"/"Electron") at once -> ((n_leptons, 13) float64, counts per event)."""
    lep = events[coll]
    counts = ak.to_numpy(ak.num(lep))

    def g(name):
        return ak.to_numpy(ak.flatten(lep[name])).astype(np.float64)

    # jetIdx > -1 ? Jet_btagDeepFlavB[jetIdx] : 0, via a global index into the flattened Jet collection
    jidx = ak.to_numpy(ak.flatten(lep.jetIdx)).astype(np.int64)
    njet = ak.to_numpy(ak.num(events.Jet))
    joff = np.concatenate([[0], np.cumsum(njet)[:-1]]).astype(np.int64)
    evt_of_lep = np.repeat(np.arange(len(counts)), counts)
    jb = ak.to_numpy(ak.flatten(events.Jet.btagDeepFlavB)).astype(np.float64)
    ok = jidx >= 0
    btag = np.zeros(len(jidx))
    btag[ok] = jb[joff[evt_of_lep[ok]] + jidx[ok]]
    last = g("segmentComp") if coll == "Muon" else g("mvaIso")
    mini_chg = g("miniPFRelIso_chg")
    with np.errstate(divide="ignore"):
        X = np.column_stack(
            [
                g("pt"), g("eta"), g("pfRelIso03_all"), mini_chg, g("miniPFRelIso_all") - mini_chg,
                g("jetNDauCharged"), g("jetPtRelv2"), btag,
                np.minimum(1.0 / (1.0 + g("jetRelIso")), 1.5),
                g("sip3d"), np.log(np.abs(g("dxy"))), np.log(np.abs(g("dz"))), last,
            ]
        )
    return X, counts


def score_features(X, coll):
    """(n, 13) features -> (n,) float32 mvaTTH score in (-1, 1)."""
    if len(X) == 0:
        return np.zeros(0, dtype=np.float32)
    sess = _session(coll)
    name = sess.get_inputs()[0].name
    out = sess.run(None, {name: np.ascontiguousarray(X, dtype=np.float32)})[0]
    return np.asarray(out, dtype=np.float32).reshape(-1)


def add_tthmva_scores(events, year):
    """Attach Muon.tthMVA and Electron.tthMVA (NanoAOD-level inputs). No-op outside TTHMVA_YEARS."""
    if year not in TTHMVA_YEARS:
        return events
    for coll in ("Muon", "Electron"):
        X, counts = lepton_features(events, coll)
        events[coll, "tthMVA"] = ak.unflatten(score_features(X, coll), counts)
    return events
