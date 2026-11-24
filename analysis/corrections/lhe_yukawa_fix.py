"""Per-event correction of LHE scale weights in the central H+c / H+b FxFx samples (2026-10-13).

The Run 3 gridpacks (MG5_aMC 2.9.9; HPlusCharm-4FS / HPlusBottom_5FS ...MuRFScaleDynX0p50... amcatnloFXFX) carry a
custom `<sample>_mg5amc.patch` that turns on a running quark Yukawa, F(muR) = (m_q(muR) / m_q,in)^2, through
`rwgt_muR_dep_fac` (H+c: ymc = 0.986 GeV at 3 GeV, nf = 4; H+b: ymb = 4.18 GeV at m_b, nf = 5). The nominal event
weight carries F (sigma 0.298 pb vs 0.264 pb for the clean Run 2 UL18 sample at 13 TeV), but the stored
LHEScaleWeight / LHEPdfWeight variations do not: they are F^-1 ~ 2x too large, by an event-dependent amount.
The (1,1) scale entry and PDF member 0 are copies of the nominal.

Correction. The PDF members are all evaluated at the central scale mu0, so per event
    r = mean(LHEPdfWeight[1:101]) / LHEPdfWeight[0] = 1 / F(mu0)      (up to the replica-mean offset, ~1e-3)
F(mu0) is inverted to mu0 with the gridpack's own running, then each scale weight gets the factor it should have:
    w(muR = k mu0, muF) -> w_stored * F(k mu0; mu0)
with F(mu0; mu0) = 1/r for k = 1 (muF-only entries: exact, no running model involved).

F is tabulated by `analysis/data/lhe_yukawa/tab.f`: the patched rwgt_muR_dep_fac with the gridpack's own runalpha /
runmass (compiled from MG5_aMC v2.9.10 reweight_xsec.f; the patch applies with the same offset as in the
gridpack build log), the gridpack's param card (MZ, aEWM1, Gf, y_q) and alpha_s(MZ) from the PDF set via LHAPDF
(325500 nf4: 0.11266; 325300: 0.11800). See `analysis/data/lhe_yukawa/build.sh`.

The tables start at mu0 = 1 GeV: MG clamps the renormalisation scale at 1 GeV, and ~0.8% of the 1-jet H-events sit
exactly there (constant ratios muR0.5 = 6.199, muR2 = 0.308 in clean samples; reproduced exactly here).

Validated: muF-only entries reproduce the clean Run 2 sample to < 1% (0.94 / 1.02); the fix is refused on samples
without the offset (median r < 1.2), where it would be wrong.
"""
import os
from functools import lru_cache

import numpy as np

_DATA = os.path.join(os.path.dirname(__file__), "..", "data", "lhe_yukawa")
_TABLES = {
    "hpluscharm_4fs": "hpluscharm_4fs_table.txt",
    "hplusbottom_5fs": "hplusbottom_5fs_table.txt",
}
# workflow dataset name (prefix; jobs run as <name>_<partition>) -> table
DATASET_TABLE = {
    "HplusCharm_4FS_HToWWTo2L2Nu": "hpluscharm_4fs",
    "HplusBottom_5FS_HToWWTo2L2Nu": "hplusbottom_5fs",
}
# NanoAOD LHEScaleWeight order: [0] (R0.5,F0.5) [1] (R0.5,F1) [2] (R0.5,F2) [3] (R1,F0.5) [4] (1,1) [5] (R1,F2)
# [6] (R2,F0.5) [7] (R2,F1) [8] (R2,F2)
_KR_COLUMN = [1, 1, 1, 0, 0, 0, 2, 2, 2]   # 0: F(mu0;mu0), 1: F(0.5 mu0; mu0), 2: F(2 mu0; mu0)
_FLOOR_TOL = 0.005                          # relative tolerance on F(mu0) above F(1 GeV), see corrected_scale_weights


def table_for_dataset(dataset):
    for name, table in DATASET_TABLE.items():
        if dataset == name or dataset.startswith(name + "_"):
            return table
    return None


@lru_cache(maxsize=None)
def _load(table):
    arr = np.loadtxt(os.path.join(_DATA, _TABLES[table]))
    lnmu, fc, fh, f2 = np.log(arr[:, 0]), arr[:, 1], arr[:, 2], arr[:, 3]
    if not np.all(np.diff(fc) < 0):
        raise RuntimeError(f"{table}: F(mu0;mu0) not monotonic, cannot invert")
    return lnmu, fc, fh, f2


def corrected_scale_weights(lhe_scale, lhe_pdf, table):
    """lhe_scale (n, 9), lhe_pdf (n, >=101) -> (corrected (n, 9), info). Entry 4 returned as 1 (nominal)."""
    sc = np.asarray(lhe_scale, dtype=np.float64)
    pdf = np.asarray(lhe_pdf, dtype=np.float64)
    if sc.ndim != 2 or sc.shape[1] != 9 or pdf.ndim != 2 or pdf.shape[1] < 101:
        raise ValueError(f"need LHEScaleWeight (n,9) and LHEPdfWeight (n,>=101), got {sc.shape} / {pdf.shape}")
    lnmu, fc, fh, f2 = _load(table)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = pdf[:, 1:101].mean(axis=1) / pdf[:, 0]
        f0 = 1.0 / r
    # the table starts at mu0 = 1 GeV, MG's scale floor ("spike" events sit exactly there); allow the replica-mean
    # noise (~1e-3) to push F(mu0) slightly above the table edge and treat those events as mu0 = 1 GeV
    ok = np.isfinite(f0) & (r > 0) & (f0 <= fc[0] * (1 + _FLOOR_TOL)) & (f0 >= fc[-1])
    # guard: only for samples that actually carry the offset (clean samples have r ~ 1, where this would be wrong)
    if ok.sum() > 0 and np.median(r[ok]) < 1.2:
        raise RuntimeError(f"lhe_yukawa_fix: median replicas/member0 = {np.median(r[ok]):.3f}; "
                           "this sample does not show the running-Yukawa offset, do not apply the fix")
    f0s = np.where(ok, np.minimum(f0, fc[0]), fc[0])
    lnmu0 = np.interp(f0s, fc[::-1], lnmu[::-1])          # fc decreasing in mu
    fk = (np.where(ok, f0, fc[0]), np.interp(lnmu0, lnmu, fh), np.interp(lnmu0, lnmu, f2))
    # events with F(mu0) below the table (mu0 > 14 TeV; ~4e-5 of events, pieces at different scales that partly cancel):
    # apply the measured factor only (exact for muF-only entries; drops the mu0 -> k*mu0 running for muR entries).
    # F(mu0) above F(1 GeV) (below MG's scale floor) only happens through near-total cancellations, where 1/r is huge
    # (seen up to 62) and multiplying by it blows the variation up; those, like r <= 0, get no variation (ratio 1).
    partial = ~ok & np.isfinite(f0) & (r > 0) & (f0 < fc[-1])
    out = np.ones_like(sc)
    for i in range(9):
        if i != 4:
            out[:, i] = np.where(ok, sc[:, i] * fk[_KR_COLUMN[i]], np.where(partial, sc[:, i] * np.where(partial, f0, 1.0), 1.0))
    info = {"n": len(r), "n_offtable": int(np.sum(partial)), "n_fallback": int(np.sum(~ok & ~partial)),
            "mu0_median": float(np.exp(np.median(lnmu0[ok]))) if ok.any() else float("nan")}
    return out, info
