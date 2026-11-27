"""Pure-numpy PNet 2D c-tag category assignment (11 categories), dependency free.

Single source of truth shared by
  * analysis/corrections/ctag2d.py       (CMS_ctag2d_* scale factors, per-event weights)
  * analysis/postprocess/inference.py    (cjet_cand_ctag2d_* one-hot MVA inputs; P0-7, 2026-11-04)
and identical to b-hive scripts/append_onehot.py / apply_ctag2d_sf.py. Kept free of coffea/correctionlib/torch
so the inference environment can import it.
"""
import numpy as np

# 2D category edges (mirror the analysis MVA one-hot binning)
X_HFVLF_EDGES = [0.0, 0.250, 0.452, 0.808, 1.000]
Y_BVC_EDGES = [0.0, 0.006, 0.017, 0.055, 0.761, 0.944, 0.985, 0.995, 1.0]
CATS = ["L0", "C0", "C1", "C2", "C3", "C4", "B0", "B1", "B2", "B3", "B4"]
CID = {n: i for i, n in enumerate(CATS)}
ONEHOT_PREFIX = "cjet_cand_ctag2d_"
ONEHOT_COLUMNS = [ONEHOT_PREFIX + c for c in CATS]


def category_np(cvsl, cvsb):
    """cvsl/cvsb (PNet CvsL / CvsB) -> integer category index into CATS (-1 if undefined)."""
    cvsl = np.asarray(cvsl, dtype=np.float64)
    cvsb = np.asarray(cvsb, dtype=np.float64)
    den = cvsl + cvsb * (1.0 - cvsl)
    with np.errstate(invalid="ignore", divide="ignore"):
        x = np.where(den != 0, cvsl / den, np.nan)
    y = 1.0 - cvsb
    good = np.isfinite(x) & np.isfinite(y)
    cat = np.full(x.shape, -1, dtype=np.int64)
    xe, ye = X_HFVLF_EDGES, Y_BVC_EDGES
    left = good & (x < xe[3])
    right = good & (x >= xe[3])

    def put(m, n):
        cat[m] = CID[n]

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


def onehot_columns(cvsl, cvsb):
    """dict {cjet_cand_ctag2d_<CAT>: int8 array}; all-zero rows where the category is undefined."""
    cat = category_np(cvsl, cvsb)
    return {ONEHOT_PREFIX + n: (cat == i).astype(np.int8) for i, n in enumerate(CATS)}
