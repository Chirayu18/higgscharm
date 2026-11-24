"""negrw validation statistics, corrected 2026-11-12 (findings F1/F2 of the v2 acceptance review).

The model is held fixed: these test the realised model on independent (held-out or SR) events.

ratio_and_err   R = sum|w|g / sum w with the delta-method sigma sqrt(sum((|w|g - R w)^2)) / |sum w|, which
                includes the numerator/denominator covariance (same events). The 2026-11-12 training output
                quoted sqrt(sum w^2)/|sum w|, the denominator only.
shape_cov       shape residual after the approved per-sample renormalisation k = sum w / sum|w|g, with the
                fitted-normalisation, shared-event covariance c_ib = (k a_i - w_i)(1[i in b] - f_b),
                f_b = A_b / A, summed over independent groups (eras). Port of the reviewer's
                resume_shape_cov.py (~/negrw_review_20261112, sha256 4dcea11e...), toy-validated in
                tests/test_negrw_stats.py. chi2 = r^T C^+ r on the non-null eigenspace (ndf = rank).
"""
import numpy as np


def ratio_and_err(w, g):
    w = np.asarray(w, float); a = np.abs(w) * np.asarray(g, float); s = w.sum()
    if s == 0:
        return float('nan'), float('nan')
    r = a.sum() / s
    return float(r), float(np.sqrt(np.sum((a - r * w) ** 2)) / abs(s))


def binned_ratio_err(idx, w, g, nb):
    """Per-bin R_b = sum_b |w|g / sum_b w and its delta-method sigma (NaN where sum_b w == 0)."""
    w = np.asarray(w, float); a = np.abs(w) * np.asarray(g, float)
    h = lambda v: np.bincount(idx, weights=v, minlength=nb)
    nw, na = h(w), h(a)
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.where(nw != 0, na / nw, np.nan)
        err = np.sqrt(h((a - np.nan_to_num(r)[idx] * w) ** 2)) / np.abs(nw)
    return r, np.where(nw != 0, err, np.nan)


def pinv_chi2(r, C, rel=1e-9):
    lam, U = np.linalg.eigh(C); keep = lam > rel * lam.max()
    y = U[:, keep].T @ r
    return float((y ** 2 / lam[keep]).sum()), int(keep.sum())


def shape_cov(idx, nb, w, g, groups, g_up=None, min_events=50):
    """Sum over independent groups (boolean masks) of the renormalised residual r = k A - N and its covariance.

    g_up (optional): the CMS_negrw_vjets Up g (clip(g + dg, -1, 1)); Delta = renormalised Up - nominal shift,
    added as one fully correlated, Gaussian-constrained shape parameter (Delta Delta^T).
    Returns None when no group has >= min_events events or fewer than 2 bins are populated.
    """
    w = np.asarray(w, float); a = np.abs(w); g = np.asarray(g, float)
    r = np.zeros(nb); C = np.zeros((nb, nb)); D = np.zeros(nb); N = np.zeros(nb); RW = np.zeros(nb); n = 0
    for m in groups:
        if m.sum() < min_events:
            continue
        n += int(m.sum())
        h = lambda v: np.bincount(idx[m], weights=v, minlength=nb)
        ai, wi, gi = a[m], w[m], g[m]
        Nb, Ab = h(wi), h(ai * gi); k = Nb.sum() / Ab.sum(); f = Ab / Ab.sum()
        u2 = h((k * ai * gi - wi) ** 2); T = u2.sum()
        C += np.diag(u2) - np.outer(f, u2) - np.outer(u2, f) + T * np.outer(f, f)
        r += k * Ab - Nb; N += Nb; RW += k * Ab
        if g_up is not None:
            Au = h(ai * np.asarray(g_up, float)[m]); D += (Nb.sum() / Au.sum()) * Au - k * Ab
    ok = np.diag(C) > 0
    if n == 0 or ok.sum() < 2:
        return None
    chi2, rank = pinv_chi2(r[ok], C[np.ix_(ok, ok)])
    out = dict(n=n, nb=int(ok.sum()), rank=rank, chi2=chi2, r=r, C=C, N=N, RW=RW, ok=ok)
    if g_up is not None:
        out['chi2_with_nuis'] = pinv_chi2(r[ok], C[np.ix_(ok, ok)] + np.outer(D[ok], D[ok]))[0]
        out['Delta'] = D
    return out
