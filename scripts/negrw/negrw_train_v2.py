#!/usr/bin/env python3
"""
negrw_train_v2.py -- retrain the negative-weight reweighting ensemble (arXiv:2510.16217)
on the hww_genrw_train_v2 parquets (current DY / jet-binned W+jets, 4 Run-3 eras pooled).

RECIPE = July 2026 production (scripts/negweight_reweight_train.py / ~/negrw_condor/negrw_diag.py):
  identical 20 FEATURES, 20x HistGradientBoostingClassifier(log_loss, max_iter 200, lr 0.05,
  max_depth 4, l2 1.0, early_stopping val 0.15), each member on a 60% subsample without
  replacement, seeds seed+m, label = weight_nominal > 0, unweighted fit.
ONLY CHANGE TO THE FIT: rows with event % 5 == 0 are HELD OUT (never seen by any member) and
used for closure / calibration / old-vs-new comparison. The July run evaluated closure on its
own training rows.

Bundle schema is unchanged ({"models","features","n_models","frac_pos"} + extra "meta"), so
analysis/processors/base.py::_score_negrw loads it as-is.

Run inside coffea-base-almalinux9:0.7.30-py3.10 (sklearn 1.7.2 = the processing workers').
"""
import argparse, json, os, sys, hashlib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score

FEATURES = [
    "lhe_njets", "lhe_nb", "lhe_nc", "lhe_nuds", "lhe_nglu", "lhe_npnlo",
    "lhe_ht", "lhe_htincoming", "lhe_vpt", "lhe_alphas",
    "genparton_multiplicity", "genparton_n_pt20", "genparton_n_pt100", "genparton_n_pt200",
    "genparton_incoming1_pdgId", "genparton_incoming2_pdgId",
    "genparton1_pt", "genparton1_eta", "genparton2_pt", "genparton2_eta",
]
N_MODELS = 20
SUBSAMPLE_FRAC = 0.6
HYPER = dict(loss="log_loss", max_iter=200, learning_rate=0.05, max_depth=4,
             l2_regularization=1.0, early_stopping=True, validation_fraction=0.15)
DATASETS = ["DYto2L_2Jets_10to50", "DYto2L_2Jets_50", "WtoLNu_2Jets_0J", "WtoLNu_2Jets_1J", "WtoLNu_2Jets_2J"]
ERAS = ["2022preEE", "2022postEE", "2023preBPix", "2023postBPix"]


def predict(models, X, chunk=500_000):
    """ensemble P+ mean and 2*std, chunked to bound memory."""
    pm = np.empty(len(X), dtype=np.float32); ps = np.empty(len(X), dtype=np.float32)
    for s in range(0, len(X), chunk):
        P = np.stack([m.predict_proba(X[s:s + chunk])[:, 1] for m in models], axis=0)
        pm[s:s + chunk] = P.mean(0); ps[s:s + chunk] = 2 * P.std(0)
    return pm, ps


def neff(w):
    s2 = np.sum(w ** 2)
    return float(np.sum(w) ** 2 / s2) if s2 > 0 else 0.0


def closure(w, g):
    """reweighted sum |w| g over nominal sum w, with N_eff before/after.

    ratio_staterr (corrected 2026-11-12, v2 acceptance review F1): delta-method sigma of R = sum|w|g / sum w
    on the same events with g fixed, sqrt(sum((|w|g - R w)^2)) / |sum w|, which includes the numerator/
    denominator covariance. The 2026-11-12 training output (negrw_v2_validation.json) quoted
    sqrt(sum w^2)/|sum w|, the denominator only; its ratio_staterr has no ratio_staterr_def key.
    """
    w = np.asarray(w, float); rw = np.abs(w) * np.asarray(g, float); s = w.sum()
    r = float(rw.sum() / s) if s != 0 else float("nan")
    return dict(n=int(len(w)), sumw=float(s), sumrw=float(rw.sum()), ratio=r,
                neff_nom=neff(w), neff_rw=neff(rw),
                ratio_staterr=float(np.sqrt(np.sum((rw - r * w) ** 2)) / abs(s)) if s != 0 else float("nan"),
                ratio_staterr_def="delta")


def binned(x, w, gs, edges):
    """per-bin sum w, sum |w| g (for each g in gs), and N_eff.

    <k>_ratio_var (added 2026-11-12): delta-method variance of the per-bin ratio R_b = sum_b|w|g / sum_b w,
    sum_b((|w|g - R_b w)^2) / (sum_b w)^2 (shared events). This is the raw, not renormalised, ratio; the
    shape test after the per-sample renormalisation is negrw_stats.shape_cov.
    """
    out = dict(edges=list(map(float, edges)))
    idx = np.digitize(np.nan_to_num(x, nan=-1e9), edges) - 1
    ok = (idx >= 0) & (idx < len(edges) - 1)
    def h(v): return np.bincount(idx[ok], weights=v[ok], minlength=len(edges) - 1)
    out["nom"] = h(w).tolist(); out["nom_var"] = h(w ** 2).tolist()
    for k, g in gs.items():
        rw = np.abs(w) * g
        out[k] = h(rw).tolist(); out[k + "_var"] = h(rw ** 2).tolist()
        nw, na = h(w), h(rw)
        with np.errstate(divide="ignore", invalid="ignore"):
            rb = np.where(nw != 0, na / nw, 0.0)
            out[k + "_ratio_var"] = np.where(nw != 0, h((rw - rb[np.clip(idx, 0, len(rb) - 1)] * w) ** 2) / nw ** 2, np.nan).tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="slim training parquet (from negrw_slim_v2.py)")
    ap.add_argument("--sr", default=None, help="slim allfix SR parquet (gen features + weight_negrw)")
    ap.add_argument("--old-model", default=None)
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout-mod", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    import sklearn, joblib
    print("sklearn", sklearn.__version__, flush=True)

    df = pd.read_parquet(args.train)
    hold = (df["event"].to_numpy() % args.holdout_mod) == 0
    X = df[FEATURES].to_numpy(dtype=np.float32)
    w = df["weight_nominal"].to_numpy(dtype=np.float64)
    y = (w > 0).astype(np.int8)
    era = df["era"].to_numpy(); ds = df["dataset"].to_numpy()
    del df
    tr = np.flatnonzero(~hold); te = np.flatnonzero(hold)
    n = len(tr); fpos = float(y[tr].mean())
    print(f"rows {len(y)}  train {n}  holdout {len(te)}  frac+ train {fpos:.4f} holdout {y[te].mean():.4f}", flush=True)

    rng = np.random.default_rng(args.seed)
    models, n_iters, curves = [], [], []
    for m in range(N_MODELS):
        idx = tr[rng.choice(n, size=int(SUBSAMPLE_FRAC * n), replace=False)]
        clf = HistGradientBoostingClassifier(random_state=args.seed + m, **HYPER)
        clf.fit(X[idx], y[idx])
        models.append(clf); n_iters.append(int(clf.n_iter_))
        curves.append(dict(train=(-np.asarray(clf.train_score_)).tolist(),
                           val=(-np.asarray(clf.validation_score_)).tolist() if clf.validation_score_ is not None else []))
        print(f"  model {m + 1}/{N_MODELS} n_iter={clf.n_iter_}", flush=True)
        del idx

    bundle = {"models": models, "features": FEATURES, "n_models": N_MODELS, "frac_pos": fpos,
              "meta": dict(workflow="hww_genrw_train_v2", eras=ERAS, datasets=DATASETS,
                           category="train (veto_emu_sr, allfix lepton definitions)",
                           holdout=f"event % {args.holdout_mod} == 0 excluded from training",
                           n_train_rows=int(n), hyperparams=HYPER, subsample_frac=SUBSAMPLE_FRAC,
                           sklearn=sklearn.__version__, date="2026-11-12")}
    mpath = os.path.join(args.outdir, "negrw_models_v2_20261112.joblib")
    joblib.dump(bundle, mpath)
    print("saved", mpath, hashlib.sha256(open(mpath, "rb").read()).hexdigest(), flush=True)

    # ---------------- validation on held-out rows ----------------
    res = dict(n_rows=int(len(y)), n_train=int(n), n_holdout=int(len(te)), frac_pos_train=fpos,
               n_iters=n_iters, hyperparams=HYPER, sklearn=sklearn.__version__)
    Xt, yt, wt, et, dt = X[te], y[te], w[te], era[te], ds[te]
    pn, sn = predict(models, Xt)
    gn = 2 * pn - 1
    # in-sample comparison on a train subsample (overfitting check)
    sub = tr[np.random.default_rng(7).choice(n, size=min(len(te), n), replace=False)]
    ptr, _ = predict(models, X[sub])
    res["new"] = dict(logloss_holdout=float(log_loss(yt, pn)), auc_holdout=float(roc_auc_score(yt, pn)),
                      logloss_train=float(log_loss(y[sub], ptr)), auc_train=float(roc_auc_score(y[sub], ptr)),
                      g_mean=float(gn.mean()), g_min=float(gn.min()), g_max=float(gn.max()),
                      dg_mean=float(sn.mean()), dg_max=float(sn.max()), dg_p99=float(np.percentile(sn, 99)))
    del X
    gs = {"rw_new": gn}
    old = None
    if args.old_model:
        old = joblib.load(args.old_model)
        assert old["features"] == FEATURES, "old model feature list differs"
        po, so = predict(old["models"], Xt)
        go = 2 * po - 1
        gs["rw_old"] = go
        res["old"] = dict(logloss_holdout=float(log_loss(yt, po)), auc_holdout=float(roc_auc_score(yt, po)),
                          g_mean=float(go.mean()), dg_mean=float(so.mean()),
                          corr_g_new_old=float(np.corrcoef(gn, go)[0, 1]),
                          mean_abs_dg_new_old=float(np.mean(np.abs(gn - go))))
    # calibration: observed frac+ in bins of predicted P+
    pe = np.linspace(0, 1, 21)
    cal = {}
    for k, p in [("new", pn)] + ([("old", po)] if old else []):
        b = np.clip(np.digitize(p, pe) - 1, 0, 19)
        cal[k] = dict(n=np.bincount(b, minlength=20).tolist(),
                      pred=(np.bincount(b, weights=p, minlength=20) / np.maximum(np.bincount(b, minlength=20), 1)).tolist(),
                      obs=(np.bincount(b, weights=yt, minlength=20) / np.maximum(np.bincount(b, minlength=20), 1)).tolist())
    res["calibration"] = dict(edges=pe.tolist(), **cal)
    # closure: total, per dataset, per era, per (era,dataset)
    clos = {"total": {k: closure(wt, g) for k, g in gs.items()}}
    for i, d in enumerate(DATASETS):
        m = dt == i
        if m.any(): clos[d] = {k: closure(wt[m], g[m]) for k, g in gs.items()}
    for i, e in enumerate(ERAS):
        m = et == i
        if m.any(): clos[e] = {k: closure(wt[m], g[m]) for k, g in gs.items()}
        for j, d in enumerate(DATASETS):
            mm = m & (dt == j)
            if mm.any(): clos[f"{e}/{d}"] = {k: closure(wt[mm], g[mm]) for k, g in gs.items()}
    res["closure_holdout"] = clos
    fi = {f: i for i, f in enumerate(FEATURES)}
    edges = {"lhe_vpt": np.linspace(0, 400, 21), "lhe_ht": np.array([0, 1, 30, 60, 100, 150, 200, 300, 400, 600, 1000, 4000.]),
             "lhe_njets": np.arange(-0.5, 4.5, 1.0), "lhe_npnlo": np.arange(-0.5, 3.5, 1.0),
             "genparton1_pt": np.array([0, 20, 40, 60, 80, 100, 150, 200, 300, 500, 2000.]),
             "lhe_nc": np.arange(-0.5, 4.5, 1.0)}
    res["binned_holdout"] = {f: binned(Xt[:, fi[f]], wt, gs, e) for f, e in edges.items()}
    # binned per W jet bin / DY sample for lhe_vpt
    res["binned_holdout_vpt_by_dataset"] = {d: binned(Xt[dt == i, fi["lhe_vpt"]], wt[dt == i], {k: g[dt == i] for k, g in gs.items()}, edges["lhe_vpt"])
                                            for i, d in enumerate(DATASETS) if (dt == i).any()}
    res["hist_g"] = {k: np.histogram(g, bins=np.linspace(-1, 1, 81))[0].tolist() for k, g in gs.items()}
    res["hist_dg_new"] = np.histogram(sn, bins=np.linspace(0, 0.2, 81))[0].tolist()

    # ---------------- allfix SR (2026-11-10 v1 TRAIN parquets) ----------------
    if args.sr:
        sr = pd.read_parquet(args.sr)
        Xs = sr[FEATURES].to_numpy(dtype=np.float32)
        ws = sr["weight_nominal"].to_numpy(float); ds_s = sr["dataset"].to_numpy()
        ps, ss = predict(models, Xs); g_sr = 2 * ps - 1
        g_old_col = sr["weight_negrw"].to_numpy(float)
        gsr = {"rw_new": g_sr, "rw_old_column": g_old_col}
        if old:
            po_s, _ = predict(old["models"], Xs); gsr["rw_old"] = 2 * po_s - 1
            res.setdefault("sr", {})["old_model_reproduces_column_maxabs"] = float(np.nanmax(np.abs(gsr["rw_old"] - g_old_col)))
        sres = res.setdefault("sr", {})
        sres["n"] = int(len(sr)); sres["frac_pos"] = float((ws > 0).mean())
        sres["closure"] = {"total": {k: closure(ws, g) for k, g in gsr.items()}}
        for i, d in enumerate(DATASETS):
            m = ds_s == i
            if m.any(): sres["closure"][d] = {k: closure(ws[m], g[m]) for k, g in gsr.items()}
        sres["g_new_mean"] = float(g_sr.mean()); sres["dg_new_mean"] = float(ss.mean()); sres["dg_new_max"] = float(ss.max())
        sres["corr_g_new_old"] = float(np.corrcoef(g_sr, g_old_col)[0, 1])
        sres["binned_vpt"] = binned(Xs[:, fi["lhe_vpt"]], ws, gsr, edges["lhe_vpt"])
        sres["hist_g"] = {k: np.histogram(g, bins=np.linspace(-1, 1, 81))[0].tolist() for k, g in gsr.items()}
        # coverage: fraction of SR events above the TRAINING-sample p99.9 / max, per feature
        trd = pd.read_parquet(args.train, columns=FEATURES)
        cov = {}
        for f in FEATURES:
            t = trd[f].to_numpy(float); t = t[np.isfinite(t)]
            s = sr[f].to_numpy(float); s = s[np.isfinite(s)]
            if len(t) == 0 or len(s) == 0: continue
            q = np.percentile(t, [0.1, 99.9])
            cov[f] = dict(train_min=float(t.min()), train_max=float(t.max()), train_p999=float(q[1]),
                          sr_min=float(s.min()), sr_max=float(s.max()),
                          sr_frac_above_p999=float((s > q[1]).mean()), sr_frac_outside_minmax=float(((s > t.max()) | (s < t.min())).mean()))
        sres["coverage"] = cov
        del trd

    json.dump(res, open(os.path.join(args.outdir, "negrw_v2_validation.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(args.outdir, "negrw_v2_curves.npz"),
                        curves=np.array([json.dumps(c) for c in curves]))
    print(json.dumps({k: res[k] for k in ("new", "old") if k in res}, indent=1), flush=True)
    print("TOTAL holdout closure:", json.dumps(clos["total"], indent=1), flush=True)
    if args.sr:
        print("SR closure:", json.dumps(res["sr"]["closure"]["total"], indent=1), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
