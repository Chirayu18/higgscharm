#!/usr/bin/env python3
"""negrw_v2_corrected_report.py -- versioned, corrected validation report for the v2 negrw model (2026-11-12).

No training and no model change. Re-evaluates, on the SAME held-out rows (event % 5 == 0 of the G1-frozen
train_v2.parquet) and the allfix SR reference, the statistics found wrong in the v2 acceptance review:
  F1  closure_holdout / sr.closure: ratio_staterr = delta-method sigma of R (numerator/denominator covariance);
      the superseded denominator-only value is kept as ratio_staterr_denominator_only_superseded.
  F1  binned_* : per-bin delta-method ratio variance <k>_ratio_var (raw ratio, not renormalised).
  F2  shape_after_renorm: residual after the approved per-(era, dataset) renormalisation with the fitted-k,
      shared-event covariance (negrw_stats.shape_cov; reviewer resume_shape_cov.py method), +CMS_negrw_vjets.
Every other section is copied verbatim from the original negrw_v2_validation.json (not affected).
Inputs are pinned by sha256. The output directory must not exist; original artifacts are never touched.

usage: negrw_v2_corrected_report.py --outdir NEW_DIR [--scores-cache NPZ] [--reviewer-shape-json JSON]
Run inside coffea-base-almalinux9:0.7.30-py3.10 (sklearn 1.7.2, as for training)."""
import argparse, hashlib, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
sys.path.insert(0, str(Path(__file__).resolve().parent))
from negrw_train_v2 import FEATURES, DATASETS, ERAS, closure, binned, predict  # noqa: E402
from negrw_stats import shape_cov  # noqa: E402

EOS = "/eos/user/c/cgupta/HToWW/b-hive/negrw_out_img"
FROZEN = EOS + "/v2_20261112/frozen_inputs"
PINNED = {  # G1 identities (negrw_models_v2_20261112.provenance.json)
    "train": (FROZEN + "/train_v2.parquet", "f603ef6de8f30dc873615eddadba63684ebd64fc0e0b7c830a8e19213e7cc412"),
    "sr": (FROZEN + "/sr_allfix_v1_20261110.parquet", "4e34a0fffada2b7ee3953b8606bd41c591c0b0a167ce096ad920b29f3692ff98"),
    "model_new": (EOS + "/negrw_models_v2_20261112.joblib", "af53567ccfeddeef94c1fa62b3a6ed3c1285937919ced3f7705c565b819155cb"),
    "model_old": (EOS + "/negrw_models.joblib", "05567a8c0762b3446e8837c7825e30406637127a6143b331a968a42bcf707417"),
    "old_json": (EOS + "/v2_20261112/negrw_v2_validation.json", "e6918b6c4e84"),  # prefix as recorded in the review
}
INF = np.inf
SHAPE_EDGES = {"lhe_vpt": np.r_[-INF, np.linspace(0, 400, 21)[1:-1], 400, INF],
               "lhe_ht": np.array([-INF, 1, 30, 60, 100, 150, 200, 300, 400, 600, 1000, INF]),
               "genparton1_pt": np.array([-INF, 20, 40, 60, 80, 100, 150, 200, 300, 500, INF]),
               "lhe_njets": np.array([-INF, 0.5, 1.5, 2.5, 3.5, INF])}
PLOT_KEYS = ["lhe_vpt/all/DYto2L_2Jets_50", "lhe_vpt/njets_ge1/DYto2L_2Jets_50", "lhe_ht/njets_ge1/DYto2L_2Jets_50",
             "genparton1_pt/njets_ge1/DYto2L_2Jets_50", "lhe_vpt/all/WtoLNu_2Jets_1J", "lhe_ht/all/WtoLNu_2Jets_1J",
             "genparton1_pt/all/WtoLNu_2Jets_2J", "lhe_ht/all/WtoLNu_2Jets_0J", "lhe_njets/all/DYto2L_2Jets_10to50"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()


def corrected_closure(w, g):
    c = closure(w, g)
    s = float(np.sum(w))
    c["ratio_staterr_denominator_only_superseded"] = float(np.sqrt(np.sum(np.asarray(w, float) ** 2)) / abs(s)) if s else float("nan")
    c["signif_ratio_minus_1"] = (c["ratio"] - 1) / c["ratio_staterr"] if c["ratio_staterr"] > 0 else float("nan")
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--scores-cache", help="npz with new_g/new_sd/old_g/old_sd on the held-out rows (spot-checked)")
    ap.add_argument("--reviewer-shape-json", help="reviewer resume_shape_cov_out.json: chi2 must be reproduced")
    ap.add_argument("--spot", type=int, default=20000, help="rows re-scored to verify a score cache")
    a = ap.parse_args()
    out = Path(a.outdir)
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    ids = {}
    for k, (p, s) in PINNED.items():
        got = sha(p)
        if not got.startswith(s):
            sys.exit(f"{k}: {p} sha256 {got} != pinned {s}")
        ids[k] = dict(path=p, sha256=got)
    R = json.load(open(PINNED["old_json"][0]))
    bund = {"rw_new": joblib.load(PINNED["model_new"][0]), "rw_old": joblib.load(PINNED["model_old"][0])}
    assert all(b["features"] == FEATURES for b in bund.values())

    df = pd.read_parquet(PINNED["train"][0], columns=FEATURES + ["weight_nominal", "event", "era", "dataset",
                                                                  "lepton1_pdgId", "lepton2_pdgId"])
    df = df[(df["event"].to_numpy() % 5) == 0].reset_index(drop=True)
    X = df[FEATURES].to_numpy(np.float32); w = df["weight_nominal"].to_numpy(float)
    dt, et = df["dataset"].to_numpy(), df["era"].to_numpy()
    assert len(w) == R["n_holdout"], (len(w), R["n_holdout"])
    G = {}
    if a.scores_cache:
        z = np.load(a.scores_cache)
        assert int(z["n"]) == len(w) and abs(float(z["sumw"]) - w.sum()) < 1e-9 * abs(w.sum())
        G = {k: (z[c + "_g"].astype(float), z[c + "_sd"].astype(float)) for k, c in (("rw_new", "new"), ("rw_old", "old"))}
        pick = np.random.default_rng(11).choice(len(w), size=min(a.spot, len(w)), replace=False)
        spot = {}
        for k, b in bund.items():
            pm, ps = predict(b["models"], X[pick])
            spot[k] = dict(max_abs_dg=float(np.max(np.abs((2 * pm - 1) - G[k][0][pick]))),
                           max_abs_dsd=float(np.max(np.abs(ps - G[k][1][pick]))))
            assert spot[k]["max_abs_dg"] < 1e-5 and spot[k]["max_abs_dsd"] < 1e-5, spot
        ids["scores"] = dict(source="cache", path=a.scores_cache, sha256=sha(a.scores_cache), spot_check_rows=len(pick), spot=spot)
    else:
        for k, b in bund.items():
            pm, ps = predict(b["models"], X); G[k] = ((2 * pm - 1).astype(float), ps.astype(float))
        ids["scores"] = dict(source="computed")

    # F1: closure with the delta-method sigma (same grouping as the training script)
    gs = {k: v[0] for k, v in G.items()}
    clos = {"total": {k: corrected_closure(w, g) for k, g in gs.items()}}
    for i, d in enumerate(DATASETS):
        m = dt == i
        if m.any(): clos[d] = {k: corrected_closure(w[m], g[m]) for k, g in gs.items()}
    for i, e in enumerate(ERAS):
        m = et == i
        if m.any(): clos[e] = {k: corrected_closure(w[m], g[m]) for k, g in gs.items()}
        for j, d in enumerate(DATASETS):
            mm = m & (dt == j)
            if mm.any(): clos[f"{e}/{d}"] = {k: corrected_closure(w[mm], g[mm]) for k, g in gs.items()}
    # the recomputed ratios must be the author's (same rows, same models)
    dev = max(abs(clos[q][k]["ratio"] - R["closure_holdout"][q][k]["ratio"]) for q in clos for k in gs)
    assert dev < 1e-6, f"ratios differ from the original validation JSON by {dev}"
    R["closure_holdout"] = clos
    fi = {f: i for i, f in enumerate(FEATURES)}
    edges = {f: np.array(B["edges"]) for f, B in R["binned_holdout"].items()}
    R["binned_holdout"] = {f: binned(X[:, fi[f]], w, gs, e) for f, e in edges.items()}
    R["binned_holdout_vpt_by_dataset"] = {d: binned(X[dt == i, fi["lhe_vpt"]], w[dt == i], {k: g[dt == i] for k, g in gs.items()},
                                                    edges["lhe_vpt"]) for i, d in enumerate(DATASETS) if (dt == i).any()}

    # F2: shape after per-(era, dataset) renormalisation, fitted-k shared-event covariance, + CMS_negrw_vjets
    l1, l2 = np.abs(df["lepton1_pdgId"].to_numpy()), np.abs(df["lepton2_pdgId"].to_numpy())
    sub = {"all": np.ones(len(w), bool), "njets_ge1": df["lhe_njets"].to_numpy() >= 1, "sf_pair": (l1 > 0) & (l1 == l2)}
    shape, ref_dev = {}, 0.0
    ref = json.load(open(a.reviewer_shape_json))["res"] if a.reviewer_shape_json else None
    for v, ed in SHAPE_EDGES.items():
        x = df[v].to_numpy(float); idx = np.digitize(x, ed[1:-1]); idx[np.isnan(x)] = len(ed) - 1; nb = len(ed)
        for sname, smask in sub.items():
            for i, ds in enumerate(DATASETS):
                groups = [(dt == i) & (et == era) & smask for era in range(len(ERAS))]
                for mk, (g, sd) in G.items():
                    S = shape_cov(idx, nb, w, g, groups, g_up=np.clip(g + sd, -1, 1))
                    key = f"{v}/{sname}/{ds}/{mk}"
                    if S is None:
                        shape[key] = dict(skipped="fewer than 2 populated bins or no group with >= 50 events"); continue
                    N, RW, C = S["N"], S["RW"], S["C"]
                    with np.errstate(divide="ignore", invalid="ignore"):
                        rel = np.where(N != 0, S["r"] / N, np.nan); rsig = np.where(N != 0, np.sqrt(np.diag(C)) / np.abs(N), np.nan)
                        nu = np.where(RW != 0, S["Delta"] / RW, np.nan)
                        pull = np.where(S["ok"], S["r"] / np.sqrt(np.where(S["ok"], np.diag(C), 1)), 0)
                    shape[key] = dict(n=S["n"], nb=S["nb"], rank=S["rank"], chi2=S["chi2"], chi2_with_nuis=S["chi2_with_nuis"],
                                      max_abs_pull_marginal=float(np.abs(pull).max()), frac_N=(N / N.sum()).tolist(),
                                      rel_resid=rel.tolist(), rel_sigma=rsig.tolist(), rel_nuis_up=nu.tolist(),
                                      marginal_pull=pull.tolist(), plot=f"{v}/{sname}/{ds}" in PLOT_KEYS)
                    if ref is not None:
                        rk = ref.get(f"{v}/{sname}/{ds}/{mk[3:]}")
                        assert rk is not None and "chi2" in rk, key
                        ref_dev = max(ref_dev, abs(rk["chi2"] - S["chi2"]) / max(1., rk["chi2"]),
                                      abs(rk["chi2_with_nuis"] - S["chi2_with_nuis"]) / max(1., rk["chi2_with_nuis"]))
    if ref is not None:
        assert ref_dev < 1e-4, f"shape chi2 differs from the reviewer output by {ref_dev}"
    R["shape_after_renorm"] = shape
    R["shape_edges"] = {k: [float(x) for x in v] for k, v in SHAPE_EDGES.items()}

    # SR reference: closure with the corrected sigma
    if "sr" in R:
        sr = pd.read_parquet(PINNED["sr"][0])
        Xs = sr[FEATURES].to_numpy(np.float32); ws = sr["weight_nominal"].to_numpy(float); ds_s = sr["dataset"].to_numpy()
        gsr = {k: 2 * predict(b["models"], Xs)[0].astype(float) - 1 for k, b in bund.items()}
        gsr["rw_old_column"] = sr["weight_negrw"].to_numpy(float)
        R["sr"]["closure"] = {"total": {k: corrected_closure(ws, g) for k, g in gsr.items()}}
        for i, d in enumerate(DATASETS):
            m = ds_s == i
            if m.any(): R["sr"]["closure"][d] = {k: corrected_closure(ws[m], g[m]) for k, g in gsr.items()}
        R["sr"]["binned_vpt"] = binned(Xs[:, fi["lhe_vpt"]], ws, gsr, edges["lhe_vpt"])
    R["correction"] = dict(
        date="2026-11-12", script=Path(__file__).name, script_sha256=sha(__file__),
        stats_sha256=sha(Path(__file__).resolve().parent / "negrw_stats.py"),
        train_script_sha256=sha(Path(__file__).resolve().parent / "negrw_train_v2.py"),
        replaced=["closure_holdout", "binned_holdout", "binned_holdout_vpt_by_dataset", "sr.closure", "sr.binned_vpt"],
        added=["shape_after_renorm", "shape_edges"], inputs=ids, reviewer_shape_json=a.reviewer_shape_json,
        reviewer_shape_max_rel_dev=ref_dev if ref is not None else None, max_ratio_dev_vs_original=dev,
        note="statistic correction only (acceptance review F1/F2); not a shape validation of either model; G3 open")
    out.mkdir(parents=True)
    with open(out / "negrw_v2_validation_corrected_20261112.json", "x") as f:
        json.dump(R, f, indent=1)
    t = clos["total"]
    print(json.dumps({k: {q: t[k][q] for q in ("ratio", "ratio_staterr", "ratio_staterr_denominator_only_superseded")} for k in t}, indent=1))
    print("wrote", out / "negrw_v2_validation_corrected_20261112.json")


if __name__ == "__main__":
    main()
