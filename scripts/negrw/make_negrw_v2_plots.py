#!/usr/bin/env python3
"""make_negrw_v2_plots.py -- plots from negrw_train_v2.py outputs (negrw_v2_validation.json + negrw_v2_curves.npz).
usage: make_negrw_v2_plots.py <outdir_of_training | validation json> <plot_outdir>
(2026-11-12: a JSON path, e.g. the corrected report negrw_v2_validation_corrected_20261112.json, is accepted)"""
import sys, os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
if src.endswith(".json"):
    R = json.load(open(src)); src = os.path.dirname(src)
else:
    R = json.load(open(os.path.join(src, "negrw_v2_validation.json")))
C = {"rw_new": "#1f77b4", "rw_old": "#d62728", "rw_old_column": "#ff9896", "nom": "black"}
L = {"rw_new": "v2 model |w|·g", "rw_old": "July model |w|·g", "rw_old_column": "stored weight_negrw", "nom": "nominal w"}


def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(dst, name), dpi=130); plt.close(fig); print("wrote", name)


# 1 loss curves
try:
    cur = [json.loads(c) for c in np.load(os.path.join(src, "negrw_v2_curves.npz"))["curves"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, c in enumerate(cur):
        ax.plot(c["train"], color="C0", alpha=.4, lw=.8, label="train" if i == 0 else None)
        if c["val"]: ax.plot(c["val"], color="C1", alpha=.4, lw=.8, label="early-stop val" if i == 0 else None)
    ax.set_xlabel("iteration"); ax.set_ylabel("log-loss"); ax.legend(); ax.set_title(f"20 members, n_iter {R['n_iters']}", fontsize=7)
    save(fig, "01_loss_curves.png")
except Exception as e:
    print("curves skipped:", e)

# 2 calibration
cal = R["calibration"]; e = np.array(cal["edges"]); ctr = .5 * (e[1:] + e[:-1])
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot([0, 1], [0, 1], "k:", lw=1)
for k, col in (("new", "rw_new"), ("old", "rw_old")):
    if k in cal:
        n = np.array(cal[k]["n"]); ok = n > 50
        ax.plot(np.array(cal[k]["pred"])[ok], np.array(cal[k]["obs"])[ok], "o-", color=C[col], label=f"{k} model (held-out)")
ax.set_xlabel("predicted P+"); ax.set_ylabel("observed fraction w>0"); ax.legend(); ax.set_xlim(.4, 1); ax.set_ylim(.4, 1)
save(fig, "02_calibration_holdout.png")

# 3 g distributions (held-out, SR)
bins = np.linspace(-1, 1, 81)
for tag, H in (("holdout", R["hist_g"]), ("allfixSR", R.get("sr", {}).get("hist_g"))):
    if not H: continue
    fig, ax = plt.subplots(figsize=(6, 4))
    for k, h in H.items():
        h = np.array(h, float); ax.stairs(h / h.sum(), bins, color=C.get(k), label=L.get(k, k))
    ax.set_xlabel("g = 2P+ − 1"); ax.set_ylabel("fraction of events"); ax.set_yscale("log"); ax.legend(); ax.set_title(tag)
    save(fig, f"03_g_{tag}.png")

# 4 closure ratio per (era, dataset) on held-out rows
clos = R["closure_holdout"]
keys = [k for k in clos if "/" in k]
fig, ax = plt.subplots(figsize=(10, 4.5))
x = np.arange(len(keys))
for j, k in enumerate(("rw_new", "rw_old")):
    if k not in clos["total"]: continue
    r = np.array([clos[q][k]["ratio"] for q in keys])
    # 2026-11-12: ratio_staterr is the sigma of R itself (delta method, ratio_staterr_def). The pre-correction
    # JSON holds the denominator-only sqrt(sum w^2)/|sum w|: no error bars are drawn from it.
    er = (np.array([clos[q][k]["ratio_staterr"] for q in keys]) if clos["total"][k].get("ratio_staterr_def") == "delta"
          else None)
    ax.errorbar(x + (j - .5) * .25, r, yerr=er, fmt="o", color=C[k], label=L[k] + f"  (total {clos['total'][k]['ratio']:.4f})"
                + ("" if er is not None else ", no valid σ in this JSON"), ms=4)
ax.axhline(1, color="k", lw=.8)
ax.set_xticks(x); ax.set_xticklabels([k.replace("_2Jets", "") for k in keys], rotation=75, fontsize=6)
ax.set_ylabel("Σ|w|g / Σw  (held-out)"); ax.legend(fontsize=7); ax.set_ylim(.9, 1.1)
save(fig, "04_closure_holdout_era_dataset.png")


def ratio_panel(B, title, name, xlabel):
    e = np.array(B["edges"]); nom = np.array(B["nom"]); nv = np.array(B["nom_var"])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6, 5.5), sharex=True, gridspec_kw=dict(height_ratios=[2, 1]))
    a1.stairs(nom, e, color="k", label=L["nom"])
    for k in ("rw_new", "rw_old", "rw_old_column"):
        if k not in B: continue
        v = np.array(B[k]); a1.stairs(v, e, color=C[k], label=L[k])
        with np.errstate(divide="ignore", invalid="ignore"):
            r = v / nom
        # 2026-11-12: per-bin delta-method sigma when the JSON has it; the old sqrt(nom_var)/|nom| was denominator-only
        er = np.sqrt(np.array(B[k + "_ratio_var"], float)) if k + "_ratio_var" in B else None
        a2.errorbar(.5 * (e[1:] + e[:-1]), r, yerr=er, fmt="o", ms=3, color=C[k])
    a1.set_yscale("log"); a1.legend(fontsize=7); a1.set_title(title, fontsize=9)
    a2.axhline(1, color="k", lw=.8); a2.set_ylim(.8, 1.2); a2.set_ylabel("rw / nominal"); a2.set_xlabel(xlabel)
    save(fig, name)


# 5 binned closure per feature (held-out)
for f, B in R["binned_holdout"].items():
    ratio_panel(B, f"held-out, all samples: {f}", f"05_binned_holdout_{f}.png", f)
# 6 lhe_vpt per dataset
for d, B in R["binned_holdout_vpt_by_dataset"].items():
    ratio_panel(B, f"held-out {d}: lhe_vpt", f"06_vpt_holdout_{d}.png", "lhe_vpt [GeV]")
# 7 SR
if "sr" in R and "binned_vpt" in R["sr"]:
    ratio_panel(R["sr"]["binned_vpt"], f"allfix eμ SR V+jets (n={R['sr']['n']}): lhe_vpt", "07_sr_vpt.png", "lhe_vpt [GeV]")
# 9 shape residual after per-sample renormalisation (corrected report only; fitted-k shared-event covariance)
for key, S in R.get("shape_after_renorm", {}).items():
    if not key.endswith("/rw_new") or "rel_resid" not in S or S.get("plot") is not True:
        continue
    var = key.split("/")[0]; ed = np.array(R["shape_edges"][var], float)
    lab = ["NaN" if i == len(ed) - 1 else "<%g" % ed[i + 1] if not np.isfinite(ed[i]) else
           ">=%g" % ed[i] if not np.isfinite(ed[i + 1]) else "%g-%g" % (ed[i], ed[i + 1]) for i in range(len(ed))]
    fig, ax = plt.subplots(figsize=(7, 3.6)); xs = np.arange(len(lab))
    for mk, off in (("rw_new", -.12), ("rw_old", .12)):
        T = R["shape_after_renorm"].get(key[:-len("rw_new")] + mk)
        if not T or "rel_resid" not in T: continue
        rr, re = np.array(T["rel_resid"], float), np.array(T["rel_sigma"], float)
        ax.errorbar(xs + off, rr, yerr=re, fmt="o", ms=3, color=C[mk],
                    label=f"{L[mk]}: χ² {T['chi2']:.1f}/{T['rank']} (+nuis {T['chi2_with_nuis']:.1f})")
    nu = np.array(S["rel_nuis_up"], float); ax.fill_between(xs, -np.abs(nu), np.abs(nu), step="mid", color="0.85", label="|CMS_negrw_vjets| (v2)")
    ax.axhline(0, color="k", lw=.8); ax.set_xticks(xs); ax.set_xticklabels(lab, rotation=60, fontsize=6)
    ax.set_ylabel("(k·Σ|w|g − Σw) / Σw"); ax.legend(fontsize=6); ax.set_title(f"held-out, renormalised per (era, dataset): {key[:-7]}", fontsize=8)
    save(fig, "09_shape_renorm_" + key[:-7].replace("/", "_") + ".png")
# 8 δg
fig, ax = plt.subplots(figsize=(6, 4))
ax.stairs(np.array(R["hist_dg_new"], float), np.linspace(0, .2, 81), color=C["rw_new"])
ax.set_yscale("log"); ax.set_xlabel("δg = 2·std(P+) over 20 members (held-out)"); ax.set_ylabel("events")
save(fig, "08_dg_holdout.png")
