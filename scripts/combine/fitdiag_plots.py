#!/usr/bin/env python3
"""Standard prefit/postfit fit diagnostics: nuisance pulls + per-process yields.

  python3 fitdiag_plots.py --input fitDiagnostics.root --outdir <dir>

Produces:
  pulls.png   -- (theta_hat - theta_0)/sigma per nuisance, with postfit constraint
  yields.png  -- per-process prefit vs postfit normalisation in the SR
"""
import argparse, uproot, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import ROOT  # noqa

ap=argparse.ArgumentParser()
ap.add_argument("--input",required=True); ap.add_argument("--outdir",required=True)
ap.add_argument("--channel",default="SR_hplusc"); ap.add_argument("--top",type=int,default=25)
a=ap.parse_args()

fr=ROOT.TFile.Open(a.input)
fit=fr.Get("fit_s")
pars=fit.floatParsFinal()
names,vals,errs=[],[],[]
for i in range(pars.getSize()):
    p=pars.at(i); n=p.GetName()
    if n=="r": continue
    names.append(n); vals.append(p.getVal()); errs.append(p.getError())
o=np.argsort([-abs(v) for v in vals])[:a.top]
names=[names[i] for i in o]; vals=[vals[i] for i in o]; errs=[errs[i] for i in o]
y=np.arange(len(names))[::-1]
fig,ax=plt.subplots(figsize=(8,max(4,0.32*len(names))))
ax.axvspan(-1,1,color="#ffe08a",alpha=.35); ax.axvspan(-2,2,color="#cfe8a9",alpha=.20,zorder=0)
ax.axvline(0,color="k",lw=1)
ax.errorbar(vals,y,xerr=errs,fmt="o",color="#1f4e79",ms=4,capsize=2)
ax.set_yticks(y); ax.set_yticklabels(names,fontsize=7)
ax.set_xlabel(r"$(\hat\theta-\theta_0)/\sigma_\theta$",fontsize=11)
ax.set_xlim(-3,3); ax.grid(alpha=.2,axis="x")
ax.set_title("Nuisance pulls and postfit constraints",fontsize=12,fontweight="bold")
fig.tight_layout(); fig.savefig(f"{a.outdir}/pulls.png",dpi=160); plt.close(fig)
print("wrote pulls.png")

f=uproot.open(a.input)
pre=f[f"norm_prefit"]; post=f[f"norm_fit_s"]
def grab(d):
    out={}
    for k in d.keys():
        kk=k.split(";")[0]
        if kk.startswith(a.channel+"/"):
            out[kk.split("/")[1]]=d[kk]
    return out
try:
    P,Q=grab(pre),grab(post)
    procs=[p for p in P if p in Q]
    pv=[float(P[p].members["fVal"]) if hasattr(P[p],"members") else float(P[p]) for p in procs]
    qv=[float(Q[p].members["fVal"]) if hasattr(Q[p],"members") else float(Q[p]) for p in procs]
    x=np.arange(len(procs)); w=.38
    fig,ax=plt.subplots(figsize=(8,4.2))
    ax.bar(x-w/2,pv,w,label="prefit",color="#8c959d")
    ax.bar(x+w/2,qv,w,label="postfit",color="#1f4e79")
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(procs,fontsize=9)
    ax.set_ylabel("events",fontsize=11); ax.legend(fontsize=9); ax.grid(alpha=.2,axis="y")
    ax.set_title(f"Prefit vs postfit yields - {a.channel}",fontsize=12,fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{a.outdir}/yields.png",dpi=160)
    print("wrote yields.png")
except Exception as e:
    print("yields failed:",type(e).__name__,e)
