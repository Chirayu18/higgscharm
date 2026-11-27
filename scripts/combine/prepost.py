#!/usr/bin/env python3
"""Prefit vs postfit with uncertainty band, signal overlay and ratio panel."""
import argparse, uproot, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

BKG=["tt","st","vjets","diboson","higgsbkg"]
COL={"tt":"#4c78a8","st":"#f58518","vjets":"#54a24b",
     "diboson":"#b279a2","higgsbkg":"#e45756"}

def draw(f,d,ch,ax,rax,title,sig_scale=20000):
    base=f"{d}/{ch}"; bot=None; edges=None
    for p in BKG:
        k=f"{base}/{p}"
        if k not in f: continue
        h=f[k]; v=h.values()
        if edges is None:
            edges=h.axis().edges(); x=(edges[:-1]+edges[1:])/2; bot=np.zeros_like(v)
        ax.bar(x,v,width=np.diff(edges),bottom=bot,label=p,color=COL[p],
               edgecolor="white",linewidth=.3)
        bot=bot+v
    tot=f.get(f"{base}/total")
    if tot is not None:
        tv=tot.values(); te=tot.errors()
        ax.bar(x,2*te,width=np.diff(edges),bottom=tv-te,color="none",
               hatch="////",edgecolor="#444",linewidth=0,label="total unc.")
        rax.fill_between(edges,np.append(1-te/tv,(1-te/tv)[-1]),
                         np.append(1+te/tv,(1+te/tv)[-1]),step="post",
                         color="#888",alpha=.3)
    sk=f"{base}/hplusc"
    if sk in f:
        sv=f[sk].values()*sig_scale
        ax.step(edges,np.append(sv,sv[-1]),where="post",color="k",lw=1.8,
                ls="--",label=f"H+c x{sig_scale//1000}k")
    rax.axhline(1.0,color="k",lw=.9); rax.set_ylim(.9,1.1)
    rax.set_xlabel("MVA discriminant bin",fontsize=10)
    rax.set_ylabel("ratio",fontsize=9)
    ax.set_title(title,fontsize=12,fontweight="bold"); ax.grid(alpha=.2,axis="y")
    return tot.values() if tot is not None else None

ap=argparse.ArgumentParser()
ap.add_argument("--input",required=True); ap.add_argument("--output",required=True)
ap.add_argument("--channel",default="SR_hplusc")
a=ap.parse_args()
f=uproot.open(a.input)
fig=plt.figure(figsize=(11,5.2))
gs=fig.add_gridspec(2,2,height_ratios=[3,1],hspace=.06,wspace=.18)
a0=fig.add_subplot(gs[0,0]); r0=fig.add_subplot(gs[1,0],sharex=a0)
a1=fig.add_subplot(gs[0,1],sharey=a0); r1=fig.add_subplot(gs[1,1],sharex=a1)
draw(f,"shapes_prefit",a.channel,a0,r0,f"Prefit - {a.channel}")
draw(f,"shapes_fit_s",a.channel,a1,r1,f"Postfit (S+B) - {a.channel}")
a0.set_ylabel("events",fontsize=11); a1.legend(fontsize=8,ncol=2)
for ax in (a0,a1): plt.setp(ax.get_xticklabels(),visible=False)
fig.savefig(a.output,dpi=160,bbox_inches="tight")
print("wrote",a.output)
