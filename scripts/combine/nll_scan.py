import uproot, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
t=uproot.open("/tmp/deepscan/higgsCombine_scan_cur.MultiDimFit.mH120.root")["limit"]
r=t["r"].array(library="np"); n=t["deltaNLL"].array(library="np")
m=np.isfinite(n)&(n>=0); r,n=r[m],n[m]; o=np.argsort(r); r,n=r[o],n[o]
q=2*n
fig,ax=plt.subplots(figsize=(8.2,4.6)); fig.subplots_adjust(left=.11,right=.97,top=.90,bottom=.15)
ax.plot(r,q,color="#1f4e79",lw=2.4,zorder=3)
ax.axhline(1.0,ls="--",color="#8c959d",lw=1.5)
ax.text(r.max()*0.985,1.08,"68% CL  ($-2\\Delta\\ln L=1$)",color="#5a6169",fontsize=9,ha="right")
ax.axvline(1.0,ls=":",color="#2f6b3c",lw=1.6)
ax.text(60,6.4,"$\\hat r = 1$  (Asimov, injected)",color="#2f6b3c",fontsize=10)
ax.set_xlabel("r  (signal strength)",fontsize=12)
ax.set_ylabel(r"$-2\Delta\ln L$",fontsize=12)
ax.set_ylim(0,7); ax.set_xlim(0,1500); ax.grid(alpha=.25)
ax.set_title("Asimov likelihood scan — profile in r",fontsize=13,fontweight="bold")
fig.savefig("/tmp/nll_scan2.png",dpi=160)
print("wrote; min at r=",r[np.argmin(n)])
