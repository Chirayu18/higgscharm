"""Combine result plots (prefit SR stack + Brazil band), yaml-driven.

Reads the combine ROOT and AsymptoticLimits result from the paths in the
workflow `combine:` block and writes:
  <combine_out_dir>/prefit_SR.png      stacked D in the signal region (S/sqrt(B) ratio)
  <combine_out_dir>/r95_brazil.png     expected 95% CL Brazil band

Usage:
  python scripts/combine/make_combine_plots.py -w hww_MVA
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.workflows.config import WorkflowConfigBuilder

mpl.rcParams["figure.dpi"] = 130

PROC_COLOR = {
    "tt": "#e36c5d", "st": "#f3a3a3", "diboson": "#9ad1ff",
    "vjets": "#88d090", "higgsbkg": "#c5a7ff", "hplusc": "k",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-w", "--workflow", required=True,
        choices=[f.stem for f in (Path.cwd() / "analysis" / "workflows").glob("*.yaml")],
    )
    parser.add_argument("--tag", default="V3")
    parser.add_argument("--lumi", type=float, default=26.67)
    parser.add_argument("--sig-scale", type=float, default=1000.0,
                        help="signal overlay scale factor")
    return parser.parse_args()


def get_hist(f, key):
    counts, edges = f[key].to_numpy()
    return counts, edges


def main():
    args = parse_args()
    cfg = WorkflowConfigBuilder(workflow=args.workflow).build_workflow_config()
    if cfg.combine is None:
        sys.exit(f"workflow {args.workflow} has no 'combine:' block")
    combine = cfg.combine

    signal = combine["signal"]
    classes = combine["classes"]
    backgrounds = [c for c in classes if c != signal]
    # signal region channel = the channel whose class == signal
    sr_channel = next(ch for ch, cls in combine["channels"].items() if cls == signal)

    out_dir = (Path.cwd() / combine["output"]["root"]).parent
    root_path = Path.cwd() / combine["output"]["root"]
    if not root_path.exists():
        sys.exit(f"combine ROOT not found: {root_path} (run make_combine_inputs.py)")

    # ---- prefit SR stack ----
    f = uproot.open(root_path)
    stack = [(p, *get_hist(f, f"{sr_channel}_{p}")) for p in backgrounds]
    c_sig, edges = get_hist(f, f"{sr_channel}_{signal}")
    centers = 0.5 * (edges[1:] + edges[:-1])
    width = edges[1] - edges[0]

    fig, (ax, axr) = plt.subplots(
        2, 1, sharex=True, figsize=(9.5, 7),
        gridspec_kw=dict(height_ratios=[3.0, 1.0], hspace=0.07),
    )
    bottom = np.zeros_like(stack[0][1])
    total = np.zeros_like(stack[0][1])
    for p, c, _ in stack:
        ax.bar(centers, c, width=width, bottom=bottom,
               color=PROC_COLOR.get(p, "#cccccc"), edgecolor="black",
               linewidth=0.4, label=p)
        bottom = bottom + c
        total = total + c
    ax.step(edges, np.r_[c_sig[0], c_sig] * args.sig_scale, where="pre",
            color="k", linewidth=1.8, label=f"{signal} x{args.sig_scale:.0f}")
    ax.set_yscale("log")
    ax.set_ylabel("Events / bin")
    ax.set_ylim(1e-2, 1e5)
    ax.legend(loc="upper right", fontsize=11, ncol=2, frameon=False)
    ax.set_title(f"Prefit {sr_channel}  ({args.lumi:.1f} fb$^{{-1}}$)", loc="left")

    sb = c_sig / np.sqrt(np.maximum(total, 1e-9))
    axr.step(edges, np.r_[sb[0], sb], where="pre", color="k", linewidth=1.4)
    axr.set_yscale("log")
    axr.set_ylim(1e-4, 1e-1)
    axr.set_ylabel(r"$S / \sqrt{B}$")
    axr.set_xlabel(r"$D = P(\mathrm{argmax})$")
    axr.set_xlim(edges[0], edges[-1])
    axr.grid(True, alpha=0.3)
    fig.savefig(out_dir / "prefit_SR.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'prefit_SR.png'}")

    # ---- Brazil band ----
    limit_root = out_dir / f"higgsCombine{args.tag}Limit.AsymptoticLimits.mH120.root"
    if not limit_root.exists():
        print(f"  [skip brazil] {limit_root} not found")
        return
    lim = uproot.open(limit_root)["limit"].arrays(library="np")["limit"]
    m2, m1, med, p1, p2 = list(lim)

    fig, ax = plt.subplots(figsize=(10.5, 3.4))
    yv = 0
    ax.barh(yv, p2 - m2, left=m2, height=0.55, color="#ffe46b", label=r"95% CL $\pm 2\sigma$")
    ax.barh(yv, p1 - m1, left=m1, height=0.55, color="#3aa54b", label=r"95% CL $\pm 1\sigma$")
    ax.plot([med, med], [yv - 0.3, yv + 0.3], color="k", linewidth=2.2, label=f"median = {med:.0f}")
    ax.text(p2 * 1.03, yv, f"{med:.0f}", va="center", fontsize=12, weight="bold")
    ax.set_yticks([yv])
    ax.set_yticklabels([f"{args.tag}"])
    ax.set_xlim(0, p2 * 1.2)
    ax.set_xlabel(rf"Expected 95% CL upper limit on $r$  (Asimov, {args.lumi:.1f} fb$^{{-1}}$)")
    ax.legend(loc="lower right", fontsize=10, frameon=False, ncol=2)
    fig.savefig(out_dir / "r95_brazil.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'r95_brazil.png'}")


if __name__ == "__main__":
    main()
