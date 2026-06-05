"""Impact-on-r_95 plot (manual, no CombineHarvester), yaml-driven.

Reads the per-nuisance AsymptoticLimits ROOTs produced by drive_combine.py's
manual impact loop:
    <combine_out_dir>/<tag>_impacts/<nuis>_{up,dn}.root
and the nominal AsymptoticLimits result
    <combine_out_dir>/higgsCombine<tag>Limit.AsymptoticLimits.mH120.root
Computes Delta r_95 per nuisance, ranks by max |Delta r_95|, draws the bars.

Output: <combine_out_dir>/impacts.png  (+ impacts.json)

Usage:
  python scripts/combine/make_impact_plot.py -w hww_MVA
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.workflows.config import WorkflowConfigBuilder

mpl.rcParams["figure.dpi"] = 130


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-w", "--workflow", required=True,
        choices=[f.stem for f in (Path.cwd() / "analysis" / "workflows").glob("*.yaml")],
    )
    parser.add_argument("--tag", default="V3", help="combine -n tag (default: V3)")
    parser.add_argument("--lumi", type=float, default=26.67, help="lumi [fb^-1] for the label")
    return parser.parse_args()


def read_r95(path):
    if not path.exists():
        return None
    try:
        a = uproot.open(path)["limit"].arrays(library="np")
        return float(a["limit"][2])  # median (index 2)
    except Exception:
        return None


def main():
    args = parse_args()
    cfg = WorkflowConfigBuilder(workflow=args.workflow).build_workflow_config()
    if cfg.combine is None:
        sys.exit(f"workflow {args.workflow} has no 'combine:' block")
    combine = cfg.combine

    out_dir = (Path.cwd() / combine["output"]["root"]).parent
    imp_dir = out_dir / f"{args.tag.lower()}_impacts"
    nominal = out_dir / f"higgsCombine{args.tag}Limit.AsymptoticLimits.mH120.root"

    r95_nom = read_r95(nominal)
    if r95_nom is None:
        sys.exit(f"nominal limit not found: {nominal}")
    print(f"nominal r_95 (median) = {r95_nom:.2f}")

    nuisances = list(combine["lnN"].keys()) + list(combine["shape_systematics"])
    rows = []
    for n in nuisances:
        up = read_r95(imp_dir / f"{n}_up.root")
        dn = read_r95(imp_dir / f"{n}_dn.root")
        if up is None or dn is None:
            print(f"  {n}: missing (up={up}, dn={dn})")
            continue
        d_up, d_dn = up - r95_nom, dn - r95_nom
        rows.append((n, d_up, d_dn, max(abs(d_up), abs(d_dn))))

    if not rows:
        sys.exit(f"no impact results in {imp_dir}; run run_combine.sh first")

    rows.sort(key=lambda r: r[3], reverse=True)
    names = [r[0] for r in rows]
    d_ups = np.array([r[1] for r in rows])
    d_dns = np.array([r[2] for r in rows])
    y = np.arange(len(rows))[::-1]

    fig, ax = plt.subplots(figsize=(10.5, max(6.0, 0.32 * len(rows) + 2.0)))
    ax.barh(y, d_ups, height=0.7, color="#3a78d6", edgecolor="black",
            linewidth=0.4, label=r"$+1\sigma$ pre-fit impact")
    ax.barh(y, d_dns, height=0.7, color="#e36c5d", edgecolor="black",
            linewidth=0.4, label=r"$-1\sigma$ pre-fit impact")
    for yi, du, dd in zip(y, d_ups, d_dns):
        x_anno = max(abs(du), abs(dd)) * 1.05
        ax.text(x_anno if du >= 0 else -x_anno, yi, f"{du:+.1f}/{dd:+.1f}",
                va="center", ha="left" if du >= 0 else "right", fontsize=9)

    xmax = max(np.max(np.abs(d_ups)), np.max(np.abs(d_dns))) * 1.35
    ax.set_xlim(-xmax, xmax)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(r"$\Delta r_{95}$  (Asimov, nuisance frozen at $\pm 1\sigma$)")
    ax.set_title(f"Nuisance impacts on expected $r_{{95}}$  (nominal = {r95_nom:.0f})",
                 fontsize=12, loc="left", pad=18)
    ax.legend(loc="lower right", fontsize=10, frameon=False)
    ax.grid(axis="x", alpha=0.3)

    png = out_dir / "impacts.png"
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png}")

    with open(out_dir / "impacts.json", "w") as f:
        json.dump({"r95_nominal": r95_nom,
                   "rows": [{"nuis": r[0], "dup": r[1], "ddn": r[2]} for r in rows]},
                  f, indent=2)
    print(f"Saved {out_dir / 'impacts.json'}")


if __name__ == "__main__":
    main()
