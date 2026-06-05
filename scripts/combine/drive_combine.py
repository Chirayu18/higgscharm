"""Drive combine fits from the workflow yaml `combine.run` block.

Run inside a CMSSW + combine environment (see run_combine.sh). Reads the
datacard/workspace paths and run options from the yaml, then executes:
  - text2workspace.py
  - combine -M AsymptoticLimits   (Asimov / blind per yaml)
  - combine -M MultiDimFit --algo grid  (NLL scan)
  - manual per-nuisance impact loop  (freeze each nuisance at +/-1)

All combine invocations run with cwd = the datacard's directory so the
output higgsCombine*.root files land next to the inputs.

Usage (from run_combine.sh, inside CMSSW):
  python3 scripts/combine/drive_combine.py -w hww_MVA
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.workflows.config import WorkflowConfigBuilder


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-w", "--workflow", required=True,
        choices=[f.stem for f in (Path.cwd() / "analysis" / "workflows").glob("*.yaml")],
        help="workflow yaml name (must contain a 'combine:' block)",
    )
    parser.add_argument("--name", default="V3", help="combine -n tag (default: V3)")
    return parser.parse_args()


def run(cmd, cwd):
    print(f"\n$ {cmd}")
    subprocess.run(cmd, shell=True, cwd=str(cwd), check=True)


def main():
    args = parse_args()
    cfg = WorkflowConfigBuilder(workflow=args.workflow).build_workflow_config()
    if cfg.combine is None:
        sys.exit(f"workflow {args.workflow} has no 'combine:' block in its yaml")
    combine = cfg.combine

    datacard = Path.cwd() / combine["output"]["datacard"]
    workdir = datacard.parent
    datacard_name = datacard.name
    workspace = combine["output"]["workspace"]
    run_cfg = combine.get("run", {})
    tag = args.name

    if not datacard.exists():
        sys.exit(f"datacard not found: {datacard} (run make_combine_inputs.py first)")

    # workspace
    run(f"text2workspace.py {datacard_name} -o {workspace}", workdir)

    # AsymptoticLimits
    al = run_cfg.get("asymptotic_limits")
    if al:
        opts = f"-M AsymptoticLimits {workspace} -n {tag}Limit --mass 120"
        if al.get("asimov"):
            opts += " -t -1"
        if al.get("blind"):
            opts += " --run blind"
        run(f"combine {opts}", workdir)

    # MultiDimFit grid scan
    md = run_cfg.get("multidimfit_scan")
    if md:
        lo, hi = md["range_r"]
        opts = (f"-M MultiDimFit {workspace} -n {tag}Grid --algo {md.get('algo', 'grid')} "
                f"--rMin {lo} --rMax {hi} --points {md.get('points', 41)} -t -1 --mass 120")
        run(f"combine {opts}", workdir)

    # manual per-nuisance impacts
    imp = run_cfg.get("impacts")
    if imp and imp.get("method") == "freeze_per_nuisance":
        nuisances = list(combine["lnN"].keys()) + list(combine["shape_systematics"])
        imp_dir = workdir / f"{tag.lower()}_impacts"
        imp_dir.mkdir(parents=True, exist_ok=True)
        for nuis in nuisances:
            for val, side in (("+1", "up"), ("-1", "dn")):
                base = f"_imp_{nuis}_{side}"
                cmd = (f"combine -M AsymptoticLimits {workspace} -t -1 --run blind --noFitAsimov "
                       f"--setParameters {nuis}={val} --freezeParameters {nuis} "
                       f"-n {base} --mass 120")
                try:
                    run(cmd, workdir)
                    src = workdir / f"higgsCombine{base}.AsymptoticLimits.mH120.root"
                    if src.exists():
                        src.rename(imp_dir / f"{nuis}_{side}.root")
                except subprocess.CalledProcessError:
                    print(f"  [warn] impact fit failed for {nuis} {side}")
        print(f"\nImpacts written to {imp_dir}")

    print("\ndrive_combine: done")


if __name__ == "__main__":
    main()
