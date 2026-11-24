from analysis.processing_modes import add_arguments as add_mode_arguments, forwarded as mode_arguments
import os
import json
import argparse
from pathlib import Path
from coffea import processor
from coffea.util import save
from coffea.nanoevents import NanoAODSchema
from analysis.utils import write_root
from analysis.processors.base import BaseProcessor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-w",
        "--workflow",
        dest="workflow",
        required=True,
        type=str,
        choices=[
            f.stem for f in (Path.cwd() / "analysis" / "workflows").glob("*.yaml")
        ],
        help="workflow to run",
    )
    parser.add_argument(
        "-y",
        "--year",
        dest="year",
        type=str,
        choices=[
            "2016preVFP",
            "2016postVFP",
            "2017",
            "2018",
            "2022preEE",
            "2022postEE",
            "2023preBPix",
            "2023postBPix",
            "2024"
        ],
        help="dataset year",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        dest="dataset",
        type=str,
        help="dataset",
    )
    parser.add_argument(
        "--partition_json",
        dest="partition_json",
        type=str,
        help="json with partition dataset",
    )
    parser.add_argument(
        "--output_path",
        dest="output_path",
        type=str,
        help="output path",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="coffea",
        choices=["coffea", "root", "parquet"],
        help="format of output histogram",
    )
    parser.add_argument(
        "--eos",
        action="store_true",
        help="Enable saving outputs to /eos",
    )
    parser.add_argument(
        "--user",
        type=str,
        help="User name",
    )
    parser.add_argument(
        "--output_location",
        type=str,
        default=None,
        help="explicit parquet output root (overrides --eos/AFS default); e.g. a node-local /tmp dir for smoke tests",
    )
    add_mode_arguments(parser)
    args = parser.parse_args()
    from analysis.processing_modes import validate
    from analysis.workflows.config import WorkflowConfigBuilder
    validate(WorkflowConfigBuilder(args.workflow).build_workflow_config(), args.mode, args.folds,
             args.model_manifest, args.allow_diagnostic_model, year=args.year)
    if args.mode != "legacy":
        args.output_format = "parquet" if args.mode in ("train", "legacy_slim") else "coffea"
    if args.mode == "run" and args.allow_diagnostic_model and getattr(args, "submit", False):
        parser.error("Diagnostic models are local-test only; cannot submit them")

    # set output location (used when --output_format parquet)
    if args.output_location:
        output_location = args.output_location.rstrip("/") + "/"
    elif args.eos:
        output_location = f"root://eosuser.cern.ch//eos/user/{args.user[0]}/{args.user}/higgscharm/outputs/"
    else:
        output_location = (
            f"/afs/cern.ch/user/{args.user[0]}/{args.user}/public/higgscharm/outputs/"
        )

    # load partition_fileset, run processor and save output
    with open(args.partition_json) as f:
        partition_fileset = json.load(f)
    out = processor.run_uproot_job(
        partition_fileset,
        treename="Events",
        processor_instance=BaseProcessor(
            workflow=args.workflow,
            year=args.year,
            output_format=args.output_format,
            output_location=output_location,
            mode=args.mode, folds=args.folds, model_manifest=args.model_manifest,
            allow_diagnostic_model=args.allow_diagnostic_model, slim_ntuple=args.slim_ntuple,
        ),
        executor=processor.futures_executor,
        # workers=2 (was 4): each worker holds a NanoAOD chunk in RAM; 4 pushed peak
        # to ~5.9 GB and OOM-truncated heavy samples at the 2-3 GB condor request.
        # 2 roughly halves peak (~3 GB) so jobs fit a modest request; ~2x slower/job.
        executor_args={"schema": NanoAODSchema, "workers": 2,
                       # 2026-10-02: coffea default xrootdtimeout is 60 s; slow-but-alive sites
                       # exceeded it ("Operation expired") -> ~35% of jobs failed on the first pass.
                       "xrootdtimeout": 600},
    )
    if args.mode != "legacy":
        from analysis.processing_checks import check_input_entries
        out["execution_checks"] = check_input_entries(partition_fileset, out, args.year)
    # coffea.util.save does not create directories; make sure the target exists.
    if not args.output_path:
        parser.error("--output_path is required to save the job output")
    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    savepath = f"{args.output_path}/{args.dataset}"
    if args.output_format in ["coffea", "parquet"]:
        save(out, f"{savepath}.coffea")
    elif args.output_format == "root":
        write_root(out, savepath, args)
