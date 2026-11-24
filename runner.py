from analysis.processing_modes import add_arguments as add_mode_arguments, forwarded as mode_arguments
import argparse
import subprocess
from pathlib import Path
from analysis.filesets.utils import (
    fileset_checker,
    get_datasets_to_run_over,
)


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
        "--nfiles",
        dest="nfiles",
        type=int,
        default=15,
        help="number of root files to include in each dataset partition (default 15)",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Enable Condor job submission. If not provided, it just builds condor files",
    )
    parser.add_argument(
        "--eos",
        action="store_true",
        help="Enable saving outputs to /eos",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="coffea",
        choices=["coffea", "parquet"],
        help="format of output file",
    )
    parser.add_argument(
        "-m",
        "--memory",
        type=str,
        default="2000",
        help="Requested memory (in MB) for the condor job",
    )
    parser.add_argument(
        "--output_root",
        default=None,
        help="absolute root replacing the default output area for .coffea and parquet outputs "
             "(<root>/outputs/<workflow>[__<mode>]/<year>/<dataset>). Unset: unchanged default.",
    )
    parser.add_argument(
        "--output_root_url",
        default=None,
        help="xrootd URL of --output_root for parquet writes (needed for non-/eos/user EOS roots)",
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

    # get datasets to run over for selected workflow and year
    datasets_to_run_over = get_datasets_to_run_over(args.workflow, args.year)

    # check if the input fileset for the given year exists, generate it otherwise
    fileset_checker(samples=datasets_to_run_over, year=args.year)

    # submit (or prepare) a job for each dataset using the given arguments
    cmd = ["python3", "submit_condor.py"]
    for dataset in datasets_to_run_over:
        cmd_args = [
            "--workflow",
            args.workflow,
            "--year",
            args.year,
            "--dataset",
            dataset,
            "--nfiles",
            str(args.nfiles),
            "--output_format",
            args.output_format,
            "--memory",
            args.memory
        ]
        if args.submit:
            cmd_args.append("--submit")
        if args.eos:
            cmd_args.append("--eos")
        if args.output_root:
            cmd_args += ["--output_root", args.output_root]
        if args.output_root_url:
            cmd_args += ["--output_root_url", args.output_root_url]
        subprocess.run(cmd + cmd_args + mode_arguments(args), check=True)
