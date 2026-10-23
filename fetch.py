import argparse
import os
import subprocess
from pathlib import Path
from analysis.filesets.utils import get_nano_version

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
            "2024",
        ],
    )
    parser.add_argument(
        "--image",
        dest="image",
        type=str,
        default="/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask-almalinux9:2025.10.1-py3.10",
    )
    parser.add_argument(
        "--samples",
        nargs="*",
        type=str,
        help="(Optional) List of samples to use. If omitted, all available samples will be used",
    )
    parser.add_argument(
        "--site",
        dest="site",
        default="root://maite.iihe.ac.be:1094",
        type=str,
        help="site from which to read the signal samples",
    )
    parser.add_argument(
        "--onlysignal", action="store_true", help="Build only signal input filesets"
    )
    parser.add_argument(
        "-s", "--signal", action="store_true", help="Build signal input filesets"
    )
    args = parser.parse_args()

    if not args.onlysignal:
        try:
            subprocess.run("voms-proxy-info -exists", shell=True, check=True)
        except subprocess.CalledProcessError:
            raise Exception(
                "VOMS proxy expired or non-existing: please run 'voms-proxy-init --voms cms'"
            )
        sites_file = Path.cwd() / "analysis" / "filesets" / f"{args.year}_sites.yaml"
        if not sites_file.exists():
            cmd = f"python3 analysis/filesets/build_sites.py --year {args.year}"
            subprocess.run(cmd, shell=True)

        samples_str = " ".join(args.samples) if args.samples else ""
        make_cmd = f"python3 -m analysis.filesets.make_filesets --year {args.year} --samples {samples_str}"
        # already inside a container (e.g. the coffea image via apptainer): run directly,
        # make_filesets loads the repo's rucio_utils by itself
        in_container = any(
            v in os.environ for v in ("APPTAINER_CONTAINER", "SINGULARITY_CONTAINER")
        )
        cmd = make_cmd if in_container else (
            f"singularity exec "
            f"--env PYTHONNOUSERSITE=1 "
            f"-B /afs "
            f"-B /cvmfs "
            f"-B analysis/filesets/rucio_utils.py:/usr/local/lib/python3.10/site-packages/coffea/dataset_tools/rucio_utils.py "
            f"{args.image} "
            f"{make_cmd}"
        )
        subprocess.run(cmd, shell=True)

    if args.signal or args.onlysignal:
        # add signal samples
        signal_cmd = f"python3 -m analysis.filesets.make_signal_filesets --year {args.year} --site {args.site}"
        subprocess.run(signal_cmd, shell=True)
