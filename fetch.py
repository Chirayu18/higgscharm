import argparse
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

    try:
        subprocess.run("voms-proxy-info -exists", shell=True, check=True)
    except subprocess.CalledProcessError:
        raise Exception(
            "VOMS proxy expired or non-existing: please run 'voms-proxy-init --voms cms'"
        )
    if not args.onlysignal:
        sites_file = Path.cwd() / "analysis" / "filesets" / f"{args.year}_sites.yaml"
        if not sites_file.exists():
            cmd = f"python3 analysis/filesets/build_sites.py --year {args.year}"
            subprocess.run(cmd, shell=True)

        samples_str = " ".join(args.samples) if args.samples else ""
        cmd = (
            f"singularity exec "
            f"--env PYTHONNOUSERSITE=1 "
            f"-B /afs "
            f"-B /cvmfs "
            f"-B analysis/filesets/rucio_utils.py:/usr/local/lib/python3.10/site-packages/coffea/dataset_tools/rucio_utils.py "
            f"{args.image} "
            f"python3 -m analysis.filesets.make_filesets --year {args.year} --samples {samples_str}"
        )
        subprocess.run(cmd, shell=True)

    if args.signal or args.onlysignal:
        # add signal samples
        signal_cmd = f"python3 -m analysis.filesets.make_signal_filesets --year {args.year} --site {args.site}"
        subprocess.run(signal_cmd, shell=True)
