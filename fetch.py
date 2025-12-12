import argparse
import subprocess
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--year",
        dest="year",
        type=str,
        choices=["2022preEE", "2022postEE", "2023preBPix", "2023postBPix"],
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
    args = parser.parse_args()

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
    cmd = f"singularity exec -B /afs -B /cvmfs {args.image} python3 analysis/filesets/make_filesets.py --year {args.year} --samples {samples_str}"
    subprocess.run(cmd, shell=True)
