import argparse
import json
import os
import subprocess
from pathlib import Path

import yaml
from analysis.filesets.utils import get_nano_version
from coffea.dataset_tools.dataset_query import DataDiscoveryCLI


def query_das_user_dataset(query, instance="prod/phys03"):
    """
    Query DAS for USER datasets using dasgoclient.

    Parameters
    ----------
    query : str
        Dataset path (e.g., /Primary/Processing/USER)
    instance : str
        DBS instance (default: prod/phys03 for USER datasets)

    Returns
    -------
    list
        List of file paths
    """
    cmd = [
        "/cvmfs/cms.cern.ch/common/dasgoclient",
        f"--query=file dataset={query} instance={instance}",
        "--limit=0"  # get all files
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        files = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
        return files
    except subprocess.CalledProcessError as e:
        print(f"Error querying DAS for {query}: {e.stderr}")
        return []


def is_user_dataset(query):
    """Check if dataset is a USER dataset."""
    return query.endswith("/USER")


if __name__ == "__main__":
    years = [
        "2016preVFP",
        "2016postVFP",
        "2017",
        "2018",
        "2022preEE",
        "2022postEE",
        "2023preBPix",
        "2023postBPix",
        "2024",
    ]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-y",
        "--year",
        dest="year",
        type=str,
        choices=years,
    )
    parser.add_argument(
        "--samples",
        nargs="*",
        type=str,
        help="(Optional) List of samples to use. If omitted, all available samples will be used",
    )
    parser.add_argument(
        "--dbs-instance",
        dest="dbs_instance",
        type=str,
        default="prod/phys03",
        help="DBS instance for USER datasets (default: prod/phys03)",
    )
    args = parser.parse_args()

    # open dataset configs
    filesets_dir = Path.cwd() / "analysis" / "filesets"
    datasets_dir = filesets_dir / f"{args.year}_nanov12.yaml"
    with open(datasets_dir, "r") as f:
        dataset_configs = yaml.safe_load(f)

    # read dataset queries
    if args.samples:
        samples_to_use = args.samples
    else:
        samples_to_use = list(dataset_configs.keys())

    das_queries = {}
    for sample in samples_to_use:
        query = dataset_configs[sample]["query"]
        if query:
            das_queries[sample] = query
        else:
            print(f"No available query for: {sample}")

    # Separate USER datasets from regular datasets
    user_datasets = {}
    rucio_datasets = {}

    for dataset_key, query in das_queries.items():
        full_query = f"/{query}"
        if is_user_dataset(full_query):
            user_datasets[dataset_key] = full_query
        else:
            rucio_datasets[dataset_key] = full_query

    # Handle regular datasets with Rucio
    new_dataset = {key: [] for key in das_queries}

    if rucio_datasets:
        # create a dataset_definition dict for Rucio datasets
        dataset_definition = {}
        for dataset_key, query in rucio_datasets.items():
            dataset_definition[query] = {"short_name": dataset_key}

        # the dataset definition is passed to a DataDiscoveryCLI
        ddc = DataDiscoveryCLI()

        # set the allow sites to look for replicas
        sites_file = filesets_dir / f"{args.year}_sites.yaml"
        with open(sites_file, "r") as f:
            sites = yaml.safe_load(f)["white"]
        ddc.do_allowlist_sites(sites)

        # query rucio and get replicas
        ddc.load_dataset_definition(
            dataset_definition,
            query_results_strategy="all",
            replicas_strategy="round-robin",
        )
        ddc.do_save(f"dataset_discovery_{args.year}.json")

        # load and reformat generated fileset
        with open(f"dataset_discovery_{args.year}.json", "r") as f:
            dataset_discovery = json.load(f)

        for dataset in dataset_discovery:
            root_files = list(dataset_discovery[dataset]["files"].keys())
            dataset_key = dataset_discovery[dataset]["metadata"]["short_name"]
            if dataset_key.startswith("Single"):
                new_dataset[dataset_key.split("_")[0]] += root_files
            else:
                new_dataset[dataset_key] = root_files

        # clean up temporary file
        os.remove(f"dataset_discovery_{args.year}.json")

    # Handle USER datasets with DAS
    if user_datasets:
        print(f"\nQuerying {len(user_datasets)} USER dataset(s) from DAS...")
        for dataset_key, query in user_datasets.items():
            print(f"Querying: {dataset_key}")
            files = query_das_user_dataset(query, instance=args.dbs_instance)
            if files:
                print(f"  Found {len(files)} files")
                # Prepend XRootD redirector for remote access.
                # USER (privately produced) datasets are NOT served by
                # cms-xrd-global -- it returns "[3011] No servers are available
                # to read the file". They live on the IIHE T2, which serves both
                # the cgupta and tvanlaer H+c/H+b productions for every era.
                new_dataset[dataset_key] = [
                    f"root://maite.iihe.ac.be:1094/{f}" for f in files
                ]
            else:
                print(f"  No files found for {dataset_key}")
                new_dataset[dataset_key] = []

    # save new fileset
    nano_version = get_nano_version(args.year)
    fileset_file = filesets_dir / f"fileset_{args.year}_nanov{nano_version}_lxplus.json"
    with open(fileset_file, "w") as json_file:
        json.dump(new_dataset, json_file, indent=4, sort_keys=True)

    print(f"\nFileset saved to: {fileset_file}")
    print(f"Total samples: {len(new_dataset)}")
    for key, files in new_dataset.items():
        print(f"  {key}: {len(files)} files")
