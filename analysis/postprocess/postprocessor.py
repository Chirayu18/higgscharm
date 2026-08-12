import copy
import yaml
import glob
import logging
import numpy as np
import pandas as pd
import dask.dataframe as dd
from pathlib import Path
from coffea.util import load, save
from coffea.processor import accumulate
from analysis.filesets.utils import get_dataset_config
from analysis.histograms import HistBuilder, fill_histogram
from analysis.postprocess.utils import (
    print_header,
    get_variations_keys,
    find_kin_and_axis,
    get_lumi_weight,
    accumulate_histograms,
    accumulate_metadata,
    save_cutflows,
    accumulate_and_save_cutflows,
)


def fill_histograms_from_parquets(
    year, sample, categories, workflow_config, output_dir
):
    """Build and fill histograms from parquet files for a given sample"""
    dataset_config = get_dataset_config(year)
    histogram_config = workflow_config.histogram_config
    variables = list(histogram_config.axes.keys())
    histograms = HistBuilder(workflow_config).build_histogram()

    for category in categories:
        logging.info(f"Filling {sample} histograms")

        # merge sample parquets
        sample_df_file = output_dir / f"{sample}.parquet"
        if sample_df_file.exists():
            sample_df = pd.read_parquet(sample_df_file)
        else:
            sample_parquets = glob.glob(
                f"{output_dir}/parquets_{sample}/{category}/*.parquet"
            )
            # skip empty partition parquets (can happen for low-stat selections);
            # fall back to the first partition only if every one is empty
            valid_parquets = [
                f for f in sample_parquets if len(pd.read_parquet(f)) > 0
            ]
            if not sample_parquets:
                # No shards at all: the selection kept zero events for this sample
                # (all jobs finished - .coffea markers present - they just wrote
                # nothing). Return correctly-shaped EMPTY histograms rather than
                # crashing on sample_parquets[0].
                logging.warning(
                    f"No partition parquets for {sample} [{category}]; "
                    f"selection kept zero events - returning empty histograms"
                )
                return copy.deepcopy(histograms)
            if not valid_parquets:
                logging.warning(
                    f"All partition parquets empty for {sample} [{category}]; "
                    f"using the first as a header-only fallback"
                )
                sample_df = pd.read_parquet(sample_parquets[0])
            else:
                sample_df = dd.read_parquet(
                    valid_parquets, engine="pyarrow", calculate_divisions=False
                ).compute()
            sample_df = sample_df.replace({None: np.nan})
            sample_df.to_parquet(
                f"{output_dir}/{sample}.parquet", engine="pyarrow", index=False
            )

        # build variables map
        variables_map = {}
        variables_mask_map = {}
        for variable in variables:
            if variable in sample_df.columns:
                variable_array = sample_df[variable].values
            else:
                logging.info(f"Could not find variable {variable} for sample {sample}")
            if variable_array.dtype.type is np.object_:
                variable_array = np.array(
                    [x if x is not None else np.nan for x in variable_array], dtype=bool
                )
            variables_map[variable] = variable_array

        # compute nominal weights
        # NOTE: weight_negrw / weight_negrw_std are DIAGNOSTIC OUTPUT COLUMNS written
        # by base.py::_score_negrw, not multiplicative weights - they are deliberately
        # never added to the weights container. weight_negrw = 2*P+(x)-1 spans -1..+1
        # and weight_negrw_std is an ensemble std-dev, so folding either into the
        # nominal product silently destroys the affected samples (DY, V+Jets were
        # suppressed ~156x). Exclude them here.
        partial_weights = list(
            set(
                [
                    w.replace("Up", "").replace("Down", "")
                    for w in sample_df.columns
                    if w.startswith("weight")
                    and "nominal" not in w
                    and "negrw" not in w
                ]
            )
        )
        nominal_weights = sample_df[partial_weights].prod(axis=1).values
        if len(partial_weights) > 0:
            logging.info(
                f"weights: {[w.replace('weight_','') for w in partial_weights]}"
            )

        # fill nominal histograms
        sample_histograms = copy.deepcopy(histograms)
        fill_args = {
            "histograms": sample_histograms,
            "histogram_config": histogram_config,
            "variables_map": variables_map,
            "category": category,
            "flow": True,
            "weights": nominal_weights,
            "variation": "nominal",
        }
        fill_histogram(**fill_args)

        # fill syst variation histograms
        if dataset_config[sample]["era"] in ["mc", "signal"]:
            for syst in partial_weights:
                for variation in ["Up", "Down"]:
                    syst_name = f"{syst}{variation}"
                    if syst_name in sample_df.columns:
                        fill_args["weights"] = sample_df[syst_name].values
                        fill_args["variation"] = syst_name.replace("weight_", "")
                        fill_histogram(**fill_args)

    return sample_histograms


def save_histograms_by_sample(
    grouped_outputs,
    sample,
    year,
    output_dir,
    categories,
    workflow_config,
    nocutflow,
    output_format,
    skipmerging,
):
    """Accumulate, scale, and save histograms for a single sample"""
    print_header(f"Processing {sample} outputs")

    # get histograms
    if output_format == "coffea":
        histograms = accumulate_histograms(grouped_outputs, sample)
    elif output_format == "parquet":
        histograms = fill_histograms_from_parquets(
            year, sample, categories, workflow_config, output_dir
        )
    else:
        raise ValueError(f"Unsupported output_format: {output_format}")

    # accumulate metadata (still used for cutflows) and compute lumi weight.
    # The weight's sumw now comes from the parquet metadata (see get_lumi_weight),
    # not metadata["sumw"], so a missing per-job .coffea no longer breaks it.
    metadata = accumulate_metadata(grouped_outputs, sample)
    weight = get_lumi_weight(year, sample, output_dir, categories)

    # scale histograms by lumi-xsec weight
    scaled_histograms = {
        variable: histograms[variable] * weight for variable in histograms
    }
    save(scaled_histograms, Path(output_dir) / f"{sample}.coffea")

    # save cutflows if requested
    if not nocutflow:
        save_cutflows(metadata, categories, sample, weight, output_dir)


def save_histograms_by_process(
    process: str,
    output_dir: str,
    process_samples_map: dict,
    categories: list,
    nocutflow: bool,
    output_format: str,
):
    """Accumulate and save all outputs for a given physics process."""
    print_header(f"Processing {process} outputs")

    # accumulate and save all histograms into a single dictionary
    coffea_files = []
    for sample in process_samples_map[process]:
        coffea_files += glob.glob(f"{output_dir}/{sample}.coffea", recursive=True)

    logging.info(f"Accumulating histograms for process {process}")
    hist_to_accumulate = [load(f) for f in coffea_files]
    output_histograms = {process: accumulate(hist_to_accumulate)}
    save(output_histograms, Path(output_dir) / f"{process}.coffea")

    # accumulate and save all parquets into a single parquet file
    if output_format == "parquet":
        logging.info(f"Accumulating parquets for process {process}")
        parquet_files = []
        for sample in process_samples_map[process]:
            parquet_files += glob.glob(
                f"{output_dir}/{sample}.parquet", recursive=True
            )
        # skip empty sample parquets; fall back to the first if all empty
        valid_parquets = [f for f in parquet_files if len(pd.read_parquet(f)) > 0]
        if not parquet_files:
            # No sample in this process wrote a parquet => every sample kept zero
            # events. Histograms/cutflows are already accumulated above; there is
            # simply nothing to concatenate, so skip <process>.parquet entirely
            # rather than crashing on parquet_files[0].
            logging.warning(
                f"No sample parquets for process {process}; all samples selected "
                f"zero events - skipping {process}.parquet"
            )
            process_df = None
        elif not valid_parquets:
            logging.warning(
                f"All sample parquets empty for process {process}; "
                f"using the first as a header-only fallback"
            )
            process_df = pd.read_parquet(parquet_files[0])
        else:
            process_df = dd.read_parquet(
                valid_parquets, engine="pyarrow", calculate_divisions=False
            ).compute()
        if process_df is not None:
            process_df.to_parquet(Path(output_dir) / f"{process}.parquet")

    # accumulate and save cutflows if requested
    if not nocutflow:
        accumulate_and_save_cutflows(
            process, process_samples_map, output_dir, categories
        )
