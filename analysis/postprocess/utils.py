import glob
import logging
import os
import pickle
from collections import defaultdict
from pathlib import Path

import dask.dataframe as dd
import hist
import numpy as np
import pandas as pd
import yaml
from coffea.processor import accumulate
from coffea.util import load, save

from analysis.filesets.utils import get_dataset_config


def setup_logger(output_dir):
    """Set up the logger to log to a file in the specified output directory."""
    output_file_path = os.path.join(output_dir, "output.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.FileHandler(output_file_path), logging.StreamHandler()],
    )


def open_output(fname: str) -> dict:
    with open(fname, "rb") as f:
        output = pickle.load(f)
    return output


def print_header(text):
    logging.info("-" * 90)
    logging.info(text)
    logging.info("-" * 90)


def divide_by_binwidth(histogram):
    bin_width = histogram.axes.edges[0][1:] - histogram.axes.edges[0][:-1]
    return histogram / bin_width


def clear_output_directory(output_dir, ext):
    """Delete all result files in the output directory with extension 'ext'"""
    files = glob.glob(os.path.join(output_dir, f"*.{ext}"))
    for file in files:
        os.remove(file)


def combine_event_tables(df1, df2, blind):
    assert all(df1.index == df2.index), "DataFrame index don't match!"
    combined = pd.DataFrame(index=df1.index)
    combined["events"] = df1["events"] + df2["events"]
    combined["stat err"] = np.sqrt(df1["stat err"] ** 2 + df2["stat err"] ** 2)
    combined["syst err up"] = np.sqrt(df1["syst err up"] ** 2 + df2["syst err up"] ** 2)
    combined["syst err down"] = np.sqrt(
        df1["syst err up"] ** 2 + df2["syst err down"] ** 2
    )
    if not blind:
        data = combined.loc["Data", "events"]
        total_bkg = combined.loc["Total background", "events"]
        combined.loc[
            "Data/Total background",
            ["events", "stat err", "syst err up", "syst err down"],
        ] = [
            data / total_bkg,
            np.nan,
            np.nan,
            np.nan,
        ]
    return combined


def combine_cutflows(df1, df2):
    if not df1.index.equals(df2.index):
        raise ValueError("Los índices (etiquetas de cortes) no coinciden.")
    combined = df1.add(df2, fill_value=0)
    return combined


def format_cutflow_with_efficiency(events_df, eff_df):
    combined = pd.DataFrame(index=events_df.index, columns=events_df.columns)
    for col in events_df.columns:
        for idx in events_df.index:
            val = events_df.loc[idx, col]
            eff = eff_df.loc[idx, col]
            combined.loc[idx, col] = f"{val} ({eff:.2f}%)"
    return combined


def get_variations_keys(processed_histograms):
    variations = {}
    for process, histogram_dict in processed_histograms.items():
        if process == "Data":
            continue
        for feature in histogram_dict:
            helper_histogram = histogram_dict[feature]
            variations = [
                var for var in helper_histogram.axes["variation"] if var != "nominal"
            ]
            break
        break
    variations = list(
        set([var.replace("Up", "").replace("Down", "") for var in variations])
    )
    return variations


def find_kin_and_axis(processed_histograms, name="multiplicity"):
    for process, histogram_dict in processed_histograms.items():
        if process == "Data":
            continue
        for kin, hist in histogram_dict.items():
            for axis_name in hist.axes.name:
                if axis_name != "variation" and name in axis_name:
                    return kin, axis_name
    raise ValueError(f"No histogram with a '{name}' axis found.")


def merge_parquet_files(input_files, output_file):
    """Concatenate parquet shards and aggregate self-normalising metadata.

    Per-shard parquets carry sumw/xsec/era in their pyarrow schema metadata
    (stamped by dump_parquet). pyarrow is used directly (not dask) so the
    schema metadata survives the round-trip; sumw is summed across shards,
    xsec/era pass through.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    input_files = sorted(input_files)
    if not input_files:
        return False

    tables = [pq.read_table(f) for f in input_files]
    total_sumw, have_sumw, xsec_b, era_b = 0.0, False, None, None
    for t in tables:
        md = t.schema.metadata or {}
        if b"sumw" in md:
            try:
                total_sumw += float(md[b"sumw"])
                have_sumw = True
            except ValueError:
                pass
        if xsec_b is None and md.get(b"xsec"):
            xsec_b = md[b"xsec"]
        if era_b is None and md.get(b"era"):
            era_b = md[b"era"]

    merged = pa.concat_tables(tables, promote_options="default")
    new_md = dict(merged.schema.metadata or {})
    if have_sumw:
        new_md[b"sumw"] = str(total_sumw).encode()
    if xsec_b is not None:
        new_md[b"xsec"] = xsec_b
    if era_b is not None:
        new_md[b"era"] = era_b
    merged = merged.replace_schema_metadata(new_md)

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(merged, output_file)
    return True


def merge_parquets(inpath, outpath, sample_name):
    outpath = Path(outpath)
    files = sorted(Path(inpath).glob("*.parquet"))
    merge_parquet_files(files, outpath / f"{sample_name}.parquet")


def accumulate_metadata(grouped_outputs, sample):
    grouped_metadata = {}
    for fname in grouped_outputs[sample]:
        output = load(fname)
        if not output:
            continue
        for meta_key, meta_val in output["metadata"].items():
            grouped_metadata.setdefault(meta_key, []).append(meta_val)
    return {k: accumulate(v) for k, v in grouped_metadata.items()}


def accumulate_histograms(grouped_outputs, sample):
    grouped_histograms = []
    for fname in grouped_outputs[sample]:
        output = load(fname)
        if output:
            grouped_histograms.append(output["histograms"])
    return accumulate(grouped_histograms)


def read_parquet_sumw(output_dir, sample, category):
    """Total generator sumw for a sample, read from the self-normalising parquet
    metadata.

    Each merged shard stores its read-chunk's pre-selection sumw in the parquet
    schema metadata (the additive self-normalising scheme), so the dataset total
    is just their sum over one category. This replaces the per-job .coffea
    metadata['sumw']: resubmitted jobs can land their shards without a cutflow
    .coffea, and keying normalisation off the parquet metadata makes it robust to
    that (and matches the .coffea sumw to ~1% on samples that have both).
    """
    import pyarrow.parquet as pq

    total = 0.0
    shards = glob.glob(f"{output_dir}/parquets_{sample}/{category}/*.parquet")
    for f in shards:
        meta = pq.ParquetFile(f).schema_arrow.metadata
        if meta and meta.get(b"sumw") is not None:
            total += float(meta[b"sumw"])
    return total


def get_lumi_weight(year, sample, output_dir, categories):
    lumi_file = Path.cwd() / "analysis" / "postprocess" / "luminosity.yaml"
    with open(lumi_file, "r") as f:
        luminosities = yaml.safe_load(f)

    dataset_config = get_dataset_config(year)
    xsec = dataset_config[sample]["xsec"]
    weight = 1
    sumw = None
    if dataset_config[sample]["era"] in ["mc", "signal"]:
        # sumw from the parquet metadata (primary category), not the .coffea
        sumw = read_parquet_sumw(output_dir, sample, list(categories)[0])
        if not sumw:
            raise ValueError(
                f"zero/missing parquet sumw for MC sample {sample!r} in "
                f"{output_dir}; cannot normalise"
            )
        weight = (luminosities[year] * xsec) / sumw

    logging.info(f"luminosity [1/pb]: {luminosities[year]}")
    logging.info(f"xsec [pb]: {xsec}")
    logging.info(f"sumw (parquet): {sumw}")
    logging.info(f"weight: {weight}")

    return weight


def save_cutflows(metadata, categories, sample, weight, output_dir):
    for category in categories:
        if category not in metadata:
            logging.info(f"zero events for category {category}")
            continue

        logging.info(f"saving {sample} cutflow for category {category}")

        category_dir = Path(output_dir) / category
        if not category_dir.exists():
            category_dir.mkdir(parents=True, exist_ok=True)

        scaled_cutflow = {
            cut: nevents * weight
            for cut, nevents in metadata[category]["cutflow"].items()
        }
        processed_cutflow = {sample: scaled_cutflow}
        cutflow_file = category_dir / f"cutflow_{category}_{sample}.coffea"
        save(processed_cutflow, cutflow_file)


def get_process_dict(output_dir, year, categories):
    folders = glob.glob(str(output_dir / "*"))
    process_dict = defaultdict(list)

    dataset_config = get_dataset_config(year)

    for folder in folders:
        folder_path = Path(folder)
        name = folder_path.name
        for category in categories:
            if name == category:
                continue
            if category not in process_dict:
                process_dict[category] = {}

            folder_content = glob.glob(f"{folder_path}/*")

            # case where there are multiple partitions and the folder includes only .coffea files
            if any(".coffea" in f for f in folder_content) and not any(
                category in f for f in folder_content
            ):
                if name not in process_dict[category]:
                    process_dict[category][name] = []
            # case where there is only one partition and the main folder includes both the .coffea file and the category folder
            elif any(".coffea" in f for f in folder_content) and any(
                category in f for f in folder_content
            ):
                for f in folder_content:
                    if category in f:
                        if name not in process_dict[category]:
                            process_dict[category][name] = [str(folder_path)]
            # case where there's only the category folder (no per-job .coffea).
            # Discover the sample from the parquet shards + dataset_config rather
            # than relying on a sibling .coffea key already existing: resubmitted
            # jobs can land their shards without the cutflow .coffea, which would
            # otherwise silently drop the whole sample from the merge.
            else:
                if not glob.glob(f"{folder_path}/{category}/*.parquet"):
                    continue
                actual_name = (
                    name if name in dataset_config else name.rsplit("_", 1)[0]
                )
                if actual_name not in dataset_config:
                    continue
                process_dict[category].setdefault(actual_name, [])
                process_dict[category][actual_name].append(str(folder_path))

    return dict(process_dict)


def merge_parquets_by_sample(output_dir, year, categories):
    """Merge parquet files from subfolders into a single output path per sample and category."""
    print_header("Merging parquet outputs by sample")
    process_dict = get_process_dict(output_dir, year, categories)
    for category in categories:
        for name, subfolders in process_dict[category].items():
            outpath = f"{output_dir}/parquets_{name}/{category}"
            if len(subfolders) != 0:
                logging.info(
                    f"Merging {name} outputs into {len(subfolders)} "
                    f"{'partition' if len(subfolders) == 1 else 'partitions'}"
                )
                for i, subfolder in enumerate(subfolders, start=1):
                    inpath = f"{subfolder}/{category}"
                    merge_parquets(inpath, outpath, f"{name}_{i}")


def merge_shifted_parquets_by_sample(output_dir, year, categories):
    """Merge object-shift parquets into <output_dir>/<shift>/<sample>.parquet.

    The runner (object_shifts: true) writes shifted kinematics to
        <output_dir>/<dataset>/<category>/<shift>/<partition>.parquet
    one level deeper than nominal. The nominal merge globs *.parquet
    non-recursively, so it ignores these; this produces the per-sample,
    per-shift merge the inference/combine path expects.

    Only the primary (first) category is merged — mirroring the nominal
    per-sample parquet, which is built from the first category
    (see fill_histograms_from_parquets). Combine reads that inclusive
    selection and channelises by argmax, so the signal-region category
    is not merged here.
    """
    output_dir = Path(output_dir)
    categories = list(categories)
    if not categories:
        return
    primary = categories[0]
    dataset_config = get_dataset_config(year)

    # group partition folders (named <dataset>_<i>) back to their sample
    shift_sample_files = defaultdict(list)
    for dataset_dir in sorted(output_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        sample = dataset_dir.name
        if sample not in dataset_config:
            stripped = sample.rsplit("_", 1)[0]  # drop the _<i> partition suffix
            if stripped in dataset_config:
                sample = stripped
            else:
                continue
        # object-shift systematics are MC-only; data is the observation and
        # carries no scale/resolution variation. The processor still emits
        # lepton-shifted data parquets, so skip them here to keep the shift
        # templates MC-only (otherwise the datacard builder would create bogus
        # data variations).
        if dataset_config[sample].get("era") not in ("mc", "signal"):
            continue
        shift_root = dataset_dir / primary
        if not shift_root.exists():
            continue
        for shift_dir in sorted(shift_root.iterdir()):
            if not shift_dir.is_dir():
                continue
            files = sorted(str(p) for p in shift_dir.glob("*.parquet"))
            if files:
                shift_sample_files[(shift_dir.name, sample)].extend(files)

    if not shift_sample_files:
        return
    print_header("Merging object-shift parquet outputs by sample")
    for (shift, sample), files in sorted(shift_sample_files.items()):
        out = output_dir / shift / f"{sample}.parquet"
        logging.info(f"Merging {sample} [{shift}] ({len(files)} partitions) -> {out}")
        merge_parquet_files(files, out)


def accumulate_and_save_cutflows(process, process_samples_map, output_dir, categories):
    """Accumulate cutflows from all samples in a process and save them per category."""
    for category in categories:
        category_dir = Path(f"{output_dir}/{category}")
        df_total = pd.DataFrame()

        for sample in process_samples_map[process]:
            cutflow_file = category_dir / f"cutflow_{category}_{sample}.coffea"
            if cutflow_file.exists():
                df_sample = pd.DataFrame(load(cutflow_file).values())
                df_total = pd.concat([df_total, df_sample])

        if not df_total.empty:
            cutflow_df = pd.DataFrame(df_total.sum())
            cutflow_df.columns = [process]
            cutflow_csv = category_dir / f"cutflow_{category}_{process}.csv"
            logging.info(
                f"Saving cutflow for process {process} and category {category}"
            )
            cutflow_df.to_csv(cutflow_csv)


def load_processed_histograms(
    year: str,
    output_dir: str,
    process_samples_map: dict,
):
    processed_histograms = {}
    for process in process_samples_map:
        processed_histograms.update(load(f"{output_dir}/{process}.coffea"))
    save(processed_histograms, f"{output_dir}/{year}_processed_histograms.coffea")
    return processed_histograms


def get_results_report(
    processed_histograms, workflow_config, category, columns_to_drop, blind
):
    kin, aux_var = find_kin_and_axis(processed_histograms)
    nominal = {}
    variations = {}
    mcstat_err = {}
    bin_error_up = {}
    bin_error_down = {}
    for process in processed_histograms:
        aux_hist = processed_histograms[process][kin]
        nominal_selector = {"variation": "nominal"}
        if "category" in aux_hist.axes.name:
            nominal_selector["category"] = category
        nominal_hist = aux_hist[nominal_selector].project(aux_var)
        nominal[process] = nominal_hist

        mcstat_err[process] = {}
        bin_error_up[process] = {}
        bin_error_down[process] = {}
        mcstat_err2 = nominal_hist.variances()
        mcstat_err[process] = np.sum(np.sqrt(mcstat_err2))
        err2_up = mcstat_err2
        err2_down = mcstat_err2

        if process == "Data":
            continue

        for variation in get_variations_keys(processed_histograms):
            if f"{variation}Up" not in aux_hist.axes["variation"]:
                continue
            selectorup = {"variation": f"{variation}Up"}
            selectordown = {"variation": f"{variation}Down"}
            if "category" in aux_hist.axes.name:
                selectorup["category"] = category
                selectordown["category"] = category
            var_up = aux_hist[selectorup].project(aux_var).values()
            var_down = aux_hist[selectordown].project(aux_var).values()
            # Compute the uncertainties corresponding to the up/down variations
            err_up = var_up - nominal_hist.values()
            err_down = var_down - nominal_hist.values()
            # Compute the flags to check which of the two variations (up and down) are pushing the nominal value up and down
            up_is_up = err_up > 0
            down_is_down = err_down < 0
            # Compute the flag to check if the uncertainty is one-sided, i.e. when both variations are up or down
            is_onesided = up_is_up ^ down_is_down
            # Sum in quadrature of the systematic uncertainties taking into account if the uncertainty is one- or double-sided
            err2_up_twosided = np.where(up_is_up, err_up**2, err_down**2)
            err2_down_twosided = np.where(up_is_up, err_down**2, err_up**2)
            err2_max = np.maximum(err2_up_twosided, err2_down_twosided)
            err2_up_onesided = np.where(is_onesided & up_is_up, err2_max, 0)
            err2_down_onesided = np.where(is_onesided & down_is_down, err2_max, 0)
            err2_up_combined = np.where(is_onesided, err2_up_onesided, err2_up_twosided)
            err2_down_combined = np.where(
                is_onesided, err2_down_onesided, err2_down_twosided
            )
            # Sum in quadrature of the systematic uncertainty corresponding to a MC sample
            err2_up += err2_up_combined
            err2_down += err2_down_combined

        bin_error_up[process] = np.sum(np.sqrt(err2_up))
        bin_error_down[process] = np.sum(np.sqrt(err2_down))

    mcs = []
    results = {}
    for process in nominal:
        results[process] = {}
        results[process]["events"] = np.sum(nominal[process].values())
        if process == "Data":
            results[process]["stat err"] = np.sqrt(np.sum(nominal[process].values()))
        else:
            if process not in columns_to_drop:
                mcs.append(process)
            results[process]["stat err"] = mcstat_err[process]
            results[process]["syst err up"] = bin_error_up[process]
            results[process]["syst err down"] = bin_error_down[process]
    df = pd.DataFrame(results)
    df["Total background"] = df.loc[["events"], mcs].sum(axis=1)
    df.loc["stat err", "Total background"] = np.sqrt(
        np.sum(df.loc["stat err", mcs] ** 2)
    )
    df.loc["syst err up", "Total background"] = np.sqrt(
        np.sum(df.loc["syst err up", mcs] ** 2)
    )
    df.loc["syst err down", "Total background"] = np.sqrt(
        np.sum(df.loc["syst err down", mcs] ** 2)
    )
    df = df.T
    if not blind:
        df.loc["Data/Total background"] = (
            df.loc["Data", ["events"]] / df.loc["Total background", ["events"]]
        )
    return df


def df_to_latex(df, blind, table_title="Events"):
    output = rf"""\begin{{table}}[h!]
\centering
\begin{{tabular}}{{@{{}} l c @{{}}}}
\hline
 & \textbf{{{table_title}}} \\
\hline
"""

    total_background_value = None
    data_value = None

    for label, row in df.iterrows():
        events = row["events"]
        stat_err = row["stat err"] if pd.notna(row["stat err"]) else None
        syst_err_up = row["syst err up"] if pd.notna(row["syst err up"]) else None
        syst_err_down = row["syst err down"] if pd.notna(row["syst err down"]) else None

        events_f = f"{float(events):.2f}"
        stat_err_f = f"{float(stat_err):.2f}" if stat_err is not None else "nan"
        syst_err_up_f = (
            f"{float(syst_err_up):.2f}" if syst_err_up is not None else "nan"
        )
        syst_err_down_f = (
            f"{float(syst_err_down):.2f}" if syst_err_down is not None else "nan"
        )

        if label not in ["Data", "Total background", "Data/Total background"]:
            output += f"{label} & ${events_f} \\pm {stat_err_f} \\, (\\text{{stat}}) \\pm {syst_err_up_f} \\, (\\text{{syst up}})$ \\pm {syst_err_down_f} \\, (\\text{{syst down}})$\\\\\n"

    output += r"\hline" + "\n"

    # Total background
    bg = df.loc["Total background"]
    total_background_value = bg["events"]
    stat_err_bg = bg["stat err"]
    syst_err_up_bg = bg["syst err up"]
    syst_err_down_bg = bg["syst err down"]

    output += f"Total Background & ${float(total_background_value):.2f} \\pm {float(stat_err_bg):.2f} \\, (\\text{{stat}}) \\pm {float(syst_err_up_bg):.2f} \\, (\\text{{syst up}})$ \\pm {float(syst_err_down_bg):.2f} \\, (\\text{{syst down}})$ \\\\ \n"

    # Data
    if not blind:
        data_value = df.loc["Data"]["events"]
        output += f"Data & ${float(data_value):.0f}$ \\\\ \n"

        output += r"\hline" + "\n"

        # Data/Total Background
        if total_background_value and data_value:
            ratio = data_value / total_background_value
            output += f"Data/Total Background & ${ratio:.2f}$ \\\\ \n"

    output += r"""\hline
\end{tabular}
\end{table}"""
    return output
