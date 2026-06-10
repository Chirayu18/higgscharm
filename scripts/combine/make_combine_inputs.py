"""Build combine input TH1Ds + datacard from scored parquets (yaml-driven).

Consumes the `combine:` block of the workflow yaml and the score-augmented
parquets written by scripts/mva/run_inference.py:

    outputs/<workflow>/<year>/<variation>/mva/<sample>.parquet

Each event is assigned to a channel by argmax over the mva_score_<class>
columns; the fit discriminant D in that channel is the winning class score.
Per-sample lumi*xsec/sumw scaling comes from the parquet's self-normalising
schema metadata (stamped by base.py / merge_parquets) — no coffea sidecar.

Weight-based shape systematics are read from the nominal parquet's
weight_<name>Up/Down columns. Output mirrors the legacy b-hive v3 builder:
6 channels x 6 processes x (nominal + 2*N_syst) TH1Ds, plus per-channel
Asimov bkg-only data_obs.

Usage:
  python scripts/combine/make_combine_inputs.py -w hww_MVA -y 2022postEE
"""

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import uproot
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.filesets.utils import get_dataset_config
from analysis.workflows.config import WorkflowConfigBuilder

OUTPUT_DIR = Path.cwd() / "outputs"
LUMI_FILE = Path.cwd() / "analysis" / "postprocess" / "luminosity.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-w", "--workflow", required=True,
        choices=[f.stem for f in (Path.cwd() / "analysis" / "workflows").glob("*.yaml")],
        help="workflow yaml name (must contain a 'combine:' block)",
    )
    parser.add_argument(
        "-y", "--year", required=True,
        choices=["2022preEE", "2022postEE", "2023preBPix", "2023postBPix"],
        help="data-taking year",
    )
    parser.add_argument(
        "--variation", default="nominal",
        help="which variation's mva/ parquets to histogram (default: nominal)",
    )
    return parser.parse_args()


def load_lumi(year):
    with open(LUMI_FILE) as f:
        return yaml.safe_load(f)[year]


def read_scale(parquet_path, lumi):
    """lumi*xsec/sumw from the parquet schema metadata; 1.0 for data."""
    md = pq.read_table(parquet_path).schema.metadata or {}
    era = md.get(b"era", b"").decode()
    if era not in ("mc", "signal"):
        return 1.0
    sumw = float(md[b"sumw"])
    xsec = float(md[b"xsec"])
    if sumw == 0:
        return 0.0
    return lumi * xsec / sumw


def gather_samples(year, process_map):
    """combine_process -> [sample, ...] for every MC/signal sample whose
    higgscharm `process` is listed under that combine process."""
    inverse = {}
    for combine_proc, hch_processes in process_map.items():
        for hch in hch_processes:
            inverse[hch] = combine_proc

    dataset_config = get_dataset_config(year)
    combine_to_samples = defaultdict(list)
    for sample, info in dataset_config.items():
        if info.get("era") not in ("mc", "signal"):
            continue
        combine_proc = inverse.get(info.get("process"))
        if combine_proc is not None:
            combine_to_samples[combine_proc].append(sample)
    return combine_to_samples


def build_variations(shape_systematics):
    variations = [("nominal", "weight_nominal")]
    for s in shape_systematics:
        variations.append((f"{s}Up", f"weight_{s}Up"))
        variations.append((f"{s}Down", f"weight_{s}Down"))
    return variations


def fill_hist(values, weights, edges):
    h, _ = np.histogram(values, bins=edges, weights=weights)
    h2, _ = np.histogram(values, bins=edges, weights=weights ** 2)
    return h, h2


def to_uproot_th1(counts, sumw2, edges, name, title=None):
    centers = 0.5 * (edges[:-1] + edges[1:])
    return uproot.writing.identify.to_TH1x(
        fName=name,
        fTitle=title or name,
        data=np.concatenate([[0.0], counts.astype(np.float64), [0.0]]),
        fEntries=float(counts.sum()),
        fTsumw=float(counts.sum()),
        fTsumw2=float(sumw2.sum()),
        fTsumwx=float(np.sum(centers * counts)),
        fTsumwx2=float(np.sum(centers ** 2 * counts)),
        fSumw2=np.concatenate([[0.0], sumw2.astype(np.float64), [0.0]]),
        fXaxis=uproot.writing.identify.to_TAxis(
            fName="xaxis",
            fTitle="D = P(argmax class)",
            fNbins=len(counts),
            fXmin=float(edges[0]),
            fXmax=float(edges[-1]),
            fXbins=edges.astype(np.float64),
        ),
    )


def process_sample(pq_path, classes, score_cols, channels_by_class, variations,
                   nbins, edges, lumi):
    """Returns {channel: {var_name: (counts, sumw2)}} for one sample."""
    df = pd.read_parquet(pq_path)
    if len(df) == 0:
        return None
    missing = [c for c in score_cols if c not in df.columns]
    if missing:
        raise KeyError(f"{pq_path.name} missing score columns {missing}; "
                       f"run scripts/mva/run_inference.py first")

    scale = read_scale(pq_path, lumi)
    scores = df[score_cols].to_numpy(dtype=np.float64)
    argmax = np.argmax(scores, axis=1)
    D = scores[np.arange(len(scores)), argmax]

    # argmax index -> channel name (class order == score_cols order)
    out = {ch: {} for ch in channels_by_class.values()}
    channel_idx = {channels_by_class[cls]: (argmax == i) for i, cls in enumerate(classes)}
    nominal_w = df["weight_nominal"].to_numpy(dtype=np.float64)
    for var_name, col in variations:
        w = (df[col].to_numpy(dtype=np.float64) if col in df.columns
             else nominal_w)
        # A non-finite systematic weight means the event has no info for that
        # variation (e.g. diboson MC produced without PSWeights -> NaN ps_isr/fsr).
        # Such events get no shift: fall back to their nominal weight, so the
        # Up/Down template stays finite (otherwise text2workspace dies on a
        # "Bogus norm nan" kappa).
        bad = ~np.isfinite(w)
        if bad.any():
            w = np.where(bad, nominal_w, w)
        w = w * scale
        for ch, mask in channel_idx.items():
            if mask.any():
                out[ch][var_name] = fill_hist(D[mask], w[mask], edges)
            else:
                out[ch][var_name] = (np.zeros(nbins), np.zeros(nbins))
    return out


def write_root(root_path, proc_hists, channels, processes, variations, edges):
    histograms = {}
    for ch in channels:
        for cp in processes:
            for var_name, _ in variations:
                counts, sumw2 = proc_hists[ch][cp][var_name]
                if var_name == "nominal":
                    hname = f"{ch}_{cp}"
                else:
                    hname = f"{ch}_{cp}_{var_name}"
                histograms[hname] = to_uproot_th1(counts, sumw2, edges, hname)
    root_path.parent.mkdir(parents=True, exist_ok=True)
    with uproot.recreate(str(root_path)) as f:
        for name, h in histograms.items():
            f[name] = h
    return len(histograms)


def write_data_obs(root_path, proc_hists, channels, backgrounds, edges):
    """Append per-channel Asimov bkg-only data_obs to the ROOT file."""
    data = {}
    for ch in channels:
        counts = np.sum([proc_hists[ch][b]["nominal"][0] for b in backgrounds], axis=0)
        sumw2 = np.sum([proc_hists[ch][b]["nominal"][1] for b in backgrounds], axis=0)
        data[ch] = (counts, sumw2)
    with uproot.update(str(root_path)) as f:
        for ch, (counts, sumw2) in data.items():
            f[f"{ch}_data_obs"] = to_uproot_th1(counts, sumw2, edges, f"{ch}_data_obs")
    return {ch: data[ch][0].sum() for ch in channels}


def fmt_cell(val, width):
    s = val if isinstance(val, str) else f"{val:.4f}"
    return s.ljust(max(width, len(s) + 1))


def write_datacard(datacard_path, root_name, combine, proc_hists, edges):
    channels = list(combine["channels"].keys())
    signal = combine["signal"]
    classes = combine["classes"]
    backgrounds = [c for c in classes if c != signal]
    processes = [signal] + backgrounds
    shape_systs = combine["shape_systematics"]
    no_scalevar = set(combine.get("no_scalevar", []))
    lnN = combine["lnN"]
    auto_mc_stats = combine.get("autoMCStats", 10)

    yields = {
        (ch, p): float(proc_hists[ch][p]["nominal"][0].sum())
        for ch in channels for p in processes
    }

    proc_ids = {signal: 0}
    for i, b in enumerate(backgrounds, start=1):
        proc_ids[b] = i

    name_w, col_w = 28, 14
    L = []
    L.append("# Datacard (HIG-24-018-style argmax channels), yaml-driven")
    L.append(f"# Built from {root_name}")
    L.append(f"imax {len(channels)}")
    L.append(f"jmax {len(backgrounds)}")
    L.append("kmax *")
    L.append("-" * 100)
    for ch in channels:
        L.append(f"shapes * {ch:<14s} {root_name} {ch}_$PROCESS {ch}_$PROCESS_$SYSTEMATIC")
    L.append("-" * 100)
    L.append("bin         " + "  ".join(f"{c:<14s}" for c in channels))
    L.append("observation " + "  ".join(f"{'-1':<14s}" for _ in channels))
    L.append("-" * 100)

    columns = [(ch, p) for ch in channels for p in processes]
    header = "bin".ljust(name_w) + " "
    proc_n = "process".ljust(name_w) + " "
    proc_i = "process".ljust(name_w) + " "
    rate_l = "rate".ljust(name_w) + " "
    for ch, p in columns:
        header += ch.ljust(col_w)
        proc_n += p.ljust(col_w)
        proc_i += str(proc_ids[p]).ljust(col_w)
        rate_l += f"{yields[(ch, p)]:.4f}".ljust(col_w)
    L += [header.rstrip(), proc_n.rstrip(), proc_i.rstrip(), rate_l.rstrip()]
    L.append("-" * 100)

    for name, mapping in lnN.items():
        row = f"{name:<{name_w-6}} lnN   "
        for ch, p in columns:
            row += fmt_cell(mapping.get(p, "-"), col_w)
        L.append(row.rstrip())

    for s in shape_systs:
        row = f"{s:<{name_w-6}} shape "
        for ch, p in columns:
            if s.startswith("scalevar") and p in no_scalevar:
                row += fmt_cell("-", col_w)
            else:
                row += fmt_cell("1", col_w)
        L.append(row.rstrip())

    L.append("-" * 100)
    for ch in channels:
        L.append(f"{ch} autoMCStats {auto_mc_stats}")
    L.append("")

    datacard_path.parent.mkdir(parents=True, exist_ok=True)
    datacard_path.write_text("\n".join(L))
    return yields


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = WorkflowConfigBuilder(workflow=args.workflow).build_workflow_config()
    if cfg.combine is None:
        sys.exit(f"workflow {args.workflow} has no 'combine:' block in its yaml")
    combine = cfg.combine

    classes = combine["classes"]
    signal = combine["signal"]
    backgrounds = [c for c in classes if c != signal]
    processes = [signal] + backgrounds
    channels_by_class = {cls: ch for ch, cls in combine["channels"].items()}
    channels = list(combine["channels"].keys())
    process_map = combine["process_map"]
    nbins = combine["binning"]["nbins"]
    edges = np.linspace(combine["binning"]["start"], combine["binning"]["stop"], nbins + 1)
    score_cols = [f"mva_score_{c}" for c in classes]
    variations = build_variations(combine["shape_systematics"])

    lumi = load_lumi(args.year)

    base_dir = OUTPUT_DIR / args.workflow / args.year
    var_dir = base_dir / args.variation if (base_dir / args.variation).exists() else base_dir
    mva_dir = var_dir / "mva"
    if not mva_dir.exists():
        sys.exit(f"scored parquets not found: {mva_dir} "
                 f"(run scripts/mva/run_inference.py first)")

    combine_to_samples = gather_samples(args.year, process_map)

    # accumulator: proc_hists[channel][combine_proc][var_name] = (counts, sumw2)
    proc_hists = {
        ch: {cp: {v: (np.zeros(nbins), np.zeros(nbins)) for v, _ in variations}
             for cp in processes}
        for ch in channels
    }

    for cp in processes:
        for sample in combine_to_samples.get(cp, []):
            pq_path = mva_dir / f"{sample}.parquet"
            if not pq_path.exists():
                logging.info(f"  [skip] {cp}/{sample}: no {pq_path.name}")
                continue
            result = process_sample(pq_path, classes, score_cols, channels_by_class,
                                    variations, nbins, edges, lumi)
            if result is None:
                continue
            for ch in channels:
                for v, hh in result[ch].items():
                    acc_c, acc_s2 = proc_hists[ch][cp][v]
                    proc_hists[ch][cp][v] = (acc_c + hh[0], acc_s2 + hh[1])
            logging.info(f"  [ok]  {cp:<10s} {sample}")

    root_path = Path.cwd() / combine["output"]["root"]
    datacard_path = Path.cwd() / combine["output"]["datacard"]

    n = write_root(root_path, proc_hists, channels, processes, variations, edges)
    logging.info(f"\nWrote {n} TH1Ds -> {root_path}")

    obs = write_data_obs(root_path, proc_hists, channels, backgrounds, edges)
    logging.info("data_obs (Asimov bkg-only) per channel:")
    for ch in channels:
        logging.info(f"  {ch:<14s} {obs[ch]:>12.3f}")

    yields = write_datacard(datacard_path, Path(combine["output"]["root"]).name,
                            combine, proc_hists, edges)
    logging.info(f"\nDatacard -> {datacard_path}")
    logging.info(f"\nPer-channel x per-process nominal yields:")
    logging.info("  " + "channel".ljust(14) + "".join(f"{p:>12s}" for p in processes) + f"{'total':>12s}")
    for ch in channels:
        total = sum(yields[(ch, p)] for p in processes)
        row = "  " + ch.ljust(14) + "".join(f"{yields[(ch, p)]:>12.3f}" for p in processes)
        logging.info(row + f"{total:>12.3f}")


if __name__ == "__main__":
    main()
