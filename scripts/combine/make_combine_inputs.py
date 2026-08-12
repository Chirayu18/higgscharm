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

import glob

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


def read_scale(sample, year, base_dir, lumi):
    """lumi*xsec/sumw for an MC/signal sample; 1.0 for data.

    SELF-NORMALISING (source #1): sumw comes from the per-chunk sumw_records written
    by dump_chunk_sumw on the PRE-selection events of every read chunk -- including
    chunks that select zero events and therefore write no data shard. This is the
    repo's own read_parquet_sumw() logic and is the correct generator sumw.

    Legacy samples produced before dump_chunk_sumw have no sumw_records; for those we
    fall back to the sidecar analysis/filesets/sumw_<year>.json. Every fallback is
    logged so the set is explicit.

    NOT USED: the per-shard schema metadata in parquets_<sample>/base/. It undercounts
    low-efficiency samples badly (WtoLNu_2Jets 5.4x, TbarQto2Q 72x) precisely because
    zero-selection chunks wrote no shard.
    """
    info = get_dataset_config(year).get(sample, {})
    era = info.get("era")
    if era not in ("mc", "signal"):
        return 1.0
    xsec = float(info["xsec"])

    import json, re as _re

    # --- source #1: sumw_records (self-normalising) ---
    rec_dirs = glob.glob(f"{base_dir}/{sample}_*/sumw_records") + glob.glob(
        f"{base_dir}/{sample}/sumw_records"
    )
    # guard against prefix collisions (DYto2L_2Jets_50 vs ..._50_ext)
    rec_dirs = [
        d for d in rec_dirs
        if _re.fullmatch(rf"{_re.escape(sample)}(_\d+)?", Path(d).parent.name)
    ]
    rec_files = [f for d in rec_dirs for f in glob.glob(f"{d}/*.parquet")]
    sumw = 0.0
    for f in rec_files:
        sumw += float(sum(pq.read_table(f, columns=["sumw"])["sumw"].to_pylist()))

    if sumw > 0:
        return lumi * xsec / sumw

    # --- fallback: sidecar, for legacy samples with no sumw_records ---
    sidecar = json.load(open(Path.cwd() / "analysis" / "filesets" / f"sumw_{year}.json"))
    sumw = sidecar.get(sample)
    if not sumw:
        raise ValueError(
            f"no sumw for MC sample {sample!r}: no sumw_records under "
            f"{base_dir}/{sample}*/sumw_records and not in sumw_{year}.json"
        )
    print(f"    [sumw] {sample}: no sumw_records -> sidecar fallback ({float(sumw):.4e})")
    return lumi * xsec / float(sumw)


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


def clip_negative_bins(proc_hists, channels, processes, variations, floor=1e-6):
    """Floor non-positive template bins to a tiny positive value, coordinated
    across the nominal and its systematic variations.

    Negative-weight NLO MC (here: low-stat vjets) can give negative bin contents,
    which combine cannot fit (a negative expected yield breaks the Poisson
    likelihood / gives a "Bogus norm"). For each (channel, process):
      * nominal bins <= floor are set to floor;
      * a systematic bin <= floor is set to floor;
      * crucially, wherever the *nominal* bin was floored, the matching
        systematic bin is forced to floor too, so kappa = 1 there instead of an
        exploding floor/large ratio that would crash text2workspace.

    Only bins that are actually non-positive change; well-populated templates are
    untouched. sumw2 (the stat error) is left as-is.
    """
    var_names = [v for v, _ in variations]
    n_clipped = 0
    for ch in channels:
        for cp in processes:
            nom_counts, nom_s2 = proc_hists[ch][cp]["nominal"]
            clipped = nom_counts <= floor
            n_clipped += int(clipped.sum())
            new_nom = np.where(clipped, floor, nom_counts)
            proc_hists[ch][cp]["nominal"] = (new_nom, nom_s2)
            for v in var_names:
                if v == "nominal":
                    continue
                c, s2 = proc_hists[ch][cp][v]
                c = np.where(c <= floor, floor, c)
                c = np.where(clipped, floor, c)  # nominal floored -> kappa = 1
                proc_hists[ch][cp][v] = (c, s2)
    return n_clipped


def smooth_shape_variations(proc_hists, channels, processes, variations,
                            nominal_name="nominal", frac=0.6):
    """Smooth each shape variation's bin-by-bin ratio to nominal (AN-23-102 7.2.1).

    Low-stat shape templates (esp. the 0.17-event signal, scalevar/ps_*, object
    shifts) have bin-to-bin MC-stat fluctuations that combine mistakes for real
    shape information -> artificial nuisance constraints that inflate the limit.
    We LOWESS-smooth the variation/nominal ratio over the populated bins and
    rescale to preserve the variation's total yield (the genuine rate effect is
    kept; only the shape noise is removed). Falls back to a 3-point moving
    average if statsmodels is unavailable.
    """
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        def _smooth(y, x):
            return lowess(y, x, frac=frac, return_sorted=False)
    except Exception:
        def _smooth(y, x):
            k = np.array([0.25, 0.5, 0.25])
            return np.convolve(np.pad(y, 1, mode="edge"), k, mode="valid")

    var_names = [v for v, _ in variations if v != nominal_name]
    n_sm = 0
    for ch in channels:
        for cp in processes:
            nom = proc_hists[ch][cp][nominal_name][0]
            if nom.max() <= 0:
                continue
            pop = nom > nom.max() * 1e-3
            if pop.sum() < 4:                  # too few bins to smooth meaningfully
                continue
            x = np.arange(len(nom))[pop].astype(float)
            for v in var_names:
                var, s2 = proc_hists[ch][cp][v]
                ratio = np.ones_like(nom)
                ratio[pop] = var[pop] / nom[pop]
                sm = ratio.copy()
                sm[pop] = _smooth(ratio[pop], x)
                new = sm * nom
                tot = new[pop].sum()
                if tot > 0:
                    new[pop] *= var[pop].sum() / tot   # preserve variation yield
                proc_hists[ch][cp][v] = (new, s2)
                n_sm += 1
    return n_sm


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


def process_sample(pq_path, sample, year, base_dir, classes, score_cols,
                   channels_by_class, variations, nbins, edges, lumi,
                   is_vjets=False, negrw_shape_name=None):
    """Returns {channel: {var_name: (counts, sumw2)}} for one sample.

    If is_vjets and a weight_negrw column is present, apply the neg-weight
    reweighting (arXiv:2510.16217): every fill weight w -> |w| * g, with
    g = weight_negrw = 2*P+(x)-1, then renormalise this sample's reweighted
    yield back to its nominal (Sum|w|g -> Sum w). This removes the SR MC-stat
    variance at the source WITHOUT changing the central yield, and REPLACES the
    DY-template smoothing (do not also smooth). g is a generator-level sign
    reweight, independent of the reco systematic, so it multiplies every
    variation. See Projects/HToWW/negrw-training/.
    """
    df = pd.read_parquet(pq_path)
    if len(df) == 0:
        return None
    missing = [c for c in score_cols if c not in df.columns]
    if missing:
        raise KeyError(f"{pq_path.name} missing score columns {missing}; "
                       f"run scripts/mva/run_inference.py first")

    scale = read_scale(sample, year, base_dir, lumi)
    scores = df[score_cols].to_numpy(dtype=np.float64)
    argmax = np.argmax(scores, axis=1)
    D = scores[np.arange(len(scores)), argmax]

    # argmax index -> channel name (class order == score_cols order)
    out = {ch: {} for ch in channels_by_class.values()}
    channel_idx = {channels_by_class[cls]: (argmax == i) for i, cls in enumerate(classes)}
    nominal_w = df["weight_nominal"].to_numpy(dtype=np.float64)

    use_negrw = is_vjets and ("weight_negrw" in df.columns)
    if use_negrw:
        g_negrw = df["weight_negrw"].to_numpy(dtype=np.float64)
        _sw = nominal_w.sum()
        _swg = (np.abs(nominal_w) * g_negrw).sum()
        negrw_renorm = (_sw / _swg) if _swg != 0 else 1.0
        logging.info(f"    [negrw] {sample}: renorm={negrw_renorm:.4f} "
                     f"(Sum w={_sw:.4g}, Sum|w|g={_swg:.4g})")

    for var_name, col in variations:
        if col == "__negrw__":
            continue   # negrw Up/Down handled in the dedicated block below
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
        if use_negrw:
            w = np.abs(w) * g_negrw * negrw_renorm   # |w|*g, yield-preserving
        w = w * scale
        for ch, mask in channel_idx.items():
            if mask.any():
                out[ch][var_name] = fill_hist(D[mask], w[mask], edges)
            else:
                out[ch][var_name] = (np.zeros(nbins), np.zeros(nbins))

    # neg-weight reweighting UNCERTAINTY (arXiv:2510.16217 sec IV): the 20-model
    # ensemble spread weight_negrw_std = 2*std(P+) -> a shape nuisance on vjets.
    # Up/Down = |w_nom| * clip(g +/- g_std, -1, 1), each renormalised to the nominal
    # yield so the nuisance is SHAPE-ONLY (rate stays = nominal, as for the other
    # shape systs here). Emitted for EVERY process so combine finds the template:
    # non-vjets get their nominal (Up==Down==nominal -> kappa 1, no effect).
    if negrw_shape_name:
        up_name, dn_name = f"{negrw_shape_name}Up", f"{negrw_shape_name}Down"
        if use_negrw and "weight_negrw_std" in df.columns:
            g_std = df["weight_negrw_std"].to_numpy(dtype=np.float64)
            for vn, gv in ((up_name, np.clip(g_negrw + g_std, -1.0, 1.0)),
                           (dn_name, np.clip(g_negrw - g_std, -1.0, 1.0))):
                swv = (np.abs(nominal_w) * gv).sum()
                rv = (nominal_w.sum() / swv) if swv != 0 else 1.0
                wv = np.abs(nominal_w) * gv * rv * scale
                for ch, mask in channel_idx.items():
                    out[ch][vn] = (fill_hist(D[mask], wv[mask], edges)
                                   if mask.any() else (np.zeros(nbins), np.zeros(nbins)))
        else:
            # non-vjets (or no std col): Up=Down=nominal so the shape row is a no-op.
            for vn in (up_name, dn_name):
                for ch in out:
                    out[ch][vn] = out[ch]["nominal"]
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


def write_datacard(datacard_path, root_name, combine, proc_hists, edges,
                   obj_shift_systs=None):
    channels = list(combine["channels"].keys())
    signal = combine["signal"]
    classes = combine["classes"]
    backgrounds = [c for c in classes if c != signal]
    processes = [signal] + backgrounds
    shape_systs = combine["shape_systematics"]
    no_scalevar = set(combine.get("no_scalevar", []))
    # processes whose normalization is taken from data via a rateParam (CR->SR):
    # drop their theory shape uncertainties (scalevar_*, ps_*), as in AN-23-102 for tt.
    no_theory = set(combine.get("no_theory", []))
    rate_params = combine.get("rate_params", [])
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
        is_theory = s.startswith("scalevar") or s.startswith("ps_")
        for ch, p in columns:
            if s.startswith("scalevar") and p in no_scalevar:
                row += fmt_cell("-", col_w)
            elif is_theory and p in no_theory:
                row += fmt_cell("-", col_w)
            else:
                row += fmt_cell("1", col_w)
        L.append(row.rstrip())

    # object-shift (kinematic) systematics: MC-wide shape, applies to all processes
    for s in (obj_shift_systs or []):
        row = f"{s:<{name_w-6}} shape "
        for ch, p in columns:
            row += fmt_cell("1", col_w)
        L.append(row.rstrip())

    # neg-weight reweighting uncertainty: shape nuisance on VJETS ONLY (the ensemble
    # spread of g). Templates {ch}_vjets_{name}{Up,Down} are in the ROOT; other
    # processes have Up==Down==nominal so "-" here keeps the row vjets-only.
    negrw_shape = combine.get("negrw_shape_name", "CMS_negrw_vjets")
    if combine.get("negrw_uncertainty", True) and "vjets" in processes:
        row = f"{negrw_shape:<{name_w-6}} shape "
        for ch, p in columns:
            row += fmt_cell("1" if p == "vjets" else "-", col_w)
        L.append(row.rstrip())

    L.append("-" * 100)
    for ch in channels:
        L.append(f"{ch} autoMCStats {auto_mc_stats}")
    # data-driven background normalizations: one shared parameter across all channels,
    # so the (background-pure) CR pins the rate and propagates it into the SR.
    for p in rate_params:
        L.append(f"rate_{p} rateParam * {p} 1.0 [0,5]")
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
    binning = combine["binning"]
    if binning.get("edges"):
        edges = np.array(binning["edges"], dtype=float)
        nbins = len(edges) - 1
    else:
        nbins = binning["nbins"]
        edges = np.linspace(binning["start"], binning["stop"], nbins + 1)
    score_cols = [f"mva_score_{c}" for c in classes]
    variations = build_variations(combine["shape_systematics"])

    # neg-weight reweighting uncertainty: one shape nuisance on vjets (built inside
    # process_sample from the 20-model ensemble spread weight_negrw_std). Enabled iff
    # the vjets templates are reweighted. Up/Down carry a sentinel col (not a real
    # weight column) -> handled specially in process_sample.
    NEGRW_SHAPE = combine.get("negrw_shape_name", "CMS_negrw_vjets")
    negrw_vars = [(f"{NEGRW_SHAPE}Up", "__negrw__"), (f"{NEGRW_SHAPE}Down", "__negrw__")]
    variations = variations + negrw_vars

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
            result = process_sample(pq_path, sample, args.year, base_dir, classes,
                                    score_cols, channels_by_class,
                                    variations, nbins, edges, lumi,
                                    is_vjets=(cp == "vjets"),
                                    negrw_shape_name=NEGRW_SHAPE)
            if result is None:
                continue
            for ch in channels:
                for v, hh in result[ch].items():
                    acc_c, acc_s2 = proc_hists[ch][cp][v]
                    proc_hists[ch][cp][v] = (acc_c + hh[0], acc_s2 + hh[1])
            logging.info(f"  [ok]  {cp:<10s} {sample}")

    # --- Object-shift (kinematic) systematics ----------------------------------
    # JES/JER/lepton-scale live in separate parquet dirs <syst>Up / <syst>Down with
    # SHIFTED kinematics -> shifted MVA scores -> events migrate argmax channels.
    # MC-only, weight_nominal only. We re-run the per-sample histogramming on each
    # shift dir's mva/ parquets (scored by run_inference) and add the templates as
    # extra datacard shape rows. A process with no shift parquet falls back to its
    # nominal template (systematic flat for it) so combine doesn't see a fake 100% shape.
    obj_shift_systs = []
    for d in sorted(glob.glob(str(base_dir / "*Up"))):
        nm = Path(d).name
        if nm.endswith("Up") and (Path(d) / "mva").is_dir() \
                and (base_dir / f"{nm[:-2]}Down" / "mva").is_dir():
            obj_shift_systs.append(nm[:-2])
    obj_shift_vars = [(f"{s}{dn}", "weight_nominal")
                      for s in obj_shift_systs for dn in ("Up", "Down")]
    for ch in channels:
        for cp in processes:
            for vn, _ in obj_shift_vars:
                proc_hists[ch][cp][vn] = (np.zeros(nbins), np.zeros(nbins))
    for vn, _ in obj_shift_vars:
        sdir = base_dir / vn / "mva"
        for cp in processes:
            for sample in combine_to_samples.get(cp, []):
                pq_path = sdir / f"{sample}.parquet"
                if not pq_path.exists():
                    continue
                res = process_sample(pq_path, sample, args.year, base_dir, classes,
                                     score_cols, channels_by_class,
                                     [(vn, "weight_nominal")], nbins, edges, lumi,
                                     is_vjets=(cp == "vjets"))
                if res is None:
                    continue
                for ch in channels:
                    ac, as2 = proc_hists[ch][cp][vn]
                    hh = res[ch][vn]
                    proc_hists[ch][cp][vn] = (ac + hh[0], as2 + hh[1])
    # fallback to nominal where a (channel, process) got no shift events
    for ch in channels:
        for cp in processes:
            nom_c, nom_s2 = proc_hists[ch][cp]["nominal"]
            for vn, _ in obj_shift_vars:
                c, _s2 = proc_hists[ch][cp][vn]
                if c.sum() <= 0 < nom_c.sum():
                    proc_hists[ch][cp][vn] = (nom_c.copy(), nom_s2.copy())
    variations = variations + obj_shift_vars
    logging.info(f"Folded {len(obj_shift_systs)} object-shift systematics: {obj_shift_systs}")

    # Floor non-positive template bins (negative-weight NLO MC, here low-stat
    # vjets) so combine can fit them; nominal+systematics clipped together.
    n_clip = clip_negative_bins(proc_hists, channels, processes, variations)
    logging.info(f"\nClipped {n_clip} non-positive nominal bins to floor")

    if combine.get("smooth_shapes"):
        n_sm = smooth_shape_variations(proc_hists, channels, processes, variations)
        logging.info(f"Smoothed {n_sm} shape variations (LOWESS, AN-23-102 7.2.1)")
        # re-floor in case smoothing produced sub-floor bins
        clip_negative_bins(proc_hists, channels, processes, variations)

    root_path = Path.cwd() / combine["output"]["root"]
    datacard_path = Path.cwd() / combine["output"]["datacard"]

    n = write_root(root_path, proc_hists, channels, processes, variations, edges)
    logging.info(f"\nWrote {n} TH1Ds -> {root_path}")

    obs = write_data_obs(root_path, proc_hists, channels, backgrounds, edges)
    logging.info("data_obs (Asimov bkg-only) per channel:")
    for ch in channels:
        logging.info(f"  {ch:<14s} {obs[ch]:>12.3f}")

    yields = write_datacard(datacard_path, Path(combine["output"]["root"]).name,
                            combine, proc_hists, edges,
                            obj_shift_systs=obj_shift_systs)
    logging.info(f"\nDatacard -> {datacard_path}")
    logging.info(f"\nPer-channel x per-process nominal yields:")
    logging.info("  " + "channel".ljust(14) + "".join(f"{p:>12s}" for p in processes) + f"{'total':>12s}")
    for ch in channels:
        total = sum(yields[(ch, p)] for p in processes)
        row = "  " + ch.ljust(14) + "".join(f"{yields[(ch, p)]:>12.3f}" for p in processes)
        logging.info(row + f"{total:>12.3f}")


if __name__ == "__main__":
    main()
