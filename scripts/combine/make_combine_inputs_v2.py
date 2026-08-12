#!/usr/bin/env python3
"""Card builder v2: per-channel binning + datacard-process/MVA-class decoupling.

A THIN WRAPPER around scripts/combine/make_combine_inputs.py. It imports that module
and reuses its functions unchanged, overriding only the two behaviours hardcoded there.
The original script is NOT modified, so the r < 1160 result stays exactly reproducible
by running it.

Both features are opt-in via the workflow yaml. With NEITHER key present this produces
a byte-identical card to the original builder (verified by diffing the outputs).

------------------------------------------------------------------------------
1. PER-CHANNEL BINNING   ->   combine.binning.per_channel
------------------------------------------------------------------------------
The original uses ONE global `combine.binning.edges` for all six channels
(make_combine_inputs.py L503-509).

Motivation: AN-23-102 line 662 makes the top CR deliberately yield-only, and AN-24-091
Table 10 (Run 3 HH->bbWW) makes EVERY control region a single bin while the SRs keep
10/6/3. Our CR_tt is 87.9% pure over 1.56M events -- there is no useful shape there, and
a 10-bin shape in an argmax-defined CR is exactly the artificial-constraint failure mode
described in AN-23-102 section 7.2.1. It also removes the argmax channel-migration
artifact from that channel (see 2026-08-11-jes-jer-bug-fixed).

    combine:
      binning:
        edges: [0.0, 0.2, 0.3, ...]     # default for any channel not listed
        per_channel:
          CR_tt: [0.0, 1.0]             # 1 bin -> yield-only

------------------------------------------------------------------------------
2. PROCESS / CLASS DECOUPLING   ->   combine.processes
------------------------------------------------------------------------------
The original derives datacard processes from `combine.classes`, the MVA output classes
(L496-499), so a datacard row is forced to BE an argmax class. Any `process_map` key
that is not a class is silently ignored, which SILENTLY DELETES those samples.

Verified 2026-08-11: adding `ggH: [ggH, ggZH]` to process_map dropped the SR total
20664 -> 20561 (~103 events), took higgsbkg 131.09 -> 28.33, and turned the
flavor_composition_ggH row into all dashes -- with the build still exiting 0.

Channels must stay one-per-class (that is what argmax means), but a datacard PROCESS is
just a row and need not be a class. This lets ggH be split out of higgsbkg WITHOUT
retraining:

    combine:
      processes: [hplusc, ggH, higgsbkg, tt, st, diboson, vjets]
      process_map:
        ggH:      [ggH, ggZH]
        higgsbkg: [H+b, VBF, ZH, ttHnonBB, ttHtoBB]
        ...
      lnN:
        flavor_composition_ggH: {ggH: 1.50}   # the AN's real 50%, on ggH alone

VALIDATION turns the silent-drop bug into a hard error: every declared process must have
a process_map entry, and every process_map key must be a declared process.

Usage (identical to the original):
    python3 scripts/combine/make_combine_inputs_v2.py -w hww_combine_2dcat -y 2022postEE
"""
import glob
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_combine_inputs as base  # noqa: E402


# ----------------------------------------------------------------------------------
# configuration resolution
# ----------------------------------------------------------------------------------
def resolve_processes(combine):
    """Datacard processes. Falls back to the original `classes` behaviour."""
    classes = combine["classes"]
    signal = combine["signal"]
    process_map = combine["process_map"]

    declared = combine.get("processes")
    if not declared:
        return [signal] + [c for c in classes if c != signal]

    missing = [p for p in declared if p not in process_map]
    if missing:
        raise KeyError(
            f"combine.processes lists {missing} with no process_map entry -- those "
            f"datacard rows would be empty."
        )
    orphan = [k for k in process_map if k not in declared]
    if orphan:
        raise KeyError(
            f"process_map has keys {orphan} absent from combine.processes -- their "
            f"samples would be SILENTLY DROPPED (the 2026-08-11 ggH bug). Add them to "
            f"combine.processes."
        )
    if signal not in declared:
        raise KeyError(f"combine.signal '{signal}' is missing from combine.processes")
    return [signal] + [p for p in declared if p != signal]


def resolve_binning(combine, channels):
    """{channel: edges}, per-channel overrides on top of the global default."""
    binning = combine["binning"]
    if binning.get("edges"):
        default = np.array(binning["edges"], dtype=float)
    else:
        default = np.linspace(binning["start"], binning["stop"], binning["nbins"] + 1)

    per_channel = binning.get("per_channel") or {}
    unknown = [ch for ch in per_channel if ch not in channels]
    if unknown:
        raise KeyError(f"binning.per_channel names unknown channels {unknown}; "
                       f"valid: {channels}")

    out = {}
    for ch in channels:
        e = np.array(per_channel[ch], dtype=float) if ch in per_channel else default
        if len(e) < 2:
            raise ValueError(f"channel {ch}: need >=2 bin edges, got {list(e)}")
        out[ch] = e
    return out


# ----------------------------------------------------------------------------------
# per-channel-edges variants of the base writers
# ----------------------------------------------------------------------------------
def write_root_v2(root_path, proc_hists, channels, processes, variations, edges_by_ch):
    histograms = {}
    for ch in channels:
        edges = edges_by_ch[ch]
        for cp in processes:
            for var_name, _ in variations:
                counts, sumw2 = proc_hists[ch][cp][var_name]
                hname = f"{ch}_{cp}" if var_name == "nominal" else f"{ch}_{cp}_{var_name}"
                histograms[hname] = base.to_uproot_th1(counts, sumw2, edges, hname)
    root_path.parent.mkdir(parents=True, exist_ok=True)
    with base.uproot.recreate(str(root_path)) as f:
        for name, h in histograms.items():
            f[name] = h
    return len(histograms)


def write_data_obs_v2(root_path, proc_hists, channels, backgrounds, edges_by_ch):
    data = {}
    for ch in channels:
        counts = np.sum([proc_hists[ch][b]["nominal"][0] for b in backgrounds], axis=0)
        sumw2 = np.sum([proc_hists[ch][b]["nominal"][1] for b in backgrounds], axis=0)
        data[ch] = (counts, sumw2)
    with base.uproot.update(str(root_path)) as f:
        for ch, (counts, sumw2) in data.items():
            f[f"{ch}_data_obs"] = base.to_uproot_th1(
                counts, sumw2, edges_by_ch[ch], f"{ch}_data_obs"
            )
    return {ch: data[ch][0].sum() for ch in channels}


def write_datacard_v2(datacard_path, root_name, combine, proc_hists, processes,
                      obj_shift_systs=None):
    """base.write_datacard, but with the process list passed in rather than
    re-derived from combine.classes. Everything else is delegated by monkey-patching
    the module-level view of `classes` for the duration of the call."""
    # base.write_datacard computes: backgrounds = [c for c in classes if c != signal]
    # We temporarily present our process list as `classes` so the derivation matches.
    saved = combine.get("classes")
    try:
        combine = dict(combine)
        combine["classes"] = processes
        return base.write_datacard(
            datacard_path, root_name, combine, proc_hists, None,
            obj_shift_systs=obj_shift_systs,
        )
    finally:
        if saved is not None:
            combine["classes"] = saved


# ----------------------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = base.parse_args()
    cfg = base.WorkflowConfigBuilder(workflow=args.workflow).build_workflow_config()
    combine = cfg.combine

    classes = combine["classes"]
    signal = combine["signal"]
    channels_by_class = {cls: ch for ch, cls in combine["channels"].items()}
    channels = list(combine["channels"].keys())
    process_map = combine["process_map"]

    processes = resolve_processes(combine)
    edges_by_ch = resolve_binning(combine, channels)
    backgrounds = [p for p in processes if p != signal]

    logging.info("v2 builder")
    logging.info("  processes (%d): %s", len(processes), processes)
    for ch in channels:
        nb = len(edges_by_ch[ch]) - 1
        logging.info("  %-14s %2d bin(s)%s", ch, nb, "  <- yield-only" if nb == 1 else "")

    score_cols = [f"mva_score_{c}" for c in classes]
    variations = base.build_variations(combine["shape_systematics"])
    NEGRW_SHAPE = combine.get("negrw_shape_name", "CMS_negrw_vjets")
    variations = variations + [(f"{NEGRW_SHAPE}Up", "__negrw__"),
                               (f"{NEGRW_SHAPE}Down", "__negrw__")]

    lumi = base.load_lumi(args.year)
    base_dir = base.OUTPUT_DIR / args.workflow / args.year
    var_dir = base_dir / args.variation if (base_dir / args.variation).exists() else base_dir
    mva_dir = var_dir / "mva"
    if not mva_dir.exists():
        sys.exit(f"scored parquets not found: {mva_dir}")

    combine_to_samples = base.gather_samples(args.year, process_map)

    # accumulator, per channel with that channel's own bin count
    proc_hists = {
        ch: {cp: {v: (np.zeros(len(edges_by_ch[ch]) - 1),
                      np.zeros(len(edges_by_ch[ch]) - 1)) for v, _ in variations}
             for cp in processes}
        for ch in channels
    }

    # process_sample fills ALL channels with one edges array, so call it once per
    # distinct binning and keep only the channels that use that binning.
    distinct = {}
    for ch in channels:
        distinct.setdefault(tuple(edges_by_ch[ch]), []).append(ch)
    logging.info("  %d distinct binning(s)", len(distinct))

    for cp in processes:
        for sample in combine_to_samples.get(cp, []):
            pq_path = mva_dir / f"{sample}.parquet"
            if not pq_path.exists():
                logging.info("  [skip] %s/%s: no %s", cp, sample, pq_path.name)
                continue
            for edges_t, chs in distinct.items():
                edges = np.array(edges_t, dtype=float)
                result = base.process_sample(
                    pq_path, sample, args.year, base_dir, classes, score_cols,
                    channels_by_class, variations, len(edges) - 1, edges, lumi,
                    is_vjets=(cp == "vjets"), negrw_shape_name=NEGRW_SHAPE,
                )
                if result is None:
                    continue
                for ch in chs:
                    for v, hh in result[ch].items():
                        acc_c, acc_s2 = proc_hists[ch][cp][v]
                        proc_hists[ch][cp][v] = (acc_c + hh[0], acc_s2 + hh[1])
            logging.info("  [ok]  %-10s %s", cp, sample)

    # --- Object-shift (kinematic) systematics -------------------------------------
    # Same logic as the original builder, but filled with each channel's own edges.
    # JES/JER/lepton-scale live in <syst>Up / <syst>Down parquet dirs with SHIFTED
    # kinematics -> shifted MVA scores -> events migrate argmax channels. MC-only,
    # weight_nominal only. A (channel, process) with no shift events falls back to its
    # nominal template so combine does not see a fake 100% shape.
    obj_shift_systs = []
    for d in sorted(glob.glob(str(base_dir / "*Up"))):
        nm = Path(d).name
        if nm.endswith("Up") and (Path(d) / "mva").is_dir() \
                and (base_dir / f"{nm[:-2]}Down" / "mva").is_dir():
            obj_shift_systs.append(nm[:-2])
    obj_shift_vars = [(f"{sname}{dn}", "weight_nominal")
                      for sname in obj_shift_systs for dn in ("Up", "Down")]

    for ch in channels:
        nb = len(edges_by_ch[ch]) - 1
        for cp in processes:
            for vn, _ in obj_shift_vars:
                proc_hists[ch][cp][vn] = (np.zeros(nb), np.zeros(nb))

    for vn, _ in obj_shift_vars:
        sdir = base_dir / vn / "mva"
        for cp in processes:
            for sample in combine_to_samples.get(cp, []):
                pq_path = sdir / f"{sample}.parquet"
                if not pq_path.exists():
                    continue
                for edges_t, chs in distinct.items():
                    edges = np.array(edges_t, dtype=float)
                    res = base.process_sample(
                        pq_path, sample, args.year, base_dir, classes, score_cols,
                        channels_by_class, [(vn, "weight_nominal")],
                        len(edges) - 1, edges, lumi, is_vjets=(cp == "vjets"),
                    )
                    if res is None:
                        continue
                    for ch in chs:
                        ac, as2 = proc_hists[ch][cp][vn]
                        hh = res[ch][vn]
                        proc_hists[ch][cp][vn] = (ac + hh[0], as2 + hh[1])

    for ch in channels:
        for cp in processes:
            nom_c, nom_s2 = proc_hists[ch][cp]["nominal"]
            for vn, _ in obj_shift_vars:
                c, _s2 = proc_hists[ch][cp][vn]
                if c.sum() <= 0 < nom_c.sum():
                    proc_hists[ch][cp][vn] = (nom_c.copy(), nom_s2.copy())

    variations = variations + obj_shift_vars
    logging.info("Folded %d object-shift systematics: %s",
                 len(obj_shift_systs), obj_shift_systs)

    n_clip = base.clip_negative_bins(proc_hists, channels, processes, variations)
    logging.info("Clipped %d non-positive nominal bins to floor", n_clip)

    if combine.get("smooth_shapes"):
        n_sm = base.smooth_shape_variations(proc_hists, channels, processes, variations)
        logging.info("Smoothed %d shape variations (LOWESS)", n_sm)
        base.clip_negative_bins(proc_hists, channels, processes, variations)

    out_root = Path(combine["output"]["root"])
    out_card = Path(combine["output"]["datacard"])
    n = write_root_v2(out_root, proc_hists, channels, processes, variations, edges_by_ch)
    write_data_obs_v2(out_root, proc_hists, channels, backgrounds, edges_by_ch)
    yields = write_datacard_v2(out_card, out_root.name, combine, proc_hists,
                               processes, obj_shift_systs=obj_shift_systs)

    logging.info("wrote %d histograms -> %s", n, out_root)
    logging.info("wrote datacard      -> %s", out_card)
    logging.info("")
    logging.info("Per-channel x per-process nominal yields:")
    hdr = "  %-14s" % "channel" + "".join("%12s" % p for p in processes) + "%12s" % "total"
    logging.info(hdr)
    for ch in channels:
        row = "  %-14s" % ch
        tot = 0.0
        for p in processes:
            y = yields[(ch, p)]
            tot += y
            row += "%12.3f" % y
        logging.info(row + "%12.3f" % tot)


if __name__ == "__main__":
    main()
