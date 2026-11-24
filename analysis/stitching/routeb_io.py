"""Route B stitch-record I/O and card-builder scale factors (PROPOSAL HTOWW-ROUTEB-SOL-20261105, agent-written).

One record per processed chunk, written next to the existing sumw_records:
    <output>/<workflow>/<year>/<dataset>/stitch_records/<partition_key>.json
The filename is the coffea partition key, so a retried chunk OVERWRITES its record instead of adding a second
one (no duplicate sumw). Readers additionally de-duplicate by partition key across shard dirs (<dataset>_<n>).

The record's `full` block is a cross-check of sumw_records, not a second normaliser: scale_factors() requires
|full.sumw - sum(sumw_records)| / |full.sumw| < 1e-6 and then uses ONE source (sumw_records) for F_s.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from analysis.stitching.hplusc_routeb import RouteBConfig, StitchSums, normalisation, SCHEMES

RECORD_DIR = "stitch_records"
FORMAT_VERSION = 1


def record_payload(sums: StitchSums, dataset: str, year: str, partition_key: str) -> dict:
    return {"format": FORMAT_VERSION, "dataset": dataset, "year": year, "partition_key": partition_key,
            "sums": sums.to_dict()}


def safe_key(partition_key: str) -> str:
    return partition_key.replace("/", "_")


def write_record_local(payload: dict, out_dir: str) -> str:
    """Write a record to a local/FUSE directory (tests, local smoke). Production should use write_record()."""
    p = Path(out_dir) / RECORD_DIR
    p.mkdir(parents=True, exist_ok=True)
    f = p / (safe_key(payload["partition_key"]) + ".json")
    f.write_text(json.dumps(payload, sort_keys=True))
    return str(f)


def write_record(payload: dict, workflow: str, year: str, output_location: str) -> None:
    """Production writer: reuse the framework's EOS-aware copy (parquet_writer._xrd_copy has the [3018] retry).
    Writes a one-row parquet so the same eos_io / pyarrow tooling as sumw_records can read it."""
    from analysis.utils.parquet_writer import dump_pa_table

    s = payload["sums"]
    row = {"payload": [json.dumps(payload, sort_keys=True)],
           "full_sumw": [s["full"]["sumw"]], "kept_sumw": [s["kept"]["sumw"]]}
    dump_pa_table(row, safe_key(payload["partition_key"]) + ".parquet", output_location,
                  [workflow, year, payload["dataset"], RECORD_DIR], extra_metadata={"routeb_format": str(FORMAT_VERSION)})


def _shard_dirs(base_dir: str, dataset: str, sub: str, glob) -> List[str]:
    dirs = glob(f"{base_dir}/{dataset}_*/{sub}") + glob(f"{base_dir}/{dataset}/{sub}")
    # same guard as make_combine_inputs.read_scale: HplusCharm_3FS_HToWW must not catch ..._ext1
    return [d for d in dirs if re.fullmatch(rf"{re.escape(dataset)}(_\d+)?", Path(d).parent.name)]


def _load_payload(path: str) -> dict:
    if path.endswith(".json"):
        return json.loads(Path(path).read_text())
    import pyarrow.parquet as pq
    return json.loads(pq.read_table(path, columns=["payload"])["payload"].to_pylist()[0])


def read_records(base_dir: str, dataset: str, glob=None) -> Tuple[StitchSums, int]:
    """Sum all stitch records of one dataset in one era dir. Returns (sums, n_unique_chunks).
    Raises on duplicate partition keys with DIFFERENT content (two incompatible processings)."""
    if glob is None:
        from analysis.utils.eos_io import glob  # server-side EOS listing (stale FUSE listings undercount)
    seen: Dict[str, dict] = {}
    for d in _shard_dirs(base_dir, dataset, RECORD_DIR, glob):
        for f in glob(f"{d}/*.json") + glob(f"{d}/*.parquet"):
            p = _load_payload(f)
            k = p["partition_key"]
            if k in seen:
                if seen[k]["sums"] != p["sums"]:
                    raise RuntimeError(f"{dataset}: conflicting stitch records for chunk {k}")
                continue
            seen[k] = p
    if not seen:
        raise FileNotFoundError(f"no {RECORD_DIR} for {dataset} under {base_dir}")
    total = sum((StitchSums.from_dict(p["sums"]) for p in seen.values()), 0)
    return total, len(seen)


def read_sumw_records(base_dir: str, dataset: str, glob=None) -> float:
    """Same algorithm as scripts/combine/make_combine_inputs.read_scale (sumw_records parquet, signed)."""
    import pyarrow.parquet as pq
    if glob is None:
        from analysis.utils.eos_io import glob
    tot = 0.0
    for d in _shard_dirs(base_dir, dataset, "sumw_records", glob):
        for f in glob(f"{d}/*.parquet"):
            tot += float(sum(pq.read_table(f, columns=["sumw"])["sumw"].to_pylist()))
    return tot


def sums_from_sink_entry(entry: Mapping) -> Dict[str, StitchSums]:
    """TRAIN/RUN path: parse output["samples"][sample]["routeb_stitch"] = {partition_key: json string} (one sample)
    into {dataset: StitchSums}. Raises if any value is not exactly one JSON object (a chunk accumulated twice)."""
    out: Dict[str, StitchSums] = {}
    for key, raw in (entry.get("routeb_stitch") or {}).items():
        try:
            p = json.loads(raw)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"routeb_stitch[{key!r}] is not a single record (chunk accumulated twice?)") from e
        if p["partition_key"] != key:
            raise RuntimeError(f"routeb_stitch key mismatch {key!r} vs {p['partition_key']!r}")
        s = StitchSums.from_dict(p["sums"])
        d = p["dataset"]
        out[d] = out[d] + s if d in out else s
    return out


def scale_factors(cfg: RouteBConfig, era_dirs: Mapping[str, str], lumi_pb: Mapping[str, float],
                  xsec_pb: Mapping[Tuple[str, str], float], glob=None, check_sumw_records: bool = True):
    """Per-(era, dataset) event normalisation for the Route B samples.

    era_dirs: era -> <output>/<workflow>/<era> ; xsec_pb[(era, dataset)] = fileset `xsec` (must be the SAME
    sigma_NNLO x BR for every Route B dataset of an era; enforced). Returns
      {(era, dataset): {"scale": L*X*c_s/F_s, "scheme", "c", "F_s"}}, {era or "pooled": StitchNorm}
    F_s is the scheme total over ALL datasets of that scheme in the era (main + ext1 share one F_s).
    """
    per_era_sums: Dict[str, Dict[str, StitchSums]] = {}
    per_ds_full: Dict[Tuple[str, str], float] = {}
    for era, base in era_dirs.items():
        xs = {xsec_pb[(era, d)] for d in cfg.datasets if (era, d) in xsec_pb}
        if len(xs) != 1:
            raise ValueError(f"{era}: Route B datasets must share one xsec (sigma_NNLO x BR); got {sorted(xs)}")
        by_scheme: Dict[str, StitchSums] = {}
        for d, scheme in cfg.datasets.items():
            try:
                s, _ = read_records(base, d, glob)
            except FileNotFoundError:
                continue  # dataset absent in this era (e.g. ext1 exists only for 2022postEE)
            if check_sumw_records:
                sw = read_sumw_records(base, d, glob)
                if abs(s.full.sumw - sw) > 1e-6 * abs(s.full.sumw):
                    raise RuntimeError(f"{era}/{d}: stitch full sumw {s.full.sumw} != sumw_records {sw} "
                                       "(missing/duplicate chunks?)")
            per_ds_full[(era, d)] = s.full.sumw
            by_scheme[scheme] = by_scheme[scheme] + s if scheme in by_scheme else s
        if set(by_scheme) != set(SCHEMES):
            raise RuntimeError(f"{era}: need both schemes, have {sorted(by_scheme)}")
        per_era_sums[era] = by_scheme

    norms = {}
    if cfg.domain == "per_era":
        for era, sums in per_era_sums.items():
            norms[era] = normalisation(sums)
    else:  # pooled_fractions: fractions from all eras, F_s still per era
        pooled = {s: sum((per_era_sums[e][s] for e in per_era_sums), 0) for s in SCHEMES}
        n = normalisation(pooled)
        for era in per_era_sums:
            norms[era] = n
        norms["pooled"] = n

    out = {}
    for (era, d), _ in per_ds_full.items():
        scheme = cfg.datasets[d]
        F = per_era_sums[era][scheme].full.sumw
        c = norms[era].c[scheme]
        out[(era, d)] = {"scheme": scheme, "c": c, "F_s": F,
                         "scale": lumi_pb[era] * xsec_pb[(era, d)] * c / F}
    return out, norms


def sink_scale_factors(cfg: RouteBConfig, samples: Mapping[str, Mapping], lumi_pb: float,
                       xsec_pb: Mapping[str, float], era_samples=None):
    """TRAIN/RUN accumulator path, ONE era (per_era domain), used by scripts/combine/build_cards_from_templates.py.

    samples: merged output["samples"] of one era; xsec_pb[sample] = fileset `xsec`. For every Route B sample
    present, checks that each processed chunk carries exactly one stitch record and that the records' full sumw
    equals the sample's (pre-mask) sumw, then returns
      {sample: {"scheme", "c", "F_s", "scale": L*X*c_s/F_s}}, StitchNorm   ({}, None if no Route B sample)
    F_s sums every dataset of the scheme in the era (main + ext1 share one F_s).
    era_samples (optional): the era's fileset sample names. If given and any Route B sample is used, every
    configured Route B sample that exists in the era must be present in `samples` (a missing ext1 would otherwise
    silently shrink F_s).
    """
    if cfg.domain != "per_era":
        raise NotImplementedError("sink_scale_factors implements the per_era domain only")
    if era_samples is not None and any(cfg.scheme_of(s) for s in samples):
        missing = sorted(s for s in cfg.datasets if s in set(era_samples) and s not in samples)
        if missing:
            raise RuntimeError(f"Route B samples of this era missing from the output: {missing}")
    per_sample, by_scheme = {}, {}
    for sample, entry in samples.items():
        scheme = cfg.scheme_of(sample)
        if scheme is None:
            continue
        keys = set(entry.get("routeb_stitch") or {})
        chunks = set(entry.get("chunks") or ())
        if keys != chunks:
            raise RuntimeError(f"{sample}: stitch records for {len(keys)} chunks, sample has {len(chunks)} "
                               f"({len(chunks - keys)} without record, {len(keys - chunks)} unknown)")
        recs = sums_from_sink_entry(entry)
        if set(recs) != {sample}:
            raise RuntimeError(f"{sample}: stitch records name datasets {sorted(recs)}")
        s = recs[sample]
        if s.scheme != scheme:
            raise RuntimeError(f"{sample}: records labelled {s.scheme}, workflow says {scheme}")
        sumw = float(entry["sumw"])
        if abs(s.full.sumw - sumw) > 1e-6 * abs(sumw):
            raise RuntimeError(f"{sample}: stitch full sumw {s.full.sumw} != sample sumw {sumw}")
        per_sample[sample] = (scheme, s)
        by_scheme[scheme] = by_scheme[scheme] + s if scheme in by_scheme else s
    if not per_sample:
        return {}, None
    if set(by_scheme) != set(SCHEMES):
        raise RuntimeError(f"Route B needs both schemes, have {sorted(by_scheme)} ({sorted(per_sample)})")
    xs = {float(xsec_pb[s]) for s in per_sample}
    if len(xs) != 1:
        raise ValueError(f"Route B samples must share one xsec (sigma_NNLO x BR); got {sorted(xs)}")
    X = xs.pop()
    norm = normalisation(by_scheme)
    out = {}
    for sample, (scheme, _) in per_sample.items():
        F = by_scheme[scheme].full.sumw
        out[sample] = {"scheme": scheme, "c": norm.c[scheme], "F_s": F, "scale": lumi_pb * X * norm.c[scheme] / F}
    return out, norm
