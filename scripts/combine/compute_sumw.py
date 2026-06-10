"""Compute the true generator sumw per sample from the NanoAOD Runs tree.

The self-normalising parquet sumw (parquet_writer.dump_parquet) only sums
genWeight over read-chunks that produced >=1 selected event; for low-efficiency
samples (vjets, hadronic) most chunks select nothing, write no shard, and their
sumw is lost -- so the parquet sumw is far too small and the lumi*xsec/sumw
normalisation comes out too large (e.g. vjets floods the signal region).

The authoritative number is the sum of `genEventSumw` over the Runs tree of every
file in the dataset -- a single pre-computed value per file, no event loop. This
script reads it (parallel) and writes a sidecar JSON the combine/postprocess
normalisation reads instead of the parquet metadata.

Run inside the coffea container (needs xrootd), with a valid proxy:

  python3 scripts/combine/compute_sumw.py -y 2022postEE \
      --samples WtoLNu_2Jets DYto2L_2Jets_50 DYto2L_2Jets_10to50
  # or omit --samples to do every MC sample in the fileset

Writes: analysis/filesets/sumw_<year>.json  ({sample: sumw})
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import uproot

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.filesets.utils import get_dataset_config, get_nano_version


def file_sumw(path):
    """Sum genEventSumw over the Runs tree of one NanoAOD file."""
    with uproot.open(path) as f:
        return float(f["Runs"]["genEventSumw"].array(library="np").sum())


def sample_sumw(files, workers=16):
    total = 0.0
    bad = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(file_sumw, p): p for p in files}
        for fut in as_completed(futs):
            try:
                total += fut.result()
            except Exception as e:  # noqa: BLE001
                bad += 1
                print(f"    [warn] {futs[fut].split('/')[-1]}: {type(e).__name__}")
    return total, bad


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-y", "--year", required=True)
    ap.add_argument("--samples", nargs="*", default=None,
                    help="samples to compute (default: every MC sample in the fileset)")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    filesets_dir = Path.cwd() / "analysis" / "filesets"
    nano = get_nano_version(args.year)
    fileset = json.load(open(filesets_dir / f"fileset_{args.year}_nanov{nano}_lxplus.json"))
    cfg = get_dataset_config(args.year)

    out_path = filesets_dir / f"sumw_{args.year}.json"
    result = json.load(open(out_path)) if out_path.exists() else {}

    samples = args.samples or [
        s for s in fileset if cfg.get(s, {}).get("era") in ("mc", "signal")
    ]
    for s in samples:
        files = fileset.get(s)
        if not files:
            print(f"{s}: no files in fileset, skip")
            continue
        sw, bad = sample_sumw(files, args.workers)
        result[s] = sw
        print(f"{s:30s} sumw={sw:.6e}  ({len(files)} files, {bad} failed)")
        json.dump(result, open(out_path, "w"), indent=2)  # checkpoint each sample

    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
