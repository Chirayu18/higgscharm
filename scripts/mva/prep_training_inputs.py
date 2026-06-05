"""Split merged per-process parquets into train/test sets for b-hive.

Reads the `mva:` block from the workflow yaml:

    mva:
      train_test_split:
        field: event           # column to mod against (NanoAOD event id)
        test_modulo: 10        # ~10% test, ~90% train
        test_remainder: 9
      labels:
        process_groups:        # ordered: defines the one-hot class order
          higgs:   [H+c, H+b, ggH, ...]
          tt:      [tt]
          ...

For each <output_dir>/<process>.parquet (one per physics process, written
by run_postprocess.py):

  1. add one-hot `is_<group>` columns derived from process_groups
  2. add a `weight` alias (b-hive expects this name)
  3. split events into train/test using `df[field] % test_modulo`
  4. write <output_dir>/training/<process>_train.parquet
            <output_dir>/training/<process>_test.parquet

Then write filelists for b-hive:
  <output_dir>/training/filelists/train.txt
  <output_dir>/training/filelists/test.txt

Usage:
  python scripts/mva/prep_training_inputs.py -w hww_MVA -y 2022postEE
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Make `analysis.*` importable when invoked from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.workflows.config import WorkflowConfigBuilder

OUTPUT_DIR = Path.cwd() / "outputs"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-w", "--workflow", required=True,
        choices=[f.stem for f in (Path.cwd() / "analysis" / "workflows").glob("*.yaml")],
        help="workflow yaml name (must contain an 'mva:' block)",
    )
    parser.add_argument(
        "-y", "--year", required=True,
        choices=["2022preEE", "2022postEE", "2023preBPix", "2023postBPix"],
        help="data-taking year",
    )
    parser.add_argument(
        "--processes", nargs="*", default=None,
        help="restrict to these process names (default: every {process}.parquet found)",
    )
    return parser.parse_args()


def build_process_to_group(process_groups: dict) -> dict:
    """Invert {group: [process,...]} into {process: group}."""
    mapping = {}
    for group, processes in process_groups.items():
        for p in processes:
            if p in mapping:
                logging.warning(
                    f"Process {p!r} mapped to both {mapping[p]!r} and {group!r}; "
                    f"keeping {group!r}"
                )
            mapping[p] = group
    return mapping


def add_labels(df: pd.DataFrame, process: str, group_names: list, p2g: dict) -> pd.DataFrame:
    """Add is_<group> one-hot columns + a 'weight' alias."""
    group = p2g.get(process.strip())
    if group is None:
        raise KeyError(
            f"Process {process!r} not present in mva.labels.process_groups; "
            f"known: {sorted(p2g)}"
        )
    for g in group_names:
        df[f"is_{g}"] = int(g == group)
    if "weight" not in df.columns:
        if "weight_nominal" in df.columns:
            df["weight"] = df["weight_nominal"]
        else:
            df["weight"] = 1.0
    return df


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    base_dir = OUTPUT_DIR / args.workflow / args.year
    # New layout: <year>/nominal/<process>.parquet. Fall back to <year>/<process>.parquet
    # for pre-variation outputs.
    nominal_dir = base_dir / "nominal"
    output_dir = nominal_dir if nominal_dir.exists() else base_dir
    if not output_dir.exists():
        sys.exit(f"output dir not found: {output_dir}")

    cfg = WorkflowConfigBuilder(workflow=args.workflow).build_workflow_config()
    if cfg.mva is None:
        sys.exit(f"workflow {args.workflow} has no 'mva:' block in its yaml")

    split = cfg.mva["train_test_split"]
    field = split["field"]
    modulo = int(split["test_modulo"])
    remainder = int(split["test_remainder"])

    process_groups = cfg.mva["labels"]["process_groups"]
    group_names = list(process_groups.keys())  # yaml order = class order
    p2g = build_process_to_group(process_groups)

    if args.processes:
        process_names = args.processes
    else:
        process_names = sorted(
            p.stem for p in output_dir.glob("*.parquet")
            if p.stem in p2g  # skip Data.parquet and anything else not labelled
        )
    if not process_names:
        sys.exit(f"no labelable {{process}}.parquet found in {output_dir}")

    out_dir = output_dir / "training"
    filelist_dir = out_dir / "filelists"
    filelist_dir.mkdir(parents=True, exist_ok=True)

    train_files, test_files = [], []
    for process in process_names:
        src = output_dir / f"{process}.parquet"
        if not src.exists():
            logging.warning(f"missing {src}, skipping")
            continue

        df = pd.read_parquet(src)
        if field not in df.columns:
            sys.exit(
                f"{src} has no {field!r} column. Re-run runner.py after the "
                f"base.py update that writes the NanoAOD event id."
            )

        df = add_labels(df, process, group_names, p2g)
        is_test = (df[field].astype("int64") % modulo) == remainder
        train_path = out_dir / f"{process}_train.parquet"
        test_path = out_dir / f"{process}_test.parquet"
        df[~is_test].to_parquet(train_path, engine="pyarrow", index=False)
        df[is_test].to_parquet(test_path, engine="pyarrow", index=False)

        logging.info(
            f"{process:>20s}  total={len(df):>8d}  "
            f"train={(~is_test).sum():>8d}  test={is_test.sum():>8d}  "
            f"group={p2g[process.strip()]}"
        )
        train_files.append(train_path)
        test_files.append(test_path)

    with open(filelist_dir / "train.txt", "w") as f:
        f.write("\n".join(str(p) for p in train_files) + "\n")
    with open(filelist_dir / "test.txt", "w") as f:
        f.write("\n".join(str(p) for p in test_files) + "\n")

    logging.info(f"\nwrote {len(train_files)} train + {len(test_files)} test parquets")
    logging.info(f"filelists at {filelist_dir}/{{train,test}}.txt")


if __name__ == "__main__":
    main()
