"""Drive b-hive MVA inference for the higgscharm fit flow.

Reads the `inference:` block from the workflow yaml and calls
`analysis.postprocess.inference.run_inference` for the requested variation(s).

    inference:
      framework: b-hive
      model_version: hww_multiclass_v32
      model_path: /eos/.../best_model.pt
      bhive_path:  /eos/.../b-hive
      bhive_config: HPlusCHToWW_kappa_hce
      bhive_model_name: SimpleMLP_MultiClass
      batch_size: 4096

For each variation, it expects per-process parquets at:
    outputs/<workflow>/<year>/<variation>/<process>.parquet
and writes score-augmented copies to:
    outputs/<workflow>/<year>/<variation>/mva/<process>.parquet

Falls back to outputs/<workflow>/<year>/ if no per-variation subdir exists.

Usage:
  python scripts/mva/run_inference.py -w hww_MVA -y 2022postEE
  python scripts/mva/run_inference.py -w hww_MVA -y 2022postEE --variation nominal
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make `analysis.*` importable when invoked from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.postprocess.inference import run_inference
from analysis.workflows.config import WorkflowConfigBuilder

OUTPUT_DIR = Path.cwd() / "outputs"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-w", "--workflow", required=True,
        choices=[f.stem for f in (Path.cwd() / "analysis" / "workflows").glob("*.yaml")],
        help="workflow yaml name (must contain an 'inference:' block)",
    )
    parser.add_argument(
        "-y", "--year", required=True,
        choices=["2022preEE", "2022postEE", "2023preBPix", "2023postBPix"],
        help="data-taking year",
    )
    parser.add_argument(
        "--variation", default=None,
        help="single variation to run (default: nominal + discovered shift subdirs, "
             "or just 'nominal' if no variations: block exists)",
    )
    # All of these override the yaml's `inference:` block if set.
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--bhive-path", default=None)
    parser.add_argument("--bhive-config", default=None)
    parser.add_argument("--bhive-model-name", default=None)
    return parser.parse_args()


def discover_variations(base_dir, requested: str | None) -> list[str]:
    """Variations to score. Explicit --variation wins; otherwise 'nominal'
    plus any object-shift subdirs that were produced (object_shifts: true)."""
    if requested is not None:
        return [requested]
    variations = ["nominal"]
    for sub in sorted(p.name for p in base_dir.iterdir() if p.is_dir()):
        # shift subdirs sit alongside the merged nominal parquets; skip the
        # bookkeeping dirs (training/, mva/, filelists/, per-sample parquet dirs)
        if sub in ("mva", "training", "filelists"):
            continue
        if (base_dir / sub / "mva").exists() or any((base_dir / sub).glob("*.parquet")):
            variations.append(sub)
    return variations


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    base_dir = OUTPUT_DIR / args.workflow / args.year
    if not base_dir.exists():
        sys.exit(f"output dir not found: {base_dir}")

    cfg = WorkflowConfigBuilder(workflow=args.workflow).build_workflow_config()
    if cfg.inference is None:
        sys.exit(f"workflow {args.workflow} has no 'inference:' block in its yaml")

    inf = cfg.inference
    model_path = args.model_path or inf["model_path"]
    bhive_path = args.bhive_path or inf["bhive_path"]
    bhive_config = args.bhive_config or inf["bhive_config"]
    bhive_model_name = args.bhive_model_name or inf["bhive_model_name"]

    if not Path(model_path).exists():
        sys.exit(f"model checkpoint not found: {model_path}")

    for variation in discover_variations(base_dir, args.variation):
        # nominal merged parquets live at <year>/; shifts at <year>/<shift>/.
        var_dir = base_dir / variation if variation != "nominal" else base_dir
        logging.info(f"=== inference: variation={variation}  dir={var_dir} ===")
        run_inference(
            output_dir=var_dir,
            model_path=model_path,
            bhive_path=bhive_path,
            config_name=bhive_config,
            model_name=bhive_model_name,
        )


if __name__ == "__main__":
    main()
