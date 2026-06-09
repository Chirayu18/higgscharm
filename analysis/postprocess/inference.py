"""
Run b-hive model inference on postprocessed parquet files.
Adds MVA score columns and saves copies to output_dir/mva/.
"""

import sys
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


def run_inference(
    output_dir,
    model_path,
    bhive_path,
    config_name,
    model_name,
    split="full",
    split_field="event",
    split_modulo=None,
    split_remainder=None,
):
    """
    Load a trained b-hive model and add MVA score columns to parquet files.

    Reads each {process}.parquet in output_dir, runs inference, and writes
    augmented copies to output_dir/mva/{process}.parquet.

    Parameters
    ----------
    output_dir : Path
        Directory containing {process}.parquet files.
    model_path : str
        Path to the trained model checkpoint (.pt file).
    bhive_path : str
        Path to the b-hive repository root.
    config_name : str
        Name of the b-hive config (e.g. 'HPlusCHToWW_multiclass').
    model_name : str
        Name of the model class (e.g. 'SimpleMLP_MultiClass').
    split : {'full', 'test', 'train'}
        Which event subset to score. 'full' (default) scores every event and
        writes to <output_dir>/mva/. 'test'/'train' score only the matching
        half of the train/test split (using the same field % modulo == remainder
        rule as prep_training_inputs.py) and write to <output_dir>/mva_<split>/,
        so the full and split outputs never clobber each other.
    split_field : str
        Event-id column the split is computed on (default 'event').
    split_modulo, split_remainder : int
        Split parameters; test = (field % modulo == remainder), train = the
        complement. Required when split != 'full'.
    """
    if split not in ("full", "test", "train"):
        raise ValueError(f"split must be full|test|train, got {split!r}")
    if split != "full" and (split_modulo is None or split_remainder is None):
        raise ValueError(
            f"split={split!r} requires split_modulo and split_remainder"
        )
    import torch

    # Add b-hive to sys.path so we can import its modules
    bhive_path = str(bhive_path)
    if bhive_path not in sys.path:
        sys.path.insert(0, bhive_path)

    from utils.config.config_loader import ConfigLoader
    from utils.models.models import BTaggingModels

    # Set B_HIVE_DIR env var if not set (needed by ConfigLoader)
    import os
    if not os.getenv("B_HIVE_DIR"):
        os.environ["B_HIVE_DIR"] = bhive_path

    # Load config and build model
    logging.info(f"Loading b-hive config: {config_name}")
    config = ConfigLoader.load_config(config_name)

    logging.info(f"Building model: {model_name}")
    model = BTaggingModels(model_name, config)
    model.create_integers_defaults()
    model.create_feature_shapes()

    # Load checkpoint
    logging.info(f"Loading checkpoint: {model_path}")
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.for_inference = True

    # Get feature list and class names from config/model
    features = config["global_features"]
    class_names = list(model.classes.keys())
    logging.info(f"Features ({len(features)}): {features}")
    logging.info(f"Classes: {class_names}")

    # Create output directory for MVA-augmented parquets. Split outputs go to a
    # sibling dir so the default 'full' mva/ (read by combine) is never clobbered.
    mva_dir = Path(output_dir) / ("mva" if split == "full" else f"mva_{split}")
    mva_dir.mkdir(parents=True, exist_ok=True)
    if split != "full":
        logging.info(
            f"split={split}: keeping events where {split_field} % {split_modulo} "
            f"{'==' if split == 'test' else '!='} {split_remainder}"
        )

    # Find process-level parquet files in output_dir (not in subdirectories)
    parquet_files = sorted(Path(output_dir).glob("*.parquet"))
    if not parquet_files:
        logging.warning(f"No parquet files found in {output_dir}")
        return

    logging.info(f"Found {len(parquet_files)} parquet file(s) to process")

    for pq_file in parquet_files:
        logging.info(f"Processing: {pq_file.name}")
        df = pd.read_parquet(pq_file)

        # restrict to the requested train/test subset (full = no filter)
        if split != "full" and len(df) > 0:
            if split_field not in df.columns:
                raise KeyError(
                    f"{pq_file.name} has no {split_field!r} column needed for "
                    f"split={split!r}"
                )
            is_test = (df[split_field].astype("int64") % split_modulo) == split_remainder
            df = df[is_test] if split == "test" else df[~is_test]
            df = df.reset_index(drop=True)

        if len(df) == 0:
            logging.warning(f"  Skipping empty file: {pq_file.name}")
            df.to_parquet(mva_dir / pq_file.name)
            continue

        # Extract features in config order, fill missing with 0
        feature_arrays = []
        for feat in features:
            if feat in df.columns:
                feature_arrays.append(df[feat].values)
            else:
                warnings.warn(f"  Feature '{feat}' missing in {pq_file.name}, filling with 0.0")
                feature_arrays.append(np.zeros(len(df), dtype=np.float32))

        X = np.column_stack(feature_arrays).astype(np.float32)

        # Replace NaN/inf with 0
        mask = ~np.isfinite(X)
        if mask.any():
            n_bad = mask.sum()
            logging.warning(f"  Replacing {n_bad} NaN/inf values with 0")
            X[mask] = 0.0

        # Run inference in batches
        batch_size = 4096
        all_scores = []

        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                batch = torch.tensor(X[start : start + batch_size])
                # Model expects (global_feat, cpf, npf) tuple
                dummy_cpf = torch.zeros(batch.shape[0], 0, 0)
                dummy_npf = torch.zeros(batch.shape[0], 0, 0)
                scores = model((batch, dummy_cpf, dummy_npf))
                all_scores.append(scores.cpu().numpy())

        all_scores = np.concatenate(all_scores, axis=0)

        # Add score columns
        for i, cls_name in enumerate(class_names):
            df[f"mva_score_{cls_name}"] = all_scores[:, i]

        # Save to mva subdirectory
        out_path = mva_dir / pq_file.name
        df.to_parquet(out_path)
        logging.info(f"  Saved {out_path} with {len(class_names)} score columns")

        # Quick sanity check
        score_cols = [c for c in df.columns if c.startswith("mva_score_")]
        score_sums = df[score_cols].sum(axis=1)
        logging.info(
            f"  Score sum: mean={score_sums.mean():.4f}, "
            f"std={score_sums.std():.4f} (should be ~1.0)"
        )

    logging.info(f"Inference complete. MVA parquets saved to: {mva_dir}")
