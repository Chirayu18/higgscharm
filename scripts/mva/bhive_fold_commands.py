#!/usr/bin/env python3
"""Write (never run) the per-fold b-hive commands for a split_folds.py output.

For model i: b-hive constructs a dataset from model_i/train.txt (four folds for
k=5) and its TrainingTask splits those constructed files into training and
validation itself (train_val_split). model_i/test.txt (fold i only) is
constructed as a separate test dataset for InferenceTask/ROC; it never enters
training or validation.

b-hive stores a constructed dataset under
$DATA_PATH/DatasetConstructorTask/<config>/<dataset_version>; the filelist is
not part of that path (luigi significant=False). Reusing a dataset version
across folds would silently reuse another fold's data, so every version here is
unique per split hash and fold, and the script refuses to start if a target
already exists. The stock train_v11_2dcats.sh is not used: it cannot set the
test dataset version, which would reuse one test dataset for every fold.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex


def build(split_dir, config, model, epochs, batch_size, lr, workers, chunk_size,
          train_val_split, prefix, split_seed=None, training_seed=None,
          data_path=None, bhive_dir=None):
    """split_seed (HToWW b-hive copy, 2026-11-12): seed of TrainingTask's shuffle
    before its internal train/validation split. training_seed: seed of every other
    RNG of a fresh TrainingTask (init, epoch file order, DataLoader workers). None
    keeps the old command line (no seed flag) for b-hive checkouts without them.
    data_path / bhive_dir: if given, the script refuses to start unless $DATA_PATH /
    $B_HIVE_DIR equal them (b-hive setup.sh sources local_setup.sh, which points
    DATA_PATH at another output area; export DATA_PATH after sourcing setup.sh)."""
    split_dir = Path(split_dir).resolve()
    manifest = json.loads((split_dir / 'split_manifest.json').read_text())
    if manifest.get('schema_version') != 2:
        raise ValueError('split manifest predates the single-validation-owner scheme')
    k = manifest['folds']
    tag = manifest['filelist_sha256'][:8]
    runs, lines = {}, []
    for i in range(k):
        m = manifest['models'][str(i)]
        train, test = split_dir / m['train_list'], split_dir / m['test_list']
        for path, key in ((train, 'train_list_sha256'), (test, 'test_list_sha256')):
            if hashlib.sha256(path.read_bytes()).hexdigest() != m[key]:
                raise ValueError(f'{path} changed after the split manifest was written')
        train_set = set(train.read_text().split())
        test_set = set(test.read_text().split())
        if train_set & test_set or not test_set or not train_set:
            raise ValueError(f'model {i}: train/test lists overlap or are empty')
        if any(f'_fold_{i}.parquet' not in p for p in test_set) or any(f'_fold_{i}.parquet' in p for p in train_set):
            raise ValueError(f'model {i}: test list must hold exactly fold {i}')
        ds, ts, tv = f'{prefix}_{tag}_f{i}_train', f'{prefix}_{tag}_f{i}_test', f'{prefix}_{tag}_f{i}'
        runs[i] = dict(dataset_version=ds, test_dataset_version=ts, training_version=tv,
                       config=config, model_name=model, epochs=epochs,
                       train_val_split=train_val_split, train_list=str(train), test_list=str(test))
        if split_seed is not None:
            runs[i]['split_seed'] = int(split_seed)
        if training_seed is not None:
            runs[i]['training_seed'] = int(training_seed)
        if data_path is not None:
            runs[i]['data_path'] = str(data_path)
        if bhive_dir is not None:
            runs[i]['bhive_dir'] = str(bhive_dir)
        q = shlex.quote
        common = f'--config {q(config)}'
        train_args = (f'{common} --filelist {q(str(train))} --dataset-version {ds} '
                      f'--training-version {tv} --model-name {q(model)} --epochs {epochs} '
                      f'--batch-size {batch_size} --learning-rate {lr}')
        if split_seed is not None:
            # same value on TrainingTask and InferenceTask (InferenceTask requires TrainingTask)
            train_args += f' --split-seed {int(split_seed)}'
        if training_seed is not None:
            train_args += f' --training-seed {int(training_seed)}'
        lines += [
            f'# ---- model {i}: train/validate on folds {m["train_folds"]} (b-hive internal '
            f'{train_val_split:g}/{1-train_val_split:g}), test fold {i} ----',
            f'guard {q(config)} {ds}; guard {q(config)} {ts}',
            f'guard_run {q(config)} {ds}',
            f'law run DatasetConstructorTask {common} --filelist {q(str(train))} '
            f'--dataset-version {ds} --coffea-worker {workers} --chunk-size {chunk_size}',
            f'law run DatasetConstructorTask {common} --filelist {q(str(test))} '
            f'--dataset-version {ts} --coffea-worker {workers} --chunk-size {chunk_size}',
            f'law run TrainingTask {train_args} --train-val-split {train_val_split} --loss-weighting',
            f'law run InferenceTask {train_args} --test-dataset-version {ts} --test-filelist {q(str(test))}',
            '',
        ]
    header = ['#!/bin/bash', '# Generated by scripts/mva/bhive_fold_commands.py. Run from the b-hive',
              '# checkout after `source setup.sh`. Needs explicit approval: this is model training.',
              'set -euo pipefail', ': "${DATA_PATH:?source b-hive setup.sh first}"',
              'guard() { if [ -e "$DATA_PATH/DatasetConstructorTask/$1/$2" ]; then',
              '  echo "refusing: $DATA_PATH/DatasetConstructorTask/$1/$2 exists (stale fold data)" >&2; exit 1; fi; }',
              '']
    # Training/inference outputs live under the train dataset version; seeds and config
    # content are not part of b-hive output paths, so an existing run directory is never reused.
    header[-1:-1] = [
        'guard_run() { for t in TrainingTask InferenceTask; do if [ -e "$DATA_PATH/$t/$1/$2" ]; then',
        '  echo "refusing: $DATA_PATH/$t/$1/$2 exists (stale training/inference output)" >&2; exit 1; fi; done; }']
    for var, want in (('DATA_PATH', data_path), ('B_HIVE_DIR', bhive_dir)):
        if want is not None:
            header[-1:-1] = [
                f'[ "${{{var}:-}}" = {shlex.quote(str(want))} ] || {{ echo "refusing: {var}=${{{var}:-}}, '
                f'expected {want} (export it after sourcing setup.sh)" >&2; exit 1; }}']
    return '\n'.join(header + lines) + '\n', runs


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split', required=True, help='split_folds.py output directory')
    p.add_argument('--output', required=True, help='new directory for run_folds.sh and bhive_runs.json')
    p.add_argument('--config', default='HPlusCHToWW_2dcats')
    p.add_argument('--model', default='SimpleMLP_MultiClass')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=1024)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--chunk_size', type=int, default=300)
    p.add_argument('--train_val_split', type=float, default=0.85)
    p.add_argument('--prefix', default='hwwtrainrun')
    p.add_argument('--split_seed', type=int, default=20261112,
                   help='seed of b-hive TrainingTask train/validation file shuffle (needs the HToWW b-hive copy); '
                        'negative = omit --split-seed for an older b-hive checkout')
    p.add_argument('--training_seed', type=int, default=20261112,
                   help='seed of all other b-hive TrainingTask RNGs (init, epoch order, DataLoader workers; '
                        'needs the HToWW b-hive copy); negative = omit --training-seed')
    p.add_argument('--data_path', default='/eos/user/c/cgupta/HToWW/b-hive/output',
                   help="required $DATA_PATH of the b-hive run ('' = do not check)")
    p.add_argument('--bhive_dir', default='/eos/user/c/cgupta/HToWW/b-hive',
                   help="required $B_HIVE_DIR (config source) of the b-hive run ('' = do not check)")
    a = p.parse_args()
    if not 0 < a.train_val_split < 1:
        raise ValueError('train_val_split must leave a non-empty validation share')
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    script, runs = build(a.split, a.config, a.model, a.epochs, a.batch_size, a.lr, a.workers,
                         a.chunk_size, a.train_val_split, a.prefix,
                         None if a.split_seed < 0 else a.split_seed,
                         None if a.training_seed < 0 else a.training_seed,
                         a.data_path or None, a.bhive_dir or None)
    out.mkdir(parents=True)
    (out / 'run_folds.sh').write_text(script)
    (out / 'bhive_runs.json').write_text(json.dumps(runs, indent=2) + '\n')
    print(out / 'run_folds.sh')
