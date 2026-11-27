#!/usr/bin/env python3
"""Freeze externally trained/exported ONNX folds with their split provenance.

Only use after validating the exports against the training implementation.
This records the declared held-out folds; it cannot independently prove how an
external training command used its inputs. The frozen training manifest remains
part of the review record.

Split schema 2: model i was trained by b-hive on model_i/train.txt (all folds
except i; b-hive's TrainingTask owns the internal validation split) and is
evaluated/scored only on held-out fold i. The b-hive config actually used is
checked against FEATURES and the class order, and its hash is frozen.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import yaml
from analysis.features.hww import FEATURES, signature, selection_signature, external_inputs
from analysis.models.registry import FoldModels
from analysis.workflows.config import WorkflowConfigBuilder


def check_bhive_config(path, classes):
    cfg = yaml.safe_load(Path(path).read_text())
    if list(cfg.get('global_features', [])) != FEATURES:
        raise ValueError(f'{path}: global_features differ from analysis FEATURES order')
    if list(cfg.get('truths', [])) != ['is_' + c for c in classes]:
        raise ValueError(f'{path}: truths {cfg.get("truths")} differ from classes {list(classes)}')
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_onnx_io(path, n_features, n_classes):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    inputs, outputs = sess.get_inputs(), sess.get_outputs()
    if len(inputs) != 1 or inputs[0].shape[-1] != n_features:
        raise ValueError(f'{path}: expected one input with {n_features} features, got {[i.shape for i in inputs]}')
    if outputs[0].shape[-1] != n_classes:
        raise ValueError(f'{path}: expected {n_classes} output classes, got {outputs[0].shape}')


def check_export_record(validation, models, bhive_sha, split, production):
    """Tie the export record (export_fold_onnx.py) to the files being frozen (2026-11-12).

    Always: a recorded per-fold onnx_sha256 must equal the model given for that fold.
    Production additionally requires the full record: per-fold ONNX hashes, the same
    b-hive config hash as --bhive_config, the split's filelist hash, and a completed
    b-hive training_provenance.json whose best_model hash is the exported checkpoint,
    with identical seeds/config/default-config hashes across folds. Since the b-hive
    LZ4Dataset dup fix (2026-11-12) production also requires each fold's training to
    record lz4dataset_test_leftover_emitted_once=True and the same LZ4Dataset hash."""
    folds = validation.get('folds') or []
    if folds or production:
        if [f.get('fold') for f in folds] != list(range(len(models))):
            raise ValueError('export record must list folds 0..k-1 in order')
        for f, src in zip(folds, models):
            want = f.get('onnx_sha256')
            if (want or production) and want != hashlib.sha256(Path(src).read_bytes()).hexdigest():
                raise ValueError(f'{src}: sha256 differs from the export record of fold {f["fold"]}')
    if not production:
        return
    if validation.get('split_filelist_sha256') != split['filelist_sha256']:
        raise ValueError('export record was made for a different split')
    seeds = set()
    for f in folds:
        if f.get('config_sha256') != bhive_sha:
            raise ValueError(f'fold {f["fold"]}: exported with a b-hive config other than --bhive_config')
        t = f.get('training_provenance') or {}
        if t.get('status') != 'completed' or t.get('best_model_sha256') != f.get('checkpoint_sha256'):
            raise ValueError(f'fold {f["fold"]}: needs a completed b-hive training_provenance.json '
                             'whose best_model is the exported checkpoint')
        if t.get('config_sha256') != bhive_sha:
            raise ValueError(f'fold {f["fold"]}: trained with a b-hive config other than --bhive_config')
        if t.get('lz4dataset_test_leftover_emitted_once') is not True or not t.get('lz4dataset_sha256'):
            raise ValueError(f'fold {f["fold"]}: trained without the b-hive LZ4Dataset validation '
                             'duplication fix (2026-11-12) or without its recorded hash')
        seeds.add((t.get('split_seed'), t.get('training_seed'), t.get('default_config_sha256'),
                   t.get('lz4dataset_sha256')))
    if len(seeds) != 1:
        raise ValueError(f'folds were trained with different seeds/default configs/LZ4Dataset code: '
                         f'{sorted(seeds, key=str)}')


def git_state():
    root = Path(__file__).resolve().parents[2]
    run = lambda *c: subprocess.check_output(['git', *c], text=True, cwd=root)
    status = run('status', '--porcelain', '--untracked-files=no')
    return dict(git_commit=run('rev-parse', 'HEAD').strip(), git_dirty=bool(status.strip()),
                git_diff_sha256=hashlib.sha256(run('diff', 'HEAD').encode()).hexdigest())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('-w','--workflow',required=True)
    p.add_argument('--split_manifest',required=True)
    p.add_argument('--models',nargs='+',required=True,help='fold_0.onnx ... fold_k-1.onnx in held-out-fold order')
    p.add_argument('--eras',nargs='+',required=True)
    p.add_argument('--validation',required=True,help='JSON with max_abs_onnx_torch_diff <= 1e-5 for each exported fold')
    p.add_argument('--bhive_config',required=True,help='the b-hive config YAML used for every fold')
    p.add_argument('--bhive_runs',help='optional JSON recording each fold\'s b-hive training/dataset versions')
    p.add_argument('--allow_dirty',action='store_true',help='record (do not refuse) an uncommitted analysis tree')
    p.add_argument('--diagnostic_only',action='store_true',
                   help='TEST ONLY: write diagnostic_only=true (RUN then needs --allow_diagnostic_model and '
                        'refuses --submit); a split whose features/selection hashes differ from the current '
                        'workflow is recorded in training_data instead of refused')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    output=Path(a.output)
    if output.exists():raise FileExistsError(f'Refusing to overwrite frozen model: {output}')
    cfg=WorkflowConfigBuilder(a.workflow).build_workflow_config()
    classes=list(cfg.combine['classes'])
    if list(cfg.mva['labels']['process_groups']) != classes:
        raise ValueError('mva.labels.process_groups order differs from combine.classes')
    split=json.loads(Path(a.split_manifest).read_text())
    if split.get('schema_version') != 2 or split.get('validation', {}).get('owner') != 'b-hive TrainingTask':
        raise ValueError('Split manifest predates the single-validation-owner scheme; regenerate it')
    k=split['folds']
    split_matches=(split['features_hash']==signature(cfg) and split['selection_hash']==selection_signature(cfg))
    if not split_matches and not a.diagnostic_only:
        raise ValueError('TRAIN split differs from current workflow features/selection')
    for i in range(k):
        m=split['models'][str(i)]
        if m['test_fold']!=i or i in m['train_folds'] or sorted(m['train_folds']+[i])!=list(range(k)):
            raise ValueError(f'Split manifest model {i} does not hold out exactly fold {i}')
    if len(a.models)!=k:raise ValueError('Exactly one ONNX model per held-out fold is required')
    validation=json.loads(Path(a.validation).read_text())
    diffs=validation.get('max_abs_onnx_torch_diff',[])
    if len(diffs)!=k or any(not 0 <= x <= 1e-5 for x in diffs):
        raise ValueError('Every fold needs a recorded ONNX/training-code equivalence result <=1e-5')
    bhive_sha=check_bhive_config(a.bhive_config, classes)
    for src in a.models:
        check_onnx_io(src, len(FEATURES), len(classes))
    check_export_record(validation, a.models, bhive_sha, split, production=not a.diagnostic_only)
    git=git_state()
    if git['git_dirty'] and not a.allow_dirty:
        raise ValueError('Analysis tree has uncommitted changes; commit first or pass --allow_dirty to record them')
    output.mkdir(parents=True)
    records=[]
    for i,src in enumerate(a.models):
        dest=output/f'fold_{i}.onnx';shutil.copy2(src,dest)
        records.append(dict(fold=i,excluded_fold=i,path=dest.name,sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
    shutil.copy2(a.split_manifest,output/'split_manifest.json')
    shutil.copy2(a.validation,output/'export_validation.json')
    shutil.copy2(a.bhive_config,output/'bhive_config.yml')
    if a.bhive_runs:
        shutil.copy2(a.bhive_runs,output/'bhive_runs.json')
    meta=dict(schema_version=1,features=FEATURES,features_hash=signature(cfg),
              selection_hash=selection_signature(cfg),classes=classes,
              fold_field='event',folds=k,diagnostic_only=bool(a.diagnostic_only),models=records,
              training_data=dict(workflow=a.workflow,eras=a.eras,
                                 filelist_sha256=split['filelist_sha256'],
                                 split_schema=2,validation_owner=split['validation']['owner'],
                                 bhive_config_sha256=bhive_sha,**git),
              external_inputs=external_inputs(cfg))
    if a.diagnostic_only:
        meta['training_data'].update(split_features_hash=split['features_hash'],
                                     split_selection_hash=split['selection_hash'],
                                     split_matches_workflow=split_matches)
    path=output/'meta.yaml';path.write_text(yaml.safe_dump(meta,sort_keys=False))
    FoldModels(path,signature(cfg),selection_signature(cfg),classes,k,allow_diagnostic=bool(a.diagnostic_only))
    print(path)


if __name__=='__main__':main()
