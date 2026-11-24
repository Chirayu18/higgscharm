#!/usr/bin/env python3
"""Export b-hive fold models (best_model.pt) to ONNX and record torch/ONNX agreement.

Run in the b-hive environment (torch, onnx, onnxruntime, pyarrow), not the coffea
container. For fold i it:
  1. builds the b-hive model exactly like analysis/postprocess/inference.py
     (ConfigLoader + BTaggingModels, load model_state_dict, eval, for_inference=True);
  2. exports a single-input wrapper: float32 (N, n_features) in config
     global_features order -> class probabilities (N, n_classes) in config truths
     order. With n_cpf = n_npf = 0 the b-hive forward is softmax(net(cat(global,
     empty, empty))) = softmax(net(global)); the wrapper calls net directly so the
     ONNX graph has no zero-size tensors;
  3. reads the fold's held-out test shards (model_i/test.txt), feeds the same
     float32 rows to the ORIGINAL torch call (model((X, zeros(N,0,0), zeros(N,0,0))))
     and to onnxruntime, and records max |p_torch - p_onnx|.
The output json is what register_folds.py --validation expects
(max_abs_onnx_torch_diff: one value per fold, <= 1e-5). Inputs are not
modified: no fill or clipping, nonfinite feature values are an error (TRAIN
writes nonfinite features as 0).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

TOLERANCE = 1e-5


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_features(paths, features):
    import pyarrow.parquet as pq
    blocks = []
    for path in paths:
        table = pq.read_table(path, columns=list(features))
        blocks.append(np.column_stack([table[f].to_numpy() for f in features]).astype(np.float32))
    X = np.concatenate(blocks) if blocks else np.zeros((0, len(features)), np.float32)
    if not np.all(np.isfinite(X)):
        raise ValueError('nonfinite feature values in the test shards')
    return X


def export_and_compare(wrapper, reference, X, onnx_path, opset=17, batch=8192):
    """Export `wrapper` (torch, x -> probabilities) to onnx_path and compare onnxruntime
    against `reference(x)` (the original torch call) on X. Returns a result dict."""
    import torch
    import onnxruntime as ort
    wrapper.eval()
    n_features = X.shape[1]
    torch.onnx.export(wrapper, torch.zeros(2, n_features), str(onnx_path), opset_version=opset,
                      input_names=['features'], output_names=['probabilities'],
                      dynamic_axes={'features': {0: 'batch'}, 'probabilities': {0: 'batch'}})
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(onnx_path), opts, providers=['CPUExecutionProvider'])
    name = sess.get_inputs()[0].name
    max_diff, max_wrap_diff, max_sum_dev = 0.0, 0.0, 0.0
    with torch.no_grad():
        for start in range(0, len(X), batch):
            x = X[start:start + batch]
            p_ref = reference(torch.from_numpy(x)).cpu().numpy()
            p_wrap = wrapper(torch.from_numpy(x)).cpu().numpy()
            p_onnx = np.asarray(sess.run(None, {name: x})[0])
            if p_onnx.shape != p_ref.shape:
                raise ValueError(f'shape mismatch onnx {p_onnx.shape} vs torch {p_ref.shape}')
            max_diff = max(max_diff, float(np.max(np.abs(p_onnx - p_ref))))
            max_wrap_diff = max(max_wrap_diff, float(np.max(np.abs(p_wrap - p_ref))))
            max_sum_dev = max(max_sum_dev, float(np.max(np.abs(p_onnx.sum(1) - 1.0))))
    return dict(n_events=int(len(X)), max_abs_onnx_torch_diff=max_diff,
                max_abs_wrapper_torch_diff=max_wrap_diff, max_abs_prob_sum_minus_1=max_sum_dev,
                onnx_sha256=sha256(onnx_path), opset=opset)


def load_bhive_model(bhive_path, config_name, model_name, checkpoint):
    import torch
    bhive_path = str(Path(bhive_path).resolve())
    if bhive_path not in sys.path:
        sys.path.insert(0, bhive_path)
    # ConfigLoader reads $B_HIVE_DIR/config: refuse a different checkout (e.g. the v11 area).
    env = os.environ.get('B_HIVE_DIR')
    if env and str(Path(env).resolve()) != bhive_path:
        raise ValueError(f'B_HIVE_DIR={env} is not --bhive_path {bhive_path}')
    os.environ['B_HIVE_DIR'] = bhive_path
    from utils.config.config_loader import ConfigLoader
    from utils.models.models import BTaggingModels
    config = ConfigLoader.load_config(config_name)
    if config.get('n_cpf_candidates', 0) or config.get('n_npf_candidates', 0):
        raise ValueError('wrapper assumes n_cpf_candidates = n_npf_candidates = 0')
    model = BTaggingModels(model_name, config)
    model.create_integers_defaults()
    model.create_feature_shapes()
    state = torch.load(checkpoint, map_location=torch.device('cpu'), weights_only=False)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    model.for_inference = True

    class Wrapper(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            return torch.softmax(self.net(x), dim=1)

    def reference(x):  # the original b-hive call, as in analysis/postprocess/inference.py
        empty = torch.zeros(x.shape[0], 0, 0)
        return model((x, empty, empty))

    return config, model, Wrapper(model.net), reference


def training_record(checkpoint):
    """Seeds/status/hash summary of the b-hive training_provenance.json next to a
    checkpoint (None for runs made before it existed)."""
    path = Path(checkpoint).resolve().parent / 'training_provenance.json'
    if not path.exists():
        return None
    rec = json.loads(path.read_text())
    keep = ('status', 'split_seed', 'training_seed', 'config_sha256', 'default_config_sha256',
            'input_file_list_sha256', 'training_files_sha256', 'validation_files_sha256',
            'lz4dataset_sha256')
    out = {k: rec.get(k) for k in keep}
    sw = rec.get('software') or {}
    # 2026-11-12 dup fix: b-hive LZ4Dataset emits validation rows once (best-epoch selection)
    out.update(lz4dataset_test_leftover_emitted_once=sw.get('lz4dataset_test_leftover_emitted_once'),
               bhive_git_diff_sha256=sw.get('bhive_git_diff_sha256'))
    out.update(path=str(path), sha256=sha256(path),
               best_model_sha256=(rec.get('outputs') or {}).get('best_model'))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bhive_path', required=True)
    p.add_argument('--config', default='HPlusCHToWW_2dcats')
    p.add_argument('--model_name', default='SimpleMLP_MultiClass')
    p.add_argument('--checkpoints', nargs='+', required=True, help='best_model.pt of fold 0..k-1, in order')
    p.add_argument('--split', required=True, help='split_folds.py output directory (model_<i>/test.txt)')
    p.add_argument('--output', required=True, help='new directory for fold_<i>.onnx and export_validation.json')
    p.add_argument('--opset', type=int, default=17)
    p.add_argument('--max_events', type=int, default=-1, help='compare on at most N test events per fold (-1 all)')
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    split = Path(a.split)
    manifest = json.loads((split / 'split_manifest.json').read_text())
    k = manifest['folds']
    if len(a.checkpoints) != k:
        raise ValueError(f'need {k} checkpoints, one per held-out fold')
    import torch
    torch.set_num_threads(1)
    out.mkdir(parents=True)
    folds = []
    for i, ckpt in enumerate(a.checkpoints):
        m = manifest['models'][str(i)]
        test_list = split / m['test_list']
        if sha256(test_list) != m['test_list_sha256']:
            raise ValueError(f'{test_list} changed after the split manifest was written')
        paths = test_list.read_text().split()
        if any(f'_fold_{i}.parquet' not in x for x in paths):
            raise ValueError(f'{test_list} must hold only fold {i}')
        config, model, wrapper, reference = load_bhive_model(a.bhive_path, a.config, a.model_name, ckpt)
        X = read_features(paths, config['global_features'])
        if a.max_events >= 0:
            X = X[:a.max_events]
        res = export_and_compare(wrapper, reference, X, out / f'fold_{i}.onnx', a.opset)
        bhive = Path(a.bhive_path).resolve()
        res.update(config_sha256=sha256(bhive / 'config' / f'{a.config}.yml'),
                   default_config_sha256=sha256(bhive / 'config' / 'default.yml'),
                   training_provenance=training_record(ckpt))
        res.update(fold=i, checkpoint=str(Path(ckpt).resolve()), checkpoint_sha256=sha256(ckpt),
                   test_list=str(test_list), test_list_sha256=m['test_list_sha256'],
                   classes=[t.removeprefix('is_') for t in config['truths']],
                   features=list(config['global_features']))
        folds.append(res)
        print(f"fold {i}: {res['n_events']} test events, max|onnx-torch| = {res['max_abs_onnx_torch_diff']:.3g}")
    record = dict(schema_version=1, tolerance=TOLERANCE, bhive_path=str(Path(a.bhive_path).resolve()),
                  config=a.config, model_name=a.model_name, split_filelist_sha256=manifest['filelist_sha256'],
                  max_abs_onnx_torch_diff=[f['max_abs_onnx_torch_diff'] for f in folds], folds=folds,
                  torch_version=torch.__version__)
    (out / 'export_validation.json').write_text(json.dumps(record, indent=2) + '\n')
    bad = [f['fold'] for f in folds if not f['max_abs_onnx_torch_diff'] <= TOLERANCE]
    if bad:
        raise SystemExit(f'folds {bad} exceed the {TOLERANCE} torch/ONNX tolerance; see {out}/export_validation.json')
    print(out / 'export_validation.json')


if __name__ == '__main__':
    main()
