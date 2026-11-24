"""Strict ONNX k-fold registry; no fallback to an old/in-sample model."""
from pathlib import Path
import hashlib
import yaml
import numpy as np
from analysis.features.hww import FEATURES, event_folds


class FoldModels:
    def __init__(self, manifest, features_hash, selection_hash, classes, folds=5,
                 allow_diagnostic=False, year=None):
        self.path = Path(manifest).resolve()
        self.meta = yaml.safe_load(self.path.read_text())
        m = self.meta
        if m.get('schema_version') != 1:
            raise ValueError('Unsupported model manifest schema')
        if m.get('features') != FEATURES or m.get('features_hash') != features_hash:
            raise ValueError('Frozen model features/hash differ from this workflow')
        if m.get('selection_hash') != selection_hash:
            raise ValueError('Frozen model selection/corrections differ from this workflow')
        if m.get('classes') != list(classes) or m.get('folds') != folds:
            raise ValueError('Frozen model class order/fold scheme mismatch')
        if m.get('fold_field') != 'event':
            raise ValueError('Only event modulo k fold assignment is supported')
        if m.get('diagnostic_only', False) and not allow_diagnostic:
            raise ValueError('Diagnostic model is forbidden for production; use --allow_diagnostic_model for a local test')
        if not m.get('diagnostic_only', False):
            provenance = m.get('training_data', {})
            if any(not provenance.get(k) for k in ('workflow', 'eras', 'git_commit', 'filelist_sha256')):
                raise ValueError('Production model needs a training-data manifest')
            # Neither hash contains the era: refuse to score an era the model never saw.
            if year is not None and year not in provenance['eras']:
                raise ValueError(f'Model trained on eras {provenance["eras"]}; refusing to score {year}')
        records = m.get('models', [])
        if len(records) != folds or {x['fold'] for x in records} != set(range(folds)):
            raise ValueError('Exactly one model for each held-out fold is required')
        self.files = {}
        for rec in records:
            fold = rec['fold']
            if rec.get('excluded_fold') != fold:
                raise ValueError(f'Model {fold} must declare its excluded training fold')
            path = (self.path.parent / rec['path']).resolve()
            if hashlib.sha256(path.read_bytes()).hexdigest() != rec.get('sha256'):
                raise ValueError(f'Model file checksum mismatch: {path}')
            self.files[fold] = path
        self.folds, self.classes, self.sessions = folds, list(classes), {}
        self.manifest_hash = hashlib.sha256(self.path.read_bytes()).hexdigest()

    def __getstate__(self):
        state = self.__dict__.copy()
        state['sessions'] = {}  # ORT sessions cannot be sent to coffea worker processes
        return state

    def score(self, X, event):
        import onnxruntime as ort
        assignment = event_folds(event, self.folds)
        if len(X) != len(assignment):
            raise ValueError('Model inputs and event IDs have different lengths')
        result = np.empty((len(X), len(self.classes)), dtype=np.float32)
        for fold in np.unique(assignment):
            fold = int(fold)
            mask = assignment == fold
            if fold not in self.sessions:
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = opts.inter_op_num_threads = 1
                self.sessions[fold] = ort.InferenceSession(str(self.files[fold]), opts,
                                                        providers=['CPUExecutionProvider'])
            sess = self.sessions[fold]
            scores = np.asarray(sess.run(None, {sess.get_inputs()[0].name: X[mask]})[0])
            if scores.shape != (int(mask.sum()), len(self.classes)):
                raise ValueError(f'Unexpected model score shape {scores.shape}')
            if (not np.all(np.isfinite(scores)) or np.any(scores < -1e-6)
                    or np.any(scores > 1 + 1e-6) or not np.allclose(scores.sum(1), 1., atol=1e-4)):
                raise ValueError('ONNX output must be finite class probabilities summing to one')
            result[mask] = scores
        return result
