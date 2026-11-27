"""G2 (2026-11-12): negrw model identity in RUN provenance and the card-builder merge guard.

Run from the repository root inside the coffea 0.7.30 container. Synthetic coffea outputs, tiny synthetic
models plus the real workflow YAMLs and the production (July) model file; no production data, no jobs.
"""
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
import joblib
import numpy as np
from coffea.util import save
from sklearn.dummy import DummyClassifier

ROOT = Path(__file__).resolve().parents[1]
YEAR = '2022postEE'
JULY = Path('/afs/cern.ch/user/c/cgupta/negrw_model/negrw_models.joblib')
JULY_SHA = '05567a8c'
V2 = Path('/afs/cern.ch/user/c/cgupta/negrw_model/negrw_models_v2_20261112.joblib')
SHA_A, SHA_B = 'a' * 64, 'b' * 64


def import_script(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def workflow(name):
    from analysis.workflows.config import WorkflowConfigBuilder
    return WorkflowConfigBuilder(name).build_workflow_config()


def tiny_bundle(p_plus):
    m = DummyClassifier(strategy='prior').fit(np.zeros((10, 1)), [1] * int(10 * p_plus) + [0] * (10 - int(10 * p_plus)))
    return {'models': [m], 'features': ['f']}


def processor(mode, negrw):
    from analysis.processors.base import BaseProcessor
    proc = BaseProcessor.__new__(BaseProcessor)
    proc.mode, proc._negrw_bundle, proc._negrw_sha256 = mode, None, None
    proc.workflow_config = types.SimpleNamespace(negrw=negrw)
    return proc


class TestRunIdentity(unittest.TestCase):
    def test_identity_is_the_consumed_bytes_even_if_file_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'm.joblib'
            joblib.dump(tiny_bundle(0.8), path)
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            proc = processor('run', {'model': str(path)})
            g, _ = proc._score_negrw(proc.workflow_config.negrw, {'f': np.zeros(3)})
            np.testing.assert_allclose(g, 0.6)
            self.assertEqual(proc._negrw_model_identity(), sha)
            joblib.dump(tiny_bundle(0.5), path)  # file replaced mid-job: identity and scores stay the loaded ones
            self.assertNotEqual(hashlib.sha256(path.read_bytes()).hexdigest(), sha)
            g, _ = proc._score_negrw(proc.workflow_config.negrw, {'f': np.zeros(3)})
            np.testing.assert_allclose(g, 0.6)
            self.assertEqual(proc._negrw_model_identity(), sha)
            # identity before any scoring (non-vjets chunk first) loads the same bundle
            fresh = processor('run', {'model': str(path)})
            self.assertEqual(fresh._negrw_model_identity(), hashlib.sha256(path.read_bytes()).hexdigest())
            np.testing.assert_allclose(fresh._score_negrw(fresh.workflow_config.negrw, {'f': np.zeros(2)})[0], 0.0)

    def test_disabled_is_explicit(self):
        for negrw in (None, {}):
            self.assertEqual(processor('run', negrw)._negrw_model_identity(), 'disabled')

    def test_train_mode_keeps_path_load_and_records_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'm.joblib'
            joblib.dump(tiny_bundle(0.8), path)
            proc = processor('train', {'model': str(path)})
            g, _ = proc._score_negrw(proc.workflow_config.negrw, {'f': np.zeros(3)})
            np.testing.assert_allclose(g, 0.6)
            self.assertIsNone(proc._negrw_sha256)

    def test_production_model_bytes_load_scores_identically(self):
        if not JULY.is_file():
            self.skipTest('production negrw model not readable')
        cfg = workflow('hww_allfix_4fs').negrw
        self.assertEqual(Path(cfg['model']), JULY)  # no model switch
        run, legacy = processor('run', cfg), processor('train', cfg)
        rng = np.random.default_rng(3)
        feats = joblib.load(JULY)['features']
        X = {f: rng.uniform(0, 200, 50).astype(np.float32) for f in feats}
        a, b = run._score_negrw(cfg, X), legacy._score_negrw(cfg, X)
        np.testing.assert_array_equal(a[0], b[0]); np.testing.assert_array_equal(a[1], b[1])
        self.assertTrue(run._negrw_model_identity().startswith(JULY_SHA))


class TestMergeGuard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = import_script('build_cards_g2', 'scripts/combine/build_cards_from_templates.py')

    def write(self, tmp, name, sha=None, mode='run', drop=False, fh='h', sh='s'):
        prov = {'mode': {mode}, 'year': {YEAR}, 'features_hash': {fh}, 'selection_hash': {sh}, 'folds': {5}}
        if not drop:
            prov['negrw_model_sha256'] = {sha}
        out = {'provenance': prov, 'execution_checks': {'files': {name: {f'/store/{name}.root'}}},
               'samples': {name: {'chunks_read': 1, 'chunks': {name + '-c0'}}}}
        path = Path(tmp) / f'{name}.coffea'
        save(out, str(path))
        return path

    def test_matching_merges(self):
        with tempfile.TemporaryDirectory() as t:
            for sha in (SHA_A, 'disabled'):
                res = self.mod.merge_outputs([self.write(t, f'x{sha[:3]}', sha), self.write(t, f'y{sha[:3]}', sha)])
                self.assertEqual(res['provenance']['negrw_model_sha256'], {sha})

    def test_mixed_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            for other in (SHA_B, 'disabled'):
                with self.assertRaisesRegex(ValueError, 'Incompatible RUN provenance: negrw_model_sha256'):
                    self.mod.merge_outputs([self.write(t, 'x', SHA_A), self.write(t, 'y' + other[:3], other)])

    def test_missing_or_malformed_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            for files in ([self.write(t, 'x', SHA_A), self.write(t, 'y', drop=True)],
                          [self.write(t, 'z', drop=True)]):
                with self.assertRaisesRegex(ValueError, 'no negrw_model_sha256'):
                    self.mod.merge_outputs(files)
            with self.assertRaisesRegex(ValueError, 'malformed negrw_model_sha256'):
                self.mod.merge_outputs([self.write(t, 'w', 'July')])
            # non-RUN outputs keep their existing refusal
            with self.assertRaisesRegex(ValueError, 'Only RUN outputs'):
                self.mod.merge_outputs([self.write(t, 'v', mode='train', drop=True)])

    def test_expected_identity_and_main_refuses_other_model(self):
        from analysis.features.hww import signature, selection_signature
        cfg = workflow('hww_allfix_4fs')
        self.assertTrue(self.mod.negrw_identity(cfg).startswith(JULY_SHA))
        off = deepcopy(cfg); off.negrw = None
        self.assertEqual(self.mod.negrw_identity(off), 'disabled')
        fh, sh = signature(cfg), selection_signature(cfg)
        with tempfile.TemporaryDirectory() as t:
            for sha, err in ((SHA_B, 'differs from the workflow negrw.model'), (self.mod.negrw_identity(cfg), None)):
                f = self.write(t, f'in{sha[:4]}', sha, fh=fh, sh=sh)
                argv = sys.argv
                sys.argv = ['b', '-w', 'hww_allfix_4fs', '-y', YEAR, '--inputs', str(f), '--output', str(Path(t) / f'o{sha[:4]}')]
                try:
                    with self.assertRaises(Exception) as ctx:
                        self.mod.main()  # synthetic outputs never pass coverage; only the negrw step is under test
                finally:
                    sys.argv = argv
                if err:
                    self.assertIn(err, str(ctx.exception))
                else:
                    self.assertNotIn('negrw model', str(ctx.exception))


class TestTrainHashesUnchanged(unittest.TestCase):
    def test_ab_hashes_and_model_switch_invariance(self):
        from analysis.features.hww import signature, selection_signature
        for wf, (fh, sh) in {'hww_allfix_4fs': ('36f655f4', '80e75bca'),
                             'hww_routeb_3fs4fs': ('36f655f4', '24b63ba3')}.items():
            cfg = workflow(wf)
            self.assertEqual((signature(cfg)[:8], selection_signature(cfg)[:8]), (fh, sh), wf)
            if V2.is_file():
                alt = deepcopy(cfg); alt.negrw = dict(cfg.negrw, model=str(V2))
                self.assertEqual((signature(alt)[:8], selection_signature(alt)[:8]), (fh, sh), wf)


if __name__ == '__main__':
    unittest.main()
