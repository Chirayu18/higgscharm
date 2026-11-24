"""b-hive fold pipeline glue (2026-11-12): seeds, run guards, diagnostic registration, ONNX export.

torch-dependent tests skip in the coffea container; run them in the b-hive env.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from analysis.features.hww import FEATURES, event_folds

ROOT = Path(__file__).resolve().parents[1]
WF = 'hww_allfix_4fs'
YEAR = '2022postEE'


def import_script(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def train_shard(path, events, label, sample='S'):
    events = np.asarray(events, dtype=np.uint64)
    arrays = {k: np.ones(len(events), dtype=np.float32) for k in FEATURES}
    arrays.update(event=events, fold=event_folds(events), label=np.asarray(label, dtype=np.int16),
                  weight_nominal=np.ones(len(events)))
    table = pa.table(arrays).replace_schema_metadata({
        b'processing_mode': b'train', b'features_hash': b'h', b'selection_hash': b's', b'folds': b'5',
        b'year': YEAR.encode(), b'sample': sample.encode()})
    pq.write_table(table, path)


def make_split(d):
    split = import_script('split_folds_mva', 'scripts/mva/split_folds.py')
    p = Path(d) / 'in.parquet'
    train_shard(p, np.arange(50), np.zeros(50))
    split.split_files([p], Path(d) / 'split', 5, {(YEAR, 'S'): 1.})
    return Path(d) / 'split'


class TestSplitSeedCommands(unittest.TestCase):
    def test_seed_is_passed_to_training_and_inference_and_recorded(self):
        gen = import_script('bhive_fold_commands_mva', 'scripts/mva/bhive_fold_commands.py')
        with tempfile.TemporaryDirectory() as d:
            split = make_split(d)
            args = (split, 'HPlusCHToWW_2dcats', 'SimpleMLP_MultiClass', 2, 1024, 1e-3, 4, 300, 0.85, 'x')
            old, _ = gen.build(*args)  # backward compatible: no seed -> old command line
            self.assertNotIn('--split-seed', old)
            script, runs = gen.build(*args, split_seed=123)
            for line in script.splitlines():
                if line.startswith('law run TrainingTask') or line.startswith('law run InferenceTask'):
                    self.assertIn('--split-seed 123', line)
                if line.startswith('law run DatasetConstructorTask'):
                    self.assertNotIn('--split-seed', line)
            self.assertEqual(script.count('--split-seed 123'), 10)
            self.assertEqual({r['split_seed'] for r in runs.values()}, {123})
            self.assertEqual(script.replace(' --split-seed 123', ''), old)


class TestGeneratedScriptGuards(unittest.TestCase):
    """c1 resume 2026-11-12: training seed flag, environment checks and run-directory guard,
    exercised by executing the generated script with a stub `law`."""

    def setUp(self):
        self.gen = import_script('bhive_fold_commands_mva2', 'scripts/mva/bhive_fold_commands.py')
        self.tmp = tempfile.TemporaryDirectory(); d = Path(self.tmp.name)
        self.split = make_split(d)
        self.data, self.bhive, self.bin = d / 'data', d / 'bhive', d / 'bin'
        for x in (self.data, self.bhive, self.bin):
            x.mkdir()
        (self.bin / 'law').write_text('#!/bin/bash\necho "$@" >> "$LAW_LOG"\n'); (self.bin / 'law').chmod(0o755)
        self.log = d / 'law.log'
        self.script, self.runs = self.gen.build(
            self.split, 'HPlusCHToWW_2dcats', 'SimpleMLP_MultiClass', 2, 1024, 1e-3, 4, 300, 0.85, 'x',
            split_seed=7, training_seed=11, data_path=self.data, bhive_dir=self.bhive)
        (d / 'run.sh').write_text(self.script)

    def tearDown(self):
        self.tmp.cleanup()

    def run_script(self, data_path=None, bhive_dir=None):
        import os
        env = dict(os.environ, PATH=f'{self.bin}:{os.environ["PATH"]}', LAW_LOG=str(self.log),
                   DATA_PATH=str(data_path or self.data), B_HIVE_DIR=str(bhive_dir or self.bhive))
        if self.log.exists():
            self.log.unlink()
        res = subprocess.run(['bash', str(Path(self.tmp.name) / 'run.sh')], env=env, capture_output=True, text=True)
        calls = self.log.read_text().splitlines() if self.log.exists() else []
        return res, calls

    def test_seeds_on_training_and_inference_only_and_recorded(self):
        for line in self.script.splitlines():
            if line.startswith(('law run TrainingTask', 'law run InferenceTask')):
                self.assertIn('--split-seed 7 --training-seed 11', line)
            if line.startswith('law run DatasetConstructorTask'):
                self.assertNotIn('seed', line)
        self.assertEqual(self.script.count('--training-seed 11'), 10)
        self.assertEqual({(r['split_seed'], r['training_seed']) for r in self.runs.values()}, {(7, 11)})
        self.assertEqual({r['data_path'] for r in self.runs.values()}, {str(self.data)})
        res = subprocess.run(['bash', '-n'], input=self.script, text=True, capture_output=True)
        self.assertEqual(res.returncode, 0, res.stderr)

    def test_runs_all_twenty_commands_in_the_intended_environment(self):
        res, calls = self.run_script()
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(len(calls), 20)
        self.assertEqual(sum(c.startswith('run TrainingTask') for c in calls), 5)

    def test_wrong_data_path_or_bhive_dir_refused_before_any_law_call(self):
        other = Path(self.tmp.name) / 'v11_output'; other.mkdir()
        for kw in (dict(data_path=other), dict(bhive_dir=other)):
            res, calls = self.run_script(**kw)
            self.assertNotEqual(res.returncode, 0)
            self.assertIn('refusing', res.stderr)
            self.assertEqual(calls, [])

    def test_existing_training_or_inference_output_refused(self):
        r = self.runs[2]
        for task in ('TrainingTask', 'InferenceTask'):
            stale = self.data / task / 'HPlusCHToWW_2dcats' / r['dataset_version']
            stale.mkdir(parents=True)
            res, calls = self.run_script()
            self.assertNotEqual(res.returncode, 0)
            self.assertIn('stale training/inference output', res.stderr)
            self.assertEqual(len(calls), 8)  # models 0 and 1 ran; model 2 refused before its first command
            shutil.rmtree(self.data / task)


class TestExportRecordChecks(unittest.TestCase):
    """register_folds.check_export_record (c1 resume 2026-11-12)."""

    def setUp(self):
        self.reg = import_script('register_folds_mva2', 'scripts/mva/register_folds.py')
        self.models = [str(ROOT / 'tests/data/mode_models' / f'fold_{i}.onnx') for i in range(5)]
        sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
        self.split = dict(filelist_sha256='F')
        prov = dict(status='completed', split_seed=1, training_seed=2, config_sha256='C',
                    default_config_sha256='D', lz4dataset_sha256='L',
                    lz4dataset_test_leftover_emitted_once=True)
        self.record = dict(split_filelist_sha256='F', max_abs_onnx_torch_diff=[0.0] * 5, folds=[
            dict(fold=i, onnx_sha256=sha(m), config_sha256='C', checkpoint_sha256=f'k{i}',
                 training_provenance=dict(prov, best_model_sha256=f'k{i}')) for i, m in enumerate(self.models)])

    def check(self, record, production=True, models=None):
        return self.reg.check_export_record(record, models or self.models, 'C', self.split, production)

    def test_complete_record_passes(self):
        self.check(self.record)

    def test_swapped_onnx_refused_even_for_diagnostic(self):
        swapped = self.models[1:] + self.models[:1]
        if len({hashlib.sha256(Path(m).read_bytes()).hexdigest() for m in self.models}) == 1:
            self.skipTest('fixture folds are byte-identical')
        for production in (True, False):
            with self.assertRaisesRegex(ValueError, 'differs from the export record'):
                self.check(self.record, production, swapped)

    def test_production_requires_full_provenance(self):
        import copy
        cases = {
            'no folds': lambda r: r.update(folds=[]),
            'other split': lambda r: r.update(split_filelist_sha256='G'),
            'other config': lambda r: r['folds'][3].update(config_sha256='X'),
            'no training record': lambda r: r['folds'][0].update(training_provenance=None),
            'unfinished training': lambda r: r['folds'][0]['training_provenance'].update(status='started'),
            'other checkpoint': lambda r: r['folds'][4].update(checkpoint_sha256='zz'),
            'trained with other config': lambda r: r['folds'][2]['training_provenance'].update(config_sha256='X'),
            'mixed seeds': lambda r: r['folds'][1]['training_provenance'].update(training_seed=3),
            'pre dup-fix training': lambda r: r['folds'][0]['training_provenance'].update(
                lz4dataset_test_leftover_emitted_once=None),
            'no LZ4Dataset hash': lambda r: r['folds'][0]['training_provenance'].update(lz4dataset_sha256=None),
            'mixed LZ4Dataset code': lambda r: r['folds'][3]['training_provenance'].update(lz4dataset_sha256='M'),
        }
        for name, mutate in cases.items():
            r = copy.deepcopy(self.record); mutate(r)
            with self.assertRaises(ValueError, msg=name):
                self.check(r)
        self.check({'max_abs_onnx_torch_diff': [0.0] * 5}, production=False)  # diagnostic: legacy record ok


class TestDiagnosticRegistration(unittest.TestCase):
    def run_register(self, d, split, *extra):
        cfg = import_script('register_folds_mva', 'scripts/mva/register_folds.py')
        classes = list(cfg.WorkflowConfigBuilder(WF).build_workflow_config().combine['classes'])
        bhive = Path(d) / 'bhive.yml'
        bhive.write_text(yaml.safe_dump(dict(global_features=FEATURES, truths=['is_' + c for c in classes])))
        val = Path(d) / 'val.json'
        val.write_text(json.dumps(dict(max_abs_onnx_torch_diff=[1e-7] * 5)))
        models = [str(ROOT / 'tests/data/mode_models' / f'fold_{i}.onnx') for i in range(5)]
        out = Path(d) / ('model' + '_'.join(x.strip('-') for x in extra))
        cmd = [sys.executable, str(ROOT / 'scripts/mva/register_folds.py'), '-w', WF,
               '--split_manifest', str(split / 'split_manifest.json'), '--models', *models,
               '--eras', YEAR, '--validation', str(val), '--bhive_config', str(bhive),
               '--output', str(out), '--allow_dirty', *extra]
        return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True), out

    def test_stale_split_refused_unless_diagnostic_and_then_marked(self):
        from analysis.models.registry import FoldModels
        from analysis.features.hww import signature, selection_signature
        from analysis.workflows.config import WorkflowConfigBuilder
        cfg = WorkflowConfigBuilder(WF).build_workflow_config()
        with tempfile.TemporaryDirectory() as d:
            split = make_split(d)  # hashes 'h'/'s' differ from the workflow
            res, out = self.run_register(d, split)
            self.assertNotEqual(res.returncode, 0)
            self.assertIn('TRAIN split differs from current workflow', res.stderr)
            self.assertFalse(out.exists())
            res, out = self.run_register(d, split, '--diagnostic_only')
            self.assertEqual(res.returncode, 0, res.stderr)
            meta = yaml.safe_load((out / 'meta.yaml').read_text())
            self.assertTrue(meta['diagnostic_only'])
            td = meta['training_data']
            self.assertEqual((td['split_features_hash'], td['split_selection_hash']), ('h', 's'))
            self.assertFalse(td['split_matches_workflow'])
            args = (out / 'meta.yaml', signature(cfg), selection_signature(cfg), cfg.combine['classes'], 5)
            with self.assertRaisesRegex(ValueError, 'Diagnostic model is forbidden'):
                FoldModels(*args)
            FoldModels(*args, allow_diagnostic=True)


class TestOnnxExport(unittest.TestCase):
    def test_read_features_refuses_nonfinite(self):
        exp = import_script('export_fold_onnx_mva', 'scripts/mva/export_fold_onnx.py')
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'a.parquet'
            pq.write_table(pa.table({'a': np.array([1., np.nan], np.float32), 'b': np.ones(2, np.float32)}), p)
            with self.assertRaisesRegex(ValueError, 'nonfinite'):
                exp.read_features([p], ['a', 'b'])
            pq.write_table(pa.table({'a': np.array([1., 2.], np.float32), 'b': np.zeros(2, np.float32)}), p)
            np.testing.assert_array_equal(exp.read_features([p], ['b', 'a']), [[0, 1], [0, 2]])

    def test_training_record_and_bhive_dir_mismatch(self):
        exp = import_script('export_fold_onnx_mva', 'scripts/mva/export_fold_onnx.py')
        with tempfile.TemporaryDirectory() as d:
            ckpt = Path(d) / 'best_model.pt'; ckpt.write_bytes(b'x')
            self.assertIsNone(exp.training_record(ckpt))
            (Path(d) / 'training_provenance.json').write_text(json.dumps(dict(
                status='completed', split_seed=1, training_seed=2, outputs=dict(best_model='abc'),
                lz4dataset_sha256='L', software=dict(lz4dataset_test_leftover_emitted_once=True))))
            rec = exp.training_record(ckpt)
            self.assertEqual((rec['status'], rec['training_seed'], rec['best_model_sha256']), ('completed', 2, 'abc'))
            self.assertEqual((rec['lz4dataset_sha256'], rec['lz4dataset_test_leftover_emitted_once']), ('L', True))
            try:
                import torch  # noqa: F401
            except ImportError:
                return
            import os
            old = os.environ.get('B_HIVE_DIR')
            os.environ['B_HIVE_DIR'] = str(Path(d) / 'elsewhere')
            try:
                with self.assertRaisesRegex(ValueError, 'is not --bhive_path'):
                    exp.load_bhive_model(d, 'cfg', 'model', ckpt)
            finally:
                if old is None:
                    os.environ.pop('B_HIVE_DIR')
                else:
                    os.environ['B_HIVE_DIR'] = old

    def test_export_matches_reference(self):
        try:
            import torch, onnx, onnxruntime  # noqa: F401
        except ImportError:
            self.skipTest('needs torch + onnx + onnxruntime (b-hive env)')
        exp = import_script('export_fold_onnx_mva', 'scripts/mva/export_fold_onnx.py')
        torch.manual_seed(0)
        net = torch.nn.Sequential(torch.nn.Linear(26, 8), torch.nn.BatchNorm1d(8), torch.nn.ReLU(),
                                  torch.nn.Dropout(0.2), torch.nn.Linear(8, 6)).eval()

        class W(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.net = net

            def forward(self, x):
                return torch.softmax(self.net(x), dim=1)
        X = np.random.default_rng(1).normal(size=(1000, 26)).astype(np.float32)
        with tempfile.TemporaryDirectory() as d:
            res = exp.export_and_compare(W(), lambda x: torch.softmax(net(x), 1), X, Path(d) / 'm.onnx')
            self.assertEqual(res['n_events'], 1000)
            self.assertLess(res['max_abs_onnx_torch_diff'], 1e-5)
            self.assertLess(res['max_abs_prob_sum_minus_1'], 1e-5)
            bad = exp.export_and_compare(W(), lambda x: torch.softmax(2 * net(x), 1), X, Path(d) / 'n.onnx')
            self.assertGreater(bad['max_abs_onnx_torch_diff'], 1e-3)


if __name__ == '__main__':
    unittest.main()
