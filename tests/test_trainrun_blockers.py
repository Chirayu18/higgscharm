"""Regression tests for the 2026-11-05 TRAIN/RUN blocker fixes (F1-F13).

Run from the repository root inside the coffea 0.7.30 container. Synthetic
inputs plus the real workflow/era YAMLs; no production data, no jobs.
"""
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from analysis.features.hww import FEATURES, event_folds

ROOT = Path(__file__).resolve().parents[1]
WF = 'hww_allfix_4fs'
YEAR = '2022postEE'
BHIVE_CONFIG = Path('/eos/user/c/cgupta/HToWW/b-hive/config/HPlusCHToWW_2dcats.yml')


def import_script(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def workflow():
    from analysis.workflows.config import WorkflowConfigBuilder
    return WorkflowConfigBuilder(WF).build_workflow_config()


def train_shard(path, events, label, sample='S', year=YEAR):
    arrays = {k: np.ones(len(events), dtype=np.float32) for k in FEATURES}
    arrays.update(event=np.asarray(events, dtype=np.uint64), fold=event_folds(np.asarray(events, dtype=np.uint64)),
                  label=np.asarray(label, dtype=np.int16), weight_nominal=np.ones(len(events)))
    table = pa.table(arrays).replace_schema_metadata({
        b'processing_mode': b'train', b'features_hash': b'h', b'selection_hash': b's', b'folds': b'5',
        b'year': year.encode(), b'sample': sample.encode()})
    pq.write_table(table, path)


class TestSingleValidationOwnerSplit(unittest.TestCase):
    def test_four_folds_train_one_fold_test_no_external_validation(self):
        mod = import_script('split_folds', 'scripts/mva/split_folds.py')
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'in.parquet'
            train_shard(p, np.arange(100), np.r_[np.zeros(90), np.full(10, -1)])
            info = mod.split_files([p], Path(d) / 'split', 5, {(YEAR, 'S'): 2.})
            labelled = set(range(90))
            for i in range(5):
                mdir = Path(d) / 'split' / f'model_{i}'
                self.assertEqual(sorted(x.name for x in mdir.iterdir()), ['test.txt', 'train.txt'])
                ev = {}
                for role in ('train', 'test'):
                    files = (mdir / f'{role}.txt').read_text().splitlines()
                    self.assertNotIn('', files)
                    ev[role] = {int(x) for f in files for x in pq.read_table(f)['event'].to_pylist()}
                self.assertFalse(ev['train'] & ev['test'])
                self.assertEqual(ev['test'], {x for x in labelled if x % 5 == i})
                self.assertEqual(ev['train'], {x for x in labelled if x % 5 != i})
                m = info['models'][i]
                self.assertEqual((m['test_fold'], m['train_folds']), (i, [f for f in range(5) if f != i]))
                self.assertNotIn('validation_fold', m)
                self.assertEqual(m['rows'], {'train': 72, 'test': 18})
            self.assertEqual(info['schema_version'], 2)
            self.assertEqual(info['validation']['owner'], 'b-hive TrainingTask')
            self.assertFalse(info['weight_column']['consumed_by_bhive_2dcats'])
            on_disk = json.loads((Path(d) / 'split' / 'split_manifest.json').read_text())
            self.assertEqual(on_disk['models']['0']['train_list_sha256'],
                             hashlib.sha256((Path(d) / 'split/model_0/train.txt').read_bytes()).hexdigest())

    def test_normalization_skips_real_data_eras_and_enforces_coverage(self):
        """F3: the __main__ normalization path with a real data sample (era E, xsec 'None')."""
        mod = import_script('split_folds2', 'scripts/mva/split_folds.py')
        import analysis.processing_checks as pc
        from analysis.filesets.utils import get_dataset_config
        cfg = get_dataset_config(YEAR)
        data = next(s for s, v in cfg.items() if v['era'] not in ('mc', 'signal'))
        mc = next(s for s, v in cfg.items() if v['era'] == 'mc' and v['process'] == 'tt')

        def acc(sample):
            return {'provenance': {'mode': {'train'}, 'year': {YEAR}},
                    'execution_checks': {'files': {sample: {f'/store/{sample}/a.root'}}},
                    'samples': {sample: {'sumw': 10., 'chunks': {f'{sample}-c0'}}}}
        saved = pc.expected_mc_samples, pc.fileset_files
        try:
            pc.expected_mc_samples = lambda w, y: {mc}
            pc.fileset_files = lambda y, f=None: {mc: {f'/store/{mc}/a.root'}}
            scales = mod.normalization([acc(data), acc(mc)], WF)
            self.assertEqual(set(scales), {(YEAR, mc)})
            pc.expected_mc_samples = lambda w, y: {mc, 'Other'}
            with self.assertRaisesRegex(ValueError, 'expected MC samples have no output'):
                mod.normalization([acc(mc)], WF)
            scales = mod.normalization([acc(mc)], WF, {YEAR: {'Other': 'not produced in this era (test)'}})
            self.assertEqual(set(scales), {(YEAR, mc)})
        finally:
            pc.expected_mc_samples, pc.fileset_files = saved


class TestSampleClassification(unittest.TestCase):
    def test_every_era_entry_is_mc_or_data(self):
        from analysis.filesets.utils import get_dataset_config
        from analysis.processing_checks import sample_kind
        for year in ('2022preEE', '2022postEE', '2023preBPix', '2023postBPix'):
            cfg = get_dataset_config(year)
            kinds = {s: sample_kind(s, v) for s, v in cfg.items()}
            self.assertEqual({s for s, k in kinds.items() if k == 'data'},
                             {s for s, v in cfg.items() if v['process'] == 'Data'})
            self.assertTrue(all(v['era'] != 'data' for v in cfg.values()))  # the old test never matched
        with self.assertRaises(ValueError):
            sample_kind('x', {'era': 'Z', 'process': 'tt'})

    def test_all_workflow_mc_processes_map_to_one_card_class_and_label(self):
        """F2: every MC process the workflow runs (incl. WG) maps once, consistently."""
        from analysis.filesets.utils import get_dataset_config
        from analysis.processing_checks import expected_mc_samples
        c = workflow()
        groups = c.mva['labels']['process_groups']
        self.assertEqual(list(groups), c.combine['classes'])
        for year in ('2022preEE', '2022postEE', '2023preBPix', '2023postBPix'):
            cfg = get_dataset_config(year)
            procs = {cfg[s]['process'].strip() for s in expected_mc_samples(WF, year)}
            self.assertIn('WG', procs)
            for proc in procs:
                card = [k for k, v in c.combine['process_map'].items() if proc in v]
                label = [k for k, v in groups.items() if proc in v]
                self.assertEqual(len(card), 1, (year, proc, card))
                self.assertEqual(card, label, (year, proc))
        self.assertEqual([k for k, v in groups.items() if 'WG' in v], ['diboson'])


class TestCoverage(unittest.TestCase):
    def test_missing_extra_stale_and_file_gaps_raise(self):
        from analysis.processing_checks import check_dataset_coverage
        import analysis.processing_checks as pc
        saved = pc.fileset_files
        try:
            pc.fileset_files = lambda y, f=None: {'A': {'/store/a1', '/store/a2'}, 'B': {'/store/b1'}}
            ok = {'A': {'/store/a1', '/store/a2'}, 'B': {'/store/b1'}}
            check_dataset_coverage(WF, YEAR, {'A', 'B'}, ok, expected={'A', 'B'})
            with self.assertRaisesRegex(ValueError, 'no output'):
                check_dataset_coverage(WF, YEAR, {'A'}, ok, expected={'A', 'B'})
            check_dataset_coverage(WF, YEAR, {'A'}, ok, {'B': 'approved: absent in era'}, expected={'A', 'B'})
            with self.assertRaisesRegex(ValueError, 'does not run over'):
                check_dataset_coverage(WF, YEAR, {'A', 'B'}, ok, {'C': 'r'}, expected={'A', 'B'})
            with self.assertRaisesRegex(ValueError, 'unprocessed'):
                check_dataset_coverage(WF, YEAR, {'A', 'B'}, {'A': {'/store/a1'}, 'B': {'/store/b1'}},
                                       expected={'A', 'B'})
            with self.assertRaisesRegex(ValueError, 'outside the workflow'):
                check_dataset_coverage(WF, YEAR, {'A', 'B'}, ok, expected={'A'})
        finally:
            pc.fileset_files = saved

    def test_store_path_is_replica_independent(self):
        from analysis.processing_checks import store_path
        self.assertEqual(store_path('root://xrootd-cms.infn.it//store/mc/X/a.root'), '/store/mc/X/a.root')
        self.assertEqual(store_path('root://eoscms.cern.ch//eos/cms/store/mc/X/a.root'), '/store/mc/X/a.root')


class TestCardBuilder(unittest.TestCase):
    """F1/F2/F5/F6/F7 against the real workflow and 2022postEE era config."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / 'scripts/combine'))
        cls.mod = import_script('build_cards', 'scripts/combine/build_cards_from_templates.py')
        from analysis.filesets.utils import get_dataset_config
        cls.cfg = cls.mod.configure_era(workflow(), YEAR)
        cls.datasets = get_dataset_config(YEAR)

    def record(self, values, chunks, kind='weight'):
        from analysis.sinks.hww import final_channels
        ch = final_channels(self.cfg.combine)
        nb = len(self.cfg.combine['binning']['edges']) - 1
        counts = {c: np.full(nb, float(values)) for c in ch}
        return {'counts': counts, 'sumw2': {c: np.abs(v) for c, v in counts.items()},
                'norm_raw': 1., 'norm_rw': 1., 'negrw': {False}, 'kind': {kind}, 'chunks': set(chunks)}

    def output(self, drop=None, partial=None, shifts_evaluated=True, shift_records=True):
        combine = self.cfg.combine
        cat = next(iter(self.cfg.event_selection['categories']))
        pick = {}
        for cls in combine['classes']:
            pick[cls] = next(s for s, v in self.datasets.items() if v['era'] == 'mc'
                             and v['process'].strip() in combine['process_map'][cls])
        wg = next(s for s, v in self.datasets.items() if v['process'] == 'WG')
        data = next(s for s, v in self.datasets.items() if v['process'] == 'Data')
        weights = ['nominal'] + [s + d for s in combine['shape_systematics'] for d in ('Up', 'Down')]
        samples = {}
        for sample in list(pick.values()) + [wg, data]:
            chunks = {sample + '-c0', sample + '-c1'}
            if self.datasets[sample]['process'] == 'Data':
                samples[sample] = {'sumw': 5., 'chunks': chunks, 'chunks_read': 2}
                continue
            recs = {v: self.record(1., chunks) for v in weights}
            if self.datasets[sample]['process'].strip() in combine['process_map']['vjets']:
                for d in ('Up', 'Down'):
                    recs['CMS_negrw_vjets' + d] = self.record(1., chunks)
            if shift_records:
                recs['JESUp'] = self.record(1.1, chunks, 'object')
                recs['JESDown'] = self.record(0.9, chunks, 'object')
            if partial and sample == pick['tt']:
                recs[partial]['chunks'] = {sample + '-c0'}
            if drop and sample == pick['tt']:
                recs.pop(drop)
            entry = {'sumw': 100., 'chunks': chunks, 'chunks_read': 2, 'templates': {cat: recs},
                     'evaluated_shifts': ({'JESUp': set(chunks), 'JESDown': set(chunks)}
                                          if shifts_evaluated else {})}
            samples[sample] = entry
        return {'samples': samples}, pick, wg

    def test_data_skipped_and_wg_lands_in_diboson(self):
        out, pick, wg = self.output()
        policy = {}
        hists, variations, shapes, _ = self.mod.make_process_hists(out, self.cfg, YEAR, policy=policy)
        self.assertEqual(len(policy['data_samples_skipped']), 1)
        self.assertEqual(shapes, ['JES'])
        lumi = self.mod.load_lumi(YEAR)
        per = lambda s: lumi * float(self.datasets[s]['xsec']) / 100.
        expected = per(pick['diboson']) + per(wg)
        np.testing.assert_allclose(hists['CR_diboson']['diboson']['nominal'][0], expected)

    def test_partial_weight_variation_raises(self):
        var = self.cfg.combine['shape_systematics'][0] + 'Up'
        out, _, _ = self.output(partial=var)
        with self.assertRaisesRegex(ValueError, 'covers 1 of 2 nominal chunks'):
            self.mod.make_process_hists(out, self.cfg, YEAR)

    def test_object_shift_real_zero_vs_missing(self):
        # Evaluated on every chunk but no selected shifted events: genuine zero, not nominal.
        out, pick, _ = self.output(drop='JESUp')
        hists, *_ = self.mod.make_process_hists(out, self.cfg, YEAR)
        self.assertEqual(hists['CR_tt']['tt']['JESUp'][0].sum(), 0.)
        self.assertGreater(hists['CR_tt']['tt']['nominal'][0].sum(), 0.)
        # Never evaluated: an error unless declared, then nominal with an audit entry.
        out, pick, _ = self.output(shifts_evaluated=False, shift_records=False)
        for s in out['samples']:
            if 'evaluated_shifts' in out['samples'][s] and s != pick['tt']:
                out['samples'][s]['evaluated_shifts'] = {'JESUp': out['samples'][s]['chunks'],
                                                         'JESDown': out['samples'][s]['chunks']}
        with self.assertRaisesRegex(ValueError, 'evaluated on 0 of 2'):
            self.mod.make_process_hists(out, self.cfg, YEAR)
        policy = {}
        hists, *_ = self.mod.make_process_hists(out, self.cfg, YEAR, {pick['tt']: ['JES']}, policy)
        np.testing.assert_allclose(hists['CR_tt']['tt']['JESUp'][0], hists['CR_tt']['tt']['nominal'][0])
        self.assertIn([pick['tt'], 'JESUp'], policy['declared_not_applicable_used'])

    def test_final_policy_matches_legacy_order_exactly(self):
        """F5 parity: same inputs through the legacy main() sequence and finalize_templates."""
        legacy = sys.modules['make_combine_inputs']
        rng = np.random.default_rng(7)
        channels, procs = ['SR', 'CR'], ['sig', 'vjets']
        names = ['nominal', 'pileupUp', 'pileupDown', 'JESUp', 'JESDown']
        hists = {ch: {p: {v: (rng.normal(1, 2, 6), rng.uniform(0, 1, 6)) for v in names} for p in procs}
                 for ch in channels}
        hists['SR']['vjets']['nominal'][0][-1] = -4.0          # review R2 negative bin
        hists['CR']['sig']['JESUp'] = (np.zeros(6), np.zeros(6))  # empty object shift
        hists['CR']['sig']['nominal'][0][:] = np.abs(hists['CR']['sig']['nominal'][0])
        for smooth in (False, True):
            a, b = deepcopy(hists), deepcopy(hists)
            # legacy make_combine_inputs.main order, verbatim logic
            for ch in channels:
                for cp in procs:
                    nom_c, nom_s2 = a[ch][cp]['nominal']
                    for vn in ('JESUp', 'JESDown'):
                        c, _s2 = a[ch][cp][vn]
                        if c.sum() <= 0 < nom_c.sum():
                            a[ch][cp][vn] = (nom_c.copy(), nom_s2.copy())
            pairs = [(v, '') for v in names]
            legacy.clip_negative_bins(a, channels, procs, pairs)
            if smooth:
                legacy.smooth_shape_variations(a, channels, procs, pairs)
                legacy.clip_negative_bins(a, channels, procs, pairs)
            policy = {}
            self.mod.finalize_templates(b, {'smooth_shapes': smooth}, names, ['JES'], policy)
            for ch in channels:
                for cp in procs:
                    for v in names:
                        np.testing.assert_array_equal(a[ch][cp][v][0], b[ch][cp][v][0])
                        np.testing.assert_array_equal(a[ch][cp][v][1], b[ch][cp][v][1])
            self.assertEqual(b['SR']['vjets']['nominal'][0][-1], 1e-6)
            self.assertEqual([x[:3] for x in policy['object_shift_empty_fallback']], [['CR', 'sig', 'JESUp']])
        with self.assertRaisesRegex(ValueError, 'is empty'):
            self.mod.finalize_templates(deepcopy(hists), {'object_shift_empty_fallback': 'error'},
                                        names, ['JES'], {})


class TestRegistryGuards(unittest.TestCase):
    def production_meta(self, directory, eras):
        records = []
        for fold in range(5):
            src = ROOT / 'tests/data/mode_models' / f'fold_{fold}.onnx'
            dest = Path(directory) / src.name; shutil.copyfile(src, dest)
            records.append(dict(fold=fold, excluded_fold=fold, path=dest.name,
                                sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
        meta = dict(schema_version=1, features=FEATURES, features_hash='f', selection_hash='s',
                    classes=list('abcdef'), folds=5, fold_field='event', diagnostic_only=False, models=records,
                    training_data=dict(workflow=WF, eras=eras, git_commit='x', filelist_sha256='y'))
        path = Path(directory) / 'meta.yaml'; path.write_text(yaml.safe_dump(meta)); return path

    def test_run_refuses_untrained_era(self):
        from analysis.models.registry import FoldModels
        with tempfile.TemporaryDirectory() as d:
            path = self.production_meta(d, ['2022preEE', '2022postEE'])
            FoldModels(path, 'f', 's', list('abcdef'), year='2022postEE')
            with self.assertRaisesRegex(ValueError, 'refusing to score 2023preBPix'):
                FoldModels(path, 'f', 's', list('abcdef'), year='2023preBPix')

    def test_bhive_config_and_onnx_io_checks(self):
        reg = import_script('register_folds', 'scripts/mva/register_folds.py')
        classes = workflow().combine['classes']
        self.assertEqual(len(reg.check_bhive_config(BHIVE_CONFIG, classes)), 64)
        with tempfile.TemporaryDirectory() as d:
            bad = yaml.safe_load(BHIVE_CONFIG.read_text())
            bad['truths'] = bad['truths'][::-1]
            Path(d, 'bad.yml').write_text(yaml.safe_dump(bad))
            with self.assertRaisesRegex(ValueError, 'truths'):
                reg.check_bhive_config(Path(d, 'bad.yml'), classes)
            bad = yaml.safe_load(BHIVE_CONFIG.read_text())
            bad['global_features'] = bad['global_features'][1:] + bad['global_features'][:1]
            Path(d, 'bad2.yml').write_text(yaml.safe_dump(bad))
            with self.assertRaisesRegex(ValueError, 'global_features'):
                reg.check_bhive_config(Path(d, 'bad2.yml'), classes)
        reg.check_onnx_io(ROOT / 'tests/data/mode_models/fold_0.onnx', 26, 6)
        with self.assertRaises(ValueError):
            reg.check_onnx_io(ROOT / 'tests/data/mode_models/fold_0.onnx', 26, 5)


class TestBhiveHandoff(unittest.TestCase):
    def test_commands_hold_out_fold_and_use_unique_versions(self):
        split = import_script('split_folds3', 'scripts/mva/split_folds.py')
        gen = import_script('bhive_fold_commands', 'scripts/mva/bhive_fold_commands.py')
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'in.parquet'
            train_shard(p, np.arange(50), np.zeros(50))
            split.split_files([p], Path(d) / 'split', 5, {(YEAR, 'S'): 1.})
            script, runs = gen.build(Path(d) / 'split', 'HPlusCHToWW_2dcats', 'SimpleMLP_MultiClass',
                                     30, 1024, 1e-3, 4, 300, 0.85, 'hwwtrainrun')
            versions = [v for r in runs.values() for v in (r['dataset_version'], r['test_dataset_version'])]
            self.assertEqual(len(versions), len(set(versions)))
            self.assertEqual(script.count('law run TrainingTask'), 5)
            self.assertEqual(script.count('--train-val-split 0.85'), 5)
            for i, r in runs.items():
                self.assertIn(f"--test-dataset-version {r['test_dataset_version']}", script)
                self.assertTrue(r['test_list'].endswith(f'model_{i}/test.txt'))
            self.assertNotIn('valid.txt', script)
            with open(Path(d) / 'split/model_2/train.txt', 'a') as f:
                f.write('extra\n')
            with self.assertRaisesRegex(ValueError, 'changed after'):
                gen.build(Path(d) / 'split', 'C', 'M', 1, 1, 1e-3, 1, 1, 0.85, 'x')


class TestCondorPreparation(unittest.TestCase):
    """F4/F10/F13 through the real submit_condor.py and condor/submit.sh."""

    def test_prepared_job_saves_into_created_mode_directory(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            work, outroot, stub = d / 'cwd', d / 'outroot', d / 'bin'
            (work / 'condor').mkdir(parents=True); stub.mkdir()
            os.symlink(ROOT / 'analysis', work / 'analysis')
            shutil.copy(ROOT / 'condor/submit.sub', work / 'condor/submit.sub')
            proxy = d / 'x509up_test'; proxy.write_text('fake')
            (stub / 'voms-proxy-info').write_text(f'#!/bin/bash\n[ "$1" = -path ] && echo {proxy}\nexit 0\n')
            (stub / 'voms-proxy-info').chmod(0o755)
            env = dict(os.environ, PATH=f'{stub}:{os.environ["PATH"]}', USER='zzfakeuser',
                       PYTHONDONTWRITEBYTECODE='1')
            fs = json.loads((ROOT / f'analysis/filesets/fileset_{YEAR}_nanov12_lxplus.json').read_text())
            dataset = next(k for k in fs if k.startswith('TTto2L2Nu'))
            subprocess.run([sys.executable, str(ROOT / 'submit_condor.py'), '-w', WF, '-y', YEAR,
                            '-d', dataset, '--mode', 'train', '--output_root', str(outroot)],
                           cwd=work, env=env, check=True, capture_output=True)
            job = work / 'condor' / f'{WF}__train' / YEAR / dataset
            args = json.loads((job / 'arguments.json').read_text())
            expect = outroot / 'outputs' / f'{WF}__train' / YEAR / dataset
            self.assertEqual(Path(args['output_path']), expect)
            self.assertTrue(expect.is_dir())                       # F4: created at preparation
            self.assertEqual(args['output_location'], str(outroot) + '/outputs/')
            # Execute condor/submit.sh in a scratch job dir with a stub submit.py that
            # records its argv and saves exactly as submit.py does.
            base, jobwd = d / 'base', d / 'jobwd'
            base.mkdir(); jobwd.mkdir()
            for f in ('arguments.json', 'partitions.json'):
                shutil.copy(job / f, jobwd / f)
            (base / 'submit.py').write_text(
                'import sys, json\n'
                'a = sys.argv[1:]\n'
                'json.dump(a, open(sys.argv[0] + ".argv.json", "w"))\n'
                'from coffea.util import save\n'
                'save({"ok": 1}, a[a.index("--output_path") + 1] + "/" + a[a.index("--dataset") + 1] + ".coffea")\n')
            r = subprocess.run(['bash', str(ROOT / 'condor/submit.sh'), '1', str(base), str(proxy)],
                               cwd=jobwd, env=env, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            argv = json.loads((base / 'submit.py.argv.json').read_text())
            self.assertEqual(argv[argv.index('--mode') + 1], 'train')
            self.assertEqual(argv[argv.index('--output_location') + 1], str(outroot) + '/outputs/')
            self.assertTrue(list(expect.glob('*.coffea')))
            # F13: the mode-options helper fails (non-string manifest). The old process
            # substitution hid this and ran legacy mode; now the job stops first.
            (base / 'submit.py.argv.json').unlink()
            (jobwd / 'arguments.json').write_text(json.dumps({**args, 'model_manifest': 123}))
            r = subprocess.run(['bash', str(ROOT / 'condor/submit.sh'), '1', str(base), str(proxy)],
                               cwd=jobwd, env=env, capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertFalse((base / 'submit.py.argv.json').exists())

    def test_output_root_rules(self):
        from analysis.utils.output_dir_maker import parquet_output_location, output_workflow_name
        ns = types.SimpleNamespace
        self.assertIsNone(parquet_output_location(ns(output_root=None)))
        self.assertEqual(parquet_output_location(ns(output_root='/eos/user/c/cgupta/X', output_root_url=None)),
                         'root://eosuser.cern.ch//eos/user/c/cgupta/X/outputs/')
        with self.assertRaises(ValueError):
            parquet_output_location(ns(output_root='/eos/cms/store/group/X', output_root_url=None))
        self.assertEqual(output_workflow_name('wf', 'legacy'), 'wf')
        self.assertEqual(output_workflow_name('wf', 'run'), 'wf__run')
        from analysis.utils.path_handler import Paths
        with self.assertRaises(ValueError):
            Paths(eos=False, root='relative/dir')


class TestJobsStatus(unittest.TestCase):
    def test_prepared_job_mode_and_output_must_match(self):
        sys.argv = ['x']
        js = import_script('jobs_status', 'jobs_status.py')
        with tempfile.TemporaryDirectory() as d:
            job_dir, out = Path(d) / 'jobs', Path(d) / 'out'
            (job_dir / 'DS').mkdir(parents=True)
            (job_dir / 'DS/arguments.json').write_text(json.dumps({'mode': 'run', 'output_path': str(out / 'DS')}))
            js.check_prepared_jobs(job_dir, out, 'run')
            with self.assertRaises(SystemExit):
                js.check_prepared_jobs(job_dir, out, 'train')
            with self.assertRaises(SystemExit):
                js.check_prepared_jobs(job_dir, Path(d) / 'elsewhere', 'run')


class TestEosParquetFallback(unittest.TestCase):
    def test_truncated_eos_file_retries_from_server(self):
        import analysis.utils.eos_io as eio
        with tempfile.TemporaryDirectory() as d:
            good = Path(d) / 'good.parquet'
            pq.write_table(pa.table({'x': [1, 2]}), good)
            bad = Path(d) / 'bad.parquet'; bad.write_bytes(b'')
            with self.assertRaises(pa.ArrowInvalid):
                eio.read_parquet(str(bad))                      # non-EOS: no fallback
            saved = eio._is_eos, eio.read_bytes
            try:
                eio._is_eos = lambda p: True
                eio.read_bytes = lambda p, prefer_server=False: good.read_bytes()
                self.assertEqual(eio.read_parquet(str(bad))['x'].to_pylist(), [1, 2])
            finally:
                eio._is_eos, eio.read_bytes = saved


if __name__ == '__main__':
    unittest.main()
