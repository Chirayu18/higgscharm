import importlib.util
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from coffea.util import save
from analysis.features.hww import FEATURES, event_folds
from scripts.combine.build_cards_from_train import accumulate_train, card_config, write_cards
from scripts.combine.build_cards_from_templates import normalize_record, write_datacard


class Scorer:
    def score(self, X, events):
        # Distinct probability/channel per event, independent of the code under test.
        a = events % 2 == 0
        return np.column_stack([np.where(a, .8, .2), np.where(a, .2, .8)])


class Negrw:
    def _score_negrw(self, cfg, variables):
        assert 'weight_negrw' not in variables
        return np.asarray(variables['gen_test']), None


class TrainCardsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg = types.SimpleNamespace(event_selection={'categories': {'base': []}}, negrw=None,
            combine={'classes': ['hplusc', 'tt'], 'channels': {'SR': 'hplusc', 'CR': 'tt'},
                     'binning': {'edges': [0., .7, 1.]}, 'signal': 'hplusc',
                     'shape_systematics': ['shape1'], 'lnN': {'rate': {'tt': 1.1}},
                     'rate_params': ['tt'], 'negrw_uncertainty': True, 'autoMCStats': 10})
        self.datasets = {'sample': {'era': 'mc', 'process': 'p', 'xsec': 2.}}
        marker = self.root/'sample_1.coffea'
        save({'provenance': {'mode': {'train'}, 'year': {'2022preEE'}, 'features_hash': {'f'},
                            'selection_hash': {'s'}, 'folds': {5}},
              'execution_checks': {'expected_entries': 100, 'processed_entries': 100,
                                   'files': {'sample': {'/store/a.root'}}},
              'samples': {'sample': {'sumw': 100., 'sumw2': 120., 'events_read': 100,
                         'chunks_read': 2, 'chunks': {'c0', 'c1'}, 'selected_events': 4}}}, str(marker))
        self.paths = []
        self.events = np.array([0, 1, 2, 2**63+7], dtype=np.uint64)
        self.weights = [2., -1., 3., -2.]
        self.g = [.5, .25, .5, .75]
        for i in range(2):
            sl = slice(2*i, 2*i+2)
            d = {k: np.full(2, .3, dtype=np.float32) for k in FEATURES}
            d.update(event=self.events[sl], fold=event_folds(self.events[sl]),
                     weight_nominal=self.weights[sl], gen_test=self.g[sl],
                     weight_negrw=[99., 99.], sample=['sample']*2, process=['p']*2)
            table = pa.table(d).replace_schema_metadata({b'processing_mode': b'train',
                      b'year': b'2022preEE', b'features_hash': b'f', b'selection_hash': b's',
                      b'folds': b'5', b'sample': b'sample', b'source_file': b'/store/a.root'})
            path = self.root/f'shard{i}.parquet'
            pq.write_table(table, path); self.paths.append(str(path))
        self.manifest = {'coffea': [str(marker)], 'parquets': self.paths,
                         'expected_partitions': [{'marker': str(marker), 'era': '2022preEE',
                                                  'sample': 'sample', 'lfns': ['/store/a.root']}]}

    def build(self, **kw):
        cfg = card_config(self.cfg, kw.pop('merge', True), True)
        return accumulate_train(self.manifest, '2022preEE', cfg, Scorer(), self.datasets,
                                'f', 's', **kw)

    def test_signed_nominal_counts_variances_and_uint64_folds(self):
        out, report = self.build()
        rec = out['samples']['sample']['templates']['base']['nominal']
        norm = normalize_record(rec, 2.)
        # Even events carry +2,+3 into SR; odd events -1,-2 into CR.
        np.testing.assert_allclose(norm['SR'][0], [0., 10.])
        np.testing.assert_allclose(norm['SR'][1], [0., 52.])
        np.testing.assert_allclose(norm['CR'][0], [0., -6.])
        np.testing.assert_allclose(norm['CR'][1], [0., 20.])
        self.assertEqual((report['rows'], report['parquets'], report['markers']), (4, 2, 1))
        self.assertEqual(out['samples']['sample']['sumw'], 100.)

    def test_negrw_rescored_and_renormalized_after_all_shards(self):
        self.cfg.negrw = {'datasets': ['sample'], 'model': 'unused'}
        out, _ = self.build(negrw_processor=Negrw())
        rec = out['samples']['sample']['templates']['base']['nominal']
        norm = normalize_record(rec, 2.)
        factor = 4./4.25
        np.testing.assert_allclose(norm['SR'][0], [0., 2.5*factor])
        np.testing.assert_allclose(norm['CR'][0], [0., 1.75*factor])
        np.testing.assert_allclose(norm['SR'][1], [0., 3.25*factor**2])
        np.testing.assert_allclose(norm['CR'][1], [0., 2.3125*factor**2])
        self.assertAlmostEqual(sum(v[0].sum() for v in norm.values()), 4.)

    def test_missing_rows_and_duplicate_inputs_rejected(self):
        self.manifest['parquets'] = self.paths[:1]
        with self.assertRaisesRegex(ValueError, 'row mismatch'): self.build()
        self.manifest['parquets'] = self.paths + self.paths[:1]
        with self.assertRaisesRegex(ValueError, 'duplicate TRAIN parquet'): self.build()

    def test_wrong_fold_and_unprocessed_source_rejected(self):
        p = self.paths[0]; table = pq.read_table(p)
        idx = table.schema.get_field_index('fold')
        pq.write_table(table.set_column(idx, 'fold', pa.array([4, 4])), p)
        with self.assertRaisesRegex(ValueError, 'stored fold'): self.build()
        md = dict(table.schema.metadata); md[b'source_file'] = b'/store/unprocessed.root'
        pq.write_table(table.replace_schema_metadata(md), p)
        with self.assertRaisesRegex(ValueError, 'unprocessed source_file'): self.build()

    def test_split_variable_requires_explicit_merge(self):
        self.cfg.combine['channel_splits'] = {'SR': {'variable': 'cjet_multiplicity',
            'bins': {'SR1': [1, 2], 'SR2': [2, None]}}}
        with self.assertRaisesRegex(ValueError, 'explicitly use --merge_channel_splits'):
            self.build(merge=False)
        self.build(merge=True)

    def test_card_keeps_statistics_and_tt_rateparam_but_no_shapes_or_lnn(self):
        cfg = card_config(self.cfg, True, True)
        h = {ch: {cp: {'nominal': (np.array([1., 2.]), np.array([.5, 1.]))}
                  for cp in cfg.combine['classes']} for ch in cfg.combine['channels']}
        path = self.root/'datacard.txt'
        write_datacard(path, 'templates.root', cfg.combine, h, [0, .7, 1], [])
        text = path.read_text()
        self.assertNotIn(' lnN ', text)
        self.assertNotIn(' shape ', text)
        self.assertEqual(text.count(' autoMCStats 10'), 2)
        self.assertIn('rate_tt rateParam * tt 1.0 [0,5]', text)
        rate_cfg = card_config(self.cfg, True, False)
        write_datacard(path, 'templates.root', rate_cfg.combine, h, [0, .7, 1], [])
        self.assertIn(' lnN ', path.read_text())
        self.assertNotIn(' shape ', path.read_text())

    def test_shared_scored_accumulator_writes_root_and_both_card_modes(self):
        import copy
        import uproot
        out, _ = self.build()
        out['samples']['other'] = copy.deepcopy(out['samples']['sample'])
        self.cfg.combine['process_map'] = {'hplusc': ['p'], 'tt': ['q']}
        ds = dict(self.datasets, other={'era': 'mc', 'process': 'q', 'xsec': 2.})
        stat = card_config(self.cfg, True, True)
        rate = card_config(self.cfg, True, False)
        with patch('scripts.combine.build_cards_from_templates.get_dataset_config', return_value=ds), \
             patch('scripts.combine.build_cards_from_templates.load_lumi', return_value=100.):
            stat_card = write_cards(out, stat, '2022preEE', self.root/'stat', {'card_mode': 'stat_only'})
            rate_card = write_cards(out, rate, '2022preEE', self.root/'rates', {'card_mode': 'rate_systematics'})
        self.assertNotIn(' lnN ', stat_card.read_text())
        self.assertIn(' lnN ', rate_card.read_text())
        self.assertEqual(out['samples']['sample']['templates']['base']['nominal']['counts']['CR'][1], -3.)
        with uproot.open(stat_card.parent/'templates.root') as a, uproot.open(rate_card.parent/'templates.root') as b:
            for name in a.keys(cycle=False):
                np.testing.assert_array_equal(a[name].values(), b[name].values())
                np.testing.assert_array_equal(a[name].variances(), b[name].variances())
            # Background-only Asimov, independently checked against the background histogram.
            np.testing.assert_array_equal(a['SR_data_obs'].values(), a['SR_tt'].values())
            self.assertAlmostEqual(a['SR_hplusc'].values().sum(), 10.000001)


if __name__ == '__main__':
    unittest.main()
