import hashlib
import importlib.util
from pathlib import Path
import tempfile
import types
import unittest
import awkward as ak
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from coffea import processor
from analysis.features.hww import FEATURES, compute, event_folds, matrix
from analysis.sinks.hww import make_templates, dataset_info

ROOT = Path(__file__).resolve().parents[1]


def import_script(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT/path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


class TestFeaturesAndSplits(unittest.TestCase):
    def test_folds_preserve_uint64_and_chunk_independence(self):
        event = np.array([0, 1, 9, 2**63+7, 2**64-1], dtype=np.uint64)
        expected = np.array([int(x) % 5 for x in event])
        np.testing.assert_array_equal(event_folds(event), expected)
        np.testing.assert_array_equal(event_folds(event), np.r_[event_folds(event[:2]), event_folds(event[2:])])
        with self.assertRaises(ValueError): event_folds(np.array([1.5]))
        with self.assertRaises(ValueError): event_folds(np.array([-1]))

    def test_features_are_ordered_and_missing_is_fatal(self):
        raw = {k: ak.Array([[1.], []]) for k in FEATURES if 'ctag2d' not in k}
        raw['cjet_cand_cvsl_pnet'] = np.array([.9, .9])
        raw['cjet_cand_cvsb_pnet'] = np.array([.5, .5])
        cols = compute(raw)
        self.assertEqual(matrix(cols).shape, (2, 26))
        self.assertEqual(matrix(cols).dtype, np.float32)
        self.assertEqual(cols['dilepton_pt'][1], 0.)
        self.assertEqual(sum(cols[k][0] for k in FEATURES if 'ctag2d' in k), 1)
        del raw['met_pt']
        with self.assertRaises(KeyError): compute(raw)

    def test_split_filelists_are_disjoint_and_exclude_data(self):
        mod = import_script('split_folds', 'scripts/mva/split_folds.py')
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'input.parquet'
            arrays = {k: np.ones(30, dtype=np.float32) for k in FEATURES}
            arrays.update(event=np.arange(30, dtype=np.uint64), fold=event_folds(np.arange(30)),
                          label=np.r_[np.zeros(25, dtype=np.int16), np.full(5, -1, dtype=np.int16)],
                          weight_nominal=np.ones(30))
            table = pa.table(arrays).replace_schema_metadata({b'processing_mode':b'train', b'features_hash':b'h',
                        b'selection_hash':b's', b'folds':b'5', b'year':b'2022postEE', b'sample':b'Hc'})
            pq.write_table(table, p)
            mod.split_files([p], Path(d)/'split', 5, {('2022postEE', 'Hc'): 3.})
            for fold in range(5):
                sets = {}
                # Single validation owner: b-hive splits train.txt internally; no valid.txt.
                self.assertFalse((Path(d)/'split'/f'model_{fold}'/'valid.txt').exists())
                for role in ('train', 'test'):
                    paths = (Path(d)/'split'/f'model_{fold}'/f'{role}.txt').read_text().splitlines()
                    rows = [pq.read_table(f).to_pydict() for f in paths]
                    sets[role] = set(x for t in rows for x in t['event'])
                    self.assertTrue(all(w == 3 for t in rows for w in t['weight']))
                self.assertFalse(sets['train'] & sets['test'])
                self.assertEqual(set.union(*sets.values()), set(range(25)))
                self.assertTrue(all(x % 5 == fold for x in sets['test']))

    def test_dataset_partition_resolution(self):
        d = {'A': {'process':'p'}, 'A_2': {'process':'q'}}
        self.assertEqual(dataset_info('A_2_7',d)[0], 'A_2')
        with self.assertRaises(ValueError): dataset_info('AB_1',d)


class TestTemplateAccumulation(unittest.TestCase):
    def test_variance_and_negrw_normalize_after_chunk_merge(self):
        from scripts.combine.build_cards_from_templates import normalize_record
        classes, channels, edges = ['a','b'], {'A':'a','B':'b'}, [0.,.5,1.]
        scores=np.array([[.8,.2],[.7,.3],[.1,.9]])
        w=np.array([2.,-1.,3.]);g=np.array([.3,-.4,.8])
        full=make_templates(scores,classes,channels,edges,{'nominal':w,'up':2*w},g,np.ones(3)*.1)
        chunks=[make_templates(scores[sl],classes,channels,edges,{'nominal':w[sl],'up':2*w[sl]},g[sl],np.ones(len(w[sl]))*.1) for sl in (slice(0,1),slice(1,3))]
        merged=processor.accumulate(chunks)
        for variation in full:
            a,b=normalize_record(full[variation],2.),normalize_record(merged[variation],2.)
            for ch in channels:
                np.testing.assert_allclose(a[ch][0],b[ch][0]);np.testing.assert_allclose(a[ch][1],b[ch][1])
        fac=2*w.sum()/(np.abs(w)*g).sum()
        out=normalize_record(merged['nominal'],2.)
        np.testing.assert_allclose(out['A'][1].sum(),np.sum((np.abs(w[:2])*g[:2]*fac)**2))

    def test_nonfinite_weights_raise_and_last_edge_is_kept(self):
        with self.assertRaises(ValueError):
            make_templates(np.array([[1.,0.]]),['a','b'],{'A':'a','B':'b'},[0.,1.],{'nominal':np.array([np.nan])})
        out=make_templates(np.array([[1.,0.]]),['a','b'],{'A':'a','B':'b'},[0.,1.],{'nominal':np.array([2.])})
        self.assertEqual(out['nominal']['counts']['A'][0],2.)




class TestFrozenModels(unittest.TestCase):
    def _registry(self, directory, **overrides):
        import yaml, shutil
        from analysis.models.registry import FoldModels
        records=[]
        for fold in range(5):
            src=ROOT/'tests/data/mode_models'/f'fold_{fold}.onnx'
            dest=Path(directory)/src.name;shutil.copyfile(src,dest)
            records.append(dict(fold=fold,excluded_fold=fold,path=dest.name,sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
        meta=dict(schema_version=1,features=FEATURES,features_hash='features',selection_hash='selection',
                  classes=list('abcdef'),folds=5,fold_field='event',diagnostic_only=True,models=records)
        meta.update(overrides);path=Path(directory)/'meta.yaml';path.write_text(yaml.safe_dump(meta))
        return path

    def test_model_uses_exact_heldout_fold_and_survives_pickle(self):
        from analysis.models.registry import FoldModels
        import pickle
        with tempfile.TemporaryDirectory() as d:
            path=self._registry(d)
            m=FoldModels(path,'features','selection',list('abcdef'),allow_diagnostic=True)
            event=np.arange(17,dtype=np.uint64)
            s=m.score(np.zeros((17,26),np.float32),event)
            np.testing.assert_array_equal(np.argmax(s,axis=1),event%5)
            restored=pickle.loads(pickle.dumps(m))
            np.testing.assert_array_equal(s,restored.score(np.zeros((17,26),np.float32),event))

    def test_diagnostic_and_mismatched_signatures_are_rejected(self):
        from analysis.models.registry import FoldModels
        with tempfile.TemporaryDirectory() as d:
            path=self._registry(d)
            with self.assertRaises(ValueError):FoldModels(path,'features','selection',list('abcdef'))
            with self.assertRaises(ValueError):FoldModels(path,'bad','selection',list('abcdef'),allow_diagnostic=True)
            with self.assertRaises(ValueError):FoldModels(path,'features','bad',list('abcdef'),allow_diagnostic=True)
            path=self._registry(d,diagnostic_only=False)
            with self.assertRaises(ValueError):FoldModels(path,'features','selection',list('abcdef'))

    def test_corrupt_model_and_missing_fold_are_rejected(self):
        from analysis.models.registry import FoldModels
        import yaml
        with tempfile.TemporaryDirectory() as d:
            path=self._registry(d)
            (Path(d)/'fold_0.onnx').write_bytes(b'invalid')
            with self.assertRaises(ValueError):FoldModels(path,'features','selection',list('abcdef'),allow_diagnostic=True)
            path=self._registry(d);meta=yaml.safe_load(path.read_text());meta['models'].pop();path.write_text(yaml.safe_dump(meta))
            with self.assertRaises(ValueError):FoldModels(path,'features','selection',list('abcdef'),allow_diagnostic=True)

class TestObjectShiftTemplates(unittest.TestCase):
    def test_object_shift_has_nominal_weight_without_a_second_nominal_slice(self):
        out=make_templates(np.array([[.8,.2]]),['a','b'],{'A':'a','B':'b'},[0.,1.],
                           {'CMS_scale_jUp':np.array([3.])},nominal_weight=np.array([3.]))
        self.assertEqual(set(out),{'CMS_scale_jUp'})
        self.assertEqual(out['CMS_scale_jUp']['counts']['A'][0],3.)
        self.assertEqual(out['CMS_scale_jUp']['sumw2']['A'][0],9.)

    def test_era_names_match_existing_wrapper_without_mutating_config(self):
        from scripts.combine.build_cards_from_templates import configure_era
        cfg=types.SimpleNamespace(combine={'shape_systematics':['CMS_eff_m_id_2022','lhe_pdf'],
                                          'lnN':{'lumi_13p6TeV':{'signal':1.014}}})
        other=configure_era(cfg,'2023postBPix')
        self.assertEqual(other.combine['shape_systematics'],['CMS_eff_m_id_2023','lhe_pdf'])
        self.assertEqual(other.combine['lnN']['lumi_13p6TeV_2023']['signal'],1.013)
        self.assertEqual(cfg.combine['shape_systematics'][0],'CMS_eff_m_id_2022')


if __name__ == '__main__': unittest.main()
