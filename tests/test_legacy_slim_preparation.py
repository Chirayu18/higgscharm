"""Bounded synthetic regressions for legacy-slim preparation; no production IO."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import awkward as ak
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/combine'))


def builder():
    import build_cards_from_templates
    return build_cards_from_templates


def objects(pt, phi):
    return ak.Array([[{'pt': p, 'phi': f} for p, f in zip(ps, fs)] for ps, fs in zip(pt, phi)])


class TestLeptonMET(unittest.TestCase):
    def test_vector_conservation_and_empty_event(self):
        from analysis.corrections.met_propagation import propagate_object_met
        met = ak.Array([{'pt': 50., 'phi': 0.}, {'pt': 20., 'phi': np.pi/2}])
        old = objects([[10., 20.], []], [[0., np.pi/2], []])
        new = objects([[15., 17.], []], [[0., np.pi/2], []])
        result = propagate_object_met(met, old, new)
        # Independent vector arithmetic: (50,0) - ((15,17)-(10,20)) = (45,3).
        np.testing.assert_allclose(result.pt, [np.hypot(45, 3), 20.])
        np.testing.assert_allclose(result.phi, [np.arctan2(3, 45), np.pi/2])
        np.testing.assert_array_equal(met.pt, [50., 20.])
        with self.assertRaisesRegex(ValueError, 'aligned'):
            propagate_object_met(met, old, objects([[1.], []], [[0.], []]))

    def test_legacy_update_helper_uses_momentum_conserving_sign(self):
        from analysis.corrections.met import update_met
        events=ak.Array({'Muon':[[{'pt_raw':10.,'pt':15.,'phi':0.},
                                  {'pt_raw':20.,'pt':17.,'phi':np.pi/2}]],
                         'PuppiMET':[{'pt':50.,'phi':0.}]})
        update_met(events,'Muon')
        np.testing.assert_allclose(events.PuppiMET.pt,[np.hypot(45.,3.)])
        np.testing.assert_allclose(events.PuppiMET.phi,[np.arctan2(3.,45.)])

    def test_manager_nominal_jes_lepton_shifts_and_opt_in(self):
        import analysis.corrections.correction_manager as cm
        events = ak.Array({'Muon': [[{'pt': 10., 'phi': 0.}]],
                           'Electron': [[{'pt': 20., 'phi': np.pi/2}]],
                           'Jet': [[{'pt': 40.}]], 'PuppiMET': [{'pt': 50., 'phi': 0.}]})
        def jet(**kw):
            base = {'Jet': events.Jet, 'MET': events.PuppiMET}
            up = dict(base, MET=ak.with_field(events.PuppiMET, [60.], 'pt'))
            return [(base, None), (up, 'JESUp')]
        def mu(**kw):
            ss = kw['shifts']
            for cols, _ in ss:
                cols['Muon'] = ak.with_field(events.Muon, [[15.]], 'pt')
            ss.append((dict(ss[0][0], Muon=ak.with_field(events.Muon, [[18.]], 'pt')), 'muUp'))
            return ss
        def ele(**kw):
            ss = kw['shifts']
            for cols, _ in ss:
                cols['Electron'] = ak.with_field(events.Electron, [[22.]], 'pt')
            ss.append((dict(ss[0][0], Electron=ak.with_field(events.Electron, [[24.]], 'pt')), 'elUp'))
            return ss
        cfg = {'object': {'jet': {}, 'muon': {}, 'electron': {}, 'propagate_leptons_to_met': True}}
        with patch.object(cm, 'apply_jerc_corrections', jet), patch.object(cm, 'apply_muon_ss_corrections', mu), patch.object(cm, 'apply_electron_ss_corrections', ele):
            _, ss = cm.object_corrector_manager(events, '2022postEE', cfg)
            for (cols, name), x, y in zip(ss, [45.,55.,42.,45.], [-2.,-2.,-2.,-4.]):
                np.testing.assert_allclose(cols['MET'].pt * np.cos(cols['MET'].phi), [x])
                np.testing.assert_allclose(cols['MET'].pt * np.sin(cols['MET'].phi), [y], atol=1e-12)
            cfg['object'].pop('propagate_leptons_to_met')
            _, original = cm.object_corrector_manager(events, '2022postEE', cfg)
            np.testing.assert_array_equal(original[0][0]['MET'].pt, [50.])
            cfg['object'] = list(cfg['object'])  # actual existing workflow schema
            _, legacy_list = cm.object_corrector_manager(events, '2022postEE', cfg)
            np.testing.assert_array_equal(legacy_list[0][0]['MET'].pt, [50.])
            np.testing.assert_array_equal(events.Muon.pt, [[10.]])
            # Changing the returned variation cannot modify nominal or jet variation.
            ss[-1][0]['MET']['pt'] = [999.]
            self.assertLess(float(ss[0][0]['MET'].pt[0]), 100.)


class TestCombinedMET(unittest.TestCase):
    def test_xy_then_both_leptons_on_every_shift(self):
        import analysis.corrections.correction_manager as cm
        from analysis.corrections.met_xy import apply_run3_met_xy
        events=ak.Array({'Muon':[[{'pt':10.,'phi':0.}]],
                         'Electron':[[{'pt':20.,'phi':np.pi/2}]],
                         'Jet':[[{'pt':40.}]], 'PuppiMET':[{'pt':80.,'phi':.3}],
                         'PV':[{'npvsGood':25}], 'genWeight':[1.]})
        def jet(**kw):
            return [({'MET':events.PuppiMET,'Jet':events.Jet},None),
                    ({'MET':ak.with_field(events.PuppiMET,[90.],'pt'),'Jet':events.Jet},'JESUp')]
        def mu(**kw):
            ss=kw['shifts']
            for cols,_ in ss: cols['Muon']=ak.with_field(events.Muon,[[15.]],'pt')
            ss.append((dict(ss[0][0],Muon=ak.with_field(events.Muon,[[18.]],'pt')),'muUp'))
            return ss
        def ele(**kw):
            ss=kw['shifts']
            for cols,_ in ss: cols['Electron']=ak.with_field(events.Electron,[[22.]],'pt')
            ss.append((dict(ss[0][0],Electron=ak.with_field(events.Electron,[[24.]],'pt')),'elUp'))
            return ss
        cfg={'object':['jet','muon','electron','met_phi','propagate_leptons_to_met'],
             'object_shifts':True}
        reference={name:cols['MET'] for cols,name in apply_run3_met_xy(events,'2022postEE',jet(),cfg)}
        with patch.object(cm,'apply_jerc_corrections',jet), patch.object(cm,'apply_muon_ss_corrections',mu), patch.object(cm,'apply_electron_ss_corrections',ele):
            _,result=cm.object_corrector_manager(events,'2022postEE',cfg)
        self.assertEqual(len(result),6)
        for cols,name in result:
            baseline=reference[name if name in reference else None]
            dx=8. if name=='muUp' else 5.
            dy=4. if name=='elUp' else 2.
            np.testing.assert_allclose(cols['MET'].pt*np.cos(cols['MET'].phi),baseline.pt*np.cos(baseline.phi)-dx,atol=1e-10)
            np.testing.assert_allclose(cols['MET'].pt*np.sin(cols['MET'].phi),baseline.pt*np.sin(baseline.phi)-dy,atol=1e-10)


class TestAlternativeTemplates(unittest.TestCase):
    def setUp(self):
        from analysis.workflows.config import WorkflowConfigBuilder
        self.mod = builder()
        self.cfg = self.mod.configure_era(WorkflowConfigBuilder('hww_allfix_4fs').build_workflow_config(), '2022postEE')
        # Reuse synthetic fixture creation, not an already-run test.
        from tests.test_trainrun_blockers import TestCardBuilder
        TestCardBuilder.cfg, TestCardBuilder.datasets = self.cfg, self.mod.get_dataset_config('2022postEE')
        fixture = TestCardBuilder()
        self.nominal, self.pick, _ = fixture.output()
        self.record = fixture.record
        self.cfg.combine['alternative_samples'] = {'tt_hdamp': {
            'process': 'tt', 'nominal_samples': [self.pick['tt']],
            'samples': {'Up': ['TTto2L2Nu_Hdamp418'], 'Down': ['TTto2L2Nu_Hdamp158']}}}
        cat = next(iter(self.cfg.event_selection['categories']))
        self.alt = {'samples': {s: {'sumw': 100., 'chunks': {s+'-c0'}, 'chunks_read': 1,
                            'templates': {cat: {'nominal': self.record(value, {s+'-c0'})}}}
                        for s, value in [('TTto2L2Nu_Hdamp418', 2.), ('TTto2L2Nu_Hdamp158', .5)]}}

    def test_replaces_dilepton_only_preserves_rest_and_variance(self):
        # Add an independent nominal hadronic tt component with a distinct pattern.
        cat = next(iter(self.cfg.event_selection['categories']))
        other = deepcopy(self.nominal['samples'][self.pick['tt']])
        for rec in other['templates'][cat].values():
            for ch in rec['counts']:
                rec['counts'][ch][:] = 3.
                rec['sumw2'][ch][:] = 7.
        self.nominal['samples']['TTto4Q'] = other
        plain_cfg = deepcopy(self.cfg)
        plain_cfg.combine.pop('alternative_samples')
        baseline, *_ = self.mod.make_process_hists(self.nominal, plain_cfg, '2022postEE')
        policy = {}
        h, variations, obj, edges = self.mod.make_process_hists(self.nominal, self.cfg, '2022postEE', policy=policy, alternative_output=self.alt)
        scale = self.mod.load_lumi('2022postEE') * 98.578 / 100.
        rest_scale = self.mod.load_lumi('2022postEE') * 418.574 / 100.
        for ch in h:
            np.testing.assert_array_equal(h[ch]['tt']['nominal'][0], baseline[ch]['tt']['nominal'][0])
            np.testing.assert_allclose(h[ch]['tt']['tt_hdampUp'][0], 3*rest_scale + 2*scale)
            np.testing.assert_allclose(h[ch]['tt']['tt_hdampDown'][0], 3*rest_scale + .5*scale)
            np.testing.assert_allclose(h[ch]['tt']['tt_hdampUp'][1], 7*rest_scale**2 + 2*scale**2)
            np.testing.assert_array_equal(h[ch]['st']['tt_hdampUp'][0], h[ch]['st']['nominal'][0])
        self.assertIn('tt_hdampUp', variations)
        self.assertNotIn('tt_hdamp', obj)
        self.mod.finalize_templates(h, self.cfg.combine, variations, obj, policy)
        from analysis.sinks.hww import card_combine, final_channels
        import uproot
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/'templates.root'; card = Path(d)/'card.txt'
            self.mod.write_root(root, h, final_channels(self.cfg.combine), self.cfg.combine['classes'], [(v,'') for v in variations], edges)
            self.mod.write_datacard(card, root.name, card_combine(self.cfg.combine), h, edges, obj)
            row = next(l for l in card.read_text().splitlines() if l.startswith('tt_hdamp ')).split()
            self.assertEqual(row[1], 'shape')
            self.assertEqual(row[2:], ['-','-','1','-','-','-'] * len(h))
            with uproot.open(root) as f:
                key = next(iter(h)) + '_tt_tt_hdampUp'
                np.testing.assert_allclose(f[key].values(), h[next(iter(h))]['tt']['tt_hdampUp'][0])
                np.testing.assert_allclose(f[key].variances(), h[next(iter(h))]['tt']['tt_hdampUp'][1])
        self.assertEqual(policy['alternative_samples']['tt_hdamp']['nominal_samples'], [self.pick['tt']])

    def test_missing_unknown_and_wrong_direction_fail(self):
        with self.assertRaisesRegex(ValueError, 'require --alternative_inputs'):
            self.mod.make_process_hists(self.nominal, self.cfg, '2022postEE')
        missing = deepcopy(self.alt); missing['samples'].pop('TTto2L2Nu_Hdamp418')
        with self.assertRaisesRegex(ValueError, 'sample set'):
            self.mod.make_process_hists(self.nominal, self.cfg, '2022postEE', alternative_output=missing)
        cfg = deepcopy(self.cfg)
        cfg.combine['alternative_samples']['tt_hdamp']['samples']['Up'] = ['TTto2L2Nu_Hdamp158']
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            self.mod.make_process_hists(self.nominal, cfg, '2022postEE', alternative_output=self.alt)

    def test_missing_alternative_channel_is_not_a_real_zero(self):
        alt=deepcopy(self.alt)
        sample=next(iter(alt['samples']))
        cat=next(iter(self.cfg.event_selection['categories']))
        record=alt['samples'][sample]['templates'][cat]['nominal']
        record['counts'].pop(next(iter(record['counts'])))
        with self.assertRaisesRegex(ValueError,'inconsistent alternative channel'):
            self.mod.make_process_hists(self.nominal,self.cfg,'2022postEE',alternative_output=alt)

    def test_provenance_and_file_coverage_fail_closed(self):
        import json
        nominal=deepcopy(self.nominal)
        nominal['provenance']={'mode':{'run'}, 'model_manifest_hash':{'good'}, 'year':{'2022postEE'}}
        alternative=deepcopy(self.alt)
        alternative['provenance']=deepcopy(nominal['provenance'])
        alternative['execution_checks']={'files':{s:{'/store/'+s+'.root'} for s in alternative['samples']}}
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'files.json'
            path.write_text(json.dumps({s:list(v) for s,v in alternative['execution_checks']['files'].items()}))
            self.mod.validate_alternative_inputs(nominal,alternative,self.cfg,'2022postEE',str(path))
            alternative['provenance']['model_manifest_hash']={'wrong-model'}
            with self.assertRaisesRegex(ValueError,'provenance differ'):
                self.mod.validate_alternative_inputs(nominal,alternative,self.cfg,'2022postEE',str(path))
            alternative['provenance']=deepcopy(nominal['provenance'])
            sample=next(iter(alternative['samples']))
            alternative['execution_checks']['files'][sample]=set()
            with self.assertRaisesRegex(ValueError,'unprocessed'):
                self.mod.validate_alternative_inputs(nominal,alternative,self.cfg,'2022postEE',str(path))

    def test_nominal_train_card_configuration_removes_alternatives(self):
        import build_cards_from_train
        for stat in (True, False):
            cfg = build_cards_from_train.card_config(self.cfg, True, stat)
            self.assertNotIn('alternative_samples', cfg.combine)
            self.assertIn('alternative_samples', self.cfg.combine)


class TestShapeOnly(unittest.TestCase):
    def test_retains_channel_migration_scales_variance_and_leaves_nominal(self):
        from template_prescriptions import normalize_shape_only
        # Nominal total 100; up total 200. After rate removal, SR is 60
        # (nominal 40) and CR is 40 (nominal 60): migration is retained.
        h = {ch: {'p': {'nominal': (np.array([n]), np.array([4.])),
                          'scaleUp': (np.array([u]), np.array([16.])),
                          'scaleDown': (np.array([v]), np.array([9.]))}}
             for ch,n,u,v in [('SR',40.,120.,10.), ('CR',60.,80.,40.)]}
        original = deepcopy(h)
        policy = {}
        normalize_shape_only(h, {'classes':['p'], 'shape_only_systematics':{'scale':['p']}},
                             ['nominal','scaleUp','scaleDown'], policy)
        self.assertEqual(float(h['SR']['p']['scaleUp'][0][0]), 60.)
        self.assertEqual(float(h['CR']['p']['scaleUp'][0][0]), 40.)
        self.assertEqual(float(h['SR']['p']['scaleUp'][1][0]), 4.)
        self.assertEqual(float(h['SR']['p']['scaleDown'][1][0]), 36.)
        for d in ('Up','Down'):
            self.assertEqual(sum(float(h[ch]['p']['scale'+d][0].sum()) for ch in h), 100.)
        for ch in h:
            np.testing.assert_array_equal(h[ch]['p']['nominal'][0], original[ch]['p']['nominal'][0])
        self.assertEqual(policy['shape_only_systematics'][0]['scope'], 'all_final_channels_per_era')
        with self.assertRaisesRegex(ValueError, 'missing paired'):
            normalize_shape_only(h, {'classes':['p'], 'shape_only_systematics':{'missing':['p']}}, [], {})
        h['SR']['p']['scaleUp'][0][:] = -40.
        h['CR']['p']['scaleUp'][0][:] = 40.
        with self.assertRaisesRegex(ValueError, 'positive finite'):
            normalize_shape_only(h, {'classes':['p'], 'shape_only_systematics':{'scale':['p']}}, ['scaleUp','scaleDown'], {})

    def test_normalization_applies_after_existing_floor(self):
        h={'SR':{'p':{'nominal':(np.array([-2.,12.]),np.array([2.,4.])),
                      'scaleUp':(np.array([-1.,21.]),np.array([3.,5.])),
                      'scaleDown':(np.array([3.,2.]),np.array([1.,1.]))}}}
        policy={}
        builder().finalize_templates(h,{'classes':['p'],'shape_only_systematics':{'scale':['p']}},
                                     ['nominal','scaleUp','scaleDown'],[],policy)
        nominal=h['SR']['p']['nominal'][0].sum()
        self.assertAlmostEqual(nominal,12.000001)
        for direction in ('Up','Down'):
            self.assertAlmostEqual(h['SR']['p']['scale'+direction][0].sum(),nominal)
        self.assertEqual(len(policy['shape_only_systematics']),2)

    def test_card_policy_does_not_mutate_selection_or_original(self):
        from template_prescriptions import load_card_policy
        cfg = SimpleNamespace(combine={'classes':['p']}, object_selection={'cut':'unchanged'})
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'policy.yaml'
            path.write_text('shape_only_systematics: {scale: [p]}\n')
            updated=load_card_policy(cfg, path)
            self.assertEqual(updated.object_selection, cfg.object_selection)
            self.assertNotIn('shape_only_systematics', cfg.combine)
            path.write_text('corrections: guessed\n')
            with self.assertRaisesRegex(ValueError, 'supports only'):
                load_card_policy(cfg, path)


class TestRateConstraint(unittest.TestCase):
    def test_replaces_free_rate_with_sourced_constraint_without_defaults(self):
        from template_prescriptions import load_card_policy
        cfg=SimpleNamespace(combine={'classes':['tt','st'], 'rate_params':['tt'], 'lnN':{},
                                     'no_theory':['tt'], 'shape_systematics':[]})
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'policy.yaml'
            # Test-only 10% example: no production numerical constraint is chosen.
            path.write_text('rate_constraints: {tt: {name: test_xsec_tt, value: 1.1, source: synthetic-test}}\n')
            new=load_card_policy(cfg,path)
            self.assertEqual(new.combine['rate_params'],[])
            self.assertEqual(new.combine['lnN'],{'test_xsec_tt':{'tt':1.1}})
            self.assertEqual(cfg.combine['rate_params'],['tt'])
            self.assertEqual(new.combine['no_theory'],['tt'])
            self.assertEqual(new.combine['rate_constraints_provenance']['tt']['source'],'synthetic-test')
            path.write_text('rate_constraints: {tt: {name: test_xsec_tt, value: 1.1, source: ""}}\n')
            with self.assertRaisesRegex(ValueError,'nonempty source'):
                load_card_policy(cfg,path)
            path.write_text('rate_constraints: {tt: {name: test_xsec_tt, value: -1, source: test}}\n')
            with self.assertRaisesRegex(ValueError,'invalid or trivial'):
                load_card_policy(cfg,path)


class TestMETXY(unittest.TestCase):
    def test_all_eras_data_mc_payloads_and_shift_independence(self):
        from analysis.corrections.met_xy import apply_run3_met_xy, EPOCHS
        from analysis.corrections.correctionlib_files import correction_files
        import correctionlib
        for year, epoch in EPOCHS.items():
            for mc in (False, True):
                data = {'PV': [{'npvsGood': 25}, {'npvsGood': 40}]}
                if mc:
                    data['genWeight'] = [1.,1.]
                events=ak.Array(data)
                raw=ak.Array([{'pt':80.,'phi':0.3}, {'pt':120.,'phi':-1.}])
                shifted=ak.with_field(raw, [90.,130.], 'pt')
                shifts=[({'MET':raw,'Jet':None},None), ({'MET':shifted,'Jet':None},'JESUp')]
                result=apply_run3_met_xy(events, year, shifts, {'object_shifts':True})
                self.assertEqual(len(result), 4 if mc else 2)
                self.assertEqual(result[1][1], 'JESUp')
                # The XY correction is a vector offset: preserve the existing
                # JES MET delta rather than replacing it with corrected nominal.
                for fn in (np.cos,np.sin):
                    before=shifted.pt*fn(shifted.phi)-raw.pt*fn(raw.phi)
                    after=result[1][0]['MET'].pt*fn(result[1][0]['MET'].phi)-result[0][0]['MET'].pt*fn(result[0][0]['MET'].phi)
                    np.testing.assert_allclose(after,before,atol=1e-10)
                np.testing.assert_array_equal(raw.pt,[80.,120.])
                self.assertFalse(np.array_equal(result[0][0]['MET'].pt, raw.pt))
                if mc:
                    self.assertEqual(result[-1][1], 'met_xy_'+year[:4]+'Down')
                    self.assertFalse(np.array_equal(result[-1][0]['MET'].pt, result[0][0]['MET'].pt))
                    result[-1][0]['MET']['pt']=[999.,999.]
                    self.assertTrue(np.all(ak.to_numpy(result[0][0]['MET'].pt)<200.))
                    self.assertEqual(len(apply_run3_met_xy(events,year,shifts,{'object_shifts':False})),2)
        with self.assertRaisesRegex(ValueError,'supported'):
            apply_run3_met_xy(ak.Array([]),'2024',[],{'object_shifts':True})


if __name__ == '__main__':
    unittest.main()
