"""No-tag, single-SR legacy-slim selection, including shifted jet boundaries."""
import copy
import unittest
from pathlib import Path
from types import SimpleNamespace

import awkward as ak
import numpy as np
import yaml
from coffea.nanoevents.methods import candidate

from analysis.features.hww import selection_signature, signature
from analysis.processing_modes import validate
from analysis.selections.object_selections import ObjectSelector
from analysis.working_points import working_points
from analysis.sinks.hww import channel_layout
from analysis.workflows.config import WorkflowConfigBuilder

WORKFLOWS = ('hww_legacy_slim_ab', 'hww_legacy_slim_ab_hdamp',
             'hww_legacy_slim_b', 'hww_legacy_slim_hdamp')
ERAS = ('2022preEE', '2022postEE', '2023preBPix', '2023postBPix')
ROOT = Path(__file__).resolve().parents[1]


def config(name):
    # Read this checkout's YAML, also when testing isolated configs with CERN code.
    builder = WorkflowConfigBuilder.__new__(WorkflowConfigBuilder)
    builder.config = yaml.safe_load((ROOT / 'analysis/workflows' / (name + '.yaml')).read_text())
    return builder.build_workflow_config()


def select(cfg, year, pts):
    # Events: untagged good jet; no jet; two untagged jets; failed ID;
    # eta boundary; lepton overlap; pT boundary; eta migration across a shift.
    jets = ak.Array([
        [dict(pt=pts[0], eta=0., phi=2., cvl=.01, jetId=6)],
        [],
        [dict(pt=25., eta=0., phi=2., cvl=.01, jetId=6),
         dict(pt=24., eta=0., phi=2., cvl=.02, jetId=6)],
        [dict(pt=25., eta=0., phi=2., cvl=.01, jetId=0)],
        [dict(pt=25., eta=2.4, phi=2., cvl=.01, jetId=6)],
        [dict(pt=25., eta=0., phi=0., cvl=.01, jetId=6)],
        [dict(pt=20., eta=0., phi=2., cvl=.01, jetId=6)],
        [dict(pt=25., eta=pts[1], phi=2., cvl=.01, jetId=6)],
    ])
    jets = ak.zip(dict(pt=jets.pt, eta=jets.eta, phi=jets.phi,
                      mass=ak.ones_like(jets.pt), charge=ak.zeros_like(jets.pt),
                      btagPNetCvL=jets.cvl, btagPNetCvB=ak.full_like(jets.pt, .01),
                      jetId=jets.jetId, muEF=ak.zeros_like(jets.pt),
                      chEmEF=ak.zeros_like(jets.pt), neHEF=ak.zeros_like(jets.pt),
                      neEmEF=ak.zeros_like(jets.pt)),
                  with_name='PtEtaPhiMCandidate', behavior=candidate.behavior)
    leptons = ak.zip(dict(pt=ak.Array([[30.]] * 8), eta=ak.Array([[0.]] * 8),
                         phi=ak.Array([[0.]] * 8), mass=ak.Array([[0.]] * 8),
                         charge=ak.Array([[1]] * 8)),
                     with_name='PtEtaPhiMCandidate', behavior=candidate.behavior)
    selector = ObjectSelector(cfg.object_selection, year)
    selector.objects = {'cjets': jets, 'leptons': leptons}
    events = SimpleNamespace(Jet=jets)
    mask = selector.get_selection_mask(events, 'cjets', cfg.object_selection['cjets']['cuts'])
    selector.objects['cjets'] = jets[mask]
    selector.select_candidate_cjet('candidate_cjet')
    objects = selector.objects
    expr = cfg.event_selection['selections']['atleast_one_goodjet']
    keep = eval(expr, {'ak': ak, 'objects': objects})
    return keep, objects, selector, events


class NoTagSlim(unittest.TestCase):
    def test_untagged_jets_and_object_shift_migrations(self):
        for name in WORKFLOWS:
            cfg = config(name)
            for year in ERAS:
                with self.subTest(workflow=name, year=year):
                    keep, objects, selector, events = select(cfg, year, (20.1, 2.39))
                    np.testing.assert_array_equal(ak.to_numpy(keep),
                                                 [True, False, True, False, False, False, False, True])
                    # 20--30 GeV good jets pass; no accidental pT>30 replacement.
                    self.assertAlmostEqual(ak.to_list(objects['candidate_cjet'].pt)[0][0], 20.1)
                    self.assertEqual(ak.to_list(objects['candidate_cjet'].btagPNetCvL)[2], [.02])
                    # These retained jets fail the actual medium tag of the old baseline.
                    old = working_points.jet_ctagging(events, 'medium', year)
                    self.assertFalse(bool(ak.any(old)))
                    shifted, _, _, _ = select(cfg, year, (19.9, 2.41))
                    np.testing.assert_array_equal(ak.to_numpy(shifted),
                                                 [False, False, True, False, False, False, False, False])

    def test_single_sr_and_model_compatibility(self):
        for name in WORKFLOWS:
            with self.subTest(workflow=name):
                cfg = config(name)
                self.assertEqual([c for c in channel_layout(cfg.combine) if c.startswith('SR_')], ['SR_hplusc'])
                self.assertFalse(cfg.combine.get('channel_splits'))
                self.assertEqual(len(channel_layout(cfg.combine)), 6)
                self.assertIsNone(validate(cfg, 'legacy_slim', 5))
                order = list(cfg.object_selection)
                for producer, consumer in [('muons', 'electrons'), ('leptons', 'cjets'),
                                           ('cjets', 'candidate_cjet'), ('met', 'mTll')]:
                    self.assertLess(order.index(producer), order.index(consumer))
                self.assertIn('atleast_one_goodjet', cfg.event_selection['categories']['base'])
                self.assertNotIn('atleast_one_cjet', cfg.event_selection['selections'])
                old = copy.deepcopy(cfg)
                old.object_selection['cjets']['cuts'].insert(3, "working_points.jet_ctagging(events, 'medium', year)")
                old.event_selection['selections']['atleast_one_cjet'] = old.event_selection['selections'].pop('atleast_one_goodjet')
                old.event_selection['selections']['atleast_two_cjets'] = "ak.num(objects['cjets']) > 1"
                old.event_selection['categories']['base'][-1] = 'atleast_one_cjet'
                self.assertEqual(signature(cfg), signature(old))
                self.assertNotEqual(selection_signature(cfg), selection_signature(old))


if __name__ == '__main__':
    unittest.main()
