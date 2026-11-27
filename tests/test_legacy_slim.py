"""Slim output must retain enough information to reproduce shifted templates."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import awkward as ak
import numpy as np
import pyarrow.parquet as pq
from coffea.analysis_tools import Weights
from analysis.features.hww import FEATURES, event_folds, matrix
from analysis.processors.base import BaseProcessor
from analysis.sinks.hww import make_templates,channel_layout
from analysis.workflows.config import WorkflowConfigBuilder
from analysis.processing_modes import validate

# The real legacy-slim workflow: one merged SR, no channel splits (Chirayu, 2026-11-27).
WORKFLOW='hww_legacy_slim_ab'
SLIM_WORKFLOWS=['hww_legacy_slim_ab','hww_legacy_slim_ab_hdamp','hww_legacy_slim_b','hww_legacy_slim_hdamp']

class SlimRoundtrip(unittest.TestCase):
 def sink(self,path,shift=None,data=False):
  p=BaseProcessor(WORKFLOW,'2022postEE','parquet',str(path),mode='legacy_slim')
  p.is_mc=not data
  n=6
  sample='DYto2L_2Jets_50' if not data else 'Muon_Run2022E'
  # Resolve the real data sample name instead of making an era assumption.
  if data:
   from analysis.filesets.utils import get_dataset_config
   sample=next(s for s,v in get_dataset_config('2022postEE').items() if v['process']=='Data')
  ev=ak.Array(dict(event=np.array([1,2,3,4,5,2**63+7],dtype=np.uint64),run=np.ones(n,dtype=np.int64),luminosityBlock=np.arange(n)))
  ev.behavior={'__events_factory__':SimpleNamespace(_partition_key='sample/uuid/0-6')}
  # NanoEvents metadata is exposed as an attribute by its record behavior.
  class Events:
   event=ev.event;run=ev.run;luminosityBlock=ev.luminosityBlock;behavior=ev.behavior
   metadata={'dataset':sample,'filename':'/store/test.root'}
   def __len__(self):return n
  values={k:np.linspace(1,2,n) for k in FEATURES if 'ctag2d' not in k}
  values.update(cjet_cand_cvsl_pnet=np.full(n,.9),cjet_cand_cvsb_pnet=np.full(n,.5),lhe_vpt=np.arange(n,dtype=float))
  if shift:values['met_pt']+=10
  weights=Weights(n);weights.add('test',np.array([1.,-1,2,3,4,5]) if not data else np.ones(n),weightUp=np.full(n,2.),weightDown=np.full(n,.5))
  out={};p._mode_sink(out,Events(),weights,values,'base',shift)
  f=next(path.rglob('*.parquet'));return p,pq.read_table(f),values,weights,out
 def test_nominal_and_shift_roundtrip(self):
  from analysis.features.hww import compute
  with tempfile.TemporaryDirectory() as t:
   for shift in [None,'CMS_scale_jUp']:
    p,tab,values,weights,out=self.sink(Path(t)/(shift or 'nominal'),shift)
    md=tab.schema.metadata
    self.assertEqual(md[b'processing_mode'],b'legacy_slim');self.assertEqual(md[b'shift'],(shift or 'nominal').encode())
    self.assertNotIn('cjet_multiplicity',tab.column_names);self.assertIn('lhe_vpt',tab.column_names)
    self.assertNotIn('weight_negrw',tab.column_names)
    np.testing.assert_array_equal(tab['fold'].to_numpy(),event_folds(tab['event'].to_numpy()))
    np.testing.assert_array_equal(tab['event'].to_numpy()[-1:],np.array([2**63+7],dtype=np.uint64))
    if shift:self.assertNotIn('weight_testUp',tab.column_names)
    else:self.assertIn('weight_testUp',tab.column_names)
    # Fixed scores isolate output fidelity from model inference; cover every
    # process class (one merged SR) and signed weights.
    scores=np.eye(6)*.7+.05;scores[1]=scores[0]
    comb=p.workflow_config.combine;layout=channel_layout(comb)
    self.assertEqual(sorted(layout),sorted(comb['channels']));self.assertTrue(all(v[1] is None for v in layout.values()))
    wm={shift or 'nominal':weights.weight()}
    if shift is None:wm.update({v:weights.weight(v) for v in weights.variations})
    reference=make_templates(scores,comb['classes'],comb['channels'],comb['binning']['edges'],wm,nominal_weight=weights.weight(),layout=layout)
    stored={k:tab['weight_nominal' if k==(shift or 'nominal') else 'weight_'+k].to_numpy() for k in wm}
    actual=make_templates(scores,comb['classes'],comb['channels'],comb['binning']['edges'],stored,nominal_weight=tab['weight_nominal'].to_numpy(),layout=layout)
    for variation,record in reference.items():
     for ch in layout:
      np.testing.assert_array_equal(actual[variation]['counts'][ch],record['counts'][ch]);np.testing.assert_array_equal(actual[variation]['sumw2'][ch],record['sumw2'][ch])
    np.testing.assert_array_equal(matrix({k:tab[k].to_numpy() for k in FEATURES}),matrix(compute(values)))
 def test_no_model_required_and_manifest_rejected(self):
  cfg=WorkflowConfigBuilder(WORKFLOW).build_workflow_config()
  self.assertIsNone(validate(cfg,'legacy_slim',5))
  with self.assertRaises(ValueError):validate(cfg,'legacy_slim',5,manifest='unused')
 def test_single_sr_no_channel_splits(self):
  from analysis.sinks.hww import split_variables
  for w in SLIM_WORKFLOWS:
   cfg=WorkflowConfigBuilder(w).build_workflow_config()
   self.assertFalse(cfg.combine.get('channel_splits'),w);self.assertEqual(split_variables(cfg.combine),[],w)
   self.assertEqual(sorted(channel_layout(cfg.combine)),sorted(cfg.combine['channels']),w);self.assertIn('SR_hplusc',channel_layout(cfg.combine),w)
  # A split re-added to a legacy-slim workflow is refused at validation.
  cfg=WorkflowConfigBuilder(WORKFLOW).build_workflow_config()
  cfg.combine['channel_splits']={'SR_hplusc':{'variable':'cjet_multiplicity','bins':{'SR_hplusc_1cj':[1,2],'SR_hplusc_2cj':[2,None]}}}
  with self.assertRaisesRegex(ValueError,'one merged SR'):validate(cfg,'legacy_slim',5)
 def test_data_weights_and_labels(self):
  with tempfile.TemporaryDirectory() as t:
   p,tab,_,_,_=self.sink(Path(t),data=True)
   np.testing.assert_array_equal(tab['weight_nominal'].to_numpy(),np.ones(6))
   np.testing.assert_array_equal(tab['label'].to_numpy(),np.full(6,-1))
 def test_negrw_not_loaded_by_slim_constructor(self):
  with patch('joblib.load',side_effect=AssertionError('model loaded')):
   with tempfile.TemporaryDirectory() as t:self.sink(Path(t))

if __name__=='__main__':unittest.main()
