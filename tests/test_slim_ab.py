import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import awkward as ak
import numpy as np
from analysis.processors.base import BaseProcessor
from analysis.stitching.hplusc_routeb import RouteBConfig,chunk_sums
from analysis.stitching.slim_ab import arm_mask

class InclusiveAB(unittest.TestCase):
 def config(self):return RouteBConfig.from_block({'4FS':['HplusCharm_4FS_HToWWTo2L2Nu'],'3FS':['HplusCharm_3FS_HToWW'],'defer_selection':True})
 def test_offline_arms_and_signed_weight_sums(self):
  cfg=self.config();n=np.array([0,1,2,0]);w=np.array([3.,-2.,5.,1.])
  for sample,scheme in cfg.datasets.items():
   sums,keep=chunk_sums(w,n,scheme)
   cols={'event':np.arange(4),'gen_routeb_cjet_multiplicity':n,'routeb_keep':keep}
   a=arm_mask(sample,cols,cfg,'A');b=arm_mask(sample,cols,cfg,'B')
   np.testing.assert_array_equal(a,np.full(4,scheme=='4FS'));np.testing.assert_array_equal(b,n==0 if scheme=='4FS' else n>=1)
   self.assertEqual(w[b].sum(),sums.kept.sumw);self.assertEqual(w.sum(),sums.full.sumw)
 def test_bad_or_missing_stitch_columns_raise(self):
  cfg=self.config();cols={'event':np.arange(2),'gen_routeb_cjet_multiplicity':np.array([0,1]),'routeb_keep':np.array([True,False])}
  self.assertTrue(arm_mask('background',{'event':np.arange(2)},cfg,'A').all())
  with self.assertRaises(ValueError):arm_mask('background',cols,cfg,'union')
  with self.assertRaises(KeyError):arm_mask('HplusCharm_4FS_HToWWTo2L2Nu',{'event':np.arange(2)},cfg,'B')
  with self.assertRaises(ValueError):arm_mask('HplusCharm_3FS_HToWW',cols,cfg,'B')
  with self.assertRaises(ValueError):RouteBConfig.from_block({'4FS':['s4'],'3FS':['s3'],'defer_selection':'true'})
 def test_deferred_processing_keeps_all_events_but_records_true_kept_sums(self):
  sample='HplusCharm_4FS_HToWWTo2L2Nu';cfg=self.config();n=np.array([0,1,2,0]);w=np.array([3.,-2.,5.,1.])
  ev=ak.Array({'event':np.arange(4),'genWeight':w},behavior={'__events_factory__':SimpleNamespace(_partition_key='chunk')})
  # Lightweight array wrapper keeps NanoEvents metadata/field-write semantics.
  class Events:
   metadata={'dataset':sample};behavior=ev.behavior
   def __init__(self,a):self.a=a
   def __len__(self):return len(self.a)
   def __getattr__(self,k):return getattr(self.a,k)
   def __setitem__(self,k,v):self.a[k]=v
   def __getitem__(self,k):return Events(self.a[k])
  events=Events(ev);seen=[]
  p=BaseProcessor('hww_legacy_slim_ab','2022postEE','parquet','/tmp/unused-ab',mode='legacy_slim')
  p.process_shift=lambda e,sh:seen.append((len(e),np.asarray(e.routeb_keep),np.asarray(e.gen_routeb_cjet_multiplicity))) or {}
  sums,keep=chunk_sums(w,n,'4FS')
  with patch('analysis.stitching.hplusc_routeb.event_sums',return_value=(sums,keep,n)),patch('analysis.processors.base.object_corrector_manager',side_effect=lambda **kw:(kw['events'],[({},None),({},'jUp')])),patch('analysis.processors.base.update',side_effect=lambda e,c:e):out=p.process(events)
  self.assertEqual([x[0] for x in seen],[4,4])
  for _,k,nn in seen:np.testing.assert_array_equal(k,keep);np.testing.assert_array_equal(nn,n)
  entry=out['samples'][sample];self.assertEqual(entry['sumw'],w.sum())
  rec=json.loads(entry['routeb_stitch']['chunk']);self.assertEqual(rec['sums']['kept']['sumw'],w[keep].sum())
  self.assertEqual(out['provenance']['signal_retention'],{'inclusive_4fs_3fs'})
  self.assertEqual(out['provenance']['legacy_slim_schema_version'],{2})
 def test_alternative_selection_signature_matches_inclusive_nominal(self):
  from analysis.workflows.config import WorkflowConfigBuilder
  from analysis.features.hww import signature,selection_signature
  a=WorkflowConfigBuilder('hww_legacy_slim_ab').build_workflow_config()
  b=WorkflowConfigBuilder('hww_legacy_slim_ab_hdamp').build_workflow_config()
  self.assertEqual(signature(a),signature(b));self.assertEqual(selection_signature(a),selection_signature(b))
 def test_deferred_workflow_cannot_be_accidentally_used_as_train_or_run(self):
  with self.assertRaisesRegex(ValueError,'Deferred signal selection'):
   BaseProcessor('hww_legacy_slim_ab','2022postEE','parquet','/tmp/unused-ab',mode='train')

if __name__=='__main__':unittest.main()
