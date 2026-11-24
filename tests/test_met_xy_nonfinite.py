import unittest
from unittest.mock import patch
import awkward as ak
import numpy as np
from analysis.corrections.met_xy import apply_run3_met_xy
class NonfiniteInput(unittest.TestCase):
 def run_corr(self,pt,phi,bad=False):
  raw=ak.Array({'pt':pt,'phi':phi})
  class Correction:
   def evaluate(self,*args):
    out=np.array(args[-3] if args[0]=='pt' else args[-2],copy=True)
    if bad:out[0]=np.nan
    return out
  events=ak.Array({'PV':{'npvsGood':np.full(len(pt),30.)},'genWeight':np.ones(len(pt))})
  with patch('analysis.corrections.met_xy.load_correction_set',return_value={'met_xy_corrections':Correction()}):
   return apply_run3_met_xy(events,'2022postEE',[({'MET':raw},None)],{'object_shifts':False})[0][0]['MET']
 def test_input_nan_does_not_abort_other_events_or_become_finite(self):
  out=self.run_corr([60.,np.nan,70.],[.1,np.nan,.2])
  np.testing.assert_array_equal(np.asarray(out.pt)[[0,2]],[60.,70.]);self.assertTrue(np.isnan(out.pt[1]))
  np.testing.assert_array_equal(np.asarray(out.pt)>45,[True,False,True])
 def test_finite_input_calibration_nan_still_fails(self):
  with self.assertRaisesRegex(ValueError,'from finite input'):self.run_corr([60.],[.1],bad=True)
class SignedVariation(unittest.TestCase):
 def test_down_crossing_zero_preserves_cartesian_vector(self):
  pt=np.array([.1]);phi=np.array([.3]);raw=ak.Array({'pt':pt,'phi':phi})
  class Correction:
   def evaluate(self,*args):
    q=args[0]
    if q.startswith('pt_stat'):return pt+1.
    return pt if q=='pt' else phi
  events=ak.Array({'PV':{'npvsGood':[30.]},'genWeight':[1.]})
  with patch('analysis.corrections.met_xy.load_correction_set',return_value={'met_xy_corrections':Correction()}):
   shifts=apply_run3_met_xy(events,'2022postEE',[({'MET':raw},None)],{'object_shifts':True})
  down=next(c['MET'] for c,n in shifts if n=='met_xy_2022Down')
  radial=pt-np.sqrt(2.)
  self.assertTrue(np.all(down.pt>=0))
  np.testing.assert_allclose(down.pt*np.cos(down.phi),radial*np.cos(phi),atol=1e-12)
  np.testing.assert_allclose(down.pt*np.sin(down.phi),radial*np.sin(phi),atol=1e-12)
if __name__=='__main__':unittest.main()
