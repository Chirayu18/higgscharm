"""F14 external-input fingerprint (2026-11-10)."""
import time
import unittest
from unittest import mock

import analysis.features.hww as hww


def cfg():
    from analysis.workflows.config import WorkflowConfigBuilder
    return WorkflowConfigBuilder('hww_allfix_4fs').build_workflow_config()


class ExternalInputs(unittest.TestCase):
    def test_contents(self):
        ext = hww.external_inputs(cfg())
        sel = ext['selection']
        self.assertFalse(any(k.startswith('negrw:') for k in sel))
        self.assertTrue(any(k.startswith('negrw:') for k in ext['weights']))
        for name in hww.SELECTION_CORRECTIONS:
            for era in hww.MODEL_ERAS:
                self.assertTrue(any(k.startswith(f'{name}/{era}:') for k in sel), (name, era))
        self.assertTrue(any(k.startswith('data/lhe_yukawa/') for k in sel))
        self.assertTrue(any(k.startswith('ctagging_2d/') for k in ext['weights']))
        self.assertFalse(set(sel) & set(ext['weights']))

    def test_selection_change_changes_hash_weight_change_does_not(self):
        c = cfg()
        base = hww.selection_signature(c)
        real = hww.external_inputs
        def sel_changed(config):
            out = real(config); out['selection'] = dict(out['selection'], extra='x'); return out
        def wgt_changed(config):
            out = real(config); out['weights'] = dict(out['weights'], extra='x'); return out
        with mock.patch.object(hww, 'external_inputs', sel_changed):
            self.assertNotEqual(hww.selection_signature(c), base)
        with mock.patch.object(hww, 'external_inputs', wgt_changed):
            self.assertEqual(hww.selection_signature(c), base)

    def test_missing_negrw_model_raises(self):
        c = cfg()
        c.negrw = dict(c.negrw, model='/nonexistent/negrw.joblib')
        with self.assertRaises(FileNotFoundError):
            hww.external_inputs(c)

    def test_cached_second_call_is_fast(self):
        c = cfg(); hww.selection_signature(c)
        t = time.time(); hww.selection_signature(c)
        self.assertLess(time.time() - t, 2.0)


if __name__ == '__main__':
    unittest.main()
