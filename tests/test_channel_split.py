"""SR split Nc-j = 1 / Nc-j > 1 (AN-23-102 l.337-338), 2026-11-11."""
import unittest
import numpy as np
from analysis.sinks.hww import channel_layout, final_channels, card_combine, make_templates

CLASSES = ['hplusc', 'higgsbkg', 'tt']
BASE = {'classes': CLASSES, 'channels': {'SR_hplusc': 'hplusc', 'CR_higgsbkg': 'higgsbkg', 'CR_tt': 'tt'}}
SPLIT = dict(BASE, channel_splits={'SR_hplusc': {'variable': 'cjet_multiplicity',
                                                 'bins': {'SR_1cj': [1, 2], 'SR_2cj': [2, None]}}})


class ChannelSplit(unittest.TestCase):
    def test_layout(self):
        self.assertEqual(final_channels(BASE), ['SR_hplusc', 'CR_higgsbkg', 'CR_tt'])
        self.assertEqual(final_channels(SPLIT), ['SR_1cj', 'SR_2cj', 'CR_higgsbkg', 'CR_tt'])
        self.assertEqual(card_combine(SPLIT)['channels'],
                         {'SR_1cj': 'hplusc', 'SR_2cj': 'hplusc', 'CR_higgsbkg': 'higgsbkg', 'CR_tt': 'tt'})
        self.assertEqual(SPLIT['channels']['SR_hplusc'], 'hplusc')  # legacy view untouched

    def test_bad_layouts(self):
        for bins in ({'a': [1, 2], 'b': [3, None]},      # gap
                     {'a': [1, None], 'b': [2, 3]},      # open bin not last
                     {'a': [2, 1], 'b': [1, None]},      # reversed
                     {'a': [1, None]}):                  # single sub-channel
            with self.assertRaises(ValueError):
                channel_layout(dict(BASE, channel_splits={'SR_hplusc': {'variable': 'n', 'bins': bins}}))
        with self.assertRaises(ValueError):
            channel_layout(dict(BASE, channel_splits={'SR_x': {'variable': 'n', 'bins': {'a': [1, 2], 'b': [2, None]}}}))

    def test_templates_partition_the_sr(self):
        r = np.random.default_rng(0)
        n = 2000
        scores = r.dirichlet([1, 1, 1], n)
        ncj = r.integers(1, 4, n)
        w = r.normal(1, 0.3, n)
        edges = [0, 0.4, 0.6, 1.0]
        plain = make_templates(scores, CLASSES, BASE['channels'], edges, {'nominal': w})['nominal']
        split = make_templates(scores, CLASSES, BASE['channels'], edges, {'nominal': w},
                               layout=channel_layout(SPLIT),
                               split_values={'cjet_multiplicity': ncj})['nominal']
        np.testing.assert_allclose(split['counts']['SR_1cj'] + split['counts']['SR_2cj'],
                                   plain['counts']['SR_hplusc'])
        np.testing.assert_allclose(split['sumw2']['SR_1cj'] + split['sumw2']['SR_2cj'],
                                   plain['sumw2']['SR_hplusc'])
        np.testing.assert_allclose(split['counts']['CR_tt'], plain['counts']['CR_tt'])
        sr = np.argmax(scores, 1) == 0
        self.assertAlmostEqual(split['counts']['SR_1cj'].sum(), w[sr & (ncj == 1)].sum())
        self.assertAlmostEqual(split['counts']['SR_2cj'].sum(), w[sr & (ncj >= 2)].sum())
        self.assertNotIn('SR_hplusc', split['counts'])

    def test_uncovered_event_raises(self):
        scores = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])
        with self.assertRaises(ValueError):  # SR event with 0 c-jets falls in no sub-channel
            make_templates(scores, CLASSES, BASE['channels'], [0, 1], {'nominal': np.ones(2)},
                           layout=channel_layout(SPLIT), split_values={'cjet_multiplicity': np.array([0, 1])})
        with self.assertRaises(ValueError):  # split variable missing
            make_templates(scores, CLASSES, BASE['channels'], [0, 1], {'nominal': np.ones(2)},
                           layout=channel_layout(SPLIT))


if __name__ == '__main__':
    unittest.main()
