"""2026-11-12: corrected negrw validation statistics (v2 acceptance review F1/F2), checked with bounded toys.

Exact model g = 2 P+(x) - 1 with |w| constant per sample: the null hypothesis of the closure/shape tests.
Run from the repository root inside the coffea 0.7.30 container (about 10 s). No production data, no jobs.
"""
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts' / 'negrw'))
import negrw_stats as st  # noqa: E402


def train_script():
    spec = importlib.util.spec_from_file_location('negrw_train_v2_t', ROOT / 'scripts/negrw/negrw_train_v2.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


P = lambda x: 0.6 + 0.35 * x  # P+(x), as in the reviewer toy


def toys(rng, ntoy, n, absw=3.7):
    x = rng.random((ntoy, n)); s = np.where(rng.random((ntoy, n)) < P(x), 1., -1.)
    return x, absw * s, 2 * P(x) - 1


class TestClosureSigma(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tr = train_script()
        cls.x, cls.w, cls.g = toys(np.random.default_rng(5), 2000, 4000)

    def test_train_closure_matches_toy_spread(self):
        res = [self.tr.closure(w, g) for w, g in zip(self.w, self.g)]
        R = np.array([c['ratio'] for c in res]); sig = np.array([c['ratio_staterr'] for c in res])
        self.assertTrue(all(c['ratio_staterr_def'] == 'delta' for c in res))
        self.assertAlmostEqual(np.std(R, ddof=1) / np.mean(sig), 1.0, delta=0.06)  # ~1.6% stat precision
        self.assertLess(abs(np.mean((R - 1) / sig)), 0.1)  # unbiased pulls under the null
        old = np.array([np.sqrt(np.sum(w ** 2)) / abs(w.sum()) for w in self.w])
        self.assertGreater(np.mean(old) / np.std(R, ddof=1), 1.15)  # superseded formula overstates (~1.23 here)
        # same numbers as the stats module
        r0, e0 = st.ratio_and_err(self.w[0], self.g[0])
        self.assertAlmostEqual(r0, res[0]['ratio']); self.assertAlmostEqual(e0, res[0]['ratio_staterr'])

    def test_binned_ratio_sigma(self):
        edges = np.array([0, .25, .6, 1.])
        R, V = [], []
        for x, w, g in zip(self.x[:1500], self.w[:1500], self.g[:1500]):
            b = self.tr.binned(x, w, {'k': g}, edges)
            R.append(np.array(b['k']) / np.array(b['nom'])); V.append(b['k_ratio_var'])
            idx = np.digitize(x, edges[1:-1])
        ratio = np.std(np.array(R), axis=0, ddof=1) / np.sqrt(np.mean(np.array(V), axis=0))
        np.testing.assert_allclose(ratio, 1.0, atol=0.08)
        r, e = st.binned_ratio_err(idx, w, g, 3)
        np.testing.assert_allclose(r, R[-1]); np.testing.assert_allclose(e ** 2, V[-1])
        # under/overflow and NaN rows are outside the binned range and must not enter any bin
        xo = np.r_[x, -1., 2., np.nan]; wo = np.r_[w, 5., 5., 5.]; go = np.r_[g, .1, .1, .1]
        b = self.tr.binned(xo, wo, {'k': go}, edges)
        np.testing.assert_allclose(b['k_ratio_var'], V[-1])
        np.testing.assert_allclose(b['nom'], self.tr.binned(x, w, {'k': g}, edges)['nom'])


class TestShapeCov(unittest.TestCase):
    def test_null_chi2_and_covariance(self):
        rng = np.random.default_rng(1); nb, n, ntoy = 6, 5000, 1000
        chi2, rs, Cs, ranks = [], [], [], set()
        for _ in range(ntoy):
            x, w, g = (v[0] for v in toys(rng, 1, n))
            # two independent groups ("eras") with different |w|: per-group k, summed covariance
            w = np.where(np.arange(n) < n // 2, w, w * 0.4)
            idx = np.minimum((x * nb).astype(int), nb - 1)
            S = st.shape_cov(idx, nb, w, g, [np.arange(n) < n // 2, np.arange(n) >= n // 2])
            chi2.append(S['chi2']); rs.append(S['r']); Cs.append(S['C']); ranks.add(S['rank'])
        self.assertEqual(ranks, {nb - 1})  # one constraint per sum: total of (kA - N) is 0 per group
        self.assertAlmostEqual(np.mean(chi2), nb - 1, delta=0.4)  # sd of mean ~0.1
        E, Cf = np.cov(np.array(rs).T), np.mean(Cs, axis=0)
        np.testing.assert_allclose(np.diag(E) / np.diag(Cf), 1.0, atol=0.15)

    def test_biased_model_is_detected_and_nuisance_reduces_chi2(self):
        rng = np.random.default_rng(2); nb, n = 6, 1000000
        x, w, g = (v[0] for v in toys(rng, 1, n))
        idx = np.minimum((x * nb).astype(int), nb - 1)
        gb = g * (1 + 0.06 * (x - 0.5))  # 3% shape bias in g
        S = st.shape_cov(idx, nb, w, gb, [np.ones(n, bool)], g_up=gb + 0.06 * (x - 0.5) * g)
        self.assertGreater(S['chi2'], 40)
        self.assertLess(S['chi2_with_nuis'], S['chi2'])

    def test_too_few_events_or_bins(self):
        idx = np.zeros(30, int); w = np.ones(30); g = np.ones(30)
        self.assertIsNone(st.shape_cov(idx, 3, w, g, [np.ones(30, bool)]))  # group below 50 events
        self.assertIsNone(st.shape_cov(np.zeros(60, int), 3, np.ones(60), np.ones(60), [np.ones(60, bool)]))


if __name__ == '__main__':
    unittest.main()
