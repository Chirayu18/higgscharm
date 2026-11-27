"""Synthetic tests for Route B stitching (PROPOSAL HTOWW-ROUTEB-SOL-20261105). Run from the repo root (or with the
proposal's new_files/ first on sys.path):  python3 -m unittest tests.test_routeb_stitching -v
No input files, no network, no EOS."""
import json
import math
import os
import tempfile
import unittest
from glob import glob as local_glob

import awkward as ak
import numpy as np

from analysis.stitching.hplusc_routeb import (
    A_SCHEME, K_FACTOR, TABLE3_NO_CUTS_FB, RouteBConfig, StitchSums, WeightSums, chunk_sums, combine,
    count_gen_cjets, keep_mask, normalisation,
)
from analysis.stitching import routeb_io


def genjets(rows):
    """rows: list of events, each a list of (pt, eta, hadronFlavour)."""
    return ak.Array([[{"pt": float(p), "eta": float(e), "hadronFlavour": int(f)} for p, e, f in ev] for ev in rows])


def saranya_count_gen_cjets_c508ecc(genjets_, pt_min=10.0):
    """Verbatim logic of HcZZ-zx-estimation c508ecc analysis/processors/base.py::count_gen_cjets (no eta cut)."""
    is_c = (genjets_.hadronFlavour == 4) & (genjets_.pt > pt_min)
    return ak.to_numpy(ak.sum(is_c, axis=1))


def random_sample(rng, n, neg_frac, p_c, scale=1.0):
    """Synthetic sample: genWeight +-scale with a negative fraction, and a GenJet list."""
    w = np.where(rng.random(n) < neg_frac, -scale, scale) * rng.uniform(0.5, 1.5, n)
    rows = []
    for _ in range(n):
        nj = rng.integers(0, 6)
        ev = []
        for _ in range(nj):
            fl = 4 if rng.random() < p_c else int(rng.choice([0, 5]))
            ev.append((rng.uniform(5, 40), rng.uniform(-3.5, 3.5), fl))
        rows.append(ev)
    return w, genjets(rows)


class TestMasks(unittest.TestCase):
    def test_complementary_and_exclusive(self):
        n = np.array([0, 1, 2, 0, 7, 3, 0])
        k4, k3 = keep_mask(n, "4FS"), keep_mask(n, "3FS")
        self.assertTrue(np.all(k4 ^ k3))           # every event in exactly one scheme's kept region
        self.assertFalse(np.any(k4 & k3))
        np.testing.assert_array_equal(k4, n == 0)
        with self.assertRaises(ValueError):
            keep_mask(n, "5FS")

    def test_pt_threshold_strict(self):
        g = genjets([[(10.0, 0.0, 4)], [(10.0 + 1e-4, 0.0, 4)], [(9.99, 0.0, 4)], [(50.0, 0.0, 4)]])
        np.testing.assert_array_equal(count_gen_cjets(g), [0, 1, 0, 1])

    def test_eta_boundary_strict(self):
        g = genjets([[(20, 2.5, 4)], [(20, -2.5, 4)], [(20, 2.4999, 4)], [(20, -2.4999, 4)], [(20, 2.6, 4)]])
        np.testing.assert_array_equal(count_gen_cjets(g), [0, 0, 1, 1, 0])

    def test_flavour_and_multiplicity(self):
        g = genjets([[], [(20, 0, 5), (20, 0, 0)], [(20, 0, 4), (30, 1, 4), (15, -1, 5)], [(20, 3.0, 4), (11, 0.1, 4)]])
        np.testing.assert_array_equal(count_gen_cjets(g), [0, 0, 2, 1])

    def test_configurable_definition(self):
        g = genjets([[(20, 2.6, 4)], [(12, 0, 4)]])
        np.testing.assert_array_equal(count_gen_cjets(g, pt_min=15, eta_max=3.0), [1, 0])

    def test_matches_saranya_without_eta_cut(self):
        rng = np.random.default_rng(3)
        _, g = random_sample(rng, 3000, 0.3, 0.4)
        np.testing.assert_array_equal(count_gen_cjets(g, eta_max=np.inf), saranya_count_gen_cjets_c508ecc(g))
        # and the |eta|<2.5 cut only ever removes c-jets
        self.assertTrue(np.all(count_gen_cjets(g) <= saranya_count_gen_cjets_c508ecc(g)))


class TestPartitionIndependence(unittest.TestCase):
    def test_sums_independent_of_chunking_and_order(self):
        rng = np.random.default_rng(11)
        w, g = random_sample(rng, 5000, 0.4, 0.35)
        n = count_gen_cjets(g)
        whole, m = chunk_sums(w, n, "3FS")
        for trial in range(5):
            cuts = np.sort(rng.choice(np.arange(1, len(w)), size=rng.integers(1, 40), replace=False))
            parts = [chunk_sums(w[a:b], n[a:b], "3FS")[0]
                     for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(w)])]
            rng.shuffle(parts)
            tot = sum(parts, 0)
            self.assertEqual(tot.n_chunks, len(parts))
            for a, b in ((tot.full, whole.full), (tot.kept, whole.kept)):
                self.assertEqual((a.n, a.n_neg), (b.n, b.n_neg))
                for x in ("sumw", "sumw2", "sumw_abs"):
                    self.assertAlmostEqual(getattr(a, x), getattr(b, x), delta=1e-9 * max(1, abs(getattr(b, x))))
        self.assertAlmostEqual(whole.kept.sumw, float(w[m].sum()), places=9)

    def test_cannot_mix_definitions_or_schemes(self):
        a = chunk_sums(np.ones(3), np.array([0, 1, 2]), "4FS")[0]
        b = chunk_sums(np.ones(3), np.array([0, 1, 2]), "3FS")[0]
        c = chunk_sums(np.ones(3), np.array([0, 1, 2]), "4FS", eta_max=2.4)[0]
        with self.assertRaises(ValueError):
            a + b
        with self.assertRaises(ValueError):
            a + c

    def test_roundtrip(self):
        s = chunk_sums(np.array([1.0, -2.0, 3.0]), np.array([0, 1, 1]), "3FS", extra={"muR_up": np.array([1, 2, 3])})[0]
        self.assertEqual(StitchSums.from_dict(json.loads(json.dumps(s.to_dict()))), s)
        self.assertAlmostEqual(s.extra_kept["muR_up"], -2 * 2 + 3 * 3)


class TestNormalisation(unittest.TestCase):
    def _two_samples(self, seed=5):
        rng = np.random.default_rng(seed)
        w4, g4 = random_sample(rng, 20000, 0.13, 0.15, scale=3.0)   # FxFx-like
        w3, g3 = random_sample(rng, 20000, 0.40, 0.45, scale=0.7)   # 3FS-like, 40% negative
        s4, m4 = chunk_sums(w4, count_gen_cjets(g4), "4FS")
        s3, m3 = chunk_sums(w3, count_gen_cjets(g3), "3FS")
        return (w4, m4, s4), (w3, m3, s3)

    def test_closure_exact(self):
        (w4, m4, s4), (w3, m3, s3) = self._two_samples()
        nm = normalisation({"4FS": s4, "3FS": s3})
        L, X = 34652.0, 0.0016626   # pb^-1, pb (2022postEE-like lumi; fileset NNLO xsec)
        tot = (nm.scale("4FS", L, X, s4.full.sumw) * w4[m4].sum()
               + nm.scale("3FS", L, X, s3.full.sumw) * w3[m3].sum())
        self.assertAlmostEqual(tot / (L * X), 1.0, places=12)
        self.assertAlmostEqual(nm.closure, 1.0, places=12)

    def test_matches_paper_two_step(self):
        (_, _, s4), (_, _, s3) = self._two_samples(7)
        nm = normalisation({"4FS": s4, "3FS": s3})
        T = TABLE3_NO_CUTS_FB
        sig = {"4FS": T["NLO_4FS"], "3FS": T["NLO_3FS"]}
        f = {"4FS": s4.fraction, "3FS": s3.fraction}
        region = {s: K_FACTOR[s] * sig[s] * f[s] for s in sig}          # step 1
        S = T["MiNNLOPS_4FS"] / sum(region.values())                    # step 2
        for s in sig:   # stitched yield of region s / sigma_NNLO must agree
            self.assertAlmostEqual(S * region[s] / T["MiNNLOPS_4FS"], nm.c[s] * f[s], places=12)
        self.assertAlmostEqual(nm.S_paper, S, places=12)
        self.assertAlmostEqual(K_FACTOR["4FS"], 0.1690 / 0.2216, places=12)
        self.assertAlmostEqual(K_FACTOR["3FS"], 0.1630 / 0.1248, places=12)

    def test_eras_pooled_vs_per_era(self):
        (_, _, a4), (_, _, a3) = self._two_samples(1)
        (_, _, b4), (_, _, b3) = self._two_samples(2)
        pooled = normalisation(combine([a4, a3, b4, b3]))
        self.assertAlmostEqual(pooled.f["3FS"], (a3.kept.sumw + b3.kept.sumw) / (a3.full.sumw + b3.full.sumw), places=12)
        self.assertAlmostEqual(pooled.closure, 1.0, places=12)

    def test_fraction_stat_unit_weights_is_binomial(self):
        n = np.r_[np.zeros(300, int), np.ones(700, int)]
        s = chunk_sums(np.ones(1000), n, "3FS")[0]
        self.assertAlmostEqual(s.fraction_stat, math.sqrt(0.7 * 0.3 / 1000), places=12)

    def test_guards(self):
        good = chunk_sums(np.ones(4), np.array([0, 0, 1, 1]), "4FS")[0]
        bad = chunk_sums(np.array([1.0, -3.0]), np.array([1, 1]), "3FS")[0]
        with self.assertRaises(ValueError):
            normalisation({"4FS": good, "3FS": bad})            # F <= 0
        zero = chunk_sums(np.ones(2), np.array([0, 0]), "3FS")[0]
        with self.assertRaises(ValueError):
            normalisation({"4FS": good, "3FS": zero})           # kept fraction 0
        with self.assertRaises(ValueError):
            normalisation({"4FS": good})


class TestRecordsAndScales(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="routeb_test_")
        self.cfg = RouteBConfig.from_block({"4FS": ["HplusCharm_4FS_HToWWTo2L2Nu"],
                                            "3FS": ["HplusCharm_3FS_HToWW", "HplusCharm_3FS_HToWW_ext1"]})

    def _write(self, era, dataset, chunks, rng, neg, pc):
        import pyarrow as pa, pyarrow.parquet as pq
        base = os.path.join(self.tmp, era)
        ws = []
        for i in range(chunks):
            w, g = random_sample(rng, 400, neg, pc)
            s, _ = chunk_sums(w, count_gen_cjets(g), self.cfg.scheme_of(dataset))
            pkey = f"file{i}.root/Events/0-400"
            routeb_io.write_record_local(routeb_io.record_payload(s, dataset, era, pkey), os.path.join(base, dataset))
            os.makedirs(os.path.join(base, dataset, "sumw_records"), exist_ok=True)
            pq.write_table(pa.table({"sumw": [float(w.sum())]}),
                           os.path.join(base, dataset, "sumw_records", routeb_io.safe_key(pkey) + ".parquet"))
            ws.append(s)
        return base, sum(ws, 0)

    def test_scale_factors_main_plus_ext_share_F(self):
        rng = np.random.default_rng(9)
        base, s4 = self._write("2022postEE", "HplusCharm_4FS_HToWWTo2L2Nu", 4, rng, 0.13, 0.15)
        _, s3m = self._write("2022postEE", "HplusCharm_3FS_HToWW", 5, rng, 0.4, 0.45)
        _, s3e = self._write("2022postEE", "HplusCharm_3FS_HToWW_ext1", 2, rng, 0.4, 0.45)
        X = 0.0016626
        xs = {("2022postEE", d): X for d in self.cfg.datasets}
        out, norms = routeb_io.scale_factors(self.cfg, {"2022postEE": base}, {"2022postEE": 1000.0}, xs,
                                             glob=local_glob)
        F3 = s3m.full.sumw + s3e.full.sumw
        self.assertAlmostEqual(out[("2022postEE", "HplusCharm_3FS_HToWW")]["F_s"], F3, places=9)
        self.assertEqual(out[("2022postEE", "HplusCharm_3FS_HToWW")]["scale"],
                         out[("2022postEE", "HplusCharm_3FS_HToWW_ext1")]["scale"])
        tot = (out[("2022postEE", "HplusCharm_4FS_HToWWTo2L2Nu")]["scale"] * s4.kept.sumw
               + out[("2022postEE", "HplusCharm_3FS_HToWW")]["scale"] * (s3m.kept.sumw + s3e.kept.sumw))
        self.assertAlmostEqual(tot / (1000.0 * X), 1.0, places=10)

    def test_retry_is_not_double_counted_and_conflict_raises(self):
        rng = np.random.default_rng(4)
        base, s = self._write("e", "HplusCharm_3FS_HToWW", 3, rng, 0.4, 0.45)
        # identical record again in a shard dir (<dataset>_2): ignored
        src = local_glob(f"{base}/HplusCharm_3FS_HToWW/stitch_records/*.json")[0]
        os.makedirs(f"{base}/HplusCharm_3FS_HToWW_2/stitch_records")
        with open(src) as fh:
            payload = json.load(fh)
        with open(f"{base}/HplusCharm_3FS_HToWW_2/stitch_records/x.json", "w") as fh:
            json.dump(payload, fh)
        tot, n = routeb_io.read_records(base, "HplusCharm_3FS_HToWW", glob=local_glob)
        self.assertEqual(n, 3)
        self.assertAlmostEqual(tot.full.sumw, s.full.sumw, places=9)
        # ext1 is not swallowed by the main dataset's prefix
        self.assertEqual(routeb_io._shard_dirs(base, "HplusCharm_3FS_HToWW", "stitch_records", local_glob).__len__(), 2)
        payload["sums"]["full"]["sumw"] += 1.0
        with open(f"{base}/HplusCharm_3FS_HToWW_2/stitch_records/x.json", "w") as fh:
            json.dump(payload, fh)
        with self.assertRaises(RuntimeError):
            routeb_io.read_records(base, "HplusCharm_3FS_HToWW", glob=local_glob)

    def test_sumw_records_crosscheck_catches_missing_chunk(self):
        rng = np.random.default_rng(8)
        base, _ = self._write("e", "HplusCharm_4FS_HToWWTo2L2Nu", 3, rng, 0.13, 0.15)
        self._write("e", "HplusCharm_3FS_HToWW", 3, rng, 0.4, 0.45)
        os.remove(local_glob(f"{base}/HplusCharm_3FS_HToWW/stitch_records/*.json")[0])
        xs = {("e", d): 0.0016626 for d in self.cfg.datasets}
        with self.assertRaises(RuntimeError):
            routeb_io.scale_factors(self.cfg, {"e": base}, {"e": 1.0}, xs, glob=local_glob)

    def test_inconsistent_xsec_rejected(self):
        rng = np.random.default_rng(2)
        base, _ = self._write("e", "HplusCharm_4FS_HToWWTo2L2Nu", 1, rng, 0.1, 0.2)
        self._write("e", "HplusCharm_3FS_HToWW", 1, rng, 0.4, 0.45)
        xs = {("e", "HplusCharm_4FS_HToWWTo2L2Nu"): 0.0016626, ("e", "HplusCharm_3FS_HToWW"): 0.0011}
        with self.assertRaises(ValueError):
            routeb_io.scale_factors(self.cfg, {"e": base}, {"e": 1.0}, xs, glob=local_glob)

    def test_sink_entry_duplicate_chunk_detected(self):
        # mimic coffea accumulate: dict values under equal keys are added (str + str = concatenation)
        s = chunk_sums(np.array([1.0, -0.5, 2.0]), np.array([0, 1, 2]), "3FS")[0]
        p = json.dumps(routeb_io.record_payload(s, "HplusCharm_3FS_HToWW", "2022postEE", "f.root/Events/0-3"),
                       sort_keys=True)
        q = json.dumps(routeb_io.record_payload(s, "HplusCharm_3FS_HToWW", "2022postEE", "g.root/Events/0-3"),
                       sort_keys=True)
        ok = routeb_io.sums_from_sink_entry({"routeb_stitch": {"f.root/Events/0-3": p, "g.root/Events/0-3": q}})
        self.assertAlmostEqual(ok["HplusCharm_3FS_HToWW"].full.sumw, 2 * s.full.sumw)
        with self.assertRaises(RuntimeError):
            routeb_io.sums_from_sink_entry({"routeb_stitch": {"f.root/Events/0-3": p + p}})

    def test_config_parsing(self):
        with self.assertRaises(ValueError):
            RouteBConfig.from_block({"4FS": ["a"], "3FS": ["a"]})
        with self.assertRaises(ValueError):
            RouteBConfig.from_block({"4FS": ["a"]})
        self.assertIsNone(self.cfg.scheme_of("HplusCharm_3FS_HToWW_ext"))   # exact match only


if __name__ == "__main__":
    unittest.main()
