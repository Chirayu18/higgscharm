"""Reproducible per-object random numbers (2026-11-10)."""
import unittest
import numpy as np
import awkward as ak
import correctionlib

from analysis.corrections.event_rng import object_uniforms, object_normals
from analysis.corrections.correctionlib_files import correction_files
from analysis.corrections.muon_ss import pt_resol


def make_events(n=400, seed=1):
    r = np.random.default_rng(seed)
    nmu = r.integers(0, 4, n)
    tot = int(nmu.sum())
    muons = ak.unflatten(ak.zip({
        "pt": r.uniform(15, 150, tot),
        "eta": r.uniform(-2.4, 2.4, tot),
        "nTrackerLayers": r.integers(8, 17, tot).astype(np.int32),
    }), nmu)
    return ak.zip({
        "run": np.full(n, 1, dtype=np.uint32),
        "luminosityBlock": r.integers(1, 50, n).astype(np.uint32),
        "event": r.permutation(np.arange(10**9, 10**9 + n)).astype(np.uint64),
        "Muon": muons,
    }, depth_limit=1)


class EventRNG(unittest.TestCase):
    def test_chunk_invariant(self):
        ev = make_events()
        full = object_uniforms(ev, "Muon", "muon_resolution")
        parts = np.concatenate([object_uniforms(ev[a:b], "Muon", "muon_resolution")
                                for a, b in [(0, 7), (7, 230), (230, 400)]])
        np.testing.assert_array_equal(full, parts)
        # reordering events moves the numbers with the events
        perm = np.random.default_rng(3).permutation(len(ev))
        np.testing.assert_array_equal(
            ak.to_numpy(ak.flatten(ak.unflatten(full, ak.num(ev.Muon))[perm])),
            object_uniforms(ev[perm], "Muon", "muon_resolution"))

    def test_range_and_streams(self):
        ev = make_events(5000)
        u = object_uniforms(ev, "Muon", "muon_resolution")
        self.assertTrue(np.all((u > 0) & (u < 1)))
        self.assertLess(abs(u.mean() - 0.5), 0.01)
        self.assertEqual(len(np.unique(u)), len(u))
        # a different purpose gives an independent stream
        ev2 = ak.with_field(ev, ev.Muon, "Electron")
        v = object_uniforms(ev2, "Electron", "electron_smearing")
        self.assertLess(abs(np.corrcoef(u, v)[0, 1]), 0.05)
        z = object_normals(ev2, "Electron", "electron_smearing")
        self.assertLess(abs(z.mean()), 0.05)
        self.assertLess(abs(z.std() - 1), 0.05)

    def test_muon_resolution_chunk_invariant_real_cset(self):
        cset = correctionlib.CorrectionSet.from_file(correction_files["muon_ss"]["2022postEE"])
        ev = make_events()

        def corr(e):
            m = e.Muon
            return ak.to_numpy(ak.flatten(pt_resol(m.pt, m.eta, m.nTrackerLayers, cset, nested=True,
                                                   uniforms=object_uniforms(e, "Muon", "muon_resolution"))))
        full = corr(ev)
        parts = np.concatenate([corr(ev[a:b]) for a, b in [(0, 100), (100, 101), (101, 400)]])
        np.testing.assert_allclose(full, parts, rtol=0, atol=0)
        self.assertFalse(np.allclose(full, ak.to_numpy(ak.flatten(ev.Muon.pt))))  # smearing applied
        m = ev.Muon
        with self.assertRaises(ValueError):
            pt_resol(m.pt, m.eta, m.nTrackerLayers, cset, nested=True)


if __name__ == "__main__":
    unittest.main()
