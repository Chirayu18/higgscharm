"""Unit tests for the 2026-11-04 'all fixes' set (HTOWW-FULLFIX-RERUN-20261104). Synthetic inputs only, no EOS/xrootd.

Run from the repo root inside the coffea image (no pytest there):
    python3 -m unittest tests.test_allfix -v
Covers: lhe_yukawa_fix + lhescale (P0-15), trigger (P0-3), c-tag 2D one-hots (P0-7), Type-1 MET selection (P0-5),
jet re-sort (P0-6), 3FS/4FS shape (gen_chad1_pt + correctionlib weights), xsec guard (P0-2).
"""
import os
import sys
import types
import unittest

import awkward as ak
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "combine"))


# --------------------------------------------------------------------------------------------------------------------
# lhe_yukawa_fix + lhescale
# --------------------------------------------------------------------------------------------------------------------
class TestYukawaFix(unittest.TestCase):
    TABLE = "hpluscharm_4fs"

    @classmethod
    def setUpClass(cls):
        from analysis.corrections import lhe_yukawa_fix as y

        cls.y = y
        cls.lnmu, cls.fc, cls.fh, cls.f2 = y._load(cls.TABLE)

    def _sample(self, mu0, true_ratio):
        """Synthetic NanoAOD-like weights for events at scale mu0: stored variations carry F^-1 too much (see module doc).
        PDF replicas are +-symmetric around r = 1/F(mu0) so the replica mean is exactly r."""
        mu0 = np.asarray(mu0, dtype=float)
        f0 = np.interp(np.log(mu0), self.lnmu, self.fc)
        r = 1.0 / f0
        n = len(mu0)
        pdf = np.ones((n, 103))
        noise = np.tile(np.r_[np.full(50, 0.01), np.full(50, -0.01)], (n, 1))
        pdf[:, 1:101] = r[:, None] * (1.0 + noise)
        pdf[:, 101:] = r[:, None]
        stored = np.tile(np.asarray(true_ratio, dtype=float), (n, 1)) * r[:, None]
        stored[:, 4] = 1.0
        return stored, pdf, f0

    def test_table_for_dataset(self):
        t = self.y.table_for_dataset
        self.assertEqual(t("HplusCharm_4FS_HToWWTo2L2Nu"), "hpluscharm_4fs")
        self.assertEqual(t("HplusCharm_4FS_HToWWTo2L2Nu_12"), "hpluscharm_4fs")
        self.assertEqual(t("HplusBottom_5FS_HToWWTo2L2Nu_3"), "hplusbottom_5fs")
        self.assertIsNone(t("TTto2L2Nu_3"))
        self.assertIsNone(t("HplusCharm_4FS_HToWWTo2L2NuX"))  # prefix match needs the '_<partition>' boundary

    def test_tables_monotonic_and_start_at_1GeV(self):
        for tab in ("hpluscharm_4fs", "hplusbottom_5fs"):
            lnmu, fc, fh, f2 = self.y._load(tab)
            self.assertAlmostEqual(float(np.exp(lnmu[0])), 1.0, places=6)
            self.assertTrue(np.all(np.diff(fc) < 0))

    def test_muF_only_entries_exact(self):
        true = [1.0, 1.0, 1.0, 0.95, 1.0, 1.02, 1.0, 1.0, 1.0]
        stored, pdf, _ = self._sample([3.0, 10.0, 50.0, 200.0], true)
        corr, info = self.y.corrected_scale_weights(stored, pdf, self.TABLE)
        np.testing.assert_allclose(corr[:, 3], 0.95, rtol=1e-3)
        np.testing.assert_allclose(corr[:, 5], 1.02, rtol=1e-3)
        np.testing.assert_allclose(corr[:, 4], 1.0)
        self.assertEqual(info["n_fallback"], 0)
        self.assertEqual(info["n_offtable"], 0)

    def test_muR_entries_get_running_factor(self):
        true = [1.0] * 9
        mu0 = np.array([5.0, 20.0, 100.0])
        stored, pdf, f0 = self._sample(mu0, true)
        corr, _ = self.y.corrected_scale_weights(stored, pdf, self.TABLE)
        fh = np.interp(np.log(mu0), self.lnmu, self.fh)  # F(0.5 mu0; mu0)
        f2 = np.interp(np.log(mu0), self.lnmu, self.f2)  # F(2 mu0; mu0)
        # corrected = stored * F(k mu0; mu0) = true * r * F(k mu0; mu0) = F(k mu0; mu0) / F(mu0; mu0)
        np.testing.assert_allclose(corr[:, 1], fh / f0, rtol=2e-3)
        np.testing.assert_allclose(corr[:, 7], f2 / f0, rtol=2e-3)

    def test_clean_sample_refused(self):
        n = 200
        pdf = np.ones((n, 103)) * (1.0 + 0.005 * np.sin(np.arange(103)))[None, :]
        pdf[:, 0] = 1.0
        with self.assertRaises(RuntimeError):
            self.y.corrected_scale_weights(np.ones((n, 9)), pdf, self.TABLE)

    def test_bad_shapes_raise(self):
        with self.assertRaises(ValueError):
            self.y.corrected_scale_weights(np.ones((5, 8)), np.ones((5, 103)), self.TABLE)
        with self.assertRaises(ValueError):
            self.y.corrected_scale_weights(np.ones((5, 9)), np.ones((5, 50)), self.TABLE)

    def test_pathological_events_get_unit_variation(self):
        true = [1.0, 1.0, 1.0, 0.9, 1.0, 1.1, 1.0, 1.0, 1.0]
        stored, pdf, _ = self._sample([5.0] * 400, true)
        pdf[0, 1:101] = -1.0  # r <= 0 -> no variation (ratio 1)
        corr, info = self.y.corrected_scale_weights(stored, pdf, self.TABLE)
        self.assertTrue(np.all(corr[0, [0, 1, 2, 3, 5, 6, 7, 8]] == 1.0))
        self.assertEqual(info["n_fallback"], 1)
        np.testing.assert_allclose(corr[1:, 3], 0.9, rtol=1e-3)


class TestScaleVarWeights(unittest.TestCase):
    """add_scalevar_weight: Up/Down labelling (HcZZ port), nominal factor 1, and the yukawa path."""

    @staticmethod
    def _events(scale, pdf=None):
        d = {"LHEScaleWeight": ak.Array(scale)}
        if pdf is not None:
            d["LHEPdfWeight"] = ak.Array(pdf)
        ev = types.SimpleNamespace(**d)
        ev.fields = list(d)
        return ev

    def test_up_down_labels_and_nominal_untouched(self):
        from coffea.analysis_tools import Weights
        from analysis.corrections.lhescale import add_scalevar_weight

        w9 = np.array([[0.5, 0.6, 0.7, 0.8, 2.0, 1.2, 1.3, 1.4, 1.5]] * 4)
        ev = self._events(w9)
        w = Weights(4, storeIndividual=True)
        w.add("gen", np.full(4, 3.0))
        add_scalevar_weight(ev, w, None)
        r = w9 / w9[:, [4]]
        np.testing.assert_allclose(w.weight(), 3.0)  # nominal weight not multiplied by LHEScaleWeight[4]
        # Up = scale x2 entries [7] muR, [5] muF, [8] both; Down = x0.5 entries [1], [3], [0]
        np.testing.assert_allclose(w.weight("scalevar_muRUp"), 3.0 * r[:, 7])
        np.testing.assert_allclose(w.weight("scalevar_muRDown"), 3.0 * r[:, 1])
        np.testing.assert_allclose(w.weight("scalevar_muFUp"), 3.0 * r[:, 5])
        np.testing.assert_allclose(w.weight("scalevar_muFDown"), 3.0 * r[:, 3])
        np.testing.assert_allclose(w.weight("scalevar_muR_muFUp"), 3.0 * r[:, 8])
        np.testing.assert_allclose(w.weight("scalevar_muR_muFDown"), 3.0 * r[:, 0])

    def test_yukawa_path_replaces_ratio_and_keeps_nominal(self):
        from coffea.analysis_tools import Weights
        from analysis.corrections import lhe_yukawa_fix as y
        from analysis.corrections.lhescale import add_scalevar_weight

        t = TestYukawaFix()
        TestYukawaFix.setUpClass()
        true = [1.0, 1.0, 1.0, 0.93, 1.0, 1.05, 1.0, 1.0, 1.0]
        stored, pdf, _ = t._sample([50.0] * 6, true)  # hard-scale events: r = 1/F(mu0) ~ 2, passes the r>1.2 guard
        ev = self._events(stored, pdf)
        w = Weights(6, storeIndividual=True)
        w.add("gen", np.ones(6))
        add_scalevar_weight(ev, w, None, yukawa_table="hpluscharm_4fs")
        np.testing.assert_allclose(w.weight(), 1.0)
        np.testing.assert_allclose(w.weight("scalevar_muFDown"), 0.93, rtol=1e-3)  # entry [3]
        np.testing.assert_allclose(w.weight("scalevar_muFUp"), 1.05, rtol=1e-3)  # entry [5]
        # without the fix the stored (F^-1 too large) ratios would leak through
        w2 = Weights(6, storeIndividual=True)
        w2.add("gen", np.ones(6))
        add_scalevar_weight(ev, w2, None, yukawa_table=None)
        self.assertGreater(float(np.mean(w2.weight("scalevar_muFDown"))), 1.0)

    def test_yukawa_needs_pdf_weights(self):
        from coffea.analysis_tools import Weights
        from analysis.corrections.lhescale import add_scalevar_weight

        ev = self._events(np.ones((3, 9)))
        w = Weights(3, storeIndividual=True)
        w.add("gen", np.ones(3))
        with self.assertRaises(RuntimeError):
            add_scalevar_weight(ev, w, None, yukawa_table="hpluscharm_4fs")


# --------------------------------------------------------------------------------------------------------------------
# trigger (P0-3)
# --------------------------------------------------------------------------------------------------------------------
class _Ev:
    def __init__(self, hlt, mc=False):
        self.HLT = {k: np.asarray(v, dtype=bool) for k, v in hlt.items()}
        self._n = len(next(iter(self.HLT.values())))
        if mc:
            self.genWeight = np.ones(self._n)

    def __len__(self):
        return self._n


HWW_HLT = {"MuonEG": ["MuEle"], "Muon": ["SingleMu"], "EGamma": ["SingleEle"]}
MUELE = "Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL"


def _hlt(mu=(0, 0, 0, 0), ele=(0, 0, 0, 0), mue=(0, 0, 0, 0)):
    # events: [mu only, ele only, mu+ele cross trigger, mu + single-mu + ele]
    d = {p: np.zeros(4, bool) for p in (
        "Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL", "Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
        "Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ", "Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
        "DiMu9_Ele9_CaloIdL_TrackIdL_DZ", "Mu8_DiEle12_CaloIdL_TrackIdL_DZ")}
    d["IsoMu24"] = np.array(mu, bool)
    d["Ele30_WPTight_Gsf"] = np.array(ele, bool)
    d[MUELE] = np.array(mue, bool)
    return d


class TestTrigger(unittest.TestCase):
    def setUp(self):
        from analysis.selections.trigger import trigger_mask

        self.tm = trigger_mask
        # event 0: IsoMu24 only; 1: Ele30 only; 2: MuEle cross only; 3: IsoMu24 + MuEle
        self.hlt = _hlt(mu=(1, 0, 0, 1), ele=(0, 1, 0, 0), mue=(0, 0, 1, 1))

    def test_data_pds_exclusive(self):
        ev = _Ev(self.hlt)
        mueg = self.tm(ev, HWW_HLT, "MuonEGRun2022E", "2022postEE")
        muon = self.tm(ev, HWW_HLT, "MuonRun2022E", "2022postEE")
        eg = self.tm(ev, HWW_HLT, "EGammaRun2022E", "2022postEE")
        np.testing.assert_array_equal(mueg, [0, 0, 1, 1])
        np.testing.assert_array_equal(muon, [1, 0, 0, 0])
        np.testing.assert_array_equal(eg, [0, 1, 0, 0])

    def test_singlemuon_2022C_behaves_like_muon(self):
        ev = _Ev(self.hlt)
        sm = self.tm(ev, HWW_HLT, "SingleMuonRun2022C", "2022preEE")
        np.testing.assert_array_equal(sm, [1, 0, 0, 0])  # NOT the OR of all triggers

    def test_doublemuon_2022C_contributes_nothing(self):
        ev = _Ev(self.hlt)
        dm = self.tm(ev, HWW_HLT, "DoubleMuonRun2022C", "2022preEE")
        self.assertFalse(dm.any())

    def test_unknown_data_dataset_raises(self):
        ev = _Ev(self.hlt)
        with self.assertRaises(ValueError):
            self.tm(ev, HWW_HLT, "JetMETRun2022C", "2022preEE")

    def test_mc_is_or_of_all(self):
        ev = _Ev(self.hlt, mc=True)
        mc = self.tm(ev, HWW_HLT, "TTto2L2Nu_3", "2022preEE")
        np.testing.assert_array_equal(mc, [1, 1, 1, 1])


# --------------------------------------------------------------------------------------------------------------------
# c-tag 2D one-hots (P0-7)
# --------------------------------------------------------------------------------------------------------------------
class TestCtag2dOneHot(unittest.TestCase):
    def test_onehot_is_exclusive_and_complete(self):
        from analysis.corrections.ctag2d_categories import ONEHOT_COLUMNS, onehot_columns

        rng = np.random.default_rng(1)
        cvsl, cvsb = rng.random(5000), rng.random(5000)
        oh = onehot_columns(cvsl, cvsb)
        self.assertEqual(list(oh), ONEHOT_COLUMNS)
        tot = sum(oh.values())
        self.assertTrue(np.all(tot == 1))  # every defined point lands in exactly one of the 11 categories

    def test_known_points(self):
        from analysis.corrections.ctag2d_categories import CATS, category_np

        # x = cvsl/(cvsl + cvsb (1-cvsl)), y = 1-cvsb; edges x: .25 .452 .808, y: .006 .017 .055 .761 .944 .985 .995
        cvsl = [0.02, 0.9, 0.9, 0.95, 0.3]
        cvsb = [0.5, 0.001, 0.99, 0.5, 0.5]
        # x=.039 -> L0 | x=.9999, y=.999 -> B4 | x=.9009, y=.01 -> C3 | x=.974, y=.5 -> B0 | x=.4615 -> C1
        expected = ["L0", "B4", "C3", "B0", "C1"]
        cat = category_np(cvsl, cvsb)
        self.assertEqual([CATS[i] for i in cat], expected)

    def test_undefined_gives_all_zero(self):
        from analysis.corrections.ctag2d_categories import onehot_columns

        oh = onehot_columns([0.0, np.nan], [0.0, 0.5])
        self.assertEqual(sum(v[0] for v in oh.values()), 0)  # den = 0 -> undefined
        self.assertEqual(sum(v[1] for v in oh.values()), 0)  # nan

    def test_matches_bhive_reference(self):
        ref = "/eos/user/c/cgupta/HToWW/b-hive/scripts/append_onehot.py"
        if not os.path.exists(ref):
            self.skipTest("b-hive reference not mounted")
        import importlib.util

        spec = importlib.util.spec_from_file_location("append_onehot_ref", ref)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        from analysis.corrections.ctag2d_categories import category_np

        rng = np.random.default_rng(7)
        cvsl, cvsb = rng.random(20000), rng.random(20000)
        cvsl[:50], cvsb[:50] = 0.0, 0.0
        np.testing.assert_array_equal(category_np(cvsl, cvsb), mod.ctag2d_category_np(cvsl, cvsb))

    def test_ctag2d_module_uses_shared_function(self):
        with open(os.path.join(REPO, "analysis", "corrections", "ctag2d.py")) as fh:
            src = fh.read()
        self.assertIn("ctag2d_categories", src)


# --------------------------------------------------------------------------------------------------------------------
# Type-1 MET selection (P0-5) and jet re-sort (P0-6)
# --------------------------------------------------------------------------------------------------------------------
class TestType1Met(unittest.TestCase):
    @staticmethod
    def _jets(orig_pt, phi, raw=None, chem=None, nhem=None, msf=None):
        return ak.zip({
            "orig_pt": ak.Array(orig_pt), "phi": ak.Array(phi),
            "pt_raw": ak.Array(raw if raw is not None else [[0.8 * x for x in e] for e in orig_pt]),
            "chEmEF": ak.Array(chem if chem is not None else [[0.1] * len(e) for e in orig_pt]),
            "neEmEF": ak.Array(nhem if nhem is not None else [[0.1] * len(e) for e in orig_pt]),
            "muonSubtrFactor": ak.Array(msf if msf is not None else [[0.0] * len(e) for e in orig_pt]),
        })

    @staticmethod
    def _met(pt, phi):
        return types.SimpleNamespace(pt=ak.Array(pt), phi=ak.Array(phi))

    def test_unchanged_jets_leave_met_unchanged(self):
        from analysis.corrections.jerc import type1_corrected_met

        jets = self._jets([[50.0, 20.0]], [[0.3, 2.0]])
        met = self._met([60.0], [1.0])
        out = type1_corrected_met(met, jets, jets.orig_pt)
        np.testing.assert_allclose(ak.to_numpy(out.pt), [60.0], rtol=1e-6)
        np.testing.assert_allclose(ak.to_numpy(out.phi), [1.0], rtol=1e-6)

    def test_jet_scaled_up_moves_met_against_it(self):
        from analysis.corrections.jerc import type1_corrected_met

        jets = self._jets([[100.0]], [[0.0]])  # jet along +x
        met = self._met([50.0], [np.pi])  # MET along -x (recoil)
        out = type1_corrected_met(met, jets, ak.Array([[110.0]]))  # jet +10 GeV -> MET_x: -50 - 10
        np.testing.assert_allclose(ak.to_numpy(out.pt), [60.0], rtol=1e-5)
        out_dn = type1_corrected_met(met, jets, ak.Array([[90.0]]))  # jet -10 GeV -> MET_x: -50 + 10
        np.testing.assert_allclose(ak.to_numpy(out_dn.pt), [40.0], rtol=1e-5)

    def test_em_dominated_and_soft_jets_do_not_move_met(self):
        from analysis.corrections.jerc import type1_corrected_met

        jets = self._jets([[10.0, 80.0]], [[0.0, 0.0]], chem=[[0.1, 0.5]], nhem=[[0.1, 0.5]])  # soft; EM fraction 1.0
        met = self._met([50.0], [np.pi])
        out = type1_corrected_met(met, jets, ak.Array([[12.0, 160.0]]))  # soft stays < 15; EM jet doubled
        np.testing.assert_allclose(ak.to_numpy(out.pt), [50.0], rtol=1e-6)

    def test_jet_entering_threshold_adds_new_minus_raw(self):
        from analysis.corrections.jerc import type1_corrected_met

        jets = self._jets([[10.0]], [[0.0]], raw=[[8.0]])  # nano pT 10 < 15: not in the NanoAOD Type-1 sum
        met = self._met([50.0], [np.pi])
        out = type1_corrected_met(met, jets, ak.Array([[20.0]]))  # new 20 > 15: enters with (20 - 8) = 12, not 20
        np.testing.assert_allclose(ak.to_numpy(out.pt), [62.0], rtol=1e-5)

    def test_muon_dominated_jet_below_threshold_excluded(self):
        from analysis.corrections.jerc import type1_jet_pt

        jets = self._jets([[100.0]], [[0.0]], msf=[[0.9]])  # muon-subtracted pt = 10 < 15
        self.assertEqual(float(ak.to_numpy(ak.flatten(type1_jet_pt(jets, jets.orig_pt)))[0]), 0.0)

    def test_missing_fields_raise(self):
        from analysis.corrections.jerc import type1_jet_pt

        jets = ak.zip({"orig_pt": ak.Array([[30.0]]), "phi": ak.Array([[0.0]])})
        with self.assertRaises(KeyError):
            type1_jet_pt(jets, jets.orig_pt)


class TestJetResort(unittest.TestCase):
    def test_jets_sorted_by_pt(self):
        from analysis.selections.object_selections import ObjectSelector

        jets = ak.zip({"pt": ak.Array([[20.0, 80.0, 40.0], [30.0]]), "eta": ak.Array([[0.1, 0.2, 0.3], [0.4]]),
                       "phi": ak.Array([[0.1, 0.2, 0.3], [0.4]]), "mass": ak.Array([[1.0, 2.0, 3.0], [4.0]])})
        fake = types.SimpleNamespace(objects={"jets": jets})
        ObjectSelector.select_leading_jets(fake, "jets")
        self.assertEqual(ak.to_list(fake.objects["jets"].pt), [[80.0, 40.0, 20.0], [30.0]])
        np.testing.assert_allclose(ak.to_numpy(fake.objects["jet1"].pt), [80.0, 30.0])
        np.testing.assert_allclose(ak.to_numpy(ak.fill_none(fake.objects["jet2"].pt, -1)), [40.0, -1])
        np.testing.assert_allclose(ak.to_numpy(ak.fill_none(fake.objects["jet3"].pt, -1)), [20.0, -1])


# --------------------------------------------------------------------------------------------------------------------
# 3FS/4FS shape
# --------------------------------------------------------------------------------------------------------------------
class TestFsShape(unittest.TestCase):
    @staticmethod
    def _genpart(events):
        """events: list of lists of (pdgId, status, mother_index, pt, eta)."""
        def col(i):
            return ak.Array([[p[i] for p in ev] for ev in events])

        return ak.zip({"pdgId": col(0), "status": col(1), "genPartIdxMother": col(2), "pt": col(3), "eta": col(4)})

    def test_gen_chad1_pt_definition(self):
        from analysis.corrections.hplusc_fs import gen_chad1_pt

        ev0 = [  # D*+ (413) -> D0 (421) pi: D0 is the weakly decaying one (pt 30); D* has a c-hadron daughter
            (413, 2, -1, 35.0, 0.1), (421, 2, 0, 30.0, 0.1), (211, 1, 0, 5.0, 0.1),
            (4122, 2, -1, 12.0, 0.2),  # Lambda_c, lower pT
        ]
        ev1 = [(521, 2, -1, 50.0, 0.0), (421, 2, 0, 20.0, 0.0)]  # D0 from a B+ -> excluded -> no c-hadron
        ev2 = [(443, 2, -1, 40.0, 0.0), (421, 2, -1, 9.0, 3.0)]  # J/psi excluded, D0 outside |eta|<2.5
        ev3 = [(11, 1, -1, 25.0, 0.0)]  # no charm
        ev4 = [(-431, 2, -1, 7.0, -1.0), (-4122, 2, -1, 3.0, 0.0)]  # anti-Ds (leading 7), anti-Lambda_c
        ev5 = [(20413, 2, -1, 15.0, 0.0)]  # 204xx excited state, weakly decaying here
        out = gen_chad1_pt(types.SimpleNamespace(GenPart=self._genpart([ev0, ev1, ev2, ev3, ev4, ev5])))
        np.testing.assert_allclose(out, [30.0, -1.0, -1.0, -1.0, 7.0, 15.0])

    def test_shape_weights_from_correctionlib(self):
        try:
            import correctionlib  # noqa: F401
        except ImportError:
            self.skipTest("no correctionlib")
        from analysis.corrections.hplusc_fs import fs_shape_weights

        pts = np.array([-1.0, 3.0, 7.0, 12.0, 25.0, 200.0, 5e5])
        up, down = fs_shape_weights(pts)
        np.testing.assert_allclose(up[[0, 1]], 0.7756648516965607)  # no c-hadron / below 5 GeV: w0
        np.testing.assert_allclose(up[2], 1.4096625444720947)
        np.testing.assert_allclose(up[3], 1.2308781813967482)
        np.testing.assert_allclose(up[5], 1.1830310675733624)
        np.testing.assert_allclose(up[6], 1.1830310675733624)  # clamp
        np.testing.assert_allclose(up + down, 2.0, atol=1e-12)  # Down = 2 - Up (mirror)

    def test_add_weight_gating_and_names(self):
        from coffea.analysis_tools import Weights
        from analysis.corrections.hplusc_fs import add_hplusc_fs_weight

        gp = self._genpart([[(421, 2, -1, 7.0, 0.0)], [(11, 1, -1, 25.0, 0.0)]])
        class _E(types.SimpleNamespace):
            def __len__(self):
                return 2

        ev = _E(GenPart=gp, fields=["GenPart"])
        ds = ["HplusCharm_4FS_HToWWTo2L2Nu"]
        w = Weights(2, storeIndividual=True)
        w.add("gen", np.ones(2))
        add_hplusc_fs_weight(ev, w, None, "HplusCharm_4FS_HToWWTo2L2Nu_5", ds)
        np.testing.assert_allclose(w.weight(), 1.0)  # nominal untouched
        np.testing.assert_allclose(w.weight("xsec_hplusc_3FS_4FSUp"), [1.4096625444720947, 0.7756648516965607])
        np.testing.assert_allclose(w.weight("xsec_hplusc_3FS_4FSDown"), [0.5903374555279053, 1.2243351483034393])
        for args in ((None, "TTto2L2Nu_1", ds), ("JESup", "HplusCharm_4FS_HToWWTo2L2Nu_5", ds),
                     (None, "HplusCharm_4FS_HToWWTo2L2Nu_5", None)):
            w2 = Weights(2, storeIndividual=True)
            w2.add("gen", np.ones(2))
            add_hplusc_fs_weight(ev, w2, args[0], args[1], args[2])
            self.assertNotIn("xsec_hplusc_3FS_4FSUp", w2.variations)


# --------------------------------------------------------------------------------------------------------------------
# xsec guard (P0-2)
# --------------------------------------------------------------------------------------------------------------------
class TestXsecGuard(unittest.TestCase):
    def test_zero_xsec_raises(self):
        import make_combine_inputs as m

        orig = m.get_dataset_config
        m.get_dataset_config = lambda year: {"WminusH_WtoLNu_Hto2Wto2L2Nu": {"era": "mc", "xsec": 0.0}}
        try:
            with self.assertRaises(ValueError):
                m.read_scale("WminusH_WtoLNu_Hto2Wto2L2Nu", "2022preEE", "/nonexistent", 1.0)
        finally:
            m.get_dataset_config = orig


# --------------------------------------------------------------------------------------------------------------------
# retrained ttH lepton MVA (tvl/MVA port, 2026-11-05): ONNX == TMVA, vectorised features, working points
# --------------------------------------------------------------------------------------------------------------------
class TestTthMVA(unittest.TestCase):
    FIXTURE = os.path.join(REPO, "tests", "data", "tthmva_ref_fixture.npz")

    def test_onnx_matches_tmva_including_ties(self):
        # fixture: real 2022postEE TTto2L2Nu leptons, every tie-sensitive one (input exactly on a TMVA split value:
        # 12 mu, 861 e out of ~25k each) + 500 random per flavour; ref = TMVA-exact (x >= cut, validated vs TMVA::Reader)
        from analysis.leptonmva.tth import score_features
        D = np.load(self.FIXTURE)
        for coll, key in (("Muon", "muons"), ("Electron", "electrons")):
            s = score_features(D[key + "_X"], coll)
            self.assertLess(float(np.max(np.abs(s - D[key + "_ref"]))), 1e-6, coll)
            tie = D[key + "_tie"]
            self.assertGreater(int(tie.sum()), 0)
            self.assertLess(float(np.max(np.abs(s[tie] - D[key + "_ref"][tie]))), 1e-6, coll + " ties")

    def _events(self):
        lep = dict(pt=[[30.0, 12.0], [], [50.0]], eta=[[0.1, -1.0], [], [2.0]], pfRelIso03_all=[[0.01, 0.2], [], [0.0]],
                   miniPFRelIso_chg=[[0.01, 0.05], [], [0.0]], miniPFRelIso_all=[[0.02, 0.3], [], [0.0]],
                   jetNDauCharged=[[3, 1], [], [0]], jetPtRelv2=[[5.0, 0.5], [], [0.0]], jetIdx=[[1, -1], [], [0]],
                   jetRelIso=[[0.1, -0.9], [], [0.5]], sip3d=[[1.0, 4.0], [], [0.5]], dxy=[[0.001, -0.02], [], [0.0]],
                   dz=[[0.002, 0.01], [], [-0.003]], segmentComp=[[0.9, 0.4], [], [0.8]], mvaIso=[[0.99, 0.5], [], [0.1]])
        jets = dict(btagDeepFlavB=[[0.1, 0.7], [0.3], [0.05, 0.2]])
        return ak.zip({"Muon": ak.zip(lep), "Electron": ak.zip(lep), "Jet": ak.zip(jets)}, depth_limit=1)

    def test_features_vectorised(self):
        from analysis.leptonmva.tth import lepton_features
        X, counts = lepton_features(self._events(), "Muon")
        self.assertEqual(list(counts), [2, 0, 1])
        self.assertEqual(X.shape, (3, 13))
        np.testing.assert_allclose(X[:, 7], [0.7, 0.0, 0.05])            # jetIdx>-1 ? Jet_btagDeepFlavB[jetIdx] : 0
        np.testing.assert_allclose(X[:, 8], [1 / 1.1, 1.5, 1 / 1.5])     # min(1/(1+jetRelIso), 1.5)
        np.testing.assert_allclose(X[:, 4], [0.01, 0.25, 0.0])           # miniPFRelIso_all - miniPFRelIso_chg
        self.assertTrue(np.isneginf(X[2, 10]))                           # log|dxy| of 0 -> -inf, as TMVA sees it
        Xe, _ = lepton_features(self._events(), "Electron")
        np.testing.assert_allclose(Xe[:, 12], [0.99, 0.5, 0.1])          # electron: mvaIso, muon: segmentComp
        np.testing.assert_allclose(X[:, 12], [0.9, 0.4, 0.8])

    def test_attach_and_working_points(self):
        from analysis.leptonmva.tth import add_tthmva_scores
        from analysis.working_points.working_points import WorkingPoints
        ev = add_tthmva_scores(self._events(), "2022postEE")
        self.assertEqual(ak.num(ev.Muon.tthMVA).tolist(), [2, 0, 1])
        self.assertTrue(np.all(np.abs(ak.to_numpy(ak.flatten(ev.Muon.tthMVA))) < 1))
        mu = ak.to_numpy(ak.flatten(WorkingPoints().muon_promptMVA(ev, "2022postEE", "tight_HWW")))
        np.testing.assert_array_equal(mu, ak.to_numpy(ak.flatten(ev.Muon.tthMVA)) > 0.67)
        ev2 = add_tthmva_scores(self._events(), "2018")                 # other years: no field
        self.assertNotIn("tthMVA", ev2.Muon.fields)


class TestPdfOnlyCombined(unittest.TestCase):
    def test_no_alphaS_combined_keeps_pdf_uncertainty(self):
        from coffea.analysis_tools import Weights
        from analysis.corrections.lhepdf import add_lhepdf_weight

        # A symmetric Hessian set with central 1 and 100 members displaced by .01 has sigma=.1.
        pdf = np.ones((2, 101))
        pdf[:, 1:] += .01
        ev = types.SimpleNamespace(LHEPdfWeight=ak.Array(pdf), fields=["LHEPdfWeight"])
        weights = Weights(2, storeIndividual=True)
        add_lhepdf_weight(ev, weights, None)
        np.testing.assert_allclose(weights.weight("lhe_pdfUp"), 1.1)
        np.testing.assert_allclose(weights.weight("lhe_pdf_alphaSUp"), 1.1)
        np.testing.assert_allclose(weights.weight("lhe_pdf_alphaSDown"), .9)
        np.testing.assert_allclose(weights.weight("lhe_alphaSUp"), 1.)


if __name__ == "__main__":
    unittest.main()
