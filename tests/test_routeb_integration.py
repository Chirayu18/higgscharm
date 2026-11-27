"""Route B integration tests (2026-11-12): base.py hook pieces, 3FS PDF replica fix, sink scale factors,
card-builder normalisation, workflow/fileset wiring. Synthetic inputs + the real workflow/era YAMLs; no jobs.

Run from the repository root inside the coffea 0.7.30 container:
    python3 -m unittest tests.test_routeb_integration
"""
import importlib.util
import json
import sys
import unittest
from pathlib import Path

import awkward as ak
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WF_B, WF_A = "hww_routeb_3fs4fs", "hww_allfix_4fs"
YEAR = "2022postEE"
S4, S3, S3X = "HplusCharm_4FS_HToWWTo2L2Nu", "HplusCharm_3FS_HToWW", "HplusCharm_3FS_HToWW_ext1"


def import_script(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bare_processor(workflow=WF_B, year=YEAR):
    """BaseProcessor without __init__ (no histogram/model setup): only the Route B attributes."""
    from analysis.processors.base import BaseProcessor
    from analysis.stitching.hplusc_routeb import RouteBConfig
    p = BaseProcessor.__new__(BaseProcessor)
    p.year, p._routeb, p._routeb_scheme_cache = year, RouteBConfig.from_workflow(workflow), {}
    return p


def pdf_events(n=400, seed=3, n_members=101, spread=0.015):
    rng = np.random.default_rng(seed)
    w = np.empty((n, n_members))
    w[:, 0] = 1.0
    w[:, 1:] = 1.0 + spread * rng.standard_normal((n, n_members - 1)) + 0.01 * rng.standard_normal((n, 1))
    return ak.Array({"LHEPdfWeight": w.tolist()}), w


def standard_container(events):
    from coffea.analysis_tools import Weights
    from analysis.corrections.lhepdf import add_lhepdf_weight
    wc = Weights(len(events), storeIndividual=True)
    wc.add("genweight", np.where(np.arange(len(events)) % 3 == 0, -1.0, 1.0) * np.linspace(0.5, 2.0, len(events)))
    add_lhepdf_weight(events, wc, shift=None)
    return wc


class TestCoffeaReAdd(unittest.TestCase):
    def test_readd_multiplies_nominal_and_overwrites_modifier(self):
        from coffea.analysis_tools import Weights
        wc = Weights(4, storeIndividual=True)
        wc.add("a", np.array([1.0, 2.0, 3.0, 4.0]))
        wc.add("x", np.ones(4), weightUp=np.full(4, 1.5), weightDown=np.full(4, 0.5))
        nominal, names = wc.weight().copy(), set(wc.variations)
        wc.add("x", np.ones(4), weightUp=np.full(4, 1.02), weightDown=np.full(4, 0.98))
        np.testing.assert_array_equal(wc.weight(), nominal)
        self.assertEqual(set(wc.variations), names)
        np.testing.assert_allclose(wc.weight("xUp"), nominal * 1.02)
        np.testing.assert_allclose(wc.weight("xDown"), nominal * 0.98)


class TestThreeFSPdf(unittest.TestCase):
    def test_3fs_replica_std_replaces_hessian(self):
        ev, w = pdf_events()
        wc = standard_container(ev)
        nominal, names = wc.weight().copy(), set(wc.variations)
        hessian_up = wc.weight("lhe_pdfUp") / nominal
        bare_processor()._routeb_3fs_pdf(ev, S3 + "_7", wc)
        delta = np.minimum(np.std(w[:, 1:101], axis=1, ddof=1), 0.5)
        np.testing.assert_array_equal(wc.weight(), nominal)
        self.assertEqual(set(wc.variations), names)
        for name in ("lhe_pdf", "lhe_pdf_alphaS"):
            np.testing.assert_allclose(wc.weight(name + "Up") / nominal, 1 + delta, rtol=1e-12)
            np.testing.assert_allclose(wc.weight(name + "Down") / nominal, 1 - delta, rtol=1e-12)
        np.testing.assert_allclose(wc.weight("lhe_alphaSUp"), nominal)
        # Hessian sum over 100 replicas ~ sqrt(100) x std: the bug this fixes
        self.assertGreater(np.median(hessian_up - 1) / np.median(delta), 8)

    def test_ext1_partition_also_fixed_and_cap_applied(self):
        ev, w = pdf_events(spread=0.8)
        wc = standard_container(ev)
        bare_processor()._routeb_3fs_pdf(ev, S3X + "_2", wc)
        r = wc.weight("lhe_pdfUp") / wc.weight()
        self.assertLessEqual(r.max(), 1.5 + 1e-12)
        np.testing.assert_allclose(r, 1 + np.minimum(np.std(w[:, 1:], axis=1, ddof=1), 0.5))

    def test_4fs_and_background_untouched(self):
        for ds in (S4 + "_3", S4, "TTto2L2Nu_12"):
            ev, _ = pdf_events()
            wc = standard_container(ev)
            before = {v: wc.weight(v).copy() for v in sorted(wc.variations)}
            bare_processor()._routeb_3fs_pdf(ev, ds, wc)
            for v, arr in before.items():
                np.testing.assert_array_equal(wc.weight(v), arr)

    def test_train_pass_without_pdf_is_noop_and_bad_member_count_raises(self):
        from coffea.analysis_tools import Weights
        ev, _ = pdf_events()
        wc = Weights(len(ev), storeIndividual=True)
        wc.add("genweight", np.ones(len(ev)))
        bare_processor()._routeb_3fs_pdf(ev, S3 + "_1", wc)
        self.assertEqual(set(wc.variations), set())
        ev103, _ = pdf_events(n_members=103)
        wc = standard_container(ev103)
        with self.assertRaisesRegex(ValueError, "101 LHEPdfWeight"):
            bare_processor()._routeb_3fs_pdf(ev103, S3 + "_1", wc)


class TestWiring(unittest.TestCase):
    def test_workflow_block_and_run_a_has_none(self):
        from analysis.stitching.hplusc_routeb import RouteBConfig
        cfg = RouteBConfig.from_workflow(WF_B)
        self.assertEqual(cfg.datasets, {S4: "4FS", S3: "3FS", S3X: "3FS"})
        self.assertEqual((cfg.pt_min, cfg.eta_max, cfg.domain), (10.0, 2.5, "per_era"))
        self.assertIsNone(RouteBConfig.from_workflow(WF_A))

    def test_scheme_resolution_of_condor_partitions(self):
        p = bare_processor()
        self.assertEqual(p._routeb_scheme(S3 + "_12"), ("3FS", S3))
        self.assertEqual(p._routeb_scheme(S3X + "_1"), ("3FS", S3X))
        self.assertEqual(p._routeb_scheme(S3X), ("3FS", S3X))
        self.assertEqual(p._routeb_scheme(S4 + "_4"), ("4FS", S4))
        self.assertEqual(p._routeb_scheme("TTto2L2Nu_3")[0], None)

    def test_workflow_differs_from_run_a_only_where_intended(self):
        from analysis.workflows.config import WorkflowConfigBuilder
        a = WorkflowConfigBuilder(WF_A).build_workflow_config()
        b = WorkflowConfigBuilder(WF_B).build_workflow_config()
        self.assertEqual(b.datasets["mc"], a.datasets["mc"][:a.datasets["mc"].index("hc_central") + 1]
                         + ["hc_3fs"] + a.datasets["mc"][a.datasets["mc"].index("hc_central") + 1:])
        wa, wb = a.corrections_config["event_weights"], b.corrections_config["event_weights"]
        self.assertEqual(wb["hpluscFSShape"], [])
        self.assertEqual({k: v for k, v in wa.items() if k != "hpluscFSShape"},
                         {k: v for k, v in wb.items() if k != "hpluscFSShape"})
        self.assertNotIn(S3, wb["lhe_renorm_datasets"])
        self.assertEqual([s for s in a.combine["shape_systematics"] if s != "xsec_hplusc_3FS_4FS"],
                         b.combine["shape_systematics"])
        self.assertEqual({k: v for k, v in a.combine.items() if k not in ("shape_systematics", "output")},
                         {k: v for k, v in b.combine.items() if k not in ("shape_systematics", "output")})
        self.assertEqual(a.object_selection, b.object_selection)
        self.assertEqual(a.event_selection, b.event_selection)
        self.assertEqual(a.mva, b.mva)

    def test_lhe_yukawa_guard_ignores_3fs(self):
        from analysis.corrections.lhe_yukawa_fix import table_for_dataset
        for ds in (S3, S3 + "_3", S3X, S3X + "_1"):
            self.assertIsNone(table_for_dataset(ds))
        self.assertEqual(table_for_dataset(S4 + "_3"), "hpluscharm_4fs")

    def test_era_filesets(self):
        from analysis.filesets.utils import get_dataset_config, get_datasets_to_run_over
        for era in ("2022preEE", "2022postEE", "2023preBPix", "2023postBPix"):
            cfg = get_dataset_config(era)
            three = [s for s in (S3, S3X) if s in cfg]
            self.assertEqual(three, [S3, S3X] if era == "2022postEE" else [S3])
            for s in three:
                self.assertEqual((cfg[s]["process"], cfg[s]["key"], cfg[s]["era"]), ("H+c", "hc_3fs", "mc"))
                self.assertEqual(float(cfg[s]["xsec"]), float(cfg[S4]["xsec"]))
            run_b, run_a = get_datasets_to_run_over(WF_B, era), get_datasets_to_run_over(WF_A, era)
            self.assertEqual(sorted(set(run_b) - set(run_a)), sorted(three))
            self.assertEqual(set(run_a) - set(run_b), set())
            fs = json.loads((ROOT / f"analysis/filesets/fileset_{era}_nanov12_lxplus.json").read_text())
            for s in three:
                self.assertTrue(fs[s] and all(f.startswith("root://maite.iihe.ac.be:1094//store/user/cgupta/")
                                              for f in fs[s]))

    def test_provenance_hashes_stitching_sources(self):
        import hashlib
        from analysis.processors.base import _routeb_provenance
        prov = json.loads(_routeb_provenance(WF_B))
        self.assertEqual(prov["stitching"]["hplusc_routeb"]["3FS"], [S3, S3X])
        for f in ("__init__.py", "hplusc_routeb.py", "routeb_io.py"):
            self.assertEqual(prov["sha256"][f],
                             hashlib.sha256((ROOT / "analysis/stitching" / f).read_bytes()).hexdigest())


def sink_entry(sample, scheme, chunks, rng, frac):
    """Synthetic accumulated TRAIN/RUN sample entry with one stitch record per chunk."""
    from analysis.stitching.hplusc_routeb import chunk_sums
    from analysis.stitching.routeb_io import record_payload
    recs, sumw = {}, 0.0
    for c in chunks:
        w = rng.choice([-1.0, 1.0], size=500, p=[0.3, 0.7]) * rng.uniform(0.5, 1.5, 500)
        n = (rng.uniform(size=500) < (frac if scheme == "3FS" else 1 - frac)).astype(int)
        s, _ = chunk_sums(w, n, scheme)
        recs[c] = json.dumps(record_payload(s, sample, YEAR, c), sort_keys=True)
        sumw += float(w.sum())
    return {"sumw": sumw, "chunks": set(chunks), "chunks_read": len(chunks), "routeb_stitch": recs}


class TestSinkScaleFactors(unittest.TestCase):
    def setUp(self):
        from analysis.stitching.hplusc_routeb import RouteBConfig
        self.cfg = RouteBConfig.from_workflow(WF_B)
        rng = np.random.default_rng(11)
        self.samples = {S4: sink_entry(S4, "4FS", ["a/0", "a/1", "a/2"], rng, 0.4),
                        S3: sink_entry(S3, "3FS", ["b/0", "b/1"], rng, 0.45),
                        S3X: sink_entry(S3X, "3FS", ["c/0"], rng, 0.45),
                        "TTto2L2Nu": {"sumw": 5.0, "chunks": {"t"}}}
        self.xsec = {S4: 0.0016626, S3: 0.0016626, S3X: 0.0016626, "TTto2L2Nu": 98.0}

    def test_closure_and_shared_F(self):
        from analysis.stitching.routeb_io import sums_from_sink_entry
        L = 26671.7
        out, norm = __import__("analysis.stitching.routeb_io", fromlist=["x"]).sink_scale_factors(
            self.cfg, self.samples, L, self.xsec)
        self.assertEqual(set(out), {S4, S3, S3X})
        F3 = self.samples[S3]["sumw"] + self.samples[S3X]["sumw"]
        self.assertAlmostEqual(out[S3]["F_s"], F3)
        self.assertAlmostEqual(out[S3X]["F_s"], F3)
        self.assertEqual(out[S3]["scale"], out[S3X]["scale"])
        kept = sum(out[s]["scale"] * sums_from_sink_entry(self.samples[s])[s].kept.sumw for s in out)
        self.assertAlmostEqual(kept / (L * 0.0016626), 1.0, places=12)
        self.assertAlmostEqual(norm.closure, 1.0, places=12)

    def test_guards(self):
        from analysis.stitching.routeb_io import sink_scale_factors
        bad = dict(self.samples)
        bad[S3] = dict(bad[S3], chunks=bad[S3]["chunks"] | {"b/9"})
        with self.assertRaisesRegex(RuntimeError, "without record"):
            sink_scale_factors(self.cfg, bad, 1.0, self.xsec)
        bad = dict(self.samples)
        bad[S4] = dict(bad[S4], sumw=bad[S4]["sumw"] * 1.01)
        with self.assertRaisesRegex(RuntimeError, "full sumw"):
            sink_scale_factors(self.cfg, bad, 1.0, self.xsec)
        with self.assertRaisesRegex(RuntimeError, "both schemes"):
            sink_scale_factors(self.cfg, {S4: self.samples[S4]}, 1.0, self.xsec)
        with self.assertRaisesRegex(ValueError, "share one xsec"):
            sink_scale_factors(self.cfg, self.samples, 1.0, dict(self.xsec, **{S3X: 0.002}))
        self.assertEqual(sink_scale_factors(self.cfg, {"TTto2L2Nu": self.samples["TTto2L2Nu"]}, 1.0,
                                            self.xsec), ({}, None))
        # a configured Route B sample of the era (here ext1) absent from the output must not shrink F_s silently
        no_ext = {s: e for s, e in self.samples.items() if s != S3X}
        with self.assertRaisesRegex(RuntimeError, "missing from the output"):
            sink_scale_factors(self.cfg, no_ext, 1.0, self.xsec, era_samples=list(self.samples))
        # ...but an era without ext1 in its fileset is fine
        out, _ = sink_scale_factors(self.cfg, no_ext, 1.0, self.xsec, era_samples=list(no_ext))
        self.assertEqual(set(out), {S4, S3})
        out, _ = sink_scale_factors(self.cfg, self.samples, 1.0, self.xsec, era_samples=list(self.samples))
        self.assertEqual(set(out), {S4, S3, S3X})


class TestCardBuilderRouteB(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "scripts/combine"))
        cls.mod = import_script("build_cards_rb", "scripts/combine/build_cards_from_templates.py")
        from analysis.filesets.utils import get_dataset_config
        from analysis.workflows.config import WorkflowConfigBuilder
        from analysis.stitching.hplusc_routeb import RouteBConfig
        cls.cfg = cls.mod.configure_era(WorkflowConfigBuilder(WF_B).build_workflow_config(), YEAR)
        cls.routeb = RouteBConfig.from_workflow(WF_B)
        cls.datasets = get_dataset_config(YEAR)

    def output(self):
        from analysis.sinks.hww import final_channels
        combine = self.cfg.combine
        cat = next(iter(self.cfg.event_selection["categories"]))
        nb = len(combine["binning"]["edges"]) - 1
        weights = ["nominal"] + [s + d for s in combine["shape_systematics"] for d in ("Up", "Down")]
        rng = np.random.default_rng(5)
        pick = {}
        for cls in combine["classes"]:
            pick[cls] = next((s for s, v in self.datasets.items() if v["era"] == "mc"
                             and v["process"].strip() in combine["process_map"][cls] and s not in (S4, S3, S3X)), None)
        samples = {}
        for sample in [s for c, s in pick.items() if c != "hplusc"] + [S4, S3, S3X]:
            chunks = [sample + "-c0", sample + "-c1"]
            entry = (sink_entry(sample, self.routeb.scheme_of(sample), chunks, rng, 0.45)
                     if self.routeb.scheme_of(sample) else {"sumw": 100., "chunks": set(chunks), "chunks_read": 2})
            rec = lambda: {"counts": {c: np.full(nb, 1.0) for c in final_channels(combine)},
                           "sumw2": {c: np.full(nb, 1.0) for c in final_channels(combine)},
                           "norm_raw": 1., "norm_rw": 1., "negrw": {False}, "kind": {"weight"},
                           "chunks": set(chunks)}
            recs = {v: rec() for v in weights}
            if self.datasets[sample]["process"].strip() in combine["process_map"]["vjets"]:
                recs.update({"CMS_negrw_vjets" + d: rec() for d in ("Up", "Down")})
            entry.update(templates={cat: recs}, evaluated_shifts={})
            samples[sample] = entry
        return {"samples": samples}

    def test_routeb_scale_used_for_hplusc(self):
        from analysis.stitching.routeb_io import sink_scale_factors
        out = self.output()
        lumi = self.mod.load_lumi(YEAR)
        policy = {}
        hists, *_ = self.mod.make_process_hists(out, self.cfg, YEAR, policy=policy, routeb=self.routeb)
        scales, _ = sink_scale_factors(self.routeb, {s: out["samples"][s] for s in (S4, S3, S3X)}, lumi,
                                       {s: 0.0016626 for s in (S4, S3, S3X)})
        ch = next(iter(hists))
        np.testing.assert_allclose(hists[ch]["hplusc"]["nominal"][0], sum(v["scale"] for v in scales.values()))
        self.assertIn("routeb_stitching", policy)
        self.assertAlmostEqual(policy["routeb_stitching"]["closure"], 1.0, places=12)
        json.dumps(policy["routeb_stitching"])
        # without the hook every H+c sample would carry the full NNLO xsec (double counting)
        plain, *_ = self.mod.make_process_hists(out, self.cfg, YEAR)
        expected_plain = sum(lumi * 0.0016626 / out["samples"][s]["sumw"] for s in (S4, S3, S3X))
        np.testing.assert_allclose(plain[ch]["hplusc"]["nominal"][0], expected_plain)
        # backgrounds identical with and without the hook
        for p in self.cfg.combine["classes"]:
            if p != "hplusc":
                np.testing.assert_array_equal(plain[ch][p]["nominal"][0], hists[ch][p]["nominal"][0])


if __name__ == "__main__":
    unittest.main()
