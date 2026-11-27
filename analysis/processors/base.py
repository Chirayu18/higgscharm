import re
import json
import numpy as np
import awkward as ak
from copy import deepcopy
from coffea import processor
from coffea.nanoevents import NanoAODSchema
from coffea.analysis_tools import PackedSelection
from coffea.nanoevents.methods.vector import LorentzVector
from analysis.utils import dump_lumi, update, add_cutflow, dump_parquet
from analysis.utils.parquet_writer import dump_chunk_sumw
from analysis.workflows.config import WorkflowConfigBuilder
from analysis.histograms import HistBuilder, fill_histograms

from analysis.corrections.correction_manager import (
    object_corrector_manager,
    weight_manager,
)
from analysis.selections import (
    ObjectSelector,
    get_lumi_mask,
    get_trigger_mask,
    get_zzto4l_trigger_mask,
    get_metfilters_mask,
    get_trigger_match_mask,
    get_stitching_mask,
)

NanoAODSchema.warn_missing_crossrefs = False


class BaseProcessor(processor.ProcessorABC):
    def __init__(
        self,
        workflow: str,
        year: str,
        output_format: str,
        output_location: str,
        mode: str = "legacy",
        folds: int = 5,
        model_manifest: str = None,
        allow_diagnostic_model: bool = False,
        slim_ntuple: bool = False,
    ):
        self.mode, self.folds, self.slim_ntuple = mode, folds, slim_ntuple
        self.year = year
        self.workflow = workflow
        self.output_format = output_format
        self.output_location = output_location

        config_builder = WorkflowConfigBuilder(workflow)
        self.workflow_config = config_builder.build_workflow_config()
        from analysis.processing_modes import validate
        from analysis.features.hww import signature, selection_signature
        self._models = validate(self.workflow_config, mode, folds, model_manifest, allow_diagnostic_model, year=year)
        self._feature_hash = signature(self.workflow_config) if mode != "legacy" else None
        self._selection_hash = selection_signature(self.workflow_config) if mode != "legacy" else None
        if mode == "train":
            # Freeze signatures before the operational shift toggle: TRAIN/RUN share nominal physics.
            self.workflow_config.corrections_config = deepcopy(self.workflow_config.corrections_config)
            self.workflow_config.corrections_config["object_shifts"] = False
        self.histogram_config = self.workflow_config.histogram_config
        # Route B H+c 3FS/4FS stitching (2026-11-12, from proposal HTOWW-ROUTEB-SOL-20261105). None unless the
        # workflow YAML has a `stitching: hplusc_routeb:` block, so every other workflow (Run A) is untouched.
        from analysis.stitching.hplusc_routeb import RouteBConfig
        self._routeb = RouteBConfig.from_workflow(workflow)
        if self._routeb is not None and self._routeb.defer_selection and mode != "legacy_slim":
            raise ValueError("Deferred signal selection requires legacy_slim output and explicit offline A/B selection")
        self._routeb_provenance = None
        self._routeb_scheme_cache = {}
        if self._routeb is not None:
            if mode == "legacy" and output_format == "coffea":
                # process_shift's metadata["sumw"] would be taken AFTER the stitch mask -> wrong normalisation
                raise NotImplementedError("Route B stitching supports parquet or TRAIN/RUN output, not legacy coffea")
            self._routeb_provenance = _routeb_provenance(workflow)
        self.histograms = HistBuilder(self.workflow_config).build_histogram()
        # lazily-loaded neg-weight reweighting ensemble (see _score_negrw)
        self._negrw_bundle = None
        # sha256 of the negrw model bytes RUN actually scored with (_negrw_model_identity)
        self._negrw_sha256 = None

    def process(self, events):
        self.is_mc = hasattr(events, "genWeight")
        # Route B: compute the stitch record on the FULL pre-veto chunk (same events as dump_chunk_sumw), keep a
        # handle on the full chunk for every sumw below, then drop the events the other scheme supplies.
        events_full, stitch_payload, stitch_keep = events, None, None
        if self.is_mc and self._routeb is not None:
            scheme, sample_name = self._routeb_scheme(events.metadata["dataset"])
            if scheme is not None:
                from analysis.stitching.hplusc_routeb import event_sums
                from analysis.stitching import routeb_io
                _s, stitch_keep, n_cjets = event_sums(events, scheme, self._routeb.pt_min, self._routeb.eta_max)
                stitch_payload = routeb_io.record_payload(
                    _s, sample_name, self.year, events.behavior["__events_factory__"]._partition_key)
                if self._routeb.defer_selection:
                    # GEN properties follow each event through nominal and
                    # shifted selections. Do not discard either scheme region.
                    events["gen_routeb_cjet_multiplicity"] = n_cjets
                    events["routeb_keep"] = stitch_keep
                    stitch_keep = None
                if self.mode == "legacy" and self.output_format == "parquet":
                    routeb_io.write_record(stitch_payload, self.workflow, self.year, self.output_location)
        # Record this chunk's full generator sumw BEFORE any selection/veto, so the
        # parquet normalisation is correct even for chunks that select zero events
        # (which otherwise write no shard and lose their sumw). Pre-veto `events`.
        if self.mode == "legacy" and self.output_format == "parquet" and self.is_mc:
            dump_chunk_sumw(events_full, self.workflow, self.year, self.output_location)
        shifts = None
        if stitch_keep is not None:
            events = events[stitch_keep]
        if stitch_keep is not None and not stitch_keep.any() and self.mode not in ("run", "legacy_slim"):
            # Route B chunk with no kept event: nothing to select. The stats below (TRAIN) still record its full
            # sumw and stitch record. RUN mode falls through so evaluated_shifts stays complete.
            output = {}
        else:
            vetoed_events, shifts = object_corrector_manager(
                events=events,
                year=self.year,
                corrections_config=self.workflow_config.corrections_config,
            )
            output = processor.accumulate(
                self.process_shift(update(vetoed_events, collections), shift)
                for collections, shift in shifts
                if self.mode != "train" or shift is None
            )
        if self.mode != "legacy":
            from analysis.filesets.utils import get_dataset_config
            from analysis.sinks.hww import dataset_info
            sample, info = dataset_info(events.metadata["dataset"], get_dataset_config(self.year))
            # BEFORE veto and selections, once per chunk, including zero-selection chunks.
            # FULL chunk (events_full), never the stitch-masked events: F_s must count every generated event.
            stats = {"sumw": float(ak.sum(events_full.genWeight)) if self.is_mc else float(len(events_full)),
                     "sumw2": float(ak.sum(events_full.genWeight**2)) if self.is_mc else float(len(events_full)),
                     "events_read": len(events_full), "chunks_read": 1,
                     "chunks": {events.behavior["__events_factory__"]._partition_key}}
            if stitch_payload is not None:
                # one JSON STRING per chunk key: coffea's accumulate adds values under equal keys, so a chunk
                # accumulated twice becomes concatenated (invalid) JSON and routeb_io.sums_from_sink_entry raises
                # instead of silently doubling F_s / K_s.
                stats["routeb_stitch"] = {stitch_payload["partition_key"]: json.dumps(stitch_payload, sort_keys=True)}
            entry = output.setdefault("samples", {}).setdefault(sample, {})
            entry.update(stats)
            entry.setdefault("selected_events", 0)
            if self.mode == "legacy_slim":
                counts = entry.setdefault("selected_events_by_shift", {})
                for _, sh in shifts:
                    counts.setdefault(sh or "nominal", 0)
            if self.mode in ("run", "legacy_slim"):
                # Which object shifts were evaluated on this chunk (before any selection),
                # so the card builder can tell a real zero-yield shift from a missing one.
                key = stats["chunks"]
                entry["evaluated_shifts"] = {s: set(key) for _, s in shifts if s is not None}
            output["provenance"] = {"mode": {self.mode}, "year": {self.year},
                                    "features_hash": {self._feature_hash},
                                    "selection_hash": {self._selection_hash}, "folds": {self.folds}}
            if self.mode == "legacy_slim":
                output["provenance"]["legacy_slim_schema_version"] = {2 if self._routeb is not None and self._routeb.defer_selection else 1}
                output["provenance"]["signal_retention"] = {"inclusive_4fs_3fs" if self._routeb is not None and self._routeb.defer_selection else "workflow_selected"}
                output["provenance"]["negrw_policy"] = {"deferred_from_generator_inputs"}
            if self._models:
                output["provenance"]["model_manifest_hash"] = {self._models.manifest_hash}
            if self.mode == "run":
                # G2 (2026-11-12): identity of the negrw model actually consumed ("disabled" without a negrw
                # block). Recorded on every RUN chunk, vjets or not, so a merge can refuse mixed models.
                output["provenance"]["negrw_model_sha256"] = {self._negrw_model_identity()}
            if self._routeb_provenance is not None:
                # stitching block + sha256 of analysis/stitching/*.py (Run A outputs carry no such key)
                output["provenance"]["routeb_stitching"] = {self._routeb_provenance}
        return output

    def _routeb_scheme(self, dataset):
        """(scheme, sample) for a Route B dataset (condor partition names <sample>_<n> resolve via dataset_info),
        (None, sample) otherwise. Only called when the workflow has a stitching block."""
        if dataset not in self._routeb_scheme_cache:
            from analysis.filesets.utils import get_dataset_config
            from analysis.sinks.hww import dataset_info
            sample, _ = dataset_info(dataset, get_dataset_config(self.year))
            self._routeb_scheme_cache[dataset] = (self._routeb.scheme_of(sample), sample)
        return self._routeb_scheme_cache[dataset]

    def _routeb_3fs_pdf(self, events, dataset, weights_container):
        """Route B 3FS only: replace lhe_pdf / lhe_pdf_alphaS with the MC-replica uncertainty.

        The private 3FS re-NANO carries LHA 321900-322000 (NNPDF31_nnlo_pch_as_0118_nf_3, member 0 + 100 MC
        replicas). lhepdf.py always uses the Hessian sum sqrt(sum_k (w_k - w_0)^2), correct for the 4FS Hessian set
        325500 but ~10x too large for replicas. Replica prescription: delta = std_k(w_k) (ddof=1), same per-event
        cap as lhepdf.py. No alphaS members -> lhe_alphaS stays 1 and lhe_pdf_alphaS = lhe_pdf (as lhepdf.py does
        for 101 members). Re-adding a name multiplies the nominal by `weight` (= 1) and overwrites the modifiers.
        """
        if "lhe_pdfUp" not in weights_container.variations:
            return  # not the nominal weight pass (e.g. TRAIN) or no PDF weights configured
        if self._routeb_scheme(dataset)[0] != "3FS":
            return
        from analysis.corrections.lhepdf import PDF_DELTA_CAP
        pdf = np.asarray(ak.to_numpy(ak.to_regular(events.LHEPdfWeight)), dtype=np.float64)
        if pdf.ndim != 2 or pdf.shape[1] != 101:
            raise ValueError(f"{dataset}: Route B 3FS expects 101 LHEPdfWeight members (LHA 321900), got {pdf.shape}")
        delta = np.minimum(np.std(pdf[:, 1:101], axis=1, ddof=1), PDF_DELTA_CAP)
        ones = np.ones(len(delta))
        for name in ("lhe_pdf", "lhe_pdf_alphaS"):
            weights_container.add(name, weight=ones, weightUp=1 + delta, weightDown=1 - delta)

    def process_shift(self, events, shift):
        year = self.year
        dataset = events.metadata["dataset"]
        histograms = deepcopy(self.histograms)

        # initialize output dictionary to store histograms/arrays and metadata
        output = {}
        output["metadata"] = {}
        if shift is None:
            # add sum of genweights (before selection) to metadata
            sumw = ak.sum(events.genWeight) if self.is_mc else len(events)
            output["metadata"].update({"sumw": sumw})

        if not self.is_mc:
            events["Jet", "hadronFlavour"] = ak.zeros_like(events.Jet.pt)

        # --------------------------------------------------------------
        # Object selection
        # --------------------------------------------------------------
        object_selection = self.workflow_config.object_selection
        object_selector = ObjectSelector(object_selection, year)
        objects = object_selector.select_objects(events)

        # --------------------------------------------------------------
        # Event selection
        # --------------------------------------------------------------
        event_selection = self.workflow_config.event_selection
        hlt_paths = event_selection["hlt_paths"]
        if not self.is_mc:
            # save (run, luminosityBlock) pairs to metadata
            lumi_mask = eval(event_selection["selections"]["lumimask"])
            dump_lumi(events[lumi_mask], output)

        selection_manager = PackedSelection()
        for selection, mask in event_selection["selections"].items():
            selection_manager.add(selection, eval(mask))

        add_cutflow(events, output, selection_manager, self.workflow_config)
        # --------------------------------------------------------------
        # Histogram filling / array dumping
        # --------------------------------------------------------------
        categories = event_selection["categories"]
        for category, category_cuts in categories.items():
            # get selection mask by category
            category_mask = selection_manager.all(*category_cuts)
            nevents_after = ak.sum(category_mask)
            if nevents_after > 0:
                # get pruned events
                pruned_ev = events[category_mask]
                # add each selected object to 'pruned_ev' as a new field
                for obj in objects:
                    pruned_ev[f"selected_{obj}"] = objects[obj][category_mask]
                # get weights container
                weights_container = weight_manager(
                    pruned_ev=pruned_ev,
                    year=year,
                    dataset=dataset,
                    workflow_config=self.workflow_config,
                    category=category,
                    shift="train_nominal" if self.mode == "train" else shift,
                )
                if self._routeb is not None and shift is None and self.is_mc:
                    self._routeb_3fs_pdf(pruned_ev, dataset, weights_container)
                # get analysis variables map
                variables_map = {}
                for variable, axis in self.histogram_config.axes.items():
                    variables_map[variable] = eval(axis.expression)[category_mask]
                # NanoAOD event id — used by scripts/mva/prep_training_inputs.py
                # for the deterministic train/test split
                variables_map["event"] = events.event[category_mask]

                # Negative-weight reweighting (arXiv:2510.16217): score the vjets events
                # with the pre-trained P+(x) ensemble on their generator features and dump
                # g = 2*P+-1 (+ ensemble std) as extra columns. Config-gated + dataset-gated
                # so no other workflow/dataset is affected. g(x) is a generator property and
                # the SR events are disjoint-by-construction from the veto_emu_sr training
                # region, so no hold-out is needed. See analysis/processors/negrw.py.
                negrw_cfg = self.workflow_config.negrw
                if self.mode != "legacy_slim" and negrw_cfg and _dataset_matches(dataset, negrw_cfg.get("datasets")):
                    g_central, g_std = self._score_negrw(negrw_cfg, variables_map)
                    variables_map["weight_negrw"] = g_central
                    variables_map["weight_negrw_std"] = g_std

                if self.mode != "legacy":
                    self._mode_sink(output, pruned_ev, weights_container, variables_map, category, shift)
                elif self.output_format == "coffea":
                    fill_histograms(
                        histogram_config=self.histogram_config,
                        weights_container=weights_container,
                        variables_map=variables_map,
                        histograms=histograms,
                        category=category,
                        is_mc=self.is_mc,
                        shift=shift,
                        flow=True,
                    )
                elif self.output_format == "parquet":
                    dump_parquet(
                        events=events,
                        weights_container=weights_container,
                        variables_map=variables_map,
                        workflow=self.workflow,
                        year=year,
                        category=category,
                        output_location=self.output_location,
                        shift=shift,
                    )

        # add histograms to output dictionary
        if self.mode == "legacy" and self.output_format == "coffea":
            output["histograms"] = histograms
        return output

    def _mode_sink(self, output, events, weights, variables, category, shift):
        from analysis.features.hww import compute, matrix, event_folds
        from analysis.sinks.hww import dataset_info, labels, make_templates, channel_layout, split_variables
        from analysis.filesets.utils import get_dataset_config
        from analysis.utils.parquet_writer import dump_pa_table
        from analysis.histograms.hist_filler import fill_histogram
        cols = compute(variables)
        event = ak.to_numpy(events.event)
        sample, info = dataset_info(events.metadata["dataset"], get_dataset_config(self.year))
        if shift is None:
            output.setdefault("samples", {}).setdefault(sample, {})["selected_events"] = len(events)
        if self.mode == "legacy_slim":
            output.setdefault("samples", {}).setdefault(sample, {}).setdefault("selected_events_by_shift", {})[shift or "nominal"] = len(events)
        nominal = np.asarray(weights.weight(), dtype=np.float64)
        if not np.all(np.isfinite(nominal)):
            raise ValueError(f"{sample}: nonfinite nominal weight in {self.mode} mode")
        if self.mode == "run":
            scores = self._models.score(matrix(cols), event)
            weight_map = ({"nominal": nominal} if shift is None else {shift: nominal})
            if self.is_mc and shift is None:
                weight_map.update({v: np.asarray(weights.weight(v), dtype=np.float64)
                                   for v in sorted(weights.variations)})
            combine = self.workflow_config.combine
            g, std = variables.get("weight_negrw"), variables.get("weight_negrw_std")
            records = make_templates(scores, combine["classes"], combine["channels"],
                                     combine["binning"]["edges"], weight_map, g,
                                     std if shift is None else None,
                                     combine.get("negrw_shape_name", "CMS_negrw_vjets")
                                     if combine.get("negrw_uncertainty", True) else None,
                                     nominal_weight=nominal,
                                     layout=channel_layout(combine),
                                     split_values={v: np.asarray(ak.to_numpy(variables[v]), dtype=np.float64)
                                                   for v in split_variables(combine)})
            chunk = events.behavior["__events_factory__"]._partition_key
            for record in records.values():
                record["kind"] = {"weight" if shift is None else "object"}
                # Chunk coverage per variation: a weight variation must cover exactly
                # the nominal's chunks (checked when building cards).
                record["chunks"] = {chunk}
            entry = output.setdefault("samples", {}).setdefault(sample, {})
            entry.setdefault("templates", {})[category] = records
            # Control histograms are nominal-only; templates contain every variation.
            if shift is None:
                controls = deepcopy(self.histograms)
                fill_histogram(controls, self.histogram_config, variables, category,
                               nominal, "nominal", flow=True)
                entry["controls"] = controls
        if self.mode in ("train", "legacy_slim") or self.slim_ntuple:
            arrays = dict(cols)
            arrays.update(event=event, run=ak.to_numpy(events.run),
                          luminosityBlock=ak.to_numpy(events.luminosityBlock),
                          fold=event_folds(event, self.folds), weight_nominal=nominal,
                          process=[info["process"]] * len(events), sample=[sample] * len(events))
            arrays.update(labels(info["process"].strip(),
                                 self.workflow_config.mva["labels"]["process_groups"], len(events)))
            if self.mode in ("run", "legacy_slim") and shift is None and self.is_mc:
                arrays.update({"weight_" + v: weights.weight(v) for v in sorted(weights.variations)})
            if self.mode == "legacy_slim":
                if self._routeb is not None and self._routeb.defer_selection and self._routeb.scheme_of(sample):
                    arrays["gen_routeb_cjet_multiplicity"] = np.asarray(ak.to_numpy(events.gen_routeb_cjet_multiplicity), dtype=np.int64)
                    arrays["routeb_keep"] = np.asarray(ak.to_numpy(events.routeb_keep), dtype=np.bool_)
                for v in ("weight_nominal", *[k for k in arrays if k.startswith("weight_")]):
                    if not np.all(np.isfinite(arrays[v])):
                        raise ValueError(f"{sample}: nonfinite slim weight {v}")
            # Deferred negrw needs generator inputs on EVERY shift: an event
            # entering a shifted selection may be absent from nominal.
            negrw = self.workflow_config.negrw
            deferred = self.mode == "legacy_slim" and negrw and _dataset_matches(sample, negrw.get("datasets"))
            if "weight_negrw" in variables or deferred:
                arrays.update({k: v for k, v in variables.items()
                               if k.startswith(("gen", "lhe_", "weight_negrw"))})
            fname = events.behavior["__events_factory__"]._partition_key.replace("/", "_") + ".parquet"
            subdirs = [self.workflow + "__" + self.mode, self.year, events.metadata["dataset"], category]
            if shift is not None:
                subdirs.append(shift)
            extra_metadata = {"processing_mode": self.mode, "features_hash": self._feature_hash,
                              "selection_hash": self._selection_hash, "folds": str(self.folds),
                              "sample": sample, "year": self.year,
                              "xsec": str(info.get("xsec", "")),
                              "source_file": events.metadata["filename"]}
            if self.mode == "legacy_slim":
                extra_metadata.update(legacy_slim_schema_version="2" if self._routeb is not None and self._routeb.defer_selection else "1",
                                      signal_retention="inclusive_4fs_3fs" if self._routeb is not None and self._routeb.defer_selection else "workflow_selected",
                                      shift=shift or "nominal",
                                      chunk=events.behavior["__events_factory__"]._partition_key,
                                      weight_variations=json.dumps(sorted(weights.variations) if shift is None else []),
                                      negrw_policy="deferred_from_generator_inputs")
            if self._routeb_provenance is not None:
                extra_metadata["routeb_stitching"] = self._routeb_provenance
            if self.mode == "run":
                extra_metadata["negrw_model_sha256"] = self._negrw_model_identity()
            dump_pa_table(arrays, fname, self.output_location, subdirs, extra_metadata=extra_metadata)

    def _score_negrw(self, negrw_cfg, variables_map):
        """Score the pre-trained P+(x) ensemble on the pruned events' generator features.

        Returns (g_central, g_std) as numpy arrays aligned with variables_map rows:
          g_central = 2*mean_m P+_m(x) - 1   (the per-event reweight factor)
          g_std     = 2*std_m  P+_m(x)       (ensemble spread -> shape systematic)

        Feature order + NaN handling mirror negweight_reweight_train.py exactly: the
        column order is the persisted `features` list, missing parton kinematics stay
        NaN (HistGradientBoosting handles NaN natively).
        """
        import joblib

        if self._negrw_bundle is None:
            if self.mode == "run":
                self._negrw_model_identity()  # loads the bundle from the hashed bytes
            else:
                self._negrw_bundle = joblib.load(negrw_cfg["model"])
        bundle = self._negrw_bundle
        models = bundle["models"]
        features = bundle["features"]

        # build the (n_events, n_features) matrix in the trained feature order.
        cols = []
        for feat in features:
            arr = variables_map[feat]
            # jagged/option axes (e.g. genparton1.pt) -> firsts; fill missing with NaN
            if getattr(arr, "ndim", 1) == 2:
                arr = ak.firsts(arr)
            arr = ak.fill_none(arr, np.nan)
            cols.append(np.asarray(ak.to_numpy(arr), dtype=np.float32))
        X = np.stack(cols, axis=1)

        # ensemble P+ over the members -> g and its spread
        P = np.stack([m.predict_proba(X)[:, 1] for m in models], axis=0)
        g_central = 2.0 * P.mean(axis=0) - 1.0
        g_std = 2.0 * P.std(axis=0)
        return g_central, g_std

    def _negrw_model_identity(self):
        """RUN: sha256 of the negrw model bytes this processor scores with; "disabled" without a negrw block.

        The hash and joblib.load use the SAME in-memory bytes, so the recorded identity is the model actually
        consumed even if the file at negrw.model is replaced while jobs run (G2, 2026-11-12). TRAIN and legacy
        modes keep loading by path and do not record it (weight-only input, not in the TRAIN hashes).
        """
        negrw_cfg = self.workflow_config.negrw
        if not negrw_cfg:
            return "disabled"
        if self._negrw_sha256 is None:
            import io, hashlib, joblib
            with open(negrw_cfg["model"], "rb") as fh:
                data = fh.read()
            bundle = joblib.load(io.BytesIO(data))
            self._negrw_bundle, self._negrw_sha256 = bundle, hashlib.sha256(data).hexdigest()
        return self._negrw_sha256

    def postprocess(self, accumulator):
        pass


def _routeb_provenance(workflow):
    """Canonical JSON: the workflow's `stitching` block + sha256 of every analysis/stitching/*.py."""
    import json, hashlib, importlib.resources
    from pathlib import Path
    import yaml
    with importlib.resources.open_text("analysis.workflows", f"{workflow}.yaml") as fh:
        block = (yaml.safe_load(fh) or {}).get("stitching")
    src = Path(__file__).resolve().parents[1] / "stitching"
    sha = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(src.glob("*.py"))}
    return json.dumps({"stitching": block, "sha256": sha}, sort_keys=True)


def _dataset_matches(dataset, names):
    """True if `dataset` is one of `names`, ignoring condor's `_<jobid>` partition
    suffix (falsy `names` -> match all).

    condor/submit.sh runs each partition as `--dataset <name>_$JOBID`, so the
    processor sees e.g. "DYto2L_2Jets_50_7" for sample "DYto2L_2Jets_50".

    Anchored, NOT substring: the reweighting is only valid for the vjets samples the
    P+(x) ensemble was trained on. A substring gate like "WtoLNu" would also catch
    the WH signal sample `WplusH_WtoLNu_Hto2Wto2L2Nu` and silently reweight a Higgs
    template with a V+jets generator model.
    """
    if not names:
        return True
    # strip a trailing "_<digits>" partition suffix, if present
    base = re.sub(r"_\d+$", "", dataset)
    return dataset in set(names) or base in set(names)
