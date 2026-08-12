import re
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
    ):
        self.year = year
        self.workflow = workflow
        self.output_format = output_format
        self.output_location = output_location

        config_builder = WorkflowConfigBuilder(workflow)
        self.workflow_config = config_builder.build_workflow_config()
        self.histogram_config = self.workflow_config.histogram_config
        self.histograms = HistBuilder(self.workflow_config).build_histogram()
        # lazily-loaded neg-weight reweighting ensemble (see _score_negrw)
        self._negrw_bundle = None

    def process(self, events):
        self.is_mc = hasattr(events, "genWeight")
        # Record this chunk's full generator sumw BEFORE any selection/veto, so the
        # parquet normalisation is correct even for chunks that select zero events
        # (which otherwise write no shard and lose their sumw). Pre-veto `events`.
        if self.output_format == "parquet" and self.is_mc:
            dump_chunk_sumw(events, self.workflow, self.year, self.output_location)
        vetoed_events, shifts = object_corrector_manager(
            events=events,
            year=self.year,
            corrections_config=self.workflow_config.corrections_config,
        )
        return processor.accumulate(
            self.process_shift(update(vetoed_events, collections), shift)
            for collections, shift in shifts
        )

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
                    shift=shift,
                )
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
                if negrw_cfg and _dataset_matches(dataset, negrw_cfg.get("datasets")):
                    g_central, g_std = self._score_negrw(negrw_cfg, variables_map)
                    variables_map["weight_negrw"] = g_central
                    variables_map["weight_negrw_std"] = g_std

                if self.output_format == "coffea":
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
        if self.output_format == "coffea":
            output["histograms"] = histograms
        return output

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

    def postprocess(self, accumulator):
        pass


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
