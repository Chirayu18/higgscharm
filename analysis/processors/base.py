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

    def postprocess(self, accumulator):
        pass
