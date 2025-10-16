import numpy as np
import awkward as ak
from copy import deepcopy
from coffea import processor
from coffea.nanoevents import NanoAODSchema
from coffea.analysis_tools import PackedSelection
from coffea.nanoevents.methods.vector import LorentzVector
from analysis.utils import dump_lumi, dump_pa_table
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
        year = self.year
        dataset = events.metadata["dataset"]

        object_selections = self.workflow_config.object_selection
        event_selection = self.workflow_config.event_selection
        hlt_paths = event_selection["hlt_paths"]
        histograms = deepcopy(self.histograms)

        # check if dataset is MC or Data
        is_mc = hasattr(events, "genWeight")
        if not is_mc:
            events["Jet", "hadronFlavour"] = ak.zeros_like(events.Jet.pt)

        # initialize output dictionary
        output = {}

        # initialize metadata info with sumw before selection
        output["metadata"] = {}
        sumw = ak.sum(events.genWeight) if is_mc else len(events)
        output["metadata"].update({"sumw": sumw})

        # --------------------------------------------------------------
        # Object corrections
        # --------------------------------------------------------------
        object_corrector_manager(
            events=events,
            year=year,
            dataset=dataset,
            workflow_config=self.workflow_config,
        )

        # --------------------------------------------------------------
        # Object selection
        # --------------------------------------------------------------
        object_selector = ObjectSelector(object_selections, year)
        objects = object_selector.select_objects(events)

        # --------------------------------------------------------------
        # Event selection
        # --------------------------------------------------------------
        if not is_mc:
            # save (run, luminosityBlock) pairs to metadata
            lumi_mask = eval(event_selection["selections"]["lumimask"])
            dump_lumi(events[lumi_mask], output)

        # initialize selection manager
        selection_manager = PackedSelection()
        # add all selections to selector manager
        for selection, mask in event_selection["selections"].items():
            selection_manager.add(selection, eval(mask))

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
                )
                # save cutflow to metadata
                output["metadata"][category] = {"cutflow": {"initial": sumw}}
                selections = []
                for cut_name in category_cuts:
                    selections.append(cut_name)
                    current_selection = selection_manager.all(*selections)
                    pruned_ev_cutflow = events[current_selection]
                    for obj in objects:
                        pruned_ev_cutflow[f"selected_{obj}"] = objects[obj][
                            current_selection
                        ]
                    weights_container_cutflow = weight_manager(
                        pruned_ev=pruned_ev_cutflow,
                        year=year,
                        dataset=dataset,
                        workflow_config=self.workflow_config,
                    )
                    output["metadata"][category]["cutflow"][cut_name] = ak.sum(
                        weights_container_cutflow.weight()
                    )

                # save number of events after selection to metadata
                weighted_final_nevents = ak.sum(weights_container.weight())
                output["metadata"][category].update(
                    {
                        "weighted_final_nevents": weighted_final_nevents,
                        "raw_final_nevents": nevents_after,
                    }
                )
                # get analysis variables map
                variables_map = {}
                for variable, axis in self.histogram_config.axes.items():
                    variables_map[variable] = eval(axis.expression)[category_mask]

                if self.output_format == "coffea":
                    fill_histograms(
                        histogram_config=self.histogram_config,
                        weights_container=weights_container,
                        variables_map=variables_map,
                        histograms=histograms,
                        variation="nominal",
                        category=category,
                        is_mc=is_mc,
                        flow=True,
                    )
                elif self.output_format == "parquet":
                    # add weights to variables map
                    if is_mc:
                        variations = ["nominal"] + list(weights_container.variations)
                        for variation in variations:
                            if variation == "nominal":
                                variables_map[f"weight_nominal"] = (
                                    weights_container.weight()
                                )
                                for (
                                    partial_weight
                                ) in weights_container.weightStatistics:
                                    variables_map[f"weight_{partial_weight}"] = (
                                        weights_container.partial_weight(
                                            include=[partial_weight]
                                        )
                                    )
                            else:
                                variables_map[f"weight_{variation}"] = (
                                    weights_container.weight(modifier=variation)
                                )
                    # save parquet files
                    fname = (
                        events.behavior["__events_factory__"]._partition_key.replace(
                            "/", "_"
                        )
                        + ".parquet"
                    )
                    subdirs = [self.workflow, self.year, dataset, category]
                    dump_pa_table(variables_map, fname, self.output_location, subdirs)

        # add histograms to output dictionary
        if self.output_format == "coffea":
            output["histograms"] = histograms
        return output

    def postprocess(self, accumulator):
        pass
