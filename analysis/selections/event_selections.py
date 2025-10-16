import yaml
import json
import numpy as np
import awkward as ak
import importlib.resources
from coffea.lumi_tools import LumiMask
from analysis.selections.trigger import trigger_mask, trigger_match_mask, zzto4l_trigger


def get_lumi_mask(events, year):
    if year.startswith("2022"):
        goldenjson = "analysis/data/Cert_Collisions2022_355100_362760_Golden.txt"
    elif year.startswith("2023"):
        goldenjson = "analysis/data/Cert_Collisions2023_366442_370790_Golden.txt"
    if hasattr(events, "genWeight"):
        lumi_mask = np.ones(len(events), dtype="bool")
    else:
        lumi_info = LumiMask(goldenjson)
        lumi_mask = lumi_info(events.run, events.luminosityBlock)
    return lumi_mask == 1


def get_zzto4l_trigger_mask(events, hlt_paths, dataset_key, year):
    return zzto4l_trigger(events, hlt_paths, dataset_key, year)


def get_trigger_mask(events, hlt_paths, dataset_key, year):
    return trigger_mask(events, hlt_paths, dataset_key, year)


def get_trigger_match_mask(events, hlt_paths, year, leptons):
    mask = trigger_match_mask(events, hlt_paths, year, leptons)
    return ak.sum(mask, axis=-1) > 0


def trigger_match_lepton(events, hlt_paths, year):
    mu_mask = get_trigger_match_mask(events, hlt_paths, year, events.Muon)
    ele_mask = get_trigger_match_mask(events, hlt_paths, year, events.Electron)
    return mu_mask | ele_mask


def get_metfilters_mask(events, year):
    with importlib.resources.path("analysis.data", "metfilters.json") as path:
        with open(path, "r") as handle:
            metfilters = json.load(handle)[year]
    metfilters_mask = np.ones(len(events), dtype="bool")
    metfilterkey = "mc" if hasattr(events, "genWeight") else "data"
    for mf in metfilters[metfilterkey]:
        if mf in events.Flag.fields:
            metfilters_mask = metfilters_mask & events.Flag[mf]
    return metfilters_mask


def get_stitching_mask(events, dataset, dataset_key, ht_value):
    stitching_mask = np.ones(len(events), dtype="bool")
    if dataset.startswith(dataset_key):
        stitching_mask = events.LHE.HT < ht_value
    return stitching_mask