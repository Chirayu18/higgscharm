import yaml
import numpy as np
import awkward as ak
import importlib.resources
from analysis.filesets.utils import get_dataset_name


def get_hltpaths_from_flag(flag, year):
    if year.startswith("2022"):
        year_key = 2022
    elif year.startswith("2023"):
        year_key = 2023
    with importlib.resources.open_text(
        f"analysis.selections", f"trigger_flags.yaml"
    ) as file:
        hlt_paths = yaml.safe_load(file)
    return hlt_paths[year_key][flag]


def trigger_from_flag(events, flag, year):
    hlt_paths = get_hltpaths_from_flag(flag, year)
    trigger_mask = np.zeros(len(events), dtype="bool")
    for hlt_path in hlt_paths:
        trigger_mask = trigger_mask | events.HLT[hlt_path]
    return trigger_mask


def zzto4l_trigger(events, hlt_paths, dataset, year):
    nevents = len(events)
    dataset_name = get_dataset_name(dataset)
    # compute all trigger masks based on the flags in hlt_paths
    trigger_flags = {}
    for flag in hlt_paths:
        trigger_flags[flag] = trigger_from_flag(events, flag, year)

    if hasattr(events, "genWeight"):
        # compute the combined OR of all flags (for MC datasets)
        mc_trigger_mask = np.zeros(nevents, dtype="bool")
        for flag in trigger_flags:
            mc_trigger_mask = mc_trigger_mask | trigger_flags[flag]
        return mc_trigger_mask

    trigger_flags["TriEle"] = np.zeros(nevents, dtype="bool")

    # ensure each event is taken only from a single PD
    pd_trigger_mask = (
        (np.zeros(nevents, dtype="bool"))
        | (
            ((dataset_name == "EGamma") * np.ones(nevents, dtype=bool))
            & (trigger_flags["DiEle"] | trigger_flags["TriEle"])
        )
        | (
            (
                ((dataset_name == "Muon") or (dataset_name == "DoubleMuon"))
                * np.ones(nevents, dtype=bool)
            )
            & (trigger_flags["DiMu"] | trigger_flags["TriMu"])
            & ~trigger_flags["DiEle"]
            & ~trigger_flags["TriEle"]
        )
        | (
            ((dataset_name == "MuonEG") * np.ones(nevents, dtype=bool))
            & trigger_flags["MuEle"]
            & ~trigger_flags["DiMu"]
            & ~trigger_flags["TriMu"]
            & ~trigger_flags["DiEle"]
            & ~trigger_flags["TriEle"]
        )
        | (
            ((dataset_name == "EGamma") * np.ones(nevents, dtype=bool))
            & trigger_flags["SingleEle"]
            & ~trigger_flags["MuEle"]
            & ~trigger_flags["DiMu"]
            & ~trigger_flags["TriMu"]
            & ~trigger_flags["DiEle"]
            & ~trigger_flags["TriEle"]
        )
        | (
            (
                ((dataset_name == "Muon") or (dataset_name == "SingleMuon"))
                * np.ones(nevents, dtype=bool)
            )
            & trigger_flags["SingleMu"]
            & ~trigger_flags["SingleEle"]
            & ~trigger_flags["MuEle"]
            & ~trigger_flags["DiMu"]
            & ~trigger_flags["TriMu"]
            & ~trigger_flags["DiEle"]
            & ~trigger_flags["TriEle"]
        )
    )
    return pd_trigger_mask


def trigger_mask(events, hlt_paths, dataset, year):
    dataset_name = get_dataset_name(dataset)
    # compute all trigger masks based on the flags in hlt_paths
    trigger_flags = {}
    for dataset_flags in hlt_paths.values():
        for flag in dataset_flags:
            trigger_flags[flag] = trigger_from_flag(events, flag, year)

    # compute the combined OR of all flags (for background)
    all_combined_mask = np.zeros(len(events), dtype="bool")
    for flag in trigger_flags:
        all_combined_mask = all_combined_mask | trigger_flags[flag]

    dataset_masks = {}
    for dataset, flags in hlt_paths.items():
        # OR of all flags for this dataset
        mask = np.zeros(len(events), dtype="bool")
        for flag in flags:
            mask = mask | trigger_flags[flag]

        if dataset_masks:
            # exclude other datasets flags
            other_masks = np.zeros(len(events), dtype="bool")
            for other_dataset in dataset_masks:
                for flag in hlt_paths[other_dataset]:
                    other_masks = other_masks | trigger_flags[flag]
            mask = mask & ~other_masks
        dataset_masks[dataset] = mask

    return dataset_masks.get(dataset_name, all_combined_mask)


def trigger_match(leptons, trigobjs, hlt_path):
    """
    Returns DeltaR matched trigger objects

    leptons:
        Leptons array
    trigobjs:
        trigobjs array
    hlt_path:
        trigger to match (IsoMu24, Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8, Ele30_WPTight_Gsf)

    how to:
    https://twiki.cern.ch/twiki/bin/viewauth/CMS/EgammaNanoAOD#Trigger_bits_how_to

    NanoAOD docs:
    https://cms-nanoaod-integration.web.cern.ch/autoDoc/NanoAODv11/2022postEE/doc_WZ_TuneCP5_13p6TeV_pythia8_Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v1.html#TrigObj
    """
    match_configs = {
        # filterbit: 3 => 1mu
        # id: 13 => mu
        "IsoMu24": {
            "pt": trigobjs.pt > 23,
            "id": abs(trigobjs.id) == 13,
            "filterbit": trigobjs.filterBits & (0x1 << 3) > 0,
        },
        # filterbit: 0 => TrkIsoVVL
        # id: 13 => mu
        "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8": {
            "pt": trigobjs.pt > 7,
            "id": abs(trigobjs.id) == 13,
            "filterbit": trigobjs.filterBits & (0x1 << 0) > 0,
        },
        # filterbit: 0 => TrkIsoVVL
        # id: 13 => mu
        "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ": {
            "pt": trigobjs.pt > 7,
            "id": abs(trigobjs.id) == 13,
            "filterbit": trigobjs.filterBits & (0x1 << 0) > 0,
        },
        # filterbit: 0 => TrkIsoVVL
        # id: 13 => mu
        "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL": {
            "pt": trigobjs.pt > 7,
            "id": abs(trigobjs.id) == 13,
            "filterbit": trigobjs.filterBits & (0x1 << 0) > 0,
        },
        # filterbit: 1 => 1e (WPTight)
        # id: 11 => ele
        "Ele30_WPTight_Gsf": {
            "pt": trigobjs.pt > 28,
            "id": abs(trigobjs.id) == 11,
            "filterbit": trigobjs.filterBits & (0x1 << 1) > 0,
        },
    }
    pass_pt = match_configs[hlt_path]["pt"]
    pass_id = match_configs[hlt_path]["id"]
    pass_filterbit = match_configs[hlt_path]["filterbit"]
    trigger_cands = trigobjs[pass_pt & pass_id & pass_filterbit]
    delta_r = leptons.metric_table(trigger_cands)
    pass_delta_r = delta_r < 0.1
    n_of_trigger_matches = ak.sum(pass_delta_r, axis=2)
    trig_matched_locs = n_of_trigger_matches >= 1
    return trig_matched_locs


def trigger_match_mask(events, hlt_paths, year, leptons):
    trigger_match_mask = np.zeros(len(events), dtype="bool")
    for dataset_flags in hlt_paths.values():
        for flag in dataset_flags:
            for hlt_path in get_hltpaths_from_flag(flag, year):
                trig_obj_mask = trigger_match(
                    leptons=leptons,
                    trigobjs=events.TrigObj,
                    hlt_path=hlt_path,
                )
                trigger_match_mask = trigger_match_mask | trig_obj_mask
    return trigger_match_mask
