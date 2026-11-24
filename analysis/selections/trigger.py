import yaml
import numpy as np
import awkward as ak
import importlib.resources
from analysis.filesets.utils import get_dataset_name


# 2026-11-05 (HTOWW-FRAMEWORK-SPEEDUP-2): trigger_flags.yaml was re-parsed on every
# get_hltpaths_from_flag call (45 x per chunk). Parse it once per process; callers get a
# copy of the path list, so the cached data cannot be mutated.
_TRIGGER_FLAGS = None


def _trigger_flags():
    global _TRIGGER_FLAGS
    if _TRIGGER_FLAGS is None:
        with importlib.resources.open_text(
            f"analysis.selections", f"trigger_flags.yaml"
        ) as file:
            _TRIGGER_FLAGS = yaml.safe_load(file)
    return _TRIGGER_FLAGS


def get_hltpaths_from_flag(flag, year):
    hlt_paths = _trigger_flags()
    trigger_flags = hlt_paths[int(year[:4])]
    if flag in trigger_flags:
        paths = trigger_flags[flag]
        return list(paths) if isinstance(paths, list) else paths
    else:
        print(f"Trigger flag {flag} is not defined for year {year}")


def trigger_from_flag(events, flag, year):
    trigger_mask = np.zeros(len(events), dtype="bool")
    hlt_paths = get_hltpaths_from_flag(flag, year)
    if hlt_paths:
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

    # FIX 2026-11-04 (P0-3): Run2022C has the SingleMuon and DoubleMuon primary datasets (the "Muon" PD only exists
    # from 2022D). Both used to miss `dataset_masks` and fall through to the OR of ALL triggers, so every
    # MuonEG/EGamma event was counted again and SingleMuon/DoubleMuon duplicated each other.
    #   SingleMuon -> same logic as "Muon" (SingleMu & ~MuEle);
    #   DoubleMuon -> no analysis trigger is a DoubleMuon-PD path, so it contributes nothing (zero mask).
    # Unknown data PDs now raise instead of silently using the OR of everything.
    is_data = not hasattr(events, "genWeight")
    if is_data:
        if dataset_name == "SingleMuon":
            dataset_name = "Muon"
        elif dataset_name == "DoubleMuon":
            return np.zeros(len(events), dtype="bool")
        elif dataset_name not in dataset_masks:
            raise ValueError(
                f"trigger_mask: no trigger logic for data primary dataset '{dataset_name}'; "
                f"known: {sorted(dataset_masks)}. Add it to hlt_paths / get_dataset_name instead of "
                "falling back to the OR of all triggers (double counting)."
            )
    return dataset_masks.get(dataset_name, all_combined_mask)


def trigger_match(leptons: ak.Array, trigobjs: ak.Array, hlt_path: str, year: str):
    """
    Returns DeltaR matched trigger objects

    leptons:
        electrons or muons arrays
    trigobjs:
        trigger objects array
    hlt_path:
        trigger to match
    year:
        dataset year {2016preVFP, 2016postVFP, 2017, 2018, 2022preEE, 2022postEE, 2023preBPix, 2023postBPix}

    https://twiki.cern.ch/twiki/bin/viewauth/CMS/EgammaNanoAOD#Trigger_bits_how_to
    """
    match_configs = {
        "run2": {
            "IsoMu24": {
                "pt": trigobjs.pt > 22,
                "id": abs(trigobjs.id) == 13,
                "filterbit": (trigobjs.filterBits & 8) > 0,
            },
            "IsoMu27": {
                "pt": trigobjs.pt > 25,
                "filterbit": (trigobjs.filterBits & 8) > 0,
                "id": abs(trigobjs.id) == 13,
            },
            "Mu50": {
                "pt": trigobjs.pt > 45,
                "filterbit": (trigobjs.filterBits & 1024) > 0,
                "id": abs(trigobjs.id) == 13,
            },
            "OldMu100": {
                "pt": trigobjs.pt > 95,
                "filterbit": (trigobjs.filterBits & 2048) > 0,
                "id": abs(trigobjs.id) == 13,
            },
            "TkMu100": {
                "pt": trigobjs.pt > 95,
                "filterbit": (trigobjs.filterBits & 2048) > 0,
                "id": abs(trigobjs.id) == 13,
            },
            "Ele35_WPTight_Gsf": {
                "pt": trigobjs.pt > 33,
                "filterbit": (trigobjs.filterBits & 2) > 0,
                "id": abs(trigobjs.id) == 11,
            },
            "Ele32_WPTight_Gsf": {
                "pt": trigobjs.pt > 30,
                "filterbit": (trigobjs.filterBits & 2) > 0,
                "id": abs(trigobjs.id) == 11,
            },
            "Ele27_WPTight_Gsf": {
                "pt": trigobjs.pt > 25,
                "filterbit": (trigobjs.filterBits & 2) > 0,
                "id": abs(trigobjs.id) == 11,
            },
            "Photon175": {
                "pt": trigobjs.pt > 25,
                "filterbit": (trigobjs.filterBits & 8192) > 0,
                "id": abs(trigobjs.id) == 11,
            },
            "Photon200": {
                "pt": trigobjs.pt > 25,
                "filterbit": (trigobjs.filterBits & 8192) > 0,
                "id": abs(trigobjs.id) == 11,
            },
            "IsoTkMu24": {
                "pt": trigobjs.pt > 22,
                "filterbit": (trigobjs.filterBits & 8) > 0,
                "id": abs(trigobjs.id) == 13,
            },
        },
        "run3": {
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
            # filterbit: 1 => 1e (WPTight)
            # id: 11 => ele
            "Ele30_WPTight_Gsf": {
                "pt": trigobjs.pt > 28,
                "id": abs(trigobjs.id) == 11,
                "filterbit": trigobjs.filterBits & (0x1 << 1) > 0,
            },
        },
    }
    run_key = "run2" if year.startswith("201") else "run3"
    pass_pt = match_configs[run_key][hlt_path]["pt"]
    pass_id = match_configs[run_key][hlt_path]["id"]
    pass_filterbit = match_configs[run_key][hlt_path]["filterbit"]
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
                    year=year,
                )
                trigger_match_mask = trigger_match_mask | trig_obj_mask
    return trigger_match_mask
