import re
import awkward as ak
from pathlib import Path

# summary of pog scale factors: https://cms-nanoaod-integration.web.cern.ch/commonJSONSFs/
# CorrectionLib files are available from
POG_CORRECTION_PATH = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration"
POG_JSONS = {
    "pileup": ["LUM", "puWeights.json.gz"],
    "muon": ["MUO", "muon_Z.json.gz"],
    "electron_id": ["EGM", "electron.json.gz"],
    "electron_hlt": ["EGM", "electronHlt.json.gz"],
    "electron_ss": ["EGM", "electronSS.json.gz"],
    "jetvetomaps": ["JME", "jetvetomaps.json.gz"],
    "jec": ["JME", "jet_jerc.json.gz"],
    "ctag": ["BTV", "ctagging.json.gz"],
}
POG_YEARS = {
    "2022preEE": "2022_Summer22",
    "2022postEE": "2022_Summer22EE",
    "2023preBPix": "2023_Summer23",
    "2023postBPix": "2023_Summer23BPix",
}
# summary of corrections: https://twiki.cern.ch/twiki/bin/view/CMS/EgammSFandSSRun3#JSONs
EGAMMA_CORRECTION_PATH = "/eos/cms/store/group/phys_egamma/ScaleFactors"
EGAMMA_YEARS = {
    "2022preEE": "Data2022/ForRe-recoBCD",
    "2022postEE": "Data2022/ForRe-recoE+PromptFG",
    "2023preBPix": "Data2023/ForPrompt23C",
    "2023postBPix": "Data2023/ForPrompt23D",
}
EGAMMA_JSONS = {
    "electron_ss": ["SS", "electronSS.json.gz"]
}
    

def get_pog_json(json_name: str, year: str) -> str:
    """
    returns pog json file path

    Parameters:
    -----------
        json_name:
            pog json name
        year:
            dataset year {2022preEE, 2022postEE, 2023preBPix, 2023postBPix}
    """
    if json_name in POG_JSONS:
        pog_json = POG_JSONS[json_name]
    else:
        print(f"No json for {json_name}")
    return f"{POG_CORRECTION_PATH}/POG/{pog_json[0]}/{POG_YEARS[year]}/{pog_json[1]}"


def get_egamma_json(year: str) -> str:
    """
    returns egamma json file path

    Parameters:
    -----------
        json_name:
            json name
        year:
            dataset year {2022preEE, 2022postEE, 2023preBPix, 2023postBPix}
    """
    """
    if json_name in EGAMMA_JSONS:
        egamma_json = EGAMMA_JSONS[json_name]
    else:
        print(f"No json for {json_name}")
    return f"{EGAMMA_CORRECTION_PATH}/{EGAMMA_YEARS[year]}/{egamma_json[0]}/{egamma_json[1]}"
    """
    return f"{Path.cwd()}/analysis/data/{year}_electronSS.json.gz"


def get_muon_hlt_json(year: str)  -> str:
    return f"{Path.cwd()}/analysis/data/{year}_Muon_HLT_Eff.json"
    

def unflat_sf(sf: ak.Array, in_limit_mask: ak.Array, n: ak.Array):
    """
    get scale factors for in-limit objects (otherwise assign 1).
    Unflat array to original shape and multiply scale factors event-wise

    Parameters:
    -----------
        sf:
            Array with 1D scale factors
        in_limit_mask:
            Array mask for events with objects within correction limits
        n:
            Array with number of objects per event
    """
    sf = ak.where(in_limit_mask, sf, ak.ones_like(sf))
    return ak.fill_none(ak.prod(ak.unflatten(sf, n), axis=1), value=1)