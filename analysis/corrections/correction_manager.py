import numpy as np
from coffea.analysis_tools import Weights
from analysis.filesets.utils import get_nano_version
from analysis.corrections.muon import MuonWeights
from analysis.corrections.ctag import CTagCorrector
from analysis.corrections.ctag2d import CTag2DCorrector
from analysis.corrections.jetvetomaps import jet_veto
from analysis.corrections.pileup import add_pileup_weight
from analysis.corrections.nnlops import add_nnlops_weight
from analysis.corrections.lhepdf import add_lhepdf_weight
from analysis.corrections.electron import ElectronWeights
from analysis.corrections.jerc import apply_jerc_corrections
from analysis.corrections.lhescale import add_scalevar_weight
from analysis.corrections.lhe_norm import lhe_variation_means
from analysis.corrections.lhe_yukawa_fix import table_for_dataset
from analysis.corrections.hplusc_fs import add_hplusc_fs_weight
from analysis.corrections.met import apply_met_phi_corrections
from analysis.corrections.met_propagation import propagate_lepton_shifts
from analysis.corrections.met_xy import apply_run3_met_xy
from analysis.corrections.muon_ss import apply_muon_ss_corrections
from analysis.corrections.partonshower import add_partonshower_weight
from analysis.corrections.toppt import add_toppt_weight
from analysis.corrections.higgs_hf import add_higgs_hf_weight
from analysis.corrections.electron_ss import apply_electron_ss_corrections
from analysis.leptonmva import add_tthmva_scores, TTHMVA_YEARS


def object_corrector_manager(events, year, corrections_config):
    """apply object level corrections"""
    objcorr_config = corrections_config["object"]
    # Workflows historically use a list; accept an explicit mapping as well.
    object_options = (objcorr_config if isinstance(objcorr_config, dict)
                      else {key: True for key in (objcorr_config or [])})

    # Keep the uncalibrated reference before either correction can replace pt.
    original_leptons = {key: events[key] for key, option in
                        (('Muon', 'muon'), ('Electron', 'electron'))
                        if option in (objcorr_config or {})}
    shifts = []
    if objcorr_config:
        # 2026-11-05: retrained ttH lepton MVA (analysis/leptonmva/tth.py). Scores are computed once, on NanoAOD-level
        # inputs, before the muon/electron scale corrections, so every shift pass below inherits Muon/Electron.tthMVA.
        if "tthMVA" in objcorr_config:
            if year not in TTHMVA_YEARS:
                raise ValueError(f"tthMVA requested for {year}; the retrained model is defined for {TTHMVA_YEARS} only")
            add_tthmva_scores(events, year)
        # apply jet energy scale and smearing corrections
        if "jet" in objcorr_config:
            shifts = apply_jerc_corrections(
                events=events,
                year=year,
                shifts=shifts,
                corrections_config=corrections_config,
            )
        else:
            met_field_key = "PuppiMET" if int(get_nano_version(year)) >= 12 else "MET"
            shifts = [({"Jet": events.Jet, "MET": events[met_field_key]}, None)]

        # XY is evaluated after JEC, before either lepton correction. Every
        # existing jet/MET shift is corrected and the new XY pair inherits the
        # nominal jet collection; leptons are added and propagated below.
        if object_options.get("met_phi", False):
            shifts = apply_run3_met_xy(events, year, shifts, corrections_config)

        if "muon" in objcorr_config:
            # apply muon scale and smearing corrections
            shifts = apply_muon_ss_corrections(
                events=events,
                year=year,
                shifts=shifts,
                corrections_config=corrections_config,
            )
        else:
            for i in range(len(shifts)):
                shifts[i][0]["Muon"] = events.Muon

        if "electron" in objcorr_config:
            # electron scale and smearing corrections are applied only to run3 datasets
            # they are already applied in nanov9 datasets
            if year.startswith("202"):
                shifts = apply_electron_ss_corrections(
                    events=events,
                    year=year,
                    shifts=shifts,
                    corrections_config=corrections_config,
                )
        else:
            for i in range(len(shifts)):
                shifts[i][0]["Electron"] = events.Electron

        # Opt-in for the new production; existing workflows keep their numerical
        # convention. Every nominal/object shift receives both lepton changes.
        if object_options.get("propagate_leptons_to_met", False):
            shifts = propagate_lepton_shifts(shifts, original_leptons)

        # apply jet veto
        if "jetveto" in objcorr_config:
            event_veto = jet_veto(events, year)
            vetoed_events = events[~event_veto]
            for collections, _ in shifts:
                for key in collections:
                    collections[key] = collections[key][~event_veto]
        else:
            vetoed_events = events

    return vetoed_events, shifts


def weight_manager(pruned_ev, year, dataset, workflow_config, category, shift):
    """apply event level corrections (weights)"""
    # get weights config info
    weights_config = workflow_config.corrections_config["event_weights"]
    # initialize weights container
    weights_container = Weights(len(pruned_ev), storeIndividual=True)
    # get nano version
    nano_version = get_nano_version(year)
    # add weights
    if hasattr(pruned_ev, "genWeight"):
        if weights_config["genWeight"]:
            weights_container.add("genweight", pruned_ev.genWeight)

        if "pileupWeight" in weights_config:
            if weights_config["pileupWeight"]:
                add_pileup_weight(
                    events=pruned_ev,
                    year=year,
                    shift=shift,
                    weights_container=weights_container,
                )
        if "partonshowerWeight" in weights_config:
            if weights_config["partonshowerWeight"]:
                if "PSWeight" in pruned_ev.fields:
                    add_partonshower_weight(
                        events=pruned_ev,
                        weights_container=weights_container,
                        shift=shift,
                    )
        # per-dataset LHE variation normalisation (lhe_norm.py), opt-in via lhe_renorm_datasets
        _lhe_norm = (None, None)
        # dataset arrives as <sample>_<partition> (condor) -> match the sample-name prefix
        if shift is None and any(dataset == d or dataset.startswith(d + "_")
                                 for d in (weights_config.get("lhe_renorm_datasets") or [])):
            _lhe_norm = lhe_variation_means(pruned_ev.metadata["filename"])
            if table_for_dataset(dataset) is None:
                raise RuntimeError(f"{dataset} is in lhe_renorm_datasets but has no lhe_yukawa table")
        if "lhepdfWeight" in weights_config:
            if weights_config["lhepdfWeight"]:
                if "LHEPdfWeight" in pruned_ev.fields:
                    add_lhepdf_weight(
                        events=pruned_ev,
                        weights_container=weights_container,
                        shift=shift,
                        pdf_norm=_lhe_norm[1],
                    )
        if "lhescaleWeight" in weights_config:
            if weights_config["lhescaleWeight"]:
                if "LHEScaleWeight" in pruned_ev.fields:
                    # lhe_renorm datasets: per-event Yukawa correction for scale (global mean kept for PDF only)
                    add_scalevar_weight(
                        events=pruned_ev,
                        weights_container=weights_container,
                        shift=shift,
                        scale_norm=None,
                        yukawa_table=table_for_dataset(dataset) if _lhe_norm[0] is not None else None,
                    )
        if "nnlopsWeight" in weights_config:
            if weights_config["nnlopsWeight"]:
                if dataset.startswith("GluGluH"):
                    add_nnlops_weight(
                        events=pruned_ev,
                        weights_container=weights_container,
                    )
        if "toppTWeight" in weights_config:
            if weights_config["toppTWeight"]:
                # tt only (AN-23-102 line 566); the module no-ops on other datasets
                add_toppt_weight(
                    events=pruned_ev,
                    weights_container=weights_container,
                    shift=shift,
                    dataset=dataset,
                )
        if "higgsHFWeight" in weights_config:
            if weights_config["higgsHFWeight"]:
                # ggH/VBF only; per-event, keyed on gen-jet heavy flavour, so it
                # replaces the mis-scoped flat lnN on the merged higgsbkg group
                add_higgs_hf_weight(
                    events=pruned_ev,
                    weights_container=weights_container,
                    shift=shift,
                    dataset=dataset,
                    # charm: AN-23-102 scopes this to the ggH+HF composition and our
                    # signal is H+c, so the c-flavour variant is the relevant one.
                    flav="c",
                )
        if "hpluscFSShape" in weights_config:
            if weights_config["hpluscFSShape"]:
                # 3FS-vs-4FS flavour-scheme SHAPE nuisance (2026-11-04), central 4FS H+c only: the value is the list of
                # datasets (prefix match) it is added to; leading GEN c-hadron pT binned weights (hplusc_fs.py)
                add_hplusc_fs_weight(
                    events=pruned_ev,
                    weights_container=weights_container,
                    shift=shift,
                    dataset=dataset,
                    datasets=weights_config["hpluscFSShape"],
                )
        if "muon" in weights_config:
            if weights_config["muon"]:
                if "selected_muons" in pruned_ev.fields:
                    muon_weights = MuonWeights(
                        events=pruned_ev,
                        year=year,
                        shift=shift,
                        weights=weights_container,
                    )
                    if "id" in weights_config["muon"]:
                        if weights_config["muon"]["id"]:
                            muon_weights.add_id_weights(
                                id_wp=weights_config["muon"]["id"]
                            )
                    if "iso" in weights_config["muon"]:
                        if weights_config["muon"]["iso"]:
                            muon_weights.add_iso_weights(
                                id_wp=weights_config["muon"]["id"],
                                iso_wp=weights_config["muon"]["iso"],
                            )
                    # 2026-11-05: retrained ttH lepton-MVA working point SF (HWW/latinos JSON), tvl/MVA port
                    if weights_config["muon"].get("promptMVA"):
                        muon_weights.add_promptMVA_weights(
                            id_wp=weights_config["muon"]["id"],
                            iso_wp=weights_config["muon"]["iso"],
                            promptMVA_wp=weights_config["muon"]["promptMVA"],
                        )
                    if "trigger" in weights_config["muon"]:
                        if weights_config["muon"]["trigger"]:
                            muon_weights.add_trigger_weights(
                                dataset=dataset,
                                id_wp=weights_config["muon"]["id"],
                                iso_wp=weights_config["muon"]["iso"],
                                hlt_paths=workflow_config.event_selection["hlt_paths"],
                            )
        if "electron" in weights_config:
            if weights_config["electron"]:
                if "selected_electrons" in pruned_ev.fields:
                    electron_weights = ElectronWeights(
                        events=pruned_ev,
                        year=year,
                        shift=shift,
                        weights=weights_container,
                    )
                    if "id" in weights_config["electron"]:
                        if weights_config["electron"]["id"]:
                            electron_weights.add_id_weights(
                                id_wp=weights_config["electron"]["id"]
                            )
                    # 2026-11-05: retrained ttH lepton-MVA working point SF (HWW/latinos JSON), tvl/MVA port
                    if weights_config["electron"].get("promptMVA"):
                        electron_weights.add_promptMVA_weights(
                            id_wp=weights_config["electron"]["id"],
                            promptMVA_wp=weights_config["electron"]["promptMVA"],
                        )
                    if "reco" in weights_config["electron"]:
                        if weights_config["electron"]["reco"]:
                            if nano_version == "9":
                                electron_weights.add_reco_weights("RecoAbove20")
                                electron_weights.add_reco_weights("RecoBelow20")
                            elif nano_version == "12":
                                electron_weights.add_reco_weights("RecoBelow20")
                                electron_weights.add_reco_weights("Reco20to75")
                                electron_weights.add_reco_weights("RecoAbove75")
                            elif nano_version == "15":
                                electron_weights.add_reco_weights("Reco20to75")
                                electron_weights.add_reco_weights("RecoAbove75")
                    if "trigger" in weights_config["electron"]:
                        if weights_config["electron"]["trigger"]:
                            # no available HLT Sfs for 2024 yet
                            if nano_version != "15":
                                electron_weights.add_hlt_weights(
                                    id_wp=weights_config["electron"]["id"],
                                )

        if "ctagging" in weights_config:
            if weights_config["ctagging"]:
                working_points = ["loose", "medium", "tight"]
                if weights_config["ctagging"]["wp"] not in working_points:
                    raise ValueError(
                        f"There are no available c-tag SFs for the working point. Please specify {working_points}"
                    )
                ctag_corrector = CTagCorrector(
                    events=pruned_ev,
                    weights=weights_container,
                    worging_point=weights_config["ctagging"]["wp"],
                    category=category,
                    year=year,
                    shift=shift,
                )
                ctag_corrector.add_ctag_weights(flavor="b")
                ctag_corrector.add_ctag_weights(flavor="c")
                ctag_corrector.add_ctag_weights(flavor="light")

        if "ctagging_2d" in weights_config:
            if weights_config["ctagging_2d"]:
                ctag2d_corrector = CTag2DCorrector(
                    events=pruned_ev,
                    weights=weights_container,
                    year=year,
                    shift=shift,
                )
                ctag2d_corrector.add_weights()
    else:
        weights_container.add("weight", np.ones(len(pruned_ev)))
    return weights_container
