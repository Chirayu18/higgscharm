# tools to apply JEC/JER and compute their uncertainties
# JME recommendations: https://cms-jerc.web.cern.ch/Recommendations/
# taken from https://github.com/cms-btv-pog/BTVNanoCommissioning/blob/master/src/BTVNanoCommissioning/utils/correction.py
import copy
import yaml
import contextlib
import correctionlib
import numpy as np
import awkward as ak
import importlib.resources
from pathlib import Path
from analysis.corrections.correctionlib_files import correction_files
from analysis.filesets.utils import get_dataset_era, get_nano_version
from coffea.lookup_tools import extractor
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory, CorrectedMETFactory
from coffea.jetmet_tools.CorrectedMETFactory import corrected_polar_met

# ----------------------------------------------------------------------------------------------------
# COFFEA'S JET/MET TOOLS
# Used for Run2 datasets and 2023preBPix campaign
# ----------------------------------------------------------------------------------------------------
with importlib.resources.open_text(
    f"analysis.corrections", f"jec_params_coffea.yaml"
) as file:
    jec_params_coffea = yaml.safe_load(file)


def jec_names_and_sources(year):
    names = {}
    algo = jec_params_coffea["algorithm"][year]
    suffix = {
        "jec_names": [
            f"_{level}_{algo}" for level in jec_params_coffea["jec_levels_mc"][year]
        ],
        "jec_names_data": [
            f"_{level}_{algo}" for level in jec_params_coffea["jec_levels_data"][year]
        ],
        "junc_names": [f"_Uncertainty_{algo}"],
        "junc_names_data": [f"_Uncertainty_{algo}"],
        "junc_sources": [f"_UncertaintySources_{algo}"],
        "junc_sources_data": [f"_UncertaintySources_{algo}"],
        "jer_names": [f"_PtResolution_{algo}"],
        "jersf_names": [f"_SF_{algo}"],
    }
    for key, suff in suffix.items():
        if "data" in key:
            names[key] = {}
            for run in jec_params_coffea["runs"][year]:
                for tag, iruns in jec_params_coffea["jec_data_tags"][year].items():
                    if run in iruns:
                        names[key].update({run: [f"{tag}{s}" for s in suff]})
        else:
            tag = (
                jec_params_coffea["jer_tags"][year]
                if "jer" in key
                else jec_params_coffea["jec_tags"][year]
            )
            names[key] = [f"{tag}{s}" for s in suff]
    return names


def get_jet_evaluator(year):
    names = jec_names_and_sources(year)
    extensions = {
        "jec_names": "jec",
        "jer_names": "jr",
        "jersf_names": "jersf",
        "junc_names": "junc",
        "junc_sources": "junc",
    }
    # prepare evaluators for JEC, JER and their systematics
    jec_ext = extractor()
    for opt, ext in extensions.items():
        # MC
        with contextlib.ExitStack() as stack:
            jec_files = [
                stack.enter_context(
                    importlib.resources.path("analysis.data.jec", f"{name}.{ext}.txt")
                )
                for name in names[opt]
            ]
            jec_ext.add_weight_sets([f"* * {file}" for file in jec_files])
        # Data
        if "jer" in opt:
            continue
        data = []
        for run, items in names[f"{opt}_data"].items():
            data.extend(items)
        data = list(set(data))
        with contextlib.ExitStack() as stack:
            jec_data_files = [
                stack.enter_context(
                    importlib.resources.path("analysis.data.jec", f"{name}.{ext}.txt")
                )
                for name in data
            ]
            jec_ext.add_weight_sets([f"* * {file}" for file in jec_data_files])

    jec_ext.finalize()
    jet_evaluator = jec_ext.make_evaluator()
    return jet_evaluator


def get_corrected_jets_coffea(events, year):
    is_mc = hasattr(events, "genWeight")
    dataset = events.metadata["dataset"]

    # add requiered variables to Jet collection
    jets = events.Jet

    # set raw pT and Mass, otherwise original pT and Mass will be used as 'raw' values
    events["Jet", "pt_raw"] = (1 - jets.rawFactor) * jets.pt
    events["Jet", "mass_raw"] = (1 - jets.rawFactor) * jets.mass
    events["Jet", "event_rho"] = (
        ak.ones_like(jets.pt) * events.Rho.fixedGridRhoFastjetAll
        if hasattr(events, "Rho")
        else ak.broadcast_arrays(events.fixedGridRhoFastjetAll, jets.pt)[0]
    )
    if is_mc:
        # set ptGenJet (required for hybrid JER smearing method)
        events["Jet", "pt_gen"] = ak.values_astype(
            ak.fill_none(jets.matched_gen.pt, 0), np.float32
        )

    # set inputs for jec, jer and junc stack
    names = jec_names_and_sources(year)
    jet_evaluator = get_jet_evaluator(year)

    stacks_def = {
        "jec_stack": ["jec_names"],
        "jer_stack": ["jer_names", "jersf_names"],
        "junc_stack": ["junc_names"],
    }
    stacks = {}
    for key, vals in stacks_def.items():
        stacks[key] = []
        for v in vals:
            stacks[key].extend(names[v])
    jec_input_options = {}
    jet_variations = ["jec", "jer", "junc"]
    for variation in jet_variations:
        jec_input_options[variation] = {}
        for name in stacks[f"{variation}_stack"]:
            jec_input_options[variation][name] = jet_evaluator[name]
    for src in names["junc_sources"]:
        for key in jet_evaluator.keys():
            if src in key:
                jec_input_options["junc"][key] = jet_evaluator[key]

    jec_options = {}
    jec_options.update(jec_input_options["jec"])
    jec_options.update(jec_input_options["junc"])
    if is_mc:
        jec_options.update(jec_input_options["jer"])

    # set jerc name map (I don't use JECStack.blank_name_map since it includes 'ptRaw' and 'massRaw' by default)
    jec_name_map = {
        "JetPt": "pt",
        "JetMass": "mass",
        "JetEta": "eta",
        "JetA": "area",
        "ptGenJet": "pt_gen",
        "Rho": "event_rho",
        "ptRaw": "pt_raw",
        "massRaw": "mass_raw",
        "METpt": "pt",
        "METphi": "phi",
        "JetPhi": "phi",
        "UnClusteredEnergyDeltaX": "MetUnclustEnUpDeltaX",
        "UnClusteredEnergyDeltaY": "MetUnclustEnUpDeltaY",
    }

    if is_mc:
        # create MC factory with jec, jer and junc stack
        stack = JECStack(jec_options)
        jec_factory = CorrectedJetsFactory(jec_name_map, stack)
    else:
        # create a separate factory for the data era
        era = get_dataset_era(dataset, year)
        jec_inputs_data = {}
        for opt in ["jec", "junc"]:
            jec_inputs_data.update(
                {name: jet_evaluator[name] for name in names[f"{opt}_names_data"][era]}
            )
        for src in names["junc_sources_data"][era]:
            for key in jet_evaluator.keys():
                if src in key:
                    jec_inputs_data[key] = jet_evaluator[key]
        jec_stack_data = JECStack(jec_inputs_data)
        jec_factory = CorrectedJetsFactory(jec_name_map, jec_stack_data)

    jets = jec_factory.build(events.Jet, events.caches[0])

    # Type-I MET correction
    met_field_key = "PuppiMET" if hasattr(events, "Rho") else "MET"
    if met_field_key == "PuppiMET":
        nocorrmet = events[met_field_key]
        met = copy.copy(nocorrmet)
        metinfo = [nocorrmet.pt, nocorrmet.phi, jets.pt, jets.phi, jets.pt_raw]
        met["pt"], met["phi"] = (
            ak.values_astype(corrected_polar_met(*metinfo).pt, np.float32),
            ak.values_astype(corrected_polar_met(*metinfo).phi, np.float32),
        )
        met["orig_pt"], met["orig_phi"] = nocorrmet["pt"], nocorrmet["phi"]
        # NanoAODv12 ships the unclustered-energy variation as ready-made branches on
        # PuppiMET. The CorrectedMETFactory path below never runs for Run 3, so the
        # `MET_UnclusteredEnergy` field the shift loop looks for is never attached --
        # take the branches directly instead. (AN-23-102 Table 16: MET unclustered.)
        if "ptUnclusteredUp" in nocorrmet.fields:
            unclust = {}
            for _dirn, _suf in (("up", "Up"), ("down", "Down")):
                _v = copy.copy(met)
                _v["pt"] = ak.values_astype(
                    nocorrmet[f"ptUnclustered{_suf}"], np.float32
                )
                _v["phi"] = ak.values_astype(
                    nocorrmet[f"phiUnclustered{_suf}"], np.float32
                )
                unclust[_dirn] = _v
            met["MET_UnclusteredEnergy"] = ak.zip(
                {"up": unclust["up"], "down": unclust["down"]}, depth_limit=1
            )
    else:
        met = CorrectedMETFactory(jec_name_map).build(events[met_field_key], jets, {})

    return jets, met


# ----------------------------------------------------------------------------------------------------
# CORRECTIONLIB
# Used for Run3 datasets execpt 2023preBPix campaign
# ----------------------------------------------------------------------------------------------------
with importlib.resources.open_text(
    f"analysis.corrections", f"jec_params_correctionlib.yaml"
) as file:
    jec_params_correctionlib = yaml.safe_load(file)


def get_corr_inputs(input_dict, corr_obj, jersyst="nom"):
    """
    Helper function for getting values of input variables
    given a dictionary and a correction object.
    """
    input_values = []
    for inputs in corr_obj.inputs:
        if "systematic" in inputs.name:
            input_values.append(jersyst)
        else:
            input_values.append(
                np.array(
                    input_dict[
                        inputs.name.replace("Jet", "")
                        .replace("Pt", "pt")
                        .replace("Phi", "phi")
                        .replace("Eta", "eta")
                        .replace("Mass", "mass")
                        .replace("Rho", "rho")
                        .replace("A", "area")
                    ]
                )
            )
    return input_values


def get_corrected_jets_correctionlib(events, year):
    dataset = events.metadata["dataset"]
    is_mc = hasattr(events, "genWeight")

    cset_jersmear_paths = [
        "/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/JER-Smearing/latest/jer_smear.json.gz",
        "/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/jer_smear.json.gz",
    ]
    for _jersmear_path in cset_jersmear_paths:
        if Path(_jersmear_path).exists:
            cset_jersmear = correctionlib.CorrectionSet.from_file(_jersmear_path)
            break
    else:
        warnings.warn(
            "JER smearing JSON not found in CVMFS. JER smearing corrections will be disabled.",
            RuntimeWarning,
            stacklevel=2,
        )
        cset_jersmear = {"JERSmear": None}
    sf_jersmear = cset_jersmear["JERSmear"]

    cset = correctionlib.CorrectionSet.from_file(correction_files["jerc"][year])

    for d in jec_params_correctionlib[year].keys():
        if (
            np.all(
                np.char.find(
                    np.array(list(cset.keys())),
                    jec_params_correctionlib[year][d],
                )
            )
            == -1
        ):
            raise (
                f"{d} has no JEC map : {jec_params_correctionlib[year][d]} available"
            )

    jecname = ""
    # https://cms-jerc.web.cern.ch/JECUncertaintySources/, currently no recommendation of reduced/full split sources
    syst_list = [
        i.split("_")[3]
        for i in cset.keys()
        if "MC" in i
        and "L1" not in i
        and "L2" not in i
        and "L3" not in i
        and i.split("_")[3] != "MC"
    ]
    if is_mc:
        jecname = jec_params_correctionlib[year]["MC"].split(" ")[0] + "_MC"
        jrname = jec_params_correctionlib[year]["MC"].split(" ")[1] + "_MC"
    else:
        jecname = [v for k, v in jec_params_correctionlib[year].items() if k in dataset]
        if len(jecname) > 1:
            raise ValueError("Multiple uncertainties match to this era")
        elif len(jecname) == 0:
            raise ValueError(
                "Available JEC variations in this era are not compatible with this file. Did you choose the correct dataset-era combination?"
            )
        else:
            jecname = jecname[0] + "_DATA"

    # Jet Energy Scale
    nocorrjet = events.Jet
    nocorrjet["pt_raw"] = (1 - nocorrjet["rawFactor"]) * nocorrjet["pt"]
    nocorrjet["mass_raw"] = (1 - nocorrjet["rawFactor"]) * nocorrjet["mass"]
    nocorrjet["rho"] = (
        ak.ones_like(nocorrjet.pt) * events.Rho.fixedGridRhoFastjetAll
        if hasattr(events, "Rho")
        else ak.broadcast_arrays(events.fixedGridRhoFastjetAll, nocorrjet.pt)[0]
    )
    nocorrjet["EventID"] = ak.broadcast_arrays(events.event, nocorrjet.pt)[0]
    nocorrjet["run"] = ak.broadcast_arrays(events.run, nocorrjet.pt)[0]
    if is_mc:
        genjetidx = ak.where(nocorrjet.genJetIdx == -1, 0, nocorrjet.genJetIdx)
        nocorrjet["Genpt"] = ak.where(
            nocorrjet.genJetIdx == -1, -1, events.GenJet[genjetidx].pt
        )
    jets = copy.deepcopy(nocorrjet)
    jets["orig_pt"] = ak.values_astype(nocorrjet["pt"], np.float32)

    j, nj = ak.flatten(nocorrjet), ak.num(nocorrjet)

    # get JES correction factor
    jec_corr = cset.compound[f"{jecname}_L1L2L3Res_AK4PFPuppi"]
    jec_input = get_corr_inputs(j, jec_corr)
    jec_flat_corr_factor = jec_corr.evaluate(*jec_input)

    # Jet Energy Resolution
    if is_mc:
        jer_sf = cset[f"{jrname}_ScaleFactor_AK4PFPuppi"]
        jer_ptres = cset[f"{jrname}_PtResolution_AK4PFPuppi"]
        # for MC, correct the jet pT with JEC first
        j["pt"] = j["pt_raw"] * jec_flat_corr_factor
        j["mass"] = j["mass_raw"] * jec_flat_corr_factor
        jer_sf_input = get_corr_inputs(j, jer_sf)
        jer_ptres_input = get_corr_inputs(j, jer_ptres)
        j["JER"] = jer_ptres.evaluate(*jer_ptres_input)
        j["JERSF"] = jer_sf.evaluate(*jer_sf_input)
        if sf_jersmear is None:
            warnings.warn(
                "JER smearing coefficients unavailable. "
                "Proceeding without applying JER smearing.",
                RuntimeWarning,
                stacklevel=2,
            )
            corr_factor = jec_flat_corr_factor
        else:
            jer_smear_input = get_corr_inputs(j, sf_jersmear)
            corr_factor = jec_flat_corr_factor * sf_jersmear.evaluate(*jer_smear_input)
    else:
        # in data only the JEC is applied
        corr_factor = jec_flat_corr_factor

    corr_factor = ak.unflatten(corr_factor, nj)

    jets["pt"] = ak.values_astype(nocorrjet["pt_raw"] * corr_factor, np.float32)
    jets["mass"] = ak.values_astype(nocorrjet["mass_raw"] * corr_factor, np.float32)

    # Type-I MET correction
    met_field_key = "PuppiMET" if int(get_nano_version(year)) >= 12 else "MET"
    nocorrmet = events[met_field_key]
    met = copy.copy(nocorrmet)
    metinfo = [nocorrmet.pt, nocorrmet.phi, jets.pt, jets.phi, jets.pt_raw]
    met["pt"], met["phi"] = (
        ak.values_astype(corrected_polar_met(*metinfo).pt, np.float32),
        ak.values_astype(corrected_polar_met(*metinfo).phi, np.float32),
    )
    met["orig_pt"], met["orig_phi"] = nocorrmet["pt"], nocorrmet["phi"]

    # JEC variations
    if is_mc:
        jesuncmap = cset[f"{jecname}_Total_AK4PFPuppi"]
        jesunc = ak.unflatten(jesuncmap.evaluate(j.eta, j.pt), nj)
        unc_jets, unc_met = {}, {}
        for var in ["up", "down"]:
            fac = 1.0 if var == "up" else -1.0
            # JES total
            unc_jets[f"JES_jes{var}"] = copy.copy(nocorrjet)
            unc_met[f"JES_jes{var}"] = copy.copy(nocorrmet)

            unc_jets[f"JES_jes{var}"]["pt"] = ak.values_astype(
                jets["pt"] * (1 + fac * jesunc),
                np.float32,
            )
            unc_jets[f"JES_jes{var}"]["mass"] = ak.values_astype(
                jets["mass"] * (1 + fac * jesunc),
                np.float32,
            )
            unc_met[f"JES_jes{var}"]["pt"] = corrected_polar_met(
                nocorrmet.pt,
                nocorrmet.phi,
                unc_jets[f"JES_jes{var}"]["pt"],
                jets.phi,
                jets.pt_raw,
            ).pt
            unc_met[f"JES_jes{var}"]["phi"] = corrected_polar_met(
                nocorrmet.pt,
                nocorrmet.phi,
                unc_jets[f"JES_jes{var}"]["pt"],
                jets.phi,
                jets.pt_raw,
            ).phi

            jer_sf_input_var = get_corr_inputs(j, jer_sf, var)

            # JER variations
            if sf_jersmear is None:
                warnings.warn(
                    "Skipping JER smearing variations because the "
                    "correction file is unavailable.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                unc_jets[f"JER{var}"] = copy.copy(jets)
                unc_met[f"JER{var}"] = copy.copy(met)
            else:
                unc_jets[f"JER{var}"] = copy.copy(nocorrjet)
                unc_met[f"JER{var}"] = copy.copy(nocorrmet)
                j["JERSF"] = jer_sf.evaluate(*jer_sf_input_var)
                jer_smear_input_var = get_corr_inputs(j, sf_jersmear)
                corrFactor_var = jec_flat_corr_factor * sf_jersmear.evaluate(
                    *jer_smear_input_var
                )
                corrFactor_var = ak.unflatten(corrFactor_var, nj)

                unc_jets[f"JER{var}"]["pt"] = ak.values_astype(
                    nocorrjet["pt_raw"] * corrFactor_var,
                    np.float32,
                )
                unc_jets[f"JER{var}"]["mass"] = ak.values_astype(
                    nocorrjet["mass_raw"] * corrFactor_var,
                    np.float32,
                )
                unc_met[f"JER{var}"]["pt"] = corrected_polar_met(
                    nocorrmet.pt,
                    nocorrmet.phi,
                    unc_jets[f"JER{var}"]["pt"],
                    jets.phi,
                    jets.pt_raw,
                ).pt
                unc_met[f"JER{var}"]["phi"] = corrected_polar_met(
                    nocorrmet.pt,
                    nocorrmet.phi,
                    unc_jets[f"JER{var}"]["pt"],
                    jets.phi,
                    jets.pt_raw,
                ).phi

        jets["JES_jes"] = ak.zip(
            {
                "up": unc_jets["JES_jesup"],
                "down": unc_jets["JES_jesdown"],
            }
        )
        jets["JER"] = ak.zip(
            {
                "up": unc_jets["JERup"],
                "down": unc_jets["JERdown"],
            }
        )
        met["JES_jes"] = ak.zip(
            {
                "up": unc_met["JES_jesup"],
                "down": unc_met["JES_jesdown"],
            }
        )
        met["JER"] = ak.zip(
            {
                "up": unc_met["JERup"],
                "down": unc_met["JERdown"],
            }
        )

    return jets, met


def apply_jerc_corrections(events, year, shifts, corrections_config):

    if year in correction_files["jerc"]:
        corr_func = get_corrected_jets_correctionlib
    else:
        corr_func = get_corrected_jets_coffea

    jets, met = corr_func(events, year)

    if hasattr(events, "genWeight") and corrections_config["object_shifts"]:
        if "JES_jes" in jets.fields and "JES_jes" in met.fields:
            shifts += [
                (
                    {
                        "Jet": jets.JES_jes.up,
                        "MET": met.JES_jes.up,
                    },
                    f"CMS_scale_j_{year[:4]}Up",
                ),
                (
                    {
                        "Jet": jets.JES_jes.down,
                        "MET": met.JES_jes.down,
                    },
                    f"CMS_scale_j_{year[:4]}Down",
                ),
            ]
        if "JER" in jets.fields and "JER" in met.fields:
            shifts += [
                (
                    {
                        "Jet": jets.JER.up,
                        "MET": met.JER.up,
                    },
                    f"CMS_res_j_{year[:4]}Up",
                ),
                (
                    {
                        "Jet": jets.JER.down,
                        "MET": met.JER.down,
                    },
                    f"CMS_res_j_{year[:4]}Down",
                ),
            ]
        if "MET_UnclusteredEnergy" in met.fields:
            shifts += [
                (
                    {
                        "Jet": jets,
                        "MET": met.MET_UnclusteredEnergy.up,
                    },
                    f"CMS_met_unclustered_{year[:4]}Up",
                ),
                (
                    {
                        "Jet": jets,
                        "MET": met.MET_UnclusteredEnergy.down,
                    },
                    f"CMS_met_unclustered_{year[:4]}Down",
                ),
            ]

    shifts.insert(0, ({"Jet": jets, "MET": met}, None))
    return shifts
