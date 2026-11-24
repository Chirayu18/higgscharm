import correctionlib
import numpy as np
import awkward as ak
from pathlib import Path
from analysis.corrections.met import update_met
from analysis.corrections.event_rng import object_normals
from analysis.corrections.correctionlib_files import correction_files


def apply_electron_ss_corrections(
    events: ak.Array,
    year: str,
    shifts: list,
    corrections_config: dict,
):
    """
    Apply electron scale and smearing corrections for Run3

    from docs https://egammapog.docs.cern.ch/Run3/SaS/

    The purpose of scale and smearing (more formal: energy scale and resolution corrections) is to correct and calibrate electron and photon energies in data and MC. This step is performed after the MC-based semi-parametric EGamma energy regression, which is applied to both MC and data (aiming to correct for inherent imperfections that are not in principle related to data/MC differences like crystal-by-crystal differences, intermodule gaps, ...). The remaining differences between the electron and photon energy scales and the resolution in the data and simulation need to be corrected for. To address this, a multistep procedure is implemented to calibrate the residual energy scale in data. Additionally, an extra smearing is applied to the electron or photon energy in simulation to ensure that the energy resolution matches that observed in data

    https://gitlab.cern.ch/cms-analysis-corrections/EGM/examples/-/blob/latest/egmScaleAndSmearingExample.py
    """
    cset = correctionlib.CorrectionSet.from_file(correction_files["electron_ss"][year])

    electrons = events.Electron
    counts = ak.num(events.Electron)

    flat_ele = ak.flatten(electrons)
    seedgain = flat_ele.seedGain
    run = np.repeat(events.run, counts)
    sceta = (
        flat_ele.superclusterEta
        if year == "2024"
        else flat_ele.eta + flat_ele.deltaEtaSC
    )
    r9 = flat_ele.r9
    pt = flat_ele.pt

    if hasattr(events, "genWeight"):
        # for MC, we apply the smearing correction
        smear = cset["SmearAndSyst"].evaluate("smear", pt, r9, sceta)
        # 2026-11-10: per-electron reproducible normals (event_rng) replace a
        # per-chunk seed=42 stream, which depended on chunk boundaries.
        random_numbers = object_normals(events, "Electron", "electron_smearing")
        assert len(random_numbers) == len(pt)
        smearing = 1 + smear * random_numbers
        pt_corrected = pt * smearing
    else:
        # the scale correction is applied to data
        scale = cset.compound["Scale"].evaluate("scale", run, sceta, r9, pt, seedgain)
        pt_corrected = pt * scale

    # add nominal correction to shifts
    electrons["pt"] = ak.unflatten(pt_corrected, counts)
    for i in range(len(shifts)):
        shifts[i][0]["Electron"] = electrons

    # uncertainties
    if corrections_config["object_shifts"]:
        ele_smear_up, ele_smear_down = events.Electron, events.Electron
        ele_scale_up, ele_scale_down = events.Electron, events.Electron
        # scale and smearing uncertainties should be evaluated on the original MC only
        if hasattr(events, "genWeight"):
            # Obtain the uncertainty on the smearing width using MC original variables (pt, r9, and ScEta).
            unc_smear = cset["SmearAndSyst"].evaluate("smear", pt, r9, sceta)
            # FIX 2026-10-06: these evaluated the NOMINAL "smear" key, so CMS_res_e Up == Down == nominal
            smear_up = cset["SmearAndSyst"].evaluate("smear_up", pt, r9, sceta)
            smear_down = np.maximum(cset["SmearAndSyst"].evaluate("smear_down", pt, r9, sceta), 0.0)
            # In 2022, the "smear_down" variation can lead to negative smearing width in some cases, which is unphysical.
            # Therefore, we use max(smear - unc_smear, 0) to ensure the smearing width is non-negative.
            smearing_up = 1 + smear_up * random_numbers
            smearing_down = 1 + smear_down * random_numbers

            ele_smear_up["pt"] = ak.unflatten(pt * smearing_up, counts)
            ele_smear_down["pt"] = ak.unflatten(pt * smearing_down, counts)

            # the scale uncertainty, is also evaluated on MC original variables (pt, r9, and ScEta) BUT applied on the smeared pt
            unc_scale = cset["SmearAndSyst"].evaluate("escale", pt, r9, sceta)
            scale_up = cset["SmearAndSyst"].evaluate("scale_up", pt, r9, sceta)
            scale_down = cset["SmearAndSyst"].evaluate("scale_down", pt, r9, sceta)

            ele_scale_up["pt"] = ak.unflatten(scale_up * pt_corrected, counts)
            ele_scale_down["pt"] = ak.unflatten(scale_down * pt_corrected, counts)

        shifts += [
            (
                {
                    "Jet": shifts[0][0]["Jet"],
                    "MET": shifts[0][0]["MET"],
                    "Muon": shifts[0][0]["Muon"],
                    "Electron": ele_scale_up,
                },
                f"CMS_scale_e_{year[:4]}Up",
            )
        ]
        shifts += [
            (
                {
                    "Jet": shifts[0][0]["Jet"],
                    "MET": shifts[0][0]["MET"],
                    "Muon": shifts[0][0]["Muon"],
                    "Electron": ele_scale_down,
                },
                f"CMS_scale_e_{year[:4]}Down",
            )
        ]
        shifts += [
            (
                {
                    "Jet": shifts[0][0]["Jet"],
                    "MET": shifts[0][0]["MET"],
                    "Muon": shifts[0][0]["Muon"],
                    "Electron": ele_smear_up,
                },
                f"CMS_res_e_{year[:4]}Up",
            )
        ]
        shifts += [
            (
                {
                    "Jet": shifts[0][0]["Jet"],
                    "MET": shifts[0][0]["MET"],
                    "Muon": shifts[0][0]["Muon"],
                    "Electron": ele_smear_down,
                },
                f"CMS_res_e_{year[:4]}Down",
            )
        ]
    return shifts
