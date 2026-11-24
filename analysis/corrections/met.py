import correctionlib
import numpy as np
import awkward as ak


def apply_met_phi_corrections(
    events: ak.Array,
    is_mc: bool,
    year: str,
):
    """
    Apply MET phi modulation corrections

    Parameters:
    -----------
        events:
            Events array
        is_mc:
            True if dataset is MC
        year:
            Year of the dataset  {'2022preEE', '2022postEE'}

    Returns:
    --------
        corrected MET pt and phi


    Docs: (from https://twiki.cern.ch/twiki/bin/viewauth/CMS/MissingETRun2Corrections#xy_Shift_Correction_MET_phi_modu)
        The xy-Shift correction reduces the MET phi modulation. The distribution of true MET is independent of phi because of the rotational symmetry of the collisions around the beam axis. However, we observe that the reconstructed MET does depend on phi. The MET phi distribution has roughly a sinusoidal curve with the period of 2pi. The possible causes of the modulation include anisotropic detector responses, inactive calorimeter cells or tracking regions, the detector misalignment, the displacement of the beam spot. The amplitude of the modulation increases roughly linearly with the number of the pile-up interactions. We can reduce the amplitude of the phi modulation by shifting the origin of the coordinate in the transverse momentum plane as a function of different particle species and in bins of eta.
    """
    cset = correctionlib.CorrectionSet.from_file(
        "analysis/data/met/met_xy_corrections.json"
    )
    events["PuppiMET", "pt_raw"] = ak.ones_like(events.PuppiMET.pt) * events.PuppiMET.pt
    events["PuppiMET", "phi_raw"] = (
        ak.ones_like(events.PuppiMET.phi) * events.PuppiMET.phi
    )

    # make sure to not cross the maximum allowed value for uncorrected met
    met_pt = events.PuppiMET.pt_raw
    met_pt = np.clip(met_pt, 0.0, 6499.0)
    met_phi = events.PuppiMET.phi_raw
    met_phi = np.clip(met_phi, -3.15, 3.15)

    # use correct run ranges when working with data, otherwise use uniform run numbers in an arbitrary large window
    run_ranges = {
        "2022preEE": [355094, 359017],
        "2022postEE": [359045, 362760],
    }
    data_kind = "mc" if is_mc else "data"
    if data_kind == "mc":
        run = np.random.randint(
            run_ranges[year][0], run_ranges[year][1], size=len(met_pt)
        )
    else:
        run = events.run
    try:
        events["PuppiMET", "pt"] = cset[f"pt_metphicorr_puppimet_{data_kind}"].evaluate(
            met_pt.to_numpy(), met_phi.to_numpy(), events.PV.npvsGood.to_numpy(), run
        )
        events["PuppiMET", "phi"] = cset[
            f"phi_metphicorr_puppimet_{data_kind}"
        ].evaluate(
            met_pt.to_numpy(), met_phi.to_numpy(), events.PV.npvsGood.to_numpy(), run
        )
    except:
        pass


def update_met(events, other_obj, met_obj="PuppiMET") -> None:
    """
    helper function to compute new MET after lepton pT correction.
    It uses the 'pt_raw' and 'pt' fields from 'leptons' to update MET 'pt' and 'phi' fields

    Parameters:
        - events:
            Events array
        - lepton:
            Lepton name {'Muon', 'Electron', Tau'}

    https://github.com/columnflow/columnflow/blob/16d35bb2f25f62f9110a8f1089e8dc5c62b29825/columnflow/calibration/util.py#L42
    https://github.com/Katsch21/hh2bbtautau/blob/e268752454a0ce0089ff08cc6c373a353be77679/hbt/calibration/tau.py#L117
    """
    # get needed lepton and MET fields
    other_pt_raw = events[other_obj, "pt_raw"]
    other_pt = events[other_obj, "pt"]
    other_phi = events[other_obj, "phi"]
    met_pt = events[met_obj].pt
    met_phi = events[met_obj].phi

    # build px and py sums before and after: we sum the time at x and the time at y of each event
    old_px = ak.sum(other_pt_raw * np.cos(other_phi), axis=1)
    old_py = ak.sum(other_pt_raw * np.sin(other_phi), axis=1)
    new_px = ak.sum(other_pt * np.cos(other_phi), axis=1)
    new_py = ak.sum(other_pt * np.sin(other_phi), axis=1)

    # get x and y changes
    delta_x = new_px - old_px
    delta_y = new_py - old_py

    # propagate changes to MET (x, y) components
    met_px = met_pt * np.cos(met_phi) - delta_x
    met_py = met_pt * np.sin(met_phi) - delta_y

    # propagate changes to MET (pT, phi) components
    met_pt = np.sqrt((met_px**2.0 + met_py**2.0))
    met_phi = np.arctan2(met_py, met_px)

    # update MET fields
    events[met_obj, "pt"] = met_pt
    events[met_obj, "phi"] = met_phi
