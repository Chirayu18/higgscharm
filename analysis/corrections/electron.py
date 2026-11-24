import correctionlib
import numpy as np
import awkward as ak
from typing import Type
from coffea.analysis_tools import Weights
from analysis.filesets.utils import get_nano_version
from analysis.selections.event_selections import get_trigger_mask
from analysis.corrections.utils import unflat_sf
from analysis.corrections.correctionlib_files import correction_files


class ElectronWeights:
    """
    Electron ID, Reco and HLT weights class

    Parameters:
    -----------
        events:
            events collection
        weights:
            Weights container
        year:
            Year of the dataset {2016preVFP, 2016postVFP, 2017, 2018, 2022preEE, 2022postEE, 2023preBPix, 2023postBPix, 2024}
        variation:
            syst variation
        id_wp:
            ID working point {wp80iso, wp90iso}

    more info: https://twiki.cern.ch/twiki/bin/view/CMS/EgammSFandSSRun3#Scale_factors_and_correction_AN1
    """

    def __init__(
        self,
        events: ak.Array,
        weights: Type[Weights],
        year: str,
        shift: str,
    ) -> None:
        self.events = events
        self.electrons = events.selected_electrons
        self.weights = weights
        self.year = year
        self.shift = shift
        self.nano_version = get_nano_version(year)

        self.flat_electrons = ak.flatten(self.electrons)
        self.electrons_counts = ak.num(self.electrons)

        # get correction set for Run2/Run3 ID and Reco
        self.cset_id = correctionlib.CorrectionSet.from_file(
            correction_files["electron_id"][year]
        )
        self.cset_reco = correctionlib.CorrectionSet.from_file(
            correction_files["electron_reco"][year]
        )

    def add_id_weights(self, id_wp):
        """
        add electron ID weights to weights container
        """
        if self.nano_version in ["12", "15"]:
            id_weight_func = self.get_id_weights_run3
        elif self.nano_version == "9":
            id_weight_func = self.get_id_weights_run2

        nominal_weights = id_weight_func(syst="sf", id_wp=id_wp)

        if self.shift is None:
            up_weights = id_weight_func(syst="sfup", id_wp=id_wp)
            down_weights = id_weight_func(syst="sfdown", id_wp=id_wp)
            # add scale factors to weights container
            self.weights.add(
                name=f"CMS_eff_e_id_{self.year[:4]}",
                weight=nominal_weights,
                weightUp=up_weights,
                weightDown=down_weights,
            )
        else:
            self.weights.add(
                name=f"CMS_eff_e_id_{self.year[:4]}",
                weight=nominal_weights,
            )

    # 2026-11-05 (tvl/MVA port): retrained ttH lepton-MVA working-point SFs from the HWW/latinos electron JSON
    PROMPTMVA_SF_NAMES = {
        ("ttHMVA_Run3", "Medium"): "cut_MediumID_tthMVA_Run3",
        ("ttHMVA_HWW", "Medium"): "cut_MediumID_tthMVA_HWW",
        ("ttHMVA_Run3", "wp90iso"): "cut_mvaWinter22V2Iso_WP90_tthMVA_Run3",
        ("ttHMVA_HWW", "wp90iso"): "cut_mvaWinter22V2Iso_WP90_tthMVA_HWW",
    }

    def add_promptMVA_weights(self, id_wp, promptMVA_wp):
        """Nuisance CMS_eff_e_promptMVA_<year>. Run-3 nanoAODv12 eras 2022-2023 only."""
        if self.nano_version != "12":
            raise ValueError(f"electron promptMVA SFs exist for nanoAODv12 (2022-2023) only, got v{self.nano_version}")
        nominal_weights = self.get_promptMVA_weights_run3("sf", id_wp, promptMVA_wp)
        if self.shift is None:
            self.weights.add(
                name=f"CMS_eff_e_promptMVA_{self.year[:4]}",
                weight=nominal_weights,
                weightUp=self.get_promptMVA_weights_run3("sfup", id_wp, promptMVA_wp),
                weightDown=self.get_promptMVA_weights_run3("sfdown", id_wp, promptMVA_wp),
            )
        else:
            self.weights.add(
                name=f"CMS_eff_e_promptMVA_{self.year[:4]}",
                weight=nominal_weights,
            )

    def get_promptMVA_weights_run3(self, syst, id_wp, promptMVA_wp):
        """
        Binning: supercluster eta in [-2.5, 2.5], pt in [10, 200) (flow=error); electrons outside get SF 1.
        Unlike Thomas's version no phi argument is passed for 2023 (these JSONs have no phi input), and the
        supercluster eta (eta + deltaEtaSC) is used as the JSON's eta input is documented as supercluster eta.
        """
        key = (promptMVA_wp, id_wp)
        if key not in self.PROMPTMVA_SF_NAMES:
            raise ValueError(f"no electron promptMVA SF for promptMVA/id = {promptMVA_wp}/{id_wp}")
        if not hasattr(self, "cset_promptMVA"):
            self.cset_promptMVA = correctionlib.CorrectionSet.from_file(correction_files["electron_HWW"][self.year])
        sc_eta = self.flat_electrons.eta + self.flat_electrons.deltaEtaSC
        in_electron_mask = (
            (self.flat_electrons.pt > 10.0) & (self.flat_electrons.pt < 200.0) & (np.abs(sc_eta) < 2.5)
        )
        in_electrons = self.flat_electrons.mask[in_electron_mask]
        electron_pt = ak.fill_none(in_electrons.pt, 10.0)
        electron_eta = ak.fill_none(in_electrons.eta + in_electrons.deltaEtaSC, 0.0)
        year_map = {
            "2022postEE": "2022Re-recoE+PromptFG",
            "2022preEE": "2022Re-recoBCD",
            "2023preBPix": "2023PromptC",
            "2023postBPix": "2023PromptD",
        }
        return unflat_sf(
            self.cset_promptMVA["Electron-ID-SF"].evaluate(
                year_map[self.year], syst, self.PROMPTMVA_SF_NAMES[key], electron_eta, electron_pt
            ),
            in_electron_mask,
            self.electrons_counts,
        )

    def add_reco_weights(self, reco_range):
        """
        add electron Reco weights to weights container
        """
        var_naming_map = {
            "RecoBelow20": f"CMS_eff_e_reco_below20_{self.year[:4]}",
            "RecoAbove20": f"CMS_eff_e_reco_above20_{self.year[:4]}",
            "Reco20to75": f"CMS_eff_e_reco_20to75_{self.year[:4]}",
            "RecoAbove75": f"CMS_eff_e_reco_above75_{self.year[:4]}",
        }
        if self.nano_version in ["12", "15"]:
            reco_weight_func = self.get_reco_weights_run3
        elif self.nano_version == "9":
            reco_weight_func = self.get_reco_weights_run2

        nominal_weights = reco_weight_func(syst="sf", reco_range=reco_range)

        if self.shift is None:
            up_weights = reco_weight_func(syst="sfup", reco_range=reco_range)
            down_weights = reco_weight_func(syst="sfdown", reco_range=reco_range)
            # add scale factors to weights container
            self.weights.add(
                name=var_naming_map[reco_range],
                weight=nominal_weights,
                weightUp=up_weights,
                weightDown=down_weights,
            )
        else:
            self.weights.add(
                name=var_naming_map[reco_range],
                weight=nominal_weights,
            )

    def add_hlt_weights(self, id_wp):
        """
        add electron HLT weights to weights container
        """
        if self.nano_version in ["12", "15"]:
            nominal_weights = self.get_hlt_weights_run3(syst="nom", id_wp=id_wp)
        elif self.nano_version == "9":
            nominal_weights = self.get_hlt_weights_run2(syst="nom", id_wp=id_wp)
        if self.shift is None:
            # add scale factors to weights container
            self.weights.add(
                name=f"CMS_eff_e_trigger_{self.year[:4]}",
                weight=nominal_weights,
                # weightUp=up_weights,
                # weightDown=down_weights,
            )
        else:
            self.weights.add(
                name=f"CMS_eff_e_trigger_{self.year[:4]}",
                weight=nominal_weights,
            )

    def get_id_weights_run3(self, syst, id_wp):
        """Compute electron ID weights for Run3 datasets"""
        # get electrons that pass the id wp, and within SF binning
        if self.nano_version != "15":
            electron_pt_mask = self.flat_electrons.pt > 10.0
            electron_eta_mask = ak.ones_like(electron_pt_mask, dtype=bool)
        else:
            electron_pt_mask = (self.flat_electrons.pt > 20.0) & (
                self.flat_electrons.pt < 1000.0
            )
            electron_eta_mask = (
                np.abs(self.flat_electrons.eta + self.flat_electrons.deltaEtaSC) < 2.5
            )

        in_electron_mask = electron_pt_mask & electron_eta_mask
        in_electrons = self.flat_electrons.mask[in_electron_mask]

        # get electrons pT and abseta (replace None values with some 'in-limit' value)
        electron_pt = ak.fill_none(in_electrons.pt, 20.0)
        electron_eta = ak.fill_none(in_electrons.eta, 0)
        electron_phi = ak.fill_none(in_electrons.phi, 0)

        year_map = {
            "2022postEE": "2022Re-recoE+PromptFG",
            "2022preEE": "2022Re-recoBCD",
            "2023preBPix": "2023PromptC",
            "2023postBPix": "2023PromptD",
        }
        cset_args = [
            year_map.get(self.year, self.year),
            syst,
            id_wp,
            electron_eta,
            electron_pt,
        ]
        if self.year.startswith("2023"):
            cset_args += [electron_phi]
        weights = unflat_sf(
            self.cset_id["Electron-ID-SF"].evaluate(*cset_args),
            in_electron_mask,
            self.electrons_counts,
        )
        return weights

    def get_id_weights_run2(self, syst, id_wp):
        """Compute electron ID weights for Run2 datasets"""
        # get electrons that pass the id wp, and within SF binning
        electron_pt_mask = (self.flat_electrons.pt > 10.0) & (
            self.flat_electrons.pt < 499.999
        )  # potential problems with pt > 500 GeV
        in_electron_mask = electron_pt_mask
        in_electrons = self.flat_electrons.mask[in_electron_mask]

        # get electrons pT and abseta (replace None values with some 'in-limit' value)
        electron_pt = ak.fill_none(in_electrons.pt, 15.0)
        electron_eta = ak.fill_none(in_electrons.eta + in_electrons.deltaEtaSC, 0)
        electron_phi = ak.fill_none(in_electrons.phi, 0)

        cset_args = [
            self.year,
            syst,
            id_wp,
            electron_eta,
            electron_pt,
        ]
        if self.year.startswith("2023"):
            cset_args += [electron_phi]
        weights = unflat_sf(
            self.cset_id["UL-Electron-ID-SF"].evaluate(*cset_args),
            in_electron_mask,
            self.electrons_counts,
        )
        return weights

    def get_reco_weights_run3(self, syst, reco_range):
        """Compute electron Reco weights for Run3 datasets"""
        # get electrons that pass the id wp, and within SF binning
        electron_pt_mask = {
            "RecoBelow20": (self.flat_electrons.pt > 10.0)
            & (self.flat_electrons.pt < 20.0),
            "Reco20to75": (self.flat_electrons.pt > 20.0)
            & (self.flat_electrons.pt < 75.0),
            "RecoAbove75": self.flat_electrons.pt > 75,
        }
        in_electrons_mask = electron_pt_mask[reco_range]
        in_electrons = self.flat_electrons.mask[in_electrons_mask]

        # get electrons pT and abseta (replace None values with some 'in-limit' value)
        electron_pt_limits = {
            "RecoBelow20": 15,
            "Reco20to75": 30,
            "RecoAbove75": 80,
        }
        electron_pt = ak.fill_none(in_electrons.pt, electron_pt_limits[reco_range])
        electron_eta = ak.fill_none(in_electrons.eta, 0)
        electron_phi = ak.fill_none(in_electrons.phi, 0)

        year_map = {
            "2022postEE": "2022Re-recoE+PromptFG",
            "2022preEE": "2022Re-recoBCD",
            "2023preBPix": "2023PromptC",
            "2023postBPix": "2023PromptD",
            "2024": "2024Prompt",
        }
        cset_args = [
            year_map.get(self.year, self.year),
            syst,
            reco_range,
            electron_eta,
            electron_pt,
        ]
        if self.year.startswith("2023"):
            cset_args += [electron_phi]
        weights = unflat_sf(
            self.cset_reco["Electron-ID-SF"].evaluate(*cset_args),
            in_electrons_mask,
            self.electrons_counts,
        )
        return weights

    def get_reco_weights_run2(self, syst, reco_range):
        """Compute electron Reco weights for Run2 datasets"""
        # get electrons that pass the id wp, and within SF binning
        electron_pt_mask = {
            "RecoAbove20": (self.flat_electrons.pt > 20)
            & (self.flat_electrons.pt < 499.999),
            "RecoBelow20": (self.flat_electrons.pt > 10)
            & (self.flat_electrons.pt < 20),
        }
        in_electrons_mask = electron_pt_mask[reco_range]
        in_electrons = self.flat_electrons.mask[in_electrons_mask]

        # get electrons pT and abseta (replace None values with some 'in-limit' value)
        electrons_pt_limits = {
            "RecoAbove20": 21,
            "RecoBelow20": 15,
            "Reco20to75": 30,
            "RecoAbove75": 80,
        }
        electron_pt = ak.fill_none(in_electrons.pt, electrons_pt_limits[reco_range])
        electron_eta = ak.fill_none(in_electrons.eta + in_electrons.deltaEtaSC, 0)
        electron_phi = ak.fill_none(in_electrons.phi, 0)

        cset_args = [
            self.year,
            syst,
            reco_range,
            electron_eta,
            electron_pt,
        ]
        weights = unflat_sf(
            self.cset_reco["UL-Electron-ID-SF"].evaluate(*cset_args),
            in_electrons_mask,
            self.electrons_counts,
        )
        return weights

    def get_hlt_weights_run3(self, syst, id_wp):
        """Compute electron HLT weights for Run3 datasets"""
        # get correction set
        cset = correctionlib.CorrectionSet.from_file(
            correction_files["electron_hlt"][self.year]
        )
        # get electrons that pass the id wp, and within SF binning
        electron_pt_mask = self.flat_electrons.pt > 25.0

        in_electrons_mask = electron_pt_mask
        in_electrons = self.flat_electrons.mask[in_electrons_mask]

        # get electrons pT and abseta (replace None values with some 'in-limit' value)
        electron_pt = ak.fill_none(in_electrons.pt, 25)
        electron_eta = ak.fill_none(in_electrons.eta, 0)

        hlt_path_id_map = {
            "wp80iso": "HLT_SF_Ele30_MVAiso80ID",
            "wp90iso": "HLT_SF_Ele30_MVAiso90ID",
        }
        year_map = {
            "2022postEE": "2022Re-recoE+PromptFG",
            "2022preEE": "2022Re-recoBCD",
            "2023preBPix": "2023PromptC",
            "2023postBPix": "2023PromptD",
            "2024": "2024Prompt",
        }
        sf_syst_map = {"nom": "sf", "up": "sfup", "down": "sfdown"}

        kind = "single" if ak.all(ak.num(self.electrons) == 1) else "double"
        if kind == "single":
            # for single electron events, compute SF from POG SF
            sf = cset["Electron-HLT-SF"].evaluate(
                year_map[self.year],
                sf_syst_map[syst],
                hlt_path_id_map[id_wp],
                electron_eta,
                electron_pt,
            )
            sf = ak.where(in_electrons_mask, sf, ak.ones_like(sf))
            sf = ak.fill_none(ak.unflatten(sf, self.electrons_counts), value=1)
            nominal_sf = ak.firsts(sf)

        elif kind == "double":
            # for double electron events, compute SF from electrons' efficiencies
            data_eff = cset["Electron-HLT-DataEff"].evaluate(
                year_map[self.year],
                syst,
                hlt_path_id_map[id_wp],
                electron_eta,
                electron_pt,
            )
            data_eff = ak.where(in_electrons_mask, data_eff, ak.ones_like(data_eff))
            data_eff = ak.unflatten(data_eff, self.electrons_counts)
            data_eff_leading = ak.firsts(data_eff)
            data_eff_subleading = ak.pad_none(data_eff, target=2)[:, 1]
            full_data_eff = (
                data_eff_leading
                + data_eff_subleading
                - data_eff_leading * data_eff_subleading
            )
            full_data_eff = ak.fill_none(full_data_eff, 1)

            mc_eff = cset["Electron-HLT-McEff"].evaluate(
                year_map[self.year],
                syst,
                hlt_path_id_map[id_wp],
                electron_eta,
                electron_pt,
            )
            mc_eff = ak.where(in_electrons_mask, mc_eff, ak.ones_like(mc_eff))
            mc_eff = ak.unflatten(mc_eff, self.electrons_counts)
            mc_eff_leading = ak.firsts(mc_eff)
            mc_eff_subleading = ak.pad_none(mc_eff, target=2)[:, 1]
            full_mc_eff = (
                mc_eff_leading + mc_eff_subleading - mc_eff_leading * mc_eff_subleading
            )
            full_mc_eff = ak.fill_none(full_mc_eff, 1)

            # compute SF from efficiencies
            nominal_sf = full_data_eff / full_mc_eff

        return nominal_sf

    def get_hlt_weights_run2(self, syst, id_wp):
        """Compute electron HLT weights for Run2 datasets"""
        hlt_paths = {
            "2016preVFP": "Ele27_WPTight_Gsf_OR_Photon175",
            "2016postVFP": "Ele27_WPTight_Gsf_OR_Photon175",
            "2017": "Ele35_WPTight_Gsf_OR_Photon200",
            "2018": "Ele32_WPTight_Gsf",
        }
        hlt_path_id_map = {
            "wp80iso": "HLT_SF_Ele30_MVAiso80ID",
            "wp90iso": "HLT_SF_Ele30_MVAiso90ID",
        }
        pt_limits = {
            "2016preVFP": 27.0,
            "2016postVFP": 27.0,
            "2017": 35.0,
            "2018": 32.0,
        }
        electron_pt_mask = (self.flat_electrons.pt > pt_limits[self.year]) & (
            self.flat_electrons.pt < 499.9
        )
        eta_mask = (
            np.abs(self.flat_electrons.eta + self.flat_electrons.deltaEtaSC) < 2.5
        )
        in_electrons_mask = electron_pt_mask & eta_mask
        in_electrons = self.flat_electrons.mask[in_electrons_mask]

        # get electrons pT and abseta (replace None values with some 'in-limit' value)
        electron_pt = ak.fill_none(in_electrons.pt, 40)
        electron_eta = ak.fill_none(
            self.flat_electrons.eta + self.flat_electrons.deltaEtaSC, 0
        )
        # check whether there are single or double electron events
        kind = "single" if ak.all(ak.num(self.electrons) == 1) else "double"

        if kind == "single":
            cset = correctionlib.CorrectionSet.from_file(
                correction_files["electron_hlt"][self.year]
            )
            sf = cset[f"HLT_SF_{hlt_paths[self.year]}_MVAiso80ID"].evaluate(
                electron_eta, electron_pt
            )
            sf = ak.where(in_electrons_mask, sf, ak.ones_like(sf))
            sf = ak.fill_none(ak.unflatten(sf, self.electrons_counts), value=1)
            nominal_sf = ak.firsts(sf)

        elif kind == "double":
            cset_data_eff = correctionlib.CorrectionSet.from_file(
                correction_files["electron_hlt_data_eff"][self.year]
            )
            data_eff = cset_data_eff[
                f"HLT_DataEff_{hlt_paths[self.year]}_MVAiso80ID"
            ].evaluate(electron_eta, electron_pt)
            data_eff = ak.where(in_electrons_mask, data_eff, ak.ones_like(data_eff))
            data_eff = ak.unflatten(data_eff, self.electrons_counts)
            data_eff_leading = ak.firsts(data_eff)
            data_eff_subleading = ak.pad_none(data_eff, target=2)[:, 1]
            full_data_eff = (
                data_eff_leading
                + data_eff_subleading
                - data_eff_leading * data_eff_subleading
            )
            full_data_eff = ak.fill_none(full_data_eff, 1)

            cset_mc_eff = correctionlib.CorrectionSet.from_file(
                correction_files["electron_hlt_mc_eff"][self.year]
            )
            mc_eff = cset_mc_eff[
                f"HLT_MCEff_{hlt_paths[self.year]}_MVAiso80ID"
            ].evaluate(electron_eta, electron_pt)
            mc_eff = ak.where(in_electrons_mask, mc_eff, ak.ones_like(mc_eff))
            mc_eff = ak.unflatten(mc_eff, self.electrons_counts)
            mc_eff_leading = ak.firsts(mc_eff)
            mc_eff_subleading = ak.pad_none(mc_eff, target=2)[:, 1]
            full_mc_eff = (
                mc_eff_leading + mc_eff_subleading - mc_eff_leading * mc_eff_subleading
            )
            full_mc_eff = ak.fill_none(full_mc_eff, 1)

            nominal_sf = full_data_eff / full_mc_eff

        return nominal_sf
