import correctionlib
import numpy as np
import awkward as ak
from typing import Type
from coffea.analysis_tools import Weights
from analysis.selections.trigger import trigger_match_mask
from analysis.selections.event_selections import get_trigger_mask
from analysis.corrections.met import update_met
from analysis.corrections.utils import get_pog_json, get_egamma_json, unflat_sf


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
            Year of the dataset {2022postEE, 2022preEE, 2023preBPix, 2023postBPix}
        variation:
            syst variation
        id_wp:
            ID working point {wp80iso, wp90iso, wp80noiso, wp90noiso, loose, medium, tight, veto}

    more info: https://twiki.cern.ch/twiki/bin/view/CMS/EgammSFandSSRun3#Scale_factors_and_correction_AN1
    """

    def __init__(
        self,
        events: ak.Array,
        weights: Type[Weights],
        year: str,
        variation: str,
    ) -> None:
        self.events = events
        self.electrons = events.selected_electrons
        self.weights = weights
        self.year = year
        self.variation = variation

        self.flat_electrons = ak.flatten(self.electrons)
        self.electrons_counts = ak.num(self.electrons)

        # set id working points map
        self.id_map = {
            "wp80iso": "wp80iso",
            "wp90iso": "wp90iso",
            "wp80noiso": "wp80noiso",
            "wp90noiso": "wp90noiso",
            "loose": "Loose",
            "medium": "Medium",
            "tight": "Tight",
            "veto": "Veto",
        }
        self.year_map = {
            "2022postEE": "2022Re-recoE+PromptFG",
            "2022preEE": "2022Re-recoBCD",
            "2023preBPix": "2023PromptC",
            "2023postBPix": "2023PromptD",
        }

    def add_id_weights(self, id_wp):
        """
        add electron ID weights to weights container
        """
        nominal_weights = self.get_id_weights(variation="sf", id_wp=id_wp)
        if self.variation == "nominal":
            # get 'up' and 'down' weights
            up_weights = self.get_id_weights(variation="sfup", id_wp=id_wp)
            down_weights = self.get_id_weights(variation="sfdown", id_wp=id_wp)
            # add scale factors to weights container
            self.weights.add(
                name=f"electron_id",
                weight=nominal_weights,
                weightUp=up_weights,
                weightDown=down_weights,
            )
        else:
            self.weights.add(
                name=f"electron_id",
                weight=nominal_weights,
            )

    def add_reco_weights(self, reco_range):
        """
        add electron ID weights to weights container
        """
        nominal_weights = self.get_reco_weights(variation="sf", reco_range=reco_range)
        if self.variation == "nominal":
            # get 'up' and 'down' weights
            up_weights = self.get_reco_weights(variation="sfup", reco_range=reco_range)
            down_weights = self.get_reco_weights(
                variation="sfdown", reco_range=reco_range
            )
            # add scale factors to weights container
            self.weights.add(
                name=f"electron_reco_{reco_range}",
                weight=nominal_weights,
                weightUp=up_weights,
                weightDown=down_weights,
            )
        else:
            self.weights.add(
                name=f"electron_reco_{reco_range}",
                weight=nominal_weights,
            )

    def add_hlt_weights(self, id_wp):
        """
        add electron HLT weights to weights container
        """
        nominal_weights = self.get_hlt_weights(variation="nom", id_wp=id_wp)
        if self.variation == "nominal":
            """
            # get 'up' and 'down' weights
            up_weights = self.get_hlt_weights(
                variation="up",
                hlt_paths=hlt_paths,
                id_wp=id_wp,
            )
            down_weights = self.get_hlt_weights(
                variation="down",
                hlt_paths=hlt_paths,
                id_wp=id_wp,
            )
            """
            # add scale factors to weights container
            self.weights.add(
                name="electron_hlt",
                weight=nominal_weights,
                # weightUp=up_weights,
                # weightDown=down_weights,
            )
        else:
            self.weights.add(
                name="electron_hlt",
                weight=nominal_weights,
            )

    def get_id_weights(self, variation, id_wp):
        """
        Compute electron ID weights

        Parameters:
        -----------
            variation:
                {sf, sfdown, sfup}
        """
        # get electron correction set
        cset = correctionlib.CorrectionSet.from_file(
            get_pog_json(json_name="electron_id", year=self.year)
        )
        # get electrons that pass the id wp, and within SF binning
        electron_pt_mask = self.flat_electrons.pt > 10.0
        in_electron_mask = electron_pt_mask
        in_electrons = self.flat_electrons.mask[in_electron_mask]

        # get electrons pT and abseta (replace None values with some 'in-limit' value)
        electron_pt = ak.fill_none(in_electrons.pt, 15.0)
        electron_eta = ak.fill_none(in_electrons.eta, 0)
        electron_phi = ak.fill_none(in_electrons.phi, 0)

        cset_args = [
            self.year_map[self.year],
            variation,
            self.id_map[id_wp],
            electron_eta,
            electron_pt,
        ]
        if self.year.startswith("2023"):
            cset_args += [electron_phi]
        weights = unflat_sf(
            cset["Electron-ID-SF"].evaluate(*cset_args),
            in_electron_mask,
            self.electrons_counts,
        )
        return weights

    def get_reco_weights(self, variation, reco_range):
        """
        Compute electron Reco weights

        Parameters:
        -----------
            variation:
                {sf, sfdown, sfup}
        """
        # get electron correction set
        cset = correctionlib.CorrectionSet.from_file(
            get_pog_json(json_name="electron_id", year=self.year)
        )
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

        cset_args = [
            self.year_map[self.year],
            variation,
            reco_range,
            electron_eta,
            electron_pt,
        ]
        if self.year.startswith("2023"):
            cset_args += [electron_phi]
        weights = unflat_sf(
            cset["Electron-ID-SF"].evaluate(*cset_args),
            in_electrons_mask,
            self.electrons_counts,
        )
        return weights

    def get_hlt_weights(self, variation, id_wp):
        """
        Compute electron HLT weights

        Parameters:
        -----------
            variation:
                {sf, sfdown, sfup}
        """
        # get electron correction set
        cset = correctionlib.CorrectionSet.from_file(
            get_pog_json(json_name="electron_hlt", year=self.year)
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
        sf_variations_map = {"nom": "sf", "up": "sfup", "down": "sfdown"}

        kind = "single" if ak.all(ak.num(self.electrons) == 1) else "double"
        if kind == "single":
            # for single electorn events, compute SF from POG SF
            sf = cset["Electron-HLT-SF"].evaluate(
                self.year_map[self.year],
                sf_variations_map[variation],
                hlt_path_id_map[id_wp],
                electron_eta,
                electron_pt,
            )
            sf = ak.where(in_electrons_mask, sf, ak.ones_like(sf))
            sf = ak.fill_none(ak.unflatten(sf, self.electrons_counts), value=1)
            nominal_sf = ak.firsts(nominal_sf)

        elif kind == "double":
            # for double electron events, compute SF from electrons' efficiencies
            data_eff = cset["Electron-HLT-DataEff"].evaluate(
                self.year_map[self.year],
                variation,
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
                self.year_map[self.year],
                variation,
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


class ElectronSS:
    """
    Electron Scale and Smearing (energy scale and resolution) corrector

        'The purpose of scale and smearing (more formal: energy scale and resolution corrections) is to correct and calibrate electron and photon energies in data and MC. This step is performed after the MC-based semi-parametric EGamma energy regression, which is applied to both MC and data (aiming to correct for inherent imperfections that are not in principle related to data/MC differences like crystal-by-crystal differences, intermodule gaps, ...). The scale and smearing values are extracted from matching the Zee peaks in data and MC. Usually, energies in data are shifted to the right since, on average, the measured energy is underestimated compared to MC. The energies in MC are smeared out stochastically since the predicted resolution in MC is optimistic'

    more info: https://twiki.cern.ch/twiki/bin/view/CMS/EgammSFandSSRun3#Scale_factors_and_correction_AN1

    coffea0.7 example: https://gitlab.cern.ch/cms-analysis/general/HiggsDNA/-/blob/master/higgs_dna/systematics/electron_systematics.py?ref_type=heads
    """

    def __init__(
        self,
        events: ak.Array,
        year: str,
        variation: str = "nominal",
    ) -> None:
        self.events = events
        self.year = year
        self.variation = variation
        if year.startswith("2022"):
            # get correction set from POG
            self.cset = correctionlib.CorrectionSet.from_file(
                get_pog_json(json_name="electron_ss", year=self.year)
            )
        if year.startswith("2023"):
            # get correction set from EG POG
            self.cset = correctionlib.CorrectionSet.from_file(
                get_egamma_json(year=self.year)
            )
        self.year_mapping = {
            "2022preEE": ["Scale", "Smearing"],
            "2022postEE": ["Scale", "Smearing"],
            "2023preBPix": ["2023PromptC_ScaleJSON", "2023PromptC_SmearingJSON"],
            "2023postBPix": ["2023PromptD_ScaleJSON", "2023PromptD_SmearingJSON"],
        }
        # define Electron pt_raw field (needed for MET recalculation)
        self.events["Electron", "pt_raw"] = (
            ak.ones_like(self.events.Electron.pt) * self.events.Electron.pt
        )
        # select electrons
        self.flat_electrons = ak.flatten(events.Electron)
        self.electrons_counts = ak.num(events.Electron)

    def apply_scale(self):
        """
        from https://twiki.cern.ch/twiki/bin/view/CMS/EgammSFandSSRun3#Scale_factors_and_correction_AN1:

             'Note that we deal with pt instead of energy in the code below. When dealing with nanoAOD in a columnar format,
             this makes sense as pt is an actual field of the photon and electron collections. Scales and smearings are equal
             for pt and energy as they are directly linearly proportional. Changing the pt also automatically changes the energy
             when loading the nanoAOD in the framework below due to Lorentzvector behaviours'.

        """
        # get correction input variables
        gain = self.flat_electrons.seedGain
        run = np.repeat(self.events.run, self.electrons_counts)
        etasc = self.flat_electrons.eta + self.flat_electrons.deltaEtaSC
        r9 = self.flat_electrons.r9
        pt = self.flat_electrons.pt
        # compute scale factor
        scale = self.cset[self.year_mapping[self.year][0]].evaluate(
            "total_correction",
            gain,
            run,
            etasc,
            r9,
            pt,
        )
        # scale is multiplicative correction, unlike smearing, it is deterministic
        if self.variation == "nominal":
            # apply scale correction only to electons with pT > 20 GeV
            corrected_flat_electrons_pt = ak.where(
                self.flat_electrons.pt > 20,
                self.flat_electrons.pt * scale,
                self.flat_electrons.pt,
            )
            self.events["Electron", "pt"] = ak.unflatten(
                corrected_flat_electrons_pt, self.electrons_counts
            )
            # propagate electron pT corrections to MET
            update_met(events=self.events, other_obj="Electron", met_obj="PuppiMET")
        else:
            # uncertainties: TO DO (https://cms-talk.web.cern.ch/t/pnoton-energy-corrections-in-nanoaod-v11/34327/2)
            pass

    def apply_smearing(self, seed=42):
        # get correction input variables
        etasc = self.flat_electrons.eta + self.flat_electrons.deltaEtaSC
        r9 = self.flat_electrons.r9
        # rho does not correspond to the pileup rho energy density,
        # instead it is the standard deviation of the Gaussian used to draw the smearing
        rho = self.cset[self.year_mapping[self.year][1]].evaluate(
            "rho",
            etasc,
            r9,
        )
        if self.variation == "nominal":
            # The smearing is done statistically, so we need some random numbers
            rng = np.random.default_rng(seed=seed)
            smearing = rng.normal(loc=1.0, scale=rho)
            # apply smearing correction only to electons with pT > 20 GeV
            corrected_flat_electrons_pt = ak.where(
                self.flat_electrons.pt > 20,
                self.flat_electrons.pt * smearing,
                self.flat_electrons.pt,
            )
            self.events["Electron", "pt"] = ak.unflatten(
                corrected_flat_electrons_pt, self.electrons_counts
            )
            # propagate electron pT corrections to MET
            update_met(events=self.events, other_obj="Electron", met_obj="PuppiMET")
        else:
            # uncertainties: TO DO
            pass
