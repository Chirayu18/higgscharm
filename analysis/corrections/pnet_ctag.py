import re
import json
import glob
import pathlib
import correctionlib
import numpy as np
import awkward as ak
import importlib.resources
from coffea import util
from typing import Type
from coffea.analysis_tools import Weights
from analysis.corrections.utils import get_btv_json, get_pnet_ctag_mask


class CTagCorrector:
    """
    CTag corrector class.

    Parameters:
    -----------
        events:
            Events collection
        weights:
            Weights container from coffea.analysis_tools
        worging_point:
            worging point {loose, medium, tight}
        year:
            dataset year {2022preEE, 2022postEE, 2023preBPix, 2023PostBPix}
        variation:
            if 'nominal' (default) add 'nominal', 'up' and 'down' variations to weights container
    """

    def __init__(
        self,
        events,
        weights: Type[Weights],
        worging_point: str,
        year: str,
        variation: str = "nominal",
    ) -> None:
        self._year = year
        self._wp = worging_point
        self._weights = weights
        self._variation = variation

        # check available ctag SFs
        self._wp_map = {"tight": "T", "medium": "M", "loose": "L"}
        if self._wp not in self._wp_map:
            raise ValueError(
                f"There are no available c-tag SFs for the working point. Please specify {list(self._wp_map.keys())}"
            )

        # load c-tagging efficiency look-up table
        with importlib.resources.path(
            "analysis.data",
            f"ctag_pnet_eff_{self._wp}_{self._year}.coffea",
        ) as filename:
            self._efflookup = util.load(str(filename))

        # load correction set
        self._cset = correctionlib.CorrectionSet.from_file(
            get_btv_json(json_name="ctag", year=year)
        )
        # select b, c and light jets
        self._flavors = {"b": 5, "c": 4, "light": 0}
        self._c_jets = events.selected_jets[
            events.selected_jets.hadronFlavour == self._flavors["c"]
        ]
        self._b_jets = events.selected_jets[
            events.selected_jets.hadronFlavour == self._flavors["b"]
        ]
        self._light_jets = events.selected_jets[
            events.selected_jets.hadronFlavour == self._flavors["light"]
        ]
        self._jet_map = {
            "c": self._c_jets,
            "b": self._b_jets,
            "light": self._light_jets,
        }
        self._jet_pass_ctag = {
            "c": get_pnet_ctag_mask(self._jet_map["c"], self._wp, self._year),
            "b": get_pnet_ctag_mask(self._jet_map["b"], self._wp, self._year),
            "light": get_pnet_ctag_mask(self._jet_map["light"], self._wp, self._year),
        }
        self.var_naming_map = {
            "c": "CMS_ctag_c",
            "b": "CMS_ctag_b",
            "light": "CMS_ctag_light",
        }

    def add_ctag_weights(self, flavor: str) -> None:
        """
        Add c-tagging weights (nominal, up and down) to weights container for b, c or light jets
        """
        # efficiencies
        eff = self.efficiency(flavor=flavor)

        # mask with events that pass the ctag working point
        pass_ctag = self._jet_pass_ctag[flavor]

        # nominal scale factors
        ctag_sf = self.get_scale_factors(flavor=flavor, syst="central")

        # nominal weights
        ctag_weight = self.get_ctag_weight(eff, ctag_sf, pass_ctag)

        if self._variation == "nominal":
            # systematics
            # up and down scale factors
            ctag_sf_up = self.get_scale_factors(flavor=flavor, syst="up")
            ctag_sf_down = self.get_scale_factors(flavor=flavor, syst="down")
            ctag_weight_up = self.get_ctag_weight(eff, ctag_sf_up, pass_ctag)
            ctag_weight_down = self.get_ctag_weight(eff, ctag_sf_down, pass_ctag)
            # add weights to Weights container
            self._weights.add(
                name=self.var_naming_map[flavor],
                weight=ctag_weight,
                weightUp=ctag_weight_up,
                weightDown=ctag_weight_down,
            )
        else:
            self._weights.add(
                name=self.var_naming_map[flavor],
                weight=ctag_weight,
            )

    def efficiency(self, flavor: str, fill_value=1) -> ak.Array:
        """compute the c-tagging efficiency"""
        return self._efflookup(
            self._jet_map[flavor].pt,
            np.abs(self._jet_map[flavor].eta),
            self._jet_map[flavor].hadronFlavour,
        )

    def get_scale_factors(self, flavor: str, syst="central", fill_value=1) -> ak.Array:
        """
        compute c-taggging scale factors
        """
        return self.get_sf(flavor=flavor, syst=syst)

    def get_sf(self, flavor: str, syst: str = "central") -> ak.Array:
        """
        compute the scale factors for b, c or light jets

        Parameters:
        -----------
            flavor:
                hadron flavor {b, c, light}
            syst:
                Name of the systematic {central, up, down}
        """
        cset_keys = {
            "b": "particleNet_tnp",
            "c": "particleNet_wc",
            "light": "particleNet_light",
        }
        # until correctionlib handles jagged data natively we have to flatten and unflatten
        j, nj = ak.flatten(self._jet_map[flavor]), ak.num(self._jet_map[flavor])

        # get 'in-limits' jets
        jet_eta_mask = np.abs(j.eta) < 2.4
        in_jet_mask = jet_eta_mask
        in_jets = j.mask[in_jet_mask]

        # get jet transverse momentum, abs pseudorapidity and hadron flavour (replace None values with some 'in-limit' value)
        jets_pt = ak.fill_none(in_jets.pt, 0.0)
        jets_eta = ak.fill_none(np.abs(in_jets.eta), 0.0)
        jets_hadron_flavour = ak.fill_none(in_jets.hadronFlavour, self._flavors[flavor])
        if flavor == "c":
            if syst != "central":
                syst += "_stat"
        sf = self._cset[cset_keys[flavor]].evaluate(
            syst,
            self._wp_map[self._wp],
            np.array(jets_hadron_flavour),
            np.array(jets_eta),
            np.array(jets_pt),
        )
        sf = ak.where(in_jet_mask, sf, ak.ones_like(sf))
        return ak.unflatten(sf, nj)

    @staticmethod
    def get_ctag_weight(eff: ak.Array, sf: ak.Array, pass_ctag: ak.Array) -> ak.Array:
        """
        compute c-tagging weights

        Parameters:
        -----------
            eff:
                c-tagging efficiencies
            sf:
                jet's scale factors
            pass_ctag:
                mask with jets that pass the c-tagging working point
        """
        # tagged SF = SF * eff / eff = SF
        tagged_sf = ak.prod(sf.mask[pass_ctag], axis=-1)

        # untagged SF = (1 - SF * eff) / (1 - eff)
        untagged_sf = ak.prod(((1 - sf * eff) / (1 - eff)).mask[~pass_ctag], axis=-1)

        return ak.fill_none(tagged_sf * untagged_sf, 1.0)
