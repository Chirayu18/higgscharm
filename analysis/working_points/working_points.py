import numpy as np
from analysis.corrections.utils import get_pnet_ctag_mask


class WorkingPoints:

    def jet_id(self, events, wp):
        wps = {
            "tightlepveto": events.Jet.jetId == 6,
            "tight": events.Jet.jetId == 2,
        }
        return wps[wp]

    def electron_id(self, events, wp):
        wps = {
            "wp80iso": events.Electron.mvaIso_WP80,
            "wp90iso": events.Electron.mvaIso_WP90,
            "wp80noiso": events.Electron.mvaNoIso_WP80,
            "wp90noiso": events.Electron.mvaNoIso_WP90,
            "fail": events.Electron.cutBased == 0,
            "veto": events.Electron.cutBased == 1,
            "loose": events.Electron.cutBased == 2,
            "medium": events.Electron.cutBased == 3,
            "tight": events.Electron.cutBased == 4,
            # WP was derived before scale corrections, so the uncorrected pt should be used when available
            "bdt": (
                (
                    (np.abs(events.Electron.eta) < 0.8)
                    & (
                        (
                            (events.Electron.pt > 5)
                            & (events.Electron.pt < 10)
                            & (events.Electron.mvaHZZIso > 0.9044)
                        )
                        | (
                            (events.Electron.pt > 10)
                            & (events.Electron.mvaHZZIso > 0.1969)
                        )
                    )
                )
                | (
                    (np.abs(events.Electron.eta) > 0.8)
                    & (np.abs(events.Electron.eta) < 1.479)
                    & (
                        (
                            (events.Electron.pt > 5)
                            & (events.Electron.pt < 10)
                            & (events.Electron.mvaHZZIso > 0.9094)
                        )
                        | (
                            (events.Electron.pt > 10)
                            & (events.Electron.mvaHZZIso > 0.0759)
                        )
                    )
                )
                | (
                    (np.abs(events.Electron.eta) > 1.479)
                    & (
                        (
                            (events.Electron.pt > 5)
                            & (events.Electron.pt < 10)
                            & (events.Electron.mvaHZZIso > 0.9444)
                        )
                        | (
                            (events.Electron.pt > 10)
                            & (events.Electron.mvaHZZIso > -0.5169)
                        )
                    )
                )
            ),
            "bdt_raw": (
                (
                    (np.abs(events.Electron.eta + events.Electron.deltaEtaSC) < 0.8)
                    & (
                        (
                            (events.Electron.pt_raw > 5)
                            & (events.Electron.pt_raw < 10)
                            & (events.Electron.mvaHZZIso > 1.6339)
                        )
                        | (
                            (events.Electron.pt_raw > 10)
                            & (events.Electron.mvaHZZIso > 0.3685)
                        )
                    )
                )
                | (
                    (np.abs(events.Electron.eta + events.Electron.deltaEtaSC) > 0.8)
                    & (np.abs(events.Electron.eta + events.Electron.deltaEtaSC) < 1.479)
                    & (
                        (
                            (events.Electron.pt_raw > 5)
                            & (events.Electron.pt_raw < 10)
                            & (events.Electron.mvaHZZIso > 1.5499)
                        )
                        | (
                            (events.Electron.pt_raw > 10)
                            & (events.Electron.mvaHZZIso > 0.2662)
                        )
                    )
                )
                | (
                    (np.abs(events.Electron.eta + events.Electron.deltaEtaSC) > 1.479)
                    & (
                        (
                            (events.Electron.pt_raw > 5)
                            & (events.Electron.pt_raw < 10)
                            & (events.Electron.mvaHZZIso > 2.0629)
                        )
                        | (
                            (events.Electron.pt_raw > 10)
                            & (events.Electron.mvaHZZIso > -0.5444)
                        )
                    )
                )
            ),
        }
        return wps[wp]

    def electron_iso(self, events, wp):
        wps = {
            "loose": (
                events.Electron.pfRelIso04_all < 0.25
                if hasattr(events.Electron, "pfRelIso04_all")
                else events.Electron.pfRelIso03_all < 0.25
            ),
            "medium": (
                events.Electron.pfRelIso04_all < 0.20
                if hasattr(events.Electron, "pfRelIso04_all")
                else events.Electron.pfRelIso03_all < 0.20
            ),
            "tight": (
                events.Electron.pfRelIso04_all < 0.15
                if hasattr(events.Electron, "pfRelIso04_all")
                else events.Electron.pfRelIso03_all < 0.15
            ),
        }
        return wps[wp]

    def muon_id(self, events, wp):
        muons_id_wps = {
            "loose": events.Muon.looseId,
            "medium": events.Muon.mediumId,
            "tight": events.Muon.tightId,
        }
        return muons_id_wps[wp]

    def muon_iso(self, events, wp):
        wps = {
            "loose": (
                events.Muon.pfRelIso04_all < 0.25
                if hasattr(events.Muon, "pfRelIso04_all")
                else events.Muon.pfRelIso03_all < 0.25
            ),
            "medium": (
                events.Muon.pfRelIso04_all < 0.20
                if hasattr(events.Muon, "pfRelIso04_all")
                else events.Muon.pfRelIso03_all < 0.20
            ),
            "tight": (
                events.Muon.pfRelIso04_all < 0.15
                if hasattr(events.Muon, "pfRelIso04_all")
                else events.Muon.pfRelIso03_all < 0.15
            ),
        }
        return wps[wp]

    def jet_particlenet_c(self, events, wp, year):
        return get_pnet_ctag_mask(jets=events.Jet, wp=wp, year=year)
