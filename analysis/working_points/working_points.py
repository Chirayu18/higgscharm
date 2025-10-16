import numpy as np


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
        # https://indico.cern.ch/event/1304360/contributions/5518916/attachments/2692786/4673101/230731_BTV.pdf
        wps = {
            "2022preEE": {
                "loose": (events.Jet.btagPNetCvB > 0.181)
                & (events.Jet.btagPNetCvL > 0.054),
                "medium": (events.Jet.btagPNetCvB > 0.306)
                & (events.Jet.btagPNetCvL > 0.160),
                "tight": (events.Jet.btagPNetCvB > 0.259)
                & (events.Jet.btagPNetCvL > 0.492),
            },
            "2022postEE": {
                "loose": (events.Jet.btagPNetCvB > 0.182)
                & (events.Jet.btagPNetCvL > 0.054),
                "medium": (events.Jet.btagPNetCvB > 0.304)
                & (events.Jet.btagPNetCvL > 0.160),
                "tight": (events.Jet.btagPNetCvB > 0.258)
                & (events.Jet.btagPNetCvL > 0.491),
            },
            "2023preBPix": {
                "loose": (events.Jet.btagPNetCvB > 0.220)
                & (events.Jet.btagPNetCvL > 0.052),
                "medium": (events.Jet.btagPNetCvB > 0.353)
                & (events.Jet.btagPNetCvL > 0.148),
                "tight": (events.Jet.btagPNetCvB > 0.300)
                & (events.Jet.btagPNetCvL > 0.434),
            },
            "2023postBPix": {
                "loose": (events.Jet.btagPNetCvB > 0.091)
                & (events.Jet.btagPNetCvL > 0.038),
                "medium": (events.Jet.btagPNetCvB > 0.157)
                & (events.Jet.btagPNetCvL > 0.109),
                "tight": (events.Jet.btagPNetCvB > 0.116)
                & (events.Jet.btagPNetCvL > 0.308),
            },
        }
        return wps[year][wp]

    def jet_particlenet_b(self, events, wp, year):
        # https://indico.cern.ch/event/1304360/contributions/5518915/attachments/2692528/4678901/BTagPerf_230808_Summer22WPs.pdf
        wps = {
            "2022preEE": {
                "loose": events.Jet.btagPNetB > 0.0438,
                "medium": events.Jet.btagPNetB > 0.2383,
                "tight": events.Jet.btagPNetB > 0.6939,
                "verytight": events.Jet.btagPNetB > 0.8111,
                "supertight": events.Jet.btagPNetB > 0.9625,
            },
            "2022postEE": {
                "loose": events.Jet.btagPNetB > 0.0458,
                "medium": events.Jet.btagPNetB > 0.2496,
                "tight": events.Jet.btagPNetB > 0.7061,
                "verytight": events.Jet.btagPNetB > 0.8184,
                "supertight": events.Jet.btagPNetB > 0.9649,
            },
            "2023preBPix": {
                "loose": events.Jet.btagPNetB > 0.0479,
                "medium": events.Jet.btagPNetB > 0.2431,
                "tight": events.Jet.btagPNetB > 0.6553,
                "verytight": events.Jet.btagPNetB > 0.7667,
                "supertight": events.Jet.btagPNetB > 0.9459,
            },
            "2023postBPix": {
                "loose": events.Jet.btagPNetB > 0.048,
                "medium": events.Jet.btagPNetB > 0.2435,
                "tight": events.Jet.btagPNetB > 0.6563,
                "verytight": events.Jet.btagPNetB > 0.7671,
                "supertight": events.Jet.btagPNetB > 0.9483,
            },
        }
        return wps[year][wp]
