import numba
import numpy as np
import awkward as ak
from coffea.nanoevents.methods import candidate


def delta_r_higher(first, second, threshold=0.4):
    # select objects from 'first' which are at least 'threshold' away from all objects in 'second'.
    mval = first.metric_table(second)
    return ak.all(mval > threshold, axis=-1)


def delta_r_lower(first, second, threshold=0.4):
    # select objects from 'first' which are at least 'threshold' within from all objects in 'second'.
    mval = first.metric_table(second)
    return ak.all(mval <= threshold, axis=-1)


def select_dileptons(objects, key):
    leptons = ak.zip(
        {
            "pt": objects[key].pt,
            "eta": objects[key].eta,
            "phi": objects[key].phi,
            "mass": objects[key].mass,
            "charge": objects[key].charge,
        },
        with_name="PtEtaPhiMCandidate",
        behavior=candidate.behavior,
    )
    # create pair combinations with all muons
    dileptons = ak.combinations(leptons, 2, fields=["l1", "l2"])
    dileptons = dileptons[ak.argsort(dileptons.l1.pt, axis=1)]
    # add dimuon 4-momentum field
    dileptons["p4"] = dileptons.l1 + dileptons.l2
    dileptons["pt"] = dileptons.p4.pt
    return dileptons


def transverse_mass(lepton, met):
    return np.sqrt(
        2.0
        * lepton.pt
        * met.pt
        * (ak.ones_like(met.pt) - np.cos(lepton.delta_phi(met)))
    )


def get_closest_lepton(fsr, lepton, axis=1):
    mval, (a, b) = fsr.metric_table(lepton, axis, return_combinations=True)
    mval = ak.where(b.is_relaxed, mval, np.inf)
    mmin = ak.argmin(mval, axis=axis + 1, keepdims=True)
    out = ak.firsts(b[mmin], axis=axis + 1)
    dR = ak.firsts(mval[mmin], axis=axis + 1)
    dREt2 = dR / fsr.pt**2
    mask = (dR < 0.5) & (dR > 0.001) & (dREt2 < 0.012)
    return ak.mask(out, mask), dREt2


def assign_lepton_fsr_idx(fsr_photons, leptons):
    leptons["fsr_idx"] = ak.full_like(leptons.pt, -1)
    # compare elements within each sublist using broadcasting
    mask = ak.any(leptons.idx[:, :, None] == fsr_photons.lepton_idx[:, None, :], axis=2)
    # identify the correct positions
    idx_positions = ak.argmax(
        leptons.idx[:, :, None] == fsr_photons.lepton_idx[:, None, :], axis=2
    )
    # create the updated fsr_idx for leptons
    idx_updated = ak.where(mask, fsr_photons.idx[idx_positions], leptons.fsr_idx)
    return ak.with_field(leptons, idx_updated, "fsr_idx")


def build_zcand(z):
    z_fields = {
        "l1": z.l1,
        "l2": z.l2,
        "p4": z.l1.p4 + z.l2.p4,
        "idx": z.idx,
        "is_ossf": z.is_ossf,
        "is_sr": z.is_sr,
    }
    if hasattr(z, "is_ss"):
        z_fields.update({"is_ss": z.is_ss})
    if hasattr(z, "is_1fcr"):
        z_fields.update({"is_1fcr": z.is_1fcr})
    if hasattr(z, "is_2fcr"):
        z_fields.update({"is_2fcr": z.is_2fcr})
    if hasattr(z, "is_sscr"):
        z_fields.update({"is_sscr": z.is_sscr})
    return ak.zip(z_fields)


def fourlepcand(z1, z2):
    """return 4vector for a 4lepton candidate adding a 'p4' field using 'dressed' leptons"""
    return ak.zip(
        {
            "z1": build_zcand(z1),
            "z2": build_zcand(z2),
        }
    )


def make_cand(zcand, kind, sort_by_mass=True, os_method=True):
    """build ZZ or ZLL candidates in a Higgs phase space"""
    if kind == "zz":
        cand = ak.combinations(zcand, 2, fields=["z1", "z2"])
        cand = fourlepcand(cand.z1, cand.z2)
    elif kind == "zll":
        cand = ak.cartesian({"z1": zcand, "z2": zcand})
        cand = fourlepcand(cand.z1, cand.z2)

    # check that the Zs are mutually exclusive (not sharing the same lepton)
    share_same_lepton_mask = (
        (cand.z1.l1.idx == cand.z2.l1.idx)
        | (cand.z1.l2.idx == cand.z2.l2.idx)
        | (cand.z1.l2.idx == cand.z2.l1.idx)
        | (cand.z1.l2.idx == cand.z2.l2.idx)
    )
    cand = cand[~share_same_lepton_mask]

    if sort_by_mass:
        # sort ZZ(ZLL) candidates by they proximity to the Z mass
        zmass = 91.1876
        dist_from_z1_to_zmass = np.abs(cand.z1.p4.mass - zmass)
        dist_from_z2_to_zmass = np.abs(cand.z2.p4.mass - zmass)
        z1 = ak.where(
            dist_from_z1_to_zmass > dist_from_z2_to_zmass,
            cand.z2,
            cand.z1,
        )
        z2 = ak.where(
            dist_from_z1_to_zmass < dist_from_z2_to_zmass,
            cand.z2,
            cand.z1,
        )
        cand = fourlepcand(z1, z2)

    # chech that Z1 mass > 40 GeV
    z1_mass_g40_mask = cand.z1.p4.mass > 40

    # ghost removal: ∆R(η, φ) > 0.02 between each of the four leptons (to protect against split tracks)
    ghost_removal_mask = (
        (cand.z1.l1.delta_r(cand.z1.l2) > 0.02)
        & (cand.z1.l1.delta_r(cand.z2.l1) > 0.02)
        & (cand.z1.l1.delta_r(cand.z2.l2) > 0.02)
        & (cand.z1.l2.delta_r(cand.z2.l1) > 0.02)
        & (cand.z1.l2.delta_r(cand.z2.l2) > 0.02)
        & (cand.z2.l1.delta_r(cand.z2.l2) > 0.02)
    )
    # trigger acceptance: two of the four selected leptons should pass pT,i > 20 GeV and pT,j > 10 (FSR photons are used)
    trigger_acceptance_mask = (
        ((cand.z1.l1.p4.pt > 20) & (cand.z1.l2.p4.pt > 10))
        | ((cand.z1.l1.p4.pt > 20) & (cand.z2.l1.p4.pt > 10))
        | ((cand.z1.l1.p4.pt > 20) & (cand.z2.l2.p4.pt > 10))
        | ((cand.z1.l2.p4.pt > 20) & (cand.z1.l1.p4.pt > 10))
        | ((cand.z1.l2.p4.pt > 20) & (cand.z2.l1.p4.pt > 10))
        | ((cand.z1.l2.p4.pt > 20) & (cand.z2.l2.p4.pt > 10))
        | ((cand.z2.l1.p4.pt > 20) & (cand.z1.l1.p4.pt > 10))
        | ((cand.z2.l1.p4.pt > 20) & (cand.z1.l2.p4.pt > 10))
        | ((cand.z2.l1.p4.pt > 20) & (cand.z2.l2.p4.pt > 10))
        | ((cand.z2.l2.p4.pt > 20) & (cand.z1.l1.p4.pt > 10))
        | ((cand.z2.l2.p4.pt > 20) & (cand.z1.l2.p4.pt > 10))
        | ((cand.z2.l2.p4.pt > 20) & (cand.z2.l1.p4.pt > 10))
    )
    # QCD suppression: all four opposite-sign pairs that can be built with the four leptons (regardless of lepton flavor) must satisfy m > 4 GeV
    # FSR photons are not used since a QCD-induced low mass dilepton (eg. Jpsi) may have photons nearby (e.g. from π0).
    qcd_suppression_mask = (
        ((cand.z1.l1 + cand.z1.l2).mass > 4)
        & ((cand.z2.l1 + cand.z2.l2).mass > 4)
        & (
            (cand.z1.l1.charge + cand.z2.l1.charge != 0)
            | (
                (cand.z1.l1.charge + cand.z2.l1.charge == 0)
                & ((cand.z1.l1 + cand.z2.l1).mass > 4)
            )
        )
        & (
            (cand.z1.l1.charge + cand.z2.l2.charge != 0)
            | (
                (cand.z1.l1.charge + cand.z2.l2.charge == 0)
                & ((cand.z1.l1 + cand.z2.l2).mass > 4)
            )
        )
        & (
            (cand.z1.l2.charge + cand.z2.l1.charge != 0)
            | (
                (cand.z1.l2.charge + cand.z2.l1.charge == 0)
                & ((cand.z1.l2 + cand.z2.l1).mass > 4)
            )
        )
        & (
            (cand.z1.l2.charge + cand.z2.l2.charge != 0)
            | (
                (cand.z1.l2.charge + cand.z2.l2.charge == 0)
                & ((cand.z1.l2 + cand.z2.l2).mass > 4)
            )
        )
    )
    # select good ZZ candidates
    if os_method:
        full_mask = (
            z1_mass_g40_mask
            & ghost_removal_mask
            & trigger_acceptance_mask
            & qcd_suppression_mask
        )
    else:
        full_mask = qcd_suppression_mask

    cand = cand[ak.fill_none(full_mask, False)]

    if os_method:
        # get alternative pairing for same-sign candidates (FSR photons are used)
        # select same flavor pairs
        sf_pairs = np.abs(cand.z1.l1.pdgId) == np.abs(cand.z2.l1.pdgId)
        cand_sf = cand.mask[sf_pairs]
        # get initial alternative pairs
        ops = cand_sf.z1.l1.pdgId == -cand_sf.z2.l1.pdgId
        za0 = ak.where(
            ops,
            cand_sf.z1.l1.p4 + cand_sf.z2.l1.p4,
            cand_sf.z1.l1.p4 + cand_sf.z2.l2.p4,
        )
        zb0 = ak.where(
            ops,
            cand_sf.z1.l2.p4 + cand_sf.z2.l2.p4,
            cand_sf.z1.l2.p4 + cand_sf.z2.l1.p4,
        )
        # get final alternative pairs selecting Za as the one closest to the Z mass
        zmass = 91.1876
        dist_from_za_to_zmass = np.abs(za0.mass - zmass)
        dist_from_zb_to_zmass = np.abs(zb0.mass - zmass)
        za = ak.where(
            dist_from_zb_to_zmass > dist_from_za_to_zmass,
            za0,
            zb0,
        )
        zb = ak.where(
            dist_from_zb_to_zmass < dist_from_za_to_zmass,
            za0,
            zb0,
        )
        smart_cut = ~(
            (np.abs(za.mass - zmass) < np.abs(cand.z1.p4.mass - zmass)) & (zb.mass < 12)
        )
        smart_cut = ak.fill_none(smart_cut, True)
        cand = cand[smart_cut]

    # add p4 and pT fields to ZLL candidates
    cand["p4"] = cand.z1.p4 + cand.z2.p4
    cand["pt"] = cand.p4.pt
    return cand


def select_best_zzcandidate(cand, cr=False):
    """
    selects best ZZ or ZLL candidate as the one with Z1 closest in mass to nominal Z boson mass
    and Z2 from the candidates whose lepton give higher pT sum

    cand: ZZ or Zll candidate
    cr: Control Region. 'False' for ZZ and 'is_1fcr', 'is_2fcr' or 'is_sscr' for Zll
    """
    if cr:
        selected_cand = cand[cand.z2[cr]]
    else:
        selected_cand = cand
    # get mask of Z1's closest to Z
    zmass = 91.1876
    z1_dist_to_z = np.abs(selected_cand.z1.p4.mass - zmass)
    min_z1_dist_to_z = ak.min(z1_dist_to_z, axis=1)
    closest_z1_mask = z1_dist_to_z == min_z1_dist_to_z
    # get mask of Z2's with higher pT sum
    z2_pt_sum = selected_cand.z2.l1.p4.pt + selected_cand.z2.l2.p4.pt
    max_z2_pt_sum = ak.max(z2_pt_sum[closest_z1_mask], axis=1)
    best_candidate_mask = (z2_pt_sum == max_z2_pt_sum) & closest_z1_mask
    return selected_cand[best_candidate_mask]


def select_candidate_mass(cand, flavor):
    z1_flavor = np.abs(cand.z1.l1.pdgId) + np.abs(cand.z1.l2.pdgId)
    z2_flavor = np.abs(cand.z2.l1.pdgId) + np.abs(cand.z2.l2.pdgId)
    flavor_masks = {
        "4e": (z1_flavor == 22) & (z2_flavor == 22),
        "4mu": (z1_flavor == 26) & (z2_flavor == 26),
        "2e2mu": (z1_flavor == 22) & (z2_flavor == 26),
        "2mu2e": (z1_flavor == 26) & (z2_flavor == 22),
    }
    return cand[flavor_masks[flavor]].p4.mass