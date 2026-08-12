# Higgs + heavy-flavour composition uncertainty (the AN's "ggH+heavy flavor jets").
#
# SOURCE (verified 2026-08-11): HiggsDNA
#   higgs_dna/systematics/event_weight_systematics.py :: Higgs_plus_HF_syst
#   https://gitlab.cern.ch/cms-analysis/general/HiggsDNA
#
# AN-23-102 section 7.1 (line 547) and Table 16: "A conservative uncertainty of heavy
# flavor modeling of ggH is assigned, 50% uncertainty on the normalisation of the yield".
#
# WHY THIS REPLACES THE lnN. Our card carried `flavor_composition_ggH` as a flat lnN on
# the whole merged `higgsbkg` group. That is wrong twice over:
#   1. ggH is only 13.1% of the merged SR yield (VBF 29.0%, ggZH 23.3%, ZH 21.0%, ...),
#      so a flat lnN either over-penalises the other 87% (the old 1.40) or has to be
#      scaled down to an effective average (the 1.066 stopgap).
#   2. A flat lnN moves every component together and cannot produce the SHAPE effect
#      that a genuine per-event heavy-flavour uncertainty has.
# This per-event weight instead selects on GEN-LEVEL jet flavour: only events that
# actually contain a heavy-flavour gen jet are varied, so the grouping stops mattering
# and the shape comes out correctly.
import awkward as ak
import numpy as np


# Higgs samples the uncertainty applies to. HiggsDNA's docstring warns
# "Make sure you apply it only on ggH or VBF samples"; AN-23-102 scopes it to ggH.
_HIGGS_PREFIXES = ("GluGluH", "VBFH")


def add_higgs_hf_weight(
    events,
    weights_container,
    shift,
    dataset,
    flav="c",
    pt_min=25.0,
    eta_max=2.5,
    rel_unc=0.5,
):
    """Flat +-rel_unc on Higgs events containing >=1 heavy-flavour GEN jet.

    Events with no such jet get exactly 1.0, so the nuisance is automatically
    confined to the phase space it is meant to cover.

    flav: "c" -> hadronFlavour == 4, "b" -> hadronFlavour == 5.
    """
    if shift is not None:
        return
    if not dataset.startswith(_HIGGS_PREFIXES):
        return
    if "GenJet" not in events.fields:
        print("No GenJet in dataset, skip systematic: Higgs+HF composition")
        return

    flav_id = {"c": 4, "b": 5}.get(flav)
    if flav_id is None:
        raise ValueError("flav must be either 'b' or 'c'")

    try:
        gj = events.GenJet
        gj = gj[(gj.pt > pt_min) & (abs(gj.eta) < eta_max)]
        n_hf = ak.sum(gj.hadronFlavour == flav_id, axis=-1)

        has_hf = ak.to_numpy(n_hf > 0)
        up = np.where(has_hf, 1.0 + rel_unc, 1.0)
        down = np.where(has_hf, 1.0 - rel_unc, 1.0)

        weights_container.add(
            name=f"higgs_plus_{flav}",
            weight=np.ones(len(events)),
            weightUp=up,
            weightDown=down,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to build Higgs+HF weight, skipping systematic: {exc}")
