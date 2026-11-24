import numpy as np
import awkward as ak

# FIX 2026-10-06: cap the per-event relative PDF / alphaS spread. A handful of events in the powheg-MiNNLO
# WH->tautau samples have spreads of O(100-1000), which made the summed lhe_pdf Up 22x the nominal and leaked
# a 6x higgsbkg lhe_pdf template into the 2022postEE card. Outliers (delta > cap) are <1% of events in every
# other sample, where the cap changes the integrated variation by <1%. Same spirit as the outlier threshold
# (0.5) of the columnflow pdf_weights producer.
PDF_DELTA_CAP = 0.5


def add_lhepdf_weight(events, weights_container, shift, pdf_norm=None):
    """
    Add PDF, αs, and combined (PDF⊕αs) systematic weight variations to the Weights container.

    Handles both standard (103 weights) and reduced (101 weights, no αs) PDF sets.

    This function follows the CMS MC Contact Report (Antonis Agapitos et al., B2G Meeting, 14 July 2020):

        δ_pdf   = sqrt( Σ_{k=1}^{100} (σ^(k) - σ^(0))² )
        δ_αs    = |σ^(102) - σ^(101)| / 2   [if αs weights exist]
        δ_total = sqrt( δ_pdf² + δ_αs² )

    where:
      • mem = 0       → central PDF (αs = 0.118)
      • mem = 1–100   → PDF eigenvector members
      • mem = 101–102 → αs variations (αs = 0.116, 0.120)

    The LHEPdfWeight branch contains ratios (w_var / w_nominal) for LHA IDs 306000–306102.
    Some samples (e.g. ST_s-channel_4f_leptonDecays) were produced with only 101 members
    and therefore lack the αs variations.

    The function computes and registers:
      • "PDFweight"       → PDF uncertainty only
      • "AlphaSweight"    → αs uncertainty only (if available)
      • "PDFAlphaSweight" → combined PDF⊕αs uncertainty (if available)

    References:
      - CMS MC Contact Report v3, slides 4–5 & 25–28:
        https://indico.cern.ch/event/938672/contributions/3943718/attachments/2073936/3482265/MC_ContactReport_v3.pdf
      - PDF4LHC recommendations, arXiv:1510.03865
    """
    if shift is None:
        if "LHEPdfWeight" in events.fields:
            pdfweights = events.LHEPdfWeight
            # optional per-file normalisation (lhe_norm.py): member k -> w_k / <w_k>_gen
            if pdf_norm is not None and len(pdf_norm) == int(ak.max(ak.num(pdfweights))):
                _n = np.where(np.asarray(pdf_norm) > 0, np.asarray(pdf_norm), 1.0)
                pdfweights = ak.from_regular(ak.to_regular(pdfweights) / _n[None, :])
                # In these samples member 0 is ALSO inconsistent with the replicas event by event (offset
                # varies per event while the replicas agree to ~0.5%), so the Hessian sum vs member 0 adds the
                # offset 100x. Use the per-event mean of the (normalised) eigen-members as the centre instead.
                _w = ak.to_numpy(ak.to_regular(pdfweights)).astype(np.float64)
                _w[:, 0] = _w[:, 1:101].mean(axis=1)
                pdfweights = ak.from_regular(ak.Array(_w))
            n_members = ak.num(pdfweights)

            # Case 2: 103 members → PDF + αs
            if ak.all(n_members == 103):
                # PDF
                w0 = pdfweights[:, 0]  # central (mem=0)
                w_all = pdfweights[:, 1:101]  # mem=1..100
                diffs = (w_all - w0[:, None]) ** 2
                delta_pdf = np.minimum(np.sqrt(ak.sum(diffs, axis=1)), PDF_DELTA_CAP)

                w_up_pdf = 1 + delta_pdf
                w_down_pdf = 1 - delta_pdf

                # αs
                w_as_low = pdfweights[:, 101]
                w_as_high = pdfweights[:, 102]
                # FIX 2026-10-29 (ported from the HcZZ branch, her 2026-08-19 fix): alphaS is a signed
                # shift, Up = alphaS 0.120. np.abs forced Up >= 1 and Down <= 1 for every event and lost the
                # per-event sign. The cap still bounds the magnitude.
                delta_alpha = np.clip(0.5 * (w_as_high - w_as_low), -PDF_DELTA_CAP, PDF_DELTA_CAP)
                w_up_alpha = 1 + delta_alpha
                w_down_alpha = 1 - delta_alpha

                # PDF + αs
                delta_total = np.sqrt(delta_pdf**2 + delta_alpha**2)
                w_up_total = 1 + delta_total
                w_down_total = 1 - delta_total

            # Case 3: 101 members → PDF only, αs = 1
            elif ak.all(n_members == 101):
                w0 = pdfweights[:, 0]
                w_all = pdfweights[:, 1:101]
                diffs = (w_all - w0[:, None]) ** 2
                delta_pdf = np.minimum(np.sqrt(ak.sum(diffs, axis=1)), PDF_DELTA_CAP)

                w_up_pdf = 1 + delta_pdf
                w_down_pdf = 1 - delta_pdf

                print("skiping AlphaS systematic Weight")
                w_up_alpha = ak.ones_like(delta_pdf)
                w_down_alpha = ak.ones_like(delta_pdf)
                # With no alphaS members, PDF (+) alphaS reduces to the PDF uncertainty.
                w_up_total = w_up_pdf
                w_down_total = w_down_pdf
            else:
                print("skiping PDF/AlphaS systematic Weight")
                delta_pdf = np.ones(len(events))
                w_up_pdf = np.ones(len(events))
                w_down_pdf = np.ones(len(events))
                w_up_alpha = np.ones(len(events))
                w_down_alpha = np.ones(len(events))
                w_up_total = np.ones(len(events))
                w_down_total = np.ones(len(events))

        else:
            print("No LHEPdf Weights in dataset, skiping PDF/AlphaS systematic Weight")
            delta_pdf = np.ones(len(events))
            w_up_pdf = np.ones(len(events))
            w_down_pdf = np.ones(len(events))
            w_up_alpha = np.ones(len(events))
            w_down_alpha = np.ones(len(events))
            w_up_total = np.ones(len(events))
            w_down_total = np.ones(len(events))

        weights_container.add(
            "lhe_pdf",
            weight=ak.ones_like(delta_pdf),
            weightUp=w_up_pdf,
            weightDown=w_down_pdf,
        )
        weights_container.add(
            "lhe_alphaS",
            weight=ak.ones_like(delta_pdf),
            weightUp=w_up_alpha,
            weightDown=w_down_alpha,
        )
        weights_container.add(
            "lhe_pdf_alphaS",
            weight=ak.ones_like(delta_pdf),
            weightUp=w_up_total,
            weightDown=w_down_total,
        )
