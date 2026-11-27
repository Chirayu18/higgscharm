"""Per-file normalisation of LHE scale / PDF variation weights (2026-10-02).

Some central MadGraph FxFx samples (HPlusCharm-4FS / HPlusBottom_5FS ...MuRFScaleDynX0p50...HToWWTo2L2Nu) store
LHEScaleWeight / LHEPdfWeight with a global offset w.r.t. the nominal: every variation except the mu=1 entry
(and PDF member 0) is ~2x the nominal on average, even the 100 PDF replicas, whose spread is ~1%. Taken at face
value that turns lhe_pdf into a x10 nuisance and scale into a same-sign x2 one.

Fix (opt-in per dataset via `lhe_renorm_datasets` in the workflow yaml): divide each variation by its
generator-level mean, read from the input file's Runs tree (LHEScaleSumw, LHEPdfSumw are already normalised to
genEventSumw; combine run entries weighted by genEventSumw). Keeps the acceptance/shape effect of each
variation, removes the spurious normalisation offset."""
from functools import lru_cache
import numpy as np
import uproot


@lru_cache(maxsize=64)
def lhe_variation_means(filename):
    """Return (scale_means[9] or None, pdf_means[n] or None), each relative to its nominal entry."""
    with uproot.open(filename, timeout=600) as f:
        runs = f["Runs"]
        keys = set(runs.keys())
        sumw = np.asarray(runs["genEventSumw"].array(library="np"), dtype=np.float64)

        def combine(branch, nominal_index):
            if branch not in keys:
                return None
            arr = runs[branch].array(library="np")
            stacked = np.stack([np.asarray(a, dtype=np.float64) for a in arr])  # (nruns, nvar)
            mean = (stacked * sumw[:, None]).sum(0) / sumw.sum()
            return mean / mean[nominal_index] if mean[nominal_index] != 0 else mean

        return combine("LHEScaleSumw", 4), combine("LHEPdfSumw", 0)
