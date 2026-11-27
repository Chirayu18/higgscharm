"""Route B: H+c flavour-scheme stitching of private 3FS and central 4FS-FxFx H->WW samples.

PROPOSAL (HTOWW-ROUTEB-SOL-20261105, 2026-11-05). Agent-written; NOT validated by Chirayu; not yet hooked into
the live processor. Standalone: importing this module changes nothing in Run A.

Physics (fixed by decisions already taken; this module does not choose them)
---------------------------------------------------------------------------
* LHCHWG-2026-007 (arXiv:2608.26863) Sec. 4.4: "phase-space splitting based on the number of charm jets with
  pT > 10 GeV" (footnote 4: 10 GeV, below the usual 25 GeV, "to resolve the soft-charm region");
  "Nc-jet = 0 from the 4FS-FxFx sample and events with Nc-jet >= 1 from the 3FS sample".
* Jet radius R = 0.4 and |eta| < 2.5 for the split: Chirayu's choice, 2026-11-05 (the paper gives neither).
  R = 0.4 is realised by NanoAOD `GenJet` (anti-kT 0.4); this module cannot change R.
* c-jet = GenJet with hadronFlavour == 4 (ghost-hadron flavour; Sec. 5.1 "jet containing a D or B hadron",
  Sec. 5.3 "jet containing at least one charm quark"). NOTE: NanoAOD GenJets cluster all stable non-neutrino
  particles, i.e. also the H->WW decay leptons; the paper (H->gamma gamma only) does not address this. Flagged,
  not corrected here.
* Normalisation (Sec. 4.4, two steps): (1) each region is normalised to its originating sample's fiducial NLO
  cross section times a flat scheme K-factor (NNLO/NLO of bbH, Table 3 "NLO+bbHfact"); (2) the sum is rescaled
  to the NNLO MiNNLOPS ccH cross section.

Normalisation prescription implemented (exact)
----------------------------------------------
For scheme s in {4FS, 3FS}, from PRE-selection, PRE-veto, signed genWeight sums over ALL chunks of ALL
datasets of that scheme in the normalisation domain (one era by default; main + ext1 together):
    F_s = sum_{all events} w ,   K_s = sum_{kept events} w ,   f_s = K_s / F_s
    kept(4FS) = [N_cjet == 0],  kept(3FS) = [N_cjet >= 1]
    A_s = sigma_s^{NLO x K_s^{bbH}} (Table 3, "No cuts", NLO+bbHfact): A_4FS = 0.1690, A_3FS = 0.1630 [fb, H->gg]
    D   = A_4FS f_4FS + A_3FS f_3FS
    c_s = A_s / D                      (dimensionless; BR(H->gg) cancels; BR(H->WW->2l2nu) must be the same in
                                        both samples -- both are H->WW, W->(e,mu,tau)nu, checked for the 3FS
                                        fragment only)
    event weight (kept events only) = L * X * c_s * w * (other per-event SFs) / F_s
with X = sigma_NNLO(ccH) x BR = the fileset `xsec` of the central 4FS entry (0.0016626 pb, already NNLO). Then
    sum_kept weight = L X (c_4 f_4 + c_3 f_3) = L X           (closure, exact, before selection)
The existing framework scale L X / F_s (read_scale) is therefore multiplied by c_s and non-kept events dropped.
Equivalent to the paper's S = sigma_NNLO / D with region weights K_s sigma_s.

Negative weights: every sum is signed; |w| sums and negative counts are recorded for diagnostics only.
sumw2 is recorded for the statistical uncertainty of f_s (correlated kept/full ratio).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, Iterable, Mapping, Optional

import numpy as np

# ------------------------------------------------------------------------------------------------------------
# Constants with provenance
# ------------------------------------------------------------------------------------------------------------
PAPER = "LHCHWG-2026-007, arXiv:2608.26863"
# Table 3, 13.6 TeV, pp -> H(gg) cc~, "No cuts" column, fb (central values; scale bands omitted).
TABLE3_NO_CUTS_FB = {
    "MiNNLOPS_4FS": 0.1664,
    "NLO_4FS": 0.2216,
    "NLO_3FS": 0.1248,
    "NLO_bbHfact_4FS": 0.1690,
    "NLO_bbHfact_3FS": 0.1630,
}
# Flat scheme K-factors implied by Table 3 (identical within rounding in all three columns):
#   K_4FS = 0.1690/0.2216 = 0.7626 ; K_3FS = 0.1630/0.1248 = 1.3061
K_FACTOR = {
    "4FS": TABLE3_NO_CUTS_FB["NLO_bbHfact_4FS"] / TABLE3_NO_CUTS_FB["NLO_4FS"],
    "3FS": TABLE3_NO_CUTS_FB["NLO_bbHfact_3FS"] / TABLE3_NO_CUTS_FB["NLO_3FS"],
}
# Inclusive NLO generator cross sections the paper's numbers correspond to (BR(gg) removed) are reproduced by
# the CMS samples: 3FS 54.97 fb (gridpack / XSDB), 4FS-FxFx 97.62 fb (XSDB, post-merging). Only ratios enter.
A_SCHEME = {
    "4FS": TABLE3_NO_CUTS_FB["NLO_bbHfact_4FS"],
    "3FS": TABLE3_NO_CUTS_FB["NLO_bbHfact_3FS"],
}

DEFAULT_PT_MIN = 10.0     # GeV, strict '>' (paper: "pT > 10 GeV")
DEFAULT_ETA_MAX = 2.5     # strict '<' (Chirayu 2026-11-05)
C_FLAVOUR = 4
SCHEMES = ("4FS", "3FS")


# ------------------------------------------------------------------------------------------------------------
# Event level
# ------------------------------------------------------------------------------------------------------------
def count_gen_cjets(genjets, pt_min: float = DEFAULT_PT_MIN, eta_max: float = DEFAULT_ETA_MAX) -> np.ndarray:
    """Per-event number of GEN c-jets: GenJet.hadronFlavour == 4, pt > pt_min, |eta| < eta_max.

    `genjets` is a jagged record with fields pt, eta, hadronFlavour (events.GenJet). Adapted from
    S. Nandakumar's count_gen_cjets (HcZZ-zx-estimation c508ecc), which has no eta cut; the |eta| cut is
    Chirayu's decision for Route B. Returns a flat int64 numpy array.
    """
    import awkward as ak

    is_c = (genjets.hadronFlavour == C_FLAVOUR) & (genjets.pt > pt_min) & (abs(genjets.eta) < eta_max)
    return np.asarray(ak.to_numpy(ak.sum(is_c, axis=1)), dtype=np.int64)


def keep_mask(n_cjets: np.ndarray, scheme: str) -> np.ndarray:
    """Route B keep mask: 4FS keeps N==0, 3FS keeps N>=1. Complementary by construction."""
    n = np.asarray(n_cjets)
    if scheme == "4FS":
        return n == 0
    if scheme == "3FS":
        return n >= 1
    raise ValueError(f"unknown scheme {scheme!r}; expected one of {SCHEMES}")


# ------------------------------------------------------------------------------------------------------------
# Additive sums
# ------------------------------------------------------------------------------------------------------------
@dataclass
class WeightSums:
    sumw: float = 0.0
    sumw2: float = 0.0
    sumw_abs: float = 0.0
    n: int = 0
    n_neg: int = 0

    @classmethod
    def of(cls, w: np.ndarray) -> "WeightSums":
        w = np.asarray(w, dtype=np.float64)
        return cls(float(w.sum()), float((w * w).sum()), float(np.abs(w).sum()), int(w.size), int((w < 0).sum()))

    def __add__(self, o: "WeightSums") -> "WeightSums":
        return WeightSums(self.sumw + o.sumw, self.sumw2 + o.sumw2, self.sumw_abs + o.sumw_abs,
                          self.n + o.n, self.n_neg + o.n_neg)


@dataclass
class StitchSums:
    """Pre-selection sums for ONE scheme. Additive across chunks, files, datasets (main+ext1) and eras.

    full: all events read; kept: events passing keep_mask(scheme); extra: optional named per-event weight
    multipliers (e.g. corrected LHE scale weights) summed as w*m over full and kept, for later variation
    renormalisation. Nothing here applies a weight to events.
    """
    scheme: str
    full: WeightSums = field(default_factory=WeightSums)
    kept: WeightSums = field(default_factory=WeightSums)
    extra_full: Dict[str, float] = field(default_factory=dict)
    extra_kept: Dict[str, float] = field(default_factory=dict)
    pt_min: float = DEFAULT_PT_MIN
    eta_max: float = DEFAULT_ETA_MAX
    n_chunks: int = 0

    def __add__(self, o: "StitchSums") -> "StitchSums":
        if (self.scheme, self.pt_min, self.eta_max) != (o.scheme, o.pt_min, o.eta_max):
            raise ValueError(f"cannot add StitchSums with different definitions: "
                             f"{(self.scheme, self.pt_min, self.eta_max)} vs {(o.scheme, o.pt_min, o.eta_max)}")
        if self.n_chunks and o.n_chunks and set(self.extra_full) != set(o.extra_full):
            raise ValueError("cannot add StitchSums with different extra-weight names")
        ef = {k: self.extra_full.get(k, 0.0) + o.extra_full.get(k, 0.0) for k in set(self.extra_full) | set(o.extra_full)}
        ek = {k: self.extra_kept.get(k, 0.0) + o.extra_kept.get(k, 0.0) for k in set(self.extra_kept) | set(o.extra_kept)}
        return StitchSums(self.scheme, self.full + o.full, self.kept + o.kept, ef, ek,
                          self.pt_min, self.eta_max, self.n_chunks + o.n_chunks)

    __radd__ = lambda self, o: self if o == 0 else o.__add__(self)  # noqa: E731  (allows sum([...]))

    @property
    def fraction(self) -> float:
        return self.kept.sumw / self.full.sumw

    @property
    def fraction_stat(self) -> float:
        """Stat. uncertainty of f = K/F with kept subset of full (signed weights):
        var f = [ (1-f)^2 sum_kept w^2 + f^2 sum_notkept w^2 ] / F^2."""
        f = self.fraction
        notkept_w2 = self.full.sumw2 - self.kept.sumw2
        return math.sqrt(max(0.0, (1 - f) ** 2 * self.kept.sumw2 + f ** 2 * notkept_w2)) / abs(self.full.sumw)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping) -> "StitchSums":
        d = dict(d)
        d["full"] = WeightSums(**d["full"])
        d["kept"] = WeightSums(**d["kept"])
        return cls(**d)


def chunk_sums(genweight, n_cjets, scheme: str, extra: Optional[Mapping[str, np.ndarray]] = None,
               pt_min: float = DEFAULT_PT_MIN, eta_max: float = DEFAULT_ETA_MAX):
    """Return (StitchSums, keep-mask) for one chunk. `genweight` must be the PRE-veto, PRE-selection
    events.genWeight (the same events dump_chunk_sumw sees)."""
    w = np.asarray(genweight, dtype=np.float64)
    m = keep_mask(n_cjets, scheme)
    if m.shape != w.shape:
        raise ValueError(f"mask/weight shape mismatch {m.shape} vs {w.shape}")
    ef, ek = {}, {}
    for name, mult in (extra or {}).items():
        mult = np.asarray(mult, dtype=np.float64)
        ef[name] = float((w * mult).sum())
        ek[name] = float((w * mult)[m].sum())
    return StitchSums(scheme, WeightSums.of(w), WeightSums.of(w[m]), ef, ek, pt_min, eta_max, 1), m


def event_sums(events, scheme: str, pt_min: float = DEFAULT_PT_MIN, eta_max: float = DEFAULT_ETA_MAX,
               extra: Optional[Mapping[str, np.ndarray]] = None):
    """Convenience wrapper on NanoEvents: (StitchSums, keep-mask, n_cjets)."""
    import awkward as ak

    if "GenJet" not in events.fields or not hasattr(events, "genWeight"):
        raise RuntimeError("Route B stitching needs MC with GenJet and genWeight")
    n = count_gen_cjets(events.GenJet, pt_min, eta_max)
    s, m = chunk_sums(ak.to_numpy(events.genWeight), n, scheme, extra, pt_min, eta_max)
    return s, m, n


# ------------------------------------------------------------------------------------------------------------
# Normalisation
# ------------------------------------------------------------------------------------------------------------
@dataclass
class StitchNorm:
    c: Dict[str, float]              # per-scheme multiplicative factor on L*X/F_s
    f: Dict[str, float]              # kept fractions
    f_stat: Dict[str, float]
    D: float
    closure: float                   # c_4 f_4 + c_3 f_3 (== 1 up to rounding)
    S_paper: float                   # sigma_NNLO / (sum_s K_s sigma_s f_s)  (paper's step-2 factor, H->gg fb units)
    neg_fraction: Dict[str, float]
    inputs: Dict[str, dict]

    def scale(self, scheme: str, lumi_pb: float, xsec_pb: float, full_sumw: float) -> float:
        """Per-event normalisation for kept events of `scheme`: L * X * c_s / F_s (multiply by genWeight)."""
        return lumi_pb * xsec_pb * self.c[scheme] / full_sumw

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1, sort_keys=True)


def normalisation(sums: Mapping[str, StitchSums], a_scheme: Mapping[str, float] = A_SCHEME,
                  sigma_nnlo_fb: float = TABLE3_NO_CUTS_FB["MiNNLOPS_4FS"]) -> StitchNorm:
    """Compute c_4FS, c_3FS from the summed (over the normalisation domain) StitchSums of both schemes."""
    missing = [s for s in SCHEMES if s not in sums]
    if missing:
        raise ValueError(f"missing schemes {missing}")
    f, fs, neg = {}, {}, {}
    defs = {(sums[s].pt_min, sums[s].eta_max) for s in SCHEMES}
    if len(defs) != 1:
        raise ValueError(f"4FS and 3FS sums use different c-jet definitions: {defs}")
    for s in SCHEMES:
        ss = sums[s]
        if ss.scheme != s:
            raise ValueError(f"sums[{s!r}] is labelled {ss.scheme!r}")
        if not ss.full.sumw > 0:
            raise ValueError(f"{s}: full sumw = {ss.full.sumw} <= 0")
        f[s] = ss.fraction
        if not 0.0 < f[s] <= 1.0 + 1e-12:
            raise ValueError(f"{s}: kept fraction {f[s]} outside (0,1] (negative-weight dominated?)")
        fs[s] = ss.fraction_stat
        neg[s] = ss.full.n_neg / ss.full.n if ss.full.n else float("nan")
    D = sum(a_scheme[s] * f[s] for s in SCHEMES)
    c = {s: a_scheme[s] / D for s in SCHEMES}
    closure = sum(c[s] * f[s] for s in SCHEMES)
    sigma_nlo = {"4FS": TABLE3_NO_CUTS_FB["NLO_4FS"], "3FS": TABLE3_NO_CUTS_FB["NLO_3FS"]}
    S_paper = sigma_nnlo_fb / sum(K_FACTOR[s] * sigma_nlo[s] * f[s] for s in SCHEMES)
    return StitchNorm(c, f, fs, D, closure, S_paper, neg,
                      {s: sums[s].to_dict() for s in SCHEMES})


def combine(records: Iterable[StitchSums]) -> Dict[str, StitchSums]:
    """Sum records per scheme (chunks/files/datasets/eras as chosen by the caller's domain)."""
    out: Dict[str, StitchSums] = {}
    for r in records:
        out[r.scheme] = out[r.scheme] + r if r.scheme in out else r
    return out


# ------------------------------------------------------------------------------------------------------------
# Workflow config
# ------------------------------------------------------------------------------------------------------------
@dataclass
class RouteBConfig:
    datasets: Dict[str, str]                 # dataset (YAML sample name) -> scheme
    pt_min: float = DEFAULT_PT_MIN
    eta_max: float = DEFAULT_ETA_MAX
    domain: str = "per_era"                  # "per_era" (default, exact per-era closure) or "pooled_fractions"
    name: str = "hplusc_routeb"
    defer_selection: bool = False           # retain both schemes for offline A/B choices

    @classmethod
    def from_block(cls, block: Mapping) -> "RouteBConfig":
        """Parse the workflow `stitching: hplusc_routeb:` block (see proposed hww_routeb_3fs4fs.yaml)."""
        ds = {}
        for scheme in SCHEMES:
            for d in block.get(scheme, []) or []:
                if d in ds:
                    raise ValueError(f"dataset {d} listed for two schemes")
                ds[d] = scheme
        if not any(v == "4FS" for v in ds.values()) or not any(v == "3FS" for v in ds.values()):
            raise ValueError("stitching block needs at least one 4FS and one 3FS dataset")
        dom = block.get("domain", "per_era")
        if dom not in ("per_era", "pooled_fractions"):
            raise ValueError(f"unknown domain {dom}")
        deferred = block.get("defer_selection", False)
        if not isinstance(deferred, bool):
            raise ValueError("defer_selection must be a boolean")
        return cls(ds, float(block.get("pt_min", DEFAULT_PT_MIN)), float(block.get("eta_max", DEFAULT_ETA_MAX)),
                   dom, block.get("name", "hplusc_routeb"), deferred)

    @classmethod
    def from_workflow(cls, workflow: str) -> Optional["RouteBConfig"]:
        """Read the raw workflow YAML (WorkflowConfigBuilder drops unknown top-level keys). None if absent."""
        import importlib.resources
        import yaml

        with importlib.resources.open_text("analysis.workflows", f"{workflow}.yaml") as fh:
            raw = yaml.safe_load(fh)
        block = (raw.get("stitching") or {}).get("hplusc_routeb")
        return cls.from_block(block) if block else None

    def scheme_of(self, dataset: str) -> Optional[str]:
        """Exact match only (no prefix matching: HplusCharm_3FS_HToWW must not catch ..._ext1 by accident)."""
        return self.datasets.get(dataset)
