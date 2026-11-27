#!/usr/bin/env python3
"""Compare two combine ROOT cards: per-channel x per-process rate and n_eff.

  python3 cmp_cards.py --base <baseline.root> --new <current.root>

n_eff = (sum w)^2 / sum(w^2), computed from the histogram values and errors.
Use this to check that a template change moved ONLY the process you intended --
a large n_eff gain confined to one process is the signature you want; broad
rate shifts across all processes mean the inputs changed for another reason.
"""
import argparse, uproot, numpy as np

CH = ["SR_hplusc","CR_higgsbkg","CR_tt","CR_st","CR_diboson","CR_vjets"]
PR = ["hplusc","higgsbkg","tt","st","diboson","vjets"]

def get(f, ch, p):
    k = f"{ch}_{p}"
    if k not in f: return None
    h = f[k]; return h.values(), h.errors()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--new", required=True)
    ap.add_argument("--flag-pct", type=float, default=2.0,
                    help="mark rows whose rate moved by more than this %%")
    a = ap.parse_args()
    b, c = uproot.open(a.base), uproot.open(a.new)
    print(f"{'channel':13s} {'proc':9s} {'base':>11s} {'new':>11s} {'d%':>7s} "
          f"{'neff_b':>9s} {'neff_n':>9s} {'ratio':>7s}")
    for ch in CH:
        for p in PR:
            x, y = get(b, ch, p), get(c, ch, p)
            if x is None or y is None: continue
            (vb, eb), (vn, en) = x, y
            sb, sn = vb.sum(), vn.sum()
            nb = sb**2/np.sum(eb**2) if np.sum(eb**2) > 0 else 0
            nn = sn**2/np.sum(en**2) if np.sum(en**2) > 0 else 0
            d  = 100*(sn-sb)/sb if sb else 0
            r  = nn/nb if nb > 0 else 0
            flag = " <<<" if abs(d) > a.flag_pct else ""
            print(f"{ch:13s} {p:9s} {sb:11.2f} {sn:11.2f} {d:+6.1f}% "
                  f"{nb:9.0f} {nn:9.0f} {r:6.2f}x{flag}")
