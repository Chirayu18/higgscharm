#!/usr/bin/env python3
"""HWW adaptation of HZZ selected-row/job completeness bookkeeping."""
import argparse
import glob
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from coffea.util import load
from analysis.processing_checks import check_train_rows

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--coffea',nargs='+',required=True)
    p.add_argument('--parquets',nargs='+',required=True)
    a=p.parse_args()
    files=[f for pat in a.parquets for f in glob.glob(pat,recursive=True)]
    result=check_train_rows([load(f) for f in a.coffea],files)
    for (year,sample),count in sorted(result.items()): print(year,sample,count,'PASS')
