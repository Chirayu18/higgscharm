#!/usr/bin/env python3
"""N1: training-region events vs allfix eμ SR events. Host python3 + pyarrow/pandas.
usage: negrw_n1_check.py <train_v2.parquet> <sr_allfix.parquet>"""
import sys, glob, numpy as np, pandas as pd, pyarrow.parquet as pq
tr = pd.read_parquet(sys.argv[1], columns=["era","dataset","event","lhe_vpt","lhe_ht","lepton1_pdgId","lepton2_pdgId","partition"])
sr = pd.read_parquet(sys.argv[2], columns=["era","dataset","event","lhe_vpt","lhe_ht","run","luminosityBlock"])
a, b = tr.lepton1_pdgId.abs(), tr.lepton2_pdgId.abs()
emu = (((a == 11) & (b == 13)) | ((a == 13) & (b == 11))).sum()
print(f"v2 training rows {len(tr)}; eμ-pair rows {int(emu)}; same-flavour pair rows {int(((a>0)&(a==b)).sum())}; pairless {int((a==0).sum())}")
key = ["era","dataset","event"]
m = tr[key+["lhe_vpt","lhe_ht"]].merge(sr[key+["lhe_vpt","lhe_ht"]], on=key, suffixes=("_t","_s"))
print(f"(era,dataset,event) matches: {len(m)}; of which identical lhe_vpt & lhe_ht (same physical event): "
      f"{int(((m.lhe_vpt_t==m.lhe_vpt_s)&(m.lhe_ht_t==m.lhe_ht_s)).sum())}")
# July training (2022postEE, old samples) vs allfix SR 2022postEE
cols = ["event","lhe_vpt","lhe_ht"]
jt = pd.concat([pd.read_parquet(f, columns=cols+["lepton1_pdgId"]).assign(ds=f.split("/")[-3]) for f in
               glob.glob("/eos/user/c/cgupta/higgscharm/outputs/hww_genrw_train/2022postEE/*/train/*.parquet")])
s22 = sr[sr.era == 1]
mj = jt.merge(s22[cols], on="event", suffixes=("_t","_s"))
print(f"July training rows {len(jt)}; event-number matches vs allfix SR 2022postEE: {len(mj)}; "
      f"identical lhe_vpt&lhe_ht: {int(((mj.lhe_vpt_t==mj.lhe_vpt_s)&(mj.lhe_ht_t==mj.lhe_ht_s)).sum())}")
