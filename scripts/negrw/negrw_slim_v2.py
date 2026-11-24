#!/usr/bin/env python3
"""Slim the hww_genrw_train_v2 training parquets (and the allfix SR parquets) to the columns the
negrw training/validation needs, tagged with era/dataset codes. Host python3 + pyarrow only.
usage: negrw_slim_v2.py train <out.parquet> | sr <out.parquet> [<sr_workflow_dir>]"""
import sys, glob, os, re
import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
FEATURES = ["lhe_njets","lhe_nb","lhe_nc","lhe_nuds","lhe_nglu","lhe_npnlo","lhe_ht","lhe_htincoming","lhe_vpt","lhe_alphas",
    "genparton_multiplicity","genparton_n_pt20","genparton_n_pt100","genparton_n_pt200","genparton_incoming1_pdgId",
    "genparton_incoming2_pdgId","genparton1_pt","genparton1_eta","genparton2_pt","genparton2_eta"]
DATASETS = ["DYto2L_2Jets_10to50","DYto2L_2Jets_50","WtoLNu_2Jets_0J","WtoLNu_2Jets_1J","WtoLNu_2Jets_2J"]
ERAS = ["2022preEE","2022postEE","2023preBPix","2023postBPix"]
OUT = "/eos/user/c/cgupta/higgscharm/outputs"
mode, outp = sys.argv[1], sys.argv[2]
if mode == "train":
    base, sub, extra = f"{OUT}/hww_genrw_train_v2", "train", ["event", "lepton1_pdgId", "lepton2_pdgId"]
else:
    base = sys.argv[3] if len(sys.argv) > 3 else f"{OUT}/hww_allfix_4fs__train_v1_20261110"
    sub, extra = "base", ["run", "luminosityBlock", "event", "weight_negrw", "weight_negrw_std"]
cols = FEATURES + ["weight_nominal"] + extra
tabs = []; nfile = {}
for ei, era in enumerate(ERAS):
    for di, ds in enumerate(DATASETS):
        for d in glob.glob(f"{base}/{era}/{ds}*"):
            name = os.path.basename(d)
            if not (name == ds or re.fullmatch(re.escape(ds) + r"_\d+", name)): continue
            for f in glob.glob(f"{d}/{sub}/*.parquet"):
                t = pq.read_table(f, columns=cols)
                if t.num_rows == 0: continue
                t = t.append_column("era", pa.array([ei] * t.num_rows, pa.int8()))
                t = t.append_column("dataset", pa.array([di] * t.num_rows, pa.int8()))
                t = t.append_column("partition", pa.array([name] * t.num_rows, pa.string()))
                tabs.append(t.cast(pa.schema([(c, pa.float32() if c in FEATURES else t.schema.field(c).type) for c in t.column_names])))
                nfile[(era, ds)] = nfile.get((era, ds), 0) + 1
T = pa.concat_tables(tabs)
pq.write_table(T, outp, compression="zstd")
print(outp, T.num_rows, "rows")
for k, v in sorted(nfile.items()): 
    m = pc.and_(pc.equal(T["era"], ERAS.index(k[0])), pc.equal(T["dataset"], DATASETS.index(k[1])))
    print(k, v, "files", pc.sum(m.cast(pa.int64())).as_py(), "rows")
