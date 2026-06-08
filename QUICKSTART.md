# Quickstart — H+c (H→WW) analysis on `migration-v2`

This branch runs the **H+c → WW** analysis end-to-end: NanoAOD → parquet
ntuples (with JES/JER/lepton systematics) → 6-class MVA scoring → Combine
datacard → r₉₅ limit. It is built on the **`uacms/MVA`** systematics line, with
content cherry-picked from Thomas's 53-variable line.

```
git checkout migration-v2
```

---

## What's different vs upstream `uacms/MVA`

| Area | Change |
|---|---|
| **Parquet output** | Self-normalising shards — `sumw`/`xsec`/`era` stored in the parquet schema metadata, plus an `event` id column. No external bookkeeping needed. |
| **Object-shift systematics** | One Condor job per dataset emits the nominal **plus every** JES/JER/lepton scale+resolution shift in a single NanoAOD read (`object_shifts: true`). Shifts land in `<dataset>/base/<shift>/`. |
| **Shift merge** | `merge_shifted_parquets_by_sample` collapses them to `<year>/<shift>/<sample>.parquet`. |
| **MVA** | Train/test split + inference extracted into `scripts/mva/{prep_training_inputs,run_inference}.py`; 6-class model, order `[hplusc, higgsbkg, tt, st, diboson, vjets]`. |
| **Combine** | New `scripts/combine/` pipeline + a `combine:` block in the workflow yaml (6 argmax channels, 12 shape systs, 9 lnNs). |
| **Variables** | 53 MVA/histogram variables (mirrored into `hww.yaml` + `hww_MVA.yaml`). |
| **Selection** | Single category `base`, now requiring **`atleast_one_cjet`** (SR category dropped). |
| **Fixes** | muon `pt_resol` clamp; empty-parquet failsafes; **2022postEE JEC tags bumped V3→V4 / JRV1→JRV2** (cvmfs `latest` drift — see Gotcha). |

Two workflow files, identical except one knob:
`hww_MVA.yaml` (`object_shifts: false`, nominal) and `hww_combine.yaml`
(`object_shifts: true`, nominal + all shifts).

---

## Run it end-to-end

### Pre-flight (once per session)
```bash
voms-proxy-init --voms cms --valid 192:00
cd .../higgscharm
ln -sfn /eos/user/c/cgupta/higgscharm/outputs outputs   # if not already linked
```

### Nominal → r₉₅ (the working path)
```bash
# 1. Produce parquets on Condor. Use --memory 6000 (default 3 GB OOMs heavy samples).
python runner.py -w hww_MVA -y 2022postEE --output_format parquet --eos --submit --memory 6000

# 2. Check completion / resubmit failures (parquet-aware):
python jobs_status.py -w hww_combine -y 2022postEE --eos --output_format parquet
#    -> reports finished/missing per dataset; answer y to blacklist bad xrootd sites + resubmit

# 3. Merge shards (base coffea/torch env)
python run_postprocess.py -w hww_MVA -y 2022postEE --postprocess --output_format parquet

# 4. MVA inference -> adds mva_score_* columns
python scripts/mva/run_inference.py -w hww_MVA -y 2022postEE

# 5. Build ROOT templates + datacard
python scripts/combine/make_combine_inputs.py -w hww_MVA -y 2022postEE

# 6. Fits (inside CMSSW — the wrapper sources it)
bash scripts/combine/run_combine.sh hww_MVA

# 7. Plots
python scripts/combine/make_combine_plots.py -w hww_MVA
python scripts/combine/make_impact_plot.py   -w hww_MVA
```

Only **step 6** runs inside CMSSW; steps 3–5 run in the base env.

### Retrain the MVA (optional, in the b-hive repo)
```bash
python scripts/mva/prep_training_inputs.py -w hww_MVA -y 2022postEE  # split + labels + filelists
cd .../b-hive && ./train_MVA.sh                                      # DatasetConstructor + Training
# then point inference.model_path in hww_MVA.yaml at the new best_model.pt
```

---

## Gotchas

- **JEC version drift.** cvmfs `latest` `jet_jerc.json.gz` moves versions; the tags
  in `analysis/corrections/jec_params_correctionlib.yaml` must match or jobs die
  with `KeyError: ..._V3_MC_L1L2L3Res_AK4PFPuppi`. **2022postEE is fixed (V4/JRV2);
  2022preEE and 2023* are still on V3/JRV1** — bump before running those years.
  Verify a key exists:
  ```python
  correctionlib.CorrectionSet.from_file(<json>).compound.keys()
  ```
- **Memory.** Runner Condor jobs need `--memory 6000`.
- **`jobs_status.py` output format.** Pass `--output_format parquet`; the default
  `coffea` counts stale `.coffea` stubs and reports wrong numbers.

---

## Status

- **Nominal limit + weight-based systematics: ready** (the v3 r₉₅ path).
- **Object-shift (JES/JER) shape templates: produced & scored, not yet folded into
  the datacard** — that's the one remaining wiring step in
  `scripts/combine/make_combine_inputs.py` (loop the `<year>/<shift>/mva/` dirs and
  emit shape rows).
