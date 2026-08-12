# H→WW + charm (H+c) — analysis pipeline

End-to-end guide for the **`hww_combine_2dcat`** workflow: NanoAOD → parquet →
MVA → datacard → limit. This documents the H→WW-specific layer built on top of
the base framework; see [`README.md`](README.md) for the underlying higgscharm
framework (filesets, workflow YAML structure, condor submission).

**Repo**: `/afs/cern.ch/user/c/cgupta/higgscharm_thomas/higgscharm_thomas_new/higgscharm`
**Env**: micromamba `b_hive` (coffea 0.7 / pandas)
**b-hive**: `/eos/home-c/cgupta/HToWW/b-hive`

```bash
export MAMBA_EXE=/eos/user/c/cgupta/EPR_task/b-hive/micromamba/micromamba
export MAMBA_ROOT_PREFIX=/eos/user/c/cgupta/EPR_task/b-hive/micromamba
$MAMBA_EXE run -n b_hive python3 <script>     # micromamba is a shell function, not on PATH
```

---

## Contents

- [Pipeline overview](#pipeline-overview)
- [0. Prerequisites](#0-prerequisites)
- [1. Process NanoAOD → parquet](#1-process-nanoaod--parquet)
- [2. Postprocess → merged parquets](#2-postprocess--merged-parquets)
- [3. MVA: training](#3-mva-training)
- [4. MVA: inference](#4-mva-inference)
- [5. Build combine inputs](#5-build-combine-inputs)
- [6. Run the limit](#6-run-the-limit)
- [7. Impacts / nuisance ranking](#7-impacts--nuisance-ranking)
- [Analysis structure](#analysis-structure)
- [Systematics](#systematics)
- [Known traps](#known-traps)

---

## Pipeline overview

```
NanoAOD (DAS/EOS)
   |  runner.py                    condor, ~21 MC+signal datasets
   v
parquet  outputs/<workflow>/<year>/<dataset>/            nominal + 6 object-shift dirs
   |  run_postprocess.py --mva
   v
merged parquet  .../<var>/                               per-process, sumw sidecar
   |  scripts/mva/run_inference.py                       6-class MVA scores
   v
mva parquet     .../<var>/mva/                           mva_score_{hplusc,higgsbkg,tt,st,diboson,vjets}
   |  scripts/combine/make_combine_inputs_v2.py
   v
datacard  outputs/combine/v11_hplusc_2dcat.{root,txt}    6 channels x 6 processes
   |  combine
   v
limit
```

The MVA is trained **separately** (section 3) using b-hive; steps 1-2 produce its
training input, and step 4 applies the trained model.

---

## 0. Prerequisites

**Grid proxy** — node-local, so re-init or use the AFS copy:
```bash
voms-proxy-init --voms cms --valid 192:00
# persistent copy (NOT cms.proxy, which is expired):
export X509_USER_PROXY=/afs/cern.ch/user/c/cgupta/private/x509up_u151861
```

**EOS space** — check before any campaign (quota node, instant, no directory walk):
```bash
eos root://eosuser.cern.ch quota /eos/user/c/cgupta
```
Campaign needs headroom. **Never delete anything on EOS without confirming first.**

**Long jobs** run in `tmux`, not `nohup` — submissions outlive the ssh session.

---

## 1. Process NanoAOD → parquet

```bash
python3 runner.py --workflow hww_combine_2dcat --year 2022postEE \
        --submit --eos --output_format parquet
```

Emits nominal + all object-shift trees (JES/JER/lepton scale+res), with
negative-weight reweighting and native 2D c-tag SF columns baked in.

Years: `2022preEE`, `2022postEE`, `2023preBPix`, `2023postBPix`.

**Monitor / resubmit:**
```bash
watch condor_q
python3 jobs_status.py --workflow hww_combine_2dcat --year 2022postEE --eos
```

> **Trap:** `jobs_status.py`'s site blacklist is **memoryless** — it re-admits
> previously-failed sites each round and cycles. Filesets list ONE replica per
> file, so blacklisting a site **deletes those files** from the fileset. Watch
> the file counts across rounds.

> **Trap:** DRPremix step2 OOMs at 4 GB / 2 cores. Resubmit at 5 GB (policy cap
> is 2500 MB/core) or bump the config to 3 cores / 7500 MB.

---

## 2. Postprocess → merged parquets

```bash
python3 run_postprocess.py --workflow hww_combine_2dcat --year 2022postEE \
        --postprocess --output_format parquet --mva
```

Merges per-dataset parquets into per-process files and writes the **sumw sidecar
json**, which is what normalisation reads.

> **Trap:** `read_scale = lumi × xsec / sumw` MUST use the sidecar
> `sumw_<year>.json`. Parquet metadata undercounts `WtoLNu` by 5.8× and would
> silently inflate V+jets by ~2.4×.

---

## 3. MVA: training

Trained with **b-hive** (law-based task graph), separate from the coffea repo.

**Model**: `SimpleMLP_MultiClass`, 6 classes
`[hplusc, higgsbkg, tt, st, diboson, vjets]`
**Config**: `HPlusCHToWW_2dcats` — 26 features including the 11 one-hot
`cjet_cand_ctag2d_*` categories (replacing raw PNet CvL/CvB scores).

### 3a. Prepare labelled + split inputs

```bash
cd /eos/home-c/cgupta/HToWW/b-hive

# attach 6-class labels from the workflow's process_groups
python3 make_mva_labeled.py \
    --input-dir /eos/user/c/cgupta/higgscharm/outputs/hww_combine_2dcat/2022postEE \
    --groups-key process_groups

# 80/20 train/test split, seed 42
python3 split_train_test.py \
    --input-dir /eos/user/c/cgupta/higgscharm/outputs/hww_combine_2dcat/2022postEE/mva_labeled
```

Produces the filelists referenced by the training script:
```
filelists/v11_train_allEras.txt
filelists/v11_test_allEras.txt
```

> The train/test split is by `event` id (`test_modulo: 10`, `test_remainder: 9`)
> so the same event never lands in both, across eras.

### 3b. Run the training

```bash
./train_v11_2dcats.sh                    # defaults: 30 epochs, batch 1024, lr 1e-3
./train_v11_2dcats.sh --epochs 50 --lr 5e-4
./train_v11_2dcats.sh --debug            # quick smoke test
```

The script runs four law tasks in sequence:

| step | task | what it does |
|---|---|---|
| 1a | `DatasetConstructorTask` | build train dataset from `v11_train_allEras.txt` |
| 1b | `DatasetConstructorTask` | build held-out test dataset |
| 2 | `TrainingTask` | train `SimpleMLP_MultiClass`, loss weighting on |
| 3 | `InferenceTask` | score the held-out test set |
| 4 | `ROCCurveTask` | per-class ROC / AUC |

Output: `output/TrainingTask/HPlusCHToWW_2dcats/hwwcom_multiclass_v11_2dcats/SimpleMLP_MultiClass/epochs_30/nominal/best_model.pt`

Key knobs (edit the script's CONFIGURATION block or pass flags):
`--config`, `--model`, `--epochs`, `--batch-size`, `--lr`, `--workers`,
`--no-loss-weighting`, `--train-filelist`, `--test-filelist`, `--version`.

> **Versioning:** `DATASET_VERSION` / `TRAINING_VERSION` namespace the outputs.
> Change them when varying the selection or feature set so runs do not collide
> — e.g. the `_2dcats` suffix keeps this separate from the baseline v11.

> **When comparing two selections**, use **single-era filelists for both** runs.
> The stock filelists are 3-era; using them for one arm and not the other
> confounds the selection change with a training-statistics change.

Other training variants: `train_v11.sh` (baseline, raw CvL/CvB),
`train_v32.sh` (13-class kappa-HCE scheme).

### 3c. Check the ROC

```bash
./check_v11_trainroc.sh
```
Most important pair: **hplusc-vs-higgsbkg** (the ggH shape degeneracy) and
hplusc-vs-tt.

---

## 4. MVA: inference

Applies the trained model to **every** parquet directory:

```bash
B_HIVE_DIR=/eos/home-c/cgupta/HToWW/b-hive \
python3 scripts/mva/run_inference.py \
    --workflow hww_combine_2dcat --year 2022postEE \
    --model-path /eos/user/c/cgupta/EPR_task/b-hive/output/TrainingTask/HPlusCHToWW_2dcats/hwwcom_v11_2dcats_train/hwwcom_multiclass_v11_2dcats/SimpleMLP_MultiClass/epochs_30/nominal/best_model.pt \
    --bhive-config HPlusCHToWW_2dcats
```

Writes `mva_score_*` into `<var>/mva/` for nominal **and every object-shift dir**.

> **Trap — this is the JES/JER bug that cost 500 units.** Inference must cover
> all shift directories, not just nominal. A partial run scored only 19 of 57
> and left object-shift templates frozen at nominal; the limit read 1676 instead
> of 1185. **Always verify the count:**
> ```bash
> find outputs/hww_combine_2dcat/2022postEE -name 'mva' -type d | wc -l
> ```
> Expect one per (dataset × shift-dir). Add new shift dirs (e.g. MET
> unclustered) to this step, not just to the processor.

---

## 5. Build combine inputs

```bash
python3 scripts/combine/make_combine_inputs_v2.py \
        --workflow hww_combine_2dcat --year 2022postEE
```

Output: `outputs/combine/v11_hplusc_2dcat.{root,txt}`

**Two builders exist:**

| builder | notes |
|---|---|
| `make_combine_inputs.py` | v1, canonical. One global binning for all channels. Derives datacard processes from `combine.classes` and **silently ignores** unmatched `process_map` keys. |
| `make_combine_inputs_v2.py` | thin wrapper. Adds per-channel binning (`combine.binning.per_channel`) and explicit `combine.processes`. **Validates** processes against process_map and raises `KeyError` instead of silently dropping. |

> **Trap:** with v1, setting `combine.processes` to split out a process (e.g.
> ggH) **deletes it from the card** while exiting 0 — SR total silently drops
> and the affected lnN row goes all-dashes. v2 raises instead. Use v2.

**Channels** — one per MVA class, assigned by argmax; the winning class's score
is the discriminant:
`SR_hplusc`, `CR_higgsbkg`, `CR_tt`, `CR_st`, `CR_diboson`, `CR_vjets`.

**Verify the card identity** before trusting any limit — builds overwrite in
place, so a card on disk may not be the configuration you think:
```bash
$MAMBA_EXE run -n b_hive python3 -c "
import uproot
f = uproot.open('outputs/combine/v11_hplusc_2dcat.root')
for ch in ['SR_hplusc','CR_higgsbkg','CR_tt','CR_st','CR_diboson','CR_vjets']:
    print(ch, len(f[ch+'_vjets'].values()))"
# baseline (1160 card) = 10 bins everywhere
```

---

## 6. Run the limit

```bash
cd outputs/combine
text2workspace.py v11_hplusc_2dcat.txt -o ws.root

# blind expected limit
combine -M AsymptoticLimits ws.root -t -1 --run blind --noFitAsimov --mass 120

# stat-only
combine -M AsymptoticLimits ws.root -t -1 --run blind --noFitAsimov \
        --freezeParameters allConstrainedNuisances --mass 120
```

Or the wrappers: `scripts/combine/run_limit.sh`, `run_combine.sh`.

Current reference (2022postEE, Asimov, 1 POI): **r95 ≈ 1160 full / 637 stat-only.**

> Combine scans take longer than a foreground shell allows — run them in
> `tmux`, not as a plain background job.

---

## 7. Impacts / nuisance ranking

Cheap approximation — freeze one nuisance at a time and compare:

```bash
for n in <nuisance list>; do
  r=$(combine -M AsymptoticLimits ws.root -t -1 --run blind --noFitAsimov \
        --freezeParameters $n -n _F$n --mass 120 2>&1 \
      | grep "Expected 50" | grep -oE "[0-9]+\.[0-9]+")
  echo "$n $r"
done
```

> When parsing result filenames, **do not** `rsplit('_', 1)` to strip the
> Up/Down suffix — nuisance names contain underscores (`ps_fsr`, `muon_id`,
> `electron_reco_RecoBelow20`) and it silently reads the wrong files.

A nuisance whose limit **improves** when frozen usually signals degeneracy with
a free-floating rateParam, not a broken nuisance.

For the real ranking use `combine -M Impacts` (three-stage: `--doInitialFit`,
`--doFits`, `-o impacts.json`).

---

## Analysis structure

**MVA classes → datacard processes** (`analysis/workflows/hww_combine_2dcat.yaml`):

```yaml
labels.process_groups:          # training labels
  hplusc:   [H+c]
  higgsbkg: [H+b, ggH, VBF, ZH, ggZH, WH, ttHnonBB, ttHtoBB, H(125)]
  tt:       [tt]
  st:       [Single Top]
  diboson:  [WW, WZ, ZZ, qqToZZ, ggToZZ]
  vjets:    [DY+Jets, V+Jets]

combine.classes: [hplusc, higgsbkg, tt, st, diboson, vjets]
combine.process_map:            # datacard processes (must agree with classes)
  ...
```

> `datasets.mc` uses **exact key matching** — no substring, no `-ext`
> expansion. A fileset entry whose `key` is absent from that list is never
> processed (`analysis/filesets/utils.py:185`).

**`tt` is a free-floating rateParam** (`rate_tt`, range [0,5]) — its
normalisation comes from data via CR_tt.

---

## Systematics

Declared in `combine.shape_systematics` (weight-based, read as
`weight_<name>Up/Down`) plus object shifts folded from the shift directories,
plus lnN rows.

| kind | examples |
|---|---|
| weight shapes | `pileup`, `ps_isr`, `ps_fsr`, `scalevar_mu{R,F,RF}`, `lhe_pdf`, `lhe_alphaS`, `CMS_ctag2d_2022`, `CMS_negrw_vjets`, lepton id/iso/reco |
| object shifts | `CMS_scale_j`, `CMS_res_j`, `CMS_scale_e`, `CMS_res_e`, `CMS_scale_m`, `CMS_res_m` |
| lnN | `lumi_13p6TeV`, `xsec_*`, `BR_HtoWW`, `BR_Htautau`, `flavor_composition_ggH` |
| MC stat | `autoMCStats 10` per channel (Barlow–Beeston-lite) |

Adding a **weight-based** systematic = emit `weight_<name>Up/Down` in the
processor + add to `shape_systematics` + **reprocess**.
Adding an **object shift** = new shift dir in the processor + rerun step 1,
**and step 4 inference over the new dirs**.

Trigger SFs are **deliberately disabled** — nominal weights carry no trigger
correction. Known, accepted gap.

---

## Known traps

| trap | detail |
|---|---|
| **Inference coverage** | MVA scoring must cover every shift dir, not just nominal. Cost 500 limit units when missed. |
| **sumw source** | Use the sidecar json, never parquet metadata (5.8× undercount on WtoLNu). |
| **Parent/`-ext` xsecs** | `-ext` cross sections are a *proportional split* of the same total, not extra rate. tt parent-only = 923.41 pb = NNLO. Adding `-ext` naively inflates tt 1.5×. Re-split, don't append. |
| **v1 builder silently drops processes** | `combine.processes` with v1 deletes unmatched processes while exiting 0. Use v2. |
| **Card identity** | Builds overwrite in place. Check per-channel bin counts before trusting a limit. |
| **`jobs_status.py` blacklist** | Memoryless; can delete files from the fileset. |
| **Proxy is node-local** | Use the AFS copy. Never `kinit`. |
| **`/tmp` is node-local** | Workspace files vanish when you reconnect to a different node. Write to AFS/EOS. |
| **Foreground timeout** | Long combine scans die at the shell timeout. Use tmux. |

---

## Related workflow variants

| workflow | difference |
|---|---|
| `hww_combine_2dcat` | **production** — 6-class MVA, 2D c-tag categories |
| `hww_2dcat_nocjet` | c-tag requirement removed (acceptance study) |
| `hww_2dcat_looseWP` | loose instead of medium c-tag WP |
| `hww_2dcat_nocjet_kin` | no c-tag + kinematic cuts in `base` |
