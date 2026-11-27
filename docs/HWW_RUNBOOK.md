# HWW runbook — source inspection, 24 November 2026

This guide accompanies the existing HWW source, not a new processing campaign.
Commands below are instructions; publication did not execute processing, training,
inference or fits. Run from the repository root in bash. Use a fresh output version.

## Revisions and supported routes

| Source revision | Use | Limit |
| --- | --- | --- |
| `5b25295c561ca8431a290e018a1641b7998656f3` | Completed nominal TRAIN A/B scoring and expected limits | Merged SR; statistical/configured rates only |
| `efd7030162128c65b922ff47054400107e16ad17` | hDamp/MET/card-policy preparation on `hww-analysis` | Not a completed full-systematics result |
| `c544605` | First model-independent `legacy_slim` implementation | Superseded for inclusive A/B retention |
| `8f0454098fc2168a713ab8dac76a0330c9c35955` | Inclusive legacy-slim A/B plus MET edge handling | Ntuples/pilots, not an offline full-systematics scorer |
| `bf0e75f6c254de3fdbd823afbba02fd79bd2fd7b` | Legacy-slim with one merged SR: Nc-jet channel split removed (decision 2026-11-27) | Same ntuple scope as `8f04540`; feature/selection hashes unchanged |

The publication baseline is a source-only replay of `bf0e75f` (child of `8f04540`) on the published `hww-analysis`
base; the original source SHAs are provenance references, not ancestors of this branch.
The source came from the committed
lxplus branches, with no uncommitted files imported. The canonical baseline checkout
is `/afs/cern.ch/user/c/cgupta/higgscharm_thomas/higgscharm_thomas_new/higgscharm`;
`/eos/user/c/cgupta/higgscharm` is an output root, not that git checkout. The laptop
`~/higgscharm` on `NewWorkflows` is older. Do not run historical QUICKSTART v11 commands
as though they reproduce current A/B results; in particular `make_combine_inputs_v2.py`
and `drive_combine.py` are empty at the inspected baseline.

To reproduce the saved nominal result, make a **separate checkout** at the full
`5b25295...` SHA **from the CERN source repository** (that historical SHA is not
provided by this source-only GitHub branch). Do not change a shared worker checkout. Newer correction source
changes invalidate the saved model selection signature, even if a correction is
opt-in. A model-guard failure is a stop condition; never edit manifest hashes.

## Clone and restore required CERN assets

On lxplus, clone into a new directory and record the published SHA:

```bash
git clone --branch worker/htoww-publish-source --single-branch https://github.com/Chirayu18/higgscharm.git higgscharm-published
cd higgscharm-published
git rev-parse HEAD
```

Restore omitted model/calibration/test assets from the unchanged CERN source checkout.
The following checks every byte hash and refuses to overwrite differing local files.
It does not restore the static-results plotting script. No data processing is started.

```bash
python3 - <<'PYASSETS'
import hashlib, json
from pathlib import Path
source = Path('/afs/cern.ch/user/c/cgupta/higgscharm_legacy_slim_ab_20261118')
for asset in json.loads(Path('docs/HWW_EXTERNAL_ASSETS.json').read_text())['excluded_files']:
    name = asset['path']
    if name.startswith('scripts/plots/'):
        continue
    data = (source / name).read_bytes()
    if hashlib.sha256(data).hexdigest() != asset['sha256']:
        raise RuntimeError('Asset hash mismatch: ' + name)
    target = Path(name)
    if target.exists():
        if target.read_bytes() != data:
            raise RuntimeError('Refusing to overwrite: ' + name)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as output:
            output.write(data)
print('Required omitted assets restored with verified hashes')
PYASSETS
```

The four actual `fileset_<era>_nanov12_lxplus.json` files are also external: use the
frozen campaign inventory for a reproduction, or explicitly generate/approve a new
one. The workflow negrw model path below must be readable. Restored files are local
runtime dependencies and must not be added to the publication branch.

For the latest model-independent writer, after environment setup below:

```bash
# Preparation only: choose a new absolute ROOT and verify filesets/proxy/quota first.
ROOT=/eos/user/c/cgupta/HToWW/new_published_slim_version
hwwpy runner.py -w hww_legacy_slim_ab -y 2022postEE --mode legacy_slim \
  --folds 5 --eos --memory 6000 --nfiles 3 --output_root "$ROOT"
```

This writes job descriptions but does not submit. A deliberate `--submit` addition
launches Condor jobs; do not use it to duplicate the existing PR #6 campaign. The
published writer is not yet a complete slim-to-training/cards/limits pipeline.

## Environment and access

Processing, EOS access and Condor preparation are supported on CERN lxplus with
AFS/EOS/CVMFS mounted. CERN login, CMS VO membership, a valid grid proxy, read access
to the input samples and write quota on the chosen output area are required.
Never commit the proxy or its contents. Renew through the normal CERN flow:

```bash
voms-proxy-init --voms cms --valid 192:00
voms-proxy-info --timeleft
```

The processing image is the actual default in `submit_condor.py`, also used by the
saved nominal scoring commands:

```bash
IMAGE=/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-base-almalinux9:0.7.30-py3.10
hwwpy() {
  apptainer exec -B /afs -B /cvmfs -B /eos -B /tmp -B /run/user \
    --env PYTHONNOUSERSITE=1 --env PYTHONDONTWRITEBYTECODE=1 \
    --env OMP_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 "$IMAGE" python3 "$@"
}
hwwpy runner.py --help
```

Inspected image runtime: Python 3.10.19, coffea 0.7.30, numpy 1.26.4,
awkward 1.10.5, pyarrow 19.0.1, correctionlib 2.7.0, uproot 4.3.7,
ONNX Runtime **1.23.2**. The image name is versioned but no immutable image digest
was recovered. Recheck versions when reproducing. Do not leak CMSSW PYTHONPATH or
LD_LIBRARY_PATH into this container. A generic pip install is not an equivalent setup.

Training uses the separate b-hive checkout `/eos/user/c/cgupta/HToWW/b-hive`, source
`195fd670dbd34c6e3194aa7962d327c78cb44b54` plus its recorded production working-tree
provenance. Its `config/HPlusCHToWW_2dcats.yml` SHA256 is
`2024ecc544912967a47927d8557320336d84cb4ebe5b82b588c2d07db095e5ed`.
Use `/eos/user/c/cgupta/EPR_task/b-hive/micromamba/envs/b_hive/bin/python` and the
existing environment; `env.yml` is a specification, not a historical lockfile.
`setup.sh` sources a machine-local `local_setup.sh`; inspect that file and override
DATA_PATH **after** setup. The private b-hive publication runbook explains the
source-only snapshot and remaining production-diff dependency.

Combine is separate: the completed fits used CMSSW **14_1_0_pre4**, Combine **10.6.0**.
Enter the existing installation's `src` directory and source CMS then `cmsenv`:

```bash
source /cvmfs/cms.cern.ch/cmsset_default.sh
# cd <existing CMSSW_14_1_0_pre4>/src
eval "$(scram runtime -sh)"
combine --version
```

No CMSSW installation or package installation is part of this guide's verification.
The exact installation directory must be supplied on a different CERN account.

## Inputs and external assets

A = `hww_allfix_4fs`; B = `hww_routeb_3fs4fs`.
Eras: `2022preEE 2022postEE 2023preBPix 2023postBPix`.
Dataset keys, DAS queries and cross sections are in
`analysis/filesets/<era>_nanov12.yaml`; actual LFNs/replicas are in
`analysis/filesets/fileset_<era>_nanov12_lxplus.json` on lxplus.
A fresh fetch is not an exact reproduction: freeze and checksum the actual JSON,
workflow and partition inventory. `fileset_checker` may fetch when a fileset is absent.
Do not silently replace private samples or omit missing datasets.

Dated correction paths are in `analysis/corrections/correctionlib_files.py`;
JEC tags are in `jec_params_correctionlib.yaml`. Keep the exact dated CVMFS payloads,
repo calibration JSONs and lepton-MVA ONNX assets. Model manifests record hashes
under `external_inputs`; a checkout alone does not supply all external dependencies.
The negrw v2 asset is referenced by the workflow at
`/afs/cern.ch/user/c/cgupta/negrw_model/negrw_models_v2_20261112.joblib`;
nominal scoring records SHA256
`af53567ccfeddeef94c1fa62b3a6ed3c1285937919ced3f7705c565b819155cb`.
Models/ROOT/parquet/plots remain CERN-side, not new git payloads.

Saved nominal evidence root:
`/afs/cern.ch/user/c/cgupta/bhive_prod_20261112_sol/{A,B}`.
A: `inputs.json`, `registered_20261113/meta.yaml`, `cards_20261113`, `limits_20261113`.
B: `inputs.json`, `registered_20261116/meta.yaml`, `cards_20261116`, `limits_20261116`.
The B manifest SHA256 is
`bab2b1cf262bda2b85d819d5b7b6062c9f5c9faf489e30ee32ac6548ecf91aff`.
Keep relative `fold_0.onnx` ... `fold_4.onnx` beside each meta.yaml.
The split/command directories are
`/eos/user/c/cgupta/HToWW/b-hive/output/prod_20261112_sol_<A-or-B>_{split,cmds}`.
These AFS/EOS files are external dependencies, not bundled by git. On another account
request CERN access or an approved CERNBox copy, preserving hashes and relative paths.

## Legacy, TRAIN and RUN command sequence

The following variable paths are examples to replace with approved new absolute
locations. Do not point ROOT at an existing campaign. Preparation writes job files
and may copy a proxy; omitting `--submit` does **not** mean a read-only dry run.

```bash
W=hww_allfix_4fs
ERA=2022postEE
ROOT=/eos/user/c/cgupta/HToWW/example_new_version
# Prepare only. Add --submit only when an actual campaign is authorized.
hwwpy runner.py -w "$W" -y "$ERA" --eos --mode legacy \
  --output_format parquet --memory 6000 --output_root "$ROOT"
# TRAIN: nominal features, event%5 folds, selected rows and completeness markers.
hwwpy runner.py -w "$W" -y "$ERA" --eos --mode train --folds 5 \
  --memory 6000 --output_root "$ROOT"
```

Condor uses 2 CPUs per processing partition and `workday` by default. `--memory 6000`
is a resource request, not a guarantee that every dataset fits. Files/partition default
is 15 (`--nfiles`); keep partitions/arguments for resumption. Outputs are
`$ROOT/outputs/$W/$ERA/<dataset>` for legacy,
`$ROOT/outputs/${W}__train/$ERA/<dataset>` for TRAIN,
and `${W}__run` for RUN. Coffea markers include zero-selected chunks; never normalize
only selected parquet shards. For non-user EOS roots, supply the matching
`--output_root_url` supported by runner/submit_condor.

`jobs_status.py -w "$W" -y "$ERA" --eos --mode train --output_root "$ROOT"`
uses the same mode/root. It can prompt for regeneration/resubmission; inspect before
accepting. Do not run two owners against the same partitions. A directory existing
is not completion: verify entries, UUID/file coverage, duplicate chunks and rows.

After assembling arrays of **all** relevant coffea markers and parquet shards, including
zero-selected job markers, use the actual helper interfaces:

```bash
# MARKERS and SHARDS are bash arrays populated from the frozen complete inventory.
hwwpy scripts/check_train_outputs.py --coffea "${MARKERS[@]}" --parquets "${SHARDS[@]}"
hwwpy scripts/mva/split_folds.py -w "$W" --coffea "${MARKERS[@]}" \
  --inputs "${SHARDS[@]}" --output "$SPLIT" --folds 5
hwwpy scripts/mva/bhive_fold_commands.py --split "$SPLIT" --output "$CMDS" \
  --prefix new_unique_version --bhive_dir "$BHIVE" --data_path "$DATA_PATH"
```

Set SPLIT/CMDS to new directories, BHIVE to the b-hive checkout and DATA_PATH to its
approved output root. This generates `split_manifest.json`, model_i train/test lists,
`run_folds.sh` and `bhive_runs.json`. The test fold i never enters model i training;
b-hive owns the internal 85/15 validation split. Exclusions require explicit sourced
reasons, not a reduced file list presented as complete. The saved nominal production
also checked the exact expected-partition inventory in `prepare.py` alongside the
saved `inputs.json` and `ready.json`; retain those if reproducing that production.

In b-hive, after setup and explicit training authorization, `bash "$CMDS/run_folds.sh"`
constructs/train/tests all folds. It refuses existing versions; it is not a generic
resume script. Keep completed tasks and provenance, and resume only the established
law/Condor task with matching version, inputs and seeds. GPU production used one GPU,
4 requested CPUs and 24000 MB per fold, `tomorrow` flavour. Queue/hardware vary;
same-seed bitwise equality is not promised across hardware/drivers. No jobs were
started for this publication.

Export in b-hive's Python environment, checkpoints in held-out-fold order:

```bash
"$BHIVE_PYTHON" scripts/mva/export_fold_onnx.py --bhive_path "$BHIVE" \
  --checkpoints "${CHECKPOINTS[@]}" --split "$SPLIT" --output "$ONNX"
hwwpy scripts/mva/register_folds.py -w "$W" --split_manifest "$SPLIT/split_manifest.json" \
  --models "$ONNX"/fold_{0..4}.onnx --eras 2022preEE 2022postEE 2023preBPix 2023postBPix \
  --validation "$ONNX/export_validation.json" --bhive_config "$BHIVE/config/HPlusCHToWW_2dcats.yml" \
  --bhive_runs "$CMDS/bhive_runs.json" --output "$MODEL"
```

Set BHIVE_PYTHON/ONNX/MODEL first. Require full held-out Torch/ONNX validation at
max absolute difference <=1e-5. B export failed in ORT 1.17.3; the unchanged model bytes
passed in production ORT 1.23.2. Preserve the failed evidence; do not loosen the guard.
No `--diagnostic_only`/`--allow_dirty` bypass is a production recipe.

```bash
# Compatible registered model required; old A/B manifests fail newer source guards.
hwwpy runner.py -w "$W" -y "$ERA" --mode run --folds 5 --model_manifest "$MODEL/meta.yaml" \
  --slim_ntuple --eos --memory 6000 --output_root "$ROOT"
# RUN_INPUTS contains every completed RUN accumulator, not TRAIN markers or parquets.
hwwpy scripts/combine/build_cards_from_templates.py -w "$W" -y "$ERA" \
  --inputs "${RUN_INPUTS[@]}" --output "$CARDS"
```

RUN emits held-out-scored nominal/shift templates and optional safety ntuples.
The card builder refuses missing configured shapes unless applicability/exclusions
are explicit. It writes `datacard.txt`, `templates.root`, `template_policy.json`;
it does not execute Combine. Full-systematics compatibility/closure is not established.

Legacy postprocessing is `run_postprocess.py -w "$W" -y "$ERA" --postprocess
--output_format parquet --mva`, followed by `scripts/mva/run_inference.py` and
`scripts/combine/make_combine_inputs.py`. Those legacy tools use historical workflow
inference configuration (v11 in A/B YAML), not the strict fold registry. There is no
verified command sequence here to turn that legacy path into a current full-systematics
result. Do not silently substitute its old event model for registered held-out models.

## Concrete saved nominal example (smallest operational unit: one era)

This is the command shape recorded in B `cards_20261116/commands.json`, using the
**5b25295** checkout and unchanged saved input/model assets. It processes one era's
retained TRAIN tables, so it is not a cheap syntax test. It was not rerun for publication.

```bash
B=/afs/cern.ch/user/c/cgupta/bhive_prod_20261112_sol/B
NEW=/eos/user/c/cgupta/HToWW/example_B_2022preEE_new
hwwpy scripts/combine/build_cards_from_train.py -w hww_routeb_3fs4fs -y 2022preEE \
  --input_manifest "$B/inputs.json" --model_manifest "$B/registered_20261116/meta.yaml" \
  --output "$NEW" --stat_only --also_rate_systematics --merge_channel_splits
```

Expected artifacts: `datacard.txt`, `templates.root`, `template_policy.json`,
`nominal_accumulator.coffea`, plus `rate_systematics/{datacard.txt,templates.root,template_policy.json}`.
Readback must show finite values/variances, nominal equality across the two card
variants, background-only Asimov data, retained autoMCStats and configured free tt rate.
The old TRAIN schema needs `--merge_channel_splits`. The analysis now uses one merged SR (decision 2026-11-27), so this matches the legacy-slim layout.
A uses its A workflow/input/model paths, not B's manifest.

For an expected limit, in the separate established CMSSW runtime and a new fit directory:

```bash
text2workspace.py "$NEW/datacard.txt" -o workspace.root
combine -M AsymptoticLimits workspace.root -t -1 --run blind \
  --mass 125.09 --rMin 0 --rMax 100000 -n _B_stat_only
```

For Run 3 combine all four era cards with explicit channel labels using `combineCards.py`,
then text2workspace and the same blind fit. Saved `limits_20261116/commands.json` gives
all commands and paths. Check exit codes and ROOT `limit` tree: five finite positive
ordered expected quantiles (.025,.16,.5,.84,.975), no observed entry. Saved A/B limits
are central statistical/rate checks, not full-systematics or a stitching-only gain.

## Inclusive legacy-slim route and limits

At `bf0e75f`, `runner.py -w hww_legacy_slim_ab -y 2022postEE --mode legacy_slim
--folds 5 --eos --memory 6000 --nfiles 3 --output_root <NEW_ABSOLUTE_ROOT>` prepares
model-independent shifted trees. The spelling is `legacy_slim`; `--slim_ntuple`
does not activate this mode. The separate `hww_legacy_slim_ab_hdamp` workflow carries
hDamp alternatives. Do not run its deferred inclusive workflow as TRAIN/RUN.
All four legacy-slim workflows (`hww_legacy_slim_ab`, `_ab_hdamp`, `_b`, `_hdamp`) define one
merged `SR_hplusc` plus five argmax CRs; slim trees carry no SR split variable, and
`--mode legacy_slim` refuses a workflow with `combine.channel_splits`.
The post-publication no-tag update (2026-11-27) removes the medium c-tag working-point
cut in all four workflows. The base requires at least one good candidate jet with
pT > 20 GeV, |eta| < 2.4, tight-lepton-veto jet ID and deltaR(lepton, jet) > 0.4.
The legacy `cjets` collection name now denotes these untagged good jets; the separate
`jets` collection still has pT > 30 GeV. Candidate selection remains max ParticleNet
CvL, now over all good candidate jets. Feature names, candidate-based 2D SF treatment
and one-hot definitions are retained; no all-jet SF product or new ranking rule is
introduced by this selection-only update. Calibration provenance remains a separate
open physics input. These objects and selections are recomputed for each object shift.

The selection signature changes. Existing medium-tagged ntuples/pilots and trained
models are not acceptance tests or compatible training for this selection. Use a new
versioned campaign and retrain before scoring; do not reuse the frozen November 18
campaign or bypass model-signature guards. This update does not submit production,
implement a slim training reader or add the offline full-systematics scorer.

Use `analysis.stitching.slim_ab.arm_mask`: A keeps all 4FS/no 3FS; B selects the
complementary GEN masks. Normalize from full preselection sums, not retained rows.

The existing frozen campaign is owned by PR #6, operational directory
`/afs/cern.ch/user/c/cgupta/legacy_slim_ab_campaign_20261118`.
Do not recreate or submit its partitions from this guide. Its pilots passed according
to that PR; full production was unsubmitted at its last report. Current proxy/queue
state was not audited here. Evidence and code patch:
[CERNBox](https://cernbox.cern.ch/files/spaces/eos/user/c/cgupta/HToWW/legacy_slim_ab_20261118/evidence).

The legacy-slim training reader and full-systematics offline scorer are not built in
this revision. Final tt constraint/correlations, Run 2 cards, external theory/c-tag
inputs and negrw closure remain separate work. The candidate
`prescriptions/legacy_slim.yaml` predates the clarified background-only scale scope;
do not adopt it unchanged as a final policy. hDamp alternatives cover dilepton tt,
not every tt decay. Do not imply that prepared MET/hDamp software validates a full
campaign or that nominal cards include all modelling uncertainties.

## Verification record

For publication: source/argparse inspection, runtime version query and CLI --help
checks only; no scientific pipeline executed. Historical checks/results above are
agent-reported from the saved command/manifests and private vault PR #6. A fresh clone
still needs external filesets, calibrations/models, b-hive production diffs/config,
and CERN permissions. Keep large artifacts on EOS, linked via CERNBox, and record
SHA256s alongside the run; do not copy generated evidence into git.

## Source-only publication assets

This branch starts from a text-source replay of `bf0e75f` (earlier `8f04540`) on the
published `hww-analysis` base, with the documented post-publication no-tag update. `HWW_SOURCE_MANIFEST.json` records current hashes and preserves baseline `source_sha256`
values for the changed workflow files. The no-tag update has not been installed in the
original CERN checkout. The older `_hdamp` workflow also restores object dependency
order (leptons before cleaned jets, jets before candidate, MET before transverse mass).
`HWW_EXTERNAL_ASSETS.json` lists the exact omitted binary/model and large calibration
payloads with hashes. Restore required calibration/model/test assets from the original
CERN checkout or an approved CERNBox copy before running; their public redistribution
is not implied. The historical static-results plotting script is also excluded.
The original source histories and working trees are preserved.
