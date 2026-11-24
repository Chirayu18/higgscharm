#!/bin/bash
# Condor executable: negrw v2 ensemble training + validation inside coffea-base-almalinux9:0.7.30-py3.10
set -euo pipefail
echo "[exec] host $(hostname) date $(date)"
python3 -c "import sklearn,numpy,pandas;print('sklearn',sklearn.__version__,'numpy',numpy.__version__,'pandas',pandas.__version__)"
mkdir -p out
python3 negrw_train_v2.py --train train_v2.parquet --sr sr_allfix_v1_20261110.parquet \
   --old-model /afs/cern.ch/user/c/cgupta/negrw_model/negrw_models.joblib --outdir out
ls -la out
echo "[exec] done $(date)"
