#!/usr/bin/env bash
# CMSSW + combine wrapper for the higgscharm combine pipeline.
#
# Sources the combine environment, then runs drive_combine.py which reads the
# combine.run block from the workflow yaml and executes text2workspace +
# AsymptoticLimits + MultiDimFit + the manual impact loop.
#
# Usage:
#   bash scripts/combine/run_combine.sh hww_MVA
#
# Override the CMSSW location with CMSSW_SRC if needed.

set -e

WORKFLOW=${1:?usage: run_combine.sh <workflow>}
CMSSW_SRC=${CMSSW_SRC:-/afs/cern.ch/user/c/cgupta/CMSSW_14_1_0_pre4/src}
HIGGSCHARM=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

source /cvmfs/cms.cern.ch/cmsset_default.sh
cd "$CMSSW_SRC"
eval "$(scram runtime -sh)"

cd "$HIGGSCHARM"
python3 scripts/combine/drive_combine.py -w "$WORKFLOW"
