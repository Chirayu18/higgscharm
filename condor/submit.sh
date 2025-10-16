#!/bin/bash -xe

JOBID=$1
BASEDIR=$2
X509PATH=$3

export HOME=`pwd`
if [ -d /afs/cern.ch/user/${USER:0:1}/$USER ]; then
  export HOME=/afs/cern.ch/user/${USER:0:1}/$USER
fi

export XRD_NETWORKSTACK=IPv4
export XRD_RUNFORKHANDLER=1
export X509_USER_PROXY=$X509PATH
voms-proxy-info -all -file $X509PATH

WORKDIR=`pwd`

declare -A ARGS
for key in workflow year output_path output_format dataset eos user; do
    ARGS[$key]=$(python3 -c "import json; print(json.load(open('$WORKDIR/arguments.json'))['$key'])")
done

OPTS="--workflow ${ARGS[workflow]} --year ${ARGS[year]} --output_path ${ARGS[output_path]} --output_format ${ARGS[output_format]} --dataset ${ARGS[dataset]}_$JOBID --user ${ARGS[user]}"

if [ "${ARGS[eos]}" = "True" ]; then
    OPTS="$OPTS --eos"
fi

# get partition fileset
python3 -c "import json; json.dump(json.load(open('$WORKDIR/partitions.json'))['$JOBID'], open('$WORKDIR/partition_fileset.json', 'w'), indent=4)"
OPTS="$OPTS --partition_json $WORKDIR/partition_fileset.json"

echo $OPTS

cd $BASEDIR
python3 submit.py $OPTS