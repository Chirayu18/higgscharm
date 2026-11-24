#!/bin/bash -xe

# Pin the same isolated Python/runtime behavior used in the validation container.
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

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
# Mode/output flags are optional for old prepared jobs. Write them NUL-separated to a
# file and check the helper's exit status: a failure inside a process substitution is
# invisible to `bash -e` and would silently run legacy mode into a __mode path.
python3 - "$WORKDIR/arguments.json" "$WORKDIR/mode_opts.bin" <<'PYMODE'
import json, sys
args = json.load(open(sys.argv[1]))
items = ['--mode', args.get('mode', 'legacy'), '--folds', str(args.get('folds', 5))]
if args.get('model_manifest'):
    items += ['--model_manifest', args['model_manifest']]
for key in ('allow_diagnostic_model', 'slim_ntuple'):
    if args.get(key): items.append('--' + key)
if args.get('output_location'):
    items += ['--output_location', args['output_location']]
open(sys.argv[2], 'wb').write(b'\0'.join(x.encode() for x in items) + b'\0')
PYMODE
MODE_OPTS=()
while IFS= read -r -d '' value; do MODE_OPTS+=("$value"); done < "$WORKDIR/mode_opts.bin"
[ "${#MODE_OPTS[@]}" -ge 4 ] || { echo "mode options missing" >&2; exit 1; }
python3 submit.py $OPTS "${MODE_OPTS[@]}"
