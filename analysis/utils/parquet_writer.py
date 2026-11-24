import os
import shutil
import pathlib
import numpy as np
import awkward as ak
from typing import List, Optional


def _xrd_copy(local_file: str, destination: str, attempts: int = 4):
    """xrootd copy with retries (ported 2026-10-29 from the HcZZ branch). Several coffea
    workers of one job write into the same new EOS directory at once (e.g. sumw_records/),
    and EOS can answer one of them with "[3018] Unable to mkdir: File exists". The
    directory exists by the next attempt, so a short retry is enough."""
    import time
    import XRootD.client

    for attempt in range(attempts):
        copyproc = XRootD.client.CopyProcess()
        copyproc.add_job(local_file, destination, force=True)
        copyproc.prepare()
        status, _ = copyproc.run()
        del copyproc
        if status.status == 0:
            return
        if attempt < attempts - 1:
            time.sleep(2 * (attempt + 1))
    raise Exception(status.message)


# 2026-11-05 (HTOWW-FRAMEWORK-SPEEDUP-2, plan P1-perf (a)): pa.Table.from_pydict() on an
# awkward-1 Array iterates it element by element (ak.Array has no __arrow_array__), which was
# 38% of processor CPU in the 2026-10-08 profile. Convert flat awkward columns to numpy first,
# reproducing EXACTLY the Arrow type the element-wise path infers from Python scalars:
#   float* -> float64 (double), int*/uint* -> int64, bool -> bool, None -> null (validity
#   mask), all-None -> Arrow `null` type, NaN stays a NaN value (not null).
# Anything else (records, ragged lists, strings, uint64 >= 2**63, conversion errors) keeps the
# original awkward object, i.e. the old slow path, so the output cannot change silently.
_INT64_MAX = np.iinfo(np.int64).max


def _to_arrow_like_pydict(array, pa):
    if not isinstance(array, ak.Array):
        return array  # numpy (weights, negrw g) and Python lists: unchanged
    try:
        if array.ndim != 1:
            return array
        np_arr = ak.to_numpy(array, allow_missing=True)
    except Exception:
        return array
    if np.ma.isMaskedArray(np_arr):
        mask = np.ma.getmaskarray(np_arr)
        data = np.ma.getdata(np_arr)
    else:
        mask = None
        data = np_arr
    if data.dtype.names is not None or data.ndim != 1:
        return array
    if mask is not None and len(mask) > 0 and mask.all():
        return pa.nulls(len(mask))  # the element-wise path infers type `null`
    kind = data.dtype.kind
    if kind == "f":
        data = data.astype(np.float64, copy=False)
    elif kind == "i":
        data = data.astype(np.int64, copy=False)
    elif kind == "u":
        valid = data if mask is None else data[~mask]
        if valid.size and valid.max() > _INT64_MAX:
            return array  # Python inference would give uint64: keep the old path
        data = data.astype(np.int64)
    elif kind == "b":
        data = data.astype(np.bool_, copy=False)
    else:
        return array
    if mask is not None and mask.any():
        return pa.array(data, mask=mask)
    return pa.array(data)


def dump_pa_table(
    arrays: dict,
    fname: str,
    location: str,
    subdirs: Optional[List[str]] = None,
    extra_metadata: Optional[dict] = None,
):
    subdirs = subdirs or []
    xrd_prefix = "root://"
    pfx_len = len(xrd_prefix)
    xrootd = False
    if xrd_prefix in location:
        try:
            import XRootD
            import XRootD.client

            xrootd = True
        except ImportError as err:
            raise ImportError(
                "Install XRootD python bindings with: conda install -c conda-forge xroot"
            ) from err
    local_file = (
        os.path.abspath(os.path.join(".", _local_tmp_name(fname)))
        if xrootd
        else os.path.join(".", _local_tmp_name(fname))
    )
    merged_subdirs = "/".join(subdirs) if xrootd else os.path.sep.join(subdirs)
    destination = (
        location + merged_subdirs + f"/{fname}"
        if xrootd
        else os.path.join(location, os.path.join(merged_subdirs, fname))
    )
    import pyarrow as pa
    import pyarrow.parquet as pq

    out = {}
    for variable, array in arrays.items():
        # plain Python lists (e.g. dump_chunk_sumw's {"sumw": [val]}) have no .ndim;
        # only awkward arrays need the 2D->firsts flattening.
        if hasattr(array, "ndim") and array.ndim == 2:
            array = ak.firsts(array)
        out[variable] = _to_arrow_like_pydict(array, pa)

    table = pa.Table.from_pydict(out)
    if extra_metadata:
        existing = dict(table.schema.metadata or {})
        for k, v in extra_metadata.items():
            kk = k.encode() if isinstance(k, str) else k
            vv = v.encode() if isinstance(v, str) else v
            existing[kk] = vv
        table = table.replace_schema_metadata(existing)
    if len(table) != 0:  # skip dataframes with empty entries
        pq.write_table(table, local_file)
        if xrootd:
            _xrd_copy(local_file, destination)
        else:
            dirname = os.path.dirname(destination)
            pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
            shutil.copy(local_file, destination)
            assert os.path.isfile(destination)
        pathlib.Path(local_file).unlink()


def _local_tmp_name(fname: str) -> str:
    """Unique local staging name for `fname` (2026-11-12). Condor jobs run with cwd = the shared AFS repo
    (condor/submit.sh: cd $BASEDIR), and workflows processing the same chunk build the same `fname`; a fixed
    local name let concurrent jobs delete or overwrite each other's staged file before the copy. The
    destination name is unchanged, so outputs are identical."""
    import uuid

    return f"{fname}.{os.getpid()}.{uuid.uuid4().hex}.tmp"


# Function taken from HiggsDNA
# https://gitlab.cern.ch/HiggsDNA-project/HiggsDNA/-/blob/master/higgs_dna/utils/dumping_utils.py
def dump_ak_array(
    akarr: ak.Array,
    fname: str,
    location: str,
    subdirs: Optional[List[str]] = None,
) -> None:
    """
    Dump an ak array to disk at location/'/'.join(subdirs)/fname.
    """
    subdirs = subdirs or []
    xrd_prefix = "root://"
    pfx_len = len(xrd_prefix)
    xrootd = False
    if xrd_prefix in location:
        try:
            import XRootD  # type: ignore
            import XRootD.client  # type: ignore

            xrootd = True
        except ImportError as err:
            raise ImportError(
                "Install XRootD python bindings with: conda install -c conda-forge xroot"
            ) from err
    local_file = (
        os.path.abspath(os.path.join(".", _local_tmp_name(fname)))
        if xrootd
        else os.path.join(".", _local_tmp_name(fname))
    )
    merged_subdirs = "/".join(subdirs) if xrootd else os.path.sep.join(subdirs)
    destination = (
        location + merged_subdirs + f"/{fname}"
        if xrootd
        else os.path.join(location, os.path.join(merged_subdirs, fname))
    )
    ak.to_parquet(ak.fill_none(akarr, [], axis=0), local_file)
    if xrootd:
        _xrd_copy(local_file, destination)
    else:
        dirname = os.path.dirname(destination)
        pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
        shutil.copy(local_file, destination)
        assert os.path.isfile(destination)
    pathlib.Path(local_file).unlink()


def dump_chunk_sumw(events, workflow, year, output_location):
    """Write the chunk's full generator sumw to a dedicated record, ALWAYS.

    The data-shard sumw (in dump_parquet) is correct per chunk, but a chunk that
    selects zero events writes no shard at all (base.py guards dump on
    nevents_after > 0, and dump_pa_table skips empty tables), so its sumw is
    silently lost. For low-efficiency samples (vjets, hadronic) most chunks select
    nothing, so the summed parquet sumw can undercount the true generator sumw by
    10-30x -- breaking the lumi*xsec/sumw normalisation.

    This writes a one-row `{sumw: <full chunk genWeight sum>}` parquet per chunk
    into <dataset>/sumw_records/, computed on the pre-selection events, regardless
    of how many events survive. Summing this directory gives the true generator
    sumw (validated to <0.5% against the .coffea cutflow and the Runs-tree
    genEventSumw). MC only.
    """
    if not hasattr(events, "genWeight"):
        return
    dataset = events.metadata["dataset"]
    chunk_sumw = float(ak.sum(events.genWeight))
    pkey = events.behavior["__events_factory__"]._partition_key.replace("/", "_")
    fname = pkey + ".parquet"
    subdirs = [workflow, year, dataset, "sumw_records"]
    dump_pa_table(
        {"sumw": [chunk_sumw]},
        fname,
        output_location,
        subdirs,
        extra_metadata={"sumw": str(chunk_sumw)},
    )


# 2026-11-05 (HTOWW-FRAMEWORK-SPEEDUP-2): dump_parquet needs only xsec/era for the shard
# metadata, but re-read and re-parsed the ~1000-line era YAML on every call (15 x per chunk,
# ~1.2 s per 86k-event chunk). The YAML cannot change inside a job, so parse it once per
# process. Keyed by (cwd, year) because get_dataset_config resolves the path from Path.cwd().
# A deep copy is handed out so no caller can mutate the cached dict.
_DATASET_CONFIG_CACHE = {}


def _dataset_config_cached(year):
    import copy
    from pathlib import Path
    from analysis.filesets.utils import get_dataset_config

    key = (str(Path.cwd()), year)
    if key not in _DATASET_CONFIG_CACHE:
        _DATASET_CONFIG_CACHE[key] = get_dataset_config(year)
    return copy.deepcopy(_DATASET_CONFIG_CACHE[key])


def dump_parquet(
    events,
    weights_container,
    variables_map,
    workflow,
    year,
    category,
    output_location,
    shift,
):
    is_mc = hasattr(events, "genWeight")
    dataset = events.metadata["dataset"]

    if shift is None:
        # nominal: full weight set (nominal + per-syst Up/Down columns)
        if is_mc:
            variations = ["nominal", *weights_container.variations]
            for variation in variations:
                if variation == "nominal":
                    variables_map[f"weight_nominal"] = weights_container.weight()
                    for partial_weight in weights_container.weightStatistics:
                        variables_map[f"weight_{partial_weight}"] = (
                            weights_container.partial_weight(include=[partial_weight])
                        )
                else:
                    variables_map[f"weight_{variation}"] = weights_container.weight(
                        modifier=variation
                    )
        subdirs = [workflow, year, dataset, category]
    else:
        # object shift: kinematics already reflect the shift; only the nominal
        # weight is needed (weight-systematics are not crossed with object shifts).
        # Output goes to its own <shift> subdir so the MVA/combine path can read
        # each variation independently.
        if is_mc:
            variables_map["weight_nominal"] = weights_container.weight()
        subdirs = [workflow, year, dataset, category, shift]

    # Self-normalising metadata: per-shard sumw (genWeight unchanged by object
    # shifts) + xsec/era from the dataset config. merge_parquets aggregates sumw
    # across shards. Lets downstream compute lumi*xsec/sumw without a sidecar.
    extra_metadata = None
    if is_mc:
        dataset_info = _dataset_config_cached(year).get(dataset, {})
        extra_metadata = {
            "sumw": str(float(ak.sum(events.genWeight))),
            "xsec": str(dataset_info.get("xsec", "")),
            "era":  str(dataset_info.get("era", "")),
        }

    fname = (
        events.behavior["__events_factory__"]._partition_key.replace("/", "_")
        + ".parquet"
    )
    dump_pa_table(variables_map, fname, output_location, subdirs, extra_metadata=extra_metadata)
