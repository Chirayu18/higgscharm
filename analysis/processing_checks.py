"""Processing coverage checks adapted from HcZZ bookkeeping (707c813).

HWW checks fail instead of displaying incomplete rows as if they were usable.
"""
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq


def check_input_entries(partition, output, year):
    import uproot
    from analysis.filesets.utils import get_dataset_config
    from analysis.sinks.hww import dataset_info
    expected = {}
    seen = set()
    for dataset, files in partition.items():
        sample, _ = dataset_info(dataset, get_dataset_config(year))
        expected.setdefault(sample, 0)
        for path in files:
            with uproot.open(path, timeout=600) as root:
                identity = (sample, str(root.file.uuid))
                if identity in seen:
                    raise ValueError(f'Duplicate NanoAOD input: {path}')
                seen.add(identity)
                expected[sample] += root['Events'].num_entries
    for sample, n in expected.items():
        actual = output.get('samples', {}).get(sample, {}).get('events_read', 0)
        if actual != n:
            raise ValueError(f'{sample}: processed {actual} entries, expected {n}')
    # 'files' (sample -> logical /store paths) lets downstream consumers prove that
    # every fileset file of every expected sample was processed by some job.
    files = {}
    for dataset, paths in partition.items():
        sample, _ = dataset_info(dataset, get_dataset_config(year))
        files.setdefault(sample, set()).update(store_path(x) for x in paths)
    return {'expected_entries': sum(expected.values()),
            'processed_entries': sum(output['samples'][s]['events_read'] for s in expected),
            'files': files}


def store_path(path):
    """Replica-independent identity of an input file (the /store/... LFN)."""
    path = str(path)
    i = path.find('/store/')
    return path[i:] if i >= 0 else path


def sample_kind(sample, info):
    """'mc' or 'data' from the era fileset entry; anything else is an error.

    Simulation is era mc/signal. Collision data carries a run-letter era (C, D,
    E, F, G, C123, C4, ...) and process 'Data'. Never test era == 'data'.
    """
    era = str(info.get('era', '')).strip()
    if era in ('mc', 'signal'):
        return 'mc'
    if str(info.get('process', '')).strip() == 'Data':
        return 'data'
    raise ValueError(f'{sample}: cannot classify era={era!r} process={info.get("process")!r} as MC or data')


def load_exclusions(path):
    """YAML {sample: reason}. Every exclusion needs a non-empty provenance reason."""
    if not path:
        return {}
    import yaml
    data = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f'{path}: exclusions must map sample -> reason')
    for sample, reason in data.items():
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f'{path}: exclusion of {sample} needs a provenance reason')
    return data


def expected_mc_samples(workflow, year):
    from analysis.filesets.utils import get_dataset_config, get_datasets_to_run_over
    config = get_dataset_config(year)
    return {s for s in get_datasets_to_run_over(workflow, year) if sample_kind(s, config[s]) == 'mc'}


def fileset_files(year, fileset_json=None):
    import json
    from analysis.filesets.utils import get_nano_version
    if fileset_json is None:
        fileset_json = (Path.cwd() / 'analysis' / 'filesets' /
                        f'fileset_{year}_nanov{get_nano_version(year)}_lxplus.json')
    data = json.loads(Path(fileset_json).read_text())
    return {k: {store_path(x) for x in v} for k, v in data.items()}


def check_dataset_coverage(workflow, year, present, processed_files, exclusions=None,
                           fileset_json=None, expected=None):
    """Every MC sample the workflow runs over is present and fully processed.

    present: samples found in the supplied outputs. processed_files: sample ->
    set of /store paths recorded in execution_checks. Excluded samples must be
    listed explicitly with a reason; stale exclusions are errors too.
    """
    exclusions = exclusions or {}
    expected = expected_mc_samples(workflow, year) if expected is None else set(expected)
    stale = set(exclusions) - expected
    if stale:
        raise ValueError(f'Exclusions name samples the workflow does not run over: {sorted(stale)}')
    overlap = set(exclusions) & set(present)
    if overlap:
        raise ValueError(f'Excluded samples are present in the inputs: {sorted(overlap)}')
    missing = expected - set(exclusions) - set(present)
    if missing:
        raise ValueError(f'{year}: {len(missing)} expected MC samples have no output: {sorted(missing)}')
    unexpected = set(present) - expected
    if unexpected:
        raise ValueError(f'{year}: outputs contain MC samples outside the workflow: {sorted(unexpected)}')
    reference = fileset_files(year, fileset_json)
    for sample in sorted(set(present)):
        if sample not in reference:
            raise ValueError(f'{sample}: absent from the {year} fileset JSON')
        got = processed_files.get(sample)
        if got is None:
            raise ValueError(f'{sample}: outputs carry no execution_checks file record')
        lost = reference[sample] - got
        extra = got - reference[sample]
        if lost or extra:
            raise ValueError(f'{sample}: {len(lost)} fileset files unprocessed, '
                             f'{len(extra)} processed files not in the fileset')
    return {'expected': sorted(expected), 'excluded': dict(sorted(exclusions.items()))}


def check_train_rows(accumulators, parquet_files):
    from coffea import processor
    expected, chunks = {}, {}
    for total in accumulators:
        if total['provenance']['mode'] != {'train'} or len(total['provenance']['year']) != 1:
            raise ValueError('TRAIN row coverage needs per-era TRAIN accumulators')
        year = next(iter(total['provenance']['year']))
        for sample, entry in total['samples'].items():
            key = (year, sample)
            source_chunks = entry.get('chunks', set())
            if chunks.get(key, set()) & source_chunks:
                raise ValueError(f'Duplicate TRAIN chunks for {key}')
            chunks.setdefault(key, set()).update(source_chunks)
            expected[key] = expected.get(key, 0) + entry.get('selected_events', 0)
    actual = {k: 0 for k in expected}
    seen = set()
    for path in parquet_files:
        path = str(Path(path).resolve())
        if path in seen:
            raise ValueError('Duplicate TRAIN parquet input')
        seen.add(path)
        pf = pq.ParquetFile(path);md=pf.schema_arrow.metadata or {}
        if md.get(b'processing_mode') != b'train':
            raise ValueError(f'Not a TRAIN shard: {path}')
        key = (md[b'year'].decode(),md[b'sample'].decode())
        if key not in actual:
            raise ValueError(f'Parquet without matching TRAIN job output: {key}')
        actual[key] += pf.metadata.num_rows
    if actual != expected:
        raise ValueError(f'Incomplete TRAIN rows: expected {expected}, found {actual}')
    return actual
