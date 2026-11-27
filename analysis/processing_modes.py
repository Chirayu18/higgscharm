"""CLI and preflight shared by local execution and Condor preparation."""
from analysis.features.hww import signature, selection_signature
from analysis.models.registry import FoldModels


def add_arguments(parser):
    parser.add_argument('--mode', choices=['legacy', 'train', 'run', 'legacy_slim'], default='legacy',
                        help='legacy outputs; nominal TRAIN ntuples; inline scored RUN templates; model-independent slim trees with all shifts')
    parser.add_argument('--folds', type=int, default=5, help='event modulo k folds, at least 2')
    parser.add_argument('--model_manifest', default=None, help='frozen k-fold ONNX meta.yaml (RUN only)')
    parser.add_argument('--allow_diagnostic_model', action='store_true',
                        help='allow explicitly diagnostic ONNX models for local smoke tests')
    parser.add_argument('--slim_ntuple', action='store_true', help='RUN: retain per-shift features/weights as insurance')


def forwarded(args):
    out = ['--mode', args.mode, '--folds', str(args.folds)]
    if args.model_manifest:
        out += ['--model_manifest', args.model_manifest]
    for key in ('allow_diagnostic_model', 'slim_ntuple'):
        if getattr(args, key):
            out.append('--' + key)
    return out


def validate(config, mode, folds, manifest=None, diagnostic=False, year=None):
    if folds < 2 or folds > 32767:
        raise ValueError('folds must be in [2, 32767]')
    if mode not in ('legacy', 'train', 'run', 'legacy_slim'):
        raise ValueError(f'Unknown processing mode: {mode}')
    if mode == 'legacy':
        if manifest:
            raise ValueError('A model manifest is only used by RUN mode')
        return None
    if config.combine is None or config.mva is None:
        raise ValueError('TRAIN/RUN require an HWW workflow with mva and combine configuration')
    signature(config)
    if len(config.event_selection['categories']) != 1:
        raise ValueError('TRAIN/RUN currently require one exclusive preselection category')
    if mode in ('train', 'legacy_slim'):
        if manifest:
            raise ValueError(f'{mode} does not score a model; do not pass --model_manifest')
        if mode == 'legacy_slim' and (config.combine or {}).get('channel_splits'):
            # Single-SR decision (Chirayu, 2026-11-27): slim trees do not carry split
            # variables, so a split here could never be built offline.
            raise ValueError('legacy_slim builds one merged SR; remove combine.channel_splits')
        return None
    if not manifest:
        raise ValueError('RUN requires --model_manifest; no fallback to the old v11 model')
    if year is None:
        raise ValueError('RUN validation needs the processing year (model era coverage)')
    if list(config.mva['labels']['process_groups']) != list(config.combine['classes']):
        raise ValueError('mva.labels.process_groups order differs from combine.classes')
    return FoldModels(manifest, signature(config), selection_signature(config),
                      config.combine['classes'], folds, diagnostic, year=year)
