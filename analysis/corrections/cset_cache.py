"""Process-level cache of correctionlib CorrectionSets (HTOWW-FRAMEWORK-SPEEDUP-2, plan P1-perf (b)).

Several correctors call CorrectionSet.from_file() on every object-shift pass (15 x per chunk),
re-parsing the same JSON(.gz) each time. A CorrectionSet is read-only once built, so the parsed
object is reused for the life of the worker process. Keyed by the path string; the files cannot
change during a job. Evaluation results are unchanged; only the parse is skipped.
"""
from functools import lru_cache

import correctionlib


@lru_cache(maxsize=None)
def _load(path: str):
    return correctionlib.CorrectionSet.from_file(path)


def load_correction_set(path):
    """Drop-in for correctionlib.CorrectionSet.from_file(path), cached per process."""
    return _load(str(path))
