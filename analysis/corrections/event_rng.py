"""Reproducible per-object random numbers.

Each number is a pure function of (run, luminosityBlock, event, index of the
object in its NanoAOD collection, purpose). So the same event gets the same
smearing whatever the chunk size, worker count or processing order.
Chirayu, 2026-11-10: "Make reproduceable".
"""
import numpy as np
import awkward as ak
from scipy.special import ndtri

# One salt per use, so that two corrections never share a random stream.
SALTS = {
    "muon_resolution": 0x6D75,   # "mu"
    "electron_smearing": 0x656C,  # "el"
}

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)


def _mix(x):
    """splitmix64 finaliser, vectorised on uint64 (wrap-around is intended)."""
    with np.errstate(over="ignore"):
        x = x + _GOLDEN
        x = (x ^ (x >> np.uint64(30))) * _M1
        x = (x ^ (x >> np.uint64(27))) * _M2
    return x ^ (x >> np.uint64(31))


def _u64(a):
    return np.asarray(ak.to_numpy(a)).astype(np.int64).view(np.uint64)


def object_uniforms(events, collection, purpose):
    """Flat numpy array of uniforms in (0, 1), one per object of events[collection]."""
    salt = np.uint64(SALTS[purpose])
    objs = events[collection]
    counts = ak.to_numpy(ak.num(objs))
    run = np.repeat(_u64(events.run), counts)
    lumi = np.repeat(_u64(events.luminosityBlock), counts)
    evt = np.repeat(_u64(events.event), counts)
    idx = _u64(ak.flatten(ak.local_index(objs)))
    h = _mix(evt)
    h = _mix(h ^ run)
    h = _mix(h ^ lumi)
    h = _mix(h ^ ((idx << np.uint64(16)) | salt))
    return ((h >> np.uint64(11)).astype(np.float64) + 0.5) * (2.0 ** -53)


def object_normals(events, collection, purpose):
    """Flat numpy array of standard-normal numbers, one per object."""
    return ndtri(object_uniforms(events, collection, purpose))
