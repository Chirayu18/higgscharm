"""
EOS listings that do not trust the FUSE mount (ported 2026-10-29 from the HcZZ branch,
Saranya's combine/scripts/eos_io.py, 2026-09-25).

On lxplus929 and lxplus971 the eosxd FUSE mount served stale, EMPTY listings for
directories whose files `eos ls` / xrootd showed fine, after ~40 min of heavy reading.
glob/os.listdir raise nothing, so readers silently got a subset (in her case 21% of one
shift template and whole sumw dirs went missing). Listings here come from the xrootd
server, which is authoritative. Paths outside /eos use the normal filesystem calls.
"""
import io
import fnmatch
import glob as _glob
import os
import re

XRD_URL = "root://eosuser.cern.ch"
_fs = None
N_DIFF = 0


def _filesystem():
    global _fs
    if _fs is None:
        from XRootD import client

        _fs = client.FileSystem(XRD_URL)
    return _fs


def _is_eos(path):
    return str(path).startswith("/eos/")


def _server_path(path):
    # /eos/home-c/cgupta/... is the same namespace as /eos/user/c/cgupta/...
    return re.sub(r"^/eos/home-(\w)/", r"/eos/user/\1/", str(path))


def listdir(path):
    """Entry names of a directory, sorted, no dot entries (EOS .sys.v# version dirs).
    Returns [] if the directory does not exist; raises on any other server error."""
    path = str(path)
    if not _is_eos(path):
        try:
            return sorted(n for n in os.listdir(path) if not n.startswith("."))
        except FileNotFoundError:
            return []
    from XRootD.client.flags import DirListFlags

    st, lst = _filesystem().dirlist(_server_path(path), DirListFlags.STAT)
    if not st.ok:
        msg = st.message.lower()
        # missing path, or a file where a directory was expected: glob.glob also gives []
        if "no such file" in msg or "not found" in msg or "not a directory" in msg:
            return []
        raise OSError(f"xrootd dirlist failed for {path}: {st.message}")
    names = sorted(e.name for e in lst if not e.name.startswith("."))
    global N_DIFF
    try:
        fuse = set(n for n in os.listdir(path) if not n.startswith("."))
    except OSError:
        fuse = set()
    if fuse != set(names):
        N_DIFF += 1
        if N_DIFF <= 20:
            print(
                f"  NOTE eos_io: FUSE listing of {path} differs from server "
                f"({len(fuse)} vs {len(names)} entries); using server listing",
                flush=True,
            )
    return names


def glob(pattern):
    """glob.glob replacement (no ** support) whose /eos listings come from the server."""
    pattern = str(pattern)
    if not _is_eos(pattern):
        return _glob.glob(pattern)
    parts = pattern.strip("/").split("/")
    paths = [""]
    for part in parts:
        nxt = []
        for base in paths:
            if _glob.has_magic(part):
                nxt += [f"{base}/{n}" for n in listdir(base or "/") if fnmatch.fnmatchcase(n, part)]
            else:
                nxt.append(f"{base}/{part}")
        paths = nxt
    # the literal tail components were not listed, so keep only those that exist
    return [p for p in paths if os.path.lexists(p) or _exists_on_server(p)]


def _exists_on_server(path):
    st, _ = _filesystem().stat(_server_path(path))
    if st.ok:
        return True
    if _missing(st):
        return False
    raise OSError(f"xrootd stat failed for {path}: {st.message}")


# Adapted from Saranya HcZZ-zx-estimation 7962523, combine/scripts/eos_io.py.
# Keep requested parquet columns strict: missing physics inputs must not be dropped.
def _missing(status):
    msg = status.message.lower()
    return any(s in msg for s in ("no such file", "not found", "not a directory"))


def isdir(path):
    if not _is_eos(path):
        return os.path.isdir(path)
    from XRootD.client.flags import StatInfoFlags
    st, info = _filesystem().stat(_server_path(path))
    if not st.ok:
        if _missing(st):
            return False
        raise OSError(f"xrootd stat failed for {path}: {st.message}")
    return bool(info.flags & StatInfoFlags.IS_DIR)


def read_bytes(path, prefer_server=False):
    path = str(path)
    if not prefer_server:
        try:
            with open(path, "rb") as stream:
                return stream.read()
        except OSError:
            if not _is_eos(path):
                raise
    from XRootD import client
    with client.File() as stream:
        status, _ = stream.open(XRD_URL + "/" + _server_path(path))
        if not status.ok:
            raise OSError(f"xrootd open failed for {path}: {status.message}")
        status, data = stream.read()
        if not status.ok:
            raise OSError(f"xrootd read failed for {path}: {status.message}")
    return data


def read_parquet(path, columns=None):
    import pyarrow as pa
    import pyarrow.parquet as pq
    try:
        return pq.read_table(path, columns=columns)
    except (OSError, pa.ArrowInvalid):
        # A stale/truncated FUSE view raises ArrowInvalid (not OSError); retry from
        # the server for /eos paths only. A server-side failure still raises.
        if not _is_eos(path):
            raise
        return pq.read_table(io.BytesIO(read_bytes(path, prefer_server=True)), columns=columns)
