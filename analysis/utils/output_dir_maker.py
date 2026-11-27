from analysis.utils.path_handler import Paths


def output_workflow_name(workflow, mode="legacy"):
    """Directory name for a processing mode: legacy keeps the bare workflow name."""
    return workflow if mode in (None, "legacy") else f"{workflow}__{mode}"


def make_output_directory(args, workflow_name=None) -> str:
    """builds output directories. Returns output path

    workflow_name overrides args.workflow for the directory (TRAIN/RUN use
    '<workflow>__<mode>'), so the directory that is CREATED is the one jobs save
    into. args.output_root (optional) replaces the default root path.
    """
    paths = Paths(eos=args.eos, root=getattr(args, "output_root", None))
    path_args = {}
    for arg in ["workflow", "year", "dataset"]:
        if arg in args:
            path_args[arg] = vars(args)[arg]
        else:
            path_args[arg] = None
    if workflow_name is not None:
        path_args["workflow"] = workflow_name
    workflow_output_path = paths.workflow_path(**path_args)
    return str(workflow_output_path)


def parquet_output_location(args) -> str:
    """Root URL/path under which parquet shards are written (always ends in '/outputs/').

    Default (no output_root): unchanged legacy behaviour. With output_root the
    parquet location is derived from the same root as the .coffea outputs; an EOS
    user-space root is written through the eosuser xrootd endpoint, as before.
    Any other /eos root needs an explicit output_root_url (no guessed endpoint).
    """
    root = getattr(args, "output_root", None)
    if not root:
        return None  # submit.py keeps its existing --eos / AFS default
    url = getattr(args, "output_root_url", None)
    if url:
        return url.rstrip("/") + "/outputs/"
    if root.startswith("/eos/user/"):
        return "root://eosuser.cern.ch/" + root.rstrip("/") + "/outputs/"
    if root.startswith("/eos/"):
        raise ValueError(f"{root}: give --output_root_url (xrootd endpoint) for non-user EOS roots")
    return root.rstrip("/") + "/outputs/"
