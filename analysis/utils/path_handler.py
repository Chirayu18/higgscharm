import os
import pathlib


class Paths:

    def __init__(self, eos: bool, root: str = None) -> None:
        if root:
            # explicit output root (--output_root); must be absolute and is never guessed
            root_path = pathlib.Path(root)
            if not root_path.is_absolute():
                raise ValueError(f"--output_root must be an absolute path: {root}")
            self.root_path = root_path
        elif eos:
            # finds the /eos user directory
            user = os.environ["USER"]
            self.root_path = pathlib.Path(f"/eos/user/{user[0]}/{user}/higgscharm")
        else:
            # finds the root path as the directory one level upwards of where this file is located
            self.root_path = pathlib.Path(__file__).resolve().parent.parent

    def workflow_path(
        self,
        workflow: str,
        year: str,
        dataset: str
    ) -> pathlib.Path:
        """
        Safely return a path by creating the parent directories to avoid errors when writing to the path.

        Parameters:
        -----------
            path: Path to optionally create and return.
            mkdir: If True, creates the parent directories. If False, it has no effect.

        Returns:
            Input path.
        """
        workflow_path = "/".join(
            [
                elem
                for elem in [
                    workflow,
                    year,
                    dataset
                ]
                if elem is not None
            ]
        )
        # make output directory
        output_path = self.root_path / "outputs" / workflow_path
        if not output_path.exists():
            output_path.mkdir(parents=True)
        return output_path