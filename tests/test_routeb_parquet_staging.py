"""parquet_writer local staging (2026-11-12): concurrent jobs sharing a cwd must not touch each other's files.

Condor jobs run with cwd = the shared AFS repo; Run A and Route B TRAIN build the same output name for the same
chunk. Run from the repository root inside the coffea 0.7.30 container:
    python3 -m unittest tests.test_routeb_parquet_staging
"""
import os
import tempfile
import unittest
from pathlib import Path

import awkward as ak
import pyarrow.parquet as pq

from analysis.utils import parquet_writer as pw

FNAME = "a138dee6-afea-11f1-a6d8-28319e86beef_%2FEvents%3B1_0-85732.parquet"


class TestLocalStaging(unittest.TestCase):
    def setUp(self):
        self._cwd = os.getcwd()
        self.tmp = tempfile.TemporaryDirectory()
        self.work = Path(self.tmp.name) / "shared_cwd"
        self.out = Path(self.tmp.name) / "out"
        self.work.mkdir()
        os.chdir(self.work)
        # another job's staged file with the same name, mid-flight in the shared cwd
        self.foreign = self.work / FNAME
        self.foreign.write_bytes(b"other job")

    def tearDown(self):
        os.chdir(self._cwd)
        self.tmp.cleanup()

    def _assert_clean(self):
        self.assertEqual(self.foreign.read_bytes(), b"other job")  # neither overwritten nor deleted
        self.assertEqual(sorted(p.name for p in self.work.iterdir()), [FNAME])  # no staging leftovers

    def test_dump_pa_table(self):
        arrays = {"x": ak.Array([1.0, 2.0, 3.0]), "n": ak.Array([1, 2, 3])}
        pw.dump_pa_table(arrays, FNAME, str(self.out), ["wf__train", "2022postEE", "ds_1", "base"],
                         extra_metadata={"sample": "ds"})
        dest = self.out / "wf__train" / "2022postEE" / "ds_1" / "base" / FNAME
        t = pq.read_table(dest)
        self.assertEqual(t.column("x").to_pylist(), [1.0, 2.0, 3.0])
        self.assertEqual(t.column("n").to_pylist(), [1, 2, 3])
        self.assertEqual(t.schema.metadata[b"sample"], b"ds")
        self._assert_clean()

    def test_dump_pa_table_empty_writes_nothing(self):
        pw.dump_pa_table({"x": ak.Array([])}, FNAME, str(self.out), ["a"])
        self.assertFalse((self.out / "a" / FNAME).exists())
        self._assert_clean()

    def test_dump_ak_array(self):
        pw.dump_ak_array(ak.Array([{"x": 1.0}, {"x": 2.0}]), FNAME, str(self.out), ["b"])
        self.assertEqual(ak.from_parquet(str(self.out / "b" / FNAME)).x.tolist(), [1.0, 2.0])
        self._assert_clean()

    def test_names_unique(self):
        a, b = pw._local_tmp_name(FNAME), pw._local_tmp_name(FNAME)
        self.assertNotEqual(a, b)
        self.assertTrue(a.startswith(FNAME + ".") and a.endswith(".tmp"))


if __name__ == "__main__":
    unittest.main()
