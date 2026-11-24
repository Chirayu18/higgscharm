import io
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import pyarrow as pa
import pyarrow.parquet as pq
from analysis.utils import eos_io
from analysis.processing_checks import check_train_rows


class TestEosSafety(unittest.TestCase):
    def test_server_stat_errors_are_not_absent_files(self):
        fs=types.SimpleNamespace(stat=lambda p:(types.SimpleNamespace(ok=False,message='Permission denied'),None))
        with patch.object(eos_io,'_filesystem',return_value=fs):
            with self.assertRaises(OSError):eos_io._exists_on_server('/eos/user/c/test/a')
        fs.stat=lambda p:(types.SimpleNamespace(ok=False,message='No such file'),None)
        with patch.object(eos_io,'_filesystem',return_value=fs):
            self.assertFalse(eos_io._exists_on_server('/eos/user/c/test/a'))

    def test_parquet_read_falls_back_and_preserves_required_columns(self):
        buf=pa.BufferOutputStream();pq.write_table(pa.table({'sumw':[3.]}),buf);data=buf.getvalue().to_pybytes()
        original=pq.read_table
        def read(source,**kw):
            if isinstance(source,str) and source.startswith('/eos/'):
                raise OSError('stale FUSE')
            return original(source,**kw)
        with patch.object(pq,'read_table',side_effect=read),patch.object(eos_io,'read_bytes',return_value=data):
            self.assertEqual(eos_io.read_parquet('/eos/user/c/test/a',columns=['sumw'])['sumw'].to_pylist(),[3.])
            with self.assertRaises(pa.ArrowInvalid):eos_io.read_parquet('/eos/user/c/test/a',columns=['missing'])

    def test_train_row_coverage_counts_empty_chunks_and_rejects_missing_rows(self):
        acc={'provenance':{'mode':{'train'},'year':{'2022postEE'}},
             'samples':{'A':{'selected_events':2},'B':{'selected_events':0}}}
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'a.parquet';t=pa.table({'event':[1,2]}).replace_schema_metadata({b'processing_mode':b'train',b'year':b'2022postEE',b'sample':b'A'});pq.write_table(t,path)
            self.assertEqual(check_train_rows([acc],[path])[('2022postEE','A')],2)
            with self.assertRaises(ValueError):check_train_rows([acc],[])
            with self.assertRaises(ValueError):check_train_rows([acc],[path,path])
