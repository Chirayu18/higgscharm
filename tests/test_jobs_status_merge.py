"""jobs_status fileset regeneration must not truncate the era fileset (2026-11-12)."""
import importlib.util, json, sys, tempfile, unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('jobs_status', ROOT / 'jobs_status.py')
js = importlib.util.module_from_spec(spec); spec.loader.exec_module(js)


class MergeFilesets(unittest.TestCase):
    def test_merge_keeps_other_datasets(self):
        prev = {'A': ['root://x//store/a1'], 'B': ['root://x//store/b1', 'root://x//store/b2']}
        regen = {'B': ['root://y//store/b2', 'root://y//store/b1']}
        out = js.merge_filesets(prev, regen, ['B'])
        self.assertEqual(out['A'], prev['A'])
        self.assertEqual(out['B'], regen['B'])

    def test_changed_store_set_refused(self):
        with self.assertRaises(RuntimeError):
            js.merge_filesets({'B': ['root://x//store/b1']}, {'B': ['root://y//store/b9']}, ['B'])

    def test_update_group_does_not_truncate(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); fs = d / 'filesets'; fs.mkdir(); jobs = d / 'jobs'
            path = fs / 'fileset_2022postEE_nanov12_lxplus.json'
            path.write_text(json.dumps({'A': ['root://x//store/a1'], 'B': ['root://x//store/b1']}))
            (jobs / 'B').mkdir(parents=True)
            (jobs / 'B' / 'arguments.json').write_text(json.dumps({'nfiles': 15}))

            def fake_fetch(cmd, check):  # mimic fetch.py: rewrites with the group only
                path.write_text(json.dumps({'B': ['root://y//store/b1']}))
            with mock.patch.object(js.subprocess, 'run', side_effect=fake_fetch), \
                 mock.patch.object(js, 'modify_site_list'), \
                 mock.patch.object(js, 'get_nano_version', return_value='12'):
                js.update_input_filesets_for_group({'T2_X'}, '2022postEE', fs, jobs, ['B'])
            out = json.loads(path.read_text())
            self.assertEqual(set(out), {'A', 'B'})
            self.assertEqual(out['B'], ['root://y//store/b1'])
            parts = json.loads((jobs / 'B' / 'partitions.json').read_text())
            self.assertEqual(list(parts.values())[0], {'B': ['root://y//store/b1']})


if __name__ == '__main__':
    unittest.main()
