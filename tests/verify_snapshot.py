"""Synthetic lifecycle regression plus consistency of the two calibrated cases."""
import importlib.util
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parent.parent
spec=importlib.util.spec_from_file_location('replay',ROOT/'tools/replay_snapshot.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def snapshot(blocks,trace):
    return {'segments':[{'device':0,'blocks':blocks,'total_size':4096,
        'allocated_size':sum(b['size'] for b in blocks if b['state']=='active_allocated'),
        'active_size':sum(b['size'] for b in blocks if b['state']!='inactive')}],
        'device_traces':[trace]}

class Lifecycle(unittest.TestCase):
    def test_alias_reuse_async_and_peaks(self):
        b={'address':4096,'size':512,'state':'active_allocated','frames':[]}
        params=[{'device':'npu:0','address':4096,'storage_bytes':512}]*2
        start=snapshot([b],[])
        trace=[{'action':'snapshot'},
            {'action':'alloc','addr':8192,'size':700,'frames':[{'filename':'/x/_fsdp_collectives.py','name':'all_gather'}]},
            {'action':'free_requested','addr':8192},
            # Async release is pending while another category is allocated.
            {'action':'alloc','addr':12288,'size':700,'frames':[{'filename':'/x/muon/step.py','name':'step'}]},
            {'action':'free_completed','addr':8192},
            {'action':'free_requested','addr':12288},
            {'action':'free_completed','addr':12288},
            # Same address, new lifetime.
            {'action':'alloc','addr':8192,'size':1},
            {'action':'free_requested','addr':8192},
            {'action':'free_completed','addr':8192}]
        r,_,_=m.replay(start,snapshot([b],trace),params,200)
        self.assertEqual(r['peak']['bytes'],1536)
        self.assertEqual(r['active_peak_bytes'],2560)
        self.assertEqual(r['end_allocated_bytes'],512)
        self.assertEqual(r['peak']['categories']['parameter_shard'],512)
        self.assertGreater(sum(r['category_independent_peaks'].values()),r['peak']['bytes'])
        with self.assertRaises(AssertionError):m.replay(start,snapshot([b],trace),params,len(trace))
    def test_real_case_consistency(self):
        data=json.loads((ROOT/'data/memory-snapshot-20261023.json').read_text())
        for arm,r in data.items():
            self.assertEqual(sum(r['peak']['categories'].values()),r['peak']['bytes'],arm)
            self.assertEqual(r['allocated_peak_error_bytes'],0)
            self.assertEqual(r['active_peak_error_bytes'],0)
            self.assertEqual(r['reserved_error_bytes'],0)
            self.assertLess(r['event_count'],200000)
            self.assertEqual(r['allocator_reserved_bytes'],r['allocator']['reserved_bytes.all.current'])
            self.assertLess(r['peak']['bytes'],r['active_peak_bytes'])
            self.assertGreater(r['category_independent_peaks']['optimizer_state_or_workspace'],13*1024**3)
        self.assertEqual(data['bf16']['parameter_unique_storage_bytes'],data['mxfp8']['parameter_unique_storage_bytes'])
    def test_embedded_fallback_matches_files(self):
        html=(ROOT/'index.html').read_text(encoding='utf-8')
        import re
        embedded=json.loads(re.search(r'var SNAPSHOT_REFERENCE = (.*);',html).group(1))
        self.assertEqual(embedded,json.loads((ROOT/'data/memory-snapshot-20261023.json').read_text()))
        presets=json.loads(re.search(r'var PRESETS = (.*?);',html,re.S).group(1))
        self.assertEqual(presets,json.loads((ROOT/'presets.json').read_text(encoding='utf-8'))['presets'])

if __name__=='__main__':unittest.main()
