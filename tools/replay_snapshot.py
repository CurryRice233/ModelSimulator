"""Replay trusted, locally collected torch_npu snapshots. No tensor payloads."""
import collections
import csv
import hashlib
import json
import math
import pickle
from pathlib import Path

import argparse
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'

def category(frames):
    for i, f in enumerate(frames):
        file=f.get('filename','').replace('\\','/')
        name=f.get('name','')
        if '_fsdp_collectives.py' in file:
            following=' '.join(x.get('name','') for x in frames[i:i+4])
            return 'fsdp_reduce_buffer' if 'reduce' in following else 'fsdp_allgather_buffer'
        if '_fsdp_param.py' in file:
            return 'fsdp_unsharded_parameter'
        if any(x in file for x in ['swap_overlap.py','/muon/','fused_mat_mul.py','packed_expert.py']) or file.endswith('/ns.py'):
            return 'optimizer_state_or_workspace'
        if '/quant.py' in file and 'ascend_deepep' in file:
            return 'expert_mxfp8_quantization'
        if file.endswith('/mx_quant.py'):
            return 'projection_mxfp8_quantization'
        if file.endswith('/mx_quant_matmul.py'):
            return 'projection_mxfp8_matmul_output'
        if file.endswith('/fused.py') and 'ascend_deepep' in file:
            return 'megamoe_output_or_workspace'
        if 'mega_moe_redispatch.py' in file or 'dispatcher_mega_moe' in file:
            return 'megamoe_redispatch_or_saved_activation'
        if '/uam/' in file or '/offload/' in file or file.endswith('/cpu.py'):
            return 'uam_restore_or_buffer'
        if any(x in file for x in ['modeling_deepseek','linear.py','mhc_','sparse_flash','swiglu']):
            return 'model_activation_or_operator_workspace'
    return 'unattributed'

def replay(start,end,params,limit):
    assert len(end['device_traces'][0])<limit
    trace=end['device_traces'][0]
    begin=len(start['device_traces'][0])
    assert trace[:begin]==start['device_traces'][0], 'Window prefix differs'
    assert trace[begin]['action']=='snapshot', 'Missing start marker'
    param_ranges={int(x['address']):x for x in params if x['device'].startswith('npu') and x['storage_bytes']}
    blocks={}
    for seg in start['segments']:
        if seg['device']!=0: continue
        for b in seg['blocks']:
            if b['state']=='inactive':continue
            c='parameter_shard' if b['address'] in param_ranges else category(b.get('frames',[]))
            blocks[b['address']]=dict(size=b['size'],allocated=b['state']=='active_allocated',category=c,frames=b.get('frames',[]),born=-1)
    allocated=sum(x['size'] for x in blocks.values() if x['allocated'])
    active=sum(x['size'] for x in blocks.values())
    categories=collections.Counter()
    for b in blocks.values():
        if b['allocated']:categories[b['category']]+=b['size']
    peak=dict(bytes=allocated,index=begin,categories=dict(categories)); active_peak=active
    category_peaks=categories.copy(); stack_traffic=collections.Counter(); rows=[]; failures=[]
    for i,e in enumerate(trace[begin+1:],begin+1):
        a=e['action']; addr=e.get('addr'); raw=e.get('size',0)
        if a=='alloc':
            assert addr not in blocks, ('Duplicate live address',i,addr)
            # This frozen allocator uses 512-byte rounding. Verify against exact end blocks and peak counters below.
            size=((raw+511)//512)*512
            c=category(e.get('frames',[]))
            blocks[addr]=dict(size=size,allocated=True,category=c,frames=e.get('frames',[]),born=i)
            allocated+=size;active+=size;categories[c]+=size
            stack=';'.join(f"{Path(f.get('filename','')).name}:{f.get('name','')}:{f.get('line',0)}" for f in e.get('frames',[])[:6])
            stack_traffic[(c,stack)]+=size
        elif a=='free_requested':
            assert addr in blocks, ('Unknown free',i,addr)
            b=blocks[addr];assert b['allocated'],('Double request',i,addr)
            allocated-=b['size'];categories[b['category']]-=b['size'];b['allocated']=False
        elif a=='free_completed':
            assert addr in blocks,('Unknown completed',i,addr)
            b=blocks.pop(addr);assert not b['allocated'],('Missing request',i,addr)
            active-=b['size']
        elif a not in ['snapshot','segment_map','segment_unmap','segment_alloc','segment_free','workspace_snapshot']:
            failures.append((i,a))
        if allocated>peak['bytes']:
            peak=dict(bytes=allocated,index=i,categories=dict(categories),blocks=[dict(address=k,**v) for k,v in blocks.items() if v['allocated']])
        active_peak=max(active_peak,active)
        for c,n in categories.items():category_peaks[c]=max(category_peaks[c],n)
        if i%50==0:rows.append(dict(event=i,allocated_bytes=allocated,active_bytes=active))
    expected={b['address']:b for s in end['segments'] if s['device']==0 for b in s['blocks'] if b['state']!='inactive'}
    assert set(blocks)==set(expected),('End address mismatch',len(blocks),len(expected))
    assert all(b['size']==expected[k]['size'] for k,b in blocks.items()),'End block size mismatch'
    assert allocated==sum(s['allocated_size'] for s in end['segments'] if s['device']==0)
    assert active==sum(s['active_size'] for s in end['segments'] if s['device']==0)
    assert not failures,failures
    return dict(peak=peak,active_peak_bytes=active_peak,end_allocated_bytes=allocated,
                end_active_bytes=active,end_reserved_bytes=sum(s['total_size'] for s in end['segments'] if s['device']==0),
                category_independent_peaks=dict(category_peaks),event_count=len(trace),
                prefix_events=begin,end_state_exact=True),rows,stack_traffic

def main():
    report={}
    for arm in ['bf16','mxfp8']:
        d=RESULTS/f'snapshot-{arm}'
        start=pickle.load(open(d/'rank-0-step12-start.pickle','rb'))
        end=pickle.load(open(d/'rank-0-step12-end.pickle','rb'))
        params=json.loads((d/'rank-0-parameters-step11.json').read_text())
        meta=json.loads((d/'rank-0-snapshot.json').read_text())
        r,rows,traffic=replay(start,end,params,meta['max_entries'])
        measured=meta['allocator_step12']
        r['allocator']=measured
        r['allocated_peak_error_bytes']=r['peak']['bytes']-measured['allocated_bytes.all.peak']
        r['active_peak_error_bytes']=r['active_peak_bytes']-measured['active_bytes.all.peak']
        workspaces={e['addr']:e['size'] for e in end['device_traces'][0] if e['action']=='workspace_snapshot'}
        r['snapshot_workspace_segment_bytes']=sum(workspaces.values())
        r['allocator_reserved_bytes']=r['end_reserved_bytes']-sum(workspaces.values())
        r['reserved_error_bytes']=r['allocator_reserved_bytes']-measured['reserved_bytes.all.current']
        assert r['reserved_error_bytes']==0, 'Unexpected segment coverage beyond allocator/workspaces'
        assert r['allocated_peak_error_bytes']==0 and r['active_peak_error_bytes']==0
        r['source_snapshot_sha256']=meta['step12-end']['sha256']
        r['parameter_dtypes']=dict(collections.Counter(x['dtype'] for x in params))
        r['parameter_unique_storage_bytes']=sum({(x['device'],x['address']):x['storage_bytes'] for x in params}.values())
        with (ROOT/f'{arm}-timeline.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
        with (ROOT/f'{arm}-allocation-traffic.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(['category','stack','cumulative_allocated_bytes_not_peak'])
            for (c,s),n in traffic.most_common():w.writerow([c,s,n])
        (ROOT/f'{arm}-peak-blocks.json').write_text(json.dumps(r['peak'].pop('blocks'),indent=2))
        report[arm]=r
        print(arm,json.dumps({k:v for k,v in r.items() if k!='allocator'},ensure_ascii=False))
    (ROOT/'snapshot-analysis.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence_root',type=Path,help='Private directory containing results/snapshot-bf16 and snapshot-mxfp8; writes analysis here')
    args=parser.parse_args()
    ROOT=args.evidence_root.resolve()
    RESULTS=ROOT/'results'
    main()
