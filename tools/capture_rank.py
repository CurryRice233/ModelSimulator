"""Task-local all-rank completed-step meter; repository sources stay frozen."""
import collections
import hashlib
import inspect
import json
import os
from pathlib import Path
import runpy
import sys
import time
import pickle

out=Path(os.environ['RUN_DIR'])
rank=int(os.environ['RANK'])
arm=os.environ['BENCH_ARM']
entry=Path(os.environ['REPO_ROOT'])/'train_fsdp2.py'
sys.argv[0]=str(entry)
# Preserve the real entrypoint's torch/NPU/transfer/logger import order before
# installing observations. A fresh module name also avoids the launcher's
# pre-existing __main__ event logger.
namespace=runpy.run_path(str(entry),run_name='precision_benchmark_entry')
torch=sys.modules['torch']
torch_npu=sys.modules['torch_npu']
TrainMonitor=namespace['TrainMonitor']
Trainer=namespace['Trainer']
from mindspeed_llm.fsdp2.data.megatron_data import megatron_dataset_generate as dataset_builder
original_dataset_config=dataset_builder.core_gpt_dataset_config_from_args


def writable_dataset_config(*args,**kwargs):
    config=original_dataset_config(*args,**kwargs)
    config.path_to_cache=str(Path(os.environ['BENCH_ROOT'])/'data-cache')
    return config


dataset_builder.core_gpt_dataset_config_from_args=writable_dataset_config
samples=[]
snapshot_rank = int(os.environ.get('SNAPSHOT_RANK', '0'))
entry_limit = int(os.environ.get('SNAPSHOT_MAX_ENTRIES', '200000'))
memory_api = torch_npu.npu.memory
capture = dict(rank=rank, capture_rank=snapshot_rank, max_entries=entry_limit,
               target_step=12, exporter='pickle(_snapshot()); same schema as _dump_snapshot, without profiler fallback')
trainer_ref = None

def storage_inventory(model, label):
    rows=[]
    for name, p in model.named_parameters():
        t = p._local_tensor if hasattr(p, '_local_tensor') else p
        if t.device.type == 'meta':
            continue
        s=t.untyped_storage()
        rows.append(dict(name=name, device=str(t.device), dtype=str(t.dtype),
                         shape=list(t.shape), numel=t.numel(), address=s.data_ptr(),
                         storage_bytes=s.nbytes(), offset=t.storage_offset()))
    (out/f'rank-{rank}-parameters-{label}.json').write_text(json.dumps(rows))

def save_snapshot(label):
    snap=memory_api._snapshot()
    path=out/f'rank-{rank}-{label}.pickle'
    with path.open('xb') as f:
        pickle.dump(snap,f,protocol=4)
    traces=snap.get('device_traces', [])
    meta=dict(path=path.name, bytes=path.stat().st_size,
              sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
              segments=len(snap.get('segments', [])),
              trace_lengths=[len(t) for t in traces],
              first_actions=[[e.get('action') for e in t[:3]] for t in traces],
              last_actions=[[e.get('action') for e in t[-3:]] for t in traces],
              event_counts=[dict(collections.Counter(e['action'] for e in t)) for t in traces])
    capture[label]=meta
    return meta

last=None
original_step=TrainMonitor.step
signature=inspect.signature(original_step)


def measured_step(self,*args,**kwargs):
    global last
    bound=signature.bind(self,*args,**kwargs).arguments
    result=original_step(self,*args,**kwargs)
    torch.npu.synchronize()
    now=time.perf_counter_ns()
    step=int(bound['current_step'])
    stats=memory_api.memory_stats()
    mem={k:stats.get(k) for k in ['allocated_bytes.all.current','allocated_bytes.all.peak',
        'active_bytes.all.current','active_bytes.all.peak','reserved_bytes.all.current',
        'reserved_bytes.all.peak','inactive_split_bytes.all.current','num_alloc_retries','num_ooms']}
    samples.append(dict(step=step, memory=mem, time_ns=time.time_ns(),
        completed_step_ms=None if last is None else (now-last)/1e6,
        grad_norm=float(bound['grad_norm']),loss=float(result[0]['avg_loss'])))
    last=now
    if step == 11 and rank == snapshot_rank:
        storage_inventory(trainer_ref.model, 'step11')
        memory_api._record_memory_history(enabled='all', context='alloc', stacks='python', max_entries=entry_limit)
        capture['start_time_ns']=time.time_ns()
        save_snapshot('step12-start')
    if step == 12 and rank == snapshot_rank:
        capture['end_time_ns']=time.time_ns()
        meta=save_snapshot('step12-end')
        capture['no_capacity_wrap']=all(n < entry_limit for n in meta['trace_lengths'])
        capture['allocator_step12']=mem
        memory_api._record_memory_history(enabled=None)
        (out/f'rank-{rank}-snapshot.json').write_text(json.dumps(capture,indent=2))
        assert capture['no_capacity_wrap'], 'Snapshot history reached capacity: rerun with larger event budget'
    memory_api.reset_peak_memory_stats()
    return result


TrainMonitor.step=measured_step
original_init=Trainer.__init__


def checked_init(self,*args,**kwargs):
    global trainer_ref
    if rank == snapshot_rank:
        memory_api._record_memory_history(enabled='state', context='alloc', stacks='python', max_entries=entry_limit)
    original_init(self,*args,**kwargs)
    trainer_ref=self
    storage_inventory(self.model, 'init')
    classes=collections.Counter(type(m).__name__ for m in self.model.modules())
    grouped=sum(hasattr(m,'_mxfp8_config') for m in self.model.modules())
    quantized=sum(n for name,n in classes.items() if 'MXLinear' in name)+grouped
    assert (quantized>0)==(arm=='mxfp8'),(arm,classes,grouped)
    assert self.parallel_args.megamoe_precision==arm
    assert not self.optimization_args.use_fused_compressor
    assert bool(self.args.profile)==(os.environ['BENCH_PROFILE']=='True')
    metadata=dict(rank=rank,arm=arm,quantized_projection_modules=quantized,
        grouped_mxfp8_modules=grouped,classes=dict(classes),steps=self.args.max_steps,
        profile=bool(self.args.profile),seed=self.args.seed,
        torch=str(torch.__version__),torch_npu_path=str(Path(torch_npu.__file__).resolve()))
    (out/f'rank-{rank}-metadata.json').write_text(json.dumps(metadata,indent=2))


Trainer.__init__=checked_init
status=dict(rank=rank,arm=arm,status='FAIL',started=time.time())
try:
    namespace['main']()
    assert len(samples)==15 and [x['step'] for x in samples]==list(range(1,16)), 'Incomplete steps'
    if arm=='bf16':
        from fsdp_turbo.distributed.expert_parallel import dispatcher_mega_moe as bf
        torch.npu.synchronize()
        torch.distributed.barrier()
        for pool in bf._WINDOW_POOL_STATE['pools'].values():
            for op in [*pool['ops'],*pool['modules'].values()]:
                op.close()
        if bf._MEGA_MOE_STATE['shmem'] is not None:
            bf._MEGA_MOE_STATE['shmem'].destroy()
    status.update(status='PASS',exitcode=0)
except BaseException as error:
    status.update(error_type=type(error).__name__,error=str(error),exitcode=1)
    raise
finally:
    libraries={}
    for line in Path('/proc/self/maps').read_text().splitlines():
        parts=line.split(maxsplit=5)
        if len(parts)==6 and parts[5].startswith('/') and '.so' in parts[5] and any(
                name in parts[5] for name in ('ascend_deepep','shmem','libtorch_npu')):
            p=Path(parts[5])
            libraries[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else 'MISSING'
    status.update(finished=time.time(),samples=samples,loaded_libraries=libraries,
        timing_boundary='completed TrainMonitor.step to next completed TrainMonitor.step; all-device synchronize')
    (out/f'rank-{rank}-exit.json').write_text(json.dumps(status,indent=2))
