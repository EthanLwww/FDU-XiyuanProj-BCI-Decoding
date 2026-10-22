"""Sample only this experiment's process memory and GPU allocations."""
import json
from pathlib import Path
import subprocess
import time
from run_scaling_residual import ROOT
from streaming_data import save_json

peaks={}
while True:
    gpu={}
    query=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory',
                          '--format=csv,noheader,nounits'],capture_output=True,text=True)
    for line in query.stdout.splitlines():
        try:
            pid,mb=[int(v.strip()) for v in line.split(',')]
            gpu[pid]=gpu.get(pid,0)+mb
        except ValueError:
            pass
    samples=[]
    for p in Path('/proc').glob('[0-9]*'):
        try:
            args=(p/'cmdline').read_bytes().split(b'\0')
            if str(ROOT).encode() not in args or b'train' not in args:
                continue
            group=args[args.index(b'--group')+1].decode()
            phase=args[args.index(b'--phase')+1].decode()
            status=dict(line.split(':',1) for line in (p/'status').read_text().splitlines() if ':' in line)
            rss=int(status.get('VmRSS','0 kB').split()[0])/1024
            key=group+'/'+phase
            row=dict(group=group,phase=phase,pid=int(p.name),rss_mib=rss,gpu_mib=gpu.get(int(p.name)))
            samples.append(row)
            peak=peaks.setdefault(key,dict(sampled_peak_rss_mib=0,sampled_peak_gpu_mib=None,samples=0))
            peak['sampled_peak_rss_mib']=max(peak['sampled_peak_rss_mib'],rss)
            if row['gpu_mib'] is not None:
                peak['sampled_peak_gpu_mib']=max(peak['sampled_peak_gpu_mib'] or 0,row['gpu_mib'])
            peak['samples']+=1
        except (FileNotFoundError,PermissionError,ProcessLookupError,ValueError):
            continue
    with (ROOT/'resource_samples.jsonl').open('a') as f:
        f.write(json.dumps(dict(unix=time.time(),processes=samples))+'\n')
    save_json(ROOT/'resource_peaks.json',dict(interval_seconds=30,peaks=peaks,
              note='Sampled process GPU memory includes CUDA context; sampling can miss transient peaks.'))
    try:
        state=json.loads((ROOT/'suite.status.json').read_text())['status']
        if state in ('complete','failed'):
            break
    except FileNotFoundError:
        pass
    time.sleep(30)
