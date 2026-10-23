"""Resource-bounded paper EEGNet queue; reuse completed folds and publish audited plots."""
import argparse
import fcntl
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiment_runners.paper_pipeline import save_json


def make_jobs(subjects, scope):
    labels = [(1,4)] if scope=='binary' else [(1,4),(1,2,4),(1,2,3,4)]
    bands = [(4,40)] if scope!='frequency' else [(4,40),(8,13),(13,30),(4,8),(.5,4)]
    jobs = []
    for band, group, task, subject in itertools.product(bands,labels,['ME','MI'],subjects):
        key=f'S{subject:02d}_{task}_{"-".join(map(str,group))}_{band[0]:g}-{band[1]:g}'
        jobs.append(dict(key=key,subject=subject,task=task,labels=list(group),band=list(band)))
    return jobs


def gpu_snapshot():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu',
                                 '--format=csv,noheader,nounits'],text=True)
    return [dict(index=int(a),uuid=b.strip(),memory_mib=int(c),utilization=int(d))
            for a,b,c,d in (line.split(',') for line in raw.strip().splitlines())]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True); parser.add_argument('--data-root',required=True)
    parser.add_argument('--subjects',nargs='+',type=int,default=list(range(1,22)))
    parser.add_argument('--scope',choices=['binary','core','frequency'],default='core')
    parser.add_argument('--gpus',nargs='+',required=True,help='Physical GPU UUIDs; jobs start only on an idle GPU')
    parser.add_argument('--plan-only',action='store_true'); args=parser.parse_args()
    root=Path(args.root); root.mkdir(parents=True,exist_ok=True)
    lock=(root/'suite.lock').open('w')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise SystemExit('A suite controller already owns this directory')
    jobs=make_jobs(args.subjects,args.scope)
    plan=dict(subjects=args.subjects,scope=args.scope,gpu_uuids=args.gpus,jobs=jobs,
              max_concurrent_jobs=len(args.gpus),independent_test_set=False,
              note='EEGNet offline only; FBCSP and ERD panels are not included')
    plan_path=root/'suite.plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text())!=plan:
        raise ValueError('Existing suite plan differs; use another directory')
    save_json(plan_path,plan)
    if args.plan_only: print(json.dumps(dict(conditions=len(jobs),folds=5*len(jobs)))); return
    pending=jobs.copy(); running={}; complete=[]; failed=[]; adopted=[]
    for job in list(pending):
        pilot=root/f'S{job["subject"]:02d}_ME_4-40'
        if job['task']=='ME' and job['labels']==[1,4] and job['band']==[4,40] and (pilot/'config.json').exists():
            # The three pilot processes were launched before this controller. Never duplicate them.
            adopted.append(dict(job=job,directory=pilot)); pending.remove(job)
    started=time.time()
    while pending or running or adopted:
        for gpu,entry in list(running.items()):
            rc=entry['process'].poll()
            if rc is None: continue
            entry['log'].close()
            if rc==0 and (entry['directory']/'result.json').exists(): complete.append(entry['job']['key'])
            else: failed.append(dict(key=entry['job']['key'],returncode=rc,log=str(entry['directory']/'run.log')))
            del running[gpu]
        for entry in adopted[:]:
            directory=entry['directory']
            if (directory/'result.json').exists(): complete.append(entry['job']['key']); adopted.remove(entry)
            elif (directory/'error.json').exists():
                failed.append(dict(key=entry['job']['key'],error=str(directory/'error.json'))); adopted.remove(entry)
        stats=gpu_snapshot()
        by_uuid={g['uuid']:g for g in stats}
        with (root/'resources.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(time=time.time(),gpus=stats,running={g:e['job']['key'] for g,e in running.items()}))+'\n')
        for gpu in args.gpus:
            if gpu in running or not pending or by_uuid[gpu]['memory_mib']>500: continue
            job=pending.pop(0); directory=root/'runs'/job['key']; directory.mkdir(parents=True,exist_ok=True)
            if (directory/'result.json').exists():
                complete.append(job['key']); continue
            cmd=[sys.executable,'-u','-m','experiment_runners.paper_reproduction',
                 '--data-root',args.data_root,'--cache',str(root/'cache'),'--out',str(directory),
                 '--subject',str(job['subject']),'--task',job['task'],'--labels',*map(str,job['labels']),
                 '--band',*map(str,job['band'])]
            env=os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2')
            log=(directory/'run.log').open('a')
            process=subprocess.Popen(cmd,cwd=Path(__file__).parents[1],env=env,stdout=log,stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL)
            running[gpu]=dict(process=process,log=log,job=job,directory=directory)
            print(f'Start {job["key"]} on {gpu}, pid={process.pid}',flush=True)
        save_json(root/'suite.status.json',dict(state='running' if pending or running or adopted else 'complete',
                  conditions_total=len(jobs),completed=complete,failed=failed,pending=len(pending),
                  running=[dict(key=e['job']['key'],pid=e['process'].pid,gpu=g) for g,e in running.items()],
                  adopted=[dict(key=e['job']['key'],directory=str(e['directory'])) for e in adopted],
                  elapsed_seconds=time.time()-started,updated_at=time.time()))
        if len(complete) and len(complete)!=getattr(main,'last_report_count',-1):
            result=subprocess.run([os.environ.get('PAPER_REPORT_PYTHON',sys.executable),'-m','experiment_runners.paper_report',str(root)],
                                  cwd=Path(__file__).parents[1],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            (root/'report.log').write_text(result.stdout)
            if result.returncode:
                raise RuntimeError('Result audit/report failed; see report.log')
            main.last_report_count=len(complete)
        if pending or running or adopted: time.sleep(30)
    if failed: raise SystemExit('One or more runs failed; see suite.status.json')


if __name__=='__main__': main()
