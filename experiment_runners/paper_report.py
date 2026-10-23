"""Audit completed paper CV outputs and produce incrementally updated PNG/CSV figures."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiment_runners.paper_pipeline import save_json, sha256


def audit_condition(directory):
    directory = Path(directory)
    summary = json.loads((directory/'result.json').read_text())
    cfg = summary['config']
    if cfg['smoke'] or summary['completed_folds'] != 5:
        raise ValueError('Smoke/partial run cannot be included as a formal result')
    if sorted(r['fold'] for r in summary['folds']) != list(range(1,6)):
        raise ValueError('Expected exactly folds 1 through 5')
    held_out, held_windows, correct, windows, true_total = [], [], 0, 0, 0
    trial_accuracies, window_accuracies, splits = [], [], []
    for row in summary['folds']:
        fd = directory/f'fold{row["fold"]}'
        split = json.loads((fd/'split.json').read_text())
        splits.append(split)
        if set(split['train_trials']) & set(split['val_trials']):
            raise ValueError(f'Trial leakage in {fd}')
        if set(split['train_windows']) & set(split['val_windows']):
            raise ValueError(f'Window leakage in {fd}')
        for partition in ('train','val'):
            for unit in ('trials','windows'):
                values = split[f'{partition}_{unit}']
                if len(values) != len(set(values)) or len(values) != row[f'n_{partition}_{unit}']:
                    raise ValueError(f'Duplicate or incorrect split size in {fd}')
        if row['checkpoint_sha256'] != sha256(fd/'best.weights.h5'):
            raise ValueError(f'Checkpoint changed: {fd}')
        predictions = np.load(fd/'predictions.npz')
        probabilities = predictions['probabilities']
        if not np.isfinite(probabilities).all():
            raise ValueError(f'Nonfinite probabilities in {fd}')
        if not np.isclose(probabilities.sum(axis=1),1,atol=1e-5).all():
            raise ValueError('Probabilities not normalized')
        ids = predictions['trial_indices']
        if probabilities.shape != (len(ids),len(cfg['labels'])) or len(ids) != row['n_val_windows']:
            raise ValueError('Prediction window count/shape mismatch')
        if np.any(probabilities<0) or np.any(probabilities>1):
            raise ValueError('Probabilities out of range')
        if sorted(np.unique(ids).tolist()) != sorted(split['val_trials']):
            raise ValueError('Prediction trial coverage mismatch')
        starts = predictions['raw_starts']
        if starts.shape != ids.shape or np.any(starts%128) or np.any(starts<0) or np.any(starts>4096):
            raise ValueError('Incorrect raw window timing')
        if len(set(zip(ids.tolist(),starts.tolist()))) != len(ids):
            raise ValueError('Duplicated raw window in validation predictions')
        truth_encoded = predictions['true_encoded_labels']
        if truth_encoded.shape != ids.shape or np.any(truth_encoded<0) or np.any(truth_encoded>=len(cfg['labels'])):
            raise ValueError('Incorrect encoded label array')
        actual_window = float(np.mean(probabilities.argmax(axis=1)==predictions['true_encoded_labels']))
        if not np.isclose(actual_window,row['window_accuracy'],atol=1e-7):
            raise ValueError('Stored window accuracy does not match predictions')
        with (fd/'trial_predictions.csv').open() as stream:
            trial_rows = list(csv.DictReader(stream))
        if len(trial_rows) != row['n_val_trials']:
            raise ValueError('Vote count mismatch')
        if sorted(int(tr['trial_index']) for tr in trial_rows) != sorted(split['val_trials']):
            raise ValueError('Vote trial coverage mismatch')
        fold_correct = 0
        for tr in trial_rows:
            mask = ids == int(tr['trial_index'])
            if len(np.unique(truth_encoded[mask])) != 1:
                raise ValueError('Inconsistent true labels within a trial')
            votes = np.bincount(probabilities[mask].argmax(axis=1),minlength=len(cfg['labels']))
            pred = cfg['labels'][int(votes.argmax())]
            truth = cfg['labels'][int(predictions['true_encoded_labels'][mask][0])]
            if int(tr['pred_label']) != pred or int(tr['true_label']) != truth:
                raise ValueError('Majority vote mismatch')
            if int(tr['n_windows']) != int(votes.sum()) or json.loads(tr['votes']) != votes.tolist():
                raise ValueError('Saved vote/window count mismatch')
            fold_correct += pred == truth
        actual_trial = fold_correct/len(trial_rows)
        if row['n_correct_trials'] != fold_correct or not np.isclose(row['trial_accuracy'],actual_trial,atol=1e-9):
            raise ValueError('Stored trial accuracy does not match predictions')
        with (fd/'history.csv').open() as stream:
            history = list(csv.DictReader(stream))
        val_acc = np.array([float(h['val_accuracy']) for h in history])
        if (len(history) != row['epochs_ran'] or not np.isfinite(val_acc).all()
                or row['checkpoint_epoch'] != int(val_acc.argmax())+1
                or not np.isclose(val_acc.max(),actual_window,atol=1e-6)
                or not row['checkpoint_reloaded_and_verified']):
            raise ValueError('Checkpoint selection does not match logged best epoch')
        trial_accuracies.append(actual_trial)
        window_accuracies.append(actual_window)
        correct += fold_correct
        true_total += len(trial_rows)
        windows += len(ids)
        held_out.extend(split['val_trials'])
        held_windows.extend(split['val_windows'])
    if len(held_out) != len(set(held_out)) or len(held_out)!=summary['n_eligible_trials']:
        raise ValueError('An eligible trial was missing or evaluated more than once')
    if windows != summary['n_windows']:
        raise ValueError('OOF window count mismatch')
    if sorted(held_windows) != list(range(summary['n_windows'])):
        raise ValueError('OOF windows missing or evaluated more than once')
    for split in splits:
        if set(split['train_trials']+split['val_trials']) != set(held_out):
            raise ValueError('Fold does not partition all eligible trials')
        if sorted(split['train_windows']+split['val_windows']) != list(range(summary['n_windows'])):
            raise ValueError('Fold does not partition all windows')
    if not np.isclose(correct/true_total,summary['pooled_trial_accuracy'],atol=1e-9):
        raise ValueError('Pooled trial accuracy mismatch')
    for field,value in [('mean_trial_accuracy',np.mean(trial_accuracies)),
                        ('mean_window_accuracy',np.mean(window_accuracies)),
                        ('std_trial_accuracy',np.std(trial_accuracies))]:
        if not np.isclose(summary[field],value,atol=1e-9):
            raise ValueError(f'Stored {field} does not match fold predictions')
    return summary


def report(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(root)
    output = root/'figures'
    output.mkdir(exist_ok=True)
    summaries, errors = [], []
    paths = sorted(root.glob('S*_ME_4-40/result.json')) + sorted(root.glob('runs/*/result.json'))
    for path in paths:
        try:
            result = audit_condition(path.parent)
            summaries.append((path.parent,result))
        except Exception as exc:
            errors.append(dict(path=str(path),error=repr(exc)))
    records = []
    for directory, result in summaries:
        cfg = result['config']
        records.append(dict(subject=cfg['subject'],task=cfg['task'],labels='-'.join(map(str,cfg['labels'])),
            band='-'.join(f'{x:g}' for x in cfg['band']),trial_accuracy=result['mean_trial_accuracy'],
            pooled_trial_accuracy=result['pooled_trial_accuracy'],window_accuracy=result['mean_window_accuracy'],
            fold_sd=result['std_trial_accuracy'],n_trials=result['n_eligible_trials'],n_windows=result['n_windows'],
            directory=str(directory)))
    if records:
        with (root/'summary.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    reference_path=root/'paper_fig6_reference.json'
    if records and reference_path.exists():
        reference=json.loads(reference_path.read_text(encoding='utf-8'))
        lookup={(r['subject'],r['task'],'-'.join(map(str,r['labels'])),
                 '-'.join(f'{v:g}' for v in r['band'])):r for r in reference['records']}
        paired=[]
        for r in records:
            key=(r['subject'],r['task'],r['labels'],r['band'])
            if key not in lookup: continue
            paper=lookup[key]
            paired.append(dict(**r,paper_accuracy=paper['published_accuracy'],
                delta_trial_pp=100*(r['trial_accuracy']-paper['published_accuracy']),
                delta_window_pp=100*(r['window_accuracy']-paper['published_accuracy']),
                paper_sheet=paper['sheet'],paper_cell=paper['cell']))
        if paired:
            with (root/'paper_comparison.csv').open('w',newline='',encoding='utf-8') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(paired[0])); writer.writeheader(); writer.writerows(paired)
            fig,axes=plt.subplots(1,2,figsize=(12,5.7))
            fig.subplots_adjust(left=.07,right=.985,bottom=.19,top=.82,wspace=.23)
            for ax,task in zip(axes,['ME','MI']):
                labels=[v for v in ['1-4','1-2-4','1-2-3-4'] if any(r['task']==task and r['labels']==v and r['band']=='4-40' for r in paired)]
                x=np.arange(len(labels)); width=.24
                for offset,field,color,name in [(-1,'paper_accuracy','#777777','Paper published accuracy'),
                    (0,'trial_accuracy','#237aab','Reproduction trial vote'),(1,'window_accuracy','#e69f00','Reproduction window')]:
                    means=[100*np.mean([r[field] for r in paired if r['task']==task and r['labels']==v and r['band']=='4-40']) for v in labels]
                    bars=ax.bar(x+offset*width,means,width,color=color,label=name)
                    ax.bar_label(bars,fmt='%.2f',fontsize=8,padding=3)
                ax.set(xticks=x,xticklabels=[f'{v}\nn={sum(r["task"]==task and r["labels"]==v and r["band"]=="4-40" for r in paired)}' for v in labels],
                       title=f'{task}: matched subject IDs',ylabel='Accuracy (%)',ylim=(0,100))
                ax.grid(axis='y',alpha=.2)
                if labels: ax.legend(fontsize=7,loc='lower left')
                else:
                    ax.set_xticks([])
                    ax.text(.5,.5,'Pending: no completed CV conditions',ha='center',va='center',transform=ax.transAxes,color='#777777')
            fig.suptitle('Fig. 6 Source Data vs reproduction | same subjects and tasks | 4-40 Hz',fontsize=13,y=.97)
            fig.text(.5,.025,'CV validation; no independent test. Paper offline model selection/aggregation details not fully published.',ha='center',fontsize=8)
            fig.savefig(output/'paper_vs_reproduction_matched.png',dpi=180); plt.close(fig)
    save_json(root/'verification.json',dict(audited_conditions=len(summaries),errors=errors,
              checks=['checkpoint SHA256','raw stride 128 samples','no split overlap','OOF trial coverage',
                      'OOF window coverage','window probabilities','majority vote recomputation',
                      'checkpoint epoch vs history','summary means vs predictions'],independent_test_set=False))
    primary=[(d,r) for d,r in summaries if r['config']['subject'] in (1,2,5)
             and r['config']['task']=='ME' and r['config']['labels']==[1,4] and r['config']['band']==[4,40]]
    primary.sort(key=lambda x:x[1]['config']['subject'])
    if primary:
        fig,axes=plt.subplots(1,2,figsize=(10,4.3),constrained_layout=True)
        names=[f'S{r["config"]["subject"]:02d}' for d,r in primary]
        values=[[f['trial_accuracy']*100 for f in r['folds']] for d,r in primary]
        axes[0].boxplot(values,labels=names)
        for i,vals in enumerate(values,1): axes[0].scatter(np.full(len(vals),i),vals,s=22,c='#237aab',zorder=3)
        axes[0].set(title='Corrected pilot: 5-fold trial accuracy',ylabel='Accuracy (%)',ylim=(45,100))
        x=np.arange(len(primary)); width=.32
        axes[1].bar(x-width/2,[r['mean_window_accuracy']*100 for d,r in primary],width,label='Window (1 s)')
        axes[1].bar(x+width/2,[r['mean_trial_accuracy']*100 for d,r in primary],width,label='Trial majority vote')
        axes[1].set(xticks=x,xticklabels=names,ylabel='Accuracy (%)',ylim=(45,100),title='Window and trial metrics')
        axes[1].legend(fontsize=8)
        for ax in axes: ax.axhline(50,color='red',ls='--',lw=1); ax.grid(axis='y',alpha=.2)
        fig.suptitle('EEGNet-8,2 | ME thumb/pinky | 4-40 Hz | CV validation, no independent test')
        fig.savefig(output/'corrected_pilot_accuracy.png',dpi=180); plt.close(fig)
        fig,axes=plt.subplots(2,len(primary),figsize=(5*len(primary),6),squeeze=False,constrained_layout=True)
        for col,(directory,result) in enumerate(primary):
            for fold in result['folds']:
                with (directory/f'fold{fold["fold"]}'/'history.csv').open() as stream: hist=list(csv.DictReader(stream))
                epochs=np.arange(1,len(hist)+1)
                axes[0,col].plot(epochs,[float(h['val_accuracy'])*100 for h in hist],label=f'Fold {fold["fold"]}')
                axes[1,col].plot(epochs,[float(h['val_loss']) for h in hist])
            axes[0,col].set(title=f'S{result["config"]["subject"]:02d}: validation acc',ylabel='Window accuracy (%)')
            axes[0,col].legend(fontsize=7,ncol=3)
            axes[1,col].set(title='Validation loss',xlabel='Epoch',ylabel='Cross entropy')
            for ax in axes[:,col]: ax.grid(alpha=.2)
        fig.savefig(output/'corrected_pilot_learning_curves.png',dpi=180); plt.close(fig)
    if records:
        for metric,filename in [('trial_accuracy','fig6_style_trial_accuracy.png'),('window_accuracy','fig6_style_window_accuracy.png')]:
            fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
            for ax,task in zip(axes,['ME','MI']):
                conditions=sorted({(r['labels'],r['band']) for r in records if r['task']==task})
                vals=[[r[metric]*100 for r in records if r['task']==task and (r['labels'],r['band'])==c] for c in conditions]
                if vals:
                    ax.boxplot(vals,labels=[f'{c[0]}\n{c[1]} Hz\nn={len(v)}' for c,v in zip(conditions,vals)])
                    ax.tick_params(axis='x',labelrotation=45,labelsize=7)
                    for i,v in enumerate(vals,1):
                        ax.scatter(i+np.linspace(-.1,.1,len(v)),v,s=14,c='#237aab',alpha=.7,zorder=3)
                        chance=100/len(conditions[i-1][0].split('-'))
                        ax.hlines(chance,i-.35,i+.35,color='red',linestyle='--',lw=1,
                                  label='Chance level' if i==1 else None)
                    ax.legend(fontsize=7,loc='lower left')
                else:
                    ax.set_xticks([])
                    ax.text(.5,.5,'Pending: no completed CV conditions',ha='center',va='center',transform=ax.transAxes,color='#777777')
                ax.set(title=task,ylabel='Subject CV accuracy (%)',ylim=(0,100)); ax.grid(axis='y',alpha=.2)
            fig.suptitle(f'EEGNet Fig. 6-style | {metric} | incomplete conditions labelled with actual n')
            fig.savefig(output/filename,dpi=180); plt.close(fig)
    return records,errors


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('root'); args=parser.parse_args()
    rows,errors=report(args.root)
    print(json.dumps(dict(conditions=len(rows),errors=errors)))
    if errors: raise SystemExit(1)


if __name__=='__main__': main()
