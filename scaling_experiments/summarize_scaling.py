"""Validate, plot, and archive the completed four-cohort S21 comparison."""
import argparse
import csv
import json
from pathlib import Path
import tarfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_scaling import GROUPS, ROOT
from streaming_data import save_json
from training_artifacts import checkpoint_hash


def summarize(root):
    rows, histories, reference = [], {}, None
    for group, subjects in GROUPS.items():
        for phase in ['pretrain','finetune']:
            path=root/group/phase
            c=json.loads(path.with_suffix('.config.json').read_text())
            h=json.loads(path.with_suffix('.history.json').read_text())
            assert c['status']=='complete' and c['seed']==42 and c['depth']==1
            assert checkpoint_hash(path.with_suffix('.h5'))==c['checkpoint_sha256']
            v=json.loads(path.with_suffix('.verification.json').read_text())
            assert v['status']=='verified' and v['checkpoint_sha256']==c['checkpoint_sha256']
            splits=dict(np.load(path.with_suffix('.split.npz')))
            assert not np.intersect1d(splits['train_window_trial_indices'],splits['val_window_trial_indices']).size
            if phase=='finetune':
                if reference is None:
                    reference=splits
                assert all(np.array_equal(reference[k],splits[k]) for k in reference)
                assert c['subjects']==[21]
            best=int(np.argmax(h['val_accuracy']))
            row=dict(group=group,pretraining_subject_count=len(subjects),phase=phase,seed=42,
                trials=c['n_trials'],train_windows=c['n_train_windows'],validation_windows=c['n_validation_windows'],
                epochs=len(h['loss']),best_epoch=best+1,best_val_accuracy=h['val_accuracy'][best],
                val_loss_at_best_accuracy=h['val_loss'][best],min_val_loss=min(h['val_loss']),
                peak_rss_mib=c.get('peak_rss_mib'),reused=(group=='s3_reference' and phase=='pretrain'),
                checkpoint_sha256=c['checkpoint_sha256'])
            rows.append(row)
            histories[(group,phase)]=h
    save_json(root/'summary.json',rows)
    with (root/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    colors=['#377eb8','#ff7f00','#4daf4a','#984ea3']
    names=['S01/S02/S05 (reused pretraining)','S01–S05','S01–S10','S01–S20']
    for metric, ylabel in [('accuracy','Accuracy (%)'),('loss','Cross-entropy loss')]:
        fig,axes=plt.subplots(1,2,figsize=(14,5.5),constrained_layout=True)
        for ax,phase in zip(axes,['pretrain','finetune']):
            for group,name,color in zip(GROUPS,names,colors):
                h=histories[(group,phase)];scale=100 if metric=='accuracy' else 1
                ax.plot(np.arange(1,len(h[metric])+1),np.array(h[metric])*scale,'--',color=color,alpha=.5)
                ax.plot(np.arange(1,len(h[metric])+1),np.array(h['val_'+metric])*scale,color=color,label=name)
            ax.set(xlabel='Epoch',ylabel=ylabel,title='Pretraining (cohort-specific validation)' if phase=='pretrain' else 'S21 Session 2 fine-tuning (identical split)')
            ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.suptitle('Baseline EEGConformer depth=1 | 3 s | ME thumb/pinky | seed 42 only\nSolid: validation; dashed: training')
        for ext in ['png','pdf']:
            fig.savefig(root/f'{metric}_comparison.{ext}',dpi=200)
        plt.close(fig)
    fine=[r for r in rows if r['phase']=='finetune']
    pre=[r for r in rows if r['phase']=='pretrain']
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    for ax,x,xlabel in [(axes[0],[len(s) for s in GROUPS.values()],'Pretraining subjects'),
                        (axes[1],[r['train_windows'] for r in pre],'Pretraining training windows')]:
        y=[r['best_val_accuracy']*100 for r in fine]
        ax.plot(x,y,color='#526776',alpha=.5)
        ax.scatter(x,y,c=colors,s=65)
        for px,py in zip(x,y):
            ax.annotate(f'{py:.2f}%',(px,py),xytext=(0,9),textcoords='offset points',ha='center')
        ax.set(xlabel=xlabel,ylabel='S21 best validation accuracy (%)');ax.grid(alpha=.2);ax.margins(y=.2)
    fig.suptitle('Pretraining data volume versus S21 validation accuracy\nOne run per cohort; no error bars or repeat-distribution boxplots')
    for ext in ['png','pdf']:
        fig.savefig(root/f'data_volume_comparison.{ext}',dpi=200)
    plt.close(fig)
    counts=[]
    for sid in range(1,22):
        meta=json.loads((root/'cache'/f'S{sid:02d}'/'ready.json').read_text())
        counts.append(dict(subject=sid,files=len(meta['files']),trials=meta['n_trials'],windows=meta['shape'][0],class_counts=meta['class_counts']))
    save_json(root/'subject_data_counts.json',counts)
    lines=['# EEGConformer 预训练数据量与 S21 迁移实验','',
        '四组各运行一次（seed=42），仅使用宽带 EEGConformer baseline。新增 S01–S05、S01–S10、S01–S20 三组预训练。',
        '参考组复用原 S01/S02/S05、seed=42 的预训练权重，四组均重新在 S21 Session 2 Base 数据上微调。原 S20 微调结果不参与本次比较。','',
        '## 结果','', '| 预训练被试 | 预训练 trial | 预训练训练窗口 | 预训练最佳验证 acc | S21 最佳验证 acc | S21 对应 loss |',
        '|---|---:|---:|---:|---:|---:|']
    for name,a,b in zip(names,pre,fine):
        lines.append(f'| {name} | {a["trials"]} | {a["train_windows"]} | {a["best_val_accuracy"]:.2%} | {b["best_val_accuracy"]:.2%} | {b["val_loss_at_best_accuracy"]:.4f} |')
    lines += ['', '## 实验设置','',
        'ME 拇指/小指二分类，128 通道，depth=1，3 秒窗口，原始 1024 Hz 降采样为 100 Hz。预训练使用离线 Movement 与在线 Session 1/2 的 Base、Finetune 数据。S21 仅用于目标微调。',
        '按 trial 做 80/20 划分后再分窗。四组 S21 trial 与窗口划分完全相同。预训练各组对自己的合并 trial 列表按 seed=42 划分，因此各组预训练验证集不同。',
        'batch=32，预训练/微调 dropout=.5/.65，Adam 学习率 .0005/.00005，最大 300/100 轮，val_loss 早停 patience=80，降学习率 patience=30、factor=.5，label smoothing=.1。微调相同四个 Dense 层。',
        'checkpoint 取首次达到最高验证 accuracy 的模型，loss 为同一个 checkpoint 对应的验证 loss。所有新增 checkpoint 均重新加载并在完整验证集复算。','',
        '## 数据与内存','',
        '数据目录：`/Dataset2/dataset_bci_decode_kxy`。逐 MAT 文件读取并做 CAR、分窗、重采样、滤波、标准化，写入每被试 float32 窗口缓存。训练仅按批读取，预取一个 batch，不将全量数据复制到 CPU/GPU Tensor。TensorFlow 显存分配上限 16 GiB，驱动上下文另有开销。',
        '新预处理与旧流程已在真实文件和合成数据上逐元素对照。原 float64 滤波结果转 float32 缓存，与旧 TensorFlow 输入到模型时的 float32 精度一致。',
        'S09 有额外的在线运行，S13 有 22 个离线 ME 文件，S19 Session 2 二分类 Finetune 有 4 个文件。使用所有已校验文件，逐文件 SHA256、实际 trial 和窗口数已保存。',
        '全部 1,924 个文件的事件配对已检查。S09 Session 1 3class_Finetune R07 有一个无对应 Target 的多余 TrialEnd（事件索引 26）。加载器按事件顺序配对，忽略并记录该标记，保留全部 20 个拇指/小指 trial。其他文件没有此异常。详情见 event_pairing_audit.json。',
        '复用的三人参考模型也通过新缓存完整复算了原 4,211 个预训练验证窗口，准确率与旧记录一致，loss 在 1e-4 容差内。',
        '新增组使用每轮完整的 NumPy 随机排列；复用参考组保留原 TensorFlow shuffle 的历史。训练随机顺序并非逐步一致。','',
        '## 解读边界','',
        '这是单随机种子的验证结果，没有独立测试集。不能据此估计重复实验方差或统计显著性，因此本轮不使用箱形图。',
        '扩大被试范围同时改变数据量和被试构成，且相同 epoch 下优化步数增加。本结果反映这一整体变化，不是严格固定构成或计算预算的数据量因果实验。','',
        '![Accuracy](accuracy_comparison.png)','', '![Loss](loss_comparison.png)','', '![Volume](data_volume_comparison.png)','']
    (root/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    save_json(root/'final_verification.json',dict(status='verified',groups=4,pretraining_new=3,finetuning_new=4,
        seed=42,s21_splits_identical=True,metrics=rows))
    # Include source manifests and model artifacts, excluding large regenerable signal caches.
    with tarfile.open(root/'results_bundle.tar.gz','w:gz') as tar:
        for p in sorted(root.rglob('*')):
            rel=p.relative_to(root)
            if p.is_file() and rel.parts[0] not in ('cache','runtime') and p.name!='results_bundle.tar.gz' and '__pycache__' not in rel.parts:
                tar.add(p,arcname=str(rel))
        for p in sorted((root/'cache').glob('S*/ready.json')):
            tar.add(p,arcname=str(Path('data_manifests')/p.parent.name/'ready.json'))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT)
    summarize(p.parse_args().root)
