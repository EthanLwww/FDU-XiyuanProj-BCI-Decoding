"""Audit both finished model suites and generate comparable scientific figures."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scaling_experiments.streaming_data import save_json
from bci.training_artifacts import checkpoint_hash

GROUPS={'s3_reference':[1,2,5], 's5':list(range(1,6)),
        's10':list(range(1,11)), 's20':list(range(1,21))}
NAMES=['S01/S02/S05','S01-S05','S01-S10','S01-S20']
ROOT=Path('/home/kxy/LYC/experiments/conformer_scaling_S21_filterbank_residual_20261021')
BASELINE=Path('/home/kxy/LYC/experiments/conformer_scaling_S21_20261021')
COLORS={'baseline':'#2474A6','residual':'#D65F2B'}

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def manifest(root,sid):
    p=root/'cache'/f'S{sid:02d}'/'ready.json'
    return read(p if p.exists() else root/'data_manifests'/f'S{sid:02d}'/'ready.json')

def compare(root,baseline):
    rows=[]; histories={}; splits={}; hashes={}
    for model,folder in [('baseline',baseline),('residual',root)]:
        for group in GROUPS:
            for phase in ['pretrain','finetune']:
                p=folder/group/phase
                c=read(p.with_suffix('.config.json')); h=read(p.with_suffix('.history.json'))
                v=read(p.with_suffix('.verification.json'))
                digest=checkpoint_hash(p.with_suffix('.h5'))
                assert c['status']=='complete' and v['status']=='verified'
                assert digest==c['checkpoint_sha256']==v['checkpoint_sha256']
                assert c['seed']==42 and c['depth']==1 and c['windowlen']==3
                assert c['filter_bank']==('off' if model=='baseline' else 'weighted')
                assert c.get('broadband_residual','off')==('off' if model=='baseline' else 'gated')
                best=int(np.argmax(h['val_accuracy']))
                assert abs(v['recomputed_validation']['accuracy']-h['val_accuracy'][best])<1e-6
                assert abs(v['recomputed_validation']['loss']-h['val_loss'][best])<1e-4
                with np.load(p.with_suffix('.split.npz')) as z: split={k:z[k].copy() for k in z.files}
                assert not np.intersect1d(split['train_window_trial_indices'],split['val_window_trial_indices']).size
                hashes[f'{model}/{group}/{phase}.h5']=digest
                splits[(model,group,phase)]=split
                histories[(model,group,phase)]=h
                rows.append(dict(model=model,group=group,phase=phase,subjects=GROUPS[group],
                    train_windows=c['n_train_windows'],validation_windows=c['n_validation_windows'],
                    trials=c['n_trials'],epochs=len(h['loss']),best_epoch=best+1,
                    accuracy=h['val_accuracy'][best],loss=h['val_loss'][best],
                    min_loss=min(h['val_loss']),parameter_count=c['parameter_count'],
                    trainable_parameter_count=c['trainable_parameter_count'],
                    peak_rss_mib=c.get('peak_rss_mib'),elapsed_seconds=c.get('elapsed_seconds'),
                    checkpoint_sha256=digest,band_weights=v.get('band_weights'),
                    residual_info=v.get('broadband_residual')))
    for group in GROUPS:
        for phase in ['pretrain','finetune']:
            a=splits[('baseline',group,phase)];b=splits[('residual',group,phase)]
            for key in a:
                np.testing.assert_array_equal(a[key],b[key])
        for model in ['baseline','residual']:
            for key,value in splits[('baseline','s5','finetune')].items():
                np.testing.assert_array_equal(value,splits[(model,group,'finetune')][key])
    counts=[]; files_checked=0
    for sid in range(1,22):
        a=manifest(baseline,sid);b=manifest(root,sid)
        assert a['shape'][:3]==b['shape'][:3] and b['shape'][3]==5
        assert a['trial_labels']==b['trial_labels'] and len(a['files'])==len(b['files'])
        for af,bf in zip(a['files'],b['files']):
            assert Path(af['path']).parts[-3:]==Path(bf['path']).parts[-3:]
            assert af['sha256']==bf['sha256'] and af['size']==bf['size']
            files_checked+=1
        counts.append(dict(subject=sid,files=len(b['files']),trials=b['n_trials'],
            windows=b['shape'][0],cache_bytes=int(np.prod(b['shape']))*4))
    by={(r['model'],r['group'],r['phase']):r for r in rows}
    pairs=[]
    for group in GROUPS:
        for phase in ['pretrain','finetune']:
            a=by[('baseline',group,phase)];b=by[('residual',group,phase)]
            assert a['train_windows']==b['train_windows'] and a['validation_windows']==b['validation_windows']
            pairs.append(dict(group=group,phase=phase,baseline_accuracy=a['accuracy'],
                residual_accuracy=b['accuracy'],delta_percentage_points=100*(b['accuracy']-a['accuracy']),
                baseline_loss=a['loss'],residual_loss=b['loss'],loss_delta=b['loss']-a['loss'],
                train_windows=b['train_windows'],validation_windows=b['validation_windows']))
    result=dict(status='verified',seed=42,rows=rows,pairs=pairs,subject_counts=counts,
                source_files_checked=files_checked,splits_identical_across_models=True,
                s21_split_identical_across_all_eight_models=True,checkpoint_sha256=hashes,
                residual_pretrain_cache_gib=sum(c['cache_bytes'] for c in counts if c['subject']!=21)/2**30,
                resource_sampling=read(root/'resource_peaks.json') if (root/'resource_peaks.json').exists() else None)
    save_json(root/'comparison.json',result)
    save_json(root/'checkpoint_sha256.json',hashes)
    with (root/'comparison.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(pairs[0]));writer.writeheader();writer.writerows(pairs)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    for phase in ['pretrain','finetune']:
        for metric in ['accuracy','loss']:
            fig,axes=plt.subplots(2,2,figsize=(14,9),constrained_layout=True)
            for ax,group,name in zip(axes.ravel(),GROUPS,NAMES):
                for model,label in [('baseline','Ordinary EEGConformer'),('residual','Filter-bank + residual')]:
                    h=histories[(model,group,phase)];scale=100 if metric=='accuracy' else 1
                    epoch=np.arange(1,len(h['loss'])+1)
                    ax.plot(epoch,np.array(h[metric])*scale,'--',alpha=.55,color=COLORS[model],label=label+' train')
                    ax.plot(epoch,np.array(h['val_'+metric])*scale,color=COLORS[model],label=label+' validation')
                ax.set(title=name,xlabel='Epoch',ylabel='Accuracy (%)' if metric=='accuracy' else 'Cross-entropy loss')
                ax.grid(alpha=.2);ax.legend(fontsize=8,loc='best')
            fig.suptitle(('Pretraining' if phase=='pretrain' else 'S21 fine-tuning')+' '+metric+' comparison | seed 42')
            for ext in ['png','pdf']:fig.savefig(root/f'{phase}_{metric}_models.{ext}',dpi=200)
            plt.close(fig)
    fine=[p for p in pairs if p['phase']=='finetune']
    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    x=np.arange(4)
    for offset,model,label in [(-.18,'baseline','Ordinary EEGConformer'),(.18,'residual','Filter-bank + residual')]:
        bars=axes[0].bar(x+offset,[p[model+'_accuracy']*100 for p in fine],.36,label=label,color=COLORS[model])
        axes[0].bar_label(bars,fmt='%.2f',fontsize=9)
    axes[0].set(ylim=(0,100),ylabel='Best S21 validation accuracy (%)');axes[0].legend(fontsize=9)
    bars=axes[1].bar(x,[p['delta_percentage_points'] for p in fine],color=COLORS['residual'])
    axes[1].bar_label(bars,fmt='%+.2f',padding=3);axes[1].axhline(0,color='#333',lw=.8)
    axes[1].set(ylabel='Residual minus ordinary (percentage points)');axes[1].margins(y=.25)
    for ax in axes:ax.set_xticks(x,NAMES);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Identical S21 validation split | one run per model and cohort')
    for ext in ['png','pdf']:fig.savefig(root/f's21_model_comparison.{ext}',dpi=200)
    plt.close(fig)
    lines=['# 频带残差与普通 EEGConformer 数据规模比较','',
        'ME 拇指/小指二分类，depth=1，3 秒窗口，seed=42，每组一次。普通模型复用上次结果，频带残差四组全部重新预训练并在 S21 Session 2 Base 微调。',
        '', '| 预训练组 | 普通 S21 acc | 频带残差 S21 acc | 差值（百分点） | 普通 loss | 频带残差 loss |',
        '|---|---:|---:|---:|---:|---:|']
    for name,p in zip(NAMES,fine):
        lines.append(f'| {name} | {p["baseline_accuracy"]:.2%} | {p["residual_accuracy"]:.2%} | {p["delta_percentage_points"]:+.2f} | {p["baseline_loss"]:.4f} | {p["residual_loss"]:.4f} |')
    lines += ['', '## 训练与评估', '',
        '预训练/微调：最大 300/100 epoch，Adam 0.0005/0.00005，dropout 0.5/0.65，batch=32。val_loss 早停 patience=80，降学习率 patience=30、factor=0.5，label smoothing=0.1。',
        '两种模型在同一被试组中直接共用上一轮保存的 trial 划分，分窗后再次核对索引。S21 为 256/64 个训练/验证 trial，对应 262/68 个有效窗口。选择首次最高验证 acc checkpoint，报告同一 checkpoint 的 loss。',
        '普通模型参数 379,306，频带残差 412,751（增加约 8.82%）。微调共同的最后四个 Dense 层，频带残差还训练 4 个频带 logits 与 1 个门控参数，其余前端冻结。',
        '', '## 前端与数据加载', '',
        '数据位于 `/Dataset2/dataset_bci_decode_kxy/data`。输入 128×300×5：四个独立标准化子带 [4,8]/[8,13]/[13,30]/[30,40] Hz，加上 [4,40] Hz 宽带。子带使用独立时间卷积与 Depthwise 空间卷积，softmax 权重融合后加上 sigmoid 门控宽带分支。',
        '逐 MAT 文件预处理，按被试保存 float32 缓存。内存映射每次读取 32 个窗口，预取 1 个 batch，保留尾 batch。TensorFlow 分配上限为每进程 16 GiB，CUDA 上下文另有开销。',
        f'20 人缓存 {result["residual_pretrain_cache_gib"]:.2f} GiB。核对了 {files_checked} 个源文件 SHA256 与窗口索引，数据内容与上一轮一致。S09 的孤立 TrialEnd 按已有规则忽略并记录。',
        '', '## 解释边界', '',
        '这是单次运行的最佳验证结果，没有独立测试集。用验证集选择 checkpoint 会带来选择偏差，不能从本轮推断统计显著性、跨随机种子稳定性或独立测试性能。',
        '比较的是完整前端插件配置。除了频带划分，还改变了空间卷积、特征归一化、参数量及微调可训练参数，因此不能把差值单独归因为频带归纳偏置。',
        '频带 softmax 和残差系数是模型级融合参数。标准化去除了绝对频带功率，它们不能直接解释为手指重要性或神经生理贡献。',
        '扩大被试范围同时改变数据量、被试构成与每轮优化步数。普通三人参考组复用了旧 TensorFlow shuffle 历史，其余新训练使用 NumPy 全排列，因此三人组随机训练顺序也存在差异。',
        '', '![S21 comparison](s21_model_comparison.png)',
        '', '![Pretraining accuracy](pretrain_accuracy_models.png)',
        '', '![Pretraining loss](pretrain_loss_models.png)',
        '', '![S21 accuracy](finetune_accuracy_models.png)',
        '', '![S21 loss](finetune_loss_models.png)', '']
    (root/'REPORT_COMPARISON.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(status='verified',source_files_checked=files_checked,checkpoints=len(hashes),s21=fine),indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--baseline',type=Path,default=BASELINE)
    args=parser.parse_args();compare(args.root,args.baseline)
