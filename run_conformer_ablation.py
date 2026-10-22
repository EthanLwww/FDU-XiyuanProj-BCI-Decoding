"""Reproducible ME binary pretraining and S20 transfer experiment."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

VARIANTS = {"baseline": ("off", "off"), "filterbank": ("weighted", "off"),
            "filterbank_residual": ("weighted", "gated")}
DEFAULT_ROOT = "/home/kxy/LYC/experiments/conformer_depth1_win3_ME2_20261020"
DATA_ROOT = "/Dataset2/dataset_bci_decode_lyc"


def prepare(root):
    import numpy as np
    from Functions import generate_paths, load_and_filter_data
    from training_artifacts import write_json
    cache = root / "data"
    cache.mkdir(parents=True, exist_ok=True)
    for phase, subjects, modeltype in [("pretrain", [1, 2, 5], "Orig"),
                                       ("finetune", [20], "Finetune")]:
        marker = cache / (phase + ".json")
        if marker.exists():
            print("Reusing completed cache", marker, flush=True)
            continue
        params = dict(maxtriallen=5, windowlen=3, depth=1, block_size=128, downsrate=100,
                      bandpass_filt=[4, 40], nclass=2, model_arch="EEGConformer",
                      seed=42, batch_size=32, subj_ids=subjects, session_num=2, task="ME",
                      modeltype=modeltype, data_folder=DATA_ROOT)
        paths = [p for subject in subjects for p in
                 generate_paths(subject, "ME", 2, 2, modeltype, DATA_ROOT)]
        expected = sorted(str(p.resolve()) for directory in paths
                          for p in Path(directory).glob("*.mat"))
        params["data_paths"] = paths
        data, labels, params = load_and_filter_data(paths, params)
        loaded = sorted(f["path"] for f in params["data_files"])
        if loaded != expected:
            raise RuntimeError("MAT file manifest mismatch: " + str(set(expected) - set(loaded)))
        if len(set(loaded)) != len(loaded):
            raise RuntimeError("Duplicate input files")
        params["class_counts"] = {str(c): int((labels == c).sum()) for c in [1, 2]}
        params["data_shape"] = list(data.shape)
        params["selected_file_count"] = len(expected)
        if data.shape[1] != 128 or any(v == 0 for v in params["class_counts"].values()):
            raise RuntimeError("Unexpected channel count or missing class")
        np.save(cache / (phase + "_data.npy"), data)
        np.save(cache / (phase + "_labels.npy"), labels)
        write_json(marker, params)
        print("Prepared", phase, data.shape, params["class_counts"], flush=True)
        del data, labels


def run(root, variant, seed=None, preprocessed_root=None):
    import numpy as np
    import tensorflow as tf
    for device in tf.config.list_physical_devices("GPU"):
        tf.config.set_logical_device_configuration(
            device, [tf.config.LogicalDeviceConfiguration(memory_limit=32768)])
    from Functions import train_models
    from training_artifacts import write_json
    out = root / variant
    out.mkdir(parents=True, exist_ok=True)
    status = out / "status.json"
    start = time.time()
    try:
        for phase in ("pretrain", "finetune"):
            checkpoint = out / (phase + ".h5")
            config = checkpoint.with_suffix(".config.json")
            if config.exists() and json.loads(config.read_text()).get("status") == "complete":
                if seed is not None and json.loads(config.read_text()).get("seed") != seed:
                    raise ValueError("Completed checkpoint uses a different seed.")
                print("Reusing completed phase", phase, flush=True)
                continue
            tf.keras.backend.clear_session()
            params = json.loads((root / "data" / (phase + ".json")).read_text())
            if seed is not None:
                params["seed"] = seed
            if preprocessed_root is not None:
                params["preprocessed_cache"] = str(Path(preprocessed_root) / phase)
            params.update(filter_bank=VARIANTS[variant][0],
                          broadband_residual=VARIANTS[variant][1],
                          gpu_memory_limit_mib=32768, input_storage="CPU",
                          epochs=300 if phase == "pretrain" else 100)
            if phase == "finetune":
                params["modelpath"] = str(out / "pretrain.h5")
                params["finetune_policy"] = "conformer_baseline"
            write_json(status, dict(status="running", phase=phase, pid=os.getpid(),
                                    started_unix=start, variant=variant))
            data = np.load(root / "data" / (phase + "_data.npy"), mmap_mode="r")
            labels = np.load(root / "data" / (phase + "_labels.npy"))
            train_models(data, labels, str(checkpoint), params)
            del data, labels
        write_json(status, dict(status="complete", variant=variant,
                                elapsed_seconds=time.time() - start))
    except BaseException:
        write_json(status, dict(status="failed", variant=variant,
                                traceback=traceback.format_exc()))
        raise


def plot(root):
    import csv
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from training_artifacts import write_json, checkpoint_hash
    labels = {"baseline": "Broadband baseline", "filterbank": "Filter bank",
              "filterbank_residual": "Filter bank + broadband residual"}
    colors = dict(zip(VARIANTS, ["#377eb8", "#ff7f00", "#4daf4a"]))
    rows, histories = [], {}
    for phase in ("pretrain", "finetune"):
        reference = None
        for variant in VARIANTS:
            path = root / variant / phase
            config = json.loads(path.with_suffix(".config.json").read_text())
            assert config["status"] == "complete", (phase, variant)
            assert config["depth"] == 1 and config["windowlen"] == 3
            assert config["checkpoint_sha256"] == checkpoint_hash(path.with_suffix(".h5"))
            split = dict(np.load(path.with_suffix(".split.npz")))
            if reference is None:
                reference = split
            assert all(np.array_equal(reference[k], split[k]) for k in reference), "Unequal splits"
            assert not np.intersect1d(split["train_trial_indices"], split["val_trial_indices"]).size
            history = json.loads(path.with_suffix(".history.json").read_text())
            histories[(phase, variant)] = history
            best = int(np.argmax(history["val_accuracy"]))
            row = dict(phase=phase, variant=variant, epochs=len(history["loss"]),
                       best_epoch=best + 1, best_val_accuracy=history["val_accuracy"][best],
                       val_loss_at_best_accuracy=history["val_loss"][best],
                       min_val_loss=min(history["val_loss"]), parameters=config["parameter_count"],
                       train_windows=config["n_train_windows"],
                       val_windows=config["n_validation_windows"])
            rows.append(row)
    for metric, ylabel in [("loss", "Cross-entropy loss"), ("accuracy", "Accuracy")]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        for ax, phase in zip(axes, ("pretrain", "finetune")):
            for variant in VARIANTS:
                history = histories[(phase, variant)]
                x = np.arange(1, len(history[metric]) + 1)
                ax.plot(x, history[metric], color=colors[variant], alpha=.45, linestyle="--")
                ax.plot(x, history["val_" + metric], color=colors[variant],
                        label=labels[variant], linewidth=1.8)
            ax.set(title="S01/S02/S05 pretraining" if phase == "pretrain" else "S20 Session 2 fine-tuning",
                   xlabel="Epoch", ylabel=ylabel)
            ax.grid(alpha=.2)
            if metric == "accuracy":
                ax.set_ylim(0, 1)
            ax.legend(loc="best", fontsize=8)
        fig.suptitle(f"EEGConformer depth=1 | 3 s | ME thumb/pinky | seed={config['seed']}\nSolid: validation; dashed: training")
        fig.savefig(root / (metric + "_comparison.png"), dpi=180)
        fig.savefig(root / (metric + "_comparison.pdf"))
        plt.close(fig)
    write_json(root / "summary.json", rows)
    with (root / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# EEGConformer three-way ablation", "",
             "ME thumb/pinky; depth 1; 3 s; seed 42; batch 32.",
             "Pretraining: S01/S02/S05 OfflineMovement and online sessions 1–2.",
             "Fine-tuning: S20 Session 2 2class_Base + 3class_Base, labels 1 and 4.",
             "Identical 80/20 trial splits across variants; overlapping windows stay in their trial partition.",
             "Best-validation-accuracy checkpoints; early stopping on validation loss (patience 80).",
             "Fine-tuning trains the same four Dense layers as the original Conformer last-12 policy (FF output + classifier), plus fusion gates; all other weights frozen.",
             "These are validation results from one seed, with no independent test set.",
             "Global band/gate coefficients are not finger-specific or causal importance.", "",
             "| Phase | Variant | Epochs | Best epoch | Best validation accuracy |",
             "|---|---|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['phase']} | {r['variant']} | {r['epochs']} | {r['best_epoch']} | {r['best_val_accuracy']:.2%} |")
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run", "plot"])
    parser.add_argument("--root", type=Path, default=Path(DEFAULT_ROOT))
    parser.add_argument("--variant", choices=list(VARIANTS))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--preprocessed-root", type=Path)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    if args.command == "run":
        if args.variant is None:
            parser.error("run requires --variant")
        run(args.root, args.variant, args.seed, args.preprocessed_root)
    elif args.command == "prepare":
        prepare(args.root)
    else:
        plot(args.root)
