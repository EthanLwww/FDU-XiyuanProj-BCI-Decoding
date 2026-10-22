"""Paired-seed EEGConformer ablation and boxplot reports."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

VARIANTS = ["baseline", "filterbank", "filterbank_residual"]
SEEDS = [42, 43, 44, 45, 46]
DEFAULT_ROOT = Path("/home/kxy/LYC/experiments/conformer_depth1_win3_ME2_repeats5_20261020")
ORIGINAL_ROOT = Path("/home/kxy/LYC/experiments/conformer_depth1_win3_ME2_20261020")


def write_status(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def summarize_histories(items):
    import numpy as np
    rows = []
    for item in items:
        h = item["history"]
        best = int(np.argmax(h["val_accuracy"]))
        rows.append(dict(seed=item["seed"], variant=item["variant"], phase=item["phase"],
                         epochs=len(h["loss"]), best_epoch=best + 1,
                         best_val_accuracy=h["val_accuracy"][best],
                         val_loss_at_best_accuracy=h["val_loss"][best],
                         min_val_loss=min(h["val_loss"])))
    return rows


def verify_variant(seed_root, variant, cache_root):
    import numpy as np
    import tensorflow as tf
    from bci.experiment_window_cache import load_cached_split
    from bci.training_artifacts import checkpoint_hash
    from bci import FilterBankModels_tf
    results = []
    for phase in ["pretrain", "finetune"]:
        path = seed_root / variant / (phase + ".h5")
        c = json.loads(path.with_suffix(".config.json").read_text())
        assert c["status"] == "complete" and c["depth"] == 1 and c["windowlen"] == 3
        assert checkpoint_hash(path) == c["checkpoint_sha256"]
        split = np.load(path.with_suffix(".split.npz"))
        assert not np.intersect1d(split["train_window_trial_indices"], split["val_window_trial_indices"]).size
        data = np.load(seed_root / "data" / (phase + "_data.npy"), mmap_mode="r")
        labels = np.load(seed_root / "data" / (phase + "_labels.npy"))
        x, y, _ = load_cached_split(cache_root / phase, data, labels, c,
                                    split["val_trial_indices"], c["filter_bank"],
                                    c["broadband_residual"])
        tf.keras.backend.clear_session()
        model = tf.keras.models.load_model(path, compile=False)
        assert sum(isinstance(layer, tf.keras.layers.MultiHeadAttention) for layer in model.layers) == 1
        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=.1),
                      metrics=["accuracy"])
        with tf.device("/CPU:0"):
            x = tf.convert_to_tensor(x)
        metrics = model.evaluate(x, tf.keras.utils.to_categorical(y - 1, 2),
                                 batch_size=32, verbose=0, return_dict=True)
        history = json.loads(path.with_suffix(".history.json").read_text())
        best = int(np.argmax(history["val_accuracy"]))
        assert abs(metrics["accuracy"] - history["val_accuracy"][best]) < 1e-6
        assert abs(metrics["loss"] - history["val_loss"][best]) < 1e-4
        results.append(dict(phase=phase, seed=c["seed"], variant=variant,
                            checkpoint_sha256=c["checkpoint_sha256"],
                            recomputed_validation=metrics))
        del x, data, model
    write_status(seed_root / variant / "verification.json", results)


def worker(root, seed, variant):
    from experiment_runners.ablation import run
    seed_root = root / f"seed_{seed}"
    try:
        run(seed_root, variant, seed=seed, preprocessed_root=root / "preprocessed")
        verify_variant(seed_root, variant, root / "preprocessed")
    except BaseException:
        write_status(seed_root / variant / "status.json",
                     dict(status="failed", traceback=traceback.format_exc()))
        raise


def collect(root):
    import numpy as np
    from bci.training_artifacts import checkpoint_hash
    items = []
    for seed in SEEDS:
        for phase in ["pretrain", "finetune"]:
            reference = None
            for variant in VARIANTS:
                path = root / f"seed_{seed}" / variant / phase
                config = json.loads(path.with_suffix(".config.json").read_text())
                assert config["status"] == "complete" and config["seed"] == seed
                assert config["depth"] == 1 and config["windowlen"] == 3
                assert checkpoint_hash(path.with_suffix(".h5")) == config["checkpoint_sha256"]
                split = dict(np.load(path.with_suffix(".split.npz")))
                if reference is None:
                    reference = split
                else:
                    assert all(np.array_equal(reference[k], split[k]) for k in reference), (seed, phase)
                assert not np.intersect1d(split["train_window_trial_indices"], split["val_window_trial_indices"]).size
                history = json.loads(path.with_suffix(".history.json").read_text())
                items.append(dict(seed=seed, variant=variant, phase=phase, history=history))
    rows = summarize_histories(items)
    assert len(rows) == 30
    return rows


def plot(root):
    import csv
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = collect(root)
    names = ["Broadband\nbaseline", "Filter bank", "Filter bank\n+ broadband residual"]
    colors = ["#377eb8", "#ff7f00", "#4daf4a"]
    stats = []
    for metric, filename, ylabel, scale in [
            ("best_val_accuracy", "accuracy_boxplot", "Best validation accuracy (%)", 100),
            ("val_loss_at_best_accuracy", "loss_boxplot", "Validation loss at best-accuracy checkpoint", 1)]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
        for ax, phase in zip(axes, ["pretrain", "finetune"]):
            data = [[next(row[metric] for row in rows if row["seed"] == seed
                          and row["phase"] == phase and row["variant"] == variant) * scale
                     for seed in SEEDS] for variant in VARIANTS]
            boxes = ax.boxplot(data, positions=[1, 2, 3], widths=.5, patch_artist=True,
                               showfliers=False, medianprops=dict(color="black", linewidth=1.6))
            for box, color in zip(boxes["boxes"], colors):
                box.set_facecolor(color)
                box.set_alpha(.35)
            offsets = np.linspace(-.10, .10, len(SEEDS))
            for index, seed in enumerate(SEEDS):
                x = np.arange(1, 4) + offsets[index]
                y = [values[index] for values in data]
                ax.plot(x, y, color="#777777", alpha=.25, linewidth=.7, zorder=2)
                ax.scatter(x, y, color=colors, edgecolors="white", linewidth=.5, s=36, zorder=3)
                for px, py in zip(x, y):
                    ax.annotate(str(seed), (px, py), xytext=(3, 3), textcoords="offset points",
                                fontsize=6, color="#444444")
            ax.set_xticks([1, 2, 3], names)
            ax.set(ylabel=ylabel, title="S01/S02/S05 pretraining" if phase == "pretrain"
                   else "S20 Session 2 fine-tuning")
            ax.grid(axis="y", alpha=.2)
            ax.margins(y=.2)
            for variant, values in zip(VARIANTS, data):
                stats.append(dict(phase=phase, variant=variant, metric=metric,
                                  unit="percent" if scale == 100 else "cross_entropy",
                                  n=len(values), mean=float(np.mean(values)),
                                  std=float(np.std(values, ddof=1)), median=float(np.median(values)),
                                  q1=float(np.percentile(values, 25)), q3=float(np.percentile(values, 75)),
                                  minimum=float(min(values)), maximum=float(max(values))))
        fig.suptitle("EEGConformer depth=1 | 3 s | ME thumb/pinky | 5 paired seeds\n"
                     "Dots: complete runs (seed labels); boxes: IQR; whiskers: 1.5 IQR")
        for suffix in ["png", "pdf"]:
            fig.savefig(root / f"{filename}.{suffix}", dpi=200)
        plt.close(fig)
    for filename, values in [("run_metrics", rows), ("descriptive_statistics", stats)]:
        write_status(root / f"{filename}.json", values)
        with (root / f"{filename}.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    lines = ["# Five-seed EEGConformer ablation", "",
             "Seeds: 42, 43, 44, 45, 46. Seed 42 reuses the verified initial experiment.",
             "Each point is one complete pretraining/fine-tuning run; epochs are not repeated samples.",
             "Seeds change both trial partitions and initialization. The three variants share identical partitions within each seed.",
             "Accuracy is best validation accuracy. Loss is measured at that same checkpoint, selected by accuracy.",
             "These are validation results with no independent test set.", "",
             "| Phase | Variant | Accuracy mean ± SD (%) | Median (%) |",
             "|---|---|---:|---:|"]
    for item in stats:
        if item["metric"] == "best_val_accuracy":
            lines.append(f"| {item['phase']} | {item['variant']} | {item['mean']:.2f} ± {item['std']:.2f} | {item['median']:.2f} |")
    lines += ["", "![Accuracy](accuracy_boxplot.png)", "", "![Loss](loss_boxplot.png)", ""]
    (root / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps(stats, indent=2))


def suite(root):
    import fcntl
    lock = (root / "suite.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = root / "suite.status.json"
    jobs = [(seed, variant) for variant in ["filterbank_residual", "filterbank", "baseline"]
            for seed in SEEDS if seed != 42]
    completed = []
    for job in list(jobs):
        if (root / f"seed_{job[0]}" / job[1] / "verification.json").exists():
            jobs.remove(job)
            completed.append(job)
    active, failures = {}, {}
    env = dict(os.environ, TF_FORCE_GPU_ALLOW_GROWTH="true", TF_CPP_MIN_LOG_LEVEL="2",
               OPENBLAS_NUM_THREADS="1", TF_NUM_INTRAOP_THREADS="4", TF_NUM_INTEROP_THREADS="2",
               MPLBACKEND="Agg", PYTHONUNBUFFERED="1")
    env["LD_LIBRARY_PATH"] = str(root / "runtime/cudnn") + ":" + env.get("LD_LIBRARY_PATH", "")
    while jobs or active:
        for gpu, (job, process, log) in list(active.items()):
            code = process.poll()
            if code is not None:
                log.close()
                del active[gpu]
                if code:
                    failures[str(job)] = code
                else:
                    completed.append(job)
        if failures and not active:
            raise RuntimeError("Worker failure: " + str(failures))
        usage = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True)
        free = [int(line.split(",")[0]) for line in usage.splitlines()
                if int(line.split(",")[1]) < 500 and int(line.split(",")[0]) not in active]
        if not failures:
            for gpu in free:
                if not jobs:
                    break
                job = jobs.pop(0)
                seed, variant = job
                log = (root / f"seed_{seed}_{variant}.log").open("a")
                process = subprocess.Popen(
                    [sys.executable, "-u", "-m", "experiment_runners.repeats", "worker",
                     "--root", str(root), "--seed", str(seed), "--variant", variant],
                    cwd=Path(__file__).resolve().parents[1], env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)),
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active[gpu] = (job, process, log)
        write_status(status, dict(status="running", completed=completed, pending=jobs,
                     active={gpu: dict(seed=job[0], variant=job[1], pid=p.pid)
                             for gpu, (job, p, _) in active.items()}, failures=failures))
        if jobs or active:
            time.sleep(15)
    plot(root)
    write_status(status, dict(status="complete", completed=completed, reused_seed=42))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["suite", "worker", "plot"])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--variant", choices=VARIANTS)
    args = parser.parse_args()
    args.root = args.root.resolve()
    if args.command == "worker":
        if args.seed is None or args.variant is None:
            parser.error("worker requires --seed and --variant")
        worker(args.root, args.seed, args.variant)
    elif args.command == "plot":
        plot(args.root)
    else:
        try:
            suite(args.root)
        except BaseException:
            write_status(args.root / "suite.status.json",
                         dict(status="failed", traceback=traceback.format_exc()))
            raise
