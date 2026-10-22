"""Shared command-line options for single- and multi-subject offline training."""
import argparse
from pathlib import Path

from bci.filter_bank import FILTER_BANK_MODES, BROADBAND_RESIDUAL_MODES, validate_residual_mode


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return value


def parse_training_args(argv=None, multi_subject=False):
    parser = argparse.ArgumentParser(description="Train EEGNet/EEGConformer with an optional filter bank.")
    parser.add_argument("subjects", help="Subject ID" if not multi_subject else "Comma-separated subject IDs")
    parser.add_argument("session_num", type=positive_int)
    parser.add_argument("nclass", type=int, choices=(2, 3))
    parser.add_argument("task", choices=("MI", "ME"))
    parser.add_argument("modeltype", choices=("Orig", "Finetune"))
    parser.add_argument("model_arch", nargs="?", default="EEGNet", choices=("EEGNet", "EEGConformer"))
    parser.add_argument("--filter-bank", choices=FILTER_BANK_MODES, default="off",
                        help="off: original; mean/weighted/concat: independent four-band frontend")
    parser.add_argument("--seed", type=int, default=None, help="Fix trial split and random initialization")
    parser.add_argument("--broadband-residual", choices=BROADBAND_RESIDUAL_MODES, default="off",
                        help="Add original broadband CNN features: off, fixed (alpha=1), gated (initial alpha=0.1)")
    parser.add_argument("--epochs", type=positive_int, default=None)
    parser.add_argument("--batch-size", type=positive_int, default=32)
    parser.add_argument("--depth", type=positive_int, default=3)
    parser.add_argument("--windowlen", type=float, default=3)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--save-dir", type=Path)
    parser.add_argument("--init-model", type=Path, help="Explicit compatible checkpoint for Finetune")
    args = parser.parse_args(argv)
    if not 0 < args.windowlen <= 5:
        parser.error("--windowlen must be positive and at most 5 seconds.")
    try:
        validate_residual_mode(args.filter_bank, args.broadband_residual)
    except ValueError as error:
        parser.error(str(error))
    try:
        ids = [int(s.strip()) for s in args.subjects.split(",")]
        if any(s <= 0 for s in ids) or (not multi_subject and len(ids) != 1):
            raise ValueError
        if len(set(ids)) != len(ids):
            raise ValueError
    except ValueError:
        parser.error("Provide distinct positive subject IDs; single-subject mode accepts one ID.")
    if args.seed is not None and not 0 <= args.seed < 2 ** 32:
        parser.error("--seed must be between 0 and 2**32 - 1.")
    if args.init_model is not None and args.modeltype != "Finetune":
        parser.error("--init-model is only valid for Finetune.")
    args.subj_ids = sorted(ids)
    return args


def experiment_name(args):
    subjects = (f"S{args.subj_ids[0]:02}" if len(args.subj_ids) == 1 else
                "MultiSubj_" + "-".join(f"S{s:02}" for s in args.subj_ids))
    variant = "" if args.filter_bank == "off" else "_FB-" + args.filter_bank
    if args.broadband_residual != "off":
        variant += "_BB-" + args.broadband_residual
    if args.seed is not None:
        variant += f"_Seed{args.seed}"
    if args.depth != 3:
        variant += f"_Depth{args.depth}"
    if args.windowlen != 3:
        variant += f"_Win{args.windowlen:g}"
    return (f"{subjects}_Sess{args.session_num:02}_{args.task}_{args.nclass}class_"
            f"{args.model_arch}{variant}_{args.modeltype}.h5")


def run_training(argv=None, multi_subject=False):
    # Keep --help and argument validation independent of TensorFlow startup.
    args = parse_training_args(argv, multi_subject)
    from bci.Functions import generate_paths, load_and_filter_data, train_models
    root = Path(__file__).resolve().parents[1]
    data_folder = args.data_dir or (root / "Data" if (root / "Data").is_dir() else Path("/Dataset4/bci"))
    save_folder = args.save_dir or root / "Save"
    save_folder.mkdir(parents=True, exist_ok=True)
    params = dict(maxtriallen=5, windowlen=args.windowlen, depth=args.depth, block_size=128, downsrate=100,
                  bandpass_filt=[4, 40], nclass=args.nclass, model_arch=args.model_arch,
                  filter_bank=args.filter_bank, seed=args.seed, batch_size=args.batch_size,
                  broadband_residual=args.broadband_residual,
                  subj_ids=args.subj_ids, session_num=args.session_num, task=args.task,
                  modeltype=args.modeltype, data_folder=str(data_folder.resolve()))
    if args.epochs is not None:
        params["epochs"] = args.epochs
    paths = []
    for subject in args.subj_ids:
        subject_paths = generate_paths(subject, args.task, args.nclass, args.session_num,
                                       args.modeltype, str(data_folder))
        if not subject_paths:
            raise FileNotFoundError(f"No data found for S{subject:02} in {data_folder}")
        paths.extend(subject_paths)
    save_name = save_folder / experiment_name(args)
    if args.modeltype == "Finetune":
        modelpath = args.init_model or save_name.with_name(save_name.name.replace("_Finetune.h5", "_Orig.h5"))
        # Preserve the existing single-subject EEGNet legacy fallback only.
        if (not modelpath.exists() and args.init_model is None and args.filter_bank == "off"
                and args.seed is None and args.model_arch == "EEGNet"):
            legacy = modelpath.with_name(modelpath.name.replace("_EEGNet_", "_"))
            if legacy.exists():
                modelpath = legacy
        if not modelpath.is_file():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {modelpath}; use --init-model.")
        params["modelpath"] = str(modelpath.resolve())
    params["data_paths"] = [str(Path(p).resolve()) for p in paths]
    data, labels, params = load_and_filter_data(paths, params)
    return train_models(data, labels, str(save_name), params)
