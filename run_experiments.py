"""Unified entry point for ablations, repeated runs and subject-scaling studies."""
import argparse
import runpy
import sys


COMMANDS = {
    "ablation": "experiment_runners.ablation",
    "repeats": "experiment_runners.repeats",
    "scaling": "scaling_experiments.run_scaling",
    "scaling-residual": "scaling_experiments.run_scaling_residual",
    "summarize": "scaling_experiments.summarize_scaling",
    "summarize-residual": "scaling_experiments.summarize_residual",
    "compare": "scaling_experiments.compare_scaling_models",
}


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Use python run_experiments.py COMMAND --help for that experiment's options.")
    parser.add_argument("command", choices=COMMANDS)
    if not args:
        parser.print_help()
        return
    command = parser.parse_args(args[:1]).command
    original_argv = sys.argv
    try:
        sys.argv = [f"{parser.prog} {command}", *args[1:]]
        runpy.run_module(COMMANDS[command], run_name="__main__")
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
