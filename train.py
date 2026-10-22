"""Train or fine-tune EEGNet/EEGConformer for one or multiple subjects.

Example: python train.py 1,2,5 2 2 ME Orig EEGConformer --depth 1 --seed 42
"""
from bci.training_cli import run_training


if __name__ == "__main__":
    run_training(multi_subject=True)
