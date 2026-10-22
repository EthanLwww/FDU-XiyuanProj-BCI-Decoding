# main_multi_subject_training.py
# Offline training for Finger-BCI-Decoding; original positional arguments remain supported.
# Example: python main_multi_subject_training.py "1,2,3" 2 2 ME Orig EEGConformer --filter-bank weighted --seed 42
from training_cli import run_training


if __name__ == "__main__":
    run_training(multi_subject=True)
