"""Check public commands and independently runnable experiment source snapshots."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from bci.source_snapshot import snapshot_sources


PROJECT = Path(__file__).resolve().parents[1]


class EntrypointTests(unittest.TestCase):
    def test_public_help_from_another_working_directory(self):
        commands = [["train.py", "--help"], ["run_experiments.py", "--help"]]
        for command in ("ablation", "repeats", "scaling", "scaling-residual",
                        "summarize", "summarize-residual", "compare"):
            commands.append(["run_experiments.py", command, "--help"])
        with tempfile.TemporaryDirectory() as directory:
            for command in commands:
                with self.subTest(command=command):
                    result = subprocess.run([sys.executable, str(PROJECT / command[0]), *command[1:]],
                                            cwd=directory, capture_output=True, text=True, timeout=60)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("usage:", result.stdout)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_snapshot_contains_dependencies_and_preserves_previous_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = snapshot_sources(root)
            manifest = json.loads((root / "source_sha256.json").read_text())
            for name, digest in manifest.items():
                self.assertEqual(hashlib.sha256((source / name).read_bytes()).hexdigest(), digest)
            self.assertIn("bci/FilterBankModels_tf.py", manifest)
            self.assertIn("scaling_experiments/streaming_data.py", manifest)
            for module in ("scaling_experiments.run_scaling", "scaling_experiments.run_scaling_residual",
                           "experiment_runners.repeats"):
                result = subprocess.run([sys.executable, "-m", module, "--help"], cwd=source,
                                        capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr)
            marker = source / "previous-revision.txt"
            marker.write_text("preserve me", encoding="utf-8")
            # Resume from the snapshot itself, as experiment workers can do.
            snapshot_sources(root, project_root=source)
            revisions = list((root / "source_revisions").iterdir())
            self.assertEqual(len(revisions), 1)
            self.assertEqual((revisions[0] / marker.name).read_text(), "preserve me")
            self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
