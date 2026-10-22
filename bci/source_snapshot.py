"""Capture runnable source packages for experiment workers and later reproduction."""
import hashlib
import json
from pathlib import Path
import shutil
import time
import uuid


def snapshot_sources(root, project_root=None):
    """Return a self-contained source tree, preserving any earlier revision.

    Workers use ``python -m`` from this directory, so shared models are captured
    with the experiment code rather than imported from a mutable live checkout.
    """
    root = Path(root).resolve()
    project = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    source = root / "source"
    staging = root / (".source-stage-" + uuid.uuid4().hex)
    staging.mkdir()
    for package in ("bci", "experiment_runners", "scaling_experiments"):
        for path in sorted((project / package).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            target = staging / path.relative_to(project)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    for name in ("train.py", "run_experiments.py", "main_online_processing.py"):
        shutil.copy2(project / name, staging / name)
    for package in ("bci", "experiment_runners", "scaling_experiments"):
        if not (staging / package / "__init__.py").is_file():
            raise FileNotFoundError(f"Source package missing: {package}")
    hashes = {p.relative_to(staging).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(staging.rglob("*.py"))}
    if source.exists():
        revision = root / "source_revisions" / f"before_resume_{time.time_ns()}"
        revision.parent.mkdir(parents=True, exist_ok=True)
        source.rename(revision)
        if (root / "source_sha256.json").exists():
            shutil.copy2(root / "source_sha256.json", revision / "source_sha256.json")
    staging.rename(source)
    temp = root / "source_sha256.json.tmp"
    temp.write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    temp.replace(root / "source_sha256.json")
    return source
