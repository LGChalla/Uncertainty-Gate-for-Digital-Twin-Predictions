"""
Versioned, append-only run log (protocol steps 3 and 37).

Every run appends one JSON line: what ran, with which code, config, seed,
inputs and outputs, plus the environment. Lines are never edited or deleted.
The same log is the audit trail and, later, the twin's update history
(a patient's next calibration starts from its last logged posterior).
"""

import hashlib
import json
import os
import platform
import subprocess
import time
import uuid

import torch


def set_deterministic(seed: int | None = None):
    """Same seed -> bit-identical output (protocol step 3)."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if seed is not None:
        torch.manual_seed(seed)


def environment() -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, timeout=5).stdout.strip() or None
    except Exception:
        commit = None
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": gpu,
        "host": platform.node(),
        "git_commit": commit,
    }


def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class RunLog:
    def __init__(self, path="runs/runlog.jsonl"):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def append(self, step: str, *, config_hash: str, seed, inputs: dict,
               outputs: dict, verdicts: dict | None = None, parent: str | None = None) -> str:
        run_id = uuid.uuid4().hex[:12]
        rec = {
            "run_id": run_id,
            "parent": parent,              # previous run this one builds on (updates)
            "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "step": step,
            "config_hash": config_hash,
            "seed": seed,
            "inputs": inputs,
            "outputs": outputs,
            "gate_verdicts": verdicts or {},
            "env": environment(),
        }
        with open(self.path, "a") as f:            # append only
            f.write(json.dumps(rec, default=str) + "\n")
        return run_id

    def read(self):
        if not os.path.exists(self.path):
            return []
        with open(self.path) as f:
            return [json.loads(line) for line in f if line.strip()]
