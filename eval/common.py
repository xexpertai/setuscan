"""Shared helpers for evaluation scripts: provenance written into every result file."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "results"
sys.path.insert(0, str(ROOT))


def provenance(script: str) -> dict:
    try:
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, timeout=10).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "setuscan"],
                                    capture_output=True, text=True, timeout=10).stdout.strip())
    except Exception:
        commit, dirty = "unknown", None
    return {"script": script, "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": commit, "setuscan_dirty": dirty, "opencv": cv2.__version__, "numpy": np.__version__,
            "python": platform.python_version(), "machine": platform.machine(), "cpu_count": __import__("os").cpu_count()}


def write_json(name: str, data: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / name
    p.write_text(json.dumps(data, indent=1, default=float))
    print("wrote", p.relative_to(ROOT))
    return p


def prf(tp: int, fp: int, fn: int, tn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    acc = (tp + tn) / max(1, tp + fp + fn + tn)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(f1, 4), "accuracy": round(acc, 4)}
