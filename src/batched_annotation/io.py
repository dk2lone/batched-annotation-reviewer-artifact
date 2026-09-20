from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def evidence_root() -> Path:
    return repository_root() / "evidence"


def load_manifest() -> dict[str, Any]:
    return json.loads((evidence_root() / "manifest.json").read_text(encoding="utf-8"))


def load_arrays(relative: str) -> np.lib.npyio.NpzFile:
    return np.load(evidence_root() / relative, allow_pickle=False)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
