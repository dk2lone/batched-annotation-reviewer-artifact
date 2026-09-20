from __future__ import annotations

import json
from typing import Any

import numpy as np

from .io import evidence_root, repository_root


def load_protocol() -> dict[str, Any]:
    return json.loads((repository_root() / "protocol.json").read_text(encoding="utf-8"))


def validate() -> dict[str, Any]:
    protocol = load_protocol()
    required_models = {
        "Qwen/Qwen3-8B",
        "ibm-granite/granite-3.3-8b-instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
    }
    models = {row["model_id"] for row in protocol["models"]}
    if models != required_models:
        raise ValueError("model panel differs from the reported study")
    required_datasets = {
        "civil_comments": {
            "dataset_id": "google/civil_comments",
            "revision": "f2970eb3a55777454c94069077cc8d9b5866312d",
            "split": "validation",
        },
        "ag_news": {
            "dataset_id": "fancyzhx/ag_news",
            "revision": "eb185aade064a813bc0b7f42de02595523103ca4",
            "split": "train",
        },
        "multi_nli": {
            "dataset_id": "nyu-mll/multi_nli",
            "revision": "da70db2af9d09693783c3320c4249840212ee221",
            "split": "train",
        },
        "imdb": {
            "dataset_id": "stanfordnlp/imdb",
            "revision": "e6281661ce1c48d982bc483cf8a173c1bbeb5d31",
            "split": "train",
        },
    }
    if protocol["datasets"] != required_datasets:
        raise ValueError("dataset sources differ from the reported study")
    if protocol["decoding"] != {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "seed": 20260912,
        "dtype": "bfloat16",
        "max_model_len": 32768,
        "engine": "vllm==0.28.0",
    }:
        raise ValueError("decoding policy differs from the reported study")
    manifest = json.loads((evidence_root() / "manifest.json").read_text(encoding="utf-8"))
    summary_path = evidence_root() / manifest["batch_size_execution_summary"]
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    for section in ("covariance_grid", "batch_size", "repeated_sampling"):
        for row in manifest[section]:
            path = evidence_root() / row["file"]
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as arrays:
                if not arrays.files:
                    raise ValueError(f"empty evidence file: {path}")
    return {
        "models": len(models),
        "datasets": len(protocol["datasets"]),
        "engine": protocol["decoding"]["engine"],
        "gpu_started": False,
        "status": "syntax_and_evidence_check_passed",
    }


def main() -> None:
    print(json.dumps(validate(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
