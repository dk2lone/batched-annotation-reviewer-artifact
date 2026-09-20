from __future__ import annotations

import argparse
import csv
import hashlib
import json
from typing import Any

from .io import evidence_root
from .protocol import load_protocol

PREFIXES = {
    "ag_news": "ag",
    "civil_comments": "cc",
    "imdb": "im",
    "multi_nli": "mn",
}


def normalize_source_row(task: str, row: dict[str, Any]) -> tuple[str, int, str]:
    """Return the normalized text, reference label, and optional source identifier."""
    if task == "multi_nli":
        premise = " ".join(row["premise"].split())
        hypothesis = " ".join(row["hypothesis"].split())
        text = f"premise: {premise}\nhypothesis: {hypothesis}"
        reference = int(row["label"])
        source_id = str(row.get("pairID") or "")
    else:
        text = " ".join(row["text"].split())
        reference = (
            int(float(row["toxicity"]) >= 0.5)
            if task == "civil_comments"
            else int(row["label"])
        )
        source_id = str(row.get("id") or "")
    return text, reference, source_id


def document_id(task: str, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:14]
    return PREFIXES[task] + digest


def load_index(task: str) -> dict[int, dict[str, str]]:
    path = evidence_root() / "dataset_indices" / f"{task}.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return {int(row["source_index"]): row for row in csv.DictReader(handle)}


def verify_task(task: str) -> dict[str, Any]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("install the data dependency group with: uv sync --extra data") from exc

    dataset = load_protocol()["datasets"][task]
    expected = load_index(task)
    stream = load_dataset(
        dataset["dataset_id"],
        split=dataset["split"],
        revision=dataset["revision"],
        streaming=True,
    )
    found = 0
    last_index = max(expected)
    for source_index, source_row in enumerate(stream):
        if source_index > last_index:
            break
        row = expected.get(source_index)
        if row is None:
            continue
        text, reference, source_id = normalize_source_row(task, source_row)
        observed = {
            "doc_id": document_id(task, text),
            "source_id": source_id,
            "reference_normalized": str(reference),
            "n_chars": str(len(text)),
        }
        if task == "civil_comments":
            observed["toxicity_score"] = format(float(source_row["toxicity"]), ".9g")
        for key, value in observed.items():
            if value != row[key]:
                raise ValueError(f"{task} row {source_index}: {key} does not match")
        found += 1
    if found != len(expected):
        raise ValueError(f"{task}: verified {found} of {len(expected)} indexed rows")
    return {"dataset": task, "rows_verified": found, "status": "passed"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify source dataset rows.")
    parser.add_argument("task", choices=tuple(PREFIXES))
    args = parser.parse_args()
    print(json.dumps(verify_task(args.task), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
