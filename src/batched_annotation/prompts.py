from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .protocol import load_protocol


def _example(docs: list[tuple[str, str]], target_ids: list[str]) -> str:
    del docs
    return '{"labels": {' + ", ".join(f'"{doc_id}": 0' for doc_id in target_ids) + "}}"


def build_messages(
    task: str,
    docs: list[tuple[str, str]],
    target_ids: list[str],
    *,
    variant: str = "baseline",
) -> list[dict[str, str]]:
    """Construct the exact joint or matched-context prompt used by the study."""
    prompts = load_protocol()["prompts"]
    if task not in prompts:
        raise ValueError(f"unknown task {task}")
    ids = [doc_id for doc_id, _ in docs]
    if (
        not ids
        or len(ids) != len(set(ids))
        or not target_ids
        or len(target_ids) != len(set(target_ids))
    ):
        raise ValueError("nonempty unique document and target IDs are required")
    if not set(target_ids) <= set(ids):
        raise ValueError("every target ID must be shown")
    if len(target_ids) not in (1, len(ids)):
        raise ValueError("request either one target or all shown documents")
    if variant not in {"baseline", "definition_last", "brief_directive"}:
        raise ValueError(f"unknown variant {variant}")

    spec = prompts[task]
    instruction = (
        spec["joint_instruction"].format(n=len(docs))
        if len(target_ids) == len(ids)
        else spec["single_instruction"]
    )
    rendered = "\n".join(f'<doc id="{doc_id}">\n{text}\n</doc>' for doc_id, text in docs)
    target_line = (
        "Return one label for every document ID above, and no other ID."
        if len(target_ids) == len(ids)
        else f'Return only the label for document ID "{target_ids[0]}". '
        "Ignore all other documents."
    )
    if variant == "baseline":
        user = (
            f"{instruction}\n\n{spec['definition']}\n\nDocuments:\n{rendered}\n\n{target_line}"
        )
    elif variant == "definition_last":
        user = (
            f"{instruction}\n\nDocuments:\n{rendered}\n\n{spec['definition']}\n\n{target_line}"
        )
    else:
        user = f"{spec['definition']}\n\n{instruction}\nDocuments:\n{rendered}\n{target_line}"
    user += "\nAnswer with this JSON object and nothing else:\n" + _example(docs, target_ids)
    return [{"role": "system", "content": spec["system"]}, {"role": "user", "content": user}]


@dataclass(frozen=True)
class ParsedLabels:
    labels: dict[str, int]


def parse_response(raw: str, doc_ids: list[str], task: str) -> ParsedLabels:
    """Apply the exact strict first-response grammar to a model response."""
    classes = {"civil_comments": 2, "imdb": 2, "ag_news": 4, "multi_nli": 3}
    if task not in classes:
        raise ValueError(f"unknown task {task}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(raw, object_pairs_hook=unique_object)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("response is not a single JSON object") from exc
    if not isinstance(payload, dict) or set(payload) != {"labels"}:
        raise ValueError("response must contain only the labels key")
    labels = payload["labels"]
    if (
        not isinstance(labels, dict)
        or set(labels) != set(doc_ids)
        or len(doc_ids) != len(set(doc_ids))
    ):
        raise ValueError("response IDs must match requested IDs exactly")
    for value in labels.values():
        if type(value) is not int or not 0 <= value < classes[task]:
            raise ValueError(f"invalid {task} label")
    return ParsedLabels(labels={doc_id: labels[doc_id] for doc_id in doc_ids})
