from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from . import stats
from .batch_size import ASSIGNMENTS, SIZES, Cell, condition, slope
from .covariance_grid import covariance
from .io import evidence_root, load_arrays, load_manifest, write_json


def civil_comments_frame() -> tuple[np.ndarray, np.ndarray]:
    path = evidence_root() / "dataset_indices/civil_comments.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    references = np.asarray([int(row["reference_normalized"]) for row in rows])
    scores = np.asarray([float(row["toxicity_score"]) for row in rows])
    return references, scores


def transformed_errors(
    errors: np.ndarray,
    schedule: np.ndarray,
    original_references: np.ndarray,
    new_references: np.ndarray,
    keep: np.ndarray,
) -> np.ndarray:
    matrix = np.asarray(errors, dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("reference sensitivity requires resolved outputs")
    old = original_references[schedule]
    predictions = (matrix.astype(int) + old) % 2
    changed = (predictions != new_references[schedule]).astype(float)
    return np.where(keep[schedule], changed, np.nan)


def grid_sensitivity(
    original_references: np.ndarray, new_references: np.ndarray, keep: np.ndarray
) -> dict[str, Any]:
    cells: dict[str, float] = {}
    for metadata in load_manifest()["covariance_grid"]:
        if metadata["task_id"] != "civil_comments":
            continue
        arrays = load_arrays(str(metadata["file"]))
        deltas = []
        for assignment in ASSIGNMENTS:
            schedule = arrays[f"schedule_assignment_{assignment}"]
            joint = transformed_errors(
                arrays[f"joint_recovered_assignment_{assignment}"],
                schedule,
                original_references,
                new_references,
                keep,
            )
            exact = transformed_errors(
                arrays[f"exact_context_recovered_assignment_{assignment}"],
                schedule,
                original_references,
                new_references,
                keep,
            )
            deltas.append(covariance(joint) - covariance(exact))
        cells[str(metadata["model_family"])] = float(np.mean(deltas))
    return {
        "equal_cell_delta": float(np.mean(list(cells.values()))),
        "positive_cells": sum(value > 0 for value in cells.values()),
        "cells": cells,
    }


def batch_size_sensitivity(
    original_references: np.ndarray, new_references: np.ndarray, keep: np.ndarray
) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    for metadata in load_manifest()["batch_size"]:
        if metadata["task_id"] != "civil_comments":
            continue
        cell = Cell(metadata)
        contrasts = []
        for size in SIZES:
            assignment_values = []
            for assignment in ASSIGNMENTS:
                schedule = cell.arrays[f"schedule_assignment_{assignment}"].reshape(-1, size)
                joint = transformed_errors(
                    cell.values(size, assignment, "joint"),
                    schedule,
                    original_references,
                    new_references,
                    keep,
                )
                exact = transformed_errors(
                    cell.values(size, assignment, "exact_context"),
                    schedule,
                    original_references,
                    new_references,
                    keep,
                )
                assignment_values.append(
                    condition(joint)["standardized"] - condition(exact)["standardized"]
                )
            contrasts.append(float(np.mean(assignment_values)))
        cells[str(metadata["model_id"])] = {
            "contrasts": dict(zip((str(size) for size in SIZES), contrasts, strict=True)),
            "slope": slope(contrasts),
        }
    return {"cells": cells}


def repeated_sampling_sensitivity(
    original_references: np.ndarray, new_references: np.ndarray, keep: np.ndarray
) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    for metadata in load_manifest()["repeated_sampling"]:
        if metadata["task_id"] != "civil_comments":
            continue
        arrays = load_arrays(str(metadata["file"]))
        reference = transformed_errors(
            arrays["reference_recovered"],
            arrays["reference_schedule"],
            original_references,
            new_references,
            keep,
        )
        evaluation = transformed_errors(
            arrays["evaluation_recovered"],
            arrays["evaluation_schedule"],
            original_references,
            new_references,
            keep,
        )
        target = float(np.mean([np.nanmean(block) for block in reference]))
        iid_intervals = []
        call_intervals = []
        retained = []
        for block in evaluation:
            flat = block[np.isfinite(block)]
            call_means = np.nanmean(block, axis=1)
            iid_intervals.append(stats.wald_interval(flat))
            call_intervals.append(stats.call_t_interval(call_means))
            retained.append(flat.size)

        def coverage(
            intervals: list[stats.Interval], target_value: float = target
        ) -> dict[str, Any]:
            return stats.coverage(
                np.asarray([interval.low for interval in intervals]),
                np.asarray([interval.high for interval in intervals]),
                target_value,
            )

        cells[str(metadata["model_id"])] = {
            "target_error_rate": target,
            "mean_retained_items": float(np.mean(retained)),
            "iid_wald": coverage(iid_intervals),
            "call_t": coverage(call_intervals),
        }
    return {"cells": cells}


def analyze(output: Path | None = None) -> dict[str, Any]:
    original_references, scores = civil_comments_frame()
    conditions = {
        "threshold_0.4": (scores >= 0.4, np.ones(scores.size, dtype=bool)),
        "threshold_0.5": (scores >= 0.5, np.ones(scores.size, dtype=bool)),
        "threshold_0.6": (scores >= 0.6, np.ones(scores.size, dtype=bool)),
        "exclude_open_0.4_0.6": (
            scores >= 0.5,
            ~((scores > 0.4) & (scores < 0.6)),
        ),
    }
    result: dict[str, Any] = {
        "experiment": "civil_comments_reference_sensitivity",
        "status": "post_hoc",
        "reference_rule": "toxic if toxicity score is at least 0.5",
        "conditions": {},
    }
    for name, (references, keep) in conditions.items():
        result["conditions"][name] = {
            "retained_frame_items": int(keep.sum()),
            "positive_reference_items": int((references & keep).sum()),
            "covariance_grid": grid_sensitivity(
                original_references, references.astype(int), keep
            ),
            "batch_size": batch_size_sensitivity(
                original_references, references.astype(int), keep
            ),
            "repeated_sampling": repeated_sampling_sensitivity(
                original_references, references.astype(int), keep
            ),
        }
    if output is not None:
        write_json(output, result)
    return result


if __name__ == "__main__":
    analyze(evidence_root().parent / "outputs/reference_sensitivity.json")
