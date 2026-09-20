from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .io import evidence_root, load_arrays, load_manifest, write_json

SIZES = (2, 4, 8, 16, 32)
ASSIGNMENTS = (1, 2, 3)
N_ITEMS = 8_192
BOOTSTRAP_DRAWS = 2_000


def covariance(values: np.ndarray) -> float:
    matrix = np.asarray(values, dtype=float)
    mask = ~np.isnan(matrix)
    mean = float(matrix[mask].mean())
    centred = np.where(mask, matrix - mean, 0.0)
    counts = mask.sum(axis=1)
    pairs = float((counts * (counts - 1)).sum())
    sums = centred.sum(axis=1)
    squares = (centred**2).sum(axis=1)
    return float(((sums**2 - squares).sum()) / pairs)


def condition(values: np.ndarray) -> dict[str, float]:
    raw = covariance(values)
    p = float(values[~np.isnan(values)].mean())
    standardized = raw / (p * (1.0 - p))
    size = values.shape[1]
    return {
        "covariance": raw,
        "marginal_error": p,
        "accuracy": 1.0 - p,
        "standardized": standardized,
        "design_effect": 1.0 + (size - 1) * standardized,
    }


def slope(values: list[float]) -> float:
    x = np.log2(np.asarray(SIZES, dtype=float))
    y = np.asarray(values, dtype=float)
    centred = x - x.mean()
    return float((centred * y).sum() / (centred**2).sum())


class Cell:
    def __init__(self, metadata: dict[str, Any]) -> None:
        self.metadata = metadata
        self.cell_key = str(metadata["cell_key"])
        self.arrays = load_arrays(str(metadata["file"]))

    def values(self, size: int, assignment: int, arm: str) -> np.ndarray:
        return self.arrays[f"{arm}_size_{size}_assignment_{assignment}"]

    def blocks(self, size: int, assignment: int, arm: str) -> np.ndarray:
        return self.values(size, assignment, arm).reshape(-1, 32)

    def bootstrap_seed(self, assignment: int, draw: int) -> int:
        return int(self.arrays[f"bootstrap_seeds_assignment_{assignment}"][draw])


def observed_table(cell: Cell) -> dict[str, Any]:
    sizes = []
    for size in SIZES:
        assignments = []
        for assignment in ASSIGNMENTS:
            joint = condition(cell.values(size, assignment, "joint"))
            exact = condition(cell.values(size, assignment, "exact_context"))
            assignments.append(
                {
                    "assignment": assignment,
                    "joint": joint,
                    "exact_context": exact,
                    "standardized_contrast": joint["standardized"] - exact["standardized"],
                    "raw_contrast": joint["covariance"] - exact["covariance"],
                }
            )
        sizes.append(
            {
                "batch_size": size,
                "standardized_contrast": float(
                    np.mean([row["standardized_contrast"] for row in assignments])
                ),
                "raw_contrast": float(np.mean([row["raw_contrast"] for row in assignments])),
                "joint_standardized": float(
                    np.mean([row["joint"]["standardized"] for row in assignments])
                ),
                "exact_context_standardized": float(
                    np.mean([row["exact_context"]["standardized"] for row in assignments])
                ),
                "mean_joint_design_effect": float(
                    np.mean([row["joint"]["design_effect"] for row in assignments])
                ),
                "assignments": assignments,
            }
        )
    return {
        **cell.metadata,
        "by_size": sizes,
        "slope": slope([row["standardized_contrast"] for row in sizes]),
    }


def draw_contrasts(cell: Cell, draw: int) -> list[float]:
    picks = {
        assignment: np.random.default_rng(cell.bootstrap_seed(assignment, draw)).integers(
            0, N_ITEMS // 32, N_ITEMS // 32
        )
        for assignment in ASSIGNMENTS
    }
    output = []
    for size in SIZES:
        assignment_values = []
        for assignment in ASSIGNMENTS:
            joint = cell.blocks(size, assignment, "joint")[picks[assignment]].reshape(-1, size)
            exact = cell.blocks(size, assignment, "exact_context")[picks[assignment]].reshape(
                -1, size
            )
            joint_row, exact_row = condition(joint), condition(exact)
            assignment_values.append(joint_row["standardized"] - exact_row["standardized"])
        output.append(float(np.mean(assignment_values)))
    return output


def interval(values: np.ndarray) -> list[float]:
    low, high = np.percentile(values, [2.5, 97.5])
    return [float(low), float(high)]


def analyze(output: Path | None = None) -> dict[str, Any]:
    cells = [Cell(meta) for meta in load_manifest()["batch_size"]]
    tables = [observed_table(cell) for cell in cells]
    assignment_pooled = {
        assignment: [
            float(
                np.mean(
                    [
                        next(
                            item
                            for item in table["by_size"][position]["assignments"]
                            if item["assignment"] == assignment
                        )["standardized_contrast"]
                        for table in tables
                    ]
                )
            )
            for position in range(len(SIZES))
        ]
        for assignment in ASSIGNMENTS
    }
    assignment_slopes = {
        str(assignment): slope(assignment_pooled[assignment])
        for assignment in ASSIGNMENTS
    }
    draws = np.empty((BOOTSTRAP_DRAWS, len(cells), len(SIZES)), dtype=float)
    for draw in range(BOOTSTRAP_DRAWS):
        for index, cell in enumerate(cells):
            draws[draw, index] = draw_contrasts(cell, draw)
    slopes = np.apply_along_axis(lambda row: slope(row.tolist()), 2, draws)
    pooled_slope = slopes.mean(axis=1)
    pooled_sizes = draws.mean(axis=1)
    pooled_observed = {
        str(size): float(
            np.mean([table["by_size"][position]["standardized_contrast"] for table in tables])
        )
        for position, size in enumerate(SIZES)
    }
    pooled_intervals = {
        str(size): interval(pooled_sizes[:, position])
        for position, size in enumerate(SIZES)
    }
    pooled_assignment_ranges = {
        str(size): [
            float(min(assignment_pooled[a][position] for a in ASSIGNMENTS)),
            float(max(assignment_pooled[a][position] for a in ASSIGNMENTS)),
        ]
        for position, size in enumerate(SIZES)
    }
    design_effects = {}
    effective_n = {}
    for size in SIZES:
        value = pooled_observed[str(size)]
        design_effects[str(size)] = {
            "point": 1.0 + (size - 1) * value,
            "assignment_range": [
                1.0 + (size - 1) * bound
                for bound in pooled_assignment_ranges[str(size)]
            ],
        }
        de = design_effects[str(size)]
        effective_n[str(size)] = {
            "point": N_ITEMS / de["point"],
            "assignment_range": [
                N_ITEMS / de["assignment_range"][1],
                N_ITEMS / de["assignment_range"][0],
            ],
        }
    scheduled_calls = int(
        len(cells)
        * len(ASSIGNMENTS)
        * sum(N_ITEMS // size + N_ITEMS for size in SIZES)
    )
    if any(
        not np.isfinite(cell.values(size, assignment, arm)).all()
        for cell in cells
        for size in SIZES
        for assignment in ASSIGNMENTS
        for arm in ("joint", "exact_context")
    ):
        raise ValueError("batch-size evidence contains an unresolved output")
    summary_path = evidence_root() / load_manifest()["batch_size_execution_summary"]
    execution = json.loads(summary_path.read_text(encoding="utf-8"))
    if not all(
        execution[key] == scheduled_calls
        for key in ("calls", "attempt_1_calls", "parse_status_ok_calls", "status_ok_calls")
    ):
        raise ValueError("batch-size execution summary does not match the scheduled calls")
    result = {
        "experiment": "batch_size_dependence",
        "design": {
            "sizes": list(SIZES),
            "assignments": len(ASSIGNMENTS),
            "items_per_cell": N_ITEMS,
            "cells": len(cells),
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "scheduled_calls": scheduled_calls,
            "first_attempt_parse_successes": execution["attempt_1_calls"],
        },
        "primary": {
            "equal_cell_slope": float(np.mean([table["slope"] for table in tables])),
            "assignment_slopes": assignment_slopes,
            "assignment_range": [
                float(min(assignment_slopes.values())),
                float(max(assignment_slopes.values())),
            ],
            "inference_status": (
                "descriptive: the three complete randomized assignments reuse the same items"
            ),
        },
        "pooled_standardized_contrast": pooled_observed,
        "pooled_standardized_contrast_assignment_range": pooled_assignment_ranges,
        "prespecified_block_bootstrap_sensitivity": {
            "equal_cell_slope_interval_95": interval(pooled_slope),
            "p_two_sided": float(
                min(1.0, 2.0 * min((pooled_slope <= 0).mean(), (pooled_slope >= 0).mean()))
            ),
            "pooled_standardized_contrast_interval_95": pooled_intervals,
            "limitation": (
                "size-32 parent groups were resampled independently by assignment, so this "
                "sensitivity does not preserve recurring item identities across assignments"
            ),
        },
        "excess_design_effect": design_effects,
        "effective_n": effective_n,
        "cells": tables,
    }
    if output is not None:
        write_json(output, result)
    return result


if __name__ == "__main__":
    analyze(evidence_root().parent / "outputs/batch_size.json")
