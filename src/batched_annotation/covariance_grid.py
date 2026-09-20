from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from . import stats
from .io import evidence_root, load_arrays, load_manifest, write_json

N_ITEMS = 8_192
BATCH_SIZE = 16
PRACTICAL_FLOOR = 0.002
ALPHA = 0.05


def covariance(values: np.ndarray, *, fill: float | None = None) -> float:
    matrix = np.asarray(values, dtype=float)
    if fill is not None:
        matrix = np.where(np.isnan(matrix), fill, matrix)
    if np.isnan(matrix).any():
        groups = np.repeat(np.arange(matrix.shape[0]), matrix.shape[1])
        return stats.pair_weighted_covariance(matrix.ravel(), groups)
    return stats.covariance_from_stratified(matrix)


def standardized(cov: float, values: np.ndarray, *, fill: float = 0.0) -> float:
    clean = np.where(np.isnan(values), fill, values)
    p = float(clean.mean())
    spread = p * (1.0 - p)
    return cov / spread if spread > 0 else math.nan


def sensitivity(joint: np.ndarray, exact: np.ndarray) -> dict[str, float]:
    groups = np.repeat(np.arange(joint.shape[0]), joint.shape[1])
    j, e = joint.ravel(), exact.ravel()
    out: dict[str, float] = {}
    for name, fill in (("unresolved_as_correct", 0.0), ("unresolved_as_error", 1.0)):
        out[name] = stats.delta_exact(
            np.where(np.isnan(j), fill, j), np.where(np.isnan(e), fill, e), groups
        )
    out["unresolved_dropped"] = stats.delta_exact(j, e, groups)
    bad = np.isnan(joint).any(axis=1) | np.isnan(exact).any(axis=1)
    out["whole_call_discarded"] = stats.delta_exact(
        joint[~bad].ravel(),
        exact[~bad].ravel(),
        np.repeat(np.arange((~bad).sum()), joint.shape[1]),
    )
    out["discarded_calls"] = int(bad.sum())
    return out


def replay_summary(observed: float, null: np.ndarray) -> dict[str, Any]:
    centred = null - null.mean()
    high, low = np.percentile(centred, [97.5, 2.5])
    return {
        "observed": observed,
        "p_value_one_sided": float((1 + int((null >= observed).sum())) / (1 + len(null))),
        "draws": int(len(null)),
        "interval_95": [observed - float(high), observed - float(low)],
        "null_mean": float(null.mean()),
        "null_sd": float(null.std(ddof=1)),
    }


def analyze(output: Path | None = None) -> dict[str, Any]:
    manifest = load_manifest()
    cells = manifest["covariance_grid"]
    replay = load_arrays("covariance_grid/randomization_null.npz")
    rows: list[dict[str, Any]] = []
    original_nulls: list[np.ndarray] = []
    recovered_nulls: list[np.ndarray] = []
    treatment_by_cell: dict[str, dict[str, float]] = {}
    recovered_observed: dict[str, float] = {}
    recovered_raw_p: dict[str, float] = {}
    per_assignment_global = np.zeros(3, dtype=float)

    for meta in cells:
        arrays = load_arrays(meta["file"])
        cell_key = str(meta["cell_key"])
        short = Path(meta["file"]).stem
        assignment_rows = []
        sensitivities = []
        recovered_deltas = []
        for assignment in range(1, 4):
            joint = arrays[f"joint_original_assignment_{assignment}"]
            exact = arrays[f"exact_context_original_assignment_{assignment}"]
            solo = arrays[f"solo_original_assignment_{assignment}"]
            joint_cov = covariance(joint, fill=0.0)
            exact_cov = covariance(exact, fill=0.0)
            solo_cov = covariance(solo, fill=0.0)
            delta = joint_cov - exact_cov
            per_assignment_global[assignment - 1] += delta / len(cells)
            assignment_rows.append(
                {
                    "assignment": assignment,
                    "joint_covariance": joint_cov,
                    "exact_context_covariance": exact_cov,
                    "solo_fake_batch_covariance": solo_cov,
                    "delta": delta,
                    "joint_error_rate": float(np.where(np.isnan(joint), 0.0, joint).mean()),
                    "exact_context_error_rate": float(
                        np.where(np.isnan(exact), 0.0, exact).mean()
                    ),
                    "joint_standardized": standardized(joint_cov, joint),
                    "exact_context_standardized": standardized(exact_cov, exact),
                }
            )
            sensitivities.append(sensitivity(joint, exact))
            repaired_joint = arrays[f"joint_recovered_assignment_{assignment}"]
            repaired_exact = arrays[f"exact_context_recovered_assignment_{assignment}"]
            recovered_deltas.append(covariance(repaired_joint) - covariance(repaired_exact))

        observed = float(np.mean([row["delta"] for row in assignment_rows]))
        treatment_names = (
            "unresolved_as_correct",
            "unresolved_as_error",
            "unresolved_dropped",
            "whole_call_discarded",
        )
        treatments = {
            name: float(np.mean([row[name] for row in sensitivities]))
            for name in treatment_names
        }
        treatment_by_cell[cell_key] = treatments
        original_null = replay[f"original_{short}"]
        recovered_null = replay[f"recovered_{short}"]
        original_nulls.append(original_null)
        recovered_nulls.append(recovered_null)
        recovered_value = float(np.mean(recovered_deltas))
        recovered_observed[cell_key] = recovered_value
        recovered_raw_p[cell_key] = float(
            (1 + int((recovered_null >= recovered_value).sum())) / (1 + len(recovered_null))
        )
        rows.append(
            {
                **meta,
                "assignments": assignment_rows,
                "delta": observed,
                "replay": replay_summary(observed, original_null),
                "malformed_output_treatments": treatments,
                "recovered_delta": recovered_value,
            }
        )

    zero_null_p = {
        row["cell_key"]: row["replay"]["p_value_one_sided"] for row in rows
    }
    zero_null_holm = stats.holm(zero_null_p, ALPHA)["adjusted_p"]
    zero_null_positive = [
        row["cell_key"]
        for row in rows
        if zero_null_holm[row["cell_key"]] <= ALPHA
    ]
    practical_floor_cells = [
        row["cell_key"]
        for row in rows
        if row["delta"] >= PRACTICAL_FLOOR
    ]
    recovered_holm = stats.holm(recovered_raw_p, ALPHA)["adjusted_p"]
    recovered_positive = [
        key
        for key, value in recovered_observed.items()
        if value >= PRACTICAL_FLOOR and recovered_holm[key] <= ALPHA
    ]

    observed_global = float(np.mean([row["delta"] for row in rows]))
    global_null = np.column_stack(original_nulls).mean(axis=1)
    global_replay = replay_summary(observed_global, global_null)
    treatment_global = {
        name: float(np.mean([values[name] for values in treatment_by_cell.values()]))
        for name in next(iter(treatment_by_cell.values()))
    }
    all_assignments = [entry for row in rows for entry in row["assignments"]]
    joint_standardized = float(np.mean([row["joint_standardized"] for row in all_assignments]))
    exact_standardized = float(
        np.mean([row["exact_context_standardized"] for row in all_assignments])
    )
    cell_assignment_joint_effective_n = []
    cell_assignment_exact_effective_n = []
    for row in rows:
        cell_assignment_joint_effective_n.extend(
            N_ITEMS / (1.0 + 15.0 * item["joint_standardized"])
            for item in row["assignments"]
        )
        cell_assignment_exact_effective_n.extend(
            N_ITEMS / (1.0 + 15.0 * item["exact_context_standardized"])
            for item in row["assignments"]
        )
        excess = float(
            np.mean(
                [
                    item["joint_standardized"] - item["exact_context_standardized"]
                    for item in row["assignments"]
                ]
            )
        )
        row["standardized_excess_dependence"] = excess
        row["excess_design_effect"] = 1.0 + 15.0 * excess
        row["effective_n"] = N_ITEMS / row["excess_design_effect"]
        row["holm_adjusted_p_against_zero"] = zero_null_holm[row["cell_key"]]
        row["meets_practical_floor"] = row["delta"] >= PRACTICAL_FLOOR

    fake_batch_values = [
        float(np.mean([item["solo_fake_batch_covariance"] for item in row["assignments"]]))
        for row in rows
    ]
    leave_one_model_out = {
        family: float(np.mean([row["delta"] for row in rows if row["model_family"] != family]))
        for family in sorted({row["model_family"] for row in rows})
    }
    leave_one_task_out = {
        task: float(np.mean([row["delta"] for row in rows if row["task_id"] != task]))
        for task in sorted({row["task_id"] for row in rows})
    }
    result = {
        "experiment": "cross_setting_covariance_grid",
        "design": {
            "models": 4,
            "tasks": 4,
            "cells": 16,
            "items_per_cell": N_ITEMS,
            "batch_size": BATCH_SIZE,
            "assignments": 3,
            "randomization_draws": int(len(global_null)),
        },
        "primary": {
            "equal_cell_delta": observed_global,
            "one_sided_p": global_replay["p_value_one_sided"],
            "interval_95": global_replay["interval_95"],
            "practical_floor": PRACTICAL_FLOOR,
            "holm_significant_against_zero_cells": zero_null_positive,
            "holm_significant_against_zero_count": len(zero_null_positive),
            "point_estimate_at_practical_floor_cells": practical_floor_cells,
            "point_estimate_at_practical_floor_count": len(practical_floor_cells),
        },
        "malformed_output_sensitivity": {
            "global_by_treatment": treatment_global,
            "range": [min(treatment_global.values()), max(treatment_global.values())],
        },
        "recovered_output_sensitivity": {
            "equal_cell_delta": float(np.mean(list(recovered_observed.values()))),
            "holm_positive_cells": recovered_positive,
            "holm_positive_count": len(recovered_positive),
            "cell_inclusion_rule": (
                "recovered delta at least 0.002 and Holm-adjusted one-sided randomization "
                "p-value against zero at most 0.05"
            ),
            "null_hypothesis": "zero excess covariance",
            "practical_floor_filter": PRACTICAL_FLOOR,
        },
        "standardized": {
            "joint": joint_standardized,
            "exact_context": exact_standardized,
            "joint_design_effect": 1.0 + 15.0 * joint_standardized,
            "exact_context_design_effect": 1.0 + 15.0 * exact_standardized,
            "mean_cell_assignment_joint_effective_n": float(
                np.mean(cell_assignment_joint_effective_n)
            ),
            "mean_cell_assignment_exact_context_effective_n": float(
                np.mean(cell_assignment_exact_effective_n)
            ),
        },
        "assignment_robustness": {
            "positive_cell_assignment_contrasts": int(
                sum(
                    item["delta"] > 0
                    for row in rows
                    for item in row["assignments"]
                )
            ),
            "global_delta_per_assignment": per_assignment_global.tolist(),
            "leave_one_assignment_out": [
                float(np.mean(np.delete(per_assignment_global, index))) for index in range(3)
            ],
            "leave_one_model_out": leave_one_model_out,
            "leave_one_task_out": leave_one_task_out,
        },
        "fake_batch_range": [min(fake_batch_values), max(fake_batch_values)],
        "cells": rows,
    }
    if output is not None:
        write_json(output, result)
    return result


if __name__ == "__main__":
    analyze(evidence_root().parent / "outputs/covariance_grid.json")
