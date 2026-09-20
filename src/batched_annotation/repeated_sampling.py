from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from . import stats
from .io import evidence_root, load_arrays, load_manifest, write_json

REFERENCE_REPLICATES = 250
EVALUATION_REPLICATES = 400
CALLS_PER_REPLICATE = 32
ITEMS_PER_CALL = 16
BOOTSTRAP_DRAWS = 2_000


def replicate_intervals(errors: np.ndarray) -> dict[str, Any]:
    flat = errors.ravel()
    call_means = errors.mean(axis=1)
    wald = stats.wald_interval(flat)
    call_t = stats.call_t_interval(call_means)
    return {
        "mean": float(flat.mean()),
        "iid_wald": wald,
        "call_t": call_t,
        "call_means": call_means,
        "nominal_variance": float(flat.mean() * (1.0 - flat.mean()) / flat.size),
    }


def wilson_interval(values: np.ndarray) -> stats.Interval:
    values = np.asarray(values, dtype=float)
    n = values.size
    p = float(values.mean())
    z = stats.Z975
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return stats.Interval(p, centre - half, centre + half)


def analyze_cell(metadata: dict[str, Any]) -> dict[str, Any]:
    arrays = load_arrays(str(metadata["file"]))
    reference = np.asarray(arrays["reference_recovered"], dtype=float)
    evaluation = np.asarray(arrays["evaluation_recovered"], dtype=float)
    if reference.shape != (REFERENCE_REPLICATES, CALLS_PER_REPLICATE, ITEMS_PER_CALL):
        raise ValueError("unexpected reference array shape")
    if evaluation.shape != (EVALUATION_REPLICATES, CALLS_PER_REPLICATE, ITEMS_PER_CALL):
        raise ValueError("unexpected evaluation array shape")
    reference_rows = [replicate_intervals(block) for block in reference]
    evaluation_rows = [replicate_intervals(block) for block in evaluation]
    reference_means = np.asarray([row["mean"] for row in reference_rows])
    evaluation_means = np.asarray([row["mean"] for row in evaluation_rows])
    target = float(reference_means.mean())
    seed = int(arrays["bootstrap_seed"][0])
    rng = np.random.default_rng(seed)
    bootstrap_intervals = [
        stats.whole_call_bootstrap(row["call_means"], BOOTSTRAP_DRAWS, rng)
        for row in evaluation_rows
    ]

    def bounds(name: str) -> tuple[np.ndarray, np.ndarray]:
        intervals = [row[name] for row in evaluation_rows]
        return (
            np.asarray([interval.low for interval in intervals]),
            np.asarray([interval.high for interval in intervals]),
        )

    iid_low, iid_high = bounds("iid_wald")
    call_low, call_high = bounds("call_t")
    boot_low = np.asarray([interval.low for interval in bootstrap_intervals])
    boot_high = np.asarray([interval.high for interval in bootstrap_intervals])
    nominal = np.asarray([row["nominal_variance"] for row in evaluation_rows])
    variance_ratio = stats.variance_ratio(
        evaluation_means, nominal, BOOTSTRAP_DRAWS, rng
    )
    mean_nominal_half_width = float(
        np.mean([row["iid_wald"].high - row["iid_wald"].point for row in evaluation_rows])
    )
    reference_precision = stats.reference_precision_check(
        reference_means, mean_nominal_half_width
    )
    reference_uncertainty = stats.coverage_with_reference_uncertainty(
        iid_low,
        iid_high,
        target,
        float(reference_precision["target_monte_carlo_se"]),
    )

    wilson = [wilson_interval(block.ravel()) for block in evaluation]
    wilson_low = np.asarray([interval.low for interval in wilson])
    wilson_high = np.asarray([interval.high for interval in wilson])
    original_reference = np.asarray(arrays["reference_original"], dtype=float)
    original_evaluation = np.asarray(arrays["evaluation_original"], dtype=float)
    reference_complete = np.isfinite(original_reference).all(axis=(1, 2))
    evaluation_complete = np.isfinite(original_evaluation).all(axis=(1, 2))
    original_target = float(original_reference[reference_complete].mean())
    repair_free_intervals = [
        stats.wald_interval(block.ravel()) for block in original_evaluation[evaluation_complete]
    ]
    repair_free = stats.coverage(
        np.asarray([interval.low for interval in repair_free_intervals]),
        np.asarray([interval.high for interval in repair_free_intervals]),
        original_target,
    )
    return {
        **metadata,
        "target_error_rate": target,
        "repaired_calls": int(metadata["repaired_calls"]),
        "reference": {
            "precision_check": reference_precision,
            "coverage_uncertainty_check": reference_uncertainty,
        },
        "coverage": {
            "iid_wald": {
                **stats.coverage(iid_low, iid_high, target),
                "mean_width": float(np.mean(iid_high - iid_low)),
            },
            "call_t": {
                **stats.coverage(call_low, call_high, target),
                "mean_width": float(np.mean(call_high - call_low)),
            },
            "whole_call_bootstrap": {
                **stats.coverage(boot_low, boot_high, target),
                "mean_width": float(np.mean(boot_high - boot_low)),
            },
            "iid_wilson": {
                **stats.coverage(wilson_low, wilson_high, target),
                "mean_width": float(np.mean(wilson_high - wilson_low)),
            },
        },
        "variance_ratio": variance_ratio,
        "repair_free_subset": {
            "reference_replicates": int(reference_complete.sum()),
            "evaluation_replicates": int(evaluation_complete.sum()),
            "target_error_rate": original_target,
            "iid_wald": repair_free,
            "selection_note": "Descriptive subset; replicate completeness was not randomized.",
        },
    }


def analyze(output: Path | None = None) -> dict[str, Any]:
    cells = [analyze_cell(meta) for meta in load_manifest()["repeated_sampling"]]
    total_calls = len(cells) * (
        REFERENCE_REPLICATES + EVALUATION_REPLICATES
    ) * CALLS_PER_REPLICATE
    result = {
        "experiment": "repeated_sampling_coverage",
        "design": {
            "reference_replicates_per_cell": REFERENCE_REPLICATES,
            "evaluation_replicates_per_cell": EVALUATION_REPLICATES,
            "calls_per_replicate": CALLS_PER_REPLICATE,
            "items_per_call": ITEMS_PER_CALL,
            "total_calls": total_calls,
            "recovered_calls": int(sum(cell["repaired_calls"] for cell in cells)),
            "bootstrap_draws": BOOTSTRAP_DRAWS,
        },
        "cells": cells,
    }
    if output is not None:
        write_json(output, result)
    return result


if __name__ == "__main__":
    analyze(evidence_root().parent / "outputs/repeated_sampling.json")
