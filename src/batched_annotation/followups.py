from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as scipy_stats

from .io import repository_root, write_json


def load_output(name: str) -> dict[str, Any]:
    return json.loads((repository_root() / "outputs" / name).read_text(encoding="utf-8"))


def analyze(output: Path | None = None) -> dict[str, Any]:
    grid = load_output("covariance_grid.json")
    batch = load_output("batch_size.json")
    coverage = load_output("repeated_sampling.json")
    grid_cells = {row["cell_key"]: row for row in grid["cells"]}
    coherence = []
    for row in coverage["cells"]:
        source = grid_cells[row["cell_key"]]
        joint_covariance = float(
            np.mean([item["joint_covariance"] for item in source["assignments"]])
        )
        error_rate = float(
            np.mean([item["joint_error_rate"] for item in source["assignments"]])
        )
        implied = 1.0 + 15.0 * joint_covariance / (error_rate * (1.0 - error_rate))
        coherence.append(
            {
                "cell_key": row["cell_key"],
                "grid_implied_design_effect": implied,
                "repeated_sampling_variance_ratio": row["variance_ratio"]["point"],
            }
        )
    implied_values = np.asarray([row["grid_implied_design_effect"] for row in coherence])
    measured_values = np.asarray(
        [row["repeated_sampling_variance_ratio"] for row in coherence]
    )
    result = {
        "batch_size_information": {
            "excess_design_effect": batch["excess_design_effect"],
            "effective_n": batch["effective_n"],
        },
        "grid_cell_information_range": {
            "excess_design_effect": [
                min(row["excess_design_effect"] for row in grid["cells"]),
                max(row["excess_design_effect"] for row in grid["cells"]),
            ],
            "effective_n": [
                min(row["effective_n"] for row in grid["cells"]),
                max(row["effective_n"] for row in grid["cells"]),
            ],
        },
        "cross_experiment_coherence": {
            "cells": coherence,
            "pearson": float(scipy_stats.pearsonr(implied_values, measured_values).statistic),
            "spearman": float(
                scipy_stats.spearmanr(implied_values, measured_values).statistic
            ),
            "scope": "Descriptive comparison across four cells, not a prediction rule.",
        },
    }
    if output is not None:
        write_json(output, result)
    return result


if __name__ == "__main__":
    analyze(repository_root() / "outputs/followups.json")
