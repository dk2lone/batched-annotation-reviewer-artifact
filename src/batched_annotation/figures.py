from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .io import repository_root


def load(name: str) -> dict[str, Any]:
    return json.loads((repository_root() / "outputs" / name).read_text(encoding="utf-8"))


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        bbox_inches="tight",
        metadata={"Creator": "Anonymous reviewer artifact", "CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def covariance_figure(grid: dict[str, Any], output: Path) -> None:
    families = ("qwen", "granite", "llama", "mistral")
    tasks = ("ag_news", "civil_comments", "imdb", "multi_nli")
    values = np.empty((len(families), len(tasks)))
    for i, family in enumerate(families):
        for j, task in enumerate(tasks):
            row = next(
                cell
                for cell in grid["cells"]
                if cell["model_family"] == family and cell["task_id"] == task
            )
            values[i, j] = 100.0 * row["delta"]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    image = ax.imshow(values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(tasks)), ["AG News", "Civil Comments", "IMDb", "MultiNLI"])
    ax.set_yticks(range(len(families)), ["Qwen", "Granite", "Llama", "Mistral"])
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(image, ax=ax, label="Joint minus exact covariance ×100")
    save(fig, output / "covariance_grid.pdf")


def batch_size_figure(batch: dict[str, Any], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    sizes = np.asarray(batch["design"]["sizes"])
    for cell in batch["cells"]:
        values = [row["standardized_contrast"] for row in cell["by_size"]]
        family = Path(cell["file"]).stem.split("--", 1)[0].title()
        ax.plot(sizes, values, marker="o", alpha=0.65, label=family)
    pooled = np.asarray([batch["pooled_standardized_contrast"][str(k)] for k in sizes])
    bounds = np.asarray(
        [batch["pooled_standardized_contrast_assignment_range"][str(k)] for k in sizes]
    )
    ax.plot(sizes, pooled, color="black", linewidth=2.2, label="Equal-cell mean")
    ax.fill_between(
        sizes,
        bounds[:, 0],
        bounds[:, 1],
        color="black",
        alpha=0.15,
        label="Assignment range",
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes, [str(value) for value in sizes])
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Standardized joint-minus-exact dependence")
    ax.legend(frameon=False, ncol=2)
    save(fig, output / "batch_size.pdf")


def coverage_figure(coverage: dict[str, Any], output: Path) -> None:
    methods = ("iid_wald", "call_t", "whole_call_bootstrap")
    labels = ("IID Wald", "Call-aware t", "Whole-call bootstrap")
    x = np.arange(len(coverage["cells"]))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for offset, (method, label) in enumerate(zip(methods, labels, strict=True)):
        rows = [cell["coverage"][method] for cell in coverage["cells"]]
        points = np.asarray([row["point"] for row in rows])
        errors = np.asarray(
            [[point - row["low"] for point, row in zip(points, rows, strict=True)],
             [row["high"] - point for point, row in zip(points, rows, strict=True)]]
        )
        ax.bar(x + (offset - 1) * width, points, width, label=label, yerr=errors, capsize=2)
    ax.axhspan(0.94, 0.96, color="gray", alpha=0.12)
    ax.set_ylim(0.7, 1.0)
    ax.set_ylabel("Empirical 95% interval coverage")
    ax.set_xticks(x, ["Granite/IMDb", "Llama/Civil", "Mistral/MultiNLI", "Qwen/Civil"])
    ax.legend(frameon=False, ncol=3, fontsize=8)
    save(fig, output / "coverage.pdf")


def source_data(
    grid: dict[str, Any], batch: dict[str, Any], coverage: dict[str, Any], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("figure", "model", "task", "series", "value", "x", "low", "high")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for cell in grid["cells"]:
            writer.writerow(
                {
                    "figure": "covariance",
                    "model": cell["model_family"],
                    "task": cell["task_id"],
                    "series": "joint_minus_exact",
                    "value": cell["delta"],
                    "x": 16,
                }
            )
        for cell in batch["cells"]:
            for row in cell["by_size"]:
                writer.writerow(
                    {
                        "figure": "batch_size",
                        "model": cell["model_id"],
                        "task": cell["task_id"],
                        "series": "standardized_contrast",
                        "value": row["standardized_contrast"],
                        "x": row["batch_size"],
                    }
                )
        for cell in coverage["cells"]:
            for method in ("iid_wald", "call_t", "whole_call_bootstrap"):
                row = cell["coverage"][method]
                writer.writerow(
                    {
                        "figure": "coverage",
                        "model": cell["model_id"],
                        "task": cell["task_id"],
                        "series": method,
                        "value": row["point"],
                        "x": 16,
                        "low": row["low"],
                        "high": row["high"],
                    }
                )


def main() -> None:
    output = repository_root() / "outputs" / "figures"
    grid = load("covariance_grid.json")
    batch = load("batch_size.json")
    coverage = load("repeated_sampling.json")
    covariance_figure(grid, output)
    batch_size_figure(batch, output)
    coverage_figure(coverage, output)
    source_data(grid, batch, coverage, output / "source_data.csv")


if __name__ == "__main__":
    main()
