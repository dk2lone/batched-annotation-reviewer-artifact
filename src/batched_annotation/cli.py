from __future__ import annotations

import argparse
import json

from . import (
    batch_size,
    covariance_grid,
    figures,
    followups,
    reference_sensitivity,
    repeated_sampling,
)
from .io import repository_root
from .protocol import validate


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the reported analyses.")
    parser.add_argument(
        "command",
        choices=(
            "all",
            "covariance-grid",
            "batch-size",
            "repeated-sampling",
            "reference-sensitivity",
            "followups",
            "figures",
            "check-gpu-protocol",
        ),
    )
    args = parser.parse_args()
    outputs = repository_root() / "outputs"
    if args.command in ("all", "covariance-grid"):
        covariance_grid.analyze(outputs / "covariance_grid.json")
    if args.command in ("all", "batch-size"):
        batch_size.analyze(outputs / "batch_size.json")
    if args.command in ("all", "repeated-sampling"):
        repeated_sampling.analyze(outputs / "repeated_sampling.json")
    if args.command in ("all", "reference-sensitivity"):
        reference_sensitivity.analyze(outputs / "reference_sensitivity.json")
    if args.command in ("all", "followups"):
        followups.analyze(outputs / "followups.json")
    if args.command in ("all", "figures"):
        figures.main()
    if args.command == "check-gpu-protocol":
        print(json.dumps(validate(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
