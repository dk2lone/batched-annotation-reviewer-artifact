from __future__ import annotations

import pytest

from batched_annotation import (
    batch_size,
    covariance_grid,
    reference_sensitivity,
    repeated_sampling,
)
from batched_annotation.datasets import document_id, normalize_source_row
from batched_annotation.prompts import build_messages, parse_response
from batched_annotation.protocol import validate


@pytest.fixture(scope="session")
def grid() -> dict:
    return covariance_grid.analyze()


@pytest.fixture(scope="session")
def sizes() -> dict:
    return batch_size.analyze()


@pytest.fixture(scope="session")
def coverage() -> dict:
    return repeated_sampling.analyze()


def test_covariance_grid(grid: dict) -> None:
    primary = grid["primary"]
    assert primary["equal_cell_delta"] == pytest.approx(0.009610428454147446)
    assert primary["interval_95"] == pytest.approx(
        [0.009359636993365054, 0.009857107425222694]
    )
    assert primary["one_sided_p"] == 0.0002
    assert primary["holm_significant_against_zero_count"] == 16
    assert primary["point_estimate_at_practical_floor_count"] == 13
    assert grid["malformed_output_sensitivity"]["range"] == pytest.approx(
        [0.007522245512483648, 0.015495075368218952]
    )
    assert grid["recovered_output_sensitivity"]["equal_cell_delta"] == pytest.approx(
        0.007898874435987737
    )
    assert grid["recovered_output_sensitivity"]["holm_positive_count"] == 13
    assert grid["recovered_output_sensitivity"]["null_hypothesis"] == (
        "zero excess covariance"
    )
    assert grid["fake_batch_range"] == pytest.approx(
        [-0.0008596632215711804, 0.0008167435725529989]
    )
    assert grid["assignment_robustness"]["positive_cell_assignment_contrasts"] == 48
    assert min(grid["assignment_robustness"]["leave_one_model_out"].values()) > 0
    assert min(grid["assignment_robustness"]["leave_one_task_out"].values()) > 0
    assert grid["standardized"]["mean_cell_assignment_joint_effective_n"] == pytest.approx(
        5021.85067449566
    )
    assert grid["standardized"][
        "mean_cell_assignment_exact_context_effective_n"
    ] == pytest.approx(8104.627930003505)


def test_batch_size(sizes: dict) -> None:
    assert sizes["design"]["scheduled_calls"] == 586_752
    assert sizes["design"]["first_attempt_parse_successes"] == 586_752
    assert sizes["primary"]["equal_cell_slope"] == pytest.approx(-0.009095205498454413)
    assert sizes["primary"]["assignment_slopes"] == pytest.approx(
        {
            "1": -0.009602436342697682,
            "2": -0.008794704394923832,
            "3": -0.00888847575774172,
        }
    )
    assert sizes["primary"]["assignment_range"] == pytest.approx(
        [-0.009602436342697682, -0.008794704394923832]
    )
    assert sizes["prespecified_block_bootstrap_sensitivity"][
        "equal_cell_slope_interval_95"
    ] == pytest.approx([-0.011181794781879095, -0.007181566258109577])
    assert sizes["pooled_standardized_contrast"] == pytest.approx(
        {
            "2": 0.0794792248558128,
            "4": 0.08065801147463807,
            "8": 0.07110971775911112,
            "16": 0.05955897733708248,
            "32": 0.04455271443231851,
        }
    )
    assert sizes["excess_design_effect"]["2"]["point"] == pytest.approx(
        1.0794792248558127
    )
    assert sizes["excess_design_effect"]["32"]["point"] == pytest.approx(
        2.3811341474018737
    )


def test_repeated_sampling(coverage: dict) -> None:
    assert coverage["design"]["total_calls"] == 83_200
    assert coverage["design"]["recovered_calls"] == 1_095
    by_cell = {row["cell_key"]: row for row in coverage["cells"]}
    expected = {
        "ibm-granite/granite-3.3-8b-instruct::imdb": (0.7775, 0.93, 0.9175),
        "Qwen/Qwen3-8B::civil_comments": (0.88, 0.96, 0.9525),
        "mistralai/Mistral-7B-Instruct-v0.3::multi_nli": (0.9075, 0.9525, 0.9475),
        "meta-llama/Llama-3.1-8B-Instruct::civil_comments": (0.9675, 0.965, 0.94),
    }
    for key, values in expected.items():
        row = by_cell[key]["coverage"]
        assert (
            row["iid_wald"]["point"],
            row["call_t"]["point"],
            row["whole_call_bootstrap"]["point"],
        ) == pytest.approx(values)
    granite = by_cell["ibm-granite/granite-3.3-8b-instruct::imdb"]
    assert granite["variance_ratio"]["point"] == pytest.approx(2.7109804286209247)
    assert granite["repair_free_subset"]["iid_wald"]["point"] == pytest.approx(
        0.7894736842105263
    )
    for cell in by_cell.values():
        assert cell["reference"]["precision_check"][
            "precise_enough_for_a_coverage_claim"
        ]
    supported = {
        key
        for key, cell in by_cell.items()
        if cell["reference"]["coverage_uncertainty_check"]["undercoverage_supported"]
    }
    assert supported == {
        "ibm-granite/granite-3.3-8b-instruct::imdb",
        "mistralai/Mistral-7B-Instruct-v0.3::multi_nli",
        "Qwen/Qwen3-8B::civil_comments",
    }


def test_civil_comments_reference_sensitivity() -> None:
    result = reference_sensitivity.analyze()
    conditions = result["conditions"]
    assert conditions["exclude_open_0.4_0.6"]["retained_frame_items"] == 7_683
    for name in ("threshold_0.4", "threshold_0.5", "threshold_0.6", "exclude_open_0.4_0.6"):
        condition = conditions[name]
        assert condition["covariance_grid"]["positive_cells"] == 4
        slopes = [
            row["slope"] for row in condition["batch_size"]["cells"].values()
        ]
        assert all(value < 0 for value in slopes)
    assert conditions["threshold_0.4"]["repeated_sampling"]["cells"][
        "Qwen/Qwen3-8B"
    ]["iid_wald"]["point"] == pytest.approx(0.915)


def test_protocol_check_starts_no_gpu() -> None:
    result = validate()
    assert result["gpu_started"] is False
    assert result["models"] == 4
    assert result["datasets"] == 4


def test_joint_and_exact_prompts_keep_identical_context() -> None:
    docs = [("cc1", "First document."), ("cc2", "Second document.")]
    joint = build_messages("civil_comments", docs, ["cc1", "cc2"])
    exact = build_messages("civil_comments", docs, ["cc1"])
    for _, text in docs:
        assert text in joint[1]["content"]
        assert text in exact[1]["content"]
    assert joint[1]["content"].index("First document.") < joint[1]["content"].index(
        "Second document."
    )
    assert exact[1]["content"].index("First document.") < exact[1]["content"].index(
        "Second document."
    )


def test_strict_parser_and_dataset_normalization() -> None:
    parsed = parse_response('{"labels":{"ag1":2}}', ["ag1"], "ag_news")
    assert parsed.labels == {"ag1": 2}
    with pytest.raises(ValueError):
        parse_response('{"labels":{"ag1":true}}', ["ag1"], "ag_news")
    text, reference, source_id = normalize_source_row(
        "civil_comments", {"text": "  a   comment ", "toxicity": 0.75}
    )
    assert (text, reference, source_id) == ("a comment", 1, "")
    assert document_id("civil_comments", text).startswith("cc")
