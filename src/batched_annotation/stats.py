"""Frozen CPU statistics for the study. Every function is design-respecting.

No function here has seen a new model output. They are tested against brute-force
references and closed forms in `tests/test_stats.py`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats as sps

Z975 = 1.959963984540054


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float

    def as_dict(self) -> dict[str, float | None]:
        def clean(x: float) -> float | None:
            return None if isinstance(x, float) and math.isnan(x) else float(x)

        return {"point": clean(self.point), "low": clean(self.low), "high": clean(self.high)}


# --------------------------------------------------------------------------------------
# Pair-weighted within-call covariance and its contrasts
# --------------------------------------------------------------------------------------


def pair_weighted_covariance(errors: np.ndarray, groups: np.ndarray) -> float:
    """C = sum_b sum_{i != j in b} (E_i - Ebar)(E_j - Ebar) / sum_b |b|(|b|-1).

    `errors` may hold NaN for an unresolved label. NaN rows are dropped, which
    shrinks the affected call and is the drop rule named in the protocol. Ebar is
    the mean of the resolved rows of this condition only.
    """
    errors = np.asarray(errors, dtype=float)
    groups = np.asarray(groups)
    keep = ~np.isnan(errors)
    errors, groups = errors[keep], groups[keep]
    if errors.size == 0:
        return math.nan
    centred = errors - errors.mean()
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=centred, minlength=counts.size)
    sumsq = np.bincount(inverse, weights=centred**2, minlength=counts.size)
    pairs = float((counts * (counts - 1)).sum())
    if pairs <= 0:
        return math.nan
    return float((sums**2 - sumsq).sum() / pairs)


def covariance_from_stratified(errors_by_call: np.ndarray) -> float:
    """Fast path for complete equal-size calls, shaped (n_calls, k). No NaN allowed."""
    matrix = np.asarray(errors_by_call, dtype=float)
    n_calls, k = matrix.shape
    centred = matrix - matrix.mean()
    sums = centred.sum(axis=1)
    return float((sums**2 - (centred**2).sum(axis=1)).sum() / (n_calls * k * (k - 1)))


def delta_exact(
    joint_errors: np.ndarray, exact_errors: np.ndarray, groups: np.ndarray
) -> float:
    """Signed joint-minus-exact excess covariance for one assignment."""
    return pair_weighted_covariance(joint_errors, groups) - pair_weighted_covariance(
        exact_errors, groups
    )


def cell_delta(
    joint_by_assignment: list[np.ndarray],
    exact_by_assignment: list[np.ndarray],
    groups_by_assignment: list[np.ndarray],
) -> dict[str, Any]:
    """Cell contrast as the unweighted mean of the per-assignment contrasts."""
    deltas = [
        delta_exact(j, e, g)
        for j, e, g in zip(joint_by_assignment, exact_by_assignment, groups_by_assignment,
                           strict=True)
    ]
    array = np.asarray(deltas, dtype=float)
    return {
        "per_assignment": [float(d) for d in deltas],
        "point": float(np.nanmean(array)),
        "between_assignment_sd": float(np.nanstd(array, ddof=1)) if array.size > 1 else math.nan,
        "aggregation": "unweighted mean over assignments; items recur across assignments",
    }


def fake_batch_contrast(
    joint_errors: np.ndarray,
    independent_errors: np.ndarray,
    real_groups: np.ndarray,
    fake_groups: np.ndarray,
) -> dict[str, float]:
    """C_joint minus the covariance of independently produced labels, regrouped.

    `independent_errors` are exact-context labels, each produced in its own call.
    `fake_groups` come from the frozen assignment generator, so the grouping rule is
    identical and only the label provenance differs.
    """
    c_joint = pair_weighted_covariance(joint_errors, real_groups)
    c_fake = pair_weighted_covariance(independent_errors, fake_groups)
    return {
        "c_joint": float(c_joint),
        "c_fake_batch": float(c_fake),
        "delta_perm": float(c_joint - c_fake),
    }


# --------------------------------------------------------------------------------------
# Design-respecting paired randomization test
# --------------------------------------------------------------------------------------


def _reshape(values: np.ndarray, stratum_matrix: np.ndarray) -> np.ndarray:
    """Arrange item values into (n_calls, k) using a stratum matrix of item indices."""
    array = np.asarray(values, dtype=float)
    if np.isnan(array).any():
        raise ValueError(
            "the randomization test needs a resolved value for every item. Apply one of the "
            "frozen unresolved-label bounds first."
        )
    reshaped: np.ndarray = array[stratum_matrix].T
    return reshaped


def _mean_contrast(
    joint_errors: list[np.ndarray],
    exact_errors: list[np.ndarray],
    matrices: list[np.ndarray],
) -> float:
    """Unweighted mean over assignments of the joint-minus-exact call covariance."""
    return float(
        np.mean(
            [
                covariance_from_stratified(_reshape(j, m))
                - covariance_from_stratified(_reshape(e, m))
                for j, e, m in zip(joint_errors, exact_errors, matrices, strict=True)
            ]
        )
    )


def randomization_test(
    joint_errors: list[np.ndarray],
    exact_errors: list[np.ndarray],
    observed_matrices: list[np.ndarray],
    n_draws: int,
    replay: Callable[[int], list[np.ndarray]],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """One-sided upper randomization test of the sharp no-assignment-effect null.

    `replay(draw)` must be the frozen assignment generator itself, which is
    `replay(draw)` rebuilds the assignment mechanism: the same 16 length strata,
    the same one-item-per-stratum call
    rule, the same 128 calls per assignment in the same call-index order, and so the
    same execution block for every item. Every item still appears once in each of the
    three assignments, so the repeated-item structure is preserved.

    The same draw regroups both arms, which is what makes the test paired. Item outputs
    are held fixed, which is the sharp null. The statistic is symmetric in the members
    of a call, so the slot matching, which the generator applies after partitioning,
    cannot change any value here.
    """
    observed = _mean_contrast(joint_errors, exact_errors, observed_matrices)
    null = np.empty(n_draws, dtype=float)
    for draw in range(n_draws):
        null[draw] = _mean_contrast(joint_errors, exact_errors, replay(draw))

    p_value = float((1 + int((null >= observed).sum())) / (1 + n_draws))
    centred = null - null.mean()
    low, high = np.percentile(centred, [100 * (1 - alpha / 2), 100 * (alpha / 2)])
    return {
        "observed": observed,
        "p_value_one_sided_upper": p_value,
        "n_draws": n_draws,
        "min_reachable_p": 1.0 / (1 + n_draws),
        "null_mean": float(null.mean()),
        "null_sd": float(null.std(ddof=1)),
        "effect_interval": Interval(
            observed, observed - float(low), observed - float(high)
        ).as_dict(),
        "effect_interval_method": (
            "design-respecting shift interval: observed minus the mean-centred replay quantiles"
        ),
        "replay_mechanism": (
            "the frozen assignment generator, replayed by draw index; strata, call sizes, call "
            "index order, each item's observed execution block, and the three repeated appearances "
            "of every item are all preserved"
        ),
        "alpha": alpha,
    }


# --------------------------------------------------------------------------------------
# Recurrence, multiplicity, sensitivity
# --------------------------------------------------------------------------------------


def recurrence_statistic(cell_deltas: dict[str, float]) -> dict[str, Any]:
    """Equal-cell-weight mean contrast over the three confirmation cells."""
    keys = sorted(cell_deltas)
    values = np.asarray([cell_deltas[k] for k in keys], dtype=float)
    return {
        "cells": keys,
        "per_cell": [float(v) for v in values],
        "equal_weight_mean": float(np.mean(values)),
        "weighting": "equal per cell; never by item count or precision",
        "n_cells": len(keys),
    }


def holm(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, Any]:
    """Holm step-down family-wise correction. Adjusted p values are monotone."""
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[name] = running
    return {
        "alpha": alpha,
        "n_tests": m,
        "order": [name for name, _ in items],
        "adjusted_p": adjusted,
        "rejected": {name: adjusted[name] <= alpha for name in p_values},
    }


def malformed_sensitivity(
    joint_errors: np.ndarray,
    exact_errors: np.ndarray,
    groups: np.ndarray,
    joint_unresolved: np.ndarray,
    exact_unresolved: np.ndarray,
) -> dict[str, float]:
    """The four mandatory bounds on unresolved labels.

    `*_errors` hold NaN where a label is unresolved. `*_unresolved` are boolean masks.
    """
    out: dict[str, float] = {}
    for name, fill in (("as_correct", 0.0), ("as_error", 1.0)):
        j = np.where(joint_unresolved, fill, joint_errors)
        e = np.where(exact_unresolved, fill, exact_errors)
        out[f"delta_unresolved_{name}"] = delta_exact(j, e, groups)
    out["delta_unresolved_dropped"] = delta_exact(joint_errors, exact_errors, groups)

    bad_calls = set(np.asarray(groups)[joint_unresolved | exact_unresolved].tolist())
    keep = np.array([g not in bad_calls for g in np.asarray(groups)])
    out["delta_whole_call_discarded"] = (
        delta_exact(joint_errors[keep], exact_errors[keep], np.asarray(groups)[keep])
        if keep.any()
        else math.nan
    )
    out["n_calls_discarded"] = float(len(bad_calls))
    return out


def failure_clustering(unresolved: np.ndarray, groups: np.ndarray) -> float:
    """Pair-weighted covariance of the failure indicator itself."""
    return pair_weighted_covariance(np.asarray(unresolved, dtype=float), groups)


# --------------------------------------------------------------------------------------
# Repeated sampling: intervals, variance ratio, coverage
# --------------------------------------------------------------------------------------


def wald_interval(errors: np.ndarray, alpha: float = 0.05) -> Interval:
    """Ordinary item-level Wald interval. No finite-population correction, by decision D7."""
    values = np.asarray(errors, dtype=float)
    n = values.size
    if n == 0:
        return Interval(math.nan, math.nan, math.nan)
    p = float(values.mean())
    z = Z975 if alpha == 0.05 else float(sps.norm.ppf(1 - alpha / 2))
    half = z * math.sqrt(max(p * (1.0 - p), 0.0) / n)
    return Interval(p, p - half, p + half)


def call_t_interval(call_means: np.ndarray, alpha: float = 0.05) -> Interval:
    """t interval over the call means. df = number of calls minus 1."""
    values = np.asarray(call_means, dtype=float)
    n = values.size
    if n < 2:
        return Interval(float(values.mean()) if n else math.nan, math.nan, math.nan)
    point = float(values.mean())
    se = float(values.std(ddof=1)) / math.sqrt(n)
    half = float(sps.t.ppf(1 - alpha / 2, df=n - 1)) * se
    return Interval(point, point - half, point + half)


def fpc(n_sample: int, n_frame: int) -> float:
    """Finite-population correction factor. Reported as a sensitivity only."""
    return math.sqrt(max(1.0 - n_sample / n_frame, 0.0))


def whole_call_bootstrap(
    call_means: np.ndarray, n_boot: int, rng: np.random.Generator, alpha: float = 0.05
) -> Interval:
    """Resample whole call means with replacement. Sensitivity for the call-aware interval."""
    values = np.asarray(call_means, dtype=float)
    if values.size < 2:
        return Interval(math.nan, math.nan, math.nan)
    draws = values[rng.integers(0, values.size, size=(n_boot, values.size))].mean(axis=1)
    low, high = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Interval(float(values.mean()), float(low), float(high))


def variance_ratio(
    replicate_means: np.ndarray,
    nominal_variances: np.ndarray,
    n_boot: int = 2000,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Empirical variance of replicate means over the mean nominal i.i.d. variance."""
    means = np.asarray(replicate_means, dtype=float)
    nominal = np.asarray(nominal_variances, dtype=float)
    if means.size < 2 or nominal.size == 0:
        return {
            "point": math.nan,
            "low": math.nan,
            "high": math.nan,
            "n_replicates": int(means.size),
        }
    empirical = float(means.var(ddof=1))
    nominal_mean = float(nominal.mean())
    point = empirical / nominal_mean if nominal_mean > 0 else math.nan
    generator = rng or np.random.default_rng(0)
    idx = generator.integers(0, means.size, size=(n_boot, means.size))
    boot = means[idx].var(axis=1, ddof=1) / nominal[idx].mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return {
        "point": point,
        "low": float(low),
        "high": float(high),
        "empirical_variance": empirical,
        "mean_nominal_variance": nominal_mean,
        "n_replicates": int(means.size),
        "floor": 1.25,
    }


def clopper_pearson(successes: int, trials: int, alpha: float = 0.05) -> Interval:
    """Exact binomial interval. Used for every measured coverage proportion."""
    if trials <= 0:
        return Interval(math.nan, math.nan, math.nan)
    point = successes / trials
    low = (
        0.0
        if successes == 0
        else float(sps.beta.ppf(alpha / 2, successes, trials - successes + 1))
    )
    high = (
        1.0
        if successes == trials
        else float(sps.beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    )
    return Interval(point, low, high)


def coverage(
    lows: np.ndarray, highs: np.ndarray, target: float, alpha: float = 0.05
) -> dict[str, Any]:
    hit = (np.asarray(lows) <= target) & (target <= np.asarray(highs))
    interval = clopper_pearson(int(hit.sum()), int(hit.size), alpha)
    return {
        "target": float(target),
        "n_replicates": int(hit.size),
        "n_covered": int(hit.sum()),
        **interval.as_dict(),
        "method": "Clopper-Pearson exact interval on the coverage proportion",
    }


def coverage_with_reference_uncertainty(
    lows: np.ndarray,
    highs: np.ndarray,
    target: float,
    target_se: float,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Coverage at the reference target and at target +- z * reference standard error.

    The least favourable of the three is what an undercoverage claim must use, so a
    favourable point estimate cannot carry the claim on its own.
    """
    offsets = {
        "at_target": target,
        "target_minus_z_se": target - Z975 * target_se,
        "target_plus_z_se": target + Z975 * target_se,
    }
    results = {name: coverage(lows, highs, value, alpha) for name, value in offsets.items()}
    worst = max(results, key=lambda name: float(results[name]["high"] or 1.0))
    return {
        "reference_target": float(target),
        "reference_se": float(target_se),
        "variants": results,
        "least_favourable_variant": worst,
        "least_favourable_upper_bound": results[worst]["high"],
        "undercoverage_supported": bool(
            results[worst]["high"] is not None and float(results[worst]["high"]) < 1 - alpha
        ),
        "claim_rule": (
            "the upper Clopper-Pearson bound of the least favourable variant must be below "
            f"{1 - alpha} before undercoverage is claimed"
        ),
    }


def reference_precision_check(
    reference_replicate_means: np.ndarray, mean_nominal_half_width: float
) -> dict[str, Any]:
    """The frozen gate on whether the reference stream is precise enough for a coverage claim."""
    means = np.asarray(reference_replicate_means, dtype=float)
    se = float(means.std(ddof=1) / math.sqrt(means.size)) if means.size > 1 else math.nan
    ratio = se / mean_nominal_half_width if mean_nominal_half_width > 0 else math.nan
    return {
        "n_reference_replicates": int(means.size),
        "target_point": float(means.mean()) if means.size else math.nan,
        "target_monte_carlo_se": se,
        "mean_nominal_half_width": float(mean_nominal_half_width),
        "se_over_half_width": ratio,
        "threshold": 0.25,
        "precise_enough_for_a_coverage_claim": bool(ratio <= 0.25),
        "if_not": (
            "report the empirical variance of replicate means and its ratio to the average "
            "nominal i.i.d. variance; make no measured-coverage claim"
        ),
    }
