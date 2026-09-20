from __future__ import annotations

import numpy as np
import pytest

from batched_annotation import stats


def test_pair_weighted_covariance_matches_equal_call_fast_path() -> None:
    values = np.asarray([[0, 1, 1], [1, 0, 0]], dtype=float)
    groups = np.repeat(np.arange(2), 3)
    assert stats.pair_weighted_covariance(values.ravel(), groups) == pytest.approx(
        stats.covariance_from_stratified(values)
    )


def test_clopper_pearson_known_coverage() -> None:
    interval = stats.clopper_pearson(311, 400)
    assert interval.point == 0.7775
    assert interval.low == pytest.approx(0.7335123674808888)
    assert interval.high == pytest.approx(0.8173264856181306)
