from __future__ import annotations

import math

import pytest

from reasoning_pipeline.reasoning.configuration import (
    ReasoningConfiguration,
)


def test_default_configuration_is_valid() -> None:
    configuration = ReasoningConfiguration()

    assert configuration.strong_support_threshold == pytest.approx(
        0.60
    )
    assert configuration.partial_support_threshold == pytest.approx(
        0.15
    )
    assert configuration.minimum_strong_support_items == 2
    assert configuration.reasoning_version == "reasoning-engine-v1"


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("strong_support_threshold", -0.01),
        ("strong_support_threshold", 1.01),
        ("partial_support_threshold", -0.01),
        ("partial_support_threshold", 1.01),
        ("conflict_ratio_threshold", -0.01),
        ("conflict_ratio_threshold", 1.01),
        ("low_signal_quality_threshold", -0.01),
        ("low_signal_quality_threshold", 1.01),
        ("low_reliability_threshold", -0.01),
        ("low_reliability_threshold", 1.01),
        ("conflict_penalty", -0.01),
        ("conflict_penalty", 1.01),
        ("insufficient_evidence_factor", -0.01),
        ("insufficient_evidence_factor", 1.01),
        ("low_signal_quality_factor", -0.01),
        ("low_signal_quality_factor", 1.01),
    ],
)
def test_configuration_rejects_values_outside_unit_interval(
    field_name: str,
    invalid_value: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="between zero and one",
    ):
        ReasoningConfiguration(
            **{field_name: invalid_value},
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        math.inf,
        -math.inf,
        math.nan,
    ],
)
def test_configuration_rejects_non_finite_values(
    invalid_value: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="must be finite",
    ):
        ReasoningConfiguration(
            conflict_penalty=invalid_value,
        )


@pytest.mark.parametrize(
    ("strong_threshold", "partial_threshold"),
    [
        (0.50, 0.50),
        (0.40, 0.50),
    ],
)
def test_strong_threshold_must_exceed_partial_threshold(
    strong_threshold: float,
    partial_threshold: float,
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "strong_support_threshold must be greater than "
            "partial_support_threshold"
        ),
    ):
        ReasoningConfiguration(
            strong_support_threshold=strong_threshold,
            partial_support_threshold=partial_threshold,
        )


@pytest.mark.parametrize("invalid_count", [0, -1])
def test_minimum_support_item_count_must_be_positive(
    invalid_count: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="must be at least one",
    ):
        ReasoningConfiguration(
            minimum_strong_support_items=invalid_count,
        )


@pytest.mark.parametrize("invalid_version", ["", " ", "\n"])
def test_reasoning_version_cannot_be_blank(
    invalid_version: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="reasoning_version cannot be empty",
    ):
        ReasoningConfiguration(
            reasoning_version=invalid_version,
        )


def test_custom_valid_configuration_is_accepted() -> None:
    configuration = ReasoningConfiguration(
        strong_support_threshold=0.75,
        partial_support_threshold=0.25,
        conflict_ratio_threshold=0.30,
        low_signal_quality_threshold=0.40,
        low_reliability_threshold=0.45,
        conflict_penalty=0.60,
        insufficient_evidence_factor=0.35,
        low_signal_quality_factor=0.25,
        minimum_strong_support_items=3,
        reasoning_version="reasoning-engine-v2-test",
    )

    assert configuration.strong_support_threshold == pytest.approx(
        0.75
    )
    assert configuration.minimum_strong_support_items == 3
    assert (
        configuration.reasoning_version
        == "reasoning-engine-v2-test"
    )
