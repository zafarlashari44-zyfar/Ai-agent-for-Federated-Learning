from __future__ import annotations

import pytest

from reasoning_pipeline.domain.enums.statuses import EvidenceDirection
from reasoning_pipeline.domain.models import EvidenceItem


def test_evidence_item_accepts_valid_data() -> None:
    evidence = EvidenceItem(
        evidence_id="RR_VARIABILITY_HIGH",
        feature_name="sdnn_ms",
        measured_value=142.5,
        unit="ms",
        interpretation="RR variability is elevated",
        direction=EvidenceDirection.SUPPORTS,
        reliability=0.91,
        source_reference="scribe_v2.rhythm.sdnn_ms",
    )

    assert evidence.direction is EvidenceDirection.SUPPORTS
    assert evidence.reliability == pytest.approx(0.91)


def test_evidence_item_rejects_invalid_reliability() -> None:
    with pytest.raises(
        ValueError,
        match="reliability must be between zero and one",
    ):
        EvidenceItem(
            evidence_id="INVALID",
            feature_name="sdnn_ms",
            measured_value=100.0,
            unit="ms",
            interpretation="Invalid test evidence",
            direction=EvidenceDirection.NEUTRAL,
            reliability=1.5,
            source_reference="test",
        )


def test_evidence_item_rejects_empty_identifier() -> None:
    with pytest.raises(ValueError, match="evidence_id cannot be empty"):
        EvidenceItem(
            evidence_id="",
            feature_name="heart_rate",
            measured_value=75.0,
            unit="bpm",
            interpretation="Normal heart rate",
            direction=EvidenceDirection.NEUTRAL,
            reliability=0.9,
            source_reference="test",
        )
