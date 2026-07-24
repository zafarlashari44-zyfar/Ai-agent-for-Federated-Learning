from enum import StrEnum


class SignalQualityStatus(StrEnum):
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNUSABLE = "unusable"


class EvidenceDirection(StrEnum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    NEUTRAL = "neutral"
    UNAVAILABLE = "unavailable"


class ConsistencyStatus(StrEnum):
    STRONGLY_SUPPORTED = "strongly_supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    LOW_SIGNAL_QUALITY = "low_signal_quality"
    OUT_OF_SCOPE = "out_of_scope"