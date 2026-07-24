from enum import Enum


class SignalQualityStatus(str, Enum):
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNUSABLE = "unusable"


class EvidenceDirection(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    NEUTRAL = "neutral"
    UNAVAILABLE = "unavailable"


class ConsistencyStatus(str, Enum):
    STRONGLY_SUPPORTED = "strongly_supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    LOW_SIGNAL_QUALITY = "low_signal_quality"
    OUT_OF_SCOPE = "out_of_scope"
