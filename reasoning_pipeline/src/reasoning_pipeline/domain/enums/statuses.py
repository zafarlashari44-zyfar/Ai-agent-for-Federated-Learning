from enum import StrEnum


class SignalQualityStatus(StrEnum):
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNUSABLE = "unusable"


class SignalSuitabilityStatus(StrEnum):
    ACCEPTED = "accepted"
    ACCEPTED_WITH_WARNINGS = "accepted_with_warnings"
    REJECTED = "rejected"


class OODStatus(StrEnum):
    IN_DISTRIBUTION_LIKE = "in_distribution_like"
    UNCERTAIN = "uncertain"
    LIKELY_OUT_OF_DISTRIBUTION = "likely_out_of_distribution"


class AnalysisScope(StrEnum):
    VALIDATED_MIT_BIH_COMPATIBLE = "validated_mit_bih_compatible"
    EXPLORATORY_EXTERNAL_SOURCE = "exploratory_external_source"
    UNSUPPORTED = "unsupported"


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
