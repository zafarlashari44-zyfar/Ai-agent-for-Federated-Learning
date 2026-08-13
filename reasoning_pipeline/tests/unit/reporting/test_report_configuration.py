from __future__ import annotations

import pytest

from reasoning_pipeline.reporting.configuration import ReportConfiguration


def test_default_report_configuration_is_valid() -> None:
    configuration = ReportConfiguration()

    assert configuration.report_version == "clinical-report-v1"
    assert "research and decision-support" in configuration.disclaimer


@pytest.mark.parametrize("invalid_version", ["", " ", "\n"])
def test_report_version_cannot_be_blank(
    invalid_version: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="report_version cannot be empty",
    ):
        ReportConfiguration(report_version=invalid_version)


@pytest.mark.parametrize("invalid_disclaimer", ["", " ", "\n"])
def test_disclaimer_cannot_be_blank(
    invalid_disclaimer: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="disclaimer cannot be empty",
    ):
        ReportConfiguration(disclaimer=invalid_disclaimer)


def test_custom_configuration_is_accepted() -> None:
    configuration = ReportConfiguration(
        report_version="clinical-report-v2",
        disclaimer="Research use only.",
    )

    assert configuration.report_version == "clinical-report-v2"
    assert configuration.disclaimer == "Research use only."
