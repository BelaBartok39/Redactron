"""Tests for the Presidio-based PII analyzer module."""

from __future__ import annotations

import pytest

from redactron.core.analyzer import analyze_text
from redactron.models.schemas import Finding


# ---------------------------------------------------------------------------
# SSN detection
# ---------------------------------------------------------------------------


class TestSSNDetection:
    def test_detect_ssn_dashed_format(self):
        # Use a valid-looking SSN (Presidio validates area/group/serial numbers)
        text = "The applicant SSN is 323-45-6789 on file."
        findings = analyze_text(text, page=0)
        entity_types = [f.entity_type for f in findings]
        assert "US_SSN" in entity_types
        ssn_findings = [f for f in findings if f.entity_type == "US_SSN"]
        assert any("323-45-6789" in f.text for f in ssn_findings)

    def test_ssn_finding_has_correct_fields(self):
        text = "My social security number is 219-09-9999"
        findings = analyze_text(text, page=5, source="hidden_text_under_redaction")
        ssn_findings = [f for f in findings if f.entity_type == "US_SSN"]
        assert len(ssn_findings) >= 1
        f = ssn_findings[0]
        assert isinstance(f, Finding)
        assert f.page == 5
        assert f.source == "hidden_text_under_redaction"
        assert f.confidence > 0
        assert f.start is not None
        assert f.end is not None


# ---------------------------------------------------------------------------
# Person name detection
# ---------------------------------------------------------------------------


class TestPersonNameDetection:
    def test_detect_person_name(self):
        text = "The defendant John Michael Smith appeared in court on Monday."
        findings = analyze_text(text, page=0)
        entity_types = [f.entity_type for f in findings]
        assert "PERSON" in entity_types

    def test_multiple_names(self):
        text = "Attorneys Jane Johnson and Robert Williams represented the parties."
        findings = analyze_text(text, page=0)
        person_findings = [f for f in findings if f.entity_type == "PERSON"]
        # Should detect at least one name
        assert len(person_findings) >= 1


# ---------------------------------------------------------------------------
# Phone number and email detection
# ---------------------------------------------------------------------------


class TestPhoneAndEmail:
    def test_detect_phone_number(self):
        text = "Contact the office at 555-123-4567 for more information."
        findings = analyze_text(text, page=0)
        entity_types = [f.entity_type for f in findings]
        assert "PHONE_NUMBER" in entity_types

    def test_detect_email_address(self):
        text = "Send documents to john.doe@example.com for review."
        findings = analyze_text(text, page=0)
        entity_types = [f.entity_type for f in findings]
        assert "EMAIL_ADDRESS" in entity_types
        email_findings = [f for f in findings if f.entity_type == "EMAIL_ADDRESS"]
        assert any("john.doe@example.com" in f.text for f in email_findings)


# ---------------------------------------------------------------------------
# Custom legal recognizers — case numbers
# ---------------------------------------------------------------------------


class TestCaseNumberDetection:
    def test_dashed_case_number(self):
        text = "See case 24-1234-CR for reference."
        findings = analyze_text(text, page=0, score_threshold=0.3)
        entity_types = [f.entity_type for f in findings]
        assert "CASE_NUMBER" in entity_types

    def test_year_prefix_case_number(self):
        text = "Filed under 2024-CF-001234 in the district court."
        findings = analyze_text(text, page=0, score_threshold=0.3)
        entity_types = [f.entity_type for f in findings]
        assert "CASE_NUMBER" in entity_types

    def test_court_prefix_case_number(self):
        text = "Refer to case 1:24-cv-00123 for precedent."
        findings = analyze_text(text, page=0, score_threshold=0.3)
        entity_types = [f.entity_type for f in findings]
        assert "CASE_NUMBER" in entity_types

    def test_compact_case_number(self):
        text = "Record number 24CR001234 was entered."
        findings = analyze_text(text, page=0, score_threshold=0.3)
        entity_types = [f.entity_type for f in findings]
        assert "CASE_NUMBER" in entity_types


# ---------------------------------------------------------------------------
# Custom legal recognizers — docket numbers
# ---------------------------------------------------------------------------


class TestDocketNumberDetection:
    def test_docket_no_format(self):
        text = "Docket No. 12-3456 is scheduled for hearing."
        findings = analyze_text(text, page=0, score_threshold=0.3)
        entity_types = [f.entity_type for f in findings]
        assert "DOCKET_NUMBER" in entity_types

    def test_docket_hash_format(self):
        text = "See Docket #12-3456 for the motion."
        findings = analyze_text(text, page=0, score_threshold=0.3)
        entity_types = [f.entity_type for f in findings]
        assert "DOCKET_NUMBER" in entity_types


# ---------------------------------------------------------------------------
# Confidence threshold filtering
# ---------------------------------------------------------------------------


class TestConfidenceThreshold:
    def test_high_threshold_filters_low_confidence(self):
        text = "The SSN is 219-09-9999."
        high_thresh = analyze_text(text, page=0, score_threshold=0.9)
        low_thresh = analyze_text(text, page=0, score_threshold=0.1)
        # With a higher threshold, we should get fewer or equal findings
        assert len(high_thresh) <= len(low_thresh)

    def test_zero_threshold_returns_all(self):
        text = "John Smith SSN 219-09-9999 email john@example.com phone 555-123-4567"
        findings = analyze_text(text, page=0, score_threshold=0.0)
        assert len(findings) > 0

    def test_threshold_one_returns_none_or_few(self):
        text = "Some text with possible PII like John Smith."
        findings = analyze_text(text, page=0, score_threshold=1.0)
        # At threshold 1.0, most or all findings should be filtered
        assert len(findings) == 0 or all(f.confidence >= 1.0 for f in findings)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestAnalyzerEdgeCases:
    def test_empty_text_returns_no_findings(self):
        findings = analyze_text("", page=0)
        assert findings == []

    def test_whitespace_only_returns_no_findings(self):
        findings = analyze_text("   \n\t  ", page=0)
        assert findings == []

    def test_source_parameter_propagated(self):
        text = "SSN 219-09-9999"
        findings = analyze_text(text, page=0, source="hidden_text_under_redaction")
        for f in findings:
            assert f.source == "hidden_text_under_redaction"

    def test_page_parameter_propagated(self):
        text = "SSN 219-09-9999"
        findings = analyze_text(text, page=42)
        for f in findings:
            assert f.page == 42
