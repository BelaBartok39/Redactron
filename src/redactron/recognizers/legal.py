"""Custom Presidio recognizers for legal document entity types."""

from __future__ import annotations

from presidio_analyzer import Pattern, PatternRecognizer


def build_case_number_recognizer() -> PatternRecognizer:
    """Return a recognizer for U.S. case number patterns.

    Covers common federal and state formats such as:
    - ``24-12345-CR`` / ``2024-CF-001234``
    - ``1:24-cv-00123``
    - ``24CR001234``
    """
    patterns = [
        Pattern(
            "case_number_dashed",
            r"\b\d{1,2}-\d{2,5}-[A-Z]{2,3}\b",
            0.75,
        ),
        Pattern(
            "case_number_year_prefix",
            r"\b\d{4}-[A-Z]{1,3}-\d{3,6}\b",
            0.80,
        ),
        Pattern(
            "case_number_court_prefix",
            r"\b\d{1,2}:\d{2,4}-[a-z]{2,3}-\d{3,6}\b",
            0.85,
        ),
        Pattern(
            "case_number_compact",
            r"\b\d{2}[A-Z]{2}\d{4,6}\b",
            0.65,
        ),
    ]
    return PatternRecognizer(
        supported_entity="CASE_NUMBER",
        patterns=patterns,
        supported_language="en",
    )


def build_docket_number_recognizer() -> PatternRecognizer:
    """Return a recognizer for docket number patterns.

    Covers formats such as:
    - ``Docket No. 12-3456`` / ``Docket #12-3456``
    - ``No. 2024-1234``
    """
    patterns = [
        Pattern(
            "docket_explicit",
            r"(?i)\bDocket\s*(?:No\.?|#)\s*\d{2,4}-\d{2,6}\b",
            0.90,
        ),
        Pattern(
            "docket_no_prefix",
            r"(?i)\bNo\.\s*\d{2,4}-\d{2,6}\b",
            0.60,
        ),
    ]
    return PatternRecognizer(
        supported_entity="DOCKET_NUMBER",
        patterns=patterns,
        supported_language="en",
    )


def get_legal_recognizers() -> list[PatternRecognizer]:
    """Return all custom legal recognizers ready for registration."""
    return [
        build_case_number_recognizer(),
        build_docket_number_recognizer(),
    ]
