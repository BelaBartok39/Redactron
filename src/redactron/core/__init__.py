"""Core engine: ingestion, extraction, OCR, analysis, and redaction checking."""

from redactron.core.redaction_checker import check_document

__all__ = ["check_document"]
