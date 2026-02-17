"""Pydantic models for documents, findings, and batch results."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A single document ingested for redaction QA."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    file_path: Path
    file_type: str
    page_count: int = 0


class Finding(BaseModel):
    """A single PII or redaction finding within a document."""

    entity_type: str
    text: str
    page: int
    confidence: float
    source: Literal["hidden_text_under_redaction", "visible_pii_missed"]
    start: int | None = None
    end: int | None = None


class BatchMetrics(BaseModel):
    """Aggregate quality metrics for a batch of documents."""

    clean_doc_rate: float
    entity_redaction_rate: float
    weighted_risk_score: float
    total_documents: int
    total_findings: int
    findings_by_type: dict[str, int]


class DocumentResult(BaseModel):
    """Processing result for a single document."""

    document: Document
    findings: list[Finding]
    redaction_box_count: int
    has_text_layer: bool
    status: Literal["processed", "error", "skipped"]
    error_message: str | None = None


class BatchResult(BaseModel):
    """Complete result for a batch processing run."""

    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    directory: str
    documents: list[DocumentResult]
    metrics: BatchMetrics
