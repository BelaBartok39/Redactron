"""SQLite persistence for batch results, documents, and findings."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from redactron.models.schemas import (
    BatchMetrics,
    BatchResult,
    Document,
    DocumentResult,
    Finding,
)

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS batches (
    id                   TEXT PRIMARY KEY,
    name                 TEXT NOT NULL,
    timestamp            TEXT NOT NULL,
    directory            TEXT NOT NULL,
    clean_doc_rate       REAL NOT NULL,
    entity_redaction_rate REAL NOT NULL,
    weighted_risk_score  REAL NOT NULL,
    total_documents      INTEGER NOT NULL,
    total_findings       INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id                  TEXT PRIMARY KEY,
    batch_id            TEXT NOT NULL REFERENCES batches(id),
    filename            TEXT NOT NULL,
    status              TEXT NOT NULL,
    finding_count       INTEGER NOT NULL,
    redaction_box_count INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS findings (
    id           TEXT PRIMARY KEY,
    document_id  TEXT NOT NULL REFERENCES documents(id),
    entity_type  TEXT NOT NULL,
    text_snippet TEXT NOT NULL,
    page         INTEGER NOT NULL,
    confidence   REAL NOT NULL,
    source       TEXT NOT NULL
);
"""


def init_db(db_path: str | Path) -> None:
    """Create the database and tables if they do not exist."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        conn.executescript(_SCHEMA)


def save_batch(db_path: str | Path, batch: BatchResult) -> None:
    """Persist a complete :class:`BatchResult` to the database."""
    init_db(db_path)
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO batches "
            "(id, name, timestamp, directory, clean_doc_rate, entity_redaction_rate, "
            "weighted_risk_score, total_documents, total_findings) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                batch.batch_id,
                batch.name,
                batch.timestamp.isoformat(),
                batch.directory,
                batch.metrics.clean_doc_rate,
                batch.metrics.entity_redaction_rate,
                batch.metrics.weighted_risk_score,
                batch.metrics.total_documents,
                batch.metrics.total_findings,
            ),
        )
        for doc_result in batch.documents:
            doc = doc_result.document
            conn.execute(
                "INSERT OR REPLACE INTO documents "
                "(id, batch_id, filename, status, finding_count, redaction_box_count) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    doc.id,
                    batch.batch_id,
                    doc.filename,
                    doc_result.status,
                    len(doc_result.findings),
                    doc_result.redaction_box_count,
                ),
            )
            for finding in doc_result.findings:
                conn.execute(
                    "INSERT OR REPLACE INTO findings "
                    "(id, document_id, entity_type, text_snippet, page, confidence, source) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        str(uuid.uuid4()),
                        doc.id,
                        finding.entity_type,
                        finding.text,
                        finding.page,
                        finding.confidence,
                        finding.source,
                    ),
                )


def load_batches(db_path: str | Path) -> list[BatchResult]:
    """Load all batches (without full document details) from the database."""
    db_path = Path(db_path)
    if not db_path.exists():
        return []
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        batch_rows = conn.execute(
            "SELECT * FROM batches ORDER BY timestamp DESC"
        ).fetchall()

    return [_load_batch_from_row(db_path, dict(row)) for row in batch_rows]


def load_batch(db_path: str | Path, batch_id: str) -> BatchResult | None:
    """Load a single batch with all documents and findings."""
    db_path = Path(db_path)
    if not db_path.exists():
        return None
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM batches WHERE id = ?", (batch_id,)
        ).fetchone()
    if row is None:
        return None
    return _load_batch_from_row(db_path, dict(row))


def _load_batch_from_row(db_path: Path, batch_row: dict) -> BatchResult:
    """Reconstruct a :class:`BatchResult` from a database batch row."""
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        doc_rows = conn.execute(
            "SELECT * FROM documents WHERE batch_id = ?", (batch_row["id"],)
        ).fetchall()

        doc_results: list[DocumentResult] = []
        findings_by_type: dict[str, int] = {}
        for doc_row in doc_rows:
            doc_dict = dict(doc_row)
            finding_rows = conn.execute(
                "SELECT * FROM findings WHERE document_id = ?", (doc_dict["id"],)
            ).fetchall()
            findings: list[Finding] = []
            for f in finding_rows:
                fd = dict(f)
                finding = Finding(
                    entity_type=fd["entity_type"],
                    text=fd["text_snippet"],
                    page=fd["page"],
                    confidence=fd["confidence"],
                    source=fd["source"],
                )
                findings.append(finding)
                findings_by_type[fd["entity_type"]] = (
                    findings_by_type.get(fd["entity_type"], 0) + 1
                )

            doc_results.append(
                DocumentResult(
                    document=Document(
                        id=doc_dict["id"],
                        filename=doc_dict["filename"],
                        file_path=Path(doc_dict["filename"]),
                        file_type=Path(doc_dict["filename"]).suffix.lower(),
                    ),
                    findings=findings,
                    redaction_box_count=doc_dict["redaction_box_count"],
                    has_text_layer=False,
                    status=doc_dict["status"],
                )
            )

    metrics = BatchMetrics(
        clean_doc_rate=batch_row["clean_doc_rate"],
        entity_redaction_rate=batch_row["entity_redaction_rate"],
        weighted_risk_score=batch_row["weighted_risk_score"],
        total_documents=batch_row["total_documents"],
        total_findings=batch_row["total_findings"],
        findings_by_type=findings_by_type,
    )

    return BatchResult(
        batch_id=batch_row["id"],
        name=batch_row["name"],
        timestamp=datetime.fromisoformat(batch_row["timestamp"]),
        directory=batch_row["directory"],
        documents=doc_results,
        metrics=metrics,
    )


def export_batch_json(batch: BatchResult, output_path: str | Path) -> None:
    """Export a :class:`BatchResult` as a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(batch.model_dump_json(indent=2), encoding="utf-8")
