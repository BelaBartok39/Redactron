"""End-to-end integration tests: synthetic PDFs -> pipeline -> verify results."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import fitz
import pytest

from redactron.core.metrics import compute_batch_metrics
from redactron.core.pipeline import process_batch, scan_directory
from redactron.models.database import init_db, load_batch, save_batch
from redactron.models.schemas import (
    BatchMetrics,
    BatchResult,
    Document,
    DocumentResult,
    Finding,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_pdf_with_pii(directory: Path, filename: str, pii_text: str) -> Path:
    """Create a PDF file with visible PII text in the given directory."""
    path = directory / filename
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), pii_text, fontsize=12)
    doc.save(str(path))
    doc.close()
    return path


def _create_pdf_with_redaction_and_hidden_pii(
    directory: Path,
    filename: str,
    hidden_text: str,
    visible_text: str,
) -> Path:
    """Create a PDF with a redaction annotation covering hidden PII."""
    path = directory / filename
    doc = fitz.open()
    page = doc.new_page()
    # Hidden text under redaction
    page.insert_text((72, 72), hidden_text, fontsize=12)
    # Visible text below
    page.insert_text((72, 300), visible_text, fontsize=12)
    # Add redaction annotation over hidden text
    redact_rect = fitz.Rect(60, 55, 500, 85)
    page.add_redact_annot(redact_rect)
    doc.save(str(path))
    doc.close()
    return path


def _create_clean_pdf(directory: Path, filename: str) -> Path:
    """Create a PDF with no PII content."""
    path = directory / filename
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "This document contains no personally identifiable information.",
        fontsize=12,
    )
    doc.save(str(path))
    doc.close()
    return path


# ---------------------------------------------------------------------------
# Tests for scan_directory
# ---------------------------------------------------------------------------


class TestScanDirectory:
    def test_finds_pdf_files(self, tmp_path):
        _create_clean_pdf(tmp_path, "doc1.pdf")
        _create_clean_pdf(tmp_path, "doc2.pdf")
        docs = scan_directory(tmp_path)
        assert len(docs) == 2
        filenames = {d.filename for d in docs}
        assert filenames == {"doc1.pdf", "doc2.pdf"}

    def test_ignores_unsupported_files(self, tmp_path):
        _create_clean_pdf(tmp_path, "doc.pdf")
        (tmp_path / "readme.txt").write_text("not a supported file")
        (tmp_path / "data.csv").write_text("a,b,c")
        docs = scan_directory(tmp_path)
        assert len(docs) == 1
        assert docs[0].filename == "doc.pdf"

    def test_empty_directory(self, tmp_path):
        docs = scan_directory(tmp_path)
        assert len(docs) == 0

    def test_document_fields(self, tmp_path):
        _create_clean_pdf(tmp_path, "test.pdf")
        docs = scan_directory(tmp_path)
        assert len(docs) == 1
        doc = docs[0]
        assert isinstance(doc, Document)
        assert doc.filename == "test.pdf"
        assert doc.file_type == ".pdf"


# ---------------------------------------------------------------------------
# Tests for process_batch (full pipeline)
# ---------------------------------------------------------------------------


class TestProcessBatch:
    def test_batch_with_visible_pii(self, tmp_path):
        """A PDF with visible SSN should produce findings."""
        _create_pdf_with_pii(
            tmp_path,
            "visible_ssn.pdf",
            "Applicant SSN: 219-09-9999 was submitted for review.",
        )
        db_path = tmp_path / "test.db"
        result = process_batch(
            tmp_path,
            name="test-visible-pii",
            max_workers=1,
            db_path=db_path,
        )
        assert isinstance(result, BatchResult)
        assert result.metrics.total_documents == 1
        # Should detect the SSN
        all_findings = []
        for doc_result in result.documents:
            all_findings.extend(doc_result.findings)
        ssn_findings = [f for f in all_findings if f.entity_type == "US_SSN"]
        assert len(ssn_findings) >= 1

    def test_batch_with_hidden_pii(self, tmp_path):
        """A PDF with PII hidden under a redaction box should detect it."""
        _create_pdf_with_redaction_and_hidden_pii(
            tmp_path,
            "hidden_ssn.pdf",
            hidden_text="SSN 323-45-6789",
            visible_text="This portion is public record.",
        )
        db_path = tmp_path / "test.db"
        result = process_batch(
            tmp_path,
            name="test-hidden-pii",
            max_workers=1,
            db_path=db_path,
        )
        assert result.metrics.total_documents == 1
        # Should find the hidden SSN
        all_findings = []
        for doc_result in result.documents:
            all_findings.extend(doc_result.findings)
        hidden_findings = [
            f for f in all_findings if f.source == "hidden_text_under_redaction"
        ]
        assert len(hidden_findings) >= 1

    def test_batch_with_clean_document(self, tmp_path):
        """A clean PDF should produce zero findings."""
        _create_clean_pdf(tmp_path, "clean.pdf")
        db_path = tmp_path / "test.db"
        result = process_batch(
            tmp_path,
            name="test-clean",
            max_workers=1,
            db_path=db_path,
        )
        assert result.metrics.total_documents == 1
        assert result.metrics.total_findings == 0
        assert result.metrics.clean_doc_rate == pytest.approx(1.0)

    def test_batch_result_structure(self, tmp_path):
        """Verify the BatchResult model has all expected fields populated."""
        _create_pdf_with_pii(
            tmp_path,
            "structure_test.pdf",
            "Name: John Smith, SSN: 219-09-9999",
        )
        db_path = tmp_path / "test.db"
        result = process_batch(
            tmp_path,
            name="test-structure",
            max_workers=1,
            db_path=db_path,
        )
        assert isinstance(result, BatchResult)
        assert result.name == "test-structure"
        assert result.directory == str(tmp_path)
        assert isinstance(result.metrics, BatchMetrics)
        assert isinstance(result.documents, list)
        assert len(result.documents) == 1

        doc_result = result.documents[0]
        assert isinstance(doc_result, DocumentResult)
        assert doc_result.status == "processed"
        assert doc_result.has_text_layer is True
        assert isinstance(doc_result.findings, list)

    def test_batch_multiple_documents(self, tmp_path):
        """Process multiple PDFs in one batch and verify aggregate metrics."""
        _create_pdf_with_pii(
            tmp_path, "with_pii.pdf", "Contact: 555-123-4567"
        )
        _create_clean_pdf(tmp_path, "clean1.pdf")
        _create_clean_pdf(tmp_path, "clean2.pdf")

        db_path = tmp_path / "test.db"
        result = process_batch(
            tmp_path,
            name="test-multi",
            max_workers=1,
            db_path=db_path,
        )
        assert result.metrics.total_documents == 3

    def test_empty_directory_batch(self, tmp_path):
        """Processing an empty directory should produce an empty batch."""
        db_path = tmp_path / "test.db"
        result = process_batch(
            tmp_path,
            name="test-empty",
            max_workers=1,
            db_path=db_path,
        )
        assert result.metrics.total_documents == 0
        assert result.metrics.total_findings == 0


# ---------------------------------------------------------------------------
# Tests for SQLite persistence
# ---------------------------------------------------------------------------


class TestDatabasePersistence:
    def test_batch_saved_to_sqlite(self, tmp_path):
        """After process_batch, the SQLite DB should contain the batch data."""
        _create_pdf_with_pii(
            tmp_path,
            "db_test.pdf",
            "SSN: 219-09-9999, Name: Jane Doe",
        )
        db_path = tmp_path / "test.db"
        result = process_batch(
            tmp_path,
            name="test-db",
            max_workers=1,
            db_path=db_path,
        )

        # Verify the database file exists
        assert db_path.exists()

        # Query the database directly
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        # Check batches table
        batches = conn.execute("SELECT * FROM batches").fetchall()
        assert len(batches) == 1
        batch_row = dict(batches[0])
        assert batch_row["name"] == "test-db"
        assert batch_row["total_documents"] == 1

        # Check documents table
        docs = conn.execute("SELECT * FROM documents").fetchall()
        assert len(docs) == 1
        doc_row = dict(docs[0])
        assert doc_row["filename"] == "db_test.pdf"
        assert doc_row["status"] == "processed"

        # Check findings table
        findings = conn.execute("SELECT * FROM findings").fetchall()
        assert len(findings) >= 1  # Should have at least the SSN finding

        conn.close()

    def test_load_batch_from_db(self, tmp_path):
        """Verify that a saved batch can be reloaded from the database."""
        _create_pdf_with_pii(
            tmp_path, "reload_test.pdf", "Phone: 555-123-4567"
        )
        db_path = tmp_path / "test.db"
        original = process_batch(
            tmp_path,
            name="test-reload",
            max_workers=1,
            db_path=db_path,
        )

        # Reload from DB
        loaded = load_batch(db_path, original.batch_id)
        assert loaded is not None
        assert loaded.batch_id == original.batch_id
        assert loaded.name == "test-reload"
        assert loaded.metrics.total_documents == original.metrics.total_documents
        assert loaded.metrics.total_findings == original.metrics.total_findings

    def test_multiple_batches_in_db(self, tmp_path):
        """Run two batches and verify both are stored."""
        _create_clean_pdf(tmp_path, "doc.pdf")
        db_path = tmp_path / "test.db"

        result1 = process_batch(
            tmp_path, name="batch-1", max_workers=1, db_path=db_path
        )
        result2 = process_batch(
            tmp_path, name="batch-2", max_workers=1, db_path=db_path
        )

        conn = sqlite3.connect(str(db_path))
        batches = conn.execute("SELECT * FROM batches").fetchall()
        conn.close()
        assert len(batches) == 2


# ---------------------------------------------------------------------------
# Tests for progress callback
# ---------------------------------------------------------------------------


class TestProgressCallback:
    def test_callback_invoked(self, tmp_path):
        """The progress callback should be called for each document."""
        _create_clean_pdf(tmp_path, "cb1.pdf")
        _create_clean_pdf(tmp_path, "cb2.pdf")
        db_path = tmp_path / "test.db"

        calls: list[tuple[int, int, str]] = []

        def on_progress(processed: int, total: int, filename: str) -> None:
            calls.append((processed, total, filename))

        process_batch(
            tmp_path,
            name="test-callback",
            max_workers=1,
            db_path=db_path,
            progress_callback=on_progress,
        )

        assert len(calls) == 2
        # All calls should have total=2
        for processed, total, filename in calls:
            assert total == 2
            assert filename.endswith(".pdf")
