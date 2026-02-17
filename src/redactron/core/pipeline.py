"""Batch processing orchestrator: scan, process in parallel, persist."""

from __future__ import annotations

import logging
import multiprocessing
import os
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from redactron import config
from redactron.core.metrics import compute_batch_metrics
from redactron.models.database import save_batch
from redactron.models.schemas import BatchResult, Document, DocumentResult

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], None]


def scan_directory(directory: str | Path) -> list[Document]:
    """Return a list of :class:`Document` for every supported file in *directory*."""
    directory = Path(directory)
    documents: list[Document] = []
    for entry in sorted(directory.rglob("*")):
        if entry.is_file() and entry.suffix.lower() in config.SUPPORTED_EXTENSIONS:
            documents.append(
                Document(
                    filename=entry.name,
                    file_path=entry,
                    file_type=entry.suffix.lower(),
                )
            )
    return documents


def _process_single(file_path: str, confidence_threshold: float) -> DocumentResult:
    """Process a single document — executed inside a worker process."""
    # Import lazily so the heavy engine modules are loaded only in workers.
    from redactron.core.redaction_checker import check_document

    try:
        return check_document(file_path, confidence_threshold=confidence_threshold)
    except Exception as exc:  # noqa: BLE001
        logger.error("Error processing %s: %s", file_path, exc)
        from redactron.core.ingestion import load_document
        document = load_document(file_path)
        return DocumentResult(
            document=document,
            findings=[],
            redaction_box_count=0,
            has_text_layer=False,
            status="error",
            error_message=str(exc),
        )


def process_batch(
    directory: str | Path,
    name: str,
    *,
    confidence_threshold: float = config.DEFAULT_CONFIDENCE_THRESHOLD,
    risk_weights: dict[str, int] | None = None,
    progress_callback: ProgressCallback | None = None,
    max_workers: int = config.MAX_WORKERS,
    db_path: str | Path | None = None,
) -> BatchResult:
    """Scan *directory*, process all documents in parallel, and return a :class:`BatchResult`.

    Parameters
    ----------
    directory:
        Path to the folder containing documents to check.
    name:
        Human-friendly name for this batch run.
    confidence_threshold:
        Findings below this confidence are discarded.
    risk_weights:
        Optional per-entity-type weights for the risk score.
    progress_callback:
        Called as ``progress_callback(processed, total, current_filename)``
        after each document finishes.
    max_workers:
        Number of parallel worker processes.  ``0`` means use all CPUs.
    db_path:
        Path to the SQLite database.  Defaults to ``<directory>/redactron.db``.
    """
    directory = Path(directory)
    documents = scan_directory(directory)

    if not documents:
        logger.warning("No supported files found in %s", directory)

    # Required on Windows where processes are spawned (not forked).
    multiprocessing.freeze_support()

    workers = max_workers if max_workers > 0 else (os.cpu_count() or 1)
    results: list[DocumentResult] = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_doc = {
            executor.submit(_process_single, str(doc.file_path), confidence_threshold): doc
            for doc in documents
        }
        for idx, future in enumerate(as_completed(future_to_doc), start=1):
            doc = future_to_doc[future]
            result = future.result()
            results.append(result)
            if progress_callback is not None:
                progress_callback(idx, len(documents), doc.filename)

    metrics = compute_batch_metrics(results, risk_weights)

    batch = BatchResult(
        batch_id=str(uuid.uuid4()),
        name=name,
        timestamp=datetime.now(timezone.utc),
        directory=str(directory),
        documents=results,
        metrics=metrics,
    )

    # Persist to database.
    resolved_db = Path(db_path) if db_path else directory / config.DATABASE_FILENAME
    save_batch(resolved_db, batch)

    return batch
