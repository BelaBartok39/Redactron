"""Document ingestion: format detection, validation, and loading."""

from __future__ import annotations

import logging
from pathlib import Path

from redactron import config
from redactron.models.schemas import Document

logger = logging.getLogger(__name__)


def detect_file_type(file_path: Path) -> str:
    """Return the lowercase file extension (e.g. '.pdf') for *file_path*.

    Raises ``ValueError`` if the extension is not in
    ``config.SUPPORTED_EXTENSIONS``.
    """
    ext = file_path.suffix.lower()
    if ext not in config.SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {sorted(config.SUPPORTED_EXTENSIONS)}"
        )
    return ext


def is_pdf(file_type: str) -> bool:
    """Return *True* if *file_type* represents a PDF."""
    return file_type == ".pdf"


def is_image(file_type: str) -> bool:
    """Return *True* if *file_type* represents an image format."""
    return file_type in {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}


def load_document(file_path: str | Path) -> Document:
    """Validate and wrap *file_path* in a :class:`Document` model.

    The function verifies that the file exists and has a supported extension,
    then constructs a ``Document`` instance.  Page-count is set to ``0`` at
    this stage — downstream processors (PDF extractor / OCR) are responsible
    for populating the real count.
    """
    path = Path(file_path).resolve()

    if not path.is_file():
        raise FileNotFoundError(f"Document not found: {path}")

    file_type = detect_file_type(path)

    doc = Document(
        filename=path.name,
        file_path=path,
        file_type=file_type,
        page_count=0,
    )
    logger.info("Loaded document %s (type=%s)", doc.filename, doc.file_type)
    return doc
