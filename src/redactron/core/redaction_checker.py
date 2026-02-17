"""Per-document redaction verification orchestrator."""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

from redactron import config
from redactron.core.analyzer import analyze_text
from redactron.core.ingestion import is_image, is_pdf, load_document
from redactron.core.ocr_engine import OCREngine
from redactron.core.pdf_extractor import (
    PDFExtractionResult,
    extract_pdf,
    render_page_to_image,
)
from redactron.models.schemas import Document, DocumentResult, Finding

logger = logging.getLogger(__name__)


def _ocr_image(
    image: Image.Image,
    page: int,
    mask_regions: list[tuple[int, int, int, int]] | None = None,
    *,
    score_threshold: float = config.DEFAULT_CONFIDENCE_THRESHOLD,
) -> list[Finding]:
    """Run OCR on *image*, then analyse the extracted text for PII."""
    engine = OCREngine()
    ocr_results = engine.run(image, mask_regions=mask_regions)
    combined_text = " ".join(r.text for r in ocr_results)
    return analyze_text(
        combined_text,
        page=page,
        source="visible_pii_missed",
        score_threshold=score_threshold,
    )


def _redaction_boxes_as_mask(
    pdf_result: PDFExtractionResult,
    page_index: int,
) -> list[tuple[int, int, int, int]]:
    """Convert redaction box rects for a page into pixel mask regions.

    The coordinates from PyMuPDF are in PDF points (1/72 inch).  We need
    to scale them to match the rendered image at ``config.PDF_RENDER_DPI``.
    """
    scale = config.PDF_RENDER_DPI / 72.0
    masks: list[tuple[int, int, int, int]] = []
    page_data = pdf_result.pages[page_index]
    for box in page_data.redaction_boxes:
        r = box.rect
        masks.append((
            int(r.x0 * scale),
            int(r.y0 * scale),
            int(r.x1 * scale),
            int(r.y1 * scale),
        ))
    return masks


def _process_pdf(
    file_path: Path,
    score_threshold: float,
) -> tuple[list[Finding], PDFExtractionResult]:
    """Extract and analyse a PDF document, returning findings and extraction data."""
    pdf_result = extract_pdf(file_path)
    findings: list[Finding] = []

    for page_data in pdf_result.pages:
        page_idx = page_data.page_index

        # 1. Hidden text under redaction boxes — critical findings
        for hidden_text in page_data.hidden_texts:
            findings.extend(
                analyze_text(
                    hidden_text,
                    page=page_idx,
                    source="hidden_text_under_redaction",
                    score_threshold=score_threshold,
                )
            )

        # 2. Visible text from the text layer
        if page_data.visible_text:
            findings.extend(
                analyze_text(
                    page_data.visible_text,
                    page=page_idx,
                    source="visible_pii_missed",
                    score_threshold=score_threshold,
                )
            )

        # 3. OCR for pages without a text layer (scanned pages)
        if not page_data.has_text_layer:
            mask_regions = _redaction_boxes_as_mask(pdf_result, page_idx)
            page_image = render_page_to_image(file_path, page_idx)
            findings.extend(
                _ocr_image(page_image, page_idx, mask_regions, score_threshold=score_threshold)
            )

    return findings, pdf_result


def _process_image(file_path: Path, score_threshold: float) -> list[Finding]:
    """OCR a standalone image file and analyse for PII."""
    image = Image.open(file_path).convert("RGB")
    return _ocr_image(image, page=0, score_threshold=score_threshold)


def check_document(
    document_or_path: str | Path | Document,
    confidence_threshold: float = config.DEFAULT_CONFIDENCE_THRESHOLD,
) -> DocumentResult:
    """Run the full redaction-verification pipeline on a single document.

    Parameters
    ----------
    document_or_path:
        Either a file path (``str`` / ``Path``) or a pre-built
        :class:`Document` model instance.
    confidence_threshold:
        Minimum Presidio confidence score for findings.

    Pipeline steps:
    1. Load and validate the document.
    2. For PDFs — detect redaction boxes, extract hidden text, extract
       visible text, and OCR scanned pages.
    3. For images — OCR the full image.
    4. Run Presidio analysis on all extracted text.
    5. Return a :class:`DocumentResult` with tagged findings.
    """
    if isinstance(document_or_path, Document):
        document = document_or_path
        path = Path(document.file_path).resolve()
    else:
        path = Path(document_or_path).resolve()
        document = load_document(path)

    findings: list[Finding] = []
    redaction_box_count = 0
    has_text_layer = True
    status = "processed"

    try:
        if is_pdf(document.file_type):
            findings, pdf_result = _process_pdf(path, confidence_threshold)
            document.page_count = pdf_result.page_count
            redaction_box_count = pdf_result.redaction_box_count
            has_text_layer = pdf_result.has_text_layer
        elif is_image(document.file_type):
            document.page_count = 1
            findings = _process_image(path, confidence_threshold)
        else:
            status = "error"
            logger.error("Unhandled file type: %s", document.file_type)
    except Exception:
        status = "error"
        logger.exception("Failed to process document: %s", path.name)

    result = DocumentResult(
        document=document,
        findings=findings,
        redaction_box_count=redaction_box_count,
        has_text_layer=has_text_layer,
        status=status,
    )

    logger.info(
        "Document %s: %d findings, %d redaction boxes, status=%s",
        document.filename,
        len(findings),
        redaction_box_count,
        status,
    )
    return result
