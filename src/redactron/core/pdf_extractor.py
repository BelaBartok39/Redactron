"""PDF processing with PyMuPDF: redaction detection, text extraction, and page rendering."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

from redactron import config

logger = logging.getLogger(__name__)

# Annotation type value for "Redact" in PDF spec
_ANNOT_REDACT = fitz.PDF_ANNOT_REDACT  # type: ignore[attr-defined]

# Threshold (0–1 per channel) below which a fill colour is considered "black"
_BLACK_THRESHOLD = 0.15


@dataclass
class RedactionBox:
    """A rectangular region identified as a redaction on a PDF page."""

    page_index: int
    rect: fitz.Rect


@dataclass
class PageExtractionResult:
    """Text and redaction data extracted from a single PDF page."""

    page_index: int
    visible_text: str = ""
    hidden_texts: list[str] = field(default_factory=list)
    redaction_boxes: list[RedactionBox] = field(default_factory=list)
    has_text_layer: bool = True


@dataclass
class PDFExtractionResult:
    """Aggregated extraction results for an entire PDF document."""

    page_count: int = 0
    pages: list[PageExtractionResult] = field(default_factory=list)
    has_text_layer: bool = True
    needs_ocr: bool = False
    redaction_box_count: int = 0


def _is_black_fill(color: tuple[float, ...] | list[float] | None) -> bool:
    """Return *True* if *color* is close to black (all channels <= threshold)."""
    if color is None:
        return False
    return all(c <= _BLACK_THRESHOLD for c in color)


def _detect_redaction_annotations(page: fitz.Page, page_index: int) -> list[RedactionBox]:
    """Find all annotations that look like redaction marks on *page*.

    This detects both explicit "Redact" annotations **and** opaque
    black-filled rectangle annotations that are commonly used as
    manual redactions.
    """
    boxes: list[RedactionBox] = []
    for annot in page.annots() or []:
        is_redact = annot.type[0] == _ANNOT_REDACT
        is_black_rect = (
            annot.type[0] == fitz.PDF_ANNOT_SQUARE
            and _is_black_fill(annot.colors.get("fill"))
        )
        if is_redact or is_black_rect:
            boxes.append(RedactionBox(page_index=page_index, rect=annot.rect))
    return boxes


def _extract_text_under_boxes(
    page: fitz.Page,
    boxes: list[RedactionBox],
) -> list[str]:
    """Return the text layer content beneath each redaction *box*.

    If the text layer contains content inside a redaction rectangle, that
    content was likely not properly removed — a critical finding.
    """
    hidden: list[str] = []
    for box in boxes:
        text = page.get_text("text", clip=box.rect).strip()
        if text:
            hidden.append(text)
    return hidden


def _extract_visible_text(
    page: fitz.Page,
    boxes: list[RedactionBox],
) -> str:
    """Extract all text from *page* that is **outside** redaction boxes.

    We get the full page text and, for each redaction box, remove any
    text that falls within it (since that text is handled separately as
    hidden text).
    """
    if not boxes:
        return page.get_text("text").strip()

    # Use text blocks for finer-grained spatial filtering
    visible_parts: list[str] = []
    for block in page.get_text("blocks"):
        # block: (x0, y0, x1, y1, text, block_no, block_type)
        if block[6] != 0:  # skip image blocks
            continue
        block_rect = fitz.Rect(block[:4])
        # Keep block only if it does not significantly overlap any redaction box
        overlaps = any(
            block_rect.intersects(b.rect) and _overlap_ratio(block_rect, b.rect) > 0.5
            for b in boxes
        )
        if not overlaps:
            text = block[4].strip() if isinstance(block[4], str) else ""
            if text:
                visible_parts.append(text)
    return "\n".join(visible_parts)


def _overlap_ratio(a: fitz.Rect, b: fitz.Rect) -> float:
    """Return the fraction of *a* that is covered by its intersection with *b*."""
    intersection = a & b  # fitz.Rect intersection
    if intersection.is_empty:
        return 0.0
    a_area = a.width * a.height
    if a_area == 0:
        return 0.0
    return (intersection.width * intersection.height) / a_area


def _page_has_text_layer(page: fitz.Page) -> bool:
    """Return *True* if *page* contains any extractable text."""
    return bool(page.get_text("text").strip())


def extract_pdf(file_path: str | Path) -> PDFExtractionResult:
    """Process a PDF file and return extraction results.

    For each page the function:
    1. Detects redaction annotations / black-filled rectangles.
    2. Extracts text hidden *under* those redaction boxes.
    3. Extracts visible (non-redacted) text.
    4. Checks whether a usable text layer exists.
    """
    path = Path(file_path)
    doc = fitz.open(str(path))
    result = PDFExtractionResult(page_count=len(doc))

    pages_without_text = 0

    for page_index in range(len(doc)):
        page = doc[page_index]

        boxes = _detect_redaction_annotations(page, page_index)
        hidden = _extract_text_under_boxes(page, boxes)
        visible = _extract_visible_text(page, boxes)
        has_text = _page_has_text_layer(page)

        if not has_text:
            pages_without_text += 1

        page_result = PageExtractionResult(
            page_index=page_index,
            visible_text=visible,
            hidden_texts=hidden,
            redaction_boxes=boxes,
            has_text_layer=has_text,
        )
        result.pages.append(page_result)
        result.redaction_box_count += len(boxes)

    doc.close()

    # If the majority of pages lack a text layer, flag for OCR
    if result.page_count > 0 and pages_without_text / result.page_count > 0.5:
        result.has_text_layer = False
        result.needs_ocr = True

    logger.info(
        "PDF %s: %d pages, %d redaction boxes, text_layer=%s",
        path.name,
        result.page_count,
        result.redaction_box_count,
        result.has_text_layer,
    )
    return result


def render_page_to_image(
    file_path: str | Path,
    page_index: int,
    dpi: int = config.PDF_RENDER_DPI,
) -> Image.Image:
    """Render a single PDF page to a PIL Image at the given *dpi*."""
    doc = fitz.open(str(file_path))
    try:
        page = doc[page_index]
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    finally:
        doc.close()
    return img
