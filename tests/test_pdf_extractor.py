"""Tests for the PDF extractor module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import fitz
import pytest

from redactron.core.pdf_extractor import (
    PDFExtractionResult,
    PageExtractionResult,
    RedactionBox,
    _detect_redaction_annotations,
    _extract_text_under_boxes,
    _extract_visible_text,
    _is_black_fill,
    _overlap_ratio,
    _page_has_text_layer,
    extract_pdf,
    render_page_to_image,
)


# ---------------------------------------------------------------------------
# Helpers for creating synthetic PDFs
# ---------------------------------------------------------------------------


def _create_pdf_with_text(text: str, *, page_count: int = 1) -> Path:
    """Create a temporary PDF with *text* inserted on each page."""
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    doc = fitz.open()
    for _ in range(page_count):
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=12)
    doc.save(tmp.name)
    doc.close()
    return Path(tmp.name)


def _create_pdf_with_redaction_annotation(
    visible_text: str,
    hidden_text: str,
) -> Path:
    """Create a PDF that has text under a Redact annotation.

    The *hidden_text* is placed at a known position, then a Redact annotation
    is drawn on top of it (without applying the redaction, so text remains).
    The *visible_text* is placed elsewhere on the page.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    doc = fitz.open()
    page = doc.new_page()

    # Place hidden text in the upper-left area
    page.insert_text((72, 72), hidden_text, fontsize=12)

    # Place visible text lower on the page
    page.insert_text((72, 300), visible_text, fontsize=12)

    # Add a Redact annotation covering the hidden text area
    # The rect should cover the area where hidden_text was inserted
    redact_rect = fitz.Rect(60, 55, 400, 85)
    annot = page.add_redact_annot(redact_rect)

    doc.save(tmp.name)
    doc.close()
    return Path(tmp.name)


def _create_pdf_with_black_rect_annotation(
    visible_text: str,
    hidden_text: str,
) -> Path:
    """Create a PDF with text under a black-filled Square annotation (manual redaction)."""
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    doc = fitz.open()
    page = doc.new_page()

    # Place hidden text
    page.insert_text((72, 72), hidden_text, fontsize=12)

    # Place visible text lower
    page.insert_text((72, 300), visible_text, fontsize=12)

    # Add a black-filled Square annotation over the hidden text area
    rect = fitz.Rect(60, 55, 400, 85)
    annot = page.add_rect_annot(rect)
    annot.set_colors(fill=(0, 0, 0))
    annot.update()

    doc.save(tmp.name)
    doc.close()
    return Path(tmp.name)


def _create_blank_pdf(page_count: int = 1) -> Path:
    """Create a PDF with blank pages (no text layer)."""
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    doc = fitz.open()
    for _ in range(page_count):
        doc.new_page()
    doc.save(tmp.name)
    doc.close()
    return Path(tmp.name)


# ---------------------------------------------------------------------------
# Tests for _is_black_fill
# ---------------------------------------------------------------------------


class TestIsBlackFill:
    def test_black_color(self):
        assert _is_black_fill((0, 0, 0)) is True

    def test_near_black_color(self):
        assert _is_black_fill((0.1, 0.1, 0.1)) is True

    def test_not_black(self):
        assert _is_black_fill((1.0, 1.0, 1.0)) is False

    def test_none_color(self):
        assert _is_black_fill(None) is False

    def test_partially_black(self):
        assert _is_black_fill((0, 0, 0.5)) is False


# ---------------------------------------------------------------------------
# Tests for _overlap_ratio
# ---------------------------------------------------------------------------


class TestOverlapRatio:
    def test_no_overlap(self):
        a = fitz.Rect(0, 0, 10, 10)
        b = fitz.Rect(20, 20, 30, 30)
        assert _overlap_ratio(a, b) == 0.0

    def test_full_overlap(self):
        a = fitz.Rect(0, 0, 10, 10)
        b = fitz.Rect(0, 0, 10, 10)
        assert _overlap_ratio(a, b) == pytest.approx(1.0)

    def test_partial_overlap(self):
        a = fitz.Rect(0, 0, 10, 10)
        b = fitz.Rect(5, 5, 15, 15)
        # Intersection is 5x5 = 25, a area is 100, ratio = 0.25
        assert _overlap_ratio(a, b) == pytest.approx(0.25)

    def test_zero_area_rect(self):
        a = fitz.Rect(0, 0, 0, 10)  # zero width
        b = fitz.Rect(0, 0, 10, 10)
        assert _overlap_ratio(a, b) == 0.0


# ---------------------------------------------------------------------------
# Tests for extract_pdf — redaction annotation detection
# ---------------------------------------------------------------------------


class TestDetectRedactionAnnotations:
    def test_detects_redact_annotation(self):
        pdf_path = _create_pdf_with_redaction_annotation(
            visible_text="This is visible.",
            hidden_text="SECRET SSN 123-45-6789",
        )
        result = extract_pdf(pdf_path)
        assert result.redaction_box_count >= 1
        assert len(result.pages) == 1
        assert len(result.pages[0].redaction_boxes) >= 1

    def test_detects_black_rect_annotation(self):
        pdf_path = _create_pdf_with_black_rect_annotation(
            visible_text="Public text.",
            hidden_text="HIDDEN NAME John Doe",
        )
        result = extract_pdf(pdf_path)
        assert result.redaction_box_count >= 1

    def test_no_redactions(self):
        pdf_path = _create_pdf_with_text("Plain document text, no redactions here.")
        result = extract_pdf(pdf_path)
        assert result.redaction_box_count == 0
        for page in result.pages:
            assert len(page.redaction_boxes) == 0


# ---------------------------------------------------------------------------
# Tests for text extraction
# ---------------------------------------------------------------------------


class TestTextExtraction:
    def test_extract_hidden_text_under_redaction(self):
        pdf_path = _create_pdf_with_redaction_annotation(
            visible_text="Visible content here.",
            hidden_text="SSN 123-45-6789",
        )
        result = extract_pdf(pdf_path)
        hidden_texts = result.pages[0].hidden_texts
        # The hidden text should contain the SSN we planted
        combined_hidden = " ".join(hidden_texts)
        assert "123-45-6789" in combined_hidden

    def test_visible_text_excludes_redacted_area(self):
        pdf_path = _create_pdf_with_redaction_annotation(
            visible_text="Public paragraph.",
            hidden_text="REDACTED CONTENT",
        )
        result = extract_pdf(pdf_path)
        visible = result.pages[0].visible_text
        # Visible text should contain the public paragraph
        assert "Public paragraph" in visible

    def test_no_redaction_all_text_visible(self):
        text = "This is a normal document with no redactions."
        pdf_path = _create_pdf_with_text(text)
        result = extract_pdf(pdf_path)
        assert result.pages[0].visible_text.strip() != ""
        assert len(result.pages[0].hidden_texts) == 0


# ---------------------------------------------------------------------------
# Tests for text layer detection
# ---------------------------------------------------------------------------


class TestTextLayerDetection:
    def test_has_text_layer(self):
        pdf_path = _create_pdf_with_text("Some text content.")
        result = extract_pdf(pdf_path)
        assert result.has_text_layer is True
        assert result.needs_ocr is False

    def test_no_text_layer_flags_ocr(self):
        pdf_path = _create_blank_pdf(page_count=2)
        result = extract_pdf(pdf_path)
        assert result.has_text_layer is False
        assert result.needs_ocr is True


# ---------------------------------------------------------------------------
# Tests for multi-page PDFs
# ---------------------------------------------------------------------------


class TestMultiPage:
    def test_multi_page_extraction(self):
        pdf_path = _create_pdf_with_text("Page content", page_count=3)
        result = extract_pdf(pdf_path)
        assert result.page_count == 3
        assert len(result.pages) == 3
        for i, page in enumerate(result.pages):
            assert page.page_index == i
            assert page.has_text_layer is True


# ---------------------------------------------------------------------------
# Tests for render_page_to_image
# ---------------------------------------------------------------------------


class TestRenderPageToImage:
    def test_render_returns_pil_image(self):
        from PIL import Image

        pdf_path = _create_pdf_with_text("Render test text.")
        img = render_page_to_image(pdf_path, page_index=0, dpi=72)
        assert isinstance(img, Image.Image)
        assert img.width > 0
        assert img.height > 0

    def test_higher_dpi_gives_larger_image(self):
        pdf_path = _create_pdf_with_text("DPI test.")
        img_low = render_page_to_image(pdf_path, page_index=0, dpi=72)
        img_high = render_page_to_image(pdf_path, page_index=0, dpi=144)
        assert img_high.width > img_low.width
        assert img_high.height > img_low.height


# ---------------------------------------------------------------------------
# Tests for PDFExtractionResult structure
# ---------------------------------------------------------------------------


class TestPDFExtractionResultStructure:
    def test_result_fields(self):
        pdf_path = _create_pdf_with_text("Structure test.")
        result = extract_pdf(pdf_path)
        assert isinstance(result, PDFExtractionResult)
        assert result.page_count == 1
        assert isinstance(result.pages, list)
        assert isinstance(result.pages[0], PageExtractionResult)
        assert isinstance(result.has_text_layer, bool)
        assert isinstance(result.needs_ocr, bool)
        assert isinstance(result.redaction_box_count, int)
