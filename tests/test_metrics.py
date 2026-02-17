"""Tests for the metrics computation module."""

from __future__ import annotations

import pytest

from redactron.core.metrics import (
    compute_batch_metrics,
    compute_clean_doc_rate,
    compute_entity_redaction_rate,
    compute_weighted_risk_score,
)
from redactron.models.schemas import (
    BatchMetrics,
    Document,
    DocumentResult,
    Finding,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_doc_result(
    *,
    findings: list[Finding] | None = None,
    redaction_box_count: int = 0,
    filename: str = "test.pdf",
) -> DocumentResult:
    """Build a minimal DocumentResult for testing."""
    return DocumentResult(
        document=Document(
            filename=filename,
            file_path=f"/tmp/{filename}",
            file_type=".pdf",
        ),
        findings=findings or [],
        redaction_box_count=redaction_box_count,
        has_text_layer=True,
        status="processed",
    )


def _make_finding(entity_type: str = "US_SSN", confidence: float = 0.85) -> Finding:
    """Build a minimal Finding for testing."""
    return Finding(
        entity_type=entity_type,
        text="[REDACTED]",
        page=0,
        confidence=confidence,
        source="visible_pii_missed",
    )


# ---------------------------------------------------------------------------
# Tests for compute_clean_doc_rate
# ---------------------------------------------------------------------------


class TestCleanDocRate:
    def test_known_ratio(self):
        # 8 clean docs out of 10
        docs = [_make_doc_result() for _ in range(8)]  # 8 clean
        docs += [
            _make_doc_result(findings=[_make_finding()]) for _ in range(2)
        ]  # 2 dirty
        rate = compute_clean_doc_rate(docs)
        assert rate == pytest.approx(0.8)

    def test_all_clean(self):
        docs = [_make_doc_result() for _ in range(5)]
        rate = compute_clean_doc_rate(docs)
        assert rate == pytest.approx(1.0)

    def test_all_dirty(self):
        docs = [_make_doc_result(findings=[_make_finding()]) for _ in range(5)]
        rate = compute_clean_doc_rate(docs)
        assert rate == pytest.approx(0.0)

    def test_empty_returns_one(self):
        rate = compute_clean_doc_rate([])
        assert rate == pytest.approx(1.0)

    def test_single_clean_doc(self):
        docs = [_make_doc_result()]
        rate = compute_clean_doc_rate(docs)
        assert rate == pytest.approx(1.0)

    def test_single_dirty_doc(self):
        docs = [_make_doc_result(findings=[_make_finding()])]
        rate = compute_clean_doc_rate(docs)
        assert rate == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests for compute_entity_redaction_rate
# ---------------------------------------------------------------------------


class TestEntityRedactionRate:
    def test_all_redacted_no_missed(self):
        docs = [_make_doc_result(redaction_box_count=10)]
        rate = compute_entity_redaction_rate(docs)
        assert rate == pytest.approx(1.0)

    def test_all_missed_no_redactions(self):
        docs = [_make_doc_result(findings=[_make_finding()] * 5)]
        rate = compute_entity_redaction_rate(docs)
        assert rate == pytest.approx(0.0)

    def test_mixed_redacted_and_missed(self):
        # 6 redacted + 4 missed = 6/10 = 0.6
        docs = [
            _make_doc_result(
                findings=[_make_finding()] * 4,
                redaction_box_count=6,
            )
        ]
        rate = compute_entity_redaction_rate(docs)
        assert rate == pytest.approx(0.6)

    def test_zero_total_returns_one(self):
        docs = [_make_doc_result()]
        rate = compute_entity_redaction_rate(docs)
        assert rate == pytest.approx(1.0)

    def test_empty_docs_returns_one(self):
        rate = compute_entity_redaction_rate([])
        assert rate == pytest.approx(1.0)

    def test_multiple_documents(self):
        # Doc1: 3 boxes, 2 missed; Doc2: 7 boxes, 3 missed
        # total redacted = 10, total missed = 5, rate = 10/15
        docs = [
            _make_doc_result(
                findings=[_make_finding()] * 2,
                redaction_box_count=3,
            ),
            _make_doc_result(
                findings=[_make_finding()] * 3,
                redaction_box_count=7,
            ),
        ]
        rate = compute_entity_redaction_rate(docs)
        assert rate == pytest.approx(10 / 15)


# ---------------------------------------------------------------------------
# Tests for compute_weighted_risk_score
# ---------------------------------------------------------------------------


class TestWeightedRiskScore:
    def test_no_findings_no_boxes_returns_one(self):
        docs = [_make_doc_result()]
        score = compute_weighted_risk_score(docs)
        assert score == pytest.approx(1.0)

    def test_boxes_only_no_missed_returns_one(self):
        docs = [_make_doc_result(redaction_box_count=10)]
        score = compute_weighted_risk_score(docs)
        assert score == pytest.approx(1.0)

    def test_all_missed_no_boxes(self):
        # All missed, no boxes: weighted_missed / weighted_total = 1.0, score = 0.0
        docs = [_make_doc_result(findings=[_make_finding("US_SSN")] * 5)]
        score = compute_weighted_risk_score(docs, weights={"US_SSN": 10})
        assert score == pytest.approx(0.0)

    def test_specific_weighted_calculation(self):
        # 2 SSN missed, 8 boxes total
        # Only one entity type: SSN (weight=10)
        # boxes_for_type = 8 * (2/2) = 8
        # total_for_type = 8 + 2 = 10
        # weighted_missed = 10 * 2 = 20
        # weighted_total = 10 * 10 = 100
        # score = 1 - 20/100 = 0.8
        docs = [
            _make_doc_result(
                findings=[_make_finding("US_SSN")] * 2,
                redaction_box_count=8,
            )
        ]
        score = compute_weighted_risk_score(docs, weights={"US_SSN": 10})
        assert score == pytest.approx(0.8)

    def test_multiple_entity_types(self):
        # 1 SSN missed (weight=10), 1 PERSON missed (weight=7)
        # total_missed = 2, total_boxes = 6
        # SSN: boxes_for_type = 6*(1/2)=3, total_for_type=3+1=4
        #   weighted_missed = 10*1=10, weighted_total = 10*4=40
        # PERSON: boxes_for_type = 6*(1/2)=3, total_for_type=3+1=4
        #   weighted_missed = 7*1=7, weighted_total = 7*4=28
        # score = 1 - (10+7)/(40+28) = 1 - 17/68 = 0.75
        docs = [
            _make_doc_result(
                findings=[
                    _make_finding("US_SSN"),
                    _make_finding("PERSON"),
                ],
                redaction_box_count=6,
            )
        ]
        score = compute_weighted_risk_score(
            docs, weights={"US_SSN": 10, "PERSON": 7}
        )
        assert score == pytest.approx(0.75)

    def test_empty_docs_returns_one(self):
        score = compute_weighted_risk_score([])
        assert score == pytest.approx(1.0)

    def test_unknown_entity_type_uses_default_weight(self):
        from redactron import config

        docs = [
            _make_doc_result(
                findings=[_make_finding("UNKNOWN_TYPE")] * 3,
                redaction_box_count=7,
            )
        ]
        score = compute_weighted_risk_score(docs, weights={})
        # UNKNOWN_TYPE -> DEFAULT_ENTITY_WEIGHT = 3
        # boxes_for_type = 7, total_for_type = 7+3 = 10
        # weighted_missed = 3*3=9, weighted_total = 3*10=30
        # score = 1 - 9/30 = 0.7
        assert score == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Tests for compute_batch_metrics (aggregate)
# ---------------------------------------------------------------------------


class TestComputeBatchMetrics:
    def test_returns_batch_metrics_model(self):
        docs = [_make_doc_result()]
        metrics = compute_batch_metrics(docs)
        assert isinstance(metrics, BatchMetrics)

    def test_total_documents_count(self):
        docs = [_make_doc_result() for _ in range(4)]
        metrics = compute_batch_metrics(docs)
        assert metrics.total_documents == 4

    def test_total_findings_count(self):
        docs = [
            _make_doc_result(findings=[_make_finding()] * 3),
            _make_doc_result(findings=[_make_finding()] * 2),
        ]
        metrics = compute_batch_metrics(docs)
        assert metrics.total_findings == 5

    def test_findings_by_type(self):
        docs = [
            _make_doc_result(
                findings=[
                    _make_finding("US_SSN"),
                    _make_finding("US_SSN"),
                    _make_finding("PERSON"),
                ]
            )
        ]
        metrics = compute_batch_metrics(docs)
        assert metrics.findings_by_type == {"US_SSN": 2, "PERSON": 1}

    def test_empty_batch(self):
        metrics = compute_batch_metrics([])
        assert metrics.total_documents == 0
        assert metrics.total_findings == 0
        assert metrics.clean_doc_rate == pytest.approx(1.0)
        assert metrics.entity_redaction_rate == pytest.approx(1.0)
        assert metrics.weighted_risk_score == pytest.approx(1.0)
        assert metrics.findings_by_type == {}

    def test_all_clean_batch(self):
        docs = [_make_doc_result(redaction_box_count=5) for _ in range(10)]
        metrics = compute_batch_metrics(docs)
        assert metrics.clean_doc_rate == pytest.approx(1.0)
        assert metrics.total_findings == 0
