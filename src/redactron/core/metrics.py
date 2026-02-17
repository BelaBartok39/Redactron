"""Metrics engine: clean-doc rate, entity redaction rate, weighted risk score."""

from __future__ import annotations

from redactron import config
from redactron.models.schemas import BatchMetrics, DocumentResult


def compute_clean_doc_rate(documents: list[DocumentResult]) -> float:
    """Fraction of documents with zero findings.

    Returns 1.0 when *documents* is empty (no docs means nothing is dirty).
    """
    if not documents:
        return 1.0
    clean = sum(1 for d in documents if len(d.findings) == 0)
    return clean / len(documents)


def compute_entity_redaction_rate(documents: list[DocumentResult]) -> float:
    """Estimated redaction coverage: redacted / (redacted + missed).

    *redacted_entities* is approximated by the total redaction-box count across
    all documents.  *missed_entities* is the total finding count (each finding
    represents a PII instance that was NOT successfully redacted).

    Returns 1.0 when both counts are zero.
    """
    redacted = sum(d.redaction_box_count for d in documents)
    missed = sum(len(d.findings) for d in documents)
    total = redacted + missed
    if total == 0:
        return 1.0
    return redacted / total


def compute_weighted_risk_score(
    documents: list[DocumentResult],
    weights: dict[str, int] | None = None,
) -> float:
    """Weighted risk score: 1 - sum(w_i * missed_i) / sum(w_i * total_i).

    For each entity type *i*:
      - missed_i  = number of findings of that type
      - total_i   = redaction_boxes (estimated) + missed_i
      - w_i       = weight from *weights* (or DEFAULT_ENTITY_WEIGHT)

    Redaction boxes are distributed proportionally across entity types based on
    the missed-finding distribution.  When there are no missed findings the
    boxes are ignored (score = 1.0).

    Returns 1.0 when there are zero entities overall.
    """
    if weights is None:
        weights = config.DEFAULT_RISK_WEIGHTS

    # Collect per-type missed counts and total redaction boxes.
    missed_by_type: dict[str, int] = {}
    total_boxes = 0
    for doc in documents:
        total_boxes += doc.redaction_box_count
        for f in doc.findings:
            missed_by_type[f.entity_type] = missed_by_type.get(f.entity_type, 0) + 1

    total_missed = sum(missed_by_type.values())
    if total_missed == 0 and total_boxes == 0:
        return 1.0
    if total_missed == 0:
        # All boxes present, nothing missed — perfect score.
        return 1.0

    # Distribute redaction boxes proportionally across entity types.
    weighted_missed = 0.0
    weighted_total = 0.0
    for etype, missed_count in missed_by_type.items():
        w = weights.get(etype, config.DEFAULT_ENTITY_WEIGHT)
        # Proportional share of boxes for this entity type.
        boxes_for_type = total_boxes * (missed_count / total_missed)
        total_for_type = boxes_for_type + missed_count
        weighted_missed += w * missed_count
        weighted_total += w * total_for_type

    if weighted_total == 0:
        return 1.0
    return 1.0 - (weighted_missed / weighted_total)


def compute_batch_metrics(
    documents: list[DocumentResult],
    weights: dict[str, int] | None = None,
) -> BatchMetrics:
    """Compute all batch-level metrics and summary statistics."""
    findings_by_type: dict[str, int] = {}
    total_findings = 0
    for doc in documents:
        for f in doc.findings:
            findings_by_type[f.entity_type] = findings_by_type.get(f.entity_type, 0) + 1
            total_findings += 1

    return BatchMetrics(
        clean_doc_rate=compute_clean_doc_rate(documents),
        entity_redaction_rate=compute_entity_redaction_rate(documents),
        weighted_risk_score=compute_weighted_risk_score(documents, weights),
        total_documents=len(documents),
        total_findings=total_findings,
        findings_by_type=findings_by_type,
    )
