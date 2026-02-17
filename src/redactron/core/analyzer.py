"""Presidio-based PII analysis with custom legal recognizers."""

from __future__ import annotations

import logging

from presidio_analyzer import AnalyzerEngine, RecognizerResult

from redactron import config
from redactron.models.schemas import Finding
from redactron.recognizers.legal import get_legal_recognizers

logger = logging.getLogger(__name__)

_analyzer: AnalyzerEngine | None = None


def _get_analyzer() -> AnalyzerEngine:
    """Return a lazily-initialised, module-level :class:`AnalyzerEngine`."""
    global _analyzer  # noqa: PLW0603
    if _analyzer is not None:
        return _analyzer

    engine = AnalyzerEngine()

    for recognizer in get_legal_recognizers():
        engine.registry.add_recognizer(recognizer)
        logger.info("Registered recognizer: %s", recognizer.supported_entities)

    _analyzer = engine
    return _analyzer


def analyze_text(
    text: str,
    *,
    language: str = "en",
    page: int = 0,
    source: str = "visible_pii_missed",
    score_threshold: float = config.DEFAULT_CONFIDENCE_THRESHOLD,
) -> list[Finding]:
    """Run Presidio analysis on *text* and return :class:`Finding` objects.

    Parameters
    ----------
    text:
        The input text to scan for PII.
    language:
        Language code for the analyzer (default ``"en"``).
    page:
        The page number to attach to each finding.
    source:
        Finding source tag — typically ``"hidden_text_under_redaction"`` or
        ``"visible_pii_missed"``.
    score_threshold:
        Minimum confidence score; findings below this are discarded.
    """
    if not text.strip():
        return []

    engine = _get_analyzer()
    results: list[RecognizerResult] = engine.analyze(
        text=text,
        language=language,
        score_threshold=score_threshold,
    )

    findings: list[Finding] = []
    for r in results:
        matched_text = text[r.start : r.end]
        findings.append(
            Finding(
                entity_type=r.entity_type,
                text=matched_text,
                page=page,
                confidence=round(r.score, 4),
                source=source,
                start=r.start,
                end=r.end,
            )
        )

    logger.debug(
        "Page %d (%s): %d findings above threshold %.2f",
        page,
        source,
        len(findings),
        score_threshold,
    )
    return findings
