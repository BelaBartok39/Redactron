"""Redactron configuration: risk weights, thresholds, PII type registry."""

from __future__ import annotations

# Supported file extensions for document ingestion
SUPPORTED_EXTENSIONS: set[str] = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}

# Default confidence threshold — findings below this score are discarded
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.35

# Default risk weights per PII entity type (used in weighted risk score)
DEFAULT_RISK_WEIGHTS: dict[str, int] = {
    "US_SSN": 10,
    "PERSON": 7,
    "DATE_OF_BIRTH": 6,
    "CASE_NUMBER": 5,
    "DOCKET_NUMBER": 5,
    "US_DRIVER_LICENSE": 5,
    "ADDRESS": 4,
    "PHONE_NUMBER": 3,
    "EMAIL_ADDRESS": 3,
    "US_PASSPORT": 8,
    "US_BANK_NUMBER": 8,
    "CREDIT_CARD": 9,
    "IP_ADDRESS": 2,
    "URL": 1,
}

# Fallback weight for entity types not listed above
DEFAULT_ENTITY_WEIGHT: int = 3

# PaddleOCR configuration
OCR_LANG: str = "en"
OCR_USE_ANGLE_CLS: bool = True
OCR_DET_DB_THRESH: float = 0.3

# PDF rendering DPI for OCR (higher = better OCR accuracy, slower processing)
PDF_RENDER_DPI: int = 200

# Maximum parallel workers (0 = use all available CPU cores)
MAX_WORKERS: int = 0

# SQLite database filename (stored alongside batch output)
DATABASE_FILENAME: str = "redactron.db"
