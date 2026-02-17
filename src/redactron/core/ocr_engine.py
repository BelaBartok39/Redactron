"""PaddleOCR wrapper: singleton engine with GPU auto-detection.

Supports PaddleOCR v3.x (``predict`` API returning ``OCRResult`` dicts
with ``rec_texts``, ``rec_scores``, ``dt_polys`` keys) as well as the
legacy v2.x ``ocr`` API.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass

import numpy as np
from PIL import Image

from redactron import config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OCRResult:
    """A single OCR detection: recognised text, bounding box, and confidence."""

    text: str
    bbox: list[list[float]]
    confidence: float


class OCREngine:
    """Thread-safe singleton wrapper around PaddleOCR.

    The engine is lazily initialised on first call to :meth:`run`.  GPU
    availability is auto-detected — if CUDA is accessible the engine runs
    on GPU, otherwise it falls back to CPU.

    Compatible with both PaddleOCR v2.x and v3.x APIs.
    """

    _instance: OCREngine | None = None
    _lock = threading.Lock()

    def __new__(cls) -> OCREngine:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    obj = super().__new__(cls)
                    obj._engine = None  # type: ignore[attr-defined]
                    obj._initialised = False  # type: ignore[attr-defined]
                    obj._api_version = 3  # type: ignore[attr-defined]
                    cls._instance = obj
        return cls._instance

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _ensure_initialised(self) -> None:
        if self._initialised:
            return
        with self._lock:
            if self._initialised:
                return
            self._engine = self._create_engine()
            self._initialised = True

    @staticmethod
    def _detect_gpu() -> bool:
        """Return *True* if a CUDA-capable GPU is available to Paddle."""
        try:
            import paddle  # type: ignore[import-untyped]
            return paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
        except Exception:
            return False

    def _create_engine(self):  # noqa: ANN202
        from paddleocr import PaddleOCR  # type: ignore[import-untyped]
        import paddleocr  # type: ignore[import-untyped]

        use_gpu = self._detect_gpu()
        device = "gpu" if use_gpu else "cpu"
        logger.info("Initialising PaddleOCR (GPU=%s, lang=%s)", use_gpu, config.OCR_LANG)

        # Suppress noisy model-source connectivity checks in PaddleOCR v3.x.
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

        major_version = int(getattr(paddleocr, "__version__", "2.0.0").split(".")[0])

        if major_version >= 3:
            # PaddleOCR v3.x API
            self._api_version = 3
            engine = PaddleOCR(
                lang=config.OCR_LANG,
                use_textline_orientation=config.OCR_USE_ANGLE_CLS,
                text_det_thresh=config.OCR_DET_DB_THRESH,
                device=device,
                enable_mkldnn=False,
            )
        else:
            # PaddleOCR v2.x (legacy) API
            self._api_version = 2
            engine = PaddleOCR(
                lang=config.OCR_LANG,
                use_angle_cls=config.OCR_USE_ANGLE_CLS,
                det_db_thresh=config.OCR_DET_DB_THRESH,
                use_gpu=use_gpu,
                show_log=False,
            )
        return engine

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        image: Image.Image,
        mask_regions: list[tuple[int, int, int, int]] | None = None,
    ) -> list[OCRResult]:
        """Run OCR on a PIL *image* and return detected text regions.

        Parameters
        ----------
        image:
            The input page/image as a PIL Image.
        mask_regions:
            Optional list of ``(x0, y0, x1, y1)`` bounding boxes to paint
            black before OCR.  This is used to mask out known redaction
            regions so the engine only reads visible text.

        Returns
        -------
        list[OCRResult]
            Detected text segments with bounding boxes and confidence scores.
        """
        self._ensure_initialised()

        img_array = np.array(image)

        if mask_regions:
            img_array = self._apply_masks(img_array, mask_regions)

        if self._api_version >= 3:
            return self._run_v3(img_array)
        return self._run_v2(img_array)

    def _run_v3(self, img_array: np.ndarray) -> list[OCRResult]:
        """Parse results from PaddleOCR v3.x ``predict`` API."""
        raw = self._engine.predict(img_array)

        results: list[OCRResult] = []
        if not raw:
            return results

        for page_result in raw:
            texts = page_result.get("rec_texts", [])
            scores = page_result.get("rec_scores", [])
            polys = page_result.get("dt_polys", [])
            for text, score, poly in zip(texts, scores, polys):
                bbox = poly.tolist() if hasattr(poly, "tolist") else poly
                results.append(OCRResult(text=text, bbox=bbox, confidence=float(score)))

        return results

    def _run_v2(self, img_array: np.ndarray) -> list[OCRResult]:
        """Parse results from PaddleOCR v2.x ``ocr`` API."""
        raw = self._engine.ocr(img_array, cls=config.OCR_USE_ANGLE_CLS)

        results: list[OCRResult] = []
        if not raw or not raw[0]:
            return results

        for line in raw[0]:
            bbox, (text, conf) = line
            results.append(OCRResult(text=text, bbox=bbox, confidence=conf))

        return results

    @staticmethod
    def _apply_masks(
        img: np.ndarray,
        regions: list[tuple[int, int, int, int]],
    ) -> np.ndarray:
        """Paint black rectangles over *regions* in *img* (in-place copy)."""
        masked = img.copy()
        for x0, y0, x1, y1 in regions:
            masked[y0:y1, x0:x1] = 0
        return masked
