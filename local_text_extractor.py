from __future__ import annotations

import os
import re
import threading
import unicodedata
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, Callable, Iterable


DEFAULT_MAX_PAGES = 3
DEFAULT_OCR_DPI = 300
SUPPORTED_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})


class TextEvidenceSource(str, Enum):
    XML = "xml"
    PDF_TEXT = "pdf_text"
    RAPIDOCR = "rapidocr"


class TextAcquisitionStatus(str, Enum):
    ACQUIRED = "acquired"
    INSUFFICIENT = "insufficient"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"


def normalize_document_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    lines = [re.sub(r"[\t\f\v ]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def is_usable_native_pdf_text(value: object) -> bool:
    text = normalize_document_text(value)
    compact = re.sub(r"\s+", "", text)
    if len(compact) < 40:
        return False
    meaningful = re.findall(r"[\u3400-\u9fffA-Za-z0-9]", compact)
    if len(meaningful) / max(1, len(compact)) < 0.6:
        return False
    lowered = text.lower()
    invoice_markers = (
        "发票",
        "开票日期",
        "价税合计",
        "购买方",
        "销售方",
        "invoice",
        "invoice number",
        "purchaser",
        "seller",
        "folio",
        "arrival",
        "departure",
        "room charge",
    )
    return len(compact) >= 120 or any(marker in lowered for marker in invoice_markers)


def _confidence(value: object) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")
    if not parsed.is_finite():
        return Decimal("0")
    return min(Decimal("1"), max(Decimal("0"), parsed))


def _bounding_box(value: object) -> tuple[tuple[float, float], ...]:
    points = []
    try:
        for point in value or ():
            if len(point) < 2:
                continue
            points.append((float(point[0]), float(point[1])))
    except (TypeError, ValueError, OverflowError):
        return ()
    return tuple(points)


@dataclass(frozen=True)
class OcrSpan:
    page_index: int
    text: str
    bounding_box: tuple[tuple[float, float], ...] = field(default_factory=tuple)
    confidence: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        object.__setattr__(self, "page_index", max(0, int(self.page_index)))
        object.__setattr__(self, "text", normalize_document_text(self.text))
        object.__setattr__(self, "bounding_box", _bounding_box(self.bounding_box))
        object.__setattr__(self, "confidence", _confidence(self.confidence))


@dataclass(frozen=True)
class TextEvidence:
    source: TextEvidenceSource
    text: str
    spans: tuple[OcrSpan, ...] = field(default_factory=tuple)
    page_count: int = 1
    average_confidence: Decimal | None = None

    def __post_init__(self) -> None:
        spans = tuple(self.spans or ())
        confidence = self.average_confidence
        if confidence is None and spans:
            confidence = sum(
                (span.confidence for span in spans), start=Decimal("0")
            ) / Decimal(len(spans))
        object.__setattr__(self, "text", normalize_document_text(self.text))
        object.__setattr__(self, "spans", spans)
        object.__setattr__(self, "page_count", max(1, int(self.page_count)))
        object.__setattr__(
            self,
            "average_confidence",
            _confidence(confidence) if confidence is not None else None,
        )


@dataclass(frozen=True)
class TextAcquisitionResult:
    status: TextAcquisitionStatus
    evidence: TextEvidence | None = None
    reason_code: str = ""

    def __post_init__(self) -> None:
        if self.status is TextAcquisitionStatus.ACQUIRED and self.evidence is None:
            raise ValueError("acquired text evidence is required")
        if self.evidence is not None and not self.evidence.text:
            raise ValueError("text evidence cannot be empty")

    @classmethod
    def acquired(cls, evidence: TextEvidence) -> "TextAcquisitionResult":
        return cls(TextAcquisitionStatus.ACQUIRED, evidence=evidence)

    @classmethod
    def terminal(
        cls, status: TextAcquisitionStatus, reason_code: str
    ) -> "TextAcquisitionResult":
        return cls(status, reason_code=reason_code)


class RapidOcrUnavailable(RuntimeError):
    reason_code = "RAPIDOCR_UNAVAILABLE"


class RapidOcrEngine:
    """Lazily load one RapidOCR instance for the owning acquisition pipeline."""

    def __init__(self, factory: Callable[[], Any] | None = None) -> None:
        self._factory = factory
        self._instance = None
        self._lock = threading.Lock()

    def _load(self):
        if self._instance is not None:
            return self._instance
        with self._lock:
            if self._instance is not None:
                return self._instance
            os.environ["ORT_DISABLE_TELEMETRY"] = "1"
            factory = self._factory
            if factory is None:
                try:
                    import onnxruntime
                    from rapidocr_onnxruntime import RapidOCR
                except ImportError as exc:
                    raise RapidOcrUnavailable(self.reason_code) from exc
                disable_telemetry = getattr(
                    onnxruntime, "disable_telemetry_events", None
                )
                if callable(disable_telemetry):
                    disable_telemetry()
                factory = RapidOCR
            self._instance = factory()
            return self._instance

    def recognize(self, source: object) -> Iterable[object]:
        response = self._load()(source)
        if isinstance(response, tuple):
            return response[0] or ()
        return response or ()

    @property
    def reason_code(self) -> str:
        return RapidOcrUnavailable.reason_code


class LocalTextExtractor:
    """Acquire local text without making any network request."""

    def __init__(
        self,
        *,
        ocr_engine: Any | None = None,
        max_pages: int = DEFAULT_MAX_PAGES,
        ocr_dpi: int = DEFAULT_OCR_DPI,
        native_text_validator: Callable[[object], bool] = is_usable_native_pdf_text,
    ) -> None:
        self.ocr_engine = ocr_engine or RapidOcrEngine()
        self.max_pages = max(1, int(max_pages))
        self.ocr_dpi = max(72, int(ocr_dpi))
        self.native_text_validator = native_text_validator

    @staticmethod
    def _spans(rows: Iterable[object], page_index: int) -> tuple[OcrSpan, ...]:
        spans = []
        for row in rows or ():
            try:
                box, text, score = row[0], row[1], row[2]
            except (IndexError, KeyError, TypeError):
                continue
            normalized = normalize_document_text(text)
            if not normalized:
                continue
            spans.append(
                OcrSpan(
                    page_index=page_index,
                    text=normalized,
                    bounding_box=_bounding_box(box),
                    confidence=_confidence(score),
                )
            )
        return tuple(spans)

    def _ocr_evidence(
        self, page_sources: Iterable[object], *, page_count: int
    ) -> TextAcquisitionResult:
        spans = []
        page_texts = []
        for page_index, source in enumerate(page_sources):
            page_spans = self._spans(
                self.ocr_engine.recognize(source), page_index=page_index
            )
            spans.extend(page_spans)
            page_text = "\n".join(span.text for span in page_spans)
            if page_text:
                page_texts.append(page_text)
        text = "\n\n".join(page_texts)
        if not text:
            return TextAcquisitionResult.terminal(
                TextAcquisitionStatus.INSUFFICIENT, "RAPIDOCR_EMPTY"
            )
        return TextAcquisitionResult.acquired(
            TextEvidence(
                source=TextEvidenceSource.RAPIDOCR,
                text=text,
                spans=tuple(spans),
                page_count=page_count,
            )
        )

    def _acquire_pdf(self, path: str) -> TextAcquisitionResult:
        try:
            import fitz

            with fitz.open(path) as document:
                page_count = min(len(document), self.max_pages)
                native_parts = [
                    document.load_page(index).get_text("text") or ""
                    for index in range(page_count)
                ]
                native_text = normalize_document_text("\n".join(native_parts))
                if self.native_text_validator(native_text):
                    return TextAcquisitionResult.acquired(
                        TextEvidence(
                            source=TextEvidenceSource.PDF_TEXT,
                            text=native_text,
                            page_count=max(1, page_count),
                        )
                    )
                page_images = [
                    document.load_page(index)
                    .get_pixmap(dpi=self.ocr_dpi, alpha=False)
                    .tobytes("png")
                    for index in range(page_count)
                ]
        except Exception:
            return TextAcquisitionResult.terminal(
                TextAcquisitionStatus.FAILED, "PDF_TEXT_ACQUISITION_FAILED"
            )
        if not page_images:
            return TextAcquisitionResult.terminal(
                TextAcquisitionStatus.INSUFFICIENT, "PDF_HAS_NO_PAGES"
            )
        try:
            return self._ocr_evidence(page_images, page_count=len(page_images))
        except RapidOcrUnavailable:
            return TextAcquisitionResult.terminal(
                TextAcquisitionStatus.FAILED, "RAPIDOCR_UNAVAILABLE"
            )
        except Exception:
            return TextAcquisitionResult.terminal(
                TextAcquisitionStatus.FAILED, "RAPIDOCR_FAILED"
            )

    def _acquire_image(self, path: str) -> TextAcquisitionResult:
        try:
            return self._ocr_evidence((path,), page_count=1)
        except RapidOcrUnavailable:
            return TextAcquisitionResult.terminal(
                TextAcquisitionStatus.FAILED, "RAPIDOCR_UNAVAILABLE"
            )
        except Exception:
            return TextAcquisitionResult.terminal(
                TextAcquisitionStatus.FAILED, "RAPIDOCR_FAILED"
            )

    @staticmethod
    def _acquire_xml(path: str) -> TextAcquisitionResult:
        try:
            root = ElementTree.parse(path).getroot()
        except (ElementTree.ParseError, OSError):
            return TextAcquisitionResult.terminal(
                TextAcquisitionStatus.FAILED, "XML_TEXT_ACQUISITION_FAILED"
            )
        fields = []
        for element in root.iter():
            text = normalize_document_text(element.text)
            if not text:
                continue
            tag = str(element.tag).rsplit("}", 1)[-1]
            fields.append(f"{tag}={text}")
        if not fields:
            return TextAcquisitionResult.terminal(
                TextAcquisitionStatus.INSUFFICIENT, "XML_TEXT_EMPTY"
            )
        return TextAcquisitionResult.acquired(
            TextEvidence(
                source=TextEvidenceSource.XML,
                text="\n".join(fields),
                page_count=1,
            )
        )

    def acquire(self, source_path: object) -> TextAcquisitionResult:
        path = os.path.abspath(str(source_path or "").strip())
        if not path or not os.path.isfile(path):
            return TextAcquisitionResult.terminal(
                TextAcquisitionStatus.FAILED, "LOCAL_TEXT_SOURCE_MISSING"
            )
        suffix = os.path.splitext(path)[1].lower()
        if suffix == ".pdf":
            return self._acquire_pdf(path)
        if suffix in SUPPORTED_IMAGE_SUFFIXES:
            return self._acquire_image(path)
        if suffix == ".xml":
            return self._acquire_xml(path)
        return TextAcquisitionResult.terminal(
            TextAcquisitionStatus.UNSUPPORTED, "LOCAL_TEXT_SOURCE_UNSUPPORTED"
        )
