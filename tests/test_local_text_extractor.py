import os
import threading
from decimal import Decimal
from types import SimpleNamespace

import fitz
import pytest

from candidate_pipeline import CandidatePipeline, CandidatePreflight
from local_text_extractor import (
    LocalTextExtractor,
    RapidOcrEngine,
    TextAcquisitionStatus,
    TextEvidence,
    TextEvidenceSource,
)


class _FailIfCalledOcr:
    @staticmethod
    def recognize(_source):
        pytest.fail("OCR must not run when the PDF text layer is usable")


class _FixedOcr:
    def __init__(self, rows):
        self.rows = rows
        self.sources = []

    def recognize(self, source):
        self.sources.append(source)
        return list(self.rows)


def _write_pdf(path, text=""):
    document = fitz.open()
    page = document.new_page()
    if text:
        page.insert_textbox(fitz.Rect(36, 36, 560, 800), text, fontsize=10)
    document.save(path)
    document.close()


def test_usable_native_pdf_text_prevents_ocr(tmp_path):
    pdf_path = tmp_path / "native.pdf"
    text = (
        "TAX INVOICE Invoice Number 12345678901234567890 "
        "Invoice Date 2026-09-01 Purchaser Example Buyer Seller Example Seller "
        "Total Amount CNY 123.45 Description Hotel accommodation "
    )
    _write_pdf(pdf_path, text)

    result = LocalTextExtractor(ocr_engine=_FailIfCalledOcr()).acquire(pdf_path)

    assert result.status is TextAcquisitionStatus.ACQUIRED
    assert result.evidence.source is TextEvidenceSource.PDF_TEXT
    assert "Invoice Number" in result.evidence.text
    assert result.evidence.spans == ()
    assert result.evidence.average_confidence is None


def test_textless_pdf_uses_rapidocr_and_preserves_evidence(tmp_path):
    pdf_path = tmp_path / "scan.pdf"
    _write_pdf(pdf_path)
    ocr = _FixedOcr(
        [
            ([[1, 2], [11, 2], [11, 8], [1, 8]], "发票号码", 0.95),
            ([[1, 10], [20, 10], [20, 18], [1, 18]], "1234567890", 0.90),
        ]
    )

    result = LocalTextExtractor(ocr_engine=ocr, ocr_dpi=72).acquire(pdf_path)

    assert result.status is TextAcquisitionStatus.ACQUIRED
    assert result.evidence.source is TextEvidenceSource.RAPIDOCR
    assert result.evidence.text == "发票号码\n1234567890"
    assert result.evidence.page_count == 1
    assert result.evidence.average_confidence == Decimal("0.925")
    assert result.evidence.spans[0].page_index == 0
    assert result.evidence.spans[0].bounding_box == (
        (1.0, 2.0),
        (11.0, 2.0),
        (11.0, 8.0),
        (1.0, 8.0),
    )
    assert result.evidence.spans[0].confidence == Decimal("0.95")
    assert isinstance(ocr.sources[0], bytes)


def test_image_uses_rapidocr_without_opening_it_as_pdf(tmp_path):
    image_path = tmp_path / "invoice.png"
    image_path.write_bytes(b"fixture image bytes")
    ocr = _FixedOcr(
        [([[0, 0], [4, 0], [4, 4], [0, 4]], "Amount 88.00", 0.8)]
    )

    result = LocalTextExtractor(ocr_engine=ocr).acquire(image_path)

    assert result.status is TextAcquisitionStatus.ACQUIRED
    assert result.evidence.text == "Amount 88.00"
    assert result.evidence.source is TextEvidenceSource.RAPIDOCR
    assert ocr.sources == [str(image_path)]


def test_xml_evidence_keeps_field_names_and_values(tmp_path):
    xml_path = tmp_path / "invoice.xml"
    xml_path.write_text(
        "<Invoice><InvoiceNumber>12345678</InvoiceNumber>"
        "<SellerName>Example Seller</SellerName>"
        "<TotalAmount>42.50</TotalAmount></Invoice>",
        encoding="utf-8",
    )

    result = LocalTextExtractor(ocr_engine=_FailIfCalledOcr()).acquire(xml_path)

    assert result.status is TextAcquisitionStatus.ACQUIRED
    assert result.evidence.source is TextEvidenceSource.XML
    assert "InvoiceNumber=12345678" in result.evidence.text
    assert "SellerName=Example Seller" in result.evidence.text
    assert result.evidence.average_confidence is None


def test_rapidocr_engine_loads_factory_once_and_disables_telemetry(monkeypatch):
    calls = []
    monkeypatch.delenv("ORT_DISABLE_TELEMETRY", raising=False)

    class Backend:
        @staticmethod
        def __call__(_source):
            return [], 0.01

    def factory():
        calls.append("load")
        return Backend()

    engine = RapidOcrEngine(factory=factory)

    assert tuple(engine.recognize(b"first")) == ()
    assert tuple(engine.recognize(b"second")) == ()
    assert calls == ["load"]
    assert os.environ["ORT_DISABLE_TELEMETRY"] == "1"


def test_unsupported_or_missing_source_fails_without_ocr(tmp_path):
    unsupported = tmp_path / "invoice.ofd"
    unsupported.write_bytes(b"ofd")
    extractor = LocalTextExtractor(ocr_engine=_FailIfCalledOcr())

    unsupported_result = extractor.acquire(unsupported)
    missing_result = extractor.acquire(tmp_path / "missing.pdf")

    assert unsupported_result.status is TextAcquisitionStatus.UNSUPPORTED
    assert unsupported_result.reason_code == "LOCAL_TEXT_SOURCE_UNSUPPORTED"
    assert missing_result.status is TextAcquisitionStatus.FAILED
    assert missing_result.reason_code == "LOCAL_TEXT_SOURCE_MISSING"


def test_text_evidence_is_immutable():
    evidence = TextEvidence(
        source=TextEvidenceSource.PDF_TEXT,
        text="Invoice Number 12345678",
    )

    with pytest.raises((AttributeError, TypeError)):
        evidence.text = "changed"


def test_preflight_stores_local_text_without_preparing_cloud_images(tmp_path):
    source = tmp_path / "unresolved.pdf"
    source.write_bytes(b"%PDF fixture" * 200)
    candidate = CandidatePipeline().collect(
        [{"filepath": str(source), "message_uid": "mail-1"}]
    )[0]
    evidence = TextEvidence(
        source=TextEvidenceSource.PDF_TEXT,
        text="Invoice Number 12345678 Seller Example Amount 42.50",
    )
    acquisition = SimpleNamespace(
        status=TextAcquisitionStatus.ACQUIRED,
        evidence=evidence,
        reason_code="",
    )
    sidecar = {}

    class ExtractorStub:
        @staticmethod
        def probe_local_only(*_args, **_kwargs):
            return SimpleNamespace(status="needs_remote")

        @staticmethod
        def pdf_to_base64_image(_path):
            pytest.fail("Local mode must not prepare cloud image payloads")

    preflight = CandidatePreflight(
        api=SimpleNamespace(),
        extractor=ExtractorStub(),
        working_history=set(),
        sidecar=sidecar,
        sidecar_lock=threading.Lock(),
        converter_factory=lambda: None,
        text_extractor=SimpleNamespace(acquire=lambda _path: acquisition),
        prepare_remote_images=False,
    )

    assert preflight(candidate) is None
    prepared = sidecar[candidate.identity.document_id]
    assert prepared["text_acquisition"] is acquisition
    assert prepared["base64_img"] is None
