import threading
from decimal import Decimal
from types import SimpleNamespace

from candidate_pipeline import CandidatePipeline, CandidatePreflight
from extraction_pipeline import ExtractionOutcome
from local_text_extractor import (
    OcrSpan,
    TextAcquisitionResult,
    TextEvidence,
    TextEvidenceSource,
)
from recognition_validation import (
    RecognitionResultValidator,
    RecognitionStatus,
    RecognitionValidator,
    extract_labeled_money_components,
    is_invoice_number_format_valid,
    is_tax_identifier_format_valid,
    is_unified_social_credit_code_valid,
)


def _candidate(path="/tmp/invoice.pdf"):
    return CandidatePipeline().collect(
        [{"filepath": str(path), "message_uid": "mail-1"}]
    )[0]


def _payload(**overrides):
    result = {
        "is_invoice": True,
        "Date": "20260901",
        "Purchaser": "示例购买方有限公司",
        "Seller": "示例销售方有限公司",
        "Amount": "106.00",
        "InvoiceCode": "",
        "InvoiceNumber": "26372000002439975871",
        "Type": "餐饮",
        "category": "餐饮",
        "Departure_Date": "",
        "Departure_City": "",
        "Destination_City": "",
    }
    result.update(overrides)
    return result


def _evidence(text, *, source=TextEvidenceSource.PDF_TEXT, confidence=None):
    spans = ()
    if source is TextEvidenceSource.RAPIDOCR:
        spans = (
            OcrSpan(
                page_index=0,
                text=text,
                bounding_box=((0, 0), (1, 0), (1, 1), (0, 1)),
                confidence=confidence if confidence is not None else Decimal("0.90"),
            ),
        )
    return TextEvidence(source=source, text=text, spans=spans)


def test_valid_result_accepts_decimal_arithmetic_and_standard_tax_code():
    candidate = _candidate()
    evidence = _evidence(
        "\n".join(
            (
                "统一社会信用代码: 91310000710920127H",
                "不含税金额: 100.00",
                "合计税额: 6.00",
                "价税合计(小写): 106.00",
            )
        )
    )

    result = RecognitionValidator().validate(
        _payload(), candidate.identity, evidence=evidence
    )

    assert result.status is RecognitionStatus.ACCEPTED
    assert result.reason_codes == ()
    assert result.record.amount == Decimal("106.00")

    invalid_tax_id = RecognitionValidator().validate(
        _payload(),
        candidate.identity,
        evidence=_evidence("统一社会信用代码: 913100007109201270"),
    )
    assert "TAX_IDENTIFIER_FORMAT_INVALID" in invalid_tax_id.reason_codes


def test_amount_tax_total_mismatch_and_payload_total_mismatch_require_review():
    candidate = _candidate()
    evidence = _evidence(
        "不含税金额: 100.00\n合计税额: 6.00\n价税合计(小写): 108.00"
    )

    result = RecognitionValidator().validate(
        _payload(Amount="106.00"), candidate.identity, evidence=evidence
    )

    assert result.status is RecognitionStatus.REVIEW
    assert "AMOUNT_TAX_TOTAL_MISMATCH" in result.reason_codes
    assert "RECOGNIZED_AMOUNT_TOTAL_MISMATCH" in result.reason_codes


def test_negative_red_invoice_amount_is_valid_and_zero_requires_review():
    candidate = _candidate()

    negative = RecognitionValidator().validate(
        _payload(Amount="-106.00"), candidate.identity
    )
    zero = RecognitionValidator().validate(
        _payload(Amount="0.00"), candidate.identity
    )

    assert negative.status is RecognitionStatus.ACCEPTED
    assert zero.status is RecognitionStatus.REVIEW
    assert "AMOUNT_ZERO" in zero.reason_codes

    float_amount = RecognitionValidator().validate(
        _payload(Amount=106.0), candidate.identity
    )
    assert "AMOUNT_TYPE_INVALID" in float_amount.reason_codes


def test_missing_invalid_and_conflicting_critical_fields_require_review():
    candidate = _candidate()

    result = RecognitionValidator().validate(
        _payload(
            Date="2026-02-30",
            Purchaser="未知购买方",
            InvoiceNumber="1234",
        ),
        candidate.identity,
        conflicts=("Seller",),
    )

    assert result.status is RecognitionStatus.REVIEW
    assert result.reason_codes[0] == "FIELD_VALUE_CONFLICT"
    assert "REQUIRED_FIELD_MISSING" in result.reason_codes
    assert "INVOICE_DATE_INVALID" in result.reason_codes
    assert "INVOICE_NUMBER_FORMAT_INVALID" in result.reason_codes


def test_low_rapidocr_confidence_prevents_acceptance():
    candidate = _candidate()
    evidence = _evidence(
        "发票号码 26372000002439975871",
        source=TextEvidenceSource.RAPIDOCR,
        confidence=Decimal("0.79"),
    )

    result = RecognitionValidator(confidence_threshold=Decimal("0.80")).validate(
        _payload(), candidate.identity, evidence=evidence
    )

    assert result.status is RecognitionStatus.REVIEW
    assert "OCR_CONFIDENCE_BELOW_THRESHOLD" in result.reason_codes


def test_blank_or_explicit_non_invoice_result_maps_to_failed():
    candidate = _candidate()
    validator = RecognitionValidator()

    blank = validator.validate(
        {key: "" for key in _payload()}, candidate.identity
    )
    non_invoice = validator.validate(
        _payload(is_invoice=False), candidate.identity
    )

    assert blank.status is RecognitionStatus.FAILED
    assert blank.primary_reason_code == "NO_USABLE_STRUCTURED_RESULT"
    assert non_invoice.status is RecognitionStatus.FAILED
    assert non_invoice.primary_reason_code == "DOCUMENT_NOT_RECOGNIZED_AS_INVOICE"


def test_complete_exempt_supporting_document_can_be_accepted():
    candidate = _candidate()

    result = RecognitionValidator().validate(
        _payload(
            is_invoice=False,
            Purchaser="个人",
            InvoiceNumber="",
            Type="打车",
            category="打车",
        ),
        candidate.identity,
    )

    assert result.status is RecognitionStatus.ACCEPTED


def test_invoice_number_and_tax_identifier_formats_are_conservative():
    assert is_invoice_number_format_valid("12345678", "餐饮")
    assert is_invoice_number_format_valid("26372000002439975871", "火车票")
    assert is_invoice_number_format_valid("Invoice #23265242", "其他")
    assert is_invoice_number_format_valid("SCCT00921845", "差旅服务费")
    assert not is_invoice_number_format_valid("SCCT00921845", "餐饮")
    assert not is_invoice_number_format_valid("1234", "其他")

    assert is_unified_social_credit_code_valid("91310000710920127H")
    assert not is_unified_social_credit_code_valid("913100007109201270")
    assert is_tax_identifier_format_valid("123456789012345")
    assert is_tax_identifier_format_valid("12345678901234567890")
    assert not is_tax_identifier_format_valid("ABC")


def test_money_component_extraction_requires_one_unambiguous_value_per_label():
    assert extract_labeled_money_components(
        "不含税金额: (100.00)\n合计税额: -6.00\n价税合计(小写): -106.00"
    ) == (Decimal("-100.00"), Decimal("-6.00"), Decimal("-106.00"))
    assert extract_labeled_money_components(
        "不含税金额: 100.00\n不含税金额: 200.00\n合计税额: 6.00\n价税合计: 106.00"
    )[0] is None


def test_result_mapper_keeps_safe_trace_and_maps_review_to_manual_review(tmp_path):
    source = tmp_path / "invoice.pdf"
    source.write_bytes(b"%PDF fixture")
    candidate = _candidate(source)
    secret_text = "private invoice text"
    evidence = _evidence(
        secret_text,
        source=TextEvidenceSource.RAPIDOCR,
        confidence=Decimal("0.50"),
    )
    mapper = RecognitionResultValidator(confidence_threshold=Decimal("0.80"))

    outcome = mapper.validate_envelope(
        candidate,
        {
            "pdf_path": str(source),
            "info_json": _payload(),
            "extraction_trace": {"raw_source_text": secret_text},
        },
        evidence,
        provenance={"Seller": "local_llm"},
        provider="local_mlx",
    )
    trace = outcome.to_legacy_trace_context()["recognition"]

    assert outcome.status == "manual_review"
    assert outcome.artifact_path == str(source)
    assert trace["recognition_status"] == "review"
    assert trace["provider"] == "local_mlx"
    assert trace["average_confidence"] == "0.50"
    assert secret_text not in repr(trace)


def test_result_mapper_annotates_existing_provider_failure_without_payload_data():
    candidate = _candidate()
    original = ExtractionOutcome(
        candidate=candidate,
        status="manual_review",
        reason_code="EXTRACTOR_ALL_ENGINES_FAILED",
    )

    outcome = RecognitionResultValidator().validate_existing(
        original, provider="glm"
    )
    trace = outcome.to_legacy_trace_context()["recognition"]

    assert outcome.status == "manual_review"
    assert trace == {
        "provider": "glm",
        "recognition_status": "review",
        "validation_reason_codes": ["EXTRACTOR_ALL_ENGINES_FAILED"],
        "validation_issues": [
            {
                "reason_code": "EXTRACTOR_ALL_ENGINES_FAILED",
                "field_name": "",
            }
        ],
    }


def test_resolved_deterministic_preflight_uses_validation_gate(tmp_path):
    source = tmp_path / "invoice.pdf"
    source.write_bytes(b"%PDF fixture" * 200)
    candidate = _candidate(source)
    mapper = RecognitionResultValidator()

    class Extractor:
        @staticmethod
        def probe_local_only(*_args, **_kwargs):
            return SimpleNamespace(
                status="resolved",
                result=_payload(Date="2026-02-30"),
                engine="local_test",
                reason_code="LOCAL_TEST",
            )

    preflight = CandidatePreflight(
        api=SimpleNamespace(),
        extractor=Extractor(),
        working_history=set(),
        sidecar={},
        sidecar_lock=threading.Lock(),
        converter_factory=lambda: None,
        resolved_validator=mapper,
    )

    outcome = preflight(candidate)

    assert outcome.status == "manual_review"
    assert outcome.reason_code == "INVOICE_DATE_INVALID"
    assert outcome.to_legacy_trace_context()["recognition"]["recognition_status"] == "review"


def test_validation_failure_can_continue_with_only_valid_deterministic_fields(tmp_path):
    source = tmp_path / "invoice.pdf"
    source.write_bytes(b"%PDF fixture" * 200)
    candidate = _candidate(source)
    sidecar = {}

    class Extractor:
        @staticmethod
        def probe_local_only(*_args, **_kwargs):
            return SimpleNamespace(
                status="resolved",
                result=_payload(Date="2026-02-30"),
                engine="local_test",
                reason_code="LOCAL_TEST",
            )

        @staticmethod
        def pdf_to_base64_image(_path):
            raise AssertionError("Local continuation must not prepare cloud images")

    preflight = CandidatePreflight(
        api=SimpleNamespace(),
        extractor=Extractor(),
        working_history=set(),
        sidecar=sidecar,
        sidecar_lock=threading.Lock(),
        converter_factory=lambda: None,
        resolved_validator=RecognitionResultValidator(),
        continue_after_validation_failure=True,
        prepare_remote_images=False,
    )

    assert preflight(candidate) is None
    deterministic = sidecar[candidate.identity.document_id]["deterministic_fields"]
    assert "Date" not in deterministic
    assert deterministic["Seller"] == "示例销售方有限公司"
