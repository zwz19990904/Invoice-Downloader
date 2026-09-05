from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Mapping, Sequence

from document_types import is_exempt_type, normalize_document_type
from invoice_domain import (
    DocumentIdentity,
    InvoiceRecord,
    parse_amount,
    parse_local_date,
)
from local_text_extractor import TextEvidence, TextEvidenceSource


DEFAULT_MONEY_TOLERANCE = Decimal("0.01")

_UNKNOWN_VALUES = frozenset(
    {
        "",
        "unknown",
        "unknowndate",
        "未知",
        "未知日期",
        "未知金额",
        "未知购买方",
        "未知开票方",
        "暂无",
        "null",
        "none",
    }
)
_NUMBER_REQUIRED_TYPES = frozenset(
    {"火车票", "住宿发票", "餐饮", "过路费", "定额发票", "其他", "差旅服务费"}
)
_ROUTE_REQUIRED_TYPES = frozenset({"火车票", "航班行程单"})
_DOMESTIC_NUMERIC_NUMBER_TYPES = frozenset(
    {"火车票", "住宿发票", "餐饮", "过路费", "定额发票"}
)
_USCC_ALPHABET = "0123456789ABCDEFGHJKLMNPQRTUWXY"
_USCC_WEIGHTS = (1, 3, 9, 27, 19, 26, 16, 17, 20, 29, 25, 13, 8, 24, 10, 30, 28)
_SAFE_TRACE_SCALARS = (
    "engine",
    "reason_code",
    "provider",
    "evidence_source",
    "average_confidence",
    "recognition_status",
)


class RecognitionStatus(str, Enum):
    ACCEPTED = "accepted"
    REVIEW = "review"
    FAILED = "failed"


@dataclass(frozen=True)
class ValidationIssue:
    reason_code: str
    field_name: str = ""


@dataclass(frozen=True)
class RecognitionValidation:
    status: RecognitionStatus
    issues: tuple[ValidationIssue, ...] = field(default_factory=tuple)
    record: InvoiceRecord | None = None

    @property
    def reason_codes(self) -> tuple[str, ...]:
        return tuple(issue.reason_code for issue in self.issues)

    @property
    def primary_reason_code(self) -> str:
        if self.issues:
            return self.issues[0].reason_code
        return f"RECOGNITION_{self.status.value.upper()}"


def _normalized_text(value: object) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).strip()


def _is_present(value: object) -> bool:
    return _normalized_text(value).casefold().replace(" ", "") not in _UNKNOWN_VALUES


def _safe_recognition_trace(value: Mapping[str, Any] | None) -> dict[str, Any]:
    source = dict(value or {})
    safe = {
        field_name: source[field_name]
        for field_name in _SAFE_TRACE_SCALARS
        if field_name in source
    }
    provenance = source.get("field_provenance")
    if isinstance(provenance, Mapping):
        safe["field_provenance"] = {
            str(field_name): str(field_source)
            for field_name, field_source in provenance.items()
        }
    for field_name in ("conflict_fields", "validation_reason_codes"):
        values = source.get(field_name)
        if isinstance(values, (list, tuple)):
            safe[field_name] = [str(item) for item in values]
    issues = source.get("validation_issues")
    if isinstance(issues, (list, tuple)):
        safe["validation_issues"] = [
            {
                "reason_code": str(issue.get("reason_code") or ""),
                "field_name": str(issue.get("field_name") or ""),
            }
            for issue in issues
            if isinstance(issue, Mapping)
        ]
    return safe


def is_unified_social_credit_code_valid(value: object) -> bool:
    """Validate an 18-character GB 32100 unified social credit code."""
    code = re.sub(r"[^0-9A-Z]", "", _normalized_text(value).upper())
    if len(code) != 18 or any(character not in _USCC_ALPHABET for character in code):
        return False
    weighted_sum = sum(
        _USCC_ALPHABET.index(character) * weight
        for character, weight in zip(code[:17], _USCC_WEIGHTS)
    )
    expected = _USCC_ALPHABET[(31 - weighted_sum % 31) % 31]
    return code[-1] == expected


def is_tax_identifier_format_valid(value: object) -> bool:
    """Accept current and retained legacy Chinese taxpayer identifier shapes."""
    identifier = re.sub(r"[^0-9A-Z]", "", _normalized_text(value).upper())
    if re.fullmatch(r"\d{15}", identifier):
        return True
    if re.fullmatch(r"\d{8}[A-Z]{2}\d{5}", identifier):
        return True
    if re.fullmatch(r"[0-9A-HJ-NPQRTUWXY]{18}", identifier):
        return True
    if re.fullmatch(r"\d{20}", identifier):
        return True
    return bool(re.fullmatch(r"L[0-9A-Z]{15,18}", identifier))


def is_invoice_number_format_valid(value: object, document_type: object = "") -> bool:
    text = _normalized_text(value)
    if text.casefold().replace(" ", "") in _UNKNOWN_VALUES:
        return False
    text = re.sub(r"(?i)^invoice\s*(?:number|no\.?|#)?\s*[:#-]?\s*", "", text)
    text = re.sub(r"^(?:发票号码|票号)\s*[:：#-]?\s*", "", text)
    compact = re.sub(r"\s+", "", text)
    normalized_type = str(normalize_document_type(document_type))
    if compact.isdigit():
        return 8 <= len(compact) <= 20
    if normalized_type in _DOMESTIC_NUMERIC_NUMBER_TYPES:
        return False
    return bool(
        4 <= len(compact) <= 40
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/#-]*", compact)
        and re.search(r"\d", compact)
    )


def _tax_identifiers(text: object) -> tuple[tuple[str, str], ...]:
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    pattern = re.compile(
        r"(?P<label>统一社会信用代码|纳税人识别号|纳税识别号|税号)"
        r"[ \t]*[:：=]?[ \t]*(?:\r?\n[ \t]*)?"
        r"(?P<value>[0-9A-Za-z][0-9A-Za-z \t-]{2,28})"
    )
    found = []
    for match in pattern.finditer(normalized):
        value = re.sub(r"[^0-9A-Z]", "", match.group("value").upper())
        if value:
            found.append((match.group("label"), value))
    return tuple(found)


_MONEY_TOKEN = r"(?:\([-+]?[$¥￥]?\s*\d[\d,]*(?:\.\d+)?\)|[-+]?[$¥￥]?\s*\d[\d,]*(?:\.\d+)?)"


def _unique_labeled_amount(text: str, labels: Sequence[str]) -> Decimal | None:
    values: set[Decimal] = set()
    for label in labels:
        pattern = re.compile(
            rf"(?:{label})[ \t]*[:：=]?[ \t]*(?:CNY|RMB)?[ \t]*(?P<value>{_MONEY_TOKEN})",
            flags=re.IGNORECASE,
        )
        for match in pattern.finditer(text):
            value = parse_amount(match.group("value").replace("$", ""))
            if value is not None:
                values.add(value)
    return next(iter(values)) if len(values) == 1 else None


def extract_labeled_money_components(
    text: object,
) -> tuple[Decimal | None, Decimal | None, Decimal | None]:
    """Return an unambiguous subtotal, tax and tax-inclusive total when labelled."""
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    subtotal = _unique_labeled_amount(
        normalized,
        (
            r"不含税金额",
            r"合计金额",
            r"sub[ \t]*total",
            r"amount[ \t]+excluding[ \t]+tax",
            r"TotalAmWithoutTax",
        ),
    )
    tax = _unique_labeled_amount(
        normalized,
        (r"合计税额", r"税额合计", r"tax[ \t]+amount", r"TotalTaxAm"),
    )
    total = _unique_labeled_amount(
        normalized,
        (
            r"价税合计(?:\(小写\))?",
            r"(?<!不)含税金额",
            r"grand[ \t]+total",
            r"amount[ \t]+due",
            r"TotalTax-includedAmount",
            r"TotalAmount",
        ),
    )
    return subtotal, tax, total


class RecognitionValidator:
    """Apply deterministic acceptance rules without guessing missing fields."""

    def __init__(
        self,
        *,
        confidence_threshold: Decimal = Decimal("0.80"),
        money_tolerance: Decimal = DEFAULT_MONEY_TOLERANCE,
    ) -> None:
        self.confidence_threshold = Decimal(str(confidence_threshold))
        self.money_tolerance = Decimal(str(money_tolerance))

    def validate(
        self,
        payload: Mapping[str, Any] | None,
        identity: DocumentIdentity,
        *,
        evidence: TextEvidence | None = None,
        conflicts: Sequence[str] = (),
    ) -> RecognitionValidation:
        if not isinstance(payload, Mapping):
            return RecognitionValidation(
                RecognitionStatus.FAILED,
                (ValidationIssue("RECOGNITION_RESULT_NOT_MAPPING"),),
            )
        raw = dict(payload)
        meaningful_fields = (
            "Date",
            "Purchaser",
            "Seller",
            "Amount",
            "InvoiceCode",
            "InvoiceNumber",
            "Departure_Date",
            "Departure_City",
            "Destination_City",
        )
        if not any(_is_present(raw.get(field_name)) for field_name in meaningful_fields):
            return RecognitionValidation(
                RecognitionStatus.FAILED,
                (ValidationIssue("NO_USABLE_STRUCTURED_RESULT"),),
            )
        try:
            record = InvoiceRecord.from_legacy(raw, identity)
        except (TypeError, ValueError):
            return RecognitionValidation(
                RecognitionStatus.FAILED,
                (ValidationIssue("INVOICE_SCHEMA_ADAPTATION_FAILED"),),
            )
        document_type = str(record.document_type)
        if not record.is_invoice and not is_exempt_type(document_type):
            return RecognitionValidation(
                RecognitionStatus.FAILED,
                (ValidationIssue("DOCUMENT_NOT_RECOGNIZED_AS_INVOICE"),),
                record,
            )

        issues: list[ValidationIssue] = []
        seen: set[tuple[str, str]] = set()

        def add(reason_code: str, field_name: str = "") -> None:
            key = (reason_code, field_name)
            if key not in seen:
                seen.add(key)
                issues.append(ValidationIssue(reason_code, field_name))

        for field_name in conflicts:
            add("FIELD_VALUE_CONFLICT", str(field_name))

        required = ["Date", "Seller", "Amount", "Type"]
        if not is_exempt_type(document_type):
            required.append("Purchaser")
        if document_type in _NUMBER_REQUIRED_TYPES:
            required.append("InvoiceNumber")
        if document_type in _ROUTE_REQUIRED_TYPES:
            required.extend(("Departure_Date", "Departure_City", "Destination_City"))
        for field_name in required:
            if not _is_present(raw.get(field_name)):
                add("REQUIRED_FIELD_MISSING", field_name)

        if _is_present(raw.get("Date")) and parse_local_date(raw.get("Date")) is None:
            add("INVOICE_DATE_INVALID", "Date")
        if _is_present(raw.get("Departure_Date")) and parse_local_date(
            raw.get("Departure_Date")
        ) is None:
            add("DEPARTURE_DATE_INVALID", "Departure_Date")
        if _is_present(raw.get("Amount")):
            if not isinstance(raw.get("Amount"), (str, Decimal)):
                add("AMOUNT_TYPE_INVALID", "Amount")
            if record.amount is None:
                add("AMOUNT_INVALID", "Amount")
            elif record.amount == 0:
                add("AMOUNT_ZERO", "Amount")
        if _is_present(raw.get("InvoiceNumber")) and not is_invoice_number_format_valid(
            raw.get("InvoiceNumber"), document_type
        ):
            add("INVOICE_NUMBER_FORMAT_INVALID", "InvoiceNumber")

        if evidence is not None:
            if (
                evidence.source is TextEvidenceSource.RAPIDOCR
                and evidence.average_confidence is not None
                and evidence.average_confidence < self.confidence_threshold
            ):
                add("OCR_CONFIDENCE_BELOW_THRESHOLD")
            for label, identifier in _tax_identifiers(evidence.text):
                valid = (
                    is_unified_social_credit_code_valid(identifier)
                    if label == "统一社会信用代码"
                    else is_tax_identifier_format_valid(identifier)
                )
                if not valid:
                    add("TAX_IDENTIFIER_FORMAT_INVALID")
            subtotal, tax, total = extract_labeled_money_components(evidence.text)
            if subtotal is not None and tax is not None and total is not None:
                if abs((subtotal + tax) - total) > self.money_tolerance:
                    add("AMOUNT_TAX_TOTAL_MISMATCH")
            if record.amount is not None and total is not None:
                if abs(record.amount - total) > self.money_tolerance:
                    add("RECOGNIZED_AMOUNT_TOTAL_MISMATCH", "Amount")

        return RecognitionValidation(
            RecognitionStatus.REVIEW if issues else RecognitionStatus.ACCEPTED,
            tuple(issues),
            record,
        )


class RecognitionResultValidator:
    """Map internal validation states to the existing extraction contract."""

    def __init__(
        self,
        *,
        confidence_threshold: Decimal = Decimal("0.80"),
        artifact_path_resolver: Any | None = None,
    ) -> None:
        self.validator = RecognitionValidator(
            confidence_threshold=confidence_threshold
        )
        self.artifact_path_resolver = artifact_path_resolver or (
            lambda candidate, envelope: str(
                (envelope or {}).get("pdf_path") or candidate.source_path
            )
        )

    @staticmethod
    def _evidence(value: Any) -> TextEvidence | None:
        if isinstance(value, TextEvidence):
            return value
        evidence = getattr(value, "evidence", None)
        return evidence if isinstance(evidence, TextEvidence) else None

    def validate_envelope(
        self,
        candidate: Any,
        envelope: Mapping[str, Any],
        evidence_or_acquisition: Any = None,
        *,
        conflicts: Sequence[str] = (),
        provenance: Mapping[str, Any] | None = None,
        provider: str = "",
    ):
        from extraction_pipeline import ExtractionOutcome

        result = dict(envelope or {})
        info_json = result.get("info_json")
        evidence = self._evidence(evidence_or_acquisition)
        validation = self.validator.validate(
            info_json,
            candidate.identity,
            evidence=evidence,
            conflicts=conflicts,
        )
        extraction_trace = dict(result.get("extraction_trace") or {})
        if provider:
            extraction_trace["provider"] = provider
        if provenance is not None:
            extraction_trace["field_provenance"] = dict(provenance)
        if conflicts:
            extraction_trace["conflict_fields"] = list(conflicts)
        extraction_trace.update(
            {
                "recognition_status": validation.status.value,
                "validation_reason_codes": list(validation.reason_codes),
                "validation_issues": [
                    {
                        "reason_code": issue.reason_code,
                        "field_name": issue.field_name,
                    }
                    for issue in validation.issues
                ],
            }
        )
        if evidence is not None:
            extraction_trace["evidence_source"] = evidence.source.value
            extraction_trace["average_confidence"] = (
                str(evidence.average_confidence)
                if evidence.average_confidence is not None
                else None
            )
        result["extraction_trace"] = extraction_trace
        trace_context = dict(candidate.trace_context)
        trace_context["recognition"] = _safe_recognition_trace(extraction_trace)
        artifact_path = str(self.artifact_path_resolver(candidate, result))
        if validation.status is RecognitionStatus.ACCEPTED:
            return ExtractionOutcome(
                candidate=candidate,
                status="resolved",
                payload=result,
                trace_context=trace_context,
            )
        if validation.status is RecognitionStatus.REVIEW:
            return ExtractionOutcome(
                candidate=candidate,
                status="manual_review",
                reason_code=validation.primary_reason_code,
                message=validation.primary_reason_code,
                artifact_path=artifact_path,
                trace_context=trace_context,
            )
        return ExtractionOutcome(
            candidate=candidate,
            status="unresolved",
            reason_code=validation.primary_reason_code,
            message=validation.primary_reason_code,
            artifact_path=artifact_path,
            trace_context=trace_context,
        )

    def validate_existing(
        self,
        outcome: Any,
        evidence_or_acquisition: Any = None,
        *,
        provider: str = "",
    ):
        status = getattr(outcome, "status", None)
        if status != "resolved":
            if not hasattr(outcome, "to_legacy_trace_context"):
                return outcome
            from extraction_pipeline import ExtractionOutcome

            trace_context = outcome.to_legacy_trace_context()
            recognition_trace = _safe_recognition_trace(
                trace_context.get("recognition")
            )
            recognition_trace.update(
                {
                    "provider": provider,
                    "recognition_status": (
                        "review" if status in {"manual_review", "retained"} else "failed"
                    ),
                    "validation_reason_codes": [
                        str(outcome.reason_code or "RECOGNITION_FAILED")
                    ],
                    "validation_issues": [
                        {
                            "reason_code": str(
                                outcome.reason_code or "RECOGNITION_FAILED"
                            ),
                            "field_name": "",
                        }
                    ],
                }
            )
            trace_context["recognition"] = recognition_trace
            return ExtractionOutcome(
                candidate=outcome.candidate,
                status=outcome.status,
                payload=outcome.payload,
                reason_code=outcome.reason_code,
                message=outcome.message,
                artifact_path=outcome.artifact_path,
                trace_context=trace_context,
            )
        payload = outcome.to_legacy_payload()
        if not isinstance(payload, Mapping):
            payload = {"info_json": payload}
        return self.validate_envelope(
            outcome.candidate,
            payload,
            evidence_or_acquisition,
            provider=provider,
        )

    def __call__(
        self,
        candidate: Any,
        envelope: Mapping[str, Any],
        acquisition: Any = None,
    ):
        return self.validate_envelope(candidate, envelope, acquisition)
