from __future__ import annotations

import json
import os
import platform
import re
import sys
import threading
import unicodedata
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence

from local_text_extractor import TextEvidence


LEGACY_INVOICE_FIELDS = (
    "is_invoice",
    "Date",
    "Purchaser",
    "Seller",
    "Amount",
    "InvoiceCode",
    "InvoiceNumber",
    "Type",
    "category",
    "Departure_Date",
    "Departure_City",
    "Destination_City",
)
FACTUAL_TEXT_FIELDS = frozenset(
    {
        "Date",
        "Purchaser",
        "Seller",
        "Amount",
        "InvoiceCode",
        "InvoiceNumber",
        "Departure_Date",
        "Departure_City",
        "Destination_City",
    }
)
DEFAULT_MAX_EVIDENCE_CHARS = 24_000

_FIELD_ALIASES = {
    "is_invoice": "is_invoice",
    "date": "Date",
    "invoice_date": "Date",
    "purchaser": "Purchaser",
    "buyer": "Purchaser",
    "seller": "Seller",
    "amount": "Amount",
    "total": "Amount",
    "invoice_code": "InvoiceCode",
    "invoicecode": "InvoiceCode",
    "invoice_number": "InvoiceNumber",
    "invoicenumber": "InvoiceNumber",
    "type": "Type",
    "category": "category",
    "departure_date": "Departure_Date",
    "departure_city": "Departure_City",
    "destination_city": "Destination_City",
}


class LocalLLMProviderError(RuntimeError):
    reason_code = "LOCAL_MODEL_FAILED"

    def __init__(self, reason_code: str | None = None) -> None:
        self.reason_code = str(reason_code or self.reason_code)
        super().__init__(self.reason_code)


class LocalLLMUnavailable(LocalLLMProviderError):
    reason_code = "LOCAL_MODEL_UNAVAILABLE"


class LocalLLMOutputError(LocalLLMProviderError):
    reason_code = "LOCAL_MODEL_INVALID_OUTPUT"


class LocalLLMGroundingError(LocalLLMOutputError):
    reason_code = "LOCAL_MODEL_UNGROUNDED_OUTPUT"

    def __init__(self, fields: Sequence[str]) -> None:
        self.fields = tuple(str(item) for item in fields)
        super().__init__(self.reason_code)


class LocalLLMBackend(Protocol):
    def load(self, model_source: str) -> Any: ...

    def generate(
        self,
        loaded: Any,
        messages: Sequence[Mapping[str, str]],
        *,
        max_tokens: int,
    ) -> str: ...


@dataclass(frozen=True)
class _MlxLoadedModel:
    model: Any
    tokenizer: Any
    generate_fn: Callable[..., Any]
    sampler_factory: Callable[..., Any]
    model_source: str


class MlxLmBackend:
    """Small lazy adapter over mlx-lm's Python API."""

    def __init__(
        self,
        *,
        loader: Callable[..., Any] | None = None,
        generator: Callable[..., Any] | None = None,
        sampler_factory: Callable[..., Any] | None = None,
    ) -> None:
        self._loader = loader
        self._generator = generator
        self._sampler_factory = sampler_factory

    def _resolve_api(self) -> tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any]]:
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("DO_NOT_TRACK", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        if self._loader and self._generator and self._sampler_factory:
            return self._loader, self._generator, self._sampler_factory
        try:
            from mlx_lm import generate, load
            from mlx_lm.sample_utils import make_sampler
        except (ImportError, OSError) as exc:
            raise LocalLLMUnavailable("MLX_LM_UNAVAILABLE") from exc
        return self._loader or load, self._generator or generate, self._sampler_factory or make_sampler

    def load(self, model_source: str) -> _MlxLoadedModel:
        loader, generator, sampler_factory = self._resolve_api()
        try:
            model, tokenizer = loader(
                model_source,
                tokenizer_config={"trust_remote_code": False},
            )
        except Exception as exc:
            raise LocalLLMUnavailable("LOCAL_MODEL_LOAD_FAILED") from exc
        return _MlxLoadedModel(
            model=model,
            tokenizer=tokenizer,
            generate_fn=generator,
            sampler_factory=sampler_factory,
            model_source=model_source,
        )

    def generate(
        self,
        loaded: _MlxLoadedModel,
        messages: Sequence[Mapping[str, str]],
        *,
        max_tokens: int,
    ) -> str:
        try:
            prompt = loaded.tokenizer.apply_chat_template(
                list(messages),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError as exc:
            if "qwen3" in loaded.model_source.lower():
                raise LocalLLMUnavailable(
                    "LOCAL_MODEL_THINKING_CONTROL_UNSUPPORTED"
                ) from exc
            prompt = loaded.tokenizer.apply_chat_template(
                list(messages),
                tokenize=False,
                add_generation_prompt=True,
            )
        try:
            sampler = loaded.sampler_factory(temp=0.0, top_p=1.0, top_k=0)
            response = loaded.generate_fn(
                loaded.model,
                loaded.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=sampler,
                verbose=False,
            )
        except Exception as exc:
            raise LocalLLMProviderError("LOCAL_MODEL_GENERATION_FAILED") from exc
        return str(response or "").strip()


@dataclass(frozen=True)
class LocalLLMExtraction:
    payload: Mapping[str, Any]
    grounded_fields: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))
        object.__setattr__(self, "grounded_fields", tuple(self.grounded_fields))


@dataclass(frozen=True)
class FieldProvenance:
    source: str
    conflict: bool = False


@dataclass(frozen=True)
class InvoiceFieldMerge:
    payload: Mapping[str, Any]
    provenance: Mapping[str, FieldProvenance]
    conflicts: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))
        object.__setattr__(
            self, "provenance", MappingProxyType(dict(self.provenance))
        )
        object.__setattr__(self, "conflicts", tuple(self.conflicts))

    def trace_provenance(self) -> dict[str, str]:
        return {
            field_name: (
                f"{item.source}:conflict" if item.conflict else item.source
            )
            for field_name, item in self.provenance.items()
        }


def _is_empty(value: object) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _compact(value: object) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return "".join(
        character
        for character in normalized
        if character.isalnum() or "\u3400" <= character <= "\u9fff"
    )


def _digits(value: object) -> str:
    return "".join(character for character in str(value or "") if character.isdigit())


def _decimal(value: object) -> Decimal | None:
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    text = (
        text.replace(",", "")
        .replace("$", "")
        .replace("¥", "")
        .replace("￥", "")
    )
    try:
        parsed = Decimal(text)
    except (InvalidOperation, TypeError, ValueError):
        return None
    return parsed if parsed.is_finite() else None


def _equivalent(field_name: str, left: object, right: object) -> bool:
    if field_name == "is_invoice":
        return left is right or bool(left) == bool(right)
    if field_name == "Amount":
        left_amount = _decimal(left)
        right_amount = _decimal(right)
        return (
            left_amount is not None
            and right_amount is not None
            and left_amount == right_amount
        )
    if field_name in {"Date", "Departure_Date", "InvoiceCode", "InvoiceNumber"}:
        return bool(_digits(left)) and _digits(left) == _digits(right)
    return _compact(left) == _compact(right)


def normalize_deterministic_fields(values: Mapping[str, Any] | None) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in dict(values or {}).items():
        canonical = key if key in LEGACY_INVOICE_FIELDS else _FIELD_ALIASES.get(str(key).casefold())
        if canonical and not _is_empty(value):
            normalized[canonical] = value
    return normalized


def is_value_grounded(field_name: str, value: object, source_text: object) -> bool:
    if _is_empty(value):
        return True
    if field_name not in FACTUAL_TEXT_FIELDS:
        return True
    if field_name == "Amount":
        target_amount = _decimal(value)
        if target_amount is None:
            return False
        normalized_source = unicodedata.normalize("NFKC", str(source_text or ""))
        amount_windows = [
            normalized_source[match.start() : match.end() + 80]
            for match in re.finditer(
                r"价税合计|小写金额|应付金额|合计|grand\s+total|total\s+amount|amount\s+due|\btotal\b",
                normalized_source,
                flags=re.IGNORECASE,
            )
        ]
        amount_windows.extend(
            match.group(0)
            for match in re.finditer(
                r"[$¥￥]\s*[-+]?(?:\d[\d,]*)(?:\.\d+)?",
                normalized_source,
            )
        )
        source_amounts = (
            _decimal(match)
            for window in amount_windows
            for match in re.findall(
                r"[-+]?(?:\d[\d,]*)(?:\.\d+)?", window
            )
        )
        return any(amount == target_amount for amount in source_amounts)
    if field_name in {"Date", "Departure_Date"}:
        target_date = _digits(value)
        normalized_source = unicodedata.normalize("NFKC", str(source_text or ""))
        source_dates = {
            f"{year}{int(month):02d}{int(day):02d}"
            for year, month, day in re.findall(
                r"(20\d{2})\s*(?:年|[-/.])?\s*(0?[1-9]|1[0-2])\s*"
                r"(?:月|[-/.])?\s*(0?[1-9]|[12]\d|3[01])",
                normalized_source,
            )
        }
        return len(target_date) == 8 and target_date in source_dates
    compact_value = _compact(value)
    compact_source = _compact(source_text)
    if len(compact_value) >= 2 and compact_value in compact_source:
        return True
    return False


def grounded_deterministic_fields(
    values: Mapping[str, Any] | None, source_text: object
) -> dict[str, Any]:
    return {
        field_name: value
        for field_name, value in normalize_deterministic_fields(values).items()
        if is_value_grounded(field_name, value, source_text)
    }


def merge_invoice_fields(
    deterministic_fields: Mapping[str, Any] | None,
    model_fields: Mapping[str, Any] | None,
) -> InvoiceFieldMerge:
    deterministic = normalize_deterministic_fields(deterministic_fields)
    model = {
        key: value
        for key, value in dict(model_fields or {}).items()
        if key in LEGACY_INVOICE_FIELDS and not _is_empty(value)
    }
    payload: dict[str, Any] = {}
    provenance: dict[str, FieldProvenance] = {}
    conflicts = []
    for field_name in LEGACY_INVOICE_FIELDS:
        deterministic_value = deterministic.get(field_name)
        model_value = model.get(field_name)
        if not _is_empty(deterministic_value):
            conflict = bool(
                not _is_empty(model_value)
                and not _equivalent(field_name, deterministic_value, model_value)
            )
            payload[field_name] = deterministic_value
            provenance[field_name] = FieldProvenance("deterministic", conflict)
            if conflict:
                conflicts.append(field_name)
            continue
        if not _is_empty(model_value):
            payload[field_name] = model_value
            provenance[field_name] = FieldProvenance("local_llm")
            continue
        payload[field_name] = False if field_name == "is_invoice" else ""
        provenance[field_name] = FieldProvenance("missing")
    return InvoiceFieldMerge(payload, provenance, tuple(conflicts))


def parse_strict_invoice_json(raw_response: object) -> dict[str, Any]:
    text = str(raw_response or "").strip()
    if not text or not text.startswith("{") or not text.endswith("}"):
        raise LocalLLMOutputError("LOCAL_MODEL_JSON_ONLY_REQUIRED")
    duplicate_keys = []

    def _strict_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                duplicate_keys.append(key)
            result[key] = value
        return result

    try:
        parsed = json.loads(
            text,
            parse_float=Decimal,
            parse_int=Decimal,
            object_pairs_hook=_strict_object,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise LocalLLMOutputError("LOCAL_MODEL_JSON_INVALID") from exc
    if not isinstance(parsed, dict):
        raise LocalLLMOutputError("LOCAL_MODEL_JSON_OBJECT_REQUIRED")
    if duplicate_keys:
        raise LocalLLMOutputError("LOCAL_MODEL_JSON_DUPLICATE_KEYS")
    keys = set(parsed)
    expected = set(LEGACY_INVOICE_FIELDS)
    if keys != expected:
        raise LocalLLMOutputError("LOCAL_MODEL_SCHEMA_MISMATCH")
    if not isinstance(parsed["is_invoice"], bool):
        raise LocalLLMOutputError("LOCAL_MODEL_SCHEMA_TYPE_INVALID")
    for field_name in LEGACY_INVOICE_FIELDS[1:]:
        if parsed[field_name] is not None and not isinstance(parsed[field_name], str):
            raise LocalLLMOutputError("LOCAL_MODEL_SCHEMA_TYPE_INVALID")
    return {
        field_name: (
            parsed[field_name]
            if field_name == "is_invoice"
            else str(parsed[field_name] or "").strip()
        )
        for field_name in LEGACY_INVOICE_FIELDS
    }


def _build_messages(evidence: TextEvidence, max_evidence_chars: int) -> tuple[dict[str, str], ...]:
    schema = """{
  "is_invoice": true,
  "Date": null,
  "Purchaser": null,
  "Seller": null,
  "Amount": null,
  "InvoiceCode": null,
  "InvoiceNumber": null,
  "Type": null,
  "category": null,
  "Departure_Date": null,
  "Departure_City": null,
  "Destination_City": null
}"""
    system_prompt = (
        "You extract invoice fields from local source text. /no_think\n"
        "Use only facts explicitly present in SOURCE_TEXT. Never guess, infer a missing "
        "company, number, date, route, or amount, and never complete partial identifiers. "
        "Use null whenever a value cannot be determined. Date and Departure_Date must be "
        "YYYYMMDD strings when known. Amount is the displayed tax-inclusive grand total "
        "and must be a decimal JSON string, never a JSON number. Return exactly one JSON "
        "object with every key shown below, "
        "no Markdown, prose, comments, or extra keys.\n"
        f"SCHEMA:\n{schema}"
    )
    source_text = evidence.text[: max(1, int(max_evidence_chars))]
    return (
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"SOURCE_TEXT_BEGIN\n{source_text}\nSOURCE_TEXT_END",
        },
    )


def is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine().casefold() == "arm64"


class LocalLLMProvider:
    """Turn local text evidence into the legacy invoice schema without network APIs."""

    def __init__(
        self,
        model_source: str,
        *,
        max_tokens: int = 384,
        max_evidence_chars: int = DEFAULT_MAX_EVIDENCE_CHARS,
        backend: LocalLLMBackend | None = None,
        platform_checker: Callable[[], bool] = is_apple_silicon,
        event_sink: Callable[[str, Mapping[str, Any]], None] | None = None,
    ) -> None:
        raw_model_source = str(model_source or "").strip()
        self.configured_model_source = raw_model_source
        self._expects_local_path = raw_model_source.startswith(("/", "~", "."))
        self.model_source = (
            os.path.abspath(os.path.expanduser(raw_model_source))
            if self._expects_local_path
            else raw_model_source
        )
        self.max_tokens = max(64, min(2_048, int(max_tokens)))
        self.max_evidence_chars = max(1_000, int(max_evidence_chars))
        self.backend = backend or MlxLmBackend()
        self.platform_checker = platform_checker
        self.event_sink = event_sink
        self._loaded = None
        self._load_failure_reason = ""
        self._load_lock = threading.Lock()
        self._generation_lock = threading.Lock()

    def _emit(self, event: str, reason_code: str = "") -> None:
        if self.event_sink is None:
            return
        payload = {
            "provider": "local_mlx",
            "model_source_kind": (
                "local_path" if os.path.exists(self.model_source) else "huggingface"
            ),
        }
        if reason_code:
            payload["reason_code"] = reason_code
        try:
            self.event_sink(event, payload)
        except Exception:
            pass

    def _ensure_loaded(self) -> Any:
        if self._loaded is not None:
            return self._loaded
        if self._load_failure_reason:
            raise LocalLLMUnavailable(self._load_failure_reason)
        with self._load_lock:
            if self._loaded is not None:
                return self._loaded
            if self._load_failure_reason:
                raise LocalLLMUnavailable(self._load_failure_reason)
            if not self.model_source:
                raise LocalLLMUnavailable("LOCAL_MODEL_SOURCE_REQUIRED")
            if self._expects_local_path and not os.path.exists(self.model_source):
                raise LocalLLMUnavailable("LOCAL_MODEL_PATH_MISSING")
            if not self.platform_checker():
                raise LocalLLMUnavailable("LOCAL_MODEL_PLATFORM_UNSUPPORTED")
            self._emit("loading")
            try:
                self._loaded = self.backend.load(self.model_source)
            except LocalLLMProviderError as exc:
                self._load_failure_reason = exc.reason_code
                self._emit("failed", exc.reason_code)
                raise
            except Exception as exc:
                self._load_failure_reason = "LOCAL_MODEL_LOAD_FAILED"
                self._emit("failed", "LOCAL_MODEL_LOAD_FAILED")
                raise LocalLLMUnavailable("LOCAL_MODEL_LOAD_FAILED") from exc
            self._emit("ready")
            return self._loaded

    @property
    def is_loaded(self) -> bool:
        return self._loaded is not None

    def extract(
        self,
        evidence: TextEvidence,
        document_context: Mapping[str, Any] | None = None,
    ) -> LocalLLMExtraction:
        del document_context
        if not isinstance(evidence, TextEvidence) or not evidence.text.strip():
            raise LocalLLMProviderError("LOCAL_TEXT_EVIDENCE_REQUIRED")
        messages = _build_messages(evidence, self.max_evidence_chars)
        loaded = self._ensure_loaded()
        with self._generation_lock:
            raw_response = self.backend.generate(
                loaded, messages, max_tokens=self.max_tokens
            )
        payload = parse_strict_invoice_json(raw_response)
        ungrounded = tuple(
            field_name
            for field_name in FACTUAL_TEXT_FIELDS
            if not is_value_grounded(
                field_name, payload.get(field_name), evidence.text
            )
        )
        if ungrounded:
            raise LocalLLMGroundingError(sorted(ungrounded))
        grounded = tuple(
            field_name
            for field_name in FACTUAL_TEXT_FIELDS
            if not _is_empty(payload.get(field_name))
        )
        return LocalLLMExtraction(payload=payload, grounded_fields=grounded)
