import json
import os
import threading
from types import SimpleNamespace

import fitz
import pytest

from candidate_pipeline import CandidatePipeline
from extraction_pipeline import ExtractionOutcome
from local_llm_provider import (
    LEGACY_INVOICE_FIELDS,
    LocalLLMExtraction,
    LocalLLMGroundingError,
    LocalLLMOutputError,
    LocalLLMProvider,
    LocalLLMUnavailable,
    MlxLmBackend,
    grounded_deterministic_fields,
    is_value_grounded,
    merge_invoice_fields,
    parse_strict_invoice_json,
)
from local_text_extractor import (
    TextAcquisitionResult,
    TextEvidence,
    TextEvidenceSource,
)
from recognition_policy import RecognitionPolicy
from recognition_router import (
    LocalEvidenceRecognitionExtractor,
    ModeAwareRecognitionExtractor,
)


def _payload(**overrides):
    result = {
        "is_invoice": True,
        "Date": "20260901",
        "Purchaser": "Example Buyer",
        "Seller": "Example Seller",
        "Amount": "42.50",
        "InvoiceCode": None,
        "InvoiceNumber": "12345678",
        "Type": "餐饮",
        "category": "餐饮",
        "Departure_Date": None,
        "Departure_City": None,
        "Destination_City": None,
    }
    result.update(overrides)
    return result


def _evidence():
    return TextEvidence(
        source=TextEvidenceSource.PDF_TEXT,
        text=(
            "TAX INVOICE\nInvoice Date 2026-09-01\nInvoice Number 12345678\n"
            "Purchaser Example Buyer\nSeller Example Seller\nTotal CNY 42.50\n餐饮"
        ),
    )


class _FakeBackend:
    def __init__(self, responses):
        self.responses = list(responses)
        self.load_calls = []
        self.generate_calls = []

    def load(self, model_source):
        self.load_calls.append(model_source)
        return object()

    def generate(self, loaded, messages, *, max_tokens):
        self.generate_calls.append((loaded, messages, max_tokens))
        return self.responses.pop(0)


def test_provider_loads_fake_backend_once_and_generates_text_only():
    backend = _FakeBackend([json.dumps(_payload()), json.dumps(_payload())])
    provider = LocalLLMProvider(
        "mlx-community/Qwen3-1.7B-4bit",
        max_tokens=320,
        backend=backend,
        platform_checker=lambda: True,
    )

    first = provider.extract(_evidence())
    second = provider.extract(_evidence())

    assert backend.load_calls == ["mlx-community/Qwen3-1.7B-4bit"]
    assert len(backend.generate_calls) == 2
    assert all(call[2] == 320 for call in backend.generate_calls)
    assert "/no_think" in backend.generate_calls[0][1][0]["content"]
    assert "image" not in repr(backend.generate_calls[0][1]).lower()
    assert first.payload["Seller"] == "Example Seller"
    assert first.payload["InvoiceCode"] == ""
    assert second.payload == first.payload
    assert provider.is_loaded is True


def test_mlx_backend_disables_thinking_and_uses_greedy_sampler(monkeypatch):
    calls = {}
    monkeypatch.delenv("HF_HUB_DISABLE_TELEMETRY", raising=False)
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls["messages"] = messages
            calls["template_kwargs"] = kwargs
            return "rendered prompt"

    def loader(source, **kwargs):
        calls["load"] = (source, kwargs)
        return object(), Tokenizer()

    def sampler_factory(**kwargs):
        calls["sampler"] = kwargs
        return "greedy-sampler"

    def generator(model, tokenizer, **kwargs):
        calls["generate"] = (model, tokenizer, kwargs)
        return json.dumps(_payload())

    backend = MlxLmBackend(
        loader=loader,
        generator=generator,
        sampler_factory=sampler_factory,
    )
    loaded = backend.load("mlx-community/Qwen3-1.7B-4bit")

    response = backend.generate(
        loaded,
        ({"role": "user", "content": "text"},),
        max_tokens=200,
    )

    assert calls["load"][1] == {
        "tokenizer_config": {"trust_remote_code": False},
    }
    assert calls["template_kwargs"] == {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    assert calls["sampler"] == {"temp": 0.0, "top_p": 1.0, "top_k": 0}
    assert calls["generate"][2]["sampler"] == "greedy-sampler"
    assert calls["generate"][2]["verbose"] is False
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert os.environ["DO_NOT_TRACK"] == "1"
    assert response.startswith("{")


@pytest.mark.parametrize(
    "response",
    [
        "```json\n{}\n```",
        json.dumps({key: None for key in LEGACY_INVOICE_FIELDS[:-1]}),
        json.dumps({**_payload(), "unexpected": "value"}),
        json.dumps(_payload(Amount=42.5)),
        json.dumps(_payload())[:-1] + ',"Seller":"Duplicate"}',
    ],
)
def test_strict_json_adapter_rejects_wrappers_schema_drift_and_numbers(response):
    with pytest.raises(LocalLLMOutputError):
        parse_strict_invoice_json(response)


def test_provider_rejects_model_values_not_present_in_source():
    backend = _FakeBackend(
        [json.dumps(_payload(Seller="Invented Corporation"))]
    )
    provider = LocalLLMProvider(
        "local-test-model",
        backend=backend,
        platform_checker=lambda: True,
    )

    with pytest.raises(LocalLLMGroundingError) as exc_info:
        provider.extract(_evidence())

    assert exc_info.value.fields == ("Seller",)


def test_platform_guard_fails_before_backend_load():
    backend = _FakeBackend([json.dumps(_payload())])
    provider = LocalLLMProvider(
        "mlx-community/Qwen3-1.7B-4bit",
        backend=backend,
        platform_checker=lambda: False,
    )

    with pytest.raises(LocalLLMUnavailable) as exc_info:
        provider.extract(_evidence())

    assert exc_info.value.reason_code == "LOCAL_MODEL_PLATFORM_UNSUPPORTED"
    assert backend.load_calls == []


def test_failed_model_load_is_not_retried_for_every_invoice():
    class FailingBackend:
        def __init__(self):
            self.load_calls = 0

        def load(self, _model_source):
            self.load_calls += 1
            raise LocalLLMUnavailable("LOCAL_MODEL_LOAD_FAILED")

    backend = FailingBackend()
    provider = LocalLLMProvider(
        "mlx-community/Qwen3-1.7B-4bit",
        backend=backend,
        platform_checker=lambda: True,
    )

    for _ in range(2):
        with pytest.raises(LocalLLMUnavailable):
            provider.extract(_evidence())

    assert backend.load_calls == 1


def test_merge_keeps_deterministic_values_and_records_conflicts():
    model = parse_strict_invoice_json(json.dumps(_payload()))
    matching = merge_invoice_fields(
        {"invoice_number": "12345678", "amount": "42.500"}, model
    )
    conflicting = merge_invoice_fields({"seller": "Different Seller"}, model)

    assert matching.payload["InvoiceNumber"] == "12345678"
    assert matching.payload["Amount"] == "42.500"
    assert matching.provenance["Amount"].source == "deterministic"
    assert matching.conflicts == ()
    assert conflicting.payload["Seller"] == "Different Seller"
    assert conflicting.conflicts == ("Seller",)
    assert conflicting.trace_provenance()["Seller"] == "deterministic:conflict"


def test_only_grounded_deterministic_fields_can_enter_merge():
    grounded = grounded_deterministic_fields(
        {
            "invoice_number": "12345678",
            "seller": "Invented Corporation",
            "amount": "42.50",
        },
        _evidence().text,
    )

    assert grounded == {"InvoiceNumber": "12345678", "Amount": "42.50"}


def test_grounding_uses_date_and_amount_context_instead_of_unrelated_digits():
    source = "开票日期 2026年9月1日 订单号 20260001 价税合计 ¥42.50"

    assert is_value_grounded("Date", "20260901", source) is True
    assert is_value_grounded("Amount", "42.50", source) is True
    assert is_value_grounded("Amount", "2026", source) is False


def test_local_recognizer_adapts_merged_result_without_copying_source_text(tmp_path):
    source = tmp_path / "invoice.pdf"
    source.write_bytes(b"%PDF fixture")
    candidate = CandidatePipeline().collect(
        [{"filepath": str(source), "message_uid": "mail-1"}]
    )[0]
    evidence = _evidence()
    acquisition = TextAcquisitionResult.acquired(evidence)
    sidecar = {
        candidate.identity.document_id: {
            "pdf_path": str(source),
            "metadata": {
                "provider_recovered_fields": {"invoice_number": "12345678"}
            },
            "text_acquisition": acquisition,
            "deterministic_fields": {},
        }
    }

    class Provider:
        @staticmethod
        def extract(_evidence_value, document_context=None):
            assert document_context is not None
            return LocalLLMExtraction(parse_strict_invoice_json(json.dumps(_payload())))

    class Owner:
        @staticmethod
        def _adapt_extraction_result(payload, **_kwargs):
            return dict(payload)

    recognizer = LocalEvidenceRecognitionExtractor(
        provider=Provider(),
        owner_extractor=Owner(),
        sidecar=sidecar,
        sidecar_lock=threading.Lock(),
    )

    outcome = recognizer(candidate)
    payload = outcome.to_legacy_payload()

    assert outcome.status == "resolved"
    assert payload["info_json"]["Seller"] == "Example Seller"
    assert payload["extraction_trace"]["field_provenance"]["InvoiceNumber"] == "deterministic"
    assert evidence.text not in repr(payload["extraction_trace"])
    assert recognizer.verified_ceiling() == 1


def test_hybrid_calls_cloud_only_after_local_provider_failure():
    candidate = CandidatePipeline().collect(
        [{"filepath": "/tmp/invoice.pdf", "message_uid": "mail-1"}]
    )[0]
    calls = []
    local_review = ExtractionOutcome(
        candidate=candidate,
        status="manual_review",
        reason_code="LOCAL_MODEL_INVALID_OUTPUT",
    )
    cloud_result = ExtractionOutcome.resolved(candidate, {"info_json": {}})
    router = ModeAwareRecognitionExtractor(
        policy=RecognitionPolicy.from_settings(
            {"recognition_mode": "hybrid", "cloud_provider": "glm"}
        ),
        local_extractor=lambda _candidate: local_review,
        cloud_extractors={"glm": lambda item: calls.append(item) or cloud_result},
    )

    assert router(candidate) is cloud_result
    assert calls == [candidate]


def test_desktop_local_policy_wires_text_evidence_to_injected_provider(
    tmp_path, monkeypatch
):
    import user_settings
    from app_api import InvoiceAppAPI

    monkeypatch.setattr(user_settings, "_is_macos", lambda: False)
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "localappdata"))
    source = tmp_path / "invoice.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_textbox(
        fitz.Rect(36, 36, 560, 800),
        _evidence().text,
        fontsize=10,
    )
    document.save(source)
    document.close()
    provider_calls = []

    class Provider:
        @staticmethod
        def extract(evidence, document_context=None):
            provider_calls.append((evidence, document_context))
            return LocalLLMExtraction(
                parse_strict_invoice_json(json.dumps(_payload()))
            )

    class Owner:
        @staticmethod
        def load_processed_records():
            return {}

        @staticmethod
        def probe_local_only(_path, document_context=None):
            del document_context
            return SimpleNamespace(status="needs_remote", result=None)

        @staticmethod
        def pdf_to_base64_image(_path):
            pytest.fail("Local mode cannot prepare a cloud image payload")

        @staticmethod
        def _adapt_extraction_result(payload, **_kwargs):
            return dict(payload)

    api = InvoiceAppAPI(revision_resolver=lambda: "a" * 40)
    api._safe_emit_stage_event = lambda *_args, **_kwargs: None
    owner = Owner()
    session = api._create_processing_pipeline_session(
        [{"filepath": str(source), "message_uid": "mail-1"}],
        "",
        str(tmp_path / "output"),
        _extractor=owner,
        _recognition_policy=RecognitionPolicy.from_settings(
            {"recognition_mode": "local"}
        ),
        _local_provider=Provider(),
    )

    try:
        outcomes = session.extract()
    finally:
        session.close()

    assert len(outcomes) == 1
    assert outcomes[0].status == "resolved"
    assert outcomes[0].to_legacy_payload()["info_json"]["Seller"] == "Example Seller"
    assert len(provider_calls) == 1
    assert provider_calls[0][0].source is TextEvidenceSource.PDF_TEXT


def test_desktop_reuses_one_provider_for_same_model_across_runs(tmp_path, monkeypatch):
    import user_settings
    from app_api import InvoiceAppAPI

    monkeypatch.setattr(user_settings, "_is_macos", lambda: False)
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "localappdata"))
    api = InvoiceAppAPI(revision_resolver=lambda: "b" * 40)
    first_policy = RecognitionPolicy.from_settings(
        {
            "recognition_mode": "local",
            "local_model_source": "mlx-community/Qwen3-1.7B-4bit",
            "local_model_max_tokens": 256,
        }
    )
    second_policy = RecognitionPolicy.from_settings(
        {
            "recognition_mode": "local",
            "local_model_source": "mlx-community/Qwen3-1.7B-4bit",
            "local_model_max_tokens": 512,
        }
    )

    first = api._get_or_create_local_llm_provider(first_policy)
    second = api._get_or_create_local_llm_provider(second_policy)

    assert second is first
    assert second.max_tokens == 512
    assert second.is_loaded is False
