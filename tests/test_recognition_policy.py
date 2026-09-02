import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from candidate_pipeline import CandidatePipeline
from extraction_pipeline import ExtractionOutcome
from recognition_policy import (
    CloudAccessDenied,
    CloudProviderId,
    RecognitionMode,
    RecognitionPolicy,
    RecognitionPolicyError,
)
from recognition_router import ModeAwareRecognitionExtractor


def _candidate(path: str = "/tmp/invoice.pdf"):
    return CandidatePipeline().collect(
        [{"filepath": path, "message_uid": "mail-1", "subject": "invoice"}]
    )[0]


def test_missing_or_invalid_mode_fails_closed_to_local():
    missing = RecognitionPolicy.from_settings({})
    invalid = RecognitionPolicy.from_settings(
        {
            "recognition_mode": "automatic-cloud-fallback",
            "cloud_provider": "glm",
            "local_model_max_tokens": 999999,
            "local_confidence_threshold": "1.5",
        }
    )

    assert missing.mode is RecognitionMode.LOCAL
    assert missing.cloud_provider is None
    assert invalid.mode is RecognitionMode.LOCAL
    assert invalid.cloud_provider is CloudProviderId.GLM
    assert invalid.local_model_max_tokens == 768
    assert str(invalid.local_confidence_threshold) == "0.80"
    invalid.validate_for_admission(credentials={})
    assert str(
        RecognitionPolicy.from_settings(
            {"local_confidence_threshold": "NaN"}
        ).local_confidence_threshold
    ) == "0.80"


def test_cloud_policy_requires_provider_and_matching_credential():
    with pytest.raises(RecognitionPolicyError) as missing_provider:
        RecognitionPolicy.from_settings(
            {"recognition_mode": "cloud"}
        ).validate_for_admission(credentials={})
    assert missing_provider.value.reason_code == "CLOUD_PROVIDER_REQUIRED"

    policy = RecognitionPolicy.from_settings(
        {"recognition_mode": "cloud", "cloud_provider": "glm"}
    )
    with pytest.raises(RecognitionPolicyError) as missing_key:
        policy.validate_for_admission(credentials={CloudProviderId.GLM: False})
    assert missing_key.value.reason_code == "CLOUD_CREDENTIAL_REQUIRED"

    policy.validate_for_admission(credentials={CloudProviderId.GLM: True})
    assert policy.assert_cloud_allowed("glm") is CloudProviderId.GLM
    with pytest.raises(CloudAccessDenied):
        policy.assert_cloud_allowed("deepseek")


def test_local_router_never_calls_cloud_extractor():
    candidate = _candidate()
    cloud_calls = []

    router = ModeAwareRecognitionExtractor(
        policy=RecognitionPolicy.from_settings(
            {
                "recognition_mode": "local",
                "cloud_provider": "glm",
            }
        ),
        cloud_extractors={CloudProviderId.GLM: lambda item: cloud_calls.append(item)},
        artifact_path_resolver=lambda _item: "/retained/invoice.pdf",
    )

    outcome = router(candidate)

    assert isinstance(outcome, ExtractionOutcome)
    assert outcome.status == "manual_review"
    assert outcome.reason_code == "LOCAL_RECOGNITION_NOT_READY"
    assert outcome.artifact_path == "/retained/invoice.pdf"
    assert cloud_calls == []
    assert router.verified_ceiling() == 1


def test_hybrid_cannot_skip_unfinished_local_stage():
    candidate = _candidate()
    cloud_calls = []
    policy = RecognitionPolicy.from_settings(
        {"recognition_mode": "hybrid", "cloud_provider": "glm"}
    )
    router = ModeAwareRecognitionExtractor(
        policy=policy,
        cloud_extractors={"glm": lambda item: cloud_calls.append(item)},
    )

    outcome = router(candidate)

    assert outcome.status == "manual_review"
    assert outcome.reason_code == "LOCAL_RECOGNITION_NOT_READY"
    assert cloud_calls == []


def test_explicit_cloud_router_calls_only_selected_provider():
    candidate = _candidate()
    calls = []
    expected = ExtractionOutcome.resolved(candidate, {"info_json": {}})

    class GlmExtractor:
        @staticmethod
        def verified_ceiling():
            return 2

        def __call__(self, item):
            calls.append(("glm", item))
            return expected

    router = ModeAwareRecognitionExtractor(
        policy=RecognitionPolicy.from_settings(
            {"recognition_mode": "cloud", "cloud_provider": "glm"}
        ),
        cloud_extractors={
            "glm": GlmExtractor(),
            "deepseek": lambda item: calls.append(("deepseek", item)),
        },
    )

    assert router(candidate) is expected
    assert calls == [("glm", candidate)]
    assert router.verified_ceiling() == 2


class _MemorySettingsStore:
    def __init__(self, path: Path, values=None):
        self.settings_path = str(path)
        self.values = dict(values or {})

    def load(self):
        return dict(self.values)

    def save(self, values):
        self.values = dict(values)
        return self.settings_path

    def clear(self):
        self.values = {}


def test_run_dependencies_pass_frozen_policy_to_extraction_session(
    tmp_path, monkeypatch
):
    import invoice_extractor
    import user_settings
    from app_api import InvoiceAppAPI
    from run_coordinator import RunRequest

    monkeypatch.setattr(user_settings, "_is_macos", lambda: False)
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "localappdata"))
    api = InvoiceAppAPI(revision_resolver=lambda: "c" * 40)
    api._settings_store = _MemorySettingsStore(tmp_path / "settings.json")
    policy = RecognitionPolicy.from_settings({"recognition_mode": "local"})
    captured = {}

    class ExtractorStub:
        def __init__(self, api_key="", output_dir="", **_kwargs):
            self.api_key = api_key
            self.output_dir = output_dir

    class SessionStub:
        @staticmethod
        def extract():
            return []

    def create_session(*_args, **kwargs):
        captured.update(kwargs)
        return SessionStub()

    monkeypatch.setattr(invoice_extractor, "InvoiceExtractor", ExtractorStub)
    monkeypatch.setattr(api, "_create_processing_pipeline_session", create_session)
    request = RunRequest(
        run_id="local-policy",
        date_from="2026-08-01",
        date_to="2026-09-01",
        save_path=str(tmp_path / "output"),
        rules_text="",
        account_id="account",
        channel_id="qq",
    )

    dependencies = api._build_run_dependencies(
        request,
        email_address="local@qq.com",
        auth_code="mail-auth",
        api_key="",
        recognition_policy=policy,
    )

    assert dependencies.extract([], request) == []
    assert captured["_recognition_policy"] is policy


def test_local_app_admission_does_not_require_api_key(tmp_path, monkeypatch):
    import user_settings
    from app_api import InvoiceAppAPI

    monkeypatch.setattr(user_settings, "_is_macos", lambda: False)
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "localappdata"))
    api = InvoiceAppAPI(revision_resolver=lambda: "a" * 40)
    api._settings_store = _MemorySettingsStore(tmp_path / "settings.json")
    monkeypatch.setattr(api, "get_default_save_path", lambda: str(tmp_path / "output"))
    captured = {}
    entered = threading.Event()
    release = threading.Event()

    def build_dependencies(_request, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(report_service=SimpleNamespace(close=lambda: None))

    def complete_worker(_request, handle, _dependencies):
        entered.set()
        assert release.wait(2)
        handle.finalize([])

    monkeypatch.setattr(api, "_build_run_dependencies", build_dependencies)
    monkeypatch.setattr(api, "_processing_worker", complete_worker)
    monkeypatch.setattr(api, "_start_truth_audit_async", lambda *_args: None)
    monkeypatch.setattr(api, "_safe_write_run_config", lambda *_args, **_kwargs: None)

    result = api.start_processing(
        "",
        str(tmp_path / "output"),
        "2026-08-01",
        "2026-09-01",
        email_address="local@qq.com",
        auth_code="mail-auth",
        api_key="",
    )
    assert entered.wait(1)
    worker = api._worker_thread

    assert result == {"success": True, "message": "任务已启动"}
    assert captured["api_key"] == ""
    assert captured["recognition_policy"].mode is RecognitionMode.LOCAL
    assert api._settings_store.values["recognition_mode"] == "local"
    release.set()
    worker.join(1)


def test_cloud_app_admission_rejects_missing_api_key(tmp_path, monkeypatch):
    import user_settings
    from app_api import InvoiceAppAPI

    monkeypatch.setattr(user_settings, "_is_macos", lambda: False)
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "localappdata"))
    api = InvoiceAppAPI(revision_resolver=lambda: "b" * 40)
    api._settings_store = _MemorySettingsStore(
        tmp_path / "settings.json",
        {"recognition_mode": "cloud", "cloud_provider": "glm"},
    )
    monkeypatch.setattr(api, "_cleanup_temp_folders", lambda **_kwargs: None)
    monkeypatch.setattr(
        api,
        "_build_run_dependencies",
        lambda *_args, **_kwargs: pytest.fail("dependencies must not be built"),
    )

    result = api.start_processing(
        "",
        str(tmp_path / "output"),
        "2026-08-01",
        "2026-09-01",
        email_address="cloud@qq.com",
        auth_code="mail-auth",
        api_key="",
    )

    assert result == {
        "success": False,
        "message": "Cloud 模式需要 GLM API Key。",
    }
