from __future__ import annotations

from typing import Any, Callable, Mapping

from candidate_pipeline import DocumentCandidate
from extraction_pipeline import ExtractionOutcome
from local_llm_provider import (
    LocalLLMProviderError,
    grounded_deterministic_fields,
    merge_invoice_fields,
)
from local_text_extractor import TextAcquisitionStatus
from recognition_policy import (
    CloudAccessDenied,
    CloudProviderId,
    RecognitionMode,
    RecognitionPolicy,
)


class LocalEvidenceRecognitionExtractor:
    """Resolve prepared local text with a run-owned text-only provider."""

    def __init__(
        self,
        *,
        provider: Any,
        owner_extractor: Any,
        sidecar: Mapping[str, Mapping[str, Any]],
        sidecar_lock: Any,
        artifact_path_resolver: Callable[[DocumentCandidate], str] | None = None,
    ) -> None:
        self.provider = provider
        self.owner_extractor = owner_extractor
        self.sidecar = sidecar
        self.sidecar_lock = sidecar_lock
        self.artifact_path_resolver = artifact_path_resolver or (
            lambda candidate: candidate.source_path
        )

    def verified_ceiling(self) -> int:
        return 1

    def _review(
        self,
        candidate: DocumentCandidate,
        reason_code: str,
        *,
        trace_context: Mapping[str, Any] | None = None,
    ) -> ExtractionOutcome:
        review_trace = dict(candidate.trace_context)
        if trace_context:
            review_trace["local_recognition"] = dict(trace_context)
        return ExtractionOutcome(
            candidate=candidate,
            status="manual_review",
            reason_code=reason_code,
            message=reason_code,
            artifact_path=str(
                self.artifact_path_resolver(candidate) or candidate.source_path
            ),
            trace_context=review_trace,
        )

    @staticmethod
    def _acquired(acquisition: Any) -> bool:
        status = getattr(acquisition, "status", None)
        return status is TextAcquisitionStatus.ACQUIRED or getattr(
            status, "value", status
        ) == TextAcquisitionStatus.ACQUIRED.value

    def __call__(self, candidate: DocumentCandidate) -> ExtractionOutcome:
        with self.sidecar_lock:
            prepared = dict(self.sidecar.get(candidate.identity.document_id, {}) or {})
        pdf_path = str(prepared.get("pdf_path") or candidate.source_path)
        metadata = dict(prepared.get("metadata") or candidate.to_legacy())
        acquisition = prepared.get("text_acquisition")
        if not self._acquired(acquisition):
            return self._review(
                candidate,
                str(
                    getattr(acquisition, "reason_code", "")
                    or "LOCAL_TEXT_EVIDENCE_UNAVAILABLE"
                ),
            )
        evidence = getattr(acquisition, "evidence", None)
        if evidence is None:
            return self._review(candidate, "LOCAL_TEXT_EVIDENCE_UNAVAILABLE")

        deterministic = {}
        deterministic.update(dict(prepared.get("deterministic_fields") or {}))
        deterministic.update(dict(metadata.get("provider_recovered_fields") or {}))
        deterministic = grounded_deterministic_fields(deterministic, evidence.text)
        try:
            model_result = self.provider.extract(
                evidence, document_context=metadata
            )
        except LocalLLMProviderError as exc:
            return self._review(candidate, exc.reason_code)
        except Exception:
            return self._review(candidate, "LOCAL_MODEL_FAILED")

        merged = merge_invoice_fields(deterministic, model_result.payload)
        trace = {
            "engine": "local_mlx_qwen",
            "reason_code": "LOCAL_LLM_STRUCTURED_RESULT",
            "provider": "local_mlx",
            "evidence_source": getattr(evidence.source, "value", str(evidence.source)),
            "average_confidence": (
                str(evidence.average_confidence)
                if evidence.average_confidence is not None
                else None
            ),
            "field_provenance": merged.trace_provenance(),
            "conflict_fields": list(merged.conflicts),
        }
        if merged.conflicts:
            return self._review(
                candidate, "LOCAL_FIELD_CONFLICT", trace_context=trace
            )
        try:
            info_json = self.owner_extractor._adapt_extraction_result(
                dict(merged.payload),
                pdf_path=pdf_path,
                document_context=metadata,
            )
        except Exception:
            return self._review(
                candidate, "LOCAL_MODEL_SCHEMA_ADAPTATION_FAILED", trace_context=trace
            )
        if not isinstance(info_json, dict):
            return self._review(
                candidate, "LOCAL_MODEL_SCHEMA_ADAPTATION_FAILED", trace_context=trace
            )
        return ExtractionOutcome.resolved(
            candidate,
            {
                "pdf_path": pdf_path,
                "metadata": metadata,
                "info_json": info_json,
                "extraction_trace": trace,
                "extraction_timing": {},
            },
        )


class ModeAwareRecognitionExtractor:
    """Route unresolved candidates through an immutable, fail-closed policy."""

    def __init__(
        self,
        *,
        policy: RecognitionPolicy,
        cloud_extractors: Mapping[object, Callable[[DocumentCandidate], Any]],
        local_extractor: Callable[[DocumentCandidate], Any] | None = None,
        artifact_path_resolver: Callable[[DocumentCandidate], str] | None = None,
    ) -> None:
        self.policy = policy
        self.local_extractor = local_extractor
        self.artifact_path_resolver = artifact_path_resolver or (
            lambda candidate: candidate.source_path
        )
        self.cloud_extractors = {
            provider: extractor
            for key, extractor in cloud_extractors.items()
            if (provider := CloudProviderId.parse(key)) is not None
        }

    def verified_ceiling(self) -> int:
        if self.policy.mode is RecognitionMode.LOCAL or self.local_extractor is None:
            if self.policy.mode is not RecognitionMode.CLOUD:
                return 1
        provider = self.policy.cloud_provider
        extractor = self.cloud_extractors.get(provider)
        ceiling = getattr(extractor, "verified_ceiling", None)
        if not callable(ceiling):
            return 1
        try:
            return max(1, min(2, int(ceiling())))
        except (TypeError, ValueError, OverflowError):
            return 1

    def _pending_local(self, candidate: DocumentCandidate) -> ExtractionOutcome:
        return ExtractionOutcome(
            candidate=candidate,
            status="manual_review",
            reason_code="LOCAL_RECOGNITION_NOT_READY",
            message="LOCAL_RECOGNITION_NOT_READY",
            artifact_path=str(
                self.artifact_path_resolver(candidate) or candidate.source_path
            ),
        )

    def _run_cloud(self, candidate: DocumentCandidate) -> Any:
        try:
            provider = self.policy.assert_cloud_allowed(self.policy.cloud_provider)
        except CloudAccessDenied:
            return ExtractionOutcome(
                candidate=candidate,
                status="manual_review",
                reason_code="CLOUD_ACCESS_DENIED",
                message="CLOUD_ACCESS_DENIED",
                artifact_path=str(
                    self.artifact_path_resolver(candidate) or candidate.source_path
                ),
            )
        extractor = self.cloud_extractors.get(provider)
        if extractor is None:
            return ExtractionOutcome(
                candidate=candidate,
                status="manual_review",
                reason_code="CLOUD_PROVIDER_UNAVAILABLE",
                message="CLOUD_PROVIDER_UNAVAILABLE",
                artifact_path=str(
                    self.artifact_path_resolver(candidate) or candidate.source_path
                ),
            )
        return extractor(candidate)

    def __call__(self, candidate: DocumentCandidate) -> Any:
        if self.policy.mode is RecognitionMode.CLOUD:
            return self._run_cloud(candidate)

        if self.local_extractor is None:
            # Hybrid cannot skip the unfinished local stage and silently become Cloud.
            return self._pending_local(candidate)

        local_outcome = self.local_extractor(candidate)
        if self.policy.mode is RecognitionMode.LOCAL:
            return local_outcome
        if isinstance(local_outcome, ExtractionOutcome) and local_outcome.status == "resolved":
            return local_outcome
        return self._run_cloud(candidate)
