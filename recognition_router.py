from __future__ import annotations

from typing import Any, Callable, Mapping

from candidate_pipeline import DocumentCandidate
from extraction_pipeline import ExtractionOutcome
from recognition_policy import (
    CloudAccessDenied,
    CloudProviderId,
    RecognitionMode,
    RecognitionPolicy,
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
