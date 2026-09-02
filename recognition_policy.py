from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Collection, Mapping


DEFAULT_LOCAL_MODEL_SOURCE = "mlx-community/Qwen3-1.7B-4bit"
DEFAULT_LOCAL_MODEL_MAX_TOKENS = 768
DEFAULT_LOCAL_CONFIDENCE_THRESHOLD = Decimal("0.80")
MIN_LOCAL_MODEL_MAX_TOKENS = 64
MAX_LOCAL_MODEL_MAX_TOKENS = 2048


class RecognitionMode(str, Enum):
    LOCAL = "local"
    HYBRID = "hybrid"
    CLOUD = "cloud"

    @classmethod
    def parse(cls, value: object) -> "RecognitionMode":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value or "").strip().lower())
        except ValueError:
            return cls.LOCAL


class CloudProviderId(str, Enum):
    GLM = "glm"
    DEEPSEEK = "deepseek"

    @classmethod
    def parse(cls, value: object) -> "CloudProviderId | None":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value or "").strip().lower())
        except ValueError:
            return None


class RecognitionPolicyError(ValueError):
    def __init__(self, reason_code: str, user_message: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code
        self.user_message = user_message


class CloudAccessDenied(PermissionError):
    reason_code = "CLOUD_ACCESS_DENIED"
    user_message = "当前识别模式不允许调用该云端服务。"

    def __init__(self, provider: object = "") -> None:
        provider_id = str(getattr(provider, "value", provider) or "").strip()
        super().__init__(
            f"{self.reason_code}:{provider_id}" if provider_id else self.reason_code
        )


def _bounded_max_tokens(value: object) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return DEFAULT_LOCAL_MODEL_MAX_TOKENS
    if not MIN_LOCAL_MODEL_MAX_TOKENS <= parsed <= MAX_LOCAL_MODEL_MAX_TOKENS:
        return DEFAULT_LOCAL_MODEL_MAX_TOKENS
    return parsed


def _confidence_threshold(value: object) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return DEFAULT_LOCAL_CONFIDENCE_THRESHOLD
    if not parsed.is_finite() or not Decimal("0") <= parsed <= Decimal("1"):
        return DEFAULT_LOCAL_CONFIDENCE_THRESHOLD
    return parsed


@dataclass(frozen=True)
class RecognitionPolicy:
    mode: RecognitionMode = RecognitionMode.LOCAL
    cloud_provider: CloudProviderId | None = None
    local_model_source: str = DEFAULT_LOCAL_MODEL_SOURCE
    local_model_max_tokens: int = DEFAULT_LOCAL_MODEL_MAX_TOKENS
    local_confidence_threshold: Decimal = DEFAULT_LOCAL_CONFIDENCE_THRESHOLD

    @classmethod
    def from_settings(cls, settings: Mapping[str, object] | None) -> "RecognitionPolicy":
        values = settings or {}
        model_source = str(
            values.get("local_model_source") or DEFAULT_LOCAL_MODEL_SOURCE
        ).strip()
        return cls(
            mode=RecognitionMode.parse(values.get("recognition_mode")),
            cloud_provider=CloudProviderId.parse(values.get("cloud_provider")),
            local_model_source=model_source or DEFAULT_LOCAL_MODEL_SOURCE,
            local_model_max_tokens=_bounded_max_tokens(
                values.get("local_model_max_tokens")
            ),
            local_confidence_threshold=_confidence_threshold(
                values.get("local_confidence_threshold")
            ),
        )

    @property
    def uses_local_recognition(self) -> bool:
        return self.mode in {RecognitionMode.LOCAL, RecognitionMode.HYBRID}

    @property
    def allows_cloud_calls(self) -> bool:
        return self.mode in {RecognitionMode.HYBRID, RecognitionMode.CLOUD}

    def to_settings(self) -> dict[str, object]:
        return {
            "recognition_mode": self.mode.value,
            "cloud_provider": (
                self.cloud_provider.value if self.cloud_provider is not None else ""
            ),
            "local_model_source": self.local_model_source,
            "local_model_max_tokens": self.local_model_max_tokens,
            "local_confidence_threshold": str(self.local_confidence_threshold),
        }

    def validate_for_admission(
        self,
        *,
        credentials: Mapping[object, bool] | None = None,
        supported_cloud_providers: Collection[object] = (CloudProviderId.GLM,),
    ) -> None:
        if not self.allows_cloud_calls:
            return
        if self.cloud_provider is None:
            raise RecognitionPolicyError(
                "CLOUD_PROVIDER_REQUIRED",
                "Hybrid 或 Cloud 模式需要选择云端识别服务。",
            )

        supported = {
            provider
            for value in supported_cloud_providers
            if (provider := CloudProviderId.parse(value)) is not None
        }
        if self.cloud_provider not in supported:
            raise RecognitionPolicyError(
                "CLOUD_PROVIDER_UNAVAILABLE",
                "当前桌面版尚不支持所选云端识别服务。",
            )

        available_credentials = {
            provider: bool(present)
            for key, present in (credentials or {}).items()
            if (provider := CloudProviderId.parse(key)) is not None
        }
        if not available_credentials.get(self.cloud_provider, False):
            provider_name = (
                "GLM" if self.cloud_provider is CloudProviderId.GLM else "DeepSeek"
            )
            raise RecognitionPolicyError(
                "CLOUD_CREDENTIAL_REQUIRED",
                f"{self.mode.value.title()} 模式需要 {provider_name} API Key。",
            )

    def assert_cloud_allowed(self, provider: object) -> CloudProviderId:
        provider_id = CloudProviderId.parse(provider)
        if (
            not self.allows_cloud_calls
            or provider_id is None
            or provider_id is not self.cloud_provider
        ):
            raise CloudAccessDenied(provider)
        return provider_id
