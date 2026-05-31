"""Shared LLM model construction for main and fallback agents."""

from dataclasses import dataclass
from typing import Any, Optional

from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.deepseek import DeepSeekProvider
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.openai import OpenAIProvider


@dataclass(frozen=True)
class ModelEndpoint:
    """Normalized endpoint identity before pydantic-ai model construction."""

    provider: str
    model_name: str
    api_format: str
    vendor: str
    api_key: Optional[str]
    base_url: Optional[str]


class VendorAdapter:
    """Provider-specific model-setting compatibility shim."""

    def apply(self, settings: dict[str, Any], config, endpoint: ModelEndpoint) -> None:
        llm = config.llm
        effort = (llm.effort or "").strip().lower()
        if not effort:
            return
        if endpoint.api_format == "openai":
            settings["openai_reasoning_effort"] = effort
        elif endpoint.api_format == "anthropic":
            settings["thinking"] = effort


class DeepSeekV4Adapter(VendorAdapter):
    """DeepSeek V4 thinking controls differ by compatibility API format."""

    @staticmethod
    def effort(value: str) -> str:
        normalized = (value or "").strip().lower()
        if normalized in {"max", "xhigh"}:
            return "max"
        return "high"

    def apply(self, settings: dict[str, Any], config, endpoint: ModelEndpoint) -> None:
        llm = config.llm
        extra_body = settings.setdefault("extra_body", {})
        effort = (llm.effort or "").strip().lower()
        thinking = (llm.thinking or "").strip().lower()

        if endpoint.api_format == "anthropic":
            if effort:
                extra_body["output_config"] = {"effort": self.effort(effort)}
        elif effort:
            settings["openai_reasoning_effort"] = self.effort(effort)

        if thinking in {"enabled", "disabled"}:
            extra_body["thinking"] = {"type": thinking}
        elif "thinking" not in extra_body:
            # DeepSeek V4 defaults to thinking enabled; keep that default explicit
            # so both OpenAI- and Anthropic-compatible paths behave identically.
            extra_body["thinking"] = {"type": "enabled"}


def _detect_api_format(provider: str, base_url: Optional[str]) -> str:
    url = (base_url or "").lower()
    if provider == "anthropic" or "/anthropic" in url:
        return "anthropic"
    if provider in {"openai", "deepseek"} or base_url:
        return "openai"
    return provider


def _detect_vendor(provider: str, model_name: str, base_url: Optional[str]) -> str:
    model = (model_name or "").lower()
    url = (base_url or "").lower()
    if provider == "deepseek" or "api.deepseek.com" in url or model.startswith("deepseek-"):
        return "deepseek"
    return provider


def _endpoint(config, model_override: Optional[str] = None) -> ModelEndpoint:
    llm = config.llm
    provider = (llm.provider or "").lower()
    model_name = model_override or llm.model
    base_url = llm.base_url or None
    return ModelEndpoint(
        provider=provider,
        model_name=model_name,
        api_format=_detect_api_format(provider, base_url),
        vendor=_detect_vendor(provider, model_name, base_url),
        api_key=llm.api_key or None,
        base_url=base_url,
    )


def _adapter_for(endpoint: ModelEndpoint) -> VendorAdapter:
    if endpoint.vendor == "deepseek" and endpoint.model_name.lower().startswith("deepseek-v4"):
        return DeepSeekV4Adapter()
    return VendorAdapter()


def build_model_settings(config, model_override: Optional[str] = None) -> dict[str, Any]:
    """Build pydantic-ai ModelSettings shared by all local agents."""
    endpoint = _endpoint(config, model_override=model_override)
    settings: dict[str, Any] = {}

    if config.llm.max_tokens:
        settings["max_tokens"] = config.llm.max_tokens
    if config.llm.extra_body:
        settings["extra_body"] = dict(config.llm.extra_body)

    _adapter_for(endpoint).apply(settings, config, endpoint)

    if not settings.get("extra_body"):
        settings.pop("extra_body", None)
    return settings


def build_model(config, model_override: Optional[str] = None):
    """Build LLM model based on configuration."""
    endpoint = _endpoint(config, model_override=model_override)
    settings = build_model_settings(config, model_override=model_override)

    if endpoint.provider == "anthropic":
        if endpoint.api_key or endpoint.base_url:
            return AnthropicModel(
                endpoint.model_name,
                provider=AnthropicProvider(
                    api_key=endpoint.api_key,
                    base_url=endpoint.base_url,
                ),
                settings=settings,
            )
        return AnthropicModel(endpoint.model_name, settings=settings)
    if endpoint.provider == "openai":
        if endpoint.api_key or endpoint.base_url:
            return OpenAIChatModel(
                endpoint.model_name,
                provider=OpenAIProvider(
                    api_key=endpoint.api_key,
                    base_url=endpoint.base_url,
                ),
                settings=settings,
            )
        return OpenAIChatModel(endpoint.model_name, settings=settings)
    if endpoint.provider == "deepseek":
        return OpenAIChatModel(
            endpoint.model_name,
            provider=DeepSeekProvider(api_key=endpoint.api_key),
            settings=settings,
        )
    if endpoint.provider == "gemini":
        if endpoint.api_key or endpoint.base_url:
            return GoogleModel(
                endpoint.model_name,
                provider=GoogleProvider(
                    api_key=endpoint.api_key,
                    base_url=endpoint.base_url,
                ),
                settings=settings,
            )
        return GoogleModel(endpoint.model_name, settings=settings)
    raise ValueError(f"Unsupported LLM provider: {endpoint.provider}")
