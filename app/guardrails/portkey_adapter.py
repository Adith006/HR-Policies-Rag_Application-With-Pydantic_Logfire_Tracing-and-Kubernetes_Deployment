
from collections.abc import AsyncIterator

from nemoguardrails.types import (
    ChatMessage,
    LLMModel,
    LLMResponse,
    LLMResponseChunk,
)


class PortkeyLLMAdapter:
    """Adapt the existing Portkey client to NeMo's LLMModel interface."""

    def __init__(self, client, model: str):
        self.client = client
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return "portkey"

    @property
    def provider_url(self) -> str:
        return "https://api.portkey.ai"

    async def generate_async(
        self,
        prompt: str | list[ChatMessage],
        *,
        stop: list[str] | None = None,
        **kwargs,
    ) -> LLMResponse:
        import asyncio

        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = [
                {
                    "role": message.role.value
                    if hasattr(message.role, "value")
                    else message.role,
                    "content": message.content or "",
                }
                for message in prompt
            ]

        request_args = {
            "model": self._model,
            "messages": messages,
            "max_tokens": kwargs.pop("max_tokens", 512),
            **kwargs,
        }
        if stop:
            request_args["stop"] = stop

        response = await asyncio.to_thread(
            self.client.chat.completions.create,
            **request_args,
        )

        content = response.choices[0].message.content or ""
        return LLMResponse(content=content, model=self._model)

    async def stream_async(
        self,
        prompt: str | list[ChatMessage],
        *,
        stop: list[str] | None = None,
        **kwargs,
    ) -> AsyncIterator[LLMResponseChunk]:
        raise NotImplementedError(
            "Streaming is not implemented for this Portkey adapter."
        )
