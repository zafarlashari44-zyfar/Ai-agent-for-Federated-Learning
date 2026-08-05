from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from ollama import Client

from reasoning_pipeline.narrative.configuration import (
    NarrativeConfiguration,
)


class NarrativeModelClient(Protocol):
    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
    ) -> str:
        ...


class OllamaClientAdapter:
    RESPONSE_SCHEMA: dict[str, Any] = {
        "type": "object",
        "properties": {
            "doctor_report": {
                "type": "string",
                "minLength": 1,
            },
            "next_of_kin_summary": {
                "type": "string",
                "minLength": 1,
            },
        },
        "required": [
            "doctor_report",
            "next_of_kin_summary",
        ],
        "additionalProperties": False,
    }

    def __init__(
        self,
        configuration: NarrativeConfiguration,
    ) -> None:
        self._configuration = configuration
        self._client = Client(
            host=configuration.host,
            timeout=configuration.timeout_seconds,
        )

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
    ) -> str:
        response = self._client.chat(
            model=self._configuration.model_name,
            messages=[
                {
                    "role": message["role"],
                    "content": message["content"],
                }
                for message in messages
            ],
            format=self.RESPONSE_SCHEMA,
            options={
                "temperature": self._configuration.temperature,
            },
            stream=False,
        )

        content = response.message.content

        if content is None:
            raise RuntimeError(
                "Ollama returned an empty message."
            )

        return content
