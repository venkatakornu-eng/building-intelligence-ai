from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


OLLAMA_URL = "http://localhost:11434/api/chat"


class BaseOllamaAgent:
    """
    Shared Ollama communication layer for specialised agents.
    """

    def __init__(
        self,
        model: str = "deepseek-r1:1.5b",
        ollama_url: str = OLLAMA_URL,
        timeout: int = 120,
    ) -> None:
        self.model = model
        self.ollama_url = ollama_url
        self.timeout = timeout

    def _request_json(
        self,
        system_prompt: str,
        user_query: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Send a structured-output request to Ollama.
        """

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_query,
                },
            ],
            "stream": False,
            "format": schema,
            "options": {
                "temperature": 0,
            },
        }

        request = urllib.request.Request(
            self.ollama_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout,
            ) as response:
                response_data = json.loads(
                    response.read().decode("utf-8")
                )

        except urllib.error.HTTPError as error:
            error_message = error.read().decode(
                "utf-8",
                errors="replace",
            )

            raise ConnectionError(
                f"Ollama returned HTTP {error.code}: "
                f"{error_message}"
            ) from error

        except urllib.error.URLError as error:
            raise ConnectionError(
                "Could not connect to Ollama. Make sure "
                "Ollama is installed and running."
            ) from error

        except TimeoutError as error:
            raise TimeoutError(
                "The local model took too long to respond."
            ) from error

        message = response_data.get("message", {})
        content = message.get("content")

        if not content:
            raise ValueError(
                "Ollama returned an empty response."
            )

        try:
            parsed_output = json.loads(content)

        except json.JSONDecodeError as error:
            raise ValueError(
                "The agent did not return valid JSON."
            ) from error

        if not isinstance(parsed_output, dict):
            raise TypeError(
                "The agent response must be a JSON object."
            )

        return parsed_output