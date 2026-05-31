import asyncio
import json
import os
from typing import Any, AsyncIterator

import httpx
from dotenv import load_dotenv
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "qwen2.5-coder:7b-instruct-q4_K_M")
DEFAULT_EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
TEMPERATURE = 0.1
SEMAPHORE_LIMIT = 2
VRAM_BUFFER_MB = 100

# Module-level semaphore — shared across all OllamaAdapter instances to cap
# concurrent VRAM usage at 2 simultaneous inference calls (6GB VRAM constraint).
_inference_semaphore = asyncio.Semaphore(SEMAPHORE_LIMIT)


async def _free_vram_mb() -> int:
    """Return free VRAM in MB from nvidia-smi, or a large sentinel if unavailable."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode == 0:
            first_line = stdout.decode().strip().split("\n")[0]
            return int(first_line.strip())
    except Exception:
        pass
    return 99_999


async def _wait_for_vram(required_mb: int = VRAM_BUFFER_MB) -> None:
    """Block until at least `required_mb` MB of VRAM is free."""
    while True:
        free = await _free_vram_mb()
        if free >= required_mb:
            return
        logger.warning(
            f"VRAM guard: {free}MB free, need {required_mb}MB — waiting 5s before retry"
        )
        await asyncio.sleep(5)


class OllamaAdapter:
    """
    Async LLM adapter for Ollama's OpenAI-compatible /v1 endpoint.

    All inference paths (generate, stream, embed) are protected by a shared
    semaphore (max 2 concurrent) and a VRAM free-memory guard.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        temperature: float = TEMPERATURE,
        max_tokens: int = 4096,
        vram_buffer_mb: int = VRAM_BUFFER_MB,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.vram_buffer_mb = vram_buffer_mb
        self._client = httpx.AsyncClient(
            base_url=f"{self.base_url}/v1",
            timeout=httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=5.0),
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        reraise=True,
    )
    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """Send a chat completion request and return the full response string."""
        async with _inference_semaphore:
            await _wait_for_vram(self.vram_buffer_mb)
            try:
                payload: dict[str, Any] = {
                    "model": model or self.model,
                    "messages": messages,
                    "temperature": temperature if temperature is not None else self.temperature,
                    "max_tokens": max_tokens or self.max_tokens,
                    "stream": False,
                }
                if response_format:
                    payload["response_format"] = response_format

                logger.debug(
                    f"LLM generate | model={payload['model']} | messages={len(messages)}"
                )
                response = await self._client.post("/chat/completions", json=payload)
                response.raise_for_status()
                data = response.json()
                content: str = data["choices"][0]["message"]["content"]
                logger.debug(f"LLM generate | response_len={len(content)}")
                return content

            except httpx.HTTPStatusError as exc:
                logger.error(
                    f"LLM generate HTTP {exc.response.status_code}: {exc.response.text[:500]}"
                )
                raise
            except Exception as exc:
                logger.error(f"LLM generate error: {exc}")
                raise

    async def stream(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream chat completion tokens.

        The semaphore is held for the entire stream duration because the model
        occupies VRAM throughout inference.
        """
        async with _inference_semaphore:
            await _wait_for_vram(self.vram_buffer_mb)
            payload: dict[str, Any] = {
                "model": model or self.model,
                "messages": messages,
                "temperature": temperature if temperature is not None else self.temperature,
                "max_tokens": max_tokens or self.max_tokens,
                "stream": True,
            }
            try:
                async with self._client.stream(
                    "POST", "/chat/completions", json=payload
                ) as response:
                    response.raise_for_status()
                    async for raw_line in response.aiter_lines():
                        if not raw_line.startswith("data: "):
                            continue
                        line_data = raw_line[6:]
                        if line_data.strip() == "[DONE]":
                            break
                        chunk = json.loads(line_data)
                        delta_content: str = (
                            chunk["choices"][0]["delta"].get("content") or ""
                        )
                        if delta_content:
                            yield delta_content
            except httpx.HTTPStatusError as exc:
                logger.error(f"LLM stream HTTP {exc.response.status_code}")
                raise
            except Exception as exc:
                logger.error(f"LLM stream error: {exc}")
                raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        reraise=True,
    )
    async def embed(
        self,
        text: str,
        model: str = DEFAULT_EMBED_MODEL,
    ) -> list[float]:
        """Return the embedding vector for `text` using the embedding model."""
        try:
            response = await self._client.post(
                "/embeddings",
                json={"model": model, "input": text},
            )
            response.raise_for_status()
            data = response.json()
            return data["data"][0]["embedding"]
        except httpx.HTTPStatusError as exc:
            logger.error(
                f"Embed HTTP {exc.response.status_code}: {exc.response.text[:300]}"
            )
            raise
        except Exception as exc:
            logger.error(f"Embed error: {exc}")
            raise

    # ──────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────────────

    async def close(self) -> None:
        await self._client.aclose()

    def ollama_v1_url(self) -> str:
        """Return the Ollama OpenAI-compatible v1 base URL."""
        return self.base_url.rstrip("/") + "/v1"

    async def __aenter__(self) -> "OllamaAdapter":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()
