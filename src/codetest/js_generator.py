"""JSTestGenerator — Sprint 9. Ollama → self-contained Vitest test code."""
from __future__ import annotations

import asyncio
from typing import Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict

from src.codetest.js_ast_parser import JSFunctionSpec
from src.llm.instructor_client import InstructorClient, StructuredGenerationError

_JS_SEMAPHORE = asyncio.Semaphore(1)
_CODER_MODEL = "qwen2.5-coder:7b-instruct-q4_K_M"

_JS_SYSTEM_PROMPT = (
    "You are a TypeScript/Vitest test engineer generating self-contained test files. "
    "STRICT RULES: "
    "(1) test_code MUST begin with a complete TypeScript implementation of the function under test — "
    "    infer it from the function signature, jsdoc, and complexity; "
    "(2) after the implementation, import only from 'vitest': "
    "    import { describe, it, expect } from 'vitest'; "
    "(3) wrap tests in describe()/it() blocks; "
    "(4) every it() block must call expect(); "
    "(5) NO setTimeout or setInterval in test body; "
    "(6) ALL async tests use async/await, never .then(); "
    "(7) for test_type='metamorphic', provide a non-null metamorphic_relation; "
    "Return ONLY valid JSON."
)


class GeneratedJSTest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    test_id: str
    func_id: str
    test_code: str
    test_type: Literal["happy_path", "edge_case", "metamorphic"]
    metamorphic_relation: str | None


def _build_prompt(spec: JSFunctionSpec) -> str:
    params_desc = ", ".join(spec.params) if spec.params else "(none)"
    ret = spec.return_type or "unknown"
    jsdoc = spec.jsdoc or "No documentation."
    meta_hint = (
        "\nIMPORTANT: complexity is high — use test_type='metamorphic' and provide "
        "a non-null, falsifiable metamorphic_relation (e.g. 'f(x+1) >= f(x) for positive x')."
        if spec.complexity >= 2
        else "\nUse test_type='happy_path' or 'edge_case'. metamorphic_relation must be null."
    )
    return (
        f"Function: {spec.func_name}({params_desc}) -> {ret}\n"
        f"Async: {spec.is_async}\n"
        f"JSDoc: {jsdoc}\n"
        f"Complexity: {spec.complexity}\n"
        f"test_id should be 'test_{spec.func_name}'.\n"
        f"Begin test_code with the TypeScript implementation, then vitest tests."
        f"{meta_hint}"
    )


class JSTestGenerator:
    """Generate self-contained Vitest test files from JSFunctionSpec objects via Ollama."""

    def __init__(self, model: str = _CODER_MODEL) -> None:
        self._model = model

    async def generate(self, specs: list[JSFunctionSpec]) -> list[GeneratedJSTest]:
        if not specs:
            return []
        results: list[GeneratedJSTest] = []
        client = InstructorClient(model=self._model)
        try:
            for spec in specs:
                test = await self._generate_one(client, spec)
                if test is not None:
                    results.append(test)
        finally:
            await client.close()
        logger.info(f"JSTestGenerator: generated {len(results)}/{len(specs)}")
        return results

    async def _generate_one(
        self, client: InstructorClient, spec: JSFunctionSpec
    ) -> GeneratedJSTest | None:
        messages = [
            {"role": "system", "content": _JS_SYSTEM_PROMPT},
            {"role": "user", "content": _build_prompt(spec)},
        ]
        async with _JS_SEMAPHORE:
            try:
                test = await client.create_structured(
                    prompt=messages,
                    response_model=GeneratedJSTest,
                    temperature=0.2,
                )
                return test.model_copy(update={"func_id": spec.func_id})
            except StructuredGenerationError as exc:
                logger.warning(f"JSTestGenerator: failed for {spec.func_name!r}: {exc!r}")
                return None


__all__ = ["GeneratedJSTest", "JSTestGenerator"]
