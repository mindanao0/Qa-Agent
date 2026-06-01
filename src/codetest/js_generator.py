"""JSTestGenerator — Sprint 9. Ollama → self-contained Vitest test code."""
from __future__ import annotations

import asyncio
import pathlib
import re
from typing import Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict

from src.codetest.js_ast_parser import JSFunctionSpec
from src.llm.instructor_client import InstructorClient, StructuredGenerationError

_JS_SEMAPHORE = asyncio.Semaphore(1)
_CODER_MODEL = "qwen2.5-coder:7b-instruct-q4_K_M"

_JS_SYSTEM_PROMPT = (
    "You are a TypeScript/Vitest test engineer. "
    "test_code MUST have ALL THREE parts — EXAMPLE:\n"
    "```\n"
    "function add(a, b) { return a + b; }\n"
    "import { describe, it, expect } from \"vitest\";\n"
    "describe(\"add\", () => {\n"
    "  it(\"adds two numbers\", () => { expect(add(1, 2)).toBe(3); });\n"
    "  it(\"handles negatives\", () => { expect(add(-1, 1)).toBe(0); });\n"
    "});\n"
    "```\n"
    "RULES: (1) Copy implementation VERBATIM as shown above; "
    "(2) ALWAYS add: import { describe, it, expect } from \"vitest\"; "
    "(3) ALWAYS wrap tests in describe()/it() blocks after the import; "
    "(4) every it() MUST call expect(); "
    "(5) NO setTimeout or setInterval; "
    "(6) metamorphic tests need non-null metamorphic_relation. "
    "Return ONLY valid JSON."
)


def _read_func_source(spec: JSFunctionSpec) -> str | None:
    """Extract the actual TypeScript function source from the .ts file."""
    try:
        path = pathlib.Path(spec.module_path)
        if not path.exists():
            return None
        content = path.read_text(encoding="utf-8")
        pattern = (
            rf'(?:export\s+)?(?:async\s+)?function\s+{re.escape(spec.func_name)}\b'
            r'|'
            rf'export\s+(?:const|let)\s+{re.escape(spec.func_name)}\s*='
        )
        match = re.search(pattern, content)
        if not match:
            return None
        # Find the 'export' keyword if it precedes the match
        start = match.start()
        # Back up to capture 'export' prefix + JSDoc if present
        pre = content[:start].rstrip()
        # Walk back to find /** ... */ jsdoc block
        if pre.endswith("*/"):
            jdoc_start = pre.rfind("/**")
            if jdoc_start != -1:
                start = jdoc_start
        elif pre.endswith("}"):
            # Previous block — don't back up
            start = match.start()
        # Find matching closing brace
        depth = 0
        i = content.index("{", match.start())
        while i < len(content):
            if content[i] == "{":
                depth += 1
            elif content[i] == "}":
                depth -= 1
                if depth == 0:
                    return content[start : i + 1].strip()
            i += 1
        return None
    except Exception:
        return None


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
    meta_hint = (
        " Use test_type='metamorphic' and provide a non-null falsifiable metamorphic_relation."
        if spec.complexity >= 2
        else " Use test_type='happy_path' or 'edge_case'. metamorphic_relation must be null."
    )
    source = _read_func_source(spec)
    if source:
        # Strip 'export' keyword so the inline copy works without module system
        bare_source = source.replace("export function ", "function ").replace("export async function ", "async function ")
        source_section = (
            f"PART 1 — copy this implementation VERBATIM into test_code:\n"
            f"```typescript\n{bare_source}\n```\n"
            f"PART 2 — add this exact line after the implementation:\n"
            f'import {{ describe, it, expect }} from "vitest";\n'
            f"PART 3 — add a describe(\"{spec.func_name}\", ...) block with 2-3 it() assertions.\n"
            f"Trace through the implementation to determine correct expected values."
        )
    else:
        jsdoc = spec.jsdoc or ""
        source_section = (
            f"JSDoc: {jsdoc}\n"
            f"Write the implementation from the function signature, then the vitest tests."
        )
    return (
        f"Function: {spec.func_name}({params_desc}) -> {ret} | Async: {spec.is_async}\n"
        f"test_id='test_{spec.func_name}'.{meta_hint}\n\n"
        f"{source_section}"
    )


def _sanitize_test_code(code: str) -> str:
    """Fix common LLM quoting errors in generated TypeScript/Vitest test code.

    Scans it('...') and describe('...') first-argument strings and fixes:
    - Embedded unescaped single quotes in description strings (e.g. it('should return 'X'...')
    - Converts such strings to double-quoted form to prevent esbuild syntax errors.
    """
    lines = code.split("\n")
    out: list[str] = []
    # Pattern: it( or describe( followed by a single-quoted first argument
    # We capture the keyword and everything up to the closing quote+comma.
    # We do a line-level scan and look for malformed single-quoted descriptions.
    _CALL_START = re.compile(r'^(\s*)(it|describe)\s*\(\'(.*)')
    for line in lines:
        m = _CALL_START.match(line)
        if m:
            indent = m.group(1)
            keyword = m.group(2)
            rest = m.group(3)  # everything after the opening single quote
            # Find the closing "', " or "'," pattern (description end + comma)
            # If it finds a well-formed close, leave alone
            close_idx = rest.find("',")
            if close_idx == -1:
                # Malformed: no proper closing ', — convert whole description to double quote
                # Take everything up to the next ', or end of line
                # Heuristic: the description ends at the last ', on the line
                # Replace the it(' or describe(' with it(" or describe("
                rest_fixed = rest.replace("'", "\\'")
                out.append(line)
                continue
            description = rest[:close_idx]
            after_close = rest[close_idx + 2:]  # after the closing ',
            if "'" in description:
                # Escape any double quotes in the description, switch to double-quoting
                description_fixed = description.replace('"', '\\"')
                out.append(f'{indent}{keyword}("{description_fixed}", {after_close}')
            else:
                out.append(line)
        else:
            out.append(line)
    return "\n".join(out)


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
                sanitized_code = _sanitize_test_code(test.test_code)
                return test.model_copy(update={"func_id": spec.func_id, "test_code": sanitized_code})
            except StructuredGenerationError as exc:
                logger.warning(f"JSTestGenerator: failed for {spec.func_name!r}: {exc!r}")
                return None


__all__ = ["GeneratedJSTest", "JSTestGenerator"]
