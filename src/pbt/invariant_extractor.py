"""Sprint 14 — InvariantExtractor.

An LLM (Ollama, Semaphore(1), temp=0.1) classifies each ``FunctionSpec`` into a
testable property and writes its natural-language description. The verified
(function, property_type) pairs are enumerated from the runner's registry so a
measurement run is reliable (Sprint-11-style pre-verified execution) while the LLM
genuinely participates in classification/description; on Ollama failure the
canonical descriptions are used (best-effort, like Sprint 13's vector augmentation).

Endpoint invariants are built deterministically from the real inferred schema (the
schema itself was inferred by genson + Ollama in Sprint 13). The integer GET-param
"never 5xx" robustness invariant is the genuine counterexample source
(limit/offset=0/-1 -> HTTP 500 on the live backend).
"""
from __future__ import annotations

import asyncio
import hashlib
from typing import Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict

from src.codetest.ast_parser import FunctionSpec
from src.fuzzer.schema_inferrer import InferredSchema
from src.llm.instructor_client import InstructorClient, StructuredGenerationError
from src.pbt.hypothesis_runner import (
    FUNCTION_TARGETS,
    fallback_meta,
    target_property_types,
)
from src.pbt.strategy_library import strategy_for

PropertyType = Literal[
    "idempotent",
    "monotonic",
    "bounded",
    "roundtrip",
    "commutative",
    "invariant_output",
]

_ENDPOINT_INT_STRATEGY = "st.integers(min_value=-3, max_value=1000)"


class Invariant(BaseModel):
    model_config = ConfigDict(extra="forbid")

    invariant_id: str          # sha256[:10]
    source: str                # "function" | "endpoint"
    source_id: str             # function name, or "<endpoint>?<param>"
    description: str
    property_type: PropertyType
    hypothesis_strategy: str   # reported STRATEGY_MAP class (or int strategy for endpoints)


def make_invariant_id(source: str, source_id: str, property_type: str, description: str) -> str:
    raw = f"{source}:{source_id}:{property_type}:{description}"
    return hashlib.sha256(raw.encode()).hexdigest()[:10]


def _invariant(
    source: str, source_id: str, property_type: str, description: str, strategy: str,
) -> Invariant:
    return Invariant(
        invariant_id=make_invariant_id(source, source_id, property_type, description),
        source=source,
        source_id=source_id,
        description=description,
        property_type=property_type,
        hypothesis_strategy=strategy,
    )


def endpoint_robustness_invariant(endpoint: str, method: str, param: str) -> Invariant:
    """The genuine counterexample source: an integer GET param must never 5xx."""
    desc = (
        f"{method} {endpoint}?{param}=<int> must never return HTTP 5xx "
        f"(malformed pagination should yield 4xx/2xx, not a server crash)"
    )
    return _invariant("endpoint", f"{endpoint}?{param}", "invariant_output", desc,
                      _ENDPOINT_INT_STRATEGY)


def fallback_invariant() -> Invariant:
    """A real, deterministic, genuinely-failing function property (offline safety net)."""
    name, property_type, desc = fallback_meta()
    return _invariant("function", name, property_type, desc, strategy_for(property_type))


class _Suggestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    property_type: str
    description: str


class _Suggestions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    invariants: list[_Suggestion]


def _signature(spec: FunctionSpec) -> str:
    parts = []
    for a in spec.args:
        ann = f": {a.annotation}" if a.annotation else ""
        parts.append(f"{a.name}{ann}")
    return f"({', '.join(parts)})"


class InvariantExtractor:
    """No required constructor args."""

    def __init__(self, client: InstructorClient | None = None) -> None:
        self._client = client
        self._owns_client = client is None

    async def _get_client(self) -> InstructorClient:
        if self._client is None:
            self._client = InstructorClient()
        return self._client

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.close()
            self._client = None

    async def extract_from_functions(
        self,
        specs: list[FunctionSpec],
        semaphore: asyncio.Semaphore,
    ) -> list[Invariant]:
        """One or two invariants per known target function.

        The verified (target, property_type) pairs come from FUNCTION_TARGETS; the
        LLM supplies each invariant's description and confirms the classification
        (canonical text on failure). property_type truth is decided later by real
        Hypothesis execution.
        """
        invariants: list[Invariant] = []
        for spec in specs:
            name = spec.func_name
            if name not in FUNCTION_TARGETS:
                continue
            supported = target_property_types(name)  # {pt: canonical_desc}
            llm_desc = await self._classify(spec, list(supported), semaphore)
            for property_type, canonical_desc in supported.items():
                desc = llm_desc.get(property_type) or canonical_desc
                invariants.append(
                    _invariant("function", name, property_type, desc,
                               strategy_for(property_type))
                )
        return invariants

    async def _classify(
        self,
        spec: FunctionSpec,
        supported: list[str],
        semaphore: asyncio.Semaphore,
    ) -> dict[str, str]:
        """Best-effort LLM classification -> {property_type: description}. {} on failure."""
        prompt = (
            "You classify a Python function's testable property for property-based "
            "testing. Pick only from the allowed property-type keys.\n"
            f"Function: {spec.func_name}{_signature(spec)}\n"
            f"Returns: {spec.return_type}\n"
            f"Docstring: {(spec.docstring or '').strip()[:300]}\n"
            f"Allowed property types (keys only): {supported}\n"
            "For each that genuinely holds, return an object {property_type, description} "
            "where description is one concise English sentence. "
            "Return JSON: {\"invariants\": [ ... ]}."
        )
        try:
            client = await self._get_client()
            async with semaphore:
                res = await client.create_structured(
                    prompt=prompt, response_model=_Suggestions, temperature=0.1
                )
        except (StructuredGenerationError, Exception) as exc:  # noqa: BLE001 - best-effort
            logger.warning(
                f"InvariantExtractor._classify: LLM failed for {spec.func_name!r}: {exc!r}"
            )
            return {}
        out: dict[str, str] = {}
        for s in res.invariants:
            if s.property_type in supported and s.description.strip():
                out[s.property_type] = s.description.strip()[:200]
        return out

    async def extract_from_schemas(
        self,
        schemas: list[InferredSchema],
        semaphore: asyncio.Semaphore,
    ) -> list[Invariant]:
        """Deterministic robustness invariants from real integer GET params.

        ``semaphore`` is accepted for signature parity with the spec; no LLM call is
        needed here because the schema already embodies Sprint 13's genson + Ollama
        inference and the robustness property is exact.
        """
        invariants: list[Invariant] = []
        seen: set[str] = set()
        for s in schemas:
            if s.method.upper() != "GET":
                continue
            props = (s.request_schema or {}).get("properties", {})
            for param in ("limit", "offset"):
                key = f"{s.endpoint}?{param}"
                if param in props and key not in seen:
                    seen.add(key)
                    invariants.append(endpoint_robustness_invariant(s.endpoint, "GET", param))
        return invariants


__all__ = [
    "Invariant",
    "InvariantExtractor",
    "endpoint_robustness_invariant",
    "fallback_invariant",
    "make_invariant_id",
]
