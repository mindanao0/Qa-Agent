from __future__ import annotations

from typing import Literal

from loguru import logger
from playwright.async_api import Page
from pydantic import BaseModel, ConfigDict

from src.llm.instructor_client import InstructorClient, StructuredGenerationError

_DUMMY_BY_TYPE = {
    "email": "qa_test@mailinator.com",
    "password": "QaTest123!",
    "text": "Test Input",
    "number": "12345",
    "tel": "0812345678",
    "date": "2026-01-01",
    "search": "test",
    "url": "https://example.com",
}


def _dummy_for_type(input_type: str) -> str:
    return _DUMMY_BY_TYPE.get((input_type or "text").lower(), "Test Input")


class _FieldValue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str
    value: str


class _FieldValues(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fields: list[_FieldValue]


class FormFiller:
    """Fills form inputs with dummy data; falls back to LLM-suggested values."""

    def __init__(self, client: InstructorClient | None = None) -> None:
        self._client = client

    async def fill_with_dummy(self, page: Page) -> int:
        """Fill every text-like input with dummy data. Returns count filled."""
        inputs = await page.query_selector_all(
            "input:not([type=hidden]):not([type=submit]):not([type=button]), textarea"
        )
        filled = 0
        for el in inputs:
            try:
                itype = (await el.get_attribute("type")) or "text"
                await el.fill(_dummy_for_type(itype), timeout=3_000)
                filled += 1
            except Exception:
                continue
        return filled

    async def fill_with_llm(self, page: Page, field_labels: list[str]) -> int:
        """Ask the LLM for plausible values when dummy data fails validation."""
        if self._client is None or not field_labels:
            return 0
        prompt = (
            "Provide realistic test values (JSON) for these form fields. "
            "Return field 'fields': list of {label, value}.\n"
            + "\n".join(f"- {lbl}" for lbl in field_labels)
        )
        try:
            resp: _FieldValues = await self._client.create_structured(
                prompt, _FieldValues, temperature=0.1
            )
        except (StructuredGenerationError, Exception) as exc:
            logger.warning(f"FormFiller: LLM fill failed — {exc!r}")
            return 0

        filled = 0
        for fv in resp.fields:
            try:
                loc = page.get_by_label(fv.label)
                if await loc.count():
                    await loc.first.fill(fv.value, timeout=3_000)
                    filled += 1
            except Exception:
                continue
        return filled


__all__ = ["FormFiller", "_dummy_for_type"]
