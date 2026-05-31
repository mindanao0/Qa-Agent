"""
PII Masker — PDPA-compliant scrubber applied before any content enters an LLM call.

Masked categories:
  EMAIL          standard RFC 5322 email addresses
  PHONE          Thai mobile/landline (local 0x and +66 international formats)
  NATIONAL_ID    Thai national ID (13-digit with and without dashes)
  PASSPORT       Thai and international passport numbers
  CREDIT_CARD    16-digit card numbers with optional separators
  BANK_ACCOUNT   10–12-digit account numbers after keyword context
  CREDENTIALS    user:pass in URLs
  IP_ADDRESS     non-RFC1918 IPv4 addresses
  THAI_NAME      Thai honorific + Thai-script name
"""

from __future__ import annotations

import re
from typing import Any

from loguru import logger

# ──────────────────────────────────────────────────────────────────────────────
# Compiled rules — ordered most-specific first to avoid partial overlaps
# ──────────────────────────────────────────────────────────────────────────────

_RULES: list[tuple[re.Pattern[str], str]] = [
    # Thai national ID with dashes: 1-2345-67890-12-3
    (
        re.compile(r"\b\d{1}-\d{4}-\d{5}-\d{2}-\d{1}\b"),
        "[NATIONAL_ID]",
    ),
    # Email addresses
    (
        re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
        "[EMAIL]",
    ),
    # URL embedded credentials: proto://user:pass@host  (replace just user:pass)
    (
        re.compile(r"(?<=://)[^:/@\s]+:[^@\s]+(?=@)"),
        "[CREDENTIALS]",
    ),
    # Thai phone — international +66
    (
        re.compile(r"(?<!\d)\+66[2-9]\d{7,8}(?!\d)"),
        "[PHONE]",
    ),
    # Thai mobile 08x / 09x (10 digits starting 08 or 09)
    (
        re.compile(r"(?<!\d)0[89]\d{8}(?!\d)"),
        "[PHONE]",
    ),
    # Thai landline 0x-xxxx-xxxx (9-10 digits starting 02–07)
    (
        re.compile(r"(?<!\d)0[2-7]\d{7,8}(?!\d)"),
        "[PHONE]",
    ),
    # Credit card: 16 digits with optional spaces/dashes
    (
        re.compile(r"(?<!\d)\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}(?!\d)"),
        "[CREDIT_CARD]",
    ),
    # Standalone 13-digit Thai national ID (no dashes)
    (
        re.compile(r"(?<!\d)\d{13}(?!\d)"),
        "[NATIONAL_ID]",
    ),
    # Bank account after keyword: "account no: 1234567890"
    (
        re.compile(
            r"(?:account|บัญชี|เลขที่บัญชี|acc\.?)\s*(?:no\.?|number|#|ที่)?"
            r"\s*[:：]?\s*(\d{10,12})",
            re.IGNORECASE,
        ),
        r"account [BANK_ACCOUNT]",
    ),
    # Passport numbers: 1–2 uppercase letters + 6–9 digits (standalone)
    (
        re.compile(r"\b[A-Z]{1,2}\d{6,9}\b"),
        "[PASSPORT]",
    ),
    # Non-RFC1918 IPv4 addresses
    # Negative lookaheads exclude private/reserved ranges; octet pattern
    # allows 0-255 per octet (including 0 in non-first octets like 203.0.113.x)
    (
        re.compile(
            r"\b"
            r"(?!127\.)"
            r"(?!10\.)"
            r"(?!192\.168\.)"
            r"(?!172\.(?:1[6-9]|2[0-9]|3[01])\.)"
            r"(?!0\.)"
            r"(?!255\.)"
            r"\d{1,3}(?:\.\d{1,3}){3}"
            r"\b"
        ),
        "[IP_ADDRESS]",
    ),
    # Thai honorific + Thai-script name (greedy word match)
    (
        re.compile(
            r"(?:นาย|นาง(?:สาว)?|เด็กชาย|เด็กหญิง|ดร\.|นพ\.|ทพ\.|พ\.ต\.ท\.?)"
            r"\s*[฀-๿]{2,}(?:\s+[฀-๿]{2,})*"
        ),
        "[THAI_NAME]",
    ),
]


# ──────────────────────────────────────────────────────────────────────────────
# PIIMasker class
# ──────────────────────────────────────────────────────────────────────────────


class PIIMasker:
    """
    Applies all PII replacement rules to text or arbitrary JSON-serialisable
    structures.

    Thread-safe (stateless per-call). A single module-level instance is
    exported for convenience; create a custom instance only when you need
    custom rules or to disable logging.
    """

    def __init__(
        self,
        rules: list[tuple[re.Pattern[str], str]] | None = None,
        log_masks: bool = True,
    ) -> None:
        self._rules = rules if rules is not None else _RULES
        self._log = log_masks

    # ── Core ─────────────────────────────────────────────────────────────────

    def mask(self, text: str) -> str:
        """Return *text* with all PII patterns replaced by placeholder tokens."""
        if not text:
            return text
        result = text
        total_replacements = 0
        for pattern, replacement in self._rules:
            result, n = pattern.subn(replacement, result)
            total_replacements += n
        if total_replacements > 0 and self._log:
            logger.debug(
                f"PIIMasker.mask: replaced {total_replacements} PII instance(s)"
            )
        return result

    def mask_json(self, data: Any) -> Any:
        """
        Recursively traverse *data* (dict / list / str) and mask PII in every
        string value.  Non-string leaves (int, float, bool, None) are returned
        unchanged.
        """
        if isinstance(data, str):
            return self.mask(data)
        if isinstance(data, dict):
            return {k: self.mask_json(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self.mask_json(item) for item in data]
        return data

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def audit(self, text: str) -> dict[str, list[str]]:
        """
        Return a mapping of {placeholder_token: [matched_original_strings]}
        for *text* without modifying it.  Useful for audits and tests.
        """
        findings: dict[str, list[str]] = {}
        for pattern, replacement in self._rules:
            # strip back-reference syntax from replacement to get the token
            token = re.sub(r"\\[0-9]+", "", replacement).strip()
            matches = pattern.findall(text)
            if matches:
                findings.setdefault(token, []).extend(
                    m if isinstance(m, str) else m[0] for m in matches
                )
        return findings


# ──────────────────────────────────────────────────────────────────────────────
# Module-level convenience API
# ──────────────────────────────────────────────────────────────────────────────

_default = PIIMasker()


def mask_text(text: str) -> str:
    """Mask PII in a single string using the default rule set."""
    return _default.mask(text)


def mask_json(data: Any) -> Any:
    """Recursively mask PII in a JSON-serialisable structure."""
    return _default.mask_json(data)
