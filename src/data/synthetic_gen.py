"""
Synthetic QA dataset generator — UPDATE 3 three-phase pipeline.

Produces ChatML JSONL training examples across 13 application domains using
three sequential phases:

  PHASE 1 — Per-domain examples (happy paths + edge cases).
            For each domain, for each scenario / edge case, ask the LLM for a
            realistic pytest-playwright (sync) test using semantic locators
            only.  Each example is stored with rich metadata so downstream
            consumers can filter / balance.

  PHASE 2 — Healing pair examples (template-driven).
            50 short examples that train the model to convert a broken
            CSS / XPath locator into a semantic Playwright locator.  Variations
            are produced by substituting domain-specific element names into a
            small set of seed templates so coverage is wide across domains.

  PHASE 3 — Validation & quality filter.
            Every assistant payload is parsed with `ast.parse`; rejected on
            SyntaxError, missing `test_` function (for full-test examples),
            or any CSS / XPath / hardcoded-wait pattern.  Statistics are
            logged at the end.  Only valid examples reach the output file.

Output file (default): ~/.qa-agent/datasets/synthetic_universal.jsonl

Each JSONL line:

    {
      "messages": [
        {"role": "system",    "content": "<phase-specific system prompt>"},
        {"role": "user",      "content": "<scenario prompt>"},
        {"role": "assistant", "content": "<generated code>"}
      ],
      "metadata": {
        "domain":   "<domain key>",
        "scenario": "<seed scenario / template>",
        "type":     "happy_path" | "edge_case" | "healing"
      }
    }

PII masking is applied to every record before writing.

Also exports `detect_domain(requirement, url)` — used by PlannerAgent for
automatic domain detection before planning.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Any

from loguru import logger
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from src.llm.adapter import OllamaAdapter
from .pii_masker import PIIMasker

# ──────────────────────────────────────────────────────────────────────────────
# Universal Domain Registry — 13 web application domains
# ──────────────────────────────────────────────────────────────────────────────

DOMAIN_REGISTRY: dict[str, dict[str, Any]] = {
    "authentication": {
        "description": "Login, logout, registration, password reset, 2FA, SSO, OAuth",
        "scenarios": [
            "login with valid credentials",
            "login with invalid password",
            "login with empty fields",
            "logout and session clear",
            "password reset via email",
            "register new account with validation",
            "2FA code verification",
            "remember me functionality",
            "account lockout after failed attempts",
            "SSO redirect flow",
        ],
        "edge_cases": [
            "SQL injection in login fields",
            "XSS in username field",
            "concurrent sessions",
            "session timeout after inactivity",
            "login with special characters in password",
        ],
    },
    "crud_operations": {
        "description": "Create Read Update Delete for any entity",
        "scenarios": [
            "create new record with all required fields",
            "create record with optional fields empty",
            "read/view record details",
            "search and filter records",
            "sort records by column",
            "paginate through records",
            "update existing record",
            "delete record with confirmation",
            "bulk delete multiple records",
            "export records to CSV/Excel",
        ],
        "edge_cases": [
            "create duplicate record",
            "update record with invalid data",
            "delete record that has dependencies",
            "search with special characters",
            "pagination boundary (first/last page)",
        ],
    },
    "forms_validation": {
        "description": "Form input validation, required fields, format checks",
        "scenarios": [
            "submit form with all valid data",
            "submit form with empty required fields",
            "submit form with invalid email format",
            "submit form with invalid phone number",
            "submit form with text in number field",
            "submit form with date in wrong format",
            "submit form with value exceeding max length",
            "submit form with value below minimum",
            "form auto-save draft",
            "multi-step form navigation",
        ],
        "edge_cases": [
            "paste very long text into field",
            "submit form twice rapidly (double submit)",
            "form with file upload validation",
            "dropdown with no options selected",
            "date picker past/future restriction",
        ],
    },
    "rbac_permissions": {
        "description": "Role-based access control, permission matrix, feature gating",
        "scenarios": [
            "admin can access all features",
            "regular user cannot access admin panel",
            "read-only user cannot create records",
            "manager can approve subordinate requests",
            "guest user sees only public content",
            "user cannot access other user data",
            "permission denied shows correct message",
            "navigation menu filtered by role",
            "API returns 403 for unauthorized role",
            "elevated privilege actions require confirmation",
        ],
        "edge_cases": [
            "user with multiple roles",
            "role changed mid-session",
            "access URL directly without permission",
            "expired session accessing protected route",
        ],
    },
    "ecommerce": {
        "description": "Product catalog, cart, checkout, orders, payments, reviews",
        "scenarios": [
            "browse product catalog",
            "search product by name",
            "filter products by category and price",
            "view product detail page",
            "add product to cart",
            "update quantity in cart",
            "remove item from cart",
            "apply discount coupon code",
            "checkout with shipping address",
            "complete payment with credit card",
            "view order confirmation",
            "track order status",
            "submit product review",
            "add product to wishlist",
            "compare products",
        ],
        "edge_cases": [
            "add out-of-stock product",
            "apply expired coupon",
            "checkout with empty cart",
            "payment declined scenario",
            "product price change during checkout",
        ],
    },
    "finance_banking": {
        "description": "Account management, transfers, payments, statements, loans",
        "scenarios": [
            "view account balance",
            "view transaction history",
            "filter transactions by date range",
            "transfer money between own accounts",
            "transfer money to other account",
            "pay bill via account number",
            "schedule recurring payment",
            "download account statement PDF",
            "view loan details and EMI schedule",
            "apply for new credit card",
        ],
        "edge_cases": [
            "transfer amount exceeds balance",
            "transfer to invalid account number",
            "duplicate transaction within 1 minute",
            "transfer during system maintenance window",
            "statement with zero transactions",
        ],
    },
    "healthcare": {
        "description": "Patient records, appointments, prescriptions, lab results, billing",
        "scenarios": [
            "register new patient",
            "search patient by ID or name",
            "view patient medical history",
            "book appointment with doctor",
            "reschedule appointment",
            "cancel appointment",
            "view doctor availability calendar",
            "create prescription",
            "view lab test results",
            "generate medical invoice",
        ],
        "edge_cases": [
            "book appointment in the past",
            "duplicate patient registration",
            "prescription with controlled substance",
            "patient with allergy conflict in prescription",
            "appointment overlap for same doctor",
        ],
    },
    "hrm_payroll": {
        "description": "Employee management, attendance, leave, payroll, Thai labor law",
        "scenarios": [
            "create new employee profile",
            "update employee department and position",
            "record daily attendance",
            "request annual leave",
            "approve leave request as manager",
            "calculate monthly payroll",
            "verify social security deduction",
            "verify income tax calculation",
            "generate payslip",
            "run overtime calculation",
        ],
        "edge_cases": [
            "payroll with prorated salary",
            "overtime on public holiday (3x rate)",
            "leave request exceeding balance",
            "employee with multiple positions",
            "payroll with mid-month joining date",
        ],
    },
    "erp_crm": {
        "description": "Sales pipeline, inventory, purchase orders, customers, suppliers",
        "scenarios": [
            "create sales lead",
            "convert lead to opportunity",
            "create quotation for customer",
            "convert quotation to sales order",
            "create purchase order to supplier",
            "receive inventory from purchase order",
            "view stock level by warehouse",
            "create customer invoice",
            "record customer payment",
            "view sales report by period",
        ],
        "edge_cases": [
            "order with item out of stock",
            "invoice already paid",
            "negative inventory adjustment",
            "purchase order with price variance",
            "customer credit limit exceeded",
        ],
    },
    "cms_content": {
        "description": "Blog, articles, media library, categories, comments, SEO",
        "scenarios": [
            "create new article with rich text",
            "publish article",
            "schedule article for future publish",
            "unpublish/archive article",
            "upload image to media library",
            "add category and tags to article",
            "moderate user comments",
            "search articles by keyword",
            "preview article before publish",
            "set SEO metadata for article",
        ],
        "edge_cases": [
            "publish article with no title",
            "upload file exceeding size limit",
            "duplicate article slug",
            "article with broken image link",
            "comment with spam content",
        ],
    },
    "dashboard_analytics": {
        "description": "Charts, KPI cards, filters, date ranges, data export",
        "scenarios": [
            "view main dashboard on load",
            "filter dashboard by date range",
            "filter dashboard by department or category",
            "view KPI card details on click",
            "switch chart between bar and line",
            "export dashboard data to Excel",
            "drill down into chart segment",
            "compare current vs previous period",
            "view real-time updating metrics",
            "save custom dashboard layout",
        ],
        "edge_cases": [
            "dashboard with no data in selected range",
            "filter combination with zero results",
            "export very large dataset",
            "chart with single data point",
            "date range start after end date",
        ],
    },
    "notifications_messaging": {
        "description": "Email notifications, in-app alerts, chat, announcements",
        "scenarios": [
            "receive in-app notification",
            "mark notification as read",
            "mark all notifications as read",
            "view notification history",
            "configure notification preferences",
            "send direct message to user",
            "view message thread",
            "attach file in message",
            "view system announcements",
            "subscribe to notification topic",
        ],
        "edge_cases": [
            "notification with very long text",
            "message to blocked user",
            "notification preferences conflict",
            "message with unsupported file type",
            "send message to offline user",
        ],
    },
    "settings_configuration": {
        "description": "Profile settings, system config, integrations, API keys",
        "scenarios": [
            "update user profile information",
            "change account password",
            "upload profile picture",
            "configure system general settings",
            "manage API key generation and revocation",
            "configure email SMTP settings",
            "set timezone and locale",
            "configure webhook endpoints",
            "manage connected integrations",
            "view system audit log",
        ],
        "edge_cases": [
            "change password to same as current",
            "upload non-image as profile picture",
            "revoke API key currently in use",
            "invalid SMTP configuration",
            "webhook endpoint returning error",
        ],
    },
}


DOMAIN_KEYS: list[str] = list(DOMAIN_REGISTRY.keys())
_FALLBACK_DOMAIN = "crud_operations"
_DEFAULT_OUTPUT = "~/.qa-agent/datasets/synthetic_universal.jsonl"


# ──────────────────────────────────────────────────────────────────────────────
# Phase-specific prompts (per UPDATE 3 spec)
# ──────────────────────────────────────────────────────────────────────────────

_PHASE1_SYSTEM_PROMPT = (
    "You are an expert QA engineer. Generate a realistic Playwright Python test "
    "for the given scenario. Output ONLY raw Python code using pytest-playwright format. "
    "Use only: get_by_role(), get_by_label(), get_by_text(), get_by_test_id(), expect(). "
    "Function name must start with test_. No markdown fences. No imports except pytest "
    "and playwright.sync_api."
)

_PHASE1_EDGE_SUFFIX = (
    "\nThis is an edge case test. Include appropriate error handling and verify "
    "the correct error message or behavior."
)

_PHASE2_HEALING_SYSTEM_PROMPT = (
    "You are a Playwright self-healing expert. "
    "Given a broken locator and current AxTree, return the correct locator. "
    "Output ONLY the corrected single line of Python code."
)

# Healing seed templates: (broken_pattern, fixed_pattern, role, accessible_name)
# The {name}/{label} tokens are filled in with domain-appropriate values to
# create 50 distinct healing examples across the 13 domains.
_HEALING_TEMPLATES: list[dict[str, str]] = [
    {
        "broken": "page.click('#{button_id}')",
        "fixed":  "page.get_by_role('button', name='{button_name}').click()",
    },
    {
        "broken": "page.fill('#{input_id}', value)",
        "fixed":  "page.get_by_label('{input_label}').fill(value)",
    },
    {
        "broken": "page.locator('.{alert_class}').text",
        "fixed":  "expect(page.get_by_role('alert')).to_be_visible()",
    },
    {
        "broken": "page.find_element('xpath=//{xpath_tag}')",
        "fixed":  "page.get_by_role('{xpath_role}', name='{xpath_name}').click()",
    },
]

# Domain-specific element vocab used when filling healing templates so the
# 50 generated examples actually look like they came from 13 different apps.
_HEALING_VOCAB: dict[str, dict[str, list[str]]] = {
    "authentication": {
        "button_id": ["submit-btn", "login-btn", "signin-btn"],
        "button_name": ["Sign in", "Log in", "Submit"],
        "input_id": ["email", "password", "username"],
        "input_label": ["Email", "Password", "Username"],
        "alert_class": ["error-msg", "auth-error", "login-error"],
        "xpath_tag": ["button[@id='login']", "button[contains(.,'Sign')]"],
        "xpath_role": ["button"],
        "xpath_name": ["Login", "Sign in"],
    },
    "crud_operations": {
        "button_id": ["save-btn", "create-btn", "delete-btn"],
        "button_name": ["Save", "Create", "Delete"],
        "input_id": ["name", "description", "search"],
        "input_label": ["Name", "Description", "Search"],
        "alert_class": ["error-msg", "validation-error"],
        "xpath_tag": ["button[@type='submit']"],
        "xpath_role": ["button"],
        "xpath_name": ["Submit", "Save record"],
    },
    "forms_validation": {
        "button_id": ["submit-form", "next-step"],
        "button_name": ["Submit", "Next"],
        "input_id": ["email", "phone", "zip"],
        "input_label": ["Email", "Phone", "Zip code"],
        "alert_class": ["field-error", "validation-msg"],
        "xpath_tag": ["span[@class='error']"],
        "xpath_role": ["button"],
        "xpath_name": ["Submit form"],
    },
    "rbac_permissions": {
        "button_id": ["admin-panel", "manage-users"],
        "button_name": ["Admin Panel", "Manage Users"],
        "input_id": ["role-name", "permission"],
        "input_label": ["Role name", "Permission"],
        "alert_class": ["access-denied", "forbidden-msg"],
        "xpath_tag": ["div[@id='denied']"],
        "xpath_role": ["button"],
        "xpath_name": ["Request access"],
    },
    "ecommerce": {
        "button_id": ["add-to-cart", "checkout-btn", "buy-now"],
        "button_name": ["Add to cart", "Checkout", "Buy now"],
        "input_id": ["coupon-code", "quantity", "shipping-address"],
        "input_label": ["Coupon code", "Quantity", "Shipping address"],
        "alert_class": ["cart-error", "out-of-stock"],
        "xpath_tag": ["button[contains(.,'Add')]"],
        "xpath_role": ["button"],
        "xpath_name": ["Add to cart"],
    },
    "finance_banking": {
        "button_id": ["transfer-btn", "pay-bill", "submit-transfer"],
        "button_name": ["Transfer", "Pay bill", "Confirm transfer"],
        "input_id": ["amount", "account-number", "ifsc"],
        "input_label": ["Amount", "Account number", "IFSC code"],
        "alert_class": ["transfer-error", "insufficient-funds"],
        "xpath_tag": ["button[@id='confirm']"],
        "xpath_role": ["button"],
        "xpath_name": ["Confirm transfer"],
    },
    "healthcare": {
        "button_id": ["book-appointment", "save-patient", "prescribe"],
        "button_name": ["Book appointment", "Save patient", "Prescribe"],
        "input_id": ["patient-id", "doctor-name", "diagnosis"],
        "input_label": ["Patient ID", "Doctor name", "Diagnosis"],
        "alert_class": ["appointment-error", "conflict-msg"],
        "xpath_tag": ["button[@class='book']"],
        "xpath_role": ["button"],
        "xpath_name": ["Book appointment"],
    },
    "hrm_payroll": {
        "button_id": ["run-payroll", "approve-leave", "save-employee"],
        "button_name": ["Run payroll", "Approve leave", "Save employee"],
        "input_id": ["employee-id", "leave-days", "base-salary"],
        "input_label": ["Employee ID", "Leave days", "Base salary"],
        "alert_class": ["payroll-error", "leave-error"],
        "xpath_tag": ["button[@type='submit']"],
        "xpath_role": ["button"],
        "xpath_name": ["Submit payroll"],
    },
    "erp_crm": {
        "button_id": ["create-lead", "convert-to-order", "send-quote"],
        "button_name": ["Create lead", "Convert to order", "Send quotation"],
        "input_id": ["company-name", "contact-email", "product-code"],
        "input_label": ["Company name", "Contact email", "Product code"],
        "alert_class": ["pipeline-error", "stock-warning"],
        "xpath_tag": ["button[@id='convert']"],
        "xpath_role": ["button"],
        "xpath_name": ["Convert"],
    },
    "cms_content": {
        "button_id": ["publish-btn", "save-draft", "upload-image"],
        "button_name": ["Publish", "Save draft", "Upload image"],
        "input_id": ["article-title", "slug", "tags"],
        "input_label": ["Title", "Slug", "Tags"],
        "alert_class": ["editor-error", "upload-failed"],
        "xpath_tag": ["button[contains(.,'Publish')]"],
        "xpath_role": ["button"],
        "xpath_name": ["Publish"],
    },
    "dashboard_analytics": {
        "button_id": ["apply-filter", "export-excel", "drill-down"],
        "button_name": ["Apply filter", "Export to Excel", "Drill down"],
        "input_id": ["date-from", "date-to", "department"],
        "input_label": ["Date from", "Date to", "Department"],
        "alert_class": ["chart-error", "no-data-msg"],
        "xpath_tag": ["button[@aria-label='Export']"],
        "xpath_role": ["button"],
        "xpath_name": ["Export"],
    },
    "notifications_messaging": {
        "button_id": ["send-message", "mark-read", "attach-file"],
        "button_name": ["Send", "Mark as read", "Attach"],
        "input_id": ["message-text", "recipient", "subject"],
        "input_label": ["Message", "Recipient", "Subject"],
        "alert_class": ["msg-error", "delivery-failed"],
        "xpath_tag": ["button[@class='send']"],
        "xpath_role": ["button"],
        "xpath_name": ["Send message"],
    },
    "settings_configuration": {
        "button_id": ["save-settings", "regenerate-key", "test-webhook"],
        "button_name": ["Save settings", "Regenerate API key", "Test webhook"],
        "input_id": ["api-key", "webhook-url", "smtp-host"],
        "input_label": ["API key", "Webhook URL", "SMTP host"],
        "alert_class": ["settings-error", "invalid-config"],
        "xpath_tag": ["button[@id='save']"],
        "xpath_role": ["button"],
        "xpath_name": ["Save"],
    },
}


# Sample AxTree fragments parameterised by role/name — kept tiny so they fit
# the training context easily.
_AXTREE_FRAGMENTS = [
    """- main:
  - heading "{heading}" [level=1]
  - form:
    - textbox "{input_label}"
    - button "{button_name}"
    - alert """,
    """- region "{heading}":
  - link "Home" [url=/]
  - textbox "{input_label}" [required]
  - button "{button_name}" [type=submit]""",
    """- dialog "{heading}":
  - textbox "{input_label}"
  - button "{button_name}"
  - button "Cancel" """,
]


# Validation regex per spec (intent-preserving): catches CSS id / class
# selectors inside string literals, raw page.locator('#…') / page.locator('.…')
# calls, and any `xpath=` prefix.  The spec's literal regex `[#.]\w+` would
# also match every `.method` call so we use the intent-preserving form.
_FORBIDDEN_RE = re.compile(
    r"""['"]\s*[#.][\w\-]+|"""    # '#id' or '.class' inside strings
    r"""xpath\s*=|"""               # xpath= prefix
    r"""find_element\s*\(|"""       # selenium-style API
    r"""wait_for_timeout|"""        # hardcoded waits
    r"""asyncio\.sleep"""
)


# ──────────────────────────────────────────────────────────────────────────────
# Domain detection (used by PlannerAgent — unchanged interface)
# ──────────────────────────────────────────────────────────────────────────────


def _build_domain_detect_messages(
    requirement: str, url: str
) -> list[dict[str, str]]:
    domain_list = "\n".join(
        f"- {name}: {info['description']}"
        for name, info in DOMAIN_REGISTRY.items()
    )
    system = (
        "You are a domain classifier for web application QA tests. "
        "Given a test requirement and target URL, you pick the SINGLE closest "
        "matching domain from a fixed list. "
        "Output ONLY the domain name (one of the listed keys, lowercase, no quotes, "
        "no explanation). If none match clearly, output: crud_operations."
    )
    user = (
        f"Available domains:\n{domain_list}\n\n"
        f"Test requirement: {requirement}\n"
        f"Target URL: {url}\n\n"
        "Domain name (one of the listed keys only):"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _parse_domain_response(raw: str) -> str:
    if not raw:
        return _FALLBACK_DOMAIN
    cleaned = raw.strip().lower()
    cleaned = cleaned.strip("`").strip('"').strip("'").strip()
    if cleaned in DOMAIN_REGISTRY:
        return cleaned
    tokens = re.findall(r"[a-z_]+", cleaned)
    for token in tokens:
        if token in DOMAIN_REGISTRY:
            return token
    for key in DOMAIN_REGISTRY:
        if key in cleaned:
            return key
    return _FALLBACK_DOMAIN


async def detect_domain(
    requirement: str,
    url: str,
    adapter: OllamaAdapter | None = None,
    model: str = "qwen2.5-coder:7b-instruct-q4_K_M",
) -> str:
    """
    Auto-detect the application domain from a natural-language requirement and
    target URL.  Returns one of DOMAIN_REGISTRY's keys (e.g. "ecommerce",
    "authentication").  Falls back to "crud_operations" when the LLM call
    fails or the response cannot be matched.
    """
    messages = _build_domain_detect_messages(requirement, url)

    async def _call(a: OllamaAdapter) -> str:
        try:
            raw = await a.generate(messages)
            detected = _parse_domain_response(raw)
            logger.info(
                f"detect_domain: requirement={requirement[:60]!r} → {detected}"
            )
            return detected
        except Exception as exc:
            logger.warning(
                f"detect_domain failed ({exc}); falling back to {_FALLBACK_DOMAIN}"
            )
            return _FALLBACK_DOMAIN

    if adapter is not None:
        return await _call(adapter)
    async with OllamaAdapter(model=model) as own_adapter:
        return await _call(own_adapter)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers — code extraction & validation (Phase 3)
# ──────────────────────────────────────────────────────────────────────────────


def _strip_code_fences(code: str) -> str:
    """Remove ```python / ``` fences that LLMs occasionally re-introduce."""
    code = code.strip()
    code = re.sub(r"^```(?:python|py)?\s*", "", code, flags=re.IGNORECASE)
    code = re.sub(r"\s*```\s*$", "", code)
    return code.strip()


def _has_test_function(code: str) -> bool:
    """Return True iff *code* contains a top-level FunctionDef starting with test_."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test_"):
                return True
    return False


def _validate_example(record: dict[str, Any]) -> tuple[bool, str]:
    """
    Validate a single ChatML record (Phase 3 filter).

    Returns (is_valid, reason).  *reason* is empty when is_valid is True.
    """
    messages = record.get("messages") or []
    if not messages or messages[-1].get("role") != "assistant":
        return False, "no assistant turn"

    code = _strip_code_fences(str(messages[-1].get("content") or ""))
    if not code:
        return False, "empty assistant code"

    example_type = (record.get("metadata") or {}).get("type", "")

    # ── SyntaxError check (always) ────────────────────────────────────────────
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc}"

    # ── test_ function check (skip for single-line healing examples) ─────────
    if example_type != "healing":
        if not _has_test_function(code):
            return False, "no test_ function found"

    # ── Forbidden patterns (CSS / XPath / hardcoded waits) ───────────────────
    if _FORBIDDEN_RE.search(code):
        match = _FORBIDDEN_RE.search(code)
        return False, f"forbidden pattern: {match.group(0)!r}"

    return True, ""


# ──────────────────────────────────────────────────────────────────────────────
# Generator
# ──────────────────────────────────────────────────────────────────────────────


class SyntheticDataGenerator:
    """
    Three-phase synthetic dataset producer.

      1. generate per-domain examples (scenarios + edge cases) via the LLM,
      2. emit 50 healing examples from templated broken→fixed pairs,
      3. validate every example with `ast.parse` + forbidden-pattern checks
         and write only the survivors to the JSONL output file.

    Resume capability:
      The output file is read at startup and any (domain, scenario, type)
      key already present is skipped so an interrupted run can pick up where
      it left off.
    """

    def __init__(
        self,
        adapter: OllamaAdapter,
        output_file: str | Path = _DEFAULT_OUTPUT,
        masker: PIIMasker | None = None,
        enabled_domains: list[str] | None = None,
        healing_count: int = 50,
    ) -> None:
        self.adapter = adapter
        self.output_file = Path(os.path.expanduser(str(output_file)))
        self.masker = masker or PIIMasker(log_masks=False)
        self.enabled_domains = enabled_domains or list(DOMAIN_REGISTRY.keys())
        self.healing_count = healing_count

        unknown = [d for d in self.enabled_domains if d not in DOMAIN_REGISTRY]
        if unknown:
            raise ValueError(f"Unknown domains in enabled_domains: {unknown}")

        # Statistics — populated during run.
        self.stats: dict[str, int] = {
            "generated": 0,
            "discarded_syntax": 0,
            "discarded_missing_test": 0,
            "discarded_forbidden": 0,
            "discarded_other": 0,
            "saved": 0,
            "skipped_resume": 0,
        }

    # ── Public entry point ───────────────────────────────────────────────────

    async def generate_dataset(
        self,
        max_per_domain: int | None = None,
        target: int | None = None,   # kept for backwards-compat (ignored)
    ) -> int:
        """
        Run the full 3-phase pipeline and return the number of examples
        successfully written.

        * max_per_domain – cap on scenarios/edge_cases per domain (None = all).
          When set, the healing budget is also reduced to max_per_domain * 4
          so smoke tests stay fast.
        * target – legacy parameter, accepted but not used.
        """
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        already = self._load_existing_keys()
        if already:
            logger.info(
                f"SyntheticDataGenerator: resume detected — "
                f"{len(already)} examples already present, will skip duplicates"
            )

        # ── Plan tasks for the progress bar ──────────────────────────────────
        phase1_tasks: list[tuple[str, str, str]] = []   # (domain, scenario, type)
        for domain in self.enabled_domains:
            info = DOMAIN_REGISTRY[domain]
            scenarios = info["scenarios"]
            edge_cases = info["edge_cases"]
            if max_per_domain is not None:
                scenarios = scenarios[:max_per_domain]
                edge_cases = edge_cases[:max_per_domain]
            for scenario in scenarios:
                phase1_tasks.append((domain, scenario, "happy_path"))
            for edge in edge_cases:
                phase1_tasks.append((domain, edge, "edge_case"))

        # Healing budget
        healing_budget = (
            self.healing_count if max_per_domain is None
            else max(len(_HEALING_TEMPLATES), max_per_domain * len(_HEALING_TEMPLATES))
        )

        total_planned = len(phase1_tasks) + healing_budget
        logger.info(
            f"SyntheticDataGenerator: planned phase1={len(phase1_tasks)} "
            f"phase2_healing={healing_budget} domains={len(self.enabled_domains)} "
            f"output={self.output_file}"
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task(
                "Generating synthetic dataset", total=total_planned
            )

            # ── PHASE 1: per-domain scenario + edge_case examples ────────────
            progress.update(task, description="[cyan]Phase 1[/cyan] per-domain examples")
            for domain, scenario, example_type in phase1_tasks:
                key = self._make_key(domain, scenario, example_type)
                if key in already:
                    self.stats["skipped_resume"] += 1
                    progress.advance(task)
                    continue
                record = await self._phase1_generate(domain, scenario, example_type)
                self._finalise_record(record)
                progress.advance(task)

            # ── PHASE 2: healing examples ────────────────────────────────────
            progress.update(task, description="[cyan]Phase 2[/cyan] healing examples")
            for record in self._phase2_healing(healing_budget, already):
                self._finalise_record(record)
                progress.advance(task)

        # ── PHASE 3 has been applied inline by _finalise_record ──────────────
        self._log_stats()
        return self.stats["saved"]

    # ── Phase 1 — LLM-generated full tests ────────────────────────────────────

    async def _phase1_generate(
        self,
        domain: str,
        scenario: str,
        example_type: str,
    ) -> dict[str, Any] | None:
        """
        Ask the LLM for a single pytest-playwright test, then return the
        ChatML record (no validation yet) or None on hard failure.
        """
        user_content = (
            f"Domain: {domain}\n"
            f"Scenario: {scenario}\n"
            "Write a complete pytest-playwright test function for this scenario."
        )
        if example_type == "edge_case":
            user_content += _PHASE1_EDGE_SUFFIX

        messages = [
            {"role": "system", "content": _PHASE1_SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]

        for attempt in range(3):
            try:
                raw = await self.adapter.generate(messages)
                code = _strip_code_fences(raw)
                if not code:
                    raise ValueError("empty LLM response")
                self.stats["generated"] += 1
                return {
                    "messages": [
                        {"role": "system", "content": _PHASE1_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": code},
                    ],
                    "metadata": {
                        "domain": domain,
                        "scenario": scenario,
                        "type": example_type,
                    },
                }
            except Exception as exc:
                logger.debug(
                    f"phase1 attempt {attempt + 1}/3 "
                    f"[{domain}/{example_type}/{scenario[:40]}…]: {exc}"
                )
                if attempt == 2:
                    logger.warning(
                        f"phase1: giving up on {domain}/{example_type}/"
                        f"{scenario[:40]}: {exc}"
                    )
        return None

    # ── Phase 2 — template-driven healing pair examples ──────────────────────

    def _phase2_healing(
        self,
        budget: int,
        already: set[str],
    ) -> list[dict[str, Any]]:
        """
        Produce healing examples by filling templated broken→fixed pairs
        with domain-specific element vocabulary.

        Returns up to *budget* records (some may be skipped if their key is
        already present in *already*).
        """
        records: list[dict[str, Any]] = []
        rng = random.Random(42)   # deterministic for testability
        domains_cycle = list(self.enabled_domains)
        idx = 0

        while len(records) < budget and idx < budget * 3:   # safety bound
            template_no = idx % len(_HEALING_TEMPLATES)
            domain = domains_cycle[(idx // len(_HEALING_TEMPLATES)) % len(domains_cycle)]
            template = _HEALING_TEMPLATES[template_no]
            vocab = _HEALING_VOCAB.get(domain, _HEALING_VOCAB["crud_operations"])

            tokens = {
                key: rng.choice(values)
                for key, values in vocab.items()
            }
            broken = template["broken"].format(**tokens)
            fixed  = template["fixed"].format(**tokens)

            ax_frag = rng.choice(_AXTREE_FRAGMENTS).format(
                heading=tokens.get("button_name", "Page"),
                input_label=tokens.get("input_label", "Input"),
                button_name=tokens.get("button_name", "Submit"),
            )

            scenario_key = f"healing_t{template_no}_{tokens.get('button_id') or tokens.get('input_id') or 'x'}"
            key = self._make_key(domain, scenario_key, "healing")
            idx += 1
            if key in already:
                self.stats["skipped_resume"] += 1
                continue

            user_content = (
                f"Broken: {broken}\n"
                f"AxTree:\n{ax_frag}"
            )
            record = {
                "messages": [
                    {"role": "system", "content": _PHASE2_HEALING_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": fixed},
                ],
                "metadata": {
                    "domain": domain,
                    "scenario": scenario_key,
                    "type": "healing",
                },
            }
            self.stats["generated"] += 1
            records.append(record)

        return records

    # ── Phase 3 — validation + persistence ────────────────────────────────────

    def _finalise_record(self, record: dict[str, Any] | None) -> None:
        """Validate, PII-mask, and append a single record (no-op for None)."""
        if record is None:
            self.stats["discarded_other"] += 1
            return

        ok, reason = _validate_example(record)
        if not ok:
            if "Syntax" in reason:
                self.stats["discarded_syntax"] += 1
            elif "test_" in reason:
                self.stats["discarded_missing_test"] += 1
            elif "forbidden" in reason:
                self.stats["discarded_forbidden"] += 1
            else:
                self.stats["discarded_other"] += 1
            meta = record.get("metadata") or {}
            logger.warning(
                f"phase3 discard [{meta.get('domain')}/"
                f"{meta.get('type')}/{str(meta.get('scenario'))[:40]}]: {reason}"
            )
            return

        masked = self.masker.mask_json(record)
        self._append_jsonl(masked)
        self.stats["saved"] += 1

    def _append_jsonl(self, record: dict[str, Any]) -> None:
        with self.output_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _log_stats(self) -> None:
        s = self.stats
        logger.info(
            "SyntheticDataGenerator finished | "
            f"generated={s['generated']} "
            f"saved={s['saved']} "
            f"skipped_resume={s['skipped_resume']} "
            f"discarded(syntax={s['discarded_syntax']}, "
            f"missing_test={s['discarded_missing_test']}, "
            f"forbidden={s['discarded_forbidden']}, "
            f"other={s['discarded_other']}) "
            f"→ {self.output_file}"
        )

    # ── Resume helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _make_key(domain: str, scenario: str, example_type: str) -> str:
        raw = f"{domain}::{example_type}::{scenario}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    def _load_existing_keys(self) -> set[str]:
        if not self.output_file.exists():
            return set()
        keys: set[str] = set()
        try:
            with self.output_file.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    meta = record.get("metadata") or {}
                    domain = str(meta.get("domain", ""))
                    scenario = str(meta.get("scenario", ""))
                    example_type = str(meta.get("type", ""))
                    if domain and scenario and example_type:
                        keys.add(self._make_key(domain, scenario, example_type))
        except OSError:
            return set()
        return keys


# ──────────────────────────────────────────────────────────────────────────────
# Module-level convenience entry points
# ──────────────────────────────────────────────────────────────────────────────


async def generate_synthetic_dataset(
    max_per_domain: int | None = None,
    model: str = "qwen2.5-coder:7b-instruct-q4_K_M",
    output_file: str = _DEFAULT_OUTPUT,
    enabled_domains: list[str] | None = None,
    healing_count: int = 50,
) -> int:
    """
    Smoke-test friendly convenience wrapper. Constructs a temporary
    OllamaAdapter, runs the three-phase pipeline, and returns the number of
    valid examples written to *output_file*.
    """
    async with OllamaAdapter(model=model) as adapter:
        gen = SyntheticDataGenerator(
            adapter=adapter,
            output_file=output_file,
            enabled_domains=enabled_domains,
            healing_count=healing_count,
        )
        return await gen.generate_dataset(max_per_domain=max_per_domain)


# Backwards-compatibility alias (older callers use generate_dataset(...))
async def generate_dataset(
    model: str = "qwen2.5-coder:7b-instruct-q4_K_M",
    output_file: str = _DEFAULT_OUTPUT,
    target: int | None = None,
    enabled_domains: list[str] | None = None,
) -> int:
    return await generate_synthetic_dataset(
        max_per_domain=None,
        model=model,
        output_file=output_file,
        enabled_domains=enabled_domains,
    )
