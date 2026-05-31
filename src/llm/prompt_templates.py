"""
PTCF (Problem / Task / Context / Format) and RTF (Role / Task / Format) prompt
templates for every agent role in the QA system.

Design conventions:
- RTF  → system prompts that establish the agent's identity and output contract.
- PTCF → user prompts that frame the specific problem instance.
- All templates expose a `.render(**kwargs)` method for safe str.format_map()
  substitution.  Unknown placeholders are left intact rather than raising.

UPDATE 2 — Universal Domain Expansion:
- Removed all hardcoded HRM/Payroll bias from system prompts.
- Added UNIVERSAL_SYSTEM_PROMPT shared by all agent personas.
- Templates accept {domain} and {domain_hint} so the same prompt works for
  authentication, e-commerce, finance, healthcare, ERP/CRM, CMS, etc.
"""

from __future__ import annotations

import string
from dataclasses import dataclass, field


class _SafeFormatter(string.Formatter):
    """Leaves unknown {placeholders} intact instead of raising KeyError."""

    def get_value(self, key: int | str, args: list, kwargs: dict) -> object:  # type: ignore[override]
        if isinstance(key, str):
            return kwargs.get(key, f"{{{key}}}")
        return super().get_value(key, args, kwargs)


_fmt = _SafeFormatter()


@dataclass
class PromptTemplate:
    system: str
    user: str
    description: str = ""
    tags: list[str] = field(default_factory=list)

    def render_system(self, **kwargs: object) -> str:
        return _fmt.format(self.system, **kwargs)

    def render_user(self, **kwargs: object) -> str:
        return _fmt.format(self.user, **kwargs)

    def to_messages(self, **kwargs: object) -> list[dict[str, str]]:
        """Return an OpenAI-style messages list ready for the adapter."""
        messages: list[dict[str, str]] = []
        rendered_system = self.render_system(**kwargs)
        if rendered_system.strip():
            messages.append({"role": "system", "content": rendered_system})
        messages.append({"role": "user", "content": self.render_user(**kwargs)})
        return messages


# ──────────────────────────────────────────────────────────────────────────────
# Universal system prompt — shared by every agent persona
# ──────────────────────────────────────────────────────────────────────────────

UNIVERSAL_SYSTEM_PROMPT = (
    "You are an expert QA automation engineer specializing in web application testing.\n"
    "You have deep knowledge of testing patterns across all domains including:\n"
    "e-commerce, finance, healthcare, ERP/CRM, HRM/payroll, CMS, authentication systems,\n"
    "and any other web application type.\n"
    "You write robust Playwright Python tests using semantic locators only.\n"
    "You think about happy paths, negative cases, edge cases, and RBAC boundaries.\n"
    "You never assume the domain — you adapt to whatever system you are given."
)


# ──────────────────────────────────────────────────────────────────────────────
# Domain-specific coding hints — injected into Generator/Planner prompts
# Kept small on purpose: 1-2 sentences each, only the highest-value reminder.
# ──────────────────────────────────────────────────────────────────────────────

DOMAIN_CODING_HINTS: dict[str, str] = {
    "authentication": (
        "Verify both success and failure flows. Assert the post-login URL/role landing "
        "and check that error messages render on bad credentials."
    ),
    "crud_operations": (
        "Cover create → read → update → delete in order. Assert list refresh after every "
        "mutation; verify confirmation dialogs and pagination boundaries."
    ),
    "forms_validation": (
        "Test all validation error messages explicitly — assert the error text for each "
        "invalid field, not just that the form failed to submit."
    ),
    "rbac_permissions": (
        "Test both allowed and denied actions. For denied actions, assert the 403 / "
        "access-denied UI element OR that the protected route redirects away."
    ),
    "ecommerce": (
        "Verify cart totals with exact decimal precision. Always check post-action state "
        "(cart count, line totals, applied coupon banner) — not just that the click worked."
    ),
    "finance_banking": (
        "Verify numbers with exact decimal precision (use string match on formatted amounts). "
        "Always check pre-condition (balance) before transfer-style actions."
    ),
    "healthcare": (
        "Verify patient/record identity in every assertion (ID, MRN, name) to guard against "
        "cross-record bleed. Date-pickers must be validated against business calendar."
    ),
    "hrm_payroll": (
        "Verify currency amounts with exact decimal precision (e.g. THB with 2 decimals). "
        "Check role-based redirects after login and progressive calculation breakdowns."
    ),
    "erp_crm": (
        "Verify state transitions explicitly (lead→opportunity→quote→order). After every "
        "transition, assert the new status badge AND that prior-stage actions are disabled."
    ),
    "cms_content": (
        "Verify publish state changes (draft → published → archived) via badge/label. "
        "For uploads, assert preview + size + filename, not just successful upload."
    ),
    "dashboard_analytics": (
        "Wait for charts to fully render before asserting — use wait_for_load_state('networkidle') "
        "or wait for a known chart element. Assert against KPI text, never against pixel values."
    ),
    "notifications_messaging": (
        "Assert badge count changes, read/unread state, AND notification content text. "
        "For real-time flows, wait for the websocket message via expect(...).to_be_visible()."
    ),
    "settings_configuration": (
        "After save, reload the page and re-assert the value persisted. Confirm sensitive "
        "fields (passwords, API keys) are masked on display."
    ),
}


def get_domain_hint(domain: str) -> str:
    """Return a short coding hint for *domain*, or a neutral default."""
    return DOMAIN_CODING_HINTS.get(
        domain,
        "Cover happy path, negative cases, edge cases, and RBAC boundaries. "
        "Use semantic Playwright locators only.",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Planner Agent  (RTF) — domain-agnostic
# Role: senior QA architect who converts requirements into structured test plans
# ──────────────────────────────────────────────────────────────────────────────

PlannerPromptTemplate = PromptTemplate(
    description="RTF — Planner agent: requirement → TestPlan JSON (universal domain)",
    tags=["planner", "rtf", "universal"],
    system=UNIVERSAL_SYSTEM_PROMPT + """

# Task
Convert the given requirement into a structured test plan that works for ANY web
application domain (authentication, e-commerce, finance, healthcare, ERP/CRM,
HRM/payroll, CMS, dashboards, settings, etc.).

Your plan must cover:
1. Happy-path scenarios for the primary flow.
2. Negative / edge-case scenarios (invalid input, boundary values, network errors).
3. RBAC boundary checks (which roles CAN and CANNOT access the feature).
4. Domain-specific business rules — e.g. for finance: check balance BEFORE
   transfer; for forms: assert every validation error string explicitly; for
   dashboards: wait for charts to fully render before asserting.

# Domain Awareness
The detected application domain is "{domain}". Use this hint to shape the plan,
but never hardcode assumptions — if the requirement contradicts the domain hint,
trust the requirement.

Domain coding hint:
{domain_hint}

# Format — respond with ONLY valid JSON matching this schema (no markdown fences):
{{
  "title": "<concise test plan title>",
  "requirement_summary": "<one-sentence summary of what is being tested>",
  "estimated_complexity": "low" | "medium" | "high",
  "domain": "{domain}",
  "domain_specific_notes": ["<domain-specific business rule or pitfall>"],
  "steps": [
    {{
      "step_number": 1,
      "description": "<what this step verifies>",
      "action": "<Playwright action to perform>",
      "expected_result": "<assertion to make>",
      "role": "<which RBAC role executes this step>",
      "preconditions": ["<optional precondition>"]
    }}
  ],
  "rbac_scenarios": ["<role> cannot access <feature>"],
  "edge_cases": ["<edge case description>"]
}}

Rules:
- Use only Playwright semantic locators: get_by_role, get_by_label, get_by_text, get_by_test_id.
- Never include CSS selectors or XPath in step descriptions.
- estimated_complexity = "high" if the flow involves currency math, multi-role
  orchestration, or more than 5 RBAC permutations.
- Temperature is 0.1 — be deterministic and precise.""",
    user="""\
# Problem
Requirement:
{requirement}

# Context
Target URL: {url}
Authenticated role: {role}
Detected application domain: {domain}

{rag_context}

# Task
Produce the TestPlan JSON described in the system prompt.
Think through every scenario — happy path, negative, edge case, RBAC — before
writing the JSON. Tailor the steps and domain_specific_notes to the "{domain}"
domain when relevant; never assume HRM/payroll context unless the requirement
clearly indicates it.""",
)


# ──────────────────────────────────────────────────────────────────────────────
# Generator Agent  (PTCF) — domain-agnostic
# Problem: test plan → executable Playwright Python code
# ──────────────────────────────────────────────────────────────────────────────

GeneratorPromptTemplate = PromptTemplate(
    description="PTCF — Generator agent: TestPlan + RAG context → PlaywrightScript JSON (universal)",
    tags=["generator", "ptcf", "universal"],
    system=UNIVERSAL_SYSTEM_PROMPT + """

# Strict Coding Rules (violations cause automatic retry)
1. Use ONLY semantic locators: page.get_by_role(), page.get_by_label(),
   page.get_by_text(), page.get_by_test_id().
2. NEVER use CSS selectors, XPath, or page.locator() with raw CSS/XPath strings.
3. NEVER call page.wait_for_timeout() or asyncio.sleep().
   Use page.wait_for_load_state(), page.wait_for_selector(), or expect() assertions.
4. Place ALL assertions at the test function level — no assertions inside helper functions.
5. Wrap every major action group in a try/except that calls the self-healing decorator.
6. Import structure: standard lib → third-party → local src imports.
7. Every test function must be async and prefixed with test_.
8. Use pytest-asyncio with @pytest.mark.asyncio.
9. IMPORTANT: page.url is a property, NOT a coroutine.
   Correct: assert 'secure' in page.url
   Wrong:   assert 'secure' in (await page.url)

# Domain-Specific Coding Hints
The current test belongs to the "{domain}" domain.
{domain_hint}

# Format — respond with ONLY valid JSON (no markdown fences):
{{
  "reasoning": "<step-by-step reasoning about implementation decisions BEFORE writing code>",
  "code": "<complete runnable Python test file as a single string>",
  "locators_used": ["get_by_role", "get_by_label", ...],
  "test_function_name": "test_<descriptive_name>"
}}""",
    user="""\
# Problem
Generate a complete Playwright Python async test for the following test plan.

# Test Plan
{test_plan_json}

# Context
Target URL: {url}
Role under test: {role}
Detected application domain: {domain}

Retrieved documentation and examples:
{rag_context}

Accessibility tree snapshot (pruned) for current page state:
{axtree}

# Task
1. Reason through each test step, choosing the best semantic locator.
2. Apply the domain-specific coding hint above when writing assertions.
3. Write the complete, runnable async test file.
4. Validate internally that no forbidden patterns are present before outputting.
5. Output the PlaywrightScript JSON.""",
)


# ──────────────────────────────────────────────────────────────────────────────
# Healer Agent  (PTCF) — fully domain-agnostic
# Problem: broken locator + AxTree → healed locator
# ──────────────────────────────────────────────────────────────────────────────

HealerPromptTemplate = PromptTemplate(
    description="PTCF — Healer agent: failed locator + AxTree → HealedLocator JSON (universal)",
    tags=["healer", "ptcf", "universal"],
    system=UNIVERSAL_SYSTEM_PROMPT + """

# Role
You are also an AI-powered test self-healing engine. Your job is to find a working
replacement for a broken Playwright locator by analysing the current page's
Accessibility Tree (AxTree). You operate identically across every domain — your
heuristics depend on the AxTree structure, never on assumed business context.

# Healing Strategy
1. Identify WHY the original locator broke (element renamed, moved, ARIA label
   changed, role mismatch, etc.).
2. Scan the AxTree for the element that best matches the INTENT of the original
   locator — match on role, accessible name, surrounding context.
3. Prefer locator strategies in this order:
   get_by_role > get_by_label > get_by_text > get_by_test_id.
4. Assign a confidence score (0.0–1.0) reflecting how certain you are the
   replacement is semantically equivalent to the original.

# Format — respond with ONLY valid JSON (no markdown fences):
{{
  "reasoning": "<diagnosis of why the locator broke and why you chose the replacement>",
  "original": "<the failing locator string>",
  "healed": "<the new Playwright locator expression>",
  "confidence": 0.0-1.0,
  "method": "ai"
}}""",
    user="""\
# Problem
The following Playwright locator failed during test execution:

Failing locator: {failed_locator}
Attempted action: {action}
Error message: {error_message}

# Context
Current page URL: {page_url}
Current page AxTree (pruned, max {max_nodes} nodes):
{axtree}

Previously known good locators for this page:
{known_locators}

# Task
Analyse the AxTree and produce a HealedLocator JSON that replaces the failing
locator with a semantically equivalent one that exists in the current DOM.
Do not make any assumptions about the application domain.""",
)


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic Data Generator  (RTF) — universal
# Role: QA expert generating realistic training examples across all domains
# ──────────────────────────────────────────────────────────────────────────────

SyntheticGenPromptTemplate = PromptTemplate(
    description="RTF — Synthetic data generator: domain → SyntheticQAExample JSON (universal)",
    tags=["synthetic", "rtf", "universal"],
    system=UNIVERSAL_SYSTEM_PROMPT + """

# Task
Generate a single synthetic training example for the given application domain
and scenario type. The example must be:
- Realistic (plausible URL structure, field names, and business logic for the
  given domain).
- Correct (generated code must follow all Playwright best practices).
- Domain-appropriate (use field names, terminology, and assertions that fit the
  given domain — DO NOT default to HRM/Payroll unless the domain is hrm_payroll).

# Coding Rules for generated output_code
- Only semantic locators: get_by_role, get_by_label, get_by_text, get_by_test_id.
- No CSS selectors, XPath, or hardcoded waits.
- Async/await throughout with @pytest.mark.asyncio.
- RBAC scenarios must assert that the forbidden role receives an access-denied response.

# Format — respond with ONLY valid JSON (no markdown fences):
{{
  "instruction": "<natural language test requirement (1-3 sentences)>",
  "input_context": "<additional context: URL, role, preconditions, domain-specific rule>",
  "output_code": "<complete Playwright Python async test as a single string>",
  "domain": "{domain}",
  "scenario_type": "{scenario_type}"
}}""",
    user="""\
# Problem
Generate one synthetic QA training example.

Domain: {domain}
Domain description: {domain_description}
Scenario type: {scenario_type}
Variation index: {variation_index}

Scenario seed (use this as the basis for the test):
{scenario_seed}

# Task
Produce a SyntheticQAExample JSON.
Make the requirement distinct from variation index {variation_index} — avoid
duplicating earlier examples in this domain. The test code must be appropriate
for the "{domain}" domain (use realistic field names, URLs, and business rules
for that domain — not HRM unless domain == hrm_payroll).""",
)


# ──────────────────────────────────────────────────────────────────────────────
# Convenience registry
# ──────────────────────────────────────────────────────────────────────────────

TEMPLATES: dict[str, PromptTemplate] = {
    "planner": PlannerPromptTemplate,
    "generator": GeneratorPromptTemplate,
    "healer": HealerPromptTemplate,
    "synthetic_gen": SyntheticGenPromptTemplate,
}


def get_template(name: str) -> PromptTemplate:
    if name not in TEMPLATES:
        raise KeyError(f"Unknown template '{name}'. Available: {list(TEMPLATES)}")
    return TEMPLATES[name]
