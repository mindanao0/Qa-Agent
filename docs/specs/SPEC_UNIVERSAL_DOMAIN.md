# SPEC_UNIVERSAL_DOMAIN — Universal Domain Expansion (UPDATE 2)

Expand the qa-agent system to be truly Universal — capable of testing ANY web application
domain without hardcoding. Currently the system is biased toward HRM/Payroll only.

Make changes across 4 files as described below.

---

## FILE 1: src/data/synthetic_gen.py

Replace the DOMAIN_SEEDS constant with this comprehensive universal domain registry.
Each domain has: name, description, scenarios list, common_locators list, edge_cases list.

DOMAIN_REGISTRY = {

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
      "SSO redirect flow"
    ],
    "edge_cases": [
      "SQL injection in login fields",
      "XSS in username field",
      "concurrent sessions",
      "session timeout after inactivity",
      "login with special characters in password"
    ]
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
      "export records to CSV/Excel"
    ],
    "edge_cases": [
      "create duplicate record",
      "update record with invalid data",
      "delete record that has dependencies",
      "search with special characters",
      "pagination boundary (first/last page)"
    ]
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
      "multi-step form navigation"
    ],
    "edge_cases": [
      "paste very long text into field",
      "submit form twice rapidly (double submit)",
      "form with file upload validation",
      "dropdown with no options selected",
      "date picker past/future restriction"
    ]
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
      "elevated privilege actions require confirmation"
    ],
    "edge_cases": [
      "user with multiple roles",
      "role changed mid-session",
      "access URL directly without permission",
      "expired session accessing protected route"
    ]
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
      "compare products"
    ],
    "edge_cases": [
      "add out-of-stock product",
      "apply expired coupon",
      "checkout with empty cart",
      "payment declined scenario",
      "product price change during checkout"
    ]
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
      "apply for new credit card"
    ],
    "edge_cases": [
      "transfer amount exceeds balance",
      "transfer to invalid account number",
      "duplicate transaction within 1 minute",
      "transfer during system maintenance window",
      "statement with zero transactions"
    ]
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
      "generate medical invoice"
    ],
    "edge_cases": [
      "book appointment in the past",
      "duplicate patient registration",
      "prescription with controlled substance",
      "patient with allergy conflict in prescription",
      "appointment overlap for same doctor"
    ]
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
      "run overtime calculation"
    ],
    "edge_cases": [
      "payroll with prorated salary",
      "overtime on public holiday (3x rate)",
      "leave request exceeding balance",
      "employee with multiple positions",
      "payroll with mid-month joining date"
    ]
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
      "view sales report by period"
    ],
    "edge_cases": [
      "order with item out of stock",
      "invoice already paid",
      "negative inventory adjustment",
      "purchase order with price variance",
      "customer credit limit exceeded"
    ]
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
      "set SEO metadata for article"
    ],
    "edge_cases": [
      "publish article with no title",
      "upload file exceeding size limit",
      "duplicate article slug",
      "article with broken image link",
      "comment with spam content"
    ]
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
      "save custom dashboard layout"
    ],
    "edge_cases": [
      "dashboard with no data in selected range",
      "filter combination with zero results",
      "export very large dataset",
      "chart with single data point",
      "date range start after end date"
    ]
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
      "subscribe to notification topic"
    ],
    "edge_cases": [
      "notification with very long text",
      "message to blocked user",
      "notification preferences conflict",
      "message with unsupported file type",
      "send message to offline user"
    ]
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
      "view system audit log"
    ],
    "edge_cases": [
      "change password to same as current",
      "upload non-image as profile picture",
      "revoke API key currently in use",
      "invalid SMTP configuration",
      "webhook endpoint returning error"
    ]
  }
}

Update generate_synthetic_dataset() function to:
- Loop through ALL domains in DOMAIN_REGISTRY
- For each domain generate:
  * 8 happy path examples from scenarios list
  * 4 negative/edge case examples from edge_cases list
  * 3 RBAC boundary examples (using rbac_permissions domain scenarios)
- Total target: minimum 200 examples across all domains
- Each example must include domain name as metadata field
- Apply PII masking to all generated content
- Save to ~/.qa-agent/datasets/synthetic_universal.jsonl

Also add a function detect_domain(requirement: str, url: str) -> str that:
- Sends requirement + url to LLM with a short prompt
- LLM picks the closest matching domain name from DOMAIN_REGISTRY keys
- Returns domain name string (e.g. "ecommerce", "authentication")
- Falls back to "crud_operations" if uncertain
- This is used by PlannerAgent to auto-detect domain context

---

## FILE 2: src/llm/prompt_templates.py

Replace ALL hardcoded HRM/Payroll references with universal domain-aware templates.

Add UNIVERSAL_SYSTEM_PROMPT constant:
"You are an expert QA automation engineer specializing in web application testing.
You have deep knowledge of testing patterns across all domains including:
e-commerce, finance, healthcare, ERP/CRM, HRM/payroll, CMS, authentication systems,
and any other web application type.
You write robust Playwright Python tests using semantic locators only.
You think about happy paths, negative cases, edge cases, and RBAC boundaries.
You never assume the domain — you adapt to whatever system you are given."

Add PLANNER_PROMPT_TEMPLATE with {domain}, {requirement}, {url}, {role} placeholders:
The prompt must instruct the planner to:
1. Identify the domain automatically from requirement and URL
2. Consider domain-specific business rules (e.g. for finance: check balance before transfer)
3. Generate test scenarios covering: happy path, negative cases, edge cases, RBAC
4. Output structured test plan regardless of domain

Add GENERATOR_PROMPT_TEMPLATE with {domain}, {plan}, {url}, {reasoning} placeholders:
Must include domain-specific coding hints:
- For finance: "verify numbers with exact decimal precision"
- For forms: "test all validation error messages explicitly"
- For RBAC: "test both allowed and denied actions"
- For dashboard: "wait for charts to fully render before asserting"

Add HEALER_PROMPT_TEMPLATE that is completely domain-agnostic.

---

## FILE 3: src/agents/planner.py

Update PlannerAgent to use domain auto-detection:

1. At start of plan() method, call detect_domain(requirement, url) from synthetic_gen.py
2. Log the detected domain: logger.info(f"PlannerAgent: detected domain={domain}")
3. Inject domain into the planning prompt using PLANNER_PROMPT_TEMPLATE
4. Add domain to the returned TestPlan object as a field
5. Store domain in LangGraph state so generator and healer can use it

Update TestPlan Pydantic schema in src/llm/structured.py to add:
  domain: str  (detected domain name)
  domain_specific_notes: list[str]  (domain-specific things to watch for)

---

## FILE 4: config/agent.yaml

Add a new domains section:

domains:
  auto_detect: true          # use LLM to detect domain from requirement+url
  fallback_domain: "crud_operations"   # used when detection confidence is low
  enabled_domains:           # list of domains active for dataset generation
    - authentication
    - crud_operations
    - forms_validation
    - rbac_permissions
    - ecommerce
    - finance_banking
    - healthcare
    - hrm_payroll
    - erp_crm
    - cms_content
    - dashboard_analytics
    - notifications_messaging
    - settings_configuration

dataset:
  output_path: "~/.qa-agent/datasets/synthetic_universal.jsonl"
  examples_per_domain_happy: 8
  examples_per_domain_edge: 4
  examples_per_domain_rbac: 3

---

## VERIFICATION

After all changes, verify with these tests:

Test 1 — domain detection works:
uv run python -c "
from src.data.synthetic_gen import detect_domain
import asyncio
result = asyncio.run(detect_domain('test add to cart and checkout', 'https://shop.example.com'))
print(f'Detected: {result}')
assert result == 'ecommerce', f'Expected ecommerce got {result}'
print('OK')
"

Test 2 — generate mode works on non-HRM domain:
uv run python main.py --mode generate \
  --requirement "test adding product to cart and applying discount coupon" \
  --url https://the-internet.herokuapp.com \
  --role admin

Test 3 — generate mode works on authentication domain:
uv run python main.py --mode generate \
  --requirement "test user registration with email validation" \
  --url https://the-internet.herokuapp.com/login \
  --role admin

Expected for both tests:
- Log shows detected domain name (not hardcoded hrm)
- Generated test uses correct domain-specific assertions
- Script saved to tests/generated/
