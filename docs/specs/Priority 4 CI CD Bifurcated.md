You are an Elite CI/CD Security Architect. Build a complete "Secure Bifurcated CI/CD Pipeline" 
for the Production-Grade Universal Local AI QA Agent project.

═══════════════════════════════════════════════════════════
KNOWLEDGE BASE CONSTRAINTS (MUST ENFORCE — NON-NEGOTIABLE)
═══════════════════════════════════════════════════════════

BIFURCATION SECURITY RULES:
1. Untrusted Zone: MUST trigger ONLY on `pull_request` event — read-only GITHUB_TOKEN, ZERO secrets
2. Privileged Zone: MUST trigger on `workflow_run` — never checkout untrusted head.sha
3. All uploaded XML/JSON artifacts MUST be treated as untrusted data payloads in the downstream job
4. NEVER use `pull_request_target` trigger — it grants untrusted code access to base secrets
5. NEVER inject PR input (titles, labels, branch names) directly into shell commands unsanitized
6. Use OIDC short-lived credentials — NEVER long-lived cloud keys

EXIT CODE PROPAGATION RULES:
- Map ALL Pytest exit codes: 0=OK, 1=FAILED, 2=INTERRUPTED, 3=INTERNAL_ERROR, 4=USAGE_ERROR, 5=NO_TESTS
- sys.exit(exit_code.value) MUST be the last call — never let wrapper implicitly return 0
- NEVER append `|| true` to test commands in shell scripts

JUNIT XML TELEMETRY RULES:
- Embed AI metadata via <properties> tags ONLY — no custom XML tags outside this schema
- Use uniquely enumerated property names: attachment1, attachment2 (never duplicate keys)
- `if: always()` conditional MUST guard artifact upload steps

GITHUB APP / GITOPS RULES:
- Use dedicated GitHub App for write operations — NEVER default GITHUB_TOKEN for write
- CODEOWNERS must protect .github/workflows/ and agent config files
- Branch protection must prevent workflow from approving its own PRs

═════════════════════════════
FILES TO CREATE (EXACT PATHS)
═════════════════════════════

──────────────────────────────────────────────────────────
FILE 1: .github/workflows/01_untrusted_test_execution.yml
──────────────────────────────────────────────────────────
GitHub Actions workflow — Untrusted Zone.
SPEC:
- trigger: pull_request (branches: [main, develop])
- permissions: contents: read, checks: read (NOTHING else)
- NO secrets injected — security guard step must fail with exit 126 if PROD_API_KEY detected
- jobs.run_tests:
    runs-on: ubuntu-latest
    steps:
      1. checkout at github.sha (base repo SHA — NOT head.sha)
      2. setup-python 3.11
      3. cache pip dependencies
      4. pip install -r requirements-ci.txt
      5. install-playwright browsers (chromium only)
      6. run: python ci/runners/untrusted_runner.py
         env: TEST_ENV=staging (NO secrets)
      7. upload-artifact (if: always()):
           name: untrusted-test-results-${{ github.run_id }}
           path: workspace/untrusted_reports/
           retention-days: 3

──────────────────────────────────────────────────────────
FILE 2: .github/workflows/02_privileged_reporting.yml
──────────────────────────────────────────────────────────
GitHub Actions workflow — Privileged Zone.
SPEC:
- trigger: workflow_run (workflows: ["01 Untrusted Test Execution"], types: [completed])
- permissions: checks: write, pull-requests: write, contents: read, id-token: write
- jobs.parse_and_report:
    runs-on: ubuntu-latest
    steps:
      1. configure-aws-credentials via OIDC (role-to-assume from secrets.AWS_ROLE_ARN)
      2. download-artifact from triggering workflow_run (using dawidd6/action-download-artifact)
      3. validate XML schema step: python ci/validators/xml_schema_validator.py 
         (treats artifact as UNTRUSTED — sanitize before parsing)
      4. publish-test-results (dorny/test-reporter): 
           reporter: java-junit, files: workspace/untrusted_reports/results.xml
      5. annotate PR with summary using GitHub API (gh pr comment) — sanitized output only
      6. quality gate step: fail job if failure_count > 0 from parsed XML

──────────────────────────────────────────────────────────
FILE 3: ci/runners/untrusted_runner.py
──────────────────────────────────────────────────────────
Async Python runner — executes in zero-secret context.
SPEC:
- class UntrustedTestRunner with async run() method
- Security guard: scan os.environ for any key containing "KEY", "SECRET", "TOKEN", "PASSWORD" → sys.exit(126)
- Run pytest via asyncio.create_subprocess_exec (NOT subprocess.run):
    args: ["pytest", "tests/e2e/", "-v", "--numprocesses=auto", 
           "--junitxml=workspace/untrusted_reports/results.xml",
           "--tb=short", "--tracing=retain-on-failure",
           "--screenshot=only-on-failure"]
- Stream stdout/stderr in real-time via asyncio (not communicate())
- Map returncode to Pytest ExitCode enum for structured logging
- Guarantee sys.exit(returncode) — never fall through
- Type: fully typed with Python 3.11 type hints

──────────────────────────────────────────────────────────
FILE 4: ci/runners/privileged_reporter.py  
──────────────────────────────────────────────────────────
Privileged artifact parser — runs ONLY in base repo context.
SPEC:
- class PrivilegedReporter
- async parse_junit_xml(xml_path: Path) → TestSuiteReport (Pydantic V2 model)
- Pydantic model TestSuiteReport: total, passed, failed, skipped, errors, 
  ai_semantic_scores: list[float], execution_latencies: list[float], 
  playwright_trace_urls: list[str]
- XML parsing: use defusedxml (NOT stdlib xml.etree) to prevent XXE attacks
- Extract <properties> tags: ai_semantic_consistency_score, e2e_execution_latency_sec, 
  attachment1..N, failure_trace
- Quality gate: async enforce_quality_gate(report) → raises QualityGateError if:
    * failed > 0
    * mean(ai_semantic_scores) < 0.85 (when scores present)
    * error count > 0
- Return structured JSON summary for PR annotation

──────────────────────────────────────────────────────────
FILE 5: ci/validators/xml_schema_validator.py
──────────────────────────────────────────────────────────
XSD-based XML schema validator — treats all artifacts as untrusted.
SPEC:
- Validate JUnit XML against embedded XSD schema (standard JUnit + properties extension)
- Use lxml for XSD validation
- Reject any XML with: DOCTYPE declarations, external entity refs, 
  non-standard root elements, properties outside <testcase>/<testsuite>
- CLI entrypoint: python xml_schema_validator.py <path_to_xml>
- sys.exit(0) = valid, sys.exit(1) = invalid (with detailed error to stderr)
- Log all validation errors without printing raw XML content (prevent log injection)

──────────────────────────────────────────────────────────
FILE 6: ci/conftest_ci.py
──────────────────────────────────────────────────────────
Pytest conftest for CI — JUnit XML telemetry + observability hooks.
SPEC:
- pytest_configure: enforce markers (e2e, smoke, regression, ai_generated)
- Fixture: async_page with tracing=retain-on-failure baked in
- Hook: pytest_runtest_makereport:
    * on failure: capture screenshot → base64 → record_property("attachment1", data_uri)
    * always: record_property("e2e_execution_latency_sec", elapsed)
    * if item has ai_score attr: record_property("ai_semantic_consistency_score", score)
    * record_property("execution_count", retry_count) for flaky detection
- Hook: pytest_sessionfinish — write summary JSON to workspace/untrusted_reports/session_summary.json
- NO secrets read in this file — zero env var access to production credentials

──────────────────────────────────────────────────────────
FILE 7: ci/security/secret_guard.py
──────────────────────────────────────────────────────────
Reusable security module imported by untrusted_runner.py.
SPEC:
- FORBIDDEN_PATTERNS: list of regex patterns for secret-like env vars
  [".*KEY.*", ".*SECRET.*", ".*TOKEN.*", ".*PASSWORD.*", ".*CREDENTIAL.*", 
   ".*API_KEY.*", "PROD_.*", "AWS_SECRET.*"]
- function scan_environment() → list[str]: returns list of detected forbidden var names
- function enforce_clean_environment() → None: 
    if violations: log each name (NOT value), sys.exit(126)
- function sanitize_shell_input(value: str) → str: 
    strip shell metacharacters: ; | & $ ` \ " ' < > ( ) { } [ ] * ? ~
    raise ValueError if sanitized != original (log warning, never execute)
- All functions fully typed, no external dependencies

──────────────────────────────────────────────────────────
FILE 8: requirements-ci.txt
──────────────────────────────────────────────────────────
CI-only dependencies (no LLM/GPU packages):
pytest==8.3.x
pytest-asyncio==0.24.x
pytest-xdist==3.6.x
pytest-timeout==2.3.x
playwright==1.44.x
pydantic==2.7.x
defusedxml==0.7.x
lxml==5.2.x
anyio==4.4.x
python-dotenv==1.0.x (load_dotenv NOT called in untrusted context)
jellyfish==1.0.x

──────────────────────────────────────────────────────────
FILE 9: CODEOWNERS
──────────────────────────────────────────────────────────
Protect critical paths:
.github/workflows/   @qa-platform-team
ci/security/         @security-team
.claude/             @qa-platform-team
AGENTS.md            @qa-platform-team

══════════════════════════════
IMPLEMENTATION RULES (STRICT)
══════════════════════════════

PYTHON RULES:
- Python 3.11+ only — use match/case for exit code mapping
- All async I/O: asyncio — no threading, no sync subprocess.run in async context  
- Every function: fully typed with return types
- dataclasses or Pydantic V2 for all structured data (no plain dicts)
- pathlib.Path everywhere — no os.path string concatenation

SECURITY RULES:
- defusedxml.ElementTree for ALL XML parsing — never stdlib xml.etree
- Input sanitization BEFORE any shell interpolation
- Never log env var values — log names only
- Never print raw XML content to stdout in validator

WORKFLOW RULES:
- Workflow 01: permissions block must be explicit read-only
- Workflow 02: OIDC id-token: write permission is ONLY in privileged workflow
- Both workflows: pin ALL action versions to SHA (not @v3 tags)
- Cache pip with hashFiles('requirements-ci.txt') key

OUTPUT DELIVERABLE:
Produce all 9 files completely. Start each file with:
# ============================================================
# FILE: <exact_path>
# ============================================================
Never use placeholders. Every file must be complete and runnable.
After all files, output a "CI/CD Architecture Summary" section explaining 
the trust boundary between Workflow 01 and Workflow 02 in 5 bullet points.