from __future__ import annotations

import base64
import datetime
import pathlib

from src.universal_qa.models import TestResult

_CSS = """
body{font-family:system-ui,sans-serif;margin:0;padding:20px;background:#f5f5f5}
h1{color:#333}.summary{display:flex;gap:16px;margin-bottom:24px}
.badge{padding:6px 16px;border-radius:6px;font-weight:700;font-size:14px}
.pass{background:#d4edda;color:#155724}.fail{background:#f8d7da;color:#721c24}
.info{background:#d1ecf1;color:#0c5460}
.filters{margin-bottom:16px}
.filters button{margin-right:8px;padding:4px 12px;cursor:pointer;border:1px solid #ccc;border-radius:4px}
.test-card{background:#fff;border-radius:8px;padding:16px;margin-bottom:12px;border-left:4px solid #ccc}
.test-card.passed{border-left-color:#28a745}.test-card.failed{border-left-color:#dc3545}
.test-title{font-size:16px;font-weight:600;margin-bottom:8px}
.test-meta{font-size:12px;color:#888;margin-bottom:8px}
.step{padding:4px 0;font-size:13px}
.step.passed::before{content:"✓ ";color:#28a745}
.step.failed::before{content:"✗ ";color:#dc3545}
.failure-reason{background:#fff3cd;padding:8px;border-radius:4px;font-size:13px;margin-top:8px}
.screenshot{margin-top:8px}
.screenshot img{max-width:100%;border:1px solid #ddd;border-radius:4px}
"""

_JS = """
function filter(type){
  document.querySelectorAll('.test-card').forEach(c=>{
    c.style.display=(type==='all'||c.dataset.type===type||c.dataset.status===type)?'':'none';
  });
}
"""


class HTMLReporter:
    """Generates a self-contained HTML QA report (no external dependencies)."""

    def __init__(self, output_dir: pathlib.Path | None = None) -> None:
        self._output_dir = output_dir or pathlib.Path("reports")

    def generate(self, results: list[TestResult]) -> pathlib.Path:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self._output_dir / f"qa_report_{ts}.html"

        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed

        cards = "\n".join(self._render_card(r) for r in results)
        html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>QA Report {ts}</title>
<style>{_CSS}</style></head>
<body>
<h1>QA Report — {ts}</h1>
<div class="summary">
  <span class="badge info">Total: {len(results)}</span>
  <span class="badge pass">Passed: {passed}</span>
  <span class="badge fail">Failed: {failed}</span>
</div>
<div class="filters">
  <button onclick="filter('all')">All</button>
  <button onclick="filter('functional')">Functional</button>
  <button onclick="filter('accessibility')">Accessibility</button>
  <button onclick="filter('security')">Security</button>
  <button onclick="filter('passed')">Passed</button>
  <button onclick="filter('failed')">Failed</button>
</div>
{cards}
<script>{_JS}</script>
</body></html>"""
        path.write_text(html, encoding="utf-8")
        return path

    def _render_card(self, result: TestResult) -> str:
        tc = result.test_case
        status = "passed" if result.passed else "failed"
        steps_html = "".join(
            f'<div class="step {t.status}">{t.step}'
            + (f' — <em>{t.error}</em>' if t.error else "")
            + f' <small style="color:#aaa">({t.detail})</small></div>'
            for t in result.steps_trace
        )
        failure_html = (
            f'<div class="failure-reason">&#9888; {result.failure_reason}</div>'
            if result.failure_reason else ""
        )
        screenshot_html = ""
        if result.screenshot_path:
            try:
                img_bytes = pathlib.Path(result.screenshot_path).read_bytes()
                b64 = base64.b64encode(img_bytes).decode()
                screenshot_html = (
                    f'<div class="screenshot">'
                    f'<img src="data:image/png;base64,{b64}" alt="screenshot"/>'
                    f'</div>'
                )
            except Exception:
                pass
        precond_html = (
            "<ul>" + "".join(f"<li>{p}</li>" for p in tc.preconditions) + "</ul>"
            if tc.preconditions else ""
        )
        return f"""<div class="test-card {status}" data-type="{tc.type}" data-status="{status}">
  <div class="test-title">{tc.title}</div>
  <div class="test-meta">{tc.type.upper()} | priority: {tc.priority} | {result.duration_ms}ms | {tc.source_url}</div>
  {precond_html}
  {steps_html}
  {failure_html}
  {screenshot_html}
</div>"""


__all__ = ["HTMLReporter"]
