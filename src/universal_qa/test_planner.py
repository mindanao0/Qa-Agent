from __future__ import annotations

from typing import Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from src.contractskill.sfg import SFGNode, SFGStore
from src.llm.instructor_client import InstructorClient, StructuredGenerationError
from src.universal_qa.models import TestCase
from src.universal_qa.explorer.nav_map import (
    ExploredAction, ExploredPage, NavigationFlow, NavigationMap,
)

_PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}
_XSS_PAYLOAD = "<script>alert('xss')</script>"
_SQLI_PAYLOAD = "' OR '1'='1"


class _FuncItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str
    priority: Literal["high", "medium", "low"] = "medium"
    preconditions: list[str] = Field(default_factory=list)
    steps: list[str] = Field(..., min_length=1)
    expected_outcome: str


class _FuncResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    test_cases: list[_FuncItem]


class UniversalTestPlanner:
    """Generates TestCase list from SFGStore nodes.

    Functional: LLM analyses each form-node's PAM content.
    Accessibility: rule-based, 1 TestCase per node.
    Security: template XSS + SQLi per form-node.
    """

    def __init__(self) -> None:
        self._client = InstructorClient()

    async def plan(self, sfg_store: SFGStore, start_url: str) -> list[TestCase]:
        nodes = sfg_store.get_nodes_by_url_prefix(start_url)
        if not nodes:
            logger.warning("UniversalTestPlanner: no SFG nodes found for %s", start_url)
            return []

        functional = await self._plan_functional(nodes, start_url)
        accessibility = self._plan_accessibility(nodes, start_url)
        security = self._plan_security(nodes, start_url)

        all_cases = functional + accessibility + security
        return self._sort_by_priority(all_cases)

    async def _plan_functional(
        self, nodes: list[SFGNode], start_url: str
    ) -> list[TestCase]:
        results: list[TestCase] = []
        form_nodes = [n for n in nodes if "form" in n.coverage_tags]
        if not form_nodes:
            form_nodes = nodes[:5]  # fallback: use first 5 pages

        for node in form_nodes[:10]:  # cap at 10 to avoid LLM overload
            prompt = (
                f"คุณเป็น QA Engineer กรุณาเขียน test case 2-3 ข้อ (อย่างน้อย 1 happy-path และ 1 negative) "
                f"สำหรับหน้าเว็บนี้ในรูปแบบ JSON **ตอบเป็นภาษาไทยทั้งหมด**\n\n"
                f"URL ของหน้านี้: {node.url}\n"
                f"ชื่อหน้า: {node.page_title}\n"
                f"สรุปเนื้อหาหน้า:\n{node.pam_content[:800]}\n\n"
                f"กฎการเขียน steps (สำคัญมาก — ใช้ภาษาไทยล้วน):\n"
                f"- ขั้นตอนเปิดหน้า ต้องใช้ URL เต็มเสมอ เช่น: เปิดหน้า {node.url}\n"
                f"- ขั้นตอนคลิก ต้องใส่ข้อความบนปุ่ม/ลิงก์ใน double quotes เช่น: คลิกปุ่ม \"Login\"\n"
                f"- ขั้นตอนกรอกข้อมูล ต้องระบุชื่อ field ใน double quotes เช่น: กรอก \"Email Address\" ด้วย test@test.com\n"
                f"- ห้ามใช้ path เช่น /login หรือ /products ให้ใช้ URL เต็มหรือชื่อปุ่มแทนเสมอ\n\n"
                f"ส่งกลับ JSON ที่มี field 'test_cases': รายการ object ที่มี "
                f"title (ชื่อ test case ภาษาไทย), priority (high/medium/low), "
                f"preconditions (รายการเงื่อนไขก่อนทดสอบ ภาษาไทย), "
                f"steps (รายการขั้นตอนภาษาไทยตามกฎด้านบน อย่างน้อย 1 ขั้นตอน), "
                f"expected_outcome (ผลลัพธ์ที่คาดหวัง ภาษาไทย)"
            )
            try:
                response: _FuncResponse = await self._client.create_structured(
                    prompt, _FuncResponse, temperature=0.1
                )
                for item in response.test_cases:
                    results.append(TestCase(
                        title=item.title,
                        type="functional",
                        priority=item.priority,
                        preconditions=item.preconditions,
                        steps=item.steps,
                        expected_outcome=item.expected_outcome,
                        source_url=node.url,
                    ))
            except (StructuredGenerationError, Exception) as exc:
                logger.warning(f"UniversalTestPlanner: LLM failed for {node.url}: {exc!r}")
                results.append(TestCase(
                    title=f"ตรวจสอบว่าหน้า {node.page_title} โหลดได้",
                    type="functional",
                    priority="medium",
                    steps=[f"เปิดหน้า {node.url}", "ตรวจสอบว่าชื่อหน้าแสดงขึ้นมา"],
                    expected_outcome="หน้าเว็บโหลดสำเร็จโดยไม่มี error",
                    source_url=node.url,
                ))
        return results

    def _plan_accessibility(
        self, nodes: list[SFGNode], start_url: str
    ) -> list[TestCase]:
        return [
            TestCase(
                title=f"ตรวจสอบ Accessibility: {node.page_title}",
                type="accessibility",
                priority="medium",
                preconditions=[f"อยู่ที่หน้า {node.url}"],
                steps=[
                    f"เปิดหน้า {node.url}",
                    "ตรวจสอบว่า element ที่โต้ตอบได้ทุกตัวมี accessible name",
                    "ตรวจสอบว่ารูปภาพทุกรูปมี alt text",
                    "ตรวจสอบว่าไม่มี decorative role บน element ที่โต้ตอบได้",
                ],
                expected_outcome="ไม่พบการละเมิดกฎ WCAG",
                source_url=node.url,
            )
            for node in nodes
        ]

    def _plan_security(
        self, nodes: list[SFGNode], start_url: str
    ) -> list[TestCase]:
        results: list[TestCase] = []
        for node in nodes:
            if "form" not in node.coverage_tags:
                continue
            results.append(TestCase(
                title=f"ทดสอบ XSS injection: {node.page_title}",
                type="security",
                priority="high",
                preconditions=[f"อยู่ที่หน้า {node.url}"],
                steps=[
                    f"เปิดหน้า {node.url}",
                    f"กรอก XSS payload ลงทุกช่องรับข้อมูล: {_XSS_PAYLOAD}",
                    "กด submit form",
                    "ตรวจสอบว่า payload ไม่ถูก execute",
                ],
                expected_outcome="หน้าเว็บไม่ execute script payload",
                source_url=node.url,
            ))
            results.append(TestCase(
                title=f"ทดสอบ SQL injection: {node.page_title}",
                type="security",
                priority="high",
                preconditions=[f"อยู่ที่หน้า {node.url}"],
                steps=[
                    f"เปิดหน้า {node.url}",
                    f"กรอก SQLi payload ลงทุกช่องรับข้อมูล: {_SQLI_PAYLOAD}",
                    "กด submit form",
                    "ตรวจสอบว่าไม่มี SQL error โชว์",
                ],
                expected_outcome="หน้าเว็บไม่เปิดเผย SQL error หรือข้อมูลที่ไม่ตั้งใจ",
                source_url=node.url,
            ))
        return results

    @staticmethod
    def _sort_by_priority(cases: list[TestCase]) -> list[TestCase]:
        return sorted(cases, key=lambda tc: _PRIORITY_ORDER.get(tc.priority, 1))

    async def plan_from_map(self, nav_map: NavigationMap) -> list[TestCase]:
        """Generate test cases from a NavigationMap (preferred over plan())."""
        per_page = await self._plan_from_pages(nav_map.pages)
        functional = await self._plan_from_pages_functional(nav_map.pages, nav_map.base_url)
        negative = await self._plan_from_pages_negative(nav_map)
        edge = await self._plan_from_pages_edge(nav_map)
        flows = self._plan_flows(nav_map.flows)
        form_val = self._plan_form_validation(nav_map.pages)
        broken = self._plan_broken_link(nav_map.pages, nav_map.base_url)
        error_pg = self._plan_error_page(nav_map.base_url)
        search = self._plan_search(nav_map.pages)
        logout = self._plan_logout_flow(nav_map.pages, nav_map.base_url)
        return self._sort_by_priority(
            per_page + functional + negative + edge + flows +
            form_val + broken + error_pg + search + logout
        )

    def _plan_form_validation(self, pages: list[ExploredPage]) -> list[TestCase]:
        results = []
        form_pages = [p for p in pages if "form" in (p.pam_content or "").lower() or "input" in (p.pam_content or "").lower()]
        for page in form_pages[:5]:
            results.append(TestCase(
                title=f"Form validation — required fields: {page.title or page.url}",
                type="form_validation",
                priority="high",
                preconditions=[f"อยู่ที่หน้า {page.url}"],
                steps=[
                    f"navigate to {page.url}",
                    'click "Submit"',
                    'verify text: "required"',
                ],
                expected_outcome="แสดง validation error สำหรับ required fields",
                source_url=page.url,
            ))
            results.append(TestCase(
                title=f"Form validation — invalid email: {page.title or page.url}",
                type="form_validation",
                priority="medium",
                preconditions=[f"อยู่ที่หน้า {page.url}"],
                steps=[
                    f"navigate to {page.url}",
                    'fill "Email" with "not-an-email"',
                    'click "Submit"',
                    'verify text: "invalid"',
                ],
                expected_outcome="แสดง error สำหรับ email format ไม่ถูกต้อง",
                source_url=page.url,
            ))
        return results

    def _plan_broken_link(self, pages: list[ExploredPage], base_url: str) -> list[TestCase]:
        results = []
        for page in pages[:5]:
            results.append(TestCase(
                title=f"Broken link check: {page.title or page.url}",
                type="broken_link",
                priority="medium",
                preconditions=[f"อยู่ที่หน้า {page.url}"],
                steps=[f"navigate to {page.url}"],
                expected_outcome="ทุก link บนหน้าต้อง return HTTP 200",
                source_url=page.url,
            ))
        return results

    def _plan_error_page(self, base_url: str) -> list[TestCase]:
        return [TestCase(
            title="Error page — 404 handling",
            type="error_page",
            priority="low",
            preconditions=[],
            steps=[
                f"navigate to {base_url}/this-page-does-not-exist-qa-test-12345",
                'verify text: "404"',
            ],
            expected_outcome="เว็บแสดง 404 error page แทนที่จะ crash",
            source_url=base_url,
        )]

    def _plan_search(self, pages: list[ExploredPage]) -> list[TestCase]:
        results = []
        search_pages = [p for p in pages if "search" in (p.pam_content or "").lower()]
        for page in search_pages[:3]:
            results.append(TestCase(
                title=f"Search functionality: {page.title or page.url}",
                type="search",
                priority="medium",
                preconditions=[f"อยู่ที่หน้า {page.url}"],
                steps=[
                    f"navigate to {page.url}",
                    'fill "Search" with "test"',
                    'click "Search"',
                ],
                expected_outcome="ผลการค้นหาปรากฏ",
                source_url=page.url,
            ))
        return results

    def _plan_logout_flow(self, pages: list[ExploredPage], base_url: str) -> list[TestCase]:
        logout_pages = [
            p for p in pages
            if any("logout" in (a.action_label or "").lower() or "sign out" in (a.action_label or "").lower()
                   for a in p.actions)
        ]
        if not logout_pages:
            return []
        page = logout_pages[0]
        return [TestCase(
            title="Logout flow",
            type="logout_flow",
            priority="high",
            preconditions=["เข้าสู่ระบบแล้ว"],
            steps=[
                f"navigate to {page.url}",
                'click "Logout"',
            ],
            expected_outcome="redirect กลับไปหน้า login หลัง logout",
            source_url=page.url,
        )]

    async def _plan_from_pages_functional(
        self, pages: list[ExploredPage], base_url: str = ""
    ) -> list[TestCase]:
        """Per-action prompts — ถาม LLM ทีละ element แทนทั้งหน้า."""
        _post_login_kw = ("inventory", "cart", "checkout", "dashboard", "profile", "account")
        _login_kw = ("login", "signin", "sign-in", "register", "signup")

        results: list[TestCase] = []
        for page in pages:
            non_destr = [a for a in page.actions if not a.is_destructive]
            if not non_destr:
                results.append(TestCase(
                    title=f"ตรวจสอบว่าหน้า {page.title or page.url} โหลดได้",
                    type="functional", priority="medium",
                    steps=[f"เปิดหน้า {page.url}", "ตรวจสอบว่าชื่อหน้าแสดงขึ้นมา"],
                    expected_outcome="หน้าเว็บโหลดสำเร็จโดยไม่มี error",
                    source_url=page.url,
                ))
                continue

            _is_post_login = (
                any(k in page.url.lower() for k in _post_login_kw)
                or (
                    page.url.rstrip("/") != base_url.rstrip("/")
                    and not any(k in page.url.lower() for k in _login_kw)
                )
            ) if base_url else any(k in page.url.lower() for k in _post_login_kw)
            login_note = (
                "ห้าม generate step เกี่ยวกับ login เช่น click Login หรือ navigate /login\n"
                if _is_post_login else ""
            )

            for action in non_destr[:20]:  # max 20 elements per page
                name = action.element_name or action.action_label
                role = action.element_role or "element"
                dest = f" (นำไปยัง {action.leads_to_url})" if action.leads_to_url else ""
                prompt = (
                    f"เขียน test case 1 ข้อสำหรับ element นี้ (JSON ภาษาไทย)\n"
                    f"{login_note}"
                    f"หน้า: {page.url}\n"
                    f"Element: {role} \"{name}\"{dest}\n\n"
                    f"กฎ steps: ภาษาไทย, ชื่อปุ่มใน double quotes\n"
                    f"ตัวอย่าง: คลิกปุ่ม \"Add to cart\"\n\n"
                    f"JSON: {{\"test_cases\": [{{\"title\": \"...\", "
                    f"\"priority\": \"high\"|\"medium\"|\"low\", "
                    f"\"preconditions\": [...], \"steps\": [\"...\", ...], "
                    f"\"expected_outcome\": \"...\"}}]}}"
                )
                try:
                    resp: _FuncResponse = await self._client.create_structured(
                        prompt, _FuncResponse, temperature=0.1
                    )
                    for item in resp.test_cases[:1]:
                        results.append(TestCase(
                            title=item.title,
                            type="functional",
                            priority=item.priority,
                            preconditions=item.preconditions,
                            steps=item.steps,
                            expected_outcome=item.expected_outcome,
                            source_url=page.url,
                        ))
                except Exception as exc:
                    logger.warning(f"Per-action LLM failed '{name}': {exc!r}")
                    step = f'คลิกปุ่ม "{name}"' if role == "button" else f'คลิก "{name}"'
                    results.append(TestCase(
                        title=f"ทดสอบ {action.action_label}",
                        type="functional", priority="medium",
                        preconditions=[f"อยู่ที่หน้า {page.url}"],
                        steps=[f"เปิดหน้า {page.url}", step],
                        expected_outcome=f"หน้าเว็บตอบสนองต่อการคลิก {name} ได้ถูกต้อง",
                        source_url=page.url,
                    ))
        return results

    async def _plan_from_pages_negative(
        self, nav_map: NavigationMap
    ) -> list[TestCase]:
        """Per-action negative test cases — focus on form-like elements only."""
        _form_roles = frozenset({"button", "textbox", "combobox", "searchbox", "spinbutton"})
        _form_kw = ("submit", "sign", "login", "register", "search", "send", "save", "add")

        results: list[TestCase] = []
        for page in nav_map.pages:
            form_actions = [
                a for a in page.actions
                if not a.is_destructive
                and (
                    (a.element_role or "") in _form_roles
                    or any(k in (a.action_label or "").lower() for k in _form_kw)
                )
            ]
            if not form_actions and "form" not in (page.pam_content or "").lower():
                continue
            if not form_actions:
                form_actions = [a for a in page.actions[:3] if not a.is_destructive]
            for action in form_actions[:3]:  # max 3 per page
                name = action.element_name or action.action_label
                role = action.element_role or "element"
                prompt = (
                    f"เขียน negative test case 1 ข้อสำหรับ element นี้ (JSON ภาษาไทย)\n"
                    f"หน้า: {page.url}\n"
                    f"Element: {role} \"{name}\"\n\n"
                    f"เน้น: กรอกข้อมูลผิด/ว่าง → validation error\n"
                    f"กฎ steps: ภาษาไทย, ชื่อ field ใน double quotes\n"
                    f"JSON: {{\"test_cases\": [{{\"title\": \"...\", \"priority\": ..., "
                    f"\"preconditions\": [...], \"steps\": [...], \"expected_outcome\": \"...\"}}]}}"
                )
                try:
                    resp = await self._client.create_structured(
                        prompt, _FuncResponse, temperature=0.1
                    )
                    for item in resp.test_cases[:1]:
                        results.append(TestCase(
                            title=item.title,
                            type="functional",
                            priority=item.priority,
                            preconditions=item.preconditions,
                            steps=item.steps,
                            expected_outcome=item.expected_outcome,
                            source_url=page.url,
                        ))
                except Exception as exc:
                    logger.warning(f"Negative gen failed '{name}': {exc!r}")
        return results

    async def _plan_from_pages_edge(
        self, nav_map: NavigationMap
    ) -> list[TestCase]:
        """Edge case per page — prompt ตัด PAM content ออก เหลือแค่ action summary."""
        results: list[TestCase] = []
        for page in nav_map.pages:
            if not page.actions:
                continue
            action_summary = ", ".join(
                f"\"{a.action_label}\"" for a in page.actions[:5] if not a.is_destructive
            ) or "ไม่พบ action"
            prompt = (
                f"เขียน edge case test case 1 ข้อสำหรับหน้า (JSON ภาษาไทย)\n"
                f"หน้า: {page.url}\n"
                f"Actions: {action_summary}\n\n"
                f"เน้น: empty state, boundary values, back button\n"
                f"กฎ steps: ภาษาไทย, ชื่อปุ่มใน double quotes\n"
                f"JSON: {{\"test_cases\": [{{\"title\": \"...\", \"priority\": ..., "
                f"\"preconditions\": [...], \"steps\": [...], \"expected_outcome\": \"...\"}}]}}"
            )
            try:
                resp = await self._client.create_structured(
                    prompt, _FuncResponse, temperature=0.1
                )
                for item in resp.test_cases[:1]:
                    results.append(TestCase(
                        title=item.title,
                        type="functional",
                        priority=item.priority,
                        preconditions=item.preconditions,
                        steps=item.steps,
                        expected_outcome=item.expected_outcome,
                        source_url=page.url,
                    ))
            except Exception as exc:
                logger.warning(f"Edge gen failed for {page.url}: {exc!r}")
        return results

    async def _plan_from_pages(self, pages: list[ExploredPage]) -> list[TestCase]:
        results: list[TestCase] = []
        for page in pages:
            # Accessibility — one per page
            results.append(TestCase(
                title=f"ตรวจสอบ Accessibility: {page.title}",
                type="accessibility", priority="medium",
                preconditions=[f"อยู่ที่หน้า {page.url}"],
                steps=[
                    f"เปิดหน้า {page.url}",
                    "ตรวจสอบว่า element ที่โต้ตอบได้ทุกตัวมี accessible name",
                    "ตรวจสอบว่ารูปภาพทุกรูปมี alt text",
                    "ตรวจสอบว่าไม่มี decorative role บน element ที่โต้ตอบได้",
                ],
                expected_outcome="ไม่พบการละเมิดกฎ WCAG",
                source_url=page.url,
            ))
            # Security — XSS + SQLi for any page that has actions
            if page.actions:
                for kind, payload in (("XSS", _XSS_PAYLOAD), ("SQL", _SQLI_PAYLOAD)):
                    results.append(TestCase(
                        title=f"ทดสอบ {kind} injection: {page.title}",
                        type="security", priority="high",
                        preconditions=[f"อยู่ที่หน้า {page.url}"],
                        steps=[
                            f"เปิดหน้า {page.url}",
                            f"กรอก {kind} payload ลงทุกช่องรับข้อมูล: {payload}",
                            "กด submit form",
                            f"ตรวจสอบว่า payload ไม่ทำงาน",
                        ],
                        expected_outcome=f"หน้าเว็บไม่ได้รับผลกระทบจาก {kind} payload",
                        source_url=page.url,
                    ))
        return results

    def _plan_flows(self, flows: list[NavigationFlow]) -> list[TestCase]:
        cases: list[TestCase] = []
        for flow in flows:
            steps = [self._action_to_step(a) for a in flow.steps]
            cases.append(TestCase(
                title=f"ทดสอบ flow {flow.name} ตั้งแต่ต้นจนจบ",
                type="functional", priority="high",
                preconditions=[f"เริ่มที่หน้า {flow.start_url}"],
                steps=steps or [f"เปิดหน้า {flow.start_url}"],
                expected_outcome=f"เข้าหน้า {flow.end_url} สำเร็จ",
                source_url=flow.start_url,
            ))
        return cases

    @staticmethod
    def _action_to_step(action: ExploredAction) -> str:
        if action.leads_to_url:
            return f"เปิดหน้า {action.leads_to_url}"
        name = action.element_name or action.action_label
        if action.element_role == "button":
            return f'คลิกปุ่ม "{name}"'
        if action.element_role == "link":
            return f'คลิก "{name}"'
        return f'คลิก "{name}"'


__all__ = ["UniversalTestPlanner"]
