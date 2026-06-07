from __future__ import annotations

from typing import Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from src.contractskill.sfg import SFGNode, SFGStore
from src.llm.instructor_client import InstructorClient, StructuredGenerationError
from src.universal_qa.models import TestCase

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
                f"กฎการเขียน steps (สำคัญมาก):\n"
                f"- ขั้นตอน navigate ต้องใช้ URL เต็มเสมอ เช่น: navigate to {node.url}\n"
                f"- ขั้นตอน click ต้องใส่ข้อความบนปุ่ม/ลิงก์ใน double quotes เช่น: click button \"Login\"\n"
                f"- ขั้นตอน fill ต้องระบุชื่อ field ใน double quotes เช่น: fill \"Email Address\" with test@test.com\n"
                f"- ห้ามใช้ path เช่น /login หรือ /products ในขั้นตอน ให้ใช้ URL เต็มหรือชื่อปุ่มแทน\n\n"
                f"ส่งกลับ JSON ที่มี field 'test_cases': รายการ object ที่มี "
                f"title (ชื่อ test case ภาษาไทย), priority (high/medium/low), "
                f"preconditions (รายการเงื่อนไขก่อนทดสอบ ภาษาไทย), "
                f"steps (รายการขั้นตอนตามกฎด้านบน อย่างน้อย 1 ขั้นตอน), "
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


__all__ = ["UniversalTestPlanner"]
