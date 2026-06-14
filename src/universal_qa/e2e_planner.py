from __future__ import annotations

from loguru import logger
from pydantic import BaseModel

from src.llm.instructor_client import InstructorClient
from src.universal_qa.explorer.nav_map import NavigationMap
from src.universal_qa.models import TestCase


class _E2EFlowItem(BaseModel):
    title: str
    priority: str  # "high" / "medium" / "low"
    preconditions: list[str]
    steps: list[str]
    expected_outcome: str


class _E2EResponse(BaseModel):
    test_cases: list[_E2EFlowItem]


class E2EFlowPlanner:
    """Generates multi-page E2E test flows using LLM."""

    def __init__(self, client: InstructorClient) -> None:
        self._client = client

    async def plan(
        self,
        nav_map: NavigationMap,
        username: str | None = None,
        password: str | None = None,
    ) -> list[TestCase]:
        if not username or not password:
            return []

        # สร้าง page summary สำหรับ LLM
        page_lines = []
        for p in nav_map.pages[:12]:
            actions = ", ".join(a.action_label for a in p.actions[:5] if not a.is_destructive)
            page_lines.append(f"- {p.url} | {p.title} | actions: {actions or 'none'}")

        page_summary = "\n".join(page_lines) or "ไม่พบหน้า"

        # หา login URL
        login_url = next(
            (p.url for p in nav_map.pages if "login" in p.url.lower() or "login" in p.title.lower()),
            nav_map.base_url,
        )

        prompt = (
            f"คุณเป็น QA Engineer กรุณาเขียน E2E test flows 2-4 flows "
            f"ที่ข้ามหลายหน้าในรูปแบบ JSON **ตอบเป็นภาษาไทยทั้งหมด**\n\n"
            f"Login URL: {login_url}\n"
            f"Credentials: username={username}, password={password}\n"
            f"หน้าที่พบ:\n{page_summary}\n\n"
            f"กฎสำคัญ:\n"
            f"- แต่ละ flow ต้องเริ่มจาก: เปิดหน้า {login_url}\n"
            f"- ต้อง login ก่อนทำอย่างอื่น: กรอก \"Username\" ด้วย {username} และ กรอก \"Password\" ด้วย {password}\n"
            f"- steps ต้องข้ามหลายหน้า (อย่างน้อย 3 URL ต่าง)\n"
            f"- ใช้ URL เต็มเสมอ เช่น: เปิดหน้า https://...\n"
            f"- ชื่อปุ่มต้องมี double quotes เช่น: คลิกปุ่ม \"Login\"\n"
            f"- ห้ามใช้ path string หรือ snake_case\n"
            f"- ตัวอย่าง flow ที่ดี: login → เพิ่มสินค้า → checkout → ยืนยันคำสั่งซื้อ\n\n"
            f"ส่งกลับ JSON: test_cases list ที่มี title, priority, preconditions, steps, expected_outcome"
        )

        try:
            response: _E2EResponse = await self._client.create_structured(
                prompt, _E2EResponse, temperature=0.1
            )
            results = []
            for item in response.test_cases:
                results.append(TestCase(
                    title=item.title,
                    type="e2e",
                    priority=item.priority if item.priority in ("high", "medium", "low") else "medium",
                    preconditions=item.preconditions,
                    steps=item.steps,
                    expected_outcome=item.expected_outcome,
                    source_url=login_url,
                ))
            logger.info(f"E2EFlowPlanner: generated {len(results)} flows")
            return results
        except Exception as exc:
            logger.warning(f"E2EFlowPlanner: failed — {exc!r}")
            return []


__all__ = ["E2EFlowPlanner"]
