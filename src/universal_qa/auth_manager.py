from __future__ import annotations

import random
import string

from loguru import logger
from playwright.async_api import Page


class AuthManager:
    """Handles authentication for any website.

    Strategy:
    1. If username + password provided → find login form → fill → submit
    2. Else → find register form → create qa_test_<rand>@mailinator.com → login
    3. If no auth form found → return None (unauthenticated)
    """

    _LOGIN_FORM_SELECTORS = [
        'form[action*="login"]', 'form[action*="signin"]',
        'form[id*="login"]', 'form[id*="signin"]',
        'input[type="password"]',
    ]
    _REGISTER_FORM_SELECTORS = [
        'form[action*="register"]', 'form[action*="signup"]',
        'form[id*="register"]', 'form[id*="signup"]',
        'a[href*="register"]', 'a[href*="signup"]',
    ]

    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        self._username = username
        self._password = password

    def _strategy(self) -> str:
        return "login" if (self._username and self._password) else "auto_register"

    async def setup(self, page: Page) -> tuple[str, str] | None:
        """Authenticate on the current page. Returns (username, password) or None."""
        if self._strategy() == "login":
            return await self._do_login(page)
        return await self._do_auto_register(page)

    async def _do_login(self, page: Page) -> tuple[str, str] | None:
        has_form = await self._has_selector(page, self._LOGIN_FORM_SELECTORS)
        if not has_form:
            logger.info("AuthManager: no login form found — skipping auth")
            return None
        try:
            await page.get_by_role("textbox", name="email").fill(
                self._username or "", timeout=5_000
            )
        except Exception:
            try:
                inputs = page.get_by_role("textbox")
                if await inputs.count() >= 1:
                    await inputs.first.fill(self._username or "", timeout=5_000)
            except Exception:
                pass
        try:
            await page.get_by_label("password").fill(self._password or "", timeout=5_000)
        except Exception:
            try:
                await page.locator("input[type=password]").first.fill(
                    self._password or "", timeout=5_000
                )
            except Exception:
                pass
        try:
            await page.get_by_role("button", name="login").click(timeout=5_000)
        except Exception:
            try:
                await page.get_by_role("button", name="sign in").click(timeout=5_000)
            except Exception:
                pass
        await page.wait_for_load_state("domcontentloaded", timeout=30_000)
        logger.info(f"AuthManager: login attempted for {self._username}")
        return (self._username or "", self._password or "")

    async def _do_auto_register(self, page: Page) -> tuple[str, str] | None:
        has_register = await self._has_selector(page, self._REGISTER_FORM_SELECTORS)
        if not has_register:
            logger.info("AuthManager: no register form found — skipping auth")
            return None
        rand = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
        email = f"qa_test_{rand}@mailinator.com"
        password = "QaTest123!"
        name = f"QA Test {rand[:4]}"

        try:
            reg_link = page.get_by_role("link", name="signup")
            if not await reg_link.count():
                reg_link = page.get_by_role("link", name="register")
            if await reg_link.count():
                await reg_link.first.click(timeout=5_000)
                await page.wait_for_load_state("domcontentloaded", timeout=30_000)
        except Exception:
            pass

        try:
            name_input = page.get_by_role("textbox", name="name")
            if await name_input.count():
                await name_input.first.fill(name, timeout=5_000)
        except Exception:
            pass
        try:
            await page.get_by_role("textbox", name="email").fill(email, timeout=5_000)
        except Exception:
            pass
        try:
            for pw_input in await page.locator("input[type=password]").all():
                await pw_input.fill(password, timeout=5_000)
        except Exception:
            pass
        try:
            await page.get_by_role("button", name="signup").click(timeout=5_000)
        except Exception:
            try:
                await page.get_by_role("button", name="register").click(timeout=5_000)
            except Exception:
                pass
        await page.wait_for_load_state("domcontentloaded", timeout=30_000)
        logger.info(f"AuthManager: auto-registered as {email}")
        self._username = email
        self._password = password
        return (email, password)

    @staticmethod
    async def _has_selector(page: Page, selectors: list[str]) -> bool:
        for sel in selectors:
            try:
                el = await page.query_selector(sel)
                if el is not None:
                    return True
            except Exception:
                pass
        return False


__all__ = ["AuthManager"]
