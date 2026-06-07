from __future__ import annotations

from loguru import logger
from playwright.async_api import Page

from src.universal_qa.auth_manager import AuthManager

_LOGIN_TITLE_KEYWORDS = ("login", "sign in", "signin", "เข้าสู่ระบบ")


class SessionGuard:
    """Detects when a session has been lost (kicked back to login) and re-authenticates."""

    def __init__(self, auth: AuthManager, max_attempts: int = 3) -> None:
        self._auth = auth
        self._max_attempts = max_attempts
        self._attempts = 0

    async def is_session_lost(self, page: Page) -> bool:
        try:
            has_password = await page.query_selector("input[type=password]")
            if has_password is not None:
                return True
        except Exception:
            pass
        try:
            title = (await page.title()) or ""
            if any(k in title.lower() for k in _LOGIN_TITLE_KEYWORDS):
                return True
        except Exception:
            pass
        return False

    async def recover(self, page: Page) -> bool:
        """Attempt re-auth. Returns False if attempts exhausted."""
        if self._attempts >= self._max_attempts:
            logger.warning("SessionGuard: max re-auth attempts reached")
            return False
        self._attempts += 1
        try:
            await self._auth.setup(page)
            logger.info(f"SessionGuard: re-authenticated (attempt {self._attempts})")
            return True
        except Exception as exc:
            logger.warning(f"SessionGuard: re-auth failed — {exc!r}")
            return False


__all__ = ["SessionGuard"]
