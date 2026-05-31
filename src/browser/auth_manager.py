import json
import os
import time
from pathlib import Path
from typing import Any

import httpx
import yaml
from loguru import logger


class AuthError(Exception):
    pass


class AuthManager:
    """
    Manages Playwright storageState (cookies + localStorage) per RBAC role.

    Login is performed via a single API POST — never via UI automation — so that
    tests can start directly in an authenticated state without re-running the
    login flow on every test execution.

    storageState files are written atomically (write → .tmp → os.replace) to
    prevent corruption from concurrent test runs.
    """

    def __init__(
        self,
        roles_config_path: str | Path = "config/roles.yaml",
        storage_dir: str | Path | None = None,
    ) -> None:
        self._config = self._load_config(Path(roles_config_path))
        self._storage_dir = Path(
            storage_dir
            or os.path.expanduser(
                self._config.get("storage_state_dir", "~/.qa-agent/auth")
            )
        )
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._http = httpx.AsyncClient(timeout=30.0)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    async def get_storage_state(self, role: str) -> dict[str, Any]:
        """
        Return a valid storageState dict for *role*.

        Checks the cached file first; if missing or expired, performs a fresh
        API login and caches the result.
        """
        cached = self._load_cached_state(role)
        if cached is not None:
            logger.debug(f"AuthManager: cache hit for role={role!r}")
            return cached

        logger.info(f"AuthManager: cache miss for role={role!r} — performing API login")
        return await self.login(role)

    async def login(self, role: str) -> dict[str, Any]:
        """Perform an API login for *role* and return the resulting storageState."""
        role_cfg = self._get_role_config(role)
        auth_cfg = self._config.get("auth", {})

        endpoint = self._resolve_env(auth_cfg.get("endpoint", ""))
        payload = {
            k: self._resolve_env(v)
            for k, v in role_cfg.get("login_payload_template", {}).items()
        }

        try:
            response = await self._http.post(
                endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            resp_data = response.json()
        except httpx.HTTPStatusError as exc:
            raise AuthError(
                f"Login failed for role={role!r}: HTTP {exc.response.status_code}"
            ) from exc
        except Exception as exc:
            raise AuthError(f"Login request failed for role={role!r}: {exc}") from exc

        token_field = auth_cfg.get("token_field", "access_token")
        refresh_field = auth_cfg.get("refresh_token_field", "refresh_token")
        expiry_field = auth_cfg.get("expiry_field", "expires_in")

        access_token: str = resp_data.get(token_field, "")
        refresh_token: str = resp_data.get(refresh_field, "")
        expires_in: int = resp_data.get(expiry_field, 3600)

        if not access_token:
            raise AuthError(
                f"Login response for role={role!r} missing field '{token_field}'"
            )

        expiry_timestamp = int(time.time()) + expires_in - auth_cfg.get(
            "expiry_buffer_sec", 60
        )
        base_url = self._resolve_env(
            os.getenv("APP_BASE_URL", "http://localhost:3000")
        )

        storage_state = self._build_storage_state(
            base_url=base_url,
            access_token=access_token,
            refresh_token=refresh_token,
            expiry_timestamp=expiry_timestamp,
        )

        self._save_storage_state(role, storage_state, expiry_timestamp)
        logger.info(
            f"AuthManager: logged in role={role!r} | "
            f"expires_in={expires_in}s | token_len={len(access_token)}"
        )
        return storage_state

    async def refresh_token(self, role: str) -> dict[str, Any]:
        """Attempt a token refresh; falls back to full re-login on failure."""
        auth_cfg = self._config.get("auth", {})
        refresh_endpoint = self._resolve_env(
            auth_cfg.get("refresh_endpoint", "")
        )

        meta = self._load_meta(role)
        stored_refresh = meta.get("refresh_token", "")

        if not refresh_endpoint or not stored_refresh:
            logger.info(f"AuthManager: no refresh endpoint — re-logging in role={role!r}")
            return await self.login(role)

        try:
            response = await self._http.post(
                refresh_endpoint,
                json={auth_cfg.get("refresh_token_field", "refresh_token"): stored_refresh},
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
        except Exception as exc:
            logger.warning(f"Token refresh failed for role={role!r}: {exc} — re-logging in")
            return await self.login(role)

        resp_data = response.json()
        new_access = resp_data.get(auth_cfg.get("token_field", "access_token"), "")
        if not new_access:
            return await self.login(role)

        expires_in = resp_data.get(auth_cfg.get("expiry_field", "expires_in"), 3600)
        expiry_timestamp = int(time.time()) + expires_in - auth_cfg.get("expiry_buffer_sec", 60)
        base_url = os.getenv("APP_BASE_URL", "http://localhost:3000")

        storage_state = self._build_storage_state(
            base_url=base_url,
            access_token=new_access,
            refresh_token=resp_data.get(auth_cfg.get("refresh_token_field", "refresh_token"), stored_refresh),
            expiry_timestamp=expiry_timestamp,
        )
        self._save_storage_state(role, storage_state, expiry_timestamp)
        logger.info(f"AuthManager: token refreshed for role={role!r}")
        return storage_state

    async def close(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "AuthManager":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _load_config(path: Path) -> dict[str, Any]:
        if not path.exists():
            logger.warning(f"AuthManager: roles config not found at {path}")
            return {}
        with path.open() as fh:
            return yaml.safe_load(fh) or {}

    def _get_role_config(self, role: str) -> dict[str, Any]:
        roles = self._config.get("roles", {})
        if role not in roles:
            raise AuthError(
                f"Role '{role}' not found in config. Available: {list(roles)}"
            )
        return roles[role]

    @staticmethod
    def _resolve_env(value: str) -> str:
        """Expand ${VAR} placeholders in config values from environment."""
        import re
        def _sub(m: re.Match) -> str:
            return os.getenv(m.group(1), m.group(0))
        return re.sub(r"\$\{([^}]+)\}", _sub, value)

    @staticmethod
    def _build_storage_state(
        base_url: str,
        access_token: str,
        refresh_token: str,
        expiry_timestamp: int,
    ) -> dict[str, Any]:
        return {
            "cookies": [
                {
                    "name": "access_token",
                    "value": access_token,
                    "domain": base_url.split("//")[-1].split(":")[0],
                    "path": "/",
                    "expires": expiry_timestamp,
                    "httpOnly": True,
                    "secure": base_url.startswith("https"),
                    "sameSite": "Lax",
                }
            ],
            "origins": [
                {
                    "origin": base_url,
                    "localStorage": [
                        {"name": "access_token", "value": access_token},
                        {"name": "refresh_token", "value": refresh_token},
                    ],
                }
            ],
        }

    def _state_path(self, role: str) -> Path:
        role_cfg = self._get_role_config(role)
        configured = role_cfg.get("storage_state_path", "")
        if configured:
            return Path(os.path.expanduser(configured))
        return self._storage_dir / f"{role}.json"

    def _meta_path(self, role: str) -> Path:
        return self._state_path(role).with_suffix(".meta.json")

    def _save_storage_state(
        self,
        role: str,
        state: dict[str, Any],
        expiry_timestamp: int,
    ) -> None:
        """Write storageState atomically via .tmp → os.replace()."""
        state_path = self._state_path(role)
        state_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_state = state_path.with_suffix(".tmp")
        tmp_state.write_text(json.dumps(state, indent=2))
        os.replace(tmp_state, state_path)

        # Save expiry metadata separately so we can check without parsing JWT
        meta = {"expiry_timestamp": expiry_timestamp, "role": role}
        # Persist refresh token in meta for the refresh flow
        for origin in state.get("origins", []):
            for item in origin.get("localStorage", []):
                if item.get("name") == "refresh_token":
                    meta["refresh_token"] = item["value"]
        meta_path = self._meta_path(role)
        tmp_meta = meta_path.with_suffix(".tmp")
        tmp_meta.write_text(json.dumps(meta, indent=2))
        os.replace(tmp_meta, meta_path)

        logger.debug(f"AuthManager: saved storageState for role={role!r} → {state_path}")

    def _load_cached_state(self, role: str) -> dict[str, Any] | None:
        """Return cached state if it exists and is not expired, else None."""
        state_path = self._state_path(role)
        if not state_path.exists():
            return None

        meta = self._load_meta(role)
        expiry = meta.get("expiry_timestamp", 0)
        if int(time.time()) >= expiry:
            logger.debug(f"AuthManager: cached state for role={role!r} is expired")
            return None

        try:
            return json.loads(state_path.read_text())
        except Exception as exc:
            logger.warning(f"AuthManager: failed to read cached state for {role!r}: {exc}")
            return None

    def _load_meta(self, role: str) -> dict[str, Any]:
        meta_path = self._meta_path(role)
        if not meta_path.exists():
            return {}
        try:
            return json.loads(meta_path.read_text())
        except Exception:
            return {}
