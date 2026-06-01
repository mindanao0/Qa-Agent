"""JSASTParser — Sprint 9. Calls Node.js ast_walker.js via subprocess."""
from __future__ import annotations

import asyncio
import hashlib
import json
import pathlib
import subprocess

from loguru import logger
from pydantic import BaseModel, ConfigDict

_WALKER_SCRIPT = pathlib.Path(__file__).parent.parent.parent / "scripts" / "ast_walker.js"
_PARSE_SEMAPHORE = asyncio.Semaphore(4)


class JSFunctionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    func_id: str
    module_path: str
    func_name: str
    params: list[str]
    return_type: str | None
    is_async: bool
    is_exported: bool
    jsdoc: str | None
    complexity: int


class JSASTParser:
    """No required constructor args."""

    async def parse_file(self, path: pathlib.Path) -> list[JSFunctionSpec]:
        async with _PARSE_SEMAPHORE:
            return await asyncio.to_thread(self._run_walker, path)

    def _run_walker(self, path: pathlib.Path) -> list[JSFunctionSpec]:
        try:
            proc = subprocess.run(
                ["node", str(_WALKER_SCRIPT), str(path)],
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            logger.warning(f"JSASTParser: walker subprocess error for {path}: {exc!r}")
            return []

        if proc.returncode != 0:
            logger.warning(f"JSASTParser: walker failed for {path}: {proc.stderr[:200]}")
            return []

        try:
            raw = json.loads(proc.stdout)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(f"JSASTParser: invalid JSON for {path}: {exc}")
            return []

        if not isinstance(raw, list):
            logger.warning(f"JSASTParser: expected list, got {type(raw).__name__} for {path}")
            return []

        specs: list[JSFunctionSpec] = []
        for item in raw:
            try:
                func_id = hashlib.sha256(
                    (str(path) + "::" + item["func_name"]).encode()
                ).hexdigest()[:10]
                specs.append(JSFunctionSpec(
                    func_id=func_id,
                    module_path=str(path),
                    func_name=item["func_name"],
                    params=item.get("params") or [],
                    return_type=item.get("return_type"),
                    is_async=bool(item.get("is_async", False)),
                    is_exported=bool(item.get("is_exported", True)),
                    jsdoc=item.get("jsdoc"),
                    complexity=int(item.get("complexity", 1)),
                ))
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning(f"JSASTParser: skipping bad item: {exc}")
                continue
        return specs

    async def parse_dir(
        self,
        root: pathlib.Path,
        glob: str = "**/*.ts",
        exclude: tuple[str, ...] = ("node_modules", "dist", ".next"),
    ) -> list[JSFunctionSpec]:
        files = [
            p for p in root.glob(glob)
            if not any(ex in p.parts for ex in exclude)
        ]
        tasks = [self.parse_file(f) for f in files]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        specs: list[JSFunctionSpec] = []
        for r in results:
            if isinstance(r, Exception):
                logger.warning(f"JSASTParser.parse_dir: {r!r}")
            else:
                specs.extend(r)
        return specs


__all__ = ["JSASTParser", "JSFunctionSpec"]
