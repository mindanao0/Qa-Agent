from __future__ import annotations

import ast
import json
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from src.llm.schemas import TestPlan

PROTECTED_NAMES = frozenset({
    "playwright", "browser", "page", "context", "expect",
    "async_playwright", "True", "False", "None",
    "asyncio", "pytest", "request", "response"
})


class BFTSecurityHalt(Exception):
    """Raised when generator output attempts to remap protected names."""


class _DocstringStripper(ast.NodeTransformer):
    """Phase 1: Remove bare string literals (docstrings, comments)."""
    def visit_Expr(self, node: ast.Expr) -> ast.AST | None:
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return node


class _NameCanonicalizer(ast.NodeTransformer):
    """Phase 2: Rename non-protected names to var_1, var_2, ..."""
    def __init__(self) -> None:
        self._mapping: dict[str, str] = {}
        self._counter = 0

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if node.id in PROTECTED_NAMES:
            return node
        if node.id not in self._mapping:
            self._counter += 1
            self._mapping[node.id] = f"var_{self._counter}"
        return ast.Name(id=self._mapping[node.id], ctx=node.ctx)

    def visit_Assign(self, node: ast.Assign) -> ast.Assign:
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in PROTECTED_NAMES:
                raise BFTSecurityHalt(
                    f"Attempt to remap protected name: {target.id}"
                )
        return self.generic_visit(node)


class ASTNormalizer:
    def normalize(self, source_code: str) -> str | None:
        """Return canonical string or None on SyntaxError / BFTSecurityHalt."""
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return None
        try:
            tree = _DocstringStripper().visit(tree)
            tree = _NameCanonicalizer().visit(tree)
        except BFTSecurityHalt:
            return None
        for node in ast.walk(tree):
            for attr in ("lineno", "col_offset", "end_lineno", "end_col_offset"):
                try:
                    delattr(node, attr)
                except AttributeError:
                    pass
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)


class BFTVoter:
    def __init__(self) -> None:
        self._normalizer = ASTNormalizer()

    def vote(self, candidates: list[str | None]) -> str | None:
        """Return raw (pre-normalized) winning code if quorum >= 2, else None."""
        valid: list[tuple[str, str]] = []
        for raw in candidates:
            if raw is None:
                continue
            norm = self._normalizer.normalize(raw)
            if norm is not None:
                valid.append((raw, norm))
        if not valid:
            return None
        groups: dict[str, list[str]] = {}
        for raw, norm in valid:
            groups.setdefault(norm, []).append(raw)
        best_norm = max(groups, key=lambda k: len(groups[k]))
        if len(groups[best_norm]) >= 2:
            return groups[best_norm][0]
        return None


class PlanVoter:
    """BFT consensus on structured TestPlan JSON.

    Vote on a canonical JSON serialization of plan.model_dump() instead of
    Python source code — structured plans are far more likely to agree across
    LLM temperatures than free-form code.
    """

    def vote(self, candidates: list["TestPlan | None"]) -> "TestPlan | None":
        valid: list[tuple["TestPlan", str]] = []
        for plan in candidates:
            if plan is None:
                continue
            valid.append((plan, self._canonical(plan)))
        if not valid:
            return None
        groups: dict[str, list["TestPlan"]] = {}
        for plan, canon in valid:
            groups.setdefault(canon, []).append(plan)
        best = max(groups, key=lambda k: len(groups[k]))
        if len(groups[best]) >= 2:
            return groups[best][0]
        return None

    def _canonical(self, plan: "TestPlan") -> str:
        return json.dumps(
            plan.model_dump(exclude={"model_config"}),
            sort_keys=True,
            ensure_ascii=False,
        )


class BFTGeneratorConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    node_id: int
    temperature: float
    top_p: float
    model: str = "qwen2.5-coder:7b-instruct-q4_K_M"


BFT_CONFIGS: tuple[BFTGeneratorConfig, ...] = (
    BFTGeneratorConfig(node_id=0, temperature=0.0, top_p=1.0),
    BFTGeneratorConfig(node_id=1, temperature=0.3, top_p=0.85),
    BFTGeneratorConfig(node_id=2, temperature=0.7, top_p=0.85),
)


__all__ = [
    "PROTECTED_NAMES",
    "BFTSecurityHalt",
    "ASTNormalizer",
    "BFTVoter",
    "PlanVoter",
    "BFTGeneratorConfig",
    "BFT_CONFIGS",
]
