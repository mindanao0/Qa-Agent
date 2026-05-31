"""
3-phase deterministic AST normalization pipeline.

The pipeline converts arbitrary Python source code into a canonical
structural fingerprint that is invariant to formatting, naming, ordering of
duplicate imports, docstrings, and source positions — but preserves the
underlying control-flow and call structure used to vote on equivalence.

Pipeline:
    Phase 1 — Strip non-functional elements (docstrings, duplicate imports).
    Phase 2 — Canonicalize local identifiers (var_0, var_1, …) while
              preserving the PROTECTED_KEYWORDS set untouched. Any attempt to
              remap a protected name triggers KS-2 (BFTSecurityViolation).
    Phase 3 — Remove position metadata and emit canonical ``ast.dump()``.
"""

from __future__ import annotations

import ast
from typing import Any

from .models import BFTSecurityViolation


SYNTAX_ERROR_FINGERPRINT: str = "SYNTAX_ERROR"


class ASTNormalizer(ast.NodeTransformer):
    """Deterministic 3-phase AST normalizer.

    Use :meth:`normalize` as the single public entry point. A fresh instance
    should be created per call because per-run state (the identifier mapping
    register) is held on ``self``.
    """

    # KILL SWITCH KS-2: these identifiers must NEVER be renamed by phase 2.
    # If we observe a phase-2 attempt to rewrite one of these, abort the run.
    PROTECTED_KEYWORDS: frozenset[str] = frozenset(
        {
            "playwright",
            "browser",
            "page",
            "context",
            "expect",
            "asyncio",
            "async_playwright",
        }
    )

    def __init__(self) -> None:
        super().__init__()
        # Phase 2 state — populated as identifiers are encountered.
        self._mapping_register: dict[str, str] = {}
        self._next_index: int = 0

    # ─────────────────────────────────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────────────────────────────────

    def normalize(self, source_code: str) -> str:
        """Run the full 3-phase pipeline and return the canonical fingerprint.

        Returns :data:`SYNTAX_ERROR_FINGERPRINT` if the input does not parse.
        Raises :class:`BFTSecurityViolation` if KS-2 fires.
        """
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return SYNTAX_ERROR_FINGERPRINT

        # Phase 1 — structural strip
        tree = self._phase1_strip(tree)

        # Phase 2 — canonicalize identifiers (uses self as NodeTransformer)
        self._mapping_register.clear()
        self._next_index = 0
        tree = self.visit(tree)
        ast.fix_missing_locations(tree)

        # Phase 3 — strip position metadata and serialize
        return self._phase3_serialize(tree)

    # ─────────────────────────────────────────────────────────────────────
    # Phase 1 — strip non-functional elements
    # ─────────────────────────────────────────────────────────────────────

    def _phase1_strip(self, tree: ast.AST) -> ast.AST:
        """Remove docstrings, standalone string-constant statements, and
        duplicate imports while leaving semantic structure intact."""

        seen_imports: set[str] = set()

        def _import_signature(node: ast.AST) -> str | None:
            if isinstance(node, ast.Import):
                return "import:" + ",".join(
                    f"{alias.name}|{alias.asname or ''}" for alias in node.names
                )
            if isinstance(node, ast.ImportFrom):
                names = ",".join(
                    f"{alias.name}|{alias.asname or ''}" for alias in node.names
                )
                return f"from:{node.module or ''}|{node.level}|{names}"
            return None

        def _filter_body(body: list[ast.stmt]) -> list[ast.stmt]:
            new_body: list[ast.stmt] = []
            for stmt in body:
                # Drop standalone string-constant expression statements
                # (covers module/function/class docstrings and stray strings).
                if (
                    isinstance(stmt, ast.Expr)
                    and isinstance(stmt.value, ast.Constant)
                    and isinstance(stmt.value.value, str)
                ):
                    continue
                # Drop *duplicate* imports only — preserve semantic imports.
                sig = _import_signature(stmt)
                if sig is not None:
                    if sig in seen_imports:
                        continue
                    seen_imports.add(sig)
                new_body.append(stmt)
            return new_body

        for node in ast.walk(tree):
            for attr in ("body", "orelse", "finalbody"):
                body = getattr(node, attr, None)
                if isinstance(body, list) and body and isinstance(body[0], ast.stmt):
                    setattr(node, attr, _filter_body(body))

        return tree

    # ─────────────────────────────────────────────────────────────────────
    # Phase 2 — canonical identifiers (NodeTransformer visitors)
    # ─────────────────────────────────────────────────────────────────────

    def _canonical(self, original: str) -> str:
        """Return the canonical ``var_N`` form for ``original``.

        Protected keywords are returned unchanged. The mapping is stable
        across one ``normalize`` call so each unique identifier gets a
        single, sequence-determined alias.
        """
        if original in self.PROTECTED_KEYWORDS:
            return original
        existing = self._mapping_register.get(original)
        if existing is not None:
            return existing
        alias = f"var_{self._next_index}"
        self._next_index += 1
        self._mapping_register[original] = alias
        return alias

    def visit_Name(self, node: ast.Name) -> ast.AST:
        # Even though we don't rewrite protected names, we still need to
        # detect any attempt to map them. The mere act of the LLM producing
        # an assignment that *replaces* a protected identifier (e.g. binding
        # a new value to "page") is allowed — what's forbidden is structural
        # remapping where a protected name was originally used but our
        # canonicalization rewrites it to var_N. Compare before vs. after.
        new_id = self._canonical(node.id)
        if node.id in self.PROTECTED_KEYWORDS and new_id != node.id:
            raise BFTSecurityViolation(
                f"ABORT: protected API remapping detected (Name {node.id!r} -> {new_id!r})"
            )
        return ast.copy_location(ast.Name(id=new_id, ctx=node.ctx), node)

    def visit_arg(self, node: ast.arg) -> ast.AST:
        new_arg = self._canonical(node.arg)
        if node.arg in self.PROTECTED_KEYWORDS and new_arg != node.arg:
            raise BFTSecurityViolation(
                f"ABORT: protected API remapping detected (arg {node.arg!r} -> {new_arg!r})"
            )
        return ast.copy_location(
            ast.arg(
                arg=new_arg,
                annotation=self.visit(node.annotation) if node.annotation else None,
                type_comment=node.type_comment,
            ),
            node,
        )

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        # Recurse into the value side first so nested Names get canonicalized.
        new_value = self.visit(node.value)
        # Attribute names (the right-hand side of `.attr`) are API surface,
        # not local identifiers — never rename them. But if the LLM produced
        # an Attribute whose attr is a protected keyword and somehow it got
        # rewritten (defensive check), trip the kill switch.
        attr_name: str = node.attr
        if attr_name in self.PROTECTED_KEYWORDS:
            # Allowed — but ensure no normalization step mutated it.
            if attr_name != node.attr:
                raise BFTSecurityViolation(
                    f"ABORT: protected API remapping detected (Attribute {node.attr!r})"
                )
        return ast.copy_location(
            ast.Attribute(value=new_value, attr=attr_name, ctx=node.ctx),
            node,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Phase 3 — strip position metadata + canonical dump
    # ─────────────────────────────────────────────────────────────────────

    def _phase3_serialize(self, tree: ast.AST) -> str:
        """Remove ``lineno``/``col_offset``/``end_*`` fields from every node
        and return ``ast.dump`` with attribute inclusion disabled."""

        for node in ast.walk(tree):
            for field in ("lineno", "col_offset", "end_lineno", "end_col_offset"):
                if hasattr(node, field):
                    try:
                        setattr(node, field, 0)
                    except AttributeError:
                        # Built-in nodes occasionally refuse the assignment;
                        # ast.dump(..., include_attributes=False) will skip
                        # these anyway, so we just continue.
                        pass

        # ``include_attributes=False`` is the deterministic switch — it drops
        # all attribute fields (positions, ctx variants where applicable)
        # from the dump, leaving only the structural tree.
        return ast.dump(tree, annotate_fields=True, include_attributes=False)


def normalize_source(source_code: str) -> str:
    """Convenience helper: one-shot normalization with a fresh normalizer."""
    return ASTNormalizer().normalize(source_code)


__all__: list[str] = [
    "ASTNormalizer",
    "SYNTAX_ERROR_FINGERPRINT",
    "normalize_source",
]


# Re-export local symbols quietly for type-checkers that walk module dicts.
_: dict[str, Any] = {}
