"""
ASTParser — Sprint 6.

Extracts public function signatures, docstrings, type hints, and McCabe
complexity from Python source files using the stdlib ast module.
"""
from __future__ import annotations

import ast
import hashlib
import pathlib

from pydantic import BaseModel, ConfigDict


class ArgSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    annotation: str | None
    default: str | None


class FunctionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    func_id: str           # sha256[:10] of module_path+func_name
    module_path: str       # path as string
    func_name: str
    class_name: str | None = None  # set when func is a class method, else None
    is_async: bool = False         # True for async def functions
    args: list[ArgSpec]
    return_type: str | None
    docstring: str | None
    decorators: list[str]
    complexity: int        # McCabe cyclomatic complexity


def _annotation_to_str(node: ast.expr | None) -> str | None:
    if node is None:
        return None
    return ast.unparse(node)


def _default_to_str(node: ast.expr | None) -> str | None:
    if node is None:
        return None
    return ast.unparse(node)


def _decorator_names(decorator_list: list[ast.expr]) -> list[str]:
    return [ast.unparse(d) for d in decorator_list]


def _is_property(decorator_list: list[ast.expr]) -> bool:
    for d in decorator_list:
        if isinstance(d, ast.Name) and d.id == "property":
            return True
        if isinstance(d, ast.Attribute) and d.attr == "property":
            return True
    return False


def _mccabe(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """Count (If + For + While + With + AsyncWith + ExceptHandler) + 1, not recursing into nested funcs."""
    count = 1
    # BFS/DFS but skip inner function bodies
    to_visit = list(ast.iter_child_nodes(func_node))
    while to_visit:
        node = to_visit.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue  # don't count or recurse into inner functions
        if isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.AsyncWith, ast.ExceptHandler)):
            count += 1
        to_visit.extend(ast.iter_child_nodes(node))
    return count


def _extract_args(args: ast.arguments) -> list[ArgSpec]:
    result: list[ArgSpec] = []

    # posonlyargs + regular args share args.defaults (right-aligned)
    pos_args = args.posonlyargs + args.args
    offset = len(pos_args) - len(args.defaults)
    for i, arg in enumerate(pos_args):
        if arg.arg in ("self", "cls"):
            continue
        default_node = args.defaults[i - offset] if i >= offset else None
        result.append(
            ArgSpec(
                name=arg.arg,
                annotation=_annotation_to_str(arg.annotation),
                default=_default_to_str(default_node),
            )
        )

    # keyword-only args use kw_defaults (parallel list, None means no default)
    for arg, default_node in zip(args.kwonlyargs, args.kw_defaults):
        result.append(
            ArgSpec(
                name=arg.arg,
                annotation=_annotation_to_str(arg.annotation),
                default=_default_to_str(default_node),
            )
        )

    return result


def parse_module(path: pathlib.Path) -> list[FunctionSpec]:
    """
    Parse a Python source file and return FunctionSpec for each public function.

    Skips functions whose names start with '_' unless they have @property decorator.
    Handles both module-level functions and class methods.
    """
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    module_path_str = str(path)
    specs: list[FunctionSpec] = []

    def _process(
        func_node: ast.FunctionDef | ast.AsyncFunctionDef,
        class_name: str | None = None,
    ) -> None:
        name = func_node.name
        decorators = _decorator_names(func_node.decorator_list)
        is_prop = _is_property(func_node.decorator_list)

        if name.startswith("_") and not is_prop:
            return

        raw_id = module_path_str + "::" + name
        func_id = hashlib.sha256(raw_id.encode()).hexdigest()[:10]

        docstring = ast.get_docstring(func_node)
        return_type = _annotation_to_str(func_node.returns)
        args = _extract_args(func_node.args)
        complexity = _mccabe(func_node)

        specs.append(
            FunctionSpec(
                func_id=func_id,
                module_path=module_path_str,
                func_name=name,
                class_name=class_name,
                is_async=isinstance(func_node, ast.AsyncFunctionDef),
                args=args,
                return_type=return_type,
                docstring=docstring,
                decorators=decorators,
                complexity=complexity,
            )
        )

    for func_node, class_name in _top_level_functions(tree):
        _process(func_node, class_name)

    return specs


def _top_level_functions(tree: ast.Module):
    """Yield (func_node, class_name) for module-level functions and direct class methods."""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node, None
        elif isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    yield child, node.name


__all__ = ["ArgSpec", "FunctionSpec", "parse_module"]
