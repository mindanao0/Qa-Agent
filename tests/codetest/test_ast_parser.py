import pathlib
import textwrap
import pytest
from src.codetest.ast_parser import parse_module, FunctionSpec, ArgSpec


@pytest.fixture
def tmp_module(tmp_path):
    src = textwrap.dedent("""\
        def add(x: int, y: int = 0) -> int:
            \"\"\"Add two numbers.\"\"\"
            return x + y

        def _private(x):
            return x

        class MyClass:
            @property
            def _prop(self) -> str:
                return "ok"

            def public_method(self, val: str) -> None:
                if val:
                    for _ in range(3):
                        pass

        def branchy(x, y, z):
            if x:
                pass
            elif y:
                pass
            while z:
                break
            return x
    """)
    p = tmp_path / "sample.py"
    p.write_text(src, encoding="utf-8")
    return p


def test_parse_returns_function_specs(tmp_module):
    specs = parse_module(tmp_module)
    assert len(specs) >= 1
    assert all(isinstance(s, FunctionSpec) for s in specs)


def test_skips_private_functions(tmp_module):
    specs = parse_module(tmp_module)
    names = [s.func_name for s in specs]
    assert "_private" not in names


def test_includes_property(tmp_module):
    specs = parse_module(tmp_module)
    names = [s.func_name for s in specs]
    assert "_prop" in names


def test_func_id_is_sha256_prefix(tmp_module):
    specs = parse_module(tmp_module)
    add_spec = next(s for s in specs if s.func_name == "add")
    assert len(add_spec.func_id) == 10
    assert all(c in "0123456789abcdef" for c in add_spec.func_id)


def test_args_and_return_type(tmp_module):
    specs = parse_module(tmp_module)
    add_spec = next(s for s in specs if s.func_name == "add")
    assert add_spec.return_type == "int"
    assert len(add_spec.args) == 2
    assert add_spec.args[0].name == "x"
    assert add_spec.args[0].annotation == "int"
    assert add_spec.args[1].default == "0"


def test_docstring_captured(tmp_module):
    specs = parse_module(tmp_module)
    add_spec = next(s for s in specs if s.func_name == "add")
    assert add_spec.docstring is not None
    assert "Add two numbers" in add_spec.docstring


def test_complexity_base(tmp_module):
    specs = parse_module(tmp_module)
    add_spec = next(s for s in specs if s.func_name == "add")
    assert add_spec.complexity == 1  # no branches


def test_complexity_branches(tmp_module):
    specs = parse_module(tmp_module)
    branchy_spec = next(s for s in specs if s.func_name == "branchy")
    # if + elif(=If node) + while = 3 branches + 1 = 4
    assert branchy_spec.complexity == 4


def test_module_path_is_relative_string(tmp_module):
    specs = parse_module(tmp_module)
    assert all(isinstance(s.module_path, str) for s in specs)


def test_public_method_included(tmp_module):
    specs = parse_module(tmp_module)
    names = [s.func_name for s in specs]
    assert "public_method" in names


def test_public_method_complexity(tmp_module):
    specs = parse_module(tmp_module)
    method = next(s for s in specs if s.func_name == "public_method")
    # if + for = 2 branches + 1 = 3
    assert method.complexity == 3


def test_kwonly_args_default(tmp_path):
    src = textwrap.dedent("""\
        def f(a: int, *, b: str = "hello") -> None:
            pass
    """)
    p = tmp_path / "kwonly.py"
    p.write_text(src, encoding="utf-8")
    specs = parse_module(p)
    assert len(specs) == 1
    spec = specs[0]
    b_arg = next(a for a in spec.args if a.name == "b")
    assert b_arg.default == "'hello'"


def test_nested_functions_excluded(tmp_path):
    src = textwrap.dedent("""\
        def outer():
            def inner():
                pass
            return inner
    """)
    p = tmp_path / "nested.py"
    p.write_text(src, encoding="utf-8")
    specs = parse_module(p)
    names = [s.func_name for s in specs]
    assert "inner" not in names
    assert "outer" in names
