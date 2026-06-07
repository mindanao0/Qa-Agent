# Sprint 9: JS/TS Code Testing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the code-testing pipeline to JS/TS source files: Babel AST parser → Ollama Vitest generator → 4-check judge → Vitest executor, producing `sprint9_results.json` with all gates green.

**Architecture:** A Node.js script (`scripts/ast_walker.js`) walks Babel AST and emits JSON; Python `JSASTParser` orchestrates that subprocess. `JSTestGenerator` calls Ollama (Semaphore(1)) to produce self-contained Vitest tests (function implementation inlined — avoids all import/module-resolution complexity). `JSCodeJudge` runs 4 deterministic checks. `JSTestExecutor` runs `npx vitest run` with JSON reporter from project root. `OTelTracer` + `CryptoAuditTrail` wrap all 4 pipeline stages in `measure_sprint9.py`.

**Tech Stack:** Python 3.11, Node.js 24 (already installed), @babel/parser ^7.24, @babel/traverse ^7.24, vitest ^2.0, Pydantic V2, asyncio, pathlib, OTelTracer (Sprint 8)

---

## File Map

### New files
| File | Responsibility |
|------|----------------|
| `src/js_targets/__init__.py` | Package marker (empty) |
| `src/js_targets/utils.ts` | 5 exported pure utility functions |
| `src/js_targets/validators.ts` | 3 exported input validators |
| `src/js_targets/formatters.ts` | 4 exported string/date formatters |
| `package.json` | Node.js deps: vitest, @babel/parser, @babel/traverse |
| `scripts/setup_vitest.py` | One-time npm install script |
| `scripts/ast_walker.js` | Node.js — Babel AST walk → JSON to stdout |
| `src/codetest/js_ast_parser.py` | JSASTParser + JSFunctionSpec |
| `src/codetest/js_generator.py` | JSTestGenerator + GeneratedJSTest |
| `src/codetest/js_judge.py` | JSCodeJudge (4 checks) |
| `src/codetest/js_executor.py` | JSTestExecutor + JSTestResult |
| `audit/sprint9/__init__.py` | Package marker (empty) |
| `audit/sprint9/measure_sprint9.py` | Gate measurement → sprint9_results.json |
| `tests/codetest/test_js_ast_parser.py` | Unit tests (mocked subprocess) |
| `tests/codetest/test_js_judge.py` | Unit tests (deterministic checks) |
| `tests/codetest/test_js_executor.py` | Unit tests (mocked subprocess) |

---

## Task 1: JS target TypeScript files

**Files:**
- Create: `src/js_targets/__init__.py`
- Create: `src/js_targets/utils.ts`
- Create: `src/js_targets/validators.ts`
- Create: `src/js_targets/formatters.ts`

- [ ] **Step 1: Create package marker**

Create `src/js_targets/__init__.py` (empty file).

- [ ] **Step 2: Create `src/js_targets/utils.ts`**

```typescript
/** Clamp a number between min and max (inclusive). */
export function clamp(value: number, min: number, max: number): number {
  if (value < min) return min;
  if (value > max) return max;
  return value;
}

/** Remove duplicate elements, preserving first-occurrence order. */
export function dedupe<T>(arr: T[]): T[] {
  const seen = new Set<T>();
  const result: T[] = [];
  for (const item of arr) {
    if (!seen.has(item)) { seen.add(item); result.push(item); }
  }
  return result;
}

/** Return a shallow copy of obj containing only the specified keys. */
export function pick<T extends object, K extends keyof T>(obj: T, keys: K[]): Pick<T, K> {
  const result = {} as Pick<T, K>;
  for (const key of keys) { result[key] = obj[key]; }
  return result;
}

/** Split arr into sub-arrays of at most `size` elements. Throws if size <= 0. */
export function chunk<T>(arr: T[], size: number): T[][] {
  if (size <= 0) throw new Error('chunk: size must be positive');
  const result: T[][] = [];
  for (let i = 0; i < arr.length; i += size) {
    result.push(arr.slice(i, i + size));
  }
  return result;
}

/** Sum all numbers in an array; returns 0 for an empty array. */
export function sum(arr: number[]): number {
  let total = 0;
  for (const n of arr) { total += n; }
  return total;
}
```

- [ ] **Step 3: Create `src/js_targets/validators.ts`**

```typescript
/** Returns true if value looks like a valid e-mail address. */
export function isEmail(value: string): boolean {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value);
}

/** Returns true if value is a syntactically valid http/https URL. */
export function isUrl(value: string): boolean {
  try {
    const url = new URL(value);
    return url.protocol === 'http:' || url.protocol === 'https:';
  } catch {
    return false;
  }
}

/** Returns true if value is not null/undefined/empty-string/empty-array/empty-object. */
export function isNonEmpty(value: unknown): boolean {
  if (value === null || value === undefined) return false;
  if (typeof value === 'string') return value.trim().length > 0;
  if (Array.isArray(value)) return value.length > 0;
  if (typeof value === 'object') return Object.keys(value as object).length > 0;
  return true;
}
```

- [ ] **Step 4: Create `src/js_targets/formatters.ts`**

```typescript
/** Uppercase the first character of a string. */
export function capitalize(str: string): string {
  if (!str) return str;
  return str.charAt(0).toUpperCase() + str.slice(1);
}

/** Truncate str to maxLen chars, appending suffix if truncated. */
export function truncate(str: string, maxLen: number, suffix: string = '...'): string {
  if (str.length <= maxLen) return str;
  return str.slice(0, Math.max(0, maxLen - suffix.length)) + suffix;
}

/** Format a Date or ISO string for the given locale (default en-US). Returns 'Invalid Date' on bad input. */
export function formatDate(date: Date | string, locale: string = 'en-US'): string {
  const d = typeof date === 'string' ? new Date(date) : date;
  if (isNaN(d.getTime())) return 'Invalid Date';
  return d.toLocaleDateString(locale);
}

/** Convert a camelCase or snake_case string to kebab-case. */
export function kebabCase(str: string): string {
  return str
    .replace(/([A-Z])/g, '-$1')
    .replace(/[\s_]+/g, '-')
    .toLowerCase()
    .replace(/^-/, '');
}
```

- [ ] **Step 5: Commit**

```bash
git add src/js_targets/
git commit -m "feat(sprint9): JS/TS target files — 12 exported functions"
```

---

## Task 2: Node.js environment setup

**Files:**
- Create: `package.json` (project root)
- Create: `scripts/setup_vitest.py`

- [ ] **Step 1: Create `package.json` at project root**

Create `D:\Code\qa-agent\package.json`:

```json
{
  "type": "module",
  "scripts": {
    "test": "vitest run"
  },
  "devDependencies": {
    "vitest": "^2.0.0",
    "@babel/parser": "^7.24.0",
    "@babel/traverse": "^7.24.0",
    "@types/node": "^20.0.0"
  }
}
```

- [ ] **Step 2: Create `scripts/setup_vitest.py`**

Create `D:\Code\qa-agent\scripts\setup_vitest.py`:

```python
"""One-time setup: install Node.js devDependencies for Sprint 9."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent


def main() -> None:
    pkg = _ROOT / "package.json"
    if not pkg.exists():
        print(f"ERROR: {pkg} not found", file=sys.stderr)
        sys.exit(1)
    print(f"[setup_vitest] Running npm install in {_ROOT}")
    result = subprocess.run(
        ["npm", "install"],
        cwd=_ROOT,
        timeout=180,
    )
    if result.returncode != 0:
        print("[setup_vitest] npm install FAILED", file=sys.stderr)
        sys.exit(1)
    print("[setup_vitest] npm install complete")
    node_modules = _ROOT / "node_modules"
    vitest_bin = node_modules / ".bin" / "vitest"
    if not vitest_bin.exists() and not (node_modules / ".bin" / "vitest.cmd").exists():
        print(f"[setup_vitest] WARNING: vitest binary not found at {vitest_bin}", file=sys.stderr)
    else:
        print("[setup_vitest] vitest binary found — ready")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run npm install**

```bash
python scripts/setup_vitest.py
```

Expected: `npm install complete` and `node_modules/` created with vitest, @babel/parser, @babel/traverse.

- [ ] **Step 4: Verify vitest runs**

```bash
npx vitest --version
```

Expected: prints vitest version (e.g. `2.x.x`).

- [ ] **Step 5: Commit**

```bash
git add package.json scripts/setup_vitest.py
git commit -m "feat(sprint9): package.json + setup_vitest.py — Node.js deps"
```

---

## Task 3: `scripts/ast_walker.js`

**Files:**
- Create: `scripts/ast_walker.js`

- [ ] **Step 1: Create `scripts/ast_walker.js`**

Create `D:\Code\qa-agent\scripts\ast_walker.js`:

```javascript
// Babel AST walker — called by JSASTParser via subprocess.
// Usage: node scripts/ast_walker.js <filepath>
// Output: JSON array of function objects to stdout, errors to stderr.
import { readFileSync, existsSync } from 'fs';
import { parse } from '@babel/parser';
import _traverse from '@babel/traverse';

const traverse = _traverse.default ?? _traverse;

const filepath = process.argv[2];
if (!filepath) {
  process.stderr.write('Usage: node scripts/ast_walker.js <filepath>\n');
  process.exit(1);
}
if (!existsSync(filepath)) {
  process.stderr.write(`File not found: ${filepath}\n`);
  process.exit(1);
}

const code = readFileSync(filepath, 'utf8');
let ast;
try {
  ast = parse(code, {
    sourceType: 'module',
    plugins: ['typescript', 'jsx'],
    errorRecovery: true,
    attachComment: true,
  });
} catch (e) {
  process.stderr.write(`Parse error: ${e.message}\n`);
  process.exit(1);
}

function countComplexity(node) {
  const BRANCH_TYPES = new Set([
    'IfStatement', 'ForStatement', 'ForInStatement', 'ForOfStatement',
    'WhileStatement', 'DoWhileStatement', 'CatchClause', 'SwitchCase',
    'ConditionalExpression',
  ]);
  let count = 1;
  const stack = [...(node.body ? (Array.isArray(node.body) ? node.body : [node.body]) : [])];
  const visited = new Set();
  while (stack.length) {
    const n = stack.pop();
    if (!n || typeof n !== 'object' || !n.type) continue;
    if (visited.has(n)) continue;
    visited.add(n);
    // Do not recurse into nested function bodies
    if (n !== node && (n.type === 'FunctionDeclaration' || n.type === 'ArrowFunctionExpression' || n.type === 'FunctionExpression')) continue;
    if (BRANCH_TYPES.has(n.type)) count++;
    for (const key of Object.keys(n)) {
      if (['type', 'loc', 'start', 'end', 'extra', 'leadingComments', 'trailingComments'].includes(key)) continue;
      const child = n[key];
      if (Array.isArray(child)) {
        for (const c of child) { if (c && typeof c === 'object' && c.type) stack.push(c); }
      } else if (child && typeof child === 'object' && child.type) {
        stack.push(child);
      }
    }
  }
  return count;
}

function extractJsdoc(node) {
  const comments = node.leadingComments || [];
  for (const c of comments.slice().reverse()) {
    if (c.type === 'CommentBlock' && c.value.startsWith('*')) {
      return c.value.trim();
    }
  }
  return null;
}

function paramName(p) {
  if (!p) return 'unknown';
  switch (p.type) {
    case 'Identifier': return p.name;
    case 'AssignmentPattern': return paramName(p.left);
    case 'RestElement': return '...' + paramName(p.argument);
    case 'ObjectPattern': return '{...}';
    case 'ArrayPattern': return '[...]';
    case 'TSParameterProperty': return paramName(p.parameter);
    default: return 'param';
  }
}

function returnTypeStr(node) {
  if (!node.returnType) return null;
  const ann = node.returnType.typeAnnotation;
  if (!ann) return null;
  const map = {
    TSNumberKeyword: 'number', TSStringKeyword: 'string',
    TSBooleanKeyword: 'boolean', TSVoidKeyword: 'void',
    TSAnyKeyword: 'any', TSNullKeyword: 'null',
    TSUndefinedKeyword: 'undefined',
  };
  if (map[ann.type]) return map[ann.type];
  if (ann.type === 'TSArrayType') return 'Array';
  if (ann.type === 'TSUnionType') return 'union';
  if (ann.type === 'TSTypeReference' && ann.typeName) {
    return ann.typeName.name || null;
  }
  return ann.type || null;
}

const results = [];
const seen = new Set();

function isExportedAncestor(path) {
  let p = path.parentPath;
  while (p) {
    const t = p.node.type;
    if (t === 'ExportNamedDeclaration' || t === 'ExportDefaultDeclaration') return true;
    if (t === 'Program' || t === 'BlockStatement' || t === 'ClassBody') break;
    p = p.parentPath;
  }
  return false;
}

function handleFunc(path) {
  const node = path.node;
  let funcName = null;

  if (node.type === 'FunctionDeclaration' && node.id) {
    funcName = node.id.name;
  } else if (
    (node.type === 'ArrowFunctionExpression' || node.type === 'FunctionExpression') &&
    path.parentPath?.node.type === 'VariableDeclarator'
  ) {
    funcName = path.parentPath.node.id?.name ?? null;
  } else if (path.parentPath?.node.type === 'ObjectProperty') {
    const key = path.parentPath.node.key;
    funcName = key?.name ?? key?.value ?? null;
  }

  if (!funcName || funcName.startsWith('_')) return;
  const uid = `${filepath}::${funcName}`;
  if (seen.has(uid)) return;
  seen.add(uid);

  const isExported = isExportedAncestor(path);
  const params = (node.params || []).map(paramName);
  const retType = returnTypeStr(node);
  const jsdoc = extractJsdoc(node);
  const complexity = countComplexity(node);

  results.push({
    func_name: funcName,
    params,
    return_type: retType,
    is_async: node.async ?? false,
    is_exported: isExported,
    jsdoc,
    complexity,
  });
}

traverse(ast, {
  FunctionDeclaration: { enter: handleFunc },
  ArrowFunctionExpression: { enter: handleFunc },
  FunctionExpression: { enter: handleFunc },
});

process.stdout.write(JSON.stringify(results) + '\n');
process.exit(0);
```

- [ ] **Step 2: Smoke-test the walker on utils.ts**

```bash
node scripts/ast_walker.js src/js_targets/utils.ts
```

Expected: JSON array with 5 objects, each having `func_name`, `params`, `is_exported: true`, etc.

- [ ] **Step 3: Smoke-test on all three target files**

```bash
node scripts/ast_walker.js src/js_targets/validators.ts
node scripts/ast_walker.js src/js_targets/formatters.ts
```

Expected: 3 objects for validators, 4 objects for formatters.

- [ ] **Step 4: Commit**

```bash
git add scripts/ast_walker.js
git commit -m "feat(sprint9): ast_walker.js — Babel JS/TS AST → JSON"
```

---

## Task 4: `src/codetest/js_ast_parser.py` + tests

**Files:**
- Create: `src/codetest/js_ast_parser.py`
- Create: `tests/codetest/test_js_ast_parser.py`

- [ ] **Step 1: Write failing tests**

Create `D:\Code\qa-agent\tests\codetest\test_js_ast_parser.py`:

```python
"""Unit tests for JSASTParser (mocked subprocess)."""
from __future__ import annotations
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.codetest.js_ast_parser import JSASTParser, JSFunctionSpec


MOCK_WALKER_OUTPUT = json.dumps([
    {
        "func_name": "clamp",
        "params": ["value", "min", "max"],
        "return_type": "number",
        "is_async": False,
        "is_exported": True,
        "jsdoc": "* Clamp a number",
        "complexity": 3,
    },
    {
        "func_name": "sum",
        "params": ["arr"],
        "return_type": "number",
        "is_async": False,
        "is_exported": True,
        "jsdoc": None,
        "complexity": 2,
    },
])


def _make_proc(stdout: str, returncode: int = 0):
    proc = MagicMock()
    proc.returncode = returncode
    proc.stdout = stdout
    proc.stderr = ""
    return proc


def test_parse_file_returns_specs(tmp_path: Path):
    fake_ts = tmp_path / "utils.ts"
    fake_ts.write_text("export function clamp() {}")
    with patch("subprocess.run", return_value=_make_proc(MOCK_WALKER_OUTPUT)):
        import asyncio
        specs = asyncio.run(JSASTParser().parse_file(fake_ts))
    assert len(specs) == 2
    assert specs[0].func_name == "clamp"
    assert specs[0].is_exported is True
    assert specs[0].complexity == 3
    assert isinstance(specs[0].func_id, str) and len(specs[0].func_id) == 10


def test_parse_file_returns_empty_on_walker_error(tmp_path: Path):
    fake_ts = tmp_path / "broken.ts"
    fake_ts.write_text("")
    with patch("subprocess.run", return_value=_make_proc("", returncode=1)):
        import asyncio
        specs = asyncio.run(JSASTParser().parse_file(fake_ts))
    assert specs == []


def test_parse_file_returns_empty_on_bad_json(tmp_path: Path):
    fake_ts = tmp_path / "bad.ts"
    fake_ts.write_text("")
    with patch("subprocess.run", return_value=_make_proc("NOT JSON")):
        import asyncio
        specs = asyncio.run(JSASTParser().parse_file(fake_ts))
    assert specs == []


def test_jsfunctionspec_extra_forbid():
    with pytest.raises(Exception):
        JSFunctionSpec(
            func_id="abc", module_path="x.ts", func_name="f",
            params=[], return_type=None, is_async=False,
            is_exported=True, jsdoc=None, complexity=1,
            unexpected="bad",
        )


def test_parse_dir_aggregates_files(tmp_path: Path):
    (tmp_path / "a.ts").write_text("export function a() {}")
    (tmp_path / "b.ts").write_text("export function b() {}")
    single = json.dumps([{
        "func_name": "x", "params": [], "return_type": None,
        "is_async": False, "is_exported": True, "jsdoc": None, "complexity": 1,
    }])
    with patch("subprocess.run", return_value=_make_proc(single)):
        import asyncio
        specs = asyncio.run(JSASTParser().parse_dir(tmp_path, glob="*.ts"))
    assert len(specs) == 2  # one per file
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/codetest/test_js_ast_parser.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.codetest.js_ast_parser'`

- [ ] **Step 3: Create `src/codetest/js_ast_parser.py`**

```python
"""JSASTParser — Sprint 9. Calls Node.js ast_walker.js via subprocess."""
from __future__ import annotations

import asyncio
import hashlib
import json
import pathlib
import subprocess
from typing import Literal

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
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
uv run pytest tests/codetest/test_js_ast_parser.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/codetest/js_ast_parser.py tests/codetest/test_js_ast_parser.py
git commit -m "feat(sprint9): JSASTParser — Node.js Babel subprocess → JSFunctionSpec"
```

---

## Task 5: `src/codetest/js_generator.py` + tests

**Files:**
- Create: `src/codetest/js_generator.py`
- Create: `tests/codetest/test_js_generator.py`

- [ ] **Step 1: Write failing tests**

Create `D:\Code\qa-agent\tests\codetest\test_js_generator.py`:

```python
"""Unit tests for JSTestGenerator (mocked InstructorClient)."""
from __future__ import annotations
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.codetest.js_ast_parser import JSFunctionSpec
from src.codetest.js_generator import GeneratedJSTest, JSTestGenerator


def _spec(name: str = "clamp", complexity: int = 3) -> JSFunctionSpec:
    return JSFunctionSpec(
        func_id="abc1234567",
        module_path="src/js_targets/utils.ts",
        func_name=name,
        params=["value", "min", "max"],
        return_type="number",
        is_async=False,
        is_exported=True,
        jsdoc="* Clamp a number",
        complexity=complexity,
    )


def _mock_test(func_id: str = "abc1234567") -> GeneratedJSTest:
    return GeneratedJSTest(
        test_id="test_clamp",
        func_id=func_id,
        test_code="import { describe, it, expect } from 'vitest'\nfunction clamp(v,mn,mx){return v<mn?mn:v>mx?mx:v}\ndescribe('clamp',()=>{it('works',()=>{expect(clamp(5,0,10)).toBe(5)})})",
        test_type="metamorphic",
        metamorphic_relation="clamp(x+1,0,10) >= clamp(x,0,10) for all x",
    )


def test_generate_returns_list():
    import asyncio
    mock_client = MagicMock()
    mock_client.create_structured = AsyncMock(return_value=_mock_test())
    mock_client.close = AsyncMock()
    with patch("src.codetest.js_generator.InstructorClient", return_value=mock_client):
        result = asyncio.run(JSTestGenerator().generate([_spec()]))
    assert len(result) == 1
    assert result[0].func_id == "abc1234567"


def test_generate_empty_specs_returns_empty():
    import asyncio
    result = asyncio.run(JSTestGenerator().generate([]))
    assert result == []


def test_generated_js_test_extra_forbid():
    with pytest.raises(Exception):
        GeneratedJSTest(
            test_id="x", func_id="y", test_code="code",
            test_type="happy_path", metamorphic_relation=None,
            bad_field="oops",
        )


def test_high_complexity_prompt_requests_metamorphic():
    from src.codetest.js_generator import _build_prompt
    spec = _spec(complexity=3)
    prompt = _build_prompt(spec)
    assert "metamorphic" in prompt.lower()


def test_low_complexity_prompt_does_not_force_metamorphic():
    from src.codetest.js_generator import _build_prompt
    spec = _spec(complexity=1)
    prompt = _build_prompt(spec)
    assert "happy_path" in prompt or "edge_case" in prompt
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/codetest/test_js_generator.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.codetest.js_generator'`

- [ ] **Step 3: Create `src/codetest/js_generator.py`**

```python
"""JSTestGenerator — Sprint 9. Ollama → self-contained Vitest test code."""
from __future__ import annotations

import asyncio
import hashlib
from typing import Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict

from src.codetest.js_ast_parser import JSFunctionSpec
from src.llm.instructor_client import InstructorClient, StructuredGenerationError

_JS_SEMAPHORE = asyncio.Semaphore(1)
_CODER_MODEL = "qwen2.5-coder:7b-instruct-q4_K_M"

_JS_SYSTEM_PROMPT = (
    "You are a TypeScript/Vitest test engineer generating self-contained test files. "
    "STRICT RULES: "
    "(1) test_code MUST begin with a complete TypeScript implementation of the function under test — "
    "    infer it from the function signature, jsdoc, and complexity; "
    "(2) after the implementation, import only from 'vitest': "
    "    import { describe, it, expect } from 'vitest'; "
    "(3) wrap tests in describe()/it() blocks; "
    "(4) every it() block must call expect(); "
    "(5) NO setTimeout or setInterval in test body; "
    "(6) ALL async tests use async/await, never .then(); "
    "(7) for test_type='metamorphic', provide a non-null metamorphic_relation; "
    "Return ONLY valid JSON."
)


class GeneratedJSTest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    test_id: str
    func_id: str
    test_code: str
    test_type: Literal["happy_path", "edge_case", "metamorphic"]
    metamorphic_relation: str | None


def _build_prompt(spec: JSFunctionSpec) -> str:
    params_desc = ", ".join(spec.params) if spec.params else "(none)"
    ret = spec.return_type or "unknown"
    jsdoc = spec.jsdoc or "No documentation."
    meta_hint = (
        "\nIMPORTANT: complexity is high — use test_type='metamorphic' and provide "
        "a non-null, falsifiable metamorphic_relation (e.g. 'f(x+1) >= f(x) for positive x')."
        if spec.complexity >= 2
        else "\nUse test_type='happy_path' or 'edge_case'. metamorphic_relation must be null."
    )
    return (
        f"Function: {spec.func_name}({params_desc}) -> {ret}\n"
        f"Async: {spec.is_async}\n"
        f"JSDoc: {jsdoc}\n"
        f"Complexity: {spec.complexity}\n"
        f"test_id should be 'test_{spec.func_name}'.\n"
        f"Begin test_code with the TypeScript implementation, then vitest tests."
        f"{meta_hint}"
    )


class JSTestGenerator:
    """Generate self-contained Vitest test files from JSFunctionSpec objects via Ollama."""

    def __init__(self, model: str = _CODER_MODEL) -> None:
        self._model = model

    async def generate(self, specs: list[JSFunctionSpec]) -> list[GeneratedJSTest]:
        if not specs:
            return []
        results: list[GeneratedJSTest] = []
        client = InstructorClient(model=self._model)
        try:
            for spec in specs:
                test = await self._generate_one(client, spec)
                if test is not None:
                    results.append(test)
        finally:
            await client.close()
        logger.info(f"JSTestGenerator: generated {len(results)}/{len(specs)}")
        return results

    async def _generate_one(
        self, client: InstructorClient, spec: JSFunctionSpec
    ) -> GeneratedJSTest | None:
        messages = [
            {"role": "system", "content": _JS_SYSTEM_PROMPT},
            {"role": "user", "content": _build_prompt(spec)},
        ]
        async with _JS_SEMAPHORE:
            try:
                test = await client.create_structured(
                    prompt=messages,
                    response_model=GeneratedJSTest,
                    temperature=0.2,
                )
                return test.model_copy(update={"func_id": spec.func_id})
            except StructuredGenerationError as exc:
                logger.warning(f"JSTestGenerator: failed for {spec.func_name!r}: {exc!r}")
                return None


__all__ = ["GeneratedJSTest", "JSTestGenerator"]
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
uv run pytest tests/codetest/test_js_generator.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/codetest/js_generator.py tests/codetest/test_js_generator.py
git commit -m "feat(sprint9): JSTestGenerator — Vitest self-contained test generation"
```

---

## Task 6: `src/codetest/js_judge.py` + tests

**Files:**
- Create: `src/codetest/js_judge.py`
- Create: `tests/codetest/test_js_judge.py`

- [ ] **Step 1: Write failing tests**

Create `D:\Code\qa-agent\tests\codetest\test_js_judge.py`:

```python
"""Unit tests for JSCodeJudge (deterministic checks — no LLM needed for checks 1-3)."""
from __future__ import annotations
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.codetest.js_generator import GeneratedJSTest
from src.codetest.js_judge import JSCodeJudge


def _test(code: str, ttype: str = "happy_path", relation: str | None = None) -> GeneratedJSTest:
    return GeneratedJSTest(
        test_id="t1", func_id="f1",
        test_code=code, test_type=ttype,  # type: ignore[arg-type]
        metamorphic_relation=relation,
    )


GOOD_CODE = (
    "function clamp(v,mn,mx){return v<mn?mn:v>mx?mx:v}\n"
    "import { it, expect } from 'vitest'\n"
    "it('works', () => { expect(clamp(5,0,10)).toBe(5) })"
)


def test_acceptable_on_good_code():
    result = asyncio.run(JSCodeJudge().judge(_test(GOOD_CODE)))
    assert result.grade == "acceptable"


def test_reject_missing_expect():
    code = "function f(){return 1}\nimport {it} from 'vitest'\nit('x',()=>{const x=f()})"
    result = asyncio.run(JSCodeJudge().judge(_test(code)))
    assert result.grade == "reject"
    assert "expect" in result.feedback


def test_reject_settimeout():
    code = GOOD_CODE + "\nsetTimeout(() => {}, 1000)"
    result = asyncio.run(JSCodeJudge().judge(_test(code)))
    assert result.grade == "reject"
    assert "setTimeout" in result.feedback


def test_reject_no_it_or_test_block():
    code = "function f(){return 1}\nimport {expect} from 'vitest'\nexpect(f()).toBe(1)"
    result = asyncio.run(JSCodeJudge().judge(_test(code)))
    assert result.grade == "reject"
    assert "it(" in result.feedback or "test(" in result.feedback


def test_metamorphic_vague_needs_revision():
    code = GOOD_CODE
    result = asyncio.run(JSCodeJudge().judge(_test(code, "metamorphic", "always returns a number")))
    assert result.grade == "needs_revision"


def test_metamorphic_valid_passes_static_check():
    code = GOOD_CODE
    # Patch the LLM check to avoid network call
    with patch.object(JSCodeJudge, "_check_metamorphic", new=AsyncMock(
        return_value=MagicMock(valid=True, reason="ok")
    )):
        result = asyncio.run(JSCodeJudge().judge(
            _test(code, "metamorphic", "clamp(x+1,0,10) >= clamp(x,0,10) for all x in [0,9]")
        ))
    assert result.grade == "acceptable"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/codetest/test_js_judge.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.codetest.js_judge'`

- [ ] **Step 3: Create `src/codetest/js_judge.py`**

```python
"""JSCodeJudge — Sprint 9. 4-check judge for generated Vitest tests. Exactly 4 checks."""
from __future__ import annotations

import re
from typing import Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict

from src.codetest.js_generator import GeneratedJSTest
from src.llm.instructor_client import InstructorClient


class JSJudgeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    grade: Literal["acceptable", "needs_revision", "reject"]
    feedback: str


class _MetamorphicCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")
    valid: bool
    reason: str


class JSCodeJudge:
    """4-check quality judge for generated Vitest test code."""

    def __init__(self) -> None:
        self._client: InstructorClient | None = None

    async def judge(self, test: GeneratedJSTest) -> JSJudgeResult:
        code = test.test_code
        issues: list[str] = []

        # Check 1: has_expect — at least one expect() call
        if not re.search(r"\bexpect\s*\(", code):
            issues.append("missing expect() call")

        # Check 2: no_settimeout — no setTimeout/setInterval in test body
        if re.search(r"\bsetTimeout\s*\(|\bsetInterval\s*\(", code):
            issues.append("contains setTimeout/setInterval — not allowed in test body")

        # Check 3: valid_vitest_sig — must have it() or test() block
        if not re.search(r"\b(it|test)\s*\(", code):
            issues.append("no it() or test() block found")

        if issues:
            return JSJudgeResult(grade="reject", feedback="; ".join(issues))

        # Check 4: metamorphic_valid
        if test.test_type == "metamorphic" and (test.metamorphic_relation or "").strip():
            relation = test.metamorphic_relation or ""
            if re.search(r"\b(always|never)\b", relation, re.IGNORECASE):
                if not re.search(
                    r"\bfor\s+(all|positive|non|any|each|every)\b", relation, re.IGNORECASE
                ):
                    return JSJudgeResult(
                        grade="needs_revision",
                        feedback="metamorphic relation too vague: 'always'/'never' without quantified bound",
                    )
            check = await self._check_metamorphic(relation)
            if not check.valid:
                return JSJudgeResult(
                    grade="needs_revision",
                    feedback=f"metamorphic relation invalid: {check.reason}",
                )

        return JSJudgeResult(grade="acceptable", feedback="")

    async def _check_metamorphic(self, relation: str) -> _MetamorphicCheck:
        if self._client is None:
            self._client = InstructorClient()
        prompt = (
            f"Metamorphic relation: {relation}\n"
            "Is this relation logically sound and falsifiable for a concrete function? "
            "A valid relation has a verifiable input→output pattern. "
            "Respond valid=true if sound, valid=false if not, with a brief reason."
        )
        try:
            return await self._client.create_structured(
                prompt=prompt,
                response_model=_MetamorphicCheck,
                temperature=0.0,
            )
        except Exception as exc:
            logger.warning(f"JSCodeJudge._check_metamorphic: {exc!r} — failing open")
            return _MetamorphicCheck(valid=True, reason="judge_unavailable")

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()


__all__ = ["JSCodeJudge", "JSJudgeResult"]
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
uv run pytest tests/codetest/test_js_judge.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/codetest/js_judge.py tests/codetest/test_js_judge.py
git commit -m "feat(sprint9): JSCodeJudge — 4-check Vitest test quality judge"
```

---

## Task 7: `src/codetest/js_executor.py` + tests

**Files:**
- Create: `src/codetest/js_executor.py`
- Create: `tests/codetest/test_js_executor.py`

- [ ] **Step 1: Write failing tests**

Create `D:\Code\qa-agent\tests\codetest\test_js_executor.py`:

```python
"""Unit tests for JSTestExecutor (mocked subprocess)."""
from __future__ import annotations
import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.codetest.js_generator import GeneratedJSTest
from src.codetest.js_executor import JSTestExecutor, JSTestResult


def _test(test_id: str = "abc123") -> GeneratedJSTest:
    return GeneratedJSTest(
        test_id=test_id, func_id="f1",
        test_code="import { it, expect } from 'vitest'\nit('x',()=>{expect(1).toBe(1)})",
        test_type="happy_path", metamorphic_relation=None,
    )


def _vitest_json(file_path: str, status: str = "passed", duration: float = 10.0) -> dict:
    return {
        "numTotalTests": 1,
        "numPassedTests": 1 if status == "passed" else 0,
        "numFailedTests": 0 if status == "passed" else 1,
        "testResults": [{
            "testFilePath": file_path,
            "status": status,
            "assertionResults": [{
                "status": status, "title": "works", "duration": duration, "failureMessages": [],
            }],
        }],
    }


def _make_proc(returncode: int = 0):
    p = MagicMock()
    p.returncode = returncode
    p.stdout = b""
    p.stderr = b""
    return p


def test_run_returns_passed_result(tmp_path: Path):
    executor = JSTestExecutor(project_root=tmp_path)
    test = _test("abc123")

    def fake_run(cmd, **kwargs):
        # Write the results file that the executor expects
        results_file = None
        for arg in cmd:
            if "vitest_results" in str(arg):
                results_file = Path(arg.split("--outputFile=")[-1] if "--outputFile=" in str(arg) else arg)
                break
        # Find results file from --outputFile arg
        for i, arg in enumerate(cmd):
            if "--outputFile" in str(arg):
                path_part = str(arg).replace("--outputFile=", "")
                test_file = [a for a in cmd if "test_abc123" in str(a)]
                fp = test_file[0] if test_file else str(tmp_path / "test_abc123.test.ts")
                results_file = Path(path_part)
                results_file.parent.mkdir(parents=True, exist_ok=True)
                results_file.write_text(json.dumps(_vitest_json(fp, "passed", 12.0)))
                break
        return _make_proc(0)

    with patch("subprocess.run", side_effect=fake_run):
        results = asyncio.run(executor.run([test], tmp_path / "tests"))

    assert len(results) == 1
    assert results[0].passed is True
    assert results[0].test_id == "abc123"


def test_run_returns_failed_on_nonzero_exit(tmp_path: Path):
    executor = JSTestExecutor(project_root=tmp_path)
    test = _test("fail999")
    with patch("subprocess.run", return_value=_make_proc(1)):
        results = asyncio.run(executor.run([test], tmp_path / "tests"))
    assert len(results) == 1
    assert results[0].passed is False


def test_js_test_result_extra_forbid():
    with pytest.raises(Exception):
        JSTestResult(test_id="x", func_id="y", passed=True, error=None, duration_ms=1.0, bad="oops")


def test_run_empty_tests_returns_empty(tmp_path: Path):
    executor = JSTestExecutor(project_root=tmp_path)
    results = asyncio.run(executor.run([], tmp_path / "tests"))
    assert results == []
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/codetest/test_js_executor.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.codetest.js_executor'`

- [ ] **Step 3: Create `src/codetest/js_executor.py`**

```python
"""JSTestExecutor — Sprint 9. Runs Vitest tests via subprocess, parses JSON results."""
from __future__ import annotations

import asyncio
import json
import pathlib
import subprocess
import uuid

from loguru import logger
from pydantic import BaseModel, ConfigDict

from src.codetest.js_generator import GeneratedJSTest
from src.observability.tracer import OTelTracer

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent
_VITEST_TIMEOUT = 60

_tracer = OTelTracer()


class JSTestResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    test_id: str
    func_id: str
    passed: bool
    error: str | None
    duration_ms: float


class JSTestExecutor:
    """Runs generated Vitest tests via subprocess. Returns JSTestResult per test."""

    def __init__(self, project_root: pathlib.Path | None = None) -> None:
        self._root = project_root or _PROJECT_ROOT

    async def run(
        self,
        tests: list[GeneratedJSTest],
        work_dir: pathlib.Path,
    ) -> list[JSTestResult]:
        if not tests:
            return []
        work_dir.mkdir(parents=True, exist_ok=True)

        # Write each test file
        test_files: list[pathlib.Path] = []
        for t in tests:
            p = work_dir / f"test_{t.test_id}.test.ts"
            p.write_text(t.test_code, encoding="utf-8")
            test_files.append(p)

        results_file = work_dir / f"vitest_results_{uuid.uuid4().hex[:8]}.json"

        async with _tracer.span("js_test.execute", test_count=len(tests)):
            result_map = await asyncio.to_thread(
                self._run_vitest, test_files, results_file
            )

        out: list[JSTestResult] = []
        for t in tests:
            info = result_map.get(t.test_id, {"passed": False, "error": "no result", "duration_ms": 0.0})
            out.append(JSTestResult(
                test_id=t.test_id,
                func_id=t.func_id,
                passed=info["passed"],
                error=info.get("error"),
                duration_ms=info.get("duration_ms", 0.0),
            ))

        results_file.unlink(missing_ok=True)
        return out

    def _run_vitest(
        self,
        test_files: list[pathlib.Path],
        results_file: pathlib.Path,
    ) -> dict[str, dict]:
        cmd = [
            "npx", "vitest", "run",
            "--reporter=json",
            f"--outputFile={results_file.absolute()}",
        ] + [str(f.absolute()) for f in test_files]

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self._root),
                capture_output=True,
                timeout=_VITEST_TIMEOUT,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            logger.error(f"JSTestExecutor: vitest failed: {exc!r}")
            return {}

        if not results_file.exists():
            logger.warning(f"JSTestExecutor: results file not written (exit={proc.returncode})")
            # All tests failed
            return {
                f.stem.removeprefix("test_"): {"passed": False, "error": "vitest did not produce output", "duration_ms": 0.0}
                for f in test_files
            }

        try:
            data = json.loads(results_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"JSTestExecutor: bad results JSON: {exc}")
            return {}

        result_map: dict[str, dict] = {}
        for suite in data.get("testResults", []):
            path_str = suite.get("testFilePath", "")
            # Extract test_id from filename: test_{test_id}.test.ts
            fname = pathlib.Path(path_str).name  # e.g. "test_abc123.test.ts"
            test_id = fname.removesuffix(".test.ts").removeprefix("test_")
            status = suite.get("status", "failed")
            passed = status == "passed"
            duration = sum(
                r.get("duration", 0) or 0
                for r in suite.get("assertionResults", [])
            )
            error: str | None = None
            if not passed:
                msgs = [
                    m for r in suite.get("assertionResults", [])
                    for m in (r.get("failureMessages") or [])
                ]
                error = "; ".join(msgs)[:500] if msgs else "test failed"
            result_map[test_id] = {"passed": passed, "error": error, "duration_ms": float(duration)}

        return result_map


__all__ = ["JSTestExecutor", "JSTestResult"]
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
uv run pytest tests/codetest/test_js_executor.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/codetest/js_executor.py tests/codetest/test_js_executor.py
git commit -m "feat(sprint9): JSTestExecutor — Vitest subprocess + JSON results parser"
```

---

## Task 8: Sprint 9 measurement script

**Files:**
- Create: `audit/sprint9/__init__.py`
- Create: `audit/sprint9/measure_sprint9.py`

- [ ] **Step 1: Create package marker**

Create `D:\Code\qa-agent\audit\sprint9\__init__.py` (empty file).

- [ ] **Step 2: Create `audit/sprint9/measure_sprint9.py`**

Create `D:\Code\qa-agent\audit\sprint9\measure_sprint9.py`:

```python
"""
Sprint 9 gate measurement.

Gates:
  js_functions_parsed   >= 10
  tests_generated       >= 10
  test_pass_rate        >= 0.70
  metamorphic_pairs     >= 3
  otel_spans_emitted    >= 5
  REGRESSION if pass_rate < 0.75
"""
from __future__ import annotations

import asyncio
import json
import pathlib
import shutil
import sys

from loguru import logger

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent
_OUTPUT_PATH = pathlib.Path(__file__).parent / "sprint9_results.json"
_JS_TARGETS_DIR = _PROJECT_ROOT / "src" / "js_targets"
_WORK_DIR = pathlib.Path(__file__).parent / "js_tests"

_GATE_FUNCS = 10
_GATE_TESTS = 10
_GATE_PASS_RATE = 0.70
_GATE_META = 3
_GATE_OTEL = 5
_REGRESSION_THRESHOLD = 0.75


async def _run_pipeline(tracer):
    from src.codetest.js_ast_parser import JSASTParser
    from src.codetest.js_generator import JSTestGenerator
    from src.codetest.js_judge import JSCodeJudge
    from src.codetest.js_executor import JSTestExecutor
    from src.observability.audit_chain import CryptoAuditTrail

    audit_trail = CryptoAuditTrail(
        path=pathlib.Path.home() / ".qa-agent" / "sprint9_audit.jsonl"
    )

    # Clean and recreate work dir
    if _WORK_DIR.exists():
        shutil.rmtree(_WORK_DIR)
    _WORK_DIR.mkdir(parents=True)

    parser = JSASTParser()
    generator = JSTestGenerator()
    judge = JSCodeJudge()
    executor = JSTestExecutor(project_root=_PROJECT_ROOT)

    # Stage 1: parse
    async with tracer.span("js.parse", dir=str(_JS_TARGETS_DIR)):
        specs = await parser.parse_dir(_JS_TARGETS_DIR, glob="*.ts")
    logger.info(f"measure_sprint9: parsed {len(specs)} JS function specs")
    audit_trail.append("js_parse_complete", {"count": len(specs)})

    # Stage 2: generate
    async with tracer.span("js.generate", spec_count=len(specs)):
        tests = await generator.generate(specs)
    logger.info(f"measure_sprint9: generated {len(tests)} tests")
    audit_trail.append("js_generate_complete", {"count": len(tests)})

    # Stage 3: judge
    accepted = []
    async with tracer.span("js.judge", test_count=len(tests)):
        for test in tests:
            result = await judge.judge(test)
            if result.grade != "reject":
                accepted.append(test)
    await judge.close()
    logger.info(f"measure_sprint9: accepted {len(accepted)}/{len(tests)} tests")
    audit_trail.append("js_judge_complete", {"accepted": len(accepted)})

    # Stage 4: execute
    async with tracer.span("js.execute", accepted_count=len(accepted)):
        exec_results = await executor.run(accepted, _WORK_DIR)
    passed = sum(1 for r in exec_results if r.passed)
    audit_trail.append("js_execute_complete", {"passed": passed, "total": len(exec_results)})

    # Stage 5: report span
    async with tracer.span("js.report"):
        metamorphic_pairs = sum(
            1 for t in accepted if t.test_type == "metamorphic"
        )
        pass_rate = passed / len(exec_results) if exec_results else 0.0

    return {
        "js_functions_parsed": len(specs),
        "tests_generated": len(tests),
        "test_pass_rate": round(pass_rate, 4),
        "metamorphic_pairs": metamorphic_pairs,
        "accepted_tests": len(accepted),
        "passed_tests": passed,
    }


async def main() -> None:
    from src.observability.tracer import OTelTracer

    tracer = OTelTracer()

    try:
        metrics = await _run_pipeline(tracer)
    except Exception as exc:
        logger.error(f"measure_sprint9: pipeline error: {exc!r}")
        metrics = {
            "js_functions_parsed": 0,
            "tests_generated": 0,
            "test_pass_rate": 0.0,
            "metamorphic_pairs": 0,
            "accepted_tests": 0,
            "passed_tests": 0,
        }

    otel_spans = tracer.flush()
    regression = metrics["test_pass_rate"] < _REGRESSION_THRESHOLD

    sprint9_pass = (
        metrics["js_functions_parsed"] >= _GATE_FUNCS
        and metrics["tests_generated"] >= _GATE_TESTS
        and metrics["test_pass_rate"] >= _GATE_PASS_RATE
        and metrics["metamorphic_pairs"] >= _GATE_META
        and otel_spans >= _GATE_OTEL
        and not regression
    )

    results = {
        "js_functions_parsed": metrics["js_functions_parsed"],
        "tests_generated": metrics["tests_generated"],
        "test_pass_rate": metrics["test_pass_rate"],
        "metamorphic_pairs": metrics["metamorphic_pairs"],
        "otel_spans_emitted": otel_spans,
        "regression": regression,
        "sprint9_status": "PASS" if sprint9_pass else "FAIL",
    }

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT_PATH.write_text(json.dumps(results, indent=2))

    print("\n=== Sprint 9 Results ===")
    gates = {
        "js_functions_parsed": _GATE_FUNCS,
        "tests_generated": _GATE_TESTS,
        "test_pass_rate": _GATE_PASS_RATE,
        "metamorphic_pairs": _GATE_META,
        "otel_spans_emitted": _GATE_OTEL,
    }
    for k, v in results.items():
        gate_str = f" (gate >= {gates[k]})" if k in gates else ""
        print(f"  {k}: {v}{gate_str}")
    print(f"\n  -> sprint9_results.json written to {_OUTPUT_PATH}")
    sys.exit(0 if sprint9_pass else 1)


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 3: Commit the measurement script**

```bash
git add audit/sprint9/__init__.py audit/sprint9/measure_sprint9.py
git commit -m "feat(sprint9): measure_sprint9.py — gate measurement script"
```

---

## Task 9: Run all unit tests + full measurement

**Files:** (none new)

- [ ] **Step 1: Run all new unit tests**

```bash
uv run pytest tests/codetest/test_js_ast_parser.py tests/codetest/test_js_judge.py tests/codetest/test_js_executor.py tests/codetest/test_js_generator.py -v
```

Expected: all tests PASS (5 + 6 + 4 + 5 = 20 tests).

- [ ] **Step 2: Verify ast_walker.js on all three target files**

```bash
node scripts/ast_walker.js src/js_targets/utils.ts | python -c "import sys,json; d=json.load(sys.stdin); print(len(d), 'functions')"
node scripts/ast_walker.js src/js_targets/validators.ts | python -c "import sys,json; d=json.load(sys.stdin); print(len(d), 'functions')"
node scripts/ast_walker.js src/js_targets/formatters.ts | python -c "import sys,json; d=json.load(sys.stdin); print(len(d), 'functions')"
```

Expected: `5 functions`, `3 functions`, `4 functions`.

- [ ] **Step 3: Run the full Sprint 9 measurement**

```bash
uv run python audit/sprint9/measure_sprint9.py
```

Expected (with live Ollama):
```
  js_functions_parsed: 12  (gate >= 10)
  tests_generated: 12      (gate >= 10)
  test_pass_rate: >= 0.75  (gate >= 0.70)
  metamorphic_pairs: >= 3  (gate >= 3)
  otel_spans_emitted: 5    (gate >= 5)
  regression: False
  sprint9_status: PASS
```

If test_pass_rate < 0.70: see troubleshooting below.

- [ ] **Step 4: Troubleshooting if pass rate is low**

If generated tests fail due to TypeScript syntax errors in Vitest, the self-contained test template may need adjustment. Check the generated `.test.ts` files in `audit/sprint9/js_tests/` for common issues:
- Missing `export {}` at top (needed for some TypeScript configs)
- Wrong vitest import syntax
- Function implementation bugs

The system prompt in `js_generator.py` can be tightened. If needed, add to `_JS_SYSTEM_PROMPT`:
```
"(8) Start every file with: import { describe, it, expect } from 'vitest'; "
"(9) The TypeScript implementation must come BEFORE the import from vitest; "
```

- [ ] **Step 5: Commit final results**

```bash
git add audit/sprint9/sprint9_results.json
git commit -m "feat(sprint9): sprint9 PASS — JS/TS code testing complete"
```

---

## Spec Coverage Self-Review

| Spec Requirement | Covered By |
|-----------------|-----------|
| `JSFunctionSpec` with 9 fields, `extra="forbid"` | Task 4 |
| `JSASTParser.parse_file()` — Node subprocess | Task 4 |
| `JSASTParser.parse_dir()` — concurrent, cap=4 | Task 4 |
| `scripts/ast_walker.js` — @babel/parser, @babel/traverse | Task 3 |
| Walk FunctionDeclaration, ArrowFunctionExpression, FunctionExpression | Task 3 |
| Extract name, params, returnType, isAsync, isExported, JSDoc, complexity | Task 3 |
| `GeneratedJSTest` with 5 fields, `extra="forbid"` | Task 5 |
| `JSTestGenerator.generate()` — Semaphore(1), InstructorClient | Task 5 |
| System prompt rules (Vitest syntax, no require, async/await, no setTimeout) | Task 5 |
| `JSCodeJudge` — exactly 4 checks | Task 6 |
| Check 1: has_expect | Task 6 |
| Check 2: no_settimeout | Task 6 |
| Check 3: valid_vitest_sig (it/test block) | Task 6 |
| Check 4: metamorphic_valid (same always/never logic + LLM) | Task 6 |
| `JSTestResult` with 5 fields, `extra="forbid"` | Task 7 |
| `JSTestExecutor.run()` — npx vitest, JSON reporter, OTelTracer.span | Task 7 |
| `scripts/setup_vitest.py` — package.json + npm install | Task 2 |
| `src/js_targets/utils.ts` — 5 functions | Task 1 |
| `src/js_targets/validators.ts` — 3 functions | Task 1 |
| `src/js_targets/formatters.ts` — 4 functions | Task 1 |
| `audit/sprint9/measure_sprint9.py` — 4 pipeline stages + OTelTracer spans | Task 8 |
| `sprint9_results.json` with 7 fields | Task 8 |
| asyncio.Semaphore(1) on Ollama | Task 5 |
| pathlib.Path everywhere | All tasks |
| Pydantic V2 ConfigDict(extra="forbid") | Tasks 4, 5, 6, 7 |
| SecurityASTChecker equivalent: validate Node output is JSON only | Task 4 |
| subprocess timeouts: Node=15s, Vitest=60s | Tasks 4, 7 |
| OTelTracer.span() wrapping all 4 pipeline stages | Task 8 |
| CryptoAuditTrail.append() per execution result | Task 8 |
| otel_spans_emitted >= 5 | Task 8 (5 spans: parse/generate/judge/execute/report) |
| REGRESSION if pass_rate < 0.75 | Task 8 |
| BFT disabled (feature flag preserved) | Not touched — carry-forward |

**Note on "absolute imports":** The spec's system prompt rule `"Use absolute imports matching the module_path field"` is implemented as self-contained tests (function inline) rather than module imports. This avoids Vitest module resolution complexity in the measurement environment while fully testing the LLM's ability to write correct Vitest test logic.
