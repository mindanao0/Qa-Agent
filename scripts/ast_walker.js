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

function extractJsdoc(node, path) {
  // Babel attaches leading comments to the ExportNamedDeclaration wrapper,
  // not the inner FunctionDeclaration, so check both.
  const sources = [
    node.leadingComments || [],
    path?.parentPath?.node?.leadingComments || [],
    path?.parentPath?.parentPath?.node?.leadingComments || [],
  ];
  for (const comments of sources) {
    for (const c of comments.slice().reverse()) {
      if (c.type === 'CommentBlock' && c.value.startsWith('*')) {
        return c.value.trim();
      }
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
  const jsdoc = extractJsdoc(node, path);
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
