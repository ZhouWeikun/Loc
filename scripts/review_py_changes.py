#!/usr/bin/env python3
"""Watch Python file changes and emit lightweight logic review reports."""

from __future__ import annotations

import argparse
import ast
import dataclasses
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable


DEFAULT_EXCLUDED_DIRS = {
    ".cache",
    ".git",
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "env",
    "evaluate_img_encoder",
    "evaluate_img_encdoer",
    "train_img_encoder",
    "train_map_multimlp",
    "train_map_singlemlp",
    "venv",
}

DEFAULT_STATE_DIR = Path(".cache/py_change_review")
DEFAULT_STATE_FILE = DEFAULT_STATE_DIR / "monitor_state.json"
DEFAULT_REPORT_DIR = DEFAULT_STATE_DIR / "reports"

CONTROL_FLOW_NODES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.Match,
)

SEVERITY_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}


@dataclasses.dataclass(frozen=True)
class Finding:
    severity: str
    line: int
    symbol: str
    message: str


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Monitor *.py file creation/modification and generate lightweight "
            "logic review reports."
        )
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Project root to scan. Defaults to the current directory.",
    )
    parser.add_argument(
        "--state-file",
        default=None,
        help="Snapshot state file path. Defaults to .cache/py_change_review/monitor_state.json.",
    )
    parser.add_argument(
        "--report-dir",
        default=None,
        help="Directory for markdown reports. Defaults to .cache/py_change_review/reports/.",
    )
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=[],
        help="Directory name to exclude from scanning. Can be passed multiple times.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "snapshot",
        help="Create or refresh the file snapshot without generating review reports.",
    )
    scan_parser = subparsers.add_parser(
        "scan",
        help="Compare the current workspace with the saved snapshot and review changed files once.",
    )
    scan_parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit the number of changed files reviewed in this run.",
    )

    watch_parser = subparsers.add_parser(
        "watch",
        help="Continuously monitor the workspace and review changed files.",
    )
    watch_parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Polling interval in seconds. Defaults to 2.0.",
    )

    git_parser = subparsers.add_parser(
        "git",
        help="Review modified/untracked Python files relative to git HEAD.",
    )
    git_parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit the number of changed files reviewed in this run.",
    )

    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, set[str]]:
    root = Path(args.root).resolve()
    state_file = Path(args.state_file).resolve() if args.state_file else (root / DEFAULT_STATE_FILE)
    report_dir = Path(args.report_dir).resolve() if args.report_dir else (root / DEFAULT_REPORT_DIR)
    excluded_dirs = set(DEFAULT_EXCLUDED_DIRS)
    excluded_dirs.update(args.exclude_dir)
    return root, state_file, report_dir, excluded_dirs


def should_skip_dir(name: str, excluded_dirs: set[str]) -> bool:
    return name.startswith(".") or name in excluded_dirs


def should_skip_rel_path(rel_path: str, excluded_dirs: set[str]) -> bool:
    parts = Path(rel_path).parts
    return any(part.startswith(".") or part in excluded_dirs for part in parts)


def iter_python_files(root: Path, excluded_dirs: set[str]) -> Iterable[Path]:
    for current_root, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            dirname
            for dirname in dirnames
            if not should_skip_dir(dirname, excluded_dirs)
        )
        for filename in sorted(filenames):
            if not filename.endswith(".py") or filename.startswith("."):
                continue
            yield Path(current_root) / filename


def count_function_args(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    total = len(node.args.posonlyargs)
    total += len(node.args.args)
    total += len(node.args.kwonlyargs)
    total += 1 if node.args.vararg else 0
    total += 1 if node.args.kwarg else 0
    return total


def is_mutable_default(node: ast.AST | None) -> bool:
    if node is None:
        return False
    if isinstance(node, (ast.List, ast.Dict, ast.Set)):
        return True
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        return node.func.id in {"dict", "list", "set"}
    return False


def iter_local_nodes(node: ast.AST) -> Iterable[ast.AST]:
    stack: list[ast.AST] = list(ast.iter_child_nodes(node))
    while stack:
        child = stack.pop()
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        yield child
        stack.extend(ast.iter_child_nodes(child))


def statement_bodies(node: ast.stmt) -> list[list[ast.stmt]]:
    bodies: list[list[ast.stmt]] = []
    for attr in ("body", "orelse", "finalbody"):
        value = getattr(node, attr, None)
        if value:
            bodies.append(value)
    if isinstance(node, ast.Try):
        for handler in node.handlers:
            if handler.body:
                bodies.append(handler.body)
    if isinstance(node, ast.Match):
        for case in node.cases:
            if case.body:
                bodies.append(case.body)
    return bodies


def max_nesting_depth(statements: list[ast.stmt], depth: int = 0) -> int:
    max_depth = depth
    for statement in statements:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(statement, CONTROL_FLOW_NODES):
            branch_depth = depth + 1
            max_depth = max(max_depth, branch_depth)
            for body in statement_bodies(statement):
                max_depth = max(max_depth, max_nesting_depth(body, branch_depth))
    return max_depth


def is_main_guard(statement: ast.stmt) -> bool:
    if not isinstance(statement, ast.If):
        return False
    test = statement.test
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    )


def render_import_names(node: ast.Import | ast.ImportFrom) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    module = node.module or ""
    rendered = []
    for alias in node.names:
        rendered.append(f"{module}.{alias.name}" if module else alias.name)
    return rendered


def build_summary(tree: ast.Module, source: str) -> dict[str, Any]:
    top_level_functions = []
    top_level_classes = []
    imports: set[str] = set()
    top_level_exec = 0

    for statement in tree.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            top_level_functions.append(
                {
                    "name": statement.name,
                    "async": isinstance(statement, ast.AsyncFunctionDef),
                    "args": count_function_args(statement),
                    "line": statement.lineno,
                }
            )
            continue
        if isinstance(statement, ast.ClassDef):
            method_count = sum(
                1
                for item in statement.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            )
            top_level_classes.append(
                {
                    "name": statement.name,
                    "methods": method_count,
                    "line": statement.lineno,
                }
            )
            continue
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            imports.update(render_import_names(statement))
            continue
        if not is_main_guard(statement):
            top_level_exec += 1

    return {
        "line_count": len(source.splitlines()),
        "top_level_functions": sorted(top_level_functions, key=lambda item: item["name"]),
        "top_level_classes": sorted(top_level_classes, key=lambda item: item["name"]),
        "imports": sorted(imports),
        "top_level_exec": top_level_exec,
    }


class LogicReviewVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.findings: list[Finding] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._review_function(node, async_mode=False)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._review_function(node, async_mode=True)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name == "*":
                self.findings.append(
                    Finding(
                        severity="MEDIUM",
                        line=node.lineno,
                        symbol="import",
                        message="Wildcard import makes data flow and symbol ownership hard to audit.",
                    )
                )
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is None:
            self.findings.append(
                Finding(
                    severity="HIGH",
                    line=node.lineno,
                    symbol="except",
                    message="Bare except swallows unexpected failures and hides root causes.",
                )
            )
        elif isinstance(node.type, ast.Name) and node.type.id in {"Exception", "BaseException"}:
            self.findings.append(
                Finding(
                    severity="MEDIUM",
                    line=node.lineno,
                    symbol="except",
                    message=f"Broad exception catch on {node.type.id} can mask logic errors.",
                )
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec"}:
            self.findings.append(
                Finding(
                    severity="HIGH",
                    line=node.lineno,
                    symbol=node.func.id,
                    message=f"Use of {node.func.id} introduces dynamic execution that is hard to reason about safely.",
                )
            )
        self.generic_visit(node)

    def _review_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        async_mode: bool,
    ) -> None:
        symbol = f"{'async ' if async_mode else ''}function `{node.name}`"
        arg_count = count_function_args(node)
        if arg_count > 8:
            self.findings.append(
                Finding(
                    severity="MEDIUM",
                    line=node.lineno,
                    symbol=symbol,
                    message=f"Has {arg_count} parameters; the call contract is wide and easy to misuse.",
                )
            )

        end_lineno = getattr(node, "end_lineno", node.lineno)
        line_span = end_lineno - node.lineno + 1
        if line_span > 80:
            self.findings.append(
                Finding(
                    severity="MEDIUM",
                    line=node.lineno,
                    symbol=symbol,
                    message=f"Spans {line_span} lines; consider splitting branches/state transitions into smaller units.",
                )
            )

        nesting = max_nesting_depth(node.body)
        if nesting > 4:
            self.findings.append(
                Finding(
                    severity="MEDIUM",
                    line=node.lineno,
                    symbol=symbol,
                    message=f"Control-flow nesting depth is {nesting}, which makes edge cases hard to validate.",
                )
            )

        defaults = list(zip(node.args.args[-len(node.args.defaults) :], node.args.defaults))
        defaults.extend(zip(node.args.kwonlyargs, node.args.kw_defaults))
        for arg_node, default_node in defaults:
            if is_mutable_default(default_node):
                self.findings.append(
                    Finding(
                        severity="HIGH",
                        line=arg_node.lineno,
                        symbol=symbol,
                        message=f"Parameter `{arg_node.arg}` uses a mutable default value.",
                    )
                )

        has_yield = False
        has_return_value = False
        has_return_none = False
        global_names: set[str] = set()
        for child in iter_local_nodes(node):
            if isinstance(child, (ast.Yield, ast.YieldFrom)):
                has_yield = True
            elif isinstance(child, ast.Return):
                if child.value is None:
                    has_return_none = True
                else:
                    has_return_value = True
            elif isinstance(child, ast.Global):
                global_names.update(child.names)

        if not has_yield and has_return_value and has_return_none:
            self.findings.append(
                Finding(
                    severity="MEDIUM",
                    line=node.lineno,
                    symbol=symbol,
                    message="Mixes value returns with bare returns; callers may receive inconsistent result shapes.",
                )
            )

        if global_names:
            self.findings.append(
                Finding(
                    severity="MEDIUM",
                    line=node.lineno,
                    symbol=symbol,
                    message=(
                        "Writes to module-level state via global "
                        f"{', '.join(sorted(global_names))}; execution order becomes important."
                    ),
                )
            )


def analyze_source(source: str) -> tuple[dict[str, Any], list[Finding]]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        line_number = exc.lineno or 1
        summary = {
            "line_count": len(source.splitlines()),
            "top_level_functions": [],
            "top_level_classes": [],
            "imports": [],
            "top_level_exec": 0,
            "parse_error": {
                "line": line_number,
                "message": exc.msg,
            },
        }
        findings = [
            Finding(
                severity="HIGH",
                line=line_number,
                symbol="syntax",
                message=f"Syntax error: {exc.msg}",
            )
        ]
        return summary, findings

    summary = build_summary(tree, source)
    visitor = LogicReviewVisitor()
    visitor.visit(tree)
    findings = sorted(
        visitor.findings,
        key=lambda item: (SEVERITY_ORDER[item.severity], item.line, item.symbol),
    )
    return summary, findings


def build_file_record(path: Path, root: Path) -> dict[str, Any]:
    source = read_text(path)
    summary, _ = analyze_source(source)
    rel_path = path.relative_to(root).as_posix()
    return {
        "path": rel_path,
        "sha1": sha1_text(source),
        "size": len(source.encode("utf-8", errors="replace")),
        "mtime_ns": path.stat().st_mtime_ns,
        "summary": summary,
    }


def build_state(root: Path, excluded_dirs: set[str]) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for path in iter_python_files(root, excluded_dirs):
        rel_path = path.relative_to(root).as_posix()
        files[rel_path] = build_file_record(path, root)
    return {
        "version": 1,
        "root": root.as_posix(),
        "updated_at": utc_now(),
        "excluded_dirs": sorted(excluded_dirs),
        "files": files,
    }


def load_state(state_file: Path) -> dict[str, Any] | None:
    if not state_file.exists():
        return None
    return json.loads(read_text(state_file))


def save_state(state: dict[str, Any], state_file: Path) -> None:
    ensure_parent(state_file)
    state_file.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def list_changed_from_state(
    previous_state: dict[str, Any] | None,
    current_state: dict[str, Any],
) -> list[tuple[str, str]]:
    previous_files = previous_state["files"] if previous_state else {}
    current_files = current_state["files"]

    changed: list[tuple[str, str]] = []
    for rel_path, record in current_files.items():
        previous = previous_files.get(rel_path)
        if previous is None:
            changed.append((rel_path, "created"))
        elif previous["sha1"] != record["sha1"]:
            changed.append((rel_path, "modified"))
    return changed


def summarize_named_delta(
    previous_items: list[dict[str, Any]],
    current_items: list[dict[str, Any]],
    key: str,
    plural_label: str,
    singular_label: str,
) -> list[str]:
    previous_map = {item["name"]: item for item in previous_items}
    current_map = {item["name"]: item for item in current_items}

    deltas: list[str] = []
    added = sorted(current_map.keys() - previous_map.keys())
    removed = sorted(previous_map.keys() - current_map.keys())
    common = sorted(current_map.keys() & previous_map.keys())

    if added:
        deltas.append(f"Added {plural_label}: {', '.join(added)}")
    if removed:
        deltas.append(f"Removed {plural_label}: {', '.join(removed)}")

    for name in common:
        previous = previous_map[name]
        current = current_map[name]
        previous_value = previous.get(key)
        current_value = current.get(key)
        if previous_value != current_value:
            deltas.append(
                f"Changed {singular_label} signature: {name} ({previous_value} -> {current_value})"
            )
    return deltas


def summarize_delta(previous_summary: dict[str, Any] | None, current_summary: dict[str, Any]) -> list[str]:
    if not previous_summary:
        return ["No prior snapshot for comparison."]

    delta: list[str] = []
    delta.extend(
        summarize_named_delta(
            previous_summary.get("top_level_functions", []),
            current_summary.get("top_level_functions", []),
            key="args",
            plural_label="functions",
            singular_label="function",
        )
    )
    delta.extend(
        summarize_named_delta(
            previous_summary.get("top_level_classes", []),
            current_summary.get("top_level_classes", []),
            key="methods",
            plural_label="classes",
            singular_label="class",
        )
    )

    previous_imports = set(previous_summary.get("imports", []))
    current_imports = set(current_summary.get("imports", []))
    added_imports = sorted(current_imports - previous_imports)
    removed_imports = sorted(previous_imports - current_imports)
    if added_imports:
        delta.append(f"Added imports: {', '.join(added_imports[:8])}")
    if removed_imports:
        delta.append(f"Removed imports: {', '.join(removed_imports[:8])}")

    previous_exec = previous_summary.get("top_level_exec", 0)
    current_exec = current_summary.get("top_level_exec", 0)
    if previous_exec != current_exec:
        delta.append(f"Top-level executable statements: {previous_exec} -> {current_exec}")

    if previous_summary.get("parse_error") != current_summary.get("parse_error"):
        if current_summary.get("parse_error"):
            delta.append(
                f"Current parse error: {current_summary['parse_error']['message']} "
                f"(line {current_summary['parse_error']['line']})"
            )
        elif previous_summary.get("parse_error"):
            delta.append("Previous parse error resolved.")

    return delta or ["No structural delta detected beyond content changes."]


def sanitize_slug(rel_path: str) -> str:
    return rel_path.replace("/", "__")


def write_report(
    report_dir: Path,
    rel_path: str,
    event: str,
    current_summary: dict[str, Any],
    delta: list[str],
    findings: list[Finding],
) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    report_path = report_dir / f"{timestamp}__{event}__{sanitize_slug(rel_path)}.md"

    lines = [
        "# Python Change Review",
        "",
        f"- Time: {utc_now()}",
        f"- Event: {event}",
        f"- File: {rel_path}",
        f"- Lines: {current_summary.get('line_count', 0)}",
        f"- Top-level functions: {len(current_summary.get('top_level_functions', []))}",
        f"- Top-level classes: {len(current_summary.get('top_level_classes', []))}",
        "",
        "## Structural Delta",
    ]
    for item in delta:
        lines.append(f"- {item}")

    lines.extend(["", "## Findings"])
    if findings:
        for finding in findings:
            lines.append(
                f"- [{finding.severity}] line {finding.line} {finding.symbol}: {finding.message}"
            )
    else:
        lines.append("- No high-risk logic patterns were flagged by the heuristic review.")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def review_workspace_file(
    root: Path,
    report_dir: Path,
    rel_path: str,
    event: str,
    previous_summary: dict[str, Any] | None,
) -> tuple[Path, list[Finding], dict[str, Any]]:
    source = read_text(root / rel_path)
    current_summary, findings = analyze_source(source)
    delta = summarize_delta(previous_summary, current_summary)
    report_path = write_report(report_dir, rel_path, event, current_summary, delta, findings)
    return report_path, findings, current_summary


def print_review_result(rel_path: str, event: str, report_path: Path, findings: list[Finding]) -> None:
    severity_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for finding in findings:
        severity_counts[finding.severity] += 1

    print(f"[{event}] {rel_path}")
    print(f"  report: {report_path}")
    print(
        "  findings: "
        f"HIGH={severity_counts['HIGH']} "
        f"MEDIUM={severity_counts['MEDIUM']} "
        f"LOW={severity_counts['LOW']}"
    )


def scan_once(
    root: Path,
    state_file: Path,
    report_dir: Path,
    excluded_dirs: set[str],
    limit: int = 0,
) -> int:
    previous_state = load_state(state_file)
    current_state = build_state(root, excluded_dirs)

    if previous_state is None:
        save_state(current_state, state_file)
        print(f"No snapshot found. Created baseline at {state_file}")
        return 0

    changed = list_changed_from_state(previous_state, current_state)
    truncated = False
    if limit > 0 and len(changed) > limit:
        changed = changed[:limit]
        truncated = True

    if not changed:
        print("No new or modified Python files detected.")
        save_state(current_state, state_file)
        return 0

    previous_files = previous_state["files"]
    for rel_path, event in changed:
        previous_summary = previous_files.get(rel_path, {}).get("summary")
        report_path, findings, _ = review_workspace_file(
            root=root,
            report_dir=report_dir,
            rel_path=rel_path,
            event=event,
            previous_summary=previous_summary,
        )
        print_review_result(rel_path, event, report_path, findings)

    if truncated:
        print(
            "Review limit reached; snapshot was not updated so the remaining changes "
            "can be reviewed in a later run."
        )
    else:
        save_state(current_state, state_file)
    return len(changed)


def run_git(
    root: Path,
    args: list[str],
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=check,
        capture_output=True,
        text=True,
    )


def list_git_changed_py(root: Path, excluded_dirs: set[str]) -> list[tuple[str, str]]:
    try:
        diff_result = run_git(root, ["diff", "--name-only", "HEAD"])
        untracked_result = run_git(
            root,
            ["ls-files", "--others", "--exclude-standard"],
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise RuntimeError(f"git query failed: {stderr}") from exc

    changed: list[tuple[str, str]] = []
    seen: set[str] = set()

    for rel_path in diff_result.stdout.splitlines():
        rel_path = rel_path.strip()
        if not rel_path or not rel_path.endswith(".py") or should_skip_rel_path(rel_path, excluded_dirs):
            continue
        if not (root / rel_path).exists():
            continue
        seen.add(rel_path)
        event = "modified" if git_has_path_at_head(root, rel_path) else "created"
        changed.append((rel_path, event))

    for rel_path in untracked_result.stdout.splitlines():
        rel_path = rel_path.strip()
        if (
            not rel_path
            or not rel_path.endswith(".py")
            or rel_path in seen
            or should_skip_rel_path(rel_path, excluded_dirs)
        ):
            continue
        if not (root / rel_path).exists():
            continue
        changed.append((rel_path, "created"))

    return sorted(changed)


def git_has_path_at_head(root: Path, rel_path: str) -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"HEAD:{rel_path}"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def git_head_text(root: Path, rel_path: str) -> str | None:
    result = subprocess.run(
        ["git", "show", f"HEAD:{rel_path}"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def review_git_changes(
    root: Path,
    report_dir: Path,
    excluded_dirs: set[str],
    limit: int = 0,
) -> int:
    changed = list_git_changed_py(root, excluded_dirs)
    if limit > 0:
        changed = changed[:limit]

    if not changed:
        print("No modified or untracked Python files relative to git HEAD.")
        return 0

    for rel_path, event in changed:
        previous_source = git_head_text(root, rel_path)
        previous_summary = analyze_source(previous_source)[0] if previous_source is not None else None
        report_path, findings, _ = review_workspace_file(
            root=root,
            report_dir=report_dir,
            rel_path=rel_path,
            event=event,
            previous_summary=previous_summary,
        )
        print_review_result(rel_path, event, report_path, findings)

    return len(changed)


def run_watch(
    root: Path,
    state_file: Path,
    report_dir: Path,
    excluded_dirs: set[str],
    interval: float,
) -> int:
    previous_state = load_state(state_file)
    if previous_state is None:
        previous_state = build_state(root, excluded_dirs)
        save_state(previous_state, state_file)
        print(f"Created baseline at {state_file}")

    print(
        f"Watching {root} every {interval:.1f}s; excluded dirs: "
        f"{', '.join(sorted(excluded_dirs))}"
    )
    try:
        while True:
            current_state = build_state(root, excluded_dirs)
            changed = list_changed_from_state(previous_state, current_state)
            for rel_path, event in changed:
                previous_summary = previous_state["files"].get(rel_path, {}).get("summary")
                report_path, findings, _ = review_workspace_file(
                    root=root,
                    report_dir=report_dir,
                    rel_path=rel_path,
                    event=event,
                    previous_summary=previous_summary,
                )
                print_review_result(rel_path, event, report_path, findings)
            if changed:
                save_state(current_state, state_file)
                previous_state = current_state
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nWatch stopped.")
        return 0


def main() -> int:
    args = parse_args()
    root, state_file, report_dir, excluded_dirs = resolve_paths(args)

    if args.command == "snapshot":
        state = build_state(root, excluded_dirs)
        save_state(state, state_file)
        print(f"Snapshot saved to {state_file}")
        return 0

    if args.command == "scan":
        scan_once(root, state_file, report_dir, excluded_dirs, limit=args.limit)
        return 0

    if args.command == "watch":
        return run_watch(
            root=root,
            state_file=state_file,
            report_dir=report_dir,
            excluded_dirs=excluded_dirs,
            interval=args.interval,
        )

    if args.command == "git":
        review_git_changes(root, report_dir, excluded_dirs, limit=args.limit)
        return 0

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
