#!/usr/bin/env python3
"""Read-only discovery and project-adoption audit for agent_workflows."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"not strict JSON-compatible YAML: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"document root must be an object: {path}")
    return value


def locate_root(explicit: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(explicit.expanduser())
    configured = os.environ.get("AGENT_WORKFLOWS_ROOT")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.append(Path.home() / ".codex" / "agent-workflows")
    candidates.append(Path("/home/zc/agent_workflows"))
    for candidate in candidates:
        resolved = candidate.resolve()
        if (resolved / "registry.yaml").is_file():
            return resolved
    joined = ", ".join(str(path) for path in candidates)
    raise ValueError(f"cannot locate agent_workflows registry; tried: {joined}")


def index_registry(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    registry = load_json(root / "registry.yaml")
    components = {entry["id"]: entry for entry in registry.get("components", [])}
    workflows = {entry["id"]: entry for entry in registry.get("workflows", [])}
    return registry, components, workflows


def list_registry(root: Path, as_json: bool) -> int:
    registry, components, workflows = index_registry(root)
    result = {
        "root": str(root),
        "schema_version": registry.get("schema_version"),
        "global_components": registry.get("global_components", {}),
        "components": sorted(components),
        "workflows": [],
    }
    for workflow_id, entry in sorted(workflows.items()):
        workflow = load_json(root / entry["descriptor"])
        result["workflows"].append(
            {"id": workflow_id, "release": workflow.get("release"), "status": workflow.get("status")}
        )
    if as_json:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    else:
        print(f"ROOT {root}")
        print("GLOBAL_COMPONENTS")
        for component_id, release in sorted(result["global_components"].items()):
            print(f"  {component_id}@{release}")
        print("WORKFLOWS")
        for item in result["workflows"]:
            print(f"  {item['id']}@{item['release']} [{item['status']}]")
        print(f"COMPONENTS {len(result['components'])}")
        for component_id in result["components"]:
            print(f"  {component_id}")
    return 0


def show_entry(root: Path, identity: str) -> int:
    _, components, workflows = index_registry(root)
    entry = components.get(identity) or workflows.get(identity)
    if not entry:
        raise ValueError(f"not registered: {identity}")
    descriptor = load_json(root / entry["descriptor"])
    print(json.dumps({"registry_entry": entry, "descriptor": descriptor}, ensure_ascii=False, indent=2))
    return 0


def audit_project(root: Path, project: Path, declaration: Path | None, as_json: bool) -> int:
    project = project.expanduser().resolve()
    declaration = declaration.expanduser().resolve() if declaration else project / ".codex" / "workflows.yaml"
    report: dict[str, Any] = {
        "project": str(project),
        "declaration": str(declaration),
        "status": "PASS",
        "checks": [],
    }

    def check(ok: bool, name: str, detail: str) -> None:
        report["checks"].append({"status": "PASS" if ok else "FAIL", "check": name, "detail": detail})
        if not ok:
            report["status"] = "FAIL"

    if not declaration.is_file():
        report["status"] = "NOT_DECLARED"
        report["checks"].append(
            {"status": "INFO", "check": "project.declaration", "detail": "no shared Workflow adoption declared"}
        )
        if as_json:
            print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        else:
            print(f"NOT_DECLARED {declaration}")
        return 2

    document = load_json(declaration)
    check(document.get("schema_version") == 1, "project.schema", str(document.get("schema_version")))
    uses = document.get("uses", [])
    check(isinstance(uses, list) and bool(uses), "project.uses", f"count={len(uses) if isinstance(uses, list) else 0}")
    _, _, workflows = index_registry(root)
    seen: set[str] = set()
    if isinstance(uses, list):
        for item in uses:
            if not isinstance(item, dict):
                check(False, "project.use", "entry must be an object")
                continue
            workflow_id = item.get("workflow")
            release = item.get("release")
            check(bool(workflow_id) and workflow_id not in seen, f"project.workflow.{workflow_id}", "unique id")
            seen.add(workflow_id)
            entry = workflows.get(workflow_id)
            check(entry is not None, f"registry.workflow.{workflow_id}", "registered")
            if not entry:
                continue
            workflow = load_json(root / entry["descriptor"])
            check(release == workflow.get("release"), f"workflow.release.{workflow_id}", f"declared={release} registered={workflow.get('release')}")
            lock = load_json(root / entry["lock"])
            check(lock.get("workflow_release") == release, f"workflow.lock.{workflow_id}", str(root / entry["lock"]))

    if as_json:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    else:
        for item in report["checks"]:
            print(f"{item['status']:4} {item['check']}: {item['detail']}")
        print(f"RESULT {report['status']}")
    return 0 if report["status"] == "PASS" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Discover shared Agent workflows and audit project adoption without changing files.")
    parser.add_argument("--root", type=Path, help="Override the shared repository root.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List registered components and workflows.")
    list_parser.add_argument("--json", action="store_true")

    show_parser = subparsers.add_parser("show", help="Show one component or workflow descriptor.")
    show_parser.add_argument("identity")

    audit_parser = subparsers.add_parser("audit-project", help="Audit <project>/.codex/workflows.yaml.")
    audit_parser.add_argument("project", type=Path)
    audit_parser.add_argument("--declaration", type=Path)
    audit_parser.add_argument("--json", action="store_true")

    args = parser.parse_args()
    try:
        root = locate_root(args.root)
        if args.command == "list":
            return list_registry(root, args.json)
        if args.command == "show":
            return show_entry(root, args.identity)
        return audit_project(root, args.project, args.declaration, args.json)
    except ValueError as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

