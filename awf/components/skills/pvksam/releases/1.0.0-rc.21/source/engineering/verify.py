#!/usr/bin/env python3
"""Verify repository manifests, version locks, installed runtimes, and artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(os.environ.get("AWF_ROOT", str(Path(__file__).resolve().parents[5])))
IGNORED_TREE_PARTS = {"__pycache__", ".pytest_cache"}
ALLOWED_COMPONENT_KINDS = {"tool", "skill", "recipe", "artifact"}
ALLOWED_RELEASE_STATUSES = {"candidate", "validated", "deprecated", "superseded", "captured"}
ALLOWED_WORKFLOW_STATUSES = {"candidate", "validated", "deprecated", "superseded"}


def read_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"manifest root is not an object: {path}")
    return value


def regular_files(root: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file()
            and not path.is_symlink()
            and not any(part in IGNORED_TREE_PARTS for part in path.relative_to(root).parts)
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(root: Path) -> str:
    """Hash sorted `sha256  ./relative/path` lines, matching the captured baseline."""
    digest = hashlib.sha256()
    for path in regular_files(root):
        relative = path.relative_to(root).as_posix()
        digest.update(f"{file_sha256(path)}  ./{relative}\n".encode("utf-8"))
    return digest.hexdigest()


def tree_file_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in regular_files(root))


def integrity_relative_path(value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
        raise ValueError(f"invalid integrity path: {value}")
    return relative


def tree_v2_paths(
    root: Path,
    coverage: str,
    paths: list[str] | None = None,
    exclude_paths: list[str] | None = None,
) -> list[Path]:
    if coverage in {"tree", "tree-excluding"}:
        excluded = {
            integrity_relative_path(value).as_posix()
            for value in (exclude_paths or [])
        }
        return sorted(
            (
                path
                for path in root.rglob("*")
                if not any(part in IGNORED_TREE_PARTS for part in path.relative_to(root).parts)
                and path.relative_to(root).as_posix() not in excluded
                and not any(
                    path.relative_to(root).as_posix().startswith(f"{prefix}/")
                    for prefix in excluded
                )
            ),
            key=lambda path: path.relative_to(root).as_posix(),
        )
    if coverage == "explicit-paths" and isinstance(paths, list):
        return [root / integrity_relative_path(value) for value in sorted(paths)]
    raise ValueError(f"unsupported integrity coverage: {coverage}")


def tree_sha256_v2(
    root: Path,
    coverage: str,
    paths: list[str] | None = None,
    exclude_paths: list[str] | None = None,
) -> str:
    """Hash canonical JSON records for files, directories, and literal symlink targets."""
    digest = hashlib.sha256()
    for path in tree_v2_paths(root, coverage, paths, exclude_paths):
        relative = path.relative_to(root).as_posix()
        metadata = path.lstat()
        record: dict[str, Any] = {
            "mode": f"{stat.S_IMODE(metadata.st_mode):04o}",
            "path": relative,
        }
        if path.is_symlink():
            record.update({"kind": "symlink", "target": os.readlink(path)})
        elif path.is_file():
            record.update(
                {
                    "bytes": metadata.st_size,
                    "kind": "file",
                    "sha256": file_sha256(path),
                }
            )
        elif path.is_dir():
            record["kind"] = "directory"
        else:
            record["kind"] = "other"
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        digest.update(f"{line}\n".encode("utf-8"))
    return digest.hexdigest()


def canonical_object_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Report:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def add(self, ok: bool, check: str, detail: str) -> None:
        self.checks.append({"status": "PASS" if ok else "FAIL", "check": check, "detail": detail})

    @property
    def ok(self) -> bool:
        return all(item["status"] == "PASS" for item in self.checks)


def split_reference(value: str) -> tuple[str, str]:
    component_id, separator, release = value.rpartition("@")
    if not separator or not component_id or not release:
        raise ValueError(f"invalid component reference: {value}")
    return component_id, release


def artifact_payload_path(payload: dict[str, Any]) -> Path:
    path_value = payload["path"]
    path_env = payload.get("path_env")
    if isinstance(path_env, str) and os.environ.get(path_env):
        path_value = os.environ[path_env]
    return Path(path_value).expanduser()


def distribution_version(python: str, distribution: str) -> tuple[bool, str]:
    code = (
        "from importlib.metadata import version; "
        f"print(version({distribution!r}))"
    )
    try:
        result = subprocess.run(
            [python, "-c", code],
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    if result.returncode != 0:
        return False, (result.stderr or result.stdout).strip()
    return True, result.stdout.strip()


PARITY_ORACLE = (
    ROOT
    / "components/skills/manage-agent-workflows/releases/1.1.0-rc.2/source/scripts/workflow_registry.py"
)


def _adapter_path_inventory() -> dict[str, dict[str, Path]]:
    home = Path.home()
    return {
        "pi": {
            "stable_root": home / ".pi/agent/agent-workflows",
            "governance_skill": home / ".pi/agent/skills/manage-agent-workflows",
            "policy": home / ".pi/agent/AGENTS.md",
        },
        "codex": {
            "stable_root": home / ".codex/agent-workflows",
            "governance_skill": home / ".codex/skills/manage-agent-workflows",
            "policy": home / ".codex/AGENTS.md",
        },
    }


def _literal_link(path: Path) -> dict[str, Any]:
    value: dict[str, Any] = {"path": str(path), "exists": path_exists_for_verify(path)}
    if path.is_symlink():
        value["literal_target"] = os.readlink(path)
        value["resolved_target"] = str(path.resolve())
    else:
        value["literal_target"] = None
        value["resolved_target"] = str(path.resolve()) if path.exists() else None
    return value


def path_exists_for_verify(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _policy_semantics(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    lowered = text.lower()
    return {
        "canonical_project_declaration": ".codex/workflows.yaml" in text,
        "canonical_namespace_not_split": (
            ".pi/workflows.yaml" not in text
            or bool(
                re.search(r"不得[^\n。.]*\.pi/workflows\.yaml", text)
                or re.search(r"(?:must not|do not)[^\n.]*\.pi/workflows\.yaml", lowered)
            )
        ),
        "global_discovery_is_not_adoption": (
            "全局可发现不等于项目采用" in text
            or "不构成采用" in text
            or "global availability is not project adoption" in lowered
            or "global availability" in lowered and "adoption" in lowered
        ),
        "immutable_release_bytes": (
            "已锁定 release" in lowered and ("不可" in text or "不得" in text)
        ) or ("locked release" in lowered and "not" in lowered),
        "shared_registry_and_locks": "registry" in lowered and "workflow lock" in lowered,
        "promotion_requires_confirmation": (
            "经用户确认" in text
            or "user confirmation" in lowered
            or "user approval" in lowered
        ),
    }


def _parity_oracle_run(
    adapter_root: Path,
    command: str,
    arguments: list[str],
) -> tuple[int, Any, str]:
    oracle_command = [sys.executable, str(PARITY_ORACLE), "--root", str(adapter_root), command, *arguments]
    if command in {"list", "audit-project"}:
        oracle_command.append("--json")
    result = subprocess.run(
        oracle_command,
        text=True,
        capture_output=True,
        check=False,
    )
    try:
        document: Any = json.loads(result.stdout)
    except json.JSONDecodeError:
        document = None
    return result.returncode, document, result.stdout + result.stderr


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _run_parity(args: argparse.Namespace) -> int:
    report = Report()
    adapters = _adapter_path_inventory()
    try:
        registry = read_manifest(ROOT / "registry.yaml")
        pin = registry.get("global_components", {}).get("skill.manage-agent-workflows")
        report.add(isinstance(pin, str) and bool(pin), "parity.registry.global_pin", str(pin))
    except Exception as exc:
        registry = {}
        pin = None
        report.add(False, "parity.registry.global_pin", str(exc))

    component_entry = None
    expected_skill = None
    if isinstance(registry, dict):
        for item in registry.get("components", []):
            if item.get("id") == "skill.manage-agent-workflows":
                component_entry = item
                break
    if isinstance(component_entry, dict) and isinstance(pin, str):
        try:
            descriptor = read_manifest(ROOT / component_entry["descriptor"])
            release = descriptor["releases"][pin]
            expected_skill = (ROOT / release["source_snapshot"]).resolve()
        except Exception as exc:
            report.add(False, "parity.registry.governance_release", str(exc))
    if expected_skill is not None:
        report.add(
            expected_skill.is_dir() and expected_skill.is_relative_to(ROOT.resolve()),
            "parity.registry.governance_release",
            str(expected_skill),
        )

    adapter_roots: dict[str, Path] = {}
    resolved_roots: dict[str, Path] = {}
    policy_values: dict[str, dict[str, Any]] = {}
    for name, values in adapters.items():
        stable = _literal_link(values["stable_root"])
        adapter_roots[name] = values["stable_root"]
        resolved = values["stable_root"].resolve()
        resolved_roots[name] = resolved
        root_ok = stable["exists"] and values["stable_root"].is_symlink() and resolved == ROOT.resolve()
        report.add(root_ok, f"parity.stable_root.{name}", json.dumps(stable, sort_keys=True))
        skill = _literal_link(values["governance_skill"])
        skill_ok = (
            skill["exists"]
            and values["governance_skill"].is_symlink()
            and expected_skill is not None
            and values["governance_skill"].resolve() == expected_skill
        )
        report.add(skill_ok, f"parity.governance_skill.{name}", json.dumps(skill, sort_keys=True))
        try:
            policy_values[name] = _policy_semantics(values["policy"])
            report.add(True, f"parity.policy.load.{name}", str(values["policy"]))
        except (OSError, UnicodeError) as exc:
            policy_values[name] = {}
            report.add(False, f"parity.policy.load.{name}", str(exc))

    if set(policy_values) == {"pi", "codex"}:
        report.add(
            policy_values["pi"] == policy_values["codex"],
            "parity.policy.semantics",
            _canonical_json(policy_values),
        )

    if not PARITY_ORACLE.is_file():
        report.add(False, "parity.oracle.helper", f"missing frozen rc.2 helper: {PARITY_ORACLE}")
    else:
        report.add(True, "parity.oracle.helper", str(PARITY_ORACLE))
        show_identity = "skill.manage-agent-workflows"
        if not isinstance(component_entry, dict):
            workflows = registry.get("workflows", []) if isinstance(registry, dict) else []
            show_identity = workflows[0]["id"] if workflows else show_identity
        projects = list(args.projects or [])
        if not projects:
            projects = [str(ROOT)]
        elif str(ROOT) not in projects:
            projects.insert(0, str(ROOT))
        for command, command_args, check_name in (
            ("list", [], "parity.oracle.list"),
            ("show", [show_identity], "parity.oracle.show"),
        ):
            observed: dict[str, tuple[int, Any]] = {}
            errors: list[str] = []
            for name, root_path in adapter_roots.items():
                code, document, detail = _parity_oracle_run(root_path, command, command_args)
                observed[name] = (code, document)
                if document is None:
                    errors.append(f"{name}: oracle output was not JSON: {detail}")
            equal = (
                not errors
                and observed.get("pi") is not None
                and observed.get("codex") is not None
                and observed["pi"][0] == observed["codex"][0]
                and _canonical_json(observed["pi"][1]) == _canonical_json(observed["codex"][1])
            )
            report.add(equal, check_name, "equal semantic JSON" if equal else "; ".join(errors) or _canonical_json(observed))
        for project in projects:
            observed = {}
            errors = []
            for name, root_path in adapter_roots.items():
                code, document, detail = _parity_oracle_run(
                    root_path, "audit-project", [project]
                )
                observed[name] = (code, document)
                if document is None:
                    errors.append(f"{name}: oracle output was not JSON: {detail}")
            equal = (
                not errors
                and observed.get("pi") is not None
                and observed.get("codex") is not None
                and observed["pi"][0] == observed["codex"][0]
                and _canonical_json(observed["pi"][1]) == _canonical_json(observed["codex"][1])
            )
            default_not_declared = project == str(ROOT)
            if default_not_declared:
                equal = equal and observed.get("pi", (None, {}))[0] == 2 and all(
                    isinstance(observed.get(name, (None, {}))[1], dict)
                    and observed[name][1].get("status") == "NOT_DECLARED"
                    for name in ("pi", "codex")
                )
            report.add(
                equal,
                "parity.oracle.audit" if default_not_declared else f"parity.oracle.audit.{project}",
                "equal semantic JSON" if equal else "; ".join(errors) or _canonical_json(observed),
            )

    output = {
        "status": "PASS" if report.ok else "FAIL",
        "mode": "parity",
        "oracle": str(PARITY_ORACLE),
        "adapters": {
            name: {
                "stable_root": _literal_link(values["stable_root"]),
                "governance_skill": _literal_link(values["governance_skill"]),
            }
            for name, values in adapters.items()
        },
        "checks": report.checks,
    }
    if args.json:
        print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    else:
        for item in report.checks:
            print(f"{item['status']:4} {item['check']}: {item['detail']}")
        print(f"RESULT {output['status']} ({len(report.checks)} checks)")
    return 0 if report.ok else 1


DOWNLOAD_APPROVAL_BYTES = 500 * 1024 * 1024


def _download_local_stat(path: Path, kind: str) -> dict[str, Any]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        return {
            "exists": False,
            "kind_matches": False,
            "error": f"{type(exc).__name__}: {exc}",
            "bytes": None,
        }
    is_file = stat.S_ISREG(metadata.st_mode)
    is_tree = stat.S_ISDIR(metadata.st_mode)
    kind_matches = is_file if kind == "file" else is_tree if kind == "tree" else False
    return {
        "exists": True,
        "kind_matches": kind_matches,
        "actual_kind": "file" if is_file else "tree" if is_tree else "other",
        "bytes": metadata.st_size,
        "mode": f"{stat.S_IMODE(metadata.st_mode):04o}",
        "mtime_ns": metadata.st_mtime_ns,
    }


def _download_fail(args: argparse.Namespace, message: str) -> int:
    output = {
        "status": "FAIL",
        "mode": "download-plan",
        "workflow": args.workflow,
        "workflow_release": args.release,
        "planning_only": True,
        "downloads_started": False,
        "payload_content_inspected": False,
        "errors": [message],
        "engineering_metadata_boundary": (
            "工程 metadata 非科学真值；计划不构成 Artifact 内容或科学结果验收。"
        ),
    }
    if args.json:
        print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    else:
        print(f"FAIL: {message}")
    return 1


def _run_download_plan(args: argparse.Namespace) -> int:
    if args.projects:
        return _download_fail(args, "--download-plan does not accept --project")
    try:
        registry = read_manifest(ROOT / "registry.yaml")
    except Exception as exc:
        return _download_fail(args, f"Registry load failed: {exc}")
    workflows = {
        entry.get("id"): entry
        for entry in registry.get("workflows", [])
        if isinstance(entry, dict) and isinstance(entry.get("id"), str)
    }
    workflow_entry = workflows.get(args.workflow)
    if workflow_entry is None:
        return _download_fail(args, f"unknown Workflow: {args.workflow}")
    catalog = workflow_entry.get("releases")
    if not isinstance(catalog, dict):
        return _download_fail(args, f"Workflow release catalog is missing: {args.workflow}")
    selected_entry: dict[str, Any] = workflow_entry
    selected_release = args.release
    if selected_release is None:
        try:
            current_descriptor = read_manifest(ROOT / workflow_entry["descriptor"])
            selected_release = current_descriptor.get("release")
        except Exception as exc:
            return _download_fail(args, f"Workflow descriptor load failed: {exc}")
    if not isinstance(selected_release, str) or selected_release not in catalog:
        return _download_fail(
            args, f"unknown Workflow Release: {args.workflow}@{selected_release}"
        )
    selected_entry = catalog[selected_release]
    if not isinstance(selected_entry, dict):
        return _download_fail(args, f"invalid Workflow Release catalog entry: {selected_release}")
    try:
        descriptor_path = ROOT / selected_entry["descriptor"]
        lock_path = ROOT / selected_entry["lock"]
        descriptor_hash = file_sha256(descriptor_path)
        lock_hash = file_sha256(lock_path)
    except (KeyError, OSError, TypeError) as exc:
        return _download_fail(args, f"Workflow Release files are unavailable: {exc}")
    if descriptor_hash != selected_entry.get("descriptor_sha256"):
        return _download_fail(args, f"Workflow descriptor digest mismatch: {args.workflow}@{selected_release}")
    if lock_hash != selected_entry.get("lock_sha256"):
        return _download_fail(args, f"Workflow lock digest mismatch: {args.workflow}@{selected_release}")
    try:
        workflow = read_manifest(descriptor_path)
        lock = read_manifest(lock_path)
    except Exception as exc:
        return _download_fail(args, f"Workflow metadata load failed: {exc}")
    if (
        workflow.get("id") != args.workflow
        or workflow.get("release") != selected_release
        or lock.get("workflow") != args.workflow
        or lock.get("workflow_release") != selected_release
    ):
        return _download_fail(args, f"Workflow/lock identity mismatch: {args.workflow}@{selected_release}")
    locked = lock.get("components")
    if not isinstance(locked, dict):
        return _download_fail(args, "Workflow lock components must be an object")
    selected_components = dict(locked)
    global_components = registry.get("global_components", {})
    if not isinstance(global_components, dict):
        return _download_fail(args, "Registry global_components must be an object")
    for component_id, release_id in global_components.items():
        if component_id in selected_components and selected_components[component_id] != release_id:
            return _download_fail(args, f"global component conflict: {component_id}")
        selected_components[component_id] = release_id
    components = {
        entry.get("id"): entry
        for entry in registry.get("components", [])
        if isinstance(entry, dict) and isinstance(entry.get("id"), str)
    }
    payloads: list[dict[str, Any]] = []
    total_declared_known = 0
    needs_download_known = 0
    unknown_count = 0
    pending_unknown_count = 0
    for component_id, release_id in sorted(selected_components.items()):
        component_entry = components.get(component_id)
        if component_entry is None:
            return _download_fail(args, f"unknown locked component: {component_id}@{release_id}")
        if component_entry.get("kind") != "artifact":
            continue
        try:
            component_descriptor = read_manifest(ROOT / component_entry["descriptor"])
            release = component_descriptor["releases"][release_id]
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return _download_fail(args, f"unknown Artifact Release: {component_id}@{release_id}: {exc}")
        if not isinstance(release, dict):
            return _download_fail(args, f"invalid Artifact Release: {component_id}@{release_id}")
        declared_payloads = release.get("payloads")
        if not isinstance(declared_payloads, list):
            return _download_fail(args, f"Artifact payload list is missing: {component_id}@{release_id}")
        for payload in declared_payloads:
            if not isinstance(payload, dict):
                return _download_fail(args, f"invalid Artifact payload: {component_id}@{release_id}")
            payload_id = payload.get("id")
            source = payload.get("source", payload.get("sources"))
            kind = payload.get("kind")
            path_value = payload.get("path")
            if not isinstance(payload_id, str) or not isinstance(kind, str) or not isinstance(path_value, str):
                return _download_fail(args, f"Artifact payload identity/path is invalid: {component_id}")
            local = _download_local_stat(Path(path_value), kind)
            present = bool(local.get("exists") and local.get("kind_matches"))
            declared_size = payload.get("file_bytes")
            if isinstance(declared_size, bool) or not isinstance(declared_size, int) or declared_size < 0:
                declared_size = None
                unknown_count += 1
                if not present:
                    pending_unknown_count += 1
            else:
                total_declared_known += declared_size
                if not present:
                    needs_download_known += declared_size
            digest_value = payload.get("sha256") or payload.get("tree_sha256")
            payloads.append(
                {
                    "component": component_id,
                    "release": release_id,
                    "id": payload_id,
                    "source": source,
                    "kind": kind,
                    "path": path_value,
                    "declared_size_bytes": declared_size,
                    "declared_size_status": "known" if declared_size is not None else "unknown",
                    "digest_metadata": {
                        "algorithm": "sha256",
                        "declared": digest_value,
                    }
                    if isinstance(digest_value, str)
                    else None,
                    "local_stat": local,
                    "present_by_stat": present,
                    "needs_download": not present,
                    "payload_content_inspected": False,
                }
            )
    approval_reasons: list[str] = []
    if pending_unknown_count:
        approval_reasons.append("pending download has unknown declared size")
    if needs_download_known > DOWNLOAD_APPROVAL_BYTES:
        approval_reasons.append("pending known download exceeds 500 MiB")
    summary = {
        "total_declared_known_bytes": total_declared_known,
        "needs_download_known_bytes": needs_download_known,
        "unknown_declared_size_count": unknown_count,
        "pending_unknown_declared_size_count": pending_unknown_count,
        "known_bytes_threshold": DOWNLOAD_APPROVAL_BYTES,
        "approval_required": bool(approval_reasons),
        "approval_reasons": approval_reasons,
    }
    output = {
        "status": "PASS",
        "mode": "download-plan",
        "workflow": args.workflow,
        "workflow_release": selected_release,
        "planning_only": True,
        "downloads_started": False,
        "payload_content_inspected": False,
        "payloads": payloads,
        "summary": summary,
        "engineering_metadata_boundary": (
            "工程 metadata 非科学真值；计划不构成 Artifact 内容或科学结果验收。"
        ),
    }
    if args.json:
        print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    else:
        print(f"DOWNLOAD_PLAN {args.workflow}@{selected_release}")
        for item in payloads:
            print(
                f"{item['component']}@{item['release']}:{item['id']} "
                f"{item['kind']} {item['path']} needs_download={item['needs_download']}"
            )
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        print("PLANNING_ONLY: no payload download/read/hash performed")
        print("BOUNDARY: engineering metadata is not scientific truth")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify an agent_workflows version lock without changing installed software."
    )
    parser.add_argument("--workflow", default="pdf-document-stack")
    parser.add_argument(
        "--release",
        help="Verify an exact Workflow Release from the Registry release catalog.",
    )
    verification_mode = parser.add_mutually_exclusive_group()
    verification_mode.add_argument(
        "--repository-only",
        action="store_true",
        help="Check manifests, locks, and source snapshots without requiring installed runtimes.",
    )
    verification_mode.add_argument(
        "--deep-artifacts",
        action="store_true",
        help="Read and hash every registered artifact payload (7.03 GB / 6.54 GiB for the current workflow).",
    )
    parser.add_argument("--json", action="store_true", help="Emit a single JSON report.")
    parser.add_argument("--parity", action="store_true", help="Compare Pi/Codex adapters using the frozen rc.2 oracle.")
    parser.add_argument(
        "--download-plan",
        action="store_true",
        help="Plan exact Artifact downloads without reading or hashing payload contents.",
    )
    parser.add_argument(
        "--project",
        dest="projects",
        action="append",
        help="Project directory to include in parity audit (repeatable).",
    )
    args = parser.parse_args()
    if args.parity:
        if args.download_plan or args.deep_artifacts:
            parser.error("--parity conflicts with --download-plan and --deep-artifacts")
        return _run_parity(args)
    if args.download_plan:
        if args.deep_artifacts:
            parser.error("--download-plan conflicts with --deep-artifacts")
        return _run_download_plan(args)

    report = Report()
    try:
        registry = read_manifest(ROOT / "registry.yaml")
    except Exception as exc:
        report.add(False, "registry.load", str(exc))
        registry = {"components": [], "workflows": []}

    report.add(
        registry.get("schema_version") == 1,
        "schema.registry",
        str(registry.get("schema_version")),
    )
    integrity_policy = registry.get("integrity_policy")
    policy_ok = (
        isinstance(integrity_policy, dict)
        and integrity_policy.get("digest_contract") == "awf-tree-v2"
        and isinstance(integrity_policy.get("required_release_attestations"), list)
        and integrity_policy.get("require_component_release_metadata") is True
    )
    report.add(
        policy_ok,
        "registry.integrity_policy",
        "awf-tree-v2" if policy_ok else "missing or invalid integrity policy",
    )
    required_release_attestations = set(
        integrity_policy.get("required_release_attestations", [])
        if isinstance(integrity_policy, dict)
        else []
    )

    components: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for entry in registry.get("components", []):
        component_id = entry.get("id", "<missing>")
        path = ROOT / entry.get("descriptor", "")
        try:
            descriptor = read_manifest(path)
            report.add(
                descriptor.get("schema_version") == 1,
                f"schema.component.{component_id}",
                str(descriptor.get("schema_version")),
            )
            ok = descriptor.get("id") == component_id and descriptor.get("kind") == entry.get("kind")
            report.add(ok, f"registry.component.{component_id}", str(path.relative_to(ROOT)))
            component_kind = entry.get("kind")
            report.add(
                component_kind in ALLOWED_COMPONENT_KINDS,
                f"schema.component_kind.{component_id}",
                str(component_kind),
            )
            for release_id, release_entry in descriptor.get("releases", {}).items():
                report.add(
                    release_id != "latest",
                    f"schema.release_id.{component_id}@{release_id}",
                    release_id,
                )
                release_status = release_entry.get("status") if isinstance(release_entry, dict) else None
                report.add(
                    release_status in ALLOWED_RELEASE_STATUSES,
                    f"schema.release_status.{component_id}@{release_id}",
                    str(release_status),
                )
            if component_id in components:
                report.add(False, f"registry.unique.{component_id}", "duplicate component id")
            components[component_id] = (entry, descriptor)
        except Exception as exc:
            report.add(False, f"registry.component.{component_id}", str(exc))

    registered_release_references = {
        f"{component_id}@{release_id}"
        for component_id, (_, descriptor) in components.items()
        for release_id in descriptor.get("releases", {})
    }
    report.add(
        required_release_attestations == registered_release_references,
        "registry.integrity_release_set",
        (
            f"required={len(required_release_attestations)} "
            f"registered={len(registered_release_references)} "
            f"missing={sorted(registered_release_references - required_release_attestations)} "
            f"extra={sorted(required_release_attestations - registered_release_references)}"
        ),
    )

    release_attestations: dict[str, dict[str, Any]] = {}
    for relative in registry.get("integrity_attestations", []):
        path = ROOT / relative
        try:
            attestation = read_manifest(path)
            report.add(
                attestation.get("schema_version") == 1,
                f"schema.attestation.{relative}",
                str(attestation.get("schema_version")),
            )
            contract_ok = attestation.get("digest_contract") == "awf-tree-v2"
            report.add(contract_ok, f"integrity.attestation_file.{relative}", str(path))
            for reference, entry in attestation.get("releases", {}).items():
                if reference in release_attestations:
                    report.add(False, f"integrity.unique.{reference}", f"duplicate in {relative}")
                elif isinstance(entry, dict):
                    release_attestations[reference] = entry
                else:
                    report.add(False, f"integrity.entry.{reference}", "entry is not an object")
        except Exception as exc:
            report.add(False, f"integrity.attestation_file.{relative}", str(exc))

    require_release_metadata = bool(
        isinstance(integrity_policy, dict)
        and integrity_policy.get("require_component_release_metadata")
    )
    for reference in sorted(required_release_attestations):
        integrity = release_attestations.get(reference)
        report.add(
            integrity is not None,
            f"integrity.required.{reference}",
            "attested" if integrity is not None else "missing required attestation",
        )
        if integrity is None:
            continue
        try:
            component_id, release_id = split_reference(reference)
            registered = components.get(component_id)
            if registered is None:
                raise ValueError(f"attested component is not registered: {component_id}")
            component_entry, component_descriptor = registered
            release_metadata = component_descriptor.get("releases", {}).get(release_id)
            if release_metadata is None:
                raise ValueError(f"attested Release is not registered: {reference}")
            if require_release_metadata:
                descriptor_path = integrity.get("descriptor")
                report.add(
                    descriptor_path == component_entry.get("descriptor"),
                    f"integrity.descriptor.{reference}",
                    f"expected={component_entry.get('descriptor')} attested={descriptor_path}",
                )
                expected_metadata_hash = integrity.get("metadata_sha256")
                actual_metadata_hash = canonical_object_sha256(release_metadata)
                report.add(
                    actual_metadata_hash == expected_metadata_hash,
                    f"integrity.metadata.{reference}",
                    f"expected={expected_metadata_hash} actual={actual_metadata_hash}",
                )
            source_snapshot = release_metadata.get("source_snapshot")
            expected_legacy_tree = release_metadata.get("tree_sha256")
            if source_snapshot or expected_legacy_tree:
                if not isinstance(source_snapshot, str) or not isinstance(expected_legacy_tree, str):
                    report.add(
                        False,
                        f"integrity.legacy_hash.{reference}",
                        "source_snapshot and tree_sha256 must be paired",
                    )
                else:
                    source_root = ROOT / integrity_relative_path(source_snapshot)
                    actual_legacy_tree = tree_sha256(source_root)
                    report.add(
                        actual_legacy_tree == expected_legacy_tree,
                        f"integrity.legacy_hash.{reference}",
                        f"expected={expected_legacy_tree} actual={actual_legacy_tree}",
                    )
            coverage = integrity.get("coverage")
            if coverage == "legacy-v1":
                report.add(True, f"integrity.coverage.{reference}", "legacy-v1")
                continue
            integrity_root = ROOT / integrity_relative_path(integrity["root"])
            integrity_paths = tree_v2_paths(
                integrity_root,
                coverage,
                integrity.get("paths"),
                integrity.get("exclude_paths"),
            )
            actual_integrity = tree_sha256_v2(
                integrity_root,
                coverage,
                integrity.get("paths"),
                integrity.get("exclude_paths"),
            )
            expected_integrity = integrity.get("tree_sha256")
            report.add(
                actual_integrity == expected_integrity,
                f"integrity.required_hash.{reference}",
                f"expected={expected_integrity} actual={actual_integrity}",
            )
            regular_bytes = sum(
                path.lstat().st_size
                for path in integrity_paths
                if not path.is_symlink() and path.is_file()
            )
            symlink_count = sum(path.is_symlink() for path in integrity_paths)
            for name, actual_value in (
                ("entry_count", len(integrity_paths)),
                ("regular_file_bytes", regular_bytes),
                ("symlink_count", symlink_count),
            ):
                expected_value = integrity.get(name)
                if expected_value is not None:
                    report.add(
                        actual_value == expected_value,
                        f"integrity.{name}.{reference}",
                        f"expected={expected_value} actual={actual_value}",
                    )
        except Exception as exc:
            report.add(False, f"integrity.required_validate.{reference}", str(exc))

    workflows: dict[str, dict[str, Any]] = {}
    for entry in registry.get("workflows", []):
        workflow_id = entry.get("id", "<missing>")
        if workflow_id in workflows:
            report.add(
                False,
                f"registry.unique_workflow.{workflow_id}",
                "duplicate workflow id",
            )
        else:
            workflows[workflow_id] = entry
        release_catalog = entry.get("releases")
        if isinstance(release_catalog, dict):
            for release_id, release_entry in release_catalog.items():
                report.add(
                    release_id != "latest",
                    f"schema.workflow_release_id.{workflow_id}@{release_id}",
                    release_id,
                )
                release_status = release_entry.get("status") if isinstance(release_entry, dict) else None
                report.add(
                    release_status in ALLOWED_WORKFLOW_STATUSES,
                    f"schema.workflow_release_status.{workflow_id}@{release_id}",
                    str(release_status),
                )
    workflow_entry = workflows.get(args.workflow)
    if not workflow_entry:
        report.add(False, "workflow.select", f"not registered: {args.workflow}")
        workflow = {}
        lock = {"components": {}}
    else:
        try:
            selected_workflow_entry = workflow_entry
            if args.release:
                selected_workflow_entry = workflow_entry.get("releases", {}).get(args.release)
                report.add(
                    selected_workflow_entry is not None,
                    "workflow.select_release",
                    f"{args.workflow}@{args.release}",
                )
                if selected_workflow_entry is None:
                    raise ValueError(f"Workflow Release is not registered: {args.workflow}@{args.release}")
            workflow_path = ROOT / selected_workflow_entry["descriptor"]
            lock_path = ROOT / selected_workflow_entry["lock"]
            expected_descriptor_hash = selected_workflow_entry.get("descriptor_sha256")
            actual_descriptor_hash = file_sha256(workflow_path)
            report.add(
                isinstance(expected_descriptor_hash, str)
                and actual_descriptor_hash == expected_descriptor_hash,
                "workflow.descriptor_hash",
                f"expected={expected_descriptor_hash} actual={actual_descriptor_hash}",
            )
            expected_lock_hash = selected_workflow_entry.get("lock_sha256")
            actual_lock_hash = file_sha256(lock_path)
            report.add(
                isinstance(expected_lock_hash, str) and actual_lock_hash == expected_lock_hash,
                "workflow.lock_hash",
                f"expected={expected_lock_hash} actual={actual_lock_hash}",
            )
            workflow = read_manifest(workflow_path)
            lock = read_manifest(lock_path)
            report.add(
                workflow.get("schema_version") == 1,
                "schema.workflow",
                str(workflow.get("schema_version")),
            )
            report.add(
                lock.get("schema_version") == 1,
                "schema.workflow_lock",
                str(lock.get("schema_version")),
            )
            report.add(
                workflow.get("status") in ALLOWED_WORKFLOW_STATUSES,
                "schema.workflow_status",
                str(workflow.get("status")),
            )
            ok = workflow.get("id") == args.workflow and lock.get("workflow") == args.workflow
            report.add(ok, "workflow.identity", args.workflow)
            descriptor_release = workflow.get("release")
            lock_release = lock.get("workflow_release")
            report.add(
                descriptor_release == lock_release
                and descriptor_release is not None
                and (args.release is None or descriptor_release == args.release),
                "workflow.release",
                f"descriptor={descriptor_release} lock={lock_release}",
            )
        except Exception as exc:
            report.add(False, "workflow.load", str(exc))
            workflow = {}
            lock = {"components": {}}

    locked: dict[str, str] = lock.get("components", {})
    selected: dict[str, dict[str, Any]] = {}
    global_locked: dict[str, str] = registry.get("global_components", {})
    releases_to_check = dict(locked)
    for component_id, release_id in global_locked.items():
        if component_id in releases_to_check and releases_to_check[component_id] != release_id:
            report.add(False, f"global.conflict.{component_id}", f"workflow={releases_to_check[component_id]} global={release_id}")
        releases_to_check[component_id] = release_id
    for component_id, release_id in releases_to_check.items():
        report.add(
            release_id != "latest",
            f"schema.lock_release.{component_id}",
            release_id,
        )
        registered = components.get(component_id)
        if not registered:
            report.add(False, f"lock.{component_id}", "component is not registered")
            continue
        descriptor = registered[1]
        release = descriptor.get("releases", {}).get(release_id)
        report.add(release is not None, f"lock.{component_id}", release_id)
        if release is not None:
            release_status = release.get("status")
            if component_id in global_locked:
                report.add(
                    release_status == "validated",
                    f"global.status.{component_id}",
                    str(release_status),
                )
            if workflow.get("status") == "validated":
                report.add(
                    release_status != "candidate",
                    f"workflow.component_status.{component_id}",
                    str(release_status),
                )
            selected[component_id] = release

    for component_id, release_id in sorted(global_locked.items()):
        report.add(component_id in selected, f"global.component.{component_id}", release_id)

    referenced = set(workflow.get("entry_skills", []))
    for stage in workflow.get("stages", []):
        if "skill" in stage:
            referenced.add(stage["skill"])
        if "tool" in stage:
            referenced.add(stage["tool"])
        if "recipe" in stage:
            referenced.add(stage["recipe"])
        referenced.update(stage.get("tools", []))
    for component_id in sorted(referenced):
        present = component_id in locked
        report.add(
            present,
            f"workflow.reference.{component_id}",
            "present in version lock" if present else "missing from version lock",
        )

    for component_id, release in selected.items():
        references: list[str] = []
        if isinstance(release.get("recipe"), str):
            references.append(release["recipe"])
        references.extend(release.get("artifacts", []))
        references.extend(release.get("depends_on", []))
        for reference in references:
            try:
                target_id, target_release = split_reference(reference)
                ok = locked.get(target_id) == target_release
                report.add(ok, f"dependency.{component_id}->{target_id}", reference)
            except ValueError as exc:
                report.add(False, f"dependency.{component_id}", str(exc))

        snapshot = release.get("source_snapshot")
        expected_tree = release.get("tree_sha256")
        if snapshot:
            snapshot_path = ROOT / snapshot
            exists = snapshot_path.is_dir()
            report.add(exists, f"snapshot.exists.{component_id}", str(snapshot_path))
            if exists and expected_tree:
                actual = tree_sha256(snapshot_path)
                report.add(actual == expected_tree, f"snapshot.hash.{component_id}", actual)

        for key in (
            "instructions",
            "environment_lock",
            "conda_history",
            "conda_explicit_lock",
            "pypi_lock",
            "installation_record",
        ):
            relative = release.get(key)
            if relative:
                descriptor_path = Path(components[component_id][0]["descriptor"])
                candidate = ROOT / descriptor_path.parent / relative
                report.add(candidate.is_file(), f"recipe.file.{component_id}.{key}", str(candidate))

        release_id = releases_to_check[component_id]
        reference = f"{component_id}@{release_id}"
        integrity = release_attestations.get(reference)
        component_kind = components[component_id][0].get("kind")
        if component_kind == "recipe":
            report.add(
                integrity is not None,
                f"integrity.attestation.{component_id}",
                reference,
            )
        if integrity is not None and integrity.get("coverage") != "legacy-v1":
            try:
                integrity_root = ROOT / integrity_relative_path(integrity["root"])
                actual_integrity = tree_sha256_v2(
                    integrity_root,
                    integrity["coverage"],
                    integrity.get("paths"),
                    integrity.get("exclude_paths"),
                )
                expected_integrity = integrity["tree_sha256"]
                report.add(
                    actual_integrity == expected_integrity,
                    f"integrity.hash.{component_id}",
                    f"expected={expected_integrity} actual={actual_integrity} root={integrity_root}",
                )
            except Exception as exc:
                report.add(False, f"integrity.hash.{component_id}", str(exc))

        if args.repository_only:
            continue

        for required in release.get("required_paths", []):
            report.add(Path(required).exists(), f"installed.path.{component_id}", required)

        active_path = release.get("active_path")
        if active_path:
            active = Path(active_path)
            active_exists = active.exists()
            report.add(
                active_exists,
                f"installed.active_path.{component_id}",
                f"present: {active}" if active_exists else f"missing: {active}",
            )
            if active_exists:
                snapshot_path = ROOT / snapshot if snapshot else None
                source_file = snapshot_path / active.name if snapshot_path else None
                expected_kind = release.get("active_path_kind")
                if not expected_kind:
                    expected_kind = "file" if source_file and source_file.is_file() else "directory"
                actual_kind = "file" if active.is_file() else "directory" if active.is_dir() else "other"
                type_matches = actual_kind == expected_kind
                report.add(
                    type_matches,
                    f"installed.active_type.{component_id}",
                    f"expected={expected_kind} actual={actual_kind}",
                )
                if active.is_symlink():
                    expected_link_target = (
                        source_file
                        if expected_kind == "file"
                        else snapshot_path
                    )
                    actual_link_target = active.resolve()
                    expected_resolved_target = (
                        expected_link_target.resolve() if expected_link_target else None
                    )
                    report.add(
                        actual_link_target == expected_resolved_target,
                        f"installed.active_link.{component_id}",
                        f"expected={expected_resolved_target} actual={actual_link_target}",
                    )
                if type_matches and actual_kind == "directory" and expected_tree:
                    actual = tree_sha256(active)
                    report.add(actual == expected_tree, f"installed.hash.{component_id}", actual)
                    if integrity is not None and integrity.get("coverage") == "tree":
                        actual_installed_integrity = tree_sha256_v2(active, "tree")
                        expected_installed_integrity = integrity["tree_sha256"]
                        report.add(
                            actual_installed_integrity == expected_installed_integrity,
                            f"installed.integrity.{component_id}",
                            f"expected={expected_installed_integrity} actual={actual_installed_integrity}",
                        )
                elif type_matches and actual_kind == "file" and source_file:
                    if source_file.is_file():
                        expected_hash = file_sha256(source_file)
                        actual_hash = file_sha256(active)
                        report.add(
                            actual_hash == expected_hash,
                            f"installed.hash.{component_id}",
                            f"expected={expected_hash} actual={actual_hash} source={source_file}",
                        )
                    else:
                        report.add(
                            False,
                            f"installed.hash.{component_id}",
                            f"no matching snapshot file: {source_file}",
                        )

        for probe in release.get("version_probes", []):
            ok, actual = distribution_version(probe["python"], probe["distribution"])
            matches = ok and actual == probe["expected"]
            detail = f"expected={probe['expected']} actual={actual}"
            report.add(matches, f"installed.version.{component_id}.{probe['distribution']}", detail)

        for payload in release.get("payloads", []):
            path = artifact_payload_path(payload)
            exists = path.is_file() if payload["kind"] == "file" else path.is_dir()
            report.add(exists, f"artifact.exists.{component_id}.{payload['id']}", str(path))
            if not exists:
                continue
            actual_bytes = path.stat().st_size if payload["kind"] == "file" else tree_file_bytes(path)
            report.add(
                actual_bytes == payload["file_bytes"],
                f"artifact.size.{component_id}.{payload['id']}",
                f"expected={payload['file_bytes']} actual={actual_bytes}",
            )
            if args.deep_artifacts:
                actual_hash = file_sha256(path) if payload["kind"] == "file" else tree_sha256(path)
                expected_hash = payload.get("sha256") or payload.get("tree_sha256")
                report.add(
                    actual_hash == expected_hash,
                    f"artifact.hash.{component_id}.{payload['id']}",
                    actual_hash,
                )
            runtime_link = payload.get("runtime_link")
            if runtime_link:
                link = Path(runtime_link)
                ok = link.exists() and link.resolve() == path.resolve()
                report.add(ok, f"artifact.link.{component_id}.{payload['id']}", runtime_link)

    output = {
        "status": "PASS" if report.ok else "FAIL",
        "workflow": args.workflow,
        "workflow_release": workflow.get("release"),
        "repository_only": args.repository_only,
        "deep_artifacts": args.deep_artifacts,
        "checks": report.checks,
    }
    if args.json:
        print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    else:
        for item in report.checks:
            print(f"{item['status']:4} {item['check']}: {item['detail']}")
        print(f"RESULT {output['status']} ({len(report.checks)} checks)")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
