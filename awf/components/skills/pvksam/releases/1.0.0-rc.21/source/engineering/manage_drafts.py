#!/usr/bin/env python3
"""Create and inspect mutable AWF drafts outside the immutable repository."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
from datetime import datetime, timezone
import difflib
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
import uuid
from typing import Any, Iterator

import verify


ROOT = Path(os.environ.get("AWF_ROOT", str(Path(__file__).resolve().parents[5])))
REGISTRY = ROOT / "registry.yaml"
DRAFT_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
DRAFT_MANIFEST_FIELDS = {
    "schema_version",
    "kind",
    "component_id",
    "base_release",
    "base_release_status",
    "base_source",
    "base_tree_sha256",
    "base_tree_sha256_v2",
    "draft_name",
    "created_at",
    "mutable",
    "registered",
    "adoptable",
}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON manifest root is not an object: {path}")
    return value


def resolve_under(root: Path, relative: str, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} must be a non-empty relative path")
    value = Path(relative)
    if value.is_absolute():
        raise ValueError(f"{label} must be relative")
    resolved = (root / value).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} escapes repository") from exc
    return resolved


def parse_component_release(value: str) -> tuple[str, str]:
    if value.count("@") != 1:
        raise ValueError("--component must be COMPONENT_ID@RELEASE")
    component_id, release = value.split("@", 1)
    if not component_id or not release:
        raise ValueError("--component must be COMPONENT_ID@RELEASE")
    return component_id, release


def validate_registry_integrity_policy(registry: dict[str, Any]) -> set[str]:
    policy = registry.get("integrity_policy")
    repository = registry.get("repository")
    repository_root = repository.get("root") if isinstance(repository, dict) else None
    if not (
        registry.get("schema_version") == 1
        and isinstance(repository, dict)
        and repository.get("id") == "agent-workflows"
        and isinstance(repository_root, str)
        and Path(repository_root).is_absolute()
        and Path(repository_root).resolve() == ROOT.resolve()
        and isinstance(registry.get("global_components"), dict)
        and isinstance(registry.get("workflows"), list)
        and isinstance(registry.get("integrity_attestations"), list)
        and bool(registry.get("integrity_attestations"))
        and isinstance(policy, dict)
        and policy.get("digest_contract") == "awf-tree-v2"
        and policy.get("require_component_release_metadata") is True
        and isinstance(policy.get("required_release_attestations"), list)
    ):
        raise ValueError("Registry integrity policy is incomplete or invalid")

    components = registry.get("components")
    if not isinstance(components, list):
        raise ValueError("Registry integrity policy cannot resolve Component set")
    registered: set[str] = set()
    component_ids: set[str] = set()
    for entry in components:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            raise ValueError("Registry integrity policy has an invalid Component entry")
        component_id = entry["id"]
        if component_id in component_ids:
            raise ValueError("Registry integrity policy has duplicate Component ids")
        component_ids.add(component_id)
        descriptor_path = resolve_under(
            ROOT, entry.get("descriptor"), "component descriptor"
        )
        descriptor = load_json(descriptor_path)
        if not isinstance(descriptor, dict):
            raise ValueError("Registry integrity policy has an invalid Component descriptor")
        releases = descriptor.get("releases")
        if descriptor.get("id") != component_id or not isinstance(releases, dict):
            raise ValueError("Registry integrity policy has an invalid Component descriptor")
        registered.update(f"{component_id}@{release_id}" for release_id in releases)

    required_values = policy["required_release_attestations"]
    if not all(isinstance(value, str) for value in required_values):
        raise ValueError("Registry integrity policy required set is invalid")
    required = set(required_values)
    if len(required) != len(required_values) or required != registered:
        raise ValueError(
            "Registry integrity policy required set does not equal registered Releases"
        )
    return required


def validate_attestation_entry_schema(reference: str, entry: dict[str, Any]) -> None:
    coverage = entry.get("coverage")
    common = {"coverage", "descriptor", "metadata_sha256"}
    tree_common = common | {
        "entry_count",
        "regular_file_bytes",
        "root",
        "symlink_count",
        "tree_sha256",
    }
    if coverage == "legacy-v1":
        required = allowed = common
    elif coverage == "tree-excluding":
        required = allowed = tree_common | {"exclude_paths"}
    elif coverage == "tree":
        required = tree_common
        allowed = tree_common | {"descriptor_file_sha256", "legacy_tree_sha256"}
    else:
        raise ValueError(f"invalid attestation entry schema: {reference}")

    keys = set(entry)
    if not required.issubset(keys) or not keys.issubset(allowed):
        raise ValueError(f"invalid attestation entry schema: {reference}")
    digest_fields = ["metadata_sha256"]
    if coverage != "legacy-v1":
        digest_fields.append("tree_sha256")
    digest_fields.extend(
        field
        for field in ("descriptor_file_sha256", "legacy_tree_sha256")
        if field in entry
    )
    if any(
        not isinstance(entry.get(field), str)
        or re.fullmatch(r"[0-9a-f]{64}", entry[field]) is None
        for field in digest_fields
    ):
        raise ValueError(f"invalid attestation entry schema: {reference}")

    resolve_under(ROOT, entry.get("descriptor"), "attested descriptor")
    if coverage != "legacy-v1":
        resolve_under(ROOT, entry.get("root"), "attested root")
        for field in ("entry_count", "regular_file_bytes", "symlink_count"):
            value = entry.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"invalid attestation entry schema: {reference}")
    if coverage == "tree-excluding":
        exclude_paths = entry.get("exclude_paths")
        if not isinstance(exclude_paths, list) or not exclude_paths:
            raise ValueError(f"invalid attestation entry schema: {reference}")
        for relative in exclude_paths:
            if (
                not isinstance(relative, str)
                or not relative
                or Path(relative).is_absolute()
                or ".." in Path(relative).parts
            ):
                raise ValueError(f"invalid attestation entry schema: {reference}")


def attested_tree_v2(
    registry: dict[str, Any],
    reference: str,
    descriptor_path: Path,
    release: dict[str, Any],
    source: Path,
) -> str:
    required = validate_registry_integrity_policy(registry)
    if reference not in required:
        raise ValueError(f"Release is not in the required attestation set: {reference}")

    attestation_paths = registry.get("integrity_attestations")
    if not isinstance(attestation_paths, list) or not attestation_paths:
        raise ValueError("Registry integrity policy has no attestation files")
    release_attestations: dict[str, dict[str, Any]] = {}
    for relative in attestation_paths:
        attestation_path = resolve_under(ROOT, relative, "integrity attestation")
        attestation = load_json(attestation_path)
        if not (
            isinstance(attestation, dict)
            and attestation.get("schema_version") == 1
            and attestation.get("digest_contract") == "awf-tree-v2"
            and isinstance(attestation.get("releases"), dict)
        ):
            raise ValueError(f"invalid integrity attestation schema: {relative}")
        for attested_reference, entry in attestation["releases"].items():
            if not isinstance(entry, dict):
                raise ValueError(f"invalid integrity attestation entry: {attested_reference}")
            validate_attestation_entry_schema(attested_reference, entry)
            if attested_reference in release_attestations:
                raise ValueError(f"duplicate integrity attestation: {attested_reference}")
            release_attestations[attested_reference] = entry
    if set(release_attestations) != required:
        raise ValueError(
            "Registry integrity policy attestation entries do not equal required set"
        )

    integrity = release_attestations[reference]
    if integrity.get("coverage") != "tree":
        raise ValueError(f"mutable drafts require tree-v2 coverage: {reference}")
    if resolve_under(ROOT, integrity.get("root"), "attested root") != source:
        raise ValueError(f"attested root mismatch: {reference}")
    if (
        resolve_under(ROOT, integrity.get("descriptor"), "attested descriptor")
        != descriptor_path
    ):
        raise ValueError(f"attested descriptor path mismatch: {reference}")
    if integrity.get("descriptor_file_sha256") != verify.file_sha256(descriptor_path):
        raise ValueError(f"attested descriptor digest mismatch: {reference}")
    if integrity.get("metadata_sha256") != verify.canonical_object_sha256(release):
        raise ValueError(f"attested Release metadata digest mismatch: {reference}")
    if integrity.get("legacy_tree_sha256") != release.get("tree_sha256"):
        raise ValueError(f"attested legacy tree digest mismatch: {reference}")

    expected = integrity.get("tree_sha256")
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ValueError(f"attestation lacks a canonical tree-v2 digest: {reference}")
    actual = verify.tree_sha256_v2(source, "tree")
    if actual != expected:
        raise ValueError(
            f"attested tree-v2 digest mismatch for {reference}: "
            f"expected={expected} actual={actual}"
        )
    return actual


def component_release(
    component_id: str, release_id: str
) -> tuple[dict[str, Any], Path, str]:
    registry = load_json(REGISTRY)
    entries = [item for item in registry.get("components", []) if item.get("id") == component_id]
    if len(entries) != 1:
        raise ValueError(f"unknown Component id: {component_id}")
    if entries[0].get("kind") != "skill" or not component_id.startswith("skill."):
        raise ValueError("mutable drafts currently support only skill Components")
    descriptor_path = resolve_under(ROOT, entries[0].get("descriptor"), "component descriptor")
    descriptor = load_json(descriptor_path)
    if descriptor.get("id") != component_id or descriptor.get("kind") != "skill":
        raise ValueError(f"component descriptor identity mismatch: {component_id}")
    release = descriptor.get("releases", {}).get(release_id)
    if not isinstance(release, dict):
        raise ValueError(f"unknown Component Release: {component_id}@{release_id}")
    source = resolve_under(ROOT, release.get("source_snapshot"), "source snapshot")
    if not source.is_dir():
        raise ValueError(f"source snapshot is not a directory: {source}")
    expected = release.get("tree_sha256")
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"Component Release lacks canonical tree_sha256: {component_id}@{release_id}")
    actual = verify.tree_sha256(source)
    if actual != expected:
        raise ValueError(
            f"source digest mismatch for {component_id}@{release_id}: expected={expected} actual={actual}"
        )
    reference = f"{component_id}@{release_id}"
    base_v2 = attested_tree_v2(registry, reference, descriptor_path, release, source)
    return release, source, base_v2


def reject_symlink_components(value: Path, option: str) -> None:
    if ".." in value.parts:
        raise ValueError(f"{option} may not contain '..'")
    current = Path(value.anchor)
    for part in value.parts[1:]:
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{option} contains symlink path component: {current}")


def open_physical_directory(path: Path, option: str) -> tuple[int, tuple[int, int]]:
    reject_symlink_components(path, option)
    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    metadata = os.fstat(descriptor)
    identity = (metadata.st_dev, metadata.st_ino)
    try:
        verify_physical_directory(descriptor, path, identity, option)
    except Exception:
        os.close(descriptor)
        raise
    return descriptor, identity


def verify_physical_directory(
    descriptor: int,
    path: Path,
    identity: tuple[int, int],
    option: str,
) -> None:
    reject_symlink_components(path, option)
    try:
        lexical = os.stat(path, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise ValueError(f"{option} directory moved during operation: {path}") from exc
    opened = os.fstat(descriptor)
    if (lexical.st_dev, lexical.st_ino) != identity or (
        opened.st_dev,
        opened.st_ino,
    ) != identity:
        raise ValueError(f"{option} directory changed during operation: {path}")


def filesystem_identity(metadata: os.stat_result) -> tuple[int, int, int]:
    return metadata.st_dev, metadata.st_ino, stat.S_IFMT(metadata.st_mode)


def verify_published_identity(
    parent_fd: int,
    name: str,
    expected: tuple[int, int, int],
    label: str,
) -> None:
    try:
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise ValueError(f"published {label} identity changed") from exc
    if filesystem_identity(current) != expected:
        raise ValueError(f"published {label} identity changed")


def rename_directory_noreplace(
    directory_fd: int, source_name: str, target_name: str
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOSYS, "renameat2 is required for no-replace Draft publication")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        directory_fd,
        os.fsencode(source_name),
        directory_fd,
        os.fsencode(target_name),
        1,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise ValueError(f"draft already exists: {target_name}")
    raise OSError(error, os.strerror(error), target_name)


def validate_external_root(value: Path, option: str) -> Path:
    if not value.is_absolute():
        raise ValueError(f"{option} must be absolute")
    reject_symlink_components(value, option)
    resolved = value.resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        return resolved
    raise ValueError(f"{option} must be outside the immutable AWF repository")


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def create_draft(component_value: str, draft_root_value: Path, name: str) -> dict[str, Any]:
    component_id, release_id = parse_component_release(component_value)
    if DRAFT_NAME.fullmatch(name) is None:
        raise ValueError("--name must be lowercase kebab-case")
    draft_root = validate_external_root(draft_root_value, "--draft-root")
    release, source, base_v2 = component_release(component_id, release_id)
    draft_parent = draft_root / component_id
    draft = draft_parent / name
    reject_symlink_components(draft, "draft path")
    if draft.exists() or draft.is_symlink():
        raise ValueError(f"draft already exists: {draft}")

    draft_parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    reject_symlink_components(draft_parent, "draft path")
    parent_fd, parent_identity = open_physical_directory(draft_parent, "draft path")
    parent_anchor = Path(f"/proc/self/fd/{parent_fd}")
    staging_name = f".{name}.tmp-{uuid.uuid4().hex}"
    staging = parent_anchor / staging_name
    try:
        source_before = verify.tree_sha256_v2(source, "tree")
        if source_before != base_v2:
            raise ValueError(
                "source or staged copy changed during Draft creation"
            )
        staging.mkdir(mode=0o700)
        staged_source = staging / "source"
        shutil.copytree(source, staged_source, symlinks=True)
        staged_v2 = verify.tree_sha256_v2(staged_source, "tree")
        source_after = verify.tree_sha256_v2(source, "tree")
        if not (
            source_before == staged_v2 == source_after == base_v2
        ):
            raise ValueError(
                "source or staged copy changed during Draft creation"
            )
        manifest = {
            "schema_version": 1,
            "kind": "awf-mutable-draft",
            "component_id": component_id,
            "base_release": release_id,
            "base_release_status": release.get("status"),
            "base_source": source.relative_to(ROOT).as_posix(),
            "base_tree_sha256": release["tree_sha256"],
            "base_tree_sha256_v2": base_v2,
            "draft_name": name,
            "created_at": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
            "mutable": True,
            "registered": False,
            "adoptable": False,
        }
        manifest_path = staging / ".awf-draft.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.chmod(manifest_path, 0o600)
        fsync_directory(staging)
        published_identity = filesystem_identity(staging.lstat())
        rename_directory_noreplace(parent_fd, staging_name, name)
        os.fsync(parent_fd)
        verify_published_identity(parent_fd, name, published_identity, "Draft")
        verify_physical_directory(parent_fd, draft_parent, parent_identity, "draft path")
        reject_symlink_components(draft, "draft path")
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        os.close(parent_fd)

    return {
        "status": "PASS",
        "action": "create",
        "draft": str(draft),
        "component": f"{component_id}@{release_id}",
        "registered": False,
        "adoptable": False,
    }


def load_draft(value: Path) -> tuple[Path, Path, dict[str, Any], Path]:
    if value.is_symlink():
        raise ValueError("draft root must be a physical directory, not a symlink")
    draft = validate_external_root(value, "--draft")
    manifest_path = draft / ".awf-draft.json"
    source = draft / "source"
    if source.is_symlink():
        raise ValueError("draft source must be a physical directory, not a symlink")
    if manifest_path.is_symlink():
        raise ValueError("draft manifest must be a physical file, not a symlink")
    if not draft.is_dir() or not source.is_dir() or not manifest_path.is_file():
        raise ValueError(f"not an AWF mutable draft: {draft}")
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict) or set(manifest) != DRAFT_MANIFEST_FIELDS:
        raise ValueError("draft manifest fields do not match schema 1")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "awf-mutable-draft"
        or manifest.get("mutable") is not True
        or manifest.get("registered") is not False
        or manifest.get("adoptable") is not False
    ):
        raise ValueError("draft manifest lifecycle contract is invalid")
    draft_name = manifest.get("draft_name")
    if not isinstance(draft_name, str) or DRAFT_NAME.fullmatch(draft_name) is None:
        raise ValueError("draft manifest name is invalid")
    if draft.name != draft_name:
        raise ValueError("draft directory name does not match manifest")
    component_id = manifest.get("component_id")
    release_id = manifest.get("base_release")
    if not isinstance(component_id, str) or not isinstance(release_id, str):
        raise ValueError("draft manifest base identity is invalid")
    release, base_source, base_v2 = component_release(component_id, release_id)
    if manifest.get("base_release_status") != release.get("status"):
        raise ValueError("draft manifest base release status mismatch")
    if manifest.get("base_source") != base_source.relative_to(ROOT).as_posix():
        raise ValueError("draft manifest base source mismatch")
    if manifest.get("base_tree_sha256") != release.get("tree_sha256"):
        raise ValueError("draft manifest base tree digest mismatch")
    if manifest.get("base_tree_sha256_v2") != base_v2:
        raise ValueError("immutable base tree-v2 digest changed")
    return draft, source, manifest, base_source


def tree_inventory(root: Path) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for path in verify.tree_v2_paths(root, "tree"):
        relative = path.relative_to(root).as_posix()
        metadata = path.lstat()
        record: dict[str, Any] = {"mode": f"{stat.S_IMODE(metadata.st_mode):04o}"}
        if path.is_symlink():
            record.update({"kind": "symlink", "target": os.readlink(path)})
        elif path.is_file():
            record.update(
                {
                    "kind": "file",
                    "bytes": metadata.st_size,
                    "sha256": verify.file_sha256(path),
                }
            )
        elif path.is_dir():
            record["kind"] = "directory"
        else:
            record["kind"] = "other"
        inventory[relative] = record
    return inventory


def compare_trees(base: Path, draft: Path) -> dict[str, list[str]]:
    base_inventory = tree_inventory(base)
    draft_inventory = tree_inventory(draft)
    base_paths = set(base_inventory)
    draft_paths = set(draft_inventory)
    added = sorted(draft_paths - base_paths)
    deleted = sorted(base_paths - draft_paths)
    modified: list[str] = []
    type_changed: list[str] = []
    for relative in sorted(base_paths & draft_paths):
        before = base_inventory[relative]
        after = draft_inventory[relative]
        if before.get("kind") != after.get("kind"):
            type_changed.append(relative)
        elif before != after:
            modified.append(relative)
    return {
        "added": added,
        "modified": modified,
        "deleted": deleted,
        "type_changed": type_changed,
    }


def change_details(
    base: Path, draft: Path, changes: dict[str, list[str]]
) -> dict[str, dict[str, dict[str, Any] | None]]:
    base_inventory = tree_inventory(base)
    draft_inventory = tree_inventory(draft)
    candidates = sorted({relative for values in changes.values() for relative in values})
    return {
        relative: {
            "before": base_inventory.get(relative),
            "after": draft_inventory.get(relative),
        }
        for relative in candidates
    }


@contextmanager
def consistent_tree_snapshots(
    base_source: Path,
    draft_source: Path,
    expected_base_v2: str,
    operation: str,
) -> Iterator[tuple[Path, Path, str]]:
    if operation not in {"status", "proposal"}:
        raise ValueError(f"unsupported snapshot operation: {operation}")
    with tempfile.TemporaryDirectory(prefix=f"awf-{operation}-snapshot-") as temporary:
        snapshot_root = Path(temporary)
        base_snapshot = snapshot_root / "base"
        draft_snapshot = snapshot_root / "draft"
        base_before = verify.tree_sha256_v2(base_source, "tree")
        draft_before = verify.tree_sha256_v2(draft_source, "tree")
        shutil.copytree(base_source, base_snapshot, symlinks=True)
        shutil.copytree(draft_source, draft_snapshot, symlinks=True)
        base_after = verify.tree_sha256_v2(base_source, "tree")
        draft_after = verify.tree_sha256_v2(draft_source, "tree")
        base_snapshot_v2 = verify.tree_sha256_v2(base_snapshot, "tree")
        draft_snapshot_v2 = verify.tree_sha256_v2(draft_snapshot, "tree")
        if not (
            base_before
            == base_after
            == base_snapshot_v2
            == expected_base_v2
        ):
            raise ValueError(
                f"immutable base changed while {operation} snapshot was created"
            )
        if not (draft_before == draft_after == draft_snapshot_v2):
            raise ValueError(f"draft changed while {operation} snapshot was created")
        yield base_snapshot, draft_snapshot, draft_snapshot_v2


def draft_status(value: Path) -> dict[str, Any]:
    draft, source, manifest, base_source = load_draft(value)
    with consistent_tree_snapshots(
        base_source,
        source,
        manifest["base_tree_sha256_v2"],
        "status",
    ) as (base_snapshot, draft_snapshot, draft_snapshot_v2):
        changes = compare_trees(base_snapshot, draft_snapshot)
        return {
            "status": "PASS",
            "action": "status",
            "draft": str(draft),
            "component": f"{manifest['component_id']}@{manifest['base_release']}",
            "base_integrity": "PASS",
            "base_tree_sha256_v2": manifest["base_tree_sha256_v2"],
            "draft_tree_sha256_v2": draft_snapshot_v2,
            "snapshot_consistent": True,
            "changed": any(changes.values()),
            "changes": changes,
            "registered": False,
            "adoptable": False,
        }


def utf8_text(path: Path | None) -> str | None:
    if path is None or not path.is_file() or path.is_symlink():
        return "" if path is None else None
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def proposal_diffs(
    base_source: Path, draft_source: Path, changes: dict[str, list[str]]
) -> tuple[dict[str, str], list[str]]:
    text_diffs: dict[str, str] = {}
    non_text_changes: list[str] = []
    candidates = sorted(
        set(
            changes["added"]
            + changes["modified"]
            + changes["deleted"]
            + changes["type_changed"]
        )
    )
    for relative in candidates:
        before_path = base_source / relative
        after_path = draft_source / relative
        before_exists = before_path.exists() or before_path.is_symlink()
        after_exists = after_path.exists() or after_path.is_symlink()
        before = utf8_text(before_path if before_exists else None)
        after = utf8_text(after_path if after_exists else None)
        if before is None or after is None:
            non_text_changes.append(relative)
            continue
        diff = "".join(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=f"a/{relative}",
                tofile=f"b/{relative}",
            )
        )
        if diff:
            text_diffs[relative] = diff
        else:
            non_text_changes.append(relative)
    return text_diffs, sorted(non_text_changes)


def atomic_create_text(path: Path, content: str) -> None:
    reject_symlink_components(path, "--output")
    if path.exists() or path.is_symlink():
        raise ValueError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    reject_symlink_components(path.parent, "--output")
    parent_fd, parent_identity = open_physical_directory(path.parent, "--output")
    staging_name = f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        descriptor = os.open(
            staging_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=parent_fd,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            published_identity = filesystem_identity(os.fstat(handle.fileno()))
        try:
            os.link(
                staging_name,
                path.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise ValueError(f"output already exists: {path}") from exc
        os.fsync(parent_fd)
        verify_published_identity(
            parent_fd, path.name, published_identity, "proposal"
        )
        verify_physical_directory(parent_fd, path.parent, parent_identity, "--output")
        reject_symlink_components(path, "--output")
    except Exception:
        raise
    finally:
        try:
            os.unlink(staging_name, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        os.close(parent_fd)


def create_proposal(
    draft_value: Path,
    output_value: Path,
    summary: str,
    source_session: str | None,
) -> dict[str, Any]:
    summary = summary.strip()
    if not summary or len(summary) > 2000:
        raise ValueError("--summary must contain 1-2000 non-whitespace characters")
    if source_session is not None:
        source_session = source_session.strip()
        if not source_session or len(source_session) > 500:
            raise ValueError("--source-session must contain 1-500 characters")
    output = validate_external_root(output_value, "--output")
    if output.suffix != ".json":
        raise ValueError("--output must name a .json file")
    draft, source, manifest, base_source = load_draft(draft_value)
    expected_base_v2 = manifest["base_tree_sha256_v2"]
    with consistent_tree_snapshots(
        base_source,
        source,
        expected_base_v2,
        "proposal",
    ) as (base_snapshot, draft_snapshot, draft_snapshot_v2):
        changes = compare_trees(base_snapshot, draft_snapshot)
        if not any(changes.values()):
            raise ValueError("draft is unchanged; no proposal can be created")
        details = change_details(base_snapshot, draft_snapshot, changes)
        text_diffs, non_text_changes = proposal_diffs(
            base_snapshot, draft_snapshot, changes
        )
        identity = {
            "component_id": manifest["component_id"],
            "base_release": manifest["base_release"],
            "base_tree_sha256_v2": expected_base_v2,
            "draft_tree_sha256_v2": draft_snapshot_v2,
            "summary": summary,
            "source_session": source_session,
            "changes": changes,
            "change_details": details,
        }
        proposal = {
            "schema_version": 1,
            "kind": "awf-draft-proposal",
            "proposal_id": verify.canonical_object_sha256(identity),
            **identity,
            "draft_name": manifest["draft_name"],
            "draft_path": str(draft),
            "created_at": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
            "snapshot_consistent": True,
            "text_diffs": text_diffs,
            "non_text_changes": non_text_changes,
            "proposal_only": True,
            "registered": False,
            "adoptable": False,
        }
        payload = json.dumps(proposal, indent=2, sort_keys=True) + "\n"
        atomic_create_text(output, payload)
    return {
        "status": "PASS",
        "action": "proposal",
        "draft": str(draft),
        "output": str(output),
        "proposal_id": proposal["proposal_id"],
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "registered": False,
        "adoptable": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage mutable AWF drafts without changing Registry or immutable Releases."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create", help="Create a physical mutable draft from an exact Release.")
    create.add_argument("--component", required=True, help="Exact COMPONENT_ID@RELEASE base.")
    create.add_argument("--draft-root", required=True, type=Path)
    create.add_argument("--name", required=True)
    create.add_argument("--json", action="store_true")
    status = subparsers.add_parser("status", help="Compare a mutable draft with its immutable base.")
    status.add_argument("--draft", required=True, type=Path)
    status.add_argument("--json", action="store_true")
    proposal = subparsers.add_parser(
        "proposal", help="Export a reviewable proposal without creating a Release."
    )
    proposal.add_argument("--draft", required=True, type=Path)
    proposal.add_argument("--output", required=True, type=Path)
    proposal.add_argument("--summary", required=True)
    proposal.add_argument("--source-session")
    proposal.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "create":
            report = create_draft(args.component, args.draft_root, args.name)
        elif args.command == "status":
            report = draft_status(args.draft)
        elif args.command == "proposal":
            report = create_proposal(
                args.draft, args.output, args.summary, args.source_session
            )
        else:
            parser.error(f"unsupported command: {args.command}")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        if getattr(args, "json", False):
            print(json.dumps({"status": "FAIL", "error": str(exc)}, sort_keys=True))
        else:
            print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    if getattr(args, "json", False):
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"PASS: {report['action']} mutable draft {report['draft']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
