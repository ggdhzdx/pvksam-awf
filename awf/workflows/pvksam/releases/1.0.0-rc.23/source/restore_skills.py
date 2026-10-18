#!/usr/bin/env python3
"""Install locked Skill snapshots as symlinks, with mandatory backup on conflict."""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(os.environ.get("AWF_ROOT", str(Path(__file__).resolve().parents[5])))
IGNORED_TREE_PARTS = {"__pycache__", ".pytest_cache"}
SKILL_COMPONENT_PATTERN = re.compile(r"skill\.[a-z0-9]+(?:-[a-z0-9]+)*")
SKILL_TARGET_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
JOURNAL_SCHEMA_VERSION = 1
JOURNAL_STATES = {"applying", "rolling-back", "rolled-back", "committed"}
OPERATION_PHASES = {
    "planned",
    "backup-intent",
    "backup-moved",
    "link-intent",
    "link-created",
    "backup-published",
}
BACKUP_MARKER_NAME = ".awf-restore-backup.json"
BACKUP_MARKER_FORMAT = "awf-skill-restore-backup-v1"
ACTIVE_ANCHOR_FORMAT = "awf-skill-restore-active-v1"
AT_FDCWD = -100
RENAME_NOREPLACE = 1
RENAME_EXCHANGE = 2


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file()
            and not path.is_symlink()
            and not any(part in IGNORED_TREE_PARTS for part in path.relative_to(root).parts)
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(f"{file_sha256(path)}  ./{relative}\n".encode("utf-8"))
    return digest.hexdigest()


def entry_sha256(root: Path) -> str:
    """Hash one directory entry without following symlinks."""
    if not path_exists(root):
        raise FileNotFoundError(root)
    digest = hashlib.sha256()

    def add(path: Path, relative: str) -> None:
        metadata = path.lstat()
        entry: dict[str, Any] = {
            "path": relative,
            "mode": f"{stat.S_IMODE(metadata.st_mode):04o}",
        }
        if stat.S_ISLNK(metadata.st_mode):
            entry.update({"kind": "symlink", "target": os.readlink(path)})
        elif stat.S_ISREG(metadata.st_mode):
            entry.update(
                {
                    "kind": "file",
                    "size": metadata.st_size,
                    "sha256": file_sha256(path),
                }
            )
        elif stat.S_ISDIR(metadata.st_mode):
            entry["kind"] = "directory"
        else:
            raise ValueError(f"unsupported target entry type: {path}")
        digest.update(
            (json.dumps(entry, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
        )
        if entry["kind"] == "directory":
            for child in sorted(path.iterdir(), key=lambda item: item.name):
                child_relative = child.name if relative == "." else f"{relative}/{child.name}"
                add(child, child_relative)

    add(root, ".")
    return digest.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def load_journal_safely(path: Path) -> dict[str, Any]:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        named = path.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_uid != os.geteuid()
            or opened.st_nlink != 1
            or stat.S_IMODE(opened.st_mode) != 0o600
            or opened.st_size > 1024 * 1024
            or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise ValueError(f"unsafe restore journal: {path}")
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            descriptor = -1
            value = json.load(handle, object_pairs_hook=reject_duplicate_keys)
        if not isinstance(value, dict):
            raise ValueError("restore journal root must be an object")
        return value
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def resolve_under(root: Path, value: str, label: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe {label}: {value}")
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes repository: {value}") from exc
    return resolved


def path_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def entry_has_identity(path: Path, operation: dict[str, Any]) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    return (metadata.st_dev, metadata.st_ino) == (
        operation.get("original_device"),
        operation.get("original_inode"),
    )


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def create_private_directory_noreplace(path: Path) -> os.stat_result:
    os.mkdir(path, 0o700)
    # A restrictive umask can only remove bits from 0700; restoring owner access
    # here never creates a group/other-writable window.
    os.chmod(path, 0o700)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        os.fchmod(descriptor, 0o700)
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        if (
            metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            raise ValueError(f"private directory invariant failed: {path}")
    finally:
        os.close(descriptor)
    fsync_directory(path.parent)
    return metadata


def fsync_entry(path: Path) -> None:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode):
        fsync_directory(path.parent)
        return
    if stat.S_ISREG(metadata.st_mode):
        flags = os.O_RDONLY | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return
    if stat.S_ISDIR(metadata.st_mode):
        for child in sorted(path.iterdir(), key=lambda item: item.name):
            fsync_entry(child)
        fsync_directory(path)
        return
    raise ValueError(f"unsupported target entry type: {path}")


def copy_entry_noreplace(source: Path, destination: Path) -> None:
    metadata = source.lstat()
    mode = stat.S_IMODE(metadata.st_mode)
    if stat.S_ISLNK(metadata.st_mode):
        destination.symlink_to(os.readlink(source))
        fsync_directory(destination.parent)
        return
    if stat.S_ISREG(metadata.st_mode):
        source_flags = os.O_RDONLY | os.O_CLOEXEC
        destination_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            source_flags |= os.O_NOFOLLOW
            destination_flags |= os.O_NOFOLLOW
        source_descriptor = os.open(source, source_flags)
        try:
            destination_descriptor = os.open(destination, destination_flags, 0o600)
            try:
                with os.fdopen(source_descriptor, "rb") as source_handle, os.fdopen(
                    destination_descriptor, "wb"
                ) as destination_handle:
                    source_descriptor = -1
                    destination_descriptor = -1
                    shutil.copyfileobj(source_handle, destination_handle, 1024 * 1024)
                    destination_handle.flush()
                    os.fchmod(destination_handle.fileno(), mode)
                    os.fsync(destination_handle.fileno())
            finally:
                if destination_descriptor >= 0:
                    os.close(destination_descriptor)
        finally:
            if source_descriptor >= 0:
                os.close(source_descriptor)
        fsync_directory(destination.parent)
        return
    if stat.S_ISDIR(metadata.st_mode):
        create_private_directory_noreplace(destination)
        for child in sorted(source.iterdir(), key=lambda item: item.name):
            copy_entry_noreplace(child, destination / child.name)
        os.chmod(destination, mode)
        fsync_directory(destination)
        fsync_directory(destination.parent)
        return
    raise ValueError(f"unsupported target entry type: {source}")


def remove_owned_entry(path: Path) -> None:
    metadata = path.lstat()
    if stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode):
        os.chmod(path, 0o700)
        for child in sorted(path.iterdir(), key=lambda item: item.name, reverse=True):
            remove_owned_entry(child)
        path.rmdir()
    else:
        path.unlink()
    fsync_directory(path.parent)


def serialized_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"


def write_private_json_noreplace(path: Path, value: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        payload = serialized_json(value).encode("utf-8")
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        rename_noreplace(temporary, path)
        maybe_crash("after-private-json-publish")
        fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()


def write_journal(path: Path, value: dict[str, Any], *, create: bool = False) -> None:
    serialized = serialized_json(value)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        if create:
            rename_noreplace(temporary, path)
            maybe_crash("after-journal-publish")
            fsync_directory(path.parent)
        else:
            existing = load_journal_safely(path)
            if existing.get("transaction_id") != value.get("transaction_id"):
                raise ValueError("restore journal transaction changed before update")
            rename_exchange(temporary, path)
            try:
                swapped = load_journal_safely(temporary)
                if swapped.get("transaction_id") != value.get("transaction_id"):
                    raise ValueError("restore journal transaction changed during update")
            except Exception:
                rename_exchange(temporary, path)
                fsync_directory(path.parent)
                raise
            maybe_crash("after-journal-exchange")
            fsync_directory(path.parent)
            temporary.unlink()
            fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def remove_journal(path: Path) -> None:
    path.unlink()
    maybe_crash("after-journal-unlink")
    fsync_directory(path.parent)


def expected_link(path: Path, literal_target: str) -> bool:
    return path.is_symlink() and os.readlink(path) == literal_target


def journal_path_for(target_root: Path) -> Path:
    return target_root.parent / f".{target_root.name}.awf-restore-journal.json"


def anchor_path_for(target_root: Path) -> Path:
    return target_root.parent / f".{target_root.name}.awf-restore-active.json"


def paths_overlap(left: Path, right: Path) -> bool:
    left = left.resolve()
    right = right.resolve()
    return left == right or left in right.parents or right in left.parents


def renameat2(left: Path, right: Path, flags: int) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    if rename is None:
        raise OSError("atomic Skill restore requires renameat2")
    rename.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    rename.restype = ctypes.c_int
    result = rename(
        AT_FDCWD,
        os.fsencode(left),
        AT_FDCWD,
        os.fsencode(right),
        flags,
    )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), f"{left} -> {right}")


def rename_exchange(left: Path, right: Path) -> None:
    renameat2(left, right, RENAME_EXCHANGE)


def rename_noreplace(left: Path, right: Path) -> None:
    renameat2(left, right, RENAME_NOREPLACE)


def maybe_crash(point: str) -> None:
    if (
        os.environ.get("AWF_RESTORE_TEST_MODE") == "1"
        and os.environ.get("AWF_RESTORE_TEST_CRASH_AT") == point
    ):
        os._exit(97)


def open_restore_lock(path: Path):
    flags = os.O_RDWR | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    created = False
    try:
        descriptor = os.open(path, flags | os.O_CREAT | os.O_EXCL, 0o600)
        created = True
    except FileExistsError:
        descriptor = os.open(path, flags)
    try:
        if created:
            os.fchmod(descriptor, 0o600)
            fsync_directory(path.parent)
        opened = os.fstat(descriptor)
        named = path.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_uid != os.geteuid()
            or opened.st_nlink != 1
            or stat.S_IMODE(opened.st_mode) != 0o600
            or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise ValueError(f"unsafe AWF restore lock: {path}")
        return os.fdopen(descriptor, "a+", encoding="utf-8")
    except Exception:
        os.close(descriptor)
        raise


def lock_path_matches(handle, path: Path) -> bool:
    try:
        opened = os.fstat(handle.fileno())
        named = path.lstat()
    except OSError:
        return False
    return (
        stat.S_ISREG(named.st_mode)
        and (opened.st_dev, opened.st_ino) == (named.st_dev, named.st_ino)
    )


def trusted_directory(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    return (
        stat.S_ISDIR(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and metadata.st_uid == os.geteuid()
        and stat.S_IMODE(metadata.st_mode) & 0o022 == 0
    )


def target_root_matches_journal(target_root: Path, journal: dict[str, Any]) -> bool:
    try:
        metadata = target_root.lstat()
    except OSError:
        return False
    return (
        trusted_directory(target_root)
        and (metadata.st_dev, metadata.st_ino)
        == (journal.get("target_root_device"), journal.get("target_root_inode"))
    )


def target_root_state_is_safe(target_root: Path, journal: dict[str, Any]) -> bool:
    if path_exists(target_root):
        return target_root_matches_journal(target_root, journal)
    return journal.get("target_root_device") is None or (
        journal.get("state") == "rolled-back"
        and not journal.get("target_root_existed")
    )


def backup_root_matches_journal(journal: dict[str, Any]) -> bool:
    backup_value = journal.get("backup_root")
    if backup_value is None:
        return True
    backup_root = Path(backup_value)
    try:
        metadata = backup_root.lstat()
    except OSError:
        return False
    return (
        trusted_directory(backup_root.parent)
        and trusted_directory(backup_root)
        and stat.S_IMODE(metadata.st_mode) == 0o700
        and (metadata.st_dev, metadata.st_ino)
        == (journal.get("backup_root_device"), journal.get("backup_root_inode"))
    )


def backup_root_state_is_safe(journal: dict[str, Any]) -> bool:
    backup_value = journal.get("backup_root")
    if backup_value is None:
        return True
    backup_root = Path(backup_value)
    if path_exists(backup_root):
        return backup_root_matches_journal(journal)
    return journal.get("backup_root_device") is None or journal.get("state") == "rolled-back"


def validate_journal(value: Any, target_root: Path) -> dict[str, Any]:
    journal_fields = {
        "schema_version",
        "transaction_id",
        "repository_root",
        "target_root",
        "target_root_existed",
        "target_root_device",
        "target_root_inode",
        "backup_root",
        "backup_root_created",
        "backup_root_device",
        "backup_root_inode",
        "backup_marker_sha256",
        "anchor_sha256",
        "state",
        "operations",
    }
    operation_fields = {
        "action",
        "source",
        "source_sha256",
        "target",
        "link_target",
        "backup_target",
        "original_sha256",
        "original_device",
        "original_inode",
        "quarantine_target",
        "phase",
    }
    if not isinstance(value, dict) or value.get("schema_version") != JOURNAL_SCHEMA_VERSION:
        raise ValueError("unsupported restore journal schema")
    if set(value) != journal_fields:
        raise ValueError("restore journal fields do not match schema")
    transaction_id = value.get("transaction_id")
    try:
        parsed_transaction_id = uuid.UUID(transaction_id)
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("invalid restore journal transaction id") from exc
    if str(parsed_transaction_id) != transaction_id:
        raise ValueError("restore journal transaction id is not canonical")
    if value.get("repository_root") != str(ROOT.resolve()):
        raise ValueError("restore journal repository root mismatch")
    if value.get("target_root") != str(target_root):
        raise ValueError("restore journal target root mismatch")
    if type(value.get("target_root_existed")) is not bool:
        raise ValueError("restore journal target_root_existed must be boolean")
    target_root_device = value.get("target_root_device")
    target_root_inode = value.get("target_root_inode")
    if (target_root_device is None) != (target_root_inode is None) or (
        target_root_device is not None
        and (type(target_root_device) is not int or type(target_root_inode) is not int)
    ):
        raise ValueError("restore journal target root identity is invalid")
    if value["target_root_existed"] and target_root_device is None:
        raise ValueError("restore journal lacks existing target root identity")
    if path_exists(target_root):
        target_metadata = target_root.lstat()
        if target_root_device is None or (
            target_metadata.st_dev,
            target_metadata.st_ino,
        ) != (target_root_device, target_root_inode):
            raise ValueError("restore journal target root identity changed")
    elif target_root_device is not None and not (
        value.get("state") == "rolled-back" and not value["target_root_existed"]
    ):
        raise ValueError("restore journal target root is missing")
    if type(value.get("backup_root_created")) is not bool:
        raise ValueError("restore journal backup_root_created must be boolean")
    if value.get("state") not in JOURNAL_STATES:
        raise ValueError("invalid restore journal state")
    anchor_sha256 = value.get("anchor_sha256")
    if not isinstance(anchor_sha256, str) or re.fullmatch(
        r"[0-9a-f]{64}", anchor_sha256
    ) is None:
        raise ValueError("restore journal active anchor digest is invalid")
    backup_value = value.get("backup_root")
    if backup_value is not None and not isinstance(backup_value, str):
        raise ValueError("restore journal backup root must be a path or null")
    backup_marker_sha256 = value.get("backup_marker_sha256")
    backup_root_device = value.get("backup_root_device")
    backup_root_inode = value.get("backup_root_inode")
    if (backup_root_device is None) != (backup_root_inode is None) or (
        backup_root_device is not None
        and (type(backup_root_device) is not int or type(backup_root_inode) is not int)
    ):
        raise ValueError("restore journal backup root identity is invalid")
    backup_root = Path(backup_value) if isinstance(backup_value, str) else None
    if backup_root is None:
        if (
            value["backup_root_created"]
            or backup_marker_sha256 is not None
            or backup_root_device is not None
        ):
            raise ValueError("restore journal marks a null backup root as created")
    else:
        if not isinstance(backup_marker_sha256, str) or re.fullmatch(
            r"[0-9a-f]{64}", backup_marker_sha256
        ) is None:
            raise ValueError("restore journal backup marker digest is missing")
        if (
            not backup_root.is_absolute()
            or backup_root != backup_root.resolve()
            or not value["backup_root_created"]
            or paths_overlap(backup_root, target_root)
            or paths_overlap(backup_root, ROOT)
            or not trusted_directory(backup_root.parent)
        ):
            raise ValueError("unsafe restore journal backup root")
        if path_exists(backup_root):
            if backup_root_device is None or not backup_root_matches_journal(value):
                raise ValueError("restore journal backup root identity changed")
        elif backup_root_device is not None and value["state"] != "rolled-back":
            raise ValueError("restore journal backup root is missing")
    operations = value.get("operations")
    if not isinstance(operations, list):
        raise ValueError("restore journal operations must be an array")
    if not operations:
        raise ValueError("restore journal operations must be non-empty")
    seen_targets: set[Path] = set()
    has_replacement = False
    for operation in operations:
        if not isinstance(operation, dict) or set(operation) != operation_fields:
            raise ValueError("restore journal operation fields do not match schema")
        action = operation.get("action")
        if action not in {"create", "replace-with-backup"}:
            raise ValueError("invalid restore journal action")
        if operation.get("phase") not in OPERATION_PHASES:
            raise ValueError("invalid restore journal operation phase")
        target = Path(operation.get("target", ""))
        source = Path(operation.get("source", ""))
        if (
            not target.is_absolute()
            or target != Path(os.path.normpath(str(target)))
            or target.parent != target_root
        ):
            raise ValueError("restore journal target escapes target root")
        if SKILL_TARGET_PATTERN.fullmatch(target.name) is None or target in seen_targets:
            raise ValueError("invalid or duplicate restore journal target")
        seen_targets.add(target)
        try:
            resolved_source = source.resolve()
            resolved_source.relative_to(ROOT.resolve())
        except ValueError as exc:
            raise ValueError("restore journal source escapes repository") from exc
        if not source.is_absolute() or source != resolved_source:
            raise ValueError("restore journal source is not canonical")
        literal_target = operation.get("link_target")
        if literal_target != os.path.relpath(source, start=target.parent):
            raise ValueError("restore journal link target mismatch")
        source_sha256 = operation.get("source_sha256")
        if not isinstance(source_sha256, str) or re.fullmatch(
            r"[0-9a-f]{64}", source_sha256
        ) is None:
            raise ValueError("restore journal operation lacks source digest")
        backup_target_value = operation.get("backup_target")
        original_sha256 = operation.get("original_sha256")
        original_device = operation.get("original_device")
        original_inode = operation.get("original_inode")
        quarantine_value = operation.get("quarantine_target")
        if action == "replace-with-backup":
            has_replacement = True
            if backup_root is None or not isinstance(backup_target_value, str):
                raise ValueError("restore journal replacement lacks backup path")
            backup_target = Path(backup_target_value)
            if (
                not backup_target.is_absolute()
                or backup_target != Path(os.path.normpath(str(backup_target)))
                or backup_target.parent != backup_root
                or backup_target.name != target.name
            ):
                raise ValueError("restore journal backup target escapes backup root")
            if not isinstance(original_sha256, str) or re.fullmatch(
                r"[0-9a-f]{64}", original_sha256
            ) is None:
                raise ValueError("restore journal replacement lacks original digest")
            if type(original_device) is not int or type(original_inode) is not int:
                raise ValueError("restore journal replacement lacks original identity")
        else:
            if (
                backup_target_value is not None
                or original_sha256 is not None
                or original_device is not None
                or original_inode is not None
            ):
                raise ValueError("restore journal create unexpectedly has backup evidence")
        expected_quarantine = target_root / (
            f".awf-restore-{transaction_id}-{target.name}"
        )
        if quarantine_value != str(expected_quarantine):
            raise ValueError("restore journal operation has invalid quarantine path")
    if has_replacement and backup_root is None:
        raise ValueError("restore journal replacements require a backup root")
    if value["state"] == "committed" and any(
        operation["phase"]
        != (
            "backup-published"
            if operation["action"] == "replace-with-backup"
            else "link-created"
        )
        for operation in operations
    ):
        raise ValueError("committed restore journal has incomplete operations")
    return value


def active_anchor_value(journal: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "format": ACTIVE_ANCHOR_FORMAT,
        "transaction_id": journal["transaction_id"],
        "repository_root": journal["repository_root"],
        "target_root": journal["target_root"],
    }


def validate_active_anchor(
    journal: dict[str, Any], anchor_path: Path
) -> tuple[bool, bool]:
    anchor_required = journal["state"] in {"applying", "rolling-back"} and (
        (not journal["target_root_existed"] and journal["target_root_device"] is not None)
        or any(
            operation["phase"] != "planned" for operation in journal["operations"]
        )
        or (
            isinstance(journal["backup_root"], str)
            and path_exists(Path(journal["backup_root"]))
        )
    )
    if not path_exists(anchor_path):
        return (not anchor_required), False
    expected = active_anchor_value(journal)
    expected_payload = serialized_json(expected).encode("utf-8")
    expected_sha256 = hashlib.sha256(expected_payload).hexdigest()
    if journal["anchor_sha256"] != expected_sha256:
        return False, True
    try:
        actual = load_journal_safely(anchor_path)
        actual_sha256 = file_sha256(anchor_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False, True
    return actual == expected and actual_sha256 == expected_sha256, True


def backup_marker_value(journal: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "format": BACKUP_MARKER_FORMAT,
        "transaction_id": journal["transaction_id"],
        "repository_root": journal["repository_root"],
        "target_root": journal["target_root"],
        "backup_root": journal["backup_root"],
    }


def validate_backup_marker(journal: dict[str, Any]) -> tuple[bool, bool]:
    backup_value = journal["backup_root"]
    if backup_value is None:
        return True, False
    backup_root = Path(backup_value)
    marker_path = backup_root / BACKUP_MARKER_NAME
    marker_required = journal["state"] != "rolled-back" and (
        journal["state"] == "committed"
        or any(operation["phase"] != "planned" for operation in journal["operations"])
    )
    if not path_exists(marker_path):
        if path_exists(backup_root) and journal["state"] != "rolled-back":
            identity_bound_empty_root = (
                journal["state"] == "applying"
                and all(
                    operation["phase"] == "planned"
                    for operation in journal["operations"]
                )
                and backup_root_matches_journal(journal)
                and not any(backup_root.iterdir())
            )
            return identity_bound_empty_root, False
        return (not marker_required), False
    if not backup_root.is_dir():
        return False, False
    expected = backup_marker_value(journal)
    expected_payload = serialized_json(expected).encode("utf-8")
    expected_sha256 = hashlib.sha256(expected_payload).hexdigest()
    if journal["backup_marker_sha256"] != expected_sha256:
        return False, True
    try:
        actual = load_journal_safely(marker_path)
        actual_sha256 = file_sha256(marker_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False, True
    return actual == expected and actual_sha256 == expected_sha256, True


def committed_invariant_failures(journal: dict[str, Any]) -> dict[str, list[str]]:
    failures: dict[str, list[str]] = {
        "marker": [],
        "target": [],
        "source": [],
        "backup": [],
        "layout": [],
    }
    marker_valid, _ = validate_backup_marker(journal)
    if not marker_valid:
        failures["marker"].append("backup transaction marker missing or changed")
    expected_backup_names = (
        {BACKUP_MARKER_NAME} if journal["backup_root"] is not None else set()
    )
    for operation in journal["operations"]:
        source = Path(operation["source"])
        try:
            source_matches = (
                source.is_dir()
                and tree_sha256(source) == operation["source_sha256"]
            )
        except OSError:
            source_matches = False
        if not source_matches:
            failures["source"].append(str(source))
        target = Path(operation["target"])
        if not expected_link(target, operation["link_target"]):
            failures["target"].append(str(target))
        quarantine = Path(operation["quarantine_target"])
        if operation["action"] == "replace-with-backup":
            try:
                staging_matches = (
                    path_exists(quarantine)
                    and entry_sha256(quarantine) == operation["original_sha256"]
                )
            except (OSError, ValueError):
                staging_matches = False
            if journal["state"] != "committed" and (
                operation["phase"] != "backup-published"
                or not staging_matches
                or entry_has_identity(quarantine, operation)
            ):
                failures["layout"].append(str(quarantine))
            backup_target = Path(operation["backup_target"])
            expected_backup_names.add(backup_target.name)
            try:
                backup_matches = (
                    path_exists(backup_target)
                    and entry_sha256(backup_target) == operation["original_sha256"]
                    and entry_has_identity(backup_target, operation)
                )
            except (OSError, ValueError):
                backup_matches = False
            if not backup_matches:
                failures["backup"].append(str(backup_target))
        elif path_exists(quarantine):
            failures["layout"].append(str(quarantine))
    backup_value = journal["backup_root"]
    if backup_value is not None:
        backup_root = Path(backup_value)
        try:
            actual_names = {path.name for path in backup_root.iterdir()}
        except OSError:
            actual_names = set()
        if actual_names != expected_backup_names:
            failures["layout"].append(str(backup_root))
    return failures


def _recover_transaction(
    target_root: Path,
    journal_path: Path,
    lock_handle,
    lock_path: Path,
    expected_journal: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    if not lock_path_matches(lock_handle, lock_path):
        return False, "RECOVERY_INCOMPLETE: restore lock path changed"
    anchor_path = anchor_path_for(target_root)
    if not path_exists(journal_path):
        if path_exists(anchor_path):
            return False, "RECOVERY_INCOMPLETE: active transaction journal is missing"
        if expected_journal is None:
            return True, "RECOVERY_NOT_NEEDED"
        if expected_journal.get("state") == "committed":
            failures = committed_invariant_failures(expected_journal)
            if not any(failures.values()):
                return True, "RECOVERED: verified committed transaction without journal"
        original_state = True
        for operation in expected_journal.get("operations", []):
            target = Path(operation["target"])
            staging = Path(operation["quarantine_target"])
            if operation["action"] == "create":
                original_state = original_state and not path_exists(target) and not path_exists(staging)
            else:
                try:
                    target_matches = (
                        path_exists(target)
                        and entry_sha256(target) == operation["original_sha256"]
                    )
                except (OSError, ValueError):
                    target_matches = False
                original_state = original_state and target_matches and not path_exists(staging)
        backup_value = expected_journal.get("backup_root")
        if isinstance(backup_value, str) and path_exists(Path(backup_value)):
            original_state = False
        if original_state:
            return True, "RECOVERED: verified original state without journal"
        return False, "RECOVERY_INCOMPLETE: active transaction journal is missing"
    try:
        journal_value = load_journal_safely(journal_path)
        if (
            journal_value.get("target_root_existed") is False
            and journal_value.get("target_root_device") is None
            and journal_value.get("target_root_inode") is None
            and journal_value.get("state") == "applying"
            and isinstance(journal_value.get("operations"), list)
            and journal_value["operations"]
            and all(
                isinstance(operation, dict)
                and operation.get("phase") == "planned"
                for operation in journal_value["operations"]
            )
            and trusted_directory(target_root)
            and not any(target_root.iterdir())
        ):
            target_metadata = target_root.lstat()
            journal_value["target_root_device"] = target_metadata.st_dev
            journal_value["target_root_inode"] = target_metadata.st_ino
            write_journal(journal_path, journal_value)
        backup_value = journal_value.get("backup_root")
        if (
            isinstance(backup_value, str)
            and journal_value.get("backup_root_device") is None
            and journal_value.get("backup_root_inode") is None
            and journal_value.get("state") == "applying"
            and isinstance(journal_value.get("operations"), list)
            and journal_value["operations"]
            and all(
                isinstance(operation, dict)
                and operation.get("phase") == "planned"
                for operation in journal_value["operations"]
            )
        ):
            unbound_backup_root = Path(backup_value)
            if (
                trusted_directory(unbound_backup_root.parent)
                and trusted_directory(unbound_backup_root)
                and stat.S_IMODE(unbound_backup_root.lstat().st_mode) == 0o700
                and {
                    path.name for path in unbound_backup_root.iterdir()
                }.issubset({BACKUP_MARKER_NAME})
            ):
                backup_metadata = unbound_backup_root.lstat()
                journal_value["backup_root_device"] = backup_metadata.st_dev
                journal_value["backup_root_inode"] = backup_metadata.st_ino
                write_journal(journal_path, journal_value)
        journal = validate_journal(journal_value, target_root)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return False, f"RECOVERY_INCOMPLETE: unsafe restore journal: {exc}"

    anchor_valid, anchor_present = validate_active_anchor(journal, anchor_path)
    if not anchor_valid:
        return False, "RECOVERY_INCOMPLETE: active transaction anchor missing or changed"
    marker_valid, marker_present = validate_backup_marker(journal)
    if not marker_valid:
        return False, (
            "RECOVERY_INCOMPLETE: backup transaction marker missing or changed"
        )

    operations = journal["operations"]
    if (
        os.environ.get("AWF_RESTORE_TEST_MODE") == "1"
        and os.environ.get("AWF_RESTORE_TEST_SWAP_TARGET_DURING_RECOVERY") == "1"
        and path_exists(target_root)
    ):
        swapped_target = target_root.with_name(
            f"{target_root.name}.awf-test-recovery-swapped"
        )
        rename_noreplace(target_root, swapped_target)
        create_private_directory_noreplace(target_root)
    if journal["state"] == "committed":
        failures = committed_invariant_failures(journal)
        if failures["target"]:
            return False, (
                "RECOVERY_INCOMPLETE: committed target changed; not finalizing: "
                + ", ".join(failures["target"])
            )
        if failures["source"]:
            return False, (
                "RECOVERY_INCOMPLETE: committed source changed; not finalizing: "
                + ", ".join(failures["source"])
            )
        if failures["backup"] or failures["layout"] or failures["marker"]:
            return False, (
                "RECOVERY_INCOMPLETE: committed backup missing or changed; not finalizing: "
                + ", ".join(
                    failures["backup"] + failures["layout"] + failures["marker"]
                )
            )
        if not lock_path_matches(lock_handle, lock_path):
            return False, "RECOVERY_INCOMPLETE: restore lock path changed"
        for operation in operations:
            if (
                not lock_path_matches(lock_handle, lock_path)
                or not target_root_matches_journal(target_root, journal)
                or not backup_root_state_is_safe(journal)
            ):
                return False, (
                    "RECOVERY_INCOMPLETE: root identity or permissions changed "
                    "during committed cleanup"
                )
            current_failures = committed_invariant_failures(journal)
            if any(current_failures.values()):
                return False, (
                    "RECOVERY_INCOMPLETE: committed invariant changed during cleanup"
                )
            if operation["action"] == "replace-with-backup":
                staging = Path(operation["quarantine_target"])
                if path_exists(staging):
                    remove_owned_entry(staging)
                    maybe_crash("after-committed-staging-remove")
        if (
            not lock_path_matches(lock_handle, lock_path)
            or not target_root_matches_journal(target_root, journal)
            or not backup_root_state_is_safe(journal)
            or any(committed_invariant_failures(journal).values())
        ):
            return False, (
                "RECOVERY_INCOMPLETE: committed invariant changed before journal cleanup"
            )
        fsync_directory(target_root)
        if anchor_present and path_exists(anchor_path):
            anchor_path.unlink()
            fsync_directory(anchor_path.parent)
            maybe_crash("after-active-anchor-remove")
        remove_journal(journal_path)
        return True, "RECOVERED: finalized committed transaction"

    if not lock_path_matches(lock_handle, lock_path):
        return False, "RECOVERY_INCOMPLETE: restore lock path changed"
    if journal["state"] != "rolled-back":
        journal["state"] = "rolling-back"
        write_journal(journal_path, journal)
    errors: list[str] = []
    for operation in reversed(operations):
        if not lock_path_matches(lock_handle, lock_path):
            errors.append("restore lock path changed during recovery")
            break
        if not target_root_state_is_safe(target_root, journal):
            errors.append("target root identity or permissions changed during recovery")
            break
        if not backup_root_state_is_safe(journal):
            errors.append("backup root identity or permissions changed during recovery")
            break
        action = operation["action"]
        target = Path(operation["target"])
        literal_target = operation["link_target"]
        if action == "create":
            quarantine = Path(operation["quarantine_target"])
            if expected_link(target, literal_target):
                if operation["phase"] == "planned":
                    errors.append(
                        f"planned create target appeared concurrently; not removed: {target}"
                    )
                    continue
                if path_exists(quarantine):
                    errors.append(f"restore quarantine is occupied: {quarantine}")
                    continue
                rename_noreplace(target, quarantine)
                fsync_directory(target.parent)
                maybe_crash("after-recovery-quarantine")
            elif path_exists(target):
                errors.append(f"target changed concurrently; not removed: {target}")
                continue
            if path_exists(quarantine):
                if expected_link(quarantine, literal_target):
                    quarantine.unlink()
                    fsync_directory(quarantine.parent)
                    maybe_crash("after-recovery-link-remove")
                else:
                    if not path_exists(target):
                        try:
                            rename_noreplace(quarantine, target)
                            fsync_directory(target.parent)
                        except OSError:
                            pass
                    errors.append(
                        f"quarantined target changed concurrently; not removed: {quarantine}"
                    )
            continue

        backup_target = Path(operation["backup_target"])
        staging = Path(operation["quarantine_target"])
        original_sha256 = operation["original_sha256"]
        target_is_link = expected_link(target, literal_target)
        original_locations = [
            path
            for path in (target, staging, backup_target)
            if entry_has_identity(path, operation)
        ]
        if len(original_locations) > 1:
            errors.append(
                f"original entry identity appears at multiple paths: {target}"
            )
            continue

        if entry_has_identity(target, operation):
            pass
        elif target_is_link and entry_has_identity(staging, operation):
            rename_exchange(target, staging)
            fsync_directory(target.parent)
            maybe_crash("after-recovery-backup-rename")
        elif target_is_link and entry_has_identity(backup_target, operation):
            if path_exists(staging):
                remove_owned_entry(staging)
            copy_entry_noreplace(backup_target, staging)
            maybe_crash("after-recovery-copy-staging")
            if (
                entry_has_identity(staging, operation)
                or entry_sha256(staging) != original_sha256
            ):
                errors.append(f"recovery staging copy invariant failed: {staging}")
                continue
            rename_exchange(backup_target, staging)
            fsync_directory(backup_target.parent)
            fsync_directory(staging.parent)
            maybe_crash("after-recovery-original-to-staging")
            if not entry_has_identity(staging, operation):
                errors.append(f"original entry did not move to staging: {staging}")
                continue
            rename_exchange(target, staging)
            fsync_directory(target.parent)
            maybe_crash("after-recovery-backup-rename")
        elif not path_exists(target) and entry_has_identity(staging, operation):
            rename_noreplace(staging, target)
            fsync_directory(target.parent)
        elif not path_exists(target) and entry_has_identity(backup_target, operation):
            if path_exists(staging):
                remove_owned_entry(staging)
            copy_entry_noreplace(backup_target, staging)
            rename_exchange(backup_target, staging)
            fsync_directory(backup_target.parent)
            fsync_directory(staging.parent)
            if not entry_has_identity(staging, operation):
                errors.append(f"original entry did not move to staging: {staging}")
                continue
            rename_noreplace(staging, target)
            fsync_directory(target.parent)
        else:
            errors.append(
                f"original entry identity is not in a recoverable location: {target}"
            )
            continue

        if not entry_has_identity(target, operation):
            errors.append(f"exact original entry was not restored: {target}")
            continue
        if path_exists(backup_target):
            if entry_has_identity(backup_target, operation):
                errors.append(f"refusing to delete original entry from backup: {backup_target}")
                continue
            remove_owned_entry(backup_target)
            maybe_crash("after-recovery-backup-remove")
        if path_exists(staging):
            if entry_has_identity(staging, operation):
                errors.append(f"refusing to delete original entry from staging: {staging}")
                continue
            remove_owned_entry(staging)
            maybe_crash("after-recovery-link-remove")
    if errors:
        return False, "RECOVERY_INCOMPLETE: " + "; ".join(errors)

    journal["state"] = "rolled-back"
    write_journal(journal_path, journal)
    maybe_crash("after-rollback-recorded")
    if not lock_path_matches(lock_handle, lock_path):
        return False, "RECOVERY_INCOMPLETE: restore lock path changed"
    backup_value = journal.get("backup_root")
    if journal.get("backup_root_created") and isinstance(backup_value, str):
        if not backup_root_state_is_safe(journal):
            return False, "RECOVERY_INCOMPLETE: backup root identity or permissions changed"
        backup_root = Path(backup_value)
        marker_path = backup_root / BACKUP_MARKER_NAME
        if marker_present and marker_path.is_file():
            marker_path.unlink()
            fsync_directory(backup_root)
            maybe_crash("after-recovery-marker-remove")
        if backup_root.is_dir() and not any(backup_root.iterdir()):
            backup_root.rmdir()
            fsync_directory(backup_root.parent)
            maybe_crash("after-recovery-backup-root-remove")
    if not journal.get("target_root_existed") and target_root.is_dir() and not any(target_root.iterdir()):
        target_metadata = target_root.lstat()
        if (target_metadata.st_dev, target_metadata.st_ino) != (
            journal["target_root_device"],
            journal["target_root_inode"],
        ):
            return False, "RECOVERY_INCOMPLETE: target root ownership changed"
        target_root.rmdir()
        fsync_directory(target_root.parent)
        maybe_crash("after-recovery-target-root-remove")
    if anchor_present and path_exists(anchor_path):
        anchor_path.unlink()
        fsync_directory(anchor_path.parent)
        maybe_crash("after-active-anchor-remove")
    remove_journal(journal_path)
    return True, "RECOVERED: rolled back interrupted transaction"


def recover_transaction(
    target_root: Path,
    journal_path: Path,
    lock_handle,
    lock_path: Path,
    expected_journal: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    try:
        return _recover_transaction(
            target_root,
            journal_path,
            lock_handle,
            lock_path,
            expected_journal,
        )
    except Exception as exc:
        return False, (
            "RECOVERY_INCOMPLETE: recovery I/O or schema error: "
            f"{type(exc).__name__}: {exc}"
        )


def load_strict_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate_keys,
    )
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _project_report(
    project: Path,
    declaration: Path,
    status: str,
    *,
    errors: list[str] | None = None,
    action: str = "plan",
    state: str | None = None,
) -> dict[str, Any]:
    target = {
        "target": "explicit",
        "action": action,
        "state": state or status,
        "status": "NOT_DECLARED" if status == "NOT_DECLARED" else "PASS" if status == "PASS" else "FAIL",
        "errors": errors or [],
    }
    return {
        "command": "restore_skills.py",
        "mode": "project",
        "status": status,
        "project": str(project),
        "declaration": str(declaration),
        "per_target": [target],
    }


def _project_cli_error(args: argparse.Namespace, message: str) -> int:
    project = args.project.expanduser().resolve()
    declaration = project / ".codex" / "workflows.yaml"
    report = _project_report(
        project,
        declaration,
        "FAIL",
        errors=[message],
        action="apply" if args.apply else "plan",
        state="REJECTED",
    )
    if args.json:
        _emit_project_report(report, True)
    return 2


def _emit_project_report(report: dict[str, Any], json_mode: bool) -> None:
    if json_mode:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return
    if report["status"] == "NOT_DECLARED":
        print(f"NOT_DECLARED {report['declaration']}")
        return
    for error in report.get("per_target", [{}])[0].get("errors", []):
        print(f"PRECHECK_FAILED: {error}")
    print(f"RESULT {report['status']}")


def _resolve_project_declaration(project_value: Path) -> dict[str, Any]:
    project = project_value.expanduser().resolve()
    declaration = project / ".codex" / "workflows.yaml"
    if not declaration.is_file():
        return {
            "project": project,
            "declaration": declaration,
            "status": "NOT_DECLARED",
            "selected": {},
            "errors": [],
        }
    errors: list[str] = []
    try:
        document = load_strict_json(declaration)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {
            "project": project,
            "declaration": declaration,
            "status": "FAIL",
            "selected": {},
            "errors": [str(exc)],
        }
    try:
        registry = load_strict_json(ROOT / "registry.yaml")
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {
            "project": project,
            "declaration": declaration,
            "status": "FAIL",
            "selected": {},
            "errors": [f"Registry is not strict JSON-compatible: {exc}"],
        }
    expected_keys = {"schema_version", "project_id", "registry", "uses"}
    if set(document) != expected_keys:
        errors.append("project declaration schema fields do not match schema_version 1")
    if document.get("schema_version") != 1:
        errors.append(f"project schema_version must be 1: {document.get('schema_version')}")
    if not isinstance(document.get("project_id"), str) or not document.get("project_id"):
        errors.append("project_id must be a non-empty string")
    declared_registry = document.get("registry")
    expected_registry = (ROOT / "registry.yaml").resolve()
    if not isinstance(declared_registry, str) or not declared_registry:
        errors.append("registry must be a non-empty path")
    else:
        declared_registry_path = Path(declared_registry).expanduser().resolve()
        if declared_registry_path != expected_registry:
            errors.append(
                f"registry mismatch: expected={expected_registry} declared={declared_registry_path}"
            )
    uses = document.get("uses")
    if not isinstance(uses, list) or not uses:
        errors.append("uses must be a non-empty list")
        uses = []

    workflows: dict[str, dict[str, Any]] = {}
    registry_workflows = registry.get("workflows")
    if not isinstance(registry_workflows, list):
        errors.append("Registry workflows must be a list")
        registry_workflows = []
    for entry in registry_workflows:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            errors.append("Registry workflow entry is invalid")
            continue
        if entry["id"] in workflows:
            errors.append(f"duplicate Registry workflow id: {entry['id']}")
        workflows[entry["id"]] = entry

    selected: dict[str, str] = {}
    seen_workflows: set[str] = set()
    for item in uses:
        if not isinstance(item, dict):
            errors.append("each project use must be an object")
            continue
        if set(item) != {"workflow", "release", "purpose"}:
            errors.append("project use fields must be workflow, release and purpose")
        workflow_id = item.get("workflow")
        release_id = item.get("release")
        if not isinstance(workflow_id, str) or not workflow_id:
            errors.append("project use workflow must be a non-empty string")
            continue
        if workflow_id in seen_workflows:
            errors.append(f"duplicate project use: {workflow_id}")
            continue
        seen_workflows.add(workflow_id)
        if not isinstance(release_id, str) or not release_id:
            errors.append(f"project release is invalid for {workflow_id}")
            continue
        workflow_entry = workflows.get(workflow_id)
        if workflow_entry is None:
            errors.append(f"unknown Workflow: {workflow_id}")
            continue
        catalog = workflow_entry.get("releases")
        if not isinstance(catalog, dict):
            errors.append(f"Workflow catalog is missing for {workflow_id}")
            continue
        catalog_entry = catalog.get(release_id)
        if not isinstance(catalog_entry, dict):
            errors.append(f"unknown Workflow Release: {workflow_id}@{release_id}")
            continue
        catalog_status = catalog_entry.get("status")
        if catalog_status not in {"validated", "superseded", "deprecated"}:
            errors.append(
                f"candidate or unknown Workflow Release is not adoptable: "
                f"{workflow_id}@{release_id} [{catalog_status}]"
            )
        try:
            descriptor_path = resolve_under(
                ROOT, catalog_entry["descriptor"], f"descriptor for {workflow_id}@{release_id}"
            )
            lock_path = resolve_under(
                ROOT, catalog_entry["lock"], f"lock for {workflow_id}@{release_id}"
            )
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(str(exc))
            continue
        expected_descriptor_hash = catalog_entry.get("descriptor_sha256")
        expected_lock_hash = catalog_entry.get("lock_sha256")
        try:
            actual_descriptor_hash = file_sha256(descriptor_path)
            actual_lock_hash = file_sha256(lock_path)
        except OSError as exc:
            errors.append(f"Workflow catalog file missing: {exc}")
            continue
        if actual_descriptor_hash != expected_descriptor_hash:
            errors.append(
                f"Workflow descriptor digest mismatch for {workflow_id}@{release_id}: "
                f"expected={expected_descriptor_hash} actual={actual_descriptor_hash}"
            )
        if actual_lock_hash != expected_lock_hash:
            errors.append(
                f"Workflow lock digest mismatch for {workflow_id}@{release_id}: "
                f"expected={expected_lock_hash} actual={actual_lock_hash}"
            )
        try:
            workflow = load_strict_json(descriptor_path)
            lock = load_strict_json(lock_path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"Workflow metadata is invalid for {workflow_id}@{release_id}: {exc}")
            continue
        if workflow.get("schema_version") != 1 or lock.get("schema_version") != 1:
            errors.append(f"Workflow schema mismatch for {workflow_id}@{release_id}")
        if workflow.get("id") != workflow_id or lock.get("workflow") != workflow_id:
            errors.append(f"Workflow identity mismatch for {workflow_id}@{release_id}")
        if workflow.get("release") != release_id or lock.get("workflow_release") != release_id:
            errors.append(f"Workflow Release identity mismatch for {workflow_id}@{release_id}")
        if workflow.get("status") != catalog_status:
            errors.append(f"Workflow catalog status mismatch for {workflow_id}@{release_id}")
        components = lock.get("components")
        if not isinstance(components, dict):
            errors.append(f"Workflow lock components are invalid for {workflow_id}@{release_id}")
            continue
        for component_id, component_release in components.items():
            if not isinstance(component_id, str) or not isinstance(component_release, str):
                errors.append(f"Workflow lock component identity is invalid: {component_id}")
                continue
            previous = selected.get(component_id)
            if previous is not None and previous != component_release:
                errors.append(
                    f"conflicting component union: {component_id}@{previous} vs "
                    f"{component_id}@{component_release}"
                )
            else:
                selected[component_id] = component_release

    global_components = registry.get("global_components", {})
    if not isinstance(global_components, dict):
        errors.append("Registry global_components must be an object")
    else:
        for component_id, release_id in global_components.items():
            previous = selected.get(component_id)
            if previous is not None and previous != release_id:
                errors.append(
                    f"global component conflict: {component_id}@{previous} vs {component_id}@{release_id}"
                )
            else:
                selected[component_id] = release_id
    return {
        "project": project,
        "declaration": declaration,
        "status": "PASS" if not errors else "FAIL",
        "selected": selected,
        "errors": errors,
    }


def _run_project_json_wrapper(args: argparse.Namespace) -> int:
    resolution = _resolve_project_declaration(args.project)
    if resolution["status"] == "NOT_DECLARED":
        report = _project_report(
            resolution["project"],
            resolution["declaration"],
            "NOT_DECLARED",
            action="apply" if args.apply else "plan",
            state="NOT_DECLARED",
        )
        _emit_project_report(report, True)
        return 2
    if resolution["status"] != "PASS":
        report = _project_report(
            resolution["project"],
            resolution["declaration"],
            "FAIL",
            errors=resolution["errors"],
            action="apply" if args.apply else "plan",
            state="PRECHECK_FAILED",
        )
        _emit_project_report(report, True)
        return 1
    command = [sys.executable, str(Path(__file__).resolve())]
    command.extend(argument for argument in sys.argv[1:] if argument != "--json")
    child_env = os.environ.copy()
    child_env["AWF_PROJECT_JSON_CHILD"] = "1"
    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
        env=child_env,
    )
    errors = [] if result.returncode == 0 else _both_error_detail(result)
    state = (
        "COMMITTED"
        if args.apply and result.returncode == 0
        else "DRY_RUN"
        if result.returncode == 0
        else "FAILED"
    )
    report = _project_report(
        resolution["project"],
        resolution["declaration"],
        "PASS" if result.returncode == 0 else "FAIL",
        errors=errors,
        action="apply" if args.apply else "plan",
        state=state,
    )
    _emit_project_report(report, True)
    return result.returncode


def _both_target_root(harness: str) -> Path:
    if harness == "pi":
        return (Path.home() / ".pi" / "agent" / "skills").resolve()
    return (Path.home() / ".codex" / "skills").resolve()


def _harness_target_root(harness: str) -> Path:
    if harness == "pi":
        return (Path.home() / ".pi" / "agent" / "skills").resolve()
    if harness == "hermes":
        return (Path.home() / ".hermes" / "skills").resolve()
    return (Path.home() / ".codex" / "skills").resolve()


def _both_backup_root(base: Path, harness: str) -> Path:
    return (base / harness).resolve()


def _both_error_detail(result: subprocess.CompletedProcess[str]) -> list[str]:
    detail = [line for line in (result.stdout + "\n" + result.stderr).splitlines() if line]
    return detail or [f"child exited with status {result.returncode}"]


def _both_report(
    status: str,
    per_target: list[dict[str, Any]],
    *,
    overall_status: str | None = None,
) -> dict[str, Any]:
    return {
        "command": "restore_skills.py",
        "mode": "both",
        "status": status,
        "overall_status": overall_status or status,
        "per_target": per_target,
        "global_atomicity": "not_claimed",
    }


def _emit_both_report(report: dict[str, Any], json_mode: bool) -> None:
    if json_mode:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return
    print(f"BOTH {report['status']}: independent per-target transactions; no global atomicity")
    for item in report["per_target"]:
        print(
            f"{item['target']}: {item['action']} {item['state']} "
            f"({item['status']})"
        )
        for error in item.get("errors", []):
            print(f"  {error}")


def _run_both(args: argparse.Namespace) -> int:
    if args.target_root is not None:
        report = _both_report(
            "FAIL",
            [
                {
                    "target": harness,
                    "action": "preflight",
                    "state": "REJECTED",
                    "status": "FAIL",
                    "errors": ["--harness both forbids --target-root"],
                }
                for harness in ("pi", "codex")
            ],
        )
        _emit_both_report(report, args.json)
        return 2
    if args.recover and (
        args.apply
        or args.backup_dir is not None
        or args.workflow is not None
        or args.release is not None
        or args.global_only
        or args.allow_candidate
    ):
        report = _both_report(
            "FAIL",
            [
                {
                    "target": harness,
                    "action": "preflight",
                    "state": "REJECTED",
                    "status": "FAIL",
                    "errors": ["both --recover accepts only --harness; no workflow/release/scope/backup options"],
                }
                for harness in ("pi", "codex")
            ],
        )
        _emit_both_report(report, args.json)
        return 2
    if args.project is not None:
        report = _both_report(
            "FAIL",
            [
                {
                    "target": harness,
                    "action": "preflight",
                    "state": "REJECTED",
                    "status": "FAIL",
                    "errors": ["--project conflicts with --harness both"],
                }
                for harness in ("pi", "codex")
            ],
        )
        _emit_both_report(report, args.json)
        return 2

    target_roots = {harness: _both_target_root(harness) for harness in ("pi", "codex")}
    backup_base = args.backup_dir.expanduser().resolve() if args.backup_dir else None
    backup_roots = (
        {harness: _both_backup_root(backup_base, harness) for harness in ("pi", "codex")}
        if backup_base is not None
        else {}
    )
    preflight_items: list[dict[str, Any]] = []
    preflight_ok = True
    namespace_errors: dict[str, list[str]] = {"pi": [], "codex": []}
    for harness, target_root in target_roots.items():
        if paths_overlap(target_root, ROOT):
            namespace_errors[harness].append(
                f"target root overlaps AWF repository: {target_root}"
            )
        if not trusted_directory(target_root.parent):
            namespace_errors[harness].append(
                "target root parent must be owner-owned and not group/other-writable: "
                f"{target_root.parent}"
            )
        if path_exists(target_root) and not trusted_directory(target_root):
            namespace_errors[harness].append(
                f"existing target root is not owner-controlled: {target_root}"
            )
    if backup_base is not None:
        if paths_overlap(backup_base, ROOT):
            for errors in namespace_errors.values():
                errors.append(f"backup directory overlaps AWF repository: {backup_base}")
        for harness, target_root in target_roots.items():
            if paths_overlap(backup_base, target_root):
                namespace_errors[harness].append(
                    f"backup base overlaps {harness} target root: {backup_base}"
                )
        if not trusted_directory(backup_base.parent):
            for errors in namespace_errors.values():
                errors.append(
                    "backup directory parent must be owner-owned and not "
                    f"group/other-writable: {backup_base.parent}"
                )
        elif not path_exists(backup_base):
            for errors in namespace_errors.values():
                errors.append(f"backup base directory must already exist: {backup_base}")
        elif not trusted_directory(backup_base):
            for errors in namespace_errors.values():
                errors.append(f"backup directory is not owner-controlled: {backup_base}")
        for harness, backup_root in backup_roots.items():
            if paths_overlap(backup_root, target_roots[harness]) or paths_overlap(
                backup_root, ROOT
            ):
                namespace_errors[harness].append(
                    f"backup directory overlaps protected target/repository: {backup_root}"
                )
        if paths_overlap(backup_roots["pi"], backup_roots["codex"]):
            for errors in namespace_errors.values():
                errors.append("Pi and Codex backup directories overlap")

    child_base = [sys.executable, str(Path(__file__).resolve())]
    common: list[str] = []
    if args.workflow is not None:
        common += ["--workflow", args.workflow]
    if args.release is not None:
        common += ["--release", args.release]
    if args.global_only:
        common.append("--global-only")
    if args.allow_candidate:
        common.append("--allow-candidate")

    def child_command(harness: str, *, apply: bool = False, recover: bool = False) -> list[str]:
        # Keep children on the parent-captured resolved target even if HOME changes.
        command = child_base + common + [
            "--harness",
            harness,
            "--target-root",
            str(target_roots[harness]),
        ]
        if backup_base is not None and not recover:
            command += ["--backup-dir", str(backup_roots[harness])]
        if apply:
            command.append("--apply")
        if recover:
            command.append("--recover")
        return command

    # Recovery has no registry/source preflight, but each target still has its own
    # lock and journal and is attempted independently.
    if args.recover:
        results: list[dict[str, Any]] = []
        for harness in ("pi", "codex"):
            result = subprocess.run(
                child_command(harness, recover=True),
                text=True,
                capture_output=True,
                check=False,
            )
            ok = result.returncode == 0
            results.append(
                {
                    "target": harness,
                    "target_root": str(target_roots[harness]),
                    "action": "recover",
                    "state": "RECOVERED" if ok else "RECOVERY_INCOMPLETE",
                    "status": "PASS" if ok else "FAIL",
                    "errors": [] if ok else _both_error_detail(result),
                }
            )
        passed = sum(item["status"] == "PASS" for item in results)
        status = "PASS" if passed == 2 else "PARTIAL" if passed else "FAIL"
        report = _both_report(status, results, overall_status=status)
        _emit_both_report(report, args.json)
        return 0 if status == "PASS" else 1

    # Read-only child plans are collected for both targets before any apply child
    # is started. This is the fail-closed boundary for --harness both.
    for harness in ("pi", "codex"):
        errors = list(namespace_errors[harness])
        if errors:
            preflight_ok = False
        else:
            result = subprocess.run(
                child_command(harness),
                text=True,
                capture_output=True,
                check=False,
            )
            if result.returncode != 0:
                preflight_ok = False
                errors.extend(_both_error_detail(result))
            elif args.apply and backup_base is None and "replace-with-backup:" in result.stdout:
                preflight_ok = False
                errors.append("existing Skill targets require --backup-dir for both apply")
            elif args.apply and backup_base is not None and backup_roots[harness].exists():
                preflight_ok = False
                errors.append(f"backup directory already exists: {backup_roots[harness]}")
        preflight_items.append(
            {
                "target": harness,
                "target_root": str(target_roots[harness]),
                "action": "preflight",
                "state": "READY" if not errors else "PRECHECK_FAILED",
                "status": "PASS" if not errors else "FAIL",
                "errors": errors,
            }
        )

    if not preflight_ok:
        report = _both_report("FAIL", preflight_items, overall_status="FAIL")
        _emit_both_report(report, args.json)
        return 1
    if not args.apply:
        for item in preflight_items:
            item["action"] = "plan"
            item["state"] = "DRY_RUN"
        report = _both_report("PASS", preflight_items)
        _emit_both_report(report, args.json)
        return 0

    results = []
    for harness in ("pi", "codex"):
        child_env = os.environ.copy()
        child_env["AWF_RESTORE_BOTH_EXECUTION"] = harness
        result = subprocess.run(
            child_command(harness, apply=True),
            text=True,
            capture_output=True,
            check=False,
            env=child_env,
        )
        ok = result.returncode == 0
        results.append(
            {
                "target": harness,
                "target_root": str(target_roots[harness]),
                "action": "apply",
                "state": "COMMITTED" if ok else "FAILED",
                "status": "PASS" if ok else "FAIL",
                "errors": [] if ok else _both_error_detail(result),
            }
        )
    passed = sum(item["status"] == "PASS" for item in results)
    status = "PASS" if passed == 2 else "PARTIAL" if passed else "FAIL"
    report = _both_report(status, results, overall_status=status)
    _emit_both_report(report, args.json)
    return 0 if status == "PASS" else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Restore AWF-locked Skills for an Agent harness. Default is a read-only plan; use --apply to change files."
    )
    parser.add_argument("--workflow")
    parser.add_argument(
        "--project",
        type=Path,
        help="Restore the exact Workflow union declared by DIR/.codex/workflows.yaml.",
    )
    parser.add_argument(
        "--release",
        help="Select an exact Workflow Release from the Registry release catalog.",
    )
    parser.add_argument(
        "--component",
        help=(
            "Select one exact registered Skill Component as ID@RELEASE for isolated "
            "evaluation; does not merge Registry global_components."
        ),
    )
    parser.add_argument(
        "--harness",
        choices=("pi", "codex", "hermes", "both"),
        default="codex",
        help="Select the default Skill target root; --target-root overrides it.",
    )
    parser.add_argument(
        "--global-only",
        action="store_true",
        help="Restore only Registry global_components, without any Workflow Skills.",
    )
    parser.add_argument("--target-root", type=Path)
    parser.add_argument("--backup-dir", type=Path, help="Required with --apply if a non-matching target already exists.")
    parser.add_argument(
        "--allow-candidate",
        action="store_true",
        help="Permit explicit candidate restoration for isolated testing; never implies adoption.",
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--recover",
        action="store_true",
        help="Recover the durable transaction for the selected target root; safe to repeat.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit one machine-readable JSON report for new restore modes.",
    )
    args = parser.parse_args()
    if args.harness == "both":
        return _run_both(args)
    # Failure injection is test-only; the harness selector alone is inert in production.
    if (
        args.apply
        and os.environ.get("AWF_RESTORE_TEST_MODE") == "1"
        and os.environ.get("AWF_RESTORE_BOTH_EXECUTION") == args.harness
        and os.environ.get("AWF_RESTORE_TEST_FAIL_HARNESS") == args.harness
    ):
        print(f"PRECHECK_FAILED: injected {args.harness} execution failure")
        return 1
    if args.project is not None:
        if args.target_root is None:
            if args.json:
                return _project_cli_error(args, "--project requires explicit --target-root")
            parser.error("--project requires explicit --target-root")
        if (
            args.workflow is not None
            or args.release is not None
            or args.component is not None
            or args.global_only
        ):
            if args.json:
                return _project_cli_error(
                    args,
                    "--project conflicts with --workflow, --release, --component and --global-only",
                )
            parser.error(
                "--project conflicts with --workflow, --release, --component and --global-only"
            )
        if args.recover:
            if args.json:
                return _project_cli_error(args, "--project conflicts with --recover")
            parser.error("--project conflicts with --recover")
        if args.json and os.environ.get("AWF_PROJECT_JSON_CHILD") != "1":
            return _run_project_json_wrapper(args)
    if args.apply and args.recover:
        parser.error("--apply and --recover are mutually exclusive")
    if args.recover and any(
        (
            args.workflow is not None,
            args.release is not None,
            args.component is not None,
            args.global_only,
            args.backup_dir is not None,
            args.allow_candidate,
        )
    ):
        parser.error("--recover accepts only --harness or --target-root")
    if (
        args.apply
        and not args.global_only
        and args.workflow is None
        and args.project is None
        and args.component is None
    ):
        parser.error(
            "--apply requires explicit --workflow or --global-only or --project "
            "or --component"
        )
    if args.global_only and args.release is not None:
        parser.error("--release cannot be combined with --global-only")
    if args.component is not None and (
        args.workflow is not None or args.release is not None or args.global_only
    ):
        parser.error(
            "--component conflicts with --workflow, --release and --global-only"
        )
    workflow_id = args.workflow or "pdf-document-stack"
    if args.target_root is not None:
        target_root = args.target_root.expanduser().resolve()
    else:
        target_root = _harness_target_root(args.harness)
    requested_backup_root = (
        args.backup_dir.expanduser().resolve() if args.backup_dir is not None else None
    )
    if paths_overlap(target_root, ROOT):
        print(f"PRECHECK_FAILED: target root overlaps AWF repository: {target_root}")
        return 1
    if requested_backup_root is not None:
        if paths_overlap(requested_backup_root, ROOT):
            print(
                f"PRECHECK_FAILED: backup directory overlaps AWF repository: {requested_backup_root}"
            )
            return 1
        if paths_overlap(requested_backup_root, target_root):
            print(
                f"PRECHECK_FAILED: backup directory overlaps target root: {requested_backup_root}"
            )
            return 1

    restore_lock = None
    restore_lock_path: Path | None = None
    journal_path = journal_path_for(target_root)
    anchor_path = anchor_path_for(target_root)
    project_resolution: dict[str, Any] | None = None
    if args.project is not None:
        project_resolution = _resolve_project_declaration(args.project)
        if project_resolution["status"] == "NOT_DECLARED":
            report = _project_report(
                project_resolution["project"],
                project_resolution["declaration"],
                "NOT_DECLARED",
                action="plan" if not args.apply else "apply",
                state="NOT_DECLARED",
            )
            _emit_project_report(report, args.json)
            return 2
        if project_resolution["status"] != "PASS":
            report = _project_report(
                project_resolution["project"],
                project_resolution["declaration"],
                "FAIL",
                errors=project_resolution["errors"],
                action="plan" if not args.apply else "apply",
                state="PRECHECK_FAILED",
            )
            _emit_project_report(report, args.json)
            return 1
    if args.project is not None:
        if not trusted_directory(target_root.parent):
            parser.error(
                "target root parent must be owner-owned and not group/other-writable "
                f"for project restore: {target_root.parent}"
            )
        if path_exists(target_root) and not trusted_directory(target_root):
            parser.error(
                "existing target root must be owner-owned and not group/other-writable "
                f"for project restore: {target_root}"
            )
    if args.apply or args.recover:
        if not trusted_directory(target_root.parent):
            parser.error(
                "target root parent must be owner-owned and not group/other-writable "
                f"for transactional restore: {target_root.parent}"
            )
        if path_exists(target_root) and not trusted_directory(target_root):
            parser.error(
                "existing target root must be owner-owned and not group/other-writable "
                f"for transactional restore: {target_root}"
            )
        restore_lock_path = target_root.parent / f".{target_root.name}.awf-restore.lock"
        try:
            restore_lock = open_restore_lock(restore_lock_path)
        except (OSError, ValueError) as exc:
            parser.error(f"unsafe AWF restore lock: {restore_lock_path}: {exc}")
        try:
            fcntl.flock(restore_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error(
                f"another AWF restore holds the target lock: {restore_lock_path}"
            )
        if not lock_path_matches(restore_lock, restore_lock_path):
            parser.error(
                f"unsafe AWF restore lock changed during acquisition: {restore_lock_path}"
            )
    if args.recover:
        assert restore_lock is not None and restore_lock_path is not None
        recovered, message = recover_transaction(
            target_root, journal_path, restore_lock, restore_lock_path
        )
        print(message)
        return 0 if recovered else 1
    if args.apply and (path_exists(journal_path) or path_exists(anchor_path)):
        parser.error(
            "unfinished AWF restore journal requires --recover: "
            f"journal={journal_path} anchor={anchor_path}"
        )

    registry = load(ROOT / "registry.yaml")
    component_entries = {item["id"]: item for item in registry["components"]}
    actions: list[tuple[str, Path, Path]] = []
    preflight_errors: list[str] = []

    merge_global_components = args.component is None
    if args.project is not None:
        assert project_resolution is not None
        selected: dict[str, str] = dict(project_resolution["selected"])
    elif args.component is not None:
        if args.component.count("@") != 1:
            parser.error("--component must be an exact ID@RELEASE")
        component_id, component_release = args.component.split("@", 1)
        if (
            SKILL_COMPONENT_PATTERN.fullmatch(component_id) is None
            or not component_release
        ):
            parser.error("--component must select a registered Skill as ID@RELEASE")
        selected = {component_id: component_release}
    elif args.global_only:
        selected = {}
    else:
        workflow_entry = next(
            (item for item in registry["workflows"] if item["id"] == workflow_id),
            None,
        )
        if workflow_entry is None:
            parser.error(f"Workflow is not registered: {workflow_id}")
        selected_workflow_entry = workflow_entry
        if args.release is not None:
            selected_workflow_entry = workflow_entry.get("releases", {}).get(args.release)
            if selected_workflow_entry is None:
                parser.error(f"Workflow Release is not registered: {workflow_id}@{args.release}")
        try:
            workflow_path = resolve_under(
                ROOT,
                selected_workflow_entry["descriptor"],
                f"Workflow descriptor for {workflow_id}",
            )
            lock_path = resolve_under(
                ROOT,
                selected_workflow_entry["lock"],
                f"Workflow lock for {workflow_id}",
            )
        except (KeyError, ValueError) as exc:
            parser.error(str(exc))
        expected_workflow_hash = selected_workflow_entry.get("descriptor_sha256")
        if not isinstance(expected_workflow_hash, str):
            parser.error(f"Workflow descriptor digest is missing: {workflow_id}")
        actual_workflow_hash = file_sha256(workflow_path)
        if actual_workflow_hash != expected_workflow_hash:
            parser.error(
                f"Workflow descriptor digest mismatch: expected={expected_workflow_hash} "
                f"actual={actual_workflow_hash}"
            )
        expected_lock_hash = selected_workflow_entry.get("lock_sha256")
        if not isinstance(expected_lock_hash, str):
            parser.error(f"Workflow lock digest is missing: {workflow_id}")
        actual_lock_hash = file_sha256(lock_path)
        if actual_lock_hash != expected_lock_hash:
            parser.error(
                f"Workflow lock digest mismatch: expected={expected_lock_hash} "
                f"actual={actual_lock_hash}"
            )
        workflow = load(workflow_path)
        lock = load(lock_path)
        workflow_release = workflow.get("release")
        lock_release = lock.get("workflow_release")
        if workflow.get("id") != workflow_id or lock.get("workflow") != workflow_id:
            parser.error(
                f"Workflow identity mismatch: descriptor={workflow.get('id')} "
                f"lock={lock.get('workflow')} expected={workflow_id}"
            )
        if workflow_release != lock_release or (
            args.release is not None and workflow_release != args.release
        ):
            parser.error(
                f"Workflow Release mismatch: descriptor={workflow_release} "
                f"lock={lock_release} requested={args.release}"
            )
        catalog_status = selected_workflow_entry.get("status")
        workflow_status = workflow.get("status")
        if catalog_status is not None and catalog_status != workflow_status:
            parser.error(
                f"Workflow status mismatch: catalog={catalog_status} descriptor={workflow_status}"
            )
        if workflow_status == "candidate" and not args.allow_candidate:
            parser.error(
                f"candidate Workflow requires --allow-candidate: {workflow_id}@{workflow_release}"
            )
        if workflow_status not in {"validated", "superseded", "deprecated", "candidate"}:
            parser.error(f"Workflow status is not restorable: {workflow_status}")
        components_value = lock.get("components")
        if not isinstance(components_value, dict):
            parser.error("Workflow lock components must be an object")
        selected = dict(components_value)
    if merge_global_components:
        for component_id, release_id in registry.get("global_components", {}).items():
            if component_id in selected and selected[component_id] != release_id:
                preflight_errors.append(
                    f"global component conflict for {component_id}: "
                    f"workflow={selected[component_id]} global={release_id}"
                )
            else:
                selected[component_id] = release_id
    for component_id, release_id in sorted(selected.items()):
        if not component_id.startswith("skill."):
            continue
        if SKILL_COMPONENT_PATTERN.fullmatch(component_id) is None:
            preflight_errors.append(f"invalid Skill component id: {component_id}")
            continue
        component_entry = component_entries.get(component_id)
        if component_entry is None:
            preflight_errors.append(f"unregistered Skill component: {component_id}")
            continue
        try:
            descriptor_path = resolve_under(
                ROOT,
                component_entry["descriptor"],
                f"descriptor path for {component_id}",
            )
            descriptor = load(descriptor_path)
            release = descriptor["releases"][release_id]
            source = resolve_under(
                ROOT,
                release["source_snapshot"],
                f"source snapshot for {component_id}@{release_id}",
            )
        except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
            preflight_errors.append(f"invalid source for {component_id}@{release_id}: {exc}")
            continue
        if release.get("status") == "candidate" and not args.allow_candidate:
            preflight_errors.append(
                f"candidate Component requires --allow-candidate: {component_id}@{release_id}"
            )
        elif release.get("status") not in {
            "validated",
            "superseded",
            "deprecated",
            "candidate",
        }:
            preflight_errors.append(
                f"Component status is not restorable: {component_id}@{release_id} "
                f"[{release.get('status')}]"
            )
        expected_tree = release.get("tree_sha256")
        if not source.is_dir():
            preflight_errors.append(f"missing source for {component_id}@{release_id}: {source}")
        elif not expected_tree:
            preflight_errors.append(f"missing source digest for {component_id}@{release_id}")
        else:
            actual_tree = tree_sha256(source)
            if actual_tree != expected_tree:
                preflight_errors.append(
                    f"source digest mismatch for {component_id}@{release_id}: "
                    f"expected={expected_tree} actual={actual_tree}"
                )
        target = target_root / component_id.removeprefix("skill.")
        if target.is_symlink() and target.resolve() == source:
            actions.append(("unchanged", source, target))
        elif target.exists() or target.is_symlink():
            actions.append(("replace-with-backup", source, target))
        else:
            actions.append(("create", source, target))

    for action, source, target in actions:
        print(f"{action}: {target} -> {source}")
    if preflight_errors:
        for error in preflight_errors:
            print(f"PRECHECK_FAILED: {error}")
        return 1
    if not args.apply:
        print("DRY-RUN: no files changed")
        return 0

    conflicts = [target for action, _, target in actions if action == "replace-with-backup"]
    if conflicts and args.backup_dir is None:
        parser.error("existing Skill targets require --backup-dir for reversible replacement")

    changed = [item for item in actions if item[0] != "unchanged"]
    if not changed:
        print("APPLIED: already matches exact lock")
        return 0

    target_root_existed = path_exists(target_root)
    if target_root_existed and not target_root.is_dir():
        print(f"PRECHECK_FAILED: target root is not a directory: {target_root}")
        return 1
    target_root_metadata = target_root.lstat() if target_root_existed else None
    backup_root = requested_backup_root
    backup_root_created = backup_root is not None
    if backup_root:
        try:
            backup_root.relative_to(target_root)
        except ValueError:
            pass
        else:
            print(f"PRECHECK_FAILED: backup directory must not be inside target root: {backup_root}")
            return 1
        if not trusted_directory(backup_root.parent):
            print(
                "PRECHECK_FAILED: backup parent must be owner-owned and not "
                f"group/other-writable: {backup_root.parent}"
            )
            return 1
        target_device_path = target_root if target_root_existed else target_root.parent
        if target_device_path.stat().st_dev != backup_root.parent.stat().st_dev:
            print("PRECHECK_FAILED: backup and target must be on the same filesystem")
            return 1
        if path_exists(backup_root):
            print(f"PRECHECK_FAILED: backup directory already exists: {backup_root}")
            return 1
    transaction_id = str(uuid.uuid4())
    operations: list[dict[str, Any]] = []
    for action, source, target in changed:
        fsync_entry(source)
        fsync_directory(source.parent)
        original_sha256 = None
        original_metadata = None
        if action == "replace-with-backup":
            fsync_entry(target)
            fsync_directory(target.parent)
            original_sha256 = entry_sha256(target)
            original_metadata = target.lstat()
        backup_target = backup_root / target.name if action == "replace-with-backup" else None
        operations.append(
            {
                "action": action,
                "source": str(source),
                "source_sha256": tree_sha256(source),
                "target": str(target),
                "link_target": os.path.relpath(source, start=target.parent),
                "backup_target": str(backup_target) if backup_target is not None else None,
                "original_sha256": original_sha256,
                "original_device": (
                    original_metadata.st_dev if original_metadata is not None else None
                ),
                "original_inode": (
                    original_metadata.st_ino if original_metadata is not None else None
                ),
                "quarantine_target": str(
                    target_root / f".awf-restore-{transaction_id}-{target.name}"
                ),
                "phase": "planned",
            }
        )
    journal: dict[str, Any] = {
        "schema_version": JOURNAL_SCHEMA_VERSION,
        "transaction_id": transaction_id,
        "repository_root": str(ROOT.resolve()),
        "target_root": str(target_root),
        "target_root_existed": target_root_existed,
        "target_root_device": (
            target_root_metadata.st_dev if target_root_metadata is not None else None
        ),
        "target_root_inode": (
            target_root_metadata.st_ino if target_root_metadata is not None else None
        ),
        "backup_root": str(backup_root) if backup_root is not None else None,
        "backup_root_created": backup_root_created,
        "backup_root_device": None,
        "backup_root_inode": None,
        "backup_marker_sha256": None,
        "anchor_sha256": None,
        "state": "applying",
        "operations": operations,
    }
    journal["anchor_sha256"] = hashlib.sha256(
        serialized_json(active_anchor_value(journal)).encode("utf-8")
    ).hexdigest()
    if backup_root is not None:
        journal["backup_marker_sha256"] = hashlib.sha256(
            serialized_json(backup_marker_value(journal)).encode("utf-8")
        ).hexdigest()
    try:
        write_journal(journal_path, journal, create=True)
        maybe_crash("after-journal-created")
        write_private_json_noreplace(anchor_path, active_anchor_value(journal))
        maybe_crash("after-anchor-created")
    except FileExistsError:
        parser.error(f"unfinished AWF restore journal requires --recover: {journal_path}")

    fail_after: int | None = None
    if os.environ.get("AWF_RESTORE_TEST_MODE") == "1":
        configured_fail_after = os.environ.get("AWF_RESTORE_TEST_FAIL_AFTER")
        if configured_fail_after:
            fail_after = int(configured_fail_after)
    changed_actions = 0
    try:
        if restore_lock is None or restore_lock_path is None or not lock_path_matches(
            restore_lock, restore_lock_path
        ):
            raise RuntimeError("restore lock changed before filesystem mutation")
        if target_root_existed:
            fsync_directory(target_root)
            fsync_directory(target_root.parent)
        else:
            target_root_metadata = create_private_directory_noreplace(target_root)
            maybe_crash("after-target-root-mkdir-before-identity")
        if not target_root_existed:
            journal["target_root_device"] = target_root_metadata.st_dev
            journal["target_root_inode"] = target_root_metadata.st_ino
            write_journal(journal_path, journal)
        maybe_crash("after-target-root-directory")
        if backup_root is not None:
            backup_root_metadata = create_private_directory_noreplace(backup_root)
            maybe_crash("after-backup-mkdir-before-identity")
            journal["backup_root_device"] = backup_root_metadata.st_dev
            journal["backup_root_inode"] = backup_root_metadata.st_ino
            write_journal(journal_path, journal)
            maybe_crash("after-backup-mkdir-before-marker")
            write_private_json_noreplace(
                backup_root / BACKUP_MARKER_NAME,
                backup_marker_value(journal),
            )
            maybe_crash("after-backup-directory")
            maybe_crash("after-backup-marker")
        if (
            os.environ.get("AWF_RESTORE_TEST_MODE") == "1"
            and os.environ.get("AWF_RESTORE_TEST_SWAP_TARGET_AFTER_SETUP") == "1"
        ):
            swapped_target = target_root.with_name(
                f"{target_root.name}.awf-test-swapped"
            )
            rename_noreplace(target_root, swapped_target)
            target_root.mkdir(mode=0o700)
            fsync_directory(target_root.parent)
        for operation in operations:
            if not lock_path_matches(restore_lock, restore_lock_path):
                raise RuntimeError("restore lock changed during filesystem mutation")
            if not target_root_matches_journal(target_root, journal):
                raise RuntimeError("target root identity changed during filesystem mutation")
            if not backup_root_state_is_safe(journal):
                raise RuntimeError("backup root identity changed during filesystem mutation")
            action = operation["action"]
            source = Path(operation["source"])
            if not source.is_dir() or tree_sha256(source) != operation["source_sha256"]:
                raise RuntimeError(f"source changed during restore: {source}")
            target = Path(operation["target"])
            backup_value = operation["backup_target"]
            backup_target = Path(backup_value) if backup_value is not None else None
            if action == "replace-with-backup":
                assert backup_target is not None and backup_root is not None
                staging = Path(operation["quarantine_target"])
                if entry_sha256(target) != operation["original_sha256"]:
                    raise RuntimeError(f"target changed before replacement: {target}")
                operation["phase"] = "backup-intent"
                write_journal(journal_path, journal)
                maybe_crash("after-backup-intent")
                copy_entry_noreplace(target, backup_target)
                if entry_sha256(backup_target) != operation["original_sha256"]:
                    raise RuntimeError(f"backup copy invariant failed: {backup_target}")
                maybe_crash("after-backup-staging-link")
                operation["phase"] = "backup-moved"
                write_journal(journal_path, journal)
                maybe_crash("after-backup-recorded")
                operation["phase"] = "link-intent"
                write_journal(journal_path, journal)
                maybe_crash("after-link-intent")
                staging.symlink_to(
                    operation["link_target"], target_is_directory=True
                )
                fsync_directory(staging.parent)
                maybe_crash("after-link-staging")
                if (
                    entry_sha256(target) != operation["original_sha256"]
                    or not expected_link(staging, operation["link_target"])
                    or entry_sha256(backup_target) != operation["original_sha256"]
                ):
                    raise RuntimeError(f"replacement entries changed before exchange: {target}")
                rename_exchange(target, staging)
                maybe_crash("after-backup-rename")
                fsync_directory(target.parent)
                if (
                    not expected_link(target, operation["link_target"])
                    or not entry_has_identity(staging, operation)
                    or entry_sha256(staging) != operation["original_sha256"]
                    or entry_has_identity(backup_target, operation)
                    or entry_sha256(backup_target) != operation["original_sha256"]
                ):
                    raise RuntimeError(f"replacement exchange invariant failed: {target}")
                maybe_crash("after-link-create")
                operation["phase"] = "link-created"
                write_journal(journal_path, journal)
                maybe_crash("after-link-recorded")
                rename_exchange(staging, backup_target)
                maybe_crash("after-original-backup-exchange")
                fsync_directory(backup_target.parent)
                maybe_crash("after-backup-destination-fsync")
                fsync_directory(staging.parent)
                maybe_crash("after-backup-source-fsync")
                if (
                    not expected_link(target, operation["link_target"])
                    or not entry_has_identity(backup_target, operation)
                    or entry_sha256(backup_target) != operation["original_sha256"]
                    or entry_has_identity(staging, operation)
                    or entry_sha256(staging) != operation["original_sha256"]
                ):
                    raise RuntimeError(f"original backup publication invariant failed: {target}")
                operation["phase"] = "backup-published"
                write_journal(journal_path, journal)
                maybe_crash("after-original-backup-recorded")
            else:
                operation["phase"] = "link-intent"
                write_journal(journal_path, journal)
                maybe_crash("after-link-intent")
                target.symlink_to(operation["link_target"], target_is_directory=True)
                fsync_directory(target.parent)
                maybe_crash("after-link-create")
                operation["phase"] = "link-created"
                write_journal(journal_path, journal)
                maybe_crash("after-link-recorded")
            changed_actions += 1
            if fail_after is not None and changed_actions >= fail_after:
                raise RuntimeError(f"injected failure after {changed_actions} action(s)")
        maybe_crash("after-all-actions")
        if os.environ.get("AWF_RESTORE_TEST_MODE") == "1":
            tampered_name = os.environ.get(
                "AWF_RESTORE_TEST_TAMPER_TARGET_AFTER_ALL"
            )
            if tampered_name:
                tampered_target = target_root / tampered_name
                if tampered_target.is_symlink():
                    tampered_target.unlink()
                elif path_exists(tampered_target):
                    raise RuntimeError(
                        f"test target is not replaceable: {tampered_target}"
                    )
                tampered_target.mkdir()
                with (tampered_target / "concurrent.txt").open(
                    "x", encoding="utf-8"
                ) as handle:
                    handle.write("concurrent\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                fsync_directory(tampered_target)
                fsync_directory(target_root)
        if restore_lock is None or restore_lock_path is None or not lock_path_matches(
            restore_lock, restore_lock_path
        ):
            raise RuntimeError("restore lock changed before commit")
        if not target_root_matches_journal(target_root, journal):
            raise RuntimeError("target root identity changed before commit")
        if not backup_root_state_is_safe(journal):
            raise RuntimeError("backup root identity changed before commit")
        failures = committed_invariant_failures(journal)
        invariant_errors = [
            f"{kind}={','.join(paths)}"
            for kind, paths in failures.items()
            if paths
        ]
        if invariant_errors:
            raise RuntimeError(
                "commit invariant failed: " + "; ".join(invariant_errors)
            )
        journal["state"] = "committed"
        write_journal(journal_path, journal)
        maybe_crash("after-commit")
        finalized, final_message = recover_transaction(
            target_root,
            journal_path,
            restore_lock,
            restore_lock_path,
            journal,
        )
        if not finalized:
            raise RuntimeError(final_message)
    except Exception as exc:
        assert restore_lock is not None and restore_lock_path is not None
        recovered, message = recover_transaction(
            target_root,
            journal_path,
            restore_lock,
            restore_lock_path,
            journal,
        )
        if recovered and "committed transaction" in message:
            print(f"APPLIED: committed transaction finalized after cleanup error: {exc}")
            return 0
        if recovered and message == "RECOVERY_NOT_NEEDED" and journal["state"] == "committed":
            print(f"APPLIED: committed transaction; journal cleanup durability warning: {exc}")
            return 0
        if recovered:
            print(f"ROLLED_BACK: {exc}")
        elif journal["state"] == "committed":
            print(f"COMMITTED_RECOVERY_REQUIRED: {exc}; {message}")
        else:
            print(f"ROLLBACK_INCOMPLETE: {exc}; {message}")
        return 1
    print("APPLIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
