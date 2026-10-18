#!/usr/bin/env python3
"""Create deterministic AWF core backups without external model payloads."""

from __future__ import annotations

import argparse
import ctypes
import gzip
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(os.environ.get("AWF_ROOT", str(Path(__file__).resolve().parents[5])))
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".git"}
AT_FDCWD = -100
RENAME_NOREPLACE = 1


def included_paths() -> list[Path]:
    return sorted(
        (
            path
            for path in ROOT.rglob("*")
            if not any(part in SKIP_PARTS for part in path.relative_to(ROOT).parts)
        ),
        key=lambda path: path.relative_to(ROOT).as_posix(),
    )


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def normalized(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    info.mtime = 0
    return info


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_text_fsynced(path: Path, content: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def atomic_write_text_noreplace(path: Path, content: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        rename_noreplace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_archive(path: Path, paths: list[Path]) -> None:
    with path.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", dereference=False) as archive:
                archive.add(ROOT, arcname="agent_workflows", recursive=False, filter=normalized)
                for source in paths:
                    relative = source.relative_to(ROOT)
                    archive.add(
                        source,
                        arcname=Path("agent_workflows") / relative,
                        recursive=False,
                        filter=normalized,
                    )
        raw.flush()
        os.fsync(raw.fileno())


def ensure_outside_repository(path: Path, parser: argparse.ArgumentParser) -> None:
    try:
        path.relative_to(ROOT)
    except ValueError:
        return
    parser.error("backup destination must be outside the repository")


def repository_provenance() -> dict[str, Any]:
    registry_path = ROOT / "registry.yaml"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    workflows: dict[str, list[str]] = {}
    for entry in registry.get("workflows", []):
        catalog = entry.get("releases")
        if isinstance(catalog, dict):
            workflows[entry["id"]] = sorted(catalog)
        else:
            descriptor = json.loads((ROOT / entry["descriptor"]).read_text(encoding="utf-8"))
            workflows[entry["id"]] = [descriptor.get("release")]
    symlinks = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "target": os.readlink(path),
        }
        for path in included_paths()
        if path.is_symlink()
    ]
    return {
        "repository": registry.get("repository", {}),
        "registry_sha256": digest(registry_path),
        "global_components": registry.get("global_components", {}),
        "workflow_releases": workflows,
        "symlinks": symlinks,
    }


def rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError("atomic no-replace directory publication requires renameat2")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        AT_FDCWD,
        os.fsencode(source),
        AT_FDCWD,
        os.fsencode(destination),
        RENAME_NOREPLACE,
    )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def publish_legacy_archive(
    output: Path,
    paths: list[Path],
    regular_bytes: int,
) -> tuple[str, Path, Path]:
    sidecar = output.with_name(output.name + ".sha256")
    completion = output.with_name(output.name + ".complete")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", dir=output.parent
    )
    os.close(temporary_descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        write_archive(temporary, paths)
        checksum = digest(temporary)
        rename_noreplace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    atomic_write_text_noreplace(sidecar, f"{checksum}  {output.name}\n")
    marker = {
        "schema_version": 1,
        "format": "legacy-sibling-receipt-v1",
        "archive": output.name,
        "sidecar": sidecar.name,
        "sha256": checksum,
        "entry_count": len(paths),
        "regular_file_bytes": regular_bytes,
        "provenance": repository_provenance(),
    }
    atomic_write_text_noreplace(
        completion,
        json.dumps(marker, ensure_ascii=False, sort_keys=True) + "\n",
    )
    fsync_directory(output.parent)
    return checksum, sidecar, completion


def publish_atomic_directory(
    destination: Path,
    paths: list[Path],
    regular_bytes: int,
) -> tuple[str, Path]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    archive = staging / "agent_workflows-core.tar.gz"
    try:
        write_archive(archive, paths)
        checksum = digest(archive)
        write_text_fsynced(staging / "SHA256SUMS", f"{checksum}  {archive.name}\n")
        manifest = {
            "schema_version": 1,
            "format": "atomic-directory-v1",
            "archive": archive.name,
            "sha256": checksum,
            "entry_count": len(paths),
            "regular_file_bytes": regular_bytes,
            "provenance": repository_provenance(),
        }
        manifest_text = json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n"
        write_text_fsynced(staging / "manifest.json", manifest_text)
        completion = {
            "schema_version": 1,
            "manifest": "manifest.json",
            "manifest_sha256": hashlib.sha256(manifest_text.encode("utf-8")).hexdigest(),
        }
        write_text_fsynced(
            staging / "COMPLETE",
            json.dumps(completion, ensure_ascii=False, sort_keys=True) + "\n",
        )
        fsync_directory(staging)
        rename_noreplace(staging, destination)
        fsync_directory(destination.parent)
        return checksum, destination
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bundle AWF core sources and Recipes; registered external model payloads are not copied."
    )
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument(
        "--bundle-dir",
        type=Path,
        help="Atomically publish a new receipt directory (recommended).",
    )
    destination.add_argument(
        "--output",
        type=Path,
        help="Publish the legacy sibling archive/sidecar/marker format.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selected = args.bundle_dir if args.bundle_dir is not None else args.output
    target = selected.expanduser().resolve()
    ensure_outside_repository(target, parser)

    if args.bundle_dir is not None:
        existing = [target] if target.exists() or target.is_symlink() else []
    else:
        existing = [
            path
            for path in (
                target,
                target.with_name(target.name + ".sha256"),
                target.with_name(target.name + ".complete"),
            )
            if path.exists() or path.is_symlink()
        ]
    if existing:
        parser.error(f"output already exists: {', '.join(str(path) for path in existing)}")

    paths = included_paths()
    regular_bytes = sum(path.stat().st_size for path in paths if path.is_file() and not path.is_symlink())
    print(f"root={ROOT}")
    print(f"entries={len(paths)}")
    print(f"regular_file_bytes={regular_bytes}")
    print(f"destination={target}")
    if args.dry_run:
        return 0

    if args.bundle_dir is not None:
        checksum, published = publish_atomic_directory(target, paths, regular_bytes)
        print(f"sha256={checksum}")
        print(f"bundle_dir={published}")
    else:
        checksum, sidecar, completion = publish_legacy_archive(target, paths, regular_bytes)
        print(f"sha256={checksum}")
        print(f"sidecar={sidecar}")
        print(f"completion={completion}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
