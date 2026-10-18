#!/usr/bin/env python3
"""Verify AWF core backup receipts and reject unsafe tar layouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any


class Report:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def add(self, ok: bool, name: str, detail: str) -> None:
        self.checks.append({"status": "PASS" if ok else "FAIL", "check": name, "detail": detail})

    @property
    def ok(self) -> bool:
        return all(item["status"] == "PASS" for item in self.checks)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def parse_checksum(path: Path) -> tuple[str | None, str | None]:
    fields = path.read_text(encoding="utf-8").strip().split(maxsplit=1)
    if len(fields) != 2:
        return None, None
    return fields[0], fields[1]


def canonical_member_name(name: str) -> str:
    return PurePosixPath(name).as_posix()


def safe_member_path(name: str) -> bool:
    path = PurePosixPath(name)
    return (
        name == path.as_posix()
        and not path.is_absolute()
        and ".." not in path.parts
        and bool(path.parts)
        and path.parts[0] == "agent_workflows"
    )


def allowed_symlink_map(provenance: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in provenance.get("symlinks", []):
        if not isinstance(item, dict):
            continue
        path = item.get("path")
        target = item.get("target")
        if isinstance(path, str) and isinstance(target, str):
            result[f"agent_workflows/{path}"] = target
    return result


def verify_tar(
    archive: Path,
    provenance: dict[str, Any],
    report: Report,
    *,
    allow_unattested_symlinks: bool = False,
) -> None:
    try:
        with tarfile.open(archive, mode="r:gz") as bundle:
            members = bundle.getmembers()
    except (OSError, tarfile.TarError) as exc:
        report.add(False, "archive.tar", str(exc))
        return

    report.add(bool(members), "archive.tar", f"members={len(members)}")
    seen: set[str] = set()
    symlink_members: dict[str, str] = {}
    member_names: list[str] = []
    for member in members:
        canonical_name = canonical_member_name(member.name)
        member_names.append(canonical_name)
        report.add(safe_member_path(member.name), f"archive.path.{member.name}", member.name)
        duplicate = canonical_name in seen
        report.add(not duplicate, f"archive.unique.{member.name}", "duplicate" if duplicate else "unique")
        seen.add(canonical_name)
        allowed_type = member.isfile() or member.isdir() or member.issym()
        report.add(allowed_type, f"archive.type.{member.name}", member.type.decode("latin1"))
        if member.issym():
            symlink_members[canonical_name] = member.linkname

    symlink_paths = set(symlink_members)
    for name in member_names:
        path = PurePosixPath(name)
        parents = {parent.as_posix() for parent in path.parents if parent != PurePosixPath(".")}
        traverses_symlink = bool(parents & symlink_paths)
        report.add(
            not traverses_symlink,
            f"archive.symlink_parent.{name}",
            "member is not nested below an archived symlink" if not traverses_symlink else "member traverses symlink",
        )

    expected_symlinks = allowed_symlink_map(provenance)
    if allow_unattested_symlinks:
        report.add(
            True,
            "archive.symlink_allowlist",
            f"legacy-unattested={symlink_members}",
        )
    else:
        report.add(
            symlink_members == expected_symlinks,
            "archive.symlink_allowlist",
            f"expected={expected_symlinks} actual={symlink_members}",
        )


def verify_archive_hashes(
    archive: Path,
    checksum_path: Path,
    expected_hash: Any,
    provenance: dict[str, Any],
    report: Report,
    *,
    allow_unattested_symlinks: bool = False,
) -> None:
    report.add(archive.is_file(), "archive.exists", str(archive))
    report.add(checksum_path.is_file(), "checksum.exists", str(checksum_path))
    if not archive.is_file() or not checksum_path.is_file():
        return
    sidecar_hash, sidecar_archive = parse_checksum(checksum_path)
    report.add(sidecar_archive == archive.name, "checksum.archive", str(sidecar_archive))
    actual_hash = file_sha256(archive)
    report.add(
        actual_hash == expected_hash,
        "archive.manifest_hash",
        f"expected={expected_hash} actual={actual_hash}",
    )
    report.add(
        actual_hash == sidecar_hash,
        "archive.checksum_hash",
        f"expected={sidecar_hash} actual={actual_hash}",
    )
    verify_tar(
        archive,
        provenance,
        report,
        allow_unattested_symlinks=allow_unattested_symlinks,
    )


def verify_directory(bundle_dir: Path, report: Report) -> str:
    expected_names = {
        "COMPLETE",
        "SHA256SUMS",
        "agent_workflows-core.tar.gz",
        "manifest.json",
    }
    actual_names = {path.name for path in bundle_dir.iterdir()} if bundle_dir.is_dir() else set()
    report.add(bundle_dir.is_dir(), "bundle_dir.exists", str(bundle_dir))
    report.add(actual_names == expected_names, "bundle_dir.members", f"expected={sorted(expected_names)} actual={sorted(actual_names)}")
    if not bundle_dir.is_dir():
        return "atomic-directory-v1"

    complete_path = bundle_dir / "COMPLETE"
    manifest_path = bundle_dir / "manifest.json"
    try:
        complete = load_object(complete_path)
        report.add(True, "completion.load", str(complete_path))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        report.add(False, "completion.load", str(exc))
        complete = {}
    report.add(complete.get("schema_version") == 1, "completion.schema", str(complete.get("schema_version")))
    report.add(complete.get("manifest") == "manifest.json", "completion.manifest", str(complete.get("manifest")))
    if manifest_path.is_file():
        actual_manifest_hash = file_sha256(manifest_path)
        report.add(
            actual_manifest_hash == complete.get("manifest_sha256"),
            "completion.manifest_hash",
            f"expected={complete.get('manifest_sha256')} actual={actual_manifest_hash}",
        )
    try:
        manifest = load_object(manifest_path)
        report.add(True, "manifest.load", str(manifest_path))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        report.add(False, "manifest.load", str(exc))
        manifest = {}
    report.add(manifest.get("schema_version") == 1, "manifest.schema", str(manifest.get("schema_version")))
    report.add(manifest.get("format") == "atomic-directory-v1", "manifest.format", str(manifest.get("format")))
    archive_name = manifest.get("archive")
    archive = bundle_dir / archive_name if isinstance(archive_name, str) and Path(archive_name).name == archive_name else bundle_dir / "invalid"
    report.add(archive_name == "agent_workflows-core.tar.gz", "manifest.archive", str(archive_name))
    verify_archive_hashes(
        archive,
        bundle_dir / "SHA256SUMS",
        manifest.get("sha256"),
        manifest.get("provenance", {}) if isinstance(manifest.get("provenance"), dict) else {},
        report,
    )
    return "atomic-directory-v1"


def verify_legacy_archive(archive: Path, report: Report, allow_legacy_sidecar: bool) -> str:
    checksum = archive.with_name(archive.name + ".sha256")
    completion = archive.with_name(archive.name + ".complete")
    marker: dict[str, Any] = {}
    marker_present = completion.is_file()
    if marker_present:
        try:
            marker = load_object(completion)
            report.add(True, "completion.load", str(completion))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            report.add(False, "completion.load", str(exc))
    elif allow_legacy_sidecar:
        report.add(True, "completion.legacy", "legacy sidecar-only receipt explicitly allowed")
    else:
        report.add(False, "completion.load", f"missing: {completion}")

    if marker_present:
        report.add(marker.get("schema_version") == 1, "completion.schema", str(marker.get("schema_version")))
        report.add(marker.get("archive") == archive.name, "completion.archive", str(marker.get("archive")))
        expected_sidecar = archive.name + ".sha256"
        report.add(marker.get("sidecar") == expected_sidecar, "completion.sidecar", str(marker.get("sidecar")))
        expected_hash = marker.get("sha256")
        provenance = marker.get("provenance", {}) if isinstance(marker.get("provenance"), dict) else {}
        receipt_format = str(marker.get("format", "invalid-marker"))
        report.add(
            receipt_format == "legacy-sibling-receipt-v1",
            "completion.format",
            receipt_format,
        )
    else:
        expected_hash, _ = parse_checksum(checksum) if checksum.is_file() else (None, None)
        provenance = {}
        receipt_format = "legacy-sidecar-only"
    verify_archive_hashes(
        archive,
        checksum,
        expected_hash,
        provenance,
        report,
        allow_unattested_symlinks=not marker_present and allow_legacy_sidecar,
    )
    return receipt_format


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify an AWF atomic bundle directory or legacy core archive receipt."
    )
    parser.add_argument("bundle", type=Path)
    parser.add_argument(
        "--allow-legacy-sidecar",
        action="store_true",
        help="Accept a historical archive with only a valid .sha256 sidecar; report LEGACY_VALID.",
    )
    parser.add_argument("--json", action="store_true", help="Emit one JSON report.")
    args = parser.parse_args()

    bundle = args.bundle.expanduser().resolve()
    report = Report()
    if bundle.is_dir():
        receipt_format = verify_directory(bundle, report)
        legacy = False
    else:
        receipt_format = verify_legacy_archive(bundle, report, args.allow_legacy_sidecar)
        legacy = receipt_format == "legacy-sidecar-only"
    status = "LEGACY_VALID" if report.ok and legacy else "PASS" if report.ok else "FAIL"
    output = {
        "status": status,
        "format": receipt_format,
        "bundle": str(bundle),
        "checks": report.checks,
    }
    if args.json:
        print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    else:
        for item in report.checks:
            print(f"{item['status']:4} {item['check']}: {item['detail']}")
        print(f"RESULT {status} ({len(report.checks)} checks)")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
