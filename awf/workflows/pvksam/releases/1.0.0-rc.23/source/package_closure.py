#!/usr/bin/env python3
"""Package exact PVKSAM rc.23 closure, adapted from MolView rc.22 owner.

Features:
- Read-only standard packaging: verifies existing frozen source_manifest.json and
  workflow.yaml descriptor anchor without touching the release tree.
- Separate explicit --prepare-manifest flag: only permitted for unregistered candidates
  under the specified root to generate source_manifest.json and anchor it into workflow.yaml.
- Fresh output directory enforcement: refuses to overwrite non-empty output directories.
- Exact closure: preserves directory modes (e.g. 0775) and empty directories to prevent
  extraction mode degradation breaking awf-tree-v2.
- Bundle manifest: covers all tar members with type, mode, bytes, and sha256 (directories
  do not include bytes/sha256). Excludes runtime binaries, project media, and caches.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tarfile
import zipfile
sys.dont_write_bytecode = True

# Co-located verify module
import verify


SKIP_EXTENSIONS = {".pyc"}
SKIP_NAMES = {"__pycache__", ".pytest_cache", ".git", ".DS_Store"}

EXPECTED_LOCKED_COMPONENTS = {}
for _cid, _kind, _name, _version in [
    ('skill.pvksam', 'skills', 'pvksam', '1.0.0-rc.21'),
    ('recipe.pvksam-ase-cueq', 'recipes', 'pvksam-ase-cueq', '1.0.0-rc.3'),
    ('artifact.mace-mpa-0-medium', 'artifacts', 'mace-mpa-0-medium', '1.0.0-rc.2'),
    ('artifact.pvksam-reference-models', 'artifacts', 'pvksam-reference-models', '1.0.0-rc.1'),
]:
    EXPECTED_LOCKED_COMPONENTS[_cid] = {
        'release': _version,
        'path': f'components/{_kind}/{_name}/releases/{_version}',
        'descriptor': f'components/{_kind}/{_name}/component.yaml',
    }
LOCKED_COMPONENT_PATHS = [p for spec in EXPECTED_LOCKED_COMPONENTS.values()
                          for p in (spec['descriptor'], spec['path'])]

WORKFLOW_ID = "pvksam"
WORKFLOW_VERSION = "1.0.0-rc.23"
WORKFLOW_RELEASE_REL = f"workflows/{WORKFLOW_ID}/releases/{WORKFLOW_VERSION}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(65536):
            digest.update(chunk)
    return digest.hexdigest()


def get_root(arg_root: Path | None = None) -> Path:
    if arg_root:
        return arg_root.expanduser().resolve()
    if "AWF_ROOT" in os.environ:
        return Path(os.environ["AWF_ROOT"]).expanduser().resolve()
    return Path("/home/zc/agent_workflows").resolve()


def is_included_file(path: Path) -> bool:
    if path.suffix.lower() in SKIP_EXTENSIONS:
        return False
    for part in path.parts:
        if part in SKIP_NAMES:
            return False
    return True


def prepare_manifest(root: Path) -> dict:
    """Generate source_manifest.json and anchor it into workflow.yaml."""
    # 1. Candidate registration check: forbidden if already registered
    registry_file = root / "registry.yaml"
    if not registry_file.is_file():
        raise RuntimeError("Registry file registry.yaml is missing; cannot prove unregistered candidate status.")
    reg_data = json.loads(registry_file.read_text(encoding="utf-8"))
    wf_entry = next((w for w in reg_data.get("workflows", []) if w["id"] == WORKFLOW_ID), None)
    if wf_entry and WORKFLOW_VERSION in wf_entry.get("releases", {}):
        raise RuntimeError("Workflow already registered; manifest preparation forbidden.")
    req_att = reg_data.get("integrity_policy", {}).get("required_release_attestations", [])
    if "skill.pvksam@1.0.0-rc.21" in req_att:
        raise RuntimeError("New component already attested; manifest preparation forbidden.")

    skill_desc_file = root / "components/skills/pvksam/component.yaml"
    if not skill_desc_file.is_file():
        raise RuntimeError("Skill descriptor component.yaml is missing; cannot prove unregistered candidate status.")
    skill_desc = json.loads(skill_desc_file.read_text(encoding="utf-8"))
    if "1.0.0-rc.21" in skill_desc.get("releases", {}):
        raise RuntimeError("New skill already registered; manifest preparation forbidden.")

    wf_dir = root / WORKFLOW_RELEASE_REL
    wf_source_dir = wf_dir / "source"
    if not wf_source_dir.is_dir():
        raise RuntimeError(f"Workflow source directory {wf_source_dir} does not exist.")

    source_coverage = {}
    for p in sorted(wf_source_dir.rglob("*")):
        if p.is_file() and is_included_file(p):
            rel = p.relative_to(wf_source_dir).as_posix()
            source_coverage[rel] = {
                "bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            }

    if not source_coverage:
        raise RuntimeError(f"No regular source files found in {wf_source_dir}.")

    legacy_tree = verify.tree_sha256(wf_source_dir)
    tree_v2 = verify.tree_sha256_v2(wf_source_dir, "tree")
    total_bytes = sum(f["bytes"] for f in source_coverage.values())

    manifest_obj = {
        "schema_version": 1,
        "format": "awf-source-manifest-v1",
        "workflow": WORKFLOW_ID,
        "release": WORKFLOW_VERSION,
        "source_directory": f"{WORKFLOW_RELEASE_REL}/source",
        "entry_count": len(source_coverage),
        "regular_file_bytes": total_bytes,
        "legacy_tree_sha256": legacy_tree,
        "tree_sha256_v2": tree_v2,
        "files": source_coverage,
        "provenance": "Independent source manifest placed outside source/ to avoid self-reference; bridges verify helper coverage gap for Workflow/source.",
    }

    manifest_path = wf_dir / "source_manifest.json"
    manifest_bytes = (json.dumps(manifest_obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    manifest_path.write_bytes(manifest_bytes)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()

    # Anchor into workflow.yaml
    wf_yaml_path = wf_dir / "workflow.yaml"
    if not wf_yaml_path.is_file():
        raise RuntimeError(f"Workflow descriptor {wf_yaml_path} does not exist.")
    wf_yaml_data = json.loads(wf_yaml_path.read_text(encoding="utf-8"))
    wf_yaml_data["source_manifest"] = "source_manifest.json"
    wf_yaml_data["source_manifest_sha256"] = manifest_sha
    wf_yaml_path.write_text(json.dumps(wf_yaml_data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")

    return {
        "status": "PASS",
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "entry_count": len(source_coverage),
        "legacy_tree_sha256": legacy_tree,
        "tree_sha256_v2": tree_v2,
    }


def verify_frozen_manifest(root: Path) -> dict:
    """Read-only verification of workflow.yaml descriptor anchor and source_manifest.json."""
    wf_dir = root / WORKFLOW_RELEASE_REL
    wf_yaml_path = wf_dir / "workflow.yaml"
    if not wf_yaml_path.is_file():
        raise RuntimeError(f"workflow.yaml missing at {wf_yaml_path}")

    wf_yaml_data = json.loads(wf_yaml_path.read_text(encoding="utf-8"))
    if "source_manifest" not in wf_yaml_data or "source_manifest_sha256" not in wf_yaml_data:
        raise RuntimeError("workflow.yaml missing source_manifest or source_manifest_sha256 anchor.")

    manifest_rel = wf_yaml_data["source_manifest"]
    expected_manifest_sha = wf_yaml_data["source_manifest_sha256"]

    manifest_path = (wf_dir / manifest_rel).resolve()
    if not manifest_path.is_file():
        raise RuntimeError(f"source_manifest file {manifest_path} not found.")

    actual_manifest_sha = sha256_file(manifest_path)
    if actual_manifest_sha != expected_manifest_sha:
        raise RuntimeError(
            f"workflow.yaml source_manifest_sha256 anchor mismatch: expected {expected_manifest_sha}, got {actual_manifest_sha}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_files = manifest.get("files", {})
    if not manifest_files:
        raise RuntimeError("source_manifest.json has empty file inventory.")

    wf_source_dir = wf_dir / "source"
    if not wf_source_dir.is_dir():
        raise RuntimeError(f"Workflow source directory {wf_source_dir} is missing.")

    actual_files = {}
    for p in sorted(wf_source_dir.rglob("*")):
        if p.is_file() and is_included_file(p):
            rel = p.relative_to(wf_source_dir).as_posix()
            actual_files[rel] = p

    # Verify complete inventory (no missing, no extra)
    missing_files = set(manifest_files.keys()) - set(actual_files.keys())
    if missing_files:
        raise RuntimeError(f"Files listed in source_manifest.json missing on disk: {sorted(missing_files)}")

    extra_files = set(actual_files.keys()) - set(manifest_files.keys())
    if extra_files:
        raise RuntimeError(f"Extra files on disk not recorded in source_manifest.json: {sorted(extra_files)}")

    # Verify per-file bytes and sha256
    for rel, p in actual_files.items():
        expected_meta = manifest_files[rel]
        actual_size = p.stat().st_size
        if actual_size != expected_meta["bytes"]:
            raise RuntimeError(f"Size mismatch for {rel}: expected {expected_meta['bytes']}, got {actual_size}")
        actual_sha = sha256_file(p)
        if actual_sha != expected_meta["sha256"]:
            raise RuntimeError(f"SHA256 mismatch for {rel}: expected {expected_meta['sha256']}, got {actual_sha}")

    # Verify tree hashes
    actual_legacy = verify.tree_sha256(wf_source_dir)
    if actual_legacy != manifest.get("legacy_tree_sha256"):
        raise RuntimeError(
            f"legacy_tree_sha256 mismatch: manifest has {manifest.get('legacy_tree_sha256')}, actual is {actual_legacy}"
        )
    actual_v2 = verify.tree_sha256_v2(wf_source_dir, "tree")
    if actual_v2 != manifest.get("tree_sha256_v2"):
        raise RuntimeError(
            f"tree_sha256_v2 mismatch: manifest has {manifest.get('tree_sha256_v2')}, actual is {actual_v2}"
        )

    return manifest


def collect_closure_paths(root: Path) -> list[Path]:
    """Collect exact Workflow source and its four locked Component trees."""
    targets = [WORKFLOW_RELEASE_REL] + LOCKED_COMPONENT_PATHS
    collected: list[Path] = []

    for t in targets:
        target_path = root / t
        if not target_path.exists():
            raise RuntimeError(f"Required target does not exist: {target_path}")

        if target_path.is_file():
            if is_included_file(target_path):
                collected.append(target_path)
        elif target_path.is_dir():
            # Add directory itself
            collected.append(target_path)
            for sub in sorted(target_path.rglob("*")):
                if any(part in SKIP_NAMES for part in sub.parts):
                    continue
                if sub.is_symlink():
                    # Reject or skip external symlinks
                    target = sub.resolve()
                    try:
                        target.relative_to(root)
                    except ValueError:
                        raise RuntimeError(f"External symlink detected: {sub} -> {target}")
                    collected.append(sub)
                    continue
                if sub.is_dir():
                    collected.append(sub)
                elif sub.is_file():
                    if is_included_file(sub):
                        collected.append(sub)

    # Sort strictly by relative POSIX path ensuring parents precede children
    return sorted(set(collected), key=lambda p: p.relative_to(root).as_posix())


def closure_metadata(root: Path) -> tuple[bytes, bytes, dict[str, bytes]]:
    """Select only the exact lock, with original release metadata/attestations."""
    registry=json.loads((root/'registry.yaml').read_text())
    lock=json.loads((root/WORKFLOW_RELEASE_REL/'versions.lock').read_text())
    expected={key:value['release'] for key,value in EXPECTED_LOCKED_COMPONENTS.items()}
    if lock['components']!=expected:raise RuntimeError('Closure lock differs from exact four components')
    selected=[];descriptor_bytes={};wanted={f'{cid}@{version}' for cid,version in expected.items()}
    for item in registry['components']:
        if item['id'] not in expected:continue
        cid=item['id'];selected.append(item)
        descriptor=json.loads((root/item['descriptor']).read_text())
        descriptor['releases']={expected[cid]:descriptor['releases'][expected[cid]]}
        descriptor_bytes[item['descriptor']]=(json.dumps(descriptor,indent=2,ensure_ascii=False)+'\n').encode()
    if len(selected)!=len(EXPECTED_LOCKED_COMPONENTS):raise RuntimeError('Registered component closure incomplete')
    records={}
    for rel in registry['integrity_attestations']:
        att=json.loads((root/rel).read_text())
        for identity,record in att.get('releases',{}).items():
            if identity in wanted:
                if identity in records:raise RuntimeError('Duplicate attestation in exact closure')
                records[identity]=record
    if set(records)!=wanted:raise RuntimeError('Exact closure attestation set missing')
    wf=next(item for item in registry['workflows'] if item['id']==WORKFLOW_ID)
    entry=wf['releases'][WORKFLOW_VERSION]
    runtime={'schema_version':registry['schema_version'],'global_components':{},'components':selected,'workflows':[dict(wf,descriptor=entry['descriptor'],descriptor_sha256=entry['descriptor_sha256'],lock=entry['lock'],lock_sha256=entry['lock_sha256'],releases={WORKFLOW_VERSION:entry})],'integrity_policy':dict(registry['integrity_policy'],required_release_attestations=sorted(wanted)),'integrity_attestations':['attestations/exact-pvksam20-closure.yaml']}
    attestation={'schema_version':1,'digest_contract':'awf-tree-v2','provenance':'Exact selected records copied from registered authoritative attestations; source and release metadata unchanged.','releases':records}
    return (json.dumps(runtime,indent=2,ensure_ascii=False)+'\n').encode(),(json.dumps(attestation,indent=2,ensure_ascii=False)+'\n').encode(),descriptor_bytes

def verify_output(directory: Path) -> dict:
    directory=Path(directory).resolve();manifest=json.loads((directory/'manifest.json').read_text())
    archive_path=directory/manifest['archive'];zip_path=directory/manifest['zip']['archive']
    if sha256_file(archive_path)!=manifest['sha256'] or sha256_file(zip_path)!=manifest['zip']['sha256']:raise RuntimeError('Archive digest mismatch')
    expected={record['path']:record for record in manifest['members']}
    if len(expected)!=len(manifest['members']):raise RuntimeError('Duplicate member record')
    with tarfile.open(archive_path,'r:gz') as archive:
        members=archive.getmembers()
        if {x.name.rstrip('/') for x in members}!=set(expected) or len(members)!=len(expected):raise RuntimeError('TAR member set mismatch')
        for item in members:
            record=expected[item.name.rstrip('/')]
            if item.mode!=int(record['mode'],8):raise RuntimeError('TAR member mode mismatch')
            if record['type']=='file':
                if not item.isfile() or hashlib.sha256(archive.extractfile(item).read()).hexdigest()!=record['sha256']:raise RuntimeError('TAR bytes differ')
            elif record['type']=='symlink':
                if not item.issym() or item.linkname!=record['target']:raise RuntimeError('TAR symlink differs')
            elif not item.isdir():raise RuntimeError('TAR directory differs')
    with zipfile.ZipFile(zip_path) as zipped:
        items=zipped.infolist()
        if len(items)!=len(expected) or {x.filename.rstrip('/') for x in items}!=set(expected) or zipped.testzip() is not None:raise RuntimeError('ZIP member set/CRC mismatch')
        for item in items:
            record=expected[item.filename.rstrip('/')];member=PurePosixPath(item.filename)
            if member.is_absolute() or '..' in member.parts:raise RuntimeError('Unsafe ZIP member')
            if stat.S_IMODE(item.external_attr>>16)!=int(record['mode'],8):raise RuntimeError('ZIP mode mismatch')
            data=zipped.read(item)
            if record['type']=='file' and hashlib.sha256(data).hexdigest()!=record['sha256']:raise RuntimeError('ZIP bytes differ')
            if record['type']=='symlink' and data.decode()!=record['target']:raise RuntimeError('ZIP symlink differs')
    return {'status':'PASS','member_count':len(expected),'TAR_and_ZIP_bytes_types_modes_and_SHA_verified':True,'no_external_upload':True}

def main():
    parser = argparse.ArgumentParser(description="Package exact PVKSAM rc.23 four-component closure")
    parser.add_argument("--awf-root", type=Path, default=None)
    parser.add_argument("--prepare-manifest", action="store_true", help="Generate source_manifest.json and anchor in workflow.yaml")
    parser.add_argument("--output-dir", type=Path, default=None, help="Fresh directory to output bundle")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--verify-output-dir", type=Path, help="Read-only archive/member verification of an existing closure receipt")
    args = parser.parse_args()

    root = get_root(args.awf_root)
    if args.verify_output_dir:
        result=verify_output(args.verify_output_dir)
        print(json.dumps(result,indent=2))
        return 0

    if args.prepare_manifest:
        result = prepare_manifest(root)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Manifest prepared and anchored successfully:")
            print(f"  Manifest: {result['manifest_path']}")
            print(f"  SHA256:   {result['manifest_sha256']}")
            print(f"  Files:    {result['entry_count']}")
        return 0

    if not args.output_dir:
        parser.error("--output-dir is required when not using --prepare-manifest")

    out_dir = args.output_dir.expanduser().resolve()

    # Fresh output directory check: reject overwrite
    if out_dir.exists():
        if any(out_dir.iterdir()):
            print(f"REJECTED: output directory {out_dir} is not empty; refusing to overwrite.", file=sys.stderr)
            return 1
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Read-only verification of frozen manifest and descriptor anchor
    manifest = verify_frozen_manifest(root)

    # 2. Collect exact closure items (directories + files)
    closure_paths = collect_closure_paths(root)

    archive_path = out_dir / "pvksam-rc20-closure.tar.gz"
    runtime_reg_bytes, closure_attestation_bytes, selected_descriptors = closure_metadata(root)
    runtime_reg_sha = hashlib.sha256(runtime_reg_bytes).hexdigest()

    # Calculate four exact component source tree legacy and v2 hashes
    component_trees = {}
    for comp_id, spec in EXPECTED_LOCKED_COMPONENTS.items():
        source_dir = root / spec["path"] / "source"
        if not source_dir.is_dir():
            raise RuntimeError(f"Source directory missing for {comp_id} at {source_dir}")
        component_trees[comp_id] = {
            "release": spec["release"],
            "path": f"{spec['path']}/source",
            "legacy_tree_sha256": verify.tree_sha256(source_dir),
            "tree_sha256_v2": verify.tree_sha256_v2(source_dir, "tree"),
        }

    archive_path = out_dir / "pvksam-rc20-closure.tar.gz"
    manifest_path = out_dir / "manifest.json"
    sums_path = out_dir / "SHA256SUMS"

    items_to_archive = []
    for path in closure_paths:
        rel_posix = path.relative_to(root).as_posix()
        arcname = f"agent_workflows/{rel_posix}"
        rel=path.relative_to(root).as_posix()
        items_to_archive.append((arcname, None if rel in selected_descriptors else path, selected_descriptors.get(rel)))
    # Extra member: agent_workflows/registry.yaml from the exact selected closure
    items_to_archive.append(("agent_workflows/registry.yaml", None, runtime_reg_bytes))
    items_to_archive.append(("agent_workflows/attestations/exact-pvksam20-closure.yaml", None, closure_attestation_bytes))
    items_to_archive.sort(key=lambda x: x[0])

    members_records = []
    # Build tar with exact mode preservation and normalized ownership/mtime
    with archive_path.open("wb") as raw_stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_stream, mtime=0) as gz_stream:
            with tarfile.open(fileobj=gz_stream, mode="w", dereference=False) as archive:
                for arcname, path, raw_data in items_to_archive:
                    if path is not None:
                        st = path.lstat()
                        mode_str = f"{stat.S_IMODE(st.st_mode):04o}"

                        tar_info = archive.gettarinfo(path, arcname=arcname)
                        tar_info.uid = 0
                        tar_info.gid = 0
                        tar_info.uname = "root"
                        tar_info.gname = "root"
                        tar_info.mtime = 0
                        tar_info.mode = stat.S_IMODE(st.st_mode)

                        if path.is_symlink():
                            archive.addfile(tar_info)
                            members_records.append({'path':arcname,'type':'symlink','mode':mode_str,'target':os.readlink(path)})
                        elif path.is_dir():
                            tar_info.type = tarfile.DIRTYPE
                            archive.addfile(tar_info)
                            members_records.append({
                                "path": arcname,
                                "type": "dir",
                                "mode": mode_str,
                            })
                        else:
                            tar_info.type = tarfile.REGTYPE
                            file_sha = sha256_file(path)
                            with path.open("rb") as f:
                                archive.addfile(tar_info, f)
                            members_records.append({
                                "path": arcname,
                                "type": "file",
                                "mode": mode_str,
                                "bytes": st.st_size,
                                "sha256": file_sha,
                            })
                    else:
                        tar_info = tarfile.TarInfo(name=arcname)
                        tar_info.uid = 0
                        tar_info.gid = 0
                        tar_info.uname = "root"
                        tar_info.gname = "root"
                        tar_info.mtime = 0
                        tar_info.mode = 0o644
                        tar_info.type = tarfile.REGTYPE
                        tar_info.size = len(raw_data)
                        archive.addfile(tar_info, io.BytesIO(raw_data))
                        members_records.append({
                            "path": arcname,
                            "type": "file",
                            "mode": "0644",
                            "bytes": len(raw_data),
                            "sha256": hashlib.sha256(raw_data).hexdigest(),
                        })

    archive_sha = sha256_file(archive_path)
    archive_bytes = archive_path.stat().st_size

    # ZIP uses the same exact member inventory and Unix entry modes. All
    # publication/project-acceptance scripts are ordinary locked members.
    zip_path=out_dir/'pvksam-rc20-closure.zip'
    with zipfile.ZipFile(zip_path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as zipped:
        for arcname,path,raw_data in items_to_archive:
            is_dir=path is not None and path.is_dir() and not path.is_symlink()
            name=arcname+'/' if is_dir else arcname
            info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));info.create_system=3
            mode=path.lstat().st_mode if path is not None else stat.S_IFREG|0o644
            info.external_attr=(mode&0xffff)<<16
            if is_dir:info.external_attr|=0x10
            info.compress_type=zipfile.ZIP_DEFLATED
            data=(b'' if is_dir else os.readlink(path).encode() if path is not None and path.is_symlink() else path.read_bytes() if path is not None else raw_data)
            zipped.writestr(info,data)
    zip_sha=sha256_file(zip_path)

    # Write SHA256SUMS
    sums_path.write_text(f"{archive_sha}  pvksam-rc20-closure.tar.gz\n{zip_sha}  pvksam-rc20-closure.zip\n", encoding="utf-8")

    # Write bundle manifest
    wf_desc_path = root / WORKFLOW_RELEASE_REL / "workflow.yaml"
    wf_lock_path = root / WORKFLOW_RELEASE_REL / "versions.lock"
    wf_sm_path = root / WORKFLOW_RELEASE_REL / "source_manifest.json"

    bundle_manifest = {
        "schema_version": 1,
        "format": "pvksam-closure-bundle-v1",
        "archive": "pvksam-rc20-closure.tar.gz",
        "zip": {'archive':zip_path.name,'sha256':zip_sha,'bytes':zip_path.stat().st_size},
        "sha256": archive_sha,
        "archive_bytes": archive_bytes,
        "workflow": {
            "id": WORKFLOW_ID,
            "release": WORKFLOW_VERSION,
            "descriptor": f"{WORKFLOW_RELEASE_REL}/workflow.yaml",
            "descriptor_sha256": sha256_file(wf_desc_path),
            "lock": f"{WORKFLOW_RELEASE_REL}/versions.lock",
            "lock_sha256": sha256_file(wf_lock_path),
            "source_manifest": f"{WORKFLOW_RELEASE_REL}/source_manifest.json",
            "source_manifest_sha256": sha256_file(wf_sm_path),
        },
        "workflow_source_manifest": {
            "root": f"{WORKFLOW_RELEASE_REL}/source",
            "entry_count": manifest.get("entry_count"),
            "legacy_tree_sha256": manifest.get("legacy_tree_sha256"),
            "tree_sha256_v2": manifest.get("tree_sha256_v2"),
            "files": manifest.get("files"),
        },
        "locked_components": LOCKED_COMPONENT_PATHS,
        "component_source_trees": component_trees,
        "total_members": len(members_records),
        "members": members_records,
        "excluded_patterns": ["*.pyc", "__pycache__", ".pytest_cache", "external runtime binaries and project data outside locked trees"],
        "provenance": "Exact PVKSAM rc.23 closure with preserved directory modes/empty directories, four locked components, selected authoritative release metadata/attestations and complete Workflow engineering scripts. External runtimes and project result data remain external; bundled reference structures retain their data notice. Inherited source/component.yaml is historical payload, never active authority.",
    }
    manifest_path.write_text(json.dumps(bundle_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    result = {
        "status": "PASS",
        "bundle_dir": str(out_dir),
        "archive": str(archive_path),
        "sha256": archive_sha,
        "archive_bytes": archive_bytes,
        "total_members": len(members_records),
        "directories_count": sum(1 for m in members_records if m["type"] == "dir"),
        "files_count": sum(1 for m in members_records if m["type"] == "file"),
        "zip": str(zip_path),
        "zip_sha256": zip_sha,
        "archive_verification": verify_output(out_dir),
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Packaged closure bundle successfully:")
        print(f"  Archive:     {archive_path} ({archive_bytes} bytes)")
        print(f"  SHA256:      {archive_sha}")
        print(f"  Members:     {len(members_records)} ({result['directories_count']} dirs, {result['files_count']} files)")
        print(f"  Manifest:    {manifest_path}")
        print(f"  SHA256SUMS:  {sums_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
