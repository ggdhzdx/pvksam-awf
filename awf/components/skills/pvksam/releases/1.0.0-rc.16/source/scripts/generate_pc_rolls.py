#!/usr/bin/env python3
"""Generate site-phase candidates from deduplicated phosphonate skeletons.

中文：对每个已完成孤立骨架后处理的膦酸 SAM 代表，固定 P 和 PO3 三个 O，
绕 P-C 轴生成唯一周期方位角网格。随后将候选交给 fixed-site-cpu 入口做
头基对齐、高度和周期碰撞检查；本脚本不筛碰撞、不算能量，也不优化坐标。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from ase.io import read


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_record_path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _rotation_matrix(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    theta = math.radians(angle_deg)
    x, y, z = axis
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return (
        np.eye(3) * math.cos(theta)
        + (1.0 - math.cos(theta)) * np.outer(axis, axis)
        + math.sin(theta) * cross
    )


def generate_rolls(
    postprocess_manifest: Path,
    output_dir: Path,
    *,
    step_deg: float = 30.0,
) -> tuple[Path, Path]:
    postprocess_manifest = postprocess_manifest.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not postprocess_manifest.is_file():
        raise FileNotFoundError(f"skeleton postprocess manifest not found: {postprocess_manifest}")
    if not math.isfinite(step_deg) or step_deg <= 0.0:
        raise ValueError("step_deg must be a positive finite angle")
    phase_count = round(360.0 / step_deg)
    if phase_count < 1 or not math.isclose(
        phase_count * step_deg, 360.0, rel_tol=0.0, abs_tol=1.0e-8
    ):
        raise ValueError("step_deg must divide the 360-degree circle into unique phases")
    parent = json.loads(postprocess_manifest.read_text(encoding="utf-8"))
    if not isinstance(parent, dict):
        raise ValueError("postprocess manifest root must be a JSON object")
    if (
        parent.get("schema") != "sam-phosphonate-skeleton-postprocess-v1"
        or parent.get("status") != "completed"
        or parent.get("grid_complete") is not True
        or parent.get("records_scope") != "post_optimization_skeleton_geometry_all_outcomes"
    ):
        raise ValueError("roll generation requires a completed, full phosphonate skeleton manifest")
    system = parent.get("system")
    representatives = parent.get("representatives")
    if not isinstance(system, dict) or not isinstance(representatives, list) or not representatives:
        raise ValueError("completed skeleton manifest lacks system metadata or representatives")
    p_index = system.get("p_atom_id_1based")
    c_index = system.get("c_atom_id_1based")
    organic_ids = system.get("organic_atom_ids_1based")
    if (
        isinstance(p_index, bool)
        or not isinstance(p_index, int)
        or isinstance(c_index, bool)
        or not isinstance(c_index, int)
        or not isinstance(organic_ids, list)
        or not organic_ids
        or not all(isinstance(index, int) and not isinstance(index, bool) for index in organic_ids)
    ):
        raise ValueError("skeleton manifest lacks valid 1-based P-C and organic atom IDs")
    p_index -= 1
    c_index -= 1
    organic_indices = [index - 1 for index in organic_ids]
    if (
        len(set(organic_indices)) != len(organic_indices)
        or p_index in organic_indices
        or c_index not in organic_indices
    ):
        raise ValueError("P-C and carbon-side atom membership is inconsistent")

    phase_angles = [float(index * step_deg) for index in range(phase_count)]
    records = []
    common_symbols = None
    for row in sorted(representatives, key=lambda item: int(item.get("cluster_index", 0))):
        cluster_index = row.get("cluster_index")
        source_value = row.get("representative_structure")
        expected_hash = row.get("representative_structure_sha256")
        if isinstance(cluster_index, bool) or not isinstance(cluster_index, int) or cluster_index < 1:
            raise ValueError("representative cluster_index must be a positive integer")
        if not isinstance(source_value, str) or not isinstance(expected_hash, str):
            raise ValueError(f"cluster {cluster_index} lacks a structure path or hash")
        source = _resolve_record_path(source_value, postprocess_manifest.parent)
        if not source.is_file() or sha256(source) != expected_hash:
            raise ValueError(f"representative structure is missing or changed: {source}")
        atoms = read(source, format="extxyz")
        symbols = atoms.get_chemical_symbols()
        positions = np.asarray(atoms.positions, dtype=float)
        if positions.ndim != 2 or positions.shape != (len(symbols), 3) or not np.isfinite(positions).all():
            raise ValueError(f"invalid coordinates in representative: {source}")
        if max(p_index, c_index, *organic_indices) >= len(symbols):
            raise ValueError(f"atom mapping exceeds representative atom count: {source}")
        if symbols[p_index] != "P" or symbols[c_index] != "C":
            raise ValueError(f"mapped P-C axis has wrong elements in {source}")
        if common_symbols is None:
            common_symbols = symbols
        elif symbols != common_symbols:
            raise ValueError("skeleton representatives do not share one atom order and composition")
        head_indices = sorted(set(range(len(symbols))) - set(organic_indices))
        if len(head_indices) != 4 or p_index not in head_indices:
            raise ValueError("carbon-side atom map must leave exactly P plus three anchor O atoms fixed")
        axis = positions[c_index] - positions[p_index]
        norm = float(np.linalg.norm(axis))
        if not math.isfinite(norm) or norm <= 1.0e-10:
            raise ValueError(f"degenerate P-C axis in {source}")
        axis /= norm
        skeleton_id = f"cluster-{cluster_index:06d}"
        for angle in phase_angles:
            rotated = positions.copy()
            rotation = _rotation_matrix(axis, angle)
            rotated[organic_indices] = (
                positions[organic_indices] - positions[p_index]
            ) @ rotation.T + positions[p_index]
            if not np.array_equal(rotated[head_indices], positions[head_indices]):
                raise RuntimeError("P/O head coordinates changed during axial rotation")
            angle_token = f"{angle:g}"
            candidate_id = f"{skeleton_id}__pc-roll-{angle_token}deg"
            records.append(
                {
                    "candidate_id": candidate_id,
                    "skeleton_id": skeleton_id,
                    "pc_roll_deg": angle,
                    "positions_A": rotated.tolist(),
                }
            )
    if len({record["candidate_id"] for record in records}) != len(records):
        raise ValueError("duplicate candidate identifiers were generated")

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"P-C roll output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    ensemble_path = output_dir / "pc-roll-candidate-ensemble.json"
    manifest_path = output_dir / "pc-roll-generation.json"
    ensemble = {
        "schema": "pvksam-fixed-site-candidate-ensemble-v1",
        "candidate_source": "P-C axial phase grid from completed isolated phosphonate skeleton representatives",
        "symbols": common_symbols,
        "records": records,
    }
    ensemble_path.write_text(
        json.dumps(ensemble, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema": "pvksam-pc-roll-generation-v1",
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": {
            "axis": "P-to-bonded-carbon; carbon-side component only",
            "fixed_head": "P plus the three P-bound anchor O atoms",
            "step_deg": float(step_deg),
            "angles_deg": phase_angles,
            "rotation_is_geometry_only": True,
            "surface_alignment_or_collision_screened": False,
            "optimization_or_energy_evaluated": False,
        },
        "source": {
            "skeleton_postprocess_manifest": str(postprocess_manifest),
            "skeleton_postprocess_manifest_sha256": sha256(postprocess_manifest),
            "skeleton_run_identity": parent.get("run_identity"),
            "representative_count": len(representatives),
        },
        "system": {
            "p_atom_id_1based": p_index + 1,
            "c_atom_id_1based": c_index + 1,
            "organic_atom_ids_1based": [index + 1 for index in organic_indices],
            "fixed_head_atom_ids_1based": [index + 1 for index in head_indices],
            "atom_count": len(common_symbols),
        },
        "summary": {
            "skeleton_count": len(representatives),
            "phase_count_per_skeleton": len(phase_angles),
            "candidate_count": len(records),
        },
        "output": {
            "candidate_ensemble": str(ensemble_path),
            "candidate_ensemble_sha256": sha256(ensemble_path),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path, ensemble_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skeleton-postprocess-manifest", type=Path, required=True,
                        help="Completed full sam-phosphonate-skeleton-postprocess-v1 manifest")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New or empty output directory")
    parser.add_argument("--step-deg", type=float, default=30.0,
                        help="Unique P-C axial phase spacing; must divide 360 (default 30 degrees)")
    args = parser.parse_args()
    try:
        manifest, ensemble = generate_rolls(
            args.skeleton_postprocess_manifest, args.output_dir, step_deg=args.step_deg
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"P-C roll generation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"manifest": str(manifest), "candidate_ensemble": str(ensemble)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
