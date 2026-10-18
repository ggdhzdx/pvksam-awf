#!/usr/bin/env python3
"""Select one surface-safe P-C axial roll per skeleton by MMFF94s energy.

中文：对已经通过高度和周期性表面碰撞筛选的 P-C 轴向方位，使用同一 H0
分子图计算 MMFF94s 单点能；每个骨架只保留能量最低的一个方位。本步骤不优化坐标，
所得气相分子内能不替代表面 MACE 弛豫或吸附能。
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np


ROLL_ID = re.compile(r"^(?P<skeleton>.+)__pc-roll-(?P<angle>-?\d+(?:\.\d+)?)deg$")
SAFE_STATUS = "passed_cpu_collision_screen"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, label: str) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be a JSON object")
    return value


def parse_roll_identity(record: dict) -> tuple[str, float]:
    candidate_id = record.get("candidate_id")
    if not isinstance(candidate_id, str):
        raise ValueError("every candidate must declare candidate_id")
    match = ROLL_ID.fullmatch(candidate_id)
    if match is None:
        raise ValueError(
            "candidate_id must end in '__pc-roll-<angle>deg' and contain its skeleton ID: "
            f"{candidate_id}"
        )
    skeleton_id = match.group("skeleton")
    roll_deg = float(match.group("angle"))
    if not math.isfinite(roll_deg):
        raise ValueError(f"non-finite roll angle in {candidate_id}")
    if record.get("skeleton_id", skeleton_id) != skeleton_id:
        raise ValueError(f"skeleton_id disagrees with candidate_id: {candidate_id}")
    declared_angle = record.get("pc_roll_deg", roll_deg)
    if not isinstance(declared_angle, (int, float)) or not math.isclose(
        float(declared_angle), roll_deg, rel_tol=0.0, abs_tol=1.0e-9
    ):
        raise ValueError(f"pc_roll_deg disagrees with candidate_id: {candidate_id}")
    return skeleton_id, roll_deg


def score_candidates(
    ensemble_path: Path,
    cpu_manifest_path: Path,
    template_sdf: Path,
    output_dir: Path,
    *,
    passed_status: str = SAFE_STATUS,
) -> tuple[Path, Path]:
    """Score the exact one-to-one candidate set and write one selected pose per skeleton."""
    # Imported only after CLI paths are resolved so this script uses the locked
    # PVKSAM adapter beside itself, never an unrelated project copy.
    from rdkit import Chem, rdBase
    from rdkit.Chem import AllChem

    script_directory = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_directory))
    import fast_ff_prefilter

    ensemble_path = ensemble_path.resolve()
    cpu_manifest_path = cpu_manifest_path.resolve()
    template_sdf = template_sdf.resolve()
    output_dir = output_dir.resolve()
    for path, label in (
        (ensemble_path, "candidate ensemble"),
        (cpu_manifest_path, "CPU screen manifest"),
        (template_sdf, "H0 topology SDF"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    ensemble = read_json(ensemble_path, "candidate ensemble")
    cpu_manifest = read_json(cpu_manifest_path, "CPU screen manifest")
    screened_input = cpu_manifest.get("parameters", {}).get("candidate_ensemble")
    if not isinstance(screened_input, dict):
        raise ValueError("CPU screen manifest does not bind a candidate ensemble")
    screened_path = Path(screened_input.get("path", "")).expanduser().resolve()
    if (
        screened_path != ensemble_path
        or screened_input.get("sha256") != sha256(ensemble_path)
    ):
        raise ValueError("CPU screen manifest does not hash-bind this exact candidate ensemble")
    symbols = ensemble.get("symbols")
    rows = ensemble.get("records")
    cpu_rows = cpu_manifest.get("records")
    if (
        not isinstance(symbols, list)
        or not symbols
        or not all(isinstance(symbol, str) for symbol in symbols)
        or not isinstance(rows, list)
        or not rows
        or not isinstance(cpu_rows, list)
    ):
        raise ValueError("ensemble symbols/records or CPU manifest records are missing")

    candidate_by_id: dict[str, dict] = {}
    by_skeleton: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("candidate records must be JSON objects")
        skeleton_id, roll_deg = parse_roll_identity(row)
        candidate_id = row["candidate_id"]
        positions = np.asarray(row.get("positions_A"), dtype=float)
        if positions.shape != (len(symbols), 3) or not np.isfinite(positions).all():
            raise ValueError(f"invalid positions_A array for {candidate_id}")
        if candidate_id in candidate_by_id:
            raise ValueError(f"duplicate candidate_id: {candidate_id}")
        candidate = {
            "candidate_id": candidate_id,
            "skeleton_id": skeleton_id,
            "pc_roll_deg": roll_deg,
            "positions_A": positions,
        }
        candidate_by_id[candidate_id] = candidate
        by_skeleton[skeleton_id].append(candidate)

    cpu_by_id: dict[str, dict] = {}
    for row in cpu_rows:
        if not isinstance(row, dict) or not isinstance(row.get("candidate_id"), str):
            raise ValueError("every CPU screen record must declare candidate_id")
        candidate_id = row["candidate_id"]
        if candidate_id in cpu_by_id:
            raise ValueError(f"duplicate CPU screen candidate_id: {candidate_id}")
        cpu_by_id[candidate_id] = row
    if set(cpu_by_id) != set(candidate_by_id):
        missing = sorted(set(candidate_by_id) - set(cpu_by_id))
        extra = sorted(set(cpu_by_id) - set(candidate_by_id))
        raise ValueError(
            f"CPU screen candidate set differs from ensemble; missing={missing[:5]}, extra={extra[:5]}"
        )

    unsafe = [
        candidate_id
        for candidate_id, record in cpu_by_id.items()
        if record.get("status") != passed_status
    ]
    safe_by_skeleton: dict[str, list[dict]] = defaultdict(list)
    for candidate_id, candidate in candidate_by_id.items():
        if cpu_by_id[candidate_id].get("status") == passed_status:
            safe_by_skeleton[candidate["skeleton_id"]].append(candidate)
    missing_safe = sorted(set(by_skeleton) - set(safe_by_skeleton))
    if missing_safe:
        raise ValueError(
            "no surface-safe P-C roll remains for skeleton(s): " + ", ".join(missing_safe[:10])
        )

    molecule, head_indices = fast_ff_prefilter._h0_rdkit_template(
        template_sdf, symbols
    )
    if [atom.GetSymbol() for atom in molecule.GetAtoms()] != symbols:
        raise ValueError("H0 SDF atom order does not match candidate ensemble symbols")
    if molecule.GetNumConformers():
        molecule.RemoveAllConformers()
    properties = AllChem.MMFFGetMoleculeProperties(molecule, mmffVariant="MMFF94s")
    if properties is None or not AllChem.MMFFHasAllMoleculeParams(molecule):
        raise ValueError("MMFF94s parameters are unavailable for the H0 topology")

    energy_records: list[dict] = []
    ranked_by_skeleton: dict[str, list[dict]] = {}
    for skeleton_id in sorted(safe_by_skeleton):
        ranked = []
        for candidate in safe_by_skeleton[skeleton_id]:
            mol = Chem.Mol(molecule)
            conformer = Chem.Conformer(len(symbols))
            for atom_index, xyz in enumerate(candidate["positions_A"]):
                conformer.SetAtomPosition(atom_index, xyz.tolist())
            mol.AddConformer(conformer, assignId=True)
            force_field = AllChem.MMFFGetMoleculeForceField(
                mol, properties, confId=0, ignoreInterfragInteractions=False
            )
            if force_field is None:
                raise ValueError(
                    f"could not construct MMFF94s evaluator for {candidate['candidate_id']}"
                )
            energy = float(force_field.CalcEnergy())
            if not math.isfinite(energy):
                raise ValueError(f"non-finite MMFF94s energy for {candidate['candidate_id']}")
            scored = {
                "candidate_id": candidate["candidate_id"],
                "skeleton_id": skeleton_id,
                "pc_roll_deg": candidate["pc_roll_deg"],
                "mmff94s_single_point_energy_kcal_mol": energy,
            }
            ranked.append(scored)
            energy_records.append(scored)
        ranked_by_skeleton[skeleton_id] = sorted(
            ranked,
            key=lambda record: (
                record["mmff94s_single_point_energy_kcal_mol"],
                record["pc_roll_deg"],
                record["candidate_id"],
            ),
        )

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"selection output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = output_dir / "selected-candidate-ensemble.json"
    manifest_path = output_dir / "single-pc-roll-selection.json"
    if selected_path.exists() or manifest_path.exists():
        raise FileExistsError(f"selection output already exists: {output_dir}")
    selected = []
    selected_records = []
    for skeleton_id, ranked in ranked_by_skeleton.items():
        chosen = dict(ranked[0])
        chosen["surface_safe_roll_count"] = len(ranked)
        chosen["next_best_energy_gap_kcal_mol"] = (
            ranked[1]["mmff94s_single_point_energy_kcal_mol"]
            - ranked[0]["mmff94s_single_point_energy_kcal_mol"]
            if len(ranked) > 1
            else None
        )
        selected_records.append(chosen)
        pose = candidate_by_id[chosen["candidate_id"]]
        selected.append(
            {
                "candidate_id": chosen["candidate_id"],
                "skeleton_id": skeleton_id,
                "pc_roll_deg": chosen["pc_roll_deg"],
                "positions_A": pose["positions_A"].tolist(),
            }
        )
    selected_ensemble = {
        "schema": "pvksam-fixed-site-candidate-ensemble-v1",
        "candidate_source": "one lowest-MMFF94s-single-point-energy surface-safe P-C roll per skeleton",
        "symbols": symbols,
        "records": selected,
        "provenance": {
            "selection_manifest": manifest_path.name,
            "selection_method": "minimum MMFF94s single-point energy among CPU-screen-passing rolls, independently per skeleton",
        },
    }
    manifest = {
        "schema": "pvksam-single-pc-roll-selection-v1",
        "status": "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": {
            "selection": "lowest MMFF94s single-point energy among rolls that passed the declared Z-height and periodic surface-collision gates",
            "force_field": "RDKit MMFF94s",
            "energy_unit": "kcal/mol",
            "coordinates_optimized_in_this_stage": False,
            "fixed_head_atom_indices_0based": head_indices,
            "tie_break": "smaller numeric P-C roll angle, then candidate_id",
            "limitation": "isolated intramolecular energy does not include ITO-induced stabilization; use only to reduce each skeleton to one starting pose before surface MACE relaxation",
        },
        "sources": {
            "candidate_ensemble": str(ensemble_path),
            "candidate_ensemble_sha256": sha256(ensemble_path),
            "cpu_screen_manifest": str(cpu_manifest_path),
            "cpu_screen_manifest_sha256": sha256(cpu_manifest_path),
            "h0_topology_sdf": str(template_sdf),
            "h0_topology_sdf_sha256": sha256(template_sdf),
            "selector_script": str(Path(__file__).resolve()),
            "selector_script_sha256": sha256(Path(__file__).resolve()),
            "h0_template_builder": str(Path(fast_ff_prefilter.__file__).resolve()),
            "h0_template_builder_sha256": sha256(Path(fast_ff_prefilter.__file__).resolve()),
            "rdkit_version": rdBase.rdkitVersion,
        },
        "summary": {
            "skeleton_count": len(by_skeleton),
            "input_roll_count": len(candidate_by_id),
            "surface_safe_roll_count": sum(map(len, safe_by_skeleton.values())),
            "screen_rejected_roll_count": len(unsafe),
            "selected_roll_count": len(selected_records),
            "surface_safe_roll_count_min": min(map(len, safe_by_skeleton.values())),
            "surface_safe_roll_count_max": max(map(len, safe_by_skeleton.values())),
        },
        "selected": selected_records,
        "all_surface_safe_roll_energies": sorted(
            energy_records,
            key=lambda record: (
                record["skeleton_id"],
                record["mmff94s_single_point_energy_kcal_mol"],
                record["pc_roll_deg"],
            ),
        ),
        "outputs": {
            "selected_candidate_ensemble": str(selected_path),
        },
    }
    selected_path.write_text(
        json.dumps(selected_ensemble, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest["outputs"]["selected_candidate_ensemble_sha256"] = sha256(selected_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path, selected_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-ensemble", type=Path, required=True)
    parser.add_argument("--cpu-manifest", type=Path, required=True)
    parser.add_argument("--template-sdf", type=Path, required=True,
                        help="Reviewed explicit-H source SDF used to derive the exact H0 graph")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New or empty directory for the selection receipt and one-pose ensemble")
    args = parser.parse_args()
    try:
        manifest, selected = score_candidates(
            args.candidate_ensemble,
            args.cpu_manifest,
            args.template_sdf,
            args.output_dir,
        )
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"single P-C roll selection failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"manifest": str(manifest), "selected_ensemble": str(selected)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
