#!/usr/bin/env python3
"""Fixed-head SAM force-field optimization and skeleton geometry postprocessing.

This stage deliberately has a separate dependency boundary from the ASE/MACE
scanner: RDKit optimizes the isolated SAM geometry while P and its three
phosphonate O atoms remain fixed.  Substrate collision auditing is performed
later by the maintained ASE collision-audit owner.
The explicit skeleton-postprocess mode uses the ASE geometry runtime and can
query a separate RDKit interpreter once for the reviewed source bond graph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


SCHEMA_VERSION = 2
DEFAULT_MAX_ITERATIONS = 75
DEFAULT_RMSD_TOLERANCE_A = 1.0
DEFAULT_MOVABLE_FMAX_EV_A = 0.03
KCAL_MOL_PER_EV = 23.06054783061903
FIXED_HEAD_DISPLACEMENT_TOLERANCE_A = 1.0e-12

_WORKER_TEMPLATE = None
_WORKER_HEAD_INDICES = None
_WORKER_SAM_SYMBOLS = None
_WORKER_SAM_START = None
_WORKER_SAM_COUNT = None
_WORKER_MAX_ITERATIONS = None
_WORKER_NON_HEAD_INDICES = None
_WORKER_FORCE_FIELD = None
_WORKER_MOVABLE_FMAX_EV_A = None


def _force_field_slug(name: str) -> str:
    """Return the stable output/schema token for a force-field choice."""

    normalized = str(name).strip().lower()
    aliases = {
        "uff": "uff",
        "rdkit uff": "uff",
        "mmff94": "mmff94",
        "rdkit mmff94": "mmff94",
        "mmff94s": "mmff94s",
        "rdkit mmff94s": "mmff94s",
    }
    if normalized not in aliases:
        raise ValueError(f"unsupported force field: {name}")
    return aliases[normalized]


def _force_field_label(slug: str) -> str:
    labels = {
        "uff": "RDKit UFF",
        "mmff94": "RDKit MMFF94",
        "mmff94s": "RDKit MMFF94s",
    }
    try:
        return labels[_force_field_slug(slug)]
    except KeyError as exc:
        raise ValueError(f"unsupported force field slug: {slug}") from exc


def _force_field_api_variant(slug: str) -> str:
    """Map CLI slugs to RDKit's case-sensitive official variant names.

    Passing ``mmff94s`` directly to RDKit silently selects MMFF94 in the
    validated runtime.  The slug identifies files; it is never an API value.
    """

    return {"uff": "UFF", "mmff94": "MMFF94", "mmff94s": "MMFF94s"}[
        _force_field_slug(slug)
    ]


def _force_field_record_values(record: dict) -> tuple[str, float]:
    """Read new generic or historical UFF energy fields from a record."""

    if "force_field_energy_kcal_mol" in record:
        return str(record.get("force_field", "unknown")), float(
            record["force_field_energy_kcal_mol"]
        )
    # Historical runs mislabeled RDKit CalcEnergy() (kcal/mol) as eV.  Keep
    # their records readable, but expose the corrected unit to new manifests.
    if "uff_energy_eV" in record:
        return "uff", float(record["uff_energy_eV"])
    raise ValueError("force-field record has no recognized energy field")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _json_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _read_extxyz_bytes(data: bytes) -> tuple[list[str], np.ndarray, str]:
    lines = data.decode("utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError("EXTXYZ has no atom block")
    try:
        count = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError("EXTXYZ first line is not an atom count") from exc
    if count <= 0 or len(lines) < count + 2:
        raise ValueError("EXTXYZ atom block is truncated")
    symbols: list[str] = []
    positions = np.empty((count, 3), dtype=float)
    for index, line in enumerate(lines[2 : count + 2]):
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"EXTXYZ atom line {index + 1} is malformed")
        symbols.append(fields[0])
        positions[index] = [float(value) for value in fields[1:4]]
    if not np.all(np.isfinite(positions)):
        raise ValueError("EXTXYZ contains non-finite coordinates")
    return symbols, positions, lines[1]


def _write_extxyz(path: Path, symbols: list[str], positions: np.ndarray, comment: str) -> None:
    positions = np.asarray(positions, dtype=float)
    if positions.shape != (len(symbols), 3):
        raise ValueError("EXTXYZ symbols and coordinates have different sizes")
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{len(symbols)}\n")
        handle.write(f"{comment}\n")
        for symbol, (x, y, z) in zip(symbols, positions):
            handle.write(f"{symbol:<2s} {x: .10f} {y: .10f} {z: .10f}\n")


def _normalize_angle(value: float) -> float:
    value = float(value)
    normalized = (value + 180.0) % 360.0 - 180.0
    if abs(normalized + 180.0) < 1.0e-8:
        return -180.0
    return float(normalized)


def _h0_rdkit_template(template_sdf: Path, h0_symbols: list[str]):
    """Build the RDKit H0 template and return its fixed P/O indices."""

    from rdkit import Chem

    source = Chem.MolFromMolFile(str(template_sdf), removeHs=False)
    if source is None:
        raise ValueError(f"RDKit could not read template SDF: {template_sdf}")
    phosphorus = [atom.GetIdx() for atom in source.GetAtoms() if atom.GetSymbol() == "P"]
    if len(phosphorus) != 1:
        raise ValueError(f"Expected one P atom in template; found {len(phosphorus)}")
    p_index = phosphorus[0]
    oxygen = sorted(
        neighbor.GetIdx()
        for neighbor in source.GetAtomWithIdx(p_index).GetNeighbors()
        if neighbor.GetSymbol() == "O"
    )
    if len(oxygen) != 3:
        raise ValueError(f"Expected exactly three P-bound O atoms; found {len(oxygen)}")

    acidic_hydrogens = []
    acidic_oxygen = []
    for oxygen_index in oxygen:
        atom = source.GetAtomWithIdx(oxygen_index)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetSymbol() == "H":
                acidic_hydrogens.append(neighbor.GetIdx())
                acidic_oxygen.append(oxygen_index)

    original_symbols = [atom.GetSymbol() for atom in source.GetAtoms()]
    retained_indices = [
        index for index in range(source.GetNumAtoms()) if index not in acidic_hydrogens
    ]
    retained_symbols = [original_symbols[index] for index in retained_indices]
    if retained_symbols != list(h0_symbols):
        if original_symbols != list(h0_symbols):
            raise ValueError(
                "Template SDF atom order does not match the CPU-screen H0 atom order"
            )
        acidic_hydrogens = []
        acidic_oxygen = []
        retained_indices = list(range(source.GetNumAtoms()))

    old_to_new = {old: new for new, old in enumerate(retained_indices)}

    if acidic_hydrogens:
        editable = Chem.RWMol(source)
        for index in sorted(acidic_hydrogens, reverse=True):
            editable.RemoveAtom(index)
        molecule = editable.GetMol()
        for old_oxygen in acidic_oxygen:
            molecule.GetAtomWithIdx(old_to_new[old_oxygen]).SetFormalCharge(-1)
        Chem.SanitizeMol(molecule)
    else:
        molecule = Chem.Mol(source)

    if [atom.GetSymbol() for atom in molecule.GetAtoms()] != list(h0_symbols):
        raise ValueError("RDKit H0 template symbols do not match CPU-screen symbols")
    head_indices = [old_to_new.get(p_index, p_index)] if acidic_hydrogens else [p_index]
    head_indices.extend(
        old_to_new.get(index, index) for index in oxygen
    )
    if len(set(head_indices)) != 4:
        raise ValueError("P/O fixed-head atom mapping is not unique")
    return molecule, sorted(head_indices)


def _worker_init(
    template_sdf: str,
    h0_symbols: list[str],
    sam_start: int,
    sam_count: int,
    max_iterations: int,
    force_field: str,
    movable_fmax_eV_A: float | None = None,
) -> None:
    global _WORKER_TEMPLATE, _WORKER_HEAD_INDICES, _WORKER_SAM_SYMBOLS
    global _WORKER_SAM_START, _WORKER_SAM_COUNT, _WORKER_MAX_ITERATIONS
    global _WORKER_NON_HEAD_INDICES, _WORKER_FORCE_FIELD, _WORKER_MOVABLE_FMAX_EV_A
    from rdkit import Chem

    _WORKER_TEMPLATE, _WORKER_HEAD_INDICES = _h0_rdkit_template(
        Path(template_sdf), h0_symbols
    )
    _WORKER_SAM_SYMBOLS = list(h0_symbols)
    _WORKER_SAM_START = int(sam_start)
    _WORKER_SAM_COUNT = int(sam_count)
    _WORKER_MAX_ITERATIONS = int(max_iterations)
    _WORKER_FORCE_FIELD = _force_field_slug(force_field)
    _WORKER_MOVABLE_FMAX_EV_A = movable_fmax_eV_A
    _WORKER_NON_HEAD_INDICES = np.asarray(
        [
            index
            for index, symbol in enumerate(h0_symbols)
            if index not in _WORKER_HEAD_INDICES
        ],
        dtype=int,
    )
    # Force one BLAS thread per process; parallelism is across candidates.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"


def _minimize_force_field(force_field) -> tuple[int | None, int, int]:
    """Minimize with a bounded budget and an independent movable-atom gate.

    RDKit exposes a termination code but no actual iteration count.  The
    recorded budget is therefore the sum of requested block limits, never
    presented as the number of iterations the optimizer actually took.
    """

    if _WORKER_MOVABLE_FMAX_EV_A is None:
        return int(force_field.Minimize(maxIts=_WORKER_MAX_ITERATIONS)), 1, int(
            _WORKER_MAX_ITERATIONS
        )
    target_kcal_mol_A = float(_WORKER_MOVABLE_FMAX_EV_A) * KCAL_MOL_PER_EV
    initial_gradient = np.asarray(force_field.CalcGrad(), dtype=float).reshape((-1, 3))
    if float(
        np.max(np.linalg.norm(initial_gradient[_WORKER_NON_HEAD_INDICES], axis=1))
    ) <= target_kcal_mol_A:
        return None, 0, 0
    requested_budget = 0
    blocks = 0
    termination = 1
    while requested_budget < _WORKER_MAX_ITERATIONS:
        block_limit = min(250, _WORKER_MAX_ITERATIONS - requested_budget)
        termination = int(
            force_field.Minimize(
                maxIts=block_limit,
                forceTol=1.0e-4,
                energyTol=1.0e-6,
            )
        )
        requested_budget += block_limit
        blocks += 1
        gradient = np.asarray(force_field.CalcGrad(), dtype=float).reshape((-1, 3))
        movable_max = float(
            np.max(np.linalg.norm(gradient[_WORKER_NON_HEAD_INDICES], axis=1))
        )
        if movable_max <= target_kcal_mol_A:
            break
        if termination not in (0, 1):
            break
    return termination, blocks, requested_budget


def _optimize_one(payload: dict) -> dict:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    candidate_path = Path(payload["structure"])
    try:
        data = candidate_path.read_bytes()
        observed_hash = _sha256_bytes(data)
        if observed_hash != payload["structure_sha256"]:
            raise ValueError(
                f"candidate hash changed: expected {payload['structure_sha256']}, "
                f"observed {observed_hash}"
            )
        symbols, full_positions, comment = _read_extxyz_bytes(data)
        start = _WORKER_SAM_START
        stop = start + _WORKER_SAM_COUNT
        if symbols[start:stop] != _WORKER_SAM_SYMBOLS:
            raise ValueError("candidate SAM atom order differs from H0 template")
        sam_positions = np.asarray(full_positions[start:stop], dtype=float)

        molecule = Chem.Mol(_WORKER_TEMPLATE)
        conformer = molecule.GetConformer()
        for index, (x, y, z) in enumerate(sam_positions):
            conformer.SetAtomPosition(index, (float(x), float(y), float(z)))
        if _WORKER_FORCE_FIELD == "uff":
            if not AllChem.UFFHasAllMoleculeParams(molecule):
                raise ValueError("RDKit UFF lacks parameters for this H0 molecule")
            force_field = AllChem.UFFGetMoleculeForceField(
                molecule, ignoreInterfragInteractions=True
            )
        else:
            variant = _force_field_api_variant(_WORKER_FORCE_FIELD)
            if not AllChem.MMFFHasAllMoleculeParams(molecule):
                raise ValueError(
                    f"RDKit {_force_field_label(variant)} lacks parameters for this H0 molecule"
                )
            properties = AllChem.MMFFGetMoleculeProperties(
                molecule, mmffVariant=variant
            )
            if properties is None:
                raise ValueError(
                    f"RDKit could not create {_force_field_label(variant)} properties"
                )
            force_field = AllChem.MMFFGetMoleculeForceField(
                molecule, properties, ignoreInterfragInteractions=True
            )
            if force_field is None:
                raise ValueError(
                    f"RDKit could not create {_force_field_label(variant)} force field"
                )
        for index in _WORKER_HEAD_INDICES:
            force_field.AddFixedPoint(int(index))
        force_field.Initialize()
        initial_energy = float(force_field.CalcEnergy())
        termination_code, minimize_blocks, requested_iteration_budget = (
            _minimize_force_field(force_field)
        )
        energy = float(force_field.CalcEnergy())
        gradient = np.asarray(force_field.CalcGrad(), dtype=float).reshape((-1, 3))
        max_gradient = float(np.max(np.linalg.norm(gradient, axis=1)))
        max_gradient_non_head = float(
            np.max(np.linalg.norm(gradient[_WORKER_NON_HEAD_INDICES], axis=1))
        )
        optimized = np.asarray(
            [conformer.GetAtomPosition(index) for index in range(_WORKER_SAM_COUNT)],
            dtype=float,
        )
        if not np.all(np.isfinite(optimized)):
            raise ValueError(
                f"{_force_field_label(_WORKER_FORCE_FIELD)} produced non-finite coordinates"
            )
        if not np.isfinite(energy) or not np.all(np.isfinite(gradient)):
            raise ValueError("force field produced non-finite energy or gradient")
        head_displacements = np.linalg.norm(
            optimized[_WORKER_HEAD_INDICES] - sam_positions[_WORKER_HEAD_INDICES],
            axis=1,
        )
        head_max_displacement = float(np.max(head_displacements))
        if head_max_displacement > FIXED_HEAD_DISPLACEMENT_TOLERANCE_A:
            raise ValueError(
                "fixed P/O atoms moved by "
                f"{head_max_displacement:.12g} A (limit "
                f"{FIXED_HEAD_DISPLACEMENT_TOLERANCE_A:g} A)"
            )
        movable_fmax_eV_A = max_gradient_non_head / KCAL_MOL_PER_EV
        converged = (
            termination_code == 0
            if _WORKER_MOVABLE_FMAX_EV_A is None
            else movable_fmax_eV_A <= float(_WORKER_MOVABLE_FMAX_EV_A)
        )
        return {
            "status": "passed",
            "grid_index": int(payload["grid_index"]),
            "safe_rank": int(payload["safe_rank"]),
            "candidate_key": list(payload["candidate_key"]),
            "start_dihedrals_deg": list(payload["start_dihedrals_deg"]),
            "structure": str(candidate_path),
            "structure_sha256": observed_hash,
            "comment": comment,
            "positions": optimized.tolist(),
            "force_field": _WORKER_FORCE_FIELD,
            "force_field_actual_variant": _force_field_api_variant(_WORKER_FORCE_FIELD),
            # RDKit CalcEnergy/CalcGrad use kcal/mol and kcal/mol/A.
            "force_field_energy_kcal_mol": energy,
            "force_field_initial_energy_kcal_mol": initial_energy,
            "force_field_energy_change_kcal_mol": energy - initial_energy,
            "force_field_not_converged": (
                termination_code != 0 if termination_code is not None else None
            ),
            "force_field_termination_code": termination_code,
            "force_field_converged": bool(converged),
            "force_field_minimize_block_count": minimize_blocks,
            "force_field_requested_iteration_budget": requested_iteration_budget,
            "force_field_max_gradient_kcal_mol_A": max_gradient,
            "force_field_max_gradient_non_head_kcal_mol_A": max_gradient_non_head,
            "force_field_max_gradient_non_head_eV_A": movable_fmax_eV_A,
            "fixed_head_max_displacement_A": head_max_displacement,
            "fixed_head_displacements_A": head_displacements.tolist(),
        }
    except Exception as exc:  # retain candidate-level evidence and continue
        return {
            "status": "failed",
            "grid_index": int(payload["grid_index"]),
            "safe_rank": int(payload["safe_rank"]),
            "candidate_key": list(payload["candidate_key"]),
            "start_dihedrals_deg": list(payload["start_dihedrals_deg"]),
            "structure": str(candidate_path),
            "structure_sha256": payload["structure_sha256"],
            "error": f"{type(exc).__name__}: {exc}",
        }


def _complete_link_clusters(
    positions: np.ndarray,
    heavy_indices: np.ndarray,
    tolerance_A: float,
) -> tuple[list[list[int]], list[int]]:
    """Deterministic complete-link clustering in the fixed surface frame."""

    if tolerance_A <= 0 or not np.isfinite(tolerance_A):
        raise ValueError("RMSD tolerance must be positive and finite")
    clusters: list[list[int]] = []
    assignments: list[int] = [-1] * len(positions)
    for index, candidate in enumerate(positions):
        candidate_heavy = candidate[heavy_indices]
        assigned = False
        for cluster_index, members in enumerate(clusters):
            member_positions = positions[np.asarray(members, dtype=int)][:, heavy_indices]
            differences = member_positions - candidate_heavy[None, :, :]
            rmsd = np.sqrt(np.mean(differences * differences, axis=(1, 2)))
            if bool(np.all(rmsd <= tolerance_A + 1.0e-12)):
                members.append(index)
                assignments[index] = cluster_index
                assigned = True
                break
        if not assigned:
            assignments[index] = len(clusters)
            clusters.append([index])
        if (index + 1) % 1000 == 0:
            print(
                f"clustered {index + 1}/{len(positions)}; "
                f"clusters={len(clusters)}",
                flush=True,
            )
    return clusters, assignments


def _complete_link_fixed_site_clusters(
    positions: np.ndarray,
    heavy_indices: np.ndarray,
    tolerance_A: float,
) -> tuple[list[list[int]], list[int]]:
    """Use the fixed-site contract: per-atom 3D RMSD, no molecular fit."""

    if tolerance_A <= 0 or not np.isfinite(tolerance_A):
        raise ValueError("fixed-site RMSD tolerance must be positive and finite")
    clusters: list[list[int]] = []
    assignments: list[int] = [-1] * len(positions)
    for index, candidate in enumerate(positions):
        candidate_heavy = candidate[heavy_indices]
        assigned = False
        for cluster_index, members in enumerate(clusters):
            member_positions = positions[np.asarray(members, dtype=int)][:, heavy_indices]
            differences = member_positions - candidate_heavy[None, :, :]
            rmsd = np.sqrt(np.mean(np.sum(differences * differences, axis=2), axis=1))
            if bool(np.all(rmsd <= tolerance_A + 1.0e-12)):
                members.append(index)
                assignments[index] = cluster_index
                assigned = True
                break
        if not assigned:
            assignments[index] = len(clusters)
            clusters.append([index])
        if (index + 1) % 100 == 0:
            print(
                f"fixed-site clustered {index + 1}/{len(positions)}; "
                f"clusters={len(clusters)}",
                flush=True,
            )
    return clusters, assignments


def _intramolecular_collision(
    positions: np.ndarray,
    symbols: list[str],
    bond_pairs: set[tuple[int, int]],
) -> dict | None:
    """Apply the maintained H-H/H-heavy/heavy-heavy post-FF gate."""

    positions = np.asarray(positions, dtype=float)
    differences = positions[:, None, :] - positions[None, :, :]
    distances = np.linalg.norm(differences, axis=2)
    upper = np.triu(np.ones(distances.shape, dtype=bool), k=1)
    for left, right in bond_pairs:
        upper[int(left), int(right)] = False
    thresholds = np.full(distances.shape, 1.8, dtype=float)
    for index, left in enumerate(symbols):
        for other, right in enumerate(symbols):
            if left == "H" and right == "H":
                thresholds[index, other] = 1.2
            elif left == "H" or right == "H":
                thresholds[index, other] = 1.4
    collisions = np.argwhere(upper & (distances < thresholds))
    if not len(collisions):
        return None
    left, right = (int(value) for value in collisions[0])
    return {
        "atom_ids_1based": [left + 1, right + 1],
        "elements": [symbols[left], symbols[right]],
        "distance_A": float(distances[left, right]),
        "threshold_A": float(thresholds[left, right]),
    }


def _recluster_existing_ff_manifest(args: argparse.Namespace) -> int:
    """Apply the post-FF intramolecular gate without repeating optimization."""

    ff_manifest_path = Path(args.recluster_ff_manifest).expanduser().resolve()
    if not ff_manifest_path.is_file():
        raise ValueError(f"fast-FF manifest does not exist: {ff_manifest_path}")
    source_manifest = json.loads(ff_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("status") != "passed_ff_pending_substrate_audit":
        raise ValueError("recluster input is not a completed isolated FF stage")
    source_force_field = _force_field_slug(
        source_manifest.get("parameters", {}).get("force_field", "uff")
    )
    records_path = Path(source_manifest["artifacts"]["optimized_records"]).resolve()
    records = [json.loads(line) for line in records_path.read_text().splitlines() if line]
    if not records or any(item.get("status") != "passed" for item in records):
        raise ValueError("recluster input contains missing or failed FF records")
    template_sdf = Path(source_manifest["sources"]["template_sdf"]).resolve()
    h0_symbols = list(source_manifest["system"]["sam_h0_formula_symbols"])
    template, head_indices = _h0_rdkit_template(template_sdf, h0_symbols)
    bond_pairs = {
        tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
        for bond in template.GetBonds()
    }
    valid_records = []
    invalid_records = []
    positions = []
    for record in records:
        path = Path(record["optimized_structure"]).resolve()
        if not path.is_file() or _sha256_path(path) != record["optimized_structure_sha256"]:
            raise ValueError(f"optimized FF structure hash changed: {path}")
        symbols, coordinates, _ = _read_extxyz_bytes(path.read_bytes())
        if symbols != h0_symbols:
            raise ValueError(f"optimized FF symbol order changed: {path}")
        collision = _intramolecular_collision(coordinates, symbols, bond_pairs)
        updated = dict(record)
        if collision is not None:
            updated["status"] = "rejected_ff_intramolecular_collision"
            updated["ff_intramolecular_collision"] = collision
            invalid_records.append(updated)
            continue
        updated["ff_intramolecular_collision"] = None
        valid_records.append(updated)
        positions.append(coordinates)
    if not valid_records:
        raise RuntimeError("post-FF intramolecular gate rejected every candidate")
    positions_array = np.asarray(positions, dtype=float)
    heavy_indices = np.asarray(
        [
            index
            for index, symbol in enumerate(h0_symbols)
            if symbol != "H" and index not in head_indices
        ],
        dtype=int,
    )
    clusters, assignments = _complete_link_clusters(
        positions_array, heavy_indices, float(args.rmsd_tolerance_A)
    )
    for index, record in enumerate(valid_records):
        record["cluster_index"] = int(assignments[index]) + 1

    cluster_records = []
    for cluster_index, member_indices in enumerate(clusters, 1):
        ordered = sorted(
            member_indices,
            key=lambda index: (
                -float(valid_records[index]["z_min_nonpo3_all_A"]),
                int(valid_records[index]["safe_rank"]),
            ),
        )
        representative_index = ordered[0]
        representative = valid_records[representative_index]
        cluster_records.append(
            {
                "cluster_index": cluster_index,
                "member_count": len(member_indices),
                "member_record_indices": [int(index) for index in member_indices],
                "member_grid_indices": [int(valid_records[index]["grid_index"]) for index in member_indices],
                "representative_record_index": int(representative_index),
                "representative_grid_index": int(representative["grid_index"]),
                "representative_structure": representative["optimized_structure"],
                "representative_structure_sha256": representative["optimized_structure_sha256"],
                "representative_candidate_key": list(representative["candidate_key"]),
                "representative_safe_rank": int(representative["safe_rank"]),
                "representative_z_min_nonpo3_all_A": float(representative["z_min_nonpo3_all_A"]),
                "representative_z_min_nonpo3_heavy_A": float(representative["z_min_nonpo3_heavy_A"]),
                "representative_force_field": source_force_field,
                "representative_force_field_energy_kcal_mol": _force_field_record_values(
                    representative
                )[1],
            }
        )
    identity = _json_hash(
        {
            "source_manifest": str(ff_manifest_path),
            "source_manifest_sha256": _sha256_path(ff_manifest_path),
            "script": str(Path(__file__).resolve()),
            "script_sha256": _sha256_path(Path(__file__).resolve()),
            "post_ff_gate": "H-H 1.2 A; H-heavy 1.4 A; heavy-heavy 1.8 A; bonded pairs excluded",
            "rmsd_tolerance_A": float(args.rmsd_tolerance_A),
        }
    )
    output_root = Path(args.output_root).expanduser().resolve()
    run_directory = output_root / f"fast-{source_force_field}-recluster-{identity[:12]}"
    if run_directory.exists():
        raise ValueError(f"immutable FF recluster directory already exists: {run_directory}")
    stage_directory = run_directory / f"01_{source_force_field}_recluster"
    stage_directory.mkdir(parents=True)
    all_records = valid_records + invalid_records
    records_output = stage_directory / "optimized-records-with-post-gate.jsonl"
    records_output.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in all_records),
        encoding="utf-8",
    )
    clusters_output = stage_directory / "clusters.json"
    clusters_output.write_text(json.dumps(cluster_records, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "schema_version": SCHEMA_VERSION + 1,
        "run_identity": identity,
        "status": "passed_ff_pending_substrate_audit",
        "scope": "complete_endpoint_deduplicated_safe_ensemble_post_ff_geometry_gate",
        "sources": {
            "parent_fast_ff_manifest": str(ff_manifest_path),
            "parent_fast_ff_manifest_sha256": _sha256_path(ff_manifest_path),
            "template_sdf": str(template_sdf),
            "template_sdf_sha256": _sha256_path(template_sdf),
            "script": str(Path(__file__).resolve()),
            "script_sha256": _sha256_path(Path(__file__).resolve()),
        },
        "parameters": {
            "post_ff_intramolecular_collision_gate": "H-H 1.2 A; H-heavy 1.4 A; heavy-heavy 1.8 A; bonded pairs excluded",
            "rmsd_tolerance_A": float(args.rmsd_tolerance_A),
            "rmsd_atoms": "all heavy atoms excluding P and its three bound O atoms",
            "representative_metric": "maximum minimum z over every atom excluding P/O head, including H",
        },
        "system": source_manifest["system"],
        "summary": {
            "parent_force_field_passed_count": len(records),
            "post_ff_geometry_safe_count": len(valid_records),
            "post_ff_intramolecular_collision_rejected_count": len(invalid_records),
            "cluster_count": len(cluster_records),
            "substrate_collision_audit_count": len(cluster_records),
            "substrate_collision_safe_count": None,
        },
        "artifacts": {
            "optimized_records": str(records_output),
            "parent_optimized_records": str(records_path),
            "clusters": str(clusters_output),
        },
        "representatives": cluster_records,
    }
    manifest_path = run_directory / "fast-ff-recluster-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (run_directory / "README.zh-CN.md").write_text(
        "\n".join(
            [
                f"# {_force_field_label(source_force_field)} 结果的分子内几何门和重新聚类",
                "",
                f"- 父阶段 {_force_field_label(source_force_field)} 候选：{len(records)}",
                f"- 通过 {_force_field_label(source_force_field)} 后分子内几何门：{len(valid_records)}",
                f"- 因 {_force_field_label(source_force_field)} 后分子内碰撞剔除：{len(invalid_records)}",
                f"- 1 Å complete-link 簇：{len(cluster_records)}",
                "- 代表选择：除 PO₃ 外所有原子（含 H）的最低 Z 最大者。",
                "- 下一阶段：重新装回固定 ITO Site Instance，并运行维护的周期性碰撞审计。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "stage": "post-FF intramolecular geometry gate and reclustering",
                "status": "passed_pending_substrate_audit",
                "run_directory": str(run_directory),
                "manifest": str(manifest_path),
                "summary": manifest["summary"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


def _recluster_audit_manifest(args: argparse.Namespace) -> int:
    """Cluster structures that already passed the fixed-site substrate audit."""

    audit_manifest_path = Path(args.recluster_audit_manifest).expanduser().resolve()
    if not audit_manifest_path.is_file():
        raise ValueError(f"fixed-site audit manifest does not exist: {audit_manifest_path}")
    audit_manifest = json.loads(audit_manifest_path.read_text(encoding="utf-8"))
    if audit_manifest.get("status") != "passed":
        raise ValueError("audit manifest is not a completed fixed-site audit")
    records = [
        item for item in audit_manifest.get("records", []) if item.get("status") == "passed"
    ]
    if not records:
        raise ValueError("audit manifest contains no collision-safe structures")

    parent_recluster_path = Path(
        audit_manifest["sources"]["fast_ff_manifest"]
    ).expanduser().resolve()
    if not parent_recluster_path.is_file():
        raise ValueError(f"parent FF recluster manifest does not exist: {parent_recluster_path}")
    parent_recluster = json.loads(parent_recluster_path.read_text(encoding="utf-8"))
    parent_ff_path = Path(
        parent_recluster["sources"].get("parent_fast_ff_manifest", "")
    ).expanduser().resolve()
    parent_ff = json.loads(parent_ff_path.read_text(encoding="utf-8")) if parent_ff_path.is_file() else {}
    force_field = _force_field_slug(
        parent_ff.get("parameters", {}).get(
            "force_field", parent_recluster.get("system", {}).get("force_field", "uff")
        )
    )
    h0_path = Path(audit_manifest["sources"]["sam_h0"]).expanduser().resolve()
    h0_symbols, _, _ = _read_extxyz_bytes(h0_path.read_bytes())
    sam_count = len(h0_symbols)
    head_ids = [
        int(value) - 1
        for value in parent_recluster.get("system", {}).get(
            "fixed_head_atom_ids_1based", [25, 26, 27, 28]
        )
    ]
    if any(index < 0 or index >= sam_count for index in head_ids):
        raise ValueError("fixed PO3 head indices are outside the SAM block")
    heavy_indices = np.asarray(
        [index for index, symbol in enumerate(h0_symbols) if symbol != "H" and index not in head_ids],
        dtype=int,
    )

    sam_positions = []
    enriched_records = []
    substrate_count = None
    full_symbols = None
    for item in records:
        source_path = Path(item["accepted_structure"]).expanduser().resolve()
        if not source_path.is_file():
            raise ValueError(f"accepted structure is missing: {source_path}")
        raw = source_path.read_bytes()
        if _sha256_bytes(raw) != item["accepted_structure_sha256"]:
            raise ValueError(f"accepted structure hash changed: {source_path}")
        symbols, coordinates, _ = _read_extxyz_bytes(raw)
        if len(symbols) < sam_count or symbols[-sam_count:] != h0_symbols:
            raise ValueError(f"accepted structure SAM block differs from H0: {source_path}")
        current_substrate_count = len(symbols) - sam_count
        if substrate_count is None:
            substrate_count = current_substrate_count
            full_symbols = symbols
        elif current_substrate_count != substrate_count or symbols != full_symbols:
            raise ValueError("accepted structures do not share one fixed Site Instance")
        sam = coordinates[-sam_count:]
        sam_positions.append(sam)
        enriched = dict(item)
        enriched["second_stage_z_min_nonpo3_all_A"] = float(np.min(sam[[i for i in range(sam_count) if i not in head_ids], 2]))
        enriched_records.append(enriched)

    positions = np.asarray(sam_positions, dtype=float)
    clusters, assignments = _complete_link_fixed_site_clusters(
        positions, heavy_indices, float(args.rmsd_tolerance_A)
    )
    for index, record in enumerate(enriched_records):
        record["second_stage_cluster_index"] = int(assignments[index]) + 1

    identity = _json_hash(
        {
            "source_audit_manifest": str(audit_manifest_path),
            "source_audit_manifest_sha256": _sha256_path(audit_manifest_path),
            "script": str(Path(__file__).resolve()),
            "script_sha256": _sha256_path(Path(__file__).resolve()),
            "rmsd_tolerance_A": float(args.rmsd_tolerance_A),
            "rmsd_atoms": "SAM heavy atoms excluding fixed P and three P-bound O",
            "representative_metric": "maximum minimum z over SAM atoms excluding fixed P/O head",
        }
    )
    output_root = Path(args.output_root).expanduser().resolve()
    run_directory = output_root / f"fast-{force_field}-audit-recluster-{identity[:12]}"
    if run_directory.exists():
        raise ValueError(f"immutable audit recluster directory already exists: {run_directory}")
    representative_directory = run_directory / "representative_structures"
    representative_directory.mkdir(parents=True)

    cluster_records = []
    for cluster_index, member_indices in enumerate(clusters, 1):
        ordered = sorted(
            member_indices,
            key=lambda index: (
                -float(enriched_records[index]["second_stage_z_min_nonpo3_all_A"]),
                int(enriched_records[index]["cluster_index"]),
            ),
        )
        representative_index = ordered[0]
        representative = enriched_records[representative_index]
        source_path = Path(representative["accepted_structure"]).resolve()
        output_path = representative_directory / f"cluster-{cluster_index:05d}.extxyz"
        shutil.copy2(source_path, output_path)
        cluster_records.append(
            {
                "cluster_index": cluster_index,
                "member_count": len(member_indices),
                "member_audit_cluster_indices": [
                    int(enriched_records[index]["cluster_index"]) for index in member_indices
                ],
                "member_structure_paths": [
                    enriched_records[index]["accepted_structure"] for index in member_indices
                ],
                "representative_audit_cluster_index": int(representative["cluster_index"]),
                "representative_structure": str(output_path),
                "representative_structure_sha256": _sha256_path(output_path),
                "representative_source_structure": str(source_path),
                "representative_source_structure_sha256": _sha256_path(source_path),
                "representative_z_min_nonpo3_all_A": float(
                    representative["second_stage_z_min_nonpo3_all_A"]
                ),
            }
        )

    records_output = run_directory / "audit-records-with-second-stage-clusters.jsonl"
    records_output.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in enriched_records),
        encoding="utf-8",
    )
    clusters_output = run_directory / "clusters.json"
    clusters_output.write_text(json.dumps(cluster_records, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "schema_version": SCHEMA_VERSION + 1,
        "run_identity": identity,
        "status": "passed_audit_recluster_pending_mace",
        "scope": "collision_safe_fixed_site_audit_second_stage_clustering",
        "sources": {
            "audit_manifest": str(audit_manifest_path),
            "audit_manifest_sha256": _sha256_path(audit_manifest_path),
            "parent_ff_recluster_manifest": str(parent_recluster_path),
            "parent_ff_recluster_manifest_sha256": _sha256_path(parent_recluster_path),
            "sam_h0": str(h0_path),
            "sam_h0_sha256": _sha256_path(h0_path),
            "script": str(Path(__file__).resolve()),
            "script_sha256": _sha256_path(Path(__file__).resolve()),
        },
        "parameters": {
            "force_field": force_field,
            "rmsd_tolerance_A": float(args.rmsd_tolerance_A),
            "rmsd_method": "deterministic complete-link in the fixed substrate frame; no molecular Kabsch fit",
            "rmsd_atoms": "SAM heavy atoms excluding fixed P and three P-bound O",
            "representative_metric": "maximum minimum z over SAM atoms excluding fixed P/O head",
            "fixed_head_atom_ids_1based_within_sam": [int(index) + 1 for index in head_ids],
        },
        "system": {
            "substrate_atom_count_in_accepted_structure": int(substrate_count),
            "sam_atom_count": sam_count,
            "sam_h0_symbols": h0_symbols,
            "input_collision_safe_count": len(records),
        },
        "summary": {
            "input_collision_safe_count": len(records),
            "second_stage_cluster_count": len(cluster_records),
            "representative_count": len(cluster_records),
        },
        "artifacts": {
            "records": str(records_output),
            "clusters": str(clusters_output),
            "representative_directory": str(representative_directory),
        },
        "representatives": cluster_records,
    }
    manifest_path = run_directory / "audit-recluster-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (run_directory / "README.zh-CN.md").write_text(
        "\n".join(
            [
                f"# {_force_field_label(force_field)} ITO 审计通过结构的第二层聚类",
                "",
                f"- 输入：{len(records)} 个已通过固定 ITO 周期碰撞审计的结构",
                f"- RMSD：{float(args.rmsd_tolerance_A):g} Å，固定基底坐标系 complete-link",
                f"- RMSD 原子：SAM 重原子，排除固定 P 和三个 P-bound O",
                f"- 代表数：{len(cluster_records)}",
                "- 代表选择：每簇除 PO3 外 SAM 原子的最低 Z 最大者；并列时取原始审计簇编号较小者。",
                "- 该层只减少候选数，不是 MACE 能量排序。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "stage": "second-stage clustering after fixed-site substrate audit",
                "status": manifest["status"],
                "run_directory": str(run_directory),
                "manifest": str(manifest_path),
                "summary": manifest["summary"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


def _finish_optimization_only(
    *,
    run_directory: Path,
    optimized_directory: Path,
    records_path: Path,
    progress_path: Path,
    identity: str,
    sources: dict,
    parameters: dict,
    h0_symbols: list[str],
    head_indices: list[int],
    cpu_manifest: dict,
    declared_count: int,
    duplicate_count: int,
    tasks: list[dict],
    optimized_records: list[dict],
    limit: int | None,
    elapsed_s: float,
) -> int:
    """Record isolated optimization outcomes without fixed-site clustering."""

    converged = [record for record in optimized_records if record["status"] == "optimized_converged"]
    unconverged = [record for record in optimized_records if record["status"] == "optimized_not_converged"]
    failures = [record for record in optimized_records if record["status"] == "failed"]
    completed = converged + unconverged
    status = (
        "optimization_completed_pending_skeleton_deduplication"
        if completed
        else "optimization_failed_no_finite_candidates"
    )
    manifest = {
        "schema": "sam-phosphonate-skeleton-ff-v1",
        "schema_version": SCHEMA_VERSION + 1,
        "run_identity": identity,
        "status": status,
        "scope": "isolated_skeleton_fixed_head_force_field_optimization_no_ITO_audit",
        "records_scope": "optimized_skeleton_all_outcomes",
        "sources": sources,
        "parameters": parameters,
        "system": {
            "sam_atom_count": len(h0_symbols),
            "sam_h0_formula_symbols": h0_symbols,
            "fixed_head_atom_ids_1based": [index + 1 for index in head_indices],
            "substrate_atom_count": 0,
            "surface_h_count": 0,
        },
        "summary": {
            "input_skeleton_safe_count": declared_count,
            "endpoint_duplicate_count": duplicate_count,
            "unique_candidates_submitted": len(tasks),
            "force_field_finite_result_count": len(completed),
            "force_field_converged_count": len(converged),
            "force_field_not_converged_count": len(unconverged),
            "force_field_failed_count": len(failures),
            "fixed_head_max_displacement_A": (
                max(record["fixed_head_max_displacement_A"] for record in completed)
                if completed else None
            ),
            "input_grid_complete": cpu_manifest.get("grid_complete"),
            "all_input_candidates_processed": len(tasks) == declared_count - duplicate_count,
            "limit": limit,
            "clustering_performed": False,
            "substrate_collision_audit_count": 0,
            "elapsed_s": elapsed_s,
        },
        "artifacts": {
            "optimized_records": str(records_path),
            "optimized_directory": str(optimized_directory),
            "progress": str(progress_path),
        },
        "next_stage": "PC_axis_invariant_skeleton_deduplication_then_shared_axial_height_and_clearance_audit",
        "not_yet_validated": [
            "PC_axis_invariant_basin_identity",
            "post_optimization_axial_height_and_clearance",
            "ITO_site_collision",
            "surface_optimized_adsorption_energy",
        ],
    }
    manifest_path = run_directory / "fast-ff-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (run_directory / "README.zh-CN.md").write_text(
        "\n".join(
            [
                f"# 固定 P/三个 O 的 {_force_field_label(parameters['force_field'])} 骨架优化",
                "",
                f"- 已处理候选：{len(tasks)} / {declared_count}；有限优化结果：{len(completed)}。",
                f"- 独立可动原子最大梯度 ≤ {parameters['movable_fmax_eV_A']:g} eV/Å：{len(converged)}；未达门：{len(unconverged)}；异常：{len(failures)}。",
                "- P 和三个 O 通过 AddFixedPoint 固定，并逐候选核验坐标位移。",
                "- 每 250 步请求预算后检查独立梯度；记录的是请求预算，不是真实迭代数。",
                "- RDKit 内部终止码单独保留；实际可动原子最大梯度决定本阶段是否收敛。",
                "- 优化中未加 Z 约束，末态未按 Z 剔除；后续统一处理 P–C 轴角与高度净空。",
                "- 本阶段未聚类，也未加入 ITO；力场能量单位 kcal/mol，不作为吸附能排序。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"stage": "isolated fixed-head skeleton optimization only", "status": status, "manifest": str(manifest_path), "summary": manifest["summary"]}, indent=2), flush=True)
    return 0 if completed else 2


def run(args: argparse.Namespace) -> int:
    optimization_only = bool(getattr(args, "optimization_only", False))
    movable_fmax_eV_A = float(
        getattr(args, "movable_fmax_eV_A", DEFAULT_MOVABLE_FMAX_EV_A)
    )
    if movable_fmax_eV_A <= 0 or not np.isfinite(movable_fmax_eV_A):
        raise ValueError("movable-fmax-eV-A must be positive and finite")
    for name in ("workers", "chunksize", "max_iterations"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    output_root = Path(args.output_root).expanduser().resolve()
    cpu_manifest_path = Path(args.cpu_manifest).expanduser().resolve()
    template_sdf = Path(args.template_sdf).expanduser().resolve()
    if not cpu_manifest_path.is_file():
        raise ValueError(f"CPU manifest does not exist: {cpu_manifest_path}")
    if not template_sdf.is_file():
        raise ValueError(f"template SDF does not exist: {template_sdf}")
    cpu_manifest = json.loads(cpu_manifest_path.read_text(encoding="utf-8"))
    skeleton_input = cpu_manifest.get("schema") == "sam-phosphonate-skeleton-cpu-v1"
    if skeleton_input and not optimization_only:
        raise ValueError(
            "isolated skeleton input requires --optimization-only; "
            "fixed-site clustering is not a P-C-invariant skeleton comparison"
        )
    required_scope = (
        "isolated_skeleton_screen_safe_only" if skeleton_input else "collision_safe_only"
    )
    if cpu_manifest.get("records_scope") != required_scope:
        raise ValueError("input CPU manifest is not a collision-safe ensemble")
    required_status = (
        "passed_skeleton_cpu_screen" if skeleton_input else "passed_cpu_collision_screen"
    )
    records = [
        record
        for record in cpu_manifest.get("records", [])
        if record.get("status") == required_status
    ]
    records.sort(key=lambda item: int(item["safe_rank"]))
    declared_count = int(
        cpu_manifest["summary"]["screen_safe_count" if skeleton_input else "collision_safe_count"]
    )
    if len(records) != declared_count or not records:
        raise ValueError(
            f"CPU manifest records do not match declared safe count: "
            f"{len(records)} versus {declared_count}"
        )
    layout = cpu_manifest["system"]
    substrate_count = 0 if skeleton_input else int(layout["substrate_atom_count"])
    surface_h_count = 0 if skeleton_input else int(layout["surface_h_count"])
    sam_count = int(layout["sam_atom_count"])
    sam_start = substrate_count + surface_h_count
    if skeleton_input:
        sam_source = layout["sam_source"]
        sam_h0_path = (cpu_manifest_path.parent / sam_source["path"]).resolve()
        if not sam_h0_path.is_file() or _sha256_path(sam_h0_path) != sam_source["sha256"]:
            raise ValueError(f"recorded H0 source is missing or its hash changed: {sam_h0_path}")
    else:
        sam_h0_path = (
            cpu_manifest_path.parent.parent.parent.parent / "sam-h0.extxyz"
        ).resolve()
    if not sam_h0_path.is_file():
        raise ValueError(f"H0 source recorded by CPU run is missing: {sam_h0_path}")
    h0_symbols, _, _ = _read_extxyz_bytes(sam_h0_path.read_bytes())
    if len(h0_symbols) != sam_count:
        raise ValueError("H0 source atom count does not match CPU manifest")
    # Resolve this once in the parent.  Re-reading and sanitizing the SDF for
    # every candidate would erase most of the benefit of the fast stage.
    _, head_indices = _h0_rdkit_template(template_sdf, h0_symbols)
    if skeleton_input and [index + 1 for index in head_indices] != sorted(
        int(value) for value in cpu_manifest["parameters"]["head_atom_ids_1based"]
    ):
        raise ValueError("RDKit P/O head mapping differs from skeleton scan manifest")

    deduplicated: list[dict] = []
    seen_keys: set[tuple[float, ...]] = set()
    duplicate_count = 0
    for record in records:
        angles = tuple(
            _normalize_angle(value) for value in record["start_dihedrals_deg"]
        )
        if angles in seen_keys:
            duplicate_count += 1
            continue
        seen_keys.add(angles)
        source = (cpu_manifest_path.parent / record["structure"]).resolve()
        if not source.is_file():
            raise ValueError(f"CPU-screen candidate is missing: {source}")
        if skeleton_input and _sha256_path(source) != record["structure_sha256"]:
            raise ValueError(f"CPU-screen candidate hash changed: {source}")
        deduplicated.append(
            {
                "grid_index": int(record["grid_index"]),
                "safe_rank": int(record["safe_rank"]),
                "candidate_key": angles,
                "start_dihedrals_deg": list(record["start_dihedrals_deg"]),
                "structure": str(source),
                "structure_sha256": record["structure_sha256"],
            }
        )
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        deduplicated = deduplicated[: int(args.limit)]
    if not deduplicated:
        raise ValueError("No candidates remain after endpoint deduplication")

    force_field = _force_field_slug(args.force_field)
    force_field_label = _force_field_label(force_field)
    parameters = {
        "force_field": force_field,
        "force_field_label": force_field_label,
        "force_field_actual_variant": _force_field_api_variant(force_field),
        "fixed_head": "P plus all three P-bound O atoms",
        "optimize_substrate": False,
        "max_iterations": int(args.max_iterations),
        "rmsd_tolerance_A": float(args.rmsd_tolerance_A),
        "rmsd_atoms": "all heavy atoms excluding P and its three bound O atoms",
        "representative_metric": "maximum minimum z over every atom excluding P/O head, including H",
        "candidate_deduplication": "normalize 180 degrees to -180 degrees in every scanned torsion",
        "worker_count": int(args.workers),
    }
    isomer_name = None
    if optimization_only:
        isomer_name = getattr(args, "isomer_name", None)
        if not isomer_name:
            source_folder = Path(deduplicated[0]["structure"]).parent.name
            suffix = "_monomer_unoptimized_conformations"
            if source_folder.endswith(suffix):
                isomer_name = source_folder[: -len(suffix)]
        if not isomer_name or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", str(isomer_name)):
            raise ValueError("--isomer-name is required and must be a safe nonempty folder token")
        parameters.update(
            {
                "optimization_only": True,
                "isomer_name": str(isomer_name),
                "movable_fmax_eV_A": movable_fmax_eV_A,
                "movable_fmax_kcal_mol_A": movable_fmax_eV_A * KCAL_MOL_PER_EV,
                "convergence_gate": "maximum movable-atom 3D gradient <= movable_fmax_eV_A; RDKit termination code separately recorded",
                "iteration_block_limit": 250,
                "iteration_count_semantics": "sum of requested per-call limits is a budget, not an observed actual iteration count",
                "fixed_head_displacement_tolerance_A": FIXED_HEAD_DISPLACEMENT_TOLERANCE_A,
                "z_constraints": False,
                "post_optimization_z_rejection": False,
                "clustering_performed": False,
            }
        )
        for irrelevant in ("rmsd_tolerance_A", "rmsd_atoms", "representative_metric"):
            parameters.pop(irrelevant)
    source_payload = {
        "cpu_manifest": str(cpu_manifest_path),
        "cpu_manifest_sha256": _sha256_path(cpu_manifest_path),
        "template_sdf": str(template_sdf),
        "template_sdf_sha256": _sha256_path(template_sdf),
        "h0_source": str(sam_h0_path),
        "h0_source_sha256": _sha256_path(sam_h0_path),
        "script": str(Path(__file__).resolve()),
        "script_sha256": _sha256_path(Path(__file__).resolve()),
    }
    identity = _json_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "sources": source_payload,
            "parameters": parameters,
            "candidate_keys": [item["candidate_key"] for item in deduplicated],
        }
    )
    run_directory = output_root / f"fast-{force_field}-prefilter-{identity[:12]}"
    if run_directory.exists():
        raise ValueError(f"immutable FF run directory already exists: {run_directory}")
    optimized_directory = run_directory / f"01_{force_field}_optimized" / (
        f"{isomer_name}_monomer_optimized_conformations" if optimization_only else "structures"
    )
    optimized_directory.mkdir(parents=True)

    print(
        json.dumps(
            {
                "stage": f"{force_field_label} isolated fixed-head optimization",
                "status": "running",
                "input_collision_safe_count": declared_count,
                "endpoint_duplicates_removed": duplicate_count,
                "unique_candidates_submitted": len(deduplicated),
                "workers": int(args.workers),
                "max_iterations": int(args.max_iterations),
            },
            indent=2,
        ),
        flush=True,
    )

    start_time = time.time()
    tasks = list(deduplicated)
    optimized_records: list[dict] = []
    optimized_positions: list[np.ndarray] = []
    records_path = run_directory / f"01_{force_field}_optimized" / "optimized-records.jsonl"
    progress_path = run_directory / "progress.json"

    def write_progress(status: str) -> None:
        counts = {
            "processed_count": len(optimized_records),
            "submitted_count": len(tasks),
            "converged_count": sum(bool(item.get("force_field_converged")) for item in optimized_records),
            "not_converged_count": sum(item.get("status") == "optimized_not_converged" for item in optimized_records),
            "failed_count": sum(item.get("status") == "failed" for item in optimized_records),
        }
        temporary = progress_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({"status": status, **counts, "elapsed_s": time.time() - start_time}, indent=2), encoding="utf-8")
        temporary.replace(progress_path)

    write_progress("running")
    context = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else None
    executor_kwargs = {
        "max_workers": int(args.workers),
        "initializer": _worker_init,
        "initargs": (
            str(template_sdf),
            h0_symbols,
            sam_start,
            sam_count,
            int(args.max_iterations),
            force_field,
            movable_fmax_eV_A if optimization_only else None,
        ),
    }
    if context is not None:
        executor_kwargs["mp_context"] = context
    with records_path.open("w", encoding="utf-8") as records_handle, ProcessPoolExecutor(**executor_kwargs) as executor:
        results = executor.map(_optimize_one, tasks, chunksize=int(args.chunksize))
        for serial_index, result in enumerate(results, 1):
            record = dict(result)
            if record["status"] == "passed":
                coordinates = np.asarray(record.pop("positions"), dtype=float)
                non_head = np.asarray(
                    [index for index in range(sam_count) if index not in head_indices],
                    dtype=int,
                )
                record["z_min_nonpo3_all_A"] = float(np.min(coordinates[non_head, 2]))
                record["z_min_nonpo3_heavy_A"] = float(
                    np.min(
                        coordinates[
                            [
                                index
                                for index in non_head
                                if h0_symbols[index] != "H"
                            ],
                            2,
                        ]
                    )
                )
                output_path = optimized_directory / f"candidate-{serial_index:06d}.extxyz"
                _write_extxyz(
                    output_path,
                    h0_symbols,
                    coordinates,
                    "Properties=species:S:1:pos:R:3 pbc=\"F F F\"",
                )
                record["optimized_structure"] = str(output_path)
                record["optimized_structure_sha256"] = _sha256_path(output_path)
                optimized_positions.append(coordinates)
                if optimization_only:
                    record["status"] = (
                        "optimized_converged" if record["force_field_converged"] else "optimized_not_converged"
                    )
            optimized_records.append(record)
            records_handle.write(json.dumps(record, sort_keys=True) + "\n")
            if serial_index % 50 == 0 or serial_index == len(tasks):
                records_handle.flush()
                write_progress("running")
            if serial_index % (100 if optimization_only else 1000) == 0 or serial_index == len(tasks):
                passed = sum(item["status"] != "failed" for item in optimized_records)
                print(
                    f"{force_field_label} optimized {serial_index}/{len(tasks)}; finite_results={passed}",
                    flush=True,
                )

    write_progress("completed")
    if optimization_only:
        return _finish_optimization_only(
            run_directory=run_directory,
            optimized_directory=optimized_directory,
            records_path=records_path,
            progress_path=progress_path,
            identity=identity,
            sources=source_payload,
            parameters=parameters,
            h0_symbols=h0_symbols,
            head_indices=head_indices,
            cpu_manifest=cpu_manifest,
            declared_count=declared_count,
            duplicate_count=duplicate_count,
            tasks=tasks,
            optimized_records=optimized_records,
            limit=args.limit,
            elapsed_s=time.time() - start_time,
        )

    failures = [item for item in optimized_records if item["status"] != "passed"]
    passed_records = [item for item in optimized_records if item["status"] == "passed"]
    if not passed_records:
        raise RuntimeError(f"All {force_field_label} candidates failed")
    positions = np.asarray(optimized_positions, dtype=float)
    heavy_indices = np.asarray(
        [
            index
            for index, symbol in enumerate(h0_symbols)
            if symbol != "H" and index not in head_indices
        ],
        dtype=int,
    )
    clusters, assignments = _complete_link_clusters(
        positions, heavy_indices, float(args.rmsd_tolerance_A)
    )
    for index, record in enumerate(passed_records):
        record["cluster_index"] = int(assignments[index]) + 1

    cluster_records = []
    for cluster_index, member_indices in enumerate(clusters, 1):
        ordered = sorted(
            member_indices,
            key=lambda index: (
                -float(passed_records[index]["z_min_nonpo3_all_A"]),
                int(passed_records[index]["safe_rank"]),
            ),
        )
        representative_index = ordered[0]
        representative = passed_records[representative_index]
        cluster_records.append(
            {
                "cluster_index": cluster_index,
                "member_count": len(member_indices),
                "member_record_indices": [int(index) for index in member_indices],
                "member_grid_indices": [
                    int(passed_records[index]["grid_index"])
                    for index in member_indices
                ],
                "representative_record_index": int(representative_index),
                "representative_grid_index": int(representative["grid_index"]),
                "representative_structure": representative["optimized_structure"],
                "representative_structure_sha256": representative[
                    "optimized_structure_sha256"
                ],
                "representative_candidate_key": list(
                    representative["candidate_key"]
                ),
                "representative_safe_rank": int(representative["safe_rank"]),
                "representative_z_min_nonpo3_all_A": float(
                    representative["z_min_nonpo3_all_A"]
                ),
                "representative_z_min_nonpo3_heavy_A": float(
                    representative["z_min_nonpo3_heavy_A"]
                ),
                "representative_force_field": force_field,
                "representative_force_field_energy_kcal_mol": float(
                    representative["force_field_energy_kcal_mol"]
                ),
            }
        )

    (run_directory / f"01_{force_field}_optimized" / "optimized-records.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in optimized_records),
        encoding="utf-8",
    )
    (run_directory / f"01_{force_field}_optimized" / "clusters.json").write_text(
        json.dumps(cluster_records, indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_identity": identity,
        "status": "passed_ff_pending_substrate_audit",
        "scope": "limited_pilot" if args.limit is not None else "complete_endpoint_deduplicated_safe_ensemble",
        "sources": source_payload,
        "parameters": parameters,
        "system": {
            "substrate_atom_count": substrate_count,
            "surface_h_count": surface_h_count,
            "sam_atom_count": sam_count,
            "sam_h0_formula_symbols": h0_symbols,
            "fixed_head_atom_ids_1based": [int(index) + 1 for index in head_indices],
            "nonpo3_heavy_atom_ids_1based": [int(index) + 1 for index in heavy_indices],
        },
        "summary": {
            "input_collision_safe_count": declared_count,
            "endpoint_duplicate_count": duplicate_count,
            "unique_candidates_submitted": len(tasks),
            "force_field_passed_count": len(passed_records),
            "force_field_failed_count": len(failures),
            "cluster_count": len(cluster_records),
            "substrate_collision_audit_count": len(cluster_records),
            "substrate_collision_safe_count": None,
            "elapsed_s": time.time() - start_time,
        },
        "artifacts": {
            "optimized_records": str(run_directory / f"01_{force_field}_optimized" / "optimized-records.jsonl"),
            "clusters": str(run_directory / f"01_{force_field}_optimized" / "clusters.json"),
        },
        "representatives": cluster_records,
    }
    manifest_path = run_directory / "fast-ff-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (run_directory / "README.zh-CN.md").write_text(
        "\n".join(
            [
                f"# 固定 PO₃ 头基的 {force_field_label} 预优化",
                "",
                f"- 状态：`{manifest['status']}`",
                f"- 输入安全候选：{declared_count}",
                f"- 端点去重后候选：{len(tasks)}",
                f"- {force_field_label} 成功：{len(passed_records)}；失败：{len(failures)}",
                f"- 1 Å complete-link 簇：{len(cluster_records)}",
                "- 每簇代表：除 PO₃ 外所有原子的最小 Z 最大者；该层包括 H。",
                "- 力场能量单位为 kcal/mol，只作诊断，不用于最终吸附能排序。",
                "- 下一阶段：将每个代表重新放回固定 ITO Site Instance，做完整周期碰撞审计。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "stage": f"{force_field_label} isolated fixed-head optimization",
                "status": "passed_pending_substrate_audit",
                "run_directory": str(run_directory),
                "manifest": str(manifest_path),
                "summary": manifest["summary"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


def _read_head_indices_from_template(template_sdf: Path, h0_symbols: list[str]) -> list[int]:
    _, head_indices = _h0_rdkit_template(template_sdf, h0_symbols)
    return head_indices


def _h0_topology_contract_local(template_sdf: Path, h0_symbols: list[str]) -> dict:
    """Read the reviewed SDF graph through the existing RDKit H0 adapter."""

    from rdkit import Chem

    molecule, head = _h0_rdkit_template(template_sdf, h0_symbols)
    source = Chem.MolFromMolFile(str(template_sdf), removeHs=False)
    if source is None:
        raise ValueError(f"RDKit could not read source SDF: {template_sdf}")
    source_symbols = [atom.GetSymbol() for atom in source.GetAtoms()]
    source_p = next(atom.GetIdx() for atom in source.GetAtoms() if atom.GetSymbol() == "P")
    removed = {
        neighbor.GetIdx()
        for oxygen in source.GetAtomWithIdx(source_p).GetNeighbors()
        if oxygen.GetSymbol() == "O"
        for neighbor in oxygen.GetNeighbors()
        if neighbor.GetSymbol() == "H"
    }
    retained = [index for index in range(len(source_symbols)) if index not in removed]
    if [source_symbols[index] for index in retained] != h0_symbols:
        if source_symbols != h0_symbols:
            raise ValueError("SDF-to-H0 atom mapping changed")
        retained = list(range(len(source_symbols)))
    p = next(atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetSymbol() == "P")
    carbon = [atom.GetIdx() for atom in molecule.GetAtomWithIdx(p).GetNeighbors()
              if atom.GetSymbol() == "C"]
    if len(carbon) != 1 or molecule.GetAtomWithIdx(p).GetDegree() != 4:
        raise ValueError("isolated skeleton postprocessing requires one P-C and three P-O bonds")
    c_index = carbon[0]
    heavy_molecule = Chem.RemoveHs(Chem.Mol(molecule))
    heavy_h0_indices = [index for index, symbol in enumerate(h0_symbols) if symbol != "H"]
    if [atom.GetSymbol() for atom in heavy_molecule.GetAtoms()] != [h0_symbols[index] for index in heavy_h0_indices]:
        raise ValueError("RDKit heavy-atom removal changed the reviewed H0 atom order")
    max_automorphisms = 10_000
    matches = heavy_molecule.GetSubstructMatches(
        heavy_molecule, uniquify=False, useChirality=True, maxMatches=max_automorphisms + 1
    )
    if len(matches) > max_automorphisms:
        raise ValueError(
            f"source heavy-graph automorphism count exceeds the safe enumeration limit {max_automorphisms}"
        )
    heavy_to_h0 = {heavy_index: h0_index for heavy_index, h0_index in enumerate(heavy_h0_indices)}
    feature_h0_indices = [index for index in heavy_h0_indices if index not in head]
    feature_index = {h0_index: row for row, h0_index in enumerate(feature_h0_indices)}
    automorphism_permutations = set()
    head_set = set(head)
    for match in matches:
        h0_map = {heavy_to_h0[source]: heavy_to_h0[target] for source, target in enumerate(match)}
        if (h0_map.get(p) != p or h0_map.get(c_index) != c_index
                or {h0_map[index] for index in head_set} != head_set
                or {h0_map[index] for index in feature_h0_indices} != set(feature_h0_indices)):
            continue
        permutation = tuple(feature_index[h0_map[index]] for index in feature_h0_indices)
        automorphism_permutations.add(permutation)
    identity = tuple(range(len(feature_h0_indices)))
    if identity not in automorphism_permutations:
        raise ValueError("exact source graph automorphisms omitted the identity correspondence")
    return {
        "p_index": p, "c_index": c_index, "head_indices": head,
        "bond_pairs": sorted([sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
                              for bond in molecule.GetBonds()]),
        "original_atom_ids_1based": [index + 1 for index in retained],
        "source_atom_count": len(source_symbols),
        "source_formal_charge": sum(atom.GetFormalCharge() for atom in source.GetAtoms()),
        "h0_formal_charge": sum(atom.GetFormalCharge() for atom in molecule.GetAtoms()),
        "graph_source": "reviewed_source_SDF_RDKit_H0_adapter",
        "rmsd_atom_automorphisms": [list(permutation) for permutation in sorted(automorphism_permutations)],
        "rdkit_version": Chem.rdBase.rdkitVersion,
    }


def _h0_topology_contract(template_sdf: Path, h0_symbols: list[str],
                          topology_python: str | Path | None = None) -> dict:
    """Bridge the separate chemistry environment once; never infer optimized bonds."""

    if topology_python is None:
        try:
            return _h0_topology_contract_local(template_sdf, h0_symbols)
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.startswith("rdkit"):
                raise ValueError("RDKit is unavailable in this interpreter; supply --topology-python") from exc
            raise
    interpreter = Path(topology_python).expanduser().absolute()
    if not interpreter.is_file():
        raise ValueError(f"topology Python does not exist: {interpreter}")
    program = (
        "import json,sys\nfrom pathlib import Path\n"
        f"sys.path.insert(0,{str(Path(__file__).resolve().parent)!r})\n"
        "import fast_ff_prefilter as ff\n"
        "data=json.load(sys.stdin)\n"
        "result=ff._h0_topology_contract_local(Path(data['sdf']),data['symbols'])\n"
        "print('TOPOLOGY_RESULT='+json.dumps(result,sort_keys=True))\n"
    )
    completed = subprocess.run([str(interpreter), "-c", program],
                               input=json.dumps({"sdf": str(template_sdf), "symbols": h0_symbols}),
                               capture_output=True, text=True, check=False, timeout=90)
    if completed.returncode:
        raise ValueError(f"RDKit topology adapter failed: {completed.stderr.strip()}")
    results = [line for line in completed.stdout.splitlines() if line.startswith("TOPOLOGY_RESULT=")]
    if len(results) != 1:
        raise ValueError("RDKit topology adapter returned no unique JSON result")
    return json.loads(results[0].split("=", 1)[1])


def _skeleton_pair_contract(symbols: list[str], bond_pairs: set[tuple[int, int]],
                            organic_indices: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exact-source nonbonded pairs, hard floors, and axis invariance."""

    organic = set(organic_indices)
    pairs, thresholds, invariant = [], [], []
    for left in range(len(symbols)):
        for right in range(left + 1, len(symbols)):
            if (left, right) in bond_pairs:
                continue
            pairs.append((left, right))
            thresholds.append(1.2 if symbols[left] == symbols[right] == "H" else
                              1.4 if "H" in (symbols[left], symbols[right]) else 1.8)
            invariant.append((left in organic) == (right in organic))
    return (np.asarray(pairs, dtype=int).reshape(-1, 2),
            np.asarray(thresholds, dtype=float), np.asarray(invariant, dtype=bool))


def _rotate_organic_pc(positions: np.ndarray, p_index: int, c_index: int,
                       organic_indices: list[int], angle_deg: float) -> np.ndarray:
    """Rotate only the complete C-side fragment, retaining P/O coordinates exactly."""

    rotated = np.asarray(positions, dtype=float).copy()
    axis = rotated[c_index] - rotated[p_index]
    axis /= np.linalg.norm(axis)
    relative = rotated[organic_indices] - rotated[p_index]
    angle = np.radians(angle_deg)
    cosine, sine = np.cos(angle), np.sin(angle)
    rotated[organic_indices] = rotated[p_index] + (
        relative * cosine + np.cross(axis, relative) * sine
        + np.outer(relative @ axis, axis) * (1.0 - cosine)
    )
    return rotated


def _complete_link_pc_axis_clusters(features: np.ndarray, tolerance_A: float,
                                   batch_size: int = 256, progress=None
                                   ) -> tuple[list[list[int]], list[int], list[float]]:
    """Deterministic insertion with every pair under the true 3D axial RMSD gate.

    RMSD rows are bounded by batch_size. Complete-link prevents a close-neighbor
    chain from joining structures whose pair distance exceeds the threshold.
    """

    from sam_structure_tools import pc_axial_rmsd_matrix

    if tolerance_A <= 0 or not np.isfinite(tolerance_A) or batch_size <= 0:
        raise ValueError("axial RMSD tolerance and batch size must be positive")
    features = np.asarray(features, dtype=float)
    clusters, diameters = [], []
    assignments = [-1] * len(features)
    for start in range(0, len(features), batch_size):
        stop = min(start + batch_size, len(features))
        distance_rows = pc_axial_rmsd_matrix(features[start:stop], features[:stop])
        for index in range(start, stop):
            distances = distance_rows[index - start]
            for cluster_index, members in enumerate(clusters):
                member_distances = distances[members]
                if np.all(member_distances <= tolerance_A + 1.0e-12):
                    diameters[cluster_index] = max(diameters[cluster_index], float(np.max(member_distances)))
                    members.append(index)
                    assignments[index] = cluster_index
                    break
            else:
                assignments[index] = len(clusters)
                clusters.append([index])
                diameters.append(0.0)
        if progress is not None:
            progress(stop, len(clusters))
    return clusters, assignments, diameters


def _energy_ordered_pc_radius_clusters(
    features: np.ndarray, energies_kcal_mol, grid_indices, safe_ranks,
    tolerance_A: float, batch_size: int = 256, progress=None, atom_permutations=None,
) -> tuple[list[list[int]], list[int], list[int], list[float]]:
    """Cover skeletons by fixed energy-ordered representatives, without replacement.

    按原 MMFF 能量、网格编号、源 rank 排序。只向当时已选代表中最近的
    合格代表归入；代表固定，后来的代表不会重新分配或替换先前的成员。
    Every member is within the radius of its representative. Representatives
    are mutually outside the radius; a family diameter can approach twice it.
    """

    from sam_structure_tools import pc_axial_rmsd_matrix

    features = np.asarray(features, dtype=float)
    if features.ndim != 3 or features.shape[-1] != 3 or not np.all(np.isfinite(features)):
        raise ValueError("radius features must be finite (candidate, atom, xyz) coordinates")
    if features.shape[1] == 0:
        raise ValueError("radius features must contain organic heavy atoms")
    energies = np.asarray(energies_kcal_mol, dtype=float)
    grids, ranks = np.asarray(grid_indices), np.asarray(safe_ranks)
    if (energies.shape != (len(features),) or grids.shape != energies.shape
            or ranks.shape != energies.shape or not np.all(np.isfinite(energies))):
        raise ValueError("source FF energies must be finite and all ordering arrays must match candidates")
    if tolerance_A <= 0.0 or not np.isfinite(tolerance_A) or batch_size <= 0:
        raise ValueError("radius tolerance and batch size must be positive and finite")
    if len(set(int(rank) for rank in ranks)) != len(ranks):
        raise ValueError("source safe ranks must be unique")
    if atom_permutations is not None:
        permutations = np.asarray(atom_permutations, dtype=int)
        identity = np.arange(features.shape[1], dtype=int)
        if (permutations.ndim != 2 or permutations.shape[1] != features.shape[1]
                or len(permutations) == 0
                or any(not np.array_equal(np.sort(row), identity) for row in permutations)
                or len({tuple(row) for row in permutations}) != len(permutations)
                or not any(np.array_equal(row, identity) for row in permutations)):
            raise ValueError("symmetry atom permutations must be unique full mappings and include identity")

    def pair_distances(reference, candidates):
        if atom_permutations is None:
            return pc_axial_rmsd_matrix(reference, candidates)
        minimum = None
        for permutation in permutations:
            distances = pc_axial_rmsd_matrix(reference, candidates[:, permutation, :])
            minimum = distances if minimum is None else np.minimum(minimum, distances)
        return minimum

    order = sorted(range(len(features)), key=lambda index: (float(energies[index]),
                                                            int(grids[index]), int(ranks[index])))
    clusters, representative_indices = [], []
    assignments = [-1] * len(features)
    representative_distances = [0.0] * len(features)
    for start in range(0, len(order), batch_size):
        block = order[start:start + batch_size]
        # Include this block in the reference columns so newly selected
        # representatives have known distances without a per-candidate fit.
        columns = representative_indices + block
        column_lookup = {index: column for column, index in enumerate(columns)}
        rows = pair_distances(features[block], features[columns])
        for row, index in enumerate(block):
            eligible = [cluster_index for cluster_index, representative in enumerate(representative_indices)
                        if rows[row, column_lookup[representative]] <= tolerance_A + 1.0e-12]
            if eligible:
                cluster_index = min(eligible, key=lambda cluster_index: (
                    float(rows[row, column_lookup[representative_indices[cluster_index]]]),
                    float(energies[representative_indices[cluster_index]]),
                    int(grids[representative_indices[cluster_index]]),
                    int(ranks[representative_indices[cluster_index]])))
                representative_distances[index] = float(rows[row, column_lookup[representative_indices[cluster_index]]])
                clusters[cluster_index].append(index)
                assignments[index] = cluster_index
            else:
                assignments[index] = len(clusters)
                clusters.append([index])
                representative_indices.append(index)
        if progress is not None:
            progress(min(start + batch_size, len(order)), len(clusters))
    return clusters, assignments, representative_indices, representative_distances


def _skeleton_dedup_parameters(method: str) -> dict:
    if method == "energy-radius-symmetry":
        return {
            "skeleton_dedup_method": method,
            "clustering": "fixed source-energy-ordered representative radius using the minimum axial RMSD over exact source-H0 heavy-graph automorphisms; process source FF energy/grid_index/safe_rank ascending; assign to nearest currently selected admissible representative; no replacement or later remapping",
            "representative_choice": "original optimized source FF energy kcal/mol ascending, grid_index ascending, safe_rank ascending; representative fixed when selected",
            "atom_correspondence": "minimum over exact RDKit H0 heavy-graph automorphisms that preserve P, the P-bonded C, and the feature/head partition; coordinates and saved original atom IDs are never reordered",
            "guarantees": "every member-to-representative symmetry-aware axial RMSD <= radius; representative pair symmetry-aware RMSD > radius; representative source energy <= every assigned member; family pair diameter may approach 2*radius",
        }
    if method == "energy-radius":
        return {
            "skeleton_dedup_method": method,
            "clustering": "fixed representative radius; process source FF energy/grid_index/safe_rank ascending; assign to nearest currently selected admissible representative; no replacement or later remapping",
            "representative_choice": "original optimized source FF energy kcal/mol ascending, grid_index ascending, safe_rank ascending; representative fixed when selected",
            "guarantees": "every member-to-representative RMSD <= radius; representative pair RMSD > radius; representative source energy <= every assigned member; family pair diameter may approach 2*radius",
        }
    if method == "complete-link":
        return {
            "skeleton_dedup_method": method,
            "clustering": "deterministic complete-link insertion in increasing source safe_rank; all within-cluster pair RMSD <= threshold",
            "representative_choice": "largest minimum organic Z including H, then smallest original grid_index, then safe_rank; no FF energy ranking",
            "guarantees": "every within-family pair RMSD <= threshold; historical representative selection retained",
        }
    raise ValueError(f"unsupported skeleton dedup method: {method}")


def _validated_source_force_field_energy(record: dict, ff_manifest: dict) -> float:
    """Require one parent FF/API variant and a finite original-coordinate energy."""

    parent_parameters = ff_manifest["parameters"]
    force_field = _force_field_slug(parent_parameters["force_field"])
    variant = _force_field_api_variant(force_field)
    if (parent_parameters.get("force_field_actual_variant") != variant or
            record.get("force_field") != force_field or
            record.get("force_field_actual_variant") != variant):
        raise ValueError("source records must use the same parent force field and actual API variant")
    energy = float(record["force_field_energy_kcal_mol"])
    if not np.isfinite(energy):
        raise ValueError("source FF energies must be finite")
    return energy


def _write_skeleton_dedup_products(valid_records: list[dict], features: np.ndarray,
                                    tolerance_A: float, batch_size: int, method: str,
                                    representatives_directory: Path, progress=None,
                                    atom_permutations=None,
                                    ) -> list[dict]:
    """Share leader selection and member mapping between fresh and replay stages."""

    from sam_structure_tools import pc_axial_rmsd_matrix

    if not valid_records:
        representatives_directory.mkdir(parents=True)
        return []
    energies = np.asarray([record["source_optimized_result"]["force_field_energy_kcal_mol"]
                           for record in valid_records], dtype=float)
    if not np.all(np.isfinite(energies)):
        raise ValueError("source FF energies must be finite")
    features = np.asarray(features, dtype=float)
    diameters = None
    if method in ("energy-radius", "energy-radius-symmetry"):
        clusters, assignments, representatives, representative_distances = _energy_ordered_pc_radius_clusters(
            features, energies, [record["grid_index"] for record in valid_records],
            [record["safe_rank"] for record in valid_records], tolerance_A, batch_size, progress,
            atom_permutations=atom_permutations if method == "energy-radius-symmetry" else None)
    elif method == "complete-link":
        clusters, assignments, diameters = _complete_link_pc_axis_clusters(features, tolerance_A, batch_size, progress)
        representatives = [min(members, key=lambda index: (
            -valid_records[index]["pose_organic_z_min_A"], valid_records[index]["grid_index"],
            valid_records[index]["safe_rank"])) for members in clusters]
        representative_distances = [0.0] * len(valid_records)
        for members, representative in zip(clusters, representatives):
            distances = pc_axial_rmsd_matrix(features[members], features[representative])[..., 0]
            for index, distance in zip(members, distances):
                representative_distances[index] = float(distance)
    else:
        raise ValueError(f"unsupported skeleton dedup method: {method}")
    representatives_directory.mkdir(parents=True)
    cluster_records = []
    for cluster_index, (members, representative_index) in enumerate(zip(clusters, representatives), 1):
        representative = valid_records[representative_index]
        representative_path = representatives_directory / f"cluster-{cluster_index:06d}.extxyz"
        shutil.copyfile(representative["pose_structure"], representative_path)
        if _sha256_path(representative_path) != representative["pose_structure_sha256"]:
            raise RuntimeError("representative copy hash verification failed")
        cluster_record = {
            "cluster_index": cluster_index, "dedup_method": method, "member_count": len(members),
            "member_valid_record_indices": members,
            "member_safe_ranks": [valid_records[index]["safe_rank"] for index in members],
            "member_grid_indices": [valid_records[index]["grid_index"] for index in members],
            "representative_safe_rank": representative["safe_rank"],
            "representative_grid_index": representative["grid_index"],
            "representative_valid_record_index": representative_index,
            "representative_organic_z_min_A": representative["pose_organic_z_min_A"],
            "representative_structure": str(representative_path),
            "representative_structure_sha256": _sha256_path(representative_path),
            "representative_pose_force_rechecked": False,
            "representative_original_head_frame_pc_unit_vector": representative["original_head_frame_pc_unit_vector"],
            "representative_pc_tilt_to_head_normal_deg": representative["pc_tilt_to_head_normal_deg"],
            "representative_pc_length_A": representative["pc_length_A"],
            "representative_source_force_field_energy_kcal_mol": float(energies[representative_index]),
            "maximum_member_to_representative_axial_rmsd_A": max(representative_distances[index] for index in members),
            "rmsd_correspondence": ("minimum exact H0 heavy-graph automorphism" if method == "energy-radius-symmetry"
                                    else "fixed original atom IDs"),
        }
        if diameters is not None:
            cluster_record["maximum_pairwise_axial_rmsd_A"] = diameters[cluster_index - 1]
        for index in members:
            record = valid_records[index]
            record.update({
                "cluster_index": cluster_index, "representative_safe_rank": representative["safe_rank"],
                "source_force_field_energy_kcal_mol": float(energies[index]),
                "assigned_representative_rmsd_A": representative_distances[index],
                "assigned_representative_rmsd_correspondence": ("minimum exact H0 heavy-graph automorphism"
                                                                if method == "energy-radius-symmetry"
                                                                else "fixed original atom IDs"),
                "deduplication_status": "representative" if index == representative_index else "duplicate",
                "skeleton_dedup_method": method,
            })
        cluster_records.append(cluster_record)
    return cluster_records


def postprocess_skeleton_manifest(args: argparse.Namespace) -> int:
    """Audit a common P-C phase, then deduplicate optimized organic skeletons.

    先对所有有机原子（含 H）求共同高度/净空可行角，再按 P-C 相位不变
    的重原子三维 RMSD 去重。默认能量优先固定代表半径；显式保留历史
    complete-link 选项。不优化，不检验 ITO 碰撞。
    """

    from sam_structure_tools import canonicalize_pc_axis, phosphonate_axial_height_intervals

    ff_path = Path(args.skeleton_postprocess_manifest).expanduser().resolve()
    source_manifest = json.loads(ff_path.read_text(encoding="utf-8"))
    if (source_manifest.get("schema") != "sam-phosphonate-skeleton-ff-v1"
            or source_manifest.get("records_scope") != "optimized_skeleton_all_outcomes"):
        raise ValueError("postprocess input must be an isolated skeleton fixed-head FF manifest")
    limit = getattr(args, "limit", None)
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be positive")
    rmsd_tolerance = float(args.rmsd_tolerance_A)
    batch_size = int(getattr(args, "rmsd_batch_size", 256))
    dedup_method = str(getattr(args, "skeleton_dedup_method", "energy-radius-symmetry"))
    dedup_parameters = _skeleton_dedup_parameters(dedup_method)
    if rmsd_tolerance <= 0 or not np.isfinite(rmsd_tolerance) or batch_size <= 0:
        raise ValueError("RMSD tolerance and batch size must be positive and finite")
    sources = source_manifest["sources"]
    for path_key in ("cpu_manifest", "template_sdf", "h0_source"):
        source = Path(sources[path_key]).resolve()
        if not source.is_file() or _sha256_path(source) != sources[f"{path_key}_sha256"]:
            raise ValueError(f"recorded source hash changed: {source}")
    h0_symbols, h0_coordinates, _ = _read_extxyz_bytes(Path(sources["h0_source"]).read_bytes())
    if h0_symbols != source_manifest["system"]["sam_h0_formula_symbols"]:
        raise ValueError("H0 symbol order differs from FF manifest")
    topology = _h0_topology_contract(Path(sources["template_sdf"]), h0_symbols,
                                     getattr(args, "topology_python", None))
    if dedup_method == "energy-radius-symmetry":
        dedup_parameters["symmetry_automorphism_count"] = len(topology["rmsd_atom_automorphisms"])
    p_index, c_index = int(topology["p_index"]), int(topology["c_index"])
    head_indices = list(topology["head_indices"])
    if [index + 1 for index in head_indices] != source_manifest["system"]["fixed_head_atom_ids_1based"]:
        raise ValueError("SDF head mapping differs from FF manifest")
    oxygen_indices = [index for index in head_indices if index != p_index]
    numeric_tolerance_A = 1.0e-8
    if (np.max(np.abs(h0_coordinates[oxygen_indices, 2])) > numeric_tolerance_A
            or h0_coordinates[p_index, 2] <= 0.0):
        raise ValueError("H0 must retain the standardized three-O Z=0 frame with P above")
    bonds = {tuple(pair) for pair in topology["bond_pairs"]}
    adjacency = [set() for _ in h0_symbols]
    for left, right in bonds:
        adjacency[left].add(right)
        adjacency[right].add(left)
    organic_set, queue = {c_index}, [c_index]
    while queue:
        current = queue.pop()
        for neighbor in adjacency[current]:
            if neighbor == p_index or neighbor in organic_set:
                continue
            organic_set.add(neighbor)
            queue.append(neighbor)
    if organic_set != set(range(len(h0_symbols))) - set(head_indices):
        raise ValueError("P-C cut must give the complete organic fragment, excluding P/O head")
    organic_indices = sorted(organic_set)
    heavy_indices = [index for index in organic_indices if h0_symbols[index] != "H"]
    if not heavy_indices:
        raise ValueError("no organic heavy atoms remain for axial RMSD")
    pairs, thresholds, invariant = _skeleton_pair_contract(h0_symbols, bonds, organic_indices)
    records_path = Path(source_manifest["artifacts"]["optimized_records"]).resolve()
    records = [json.loads(line) for line in records_path.read_text().splitlines() if line]
    if len(records) != int(source_manifest["summary"]["unique_candidates_submitted"]):
        raise ValueError("optimized record count differs from FF manifest")
    records.sort(key=lambda record: int(record["safe_rank"]))
    if len({int(record["safe_rank"]) for record in records}) != len(records):
        raise ValueError("optimized safe ranks must be unique")
    # Validate recorded geometry bytes before making any output directory.
    for record in records:
        for path_key in ("structure", "optimized_structure"):
            if path_key not in record:
                continue
            source = Path(record[path_key]).resolve()
            if not source.is_file() or _sha256_path(source) != record[f"{path_key}_sha256"]:
                raise ValueError(f"recorded candidate hash changed: {source}")
    selected_records = records if limit is None else records[:limit]
    complete = (len(selected_records) == len(records)
                and bool(source_manifest["summary"].get("all_input_candidates_processed"))
                and bool(source_manifest["summary"].get("input_grid_complete")))
    force_gate = float(source_manifest["parameters"]["movable_fmax_eV_A"])
    if force_gate <= 0.0 or not np.isfinite(force_gate):
        raise ValueError("parent force convergence gate must be finite and positive")
    import sam_structure_tools
    source_payload = {
        "parent_fast_ff_manifest": str(ff_path), "parent_fast_ff_manifest_sha256": _sha256_path(ff_path),
        "parent_optimized_records": str(records_path), "parent_optimized_records_sha256": _sha256_path(records_path),
        "template_sdf": sources["template_sdf"], "template_sdf_sha256": sources["template_sdf_sha256"],
        "h0_source": sources["h0_source"], "h0_source_sha256": sources["h0_source_sha256"],
        "script": str(Path(__file__).resolve()), "script_sha256": _sha256_path(Path(__file__).resolve()),
        "geometry_helpers": str(Path(sam_structure_tools.__file__).resolve()),
        "geometry_helpers_sha256": _sha256_path(Path(sam_structure_tools.__file__).resolve()),
    }
    parameters = {
        "isomer_name": str(getattr(args, "isomer_name", None) or source_manifest["parameters"]["isomer_name"]),
        "rmsd_tolerance_A": rmsd_tolerance, "rmsd_batch_size": batch_size, "limit": limit,
        "rmsd_definition": ("minimum exact-graph-automorphism-mapped sqrt(mean_atom(sum_xyz(delta^2))) after proper P-C-axis canonicalization and one common axial phase fit"
                            if dedup_method == "energy-radius-symmetry" else
                            "sqrt(mean_atom(sum_xyz(delta^2))) after only proper P-C-axis canonicalization and one common axial phase fit"),
        "rmsd_atoms": ("organic heavy atoms excluding P and three O; minimum over exact source-H0 heavy-graph automorphisms preserving P, P-bonded C, and head/feature partition; original saved atom mapping unchanged"
                        if dedup_method == "energy-radius-symmetry" else
                        "organic heavy atoms excluding P and three O; original atom mapping fixed; no symmetry permutations"),
        "height_floor_A": 0.0, "height_atoms": "all C-side organic atoms including H",
        "numerical_tolerance_A": numeric_tolerance_A,
        "nonbonded_distance_floors_A": {"H-H": 1.2, "H-heavy": 1.4, "heavy-heavy": 1.8},
        "bonded_pairs": "only direct covalent bonds in reviewed SDF graph excluded",
        "phase_choice": "midpoint of widest circular connected common height-and-clearance arc; lowest normalized midpoint breaks exact ties",
        **dedup_parameters,
        "pose_force_rechecked": False, "optimization_performed": False,
        "force_field_metadata_scope": "source_optimized_result fields describe original optimized coordinates only; no force or energy evaluation of phase-rotated poses",
        "comparison_only_axis_alignment": "P is translated to origin and P-C axis aligned to +z; head-relative tilt removed only for shape comparison; full original head-frame poses retained",
        "source_movable_fmax_eV_A": force_gate,
    }
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", parameters["isomer_name"]):
        raise ValueError("isomer name must be a safe nonempty folder token")
    identity = _json_hash({"sources": source_payload, "parameters": parameters})
    run_directory = Path(args.output_root).expanduser().resolve() / f"skeleton-postprocess-{identity[:12]}"
    if run_directory.exists():
        raise ValueError(f"immutable skeleton postprocess directory already exists: {run_directory}")
    poses_directory = run_directory / f"{parameters['isomer_name']}_monomer_optimized_conformations"
    poses_directory.mkdir(parents=True)
    outcomes_path = run_directory / "postprocess-outcomes.jsonl"
    progress_path = run_directory / "progress.json"
    started = time.monotonic()
    outcomes, valid_records, features = [], [], []
    counts: dict[str, int] = {}

    def write_progress(stage: str, clustered_count: int = 0, cluster_count: int = 0) -> None:
        payload = {"stage": stage, "processed_count": len(outcomes),
                   "submitted_count": len(selected_records), "phase_safe_count": len(valid_records),
                   "outcome_counts": counts, "clustered_count": clustered_count,
                   "cluster_count": cluster_count, "elapsed_s": time.monotonic() - started}
        temporary = progress_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
        temporary.replace(progress_path)

    write_progress("shared_phase_audit_running")
    with outcomes_path.open("w", encoding="utf-8") as handle:
        for record in selected_records:
            updated = {
                "safe_rank": int(record["safe_rank"]), "grid_index": int(record["grid_index"]),
                "candidate_key": record["candidate_key"], "source_optimized_result": record,
                "pose_force_rechecked": False, "pose_optimized_convergence_claim": False,
            }
            fmax = float(record.get("force_field_max_gradient_non_head_eV_A", float("nan")))
            if (record.get("status") != "optimized_converged" or
                    not record.get("force_field_converged") or
                    not np.isfinite(fmax) or fmax > force_gate + 1.0e-12):
                updated["status"] = ("rejected_source_force_field_failed" if record.get("status") == "failed"
                                     else "rejected_source_force_field_not_converged")
            else:
                updated["source_force_field_energy_kcal_mol"] = _validated_source_force_field_energy(record, source_manifest)
                symbols, coordinates, _ = _read_extxyz_bytes(Path(record["optimized_structure"]).read_bytes())
                if symbols != h0_symbols:
                    raise ValueError(f"optimized atom order changed for safe_rank={record['safe_rank']}")
                input_symbols, input_coordinates, _ = _read_extxyz_bytes(Path(record["structure"]).read_bytes())
                if input_symbols != h0_symbols:
                    raise ValueError("source candidate atom order changed")
                head_displacement = float(np.max(np.linalg.norm(coordinates[head_indices] - input_coordinates[head_indices], axis=1)))
                if (head_displacement > FIXED_HEAD_DISPLACEMENT_TOLERANCE_A or
                        np.max(np.abs(coordinates[head_indices] - h0_coordinates[head_indices])) > FIXED_HEAD_DISPLACEMENT_TOLERANCE_A):
                    raise ValueError(f"fixed P/O coordinates changed for safe_rank={record['safe_rank']}")
                updated["fixed_head_source_displacement_A"] = head_displacement
                updated["source_organic_z_min_A"] = float(np.min(coordinates[organic_indices, 2]))
                pc_vector = coordinates[c_index] - coordinates[p_index]
                pc_length = float(np.linalg.norm(pc_vector))
                pc_unit = pc_vector / pc_length
                updated["original_head_frame_pc_unit_vector"] = pc_unit.tolist()
                updated["pc_tilt_to_head_normal_deg"] = float(np.degrees(np.arccos(np.clip(pc_unit[2], -1.0, 1.0))))
                updated["pc_length_A"] = pc_length
                invariant_pairs, invariant_thresholds = pairs[invariant], thresholds[invariant]
                invariant_distances = np.linalg.norm(coordinates[invariant_pairs[:, 0]] - coordinates[invariant_pairs[:, 1]], axis=1)
                collision_indices = np.flatnonzero(invariant_distances < invariant_thresholds - numeric_tolerance_A)
                if len(collision_indices):
                    first = int(collision_indices[0])
                    updated["status"] = "rejected_invariant_intramolecular_collision"
                    updated["collision"] = {"atom_ids_1based": (invariant_pairs[first] + 1).tolist(),
                                            "distance_A": float(invariant_distances[first]),
                                            "threshold_A": float(invariant_thresholds[first])}
                else:
                    phase = phosphonate_axial_height_intervals(
                        coordinates, p_index, c_index, organic_indices, z_floor=0.0,
                        tolerance=numeric_tolerance_A, collision_pairs_0based=pairs[~invariant],
                        collision_min_distances_A=thresholds[~invariant])
                    updated["shared_pc_phase"] = {key: phase[key] for key in (
                        "method", "feasible", "height_only_feasible", "height_only_intervals_deg",
                        "feasible_intervals_deg", "feasible_angle_measure_deg", "representative_angle_deg")}
                    if not phase["feasible"]:
                        updated["status"] = ("rejected_height_no_common_pc_phase" if not phase["height_only_feasible"]
                                             else "rejected_height_and_clearance_no_common_pc_phase")
                    else:
                        rotated = _rotate_organic_pc(coordinates, p_index, c_index, organic_indices,
                                                     float(phase["representative_angle_deg"]))
                        pose_path = poses_directory / f"candidate-{int(record['safe_rank']):06d}.extxyz"
                        _write_extxyz(pose_path, symbols, rotated,
                                      'Properties=species:S:1:pos:R:3 pbc="F F F" pose_force_rechecked=F')
                        _, written, _ = _read_extxyz_bytes(pose_path.read_bytes())
                        distances = np.linalg.norm(written[pairs[:, 0]] - written[pairs[:, 1]], axis=1)
                        height = float(np.min(written[organic_indices, 2]))
                        clearance = float(np.min(distances - thresholds))
                        final_head_displacement = float(np.max(np.linalg.norm(written[head_indices] - coordinates[head_indices], axis=1)))
                        if (height < -1.1e-8 or clearance < -1.1e-8 or
                                final_head_displacement > FIXED_HEAD_DISPLACEMENT_TOLERANCE_A):
                            raise RuntimeError("independent written-pose height/clearance/fixed-head verification failed")
                        updated.update({"status": "passed_shared_pc_phase_geometry", "pose_structure": str(pose_path),
                                        "pose_structure_sha256": _sha256_path(pose_path),
                                        "pose_organic_z_min_A": height, "pose_minimum_nonbonded_clearance_A": clearance,
                                        "pose_fixed_head_displacement_A": final_head_displacement})
                        valid_records.append(updated)
                        canonical, _ = canonicalize_pc_axis(coordinates, p_index, c_index)
                        features.append(canonical[heavy_indices])
            outcomes.append(updated)
            counts[updated["status"]] = counts.get(updated["status"], 0) + 1
            handle.write(json.dumps(updated, sort_keys=True) + "\n")
            if len(outcomes) % 100 == 0 or len(outcomes) == len(selected_records):
                handle.flush()
                write_progress("shared_phase_audit_running")
                print(f"shared P-C phase audited {len(outcomes)}/{len(selected_records)}; safe={len(valid_records)}", flush=True)

    def cluster_progress(clustered_count: int, cluster_count: int) -> None:
        write_progress(f"axial_{dedup_method}_dedup_running", clustered_count, cluster_count)
        print(f"P-C-invariant {dedup_method} processed {clustered_count}/{len(valid_records)}; representatives={cluster_count}", flush=True)

    representatives_directory = poses_directory / "representatives"
    cluster_records = _write_skeleton_dedup_products(
        valid_records, np.asarray(features), rmsd_tolerance, batch_size, dedup_method,
        representatives_directory, cluster_progress,
        atom_permutations=(topology["rmsd_atom_automorphisms"]
                           if dedup_method == "energy-radius-symmetry" else None))
    outcomes_path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in outcomes))
    valid_path = run_directory / "phase-safe-records.jsonl"
    valid_path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in valid_records))
    clusters_path = run_directory / "clusters.json"
    clusters_path.write_text(json.dumps(cluster_records, indent=2, sort_keys=True))
    summary = {"parent_ff_record_count": len(records), "processed_count": len(outcomes),
               "phase_safe_count": len(valid_records), "rejected_count": len(outcomes) - len(valid_records),
               "outcome_counts": counts, "cluster_count": len(cluster_records),
               "duplicate_phase_safe_count": len(valid_records) - len(cluster_records),
               "skeleton_dedup_method": dedup_method,
               "all_parent_records_processed": len(outcomes) == len(records), "input_grid_complete": complete,
               "parent_raw_negative_z_count_processed": sum(record.get("source_organic_z_min_A", 0.0) < 0.0 for record in outcomes),
               "raw_negative_z_rescued_count": sum(record["source_organic_z_min_A"] < 0.0 for record in valid_records),
               "source_failed_count_processed": sum(record.get("status") == "failed" for record in selected_records),
               "source_not_converged_count_processed": sum(record.get("status") == "optimized_not_converged" for record in selected_records),
               "optimization_started_count": 0, "ITO_collision_audit_count": 0,
               "elapsed_s": time.monotonic() - started}
    manifest = {
        "schema": "sam-phosphonate-skeleton-postprocess-v1", "schema_version": 1,
        "run_identity": identity, "status": "completed" if complete else "partial_completed",
        "scope": "isolated_skeleton_shared_phase_geometry_and_pc_invariant_clusters_not_ITO_audit",
        "records_scope": "post_optimization_skeleton_geometry_all_outcomes", "grid_complete": complete,
        "sources": source_payload, "parameters": parameters,
        "system": {**source_manifest["system"], "p_atom_id_1based": p_index + 1,
                   "c_atom_id_1based": c_index + 1,
                   "organic_atom_ids_1based": [index + 1 for index in organic_indices],
                   "rmsd_atom_ids_1based": [index + 1 for index in heavy_indices], "topology": topology},
        "summary": summary, "representatives": cluster_records,
        "artifacts": {"outcomes": str(outcomes_path), "phase_safe_records": str(valid_path),
                      "clusters": str(clusters_path), "poses_directory": str(poses_directory),
                      "representatives_directory": str(representatives_directory), "progress": str(progress_path)},
        "not_yet_validated": ["phase_rotated_pose_force_convergence", "ITO_collision", "surface_adsorption_energy"],
        "cluster_identity_scope": f"organic {rmsd_tolerance:g}-A geometry families under axis alignment and phase quotient; not strict FF energy basins or equivalent adsorption poses",
        "next_stage": "align_head_to_ITO_sites_then_surface_collision_screen_then_restrained_surface_optimization",
    }
    manifest_path = run_directory / "skeleton-postprocess-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    write_progress("completed" if complete else "partial_completed", len(valid_records), len(cluster_records))
    print(json.dumps({"stage": f"shared P-C phase audit and invariant skeleton {dedup_method} deduplication",
                      "manifest": str(manifest_path), "summary": summary}, indent=2), flush=True)
    return 0


def _read_jsonl_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _verified_hashed_source(path: str | Path, expected_sha256: str) -> Path:
    source = Path(path).expanduser().resolve()
    if not source.is_file() or _sha256_path(source) != expected_sha256:
        raise ValueError(f"recorded source hash changed: {source}")
    return source


def _load_skeleton_rededup_input(parent_path: Path, topology_python=None) -> dict:
    """Replay immutable source geometry and all members, without solving phases.

    Historical implementation paths are provenance, not mutable scientific
    input paths. Their recorded digests remain in the parent manifest; current
    implementations are independently identified in the new run.
    """

    from sam_structure_tools import canonicalize_pc_axis

    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if (parent.get("schema") != "sam-phosphonate-skeleton-postprocess-v1"
            or parent.get("status") != "completed"
            or parent.get("records_scope") != "post_optimization_skeleton_geometry_all_outcomes"):
        raise ValueError("rededup requires a completed isolated skeleton postprocess manifest")
    if parent.get("run_identity") != _json_hash({"sources": parent["sources"], "parameters": parent["parameters"]}):
        raise ValueError("postprocess parent source/parameter identity changed")
    parent_sources = parent["sources"]
    ff_path = _verified_hashed_source(parent_sources["parent_fast_ff_manifest"],
                                       parent_sources["parent_fast_ff_manifest_sha256"])
    ff_manifest = json.loads(ff_path.read_text(encoding="utf-8"))
    if ff_manifest.get("schema") != "sam-phosphonate-skeleton-ff-v1":
        raise ValueError("rededup parent FF schema is not isolated skeleton optimization")
    ff_sources = ff_manifest["sources"]
    for name in ("cpu_manifest", "template_sdf", "h0_source"):
        _verified_hashed_source(ff_sources[name], ff_sources[f"{name}_sha256"])
        if name in parent_sources and (Path(parent_sources[name]).resolve() != Path(ff_sources[name]).resolve()
                                       or parent_sources[f"{name}_sha256"] != ff_sources[f"{name}_sha256"]):
            raise ValueError(f"postprocess and FF source identities differ: {name}")
    ff_records_path = _verified_hashed_source(parent_sources["parent_optimized_records"],
                                               parent_sources["parent_optimized_records_sha256"])
    if ff_records_path != Path(ff_manifest["artifacts"]["optimized_records"]).resolve():
        raise ValueError("FF optimized records path differs from postprocess source")
    ff_records = _read_jsonl_records(ff_records_path)
    if len(ff_records) != int(ff_manifest["summary"]["unique_candidates_submitted"]):
        raise ValueError("FF source record count changed")
    ff_by_rank = {int(record["safe_rank"]): record for record in ff_records}
    if len(ff_by_rank) != len(ff_records):
        raise ValueError("FF safe ranks are not unique")
    for record in ff_records:
        for key in ("structure", "optimized_structure"):
            if key in record:
                _verified_hashed_source(record[key], record[f"{key}_sha256"])
    h0_symbols, h0_coordinates, _ = _read_extxyz_bytes(Path(ff_sources["h0_source"]).read_bytes())
    if (h0_symbols != ff_manifest["system"]["sam_h0_formula_symbols"]
            or h0_symbols != parent["system"]["sam_h0_formula_symbols"]
            or len(h0_symbols) != int(parent["system"]["sam_atom_count"])):
        raise ValueError("rededup source composition/atom order changed")
    topology = _h0_topology_contract(Path(ff_sources["template_sdf"]), h0_symbols, topology_python)
    for key in ("p_index", "c_index", "head_indices", "bond_pairs", "original_atom_ids_1based",
                "source_atom_count", "source_formal_charge", "h0_formal_charge"):
        if topology[key] != parent["system"]["topology"].get(key):
            raise ValueError(f"rededup original SDF atom mapping/topology changed: {key}")
    if ("rmsd_atom_automorphisms" in parent["system"]["topology"]
            and topology["rmsd_atom_automorphisms"] != parent["system"]["topology"]["rmsd_atom_automorphisms"]):
        raise ValueError("rededup exact source-graph symmetry mappings changed")
    p_index, c_index = topology["p_index"], topology["c_index"]
    head = topology["head_indices"]
    organic = sorted(set(range(len(h0_symbols))) - set(head))
    heavy = [index for index in organic if h0_symbols[index] != "H"]
    expected_mapping = {
        "fixed_head_atom_ids_1based": [index + 1 for index in head],
        "p_atom_id_1based": p_index + 1, "c_atom_id_1based": c_index + 1,
        "organic_atom_ids_1based": [index + 1 for index in organic],
        "rmsd_atom_ids_1based": [index + 1 for index in heavy],
    }
    if any(parent["system"].get(key) != values for key, values in expected_mapping.items()):
        raise ValueError("rededup original atom mapping changed")
    if not topology["rmsd_atom_automorphisms"] or any(
            len(permutation) != len(heavy) for permutation in topology["rmsd_atom_automorphisms"]):
        raise ValueError("exact source-graph symmetry maps do not match the RMSD heavy-atom mapping")
    if ff_manifest["system"]["fixed_head_atom_ids_1based"] != expected_mapping["fixed_head_atom_ids_1based"]:
        raise ValueError("FF and postprocess fixed-head atom mappings differ")
    pairs, floors, _ = _skeleton_pair_contract(h0_symbols, {tuple(pair) for pair in topology["bond_pairs"]}, organic)
    oxygen = [index for index in head if index != p_index]
    if np.max(np.abs(h0_coordinates[oxygen, 2])) > 1.0e-8 or h0_coordinates[p_index, 2] <= 0.0:
        raise ValueError("recorded H0 no longer defines the three-O Z=0 head frame")
    if (parent["parameters"].get("height_floor_A") != 0.0 or
            parent["parameters"].get("nonbonded_distance_floors_A") != {"H-H": 1.2, "H-heavy": 1.4, "heavy-heavy": 1.8}):
        raise ValueError("parent geometry gate differs from the reviewed fixed-head protocol")
    outcome_path = Path(parent["artifacts"]["outcomes"]).resolve()
    safe_path = Path(parent["artifacts"]["phase_safe_records"]).resolve()
    outcomes, safe = _read_jsonl_records(outcome_path), _read_jsonl_records(safe_path)
    outcomes_by_rank = {int(record["safe_rank"]): record for record in outcomes}
    if (len(outcomes_by_rank) != len(outcomes) or set(outcomes_by_rank) != set(ff_by_rank)
            or len(outcomes) != int(parent["summary"]["processed_count"])
            or len(outcomes) != int(parent["summary"]["parent_ff_record_count"])):
        raise ValueError("postprocess outcomes must cover all parent FF source records exactly once")
    for rank, outcome in outcomes_by_rank.items():
        if outcome.get("source_optimized_result") != ff_by_rank[rank]:
            raise ValueError("postprocess source force-field result differs from parent FF records")
    safe.sort(key=lambda record: int(record["safe_rank"]))
    if (len({int(record["safe_rank"]) for record in safe}) != len(safe)
            or len(safe) != int(parent["summary"]["phase_safe_count"])
            or {int(record["safe_rank"]) for record in safe} != {
                rank for rank, outcome in outcomes_by_rank.items() if outcome.get("status") == "passed_shared_pc_phase_geometry"}):
        raise ValueError("rededup must read every parent phase-safe member, not just representatives")
    gate = float(ff_manifest["parameters"]["movable_fmax_eV_A"])
    if not np.isfinite(gate) or gate <= 0.0:
        raise ValueError("parent FF force gate is invalid")
    features = []
    for record in safe:
        rank = int(record["safe_rank"])
        source = ff_by_rank[rank]
        if (source != record.get("source_optimized_result") or
                source.get("status") != "optimized_converged" or not source.get("force_field_converged") or
                not np.isfinite(source.get("force_field_max_gradient_non_head_eV_A", float("nan"))) or
                float(source["force_field_max_gradient_non_head_eV_A"]) > gate + 1.0e-12):
            raise ValueError("phase-safe member does not have a converged parent FF source")
        _validated_source_force_field_energy(source, ff_manifest)
        for key in ("source_optimized_result", "status", "pose_structure", "pose_structure_sha256",
                    "pose_force_rechecked", "pose_optimized_convergence_claim", "shared_pc_phase",
                    "original_head_frame_pc_unit_vector", "pc_tilt_to_head_normal_deg", "pc_length_A"):
            if record.get(key) != outcomes_by_rank[rank].get(key):
                raise ValueError(f"safe record and original outcome differ: {key}")
        if record.get("pose_force_rechecked") is not False or record.get("pose_optimized_convergence_claim") is not False:
            raise ValueError("phase pose must retain its explicit unverified-force status")
        pose_path = _verified_hashed_source(record["pose_structure"], record["pose_structure_sha256"])
        symbols, pose, _ = _read_extxyz_bytes(pose_path.read_bytes())
        source_symbols, optimized, _ = _read_extxyz_bytes(Path(source["optimized_structure"]).read_bytes())
        input_symbols, input_coordinates, _ = _read_extxyz_bytes(Path(source["structure"]).read_bytes())
        if symbols != h0_symbols or source_symbols != h0_symbols or input_symbols != h0_symbols:
            raise ValueError("replayed source/pose atom order changed")
        if (np.max(np.linalg.norm(pose[head] - optimized[head], axis=1)) > 1.0e-12 or
                np.max(np.linalg.norm(optimized[head] - input_coordinates[head], axis=1)) > 1.0e-12 or
                np.max(np.linalg.norm(optimized[head] - h0_coordinates[head], axis=1)) > 1.0e-12):
            raise ValueError("replayed pose or optimized fixed P/O head coordinates changed")
        distances = np.linalg.norm(pose[pairs[:, 0]] - pose[pairs[:, 1]], axis=1)
        if np.min(pose[organic, 2]) < -1.1e-8 or np.min(distances - floors) < -1.1e-8:
            raise ValueError("replayed written pose fails height/nonbonded geometry verification")
        angle = float(record["shared_pc_phase"]["representative_angle_deg"])
        expected_pose = _rotate_organic_pc(optimized, p_index, c_index, organic, angle)
        if np.max(np.abs(expected_pose - pose)) > 1.1e-10:
            raise ValueError("written pose differs from the recorded source P-C rotation")
        pc_vector = optimized[c_index] - optimized[p_index]
        pc_length = float(np.linalg.norm(pc_vector))
        pc_unit = pc_vector / pc_length
        pc_tilt = float(np.degrees(np.arccos(np.clip(pc_unit[2], -1.0, 1.0))))
        if (np.max(np.abs(pc_unit - record["original_head_frame_pc_unit_vector"])) > 1.0e-10 or
                abs(pc_length - float(record["pc_length_A"])) > 1.0e-10 or
                abs(pc_tilt - float(record["pc_tilt_to_head_normal_deg"])) > 1.0e-10):
            raise ValueError("retained original head-frame P-C geometry metadata changed")
        canonical, _ = canonicalize_pc_axis(optimized, p_index, c_index)
        features.append(canonical[heavy])
    return {"parent": parent, "ff_manifest": ff_manifest, "ff_path": ff_path,
            "ff_records_path": ff_records_path, "outcomes_path": outcome_path,
            "safe_records_path": safe_path, "outcomes": outcomes, "safe_records": safe,
            "features": np.asarray(features).reshape(len(safe), len(heavy), 3), "topology": topology}


def rededuplicate_skeleton_manifest(args: argparse.Namespace) -> int:
    """Reuse every audited safe pose and its original FF energy; no new phase solve."""

    import sam_structure_tools

    parent_path = Path(args.skeleton_rededup_manifest).expanduser().resolve()
    replay = _load_skeleton_rededup_input(parent_path, getattr(args, "topology_python", None))
    parent, ff_manifest = replay["parent"], replay["ff_manifest"]
    tolerance = float(args.rmsd_tolerance_A)
    batch_size = int(getattr(args, "rmsd_batch_size", 256))
    method = str(getattr(args, "skeleton_dedup_method", "energy-radius-symmetry"))
    limit = getattr(args, "limit", None)
    if tolerance <= 0.0 or not np.isfinite(tolerance) or batch_size <= 0 or (limit is not None and limit <= 0):
        raise ValueError("rededup RMSD radius, batch size, and optional limit must be positive")
    selected_count = len(replay["safe_records"]) if limit is None else min(limit, len(replay["safe_records"]))
    valid_records = [dict(record) for record in replay["safe_records"][:selected_count]]
    for record in valid_records:
        record["parent_cluster_index"] = record.pop("cluster_index", None)
    features = replay["features"][:selected_count]
    complete = selected_count == len(replay["safe_records"]) and bool(parent.get("grid_complete"))
    source_payload = {
        "parent_postprocess_manifest": str(parent_path), "parent_postprocess_manifest_sha256": _sha256_path(parent_path),
        "parent_safe_records": str(replay["safe_records_path"]), "parent_safe_records_sha256": _sha256_path(replay["safe_records_path"]),
        "parent_all_outcomes": str(replay["outcomes_path"]), "parent_all_outcomes_sha256": _sha256_path(replay["outcomes_path"]),
        "parent_fast_ff_manifest": str(replay["ff_path"]), "parent_fast_ff_manifest_sha256": _sha256_path(replay["ff_path"]),
        "parent_optimized_records": str(replay["ff_records_path"]), "parent_optimized_records_sha256": _sha256_path(replay["ff_records_path"]),
        "template_sdf": ff_manifest["sources"]["template_sdf"], "template_sdf_sha256": ff_manifest["sources"]["template_sdf_sha256"],
        "h0_source": ff_manifest["sources"]["h0_source"], "h0_source_sha256": ff_manifest["sources"]["h0_source_sha256"],
        "script": str(Path(__file__).resolve()), "script_sha256": _sha256_path(Path(__file__).resolve()),
        "geometry_helpers": str(Path(sam_structure_tools.__file__).resolve()),
        "geometry_helpers_sha256": _sha256_path(Path(sam_structure_tools.__file__).resolve()),
    }
    parameters = {
        **_skeleton_dedup_parameters(method), "rmsd_tolerance_A": tolerance,
        "rmsd_batch_size": batch_size, "limit": limit,
        "isomer_name": str(getattr(args, "isomer_name", None) or parent["parameters"]["isomer_name"]),
        "rmsd_definition": ("minimum exact-graph-automorphism-mapped sqrt(mean_atom(sum_xyz(delta^2))) after proper P-C-axis canonicalization and one common axial phase fit"
                            if method == "energy-radius-symmetry" else parent["parameters"]["rmsd_definition"]),
        "rmsd_atoms": ("organic heavy atoms excluding P and three O; minimum over exact source-H0 heavy-graph automorphisms preserving P, P-bonded C, and head/feature partition; original saved atom mapping unchanged"
                        if method == "energy-radius-symmetry" else parent["parameters"]["rmsd_atoms"]),
        "symmetry_automorphism_count": (len(replay["topology"]["rmsd_atom_automorphisms"])
                                        if method == "energy-radius-symmetry" else 0),
        "energy_scope": "original source FF optimized coordinates only; no energy computed for phase-rotated pose",
        "source_force_field": ff_manifest["parameters"]["force_field"],
        "source_force_field_actual_variant": ff_manifest["parameters"]["force_field_actual_variant"],
        "source_h0_formal_charge": replay["topology"]["h0_formal_charge"],
        "phase_recomputed": False, "optimization_performed": False, "pose_force_rechecked": False,
    }
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", parameters["isomer_name"]):
        raise ValueError("isomer name must be a safe nonempty folder token")
    identity = _json_hash({"sources": source_payload, "parameters": parameters})
    run_directory = Path(args.output_root).expanduser().resolve() / f"skeleton-dedup-{identity[:12]}"
    if run_directory.exists():
        raise ValueError(f"immutable skeleton dedup directory already exists: {run_directory}")
    run_directory.mkdir(parents=True)
    started = time.monotonic()
    progress_path = run_directory / "progress.json"

    def write_progress(processed: int, representative_count: int, status: str = "running") -> None:
        payload = {"status": status, "stage": ("replayed_energy_radius_symmetry_dedup" if method == "energy-radius-symmetry"
                                                 else "replayed_energy_radius_dedup" if method == "energy-radius"
                                                 else "replayed_complete_link_dedup"),
                   "processed_safe_count": processed, "eligible_safe_count": len(replay["safe_records"]),
                   "submitted_safe_count": selected_count, "representative_count": representative_count,
                   "phase_recomputed_count": 0, "optimization_started_count": 0,
                   "elapsed_s": time.monotonic() - started}
        temporary = progress_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
        temporary.replace(progress_path)
        print(f"replayed {method}: {processed}/{selected_count}; representatives={representative_count}", flush=True)

    write_progress(0, 0)
    representatives_directory = run_directory / f"{parameters['isomer_name']}_monomer_optimized_conformations" / "representatives"
    cluster_records = _write_skeleton_dedup_products(
        valid_records, features, tolerance, batch_size, method, representatives_directory, write_progress,
        atom_permutations=(replay["topology"]["rmsd_atom_automorphisms"]
                           if method == "energy-radius-symmetry" else None))
    safe_path = run_directory / "phase-safe-records.jsonl"
    safe_path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in valid_records))
    updated_by_rank = {record["safe_rank"]: record for record in valid_records}
    outcomes = []
    for outcome in replay["outcomes"]:
        updated = dict(updated_by_rank.get(int(outcome["safe_rank"]), outcome))
        if outcome["status"] == "passed_shared_pc_phase_geometry" and int(outcome["safe_rank"]) not in updated_by_rank:
            updated["deduplication_status"] = "not_processed_in_partial_run"
        outcomes.append(updated)
    outcomes_path = run_directory / "dedup-outcomes.jsonl"
    outcomes_path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in outcomes))
    clusters_path = run_directory / "clusters.json"
    clusters_path.write_text(json.dumps(cluster_records, indent=2, sort_keys=True))
    summary = {
        "eligible_phase_safe_count": len(replay["safe_records"]), "processed_phase_safe_count": selected_count,
        "phase_safe_count": selected_count, "cluster_count": len(cluster_records),
        "representative_count": len(cluster_records), "duplicate_phase_safe_count": selected_count - len(cluster_records),
        "parent_representative_count": int(parent["summary"]["cluster_count"]),
        "reused_source_outcome_count": len(outcomes),
        "retained_source_rejected_count": sum(record["status"] != "passed_shared_pc_phase_geometry" for record in outcomes),
        "all_eligible_safe_members_processed": selected_count == len(replay["safe_records"]),
        "input_grid_complete": complete, "parent_input_grid_complete": bool(parent.get("grid_complete")),
        "skeleton_dedup_method": method,
        "phase_recomputed_count": 0, "optimization_started_count": 0, "ITO_collision_audit_count": 0,
        "elapsed_s": time.monotonic() - started,
    }
    manifest = {
        "schema": "sam-phosphonate-skeleton-dedup-v1", "schema_version": 1,
        "run_identity": identity, "status": "completed" if complete else "partial_completed",
        "scope": "pure_dedup_replay_of_all_phase_safe_isolated_skeleton_members_no_optimization_or_phase_search",
        "records_scope": "phase_safe_members_with_dedup_mapping_and_complete_parent_outcomes", "grid_complete": complete,
        "sources": source_payload, "parameters": parameters,
        "system": {**parent["system"], "topology": {**parent["system"]["topology"],
                                                       "rmsd_atom_automorphisms": replay["topology"]["rmsd_atom_automorphisms"]}},
        "summary": summary, "representatives": cluster_records,
        "artifacts": {"outcomes": str(outcomes_path), "phase_safe_records": str(safe_path),
                      "clusters": str(clusters_path), "representatives_directory": str(representatives_directory),
                      "poses_directory": parent["artifacts"]["poses_directory"],
                      "poses_reference_scope": "all safe pose files remain in the hashed immutable parent; only new representatives copied",
                      "progress": str(progress_path)},
        "cluster_identity_scope": f"organic {tolerance:g}-A {method} shape families under PC-axis alignment and phase quotient; not strict FF basins or equivalent adsorption poses",
        "not_yet_validated": ["phase_rotated_pose_force_convergence", "ITO_collision", "surface_adsorption_energy"],
        "next_stage": parent["next_stage"],
    }
    manifest_path = run_directory / "skeleton-dedup-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    write_progress(selected_count, len(cluster_records), manifest["status"])
    print(json.dumps({"manifest": str(manifest_path), "summary": summary}, indent=2), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-manifest", type=Path)
    parser.add_argument("--template-sdf", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--optimization-only",
        action="store_true",
        help="Optimize isolated skeletons and record convergence; do not cluster or imply an ITO audit",
    )
    parser.add_argument(
        "--isomer-name",
        help="Isomer token for <isomer>_monomer_optimized_conformations (infer from input folder when possible)",
    )
    parser.add_argument(
        "--movable-fmax-eV-A",
        dest="movable_fmax_eV_A",
        type=float,
        default=DEFAULT_MOVABLE_FMAX_EV_A,
        help="Independent maximum movable-atom gradient gate in eV/A, used by --optimization-only (default: 0.03)",
    )
    parser.add_argument(
        "--recluster-ff-manifest",
        type=Path,
        help="Reuse a completed FF manifest and apply the post-FF geometry gate",
    )
    parser.add_argument(
        "--recluster-audit-manifest",
        type=Path,
        help="Cluster structures that already passed the fixed-site substrate audit",
    )
    parser.add_argument(
        "--skeleton-postprocess-manifest", type=Path,
        help="Audit common continuous P-C height/clearance phases and deduplicate converged isolated skeletons; no optimization",
    )
    parser.add_argument(
        "--skeleton-rededup-manifest", type=Path,
        help="Reuse every safe member of a completed skeleton postprocess stage; no repeated optimization or phase search",
    )
    parser.add_argument(
        "--skeleton-dedup-method", choices=("energy-radius-symmetry", "energy-radius", "complete-link"),
        default="energy-radius-symmetry",
        help="Skeleton method: exact graph-symmetry-aware energy radius (default); legacy fixed-atom-map radius; or historical complete-link",
    )
    parser.add_argument(
        "--topology-python", type=Path,
        help="Python with RDKit for a single exact source-SDF graph adapter when geometry runtime lacks RDKit",
    )
    parser.add_argument(
        "--rmsd-batch-size", type=int, default=256,
        help="Bounded RMSD row batch for skeleton postprocessing (default: 256)",
    )
    parser.add_argument("--workers", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    parser.add_argument("--chunksize", type=int, default=32)
    parser.add_argument(
        "--force-field",
        choices=("uff", "mmff94", "mmff94s"),
        default="uff",
        help="Isolated RDKit force field (default: uff; MMFF uses the named variant)",
    )
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument("--rmsd-tolerance-A", type=float, default=DEFAULT_RMSD_TOLERANCE_A)
    parser.add_argument("--limit", type=int, help="Run only the first N deduplicated candidates for a pilot")
    args = parser.parse_args()
    if args.workers <= 0 or args.chunksize <= 0 or args.max_iterations <= 0:
        parser.error("workers, chunksize, and max-iterations must be positive")
    if args.rmsd_tolerance_A <= 0 or not np.isfinite(args.rmsd_tolerance_A):
        parser.error("rmsd-tolerance-A must be positive and finite")
    if args.movable_fmax_eV_A <= 0 or not np.isfinite(args.movable_fmax_eV_A):
        parser.error("movable-fmax-eV-A must be positive and finite")
    reuse_modes = [args.recluster_ff_manifest, args.recluster_audit_manifest,
                   args.skeleton_postprocess_manifest, args.skeleton_rededup_manifest]
    if sum(mode is not None for mode in reuse_modes) > 1:
        parser.error("select only one completed-manifest processing mode")
    if args.optimization_only and any(mode is not None for mode in reuse_modes):
        parser.error("--optimization-only cannot be combined with a reclustering mode")
    if args.skeleton_postprocess_manifest is not None:
        try:
            return postprocess_skeleton_manifest(args)
        except (OSError, ValueError, RuntimeError, ImportError, subprocess.SubprocessError) as exc:
            print(f"fast_ff_prefilter skeleton postprocess failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
    if args.skeleton_rededup_manifest is not None:
        try:
            return rededuplicate_skeleton_manifest(args)
        except (OSError, ValueError, RuntimeError, ImportError, subprocess.SubprocessError) as exc:
            print(f"fast_ff_prefilter skeleton rededup failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
    if args.recluster_ff_manifest is not None:
        try:
            return _recluster_existing_ff_manifest(args)
        except (OSError, ValueError, RuntimeError) as exc:
            print(
                f"fast_ff_prefilter recluster failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return 2
    if args.recluster_audit_manifest is not None:
        try:
            return _recluster_audit_manifest(args)
        except (OSError, ValueError, RuntimeError) as exc:
            print(
                f"fast_ff_prefilter audit recluster failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return 2
    if args.cpu_manifest is None or args.template_sdf is None:
        parser.error(
            "the normal FF stage requires --cpu-manifest and --template-sdf; "
            "use --recluster-ff-manifest for a completed FF stage"
        )
    try:
        return run(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"fast_ff_prefilter failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
