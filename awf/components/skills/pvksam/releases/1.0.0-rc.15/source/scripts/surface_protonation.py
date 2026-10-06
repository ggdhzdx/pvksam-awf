#!/usr/bin/env python3
"""Globally place surface protons after every SAM molecule is fixed.

The input contract is a clean substrate followed by contiguous, equal-sized SAM
blocks.  Substrate size, molecule count, block size, proton stoichiometry, anchor
element, and geometric thresholds are CLI parameters.  The solver builds
collision-free proton groups near each headgroup, rewards multi-O hydrogen-bond
contacts, and selects all groups globally without insertion-order bias.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from ase import Atoms
from ase.io import read, write
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from sam_structure_tools import distance_within_closed_window


@dataclass(frozen=True)
class ProtonationConfig:
    substrate_atoms: int
    molecule_count: int
    atoms_per_molecule: int
    protons_per_molecule: int
    anchor_element: str = "P"
    anchor_atom_index_within_molecule_0based: int | None = None
    parent_element: str = "O"
    acceptor_element: str = "O"
    headgroup_count: int = 3
    headgroup_bond_cutoff: float = 1.85
    oh_bond: float = 0.98
    hh_min: float = 1.20
    h_heavy_min: float = 1.40
    hbond_max: float = 2.40
    hbond_angle_min: float = 105.0
    parent_xy_max: float = 5.80
    parent_dz_min: float = 0.25
    parent_dz_max: float = 4.50
    acceptor_parent_min: float = 2.35
    acceptor_parent_max: float = 3.65
    surface_h_clearance: float = 0.0
    surface_normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
    surface_periodic_axes: tuple[int, int] = (0, 1)
    frozen_headgroup_local_indices: tuple[int, ...] | None = None


def mic_vectors(
    origin: np.ndarray,
    targets: np.ndarray,
    cell: np.ndarray,
    periodic_axes: tuple[int, ...] = (0, 1, 2),
) -> np.ndarray:
    inv_cell = np.linalg.inv(cell)
    fractional = (targets - origin) @ inv_cell
    for axis in periodic_axes:
        fractional[:, axis] -= np.round(fractional[:, axis])
    return fractional @ cell


def unique_directions(directions: list[np.ndarray]) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    seen: set[tuple[float, float, float]] = set()
    for direction in directions:
        norm = float(np.linalg.norm(direction))
        if norm < 1.0e-8:
            continue
        unit = direction / norm
        if unit[2] < -0.10:
            continue
        key = tuple(np.round(unit, 5))
        if key not in seen:
            seen.add(key)
            result.append(unit)
    return result


def headgroup_indices(
    atoms: Atoms, molecule_number: int, config: ProtonationConfig
) -> tuple[int, np.ndarray, np.ndarray]:
    start = config.substrate_atoms + molecule_number * config.atoms_per_molecule
    block = np.arange(start, start + config.atoms_per_molecule, dtype=int)
    symbols = np.asarray(atoms.get_chemical_symbols())
    if config.anchor_atom_index_within_molecule_0based is None:
        anchor = block[symbols[block] == config.anchor_element]
        if len(anchor) != 1:
            raise ValueError(
                f"SAM {molecule_number + 1} has {len(anchor)} "
                f"{config.anchor_element} anchor atoms"
            )
        anchor_index = int(anchor[0])
    else:
        local_anchor = int(config.anchor_atom_index_within_molecule_0based)
        if not 0 <= local_anchor < config.atoms_per_molecule:
            raise ValueError("local anchor atom index is outside the SAM block")
        anchor_index = int(block[local_anchor])
        if symbols[anchor_index] != config.anchor_element:
            raise ValueError("local anchor atom element does not match the contract")
    if config.frozen_headgroup_local_indices is not None:
        local = np.asarray(config.frozen_headgroup_local_indices, dtype=int)
        if (len(local) != config.headgroup_count or len(set(local)) != len(local)
                or np.any(local < 0) or np.any(local >= config.atoms_per_molecule)):
            raise ValueError("Invalid sealed headgroup identities")
        indices = block[local]
        if not np.all(symbols[indices] == config.acceptor_element):
            raise ValueError("Sealed headgroup element mismatch")
        return anchor_index, indices, block
    acceptors = block[symbols[block] == config.acceptor_element]
    vectors = mic_vectors(
        atoms.positions[anchor_index],
        atoms.positions[acceptors],
        np.asarray(atoms.cell),
        config.surface_periodic_axes,
    )
    distances = np.linalg.norm(vectors, axis=1)
    head_acceptors = acceptors[distances < config.headgroup_bond_cutoff]
    if len(head_acceptors) != config.headgroup_count:
        raise ValueError(
            f"SAM {molecule_number + 1} has {len(head_acceptors)} covalent "
            f"{config.anchor_element}-{config.acceptor_element} headgroup atoms; "
            f"expected {config.headgroup_count}"
        )
    return anchor_index, head_acceptors, block


def evaluate_position(
    atoms: Atoms,
    parent_o: int,
    h_position: np.ndarray,
    all_acceptors: np.ndarray,
    head_acceptors: set[int],
    minimum_h_z: float,
    config: ProtonationConfig,
) -> dict | None:
    normal = np.asarray(config.surface_normal, dtype=float)
    normal /= np.linalg.norm(normal)
    height_above_surface = float(np.dot(h_position, normal) - minimum_h_z)
    if height_above_surface < 0.0:
        return None
    symbols = np.asarray(atoms.get_chemical_symbols())
    cell = np.asarray(atoms.cell)
    vectors = mic_vectors(
        h_position, atoms.positions, cell, config.surface_periodic_axes
    )
    distances = np.linalg.norm(vectors, axis=1)
    keep = np.ones(len(atoms), dtype=bool)
    keep[parent_o] = False
    thresholds = np.where(symbols == "H", config.hh_min, config.h_heavy_min)
    ratios = distances[keep] / thresholds[keep]
    minimum_ratio = float(np.min(ratios))
    if minimum_ratio < 1.0:
        return None

    parent_vector = mic_vectors(
        h_position,
        atoms.positions[[parent_o]],
        cell,
        config.surface_periodic_axes,
    )[0]
    parent_vector /= np.linalg.norm(parent_vector)
    interactions = []
    for o_index in all_acceptors:
        if int(o_index) == parent_o:
            continue
        distance = float(distances[o_index])
        if not distance_within_closed_window(
            distance,
            config.h_heavy_min,
            config.hbond_max,
        ):
            continue
        acceptor_vector = vectors[o_index] / distance
        angle = float(
            np.degrees(
                np.arccos(np.clip(np.dot(parent_vector, acceptor_vector), -1.0, 1.0))
            )
        )
        if angle < config.hbond_angle_min:
            continue
        distance_score = float(np.exp(-((distance - 1.75) / 0.45) ** 2))
        angle_score = min(
            1.0, max(0.0, (angle - config.hbond_angle_min) / 65.0)
        )
        weight = 1.20 if int(o_index) in head_acceptors else 1.0
        interactions.append(
            {
                "o_index": int(o_index),
                "distance_A": distance,
                "angle_deg": angle,
                "score": weight * distance_score * (0.35 + 0.65 * angle_score),
                "is_same_SAM_headgroup_acceptor": int(o_index) in head_acceptors,
            }
        )
    if not interactions:
        return None
    interactions.sort(key=lambda item: (-item["score"], item["distance_A"]))
    coordination_count = len(interactions)
    interaction_score = float(sum(item["score"] for item in interactions))
    return {
        "position": h_position,
        "height_above_surface_plane_A": height_above_surface,
        "minimum_clearance_ratio": minimum_ratio,
        "coordination_count": coordination_count,
        "interaction_score": interaction_score,
        "interactions": interactions,
    }


def single_candidates_for_sam(
    atoms: Atoms,
    molecule_number: int,
    substrate_parent_atoms: np.ndarray,
    all_acceptors: np.ndarray,
    parents_per_sam: int,
    directions_per_parent: int,
    minimum_h_z: float,
    config: ProtonationConfig,
) -> tuple[list[dict], dict]:
    cell = np.asarray(atoms.cell)
    anchor_index, head_acceptor_indices, _ = headgroup_indices(
        atoms, molecule_number, config
    )
    anchor_to_parent = mic_vectors(
        atoms.positions[anchor_index],
        atoms.positions[substrate_parent_atoms],
        cell,
        config.surface_periodic_axes,
    )
    normal = np.asarray(config.surface_normal, dtype=float)
    normal /= np.linalg.norm(normal)
    normal_components = anchor_to_parent @ normal
    lateral_vectors = anchor_to_parent - normal_components[:, None] * normal
    xy = np.linalg.norm(lateral_vectors, axis=1)
    anchor_minus_parent_z = -normal_components
    local = np.flatnonzero(
        (xy <= config.parent_xy_max)
        & (anchor_minus_parent_z >= config.parent_dz_min)
        & (anchor_minus_parent_z <= config.parent_dz_max)
    )
    association_cost = (xy / 4.0) ** 2 + (
        (anchor_minus_parent_z - 2.2) / 1.5
    ) ** 2
    if len(local) < 8:
        local = np.argsort(association_cost)[: max(8, parents_per_sam)]
    local = local[np.argsort(association_cost[local])[:parents_per_sam]]

    head_acceptor_set = set(int(index) for index in head_acceptor_indices)
    singles: list[dict] = []
    parent_diagnostics = []
    for local_index in local:
        parent_atom = int(substrate_parent_atoms[local_index])
        parent_to_acceptors = mic_vectors(
            atoms.positions[parent_atom],
            atoms.positions[all_acceptors],
            cell,
            config.surface_periodic_axes,
        )
        parent_acceptor_distances = np.linalg.norm(parent_to_acceptors, axis=1)
        acceptor_in_window = np.fromiter(
            (
                distance_within_closed_window(
                    float(distance),
                    config.acceptor_parent_min,
                    config.acceptor_parent_max,
                )
                for distance in parent_acceptor_distances
            ),
            dtype=bool,
            count=len(parent_acceptor_distances),
        )
        acceptor_local = np.flatnonzero(
            (all_acceptors != parent_atom)
            & acceptor_in_window
        )
        acceptor_local = acceptor_local[
            np.argsort(parent_acceptor_distances[acceptor_local])[:10]
        ]
        base_directions = [parent_to_acceptors[index] for index in acceptor_local]
        unit_base = unique_directions(base_directions)
        combined = list(unit_base)
        for first, second in itertools.combinations(unit_base[:8], 2):
            combined.append(first + second)
        for direction in unit_base[:8]:
            combined.append(0.80 * direction + 0.20 * normal)
        combined.append(normal)
        directions = unique_directions(combined)

        parent_records = []
        for direction in directions:
            h_position = atoms.positions[parent_atom] + config.oh_bond * direction
            evaluated = evaluate_position(
                atoms,
                parent_atom,
                h_position,
                all_acceptors,
                head_acceptor_set,
                minimum_h_z,
                config,
            )
            if evaluated is None:
                continue
            # Strongly prefer additional O coordination; the smaller terms
            # break ties by interaction geometry, locality, and clearance.
            cost = (
                float(association_cost[local_index]) * 0.12
                - 4.0 * min(evaluated["coordination_count"], 3)
                - evaluated["interaction_score"]
                - 0.05 * min(evaluated["minimum_clearance_ratio"], 2.0)
            )
            parent_records.append(
                {
                    "molecule_number": molecule_number,
                    "anchor_index": anchor_index,
                    "parent_atom": parent_atom,
                    "position": h_position,
                    "direction": direction,
                    "anchor_parent_xy_A": float(xy[local_index]),
                    "anchor_minus_parent_z_A": float(
                        anchor_minus_parent_z[local_index]
                    ),
                    "cost": cost,
                    **evaluated,
                }
            )
        parent_records.sort(
            key=lambda item: (
                item["cost"],
                -item["coordination_count"],
                -item["minimum_clearance_ratio"],
            )
        )
        singles.extend(parent_records[:directions_per_parent])
        parent_diagnostics.append(
            {
                "parent_atom": parent_atom,
                "generated_direction_count": len(directions),
                "feasible_direction_count": len(parent_records),
            }
        )

    singles.sort(key=lambda item: item["cost"])
    diagnostics = {
        "anchor_index": anchor_index,
        "local_parent_count": len(local),
        "single_candidate_count": len(singles),
        "parents_with_candidates": len({item["parent_atom"] for item in singles}),
        "parent_diagnostics": parent_diagnostics,
    }
    return singles, diagnostics


def proton_group_candidates_for_sam(
    singles: list[dict],
    cell: np.ndarray,
    groups_per_sam: int,
    config: ProtonationConfig,
) -> list[dict]:
    # Keep the best directional realization of every distinct parent-atom group.
    # Merely truncating all directional groups by score overrepresents a few
    # attractive sites and can make the later global unique-parent problem
    # artificially infeasible.
    best_by_parent_group: dict[tuple[int, ...], dict] = {}
    for member_numbers in itertools.combinations(
        range(len(singles)), config.protons_per_molecule
    ):
        protons = tuple(singles[index] for index in member_numbers)
        parent_atoms = tuple(proton["parent_atom"] for proton in protons)
        if len(set(parent_atoms)) != config.protons_per_molecule:
            continue
        minimum_hh = np.inf
        for first, second in itertools.combinations(protons, 2):
            distance = float(
                np.linalg.norm(
                    mic_vectors(
                        first["position"],
                        second["position"][None, :],
                        cell,
                        config.surface_periodic_axes,
                    )[0]
                )
            )
            minimum_hh = min(minimum_hh, distance)
        if minimum_hh < config.hh_min:
            continue
        acceptor_union = {
            item["o_index"]
            for proton in protons
            for item in proton["interactions"]
        }
        group_cost = (
            sum(proton["cost"] for proton in protons)
            - 0.20 * min(len(acceptor_union), 2 * config.protons_per_molecule)
        )
        record = {
            "molecule_number": protons[0]["molecule_number"],
            "protons": protons,
            "parent_atoms": parent_atoms,
            "intramolecular_min_HH_A": float(minimum_hh),
            "distinct_acceptor_count": len(acceptor_union),
            "cost": group_cost,
        }
        parent_group = tuple(sorted(parent_atoms))
        incumbent = best_by_parent_group.get(parent_group)
        if incumbent is None or record["cost"] < incumbent["cost"]:
            best_by_parent_group[parent_group] = record
    groups = list(best_by_parent_group.values())
    groups.sort(
        key=lambda item: (
            item["cost"],
            -item["distinct_acceptor_count"],
            -item["intramolecular_min_HH_A"],
        )
    )
    return groups[:groups_per_sam]


def group_group_hh_distance(
    first: dict,
    second: dict,
    cell: np.ndarray,
    periodic_axes: tuple[int, ...] = (0, 1, 2),
) -> float:
    first_positions = np.asarray([item["position"] for item in first["protons"]])
    second_positions = np.asarray([item["position"] for item in second["protons"]])
    minimum = np.inf
    for position in first_positions:
        vectors = mic_vectors(position, second_positions, cell, periodic_axes)
        minimum = min(minimum, float(np.min(np.linalg.norm(vectors, axis=1))))
    return minimum


def solve_global_groups(
    domains: list[list[dict]], cell: np.ndarray, config: ProtonationConfig
) -> tuple[list[dict], dict]:
    variables = []
    for molecule_number, domain in enumerate(domains):
        for group in domain:
            variables.append({**group, "molecule_number": molecule_number})
    objective = np.asarray([item["cost"] for item in variables], dtype=float)
    forbidden: set[tuple[int, int]] = set()

    for round_number in range(1, 101):
        rows: list[int] = []
        cols: list[int] = []
        values: list[float] = []
        lower: list[float] = []
        upper: list[float] = []
        row = 0
        for molecule_number in range(config.molecule_count):
            relevant = [
                index
                for index, item in enumerate(variables)
                if item["molecule_number"] == molecule_number
            ]
            rows.extend([row] * len(relevant))
            cols.extend(relevant)
            values.extend([1.0] * len(relevant))
            lower.append(1.0)
            upper.append(1.0)
            row += 1

        parent_to_variables: dict[int, list[int]] = {}
        for index, item in enumerate(variables):
            for parent_atom in item["parent_atoms"]:
                parent_to_variables.setdefault(parent_atom, []).append(index)
        for relevant in parent_to_variables.values():
            if len(relevant) < 2:
                continue
            rows.extend([row] * len(relevant))
            cols.extend(relevant)
            values.extend([1.0] * len(relevant))
            lower.append(-np.inf)
            upper.append(1.0)
            row += 1

        for first, second in sorted(forbidden):
            rows.extend((row, row))
            cols.extend((first, second))
            values.extend((1.0, 1.0))
            lower.append(-np.inf)
            upper.append(1.0)
            row += 1

        matrix = coo_matrix(
            (values, (rows, cols)), shape=(row, len(variables))
        ).tocsr()
        result = milp(
            c=objective,
            integrality=np.ones(len(variables), dtype=int),
            bounds=Bounds(np.zeros(len(variables)), np.ones(len(variables))),
            constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
            options={"time_limit": 180.0, "mip_rel_gap": 0.0},
        )
        if result.x is None or not result.success:
            raise RuntimeError(
                f"Global proton-group assignment failed: status={result.status}, "
                f"message={result.message}"
            )
        chosen_numbers = np.flatnonzero(result.x > 0.5)
        chosen = [variables[int(index)] for index in chosen_numbers]
        new_forbidden = set()
        for offset, first_number in enumerate(chosen_numbers):
            first = variables[int(first_number)]
            for second_number in chosen_numbers[offset + 1 :]:
                second = variables[int(second_number)]
                if group_group_hh_distance(
                    first, second, cell, config.surface_periodic_axes
                ) < config.hh_min:
                    new_forbidden.add(
                        tuple(sorted((int(first_number), int(second_number))))
                    )
        if not new_forbidden:
            chosen.sort(key=lambda item: item["molecule_number"])
            return chosen, {
                "rounds": round_number,
                "variable_count": len(variables),
                "lazy_HH_conflict_count": len(forbidden),
                "objective": float(result.fun),
            }
        forbidden.update(new_forbidden)
    raise RuntimeError("Global proton assignment exceeded 100 lazy-HH rounds")


def clean_lammps_masses_comments(path: Path) -> None:
    lines = path.read_text().splitlines()
    in_masses = False
    cleaned = []
    for line in lines:
        if line.strip() == "Masses":
            in_masses = True
        elif line.startswith("Atoms"):
            in_masses = False
        if in_masses and "#" in line:
            line = line.split("#", 1)[0].rstrip()
        cleaned.append(line)
    path.write_text("\n".join(cleaned) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_promoted_h0_parent(input_h0: Path, promotion_manifest_path: Path) -> dict:
    """Verify that protonation consumes the exact H0 sealed by physical promotion."""

    input_h0 = Path(input_h0).expanduser().resolve()
    promotion_manifest_path = Path(promotion_manifest_path).expanduser().resolve()
    try:
        manifest = json.loads(promotion_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("cannot read physical-interface promotion manifest") from exc
    promotion_schema = manifest.get("schema")
    if promotion_schema not in {
        "samflow-sized-h0-physical-interface-promotion-v1",
        "samflow-geometric-h0-physical-interface-promotion-v1",
    }:
        raise ValueError("unsupported physical-interface promotion manifest schema")
    if manifest.get("chemistry_state") != "H0_surface_protons_zero":
        raise ValueError("protonation requires an H0 physical-promotion parent")
    if not (
        manifest.get("promotion_eligible") is True
        and manifest.get("physical_interface_pass") is True
    ):
        raise ValueError("physical-interface parent is not promotion eligible")
    is_geometric = promotion_schema == (
        "samflow-geometric-h0-physical-interface-promotion-v1"
    )
    basis_hash = (
        manifest.get("promotion_basis_sha256")
        if is_geometric
        else manifest.get("parent", {}).get("approval_plan_sha256")
    )
    if (
        not isinstance(basis_hash, str)
        or len(basis_hash) != 64
        or any(character not in "0123456789abcdef" for character in basis_hash)
    ):
        raise ValueError("physical-interface parent lacks a valid promotion basis hash")
    parent_record = manifest.get("parent", {})
    implementation_identities = parent_record.get(
        "implementation_identities"
        if is_geometric
        else "approval_implementation_identities",
        {},
    )
    if implementation_identities.get(
        "surface_protonation_implementation_sha256"
    ) != _sha256(Path(__file__).resolve()):
        raise ValueError(
            "surface protonation implementation changed after the sized monolayer "
            "plan was resolved"
        )
    if is_geometric and implementation_identities.get(
        "interface_validator_implementation_sha256"
    ) != _sha256(Path(__file__).with_name("validate_monolayer.py")):
        raise ValueError("geometric H0 interface validator changed after promotion")

    structure_record = manifest.get("structure", {})
    recorded_structure = promotion_manifest_path.parent / str(
        structure_record.get("path", "")
    )
    expected_structure_hash = structure_record.get("sha256")
    if not recorded_structure.is_file() or _sha256(recorded_structure) != (
        expected_structure_hash
    ):
        raise ValueError("promoted H0 structure hash mismatch")
    if not input_h0.is_file() or _sha256(input_h0) != expected_structure_hash:
        raise ValueError("protonation input structure hash mismatch")

    validation_record = manifest.get("validation", {})
    validation_path = promotion_manifest_path.parent / str(
        validation_record.get("path", "")
    )
    if not validation_path.is_file() or _sha256(validation_path) != validation_record.get(
        "sha256"
    ):
        raise ValueError("physical-interface validation hash mismatch")
    try:
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("physical-interface validation is not valid JSON") from exc
    if validation.get("physical_interface_gate", {}).get(
        "physical_interface_pass"
    ) is not True:
        raise ValueError("physical-interface validation does not pass its promotion gate")
    if is_geometric:
        requirements_record = manifest.get("protonation_requirements", {})
        requirements_path = promotion_manifest_path.parent / str(
            requirements_record.get("path", "")
        )
        if not requirements_path.is_file() or _sha256(requirements_path) != (
            requirements_record.get("sha256")
        ):
            raise ValueError("geometric H0 protonation requirements hash mismatch")
        requirements = json.loads(requirements_path.read_text(encoding="utf-8"))
        if (
            requirements.get("schema") != "samflow-surface-proton-requirements-v1"
            or requirements.get("assignment_policy")
            != "global_solver_after_physical_interface_promotion"
            or requirements.get("fixed_parent_assignments") != []
        ):
            raise ValueError("geometric H0 protonation requirements are unsupported")
        geometry = manifest.get("geometry", {})
        geometry_counts = (
            geometry.get("substrate_atom_count"),
            geometry.get("molecule_count"),
            geometry.get("atoms_per_sam_h0"),
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in geometry_counts
        ) or geometry.get("surface_proton_count") != 0:
            raise ValueError("geometric H0 promotion has an invalid atom layout")
        if structure_record.get("atom_count") != (
            geometry_counts[0] + geometry_counts[1] * geometry_counts[2]
        ):
            raise ValueError("geometric H0 promotion atom count disagrees with its layout")
        released_per_sam = requirements.get("released_protons_per_sam")
        required_total = requirements.get("required_surface_proton_count")
        if (
            isinstance(released_per_sam, bool)
            or not isinstance(released_per_sam, int)
            or released_per_sam <= 0
            or required_total != geometry_counts[1] * released_per_sam
        ):
            raise ValueError("geometric H0 proton inventory disagrees with its layout")
        contract = manifest.get("interface_validation_contract", {})
        if (
            contract.get("schema")
            != "samflow-phosphonate-o-triangle-height-contract-v1"
            or contract.get("anchor_oxygen_count") != 3
            or contract.get("exact_oxygen_metal_pair_required") is not False
        ):
            raise ValueError("geometric H0 promotion has an unsupported interface contract")
        source_hash = parent_record.get("source_structure_sha256")
        if (
            not isinstance(source_hash, str)
            or len(source_hash) != 64
            or any(character not in "0123456789abcdef" for character in source_hash)
        ):
            raise ValueError("geometric H0 promotion lacks its source structure hash")
        basis_payload = {
            "source_structure_sha256": source_hash,
            "interface_contract": contract,
            "geometry": geometry,
            "protonation_requirements": requirements,
            "validator_implementation_sha256": implementation_identities[
                "interface_validator_implementation_sha256"
            ],
            "surface_protonation_implementation_sha256": implementation_identities[
                "surface_protonation_implementation_sha256"
            ],
        }
        expected_basis_hash = hashlib.sha256(
            json.dumps(
                basis_payload, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        if expected_basis_hash != basis_hash:
            raise ValueError("geometric H0 promotion basis hash mismatch")
    return manifest


def protonate_promoted_h0(
    promotion_manifest_path: Path,
    output_root: Path,
    *,
    parents_per_sam: int = 24,
    directions_per_parent: int = 4,
    groups_per_sam: int = 276,
) -> dict:
    """Globally assign surface protons from one hash-sealed promoted H0 parent."""

    promotion_manifest_path = Path(promotion_manifest_path).expanduser().resolve()
    try:
        promotion = json.loads(promotion_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("cannot read physical-interface promotion manifest") from exc
    structure_record = promotion.get("structure", {})
    input_h0 = promotion_manifest_path.parent / str(structure_record.get("path", ""))
    load_promoted_h0_parent(input_h0, promotion_manifest_path)

    is_geometric = promotion.get("schema") == (
        "samflow-geometric-h0-physical-interface-promotion-v1"
    )
    parent_record = promotion.get("parent", {})
    if is_geometric:
        geometry = promotion.get("geometry", {})
        requirements_record = promotion.get("protonation_requirements", {})
        requirements_path = promotion_manifest_path.parent / str(
            requirements_record.get("path", "")
        )
        builder = {
            "substrate_atom_count": geometry.get("substrate_atom_count"),
            "actual_sam_count": geometry.get("molecule_count"),
            "atoms_per_sam_h0": geometry.get("atoms_per_sam_h0"),
        }
    else:
        builder_manifest_path = Path(str(parent_record.get("manifest_path", ""))).resolve()
        if not builder_manifest_path.is_file() or _sha256(builder_manifest_path) != (
            parent_record.get("manifest_sha256")
        ):
            raise ValueError("sized H0 builder manifest hash mismatch")
        builder = json.loads(builder_manifest_path.read_text(encoding="utf-8"))
        requirements_record = builder.get("protonation_requirements", {})
        requirements_path = builder_manifest_path.parent / str(
            requirements_record.get("path", "")
        )
    if not requirements_path.is_file() or _sha256(requirements_path) != (
        requirements_record.get("sha256")
    ):
        raise ValueError("surface-proton requirements hash mismatch")
    requirements = json.loads(requirements_path.read_text(encoding="utf-8"))
    if (
        requirements.get("schema") != "samflow-surface-proton-requirements-v1"
        or requirements.get("assignment_policy")
        != "global_solver_after_physical_interface_promotion"
        or requirements.get("fixed_parent_assignments") != []
    ):
        raise ValueError("surface-proton requirements do not permit global assignment")

    contract = promotion.get("interface_validation_contract", {})
    normal = np.asarray(contract.get("surface_normal_cartesian_unit"), dtype=float)
    if normal.shape != (3,) or not np.isfinite(normal).all() or np.linalg.norm(normal) == 0:
        raise ValueError("physical-interface contract lacks a valid surface normal")
    normal /= np.linalg.norm(normal)
    periodic_axes = tuple(int(axis) for axis in contract.get("surface_periodic_axes", []))
    if len(periodic_axes) != 2 or len(set(periodic_axes)) != 2:
        raise ValueError("physical-interface contract lacks two surface-periodic axes")

    config = ProtonationConfig(
        substrate_atoms=int(builder["substrate_atom_count"]),
        molecule_count=int(builder["actual_sam_count"]),
        atoms_per_molecule=int(builder["atoms_per_sam_h0"]),
        protons_per_molecule=int(requirements["released_protons_per_sam"]),
        anchor_element=str(contract["anchor_element"]),
        anchor_atom_index_within_molecule_0based=int(
            contract["anchor_atom_index_within_sam_h0_0based"]
        ),
        parent_element="O",
        acceptor_element=str(contract.get("donor_element", "O")),
        headgroup_count=int(
            contract.get("headgroup_oxygen_count", contract.get("anchor_oxygen_count", 3))
        ),
        surface_normal=tuple(float(value) for value in normal),
        surface_periodic_axes=periodic_axes,
    )
    atoms = read(input_h0)
    symbols = np.asarray(atoms.get_chemical_symbols())
    expected_atoms = config.substrate_atoms + (
        config.molecule_count * config.atoms_per_molecule
    )
    if len(atoms) != expected_atoms:
        raise ValueError("promoted H0 atom layout does not match its builder manifest")
    substrate_symbols = symbols[: config.substrate_atoms]
    if np.any(substrate_symbols == "H"):
        raise ValueError("promoted H0 substrate still contains H")
    if requirements["required_surface_proton_count"] != (
        config.molecule_count * config.protons_per_molecule
    ):
        raise ValueError("surface-proton inventory does not match the promoted H0")

    surface_coordinates = atoms.positions[: config.substrate_atoms] @ normal
    minimum_h_coordinate = float(np.max(surface_coordinates)) + config.surface_h_clearance
    substrate_parent_atoms = np.flatnonzero(
        substrate_symbols == config.parent_element
    )
    all_acceptors = np.flatnonzero(symbols == config.acceptor_element)
    if len(substrate_parent_atoms) < requirements["required_surface_proton_count"]:
        raise ValueError("not enough substrate oxygen parents for proton inventory")

    domains: list[list[dict]] = []
    diagnostics = []
    for molecule_number in range(config.molecule_count):
        singles, diagnostic = single_candidates_for_sam(
            atoms,
            molecule_number,
            substrate_parent_atoms,
            all_acceptors,
            parents_per_sam,
            directions_per_parent,
            minimum_h_coordinate,
            config,
        )
        groups = proton_group_candidates_for_sam(
            singles, np.asarray(atoms.cell), groups_per_sam, config
        )
        if not groups:
            raise RuntimeError(
                f"SAM {molecule_number + 1} has no feasible global-proton domain"
            )
        diagnostic["proton_group_candidate_count"] = len(groups)
        diagnostics.append(diagnostic)
        domains.append(groups)
    chosen, solver = solve_global_groups(domains, np.asarray(atoms.cell), config)
    h_records = [proton for group in chosen for proton in group["protons"]]
    h_positions = np.asarray([item["position"] for item in h_records])
    surface_h_count = len(h_records)
    if surface_h_count != requirements["required_surface_proton_count"]:
        raise AssertionError("global solver did not satisfy surface-proton inventory")

    substrate = atoms[: config.substrate_atoms].copy()
    added_h = Atoms(
        ["H"] * surface_h_count,
        positions=h_positions,
        cell=atoms.cell,
        pbc=atoms.pbc,
    )
    sam = atoms[config.substrate_atoms :].copy()
    output = substrate + added_h + sam
    output.set_cell(atoms.cell)
    output.set_pbc(atoms.pbc)
    output.wrap()
    output.arrays.pop("type", None)

    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "promotion_manifest_sha256": _sha256(promotion_manifest_path),
                "requirements_sha256": _sha256(requirements_path),
                "solver": "global_proton_group_milp_v1",
                "parents_per_sam": parents_per_sam,
                "directions_per_parent": directions_per_parent,
                "groups_per_sam": groups_per_sam,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    output_root = Path(output_root).expanduser().resolve()
    package_directory = output_root / f"surface-protonation-{fingerprint[:12]}"
    if package_directory.exists():
        raise ValueError(f"Immutable surface protonation already exists: {package_directory}")
    package_directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        dir=package_directory.parent, prefix=f".{package_directory.name}-"
    ) as temporary_directory:
        temporary = Path(temporary_directory)
        structure_name = "protonated-interface.extxyz"
        write(temporary / structure_name, output)
        manifest = {
            "schema": "samflow-global-surface-protonation-v1",
            "schema_version": 1,
            "status": "passed_global_surface_protonation",
            "output_directory": str(package_directory),
            "parent_physical_interface_promotion": {
                "manifest_path": str(promotion_manifest_path),
                "manifest_sha256": _sha256(promotion_manifest_path),
                **(
                    {
                        "promotion_basis_sha256": promotion[
                            "promotion_basis_sha256"
                        ]
                    }
                    if is_geometric
                    else {
                        "approval_plan_sha256": promotion["parent"][
                            "approval_plan_sha256"
                        ]
                    }
                ),
                "physical_interface_pass": True,
            },
            "protonation_requirements": {
                "path": str(requirements_path),
                "sha256": _sha256(requirements_path),
            },
            "structure": {
                "path": structure_name,
                "sha256": _sha256(temporary / structure_name),
                "atom_count": len(output),
            },
            "input_substrate_atom_count": config.substrate_atoms,
            "protonated_substrate_atom_count": config.substrate_atoms
            + surface_h_count,
            "SAM_count": config.molecule_count,
            "atoms_per_SAM": config.atoms_per_molecule,
            "surface_H_count": surface_h_count,
            "surface_H_per_SAM": config.protons_per_molecule,
            "surface_frame": {
                "outward_normal_cartesian_unit": normal.tolist(),
                "periodic_fractional_axes": list(periodic_axes),
            },
            "solver": solver,
            "candidate_diagnostics": diagnostics,
            "selected_groups": [
                {
                    "SAM_number": group["molecule_number"] + 1,
                    "parent_atom_indices_0based": list(group["parent_atoms"]),
                    "protons": [
                        {
                            "parent_atom_index_0based": proton["parent_atom"],
                            "position_A": np.asarray(proton["position"]).tolist(),
                        }
                        for proton in group["protons"]
                    ],
                }
                for group in chosen
            ],
            "next_stage": "stage0_input_preparation",
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(package_directory)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Globally place surface protons after all SAMs are fixed."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output_prefix", type=Path)
    parser.add_argument(
        "--parent-promotion-manifest",
        type=Path,
        required=True,
        help="hash-sealed physical-interface promotion manifest for the exact H0 input",
    )
    parser.add_argument("--substrate-atoms", type=int, required=True)
    parser.add_argument("--molecules", type=int, required=True)
    parser.add_argument("--atoms-per-molecule", type=int, required=True)
    parser.add_argument("--protons-per-molecule", type=int, default=2)
    parser.add_argument("--anchor-element", default="P")
    parser.add_argument("--parent-element", default="O")
    parser.add_argument("--acceptor-element", default="O")
    parser.add_argument("--headgroup-count", type=int, default=3)
    parser.add_argument("--headgroup-bond-cutoff", type=float, default=1.85)
    parser.add_argument("--oh-bond", type=float, default=0.98)
    parser.add_argument("--hh-min", type=float, default=1.20)
    parser.add_argument("--h-heavy-min", type=float, default=1.40)
    parser.add_argument("--hbond-max", type=float, default=2.40)
    parser.add_argument("--hbond-angle-min", type=float, default=105.0)
    parser.add_argument("--parent-xy-max", type=float, default=5.80)
    parser.add_argument("--parent-dz-min", type=float, default=0.25)
    parser.add_argument("--parent-dz-max", type=float, default=4.50)
    parser.add_argument("--acceptor-parent-min", type=float, default=2.35)
    parser.add_argument("--acceptor-parent-max", type=float, default=3.65)
    parser.add_argument(
        "--surface-h-clearance",
        type=float,
        default=0.0,
        help=(
            "Minimum H height above the global top ITO atom plane in A; "
            "must be nonnegative"
        ),
    )
    parser.add_argument(
        "--specorder", default="H,C,N,O,P,S,In,Sn", help="Comma-separated LAMMPS species order"
    )
    parser.add_argument("--parents-per-sam", type=int, default=24)
    parser.add_argument("--directions-per-parent", type=int, default=4)
    parser.add_argument("--groups-per-sam", type=int, default=276)
    args = parser.parse_args()
    promotion_parent = load_promoted_h0_parent(
        args.input, args.parent_promotion_manifest
    )

    config = ProtonationConfig(
        substrate_atoms=args.substrate_atoms,
        molecule_count=args.molecules,
        atoms_per_molecule=args.atoms_per_molecule,
        protons_per_molecule=args.protons_per_molecule,
        anchor_element=args.anchor_element,
        parent_element=args.parent_element,
        acceptor_element=args.acceptor_element,
        headgroup_count=args.headgroup_count,
        headgroup_bond_cutoff=args.headgroup_bond_cutoff,
        oh_bond=args.oh_bond,
        hh_min=args.hh_min,
        h_heavy_min=args.h_heavy_min,
        hbond_max=args.hbond_max,
        hbond_angle_min=args.hbond_angle_min,
        parent_xy_max=args.parent_xy_max,
        parent_dz_min=args.parent_dz_min,
        parent_dz_max=args.parent_dz_max,
        acceptor_parent_min=args.acceptor_parent_min,
        acceptor_parent_max=args.acceptor_parent_max,
        surface_h_clearance=args.surface_h_clearance,
    )
    if config.protons_per_molecule < 1:
        raise ValueError("--protons-per-molecule must be positive")
    if config.surface_h_clearance < 0.0:
        raise ValueError("--surface-h-clearance must be nonnegative")

    atoms = read(args.input)
    symbols = np.asarray(atoms.get_chemical_symbols())
    input_counts = dict(sorted(Counter(symbols).items()))
    expected_input_atoms = (
        config.substrate_atoms
        + config.molecule_count * config.atoms_per_molecule
    )
    if len(atoms) != expected_input_atoms:
        raise ValueError(
            f"Input has {len(atoms)} atoms; expected {expected_input_atoms} from "
            "--substrate-atoms + --molecules * --atoms-per-molecule"
        )
    substrate_symbols = symbols[: config.substrate_atoms]
    if np.any(substrate_symbols == "H"):
        raise ValueError("H0 substrate still contains H")
    surface_plane_z = float(
        np.max(atoms.positions[: config.substrate_atoms, 2])
    )
    minimum_h_z = surface_plane_z + config.surface_h_clearance

    substrate_parent_atoms = np.flatnonzero(
        substrate_symbols == config.parent_element
    )
    all_acceptors = np.flatnonzero(symbols == config.acceptor_element)
    if len(substrate_parent_atoms) < config.molecule_count * config.protons_per_molecule:
        raise ValueError("Not enough substrate parent atoms for the requested protons")
    domains: list[list[dict]] = []
    diagnostics = []
    for molecule_number in range(config.molecule_count):
        singles, diagnostic = single_candidates_for_sam(
            atoms,
            molecule_number,
            substrate_parent_atoms,
            all_acceptors,
            args.parents_per_sam,
            args.directions_per_parent,
            minimum_h_z,
            config,
        )
        groups = proton_group_candidates_for_sam(
            singles, np.asarray(atoms.cell), args.groups_per_sam, config
        )
        diagnostic["proton_group_candidate_count"] = len(groups)
        diagnostics.append(diagnostic)
        domains.append(groups)
        print(
            f"SAM {molecule_number + 1:02d}: {len(singles)} single-H, "
            f"{len(groups)} {config.protons_per_molecule}-H groups",
            flush=True,
        )
        if not groups:
            raise RuntimeError(
                f"SAM {molecule_number + 1} has no feasible proton group"
            )

    chosen, solver = solve_global_groups(domains, np.asarray(atoms.cell), config)
    h_records = [proton for group in chosen for proton in group["protons"]]
    h_positions = np.asarray([item["position"] for item in h_records])

    surface_h_count = config.molecule_count * config.protons_per_molecule
    substrate = atoms[: config.substrate_atoms].copy()
    added_h = Atoms(
        ["H"] * surface_h_count,
        positions=h_positions,
        cell=atoms.cell,
        pbc=atoms.pbc,
    )
    sam = atoms[config.substrate_atoms :].copy()
    output = substrate + added_h + sam
    output.set_cell(atoms.cell)
    output.set_pbc(atoms.pbc)
    output.wrap()
    output.arrays.pop("type", None)
    output_counts = dict(sorted(Counter(output.get_chemical_symbols()).items()))
    expected_output_counts = dict(input_counts)
    expected_output_counts["H"] = expected_output_counts.get("H", 0) + surface_h_count
    expected_output_atoms = len(atoms) + surface_h_count
    if len(output) != expected_output_atoms or output_counts != expected_output_counts:
        raise AssertionError(
            f"Output structure invalid: atoms={len(output)}, counts={output_counts}; "
            f"expected atoms={expected_output_atoms}, counts={expected_output_counts}"
        )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    cif_path = Path(f"{args.output_prefix}.cif")
    data_path = Path(f"{args.output_prefix}.data")
    manifest_path = Path(f"{args.output_prefix}_manifest.json")
    write(cif_path, output)
    cif_lines = cif_path.read_text().splitlines(keepends=True)
    cif_path.write_text(
        "".join(
            line
            for line in cif_lines
            if not line.startswith("_chemical_formula_structural")
        )
    )
    write(
        data_path,
        output,
        format="lammps-data",
        atom_style="atomic",
        specorder=[item.strip() for item in args.specorder.split(",") if item.strip()],
        masses=True,
    )
    clean_lammps_masses_comments(data_path)

    manifest = {
        "algorithm": "post-SAM global proton-group multi-acceptor optimization",
        "parent_physical_interface_promotion": {
            "manifest_path": str(args.parent_promotion_manifest.resolve()),
            "manifest_sha256": _sha256(args.parent_promotion_manifest.resolve()),
            "approval_plan_sha256": promotion_parent["parent"][
                "approval_plan_sha256"
            ],
            "physical_interface_pass": True,
        },
        "input_H0": str(args.input),
        "output_cif": str(cif_path),
        "output_cif_sha256": _sha256(cif_path),
        "output_data": str(data_path),
        "output_data_sha256": _sha256(data_path),
        "atom_count": len(output),
        "element_counts": output_counts,
        "input_substrate_atom_count": config.substrate_atoms,
        "protonated_substrate_atom_count": config.substrate_atoms + surface_h_count,
        "SAM_count": config.molecule_count,
        "atoms_per_SAM": config.atoms_per_molecule,
        "surface_H_count": len(h_records),
        "surface_H_per_SAM": config.protons_per_molecule,
        "anchor_element": config.anchor_element,
        "parent_element": config.parent_element,
        "acceptor_element": config.acceptor_element,
        "parent_H_bond_length_A": config.oh_bond,
        "surface_plane_z_A": surface_plane_z,
        "surface_H_minimum_z_A": minimum_h_z,
        "surface_H_clearance_A": config.surface_h_clearance,
        "minimum_selected_H_height_above_surface_A": float(
            np.min([item["height_above_surface_plane_A"] for item in h_records])
        ),
        "secondary_acceptor_window_A": [config.h_heavy_min, config.hbond_max],
        "solver": solver,
        "candidate_diagnostics": diagnostics,
        "selected_groups": [
            {
                "SAM_number": group["molecule_number"] + 1,
                "intramolecular_min_HH_A": group["intramolecular_min_HH_A"],
                "distinct_acceptor_count": group["distinct_acceptor_count"],
                "protons": [
                    {
                        "parent_atom_index_0based": proton["parent_atom"],
                        "anchor_index_0based": proton["anchor_index"],
                        "anchor_parent_xy_A": proton["anchor_parent_xy_A"],
                        "anchor_minus_parent_z_A": proton[
                            "anchor_minus_parent_z_A"
                        ],
                        "coordination_count": proton["coordination_count"],
                        "interaction_score": proton["interaction_score"],
                        "minimum_clearance_ratio": proton[
                            "minimum_clearance_ratio"
                        ],
                        "height_above_surface_plane_A": proton[
                            "height_above_surface_plane_A"
                        ],
                        "position_A": np.asarray(proton["position"]).tolist(),
                        "secondary_acceptors": proton["interactions"],
                    }
                    for proton in group["protons"]
                ],
            }
            for group in chosen
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    summary = {
        key: value
        for key, value in manifest.items()
        if key not in {"selected_groups", "candidate_diagnostics"}
    }
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def protonate_restrained_h0(input_h0, output_root, *, substrate_atoms,
                            molecules, atoms_per_molecule, sealed_spec,
                            parents_per_sam=24, directions_per_parent=4,
                            groups_per_sam=276):
    """PVKSAM: global H assignment with frozen identity and no global height gate.

    全局补氢；保留几何筛选与唯一表面O分配，不伪造旧流程的界面验收。
    The caller seals spec from intact H0 before optimization. New O-H pairs are
    placement geometry only, never constraints on subsequent relaxation or MD.
    """
    from ase.geometry import find_mic
    from relax_monolayer import read_pvksam_structure, save, sha
    import time
    output_root=Path(output_root)
    output_root.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    def json_safe(value):
        if isinstance(value,np.ndarray):return json_safe(value.tolist())
        if isinstance(value,np.generic):return json_safe(value.item())
        if isinstance(value,float) and not np.isfinite(value):return None
        if isinstance(value,dict):return {str(k):json_safe(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return [json_safe(v) for v in value]
        return value
    try:
        atoms=read_pvksam_structure(input_h0);atoms.set_pbc([True,True,False])
        n=substrate_atoms;nh=molecules*sealed_spec['protons_per_molecule']
        if len(atoms)!=n+molecules*atoms_per_molecule or np.any(atoms.numbers[:n]==1):
            raise ValueError('Relaxed H0 layout mismatch')
        if not np.allclose(atoms.cell[:],np.diag(atoms.cell.diagonal()),atol=1e-10):
            raise ValueError('PVKSAM protonation requires orthogonal +z surface frame')
        head=sealed_spec['headgroups'][0]
        config=ProtonationConfig(n,molecules,atoms_per_molecule,sealed_spec['protons_per_molecule'],
            anchor_element='P' if head['family']=='phosphonate' else 'C',
            anchor_atom_index_within_molecule_0based=head['local_center'],
            headgroup_count=len(head['local_oxygen']),
            frozen_headgroup_local_indices=tuple(head['local_oxygen']))
        sy=np.asarray(atoms.get_chemical_symbols())
        parents=np.flatnonzero(sy[:n]=='O');acceptors=np.flatnonzero(sy=='O')
        if len(parents)<nh:raise ValueError('Insufficient surface-O parents for released protons')
        minimum_h=float(atoms.positions[parents,2].min())-config.oh_bond-.1
        domains=[];diagnostics=[]
        for m in range(molecules):
            singles,d=single_candidates_for_sam(atoms,m,parents,acceptors,
                parents_per_sam,directions_per_parent,minimum_h,config)
            groups=proton_group_candidates_for_sam(singles,atoms.cell[:],groups_per_sam,config)
            d['proton_group_candidate_count']=len(groups)
            diagnostics.append(d);domains.append(groups)
            save(output_root/'candidate-diagnostics.json',json_safe(diagnostics))
            save(output_root/'progress.json',{'status':'candidate_generation','molecule':m+1,
                'groups':len(groups),'elapsed_s':time.monotonic()-start})
            if not groups:raise ValueError(f'No feasible proton group for SAM {m+1}')
        chosen,solver=solve_global_groups(domains,atoms.cell[:],config)
        hs=[h for g in chosen for h in g['protons']]
        if len(hs)!=nh or len({h['parent_atom'] for h in hs})!=nh:
            raise ValueError('Global proton count/unique-parent invariant failed')
        hpos=np.asarray([h['position'] for h in hs]);oh=[];hh=[];clearance=[]
        for k,h in enumerate(hs):
            _,d=find_mic(atoms.positions-hpos[k],atoms.cell,[True,True,False])
            mask=np.arange(len(atoms))!=h['parent_atom']
            threshold=np.where(atoms.numbers==1,config.hh_min,config.h_heavy_min)
            clearance.append(float(np.min(d[mask]/threshold[mask])))
            oh.append(float(d[h['parent_atom']]))
            if k+1<nh:
                _,dh=find_mic(hpos[k+1:]-hpos[k],atoms.cell,[True,True,False]);hh.append(float(dh.min()))
        if (min(clearance)<1-1e-10 or (hh and min(hh)<config.hh_min-1e-10)
                or not np.allclose(oh,config.oh_bond,atol=1e-9)):
            raise ValueError('Independent added-H geometry check failed')
        added=Atoms(['H']*nh,positions=hpos,cell=atoms.cell,pbc=atoms.pbc)
        output=atoms[:n]+added+atoms[n:];output.set_constraint();output.arrays.pop('type',None)
        if not (np.array_equal(output.positions[:n],atoms.positions[:n]) and
                np.array_equal(output.positions[n+nh:],atoms.positions[n:])):
            raise ValueError('Protonation changed existing coordinates')
        for suffix,kw in [('extxyz',{}),('cif',{}),('data',{'format':'lammps-data',
                'atom_style':'atomic','masses':True,'specorder':sorted(set(output.get_chemical_symbols()))})]:
            write(output_root/f'protonated-interface.{suffix}',output,**kw)
        reread=read(output_root/'protonated-interface.extxyz')
        if not np.array_equal(reread.numbers,output.numbers) or not np.allclose(reread.positions,output.positions,atol=1e-7,rtol=0):
            raise ValueError('Protonation readback mismatch')
        save(output_root/'selected-groups.json',json_safe(chosen))
        result={'schema':'pvksam-global-protonation-v1','status':'completed','physical_interface_pass':None,
            'input_sha256':sha(input_h0),'surface_H_count':nh,'surface_H_per_SAM':config.protons_per_molecule,
            'H_height_policy':'none','surface_OH_restraints':0,'solver':json_safe(solver),
            'added_H_audit':{'OH_min_A':min(oh),'OH_max_A':max(oh),'minimum_added_H_H_A':min(hh) if hh else None,
                'minimum_existing_atom_clearance_ratio':min(clearance),'unique_O_parents':nh,
                'source_coordinates_unchanged':True},
            'artifacts':{p.name:sha(p) for p in output_root.glob('protonated-interface.*')}}
        save(output_root/'manifest.json',result);save(output_root/'progress.json',result)
        return result
    except BaseException as exc:
        save(output_root/'failure.json',{'type':type(exc).__name__,'message':str(exc)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
