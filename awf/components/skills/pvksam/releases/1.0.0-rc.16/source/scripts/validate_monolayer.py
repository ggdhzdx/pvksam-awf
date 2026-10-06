#!/usr/bin/env python3
"""Validate a contiguous-block SAM/substrate structure before relaxation or MD.

The builder must keep the substrate first and append each SAM as one contiguous
block.  All counts, species expectations, proton stoichiometry, anchor element,
and geometric thresholds are explicit parameters or derived from the structure.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase.geometry import find_mic
from ase.io import read
from ase.neighborlist import neighbor_list
from ase.data import atomic_numbers, covalent_radii

from sam_structure_tools import (
    closed_distance_window_tolerance_A,
    distance_within_closed_window,
)


@dataclass(frozen=True)
class ValidationConfig:
    substrate_atoms: int
    molecules: int
    atoms_per_molecule: int
    protons_per_molecule: int = 2
    anchor_element: str = "P"
    anchor_atom_index_within_molecule_0based: int | None = None
    parent_element: str = "O"
    acceptor_element: str = "O"
    marker_element: str | None = None
    marker_carbon_neighbors: int | None = None
    forbid_marker_h: bool = False
    parent_h_bond_max: float = 1.25
    secondary_min: float = 1.40
    secondary_max: float = 2.40
    secondary_angle_min: float = 105.0
    hh_min: float = 1.20
    h_heavy_min: float = 1.40
    heavy_heavy_min: float = 1.80
    expected_formula: dict[str, int] | None = None
    expected_substrate_formula: dict[str, int] | None = None
    expected_molecule_formula: dict[str, int] | None = None
    surface_h_policy: str = "network"
    interface_pair_policy: str = "exact"
    mobile_interface_min: float = 1.70
    mobile_interface_max: float = 2.80
    mobile_interface_metal_elements: tuple[str, ...] = ("In", "Sn")
    mobile_interface_donor_element: str = "O"
    surface_metal_depth_A: float = 0.7
    interface_height_clearance_A: float = 1.0
    surface_normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
    triangle_anchor_molecule_numbers: tuple[int, ...] | None = None
    allowed_interface_pairs: tuple[tuple[str, str], ...] = ()
    registered_interface_bonds: tuple[dict, ...] = ()
    surface_periodic_axes: tuple[int, int] = (0, 1)


def parse_element_pair(value: str) -> tuple[str, str]:
    fields = tuple(
        part.strip()
        for part in value.replace(":", "-").split("-")
        if part.strip()
    )
    if len(fields) != 2 or any(field not in atomic_numbers for field in fields):
        raise argparse.ArgumentTypeError(
            f"Expected an element pair such as O-P, got {value!r}"
        )
    return fields


def load_registered_interface_bonds(path: Path) -> tuple[dict, ...]:
    """Load exact global-index interface records from a strict JSON document."""

    path = Path(path)
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load registered interface bonds from {path}") from exc
    if isinstance(payload, dict):
        records = payload.get("registered_interface_bonds")
    else:
        records = payload
    if not isinstance(records, list):
        raise ValueError(
            "Registered interface bond JSON must be a list or contain "
            "registered_interface_bonds"
        )
    if any(not isinstance(record, dict) for record in records):
        raise ValueError("Every registered interface bond must be a JSON object")
    return tuple(dict(record) for record in records)


def _validated_expected_formula(value, *, name: str) -> dict[str, int] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an element-count object")
    formula = {}
    for element, count in value.items():
        element = str(element)
        if element not in atomic_numbers:
            raise ValueError(f"{name} contains unknown element {element!r}")
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)):
            raise ValueError(f"{name} counts must be non-negative integers")
        count = int(count)
        if count < 0:
            raise ValueError(f"{name} counts must be non-negative integers")
        if count:
            formula[element] = count
    return dict(sorted(formula.items()))


def _normalize_registered_interface_bonds(
    records, *, symbols, atom_count: int, substrate_atoms: int, atoms, periodic_axes
) -> tuple[list[dict], set[tuple[int, int]], set[tuple[int, int]]]:
    """Validate and independently measure exact substrate/SAM bond records."""

    if records is None:
        records = ()
    if isinstance(records, (str, bytes, dict)):
        raise ValueError("registered_interface_bonds must be a sequence of records")
    try:
        raw_records = list(records)
    except TypeError as exc:
        raise ValueError(
            "registered_interface_bonds must be a sequence of records"
        ) from exc
    normalized = []
    all_pairs = set()
    passed_pairs = set()
    mapped_sam_atoms = set()
    for record_number, record in enumerate(raw_records, 1):
        if not isinstance(record, dict):
            raise ValueError(
                f"Registered interface bond {record_number} must be an object"
            )
        substrate_index = record.get("substrate_atom_index_0based")
        sam_index = record.get("sam_atom_index_0based")
        for field_name, value in (
            ("substrate atom index", substrate_index),
            ("SAM atom index", sam_index),
        ):
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise ValueError(
                    f"Registered interface bond {record_number} {field_name} "
                    "must be a global 0-based integer"
                )
        substrate_index = int(substrate_index)
        sam_index = int(sam_index)
        if not 0 <= substrate_index < substrate_atoms:
            raise ValueError(
                f"Registered interface bond {record_number} substrate atom index "
                "must identify the substrate"
            )
        if not substrate_atoms <= sam_index < atom_count:
            if 0 <= sam_index < substrate_atoms:
                raise ValueError(
                    f"Registered interface bond {record_number} SAM atom index "
                    "must identify the SAM"
                )
            raise ValueError(
                f"Registered interface bond {record_number} SAM atom index is outside "
                "the structure"
            )
        pair = (substrate_index, sam_index)
        if pair in all_pairs:
            raise ValueError("Registered interface bond pairs must be unique; duplicate pair")
        if sam_index in mapped_sam_atoms:
            raise ValueError("Each registered SAM donor atom must be unique")
        all_pairs.add(pair)
        mapped_sam_atoms.add(sam_index)

        try:
            minimum = float(record["minimum_distance_A"])
            maximum = float(record["maximum_distance_A"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Registered interface bond {record_number} requires minimum_distance_A "
                "and maximum_distance_A"
            ) from exc
        if (
            not np.isfinite([minimum, maximum]).all()
            or minimum <= 0.0
            or maximum < minimum
        ):
            raise ValueError(
                f"Registered interface bond {record_number} has an invalid distance window"
            )
        expected_substrate = record.get("expected_substrate_element")
        expected_sam = record.get("expected_sam_element")
        if expected_substrate is not None:
            if (
                not isinstance(expected_substrate, str)
                or expected_substrate not in atomic_numbers
            ):
                raise ValueError("Registered bond expected substrate element is invalid")
            if str(symbols[substrate_index]) != expected_substrate:
                raise ValueError(
                    "Registered bond expected substrate element does not match the structure"
                )
        if expected_sam is not None:
            if not isinstance(expected_sam, str) or expected_sam not in atomic_numbers:
                raise ValueError("Registered bond expected SAM element is invalid")
            if str(symbols[sam_index]) != expected_sam:
                raise ValueError(
                    "Registered bond expected SAM element does not match the structure"
                )
        distance = float(
            _surface_distances(
                atoms, substrate_index, [sam_index], periodic_axes
            )[0]
        )
        comparison_tolerance_A = closed_distance_window_tolerance_A(
            distance,
            minimum,
            maximum,
        )
        window_passed = distance_within_closed_window(distance, minimum, maximum)
        if window_passed:
            passed_pairs.add(pair)
        normalized_record = {
            "substrate_atom_index_0based": substrate_index,
            "sam_atom_index_0based": sam_index,
            "substrate_element": str(symbols[substrate_index]),
            "sam_element": str(symbols[sam_index]),
            "minimum_distance_A": minimum,
            "maximum_distance_A": maximum,
            "distance_A": distance,
            "comparison_tolerance_A": comparison_tolerance_A,
            "window_passed": window_passed,
        }
        if expected_substrate is not None:
            normalized_record["expected_substrate_element"] = expected_substrate
        if expected_sam is not None:
            normalized_record["expected_sam_element"] = expected_sam
        normalized.append(normalized_record)
    return normalized, all_pairs, passed_pairs


def is_allowed_interface_pair(
    symbol_i: str,
    symbol_j: str,
    label_i: int,
    label_j: int,
    allowed_pairs: set[frozenset[str]],
) -> bool:
    is_substrate_sam_interface = (label_i == 0) != (label_j == 0)
    return (
        is_substrate_sam_interface
        and frozenset((symbol_i, symbol_j)) in allowed_pairs
    )


def classify_surface_proton(
    substrate_o_distance: float,
    sam_o_distance: float,
    bond_max: float,
) -> str:
    substrate_bound = substrate_o_distance < bond_max
    sam_bound = sam_o_distance < bond_max
    if substrate_bound and sam_bound:
        return "shared"
    if substrate_bound:
        return "substrate"
    if sam_bound:
        return "sam"
    return "unbound"


def _validated_mic_periodic_axes(periodic_axes) -> tuple[int, ...]:
    try:
        axes = tuple(int(axis) for axis in periodic_axes)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "periodic_axes must contain distinct cell axes chosen from 0, 1, 2"
        ) from exc
    if (
        not axes
        or len(axes) > 3
        or len(set(axes)) != len(axes)
        or any(axis not in (0, 1, 2) for axis in axes)
    ):
        raise ValueError(
            "periodic_axes must contain distinct axes chosen from 0, 1, 2"
        )
    return axes


def _validated_surface_periodic_axes(periodic_axes) -> tuple[int, int]:
    try:
        axes = _validated_mic_periodic_axes(periodic_axes)
    except ValueError as exc:
        raise ValueError(
            "surface_periodic_axes must contain two distinct axes chosen from 0, 1, 2"
        ) from exc
    if len(axes) != 2:
        raise ValueError(
            "surface_periodic_axes must contain two distinct axes chosen from 0, 1, 2"
        )
    return axes


def mic_vectors(
    origin: np.ndarray,
    targets: np.ndarray,
    cell: np.ndarray,
    periodic_axes=(0, 1, 2),
) -> np.ndarray:
    """Return true nearest-image vectors along the declared periodic cell axes.

    The three-argument call retains the historical full three-dimensional MIC.
    Validator callers pass their two Surface Frame lattice axes explicitly so
    the slab-normal direction remains nonperiodic.
    """

    axes = _validated_mic_periodic_axes(periodic_axes)
    pbc = np.zeros(3, dtype=bool)
    pbc[list(axes)] = True
    vectors = np.asarray(targets, dtype=float) - np.asarray(origin, dtype=float)
    minimum_vectors, _ = find_mic(
        vectors,
        cell=np.asarray(cell, dtype=float),
        pbc=pbc,
    )
    return np.asarray(minimum_vectors, dtype=float)


def _surface_distances(atoms, origin_index: int, target_indices, periodic_axes):
    targets = np.asarray(target_indices, dtype=int)
    vectors = mic_vectors(
        np.asarray(atoms.positions[int(origin_index)], dtype=float),
        np.asarray(atoms.positions[targets], dtype=float),
        np.asarray(atoms.cell, dtype=float),
        periodic_axes,
    )
    return np.linalg.norm(vectors, axis=1)


def _point_in_triangle_2d(point: np.ndarray, triangle: np.ndarray) -> bool:
    """Test an inclusive 2-D triangle using barycentric coordinates."""

    first, second, third = np.asarray(triangle, dtype=float)
    vector_a = second - first
    vector_b = third - first
    relative = np.asarray(point, dtype=float) - first
    determinant = vector_a[0] * vector_b[1] - vector_b[0] * vector_a[1]
    if abs(float(determinant)) <= 1.0e-12:
        return False
    weight_a = (relative[0] * vector_b[1] - vector_b[0] * relative[1]) / determinant
    weight_b = (vector_a[0] * relative[1] - relative[0] * vector_a[1]) / determinant
    tolerance = 1.0e-10
    return (
        weight_a >= -tolerance
        and weight_b >= -tolerance
        and weight_a + weight_b <= 1.0 + tolerance
    )


def _phosphonate_triangle_height_gate(
    atoms,
    *,
    config: ValidationConfig,
    symbols: np.ndarray,
    substrate_symbols: np.ndarray,
    triangle_anchor_molecule_numbers: tuple[int, ...],
) -> dict:
    """Check one accessible metal-in-anchor-triangle and non-anchor height per SAM."""

    normal = np.asarray(config.surface_normal, dtype=float)
    if normal.shape != (3,) or not np.isfinite(normal).all():
        raise ValueError("surface_normal must be a finite 3-vector")
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 0.0:
        raise ValueError("surface_normal must be nonzero")
    normal /= normal_norm
    if (
        not np.isfinite(config.surface_metal_depth_A)
        or config.surface_metal_depth_A < 0.0
    ):
        raise ValueError("surface_metal_depth_A must be finite and nonnegative")
    if (
        not np.isfinite(config.interface_height_clearance_A)
        or config.interface_height_clearance_A < 0.0
    ):
        raise ValueError("interface_height_clearance_A must be finite and nonnegative")

    cell = np.asarray(atoms.cell, dtype=float)
    periodic_axes = tuple(config.surface_periodic_axes)
    in_plane_axis = cell[periodic_axes[0]]
    in_plane_axis = in_plane_axis - np.dot(in_plane_axis, normal) * normal
    axis_norm = float(np.linalg.norm(in_plane_axis))
    if axis_norm <= 1.0e-12:
        raise ValueError("first periodic cell vector is parallel to surface_normal")
    basis_u = in_plane_axis / axis_norm
    basis_v = np.cross(normal, basis_u)
    basis_v /= np.linalg.norm(basis_v)

    substrate_count = config.substrate_atoms
    substrate_positions = np.asarray(atoms.positions[:substrate_count], dtype=float)
    substrate_top = float(np.max(substrate_positions @ normal))
    substrate_metals = np.flatnonzero(
        np.isin(substrate_symbols, config.mobile_interface_metal_elements)
    )
    if len(substrate_metals):
        metal_surface_coordinates = substrate_positions[substrate_metals] @ normal
        exposed_metal_coordinate = float(np.max(metal_surface_coordinates))
        accessible_metals = substrate_metals[
            exposed_metal_coordinate - metal_surface_coordinates
            <= config.surface_metal_depth_A + 1.0e-10
        ]
    else:
        exposed_metal_coordinate = None
        accessible_metals = np.asarray([], dtype=int)

    molecule_records = []
    for molecule_number in range(config.molecules):
        start = substrate_count + molecule_number * config.atoms_per_molecule
        stop = start + config.atoms_per_molecule
        block = np.arange(start, stop, dtype=int)
        block_symbols = symbols[block]
        anchors = (
            np.asarray([block[config.anchor_atom_index_within_molecule_0based]])
            if config.anchor_atom_index_within_molecule_0based is not None
            else block[block_symbols == config.anchor_element]
        )
        anchor_oxygen = np.asarray([], dtype=int)
        if len(anchors) == 1 and symbols[int(anchors[0])] == config.anchor_element:
            oxygen_indices = block[block_symbols == "O"]
            cutoff = covalent_cutoff(config.anchor_element, "O")
            if cutoff is not None:
                distances = _surface_distances(
                    atoms, int(anchors[0]), oxygen_indices, periodic_axes
                )
                anchor_oxygen = oxygen_indices[distances < cutoff]
        anchor_topology_pass = len(anchors) == 1 and len(anchor_oxygen) == 3
        metal_hit = None
        if anchor_topology_pass and len(accessible_metals):
            reference_oxygen = int(anchor_oxygen[0])
            oxygen_vectors = mic_vectors(
                atoms.positions[reference_oxygen],
                atoms.positions[anchor_oxygen],
                cell,
                periodic_axes,
            )
            triangle = np.column_stack(
                (oxygen_vectors @ basis_u, oxygen_vectors @ basis_v)
            )
            metal_vectors = mic_vectors(
                atoms.positions[reference_oxygen],
                atoms.positions[accessible_metals],
                cell,
                periodic_axes,
            )
            for local_index, vector in enumerate(metal_vectors):
                point = np.asarray([np.dot(vector, basis_u), np.dot(vector, basis_v)])
                if _point_in_triangle_2d(point, triangle):
                    metal_index = int(accessible_metals[local_index])
                    metal_hit = {
                        "substrate_atom_index_0based": metal_index,
                        "element": str(symbols[metal_index]),
                    }
                    break
        anchor_group = set(int(index) for index in anchors)
        anchor_group.update(int(index) for index in anchor_oxygen)
        non_anchor = np.asarray(
            [int(index) for index in block if int(index) not in anchor_group],
            dtype=int,
        )
        non_anchor_clearances = (
            np.asarray(atoms.positions[non_anchor], dtype=float) @ normal - substrate_top
        )
        minimum_height = (
            float(np.min(non_anchor_clearances))
            if len(non_anchor_clearances)
            else None
        )
        height_pass = bool(
            len(non_anchor_clearances)
            and np.all(
                non_anchor_clearances + 1.0e-10
                >= config.interface_height_clearance_A
            )
        )
        molecule_records.append(
            {
                "molecule": molecule_number + 1,
                "anchor_atom_index_0based": (
                    int(anchors[0]) if len(anchors) == 1 else None
                ),
                "anchor_oxygen_indices_0based": [int(index) for index in anchor_oxygen],
                "anchor_topology_pass": anchor_topology_pass,
                "metal_inside_anchor_o_triangle": metal_hit is not None,
                "qualifying_surface_metal": metal_hit,
                "non_anchor_atom_count": int(len(non_anchor)),
                "minimum_non_anchor_height_A": minimum_height,
                "non_anchor_height_pass": height_pass,
            }
        )
    return {
        "surface_normal_cartesian_unit": normal.tolist(),
        "substrate_top_coordinate_A": substrate_top,
        "accessible_surface_metal_count": int(len(accessible_metals)),
        "maximum_surface_metal_coordinate_A": exposed_metal_coordinate,
        "surface_metal_depth_A": float(config.surface_metal_depth_A),
        "minimum_non_anchor_height_A": float(config.interface_height_clearance_A),
        "molecules": molecule_records,
        "triangle_anchor_molecule_numbers": list(triangle_anchor_molecule_numbers),
        "anchor_triangles_passed": bool(triangle_anchor_molecule_numbers)
        and all(
            record["anchor_topology_pass"] and record["metal_inside_anchor_o_triangle"]
            for record in molecule_records
            if record["molecule"] in triangle_anchor_molecule_numbers
        ),
        "non_anchor_heights_passed": bool(molecule_records)
        and all(record["non_anchor_height_pass"] for record in molecule_records),
    }


def covalent_cutoff(symbol_a: str, symbol_b: str) -> float | None:
    pair = frozenset((symbol_a, symbol_b))
    cutoffs = {
        frozenset(("C", "H")): 1.25,
        frozenset(("N", "H")): 1.25,
        frozenset(("O", "H")): 1.25,
        frozenset(("C", "C")): 1.75,
        frozenset(("C", "N")): 1.70,
        frozenset(("C", "S")): 2.05,
        frozenset(("C", "P")): 2.15,
        frozenset(("P", "O")): 1.85,
    }
    if pair in cutoffs:
        return cutoffs[pair]
    try:
        radius_sum = (
            covalent_radii[atomic_numbers[symbol_a]]
            + covalent_radii[atomic_numbers[symbol_b]]
        )
    except KeyError:
        return None
    return min(2.20, 1.20 * float(radius_sum))


def molecule_component_size(
    atoms,
    indices: np.ndarray,
    anchor_element: str,
    periodic_axes=(0, 1, 2),
    anchor_index: int | None = None,
) -> int:
    symbols = np.asarray(atoms.get_chemical_symbols())
    positions = atoms.positions
    cell = np.asarray(atoms.cell)
    axes = _validated_mic_periodic_axes(periodic_axes)
    local = set(int(i) for i in indices)
    adjacency = {int(i): set() for i in indices}

    for offset, i in enumerate(indices):
        for j in indices[offset + 1 :]:
            cutoff = covalent_cutoff(symbols[i], symbols[j])
            if cutoff is None:
                continue
            delta = mic_vectors(
                positions[i], positions[[j]], cell, axes
            )[0]
            distance = np.linalg.norm(delta)
            if distance < cutoff:
                adjacency[int(i)].add(int(j))
                adjacency[int(j)].add(int(i))

    anchor_indices = (
        [int(anchor_index)]
        if anchor_index is not None
        and int(anchor_index) in local
        and symbols[int(anchor_index)] == anchor_element
        else [int(i) for i in indices if symbols[i] == anchor_element]
    )
    if len(anchor_indices) != 1:
        return 0
    visited = {anchor_indices[0]}
    stack = [anchor_indices[0]]
    while stack:
        current = stack.pop()
        for adjacent in adjacency[current]:
            if adjacent in local and adjacent not in visited:
                visited.add(adjacent)
                stack.append(adjacent)
    return len(visited)


def validate(path: Path, config: ValidationConfig) -> dict:
    periodic_axes = _validated_surface_periodic_axes(config.surface_periodic_axes)
    expected_formula = _validated_expected_formula(
        config.expected_formula, name="expected_formula"
    )
    expected_substrate_formula = _validated_expected_formula(
        config.expected_substrate_formula, name="expected_substrate_formula"
    )
    expected_molecule_formula = _validated_expected_formula(
        config.expected_molecule_formula, name="expected_molecule_formula"
    )
    collision_thresholds_A = {
        "H-H": float(config.hh_min),
        "H-heavy": float(config.h_heavy_min),
        "heavy-heavy": float(config.heavy_heavy_min),
    }
    if any(
        not np.isfinite(value) or value <= 0.0
        for value in collision_thresholds_A.values()
    ):
        raise ValueError("collision thresholds must be finite positive numbers")
    collision_neighbor_cutoff_A = max(collision_thresholds_A.values())
    if config.surface_h_policy not in {"h0", "network", "retained", "mobile"}:
        raise ValueError(
            f"Unsupported surface-H policy: {config.surface_h_policy}"
        )
    if config.surface_h_policy == "h0" and config.protons_per_molecule != 0:
        raise ValueError("surface_h_policy=h0 requires protons_per_molecule=0")
    if config.interface_pair_policy not in {
        "exact",
        "mobile",
        "phosphonate_triangle_height",
    }:
        raise ValueError(
            f"Unsupported interface-pair policy: {config.interface_pair_policy}"
        )
    if (
        not np.isfinite([config.mobile_interface_min, config.mobile_interface_max]).all()
        or config.mobile_interface_min <= 0.0
        or config.mobile_interface_max < config.mobile_interface_min
    ):
        raise ValueError("mobile interface distance window is invalid")
    if not config.mobile_interface_metal_elements or any(
        element not in atomic_numbers
        for element in config.mobile_interface_metal_elements
    ):
        raise ValueError("mobile interface metal elements must be known elements")
    if config.mobile_interface_donor_element not in atomic_numbers:
        raise ValueError("mobile interface donor element must be a known element")
    anchor_local_index = config.anchor_atom_index_within_molecule_0based
    if anchor_local_index is not None and (
        isinstance(anchor_local_index, bool)
        or not isinstance(anchor_local_index, (int, np.integer))
        or not 0 <= int(anchor_local_index) < config.atoms_per_molecule
    ):
        raise ValueError(
            "anchor_atom_index_within_molecule_0based must identify one atom in each SAM"
        )
    if config.triangle_anchor_molecule_numbers is not None:
        triangle_molecules = tuple(config.triangle_anchor_molecule_numbers)
        if (
            not triangle_molecules
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or not 1 <= int(value) <= config.molecules
                for value in triangle_molecules
            )
            or len(set(int(value) for value in triangle_molecules))
            != len(triangle_molecules)
        ):
            raise ValueError(
                "triangle_anchor_molecule_numbers must be unique valid 1-based SAM numbers"
            )
    allowed_interface_pairs = {
        frozenset(pair) for pair in config.allowed_interface_pairs
    }
    if path.suffix.lower() == ".data":
        atoms = read(
            path,
            format="lammps-data",
            atom_style="atomic",
        )
    else:
        atoms = read(path)
    symbols = np.asarray(atoms.get_chemical_symbols())
    if (
        isinstance(config.substrate_atoms, bool)
        or not isinstance(config.substrate_atoms, (int, np.integer))
        or not 0 <= int(config.substrate_atoms) <= len(atoms)
    ):
        raise ValueError("substrate_atoms must identify a valid leading atom block")
    registered_bonds, registered_all_pairs, registered_passed_pairs = (
        _normalize_registered_interface_bonds(
            config.registered_interface_bonds,
            symbols=symbols,
            atom_count=len(atoms),
            substrate_atoms=int(config.substrate_atoms),
            atoms=atoms,
            periodic_axes=periodic_axes,
        )
    )
    result: dict = {
        "path": str(path),
        "atom_count": len(atoms),
        "cell_A": np.asarray(atoms.cell).round(6).tolist(),
        "element_counts": dict(sorted(Counter(symbols).items())),
        "checks": {},
    }
    checks = result["checks"]
    checks["registered_interface_bonds_in_window"] = all(
        record["window_passed"] for record in registered_bonds
    )
    result["registered_interface_bonds"] = registered_bonds
    result["registered_interface_bond_violation_count"] = sum(
        not record["window_passed"] for record in registered_bonds
    )

    expected_total_atoms = (
        config.substrate_atoms + config.molecules * config.atoms_per_molecule
    )
    checks["expected_total_atoms"] = len(atoms) == expected_total_atoms
    if expected_formula is not None:
        checks["expected_element_counts"] = (
            result["element_counts"] == expected_formula
        )
    if anchor_local_index is None:
        checks["total_anchor_count"] = (
            int(np.count_nonzero(symbols == config.anchor_element)) == config.molecules
        )
    else:
        expected_anchor_indices = [
            config.substrate_atoms
            + molecule_number * config.atoms_per_molecule
            + int(anchor_local_index)
            for molecule_number in range(config.molecules)
        ]
        checks["total_anchor_count"] = all(
            symbols[index] == config.anchor_element for index in expected_anchor_indices
        )
    if config.marker_element:
        checks["total_marker_count"] = (
            int(np.count_nonzero(symbols == config.marker_element))
            == config.molecules
        )

    substrate_symbols = symbols[: config.substrate_atoms]
    substrate_formula = dict(sorted(Counter(substrate_symbols).items()))
    result["substrate_formula"] = substrate_formula
    if expected_substrate_formula is not None:
        checks["expected_substrate_formula"] = (
            substrate_formula == expected_substrate_formula
        )
    excluded_substrate_elements = [config.anchor_element]
    if config.marker_element:
        excluded_substrate_elements.append(config.marker_element)
    checks["substrate_contains_no_molecular_markers"] = not np.any(
        np.isin(substrate_symbols, excluded_substrate_elements)
    )

    substrate_indices = np.arange(config.substrate_atoms, dtype=int)
    substrate_h = substrate_indices[substrate_symbols == "H"]
    substrate_parent_atoms = substrate_indices[
        substrate_symbols == config.parent_element
    ]
    molecular_acceptors = np.flatnonzero(symbols == config.acceptor_element)
    molecular_acceptors = molecular_acceptors[
        molecular_acceptors >= config.substrate_atoms
    ]
    parent_atoms = []
    parent_h_distances = []
    proton_location_records = []
    secondary_acceptor_counts = []
    secondary_acceptor_records = []
    substrate_h_collisions = []
    for h_index in substrate_h:
        distances_to_parent = _surface_distances(
            atoms, int(h_index), substrate_parent_atoms, periodic_axes
        )
        nearest = int(np.argmin(distances_to_parent))
        parent = int(substrate_parent_atoms[nearest])
        parent_atoms.append(parent)
        parent_distance = float(distances_to_parent[nearest])
        parent_h_distances.append(parent_distance)

        if len(molecular_acceptors):
            distances_to_molecular_o = _surface_distances(
                atoms, int(h_index), molecular_acceptors, periodic_axes
            )
            molecular_nearest = int(np.argmin(distances_to_molecular_o))
            nearest_molecular_o = int(molecular_acceptors[molecular_nearest])
            molecular_distance = float(distances_to_molecular_o[molecular_nearest])
        else:
            nearest_molecular_o = None
            molecular_distance = float("inf")
        location = classify_surface_proton(
            parent_distance, molecular_distance, config.parent_h_bond_max
        )
        proton_location_records.append(
            {
                "H_index": int(h_index),
                "state": location,
                "nearest_substrate_O": parent,
                "H_substrate_O_distance_A": round(parent_distance, 6),
                "nearest_SAM_O": nearest_molecular_o,
                "H_SAM_O_distance_A": (
                    round(molecular_distance, 6)
                    if np.isfinite(molecular_distance)
                    else None
                ),
            }
        )

        h_to_parent = mic_vectors(
            atoms.positions[int(h_index)],
            atoms.positions[[parent]],
            np.asarray(atoms.cell),
            periodic_axes,
        )[0]
        h_to_parent /= np.linalg.norm(h_to_parent)
        secondary = []
        substrate_acceptors = substrate_indices[
            substrate_symbols == config.acceptor_element
        ]
        distances_to_substrate_acceptors = _surface_distances(
            atoms, int(h_index), substrate_acceptors, periodic_axes
        )
        for acceptor_index, distance in zip(
            substrate_acceptors, distances_to_substrate_acceptors
        ):
            if int(acceptor_index) == parent or not distance_within_closed_window(
                distance,
                config.secondary_min,
                config.secondary_max,
            ):
                continue
            h_to_acceptor = mic_vectors(
                atoms.positions[int(h_index)],
                atoms.positions[[int(acceptor_index)]],
                np.asarray(atoms.cell),
                periodic_axes,
            )[0]
            h_to_acceptor /= np.linalg.norm(h_to_acceptor)
            angle = float(
                np.degrees(
                    np.arccos(
                        np.clip(np.dot(h_to_parent, h_to_acceptor), -1.0, 1.0)
                    )
                )
            )
            if angle >= config.secondary_angle_min:
                secondary.append(
                    {
                        "acceptor_index": int(acceptor_index),
                        "distance_A": round(float(distance), 6),
                        "comparison_tolerance_A": (
                            closed_distance_window_tolerance_A(
                                distance,
                                config.secondary_min,
                                config.secondary_max,
                            )
                        ),
                        "angle_deg": round(angle, 3),
                    }
                )
        if len(molecular_acceptors):
            molecular_distances = _surface_distances(
                atoms, int(h_index), molecular_acceptors, periodic_axes
            )
            for acceptor_index, distance in zip(
                molecular_acceptors, molecular_distances
            ):
                if not distance_within_closed_window(
                    distance,
                    config.secondary_min,
                    config.secondary_max,
                ):
                    continue
                h_to_acceptor = mic_vectors(
                    atoms.positions[int(h_index)],
                    atoms.positions[[int(acceptor_index)]],
                    np.asarray(atoms.cell),
                    periodic_axes,
                )[0]
                h_to_acceptor /= np.linalg.norm(h_to_acceptor)
                angle = float(
                    np.degrees(
                        np.arccos(
                            np.clip(
                                np.dot(h_to_parent, h_to_acceptor), -1.0, 1.0
                            )
                        )
                    )
                )
                if angle >= config.secondary_angle_min:
                    secondary.append(
                        {
                            "acceptor_index": int(acceptor_index),
                            "distance_A": round(float(distance), 6),
                            "comparison_tolerance_A": (
                                closed_distance_window_tolerance_A(
                                    distance,
                                    config.secondary_min,
                                    config.secondary_max,
                                )
                            ),
                            "angle_deg": round(angle, 3),
                        }
                    )
        secondary_acceptor_counts.append(len(secondary))
        secondary_acceptor_records.append(
            {"H_index": int(h_index), "secondary_acceptors": secondary}
        )

        distances = _surface_distances(
            atoms, int(h_index), np.arange(len(atoms)), periodic_axes
        )
        for other_index, distance in enumerate(distances):
            if other_index in (int(h_index), parent):
                continue
            threshold = (
                config.hh_min
                if symbols[other_index] == "H"
                else config.h_heavy_min
            )
            if distance < threshold:
                substrate_h_collisions.append(
                    {
                        "H_index": int(h_index),
                        "other_index": int(other_index),
                        "other_element": str(symbols[other_index]),
                        "distance_A": round(float(distance), 6),
                        "threshold_A": threshold,
                    }
                )
    expected_surface_h = config.protons_per_molecule * config.molecules
    checks["substrate_H_count_per_SAM"] = len(substrate_h) == expected_surface_h
    no_surface_h_expected = expected_surface_h == 0
    checks["all_substrate_H_bonded_to_parent"] = (
        not len(substrate_h)
        if no_surface_h_expected
        else bool(parent_h_distances)
        and all(distance < config.parent_h_bond_max for distance in parent_h_distances)
    )
    checks["unique_parent_atom_per_H"] = len(parent_atoms) == len(set(parent_atoms))
    location_counts = Counter(record["state"] for record in proton_location_records)
    checks["no_unbound_surface_H"] = (
        not len(substrate_h)
        if no_surface_h_expected
        else bool(proton_location_records) and not location_counts["unbound"]
    )
    checks["no_substrate_H_nonparent_collisions"] = not substrate_h_collisions
    checks["all_substrate_H_have_secondary_acceptor"] = (
        not len(substrate_h)
        if no_surface_h_expected
        else bool(secondary_acceptor_counts)
        and all(count >= 1 for count in secondary_acceptor_counts)
    )
    result["substrate_hydrogen"] = {
        "count": len(substrate_h),
        "surface_H_per_SAM": len(substrate_h) / config.molecules if config.molecules else None,
        "unique_parent_atom_count": len(set(parent_atoms)),
        "proton_location_counts": dict(sorted(location_counts.items())),
        "non_substrate_only_protons": [
            record
            for record in proton_location_records
            if record["state"] != "substrate"
        ],
        "parent_H_distance_A": {
            "min": round(float(np.min(parent_h_distances)), 6) if parent_h_distances else None,
            "mean": round(float(np.mean(parent_h_distances)), 6) if parent_h_distances else None,
            "max": round(float(np.max(parent_h_distances)), 6) if parent_h_distances else None,
        },
        "nonparent_collision_count": len(substrate_h_collisions),
        "secondary_acceptor_contact_count": {
            "min": int(np.min(secondary_acceptor_counts)) if secondary_acceptor_counts else None,
            "mean": float(np.mean(secondary_acceptor_counts)) if secondary_acceptor_counts else None,
            "max": int(np.max(secondary_acceptor_counts)) if secondary_acceptor_counts else None,
            "distribution": dict(sorted(Counter(secondary_acceptor_counts).items())),
        },
        "secondary_acceptor_contacts": secondary_acceptor_records,
        "shortest_nonparent_collisions": sorted(
            substrate_h_collisions, key=lambda item: item["distance_A"]
        )[:20],
    }

    sam_atom_count = len(atoms) - config.substrate_atoms
    block_size = config.atoms_per_molecule
    checks["integer_molecule_block_size"] = (
        config.molecules > 0
        and sam_atom_count == config.molecules * config.atoms_per_molecule
    )
    result["sam_atom_count"] = sam_atom_count
    result["molecule_block_size"] = block_size

    molecule_issues: list[dict] = []
    molecule_formula_mismatches: list[dict] = []
    formulas: list[tuple[tuple[str, int], ...]] = []
    labels = np.zeros(len(atoms), dtype=np.int32)
    marker_exposures: list[float] = []

    if checks["integer_molecule_block_size"]:
        for molecule_number in range(config.molecules):
            start = config.substrate_atoms + molecule_number * block_size
            stop = start + block_size
            indices = np.arange(start, stop, dtype=int)
            labels[indices] = molecule_number + 1
            block_symbols = symbols[indices]
            formula = tuple(sorted(Counter(block_symbols).items()))
            formulas.append(formula)
            actual_block_formula = dict(formula)
            if (
                expected_molecule_formula is not None
                and actual_block_formula != expected_molecule_formula
            ):
                molecule_formula_mismatches.append(
                    {
                        "molecule": molecule_number + 1,
                        "actual_formula": actual_block_formula,
                        "expected_formula": expected_molecule_formula,
                    }
                )

            anchor_local = (
                np.asarray([indices[int(anchor_local_index)]], dtype=int)
                if anchor_local_index is not None
                else indices[block_symbols == config.anchor_element]
            )
            issues: list[str] = []
            if len(anchor_local) != 1:
                issues.append(
                    f"{config.anchor_element}_count={len(anchor_local)}"
                )

            marker_local = (
                indices[block_symbols == config.marker_element]
                if config.marker_element
                else np.asarray([], dtype=int)
            )
            if config.marker_element and len(marker_local) != 1:
                issues.append(
                    f"{config.marker_element}_count={len(marker_local)}"
                )

            if len(marker_local) == 1:
                marker_index = int(marker_local[0])
                distances = _surface_distances(
                    atoms, marker_index, indices, periodic_axes
                )
                carbon_neighbors = int(
                    np.count_nonzero((block_symbols == "C") & (distances < 2.05))
                )
                hydrogen_neighbors = int(
                    np.count_nonzero((block_symbols == "H") & (distances < 1.60))
                )
                if (
                    config.marker_carbon_neighbors is not None
                    and carbon_neighbors != config.marker_carbon_neighbors
                ):
                    issues.append(
                        f"{config.marker_element}-C_count={carbon_neighbors}"
                    )
                if config.forbid_marker_h and hydrogen_neighbors:
                    issues.append(
                        f"{config.marker_element}-H_count={hydrogen_neighbors}"
                    )
                heavy = indices[block_symbols != "H"]
                marker_exposures.append(
                    float(
                        np.max(atoms.positions[heavy, 2])
                        - atoms.positions[marker_index, 2]
                    )
                )

            component_size = molecule_component_size(
                atoms,
                indices,
                config.anchor_element,
                periodic_axes,
                anchor_index=(int(anchor_local[0]) if len(anchor_local) == 1 else None),
            )
            if component_size != block_size:
                issues.append(f"connected_from_P={component_size}/{block_size}")
            if issues:
                molecule_issues.append(
                    {"molecule": molecule_number + 1, "issues": issues}
                )

    checks["uniform_molecule_formula"] = bool(formulas) and len(set(formulas)) == 1
    if expected_molecule_formula is not None:
        checks["each_molecule_matches_expected_formula"] = (
            bool(formulas)
            and len(formulas) == config.molecules
            and not molecule_formula_mismatches
        )
    checks["all_molecule_topologies_valid"] = not molecule_issues and bool(formulas)
    result["molecule_formula"] = dict(formulas[0]) if formulas else None
    result["expected_molecule_formula"] = expected_molecule_formula
    result["molecule_formula_mismatches"] = molecule_formula_mismatches
    result["molecule_issues"] = molecule_issues

    triangle_height_report = None
    if config.interface_pair_policy == "phosphonate_triangle_height":
        if config.surface_h_policy != "h0":
            raise ValueError(
                "phosphonate_triangle_height requires surface_h_policy=h0"
            )
        triangle_ids = (
            tuple(range(1, config.molecules + 1))
            if config.triangle_anchor_molecule_numbers is None
            else tuple(int(value) for value in config.triangle_anchor_molecule_numbers)
        )
        triangle_height_report = _phosphonate_triangle_height_gate(
            atoms,
            config=config,
            symbols=symbols,
            substrate_symbols=substrate_symbols,
            triangle_anchor_molecule_numbers=triangle_ids,
        )
        checks[
            "configured_triangle_anchor_sams_have_surface_metal_inside_anchor_o_triangle"
        ] = (
            triangle_height_report["anchor_triangles_passed"]
        )
        checks["all_non_anchor_atoms_clear_surface_height"] = (
            triangle_height_report["non_anchor_heights_passed"]
        )

    mobile_interface_contacts: list[dict] = []
    mobile_interface_too_short: list[dict] = []
    mobile_interface_passed_pairs: set[tuple[int, int]] = set()
    fully_detached_molecules: list[int] = []
    mobile_contact_counts: list[int] = []
    if config.interface_pair_policy in {"mobile", "phosphonate_triangle_height"}:
        substrate_metal_indices = substrate_indices[
            np.isin(substrate_symbols, config.mobile_interface_metal_elements)
        ]
        if not len(substrate_metal_indices):
            raise ValueError("mobile interface policy found no configured substrate metals")
        for molecule_number in range(config.molecules):
            start = config.substrate_atoms + molecule_number * block_size
            stop = start + block_size
            indices = np.arange(start, stop, dtype=int)
            block_symbols = symbols[indices]
            anchors = (
                np.asarray([indices[int(anchor_local_index)]], dtype=int)
                if anchor_local_index is not None
                else indices[block_symbols == config.anchor_element]
            )
            donors = indices[block_symbols == config.mobile_interface_donor_element]
            if len(anchors) != 1:
                donor_indices = np.asarray([], dtype=int)
            else:
                anchor_distances = _surface_distances(
                    atoms, int(anchors[0]), donors, periodic_axes
                )
                donor_cutoff = covalent_cutoff(
                    config.anchor_element, config.mobile_interface_donor_element
                )
                if donor_cutoff is None:
                    raise ValueError(
                        "mobile interface donor-anchor covalent cutoff is undefined"
                    )
                donor_indices = donors[anchor_distances < donor_cutoff]

            molecule_contact_count = 0
            for donor_index in donor_indices:
                distances = _surface_distances(
                    atoms,
                    int(donor_index),
                    substrate_metal_indices,
                    periodic_axes,
                )
                for metal_index, distance in zip(substrate_metal_indices, distances):
                    distance = float(distance)
                    record = {
                        "molecule": molecule_number + 1,
                        "metal_atom_index_0based": int(metal_index),
                        "metal_element": str(symbols[metal_index]),
                        "donor_atom_index_0based": int(donor_index),
                        "donor_element": str(symbols[donor_index]),
                        "distance_A": round(distance, 6),
                    }
                    if distance_within_closed_window(
                        distance,
                        config.mobile_interface_min,
                        config.mobile_interface_max,
                    ):
                        mobile_interface_contacts.append(record)
                        mobile_interface_passed_pairs.add(
                            (int(metal_index), int(donor_index))
                        )
                        molecule_contact_count += 1
                    elif distance < config.mobile_interface_min:
                        mobile_interface_too_short.append(record)
            mobile_contact_counts.append(molecule_contact_count)
            if molecule_contact_count == 0:
                fully_detached_molecules.append(molecule_number + 1)

        mobile_interface_contacts.sort(
            key=lambda item: (
                item["molecule"],
                item["donor_atom_index_0based"],
                item["metal_atom_index_0based"],
            )
        )
        mobile_interface_too_short.sort(key=lambda item: item["distance_A"])
        checks["all_SAM_molecules_have_mobile_interface_contact"] = not (
            fully_detached_molecules
        )
        checks["no_mobile_interface_contacts_below_minimum"] = not (
            mobile_interface_too_short
        )
        if config.interface_pair_policy == "phosphonate_triangle_height":
            triangle_ids = set(
                triangle_height_report["triangle_anchor_molecule_numbers"]
            )
            non_triangle_molecules = set(range(1, config.molecules + 1)) - triangle_ids
            non_triangle_detached = [
                molecule
                for molecule in fully_detached_molecules
                if molecule in non_triangle_molecules
            ]
            non_triangle_too_short = [
                record
                for record in mobile_interface_too_short
                if record["molecule"] in non_triangle_molecules
            ]
            checks["all_other_sams_have_mobile_anchor_metal_contact"] = not (
                non_triangle_detached
            )
            checks["no_other_sam_mobile_interface_contacts_below_minimum"] = not (
                non_triangle_too_short
            )
            triangle_height_report["non_triangle_mobile_interface_failures"] = {
                "fully_detached_molecules": non_triangle_detached,
                "too_short_contact_count": len(non_triangle_too_short),
            }

    result["mobile_interface"] = {
        "policy": config.interface_pair_policy,
        "metal_elements": list(config.mobile_interface_metal_elements),
        "donor_element": config.mobile_interface_donor_element,
        "distance_window_A": [
            config.mobile_interface_min,
            config.mobile_interface_max,
        ],
        "contact_count": len(mobile_interface_contacts),
        "contact_count_per_molecule": mobile_contact_counts,
        "fully_detached_molecules": fully_detached_molecules,
        "too_short_contact_count": len(mobile_interface_too_short),
        "too_short_contacts": mobile_interface_too_short,
        "contacts": mobile_interface_contacts,
    }

    # Count only cross-component contacts. Intramolecular covalent bonds and
    # substrate-substrate bonds are deliberately excluded.
    collision_atoms = atoms.copy()
    collision_pbc = np.zeros(3, dtype=bool)
    collision_pbc[list(periodic_axes)] = True
    collision_atoms.set_pbc(collision_pbc)
    i_indices, j_indices, distances = neighbor_list(
        "ijd", collision_atoms, collision_neighbor_cutoff_A
    )
    unique = i_indices < j_indices
    i_indices = i_indices[unique]
    j_indices = j_indices[unique]
    distances = distances[unique]
    cross_component = labels[i_indices] != labels[j_indices]
    includes_sam = (labels[i_indices] != 0) | (labels[j_indices] != 0)
    evaluate = cross_component & includes_sam

    collision_records: list[dict] = []
    required_collision_records: list[dict] = []
    legacy_allowed_interface_contacts: list[dict] = []
    allowed_interface_contacts_compat: list[dict] = []
    registered_interface_contacts: list[dict] = []
    registered_bond_by_pair = {
        (
            record["substrate_atom_index_0based"],
            record["sam_atom_index_0based"],
        ): record
        for record in registered_bonds
    }
    for i, j, distance in zip(
        i_indices[evaluate], j_indices[evaluate], distances[evaluate]
    ):
        symbol_i, symbol_j = symbols[i], symbols[j]
        if symbol_i == "H" and symbol_j == "H":
            threshold = config.hh_min
        elif "H" in (symbol_i, symbol_j):
            threshold = config.h_heavy_min
        else:
            threshold = config.heavy_heavy_min
        if distance < threshold:
            record = {
                "i": int(i),
                "j": int(j),
                "elements": f"{symbol_i}-{symbol_j}",
                "distance_A": round(float(distance), 6),
                "threshold_A": threshold,
                "components": [int(labels[i]), int(labels[j])],
            }
            collision_records.append(record)
            exact_pair = (int(i), int(j))
            if (
                config.interface_pair_policy == "mobile"
                and exact_pair in mobile_interface_passed_pairs
            ):
                continue
            if exact_pair in registered_passed_pairs:
                bond = registered_bond_by_pair[exact_pair]
                registered_interface_contacts.append(
                    {
                        **record,
                        "classification": "exact_registered_interface_bond",
                        "minimum_distance_A": bond["minimum_distance_A"],
                        "maximum_distance_A": bond["maximum_distance_A"],
                        "window_passed": True,
                    }
                )
                continue
            # An explicitly registered pair outside its window cannot fall
            # back to the broad legacy element whitelist.
            registered_but_failed = exact_pair in registered_all_pairs
            legacy_allowed_interface = (
                not registered_but_failed
                and is_allowed_interface_pair(
                    str(symbol_i),
                    str(symbol_j),
                    int(labels[i]),
                    int(labels[j]),
                    allowed_interface_pairs,
                )
            )
            if legacy_allowed_interface:
                allowed_interface_contacts_compat.append(record)
                legacy_allowed_interface_contacts.append(
                    {**record, "classification": "legacy_element_pair_whitelist"}
                )
            else:
                is_substrate_sam_interface = (labels[i] == 0) != (labels[j] == 0)
                triangle_height_interface_diagnostic = (
                    config.interface_pair_policy == "phosphonate_triangle_height"
                    and is_substrate_sam_interface
                )
                surface_h_policy_exemption = (
                    is_substrate_sam_interface
                    and config.surface_h_policy in {"retained", "mobile"}
                    and "H" in (symbol_i, symbol_j)
                )
                if not surface_h_policy_exemption and not triangle_height_interface_diagnostic:
                    required_collision_records.append(record)

    collision_records.sort(key=lambda item: item["distance_A"])
    required_collision_records.sort(key=lambda item: item["distance_A"])
    checks["no_cross_component_collisions"] = not required_collision_records
    if config.interface_pair_policy == "phosphonate_triangle_height":
        sam_sam_collision_records = [
            record
            for record in required_collision_records
            if record["components"][0] != 0 and record["components"][1] != 0
        ]
        checks["no_sam_sam_collisions"] = not sam_sam_collision_records
    result["cross_component_collision_count"] = len(collision_records)
    result["required_cross_component_collision_count"] = len(
        required_collision_records
    )
    result["shortest_collisions"] = collision_records[:20]
    result["shortest_required_collisions"] = required_collision_records[:20]
    if triangle_height_report is not None:
        result["phosphonate_triangle_height_interface"] = triangle_height_report
    result["registered_interface_contacts"] = registered_interface_contacts
    result["legacy_allowed_interface_contacts"] = legacy_allowed_interface_contacts
    # Backward-compatible report alias retains the old record shape; formal
    # consumers should use the explicitly classified legacy/exact sections.
    result["allowed_interface_contacts"] = allowed_interface_contacts_compat

    if marker_exposures:
        result["marker_exposure_A"] = {
            "element": config.marker_element,
            "min": round(float(np.min(marker_exposures)), 6),
            "mean": round(float(np.mean(marker_exposures)), 6),
            "max": round(float(np.max(marker_exposures)), 6),
            "count_le_0.5": int(
                np.count_nonzero(np.asarray(marker_exposures) <= 0.5)
            ),
        }

    diagnostic_checks: list[str] = []
    if config.surface_h_policy == "retained":
        diagnostic_checks.extend(
            (
                "no_substrate_H_nonparent_collisions",
                "all_substrate_H_have_secondary_acceptor",
            )
        )
    elif config.surface_h_policy == "mobile":
        diagnostic_checks.extend(
            (
                "all_substrate_H_bonded_to_parent",
                "unique_parent_atom_per_H",
                "no_substrate_H_nonparent_collisions",
                "all_substrate_H_have_secondary_acceptor",
            )
        )
    if config.interface_pair_policy == "mobile":
        diagnostic_checks.append("registered_interface_bonds_in_window")
    if config.interface_pair_policy == "phosphonate_triangle_height":
        diagnostic_checks.append("no_cross_component_collisions")
        diagnostic_checks.extend(
            (
                "all_SAM_molecules_have_mobile_interface_contact",
                "no_mobile_interface_contacts_below_minimum",
            )
        )
    required_checks = [name for name in checks if name not in diagnostic_checks]
    nonperiodic_cell_axes = [
        axis for axis in range(3) if axis not in periodic_axes
    ]
    result["validation_policy"] = {
        "surface_h_policy": config.surface_h_policy,
        "interface_pair_policy": config.interface_pair_policy,
        "allowed_interface_pairs": [
            "-".join(pair) for pair in config.allowed_interface_pairs
        ],
        "legacy_allowed_interface_pairs": [
            "-".join(pair) for pair in config.allowed_interface_pairs
        ],
        "registered_interface_contract": {
            "schema": "global-0based-exact-substrate-sam-bonds-v1",
            "record_count": len(registered_bonds),
            "only_exact_index_pairs_exempt": True,
            "distance_window_must_pass_for_exemption": True,
            "independent_surface_only_mic_distance_check": True,
            "sam_sam_exemption": False,
        },
        "mobile_interface_contract": {
            "metal_elements": list(config.mobile_interface_metal_elements),
            "donor_element": config.mobile_interface_donor_element,
            "distance_window_A": [
                config.mobile_interface_min,
                config.mobile_interface_max,
            ],
            "donor_definition": "covalently bonded to the per-molecule anchor",
            "minimum_contacts_per_molecule": 1,
            "exact_registered_pair_retention": (
                "diagnostic_only"
                if config.interface_pair_policy == "mobile"
                else "required"
            ),
        },
        "triangle_height_interface_contract": (
            {
                "anchor_element": config.anchor_element,
                "anchor_oxygen_count": 3,
                "anchor_group_excluded_from_height_test": True,
                "accessible_surface_metal_elements": list(
                    config.mobile_interface_metal_elements
                ),
                "accessible_surface_metal_selection": (
                    "within_surface_metal_depth_A_of_highest_substrate_metal"
                ),
                "surface_metal_depth_A": float(config.surface_metal_depth_A),
                "minimum_non_anchor_height_A": float(
                    config.interface_height_clearance_A
                ),
                "exact_oxygen_metal_pair_required": False,
                "triangle_anchor_molecule_numbers": (
                    list(config.triangle_anchor_molecule_numbers)
                    if config.triangle_anchor_molecule_numbers is not None
                    else list(range(1, config.molecules + 1))
                ),
                "other_sam_anchor_policy": (
                    "at-least-one-covalent-anchor-O-to-configured-metal-contact-"
                    "inside-mobile-interface-window"
                ),
                "substrate_sam_collision_handling": (
                    "diagnostic_only; acceptance uses anchor triangle and all-"
                    "non-anchor height, while SAM-SAM collisions remain required"
                ),
            }
            if config.interface_pair_policy == "phosphonate_triangle_height"
            else None
        ),
        "collision_contract": {
            "thresholds_A": collision_thresholds_A,
            "neighbor_search_cutoff_A": collision_neighbor_cutoff_A,
            "neighbor_search_cutoff_definition": "max(H-H,H-heavy,heavy-heavy)",
            "surface_periodic_axes": list(periodic_axes),
            "nonperiodic_cell_axes": nonperiodic_cell_axes,
            "slab_normal_axis": nonperiodic_cell_axes[0],
            "slab_normal_wrapped": False,
            "minimum_image_convention": "declared_surface_periodic_axes_only",
        },
        "required_checks": required_checks,
        "diagnostic_checks": diagnostic_checks,
    }
    result["passed"] = all(checks[name] for name in required_checks)
    return result


class _SurfacePeriodicAxesAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        try:
            axes = _validated_surface_periodic_axes(values)
        except ValueError as exc:
            raise argparse.ArgumentError(self, str(exc)) from exc
        setattr(namespace, self.dest, axes)


def _positive_collision_distance(value):
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise argparse.ArgumentTypeError(
            "collision distances must be finite positive numbers"
        )
    return number


def build_parser() -> argparse.ArgumentParser:
    """Build the public, auditable validator CLI without executing it."""

    parser = argparse.ArgumentParser(
        description="Validate a substrate-first contiguous-block SAM structure."
    )
    parser.add_argument("structure", type=Path)
    parser.add_argument("--substrate-atoms", type=int, required=True)
    parser.add_argument("--molecules", type=int, required=True)
    parser.add_argument("--atoms-per-molecule", type=int, required=True)
    parser.add_argument("--protons-per-molecule", type=int, default=2)
    parser.add_argument("--anchor-element", default="P")
    parser.add_argument("--parent-element", default="O")
    parser.add_argument("--acceptor-element", default="O")
    parser.add_argument("--marker-element")
    parser.add_argument("--marker-carbon-neighbors", type=int)
    parser.add_argument("--forbid-marker-h", action="store_true")
    parser.add_argument(
        "--surface-h-policy",
        choices=("h0", "network", "retained", "mobile"),
        default="network",
        help=(
            "h0 requires an explicitly proton-free construction snapshot; network "
            "requires secondary O contacts and rejects close non-parent H contacts; "
            "retained requires that surface H remains bonded to substrate O; mobile "
            "permits transfer to SAM O and fails only when a surface H is unbound "
            "from both"
        ),
    )
    parser.add_argument(
        "--allowed-interface-pair",
        action="append",
        type=parse_element_pair,
        default=[],
        help=(
            "Legacy repeatable unordered element pair allowed only across the "
            "substrate-SAM interface, for example O-P; prefer exact registered "
            "interface bond records for formal validation"
        ),
    )
    parser.add_argument(
        "--registered-interface-bonds-json",
        type=Path,
        help=(
            "JSON list (or object containing registered_interface_bonds) of exact "
            "global 0-based substrate/SAM pairs and distance windows"
        ),
    )
    parser.add_argument(
        "--interface-pair-policy",
        choices=("exact", "mobile", "phosphonate_triangle_height"),
        default="exact",
        help=(
            "exact requires every registered substrate-SAM pair to remain in its "
            "window; mobile keeps exact retention diagnostic and instead requires "
            "at least one donor-to-metal contact per SAM; "
            "phosphonate_triangle_height requires an accessible metal projection "
            "inside each three-O anchor triangle and all other SAM atoms above "
            "the declared surface clearance"
        ),
    )
    parser.add_argument("--mobile-interface-min", type=float, default=1.70)
    parser.add_argument("--mobile-interface-max", type=float, default=2.80)
    parser.add_argument(
        "--mobile-interface-metal-element",
        action="append",
        default=None,
        help="Repeatable substrate metal element; default: In and Sn",
    )
    parser.add_argument("--mobile-interface-donor-element", default="O")
    parser.add_argument(
        "--triangle-anchor-molecule",
        action="append",
        type=int,
        default=None,
        help=(
            "1-based SAM block required to satisfy the projected three-O triangle "
            "rule; repeatable; default: every SAM block"
        ),
    )
    parser.add_argument(
        "--surface-metal-depth",
        type=float,
        default=0.7,
        help="Accessible metal depth below the highest configured substrate metal (A)",
    )
    parser.add_argument(
        "--interface-height-clearance",
        type=float,
        default=1.0,
        help="Minimum height of non-anchor SAM atoms above the highest substrate atom (A)",
    )
    parser.add_argument(
        "--surface-normal",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 1.0),
        metavar=("NX", "NY", "NZ"),
    )
    parser.add_argument("--parent-h-bond-max", type=float, default=1.25)
    parser.add_argument("--secondary-min", type=float, default=1.40)
    parser.add_argument("--secondary-max", type=float, default=2.40)
    parser.add_argument("--secondary-angle-min", type=float, default=105.0)
    parser.add_argument("--hh-min", type=_positive_collision_distance, default=1.20)
    parser.add_argument(
        "--h-heavy-min", type=_positive_collision_distance, default=1.40
    )
    parser.add_argument(
        "--heavy-heavy-min", type=_positive_collision_distance, default=1.80
    )
    parser.add_argument(
        "--surface-periodic-axes",
        nargs=2,
        type=int,
        action=_SurfacePeriodicAxesAction,
        default=(0, 1),
        metavar=("AXIS_U", "AXIS_V"),
        help=(
            "two distinct periodic cell axes used by every collision MIC; "
            "default: 0 1 (the remaining slab-normal axis is not wrapped)"
        ),
    )
    parser.add_argument(
        "--expected-formula-json",
        help='Optional JSON object, e.g. {"C":1408,"H":1280}',
    )
    parser.add_argument("--expected-substrate-formula-json")
    parser.add_argument(
        "--expected-molecule-formula-json",
        help="Explicit formula required independently for every contiguous SAM block",
    )
    parser.add_argument("--json-output", type=Path)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = ValidationConfig(
        substrate_atoms=args.substrate_atoms,
        molecules=args.molecules,
        atoms_per_molecule=args.atoms_per_molecule,
        protons_per_molecule=args.protons_per_molecule,
        anchor_element=args.anchor_element,
        parent_element=args.parent_element,
        acceptor_element=args.acceptor_element,
        marker_element=args.marker_element,
        marker_carbon_neighbors=args.marker_carbon_neighbors,
        forbid_marker_h=args.forbid_marker_h,
        parent_h_bond_max=args.parent_h_bond_max,
        secondary_min=args.secondary_min,
        secondary_max=args.secondary_max,
        secondary_angle_min=args.secondary_angle_min,
        hh_min=args.hh_min,
        h_heavy_min=args.h_heavy_min,
        heavy_heavy_min=args.heavy_heavy_min,
        expected_formula=(
            json.loads(args.expected_formula_json)
            if args.expected_formula_json
            else None
        ),
        expected_substrate_formula=(
            json.loads(args.expected_substrate_formula_json)
            if args.expected_substrate_formula_json
            else None
        ),
        expected_molecule_formula=(
            json.loads(args.expected_molecule_formula_json)
            if args.expected_molecule_formula_json
            else None
        ),
        surface_h_policy=args.surface_h_policy,
        interface_pair_policy=args.interface_pair_policy,
        mobile_interface_min=args.mobile_interface_min,
        mobile_interface_max=args.mobile_interface_max,
        mobile_interface_metal_elements=tuple(
            args.mobile_interface_metal_element or ("In", "Sn")
        ),
        mobile_interface_donor_element=args.mobile_interface_donor_element,
        surface_metal_depth_A=args.surface_metal_depth,
        interface_height_clearance_A=args.interface_height_clearance,
        surface_normal=tuple(args.surface_normal),
        triangle_anchor_molecule_numbers=(
            tuple(args.triangle_anchor_molecule)
            if args.triangle_anchor_molecule is not None
            else None
        ),
        allowed_interface_pairs=tuple(args.allowed_interface_pair),
        registered_interface_bonds=(
            load_registered_interface_bonds(args.registered_interface_bonds_json)
            if args.registered_interface_bonds_json
            else ()
        ),
        surface_periodic_axes=tuple(args.surface_periodic_axes),
    )
    result = validate(args.structure, config)
    rendered = json.dumps(result, indent=2, allow_nan=False)
    print(rendered)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n")
    return 0 if result["passed"] else 1


def audit_short_contacts(path,a,registered):
 import hashlib, collections
 path=Path(path)
 atoms=read(path);atoms.set_pbc([True,True,False]);sy=atoms.get_chemical_symbols()
 start=a.substrate_atoms+a.surface_h
 assert len(atoms)==start+a.molecules*a.mol_size
 labels=np.zeros(len(atoms),int);labels[a.substrate_atoms:start]=-1
 labels[start:]=np.repeat(np.arange(1,a.molecules+1),a.mol_size)
 ii,jj,dd=neighbor_list('ijd',atoms,3.)
 pairs={}
 for i,j,d in zip(ii,jj,dd):
  if i<j:pairs[(int(i),int(j))]=min(float(d),pairs.get((int(i),int(j)),float('inf')))
 parents={}
 for (i,j),d in pairs.items():
  for h,o in [(i,j),(j,i)]:
   if a.substrate_atoms<=h<start and sy[o]=='O' and d<a.parent_max:
    if h not in parents or d<parents[h][1]:parents[h]=(o,d)
 def row(i,j,d,threshold=None):
  return {'i_1based':i+1,'element_i':sy[i],'component_i':int(labels[i]),'j_1based':j+1,'element_j':sy[j],'component_j':int(labels[j]),'distance_A':d,'threshold_A':threshold}
 hits=[];parentrows=[];regrows=[];minima={}
 for (i,j),d in pairs.items():
  if labels[i]>0 and labels[i]==labels[j]:continue
  kind=('surface_H' if labels[i]==-1 or labels[j]==-1 else 'substrate_internal' if labels[i]==labels[j]==0 else 'SAM_substrate' if labels[i]==0 or labels[j]==0 else 'SAM_SAM')
  if parents.get(i,(None,))[0]==j or parents.get(j,(None,))[0]==i:
   parentrows.append(row(i,j,d));continue
  if (i,j) in registered:regrows.append(row(i,j,d))
  key=kind+':'+ '-'.join(sorted([sy[i],sy[j]]))
  if key not in minima or d<minima[key]['distance_A']:minima[key]=row(i,j,d)
  t=a.hh if sy[i]==sy[j]=='H' else a.h_heavy if 'H' in (sy[i],sy[j]) else a.heavy
  if d<t:hits.append(dict(row(i,j,d,t),category=kind,registered_interface=(i,j) in registered))
 hits.sort(key=lambda r:r['distance_A'])
 return {'input':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'count':len(hits),'by_category':dict(collections.Counter(x['category'] for x in hits)),'by_elements':dict(collections.Counter('-'.join(sorted([x['element_i'],x['element_j']])) for x in hits)),'contacts':hits,'current_surface_OH_contacts':sorted(parentrows,key=lambda x:x['distance_A']),'surface_H_without_O_within_parent_max':[h+1 for h in range(a.substrate_atoms,start) if h not in parents],'registered_interface_contacts_within_3A':regrows,'nonparent_minima_within_3A':minima}


def audit_pvksam_adsorption_metals(atoms, substrate_atoms, interface_bonds,
                                   metal_elements=("In", "Sn")):
    """Require CN <=5 before H0 relaxation / 优化前核对吸附金属资格.

    Count only original substrate O, under a/b MIC, at the unchanged 2.7 A
    cutoff. This is an adsorption-site eligibility rule, not a time-dependent
    coordination restraint. Every mapped metal of every denticity must pass.
    """
    from sam_structure_tools import calculate_substrate_metal_coordinations
    slab=atoms[:substrate_atoms]
    audit=calculate_substrate_metal_coordinations(
        slab.positions,slab.get_chemical_symbols(),slab.cell,
        periodic_axes=(0,1),normal_axis=2,cutoff_A=2.7,
        target_elements=metal_elements,ligand_elements=("O",))
    by_index=audit["metal_records_by_working_index"]
    mapped=sorted({int(b["metal"])-1 for b in interface_bonds})
    if not mapped:raise ValueError("PVKSAM requires registered adsorption metals")
    if any(i not in by_index for i in mapped):
        raise ValueError("PVKSAM interface has an unknown/non-substrate metal")
    records=[by_index[i] for i in mapped]
    forbidden=[r["working_atom_index_0based"]+1 for r in records if r["coordination_number"]>=6]
    return {"schema":"pvksam-adsorption-metal-eligibility-v1",
            "mode":"exclude-six-coordinated","cutoff_A":2.7,
            "max_allowed_coordination":5,"ligands":"substrate_O_only",
            "passed":not forbidden,"forbidden_metal_ids_1based":forbidden,
            "mapped_metal_records":records,"substrate_inventory":audit["inventory"]}


if __name__ == "__main__":
    raise SystemExit(main())
