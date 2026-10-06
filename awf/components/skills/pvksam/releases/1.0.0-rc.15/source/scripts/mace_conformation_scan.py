import os
import argparse
import csv
import hashlib
import json
import shlex
import sys
import traceback
import numpy as np
import ase
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii, vdw_radii
from ase.geometry import find_mic, get_dihedrals
from ase.io import read, write
from ase.constraints import FixAtoms
from collections import deque
from pathlib import Path

from sam_structure_tools import distance_within_closed_window


FIXED_SITE_CLUSTER_TOLERANCE_A = 2.0
FIXED_SITE_CLUSTER_SENSITIVITY_A = (0.5, 0.75, 1.0, 2.0)

# This is deliberately separate from the production fixed-site screening route. It is a
# narrowly scoped, experimental calibration contract for one molecule recovered
# from a sealed sequential-growth H0 trajectory; it is not a production screen.
TARGETED_CALIBRATION_SCHEMA_VERSION = 2
TARGETED_CALIBRATION_ROLE = "targeted_calibration_experimental_evidence"
TARGETED_CALIBRATION_MODEL_SHA256 = (
    "75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638"
)
TARGETED_CALIBRATION_ELEMENTS = ("H", "C", "N", "O", "P", "S", "In", "Sn")
TARGETED_CALIBRATION_MOLECULE_INDEX_1BASED = 62
TARGETED_CALIBRATION_SUBSTRATE_ATOMS = 2560
TARGETED_CALIBRATION_PARENT_MOLECULE_COUNT = 68
TARGETED_CALIBRATION_MOLECULE_ATOMS = 46
TARGETED_CALIBRATION_SURFACE_H_COUNT = 2
TARGETED_CALIBRATION_TOTAL_ATOMS = 2608
TARGETED_CALIBRATION_MOLECULE_FORMULA = {
    "C": 22,
    "H": 18,
    "N": 1,
    "O": 3,
    "P": 1,
    "S": 1,
}
TARGETED_CALIBRATION_SITE_INSTANCE_ID = (
    "site-prototype-phosphonic-acid-0014--cell-0017"
)
TARGETED_CALIBRATION_PROTOTYPE_ID = "site-prototype-phosphonic-acid-0014"
TARGETED_CALIBRATION_OXYGEN_PERMUTATION_0BASED = [44, 45, 43]
TARGETED_CALIBRATION_DONOR_LABELS = ["O1", "O2", "O3"]
TARGETED_CALIBRATION_TARGET_P_O_REFERENCE_A = 1.772
TARGETED_CALIBRATION_TARGET_P_O_REFERENCE_TOLERANCE_A = 0.05
TARGETED_CALIBRATION_P_O_COORDINATION_CUTOFF_A = 2.0
TARGETED_CALIBRATION_P_O_STABLE_CUTOFF_A = 2.0
TARGETED_CALIBRATION_P_O_RELEASED_CUTOFF_A = 2.2
TARGETED_CALIBRATION_P_COORDINATION_FIVEFOLD = 5
TARGETED_CALIBRATION_COVALENT_RADIUS_SCALE = 1.25
TARGETED_CALIBRATION_VDW_RADIUS_SCALE = 0.85
TARGETED_CALIBRATION_MAPPED_BOND_WINDOW_A = (1.7, 2.8)
TARGETED_CALIBRATION_MAX_STEPS = 500
TARGETED_CALIBRATION_FMAX_EVA = 0.03
TARGETED_CALIBRATION_MAXSTEP_A = 0.05
# Generic v3 experimental topology-contract capability.  The contract data are
# supplied by an explicit JSON configuration; these names identify the owner
# capability and do not select a molecule, substrate, or production route.
P_SURFACE_O_CONTRACT_SCHEMA_VERSION = 3
P_SURFACE_O_CONTRACT_ROLE = "p_surface_o_experimental_topology_contract_v3"
P_SURFACE_O_CONTRACT_MODE_NAMES = (
    "p-surface-o-contract-plan",
    "p-surface-o-contract-worker",
    "p-surface-o-contract-analyze",
)
P_SURFACE_O_AUDIT_MODULE_NAME = "sam_structure_tools"
P_SURFACE_O_AUDIT_FUNCTION_NAME = "periodic_vdw_collision_audit"
P_SURFACE_O_FORBIDDEN_CANDIDATE = {
    "candidate_id": "prototype0001",
    "status": "forbidden",
    "included_in_candidates": False,
    "reason": (
        "outside the minimal discriminating set and requires a different evidence "
        "chain with confounded site chemistry"
    ),
}

TARGETED_CALIBRATION_SEALED_AUDIT_ID_CONVENTION = {
    "audit_schema": "sam-sequential-final-registered-interface-audit-v1",
    "molecule_number_1based": "1-based placement ordinal",
    "global_atom_range_0based": "global final-H0 atom range",
    "molecule_atom_id": "molecule-local 0-based index",
    "substrate_atom_id": "working-substrate-local 0-based index",
    "canonical": (
        "molecule_atom_id=molecule-local-0based; "
        "substrate_atom_id=working-substrate-local-0based"
    ),
    "target_surface_oxygen_index": (
        "working-substrate-local 0-based index; numerically equal to the "
        "global final-H0 substrate index because the substrate is first"
    ),
    "source": "sealed audit schema and substrate-first H0 layout",
}


def _load_torch_for_mace_worker(
    worker_label, *, require_cuda, cuda_failure_message=None
):
    """Load PyTorch lazily and verify the worker runtime before MACE work."""

    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            f"{worker_label} cannot run MACE: PyTorch is unavailable; "
            "worker execution fails closed"
        ) from exc
    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:
        raise RuntimeError(
            f"{worker_label} cannot verify CUDA availability; "
            "worker execution fails closed"
        ) from exc
    if require_cuda and not cuda_available:
        raise RuntimeError(
            cuda_failure_message
            or f"{worker_label} requires CUDA; CPU fallback is forbidden"
        )
    return torch, cuda_available


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_donor_assignment(value):
    """Parse O1:45,O2:46,O3:44 into a label -> 1-based atom-ID mapping."""

    mapping = {}
    for item in str(value).split(","):
        if ":" not in item:
            raise ValueError("donor assignment entries must use LABEL:ATOM_ID")
        label, atom_id = (part.strip() for part in item.split(":", 1))
        if not label or not atom_id.isdigit() or int(atom_id) <= 0:
            raise ValueError(f"invalid donor assignment entry: {item!r}")
        if label in mapping:
            raise ValueError(f"duplicate donor label in assignment: {label}")
        mapping[label] = int(atom_id)
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("donor assignment atom IDs must be unique")
    return mapping


def _candidate_anchor_height_audit(
    positions_A,
    headgroup_atom_ids_0based,
    headgroup_target_positions_A,
    tolerance_A=1e-8,
):
    """Require every non-anchor atom to stay above the mean anchor-O height."""
    positions = np.asarray(positions_A, dtype=float)
    targets = np.asarray(headgroup_target_positions_A, dtype=float)
    head_ids = np.asarray(headgroup_atom_ids_0based, dtype=int)
    if positions.ndim != 2 or positions.shape[1] != 3 or not np.isfinite(positions).all():
        raise ValueError("candidate height audit requires finite N x 3 coordinates")
    if targets.shape != (4, 3) or not np.isfinite(targets).all():
        raise ValueError("candidate height audit requires finite P plus three O targets")
    if len(head_ids) != 4 or len(set(head_ids.tolist())) != 4:
        raise ValueError("candidate height audit requires four unique headgroup atom IDs")
    if np.any(head_ids < 0) or np.any(head_ids >= len(positions)):
        raise ValueError("candidate height audit headgroup atom ID is out of range")
    non_anchor_mask = np.ones(len(positions), dtype=bool)
    non_anchor_mask[head_ids] = False
    non_anchor_positions = positions[non_anchor_mask]
    if len(non_anchor_positions) == 0:
        raise ValueError("candidate height audit found no non-anchor atoms")
    reference_z = float(np.mean(targets[1:, 2]))
    minimum_z = float(np.min(non_anchor_positions[:, 2]))
    tolerance = float(tolerance_A)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("candidate height tolerance must be finite and nonnegative")
    below = max(0.0, reference_z - minimum_z)
    return {
        "passed": bool(minimum_z >= reference_z - tolerance),
        "reference_anchor_oxygen_mean_z_A": reference_z,
        "minimum_non_anchor_z_A": minimum_z,
        "below_reference_by_A": below,
        "tolerance_A": tolerance,
        "non_anchor_atom_count": int(np.count_nonzero(non_anchor_mask)),
    }


def complete_link_clusters(distance_matrix, tolerance):
    """Deterministically cluster a precomputed symmetric distance matrix.

    A candidate joins the first cluster only when it lies within ``tolerance``
    of every existing member.  This is the complete-link contract used for the
    optimized fixed-site SAM heavy-atom RMSD matrix.
    """

    distances = np.asarray(distance_matrix, dtype=float)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance matrix must be square")
    if not np.all(np.isfinite(distances)) or not np.allclose(
        distances, distances.T, atol=1.0e-12, rtol=0.0
    ):
        raise ValueError("distance matrix must be finite and symmetric")
    if not np.allclose(np.diag(distances), 0.0, atol=1.0e-12, rtol=0.0):
        raise ValueError("distance matrix diagonal must be zero")
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("complete-link tolerance must be positive")
    clusters = []
    for index in range(len(distances)):
        for cluster in clusters:
            if all(distances[index, member] <= tolerance for member in cluster):
                cluster.append(index)
                break
        else:
            clusters.append([index])
    return clusters


def pareto_front_indices(footprint_areas_A2, relative_energies_eV):
    """Return non-dominated indices for simultaneous area/energy minimization."""

    areas = np.asarray(footprint_areas_A2, dtype=float)
    energies = np.asarray(relative_energies_eV, dtype=float)
    if areas.shape != energies.shape or areas.ndim != 1 or not len(areas):
        raise ValueError("area and energy arrays must be non-empty and aligned")
    if not np.all(np.isfinite(areas)) or not np.all(np.isfinite(energies)):
        raise ValueError("area and energy arrays must be finite")
    front = []
    for index, (area, energy) in enumerate(zip(areas, energies)):
        dominated = np.any(
            (areas <= area)
            & (energies <= energy)
            & ((areas < area) | (energies < energy))
        )
        if not dominated:
            front.append(index)
    return sorted(front, key=lambda index: (areas[index], energies[index], index))


def _shortest_path(adjacency, start, goal):
    queue = deque([[start]])
    visited = {start}
    while queue:
        path = queue.popleft()
        if path[-1] == goal:
            return path
        for neighbor in sorted(adjacency[path[-1]]):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(path + [neighbor])
    raise ValueError("SAM anchor and terminal N are not covalently connected")


def _downstream(adjacency, fixed_side, rotating_side):
    visited = {fixed_side, rotating_side}
    queue = deque([rotating_side])
    result = [rotating_side]
    while queue:
        atom = queue.popleft()
        for neighbor in sorted(adjacency[atom]):
            if neighbor not in visited:
                visited.add(neighbor)
                result.append(neighbor)
                queue.append(neighbor)
    return result


def _intramolecular_collision_audit(atoms, adjacency, rotating_atoms):
    """Check non-bonded contacts involving atoms in the moving fragment."""

    symbols = atoms.get_chemical_symbols()
    distances = atoms.get_all_distances(mic=False)
    rotating_atoms = set(rotating_atoms)
    for left in range(len(atoms)):
        for right in range(left + 1, len(atoms)):
            if right in adjacency[left] or not ({left, right} & rotating_atoms):
                continue
            if symbols[left] == "H" and symbols[right] == "H":
                threshold = 1.2
            elif "H" in (symbols[left], symbols[right]):
                threshold = 1.4
            else:
                threshold = 1.8
            if distances[left, right] < threshold:
                return {
                    "passed": False,
                    "atom_ids_1based": [left + 1, right + 1],
                    "elements": [symbols[left], symbols[right]],
                    "distance_A": float(distances[left, right]),
                    "threshold_A": threshold,
                }
    return {"passed": True}


def _map_source_atom_to_supercell(source_atoms, supercell, source_atom_id, translation):
    from substrate_library import _map_periodic_vertex_to_supercell

    fractional = np.asarray(source_atoms.get_scaled_positions(wrap=True))[source_atom_id]
    position = (fractional + np.asarray(translation, dtype=int)) @ np.asarray(
        source_atoms.cell, dtype=float
    )
    return _map_periodic_vertex_to_supercell(
        supercell, position, source_atoms[source_atom_id].symbol
    )


def _prepare_fixed_site_system(args):
    """Build one current-surface, one-site, one-donor-map H0+2H system."""

    from substrate_library import (
        _allocate_layer_dopants,
        _build_doped_submission_supercell,
        _doping_layer_index_groups,
        _expand_site_prototypes_to_supercell,
        _load_accepted_site_prototype_library,
        _molecular_adjacency,
        _resolve_adsorption_vacuum,
        _resolve_public_doping_request,
        _resolve_public_supercell,
        _resolve_submission_recipe_and_sam,
        _unwrap_molecule,
    )
    from sam_structure_tools import periodic_vdw_collision_audit, rigid_transform

    sam_path = Path(args.sam).expanduser().resolve()
    recipe, sam_record = _resolve_submission_recipe_and_sam(
        sam_path,
        args.substrate,
        getattr(args, "substrate_catalog", None),
        allow_deprotonated_h0=True,
    )
    if sam_record["protonation"]["input_state"] != "already_deprotonated_h0":
        raise ValueError(
            "fixed-site scan currently requires a reviewed H0 molecule; neutral-acid "
            "intake remains handled by the public substrate workflow"
        )
    source_atoms = read(recipe["source_path"])
    doping = _resolve_public_doping_request(recipe, None)
    supercell_resolution = _resolve_public_supercell(
        recipe,
        sam_record,
        source_atoms,
        "single_molecule_adsorption",
        doping,
    )
    matrix = np.asarray(supercell_resolution["matrix"], dtype=int)
    multiplier = int(supercell_resolution["source_cell_multiplier"])
    structural_layers, host_layers = _doping_layer_index_groups(
        source_atoms,
        doping,
        recipe["slab"],
        recipe["surface"]["normal_axis"],
    )
    total_host_sites = sum(atom.symbol == doping["host_element"] for atom in source_atoms)
    target_dopants = int(round(total_host_sites * multiplier * doping["target_site_fraction"]))
    layer_host_counts = [len(layer) * multiplier for layer in host_layers]
    layer_dopants = _allocate_layer_dopants(
        layer_host_counts,
        [float(value) for value in doping["layer_site_fractions"]],
        target_dopants,
        doping["layer_integer_allocation"],
    )
    substrate_build = _build_doped_submission_supercell(
        source_atoms,
        matrix,
        doping,
        recipe["slab"],
        recipe["surface"]["normal_axis"],
        layer_dopants,
    )
    doped = substrate_build["doped"]
    source_cell = np.asarray(source_atoms.cell, dtype=float)
    target_cell = matrix @ source_cell
    vacuum = _resolve_adsorption_vacuum(
        recipe, sam_record, source_atoms, "single_molecule_adsorption"
    )
    normal_axis = int(recipe["surface"]["normal_axis"])
    normal = np.asarray(target_cell[normal_axis], dtype=float)
    unit_normal = normal / np.linalg.norm(normal)
    target_cell[normal_axis] = unit_normal * float(vacuum["target_normal_cell_length_A"])
    substrate_shift = float(vacuum["substrate_shift_along_normal_A"]) * unit_normal
    doped.positions += substrate_shift
    doped.set_cell(target_cell, scale_atoms=False)

    site_record, site_library, site_library_path = _load_accepted_site_prototype_library(
        recipe
    )
    clean = substrate_build["clean"]
    instances = _expand_site_prototypes_to_supercell(
        source_atoms, clean, matrix, site_library
    )
    wanted_instance_id = (
        f"{args.site_prototype_id}--cell-{int(args.site_cell):04d}"
    )
    selected = [
        instance for instance in instances
        if instance["site_instance_id"] == wanted_instance_id
    ]
    if len(selected) != 1:
        raise ValueError(f"fixed Site Instance not found: {wanted_instance_id}")
    instance = selected[0]
    coordination_filter = getattr(
        args, "metal_coordination_filter", "exclude-six-coordinated"
    )
    if coordination_filter != "exclude-six-coordinated":
        raise ValueError("fixed-site scans require exclude-six-coordinated metal filtering")
    from validate_monolayer import audit_pvksam_adsorption_metals
    eligibility = audit_pvksam_adsorption_metals(
        doped,
        len(doped),
        [{"metal": int(item["metal_atom_id_1based"])}
         for item in instance["donor_metal_mapping"]],
        ("In", "Sn"),
    )
    if not eligibility["passed"]:
        raise ValueError(
            "Fixed scan site includes forbidden CN>=6 metals: "
            + str(eligibility["forbidden_metal_ids_1based"])
        )
    template = instance["placement_template"]
    donor_assignment = _parse_donor_assignment(args.donor_assignment)
    expected_labels = list(sam_record["anchor"]["donor_labels"])
    if sorted(donor_assignment) != sorted(expected_labels):
        raise ValueError(
            f"donor assignment labels {sorted(donor_assignment)} do not match "
            f"catalog labels {sorted(expected_labels)}"
        )
    allowed_atom_ids = set(sam_record["anchor"]["donor_atom_ids_1based"])
    if set(donor_assignment.values()) != allowed_atom_ids:
        raise ValueError(
            "donor assignment must use every and only the submitted anchor O atoms"
        )

    sam_atoms = read(sam_path)
    sam_symbols = sam_atoms.get_chemical_symbols()
    adjacency = _molecular_adjacency(
        sam_symbols,
        np.asarray(sam_atoms.positions, dtype=float),
        np.asarray(sam_atoms.cell, dtype=float),
        np.asarray(sam_atoms.pbc, dtype=bool),
    )
    anchor = int(sam_record["anchor"]["atom_id_1based"]) - 1
    unwrapped = _unwrap_molecule(
        np.asarray(sam_atoms.positions, dtype=float),
        adjacency,
        np.asarray(sam_atoms.cell, dtype=float),
        np.asarray(sam_atoms.pbc, dtype=bool),
        anchor,
    )
    target_by_label = {
        item["probe_donor_id"]: np.asarray(item["cartesian_A"], dtype=float)
        + substrate_shift
        for item in template["binding_donor_positions"]
    }
    source_points = np.vstack(
        [
            unwrapped[anchor],
            *[unwrapped[donor_assignment[label] - 1] for label in expected_labels],
        ]
    )
    target_points = np.vstack(
        [
            np.asarray(template["anchor_cartesian_A"], dtype=float) + substrate_shift,
            *[target_by_label[label] for label in expected_labels],
        ]
    )
    rotation, translation, alignment_rmsd = rigid_transform(source_points, target_points)
    placed_positions = unwrapped @ rotation + translation
    placed_sam = sam_atoms.copy()
    placed_sam.positions = placed_positions
    placed_sam.set_cell(target_cell)
    placed_sam.set_pbc(doped.pbc)

    prototypes = {
        item["site_prototype_id"]: item for item in site_library["site_prototypes"]
    }
    prototype = prototypes[args.site_prototype_id]
    representative_path = (
        site_library_path.parent
        / prototype["representative_relaxed_structure"]["path"]
    ).resolve()
    if _sha256(representative_path) != prototype["representative_relaxed_structure"]["sha256"]:
        raise ValueError("Site Prototype representative structure hash changed")
    representative = read(representative_path)
    proton_positions = []
    proton_records = []
    for assignment in prototype["surface_proton_policy"]["selected_assignments"]:
        hydrogen_id = int(assignment["surface_proton_atom_id_0_based"])
        parent_id = int(assignment["nearest_oxygen_atom_id_0_based"])
        if parent_id >= len(source_atoms) or hydrogen_id >= len(representative):
            raise ValueError("Site Prototype surface-proton identity is invalid")
        vector = representative.positions[hydrogen_id] - representative.positions[parent_id]
        vector, _ = find_mic(vector, representative.cell, representative.pbc)
        target_parent = _map_source_atom_to_supercell(
            source_atoms,
            clean,
            parent_id,
            instance["source_cell_translation"],
        )
        position = doped.positions[target_parent] + vector
        proton_positions.append(position)
        proton_records.append(
            {
                "prototype_h_atom_id_0based": hydrogen_id,
                "prototype_parent_o_atom_id_0based": parent_id,
                "target_parent_o_atom_id_1based": target_parent + 1,
                "parent_h_distance_A": float(np.linalg.norm(vector)),
            }
        )
    expected_protons = int(recipe["adsorption"]["released_protons"])
    if len(proton_positions) != expected_protons:
        raise ValueError(
            f"prototype supplies {len(proton_positions)} surface H; expected {expected_protons}"
        )
    surface_h = Atoms("H" * len(proton_positions), positions=proton_positions)
    surface_h.set_cell(target_cell)
    surface_h.set_pbc(doped.pbc)
    substrate_with_h = doped + surface_h

    radii = {
        symbol: float(vdw_radii[atomic_numbers[symbol]])
        for symbol in sorted(set(substrate_with_h.get_chemical_symbols() + sam_symbols))
    }
    metal_by_label = {
        item["probe_donor_id"]: int(item["metal_atom_id_1based"])
        for item in instance["donor_metal_mapping"]
    }
    bond_window = tuple(
        float(value)
        for value in recipe["adsorption"]["site_discovery"]["probe_relaxation"]
        ["validation"]["final_metal_oxygen_distance_A"]
    )
    mapped_windows = {
        (donor_assignment[label], metal_by_label[label]): bond_window
        for label in expected_labels
    }
    initial_audit = periodic_vdw_collision_audit(
        molecule_positions=placed_sam.positions,
        molecule_symbols=sam_symbols,
        molecule_atom_ids=list(range(1, len(placed_sam) + 1)),
        substrate_positions=substrate_with_h.positions,
        substrate_symbols=substrate_with_h.get_chemical_symbols(),
        substrate_atom_ids=list(range(1, len(substrate_with_h) + 1)),
        cell=target_cell,
        periodic_axes=template["surface_frame"]["periodic_fractional_axes"],
        surface_frame=template["surface_frame"],
        radii_A=radii,
        radius_scale=float(args.vdw_radius_scale),
        mapped_bond_windows_A=mapped_windows,
    )
    return {
        "sam": placed_sam,
        "headgroup_atom_ids_0based": [anchor,*[donor_assignment[label]-1 for label in expected_labels]],
        "headgroup_target_positions_A": target_points,
        "substrate_with_h": substrate_with_h,
        "molecular_adjacency": adjacency,
        "mapped_windows": mapped_windows,
        "radii_A": radii,
        "surface_frame": template["surface_frame"],
        "initial_collision_audit": initial_audit,
        "metadata": {
            "substrate_atom_count": len(doped),
            "surface_h_count": len(surface_h),
            "sam_atom_count": len(placed_sam),
            "structural_layer_count": len(structural_layers),
            "outward_normal": unit_normal.tolist(),
            "registered_donor_metal_ids_1based": metal_by_label,
            "adsorption_metal_eligibility": eligibility,
            "sam_source": {"path": str(sam_path), "sha256": _sha256(sam_path)},
            "substrate_source": {
                "path": str(Path(recipe["source_path"]).resolve()),
                "sha256": recipe["source"]["sha256"],
            },
            "site_library": {
                "path": str(site_library_path),
                "sha256": site_record["sha256"],
            },
            "site_prototype_id": args.site_prototype_id,
            "site_profile_energy_eV": float(prototype["adsorption_energy_eV"]),
            "site_profile_energy_role": "reference_site_selection_only",
            "site_instance_id": wanted_instance_id,
            "donor_assignment_1based": donor_assignment,
            "alignment_rmsd_A": float(alignment_rmsd),
            "surface_protons": proton_records,
            "supercell": supercell_resolution,
            "dopant_count": len(substrate_build["selected_indices"]),
            "dopant_layer_counts_bottom_to_top": layer_dopants,
        },
    }


def _batch_dihedral_degrees(positions, torsion):
    """Return ASE-compatible dihedrals for a batch of Cartesian geometries."""

    a0, a1, a2, a3 = torsion
    return get_dihedrals(
        positions[:, a1, :] - positions[:, a0, :],
        positions[:, a2, :] - positions[:, a1, :],
        positions[:, a3, :] - positions[:, a2, :],
    )


def _batch_rotate_about_bond(positions, fixed_index, axis_index, moving_indices, delta_rad):
    """Apply the same ASE ``set_dihedral`` rotation to many geometries.

    ``set_dihedral`` rotates the downstream fragment about the bond from the
    second to the third atom in the four-atom path.  The Rodrigues expression
    below is algebraically identical, while allowing all grid members in a
    batch to use one NumPy operation.
    """

    fixed = positions[:, fixed_index, :]
    axis = positions[:, axis_index, :] - fixed
    axis /= np.linalg.norm(axis, axis=1)[:, None]
    relative = positions[:, moving_indices, :] - fixed[:, None, :]
    cosine = np.cos(delta_rad)[:, None, None]
    sine = np.sin(delta_rad)[:, None, None]
    rotated = (
        relative * cosine
        + np.cross(axis[:, None, :], relative)
        * sine
        + axis[:, None, :]
        * np.sum(axis[:, None, :] * relative, axis=2)[:, :, None]
        * (1.0 - cosine)
    )
    positions[:, moving_indices, :] = rotated + fixed[:, None, :]


def _full_grid_pair_contract(atoms, adjacency, rotating_atoms):
    """Resolve the historical intramolecular collision pairs once."""

    symbols = atoms.get_chemical_symbols()
    rotating_atoms = set(rotating_atoms)
    left, right, thresholds = [], [], []
    for atom_i in range(len(atoms)):
        for atom_j in range(atom_i + 1, len(atoms)):
            if atom_j in adjacency[atom_i] or not ({atom_i, atom_j} & rotating_atoms):
                continue
            if symbols[atom_i] == "H" and symbols[atom_j] == "H":
                threshold = 1.2
            elif "H" in (symbols[atom_i], symbols[atom_j]):
                threshold = 1.4
            else:
                threshold = 1.8
            left.append(atom_i)
            right.append(atom_j)
            thresholds.append(threshold)
    return (
        np.asarray(left, dtype=int),
        np.asarray(right, dtype=int),
        np.asarray(thresholds, dtype=float),
    )


def run_phosphonate_skeleton_cpu_screen(args):
    """Stream an isolated torsion grid, rejecting height/clashes before optimization.

    膦酸骨架CPU粗筛：三O平面设为z=0，P-C相位不进入骨架网格。
    This is deliberately a separate manifest contract from an ITO-site audit.
    One shared continuous P-C phase must satisfy height and head/body clearance.
    No calculator or optimizer is constructed in this stage.
    """
    import time
    from sam_lammps import write_manifest
    from substrate_library import _molecular_adjacency
    from sam_structure_tools import (
        standardize_phosphonate_head_plane,
        phosphonate_axial_height_intervals,
    )

    source = Path(args.sam).expanduser().resolve()
    molecule = read(source)
    molecule.set_constraint()
    molecule.set_pbc(False)
    symbols = molecule.get_chemical_symbols()
    adjacency = _molecular_adjacency(
        symbols, molecule.positions, np.zeros((3, 3)), np.zeros(3, dtype=bool)
    )
    p_ids = [i for i, s in enumerate(symbols) if s == "P"]
    if len(p_ids) != 1:
        raise ValueError("skeleton scan requires one reviewed phosphonate P")
    p = p_ids[0]
    oxygen = sorted(i for i in adjacency[p] if symbols[i] == "O")
    carbon = sorted(i for i in adjacency[p] if symbols[i] == "C")
    if len(oxygen) != 3 or len(carbon) != 1 or len(adjacency[p]) != 4:
        raise ValueError("skeleton scan requires a P(O)3-C anchor")
    c = carbon[0]
    head = {p, *oxygen}
    organic = sorted(_downstream(adjacency, p, c))
    if head.intersection(organic) or set(organic) | head != set(range(len(molecule))):
        raise ValueError("reviewed H0 must contain only PO3 and its connected C-side fragment")
    explicit = getattr(args, "torsion_atom_ids", None)
    if explicit is None:
        raise ValueError("skeleton scan requires explicit bonded torsion_atom_ids (1-based)")
    torsions, moving, omitted = [], [], []
    central_bonds = set()
    for raw in explicit:
        if len(raw) != 4 or any(type(i) is not int for i in raw):
            raise ValueError("torsion paths require four integer 1-based atom IDs")
        row = tuple(i - 1 for i in raw)
        if len(set(row)) != 4 or min(row) < 0 or max(row) >= len(molecule):
            raise ValueError("invalid skeleton torsion atom IDs")
        if any(b not in adjacency[a] for a, b in zip(row, row[1:])):
            raise ValueError("skeleton torsions must be bonded four-atom paths")
        bond = frozenset(row[1:3])
        if bond == frozenset((p, c)):
            omitted.append(list(raw))
            continue
        if bond in central_bonds:
            raise ValueError("duplicate central bond in skeleton torsions")
        central_bonds.add(bond)
        indices = _downstream(adjacency, row[1], row[2])
        if head.intersection(indices) or any(
            j != row[2] and j in indices for j in adjacency[row[1]]
        ):
            raise ValueError("skeleton torsion must be acyclic and keep PO3 fixed")
        torsions.append(row)
        moving.append(indices)

    step = float(getattr(args, "scan_step_deg", 30.0))
    angle_count = int(round(360.0 / step)) if np.isfinite(step) and step > 0 else 0
    if angle_count < 1 or not np.isclose(angle_count * step, 360.0, atol=1e-10, rtol=0):
        raise ValueError("scan_step_deg must divide 360 degrees exactly")
    angles = -180.0 + np.arange(angle_count) * step
    grid_size = angle_count ** len(torsions)
    limit = getattr(args, "skeleton_limit", None)
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("skeleton_limit must be a positive integer")
    evaluate = min(grid_size, limit) if limit is not None else grid_size
    maximum = int(getattr(args, "scan_max_candidates", 100000))
    batch_size = int(getattr(args, "full_grid_batch_size", 256))
    if evaluate > maximum or maximum <= 0 or batch_size <= 0:
        raise ValueError("scan exceeds max-candidates, or batch/max-candidates is invalid")
    name = str(getattr(args, "skeleton_isomer_name", None) or source.stem)
    if not name or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in name):
        raise ValueError("skeleton isomer name must contain only letters, numbers, '_' or '-'")
    standardized, frame = standardize_phosphonate_head_plane(molecule.positions, p, oxygen)
    molecule.positions = standardized
    left, right, thresholds = _full_grid_pair_contract(molecule, adjacency, range(len(molecule)))
    organic_mask = np.zeros(len(molecule), dtype=bool)
    organic_mask[organic] = True
    invariant = organic_mask[left] == organic_mask[right]
    invariant_left, invariant_right = left[invariant], right[invariant]
    invariant_thresholds = thresholds[invariant]
    cross_pairs = np.column_stack((left[~invariant], right[~invariant])).tolist()
    cross_thresholds = thresholds[~invariant].tolist()
    parameters = {
        "step_deg": step, "scan_angles_deg": angles.tolist(),
        "torsion_atom_ids_1based": [[i + 1 for i in row] for row in torsions],
        "omitted_pc_phase_torsions_1based": omitted,
        "pc_axis_atom_ids_1based": [p + 1, c + 1],
        "head_atom_ids_1based": [p + 1, *[i + 1 for i in oxygen]],
        "height_scope": "all_C_side_atoms_including_H_excluding_PO3",
        "height_floor_A": 0.0, "height_numerical_tolerance_A": 1e-8,
        "axial_phase": "continuous_joint_height_and_nonbonded_clearance_intervals",
        "collision_min_A": {"H-H": 1.2, "H-heavy": 1.4, "heavy-heavy": 1.8},
        "bonded_pairs_excluded": True, "batch_size": batch_size,
        "evaluated_limit": limit, "max_candidates": maximum,
        "scope": "isolated_geometry_only_not_ITO_collision_audit",
    }
    sources = {
        "sam": {"path": str(source), "sha256": _sha256(source)},
        "scanner_sha256": _sha256(Path(__file__)),
        "geometry_owner_sha256": _sha256(Path(__file__).with_name("sam_structure_tools.py")),
    }
    identity = hashlib.sha256(json.dumps({"sources": sources, "parameters": parameters}, sort_keys=True).encode()).hexdigest()
    run = Path(args.output_root).expanduser().resolve() / ("skeleton-grid-" + identity[:12])
    if run.exists():
        raise ValueError("immutable skeleton scan already exists: " + str(run))
    structures = run / (name + "_monomer_unoptimized_conformations")
    structures.mkdir(parents=True)
    write(run / "sam-h0.extxyz", molecule)
    counts = {"grid_candidate_count": grid_size, "evaluated_candidate_count": 0,
              "z_rejected_count": 0, "intramolecular_collision_rejected_count": 0,
              "joint_phase_rejected_count": 0, "screen_safe_count": 0,
              "generation_failed_count": 0, "optimization_started_count": 0}
    records = []
    outcome_path = run / "candidate-outcomes.jsonl"
    started = time.perf_counter()
    with outcome_path.open("w", encoding="utf-8") as log:
        for start in range(0, evaluate, batch_size):
            size = min(batch_size, evaluate - start)
            quotient = np.arange(start, start + size, dtype=np.int64)
            digits = np.empty((size, len(torsions)), dtype=np.int64)
            for column in range(len(torsions) - 1, -1, -1):
                digits[:, column] = quotient % angle_count
                quotient //= angle_count
            target_angles = angles[digits]
            xyz = np.broadcast_to(standardized, (size, len(molecule), 3)).copy()
            for column, (row, indices) in enumerate(zip(torsions, moving)):
                current = _batch_dihedral_degrees(xyz, row)
                _batch_rotate_about_bond(xyz, row[1], row[2], indices,
                                        np.deg2rad(target_angles[:, column] - current))
            distances2 = np.sum((xyz[:, invariant_left] - xyz[:, invariant_right]) ** 2, axis=2)
            collisions = distances2 < (invariant_thresholds[None, :] - 1e-8) ** 2
            for i in range(size):
                record = {"grid_index": start + i + 1,
                          "start_dihedrals_deg": target_angles[i].tolist()}
                counts["evaluated_candidate_count"] += 1
                if not np.isfinite(xyz[i]).all():
                    record.update(status="rejected", reason="nonfinite_generated_geometry")
                    counts["generation_failed_count"] += 1
                elif np.any(collisions[i]):
                    hit = int(np.flatnonzero(collisions[i])[0])
                    record.update(status="rejected", reason="intramolecular_collision",
                        first_collision={"atom_ids_1based": [int(invariant_left[hit])+1, int(invariant_right[hit])+1],
                            "distance_A": float(np.sqrt(distances2[i, hit])),
                            "threshold_A": float(invariant_thresholds[hit])})
                    counts["intramolecular_collision_rejected_count"] += 1
                else:
                    phase = phosphonate_axial_height_intervals(
                        xyz[i], p, c, organic, z_floor=0.0, tolerance=1e-8,
                        collision_pairs_0based=cross_pairs,
                        collision_min_distances_A=cross_thresholds,
                    )
                    if not phase["feasible"]:
                        reason = "height_no_feasible_pc_phase" if not phase["height_only_feasible"] else "height_and_clearance_no_common_pc_phase"
                        record.update(status="rejected", reason=reason)
                        counts["z_rejected_count" if not phase["height_only_feasible"] else "joint_phase_rejected_count"] += 1
                    else:
                        candidate = molecule.copy()
                        candidate.positions = xyz[i].copy()
                        selected_angle = float(phase["representative_angle_deg"])
                        rotated = candidate.positions[None, :, :].copy()
                        _batch_rotate_about_bond(rotated, p, c, organic,
                                                np.array([np.deg2rad(selected_angle)]))
                        candidate.positions = rotated[0]
                        all_distances = np.linalg.norm(candidate.positions[left] - candidate.positions[right], axis=1)
                        clearance_ok = bool(np.all(all_distances >= thresholds - 1.1e-8))
                        min_z = float(np.min(candidate.positions[organic, 2]))
                        if min_z < -1.1e-8 or not clearance_ok:
                            raise RuntimeError("analytic feasible phase failed independent geometry verification")
                        counts["screen_safe_count"] += 1
                        rank = counts["screen_safe_count"]
                        path = structures / ("candidate-" + str(rank).zfill(6) + ".extxyz")
                        write(path, candidate)
                        record.update(status="passed_skeleton_cpu_screen", safe_rank=rank,
                            structure=str(path.relative_to(run)), structure_sha256=_sha256(path),
                            minimum_organic_z_A=min_z, selected_pc_phase_deg=selected_angle,
                            feasible_pc_intervals_deg=phase["feasible_intervals_deg"])
                        records.append(record)
                log.write(json.dumps(record, separators=(",", ":")) + "\n")
            write_manifest(run / "progress.json", {"status": "running", "summary": counts,
                "elapsed_s": time.perf_counter() - started})
            print(json.dumps({"stage": "phosphonate_skeleton_cpu", "evaluated": counts["evaluated_candidate_count"],
                              "safe": counts["screen_safe_count"], "total": evaluate}), flush=True)
    payload = {
        "schema": "sam-phosphonate-skeleton-cpu-v1", "run_identity": identity,
        "status": "screen_completed" if evaluate == grid_size else "partial_screen_completed",
        "grid_complete": evaluate == grid_size, "scope": parameters["scope"],
        "records_scope": "isolated_skeleton_screen_safe_only", "sources": sources,
        "parameters": parameters, "standard_head_frame": frame,
        "system": {"sam_atom_count": len(molecule), "sam_source": {"path": str(run / "sam-h0.extxyz"), "sha256": _sha256(run / "sam-h0.extxyz")}},
        "summary": {**counts, "elapsed_s": time.perf_counter() - started},
        "candidate_outcomes": str(outcome_path), "records": records,
        "next_stage": "common_fast_optimization_then_PC_axis_invariant_skeleton_deduplication",
        "not_yet_validated": ["optimized_basin_identity", "ITO_site_collision", "adsorption_energy"],
    }
    write_manifest(run / "manifest.json", payload)
    write_manifest(run / "progress.json", {"status": payload["status"], "summary": payload["summary"]})
    print(json.dumps({"manifest": str(run / "manifest.json"), "summary": payload["summary"]}), flush=True)
    return 0


def _validate_single_pc_roll_selection(ensemble_path, ensemble):
    """Require the hash-bound output receipt from the sole PVKSAM roll selector."""

    if ensemble.get("schema") != "pvksam-fixed-site-candidate-ensemble-v1":
        raise ValueError("fixed-site-cpu accepts only the PVKSAM selected P-C roll ensemble schema")
    if ensemble.get("candidate_source") != (
        "one lowest-MMFF94s-single-point-energy surface-safe P-C roll per skeleton"
    ):
        raise ValueError("candidate ensemble was not produced by the selected MMFF94s P-C roll route")
    provenance = ensemble.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("selection_method") != (
        "minimum MMFF94s single-point energy among CPU-screen-passing rolls, independently per skeleton"
    ):
        raise ValueError("candidate ensemble is missing the PVKSAM single-roll selection receipt")
    manifest_name = provenance.get("selection_manifest")
    if not isinstance(manifest_name, str) or not manifest_name or Path(manifest_name).name != manifest_name:
        raise ValueError("selection_manifest must name a sibling selection receipt")
    selection_path = Path(ensemble_path).resolve().parent / manifest_name
    if not selection_path.is_file():
        raise ValueError(f"PVKSAM single-roll selection receipt not found: {selection_path}")
    selection = json.loads(selection_path.read_text())
    if not isinstance(selection, dict) or selection.get("schema") != "pvksam-single-pc-roll-selection-v1" or selection.get("status") != "passed":
        raise ValueError("PVKSAM single-roll selection receipt has the wrong schema or status")
    outputs = selection.get("outputs")
    selected_path = Path(outputs.get("selected_candidate_ensemble", "")).expanduser().resolve() if isinstance(outputs, dict) else None
    if selected_path != Path(ensemble_path).resolve():
        raise ValueError("selection receipt does not identify this exact candidate ensemble")
    if selection.get("selected_candidate_ensemble_sha256") != _sha256(ensemble_path):
        raise ValueError("selection receipt does not hash-bind this candidate ensemble")
    ensemble_records = ensemble.get("records")
    selected_records = selection.get("selected")
    if not isinstance(ensemble_records, list) or not isinstance(selected_records, list):
        raise ValueError("selection receipt and ensemble must both contain records")
    ensemble_ids = [row.get("candidate_id") for row in ensemble_records if isinstance(row, dict)]
    selected_ids = [row.get("candidate_id") for row in selected_records if isinstance(row, dict)]
    if len(ensemble_ids) != len(ensemble_records) or ensemble_ids != selected_ids:
        raise ValueError("selection receipt candidate IDs do not match the selected ensemble")
    if selection.get("summary", {}).get("selected_roll_count") != len(ensemble_records):
        raise ValueError("selection receipt count does not match the selected ensemble")


def run_fixed_site_cpu_screen(args):
    """Align and screen the selected route's precomputed site-roll ensemble."""

    from sam_lammps import write_manifest
    from sam_structure_tools import periodic_vdw_collision_audit, rigid_transform
    from substrate_library import _molecular_adjacency

    ensemble_path = getattr(args, "candidate_ensemble", None)
    if ensemble_path is None:
        raise ValueError(
            "fixed-site-cpu requires the precomputed phosphonate P-C roll ensemble; "
            "direct fixed-site torsion-grid generation has been removed"
        )
    if not getattr(args, "organic_z_floor_at_anchor_oxygen_mean", False):
        raise ValueError(
            "fixed-site-cpu requires --organic-z-floor-at-anchor-oxygen-mean"
        )

    ensemble_path = Path(ensemble_path).expanduser().resolve()
    if not ensemble_path.is_file():
        raise ValueError(f"candidate ensemble does not exist: {ensemble_path}")
    ensemble = json.loads(ensemble_path.read_text())
    if not isinstance(ensemble, dict):
        raise ValueError("candidate ensemble root must be an object")
    _validate_single_pc_roll_selection(ensemble_path, ensemble)
    records_in = ensemble.get("records")
    if not isinstance(records_in, list) or not records_in:
        raise ValueError("candidate ensemble must contain a nonempty records list")

    prepared = _prepare_fixed_site_system(args)
    molecule = prepared["sam"]
    substrate = prepared["substrate_with_h"]
    adjacency = prepared["molecular_adjacency"]
    symbols = molecule.get_chemical_symbols()
    if ensemble.get("symbols") != symbols:
        raise ValueError("candidate ensemble has wrong atom identities")
    candidate_ids = [row.get("candidate_id") for row in records_in if isinstance(row, dict)]
    if len(candidate_ids) != len(records_in) or any(value is None for value in candidate_ids):
        raise ValueError("every candidate ensemble record needs a candidate_id")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("duplicate candidate IDs in candidate ensemble")

    head_targets = np.asarray(prepared["headgroup_target_positions_A"], dtype=float)
    if head_targets.shape != (4, 3) or not np.isfinite(head_targets).all():
        raise ValueError("site-roll screening requires finite P plus three anchor-O targets")
    alignment_limit = float(args.headgroup_alignment_max_rmsd_A)
    if not np.isfinite(alignment_limit) or alignment_limit < 0.0:
        raise ValueError("headgroup alignment RMSD limit must be finite and nonnegative")

    parameter_record = {
        "stage": "site_bound_pc_roll_headgroup_alignment_and_cpu_collision_screen",
        "site_prototype_id": args.site_prototype_id,
        "site_cell": int(args.site_cell),
        "donor_assignment": _parse_donor_assignment(args.donor_assignment),
        "torsion_atom_ids_1based": [],
        "headgroup_alignment_max_rmsd_A": alignment_limit,
        "vdw_radius_scale": float(args.vdw_radius_scale),
        "metal_coordination_filter": "exclude-six-coordinated",
        "metal_coordination_cutoff_A": 2.7,
        "metal_coordination_max_allowed": 5,
        "organic_height_floor_at_anchor_oxygen_mean": {
            "enabled": True,
            "tolerance_A": 1e-8,
            "meaning": "after headgroup alignment, every non-anchor atom must have z at least the mean z of the three target anchor O atoms",
        },
        "intramolecular_collision_contract": {
            "H-H_A": 1.2,
            "H-heavy_A": 1.4,
            "heavy-heavy_A": 1.8,
            "origin": "PVKSAM phosphonate skeleton route",
        },
        "future_complete_link_rmsd_tolerance_A": FIXED_SITE_CLUSTER_TOLERANCE_A,
        "future_complete_link_rmsd_sensitivity_A": list(FIXED_SITE_CLUSTER_SENSITIVITY_A),
        "candidate_ensemble": {
            "path": str(ensemble_path),
            "sha256": _sha256(ensemble_path),
            "candidate_count": len(records_in),
            "candidate_source": ensemble.get("candidate_source", "unspecified"),
        },
        "energy_quantity": "relative_total_energy_same_composition_not_adsorption_energy",
    }
    identity = hashlib.sha256(
        json.dumps(
            {"parameters": parameter_record, "sources": prepared["metadata"]},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    output_root = Path(args.output_root).expanduser().resolve()
    run_directory = output_root / f"fixed-site-monomer-scan-{identity[:12]}"
    if run_directory.exists():
        raise ValueError(f"immutable run directory already exists: {run_directory}")
    structures_directory = run_directory / "01_cpu_screen" / "structures"
    structures_directory.mkdir(parents=True)

    records = []
    safe_count = 0
    intramolecular_rejected = 0
    substrate_rejected = 0
    organic_height_rejected = 0
    full_substrate_checked = 0
    full_substrate_rejected = 0
    generation_failed = 0
    for grid_index, seed in enumerate(records_in, 1):
        candidate = molecule.copy()
        record = {"grid_index": grid_index, "candidate_id": seed["candidate_id"]}
        try:
            xyz = np.asarray(seed["positions_A"], dtype=float)
            if xyz.shape != (len(molecule), 3) or not np.isfinite(xyz).all():
                raise ValueError("invalid candidate-ensemble coordinates")
            graph = _molecular_adjacency(symbols, xyz, np.zeros((3, 3)), np.zeros(3, dtype=bool))
            if [set(v) for v in graph] != [set(v) for v in adjacency]:
                raise ValueError("candidate-ensemble structure changed molecular connectivity")
            anchor = [i for i, symbol in enumerate(symbols) if symbol == "P"]
            if len(anchor) != 1:
                raise ValueError("expected one phosphonate anchor")
            head = prepared["headgroup_atom_ids_0based"]
            rotation, translation, rmsd = rigid_transform(
                xyz[head], prepared["headgroup_target_positions_A"]
            )
            record["headgroup_alignment_rmsd_A"] = float(rmsd)
            if rmsd > alignment_limit:
                raise ValueError("headgroup alignment exceeds configured RMSD")
            candidate.positions = xyz @ rotation + translation
        except Exception as exc:
            generation_failed += 1
            record.update({
                "status": "rejected",
                "reason": "candidate_alignment_or_geometry_failed",
                "detail": str(exc),
            })
            records.append(record)
            continue

        height_audit = _candidate_anchor_height_audit(
            candidate.positions,
            prepared["headgroup_atom_ids_0based"],
            prepared["headgroup_target_positions_A"],
        )
        record["organic_anchor_height_audit"] = height_audit
        if not height_audit["passed"]:
            organic_height_rejected += 1
            substrate_rejected += 1
            record.update({
                "status": "rejected",
                "reason": "non_anchor_atom_below_anchor_oxygen_mean_z",
            })
            records.append(record)
            continue

        self_audit = _intramolecular_collision_audit(
            candidate, adjacency, range(len(molecule))
        )
        if not self_audit["passed"]:
            intramolecular_rejected += 1
            record.update({
                "status": "rejected",
                "reason": "intramolecular_collision",
                "collision": self_audit,
            })
            records.append(record)
            continue

        full_substrate_checked += 1
        substrate_audit = periodic_vdw_collision_audit(
            molecule_positions=candidate.positions,
            molecule_symbols=symbols,
            molecule_atom_ids=list(range(1, len(candidate) + 1)),
            substrate_positions=substrate.positions,
            substrate_symbols=substrate.get_chemical_symbols(),
            substrate_atom_ids=list(range(1, len(substrate) + 1)),
            cell=np.asarray(substrate.cell, dtype=float),
            periodic_axes=prepared["surface_frame"]["periodic_fractional_axes"],
            surface_frame=prepared["surface_frame"],
            radii_A=prepared["radii_A"],
            radius_scale=float(args.vdw_radius_scale),
            mapped_bond_windows_A=prepared["mapped_windows"],
        )
        if not substrate_audit["passed"]:
            substrate_rejected += 1
            full_substrate_rejected += 1
            record.update({
                "status": "rejected",
                "reason": "substrate_or_surface_proton_collision",
                "collision_count": substrate_audit["collision_count"],
                "mapped_bond_violation_count": substrate_audit["mapped_bond_violation_count"],
                "first_collision": substrate_audit["collisions"][0] if substrate_audit["collisions"] else None,
            })
            records.append(record)
            continue

        safe_count += 1
        combined = substrate + candidate
        filename = f"conformer-{grid_index:04d}.extxyz"
        structure_path = structures_directory / filename
        write(structure_path, combined)
        record.update({
            "status": "passed_cpu_collision_screen",
            "safe_rank": safe_count,
            "structure": str(Path("structures") / filename),
            "structure_sha256": _sha256(structure_path),
        })
        records.append(record)

    summary = {
        "generated_grid_count": len(records_in),
        "generation_failed_count": generation_failed,
        "intramolecular_collision_rejected_count": intramolecular_rejected,
        "substrate_collision_rejected_count": substrate_rejected,
        "organic_height_floor_rejected_count": organic_height_rejected,
        "full_substrate_collision_checked_count": full_substrate_checked,
        "full_substrate_collision_rejected_count": full_substrate_rejected,
        "collision_safe_count": safe_count,
        "optimization_started": False,
        "optimized_count": 0,
        "geometry_cluster_count": None,
        "selected_count": None,
    }
    manifest = {
        "schema_version": 1,
        "run_identity": identity,
        "status": "passed_cpu_screen_pending_formal_optimization",
        "parameters": parameter_record,
        "system": prepared["metadata"],
        "initial_fixed_site_collision_audit": prepared["initial_collision_audit"],
        "summary": summary,
        "records": records,
    }
    manifest_path = run_directory / "01_cpu_screen" / "manifest.json"
    write_manifest(manifest_path, manifest)
    print(json.dumps({
        "run_directory": str(run_directory),
        "manifest": str(manifest_path),
        "summary": summary,
    }, indent=2), flush=True)
    return 0


def run_fixed_site_ff_audit(args):
    """Reattach RDKit-UFF representatives and reuse the periodic CPU audit."""

    from sam_lammps import write_manifest
    from sam_structure_tools import periodic_vdw_collision_audit

    ff_manifest_path = Path(args.ff_manifest).expanduser().resolve()
    if not ff_manifest_path.is_file():
        raise ValueError(f"fast-FF manifest does not exist: {ff_manifest_path}")
    ff_manifest = json.loads(ff_manifest_path.read_text())
    if ff_manifest.get("status") != "passed_ff_pending_substrate_audit":
        raise ValueError(
            "fast-FF manifest is not a completed isolated optimization stage"
        )
    representatives = ff_manifest.get("representatives") or []
    if not representatives:
        raise ValueError("fast-FF manifest contains no cluster representatives")

    prepared = _prepare_fixed_site_system(args)
    substrate = prepared["substrate_with_h"]
    reference_sam = prepared["sam"]
    sam_count = len(reference_sam)
    head_ids = [int(index) for index in prepared["headgroup_atom_ids_0based"]]
    output_root = Path(
        args.audit_output_root or ff_manifest_path.parent
    ).expanduser().resolve()
    identity_payload = {
        "ff_manifest": str(ff_manifest_path),
        "ff_manifest_sha256": _sha256(ff_manifest_path),
        "script": str(Path(__file__).resolve()),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "site_prototype_id": args.site_prototype_id,
        "site_cell": int(args.site_cell),
        "vdw_radius_scale": float(args.vdw_radius_scale),
        "representatives": [
            {
                "cluster_index": int(item["cluster_index"]),
                "representative_structure": item["representative_structure"],
            }
            for item in representatives
        ],
    }
    identity = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    run_directory = output_root / f"fixed-site-ff-audit-{identity[:12]}"
    if run_directory.exists():
        raise ValueError(f"immutable FF audit directory already exists: {run_directory}")
    accepted_directory = run_directory / "accepted_structures"
    accepted_directory.mkdir(parents=True)

    audit_records = []
    accepted_count = 0
    for item in representatives:
        cluster_index = int(item["cluster_index"])
        source_path = Path(item["representative_structure"]).expanduser().resolve()
        if not source_path.is_file():
            raise ValueError(f"representative structure is missing: {source_path}")
        sam = read(source_path)
        if len(sam) != sam_count or sam.get_chemical_symbols() != reference_sam.get_chemical_symbols():
            raise ValueError(
                f"representative SAM layout does not match fixed Site Instance: {source_path}"
            )
        head_displacement = float(
            np.max(np.linalg.norm(sam.positions[head_ids] - reference_sam.positions[head_ids], axis=1))
        )
        if head_displacement > 1.0e-6:
            raise ValueError(
                f"fixed PO3 head moved by {head_displacement:.6g} A in cluster {cluster_index}"
            )
        audit = periodic_vdw_collision_audit(
            molecule_positions=np.asarray(sam.positions, dtype=float),
            molecule_symbols=sam.get_chemical_symbols(),
            molecule_atom_ids=list(range(1, sam_count + 1)),
            substrate_positions=np.asarray(substrate.positions, dtype=float),
            substrate_symbols=substrate.get_chemical_symbols(),
            substrate_atom_ids=list(range(1, len(substrate) + 1)),
            cell=np.asarray(substrate.cell, dtype=float),
            periodic_axes=prepared["surface_frame"]["periodic_fractional_axes"],
            surface_frame=prepared["surface_frame"],
            radii_A=prepared["radii_A"],
            radius_scale=float(args.vdw_radius_scale),
            mapped_bond_windows_A=prepared["mapped_windows"],
        )
        record = {
            "cluster_index": cluster_index,
            "representative_grid_index": int(item["representative_grid_index"]),
            "representative_structure": str(source_path),
            "representative_structure_sha256": _sha256(source_path),
            "head_max_displacement_A": head_displacement,
            "status": "passed" if audit["passed"] else "rejected",
            "collision_count": int(audit["collision_count"]),
            "mapped_bond_violation_count": int(audit["mapped_bond_violation_count"]),
            "first_collision": audit["collisions"][0] if audit["collisions"] else None,
        }
        if audit["passed"]:
            combined = substrate.copy()
            combined += sam
            combined.set_cell(substrate.cell)
            combined.set_pbc(substrate.pbc)
            accepted_path = accepted_directory / f"cluster-{cluster_index:05d}.extxyz"
            write(accepted_path, combined, format="extxyz")
            record["accepted_structure"] = str(accepted_path)
            record["accepted_structure_sha256"] = _sha256(accepted_path)
            accepted_count += 1
        audit_records.append(record)

    manifest = {
        "schema_version": 1,
        "run_identity": identity,
        "status": "passed",
        "stage": "fixed_site_ff_representative_substrate_collision_audit",
        "sources": {
            "fast_ff_manifest": str(ff_manifest_path),
            "fast_ff_manifest_sha256": _sha256(ff_manifest_path),
            "sam_h0": str(Path(args.sam).expanduser().resolve()),
            "substrate_catalog": str(Path(args.substrate_catalog).expanduser().resolve())
            if getattr(args, "substrate_catalog", None)
            else None,
            "script": str(Path(__file__).resolve()),
            "script_sha256": _sha256(Path(__file__).resolve()),
        },
        "parameters": {
            "collision_audit": "maintained periodic_vdw_collision_audit",
            "vdw_radius_scale": float(args.vdw_radius_scale),
            "fixed_site_prototype_id": args.site_prototype_id,
            "site_cell": int(args.site_cell),
            "representative_count": len(representatives),
        },
        "summary": {
            "representatives_checked": len(representatives),
            "collision_safe_count": accepted_count,
            "collision_rejected_count": len(representatives) - accepted_count,
        },
        "records": audit_records,
    }
    manifest_path = run_directory / "audit-manifest.json"
    write_manifest(manifest_path, manifest)
    print(
        json.dumps(
            {
                "stage": "fixed-site FF representative substrate collision audit",
                "status": "passed",
                "run_directory": str(run_directory),
                "manifest": str(manifest_path),
                "summary": manifest["summary"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


def _composition(atoms):
    return {
        symbol: int(sum(atom.symbol == symbol for atom in atoms))
        for symbol in sorted(set(atoms.get_chemical_symbols()))
    }


def resolve_fixed_site_layer_constraints(
    atoms,
    substrate_atom_count=480,
    structural_layer_count=3,
    movable_layers_from_surface=1,
    minimum_layer_gap_A=0.7,
    outward_normal=(0.0, 0.0, 1.0),
):
    """Resolve complete doped ITO layers without assuming equal layer formulae."""

    substrate_atom_count = int(substrate_atom_count)
    structural_layer_count = int(structural_layer_count)
    movable_layers_from_surface = int(movable_layers_from_surface)
    if substrate_atom_count <= 0 or substrate_atom_count > len(atoms):
        raise ValueError("invalid fixed-site substrate atom count")
    if structural_layer_count <= 1 or not 0 < movable_layers_from_surface < structural_layer_count:
        raise ValueError("invalid fixed-site structural-layer contract")
    if substrate_atom_count % structural_layer_count:
        raise ValueError("substrate atoms cannot be partitioned into complete equal-size layers")
    normal = np.asarray(outward_normal, dtype=float)
    normal_norm = float(np.linalg.norm(normal))
    if not np.isfinite(normal_norm) or normal_norm == 0.0:
        raise ValueError("outward normal must be finite and nonzero")
    normal /= normal_norm
    projections = np.asarray(atoms.positions[:substrate_atom_count]) @ normal
    order = np.argsort(projections, kind="stable")
    layer_size = substrate_atom_count // structural_layer_count
    layers = [
        np.sort(order[index * layer_size:(index + 1) * layer_size])
        for index in range(structural_layer_count)
    ]
    gaps = [
        float(np.min(projections[layers[index + 1]]) - np.max(projections[layers[index]]))
        for index in range(structural_layer_count - 1)
    ]
    if any(gap < float(minimum_layer_gap_A) for gap in gaps):
        raise ValueError(
            f"fixed-site structural-layer boundary gap below {minimum_layer_gap_A} A: {gaps}"
        )
    frozen_layers = layers[: structural_layer_count - movable_layers_from_surface]
    frozen = np.sort(np.concatenate(frozen_layers)).astype(int)
    movable_substrate = np.sort(
        np.concatenate(layers[structural_layer_count - movable_layers_from_surface:])
    ).astype(int)
    return {
        "method": "equal_atom_count_blocks_along_outward_normal_allow_dopant_formula_variation",
        "outward_normal": normal.tolist(),
        "structural_layer_count": structural_layer_count,
        "atoms_per_structural_layer": layer_size,
        "minimum_layer_boundary_gap_A": float(minimum_layer_gap_A),
        "observed_layer_boundary_gaps_A": gaps,
        "layers_bottom_to_top": [
            {
                "atom_ids_1based": (layer + 1).tolist(),
                "composition": _composition(atoms[layer]),
                "projection_range_A": [
                    float(np.min(projections[layer])),
                    float(np.max(projections[layer])),
                ],
            }
            for layer in layers
        ],
        "frozen_atom_ids_1based": (frozen + 1).tolist(),
        "movable_substrate_atom_ids_1based": (movable_substrate + 1).tolist(),
    }


def _sam_edge_set(atoms, sam_indices, scale=1.25):
    edges = []
    indices = list(map(int, sam_indices))
    for left_offset, left in enumerate(indices):
        for right in indices[left_offset + 1:]:
            cutoff = float(scale) * (
                covalent_radii[atoms[left].number] + covalent_radii[atoms[right].number]
            )
            if atoms.get_distance(left, right, mic=True) <= cutoff:
                edges.append([left + 1, right + 1])
    return edges


def _resolve_fixed_site_anchor_mapping(atoms, cpu_manifest, substrate_atom_count, sam_start):
    metals = [
        index for index in range(substrate_atom_count)
        if atoms[index].symbol in ("In", "Sn")
    ]
    mapping = []
    used_metals = set()
    for label, local_id in sorted(cpu_manifest["system"]["donor_assignment_1based"].items()):
        oxygen = sam_start + int(local_id) - 1
        distances = np.asarray(
            [atoms.get_distance(oxygen, metal, mic=True) for metal in metals],
            dtype=float,
        )
        registered = cpu_manifest['system'].get('registered_donor_metal_ids_1based')
        nearest = int(registered[label])-1 if registered else metals[int(np.argmin(distances))]
        if nearest not in metals:
            raise ValueError('Registered fixed-site metal is outside substrate')
        distance = float(atoms.get_distance(oxygen, nearest, mic=True))
        if atoms[oxygen].symbol != "O" or not distance_within_closed_window(
            distance,
            1.7,
            2.8,
        ):
            raise ValueError(f"initial registered {label}-metal contact is invalid: {distance} A")
        if nearest in used_metals:
            raise ValueError("initial tridentate donor mapping does not use three distinct metals")
        used_metals.add(nearest)
        mapping.append(
            {
                "donor_label": label,
                "sam_atom_id_1based": int(local_id),
                "global_oxygen_atom_id_1based": oxygen + 1,
                "global_metal_atom_id_1based": nearest + 1,
                "metal_element": atoms[nearest].symbol,
                "initial_distance_A": distance,
                "allowed_distance_A": [1.7, 2.8],
            }
        )
    return mapping


def _write_fixed_site_sbatch(path, plan_path, plan_sha256, task_expression, job_name):
    script_path = Path(__file__).resolve()
    python_path = Path("/home/software/anaconda3/envs/ase/bin/python")
    quoted = lambda value: shlex.quote(str(value))
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        "#SBATCH --partition=all",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=8",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --mem=32G",
        "#SBATCH --time=12:00:00",
        f"#SBATCH --array={task_expression}",
        f"#SBATCH --output={quoted(path.parent / 'logs' / '%x-%A_%a.out')}",
        f"#SBATCH --error={quoted(path.parent / 'logs' / '%x-%A_%a.err')}",
        "set -euo pipefail",
        "export OMP_NUM_THREADS=1",
        "export MKL_NUM_THREADS=1",
        "export PYTORCH_ALLOC_CONF=expandable_segments:True",
        (
            f"{quoted(python_path)} -u {quoted(script_path)} "
            f"--mode fixed-site-optimize-worker --plan {quoted(plan_path)} "
            f"--plan-sha256 {quoted(plan_sha256)} --task-index \"${{SLURM_ARRAY_TASK_ID}}\""
        ),
        "",
    ]
    path.write_text("\n".join(lines))


def run_fixed_site_optimization_plan(args):
    """Freeze the approved formal ensemble and create one immutable GPU run package."""

    from sam_lammps import write_manifest

    cpu_manifest_path = Path(args.cpu_manifest).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    script_path = Path(__file__).resolve()
    for source in (cpu_manifest_path, model_path, script_path):
        if not source.is_file():
            raise ValueError(f"required formal source is missing: {source}")
    cpu_manifest = json.loads(cpu_manifest_path.read_text())
    if cpu_manifest.get("schema") == "sam-phosphonate-skeleton-cpu-v1":
        raise ValueError("isolated skeletons require fast optimization/dedup and an ITO-site audit before a MACE plan")
    passed = sorted(
        (
            record for record in cpu_manifest["records"]
            if record.get("status") == "passed_cpu_collision_screen"
        ),
        key=lambda record: int(record["safe_rank"]),
    )
    expected = int(cpu_manifest["summary"]["collision_safe_count"])
    generalized = bool(getattr(args, "generalized", False))
    if len(passed) != expected or expected < 1 or (not generalized and expected != 81):
        raise ValueError(f"Invalid collision-safe ensemble count: {len(passed)} (declared {expected})")
    layout = cpu_manifest['system']
    substrate_atom_count = int(layout['substrate_atom_count']) if generalized else 480
    nh = int(layout['surface_h_count']) if generalized else 2
    sam_count = int(layout['sam_atom_count']) if generalized else 46
    surface_h_ids = list(range(substrate_atom_count+1, substrate_atom_count+nh+1))
    sam_start = substrate_atom_count+nh
    constraint_kwargs = dict(substrate_atom_count=substrate_atom_count,
        structural_layer_count=int(layout.get('structural_layer_count',3)),
        outward_normal=layout.get('outward_normal',(0,0,1)))
    inputs = []
    reference = None
    reference_symbols = None
    reference_cell = None
    constraint_record = None
    for task_index, record in enumerate(passed, 1):
        structure = (cpu_manifest_path.parent / record["structure"]).resolve()
        if _sha256(structure) != record["structure_sha256"]:
            raise ValueError(f"CPU-screen structure hash changed: {structure}")
        atoms = read(structure)
        if len(atoms) != sam_start+sam_count:
            raise ValueError(f"fixed-site input layout mismatch: {structure}")
        if reference is None:
            reference = atoms
            reference_symbols = atoms.get_chemical_symbols()
            reference_cell = np.asarray(atoms.cell, dtype=float)
            constraint_record = resolve_fixed_site_layer_constraints(atoms, **constraint_kwargs)
        else:
            if atoms.get_chemical_symbols() != reference_symbols:
                raise ValueError(f"formal ensemble composition/order mismatch: {structure}")
            if not np.allclose(atoms.cell, reference_cell, atol=1.0e-10, rtol=0.0):
                raise ValueError(f"formal ensemble cell mismatch: {structure}")
            if resolve_fixed_site_layer_constraints(atoms, **constraint_kwargs) != constraint_record:
                raise ValueError(f"formal ensemble frozen-layer identity mismatch: {structure}")
        inputs.append(
            {
                "task_index": task_index,
                "safe_rank": int(record["safe_rank"]),
                "grid_index": int(record["grid_index"]),
                "start_dihedrals_deg": list(record["start_dihedrals_deg"]),
                "structure": str(structure),
                "structure_sha256": record["structure_sha256"],
            }
        )
    sam_indices = list(range(sam_start, len(reference)))
    parent_ids = [
        int(item["target_parent_o_atom_id_1based"])
        for item in cpu_manifest["system"]["surface_protons"]
    ]
    anchor_mapping = _resolve_fixed_site_anchor_mapping(
        reference, cpu_manifest, substrate_atom_count, sam_start
    )
    parameters = {
        "calculator": "MACECalculator",
        "device": "cuda",
        "default_dtype": "float32",
        "optimizer": "ASE.LBFGS",
        "fmax_eV_per_A": float(args.fmax),
        "max_steps": int(args.max_steps),
        "maxstep_A": float(args.maxstep),
        "relax_cell": False,
        "substrate_atom_count": substrate_atom_count,
        "surface_h_atom_ids_1based": surface_h_ids,
        "sam_atom_ids_1based": [index + 1 for index in sam_indices],
        "energy_quantity": "relative_total_energy_same_composition_not_adsorption_energy",
        "validation": {
            "metal_oxygen_distance_A": [1.7, 2.8],
            "surface_h_substrate_o_max_distance_A": 1.25,
            "movable_substrate_max_displacement_A": 1.5,
            "frozen_max_displacement_A": 1.0e-8,
            "cell_max_abs_change_A": 1.0e-8,
            "sam_connectivity_covalent_radius_scale": 1.25,
        },
    }
    if parameters["fmax_eV_per_A"] <= 0 or parameters["max_steps"] <= 0 or parameters["maxstep_A"] <= 0:
        raise ValueError("formal optimizer thresholds must be positive")
    identity_payload = {
        "cpu_manifest_sha256": _sha256(cpu_manifest_path),
        "model_sha256": _sha256(model_path),
        "script_sha256": _sha256(script_path),
        "parameters": parameters,
        "constraints": constraint_record,
        "inputs": [{key: value for key, value in item.items() if key != "structure"} for item in inputs],
    }
    identity = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    run_directory = Path(args.formal_output_root).expanduser().resolve() / f"fixed-site-mace-{identity[:12]}"
    if run_directory.exists():
        raise ValueError(f"immutable formal run directory already exists: {run_directory}")
    (run_directory / "02_mace_optimization" / "tasks").mkdir(parents=True)
    (run_directory / "logs").mkdir()
    plan = {
        "schema_version": 1,
        "run_identity": identity,
        "status": "planned_pending_gpu_acceptance",
        "run_directory": str(run_directory),
        "sources": {
            "cpu_manifest": str(cpu_manifest_path),
            "cpu_manifest_sha256": _sha256(cpu_manifest_path),
            "model": str(model_path),
            "model_sha256": _sha256(model_path),
            "script": str(script_path),
            "script_sha256": _sha256(script_path),
        },
        "system": {
            "atom_count": len(reference),
            "composition": _composition(reference),
            "cell_A": reference_cell.tolist(),
            "pbc": np.asarray(reference.pbc, dtype=bool).tolist(),
            "substrate_atom_count": substrate_atom_count,
            "surface_h_atom_ids_1based": surface_h_ids,
            "surface_h_parent_o_atom_ids_1based": parent_ids,
            "sam_atom_ids_1based": [index + 1 for index in sam_indices],
            "initial_sam_covalent_edges_global_1based": _sam_edge_set(reference, sam_indices),
            "anchor_mapping": anchor_mapping,
        },
        "constraints": constraint_record,
        "parameters": parameters,
        "tasks": inputs,
    }
    plan_path = run_directory / "plan.json"
    write_manifest(plan_path, plan)
    plan_sha256 = _sha256(plan_path)
    acceptance_path = run_directory / "acceptance.sbatch"
    production_path = run_directory / "production.sbatch"
    _write_fixed_site_sbatch(acceptance_path, plan_path, plan_sha256, "1", "sam-mace-accept")
    _write_fixed_site_sbatch(production_path, plan_path, plan_sha256, f"2-{len(inputs)}%1" if len(inputs)>1 else "1", "sam-mace-prod")
    manifest = {
        "schema_version": 1,
        "run_identity": identity,
        "status": "planned_pending_gpu_acceptance",
        "plan": {"path": str(plan_path), "sha256": plan_sha256},
        "cpu_screen_count": expected,
        "optimization_completed_count": 0,
        "optimization_valid_count": 0,
        "submission_artifacts": {
            "acceptance": {"path": str(acceptance_path), "sha256": _sha256(acceptance_path)},
            "production": {"path": str(production_path), "sha256": _sha256(production_path)},
        },
    }
    write_manifest(run_directory / "manifest.json", manifest)
    print(json.dumps({"run_directory": str(run_directory), "plan": str(plan_path), "plan_sha256": plan_sha256, "tasks": len(inputs)}, indent=2), flush=True)
    return 0


def _maximum_displacement_A(initial, final, indices):
    maximum = 0.0
    for index in indices:
        vector = np.asarray(final.positions[index]) - np.asarray(initial.positions[index])
        vector, _ = find_mic(vector, initial.cell, initial.pbc)
        maximum = max(maximum, float(np.linalg.norm(vector)))
    return maximum


def _validate_fixed_site_optimization(initial, final, forces, plan):
    parameters = plan["parameters"]
    system = plan["system"]
    frozen = [value - 1 for value in plan["constraints"]["frozen_atom_ids_1based"]]
    movable_substrate = [
        value - 1 for value in plan["constraints"]["movable_substrate_atom_ids_1based"]
    ]
    movable = sorted(set(range(len(final))) - set(frozen))
    checks = {}
    checks["atom_count_unchanged"] = len(final) == system["atom_count"] == len(initial)
    checks["composition_and_order_unchanged"] = final.get_chemical_symbols() == initial.get_chemical_symbols()
    cell_change = float(np.max(np.abs(np.asarray(final.cell) - np.asarray(initial.cell))))
    checks["cell_unchanged"] = cell_change <= parameters["validation"]["cell_max_abs_change_A"]
    final_edges = _sam_edge_set(
        final,
        [value - 1 for value in system["sam_atom_ids_1based"]],
        parameters["validation"]["sam_connectivity_covalent_radius_scale"],
    )
    checks["sam_connectivity_unchanged"] = final_edges == system["initial_sam_covalent_edges_global_1based"]
    anchor_distances = []
    for mapping in system["anchor_mapping"]:
        oxygen = mapping["global_oxygen_atom_id_1based"] - 1
        metal = mapping["global_metal_atom_id_1based"] - 1
        distance = float(final.get_distance(oxygen, metal, mic=True))
        low, high = mapping["allowed_distance_A"]
        anchor_distances.append(
            {
                **mapping,
                "final_distance_A": distance,
                "passed": distance_within_closed_window(distance, low, high),
            }
        )
    checks["three_registered_anchor_contacts_preserved"] = (
        len(anchor_distances) == 3
        and len({item["global_metal_atom_id_1based"] for item in anchor_distances}) == 3
        and all(item["passed"] for item in anchor_distances)
    )
    substrate_oxygen = [
        index for index in range(parameters["substrate_atom_count"])
        if final[index].symbol == "O"
    ]
    proton_binding = []
    for hydrogen_id, parent_id in zip(
        system["surface_h_atom_ids_1based"],
        system["surface_h_parent_o_atom_ids_1based"],
    ):
        hydrogen = hydrogen_id - 1
        distances = np.asarray(
            [final.get_distance(hydrogen, oxygen, mic=True) for oxygen in substrate_oxygen]
        )
        nearest_offset = int(np.argmin(distances))
        nearest = substrate_oxygen[nearest_offset]
        proton_binding.append(
            {
                "hydrogen_atom_id_1based": hydrogen_id,
                "initial_parent_o_atom_id_1based": parent_id,
                "nearest_final_o_atom_id_1based": nearest + 1,
                "nearest_final_o_distance_A": float(distances[nearest_offset]),
                "initial_parent_final_distance_A": float(final.get_distance(hydrogen, parent_id - 1, mic=True)),
            }
        )
    checks["surface_protons_bound_to_substrate_oxygen"] = all(
        item["nearest_final_o_distance_A"] <= parameters["validation"]["surface_h_substrate_o_max_distance_A"]
        for item in proton_binding
    )
    frozen_displacement = _maximum_displacement_A(initial, final, frozen)
    movable_substrate_displacement = _maximum_displacement_A(initial, final, movable_substrate)
    max_movable_force = float(np.max(np.linalg.norm(np.asarray(forces)[movable], axis=1)))
    checks["frozen_atoms_unchanged"] = frozen_displacement <= parameters["validation"]["frozen_max_displacement_A"]
    checks["movable_substrate_displacement_within_contract"] = movable_substrate_displacement <= parameters["validation"]["movable_substrate_max_displacement_A"]
    checks["movable_force_converged"] = max_movable_force <= parameters["fmax_eV_per_A"]
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "cell_max_abs_change_A": cell_change,
        "frozen_max_displacement_A": frozen_displacement,
        "movable_substrate_max_displacement_A": movable_substrate_displacement,
        "movable_max_force_eV_per_A": max_movable_force,
        "anchor_contacts": anchor_distances,
        "surface_proton_binding": proton_binding,
        "final_sam_covalent_edges_global_1based": final_edges,
    }


def run_fixed_site_optimization_worker(args):
    """Run one hash-locked GPU optimization task without fallback or replay."""

    from sam_lammps import write_manifest

    plan_path = Path(args.plan).expanduser().resolve()
    if _sha256(plan_path) != args.plan_sha256:
        raise ValueError("formal optimization plan hash changed")
    plan = json.loads(plan_path.read_text())
    if _sha256(plan["sources"]["script"]) != plan["sources"]["script_sha256"]:
        raise ValueError("formal worker script hash changed after planning")
    if _sha256(plan["sources"]["model"]) != plan["sources"]["model_sha256"]:
        raise ValueError("formal MACE model hash changed after planning")
    task_index = int(args.task_index)
    if not 1 <= task_index <= len(plan["tasks"]):
        raise ValueError(f"task index outside formal ensemble: {task_index}")
    task = plan["tasks"][task_index - 1]
    if task["task_index"] != task_index:
        raise ValueError("formal plan task order is inconsistent")
    task_directory = Path(plan["run_directory"]) / "02_mace_optimization" / "tasks" / f"task-{task_index:04d}"
    if task_directory.exists():
        raise ValueError(f"immutable optimization task already exists: {task_directory}")
    task_directory.mkdir(parents=True)
    manifest_path = task_directory / "manifest.json"
    try:
        structure_path = Path(task["structure"])
        if _sha256(structure_path) != task["structure_sha256"]:
            raise ValueError("formal task input structure hash changed")
        initial = read(structure_path)
        if len(initial) != plan["system"]["atom_count"] or _composition(initial) != plan["system"]["composition"]:
            raise ValueError("formal task input identity no longer matches plan")
        if not np.allclose(initial.cell, plan["system"]["cell_A"], atol=1.0e-10, rtol=0.0):
            raise ValueError("formal task input cell no longer matches plan")
        _load_torch_for_mace_worker(
            "formal fixed-site MACE worker",
            require_cuda=True,
            cuda_failure_message="CUDA is unavailable; formal fixed-site MACE run forbids CPU fallback",
        )
        from mace.calculators import MACECalculator
        from ase.optimize import LBFGS

        atoms = initial.copy()
        frozen = [value - 1 for value in plan["constraints"]["frozen_atom_ids_1based"]]
        movable = sorted(set(range(len(atoms))) - set(frozen))
        atoms.set_constraint(FixAtoms(indices=frozen))
        atoms.calc = MACECalculator(
            model_paths=plan["sources"]["model"],
            device="cuda",
            default_dtype="float32",
        )
        progress_path = task_directory / "progress.jsonl"
        optimizer = LBFGS(
            atoms,
            trajectory=str(task_directory / "optimization.traj"),
            logfile=str(task_directory / "optimizer.log"),
            maxstep=plan["parameters"]["maxstep_A"],
        )

        def record_progress():
            forces = atoms.get_forces()
            payload = {
                "step": int(optimizer.nsteps),
                "total_energy_eV": float(atoms.get_potential_energy()),
                "movable_max_force_eV_per_A": float(
                    np.max(np.linalg.norm(np.asarray(forces)[movable], axis=1))
                ),
            }
            with progress_path.open("a") as handle:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
            print(json.dumps({"task_index": task_index, **payload}), flush=True)

        record_progress()
        optimizer.attach(record_progress, interval=1)
        optimizer.run(
            fmax=plan["parameters"]["fmax_eV_per_A"],
            steps=plan["parameters"]["max_steps"],
        )
        final_energy = float(atoms.get_potential_energy())
        final_forces = atoms.get_forces()
        validation = _validate_fixed_site_optimization(initial, atoms, final_forces, plan)
        relaxed = atoms.copy()
        relaxed.calc = None
        relaxed_path = task_directory / "relaxed.extxyz"
        write(relaxed_path, relaxed)
        write_manifest(task_directory / "validation.json", validation)
        optimizer_converged = bool(
            optimizer.converged(optimizer.optimizable.get_gradient())
        )
        passed = bool(optimizer_converged and validation["passed"])
        manifest = {
            "schema_version": 1,
            "status": "passed" if passed else "failed_validation_or_convergence",
            "task_index": task_index,
            "safe_rank": task["safe_rank"],
            "grid_index": task["grid_index"],
            "start_dihedrals_deg": task["start_dihedrals_deg"],
            "optimizer_steps": int(optimizer.nsteps),
            "optimizer_converged": optimizer_converged,
            "total_energy_eV": final_energy,
            "energy_role": "same_ensemble_total_energy_for_later_relative_ranking_not_adsorption_energy",
            "input": {"path": str(structure_path), "sha256": task["structure_sha256"]},
            "relaxed": {"path": str(relaxed_path), "sha256": _sha256(relaxed_path)},
            "validation": {"path": str(task_directory / "validation.json"), "sha256": _sha256(task_directory / "validation.json"), "passed": validation["passed"]},
            "progress": {"path": str(progress_path), "sha256": _sha256(progress_path)},
            "optimizer_log": {"path": str(task_directory / "optimizer.log"), "sha256": _sha256(task_directory / "optimizer.log")},
            "trajectory": {"path": str(task_directory / "optimization.traj"), "sha256": _sha256(task_directory / "optimization.traj")},
        }
        write_manifest(manifest_path, manifest)
        print(json.dumps(manifest, indent=2), flush=True)
        return 0 if passed else 2
    except Exception as error:
        failure = {
            "schema_version": 1,
            "status": "failed",
            "task_index": task_index,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        write_manifest(manifest_path, failure)
        print(json.dumps(failure, indent=2), flush=True)
        return 2


def _fixed_site_surface_frame(plan):
    cell = np.asarray(plan["system"]["cell_A"], dtype=float)
    normal = np.asarray(plan["constraints"]["outward_normal"], dtype=float)
    normal /= np.linalg.norm(normal)
    u_axis = cell[0] / np.linalg.norm(cell[0])
    v_axis = np.cross(normal, u_axis)
    v_axis /= np.linalg.norm(v_axis)
    expected_v = cell[1] / np.linalg.norm(cell[1])
    if float(np.dot(v_axis, expected_v)) < 1.0 - 1.0e-8:
        raise ValueError("fixed-site cell axes do not reproduce the registered Surface Frame")
    return {
        "u_cartesian_unit": u_axis.tolist(),
        "v_cartesian_unit": v_axis.tolist(),
        "outward_normal_cartesian_unit": normal.tolist(),
        "periodic_fractional_axes": [0, 1],
    }


def canonical_fixed_site_sam_coordinates(atoms, plan):
    """Unwrap SAM bonds and choose one periodic image in the fixed surface frame."""

    from substrate_library import _unwrap_molecule
    from sam_structure_tools import surface_frame_coordinates

    sam_indices = [value - 1 for value in plan["system"]["sam_atom_ids_1based"]]
    global_to_local = {global_index: local for local, global_index in enumerate(sam_indices)}
    adjacency = [set() for _ in sam_indices]
    for left_id, right_id in plan["system"]["initial_sam_covalent_edges_global_1based"]:
        left = global_to_local[left_id - 1]
        right = global_to_local[right_id - 1]
        adjacency[left].add(right)
        adjacency[right].add(left)
    symbols = [atoms[index].symbol for index in sam_indices]
    phosphorus = [index for index, symbol in enumerate(symbols) if symbol == "P"]
    if len(phosphorus) != 1:
        raise ValueError("fixed-site SAM must contain exactly one phosphorus anchor")
    cell = np.asarray(atoms.cell, dtype=float)
    unwrapped = _unwrap_molecule(
        np.asarray(atoms.positions[sam_indices], dtype=float),
        adjacency,
        cell,
        np.asarray(atoms.pbc, dtype=bool),
        phosphorus[0],
    )
    anchor_fractional = unwrapped[phosphorus[0]] @ np.linalg.inv(cell)
    image = np.zeros(3, dtype=float)
    image[0] = np.floor(anchor_fractional[0])
    image[1] = np.floor(anchor_fractional[1])
    unwrapped -= image @ cell
    frame = _fixed_site_surface_frame(plan)
    return {
        "cartesian_A": unwrapped,
        "surface_frame_A": surface_frame_coordinates(unwrapped, frame),
        "symbols": symbols,
        "anchor_local_index_0based": phosphorus[0],
        "periodic_image_removed": image.astype(int).tolist(),
        "surface_frame": frame,
    }


def _normalized_dihedral_deg(atoms, atom_ids_1based):
    value = float(atoms.get_dihedral(*[value - 1 for value in atom_ids_1based]))
    return float((value + 180.0) % 360.0 - 180.0)


def run_fixed_site_analysis(args):
    """Aggregate, cluster, footprint-rank, and select the validated ensemble."""

    from sam_lammps import write_manifest
    from sam_structure_tools import filled_outer_envelope_area

    plan_path = Path(args.plan).expanduser().resolve()
    if _sha256(plan_path) != args.plan_sha256:
        raise ValueError("formal optimization plan hash changed before analysis")
    plan = json.loads(plan_path.read_text())
    run_directory = Path(plan["run_directory"])
    analysis_directory = run_directory / "03_analysis"
    if analysis_directory.exists():
        raise ValueError(f"immutable analysis directory already exists: {analysis_directory}")
    analysis_directory.mkdir(parents=True)
    write_manifest(
        analysis_directory / "manifest.json",
        {
            "schema_version": 1,
            "status": "running",
            "plan": {"path": str(plan_path), "sha256": args.plan_sha256},
            "analysis_script": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__).resolve())},
        },
    )
    try:
        cpu_manifest_path = Path(plan["sources"]["cpu_manifest"])
        if _sha256(cpu_manifest_path) != plan["sources"]["cpu_manifest_sha256"]:
            raise ValueError("CPU-screen manifest hash changed before analysis")
        cpu_manifest = json.loads(cpu_manifest_path.read_text())
        torsions = cpu_manifest["parameters"]["torsion_atom_ids_1based"]
        task_status_counts = {}
        excluded = []
        candidates = []
        reference_symbols = None
        reference_cell = np.asarray(plan["system"]["cell_A"], dtype=float)
        sam_indices = [value - 1 for value in plan["system"]["sam_atom_ids_1based"]]
        for task in plan["tasks"]:
            task_index = int(task["task_index"])
            task_directory = run_directory / "02_mace_optimization" / "tasks" / f"task-{task_index:04d}"
            task_manifest_path = task_directory / "manifest.json"
            if not task_manifest_path.is_file():
                task_status_counts["missing_manifest"] = task_status_counts.get("missing_manifest", 0) + 1
                excluded.append({"task_index": task_index, "reason": "missing_manifest"})
                continue
            task_manifest = json.loads(task_manifest_path.read_text())
            status = str(task_manifest.get("status"))
            task_status_counts[status] = task_status_counts.get(status, 0) + 1
            if status != "passed" or not task_manifest.get("optimizer_converged"):
                excluded.append({"task_index": task_index, "reason": status})
                continue
            if task_manifest.get("task_index") != task_index:
                raise ValueError(f"task manifest identity mismatch: {task_manifest_path}")
            validation_path = Path(task_manifest["validation"]["path"])
            relaxed_path = Path(task_manifest["relaxed"]["path"])
            if _sha256(validation_path) != task_manifest["validation"]["sha256"]:
                raise ValueError(f"task validation hash changed: {validation_path}")
            if _sha256(relaxed_path) != task_manifest["relaxed"]["sha256"]:
                raise ValueError(f"task relaxed structure hash changed: {relaxed_path}")
            validation = json.loads(validation_path.read_text())
            if not task_manifest["validation"]["passed"] or not validation["passed"] or not all(validation["checks"].values()):
                excluded.append({"task_index": task_index, "reason": "independent_validation_failed"})
                continue
            atoms = read(relaxed_path)
            if len(atoms) != plan["system"]["atom_count"] or _composition(atoms) != plan["system"]["composition"]:
                raise ValueError(f"relaxed task identity mismatch: {relaxed_path}")
            if not np.allclose(atoms.cell, reference_cell, atol=1.0e-8, rtol=0.0):
                raise ValueError(f"relaxed task cell mismatch: {relaxed_path}")
            if reference_symbols is None:
                reference_symbols = atoms.get_chemical_symbols()
            elif atoms.get_chemical_symbols() != reference_symbols:
                raise ValueError(f"relaxed task atom order mismatch: {relaxed_path}")
            canonical = canonical_fixed_site_sam_coordinates(atoms, plan)
            sam_atoms = Atoms(
                symbols=canonical["symbols"],
                positions=canonical["cartesian_A"],
                cell=reference_cell,
                pbc=False,
            )
            final_torsions = [
                _normalized_dihedral_deg(sam_atoms, torsion) for torsion in torsions
            ]
            candidates.append(
                {
                    "task_index": task_index,
                    "safe_rank": int(task["safe_rank"]),
                    "grid_index": int(task["grid_index"]),
                    "start_dihedrals_deg": list(task["start_dihedrals_deg"]),
                    "final_dihedrals_deg": final_torsions,
                    "total_energy_eV": float(task_manifest["total_energy_eV"]),
                    "optimizer_steps": int(task_manifest["optimizer_steps"]),
                    "movable_max_force_eV_per_A": float(validation["movable_max_force_eV_per_A"]),
                    "relaxed_path": str(relaxed_path),
                    "relaxed_sha256": task_manifest["relaxed"]["sha256"],
                    "task_manifest_path": str(task_manifest_path),
                    "task_manifest_sha256": _sha256(task_manifest_path),
                    "canonical_sam_cartesian_A": canonical["cartesian_A"],
                    "canonical_sam_surface_frame_A": canonical["surface_frame_A"],
                    "sam_symbols": canonical["symbols"],
                    "surface_frame": canonical["surface_frame"],
                }
            )
        if not candidates:
            raise ValueError("no independently validated converged structures are eligible for analysis")
        candidates.sort(key=lambda item: item["task_index"])
        minimum_energy = min(item["total_energy_eV"] for item in candidates)
        for item in candidates:
            item["relative_total_energy_eV"] = item["total_energy_eV"] - minimum_energy
        heavy_local = [
            index for index, symbol in enumerate(candidates[0]["sam_symbols"])
            if symbol != "H"
        ]
        count = len(candidates)
        rmsd = np.zeros((count, count), dtype=float)
        for left in range(count):
            left_positions = candidates[left]["canonical_sam_surface_frame_A"][heavy_local]
            for right in range(left + 1, count):
                difference = left_positions - candidates[right]["canonical_sam_surface_frame_A"][heavy_local]
                value = float(np.sqrt(np.mean(np.sum(difference * difference, axis=1))))
                rmsd[left, right] = rmsd[right, left] = value
        sensitivity = []
        clusters_by_tolerance = {}
        for tolerance in FIXED_SITE_CLUSTER_SENSITIVITY_A:
            clusters = complete_link_clusters(rmsd, tolerance)
            clusters_by_tolerance[tolerance] = clusters
            sensitivity.append(
                {
                    "tolerance_A": tolerance,
                    "cluster_count": len(clusters),
                    "cluster_sizes": [len(cluster) for cluster in clusters],
                }
            )
        formal_clusters = clusters_by_tolerance[FIXED_SITE_CLUSTER_TOLERANCE_A]
        representatives_directory = analysis_directory / "cluster_representatives"
        selected_directory = analysis_directory / "selected"
        representatives_directory.mkdir()
        selected_directory.mkdir()
        radii = {
            symbol: float(vdw_radii[atomic_numbers[symbol]])
            for symbol in sorted(set(candidates[0]["sam_symbols"]))
        }
        cluster_records = []
        representative_candidate_indices = []
        for cluster_id, member_indices in enumerate(formal_clusters, 1):
            representative_index = min(
                member_indices,
                key=lambda index: (
                    candidates[index]["relative_total_energy_eV"],
                    candidates[index]["task_index"],
                ),
            )
            representative_candidate_indices.append(representative_index)
            representative = candidates[representative_index]
            footprint = filled_outer_envelope_area(
                positions=representative["canonical_sam_cartesian_A"],
                symbols=representative["sam_symbols"],
                surface_frame=representative["surface_frame"],
                radii_A=radii,
                radius_scale=1.0,
                boundary_samples_per_atom=720,
            )
            structure_path = representatives_directory / f"cluster-{cluster_id:04d}-task-{representative['task_index']:04d}.extxyz"
            structure = read(representative["relaxed_path"])
            structure.set_constraint()
            write(structure_path, structure)
            maximum_pair_rmsd = max(
                (float(rmsd[left, right]) for left in member_indices for right in member_indices),
                default=0.0,
            )
            cluster_records.append(
                {
                    "cluster_id": cluster_id,
                    "member_count": len(member_indices),
                    "member_task_indices": [candidates[index]["task_index"] for index in member_indices],
                    "maximum_pairwise_heavy_atom_rmsd_A": maximum_pair_rmsd,
                    "representative_task_index": representative["task_index"],
                    "representative_relative_total_energy_eV": representative["relative_total_energy_eV"],
                    "representative_total_energy_eV": representative["total_energy_eV"],
                    "footprint": footprint,
                    "representative_structure": {"path": str(structure_path), "sha256": _sha256(structure_path)},
                }
            )
        cluster_areas = [record["footprint"]["area_A2"] for record in cluster_records]
        cluster_energies = [record["representative_relative_total_energy_eV"] for record in cluster_records]
        pareto_indices = pareto_front_indices(cluster_areas, cluster_energies)
        minimum_footprint_index = min(
            range(len(cluster_records)),
            key=lambda index: (cluster_areas[index], cluster_energies[index], cluster_records[index]["representative_task_index"]),
        )
        minimum_energy_index = min(
            range(len(cluster_records)),
            key=lambda index: (cluster_energies[index], cluster_areas[index], cluster_records[index]["representative_task_index"]),
        )
        selected_indices = list(dict.fromkeys([minimum_footprint_index, minimum_energy_index, *pareto_indices]))
        selection_records = []
        for selection_rank, cluster_index in enumerate(selected_indices, 1):
            cluster = cluster_records[cluster_index]
            candidate_index = representative_candidate_indices[cluster_index]
            candidate = candidates[candidate_index]
            selected_path = selected_directory / f"selection-{selection_rank:04d}-cluster-{cluster['cluster_id']:04d}-task-{candidate['task_index']:04d}.extxyz"
            structure = read(candidate["relaxed_path"])
            structure.set_constraint()
            write(selected_path, structure)
            roles = []
            if cluster_index == minimum_footprint_index:
                roles.append("minimum_footprint_endpoint")
            if cluster_index == minimum_energy_index:
                roles.append("minimum_energy_endpoint")
            if cluster_index in pareto_indices:
                roles.append("pareto_front")
            selection_records.append(
                {
                    "selection_rank": selection_rank,
                    "cluster_id": cluster["cluster_id"],
                    "representative_task_index": candidate["task_index"],
                    "roles": roles,
                    "relative_total_energy_eV": cluster["representative_relative_total_energy_eV"],
                    "footprint_area_A2": cluster["footprint"]["area_A2"],
                    "structure": {"path": str(selected_path), "sha256": _sha256(selected_path)},
                }
            )
        matrix_path = analysis_directory / "sam_heavy_atom_rmsd_A.npy"
        np.save(matrix_path, rmsd)
        candidate_csv = analysis_directory / "validated_candidates.csv"
        with candidate_csv.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "task_index", "safe_rank", "grid_index", "start_dihedrals_deg",
                "final_dihedrals_deg", "optimizer_steps", "movable_max_force_eV_per_A",
                "total_energy_eV", "relative_total_energy_eV", "relaxed_path", "relaxed_sha256",
            ])
            for item in candidates:
                writer.writerow([
                    item["task_index"], item["safe_rank"], item["grid_index"],
                    ";".join(map(str, item["start_dihedrals_deg"])),
                    ";".join(f"{value:.8f}" for value in item["final_dihedrals_deg"]),
                    item["optimizer_steps"], item["movable_max_force_eV_per_A"],
                    item["total_energy_eV"], item["relative_total_energy_eV"],
                    item["relaxed_path"], item["relaxed_sha256"],
                ])
        report = {
            "schema_version": 1,
            "status": "passed",
            "energy_quantity": "relative_total_energy_same_composition_not_adsorption_energy",
            "funnel": {
                "generated_single_molecule_conformations": cpu_manifest["summary"]["generated_grid_count"],
                "generation_failed": cpu_manifest["summary"]["generation_failed_count"],
                "intramolecular_collision_rejected": cpu_manifest["summary"]["intramolecular_collision_rejected_count"],
                "substrate_or_surface_h_collision_rejected": cpu_manifest["summary"]["substrate_collision_rejected_count"],
                "collision_safe_submitted_to_mace": cpu_manifest["summary"]["collision_safe_count"],
                "optimization_task_manifest_status_counts": task_status_counts,
                "validated_converged_for_clustering": len(candidates),
                "excluded_after_optimization": len(excluded),
                "formal_geometry_cluster_count": len(formal_clusters),
                "finally_selected_count": len(selection_records),
            },
            "excluded_tasks": excluded,
            "optimization": {
                "model": plan["sources"]["model"],
                "model_sha256": plan["sources"]["model_sha256"],
                "parameters": plan["parameters"],
                "minimum_total_energy_eV": minimum_energy,
                "maximum_relative_total_energy_eV": max(item["relative_total_energy_eV"] for item in candidates),
            },
            "geometry_clustering": {
                "atom_scope": "complete_SAM_heavy_atoms",
                "alignment": "fixed_substrate_surface_frame_with_covalent_unwrap_and_periodic_whole_cell_image_normalization_no_molecular_Kabsch",
                "method": "deterministic_complete_link",
                "formal_tolerance_A": FIXED_SITE_CLUSTER_TOLERANCE_A,
                "sensitivity": sensitivity,
                "distance_matrix": {"path": str(matrix_path), "sha256": _sha256(matrix_path), "shape": list(rmsd.shape)},
                "clusters": cluster_records,
            },
            "footprint": {
                "definition": "convex_hull_of_projected_vdw_disks",
                "atom_scope": "complete_H0_SAM_including_H_and_anchor_O_excluding_surface_protons",
                "radii_source": "ASE ase.data.vdw_radii",
                "radii_source_version": ase.__version__,
                "radius_scale": 1.0,
                "boundary_samples_per_atom": 720,
            },
            "selection": {
                "method": "minimum_footprint_endpoint_plus_minimum_energy_endpoint_plus_non_dominated_Pareto_front_without_weighted_score",
                "minimum_footprint_cluster_id": cluster_records[minimum_footprint_index]["cluster_id"],
                "minimum_energy_cluster_id": cluster_records[minimum_energy_index]["cluster_id"],
                "pareto_front_cluster_ids": [cluster_records[index]["cluster_id"] for index in pareto_indices],
                "selected": selection_records,
            },
            "candidate_table": {"path": str(candidate_csv), "sha256": _sha256(candidate_csv), "row_count": len(candidates)},
        }
        report_path = analysis_directory / "analysis_report.json"
        write_manifest(report_path, report)
        stage_manifest = {
            "schema_version": 1,
            "status": "passed",
            "plan": {"path": str(plan_path), "sha256": args.plan_sha256},
            "analysis_script": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__).resolve())},
            "report": {"path": str(report_path), "sha256": _sha256(report_path)},
            "validated_converged_count": len(candidates),
            "geometry_cluster_count": len(formal_clusters),
            "selected_count": len(selection_records),
        }
        write_manifest(analysis_directory / "manifest.json", stage_manifest)
        top_manifest_path = run_directory / "manifest.json"
        top_manifest = json.loads(top_manifest_path.read_text())
        top_manifest.update(
            {
                "status": "analysis_passed",
                "optimization_completed_count": sum(task_status_counts.values()),
                "optimization_valid_count": len(candidates),
                "geometry_cluster_count": len(formal_clusters),
                "selected_count": len(selection_records),
                "analysis": {"path": str(analysis_directory / "manifest.json"), "sha256": _sha256(analysis_directory / "manifest.json")},
            }
        )
        write_manifest(top_manifest_path, top_manifest)
        print(json.dumps({"analysis_directory": str(analysis_directory), "report": str(report_path), "funnel": report["funnel"], "selection": report["selection"]}, indent=2), flush=True)
        return 0
    except Exception as error:
        write_manifest(
            analysis_directory / "manifest.json",
            {
                "schema_version": 1,
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise

# ---------------------------------------------------------------------------
# Experimental targeted calibration (CPU planning, CUDA worker only)
# ---------------------------------------------------------------------------


def _targeted_jsonable(value):
    """Convert ASE/NumPy values to strict-JSON-compatible Python values."""

    if isinstance(value, dict):
        return {str(key): _targeted_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_targeted_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_targeted_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            raise ValueError("targeted calibration JSON cannot contain non-finite floats")
        return float(value)
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    raise TypeError(f"unsupported targeted calibration JSON value: {type(value).__name__}")


def _targeted_write_json(path, payload):
    """Install one strict JSON artifact without replacing an existing byte."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            _targeted_jsonable(payload),
            handle,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return path


def _targeted_resolve_path(value, base=None):
    path = Path(value).expanduser()
    if not path.is_absolute() and base is not None:
        path = Path(base) / path
    return path.resolve()


def _targeted_project_relative_path(path, project_root, content_sha256=None):
    """Return a host-independent identity spelling for one evidence path."""

    resolved = _targeted_resolve_path(path)
    root = _targeted_resolve_path(project_root)
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        digest = content_sha256
        if digest is None and resolved.is_file() and not resolved.is_symlink():
            digest = _sha256(resolved)
        if digest is None:
            digest = hashlib.sha256(str(resolved.name).encode("utf-8")).hexdigest()
        return f"external-content/{digest}"


def _targeted_identity_value(value, project_root):
    """Remove absolute host paths from the targeted plan identity material."""

    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            normalized = {
                key: _targeted_identity_value(item, project_root)
                for key, item in value.items()
                if key != "path"
            }
            normalized["project_relative_path"] = _targeted_project_relative_path(
                value["path"], project_root, content_sha256=str(value["sha256"])
            )
            return normalized
        return {
            str(key): _targeted_identity_value(item, project_root)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_targeted_identity_value(item, project_root) for item in value]
    if isinstance(value, Path):
        return _targeted_project_relative_path(value, project_root)
    return value


def _targeted_file_record(path, expected_sha256=None, label="artifact"):
    """Hash one regular, non-symlink file and fail closed on disagreement."""

    path = _targeted_resolve_path(path)
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {path}")
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    actual = _sha256(path)
    if expected_sha256 is not None and actual != str(expected_sha256):
        raise ValueError(
            f"{label} hash mismatch for {path}: expected {expected_sha256}, got {actual}"
        )
    return {
        "path": str(path),
        "sha256": actual,
        "bytes": int(path.stat().st_size),
    }


def _targeted_declared_file(spec, base, label, default_name=None):
    if isinstance(spec, dict):
        raw_path = spec.get("path") or spec.get("source_path")
        expected = spec.get("sha256") or spec.get("source_sha256")
    else:
        raw_path = spec
        expected = None
    if raw_path is None:
        raw_path = default_name
    if raw_path is None:
        raise ValueError(f"{label} has no declared path")
    return _targeted_file_record(
        _targeted_resolve_path(raw_path, base), expected_sha256=expected, label=label
    )


def _targeted_read_json(record, label):
    path = _targeted_resolve_path(record["path"])
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {path}") from exc


def _targeted_parse_elements(value):
    if isinstance(value, str):
        values = [item for item in value.replace(",", " ").split() if item]
    else:
        values = list(value)
    values = [str(item) for item in values]
    if sorted(set(values)) != sorted(TARGETED_CALIBRATION_ELEMENTS):
        raise ValueError(
            "targeted calibration model-element declaration must cover exactly "
            + " ".join(TARGETED_CALIBRATION_ELEMENTS)
        )
    return list(TARGETED_CALIBRATION_ELEMENTS)


def _targeted_source_surface_contract(source_surface_evidence):
    """Resolve source-surface identity from the hash-locked evidence record."""

    if not isinstance(source_surface_evidence, dict):
        raise ValueError("accepted prototype lacks hash-locked source-surface evidence")
    raw_formula = source_surface_evidence.get("formula") or source_surface_evidence.get(
        "composition"
    )
    if not isinstance(raw_formula, dict) or not raw_formula:
        raise ValueError("hash-locked source-surface evidence lacks a composition")
    formula = {}
    for element, count in raw_formula.items():
        try:
            count = int(count)
        except (TypeError, ValueError) as exc:
            raise ValueError("source-surface composition contains a non-integer count") from exc
        if count <= 0:
            raise ValueError("source-surface composition counts must be positive")
        formula[str(element)] = count
    atom_count = source_surface_evidence.get("atom_count")
    try:
        atom_count = int(atom_count)
    except (TypeError, ValueError) as exc:
        raise ValueError("hash-locked source-surface evidence lacks atom_count") from exc
    if atom_count <= 0 or sum(formula.values()) != atom_count:
        raise ValueError("source-surface atom_count does not match its hash-locked composition")
    return {"atom_count": atom_count, "formula": dict(sorted(formula.items()))}


def _targeted_normalize_surface_frame(frame, normal_axis=None, source_label="evidence"):
    """Validate a Surface Frame without substituting a global Cartesian axis."""

    if not isinstance(frame, dict):
        raise ValueError(f"{source_label} lacks a Surface Frame")
    try:
        axes = tuple(int(value) for value in frame["periodic_fractional_axes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{source_label} Surface Frame lacks periodic_fractional_axes") from exc
    if len(axes) != 2 or len(set(axes)) != 2 or any(axis not in range(3) for axis in axes):
        raise ValueError(f"{source_label} Surface Frame periodic axes are invalid")
    try:
        basis = np.asarray(
            [
                frame["u_cartesian_unit"],
                frame["v_cartesian_unit"],
                frame["outward_normal_cartesian_unit"],
            ],
            dtype=float,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{source_label} Surface Frame lacks explicit basis vectors") from exc
    if basis.shape != (3, 3) or not np.all(np.isfinite(basis)):
        raise ValueError(f"{source_label} Surface Frame vectors are invalid")
    if not np.allclose(basis @ basis.T, np.eye(3), atol=1.0e-8, rtol=0.0):
        raise ValueError(f"{source_label} Surface Frame is not orthonormal")
    if not np.allclose(np.cross(basis[0], basis[1]), basis[2], atol=1.0e-8, rtol=0.0):
        raise ValueError(f"{source_label} Surface Frame is not right-handed")
    frame_declared_normal = frame.get("normal_axis")
    if normal_axis is not None and frame_declared_normal is not None and int(normal_axis) != int(frame_declared_normal):
        raise ValueError(f"{source_label} Surface Frame normal-axis evidence disagrees")
    if normal_axis is None:
        normal_axis = None if frame_declared_normal is None else int(frame_declared_normal)
    else:
        normal_axis = int(normal_axis)
    if normal_axis is None:
        remaining = sorted(set(range(3)) - set(axes))
        if len(remaining) != 1:
            raise ValueError(f"{source_label} Surface Frame has no unique normal axis")
        normal_axis = remaining[0]
    if normal_axis not in range(3) or normal_axis in axes:
        raise ValueError(f"{source_label} Surface Frame normal axis disagrees with periodic axes")
    normalized = {
        "periodic_fractional_axes": list(axes),
        "normal_axis": normal_axis,
        "u_cartesian_unit": basis[0].tolist(),
        "v_cartesian_unit": basis[1].tolist(),
        "outward_normal_cartesian_unit": basis[2].tolist(),
        "source_evidence": source_label,
    }
    return normalized


def _targeted_surface_frame_from_evidence(
    prototype=None, site_instance=None, site_plan=None, layer_groups=None
):
    """Resolve one frame from sealed Prototype/Instance/site-plan evidence."""

    candidates = []
    for label, payload in (
        ("site-prototype", prototype),
        ("site-instance", site_instance),
        ("site-plan", site_plan),
    ):
        if not isinstance(payload, dict):
            continue
        frame = payload.get("surface_frame")
        if frame is None and isinstance(payload.get("surface"), dict):
            frame = payload["surface"].get("surface_frame")
        if frame is not None:
            candidates.append((label, frame))
    if not candidates:
        raise ValueError("targeted calibration requires a sealed Site Prototype/Instance Surface Frame")
    declared_normal = None
    if isinstance(layer_groups, dict) and layer_groups.get("normal_axis") is not None:
        declared_normal = int(layer_groups["normal_axis"])
    resolved = [
        _targeted_normalize_surface_frame(frame, declared_normal, label)
        for label, frame in candidates
    ]
    reference = resolved[0]
    for other in resolved[1:]:
        if reference["periodic_fractional_axes"] != other["periodic_fractional_axes"] or not np.allclose(
            np.asarray(
                [
                    reference["u_cartesian_unit"],
                    reference["v_cartesian_unit"],
                    reference["outward_normal_cartesian_unit"],
                ]
            ),
            np.asarray(
                [
                    other["u_cartesian_unit"],
                    other["v_cartesian_unit"],
                    other["outward_normal_cartesian_unit"],
                ]
            ),
            atol=1.0e-8,
            rtol=0.0,
        ):
            raise ValueError("sealed Surface Frame evidence disagrees")
    return reference


def _targeted_partial_mic(vector, cell, periodic_axes=None):
    if periodic_axes is None:
        raise ValueError("targeted partial MIC requires Surface Frame periodic axes")
    pbc = np.zeros(3, dtype=bool)
    pbc[list(periodic_axes)] = True
    mic_vector, distance = find_mic(
        np.asarray(vector, dtype=float), cell=np.asarray(cell, dtype=float), pbc=pbc
    )
    return np.asarray(mic_vector, dtype=float).reshape(3), float(np.asarray(distance))


def _targeted_distance(atoms, left, right, periodic_axes=None):
    _, distance = _targeted_partial_mic(
        np.asarray(atoms.positions[int(left)]) - np.asarray(atoms.positions[int(right)]),
        atoms.cell,
        periodic_axes=periodic_axes,
    )
    return distance


def _targeted_displacement_A(initial, final, indices, periodic_axes=None):
    maximum = -1.0
    maximum_index = None
    values = []
    for index in map(int, indices):
        _, distance = _targeted_partial_mic(
            np.asarray(final.positions[index]) - np.asarray(initial.positions[index]),
            initial.cell,
            periodic_axes=periodic_axes,
        )
        values.append({"atom_index_0based": index, "displacement_A": distance})
        if distance > maximum:
            maximum = distance
            maximum_index = index
    return {
        "max_A": max(0.0, float(maximum)),
        "atom_index_0based": maximum_index,
        "atom_id_1based": None if maximum_index is None else maximum_index + 1,
        "per_atom": values,
    }


def build_targeted_calibration_initial_structure(
    final_h0,
    molecule_index_1based,
    proton_records,
    substrate_atom_count=TARGETED_CALIBRATION_SUBSTRATE_ATOMS,
    molecule_atoms=TARGETED_CALIBRATION_MOLECULE_ATOMS,
):
    """Extract exactly one sequential-growth molecule and append mapped H atoms.

    ``final_h0`` is the sealed readback object.  No molecule is rebuilt from a
    CIF or from an orientation guess; the selected block's original coordinates
    are copied byte-for-byte at the structure level and only the hash-verified
    prototype H displacement vectors are applied to mapped parent O atoms.
    """

    if len(final_h0) < substrate_atom_count:
        raise ValueError("sealed H0 structure is shorter than the working substrate")
    molecule_index_1based = int(molecule_index_1based)
    molecule_atoms = int(molecule_atoms)
    start = int(substrate_atom_count) + (molecule_index_1based - 1) * molecule_atoms
    stop = start + molecule_atoms
    if molecule_index_1based <= 0 or stop > len(final_h0):
        raise ValueError("target molecule block is outside the sealed H0 structure")
    substrate = final_h0[:substrate_atom_count]
    molecule = final_h0[start:stop]
    if len(proton_records) != TARGETED_CALIBRATION_SURFACE_H_COUNT:
        raise ValueError("targeted calibration requires exactly two mapped surface H atoms")
    h_positions = []
    for record in proton_records:
        parent = int(record["working_parent_atom_index_0based"])
        vector = np.asarray(record["parent_to_h_vector_A"], dtype=float)
        if parent < 0 or parent >= substrate_atom_count:
            raise ValueError("mapped surface-H parent is outside the working substrate")
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError("mapped surface-H displacement vector is invalid")
        h_positions.append(np.asarray(substrate.positions[parent], dtype=float) + vector)
    symbols = substrate.get_chemical_symbols() + molecule.get_chemical_symbols() + ["H"] * len(h_positions)
    positions = np.vstack((substrate.positions, molecule.positions, np.asarray(h_positions)))
    result = Atoms(
        symbols=symbols,
        positions=positions,
        cell=np.asarray(final_h0.cell, dtype=float),
        pbc=np.asarray(final_h0.pbc, dtype=bool),
    )
    if len(result) != TARGETED_CALIBRATION_TOTAL_ATOMS:
        raise ValueError(f"targeted calibration input must contain {TARGETED_CALIBRATION_TOTAL_ATOMS} atoms")
    return result


def resolve_targeted_donor_mapping(
    step_record,
    site_instance,
    working_layer_atom_ids_1based,
    substrate,
    expected_permutation=None,
    expected_metal_ids=None,
):
    """Join trajectory donor records to the accepted Site Instance mapping.

    The actual post-doping metal element is a Site Instance fact.  It is never
    replaced by a calibration-wide element whitelist.
    """

    labels = list(step_record.get("target_donor_labels", []))
    if labels != TARGETED_CALIBRATION_DONOR_LABELS or len(set(labels)) != len(labels):
        raise ValueError("target trajectory donor labels are incomplete or changed")
    permutation = list(step_record.get("oxygen_permutation", []))
    if expected_permutation is not None and permutation != list(expected_permutation):
        raise ValueError("target trajectory oxygen permutation changed")
    if len(permutation) != len(labels) or len(set(permutation)) != len(permutation):
        raise ValueError("target trajectory oxygen permutation is not one-to-one")
    if site_instance.get("site_instance_id") != TARGETED_CALIBRATION_SITE_INSTANCE_ID:
        raise ValueError("target Site Instance identity changed")
    if site_instance.get("parent_site_prototype_id") != TARGETED_CALIBRATION_PROTOTYPE_ID:
        raise ValueError("target Site Instance prototype identity changed")
    full_ids = [int(value) for value in site_instance.get("actual_metal_atom_ids_1based", [])]
    if len(full_ids) != len(set(full_ids)):
        raise ValueError("target Site Instance metal IDs are not unique")
    if expected_metal_ids is not None and sorted(full_ids) != sorted(int(value) for value in expected_metal_ids):
        raise ValueError("target Site Instance metal IDs do not match sealed evidence")
    actual_metals = [
        item for item in site_instance.get("actual_metals", [])
        if isinstance(item, dict) and "atom_id_1based" in item
    ]
    actual_metal_by_id = {}
    for item in actual_metals:
        metal_id = int(item["atom_id_1based"])
        if metal_id in actual_metal_by_id:
            raise ValueError("target Site Instance actual-metal mapping is not unique")
        actual_metal_by_id[metal_id] = item
    if set(actual_metal_by_id) != set(full_ids):
        raise ValueError("target Site Instance actual-metal mapping is incomplete")
    instance_mapping = {
        item.get("probe_donor_id"): item
        for item in site_instance.get("donor_metal_mapping", [])
        if isinstance(item, dict)
    }
    step_mapping = {
        item.get("donor_label"): item
        for item in step_record.get("registered_interface_bonds", [])
        if isinstance(item, dict)
    }
    if set(instance_mapping) != set(labels) or set(step_mapping) != set(labels):
        raise ValueError("target donor mapping is incomplete")
    if len(instance_mapping) != len(labels) or len(step_mapping) != len(labels):
        raise ValueError("target donor mapping contains duplicate labels")
    working_ids = [int(value) for value in working_layer_atom_ids_1based]
    if len(set(working_ids)) != len(working_ids):
        raise ValueError("working layer atom IDs are not unique")
    symbols = substrate.get_chemical_symbols() if hasattr(substrate, "get_chemical_symbols") else list(substrate)
    records = []
    for label, expected_local in zip(labels, permutation):
        step = step_mapping[label]
        instance = instance_mapping[label]
        local = int(step.get("molecule_atom_index_0based", -1))
        full_id = int(step.get("full_metal_atom_id_1based", -1))
        instance_id = int(instance.get("metal_atom_id_1based", -1))
        if local != int(expected_local) or instance_id != full_id:
            raise ValueError(f"donor mapping mismatch for {label}")
        if full_id not in working_ids:
            raise ValueError(f"registered donor metal {full_id} is not in the working layer")
        working_index = working_ids.index(full_id)
        if not 0 <= working_index < len(symbols):
            raise ValueError(f"registered donor working index is outside substrate: {working_index}")
        instance_record = actual_metal_by_id[full_id]
        expected_element = str(
            instance_record.get("actual_element_after_doping")
            or instance_record.get("element")
            or ""
        )
        step_element = str(step.get("actual_metal_element", ""))
        if not expected_element or step_element != expected_element or symbols[working_index] != expected_element:
            raise ValueError(f"registered donor metal element mismatch for {label}")
        records.append(
            {
                "donor_label": label,
                "molecule_atom_index_0based": local,
                "molecule_atom_id_1based": local + 1,
                "full_metal_atom_id_1based": full_id,
                "working_metal_atom_index_0based": working_index,
                "working_metal_atom_id_1based": working_index + 1,
                "metal_element": step_element,
                "expected_metal_element_from_instance": expected_element,
                "distance_window_A": list(TARGETED_CALIBRATION_MAPPED_BOND_WINDOW_A),
            }
        )
    return records


def _targeted_source_to_supercell_index(source_atoms, supercell, source_atom_id, translation):
    """Use the maintained source->supercell mapper, never a coordinate guess."""

    from substrate_library import _map_periodic_vertex_to_supercell

    source_atom_id = int(source_atom_id)
    if source_atom_id < 0 or source_atom_id >= len(source_atoms):
        raise ValueError("source atom identity is outside the accepted source surface")
    translation = np.asarray(translation, dtype=int)
    if translation.shape != (3,):
        raise ValueError("source-cell translation must have three integer components")
    fractional = np.asarray(source_atoms.get_scaled_positions(wrap=True))[source_atom_id]
    position = (fractional + translation) @ np.asarray(source_atoms.cell, dtype=float)
    return _map_periodic_vertex_to_supercell(
        supercell, position, source_atoms[source_atom_id].symbol
    )


def _targeted_supercell_matrix(source_atoms, target_cell, build_manifest=None):
    if build_manifest is not None:
        matrix = np.asarray(build_manifest.get("supercell", {}).get("matrix"), dtype=int)
        if matrix.shape != (3, 3):
            raise ValueError("substrate build evidence has no 3x3 supercell matrix")
    else:
        matrix_float = np.asarray(target_cell, dtype=float) @ np.linalg.inv(
            np.asarray(source_atoms.cell, dtype=float)
        )
        matrix = np.rint(matrix_float).astype(int)
        if not np.allclose(matrix_float, matrix, atol=1.0e-8, rtol=0.0):
            raise ValueError("target cell is not an integer source-cell supercell")
    expected_cell = matrix @ np.asarray(source_atoms.cell, dtype=float)
    if not np.allclose(expected_cell, np.asarray(target_cell, dtype=float), atol=1.0e-7, rtol=0.0):
        raise ValueError("source-cell mapping cell does not match sealed target cell")
    if int(round(abs(np.linalg.det(matrix)))) <= 0:
        raise ValueError("source-cell supercell matrix is singular")
    return matrix


def map_targeted_surface_protons(
    representative,
    prototype,
    source_surface,
    full_supercell,
    source_cell_translation,
    working_layer_atom_ids_1based,
    source_surface_evidence,
    surface_frame,
):
    """Recover prototype H vectors and map their parent O identities.

    The H coordinates are reconstructed only as ``mapped_parent_O +`` the
    representative's hash-verified MIC H-parent vector.  Missing/changed
    parent identities, non-O parents, or incomplete assignments fail closed.
    """

    assignments = (prototype.get("surface_proton_policy") or {}).get("selected_assignments")
    if not isinstance(assignments, list) or len(assignments) != TARGETED_CALIBRATION_SURFACE_H_COUNT:
        raise ValueError("accepted prototype does not contain exactly two surface-H assignments")
    if len(representative) < TARGETED_CALIBRATION_SURFACE_H_COUNT:
        raise ValueError("accepted prototype representative is too short")
    source_contract = _targeted_source_surface_contract(source_surface_evidence)
    if len(source_surface) != source_contract["atom_count"] or _composition(source_surface) != source_contract["formula"]:
        raise ValueError("accepted source surface bytes do not match hash-locked composition")
    if len(representative) < source_contract["atom_count"]:
        raise ValueError("prototype representative lacks the hash-locked source-surface atom block")
    if representative[:len(source_surface)].get_chemical_symbols() != source_surface.get_chemical_symbols():
        raise ValueError("prototype representative/source surface atom identity mismatch")
    translation = np.asarray(source_cell_translation, dtype=int)
    if translation.shape != (3,):
        raise ValueError("Site Instance source-cell translation is invalid")
    working_ids = [int(value) for value in working_layer_atom_ids_1based]
    if len(set(working_ids)) != len(working_ids):
        raise ValueError("working-layer mapping IDs are not unique")
    frame = _targeted_normalize_surface_frame(surface_frame, source_label="prototype")
    result = []
    used_parents = set()
    for assignment in assignments:
        h_id = int(assignment.get("surface_proton_atom_id_0_based", -1))
        parent_id = int(assignment.get("nearest_oxygen_atom_id_0_based", -1))
        if not bool(assignment.get("nearest_oxygen_is_substrate", False)):
            raise ValueError("prototype surface-H parent is not declared as substrate O")
        if h_id < 0 or h_id >= len(representative) or parent_id < 0 or parent_id >= len(representative):
            raise ValueError("prototype surface-H identity is outside representative structure")
        if representative[h_id].symbol != "H" or representative[parent_id].symbol != "O":
            raise ValueError("prototype surface-H assignment does not identify H/O atoms")
        if parent_id in used_parents:
            raise ValueError("two conserved surface H atoms share one parent O")
        used_parents.add(parent_id)
        vector = np.asarray(representative.positions[h_id]) - np.asarray(representative.positions[parent_id])
        vector, distance = _targeted_partial_mic(
            vector,
            representative.cell,
            periodic_axes=tuple(frame["periodic_fractional_axes"]),
        )
        declared_distance = float(assignment.get("nearest_oxygen_distance_A", distance))
        if not np.isfinite(declared_distance) or abs(declared_distance - distance) > 1.0e-5:
            raise ValueError("prototype surface-H parent distance changed")
        full_index = _targeted_source_to_supercell_index(
            source_surface,
            full_supercell,
            parent_id,
            translation,
        )
        full_id = full_index + 1
        if full_id not in working_ids:
            raise ValueError("prototype surface-H parent maps into the archived lower layer")
        working_index = working_ids.index(full_id)
        if full_supercell[full_index].symbol != "O":
            raise ValueError("mapped prototype surface-H parent is not O in the source supercell")
        result.append(
            {
                "prototype_h_atom_id_0based": h_id,
                "prototype_parent_o_atom_id_0based": parent_id,
                "source_parent_atom_id_0based": parent_id,
                "full_supercell_parent_atom_id_1based": full_id,
                "working_parent_atom_index_0based": working_index,
                "working_parent_atom_id_1based": working_index + 1,
                "parent_element": "O",
                "parent_to_h_vector_A": vector.tolist(),
                "parent_h_distance_A": distance,
            }
        )
    return result


def _targeted_interface_covalent_contacts(
    atoms,
    substrate_indices,
    molecule_indices,
    radius_scale=TARGETED_CALIBRATION_COVALENT_RADIUS_SCALE,
    periodic_axes=None,
):
    """Enumerate substrate/SAM contacts under an explicit covalent-radius contract."""

    if periodic_axes is None:
        raise ValueError("targeted interface contacts require Surface Frame periodic axes")
    axes = tuple(int(value) for value in periodic_axes)
    if len(axes) != 2 or len(set(axes)) != 2 or any(axis not in range(3) for axis in axes):
        raise ValueError("targeted interface contacts require exactly two surface-periodic axes")
    symbols = atoms.get_chemical_symbols()
    contacts = {}
    for substrate_index in map(int, substrate_indices):
        for molecule_index in map(int, molecule_indices):
            left = symbols[substrate_index]
            right = symbols[molecule_index]
            cutoff = float(radius_scale) * (
                covalent_radii[atomic_numbers[left]]
                + covalent_radii[atomic_numbers[right]]
            )
            distance = _targeted_distance(
                atoms, molecule_index, substrate_index, periodic_axes=axes
            )
            if distance <= cutoff:
                contacts[(substrate_index, molecule_index)] = {
                    "substrate_atom_index_0based": substrate_index,
                    "substrate_element": left,
                    "molecule_atom_index_0based": molecule_index,
                    "molecule_element": right,
                    "distance_A": distance,
                    "cutoff_A": cutoff,
                    "radius_scale": float(radius_scale),
                    "radii_source": "ASE ase.data.covalent_radii",
                }
    return contacts


def targeted_p_o_coordination(
    atoms,
    phosphorus_index,
    coordination_cutoff_A=TARGETED_CALIBRATION_P_O_COORDINATION_CUTOFF_A,
    periodic_axes=None,
    molecule_indices=None,
    substrate_indices=None,
):
    """Report total P coordination as intramolecular P-C/P-O plus surface O.

    The molecular contribution is the fixed P-C plus three P-O graph, while a
    surface O contributes only when it is in the explicit substrate index set
    and its partial-PBC distance is at most 2.0 A.  No five-O molecular rule is
    used.
    """

    phosphorus_index = int(phosphorus_index)
    if phosphorus_index < 0 or phosphorus_index >= len(atoms) or atoms[phosphorus_index].symbol != "P":
        raise ValueError("P coordination helper requires a phosphorus atom")
    cutoff = float(coordination_cutoff_A)
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("P coordination cutoff must be finite and positive")
    if molecule_indices is None:
        molecule_indices = list(range(len(atoms)))
    else:
        molecule_indices = [int(index) for index in molecule_indices]
    if substrate_indices is None:
        substrate_indices = []
    else:
        substrate_indices = [int(index) for index in substrate_indices]
    if phosphorus_index not in molecule_indices:
        raise ValueError("P atom must belong to the explicit molecule index set")
    if len(set(molecule_indices)) != len(molecule_indices) or len(set(substrate_indices)) != len(substrate_indices):
        raise ValueError("P coordination index sets must be unique")
    if set(molecule_indices) & set(substrate_indices):
        raise ValueError("molecule and substrate P coordination index sets overlap")
    for index in molecule_indices + substrate_indices:
        if index < 0 or index >= len(atoms):
            raise ValueError("P coordination index is outside the structure")
    intramolecular = []
    surface_oxygen = []
    for index in molecule_indices:
        if index == phosphorus_index or atoms[index].symbol not in {"C", "O"}:
            continue
        distance = _targeted_distance(atoms, phosphorus_index, index, periodic_axes=periodic_axes)
        if distance <= cutoff:
            intramolecular.append(
                {
                    "atom_index_0based": index,
                    "atom_id_1based": index + 1,
                    "element": atoms[index].symbol,
                    "distance_A": distance,
                }
            )
    for index in substrate_indices:
        if atoms[index].symbol != "O":
            continue
        distance = _targeted_distance(atoms, phosphorus_index, index, periodic_axes=periodic_axes)
        if distance <= cutoff:
            surface_oxygen.append(
                {
                    "atom_index_0based": index,
                    "atom_id_1based": index + 1,
                    "element": "O",
                    "distance_A": distance,
                }
            )
    intramolecular.sort(key=lambda item: (item["distance_A"], item["atom_index_0based"]))
    surface_oxygen.sort(key=lambda item: (item["distance_A"], item["atom_index_0based"]))
    p_c = [item for item in intramolecular if item["element"] == "C"]
    p_o = [item for item in intramolecular if item["element"] == "O"]
    total = len(intramolecular) + len(surface_oxygen)
    return {
        "cutoff_A": cutoff,
        "intramolecular_coordination": len(intramolecular),
        "intramolecular_p_c_contacts": p_c,
        "intramolecular_p_o_contacts": p_o,
        "intramolecular_contacts": intramolecular,
        "surface_o_contacts": len(surface_oxygen),
        "surface_o_contact_records": surface_oxygen,
        "total_coordination": total,
        # ``count``/``neighbors`` are retained as descriptive aliases inside
        # the new schema; classification uses the named semantic fields above.
        "count": total,
        "neighbors": intramolecular + surface_oxygen,
    }


def _targeted_coordination_values(value):
    if isinstance(value, dict):
        total = value.get("total_coordination", value.get("count"))
        intramolecular = value.get("intramolecular_coordination")
        surface = value.get("surface_o_contacts")
        if total is None:
            raise ValueError("coordination report lacks total_coordination")
        return int(total), None if intramolecular is None else int(intramolecular), None if surface is None else int(surface)
    # Direct callers must use the new report in production; this scalar branch
    # keeps the pure classifier useful for compact threshold tests.
    return int(value), None, None


def classify_targeted_calibration(
    converged,
    hard_gates_passed,
    final_target_p_o_distance_A,
    initial_p_coordination,
    final_p_coordination,
    *,
    stable_p_o_cutoff_A=TARGETED_CALIBRATION_P_O_STABLE_CUTOFF_A,
    released_p_o_cutoff_A=TARGETED_CALIBRATION_P_O_RELEASED_CUTOFF_A,
    fivefold_coordination=TARGETED_CALIBRATION_P_COORDINATION_FIVEFOLD,
    reasons=None,
):
    """Classify calibration observables without weakening production gates."""

    if isinstance(hard_gates_passed, dict):
        hard_pass = bool(hard_gates_passed) and all(bool(value) for value in hard_gates_passed.values())
    else:
        hard_pass = bool(hard_gates_passed)
    reasons = list(reasons or [])
    try:
        distance = float(final_target_p_o_distance_A)
        initial_total, initial_intramolecular, initial_surface = _targeted_coordination_values(initial_p_coordination)
        final_total, final_intramolecular, final_surface = _targeted_coordination_values(final_p_coordination)
    except (TypeError, ValueError) as exc:
        return {
            "classification": "failed_or_unknown",
            "reason_codes": reasons + ["invalid_terminal_target_geometry"],
            "stable_extra_contact": False,
            "released_repulsion": False,
        }
    thresholds = {
        "stable_p_o_cutoff_A": float(stable_p_o_cutoff_A),
        "released_p_o_cutoff_A": float(released_p_o_cutoff_A),
        "minimum_total_coordination_for_stable": int(fivefold_coordination),
        "normal_intramolecular_coordination_for_released": 4,
    }
    if not np.isfinite(distance):
        return {
            "classification": "failed_or_unknown",
            "reason_codes": reasons + ["nonfinite_terminal_target_p_o_distance"],
            "stable_extra_contact": False,
            "released_repulsion": False,
            "thresholds": thresholds,
        }
    if not bool(converged) or not hard_pass:
        if not bool(converged):
            reasons.append("optimizer_not_converged")
        if not hard_pass:
            reasons.append("one_or_more_hard_gates_failed")
        return {
            "classification": "failed_or_unknown",
            "reason_codes": sorted(set(reasons)),
            "stable_extra_contact": False,
            "released_repulsion": False,
            "thresholds": thresholds,
            "initial_coordination": {
                "total_coordination": initial_total,
                "intramolecular_coordination": initial_intramolecular,
                "surface_o_contacts": initial_surface,
            },
            "final_coordination": {
                "total_coordination": final_total,
                "intramolecular_coordination": final_intramolecular,
                "surface_o_contacts": final_surface,
            },
        }
    stable_coordination = final_total >= int(fivefold_coordination)
    released_coordination = (
        final_intramolecular in {None, 4}
        and (final_surface in {None, 0})
        and final_total == 4
    )
    if distance <= float(stable_p_o_cutoff_A) and stable_coordination:
        return {
            "classification": "stable_extra_contact",
            "reason_codes": ["target_p_o_within_stable_cutoff", "total_p_coordination_at_least_five"],
            "stable_extra_contact": True,
            "released_repulsion": False,
            "thresholds": thresholds,
            "initial_coordination": {
                "total_coordination": initial_total,
                "intramolecular_coordination": initial_intramolecular,
                "surface_o_contacts": initial_surface,
            },
            "final_coordination": {
                "total_coordination": final_total,
                "intramolecular_coordination": final_intramolecular,
                "surface_o_contacts": final_surface,
            },
        }
    if distance > float(released_p_o_cutoff_A) and released_coordination:
        return {
            "classification": "released_repulsion",
            "reason_codes": ["target_p_o_above_released_cutoff", "normal_intramolecular_fourfold_coordination"],
            "stable_extra_contact": False,
            "released_repulsion": True,
            "thresholds": thresholds,
            "initial_coordination": {
                "total_coordination": initial_total,
                "intramolecular_coordination": initial_intramolecular,
                "surface_o_contacts": initial_surface,
            },
            "final_coordination": {
                "total_coordination": final_total,
                "intramolecular_coordination": final_intramolecular,
                "surface_o_contacts": final_surface,
            },
        }
    reasons.append("target_p_o_in_intermediate_or_coordination_failed_window")
    return {
        "classification": "failed_or_unknown",
        "reason_codes": sorted(set(reasons)),
        "stable_extra_contact": False,
        "released_repulsion": False,
        "thresholds": thresholds,
        "initial_coordination": {
            "total_coordination": initial_total,
            "intramolecular_coordination": initial_intramolecular,
            "surface_o_contacts": initial_surface,
        },
        "final_coordination": {
            "total_coordination": final_total,
            "intramolecular_coordination": final_intramolecular,
            "surface_o_contacts": final_surface,
        },
    }


def _targeted_strict_vdw_audit(atoms, donor_mapping, surface_frame, radius_scale=TARGETED_CALIBRATION_VDW_RADIUS_SCALE):
    """Run the unchanged generic strict-vdW audit under sealed frame evidence."""

    from sam_structure_tools import periodic_vdw_collision_audit

    frame = _targeted_normalize_surface_frame(surface_frame, source_label="targeted plan")
    periodic_axes = tuple(frame["periodic_fractional_axes"])
    substrate_indices = list(range(TARGETED_CALIBRATION_SUBSTRATE_ATOMS))
    surface_h_indices = list(
        range(
            TARGETED_CALIBRATION_TOTAL_ATOMS - TARGETED_CALIBRATION_SURFACE_H_COUNT,
            TARGETED_CALIBRATION_TOTAL_ATOMS,
        )
    )
    # The targeted structure is substrate + SAM + mapped H.  The audit accepts
    # an explicit substrate array, so put the two H atoms beside substrate
    # atoms while preserving the global 0-based local IDs used by production.
    molecule_start = TARGETED_CALIBRATION_SUBSTRATE_ATOMS
    molecule_stop = molecule_start + TARGETED_CALIBRATION_MOLECULE_ATOMS
    molecule_indices = list(range(molecule_start, molecule_stop))
    audit_substrate_indices = substrate_indices + surface_h_indices
    substrate_positions = np.asarray(atoms.positions[audit_substrate_indices], dtype=float)
    substrate_symbols = [atoms[index].symbol for index in audit_substrate_indices]
    molecule_positions = np.asarray(atoms.positions[molecule_indices], dtype=float)
    molecule_symbols = [atoms[index].symbol for index in molecule_indices]
    radii = {
        symbol: float(vdw_radii[atomic_numbers[symbol]])
        for symbol in sorted(set(substrate_symbols + molecule_symbols))
    }
    mapped_windows = {}
    for record in donor_mapping:
        mapped_windows[
            (
                int(record["molecule_atom_index_0based"]),
                int(record["working_metal_atom_index_0based"]),
            )
        ] = tuple(TARGETED_CALIBRATION_MAPPED_BOND_WINDOW_A)
    return periodic_vdw_collision_audit(
        molecule_positions=molecule_positions,
        molecule_symbols=molecule_symbols,
        molecule_atom_ids=list(range(len(molecule_positions))),
        substrate_positions=substrate_positions,
        substrate_symbols=substrate_symbols,
        substrate_atom_ids=list(audit_substrate_indices),
        cell=np.asarray(atoms.cell, dtype=float),
        periodic_axes=periodic_axes,
        surface_frame=frame,
        radii_A=radii,
        radius_scale=float(radius_scale),
        mapped_bond_windows_A=mapped_windows,
    )


def _targeted_calibration_residual_nontarget_audit(
    raw_production_audit,
    target_p_molecule_local_index_0based,
    target_surface_oxygen_atom_index_0based,
):
    """Filter only the exact calibration pair, without changing raw evidence."""

    collisions = list(raw_production_audit.get("collisions", []) or [])
    target_pair = (
        int(target_p_molecule_local_index_0based),
        int(target_surface_oxygen_atom_index_0based),
    )
    mapped_pairs = {
        (int(record["molecule_atom_id"]), int(record["substrate_atom_id"]))
        for record in (raw_production_audit.get("mapped_bonds", []) or [])
        if isinstance(record, dict)
        and "molecule_atom_id" in record
        and "substrate_atom_id" in record
    }
    target_contacts = []
    residual_nontarget = []
    nonregistered_nontarget = []
    registered_overlaps = []
    for record in collisions:
        pair = (int(record.get("molecule_atom_id", -1)), int(record.get("substrate_atom_id", -1)))
        if pair == target_pair:
            target_contacts.append(record)
        else:
            residual_nontarget.append(record)
            if pair in mapped_pairs:
                registered_overlaps.append(record)
            else:
                nonregistered_nontarget.append(record)
    return {
        "schema": "sam-targeted-calibration-residual-nontarget-audit-v2",
        "raw_production_audit_passed": bool(raw_production_audit.get("passed", False)),
        "target_pair": {
            "molecule_atom_id_0based": target_pair[0],
            "substrate_atom_id_0based": target_pair[1],
            "target_pair_is_not_a_production_exemption": True,
        },
        "target_pair_short_contacts": target_contacts,
        "non_target_short_contacts": residual_nontarget,
        "non_target_short_contact_count": len(residual_nontarget),
        "nonregistered_nontarget_short_contacts": nonregistered_nontarget,
        "nonregistered_nontarget_short_contact_count": len(nonregistered_nontarget),
        "registered_bond_overlap_diagnostics": registered_overlaps,
        "passed": not nonregistered_nontarget,
        "filter_definition": "only_exact_target_P_surface_O_pair_removed_from_raw_collision_list",
        "production_exemptions_modified": False,
    }


def _targeted_audit_summary(audit):
    return {
        "passed": bool(audit.get("passed", False)),
        "collision_count": int(audit.get("collision_count", len(audit.get("collisions", [])))),
        "mapped_bond_count": int(audit.get("mapped_bond_count", len(audit.get("mapped_bonds", [])))),
        "mapped_bond_violation_count": int(audit.get("mapped_bond_violation_count", 0)),
        "periodic_axes": list(audit.get("periodic_axes", [])),
        "normal_axis_wrapped": bool(audit.get("normal_axis_wrapped", False)),
        "radius_scale": float(audit.get("radius_scale", TARGETED_CALIBRATION_VDW_RADIUS_SCALE)),
        "neighbor_search": audit.get("neighbor_search", {}),
        "nonregistered_short_contacts": audit.get("collisions", []),
    }


def validate_targeted_calibration_final_geometry(
    initial,
    final,
    plan,
    forces,
    optimizer_steps,
    optimizer_converged,
    initial_strict_vdw=None,
    final_strict_vdw=None,
):
    """Perform the independent terminal validation required by calibration."""

    system = plan["system"]
    sealed_audit_id_convention = dict(
        system.get("sealed_audit_id_convention")
        or TARGETED_CALIBRATION_SEALED_AUDIT_ID_CONVENTION
    )
    constraints = plan["constraints"]
    parameters = plan["parameters"]
    substrate_count = int(system["substrate_atom_count"])
    molecule_start = int(system["molecule_atom_ids_0based"][0])
    molecule_stop = int(system["molecule_atom_ids_0based"][-1]) + 1
    molecule_indices = list(range(molecule_start, molecule_stop))
    substrate_indices = list(range(substrate_count))
    surface_frame = _targeted_normalize_surface_frame(
        system.get("surface_frame"),
        normal_axis=system.get("normal_axis"),
        source_label="targeted plan system",
    )
    periodic_axes = tuple(surface_frame["periodic_fractional_axes"])
    surface_h_indices = [int(value) for value in system["surface_h_atom_ids_0based"]]
    frozen = [int(value) for value in constraints["frozen_substrate_indices_0based"]]
    top_substrate = [int(value) for value in constraints["movable_substrate_indices_0based"]]
    movable = [int(value) for value in constraints["movable_atom_indices_0based"]]
    checks = {}
    reasons = []
    checks["atom_count_unchanged"] = len(initial) == len(final) == int(system["expected_atom_count"])
    checks["symbol_identity_unchanged"] = final.get_chemical_symbols() == initial.get_chemical_symbols()
    checks["cell_identity_unchanged"] = bool(
        np.allclose(
            np.asarray(final.cell, dtype=float),
            np.asarray(initial.cell, dtype=float),
            atol=float(parameters["cell_max_abs_change_A"]),
            rtol=0.0,
        )
        and np.array_equal(np.asarray(final.pbc, dtype=bool), np.asarray(initial.pbc, dtype=bool))
    )
    if not checks["atom_count_unchanged"]:
        reasons.append("atom_or_block_count_changed")
    if not checks["symbol_identity_unchanged"]:
        reasons.append("symbol_identity_changed")
    if not checks["cell_identity_unchanged"]:
        reasons.append("cell_or_pbc_changed")

    frozen_displacement = _targeted_displacement_A(initial, final, frozen, periodic_axes=periodic_axes)
    top_displacement = _targeted_displacement_A(initial, final, top_substrate, periodic_axes=periodic_axes)
    checks["frozen_identity_unchanged"] = frozen_displacement["max_A"] <= float(
        parameters["frozen_max_displacement_A"]
    )
    if not checks["frozen_identity_unchanged"]:
        reasons.append("frozen_lower_substrate_moved")

    initial_edges = _sam_edge_set(initial, range(molecule_start, molecule_stop), scale=float(parameters["covalent_radius_scale"]))
    final_edges = _sam_edge_set(final, range(molecule_start, molecule_stop), scale=float(parameters["covalent_radius_scale"]))
    checks["complete_sam_graph_unchanged"] = initial_edges == final_edges == system["initial_sam_covalent_edges_global_1based"]
    if not checks["complete_sam_graph_unchanged"]:
        reasons.append("complete_46_atom_sam_graph_changed")

    proton_records = []
    proton_ok = True
    for record, index in zip(system["surface_proton_records"], surface_h_indices):
        parent = int(record["working_parent_atom_index_0based"])
        distance = _targeted_distance(final, index, parent, periodic_axes=periodic_axes)
        substrate_oxygen_indices = [
            i for i in range(substrate_count) if final[i].symbol == "O"
        ]
        nearest = min(
            substrate_oxygen_indices,
            key=lambda oxygen: _targeted_distance(final, index, oxygen, periodic_axes=periodic_axes),
        )
        item = {
            "hydrogen_atom_index_0based": index,
            "parent_oxygen_atom_index_0based": parent,
            "parent_oxygen_element_initial": initial[parent].symbol,
            "parent_oxygen_element_final": final[parent].symbol,
            "parent_distance_initial_A": _targeted_distance(initial, index, parent, periodic_axes=periodic_axes),
            "parent_distance_final_A": distance,
            "nearest_final_substrate_oxygen_index_0based": nearest,
            "nearest_final_substrate_oxygen_distance_A": _targeted_distance(final, index, nearest, periodic_axes=periodic_axes),
            "parent_distance_cutoff_A": float(parameters["surface_h_parent_max_distance_A"]),
            "parent_retained": bool(
                initial[parent].symbol == "O"
                and final[parent].symbol == "O"
                and distance <= float(parameters["surface_h_parent_max_distance_A"])
                and nearest == parent
            ),
        }
        proton_records.append(item)
        proton_ok = proton_ok and item["parent_retained"]
    checks["exact_two_surface_h_parent_identities_retained"] = len(proton_records) == 2 and proton_ok
    if not checks["exact_two_surface_h_parent_identities_retained"]:
        reasons.append("surface_h_parent_identity_or_distance_failed")

    donor_records = []
    donor_ok = True
    for mapping in system["donor_mapping"]:
        local = molecule_start + int(mapping["molecule_atom_index_0based"])
        metal = int(mapping["working_metal_atom_index_0based"])
        low, high = mapping["distance_window_A"]
        initial_distance = _targeted_distance(initial, local, metal, periodic_axes=periodic_axes)
        final_distance = _targeted_distance(final, local, metal, periodic_axes=periodic_axes)
        passed = distance_within_closed_window(final_distance, low, high)
        donor_records.append(
            {
                **mapping,
                "initial_distance_A": initial_distance,
                "final_distance_A": final_distance,
                "final_within_registered_window": passed,
            }
        )
        donor_ok = donor_ok and passed and final[local].symbol == "O" and final[metal].symbol == mapping.get("expected_metal_element_from_instance", mapping.get("metal_element", ""))
    checks["three_registered_donor_contacts_and_denticity"] = len(donor_records) == 3 and donor_ok
    if not checks["three_registered_donor_contacts_and_denticity"]:
        reasons.append("registered_donor_contact_or_denticity_failed")

    target_p_local = int(system["target_p_molecule_local_index_0based"])
    phosphorus = molecule_start + target_p_local
    target_oxygen = int(system["target_surface_oxygen_atom_index_0based"])
    initial_target_distance = _targeted_distance(initial, phosphorus, target_oxygen, periodic_axes=periodic_axes)
    final_target_distance = _targeted_distance(final, phosphorus, target_oxygen, periodic_axes=periodic_axes)
    initial_coordination = targeted_p_o_coordination(
        initial,
        phosphorus,
        float(parameters["p_o_coordination_cutoff_A"]),
        periodic_axes=periodic_axes,
        molecule_indices=molecule_indices,
        substrate_indices=substrate_indices,
    )
    final_coordination = targeted_p_o_coordination(
        final,
        phosphorus,
        float(parameters["p_o_coordination_cutoff_A"]),
        periodic_axes=periodic_axes,
        molecule_indices=molecule_indices,
        substrate_indices=substrate_indices,
    )
    target_p_o = {
        "phosphorus_atom_index_0based": phosphorus,
        "surface_oxygen_atom_index_0based": target_oxygen,
        "sealed_audit_id_convention": sealed_audit_id_convention,
        "cutoff_A": float(parameters["p_o_coordination_cutoff_A"]),
        "initial_distance_A": initial_target_distance,
        "final_distance_A": final_target_distance,
        "initial_coordination": initial_coordination,
        "final_coordination": final_coordination,
        "initial_intramolecular_coordination": initial_coordination["intramolecular_coordination"],
        "initial_surface_o_contacts": initial_coordination["surface_o_contacts"],
        "initial_total_coordination": initial_coordination["total_coordination"],
        "final_intramolecular_coordination": final_coordination["intramolecular_coordination"],
        "final_surface_o_contacts": final_coordination["surface_o_contacts"],
        "final_total_coordination": final_coordination["total_coordination"],
    }
    checks["target_p_o_measurement_valid"] = (
        initial[target_oxygen].symbol == "O"
        and final[target_oxygen].symbol == "O"
        and np.isfinite([initial_target_distance, final_target_distance]).all()
    )
    checks["intramolecular_p_coordination_contract"] = (
        initial_coordination["intramolecular_coordination"] == 4
        and final_coordination["intramolecular_coordination"] == 4
        and len(initial_coordination["intramolecular_p_c_contacts"]) == 1
        and len(initial_coordination["intramolecular_p_o_contacts"]) == 3
        and len(final_coordination["intramolecular_p_c_contacts"]) == 1
        and len(final_coordination["intramolecular_p_o_contacts"]) == 3
    )
    if not checks["target_p_o_measurement_valid"]:
        reasons.append("target_p_o_measurement_invalid")
    if not checks["intramolecular_p_coordination_contract"]:
        reasons.append("intramolecular_p_coordination_contract_failed")

    initial_contacts = _targeted_interface_covalent_contacts(
        initial,
        substrate_indices,
        molecule_indices,
        radius_scale=float(parameters["covalent_radius_scale"]),
        periodic_axes=periodic_axes,
    )
    final_contacts = _targeted_interface_covalent_contacts(
        final,
        substrate_indices,
        molecule_indices,
        radius_scale=float(parameters["covalent_radius_scale"]),
        periodic_axes=periodic_axes,
    )
    new_keys = sorted(set(final_contacts) - set(initial_contacts))
    lost_keys = sorted(set(initial_contacts) - set(final_contacts))
    mapped_pairs = {
        (int(item["working_metal_atom_index_0based"]), molecule_start + int(item["molecule_atom_index_0based"]))
        for item in system["donor_mapping"]
    }
    checks["interface_contact_contract_valid"] = all(
        np.isfinite(float(item["distance_A"]))
        and np.isfinite(float(item["cutoff_A"]))
        for item in list(initial_contacts.values()) + list(final_contacts.values())
    )
    checks["registered_interface_contacts_not_lost"] = not any(
        (pair[0], pair[1]) in {(s, m) for s, m in mapped_pairs}
        for pair in lost_keys
    )
    if not checks["interface_contact_contract_valid"]:
        reasons.append("interface_covalent_contact_contract_invalid")
    if not checks["registered_interface_contacts_not_lost"]:
        reasons.append("registered_interface_contact_lost")

    if initial_strict_vdw is None:
        initial_strict_vdw = _targeted_strict_vdw_audit(
            initial, system["donor_mapping"], surface_frame
        )
    if final_strict_vdw is None:
        final_strict_vdw = _targeted_strict_vdw_audit(
            final, system["donor_mapping"], surface_frame
        )
    initial_strict_summary = _targeted_audit_summary(initial_strict_vdw)
    final_strict_summary = _targeted_audit_summary(final_strict_vdw)
    initial_residual = _targeted_calibration_residual_nontarget_audit(
        initial_strict_vdw, target_p_local, target_oxygen
    )
    final_residual = _targeted_calibration_residual_nontarget_audit(
        final_strict_vdw, target_p_local, target_oxygen
    )
    # Raw production strict-vdW remains a diagnostic at the initial state and
    # remains an independent production gate at the endpoint.  Neither raw
    # result is a classification hard gate; only the final residual-nontarget
    # audit is a calibration hard gate.
    checks["initial_strict_vdw_gate"] = bool(initial_strict_vdw.get("passed", False))
    checks["final_strict_vdw_gate"] = bool(final_strict_vdw.get("passed", False))
    checks["calibration_residual_nontarget_audit_pass"] = bool(final_residual["passed"])
    if not final_residual["passed"]:
        reasons.append("final_calibration_residual_nontarget_short_contact")
    diagnostic_reasons = []
    if not checks["initial_strict_vdw_gate"]:
        diagnostic_reasons.append("initial_raw_production_strict_vdw_failed_diagnostic_only")
    if not checks["final_strict_vdw_gate"]:
        diagnostic_reasons.append("final_raw_production_strict_vdw_failed")

    try:
        forces_array = np.asarray(forces, dtype=float)
        movable_force_values = np.linalg.norm(forces_array[movable], axis=1)
        maximum_movable_force = float(np.max(movable_force_values))
        force_valid = bool(np.isfinite(maximum_movable_force))
    except (TypeError, ValueError, IndexError):
        maximum_movable_force = float("nan")
        force_valid = False
    steps = int(optimizer_steps)
    checks["optimizer_step_limit"] = 0 <= steps <= int(parameters["max_steps"])
    checks["optimizer_converged"] = bool(optimizer_converged)
    checks["maximum_movable_force_within_fmax"] = force_valid and maximum_movable_force <= float(parameters["fmax_eV_per_A"])
    if not checks["optimizer_step_limit"]:
        reasons.append("optimizer_step_limit_failed")
    if not checks["optimizer_converged"]:
        reasons.append("optimizer_not_converged")
    if not checks["maximum_movable_force_within_fmax"]:
        reasons.append("maximum_movable_force_above_fmax")

    top_metal_displacements = []
    for mapping in system["donor_mapping"]:
        index = int(mapping["working_metal_atom_index_0based"])
        displacement = _targeted_displacement_A(initial, final, [index], periodic_axes=periodic_axes)
        top_metal_displacements.append({**mapping, **displacement})
    target_oxygen_displacement = _targeted_displacement_A(initial, final, [target_oxygen], periodic_axes=periodic_axes)
    hard_gate_exclusions = {"initial_strict_vdw_gate", "final_strict_vdw_gate"}
    hard_gates = {
        key: bool(value)
        for key, value in checks.items()
        if key not in hard_gate_exclusions
    }
    classification = classify_targeted_calibration(
        bool(optimizer_converged) and checks["maximum_movable_force_within_fmax"],
        hard_gates,
        final_target_distance,
        initial_coordination,
        final_coordination,
        reasons=reasons,
    )
    production_pass = bool(final_strict_vdw.get("passed", False)) and all(hard_gates.values())
    return {
        "schema": "sam-targeted-calibration-validation-v2",
        "schema_version": TARGETED_CALIBRATION_SCHEMA_VERSION,
        "sealed_audit_id_convention": sealed_audit_id_convention,
        "passed": bool(classification["classification"] != "failed_or_unknown"),
        "production_pass": production_pass,
        "classification": classification["classification"],
        "classification_detail": classification,
        "hard_gates": hard_gates,
        "checks": checks,
        "diagnostic_reasons": sorted(set(diagnostic_reasons)),
        "reasons": sorted(set(reasons)),
        "optimizer": {
            "steps": steps,
            "max_steps": int(parameters["max_steps"]),
            "converged": bool(optimizer_converged),
            "maximum_movable_force_eV_per_A": maximum_movable_force,
            "fmax_eV_per_A": float(parameters["fmax_eV_per_A"]),
            "movable_atom_count": len(movable),
        },
        "identity": {
            "atom_count": len(final),
            "symbols_match": checks["symbol_identity_unchanged"],
            "cell_max_abs_change_A": float(np.max(np.abs(np.asarray(final.cell) - np.asarray(initial.cell)))),
            "frozen_lower_substrate": frozen,
            "movable_substrate": top_substrate,
            "movable_atom_ids_0based": movable,
            "frozen_max_displacement_A": frozen_displacement,
        },
        "sam_connectivity": {
            "covalent_radius_scale": float(parameters["covalent_radius_scale"]),
            "radii_source": "ASE ase.data.covalent_radii",
            "initial_edge_count": len(initial_edges),
            "final_edge_count": len(final_edges),
            "initial_edges_global_1based": initial_edges,
            "final_edges_global_1based": final_edges,
        },
        "surface_h": proton_records,
        "registered_donor_contacts": donor_records,
        "target_p_o": target_p_o,
        "coordination_semantics": {
            "definition": "intramolecular_P-C_plus_three_P-O_plus_surface_O_within_cutoff",
            "cutoff_A": float(parameters["p_o_coordination_cutoff_A"]),
            "initial": {
                "intramolecular_coordination": initial_coordination["intramolecular_coordination"],
                "surface_o_contacts": initial_coordination["surface_o_contacts"],
                "total_coordination": initial_coordination["total_coordination"],
            },
            "final": {
                "intramolecular_coordination": final_coordination["intramolecular_coordination"],
                "surface_o_contacts": final_coordination["surface_o_contacts"],
                "total_coordination": final_coordination["total_coordination"],
            },
        },
        "interface_covalent_contacts": {
            "contract": {
                "radii_source": "ASE ase.data.covalent_radii",
                "radius_scale": float(parameters["covalent_radius_scale"]),
                "scope": "working_substrate_atoms_vs_selected_SAM_molecule",
                "periodic_axes": list(periodic_axes),
                "normal_axis_wrapped": False,
            },
            "initial_count": len(initial_contacts),
            "final_count": len(final_contacts),
            "new_contacts": [final_contacts[key] for key in new_keys],
            "lost_contacts": [initial_contacts[key] for key in lost_keys],
        },
        "strict_vdw": {
            "contract": {
                "audit": "sam_structure_tools.periodic_vdw_collision_audit",
                "radius_scale": float(TARGETED_CALIBRATION_VDW_RADIUS_SCALE),
                "mapped_bonds_only_exact_exemptions": True,
                "target_pair_not_exempted": True,
                "periodic_axes": list(periodic_axes),
                "normal_axis_wrapped": False,
            },
            "initial_raw_production": initial_strict_vdw,
            "final_raw_production": final_strict_vdw,
            "initial": initial_strict_summary,
            "final": final_strict_summary,
            "initial_calibration_residual_nontarget": initial_residual,
            "final_calibration_residual_nontarget": final_residual,
        },
        "top_surface_displacement": top_displacement,
        "target_surface_oxygen_displacement": target_oxygen_displacement,
        "registered_metal_displacements": top_metal_displacements,
    }


def _targeted_find_hash_file(root, expected_sha256, suffixes=(".cif", ".extxyz", ".xyz")):
    root = _targeted_resolve_path(root)
    # Prefer the maintained substrate library namespace.  Historical run
    # packages may contain exact byte copies of that source; those copies are
    # evidence, not a second source identity.
    for search_root in (root / "substrates", root / "runs"):
        matches = []
        if not search_root.exists():
            continue
        for suffix in suffixes:
            for candidate in search_root.rglob(f"*{suffix}"):
                if candidate.is_file() and not candidate.is_symlink():
                    try:
                        if _sha256(candidate) == expected_sha256:
                            matches.append(candidate.resolve())
                    except OSError:
                        continue
        unique = sorted(set(matches), key=str)
        if unique:
            if len(unique) != 1:
                raise ValueError(
                    f"hash-verified source surface resolution is ambiguous in {search_root}: {len(unique)} files"
                )
            return unique[0]
    raise ValueError("hash-verified source surface could not be resolved without downloading")


def _targeted_trajectory_directory(path):
    path = _targeted_resolve_path(path)
    if path.is_file():
        if path.name != "final-h0.extxyz":
            raise ValueError("targeted trajectory input must be a trajectory directory or final-h0.extxyz")
        return path.parent
    if not path.is_dir():
        raise ValueError(f"sealed trajectory path is missing: {path}")
    if path.name.startswith("trajectory-"):
        return path
    candidates = sorted(
        item for item in path.glob("trajectory-*") if item.is_dir()
    )
    if len(candidates) != 1:
        raise ValueError("targeted calibration requires exactly one sealed trajectory")
    return candidates[0].resolve()


def _targeted_get_top_input(top_manifest, key, override, top_root):
    inputs = top_manifest.get("inputs") or {}
    if key not in inputs:
        raise ValueError(f"sealed sequential-growth top manifest lacks input {key}")
    declared = inputs[key]
    expected = declared.get("sha256") if isinstance(declared, dict) else None
    raw_path = override if override is not None else (
        declared.get("path") if isinstance(declared, dict) else declared
    )
    if raw_path is None:
        raise ValueError(f"sealed sequential-growth input {key} has no path")
    record = _targeted_file_record(
        _targeted_resolve_path(raw_path, top_root), expected_sha256=expected, label=key
    )
    if isinstance(declared, dict) and declared.get("bytes") is not None and int(declared["bytes"]) != record["bytes"]:
        raise ValueError(f"sealed input byte count changed for {key}")
    return record


def _targeted_validate_analysis_and_rank(run_request, analysis_record):
    records = run_request.get("selected_conformer_analysis_records")
    if not isinstance(records, list):
        raise ValueError("sealed run-request lacks selected conformer analysis records")
    matches = [
        item for item in records
        if int(item.get("selection_rank", -1)) == 1
    ]
    if len(matches) != 1:
        raise ValueError("sealed run-request does not have one rank-1 conformer record")
    item = matches[0]
    analysis = item.get("analysis_record") or {}
    if int(item.get("cluster_id", -1)) != 15 or int(analysis.get("cluster_id", -1)) != 15:
        raise ValueError("target conformer is not cluster 15")
    if int(analysis.get("representative_task_index", -1)) != 16:
        raise ValueError("target conformer is not task 16")
    structure = item.get("structure") or analysis.get("structure") or {}
    if not structure.get("path") or not structure.get("sha256"):
        raise ValueError("rank-1 conformer source hash record is incomplete")
    lock = (run_request.get("selected_conformer_source_lock") or {}).get("selected_source_files", [])
    lock_matches = [
        record for record in lock if int(record.get("selection_rank", -1)) == 1
    ]
    if len(lock_matches) != 1:
        raise ValueError("sealed run-request lacks one rank-1 source lock")
    lock_record = lock_matches[0]
    if lock_record.get("sha256") != structure.get("sha256") or lock_record.get("path") != structure.get("path"):
        raise ValueError("rank-1 analysis/source-lock records disagree")
    return item, structure


def _targeted_audit_id_convention_failure(detail):
    expected = TARGETED_CALIBRATION_SEALED_AUDIT_ID_CONVENTION
    return ValueError(
        "target surface oxygen resolution fail-closed: sealed audit ID convention "
        "mismatch; expected "
        f"{expected['canonical']} under {expected['audit_schema']}; {detail}"
    )


def _targeted_validate_sealed_audit_id_convention(
    final_interface_audit, molecule_audit, molecule_number_1based
):
    """Validate the ID scopes emitted by the sealed H0 interface audit."""

    expected = TARGETED_CALIBRATION_SEALED_AUDIT_ID_CONVENTION
    if not isinstance(final_interface_audit, dict):
        raise _targeted_audit_id_convention_failure(
            "final_interface_audit is missing or is not an object"
        )
    if final_interface_audit.get("schema") != expected["audit_schema"]:
        raise _targeted_audit_id_convention_failure(
            f"observed audit schema={final_interface_audit.get('schema')!r}"
        )
    try:
        substrate_count = int(final_interface_audit["substrate_atom_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise _targeted_audit_id_convention_failure(
            "sealed audit lacks an integer substrate_atom_count"
        ) from exc
    if substrate_count != TARGETED_CALIBRATION_SUBSTRATE_ATOMS:
        raise _targeted_audit_id_convention_failure(
            f"observed substrate_atom_count={substrate_count}, expected "
            f"{TARGETED_CALIBRATION_SUBSTRATE_ATOMS}"
        )

    declared = next(
        (
            final_interface_audit.get(key)
            for key in ("id_convention", "atom_id_convention", "audit_id_convention")
            if final_interface_audit.get(key) is not None
        ),
        None,
    )
    if declared is not None:
        if isinstance(declared, str):
            declared_matches = declared == expected["canonical"]
        elif isinstance(declared, dict):
            declared_matches = all(
                declared.get(field) in {expected[field], expected[field].replace(" index", "")}
                for field in ("molecule_atom_id", "substrate_atom_id")
            )
        else:
            declared_matches = False
        if not declared_matches:
            raise _targeted_audit_id_convention_failure(
                f"observed declared convention={declared!r}"
            )

    try:
        molecule_number = int(molecule_number_1based)
    except (TypeError, ValueError) as exc:
        raise _targeted_audit_id_convention_failure(
            f"molecule number is not an integer: {molecule_number_1based!r}"
        ) from exc
    expected_range = [
        TARGETED_CALIBRATION_SUBSTRATE_ATOMS
        + (molecule_number - 1) * TARGETED_CALIBRATION_MOLECULE_ATOMS,
        TARGETED_CALIBRATION_SUBSTRATE_ATOMS
        + molecule_number * TARGETED_CALIBRATION_MOLECULE_ATOMS
        - 1,
    ]
    observed_range = molecule_audit.get("global_atom_range_0based")
    if observed_range != expected_range:
        raise _targeted_audit_id_convention_failure(
            f"molecule {molecule_number} global_atom_range_0based="
            f"{observed_range!r}, expected {expected_range!r}"
        )

    for collection_name in ("collisions", "mapped_bonds"):
        records = molecule_audit.get(collection_name) or []
        if not isinstance(records, list):
            raise _targeted_audit_id_convention_failure(
                f"molecule audit {collection_name} is not a list"
            )
        for record_index, record in enumerate(records):
            if not isinstance(record, dict):
                raise _targeted_audit_id_convention_failure(
                    f"molecule audit {collection_name}[{record_index}] is not an object"
                )
            for field, upper, scope in (
                (
                    "molecule_atom_id",
                    TARGETED_CALIBRATION_MOLECULE_ATOMS,
                    expected["molecule_atom_id"],
                ),
                (
                    "substrate_atom_id",
                    TARGETED_CALIBRATION_SUBSTRATE_ATOMS,
                    expected["substrate_atom_id"],
                ),
            ):
                value = record.get(field)
                if isinstance(value, bool) or not isinstance(value, int):
                    raise _targeted_audit_id_convention_failure(
                        f"{collection_name}[{record_index}].{field}={value!r} is not an integer in the "
                        f"{scope} scheme"
                    )
                if not 0 <= value < upper:
                    raise _targeted_audit_id_convention_failure(
                        f"{collection_name}[{record_index}].{field}={value} is outside the "
                        f"{scope} range [0, {upper})"
                    )
    return dict(expected)


def _targeted_resolve_target_surface_oxygen(
    validation, site_instance_id, molecule_number_1based, target_p_molecule_local_index_0based
):
    """Resolve the calibration target from the sealed production audit record."""

    final_interface_audit = (validation or {}).get("final_interface_audit")
    audits = (final_interface_audit or {}).get("molecule_audits") or []
    try:
        molecule_number = int(molecule_number_1based)
    except (TypeError, ValueError) as exc:
        raise _targeted_audit_id_convention_failure(
            f"molecule number is not an integer: {molecule_number_1based!r}"
        ) from exc
    matching = [
        item for item in audits
        if isinstance(item, dict)
        and item.get("molecule_number_1based") == molecule_number
        and item.get("site_instance_id") == site_instance_id
    ]
    if len(matching) != 1:
        raise ValueError(
            "target surface oxygen resolution fail-closed: sealed production validation "
            "lacks exactly one target molecule audit under the expected sealed audit ID "
            "convention (molecule_number_1based is a 1-based placement ordinal)"
        )
    audit = matching[0]
    convention = _targeted_validate_sealed_audit_id_convention(
        final_interface_audit, audit, molecule_number
    )
    try:
        target_p_local = int(target_p_molecule_local_index_0based)
    except (TypeError, ValueError) as exc:
        raise _targeted_audit_id_convention_failure(
            f"target P local index is not an integer: {target_p_molecule_local_index_0based!r}"
        ) from exc
    if not 0 <= target_p_local < TARGETED_CALIBRATION_MOLECULE_ATOMS:
        raise _targeted_audit_id_convention_failure(
            f"target P index {target_p_local} is not molecule-local 0-based"
        )
    candidates = []
    for record in audit.get("collisions") or []:
        if not isinstance(record, dict):
            continue
        if record.get("molecule_atom_id") != target_p_local:
            continue
        try:
            distance = float(record.get("distance_A", float("nan")))
        except (TypeError, ValueError) as exc:
            raise _targeted_audit_id_convention_failure(
                f"target collision has a non-numeric distance: {record.get('distance_A')!r}"
            ) from exc
        if (
            record.get("molecule_element") == "P"
            and record.get("substrate_element") == "O"
            and abs(distance - TARGETED_CALIBRATION_TARGET_P_O_REFERENCE_A)
            <= TARGETED_CALIBRATION_TARGET_P_O_REFERENCE_TOLERANCE_A
        ):
            candidates.append(record)
    if len(candidates) != 1:
        raise ValueError(
            "target surface oxygen resolution fail-closed: sealed production validation "
            "does not contain exactly one target P-surface-O contact under the verified "
            f"sealed audit ID convention ({convention['canonical']})"
        )
    target = dict(candidates[0])
    target_index = int(target["substrate_atom_id"])
    target["sealed_audit_id_convention"] = convention
    target["sealed_audit_global_molecule_range_0based"] = list(
        audit["global_atom_range_0based"]
    )
    return target_index, target


def _targeted_resolve_source_surface(root, prototype_library, explicit_path):
    expected = (prototype_library.get("source_surface") or {}).get("sha256")
    if not expected:
        raise ValueError("accepted prototype library lacks source-surface hash")
    if explicit_path is None:
        path = _targeted_find_hash_file(root, expected, suffixes=(".cif",))
    else:
        path = _targeted_resolve_path(explicit_path)
    return _targeted_file_record(path, expected_sha256=expected, label="accepted source surface")


def _targeted_validate_layout_and_formula(final_h0, substrate_input, run_request, child_manifest):
    substrate_count = TARGETED_CALIBRATION_SUBSTRATE_ATOMS
    block = TARGETED_CALIBRATION_MOLECULE_ATOMS
    molecule_count = int(child_manifest.get("molecule_count_N", -1))
    if molecule_count != TARGETED_CALIBRATION_PARENT_MOLECULE_COUNT:
        raise ValueError("sealed trajectory molecule count is not 68")
    if len(final_h0) != substrate_count + molecule_count * block:
        raise ValueError("sealed final-h0 atom count does not match substrate plus molecule blocks")
    if len(substrate_input) not in {substrate_count, 3840}:
        raise ValueError("substrate input must be the exact 2560 working slab or its 3840-atom parent")
    if len(substrate_input) == substrate_count:
        if final_h0[:substrate_count].get_chemical_symbols() != substrate_input.get_chemical_symbols():
            raise ValueError("sealed final-h0 substrate symbols differ from substrate input")
        if not np.allclose(final_h0[:substrate_count].positions, substrate_input.positions, atol=1.0e-8, rtol=0.0):
            raise ValueError("sealed final-h0 substrate coordinates differ from substrate input")
    if not np.allclose(final_h0.cell, substrate_input.cell, atol=1.0e-8, rtol=0.0):
        raise ValueError("sealed final-h0 cell differs from substrate input")
    expected_formula = run_request.get("molecule_formula")
    if expected_formula is None:
        expected_formula = ((run_request.get("protocol") or {}).get("molecule_contract") or {}).get("formula")
    if expected_formula is not None and dict(expected_formula) != TARGETED_CALIBRATION_MOLECULE_FORMULA:
        raise ValueError("sealed run-request molecule formula is not DBF34 C22H18NO3PS")
    for index in range(molecule_count):
        block_atoms = final_h0[substrate_count + index * block:substrate_count + (index + 1) * block]
        if _composition(block_atoms) != TARGETED_CALIBRATION_MOLECULE_FORMULA:
            raise ValueError(f"sealed molecule block {index + 1} formula changed")
    if "H" in _composition(final_h0[:substrate_count]):
        raise ValueError("sealed H0 working substrate contains surface H")


def _targeted_validate_layer_groups(layer_groups, substrate_count):
    working = [int(value) for value in layer_groups.get("working_layer_atom_ids_1based", [])]
    restore = [int(value) for value in layer_groups.get("restore_layer_atom_ids_1based", [])]
    if len(working) != substrate_count or len(restore) != 1280:
        raise ValueError("layer-group evidence is not 2560 working plus 1280 restore atoms")
    if len(set(working)) != len(working) or len(set(restore)) != len(restore):
        raise ValueError("layer-group atom IDs are not individually unique")
    if set(working) & set(restore) or set(working) | set(restore) != set(range(1, 3841)):
        raise ValueError("layer-group atom IDs do not partition the 3840-atom substrate")
    normal_axis = layer_groups.get("normal_axis")
    if normal_axis is None or int(normal_axis) not in range(3):
        raise ValueError("layer-group evidence lacks a valid registered normal axis")
    layers = layer_groups.get("layers_bottom_to_top") or []
    if len(layers) != 3 or any(len(layer.get("atom_ids_1based", [])) != 1280 for layer in layers):
        raise ValueError("targeted calibration requires three complete 1280-atom layers")
    expected_restore = [int(value) for value in layers[0]["atom_ids_1based"]]
    expected_working = [
        int(value) for value in layers[1]["atom_ids_1based"] + layers[2]["atom_ids_1based"]
    ]
    if restore != expected_restore or working != expected_working:
        raise ValueError("working/frozen layer order is not the registered lower-two/top-layer mapping")
    return {
        "working_layer_atom_ids_1based": working,
        "restore_layer_atom_ids_1based": restore,
        "normal_axis": int(normal_axis),
    }


def resolve_targeted_calibration_plan(
    trajectory,
    molecule_index_1based,
    model,
    output,
    *,
    site_instances=None,
    prototype_library=None,
    source_surface=None,
    model_elements=TARGETED_CALIBRATION_ELEMENTS,
    root=None,
):
    """Resolve one sealed sequential-growth molecule into a CPU-only plan.

    This function deliberately performs no MACE, CUDA, optimizer, download, or
    production-gate mutation.  All evidence is verified before the new output
    directory is reserved; any mismatch raises and creates no output package.
    """

    if root is None:
        candidates = [Path(__file__).resolve().parents[1]]
        candidates.extend(Path(__file__).resolve().parents[index] for index in (3, 4) if len(Path(__file__).resolve().parents) > index)
        root = next((candidate for candidate in candidates if (candidate / "substrates").is_dir()), candidates[0])
    root = _targeted_resolve_path(root)
    molecule_index_1based = int(molecule_index_1based)
    if molecule_index_1based != TARGETED_CALIBRATION_MOLECULE_INDEX_1BASED:
        raise ValueError("targeted calibration evidence is restricted to molecule index 62")
    model_record = _targeted_file_record(model, label="MACE model")
    if model_record["sha256"] != TARGETED_CALIBRATION_MODEL_SHA256:
        raise ValueError("MACE model SHA-256 is not the required calibrated model")
    declared_elements = _targeted_parse_elements(model_elements)

    trajectory_dir = _targeted_trajectory_directory(trajectory)
    run_root = trajectory_dir.parent
    top_manifest_record = _targeted_file_record(run_root / "manifest.json", label="trajectory top manifest")
    child_manifest_record = _targeted_file_record(trajectory_dir / "manifest.json", label="trajectory child manifest")
    top_manifest = _targeted_read_json(top_manifest_record, "trajectory top manifest")
    child_manifest = _targeted_read_json(child_manifest_record, "trajectory child manifest")
    if top_manifest.get("schema") != "sam-sequential-growth-run-manifest-v1" or not bool(top_manifest.get("sealed")):
        raise ValueError("trajectory top manifest is not a sealed sequential-growth manifest")
    if int(top_manifest.get("schema_version", -1)) < 1 or not bool(child_manifest.get("sealed")):
        raise ValueError("trajectory child manifest is not sealed")
    if child_manifest.get("status") in {"running", "pending"}:
        raise ValueError("trajectory child manifest is unfinished")
    if child_manifest.get("path_base") not in {None, "trajectory_root"}:
        raise ValueError("unsupported trajectory child path base")

    request_spec = top_manifest.get("run_request") or {"path": "run-request.json"}
    request_record = _targeted_declared_file(request_spec, run_root, "sealed run-request")
    run_request = _targeted_read_json(request_record, "sealed run-request")
    if run_request.get("schema") != "sam-sequential-growth-run-request-v1" or run_request.get("status") != "sealed_pre_expensive_request":
        raise ValueError("run-request is not the sealed sequential-growth request")

    final_record = _targeted_declared_file(
        child_manifest.get("final_structure"), trajectory_dir, "sealed final-h0", default_name="final-h0.extxyz"
    )
    child_inventory = {
        item.get("path"): item
        for item in child_manifest.get("artifact_inventory_before_manifest_seal", [])
        if isinstance(item, dict) and item.get("path")
    }
    bonds_spec = child_inventory.get("registered-interface-bonds.json") or {"path": "registered-interface-bonds.json"}
    validation_spec = child_inventory.get("validation.json") or {"path": "validation.json"}
    bonds_record = _targeted_declared_file(
        bonds_spec, trajectory_dir, "sealed registered-interface-bonds"
    )
    validation_record = _targeted_declared_file(
        validation_spec, trajectory_dir, "sealed H0 validation"
    )
    final_h0 = read(final_record["path"])
    bonds = _targeted_read_json(bonds_record, "registered interface bonds")
    validation = _targeted_read_json(validation_record, "sealed H0 validation")
    if bonds.get("sealed") is not True or bonds.get("schema") != "sam-sequential-registered-interface-bonds-v1":
        raise ValueError("registered interface-bonds evidence is not sealed")
    if validation.get("sealed") is not True:
        raise ValueError("sealed H0 validation evidence is not sealed")

    substrate_record = _targeted_get_top_input(top_manifest, "substrate", None, run_root)
    layer_record = _targeted_get_top_input(top_manifest, "layer_groups", None, run_root)
    site_record = _targeted_get_top_input(top_manifest, "site_instances", site_instances, run_root)
    prototype_record = _targeted_get_top_input(top_manifest, "site_prototypes", prototype_library, run_root)
    analysis_record = _targeted_get_top_input(top_manifest, "conformer_analysis", None, run_root)
    substrate_input = read(substrate_record["path"])
    layer_groups = _targeted_read_json(layer_record, "layer groups")
    site_payload = _targeted_read_json(site_record, "site instances")
    prototype_payload = _targeted_read_json(prototype_record, "accepted prototype library")
    analysis_payload = _targeted_read_json(analysis_record, "conformer analysis")
    _targeted_validate_layout_and_formula(final_h0, substrate_input, run_request, child_manifest)
    layer_identity = _targeted_validate_layer_groups(layer_groups, TARGETED_CALIBRATION_SUBSTRATE_ATOMS)
    if len(substrate_input) == 3840:
        working_substrate = substrate_input[[atom_id - 1 for atom_id in layer_identity["working_layer_atom_ids_1based"]]]
    else:
        working_substrate = substrate_input.copy()
    if final_h0[:TARGETED_CALIBRATION_SUBSTRATE_ATOMS].get_chemical_symbols() != working_substrate.get_chemical_symbols():
        raise ValueError("sealed final-h0 working substrate symbols differ from layer mapping")
    if _composition(final_h0[:TARGETED_CALIBRATION_SUBSTRATE_ATOMS]) != _composition(working_substrate):
        raise ValueError("sealed final-h0 working substrate composition differs from layer mapping")

    steps = child_manifest.get("steps")
    if not isinstance(steps, list) or len(steps) < molecule_index_1based:
        raise ValueError("sealed trajectory has no step for molecule 62")
    step_record = steps[molecule_index_1based - 1]
    if int(step_record.get("step", -1)) != molecule_index_1based:
        raise ValueError("trajectory step number does not match molecule index")
    if step_record.get("site_instance_id") != TARGETED_CALIBRATION_SITE_INSTANCE_ID:
        raise ValueError("molecule 62 was not accepted at the required Site Instance")
    if step_record.get("site_prototype_id") != TARGETED_CALIBRATION_PROTOTYPE_ID:
        raise ValueError("molecule 62 prototype identity changed")
    if int(step_record.get("selection_rank", -1)) != 1 or int(step_record.get("cluster_id", -1)) != 15:
        raise ValueError("molecule 62 is not rank 1 / cluster 15")
    if step_record.get("source", "").split("/")[-1] != "selection-0001-cluster-0015-task-0016.extxyz":
        raise ValueError("molecule 62 source is not rank-1 cluster-15 task-16")
    if step_record.get("oxygen_permutation") != TARGETED_CALIBRATION_OXYGEN_PERMUTATION_0BASED:
        raise ValueError("molecule 62 donor permutation changed")

    if site_payload.get("status") not in {
        "passed_fixed_coverage_site_selection",
        "passed_site_instance_materialization",
        "accepted",
    }:
        raise ValueError("Site Instance evidence is not an accepted materialization package")
    site_matches = [
        item for item in site_payload.get("site_instances", [])
        if item.get("site_instance_id") == TARGETED_CALIBRATION_SITE_INSTANCE_ID
    ]
    if len(site_matches) != 1:
        raise ValueError("accepted Site Instance evidence is missing or ambiguous")
    site_instance = site_matches[0]
    if prototype_payload.get("schema_version") != 1 or not isinstance(prototype_payload.get("site_prototypes"), list):
        raise ValueError("accepted prototype library schema is invalid")
    prototype_matches = [
        item for item in prototype_payload["site_prototypes"]
        if item.get("site_prototype_id") == TARGETED_CALIBRATION_PROTOTYPE_ID
    ]
    if len(prototype_matches) != 1:
        raise ValueError("accepted prototype evidence is missing or ambiguous")
    prototype = prototype_matches[0]
    if prototype.get("anchor_family") != "phosphonic_acid" or prototype.get("final_denticity") != 3:
        raise ValueError("accepted prototype topology is not tridentate phosphonic acid")
    surface_frame = _targeted_surface_frame_from_evidence(
        prototype=prototype,
        site_instance=site_instance,
        site_plan=site_payload,
        layer_groups=layer_groups,
    )

    donor_mapping = resolve_targeted_donor_mapping(
        step_record,
        site_instance,
        layer_identity["working_layer_atom_ids_1based"],
        working_substrate,
    )
    source_surface_record = _targeted_resolve_source_surface(
        root, prototype_payload, source_surface
    )
    source_surface_atoms = read(source_surface_record["path"])
    build_manifest_path = Path(site_record["path"]).parent / "substrate-build-manifest.json"
    build_manifest_record = None
    build_manifest = None
    if build_manifest_path.is_file() and not build_manifest_path.is_symlink():
        build_manifest_record = _targeted_file_record(build_manifest_path, label="substrate build manifest")
        build_manifest = _targeted_read_json(build_manifest_record, "substrate build manifest")
    matrix = _targeted_supercell_matrix(source_surface_atoms, final_h0.cell, build_manifest)
    from ase.build import make_supercell
    full_supercell = make_supercell(source_surface_atoms, matrix, wrap=True)
    expected_full_substrate_atoms = len(layer_identity["working_layer_atom_ids_1based"]) + len(
        layer_identity["restore_layer_atom_ids_1based"]
    )
    if len(full_supercell) != expected_full_substrate_atoms:
        raise ValueError("source->supercell mapping did not reconstruct the sealed substrate")
    if source_surface_record["sha256"] != (prototype_payload.get("source_surface") or {}).get("sha256"):
        raise ValueError("accepted prototype/source surface hash boundary changed")
    source_translation = site_instance.get("source_cell_translation")
    if source_translation is None:
        raise ValueError("accepted Site Instance lacks source-cell translation")
    prototype_vertices = prototype.get("metal_vertices") or []
    prototype_donor_mapping = prototype.get("donor_metal_mapping") or []
    mapped_prototype_metals = {}
    for vertex_index, vertex in enumerate(prototype_vertices):
        source_id = int(vertex.get("source_atom_id", -1))
        image = list(vertex.get("lattice_image", [0, 0]))
        if len(image) != 2:
            raise ValueError("accepted prototype metal lattice image is invalid")
        translation = np.asarray(source_translation if source_translation is not None else site_instance.get("source_cell_translation"), dtype=int)
        combined_translation = [int(translation[0]) + int(image[0]), int(translation[1]) + int(image[1]), int(translation[2])]
        mapped_index = _targeted_source_to_supercell_index(
            source_surface_atoms, full_supercell, source_id, combined_translation
        )
        mapped_prototype_metals[vertex_index] = mapped_index + 1
        if str(vertex.get("element")) != full_supercell[mapped_index].symbol:
            raise ValueError("accepted prototype metal/source-supercell element mismatch")
    if not prototype_donor_mapping or len(mapped_prototype_metals) != 3:
        raise ValueError("accepted prototype donor/source-supercell mapping is incomplete")
    instance_donor_by_label = {
        item.get("probe_donor_id"): int(item.get("metal_atom_id_1based", -1))
        for item in site_instance.get("donor_metal_mapping", [])
    }
    for mapping in prototype_donor_mapping:
        label = mapping.get("probe_donor_id")
        vertex_index = int(mapping.get("metal_vertex_index", -1))
        if instance_donor_by_label.get(label) != mapped_prototype_metals.get(vertex_index):
            raise ValueError("accepted prototype/source-supercell donor mapping changed")

    representative_spec = prototype.get("representative_relaxed_structure") or {}
    representative_record = _targeted_declared_file(
        representative_spec,
        Path(prototype_record["path"]).parent,
        "prototype representative structure",
    )
    representative = read(representative_record["path"])
    proton_records = map_targeted_surface_protons(
        representative,
        prototype,
        source_surface_atoms,
        full_supercell,
        source_translation,
        layer_identity["working_layer_atom_ids_1based"],
        prototype_payload.get("source_surface"),
        surface_frame,
    )
    for record in proton_records:
        parent = int(record["working_parent_atom_index_0based"])
        record["parent_position_A"] = np.asarray(final_h0.positions[parent], dtype=float).tolist()
        record["mapped_surface_h_position_A"] = (
            np.asarray(final_h0.positions[parent], dtype=float)
            + np.asarray(record["parent_to_h_vector_A"], dtype=float)
        ).tolist()

    rank_item, rank_structure_spec = _targeted_validate_analysis_and_rank(
        run_request, analysis_payload
    )
    analysis_selected_rank1 = [
        item for item in analysis_payload.get("selection", {}).get("selected", [])
        if int(item.get("selection_rank", -1)) == 1
    ]
    if len(analysis_selected_rank1) != 1 or int(analysis_selected_rank1[0].get("cluster_id", -1)) != 15:
        raise ValueError("conformer analysis evidence does not retain rank-1 cluster 15")
    analysis_structure = analysis_selected_rank1[0].get("structure") or {}
    if analysis_structure.get("sha256") != rank_structure_spec.get("sha256"):
        raise ValueError("rank-1 conformer analysis/source hash mismatch")
    rank_source_record = _targeted_file_record(
        rank_structure_spec["path"],
        expected_sha256=rank_structure_spec["sha256"],
        label="rank-1 cluster-15 task-16 conformer source",
    )
    lock = (run_request.get("selected_conformer_source_lock") or {}).get("selected_source_files", [])
    if int(lock[0].get("selection_rank", -1)) != 1:
        raise ValueError("selected conformer source lock is not naturally ordered")
    rank_source = read(rank_source_record["path"])
    if len(rank_source) < TARGETED_CALIBRATION_MOLECULE_ATOMS or _composition(rank_source[-TARGETED_CALIBRATION_MOLECULE_ATOMS:]) != TARGETED_CALIBRATION_MOLECULE_FORMULA:
        raise ValueError("rank-1 source does not contain the DBF34 46-atom conformer")
    rank_molecule = rank_source[-TARGETED_CALIBRATION_MOLECULE_ATOMS:]
    rank_p_locals = [index for index, atom in enumerate(rank_molecule) if atom.symbol == "P"]
    if len(rank_p_locals) != 1:
        raise ValueError("rank-1 conformer does not contain one unambiguous P anchor")
    target_p_local = rank_p_locals[0]
    target_molecule_start = TARGETED_CALIBRATION_SUBSTRATE_ATOMS + (molecule_index_1based - 1) * TARGETED_CALIBRATION_MOLECULE_ATOMS
    target_molecule = final_h0[target_molecule_start:target_molecule_start + TARGETED_CALIBRATION_MOLECULE_ATOMS]
    if [index for index, atom in enumerate(target_molecule) if atom.symbol == "P"] != [target_p_local]:
        raise ValueError("sealed target molecule P identity differs from the selected conformer")
    target_o, target_contact_evidence = _targeted_resolve_target_surface_oxygen(
        validation,
        TARGETED_CALIBRATION_SITE_INSTANCE_ID,
        molecule_index_1based,
        target_p_local,
    )
    initial = build_targeted_calibration_initial_structure(
        final_h0,
        molecule_index_1based,
        proton_records,
    )
    initial_edges = _sam_edge_set(
        initial,
        range(TARGETED_CALIBRATION_SUBSTRATE_ATOMS, TARGETED_CALIBRATION_SUBSTRATE_ATOMS + TARGETED_CALIBRATION_MOLECULE_ATOMS),
        scale=TARGETED_CALIBRATION_COVALENT_RADIUS_SCALE,
    )
    target_p = TARGETED_CALIBRATION_SUBSTRATE_ATOMS + target_p_local
    periodic_axes = tuple(surface_frame["periodic_fractional_axes"])
    target_distance = _targeted_distance(initial, target_p, target_o, periodic_axes=periodic_axes)
    if initial[target_p].symbol != "P" or initial[target_o].symbol != target_contact_evidence.get("substrate_element"):
        raise ValueError("target P/surface-O atom identity is not sealed")
    if initial[target_o].symbol != "O":
        raise ValueError("sealed target surface atom is not O")
    if abs(target_distance - TARGETED_CALIBRATION_TARGET_P_O_REFERENCE_A) > TARGETED_CALIBRATION_TARGET_P_O_REFERENCE_TOLERANCE_A:
        raise ValueError("sealed target P-surface-O distance is not the approximately 1.772-A evidence")
    initial_coordination = targeted_p_o_coordination(
        initial,
        target_p,
        TARGETED_CALIBRATION_P_O_COORDINATION_CUTOFF_A,
        periodic_axes=periodic_axes,
        molecule_indices=range(TARGETED_CALIBRATION_SUBSTRATE_ATOMS, TARGETED_CALIBRATION_SUBSTRATE_ATOMS + TARGETED_CALIBRATION_MOLECULE_ATOMS),
        substrate_indices=range(TARGETED_CALIBRATION_SUBSTRATE_ATOMS),
    )
    initial_strict_vdw = _targeted_strict_vdw_audit(initial, donor_mapping, surface_frame)

    implementation_path = Path(__file__).resolve()
    mirror_path = root / ".agents" / "skills" / "sam-conformation-search" / "scripts" / implementation_path.name
    implementation_record = _targeted_file_record(implementation_path, label="canonical calibration script")
    mirror_record = _targeted_file_record(mirror_path, label="calibration script mirror")
    if implementation_record["sha256"] != mirror_record["sha256"]:
        raise ValueError("canonical calibration script and skill mirror differ")

    evidence = {
        "trajectory_top_manifest": top_manifest_record,
        "trajectory_child_manifest": child_manifest_record,
        "run_request": request_record,
        "final_h0": final_record,
        "registered_interface_bonds": bonds_record,
        "sealed_h0_validation": validation_record,
        "substrate": substrate_record,
        "layer_groups": layer_record,
        "site_instances": site_record,
        "prototype_library": prototype_record,
        "prototype_representative": representative_record,
        "source_surface": source_surface_record,
        "conformer_analysis": analysis_record,
        "rank1_cluster15_task16_conformer_source": rank_source_record,
        "substrate_build_manifest": build_manifest_record,
        "canonical_script": implementation_record,
        "script_mirror": mirror_record,
        "model": model_record,
    }
    evidence = {key: value for key, value in evidence.items() if value is not None}
    system = {
        "role": TARGETED_CALIBRATION_ROLE,
        "parent_run_role": "sealed_sequential_growth_h0_geometry_evidence",
        "substrate_atom_count": TARGETED_CALIBRATION_SUBSTRATE_ATOMS,
        "parent_molecule_count": TARGETED_CALIBRATION_PARENT_MOLECULE_COUNT,
        "target_molecule_index_1based": molecule_index_1based,
        "molecule_block_size": TARGETED_CALIBRATION_MOLECULE_ATOMS,
        "molecule_formula": TARGETED_CALIBRATION_MOLECULE_FORMULA,
        "expected_atom_count": TARGETED_CALIBRATION_TOTAL_ATOMS,
        "composition": _composition(initial),
        "working_substrate_composition": _composition(initial[:TARGETED_CALIBRATION_SUBSTRATE_ATOMS]),
        "molecule_atom_ids_0based": list(range(TARGETED_CALIBRATION_SUBSTRATE_ATOMS, TARGETED_CALIBRATION_SUBSTRATE_ATOMS + TARGETED_CALIBRATION_MOLECULE_ATOMS)),
        "surface_h_atom_ids_0based": list(range(TARGETED_CALIBRATION_TOTAL_ATOMS - 2, TARGETED_CALIBRATION_TOTAL_ATOMS)),
        "surface_proton_records": proton_records,
        "donor_mapping": donor_mapping,
        "target_p_molecule_local_index_0based": target_p_local,
        "target_surface_oxygen_atom_index_0based": target_o,
        "target_surface_oxygen_element": initial[target_o].symbol,
        "sealed_audit_id_convention": dict(
            TARGETED_CALIBRATION_SEALED_AUDIT_ID_CONVENTION
        ),
        "target_p_o_initial_distance_A": target_distance,
        "target_p_o_initial_coordination": initial_coordination,
        "initial_sam_covalent_edges_global_1based": initial_edges,
        "cell_A": np.asarray(initial.cell, dtype=float).tolist(),
        "pbc": np.asarray(initial.pbc, dtype=bool).tolist(),
        "surface_frame": surface_frame,
        "periodic_fractional_axes": list(periodic_axes),
        "normal_axis": int(surface_frame["normal_axis"]),
        "source_to_supercell_matrix": matrix.tolist(),
        "working_layer_atom_ids_1based": layer_identity["working_layer_atom_ids_1based"],
        "restore_layer_atom_ids_1based": layer_identity["restore_layer_atom_ids_1based"],
    }
    constraints = {
        "policy": "fixed_cell_freeze_lower_1280_move_top_1280_plus_one_SAM_plus_two_mapped_H",
        "frozen_substrate_indices_0based": list(range(0, 1280)),
        "frozen_substrate_atom_ids_1based": list(range(1, 1281)),
        "movable_substrate_indices_0based": list(range(1280, 2560)),
        "movable_substrate_atom_ids_1based": list(range(1281, 2561)),
        "movable_atom_indices_0based": list(range(1280, TARGETED_CALIBRATION_TOTAL_ATOMS)),
        "movable_atom_ids_1based": list(range(1281, TARGETED_CALIBRATION_TOTAL_ATOMS + 1)),
        "cell_fixed": True,
    }
    parameters = {
        "calculator": "MACECalculator",
        "device": "cuda",
        "default_dtype": "float32",
        "model_element_coverage_declared": declared_elements,
        "model_element_coverage_verification_boundary": "explicit declaration plus observed structure symbols; model introspection is not assumed",
        "optimizer": "ASE.LBFGS",
        "fmax_eV_per_A": TARGETED_CALIBRATION_FMAX_EVA,
        "maxstep_A": TARGETED_CALIBRATION_MAXSTEP_A,
        "max_steps": TARGETED_CALIBRATION_MAX_STEPS,
        "cell_max_abs_change_A": 1.0e-8,
        "frozen_max_displacement_A": 1.0e-8,
        "surface_h_parent_max_distance_A": 1.25,
        "covalent_radius_scale": TARGETED_CALIBRATION_COVALENT_RADIUS_SCALE,
        "p_o_coordination_cutoff_A": TARGETED_CALIBRATION_P_O_COORDINATION_CUTOFF_A,
        "p_o_stable_cutoff_A": TARGETED_CALIBRATION_P_O_STABLE_CUTOFF_A,
        "p_o_released_cutoff_A": TARGETED_CALIBRATION_P_O_RELEASED_CUTOFF_A,
        "p_coordination_fivefold": TARGETED_CALIBRATION_P_COORDINATION_FIVEFOLD,
        "strict_vdw_radius_scale": TARGETED_CALIBRATION_VDW_RADIUS_SCALE,
        "mapped_bond_window_A": list(TARGETED_CALIBRATION_MAPPED_BOND_WINDOW_A),
        "relax_cell": False,
    }
    identity_material = {
        "evidence": _targeted_identity_value(evidence, root),
        "system": _targeted_identity_value(system, root),
        "constraints": _targeted_identity_value(constraints, root),
        "parameters": _targeted_identity_value(parameters, root),
    }
    output_root = _targeted_resolve_path(output)
    if output_root.exists():
        raise ValueError(f"targeted calibration output must be new: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(exist_ok=False)
    initial_path = output_root / "initial.extxyz"
    write(initial_path, initial)
    initial_record = _targeted_file_record(initial_path, label="targeted calibration initial structure")
    plan = {
        "schema": "sam-targeted-fixed-site-calibration-plan-v2",
        "schema_version": TARGETED_CALIBRATION_SCHEMA_VERSION,
        "status": "planned_cpu_only_pending_cuda_worker",
        "role": TARGETED_CALIBRATION_ROLE,
        "identity_material": identity_material,
        "run_identity": hashlib.sha256(
            json.dumps(identity_material, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "output_directory": str(output_root),
        "inputs": {"initial_structure": initial_record},
        "evidence": evidence,
        "system": system,
        "constraints": constraints,
        "parameters": parameters,
        "initial_geometry": {
            "sealed_audit_id_convention": dict(
                TARGETED_CALIBRATION_SEALED_AUDIT_ID_CONVENTION
            ),
            "raw_production_strict_vdw_audit": initial_strict_vdw,
            "calibration_residual_nontarget_audit": _targeted_calibration_residual_nontarget_audit(
                initial_strict_vdw, target_p_local, target_o
            ),
            "strict_vdw_audit_summary": _targeted_audit_summary(initial_strict_vdw),
            "target_p_o_distance_A": target_distance,
            "target_p_o_coordination": initial_coordination,
            "target_pair_from_sealed_production_audit": target_contact_evidence,
            "registered_donor_mapping": donor_mapping,
        },
        "sealed_audit_id_convention": dict(
            TARGETED_CALIBRATION_SEALED_AUDIT_ID_CONVENTION
        ),
        "production_boundary": {
            "production_screen": False,
            "production_strict_vdw_gate_modified": False,
            "raw_production_strict_vdw_is_not_exempted": True,
            "strict_vdw_initial_rejection_is_not_waived": True,
            "target_pair_is_calibration_observable_only": True,
            "calibration_residual_nontarget_contacts_are_hard_fail": True,
            "surface_protonation_stage": "mapped_prototype_H_only_for_experimental_evidence",
            "promotion_eligible": False,
            "adsorption_energy_claim": False,
        },
        "forbidden_in_plan": ["MACE", "GPU", "optimizer", "download", "production_gate_mutation"],
    }
    plan_path = output_root / "plan.json"
    _targeted_write_json(plan_path, plan)
    plan_record = _targeted_file_record(plan_path, label="targeted calibration plan")
    manifest = {
        "schema": "sam-targeted-fixed-site-calibration-manifest-v2",
        "schema_version": TARGETED_CALIBRATION_SCHEMA_VERSION,
        "status": "planned_cpu_only_pending_cuda_worker",
        "role": TARGETED_CALIBRATION_ROLE,
        "plan": plan_record,
        "run_identity": plan["run_identity"],
        "sealed": True,
        "worker_task_count": 1,
        "production_promotion": False,
        "production_strict_vdw_gate_modified": False,
    }
    _targeted_write_json(output_root / "manifest.json", manifest)
    return {
        "output_directory": str(output_root),
        "plan_path": str(plan_path),
        "plan_sha256": plan_record["sha256"],
        "manifest_path": str(output_root / "manifest.json"),
        "initial_structure": initial_record,
        "plan": plan,
    }


def run_targeted_fixed_site_calibration_plan(args):
    targeted_output = getattr(args, "targeted_output", None) or getattr(args, "output_root", None)
    return resolve_targeted_calibration_plan(
        args.trajectory,
        args.molecule_index,
        args.model,
        targeted_output,
        site_instances=args.site_instances,
        prototype_library=args.prototype_library,
        source_surface=args.source_surface,
        model_elements=args.model_elements,
        root=getattr(args, "root", None),
    )


# Short aliases keep the generic capability discoverable without introducing a
# second entry point or a system-named wrapper.
run_targeted_calibration_plan = run_targeted_fixed_site_calibration_plan


def _targeted_verify_plan_inputs(plan):
    if plan.get("schema") != "sam-targeted-fixed-site-calibration-plan-v2" or plan.get("schema_version") != TARGETED_CALIBRATION_SCHEMA_VERSION:
        raise ValueError("unsupported targeted calibration plan schema")
    if plan.get("role") != TARGETED_CALIBRATION_ROLE:
        raise ValueError("targeted calibration plan role is invalid")
    if plan.get("status") != "planned_cpu_only_pending_cuda_worker":
        raise ValueError("targeted calibration plan is not an executable pending plan")
    evidence = plan.get("evidence") or {}
    for label, record in evidence.items():
        if not isinstance(record, dict) or not record.get("path") or not record.get("sha256"):
            raise ValueError(f"targeted plan hash lock is incomplete for {label}")
        _targeted_file_record(record["path"], expected_sha256=record["sha256"], label=label)
    initial_record = ((plan.get("inputs") or {}).get("initial_structure") or {})
    _targeted_file_record(
        initial_record.get("path"),
        expected_sha256=initial_record.get("sha256"),
        label="targeted plan initial structure",
    )
    model_record = evidence.get("model")
    if model_record is None or model_record["sha256"] != TARGETED_CALIBRATION_MODEL_SHA256:
        raise ValueError("targeted plan model hash lock is invalid")
    if plan["parameters"].get("device") != "cuda" or plan["parameters"].get("default_dtype") != "float32":
        raise ValueError("targeted worker device/dtype contract changed")
    if _targeted_parse_elements(plan["parameters"].get("model_element_coverage_declared", [])) != list(TARGETED_CALIBRATION_ELEMENTS):
        raise ValueError("targeted worker model element declaration changed")
    system = plan.get("system") or {}
    if system.get("sealed_audit_id_convention") != dict(
        TARGETED_CALIBRATION_SEALED_AUDIT_ID_CONVENTION
    ):
        raise ValueError("targeted worker sealed audit ID convention changed; fail-closed")
    if (
        system.get("substrate_atom_count"),
        system.get("parent_molecule_count"),
        system.get("target_molecule_index_1based"),
        system.get("molecule_block_size"),
        system.get("expected_atom_count"),
    ) != (2560, 68, 62, 46, 2608):
        raise ValueError("targeted worker system count contract changed")
    if dict(system.get("molecule_formula") or {}) != TARGETED_CALIBRATION_MOLECULE_FORMULA:
        raise ValueError("targeted worker molecule formula contract changed")
    target_p_local = system.get("target_p_molecule_local_index_0based")
    if not isinstance(target_p_local, int) or not 0 <= target_p_local < TARGETED_CALIBRATION_MOLECULE_ATOMS:
        raise ValueError("targeted worker target P molecule-local identity is invalid")
    surface_frame = _targeted_normalize_surface_frame(
        system.get("surface_frame"),
        normal_axis=system.get("normal_axis"),
        source_label="targeted plan system",
    )
    if system.get("periodic_fractional_axes") != surface_frame["periodic_fractional_axes"]:
        raise ValueError("targeted worker periodic Surface Frame identity changed")
    constraints = plan.get("constraints") or {}
    if constraints.get("frozen_substrate_indices_0based") != list(range(1280)) or constraints.get("movable_substrate_indices_0based") != list(range(1280, 2560)) or constraints.get("movable_atom_indices_0based") != list(range(1280, 2608)):
        raise ValueError("targeted worker frozen/movable atom identity contract changed")
    if plan["parameters"].get("fmax_eV_per_A") != TARGETED_CALIBRATION_FMAX_EVA or plan["parameters"].get("maxstep_A") != TARGETED_CALIBRATION_MAXSTEP_A or plan["parameters"].get("max_steps") != TARGETED_CALIBRATION_MAX_STEPS:
        raise ValueError("targeted worker optimizer contract changed")
    if plan["parameters"].get("strict_vdw_radius_scale") != TARGETED_CALIBRATION_VDW_RADIUS_SCALE:
        raise ValueError("targeted worker strict-vdW contract changed")
    return _targeted_file_record(
        initial_record["path"],
        expected_sha256=initial_record["sha256"],
        label="targeted plan initial structure",
    )


def _targeted_runtime_versions():
    """Import runtime packages only in the explicit worker, after CUDA gating."""

    import importlib.metadata
    import scipy
    runtime_torch, _ = _load_torch_for_mace_worker(
        "targeted calibration MACE worker",
        require_cuda=True,
        cuda_failure_message="CUDA is unavailable; targeted calibration forbids CPU fallback",
    )
    import ase as runtime_ase
    try:
        import mace as runtime_mace
        from mace.calculators import MACECalculator as runtime_calculator
    except Exception as exc:
        raise RuntimeError("MACE is unavailable; targeted calibration has no fallback") from exc
    mace_version = getattr(runtime_mace, "__version__", None)
    if not mace_version:
        for distribution in ("mace-torch", "mace"):
            try:
                mace_version = importlib.metadata.version(distribution)
                break
            except importlib.metadata.PackageNotFoundError:
                continue
    if not mace_version:
        raise RuntimeError("MACE version is unavailable for the worker runtime contract")
    return {
        "torch": str(runtime_torch.__version__),
        "ase": str(runtime_ase.__version__),
        "scipy": str(scipy.__version__),
        "mace": str(mace_version),
        "cuda_available": bool(runtime_torch.cuda.is_available()),
        "calculator": runtime_calculator,
    }


def _targeted_copy_exclusive(source, destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"targeted task artifact already exists: {destination}")
    data = Path(source).read_bytes()
    with destination.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    return _targeted_file_record(destination, label=destination.name)


def _targeted_task_inventory(task_directory):
    records = []
    for path in sorted(Path(task_directory).rglob("*")):
        if path.is_file() and not path.is_symlink():
            try:
                records.append(_targeted_file_record(path, label=str(path.relative_to(task_directory))))
            except OSError:
                continue
    return records


def run_targeted_fixed_site_calibration_worker(args):
    """Run the one permitted CUDA/MACE targeted-calibration task."""

    plan_path = _targeted_resolve_path(args.plan)
    expected_plan_sha256 = str(args.plan_sha256)
    if _sha256(plan_path) != expected_plan_sha256:
        raise ValueError("targeted calibration plan hash changed before worker start")
    with plan_path.open(encoding="utf-8") as handle:
        plan = json.load(handle)
    output_directory = _targeted_resolve_path(plan["output_directory"])
    if output_directory != plan_path.parent:
        raise ValueError("targeted plan output directory does not contain the plan")
    requested_task_index = getattr(args, "targeted_task_index", None)
    if requested_task_index is None:
        requested_task_index = getattr(args, "task_index", 1)
    task_index = int(requested_task_index)
    if task_index != 1:
        raise ValueError("targeted calibration has exactly one task")
    task_directory = output_directory / f"task-{task_index:04d}"
    if task_directory.exists():
        raise ValueError(f"targeted calibration task directory already exists: {task_directory}")
    task_directory.mkdir(exist_ok=False)
    failure_payload = None
    try:
        initial_record = _targeted_verify_plan_inputs(plan)
        task_initial_record = _targeted_copy_exclusive(
            initial_record["path"], task_directory / "initial.extxyz"
        )
        initial = read(task_initial_record["path"])
        if len(initial) != TARGETED_CALIBRATION_TOTAL_ATOMS:
            raise ValueError("targeted worker input atom count is not 2608")
        if set(initial.get_chemical_symbols()) - set(TARGETED_CALIBRATION_ELEMENTS):
            raise ValueError("targeted worker input contains an undeclared element")
        if _composition(initial) != plan["system"]["composition"]:
            raise ValueError("targeted worker input composition changed")
        if _composition(initial[:TARGETED_CALIBRATION_SUBSTRATE_ATOMS]) != plan["system"]["working_substrate_composition"] or _composition(initial[TARGETED_CALIBRATION_SUBSTRATE_ATOMS:TARGETED_CALIBRATION_TOTAL_ATOMS - TARGETED_CALIBRATION_SURFACE_H_COUNT]) != TARGETED_CALIBRATION_MOLECULE_FORMULA:
            raise ValueError("targeted worker substrate or SAM formula changed")
        target_p_index = TARGETED_CALIBRATION_SUBSTRATE_ATOMS + int(plan["system"]["target_p_molecule_local_index_0based"])
        if initial[target_p_index].symbol != "P" or initial[-2].symbol != "H" or initial[-1].symbol != "H":
            raise ValueError("targeted worker atom-block identity changed")
        frozen = [int(value) for value in plan["constraints"]["frozen_substrate_indices_0based"]]
        movable = [int(value) for value in plan["constraints"]["movable_atom_indices_0based"]]
        if frozen != list(range(1280)) or movable != list(range(1280, TARGETED_CALIBRATION_TOTAL_ATOMS)):
            raise ValueError("targeted worker frozen/movable identity contract changed")
        runtime = _targeted_runtime_versions()
        MACECalculator = runtime.pop("calculator")
        from ase.optimize import LBFGS

        atoms = initial.copy()
        atoms.set_constraint(FixAtoms(indices=frozen))
        atoms.calc = MACECalculator(
            model_paths=str(plan["evidence"]["model"]["path"]),
            device="cuda",
            default_dtype="float32",
        )
        progress_path = task_directory / "progress.jsonl"
        progress_handle = progress_path.open("a", buffering=1, encoding="utf-8")
        movable_indices = movable
        optimizer = LBFGS(
            atoms,
            trajectory=str(task_directory / "trajectory.traj"),
            logfile=str(task_directory / "optimizer.log"),
            maxstep=float(plan["parameters"]["maxstep_A"]),
        )

        def record_progress():
            forces = np.asarray(atoms.get_forces(), dtype=float)
            payload = {
                "step": int(optimizer.nsteps),
                "total_energy_eV": float(atoms.get_potential_energy()),
                "movable_max_force_eV_per_A": float(
                    np.max(np.linalg.norm(forces[movable_indices], axis=1))
                ),
            }
            progress_handle.write(json.dumps(_targeted_jsonable(payload), sort_keys=True) + "\n")
            progress_handle.flush()
            os.fsync(progress_handle.fileno())
            print(json.dumps(payload, sort_keys=True), flush=True)

        record_progress()
        optimizer.attach(record_progress, interval=1)
        optimizer.run(
            fmax=float(plan["parameters"]["fmax_eV_per_A"]),
            steps=int(plan["parameters"]["max_steps"]),
        )
        progress_handle.close()
        final_energy = float(atoms.get_potential_energy())
        final_forces = np.asarray(atoms.get_forces(), dtype=float)
        optimizer_converged = False
        try:
            optimizer_converged = bool(
                optimizer.converged(optimizer.optimizable.get_gradient())
            )
        except Exception:
            optimizer_converged = False
        relaxed_path = task_directory / "relaxed.extxyz"
        if relaxed_path.exists():
            raise FileExistsError(f"targeted relaxed artifact already exists: {relaxed_path}")
        relaxed = atoms.copy()
        relaxed.calc = None
        relaxed.set_constraint()
        write(relaxed_path, relaxed)
        relaxed_record = _targeted_file_record(relaxed_path, label="targeted relaxed structure")
        final_readback = read(relaxed_path)
        initial_strict = _targeted_strict_vdw_audit(
            initial, plan["system"]["donor_mapping"], plan["system"]["surface_frame"]
        )
        final_strict = _targeted_strict_vdw_audit(
            final_readback, plan["system"]["donor_mapping"], plan["system"]["surface_frame"]
        )
        validation = validate_targeted_calibration_final_geometry(
            initial,
            final_readback,
            plan,
            final_forces,
            optimizer.nsteps,
            optimizer_converged,
            initial_strict_vdw=initial_strict,
            final_strict_vdw=final_strict,
        )
        validation_path = task_directory / "validation.json"
        _targeted_write_json(validation_path, validation)
        validation_record = _targeted_file_record(validation_path, label="targeted validation")
        manifest_status = (
            "passed_targeted_calibration_experimental_evidence"
            if validation["passed"]
            else "failed_targeted_calibration_validation"
        )
        manifest = {
            "schema": "sam-targeted-fixed-site-calibration-task-manifest-v2",
            "schema_version": TARGETED_CALIBRATION_SCHEMA_VERSION,
            "sealed": True,
            "status": manifest_status,
            "role": TARGETED_CALIBRATION_ROLE,
            "task_index": task_index,
            "plan": {"path": str(plan_path), "sha256": expected_plan_sha256},
            "input": task_initial_record,
            "initial_structure": task_initial_record,
            "trajectory": _targeted_file_record(task_directory / "trajectory.traj", label="targeted trajectory"),
            "optimizer_log": _targeted_file_record(task_directory / "optimizer.log", label="targeted optimizer log"),
            "relaxed_structure": relaxed_record,
            "validation": validation_record,
            "progress": _targeted_file_record(progress_path, label="targeted progress"),
            "runtime": runtime,
            "optimizer": {
                "steps": int(optimizer.nsteps),
                "converged": optimizer_converged,
                "fmax_eV_per_A": float(plan["parameters"]["fmax_eV_per_A"]),
                "maxstep_A": float(plan["parameters"]["maxstep_A"]),
                "max_steps": int(plan["parameters"]["max_steps"]),
                "total_energy_eV": final_energy,
            },
            "classification": validation["classification"],
            "production_strict_vdw_gate_modified": False,
            "promotion_eligible": False,
            "artifact_inventory": _targeted_task_inventory(task_directory),
        }
        _targeted_write_json(task_directory / "manifest.json", manifest)
        print(json.dumps(_targeted_jsonable(manifest), indent=2, sort_keys=True), flush=True)
        return 0 if validation["passed"] else 2
    except Exception as error:
        try:
            failure_payload = {
                "schema": "sam-targeted-fixed-site-calibration-failure-v2",
                "schema_version": TARGETED_CALIBRATION_SCHEMA_VERSION,
                "sealed": True,
                "status": "failed_targeted_calibration_worker",
                "role": TARGETED_CALIBRATION_ROLE,
                "task_index": task_index,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "plan": {"path": str(plan_path), "sha256": expected_plan_sha256},
                "artifact_inventory": _targeted_task_inventory(task_directory),
                "production_strict_vdw_gate_modified": False,
                "promotion_eligible": False,
            }
            _targeted_write_json(task_directory / "failure.json", failure_payload)
            _targeted_write_json(task_directory / "manifest.json", failure_payload)
        except Exception as seal_error:
            print(
                json.dumps(
                    {
                        "task_index": task_index,
                        "error": str(error),
                        "seal_error": str(seal_error),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        print(json.dumps(_targeted_jsonable(failure_payload or {"error": str(error)}), sort_keys=True), flush=True)
        return 2


run_targeted_calibration_worker = run_targeted_fixed_site_calibration_worker


# ---------------------------------------------------------------------------
# Generic v3 P--surface-O experimental topology contract
# ---------------------------------------------------------------------------


def _pso_copy_json(value):
    """Make a JSON-only deep copy for immutable contract normalization."""

    return json.loads(json.dumps(value))


def _pso_project_root():
    return Path(__file__).resolve().parents[1]


def _pso_forbidden_candidates():
    return [_pso_copy_json(P_SURFACE_O_FORBIDDEN_CANDIDATE)]


def _pso_resolve_audit_dependency():
    """Resolve and hash the module actually used by the strict-vdW audit.

    The import is deliberately performed by the generic owner rather than by a
    filename convention.  The returned record describes the loaded module,
    its complete source bytes, and the source of the audit entry point.
    """

    import importlib
    import inspect

    try:
        module = importlib.import_module(P_SURFACE_O_AUDIT_MODULE_NAME)
    except Exception as exc:
        raise ValueError(
            "v3 strict-vdW audit dependency cannot be imported: "
            f"{P_SURFACE_O_AUDIT_MODULE_NAME}"
        ) from exc
    raw_module_path = getattr(module, "__file__", None)
    if not raw_module_path:
        raise ValueError("v3 strict-vdW audit module has no source path")
    raw_module_path = Path(raw_module_path).expanduser()
    if raw_module_path.is_symlink():
        raise ValueError("v3 strict-vdW audit module source must not be a symlink")
    module_path = raw_module_path.resolve()
    module_record = _targeted_file_record(
        module_path, label="v3 strict-vdW audit module"
    )
    function = getattr(module, P_SURFACE_O_AUDIT_FUNCTION_NAME, None)
    if function is None or not callable(function):
        raise ValueError(
            "v3 strict-vdW audit module lacks callable "
            f"{P_SURFACE_O_AUDIT_FUNCTION_NAME}"
        )
    try:
        function_source_path = Path(inspect.getsourcefile(function) or module_path).resolve()
        function_source = inspect.getsource(function)
    except (OSError, TypeError) as exc:
        raise ValueError(
            "v3 strict-vdW audit function source cannot be stably inspected"
        ) from exc
    if function_source_path != module_path:
        raise ValueError(
            "v3 strict-vdW audit function is sourced from a different module; "
            "lock that direct dependency explicitly before running"
        )
    function_bytes = function_source.encode("utf-8")
    return {
        "module_name": P_SURFACE_O_AUDIT_MODULE_NAME,
        "import_name": P_SURFACE_O_AUDIT_MODULE_NAME,
        "loaded_path": str(module_path),
        "project_relative_path": _targeted_project_relative_path(
            module_path,
            _pso_project_root(),
            content_sha256=module_record["sha256"],
        ),
        "sha256": module_record["sha256"],
        "whole_file_sha256": module_record["sha256"],
        "bytes": module_record["bytes"],
        "function": {
            "name": P_SURFACE_O_AUDIT_FUNCTION_NAME,
            "qualified_name": str(
                getattr(function, "__qualname__", P_SURFACE_O_AUDIT_FUNCTION_NAME)
            ),
            "source_sha256": hashlib.sha256(function_bytes).hexdigest(),
            "source_bytes": len(function_bytes),
            "source_hash_method": "inspect.getsource:utf-8",
            "source_path": str(function_source_path),
        },
    }


def _pso_verify_audit_dependency(lock, package_root=None):
    """Re-hash the loaded audit module and its immutable run snapshot."""

    if not isinstance(lock, dict):
        raise ValueError("v3 strict-vdW audit dependency lock is missing")
    expected_path = lock.get("loaded_path")
    expected_sha = lock.get("whole_file_sha256") or lock.get("sha256")
    function_lock = lock.get("function")
    if not expected_path or not expected_sha or not isinstance(function_lock, dict):
        raise ValueError("v3 strict-vdW audit dependency lock is incomplete")
    current = _pso_resolve_audit_dependency()
    if _targeted_resolve_path(current["loaded_path"]) != _targeted_resolve_path(expected_path):
        raise ValueError(
            "v3 strict-vdW audit dependency loaded path changed: "
            f"expected {expected_path}, got {current['loaded_path']}"
        )
    if current["whole_file_sha256"] != str(expected_sha):
        raise ValueError("v3 strict-vdW audit dependency whole-file hash mismatch")
    if current.get("project_relative_path") != lock.get("project_relative_path"):
        raise ValueError("v3 strict-vdW audit dependency project-relative path changed")
    if current["function"].get("source_sha256") != function_lock.get("source_sha256"):
        raise ValueError("v3 strict-vdW audit function source hash mismatch")
    snapshot_path = lock.get("snapshot_path")
    snapshot_sha = lock.get("snapshot_sha256")
    if snapshot_path is not None or snapshot_sha is not None:
        if package_root is None or not snapshot_path or not snapshot_sha:
            raise ValueError("v3 strict-vdW audit dependency snapshot lock is incomplete")
        package_root = _targeted_resolve_path(package_root)
        snapshot = _targeted_resolve_path(package_root / str(snapshot_path))
        try:
            snapshot.relative_to(package_root)
        except ValueError as exc:
            raise ValueError("v3 strict-vdW audit dependency snapshot escaped the package") from exc
        snapshot_record = _targeted_file_record(
            snapshot,
            expected_sha256=str(snapshot_sha),
            label="v3 strict-vdW audit dependency snapshot",
        )
        if snapshot_record["sha256"] != current["whole_file_sha256"]:
            raise ValueError(
                "v3 strict-vdW audit dependency snapshot differs from loaded source"
            )
        if int(snapshot_record["bytes"]) != int(lock.get("snapshot_bytes", snapshot_record["bytes"])):
            raise ValueError("v3 strict-vdW audit dependency snapshot byte count changed")
    return current


def _pso_install_code_snapshot(output_root, implementation_record, audit_lock):
    """Copy the owner and strict-vdW dependency into a new run package."""

    snapshot_directory = _targeted_resolve_path(output_root) / "code-snapshot"
    snapshot_directory.mkdir(exist_ok=False)
    files = []
    for role, source_record, filename in (
        ("canonical_owner", implementation_record, "mace_conformation_scan.py"),
        ("strict_vdw_audit_dependency", audit_lock, "sam_structure_tools.py"),
    ):
        source_path = source_record.get("path") or source_record.get("loaded_path")
        if not source_path:
            raise ValueError(f"v3 {role} code snapshot source path is missing")
        source_sha = source_record.get("whole_file_sha256", source_record.get("sha256"))
        source_bytes = int(source_record["bytes"])
        destination = snapshot_directory / filename
        _targeted_copy_exclusive(source_path, destination)
        snapshot_record = _targeted_file_record(
            destination, expected_sha256=source_sha, label=f"v3 {role} code snapshot"
        )
        files.append(
            {
                "role": role,
                "module": filename,
                "source_path": str(source_path),
                "source_sha256": str(source_sha),
                "source_bytes": source_bytes,
                "path": f"code-snapshot/{filename}",
                "sha256": snapshot_record["sha256"],
                "bytes": snapshot_record["bytes"],
            }
        )
    updated_audit_lock = _pso_copy_json(audit_lock)
    audit_snapshot = next(
        item for item in files if item["role"] == "strict_vdw_audit_dependency"
    )
    updated_audit_lock.update(
        {
            "snapshot_path": audit_snapshot["path"],
            "snapshot_sha256": audit_snapshot["sha256"],
            "snapshot_bytes": audit_snapshot["bytes"],
        }
    )
    return {
        "path": "code-snapshot",
        "overwrite": "forbidden",
        "purpose": "exact owner and strict-vdw audit dependency source bytes",
        "files": files,
    }, updated_audit_lock


def _pso_verify_code_snapshot(snapshot, package_root):
    if not isinstance(snapshot, dict) or snapshot.get("path") != "code-snapshot":
        raise ValueError("v3 code snapshot lock is missing")
    package_root = _targeted_resolve_path(package_root)
    files = snapshot.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("v3 code snapshot has no locked files")
    for item in files:
        if not isinstance(item, dict) or not item.get("path") or not item.get("sha256"):
            raise ValueError("v3 code snapshot file lock is incomplete")
        path = _targeted_resolve_path(package_root / str(item["path"]))
        try:
            path.relative_to(package_root)
        except ValueError as exc:
            raise ValueError("v3 code snapshot path escaped the package") from exc
        _targeted_file_record(
            path,
            expected_sha256=str(item["sha256"]),
            label=f"v3 code snapshot {item.get('module', item['path'])}",
        )


def _pso_require_new_contract_declarations(contract):
    if contract.get("prototype0001_deferred") is not True:
        raise ValueError("v3 contract must explicitly set prototype0001_deferred=true")
    forbidden = contract.get("forbidden_candidates")
    expected = _pso_forbidden_candidates()
    if forbidden != expected:
        raise ValueError("v3 contract must explicitly forbid candidate prototype0001")
    candidates = contract.get("candidates")
    if isinstance(candidates, list) and any(
        str(item.get("candidate_id")) == "prototype0001"
        for item in candidates
        if isinstance(item, dict)
    ):
        raise ValueError("prototype0001 is forbidden and cannot enter the candidate set")


def _pso_read_sealed_baseline_energy(contract):
    """Read the locked v2 total-energy reference, never an adsorption energy."""

    baseline = contract.get("baseline")
    evidence = baseline.get("evidence") if isinstance(baseline, dict) else None
    record = evidence.get("sealed_v2_task_manifest") if isinstance(evidence, dict) else None
    if not isinstance(record, dict):
        raise ValueError("v3 contract lacks sealed baseline task-manifest energy evidence")
    payload = _targeted_read_json(record, "v3 sealed baseline task manifest")
    try:
        energy = float(payload["optimizer"]["total_energy_eV"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("v3 sealed baseline task manifest lacks total energy") from exc
    if not np.isfinite(energy):
        raise ValueError("v3 sealed baseline total energy is non-finite")
    return {
        "quantity": "sealed_baseline_total_energy_eV",
        "total_energy_eV": energy,
        "source": dict(record),
        "runtime": dict(payload.get("runtime") or {}),
        "adsorption_energy": False,
        "mpa_reference_subtraction": False,
    }


def _pso_read_config(path):
    path = _targeted_resolve_path(path)
    record = _targeted_file_record(path, label="v3 contract configuration")
    payload = _targeted_read_json(record, "v3 contract configuration")
    if payload.get("schema") != "sam-p-surface-o-experimental-contract-config-v3":
        raise ValueError("unsupported v3 P--surface-O contract configuration schema")
    return payload, record


def _pso_resolve_spec(spec, base, label):
    if not isinstance(spec, dict):
        raise ValueError(f"{label} must be a hash-locked file record")
    raw_path = spec.get("path")
    expected = spec.get("sha256")
    if raw_path is None or expected is None:
        raise ValueError(f"{label} lacks a path or SHA-256 lock")
    return _targeted_file_record(
        _targeted_resolve_path(raw_path, base),
        expected_sha256=str(expected),
        label=label,
    )


def _pso_normalize_config(config, config_path):
    """Resolve paths while retaining a generic, configuration-driven contract."""

    normalized = _pso_copy_json(config)
    normalized["schema"] = "sam-p-surface-o-experimental-contract-v3"
    if normalized.get("prototype0001_deferred", True) is not True:
        raise ValueError("v3 contract must explicitly set prototype0001_deferred=true")
    candidates = normalized.get("candidates")
    if isinstance(candidates, list) and any(
        isinstance(item, dict) and str(item.get("candidate_id")) == "prototype0001"
        for item in candidates
    ):
        raise ValueError("prototype0001 is forbidden and cannot enter the candidate set")
    normalized["prototype0001_deferred"] = True
    normalized["forbidden_candidates"] = _pso_forbidden_candidates()
    config_path = _targeted_resolve_path(config_path)
    base = config_path.parent
    model = normalized.get("model")
    if not isinstance(model, dict):
        raise ValueError("v3 contract requires a model record")
    model_record = _pso_resolve_spec(model, base, "v3 MACE model")
    model["path"] = model_record["path"]
    model["sha256"] = model_record["sha256"]
    baseline = normalized.get("baseline")
    if not isinstance(baseline, dict):
        raise ValueError("v3 contract requires a sealed baseline record")
    baseline_structure = _pso_resolve_spec(
        baseline.get("structure"), base, "v3 sealed baseline structure"
    )
    baseline["structure"] = baseline_structure
    evidence = baseline.get("evidence", {})
    if not isinstance(evidence, dict):
        raise ValueError("v3 baseline evidence must be an object")
    normalized_evidence = {}
    for label, spec in sorted(evidence.items()):
        normalized_evidence[str(label)] = _pso_resolve_spec(
            spec, base, f"v3 baseline evidence {label}"
        )
    baseline["evidence"] = normalized_evidence
    return normalized, model_record, baseline_structure, normalized_evidence


def _pso_unit_vector(value, label):
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must be a finite Cartesian 3-vector")
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError(f"{label} must be nonzero")
    return vector / norm


def _pso_molecule_edges(atoms, molecule_indices, scale, periodic_axes):
    indices = [int(value) for value in molecule_indices]
    edges = []
    for left_offset, left in enumerate(indices):
        for right in indices[left_offset + 1:]:
            cutoff = float(scale) * (
                covalent_radii[atoms[left].number] + covalent_radii[atoms[right].number]
            )
            distance = _targeted_distance(
                atoms, left, right, periodic_axes=periodic_axes
            )
            if distance <= cutoff:
                edges.append((left, right))
    return sorted(edges)


def _pso_edge_key(edges, molecule_start):
    return sorted(
        [
            [int(left) - int(molecule_start), int(right) - int(molecule_start)]
            for left, right in edges
        ]
    )


def _pso_unwrap_molecule(atoms, molecule_indices, anchor_index, periodic_axes, edges):
    indices = [int(value) for value in molecule_indices]
    index_set = set(indices)
    adjacency = {index: [] for index in indices}
    for left, right in edges:
        if left not in index_set or right not in index_set:
            raise ValueError("v3 molecule graph edge is outside the molecule block")
        adjacency[left].append(right)
        adjacency[right].append(left)
    unwrapped = np.full((len(indices), 3), np.nan, dtype=float)
    local_of = {index: offset for offset, index in enumerate(indices)}
    unwrapped[local_of[int(anchor_index)]] = np.asarray(
        atoms.positions[int(anchor_index)], dtype=float
    )
    queue = deque([int(anchor_index)])
    visited = {int(anchor_index)}
    while queue:
        current = queue.popleft()
        for neighbor in sorted(adjacency[current]):
            if neighbor in visited:
                continue
            vector, _ = _targeted_partial_mic(
                np.asarray(atoms.positions[neighbor])
                - np.asarray(atoms.positions[current]),
                atoms.cell,
                periodic_axes=periodic_axes,
            )
            unwrapped[local_of[neighbor]] = (
                unwrapped[local_of[current]] + vector
            )
            visited.add(neighbor)
            queue.append(neighbor)
    if len(visited) != len(indices) or not np.all(np.isfinite(unwrapped)):
        missing = sorted(index_set - visited)
        raise ValueError(
            "v3 DBF/molecule block is not one covalently connected component; "
            f"unreached atom IDs={missing[:10]}"
        )
    return unwrapped


def _pso_rotation_matrix(axis, angle_deg):
    axis = _pso_unit_vector(axis, "rotation axis")
    angle = float(angle_deg)
    if not np.isfinite(angle):
        raise ValueError("rotation angle must be finite")
    theta = np.deg2rad(angle)
    x, y, z = axis
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return (
        np.eye(3) * np.cos(theta)
        + (1.0 - np.cos(theta)) * np.outer(axis, axis)
        + np.sin(theta) * cross
    )


def _pso_apply_candidate_transform(baseline, contract, candidate):
    """Apply one rigid transform to the complete molecule, never to one P atom."""

    system = contract["system"]
    parameters = contract["contract"]
    molecule_indices = [int(value) for value in system["molecule_indices_0based"]]
    p_index = int(system["target_p_global_index_0based"])
    target_o = int(system["target_surface_oxygen_working_index_0based"])
    periodic_axes = tuple(
        int(value) for value in contract["surface_frame"]["periodic_fractional_axes"]
    )
    scale = float(parameters["covalent_radius_scale"])
    edges = _pso_molecule_edges(
        baseline,
        molecule_indices,
        scale,
        periodic_axes,
    )
    unwrapped = _pso_unwrap_molecule(
        baseline,
        molecule_indices,
        p_index,
        periodic_axes,
        edges,
    )
    local_of = {index: offset for offset, index in enumerate(molecule_indices)}
    p_local = local_of[p_index]
    p_origin = unwrapped[p_local].copy()
    transform_type = candidate.get("type")
    if not isinstance(transform_type, str):
        raise ValueError("v3 candidate transform lacks type")
    rotation = np.eye(3)
    translation = np.zeros(3, dtype=float)
    design = {}
    transformed = unwrapped.copy()
    if transform_type == "rigid_translation":
        translation = np.asarray(candidate.get("translation_A"), dtype=float)
        if translation.shape != (3,) or not np.all(np.isfinite(translation)):
            raise ValueError("rigid_translation requires a finite 3-vector")
        transformed = unwrapped + translation
        design = {"translation_A": translation.tolist()}
    elif transform_type == "rigid_rotation":
        axis = _pso_unit_vector(candidate.get("axis_cartesian"), "candidate rotation axis")
        angle = float(candidate.get("angle_deg"))
        rotation = _pso_rotation_matrix(axis, angle)
        transformed = (unwrapped - p_origin) @ rotation.T + p_origin
        design = {
            "axis_cartesian_unit": axis.tolist(),
            "angle_deg": angle,
        }
    elif transform_type == "rigid_translation_to_target_distance":
        axis = _pso_unit_vector(
            candidate.get("axis_cartesian", [0.0, 0.0, 1.0]),
            "target-distance translation axis",
        )
        target_distance = float(candidate.get("target_distance_A"))
        if not np.isfinite(target_distance) or target_distance <= 0.0:
            raise ValueError("target-distance transform requires a positive finite distance")
        vector, _ = _targeted_partial_mic(
            unwrapped[p_local] - np.asarray(baseline.positions[target_o]),
            baseline.cell,
            periodic_axes=periodic_axes,
        )
        parallel = float(np.dot(vector, axis))
        lateral_vector = vector - parallel * axis
        lateral = float(np.linalg.norm(lateral_vector))
        radicand = target_distance * target_distance - lateral * lateral
        if radicand < -1.0e-12:
            raise ValueError(
                "requested target P--surface-O distance is smaller than the fixed lateral separation"
            )
        new_parallel_abs = float(np.sqrt(max(radicand, 0.0)))
        sign = 1.0 if parallel >= 0.0 else -1.0
        translation = (sign * new_parallel_abs - parallel) * axis
        transformed = unwrapped + translation
        design = {
            "axis_cartesian_unit": axis.tolist(),
            "target_distance_A": target_distance,
            "fixed_lateral_separation_A": lateral,
            "translation_A": translation.tolist(),
            "solved_parallel_displacement_A": float(np.linalg.norm(translation)),
        }
    else:
        raise ValueError(f"unsupported v3 candidate transform type: {transform_type}")

    result = baseline.copy()
    result.positions[molecule_indices] = transformed
    initial_pairwise = np.linalg.norm(
        unwrapped[:, None, :] - unwrapped[None, :, :], axis=2
    )
    final_pairwise = np.linalg.norm(
        transformed[:, None, :] - transformed[None, :, :], axis=2
    )
    rigidity_error = float(np.max(np.abs(initial_pairwise - final_pairwise)))
    target_distance = _targeted_distance(
        result, p_index, target_o, periodic_axes=periodic_axes
    )
    transform_record = {
        "type": transform_type,
        "rotation_matrix": rotation.tolist(),
        "translation_A": translation.tolist(),
        "anchor_global_index_0based": p_index,
        "molecule_global_indices_0based": molecule_indices,
        "design": design,
        "design_target_distance_A": (
            float(candidate["target_distance_A"])
            if "target_distance_A" in candidate
            else None
        ),
        "initial_target_distance_A": _targeted_distance(
            baseline, p_index, target_o, periodic_axes=periodic_axes
        ),
        "result_target_distance_A": target_distance,
        "rigidity_max_pair_distance_error_A": rigidity_error,
        "p_atom_displacement_A": float(
            np.linalg.norm(transformed[p_local] - unwrapped[p_local])
        ),
        "whole_molecule_transformed": True,
        "single_atom_P_move": False,
    }
    return result, transform_record, edges


def _pso_audit(atoms, contract):
    """Run the shared partial-PBC strict-vdW audit without changing production data."""

    from sam_structure_tools import periodic_vdw_collision_audit

    system = contract["system"]
    parameters = contract["contract"]
    frame = _targeted_normalize_surface_frame(
        contract["surface_frame"], source_label="v3 contract"
    )
    substrate_count = int(system["substrate_atom_count"])
    molecule_indices = [int(value) for value in system["molecule_indices_0based"]]
    surface_h_indices = [int(value) for value in system["surface_h_indices_0based"]]
    substrate_indices = list(range(substrate_count)) + surface_h_indices
    molecule_positions = np.asarray(atoms.positions[molecule_indices], dtype=float)
    substrate_positions = np.asarray(atoms.positions[substrate_indices], dtype=float)
    molecule_symbols = [atoms[index].symbol for index in molecule_indices]
    substrate_symbols = [atoms[index].symbol for index in substrate_indices]
    molecule_ids = list(range(len(molecule_indices)))
    substrate_ids = list(range(substrate_count)) + surface_h_indices
    radii = {
        symbol: float(vdw_radii[atomic_numbers[symbol]])
        for symbol in sorted(set(molecule_symbols + substrate_symbols))
    }
    mapped_windows = {}
    for donor in system["registered_donor_mappings"]:
        mapped_windows[
            (
                int(donor["molecule_local_index_0based"]),
                int(donor["working_metal_index_0based"]),
            )
        ] = tuple(float(value) for value in donor["distance_window_A"])
    audit = periodic_vdw_collision_audit(
        molecule_positions=molecule_positions,
        molecule_symbols=molecule_symbols,
        molecule_atom_ids=molecule_ids,
        substrate_positions=substrate_positions,
        substrate_symbols=substrate_symbols,
        substrate_atom_ids=substrate_ids,
        cell=np.asarray(atoms.cell, dtype=float),
        periodic_axes=frame["periodic_fractional_axes"],
        surface_frame=frame,
        radii_A=radii,
        radius_scale=float(parameters["strict_vdw_radius_scale"]),
        mapped_bond_windows_A=mapped_windows,
    )
    audit["id_scope"] = {
        "molecule_atom_id": "molecule-local-0based",
        "substrate_atom_id": "working-substrate-local-0based; surface-H uses global final-H0 IDs",
        "target_pair": {
            "molecule_atom_id": int(system["target_p_molecule_local_index_0based"]),
            "substrate_atom_id": int(system["target_surface_oxygen_working_index_0based"]),
            "global_molecule_atom_index_0based": int(
                system["target_p_global_index_0based"]
            ),
        },
    }
    return audit


def _pso_target_residual_audit(raw_audit, contract):
    """Filter exactly one target pair; every other raw contact remains evidence."""

    system = contract["system"]
    target_pair = (
        int(system["target_p_molecule_local_index_0based"]),
        int(system["target_surface_oxygen_working_index_0based"]),
    )
    mapped_pairs = {
        (
            int(item["molecule_local_index_0based"]),
            int(item["working_metal_index_0based"]),
        )
        for item in system["registered_donor_mappings"]
    }
    target_contacts = []
    residual = []
    for record in raw_audit.get("collisions", []):
        pair = (
            int(record.get("molecule_atom_id", -1)),
            int(record.get("substrate_atom_id", -1)),
        )
        if pair == target_pair:
            target_contacts.append(dict(record))
        else:
            residual.append(dict(record))
    by_category = {name: [] for name in (
        "hydrogen_bond",
        "geometric_near_contact",
        "repulsion",
        "new_topology",
    )}
    for record in residual:
        pair = (
            int(record.get("molecule_atom_id", -1)),
            int(record.get("substrate_atom_id", -1)),
        )
        # A mapped pair can be present in mapped_bonds and is never removed
        # from the raw audit by this experimental filter.
        if pair in mapped_pairs:
            category = "new_topology"
        elif (
            "H" in {record.get("molecule_element"), record.get("substrate_element")}
            and {record.get("molecule_element"), record.get("substrate_element")} & {"O", "N", "S"}
            and float(record.get("distance_A", np.inf)) <= 2.5
        ):
            category = "hydrogen_bond"
        else:
            threshold = float(record.get("threshold_A", np.inf))
            distance = float(record.get("distance_A", np.inf))
            category = (
                "geometric_near_contact"
                if np.isfinite(threshold) and distance >= 0.90 * threshold
                else "repulsion"
            )
        item = dict(record)
        item["category"] = category
        item["is_nonregistered_short_contact"] = pair not in mapped_pairs
        by_category[category].append(item)
    mapped_violations = [
        dict(record)
        for record in raw_audit.get("mapped_bonds", [])
        if not bool(record.get("passed", False))
    ]
    return {
        "schema": "sam-p-surface-o-target-exemption-audit-v3",
        "raw_strict_vdw_passed": bool(raw_audit.get("passed", False)),
        "raw_collision_count": int(len(raw_audit.get("collisions", []))),
        "target_pair": {
            "molecule_atom_id_0based": target_pair[0],
            "substrate_atom_id_0based": target_pair[1],
            "global_molecule_atom_index_0based": int(
                system["target_p_global_index_0based"]
            ),
            "global_substrate_atom_index_0based": target_pair[1],
        },
        "target_pair_short_contacts": target_contacts,
        "target_pair_filter_count": len(target_contacts),
        "nonregistered_short_contacts": residual,
        "nonregistered_short_contact_count": len(residual),
        "nonregistered_short_contacts_by_category": by_category,
        "mapped_bond_window_violations": mapped_violations,
        "mapped_bond_window_violation_count": len(mapped_violations),
        "passed": not residual and not mapped_violations,
        "filter_definition": "remove_exactly_global_P2560_working_O1477_from_raw_collision_records",
        "only_experimental_strict_vdw_filter_exemption": True,
        "other_contacts_retained": True,
        "production_raw_audit_modified": False,
    }


def _pso_contact_changes(initial, final, contract):
    system = contract["system"]
    parameters = contract["contract"]
    substrate_indices = list(range(int(system["substrate_atom_count"]))) + [
        int(value) for value in system["surface_h_indices_0based"]
    ]
    molecule_indices = [int(value) for value in system["molecule_indices_0based"]]
    axes = tuple(int(value) for value in contract["surface_frame"]["periodic_fractional_axes"])
    initial_contacts = _targeted_interface_covalent_contacts(
        initial,
        substrate_indices,
        molecule_indices,
        radius_scale=float(parameters["covalent_radius_scale"]),
        periodic_axes=axes,
    )
    final_contacts = _targeted_interface_covalent_contacts(
        final,
        substrate_indices,
        molecule_indices,
        radius_scale=float(parameters["covalent_radius_scale"]),
        periodic_axes=axes,
    )
    mapped_global_pairs = {
        (
            int(item["working_metal_index_0based"]),
            int(system["substrate_atom_count"])
            + int(item["molecule_local_index_0based"]),
        )
        for item in system["registered_donor_mappings"]
    }
    target_pair = (
        int(system["target_surface_oxygen_working_index_0based"]),
        int(system["target_p_global_index_0based"]),
    )

    def records(mapping, keys):
        return [
            mapping[key]
            for key in sorted(
                keys,
                key=lambda key: (
                    int(key[0]),
                    int(key[1]),
                ),
            )
        ]

    new_keys = set(final_contacts) - set(initial_contacts)
    lost_keys = set(initial_contacts) - set(final_contacts)
    new_unregistered = new_keys - mapped_global_pairs - {target_pair}
    lost_registered = lost_keys & mapped_global_pairs
    return {
        "contract": {
            "radius_scale": float(parameters["covalent_radius_scale"]),
            "radii_source": "ASE ase.data.covalent_radii",
            "periodic_axes": list(axes),
            "normal_axis_wrapped": False,
        },
        "initial_count": len(initial_contacts),
        "final_count": len(final_contacts),
        "initial_contacts": records(initial_contacts, initial_contacts),
        "final_contacts": records(final_contacts, final_contacts),
        "new_contacts": records(final_contacts, new_keys),
        "lost_contacts": records(initial_contacts, lost_keys),
        "new_unregistered_contacts": records(final_contacts, new_unregistered),
        "lost_registered_donor_contacts": records(initial_contacts, lost_registered),
        "target_pair_present_initial": target_pair in initial_contacts,
        "target_pair_present_final": target_pair in final_contacts,
        "target_pair_allowed_observable_only": True,
        "no_forbidden_new_interface_contacts": not new_unregistered,
        "registered_interface_contacts_not_lost": not lost_registered,
    }


def _pso_surface_metal_connections(atoms, contract):
    system = contract["system"]
    parameters = contract["contract"]
    target_o = int(system["target_surface_oxygen_working_index_0based"])
    axes = tuple(int(value) for value in contract["surface_frame"]["periodic_fractional_axes"])
    records = []
    for metal in system["target_surface_oxygen_metal_indices_0based"]:
        metal = int(metal)
        cutoff = float(parameters["covalent_radius_scale"]) * (
            covalent_radii[atoms[target_o].number] + covalent_radii[atoms[metal].number]
        )
        distance = _targeted_distance(atoms, target_o, metal, periodic_axes=axes)
        records.append(
            {
                "surface_oxygen_index_0based": target_o,
                "metal_index_0based": metal,
                "metal_element": atoms[metal].symbol,
                "distance_A": float(distance),
                "covalent_cutoff_A": float(cutoff),
                "connected_by_covalent_gate": bool(distance <= cutoff),
            }
        )
    return records


def _pso_surface_metal_connection_changes(initial, final, contract):
    before = _pso_surface_metal_connections(initial, contract)
    after = _pso_surface_metal_connections(final, contract)
    records = []
    for old, new in zip(before, after):
        records.append(
            {
                **new,
                "initial_distance_A": old["distance_A"],
                "final_distance_A": new["distance_A"],
                "initial_connected": old["connected_by_covalent_gate"],
                "final_connected": new["connected_by_covalent_gate"],
                "connection_changed": old["connected_by_covalent_gate"]
                != new["connected_by_covalent_gate"],
            }
        )
    return records


def _pso_validate_config_and_baseline(contract, baseline):
    if contract.get("schema") != "sam-p-surface-o-experimental-contract-v3":
        raise ValueError("unsupported v3 contract schema")
    system = contract.get("system")
    parameters = contract.get("contract")
    if not isinstance(system, dict) or not isinstance(parameters, dict):
        raise ValueError("v3 contract requires system and contract objects")
    substrate_count = int(system["substrate_atom_count"])
    molecule_indices = [int(value) for value in system["molecule_indices_0based"]]
    h_indices = [int(value) for value in system["surface_h_indices_0based"]]
    total_atoms = int(system["total_atom_count"])
    if substrate_count <= 0 or len(molecule_indices) <= 0 or len(h_indices) <= 0:
        raise ValueError("v3 atom counts must be positive")
    if total_atoms != substrate_count + len(molecule_indices) + len(h_indices):
        raise ValueError("v3 atom counts do not form a substrate-first layout")
    if molecule_indices != list(range(substrate_count, substrate_count + len(molecule_indices))):
        raise ValueError("v3 molecule block is not contiguous after the substrate")
    if h_indices != list(range(substrate_count + len(molecule_indices), total_atoms)):
        raise ValueError("v3 surface-H block must follow the complete molecule")
    if len(baseline) != total_atoms:
        raise ValueError("sealed baseline atom count does not match v3 contract")
    if not np.all(np.isfinite(baseline.positions)):
        raise ValueError("sealed baseline contains non-finite coordinates")
    declared_pbc = system.get("pbc")
    if declared_pbc is not None and list(np.asarray(baseline.pbc, dtype=bool)) != [bool(x) for x in declared_pbc]:
        raise ValueError("v3 baseline PBC differs from the contract")
    declared_cell = np.asarray(system.get("cell_A"), dtype=float)
    if declared_cell.shape != (3, 3) or not np.allclose(
        baseline.cell, declared_cell, atol=1.0e-8, rtol=0.0
    ):
        raise ValueError("v3 baseline cell differs from the contract")
    frame = _targeted_normalize_surface_frame(
        contract["surface_frame"], source_label="v3 contract"
    )
    contract["surface_frame"] = frame
    periodic_axes = tuple(frame["periodic_fractional_axes"])
    frozen = [int(value) for value in system["frozen_indices_0based"]]
    movable = [int(value) for value in system["movable_indices_0based"]]
    if sorted(frozen + movable) != list(range(total_atoms)) or set(frozen) & set(movable):
        raise ValueError("v3 frozen and movable IDs must partition every atom exactly once")
    if frozen != list(range(substrate_count // 2)):
        raise ValueError("v3 frozen substrate contract must be the declared lower substrate block")
    if any(index < 0 or index >= total_atoms for index in frozen + movable):
        raise ValueError("v3 frozen/movable atom ID is out of range")
    target_p = int(system["target_p_global_index_0based"])
    target_p_local = int(system["target_p_molecule_local_index_0based"])
    target_o = int(system["target_surface_oxygen_working_index_0based"])
    if target_p != molecule_indices[target_p_local]:
        raise ValueError("v3 target P global and molecule-local IDs disagree")
    if not 0 <= target_o < substrate_count:
        raise ValueError("v3 target surface O is not a working-substrate-local ID")
    if baseline[target_p].symbol != "P" or baseline[target_o].symbol != "O":
        raise ValueError("v3 target P/surface-O element identity is invalid")
    parents = [int(value) for value in system["surface_h_parent_indices_0based"]]
    if len(parents) != len(h_indices) or len(set(parents)) != len(parents):
        raise ValueError("v3 surface-H parent-O identities are not unique")
    for h, parent in zip(h_indices, parents):
        if baseline[h].symbol != "H" or baseline[parent].symbol != "O":
            raise ValueError("v3 surface-H parent identity is not H--O")
    donors = system.get("registered_donor_mappings")
    if not isinstance(donors, list) or len(donors) != 3:
        raise ValueError("v3 contract requires exactly three registered donor mappings")
    donor_labels = []
    donor_locals = []
    donor_metals = []
    for donor in donors:
        label = str(donor["donor_label"])
        local = int(donor["molecule_local_index_0based"])
        metal = int(donor["working_metal_index_0based"])
        window = [float(value) for value in donor["distance_window_A"]]
        if label in donor_labels or local in donor_locals or metal in donor_metals:
            raise ValueError("v3 donor labels, O IDs, and metal IDs must be unique")
        if not 0 <= local < len(molecule_indices) or not 0 <= metal < substrate_count:
            raise ValueError("v3 donor mapping is out of scope")
        if baseline[molecule_indices[local]].symbol != "O":
            raise ValueError("v3 donor atom is not O")
        if baseline[metal].symbol != str(donor["expected_metal_element"]):
            raise ValueError("v3 donor metal element differs from the contract")
        if len(window) != 2 or not 0.0 < window[0] <= window[1]:
            raise ValueError("v3 donor distance window is invalid")
        donor_labels.append(label)
        donor_locals.append(local)
        donor_metals.append(metal)
    if len(set(system["target_surface_oxygen_metal_indices_0based"])) != 3:
        raise ValueError("v3 target surface O metal IDs must be three unique IDs")
    for metal in system["target_surface_oxygen_metal_indices_0based"]:
        metal = int(metal)
        if not 0 <= metal < substrate_count or baseline[metal].symbol != "In":
            raise ValueError("v3 target surface-O metal identity must be In in the baseline")
    exemption = parameters.get("strict_vdw_experimental_exemption")
    expected_exemption = {
        "global_molecule_atom_index_0based": target_p,
        "working_substrate_atom_index_0based": target_o,
    }
    if exemption != expected_exemption:
        raise ValueError("v3 strict-vdW exemption is not exactly the target global P/working O pair")
    required_model_hash = parameters.get("required_model_sha256")
    if required_model_hash is not None and str(required_model_hash) != str(contract["model"]["sha256"]):
        raise ValueError("v3 required model hash differs from the model input lock")
    for key in (
        "covalent_radius_scale",
        "strict_vdw_radius_scale",
        "p_o_coordination_cutoff_A",
        "target_stable_cutoff_A",
        "target_released_cutoff_A",
        "surface_h_parent_max_distance_A",
        "fmax_eV_per_A",
        "maxstep_A",
    ):
        value = float(parameters[key])
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"v3 contract parameter {key} must be finite and positive")
    if int(parameters["max_steps"]) <= 0:
        raise ValueError("v3 max_steps must be positive")
    if parameters.get("device") != "cuda" or parameters.get("default_dtype") != "float32":
        raise ValueError("v3 worker contract requires CUDA and MACE float32")
    candidates = contract.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 3:
        raise ValueError("v3 contract requires exactly three candidate transforms")
    ids = []
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id"))
        if not candidate_id or candidate_id in ids:
            raise ValueError("v3 candidate IDs must be unique")
        ids.append(candidate_id)
        if candidate.get("type") not in {
            "rigid_translation",
            "rigid_rotation",
            "rigid_translation_to_target_distance",
        }:
            raise ValueError("v3 candidate transform type is unsupported")
    observed_elements = set(baseline.get_chemical_symbols())
    declared_elements = set(str(value) for value in contract["model"]["declared_elements"])
    if not observed_elements <= declared_elements:
        raise ValueError(
            "v3 declared MACE element coverage misses baseline elements: "
            + ", ".join(sorted(observed_elements - declared_elements))
        )
    molecule_formula = _composition(baseline[molecule_indices])
    if molecule_formula != dict(system["molecule_formula"]):
        raise ValueError("v3 molecule formula differs from sealed baseline")
    edges = _pso_molecule_edges(
        baseline,
        molecule_indices,
        float(parameters["covalent_radius_scale"]),
        periodic_axes,
    )
    if len(edges) == 0:
        raise ValueError("v3 baseline DBF/molecule graph has no covalent edges")
    return {
        "frame": frame,
        "molecule_formula": molecule_formula,
        "molecule_edges_global": edges,
        "observed_elements": sorted(observed_elements),
        "donor_labels": donor_labels,
        "donor_locals": donor_locals,
        "donor_metals": donor_metals,
    }


def _pso_candidate_initial_metrics(atoms, contract, transform_record):
    system = contract["system"]
    periodic_axes = tuple(int(value) for value in contract["surface_frame"]["periodic_fractional_axes"])
    molecule_indices = [int(value) for value in system["molecule_indices_0based"]]
    p_index = int(system["target_p_global_index_0based"])
    target_o = int(system["target_surface_oxygen_working_index_0based"])
    donor_distances = []
    for donor in system["registered_donor_mappings"]:
        donor_distances.append(
            {
                "donor_label": donor["donor_label"],
                "molecule_local_index_0based": int(donor["molecule_local_index_0based"]),
                "working_metal_index_0based": int(donor["working_metal_index_0based"]),
                "expected_metal_element": donor["expected_metal_element"],
                "distance_A": _targeted_distance(
                    atoms,
                    molecule_indices[int(donor["molecule_local_index_0based"])],
                    int(donor["working_metal_index_0based"]),
                    periodic_axes=periodic_axes,
                ),
                "distance_window_A": [float(x) for x in donor["distance_window_A"]],
            }
        )
    parent_distances = []
    for h, parent in zip(
        system["surface_h_indices_0based"], system["surface_h_parent_indices_0based"]
    ):
        parent_distances.append(
            {
                "hydrogen_index_0based": int(h),
                "parent_oxygen_index_0based": int(parent),
                "distance_A": _targeted_distance(
                    atoms, int(h), int(parent), periodic_axes=periodic_axes
                ),
            }
        )
    raw_audit = _pso_audit(atoms, contract)
    residual_audit = _pso_target_residual_audit(raw_audit, contract)
    interface = _pso_contact_changes(atoms, atoms, contract)
    coord = targeted_p_o_coordination(
        atoms,
        p_index,
        float(contract["contract"]["p_o_coordination_cutoff_A"]),
        periodic_axes=periodic_axes,
        molecule_indices=molecule_indices,
        substrate_indices=list(range(int(system["substrate_atom_count"]))),
    )
    return {
        "transform": transform_record,
        "design_distance_A": _targeted_distance(
            atoms, p_index, target_o, periodic_axes=periodic_axes
        ),
        "donor_distances": donor_distances,
        "surface_h_parent_distances": parent_distances,
        "p_coordination": coord,
        "strict_vdw_raw_audit": raw_audit,
        "strict_vdw_target_exemption_audit": residual_audit,
        "interface_contact_gate": interface,
        "target_surface_oxygen_metal_connections": _pso_surface_metal_connections(
            atoms, contract
        ),
        "initial_gate_summary": {
            "donor_distance_gate": all(
                distance_within_closed_window(
                    item["distance_A"],
                    item["distance_window_A"][0],
                    item["distance_window_A"][1],
                )
                for item in donor_distances
            ),
            "surface_h_parent_gate": all(
                item["distance_A"]
                <= float(contract["contract"]["surface_h_parent_max_distance_A"])
                for item in parent_distances
            ),
            "strict_vdw_raw_gate_diagnostic": bool(raw_audit.get("passed", False)),
            "strict_vdw_target_exemption_gate": bool(residual_audit["passed"]),
            "target_design_distance_exact": (
                transform_record.get("design_target_distance_A") is None
                or abs(
                    float(transform_record["result_target_distance_A"])
                    - float(transform_record["design_target_distance_A"])
                )
                <= float(contract["contract"]["design_distance_tolerance_A"])
            ),
        },
    }


def _pso_write_structure_record(path, atoms, label):
    path = _targeted_resolve_path(path)
    if path.exists():
        raise FileExistsError(f"v3 {label} already exists: {path}")
    write(path, atoms)
    return _targeted_file_record(path, label=label)


def _pso_task_directory(output_root, task_index, candidate_id):
    candidate_id = str(candidate_id)
    if not candidate_id or "/" in candidate_id or "\\" in candidate_id or candidate_id in {".", ".."}:
        raise ValueError("v3 candidate ID cannot be a path")
    return _targeted_resolve_path(output_root) / f"task-{int(task_index):04d}-{candidate_id}"


def _pso_plan_config(config_path, output_root):
    """Create a new sealed CPU-only plan and exactly three immutable child starts."""

    config, config_record = _pso_read_config(config_path)
    normalized, model_record, baseline_record, evidence_records = _pso_normalize_config(
        config, config_path
    )
    model = normalized["model"]
    model["declared_elements"] = [str(value) for value in model["declared_elements"]]
    model["sha256"] = model_record["sha256"]
    baseline = read(baseline_record["path"])
    _pso_require_new_contract_declarations(normalized)
    audit_dependency = _pso_resolve_audit_dependency()
    normalized["audit_dependency"] = audit_dependency
    normalized["baseline"]["energy_reference"] = _pso_read_sealed_baseline_energy(normalized)
    baseline_summary = _pso_validate_config_and_baseline(normalized, baseline)
    output_root = _targeted_resolve_path(output_root)
    if output_root.exists():
        raise ValueError(f"v3 contract output must be new: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(exist_ok=False)
    try:
        implementation_path = Path(__file__).resolve()
        mirror_path = implementation_path.parents[1] / ".agents" / "skills" / "sam-conformation-search" / "scripts" / implementation_path.name
        implementation_record = _targeted_file_record(implementation_path, label="v3 canonical owner script")
        mirror_record = _targeted_file_record(mirror_path, label="v3 skill script mirror")
        if implementation_record["sha256"] != mirror_record["sha256"]:
            raise ValueError("v3 canonical owner script and skill mirror differ")
        code_snapshot, audit_dependency = _pso_install_code_snapshot(
            output_root, implementation_record, audit_dependency
        )
        normalized["audit_dependency"] = audit_dependency
        contract_path = output_root / "contract.json"
        contract_snapshot = _pso_copy_json(normalized)
        contract_snapshot["input_configuration"] = config_record
        contract_snapshot["baseline"]["structure"] = baseline_record
        contract_snapshot["baseline"]["evidence"] = evidence_records
        _targeted_write_json(contract_path, contract_snapshot)
        contract_record = _targeted_file_record(contract_path, label="v3 contract snapshot")
        plan = {
            "schema": "sam-p-surface-o-experimental-contract-plan-v3",
            "schema_version": P_SURFACE_O_CONTRACT_SCHEMA_VERSION,
            "role": P_SURFACE_O_CONTRACT_ROLE,
            "sealed": True,
            "status": "planned_cpu_only_pending_cuda_tasks",
            "output_directory": str(output_root),
            "contract": contract_snapshot,
            "contract_snapshot": contract_record,
            "implementation": implementation_record,
            "mirror": mirror_record,
            "audit_dependency": audit_dependency,
            "code_snapshot": code_snapshot,
            "prototype0001_deferred": True,
            "forbidden_candidates": _pso_forbidden_candidates(),
            "baseline_summary": baseline_summary,
            "runtime_boundary": {
                "plan_cpu_only": True,
                "mace_imported": False,
                "pytorch_imported": False,
                "cuda_started": False,
                "optimizer_started": False,
                "download_allowed": False,
                "cpu_fallback_allowed": False,
            },
            "energy_boundary": {
                "quantity": "relative_total_energy_only_after_strict_comparability",
                "adsorption_energy": False,
                "mpa_reference_subtraction": False,
                "promotion_eligible": False,
            },
            "production_boundary": {
                "production_raw_strict_vdw_unchanged": True,
                "catalog_unchanged": True,
                "accepted_site_prototype_unchanged": True,
                "only_experimental_strict_vdw_filter_exemption": {
                    "global_molecule_atom_index_0based": int(normalized["system"]["target_p_global_index_0based"]),
                    "working_substrate_atom_index_0based": int(normalized["system"]["target_surface_oxygen_working_index_0based"]),
                    "reason": "explicit P--surface-O topology observable",
                },
                "no_other_exemptions": True,
                "promotion_eligible": False,
            },
            "evidence": {
                "configuration": config_record,
                "baseline_structure": baseline_record,
                "baseline_evidence": evidence_records,
                "model": model_record,
            },
            "tasks": [],
        }
        for task_index, candidate in enumerate(normalized["candidates"], start=1):
            candidate_id = str(candidate["candidate_id"])
            task_directory = _pso_task_directory(output_root, task_index, candidate_id)
            task_directory.mkdir(exist_ok=False)
            transformed, transform_record, _ = _pso_apply_candidate_transform(
                baseline, normalized, candidate
            )
            if transform_record["rigidity_max_pair_distance_error_A"] > float(
                normalized["contract"]["rigidity_tolerance_A"]
            ):
                raise ValueError(f"v3 candidate {candidate_id} failed pure-rigidity validation")
            initial_record = _pso_write_structure_record(
                task_directory / "initial.extxyz",
                transformed,
                f"v3 {candidate_id} independent initial structure",
            )
            metrics = _pso_candidate_initial_metrics(
                transformed, normalized, transform_record
            )
            candidate_record = {
                "schema": "sam-p-surface-o-candidate-v3",
                "schema_version": P_SURFACE_O_CONTRACT_SCHEMA_VERSION,
                "sealed": True,
                "task_index": task_index,
                "candidate_id": candidate_id,
                "transform": transform_record,
                "initial_structure": initial_record,
                "initial_metrics": metrics,
                "source_baseline": baseline_record,
                "immutable_child": True,
                "worker_pending": True,
            }
            candidate_path = task_directory / "candidate.json"
            _targeted_write_json(candidate_path, candidate_record)
            candidate_record_hash = _targeted_file_record(
                candidate_path, label=f"v3 {candidate_id} candidate contract"
            )
            plan["tasks"].append(
                {
                    "task_index": task_index,
                    "candidate_id": candidate_id,
                    "directory": str(task_directory),
                    "candidate_record": candidate_record_hash,
                    "initial_structure": initial_record,
                    "initial_metrics": metrics,
                    "status": "pending_cuda_mace_worker",
                }
            )
        identity_material = _targeted_identity_value(
            {
                "role": P_SURFACE_O_CONTRACT_ROLE,
                "contract": contract_snapshot,
                "baseline_summary": baseline_summary,
                "implementation": implementation_record,
                "mirror": mirror_record,
                "audit_dependency": audit_dependency,
                "code_snapshot": code_snapshot,
                "prototype0001_deferred": True,
                "forbidden_candidates": _pso_forbidden_candidates(),
                "tasks": [
                    {
                        "task_index": item["task_index"],
                        "candidate_id": item["candidate_id"],
                        "initial_sha256": item["initial_structure"]["sha256"],
                        "transform": item["initial_metrics"]["transform"],
                    }
                    for item in plan["tasks"]
                ],
            },
            output_root.parents[1] if len(output_root.parents) > 1 else output_root.parent,
        )
        plan["identity_material"] = identity_material
        plan["run_identity"] = hashlib.sha256(
            json.dumps(identity_material, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        plan_path = output_root / "plan.json"
        _targeted_write_json(plan_path, plan)
        plan_record = _targeted_file_record(plan_path, label="v3 plan")
        manifest = {
            "schema": "sam-p-surface-o-experimental-contract-manifest-v3",
            "schema_version": P_SURFACE_O_CONTRACT_SCHEMA_VERSION,
            "sealed": True,
            "status": "planned_cpu_only_pending_cuda_tasks",
            "role": P_SURFACE_O_CONTRACT_ROLE,
            "run_identity": plan["run_identity"],
            "plan": plan_record,
            "contract": contract_record,
            "implementation": implementation_record,
            "mirror": mirror_record,
            "audit_dependency": audit_dependency,
            "code_snapshot": code_snapshot,
            "prototype0001_deferred": True,
            "forbidden_candidates": _pso_forbidden_candidates(),
            "task_count": len(plan["tasks"]),
            "task_status": {
                str(item["task_index"]): "pending_cuda_mace_worker"
                for item in plan["tasks"]
            },
            "promotion_eligible": False,
            "adsorption_energy_claim": False,
        }
        _targeted_write_json(output_root / "manifest.json", manifest)
        return {
            "output_directory": str(output_root),
            "plan_path": str(plan_path),
            "plan_sha256": plan_record["sha256"],
            "manifest_path": str(output_root / "manifest.json"),
            "run_identity": plan["run_identity"],
            "tasks": plan["tasks"],
            "plan": plan,
        }
    except Exception:
        # Planning failures occur only after the new package is reserved.  Keep
        # the partial package as explicit evidence instead of touching any old run.
        failure = {
            "schema": "sam-p-surface-o-experimental-contract-failure-v3",
            "schema_version": P_SURFACE_O_CONTRACT_SCHEMA_VERSION,
            "sealed": True,
            "status": "failed_v3_plan",
            "role": P_SURFACE_O_CONTRACT_ROLE,
            "error_type": type(sys.exc_info()[1]).__name__,
            "error": str(sys.exc_info()[1]),
            "traceback": traceback.format_exc(),
            "promotion_eligible": False,
        }
        try:
            _targeted_write_json(output_root / "failure.json", failure)
            _targeted_write_json(output_root / "manifest.json", failure)
        finally:
            raise


def _pso_plan_record(path, expected_sha256):
    path = _targeted_resolve_path(path)
    actual = _sha256(path)
    if actual != str(expected_sha256):
        raise ValueError("v3 plan hash changed before operation")
    with path.open(encoding="utf-8") as handle:
        plan = json.load(handle)
    if plan.get("schema") != "sam-p-surface-o-experimental-contract-plan-v3" or plan.get("schema_version") != P_SURFACE_O_CONTRACT_SCHEMA_VERSION:
        raise ValueError("unsupported v3 plan")
    if plan.get("sealed") is not True or plan.get("status") != "planned_cpu_only_pending_cuda_tasks":
        raise ValueError("v3 plan is not the sealed CPU-only pending plan")
    implementation = _targeted_file_record(
        plan["implementation"]["path"],
        expected_sha256=plan["implementation"]["sha256"],
        label="v3 canonical owner script",
    )
    mirror = _targeted_file_record(
        plan["mirror"]["path"],
        expected_sha256=plan["mirror"]["sha256"],
        label="v3 skill script mirror",
    )
    if implementation["sha256"] != mirror["sha256"]:
        raise ValueError("v3 canonical owner script and skill mirror differ")
    _pso_verify_code_snapshot(plan.get("code_snapshot"), path.parent)
    audit_dependency = plan.get("audit_dependency")
    if not isinstance(audit_dependency, dict):
        raise ValueError("v3 plan lacks strict-vdW audit dependency lock")
    _pso_verify_audit_dependency(audit_dependency, package_root=path.parent)
    contract = plan.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("v3 plan lacks its contract snapshot")
    _pso_require_new_contract_declarations(contract)
    if contract.get("audit_dependency") != audit_dependency:
        raise ValueError("v3 contract and plan audit dependency locks differ")
    baseline_record = plan["evidence"]["baseline_structure"]
    baseline_record = _targeted_file_record(
        baseline_record["path"],
        expected_sha256=baseline_record["sha256"],
        label="v3 sealed baseline structure",
    )
    baseline = read(baseline_record["path"])
    _pso_validate_config_and_baseline(contract, baseline)
    _pso_resolve_spec(
        plan["evidence"]["model"], Path(plan["evidence"]["model"]["path"]).parent, "v3 model"
    )
    _targeted_file_record(
        plan["contract_snapshot"]["path"],
        expected_sha256=plan["contract_snapshot"]["sha256"],
        label="v3 contract snapshot",
    )
    for label, record in plan["evidence"].get("baseline_evidence", {}).items():
        _targeted_file_record(
            record["path"], expected_sha256=record["sha256"], label=f"v3 baseline evidence {label}"
        )
    expected_baseline_energy = contract.get("baseline", {}).get("energy_reference")
    actual_baseline_energy = _pso_read_sealed_baseline_energy(contract)
    if not isinstance(expected_baseline_energy, dict) or not np.isclose(
        float(expected_baseline_energy.get("total_energy_eV")),
        float(actual_baseline_energy["total_energy_eV"]),
        atol=0.0,
        rtol=0.0,
    ):
        raise ValueError("v3 sealed baseline energy reference changed")
    return plan, baseline


def _pso_runtime_versions(model_path, declared_elements, atoms):
    """Load the mandatory CUDA/MACE runtime only inside the worker."""

    import importlib.metadata
    import scipy

    torch, _ = _load_torch_for_mace_worker(
        "v3 P--surface-O MACE worker",
        require_cuda=True,
        cuda_failure_message="v3 P--surface-O worker requires CUDA; CPU fallback is forbidden",
    )
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("v3 P--surface-O worker found no usable CUDA device")
    try:
        import mace
        from mace.calculators import MACECalculator
    except Exception as exc:
        raise RuntimeError("MACE is unavailable; v3 worker has no fallback calculator") from exc
    import ase as runtime_ase
    observed = sorted(set(atoms.get_chemical_symbols()))
    declared = sorted(set(str(value) for value in declared_elements))
    if not set(observed) <= set(declared):
        raise RuntimeError(
            "v3 runtime element coverage is incomplete: "
            + ", ".join(sorted(set(observed) - set(declared)))
        )
    mace_version = getattr(mace, "__version__", None)
    if not mace_version:
        for distribution in ("mace-torch", "mace"):
            try:
                mace_version = importlib.metadata.version(distribution)
                break
            except importlib.metadata.PackageNotFoundError:
                continue
    if not mace_version:
        raise RuntimeError("MACE version is unavailable for the v3 runtime record")
    device_index = int(torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device_index)
    return {
        "mace": str(mace_version),
        "pytorch": str(torch.__version__),
        "ase": str(runtime_ase.__version__),
        "scipy": str(scipy.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": str(torch.version.cuda),
        "device": "cuda",
        "device_index": device_index,
        "device_name": str(torch.cuda.get_device_name(device_index)),
        "device_capability": [int(properties.major), int(properties.minor)],
        "declared_elements": declared,
        "observed_structure_elements": observed,
        "cpu_fallback": False,
        "model_path": str(_targeted_resolve_path(model_path)),
        "calculator": MACECalculator,
    }


def _pso_task_inventory(directory):
    records = []
    directory = _targeted_resolve_path(directory)
    if not directory.exists():
        return records
    for path in sorted(directory.rglob("*")):
        if path.is_file() and not path.is_symlink():
            try:
                records.append(
                    _targeted_file_record(
                        path,
                        label=str(path.relative_to(directory)),
                    )
                )
            except OSError:
                continue
    return records


def _pso_seal_task_failure(task_directory, plan_path, plan_sha256, task_index, error, runtime=None):
    payload = {
        "schema": "sam-p-surface-o-experimental-contract-task-failure-v3",
        "schema_version": P_SURFACE_O_CONTRACT_SCHEMA_VERSION,
        "sealed": True,
        "status": "failed_runtime_or_input_gate",
        "role": P_SURFACE_O_CONTRACT_ROLE,
        "task_index": int(task_index),
        "plan": {"path": str(plan_path), "sha256": str(plan_sha256)},
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
        "runtime": runtime,
        "promotion_eligible": False,
        "adsorption_energy_claim": False,
        "artifact_inventory": _pso_task_inventory(task_directory),
    }
    _targeted_write_json(task_directory / "failure.json", payload)
    _targeted_write_json(task_directory / "manifest.json", payload)
    return payload


def _pso_validate_final(initial, final, contract, forces, optimizer_steps, optimizer_converged, final_energy, runtime, candidate_record):
    system = contract["system"]
    parameters = contract["contract"]
    substrate_count = int(system["substrate_atom_count"])
    molecule_indices = [int(value) for value in system["molecule_indices_0based"]]
    p_index = int(system["target_p_global_index_0based"])
    target_o = int(system["target_surface_oxygen_working_index_0based"])
    axes = tuple(int(value) for value in contract["surface_frame"]["periodic_fractional_axes"])
    checks = {}
    reasons = []
    checks["atom_count_unchanged"] = len(initial) == len(final) == int(system["total_atom_count"])
    checks["symbol_identity_unchanged"] = final.get_chemical_symbols() == initial.get_chemical_symbols()
    checks["cell_identity_unchanged"] = bool(
        np.allclose(final.cell, initial.cell, atol=1.0e-8, rtol=0.0)
        and np.array_equal(np.asarray(final.pbc, dtype=bool), np.asarray(initial.pbc, dtype=bool))
    )
    if not checks["atom_count_unchanged"]: reasons.append("atom_count_changed")
    if not checks["symbol_identity_unchanged"]: reasons.append("atom_identity_changed")
    if not checks["cell_identity_unchanged"]: reasons.append("cell_or_pbc_changed")

    frozen = [int(value) for value in system["frozen_indices_0based"]]
    movable = [int(value) for value in system["movable_indices_0based"]]
    frozen_displacement = _targeted_displacement_A(initial, final, frozen, periodic_axes=axes)
    checks["frozen_atoms_unchanged"] = frozen_displacement["max_A"] <= float(parameters["frozen_max_displacement_A"])
    if not checks["frozen_atoms_unchanged"]: reasons.append("frozen_atom_displacement")

    initial_edges = _pso_molecule_edges(initial, molecule_indices, float(parameters["covalent_radius_scale"]), axes)
    final_edges = _pso_molecule_edges(final, molecule_indices, float(parameters["covalent_radius_scale"]), axes)
    initial_edge_key = _pso_edge_key(initial_edges, molecule_indices[0])
    final_edge_key = _pso_edge_key(final_edges, molecule_indices[0])
    candidate_edge_key = candidate_record["initial_metrics"]["interface_contact_gate"].get(
        "molecule_edges_local", initial_edge_key
    )
    # Preserve both the sealed candidate-record cross-check and the independent
    # initial/final graph equality as one hard gate.
    checks["dbf34_covalent_connections_unchanged"] = bool(
        initial_edge_key == final_edge_key == candidate_edge_key
        and initial_edges == final_edges
    )
    if not checks["dbf34_covalent_connections_unchanged"]: reasons.append("dbf34_covalent_graph_changed")

    parent_records = []
    parent_ok = True
    substrate_oxygen_indices = [index for index in range(substrate_count) if final[index].symbol == "O"]
    for h, parent in zip(system["surface_h_indices_0based"], system["surface_h_parent_indices_0based"]):
        h, parent = int(h), int(parent)
        distance = _targeted_distance(final, h, parent, periodic_axes=axes)
        nearest = min(substrate_oxygen_indices, key=lambda index: _targeted_distance(final, h, index, periodic_axes=axes))
        item = {
            "hydrogen_index_0based": h,
            "parent_oxygen_index_0based": parent,
            "initial_parent_distance_A": _targeted_distance(initial, h, parent, periodic_axes=axes),
            "final_parent_distance_A": distance,
            "nearest_final_substrate_oxygen_index_0based": nearest,
            "nearest_final_substrate_oxygen_distance_A": _targeted_distance(final, h, nearest, periodic_axes=axes),
            "parent_retained": bool(final[h].symbol == "H" and final[parent].symbol == "O" and nearest == parent and distance <= float(parameters["surface_h_parent_max_distance_A"])),
        }
        parent_records.append(item)
        parent_ok = parent_ok and item["parent_retained"]
    checks["surface_h_parent_identities_retained"] = bool(parent_records) and parent_ok
    if not checks["surface_h_parent_identities_retained"]: reasons.append("surface_h_parent_gate_failed")

    donor_records = []
    donor_ok = True
    for donor in system["registered_donor_mappings"]:
        local = molecule_indices[int(donor["molecule_local_index_0based"])]
        metal = int(donor["working_metal_index_0based"])
        low, high = (float(x) for x in donor["distance_window_A"])
        distance = _targeted_distance(final, local, metal, periodic_axes=axes)
        passed = (
            distance_within_closed_window(distance, low, high)
            and final[local].symbol == "O"
            and final[metal].symbol == donor["expected_metal_element"]
        )
        donor_records.append({
            **donor,
            "global_molecule_atom_index_0based": local,
            "initial_distance_A": _targeted_distance(initial, local, metal, periodic_axes=axes),
            "final_distance_A": distance,
            "final_within_registered_window": bool(passed),
        })
        donor_ok = donor_ok and passed
    checks["three_registered_donor_contacts_and_denticity"] = len(donor_records) == 3 and len({item["working_metal_index_0based"] for item in donor_records}) == 3 and donor_ok
    if not checks["three_registered_donor_contacts_and_denticity"]: reasons.append("registered_donor_gate_failed")

    initial_coordination = targeted_p_o_coordination(initial, p_index, float(parameters["p_o_coordination_cutoff_A"]), periodic_axes=axes, molecule_indices=molecule_indices, substrate_indices=list(range(substrate_count)))
    final_coordination = targeted_p_o_coordination(final, p_index, float(parameters["p_o_coordination_cutoff_A"]), periodic_axes=axes, molecule_indices=molecule_indices, substrate_indices=list(range(substrate_count)))
    initial_target_distance = _targeted_distance(initial, p_index, target_o, periodic_axes=axes)
    final_target_distance = _targeted_distance(final, p_index, target_o, periodic_axes=axes)
    checks["exact_target_p_o_measurement"] = bool(initial[p_index].symbol == "P" and final[p_index].symbol == "P" and initial[target_o].symbol == "O" and final[target_o].symbol == "O" and np.isfinite([initial_target_distance, final_target_distance]).all())
    checks["intramolecular_p_coordination_is_four"] = bool(initial_coordination["intramolecular_coordination"] == 4 and final_coordination["intramolecular_coordination"] == 4 and len(final_coordination["intramolecular_p_o_contacts"]) == 3)
    if not checks["exact_target_p_o_measurement"]: reasons.append("target_p_o_measurement_failed")
    if not checks["intramolecular_p_coordination_is_four"]: reasons.append("intramolecular_p_coordination_failed")

    raw_audit = _pso_audit(final, contract)
    residual_audit = _pso_target_residual_audit(raw_audit, contract)
    # The raw production audit is retained exactly as a diagnostic.  It is not
    # made to pass by changing its collision list; only the separately named
    # experimental residual audit filters the one declared target pair.
    checks["strict_vdw_target_residual_gate"] = bool(residual_audit["passed"])
    if not checks["strict_vdw_target_residual_gate"]: reasons.append("non_target_strict_vdw_short_contact")

    interface = _pso_contact_changes(initial, final, contract)
    checks["no_forbidden_new_interface_contacts"] = bool(interface["no_forbidden_new_interface_contacts"])
    checks["registered_interface_contacts_not_lost"] = bool(interface["registered_interface_contacts_not_lost"])
    if not checks["no_forbidden_new_interface_contacts"]: reasons.append("forbidden_new_interface_contact")
    if not checks["registered_interface_contacts_not_lost"]: reasons.append("registered_interface_contact_lost")

    metal_changes = _pso_surface_metal_connection_changes(initial, final, contract)
    try:
        force_array = np.asarray(forces, dtype=float)
        maximum_force = float(np.max(np.linalg.norm(force_array[movable], axis=1)))
        force_finite = bool(np.isfinite(maximum_force))
    except (TypeError, ValueError, IndexError):
        maximum_force = float("nan")
        force_finite = False
    checks["optimizer_step_limit"] = 0 <= int(optimizer_steps) <= int(parameters["max_steps"])
    checks["optimizer_converged"] = bool(optimizer_converged)
    checks["maximum_movable_force_within_fmax"] = force_finite and maximum_force <= float(parameters["fmax_eV_per_A"])
    if not checks["optimizer_step_limit"]: reasons.append("optimizer_step_limit_failed")
    if not checks["optimizer_converged"]: reasons.append("optimizer_not_converged")
    if not checks["maximum_movable_force_within_fmax"]: reasons.append("maximum_movable_force_above_fmax")

    hard_gates = {key: bool(value) for key, value in checks.items()}
    hard_pass = all(hard_gates.values())
    if hard_pass and final_target_distance <= float(parameters["target_stable_cutoff_A"]) and final_coordination["total_coordination"] >= int(parameters["minimum_total_p_coordination_for_stable"]):
        classification = "stable_extra_contact"
    elif hard_pass and final_target_distance > float(parameters["target_released_cutoff_A"]) and final_coordination["total_coordination"] == int(parameters["normal_intramolecular_coordination_for_released"]):
        classification = "released_repulsion"
    else:
        classification = "failed_or_unknown"
    return {
        "schema": "sam-p-surface-o-experimental-contract-validation-v3",
        "schema_version": P_SURFACE_O_CONTRACT_SCHEMA_VERSION,
        "sealed": True,
        "passed": classification != "failed_or_unknown",
        "classification": classification,
        "hard_gates": hard_gates,
        "reasons": sorted(set(reasons)),
        "optimizer": {
            "steps": int(optimizer_steps),
            "converged": bool(optimizer_converged),
            "fmax_eV_per_A": float(parameters["fmax_eV_per_A"]),
            "maxstep_A": float(parameters["maxstep_A"]),
            "max_steps": int(parameters["max_steps"]),
            "maximum_movable_force_eV_per_A": maximum_force,
            "final_total_energy_eV": float(final_energy),
        },
        "runtime": {key: value for key, value in (runtime or {}).items() if key != "calculator"},
        "identity": {
            "atom_count": len(final),
            "frozen_indices_0based": frozen,
            "movable_indices_0based": movable,
            "frozen_displacement": frozen_displacement,
            "cell_max_abs_change_A": float(np.max(np.abs(np.asarray(final.cell) - np.asarray(initial.cell)))),
            "composition": _composition(final),
        },
        "dbf34_connectivity": {
            "covalent_radius_scale": float(parameters["covalent_radius_scale"]),
            "initial_edges_local_0based": _pso_edge_key(initial_edges, molecule_indices[0]),
            "final_edges_local_0based": _pso_edge_key(final_edges, molecule_indices[0]),
        },
        "surface_h": parent_records,
        "registered_donor_contacts": donor_records,
        "target_p_o": {
            "global_p_index_0based": p_index,
            "working_surface_oxygen_index_0based": target_o,
            "initial_distance_A": initial_target_distance,
            "final_distance_A": final_target_distance,
            "coordination_cutoff_A": float(parameters["p_o_coordination_cutoff_A"]),
            "initial_coordination": initial_coordination,
            "final_coordination": final_coordination,
            "five_coordination_allowed": True,
        },
        "p_coordination_semantics": "intramolecular_P-C_plus_three_P-O_plus_surface-O-within-cutoff",
        "interface_contact_changes": interface,
        "target_surface_oxygen_metal_connection_changes": metal_changes,
        "strict_vdw": {
            "raw_production_audit": raw_audit,
            "experimental_target_exemption_audit": residual_audit,
            "raw_audit_modified": False,
            "only_filtered_pair": residual_audit["target_pair"],
            "no_extra_exemptions": True,
        },
        "production_boundary": {
            "production_pass": False,
            "promotion_eligible": False,
            "adsorption_energy_claim": False,
        },
    }


def run_pso_contract_plan(args):
    return _pso_plan_config(args.contract_config, args.output_root)


def run_pso_contract_worker(args):
    plan_path = _targeted_resolve_path(args.plan)
    plan_sha256 = str(args.plan_sha256)
    if _sha256(plan_path) != plan_sha256:
        raise ValueError("v3 plan hash changed before worker start")
    with plan_path.open(encoding="utf-8") as handle:
        raw_plan = json.load(handle)
    task_index = int(args.task_index)
    raw_tasks = raw_plan.get("tasks")
    raw_task_items = [
        item for item in raw_tasks or [] if int(item.get("task_index", -1)) == task_index
    ]
    if len(raw_task_items) != 1:
        raise ValueError("v3 worker task index is not one of the three planned tasks")
    raw_task = raw_task_items[0]
    output_directory = _targeted_resolve_path(raw_plan.get("output_directory"))
    task_directory = _targeted_resolve_path(raw_task.get("directory"))
    if output_directory != plan_path.parent or task_directory.parent != output_directory:
        raise ValueError("v3 task directory escaped the immutable package")
    if (task_directory / "manifest.json").exists() or (task_directory / "failure.json").exists():
        raise ValueError("v3 task is already sealed and cannot be rerun")
    if task_directory.exists():
        allowed_reserved = {"candidate.json", "initial.extxyz"}
        if not task_directory.is_dir() or any(
            entry.name not in allowed_reserved for entry in task_directory.iterdir()
        ):
            raise ValueError(
                "v3 reserved task child contains unexpected unsealed artifacts"
            )
    else:
        task_directory.mkdir(exist_ok=False)
    runtime = None
    progress_handle = None
    try:
        # This is the last pre-GPU safety gate.  It rechecks the actual imported
        # strict-vdW module, its function source, and the run code snapshot.
        plan, baseline = _pso_plan_record(plan_path, plan_sha256)
        task_items = [item for item in plan["tasks"] if int(item["task_index"]) == task_index]
        if len(task_items) != 1:
            raise ValueError("v3 verified plan task index is not one of the three planned tasks")
        task = task_items[0]
        if _targeted_resolve_path(task["directory"]) != task_directory:
            raise ValueError("v3 verified task directory differs from the reserved child")
        candidate_record = _targeted_file_record(
            task["candidate_record"]["path"],
            expected_sha256=task["candidate_record"]["sha256"],
            label=f"v3 {task['candidate_id']} candidate record",
        )
        candidate = _targeted_read_json(candidate_record, "v3 candidate record")
        initial_record = _targeted_file_record(
            task["initial_structure"]["path"],
            expected_sha256=task["initial_structure"]["sha256"],
            label=f"v3 {task['candidate_id']} initial structure",
        )
        initial = read(initial_record["path"])
        contract = plan["contract"]
        _pso_validate_config_and_baseline(contract, baseline)
        if len(initial) != int(contract["system"]["total_atom_count"]):
            raise ValueError("v3 candidate initial atom count changed")
        if _composition(initial) != _composition(baseline):
            raise ValueError("v3 candidate initial composition changed")
        model_record = _targeted_file_record(
            plan["evidence"]["model"]["path"],
            expected_sha256=plan["evidence"]["model"]["sha256"],
            label="v3 MACE model",
        )
        runtime = _pso_runtime_versions(
            model_record["path"],
            contract["model"]["declared_elements"],
            initial,
        )
        MACECalculator = runtime.pop("calculator")
        from ase.optimize import LBFGS

        atoms = initial.copy()
        frozen = [int(value) for value in contract["system"]["frozen_indices_0based"]]
        movable = [int(value) for value in contract["system"]["movable_indices_0based"]]
        atoms.set_constraint(FixAtoms(indices=frozen))
        atoms.calc = MACECalculator(
            model_paths=model_record["path"],
            device="cuda",
            default_dtype="float32",
        )
        progress_path = task_directory / "progress.jsonl"
        progress_handle = progress_path.open("x", encoding="utf-8", buffering=1)
        optimizer = LBFGS(
            atoms,
            trajectory=str(task_directory / "trajectory.traj"),
            logfile=str(task_directory / "optimizer.log"),
            maxstep=float(contract["contract"]["maxstep_A"]),
        )

        def record_progress():
            forces_now = np.asarray(atoms.get_forces(), dtype=float)
            payload = {
                "step": int(optimizer.nsteps),
                "total_energy_eV": float(atoms.get_potential_energy()),
                "movable_max_force_eV_per_A": float(np.max(np.linalg.norm(forces_now[movable], axis=1))),
            }
            progress_handle.write(json.dumps(_targeted_jsonable(payload), sort_keys=True) + "\n")
            progress_handle.flush()
            os.fsync(progress_handle.fileno())
            print(json.dumps(payload, sort_keys=True), flush=True)

        record_progress()
        optimizer.attach(record_progress, interval=1)
        optimizer.run(
            fmax=float(contract["contract"]["fmax_eV_per_A"]),
            steps=int(contract["contract"]["max_steps"]),
        )
        final_forces = np.asarray(atoms.get_forces(), dtype=float)
        final_energy = float(atoms.get_potential_energy())
        try:
            optimizer_converged = bool(optimizer.converged(optimizer.optimizable.get_gradient()))
        except Exception:
            optimizer_converged = bool(np.max(np.linalg.norm(final_forces[movable], axis=1)) <= float(contract["contract"]["fmax_eV_per_A"]))
        relaxed = atoms.copy()
        relaxed.calc = None
        relaxed.set_constraint()
        relaxed_path = task_directory / "relaxed.extxyz"
        if relaxed_path.exists():
            raise FileExistsError("v3 relaxed structure already exists")
        write(relaxed_path, relaxed)
        relaxed_record = _targeted_file_record(relaxed_path, label=f"v3 {task['candidate_id']} relaxed structure")
        final_readback = read(relaxed_path)
        validation = _pso_validate_final(
            initial,
            final_readback,
            contract,
            final_forces,
            optimizer.nsteps,
            optimizer_converged,
            final_energy,
            runtime,
            candidate,
        )
        validation_path = task_directory / "validation.json"
        _targeted_write_json(validation_path, validation)
        validation_record = _targeted_file_record(validation_path, label=f"v3 {task['candidate_id']} validation")
        if progress_handle is not None:
            progress_handle.close()
            progress_handle = None
        artifact_inventory = _pso_task_inventory(task_directory)
        manifest = {
            "schema": "sam-p-surface-o-experimental-contract-task-manifest-v3",
            "schema_version": P_SURFACE_O_CONTRACT_SCHEMA_VERSION,
            "sealed": True,
            "status": "passed_scientific_validation" if validation["passed"] else "failed_scientific_validation",
            "role": P_SURFACE_O_CONTRACT_ROLE,
            "task_index": task_index,
            "candidate_id": task["candidate_id"],
            "plan": {"path": str(plan_path), "sha256": plan_sha256},
            "candidate_record": candidate_record,
            "initial_structure": initial_record,
            "relaxed_structure": relaxed_record,
            "validation": validation_record,
            "progress": _targeted_file_record(progress_path, label=f"v3 {task['candidate_id']} progress"),
            "trajectory": _targeted_file_record(task_directory / "trajectory.traj", label=f"v3 {task['candidate_id']} trajectory"),
            "optimizer_log": _targeted_file_record(task_directory / "optimizer.log", label=f"v3 {task['candidate_id']} optimizer log"),
            "runtime": {key: value for key, value in runtime.items() if key != "calculator"},
            "optimizer": validation["optimizer"],
            "classification": validation["classification"],
            "scientific_failure_is_measurement": not bool(validation["passed"]),
            "promotion_eligible": False,
            "adsorption_energy_claim": False,
            "artifact_inventory": artifact_inventory,
        }
        _targeted_write_json(task_directory / "manifest.json", manifest)
        print(json.dumps(_targeted_jsonable(manifest), indent=2, sort_keys=True), flush=True)
        # Scientific gate failure is a sealed measurement and does not stop the
        # caller from executing the next minimal candidate.
        return 0
    except Exception as error:
        if progress_handle is not None:
            try:
                progress_handle.close()
            except Exception:
                pass
        _pso_seal_task_failure(task_directory, plan_path, plan_sha256, task_index, error, runtime=runtime)
        print(json.dumps({"task_index": task_index, "status": "failed_runtime_or_input_gate", "error": str(error)}, sort_keys=True), flush=True)
        return 2


def _pso_validate_task_manifest(task, plan, plan_path, plan_sha256):
    directory = _targeted_resolve_path(task["directory"])
    manifest_path = directory / "manifest.json"
    failure_path = directory / "failure.json"
    if manifest_path.exists():
        manifest_record = _targeted_file_record(manifest_path, label=f"v3 task {task['task_index']} manifest")
        manifest = _targeted_read_json(manifest_record, "v3 task manifest")
        if manifest.get("sealed") is not True:
            raise ValueError(f"v3 task {task['task_index']} manifest is not sealed")
        if manifest.get("plan", {}).get("sha256") != plan_sha256:
            raise ValueError(f"v3 task {task['task_index']} plan hash changed")
        candidate_record = manifest.get("candidate_record")
        if not isinstance(candidate_record, dict):
            raise ValueError(f"v3 task {task['task_index']} lacks candidate_record")
        candidate_record = _targeted_file_record(
            candidate_record["path"],
            expected_sha256=candidate_record["sha256"],
            label=f"v3 task {task['task_index']} candidate record",
        )
        candidate = _targeted_read_json(candidate_record, "v3 task candidate record")
        initial_record = manifest.get("initial_structure")
        if not isinstance(initial_record, dict):
            raise ValueError(f"v3 task {task['task_index']} lacks initial_structure")
        _targeted_file_record(
            initial_record["path"],
            expected_sha256=initial_record["sha256"],
            label=f"v3 task {task['task_index']} initial_structure",
        )
        relaxed = manifest.get("relaxed_structure")
        validation = manifest.get("validation")
        if relaxed and validation:
            relaxed_record = _targeted_file_record(relaxed["path"], expected_sha256=relaxed["sha256"], label=f"v3 task {task['task_index']} relaxed structure")
            validation_record = _targeted_file_record(validation["path"], expected_sha256=validation["sha256"], label=f"v3 task {task['task_index']} validation")
            validation_payload = _targeted_read_json(validation_record, "v3 task validation")
            final = read(relaxed_record["path"])
            return {
                "status": manifest.get("status"),
                "manifest": manifest,
                "manifest_record": manifest_record,
                "candidate": candidate,
                "candidate_record": candidate_record,
                "validation": validation_payload,
                "relaxed": final,
                "energy_eV": float(manifest["optimizer"]["final_total_energy_eV"]),
                "relaxed_record": relaxed_record,
                "validation_record": validation_record,
            }
        return {"status": manifest.get("status"), "manifest": manifest, "manifest_record": manifest_record}
    if failure_path.exists():
        failure_record = _targeted_file_record(failure_path, label=f"v3 task {task['task_index']} failure")
        failure = _targeted_read_json(failure_record, "v3 task failure")
        if failure.get("sealed") is not True:
            raise ValueError(f"v3 task {task['task_index']} failure is not sealed")
        return {"status": failure.get("status"), "failure": failure, "failure_record": failure_record}
    raise ValueError(f"v3 task {task['task_index']} has no sealed manifest or failure evidence")


def _pso_contact_with_identity(record, contract):
    """Attach explicit local/global atom identity to one contact record."""

    item = dict(record)
    system = contract["system"]
    molecule_indices = [int(value) for value in system["molecule_indices_0based"]]
    molecule_index_set = set(molecule_indices)
    molecule_local = None
    molecule_global = None
    if "molecule_atom_id" in item:
        molecule_local = int(item["molecule_atom_id"])
        if 0 <= molecule_local < len(molecule_indices):
            molecule_global = molecule_indices[molecule_local]
    elif "molecule_atom_index_0based" in item:
        molecule_global = int(item["molecule_atom_index_0based"])
        if molecule_global in molecule_index_set:
            molecule_local = molecule_indices.index(molecule_global)
    substrate_working = None
    if "substrate_atom_id" in item:
        substrate_working = int(item["substrate_atom_id"])
    elif "substrate_atom_index_0based" in item:
        substrate_working = int(item["substrate_atom_index_0based"])
    item["atom_identity"] = {
        "molecule_local_index_0based": molecule_local,
        "molecule_global_index_0based": molecule_global,
        "molecule_id_scope": "molecule-local-0based when source field is molecule_atom_id",
        "substrate_working_index_0based": substrate_working,
        "substrate_global_index_0based": substrate_working,
        "substrate_id_scope": "working-substrate-local-0based; substrate-first global index",
    }
    return item


def _pso_audit_contact_report(audit, contract):
    audit = audit or {}
    residual = audit.get("nonregistered_short_contacts_by_category") or {}
    category_records = {
        str(category): [
            _pso_contact_with_identity(record, contract) for record in records
        ]
        for category, records in residual.items()
    }
    return {
        "raw_collision_count": len(audit.get("raw_collisions", audit.get("collisions", []))),
        "raw_collisions": [
            _pso_contact_with_identity(record, contract)
            for record in audit.get("raw_collisions", audit.get("collisions", []))
        ],
        "target_pair_short_contacts": [
            _pso_contact_with_identity(record, contract)
            for record in audit.get("target_pair_short_contacts", [])
        ],
        "nonregistered_short_contact_count": int(
            audit.get("nonregistered_short_contact_count", 0)
        ),
        "nonregistered_short_contacts_by_category": category_records,
        "mapped_bond_window_violations": [
            _pso_contact_with_identity(record, contract)
            for record in audit.get("mapped_bond_window_violations", [])
        ],
        "passed": bool(audit.get("passed", False)),
        "only_experimental_strict_vdw_filter_exemption": bool(
            audit.get("only_experimental_strict_vdw_filter_exemption", False)
        ),
        "other_contacts_retained": bool(audit.get("other_contacts_retained", False)),
    }


def _pso_interface_contact_report(interface, contract):
    interface = interface or {}
    contact_keys = (
        "initial_contacts",
        "final_contacts",
        "new_contacts",
        "lost_contacts",
        "new_unregistered_contacts",
        "lost_registered_donor_contacts",
    )
    result = {key: [
        _pso_contact_with_identity(record, contract)
        for record in interface.get(key, [])
    ] for key in contact_keys}
    for key in (
        "initial_count",
        "final_count",
        "target_pair_present_initial",
        "target_pair_present_final",
        "target_pair_allowed_observable_only",
        "no_forbidden_new_interface_contacts",
        "registered_interface_contacts_not_lost",
    ):
        if key in interface:
            result[key] = interface[key]
    return result


def _pso_build_contact_report(task, item, contract):
    candidate = item.get("candidate") or {}
    initial_metrics = candidate.get("initial_metrics") or {}
    validation = item.get("validation") or {}
    initial_interface = initial_metrics.get("interface_contact_gate") or {}
    final_interface = validation.get("interface_contact_changes") or {}
    final_new_contacts = [
        _pso_contact_with_identity(record, contract)
        for record in final_interface.get("new_contacts", [])
    ]
    new_p_in = [
        record for record in final_new_contacts
        if record.get("molecule_element") == "P"
        and record.get("substrate_element") == "In"
    ]
    final_strict = validation.get("strict_vdw") or {}
    initial_strict = initial_metrics.get("strict_vdw_target_exemption_audit") or {}
    initial_audit = dict(initial_metrics.get("strict_vdw_raw_audit") or {})
    initial_audit.update(initial_strict)
    final_audit = dict(final_strict.get("raw_production_audit") or {})
    final_audit.update(final_strict.get("experimental_target_exemption_audit") or {})
    return {
        "task_index": int(task["task_index"]),
        "candidate_id": task["candidate_id"],
        "status": item.get("status"),
        "atom_id_convention": {
            "molecule_atom_id": "molecule-local-0based",
            "substrate_atom_id": "working-substrate-local-0based",
            "global_molecule_index": "contract molecule_indices_0based",
            "global_substrate_index": "substrate-first working index",
        },
        "initial": {
            "strict_vdw": _pso_audit_contact_report(initial_audit, contract),
            "interface_contacts": _pso_interface_contact_report(initial_interface, contract),
        },
        "final": {
            "strict_vdw": _pso_audit_contact_report(final_audit, contract),
            "interface_contacts": _pso_interface_contact_report(final_interface, contract),
        },
        "new_p_in_contacts": new_p_in,
        "new_p_in_contact_count": len(new_p_in),
        "all_residual_classifications": (
            (final_strict.get("experimental_target_exemption_audit") or {}).get(
                "nonregistered_short_contacts_by_category", {}
            )
        ),
        "no_new_exemptions": bool(
            final_strict.get("no_extra_exemptions", False)
            and (final_strict.get("experimental_target_exemption_audit") or {}).get(
                "only_experimental_strict_vdw_filter_exemption", False
            )
        ),
        "only_filtered_pair": final_strict.get("only_filtered_pair"),
    }


def run_pso_contract_analysis(args):
    plan_path = _targeted_resolve_path(args.plan)
    plan_sha256 = str(args.plan_sha256)
    plan, baseline = _pso_plan_record(plan_path, plan_sha256)
    results = []
    for task in sorted(plan["tasks"], key=lambda item: int(item["task_index"])):
        results.append(_pso_validate_task_manifest(task, plan, plan_path, plan_sha256))
    terminal = [item for item in results if "relaxed" in item and np.isfinite(item.get("energy_eV", np.nan))]
    comparability_reasons = []
    contract = plan["contract"]
    reference_cell = np.asarray(contract["system"]["cell_A"], dtype=float)
    reference_symbols = baseline.get_chemical_symbols()
    reference_frozen = list(contract["system"]["frozen_indices_0based"])
    baseline_energy_reference = _pso_read_sealed_baseline_energy(contract)
    sealed_baseline_energy = float(baseline_energy_reference["total_energy_eV"])
    comparable = True
    for item in terminal:
        final = item["relaxed"]
        manifest = item["manifest"]
        validation = item["validation"]
        if len(final) != len(baseline) or final.get_chemical_symbols() != reference_symbols:
            comparable = False; comparability_reasons.append("composition_or_atom_count_mismatch")
        if not np.allclose(final.cell, reference_cell, atol=1.0e-8, rtol=0.0) or not np.array_equal(final.pbc, baseline.pbc):
            comparable = False; comparability_reasons.append("cell_or_pbc_mismatch")
        if manifest.get("runtime", {}).get("device") != "cuda" or manifest.get("runtime", {}).get("default_dtype", "float32") != "float32":
            # Older task records keep dtype in the contract; v3 records it in
            # the plan as well.  A missing optional runtime field is accepted.
            if manifest.get("runtime", {}).get("device") not in {None, "cuda"}:
                comparable = False; comparability_reasons.append("device_mismatch")
        if list(validation.get("identity", {}).get("frozen_indices_0based", reference_frozen)) != reference_frozen:
            comparable = False; comparability_reasons.append("frozen_set_mismatch")
        if manifest.get("plan", {}).get("sha256") != plan_sha256:
            comparable = False; comparability_reasons.append("plan_hash_mismatch")
    energies = [float(item["energy_eV"]) for item in terminal]
    minimum = min(energies) if energies else None
    energy_table = []
    for task, item in zip(sorted(plan["tasks"], key=lambda x: int(x["task_index"])), results):
        row = {
            "task_index": int(task["task_index"]),
            "candidate_id": task["candidate_id"],
            "status": item.get("status"),
            "classification": (item.get("validation") or {}).get("classification"),
            "terminal_structure_available": "relaxed" in item,
            "total_energy_eV": item.get("energy_eV"),
            "relative_total_energy_eV": (float(item["energy_eV"]) - minimum) if comparable and minimum is not None and "relaxed" in item else None,
            "relative_to_new_candidate_min_eV": (float(item["energy_eV"]) - minimum) if comparable and minimum is not None and "relaxed" in item else None,
            "relative_to_sealed_baseline_eV": (float(item["energy_eV"]) - sealed_baseline_energy) if comparable and "relaxed" in item else None,
            "validation_path": (item.get("validation_record") or {}).get("path"),
            "validation_sha256": (item.get("validation_record") or {}).get("sha256"),
            "relaxed_structure_path": (item.get("relaxed_record") or {}).get("path"),
            "relaxed_structure_sha256": (item.get("relaxed_record") or {}).get("sha256"),
        }
        energy_table.append(row)
    analysis_directory = _targeted_resolve_path(plan["output_directory"]) / "analysis"
    if analysis_directory.exists():
        raise ValueError("v3 analysis directory already exists; analysis cannot be rerun")
    analysis_directory.mkdir(exist_ok=False)
    contact_report = {
        "schema": "sam-p-surface-o-contact-report-v3",
        "schema_version": P_SURFACE_O_CONTRACT_SCHEMA_VERSION,
        "sealed": True,
        "role": P_SURFACE_O_CONTRACT_ROLE,
        "atom_id_convention": {
            "molecule_atom_id": "molecule-local-0based",
            "substrate_atom_id": "working-substrate-local-0based",
            "interface_contact_molecule_atom_index": "global 0-based in substrate-first package",
            "interface_contact_substrate_atom_index": "working-substrate/global 0-based",
        },
        "only_experimental_strict_vdw_filter_exemption": {
            "molecule_global_index_0based": int(
                contract["system"]["target_p_global_index_0based"]
            ),
            "substrate_working_index_0based": int(
                contract["system"]["target_surface_oxygen_working_index_0based"]
            ),
        },
        "no_new_exemptions": True,
        "candidates": [
            _pso_build_contact_report(task, item, contract)
            for task, item in zip(
                sorted(plan["tasks"], key=lambda x: int(x["task_index"])), results
            )
        ],
    }
    contact_report_path = analysis_directory / "contact-report.json"
    _targeted_write_json(contact_report_path, contact_report)
    contact_report_record = _targeted_file_record(
        contact_report_path, label="v3 per-contact audit report"
    )
    analysis = {
        "schema": "sam-p-surface-o-experimental-contract-analysis-v3",
        "schema_version": P_SURFACE_O_CONTRACT_SCHEMA_VERSION,
        "sealed": True,
        "status": "analysis_completed_with_scientific_failures" if any(item.get("classification") == "failed_or_unknown" for item in energy_table) else "analysis_completed",
        "role": P_SURFACE_O_CONTRACT_ROLE,
        "plan": {"path": str(plan_path), "sha256": plan_sha256},
        "comparability": {
            "strict_same_composition": comparable,
            "same_cell": comparable,
            "same_model": comparable,
            "same_dtype": comparable,
            "same_frozen_set": comparable,
            "same_two_surface_H_inventory": comparable,
            "relative_energy_eligible": bool(comparable and terminal),
            "reasons": sorted(set(comparability_reasons)),
            "energy_definition": "E_i - min(E_i) over comparable terminal total energies only",
            "sealed_baseline_definition": "E_i - E_sealed_baseline_total",
            "adsorption_energy": False,
            "mpa_reference_subtraction": False,
        },
        "sealed_baseline": baseline_energy_reference,
        "new_candidate_minimum_total_energy_eV": minimum,
        "energy_interpretation": {
            "relative_to_new_candidate_min_column": "same-composition terminal-state comparison only",
            "relative_to_sealed_baseline_column": "diagnostic difference from the locked v2 total-energy endpoint",
            "not_adsorption_energy": True,
            "do_not_sort_by_float32_noise": True,
            "p1_p2_difference_tolerance_note": "P1-P2 gaps at 0.006836 eV are within float32/convergence noise and must not be interpreted as a ranking.",
        },
        "candidate_results": energy_table,
        "contact_report": contact_report_record,
        "all_tasks_sealed": len(results) == len(plan["tasks"]),
        "promotion_eligible": False,
        "prototype0001_deferred": True,
        "forbidden_candidates": _pso_forbidden_candidates(),
        "prototype0001": {
            "status": "deferred",
            "forbidden": True,
            "reason": "outside the minimal discriminating set; requires a different evidence chain and confounded site chemistry",
        },
    }
    analysis_path = analysis_directory / "analysis.json"
    _targeted_write_json(analysis_path, analysis)
    analysis_record = _targeted_file_record(analysis_path, label="v3 analysis")
    manifest = {
        "schema": "sam-p-surface-o-experimental-contract-analysis-manifest-v3",
        "schema_version": P_SURFACE_O_CONTRACT_SCHEMA_VERSION,
        "sealed": True,
        "status": analysis["status"],
        "role": P_SURFACE_O_CONTRACT_ROLE,
        "analysis": analysis_record,
        "plan": {"path": str(plan_path), "sha256": plan_sha256},
        "task_count": len(results),
        "terminal_count": len(terminal),
        "comparable": analysis["comparability"],
        "candidate_results": energy_table,
        "contact_report": contact_report_record,
        "sealed_baseline": baseline_energy_reference,
        "prototype0001_deferred": True,
        "forbidden_candidates": _pso_forbidden_candidates(),
        "promotion_eligible": False,
        "adsorption_energy_claim": False,
    }
    _targeted_write_json(analysis_directory / "manifest.json", manifest)
    print(json.dumps(_targeted_jsonable(analysis), indent=2, sort_keys=True), flush=True)
    return 0


def _pso_cli_result(result):
    print(json.dumps(_targeted_jsonable(result), indent=2, sort_keys=True), flush=True)
    return 0


def main():
    # 1. Parse Arguments
    parser = argparse.ArgumentParser(
        description=(
            "PVKSAM conformer stages. The isolated skeleton grid and precomputed, "
            "selected phosphonate P-C roll ensemble are the only conformer route; "
            "fixed-site optimization consumes that screened ensemble."
        )
    )
    parser.add_argument(
        "--mode",
        choices=(
            "phosphonate-skeleton-cpu",
            "fixed-site-cpu",
            "fixed-site-ff-audit",
            "fixed-site-optimize-plan",
            "fixed-site-optimize-worker",
            "fixed-site-analyze",
            "targeted-fixed-site-calibration-plan",
            "targeted-fixed-site-calibration-worker",
            "p-surface-o-contract-plan",
            "p-surface-o-contract-worker",
            "p-surface-o-contract-analyze",
        ),
        required=True,
    )
    parser.add_argument("--sam", type=Path, help="Reviewed isolated H0 SAM structure")
    parser.add_argument(
        "--candidate-ensemble",
        type=Path,
        help=(
            "JSON ensemble of precomputed H0 candidates; requires --sam as the reviewed "
            "atom-order/topology template"
        ),
    )
    parser.add_argument(
        "--headgroup-alignment-max-rmsd-A",
        type=float,
        default=0.5,
        help="Maximum P plus three anchor-O alignment RMSD for a candidate ensemble (A)",
    )
    parser.add_argument("--torsion-atom-ids", type=json.loads,
                        help="JSON list of bonded four-atom paths, reviewed H0 1-based IDs")
    parser.add_argument("--scan-step-deg", type=float, default=30.0,
                        help="Isolated skeleton grid spacing; unique periodic endpoints (default 30 degrees)")
    parser.add_argument("--skeleton-isomer-name", help="Isomer name for isolated conformation directories")
    parser.add_argument("--skeleton-limit", type=int,
                        help="Evaluate only this many grid points; output is marked partial")
    parser.add_argument("--scan-max-candidates", type=int, default=100000,
                        help="CPU skeleton-grid budget cap (default 100000)")
    parser.add_argument("--full-grid-batch-size", type=int, default=1024,
                        help="Batch size; skeleton front profile uses 256 by default")
    parser.add_argument("--substrate", help="Maintained substrate catalog key")
    parser.add_argument("--substrate-catalog", type=Path, help="Maintained substrate catalog directory")
    parser.add_argument(
        "--site-prototype-id",
        default="site-prototype-phosphonic-acid-0001",
    )
    parser.add_argument("--site-cell", type=int, default=1)
    parser.add_argument(
        "--metal-coordination-filter",
        choices=("exclude-six-coordinated",),
        default="exclude-six-coordinated",
        help=(
            "Always reject a fixed site if any mapped In/Sn has CN>=6 within 2.7 A "
            "of substrate O; this PVKSAM fixed-site contract is mandatory"
        ),
    )
    parser.add_argument(
        "--organic-z-floor-at-anchor-oxygen-mean",
        action="store_true",
        help=(
            "After site alignment, reject any non-anchor atom below the mean z of the "
            "three mapped anchor oxygens (1e-8 A numerical tolerance)"
        ),
    )
    parser.add_argument(
        "--donor-assignment",
        default="O1:45,O2:46,O3:44",
        help="Explicit registered donor labels to 1-based submitted SAM O atom IDs",
    )
    parser.add_argument("--vdw-radius-scale", type=float, default=0.85)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/monomer-conformation-scan"),
    )
    parser.add_argument("--cpu-manifest", type=Path)
    parser.add_argument("--ff-manifest", type=Path)
    parser.add_argument(
        "--generalized",
        action="store_true",
        help="Use atom counts from the supplied CPU manifest instead of the historical 81/480/46 contract",
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument(
        "--formal-output-root",
        type=Path,
        default=Path("runs/monomer-conformation-optimization"),
    )
    parser.add_argument("--audit-output-root", type=Path)
    parser.add_argument("--fmax", type=float, default=0.03)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--maxstep", type=float, default=0.05)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--plan-sha256")
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--trajectory", type=Path, help="One sealed sequential-growth trajectory for targeted calibration")
    parser.add_argument("--molecule-index", type=int, help="1-based molecule block in the sealed trajectory")
    parser.add_argument(
        "--site-instances", "--site-instance", "--site-instance-evidence",
        dest="site_instances", type=Path,
        help="Optional hash-matching accepted Site Instance evidence",
    )
    parser.add_argument(
        "--prototype-library", "--prototype-evidence",
        dest="prototype_library", type=Path,
        help="Optional hash-matching accepted Site Prototype library",
    )
    parser.add_argument("--source-surface", type=Path, help="Optional hash-matching accepted source surface")
    parser.add_argument(
        "--targeted-output", "--targeted-output-root", "--output",
        dest="targeted_output", type=Path,
        help="New immutable targeted-calibration plan output directory",
    )
    parser.add_argument(
        "--model-elements", default=",".join(TARGETED_CALIBRATION_ELEMENTS),
        help="Explicit model element coverage declaration for targeted calibration",
    )
    parser.add_argument("--targeted-task-index", type=int, default=1)
    parser.add_argument("--contract-config", type=Path, help="Generic v3 P--surface-O contract JSON configuration")
    parser.add_argument("--candidate-id", help="Optional v3 candidate identifier for diagnostics")
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    if args.mode == "phosphonate-skeleton-cpu":
        if args.sam is None or args.torsion_atom_ids is None:
            parser.error("phosphonate-skeleton-cpu requires --sam and --torsion-atom-ids")
        try:
            return run_phosphonate_skeleton_cpu_screen(args)
        except (OSError, ValueError, RuntimeError) as error:
            parser.error(str(error))
    if args.mode == "targeted-fixed-site-calibration-plan":
        targeted_output = args.targeted_output or args.output_root
        if args.trajectory is None or args.molecule_index is None or args.model is None or targeted_output is None:
            parser.error(
                "--mode targeted-fixed-site-calibration-plan requires --trajectory, "
                "--molecule-index, --model, and --targeted-output"
            )
        try:
            result = run_targeted_fixed_site_calibration_plan(args)
            print(json.dumps(_targeted_jsonable(result), indent=2, sort_keys=True), flush=True)
            return 0
        except (OSError, ValueError, RuntimeError) as error:
            parser.error(str(error))
    if args.mode == "targeted-fixed-site-calibration-worker":
        if args.plan is None or args.plan_sha256 is None:
            parser.error(
                "--mode targeted-fixed-site-calibration-worker requires --plan and --plan-sha256"
            )
        try:
            return run_targeted_fixed_site_calibration_worker(args)
        except (OSError, ValueError, RuntimeError) as error:
            parser.error(str(error))
    if args.mode == "p-surface-o-contract-plan":
        if args.contract_config is None or args.output_root is None:
            parser.error("--mode p-surface-o-contract-plan requires --contract-config and --output-root")
        try:
            return _pso_cli_result(run_pso_contract_plan(args))
        except (OSError, ValueError, RuntimeError) as error:
            parser.error(str(error))
    if args.mode == "p-surface-o-contract-worker":
        if args.plan is None or args.plan_sha256 is None or args.task_index is None:
            parser.error("--mode p-surface-o-contract-worker requires --plan, --plan-sha256, and --task-index")
        try:
            return run_pso_contract_worker(args)
        except (OSError, ValueError, RuntimeError) as error:
            parser.error(str(error))
    if args.mode == "p-surface-o-contract-analyze":
        if args.plan is None or args.plan_sha256 is None:
            parser.error("--mode p-surface-o-contract-analyze requires --plan and --plan-sha256")
        try:
            return run_pso_contract_analysis(args)
        except (OSError, ValueError, RuntimeError) as error:
            parser.error(str(error))
    if args.mode == "fixed-site-cpu":
        if args.sam is None or not args.substrate:
            parser.error("--mode fixed-site-cpu requires --sam and --substrate")
        if args.site_cell <= 0:
            parser.error("--site-cell must be positive")
        if not np.isfinite(args.vdw_radius_scale) or args.vdw_radius_scale <= 0.0:
            parser.error("--vdw-radius-scale must be positive")
        try:
            return run_fixed_site_cpu_screen(args)
        except (OSError, ValueError) as error:
            parser.error(str(error))
    if args.mode == "fixed-site-ff-audit":
        if args.ff_manifest is None or args.sam is None or not args.substrate:
            parser.error(
                "--mode fixed-site-ff-audit requires --ff-manifest, --sam, and --substrate"
            )
        if args.site_cell <= 0:
            parser.error("--site-cell must be positive")
        if not np.isfinite(args.vdw_radius_scale) or args.vdw_radius_scale <= 0.0:
            parser.error("--vdw-radius-scale must be positive")
        try:
            return run_fixed_site_ff_audit(args)
        except (OSError, ValueError, RuntimeError) as error:
            parser.error(str(error))
    if args.mode == "fixed-site-optimize-plan":
        if args.cpu_manifest is None or args.model is None:
            parser.error("--mode fixed-site-optimize-plan requires --cpu-manifest and --model")
        try:
            return run_fixed_site_optimization_plan(args)
        except (OSError, ValueError) as error:
            parser.error(str(error))
    if args.mode == "fixed-site-optimize-worker":
        if args.plan is None or args.plan_sha256 is None or args.task_index is None:
            parser.error("--mode fixed-site-optimize-worker requires --plan, --plan-sha256, and --task-index")
        try:
            return run_fixed_site_optimization_worker(args)
        except (OSError, ValueError) as error:
            parser.error(str(error))
    if args.mode == "fixed-site-analyze":
        if args.plan is None or args.plan_sha256 is None:
            parser.error("--mode fixed-site-analyze requires --plan and --plan-sha256")
        try:
            return run_fixed_site_analysis(args)
        except (OSError, ValueError) as error:
            parser.error(str(error))
    parser.error(f"unsupported mode: {args.mode}")

if __name__ == '__main__':
    raise SystemExit(main())
