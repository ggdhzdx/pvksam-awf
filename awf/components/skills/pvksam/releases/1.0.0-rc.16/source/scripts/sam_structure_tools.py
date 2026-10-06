"""Topology utilities for chemically safe SAM/substrate reconstruction."""

from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass

import numpy as np
from ase.geometry import find_mic
from ase.neighborlist import neighbor_list, primitive_neighbor_list
from ase.data import atomic_numbers, covalent_radii
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull


DISTANCE_WINDOW_COMPARISON_ULPS = 8


def closed_distance_window_tolerance_A(
    distance_A: float,
    minimum_A: float,
    maximum_A: float,
) -> float:
    """Return a machine-precision tolerance, not a physical window expansion."""

    values = tuple(float(value) for value in (distance_A, minimum_A, maximum_A))
    _, minimum_A, maximum_A = values
    if not math.isfinite(minimum_A + maximum_A) or maximum_A < minimum_A:
        raise ValueError("distance window must be finite and ordered")
    finite_scale = max(
        [abs(value) for value in values if math.isfinite(value)],
        default=0.0,
    )
    return DISTANCE_WINDOW_COMPARISON_ULPS * math.ulp(finite_scale)


def distance_within_closed_window(
    distance_A: float,
    minimum_A: float,
    maximum_A: float,
) -> bool:
    """Test an inclusive distance window with only ULP-level roundoff repair."""

    distance_A = float(distance_A)
    minimum_A = float(minimum_A)
    maximum_A = float(maximum_A)
    tolerance_A = closed_distance_window_tolerance_A(
        distance_A,
        minimum_A,
        maximum_A,
    )
    return bool(
        math.isfinite(distance_A)
        and minimum_A - tolerance_A <= distance_A <= maximum_A + tolerance_A
    )


@dataclass(frozen=True)
class MoleculeSpec:
    formula: dict[str, int]
    anchor_element: str = "P"
    headgroup_element: str = "O"
    headgroup_count: int = 3
    headgroup_cutoff: float = 1.85
    skeleton_formula: dict[str, int] | None = None
    hydrogen_parent_elements: tuple[str, ...] = ("C", "N")
    chain_element: str = "C"
    chain_length: int = 4
    core_elements: tuple[str, ...] = ("C", "N", "S")
    marker_element: str | None = "S"

    def resolved_skeleton_formula(self) -> dict[str, int]:
        if self.skeleton_formula is not None:
            return dict(self.skeleton_formula)
        result = dict(self.formula)
        result.pop("H", None)
        remaining_headgroup = result.get(self.headgroup_element, 0) - self.headgroup_count
        if remaining_headgroup > 0:
            result[self.headgroup_element] = remaining_headgroup
        else:
            result.pop(self.headgroup_element, None)
        return result

    @property
    def hydrogen_count(self) -> int:
        return int(self.formula.get("H", 0))


@dataclass(frozen=True)
class MoleculeComponent:
    indices: tuple[int, ...]
    anchor_index: int
    headgroup_indices: tuple[int, ...]
    heavy_skeleton: tuple[int, ...]
    h_indices: tuple[int, ...]

    # Temporary source compatibility for callers that used phosphonate names.
    @property
    def p_index(self) -> int:
        return self.anchor_index

    @property
    def o_indices(self) -> tuple[int, ...]:
        return self.headgroup_indices


def _skeleton_cutoff(symbol_a: str, symbol_b: str) -> float | None:
    overrides = {
        frozenset(("C", "C")): 1.75,
        frozenset(("C", "N")): 1.70,
        frozenset(("C", "S")): 2.05,
        frozenset(("C", "P")): 2.15,
    }
    pair = frozenset((symbol_a, symbol_b))
    if pair in overrides:
        return overrides[pair]
    try:
        radius_sum = (
            covalent_radii[atomic_numbers[symbol_a]]
            + covalent_radii[atomic_numbers[symbol_b]]
        )
    except KeyError:
        return None
    return min(2.20, 1.20 * float(radius_sum))


def _mic_distance_vectors(vectors: np.ndarray, cell: np.ndarray) -> np.ndarray:
    inv_cell = np.linalg.inv(cell)
    fractional = vectors @ inv_cell
    fractional -= np.round(fractional)
    return fractional @ cell


def molecular_components(atoms, spec: MoleculeSpec) -> list[MoleculeComponent]:
    """Return one validated molecular component for every anchor atom.

    The heavy skeleton is found first. Headgroup atoms are then selected around
    each anchor. Hydrogens are assigned only through configured parent
    elements, so transferred surface hydroxyl protons remain substrate.
    """

    symbols = np.asarray(atoms.get_chemical_symbols())
    cell = np.asarray(atoms.cell)
    anchor_indices = np.flatnonzero(symbols == spec.anchor_element)
    if not len(anchor_indices):
        raise ValueError(f"No {spec.anchor_element} anchor atoms found")

    skeleton_formula = spec.resolved_skeleton_formula()
    skeleton_elements = set(skeleton_formula)

    adjacency = [set() for _ in atoms]
    i_values, j_values, distances = neighbor_list("ijd", atoms, 2.2)
    for i, j, distance in zip(i_values, j_values, distances):
        if i >= j:
            continue
        symbol_i, symbol_j = symbols[i], symbols[j]
        if symbol_i not in skeleton_elements or symbol_j not in skeleton_elements:
            continue
        threshold = _skeleton_cutoff(symbol_i, symbol_j)
        if threshold is not None and distance < threshold:
            adjacency[int(i)].add(int(j))
            adjacency[int(j)].add(int(i))

    skeletons: list[set[int]] = []
    for anchor_index_raw in anchor_indices:
        anchor_index = int(anchor_index_raw)
        visited = {anchor_index}
        queue = deque([anchor_index])
        while queue:
            current = queue.popleft()
            for adjacent in adjacency[current]:
                if adjacent not in visited:
                    visited.add(adjacent)
                    queue.append(adjacent)
        formula = Counter(symbols[sorted(visited)])
        expected = skeleton_formula
        if dict(sorted(formula.items())) != expected:
            raise ValueError(
                f"Anchor atom {anchor_index} has invalid heavy skeleton {dict(formula)}; "
                f"expected {expected}"
            )
        skeletons.append(visited)

    used_heavy: set[int] = set()
    for skeleton in skeletons:
        overlap = used_heavy.intersection(skeleton)
        if overlap:
            raise ValueError(f"Heavy skeletons overlap at atoms {sorted(overlap)[:10]}")
        used_heavy.update(skeleton)

    headgroup_groups: list[tuple[int, ...]] = []
    headgroup_indices = np.flatnonzero(symbols == spec.headgroup_element)
    for anchor_index_raw in anchor_indices:
        anchor_index = int(anchor_index_raw)
        distances_to_headgroup = atoms.get_distances(
            anchor_index, headgroup_indices, mic=True
        )
        order = np.argsort(distances_to_headgroup)
        selected = [
            int(headgroup_indices[position])
            for position in order
            if distances_to_headgroup[position] < spec.headgroup_cutoff
        ][: spec.headgroup_count]
        if len(selected) != spec.headgroup_count:
            raise ValueError(
                f"Anchor atom {anchor_index} has {len(selected)} headgroup atoms "
                f"below {spec.headgroup_cutoff:.3f} A; expected {spec.headgroup_count}"
            )
        headgroup_groups.append(tuple(selected))

    all_selected_headgroup = [item for group in headgroup_groups for item in group]
    if len(set(all_selected_headgroup)) != len(all_selected_headgroup):
        raise ValueError("A headgroup atom was assigned to more than one anchor")

    # Assign organic H atoms to the nearest C/N heavy atom belonging to a
    # skeleton.  Surface O-H protons are intentionally excluded.
    h_indices = np.flatnonzero(symbols == "H")
    # Global capacitated assignment: every molecule receives exactly the
    # configured number of H atoms while each H belongs to at most one molecule. The unchosen
    # H atoms are surface hydroxyl protons.  This avoids local nearest-owner
    # swaps in tightly packed finite-temperature structures.
    owner_costs = np.empty((len(skeletons), len(h_indices)), dtype=float)
    for owner, skeleton in enumerate(skeletons):
        organic_heavy = np.asarray(
            sorted(
                index
                for index in skeleton
                if symbols[index] in spec.hydrogen_parent_elements
            ),
            dtype=int,
        )
        for h_position, h_index_raw in enumerate(h_indices):
            vectors = atoms.positions[organic_heavy] - atoms.positions[int(h_index_raw)]
            vectors = _mic_distance_vectors(vectors, cell)
            owner_costs[owner, h_position] = float(
                np.min(np.linalg.norm(vectors, axis=1))
            )
    slot_owners = np.repeat(
        np.arange(len(skeletons), dtype=int), spec.hydrogen_count
    )
    slot_costs = owner_costs[slot_owners]
    row_indices, h_positions = linear_sum_assignment(slot_costs)
    assigned_by_owner: list[list[tuple[float, int]]] = [
        [] for _ in skeletons
    ]
    for row, h_position in zip(row_indices, h_positions):
        owner = int(slot_owners[row])
        distance = float(owner_costs[owner, h_position])
        assigned_by_owner[owner].append((distance, int(h_indices[h_position])))

    components = []
    for owner, (p_index_raw, skeleton, oxygens) in enumerate(
        zip(anchor_indices, skeletons, headgroup_groups)
    ):
        assigned_h = sorted(assigned_by_owner[owner])
        if len(assigned_h) != spec.hydrogen_count:
            raise ValueError(
                f"Anchor atom {int(p_index_raw)} has {len(assigned_h)} organic H "
                f"atoms; expected {spec.hydrogen_count}"
            )
        if assigned_h and assigned_h[-1][0] > 1.80:
            raise ValueError(
                f"Anchor atom {int(p_index_raw)} requires an implausible H assignment at "
                f"{assigned_h[-1][0]:.3f} A"
            )
        h_group = tuple(index for _, index in assigned_h)
        indices = tuple(sorted(set(skeleton).union(oxygens, h_group)))
        formula = dict(sorted(Counter(symbols[list(indices)]).items()))
        if formula != dict(sorted(spec.formula.items())):
            raise ValueError(
                f"Anchor atom {int(p_index_raw)} component formula {formula}; "
                f"expected {dict(sorted(spec.formula.items()))}"
            )
        components.append(
            MoleculeComponent(
                indices=indices,
                anchor_index=int(p_index_raw),
                headgroup_indices=oxygens,
                heavy_skeleton=tuple(sorted(skeleton)),
                h_indices=h_group,
            )
        )
    return components


def recover_substrate_and_headgroups(
    atoms,
    spec: MoleculeSpec,
    substrate_elements: tuple[str, ...],
    surface_h_per_molecule: int = 0,
    surface_parent_element: str = "O",
    surface_h_repair_cutoff: float = 1.30,
    surface_h_max_distance: float = 2.20,
    surface_h_bond: float = 0.98,
    require_complete_skeleton: bool = True,
):
    """Recover substrate atoms and anchor/headgroup targets from a template.

    Organic hydrogen assignment is deliberately skipped. This makes the
    function suitable for a damaged molecular layer: only the configured
    substrate elements, selected surface H atoms, and adsorption headgroups
    survive into a rebuilt candidate.
    """

    symbols = np.asarray(atoms.get_chemical_symbols())
    cell = np.asarray(atoms.cell)
    anchor_indices = np.flatnonzero(symbols == spec.anchor_element)
    skeleton_formula = spec.resolved_skeleton_formula()
    skeleton_elements = set(skeleton_formula)
    adjacency = [set() for _ in atoms]
    i_values, j_values, distances = neighbor_list("ijd", atoms, 2.2)
    for i, j, distance in zip(i_values, j_values, distances):
        if i >= j:
            continue
        symbol_i, symbol_j = symbols[i], symbols[j]
        if symbol_i not in skeleton_elements or symbol_j not in skeleton_elements:
            continue
        threshold = _skeleton_cutoff(symbol_i, symbol_j)
        if threshold is not None and distance < threshold:
            adjacency[int(i)].add(int(j))
            adjacency[int(j)].add(int(i))

    headgroups = []
    selected_headgroup_atoms: set[int] = set()
    for anchor_index_raw in anchor_indices:
        anchor_index = int(anchor_index_raw)
        skeleton = {anchor_index}
        queue = deque([anchor_index])
        while queue:
            current = queue.popleft()
            for adjacent in adjacency[current]:
                if adjacent not in skeleton:
                    skeleton.add(adjacent)
                    queue.append(adjacent)
        formula = dict(sorted(Counter(symbols[sorted(skeleton)]).items()))
        if formula != dict(sorted(skeleton_formula.items())):
            if require_complete_skeleton:
                raise ValueError(
                    f"Anchor atom {anchor_index} has invalid target heavy skeleton {formula}; "
                    f"expected {skeleton_formula}"
                )
            anchor_count = sum(
                symbols[index] == spec.anchor_element for index in skeleton
            )
            if anchor_count != 1:
                raise ValueError(
                    f"Anchor atom {anchor_index} is connected to {anchor_count} "
                    f"{spec.anchor_element} atoms in damaged-template mode"
                )
        candidates = np.flatnonzero(symbols == spec.headgroup_element)
        distances = atoms.get_distances(anchor_index, candidates, mic=True)
        order = np.argsort(distances)
        selected = tuple(
            int(candidates[position])
            for position in order
            if distances[position] < spec.headgroup_cutoff
        )[: spec.headgroup_count]
        if len(selected) != spec.headgroup_count:
            raise ValueError(
                f"Anchor atom {anchor_index} has {len(selected)} target headgroup bonds; "
                f"expected {spec.headgroup_count}"
            )
        if selected_headgroup_atoms.intersection(selected):
            raise ValueError("Target headgroup assignment overlaps")
        selected_headgroup_atoms.update(selected)
        indices = tuple(sorted(skeleton.union(selected)))
        headgroups.append(
            MoleculeComponent(
                indices=indices,
                anchor_index=anchor_index,
                headgroup_indices=selected,
                heavy_skeleton=tuple(sorted(skeleton)),
                h_indices=(),
            )
        )

    # Surface protons are H atoms covalently bonded to substrate O atoms.  H
    # atoms belonging to the damaged old organic layer are never retained.
    substrate_parent_atoms = [
        int(index)
        for index in np.flatnonzero(symbols == surface_parent_element)
        if int(index) not in selected_headgroup_atoms
    ]
    surface_h_candidates = []
    for h_index_raw in np.flatnonzero(symbols == "H"):
        h_index = int(h_index_raw)
        vectors = atoms.positions[substrate_parent_atoms] - atoms.positions[h_index]
        vectors = _mic_distance_vectors(vectors, cell)
        surface_h_candidates.append(
            (float(np.min(np.linalg.norm(vectors, axis=1))), h_index)
        )
    surface_h_candidates.sort()
    expected_surface_h = len(anchor_indices) * surface_h_per_molecule
    surface_h_selected = surface_h_candidates[:expected_surface_h]
    if len(surface_h_selected) != expected_surface_h:
        raise ValueError(
            f"Found {len(surface_h_selected)} surface H candidates; expected {expected_surface_h}"
        )
    if surface_h_selected and surface_h_selected[-1][0] > surface_h_max_distance:
        raise ValueError(
            f"The {expected_surface_h}th surface O-H assignment is implausible at "
            f"{surface_h_selected[-1][0]:.3f} A"
        )
    surface_h = [index for _, index in surface_h_selected]
    surface_h_repairs = []
    for distance, h_index in surface_h_selected:
        if distance <= surface_h_repair_cutoff:
            continue
        vectors = atoms.positions[substrate_parent_atoms] - atoms.positions[h_index]
        vectors = _mic_distance_vectors(vectors, cell)
        nearest_position = int(np.argmin(np.linalg.norm(vectors, axis=1)))
        parent_index = substrate_parent_atoms[nearest_position]
        # vectors points H -> O; reverse it to retain the observed O -> H
        # direction while restoring a normal hydroxyl bond length.
        o_to_h = -vectors[nearest_position]
        o_to_h /= np.linalg.norm(o_to_h)
        atoms.positions[h_index] = atoms.positions[parent_index] + surface_h_bond * o_to_h
        surface_h_repairs.append(
            {
                "H_index": h_index,
                "parent_index": parent_index,
                "old_distance_A": distance,
                "new_distance_A": surface_h_bond,
            }
        )

    substrate_indices = sorted(
        [
            int(index)
            for index in np.flatnonzero(np.isin(symbols, substrate_elements))
            if int(index) not in selected_headgroup_atoms
        ]
        + surface_h
    )
    return substrate_indices, headgroups, surface_h_repairs


def unwrap_component(
    atoms,
    component: MoleculeComponent,
    hydrogen_parent_elements: tuple[str, ...] = ("C", "N"),
) -> tuple[np.ndarray, list[str]]:
    """Return one Cartesian-continuous molecule, ordered by component.indices."""

    indices = list(component.indices)
    symbols_all = np.asarray(atoms.get_chemical_symbols())
    symbols = [str(symbols_all[index]) for index in indices]
    positions = atoms.positions
    cell = np.asarray(atoms.cell)
    local_set = set(indices)
    adjacency = {index: set() for index in indices}

    # Heavy skeleton edges.
    for offset, i in enumerate(component.heavy_skeleton):
        for j in component.heavy_skeleton[offset + 1 :]:
            threshold = _skeleton_cutoff(symbols_all[i], symbols_all[j])
            if threshold is None:
                continue
            vector = _mic_distance_vectors((positions[j] - positions[i])[None, :], cell)[0]
            if np.linalg.norm(vector) < threshold:
                adjacency[i].add(j)
                adjacency[j].add(i)

    # P-O edges.
    for oxygen in component.o_indices:
        adjacency[component.p_index].add(oxygen)
        adjacency[oxygen].add(component.p_index)

    # Attach every organic H to its closest C/N within the component.
    organic_heavy = [
        index
        for index in component.heavy_skeleton
        if symbols_all[index] in hydrogen_parent_elements
    ]
    for hydrogen in component.h_indices:
        vectors = positions[organic_heavy] - positions[hydrogen]
        vectors = _mic_distance_vectors(vectors, cell)
        distances = np.linalg.norm(vectors, axis=1)
        parent = organic_heavy[int(np.argmin(distances))]
        adjacency[hydrogen].add(parent)
        adjacency[parent].add(hydrogen)

    unwrapped_by_index = {component.p_index: positions[component.p_index].copy()}
    queue = deque([component.p_index])
    while queue:
        current = queue.popleft()
        for adjacent in adjacency[current]:
            if adjacent in unwrapped_by_index:
                continue
            vector = _mic_distance_vectors(
                (positions[adjacent] - positions[current])[None, :], cell
            )[0]
            unwrapped_by_index[adjacent] = unwrapped_by_index[current] + vector
            queue.append(adjacent)
    if set(unwrapped_by_index) != local_set:
        missing = sorted(local_set - set(unwrapped_by_index))
        raise ValueError(f"Failed to unwrap component; missing atoms {missing[:10]}")
    return np.asarray([unwrapped_by_index[index] for index in indices]), symbols


def component_metrics(
    atoms, component: MoleculeComponent, spec: MoleculeSpec
) -> dict[str, float]:
    coordinates, symbols_list = unwrap_component(
        atoms, component, spec.hydrogen_parent_elements
    )
    symbols = np.asarray(symbols_list)
    global_to_local = {global_index: local for local, global_index in enumerate(component.indices)}

    # Walk outward from the anchor along the configured chain element.
    p_local = global_to_local[component.p_index]
    chain_locals = np.flatnonzero(symbols == spec.chain_element)
    if len(chain_locals) < spec.chain_length:
        raise ValueError(
            f"Need {spec.chain_length} {spec.chain_element} chain atoms; "
            f"found {len(chain_locals)}"
        )
    distances_pc = np.linalg.norm(coordinates[chain_locals] - coordinates[p_local], axis=1)
    chain = [int(chain_locals[int(np.argmin(distances_pc))])]
    for _ in range(spec.chain_length - 1):
        remaining = [index for index in chain_locals if int(index) not in chain]
        distances_next = [
            np.linalg.norm(coordinates[int(index)] - coordinates[chain[-1]])
            for index in remaining
        ]
        chain.append(int(remaining[int(np.argmin(distances_next))]))

    core = [
        index
        for index, symbol in enumerate(symbols)
        if symbol in spec.core_elements and index not in chain
    ]
    core_coordinates = coordinates[core]
    centered = core_coordinates - np.mean(core_coordinates, axis=0)
    _, _, vh = np.linalg.svd(centered)
    normal = vh[2] / np.linalg.norm(vh[2])
    vertical_plane_angle = float(
        np.degrees(np.arccos(np.clip(abs(normal[2]), 0.0, 1.0)))
    )
    result = {"vertical_plane_angle_deg": vertical_plane_angle}
    if spec.marker_element:
        marker_locals = np.flatnonzero(symbols == spec.marker_element)
        if len(marker_locals):
            heavy = np.flatnonzero(symbols != "H")
            marker_z = float(np.mean(coordinates[marker_locals, 2]))
            result["marker_exposure_A"] = float(
                np.max(coordinates[heavy, 2]) - marker_z
            )
    return result


def surface_frame_coordinates(vectors, frame: dict) -> np.ndarray:
    """Project Cartesian row vectors onto a registered Surface Frame.

    The returned columns are the signed components along ``u``, ``v``, and the
    outward surface normal.  A Surface Frame is treated as scientific input and
    must therefore be finite, orthonormal, and right-handed rather than being
    silently repaired.
    """

    try:
        basis_vectors = np.asarray(
            [
                frame["u_cartesian_unit"],
                frame["v_cartesian_unit"],
                frame["outward_normal_cartesian_unit"],
            ],
            dtype=float,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Surface Frame requires explicit u, v, and normal vectors") from exc
    if basis_vectors.shape != (3, 3) or not np.all(np.isfinite(basis_vectors)):
        raise ValueError("Surface Frame vectors must be finite three-vectors")
    gram = basis_vectors @ basis_vectors.T
    right_handed = np.cross(basis_vectors[0], basis_vectors[1])
    if not np.allclose(gram, np.eye(3), atol=1.0e-8, rtol=0.0) or not np.allclose(
        right_handed, basis_vectors[2], atol=1.0e-8, rtol=0.0
    ):
        raise ValueError("Surface Frame must be orthonormal right-handed")
    cartesian = np.asarray(vectors, dtype=float)
    if cartesian.ndim == 1:
        cartesian = cartesian.reshape(1, -1)
    if cartesian.ndim != 2 or cartesian.shape[1] != 3:
        raise ValueError("Cartesian vectors must have shape (N, 3)")
    if not np.all(np.isfinite(cartesian)):
        raise ValueError("Cartesian vectors must be finite")
    return cartesian @ basis_vectors.T


def periodic_vdw_collision_audit(
    *,
    molecule_positions,
    molecule_symbols,
    molecule_atom_ids,
    substrate_positions,
    substrate_symbols,
    substrate_atom_ids,
    cell,
    periodic_axes,
    surface_frame: dict,
    radii_A: dict[str, float],
    radius_scale: float,
    mapped_bond_windows_A: dict[tuple[int, int], tuple[float, float]] | None = None,
) -> dict:
    """Audit exact partial-PBC molecule/substrate vdW overlaps.

    One ASE ``primitive_neighbor_list`` query uses per-atom scaled vdW radii,
    so arbitrarily skew periodic lattices are enumerated without a fixed image
    stencil or a Python ``N_molecule * N_substrate`` loop.  Only the declared
    two surface axes are periodic; the remaining slab-normal axis is not
    wrapped.  Exact mapped pairs are measured separately with ASE ``find_mic``
    because their validation windows may extend beyond the vdW neighbor cutoff.
    """

    molecule_positions = np.asarray(molecule_positions, dtype=float)
    substrate_positions = np.asarray(substrate_positions, dtype=float)
    molecule_symbols = [str(value) for value in molecule_symbols]
    substrate_symbols = [str(value) for value in substrate_symbols]
    molecule_atom_ids = [int(value) for value in molecule_atom_ids]
    substrate_atom_ids = [int(value) for value in substrate_atom_ids]
    if molecule_positions.shape != (len(molecule_symbols), 3) or len(
        molecule_atom_ids
    ) != len(molecule_symbols):
        raise ValueError("Molecule positions, symbols, and atom IDs must align")
    if substrate_positions.shape != (len(substrate_symbols), 3) or len(
        substrate_atom_ids
    ) != len(substrate_symbols):
        raise ValueError("Substrate positions, symbols, and atom IDs must align")
    if not np.isfinite(molecule_positions).all() or not np.isfinite(
        substrate_positions
    ).all():
        raise ValueError("Collision-audit positions must be finite")
    if len(set(molecule_atom_ids)) != len(molecule_atom_ids):
        raise ValueError("Molecule atom IDs must be unique")
    if len(set(substrate_atom_ids)) != len(substrate_atom_ids):
        raise ValueError("Substrate atom IDs must be unique")

    cell = np.asarray(cell, dtype=float)
    if (
        cell.shape != (3, 3)
        or not np.isfinite(cell).all()
        or abs(float(np.linalg.det(cell))) <= 1.0e-12
    ):
        raise ValueError("Periodic collision audit requires a nonsingular finite 3x3 cell")
    axes = tuple(int(value) for value in periodic_axes)
    if len(axes) != 2 or len(set(axes)) != 2 or any(axis not in range(3) for axis in axes):
        raise ValueError("periodic_axes must identify two distinct cell directions")
    pbc = np.zeros(3, dtype=bool)
    pbc[list(axes)] = True
    surface_frame_coordinates(np.zeros((1, 3)), surface_frame)

    if (
        isinstance(radius_scale, (bool, np.bool_))
        or not isinstance(radius_scale, (int, float, np.integer, np.floating))
        or not np.isfinite(radius_scale)
    ):
        raise ValueError("radius_scale must be a finite positive number")
    radius_scale = float(radius_scale)
    if radius_scale <= 0.0:
        raise ValueError("radius_scale must be a finite positive number")
    required_elements = sorted(set(molecule_symbols + substrate_symbols))
    missing = [element for element in required_elements if element not in radii_A]
    if missing:
        raise ValueError("Missing van-der-Waals radii for: " + ", ".join(missing))
    resolved_radii = {element: float(radii_A[element]) for element in required_elements}
    if any(not np.isfinite(value) or value <= 0.0 for value in resolved_radii.values()):
        raise ValueError("van-der-Waals radii must be finite positive numbers")

    mapped_windows = {}
    for key, window in (mapped_bond_windows_A or {}).items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError("Mapped bond keys must be molecule/substrate atom-ID pairs")
        if not isinstance(window, (tuple, list)) or len(window) != 2:
            raise ValueError("Mapped bond windows must contain minimum and maximum distance")
        minimum, maximum = (float(window[0]), float(window[1]))
        if not 0.0 < minimum <= maximum or not np.isfinite([minimum, maximum]).all():
            raise ValueError("Mapped bond windows must be finite positive ranges")
        normalized_key = (int(key[0]), int(key[1]))
        if normalized_key in mapped_windows:
            raise ValueError("Mapped bond atom-ID pairs must be unique")
        mapped_windows[normalized_key] = (minimum, maximum)
    molecule_index_by_id = {
        atom_id: index for index, atom_id in enumerate(molecule_atom_ids)
    }
    substrate_index_by_id = {
        atom_id: index for index, atom_id in enumerate(substrate_atom_ids)
    }
    unknown_pairs = [
        pair
        for pair in mapped_windows
        if pair[0] not in molecule_index_by_id
        or pair[1] not in substrate_index_by_id
    ]
    if unknown_pairs:
        raise ValueError(f"Mapped bond atom IDs are absent from the audit: {unknown_pairs}")

    inverse_cell = np.linalg.inv(cell)
    mapped_bonds = []
    mapped_pass_by_indices = {}
    mapped_find_mic_query_count = 0
    if mapped_windows:
        ordered_mapped_pairs = sorted(
            mapped_windows,
            key=lambda pair: (
                molecule_index_by_id[pair[0]],
                substrate_index_by_id[pair[1]],
            ),
        )
        mapped_indices = [
            (
                molecule_index_by_id[molecule_atom_id],
                substrate_index_by_id[substrate_atom_id],
            )
            for molecule_atom_id, substrate_atom_id in ordered_mapped_pairs
        ]
        raw_vectors = np.asarray(
            [
                molecule_positions[molecule_index]
                - substrate_positions[substrate_index]
                for molecule_index, substrate_index in mapped_indices
            ],
            dtype=float,
        )
        mic_vectors, mic_distances = find_mic(raw_vectors, cell=cell, pbc=pbc)
        mapped_find_mic_query_count = 1
        mic_vectors = np.asarray(mic_vectors, dtype=float).reshape(-1, 3)
        mic_distances = np.asarray(mic_distances, dtype=float).reshape(-1)
        image_fractions = (raw_vectors - mic_vectors) @ inverse_cell
        images = np.rint(image_fractions).astype(int)
        if not np.allclose(image_fractions, images, atol=1.0e-8, rtol=0.0):
            raise RuntimeError("ASE MIC vector cannot be reconstructed by a lattice image")
        nonperiodic_axes = [axis for axis in range(3) if axis not in axes]
        if np.any(images[:, nonperiodic_axes] != 0):
            raise RuntimeError("ASE MIC unexpectedly wrapped a nonperiodic cell axis")
        reconstructed = raw_vectors - images @ cell
        if not np.allclose(reconstructed, mic_vectors, atol=1.0e-8, rtol=0.0):
            raise RuntimeError("Recorded substrate image does not reconstruct ASE MIC")
        local_vectors = surface_frame_coordinates(mic_vectors, surface_frame)
        for position, (
            pair_ids,
            pair_indices,
            distance,
            local,
            image,
        ) in enumerate(
            zip(
                ordered_mapped_pairs,
                mapped_indices,
                mic_distances,
                local_vectors,
                images,
            )
        ):
            molecule_atom_id, substrate_atom_id = pair_ids
            molecule_index, substrate_index = pair_indices
            minimum, maximum = mapped_windows[pair_ids]
            threshold = radius_scale * (
                resolved_radii[molecule_symbols[molecule_index]]
                + resolved_radii[substrate_symbols[substrate_index]]
            )
            lateral = float(np.linalg.norm(local[:2]))
            normal_signed = float(local[2])
            comparison_tolerance_A = closed_distance_window_tolerance_A(
                distance,
                minimum,
                maximum,
            )
            passed = distance_within_closed_window(distance, minimum, maximum)
            mapped_pass_by_indices[pair_indices] = passed
            mapped_bonds.append(
                {
                    "reason": "mapped_donor_metal_bond",
                    "molecule_atom_id": molecule_atom_id,
                    "molecule_element": molecule_symbols[molecule_index],
                    "substrate_atom_id": substrate_atom_id,
                    "substrate_element": substrate_symbols[substrate_index],
                    "lateral_separation_A": lateral,
                    "normal_separation_A": abs(normal_signed),
                    "signed_normal_separation_A": normal_signed,
                    "distance_A": float(distance),
                    "accepted_distance_window_A": [minimum, maximum],
                    "comparison_tolerance_A": comparison_tolerance_A,
                    "passed": passed,
                    "vdw_threshold_A": float(threshold),
                    "vdw_clearance_ratio": float(distance) / float(threshold),
                    "substrate_periodic_image": image.tolist(),
                }
            )

    combined_positions = np.vstack((molecule_positions, substrate_positions))
    combined_symbols = molecule_symbols + substrate_symbols
    per_atom_cutoffs = radius_scale * np.asarray(
        [resolved_radii[symbol] for symbol in combined_symbols], dtype=float
    )
    first, second, displacement, shift = primitive_neighbor_list(
        "ijDS",
        pbc,
        cell,
        combined_positions,
        per_atom_cutoffs,
        self_interaction=False,
    )
    first = np.asarray(first, dtype=int)
    second = np.asarray(second, dtype=int)
    displacement = np.asarray(displacement, dtype=float)
    shift = np.asarray(shift, dtype=int)
    molecule_count = len(molecule_positions)
    forward = (first < molecule_count) & (second >= molecule_count)
    reverse = (first >= molecule_count) & (second < molecule_count)
    cross = forward | reverse

    # Canonicalize both directed neighbor-list rows to molecule -> substrate,
    # retaining each exact periodic image once.  This loop scales with returned
    # neighbor evidence, never with the full Cartesian cross product.
    image_evidence = {}
    for row in np.flatnonzero(cross):
        if forward[row]:
            molecule_index = int(first[row])
            substrate_index = int(second[row] - molecule_count)
            local_cartesian = -displacement[row]
            substrate_image = shift[row]
        else:
            molecule_index = int(second[row])
            substrate_index = int(first[row] - molecule_count)
            local_cartesian = displacement[row]
            substrate_image = -shift[row]
        evidence_key = (
            molecule_index,
            substrate_index,
            *(int(value) for value in substrate_image),
        )
        image_evidence[evidence_key] = (
            np.asarray(local_cartesian, dtype=float),
            np.asarray(substrate_image, dtype=int),
        )

    overlap_images_by_pair = {}
    for evidence_key, (local_cartesian, substrate_image) in image_evidence.items():
        molecule_index, substrate_index = evidence_key[:2]
        threshold = radius_scale * (
            resolved_radii[molecule_symbols[molecule_index]]
            + resolved_radii[substrate_symbols[substrate_index]]
        )
        distance = float(np.linalg.norm(local_cartesian))
        if not distance < threshold:
            continue
        pair = (molecule_index, substrate_index)
        overlap_images_by_pair.setdefault(pair, []).append(
            (distance, tuple(int(value) for value in substrate_image), local_cartesian)
        )

    collisions = []
    for (molecule_index, substrate_index), overlap_images in sorted(
        overlap_images_by_pair.items()
    ):
        if mapped_pass_by_indices.get((molecule_index, substrate_index), False):
            continue
        distance, image_tuple, local_cartesian = min(
            overlap_images, key=lambda item: (item[0], item[1])
        )
        substrate_image = np.asarray(image_tuple, dtype=int)
        raw_vector = (
            molecule_positions[molecule_index] - substrate_positions[substrate_index]
        )
        reconstructed = raw_vector - substrate_image @ cell
        if not np.allclose(reconstructed, local_cartesian, atol=1.0e-8, rtol=0.0):
            raise RuntimeError("Recorded substrate image does not reconstruct neighbor vector")
        local = surface_frame_coordinates(local_cartesian, surface_frame)[0]
        lateral = float(np.linalg.norm(local[:2]))
        normal_signed = float(local[2])
        threshold = radius_scale * (
            resolved_radii[molecule_symbols[molecule_index]]
            + resolved_radii[substrate_symbols[substrate_index]]
        )
        required_normal = float(
            np.sqrt(max(float(threshold) ** 2 - lateral**2, 0.0))
        )
        collisions.append(
            {
                "reason": "vdw_overlap",
                "molecule_atom_id": molecule_atom_ids[molecule_index],
                "molecule_element": molecule_symbols[molecule_index],
                "substrate_atom_id": substrate_atom_ids[substrate_index],
                "substrate_element": substrate_symbols[substrate_index],
                "lateral_separation_A": lateral,
                "normal_separation_A": abs(normal_signed),
                "signed_normal_separation_A": normal_signed,
                "distance_A": float(distance),
                "threshold_A": float(threshold),
                "required_normal_separation_A": required_normal,
                "substrate_periodic_image": substrate_image.tolist(),
                "overlapping_periodic_image_count": len(overlap_images),
            }
        )
    collisions.sort(
        key=lambda record: (
            record["distance_A"],
            record["molecule_atom_id"],
            record["substrate_atom_id"],
            record["substrate_periodic_image"],
        )
    )

    mapped_violations = [record for record in mapped_bonds if not record["passed"]]
    audited_distances = [record["distance_A"] for record in mapped_bonds + collisions]
    audited_ratios = [
        (
            record["vdw_clearance_ratio"]
            if "vdw_clearance_ratio" in record
            else record["distance_A"] / record["threshold_A"]
        )
        for record in mapped_bonds + collisions
    ]
    return {
        "passed": not collisions and not mapped_violations,
        "radius_scale": radius_scale,
        "periodic_axes": list(axes),
        "normal_axis_wrapped": False,
        "collision_count": len(collisions),
        "collisions": collisions,
        "mapped_bond_count": len(mapped_bonds),
        "mapped_bond_violation_count": len(mapped_violations),
        "mapped_bonds": mapped_bonds,
        "minimum_audited_distance_A": (
            min(audited_distances) if audited_distances else None
        ),
        "minimum_audited_clearance_ratio": (
            min(audited_ratios) if audited_ratios else None
        ),
        "minimum_diagnostics_definition": (
            "minimum_over_mapped_bonds_and_vdw_collision_neighbor_evidence_only"
        ),
        "neighbor_search": {
            "method": "ase.neighborlist.primitive_neighbor_list_per_atom_vdw_cutoffs",
            "primitive_neighbor_list_query_count": 1,
            "mapped_find_mic_query_count": mapped_find_mic_query_count,
            "cross_pair_count": len(molecule_positions) * len(substrate_positions),
            "returned_unique_cross_image_count": len(image_evidence),
            "full_cartesian_pair_matrix_materialized": False,
            "python_cross_pair_loop": False,
            "python_neighbor_evidence_loop": True,
            "collision_completeness": "all_cross_pairs_and_periodic_images_within_pair_vdw_cutoff",
        },
    }


def filled_outer_envelope_area(
    *,
    positions,
    symbols,
    surface_frame: dict,
    radii_A: dict[str, float],
    radius_scale: float = 1.0,
    boundary_samples_per_atom: int = 720,
) -> dict:
    """Return the filled convex outer envelope of projected vdW disks.

    Each disk boundary is sampled deterministically and the two-dimensional
    convex hull is filled, so internal holes and gaps are counted as occupied.
    The sampling resolution is returned as audit data rather than hidden.
    """

    positions = np.asarray(positions, dtype=float)
    symbols = [str(value) for value in symbols]
    if positions.shape != (len(symbols), 3) or not len(symbols):
        raise ValueError("Footprint positions and symbols must describe at least one atom")
    if (
        not isinstance(boundary_samples_per_atom, int)
        or isinstance(boundary_samples_per_atom, bool)
        or boundary_samples_per_atom < 12
    ):
        raise ValueError("boundary_samples_per_atom must be an integer of at least 12")
    if not isinstance(radius_scale, (int, float)) or not np.isfinite(radius_scale):
        raise ValueError("radius_scale must be a finite positive number")
    radius_scale = float(radius_scale)
    if radius_scale <= 0.0:
        raise ValueError("radius_scale must be a finite positive number")
    missing = sorted(set(symbols) - set(radii_A))
    if missing:
        raise ValueError("Missing van-der-Waals radii for: " + ", ".join(missing))
    radii = np.asarray([float(radii_A[symbol]) for symbol in symbols]) * radius_scale
    if np.any(~np.isfinite(radii)) or np.any(radii <= 0.0):
        raise ValueError("van-der-Waals radii must be finite positive numbers")
    centers = surface_frame_coordinates(positions, surface_frame)[:, :2]
    angles = np.linspace(
        0.0, 2.0 * np.pi, boundary_samples_per_atom, endpoint=False
    )
    circle = np.column_stack((np.cos(angles), np.sin(angles)))
    boundary_points = np.concatenate(
        [center + radius * circle for center, radius in zip(centers, radii)], axis=0
    )
    hull = ConvexHull(boundary_points)
    hull_vertices = np.asarray(boundary_points[hull.vertices], dtype=float)
    bounds = np.asarray([
        [hull_vertices[:, 0].min(), hull_vertices[:, 1].min()],
        [hull_vertices[:, 0].max(), hull_vertices[:, 1].max()],
    ])
    return {
        "area_A2": float(hull.volume),
        "definition": "convex_hull_of_projected_vdw_disks",
        "numerical_method": "sample_disk_boundaries_then_scipy_spatial_convex_hull",
        "boundary_samples_per_atom": boundary_samples_per_atom,
        "radius_scale": radius_scale,
        "atom_count": len(symbols),
        "hull_vertex_count": len(hull.vertices),
        "hull_vertices_uv_A": hull_vertices.tolist(),
        "bounds_uv_A": bounds.tolist(),
    }


def filled_outer_envelope_metrics(
    *,
    positions,
    symbols,
    surface_frame: dict,
    radii_A: dict[str, float],
    radius_scale: float = 1.0,
    boundary_samples_per_atom: int = 720,
) -> dict:
    """Return finite shape metrics for the filled outer vdW envelope.

    The envelope is the same sampled convex hull returned by
    :func:`filled_outer_envelope_area`.  Perimeter and uniform-area second
    moments are evaluated from that filled polygon, so internal gaps remain
    occupied and do not enter the metrics.  The principal orientation is
    explicitly modulo ``pi``; its ``cos(2*theta)``/``sin(2*theta)`` pair is
    included for deterministic orientation clustering without a 0/180 degree
    discontinuity.
    """

    footprint = filled_outer_envelope_area(
        positions=positions,
        symbols=symbols,
        surface_frame=surface_frame,
        radii_A=radii_A,
        radius_scale=radius_scale,
        boundary_samples_per_atom=boundary_samples_per_atom,
    )
    vertices = np.asarray(footprint["hull_vertices_uv_A"], dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
        raise ValueError("Filled envelope hull must contain at least three vertices")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("Filled envelope hull vertices must be finite")
    first = vertices
    second = np.roll(vertices, -1, axis=0)
    cross = first[:, 0] * second[:, 1] - second[:, 0] * first[:, 1]
    signed_area = 0.5 * float(np.sum(cross))
    if not np.isfinite(signed_area) or abs(signed_area) <= 1.0e-14:
        raise ValueError("Filled envelope polygon must have positive area")
    if signed_area < 0.0:
        vertices = vertices[::-1]
        first = vertices
        second = np.roll(vertices, -1, axis=0)
        cross = first[:, 0] * second[:, 1] - second[:, 0] * first[:, 1]
    area = 0.5 * float(np.sum(cross))
    if area <= 0.0 or not np.isfinite(area):
        raise ValueError("Filled envelope polygon orientation is invalid")

    centroid = np.sum((first + second) * cross[:, None], axis=0) / (6.0 * area)
    integral_x2 = float(
        np.sum((first[:, 0] ** 2 + first[:, 0] * second[:, 0] + second[:, 0] ** 2) * cross)
        / 12.0
    )
    integral_y2 = float(
        np.sum((first[:, 1] ** 2 + first[:, 1] * second[:, 1] + second[:, 1] ** 2) * cross)
        / 12.0
    )
    integral_xy = float(
        np.sum(
            (
                first[:, 0] * second[:, 1]
                + 2.0 * first[:, 0] * first[:, 1]
                + 2.0 * second[:, 0] * second[:, 1]
                + second[:, 0] * first[:, 1]
            )
            * cross
        )
        / 24.0
    )
    covariance = np.asarray(
        [
            [integral_x2 / area - centroid[0] ** 2, integral_xy / area - centroid[0] * centroid[1]],
            [integral_xy / area - centroid[0] * centroid[1], integral_y2 / area - centroid[1] ** 2],
        ],
        dtype=float,
    )
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if not np.all(np.isfinite(eigenvalues)) or np.any(eigenvalues < -1.0e-10):
        raise ValueError("Filled envelope second moments are not finite positive")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    order = np.argsort(eigenvalues)[::-1]
    principal = eigenvalues[order]
    major = float(principal[0])
    minor = float(principal[1])
    if major <= 1.0e-14:
        raise ValueError("Filled envelope principal moment is degenerate")
    aspect_ratio = math.sqrt(max(major / max(minor, 1.0e-14), 1.0))
    if not np.isfinite(aspect_ratio):
        raise ValueError("Filled envelope aspect ratio is non-finite")
    anisotropy = float((major - minor) / (major + minor))
    if major - minor > 1.0e-12:
        direction = eigenvectors[:, order[0]]
        orientation = float(math.atan2(float(direction[1]), float(direction[0])) % math.pi)
        orientation_defined = True
    else:
        orientation = 0.0
        orientation_defined = False
    perimeter = float(np.sum(np.linalg.norm(second - first, axis=1)))
    compactness = float(4.0 * math.pi * area / (perimeter * perimeter))
    if not np.all(np.isfinite([perimeter, compactness, anisotropy, orientation])):
        raise ValueError("Filled envelope shape metrics must be finite")
    return {
        **footprint,
        "area_A2": float(area),
        "perimeter_A": perimeter,
        "compactness": compactness,
        "second_moment_centroid_uv_A": [float(value) for value in centroid],
        "second_moment_covariance_A2": covariance.tolist(),
        "second_moment_principal_A2": [major, minor],
        "anisotropy": anisotropy,
        "aspect_ratio": aspect_ratio,
        "principal_orientation_rad_mod_pi": orientation,
        "principal_orientation_deg_mod_pi": float(np.degrees(orientation)),
        "principal_orientation_cos2": float(np.cos(2.0 * orientation)),
        "principal_orientation_sin2": float(np.sin(2.0 * orientation)),
        "orientation_defined": orientation_defined,
        "moment_definition": "uniform_area_filled_convex_envelope_central_second_moment",
    }


def _require_shapely():
    """Lazily import Shapely for projection-aware geometry.

    Uniform sequential growth must import and run without constructing any
    projection geometry, so Shapely/GEOS are loaded only when a projection
    operation is actually requested.
    """

    try:
        from shapely import affinity
        from shapely.geometry import Point, Polygon, box
        from shapely.ops import unary_union
    except ImportError as exc:
        raise RuntimeError(
            "projection-aware selection requires Shapely/GEOS"
        ) from exc
    return affinity, Point, Polygon, box, unary_union


def _surface_lattice_uv(cell, periodic_axes, surface_frame):
    """Return the Surface-Frame 2x2 lattice and its fractional area scale.

    The two registered surface-periodic cell rows are projected to the Surface
    Frame UV plane; the columns of the returned matrix are the lattice vectors
    a_uv and b_uv.  The area scale is ``abs(det(lattice))`` and converts unit
    square (fractional) areas into squared Angstroms.
    """

    cell = np.asarray(cell, dtype=float)
    if cell.shape != (3, 3) or not np.all(np.isfinite(cell)):
        raise ValueError("cell must be a finite 3x3 matrix")
    axes = tuple(periodic_axes)
    if (
        len(axes) != 2
        or any(axis not in (0, 1, 2) for axis in axes)
        or axes[0] == axes[1]
    ):
        raise ValueError("periodic_axes must name two distinct cell axes")
    a_uv = surface_frame_coordinates(cell[axes[0]][None, :], surface_frame)[0, :2]
    b_uv = surface_frame_coordinates(cell[axes[1]][None, :], surface_frame)[0, :2]
    lattice = np.column_stack([a_uv, b_uv])
    if not np.all(np.isfinite(lattice)):
        raise ValueError("surface lattice must be finite")
    determinant = float(np.linalg.det(lattice))
    if not np.isfinite(determinant) or abs(determinant) < 1.0e-12:
        raise ValueError("surface lattice must be nonsingular")
    return lattice, abs(determinant)


def periodic_surface_polygon(
    vertices_uv_A, *, cell, periodic_axes, surface_frame
):
    """Return the periodic image of a projected footprint clipped to one cell.

    Vertices live in the Surface Frame UV plane.  The registered cell rows are
    projected to that frame to form a finite nonsingular 2x2 lattice; polygon
    vertices are mapped to fractional coordinates, every integer image whose
    bounds intersect the unit square is enumerated, clipped to the unit square,
    and unioned.  Returns ``(geometry, area_scale_A2)`` where ``area_scale_A2``
    converts fractional unit-square areas to squared Angstroms.  Invalid or
    area-losing geometry is rejected rather than silently repaired.
    """

    _, Point, Polygon, box, unary_union = _require_shapely()
    vertices_uv_A = np.asarray(vertices_uv_A, dtype=float)
    if (
        vertices_uv_A.ndim != 2
        or vertices_uv_A.shape[1] != 2
        or len(vertices_uv_A) < 3
    ):
        raise ValueError(
            "vertices_uv_A must be at least three 2D surface-frame vertices"
        )
    if not np.all(np.isfinite(vertices_uv_A)):
        raise ValueError("vertices_uv_A must be finite")
    lattice, area_scale_A2 = _surface_lattice_uv(cell, periodic_axes, surface_frame)
    inverse = np.linalg.inv(lattice)
    fractional = np.asarray(vertices_uv_A) @ inverse.T
    base = Polygon(fractional.tolist())
    if not base.is_valid or base.area <= 0.0:
        raise ValueError("periodic polygon must be a valid positive-area polygon")
    min_frac = fractional.min(axis=0)
    max_frac = fractional.max(axis=0)
    unit_square = box(0.0, 0.0, 1.0, 1.0)
    fragments = []
    image_u_lo = int(np.ceil(-max_frac[0]))
    image_u_hi = int(np.floor(1.0 - min_frac[0]))
    image_v_lo = int(np.ceil(-max_frac[1]))
    image_v_hi = int(np.floor(1.0 - min_frac[1]))
    for image_u in range(image_u_lo, image_u_hi + 1):
        for image_v in range(image_v_lo, image_v_hi + 1):
            shift = np.array([float(image_u), float(image_v)])
            clipped = Polygon((fractional + shift).tolist()).intersection(
                unit_square
            )
            if not clipped.is_empty:
                fragments.append(clipped)
    if not fragments:
        raise ValueError("periodic polygon must intersect the unit surface cell")
    geometry = unary_union(fragments)
    if base.area <= 1.0 and geometry.area < base.area - 1.0e-9:
        raise ValueError("periodic polygon lost area during cell clipping")
    return geometry, area_scale_A2


def periodic_polygon_coverage_mask(geometry, anchor_fractional_uv):
    """Return an integer bitset of anchors covered by the periodic geometry.

    Anchor fractional coordinates are reduced modulo one so periodic images
    are covered deterministically; Shapely ``covers`` counts boundary points
    as covered.  Bit ``i`` is set when anchor ``i`` is covered.
    """

    _, Point, _, _, _ = _require_shapely()
    anchors = np.asarray(anchor_fractional_uv, dtype=float)
    if anchors.ndim != 2 or anchors.shape[1] != 2:
        raise ValueError(
            "anchor_fractional_uv must be an (n, 2) fractional-coordinate array"
        )
    if not np.all(np.isfinite(anchors)):
        raise ValueError("anchor_fractional_uv must be finite")
    wrapped = np.mod(anchors, 1.0)
    mask = 0
    for index, (u, v) in enumerate(wrapped):
        if geometry.covers(Point(float(u), float(v))):
            mask |= 1 << index
    return mask


def periodic_union_increment_A2(current_union, candidate_geometry, area_scale_A2):
    """Return the exact union-area increment and the merged union.

    The increment is the fractional union growth multiplied by
    ``area_scale_A2``.  An increment below ``-1.0e-9 A2`` is a hard error;
    smaller negative floating-point noise is clamped to zero.
    """

    _, _, _, _, unary_union = _require_shapely()
    if (
        not isinstance(area_scale_A2, (int, float))
        or not np.isfinite(area_scale_A2)
    ):
        raise ValueError("area_scale_A2 must be a finite positive number")
    area_scale_A2 = float(area_scale_A2)
    if area_scale_A2 <= 0.0:
        raise ValueError("area_scale_A2 must be a finite positive number")
    if current_union is None:
        increment_A2 = candidate_geometry.area * area_scale_A2
        merged = candidate_geometry
    else:
        merged = unary_union([current_union, candidate_geometry])
        increment_A2 = (merged.area - current_union.area) * area_scale_A2
    if increment_A2 < -1.0e-9:
        raise ValueError("periodic union increment must not be negative")
    if increment_A2 < 0.0:
        increment_A2 = 0.0
    return float(increment_A2), merged


def _positive_area_polygon_components(geometry):
    """Return every positive-area Polygon without repairing the geometry."""

    _, _, Polygon, _, _ = _require_shapely()
    if geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry] if geometry.area > 0.0 else []
    components = []
    for child in getattr(geometry, "geoms", ()):  # MultiPolygon/GeometryCollection
        components.extend(_positive_area_polygon_components(child))
    return components


def _validated_surface_lattice_uv_A(surface_lattice_uv_A):
    lattice = np.asarray(surface_lattice_uv_A, dtype=float)
    if lattice.shape != (2, 2) or not np.all(np.isfinite(lattice)):
        raise ValueError("surface_lattice_uv_A must be a finite 2x2 matrix")
    determinant = float(np.linalg.det(lattice))
    if abs(determinant) < 1.0e-12:
        raise ValueError("surface_lattice_uv_A must be nonsingular")
    return lattice


def _validated_fractional_origin(cell_origin_fractional):
    origin = np.asarray(cell_origin_fractional, dtype=float)
    if origin.shape != (2,) or not np.all(np.isfinite(origin)):
        raise ValueError("cell_origin_fractional must be a finite two-vector")
    return origin


def _merged_fractional_intervals(intervals, coordinate_tolerance):
    merged = []
    for start, stop in sorted(intervals):
        start = float(start)
        stop = float(stop)
        if stop - start <= coordinate_tolerance:
            continue
        if merged and start <= merged[-1][1] + coordinate_tolerance:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])
    return [tuple(interval) for interval in merged]


def _fractional_boundary_parts(
    geometry, *, cell_origin_fractional, seam_coordinate_tolerance_fractional
):
    """Split polygon boundary segments into cell seams and true interior lines."""

    origin_u, origin_v = (
        float(value) for value in _validated_fractional_origin(cell_origin_fractional)
    )
    tolerance = float(seam_coordinate_tolerance_fractional)
    intervals = {"u_low": [], "u_high": [], "v_low": [], "v_high": []}
    interior_segments = []
    for polygon in _positive_area_polygon_components(geometry):
        for ring in [polygon.exterior, *polygon.interiors]:
            coordinates = list(ring.coords)
            for first, second in zip(coordinates, coordinates[1:]):
                first_xy = np.asarray(first[:2], dtype=float)
                second_xy = np.asarray(second[:2], dtype=float)
                if np.linalg.norm(second_xy - first_xy) <= tolerance:
                    continue
                x1, y1 = first_xy
                x2, y2 = second_xy
                if abs(x1 - origin_u) <= tolerance and abs(x2 - origin_u) <= tolerance:
                    intervals["u_low"].append(tuple(sorted((y1, y2))))
                elif abs(x1 - (origin_u + 1.0)) <= tolerance and abs(
                    x2 - (origin_u + 1.0)
                ) <= tolerance:
                    intervals["u_high"].append(tuple(sorted((y1, y2))))
                elif abs(y1 - origin_v) <= tolerance and abs(y2 - origin_v) <= tolerance:
                    intervals["v_low"].append(tuple(sorted((x1, x2))))
                elif abs(y1 - (origin_v + 1.0)) <= tolerance and abs(
                    y2 - (origin_v + 1.0)
                ) <= tolerance:
                    intervals["v_high"].append(tuple(sorted((x1, x2))))
                else:
                    interior_segments.append((first_xy, second_xy))
    return (
        {
            key: _merged_fractional_intervals(value, tolerance)
            for key, value in intervals.items()
        },
        interior_segments,
    )


def _tolerance_clustered_interval_xor_length(
    left, right, coordinate_tolerance
):
    """Return paired-seam interval XOR after local endpoint clustering.

    Endpoint clusters have diameter at most ``coordinate_tolerance``; this
    prevents transitive snapping across a wider range.  The canonical intervals
    are then partitioned at every clustered endpoint and XOR is measured only
    on atomic intervals covered by exactly one seam side.
    """

    tagged_endpoints = []
    for side, intervals in (("left", left), ("right", right)):
        for interval_index, (start, stop) in enumerate(intervals):
            tagged_endpoints.extend(
                [
                    (float(start), side, interval_index, 0),
                    (float(stop), side, interval_index, 1),
                ]
            )
    if not tagged_endpoints:
        return 0.0, {
            "endpoint_clusters_fractional": [],
            "canonical_left_intervals_fractional": [],
            "canonical_right_intervals_fractional": [],
        }

    tagged_endpoints.sort(key=lambda record: record)
    clusters = []
    current = []
    for endpoint in tagged_endpoints:
        if (
            current
            and endpoint[0] - current[0][0] > coordinate_tolerance
        ):
            clusters.append(current)
            current = []
        current.append(endpoint)
    clusters.append(current)

    canonical_endpoints = {}
    cluster_audit = []
    for cluster in clusters:
        minimum = float(cluster[0][0])
        maximum = float(cluster[-1][0])
        representative = 0.5 * (minimum + maximum)
        for _, side, interval_index, endpoint_index in cluster:
            canonical_endpoints[(side, interval_index, endpoint_index)] = (
                representative
            )
        cluster_audit.append(
            {
                "minimum": minimum,
                "maximum": maximum,
                "representative": representative,
                "member_count": len(cluster),
            }
        )

    def canonical_intervals(side, intervals):
        canonical = []
        for interval_index in range(len(intervals)):
            start = canonical_endpoints[(side, interval_index, 0)]
            stop = canonical_endpoints[(side, interval_index, 1)]
            if stop > start:
                canonical.append((start, stop))
        return _merged_fractional_intervals(canonical, 0.0)

    canonical_left = canonical_intervals("left", left)
    canonical_right = canonical_intervals("right", right)
    boundaries = sorted(
        {
            endpoint
            for intervals in (canonical_left, canonical_right)
            for interval in intervals
            for endpoint in interval
        }
    )

    def covered(intervals, coordinate):
        return any(start < coordinate < stop for start, stop in intervals)

    xor_length = 0.0
    for start, stop in zip(boundaries, boundaries[1:]):
        if stop <= start:
            continue
        midpoint = 0.5 * (start + stop)
        if covered(canonical_left, midpoint) != covered(
            canonical_right, midpoint
        ):
            xor_length += stop - start
    return float(xor_length), {
        "endpoint_clusters_fractional": cluster_audit,
        "canonical_left_intervals_fractional": [
            list(interval) for interval in canonical_left
        ],
        "canonical_right_intervals_fractional": [
            list(interval) for interval in canonical_right
        ],
    }


def _periodic_torus_perimeter_audit(
    geometry,
    surface_lattice_uv_A,
    *,
    cell_origin_fractional=(0.0, 0.0),
    seam_coordinate_tolerance_fractional=1.0e-12,
    geometry_area_tolerance_fractional=1.0e-12,
):
    """Measure polygon boundary on a flat torus using the physical lattice metric."""

    _, _, _, box, _ = _require_shapely()
    lattice = _validated_surface_lattice_uv_A(surface_lattice_uv_A)
    origin = _validated_fractional_origin(cell_origin_fractional)
    for name, value in (
        (
            "seam_coordinate_tolerance_fractional",
            seam_coordinate_tolerance_fractional,
        ),
        ("geometry_area_tolerance_fractional", geometry_area_tolerance_fractional),
    ):
        if (
            not isinstance(value, (int, float))
            or not np.isfinite(value)
            or float(value) < 0.0
        ):
            raise ValueError(f"{name} must be finite and non-negative")
    seam_coordinate_tolerance_fractional = float(
        seam_coordinate_tolerance_fractional
    )
    geometry_area_tolerance_fractional = float(
        geometry_area_tolerance_fractional
    )
    origin_u, origin_v = (float(value) for value in origin)
    cell_box = box(origin_u, origin_v, origin_u + 1.0, origin_v + 1.0)
    if geometry is None or geometry.is_empty:
        clipped = cell_box.difference(cell_box)
        outside_area = 0.0
    else:
        if not geometry.is_valid:
            raise ValueError("geometry must be valid; geometry repair is disabled")
        outside_area = float(geometry.difference(cell_box).area)
        if outside_area > geometry_area_tolerance_fractional:
            raise ValueError("geometry extends outside the declared fractional cell")
        clipped = cell_box.intersection(geometry)

    seam_intervals, interior_segments = _fractional_boundary_parts(
        clipped,
        cell_origin_fractional=origin,
        seam_coordinate_tolerance_fractional=(
            seam_coordinate_tolerance_fractional
        ),
    )
    interior_A = float(
        sum(
            np.linalg.norm(lattice @ (second - first))
            for first, second in interior_segments
        )
    )
    seam_fractional = {}
    seam_A = {}
    seam_xor_audit = {}
    for axis, low_side, high_side, tangent_vector in (
        ("u", "u_low", "u_high", lattice[:, 1]),
        ("v", "v_low", "v_high", lattice[:, 0]),
    ):
        symmetric_difference, xor_audit = (
            _tolerance_clustered_interval_xor_length(
                seam_intervals[low_side],
                seam_intervals[high_side],
                seam_coordinate_tolerance_fractional,
            )
        )
        seam_fractional[axis] = symmetric_difference
        seam_A[axis] = symmetric_difference * float(np.linalg.norm(tangent_vector))
        seam_xor_audit[axis] = xor_audit
    perimeter_A = interior_A + seam_A["u"] + seam_A["v"]
    return {
        "metric_version": "physical-flat-torus-perimeter-v1",
        "perimeter_A": float(perimeter_A),
        "unit": "A",
        "surface_lattice_uv_A": lattice.tolist(),
        "cell_origin_fractional": origin.tolist(),
        "interior_boundary_A": interior_A,
        "true_interface_on_paired_seams_A": seam_A,
        "paired_seam_symmetric_difference_fractional": seam_fractional,
        "paired_seam_interval_xor": seam_xor_audit,
        "seam_intervals_fractional": {
            key: [list(interval) for interval in value]
            for key, value in seam_intervals.items()
        },
        "artificial_cell_seams_counted": False,
        "seam_coordinate_tolerance_fractional": (
            seam_coordinate_tolerance_fractional
        ),
        "geometry_area_tolerance_fractional": geometry_area_tolerance_fractional,
        "outside_cell_area_fractional": outside_area,
        "buffer_used": False,
        "half_factor_used": False,
    }


def periodic_torus_perimeter_A(
    geometry,
    surface_lattice_uv_A,
    *,
    cell_origin_fractional=(0.0, 0.0),
    seam_coordinate_tolerance_fractional=1.0e-12,
    geometry_area_tolerance_fractional=1.0e-12,
    return_diagnostics=False,
):
    """Return the physical boundary length of a fractional geometry on a torus."""

    audit = _periodic_torus_perimeter_audit(
        geometry,
        surface_lattice_uv_A,
        cell_origin_fractional=cell_origin_fractional,
        seam_coordinate_tolerance_fractional=(
            seam_coordinate_tolerance_fractional
        ),
        geometry_area_tolerance_fractional=geometry_area_tolerance_fractional,
    )
    return audit if return_diagnostics else audit["perimeter_A"]


def periodic_union_increment_perimeter(
    current_union,
    candidate_geometry,
    surface_lattice_uv_A=None,
    *,
    cell_origin_fractional=(0.0, 0.0),
    seam_coordinate_tolerance_fractional=1.0e-12,
    current_union_perimeter_A=None,
):
    """Return union-perimeter increment and merged geometry.

    Supplying ``surface_lattice_uv_A`` selects the physical flat-torus metric
    in Angstroms and excludes artificial cell cuts.  Omitting it preserves the
    legacy two-argument fractional/Shapely behavior for existing callers.
    Passing ``current_union_perimeter_A`` avoids recomputing the base union
    perimeter when evaluating multiple candidate geometries against the same union.
    """

    _, _, _, _, unary_union = _require_shapely()
    if current_union is None:
        merged = candidate_geometry
    else:
        merged = unary_union([current_union, candidate_geometry])
    if surface_lattice_uv_A is None:
        before = 0.0 if current_union is None else float(current_union.length)
        return float(merged.length - before), merged
    if current_union_perimeter_A is not None:
        before = float(current_union_perimeter_A)
    else:
        before = periodic_torus_perimeter_A(
            current_union,
            surface_lattice_uv_A,
            cell_origin_fractional=cell_origin_fractional,
            seam_coordinate_tolerance_fractional=(
                seam_coordinate_tolerance_fractional
            ),
        )
    after = periodic_torus_perimeter_A(
        merged,
        surface_lattice_uv_A,
        cell_origin_fractional=cell_origin_fractional,
        seam_coordinate_tolerance_fractional=(
            seam_coordinate_tolerance_fractional
        ),
    )
    return float(after - before), merged


def _gauss_reduce_2d_lattice(lattice):
    """Return ``B=A@U`` with a deterministic 2D Gauss-reduced basis.

    ``U`` is an integer unimodular matrix, so enumerating integer vectors in the
    reduced basis is exactly the same lattice-image set as enumerating them in
    the input basis.  Reduction is used for a tight adaptive CVP bound; it is
    never used as a finite-neighbour heuristic.
    """

    original = _validated_surface_lattice_uv_A(lattice)
    reduced = np.asarray(original, dtype=float).copy()
    transform = np.eye(2, dtype=np.int64)
    iterations = 0
    for iterations in range(1, 257):
        first_norm2 = float(np.dot(reduced[:, 0], reduced[:, 0]))
        second_norm2 = float(np.dot(reduced[:, 1], reduced[:, 1]))
        if first_norm2 > second_norm2:
            reduced = reduced[:, [1, 0]]
            transform = transform[:, [1, 0]]
            continue
        coefficient = int(
            np.rint(float(np.dot(reduced[:, 0], reduced[:, 1])) / first_norm2)
        )
        if coefficient:
            reduced[:, 1] -= coefficient * reduced[:, 0]
            transform[:, 1] -= coefficient * transform[:, 0]
            continue
        break
    else:  # pragma: no cover - finite 2D Gauss reduction should never reach it
        raise RuntimeError("2D Gauss lattice reduction did not converge")

    determinant = int(round(float(np.linalg.det(transform))))
    if abs(determinant) != 1:
        raise RuntimeError("Gauss reduction lost its unimodular lattice transform")
    if not np.allclose(
        original @ transform, reduced, atol=1.0e-11, rtol=1.0e-12
    ):
        raise RuntimeError("Gauss-reduced lattice disagrees with A@U")
    first_norm2 = float(np.dot(reduced[:, 0], reduced[:, 0]))
    second_norm2 = float(np.dot(reduced[:, 1], reduced[:, 1]))
    dot = float(np.dot(reduced[:, 0], reduced[:, 1]))
    scale = max(first_norm2, second_norm2, 1.0)
    if first_norm2 > second_norm2 + 1.0e-12 * scale or abs(2.0 * dot) > (
        first_norm2 + 1.0e-12 * scale
    ):
        raise RuntimeError("2D lattice basis did not satisfy Gauss reduction gates")
    return reduced, transform, iterations


def _transformed_box_bounds(lower, upper, transform):
    corners = np.asarray(
        [
            [lower[0], lower[1]],
            [lower[0], upper[1]],
            [upper[0], lower[1]],
            [upper[0], upper[1]],
        ],
        dtype=float,
    )
    transformed = corners @ np.asarray(transform, dtype=float).T
    return transformed.min(axis=0), transformed.max(axis=0)


def periodic_physical_geometry_metrics(
    left_geometry,
    right_geometry,
    surface_lattice_uv_A,
    *,
    cell_origin_fractional=(0.0, 0.0),
    geometry_area_tolerance_fractional=1.0e-12,
):
    """Return complete flat-torus distance/intersection/area in physical units.

    A deterministic 2D Gauss reduction supplies an equivalent unimodular basis.
    A finite seed image gives an upper distance bound ``D``.  If another image
    can tie or improve it, each reduced fractional coordinate differs from the
    geometry-difference box by at most ``D * ||row_i(B^-1)||`` (Cauchy--Schwarz).
    The resulting adaptive integer ranges therefore enumerate every potentially
    minimizing or intersecting image, including images far outside a fixed 3x3
    stencil for a non-reduced skew lattice.
    """

    affinity, _, _, box, unary_union = _require_shapely()
    lattice = _validated_surface_lattice_uv_A(surface_lattice_uv_A)
    origin = _validated_fractional_origin(cell_origin_fractional)
    if (
        not isinstance(geometry_area_tolerance_fractional, (int, float))
        or not np.isfinite(geometry_area_tolerance_fractional)
        or float(geometry_area_tolerance_fractional) < 0.0
    ):
        raise ValueError(
            "geometry_area_tolerance_fractional must be finite and non-negative"
        )
    tolerance = float(geometry_area_tolerance_fractional)
    origin_u, origin_v = (float(value) for value in origin)
    cell_box = box(origin_u, origin_v, origin_u + 1.0, origin_v + 1.0)

    normalized = []
    for name, geometry in (
        ("left_geometry", left_geometry),
        ("right_geometry", right_geometry),
    ):
        if geometry is None or geometry.is_empty:
            normalized.append(cell_box.difference(cell_box))
            continue
        if not geometry.is_valid:
            raise ValueError(f"{name} must be valid; geometry repair is disabled")
        outside = float(geometry.difference(cell_box).area)
        if outside > tolerance:
            raise ValueError(f"{name} extends outside the declared fractional cell")
        normalized.append(cell_box.intersection(geometry))
    left, right = normalized

    transform = [
        float(lattice[0, 0]),
        float(lattice[0, 1]),
        float(lattice[1, 0]),
        float(lattice[1, 1]),
        0.0,
        0.0,
    ]
    left_physical = affinity.affine_transform(left, transform)
    reduced, unimodular, reduction_iterations = _gauss_reduce_2d_lattice(
        lattice
    )
    area_scale_A2 = abs(float(np.linalg.det(lattice)))

    if left.is_empty or right.is_empty:
        empty_intersection = left_physical.difference(left_physical)
        audit = {
            "method": "gauss_reduced_adaptive_complete_cvp_enumeration",
            "complete": True,
            "fixed_stencil_used": False,
            "empty_operand": True,
            "image_count": 0,
            "original_lattice_A": lattice.tolist(),
            "reduced_lattice_A": reduced.tolist(),
            "unimodular_transform": unimodular.tolist(),
            "reduction_iterations": int(reduction_iterations),
            "integer_bounds_reduced": None,
            "maximum_absolute_original_image_index": 0,
        }
        return {
            "metric_version": "periodic-physical-geometry-v2-complete-cvp",
            "minimum_distance_A": 0.0,
            "intersects": False,
            "intersection_area_A2": float(empty_intersection.area),
            "left_area_A2": float(left.area) * area_scale_A2,
            "right_area_A2": float(right.area) * area_scale_A2,
            "surface_lattice_uv_A": lattice.tolist(),
            "cell_origin_fractional": origin.tolist(),
            "periodic_images_evaluated": 0,
            "image_enumeration_audit": audit,
            "fractional_distance_used": False,
        }

    left_bounds = np.asarray(left.bounds, dtype=float)
    right_bounds = np.asarray(right.bounds, dtype=float)
    difference_lower = np.asarray(
        [left_bounds[0] - right_bounds[2], left_bounds[1] - right_bounds[3]]
    )
    difference_upper = np.asarray(
        [left_bounds[2] - right_bounds[0], left_bounds[3] - right_bounds[1]]
    )
    inverse_unimodular = np.linalg.inv(unimodular).round().astype(np.int64)
    reduced_difference_lower, reduced_difference_upper = _transformed_box_bounds(
        difference_lower, difference_upper, inverse_unimodular
    )

    def physical_image(original_image):
        shifted = affinity.translate(
            right,
            xoff=float(original_image[0]),
            yoff=float(original_image[1]),
        )
        return affinity.affine_transform(shifted, transform)

    representative_difference = np.asarray(
        left.representative_point().coords[0], dtype=float
    ) - np.asarray(right.representative_point().coords[0], dtype=float)
    seed_differences = [
        np.zeros(2, dtype=float),
        representative_difference,
        difference_lower,
        difference_upper,
        np.asarray([difference_lower[0], difference_upper[1]]),
        np.asarray([difference_upper[0], difference_lower[1]]),
    ]
    seed_images = {(0, 0)}
    for difference in seed_differences[1:]:
        reduced_coordinate = inverse_unimodular @ difference
        reduced_image = np.rint(reduced_coordinate).astype(np.int64)
        original_image = unimodular @ reduced_image
        seed_images.add(tuple(int(value) for value in original_image))
    seed_distances = {
        image: float(left_physical.distance(physical_image(image)))
        for image in sorted(seed_images)
    }
    upper_bound_A = min(seed_distances.values())
    if not np.isfinite(upper_bound_A) or upper_bound_A < 0.0:
        raise RuntimeError("Failed to construct a finite periodic-distance bound")

    inverse_reduced = np.linalg.inv(reduced)
    coordinate_radii = upper_bound_A * np.linalg.norm(inverse_reduced, axis=1)
    magnitude = max(
        1.0,
        float(np.max(np.abs(reduced_difference_lower))),
        float(np.max(np.abs(reduced_difference_upper))),
        float(np.max(np.abs(coordinate_radii))),
        float(np.max(np.abs(unimodular))),
    )
    roundoff = 256.0 * np.finfo(float).eps * magnitude
    integer_lower = np.ceil(
        np.nextafter(
            reduced_difference_lower - coordinate_radii - roundoff, -np.inf
        )
    ).astype(np.int64)
    integer_upper = np.floor(
        np.nextafter(
            reduced_difference_upper + coordinate_radii + roundoff, np.inf
        )
    ).astype(np.int64)
    if np.any(integer_lower > integer_upper):
        raise RuntimeError("Adaptive periodic-image bounds excluded every image")

    distances = []
    original_images = []
    intersects = False
    intersections = []
    winning_images = []
    minimum_distance = float("inf")
    for reduced_u in range(int(integer_lower[0]), int(integer_upper[0]) + 1):
        for reduced_v in range(int(integer_lower[1]), int(integer_upper[1]) + 1):
            reduced_image = np.asarray([reduced_u, reduced_v], dtype=np.int64)
            original_image_array = unimodular @ reduced_image
            original_image = tuple(int(value) for value in original_image_array)
            physical = physical_image(original_image)
            distance = float(left_physical.distance(physical))
            if not np.isfinite(distance) or distance < 0.0:
                raise RuntimeError("Periodic image produced an invalid distance")
            distances.append(distance)
            original_images.append(original_image)
            if distance < minimum_distance:
                minimum_distance = distance
                winning_images = [original_image]
            elif abs(distance - minimum_distance) <= 1.0e-12:
                winning_images.append(original_image)
            if left_physical.intersects(physical):
                intersects = True
            intersection = left_physical.intersection(physical)
            if not intersection.is_empty and float(intersection.area) > 0.0:
                intersections.append(intersection)
    intersection_geometry = (
        unary_union(intersections)
        if intersections
        else left_physical.difference(left_physical)
    )
    maximum_image_index = max(
        (max(abs(value) for value in image) for image in original_images),
        default=0,
    )
    audit = {
        "method": "gauss_reduced_adaptive_complete_cvp_enumeration",
        "complete": True,
        "completeness_bound": (
            "for any image with distance<=seed_upper_bound_D, Cauchy-Schwarz "
            "gives |q_i-m_i|<=D*||row_i(B^-1)||; all integers in those "
            "outward-rounded ranges are enumerated"
        ),
        "fixed_stencil_used": False,
        "empty_operand": False,
        "original_lattice_A": lattice.tolist(),
        "reduced_lattice_A": reduced.tolist(),
        "unimodular_transform": unimodular.tolist(),
        "unimodular_determinant": int(round(float(np.linalg.det(unimodular)))),
        "reduction_iterations": int(reduction_iterations),
        "difference_bounds_original_fractional": [
            difference_lower.tolist(),
            difference_upper.tolist(),
        ],
        "difference_bounds_reduced_fractional": [
            reduced_difference_lower.tolist(),
            reduced_difference_upper.tolist(),
        ],
        "seed_images_original": [list(image) for image in sorted(seed_images)],
        "seed_distances_A": [
            {"image": list(image), "distance_A": seed_distances[image]}
            for image in sorted(seed_distances)
        ],
        "seed_upper_bound_A": float(upper_bound_A),
        "dual_coordinate_radii": coordinate_radii.tolist(),
        "outward_roundoff_fractional": float(roundoff),
        "integer_bounds_reduced": [
            [int(integer_lower[0]), int(integer_upper[0])],
            [int(integer_lower[1]), int(integer_upper[1])],
        ],
        "image_count": len(original_images),
        "maximum_absolute_original_image_index": int(maximum_image_index),
        "winning_original_images": [list(image) for image in winning_images],
        "central_image_included": (0, 0) in original_images,
        "central_image_seeded_for_upper_bound": True,
    }
    return {
        "metric_version": "periodic-physical-geometry-v2-complete-cvp",
        "minimum_distance_A": float(minimum_distance),
        "intersects": bool(intersects),
        "intersection_area_A2": float(intersection_geometry.area),
        "left_area_A2": float(left.area) * area_scale_A2,
        "right_area_A2": float(right.area) * area_scale_A2,
        "surface_lattice_uv_A": lattice.tolist(),
        "cell_origin_fractional": origin.tolist(),
        "periodic_images_evaluated": len(original_images),
        "image_enumeration_audit": audit,
        "fractional_distance_used": False,
    }


def _torus_uncovered_component_records(
    current_union,
    area_scale_A2,
    *,
    physical_lattice,
    cell_origin_fractional,
    seam_length_tolerance_fractional,
    seam_coordinate_tolerance_fractional,
):
    """Build the one canonical positive-line seam partition of torus holes."""

    _, _, _, box, unary_union = _require_shapely()
    origin_u, origin_v = (float(value) for value in cell_origin_fractional)
    unit_square = box(origin_u, origin_v, origin_u + 1.0, origin_v + 1.0)
    if current_union is None:
        covered = unit_square.difference(unit_square)
    else:
        if not current_union.is_valid:
            raise ValueError("current_union must be valid; geometry repair is disabled")
        outside = float(current_union.difference(unit_square).area)
        if outside > 1.0e-12:
            raise ValueError("current_union extends outside the declared fractional cell")
        covered = unit_square.intersection(current_union)
    uncovered = unit_square.difference(covered)
    fragments = _positive_area_polygon_components(uncovered)
    fragments.sort(
        key=lambda geometry: (
            float(geometry.bounds[0]),
            float(geometry.bounds[1]),
            float(geometry.bounds[2]),
            float(geometry.bounds[3]),
            float(geometry.area),
        )
    )

    parents = list(range(len(fragments)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def join(left, right):
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return False
        parents[right_root] = left_root
        return True

    fragment_seams = [
        _fractional_boundary_parts(
            fragment,
            cell_origin_fractional=cell_origin_fractional,
            seam_coordinate_tolerance_fractional=(
                seam_coordinate_tolerance_fractional
            ),
        )[0]
        for fragment in fragments
    ]
    contacts = []
    accepted_contact_count = 0
    accepted_merge_count = 0
    point_contact_count = 0
    degenerate_linear_contact_count = 0
    paired_sides = (
        ("u_low", "u_high", "u", [-1, 0]),
        ("u_high", "u_low", "u", [1, 0]),
        ("v_low", "v_high", "v", [0, -1]),
        ("v_high", "v_low", "v", [0, 1]),
    )
    for left in range(len(fragments)):
        for right in range(left + 1, len(fragments)):
            for left_side, right_side, seam_axis, translation in paired_sides:
                for left_interval in fragment_seams[left][left_side]:
                    for right_interval in fragment_seams[right][right_side]:
                        overlap_start = max(left_interval[0], right_interval[0])
                        overlap_stop = min(left_interval[1], right_interval[1])
                        line_length = float(overlap_stop - overlap_start)
                        if line_length > seam_length_tolerance_fractional:
                            classification = "positive_line_contact_merged"
                            accepted_contact_count += 1
                            if join(left, right):
                                accepted_merge_count += 1
                        elif line_length > 0.0:
                            classification = (
                                "sub_tolerance_linear_degeneracy_not_merged"
                            )
                            degenerate_linear_contact_count += 1
                        elif line_length >= -seam_coordinate_tolerance_fractional:
                            classification = "point_contact_not_merged"
                            point_contact_count += 1
                            line_length = 0.0
                        else:
                            continue
                        contacts.append(
                            {
                                "left_planar_fragment": left,
                                "right_planar_fragment": right,
                                "left_seam_side": left_side,
                                "right_seam_side": right_side,
                                "seam_axis": seam_axis,
                                "translation_fractional": translation,
                                "left_interval_fractional": list(left_interval),
                                "right_interval_fractional": list(right_interval),
                                "line_length_fractional": line_length,
                                "classification": classification,
                            }
                        )

    groups: dict[int, list[int]] = {}
    for index in range(len(fragments)):
        groups.setdefault(find(index), []).append(index)
    components = []
    for member_indices in groups.values():
        component_geometry = unary_union(
            [fragments[index] for index in member_indices]
        )
        record = {
            "planar_fragment_indices": member_indices,
            "planar_fragment_count": len(member_indices),
            "fragment_geometries": tuple(
                fragments[index] for index in member_indices
            ),
            "geometry": component_geometry,
            "area_A2": float(
                sum(float(fragments[index].area) for index in member_indices)
                * area_scale_A2
            ),
        }
        if physical_lattice is not None:
            record["perimeter_A"] = periodic_torus_perimeter_A(
                component_geometry,
                physical_lattice,
                cell_origin_fractional=cell_origin_fractional,
                seam_coordinate_tolerance_fractional=(
                    seam_coordinate_tolerance_fractional
                ),
            )
        components.append(record)
    components.sort(
        key=lambda record: (
            -record["area_A2"], tuple(record["planar_fragment_indices"])
        )
    )
    for component_number, record in enumerate(components, 1):
        record["component_id"] = f"torus-hole-{component_number:04d}"

    merge_diagnostics = {
        "planar_fragment_count": len(fragments),
        "torus_component_count": len(components),
        "accepted_positive_line_contact_count": accepted_contact_count,
        "accepted_merge_count": accepted_merge_count,
        "point_contact_count": point_contact_count,
        "degenerate_linear_contact_count": degenerate_linear_contact_count,
        "point_contact_semantics": "does_not_merge",
        "positive_line_contact_rule": (
            "merge_only_when_paired_seam_interval_overlap_exceeds_"
            "seam_length_tolerance_fractional"
        ),
        "seam_length_tolerance_fractional": seam_length_tolerance_fractional,
        "seam_coordinate_tolerance_fractional": (
            seam_coordinate_tolerance_fractional
        ),
        "buffer_used": False,
        "contacts": contacts,
    }
    return {
        "covered": covered,
        "uncovered": uncovered,
        "fragments": fragments,
        "components": components,
        "seam_merge_diagnostics": merge_diagnostics,
    }


def periodic_uncovered_component_geometries(
    current_union,
    area_scale_A2,
    *,
    surface_lattice_uv_A=None,
    cell_origin_fractional=(0.0, 0.0),
    lattice_area_tolerance_A2=1.0e-8,
    seam_length_tolerance_fractional=1.0e-12,
    seam_coordinate_tolerance_fractional=1.0e-12,
):
    """Return torus-hole records carrying live Shapely fragment geometries.

    These records are an in-memory geometry seam for local search and must not
    be serialized directly.  ``periodic_hole_diagnostics`` strips the geometry
    objects while using this exact same component partition.
    """

    if (
        not isinstance(area_scale_A2, (int, float))
        or not np.isfinite(area_scale_A2)
        or float(area_scale_A2) <= 0.0
    ):
        raise ValueError("area_scale_A2 must be a finite positive number")
    area_scale_A2 = float(area_scale_A2)
    origin = _validated_fractional_origin(cell_origin_fractional)
    physical_lattice = None
    if surface_lattice_uv_A is not None:
        physical_lattice = _validated_surface_lattice_uv_A(surface_lattice_uv_A)
        lattice_area = abs(float(np.linalg.det(physical_lattice)))
        if abs(lattice_area - area_scale_A2) > float(lattice_area_tolerance_A2):
            raise ValueError("surface lattice determinant disagrees with area_scale_A2")
    for name, value in (
        ("seam_length_tolerance_fractional", seam_length_tolerance_fractional),
        ("seam_coordinate_tolerance_fractional", seam_coordinate_tolerance_fractional),
    ):
        if (
            not isinstance(value, (int, float))
            or not np.isfinite(value)
            or float(value) < 0.0
        ):
            raise ValueError(f"{name} must be finite and non-negative")
    partition = _torus_uncovered_component_records(
        current_union,
        area_scale_A2,
        physical_lattice=physical_lattice,
        cell_origin_fractional=origin,
        seam_length_tolerance_fractional=float(
            seam_length_tolerance_fractional
        ),
        seam_coordinate_tolerance_fractional=float(
            seam_coordinate_tolerance_fractional
        ),
    )
    return partition["components"]


def periodic_hole_diagnostics(
    current_union,
    area_scale_A2,
    *,
    surface_lattice_uv_A=None,
    cell_origin_fractional=(0.0, 0.0),
    area_conservation_tolerance_A2=1.0e-8,
    lattice_area_tolerance_A2=1.0e-8,
    seam_length_tolerance_fractional=1.0e-12,
    seam_coordinate_tolerance_fractional=1.0e-12,
):
    """Report uncovered connected components on the two-dimensional torus.

    The unit-cell difference can split one periodic component at U, V, or both
    cuts.  Two planar fragments are joined only when translating one by an
    integer neighboring-cell vector makes their boundaries share linework
    longer than ``seam_length_tolerance_fractional``.  A point contact never
    joins components.  Sub-threshold linear contacts are reported as numerical
    degeneracies and are likewise not joined; no buffering or topology repair
    is used.

    Legacy area fields are retained.  Additional fields expose the planar-to-
    torus merge graph, cell area/fraction, and an explicit area-conservation
    tolerance.
    """

    _, _, _, box, unary_union = _require_shapely()
    if (
        not isinstance(area_scale_A2, (int, float))
        or not np.isfinite(area_scale_A2)
    ):
        raise ValueError("area_scale_A2 must be a finite positive number")
    area_scale_A2 = float(area_scale_A2)
    if area_scale_A2 <= 0.0:
        raise ValueError("area_scale_A2 must be a finite positive number")
    if (
        not isinstance(area_conservation_tolerance_A2, (int, float))
        or not np.isfinite(area_conservation_tolerance_A2)
        or float(area_conservation_tolerance_A2) < 0.0
    ):
        raise ValueError(
            "area_conservation_tolerance_A2 must be finite and non-negative"
        )
    area_conservation_tolerance_A2 = float(area_conservation_tolerance_A2)
    if (
        not isinstance(lattice_area_tolerance_A2, (int, float))
        or not np.isfinite(lattice_area_tolerance_A2)
        or float(lattice_area_tolerance_A2) < 0.0
    ):
        raise ValueError("lattice_area_tolerance_A2 must be finite and non-negative")
    lattice_area_tolerance_A2 = float(lattice_area_tolerance_A2)
    physical_lattice = None
    if surface_lattice_uv_A is not None:
        physical_lattice = _validated_surface_lattice_uv_A(
            surface_lattice_uv_A
        )
        lattice_cell_area_A2 = abs(float(np.linalg.det(physical_lattice)))
        if abs(lattice_cell_area_A2 - area_scale_A2) > lattice_area_tolerance_A2:
            raise ValueError(
                "surface lattice determinant disagrees with area_scale_A2 beyond "
                "lattice_area_tolerance_A2"
            )
    if (
        not isinstance(seam_length_tolerance_fractional, (int, float))
        or not np.isfinite(seam_length_tolerance_fractional)
        or float(seam_length_tolerance_fractional) < 0.0
    ):
        raise ValueError(
            "seam_length_tolerance_fractional must be finite and non-negative"
        )
    seam_length_tolerance_fractional = float(
        seam_length_tolerance_fractional
    )
    if (
        not isinstance(seam_coordinate_tolerance_fractional, (int, float))
        or not np.isfinite(seam_coordinate_tolerance_fractional)
        or float(seam_coordinate_tolerance_fractional) < 0.0
    ):
        raise ValueError(
            "seam_coordinate_tolerance_fractional must be finite and non-negative"
        )
    seam_coordinate_tolerance_fractional = float(
        seam_coordinate_tolerance_fractional
    )
    cell_origin_fractional = np.asarray(cell_origin_fractional, dtype=float)
    if (
        cell_origin_fractional.shape != (2,)
        or not np.all(np.isfinite(cell_origin_fractional))
    ):
        raise ValueError("cell_origin_fractional must be a finite two-vector")
    origin_u, origin_v = (float(value) for value in cell_origin_fractional)

    partition = _torus_uncovered_component_records(
        current_union,
        area_scale_A2,
        physical_lattice=physical_lattice,
        cell_origin_fractional=cell_origin_fractional,
        seam_length_tolerance_fractional=(
            seam_length_tolerance_fractional
        ),
        seam_coordinate_tolerance_fractional=(
            seam_coordinate_tolerance_fractional
        ),
    )
    covered = partition["covered"]
    uncovered = partition["uncovered"]
    component_records = partition["components"]
    holes = [
        {
            key: value
            for key, value in record.items()
            if key not in {"fragment_geometries", "geometry"}
        }
        for record in component_records
    ]

    hole_areas_A2 = [float(record["area_A2"]) for record in holes]
    hole_perimeters_A = (
        [float(record["perimeter_A"]) for record in holes]
        if physical_lattice is not None
        else None
    )
    total_hole_area_A2 = float(sum(hole_areas_A2))
    covered_area_A2 = float(covered.area) * area_scale_A2
    difference_uncovered_area_A2 = float(uncovered.area) * area_scale_A2
    residual_A2 = area_scale_A2 - covered_area_A2 - difference_uncovered_area_A2
    component_residual_A2 = difference_uncovered_area_A2 - total_hole_area_A2
    conservation_passed = (
        abs(residual_A2) <= area_conservation_tolerance_A2
        and abs(component_residual_A2) <= area_conservation_tolerance_A2
    )

    return {
        "metric_version": "periodic-torus-holes-v2",
        "hole_count": len(holes),
        "holes": holes,
        "hole_areas_A2": hole_areas_A2,
        "total_hole_area_A2": total_hole_area_A2,
        "maximum_hole_area_A2": hole_areas_A2[0] if hole_areas_A2 else 0.0,
        "hole_perimeters_A": hole_perimeters_A,
        "total_hole_perimeter_A": (
            float(sum(hole_perimeters_A)) if hole_perimeters_A is not None else None
        ),
        "maximum_hole_perimeter_A": (
            max(hole_perimeters_A) if hole_perimeters_A else 0.0
        ) if hole_perimeters_A is not None else None,
        "perimeter_metric": (
            {
                "metric_version": "physical-flat-torus-perimeter-v1",
                "unit": "A",
                "surface_lattice_uv_A": physical_lattice.tolist(),
                "artificial_cell_seams_counted": False,
                "half_factor_used": False,
                "buffer_used": False,
                "lattice_area_tolerance_A2": lattice_area_tolerance_A2,
                "seam_coordinate_tolerance_fractional": (
                    seam_coordinate_tolerance_fractional
                ),
            }
            if physical_lattice is not None
            else None
        ),
        "cell_area_A2": area_scale_A2,
        "cell_origin_fractional": [origin_u, origin_v],
        "uncovered_area_fraction": total_hole_area_A2 / area_scale_A2,
        "area_conservation": {
            "covered_area_A2": covered_area_A2,
            "uncovered_area_A2": difference_uncovered_area_A2,
            "component_area_sum_A2": total_hole_area_A2,
            "residual_A2": residual_A2,
            "component_residual_A2": component_residual_A2,
            "tolerance_A2": area_conservation_tolerance_A2,
            "passed": conservation_passed,
        },
        "seam_merge_diagnostics": partition["seam_merge_diagnostics"],
    }


def cluster_1d_by_tolerance(values, *, tolerance: float) -> list[dict]:
    """Cluster finite scalar values by deterministic complete-span tolerance.

    Values are sorted with their original indices.  A new cluster begins when
    adding a value would make that cluster's maximum-minus-minimum span exceed
    the explicit tolerance, so no requested number of clusters is needed.
    """

    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or np.any(~np.isfinite(values)):
        raise ValueError("Cluster values must be a finite one-dimensional sequence")
    if not isinstance(tolerance, (int, float)) or not np.isfinite(tolerance):
        raise ValueError("Cluster tolerance must be a finite non-negative number")
    tolerance = float(tolerance)
    if tolerance < 0.0:
        raise ValueError("Cluster tolerance must be a finite non-negative number")
    if not len(values):
        return []
    ordered = sorted(range(len(values)), key=lambda index: (values[index], index))
    member_groups: list[list[int]] = []
    for index in ordered:
        if not member_groups:
            member_groups.append([index])
            continue
        current = member_groups[-1]
        if float(values[index] - values[current[0]]) <= tolerance + 1.0e-12:
            current.append(index)
        else:
            member_groups.append([index])
    return [
        {
            "cluster_id": f"cluster-{number:04d}",
            "member_indices": members,
            "member_count": len(members),
            "minimum": float(np.min(values[members])),
            "maximum": float(np.max(values[members])),
            "mean": float(np.mean(values[members])),
        }
        for number, members in enumerate(member_groups, 1)
    ]


def rigid_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return row-vector rotation, translation, and RMSD for source -> target."""

    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    u, _, vt = np.linalg.svd(source_zero.T @ target_zero)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    translation = target_center - source_center @ rotation
    fitted = source @ rotation + translation
    rmsd = float(np.sqrt(np.mean(np.sum((fitted - target) ** 2, axis=1))))
    return rotation, translation, rmsd


def _validated_molecular_positions(positions) -> np.ndarray:
    coordinates = np.asarray(positions, dtype=float)
    if (
        coordinates.ndim != 2
        or coordinates.shape[1] != 3
        or not len(coordinates)
        or np.any(~np.isfinite(coordinates))
    ):
        raise ValueError("positions must be a finite, nonempty (N, 3) array")
    return coordinates


def _validated_atom_indices(indices, atom_count: int, *, name: str) -> list[int]:
    result = []
    for index in indices:
        if (
            isinstance(index, (bool, np.bool_))
            or not isinstance(index, (int, np.integer))
            or not 0 <= index < atom_count
        ):
            raise ValueError(f"{name} must contain valid integer atom indices")
        result.append(int(index))
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must contain distinct atom indices")
    return result


def standardize_phosphonate_head_plane(
    positions, p_index: int, o_indices
) -> tuple[np.ndarray, dict]:
    """Place the PO3 oxygen plane at z=0 with one proper rigid transform.

    将整个分子作同一刚体变换，令三个 O 的质心为原点、P 位于正 z。
    Input coordinates must describe an unwrapped molecule in Cartesian A.
    The ordered, noncollinear O triplet defines the plane. Its centroid is the
    origin, +x points from O[0] to O[1], and +z points toward P; +y completes a
    right-handed frame. P in the O plane cannot define the positive side and
    is rejected. No atom is independently moved, flattened, or reflected.

    Returns ``(coordinates, metadata)``. In the existing ``rigid_transform``
    row-vector convention, the complete transformation is
    ``coordinates = positions @ rotation_matrix + translation_A``. Plane
    residuals are floating-point roundoff; clipping individual O coordinates
    to exact zero would cease to be a rigid transformation.
    """

    coordinates = _validated_molecular_positions(positions)
    p_index = _validated_atom_indices(
        [p_index], len(coordinates), name="p_index"
    )[0]
    o_indices = _validated_atom_indices(
        o_indices, len(coordinates), name="o_indices"
    )
    if len(o_indices) != 3 or p_index in o_indices:
        raise ValueError("o_indices must identify exactly three O atoms distinct from P")

    oxygens = coordinates[o_indices]
    origin = np.mean(oxygens, axis=0)
    edge_x = oxygens[1] - oxygens[0]
    edge_other = oxygens[2] - oxygens[0]
    edge_scale = max(np.linalg.norm(edge_x), np.linalg.norm(edge_other))
    normal = np.cross(edge_x, edge_other)
    normal_length = float(np.linalg.norm(normal))
    roundoff = 64.0 * np.finfo(float).eps
    if edge_scale == 0.0 or normal_length <= roundoff * edge_scale**2:
        raise ValueError("The three PO3 O atoms must be noncollinear")
    normal /= normal_length
    p_vector = coordinates[p_index] - origin
    p_height = float(np.dot(p_vector, normal))
    height_roundoff_A = roundoff * max(edge_scale, np.linalg.norm(p_vector))
    if abs(p_height) <= height_roundoff_A:
        raise ValueError("P must lie outside the PO3 O plane to define positive z")
    if p_height < 0.0:
        normal *= -1.0

    x_axis = edge_x / np.linalg.norm(edge_x)
    y_axis = np.cross(normal, x_axis)
    rotation = np.column_stack((x_axis, y_axis, normal))
    translation = -origin @ rotation
    # Center before multiplying to reduce cancellation for translated inputs.
    standardized = (coordinates - origin) @ rotation
    metadata = {
        "method": "proper_rigid_po3_plane_standardization",
        "coordinate_unit": "A",
        "row_vector_transform": "positions @ rotation_matrix + translation_A",
        "rotation_matrix": rotation.tolist(),
        "translation_A": translation.tolist(),
        "oxygen_centroid_input_A": origin.tolist(),
        "p_index": p_index,
        "o_indices": o_indices,
        "p_height_A": float(standardized[p_index, 2]),
        "maximum_oxygen_plane_residual_A": float(
            np.max(np.abs(standardized[o_indices, 2]))
        ),
        "rotation_determinant": float(np.linalg.det(rotation)),
        "reflection_used": False,
    }
    return standardized, metadata


def canonicalize_pc_axis(
    positions, p_index: int, c_index: int
) -> tuple[np.ndarray, dict]:
    """Return a proper rigid P-origin, P-to-C +z feature frame.

    对全分子作 det=+1 的刚体变换，仅用于 P-C 相位不变的形状特征。
    Cartesian coordinates are in A and must describe an unwrapped molecule.
    P is translated to the origin and the directed P-to-C bond is aligned
    with +z. A deterministic Cartesian reference fixes the otherwise
    arbitrary x/y gauge; ``pc_axial_rmsd_matrix`` subsequently minimizes over
    that remaining SO(2) phase. No atom labels are permuted, no reflection
    is allowed, and the P-C bond length is retained.

    This frame is for comparison features only. Replacing a saved physical
    geometry with it would rotate its fixed PO3 head away from the surface
    frame. The caller chooses the same organic-heavy-atom indices from every
    returned structure. Excluding PO3 discards the body's azimuth and its
    inclination relative to the head, while preserving internal shape and
    each atom's axial distance from P. No independent Kabsch fit is performed.
    """

    coordinates = _validated_molecular_positions(positions)
    p_index, c_index = _validated_atom_indices(
        [p_index, c_index], len(coordinates), name="p_index/c_index"
    )
    origin = coordinates[p_index]
    axis = coordinates[c_index] - origin
    axis_length = float(np.linalg.norm(axis))
    if not np.isfinite(axis_length) or axis_length == 0.0:
        raise ValueError("P and C must define a finite, nonzero directed axis")
    z_axis = axis / axis_length
    # The least-parallel Cartesian basis vector avoids a nearly singular
    # cross product even when the P-C axis is exactly +/-z or nearly parallel.
    cartesian_reference = np.eye(3)[int(np.argmin(np.abs(z_axis)))]
    x_axis = cartesian_reference - np.dot(cartesian_reference, z_axis) * z_axis
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    rotation = np.column_stack((x_axis, y_axis, z_axis))
    aligned = (coordinates - origin) @ rotation
    if np.any(~np.isfinite(aligned)):
        raise ValueError("The P-C feature transform produced nonfinite coordinates")
    metadata = {
        "method": "proper_rigid_pc_axis_feature_frame",
        "coordinate_unit": "A",
        "p_index": p_index,
        "c_index": c_index,
        "p_c_bond_length_A": axis_length,
        "row_vector_transform": "positions @ rotation_matrix + translation_A",
        "rotation_matrix": rotation.tolist(),
        "translation_A": (-origin @ rotation).tolist(),
        "rotation_determinant": float(np.linalg.det(rotation)),
        "reflection_used": False,
        "physical_output_coordinates_replaced": False,
    }
    return aligned, metadata


def _validated_pc_feature_stack(features, *, name: str) -> np.ndarray:
    coordinates = np.asarray(features, dtype=float)
    if coordinates.ndim == 2:
        coordinates = coordinates[None, :, :]
    if (
        coordinates.ndim != 3
        or coordinates.shape[2] != 3
        or coordinates.shape[0] == 0
        or coordinates.shape[1] == 0
        or np.any(~np.isfinite(coordinates))
    ):
        raise ValueError(f"{name} must be a finite, nonempty (M, N, 3) or (N, 3) array")
    return coordinates


def pc_axial_rmsd_matrix(
    reference_features, candidate_features=None, *, return_angles: bool = False
):
    """Compute fixed-mapping 3D RMSD after one common optimal P-C phase.

    输入为 ``canonicalize_pc_axis`` 后的同一组有机重原子，不作自由 Kabsch。
    Each feature set is (N, 3) or a batch (M, N, 3), in A, in its own P-origin
    P-to-C +z frame. Corresponding rows must represent the same original atom
    labels, in the same order. The caller must include the C bonded to P and
    exclude P/PO3 and organic H. This helper does not infer chemistry.

    One proper rotation around +z is minimized for the *whole* candidate;
    no independent per-atom phase, translation, tilt, reflection, permutation,
    or molecular Kabsch fit is allowed. Z offsets and P-C bond-length changes
    remain in the true per-atom 3D RMSD, sqrt(sum_i |A_i-Rz(theta)B_i|^2/N).
    This measures body shape modulo P-C phase, not force-field-basin identity.

    Returns a (M, K) matrix. Omitting candidate_features gives a symmetric
    self-comparison matrix. With return_angles=True, also returns a (M, K)
    matrix of right-hand candidate-to-reference angles in [0, 360) degrees.
    For an undetermined phase (zero x/y correlation), the angle is zero.
    Bulk correlations use matrix products, suitable for complete-link input.
    Near-zero differences are recomputed directly to avoid catastrophic
    cancellation; negative squared distances beyond roundoff are rejected.
    """

    reference = _validated_pc_feature_stack(reference_features, name="reference_features")
    candidate = (
        reference
        if candidate_features is None
        else _validated_pc_feature_stack(candidate_features, name="candidate_features")
    )
    if reference.shape[1] != candidate.shape[1]:
        raise ValueError("Reference and candidate features must use the same atom mapping")
    if not isinstance(return_angles, (bool, np.bool_)):
        raise ValueError("return_angles must be boolean")
    rx, ry, rz = (reference[:, :, dimension] for dimension in range(3))
    cx, cy, cz = (candidate[:, :, dimension] for dimension in range(3))
    with np.errstate(over="ignore", invalid="ignore"):
        cosine = rx @ cx.T + ry @ cy.T
        # A dot Rz(theta)B = cosine*cos(theta) + sine*sin(theta) + Az.Bz.
        sine = ry @ cx.T - rx @ cy.T
        z_correlation = rz @ cz.T
        reference_norm = np.sum(reference * reference, axis=(1, 2))
        candidate_norm = np.sum(candidate * candidate, axis=(1, 2))
        correlation_amplitude = np.hypot(cosine, sine)
        squared_distance = reference_norm[:, None] + candidate_norm[None, :]
        squared_distance -= 2.0 * (z_correlation + correlation_amplitude)
    if any(
        np.any(~np.isfinite(array))
        for array in (cosine, sine, squared_distance, reference_norm, candidate_norm)
    ):
        raise ValueError("Feature magnitudes overflow finite RMSD arithmetic")

    # The uncertainty budget scales with norms, never a physical RMSD floor.
    numerical_budget = 128.0 * np.finfo(float).eps * (
        reference_norm[:, None] + candidate_norm[None, :]
    )
    if np.any(squared_distance < -numerical_budget):
        raise ValueError("Negative squared RMSD exceeds the floating-point roundoff budget")
    near_rows, near_columns = np.nonzero(squared_distance <= numerical_budget)
    # Direct residuals also recover small positive RMSDs that a norm-based
    # expression could lose. Bounded chunks avoid an M*K*N intermediate.
    for start in range(0, len(near_rows), 4096):
        rows = near_rows[start:start + 4096]
        columns = near_columns[start:start + 4096]
        amplitude = correlation_amplitude[rows, columns]
        cos_phase = np.ones_like(amplitude)
        sin_phase = np.zeros_like(amplitude)
        determined = amplitude > 0.0
        cos_phase[determined] = cosine[rows[determined], columns[determined]] / amplitude[determined]
        sin_phase[determined] = sine[rows[determined], columns[determined]] / amplitude[determined]
        dx = rx[rows] - (
            cx[columns] * cos_phase[:, None] - cy[columns] * sin_phase[:, None]
        )
        dy = ry[rows] - (
            cx[columns] * sin_phase[:, None] + cy[columns] * cos_phase[:, None]
        )
        dz = rz[rows] - cz[columns]
        squared_distance[rows, columns] = np.sum(dx * dx + dy * dy + dz * dz, axis=1)
    rmsd = np.sqrt(squared_distance / reference.shape[1])
    if not return_angles:
        return rmsd
    angles = np.degrees(np.arctan2(sine, cosine)) % 360.0
    angles[correlation_amplitude == 0.0] = 0.0
    return rmsd, angles


def _closed_trigonometric_angle_intervals(constant, cosine, sine, floor):
    """Solve constant + cosine*cos(theta) + sine*sin(theta) >= floor."""

    period = 2.0 * math.pi
    amplitude = math.hypot(cosine, sine)
    delta = floor - constant
    if amplitude == 0.0:
        return [(0.0, period)] if delta <= 0.0 else []
    if delta <= -amplitude:
        return [(0.0, period)]
    if delta > amplitude:
        return []
    half_width = math.acos(max(-1.0, min(1.0, delta / amplitude)))
    center = math.atan2(sine, cosine) % period
    start, end = center - half_width, center + half_width
    if start < 0.0:
        return [(0.0, end), (start + period, period)]
    if end >= period:
        return [(0.0, end - period), (start, period)]
    return [(start, end)]


def _intersect_closed_angle_intervals(left, right):
    intersection = []
    for start_left, end_left in left:
        for start_right, end_right in right:
            start = max(start_left, start_right)
            end = min(end_left, end_right)
            if start <= end:
                intersection.append((start, end))
    merged = []
    for start, end in sorted(intersection):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def phosphonate_axial_height_intervals(
    positions,
    p_index: int,
    c_index: int,
    organic_indices,
    z_floor: float = 0.0,
    tolerance: float = 1.0e-12,
    *,
    collision_pairs_0based=None,
    collision_min_distances_A=None,
) -> dict:
    """Find every continuous P-C axial angle satisfying the organic z floor.

    连续求解整圈共同可行角，P/PO3 不动，仅 C 侧完整片段作刚体旋转。
    The input must be an unwrapped molecule in the standardized Cartesian
    frame returned by ``standardize_phosphonate_head_plane``: the O plane is
    z=0 and P lies at positive z. This function does not infer a surface frame
    or atom chemistry. The caller must supply *all* C-side organic indices,
    including attached H and c_index, and exclude P, the PO3 O atoms, and any
    O-bound headgroup H. Only this fragment may rotate; all unlisted atoms
    remain fixed. This is an axial torsion, not a rotation of the whole SAM.

    For a right-hand angle theta about the unit axis u from P to C, each atom
    has height ``d + a*cos(theta) + b*sin(theta)`` by Rodrigues' formula. The
    returned closed intervals solve the intersection of all inequalities
    ``height >= z_floor - tolerance`` analytically. ``tolerance`` is in A and
    defaults to numerical roundoff, not a physical clearance allowance.
    Neither sampled angles nor each atom's separate maximum height proves
    that a common feasible angle exists.

    ``feasible_intervals_deg`` is a sorted list of [start, end] in [0, 360];
    an arc crossing zero is split, and 0 and 360 represent the same angle.
    A point interval remains feasible. The deterministic representative is
    the midpoint of the widest *circular* connected interval, with the
    smallest normalized midpoint breaking exact ties (0 for a full circle).
    It is None for an empty intersection.

    Optionally provide ``collision_pairs_0based`` and the corresponding
    ``collision_min_distances_A`` together. Every pair must contain exactly
    one moving organic atom and one fixed atom. Its squared distance is also
    a constant plus cosine and sine, so the joint height/collision intervals
    are solved at the *same* angle. Distances must be at least the threshold
    minus the same numerical A tolerance. ``height_only_feasible`` records
    feasibility before these constraints. Organic-organic and fixed-fixed
    distances are invariant; callers must screen those pairs separately.
    """

    coordinates = _validated_molecular_positions(positions)
    p_index, c_index = _validated_atom_indices(
        [p_index, c_index], len(coordinates), name="p_index/c_index"
    )
    organic_indices = _validated_atom_indices(
        organic_indices, len(coordinates), name="organic_indices"
    )
    if not organic_indices or c_index not in organic_indices or p_index in organic_indices:
        raise ValueError("organic_indices must include C, exclude P, and be nonempty")
    z_floor, tolerance = float(z_floor), float(tolerance)
    if not math.isfinite(z_floor) or not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("z_floor and nonnegative tolerance must be finite")
    if coordinates[p_index, 2] <= 0.0:
        raise ValueError("positions must be in the standardized PO3 frame with P at positive z")
    axis = coordinates[c_index] - coordinates[p_index]
    axis_length = float(np.linalg.norm(axis))
    if axis_length == 0.0:
        raise ValueError("P and C must define a nonzero rotation axis")
    axis /= axis_length
    relative = coordinates[organic_indices] - coordinates[p_index]
    axial_projection = relative @ axis
    constant = coordinates[p_index, 2] + axis[2] * axial_projection
    cosine = relative[:, 2] - axis[2] * axial_projection
    sine = np.cross(axis, relative)[:, 2]
    period = 2.0 * math.pi
    intervals = [(0.0, period)]
    atom_records = []
    individually_impossible = []
    for atom_index, d, a, b in zip(organic_indices, constant, cosine, sine):
        allowed = _closed_trigonometric_angle_intervals(d, a, b, z_floor - tolerance)
        amplitude = math.hypot(a, b)
        atom_records.append({
            "atom_index": atom_index,
            "constant_height_A": float(d),
            "cosine_height_A": float(a),
            "sine_height_A": float(b),
            "minimum_height_A": float(d - amplitude),
            "maximum_height_A": float(d + amplitude),
            "feasible_intervals_deg": [
                [math.degrees(start), math.degrees(end)] for start, end in allowed
            ],
        })
        if not allowed:
            individually_impossible.append(atom_index)
        intervals = _intersect_closed_angle_intervals(intervals, allowed)

    height_intervals = list(intervals)
    collision_records = []
    if (collision_pairs_0based is None) != (collision_min_distances_A is None):
        raise ValueError("collision pairs and minimum distances must be provided together")
    if collision_pairs_0based is not None:
        collision_pairs = list(collision_pairs_0based)
        minimum_distances = np.asarray(collision_min_distances_A, dtype=float)
        if (
            minimum_distances.shape != (len(collision_pairs),)
            or np.any(~np.isfinite(minimum_distances))
            or np.any(minimum_distances < 0.0)
        ):
            raise ValueError("collision minimum distances must be finite nonnegative A values")
        organic_set = set(organic_indices)
        seen_pairs = set()
        for pair, minimum_distance in zip(collision_pairs, minimum_distances):
            pair = _validated_atom_indices(pair, len(coordinates), name="collision pair")
            if len(pair) != 2 or len(organic_set.intersection(pair)) != 1:
                raise ValueError("each collision pair must have one moving and one fixed atom")
            unordered_pair = tuple(sorted(pair))
            if unordered_pair in seen_pairs:
                raise ValueError("collision pairs must be distinct")
            seen_pairs.add(unordered_pair)
            moving_index = next(index for index in pair if index in organic_set)
            fixed_index = next(index for index in pair if index not in organic_set)
            moving = coordinates[moving_index] - coordinates[p_index]
            fixed = coordinates[fixed_index] - coordinates[p_index]
            parallel = axis * np.dot(axis, moving)
            perpendicular = moving - parallel
            cross = np.cross(axis, moving)
            offset = float(np.dot(moving, moving) + np.dot(fixed, fixed)
                           - 2.0 * np.dot(fixed, parallel))
            cos_squared_distance = float(-2.0 * np.dot(fixed, perpendicular))
            sin_squared_distance = float(-2.0 * np.dot(fixed, cross))
            floor_squared_distance = max(0.0, float(minimum_distance) - tolerance)**2
            allowed = _closed_trigonometric_angle_intervals(
                offset, cos_squared_distance, sin_squared_distance, floor_squared_distance
            )
            collision_records.append({
                "pair_0based": pair,
                "moving_atom_index": moving_index,
                "fixed_atom_index": fixed_index,
                "minimum_distance_A": float(minimum_distance),
                "constant_squared_distance_A2": offset,
                "cosine_squared_distance_A2": cos_squared_distance,
                "sine_squared_distance_A2": sin_squared_distance,
                "feasible_intervals_deg": [
                    [math.degrees(start), math.degrees(end)] for start, end in allowed
                ],
            })
            intervals = _intersect_closed_angle_intervals(intervals, allowed)

    representative_rad = None
    if intervals == [(0.0, period)]:
        representative_rad = 0.0
    elif intervals:
        circular_intervals = list(intervals)
        if len(intervals) > 1 and intervals[0][0] == 0.0 and intervals[-1][1] == period:
            circular_intervals = intervals[1:-1] + [
                (intervals[-1][0], intervals[0][1] + period)
            ]
        _, representative_rad = min(
            (
                (-(end - start), ((start + end) / 2.0) % period)
                for start, end in circular_intervals
            )
        )
    representative_minimum_height = None
    if representative_rad is not None:
        representative_minimum_height = float(np.min(
            constant + cosine * math.cos(representative_rad)
            + sine * math.sin(representative_rad)
        ))
    return {
        "method": "continuous_analytic_axial_height_collision_intersection",
        "angular_grid_used": False,
        "feasible": bool(intervals),
        "height_only_feasible": bool(height_intervals),
        "height_only_intervals_deg": [
            [math.degrees(start), math.degrees(end)] for start, end in height_intervals
        ],
        "feasible_intervals_deg": [
            [math.degrees(start), math.degrees(end)] for start, end in intervals
        ],
        "feasible_angle_measure_deg": math.degrees(
            sum(end - start for start, end in intervals)
        ),
        "representative_angle_deg": (
            None if representative_rad is None else math.degrees(representative_rad)
        ),
        "representative_minimum_height_A": representative_minimum_height,
        "axis_origin_A": coordinates[p_index].tolist(),
        "axis_unit": axis.tolist(),
        "angle_convention": "right_hand_about_P_to_C",
        "organic_indices": organic_indices,
        "z_floor_A": z_floor,
        "tolerance_A": tolerance,
        "atom_height_coefficients": atom_records,
        "individually_impossible_atom_indices": individually_impossible,
        "collision_constraints_count": len(collision_records),
        "collision_squared_distance_coefficients": collision_records,
    }


def calculate_substrate_metal_coordinations(
    substrate_positions: np.ndarray,
    substrate_symbols: np.ndarray,
    cell: np.ndarray,
    *,
    periodic_axes: tuple[int, int] = (0, 1),
    normal_axis: int = 2,
    cutoff_A: float = 2.7,
    target_elements: set[str] | tuple[str, ...] | list[str] = ("In", "Sn"),
    ligand_elements: set[str] | tuple[str, ...] | list[str] = ("O",),
    full_to_working_atom_map: dict[int, dict] | None = None,
) -> dict:
    """Calculate pre-adsorption substrate-O coordination numbers for substrate metals.

    Distances are evaluated under 2D periodic boundary conditions along periodic_axes
    (Surface Frame in-plane fractional directions) with strictly no wrapping along normal_axis,
    using ASE's find_mic to guarantee the exact minimum-image distance in general oblique cells.
    """
    substrate_positions = np.asarray(substrate_positions, dtype=float)
    substrate_symbols = np.asarray(substrate_symbols, dtype=str)
    cell = np.asarray(cell, dtype=float)
    if substrate_positions.ndim != 2 or substrate_positions.shape[1] != 3:
        raise ValueError("substrate_positions must have shape (N, 3)")
    if len(substrate_symbols) != len(substrate_positions):
        raise ValueError("substrate_symbols and substrate_positions length mismatch")
    if cell.shape != (3, 3):
        raise ValueError("cell must have shape (3, 3)")

    cutoff = float(cutoff_A)
    if not math.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("cutoff_A must be positive and finite")

    target_set = set(target_elements)
    ligand_set = set(ligand_elements)
    axes = tuple(int(a) for a in periodic_axes)
    if len(axes) != 2:
        raise ValueError("periodic_axes must contain exactly 2 axes")
    norm_ax = int(normal_axis)
    if norm_ax in axes or norm_ax not in (0, 1, 2):
        raise ValueError("normal_axis must be the third axis distinct from periodic_axes")

    pbc = np.zeros(3, dtype=bool)
    pbc[list(axes)] = True

    metal_indices = [i for i, s in enumerate(substrate_symbols) if s in target_set]
    ligand_indices = [i for i, s in enumerate(substrate_symbols) if s in ligand_set]

    inverse_full_map = {}
    if full_to_working_atom_map is not None:
        for full_id, record in full_to_working_atom_map.items():
            w_idx = int(record["working_atom_index_0based"])
            inverse_full_map[w_idx] = int(full_id)

    metal_records_by_working_index = {}
    metal_records_by_full_id = {}
    inventory_by_elem = defaultdict(lambda: {"total": 0, "undercoordinated": 0, "six_coordinated": 0, "coordination_distribution": Counter()})
    overall_coord_dist = Counter()

    for m_idx in metal_indices:
        m_pos = substrate_positions[m_idx]
        m_elem = str(substrate_symbols[m_idx])
        full_id = inverse_full_map.get(m_idx, m_idx + 1)

        if len(ligand_indices) > 0:
            diff = substrate_positions[ligand_indices] - m_pos
            _, distances = find_mic(diff, cell, pbc=pbc)
            neighbor_distances = [
                float(d) for d in distances
                if distance_within_closed_window(float(d), 0.0, cutoff)
            ]
            neighbor_distances.sort()
        else:
            neighbor_distances = []
        cn = len(neighbor_distances)

        is_six = (cn >= 6)
        is_under = (cn <= 5)

        rec = {
            "working_atom_index_0based": int(m_idx),
            "full_metal_atom_id_1based": int(full_id),
            "element": m_elem,
            "coordination_number": int(cn),
            "is_six_coordinated": bool(is_six),
            "is_undercoordinated": bool(is_under),
            "neighbor_oxygen_distances_A": neighbor_distances,
        }
        metal_records_by_working_index[m_idx] = rec
        metal_records_by_full_id[full_id] = rec

        inventory_by_elem[m_elem]["total"] += 1
        if is_six:
            inventory_by_elem[m_elem]["six_coordinated"] += 1
        if is_under:
            inventory_by_elem[m_elem]["undercoordinated"] += 1
        inventory_by_elem[m_elem]["coordination_distribution"][str(cn)] += 1
        overall_coord_dist[str(cn)] += 1

    total_metals = len(metal_indices)
    total_under = sum(inv["undercoordinated"] for inv in inventory_by_elem.values())
    total_six = sum(inv["six_coordinated"] for inv in inventory_by_elem.values())

    inventory = {
        "total_metal_count": int(total_metals),
        "undercoordinated_count": int(total_under),
        "six_coordinated_count": int(total_six),
        "by_element": {
            elem: {
                "total": int(data["total"]),
                "undercoordinated": int(data["undercoordinated"]),
                "six_coordinated": int(data["six_coordinated"]),
                "coordination_distribution": dict(sorted(data["coordination_distribution"].items())),
            }
            for elem, data in sorted(inventory_by_elem.items())
        },
        "overall_coordination_distribution": dict(sorted(overall_coord_dist.items())),
    }

    return {
        "inventory": inventory,
        "metal_records_by_working_index": metal_records_by_working_index,
        "metal_records_by_full_id": metal_records_by_full_id,
        "cutoff_A": cutoff,
        "periodic_axes": list(axes),
        "normal_axis": norm_ax,
    }
