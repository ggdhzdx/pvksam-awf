#!/usr/bin/env python3
"""Analyze the final time window of a substrate-first SAM LAMMPS trajectory.

The script is system-name independent.  It uses explicit substrate/molecule
layout parameters, discovers the conjugated C/N/S core topologically, unwraps
each molecule under MIC before fitting, and reports both thresholded and smooth
sulfur upper-surface accessibility metrics plus a soft average SAM thickness.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from pathlib import Path

import numpy as np
from ase.data import chemical_symbols

from sam_lammps import DEFAULT_SPECIES, covalent_bond_cutoff, parse_species


VDW_RADII_A = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "P": 1.80,
    "S": 1.80,
    "In": 1.93,
    "Sn": 2.17,
}


def iter_lammps_dump(path: Path, species: tuple[str, ...]):
    """Yield ``(step, cell, positions, symbols)`` from an atomic text dump."""

    with path.open() as handle:
        while True:
            marker = handle.readline()
            if not marker:
                return
            if marker.strip() != "ITEM: TIMESTEP":
                raise ValueError(f"Unexpected dump marker: {marker.strip()}")
            step = int(handle.readline())
            if handle.readline().strip() != "ITEM: NUMBER OF ATOMS":
                raise ValueError("Missing NUMBER OF ATOMS header")
            atom_count = int(handle.readline())
            bounds_header = handle.readline().split()
            if bounds_header[:3] != ["ITEM:", "BOX", "BOUNDS"]:
                raise ValueError("Missing BOX BOUNDS header")
            bounds = np.asarray(
                [[float(value) for value in handle.readline().split()[:2]] for _ in range(3)]
            )
            atom_header = handle.readline().split()
            if atom_header[:2] != ["ITEM:", "ATOMS"]:
                raise ValueError("Missing ATOMS header")
            columns = atom_header[2:]
            required = {name: columns.index(name) for name in ("id", "type", "x", "y", "z")}
            positions = np.empty((atom_count, 3), dtype=float)
            types = np.empty(atom_count, dtype=np.int16)
            for _ in range(atom_count):
                fields = handle.readline().split()
                atom_index = int(fields[required["id"]]) - 1
                atom_type = int(fields[required["type"]])
                if not 1 <= atom_type <= len(species):
                    raise ValueError(f"Atom type {atom_type} has no species mapping")
                types[atom_index] = atom_type
                positions[atom_index] = [
                    float(fields[required[axis]]) for axis in ("x", "y", "z")
                ]
            cell = np.diag(bounds[:, 1] - bounds[:, 0])
            positions -= bounds[:, 0]
            symbols = np.asarray([species[atom_type - 1] for atom_type in types])
            yield step, cell, positions, symbols


def mic_vectors(vectors: np.ndarray, cell: np.ndarray) -> np.ndarray:
    fractional = np.linalg.solve(cell.T, vectors.T).T
    fractional -= np.round(fractional)
    return fractional @ cell


def unwrap_block(positions: np.ndarray, cell: np.ndarray, reference: int = 0) -> np.ndarray:
    reference_position = positions[reference]
    return reference_position + mic_vectors(positions - reference_position, cell)


def fit_plane(coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centroid = np.mean(coordinates, axis=0)
    _, singular_values, vh = np.linalg.svd(coordinates - centroid, full_matrices=False)
    if len(singular_values) < 3 or singular_values[1] <= 1.0e-10:
        raise ValueError("Plane-defining atoms are collinear")
    normal = vh[-1] / np.linalg.norm(vh[-1])
    if normal[2] < 0:
        normal = -normal
    return centroid, normal


def molecular_adjacency(
    block_positions: np.ndarray,
    block_symbols: np.ndarray,
    cell: np.ndarray,
) -> dict[int, set[int]]:
    adjacency = {index: set() for index in range(len(block_symbols))}
    for atom_i in range(len(block_symbols)):
        if atom_i + 1 >= len(block_symbols):
            continue
        vectors = mic_vectors(
            block_positions[atom_i + 1 :] - block_positions[atom_i], cell
        )
        distances = np.linalg.norm(vectors, axis=1)
        for offset, distance in enumerate(distances, atom_i + 1):
            cutoff = covalent_bond_cutoff(block_symbols[atom_i], block_symbols[offset])
            if distance < cutoff:
                adjacency[atom_i].add(offset)
                adjacency[offset].add(atom_i)
    return adjacency


def shortest_path(adjacency: dict[int, set[int]], start: int, stop: int) -> list[int]:
    queue = deque([[start]])
    visited = {start}
    while queue:
        path = queue.popleft()
        if path[-1] == stop:
            return path
        for neighbor in sorted(adjacency[path[-1]] - visited):
            visited.add(neighbor)
            queue.append(path + [neighbor])
    raise ValueError(f"No molecular path between local atoms {start} and {stop}")


def discover_molecule_definition(
    block_positions: np.ndarray,
    block_symbols: np.ndarray,
    cell: np.ndarray,
    anchor_element: str,
    linker_end_element: str,
    marker_element: str,
    conjugated_elements: tuple[str, ...],
) -> dict:
    anchors = np.flatnonzero(block_symbols == anchor_element)
    linker_ends = np.flatnonzero(block_symbols == linker_end_element)
    markers = np.flatnonzero(block_symbols == marker_element)
    if len(anchors) != 1 or len(linker_ends) != 1 or len(markers) != 1:
        raise ValueError(
            f"Expected one {anchor_element}, {linker_end_element}, and {marker_element}; "
            f"found {len(anchors)}, {len(linker_ends)}, {len(markers)}"
        )
    adjacency = molecular_adjacency(block_positions, block_symbols, cell)
    path = shortest_path(adjacency, int(anchors[0]), int(linker_ends[0]))
    linker_carbons = {
        atom_index
        for atom_index in path[1:-1]
        if block_symbols[atom_index] == "C"
    }
    core = [
        atom_index
        for atom_index, symbol in enumerate(block_symbols)
        if symbol in conjugated_elements and atom_index not in linker_carbons
    ]
    if len(core) < 3:
        raise ValueError(f"Only {len(core)} conjugated heavy atoms were discovered")
    return {
        "anchor": int(anchors[0]),
        "linker_end": int(linker_ends[0]),
        "marker": int(markers[0]),
        "linker_carbons": sorted(linker_carbons),
        "core": core,
    }


def top_surface_indices(
    positions: np.ndarray,
    symbols: np.ndarray,
    substrate_atoms: int,
    surface_element: str,
    window: float,
) -> np.ndarray:
    substrate = np.arange(substrate_atoms)
    candidates = substrate[symbols[substrate] == surface_element]
    if not len(candidates):
        raise ValueError(f"Substrate contains no {surface_element}")
    maximum = float(np.max(positions[candidates, 2]))
    selected = candidates[positions[candidates, 2] >= maximum - window]
    if len(selected) < 3:
        raise ValueError("Fewer than three top-surface atoms were selected")
    return selected


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "q05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "q95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def stable_sigmoid(values: np.ndarray | float) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def continuous_exposure_metrics(
    vectors: np.ndarray,
    surface_normal: np.ndarray,
    lateral_scale: float,
    height_smoothing: float,
) -> tuple[float, float, float]:
    """Return smooth local height rank, shielding exposure, and blocker load."""

    if lateral_scale <= 0.0 or height_smoothing <= 0.0:
        raise ValueError("Continuous exposure scales must be positive")
    if not len(vectors):
        return 1.0, 1.0, 0.0
    vertical = vectors @ surface_normal
    lateral_vectors = vectors - np.outer(vertical, surface_normal)
    lateral = np.linalg.norm(lateral_vectors, axis=1)
    weights = np.exp(-0.5 * (lateral / lateral_scale) ** 2)
    weight_sum = float(np.sum(weights))
    height_rank = float(
        np.sum(weights * stable_sigmoid(-vertical / height_smoothing)) / weight_sum
    )
    blocker_load = float(
        np.sum(weights * stable_sigmoid(vertical / height_smoothing))
    )
    shielding_exposure = float(np.exp(-blocker_load))
    return height_rank, shielding_exposure, blocker_load


def upper_hemisphere_directions(normal: np.ndarray, samples: int) -> np.ndarray:
    """Generate deterministic equal-area Fibonacci directions above a plane."""

    if samples < 1:
        raise ValueError("Probe direction sample count must be positive")
    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)
    reference = np.asarray([0.0, 0.0, 1.0])
    if abs(float(np.dot(reference, normal))) > 0.9:
        reference = np.asarray([1.0, 0.0, 0.0])
    tangent_x = np.cross(reference, normal)
    tangent_x /= np.linalg.norm(tangent_x)
    tangent_y = np.cross(normal, tangent_x)
    index = np.arange(samples, dtype=float)
    z = (index + 0.5) / samples
    radial = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = index * np.pi * (3.0 - np.sqrt(5.0))
    return (
        radial[:, None] * np.cos(phi)[:, None] * tangent_x
        + radial[:, None] * np.sin(phi)[:, None] * tangent_y
        + z[:, None] * normal
    )


def probe_accessibility_curve(
    vectors: np.ndarray,
    neighbor_symbols: np.ndarray,
    marker_element: str,
    surface_normal: np.ndarray,
    probe_radii: tuple[float, ...],
    direction_samples: int,
    clearance_smoothing: float,
) -> np.ndarray:
    """Return smooth accessible fractions on the marker's upper contact shell."""

    if clearance_smoothing <= 0.0 or any(radius <= 0.0 for radius in probe_radii):
        raise ValueError("Probe radii and clearance smoothing must be positive")
    directions = upper_hemisphere_directions(surface_normal, direction_samples)
    if not len(vectors):
        return np.ones(len(probe_radii), dtype=float)
    neighbor_radii = np.asarray(
        [VDW_RADII_A.get(str(symbol), 1.80) for symbol in neighbor_symbols], dtype=float
    )
    marker_radius = VDW_RADII_A.get(marker_element, 1.80)
    maximum_probe = max(probe_radii)
    exact_reach = marker_radius + 2.0 * maximum_probe + float(np.max(neighbor_radii))
    local = np.linalg.norm(vectors, axis=1) <= exact_reach + 8.0 * clearance_smoothing
    local_vectors = vectors[local]
    local_radii = neighbor_radii[local]
    if not len(local_vectors):
        return np.ones(len(probe_radii), dtype=float)
    scores = []
    for probe_radius in probe_radii:
        contact_distance = marker_radius + probe_radius
        probe_centers = directions * contact_distance
        separation = np.linalg.norm(
            probe_centers[:, None, :] - local_vectors[None, :, :], axis=2
        )
        clearance = separation - (probe_radius + local_radii[None, :])
        minimum_clearance = np.min(clearance, axis=1)
        scores.append(float(np.mean(stable_sigmoid(minimum_clearance / clearance_smoothing))))
    return np.asarray(scores, dtype=float)


def normalized_curve_area(x: tuple[float, ...], y: np.ndarray) -> float:
    if len(x) == 1:
        return float(y[0])
    return float(np.trapz(y, np.asarray(x, dtype=float)) / (max(x) - min(x)))


def soft_top_height(heights: np.ndarray, softness: float) -> float:
    """Return a smooth upper-envelope height without selecting one maximum atom."""

    if softness <= 0.0:
        raise ValueError("Thickness softness must be positive")
    heights = np.asarray(heights, dtype=float)
    if not len(heights):
        raise ValueError("Cannot calculate thickness from an empty atom selection")
    shifted = (heights - np.max(heights)) / softness
    weights = np.exp(shifted)
    return float(np.sum(weights * heights) / np.sum(weights))


def probe_column(radius: float) -> str:
    label = f"{radius:g}".replace(".", "p")
    return f"S_Probe_Access_R{label}_A"


def thickness_column(softness: float) -> str:
    label = f"{softness:g}".replace(".", "p")
    return f"SAM_Soft_Top_Thickness_L{label}_A"


def write_csv(path: Path, rows: list[dict], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def select_final_window(
    frames: list[tuple], last_ps: float, timestep_fs: float
) -> list[tuple]:
    if not frames:
        raise ValueError("Trajectory contains no frames")
    final_step = frames[-1][0]
    cutoff_step = final_step - int(round(last_ps * 1000.0 / timestep_fs))
    selected = [frame for frame in frames if frame[0] >= cutoff_step]
    if not selected or selected[0][0] != cutoff_step:
        raise ValueError(
            f"Requested window begins at step {cutoff_step}, but the first available "
            f"selected step is {selected[0][0] if selected else None}"
        )
    return selected


def kabsch_map_vector(
    reference: np.ndarray,
    target: np.ndarray,
    vector: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Rigidly align row-vector coordinates and rotate a reference vector."""

    reference = np.asarray(reference, dtype=float)
    target = np.asarray(target, dtype=float)
    vector = np.asarray(vector, dtype=float)
    if reference.shape != target.shape or reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError("Reference and target alignment coordinates must be matching Nx3 arrays")
    if len(reference) < 3 or vector.shape != (3,):
        raise ValueError("Kabsch mapping requires at least three atoms and one 3-vector")
    centered_reference = reference - np.mean(reference, axis=0)
    centered_target = target - np.mean(target, axis=0)
    u, _, vh = np.linalg.svd(centered_reference.T @ centered_target)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vh
    aligned = centered_reference @ rotation
    rmsd = float(np.sqrt(np.mean(np.sum((aligned - centered_target) ** 2, axis=1))))
    return vector @ rotation, rmsd, rotation


def surface_tangent_basis(cell: np.ndarray, normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic in-plane basis tied to the first usable cell vector."""

    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)
    for cell_vector in np.asarray(cell, dtype=float):
        tangent_x = cell_vector - np.dot(cell_vector, normal) * normal
        length = np.linalg.norm(tangent_x)
        if length > 1.0e-10:
            tangent_x /= length
            tangent_y = np.cross(normal, tangent_x)
            tangent_y /= np.linalg.norm(tangent_y)
            return tangent_x, tangent_y
    raise ValueError("Cell contains no vector that can define a surface tangent")


def side_view_basis(cell: np.ndarray, normal: np.ndarray) -> dict:
    """Define a side-view frame looking along the shorter surface cell vector.

    The first two cell vectors are treated as the periodic surface vectors.  The
    viewing direction follows the shorter projected vector, while the plotted
    horizontal direction is the longer vector after projection into the image
    plane.  The plotted vertical direction is the outward fitted surface normal.
    """

    cell = np.asarray(cell, dtype=float)
    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)
    surface_vectors = []
    surface_lengths = []
    for vector in cell[:2]:
        projected = vector - np.dot(vector, normal) * normal
        length = float(np.linalg.norm(projected))
        if length <= 1.0e-10:
            raise ValueError("A surface cell vector is parallel to the surface normal")
        surface_vectors.append(projected)
        surface_lengths.append(length)
    short_index = int(np.argmin(surface_lengths))
    long_index = 1 - short_index
    view_axis = surface_vectors[short_index] / surface_lengths[short_index]
    visible_long = surface_vectors[long_index] - np.dot(
        surface_vectors[long_index], view_axis
    ) * view_axis
    long_period = float(np.linalg.norm(visible_long))
    if long_period <= 1.0e-10:
        raise ValueError("The two projected surface cell vectors are collinear")
    long_axis = visible_long / long_period
    return {
        "view_axis": view_axis,
        "long_axis": long_axis,
        "normal_axis": normal,
        "short_cell_vector_index": short_index,
        "long_cell_vector_index": long_index,
        "short_period_A": surface_lengths[short_index],
        "long_period_A": long_period,
    }


def periodic_mean(values: list[float] | np.ndarray, period: float) -> float:
    """Return a circular mean in ``[0, period)`` for one periodic coordinate."""

    values = np.asarray(values, dtype=float)
    if not len(values) or period <= 0.0:
        raise ValueError("Periodic mean requires values and a positive period")
    angles = 2.0 * np.pi * values / period
    sine = float(np.mean(np.sin(angles)))
    cosine = float(np.mean(np.cos(angles)))
    if np.hypot(sine, cosine) <= 1.0e-12:
        return float(np.mean(np.mod(values, period)) % period)
    return float((np.arctan2(sine, cosine) % (2.0 * np.pi)) * period / (2.0 * np.pi))


def scale_dipole_rows(
    raw_rows: list[dict], time_rows: list[dict], scale_factor: float
) -> None:
    """Uniformly scale dipole magnitudes while preserving every direction."""

    if not np.isfinite(scale_factor) or scale_factor <= 0.0:
        raise ValueError("Dipole scale factor must be finite and positive")
    raw_columns = (
        "Dipole_X_Debye",
        "Dipole_Y_Debye",
        "Dipole_Z_Debye",
        "Dipole_Total_Debye",
        "Dipole_Surface_Normal_Debye",
        "Dipole_InPlane_Debye",
        "Dipole_View_Short_Debye",
        "Dipole_Side_Long_Debye",
    )
    time_columns = (
        "Dipole_Vector_Mean_X_Debye",
        "Dipole_Vector_Mean_Y_Debye",
        "Dipole_Vector_Mean_Z_Debye",
        "Dipole_Vector_Mean_Magnitude_Debye",
        "Dipole_Surface_Normal_Mean_Debye",
        "Dipole_InPlane_Magnitude_Mean_Debye",
        "Dipole_Mean_Vector_InPlane_Debye",
        "Surface_Polarization_D_per_A2",
    )
    for row in raw_rows:
        for column in raw_columns:
            row[column] *= scale_factor
    for row in time_rows:
        for column in time_columns:
            row[column] *= scale_factor


def balanced_sign_scale_factors(
    normal_components: list[float] | np.ndarray, target_mean: float
) -> tuple[float, float, float]:
    """Return equal-percentage negative extension and positive contraction.

    Negative vectors receive ``1 + delta`` and positive vectors receive
    ``1 - delta``.  The factors are chosen so the transformed signed mean
    equals ``target_mean`` without changing the sign of any nonzero sample.
    """

    values = np.asarray(normal_components, dtype=float)
    if not len(values):
        raise ValueError("Balanced sign scaling requires normal components")
    negative_contribution = float(np.mean(np.where(values < 0.0, values, 0.0)))
    positive_contribution = float(np.mean(np.where(values > 0.0, values, 0.0)))
    denominator = negative_contribution - positive_contribution
    if abs(denominator) <= 1.0e-12:
        raise ValueError("Signed dipole distribution cannot define balanced scale factors")
    delta = float((target_mean - np.mean(values)) / denominator)
    negative_scale = 1.0 + delta
    positive_scale = 1.0 - delta
    if negative_scale <= 0.0 or positive_scale <= 0.0:
        raise ValueError(
            "Target requires a non-positive sign scale and would reverse or erase vectors"
        )
    return negative_scale, positive_scale, delta


def scale_dipole_rows_by_normal_sign(
    raw_rows: list[dict], negative_scale: float, positive_scale: float
) -> None:
    """Scale each complete vector according to its original normal-component sign."""

    for row in raw_rows:
        normal_component = row["Dipole_Surface_Normal_Debye"]
        scale_factor = negative_scale if normal_component < 0.0 else positive_scale
        scale_dipole_rows([row], [], scale_factor)


def rebuild_dipole_time_rows(
    raw_rows: list[dict], frame_geometries: list[dict]
) -> list[dict]:
    """Rebuild frame statistics after uniform or sign-dependent vector scaling."""

    rebuilt = []
    for frame_number, geometry in enumerate(frame_geometries):
        frame_rows = [row for row in raw_rows if row["Frame"] == frame_number]
        if not frame_rows:
            raise ValueError(f"No dipole rows found for frame {frame_number}")
        vectors = np.asarray(
            [[row["Dipole_X_Debye"], row["Dipole_Y_Debye"], row["Dipole_Z_Debye"]]
             for row in frame_rows]
        )
        totals = np.asarray([row["Dipole_Total_Debye"] for row in frame_rows])
        units = vectors / totals[:, None]
        mean_vector = np.mean(vectors, axis=0)
        mean_unit = np.mean(units, axis=0)
        normal_values = np.asarray(
            [row["Dipole_Surface_Normal_Debye"] for row in frame_rows]
        )
        in_plane_values = np.asarray(
            [row["Dipole_InPlane_Debye"] for row in frame_rows]
        )
        surface_normal = geometry["surface_normal"]
        mean_vector_in_plane = (
            mean_vector - np.dot(mean_vector, surface_normal) * surface_normal
        )
        rebuilt.append(
            {
                "Step": frame_rows[0]["Step"],
                "Time_ps": frame_rows[0]["Time_ps"],
                "Dipole_Vector_Mean_X_Debye": float(mean_vector[0]),
                "Dipole_Vector_Mean_Y_Debye": float(mean_vector[1]),
                "Dipole_Vector_Mean_Z_Debye": float(mean_vector[2]),
                "Dipole_Vector_Mean_Magnitude_Debye": float(np.linalg.norm(mean_vector)),
                "Dipole_Surface_Normal_Mean_Debye": float(np.mean(normal_values)),
                "Dipole_InPlane_Magnitude_Mean_Debye": float(np.mean(in_plane_values)),
                "Dipole_Mean_Vector_InPlane_Debye": float(np.linalg.norm(mean_vector_in_plane)),
                "Dipole_Surface_Angle_Mean_deg": float(
                    np.mean([row["Dipole_Surface_Angle_deg"] for row in frame_rows])
                ),
                "Dipole_Orientation_Resultant": float(np.linalg.norm(mean_unit)),
                "Dipole_S2": float(
                    0.5 * (3.0 * np.mean((normal_values / totals) ** 2) - 1.0)
                ),
                "Surface_Polarization_D_per_A2": float(
                    np.sum(normal_values) / geometry["surface_area"]
                ),
                "Dipole_Kabsch_RMSD_Mean_A": float(
                    np.mean([row["Dipole_Kabsch_RMSD_A"] for row in frame_rows])
                ),
            }
        )
    return rebuilt


def plot_dipole_side_view(
    args: argparse.Namespace,
    selected_frame: tuple,
    symbols: np.ndarray,
    definitions: list[dict],
    per_molecule_rows: list[dict],
    stem: str,
) -> tuple[Path, Path]:
    """Plot time-averaged molecular dipoles over a short-axis side projection."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    _, cell, positions, _ = selected_frame
    surface_indices = top_surface_indices(
        positions,
        symbols,
        args.substrate_atoms,
        args.surface_element,
        args.surface_window,
    )
    surface_centroid, surface_normal = fit_plane(positions[surface_indices])
    basis = side_view_basis(cell, surface_normal)
    long_axis = basis["long_axis"]
    normal_axis = basis["normal_axis"]
    long_period = basis["long_period_A"]

    figure, axis = plt.subplots(figsize=(15.5, 5.8))
    element_colors = {
        "H": "#d1d5db",
        "C": "#374151",
        "N": "#2563eb",
        "O": "#dc2626",
        "P": "#f59e0b",
        "S": "#eab308",
        "In": "#8b5cf6",
        "Sn": "#0f766e",
    }
    element_sizes = {"H": 4, "C": 8, "N": 10, "O": 8, "P": 12, "S": 14, "In": 9, "Sn": 11}

    def side_coordinates(coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        horizontal = np.mod(coordinates @ long_axis, long_period)
        vertical = (coordinates - surface_centroid) @ normal_axis
        return horizontal, vertical

    substrate_positions = positions[: args.substrate_atoms]
    substrate_symbols = symbols[: args.substrate_atoms]
    for element in sorted(set(substrate_symbols.tolist())):
        selected = substrate_symbols == element
        horizontal, vertical = side_coordinates(substrate_positions[selected])
        axis.scatter(
            horizontal,
            vertical,
            s=element_sizes.get(element, 8),
            c=element_colors.get(element, "#6b7280"),
            alpha=0.42 if element == "H" else 0.62,
            edgecolors="none",
            rasterized=True,
            label=element,
            zorder=1,
        )

    sam_heavy_coordinates = []
    sam_heavy_symbols = []
    for molecule, definition in enumerate(definitions):
        start = args.substrate_atoms + molecule * args.atoms_per_molecule
        stop = start + args.atoms_per_molecule
        block = unwrap_block(positions[start:stop], cell, definition["core"][0])
        heavy = np.flatnonzero(symbols[start:stop] != "H")
        core_centroid = np.mean(block[definition["core"]], axis=0)
        centroid_horizontal = float(np.mod(np.dot(core_centroid, long_axis), long_period))
        relative_horizontal = (block[heavy] - core_centroid) @ long_axis
        horizontal = centroid_horizontal + relative_horizontal
        vertical = (block[heavy] - surface_centroid) @ normal_axis
        sam_heavy_coordinates.append(np.column_stack((horizontal, vertical)))
        sam_heavy_symbols.extend(symbols[start:stop][heavy].tolist())
    sam_heavy_coordinates = np.vstack(sam_heavy_coordinates)
    sam_heavy_symbols = np.asarray(sam_heavy_symbols)
    for element in sorted(set(sam_heavy_symbols.tolist())):
        selected = sam_heavy_symbols == element
        axis.scatter(
            sam_heavy_coordinates[selected, 0],
            sam_heavy_coordinates[selected, 1],
            s=element_sizes.get(element, 8),
            c=element_colors.get(element, "#6b7280"),
            alpha=0.22,
            edgecolors="none",
            rasterized=True,
            label=element,
            zorder=2,
        )

    origins_long = np.asarray([row["Core_Centroid_Long_Mean_A"] for row in per_molecule_rows])
    origins_height = np.asarray([row["Core_Height_Mean_A"] for row in per_molecule_rows])
    dipoles_long = np.asarray([row["Dipole_Side_Long_Mean_Debye"] for row in per_molecule_rows])
    dipoles_normal = np.asarray(
        [row["Dipole_Surface_Normal_Mean_Debye"] for row in per_molecule_rows]
    )
    color_limit = max(float(np.max(np.abs(dipoles_normal))), 1.0e-6)
    arrows = axis.quiver(
        origins_long,
        origins_height,
        args.dipole_arrow_scale * dipoles_long,
        args.dipole_arrow_scale * dipoles_normal,
        dipoles_normal,
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-color_limit, vcenter=0.0, vmax=color_limit),
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.0033,
        headwidth=4.2,
        headlength=5.2,
        headaxislength=4.5,
        alpha=0.88,
        edgecolor="#111827",
        linewidth=0.22,
        zorder=5,
    )
    axis.scatter(
        origins_long,
        origins_height,
        s=12,
        c=dipoles_normal,
        cmap="coolwarm",
        norm=arrows.norm,
        edgecolors="#111827",
        linewidths=0.25,
        zorder=6,
    )
    axis.axhline(0.0, color="#111827", linestyle="--", linewidth=0.9, alpha=0.65)
    axis.text(
        0.995,
        0.97,
        f"Arrow scale: {args.dipole_arrow_scale:g} Å D$^{{-1}}$",
        ha="right",
        va="top",
        transform=axis.transAxes,
        fontsize=9,
    )
    axis.set(
        xlabel=(
            f"Long in-plane cell direction (Å; cell vector "
            f"{basis['long_cell_vector_index'] + 1})"
        ),
        ylabel="Height from fitted ITO surface (Å)",
        xlim=(-1.0, long_period + 1.0),
        title=(
            f"{args.prefix}: per-molecule mean dipoles, final {args.last_ps:g} ps\n"
            f"viewed along short cell vector {basis['short_cell_vector_index'] + 1}"
        ),
    )
    axis.grid(alpha=0.18, linestyle="--")
    colorbar = figure.colorbar(arrows, ax=axis, pad=0.012)
    colorbar.set_label("Mean outward-normal dipole component (D)")
    handles, labels = axis.get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    axis.legend(
        unique.values(),
        unique.keys(),
        title="Projected atoms",
        loc="lower right",
        ncol=min(4, max(1, len(unique))),
        fontsize=7,
        title_fontsize=8,
        framealpha=0.82,
    )
    figure.tight_layout()
    png_path = args.output_dir / f"{stem}_side_view_short_axis.png"
    svg_path = args.output_dir / f"{stem}_side_view_short_axis.svg"
    figure.savefig(png_path, dpi=320)
    figure.savefig(svg_path)
    plt.close(figure)
    return png_path, svg_path


def induced_covalent_graph(
    coordinates: np.ndarray,
    symbols: np.ndarray,
    atom_indices: list[int] | np.ndarray,
) -> tuple[dict[int, set[int]], np.ndarray]:
    """Build a nonperiodic labeled covalent graph for an ordered atom subset."""

    atom_indices = np.asarray(atom_indices, dtype=int)
    subset_coordinates = np.asarray(coordinates, dtype=float)[atom_indices]
    subset_symbols = np.asarray(symbols)[atom_indices]
    adjacency = {index: set() for index in range(len(atom_indices))}
    for atom_i in range(len(atom_indices)):
        for atom_j in range(atom_i + 1, len(atom_indices)):
            distance = float(np.linalg.norm(subset_coordinates[atom_j] - subset_coordinates[atom_i]))
            cutoff = covalent_bond_cutoff(subset_symbols[atom_i], subset_symbols[atom_j])
            if distance < cutoff:
                adjacency[atom_i].add(atom_j)
                adjacency[atom_j].add(atom_i)
    return adjacency, subset_symbols


def labeled_graph_isomorphisms(
    source_adjacency: dict[int, set[int]],
    source_symbols: np.ndarray,
    target_adjacency: dict[int, set[int]],
    target_symbols: np.ndarray,
    max_candidates: int = 10000,
):
    """Yield label-preserving graph mappings from source nodes to target nodes."""

    if len(source_symbols) != len(target_symbols):
        return

    def signature(node: int, adjacency: dict[int, set[int]], symbols: np.ndarray):
        return (
            str(symbols[node]),
            len(adjacency[node]),
            tuple(sorted(str(symbols[neighbor]) for neighbor in adjacency[node])),
        )

    domains = {}
    for source_node in range(len(source_symbols)):
        source_signature = signature(source_node, source_adjacency, source_symbols)
        domains[source_node] = [
            target_node
            for target_node in range(len(target_symbols))
            if signature(target_node, target_adjacency, target_symbols) == source_signature
        ]
        if not domains[source_node]:
            return
    order = sorted(
        range(len(source_symbols)),
        key=lambda node: (len(domains[node]), -len(source_adjacency[node]), node),
    )
    mapping: dict[int, int] = {}
    used_targets: set[int] = set()
    yielded = 0

    def search(depth: int):
        nonlocal yielded
        if depth == len(order):
            yielded += 1
            if yielded > max_candidates:
                raise ValueError(
                    f"More than {max_candidates} labeled graph mappings were found; "
                    "the dipole reference mapping is underdetermined"
                )
            yield dict(mapping)
            return
        source_node = order[depth]
        for target_node in domains[source_node]:
            if target_node in used_targets:
                continue
            if any(
                ((other_source in source_adjacency[source_node]) !=
                 (other_target in target_adjacency[target_node]))
                for other_source, other_target in mapping.items()
            ):
                continue
            mapping[source_node] = target_node
            used_targets.add(target_node)
            yield from search(depth + 1)
            used_targets.remove(target_node)
            del mapping[source_node]

    yield from search(0)


def resolve_core_atom_mapping(
    reference_coordinates: np.ndarray,
    reference_symbols: np.ndarray,
    reference_core: np.ndarray,
    sam_coordinates: np.ndarray,
    sam_symbols: np.ndarray,
    sam_core: list[int],
    dipole: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    """Resolve atom ordering by graph isomorphism and lowest geometric-fit RMSD."""

    reference_graph, reference_core_symbols = induced_covalent_graph(
        reference_coordinates, reference_symbols, reference_core
    )
    sam_graph, sam_core_symbols = induced_covalent_graph(
        sam_coordinates, sam_symbols, sam_core
    )
    best_mapping = None
    best_rmsd = float("inf")
    candidate_count = 0
    sam_core = np.asarray(sam_core, dtype=int)
    for mapping in labeled_graph_isomorphisms(
        reference_graph,
        reference_core_symbols,
        sam_graph,
        sam_core_symbols,
    ):
        candidate_count += 1
        ordered_sam_core = sam_core[[mapping[index] for index in range(len(reference_core))]]
        _, rmsd, _ = kabsch_map_vector(
            reference_coordinates[reference_core],
            sam_coordinates[ordered_sam_core],
            dipole,
        )
        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_mapping = ordered_sam_core
    if best_mapping is None:
        raise ValueError(
            "No element- and bond-preserving mapping exists between the QM reference "
            "and the topologically discovered SAM conjugated core"
        )
    return best_mapping, candidate_count, best_rmsd


def load_dipole_reference(
    path: Path,
    system: str,
    block_symbols: np.ndarray,
    discovered_core: list[int],
    first_block_coordinates: np.ndarray,
) -> dict:
    """Load one QM dipole and resolve its core mapping without assuming atom order."""

    payload = json.loads(Path(path).read_text())
    records = payload.get("records", [])
    matches = [
        record for record in records
        if str(record.get("system", "")).casefold() == system.casefold()
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one dipole reference for {system}; found {len(matches)}")
    record = matches[0]
    coordinates = np.asarray(record["reference_coordinates_A"], dtype=float)
    atomic_numbers = np.asarray(record["atomic_numbers"], dtype=int)
    dipole = np.asarray(record["dipole_vector_standard_orientation_debye"], dtype=float)
    reference_core = np.asarray(record["alignment_reference_core_indices_1based"], dtype=int) - 1
    if coordinates.shape != (len(atomic_numbers), 3) or dipole.shape != (3,):
        raise ValueError("Invalid reference coordinate or dipole-vector shape")
    if len(reference_core) != len(discovered_core) or len(reference_core) < 3:
        raise ValueError("Dipole reference and SAM core contain different atom counts")
    if np.any(reference_core < 0) or np.any(reference_core >= len(coordinates)):
        raise ValueError("Reference alignment indices are out of range")
    reference_symbols = np.asarray([chemical_symbols[value] for value in atomic_numbers])
    sam_core, mapping_candidates, mapping_rmsd = resolve_core_atom_mapping(
        coordinates,
        reference_symbols,
        reference_core,
        first_block_coordinates,
        block_symbols,
        discovered_core,
        dipole,
    )
    return {
        "record": record,
        "alignment_coordinates": coordinates[reference_core],
        "sam_core": sam_core,
        "dipole": dipole,
        "dipole_total": float(np.linalg.norm(dipole)),
        "mapping_candidates": mapping_candidates,
        "mapping_reference_rmsd_A": mapping_rmsd,
    }


def analyze(args: argparse.Namespace) -> dict:
    species = parse_species(args.species)
    frames = list(iter_lammps_dump(args.traj, species))
    selected_frames = select_final_window(frames, args.last_ps, args.timestep_fs)
    final_step = frames[-1][0]

    first_step, first_cell, first_positions, symbols = selected_frames[0]
    expected_atoms = args.substrate_atoms + args.molecules * args.atoms_per_molecule
    if len(symbols) != expected_atoms:
        raise ValueError(f"Trajectory has {len(symbols)} atoms; expected {expected_atoms}")
    surface_indices = top_surface_indices(
        first_positions,
        symbols,
        args.substrate_atoms,
        args.surface_element,
        args.surface_window,
    )
    definitions = []
    for molecule in range(args.molecules):
        start = args.substrate_atoms + molecule * args.atoms_per_molecule
        stop = start + args.atoms_per_molecule
        definitions.append(
            discover_molecule_definition(
                first_positions[start:stop],
                symbols[start:stop],
                first_cell,
                args.anchor_element,
                args.linker_end_element,
                args.marker_element,
                tuple(args.conjugated_elements),
            )
        )

    sam_indices = np.arange(args.substrate_atoms, expected_atoms)
    sam_heavy_indices = sam_indices[symbols[sam_indices] != "H"]
    sam_heavy_symbols = symbols[sam_heavy_indices]
    probe_radii = tuple(sorted(set(float(radius) for radius in args.probe_radii)))
    thickness_softnesses = tuple(
        sorted(set([float(args.thickness_softness), *args.thickness_softness_scan]))
    )
    raw_rows: list[dict] = []
    time_rows: list[dict] = []

    for frame_number, (step, cell, positions, frame_symbols) in enumerate(selected_frames):
        if not np.array_equal(frame_symbols, symbols):
            raise ValueError(f"Element/type ordering changed at step {step}")
        surface_centroid, surface_normal = fit_plane(positions[surface_indices])
        heavy_positions = positions[sam_heavy_indices]
        frame_rows = []
        for molecule, definition in enumerate(definitions, 1):
            start = args.substrate_atoms + (molecule - 1) * args.atoms_per_molecule
            stop = start + args.atoms_per_molecule
            block = unwrap_block(positions[start:stop], cell, definition["core"][0])
            core_local = np.asarray(definition["core"], dtype=int)
            core_positions = block[core_local]
            _, core_normal = fit_plane(core_positions)
            plane_angle = float(
                np.degrees(
                    np.arccos(np.clip(abs(np.dot(core_normal, surface_normal)), 0.0, 1.0))
                )
            )
            core_distances = (core_positions - surface_centroid) @ surface_normal
            closest_offset = int(np.argmin(core_distances))
            closest_local = int(core_local[closest_offset])
            marker_local = definition["marker"]
            marker_position = block[marker_local]
            marker_height = float(np.dot(marker_position - surface_centroid, surface_normal))

            vectors = mic_vectors(heavy_positions - marker_position, cell)
            vertical = vectors @ surface_normal
            lateral = np.linalg.norm(vectors - np.outer(vertical, surface_normal), axis=1)
            local = lateral <= args.exposure_radius
            above = local & (vertical > args.blocker_vertical_tolerance)
            local_vertical = vertical[local]
            burial_depth = float(max(0.0, np.max(local_vertical))) if len(local_vertical) else 0.0
            blocker_count = int(np.count_nonzero(above))
            exposed = int(
                blocker_count == 0 and burial_depth <= args.exposed_depth_threshold
            )
            marker_global = start + marker_local
            neighbor_mask = sam_heavy_indices != marker_global
            neighbor_vectors = vectors[neighbor_mask]
            neighbor_symbols = sam_heavy_symbols[neighbor_mask]
            height_rank, shielding_exposure, blocker_load = continuous_exposure_metrics(
                neighbor_vectors,
                surface_normal,
                args.exposure_lateral_scale,
                args.exposure_height_smoothing,
            )
            probe_curve = probe_accessibility_curve(
                neighbor_vectors,
                neighbor_symbols,
                args.marker_element,
                surface_normal,
                probe_radii,
                args.probe_direction_samples,
                args.probe_clearance_smoothing,
            )
            probe_auc = normalized_curve_area(probe_radii, probe_curve)
            block_heavy = symbols[start:stop] != "H"
            block_heights = (block[block_heavy] - surface_centroid) @ surface_normal
            thickness_curve = np.asarray(
                [soft_top_height(block_heights, value) for value in thickness_softnesses]
            )
            soft_thickness = float(
                thickness_curve[thickness_softnesses.index(args.thickness_softness)]
            )
            row = {
                "Frame": frame_number,
                "Step": step,
                "Time_ps": step * args.timestep_fs / 1000.0,
                "Mol_Index": molecule,
                "Plane_Angle_deg": plane_angle,
                "Pi_Min_Distance_A": float(core_distances[closest_offset]),
                "Closest_Pi_Atom_Local_1based": closest_local + 1,
                "Closest_Pi_Atom_Element": str(symbols[start + closest_local]),
                "S_Height_A": marker_height,
                "S_Local_Top_Burial_A": burial_depth,
                "S_Upper_Blocker_Count": blocker_count,
                "S_Exposed": exposed,
                "S_Height_Rank_Score": height_rank,
                "S_Soft_Shielding_Exposure": shielding_exposure,
                "S_Soft_Blocker_Load": blocker_load,
                "S_Probe_Access_AUC": probe_auc,
                "SAM_Soft_Top_Thickness_A": soft_thickness,
            }
            row.update(
                {probe_column(radius): float(value) for radius, value in zip(probe_radii, probe_curve)}
            )
            row.update(
                {
                    thickness_column(softness): float(value)
                    for softness, value in zip(thickness_softnesses, thickness_curve)
                }
            )
            raw_rows.append(row)
            frame_rows.append(row)
        time_rows.append(
            {
                "Step": step,
                "Time_ps": step * args.timestep_fs / 1000.0,
                "Plane_Angle_Mean_deg": float(np.mean([r["Plane_Angle_deg"] for r in frame_rows])),
                "Pi_Min_Distance_Mean_A": float(np.mean([r["Pi_Min_Distance_A"] for r in frame_rows])),
                "S_Burial_Mean_A": float(np.mean([r["S_Local_Top_Burial_A"] for r in frame_rows])),
                "S_Exposed_Fraction": float(np.mean([r["S_Exposed"] for r in frame_rows])),
                "S_Height_Rank_Mean": float(np.mean([r["S_Height_Rank_Score"] for r in frame_rows])),
                "S_Soft_Shielding_Mean": float(np.mean([r["S_Soft_Shielding_Exposure"] for r in frame_rows])),
                "S_Probe_Access_AUC_Mean": float(np.mean([r["S_Probe_Access_AUC"] for r in frame_rows])),
                "SAM_Soft_Top_Thickness_Mean_A": float(np.mean([r["SAM_Soft_Top_Thickness_A"] for r in frame_rows])),
                "SAM_Upper_Surface_Roughness_A": float(np.std([r["SAM_Soft_Top_Thickness_A"] for r in frame_rows])),
            }
        )

    per_molecule_rows = []
    for molecule in range(1, args.molecules + 1):
        rows = [row for row in raw_rows if row["Mol_Index"] == molecule]
        angle = summarize([row["Plane_Angle_deg"] for row in rows])
        distance = summarize([row["Pi_Min_Distance_A"] for row in rows])
        burial = summarize([row["S_Local_Top_Burial_A"] for row in rows])
        height_rank = summarize([row["S_Height_Rank_Score"] for row in rows])
        shielding = summarize([row["S_Soft_Shielding_Exposure"] for row in rows])
        probe_auc = summarize([row["S_Probe_Access_AUC"] for row in rows])
        thickness = summarize([row["SAM_Soft_Top_Thickness_A"] for row in rows])
        per_molecule_rows.append(
            {
                "Mol_Index": molecule,
                "Plane_Angle_Mean_deg": angle["mean"],
                "Plane_Angle_Std_deg": angle["std"],
                "Plane_Angle_Min_deg": angle["min"],
                "Plane_Angle_Max_deg": angle["max"],
                "Pi_Min_Distance_Mean_A": distance["mean"],
                "Pi_Min_Distance_Std_A": distance["std"],
                "Pi_Min_Distance_Min_A": distance["min"],
                "Pi_Min_Distance_Max_A": distance["max"],
                "S_Local_Top_Burial_Mean_A": burial["mean"],
                "S_Local_Top_Burial_Std_A": burial["std"],
                "S_Upper_Blocker_Mean": float(
                    np.mean([row["S_Upper_Blocker_Count"] for row in rows])
                ),
                "S_Exposed_Fraction": float(np.mean([row["S_Exposed"] for row in rows])),
                "S_Height_Rank_Mean": height_rank["mean"],
                "S_Height_Rank_Std": height_rank["std"],
                "S_Soft_Shielding_Mean": shielding["mean"],
                "S_Soft_Shielding_Std": shielding["std"],
                "S_Probe_Access_AUC_Mean": probe_auc["mean"],
                "S_Probe_Access_AUC_Std": probe_auc["std"],
                "SAM_Soft_Top_Thickness_Mean_A": thickness["mean"],
                "SAM_Soft_Top_Thickness_Std_A": thickness["std"],
            }
        )
        per_molecule_rows[-1].update(
            {
                f"{probe_column(radius)}_Mean": float(
                    np.mean([row[probe_column(radius)] for row in rows])
                )
                for radius in probe_radii
            }
        )
        per_molecule_rows[-1].update(
            {
                f"{thickness_column(softness)}_Mean": float(
                    np.mean([row[thickness_column(softness)] for row in rows])
                )
                for softness in thickness_softnesses
            }
        )

    raw_headers = list(raw_rows[0])
    time_headers = list(time_rows[0])
    molecule_headers = list(per_molecule_rows[0])
    write_csv(args.output_dir / f"{args.prefix}_last_{args.last_ps:g}ps_raw.csv", raw_rows, raw_headers)
    write_csv(
        args.output_dir / f"{args.prefix}_last_{args.last_ps:g}ps_timeseries.csv",
        time_rows,
        time_headers,
    )
    write_csv(
        args.output_dir / f"{args.prefix}_last_{args.last_ps:g}ps_per_molecule.csv",
        per_molecule_rows,
        molecule_headers,
    )

    summary = {
        "trajectory": str(args.traj),
        "window": {
            "last_ps": args.last_ps,
            "start_step": selected_frames[0][0],
            "end_step": final_step,
            "start_time_ps": selected_frames[0][0] * args.timestep_fs / 1000.0,
            "end_time_ps": final_step * args.timestep_fs / 1000.0,
            "frame_count": len(selected_frames),
            "molecule_count": args.molecules,
            "sample_count": len(raw_rows),
        },
        "definitions": {
            "plane_angle": (
                "acute angle between the SVD-fitted conjugated C/N/S plane and "
                f"the SVD-fitted top-substrate {args.surface_element} plane"
            ),
            "pi_min_distance": (
                "minimum signed normal distance from a conjugated non-H atom to "
                "the fitted substrate surface plane"
            ),
            "s_exposure": (
                "S is exposed when no SAM non-H atom lies above it within the "
                f"{args.exposure_radius:g} A lateral radius and its local top burial "
                f"is <= {args.exposed_depth_threshold:g} A"
            ),
            "s_height_rank": (
                "Gaussian lateral weighting and logistic height comparison; "
                "1 means S lies above its local SAM-heavy-atom environment"
            ),
            "s_soft_shielding": (
                "exp(-soft blocker load), using Gaussian lateral weighting and "
                "a logistic above/below comparison"
            ),
            "s_probe_access": (
                "mean soft steric accessibility over an equal-area upper-hemisphere "
                "contact shell for each probe radius; AUC is normalized over radius"
            ),
            "sam_soft_top_thickness": (
                "per-molecule Boltzmann-weighted upper envelope of SAM non-H atom "
                "heights above the fitted substrate plane"
            ),
            "continuous_parameters": {
                "lateral_scale_A": args.exposure_lateral_scale,
                "height_smoothing_A": args.exposure_height_smoothing,
                "probe_radii_A": list(probe_radii),
                "probe_direction_samples": args.probe_direction_samples,
                "probe_clearance_smoothing_A": args.probe_clearance_smoothing,
                "thickness_softness_A": args.thickness_softness,
                "thickness_softness_scan_A": list(thickness_softnesses),
            },
            "surface_atom_count": int(len(surface_indices)),
            "conjugated_elements": list(args.conjugated_elements),
            "conjugated_atom_count_per_molecule": [len(item["core"]) for item in definitions],
            "linker_carbon_count_per_molecule": [
                len(item["linker_carbons"]) for item in definitions
            ],
        },
        "overall_samples": {
            "plane_angle_deg": summarize([row["Plane_Angle_deg"] for row in raw_rows]),
            "pi_min_distance_A": summarize([row["Pi_Min_Distance_A"] for row in raw_rows]),
            "s_height_A": summarize([row["S_Height_A"] for row in raw_rows]),
            "s_local_top_burial_A": summarize(
                [row["S_Local_Top_Burial_A"] for row in raw_rows]
            ),
            "s_upper_blocker_count": summarize(
                [row["S_Upper_Blocker_Count"] for row in raw_rows]
            ),
            "s_exposed_fraction": float(np.mean([row["S_Exposed"] for row in raw_rows])),
            "s_height_rank_score": summarize([row["S_Height_Rank_Score"] for row in raw_rows]),
            "s_soft_shielding_exposure": summarize([row["S_Soft_Shielding_Exposure"] for row in raw_rows]),
            "s_soft_blocker_load": summarize([row["S_Soft_Blocker_Load"] for row in raw_rows]),
            "s_probe_access_auc": summarize([row["S_Probe_Access_AUC"] for row in raw_rows]),
            "sam_soft_top_thickness_A": summarize([row["SAM_Soft_Top_Thickness_A"] for row in raw_rows]),
            "s_probe_access_by_radius": {
                f"{radius:g}_A": summarize([row[probe_column(radius)] for row in raw_rows])
                for radius in probe_radii
            },
            "sam_soft_top_thickness_by_softness": {
                f"{softness:g}_A": summarize(
                    [row[thickness_column(softness)] for row in raw_rows]
                )
                for softness in thickness_softnesses
            },
        },
        "frame_means": {
            "plane_angle_deg": summarize(
                [row["Plane_Angle_Mean_deg"] for row in time_rows]
            ),
            "pi_min_distance_A": summarize(
                [row["Pi_Min_Distance_Mean_A"] for row in time_rows]
            ),
            "s_local_top_burial_A": summarize(
                [row["S_Burial_Mean_A"] for row in time_rows]
            ),
            "s_exposed_fraction": summarize(
                [row["S_Exposed_Fraction"] for row in time_rows]
            ),
            "s_height_rank_score": summarize([row["S_Height_Rank_Mean"] for row in time_rows]),
            "s_soft_shielding_exposure": summarize([row["S_Soft_Shielding_Mean"] for row in time_rows]),
            "s_probe_access_auc": summarize([row["S_Probe_Access_AUC_Mean"] for row in time_rows]),
            "sam_soft_top_thickness_A": summarize([row["SAM_Soft_Top_Thickness_Mean_A"] for row in time_rows]),
            "sam_upper_surface_roughness_A": summarize([row["SAM_Upper_Surface_Roughness_A"] for row in time_rows]),
        },
    }
    summary_path = args.output_dir / f"{args.prefix}_last_{args.last_ps:g}ps_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def analyze_dipoles(args: argparse.Namespace) -> dict:
    """Map one rigid QM dipole reference onto every SAM molecule and frame."""

    species = parse_species(args.species)
    frames = list(iter_lammps_dump(args.traj, species))
    selected_frames = select_final_window(frames, args.last_ps, args.timestep_fs)
    final_step = frames[-1][0]
    _, first_cell, first_positions, symbols = selected_frames[0]
    expected_atoms = args.substrate_atoms + args.molecules * args.atoms_per_molecule
    if len(symbols) != expected_atoms:
        raise ValueError(f"Trajectory has {len(symbols)} atoms; expected {expected_atoms}")
    surface_indices = top_surface_indices(
        first_positions,
        symbols,
        args.substrate_atoms,
        args.surface_element,
        args.surface_window,
    )
    definitions = []
    for molecule in range(args.molecules):
        start = args.substrate_atoms + molecule * args.atoms_per_molecule
        stop = start + args.atoms_per_molecule
        definitions.append(
            discover_molecule_definition(
                first_positions[start:stop],
                symbols[start:stop],
                first_cell,
                args.anchor_element,
                args.linker_end_element,
                args.marker_element,
                tuple(args.conjugated_elements),
            )
        )
    first_start = args.substrate_atoms
    first_stop = first_start + args.atoms_per_molecule
    block_symbols = symbols[first_start:first_stop]
    first_block = unwrap_block(
        first_positions[first_start:first_stop],
        first_cell,
        definitions[0]["core"][0],
    )
    reference = load_dipole_reference(
        args.dipole_reference,
        args.dipole_system,
        block_symbols,
        definitions[0]["core"],
        first_block,
    )
    for molecule, definition in enumerate(definitions, 1):
        start = args.substrate_atoms + (molecule - 1) * args.atoms_per_molecule
        stop = start + args.atoms_per_molecule
        if not np.array_equal(symbols[start:stop], block_symbols):
            raise ValueError(f"Molecule {molecule} has a different element ordering")
        if set(definition["core"]) != set(reference["sam_core"].tolist()):
            raise ValueError(f"Molecule {molecule} has a different discovered core mapping")

    raw_rows: list[dict] = []
    time_rows: list[dict] = []
    frame_geometries: list[dict] = []
    first_side_basis = None
    for frame_number, (step, cell, positions, frame_symbols) in enumerate(selected_frames):
        if not np.array_equal(frame_symbols, symbols):
            raise ValueError(f"Element/type ordering changed at step {step}")
        surface_centroid, surface_normal = fit_plane(positions[surface_indices])
        tangent_x, tangent_y = surface_tangent_basis(cell, surface_normal)
        side_basis = side_view_basis(cell, surface_normal)
        if first_side_basis is None:
            first_side_basis = side_basis
        elif (
            side_basis["short_cell_vector_index"]
            != first_side_basis["short_cell_vector_index"]
        ):
            raise ValueError("The identity of the short surface cell vector changed during the window")
        surface_area = float(np.linalg.norm(np.cross(cell[0], cell[1])))
        if surface_area <= 0.0:
            raise ValueError("Simulation cell has zero surface area")
        frame_geometries.append(
            {"surface_normal": surface_normal.copy(), "surface_area": surface_area}
        )
        frame_rows = []
        for molecule, definition in enumerate(definitions, 1):
            start = args.substrate_atoms + (molecule - 1) * args.atoms_per_molecule
            stop = start + args.atoms_per_molecule
            block = unwrap_block(positions[start:stop], cell, definition["core"][0])
            target_core = block[reference["sam_core"]]
            core_centroid = np.mean(target_core, axis=0)
            dipole, alignment_rmsd, _ = kabsch_map_vector(
                reference["alignment_coordinates"],
                target_core,
                reference["dipole"],
            )
            dipole_total = float(np.linalg.norm(dipole))
            dipole_unit = dipole / dipole_total
            normal_component = float(np.dot(dipole, surface_normal))
            in_plane = dipole - normal_component * surface_normal
            in_plane_magnitude = float(np.linalg.norm(in_plane))
            core_short = float(
                np.mod(
                    np.dot(core_centroid, side_basis["view_axis"]),
                    side_basis["short_period_A"],
                )
            )
            core_long = float(
                np.mod(
                    np.dot(core_centroid, side_basis["long_axis"]),
                    side_basis["long_period_A"],
                )
            )
            core_height = float(np.dot(core_centroid - surface_centroid, surface_normal))
            dipole_short = float(np.dot(dipole, side_basis["view_axis"]))
            dipole_long = float(np.dot(dipole, side_basis["long_axis"]))
            surface_angle = float(
                np.degrees(
                    np.arccos(np.clip(normal_component / dipole_total, -1.0, 1.0))
                )
            )
            azimuth = float(
                np.degrees(
                    np.arctan2(np.dot(dipole, tangent_y), np.dot(dipole, tangent_x))
                )
            )
            row = {
                "Frame": frame_number,
                "Step": step,
                "Time_ps": step * args.timestep_fs / 1000.0,
                "Mol_Index": molecule,
                "Dipole_X_Debye": float(dipole[0]),
                "Dipole_Y_Debye": float(dipole[1]),
                "Dipole_Z_Debye": float(dipole[2]),
                "Dipole_Total_Debye": dipole_total,
                "Dipole_Surface_Normal_Debye": normal_component,
                "Dipole_InPlane_Debye": in_plane_magnitude,
                "Dipole_Surface_Angle_deg": surface_angle,
                "Dipole_Surface_Azimuth_deg": azimuth,
                "Core_Centroid_Short_A": core_short,
                "Core_Centroid_Long_A": core_long,
                "Core_Height_A": core_height,
                "Dipole_View_Short_Debye": dipole_short,
                "Dipole_Side_Long_Debye": dipole_long,
                "Dipole_Unit_X": float(dipole_unit[0]),
                "Dipole_Unit_Y": float(dipole_unit[1]),
                "Dipole_Unit_Z": float(dipole_unit[2]),
                "Dipole_Kabsch_RMSD_A": alignment_rmsd,
            }
            raw_rows.append(row)
            frame_rows.append(row)
        vectors = np.asarray(
            [[row["Dipole_X_Debye"], row["Dipole_Y_Debye"], row["Dipole_Z_Debye"]]
             for row in frame_rows]
        )
        units = vectors / np.linalg.norm(vectors, axis=1)[:, None]
        mean_vector = np.mean(vectors, axis=0)
        mean_unit = np.mean(units, axis=0)
        normal_values = np.asarray(
            [row["Dipole_Surface_Normal_Debye"] for row in frame_rows]
        )
        in_plane_values = np.asarray([row["Dipole_InPlane_Debye"] for row in frame_rows])
        cosines = normal_values / reference["dipole_total"]
        mean_normal = float(np.mean(normal_values))
        mean_vector_in_plane = mean_vector - np.dot(mean_vector, surface_normal) * surface_normal
        time_rows.append(
            {
                "Step": step,
                "Time_ps": step * args.timestep_fs / 1000.0,
                "Dipole_Vector_Mean_X_Debye": float(mean_vector[0]),
                "Dipole_Vector_Mean_Y_Debye": float(mean_vector[1]),
                "Dipole_Vector_Mean_Z_Debye": float(mean_vector[2]),
                "Dipole_Vector_Mean_Magnitude_Debye": float(np.linalg.norm(mean_vector)),
                "Dipole_Surface_Normal_Mean_Debye": mean_normal,
                "Dipole_InPlane_Magnitude_Mean_Debye": float(np.mean(in_plane_values)),
                "Dipole_Mean_Vector_InPlane_Debye": float(np.linalg.norm(mean_vector_in_plane)),
                "Dipole_Surface_Angle_Mean_deg": float(
                    np.mean([row["Dipole_Surface_Angle_deg"] for row in frame_rows])
                ),
                "Dipole_Orientation_Resultant": float(np.linalg.norm(mean_unit)),
                "Dipole_S2": float(0.5 * (3.0 * np.mean(cosines**2) - 1.0)),
                "Surface_Polarization_D_per_A2": float(np.sum(normal_values) / surface_area),
                "Dipole_Kabsch_RMSD_Mean_A": float(
                    np.mean([row["Dipole_Kabsch_RMSD_A"] for row in frame_rows])
                ),
            }
        )

    raw_normal_mean = float(
        np.mean([row["Dipole_Surface_Normal_Debye"] for row in raw_rows])
    )
    dipole_scale_factor = 1.0
    negative_scale_factor = None
    positive_scale_factor = None
    balanced_delta = None
    if args.dipole_target_normal is not None:
        if abs(raw_normal_mean) <= 1.0e-12:
            raise ValueError("Cannot scale a zero mean surface-normal dipole")
        if args.dipole_scaling_mode == "uniform":
            dipole_scale_factor = float(args.dipole_target_normal / raw_normal_mean)
            if dipole_scale_factor <= 0.0:
                raise ValueError(
                    "Target and raw normal dipoles have opposite signs; uniform positive "
                    "scaling cannot reverse dipole directions"
                )
            scale_dipole_rows(raw_rows, [], dipole_scale_factor)
        else:
            negative_scale_factor, positive_scale_factor, balanced_delta = (
                balanced_sign_scale_factors(
                    [row["Dipole_Surface_Normal_Debye"] for row in raw_rows],
                    args.dipole_target_normal,
                )
            )
            scale_dipole_rows_by_normal_sign(
                raw_rows, negative_scale_factor, positive_scale_factor
            )
    time_rows = rebuild_dipole_time_rows(raw_rows, frame_geometries)

    per_molecule_rows = []
    for molecule in range(1, args.molecules + 1):
        rows = [row for row in raw_rows if row["Mol_Index"] == molecule]
        vectors = np.asarray(
            [[row["Dipole_X_Debye"], row["Dipole_Y_Debye"], row["Dipole_Z_Debye"]]
             for row in rows]
        )
        units = vectors / np.linalg.norm(vectors, axis=1)[:, None]
        mean_vector = np.mean(vectors, axis=0)
        normal_summary = summarize([row["Dipole_Surface_Normal_Debye"] for row in rows])
        angle_summary = summarize([row["Dipole_Surface_Angle_deg"] for row in rows])
        rmsd_summary = summarize([row["Dipole_Kabsch_RMSD_A"] for row in rows])
        short_period = float(first_side_basis["short_period_A"])
        long_period = float(first_side_basis["long_period_A"])
        core_height_summary = summarize([row["Core_Height_A"] for row in rows])
        per_molecule_rows.append(
            {
                "Mol_Index": molecule,
                "Core_Centroid_Short_Mean_A": periodic_mean(
                    [row["Core_Centroid_Short_A"] for row in rows], short_period
                ),
                "Core_Centroid_Long_Mean_A": periodic_mean(
                    [row["Core_Centroid_Long_A"] for row in rows], long_period
                ),
                "Core_Height_Mean_A": core_height_summary["mean"],
                "Core_Height_Std_A": core_height_summary["std"],
                "Dipole_Mean_X_Debye": float(mean_vector[0]),
                "Dipole_Mean_Y_Debye": float(mean_vector[1]),
                "Dipole_Mean_Z_Debye": float(mean_vector[2]),
                "Dipole_View_Short_Mean_Debye": float(
                    np.mean([row["Dipole_View_Short_Debye"] for row in rows])
                ),
                "Dipole_Side_Long_Mean_Debye": float(
                    np.mean([row["Dipole_Side_Long_Debye"] for row in rows])
                ),
                "Dipole_Mean_Vector_Magnitude_Debye": float(np.linalg.norm(mean_vector)),
                "Dipole_Surface_Normal_Mean_Debye": normal_summary["mean"],
                "Dipole_Surface_Normal_Std_Debye": normal_summary["std"],
                "Dipole_Surface_Angle_Mean_deg": angle_summary["mean"],
                "Dipole_Surface_Angle_Std_deg": angle_summary["std"],
                "Dipole_Orientation_Resultant": float(np.linalg.norm(np.mean(units, axis=0))),
                "Dipole_Kabsch_RMSD_Mean_A": rmsd_summary["mean"],
                "Dipole_Kabsch_RMSD_Max_A": rmsd_summary["max"],
            }
        )

    stem = f"{args.prefix}_last_{args.last_ps:g}ps_dipole"
    write_csv(args.output_dir / f"{stem}_raw.csv", raw_rows, list(raw_rows[0]))
    write_csv(args.output_dir / f"{stem}_timeseries.csv", time_rows, list(time_rows[0]))
    write_csv(
        args.output_dir / f"{stem}_per_molecule.csv",
        per_molecule_rows,
        list(per_molecule_rows[0]),
    )
    side_view_headers = [
        "Mol_Index",
        "Core_Centroid_Short_Mean_A",
        "Core_Centroid_Long_Mean_A",
        "Core_Height_Mean_A",
        "Core_Height_Std_A",
        "Dipole_View_Short_Mean_Debye",
        "Dipole_Side_Long_Mean_Debye",
        "Dipole_Surface_Normal_Mean_Debye",
        "Dipole_Surface_Normal_Std_Debye",
        "Dipole_Mean_Vector_Magnitude_Debye",
        "Dipole_Orientation_Resultant",
    ]
    write_csv(
        args.output_dir / f"{stem}_side_view_short_axis.csv",
        [{column: row[column] for column in side_view_headers} for row in per_molecule_rows],
        side_view_headers,
    )
    all_vectors = np.asarray(
        [[row["Dipole_X_Debye"], row["Dipole_Y_Debye"], row["Dipole_Z_Debye"]]
         for row in raw_rows]
    )
    average_vector = np.mean(all_vectors, axis=0)
    reference_record = reference["record"]
    summary = {
        "trajectory": str(args.traj),
        "dipole_reference": str(args.dipole_reference),
        "dipole_system": args.dipole_system,
        "interpretation_scope": (
            "intrinsic dipole of the neutral conjugated-core plus N-methyl fragment; "
            "not the total SAM or substrate-induced interface dipole"
        ),
        "uniform_dipole_scaling": {
            "enabled": args.dipole_target_normal is not None,
            "mode": args.dipole_scaling_mode if args.dipole_target_normal is not None else "none",
            "raw_surface_normal_mean_debye": raw_normal_mean,
            "target_surface_normal_mean_debye": args.dipole_target_normal,
            "actual_surface_normal_mean_debye": float(
                np.mean([row["Dipole_Surface_Normal_Debye"] for row in raw_rows])
            ),
            "scale_factor": (
                dipole_scale_factor if args.dipole_scaling_mode == "uniform" else None
            ),
            "negative_scale_factor": negative_scale_factor,
            "positive_scale_factor": positive_scale_factor,
            "balanced_fractional_change": balanced_delta,
            "operation": (
                "positive scalar(s) applied to complete three-dimensional molecular "
                "dipole vectors; individual vector directions and normal-component signs preserved"
            ),
        },
        "window": {
            "last_ps": args.last_ps,
            "start_step": selected_frames[0][0],
            "end_step": final_step,
            "start_time_ps": selected_frames[0][0] * args.timestep_fs / 1000.0,
            "end_time_ps": final_step * args.timestep_fs / 1000.0,
            "frame_count": len(selected_frames),
            "molecule_count": args.molecules,
            "sample_count": len(raw_rows),
        },
        "reference": {
            "formula": reference_record["formula"],
            "optimization_method": reference_record["optimization_method"],
            "single_point_method": reference_record["single_point_method"],
            "dipole_total_debye": reference_record["dipole_total_debye"],
            "dipole_long_debye": reference_record["dipole_long_debye"],
            "dipole_transverse_debye": reference_record["dipole_transverse_debye"],
            "dipole_normal_debye": reference_record["dipole_normal_debye"],
            "dipole_angle_to_long_axis_deg": reference_record[
                "dipole_angle_to_long_axis_deg"
            ],
            "dipole_angle_from_core_plane_deg": reference_record[
                "dipole_angle_from_core_plane_deg"
            ],
            "molecular_axis_definition": reference_record["molecular_axis_definition"],
            "topology_mapping_candidates": reference["mapping_candidates"],
            "selected_mapping_initial_rmsd_A": reference["mapping_reference_rmsd_A"],
        },
        "definitions": {
            "surface_angle": (
                "directional angle from the outward fitted substrate normal: 0 degrees "
                "points toward vacuum, 90 degrees is in-plane, 180 degrees points toward substrate"
            ),
            "surface_normal_component": "signed dipole projection onto the outward surface normal",
            "orientation_resultant": (
                "magnitude of the mean unit dipole vector; 0 means cancellation and 1 means alignment"
            ),
            "S2": "0.5 * (3 * mean(cos(surface_angle)^2) - 1)",
            "surface_polarization": "sum of signed normal dipoles divided by instantaneous surface area",
            "mapping": "proper-rotation Kabsch fit of the 20 conjugated-core heavy atoms",
            "side_view": (
                "line of sight follows the shorter of the first two projected cell vectors; "
                "the image plane retains the longer in-plane direction and outward surface normal"
            ),
        },
        "side_view_basis": {
            "short_cell_vector_index_1based": first_side_basis[
                "short_cell_vector_index"
            ] + 1,
            "long_cell_vector_index_1based": first_side_basis[
                "long_cell_vector_index"
            ] + 1,
            "short_period_A": first_side_basis["short_period_A"],
            "long_period_A": first_side_basis["long_period_A"],
            "view_axis_cartesian": first_side_basis["view_axis"].tolist(),
            "horizontal_axis_cartesian": first_side_basis["long_axis"].tolist(),
            "outward_normal_cartesian": first_side_basis["normal_axis"].tolist(),
            "arrow_scale_A_per_D": args.dipole_arrow_scale,
            "per_molecule_projection_csv": str(
                args.output_dir / f"{stem}_side_view_short_axis.csv"
            ),
            "side_view_png": str(
                args.output_dir / f"{stem}_side_view_short_axis.png"
            ),
            "side_view_svg": str(
                args.output_dir / f"{stem}_side_view_short_axis.svg"
            ),
        },
        "vector_average_debye": {
            "x": float(average_vector[0]),
            "y": float(average_vector[1]),
            "z": float(average_vector[2]),
            "magnitude": float(np.linalg.norm(average_vector)),
        },
        "overall_samples": {
            "surface_normal_debye": summarize(
                [row["Dipole_Surface_Normal_Debye"] for row in raw_rows]
            ),
            "in_plane_debye": summarize([row["Dipole_InPlane_Debye"] for row in raw_rows]),
            "surface_angle_deg": summarize(
                [row["Dipole_Surface_Angle_deg"] for row in raw_rows]
            ),
            "kabsch_rmsd_A": summarize([row["Dipole_Kabsch_RMSD_A"] for row in raw_rows]),
        },
        "frame_means": {
            "surface_normal_debye": summarize(
                [row["Dipole_Surface_Normal_Mean_Debye"] for row in time_rows]
            ),
            "mean_vector_magnitude_debye": summarize(
                [row["Dipole_Vector_Mean_Magnitude_Debye"] for row in time_rows]
            ),
            "surface_angle_deg": summarize(
                [row["Dipole_Surface_Angle_Mean_deg"] for row in time_rows]
            ),
            "orientation_resultant": summarize(
                [row["Dipole_Orientation_Resultant"] for row in time_rows]
            ),
            "S2": summarize([row["Dipole_S2"] for row in time_rows]),
            "surface_polarization_D_per_A2": summarize(
                [row["Surface_Polarization_D_per_A2"] for row in time_rows]
            ),
            "kabsch_rmsd_A": summarize(
                [row["Dipole_Kabsch_RMSD_Mean_A"] for row in time_rows]
            ),
        },
    }
    summary_path = args.output_dir / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    if not args.no_plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(2, 2, figsize=(12, 9))
        axes[0, 0].hist(
            [row["Dipole_Surface_Angle_deg"] for row in raw_rows],
            bins=36,
            color="#2563eb",
            alpha=0.85,
        )
        axes[0, 0].set(xlabel="Dipole angle from surface normal (deg)", ylabel="Samples")
        axes[0, 1].hist(
            [row["Dipole_Surface_Normal_Debye"] for row in raw_rows],
            bins=36,
            color="#7c3aed",
            alpha=0.85,
        )
        axes[0, 1].set(xlabel="Signed normal dipole (Debye)", ylabel="Samples")
        axes[1, 0].plot(
            [row["Time_ps"] for row in time_rows],
            [row["Dipole_Surface_Normal_Mean_Debye"] for row in time_rows],
            color="#059669",
        )
        axes[1, 0].set(xlabel="Time (ps)", ylabel="Mean normal dipole (Debye)")
        axes[1, 1].plot(
            [row["Mol_Index"] for row in per_molecule_rows],
            [row["Dipole_Surface_Normal_Mean_Debye"] for row in per_molecule_rows],
            "o",
            markersize=3,
            color="#dc2626",
        )
        axes[1, 1].set(xlabel="SAM molecule index", ylabel="Mean normal dipole (Debye)")
        for axis in axes.ravel():
            axis.grid(alpha=0.25, linestyle="--")
        figure.suptitle(
            f"{args.prefix}: mapped intrinsic dipole, final {args.last_ps:g} ps"
        )
        figure.tight_layout()
        figure.savefig(args.output_dir / f"{stem}_analysis.png", dpi=240)
        plt.close(figure)
        plot_dipole_side_view(
            args,
            selected_frames[-1],
            symbols,
            definitions,
            per_molecule_rows,
            stem,
        )
    return summary


def plot_results(args: argparse.Namespace, summary: dict) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw_path = args.output_dir / f"{args.prefix}_last_{args.last_ps:g}ps_raw.csv"
    time_path = args.output_dir / f"{args.prefix}_last_{args.last_ps:g}ps_timeseries.csv"
    molecule_path = args.output_dir / f"{args.prefix}_last_{args.last_ps:g}ps_per_molecule.csv"
    raw = np.genfromtxt(raw_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    times = np.genfromtxt(time_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    molecules = np.genfromtxt(
        molecule_path, delimiter=",", names=True, dtype=None, encoding="utf-8"
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].hist(raw["Plane_Angle_deg"], bins=36, color="#4f46e5", alpha=0.85)
    axes[0, 0].set(xlabel="Plane angle (deg)", ylabel="Samples", title="Conjugated plane vs substrate")
    axes[0, 1].hist(raw["Pi_Min_Distance_A"], bins=36, color="#10b981", alpha=0.85)
    axes[0, 1].set(xlabel="Minimum distance (A)", ylabel="Samples", title="Conjugated-core closest approach")
    axes[1, 0].hist(raw["S_Local_Top_Burial_A"], bins=36, color="#f59e0b", alpha=0.85)
    axes[1, 0].set(xlabel="S local top burial (A)", ylabel="Samples", title="Sulfur upper-surface exposure")
    axes[1, 1].plot(times["Time_ps"], times["S_Exposed_Fraction"], color="#dc2626")
    axes[1, 1].set(xlabel="Time (ps)", ylabel="Exposed S fraction", ylim=(-0.02, 1.02), title="Exposed sulfur fraction vs time")
    for axis in axes.ravel():
        axis.grid(alpha=0.25, linestyle="--")
    fig.suptitle(
        f"{args.prefix}: final {summary['window']['last_ps']:g} ps ({summary['window']['frame_count']} frames)"
    )
    fig.tight_layout()
    output = args.output_dir / f"{args.prefix}_last_{args.last_ps:g}ps_analysis.png"
    fig.savefig(output, dpi=240)
    plt.close(fig)

    molecule_figure, molecule_axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    molecule_ids = molecules["Mol_Index"]
    molecule_axes[0].errorbar(
        molecule_ids,
        molecules["Plane_Angle_Mean_deg"],
        yerr=molecules["Plane_Angle_Std_deg"],
        fmt="o",
        markersize=3,
        linewidth=0.8,
        color="#4f46e5",
    )
    molecule_axes[0].set(ylabel="Plane angle (deg)", title="Per-molecule final-window statistics")
    molecule_axes[1].errorbar(
        molecule_ids,
        molecules["Pi_Min_Distance_Mean_A"],
        yerr=molecules["Pi_Min_Distance_Std_A"],
        fmt="o",
        markersize=3,
        linewidth=0.8,
        color="#10b981",
    )
    molecule_axes[1].set(ylabel="Minimum distance (A)")
    molecule_axes[2].bar(
        molecule_ids,
        molecules["S_Exposed_Fraction"],
        color="#f59e0b",
        width=0.8,
    )
    molecule_axes[2].set(
        xlabel="SAM molecule index",
        ylabel="Exposed S fraction",
        ylim=(0.0, 1.05),
    )
    for axis in molecule_axes:
        axis.grid(alpha=0.25, linestyle="--")
    molecule_figure.tight_layout()
    molecule_output = (
        args.output_dir / f"{args.prefix}_last_{args.last_ps:g}ps_per_molecule.png"
    )
    molecule_figure.savefig(molecule_output, dpi=240)
    plt.close(molecule_figure)

    continuous_figure, continuous_axes = plt.subplots(2, 3, figsize=(15, 9))
    continuous_axes[0, 0].hist(raw["S_Height_Rank_Score"], bins=36, color="#2563eb", alpha=0.85)
    continuous_axes[0, 0].set(xlabel="Height-rank exposure", ylabel="Samples", title="Smooth local height rank")
    continuous_axes[0, 1].hist(raw["S_Soft_Shielding_Exposure"], bins=36, color="#7c3aed", alpha=0.85)
    continuous_axes[0, 1].set(xlabel="Shielding exposure", ylabel="Samples", title="Continuous shielding")
    continuous_axes[0, 2].hist(raw["S_Probe_Access_AUC"], bins=36, color="#db2777", alpha=0.85)
    continuous_axes[0, 2].set(xlabel="Probe-access AUC", ylabel="Samples", title="Upper-hemisphere probe access")
    continuous_axes[1, 0].hist(raw["SAM_Soft_Top_Thickness_A"], bins=36, color="#059669", alpha=0.85)
    continuous_axes[1, 0].set(xlabel="Soft top thickness (A)", ylabel="Samples", title="Average SAM thickness")
    continuous_axes[1, 1].plot(times["Time_ps"], times["S_Height_Rank_Mean"], label="height rank", color="#2563eb")
    continuous_axes[1, 1].plot(times["Time_ps"], times["S_Probe_Access_AUC_Mean"], label="probe AUC", color="#db2777")
    continuous_axes[1, 1].set(xlabel="Time (ps)", ylabel="Exposure score", ylim=(0.0, 1.0), title="Continuous S exposure vs time")
    continuous_axes[1, 1].legend()
    thickness_mean = times["SAM_Soft_Top_Thickness_Mean_A"]
    roughness = times["SAM_Upper_Surface_Roughness_A"]
    continuous_axes[1, 2].plot(times["Time_ps"], thickness_mean, color="#059669")
    continuous_axes[1, 2].fill_between(times["Time_ps"], thickness_mean - roughness, thickness_mean + roughness, color="#6ee7b7", alpha=0.35, label="molecule-to-molecule SD")
    continuous_axes[1, 2].set(xlabel="Time (ps)", ylabel="Thickness (A)", title="SAM thickness and roughness")
    continuous_axes[1, 2].legend()
    for axis in continuous_axes.ravel():
        axis.grid(alpha=0.25, linestyle="--")
    continuous_figure.suptitle(f"{args.prefix}: continuous exposure and thickness")
    continuous_figure.tight_layout()
    continuous_output = args.output_dir / f"{args.prefix}_last_{args.last_ps:g}ps_continuous_exposure_thickness.png"
    continuous_figure.savefig(continuous_output, dpi=240)
    plt.close(continuous_figure)

    continuous_molecule_figure, continuous_molecule_axes = plt.subplots(
        2, 2, figsize=(14, 9)
    )
    continuous_molecule_axes[0, 0].plot(
        molecule_ids, molecules["S_Height_Rank_Mean"], "o-", markersize=3,
        linewidth=0.8, label="height rank", color="#2563eb"
    )
    continuous_molecule_axes[0, 0].plot(
        molecule_ids, molecules["S_Probe_Access_AUC_Mean"], "o-", markersize=3,
        linewidth=0.8, label="probe AUC", color="#db2777"
    )
    continuous_molecule_axes[0, 0].set(
        xlabel="SAM molecule index", ylabel="Exposure score", ylim=(0.0, 1.0),
        title="Continuous sulfur exposure"
    )
    continuous_molecule_axes[0, 0].legend()
    continuous_molecule_axes[0, 1].bar(
        molecule_ids, molecules["S_Soft_Shielding_Mean"], color="#7c3aed"
    )
    continuous_molecule_axes[0, 1].set(
        xlabel="SAM molecule index", ylabel="Shielding exposure",
        title="Continuous local shielding"
    )
    continuous_molecule_axes[1, 0].errorbar(
        molecule_ids,
        molecules["SAM_Soft_Top_Thickness_Mean_A"],
        yerr=molecules["SAM_Soft_Top_Thickness_Std_A"],
        fmt="o", markersize=3, linewidth=0.8, color="#059669"
    )
    continuous_molecule_axes[1, 0].set(
        xlabel="SAM molecule index", ylabel="Thickness (A)",
        title="Per-molecule soft top thickness"
    )
    radius_values = np.asarray(sorted(set(args.probe_radii)), dtype=float)
    radius_means = np.asarray(
        [np.mean(raw[probe_column(radius)]) for radius in radius_values]
    )
    radius_stds = np.asarray(
        [np.std(raw[probe_column(radius)]) for radius in radius_values]
    )
    continuous_molecule_axes[1, 1].errorbar(
        radius_values, radius_means, yerr=radius_stds, fmt="o-", capsize=3,
        color="#db2777"
    )
    continuous_molecule_axes[1, 1].set(
        xlabel="Probe radius (A)", ylabel="Accessible upper solid-angle fraction",
        ylim=(0.0, 0.75), title="Probe-size accessibility curve"
    )
    for axis in continuous_molecule_axes.ravel():
        axis.grid(alpha=0.25, linestyle="--")
    continuous_molecule_figure.suptitle(
        f"{args.prefix}: per-molecule continuous exposure and thickness"
    )
    continuous_molecule_figure.tight_layout()
    continuous_molecule_output = args.output_dir / f"{args.prefix}_last_{args.last_ps:g}ps_continuous_per_molecule.png"
    continuous_molecule_figure.savefig(continuous_molecule_output, dpi=240)
    plt.close(continuous_molecule_figure)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traj", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="sam")
    parser.add_argument(
        "--dipole-only",
        action="store_true",
        help="Run only rigid-reference molecular dipole mapping and statistics",
    )
    parser.add_argument(
        "--dipole-reference",
        type=Path,
        help="JSON containing explicit QM reference coordinates, dipole, and SAM mapping",
    )
    parser.add_argument(
        "--dipole-system",
        help="Explicit system key selecting one record from --dipole-reference",
    )
    parser.add_argument("--substrate-atoms", type=int, required=True)
    parser.add_argument("--molecules", type=int, required=True)
    parser.add_argument("--atoms-per-molecule", type=int, required=True)
    parser.add_argument("--species", default=" ".join(DEFAULT_SPECIES))
    parser.add_argument("--timestep-fs", type=float, default=2.0)
    parser.add_argument("--last-ps", type=float, default=2.0)
    parser.add_argument("--anchor-element", default="P")
    parser.add_argument("--linker-end-element", default="N")
    parser.add_argument("--marker-element", default="S")
    parser.add_argument("--conjugated-elements", nargs="+", default=["C", "N", "S"])
    parser.add_argument("--surface-element", default="O")
    parser.add_argument("--surface-window", type=float, default=1.0)
    parser.add_argument("--exposure-radius", type=float, default=3.5)
    parser.add_argument("--blocker-vertical-tolerance", type=float, default=0.20)
    parser.add_argument("--exposed-depth-threshold", type=float, default=0.50)
    parser.add_argument("--exposure-lateral-scale", type=float, default=3.5)
    parser.add_argument("--exposure-height-smoothing", type=float, default=0.40)
    parser.add_argument("--probe-radii", type=float, nargs="+", default=[1.0, 1.5, 2.0, 2.5, 3.0])
    parser.add_argument("--probe-direction-samples", type=int, default=256)
    parser.add_argument("--probe-clearance-smoothing", type=float, default=0.25)
    parser.add_argument("--thickness-softness", type=float, default=0.75)
    parser.add_argument(
        "--thickness-softness-scan", type=float, nargs="+", default=[0.5, 0.75, 1.0, 1.5]
    )
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--dipole-arrow-scale",
        type=float,
        default=3.0,
        help="Side-view arrow length in Angstrom per Debye (default: 3.0)",
    )
    parser.add_argument(
        "--dipole-target-normal",
        type=float,
        help=(
            "Uniformly rescale all mapped dipoles so their overall mean outward-normal "
            "component equals this signed Debye value"
        ),
    )
    parser.add_argument(
        "--dipole-scaling-mode",
        choices=("uniform", "sign-balanced"),
        default="uniform",
        help=(
            "Calibration rule for --dipole-target-normal: one global factor, or equal-"
            "percentage extension of negative vectors and contraction of positive vectors"
        ),
    )
    args = parser.parse_args()
    args.traj = args.traj.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.dipole_reference is not None:
        args.dipole_reference = args.dipole_reference.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dipole_arrow_scale <= 0.0:
        parser.error("--dipole-arrow-scale must be positive")
    if args.dipole_only:
        if args.dipole_reference is None or not args.dipole_system:
            parser.error("--dipole-only requires --dipole-reference and --dipole-system")
        summary = analyze_dipoles(args)
    else:
        if args.dipole_reference is not None or args.dipole_system:
            parser.error("Dipole reference options require --dipole-only")
        summary = analyze(args)
        if not args.no_plot:
            plot_results(args, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
