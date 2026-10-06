#!/usr/bin/env python3
"""Rebuild a chemically complete SAM by fitting conformers to target headgroups.

This implementation never slices atoms by their file position.  It recovers
the substrate and all target anchor headgroups from a validated adsorption
template using configurable covalent topology, aligns selected standing/marker-exposed
conformers to those headgroups, and rejects every cross-component collision.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import re
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from sam_structure_tools import (
    MoleculeSpec,
    molecular_components,
    component_metrics,
    recover_substrate_and_headgroups,
    rigid_transform,
    unwrap_component,
)
from sam_lammps import read_typed_structure


def prepare_template(path: Path, repeat: tuple[int, int, int], orthogonalize: bool):
    """Load, optionally repeat, and optionally orthogonalize a template."""

    template = read_typed_structure(path)
    if repeat != (1, 1, 1):
        template = template * repeat
    if orthogonalize:
        doubled_cell = np.asarray(template.cell)
        orthogonal_cell = np.asarray(
            [doubled_cell[0], doubled_cell[0] + doubled_cell[1], doubled_cell[2]]
        )
        off_diagonal = orthogonal_cell.copy()
        np.fill_diagonal(off_diagonal, 0.0)
        if np.max(np.abs(off_diagonal)) > 1.0e-5:
            raise ValueError(
                f"Template does not yield an orthogonal cell: {orthogonal_cell}"
            )
        template.set_cell(orthogonal_cell, scale_atoms=False)
        template.wrap()
    return template


def component_headgroup_coordinates(atoms, component, oxygen_order=None):
    coordinates, _ = unwrap_component(atoms, component)
    local = {global_index: position for position, global_index in enumerate(component.indices)}
    oxygen_order = oxygen_order or component.o_indices
    ordered = (component.p_index,) + tuple(oxygen_order)
    return coordinates[[local[index] for index in ordered]]


def collision_report(
    new_coordinates: np.ndarray,
    new_symbols: np.ndarray,
    existing_coordinates: np.ndarray,
    existing_symbols: np.ndarray,
    cell: np.ndarray,
) -> tuple[int, float, float]:
    if len(existing_coordinates) == 0:
        return 0, float("inf"), float("inf")
    delta = new_coordinates[:, None, :] - existing_coordinates[None, :, :]
    inv_cell = np.linalg.inv(cell)
    fractional = delta @ inv_cell
    fractional -= np.round(fractional)
    distances = np.linalg.norm(fractional @ cell, axis=2)

    new_h = new_symbols[:, None] == "H"
    old_h = existing_symbols[None, :] == "H"
    thresholds = np.where(new_h & old_h, 1.20, np.where(new_h | old_h, 1.40, 1.80))
    collisions = distances < thresholds
    count = int(np.count_nonzero(collisions))
    minimum = float(np.min(distances))
    minimum_ratio = float(np.min(distances / thresholds))
    return count, minimum, minimum_ratio


def collision_details(
    new_coordinates,
    new_symbols,
    existing_coordinates,
    existing_symbols,
    cell,
    limit=12,
):
    delta = new_coordinates[:, None, :] - existing_coordinates[None, :, :]
    inv_cell = np.linalg.inv(cell)
    fractional = delta @ inv_cell
    fractional -= np.round(fractional)
    distances = np.linalg.norm(fractional @ cell, axis=2)
    new_h = new_symbols[:, None] == "H"
    old_h = existing_symbols[None, :] == "H"
    thresholds = np.where(new_h & old_h, 1.20, np.where(new_h | old_h, 1.40, 1.80))
    rows, columns = np.nonzero(distances < thresholds)
    details = []
    for row, column in zip(rows, columns):
        details.append(
            {
                "new_local": int(row),
                "new_element": str(new_symbols[row]),
                "existing_local": int(column),
                "existing_element": str(existing_symbols[column]),
                "distance_A": round(float(distances[row, column]), 6),
                "threshold_A": float(thresholds[row, column]),
            }
        )
    details.sort(key=lambda item: item["distance_A"])
    return details[:limit]


def solve_min_conflicts(domains, target_atoms, target_components, cell, rng, restarts=24, iterations=2500):
    anchor_positions = np.asarray(
        [target_atoms.positions[component.p_index] for component in target_components]
    )
    inv_cell = np.linalg.inv(cell)

    # Conservative neighbor graph based on the largest atom-to-P extent at
    # each site.  Pairs outside this bound cannot collide under any assignment.
    radii = []
    for site, domain in enumerate(domains):
        anchor = anchor_positions[site]
        maximum = 0.0
        for candidate in domain:
            vectors = candidate["coordinates"] - anchor
            fractional = vectors @ inv_cell
            fractional -= np.round(fractional)
            maximum = max(maximum, float(np.max(np.linalg.norm(fractional @ cell, axis=1))))
        radii.append(maximum)
    edges = []
    neighbors = [set() for _ in domains]
    for i in range(len(domains)):
        for j in range(i + 1, len(domains)):
            vector = anchor_positions[j] - anchor_positions[i]
            fractional = vector @ inv_cell
            fractional -= np.round(fractional)
            anchor_distance = float(np.linalg.norm(fractional @ cell))
            if anchor_distance < radii[i] + radii[j] + 1.80:
                edges.append((i, j))
                neighbors[i].add(j)
                neighbors[j].add(i)

    def pair_collision(site_a, choice_a, site_b, choice_b):
        candidate_a = domains[site_a][choice_a]
        candidate_b = domains[site_b][choice_b]
        count, minimum, ratio = collision_report(
            candidate_a["coordinates"],
            candidate_a["symbols"],
            candidate_b["coordinates"],
            candidate_b["symbols"],
            cell,
        )
        return count, minimum, ratio

    print(f"Global conflict graph: {len(edges)} potentially interacting site pairs", flush=True)
    best_state = {"conflict_pairs": len(edges) + 1, "restart": None, "iteration": None}
    for restart in range(1, restarts + 1):
        assignment = [rng.randrange(len(domain)) for domain in domains]
        for iteration in range(1, iterations + 1):
            site_conflicts = [0] * len(domains)
            conflict_pairs = 0
            for i, j in edges:
                count, _, _ = pair_collision(i, assignment[i], j, assignment[j])
                if count:
                    conflict_pairs += 1
                    site_conflicts[i] += 1
                    site_conflicts[j] += 1
            if conflict_pairs < best_state["conflict_pairs"]:
                best_state = {
                    "conflict_pairs": conflict_pairs,
                    "restart": restart,
                    "iteration": iteration,
                }
                print(f"Min-conflicts best: {best_state}", flush=True)
            if conflict_pairs == 0:
                return [domains[site][choice] for site, choice in enumerate(assignment)], best_state

            maximum_conflicts = max(site_conflicts)
            worst_sites = [
                site for site, count in enumerate(site_conflicts) if count == maximum_conflicts
            ]
            site = rng.choice(worst_sites)
            scored = []
            for choice in range(len(domains[site])):
                conflicting_neighbors = 0
                atom_collisions = 0
                minimum_ratio = float("inf")
                for neighbor in neighbors[site]:
                    count, _, ratio = pair_collision(
                        site, choice, neighbor, assignment[neighbor]
                    )
                    if count:
                        conflicting_neighbors += 1
                        atom_collisions += count
                    minimum_ratio = min(minimum_ratio, ratio)
                scored.append(
                    (conflicting_neighbors, atom_collisions, -minimum_ratio, choice)
                )
            best_score = min(score[:3] for score in scored)
            best_choices = [
                choice
                for conflicts, atoms_count, negative_ratio, choice in scored
                if (conflicts, atoms_count, negative_ratio) == best_score
            ]
            assignment[site] = rng.choice(best_choices)
    raise RuntimeError(f"Min-conflicts did not find a zero-conflict assignment; best={best_state}")


def solve_lazy_milp(domains, target_atoms, target_components, cell, seed, max_rounds=500):
    anchor_positions = np.asarray(
        [target_atoms.positions[component.p_index] for component in target_components]
    )
    inv_cell = np.linalg.inv(cell)
    radii = []
    for site, domain in enumerate(domains):
        anchor = anchor_positions[site]
        maximum = 0.0
        for candidate in domain:
            vectors = candidate["coordinates"] - anchor
            fractional = vectors @ inv_cell
            fractional -= np.round(fractional)
            maximum = max(maximum, float(np.max(np.linalg.norm(fractional @ cell, axis=1))))
        radii.append(maximum)
    edges = []
    for i in range(len(domains)):
        for j in range(i + 1, len(domains)):
            vector = anchor_positions[j] - anchor_positions[i]
            fractional = vector @ inv_cell
            fractional -= np.round(fractional)
            anchor_distance = float(np.linalg.norm(fractional @ cell))
            if anchor_distance < radii[i] + radii[j] + 1.80:
                edges.append((i, j))

    offsets = np.cumsum([0] + [len(domain) for domain in domains])
    variable_count = int(offsets[-1])
    rng = np.random.default_rng(seed)
    objective = np.empty(variable_count, dtype=float)
    for site, domain in enumerate(domains):
        for choice, candidate in enumerate(domain):
            variable = int(offsets[site] + choice)
            objective[variable] = (
                candidate["headgroup_rmsd_A"]
                - 0.01 * candidate["substrate_clearance_ratio"]
                + float(rng.uniform(0.0, 1.0e-5))
            )

    forbidden_pairs: set[tuple[int, int]] = set()
    print(
        f"Lazy MILP: {variable_count} binary variables, {len(edges)} potential site pairs",
        flush=True,
    )
    for round_number in range(1, max_rounds + 1):
        row_indices = []
        column_indices = []
        values = []
        lower = []
        upper = []
        row = 0
        for site, domain in enumerate(domains):
            for choice in range(len(domain)):
                row_indices.append(row)
                column_indices.append(int(offsets[site] + choice))
                values.append(1.0)
            lower.append(1.0)
            upper.append(1.0)
            row += 1
        for variable_a, variable_b in sorted(forbidden_pairs):
            row_indices.extend((row, row))
            column_indices.extend((variable_a, variable_b))
            values.extend((1.0, 1.0))
            lower.append(-np.inf)
            upper.append(1.0)
            row += 1
        matrix = coo_matrix(
            (values, (row_indices, column_indices)),
            shape=(row, variable_count),
        ).tocsr()
        result = milp(
            c=objective,
            integrality=np.ones(variable_count, dtype=int),
            bounds=Bounds(np.zeros(variable_count), np.ones(variable_count)),
            constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
            options={"time_limit": 60.0, "mip_rel_gap": 0.0},
        )
        if not result.success or result.x is None:
            raise RuntimeError(
                f"Lazy MILP failed at round {round_number}: status={result.status}, "
                f"message={result.message}, forbidden_pairs={len(forbidden_pairs)}"
            )
        assignment = []
        for site, domain in enumerate(domains):
            segment = result.x[offsets[site] : offsets[site + 1]]
            assignment.append(int(np.argmax(segment)))

        new_forbidden = set()
        colliding_site_pair_count = 0
        atom_collision_count = 0
        for site_a, site_b in edges:
            choice_a = assignment[site_a]
            choice_b = assignment[site_b]
            candidate_a = domains[site_a][choice_a]
            candidate_b = domains[site_b][choice_b]
            count, _, _ = collision_report(
                candidate_a["coordinates"],
                candidate_a["symbols"],
                candidate_b["coordinates"],
                candidate_b["symbols"],
                cell,
            )
            if count:
                colliding_site_pair_count += 1
                atom_collision_count += count
                variable_a = int(offsets[site_a] + choice_a)
                variable_b = int(offsets[site_b] + choice_b)
                new_forbidden.add(tuple(sorted((variable_a, variable_b))))

                # Learn a full row and column of the local incompatibility
                # table.  Adding only the currently selected pair makes the
                # lazy MILP rediscover nearly identical clashes for hundreds
                # of rounds on this dense monolayer.
                for other_choice_b, other_b in enumerate(domains[site_b]):
                    other_count, _, _ = collision_report(
                        candidate_a["coordinates"],
                        candidate_a["symbols"],
                        other_b["coordinates"],
                        other_b["symbols"],
                        cell,
                    )
                    if other_count:
                        other_variable_b = int(offsets[site_b] + other_choice_b)
                        new_forbidden.add(
                            tuple(sorted((variable_a, other_variable_b)))
                        )
                for other_choice_a, other_a in enumerate(domains[site_a]):
                    other_count, _, _ = collision_report(
                        other_a["coordinates"],
                        other_a["symbols"],
                        candidate_b["coordinates"],
                        candidate_b["symbols"],
                        cell,
                    )
                    if other_count:
                        other_variable_a = int(offsets[site_a] + other_choice_a)
                        new_forbidden.add(
                            tuple(sorted((other_variable_a, variable_b)))
                        )
        if not new_forbidden:
            print(
                f"Lazy MILP converged at round {round_number} with zero collisions",
                flush=True,
            )
            return [
                domains[site][choice] for site, choice in enumerate(assignment)
            ], {
                "method": "lazy-milp",
                "rounds": round_number,
                "forbidden_pairs": len(forbidden_pairs),
            }
        previous_count = len(forbidden_pairs)
        forbidden_pairs.update(new_forbidden)
        print(
            f"Lazy MILP round {round_number}: "
            f"colliding_site_pairs={colliding_site_pair_count}, "
            f"atom_collisions={atom_collision_count}, "
            f"new_constraints={len(forbidden_pairs) - previous_count}, "
            f"total_constraints={len(forbidden_pairs)}",
            flush=True,
        )
    raise RuntimeError(
        f"Lazy MILP exceeded {max_rounds} rounds with {len(forbidden_pairs)} constraints"
    )


def _natural_path_key(path: Path):
    """Sort numbered conformer files by numeric rank, not lexicographically."""

    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.lower())
        for part in re.split(r"(\d+)", path.name)
    )


def selected_conformer_paths(
    root: Path, conformer_glob: str, maximum_source_clusters: int | None
) -> tuple[list[Path], list[Path]]:
    """Return all matches and the selected energy-ranked source prefix."""

    matched = sorted(root.glob(conformer_glob), key=_natural_path_key)
    if maximum_source_clusters is not None:
        if maximum_source_clusters <= 0:
            raise ValueError("maximum_source_clusters must be positive")
        selected = matched[:maximum_source_clusters]
    else:
        selected = matched
    if not selected:
        raise ValueError(f"No molecular conformers matched {conformer_glob!r}")
    return matched, selected


def _fsync_cache_directory(path: Path) -> None:
    """Durably persist cache link/unlink metadata on Linux."""

    directory_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _install_cache_file_noclobber(destination: Path, writer) -> dict:
    """Install one generated cache file through a same-directory hard link."""

    destination = Path(destination)
    parent = destination.parent
    if parent.is_symlink() or not parent.is_dir():
        raise FileNotFoundError(
            f"Immutable cache parent must be a real directory: {parent}"
        )
    fd, temporary_name = tempfile.mkstemp(
        dir=parent, prefix=f".{destination.name}-", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w+b") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, destination)
        temporary_path.unlink()
        _fsync_cache_directory(parent)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return {
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "installation": "same_directory_hard_link_noreplace",
    }


def source_conformers(
    root: Path,
    conformer_glob: str,
    cache_prefix: Path,
    spec: MoleculeSpec,
    minimum_vertical_angle: float,
    maximum_marker_exposure: float,
    maximum_source_clusters: int | None = None,
    immutable_new_cache: bool = False,
) -> list[dict]:
    cache_npz = Path(f"{cache_prefix}.npz")
    cache_json = Path(f"{cache_prefix}.json")
    if immutable_new_cache:
        existing = [
            str(path)
            for path in (cache_npz, cache_json)
            if os.path.lexists(os.fspath(path))
        ]
        if existing:
            raise FileExistsError(
                "Immutable conformer cache artifacts must be absent: "
                + ", ".join(existing)
            )
    _, source_paths = selected_conformer_paths(
        root, conformer_glob, maximum_source_clusters
    )
    source_records = [
        {
            "path": str(path),
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in source_paths
    ]
    all_conformers = []
    cache_is_valid = False
    metadata_cache = None
    if (
        not immutable_new_cache
        and cache_npz.exists()
        and cache_json.exists()
    ):
        metadata_payload = json.loads(cache_json.read_text())
        cache_is_valid = (
            isinstance(metadata_payload, dict)
            and metadata_payload.get("schema_version") == 2
            and metadata_payload.get("source_records") == source_records
            and metadata_payload.get("maximum_source_clusters")
            == maximum_source_clusters
        )
        if cache_is_valid:
            metadata_cache = metadata_payload["items"]

    if cache_is_valid:
        cached = np.load(cache_npz, allow_pickle=False)
        for position, item in enumerate(metadata_cache):
            all_conformers.append(
                {
                    **item,
                    "coordinates": cached["coordinates"][position],
                    "symbols": cached["symbols"].astype(str),
                }
            )
    else:
        coordinate_cache = []
        metadata_cache = []
        cached_symbols = None
        for source_rank, path in enumerate(source_paths, 1):
            number_matches = re.findall(r"\d+", path.stem)
            cluster_number = int(number_matches[-1]) if number_matches else source_rank
            atoms = read_typed_structure(path)
            components = molecular_components(atoms, spec)
            for molecule_number, component in enumerate(components, 1):
                metrics = component_metrics(atoms, component, spec)
                coordinates, symbols = unwrap_component(
                    atoms, component, spec.hydrogen_parent_elements
                )
                local = {
                    global_index: local_index
                    for local_index, global_index in enumerate(component.indices)
                }
                item = {
                    "source": str(path),
                    "cluster": cluster_number,
                    "source_rank": source_rank,
                    "molecule": molecule_number,
                    "metrics": metrics,
                    "p_local": local[component.p_index],
                    "o_locals": [local[index] for index in component.o_indices],
                }
                all_conformers.append(
                    {
                        **item,
                        "coordinates": coordinates,
                        "symbols": np.asarray(symbols),
                    }
                )
                coordinate_cache.append(coordinates)
                metadata_cache.append(item)
                if cached_symbols is None:
                    cached_symbols = np.asarray(symbols)
        cache_document = {
            "schema_version": 2,
            "maximum_source_clusters": maximum_source_clusters,
            "source_records": source_records,
            "items": metadata_cache,
        }
        if immutable_new_cache:
            cache_parent = cache_npz.parent
            if os.path.lexists(os.fspath(cache_parent)):
                if cache_parent.is_symlink() or not cache_parent.is_dir():
                    raise FileExistsError(
                        "Immutable conformer cache parent is not a real directory: "
                        f"{cache_parent}"
                    )
            else:
                cache_parent.mkdir(parents=True, exist_ok=False)
                _fsync_cache_directory(cache_parent.parent)
            _install_cache_file_noclobber(
                cache_npz,
                lambda handle: np.savez_compressed(
                    handle,
                    coordinates=np.stack(coordinate_cache),
                    symbols=cached_symbols,
                ),
            )
            try:
                rendered_metadata = (
                    json.dumps(
                        cache_document,
                        indent=2,
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n"
                ).encode("utf-8")
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    "Conformer cache metadata is not strict JSON serializable"
                ) from exc
            _install_cache_file_noclobber(
                cache_json, lambda handle: handle.write(rendered_metadata)
            )
        else:
            cache_npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_npz,
                coordinates=np.stack(coordinate_cache),
                symbols=cached_symbols,
            )
            cache_json.write_text(json.dumps(cache_document, indent=2))

    # Surface-relative orientation changes during the P/O headgroup fit.  Do
    # not filter on the source slab's axes here; transformed_candidates applies
    # these requirements after each conformer is anchored to the target site.
    if not all_conformers:
        raise ValueError("No conformers were found")
    return all_conformers


def coordinate_metrics(coordinates, symbols, anchor_local, spec: MoleculeSpec):
    """Surface-relative orientation metrics after target anchoring."""
    symbols = np.asarray(symbols)
    carbon_locals = np.flatnonzero(symbols == spec.chain_element)
    if len(carbon_locals) < spec.chain_length:
        raise ValueError(
            f"Need {spec.chain_length} {spec.chain_element} chain atoms; "
            f"found {len(carbon_locals)}"
        )
    distances_pc = np.linalg.norm(
        coordinates[carbon_locals] - coordinates[anchor_local], axis=1
    )
    chain = [int(carbon_locals[int(np.argmin(distances_pc))])]
    for _ in range(spec.chain_length - 1):
        remaining = [index for index in carbon_locals if int(index) not in chain]
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
    centered = coordinates[core] - np.mean(coordinates[core], axis=0)
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


def transformed_candidates(
    target_atoms,
    target_component,
    conformers,
    spec,
    minimum_vertical_angle,
    maximum_marker_exposure,
    permutations_per_source=2,
):
    target_headgroup = component_headgroup_coordinates(target_atoms, target_component)
    candidates = []
    for conformer_number, conformer in enumerate(conformers):
        source_coordinates = conformer["coordinates"]
        source_options = []
        for permutation in itertools.permutations(conformer["o_locals"]):
            source_order = (conformer["p_local"],) + tuple(permutation)
            source_headgroup = source_coordinates[list(source_order)]
            rotation, translation, rmsd = rigid_transform(
                source_headgroup, target_headgroup
            )
            transformed = source_coordinates @ rotation + translation
            organic_heavy = np.isin(conformer["symbols"], spec.core_elements)
            organic_centroid_above_p = float(
                np.mean(transformed[organic_heavy, 2])
                - transformed[conformer["p_local"], 2]
            )
            if organic_centroid_above_p < 2.0:
                continue
            transformed_metrics = coordinate_metrics(
                transformed, conformer["symbols"], conformer["p_local"], spec
            )
            marker_exposure = transformed_metrics.get("marker_exposure_A", 0.0)
            if (
                transformed_metrics["vertical_plane_angle_deg"]
                < minimum_vertical_angle
                or marker_exposure > maximum_marker_exposure
            ):
                continue
            source_options.append(
                {
                    "conformer_number": conformer_number,
                    "oxygen_permutation": [int(index) for index in permutation],
                    "coordinates": transformed,
                    "symbols": conformer["symbols"],
                    "headgroup_rmsd_A": rmsd,
                    "organic_centroid_above_P_A": organic_centroid_above_p,
                    "transformed_metrics": transformed_metrics,
                }
            )
        source_options.sort(key=lambda candidate: candidate["headgroup_rmsd_A"])
        candidates.extend(source_options[:permutations_per_source])
    # Grossly mismatched headgroups are never acceptable even if collision-free.
    candidates = [candidate for candidate in candidates if candidate["headgroup_rmsd_A"] <= 0.35]
    if not candidates:
        raise ValueError(
            f"No source headgroup fits target anchor atom {target_component.p_index} "
            "below 0.35 A RMSD"
        )
    return candidates


def assemble(
    template,
    target_components,
    substrate_indices,
    conformers,
    spec,
    seed,
    attempts,
    minimum_vertical_angle,
    maximum_marker_exposure,
    domain_cache_prefix: Path | None = None,
):
    cell = np.asarray(template.cell)
    substrate = template[substrate_indices].copy()
    substrate.set_cell(template.cell)
    substrate.set_pbc(template.pbc)
    base_coordinates = substrate.positions.copy()
    base_symbols = np.asarray(substrate.get_chemical_symbols())

    cache_npz = Path(f"{domain_cache_prefix}.npz") if domain_cache_prefix else None
    cache_json = Path(f"{domain_cache_prefix}.json") if domain_cache_prefix else None
    substrate_filtered = []
    if cache_npz and cache_json and cache_npz.exists() and cache_json.exists():
        cached = np.load(cache_npz, allow_pickle=False)
        metadata = json.loads(cache_json.read_text())
        offsets = cached["offsets"]
        for site in range(len(offsets) - 1):
            domain = []
            for flat_index in range(int(offsets[site]), int(offsets[site + 1])):
                domain.append(
                    {
                        **metadata[flat_index],
                        "coordinates": cached["coordinates"][flat_index],
                        "symbols": cached["symbols"][flat_index].astype(str),
                    }
                )
            substrate_filtered.append(domain)
        print(f"Loaded cached substrate-compatible domains from {cache_npz}", flush=True)
    else:
        precomputed = [
            transformed_candidates(
                template,
                component,
                conformers,
                spec,
                minimum_vertical_angle,
                maximum_marker_exposure,
            )
            for component in target_components
        ]
        # Filter fixed substrate collisions exactly once per site/candidate.
        for site, domain in enumerate(precomputed):
            accepted = []
            rejected = []
            for candidate in domain:
                count, minimum, ratio = collision_report(
                    candidate["coordinates"],
                    candidate["symbols"],
                    base_coordinates,
                    base_symbols,
                    cell,
                )
                if count == 0:
                    candidate = dict(candidate)
                    candidate["substrate_minimum_distance_A"] = minimum
                    candidate["substrate_clearance_ratio"] = ratio
                    accepted.append(candidate)
                else:
                    rejected.append((count, candidate))
            if not accepted:
                rejected.sort(key=lambda item: item[0])
                best_count, best_candidate = rejected[0]
                details = collision_details(
                    best_candidate["coordinates"],
                    best_candidate["symbols"],
                    base_coordinates,
                    base_symbols,
                    cell,
                )
                raise RuntimeError(
                    f"Site {site + 1} has no substrate-compatible conformer; "
                    f"best_collision_count={best_count}; details={details}"
                )
            substrate_filtered.append(accepted)
            print(f"Site {site + 1:02d}: {len(accepted)}/{len(domain)} substrate-compatible candidates", flush=True)
        if cache_npz and cache_json:
            cache_npz.parent.mkdir(parents=True, exist_ok=True)
            flat = [candidate for domain in substrate_filtered for candidate in domain]
            offsets = np.cumsum([0] + [len(domain) for domain in substrate_filtered])
            np.savez_compressed(
                cache_npz,
                coordinates=np.stack([candidate["coordinates"] for candidate in flat]),
                symbols=np.stack([candidate["symbols"] for candidate in flat]),
                offsets=offsets,
            )
            cache_json.write_text(
                json.dumps(
                    [
                        {
                            key: value
                            for key, value in candidate.items()
                            if key not in ("coordinates", "symbols")
                        }
                        for candidate in flat
                    ],
                    indent=2,
                )
            )
            print(f"Saved substrate-compatible domain cache to {cache_npz}", flush=True)
    master_rng = random.Random(seed)
    best_partial = {"placed": 0, "attempt": None, "failed_site": None, "best_rejected_collisions": None}

    for attempt in range(1, attempts + 1):
        order = list(range(len(target_components)))
        master_rng.shuffle(order)
        order.sort(key=lambda site: len(substrate_filtered[site]))
        placed = []
        existing_coordinates = np.empty((0, 3), dtype=float)
        existing_symbols = np.empty((0,), dtype=str)
        failed_site = None

        for site in order:
            pool = list(substrate_filtered[site])
            master_rng.shuffle(pool)
            feasible = []
            rejected_counts = []
            for candidate in pool:
                count, minimum, ratio = collision_report(
                    candidate["coordinates"],
                    candidate["symbols"],
                    existing_coordinates,
                    existing_symbols,
                    cell,
                )
                if count == 0:
                    feasible.append((ratio, minimum, candidate))
                else:
                    rejected_counts.append(count)
            if not feasible:
                failed_site = site
                if len(placed) > best_partial["placed"]:
                    best_partial = {
                        "placed": len(placed),
                        "attempt": attempt,
                        "failed_site": site + 1,
                        "best_rejected_collisions": min(rejected_counts) if rejected_counts else None,
                    }
                break
            # Prefer the candidate with the largest normalized clearance;
            # headgroup RMSD breaks nearly equal packing scores.
            feasible.sort(
                key=lambda item: (
                    item[0],
                    item[1],
                    -item[2]["headgroup_rmsd_A"],
                ),
                reverse=True,
            )
            ratio, minimum, chosen = feasible[0]
            chosen = dict(chosen)
            chosen["site"] = site
            chosen["minimum_cross_distance_A"] = minimum
            chosen["minimum_clearance_ratio"] = ratio
            placed.append(chosen)
            existing_coordinates = np.vstack((existing_coordinates, chosen["coordinates"]))
            existing_symbols = np.concatenate((existing_symbols, chosen["symbols"]))

        if failed_site is None and len(placed) == len(target_components):
            placed.sort(key=lambda item: item["site"])
            assembled = substrate.copy()
            for item in placed:
                molecule = Atoms(
                    symbols=item["symbols"].tolist(),
                    positions=item["coordinates"],
                    cell=template.cell,
                    pbc=template.pbc,
                )
                assembled.extend(molecule)
            assembled.set_cell(template.cell)
            assembled.set_pbc(template.pbc)
            return assembled, placed, attempt
    print(
        f"Greedy assembly exhausted; best_partial={best_partial}. "
        "Switching to full min-conflicts assignment.",
        flush=True,
    )
    chosen, solver_state = solve_lazy_milp(
        substrate_filtered,
        template,
        target_components,
        cell,
        seed,
    )
    placed = []
    for site, candidate_raw in enumerate(chosen):
        candidate = dict(candidate_raw)
        other_coordinates = [base_coordinates]
        other_symbols = [base_symbols]
        for other_site, other in enumerate(chosen):
            if other_site != site:
                other_coordinates.append(other["coordinates"])
                other_symbols.append(other["symbols"])
        count, minimum, ratio = collision_report(
            candidate["coordinates"],
            candidate["symbols"],
            np.vstack(other_coordinates),
            np.concatenate(other_symbols),
            cell,
        )
        if count:
            raise AssertionError(
                f"Min-conflicts returned a colliding solution at site {site + 1}"
            )
        candidate["site"] = site
        candidate["minimum_cross_distance_A"] = minimum
        candidate["minimum_clearance_ratio"] = ratio
        placed.append(candidate)

    assembled = substrate.copy()
    for item in placed:
        molecule = Atoms(
            symbols=item["symbols"].tolist(),
            positions=item["coordinates"],
            cell=template.cell,
            pbc=template.pbc,
        )
        assembled.extend(molecule)
    assembled.set_cell(template.cell)
    assembled.set_pbc(template.pbc)
    return assembled, placed, solver_state


def _json_formula(value: str | None) -> dict[str, int] | None:
    if value is None:
        return None
    result = {str(key): int(count) for key, count in json.loads(value).items()}
    if any(count < 0 for count in result.values()):
        raise argparse.ArgumentTypeError("Formula counts must be non-negative")
    return result


def _resolved(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def headgroup_fit_main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a SAM by fitting complete conformers to template headgroups."
    )
    parser.add_argument("--root", type=Path, default=Path("."), help="Project root")
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--conformer-glob", required=True)
    parser.add_argument(
        "--max-source-clusters",
        type=int,
        help=(
            "Use only the first N naturally numbered conformer files. This is "
            "appropriate when the cluster suffix is an energy rank."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--conformer-cache", type=Path, default=Path(".cache/sam_conformers"))
    parser.add_argument("--domain-cache", type=Path, default=Path(".cache/sam_domains"))
    parser.add_argument("--template-repeat", nargs=3, type=int, default=(1, 1, 1))
    parser.add_argument("--orthogonalize", action="store_true")
    parser.add_argument("--molecule-formula-json", required=True)
    parser.add_argument("--skeleton-formula-json")
    parser.add_argument("--anchor-element", default="P")
    parser.add_argument("--headgroup-element", default="O")
    parser.add_argument("--headgroup-count", type=int, default=3)
    parser.add_argument("--headgroup-cutoff", type=float, default=1.85)
    parser.add_argument("--hydrogen-parent-elements", nargs="+", default=["C", "N"])
    parser.add_argument("--chain-element", default="C")
    parser.add_argument("--chain-length", type=int, default=4)
    parser.add_argument("--core-elements", nargs="+", default=["C", "N", "S"])
    parser.add_argument("--marker-element", default="S")
    parser.add_argument("--substrate-elements", nargs="+", required=True)
    parser.add_argument("--surface-h-per-molecule", type=int, default=0)
    parser.add_argument("--surface-parent-element", default="O")
    parser.add_argument("--min-vertical-angle", type=float, default=40.0)
    parser.add_argument("--max-marker-exposure", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=340021)
    parser.add_argument("--attempts", type=int, default=30)
    parser.add_argument(
        "--clean-substrate",
        action="store_true",
        help="Remove recovered substrate H before placing all SAM molecules.",
    )
    parser.add_argument(
        "--allow-incomplete-template-skeleton",
        action="store_true",
        help=(
            "Recover only unique P/O adsorption targets from a damaged old SAM "
            "layer. The old organic skeleton is discarded and mismatches are "
            "recorded in the output manifest."
        ),
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()
    template_path = _resolved(root, args.template)
    output_path = _resolved(root, args.output)
    manifest_path = (
        _resolved(root, args.manifest)
        if args.manifest
        else output_path.with_suffix(".manifest.json")
    )
    spec = MoleculeSpec(
        formula=_json_formula(args.molecule_formula_json) or {},
        skeleton_formula=_json_formula(args.skeleton_formula_json),
        anchor_element=args.anchor_element,
        headgroup_element=args.headgroup_element,
        headgroup_count=args.headgroup_count,
        headgroup_cutoff=args.headgroup_cutoff,
        hydrogen_parent_elements=tuple(args.hydrogen_parent_elements),
        chain_element=args.chain_element,
        chain_length=args.chain_length,
        core_elements=tuple(args.core_elements),
        marker_element=(None if args.marker_element.lower() == "none" else args.marker_element),
    )

    template = prepare_template(
        template_path, tuple(args.template_repeat), args.orthogonalize
    )
    substrate_indices, target_components, surface_h_repairs = (
        recover_substrate_and_headgroups(
            template,
            spec,
            tuple(args.substrate_elements),
            args.surface_h_per_molecule,
            args.surface_parent_element,
            require_complete_skeleton=not args.allow_incomplete_template_skeleton,
        )
    )
    if not target_components:
        raise ValueError("Template contains no recoverable anchor/headgroup targets")
    template_symbols = np.asarray(template.get_chemical_symbols())
    substrate_formula = dict(
        sorted(Counter(template_symbols[substrate_indices]).items())
    )
    inherited_surface_h_count = substrate_formula.get("H", 0)
    if args.clean_substrate:
        substrate_indices = np.asarray(
            [index for index in substrate_indices if template_symbols[index] != "H"],
            dtype=int,
        )
        substrate_formula = dict(
            sorted(Counter(template_symbols[substrate_indices]).items())
        )

    conformers = source_conformers(
        root,
        args.conformer_glob,
        _resolved(root, args.conformer_cache),
        spec,
        args.min_vertical_angle,
        args.max_marker_exposure,
        args.max_source_clusters,
    )
    assembled, placements, successful_attempt = assemble(
        template,
        target_components,
        substrate_indices,
        conformers,
        spec,
        args.seed,
        args.attempts,
        args.min_vertical_angle,
        args.max_marker_exposure,
        _resolved(root, args.domain_cache),
    )
    molecule_count = len(target_components)
    expected_formula = Counter(substrate_formula)
    for element, count in spec.formula.items():
        expected_formula[element] += molecule_count * count
    expected_formula = dict(sorted(expected_formula.items()))
    formula = dict(sorted(Counter(assembled.get_chemical_symbols()).items()))
    expected_atom_count = len(substrate_indices) + molecule_count * sum(spec.formula.values())
    if len(assembled) != expected_atom_count or formula != expected_formula:
        raise AssertionError(
            f"Assembled structure mismatch: atoms={len(assembled)}/{expected_atom_count}, "
            f"formula={formula}/{expected_formula}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write(output_path, assembled)
    if output_path.suffix.lower() == ".cif":
        lines = output_path.read_text().splitlines(keepends=True)
        output_path.write_text(
            "".join(
                line
                for line in lines
                if not line.startswith("_chemical_formula_structural")
            )
        )

    placement_records = []
    for placement in placements:
        conformer = conformers[placement["conformer_number"]]
        placement_records.append(
            {
                "site": placement["site"] + 1,
                "source": conformer["source"],
                "source_group": conformer["cluster"],
                "source_molecule": conformer["molecule"],
                "source_metrics": conformer["metrics"],
                "metrics": placement["transformed_metrics"],
                "headgroup_rmsd_A": placement["headgroup_rmsd_A"],
                "minimum_cross_distance_A": placement["minimum_cross_distance_A"],
                "minimum_clearance_ratio": placement["minimum_clearance_ratio"],
            }
        )
    matched_source_paths, selected_source_paths = selected_conformer_paths(
        root, args.conformer_glob, args.max_source_clusters
    )
    expected_template_skeleton = dict(sorted(spec.resolved_skeleton_formula().items()))
    template_skeleton_issues = []
    for component in target_components:
        actual = dict(
            sorted(
                Counter(template_symbols[list(component.heavy_skeleton)]).items()
            )
        )
        if actual != expected_template_skeleton:
            template_skeleton_issues.append(
                {
                    "anchor_index": component.anchor_index,
                    "actual_formula": actual,
                    "expected_formula": expected_template_skeleton,
                }
            )
    manifest = {
        "builder": "monolayer_headgroup_fit.py",
        "template": str(template_path),
        "output": str(output_path),
        "seed": args.seed,
        "successful_attempt": successful_attempt,
        "atom_count": len(assembled),
        "molecule_count": molecule_count,
        "molecule_formula": spec.formula,
        "molecule_spec": {
            "anchor_element": spec.anchor_element,
            "headgroup_element": spec.headgroup_element,
            "headgroup_count": spec.headgroup_count,
            "skeleton_formula": spec.resolved_skeleton_formula(),
        },
        "substrate_atom_count": len(substrate_indices),
        "substrate_formula": substrate_formula,
        "clean_substrate_H0": args.clean_substrate,
        "inherited_surface_H_removed": inherited_surface_h_count if args.clean_substrate else 0,
        "allow_incomplete_template_skeleton": args.allow_incomplete_template_skeleton,
        "template_skeleton_issues": template_skeleton_issues,
        "source_conformer_count": len(conformers),
        "source_selection": {
            "ordering": "natural_filename_rank",
            "maximum_source_clusters": args.max_source_clusters,
            "matched_source_file_count": len(matched_source_paths),
            "selected_source_file_count": len(selected_source_paths),
            "selected_source_files": [str(path) for path in selected_source_paths],
        },
        "minimum_vertical_angle_deg": args.min_vertical_angle,
        "maximum_marker_exposure_A": args.max_marker_exposure,
        "surface_h_repairs": surface_h_repairs,
        "placements": placement_records,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(
        json.dumps(
            {key: value for key, value in manifest.items() if key != "placements"},
            indent=2,
        )
    )
    return 0


def main() -> int:
    return headgroup_fit_main()


if __name__ == "__main__":
    raise SystemExit(main())
