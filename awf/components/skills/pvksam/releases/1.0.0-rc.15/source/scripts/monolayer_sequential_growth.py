#!/usr/bin/env python3
"""Build irreversible SAM growth or an immutable post-growth local repair.

``--operation grow`` opens every catalogued Site Instance, adds one locally
feasible molecule at a time, and stops only when no candidate remains.
``--operation repair`` preserves that parent history byte-for-byte and writes a
separate CPU-only H0 postprocessed/rearranged child.  Neither operation is a
global packing solver.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import importlib.metadata
import io
import itertools
import json
import math
import numbers
import os
import platform
import random
import stat
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from ase.io import read, write

from monolayer_headgroup_fit import selected_conformer_paths, source_conformers
from sam_lammps import read_typed_structure
from validate_monolayer import ValidationConfig, validate as validate_monolayer_structure
import sam_structure_tools
from sam_structure_tools import (
    MoleculeSpec,
    _require_shapely,
    _surface_lattice_uv,
    calculate_substrate_metal_coordinations,
    filled_outer_envelope_area,
    filled_outer_envelope_metrics,
    closed_distance_window_tolerance_A,
    distance_within_closed_window,
    molecular_components,
    periodic_hole_diagnostics,
    periodic_physical_geometry_metrics,
    periodic_polygon_coverage_mask,
    periodic_surface_polygon,
    periodic_union_increment_A2,
    periodic_vdw_collision_audit,
    periodic_union_increment_perimeter,
    periodic_uncovered_component_geometries,
    rigid_transform,
    surface_frame_coordinates,
    unwrap_component,
)


CAPACITY_RANKING_AUDIT_SCHEMA = "sam-sequential-capacity-ranking-step-v1"
CAPACITY_RANKING_SUMMARY_SCHEMA = "sam-sequential-capacity-ranking-summary-v1"


_STAGE3_SHAPE_FEATURE_NAMES = (
    "log_area_A2", "log_perimeter_A", "compactness", "anisotropy",
    "log_aspect_ratio", "principal_orientation_cos2",
    "principal_orientation_sin2",
)


def _stage3_shape_record_key(record: dict) -> str:
    key = record.get("source_key")
    if not isinstance(key, str) or not key:
        key = f"cluster-{int(record['cluster_id']):04d}-task-{int(record['source_task_index']):04d}"
    return key


def stage3_standardized_shape_features(records: list[dict]) -> dict:
    """Build deterministic standardized geometry features without energy."""
    if not isinstance(records, list) or not records:
        raise ValueError("Stage3 shape feature records must be a non-empty list")
    keys = [_stage3_shape_record_key(record) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Stage3 shape feature source keys must be unique")
    rows = []
    for record in records:
        metrics = record.get("shape_metrics")
        if not isinstance(metrics, dict):
            raise ValueError("Stage3 shape record lacks shape_metrics")
        try:
            area = float(metrics["area_A2"])
            perimeter = float(metrics["perimeter_A"])
            compactness = float(metrics["compactness"])
            anisotropy = float(metrics["anisotropy"])
            aspect = float(metrics["aspect_ratio"])
            cos2 = float(metrics["principal_orientation_cos2"])
            sin2 = float(metrics["principal_orientation_sin2"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Stage3 shape metrics are incomplete") from exc
        values = [area, perimeter, compactness, anisotropy, aspect, cos2, sin2]
        if (not np.all(np.isfinite(values)) or area <= 0.0 or perimeter <= 0.0 or aspect <= 0.0):
            raise ValueError("Stage3 shape metrics must be finite and positive where required")
        rows.append([np.log(area), np.log(perimeter), compactness, anisotropy, np.log(aspect), cos2, sin2])
    raw = np.asarray(rows, dtype=float)
    mean = np.mean(raw, axis=0)
    scale = np.where(np.std(raw, axis=0) > 1.0e-12, np.std(raw, axis=0), 1.0)
    # Keep the circular orientation pair on its natural [-1, 1] scale.  A
    # tiny sample variance must not turn the sin(2 theta) seam into a false
    # distance between theta≈0 and theta≈pi.
    scale[-2:] = np.maximum(scale[-2:], 1.0)
    standardized = (raw - mean) / scale
    if not np.all(np.isfinite(standardized)):
        raise ValueError("Stage3 standardized shape features must be finite")
    return {
        "schema": "sam-stage3-standardized-shape-features-v1",
        "feature_names": list(_STAGE3_SHAPE_FEATURE_NAMES),
        "orientation_encoding": "cos2theta_sin2theta",
        "standardization": "population_mean_and_population_std_zero_std_scale_one",
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "record_order": keys,
        "raw_features": {key: row.tolist() for key, row in zip(keys, raw)},
        "standardized_features": {key: row.tolist() for key, row in zip(keys, standardized)},
    }


def stage3_complete_link_clusters(records, standardized_features, *, threshold: float) -> dict:
    """Deterministically complete-link cluster standardized shape features."""
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("Stage3 shape-cluster threshold must be finite and non-negative")
    threshold = float(threshold)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("Stage3 shape-cluster threshold must be finite and non-negative")
    keys = sorted(_stage3_shape_record_key(record) for record in records)
    if len(keys) != len(set(keys)) or set(keys) != set(standardized_features):
        raise ValueError("Stage3 complete-link feature keys do not match records")
    vectors = {key: np.asarray(standardized_features[key], dtype=float) for key in keys}
    if any(vector.ndim != 1 or not np.all(np.isfinite(vector)) for vector in vectors.values()):
        raise ValueError("Stage3 complete-link feature vectors must be finite one-vectors")
    clusters = [[key] for key in keys]
    while len(clusters) > 1:
        candidates = []
        for left_index in range(len(clusters)):
            for right_index in range(left_index + 1, len(clusters)):
                left, right = clusters[left_index], clusters[right_index]
                distance = max(float(np.linalg.norm(vectors[a] - vectors[b])) for a in left for b in right)
                candidates.append((distance, tuple(sorted(left + right)), left_index, right_index))
        distance, _, left_index, right_index = min(candidates)
        if distance > threshold + 1.0e-12:
            break
        merged = sorted(clusters[left_index] + clusters[right_index])
        clusters = [cluster for index, cluster in enumerate(clusters) if index not in {left_index, right_index}]
        clusters.append(merged)
        clusters.sort(key=tuple)
    clusters.sort(key=tuple)
    assignments, cluster_records = {}, []
    for cluster_id, members in enumerate(clusters, 1):
        for key in members:
            assignments[key] = cluster_id
        distances = [float(np.linalg.norm(vectors[a] - vectors[b])) for i, a in enumerate(members) for b in members[i + 1:]]
        cluster_records.append({
            "shape_cluster_id": cluster_id,
            "member_source_keys": members,
            "member_count": len(members),
            "maximum_complete_link_distance": max(distances) if distances else 0.0,
        })
    return {
        "schema": "sam-stage3-deterministic-complete-link-shape-clusters-v1",
        "threshold": threshold,
        "metric": "euclidean_distance_on_standardized_shape_features",
        "orientation_encoding": "cos2theta_sin2theta",
        "contract": {
            "algorithm": "deterministic_complete_link_agglomeration",
            "tie_break": "distance_then_sorted_member_source_keys",
            "preselected_cluster_count": False,
            "orientation_encoding": "cos2theta_sin2theta",
        },
        "cluster_count": len(cluster_records),
        "assignments": assignments,
        "clusters": cluster_records,
    }


def stage3_choose_shape_diverse_next(
    records,
    selected_source_keys,
    assignments,
    standardized_features,
    parent_hole_match_scores,
):
    """Choose the next pool member by diversity, hole match, then energy."""
    selected = {str(value) for value in selected_source_keys}
    if not selected or not isinstance(assignments, dict):
        raise ValueError("Stage3 shape selection requires selected keys and assignments")
    vectors = {
        str(key): np.asarray(value, dtype=float)
        for key, value in standardized_features.items()
    }
    by_key = {_stage3_shape_record_key(record): record for record in records}
    if set(by_key) != set(assignments) or set(by_key) != set(vectors):
        raise ValueError("Stage3 shape selection records and assignments disagree")
    covered = {int(assignments[key]) for key in selected}
    ranking = []
    for key, record in by_key.items():
        if key in selected:
            continue
        distances = [float(np.linalg.norm(vectors[key] - vectors[other])) for other in selected]
        metrics = record.get("shape_metrics") or {}
        energy = float(record.get("relative_total_energy_eV", float("nan")))
        hole_score = float(parent_hole_match_scores[key])
        if not np.all(np.isfinite(distances + [energy, hole_score])):
            raise ValueError("Stage3 shape selection scores must be finite")
        ranking.append({
            "source_key": key,
            "shape_cluster_id": int(assignments[key]),
            "shape_cluster_uncovered": int(assignments[key]) not in covered,
            "minimum_distance_to_selected": min(distances),
            "parent_hole_match_score": hole_score,
            "relative_total_energy_eV": energy,
            "footprint_area_A2": float(metrics.get("area_A2", float("nan"))),
        })
    if not ranking:
        raise ValueError("Stage3 shape selection has no remaining source")
    ranking.sort(key=lambda row: (
        -int(row["shape_cluster_uncovered"]),
        -float(row["minimum_distance_to_selected"]),
        float(row["parent_hole_match_score"]),
        float(row["relative_total_energy_eV"]),
        str(row["source_key"]),
    ))
    return {
        "selected_source_key": ranking[0]["source_key"],
        "ranking": ranking,
        "contract": {
            "priority": [
                "uncovered_shape_cluster",
                "farthest_minimum_distance_to_selected",
                "parent_largest_hole_area_scale_and_periodic_distance_match",
                "relative_energy_final_tie_break_only",
                "source_key_stable_tie_break",
            ],
            "energy_used_as_primary": False,
        },
    }


@dataclass(frozen=True)
class SamplingPolicy:
    """Auditable conformer prior for sequential adsorption."""

    energy_scale_eV: float = 1.0
    area_coefficient: float = 2.0
    rank_exponent: float = 0.25

    def validate(self) -> None:
        values = (
            self.energy_scale_eV,
            self.area_coefficient,
            self.rank_exponent,
        )
        if not np.isfinite(values).all():
            raise ValueError("Sampling-policy parameters must be finite")
        if self.energy_scale_eV <= 0.0:
            raise ValueError("energy_scale_eV must be positive")
        if self.area_coefficient < 0.0 or self.rank_exponent < 0.0:
            raise ValueError("area_coefficient and rank_exponent must be non-negative")


def conformer_sampling_table(records: list[dict], policy: SamplingPolicy) -> list[dict]:
    """Return normalized base probabilities from rank, energy, and footprint."""

    policy.validate()
    if not records:
        raise ValueError("At least one conformer record is required")
    ranks = [int(record["selection_rank"]) for record in records]
    if any(rank <= 0 for rank in ranks) or len(set(ranks)) != len(ranks):
        raise ValueError("Conformer selection ranks must be unique positive integers")
    energies = np.asarray(
        [float(record["relative_total_energy_eV"]) for record in records], dtype=float
    )
    areas = np.asarray([float(record["footprint_area_A2"]) for record in records])
    if not np.isfinite(energies).all() or not np.isfinite(areas).all():
        raise ValueError("Conformer energies and footprint areas must be finite")
    if np.any(areas <= 0.0):
        raise ValueError("Conformer footprint areas must be positive")
    area_span = float(np.max(areas) - np.min(areas))
    normalized_area = (
        np.zeros_like(areas) if area_span <= 1.0e-12 else (areas - np.min(areas)) / area_span
    )
    log_weights = np.asarray(
        [
            -policy.rank_exponent * math.log(rank)
            - energies[position] / policy.energy_scale_eV
            - policy.area_coefficient * normalized_area[position]
            for position, rank in enumerate(ranks)
        ],
        dtype=float,
    )
    log_weights -= float(np.max(log_weights))
    weights = np.exp(log_weights)
    probabilities = weights / np.sum(weights)
    table = []
    for record, area_normalized, weight, probability in zip(
        records, normalized_area, weights, probabilities
    ):
        table.append(
            {
                "selection_rank": int(record["selection_rank"]),
                "cluster_id": int(record["cluster_id"]),
                "relative_total_energy_eV": float(
                    record["relative_total_energy_eV"]
                ),
                "footprint_area_A2": float(record["footprint_area_A2"]),
                "normalized_footprint_area": float(area_normalized),
                "unnormalized_weight": float(weight),
                "base_probability": float(probability),
                "structure": dict(record["structure"]),
            }
        )
    return table


def surface_collision_report(
    new_coordinates,
    new_symbols,
    existing_coordinates,
    existing_symbols,
    cell,
    periodic_axes=(0, 1),
    thresholds_A=None,
) -> tuple[int, float, float]:
    """Use the historical hard-distance contract without wrapping slab normal.

    The registered collision thresholds default to the historical H-H 1.20 A,
    H-heavy 1.40 A, heavy-heavy 1.80 A contract and may be overridden
    explicitly; the collision-contract fingerprint always records the exact
    thresholds used so a different threshold set can never reuse a cache.
    """

    thresholds_A = _validate_collision_thresholds(thresholds_A)
    hh_min = thresholds_A["H-H"]
    h_heavy_min = thresholds_A["H-heavy"]
    heavy_min = thresholds_A["heavy-heavy"]
    new_coordinates = np.asarray(new_coordinates, dtype=float)
    existing_coordinates = np.asarray(existing_coordinates, dtype=float)
    new_symbols = np.asarray(new_symbols, dtype=str)
    existing_symbols = np.asarray(existing_symbols, dtype=str)
    if len(existing_coordinates) == 0:
        return 0, float("inf"), float("inf")
    cell = np.asarray(cell, dtype=float)
    axes = _validated_surface_axes(periodic_axes)
    delta = new_coordinates[:, None, :] - existing_coordinates[None, :, :]
    partial_pbc = np.zeros(3, dtype=bool)
    partial_pbc[list(axes)] = True
    mic_vectors, mic_distances = find_mic(
        delta.reshape((-1, 3)), cell=cell, pbc=partial_pbc
    )
    mic_vectors = np.asarray(mic_vectors, dtype=float).reshape(delta.shape)
    distances = np.asarray(mic_distances, dtype=float).reshape(delta.shape[:2])
    if not np.all(np.isfinite(mic_vectors)) or not np.all(np.isfinite(distances)):
        raise RuntimeError("ASE partial-PBC find_mic returned non-finite evidence")
    new_h = new_symbols[:, None] == "H"
    old_h = existing_symbols[None, :] == "H"
    thresholds = np.where(
        new_h & old_h, hh_min, np.where(new_h | old_h, h_heavy_min, heavy_min)
    )
    ratios = distances / thresholds
    return (
        int(np.count_nonzero(distances < thresholds)),
        float(np.min(distances)),
        float(np.min(ratios)),
    )


def periodic_xy_sam_sam_collision(
    new_coordinates,
    new_symbols,
    existing_coordinates,
    existing_symbols,
    cell,
    periodic_axes=(0, 1),
    thresholds_A=None,
    surface_frame=None,
) -> tuple[int, float, float]:
    """Report periodic XY-projected SAM-SAM hard collisions.

    Calculates minimum-image displacement vectors in the periodic surface cell,
    computes 2D distance in the surface plane (XY/UV), and evaluates against
    the element-pair collision thresholds (H-H, H-heavy, heavy-heavy).
    """
    thresholds_A = _validate_collision_thresholds(thresholds_A)
    hh_min = thresholds_A["H-H"]
    h_heavy_min = thresholds_A["H-heavy"]
    heavy_min = thresholds_A["heavy-heavy"]
    new_coordinates = np.asarray(new_coordinates, dtype=float)
    existing_coordinates = np.asarray(existing_coordinates, dtype=float)
    new_symbols = np.asarray(new_symbols, dtype=str)
    existing_symbols = np.asarray(existing_symbols, dtype=str)
    if len(existing_coordinates) == 0 or len(new_coordinates) == 0:
        return 0, float("inf"), float("inf")
    cell = np.asarray(cell, dtype=float)
    axes = _validated_surface_axes(periodic_axes)
    delta = new_coordinates[:, None, :] - existing_coordinates[None, :, :]
    partial_pbc = np.zeros(3, dtype=bool)
    partial_pbc[list(axes)] = True
    mic_vectors, _ = find_mic(
        delta.reshape((-1, 3)), cell=cell, pbc=partial_pbc
    )
    mic_vectors = np.asarray(mic_vectors, dtype=float).reshape(delta.shape)
    if not np.all(np.isfinite(mic_vectors)):
        raise RuntimeError("ASE partial-PBC find_mic returned non-finite evidence")

    if surface_frame is not None:
        coords_uvn = surface_frame_coordinates(mic_vectors.reshape((-1, 3)), surface_frame).reshape(delta.shape)
        distances_xy = np.linalg.norm(coords_uvn[..., :2], axis=-1)
    else:
        distances_xy = np.linalg.norm(mic_vectors[..., list(axes)], axis=-1)

    new_h = new_symbols[:, None] == "H"
    old_h = existing_symbols[None, :] == "H"
    thresholds = np.where(
        new_h & old_h, hh_min, np.where(new_h | old_h, h_heavy_min, heavy_min)
    )
    ratios = distances_xy / thresholds
    return (
        int(np.count_nonzero(distances_xy < thresholds)),
        float(np.min(distances_xy)),
        float(np.min(ratios)),
    )


def _weighted_rank_choice(rng: random.Random, ranks: list[int], weights: dict[int, float]):
    total = sum(weights[rank] for rank in ranks)
    threshold = rng.random() * total
    cumulative = 0.0
    for rank in sorted(ranks):
        cumulative += weights[rank]
        if threshold <= cumulative:
            return rank
    return sorted(ranks)[-1]


def _sites_mask(site_ids, site_index) -> int:
    mask = 0
    for site_id in site_ids:
        mask |= 1 << site_index[site_id]
    return mask


class CollisionQueryProfiler:
    """Time and deduplicate direct 3D collision queries across trajectories.

    Queries are observed after the direct ``surface_collision_report`` call as
    unordered candidate pairs; repeats inside one trajectory are distinguished
    from repeats across trajectories.  Phase 1 deliberately caches no answers.
    """
    def __init__(self) -> None:
        self._query_count = 0
        self._seen_global: set[tuple[int, int]] = set()
        self._current_trajectory: set[tuple[int, int]] = set()
        self._within_trajectory_repeats = 0
        self._cross_trajectory_repeats = 0
        self._collision_seconds = 0.0
        self._overhead_seconds = 0.0

    def begin_trajectory(self) -> None:
        """Reset the within-trajectory repeat tracking at a trajectory boundary."""

        self._current_trajectory = set()

    def observe(self, left_index: int, right_index: int, seconds: float) -> None:
        """Record one unordered collision query pair and its direct cost."""

        started = time.perf_counter()
        pair = tuple(sorted((int(left_index), int(right_index))))
        self._query_count += 1
        self._collision_seconds += float(seconds)
        if pair in self._current_trajectory:
            self._within_trajectory_repeats += 1
        elif pair in self._seen_global:
            self._cross_trajectory_repeats += 1
        self._current_trajectory.add(pair)
        self._seen_global.add(pair)
        self._overhead_seconds += time.perf_counter() - started

    def summary(self) -> dict:
        unique = len(self._seen_global)
        return {
            "total_query_count": self._query_count,
            "unique_pair_count": unique,
            "repeated_query_count": self._query_count - unique,
            "within_trajectory_repeat_count": self._within_trajectory_repeats,
            "cross_trajectory_repeat_count": self._cross_trajectory_repeats,
            "collision_seconds": self._collision_seconds,
            "profiler_overhead_seconds": self._overhead_seconds,
        }


def _collision_contract_fingerprint(
    *, periodic_axes, thresholds_A, algorithm_version,
    p_surface_o_exclusion_contract=None,
) -> str:
    """Hash the exact SAM--SAM collision contract.

    Periodic axes, every registered threshold, and the algorithm version
    participate; seed, selection policy, conformer weights, footprint samples,
    and area tolerance deliberately do not, because they cannot change a fixed
    candidate-pair collision answer.
    """

    digest = hashlib.sha256()
    digest.update(b"collision-contract-v1")
    digest.update(str(tuple(int(axis) for axis in periodic_axes)).encode())
    thresholds = {
        str(key): float(value)
        for key, value in sorted(thresholds_A.items(), key=lambda item: str(item[0]))
    }
    digest.update(
        json.dumps(thresholds, separators=(",", ":")).encode("utf-8", "surrogatepass")
    )
    digest.update(str(algorithm_version).encode())
    if p_surface_o_exclusion_contract is not None:
        digest.update(
            json.dumps(
                p_surface_o_exclusion_contract,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
    return digest.hexdigest()


class LazyCollisionCache:
    """Run-shared symmetric lazy SAM--SAM collision cache.

    Each allocated candidate row is represented by Python-integer ``known``
    and ``collides`` bitsets; no eager NxN matrix is ever allocated.  The
    direct query remains the reference; a cached answer is only ever read
    after the exact pair was computed by ``direct_query`` earlier.
    """

    def __init__(
        self,
        *,
        candidate_domain_fingerprint: str,
        collision_contract_fingerprint: str,
        candidate_count: int,
        direct_query,
    ) -> None:
        self.candidate_domain_fingerprint = str(candidate_domain_fingerprint)
        self.collision_contract_fingerprint = str(collision_contract_fingerprint)
        self.candidate_count = int(candidate_count)
        self._direct_query = direct_query
        self._known_rows: dict[int, int] = {}
        self._collides_rows: dict[int, int] = {}
        self._hits = 0
        self._misses = 0
        self._direct_seconds = 0.0
        self._lookup_seconds = 0.0

    def query(self, left_index: int, right_index: int) -> bool:
        left = int(left_index)
        right = int(right_index)
        if left == right:
            raise ValueError("Collision query requires two distinct candidates")
        for index in (left, right):
            if index < 0 or index >= self.candidate_count:
                raise ValueError(f"candidate index {index} outside allocated range")
        started = time.perf_counter()
        known = bool((self._known_rows.get(left, 0) >> right) & 1)
        self._lookup_seconds += time.perf_counter() - started
        if known:
            self._hits += 1
            return bool((self._collides_rows.get(left, 0) >> right) & 1)
        query_started = time.perf_counter()
        collides = bool(self._direct_query(left, right))
        self._direct_seconds += time.perf_counter() - query_started
        self._misses += 1
        self._known_rows[left] = self._known_rows.get(left, 0) | (1 << right)
        self._known_rows[right] = self._known_rows.get(right, 0) | (1 << left)
        if collides:
            self._collides_rows[left] = self._collides_rows.get(left, 0) | (1 << right)
            self._collides_rows[right] = self._collides_rows.get(right, 0) | (1 << left)
        return collides

    def summary(self) -> dict:
        return {
            "candidate_count": self.candidate_count,
            "total_queries": self._hits + self._misses,
            "hits": self._hits,
            "misses": self._misses,
            "unique_pairs": self._misses,
            "allocated_rows": len(self._known_rows),
            "direct_seconds": self._direct_seconds,
            "lookup_seconds": self._lookup_seconds,
            "known_bitset_bytes": sum(
                (row.bit_length() + 7) // 8 for row in self._known_rows.values()
            ),
            "collides_bitset_bytes": sum(
                (row.bit_length() + 7) // 8 for row in self._collides_rows.values()
            ),
        }


def _normalized_conflict_edges(feasible_site_ids, site_conflicts):
    """Return unique undirected conflict edges within ``feasible_site_ids``."""

    feasible = {str(site_id) for site_id in feasible_site_ids}
    edges = set()
    for left, neighbors in (site_conflicts or {}).items():
        left = str(left)
        if left not in feasible:
            continue
        for right in neighbors:
            right = str(right)
            if right in feasible and right != left:
                edges.add(tuple(sorted((left, right))))
    return sorted(edges)


def solve_static_conflict_capacity(
    feasible_site_ids,
    metal_constraints,
    *,
    site_conflicts=None,
    time_limit_seconds=30.0,
) -> dict:
    """Return a fail-closed optimal shared-metal capacity certificate.

    The MILP maximizes Site Instance cardinality subject to metal capacity one
    and any explicit pairwise static conflicts.  Only HiGHS/SciPy status 0 with
    a zero reported MIP gap is accepted.  Time limits, non-optimal statuses,
    nonzero gaps, malformed integer solutions, and inconsistent bounds raise
    before a caller can commit a candidate transition.
    """

    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix

    started = time.perf_counter()
    try:
        time_limit_seconds = float(time_limit_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError("capacity time_limit_seconds must be finite and positive") from exc
    if not np.isfinite(time_limit_seconds) or time_limit_seconds <= 0.0:
        raise ValueError("capacity time_limit_seconds must be finite and positive")
    feasible = [str(site_id) for site_id in feasible_site_ids]
    if len(feasible) != len(set(feasible)):
        raise ValueError("feasible Site Instance IDs must be unique")
    feasible.sort()
    metal_constraints = metal_constraints or {}
    conflict_edges = _normalized_conflict_edges(feasible, site_conflicts)

    metal_to_sites = defaultdict(list)
    for site_id in feasible:
        for metal in set(metal_constraints.get(site_id, ())):
            metal_to_sites[metal].append(site_id)
    constrained_metals = sorted(
        metal_to_sites,
        key=lambda value: (type(value).__name__, repr(value)),
    )
    row_count = len(constrained_metals) + len(conflict_edges)
    if not feasible or row_count == 0:
        objective = len(feasible)
        return {
            "status": "optimal",
            "solver_status_code": 0,
            "success": True,
            "message": "Trivial exact capacity: no active static constraints",
            "objective": objective,
            "objective_bound": float(objective),
            "mip_dual_bound": -float(objective),
            "raw_minimization_objective": -float(objective),
            "mip_gap": 0.0,
            "mip_node_count": 0,
            "optimality_certified": True,
            "solve_time_seconds": time.perf_counter() - started,
            "time_limit_seconds": time_limit_seconds,
            "feasible_site_count": len(feasible),
            "metal_constraint_count": len(constrained_metals),
            "explicit_conflict_edge_count": len(conflict_edges),
            "constraint_row_count": row_count,
            "solver": "scipy.optimize.milp/HiGHS",
        }

    site_index = {site_id: index for index, site_id in enumerate(feasible)}
    incidence = lil_matrix((row_count, len(feasible)), dtype=float)
    row = 0
    for metal in constrained_metals:
        for site_id in metal_to_sites[metal]:
            incidence[row, site_index[site_id]] = 1.0
        row += 1
    for left, right in conflict_edges:
        incidence[row, site_index[left]] = 1.0
        incidence[row, site_index[right]] = 1.0
        row += 1

    result = milp(
        c=-np.ones(len(feasible)),
        integrality=np.ones(len(feasible)),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(
            incidence.tocsr(), -np.inf, np.ones(row_count)
        ),
        options={"time_limit": time_limit_seconds, "mip_rel_gap": 0.0},
    )
    elapsed = time.perf_counter() - started
    status_code = int(getattr(result, "status", -1))
    success = bool(getattr(result, "success", False))
    message = str(getattr(result, "message", "missing solver message"))
    raw_gap = getattr(result, "mip_gap", None)
    try:
        mip_gap = None if raw_gap is None else float(raw_gap)
    except (TypeError, ValueError, OverflowError):
        mip_gap = float("nan")
    raw_bound = getattr(result, "mip_dual_bound", None)
    try:
        raw_bound_value = None if raw_bound is None else float(raw_bound)
    except (TypeError, ValueError, OverflowError):
        raw_bound_value = float("nan")
    objective_bound = (
        None if raw_bound_value is None else -raw_bound_value
    )
    failure_details = (
        f"status={status_code}, success={success}, gap={mip_gap}, "
        f"mip_dual_bound={raw_bound_value}, message={message}"
    )
    if status_code != 0 or not success:
        raise RuntimeError(
            "Static capacity MILP did not return certified optimal status; "
            + failure_details
        )
    if mip_gap is None or not np.isfinite(mip_gap) or mip_gap != 0.0:
        raise RuntimeError(
            "Static capacity MILP requires a present, finite, exactly zero "
            "optimality gap; "
            + failure_details
        )
    if raw_bound_value is None or not np.isfinite(raw_bound_value):
        raise RuntimeError(
            "Static capacity MILP requires a present, finite dual bound; "
            + failure_details
        )
    fun = float(getattr(result, "fun", float("nan")))
    solution = np.asarray(getattr(result, "x", []), dtype=float)
    if not np.isfinite(fun) or solution.shape != (len(feasible),):
        raise RuntimeError(
            "Static capacity MILP returned a malformed objective/solution; "
            + failure_details
        )
    rounded = np.rint(solution)
    if (
        not np.all(np.isfinite(solution))
        or not np.allclose(solution, rounded, atol=1.0e-7, rtol=0.0)
        or np.any(rounded < 0.0)
        or np.any(rounded > 1.0)
    ):
        raise RuntimeError("Static capacity MILP returned a non-binary solution")
    objective = int(np.sum(rounded))
    if abs(fun + objective) > 1.0e-7:
        raise RuntimeError("Static capacity MILP objective disagrees with its solution")
    loads = np.asarray(incidence.tocsr() @ rounded, dtype=float).reshape(-1)
    if np.any(loads > 1.0 + 1.0e-7):
        raise RuntimeError("Static capacity MILP solution violates a capacity row")
    if abs(objective_bound - objective) > 1.0e-7:
        raise RuntimeError(
            "Static capacity MILP bound does not close on the integer objective "
            "after converting the minimization dual bound to maximization; "
            + failure_details
        )

    return {
        "status": "optimal",
        "solver_status_code": status_code,
        "success": success,
        "message": message,
        "objective": objective,
        "objective_bound": objective_bound,
        "mip_dual_bound": raw_bound_value,
        "raw_minimization_objective": fun,
        "mip_gap": mip_gap,
        "mip_node_count": (
            None
            if getattr(result, "mip_node_count", None) is None
            else int(result.mip_node_count)
        ),
        "optimality_certified": True,
        "solve_time_seconds": elapsed,
        "time_limit_seconds": time_limit_seconds,
        "feasible_site_count": len(feasible),
        "metal_constraint_count": len(constrained_metals),
        "explicit_conflict_edge_count": len(conflict_edges),
        "constraint_row_count": row_count,
        "solver": "scipy.optimize.milp/HiGHS",
    }


def _static_conflict_compatible_capacity(
    feasible_site_ids, metal_constraints, site_conflicts=None
) -> int:
    """Compatibility wrapper returning only the certified optimal objective."""

    return solve_static_conflict_capacity(
        feasible_site_ids,
        metal_constraints,
        site_conflicts=site_conflicts,
    )["objective"]


def capacity_debt_metrics(*, K_before, K_after_static, K_after_collision) -> dict:
    """Return the exact one-step capacity-debt definitions and safe bound scope."""

    values = {
        "K_before": K_before,
        "K_after_static": K_after_static,
        "K_after_collision": K_after_collision,
    }
    normalized = {}
    for name, value in values.items():
        if isinstance(value, bool) or int(value) != value or int(value) < 0:
            raise ValueError(f"{name} must be a non-negative integer")
        normalized[name] = int(value)
    K_before = normalized["K_before"]
    K_after_static = normalized["K_after_static"]
    K_after_collision = normalized["K_after_collision"]
    static_debt = K_before - 1 - K_after_static
    collision_debt = K_after_static - K_after_collision
    total_debt = K_before - 1 - K_after_collision
    if min(static_debt, collision_debt, total_debt) < 0:
        raise ValueError(
            "capacity debts must be non-negative under an exact candidate transition"
        )
    return {
        **normalized,
        "static_debt": static_debt,
        "collision_debt": collision_debt,
        "total_debt": total_debt,
        "one_plus_K_after_collision_upper_bound": 1 + K_after_collision,
        "upper_bound_scope": "shared_metal_and_individual_feasibility_only",
        "includes_future_candidate_pair_collisions": False,
        "realizable_terminal_packing_claim": False,
    }


def _static_blocked_site_ids(
    chosen_site_id, all_site_ids, site_conflicts, metal_constraints
) -> set[str]:
    """Return chosen plus every explicit or actual shared-metal conflict."""

    chosen_site_id = str(chosen_site_id)
    all_sites = {str(site_id) for site_id in all_site_ids}
    blocked = {chosen_site_id}
    for left, neighbors in (site_conflicts or {}).items():
        left = str(left)
        normalized_neighbors = {str(value) for value in neighbors}
        if left == chosen_site_id:
            blocked.update(normalized_neighbors)
        elif chosen_site_id in normalized_neighbors:
            blocked.add(left)
    chosen_metals = set((metal_constraints or {}).get(chosen_site_id, ()))
    if chosen_metals:
        for site_id in all_sites:
            if chosen_metals.intersection(
                set((metal_constraints or {}).get(site_id, ()))
            ):
                blocked.add(site_id)
    return blocked.intersection(all_sites)


def _resolve_transition_capacity(
    site_ids,
    *,
    metal_constraints,
    site_conflicts,
    time_limit_seconds,
    capacity_query,
):
    if capacity_query is None:
        result = solve_static_conflict_capacity(
            site_ids,
            metal_constraints,
            site_conflicts=site_conflicts,
            time_limit_seconds=time_limit_seconds,
        )
        return dict(result), False
    queried = capacity_query(tuple(sorted(str(value) for value in site_ids)))
    if isinstance(queried, tuple) and len(queried) == 2:
        result, cache_hit = queried
    else:
        result, cache_hit = queried, False
    result = dict(result)
    if result.get("status") != "optimal" or not result.get("optimality_certified"):
        raise RuntimeError("Capacity query did not provide a certified optimal result")
    return result, bool(cache_hit)


def simulate_candidate_transition(
    *,
    remaining_domains,
    chosen,
    site_conflicts,
    metal_constraints,
    cell,
    periodic_axes=(0, 1),
    collision_thresholds_A=None,
    mode="preview",
    capacity_time_limit_seconds=30.0,
    capacity_query=None,
    collision_cache=None,
    collision_profiler=None,
) -> dict:
    """Purely preview or materialize one irreversible candidate transition.

    The input domain mapping, every input list/candidate/coordinate array, and
    ``chosen`` are treated as immutable.  Static chosen/shared-metal removals
    happen before exact chosen-versus-candidate collision checks.  ``preview``
    stops at the first compatible candidate at each Site Instance, which is
    sufficient for the exact live-site set and capacity.  ``materialize`` (or
    ``commit``) retains every compatible candidate and reports exact pruning.
    """

    if mode not in {"preview", "materialize", "commit"}:
        raise ValueError("transition mode must be preview, materialize, or commit")
    materialize = mode in {"materialize", "commit"}
    if not isinstance(remaining_domains, dict):
        raise TypeError("remaining_domains must be a dictionary")
    chosen_site = str(chosen["site_instance_id"])
    if chosen_site not in remaining_domains:
        raise ValueError("chosen Site Instance is absent from remaining domains")
    chosen_id = chosen.get("candidate_id")
    chosen_index = chosen.get("candidate_index")
    chosen_is_live = any(
        (
            chosen_id is not None
            and candidate.get("candidate_id") == chosen_id
        )
        or (
            chosen_id is None
            and chosen_index is not None
            and candidate.get("candidate_index") == chosen_index
        )
        or (
            chosen_id is None
            and chosen_index is None
            and candidate is chosen
        )
        for candidate in remaining_domains[chosen_site]
    )
    if not chosen_is_live:
        raise ValueError("chosen candidate is absent from its current live domain")
    thresholds_A = _validate_collision_thresholds(collision_thresholds_A)
    blocked_sites = _static_blocked_site_ids(
        chosen_site,
        remaining_domains,
        site_conflicts,
        metal_constraints,
    )
    static_live_sites = sorted(
        site_id
        for site_id, candidates in remaining_domains.items()
        if candidates and str(site_id) not in blocked_sites
    )
    static_capacity, static_cache_hit = _resolve_transition_capacity(
        static_live_sites,
        metal_constraints=metal_constraints,
        site_conflicts=site_conflicts,
        time_limit_seconds=capacity_time_limit_seconds,
        capacity_query=capacity_query,
    )

    output_domains = (
        {
            site_id: ([] if str(site_id) in blocked_sites else list(candidates))
            for site_id, candidates in remaining_domains.items()
        }
        if materialize
        else None
    )
    live_sites = []
    pruned_to_empty = []
    collision_query_count = 0
    direct_query_count = 0
    cache_query_count = 0
    collision_seconds = 0.0
    collision_pruned = 0
    for site_id in static_live_sites:
        compatible = []
        for candidate in remaining_domains[site_id]:
            query_started = time.perf_counter()
            if collision_cache is not None:
                collides = bool(
                    collision_cache.query(
                        candidate["candidate_index"], chosen["candidate_index"]
                    )
                )
                cache_query_count += 1
                direct_seconds = 0.0
            else:
                count, _, _ = surface_collision_report(
                    candidate["coordinates"],
                    candidate["symbols"],
                    chosen["coordinates"],
                    chosen["symbols"],
                    cell,
                    periodic_axes,
                    thresholds_A=thresholds_A,
                )
                collides = count > 0
                direct_query_count += 1
                direct_seconds = time.perf_counter() - query_started
            elapsed = time.perf_counter() - query_started
            collision_seconds += elapsed
            collision_query_count += 1
            if collision_profiler is not None:
                collision_profiler.observe(
                    candidate["candidate_index"],
                    chosen["candidate_index"],
                    direct_seconds,
                )
            if collides:
                if materialize:
                    collision_pruned += 1
                continue
            compatible.append(candidate)
            if not materialize:
                break
        if compatible:
            live_sites.append(site_id)
        else:
            pruned_to_empty.append(site_id)
        if materialize:
            output_domains[site_id] = compatible

    collision_capacity, collision_cache_hit = _resolve_transition_capacity(
        live_sites,
        metal_constraints=metal_constraints,
        site_conflicts=site_conflicts,
        time_limit_seconds=capacity_time_limit_seconds,
        capacity_query=capacity_query,
    )
    static_removed_candidate_count = sum(
        len(candidates)
        for site_id, candidates in remaining_domains.items()
        if str(site_id) in blocked_sites
    )
    return {
        "mode": "materialize" if materialize else "preview",
        "chosen_candidate_id": chosen.get("candidate_id"),
        "chosen_site_instance_id": chosen_site,
        "collision_thresholds_A": thresholds_A,
        "static_blocked_site_ids": sorted(blocked_sites),
        "static_removed_candidate_count": static_removed_candidate_count,
        "static_live_site_ids": static_live_sites,
        "live_site_ids": live_sites,
        "K_after_static": int(static_capacity["objective"]),
        "K_after_collision": int(collision_capacity["objective"]),
        "capacity_after_static": static_capacity,
        "capacity_after_static_cache_hit": static_cache_hit,
        "capacity_after_collision": collision_capacity,
        "capacity_after_collision_cache_hit": collision_cache_hit,
        "collision_query_count": collision_query_count,
        "direct_collision_query_count": direct_query_count,
        "cached_collision_query_count": cache_query_count,
        "collision_query_seconds": collision_seconds,
        "collision_pruned_candidate_count": (
            collision_pruned if materialize else None
        ),
        "total_pruned_candidate_count": (
            static_removed_candidate_count + collision_pruned
            if materialize
            else None
        ),
        "remaining_candidate_count": (
            sum(len(candidates) for candidates in output_domains.values())
            if materialize
            else None
        ),
        "collision_pruned_count_is_exact": materialize,
        "collision_pruned_to_empty_sites": pruned_to_empty,
        "remaining_domains": output_domains,
    }


def choose_projection_candidate(
    *,
    candidates,
    site_conflicts,
    feasible_site_mask,
    accepted_coverage_mask,
    accepted_union,
    area_scale_A2,
    conformer_weights,
    rng,
    area_tie_tolerance_A2,
    perimeter_tie_tolerance_A=1.0e-6,
    projection_cache,
    selection_criterion="uncovered-anchors",
) -> tuple[dict, dict]:
    """Choose one candidate by the auditable lexicographic projection policy.

    Uncovered-anchor counts remain exact integers.  Floating area and physical
    flat-torus perimeter comparisons use independent tolerances in A2 and A,
    respectively; only candidates tied under both declared criteria reach the
    renormalized conformer prior.
    """

    if not candidates:
        raise ValueError("projection-aware scoring requires at least one candidate")
    if (
        not isinstance(area_tie_tolerance_A2, (int, float))
        or not np.isfinite(area_tie_tolerance_A2)
        or float(area_tie_tolerance_A2) < 0.0
    ):
        raise ValueError("area_tie_tolerance_A2 must be finite and non-negative")
    if (
        not isinstance(perimeter_tie_tolerance_A, (int, float))
        or not np.isfinite(perimeter_tie_tolerance_A)
        or float(perimeter_tie_tolerance_A) <= 0.0
    ):
        raise ValueError("perimeter_tie_tolerance_A must be finite and positive")
    area_tie_tolerance_A2 = float(area_tie_tolerance_A2)
    perimeter_tie_tolerance_A = float(perimeter_tie_tolerance_A)
    site_index = {
        site_id: index for index, site_id in enumerate(projection_cache.site_ids)
    }
    scored = []
    for candidate in candidates:
        candidate_mask = projection_cache.coverage_masks[
            candidate["candidate_index"]
        ]
        if selection_criterion == "uncovered-anchors":
            static_conflict_mask = _sites_mask(
                site_conflicts.get(candidate["site_instance_id"], set()), site_index
            )
            available_after_static = feasible_site_mask & ~static_conflict_mask
            uncovered_after_trial = (
                available_after_static
                & ~(accepted_coverage_mask | candidate_mask)
            )
            primary_value = uncovered_after_trial.bit_count()
            score = primary_value
        elif selection_criterion == "perimeter":
            geometry = projection_cache.candidate_geometries[
                candidate["candidate_index"]
            ]
            primary_value, _ = periodic_union_increment_perimeter(
                accepted_union,
                geometry,
                projection_cache.surface_lattice_uv_A,
            )
            score = -primary_value
        elif selection_criterion == "compact":
            geometry = projection_cache.candidate_geometries[
                candidate["candidate_index"]
            ]
            primary_value, _ = periodic_union_increment_A2(
                accepted_union, geometry, area_scale_A2
            )
            score = -primary_value
        else:
            raise ValueError(
                f"Unknown selection criterion {selection_criterion!r}"
            )
        scored.append((candidate, candidate_mask, primary_value, score))

    if selection_criterion == "uncovered-anchors":
        primary_extreme = max(value for _, _, value, _ in scored)
        score_finalists = [
            (candidate, mask)
            for candidate, mask, value, _ in scored
            if value == primary_extreme  # exact integer comparison by contract
        ]
        primary_scoring = {
            "name": "uncovered_feasible_anchor_proxies_after_trial",
            "direction": "maximize",
            "unit": "count",
            "tie_tolerance": 0,
            "tie_tolerance_unit": "count",
            "comparison": "exact_integer",
        }
    elif selection_criterion == "perimeter":
        primary_extreme = min(value for _, _, value, _ in scored)
        score_finalists = [
            (candidate, mask)
            for candidate, mask, value, _ in scored
            if value <= primary_extreme + perimeter_tie_tolerance_A
        ]
        primary_scoring = {
            "name": "periodic_union_perimeter_increment",
            "direction": "minimize",
            "unit": "A",
            "tie_tolerance": perimeter_tie_tolerance_A,
            "tie_tolerance_unit": "A",
        }
    else:
        primary_extreme = min(value for _, _, value, _ in scored)
        score_finalists = [
            (candidate, mask)
            for candidate, mask, value, _ in scored
            if value <= primary_extreme + area_tie_tolerance_A2
        ]
        primary_scoring = {
            "name": "periodic_union_area_increment",
            "direction": "minimize",
            "unit": "A2",
            "tie_tolerance": area_tie_tolerance_A2,
            "tie_tolerance_unit": "A2",
        }

    secondary_criterion = "area" if selection_criterion != "compact" else "perimeter"
    secondary_results = []
    for candidate, _ in score_finalists:
        geometry = projection_cache.candidate_geometries[
            candidate["candidate_index"]
        ]
        if secondary_criterion == "area":
            increment, merged = periodic_union_increment_A2(
                accepted_union, geometry, area_scale_A2
            )
        else:
            increment, merged = periodic_union_increment_perimeter(
                accepted_union,
                geometry,
                projection_cache.surface_lattice_uv_A,
            )
        secondary_results.append((candidate, increment, merged))
    minimum_value = min(increment for _, increment, _ in secondary_results)
    if secondary_criterion == "area":
        secondary_tolerance = area_tie_tolerance_A2
        secondary_scoring = {
            "name": "periodic_union_area_increment",
            "direction": "minimize",
            "unit": "A2",
            "tie_tolerance": area_tie_tolerance_A2,
            "tie_tolerance_unit": "A2",
        }
    else:
        secondary_tolerance = perimeter_tie_tolerance_A
        secondary_scoring = {
            "name": "periodic_union_perimeter_increment",
            "direction": "minimize",
            "unit": "A",
            "tie_tolerance": perimeter_tie_tolerance_A,
            "tie_tolerance_unit": "A",
        }
    finalists = [
        (candidate, increment, merged)
        for candidate, increment, merged in secondary_results
        if increment <= minimum_value + secondary_tolerance
    ]

    rank_probabilities = {}
    if len(finalists) == 1:
        winner = finalists[0][0]
    else:
        final_ranks = sorted(
            {int(candidate["selection_rank"]) for candidate, _, _ in finalists}
        )
        denominator = sum(conformer_weights[rank] for rank in final_ranks)
        rank_probabilities = {
            str(rank): conformer_weights[rank] / denominator for rank in final_ranks
        }
        rank = _weighted_rank_choice(rng, final_ranks, conformer_weights)
        rank_finalists = sorted(
            (
                candidate
                for candidate, _, _ in finalists
                if int(candidate["selection_rank"]) == rank
            ),
            key=lambda item: item["candidate_id"],
        )
        winner = rank_finalists[rng.randrange(len(rank_finalists))]
    winner_mask = next(
        mask
        for candidate, mask, _, _ in scored
        if candidate["candidate_id"] == winner["candidate_id"]
    )
    maximum_score = max(score for _, _, _, score in scored)
    audit = {
        "policy": "projection-aware",
        "selection_criterion": selection_criterion,
        "feasible_site_mask": feasible_site_mask,
        "accepted_coverage_mask": accepted_coverage_mask,
        "candidate_scores": {
            candidate["candidate_id"]: score
            for candidate, _, _, score in scored
        },
        "candidate_primary_values": {
            candidate["candidate_id"]: value
            for candidate, _, value, _ in scored
        },
        "primary_scoring": primary_scoring,
        "primary_extreme_value": primary_extreme,
        "primary_finalist_candidate_ids": [
            candidate["candidate_id"] for candidate, _ in score_finalists
        ],
        "maximum_uncovered_proxy_score": (
            primary_extreme
            if selection_criterion == "uncovered-anchors"
            else None
        ),
        "maximum_legacy_score": maximum_score,
        "secondary_criterion": secondary_criterion,
        "secondary_scoring": secondary_scoring,
        "secondary_evaluated_candidate_ids": [
            candidate["candidate_id"] for candidate, _, _ in secondary_results
        ],
        "finalist_candidate_ids": [
            candidate["candidate_id"] for candidate, _, _ in finalists
        ],
        "finalist_increments": {
            candidate["candidate_id"]: increment
            for candidate, increment, _ in secondary_results
        },
        "minimum_secondary_increment": minimum_value,
        "secondary_tie_tolerance": secondary_tolerance,
        "area_tie_tolerance_A2": area_tie_tolerance_A2,
        "perimeter_tie_tolerance_A": perimeter_tie_tolerance_A,
        "final_tie_ranks": list(rank_probabilities),
        "rank_probabilities": rank_probabilities,
        "winner_candidate_id": winner["candidate_id"],
        "winner_coverage_mask": winner_mask,
    }
    return winner, audit


def _cohesive_candidate_records(records):
    """Validate and deterministically order cohesive-policy candidate records."""
    if not isinstance(records, (list, tuple)):
        raise TypeError("records must be a list or tuple")
    seen = set()
    ordered = []
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("each candidate record must be a dict")
        candidate_id = record.get("candidate_id")
        if not isinstance(candidate_id, str):
            raise TypeError("candidate_id must be a string")
        if not candidate_id.strip():
            raise ValueError("candidate_id must be non-empty")
        if candidate_id in seen:
            raise ValueError(f"duplicate candidate_id: {candidate_id}")
        seen.add(candidate_id)
        ordered.append(record)
    return sorted(ordered, key=lambda record: record["candidate_id"])


def _cohesive_real(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number and not bool")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _cohesive_nonnegative(value, name):
    value = _cohesive_real(value, name)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _cohesive_relative_tolerance(value):
    value = _cohesive_real(value, "relative_tolerance")
    if not 0.0 <= value < 1.0:
        raise ValueError("relative_tolerance must be in [0, 1)")
    return value


DEFAULT_COHESIVE_PERIMETER_RELATIVE_TOLERANCE = 0.10
DEFAULT_COHESIVE_CONTACT_RELATIVE_TOLERANCE = 0.20
DEFAULT_COHESIVE_AREA_RELATIVE_TOLERANCE = 0.20


def resolve_cohesive_relative_tolerances(
    *,
    perimeter=None,
    contact=None,
    area=None,
    relative_tolerance=None,
) -> dict[str, float]:
    """Resolve cohesive-frontier relative tolerances with backward compatibility.

    Precedence per layer:
      1. Layer-specific override (perimeter, contact, area)
      2. Legacy umbrella tolerance (relative_tolerance)
      3. Formal canonical default (perimeter=0.10, contact=0.20, area=0.20)
    """
    umbrella = (
        None
        if relative_tolerance is None
        else _cohesive_relative_tolerance(relative_tolerance)
    )
    p = (
        _cohesive_relative_tolerance(perimeter)
        if perimeter is not None
        else (
            umbrella
            if umbrella is not None
            else DEFAULT_COHESIVE_PERIMETER_RELATIVE_TOLERANCE
        )
    )
    c = (
        _cohesive_relative_tolerance(contact)
        if contact is not None
        else (
            umbrella
            if umbrella is not None
            else DEFAULT_COHESIVE_CONTACT_RELATIVE_TOLERANCE
        )
    )
    a = (
        _cohesive_relative_tolerance(area)
        if area is not None
        else (
            umbrella
            if umbrella is not None
            else DEFAULT_COHESIVE_AREA_RELATIVE_TOLERANCE
        )
    )
    return {
        "perimeter": p,
        "contact": c,
        "area": a,
    }


def resolve_cohesive_cli_tolerances(args) -> dict[str, float]:
    """Resolve cohesive tolerances from parsed CLI arguments."""
    return resolve_cohesive_relative_tolerances(
        perimeter=getattr(args, "cohesive_perimeter_relative_tolerance", None),
        contact=getattr(args, "cohesive_contact_relative_tolerance", None),
        area=getattr(args, "cohesive_area_relative_tolerance", None),
        relative_tolerance=getattr(args, "cohesive_relative_tolerance", None),
    )


def resolve_cohesive_umbrella_audit(
    args=None,
    *,
    relative_tolerance: float | None = None,
    perimeter: float | None = None,
    contact: float | None = None,
    area: float | None = None,
) -> dict:
    """Analyze legacy umbrella CLI parameter usage and layer application."""
    if args is not None:
        umbrella_val = getattr(args, "cohesive_relative_tolerance", None)
        p_val = getattr(args, "cohesive_perimeter_relative_tolerance", None)
        c_val = getattr(args, "cohesive_contact_relative_tolerance", None)
        a_val = getattr(args, "cohesive_area_relative_tolerance", None)
    else:
        umbrella_val = relative_tolerance
        p_val = perimeter
        c_val = contact
        a_val = area

    provided = umbrella_val is not None
    applied_layers = []
    if provided:
        if p_val is None:
            applied_layers.append("perimeter")
        if c_val is None:
            applied_layers.append("contact")
        if a_val is None:
            applied_layers.append("area")

    return {
        "legacy_umbrella_relative_tolerance": (
            _cohesive_relative_tolerance(umbrella_val) if provided else None
        ),
        "legacy_umbrella_provided": provided,
        "legacy_umbrella_applied_layers": applied_layers,
        "legacy_umbrella_used": bool(applied_layers),
    }


def cohesive_perimeter_condensation_gain(
    L_before, L_candidate, L_after
):
    """Return ``(L_before + L_candidate - L_after) / L_candidate``."""
    L_before = _cohesive_nonnegative(L_before, "L_before")
    L_candidate = _cohesive_real(L_candidate, "L_candidate")
    L_after = _cohesive_nonnegative(L_after, "L_after")
    if L_candidate <= 0.0:
        raise ValueError("L_candidate must be positive")
    return float((L_before + L_candidate - L_after) / L_candidate)


def cohesive_filter_perimeter_increment(
    records,
    relative_tolerance=DEFAULT_COHESIVE_PERIMETER_RELATIVE_TOLERANCE,
    numerical_tolerance=1.0e-9,
):
    """Minimize ``increment_A`` within a minimum-candidate-perimeter band.

    Every candidate in one growth step uses the same absolute tolerance,
    ``relative_tolerance * min(candidate_perimeter_A)``.  The diagnostic
    condensation gain remains validated and recorded, but it does not rank
    candidates.
    """
    relative_tolerance = _cohesive_relative_tolerance(relative_tolerance)
    numerical_tolerance = _cohesive_nonnegative(
        numerical_tolerance, "numerical_tolerance"
    )
    ordered = _cohesive_candidate_records(records)
    audit = {
        "metric": "periodic_union_perimeter_increment_A",
        "direction": "minimize",
        "relative_tolerance": relative_tolerance,
        "tolerance_scale": "minimum_candidate_perimeter_A",
        "numerical_tolerance": numerical_tolerance,
        "invariant": "periodic_union_perimeter_subadditivity_requires_gain_nonnegative",
        "input_candidate_ids": [record["candidate_id"] for record in ordered],
        "minimum_increment_A": None,
        "minimum_candidate_perimeter_A": None,
        "allowed_increment_excess_A": None,
        "cutoff_increment_A": None,
        "comparison_operator": "<=",
        "retained_candidate_ids": [],
    }
    if not ordered:
        return [], audit

    annotated = []
    for record in ordered:
        candidate_id = record["candidate_id"]
        if "increment_A" not in record:
            raise KeyError(f"candidate {candidate_id!r} lacks increment_A")
        if "candidate_perimeter_A" not in record:
            raise KeyError(
                f"candidate {candidate_id!r} lacks candidate_perimeter_A"
            )
        if "perimeter_condensation_gain" not in record:
            raise KeyError(
                f"candidate {candidate_id!r} lacks perimeter_condensation_gain"
            )
        increment_A = _cohesive_real(
            record["increment_A"], f"increment_A[{candidate_id}]"
        )
        candidate_perimeter_A = _cohesive_real(
            record["candidate_perimeter_A"],
            f"candidate_perimeter_A[{candidate_id}]",
        )
        if candidate_perimeter_A <= 0.0:
            raise ValueError(
                f"candidate_perimeter_A[{candidate_id}] must be positive"
            )
        gain = _cohesive_real(
            record["perimeter_condensation_gain"],
            f"perimeter_condensation_gain[{candidate_id}]",
        )
        if gain < -numerical_tolerance:
            raise ValueError(
                "periodic union perimeter invariant violated: "
                f"candidate {candidate_id!r} has gain {gain} below "
                f"-{numerical_tolerance}"
            )
        if gain < 0.0:
            gain = 0.0
        annotated_record = dict(record)
        annotated_record["increment_A"] = float(increment_A)
        annotated_record["candidate_perimeter_A"] = float(candidate_perimeter_A)
        annotated_record["perimeter_condensation_gain"] = float(gain)
        annotated.append(annotated_record)

    minimum_increment_A = min(record["increment_A"] for record in annotated)
    minimum_candidate_perimeter_A = min(
        record["candidate_perimeter_A"] for record in annotated
    )
    allowed_increment_excess_A = (
        relative_tolerance * minimum_candidate_perimeter_A
    )
    cutoff_increment_A = minimum_increment_A + allowed_increment_excess_A
    finalists = [
        record
        for record in annotated
        if record["increment_A"] <= cutoff_increment_A + numerical_tolerance
    ]
    audit.update(
        {
            "minimum_increment_A": float(minimum_increment_A),
            "minimum_candidate_perimeter_A": float(
                minimum_candidate_perimeter_A
            ),
            "allowed_increment_excess_A": float(allowed_increment_excess_A),
            "cutoff_increment_A": float(cutoff_increment_A),
            "retained_candidate_ids": [
                record["candidate_id"] for record in finalists
            ],
        }
    )
    return finalists, audit


def cohesive_filter_distinct_neighbors(records):
    """Retain candidates maximizing the count of distinct contacted adsorbed neighbors.

    Implements criterion 3a: exact maximum (0 deficit). No tolerance percentage is applied;
    only candidates achieving the strict maximum distinct contacted neighbor count are retained.
    Subsequent pair-count filtering (criterion 3b) then applies the relative tolerance (default 20%).
    """
    ordered = _cohesive_candidate_records(records)
    audit = {
        "metric": "distinct_neighbor_count",
        "direction": "maximize",
        "tolerance_policy": "exact_maximum",
        "allowed_count_deficit": 0,
        "input_candidate_ids": [record["candidate_id"] for record in ordered],
        "maximum_distinct_neighbor_count": None,
        "cutoff_distinct_neighbor_count": None,
        "retained_candidate_ids": [],
    }
    if not ordered:
        return [], audit

    annotated = []
    for record in ordered:
        candidate_id = record["candidate_id"]
        count = record.get("distinct_neighbor_count")
        if count is None:
            count = record.get(
                "distinct_contacted_neighbor_count",
                record.get("contacted_neighbor_count"),
            )
        if isinstance(count, bool) or not isinstance(count, numbers.Integral):
            raise TypeError(
                f"distinct_neighbor_count[{candidate_id}] must be an integer"
            )
        count = int(count)
        if count < 0:
            raise ValueError(
                f"distinct_neighbor_count[{candidate_id}] must be non-negative"
            )
        annotated_record = dict(record)
        annotated_record["distinct_neighbor_count"] = count
        annotated.append(annotated_record)

    maximum = max(record["distinct_neighbor_count"] for record in annotated)
    cutoff = maximum
    finalists = [
        record for record in annotated if record["distinct_neighbor_count"] >= cutoff
    ]
    audit.update(
        {
            "maximum_distinct_neighbor_count": int(maximum),
            "cutoff_distinct_neighbor_count": int(cutoff),
            "retained_candidate_ids": [
                record["candidate_id"] for record in finalists
            ],
        }
    )
    return finalists, audit


def cohesive_filter_side_contacts(records, relative_tolerance=0.20):
    """Retain integer side-contact counts within the declared relative band."""
    relative_tolerance = _cohesive_relative_tolerance(relative_tolerance)
    ordered = _cohesive_candidate_records(records)
    audit = {
        "metric": "side_contact_count",
        "direction": "maximize",
        "relative_tolerance": relative_tolerance,
        "input_candidate_ids": [record["candidate_id"] for record in ordered],
        "maximum_contact_count": None,
        "allowed_count_deficit": None,
        "cutoff_contact_count": None,
        "retained_candidate_ids": [],
    }
    if not ordered:
        return [], audit

    annotated = []
    for record in ordered:
        candidate_id = record["candidate_id"]
        count = record.get("side_contact_count")
        if isinstance(count, bool) or not isinstance(count, numbers.Integral):
            raise TypeError(f"side_contact_count[{candidate_id}] must be an integer")
        count = int(count)
        if count < 0:
            raise ValueError(
                f"side_contact_count[{candidate_id}] must be non-negative"
            )
        annotated_record = dict(record)
        annotated_record["side_contact_count"] = count
        annotated.append(annotated_record)

    maximum = max(record["side_contact_count"] for record in annotated)
    deficit = max(1, math.ceil(relative_tolerance * maximum))
    cutoff = maximum - deficit
    finalists = [
        record for record in annotated if record["side_contact_count"] >= cutoff
    ]
    audit.update(
        {
            "maximum_contact_count": int(maximum),
            "allowed_count_deficit": int(deficit),
            "cutoff_contact_count": int(cutoff),
            "retained_candidate_ids": [
                record["candidate_id"] for record in finalists
            ],
        }
    )
    return finalists, audit


def cohesive_lazy_side_contacts(records, callback, relative_tolerance=0.20):
    """Evaluate side contacts only for already selected perimeter finalists."""
    ordered = _cohesive_candidate_records(records)
    annotated = []
    evaluated = []
    for record in ordered:
        count = callback(record)
        annotated_record = dict(record)
        annotated_record["side_contact_count"] = count
        annotated.append(annotated_record)
        evaluated.append(record["candidate_id"])
    finalists, audit = cohesive_filter_side_contacts(
        annotated, relative_tolerance=relative_tolerance
    )
    audit["evaluated_candidate_ids"] = evaluated
    return finalists, audit


def cohesive_filter_projected_area(records, relative_tolerance=0.20):
    """Retain fitted single-candidate footprints within 120% by default."""
    relative_tolerance = _cohesive_relative_tolerance(relative_tolerance)
    ordered = _cohesive_candidate_records(records)
    audit = {
        "metric": "projected_area_A2",
        "direction": "minimize",
        "relative_tolerance": relative_tolerance,
        "input_candidate_ids": [record["candidate_id"] for record in ordered],
        "minimum_area_A2": None,
        "cutoff_area_A2": None,
        "retained_candidate_ids": [],
    }
    if not ordered:
        return [], audit

    annotated = []
    for record in ordered:
        candidate_id = record["candidate_id"]
        area = _cohesive_real(
            record.get("projected_area_A2"), f"projected_area_A2[{candidate_id}]"
        )
        if area <= 0.0:
            raise ValueError(f"projected_area_A2[{candidate_id}] must be positive")
        annotated_record = dict(record)
        annotated_record["projected_area_A2"] = area
        annotated.append(annotated_record)
    minimum = min(record["projected_area_A2"] for record in annotated)
    cutoff = (1.0 + relative_tolerance) * minimum
    finalists = [
        record for record in annotated if record["projected_area_A2"] <= cutoff
    ]
    audit.update(
        {
            "minimum_area_A2": float(minimum),
            "cutoff_area_A2": float(cutoff),
            "retained_candidate_ids": [
                record["candidate_id"] for record in finalists
            ],
        }
    )
    return finalists, audit


def cohesive_filter_conformer_energy(records, tolerance_eV=0.03):
    """Retain conformers within the declared energy band above the minimum."""
    tolerance_eV = _cohesive_nonnegative(tolerance_eV, "tolerance_eV")
    ordered = _cohesive_candidate_records(records)
    audit = {
        "metric": "conformer_energy_eV",
        "direction": "minimize",
        "tolerance_eV": tolerance_eV,
        "input_candidate_ids": [record["candidate_id"] for record in ordered],
        "minimum_energy_eV": None,
        "cutoff_energy_eV": None,
        "retained_candidate_ids": [],
    }
    if not ordered:
        return [], audit

    annotated = []
    for record in ordered:
        candidate_id = record["candidate_id"]
        energy = _cohesive_real(
            record.get("conformer_energy_eV"),
            f"conformer_energy_eV[{candidate_id}]",
        )
        annotated_record = dict(record)
        annotated_record["conformer_energy_eV"] = energy
        annotated.append(annotated_record)
    minimum = min(record["conformer_energy_eV"] for record in annotated)
    cutoff = minimum + tolerance_eV
    finalists = [
        record
        for record in annotated
        if record["conformer_energy_eV"] <= cutoff
    ]
    audit.update(
        {
            "minimum_energy_eV": float(minimum),
            "cutoff_energy_eV": float(cutoff),
            "retained_candidate_ids": [
                record["candidate_id"] for record in finalists
            ],
        }
    )
    return finalists, audit


def _cohesive_contact_candidate(candidate, label):
    """Validate one candidate and expose its full and non-headgroup atoms."""
    if not isinstance(candidate, dict):
        raise TypeError(f"{label} must be a candidate dictionary")
    candidate_id = candidate.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise ValueError(f"{label} candidate_id must be a non-empty string")
    coordinates = np.asarray(candidate.get("coordinates"), dtype=float)
    symbols = np.asarray(candidate.get("symbols"), dtype=str)
    if coordinates.ndim != 2 or coordinates.shape[1:] != (3,):
        raise ValueError(f"{label} coordinates must have shape (N, 3)")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError(f"{label} coordinates must be finite")
    if symbols.ndim != 1 or len(symbols) != len(coordinates):
        raise ValueError(f"{label} symbols must match its coordinates")
    raw_headgroup = candidate.get("headgroup_local_indices_0based")
    if not isinstance(raw_headgroup, (list, tuple, np.ndarray)):
        raise ValueError(f"{label} headgroup indices must be a sequence")
    if any(
        isinstance(index, (bool, np.bool_))
        or not isinstance(index, (int, np.integer))
        for index in raw_headgroup
    ):
        raise ValueError(f"{label} headgroup indices must be integers")
    headgroup = [int(index) for index in raw_headgroup]
    if not headgroup or len(headgroup) != len(set(headgroup)):
        raise ValueError(f"{label} headgroup indices must be nonempty and unique")
    if min(headgroup) < 0 or max(headgroup) >= len(coordinates):
        raise ValueError(f"{label} headgroup index is outside the molecule")
    headgroup_set = set(headgroup)
    body_indices = np.asarray(
        [index for index in range(len(coordinates)) if index not in headgroup_set],
        dtype=int,
    )
    if not len(body_indices):
        raise ValueError(f"{label} has no body atoms after headgroup exclusion")
    radii_by_symbol = _resolved_ase_vdw_radii(symbols)
    radii = np.asarray([radii_by_symbol[str(symbol)] for symbol in symbols], dtype=float)
    return {
        "candidate_id": candidate_id,
        "coordinates": coordinates,
        "symbols": symbols,
        "radii_A": radii,
        "body_indices": body_indices,
        "headgroup_indices": headgroup,
    }


def cohesive_side_contact_report(
    new_candidate,
    accepted_candidates,
    *,
    cell,
    periodic_axes,
    collision_thresholds_A,
    contact_vdw_radius_scale=1.10,
    stop_after_first=False,
):
    """Count non-headgroup intermolecular contacts under partial periodicity."""
    scale = _cohesive_real(
        contact_vdw_radius_scale, "contact_vdw_radius_scale"
    )
    if scale < 1.0:
        raise ValueError("contact_vdw_radius_scale must be at least 1.0")
    if not isinstance(stop_after_first, bool):
        raise TypeError("stop_after_first must be bool")
    if not isinstance(accepted_candidates, (list, tuple)):
        raise TypeError("accepted_candidates must be a list or tuple")
    axes = _validated_surface_axes(periodic_axes)
    cell = np.asarray(cell, dtype=float)
    if cell.shape != (3, 3) or not np.all(np.isfinite(cell)):
        raise ValueError("cell must be a finite 3x3 matrix")
    thresholds_A = _validate_collision_thresholds(collision_thresholds_A)
    new = _cohesive_contact_candidate(new_candidate, "new_candidate")

    accepted = [
        _cohesive_contact_candidate(candidate, "accepted_candidate")
        for candidate in accepted_candidates
    ]
    accepted.sort(key=lambda record: record["candidate_id"])
    accepted_ids = [record["candidate_id"] for record in accepted]
    if len(accepted_ids) != len(set(accepted_ids)):
        raise ValueError("accepted candidate IDs must be unique")
    if new["candidate_id"] in set(accepted_ids):
        raise ValueError("new candidate ID must differ from accepted candidate IDs")

    pbc = np.zeros(3, dtype=bool)
    pbc[list(axes)] = True
    contact_count = 0
    evaluated_ids = []
    contacted_neighbor_ids = []
    contact_pair_counts_by_neighbor_id = {}
    stopped = False
    for existing in accepted:
        evaluated_ids.append(existing["candidate_id"])
        delta = (
            new["coordinates"][:, None, :]
            - existing["coordinates"][None, :, :]
        )
        mic_vectors, mic_distances = find_mic(
            delta.reshape((-1, 3)), cell=cell, pbc=pbc
        )
        mic_vectors = np.asarray(mic_vectors, dtype=float).reshape(delta.shape)
        distances = np.asarray(mic_distances, dtype=float).reshape(delta.shape[:2])
        if not np.all(np.isfinite(mic_vectors)) or not np.all(np.isfinite(distances)):
            raise RuntimeError("ASE partial-PBC contact MIC returned non-finite values")

        new_h = new["symbols"][:, None] == "H"
        existing_h = existing["symbols"][None, :] == "H"
        hard_thresholds = np.where(
            new_h & existing_h,
            thresholds_A["H-H"],
            np.where(
                new_h | existing_h,
                thresholds_A["H-heavy"],
                thresholds_A["heavy-heavy"],
            ),
        )
        if np.any(distances < hard_thresholds):
            raise RuntimeError(
                "cohesive contact input violates the prior SAM-SAM collision gate"
            )

        body_distances = distances[
            np.ix_(new["body_indices"], existing["body_indices"])
        ]
        body_upper = scale * (
            new["radii_A"][new["body_indices"]][:, None]
            + existing["radii_A"][existing["body_indices"]][None, :]
        )
        comparison_tolerance = 8.0 * np.maximum(
            np.spacing(np.abs(body_distances)), np.spacing(np.abs(body_upper))
        )
        pair_count = int(
            np.count_nonzero(
                body_distances <= body_upper + comparison_tolerance
            )
        )
        if pair_count:
            contact_count += pair_count
            contacted_neighbor_ids.append(existing["candidate_id"])
            contact_pair_counts_by_neighbor_id[existing["candidate_id"]] = pair_count
            if stop_after_first:
                stopped = True
                break

    is_exact = not stopped
    distinct_neighbor_count = len(contacted_neighbor_ids)
    return {
        "contact_count": int(contact_count),
        "contact_count_exact": is_exact,
        "has_contact": bool(contact_count),
        "contacted_neighbor_ids": list(contacted_neighbor_ids),
        "distinct_contacted_neighbor_count": int(distinct_neighbor_count),
        "distinct_contacted_neighbor_count_exact": is_exact,
        "contacted_neighbor_count": int(distinct_neighbor_count),
        "contacted_neighbor_count_exact": is_exact,
        "contact_pair_counts_by_neighbor_id": dict(contact_pair_counts_by_neighbor_id),
        "evaluated_existing_candidate_ids": evaluated_ids,
        "stopped_after_first": stopped,
        "contact_vdw_radius_scale": scale,
        "radii_source": "ASE ase.data.vdw_radii",
        "periodic_axes": list(axes),
        "normal_wrapped": False,
        "headgroup_exclusion": {
            "new_candidate_atom_count": len(new["headgroup_indices"]),
            "accepted_candidate_atom_counts": {
                record["candidate_id"]: len(record["headgroup_indices"])
                for record in accepted
            },
        },
        "contact_definition": (
            "non-headgroup pairs outside the hard collision gate and at or below "
            "contact_vdw_radius_scale times the ASE vdW-radius sum"
        ),
        "upper_bound_roundoff_policy": "inclusive_with_eight_ULPs_only",
    }


def cohesive_xy_bound(candidate):
    """Return a conservative body-atom XY bound about the adsorption anchor."""
    validated = _cohesive_contact_candidate(candidate, "candidate")
    anchor = np.asarray(candidate.get("anchor_cartesian_A"), dtype=float)
    if anchor.shape != (3,) or not np.all(np.isfinite(anchor)):
        raise ValueError("cohesive candidate requires a finite anchor_cartesian_A")
    frame = candidate.get("surface_frame")
    if not isinstance(frame, dict):
        raise ValueError("cohesive candidate requires a Surface Frame")
    body_indices = validated["body_indices"]
    relative_uv = surface_frame_coordinates(
        validated["coordinates"][body_indices] - anchor[None, :], frame
    )[:, :2]
    return {
        "candidate_id": validated["candidate_id"],
        "anchor_cartesian_A": anchor,
        "maximum_body_center_offset_xy_A": float(
            np.max(np.linalg.norm(relative_uv, axis=1))
        ),
        "maximum_body_vdw_radius_A": float(
            np.max(validated["radii_A"][body_indices])
        ),
        "surface_frame": frame,
    }


def cohesive_xy_may_contact(
    new_candidate,
    accepted_candidates,
    *,
    cell,
    periodic_axes,
    contact_vdw_radius_scale=1.10,
    bounds_by_candidate_id=None,
):
    """Conservatively reject molecule pairs too far apart in periodic XY.

    The triangle-inequality bound is intentionally a superset of the exact
    body-atom contact shell, so a False result cannot hide a real contact.
    """
    scale = _cohesive_real(contact_vdw_radius_scale, "contact_vdw_radius_scale")
    if scale < 1.0:
        raise ValueError("contact_vdw_radius_scale must be at least 1.0")
    axes = _validated_surface_axes(periodic_axes)
    cell = np.asarray(cell, dtype=float)
    pbc = np.zeros(3, dtype=bool)
    pbc[list(axes)] = True
    supplied = {} if bounds_by_candidate_id is None else bounds_by_candidate_id

    def resolve(candidate):
        candidate_id = candidate.get("candidate_id")
        bound = supplied.get(candidate_id)
        return cohesive_xy_bound(candidate) if bound is None else bound

    new_bound = resolve(new_candidate)
    existing_bounds = [
        resolve(candidate)
        for candidate in sorted(
            accepted_candidates, key=lambda item: item["candidate_id"]
        )
    ]
    if not existing_bounds:
        possible_ids = []
    else:
        deltas = new_bound["anchor_cartesian_A"][None, :] - np.asarray(
            [bound["anchor_cartesian_A"] for bound in existing_bounds]
        )
        mic, _ = find_mic(deltas, cell=cell, pbc=pbc)
        distances_xy_A = np.linalg.norm(
            surface_frame_coordinates(mic, new_bound["surface_frame"])[:, :2],
            axis=1,
        )
        cutoffs_A = np.asarray(
            [
                new_bound["maximum_body_center_offset_xy_A"]
                + bound["maximum_body_center_offset_xy_A"]
                + scale
                * (
                    new_bound["maximum_body_vdw_radius_A"]
                    + bound["maximum_body_vdw_radius_A"]
                )
                for bound in existing_bounds
            ]
        )
        tolerances_A = 8.0 * np.maximum(
            np.abs(np.spacing(distances_xy_A)), np.abs(np.spacing(cutoffs_A))
        )
        possible_ids = [
            bound["candidate_id"]
            for bound, possible in zip(
                existing_bounds, distances_xy_A <= cutoffs_A + tolerances_A
            )
            if bool(possible)
        ]
    return {
        "may_contact": bool(possible_ids),
        "possible_existing_candidate_ids": possible_ids,
        "zero_false_negative_contract": "periodic_XY_triangle_inequality_body_bound",
    }


def _cohesive_polygon_pieces(geometry):
    """Return positive-area Polygon pieces from one clipped torus geometry."""
    _, _, Polygon, _, _ = _require_shapely()
    if geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry] if geometry.area > 0.0 else []
    pieces = []
    for child in getattr(geometry, "geoms", ()):
        pieces.extend(_cohesive_polygon_pieces(child))
    return pieces


def _cohesive_periodic_intervals(start, stop):
    width = float(stop) - float(start)
    if not np.isfinite(width) or width < 0.0:
        raise ValueError("periodic query interval must have finite nonnegative width")
    if width >= 1.0:
        return [(0.0, 1.0)]
    wrapped_start = float(start) % 1.0
    wrapped_stop = wrapped_start + width
    if wrapped_stop <= 1.0:
        return [(wrapped_start, wrapped_stop)]
    return [(wrapped_start, 1.0), (0.0, wrapped_stop - 1.0)]


class CohesiveProjectionNeighborIndex:
    """Small periodic STRtree over accepted projection fragments.

    The tree is rebuilt only after an accepted molecule is added. Candidate
    queries therefore scale with local projection fragments rather than with
    the complete accepted history.
    """

    def __init__(
        self,
        *,
        projection_cache,
        candidates,
        collision_thresholds_A,
        contact_vdw_radius_scale=1.10,
        extra_symbols=None,
    ):
        self.projection_cache = projection_cache
        candidates = list(candidates)
        self.candidate_by_id = {
            candidate["candidate_id"]: candidate for candidate in candidates
        }
        if len(self.candidate_by_id) != len(candidates):
            raise ValueError("cohesive projection index candidate IDs must be unique")
        lattice = np.asarray(projection_cache.surface_lattice_uv_A, dtype=float)
        if lattice.shape != (2, 2) or not np.all(np.isfinite(lattice)):
            raise ValueError("cohesive projection index requires a finite 2x2 lattice")
        inverse = np.linalg.inv(lattice)
        projection_scale = float(
            getattr(projection_cache, "contract", {}).get("radius_scale", 1.0)
        )
        if not np.isfinite(projection_scale) or projection_scale <= 0.0:
            raise ValueError("projection radius scale must be finite and positive")
        contact_scale = _cohesive_real(
            contact_vdw_radius_scale, "contact_vdw_radius_scale"
        )
        thresholds = _validate_collision_thresholds(collision_thresholds_A)
        cand_symbols = {
            str(symbol)
            for candidate in self.candidate_by_id.values()
            for symbol in candidate["symbols"]
        }
        if extra_symbols:
            cand_symbols.update(str(s) for s in extra_symbols)
        symbols = sorted(cand_symbols)
        if not symbols:
            raise ValueError(
                "cohesive projection index requires at least one candidate or extra_symbols to resolve element vdW radii"
            )
        radii = _resolved_ase_vdw_radii(symbols)
        extra = 0.0
        maximum_radius = max(radii.values())
        for left in symbols:
            for right in symbols:
                radius_sum = radii[left] + radii[right]
                hard = thresholds[
                    "H-H"
                    if left == right == "H"
                    else ("H-heavy" if "H" in (left, right) else "heavy-heavy")
                ]
                exact_cutoff = max(contact_scale * radius_sum, hard)
                extra = max(extra, exact_cutoff - projection_scale * radius_sum)
        boundary_samples = int(
            getattr(projection_cache, "contract", {}).get(
                "boundary_samples_per_atom", 720
            )
        )
        if boundary_samples < 12:
            raise ValueError("projection boundary sampling is too small")
        chord_sagitta = (
            projection_scale
            * maximum_radius
            * (1.0 - math.cos(math.pi / boundary_samples))
        )
        self.physical_shell_A = float(max(0.0, extra) + 2.0 * chord_sagitta)
        self.fractional_shell_halfwidths = (
            self.physical_shell_A * np.linalg.norm(inverse, axis=1)
        )
        self.accepted_ids = []
        self._tree = None
        self._piece_to_candidate_id = []
        self._geometry_id_to_candidate_id = {}
        self._candidate_perimeter_A = {}

    def _rebuild(self):
        from shapely.strtree import STRtree

        pieces = []
        piece_ids = []
        for candidate_id in self.accepted_ids:
            candidate = self.candidate_by_id[candidate_id]
            geometry = self.projection_cache.candidate_geometries[
                candidate["candidate_index"]
            ]
            for piece in _cohesive_polygon_pieces(geometry):
                pieces.append(piece)
                piece_ids.append(candidate_id)
        self._tree = STRtree(pieces) if pieces else None
        self._piece_to_candidate_id = piece_ids
        self._geometry_id_to_candidate_id = {
            id(piece): candidate_id for piece, candidate_id in zip(pieces, piece_ids)
        }

    def add(self, candidate):
        candidate_id = candidate["candidate_id"]
        if candidate_id not in self.candidate_by_id:
            raise ValueError("accepted candidate is outside projection index domain")
        if candidate_id in set(self.accepted_ids):
            raise ValueError("accepted candidate already exists in projection index")
        self.accepted_ids.append(candidate_id)
        self._rebuild()

    def _query_box_ids(self, query_box):
        if self._tree is None:
            return set()
        matches = self._tree.query(query_box)
        if len(matches) == 0:
            return set()
        first = matches[0]
        if isinstance(first, (int, np.integer)):
            return {
                self._piece_to_candidate_id[int(index)] for index in matches
            }
        return {
            self._geometry_id_to_candidate_id[id(geometry)] for geometry in matches
        }

    def query(self, candidate):
        _, _, _, box, _ = _require_shapely()
        geometry = self.projection_cache.candidate_geometries[
            candidate["candidate_index"]
        ]
        delta_u, delta_v = self.fractional_shell_halfwidths
        neighbor_ids = set()
        query_box_count = 0
        for piece in _cohesive_polygon_pieces(geometry):
            min_u, min_v, max_u, max_v = piece.bounds
            u_intervals = _cohesive_periodic_intervals(
                min_u - delta_u, max_u + delta_u
            )
            v_intervals = _cohesive_periodic_intervals(
                min_v - delta_v, max_v + delta_v
            )
            for u0, u1 in u_intervals:
                for v0, v1 in v_intervals:
                    query_box_count += 1
                    neighbor_ids.update(self._query_box_ids(box(u0, v0, u1, v1)))
        ordered_ids = sorted(neighbor_ids)
        return (
            [self.candidate_by_id[candidate_id] for candidate_id in ordered_ids],
            {
                "global_accepted_count": len(self.accepted_ids),
                "local_neighbor_count": len(ordered_ids),
                "local_neighbor_candidate_ids": ordered_ids,
                "query_box_count": query_box_count,
                "physical_shell_A": self.physical_shell_A,
                "fractional_shell_halfwidths": self.fractional_shell_halfwidths.tolist(),
                "zero_false_negative_contract": (
                    "projected_vdw_envelope_plus_exact_cutoff_remainder_and_chord_sagitta"
                ),
            },
        )

    def perimeter_trial(self, candidate, local_neighbors):
        """Return exact torus perimeter gain after distant cancellation."""
        _, _, _, _, unary_union = _require_shapely()
        lattice = self.projection_cache.surface_lattice_uv_A
        candidate_index = candidate["candidate_index"]
        candidate_geometry = self.projection_cache.candidate_geometries[
            candidate_index
        ]
        if candidate_index not in self._candidate_perimeter_A:
            value, _ = periodic_union_increment_perimeter(
                None, candidate_geometry, lattice
            )
            self._candidate_perimeter_A[candidate_index] = float(value)
        candidate_perimeter_A = self._candidate_perimeter_A[candidate_index]
        local_geometries = [
            self.projection_cache.candidate_geometries[item["candidate_index"]]
            for item in local_neighbors
        ]
        local_union = unary_union(local_geometries) if local_geometries else None
        if local_union is None:
            before_A = 0.0
        else:
            before_A, _ = periodic_union_increment_perimeter(
                None, local_union, lattice
            )
        increment_A, _ = periodic_union_increment_perimeter(
            local_union, candidate_geometry, lattice
        )
        after_A = before_A + increment_A
        gain_A = cohesive_perimeter_condensation_gain(
            before_A, candidate_perimeter_A, after_A
        )
        return {
            "before_perimeter_A": float(before_A),
            "candidate_perimeter_A": float(candidate_perimeter_A),
            "after_perimeter_A": float(after_A),
            "increment_A": float(increment_A),
            "perimeter_condensation_gain": float(gain_A),
            "perimeter_scope": "local_projection_neighbors_with_distant_exact_cancellation",
            "local_neighbor_candidate_ids": sorted(
                item["candidate_id"] for item in local_neighbors
            ),
        }


class CandidateCollisionProjectionIndex:
    """Static periodic STRtree over candidate projection footprints for hard-collision screening.

    Zero-false-negative local spatial filter built once from candidate projection
    geometries on a periodic surface lattice. For any newly chosen candidate,
    only candidates whose projected footprints fall within the conservative hard
    collision reach across periodic boundaries are returned for exact collision
    evaluation. Distant candidates are provably collision-free and are retained
    without performing exact collision checks.
    """

    def __init__(
        self,
        *,
        projection_cache,
        candidates,
        collision_thresholds_A,
    ):
        from shapely.strtree import STRtree

        self.projection_cache = projection_cache
        candidates = list(candidates)
        if not candidates:
            raise ValueError("candidate collision index requires at least one candidate")
        self.candidate_by_id = {}
        for candidate in candidates:
            cid = str(candidate.get("candidate_id", candidate.get("candidate_index")))
            if cid in self.candidate_by_id:
                raise ValueError(
                    f"candidate collision index candidate IDs must be unique: duplicate {cid!r}"
                )
            self.candidate_by_id[cid] = candidate

        lattice = np.asarray(projection_cache.surface_lattice_uv_A, dtype=float)
        if lattice.shape != (2, 2) or not np.all(np.isfinite(lattice)):
            raise ValueError("candidate collision index requires a finite 2x2 lattice")
        det = float(np.linalg.det(lattice))
        if not np.isfinite(det) or abs(det) < 1.0e-12:
            raise ValueError("candidate collision index requires a nonsingular 2x2 lattice")
        inverse = np.linalg.inv(lattice)

        projection_scale = float(
            getattr(projection_cache, "contract", {}).get("radius_scale", 1.0)
        )
        if not np.isfinite(projection_scale) or projection_scale <= 0.0:
            raise ValueError("projection radius scale must be finite and positive")
        boundary_samples = int(
            getattr(projection_cache, "contract", {}).get(
                "boundary_samples_per_atom", 720
            )
        )
        if boundary_samples < 12:
            raise ValueError("projection boundary sampling is too small")

        thresholds = _validate_collision_thresholds(collision_thresholds_A)
        symbols = sorted(
            {
                str(symbol)
                for candidate in self.candidate_by_id.values()
                for symbol in candidate["symbols"]
            }
        )
        radii = _resolved_ase_vdw_radii(symbols)
        maximum_radius = max(radii.values())
        extra = 0.0
        for left in symbols:
            for right in symbols:
                radius_sum = radii[left] + radii[right]
                hard = thresholds[
                    "H-H"
                    if left == right == "H"
                    else ("H-heavy" if "H" in (left, right) else "heavy-heavy")
                ]
                extra = max(extra, hard - projection_scale * radius_sum)

        chord_sagitta = (
            projection_scale
            * maximum_radius
            * (1.0 - math.cos(math.pi / boundary_samples))
        )
        self.physical_shell_A = float(max(0.0, extra) + 2.0 * chord_sagitta)
        self.fractional_shell_halfwidths = (
            self.physical_shell_A * np.linalg.norm(inverse, axis=1)
        )

        pieces = []
        piece_ids = []
        for cid, candidate in self.candidate_by_id.items():
            candidate_index = candidate["candidate_index"]
            geometry = self.projection_cache.candidate_geometries[candidate_index]
            for piece in _cohesive_polygon_pieces(geometry):
                pieces.append(piece)
                piece_ids.append(cid)

        self._tree = STRtree(pieces) if pieces else None
        self._piece_to_candidate_id = piece_ids
        self._geometry_id_to_candidate_id = {
            id(piece): cid for piece, cid in zip(pieces, piece_ids)
        }

    def _query_box_ids(self, query_box):
        if self._tree is None:
            return set()
        matches = self._tree.query(query_box)
        if len(matches) == 0:
            return set()
        first = matches[0]
        if isinstance(first, (int, np.integer)):
            return {
                self._piece_to_candidate_id[int(index)] for index in matches
            }
        return {
            self._geometry_id_to_candidate_id[id(geometry)] for geometry in matches
        }

    def query_candidate_ids(self, candidate) -> set[str]:
        """Return candidate IDs whose projected footprint could collide with candidate."""
        _, _, _, box, _ = _require_shapely()
        candidate_index = candidate["candidate_index"]
        geometry = self.projection_cache.candidate_geometries[candidate_index]
        delta_u, delta_v = self.fractional_shell_halfwidths
        neighbor_ids = set()
        for piece in _cohesive_polygon_pieces(geometry):
            min_u, min_v, max_u, max_v = piece.bounds
            u_intervals = _cohesive_periodic_intervals(
                min_u - delta_u, max_u + delta_u
            )
            v_intervals = _cohesive_periodic_intervals(
                min_v - delta_v, max_v + delta_v
            )
            for u0, u1 in u_intervals:
                for v0, v1 in v_intervals:
                    neighbor_ids.update(self._query_box_ids(box(u0, v0, u1, v1)))
        own_id = str(candidate.get("candidate_id", candidate.get("candidate_index")))
        neighbor_ids.discard(own_id)
        return neighbor_ids

    def query(self, candidate):
        """Return candidates and diagnostic audit that may collide with candidate."""
        neighbor_ids = self.query_candidate_ids(candidate)
        ordered_ids = sorted(neighbor_ids)
        return (
            [self.candidate_by_id[candidate_id] for candidate_id in ordered_ids],
            {
                "indexed_candidate_count": len(self.candidate_by_id),
                "local_candidate_count": len(ordered_ids),
                "local_candidate_ids": ordered_ids,
                "physical_shell_A": self.physical_shell_A,
                "fractional_shell_halfwidths": self.fractional_shell_halfwidths.tolist(),
                "zero_false_negative_contract": (
                    "projected_envelope_plus_hard_threshold_remainder_and_chord_sagitta"
                ),
            },
        )


# Compatibility seam for callers/tests that imported the former private name.
_choose_projection_candidate = choose_projection_candidate


def _capacity_beam(records, *, beam_width, area_tie_tolerance_A2):
    """Return a deterministic static-debt/hole beam with full boundary closure."""

    if isinstance(beam_width, bool) or int(beam_width) != beam_width or int(beam_width) <= 0:
        raise ValueError("capacity beam width must be a positive integer")
    beam_width = int(beam_width)
    if (
        not isinstance(area_tie_tolerance_A2, (int, float))
        or not np.isfinite(area_tie_tolerance_A2)
        or float(area_tie_tolerance_A2) < 0.0
    ):
        raise ValueError("area_tie_tolerance_A2 must be finite and non-negative")
    tolerance = float(area_tie_tolerance_A2)
    if not records:
        raise ValueError("capacity beam requires at least one candidate record")
    candidate_ids = [str(record["candidate_id"]) for record in records]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("capacity beam candidate IDs must be unique")
    for record in records:
        debt = record["static_debt"]
        hole = float(record["total_hole_area_after_trial_A2"])
        if isinstance(debt, bool) or int(debt) != debt or int(debt) < 0:
            raise ValueError("capacity beam static_debt must be a non-negative integer")
        if not np.isfinite(hole) or hole < 0.0:
            raise ValueError("capacity beam total hole areas must be finite and non-negative")
    ordered = sorted(
        records,
        key=lambda record: (
            int(record["static_debt"]),
            float(record["total_hole_area_after_trial_A2"]),
            str(record["candidate_id"]),
        ),
    )
    base_count = min(beam_width, len(ordered))
    beam = list(ordered[:base_count])
    closure = []
    boundary = ordered[base_count - 1]
    if base_count < len(ordered):
        boundary_debt = int(boundary["static_debt"])
        boundary_hole = float(boundary["total_hole_area_after_trial_A2"])
        for record in ordered[base_count:]:
            if int(record["static_debt"]) != boundary_debt:
                break
            if (
                float(record["total_hole_area_after_trial_A2"])
                <= boundary_hole + tolerance
            ):
                closure.append(record)
            else:
                break
        beam.extend(closure)
    boundary_effective = len(beam) < len(ordered)
    audit = {
        "requested_beam_width": beam_width,
        "full_domain_candidate_count": len(ordered),
        "base_beam_candidate_count": base_count,
        "beam_candidate_count_after_tie_closure": len(beam),
        "base_beam_candidate_ids": [record["candidate_id"] for record in ordered[:base_count]],
        "tie_closure_candidate_ids": [record["candidate_id"] for record in closure],
        "tie_closure_added_count": len(closure),
        "beam_boundary_effective": boundary_effective,
        "beam_boundary_candidate_id": (
            boundary["candidate_id"] if boundary_effective else None
        ),
        "beam_boundary_static_debt": (
            int(boundary["static_debt"]) if boundary_effective else None
        ),
        "beam_boundary_total_hole_area_A2": (
            float(boundary["total_hole_area_after_trial_A2"])
            if boundary_effective
            else None
        ),
        "beam_boundary_area_tolerance_A2": tolerance,
        "beam_boundary_tie_truncated": False if boundary_effective else None,
        "candidates_outside_beam_count": len(ordered) - len(beam),
        "optimization_scope": "exact_one_step_within_deterministic_beam",
        "global_candidate_optimality": False,
        "global_packing_certificate": False,
    }
    return beam, audit


def _select_capacity_finalist(
    records,
    *,
    conformer_weights,
    rng,
    area_tie_tolerance_A2,
    perimeter_tie_tolerance_A,
):
    """Apply the unweighted capacity-aware lexicographic final selection."""

    if not records:
        raise ValueError("capacity-aware final selection requires beam records")
    area_tolerance = float(area_tie_tolerance_A2)
    perimeter_tolerance = float(perimeter_tie_tolerance_A)
    if not np.isfinite(area_tolerance) or area_tolerance < 0.0:
        raise ValueError("area_tie_tolerance_A2 must be finite and non-negative")
    if not np.isfinite(perimeter_tolerance) or perimeter_tolerance <= 0.0:
        raise ValueError("perimeter_tie_tolerance_A must be finite and positive")
    metric_fields = (
        "total_hole_area_after_trial_A2",
        "maximum_torus_hole_area_after_trial_A2",
        "total_physical_hole_perimeter_after_trial_A",
    )
    for record in records:
        debt = record["total_debt"]
        if isinstance(debt, bool) or int(debt) != debt or int(debt) < 0:
            raise ValueError("capacity-aware total_debt must be a non-negative integer")
        for field_name in metric_fields:
            value = float(record[field_name])
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and non-negative")

    debt_minimum = min(int(record["total_debt"]) for record in records)
    debt_finalists = [
        record for record in records if int(record["total_debt"]) == debt_minimum
    ]

    def tolerance_finalists(current, field_name, tolerance):
        minimum = min(float(record[field_name]) for record in current)
        return (
            [
                record
                for record in current
                if float(record[field_name]) <= minimum + tolerance
            ],
            minimum,
        )

    total_hole_finalists, total_hole_minimum = tolerance_finalists(
        debt_finalists, "total_hole_area_after_trial_A2", area_tolerance
    )
    maximum_hole_finalists, maximum_hole_minimum = tolerance_finalists(
        total_hole_finalists,
        "maximum_torus_hole_area_after_trial_A2",
        area_tolerance,
    )
    perimeter_finalists, perimeter_minimum = tolerance_finalists(
        maximum_hole_finalists,
        "total_physical_hole_perimeter_after_trial_A",
        perimeter_tolerance,
    )
    rank_probabilities = {}
    prior_reached = len(perimeter_finalists) > 1
    if not prior_reached:
        winner = perimeter_finalists[0]
    else:
        final_ranks = sorted(
            {int(record["selection_rank"]) for record in perimeter_finalists}
        )
        missing = [rank for rank in final_ranks if rank not in conformer_weights]
        if missing:
            raise ValueError(f"Missing conformer weights for ranks {missing}")
        weights = {rank: float(conformer_weights[rank]) for rank in final_ranks}
        if any(not np.isfinite(value) or value <= 0.0 for value in weights.values()):
            raise ValueError("capacity-aware conformer weights must be finite and positive")
        denominator = sum(weights.values())
        rank_probabilities = {
            str(rank): weights[rank] / denominator for rank in final_ranks
        }
        selected_rank = _weighted_rank_choice(rng, final_ranks, weights)
        rank_finalists = sorted(
            (
                record
                for record in perimeter_finalists
                if int(record["selection_rank"]) == selected_rank
            ),
            key=lambda record: str(record["candidate_id"]),
        )
        winner = rank_finalists[rng.randrange(len(rank_finalists))]
    audit = {
        "lexicographic_objectives": [
            {"name": "total_debt", "direction": "minimize", "comparison": "exact_integer"},
            {
                "name": "total_hole_area_after_trial_A2",
                "direction": "minimize",
                "tie_tolerance": area_tolerance,
                "unit": "A2",
            },
            {
                "name": "maximum_torus_hole_area_after_trial_A2",
                "direction": "minimize",
                "tie_tolerance": area_tolerance,
                "unit": "A2",
            },
            {
                "name": "total_physical_hole_perimeter_after_trial_A",
                "direction": "minimize",
                "tie_tolerance": perimeter_tolerance,
                "unit": "A",
            },
            {
                "name": "approved_conformer_prior_then_seed",
                "direction": "tie_break_only",
                "weighted_objective": False,
            },
        ],
        "minimum_total_debt": debt_minimum,
        "total_debt_finalist_candidate_ids": sorted(
            record["candidate_id"] for record in debt_finalists
        ),
        "minimum_total_hole_area_after_trial_A2": total_hole_minimum,
        "total_hole_finalist_candidate_ids": sorted(
            record["candidate_id"] for record in total_hole_finalists
        ),
        "minimum_maximum_torus_hole_area_after_trial_A2": maximum_hole_minimum,
        "maximum_hole_finalist_candidate_ids": sorted(
            record["candidate_id"] for record in maximum_hole_finalists
        ),
        "minimum_total_physical_hole_perimeter_after_trial_A": perimeter_minimum,
        "perimeter_finalist_candidate_ids": sorted(
            record["candidate_id"] for record in perimeter_finalists
        ),
        "prior_reached": prior_reached,
        "rank_probabilities": rank_probabilities,
        "winner_candidate_id": winner["candidate_id"],
        "weighted_score_used": False,
    }
    return winner, audit


def choose_capacity_candidate(
    *,
    remaining_domains,
    site_conflicts,
    metal_constraints,
    cell,
    periodic_axes,
    collision_thresholds_A,
    accepted_union,
    projection_cache,
    conformer_weights,
    rng,
    capacity_beam_width,
    area_tie_tolerance_A2,
    perimeter_tie_tolerance_A,
    capacity_time_limit_seconds=30.0,
    collision_cache=None,
    collision_profiler=None,
) -> tuple[dict, dict]:
    """Evaluate the exact capacity-aware one-step objective within its beam."""

    started = time.perf_counter()
    thresholds_A = _validate_collision_thresholds(collision_thresholds_A)
    feasible_sites = sorted(
        site_id for site_id, candidates in remaining_domains.items() if candidates
    )
    if not feasible_sites:
        raise ValueError("capacity-aware selection requires a live candidate domain")
    if projection_cache is None:
        raise ValueError("capacity-aware selection requires a ProjectionCache")
    if projection_cache.site_ids != sorted(remaining_domains):
        raise ValueError("projection cache Site Instance order mismatch")
    candidate_pool = sorted(
        (
            candidate
            for site_id in feasible_sites
            for candidate in remaining_domains[site_id]
        ),
        key=lambda candidate: str(candidate["candidate_id"]),
    )
    required_fields = (
        "candidate_id",
        "candidate_index",
        "site_instance_id",
        "selection_rank",
        "coordinates",
        "symbols",
    )
    for candidate in candidate_pool:
        for field_name in required_fields:
            if field_name not in candidate:
                raise ValueError(
                    f"Candidate missing required capacity field {field_name}"
                )
    candidate_ids = [candidate["candidate_id"] for candidate in candidate_pool]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("capacity-aware candidate IDs must be unique")

    capacity_cache = {}
    capacity_stats = {
        "request_count": 0,
        "cache_hit_count": 0,
        "solve_count": 0,
        "total_solver_seconds": 0.0,
    }

    def capacity_query(site_ids):
        key = tuple(sorted(str(site_id) for site_id in site_ids))
        capacity_stats["request_count"] += 1
        if key in capacity_cache:
            capacity_stats["cache_hit_count"] += 1
            return capacity_cache[key], True
        result = solve_static_conflict_capacity(
            key,
            metal_constraints,
            site_conflicts=site_conflicts,
            time_limit_seconds=capacity_time_limit_seconds,
        )
        capacity_cache[key] = result
        capacity_stats["solve_count"] += 1
        capacity_stats["total_solver_seconds"] += result["solve_time_seconds"]
        return result, False

    before_capacity, _ = capacity_query(feasible_sites)
    K_before = int(before_capacity["objective"])
    site_index = {
        site_id: index for index, site_id in enumerate(projection_cache.site_ids)
    }
    static_by_site = {}
    static_groups = {}
    static_started = time.perf_counter()
    for site_id in feasible_sites:
        blocked = _static_blocked_site_ids(
            site_id,
            remaining_domains,
            site_conflicts,
            metal_constraints,
        )
        static_live = tuple(
            other_site
            for other_site in feasible_sites
            if other_site not in blocked
        )
        capacity_result, cache_hit = capacity_query(static_live)
        K_after_static = int(capacity_result["objective"])
        static_debt = K_before - 1 - K_after_static
        if static_debt < 0:
            raise RuntimeError("exact static capacity produced a negative static debt")
        static_live_mask = _sites_mask(static_live, site_index)
        static_by_site[site_id] = {
            "static_blocked_site_ids": sorted(blocked),
            "static_live_site_ids": list(static_live),
            "static_live_site_mask": static_live_mask,
            "K_after_static": K_after_static,
            "static_debt": static_debt,
            "capacity_milp": capacity_result,
            "capacity_cache_hit": cache_hit,
        }
        group = static_groups.setdefault(
            static_live,
            {
                "static_live_mask_id": f"static-mask-{len(static_groups):04d}",
                "static_live_site_mask_hex": hex(static_live_mask),
                "static_live_site_count": len(static_live),
                "site_instance_ids": [],
                "K_after_static": K_after_static,
                "capacity_milp": capacity_result,
            },
        )
        static_by_site[site_id]["static_live_mask_id"] = group[
            "static_live_mask_id"
        ]
        group["site_instance_ids"].append(site_id)
    static_seconds = time.perf_counter() - static_started

    projection_started = time.perf_counter()
    all_records = []
    for candidate in candidate_pool:
        site_id = str(candidate["site_instance_id"])
        if site_id not in static_by_site:
            raise ValueError("candidate belongs to a non-live Site Instance")
        geometry = projection_cache.candidate_geometries[
            candidate["candidate_index"]
        ]
        _, merged = periodic_union_increment_A2(
            accepted_union, geometry, projection_cache.area_scale_A2
        )
        total_hole_A2 = (
            1.0 - float(merged.area)
        ) * float(projection_cache.area_scale_A2)
        if total_hole_A2 < -1.0e-8:
            raise RuntimeError("trial projection union exceeds the periodic cell area")
        total_hole_A2 = max(0.0, total_hole_A2)
        all_records.append(
            {
                "candidate_id": candidate["candidate_id"],
                "candidate_index": int(candidate["candidate_index"]),
                "site_instance_id": site_id,
                "site_prototype_id": candidate.get("site_prototype_id"),
                "selection_rank": int(candidate["selection_rank"]),
                "static_live_mask_id": static_by_site[site_id][
                    "static_live_mask_id"
                ],
                "K_before": K_before,
                "K_after_static": static_by_site[site_id]["K_after_static"],
                "static_debt": static_by_site[site_id]["static_debt"],
                "total_hole_area_after_trial_A2": float(total_hole_A2),
                "_candidate": candidate,
            }
        )
    projection_seconds = time.perf_counter() - projection_started
    beam_records, beam_audit = _capacity_beam(
        all_records,
        beam_width=capacity_beam_width,
        area_tie_tolerance_A2=area_tie_tolerance_A2,
    )
    base_beam_ids = set(beam_audit["base_beam_candidate_ids"])
    closure_ids = set(beam_audit["tie_closure_candidate_ids"])
    beam_ids = base_beam_ids | closure_ids

    preview_started = time.perf_counter()
    full_records = []
    preview_query_count = 0
    preview_query_seconds = 0.0
    preview_worst_case_queries = 0
    torus_seconds = 0.0
    for record in beam_records:
        candidate = record["_candidate"]
        static_record = static_by_site[record["site_instance_id"]]
        preview_worst_case_queries += sum(
            len(remaining_domains[site_id])
            for site_id in static_record["static_live_site_ids"]
        )
        transition = simulate_candidate_transition(
            remaining_domains=remaining_domains,
            chosen=candidate,
            site_conflicts=site_conflicts,
            metal_constraints=metal_constraints,
            cell=cell,
            periodic_axes=periodic_axes,
            collision_thresholds_A=thresholds_A,
            mode="preview",
            capacity_time_limit_seconds=capacity_time_limit_seconds,
            capacity_query=capacity_query,
            collision_cache=collision_cache,
            collision_profiler=collision_profiler,
        )
        if transition["static_live_site_ids"] != static_record["static_live_site_ids"]:
            raise RuntimeError("capacity preview static live-site set disagrees with beam audit")
        if transition["K_after_static"] != record["K_after_static"]:
            raise RuntimeError("capacity preview K_after_static disagrees with cached value")
        debt = capacity_debt_metrics(
            K_before=K_before,
            K_after_static=transition["K_after_static"],
            K_after_collision=transition["K_after_collision"],
        )
        geometry = projection_cache.candidate_geometries[
            candidate["candidate_index"]
        ]
        _, trial_union = periodic_union_increment_A2(
            accepted_union, geometry, projection_cache.area_scale_A2
        )
        torus_started = time.perf_counter()
        torus = periodic_hole_diagnostics(
            trial_union,
            projection_cache.area_scale_A2,
            surface_lattice_uv_A=projection_cache.surface_lattice_uv_A,
        )
        torus_seconds += time.perf_counter() - torus_started
        if torus["metric_version"] != "periodic-torus-holes-v2":
            raise RuntimeError("capacity-aware scoring requires periodic-torus-holes-v2")
        if torus["total_hole_perimeter_A"] is None:
            raise RuntimeError("capacity-aware scoring requires physical hole perimeter")
        if abs(
            float(torus["total_hole_area_A2"])
            - float(record["total_hole_area_after_trial_A2"])
        ) > max(1.0e-8, float(area_tie_tolerance_A2)):
            raise RuntimeError("trial union area disagrees with torus hole diagnostics")
        preview_query_count += transition["collision_query_count"]
        preview_query_seconds += transition["collision_query_seconds"]
        full_records.append(
            {
                **{key: value for key, value in record.items() if key != "_candidate"},
                **debt,
                "maximum_torus_hole_area_after_trial_A2": float(
                    torus["maximum_hole_area_A2"]
                ),
                "total_physical_hole_perimeter_after_trial_A": float(
                    torus["total_hole_perimeter_A"]
                ),
                "torus_hole_diagnostics": {
                    "metric_version": torus["metric_version"],
                    "hole_count": int(torus["hole_count"]),
                    "total_hole_area_A2": float(torus["total_hole_area_A2"]),
                    "maximum_hole_area_A2": float(torus["maximum_hole_area_A2"]),
                    "total_hole_perimeter_A": float(torus["total_hole_perimeter_A"]),
                    "perimeter_metric": torus["perimeter_metric"],
                },
                "collision_preview": {
                    "live_site_ids": transition["live_site_ids"],
                    "live_site_count": len(transition["live_site_ids"]),
                    "collision_query_count": transition["collision_query_count"],
                    "direct_collision_query_count": transition[
                        "direct_collision_query_count"
                    ],
                    "cached_collision_query_count": transition[
                        "cached_collision_query_count"
                    ],
                    "collision_query_seconds": transition[
                        "collision_query_seconds"
                    ],
                    "short_circuit_after_first_compatible_per_site": True,
                },
                "capacity_milp_after_static": transition[
                    "capacity_after_static"
                ],
                "capacity_milp_after_collision": transition[
                    "capacity_after_collision"
                ],
                "_candidate": candidate,
            }
        )
    preview_seconds = time.perf_counter() - preview_started
    winner_record, finalist_audit = _select_capacity_finalist(
        full_records,
        conformer_weights=conformer_weights,
        rng=rng,
        area_tie_tolerance_A2=area_tie_tolerance_A2,
        perimeter_tie_tolerance_A=perimeter_tie_tolerance_A,
    )
    winner = winner_record["_candidate"]

    all_candidate_audit = []
    for record in all_records:
        candidate_id = record["candidate_id"]
        all_candidate_audit.append(
            {
                **{key: value for key, value in record.items() if key != "_candidate"},
                "in_base_beam": candidate_id in base_beam_ids,
                "in_beam_tie_closure": candidate_id in closure_ids,
                "in_evaluated_beam": candidate_id in beam_ids,
            }
        )
    full_record_audit = [
        {key: value for key, value in record.items() if key != "_candidate"}
        for record in full_records
    ]
    capacity_summary = {
        **capacity_stats,
        "unique_live_site_set_count": len(capacity_cache),
        "all_results_certified_optimal": all(
            result["status"] == "optimal" and result["optimality_certified"]
            for result in capacity_cache.values()
        ),
    }
    audit = {
        "policy": "capacity-aware",
        "optimization_scope": "exact_one_step_within_deterministic_beam",
        "global_candidate_optimality": False,
        "global_packing_certificate": False,
        "capacity_bound_interpretation": {
            "one_plus_K_after_collision": (
                "shared-metal/individual-feasibility upper bound only"
            ),
            "includes_future_candidate_pair_collisions": False,
            "realizable_terminal_state_claim": False,
            "global_packing_claim": False,
        },
        "live_site_count_before_step": len(feasible_sites),
        "full_domain_candidate_count": len(candidate_pool),
        "K_before": K_before,
        "capacity_milp_before": before_capacity,
        "projection_site_order": list(projection_cache.site_ids),
        "site_mask_encoding": {
            "format": "hexadecimal_integer_lsb0",
            "bit_index_source": "projection_site_order",
            "reconstruction": "bit_i_set_means_projection_site_order_i_is_live",
        },
        "static_live_mask_unique_count": len(static_groups),
        "static_capacity_by_unique_live_mask": [
            {**group, "site_instance_count": len(group["site_instance_ids"])}
            for group in static_groups.values()
        ],
        "all_candidates": all_candidate_audit,
        "beam": beam_audit,
        "beam_candidates": full_record_audit,
        "final_selection": finalist_audit,
        "winner_candidate_id": winner["candidate_id"],
        "collision_thresholds_A": thresholds_A,
        "capacity_milp_queries": capacity_summary,
        "timing": {
            "static_capacity_seconds": static_seconds,
            "all_candidate_true_union_area_seconds": projection_seconds,
            "beam_preview_and_torus_seconds": preview_seconds,
            "beam_torus_diagnostics_seconds": torus_seconds,
            "beam_collision_query_seconds": preview_query_seconds,
            "total_selection_seconds": time.perf_counter() - started,
        },
        "query_counts": {
            "all_candidate_true_union_area_query_count": len(candidate_pool),
            "beam_torus_diagnostics_query_count": len(full_records),
            "beam_preview_collision_query_count": preview_query_count,
            "beam_preview_collision_query_upper_bound": preview_worst_case_queries,
        },
        "complexity_bound": {
            "symbols": {
                "S": len(feasible_sites),
                "N": len(candidate_pool),
                "U": len(static_groups),
                "B": len(full_records),
            },
            "capacity_milp_solve_upper_bound_including_winner_commit": (
                1 + len(static_groups) + len(full_records) + 2
            ),
            "preview_collision_query_upper_bound": preview_worst_case_queries,
            "winner_materialize_collision_query_upper_bound": len(candidate_pool),
            "asymptotic": (
                "O(N*G_union + U*MILP(S) + B*(N*Q_collision + "
                "MILP(S) + G_torus) + N*Q_collision_for_commit)"
            ),
            "future_candidate_pair_collision_matrix_built": False,
        },
    }
    return winner, audit


def cohesive_active_denticity(candidates):
    """Return the active registered denticity under the strict 3 -> 2 -> 1 order."""
    ordered = _cohesive_candidate_records(candidates)
    denticities = set()
    for candidate in ordered:
        if candidate.get("site_kind") == "single_metal_triangle" or int(candidate.get("phase_priority", 0)) == 1:
            denticities.add(1)
            continue
        labels = candidate.get("target_donor_labels")
        if not isinstance(labels, (list, tuple)):
            raise ValueError("cohesive candidate lacks target_donor_labels")
        denticity = len(labels)
        if denticity not in (1, 2, 3):
            raise ValueError("cohesive candidate denticity must be 1, 2, or 3")
        denticities.add(denticity)
    return next((value for value in (3, 2, 1) if value in denticities), None)


def _cohesive_candidate_area_record(candidate, projection_cache):
    geometry = projection_cache.candidate_geometries[candidate["candidate_index"]]
    area_A2, _ = periodic_union_increment_A2(
        None, geometry, projection_cache.area_scale_A2
    )
    record = dict(candidate)
    record["projected_area_A2"] = float(area_A2)
    return record


def _cohesive_energy_record(candidate):
    record = dict(candidate)
    record["conformer_energy_eV"] = _cohesive_real(
        candidate.get("conformer_energy_eV"),
        f"conformer_energy_eV[{candidate['candidate_id']}]",
    )
    return record


def _cohesive_seed_choice(finalists, rng):
    ordered = _cohesive_candidate_records(finalists)
    if not ordered:
        raise RuntimeError("cohesive selection produced no finalist")
    return dict(ordered[rng.randrange(len(ordered))])


def extract_frontier_boundary_rings(accepted_union, surface_lattice_uv_A=None) -> list[dict]:
    """Extract exterior boundary rings and interior hole rings from accepted_union."""
    if accepted_union is None or getattr(accepted_union, "is_empty", True):
        return []
    _, _, Polygon, _, _ = _require_shapely()
    polygons = accepted_union.geoms if hasattr(accepted_union, "geoms") else [accepted_union]
    rings = []
    det_scale = 1.0
    if surface_lattice_uv_A is not None:
        surface_lattice = np.asarray(surface_lattice_uv_A, dtype=float)
        if surface_lattice.shape == (2, 2):
            min_x, min_y, max_x, max_y = accepted_union.bounds
            if not (max_x > 1.0 + 1e-4 or max_y > 1.0 + 1e-4 or min_x < -1e-4 or min_y < -1e-4):
                det_scale = abs(float(np.linalg.det(surface_lattice)))

    for poly in polygons:
        if poly.is_empty:
            continue
        rings.append({
            "ring": poly.exterior,
            "target_type": "exterior",
            "hole_area_A2": 0.0,
        })
        for interior in poly.interiors:
            if interior.is_empty:
                continue
            hole_area = float(Polygon(interior).area) * det_scale
            rings.append({
                "ring": interior,
                "target_type": "hole",
                "hole_area_A2": float(hole_area),
            })
    return rings


def _wrap_to_unit_cell(poly):
    """Wrap any polygon extending outside [0, 1] x [0, 1] back into the unit cell."""
    affinity, _, _, box, unary_union = _require_shapely()
    if poly is None or poly.is_empty:
        return poly
    unit_square = box(0.0, 0.0, 1.0, 1.0)
    if unit_square.contains(poly):
        return poly
    min_x, min_y, max_x, max_y = poly.bounds
    u_lo = int(np.floor(min_x - 1e-9))
    u_hi = int(np.ceil(max_x + 1e-9))
    v_lo = int(np.floor(min_y - 1e-9))
    v_hi = int(np.ceil(max_y + 1e-9))
    fragments = []
    for du in range(u_lo, u_hi + 1):
        for dv in range(v_lo, v_hi + 1):
            shifted = affinity.translate(poly, xoff=-float(du), yoff=-float(dv))
            clipped = shifted.intersection(unit_square)
            if not clipped.is_empty and clipped.area > 1e-12:
                fragments.append(clipped)
    if not fragments:
        return poly.intersection(unit_square)
    return unary_union(fragments)


def generate_continuous_frontier_ideal_positions(
    accepted_union,
    candidate_templates: dict | list,
    surface_lattice_uv_A: np.ndarray,
) -> list[dict]:
    """Generate deterministic ideal positions q* on continuous frontier boundary rings.

    Calculates ideal kissing positions q* via outward normals and support vectors
    from exterior and interior hole boundary rings of the accepted projection union.
    Explicitly performs only 2D vector and support point arithmetic without constructing
    virtual candidate polygons, computing virtual polygon intersections, or evaluating
    virtual periodic perimeter increments.
    """
    if accepted_union is None or getattr(accepted_union, "is_empty", True):
        return []
    affinity, Point, Polygon, box, unary_union = _require_shapely()
    if surface_lattice_uv_A is None:
        raise ValueError("surface_lattice_uv_A is required and cannot be None")
    surface_lattice = np.asarray(surface_lattice_uv_A, dtype=float)
    if surface_lattice.shape != (2, 2) or not np.all(np.isfinite(surface_lattice)):
        raise ValueError("surface_lattice_uv_A must be a finite 2x2 matrix")
    det_val = float(np.linalg.det(surface_lattice))
    if not np.isfinite(det_val) or abs(det_val) < 1e-12:
        raise ValueError("surface_lattice_uv_A must be invertible (non-zero determinant)")
    lattice_inv = np.linalg.inv(surface_lattice)

    min_x, min_y, max_x, max_y = accepted_union.bounds
    if max_x > 1.0 + 1e-4 or max_y > 1.0 + 1e-4 or min_x < -1e-4 or min_y < -1e-4:
        matrix_to_frac = [lattice_inv[0, 0], lattice_inv[1, 0], lattice_inv[0, 1], lattice_inv[1, 1], 0.0, 0.0]
        union_frac = affinity.affine_transform(accepted_union, matrix_to_frac)
        union_frac = _wrap_to_unit_cell(union_frac)
    else:
        union_frac = accepted_union

    rings = extract_frontier_boundary_rings(union_frac, surface_lattice)
    if not rings:
        return []

    if isinstance(candidate_templates, list):
        templates_dict = {t["template_key"]: t for t in candidate_templates}
    elif isinstance(candidate_templates, dict):
        templates_dict = candidate_templates
    else:
        templates_dict = {}

    precomputed_templates = []
    for tpl_key, tpl in sorted(templates_dict.items(), key=lambda item: str(item[0])):
        rel_verts_A = np.asarray(tpl["relative_vertices_uv_A"], dtype=float)
        if len(rel_verts_A) < 3:
            continue
        rel_verts_frac = rel_verts_A @ lattice_inv
        precomputed_templates.append({
            "tpl_key": tpl_key,
            "template_record": tpl,
            "rel_verts_A": rel_verts_A,
            "rel_verts_frac": rel_verts_frac,
        })

    if not precomputed_templates:
        return []

    ideal_proposals = []
    seen_keys = set()

    for ring_info in rings:
        ring = ring_info["ring"]
        coords = np.asarray(ring.coords, dtype=float)
        if len(coords) < 3:
            continue
        vertices = coords[:-1] if np.allclose(coords[0], coords[-1]) else coords
        n_verts = len(vertices)
        hole_area_A2 = float(ring_info["hole_area_A2"])
        target_type = ring_info["target_type"]

        for i, vertex_w in enumerate(vertices):
            prev_v = vertices[(i - 1) % n_verts]
            next_v = vertices[(i + 1) % n_verts]
            edge_in = (vertex_w - prev_v) @ surface_lattice
            edge_out = (next_v - vertex_w) @ surface_lattice
            len_in = float(np.linalg.norm(edge_in))
            len_out = float(np.linalg.norm(edge_out))
            if len_in > 1e-12:
                edge_in /= len_in
            if len_out > 1e-12:
                edge_out /= len_out

            n_in = np.array([edge_in[1], -edge_in[0]])
            n_out = np.array([edge_out[1], -edge_out[0]])
            normal_phys = n_in + n_out
            norm_mag = float(np.linalg.norm(normal_phys))
            if norm_mag > 1e-12:
                normal_phys /= norm_mag
            else:
                normal_phys = n_in

            normal_frac = normal_phys @ lattice_inv
            test_pt = Point(vertex_w[0] + 0.01 * normal_frac[0], vertex_w[1] + 0.01 * normal_frac[1])
            if union_frac.contains(test_pt):
                normal_phys = -normal_phys
                normal_frac = -normal_frac

            for item in precomputed_templates:
                tpl_key = item["tpl_key"]
                rel_verts_A = item["rel_verts_A"]
                rel_verts_frac = item["rel_verts_frac"]

                projections = rel_verts_A @ normal_phys
                contact_idx = int(np.argmin(projections))
                u_contact_frac = rel_verts_frac[contact_idx]

                q_frac = vertex_w - u_contact_frac
                q_wrapped_frac = q_frac % 1.0
                q_wrapped_A = q_wrapped_frac @ surface_lattice

                prop_hash = (
                    str(tpl_key),
                    round(float(q_wrapped_A[0]), 4),
                    round(float(q_wrapped_A[1]), 4),
                )
                if prop_hash in seen_keys:
                    continue
                seen_keys.add(prop_hash)

                ideal_proposals.append({
                    "q_uv_A": q_wrapped_A,
                    "q_frac": q_wrapped_frac,
                    "template_key": tpl_key,
                    "target_type": target_type,
                    "hole_area_A2": float(hole_area_A2),
                })

    ideal_proposals.sort(
        key=lambda p: (
            p["target_type"],
            str(p["template_key"]),
            round(float(p["q_uv_A"][0]), 6),
            round(float(p["q_uv_A"][1]), 6),
        )
    )
    return ideal_proposals


def generate_continuous_frontier_proposals(
    accepted_union,
    candidate_templates,
    surface_lattice_uv_A,
    area_scale_A2=1.0,
) -> list[dict]:
    """Generate deterministic kissing proposals q* on the continuous frontier."""
    if accepted_union is None or getattr(accepted_union, "is_empty", True):
        return []
    affinity, Point, Polygon, box, unary_union = _require_shapely()
    if surface_lattice_uv_A is None:
        raise ValueError("surface_lattice_uv_A is required and cannot be None")
    surface_lattice = np.asarray(surface_lattice_uv_A, dtype=float)
    if surface_lattice.shape != (2, 2) or not np.all(np.isfinite(surface_lattice)):
        raise ValueError("surface_lattice_uv_A must be a finite 2x2 matrix")
    det_val = float(np.linalg.det(surface_lattice))
    if not np.isfinite(det_val) or abs(det_val) < 1e-12:
        raise ValueError("surface_lattice_uv_A must be invertible (non-zero determinant)")
    lattice_inv = np.linalg.inv(surface_lattice)
    det_lattice = abs(det_val)

    # Determine whether accepted_union is in fractional or physical Angstrom coordinates
    min_x, min_y, max_x, max_y = accepted_union.bounds
    if max_x > 1.0 + 1e-4 or max_y > 1.0 + 1e-4 or min_x < -1e-4 or min_y < -1e-4:
        matrix_to_frac = [lattice_inv[0, 0], lattice_inv[1, 0], lattice_inv[0, 1], lattice_inv[1, 1], 0.0, 0.0]
        union_frac = affinity.affine_transform(accepted_union, matrix_to_frac)
        union_frac = _wrap_to_unit_cell(union_frac)
    else:
        union_frac = accepted_union

    rings = extract_frontier_boundary_rings(union_frac, surface_lattice)
    if not rings:
        return []

    if isinstance(candidate_templates, list):
        templates_dict = {t["template_key"]: t for t in candidate_templates}
    elif isinstance(candidate_templates, dict):
        templates_dict = candidate_templates
    else:
        templates_dict = {}

    # A1: Template loop invariant precomputation
    precomputed_templates = []
    perimeters = []
    for tpl_key, tpl in sorted(templates_dict.items(), key=lambda item: str(item[0])):
        rel_verts_A = np.asarray(tpl["relative_vertices_uv_A"], dtype=float)
        if len(rel_verts_A) < 3:
            continue
        rel_verts_frac = rel_verts_A @ lattice_inv

        if "projected_area_A2" not in tpl or tpl["projected_area_A2"] is None:
            raise KeyError(f"Template {tpl_key} lacks projected_area_A2")
        proj_area = float(tpl["projected_area_A2"])
        if not np.isfinite(proj_area) or proj_area <= 0.0:
            raise ValueError(f"Template {tpl_key} projected_area_A2 must be finite and positive: {proj_area}")

        if "conformer_energy_eV" not in tpl or tpl["conformer_energy_eV"] is None:
            raise KeyError(f"Template {tpl_key} lacks conformer_energy_eV")
        conf_energy = float(tpl["conformer_energy_eV"])
        if not np.isfinite(conf_energy):
            raise ValueError(f"Template {tpl_key} conformer_energy_eV must be finite: {conf_energy}")

        tpl_poly = Polygon(rel_verts_A)
        perimeters.append(float(tpl_poly.length))

        precomputed_templates.append({
            "tpl_key": tpl_key,
            "template_record": tpl,
            "rel_verts_A": rel_verts_A,
            "rel_verts_frac": rel_verts_frac,
            "projected_area_A2": proj_area,
            "conformer_energy_eV": conf_energy,
        })

    # A3: Single base torus perimeter calculation
    base_perimeter_A = sam_structure_tools.periodic_torus_perimeter_A(
        union_frac,
        surface_lattice,
    )

    raw_proposals = []
    seen_keys = set()

    for ring_info in rings:
        ring = ring_info["ring"]
        coords = np.asarray(ring.coords, dtype=float)
        if len(coords) < 3:
            continue
        vertices = coords[:-1] if np.allclose(coords[0], coords[-1]) else coords
        n_verts = len(vertices)

        hole_area_A2 = float(ring_info["hole_area_A2"])

        for i, vertex_w in enumerate(vertices):
            prev_v = vertices[(i - 1) % n_verts]
            next_v = vertices[(i + 1) % n_verts]
            edge_in = (vertex_w - prev_v) @ surface_lattice
            edge_out = (next_v - vertex_w) @ surface_lattice
            len_in = float(np.linalg.norm(edge_in))
            len_out = float(np.linalg.norm(edge_out))
            if len_in > 1e-12:
                edge_in /= len_in
            if len_out > 1e-12:
                edge_out /= len_out

            n_in = np.array([edge_in[1], -edge_in[0]])
            n_out = np.array([edge_out[1], -edge_out[0]])
            normal_phys = n_in + n_out
            norm_mag = float(np.linalg.norm(normal_phys))
            if norm_mag > 1e-12:
                normal_phys /= norm_mag
            else:
                normal_phys = n_in

            normal_frac = normal_phys @ lattice_inv
            test_pt = Point(vertex_w[0] + 0.01 * normal_frac[0], vertex_w[1] + 0.01 * normal_frac[1])
            if union_frac.contains(test_pt):
                normal_phys = -normal_phys
                normal_frac = -normal_frac

            for item in precomputed_templates:
                tpl_key = item["tpl_key"]
                rel_verts_A = item["rel_verts_A"]
                rel_verts_frac = item["rel_verts_frac"]

                projections = rel_verts_A @ normal_phys
                contact_idx = int(np.argmin(projections))
                u_contact_A = rel_verts_A[contact_idx]
                u_contact_frac = rel_verts_frac[contact_idx]

                q_frac = vertex_w - u_contact_frac
                q_wrapped_frac = q_frac % 1.0
                q_wrapped_A = q_wrapped_frac @ surface_lattice

                cand_poly_unwrapped_frac = Polygon(rel_verts_frac + q_frac)
                cand_poly_frac = _wrap_to_unit_cell(cand_poly_unwrapped_frac)

                intersection_area = cand_poly_frac.intersection(union_frac).area
                if intersection_area > 1e-3 * cand_poly_frac.area:
                    continue

                # A2: Equivalent early deduplication
                prop_hash = (
                    str(tpl_key),
                    round(float(q_wrapped_A[0]), 3),
                    round(float(q_wrapped_A[1]), 3),
                )
                if prop_hash in seen_keys:
                    continue
                seen_keys.add(prop_hash)

                # A3: Single base torus perimeter calculation passed to increment
                delta_L_pre, _ = periodic_union_increment_perimeter(
                    union_frac,
                    cand_poly_frac,
                    surface_lattice,
                    current_union_perimeter_A=base_perimeter_A,
                )

                raw_proposals.append({
                    "q_uv_A": q_wrapped_A,
                    "template_key": tpl_key,
                    "template_record": item["template_record"],
                    "target_type": ring_info["target_type"],
                    "hole_area_A2": float(hole_area_A2),
                    "delta_L_pre_A": float(delta_L_pre),
                    "projected_area_A2": item["projected_area_A2"],
                    "conformer_energy_eV": item["conformer_energy_eV"],
                })

    if not raw_proposals:
        return []

    min_delta_L = min(p["delta_L_pre_A"] for p in raw_proposals)
    min_cand_perimeter = min(perimeters) if perimeters else 10.0
    allowed_excess_L = 0.10 * min_cand_perimeter
    cutoff_L = min_delta_L + allowed_excess_L
    survivors_L = [p for p in raw_proposals if p["delta_L_pre_A"] <= cutoff_L + 1e-8]

    min_area = min(p["projected_area_A2"] for p in survivors_L)
    cutoff_area = min_area * 1.20 if min_area > 0 else 0.0
    survivors_area = [
        p for p in survivors_L
        if min_area <= 0 or p["projected_area_A2"] <= cutoff_area + 1e-8
    ]

    min_energy = min(p["conformer_energy_eV"] for p in survivors_area)
    survivors_energy = [
        p for p in survivors_area
        if p["conformer_energy_eV"] <= min_energy + 0.03 + 1e-8
    ]

    survivors_energy.sort(
        key=lambda p: (
            p["delta_L_pre_A"],
            p["projected_area_A2"],
            p["conformer_energy_eV"],
            str(p["template_key"]),
            tuple(np.round(p["q_uv_A"], 5)),
        )
    )
    return survivors_energy


class SameTemplatePeriodicSpatialIndex:
    """Deterministic periodic spatial index for candidate querying grouped by template."""

    def __init__(
        self,
        candidates: list[dict],
        surface_lattice_uv_A: np.ndarray,
    ):
        if surface_lattice_uv_A is None:
            raise ValueError("surface_lattice_uv_A is required and cannot be None")
        self.surface_lattice = np.asarray(surface_lattice_uv_A, dtype=float)
        if self.surface_lattice.shape != (2, 2) or not np.all(np.isfinite(self.surface_lattice)):
            raise ValueError("surface_lattice_uv_A must be a finite 2x2 matrix")
        det = float(np.linalg.det(self.surface_lattice))
        if not np.isfinite(det) or abs(det) < 1e-12:
            raise ValueError("surface_lattice_uv_A must be invertible (non-zero determinant)")
        self.lattice_inv = np.linalg.inv(self.surface_lattice)

        self.candidates_by_template: dict[str, list[dict]] = {}
        self.anchors_by_template: dict[str, np.ndarray] = {}

        for cand in candidates:
            tpl_key = cand.get("template_key")
            if tpl_key is None:
                continue
            anchor_uv = cand.get("anchor_uv_A")
            if anchor_uv is None:
                raise KeyError(
                    f"Candidate {cand.get('candidate_id')} lacks anchor_uv_A in Surface Frame"
                )
            anc = np.asarray(anchor_uv, dtype=float)
            if anc.shape != (2,) or not np.all(np.isfinite(anc)):
                raise ValueError(
                    f"Candidate {cand.get('candidate_id')} anchor_uv_A must be a finite 2D vector in Surface Frame"
                )
            self.candidates_by_template.setdefault(tpl_key, []).append(cand)

        for tpl_key, c_list in self.candidates_by_template.items():
            self.anchors_by_template[tpl_key] = np.asarray(
                [c["anchor_uv_A"] for c in c_list], dtype=float
            )

    def query(
        self,
        proposal: dict,
        snap_radius_A: float = 4.0,
    ) -> list[dict]:
        """Query real candidates matching proposal template_key within 2D periodic snap_radius_A."""
        if proposal is None or "template_key" not in proposal or "q_uv_A" not in proposal:
            raise ValueError("proposal must contain template_key and q_uv_A")
        tpl_key = proposal["template_key"]
        q_uv = np.asarray(proposal["q_uv_A"], dtype=float)
        if q_uv.shape != (2,) or not np.all(np.isfinite(q_uv)):
            raise ValueError("proposal q_uv_A must be a finite 2D vector")

        if tpl_key not in self.candidates_by_template:
            return []

        c_list = self.candidates_by_template[tpl_key]
        anchors = self.anchors_by_template[tpl_key]

        diff = anchors - q_uv
        diff_frac = diff @ self.lattice_inv
        mic_frac = diff_frac - np.round(diff_frac)
        mic_dist = np.linalg.norm(mic_frac @ self.surface_lattice, axis=1)
        matching_indices = np.nonzero(mic_dist <= snap_radius_A + 1e-8)[0]
        return [c_list[i] for i in matching_indices]

    def query_proposals(
        self,
        proposals: list[dict],
        snap_radius_A: float = 4.0,
    ) -> tuple[int, set[str], dict[str, float], dict[str, dict]]:
        """Vectorized query of all proposals against template-grouped candidates.

        Returns:
            snap_match_pair_count (int): total number of (proposal, candidate) pairs within snap_radius_A.
            focused_candidate_ids (set[str]): unique set of candidate IDs within snap_radius_A of any proposal.
            candidate_snap_distances (dict[str, float]): candidate_id -> minimum snap distance in Angstrom.
            candidate_recalling_proposals (dict[str, dict]): candidate_id -> proposal corresponding to min snap distance.
        """
        if not proposals:
            return 0, set(), {}, {}

        proposals_by_template: dict[str, list[dict]] = {}
        for prop in proposals:
            tpl_key = prop.get("template_key")
            if tpl_key is not None:
                proposals_by_template.setdefault(tpl_key, []).append(prop)

        snap_match_pair_count = 0
        focused_candidate_ids: set[str] = set()
        candidate_snap_distances: dict[str, float] = {}
        candidate_recalling_proposals: dict[str, dict] = {}

        for tpl_key, tpl_props in proposals_by_template.items():
            if tpl_key not in self.candidates_by_template:
                continue

            c_list = self.candidates_by_template[tpl_key]
            anchors = self.anchors_by_template[tpl_key]
            n_cands = len(c_list)
            if n_cands == 0:
                continue

            q_array = np.asarray([p["q_uv_A"] for p in tpl_props], dtype=float)

            diff = anchors[:, None, :] - q_array[None, :, :]
            diff_frac = diff @ self.lattice_inv
            mic_frac = diff_frac - np.round(diff_frac)
            mic_dist = np.linalg.norm(mic_frac @ self.surface_lattice, axis=-1)

            mask = mic_dist <= (snap_radius_A + 1e-8)
            cand_indices, prop_indices = np.nonzero(mask)

            match_count = len(cand_indices)
            snap_match_pair_count += match_count

            for c_idx, p_idx in zip(cand_indices, prop_indices):
                cand = c_list[c_idx]
                cid = cand["candidate_id"]
                dist = float(mic_dist[c_idx, p_idx])
                prop = tpl_props[p_idx]

                focused_candidate_ids.add(cid)
                if cid not in candidate_snap_distances or dist < candidate_snap_distances[cid]:
                    candidate_snap_distances[cid] = dist
                    candidate_recalling_proposals[cid] = prop

        return (
            snap_match_pair_count,
            focused_candidate_ids,
            candidate_snap_distances,
            candidate_recalling_proposals,
        )


def focus_continuous_frontier_candidates_streaming(
    accepted_union,
    candidate_templates: dict | list,
    spatial_index,
    surface_lattice_uv_A: np.ndarray,
    snap_radius_A: float = 4.0,
) -> tuple[int, int, set[str], dict[str, float], dict[str, dict]]:
    """Stream continuous frontier ideal kissing proposals per template and focus candidates.

    Precomputes boundary vertex and outward normal metadata in O(V). Iterates through
    templates, vectorizes q* calculation, deduplicates per template according to the 4-decimal
    place rule, and immediately queries candidates of the same template in periodic MIC distance.
    Operates in bounded O(V) proposal memory without constructing or retaining a global O(V*T) list.

    Returns:
        ideal_proposal_count (int): Total unique proposals generated across all templates.
        snap_match_pair_count (int): Total (proposal, candidate) pairs within snap_radius_A.
        focused_candidate_ids (set[str]): Candidate IDs within snap_radius_A of any proposal.
        candidate_snap_distances (dict[str, float]): Minimum snap distance per focused candidate.
        candidate_recalling_proposals (dict[str, dict]): Deterministic recalling proposal per focused candidate.
    """
    if accepted_union is None or getattr(accepted_union, "is_empty", True):
        return 0, 0, set(), {}, {}

    affinity, Point, Polygon, box, unary_union = _require_shapely()
    if surface_lattice_uv_A is None:
        raise ValueError("surface_lattice_uv_A is required and cannot be None")
    surface_lattice = np.asarray(surface_lattice_uv_A, dtype=float)
    if surface_lattice.shape != (2, 2) or not np.all(np.isfinite(surface_lattice)):
        raise ValueError("surface_lattice_uv_A must be a finite 2x2 matrix")
    det_val = float(np.linalg.det(surface_lattice))
    if not np.isfinite(det_val) or abs(det_val) < 1e-12:
        raise ValueError("surface_lattice_uv_A must be invertible (non-zero determinant)")
    lattice_inv = np.linalg.inv(surface_lattice)

    min_x, min_y, max_x, max_y = accepted_union.bounds
    if max_x > 1.0 + 1e-4 or max_y > 1.0 + 1e-4 or min_x < -1e-4 or min_y < -1e-4:
        matrix_to_frac = [lattice_inv[0, 0], lattice_inv[1, 0], lattice_inv[0, 1], lattice_inv[1, 1], 0.0, 0.0]
        union_frac = affinity.affine_transform(accepted_union, matrix_to_frac)
        union_frac = _wrap_to_unit_cell(union_frac)
    else:
        union_frac = accepted_union

    rings = extract_frontier_boundary_rings(union_frac, surface_lattice)
    if not rings:
        return 0, 0, set(), {}, {}

    all_vertices_w = []
    all_normals_phys = []
    all_target_types = []
    all_hole_areas = []

    for ring_info in rings:
        ring = ring_info["ring"]
        coords = np.asarray(ring.coords, dtype=float)
        if len(coords) < 3:
            continue
        vertices = coords[:-1] if np.allclose(coords[0], coords[-1]) else coords
        n_verts = len(vertices)
        hole_area_A2 = float(ring_info["hole_area_A2"])
        target_type = ring_info["target_type"]

        for i, vertex_w in enumerate(vertices):
            prev_v = vertices[(i - 1) % n_verts]
            next_v = vertices[(i + 1) % n_verts]
            edge_in = (vertex_w - prev_v) @ surface_lattice
            edge_out = (next_v - vertex_w) @ surface_lattice
            len_in = float(np.linalg.norm(edge_in))
            len_out = float(np.linalg.norm(edge_out))
            if len_in > 1e-12:
                edge_in /= len_in
            if len_out > 1e-12:
                edge_out /= len_out

            n_in = np.array([edge_in[1], -edge_in[0]])
            n_out = np.array([edge_out[1], -edge_out[0]])
            normal_phys = n_in + n_out
            norm_mag = float(np.linalg.norm(normal_phys))
            if norm_mag > 1e-12:
                normal_phys /= norm_mag
            else:
                normal_phys = n_in

            normal_frac = normal_phys @ lattice_inv
            test_pt = Point(vertex_w[0] + 0.01 * normal_frac[0], vertex_w[1] + 0.01 * normal_frac[1])
            if union_frac.contains(test_pt):
                normal_phys = -normal_phys

            all_vertices_w.append(vertex_w)
            all_normals_phys.append(normal_phys)
            all_target_types.append(target_type)
            all_hole_areas.append(hole_area_A2)

    if not all_vertices_w:
        return 0, 0, set(), {}, {}

    vertices_w = np.asarray(all_vertices_w, dtype=float)
    normals_phys = np.asarray(all_normals_phys, dtype=float)
    n_boundary_verts = len(vertices_w)

    if isinstance(candidate_templates, list):
        templates_dict = {t["template_key"]: t for t in candidate_templates}
    elif isinstance(candidate_templates, dict):
        templates_dict = candidate_templates
    else:
        templates_dict = {}

    precomputed_templates = []
    for tpl_key, tpl in sorted(templates_dict.items(), key=lambda item: str(item[0])):
        rel_verts_A = np.asarray(tpl["relative_vertices_uv_A"], dtype=float)
        if len(rel_verts_A) < 3:
            continue
        rel_verts_frac = rel_verts_A @ lattice_inv
        precomputed_templates.append({
            "tpl_key": tpl_key,
            "template_record": tpl,
            "rel_verts_A": rel_verts_A,
            "rel_verts_frac": rel_verts_frac,
        })

    if not precomputed_templates:
        return 0, 0, set(), {}, {}

    if not isinstance(spatial_index, SameTemplatePeriodicSpatialIndex):
        spatial_index = SameTemplatePeriodicSpatialIndex(spatial_index, surface_lattice)

    ideal_proposal_count = 0
    snap_match_pair_count = 0
    focused_candidate_ids: set[str] = set()
    candidate_snap_distances: dict[str, float] = {}
    candidate_recalling_proposals: dict[str, dict] = {}

    for item in precomputed_templates:
        tpl_key = item["tpl_key"]
        rel_verts_A = item["rel_verts_A"]
        rel_verts_frac = item["rel_verts_frac"]

        projections = rel_verts_A @ normals_phys.T
        contact_indices = np.argmin(projections, axis=0)
        u_contact_frac = rel_verts_frac[contact_indices]

        q_frac = vertices_w - u_contact_frac
        q_wrapped_frac = q_frac % 1.0
        q_wrapped_A = q_wrapped_frac @ surface_lattice

        seen_keys = set()
        tpl_props = []
        for vi in range(n_boundary_verts):
            q_w_A = q_wrapped_A[vi]
            prop_hash = (
                str(tpl_key),
                round(float(q_w_A[0]), 4),
                round(float(q_w_A[1]), 4),
            )
            if prop_hash in seen_keys:
                continue
            seen_keys.add(prop_hash)
            tpl_props.append({
                "q_uv_A": q_w_A,
                "q_frac": q_wrapped_frac[vi],
                "template_key": tpl_key,
                "target_type": all_target_types[vi],
                "hole_area_A2": float(all_hole_areas[vi]),
            })

        ideal_proposal_count += len(tpl_props)

        tpl_props.sort(
            key=lambda p: (
                p["target_type"],
                str(p["template_key"]),
                round(float(p["q_uv_A"][0]), 6),
                round(float(p["q_uv_A"][1]), 6),
            )
        )

        if tpl_key in spatial_index.candidates_by_template and tpl_props:
            c_list = spatial_index.candidates_by_template[tpl_key]
            anchors = spatial_index.anchors_by_template[tpl_key]
            n_cands = len(c_list)
            if n_cands > 0:
                q_array = np.asarray([p["q_uv_A"] for p in tpl_props], dtype=float)
                diff = anchors[:, None, :] - q_array[None, :, :]
                diff_frac = diff @ lattice_inv
                mic_frac = diff_frac - np.round(diff_frac)
                mic_dist = np.linalg.norm(mic_frac @ surface_lattice, axis=-1)

                mask = mic_dist <= (snap_radius_A + 1e-8)
                cand_indices, prop_indices = np.nonzero(mask)

                snap_match_pair_count += len(cand_indices)

                for c_idx, p_idx in zip(cand_indices, prop_indices):
                    cand = c_list[c_idx]
                    cid = cand["candidate_id"]
                    dist = float(mic_dist[c_idx, p_idx])
                    prop = tpl_props[p_idx]

                    focused_candidate_ids.add(cid)
                    if cid not in candidate_snap_distances or dist < candidate_snap_distances[cid]:
                        candidate_snap_distances[cid] = dist
                        candidate_recalling_proposals[cid] = prop

                del q_array, diff, diff_frac, mic_frac, mic_dist, mask, cand_indices, prop_indices

        del tpl_props, projections, contact_indices, u_contact_frac, q_frac, q_wrapped_frac, q_wrapped_A, seen_keys

    return (
        ideal_proposal_count,
        snap_match_pair_count,
        focused_candidate_ids,
        candidate_snap_distances,
        candidate_recalling_proposals,
    )


def query_snap_candidates_for_proposal(
    proposal: dict,
    candidates: list[dict],
    surface_lattice_uv_A: np.ndarray,
    snap_radius_A: float = 4.0,
    *,
    spatial_index: SameTemplatePeriodicSpatialIndex | None = None,
) -> list[dict]:
    """Query real candidates matching proposal template_key within 2D periodic snap_radius_A."""
    if spatial_index is not None:
        return spatial_index.query(proposal, snap_radius_A=snap_radius_A)

    if proposal is None or "template_key" not in proposal or "q_uv_A" not in proposal:
        raise ValueError("proposal must contain template_key and q_uv_A")
    tpl_key = proposal["template_key"]
    q_uv = np.asarray(proposal["q_uv_A"], dtype=float)
    if q_uv.shape != (2,) or not np.all(np.isfinite(q_uv)):
        raise ValueError("proposal q_uv_A must be a finite 2D vector")

    if surface_lattice_uv_A is None:
        raise ValueError("surface_lattice_uv_A is required and cannot be None")
    surface_lattice = np.asarray(surface_lattice_uv_A, dtype=float)
    if surface_lattice.shape != (2, 2) or not np.all(np.isfinite(surface_lattice)):
        raise ValueError("surface_lattice_uv_A must be a finite 2x2 matrix")
    det = float(np.linalg.det(surface_lattice))
    if not np.isfinite(det) or abs(det) < 1e-12:
        raise ValueError("surface_lattice_uv_A must be invertible (non-zero determinant)")
    lattice_inv = np.linalg.inv(surface_lattice)

    recalled = []
    for cand in candidates:
        if cand.get("template_key") != tpl_key:
            continue
        anchor_uv = cand.get("anchor_uv_A")
        if anchor_uv is None:
            raise KeyError(
                f"Candidate {cand.get('candidate_id')} lacks anchor_uv_A in Surface Frame"
            )
        diff = np.asarray(anchor_uv, dtype=float) - q_uv
        diff_frac = diff @ lattice_inv
        mic_frac = diff_frac - np.round(diff_frac)
        mic_dist = float(np.linalg.norm(mic_frac @ surface_lattice))
        if mic_dist <= snap_radius_A + 1e-8:
            recalled.append(cand)

    return recalled


def _evaluate_cohesive_candidates(
    candidates: list[dict],
    *,
    placed: list[dict],
    accepted_union,
    projection_cache,
    surface_lattice: np.ndarray,
    cell,
    periodic_axes,
    collision_thresholds_A: dict,
    rng,
    perimeter_tol: float,
    contact_tol: float,
    area_tol: float,
    energy_tolerance_eV: float,
    contact_vdw_radius_scale: float = 1.10,
    local_neighbors_by_candidate_id: dict | None = None,
    projection_neighbor_index=None,
) -> tuple[dict, dict]:
    """Execute canonical cohesive lexicographic filtering pipeline on a candidate subset.

    Order of criteria:
      1) Min periodic union-perimeter increment (relative tolerance, default 10%)
      2) Min single-candidate XY projected area (relative tolerance, default 20%)
      3a) Maximize distinct contacted adsorbed neighbors (exact maximum, 0 deficit)
      3b) Maximize side vdW contact pair count (relative tolerance, default 20%, >= 1 pair)
      4) Min conformer energy (tolerance_eV, default 0.03 eV)
      5) Seeded exact tie-break
    """
    if not candidates:
        raise ValueError("Candidates list cannot be empty for cohesive evaluation")

    before_perimeter_A, _ = periodic_union_increment_perimeter(
        None, accepted_union, surface_lattice
    )
    perimeter_records = []
    perimeter_trials = {}

    for candidate in candidates:
        cand_id = candidate["candidate_id"]
        if projection_neighbor_index is None:
            geometry = projection_cache.candidate_geometries[
                candidate["candidate_index"]
            ]
            candidate_perimeter_A, _ = periodic_union_increment_perimeter(
                None, geometry, surface_lattice
            )
            increment_A, _ = periodic_union_increment_perimeter(
                accepted_union,
                geometry,
                surface_lattice,
                current_union_perimeter_A=before_perimeter_A,
            )
            after_perimeter_A = before_perimeter_A + increment_A
            gain = cohesive_perimeter_condensation_gain(
                before_perimeter_A, candidate_perimeter_A, after_perimeter_A
            )
            trial = {
                "before_perimeter_A": float(before_perimeter_A),
                "candidate_perimeter_A": float(candidate_perimeter_A),
                "after_perimeter_A": float(after_perimeter_A),
                "increment_A": float(increment_A),
                "perimeter_condensation_gain": float(gain),
                "perimeter_scope": "global_accepted_union",
            }
        else:
            neighbors = (
                local_neighbors_by_candidate_id[cand_id]
                if local_neighbors_by_candidate_id is not None
                else placed
            )
            trial = projection_neighbor_index.perimeter_trial(
                candidate, neighbors
            )
            gain = trial["perimeter_condensation_gain"]

        record = dict(candidate)
        record["increment_A"] = trial["increment_A"]
        record["candidate_perimeter_A"] = trial["candidate_perimeter_A"]
        record["perimeter_condensation_gain"] = gain
        perimeter_records.append(record)
        perimeter_trials[cand_id] = trial

    perimeter_finalists, perimeter_audit = cohesive_filter_perimeter_increment(
        perimeter_records, relative_tolerance=perimeter_tol
    )

    # Step 2: Projected area filtering evaluated on perimeter finalists
    area_records = [
        _cohesive_candidate_area_record(candidate, projection_cache)
        for candidate in perimeter_finalists
    ]
    area_finalists, area_audit = cohesive_filter_projected_area(
        area_records, relative_tolerance=area_tol
    )

    # Step 3: Expensive full side contact evaluated only for area survivors
    full_contact_audits = {}
    contact_records = []
    for candidate in area_finalists:
        cand_id = candidate["candidate_id"]
        neighbors = (
            local_neighbors_by_candidate_id[cand_id]
            if local_neighbors_by_candidate_id is not None
            else placed
        )
        report = cohesive_side_contact_report(
            candidate,
            neighbors,
            cell=cell,
            periodic_axes=periodic_axes,
            collision_thresholds_A=collision_thresholds_A,
            contact_vdw_radius_scale=contact_vdw_radius_scale,
            stop_after_first=False,
        )
        full_contact_audits[cand_id] = report

        distinct_count = report.get("distinct_contacted_neighbor_count")
        if distinct_count is None:
            distinct_count = report.get("contacted_neighbor_count")
        if distinct_count is None:
            contacted_ids = report.get("contacted_neighbor_ids")
            if contacted_ids is not None:
                distinct_count = len(contacted_ids)
            elif report.get("has_contact") or report.get("contact_count", 0) > 0:
                distinct_count = 1
            else:
                distinct_count = 0

        pair_count = report.get("contact_count", 0)

        record = dict(candidate)
        record["distinct_neighbor_count"] = int(distinct_count)
        record["side_contact_count"] = int(pair_count)
        if "contacted_neighbor_ids" in report:
            record["contacted_neighbor_ids"] = list(report["contacted_neighbor_ids"])
        contact_records.append(record)

    # Step 3a: Maximize distinct contacted neighbors (strict maximum, 0 deficit)
    neighbor_finalists, neighbor_audit = cohesive_filter_distinct_neighbors(
        contact_records
    )

    # Step 3b: Maximize side contact pair count among 3a survivors (20% relative tolerance)
    contact_finalists, contact_audit = cohesive_filter_side_contacts(
        neighbor_finalists,
        relative_tolerance=contact_tol,
    )
    contact_audit["evaluated_candidate_ids"] = [
        candidate["candidate_id"] for candidate in area_finalists
    ]

    # Step 4: Conformer energy filtering
    energy_records = [_cohesive_energy_record(record) for record in contact_finalists]
    energy_finalists, energy_audit = cohesive_filter_conformer_energy(
        energy_records, tolerance_eV=energy_tolerance_eV
    )

    # Step 5: Seeded exact tie-break
    winner = _cohesive_seed_choice(energy_finalists, rng)

    pipeline_audit = {
        "perimeter_trials": perimeter_trials,
        "perimeter_filter": perimeter_audit,
        "area_filter": area_audit,
        "full_contact_audits": full_contact_audits,
        "distinct_neighbor_filter": neighbor_audit,
        "full_contact_filter": contact_audit,
        "contact_pair_filter": contact_audit,
        "energy_filter": energy_audit,
        "seed_finalist_candidate_ids": [
            record["candidate_id"] for record in energy_finalists
        ],
        "winner_candidate_id": winner["candidate_id"],
    }
    return winner, pipeline_audit


def choose_cohesive_frontier_candidate(
    *,
    candidates,
    placed,
    accepted_union,
    projection_cache,
    cell,
    periodic_axes,
    collision_thresholds_A,
    rng,
    relative_tolerance=None,
    perimeter_relative_tolerance=None,
    contact_relative_tolerance=None,
    area_relative_tolerance=None,
    energy_tolerance_eV=0.03,
    contact_vdw_radius_scale=1.10,
    xy_bounds_by_candidate_id=None,
    projection_neighbor_index=None,
    cohesive_frontier_mode="discrete",
    frontier_snap_radius_A=4.0,
):
    """Choose one current-denticity candidate by local cohesive-frontier growth."""
    ordered = _cohesive_candidate_records(candidates)
    if not ordered:
        raise ValueError("cohesive-frontier selection requires candidates")
    if not isinstance(placed, (list, tuple)):
        raise TypeError("placed must be a list or tuple")
    denticities = {
        1
        if (
            candidate.get("site_kind") == "single_metal_triangle"
            or int(candidate.get("phase_priority", 0)) == 1
        )
        else len(candidate.get("target_donor_labels", ()))
        for candidate in ordered
    }
    if len(denticities) != 1 or next(iter(denticities)) not in (1, 2, 3):
        raise ValueError("cohesive selector requires one valid denticity phase")
    denticity = next(iter(denticities))
    is_single_metal = all(
        candidate.get("site_kind") == "single_metal_triangle"
        or int(candidate.get("phase_priority", 0)) == 1
        for candidate in ordered
    )
    resolved_tolerances = resolve_cohesive_relative_tolerances(
        perimeter=perimeter_relative_tolerance,
        contact=contact_relative_tolerance,
        area=area_relative_tolerance,
        relative_tolerance=relative_tolerance,
    )
    perimeter_tol = resolved_tolerances["perimeter"]
    contact_tol = resolved_tolerances["contact"]
    area_tol = resolved_tolerances["area"]
    energy_tolerance_eV = _cohesive_nonnegative(
        energy_tolerance_eV, "energy_tolerance_eV"
    )

    early_contact_audits = {}
    xy_prefilter_audits = {}
    frontier = []
    local_neighbors_by_candidate_id = {}
    if placed:
        for candidate in ordered:
            if projection_neighbor_index is None:
                xy_report = cohesive_xy_may_contact(
                    candidate,
                    placed,
                    cell=cell,
                    periodic_axes=periodic_axes,
                    contact_vdw_radius_scale=contact_vdw_radius_scale,
                    bounds_by_candidate_id=xy_bounds_by_candidate_id,
                )
                possible_ids = set(
                    xy_report["possible_existing_candidate_ids"]
                )
                local_neighbors = [
                    accepted
                    for accepted in placed
                    if accepted["candidate_id"] in possible_ids
                ]
            else:
                local_neighbors, xy_report = projection_neighbor_index.query(
                    candidate
                )
            xy_prefilter_audits[candidate["candidate_id"]] = xy_report
            local_neighbors_by_candidate_id[candidate["candidate_id"]] = (
                local_neighbors
            )
            if not local_neighbors:
                continue
            report = cohesive_side_contact_report(
                candidate,
                local_neighbors,
                cell=cell,
                periodic_axes=periodic_axes,
                collision_thresholds_A=collision_thresholds_A,
                contact_vdw_radius_scale=contact_vdw_radius_scale,
                stop_after_first=True,
            )
            early_contact_audits[candidate["candidate_id"]] = report
            if report["has_contact"]:
                frontier.append(candidate)

    audit = {
        "policy": "cohesive-frontier",
        "denticity_phase": denticity,
        "perimeter_relative_tolerance": perimeter_tol,
        "contact_relative_tolerance": contact_tol,
        "area_relative_tolerance": area_tol,
        "relative_tolerances": dict(resolved_tolerances),
        "legacy_relative_tolerance": (
            None
            if relative_tolerance is None
            else _cohesive_relative_tolerance(relative_tolerance)
        ),
        "relative_tolerance": (
            None
            if relative_tolerance is None
            else _cohesive_relative_tolerance(relative_tolerance)
        ),
        "energy_tolerance_eV": energy_tolerance_eV,
        "contact_vdw_radius_scale": float(contact_vdw_radius_scale),
        "capacity_evaluated": False,
        "capacity_bypass_reason": "local_cohesive_growth_has_no_future_capacity_objective",
        "input_candidate_count": len(ordered),
        "input_candidate_ids_sha256": hashlib.sha256(
            "\n".join(candidate["candidate_id"] for candidate in ordered).encode(
                "utf-8", "surrogatepass"
            )
        ).hexdigest(),
        "frontier_boolean_evaluated_count": len(early_contact_audits),
        "frontier_candidate_ids": [candidate["candidate_id"] for candidate in frontier],
        "frontier_contact_audits": {
            candidate["candidate_id"]: early_contact_audits[candidate["candidate_id"]]
            for candidate in frontier
        },
        "projection_prefilter": {
            "implemented": True,
            "correctness": "zero_false_negative_conservative_XY_bound_then_exact_3D",
            "input_count": len(ordered) if placed else 0,
            "exact_3D_evaluated_count": len(early_contact_audits),
            "nonempty_local_neighbor_candidate_count": sum(
                bool(record.get("local_neighbor_count", len(record.get(
                    "possible_existing_candidate_ids", []
                ))))
                for record in xy_prefilter_audits.values()
            ),
            "total_local_neighbor_references": sum(
                int(record.get("local_neighbor_count", len(record.get(
                    "possible_existing_candidate_ids", []
                ))))
                for record in xy_prefilter_audits.values()
            ),
            "maximum_local_neighbor_count": max(
                (
                    int(record.get("local_neighbor_count", len(record.get(
                        "possible_existing_candidate_ids", []
                    ))))
                    for record in xy_prefilter_audits.values()
                ),
                default=0,
            ),
        },
    }

    if not frontier:
        area_records = [
            _cohesive_candidate_area_record(candidate, projection_cache)
            for candidate in ordered
        ]
        area_finalists, area_audit = cohesive_filter_projected_area(
            area_records, relative_tolerance=area_tol
        )
        energy_records = [_cohesive_energy_record(record) for record in area_finalists]
        energy_finalists, energy_audit = cohesive_filter_conformer_energy(
            energy_records, tolerance_eV=energy_tolerance_eV
        )
        winner = _cohesive_seed_choice(energy_finalists, rng)
        audit.update(
            {
                "growth_mode": "new-nucleus",
                "new_nucleus_event": True,
                "nucleus_reason": (
                    "initial_molecule" if not placed else "no_exact_frontier_contact"
                ),
                "perimeter_filter": None,
                "area_filter": area_audit,
                "distinct_neighbor_filter": None,
                "full_contact_filter": None,
                "contact_pair_filter": None,
                "energy_filter": energy_audit,
                "seed_finalist_candidate_ids": [
                    record["candidate_id"] for record in energy_finalists
                ],
                "winner_candidate_id": winner["candidate_id"],
            }
        )
        if cohesive_frontier_mode == "continuous-snap":
            audit["cohesive_frontier_mode"] = "continuous-snap"
            if is_single_metal:
                audit["continuous_snap_bypass_reason"] = (
                    "single_metal_triangle_uses_discrete_uncovered_metal_anchors"
                )
            audit["snap_radius_A"] = float(frontier_snap_radius_A)
            audit["frontier_snap_radius_A"] = float(frontier_snap_radius_A)
            audit["algorithm_scope"] = "frontier-ideal-position real-candidate focusing"
            audit["continuous_global_optimum_claimed"] = False
            audit["continuous_search_qualification"] = (
                "Continuous-snap mode operates as a deterministic frontier-ideal-position real-candidate focusing heuristic on discrete boundary vertices and outward normals, bounded by snap_radius and discrete candidate domain; no global geometric continuous optimum is claimed."
            )
            audit["pre_snap_side_contact"] = "not_evaluated_no_3d_pose"
            audit["pre_snap_perimeter"] = "not_evaluated_virtual_geometry"
            audit["virtual_polygon_constructed"] = False
            audit["ideal_proposal_count"] = 0
            audit["proposal_count"] = 0
            audit["snap_match_pair_count"] = 0
            audit["focused_candidate_count"] = 0
            audit["focused_frontier_candidate_count"] = 0
            audit["omitted_candidate_count"] = 0
        return winner, audit

    if accepted_union is None:
        raise RuntimeError("frontier candidates require an accepted projection union")

    surface_lattice = getattr(projection_cache, "surface_lattice_uv_A", None)
    if surface_lattice is None:
        raise KeyError("projection_cache lacks required surface_lattice_uv_A")
    surface_lattice = np.asarray(surface_lattice, dtype=float)
    if surface_lattice.shape != (2, 2) or not np.all(np.isfinite(surface_lattice)):
        raise ValueError("projection_cache.surface_lattice_uv_A must be a finite 2x2 matrix")
    det = float(np.linalg.det(surface_lattice))
    if not np.isfinite(det) or abs(det) < 1e-12:
        raise ValueError("projection_cache.surface_lattice_uv_A must be invertible (non-zero determinant)")

    if cohesive_frontier_mode == "continuous-snap" and not is_single_metal:
        template_records = getattr(projection_cache, "template_records", None)
        if not template_records or not isinstance(template_records, list):
            raise KeyError("projection_cache lacks valid template_records")

        template_index_arr = getattr(projection_cache, "template_index", None)
        anchor_uv_arr = getattr(projection_cache, "anchor_uv", None)

        for cand in ordered:
            if "template_key" not in cand:
                cand_idx = cand.get("candidate_index")
                if cand_idx is None or template_index_arr is None:
                    raise KeyError(
                        f"Candidate {cand.get('candidate_id')} lacks template_key and projection_cache lacks template_index"
                    )
                cand_idx = int(cand_idx)
                if cand_idx < 0 or cand_idx >= len(template_index_arr):
                    raise IndexError(f"Candidate index {cand_idx} out of range for template_index")
                t_pos = int(template_index_arr[cand_idx])
                if t_pos < 0 or t_pos >= len(template_records):
                    raise IndexError(f"Template index {t_pos} out of range for template_records")
                cand["template_key"] = template_records[t_pos]["template_key"]

            if "anchor_uv_A" not in cand:
                cand_idx = cand.get("candidate_index")
                if cand_idx is None or anchor_uv_arr is None:
                    raise KeyError(
                        f"Candidate {cand.get('candidate_id')} lacks anchor_uv_A in Surface Frame and projection_cache lacks anchor_uv"
                    )
                cand_idx = int(cand_idx)
                if cand_idx < 0 or cand_idx >= len(anchor_uv_arr):
                    raise IndexError(f"Candidate index {cand_idx} out of range for anchor_uv")
                cand["anchor_uv_A"] = np.asarray(anchor_uv_arr[cand_idx], dtype=float)

            anc_uv = np.asarray(cand["anchor_uv_A"], dtype=float)
            if anc_uv.shape != (2,) or not np.all(np.isfinite(anc_uv)):
                raise ValueError(
                    f"Candidate {cand.get('candidate_id')} anchor_uv_A must be a finite 2D vector in Surface Frame"
                )

        template_energies = {}
        for cand in ordered:
            tpl_key = cand["template_key"]
            if "conformer_energy_eV" not in cand or cand["conformer_energy_eV"] is None:
                raise KeyError(f"Candidate {cand.get('candidate_id')} lacks conformer_energy_eV")
            energy = float(cand["conformer_energy_eV"])
            if not np.isfinite(energy):
                raise ValueError(f"Candidate {cand.get('candidate_id')} conformer_energy_eV must be finite: {energy}")
            if tpl_key not in template_energies:
                template_energies[tpl_key] = energy
            else:
                if abs(template_energies[tpl_key] - energy) > 1e-9:
                    raise ValueError(
                        f"Inconsistent conformer_energy_eV for template_key '{tpl_key}': "
                        f"{template_energies[tpl_key]} vs {energy}"
                    )

        _, _, Polygon, _, _ = _require_shapely()
        candidate_templates = {}
        for rec in template_records:
            tpl_key = rec.get("template_key")
            if not tpl_key:
                raise KeyError("Template record missing template_key")
            if tpl_key not in template_energies:
                continue
            rel_verts = rec.get("relative_vertices_uv_A")
            if rel_verts is None:
                raise KeyError(f"Template record {tpl_key} missing relative_vertices_uv_A")
            rel_verts_arr = np.asarray(rel_verts, dtype=float)
            if len(rel_verts_arr) < 3:
                raise ValueError(f"Template record {tpl_key} relative_vertices_uv_A has fewer than 3 vertices")
            poly = Polygon(rel_verts_arr)
            phys_area = float(poly.area)
            if not np.isfinite(phys_area) or phys_area <= 0.0:
                raise ValueError(f"Template {tpl_key} physical projected area must be finite and positive: {phys_area}")

            candidate_templates[tpl_key] = {
                "template_key": tpl_key,
                "relative_vertices_uv_A": rec["relative_vertices_uv_A"],
                "selection_rank": rec.get("selection_rank", 1),
                "conformer_energy_eV": template_energies[tpl_key],
                "projected_area_A2": phys_area,
            }

        if not candidate_templates:
            raise ValueError("No templates with matching candidate conformer energies found")

        spatial_index = SameTemplatePeriodicSpatialIndex(ordered, surface_lattice)
        (
            ideal_proposal_count,
            snap_match_pair_count,
            snap_focused_candidate_ids,
            candidate_snap_distances,
            candidate_recalling_proposals,
        ) = focus_continuous_frontier_candidates_streaming(
            accepted_union=accepted_union,
            candidate_templates=candidate_templates,
            spatial_index=spatial_index,
            surface_lattice_uv_A=surface_lattice,
            snap_radius_A=frontier_snap_radius_A,
        )

        frontier_id_set = {c["candidate_id"] for c in frontier}
        focused_frontier_candidates = [
            c for c in ordered
            if c["candidate_id"] in snap_focused_candidate_ids and c["candidate_id"] in frontier_id_set
        ]

        # Freeze RNG state for deterministic, independent branch evaluation
        frozen_rng_state = rng.getstate()
        benchmark_rng = random.Random()
        benchmark_rng.setstate(frozen_rng_state)
        focused_rng = random.Random()
        focused_rng.setstate(frozen_rng_state)

        # Global discrete benchmark on all contact frontier candidates
        global_winner, global_pipeline_audit = _evaluate_cohesive_candidates(
            frontier,
            placed=placed,
            accepted_union=accepted_union,
            projection_cache=projection_cache,
            surface_lattice=surface_lattice,
            cell=cell,
            periodic_axes=periodic_axes,
            collision_thresholds_A=collision_thresholds_A,
            rng=benchmark_rng,
            perimeter_tol=perimeter_tol,
            contact_tol=contact_tol,
            area_tol=area_tol,
            energy_tolerance_eV=energy_tolerance_eV,
            contact_vdw_radius_scale=contact_vdw_radius_scale,
            local_neighbors_by_candidate_id=local_neighbors_by_candidate_id,
            projection_neighbor_index=projection_neighbor_index,
        )

        global_winner_id = global_winner["candidate_id"]
        global_winner_delta_L = float(global_pipeline_audit["perimeter_trials"][global_winner_id]["increment_A"])
        global_min_delta_L = float(global_pipeline_audit["perimeter_filter"]["minimum_increment_A"])
        global_cutoff_delta_L = float(global_pipeline_audit["perimeter_filter"]["cutoff_increment_A"])
        global_winner_in_focused = bool(global_winner_id in snap_focused_candidate_ids)

        focused_count = len(snap_focused_candidate_ids)
        omitted_count = max(0, len(ordered) - focused_count)

        if not focused_frontier_candidates:
            # Fallback to global discrete frontier within the same denticity
            fallback = True
            chosen_candidate = global_winner
            chosen_pipeline_audit = global_pipeline_audit
            rng.setstate(benchmark_rng.getstate())
            local_winner_delta_L = global_winner_delta_L
            delta_L_excess = 0.0
            within_global_band = True
            chosen_proposal = None
            chosen_snap_dist = None
        else:
            fallback = False
            local_winner, local_pipeline_audit = _evaluate_cohesive_candidates(
                focused_frontier_candidates,
                placed=placed,
                accepted_union=accepted_union,
                projection_cache=projection_cache,
                surface_lattice=surface_lattice,
                cell=cell,
                periodic_axes=periodic_axes,
                collision_thresholds_A=collision_thresholds_A,
                rng=focused_rng,
                perimeter_tol=perimeter_tol,
                contact_tol=contact_tol,
                area_tol=area_tol,
                energy_tolerance_eV=energy_tolerance_eV,
                contact_vdw_radius_scale=contact_vdw_radius_scale,
                local_neighbors_by_candidate_id=local_neighbors_by_candidate_id,
                projection_neighbor_index=projection_neighbor_index,
            )
            chosen_candidate = local_winner
            chosen_pipeline_audit = local_pipeline_audit
            rng.setstate(focused_rng.getstate())

            local_winner_id = chosen_candidate["candidate_id"]
            local_winner_delta_L = float(chosen_pipeline_audit["perimeter_trials"][local_winner_id]["increment_A"])
            delta_L_excess = float(local_winner_delta_L - global_winner_delta_L)
            within_global_band = bool(local_winner_delta_L <= global_cutoff_delta_L + 1e-9)

            recalling_prop = candidate_recalling_proposals.get(local_winner_id)
            chosen_snap_dist = candidate_snap_distances.get(local_winner_id)
            if recalling_prop is not None:
                chosen_proposal = {
                    "q_uv_A": [float(x) for x in recalling_prop["q_uv_A"]],
                    "template_key": recalling_prop["template_key"],
                    "target_type": recalling_prop.get("target_type", "exterior"),
                    "hole_area_A2": float(recalling_prop.get("hole_area_A2", 0.0)),
                    "snap_distance_A": float(chosen_snap_dist) if chosen_snap_dist is not None else None,
                    "delta_L_pre_A": "not_evaluated_virtual_geometry",
                }
            else:
                chosen_proposal = None

        omission_audit = {
            "benchmark_policy": "global_discrete_cohesive_frontier",
            "benchmark_semantics": (
                "Full discrete evaluation on all contact frontier candidates without continuous-snap spatial filtering"
            ),
            "global_candidate_count": len(ordered),
            "frontier_candidate_count": len(frontier),
            "focused_candidate_count": focused_count,
            "focused_frontier_candidate_count": len(focused_frontier_candidates),
            "omitted_candidate_count": omitted_count,
            "ideal_proposal_count": ideal_proposal_count,
            "snap_match_pair_count": snap_match_pair_count,
            "pre_snap_perimeter": "not_evaluated_virtual_geometry",
            "virtual_polygon_constructed": False,
            "global_discrete_winner_candidate_id": global_winner_id,
            "global_discrete_winner_in_focused": global_winner_in_focused,
            "global_min_delta_L_A": global_min_delta_L,
            "global_discrete_winner_delta_L_A": global_winner_delta_L,
            "local_winner_delta_L_A": local_winner_delta_L,
            "delta_L_excess_A": delta_L_excess,
            "within_global_perimeter_band": within_global_band,
            "global_cutoff_increment_A": global_cutoff_delta_L,
            "fallback_to_global": fallback,
        }

        audit.update(
            {
                "growth_mode": "frontier",
                "new_nucleus_event": False,
                "nucleus_reason": None,
                "cohesive_frontier_mode": "continuous-snap",
                "frontier_snap_radius_A": float(frontier_snap_radius_A),
                "snap_radius_A": float(frontier_snap_radius_A),
                "algorithm_scope": "frontier-ideal-position real-candidate focusing",
                "continuous_global_optimum_claimed": False,
                "continuous_search_qualification": (
                    "Continuous-snap mode operates as a deterministic frontier-ideal-position real-candidate focusing heuristic on discrete boundary vertices and outward normals, bounded by snap_radius and discrete candidate domain; no global geometric continuous optimum is claimed."
                ),
                "pre_snap_side_contact": "not_evaluated_no_3d_pose",
                "pre_snap_perimeter": "not_evaluated_virtual_geometry",
                "virtual_polygon_constructed": False,
                "ideal_proposal_count": ideal_proposal_count,
                "proposal_count": ideal_proposal_count,
                "snap_match_pair_count": snap_match_pair_count,
                "focused_candidate_count": focused_count,
                "focused_frontier_candidate_count": len(focused_frontier_candidates),
                "omitted_candidate_count": omitted_count,
                "chosen_proposal": chosen_proposal,
                "snap_distance_A": float(chosen_snap_dist) if chosen_snap_dist is not None else None,
                "fallback_to_global": fallback,
                "omission_audit": omission_audit,
                "perimeter_trials": chosen_pipeline_audit["perimeter_trials"],
                "perimeter_filter": chosen_pipeline_audit["perimeter_filter"],
                "area_filter": chosen_pipeline_audit["area_filter"],
                "full_contact_audits": chosen_pipeline_audit["full_contact_audits"],
                "distinct_neighbor_filter": chosen_pipeline_audit["distinct_neighbor_filter"],
                "full_contact_filter": chosen_pipeline_audit["full_contact_filter"],
                "contact_pair_filter": chosen_pipeline_audit["contact_pair_filter"],
                "energy_filter": chosen_pipeline_audit["energy_filter"],
                "seed_finalist_candidate_ids": chosen_pipeline_audit["seed_finalist_candidate_ids"],
                "winner_candidate_id": chosen_candidate["candidate_id"],
            }
        )
        return chosen_candidate, audit

    # Discrete cohesive-frontier evaluation
    winner, pipeline_audit = _evaluate_cohesive_candidates(
        frontier,
        placed=placed,
        accepted_union=accepted_union,
        projection_cache=projection_cache,
        surface_lattice=surface_lattice,
        cell=cell,
        periodic_axes=periodic_axes,
        collision_thresholds_A=collision_thresholds_A,
        rng=rng,
        perimeter_tol=perimeter_tol,
        contact_tol=contact_tol,
        area_tol=area_tol,
        energy_tolerance_eV=energy_tolerance_eV,
        contact_vdw_radius_scale=contact_vdw_radius_scale,
        local_neighbors_by_candidate_id=local_neighbors_by_candidate_id,
        projection_neighbor_index=projection_neighbor_index,
    )
    audit.update(
        {
            "growth_mode": "frontier",
            "new_nucleus_event": False,
            "nucleus_reason": None,
            "cohesive_frontier_mode": "discrete",
            "fallback_to_global": False,
            "perimeter_trials": pipeline_audit["perimeter_trials"],
            "perimeter_filter": pipeline_audit["perimeter_filter"],
            "area_filter": pipeline_audit["area_filter"],
            "full_contact_audits": pipeline_audit["full_contact_audits"],
            "distinct_neighbor_filter": pipeline_audit["distinct_neighbor_filter"],
            "full_contact_filter": pipeline_audit["full_contact_filter"],
            "contact_pair_filter": pipeline_audit["contact_pair_filter"],
            "energy_filter": pipeline_audit["energy_filter"],
            "seed_finalist_candidate_ids": pipeline_audit["seed_finalist_candidate_ids"],
            "winner_candidate_id": winner["candidate_id"],
        }
    )
    if is_single_metal and cohesive_frontier_mode == "continuous-snap":
        audit["continuous_snap_bypass_reason"] = (
            "single_metal_triangle_uses_discrete_uncovered_metal_anchors"
        )
    return winner, audit


def capacity_ranking_artifact_contract() -> dict:
    """Describe the immutable per-step capacity-ranking audit artifact."""

    return {
        "schema": CAPACITY_RANKING_AUDIT_SCHEMA,
        "path_pattern": "ranking/step-XXXX.json.gz",
        "serialization": {
            "format": "strict_json",
            "encoding": "utf-8",
            "allow_nan": False,
        },
        "compression": {"format": "gzip", "mtime": 0},
        "hash": {
            "algorithm": "sha256",
            "scope": "compressed_file_bytes",
        },
        "byte_counts": ["uncompressed_bytes", "compressed_bytes"],
        "installation": {
            "method": "same_directory_hard_link_noreplace",
            "temporary_file_fsync_before_install": True,
            "destination_clobber": False,
            "temporary_unlink_after_install": True,
            "ranking_directory_fsync_after_unlink": True,
        },
        "retained_state_with_sink": {
            "trajectory_step": "bounded_summary_and_artifact_reference",
            "full_payload": "released_after_artifact_persisted",
        },
        "capacity_correctness_scope": {
            "static_capacity": (
                "exact_integer_maximum_cardinality_under_shared_metal_and_"
                "explicit_static_conflicts"
            ),
            "milp_trivial_certificate": (
                "explicit_zero_gap_and_bound_equal_to_exact_integer_objective"
            ),
            "milp_nontrivial_certificate": (
                "status_optimal;gap_and_dual_bound_present_finite;gap_exactly_zero;"
                "max_objective_bound_closes_on_integer_objective"
            ),
            "K_after_collision": (
                "exact_static_capacity_of_post_preview_individually_feasible_sites"
            ),
            "future_candidate_pair_collisions_in_capacity": False,
            "one_plus_K_after_collision_is_upper_bound_only": True,
            "global_candidate_optimality": False,
            "global_packing_certificate": False,
        },
    }


def _strict_json_bytes(payload, *, indent: int | None = None) -> bytes:
    """Serialize one payload as deterministic standards-compliant JSON bytes."""

    try:
        text = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=indent,
            separators=(",", ":") if indent is None else None,
            sort_keys=True,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Payload is not strict JSON serializable") from exc
    return (text + "\n").encode("utf-8") if indent is not None else text.encode("utf-8")


def _fsync_directory(path: Path) -> None:
    """Durably persist same-directory link/unlink metadata on Linux."""

    directory_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _atomic_install_noclobber_bytes(
    destination: Path,
    payload: bytes,
    *,
    temporary_prefix: str | None = None,
) -> bytes:
    """Install bytes through a same-directory hard link without replacement."""

    destination = Path(destination)
    parent = destination.parent
    if not parent.is_dir():
        raise FileNotFoundError(f"Artifact parent directory does not exist: {parent}")
    prefix = temporary_prefix or f".{destination.name}-"
    fd, temporary_name = tempfile.mkstemp(dir=parent, prefix=prefix, suffix=".tmp")
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, destination)
        temporary_path.unlink()
        _fsync_directory(parent)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    written = destination.read_bytes()
    if written != payload:
        raise RuntimeError(f"Artifact verification failed after write: {destination}")
    return written


def _atomic_install_noclobber_generated_file(
    destination: Path, writer, *, temporary_prefix: str | None = None
) -> dict:
    """Generate, fsync, and hard-link one file without loading it into memory."""

    destination = Path(destination)
    parent = destination.parent
    if parent.is_symlink() or not parent.is_dir():
        raise FileNotFoundError(f"Artifact parent is not a real directory: {parent}")
    prefix = temporary_prefix or f".{destination.name}-"
    fd, temporary_name = tempfile.mkstemp(dir=parent, prefix=prefix, suffix=".tmp")
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w+b") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_identity = {
            "sha256": _sha256(temporary_path),
            "bytes": temporary_path.stat().st_size,
        }
        os.link(temporary_path, destination)
        temporary_path.unlink()
        _fsync_directory(parent)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    installed_identity = {
        "sha256": _sha256(destination),
        "bytes": destination.stat().st_size,
    }
    if installed_identity != temporary_identity:
        raise RuntimeError(
            f"Generated artifact verification failed after write: {destination}"
        )
    return {
        "path": str(destination),
        **installed_identity,
        "installation": "same_directory_hard_link_noreplace",
    }


def write_strict_json_noclobber(destination: Path, payload: object) -> dict:
    """Write one strict JSON document atomically and return its byte identity."""

    destination = Path(destination)
    rendered = _strict_json_bytes(payload, indent=2)
    written = _atomic_install_noclobber_bytes(destination, rendered)
    return {
        "path": str(destination),
        "sha256": hashlib.sha256(written).hexdigest(),
        "bytes": len(written),
        "serialization": "strict_json_allow_nan_false",
        "installation": "same_directory_hard_link_noreplace",
    }


def write_live_progress_atomic(destination: Path, payload: object) -> dict:
    """Atomically write or overwrite live progress JSON with durability guarantees."""

    destination = Path(destination)
    parent = destination.parent
    if not parent.is_dir():
        raise FileNotFoundError(f"Destination parent directory does not exist: {parent}")
    if destination.is_symlink():
        raise ValueError(f"Destination must not be a symlink: {destination}")

    rendered = _strict_json_bytes(payload, indent=2)

    fd, temporary_name = tempfile.mkstemp(
        dir=parent,
        prefix=f".{destination.name}-",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        _fsync_directory(parent)
    finally:
        temporary_path.unlink(missing_ok=True)

    return {
        "path": str(destination),
        "sha256": hashlib.sha256(rendered).hexdigest(),
        "bytes": len(rendered),
    }


def write_capacity_ranking_audit(
    trajectory_dir: Path,
    *,
    step: int,
    audit_document: dict,
) -> dict:
    """Atomically write one deterministic immutable capacity audit artifact."""

    if isinstance(step, bool) or int(step) != step or int(step) <= 0:
        raise ValueError("capacity ranking audit step must be a positive integer")
    step = int(step)
    trajectory_dir = Path(trajectory_dir)
    if not trajectory_dir.is_dir():
        raise FileNotFoundError(
            f"Capacity ranking trajectory directory does not exist: {trajectory_dir}"
        )
    if not isinstance(audit_document, dict):
        raise TypeError("capacity ranking audit document must be a dictionary")
    if audit_document.get("schema") != CAPACITY_RANKING_AUDIT_SCHEMA:
        raise ValueError("capacity ranking audit document schema mismatch")
    if audit_document.get("step") != step:
        raise ValueError("capacity ranking audit document step mismatch")

    uncompressed = _strict_json_bytes(audit_document)
    buffer = io.BytesIO()
    with gzip.GzipFile(
        filename="",
        mode="wb",
        fileobj=buffer,
        compresslevel=9,
        mtime=0,
    ) as handle:
        handle.write(uncompressed)
    compressed = buffer.getvalue()

    ranking_dir = trajectory_dir / "ranking"
    if ranking_dir.is_symlink():
        raise ValueError("capacity ranking directory must not be a symlink")
    try:
        ranking_dir.mkdir(exist_ok=False)
    except FileExistsError:
        if ranking_dir.is_symlink():
            raise ValueError("capacity ranking directory must not be a symlink")
        if not ranking_dir.is_dir():
            raise ValueError("capacity ranking path must be a directory")
    destination = ranking_dir / f"step-{step:04d}.json.gz"
    written = _atomic_install_noclobber_bytes(
        destination,
        compressed,
        temporary_prefix=f".step-{step:04d}-",
    )
    return {
        "schema": CAPACITY_RANKING_AUDIT_SCHEMA,
        "path": destination.relative_to(trajectory_dir).as_posix(),
        "compression": "gzip",
        "gzip_mtime": 0,
        "sha256": hashlib.sha256(written).hexdigest(),
        "hash_algorithm": "sha256",
        "hash_scope": "compressed_file_bytes",
        "uncompressed_bytes": len(uncompressed),
        "compressed_bytes": len(written),
    }


def _validate_capacity_audit_artifact_reference(
    reference: dict, *, step: int
) -> dict:
    """Validate the bounded reference returned by a capacity audit sink."""

    if not isinstance(reference, dict):
        raise TypeError(
            "capacity_audit_sink must return an artifact-reference dictionary"
        )
    _strict_json_bytes(reference)
    expected_path = f"ranking/step-{int(step):04d}.json.gz"
    expected_values = {
        "schema": CAPACITY_RANKING_AUDIT_SCHEMA,
        "path": expected_path,
        "compression": "gzip",
        "gzip_mtime": 0,
        "hash_algorithm": "sha256",
        "hash_scope": "compressed_file_bytes",
    }
    for name, expected in expected_values.items():
        if reference.get(name) != expected:
            raise ValueError(
                f"capacity audit artifact reference {name} must be {expected!r}"
            )
    sha256 = reference.get("sha256")
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
    ):
        raise ValueError("capacity audit artifact reference has an invalid sha256")
    for name in ("uncompressed_bytes", "compressed_bytes"):
        value = reference.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"capacity audit artifact reference {name} must be a positive integer"
            )
    return dict(reference)


def _capacity_audit_summary(capacity_audit: dict) -> dict:
    """Return the bounded per-step summary retained beside an external audit."""

    winner_id = str(capacity_audit["winner_candidate_id"])
    winner_record = next(
        record
        for record in capacity_audit["beam_candidates"]
        if str(record["candidate_id"]) == winner_id
    )
    query_summary = capacity_audit["capacity_milp_queries"]
    beam = capacity_audit["beam"]
    return {
        "schema": CAPACITY_RANKING_SUMMARY_SCHEMA,
        "externalized": True,
        "policy": "capacity-aware",
        "optimization_scope": capacity_audit["optimization_scope"],
        "global_candidate_optimality": False,
        "global_packing_certificate": False,
        "full_payload_retention": "released_after_artifact_persisted",
        "winner_candidate_id": winner_id,
        "full_domain_candidate_count": int(
            capacity_audit["full_domain_candidate_count"]
        ),
        "beam_candidate_count": int(
            beam["beam_candidate_count_after_tie_closure"]
        ),
        "beam_tie_closure_added_count": int(beam["tie_closure_added_count"]),
        "K_before": int(capacity_audit["K_before"]),
        "winner_capacity": {
            name: int(winner_record[name])
            for name in (
                "K_before",
                "K_after_static",
                "K_after_collision",
                "static_debt",
                "collision_debt",
                "total_debt",
            )
        },
        "capacity_milp": {
            "solve_count": int(query_summary["solve_count"]),
            "cache_hit_count": int(query_summary["cache_hit_count"]),
            "all_results_certified_optimal": bool(
                query_summary["all_results_certified_optimal"]
            ),
        },
    }


def _materialized_transition_summary(transition: dict) -> dict:
    """Return bounded winner-transition evidence for the trajectory manifest."""

    return {
        "chosen_candidate_id": transition["chosen_candidate_id"],
        "chosen_site_instance_id": transition["chosen_site_instance_id"],
        "static_blocked_site_count": len(transition["static_blocked_site_ids"]),
        "static_live_site_count": len(transition["static_live_site_ids"]),
        "live_site_count": len(transition["live_site_ids"]),
        "K_after_static": int(transition["K_after_static"]),
        "K_after_collision": int(transition["K_after_collision"]),
        "collision_query_count": int(transition["collision_query_count"]),
        "collision_pruned_candidate_count": int(
            transition["collision_pruned_candidate_count"]
        ),
        "remaining_candidate_count": int(transition["remaining_candidate_count"]),
        "collision_pruned_count_is_exact": bool(
            transition["collision_pruned_count_is_exact"]
        ),
    }


def _capacity_audit_release_checkpoint(*, step: int, retained_step: dict) -> None:
    """Instrumentation seam called after externalized full payload references die."""

    return None


def _capacity_ranking_audit_document(
    *,
    step: int,
    seed: int,
    capacity_audit: dict,
    materialized_transition: dict,
) -> dict:
    """Build the complete strict-JSON document handed to an audit sink."""

    return {
        "schema": CAPACITY_RANKING_AUDIT_SCHEMA,
        "schema_version": 1,
        "step": int(step),
        "seed": int(seed),
        "selection_policy": "capacity-aware",
        "capacity_correctness_scope": capacity_ranking_artifact_contract()[
            "capacity_correctness_scope"
        ],
        "capacity_audit": capacity_audit,
        "materialized_transition": materialized_transition,
    }


def run_sequential_domains(
    *,
    domains: dict[str, list[dict]],
    site_conflicts: dict[str, set[str]],
    conformer_weights: dict[int, float],
    cell,
    periodic_axes=(0, 1),
    seed: int,
    selection_policy: str = "cohesive-frontier",
    projection_cache=None,
    area_tie_tolerance_A2: float = 1.0e-6,
    perimeter_tie_tolerance_A: float = 1.0e-6,
    collision_profiler=None,
    metal_constraints=None,
    collision_cache=None,
    collision_thresholds_A=None,
    selection_criterion="uncovered-anchors",
    capacity_beam_width: int = 32,
    capacity_time_limit_seconds: float = 30.0,
    capacity_audit_sink=None,
    candidate_domain_fingerprint: str | None = None,
    cohesive_relative_tolerance: float | None = None,
    cohesive_perimeter_relative_tolerance: float | None = None,
    cohesive_contact_relative_tolerance: float | None = None,
    cohesive_area_relative_tolerance: float | None = None,
    cohesive_energy_tolerance_eV: float = 0.03,
    cohesive_contact_vdw_radius_scale: float = 1.10,
    cohesive_xy_bounds_by_candidate_id=None,
    candidate_collision_index=None,
    cohesive_frontier_mode: str = "discrete",
    frontier_snap_radius_A: float = 4.0,
    progress_sink=None,
    single_metal_context: dict | None = None,
) -> dict:
    """Consume candidate domains one molecule at a time until geometric jamming.

    A capacity-aware caller may provide ``capacity_audit_sink(step, document)``.
    The sink must durably write the complete document and return the declared
    bounded artifact reference.  It is invoked and validated before each
    accepted-state commit.  Without a sink, full audits remain in ``steps`` for
    library/API compatibility.
    """

    started_perf = time.perf_counter()
    rng = random.Random(int(seed))
    remaining = {site: list(candidates) for site, candidates in domains.items()}
    initial_candidate_count = sum(len(candidates) for candidates in remaining.values())
    placed = []
    steps = []
    single_metal_phase_entered = False
    blocked_sites: dict[str, dict] = {}
    site_ids = sorted(domains)
    site_index = {site_id: index for index, site_id in enumerate(site_ids)}
    accepted_coverage_mask = 0
    accepted_union = None
    cache_status = "off"
    active_cache = None
    thresholds_A = _validate_collision_thresholds(collision_thresholds_A)
    cohesive_projection_neighbor_index = None
    if single_metal_context is not None:
        if projection_cache is not None:
            projection_cache = copy.copy(projection_cache)
            if hasattr(projection_cache, "candidate_geometries") and isinstance(projection_cache.candidate_geometries, dict):
                projection_cache.candidate_geometries = dict(projection_cache.candidate_geometries)
            if hasattr(projection_cache, "coverage_masks") and isinstance(projection_cache.coverage_masks, dict):
                projection_cache.coverage_masks = dict(projection_cache.coverage_masks)
            if hasattr(projection_cache, "candidate_ids") and isinstance(projection_cache.candidate_ids, list):
                projection_cache.candidate_ids = list(projection_cache.candidate_ids)
        if cohesive_xy_bounds_by_candidate_id is not None:
            cohesive_xy_bounds_by_candidate_id = dict(cohesive_xy_bounds_by_candidate_id)
    if collision_cache is not None:
        resolved_domain_fingerprint = (
            _candidate_domain_fingerprint(
                domains, cell=cell, periodic_axes=periodic_axes
            )
            if candidate_domain_fingerprint is None
            else str(candidate_domain_fingerprint)
        )
        contract_fingerprint = _collision_contract_fingerprint(
            periodic_axes=periodic_axes,
            thresholds_A=thresholds_A,
            algorithm_version=_COLLISION_ALGORITHM_VERSION,
        )
        if (
            collision_cache.candidate_domain_fingerprint
            == resolved_domain_fingerprint
            and collision_cache.collision_contract_fingerprint == contract_fingerprint
        ):
            active_cache = collision_cache
            cache_status = "lazy"
        else:
            cache_status = "rejected_cache_namespace"
    if selection_policy in {
        "projection-aware",
        "capacity-aware",
        "cohesive-frontier",
    }:
        if projection_cache is None:
            raise ValueError(
                f"{selection_policy} selection requires a projection cache"
            )
        if projection_cache.site_ids != site_ids:
            raise ValueError("projection cache Site Instance order mismatch")
    else:
        raise ValueError(f"Unknown selection policy {selection_policy!r}")
    cohesive_tolerances = None
    if selection_policy == "cohesive-frontier":
        cohesive_tolerances = resolve_cohesive_relative_tolerances(
            perimeter=cohesive_perimeter_relative_tolerance,
            contact=cohesive_contact_relative_tolerance,
            area=cohesive_area_relative_tolerance,
            relative_tolerance=cohesive_relative_tolerance,
        )
        extra_symbols = set()
        if single_metal_context and single_metal_context.get("conformers"):
            for conf in single_metal_context["conformers"]:
                if hasattr(conf, "get_chemical_symbols"):
                    extra_symbols.update(conf.get_chemical_symbols())
                elif hasattr(conf, "symbols"):
                    extra_symbols.update(conf.symbols)
                elif isinstance(conf, dict) and "symbols" in conf:
                    extra_symbols.update(conf["symbols"])
        cohesive_projection_neighbor_index = CohesiveProjectionNeighborIndex(
            projection_cache=projection_cache,
            candidates=[
                candidate
                for candidates in domains.values()
                for candidate in candidates
            ],
            collision_thresholds_A=thresholds_A,
            contact_vdw_radius_scale=cohesive_contact_vdw_radius_scale,
            extra_symbols=extra_symbols,
        )
    if (
        candidate_collision_index is None
        and projection_cache is not None
        and selection_policy == "cohesive-frontier"
    ):
        all_domain_candidates = [
            candidate
            for candidates in domains.values()
            for candidate in candidates
        ]
        if (
            all_domain_candidates
            and getattr(projection_cache, "candidate_geometries", None)
            and getattr(projection_cache, "surface_lattice_uv_A", None) is not None
        ):
            candidate_collision_index = CandidateCollisionProjectionIndex(
                projection_cache=projection_cache,
                candidates=all_domain_candidates,
                collision_thresholds_A=thresholds_A,
            )
    elif candidate_collision_index is False or candidate_collision_index == "off":
        candidate_collision_index = None
    if capacity_audit_sink is not None:
        if selection_policy != "capacity-aware":
            raise ValueError(
                "capacity_audit_sink is only valid for capacity-aware selection"
            )
        if not callable(capacity_audit_sink):
            raise TypeError("capacity_audit_sink must be callable")
    if progress_sink is not None and not callable(progress_sink):
        raise TypeError("progress_sink must be callable")
    if (
        isinstance(capacity_beam_width, bool)
        or int(capacity_beam_width) != capacity_beam_width
        or int(capacity_beam_width) <= 0
    ):
        raise ValueError("capacity_beam_width must be a positive integer")

    next_candidate_index = 0
    if projection_cache is not None and getattr(projection_cache, "candidate_geometries", None):
        next_candidate_index = max(projection_cache.candidate_geometries.keys(), default=-1) + 1

    while True:
        live_candidates = [
            candidate
            for site_candidates in remaining.values()
            for candidate in site_candidates
        ]
        active_denticity = None
        if selection_policy == "cohesive-frontier":
            active_denticity = cohesive_active_denticity(live_candidates)
            phase_candidates = [
                candidate
                for candidate in live_candidates
                if (
                    len(candidate.get("target_donor_labels", ())) == active_denticity
                    or (
                        active_denticity == 1
                        and (
                            candidate.get("site_kind") == "single_metal_triangle"
                            or int(candidate.get("phase_priority", 0)) == 1
                        )
                    )
                )
            ] if active_denticity is not None else []
            feasible_sites = sorted(
                {candidate["site_instance_id"] for candidate in phase_candidates}
            )
        else:
            phase_candidates = None
            feasible_sites = sorted(
                site for site, candidates in remaining.items() if candidates
            )
        if not feasible_sites:
            if (
                single_metal_context is not None
                and single_metal_context.get("uncovered_metals")
                and not single_metal_phase_entered
            ):
                single_metal_phase_entered = True
                occupied_metals = set()
                for p in placed:
                    for mid in p.get("mapped_metal_atom_ids", ()):
                        occupied_metals.add(int(mid))
                    for mid in p.get("occupied_metal_ids", ()):
                        occupied_metals.add(int(mid))

                for metal_rec in single_metal_context["uncovered_metals"]:
                    fid = int(metal_rec["full_metal_atom_id_1based"])
                    if fid in occupied_metals:
                        continue
                    sm_site_id = f"site-single-metal-triangle-atom{fid:04d}"
                    sm_candidates = _generate_single_metal_triangle_candidates(
                        metal_record=metal_rec,
                        substrate_positions=single_metal_context["substrate_positions"],
                        substrate_symbols=single_metal_context["substrate_symbols"],
                        cell=single_metal_context["cell"],
                        periodic_axes=single_metal_context["periodic_axes"],
                        surface_frame=single_metal_context["surface_frame"],
                        conformers=single_metal_context["conformers"],
                        metrics_by_rank=single_metal_context["metrics_by_rank"],
                        substrate_height_clearance_A=single_metal_context["substrate_height_clearance_A"],
                        rotation_step_deg=single_metal_context["rotation_step_deg"],
                        translation_offsets_A=single_metal_context["translation_offsets_A"],
                    )
                    surviving = []
                    for cand in sm_candidates:
                        collides = False
                        for p in placed:
                            c_count, _, _ = periodic_xy_sam_sam_collision(
                                cand["coordinates"],
                                cand["symbols"],
                                p["coordinates"],
                                p["symbols"],
                                cell,
                                periodic_axes,
                                thresholds_A=thresholds_A,
                                surface_frame=single_metal_context.get("surface_frame"),
                            )
                            if c_count > 0:
                                collides = True
                                break
                        if not collides:
                            cand_idx = next_candidate_index
                            next_candidate_index += 1
                            cand["candidate_index"] = cand_idx
                            if projection_cache is not None:
                                footprint = filled_outer_envelope_area(
                                    positions=cand["coordinates"],
                                    symbols=cand["symbols"],
                                    surface_frame=single_metal_context["surface_frame"],
                                    radii_A=_resolved_ase_vdw_radii(cand["symbols"]),
                                    radius_scale=float(getattr(projection_cache, "contract", {}).get("radius_scale", 1.0)),
                                    boundary_samples_per_atom=int(getattr(projection_cache, "contract", {}).get("boundary_samples_per_atom", 720)),
                                )
                                geometry, scale = periodic_surface_polygon(
                                    footprint["hull_vertices_uv_A"],
                                    cell=cell,
                                    periodic_axes=periodic_axes,
                                    surface_frame=single_metal_context["surface_frame"],
                                )
                                projection_cache.candidate_geometries[cand_idx] = geometry
                                if hasattr(projection_cache, "coverage_masks") and isinstance(projection_cache.coverage_masks, dict):
                                    projection_cache.coverage_masks[cand_idx] = 0
                                if hasattr(projection_cache, "candidate_ids") and isinstance(projection_cache.candidate_ids, list):
                                    projection_cache.candidate_ids.append(cand["candidate_id"])
                                if cohesive_xy_bounds_by_candidate_id is not None:
                                    cohesive_xy_bounds_by_candidate_id[cand["candidate_id"]] = cohesive_xy_bound(cand)
                                if cohesive_projection_neighbor_index is not None:
                                    cohesive_projection_neighbor_index.candidate_by_id[cand["candidate_id"]] = cand
                            surviving.append(cand)
                    if surviving:
                        remaining[sm_site_id] = surviving
                        if sm_site_id not in site_index:
                            site_index[sm_site_id] = len(site_index)

                if remaining:
                    continue
            break
        candidate_count_before_step = sum(
            len(candidates) for candidates in remaining.values()
        )
        policy_audit = None
        capacity_audit = None
        capacity_audit_artifact = None
        capacity_audit_step_summary = None
        materialized_transition = None
        materialized_transition_step_summary = None
        if selection_policy == "capacity-aware":
            chosen, capacity_audit = choose_capacity_candidate(
                remaining_domains=remaining,
                site_conflicts=site_conflicts,
                metal_constraints=metal_constraints or {},
                cell=cell,
                periodic_axes=periodic_axes,
                collision_thresholds_A=thresholds_A,
                accepted_union=accepted_union,
                projection_cache=projection_cache,
                conformer_weights=conformer_weights,
                rng=rng,
                capacity_beam_width=capacity_beam_width,
                area_tie_tolerance_A2=area_tie_tolerance_A2,
                perimeter_tie_tolerance_A=perimeter_tie_tolerance_A,
                capacity_time_limit_seconds=capacity_time_limit_seconds,
                collision_cache=active_cache,
                collision_profiler=collision_profiler,
            )
            chosen = dict(chosen)
            site = chosen["site_instance_id"]
            rank = int(chosen["selection_rank"])
            conditional_probabilities = {}
            static_capacity = int(capacity_audit["K_before"])
            capacity_seconds = float(
                capacity_audit["timing"]["total_selection_seconds"]
            )
        else:
            if selection_policy == "cohesive-frontier":
                static_capacity = None
                capacity_seconds = 0.0
            else:
                capacity_started = time.perf_counter()
                static_capacity = _static_conflict_compatible_capacity(
                    feasible_sites, metal_constraints or {}
                )
                capacity_seconds = time.perf_counter() - capacity_started
            if selection_policy == "projection-aware":
                candidate_pool = [
                    candidate
                    for site in feasible_sites
                    for candidate in remaining[site]
                ]
                required_fields = (
                    "candidate_id",
                    "candidate_index",
                    "anchor_cartesian_A",
                    "surface_frame",
                    "conformer_sha256",
                    "oxygen_permutation",
                )
                for candidate in candidate_pool:
                    for key in required_fields:
                        if key not in candidate:
                            raise ValueError(
                                f"Candidate missing required projection field {key}"
                            )
                chosen, policy_audit = choose_projection_candidate(
                    candidates=candidate_pool,
                    site_conflicts=site_conflicts,
                    feasible_site_mask=_sites_mask(feasible_sites, site_index),
                    accepted_coverage_mask=accepted_coverage_mask,
                    accepted_union=accepted_union,
                    area_scale_A2=projection_cache.area_scale_A2,
                    conformer_weights=conformer_weights,
                    rng=rng,
                    area_tie_tolerance_A2=area_tie_tolerance_A2,
                    perimeter_tie_tolerance_A=perimeter_tie_tolerance_A,
                    projection_cache=projection_cache,
                    selection_criterion=selection_criterion,
                )
                accepted_coverage_mask |= policy_audit["winner_coverage_mask"]
                _, accepted_union = periodic_union_increment_A2(
                    accepted_union,
                    projection_cache.candidate_geometries[
                        chosen["candidate_index"]
                    ],
                    projection_cache.area_scale_A2,
                )
                chosen = dict(chosen)
                site = chosen["site_instance_id"]
                rank = int(chosen["selection_rank"])
                conditional_probabilities = {}
            elif selection_policy == "cohesive-frontier":
                chosen, policy_audit = choose_cohesive_frontier_candidate(
                    candidates=phase_candidates,
                    placed=placed,
                    accepted_union=accepted_union,
                    projection_cache=projection_cache,
                    cell=cell,
                    periodic_axes=periodic_axes,
                    collision_thresholds_A=thresholds_A,
                    rng=rng,
                    relative_tolerance=cohesive_relative_tolerance,
                    perimeter_relative_tolerance=cohesive_tolerances[
                        "perimeter"
                    ],
                    contact_relative_tolerance=cohesive_tolerances[
                        "contact"
                    ],
                    area_relative_tolerance=cohesive_tolerances[
                        "area"
                    ],
                    energy_tolerance_eV=cohesive_energy_tolerance_eV,
                    contact_vdw_radius_scale=cohesive_contact_vdw_radius_scale,
                    xy_bounds_by_candidate_id=cohesive_xy_bounds_by_candidate_id,
                    projection_neighbor_index=cohesive_projection_neighbor_index,
                    cohesive_frontier_mode=cohesive_frontier_mode,
                    frontier_snap_radius_A=frontier_snap_radius_A,
                )
                accepted_coverage_mask |= projection_cache.coverage_masks.get(
                    chosen["candidate_index"], 0
                )
                _, accepted_union = periodic_union_increment_A2(
                    accepted_union,
                    projection_cache.candidate_geometries[
                        chosen["candidate_index"]
                    ],
                    projection_cache.area_scale_A2,
                )
                chosen = dict(chosen)
                site = chosen["site_instance_id"]
                rank = int(chosen["selection_rank"])
                conditional_probabilities = {}

        if selection_policy == "capacity-aware":
            commit_started = time.perf_counter()
            transition = simulate_candidate_transition(
                remaining_domains=remaining,
                chosen=chosen,
                site_conflicts=site_conflicts,
                metal_constraints=metal_constraints or {},
                cell=cell,
                periodic_axes=periodic_axes,
                collision_thresholds_A=thresholds_A,
                mode="materialize",
                capacity_time_limit_seconds=capacity_time_limit_seconds,
                collision_cache=active_cache,
                collision_profiler=collision_profiler,
            )
            winner_preview = next(
                record
                for record in capacity_audit["beam_candidates"]
                if record["candidate_id"] == chosen["candidate_id"]
            )
            if transition["live_site_ids"] != winner_preview["collision_preview"]["live_site_ids"]:
                raise RuntimeError(
                    "materialized transition live sites disagree with winner preview"
                )
            if (
                transition["K_after_static"] != winner_preview["K_after_static"]
                or transition["K_after_collision"]
                != winner_preview["K_after_collision"]
            ):
                raise RuntimeError(
                    "materialized transition capacity disagrees with winner preview"
                )
            _, committed_union = periodic_union_increment_A2(
                accepted_union,
                projection_cache.candidate_geometries[
                    chosen["candidate_index"]
                ],
                projection_cache.area_scale_A2,
            )
            previous_remaining = remaining
            next_remaining = transition["remaining_domains"]
            step_number = len(placed) + 1
            actual_shared_metal_blocked = {
                blocked
                for blocked in transition["static_blocked_site_ids"]
                if blocked != site and previous_remaining.get(blocked)
            }
            collision_pruned = int(
                transition["collision_pruned_candidate_count"]
            )
            sites_pruned_to_empty = list(
                transition["collision_pruned_to_empty_sites"]
            )
            materialized_transition = {
                key: value
                for key, value in transition.items()
                if key != "remaining_domains"
            }
            capacity_audit["timing"]["winner_materialize_seconds"] = (
                time.perf_counter() - commit_started
            )
            capacity_audit["query_counts"][
                "winner_materialize_collision_query_count"
            ] = transition["collision_query_count"]
            capacity_audit["query_counts"][
                "winner_materialize_collision_query_seconds"
            ] = transition["collision_query_seconds"]
            capacity_audit["query_counts"][
                "winner_materialize_capacity_query_count"
            ] = 2
            capacity_audit["timing"][
                "winner_materialize_capacity_seconds"
            ] = (
                transition["capacity_after_static"]["solve_time_seconds"]
                + transition["capacity_after_collision"]["solve_time_seconds"]
            )

            if capacity_audit_sink is not None:
                capacity_audit_step_summary = _capacity_audit_summary(
                    capacity_audit
                )
                materialized_transition_step_summary = (
                    _materialized_transition_summary(transition)
                )
                audit_document = _capacity_ranking_audit_document(
                    step=step_number,
                    seed=seed,
                    capacity_audit=capacity_audit,
                    materialized_transition=materialized_transition,
                )
                # Validate the complete document before invoking a sink with
                # side effects.  Sink and reference failures therefore occur
                # before this transition mutates accepted state.
                _strict_json_bytes(audit_document)
                capacity_audit_artifact = (
                    _validate_capacity_audit_artifact_reference(
                        capacity_audit_sink(step_number, audit_document),
                        step=step_number,
                    )
                )

            # This is the commit boundary.  All exact capacity checks,
            # transition agreement, strict serialization, and sink writes have
            # succeeded before accepted state changes.
            placed.append(chosen)
            for blocked in transition["static_blocked_site_ids"]:
                removed = len(previous_remaining.get(blocked, []))
                blocked_sites.setdefault(
                    blocked,
                    {
                        "blocked_at_step": step_number,
                        "blocking_site": site,
                        "removed_candidate_count": removed,
                    },
                )
            remaining = next_remaining
            accepted_coverage_mask |= projection_cache.coverage_masks[
                chosen["candidate_index"]
            ]
            accepted_union = committed_union
        else:
            placed.append(chosen)
            if (
                cohesive_projection_neighbor_index is not None
                and chosen.get("candidate_id") in getattr(cohesive_projection_neighbor_index, "candidate_by_id", {})
            ):
                cohesive_projection_neighbor_index.add(chosen)
            newly_blocked = {site}.union(site_conflicts.get(site, set()))
            actual_shared_metal_blocked = {
                blocked
                for blocked in site_conflicts.get(site, set())
                if remaining.get(blocked)
            }
            chosen_metals = set(chosen.get("occupied_metal_ids", ())) | set(chosen.get("mapped_metal_atom_ids", ()))
            if chosen_metals:
                for rem_site, rem_cands in list(remaining.items()):
                    if rem_cands:
                        first_cand = rem_cands[0]
                        first_metals = set(first_cand.get("occupied_metal_ids", ())) | set(first_cand.get("mapped_metal_atom_ids", ()))
                        if chosen_metals & first_metals:
                            newly_blocked.add(rem_site)

            for blocked in sorted(newly_blocked):
                if blocked in remaining:
                    removed = len(remaining[blocked])
                    remaining[blocked] = []
                    blocked_sites.setdefault(
                        blocked,
                        {
                            "blocked_at_step": len(placed),
                            "blocking_site": site,
                            "removed_candidate_count": removed,
                        },
                    )

            collision_pruned = 0
            sites_pruned_to_empty = []
            chosen_coordinates = chosen["coordinates"]
            chosen_symbols = chosen["symbols"]
            potential_collision_candidate_ids = (
                candidate_collision_index.query_candidate_ids(chosen)
                if (
                    candidate_collision_index is not None
                    and str(chosen.get("candidate_id", chosen.get("candidate_index"))) in candidate_collision_index.candidate_by_id
                )
                else None
            )
            for other_site in sorted(remaining):
                if not remaining[other_site]:
                    continue
                if (
                    potential_collision_candidate_ids is not None
                    and not any(
                        str(candidate.get("candidate_id", candidate.get("candidate_index")))
                        in potential_collision_candidate_ids
                        for candidate in remaining[other_site]
                    )
                ):
                    continue
                kept = []
                for candidate in remaining[other_site]:
                    cand_id = str(
                        candidate.get("candidate_id", candidate.get("candidate_index"))
                    )
                    if (
                        potential_collision_candidate_ids is not None
                        and cand_id not in potential_collision_candidate_ids
                    ):
                        kept.append(candidate)
                        continue
                    if (
                        chosen.get("site_kind") == "single_metal_triangle"
                        or candidate.get("site_kind") == "single_metal_triangle"
                    ):
                        c_count, _, _ = periodic_xy_sam_sam_collision(
                            candidate["coordinates"],
                            candidate["symbols"],
                            chosen_coordinates,
                            chosen_symbols,
                            cell,
                            periodic_axes,
                            thresholds_A=thresholds_A,
                            surface_frame=single_metal_context.get("surface_frame") if single_metal_context else None,
                        )
                        collides = c_count > 0
                    elif active_cache is not None and "candidate_index" in candidate and "candidate_index" in chosen:
                        collides = active_cache.query(
                            candidate["candidate_index"], chosen["candidate_index"]
                        )
                        if collision_profiler is not None:
                            collision_profiler.observe(
                                candidate["candidate_index"],
                                chosen["candidate_index"],
                                0.0,
                            )
                    else:
                        if collision_profiler is not None:
                            query_started = time.perf_counter()
                            count, _, _ = surface_collision_report(
                                candidate["coordinates"],
                                candidate["symbols"],
                                chosen_coordinates,
                                chosen_symbols,
                                cell,
                                periodic_axes,
                                thresholds_A=thresholds_A,
                            )
                            if "candidate_index" in candidate and "candidate_index" in chosen:
                                collision_profiler.observe(
                                    candidate["candidate_index"],
                                    chosen["candidate_index"],
                                    time.perf_counter() - query_started,
                                )
                        else:
                            count, _, _ = surface_collision_report(
                                candidate["coordinates"],
                                candidate["symbols"],
                                chosen_coordinates,
                                chosen_symbols,
                                cell,
                                periodic_axes,
                                thresholds_A=thresholds_A,
                            )
                        collides = count > 0
                    if collides:
                        collision_pruned += 1
                        continue
                    kept.append(candidate)
                if remaining[other_site] and not kept:
                    sites_pruned_to_empty.append(other_site)
                remaining[other_site] = kept

        step = {
            "step": len(placed),
            "site_instance_id": site,
            "site_prototype_id": chosen["site_prototype_id"],
            "selection_rank": rank,
            "cluster_id": int(chosen["cluster_id"]),
            "source": chosen["source"],
            "target_donor_labels": list(chosen.get("target_donor_labels", [])),
            "oxygen_permutation": [int(value) for value in chosen["oxygen_permutation"]],
            "registered_interface_bonds": [
                dict(record)
                for record in chosen.get("registered_interface_bonds", [])
            ],
            "headgroup_rmsd_A": float(chosen["headgroup_rmsd_A"]) if chosen.get("headgroup_rmsd_A") is not None else None,
            "site_kind": chosen.get("site_kind", "mapped_interface"),
            "occupied_metal_ids": list(chosen.get("occupied_metal_ids", [])),
            "conformer_anchor_height_A": chosen.get("conformer_anchor_height_A"),
            "conditional_conformer_probabilities": conditional_probabilities,
            "feasible_site_count_before_step": len(feasible_sites),
            "candidate_count_before_step": candidate_count_before_step,
            "shared_metal_blocked_site_count": len(actual_shared_metal_blocked),
            "collision_pruned_candidate_count": collision_pruned,
            "collision_pruned_to_empty_sites": sites_pruned_to_empty,
            "remaining_feasible_site_count": sum(bool(candidates) for candidates in remaining.values()),
            "remaining_candidate_count": sum(len(candidates) for candidates in remaining.values()),
            "static_capacity": static_capacity,
            "static_capacity_seconds": capacity_seconds,
        }
        if policy_audit is not None:
            step["policy_audit"] = policy_audit
        if selection_policy == "cohesive-frontier":
            step["denticity_phase"] = int(active_denticity)
            step["phase_candidate_count_before_step"] = len(phase_candidates)
            step["capacity_not_computed_reason"] = (
                "cohesive-frontier_has_no_future_capacity_objective"
            )
        if capacity_audit is not None:
            if capacity_audit_sink is None:
                step["capacity_audit"] = capacity_audit
                step["materialized_transition"] = materialized_transition
            else:
                step["capacity_audit"] = capacity_audit_step_summary
                step["capacity_audit_artifact"] = capacity_audit_artifact
                step["materialized_transition"] = (
                    materialized_transition_step_summary
                )
        steps.append(step)
        if progress_sink is not None:
            progress_sink(
                {
                    "schema": "sam-sequential-live-progress-v1",
                    "status": "running",
                    "seed": int(seed),
                    "step": int(step["step"]),
                    "current_N": len(placed),
                    "site_instance_id": step["site_instance_id"],
                    "candidate_id": chosen.get("candidate_id"),
                    "denticity_phase": step.get("denticity_phase"),
                    "growth_mode": (
                        policy_audit.get("growth_mode")
                        if isinstance(policy_audit, dict)
                        else None
                    ),
                    "remaining_candidate_count": int(
                        step["remaining_candidate_count"]
                    ),
                    "remaining_feasible_site_count": int(
                        step["remaining_feasible_site_count"]
                    ),
                    "elapsed_seconds": float(time.perf_counter() - started_perf),
                }
            )
        if capacity_audit_sink is not None and selection_policy == "capacity-aware":
            # The immutable artifact now owns the complete scientific evidence;
            # retain only the bounded step summary/reference.  In particular,
            # do not carry the previous full audit into construction of the
            # next step through audit_document or transition aliases.
            del winner_preview
            del audit_document
            del materialized_transition
            del transition
            del capacity_audit
            _capacity_audit_release_checkpoint(
                step=step["step"], retained_step=step
            )
    result = {
        "status": "jammed_no_feasible_candidate",
        "seed": int(seed),
        "initial_site_count": len(domains),
        "initial_candidate_count": initial_candidate_count,
        "emergent_molecule_count": len(placed),
        "steps": steps,
        "placed": placed,
        "blocked_sites": blocked_sites,
        "remaining_candidate_count": sum(len(candidates) for candidates in remaining.values()),
        "collision_cache_status": cache_status,
        "collision_cache_summary": (
            dict(active_cache.summary()) if active_cache is not None else {}
        ),
    }
    if selection_policy == "cohesive-frontier":
        result["cohesive_frontier_summary"] = {
            "denticity_order": [3, 2, 1],
            "capacity_evaluated": False,
            "perimeter_relative_tolerance": cohesive_tolerances["perimeter"],
            "contact_relative_tolerance": cohesive_tolerances["contact"],
            "area_relative_tolerance": cohesive_tolerances["area"],
            "relative_tolerances": dict(cohesive_tolerances),
            "legacy_relative_tolerance": (
                None
                if cohesive_relative_tolerance is None
                else _cohesive_relative_tolerance(cohesive_relative_tolerance)
            ),
            "new_nucleus_event_count": sum(
                bool(step["policy_audit"]["new_nucleus_event"])
                for step in steps
            ),
            "frontier_growth_step_count": sum(
                step["policy_audit"]["growth_mode"] == "frontier"
                for step in steps
            ),
        }
    if selection_policy in {
        "projection-aware",
        "capacity-aware",
        "cohesive-frontier",
    }:
        from shapely import wkb

        result["accepted_union_wkb_hex"] = wkb.dumps(accepted_union, hex=True)
        result["periodic_hole_diagnostics"] = periodic_hole_diagnostics(
            accepted_union,
            projection_cache.area_scale_A2,
            surface_lattice_uv_A=projection_cache.surface_lattice_uv_A,
        )
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict:
    path = Path(path).expanduser().resolve()
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _capture_directory_tree_ledger(root: Path) -> dict:
    """Capture every immutable parent path, including empty directories."""

    root = Path(root).expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError(f"Immutable tree root is not a real directory: {root}")
    records = []

    def visit(directory: Path, relative: Path) -> None:
        metadata = directory.lstat()
        records.append(
            {
                "path": "." if not relative.parts else relative.as_posix(),
                "kind": "directory",
                "mode": stat.S_IMODE(metadata.st_mode),
            }
        )
        with os.scandir(directory) as iterator:
            entries = sorted(iterator, key=lambda entry: entry.name)
        for entry in entries:
            path = directory / entry.name
            child_relative = relative / entry.name
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise RuntimeError(
                    "Immutable parent tree may not contain symlink paths: "
                    f"{child_relative.as_posix()}"
                )
            if stat.S_ISDIR(metadata.st_mode):
                visit(path, child_relative)
            elif stat.S_ISREG(metadata.st_mode):
                records.append(
                    {
                        "path": child_relative.as_posix(),
                        "kind": "file",
                        "mode": stat.S_IMODE(metadata.st_mode),
                        "sha256": _sha256(path),
                        "bytes": int(metadata.st_size),
                    }
                )
            else:
                raise RuntimeError(
                    "Immutable parent tree contains an unsupported path kind: "
                    f"{child_relative.as_posix()}"
                )

    visit(root, Path())
    digest_material = {
        "schema": "sam-immutable-directory-tree-ledger-v1",
        "records": records,
    }
    return {
        **digest_material,
        "root": str(root),
        "tree_sha256": _payload_sha256(digest_material),
        "path_count": len(records),
        "directory_count": sum(record["kind"] == "directory" for record in records),
        "file_count": sum(record["kind"] == "file" for record in records),
    }


def _paths_overlap(left: Path, right: Path) -> bool:
    """Return whether two resolved paths are equal or ancestor/descendant."""

    left = Path(left).resolve()
    right = Path(right).resolve()
    try:
        right.relative_to(left)
        return True
    except ValueError:
        pass
    try:
        left.relative_to(right)
        return True
    except ValueError:
        return False


def _validate_repair_output_evidence_isolation(
    *, output_dir: Path, parent_run: Path, reference_metrics: Path
) -> dict:
    """Reject any resolved output/evidence namespace overlap before reservation."""

    output = Path(output_dir).expanduser().resolve()
    parent = Path(parent_run).expanduser().resolve()
    requested_reference = Path(reference_metrics).expanduser().resolve()
    reference_package = (
        requested_reference
        if requested_reference.is_dir()
        else requested_reference.parent
    )
    evidence_paths = {
        "parent_run": parent,
        "reference_metrics_requested": requested_reference,
        "reference_metrics_package": reference_package,
    }
    overlaps = [
        {"role": role, "path": str(path)}
        for role, path in evidence_paths.items()
        if _paths_overlap(output, path)
    ]
    if overlaps:
        raise ValueError(
            "Repair output must be isolated from immutable parent/reference "
            "evidence (equal and ancestor/descendant paths are forbidden): "
            + json.dumps(
                {"output": str(output), "overlaps": overlaps}, sort_keys=True
            )
        )
    return {
        "output": str(output),
        "parent_run": str(parent),
        "reference_metrics_requested": str(requested_reference),
        "reference_metrics_package": str(reference_package),
        "resolved_before_output_reservation": True,
        "symlinks_resolved": True,
        "overlap": False,
    }


_PARENT_PRIMARY_INPUT_NAMES = (
    "substrate",
    "layer_groups",
    "site_instances",
    "site_prototypes",
    "conformer_analysis",
)

_SCHEMA2_ADAPTER_TOP_SHA256 = (
    "7f839bbdfc3a222c3e4f7739eca37e6331ffa67d83dd97f6d02f68449d52fcdc"
)
_SCHEMA2_ADAPTER_TRAJECTORY_SHA256 = (
    "a15839aae1a9c666a843d3913a4e5182b912af72905242d93926ba96e6415ead"
)
_SCHEMA2_ADAPTER_FINAL_SHA256 = (
    "ac273e3e03fe2f5bd0fed2ae963083a75543d232634f1ec9f1ff319e6026f929"
)
_SCHEMA2_ADAPTER_IMPLEMENTATION_SHA256 = (
    "539dc0d45d08345e1b147fea4afab556a4dc3c72a3df73f9629f5ed8989d131f"
)
_SCHEMA2_ADAPTER_IMPLEMENTATION_BLOB = "15cdba58960044ce8fe1a3bae279749bec939028"
_SCHEMA2_ADAPTER_COMMITS = {
    "implementation_alignment_commit": {
        "commit": "fc513e61a6c0a11b1aa98690ee4c40645a10d50f",
        "paths": ("scripts/monolayer_sequential_growth.py",),
    },
    "review_record_commit": {
        "commit": "e99ad72dab64fb4068de5cf2bfdb11b424665748",
        "paths": (
            "scripts/monolayer_sequential_growth.py",
            "docs/dbf34-skip-collision-aligned-assembly-2026-08-08.zh-CN.md",
        ),
    },
}


def _git_command(repository_root: Path, *arguments: str, binary=False):
    completed = subprocess.run(
        ["git", "-C", str(repository_root), *arguments],
        check=False,
        capture_output=True,
        timeout=30.0,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(
            f"Git provenance query failed ({' '.join(arguments)}): {message}"
        )
    return completed.stdout if binary else completed.stdout.decode("utf-8").strip()


def _schema2_git_implementation_evidence() -> dict:
    repository_root = _repository_root_for_script(Path(__file__).resolve())
    if repository_root is None or not (repository_root / ".git").exists():
        raise RuntimeError(
            "Historical schema-2 adapter requires the project Git object database"
        )
    object_type = _git_command(
        repository_root, "cat-file", "-t", _SCHEMA2_ADAPTER_IMPLEMENTATION_BLOB
    )
    if object_type != "blob":
        raise RuntimeError("Historical implementation object is not a Git blob")
    blob_bytes = _git_command(
        repository_root,
        "cat-file",
        "blob",
        _SCHEMA2_ADAPTER_IMPLEMENTATION_BLOB,
        binary=True,
    )
    blob_sha256 = hashlib.sha256(blob_bytes).hexdigest()
    if blob_sha256 != _SCHEMA2_ADAPTER_IMPLEMENTATION_SHA256:
        raise RuntimeError(
            "Historical Git blob bytes do not match the parent-declared SHA-256"
        )
    commit_records = {}
    for role, declaration in _SCHEMA2_ADAPTER_COMMITS.items():
        commit = declaration["commit"]
        resolved_commit = _git_command(
            repository_root, "rev-parse", f"{commit}^{{commit}}"
        )
        if resolved_commit != commit:
            raise RuntimeError(f"Historical commit evidence is ambiguous for {role}")
        path_records = []
        for path in declaration["paths"]:
            output = _git_command(repository_root, "ls-tree", commit, "--", path)
            lines = [line for line in output.splitlines() if line]
            if len(lines) != 1:
                raise RuntimeError(
                    f"Historical commit {commit} lacks unique path evidence for {path}"
                )
            metadata, recorded_path = lines[0].split("\t", 1)
            mode, kind, object_id = metadata.split()
            if recorded_path != path or kind != "blob":
                raise RuntimeError(
                    f"Historical commit path evidence is malformed for {path}"
                )
            if path == "scripts/monolayer_sequential_growth.py" and (
                object_id != _SCHEMA2_ADAPTER_IMPLEMENTATION_BLOB
            ):
                raise RuntimeError(
                    f"Historical commit {commit} points to the wrong implementation blob"
                )
            path_records.append(
                {
                    "path": recorded_path,
                    "mode": mode,
                    "object_type": kind,
                    "git_object": object_id,
                }
            )
        commit_records[role] = {
            "commit": resolved_commit,
            "path_records": path_records,
        }
    return {
        "mode": "git_object_database_exact_blob_and_commit_path_evidence",
        "repository_root": str(repository_root.resolve()),
        "blob": {
            "git_object": _SCHEMA2_ADAPTER_IMPLEMENTATION_BLOB,
            "object_type": object_type,
            "sha256": blob_sha256,
            "bytes": len(blob_bytes),
        },
        "commits": commit_records,
        "current_worktree_path_hash_required": False,
    }


def _resolve_parent_declared_path(
    declaration,
    *,
    parent_root: Path,
    local_base: Path,
    label: str,
) -> tuple[Path, dict]:
    if isinstance(declaration, str):
        raw_path = declaration
        reference = {"path": declaration}
    elif isinstance(declaration, dict):
        raw_path = declaration.get("path")
        reference = dict(declaration)
    else:
        raise ValueError(f"Parent {label} declaration is missing or malformed")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"Parent {label} path is missing")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path_base = reference.get("path_base")
        if path_base == "run_root":
            path = parent_root / path
        else:
            path = local_base / path
    return path.resolve(), reference


def _hash_check_parent_file(
    path: Path,
    reference: dict,
    *,
    label: str,
    require_declared_hash: bool,
) -> dict:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"Parent {label} is missing or is not a regular file: {path}")
    actual = _file_identity(path)
    declared_hash = reference.get("sha256")
    if require_declared_hash and not isinstance(declared_hash, str):
        raise ValueError(f"Parent {label} lacks required SHA-256 evidence")
    if declared_hash is not None and actual["sha256"] != str(declared_hash):
        raise RuntimeError(
            f"Parent {label} SHA-256 mismatch: expected {declared_hash}, "
            f"found {actual['sha256']}"
        )
    declared_bytes = reference.get("bytes")
    if declared_bytes is not None and int(declared_bytes) != int(actual["bytes"]):
        raise RuntimeError(
            f"Parent {label} byte-count mismatch: expected {declared_bytes}, "
            f"found {actual['bytes']}"
        )
    return actual


def capture_immutable_parent_reference(parent_run, *, expected_inputs) -> dict:
    """Hash-check one unambiguous schema-4 or explicit schema-2 parent package."""

    parent_root = Path(parent_run).expanduser().resolve()
    if parent_root.is_symlink() or not parent_root.is_dir():
        raise FileNotFoundError(f"Parent run is not a real directory: {parent_root}")
    top_path = parent_root / "manifest.json"
    top_identity = _hash_check_parent_file(
        top_path, {}, label="top manifest", require_declared_hash=False
    )
    top = json.loads(top_path.read_text())
    schema_version = top.get("schema_version")
    if schema_version not in (2, 4):
        raise ValueError(
            "Parent manifest schema_version must be current 4 or explicit historical 2"
        )
    schema_version = int(schema_version)
    if schema_version == 2 and (
        top_identity["sha256"] != _SCHEMA2_ADAPTER_TOP_SHA256
    ):
        raise ValueError(
            "Historical schema-2 compatibility is restricted to the sealed "
            "thr-nosub-aligned Stage1 baseline package"
        )
    if schema_version == 4:
        if top.get("schema") != "sam-sequential-growth-run-manifest-v1":
            raise ValueError("Current parent top manifest schema is not sequential growth")
        if top.get("sealed") is not True:
            raise ValueError("Current parent top manifest is not sealed")
    if top.get("method") != "irreversible_sequential_adsorption_without_target_coverage":
        raise ValueError("Parent is not an irreversible sequential-growth package")
    trajectories = top.get("trajectories")
    if not isinstance(trajectories, list) or len(trajectories) != 1:
        raise ValueError(
            "--parent-run must identify exactly one trajectory; missing or ambiguous parent"
        )
    trajectory_summary = trajectories[0]
    if not isinstance(trajectory_summary, dict):
        raise ValueError("Parent trajectory summary is malformed")
    trajectory_path, trajectory_reference = _resolve_parent_declared_path(
        trajectory_summary.get("manifest"),
        parent_root=parent_root,
        local_base=parent_root,
        label="trajectory manifest",
    )
    trajectory_identity = _hash_check_parent_file(
        trajectory_path,
        trajectory_reference,
        label="trajectory manifest",
        require_declared_hash=schema_version == 4,
    )
    trajectory = json.loads(trajectory_path.read_text())
    if schema_version == 2:
        expected_trajectory = parent_root / "trajectory-0001" / "manifest.json"
        if (
            trajectory_path != expected_trajectory
            or trajectory_identity["sha256"]
            != _SCHEMA2_ADAPTER_TRAJECTORY_SHA256
        ):
            raise ValueError(
                "Historical schema-2 trajectory does not match the narrow adapter"
            )
    if int(trajectory.get("schema_version", -1)) != schema_version:
        raise ValueError("Parent top and trajectory schema versions disagree")
    if schema_version == 4 and trajectory.get("sealed") is not True:
        raise ValueError("Current parent trajectory manifest is not sealed")
    final_path, final_reference = _resolve_parent_declared_path(
        trajectory.get("final_structure"),
        parent_root=parent_root,
        local_base=trajectory_path.parent,
        label="final H0 structure",
    )
    final_identity = _hash_check_parent_file(
        final_path,
        final_reference,
        label="final H0 structure",
        require_declared_hash=True,
    )
    if schema_version == 2:
        expected_final = parent_root / "trajectory-0001" / "final-h0.extxyz"
        if (
            final_path != expected_final
            or final_identity["sha256"] != _SCHEMA2_ADAPTER_FINAL_SHA256
        ):
            raise ValueError(
                "Historical schema-2 final structure does not match the narrow adapter"
            )
    top_final = trajectory_summary.get("final_structure")
    if isinstance(top_final, dict):
        if top_final.get("sha256") != final_identity["sha256"]:
            raise RuntimeError("Parent top/final structure SHA-256 references disagree")
        if top_final.get("bytes") is not None and int(
            top_final["bytes"]
        ) != int(final_identity["bytes"]):
            raise RuntimeError("Parent top/final structure byte references disagree")

    parent_inputs = top.get("inputs")
    if not isinstance(parent_inputs, dict):
        raise ValueError("Parent top manifest lacks input evidence")
    if set(expected_inputs) != set(_PARENT_PRIMARY_INPUT_NAMES):
        raise ValueError("Expected parent input contract must contain exactly five inputs")
    input_identities = {}
    for name in _PARENT_PRIMARY_INPUT_NAMES:
        declaration = parent_inputs.get(name)
        path, reference = _resolve_parent_declared_path(
            declaration,
            parent_root=parent_root,
            local_base=parent_root,
            label=f"input {name}",
        )
        actual = _hash_check_parent_file(
            path,
            reference,
            label=f"input {name}",
            require_declared_hash=True,
        )
        expected = expected_inputs[name]
        if actual["sha256"] != str(expected["sha256"]):
            raise ValueError(
                f"Current scientific input {name} does not match parent SHA-256"
            )
        if expected.get("bytes") is not None and int(actual["bytes"]) != int(
            expected["bytes"]
        ):
            raise ValueError(
                f"Current scientific input {name} does not match parent bytes"
            )
        input_identities[name] = {
            **actual,
            "current_path": str(Path(expected["path"]).resolve()),
            "parent_declared_path": str(path),
            "paths_equal": Path(expected["path"]).resolve() == path,
            "identity_comparison": "sha256_and_bytes_not_host_absolute_path",
        }

    ledger = {
        "top_manifest": top_identity,
        "trajectory_manifest": trajectory_identity,
        "final_structure": final_identity,
    }
    ledger.update(
        {f"input.{name}": identity for name, identity in input_identities.items()}
    )
    code_identities = {}
    run_request_identity = None
    run_request_document = None
    schema2_git_evidence = None
    if schema_version == 4:
        run_request_path, run_request_reference = _resolve_parent_declared_path(
            top.get("run_request"),
            parent_root=parent_root,
            local_base=parent_root,
            label="run request",
        )
        run_request_identity = _hash_check_parent_file(
            run_request_path,
            run_request_reference,
            label="run request",
            require_declared_hash=True,
        )
        request = json.loads(run_request_path.read_text())
        run_request_document = request
        request_inputs = request.get("primary_inputs")
        if request_inputs != {
            name: parent_inputs[name] for name in _PARENT_PRIMARY_INPUT_NAMES
        }:
            raise ValueError("Parent run-request and top input evidence disagree")
        ledger["run_request"] = run_request_identity
        snapshot = top.get("code_snapshot")
        files = snapshot.get("files") if isinstance(snapshot, dict) else None
        if not isinstance(files, list) or not files:
            raise ValueError("Current parent lacks immutable code-snapshot evidence")
        implementation_hashes = {}
        implementation = top.get("implementation") or {}
        main = implementation.get("implementation")
        if isinstance(main, dict):
            implementation_hashes[Path(main.get("path", "")).name] = main.get("sha256")
        for module, record in (implementation.get("dependencies") or {}).items():
            runtime = record.get("implementation") if isinstance(record, dict) else None
            if isinstance(runtime, dict):
                implementation_hashes[str(module)] = runtime.get("sha256")
        for file_record in files:
            module = str(file_record.get("module", ""))
            path, reference = _resolve_parent_declared_path(
                file_record,
                parent_root=parent_root,
                local_base=parent_root,
                label=f"code snapshot {module}",
            )
            actual = _hash_check_parent_file(
                path,
                reference,
                label=f"code snapshot {module}",
                require_declared_hash=True,
            )
            expected_hash = implementation_hashes.get(module)
            if expected_hash is not None and actual["sha256"] != expected_hash:
                raise RuntimeError(
                    f"Parent code snapshot {module} disagrees with implementation evidence"
                )
            code_identities[module] = actual
            ledger[f"code.{module}"] = actual
        if set(implementation_hashes) - set(code_identities):
            raise ValueError("Current parent code-snapshot module set is incomplete")
        code_evidence_mode = "schema4_immutable_code_snapshot_hash_verified"
    else:
        implementation_reference = parent_inputs.get("implementation")
        if not isinstance(implementation_reference, dict):
            raise ValueError(
                "Historical schema-2 parent lacks its implementation declaration"
            )
        declared_path = implementation_reference.get("path")
        declared_sha256 = implementation_reference.get("sha256")
        if (
            not isinstance(declared_path, str)
            or Path(declared_path).name != "monolayer_sequential_growth.py"
            or declared_sha256 != _SCHEMA2_ADAPTER_IMPLEMENTATION_SHA256
        ):
            raise ValueError(
                "Historical schema-2 parent implementation declaration is not "
                "the reviewed alignment implementation"
            )
        schema2_git_evidence = _schema2_git_implementation_evidence()
        blob = schema2_git_evidence["blob"]
        code_identities["monolayer_sequential_growth.py"] = {
            "path": f"git-object:{blob['git_object']}",
            "declared_historical_path": declared_path,
            "sha256": blob["sha256"],
            "bytes": int(blob["bytes"]),
            "object_type": blob["object_type"],
        }
        code_evidence_mode = (
            "schema2_declared_sha256_git_blob_and_commit_paths_verified_"
            "without_current_worktree_path_hash"
        )

    parent_tree_ledger = _capture_directory_tree_ledger(parent_root)

    # Store one normalized immutable before-ledger; after verification reuses
    # these exact paths, hashes, and byte counts without trusting live manifests.
    ledger = {
        label: {
            "path": identity["path"],
            "sha256": identity["sha256"],
            "bytes": int(identity["bytes"]),
        }
        for label, identity in sorted(ledger.items())
    }
    return {
        "schema": "sam-local-repair-parent-reference-v1",
        "schema_version": schema_version,
        "parent_role": "irreversible_sequential_adsorption_parent",
        "parent_run": str(parent_root),
        "parent_top_status": top.get("status"),
        "parent_method": top.get("method"),
        "top_manifest": top,
        "trajectory_manifest": trajectory,
        "trajectory_number": int(trajectory_summary.get("trajectory", 1)),
        "trajectory_seed": int(
            trajectory_summary.get("seed", trajectory.get("seed", 0))
        ),
        "trajectory_path": str(trajectory_path),
        "final_path": str(final_path),
        "input_identities": input_identities,
        "code_identities": code_identities,
        "code_evidence_mode": code_evidence_mode,
        "run_request_document": run_request_document,
        "schema2_git_evidence": schema2_git_evidence,
        "historical_schema2_explicit_compatibility": schema_version == 2,
        "trajectory_manifest_cross_hash_available": schema_version == 4,
        "hash_ledger_before": ledger,
        "parent_tree_ledger_before": parent_tree_ledger,
    }


def _repair_cli_metric_contract(args) -> dict:
    contract = {
        "footprint_boundary_samples": int(args.footprint_boundary_samples),
        "projection_radius_scale": float(args.projection_radius_scale),
        "area_tie_tolerance_A2": float(args.area_tie_tolerance_A2),
        "perimeter_tie_tolerance_A": float(args.perimeter_tie_tolerance_A),
    }
    if contract["footprint_boundary_samples"] < 12:
        raise ValueError("Repair footprint sampling is below the formal minimum")
    for name in (
        "projection_radius_scale",
        "area_tie_tolerance_A2",
        "perimeter_tie_tolerance_A",
    ):
        if not np.isfinite(contract[name]):
            raise ValueError(f"Repair metric contract {name} must be finite")
    if contract["projection_radius_scale"] <= 0.0:
        raise ValueError("Repair projection radius scale must be positive")
    if contract["area_tie_tolerance_A2"] < 0.0:
        raise ValueError("Repair area tolerance must be non-negative")
    if contract["perimeter_tie_tolerance_A"] <= 0.0:
        raise ValueError("Repair perimeter tolerance must be positive")
    return contract


def _schema2_top_projection_metric_contract(top: dict, *, label: str) -> dict:
    policy = top.get("projection_policy")
    if not isinstance(policy, dict):
        raise ValueError(f"{label} lacks its schema-2 projection policy")
    cache = policy.get("cache")
    cache_contract = cache.get("contract") if isinstance(cache, dict) else None
    if not isinstance(cache_contract, dict):
        raise ValueError(f"{label} lacks its sealed projection-cache contract")
    samples = int(policy.get("footprint_boundary_samples", -1))
    radius = float(policy.get("radius_scale", float("nan")))
    area_tolerance = float(policy.get("area_tie_tolerance_A2", float("nan")))
    if (
        int(cache_contract.get("boundary_samples_per_atom", -2)) != samples
        or float(cache_contract.get("radius_scale", float("nan"))) != radius
    ):
        raise ValueError(f"{label} projection policy/cache contract disagrees")
    perimeter_value = policy.get("perimeter_tie_tolerance_A")
    if perimeter_value is None:
        # The reviewed schema-2 implementation predates this named CLI field.
        # The narrow Stage2 adapter fixes the later formal tie contract at the
        # exact maintained value instead of treating absence as a wildcard.
        perimeter = 1.0e-6
        perimeter_provenance = (
            "schema2_field_absent_narrow_adapter_requires_exact_1e-6"
        )
    else:
        perimeter = float(perimeter_value)
        perimeter_provenance = "schema2_top_projection_policy"
    contract = {
        "footprint_boundary_samples": samples,
        "projection_radius_scale": radius,
        "area_tie_tolerance_A2": area_tolerance,
        "perimeter_tie_tolerance_A": perimeter,
    }
    if not all(
        np.isfinite(value)
        for value in (
            radius,
            area_tolerance,
            perimeter,
        )
    ):
        raise ValueError(f"{label} projection metric values must be finite")
    return {
        "contract": contract,
        "perimeter_tolerance_provenance": perimeter_provenance,
        "projection_cache_contract": cache_contract,
    }


def _parent_requested_metric_contract(parent_reference: dict) -> dict:
    if int(parent_reference["schema_version"]) == 2:
        return _schema2_top_projection_metric_contract(
            parent_reference["top_manifest"], label="historical parent"
        )
    request = parent_reference.get("run_request_document")
    if not isinstance(request, dict):
        raise ValueError("Current parent lacks its sealed run-request document")
    if (
        request.get("schema") != "sam-sequential-growth-run-request-v1"
        or request.get("status") != "sealed_pre_expensive_request"
    ):
        raise ValueError("Current parent run request is not sealed growth evidence")
    # Current sealed run requests store the metric contract under the public
    # ``protocol`` key.  Keep the older name as a read-only compatibility
    # fallback so repair never silently drops a previously sealed request.
    protocol = request.get("protocol")
    if protocol is None:
        protocol = request.get("scientific_protocol")
    footprint = protocol.get("footprint_contract") if isinstance(protocol, dict) else None
    selection = protocol.get("selection") if isinstance(protocol, dict) else None
    if not isinstance(footprint, dict) or not isinstance(selection, dict):
        raise ValueError("Current parent run request lacks its metric contract")
    return {
        "contract": {
            "footprint_boundary_samples": int(
                footprint.get("boundary_samples_per_atom", -1)
            ),
            "projection_radius_scale": float(
                footprint.get("radius_scale", float("nan"))
            ),
            "area_tie_tolerance_A2": float(
                selection.get("area_tie_tolerance_A2", float("nan"))
            ),
            "perimeter_tie_tolerance_A": float(
                selection.get("perimeter_tie_tolerance_A", float("nan"))
            ),
        },
        "perimeter_tolerance_provenance": "schema4_sealed_run_request",
        "projection_cache_contract": None,
    }


def _resolve_r0_record_path(
    record: dict, *, package_root: Path, project_root: Path, label: str
) -> Path:
    if not isinstance(record, dict):
        raise ValueError(f"R0 {label} record is malformed")
    raw = record.get("path")
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"R0 {label} path is missing")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path_base = record.get("path_base")
        if path_base == "project_root":
            path = project_root / path
        elif path_base == "run_root":
            path = package_root / path
        else:
            raise ValueError(f"R0 {label} has unsupported path_base {path_base!r}")
    return path.resolve()


def _top10_hash_contract(top: dict, *, label: str) -> list[dict]:
    if int(top.get("max_conformers", -1)) != 10:
        raise ValueError(f"{label} is not the sealed top-10 conformer contract")
    table = (top.get("sampling_policy") or {}).get("table")
    if not isinstance(table, list) or len(table) != 10:
        raise ValueError(f"{label} lacks exactly ten sampling records")
    result = []
    for expected_rank, record in enumerate(table, 1):
        structure = record.get("structure") if isinstance(record, dict) else None
        sha256 = structure.get("sha256") if isinstance(structure, dict) else None
        if int(record.get("selection_rank", -1)) != expected_rank or (
            not isinstance(sha256, str) or len(sha256) != 64
        ):
            raise ValueError(f"{label} top-10 structure evidence is malformed")
        result.append({"selection_rank": expected_rank, "sha256": sha256})
    return result


def capture_reference_metrics(
    reference_metrics,
    *,
    parent_reference,
    expected_inputs,
    selected_source_lock,
    repair_metric_contract,
    current_substrate,
) -> dict:
    """Hash-check and jointly bind the sealed Stage1-R0 metric package."""

    requested = Path(reference_metrics).expanduser().resolve()
    if requested.is_dir():
        manifest_path = requested / "manifest.json"
        metrics_path = requested / "reference-metrics.json"
    elif requested.name == "manifest.json":
        manifest_path = requested
        preliminary_manifest = json.loads(manifest_path.read_text())
        metrics_path, _ = _resolve_parent_declared_path(
            preliminary_manifest.get("reference_metrics"),
            parent_root=manifest_path.parent,
            local_base=manifest_path.parent,
            label="reference metrics",
        )
    else:
        metrics_path = requested
        manifest_path = requested.parent / "manifest.json"
    package_root = manifest_path.parent.resolve()
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise FileNotFoundError(
            "Reference metrics require their sealed sibling manifest.json"
        )
    manifest_identity = _file_identity(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") not in {
        "sam-h0-corrected-reference-metrics-run-manifest-v1",
        "dbf34-h0-corrected-reference-metrics-run-manifest-v1",
    }:
        raise ValueError("Reference metrics manifest schema is unsupported")
    corrected_gate = manifest.get("corrected_reference_metric_gate")
    if (
        manifest.get("sealed") is not True
        or manifest.get("stage") != "Stage1-R0"
        or not isinstance(corrected_gate, dict)
        or corrected_gate.get("passed") is not True
        or corrected_gate.get("metric_version") != "periodic-torus-holes-v2"
    ):
        raise ValueError(
            "Reference metrics manifest is not sealed passed torus-v2 Stage1-R0 evidence"
        )
    declared_path, reference = _resolve_parent_declared_path(
        manifest.get("reference_metrics"),
        parent_root=package_root,
        local_base=package_root,
        label="reference metrics",
    )
    if declared_path != metrics_path:
        raise ValueError("Requested reference metrics path disagrees with its manifest")
    metrics_identity = _hash_check_parent_file(
        metrics_path,
        reference,
        label="reference metrics",
        require_declared_hash=True,
    )
    inventory = manifest.get("artifact_inventory")
    matching_inventory = [
        record
        for record in inventory or []
        if isinstance(record, dict) and record.get("path") == "reference-metrics.json"
    ]
    if len(matching_inventory) != 1 or {
        "sha256": matching_inventory[0].get("sha256"),
        "bytes": matching_inventory[0].get("bytes"),
    } != {
        "sha256": metrics_identity["sha256"],
        "bytes": metrics_identity["bytes"],
    }:
        raise ValueError("R0 manifest inventory does not seal the payload hash/bytes")

    payload = json.loads(metrics_path.read_text())
    if (
        payload.get("sealed") is not True
        or payload.get("stage") != "Stage1-R0"
        or payload.get("schema")
        not in {
            "sam-h0-corrected-reference-metrics-torus-v2-v1",
            "dbf34-h0-corrected-reference-metrics-torus-v2-v1",
        }
        or (payload.get("corrected_reference_metric_gate") or {}).get("passed")
        is not True
    ):
        raise ValueError("Reference metrics payload is not sealed Stage1-R0 evidence")
    metric_contract = payload.get("metric_contract")
    if (
        not isinstance(metric_contract, dict)
        or metric_contract.get("primary_metric") != "periodic-torus-holes-v2"
        or float(metric_contract.get("expected_value_absolute_tolerance_A2", -1.0))
        != 1.0e-6
        or metric_contract.get("physical_perimeter_metric")
        != "physical-flat-torus-perimeter-v1"
    ):
        raise ValueError("R0 metric version/tolerance/perimeter contract is invalid")
    definitions = payload.get("future_A1_gate_definitions")
    if not isinstance(definitions, dict) or set(definitions) != {"primary", "strong"}:
        raise ValueError("Reference metrics lack primary/strong corrected gates")
    for name in ("primary", "strong"):
        gate = definitions[name]
        reference_values = gate.get("reference") if isinstance(gate, dict) else None
        comparison = gate.get("comparison_rule") if isinstance(gate, dict) else None
        if (
            not isinstance(reference_values, dict)
            or reference_values.get("metric_version") != "periodic-torus-holes-v2"
            or not isinstance(comparison, dict)
            or float(comparison.get("area_comparison_absolute_tolerance_A2", -1.0))
            != 1.0e-6
        ):
            raise ValueError(f"Reference metrics {name} gate is malformed")
        for field_name in (
            "molecule_count_N",
            "total_hole_area_A2",
            "maximum_hole_area_A2",
        ):
            if field_name not in reference_values:
                raise ValueError(f"Reference metrics {name} lacks {field_name}")
    if manifest.get("future_A1_gate_definitions") != definitions:
        raise ValueError("R0 manifest and payload corrected-gate definitions disagree")
    if manifest.get("source_bundle_sha256") != (
        payload.get("source_integrity") or {}
    ).get("source_bundle_sha256"):
        raise ValueError("R0 manifest/payload source-bundle seals disagree")

    project_root = _repository_root_for_script(Path(__file__).resolve())
    if project_root is None:
        raise RuntimeError("R0 source validation requires the project repository root")
    project_root = project_root.resolve()
    if set(expected_inputs) != set(_PARENT_PRIMARY_INPUT_NAMES):
        raise ValueError("R0/current input proof requires exactly five primary inputs")
    expected_hashes = {
        name: str(expected_inputs[name]["sha256"])
        for name in _PARENT_PRIMARY_INPUT_NAMES
    }
    parent_hashes = {
        name: str(parent_reference["input_identities"][name]["sha256"])
        for name in _PARENT_PRIMARY_INPUT_NAMES
    }
    if parent_hashes != expected_hashes:
        raise ValueError("R0 proof sees parent/current primary-input disagreement")
    cli_contract = {
        "footprint_boundary_samples": int(
            repair_metric_contract["footprint_boundary_samples"]
        ),
        "projection_radius_scale": float(
            repair_metric_contract["projection_radius_scale"]
        ),
        "area_tie_tolerance_A2": float(
            repair_metric_contract["area_tie_tolerance_A2"]
        ),
        "perimeter_tie_tolerance_A": float(
            repair_metric_contract["perimeter_tie_tolerance_A"]
        ),
    }
    if cli_contract["area_tie_tolerance_A2"] != 1.0e-6:
        raise ValueError("Formal repair requires the sealed R0 area tolerance 1e-6")
    parent_metric_evidence = _parent_requested_metric_contract(parent_reference)
    if parent_metric_evidence["contract"] != cli_contract:
        raise ValueError(
            "Formal repair metric CLI differs from the parent sealed request/top"
        )

    selected_files = selected_source_lock.get("selected_source_files")
    if not isinstance(selected_files, list) or len(selected_files) != 10:
        raise ValueError("Formal R0 repair requires exactly the sealed top-10 sources")
    current_top10 = [
        {
            "selection_rank": int(record["selection_rank"]),
            "sha256": str(record["sha256"]),
        }
        for record in selected_files
    ]
    if [record["selection_rank"] for record in current_top10] != list(range(1, 11)):
        raise ValueError("Current top-10 source lock ranks are not exact 1..10")

    independent = payload.get("independent_current_validation")
    independent_inputs = independent.get("primary_inputs") if isinstance(independent, dict) else None
    if not isinstance(independent_inputs, dict) or {
        name: str((independent_inputs.get(name) or {}).get("sha256"))
        for name in _PARENT_PRIMARY_INPUT_NAMES
    } != expected_hashes:
        raise ValueError("R0 independent-validation primary inputs differ from current")
    conformer_selection = independent.get("conformer_selection") or {}
    if (
        int(conformer_selection.get("max_conformers", -1)) != 10
        or str(conformer_selection.get("glob"))
        != str(selected_source_lock.get("conformer_glob"))
    ):
        raise ValueError("R0 top-10 conformer selection differs from current request")

    source_integrity = payload.get("source_integrity")
    source_records = source_integrity.get("source_records") if isinstance(source_integrity, dict) else None
    if not isinstance(source_records, list):
        raise ValueError("R0 lacks sealed source records")
    records_by_role = defaultdict(list)
    for record in source_records:
        if isinstance(record, dict):
            records_by_role[str(record.get("role"))].append(record)
    required_roles = {"baseline", "perimeter", "compact"}
    if set(records_by_role) != required_roles or any(
        len(records_by_role[role]) != 1 for role in required_roles
    ):
        raise ValueError("R0 source roles must be exactly baseline/perimeter/compact")

    source_proofs = {}
    source_artifact_ledger = {}
    parent_lattice = None
    parent_cell = None
    baseline_record = None
    for role in sorted(required_roles):
        source_record = records_by_role[role][0]
        source_run = source_record.get("source_run")
        if not isinstance(source_run, str) or not source_run:
            raise ValueError(f"R0 {role} source_run is missing")
        source_root = (project_root / source_run).resolve()
        artifacts = source_record.get("artifacts")
        if not isinstance(artifacts, dict):
            raise ValueError(f"R0 {role} source artifacts are malformed")
        artifact_identities = {}
        for artifact_role, artifact_record in sorted(artifacts.items()):
            artifact_path = _resolve_r0_record_path(
                artifact_record,
                package_root=package_root,
                project_root=project_root,
                label=f"{role} {artifact_role}",
            )
            actual = _hash_check_parent_file(
                artifact_path,
                artifact_record,
                label=f"R0 {role} {artifact_role}",
                require_declared_hash=True,
            )
            artifact_identities[artifact_role] = actual
            source_artifact_ledger[f"source.{role}.{artifact_role}"] = actual
        top_identity = artifact_identities.get("top_manifest")
        if top_identity is None or Path(top_identity["path"]) != source_root / "manifest.json":
            raise ValueError(f"R0 {role} top-manifest path disagrees with source_run")
        source_top = json.loads(Path(top_identity["path"]).read_text())
        source_input_checks = source_record.get("top_manifest_input_hash_checks")
        source_top_inputs = source_top.get("inputs")
        if not isinstance(source_input_checks, dict) or not isinstance(source_top_inputs, dict):
            raise ValueError(f"R0 {role} input-hash proof is missing")
        for name in _PARENT_PRIMARY_INPUT_NAMES:
            recorded = source_input_checks.get(name) or {}
            top_record = source_top_inputs.get(name) or {}
            if (
                str(recorded.get("recorded_sha256")) != expected_hashes[name]
                or str(top_record.get("sha256")) != expected_hashes[name]
                or recorded.get("matches_recorded_historical_bytes") is not True
            ):
                raise ValueError(f"R0 {role} input {name} differs from parent/current")
        source_top10 = _top10_hash_contract(source_top, label=f"R0 {role} source top")
        if source_top10 != current_top10:
            raise ValueError(f"R0 {role} top-10 projection sources differ from current")
        source_metric = _schema2_top_projection_metric_contract(
            source_top, label=f"R0 {role} source top"
        )
        if source_metric["contract"] != cli_contract:
            raise ValueError(f"R0 {role} source projection metric contract mismatch")
        cache_contract = source_metric["projection_cache_contract"]
        source_cell = np.asarray(cache_contract.get("cell"), dtype=float)
        source_axes = tuple(int(value) for value in cache_contract.get("periodic_axes", ()))
        source_frame = cache_contract.get("surface_frame")
        lattice, cell_area = _surface_lattice_uv(
            source_cell, source_axes, source_frame
        )
        reference_record = (payload.get("references") or {}).get(role)
        physical = (
            reference_record.get("physical_surface_lattice_derivation")
            if isinstance(reference_record, dict)
            else None
        )
        if (
            not isinstance(reference_record, dict)
            or reference_record.get("role") != role
            or reference_record.get("source_run") != source_run
            or float(reference_record.get("expected_match_absolute_tolerance_A2", -1.0))
            != 1.0e-6
            or reference_record.get("expected_match_passed") is not True
            or not isinstance(physical, dict)
            or not np.allclose(
                np.asarray(physical.get("cell_A"), dtype=float),
                source_cell,
                atol=1.0e-10,
                rtol=0.0,
            )
            or not np.allclose(
                np.asarray(physical.get("surface_lattice_uv_A"), dtype=float),
                lattice,
                atol=1.0e-10,
                rtol=0.0,
            )
            or abs(float(physical.get("cell_area_A2", -1.0)) - cell_area) > 1.0e-6
        ):
            raise ValueError(f"R0 {role} physical cell/lattice proof is invalid")
        if parent_lattice is None:
            parent_lattice = lattice
            parent_cell = source_cell
        elif not np.allclose(lattice, parent_lattice, atol=1.0e-10, rtol=0.0):
            raise ValueError("R0 source-role surface lattices disagree")
        source_proofs[role] = {
            "source_run": source_run,
            "artifacts": artifact_identities,
            "five_primary_input_hashes": expected_hashes,
            "top10_structure_hashes": source_top10,
            "projection_metric_contract": source_metric,
            "physical_surface_lattice_uv_A": lattice.tolist(),
            "physical_cell_area_A2": float(cell_area),
        }
        if role == "baseline":
            baseline_record = source_record

    baseline_artifacts = source_proofs["baseline"]["artifacts"]
    parent_expected = {
        "top_manifest": parent_reference["hash_ledger_before"]["top_manifest"],
        "trajectory_manifest": parent_reference["hash_ledger_before"]["trajectory_manifest"],
        "final_structure": parent_reference["hash_ledger_before"]["final_structure"],
    }
    parent_schema_version = int(parent_reference["schema_version"])
    # The narrow schema-2 adapter intentionally repairs the reviewed R0
    # baseline itself.  A current schema-4 parent is a separate immutable
    # strict-vdW candidate package: bind it to the same five inputs and metric
    # contract, but do not falsely require its trajectory/final bytes to equal
    # the legacy baseline used to define the R0 gates.
    if parent_schema_version == 2:
        for artifact_role, expected in parent_expected.items():
            actual = baseline_artifacts.get(artifact_role)
            if actual is None or {
                key: actual[key] for key in ("sha256", "bytes")
            } != {
                key: expected[key] for key in ("sha256", "bytes")
            }:
                raise ValueError(
                    f"R0 baseline {artifact_role} does not exactly match selected parent"
                )
        if {
            "top": parent_expected["top_manifest"]["sha256"],
            "trajectory": parent_expected["trajectory_manifest"]["sha256"],
            "final": parent_expected["final_structure"]["sha256"],
        } != {
            "top": _SCHEMA2_ADAPTER_TOP_SHA256,
            "trajectory": _SCHEMA2_ADAPTER_TRAJECTORY_SHA256,
            "final": _SCHEMA2_ADAPTER_FINAL_SHA256,
        }:
            raise ValueError("R0/schema-2 parent exact hash triple is not reviewed")

    current_cell = np.asarray(current_substrate.cell, dtype=float)
    if not np.allclose(current_cell, parent_cell, atol=1.0e-10, rtol=0.0):
        raise ValueError("R0 physical cell differs from the current substrate")
    parent_cache_contract = parent_metric_evidence.get("projection_cache_contract")
    if parent_cache_contract is not None:
        if (
            not np.allclose(
                np.asarray(parent_cache_contract.get("cell"), dtype=float),
                parent_cell,
                atol=1.0e-10,
                rtol=0.0,
            )
            or int(parent_cache_contract.get("boundary_samples_per_atom", -1))
            != cli_contract["footprint_boundary_samples"]
            or float(parent_cache_contract.get("radius_scale", float("nan")))
            != cli_contract["projection_radius_scale"]
        ):
            raise ValueError("Parent projection-cache proof differs from R0/current")

    independent_runs = independent.get("runs") if isinstance(independent, dict) else None
    baseline_validation = independent_runs.get("baseline") if isinstance(independent_runs, dict) else None
    reconstruction_checks = (
        baseline_validation.get("reconstruction_checks")
        if isinstance(baseline_validation, dict)
        else None
    )
    if (
        not isinstance(baseline_validation, dict)
        or baseline_validation.get("source_run") != baseline_record.get("source_run")
        or int(baseline_validation.get("step_count", -1)) != 96
        or int(baseline_validation.get("placed_match_count", -1)) != 96
        or baseline_validation.get("ambiguous_matches") != []
        or baseline_validation.get("missing_matches") != []
        or not isinstance(reconstruction_checks, dict)
        or not all(reconstruction_checks.values())
    ):
        raise ValueError("R0 baseline reconstruction is not 96/96 unambiguous")
    summary = independent.get("summary") or {}
    if (
        baseline_validation.get("passed") is not False
        or baseline_validation.get("status")
        != "failed_current_exact_bond_and_strict_final_interface_validation"
        or summary.get("all_source_runs_reconstructed") is not True
        or summary.get("legacy_skip_scientific_validation_failures_preserved")
        is not True
        or summary.get("all_current_exact_bond_and_strict_audits_passed")
        is not False
    ):
        raise ValueError("R0 strict-validation failure debt is not explicitly preserved")
    if manifest.get("independent_validation_summary") != summary:
        raise ValueError("R0 manifest/payload independent-validation summaries disagree")

    proof = {
        "manifest_to_payload_hash_chain": {
            "manifest_sha256": manifest_identity["sha256"],
            "payload_sha256": metrics_identity["sha256"],
            "manifest_reference_matches_payload": True,
            "artifact_inventory_matches_payload": True,
        },
        "metric_contract": {
            "metric_version": "periodic-torus-holes-v2",
            "area_tolerance_A2": 1.0e-6,
            "repair_cli": cli_contract,
            "parent": parent_metric_evidence,
            "all_exactly_equal": True,
        },
        "five_primary_input_hashes": expected_hashes,
        "top10_structure_hashes": current_top10,
        "source_roles": source_proofs,
        "baseline_parent_exact_hash_triple": {
            role: baseline_artifacts[role]["sha256"]
            for role in ("top_manifest", "trajectory_manifest", "final_structure")
        },
        "selected_parent_hash_triple": {
            role: record["sha256"] for role, record in parent_expected.items()
        },
        "selected_parent_matches_r0_baseline": parent_schema_version == 2,
        "baseline_reconstruction": {
            "step_count": 96,
            "placed_match_count": 96,
            "ambiguous_match_count": 0,
            "missing_match_count": 0,
            "unambiguous": True,
        },
        "strict_validation_debt": {
            "status": baseline_validation["status"],
            "passed": False,
            "legacy_skip_scientific_validation_failures_preserved": True,
            "not_waived_by_metric_gate": True,
        },
        "physical_surface_lattice_uv_A": parent_lattice.tolist(),
        "physical_cell_area_A2": abs(float(np.linalg.det(parent_lattice))),
    }
    ledger = {
        "manifest": manifest_identity,
        "reference_metrics": metrics_identity,
        **source_artifact_ledger,
    }
    ledger = {
        label: {
            "path": identity["path"],
            "sha256": identity["sha256"],
            "bytes": int(identity["bytes"]),
        }
        for label, identity in sorted(ledger.items())
    }
    return {
        "schema": "sam-local-repair-reference-metrics-v1",
        "manifest": manifest,
        "payload": payload,
        "manifest_identity": manifest_identity,
        "metrics_identity": metrics_identity,
        "future_A1_gate_definitions": definitions,
        "validation_proof": proof,
        "strict_validation_debt": proof["strict_validation_debt"],
        "hash_ledger_before": ledger,
    }


def verify_reference_metrics_unchanged(reference) -> dict:
    before = reference["hash_ledger_before"]
    after = {}
    for label, expected in sorted(before.items()):
        actual = _file_identity(Path(expected["path"]))
        after[label] = actual
        if {
            key: actual[key] for key in ("path", "sha256", "bytes")
        } != {
            key: expected[key] for key in ("path", "sha256", "bytes")
        }:
            raise RuntimeError("Reference metrics changed during local repair")
    return {"unchanged": True, "before": before, "after": after}


def _parent_step_candidate_id(step: dict) -> str | None:
    evidence = []
    direct = step.get("candidate_id")
    if direct is not None:
        evidence.append(str(direct))
    for key in ("capacity_audit", "policy_audit"):
        audit = step.get(key)
        if isinstance(audit, dict) and audit.get("winner_candidate_id") is not None:
            evidence.append(str(audit["winner_candidate_id"]))
    unique = sorted(set(evidence))
    if len(unique) > 1:
        raise ValueError("Parent step candidate-ID evidence is contradictory")
    return unique[0] if unique else None


def reconstruct_parent_placements(
    parent_reference,
    *,
    domains,
    substrate,
    surface_frame=None,
    position_tolerance_A=1.0e-8,
) -> dict:
    """Rebuild every accepted parent step from the current same-contract domain."""

    trajectory = parent_reference["trajectory_manifest"]
    steps = trajectory.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("Parent trajectory has no accepted sequential steps")
    by_key = defaultdict(list)
    for candidates in domains.values():
        for candidate in candidates:
            key = (
                str(candidate["site_instance_id"]),
                int(candidate["selection_rank"]),
                tuple(int(value) for value in candidate["oxygen_permutation"]),
            )
            by_key[key].append(candidate)
    placed = []
    reconstruction = []
    for expected_step, step in enumerate(steps, 1):
        if int(step.get("step", -1)) != expected_step:
            raise ValueError("Parent sequential step numbering is missing or ambiguous")
        try:
            key = (
                str(step["site_instance_id"]),
                int(step["selection_rank"]),
                tuple(int(value) for value in step["oxygen_permutation"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Parent step lacks site/rank/permutation identity") from exc
        matches = list(by_key.get(key, ()))
        candidate_id = _parent_step_candidate_id(step)
        if candidate_id is not None:
            matches = [
                candidate
                for candidate in matches
                if str(candidate.get("candidate_id")) == candidate_id
            ]
        if len(matches) != 1:
            raise ValueError(
                "Parent step reconstruction is missing or ambiguous for "
                f"site/rank/permutation {key}"
            )
        candidate = matches[0]
        for field_name in ("site_prototype_id", "cluster_id", "source"):
            if field_name in step and str(candidate.get(field_name)) != str(
                step[field_name]
            ):
                raise ValueError(
                    f"Parent step {expected_step} {field_name} disagrees with current domain"
                )
        placed.append(candidate)
        reconstruction.append(
            {
                "step": expected_step,
                "site_instance_id": key[0],
                "selection_rank": key[1],
                "oxygen_permutation": list(key[2]),
                "candidate_id": str(candidate["candidate_id"]),
                "candidate_id_parent_evidence": candidate_id,
                "candidate_id_verified_when_available": candidate_id is not None,
            }
        )
    expected_count = trajectory.get("molecule_count_N")
    if expected_count is None:
        expected_count = trajectory.get("emergent_molecule_count")
    if expected_count is None or int(expected_count) != len(placed):
        raise ValueError("Parent trajectory molecule count disagrees with accepted steps")
    frame = surface_frame or _common_surface_frame(domains)
    as_recorded = _assemble(substrate, placed)
    alignment_candidates = {
        "as_recorded": as_recorded,
        "normal_bottom_aligned": _align_to_cell_bottom(
            as_recorded.copy(), surface_frame=frame
        ),
    }
    readback = read_typed_structure(Path(parent_reference["final_path"]))
    readback_symbols = readback.get_chemical_symbols()
    trials = {}
    matching_modes = []
    for mode, rebuilt_candidate in alignment_candidates.items():
        atom_count_equal = len(rebuilt_candidate) == len(readback)
        checks = {
            "symbols_equal": bool(
                rebuilt_candidate.get_chemical_symbols() == readback_symbols
            ),
            "cell_equal": bool(
                np.allclose(
                    np.asarray(rebuilt_candidate.cell, dtype=float),
                    np.asarray(readback.cell, dtype=float),
                    atol=1.0e-10,
                    rtol=0.0,
                )
            ),
            "positions_equal": bool(
                atom_count_equal
                and np.allclose(
                    np.asarray(rebuilt_candidate.positions, dtype=float),
                    np.asarray(readback.positions, dtype=float),
                    atol=float(position_tolerance_A),
                    rtol=0.0,
                )
            ),
            "atom_count_equal": bool(atom_count_equal),
        }
        matched = all(checks.values())
        trials[mode] = {
            "matched": matched,
            "checks": checks,
            "rebuilt_structure_identity_sha256": _readback_structure_identity(
                rebuilt_candidate
            ),
        }
        if matched:
            matching_modes.append(mode)
    if len(matching_modes) != 1:
        raise ValueError(
            "Parent reconstruction requires exactly one alignment mode match; "
            + json.dumps(
                {"matching_modes": matching_modes, "trials": trials},
                sort_keys=True,
            )
        )
    alignment_mode = matching_modes[0]
    rebuilt = alignment_candidates[alignment_mode]
    checks = trials[alignment_mode]["checks"]
    return {
        "passed": True,
        "placed": placed,
        "step_reconstruction": reconstruction,
        "accepted_step_count": len(reconstruction),
        "unambiguous_step_count": len(reconstruction),
        "ambiguous_step_count": 0,
        "alignment_mode": alignment_mode,
        "alignment_trials": trials,
        "alignment_mode_unique": True,
        "readback_checks": checks,
        "position_tolerance_A": float(position_tolerance_A),
        "readback_structure_identity_sha256": _readback_structure_identity(
            readback
        ),
        "rebuilt_structure_identity_sha256": _readback_structure_identity(
            rebuilt
        ),
    }


def verify_parent_unchanged(parent_reference) -> dict:
    """Re-hash the complete captured parent ledger after repair and fail closed."""

    before = parent_reference.get("hash_ledger_before")
    if not isinstance(before, dict) or not before:
        raise ValueError("Parent reference lacks its before-hash ledger")
    tree_before = parent_reference.get("parent_tree_ledger_before")
    if not isinstance(tree_before, dict) or not tree_before.get("records"):
        raise ValueError("Parent reference lacks its complete before-tree ledger")
    tree_after = _capture_directory_tree_ledger(Path(tree_before["root"]))
    if tree_after != tree_before:
        raise RuntimeError(
            "Immutable parent tree changed during local repair: "
            + json.dumps(
                {
                    "before_tree_sha256": tree_before.get("tree_sha256"),
                    "after_tree_sha256": tree_after.get("tree_sha256"),
                    "before_path_count": tree_before.get("path_count"),
                    "after_path_count": tree_after.get("path_count"),
                },
                sort_keys=True,
            )
        )
    after = {}
    changes = []
    for label, expected in sorted(before.items()):
        path = Path(expected["path"])
        if path.is_symlink() or not path.is_file():
            changes.append({"label": label, "reason": "missing_or_aliased"})
            continue
        actual = _file_identity(path)
        normalized = {
            "path": actual["path"],
            "sha256": actual["sha256"],
            "bytes": int(actual["bytes"]),
        }
        after[label] = normalized
        if normalized != expected:
            changes.append(
                {"label": label, "before": expected, "after": normalized}
            )
    if changes:
        raise RuntimeError(
            "Immutable parent changed during local repair: "
            + json.dumps(changes, sort_keys=True)
        )
    git_evidence_before = parent_reference.get("schema2_git_evidence")
    git_evidence_after = None
    if git_evidence_before is not None:
        git_evidence_after = _schema2_git_implementation_evidence()
        if git_evidence_after != git_evidence_before:
            raise RuntimeError(
                "Historical schema-2 Git object/commit evidence changed during repair"
            )
    return {
        "unchanged": True,
        "before": before,
        "after": after,
        "checked_file_count": len(before),
        "tree_ledger": {
            "before": tree_before,
            "after": tree_after,
            "unchanged": True,
        },
        "schema2_git_evidence": {
            "before": git_evidence_before,
            "after": git_evidence_after,
            "unchanged": (
                None
                if git_evidence_before is None
                else git_evidence_after == git_evidence_before
            ),
        },
        "changes": [],
    }


def _payload_sha256(payload: object) -> str:
    return hashlib.sha256(_strict_json_bytes(payload)).hexdigest()


_RUN_IDENTITY_DEPENDENCY_SCRIPTS = (
    "sam_structure_tools.py",
    "validate_monolayer.py",
    "monolayer_headgroup_fit.py",
    "monolayer_builder.py",
)


def _repository_root_for_script(implementation: Path) -> Path | None:
    for parent in implementation.parents:
        canonical = parent / "scripts" / implementation.name
        mirror = (
            parent
            / ".agents"
            / "skills"
            / "sam-conformation-search"
            / "scripts"
            / implementation.name
        )
        if canonical.is_file() and mirror.is_file():
            return parent
    return None


def _script_identity(
    runtime_path: Path, *, repository_root: Path | None
) -> dict:
    runtime_path = Path(runtime_path).resolve()
    result = {"implementation": _file_identity(runtime_path)}
    if repository_root is None:
        result.update(
            {
                "canonical": result["implementation"],
                "mirror": None,
                "canonical_mirror_bytes_equal": None,
            }
        )
        return result
    canonical = repository_root / "scripts" / runtime_path.name
    mirror = (
        repository_root
        / ".agents"
        / "skills"
        / "sam-conformation-search"
        / "scripts"
        / runtime_path.name
    )
    if not canonical.is_file() or not mirror.is_file():
        raise FileNotFoundError(
            f"Canonical/mirror implementation pair is incomplete for {runtime_path.name}"
        )
    result.update(
        {
            "canonical": _file_identity(canonical),
            "mirror": _file_identity(mirror),
        }
    )
    result["canonical_mirror_bytes_equal"] = (
        result["canonical"]["sha256"] == result["mirror"]["sha256"]
    )
    return result


def _implementation_identity(path: Path | None = None) -> dict:
    implementation = Path(path or __file__).resolve()
    repository_root = _repository_root_for_script(implementation)
    result = _script_identity(
        implementation, repository_root=repository_root
    )
    dependencies = {}
    for filename in _RUN_IDENTITY_DEPENDENCY_SCRIPTS:
        runtime_path = implementation.parent / filename
        if not runtime_path.is_file():
            raise FileNotFoundError(
                f"Sequential-growth runtime dependency is missing: {runtime_path}"
            )
        dependencies[filename] = _script_identity(
            runtime_path, repository_root=repository_root
        )
    result["dependencies"] = dependencies
    return result


def _safe_capture_error(error: BaseException) -> dict:
    try:
        message = str(error)
    except BaseException:
        message = "<error message unavailable>"
    return {"type": type(error).__name__, "message": message}


def _capture_implementation_identity_best_effort() -> dict:
    """Capture identity before reservation without making sealing depend on it."""

    try:
        identity = _implementation_identity()
        failure_document_identity = identity["implementation"]
        if not isinstance(failure_document_identity, dict):
            raise TypeError("implementation identity record must be a dictionary")
        _strict_json_bytes(identity)
    except BaseException as error:
        return {
            "status": "unavailable",
            "identity": None,
            "failure_document_identity": {
                "status": "unavailable",
                "capture_error": _safe_capture_error(error),
            },
        }
    return {
        "status": "captured",
        "identity": identity,
        "failure_document_identity": failure_document_identity,
    }


def _code_snapshot_plan(implementation: dict) -> dict:
    runtime_sources = {
        Path(implementation["implementation"]["path"]).name: implementation[
            "implementation"
        ],
        **{
            filename: identity["implementation"]
            for filename, identity in implementation["dependencies"].items()
        },
    }
    files = []
    for filename, source in sorted(runtime_sources.items()):
        files.append(
            {
                "module": filename,
                "path": f"code-snapshot/{filename}",
                "source": dict(source),
                "sha256": source["sha256"],
                "bytes": int(source["bytes"]),
            }
        )
    return {
        "path": "code-snapshot",
        "purpose": "exact_runtime_bytes_for_uncommitted_immutable_run_provenance",
        "installation": "same_directory_hard_link_noreplace_after_file_fsync",
        "overwrite": "forbidden",
        "files": files,
    }


def _write_code_snapshot(output_dir: Path, plan: dict) -> dict:
    output_dir = Path(output_dir)
    snapshot_dir = output_dir / plan["path"]
    snapshot_dir.mkdir(exist_ok=False)
    _fsync_directory(output_dir)
    written_files = []
    for planned in plan["files"]:
        source_path = Path(planned["source"]["path"])
        payload = source_path.read_bytes()
        actual_sha256 = hashlib.sha256(payload).hexdigest()
        if (
            actual_sha256 != planned["sha256"]
            or len(payload) != int(planned["bytes"])
        ):
            raise RuntimeError(
                f"Runtime code changed after run-request sealing: {source_path}"
            )
        destination = output_dir / planned["path"]
        written = _atomic_install_noclobber_bytes(destination, payload)
        written_files.append(
            {
                "module": planned["module"],
                "path": destination.relative_to(output_dir).as_posix(),
                "sha256": hashlib.sha256(written).hexdigest(),
                "bytes": len(written),
            }
        )
    return {
        "path": snapshot_dir.relative_to(output_dir).as_posix(),
        "path_base": "run_root",
        "reuse": "new_output_internal_no_prior_reuse",
        "overwrite": "forbidden",
        "files": written_files,
    }


def _verify_regular_artifact_reference(
    path: Path, reference: dict, *, label: str
) -> dict:
    """Re-hash one immutable regular file and reject symlink substitution."""

    path = Path(path)
    if path.is_symlink() or not os.path.lexists(os.fspath(path)):
        raise RuntimeError(f"{label} is missing or is a symlink before top seal: {path}")
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"{label} is not a regular file before top seal: {path}")
    actual = {"sha256": _sha256(path), "bytes": metadata.st_size}
    expected = {
        "sha256": str(reference["sha256"]),
        "bytes": int(reference["bytes"]),
    }
    if actual != expected:
        raise RuntimeError(
            f"{label} changed before top manifest seal: {path}; "
            f"expected {expected}, found {actual}"
        )
    return {
        "path": str(path),
        **actual,
        "regular_file": True,
        "symlink": False,
        "hash_matches_initial_reference": True,
    }


def _verify_code_snapshot(output_dir: Path, snapshot: dict) -> dict:
    """Re-hash every installed code snapshot immediately before top sealing."""

    output_dir = Path(output_dir)
    snapshot_dir = output_dir / snapshot["path"]
    if (
        not os.path.lexists(os.fspath(snapshot_dir))
        or snapshot_dir.is_symlink()
        or not stat.S_ISDIR(snapshot_dir.lstat().st_mode)
    ):
        raise RuntimeError(
            f"Code snapshot directory is missing or aliased before top seal: {snapshot_dir}"
        )
    files = []
    for reference in snapshot["files"]:
        verified = _verify_regular_artifact_reference(
            output_dir / reference["path"],
            reference,
            label=f"code snapshot {reference['module']}",
        )
        files.append(
            {
                "module": reference["module"],
                "path": reference["path"],
                "sha256": verified["sha256"],
                "bytes": verified["bytes"],
                "hash_matches_initial_reference": True,
            }
        )
    return {
        "checkpoint": "immediately_before_top_manifest_seal",
        "all_hashes_match_initial_references": True,
        "files": files,
    }


def _jsonable_cli_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable_cli_value(item) for item in value]
    if isinstance(value, list):
        return [_jsonable_cli_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _jsonable_cli_value(item) for key, item in value.items()
        }
    return value


def _selected_conformer_evidence(
    records: list[dict], *, root: Path, analysis_path: Path
) -> list[dict]:
    evidence = []
    for record in records:
        structure = record.get("structure")
        if not isinstance(structure, dict):
            raise ValueError("Selected conformer record lacks a structure object")
        declared_path = structure.get("path")
        declared_sha256 = structure.get("sha256")
        if not isinstance(declared_path, str) or not declared_path:
            raise ValueError("Selected conformer structure path must be non-empty")
        if (
            not isinstance(declared_sha256, str)
            or len(declared_sha256) != 64
            or any(character not in "0123456789abcdef" for character in declared_sha256)
        ):
            raise ValueError("Selected conformer structure SHA-256 is invalid")
        raw_path = Path(declared_path).expanduser()
        candidates = (
            [raw_path]
            if raw_path.is_absolute()
            else [root / raw_path, analysis_path.parent / raw_path]
        )
        resolved = next((path.resolve() for path in candidates if path.is_file()), None)
        if resolved is None:
            raise FileNotFoundError(
                f"Selected conformer structure does not exist: {declared_path}"
            )
        actual = _file_identity(resolved)
        if actual["sha256"] != declared_sha256:
            raise ValueError(
                "Selected conformer structure SHA-256 does not match analysis record: "
                f"{resolved}"
            )
        copied_record = json.loads(json.dumps(record, allow_nan=False))
        evidence.append(
            {
                "selection_rank": int(record["selection_rank"]),
                "cluster_id": int(record["cluster_id"]),
                "analysis_record": copied_record,
                "structure": {
                    **actual,
                    "declared_path": declared_path,
                    "declared_sha256": declared_sha256,
                    "hash_verified": True,
                },
            }
        )
    _strict_json_bytes(evidence)
    return evidence


def _lock_selected_conformer_paths(
    *,
    root: Path,
    conformer_glob: str,
    maximum: int,
    selected_evidence: list[dict],
) -> dict:
    """Bind the actual natural-order glob prefix to analysis-selected evidence."""

    matched, selected = selected_conformer_paths(
        root, conformer_glob, maximum
    )
    matched_resolved = []
    for path in matched:
        if not path.is_file():
            raise ValueError(f"Conformer glob matched a non-file path: {path}")
        matched_resolved.append(path.resolve())
    selected_resolved = [path.resolve() for path in selected]
    evidence_resolved = [
        Path(record["structure"]["path"]).resolve()
        for record in selected_evidence
    ]
    if selected_resolved != evidence_resolved:
        raise ValueError(
            "Ordered --conformer-glob/--max-conformers selection does not exactly "
            "match the analysis selected evidence"
        )
    if len(selected_resolved) != maximum:
        raise ValueError(
            f"Conformer glob selected {len(selected_resolved)} files; expected {maximum}"
        )
    if len(set(selected_resolved)) != len(selected_resolved):
        raise ValueError("Selected conformer paths must resolve uniquely")
    selected_records = []
    for rank, (path, evidence) in enumerate(
        zip(selected_resolved, selected_evidence), 1
    ):
        identity = _file_identity(path)
        expected_sha256 = evidence["structure"]["sha256"]
        if identity["sha256"] != expected_sha256:
            raise ValueError(
                f"Selected conformer source SHA-256 changed before request: {path}"
            )
        selected_records.append(
            {
                "selection_rank": rank,
                "path": identity["path"],
                "sha256": identity["sha256"],
                "bytes": identity["bytes"],
                "analysis_declared_path": evidence["structure"]["declared_path"],
                "analysis_declared_sha256": evidence["structure"][
                    "declared_sha256"
                ],
            }
        )
    lock = {
        "ordering": "monolayer_headgroup_fit.selected_conformer_paths_natural_filename_rank",
        "conformer_glob": str(conformer_glob),
        "maximum_source_clusters": int(maximum),
        "matched_source_file_count": len(matched_resolved),
        "matched_source_files": [str(path) for path in matched_resolved],
        "selected_source_files": selected_records,
        "ordered_exact_match_to_analysis_evidence": True,
        "identity_basis": "resolved_path_plus_sha256_bytes_not_size_mtime",
    }
    _strict_json_bytes(lock)
    return lock


def _verify_locked_conformer_sources(lock: dict, *, checkpoint: str) -> list[dict]:
    """Re-hash every locked source; size/mtime are never accepted as identity."""

    verified = []
    for expected in lock["selected_source_files"]:
        path = Path(expected["path"])
        if not path.is_file():
            raise FileNotFoundError(
                f"Locked conformer source is missing at {checkpoint}: {path}"
            )
        actual = _file_identity(path)
        if (
            actual["sha256"] != expected["sha256"]
            or int(actual["bytes"]) != int(expected["bytes"])
        ):
            raise ValueError(
                f"Locked conformer source bytes changed at {checkpoint}: {path}"
            )
        verified.append(
            {
                "selection_rank": int(expected["selection_rank"]),
                "path": actual["path"],
                "sha256": actual["sha256"],
                "bytes": actual["bytes"],
                "checkpoint": checkpoint,
                "hash_verified": True,
            }
        )
    return verified


def _coordinate_array_sha256(coordinates) -> str:
    digest = hashlib.sha256()
    digest.update(b"conformer-coordinate-array-v1")
    _hash_update_array(digest, np.asarray(coordinates, dtype=float))
    return digest.hexdigest()


def _symbol_sequence_sha256(symbols) -> str:
    return _payload_sha256(
        {
            "schema": "conformer-symbol-sequence-v1",
            "symbols": [str(symbol) for symbol in np.asarray(symbols).tolist()],
        }
    )


def _direct_source_conformer_geometry(path: Path, spec: MoleculeSpec) -> dict:
    atoms = read_typed_structure(path)
    components = molecular_components(atoms, spec)
    if len(components) != 1:
        raise ValueError(
            f"Each selected conformer source must contain exactly one molecule; "
            f"found {len(components)} in {path}"
        )
    coordinates, symbols = unwrap_component(
        atoms, components[0], spec.hydrogen_parent_elements
    )
    coordinates = np.asarray(coordinates, dtype=float)
    symbols = np.asarray(symbols, dtype=str)
    if not np.all(np.isfinite(coordinates)):
        raise ValueError(f"Selected conformer has non-finite coordinates: {path}")
    return {
        "coordinates": coordinates,
        "symbols": symbols,
        "coordinate_sha256": _coordinate_array_sha256(coordinates),
        "symbol_sequence_sha256": _symbol_sequence_sha256(symbols),
        "geometry_sha256": _conformer_geometry_sha256(coordinates, symbols),
    }


def _audit_loaded_conformer_ensemble(
    conformers: list[dict],
    *,
    locked_selection: dict,
    spec: MoleculeSpec,
    root: Path,
) -> list[dict]:
    """Prove cache-loaded rank/source/symbol/coordinate identity before domain work."""

    expected_records = locked_selection["selected_source_files"]
    if len(conformers) != len(expected_records):
        raise ValueError(
            "Loaded conformer ensemble must contain exactly one conformer per "
            f"selected rank; loaded {len(conformers)}, expected {len(expected_records)}"
        )
    by_rank = {}
    for conformer in conformers:
        rank = int(conformer["source_rank"])
        if rank in by_rank:
            raise ValueError(f"Loaded conformer rank {rank} is not unique")
        by_rank[rank] = conformer
    expected_ranks = list(range(1, len(expected_records) + 1))
    if sorted(by_rank) != expected_ranks:
        raise ValueError(
            f"Loaded conformer ranks {sorted(by_rank)} do not equal {expected_ranks}"
        )

    audit = []
    for expected in expected_records:
        rank = int(expected["selection_rank"])
        conformer = by_rank[rank]
        raw_source = Path(str(conformer.get("source", ""))).expanduser()
        loaded_source = (
            raw_source.resolve()
            if raw_source.is_absolute()
            else (root / raw_source).resolve()
        )
        expected_source = Path(expected["path"]).resolve()
        if loaded_source != expected_source:
            raise ValueError(
                f"Loaded conformer rank {rank} source path does not match the locked "
                f"source: {loaded_source} != {expected_source}"
            )
        direct = _direct_source_conformer_geometry(expected_source, spec)
        loaded_coordinates = np.asarray(conformer["coordinates"], dtype=float)
        loaded_symbols = np.asarray(conformer["symbols"], dtype=str)
        if loaded_coordinates.ndim != 2 or loaded_coordinates.shape[1:] != (3,):
            raise ValueError(f"Loaded conformer rank {rank} coordinates are malformed")
        if loaded_symbols.ndim != 1 or len(loaded_symbols) != len(loaded_coordinates):
            raise ValueError(f"Loaded conformer rank {rank} symbols are malformed")
        if not np.all(np.isfinite(loaded_coordinates)):
            raise ValueError(f"Loaded conformer rank {rank} coordinates are non-finite")
        loaded_hashes = {
            "coordinate_sha256": _coordinate_array_sha256(loaded_coordinates),
            "symbol_sequence_sha256": _symbol_sequence_sha256(loaded_symbols),
            "geometry_sha256": _conformer_geometry_sha256(
                loaded_coordinates, loaded_symbols
            ),
        }
        expected_hashes = {
            key: direct[key]
            for key in (
                "coordinate_sha256",
                "symbol_sequence_sha256",
                "geometry_sha256",
            )
        }
        if loaded_hashes != expected_hashes:
            raise ValueError(
                f"Loaded conformer rank {rank} symbol/coordinate geometry hashes "
                "do not match direct source parsing; refusing stale cache"
            )
        audit.append(
            {
                "selection_rank": rank,
                "source_resolved_path": str(expected_source),
                "source_sha256": expected["sha256"],
                "source_bytes": int(expected["bytes"]),
                "loaded_source_path_matches": True,
                "unique_rank": True,
                "loaded": loaded_hashes,
                "direct_source": expected_hashes,
                "geometry_hashes_match": True,
            }
        )
    _strict_json_bytes(audit)
    return audit


def _cpu_environment() -> dict:
    return {
        "execution_device_scope": "CPU_only",
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "logical_cpu_count": os.cpu_count(),
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "package_versions": _package_versions(),
    }


def _validate_run_request_document(document: dict) -> None:
    if not isinstance(document, dict):
        raise ValueError("Run request must be a JSON object")
    if document.get("schema") not in {
        "sam-sequential-growth-run-request-v1",
        "sam-local-repair-run-request-v1",
    }:
        raise ValueError("Run request schema mismatch")
    if set(document.get("primary_inputs", {})) != {
        "substrate",
        "layer_groups",
        "site_instances",
        "site_prototypes",
        "conformer_analysis",
    }:
        raise ValueError("Run request must identify exactly five primary inputs")
    axes = document.get("resolved_periodic_fractional_axes")
    _validated_surface_axes(axes)
    identity = document.get("run_identity", {})
    sha256 = identity.get("sha256")
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
    ):
        raise ValueError("Run request identity hash is invalid")
    identity_material = document.get("identity_material")
    if not isinstance(identity_material, dict) or _payload_sha256(
        identity_material
    ) != sha256:
        raise ValueError("Run request identity material does not match its SHA-256")
    dependencies = document.get("implementation", {}).get("dependencies", {})
    if set(dependencies) != set(_RUN_IDENTITY_DEPENDENCY_SCRIPTS):
        raise ValueError("Run request dependency implementation set is incomplete")
    snapshot = document.get("code_snapshot", {})
    if snapshot.get("path") != "code-snapshot" or snapshot.get("overwrite") != "forbidden":
        raise ValueError("Run request code snapshot contract is incomplete")
    _strict_json_bytes(document)


def _is_strict_descendant(path: Path, parent: Path) -> bool:
    path = Path(path).resolve()
    parent = Path(parent).resolve()
    try:
        relative = path.relative_to(parent)
    except ValueError:
        return False
    return relative != Path(".")


def _paths_overlap(left: Path, right: Path) -> bool:
    """Return whether either lexical namespace contains the other."""

    left = Path(left)
    right = Path(right)
    if left == right:
        return True
    return left in right.parents or right in left.parents


def _cache_reuse_contract(
    *,
    args,
    output_dir: Path,
    resolved_paths: dict[str, Path],
    lexical_paths: dict[str, Path] | None = None,
) -> dict:
    formal = args.independent_validation != "off"
    output_dir = Path(output_dir).resolve()
    if output_dir.is_symlink() or not stat.S_ISDIR(output_dir.lstat().st_mode):
        raise ValueError("Sequential-growth output reservation must be a real directory")

    internal_cache_root = output_dir / "internal-cache"
    expected_conformer_prefix = internal_cache_root / "conformers"
    expected_projection_directory = internal_cache_root / "projection"
    expected = {
        "conformer_cache": expected_conformer_prefix,
        "projection_cache": expected_projection_directory,
    }
    initial_resolved = {
        "conformer_cache": Path(resolved_paths["conformer_cache"]),
        "projection_cache": Path(resolved_paths["projection_cache"]),
    }
    lexical = {
        name: Path((lexical_paths or {}).get(name, initial_resolved[name]))
        for name in initial_resolved
    }

    startup_paths = [
        internal_cache_root,
        expected_conformer_prefix,
        Path(f"{expected_conformer_prefix}.json"),
        Path(f"{expected_conformer_prefix}.npz"),
        expected_projection_directory,
        expected_projection_directory / "projection-cache-data.npz",
        expected_projection_directory / "projection-cache-manifest.json",
    ]
    if formal:
        existing = [
            str(path)
            for path in startup_paths
            if os.path.lexists(os.fspath(path))
        ]
        if existing:
            raise FileExistsError(
                "Formal internal cache paths and parents must not exist at run startup: "
                + ", ".join(existing)
            )

    raw_namespaces = {
        "conformer cache prefix": lexical["conformer_cache"],
        "conformer cache JSON": Path(f"{lexical['conformer_cache']}.json"),
        "conformer cache NPZ": Path(f"{lexical['conformer_cache']}.npz"),
        "projection cache directory": lexical["projection_cache"],
        "projection cache data": (
            lexical["projection_cache"] / "projection-cache-data.npz"
        ),
        "projection cache manifest": (
            lexical["projection_cache"] / "projection-cache-manifest.json"
        ),
        "projection cache failure record": (
            lexical["projection_cache"] / "failure.json"
        ),
    }
    lexical_namespaces = {
        label: Path(os.path.abspath(os.fspath(path)))
        for label, path in raw_namespaces.items()
    }
    try:
        resolved_namespaces = {
            label: path.resolve() for label, path in raw_namespaces.items()
        }
    except (OSError, RuntimeError) as exc:
        raise ValueError(
            "Cache namespace contains an unresolvable symlink alias"
        ) from exc

    current_resolved = {
        "conformer_cache": resolved_namespaces["conformer cache prefix"],
        "projection_cache": resolved_namespaces["projection cache directory"],
    }
    for name in initial_resolved:
        if current_resolved[name] != initial_resolved[name]:
            raise ValueError(
                f"{name} resolution changed before cache namespace validation: "
                f"initial={initial_resolved[name]}, current={current_resolved[name]}"
            )

    reserved_roots = {
        output_dir / "run-request.json",
        output_dir / "manifest.json",
        output_dir / "failure.json",
        output_dir / "code-snapshot",
    }
    for namespaces in (lexical_namespaces, resolved_namespaces):
        for label, path in namespaces.items():
            if any(_paths_overlap(path, reserved) for reserved in reserved_roots):
                raise ValueError(
                    f"Sequential-growth {label} aliases a reserved run artifact "
                    f"namespace: {path}"
                )
            try:
                relative = path.relative_to(output_dir)
            except ValueError:
                relative = None
            if (
                relative is not None
                and relative.parts
                and relative.parts[0].startswith("trajectory-")
            ):
                raise ValueError(
                    f"Sequential-growth {label} aliases a reserved trajectory "
                    f"namespace: {path}"
                )

    conformer_labels = (
        "conformer cache prefix",
        "conformer cache JSON",
        "conformer cache NPZ",
    )
    projection_labels = (
        "projection cache directory",
        "projection cache data",
        "projection cache manifest",
        "projection cache failure record",
    )
    for namespaces in (lexical_namespaces, resolved_namespaces):
        if any(
            _paths_overlap(namespaces[conformer_label], namespaces[projection_label])
            for conformer_label in conformer_labels
            for projection_label in projection_labels
        ):
            raise ValueError(
                "Sequential-growth conformer and projection cache namespaces "
                "must not overlap"
            )

    for label, path in resolved_namespaces.items():
        if path == output_dir or output_dir in path.parents:
            if path != internal_cache_root and internal_cache_root not in path.parents:
                raise ValueError(
                    f"Sequential-growth {label} resolves inside the output but outside "
                    f"its safe internal-cache namespace: {path}"
                )
    for label in raw_namespaces:
        lexical_path = lexical_namespaces[label]
        resolved_path = resolved_namespaces[label]
        lexical_is_internal = (
            lexical_path == output_dir or output_dir in lexical_path.parents
        )
        resolved_is_internal = (
            resolved_path == output_dir or output_dir in resolved_path.parents
        )
        if (lexical_is_internal or resolved_is_internal) and (
            raw_namespaces[label] != resolved_path
        ):
            raise ValueError(
                f"Output-internal {label} must use an exact non-alias path without "
                f"'..' or symlinks: lexical={raw_namespaces[label]}, "
                f"resolved={resolved_path}"
            )

    if not formal:
        return {
            "formal": False,
            "mode": "backward_compatible_contract_checked_cache_semantics",
            "reuse": "prior_cache_permitted_under_existing_cache_contracts",
            "startup_absence_required": False,
            "external_shared_cache_reuse_permitted": True,
            "output_internal_namespace": "internal-cache",
            "output_internal_aliases_rejected": True,
            "reserved_namespaces_rejected": True,
            "cache_namespaces_nonoverlapping": True,
        }

    for name, required_path in expected.items():
        if current_resolved[name] != required_path:
            raise ValueError(
                f"Formal {name} must exactly equal {required_path}; "
                f"received {current_resolved[name]}"
            )
        if lexical[name] != required_path:
            raise ValueError(
                f"Formal {name} must use the exact non-alias path {required_path}; "
                f"received lexical path {lexical[name]}"
            )

    return {
        "formal": True,
        "mode": "new_output_internal_no_prior_reuse",
        "reuse": "new_output_internal_no_prior_reuse",
        "startup_absence_required": True,
        "startup_absence_verified": True,
        "exact_namespace_verified": True,
        "lexical_aliases_rejected": True,
        "symlinks_rejected": True,
        "reserved_namespaces_rejected": True,
        "cache_namespaces_nonoverlapping": True,
        "expected_namespace": {
            "conformer_cache_prefix": "internal-cache/conformers",
            "projection_cache_directory": "internal-cache/projection",
        },
        "checked_paths": [str(path) for path in startup_paths],
        "write_contract": {
            "conformer": "same_directory_temp_fsync_hard_link_noreplace_directory_fsync",
            "projection": "same_directory_temp_fsync_hard_link_noreplace_directory_fsync",
            "destination_clobber": False,
        },
        "failure_retention": "cache_artifacts_remain_inside_failed_immutable_run",
    }


def _normalized_identity_arguments(
    *,
    args,
    root: Path,
    resolved_paths: dict[str, Path],
) -> dict:
    raw = {
        key: _jsonable_cli_value(value)
        for key, value in sorted(vars(args).items())
    }
    values = dict(raw)
    values["root"] = str(Path(root).resolve())
    for name in (
        "substrate",
        "layer_groups",
        "site_instances",
        "site_prototypes",
        "conformer_analysis",
    ):
        values[name] = str(resolved_paths[name])
    for name in (
        "p_surface_o_contract",
        "p_surface_o_source_report",
        "p_surface_o_source_config",
    ):
        if name in resolved_paths:
            values[name] = str(resolved_paths[name])
    values["molecule_formula_json"] = _json_formula(
        args.molecule_formula_json
    )
    values["skeleton_formula_json"] = _json_formula(
        args.skeleton_formula_json
    )
    excluded = {
        "output_dir": "excluded_pure_output_absolute_path",
        "conformer_cache": "excluded_cache_absolute_path_reuse_mode_hashed_separately",
        "projection_cache": "excluded_cache_absolute_path_reuse_mode_hashed_separately",
    }
    for name, marker in excluded.items():
        values[name] = {"run_identity_path_treatment": marker}
    if set(values) != set(raw):
        raise RuntimeError("Normalized run-identity arguments are incomplete")
    return {
        "schema": "sam-sequential-normalized-parsed-arguments-v1",
        "values": values,
        "complete_parser_namespace": True,
        "excluded_absolute_path_fields": sorted(excluded),
        "exclusion_rationale": excluded,
        "scientific_values_excluded": False,
    }


def _build_run_request_document(
    *,
    argv: list[str],
    args,
    root: Path,
    output_dir: Path,
    resolved_paths: dict[str, Path],
    implementation: dict,
    primary_inputs: dict,
    selected_conformers: list[dict],
    selected_source_lock: dict,
    sampling_table: list[dict],
    periodic_axes,
    normal_axis: int,
    collision_thresholds_A: dict,
    mapped_bond_window_A,
    substrate_collision_contract: dict,
    molecule_spec: MoleculeSpec,
    cache_reuse_contract: dict,
    code_snapshot: dict,
    operation_evidence: dict | None = None,
    p_surface_o_exclusion_contract: dict | None = None,
    metal_coordination_contract: dict | None = None,
) -> dict:
    parsed_arguments = {
        key: _jsonable_cli_value(value)
        for key, value in sorted(vars(args).items())
    }
    normalized_arguments = _normalized_identity_arguments(
        args=args, root=root, resolved_paths=resolved_paths
    )
    operation = str(getattr(args, "operation", "grow"))
    execution_contract = {
        "scope": (
            "CPU_only_H0_irreversible_sequential_geometry_growth"
            if operation == "grow"
            else "CPU_only_H0_postprocessed_local_repair"
        ),
        "chemistry_state": "H0_surface_protons_deferred",
        "independent_validation_mode": args.independent_validation,
        "forbidden_stages": [
            "calculator_initialization",
            "energy_evaluation",
            "surface_proton_placement",
            "force_relaxation",
            "dynamics",
            "production_simulation",
        ],
        "claims_excluded": [
            "energetic_ranking_of_the_assembled_interface",
            "protonated_interface_readiness",
            "dynamical_trajectory_evidence",
        ],
    }
    protocol = {
        "operation": operation,
        "seed": int(args.seed),
        "trajectory_count": int(args.trajectory_count),
        "max_conformers": int(args.max_conformers),
        "molecule_contract": {
            "formula": dict(molecule_spec.formula),
            "skeleton_formula_argument": (
                None
                if molecule_spec.skeleton_formula is None
                else dict(molecule_spec.skeleton_formula)
            ),
            "resolved_skeleton_formula": molecule_spec.resolved_skeleton_formula(),
            "anchor_element": molecule_spec.anchor_element,
            "headgroup_element": molecule_spec.headgroup_element,
            "headgroup_count": int(molecule_spec.headgroup_count),
        },
        "conformer_selection_contract": selected_source_lock,
        "sampling_contract": {
            "formula": (
                "rank^(-rank_exponent)*exp(-relative_energy/energy_scale_eV)*"
                "exp(-area_coefficient*normalized_footprint_area)"
            ),
            "energy_scale_eV": float(args.energy_scale_eV),
            "area_coefficient": float(args.area_coefficient),
            "rank_exponent": float(args.rank_exponent),
            "auditable_weight_table": sampling_table,
        },
        "headgroup_fit_contract": {
            "rmsd_max_A": float(args.headgroup_rmsd_max_A),
            "conformer_glob": str(args.conformer_glob),
        },
        "footprint_contract": {
            "boundary_samples_per_atom": int(args.footprint_boundary_samples),
            "radius_scale": float(args.projection_radius_scale),
            "radii_source": "ASE ase.data.vdw_radii",
        },
        "sam_sam_collision_thresholds_A": collision_thresholds_A,
        "substrate_collision_contract_parameters": {
            "mode": args.substrate_collision,
            "vdw_radius_scale": float(args.substrate_vdw_radius_scale),
            "mapped_bond_distance_window_A": list(mapped_bond_window_A),
            "resolved_contract": substrate_collision_contract,
        },
        "selection": {
            "policy": args.selection_policy,
            "criterion": args.selection_criterion,
            "capacity_beam_width": int(args.capacity_beam_width),
            "area_tie_tolerance_A2": float(args.area_tie_tolerance_A2),
            "perimeter_tie_tolerance_A": float(args.perimeter_tie_tolerance_A),
            **(
                {
                    "cohesive_perimeter_relative_tolerance": float(
                        resolve_cohesive_cli_tolerances(args)["perimeter"]
                    ),
                    "cohesive_contact_relative_tolerance": float(
                        resolve_cohesive_cli_tolerances(args)["contact"]
                    ),
                    "cohesive_area_relative_tolerance": float(
                        resolve_cohesive_cli_tolerances(args)["area"]
                    ),
                    "cohesive_relative_tolerances": {
                        "perimeter": float(
                            resolve_cohesive_cli_tolerances(args)["perimeter"]
                        ),
                        "contact": float(
                            resolve_cohesive_cli_tolerances(args)["contact"]
                        ),
                        "area": float(
                            resolve_cohesive_cli_tolerances(args)["area"]
                        ),
                    },
                    "cohesive_relative_tolerance": (
                        float(args.cohesive_relative_tolerance)
                        if getattr(args, "cohesive_relative_tolerance", None) is not None
                        else None
                    ),
                    "cohesive_frontier_mode": str(getattr(args, "cohesive_frontier_mode", "discrete")),
                    "frontier_snap_radius_A": float(getattr(args, "frontier_snap_radius_A", 4.0)),
                }
                if args.selection_policy == "cohesive-frontier"
                else {}
            ),
        },
        "cache_modes": {
            "collision": args.collision_cache,
            "conformer_and_projection_reuse": cache_reuse_contract["reuse"],
            "formal_internal_cache": bool(cache_reuse_contract["formal"]),
        },
        "resolved_periodic_fractional_axes": list(periodic_axes),
        "normal_axis": int(normal_axis),
    }
    if p_surface_o_exclusion_contract is not None:
        protocol["p_surface_o_exclusion_experiment"] = p_surface_o_exclusion_contract
    if metal_coordination_contract is not None:
        protocol["metal_coordination_filter"] = metal_coordination_contract
    if operation == "repair":
        protocol["repair"] = {
            "parent_run": str(args.parent_run),
            "reference_metrics": str(args.reference_metrics),
            "max_k": int(args.repair_max_k),
            "neighborhood_radius_A": float(args.repair_neighborhood_radius_A),
            "max_frontier": int(args.repair_max_frontier),
            "max_local_candidates": int(args.repair_max_local_candidates),
            "max_solutions": int(args.repair_max_solutions),
            "time_limit_s": float(args.repair_time_limit_s),
            "max_fixed_replacements": int(
                args.repair_max_fixed_replacements
            ),
            "output_contract": repair_output_contract(),
        }
    paths = {
        "output": {
            "path": str(output_dir),
            "reuse": "forbidden_reserved_by_mkdir_exist_ok_false",
        },
        "conformer_cache": {
            "path": str(resolved_paths["conformer_cache"]),
            "reuse": cache_reuse_contract["reuse"],
        },
        "projection_cache": {
            "path": str(resolved_paths["projection_cache"]),
            "reuse": cache_reuse_contract["reuse"],
        },
        "cache_reuse_contract": cache_reuse_contract,
    }
    identity_cache_contract = {
        key: value
        for key, value in cache_reuse_contract.items()
        if key != "checked_paths"
    }
    identity_material = {
        "schema": (
            "sam-sequential-growth-run-identity-v2"
            if operation == "grow"
            else "sam-local-repair-run-identity-v1"
        ),
        "normalized_parsed_arguments": normalized_arguments,
        "implementation_and_dependencies": implementation,
        "primary_inputs": primary_inputs,
        "selected_conformer_analysis_evidence": selected_conformers,
        "selected_conformer_source_lock": selected_source_lock,
        "code_snapshot_plan": code_snapshot,
        "execution_contract": execution_contract,
        "scientific_protocol": protocol,
        "cache_reuse_contract_without_absolute_paths": identity_cache_contract,
        "operation_evidence": operation_evidence,
    }
    document = {
        "schema": (
            "sam-sequential-growth-run-request-v1"
            if operation == "grow"
            else "sam-local-repair-run-request-v1"
        ),
        "schema_version": 2,
        "status": "sealed_pre_expensive_request",
        "operation": operation,
        "operation_evidence": operation_evidence,
        "argv": list(argv),
        "parsed_arguments": parsed_arguments,
        "normalized_identity_arguments": normalized_arguments,
        "implementation": implementation,
        "primary_inputs": primary_inputs,
        "selected_conformer_analysis_records": selected_conformers,
        "selected_conformer_source_lock": selected_source_lock,
        "selected_conformer_structure_hashes": [
            {
                "selection_rank": record["selection_rank"],
                "path": record["structure"]["path"],
                "sha256": record["structure"]["sha256"],
            }
            for record in selected_conformers
        ],
        "source_verification_contract": {
            "identity": "SHA256_and_bytes_never_size_mtime",
            "checkpoints": [
                "before_run_request",
                "after_run_request",
                "after_source_conformers",
                "after_direct_geometry_audit",
            ],
            "loaded_ensemble": (
                "one_unique_rank_each;resolved_source_path_exact;direct_source_"
                "symbol_coordinate_geometry_hash_exact"
            ),
            "candidate_domain_forbidden_before_all_checks_pass": True,
        },
        "code_snapshot": code_snapshot,
        "execution_contract": execution_contract,
        "protocol": protocol,
        "paths": paths,
        "resolved_periodic_fractional_axes": list(periodic_axes),
        "p_surface_o_exclusion_experiment": p_surface_o_exclusion_contract,
        "metal_coordination_filter": metal_coordination_contract,
        "identity_material": identity_material,
        "run_identity": {
            "algorithm": "sha256",
            "scope": "canonical_strict_json_run_identity_material_v2",
            "included_fields": [
                "complete_normalized_parsed_arguments",
                "implementation_and_dependency_hashes",
                "five_primary_input_hashes",
                "selected_analysis_and_actual_source_ensemble",
                "all_scientific_protocol_contracts",
                "cache_reuse_mode_without_absolute_cache_paths",
            ],
            "excluded_fields": [
                "output_dir_absolute_path",
                "conformer_cache_absolute_path",
                "projection_cache_absolute_path",
            ],
            "scientific_values_excluded": False,
            "sha256": _payload_sha256(identity_material),
        },
    }
    _validate_run_request_document(document)
    return document


def _artifact_inventory(output_dir: Path, *, excluded_names=()) -> list[dict]:
    output_dir = Path(output_dir)
    excluded = set(excluded_names)
    records = []
    for path in sorted(output_dir.rglob("*")):
        relative_path = path.relative_to(output_dir).as_posix()
        if not path.is_file() or relative_path in excluded:
            continue
        identity = _file_identity(path)
        records.append(
            {
                "path": relative_path,
                "sha256": identity["sha256"],
                "bytes": identity["bytes"],
            }
        )
    return records


def _write_failure_document(
    *,
    output_dir: Path,
    error: BaseException,
    argv_sha256: str,
    implementation_identity: dict,
    operation: str = "grow",
) -> dict:
    """Seal a minimal strict failure without invoking fallible identity again."""

    output_dir = Path(output_dir)
    request_path = output_dir / "run-request.json"
    run_request = None
    if not os.path.lexists(os.fspath(request_path)):
        request_capture = {"status": "absent", "capture_error": None}
    else:
        try:
            if request_path.is_symlink():
                raise RuntimeError("run-request.json is a symlink")
            metadata = request_path.lstat()
            if not stat.S_ISREG(metadata.st_mode):
                raise RuntimeError("run-request.json is not a regular file")
            run_request = {
                "path": "run-request.json",
                "sha256": _sha256(request_path),
                "bytes": metadata.st_size,
            }
            request_capture = {"status": "captured", "capture_error": None}
        except BaseException as capture_error:
            run_request = None
            request_capture = {
                "status": "unavailable",
                "capture_error": _safe_capture_error(capture_error),
            }

    try:
        artifacts = _artifact_inventory(
            output_dir, excluded_names={"failure.json"}
        )
        _strict_json_bytes(artifacts)
        inventory_capture = {
            "status": "captured",
            "capture_error": None,
            "artifact_count": len(artifacts),
        }
    except BaseException as inventory_error:
        artifacts = []
        inventory_capture = {
            "status": "unavailable",
            "capture_error": _safe_capture_error(inventory_error),
            "artifact_count": None,
        }

    document = {
        "schema": (
            "sam-sequential-growth-failure-v1"
            if operation == "grow"
            else "sam-local-repair-failure-v1"
        ),
        "operation": operation,
        "schema_version": 1,
        "status": "failed_runtime_exception",
        "sealed": True,
        "error": {
            **_safe_capture_error(error),
            "traceback": traceback.format_exc(),
        },
        "implementation": implementation_identity,
        "request": {
            "request_intent_sha256": argv_sha256,
            "request_intent_hash_scope": "canonical_strict_json_argv",
            "argv_sha256": argv_sha256,
            "run_request": run_request,
            "run_request_path": (
                None if run_request is None else run_request["path"]
            ),
            "run_request_sha256": (
                None if run_request is None else run_request["sha256"]
            ),
            "run_request_capture": request_capture,
        },
        "artifact_inventory_capture": inventory_capture,
        "artifacts": artifacts,
    }
    write_strict_json_noclobber(output_dir / "failure.json", document)
    return document


def _json_formula(value: str | None) -> dict[str, int] | None:
    if value is None:
        return None
    formula = {str(key): int(count) for key, count in json.loads(value).items()}
    if any(count < 0 for count in formula.values()):
        raise argparse.ArgumentTypeError("Formula counts must be non-negative")
    return formula


_P_SURFACE_O_CONTRACT_SCHEMA = "sam-p-surface-o-experimental-contract-v3"
_P_SURFACE_O_CONTRACT_ROLE = "p_surface_o_experimental_topology_contract_v3"
_P_SURFACE_O_SENSITIVITY_THRESHOLDS_A = (1.8, 1.9, 2.0, 2.1, 2.2)
_P_SURFACE_O_EXCLUSION_SCHEMA = "sam-sequential-p-surface-o-initial-exclusion-v1"


def _verified_p_surface_o_source(path: Path, expected_sha256: str, label: str) -> dict:
    """Read-only hash-lock one external P--surface-O evidence source."""

    path = Path(path).expanduser()
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a regular file: {path}")
    expected = str(expected_sha256).strip().lower()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError(f"{label} expected SHA-256 must be 64 lowercase hexadecimal characters")
    resolved = path.resolve()
    actual = _sha256(resolved)
    if actual != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected}, observed {actual}"
        )
    return {"path": str(resolved), "sha256": actual, "bytes": int(resolved.stat().st_size)}


def _p_surface_o_window(payload: dict, label: str) -> tuple[float, float]:
    """Extract and cross-check the sealed v3 P--O contract window."""

    try:
        parameters = payload["contract"]
        upper = float(parameters["p_o_coordination_cutoff_A"])
        stable = float(parameters["target_stable_cutoff_A"])
        released = float(parameters["target_released_cutoff_A"])
        sensitivity = payload["source_sensitivity"]["p_o_distance_window_A"]
        sensitivity_stable = float(sensitivity["stable_extra_contact_max"])
        sensitivity_released = float(sensitivity["released_repulsion_min_exclusive"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} lacks an unambiguous P--surface-O window") from exc
    values = (upper, stable, released, sensitivity_stable, sensitivity_released)
    if not np.all(np.isfinite(values)) or upper <= 0.0:
        raise ValueError(f"{label} P--surface-O window is not finite and positive")
    if not (
        upper == stable == sensitivity_stable
        and released == sensitivity_released
        and released > upper
    ):
        raise ValueError(
            f"{label} P--surface-O contract/window fields disagree: "
            f"upper={upper}, stable={stable}, released={released}"
        )
    if payload["source_sensitivity"].get("p_o_distance_window_A", {}).get(
        "intermediate_failed"
    ) is not True:
        raise ValueError(f"{label} must preserve the sealed intermediate-failed window evidence")
    return upper, released


def load_p_surface_o_exclusion_contract(
    *,
    contract_path: Path,
    contract_sha256: str,
    source_report_path: Path,
    source_report_sha256: str,
    source_config_path: Path,
    source_config_sha256: str,
) -> dict:
    """Resolve the generic P--surface-O candidate gate from locked evidence.

    The 2.0-A value is deliberately never a runtime default here: it is read
    from the sealed contract and cross-validated against the scratch source,
    report, schema, and inclusive window semantics.
    """

    contract_record = _verified_p_surface_o_source(
        contract_path, contract_sha256, "P--surface-O contract"
    )
    report_record = _verified_p_surface_o_source(
        source_report_path, source_report_sha256, "P--surface-O source report"
    )
    source_record = _verified_p_surface_o_source(
        source_config_path, source_config_sha256, "P--surface-O scratch source"
    )
    try:
        contract_payload = json.loads(Path(contract_record["path"]).read_text(encoding="utf-8"))
        source_payload = json.loads(Path(source_record["path"]).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("P--surface-O evidence JSON could not be parsed") from exc
    for payload, label, expected_schema in (
        (contract_payload, "sealed P--surface-O contract", _P_SURFACE_O_CONTRACT_SCHEMA),
        (source_payload, "P--surface-O scratch source", "sam-p-surface-o-experimental-contract-config-v3"),
    ):
        if not isinstance(payload, dict) or payload.get("schema") != expected_schema:
            raise ValueError(f"{label} schema is not the sealed v3 schema")
        if int(payload.get("schema_version", -1)) != 3:
            raise ValueError(f"{label} schema_version is not 3")
        if payload.get("role") != _P_SURFACE_O_CONTRACT_ROLE:
            raise ValueError(f"{label} role is not {_P_SURFACE_O_CONTRACT_ROLE}")
    if contract_payload.get("prototype0001_deferred") is not True:
        raise ValueError("Sealed P--surface-O contract must defer prototype0001")
    if not isinstance(contract_payload.get("forbidden_candidates"), list):
        raise ValueError("Sealed P--surface-O contract lacks forbidden-candidate evidence")
    contract_upper, contract_released = _p_surface_o_window(
        contract_payload, "sealed P--surface-O contract"
    )
    source_upper, source_released = _p_surface_o_window(
        source_payload, "P--surface-O scratch source"
    )
    if (source_upper, source_released) != (contract_upper, contract_released):
        raise ValueError("P--surface-O contract and scratch source windows disagree")
    input_configuration = contract_payload.get("input_configuration")
    if not isinstance(input_configuration, dict) or input_configuration.get("sha256") != source_record["sha256"]:
        raise ValueError("Sealed P--surface-O contract does not bind the supplied scratch source hash")

    report_text = Path(report_record["path"]).read_text(encoding="utf-8")
    required_report_evidence = (
        "P–surface-O",
        contract_record["sha256"],
        source_record["sha256"],
        "raw production-style audit",
        "2.0",
        "2.2",
    )
    missing_report_evidence = [item for item in required_report_evidence if item not in report_text]
    if missing_report_evidence:
        raise ValueError(
            "P--surface-O source report does not cross-validate the sealed contract: "
            + ", ".join(missing_report_evidence)
        )

    frame = contract_payload.get("surface_frame")
    if not isinstance(frame, dict):
        raise ValueError("Sealed P--surface-O contract lacks a Surface Frame")
    axes = _validated_surface_axes(frame.get("periodic_fractional_axes"))
    normal_axis = frame.get("normal_axis")
    if (
        isinstance(normal_axis, bool)
        or not isinstance(normal_axis, (int, np.integer))
        or int(normal_axis) != next(axis for axis in range(3) if axis not in axes)
    ):
        raise ValueError("Sealed P--surface-O Surface Frame normal is not the nonperiodic complement")
    p_identity = contract_payload.get("system", {}).get("target_p_molecule_local_index_0based")
    p_global = contract_payload.get("system", {}).get("target_p_global_index_0based")
    molecule_indices = contract_payload.get("system", {}).get("molecule_indices_0based")
    molecule_formula = contract_payload.get("system", {}).get("molecule_formula")
    if (
        isinstance(p_identity, bool)
        or not isinstance(p_identity, (int, np.integer))
        or isinstance(p_global, bool)
        or not isinstance(p_global, (int, np.integer))
        or not isinstance(molecule_indices, list)
        or not 0 <= int(p_identity) < len(molecule_indices)
        or int(molecule_indices[int(p_identity)]) != int(p_global)
        or not isinstance(molecule_formula, dict)
        or int(molecule_formula.get("P", 0)) != 1
    ):
        raise ValueError("Sealed P--surface-O P identity is not unambiguous")
    sensitivity = {
        "thresholds_A": [float(value) for value in _P_SURFACE_O_SENSITIVITY_THRESHOLDS_A],
        "required_counts": ["<=1.8", "<=1.9", "<=2.0", "<=2.1", "<=2.2"],
        "contract_window_bins": [
            {"label": "<=contract_upper_bound", "upper_inclusive_A": contract_upper},
            {"label": "(contract_upper_bound, released_cutoff]", "lower_exclusive_A": contract_upper, "upper_inclusive_A": contract_released},
            {"label": ">released_cutoff", "lower_exclusive_A": contract_released},
        ],
    }
    return {
        "schema": _P_SURFACE_O_EXCLUSION_SCHEMA,
        "mode": "p-to-any-working-surface-o-initial-candidate-exclusion",
        "sealed_contract_schema": _P_SURFACE_O_CONTRACT_SCHEMA,
        "sealed_contract_schema_version": 3,
        "role": _P_SURFACE_O_CONTRACT_ROLE,
        "contact_upper_bound_A": contract_upper,
        "contact_upper_bound_inclusive": True,
        "released_cutoff_A": contract_released,
        "surface_axes": list(axes),
        "normal_axis": int(normal_axis),
        "normal_wrapped": False,
        "p_identity_rule": "complete_SAM_topology_unique_anchor_P_molecule_local_0based",
        "surface_o_identity_rule": "every_unique_working_substrate_local_atom_with_element_O",
        "other_substrate_pairs": "completely_skipped_and_never_rejecting",
        "sensitivity": sensitivity,
        "sources": {
            "contract": contract_record,
            "source_report": report_record,
            "scratch_source": source_record,
        },
        "sealed_contract": contract_payload,
        "source_report_cross_validation": {
            "required_markers": list(required_report_evidence),
            "all_markers_present": True,
        },
        "scratch_source_schema": source_payload.get("schema"),
        "provenance_note": (
            "authorized unrelaxed candidate gate; not the historical single-pair "
            "residual filter, production exemption, or stable-chemistry conclusion"
        ),
    }


def verify_p_surface_o_exclusion_contract_sources(contract: dict) -> dict:
    """Re-hash all locked P--surface-O sources without writing them."""

    if not isinstance(contract, dict) or contract.get("schema") != _P_SURFACE_O_EXCLUSION_SCHEMA:
        raise ValueError("Invalid normalized P--surface-O exclusion contract")
    records = {}
    for key, label in (
        ("contract", "P--surface-O contract"),
        ("source_report", "P--surface-O source report"),
        ("scratch_source", "P--surface-O scratch source"),
    ):
        record = contract.get("sources", {}).get(key)
        if not isinstance(record, dict):
            raise ValueError(f"P--surface-O contract lacks source record {key}")
        current = _verified_p_surface_o_source(
            Path(record["path"]), record["sha256"], label
        )
        if current != record:
            raise ValueError(f"P--surface-O source identity changed for {key}")
        records[key] = current
    return records


def _selected_analysis_records(path: Path, maximum: int) -> list[dict]:
    payload = json.loads(path.read_text())
    records = payload.get("selection", {}).get("selected", [])
    if len(records) < maximum:
        raise ValueError(f"Analysis contains {len(records)} selected conformers; need {maximum}")
    selected = [dict(record) for record in records[:maximum]]
    expected_ranks = list(range(1, maximum + 1))
    actual_ranks = [int(record["selection_rank"]) for record in selected]
    if actual_ranks != expected_ranks:
        raise ValueError(f"Expected selection ranks {expected_ranks}; found {actual_ranks}")
    return selected


def build_full_to_working_atom_map(
    *, working_layer_atom_ids_1based, working_symbols
) -> dict[int, dict]:
    """Map full-substrate 1-based IDs to working-substrate 0-based indices.

    The order of ``working_layer_atom_ids_1based`` is the atom order of the
    peeled working structure.  Symbols are required explicitly so later Site
    Instance records can be checked against the actual post-doping elements
    rather than against a material-name whitelist.
    """

    raw_ids = list(working_layer_atom_ids_1based)
    symbols = [str(value) for value in working_symbols]
    if not raw_ids:
        raise ValueError("Working-layer atom IDs must not be empty")
    if len(raw_ids) != len(symbols):
        raise ValueError("Working-layer atom IDs and symbols must align")
    atom_ids = []
    for value in raw_ids:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise ValueError("Working-layer atom IDs must be 1-based integers")
        atom_id = int(value)
        if atom_id < 1:
            raise ValueError("Working-layer atom IDs must be positive")
        atom_ids.append(atom_id)
    if len(set(atom_ids)) != len(atom_ids):
        raise ValueError("Working-layer atom IDs contain duplicate values")
    if any(not symbol for symbol in symbols):
        raise ValueError("Working-layer elements must be non-empty symbols")
    return {
        full_id: {
            "full_atom_id_1based": full_id,
            "working_atom_index_0based": working_index,
            "actual_element": symbols[working_index],
        }
        for working_index, full_id in enumerate(atom_ids)
    }


_METAL_COORDINATION_FILTER_SCHEMA = "samflow-metal-coordination-filter-v1"


def build_metal_coordination_filter_contract(
    *,
    mode: str = "exclude-six-coordinated",
    cutoff_A: float = 2.7,
    max_allowed_coordination: int = 5,
    target_elements: tuple[str, ...] | list[str] = ("In", "Sn"),
    ligand_elements: tuple[str, ...] | list[str] = ("O",),
    periodic_axes: tuple[int, int] = (0, 1),
    normal_axis: int = 2,
) -> dict:
    """Build the configuration contract for pre-adsorption metal coordination candidate filtering."""
    if mode not in ("off", "exclude-six-coordinated"):
        raise ValueError(f"Invalid metal coordination filter mode: {mode}")
    axes = _validated_surface_axes(periodic_axes)
    norm_ax = int(normal_axis)
    if norm_ax not in (0, 1, 2) or norm_ax in axes:
        raise ValueError("normal_axis must be the third axis distinct from periodic_axes")
    cutoff = float(cutoff_A)
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("Metal coordination cutoff must be positive and finite")
    max_coord = int(max_allowed_coordination)
    if max_coord < 0:
        raise ValueError("Max allowed coordination must be non-negative")
    return {
        "schema": _METAL_COORDINATION_FILTER_SCHEMA,
        "mode": mode,
        "cutoff_A": cutoff,
        "max_allowed_coordination": max_coord,
        "target_elements": [str(e) for e in target_elements],
        "ligand_elements": [str(e) for e in ligand_elements],
        "periodic_axes": list(axes),
        "normal_axis": norm_ax,
        "normal_wrapped": False,
        "all_mapped_metals_must_pass": True,
        "hypothesis_testing_note": (
            "Exploratory test of the hypothesis that pre-adsorption 6-coordinate "
            "In/Sn cannot act as direct phosphate-O binding metals"
        ),
    }


def audit_candidate_metal_coordinations(
    *,
    registered_interface_bonds: list[dict],
    metal_records_by_full_id: dict[int, dict],
    metal_records_by_working_index: dict[int, dict] | None = None,
    max_allowed_coordination: int = 5,
) -> dict:
    """Audit mapped substrate metals for coordination number before adsorption."""
    mapped_metals = []
    forbidden_metals = []
    max_cn = 0
    unique_full_ids = set()

    for bond in registered_interface_bonds:
        full_id = bond.get("full_metal_atom_id_1based")
        working_idx = bond.get("working_metal_atom_index_0based")
        if full_id is not None:
            full_id = int(full_id)
            if full_id in unique_full_ids:
                continue
            unique_full_ids.add(full_id)
            record = metal_records_by_full_id.get(full_id)
        elif working_idx is not None and metal_records_by_working_index is not None:
            working_idx = int(working_idx)
            if working_idx in unique_full_ids:
                continue
            unique_full_ids.add(working_idx)
            record = metal_records_by_working_index.get(working_idx)
        else:
            record = None

        if record is None:
            raise ValueError(
                f"Mapped metal (full_id={full_id}, working_idx={working_idx}) not found in substrate coordination records"
            )

        cn = int(record["coordination_number"])
        elem = str(record["element"])
        max_cn = max(max_cn, cn)
        item = {
            "full_metal_atom_id_1based": full_id,
            "working_metal_atom_index_0based": int(record["working_atom_index_0based"]),
            "element": elem,
            "coordination_number": cn,
            "is_six_coordinated": cn > max_allowed_coordination,
        }
        mapped_metals.append(item)
        if cn > max_allowed_coordination:
            forbidden_metals.append(item)

    forbidden = len(forbidden_metals) > 0
    elems = {m["element"] for m in mapped_metals}
    if elems == {"In"}:
        elem_category = "In_only"
    elif elems == {"Sn"}:
        elem_category = "Sn_only"
    elif elems == {"In", "Sn"}:
        elem_category = "mixed_In_Sn"
    else:
        elem_category = "_".join(sorted(elems))

    denticity = len(registered_interface_bonds)
    return {
        "passed": not forbidden,
        "forbidden": forbidden,
        "denticity": denticity,
        "unique_metal_count": len(mapped_metals),
        "element_category": elem_category,
        "max_coordination_number": max_cn,
        "mapped_metals": mapped_metals,
        "forbidden_metals": forbidden_metals,
    }


def audit_site_instance_metal_coordinations(
    *,
    instance: dict,
    metal_records_by_full_id: dict[int, dict],
    max_allowed_coordination: int = 5,
) -> dict:
    """Audit one Site Instance's registered mapped metals for coordination number before adsorption."""
    site_id = instance.get("site_instance_id", "<unknown-site>")
    donor_mapping = instance.get("donor_metal_mapping", [])
    raw_actual_ids = instance.get("actual_metal_atom_ids_1based", [])

    donor_metal_ids = {
        int(mapping["metal_atom_id_1based"])
        for mapping in donor_mapping
        if mapping.get("metal_atom_id_1based") is not None
    }
    actual_metal_ids = {
        int(mid) for mid in raw_actual_ids
    }
    if donor_metal_ids and actual_metal_ids and donor_metal_ids != actual_metal_ids:
        raise ValueError(
            f"{site_id}: donor_metal_mapping metals {sorted(donor_metal_ids)} disagree with "
            f"actual_metal_atom_ids_1based {sorted(actual_metal_ids)}"
        )
    mapped_metal_ids = donor_metal_ids or actual_metal_ids

    if not mapped_metal_ids:
        raise ValueError(f"{site_id}: site instance lacks mapped metal atom IDs")

    mapped_metals = []
    forbidden_metals = []
    max_cn = 0

    for full_id in sorted(mapped_metal_ids):
        record = metal_records_by_full_id.get(full_id)
        if record is None:
            raise ValueError(
                f"{site_id}: mapped metal ID {full_id} not found in substrate coordination records"
            )
        cn = int(record["coordination_number"])
        elem = str(record["element"])
        max_cn = max(max_cn, cn)
        item = {
            "full_metal_atom_id_1based": full_id,
            "working_metal_atom_index_0based": int(record["working_atom_index_0based"]),
            "element": elem,
            "coordination_number": cn,
            "is_six_coordinated": cn > max_allowed_coordination,
        }
        mapped_metals.append(item)
        if cn > max_allowed_coordination:
            forbidden_metals.append(item)

    forbidden = len(forbidden_metals) > 0
    elems = {m["element"] for m in mapped_metals}
    if elems == {"In"}:
        elem_category = "In_only"
    elif elems == {"Sn"}:
        elem_category = "Sn_only"
    elif elems == {"In", "Sn"}:
        elem_category = "mixed_In_Sn"
    else:
        elem_category = "_".join(sorted(elems))

    denticity = int(instance.get("final_denticity", len(donor_mapping)))
    return {
        "site_instance_id": str(site_id),
        "passed": not forbidden,
        "forbidden": forbidden,
        "denticity": denticity,
        "unique_metal_count": len(mapped_metals),
        "element_category": elem_category,
        "max_coordination_number": max_cn,
        "mapped_metals": mapped_metals,
        "forbidden_metals": forbidden_metals,
    }


def _validated_registered_bond_window(distance_window_A) -> tuple[float, float]:
    try:
        minimum, maximum = (float(value) for value in distance_window_A)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Registered bond distance window must contain minimum and maximum"
        ) from exc
    if (
        not np.isfinite([minimum, maximum]).all()
        or minimum <= 0.0
        or maximum < minimum
    ):
        raise ValueError(
            "Registered bond distance window must be a finite positive range"
        )
    return minimum, maximum


def resolve_candidate_registered_bonds(
    *,
    instance: dict,
    target_donor_labels,
    oxygen_permutation,
    molecule_symbols,
    full_to_working_atom_map: dict[int, dict],
    distance_window_A,
) -> list[dict]:
    """Resolve one candidate's exact donor-to-working-metal bond records.

    Site Instance IDs are expressed in the full materialized substrate while
    sequential growth uses a peeled working layer.  This gate validates both
    schemas and joins target donor labels to the ordered molecule-local oxygen
    permutation without element-based inference.
    """

    site_id = str(instance.get("site_instance_id", "<unknown-site>"))
    if distance_window_A is None:
        minimum = maximum = None
    else:
        minimum, maximum = _validated_registered_bond_window(distance_window_A)
    labels = [str(value) for value in target_donor_labels]
    if not labels or any(not value for value in labels):
        raise ValueError(f"{site_id}: target donor labels must be non-empty")
    if len(set(labels)) != len(labels):
        raise ValueError(f"{site_id}: target donor labels contain duplicates")

    permutation = []
    for value in oxygen_permutation:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise ValueError(f"{site_id}: oxygen permutation must contain local integers")
        permutation.append(int(value))
    if len(permutation) != len(labels):
        raise ValueError(
            f"{site_id}: donor-label/oxygen-permutation cardinality mismatch"
        )
    if len(set(permutation)) != len(permutation):
        raise ValueError(f"{site_id}: oxygen permutation contains duplicate atoms")

    symbols = [str(value) for value in molecule_symbols]
    if any(index < 0 or index >= len(symbols) for index in permutation):
        raise ValueError(f"{site_id}: oxygen permutation index is outside the molecule")
    if any(symbols[index] != "O" for index in permutation):
        raise ValueError(f"{site_id}: mapped donor atom element must be O")

    raw_actual_ids = instance.get("actual_metal_atom_ids_1based")
    actual_metals = instance.get("actual_metals")
    donor_mapping = instance.get("donor_metal_mapping")
    if not isinstance(raw_actual_ids, list) or not raw_actual_ids:
        raise ValueError(f"{site_id}: missing actual metal atom IDs")
    if not isinstance(actual_metals, list) or not actual_metals:
        raise ValueError(f"{site_id}: missing actual metal records")
    if not isinstance(donor_mapping, list) or not donor_mapping:
        raise ValueError(f"{site_id}: missing donor-metal mapping")

    actual_ids = []
    for value in raw_actual_ids:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise ValueError(f"{site_id}: actual metal IDs must be integers")
        actual_ids.append(int(value))
    if len(set(actual_ids)) != len(actual_ids):
        raise ValueError(f"{site_id}: actual metal IDs contain duplicates")

    actual_by_full_id = {}
    for metal in actual_metals:
        if not isinstance(metal, dict):
            raise ValueError(f"{site_id}: malformed actual metal record")
        value = metal.get("atom_id_1based")
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise ValueError(f"{site_id}: actual metal record lacks a valid full ID")
        full_id = int(value)
        if full_id in actual_by_full_id:
            raise ValueError(f"{site_id}: actual metal records contain duplicate IDs")
        if full_id not in full_to_working_atom_map:
            raise ValueError(
                f"{site_id}: full metal atom ID {full_id} is not in the working layer"
            )
        actual_element = metal.get("actual_element_after_doping")
        if not isinstance(actual_element, str) or not actual_element:
            raise ValueError(
                f"{site_id}: actual metal {full_id} lacks its post-doping element"
            )
        working_record = full_to_working_atom_map[full_id]
        if actual_element != working_record.get("actual_element"):
            raise ValueError(
                f"{site_id}: actual metal {full_id} element {actual_element!r} "
                f"does not match working element {working_record.get('actual_element')!r}"
            )
        actual_by_full_id[full_id] = {
            "working_atom_index_0based": int(
                working_record["working_atom_index_0based"]
            ),
            "actual_element": actual_element,
        }
    if set(actual_ids) != set(actual_by_full_id):
        raise ValueError(
            f"{site_id}: actual metal ID list and actual metal records disagree"
        )

    mapping_by_label = {}
    for mapping in donor_mapping:
        if not isinstance(mapping, dict):
            raise ValueError(f"{site_id}: malformed donor-metal mapping record")
        label = mapping.get("probe_donor_id")
        if not isinstance(label, str) or not label:
            raise ValueError(f"{site_id}: donor-metal mapping lacks a donor label")
        if label in mapping_by_label:
            raise ValueError(f"{site_id}: duplicate donor label {label!r}")
        full_id = mapping.get("metal_atom_id_1based")
        if isinstance(full_id, (bool, np.bool_)) or not isinstance(
            full_id, (int, np.integer)
        ):
            raise ValueError(f"{site_id}: donor {label} lacks a valid full metal ID")
        full_id = int(full_id)
        if full_id not in full_to_working_atom_map:
            raise ValueError(
                f"{site_id}: full metal atom ID {full_id} is not in the working layer"
            )
        if full_id not in actual_by_full_id:
            raise ValueError(
                f"{site_id}: donor {label} maps to an undeclared actual metal"
            )
        mapping_by_label[label] = full_id
    if set(mapping_by_label) != set(labels):
        raise ValueError(
            f"{site_id}: donor-metal mapping labels do not match target donor labels"
        )
    if set(mapping_by_label.values()) != set(actual_ids):
        raise ValueError(
            f"{site_id}: donor-metal mapping and actual metal identities disagree"
        )
    denticity = instance.get("final_denticity")
    if isinstance(denticity, (bool, np.bool_)) or not isinstance(
        denticity, (int, np.integer)
    ):
        raise ValueError(f"{site_id}: final_denticity must be an integer")
    if int(denticity) != len(labels) or len(mapping_by_label) != len(labels):
        raise ValueError(f"{site_id}: registered donor mapping cardinality mismatch")

    records = []
    for label, molecule_index in zip(labels, permutation):
        full_id = mapping_by_label[label]
        actual = actual_by_full_id[full_id]
        record = {
                "donor_label": label,
                "molecule_atom_index_0based": molecule_index,
                "molecule_element": symbols[molecule_index],
                "full_metal_atom_id_1based": full_id,
                "working_metal_atom_index_0based": actual[
                    "working_atom_index_0based"
                ],
                "actual_metal_element": actual["actual_element"],
            }
        if minimum is not None:
            record["distance_window_A"] = [minimum, maximum]
        records.append(record)
    return records


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
    return bool(
        weight_a >= -tolerance
        and weight_b >= -tolerance
        and weight_a + weight_b <= 1.0 + tolerance
    )


def _discover_uncovered_cn5_metals(
    *,
    substrate_positions: np.ndarray,
    substrate_symbols: np.ndarray,
    cell: np.ndarray,
    periodic_axes: tuple[int, int] = (0, 1),
    normal_axis: int = 2,
    retained_site_instances: list[dict],
    full_to_working_atom_map: dict[int, dict],
    cutoff_A: float = 2.7,
    top_surface_depth_A: float = 0.7,
    target_elements: tuple[str, ...] | list[str] = ("In", "Sn"),
    ligand_elements: tuple[str, ...] | list[str] = ("O",),
) -> list[dict]:
    """Dynamically discover top-surface CN == 5 metals not covered by any retained multidentate sites."""

    coordination_data = calculate_substrate_metal_coordinations(
        substrate_positions=substrate_positions,
        substrate_symbols=substrate_symbols,
        cell=cell,
        periodic_axes=periodic_axes,
        normal_axis=normal_axis,
        cutoff_A=cutoff_A,
        target_elements=target_elements,
        ligand_elements=ligand_elements,
        full_to_working_atom_map=full_to_working_atom_map,
    )
    records_by_full_id = coordination_data["metal_records_by_full_id"]

    target_set = set(target_elements)
    metal_working_indices = [
        i for i, s in enumerate(substrate_symbols) if s in target_set
    ]
    if not metal_working_indices:
        return []

    metal_normals = substrate_positions[metal_working_indices, normal_axis]
    z_max_metal = float(np.max(metal_normals))
    top_threshold = z_max_metal - top_surface_depth_A - 1.0e-8

    top_cn5_metals = []
    for full_id, record in records_by_full_id.items():
        w_idx = int(record["working_atom_index_0based"])
        z_pos = float(substrate_positions[w_idx, normal_axis])
        cn = int(record["coordination_number"])
        if z_pos >= top_threshold and cn == 5:
            top_cn5_metals.append(record)

    covered_metal_ids = set()
    for instance in retained_site_instances:
        raw_ids = instance.get("actual_metal_atom_ids_1based") or []
        for mid in raw_ids:
            covered_metal_ids.add(int(mid))
        for mapping in instance.get("donor_metal_mapping") or []:
            if mapping.get("metal_atom_id_1based") is not None:
                covered_metal_ids.add(int(mapping["metal_atom_id_1based"]))

    uncovered = [
        record for record in top_cn5_metals
        if int(record["full_metal_atom_id_1based"]) not in covered_metal_ids
    ]
    uncovered.sort(key=lambda rec: int(rec["full_metal_atom_id_1based"]))
    return uncovered


def _resolve_conformer_source_anchor_height(
    conformer: dict, normal: np.ndarray
) -> float | None:
    """Resolve the relaxed 3-O centroid height above its source top metal for a conformer."""
    if "source_anchor_height_A" in conformer and conformer["source_anchor_height_A"] is not None:
        try:
            val = float(conformer["source_anchor_height_A"])
            if np.isfinite(val) and val > 0.0:
                return val
        except (TypeError, ValueError):
            pass

    source_path = conformer.get("source")
    if not source_path or not Path(source_path).is_file():
        return None
    try:
        atoms = read_typed_structure(source_path)
        symbols = np.asarray(atoms.get_chemical_symbols(), dtype=str)
        positions = np.asarray(atoms.positions, dtype=float)
        o_locals = conformer.get("o_locals", [])
        if not o_locals:
            return None
        metal_indices = np.flatnonzero(np.isin(symbols, ["In", "Sn"]))
        if len(metal_indices) == 0:
            return None
        conformer_coords = np.asarray(conformer.get("coordinates", []), dtype=float)
        if len(conformer_coords) == 0 or len(atoms) < len(conformer_coords):
            return None
        sam_offset = len(atoms) - len(conformer_coords)
        sub_metal_indices = [i for i in metal_indices if i < sam_offset]
        if not sub_metal_indices:
            sub_metal_indices = metal_indices
        source_metal_z = float(np.max(positions[sub_metal_indices] @ normal))
        sam_o_positions = positions[sam_offset:][o_locals]
        source_o_z = float(np.mean(sam_o_positions @ normal))
        rel_height = source_o_z - source_metal_z
        if np.isfinite(rel_height) and rel_height > 0.0:
            return float(rel_height)
    except Exception:
        pass
    return None


def _generate_single_metal_triangle_candidates(
    *,
    metal_record: dict,
    substrate_positions: np.ndarray,
    substrate_symbols: np.ndarray,
    cell: np.ndarray,
    periodic_axes: tuple[int, int],
    surface_frame: dict,
    conformers: list[dict],
    metrics_by_rank: dict[int, dict],
    substrate_height_clearance_A: float = 1.0,
    rotation_step_deg: float = 30.0,
    translation_offsets_A: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0),
) -> list[dict]:
    """Generate discrete candidate poses where the metal is inside the phosphonate 3-O XY projected triangle."""

    u_axis = np.asarray(surface_frame["u_cartesian_unit"], dtype=float)
    v_axis = np.asarray(surface_frame["v_cartesian_unit"], dtype=float)
    normal = np.asarray(surface_frame["outward_normal_cartesian_unit"], dtype=float)
    normal /= np.linalg.norm(normal)

    w_idx = int(metal_record["working_atom_index_0based"])
    r_metal = np.asarray(substrate_positions[w_idx], dtype=float)
    z_sub_max = float(np.max(substrate_positions @ normal))
    full_metal_id = int(metal_record["full_metal_atom_id_1based"])
    metal_elem = str(metal_record["element"])

    pbc = np.zeros(3, dtype=bool)
    pbc[list(periodic_axes)] = True

    candidates = []
    step_deg = max(1.0, float(rotation_step_deg))
    angles_deg = np.arange(0.0, 360.0, step_deg)

    for conformer in conformers:
        anchor_z_offset_A = _resolve_conformer_source_anchor_height(conformer, normal)
        if anchor_z_offset_A is None:
            continue

        rank = int(conformer["source_rank"])
        metrics = metrics_by_rank.get(rank, {"cluster_id": rank})
        source_coords = np.asarray(conformer["coordinates"], dtype=float)
        conformer_symbols = np.asarray(conformer["symbols"], dtype=str)
        conformer_sha256 = _conformer_geometry_sha256(source_coords, conformer_symbols)

        p_idx = int(conformer["p_local"])
        o_indices = [int(idx) for idx in conformer["o_locals"]]
        non_anchor_indices = [
            i for i in range(len(source_coords))
            if i != p_idx and i not in o_indices
        ]

        # Center conformer at the centroid of the 3 phosphonate oxygen atoms
        o_centroid = np.mean(source_coords[o_indices], axis=0)
        coords_centered = source_coords - o_centroid

        for angle_deg in angles_deg:
            rad = np.radians(angle_deg)
            cos_a = np.cos(rad)
            sin_a = np.sin(rad)
            # Rodrigues rotation matrix around normal
            rot_mat = (
                cos_a * np.eye(3)
                + sin_a * np.array([
                    [0, -normal[2], normal[1]],
                    [normal[2], 0, -normal[0]],
                    [-normal[1], normal[0], 0],
                ])
                + (1.0 - cos_a) * np.outer(normal, normal)
            )
            rotated = coords_centered @ rot_mat.T

            for dx in translation_offsets_A:
                for dy in translation_offsets_A:
                    base_pos = (
                        r_metal
                        + dx * u_axis
                        + dy * v_axis
                        + anchor_z_offset_A * normal
                    )
                    transformed = rotated + base_pos

                    # 1. Barycentric test: metal inside the 3-O projected triangle
                    o_trans = transformed[o_indices]
                    delta_o = o_trans - o_trans[0]
                    delta_m = r_metal - o_trans[0]
                    mic_o, _ = find_mic(delta_o, cell=cell, pbc=pbc)
                    mic_m, _ = find_mic(delta_m[None, :], cell=cell, pbc=pbc)
                    mic_o = np.asarray(mic_o, dtype=float)
                    mic_m = np.asarray(mic_m, dtype=float)

                    triangle_uv = np.column_stack((mic_o @ u_axis, mic_o @ v_axis))
                    metal_uv = np.asarray([np.dot(mic_m[0], u_axis), np.dot(mic_m[0], v_axis)])

                    if not _point_in_triangle_2d(metal_uv, triangle_uv):
                        continue

                    # 2. Non-anchor height clearance gate: >= z_sub_max + substrate_height_clearance_A
                    if non_anchor_indices:
                        non_anchor_heights = transformed[non_anchor_indices] @ normal
                        if float(np.min(non_anchor_heights)) < z_sub_max + float(substrate_height_clearance_A) - 1.0e-8:
                            continue

                    site_id = f"site-single-metal-triangle-atom{full_metal_id:04d}"
                    cand_id = (
                        f"{site_id}-rank{rank:02d}-rot{int(round(angle_deg)):03d}"
                        f"-dx{int(round(dx * 100)):+04d}-dy{int(round(dy * 100)):+04d}"
                    )

                    candidate = {
                        "site_instance_id": site_id,
                        "site_prototype_id": "site-prototype-phosphonic-acid-single-metal",
                        "site_kind": "single_metal_triangle",
                        "phase_priority": 1,
                        "selection_rank": rank,
                        "cluster_id": int(metrics.get("cluster_id", rank)),
                        "source": conformer.get("source", f"rank-{rank}"),
                        "target_donor_labels": (),
                        "oxygen_permutation": [int(idx) for idx in conformer["o_locals"]],
                        "registered_interface_bonds": [],
                        "occupied_metal_ids": [full_metal_id],
                        "anchor_metal_atom_id_1based": full_metal_id,
                        "anchor_metal_working_index_0based": w_idx,
                        "anchor_metal_element": metal_elem,
                        "conformer_anchor_height_A": float(anchor_z_offset_A),
                        "headgroup_rmsd_A": None,
                        "substrate_minimum_distance_A": None,
                        "substrate_clearance_ratio": None,
                        "conformer_sha256": conformer_sha256,
                        "p_molecule_local_index_0based": p_idx,
                        "headgroup_local_indices_0based": [p_idx, *o_indices],
                        "surface_frame": surface_frame,
                        "anchor_cartesian_A": r_metal.tolist(),
                        "candidate_id": cand_id,
                        "coordinates": transformed,
                        "symbols": conformer_symbols,
                    }
                    if "relative_total_energy_eV" in metrics:
                        candidate["conformer_energy_eV"] = float(
                            metrics["relative_total_energy_eV"]
                        )
                    candidates.append(candidate)

    return candidates


def _validated_partition_atom_ids(
    groups: dict, *, key: str, label: str, substrate_atom_count: int
) -> list[int]:
    raw_ids = groups.get(key)
    if not isinstance(raw_ids, list):
        raise ValueError(f"Layer groups lack {key}")
    if not raw_ids:
        raise ValueError(f"{label} atom IDs must not be empty")
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        for value in raw_ids
    ):
        raise ValueError(f"{label} atom IDs must be 1-based integers")
    atom_ids = [int(value) for value in raw_ids]
    if len(set(atom_ids)) != len(atom_ids):
        raise ValueError(f"{label} atom IDs contain duplicates")
    if min(atom_ids) < 1 or max(atom_ids) > substrate_atom_count:
        raise ValueError(f"{label} atom IDs fall outside substrate")
    return atom_ids


def _load_working_substrate(substrate_path: Path, layer_groups_path: Path):
    substrate = read_typed_structure(substrate_path)
    groups = json.loads(layer_groups_path.read_text())
    if not isinstance(groups, dict):
        raise ValueError("Layer groups must be a JSON object")

    # Validate the complete two-way partition before taking any atom slice.
    restore_ids = _validated_partition_atom_ids(
        groups,
        key="restore_layer_atom_ids_1based",
        label="Restore-layer",
        substrate_atom_count=len(substrate),
    )
    working_ids = _validated_partition_atom_ids(
        groups,
        key="working_layer_atom_ids_1based",
        label="Working-layer",
        substrate_atom_count=len(substrate),
    )
    restore_set = set(restore_ids)
    working_set = set(working_ids)
    if restore_set.intersection(working_set):
        raise ValueError("Restore-layer and working-layer atom IDs must be mutually exclusive")
    if restore_set.union(working_set) != set(range(1, len(substrate) + 1)):
        raise ValueError(
            "Layer-group union must contain every substrate atom exactly once"
        )

    working = substrate[[value - 1 for value in working_ids]].copy()
    working.set_cell(substrate.cell)
    working.set_pbc(substrate.pbc)
    build_full_to_working_atom_map(
        working_layer_atom_ids_1based=working_ids,
        working_symbols=working.get_chemical_symbols(),
    )
    return working, groups


def _prototype_map(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text())
    prototypes = payload.get("site_prototypes", [])
    result = {record["site_prototype_id"]: record for record in prototypes}
    if len(result) != len(prototypes) or not result:
        raise ValueError("Site Prototype library is empty or has duplicate IDs")
    return result


def _target_headgroup(instance: dict, prototype: dict) -> tuple[np.ndarray, list[str]]:
    instance_ideal = np.asarray(instance["ideal_site_location"]["cartesian_A"], dtype=float)
    prototype_ideal = np.asarray(prototype["ideal_site_location"]["cartesian_A"], dtype=float)
    translation = instance_ideal - prototype_ideal
    relaxed = prototype["relaxed_adsorption_location"]
    anchor = np.asarray(relaxed["anchor_cartesian_A"], dtype=float) + translation
    donors = sorted(
        relaxed["binding_oxygen_positions"], key=lambda item: item["probe_donor_id"]
    )
    donor_labels = [str(item["probe_donor_id"]) for item in donors]
    expected_denticity = int(prototype["final_denticity"])
    if len(donors) != expected_denticity or expected_denticity not in (1, 2, 3):
        raise ValueError(
            f"Prototype {prototype['site_prototype_id']} provides {len(donors)} donor "
            f"targets for final_denticity={expected_denticity}; only monodentate, "
            "bidentate, and tridentate targets are supported"
        )
    if len(set(donor_labels)) != len(donor_labels):
        raise ValueError(
            f"Prototype {prototype['site_prototype_id']} has duplicate donor labels"
        )
    coordinates = np.vstack(
        [anchor]
        + [np.asarray(item["cartesian_A"], dtype=float) + translation for item in donors]
    )
    return coordinates, donor_labels


def _hash_update_array(digest, values, dtype="<f8"):
    """Hash an array without rounding its values.

    Arrays are reordered to C-contiguous little-endian bytes so the digest
    depends on every value exactly; shapes are included so shape changes are
    detected too.
    """

    array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    digest.update(str(array.shape).encode("utf-8", "surrogatepass"))
    digest.update(array.tobytes())


def _conformer_geometry_sha256(coordinates, symbols) -> str:
    """Return the structure hash of one isolated conformer geometry."""

    digest = hashlib.sha256()
    digest.update(b"conformer-geometry-v1")
    _hash_update_array(digest, np.asarray(coordinates, dtype=float))
    _hash_update_array(digest, np.asarray(symbols, dtype=str), dtype="<U")
    return digest.hexdigest()


def _candidate_id_from_fields(
    *,
    site_instance_id,
    site_prototype_id,
    prototype_evidence,
    selection_rank,
    conformer_sha256,
    donor_labels,
    oxygen_permutation,
    symbols,
) -> str:
    """Return a stable candidate identity from its full provenance.

    The ID covers the Site Instance, the Site Prototype plus its evidence hash,
    the selection rank, the conformer structure hash, the ordered donor
    labels, the oxygen permutation, and the atom-order/symbol hash.
    """

    digest = hashlib.sha256()
    digest.update(b"candidate-id-v1")
    for field in (
        site_instance_id,
        site_prototype_id,
        prototype_evidence,
        str(int(selection_rank)),
        conformer_sha256,
        json.dumps([str(label) for label in donor_labels], separators=(",", ":")),
        json.dumps([int(value) for value in oxygen_permutation], separators=(",", ":")),
    ):
        digest.update(str(field).encode("utf-8", "surrogatepass"))
    _hash_update_array(digest, np.asarray(symbols, dtype=str), dtype="<U")
    return "candidate-" + digest.hexdigest()[:32]


def _common_surface_frame(domains) -> dict:
    """Return the single Surface Frame shared by every candidate.

    Every candidate must carry a finite orthonormal right-handed Surface Frame
    and all frames must agree within ``1.0e-8``; disagreement is a hard error
    rather than a silently chosen frame.
    """

    frames = []
    for candidates in domains.values():
        for candidate in candidates:
            frame = candidate.get("surface_frame")
            if frame is None:
                raise ValueError("Candidate missing surface_frame")
            frames.append(frame)
    if not frames:
        raise ValueError("Candidate domains contain no candidates")
    reference = frames[0]
    surface_frame_coordinates(np.zeros((1, 3)), reference)
    for frame in frames[1:]:
        surface_frame_coordinates(np.zeros((1, 3)), frame)
        for key in ("u_cartesian_unit", "v_cartesian_unit", "outward_normal_cartesian_unit"):
            if not np.allclose(
                np.asarray(frame[key], dtype=float),
                np.asarray(reference[key], dtype=float),
                atol=1.0e-8,
                rtol=0.0,
            ):
                raise ValueError(
                    "Candidate domains disagree on the common Surface Frame"
                )
    return dict(reference)


def _candidate_domain_fingerprint(
    domains, *, cell, periodic_axes, substrate_collision_contract=None,
    p_surface_o_exclusion_contract=None,
    metal_coordination_contract=None,
) -> str:
    """Hash the candidate domain without ordering or seed/policy input.

    ``None`` and the explicit legacy ``skip`` contract execute the historical
    v1 byte path exactly.  Corrected ``check`` and ``strict-vdw`` contracts use
    a separate namespace and include both the collision contract and every
    exact registered interface bond, preventing cross-mode cache reuse.
    """

    metal_coord_active = (
        metal_coordination_contract is not None
        and metal_coordination_contract.get("mode") != "off"
    )
    p_surface_o_active = p_surface_o_exclusion_contract is not None
    legacy_skip = (
        not metal_coord_active
        and not p_surface_o_active
        and (
            substrate_collision_contract is None
            or (
                isinstance(substrate_collision_contract, dict)
                and substrate_collision_contract.get("mode") == "skip"
            )
        )
    )
    digest = hashlib.sha256()
    if legacy_skip:
        digest.update(b"candidate-domain-fingerprint-v1")
    elif metal_coord_active and p_surface_o_active:
        digest.update(b"candidate-domain-fingerprint-metal-coord-p-surface-o-v1")
    elif metal_coord_active:
        digest.update(b"candidate-domain-fingerprint-metal-coord-v1")
    elif p_surface_o_active:
        digest.update(b"candidate-domain-fingerprint-p-surface-o-v1")
    else:
        digest.update(b"candidate-domain-fingerprint-interface-v2")

    if not legacy_skip:
        if substrate_collision_contract is not None:
            digest.update(
                json.dumps(
                    substrate_collision_contract,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            )
        if p_surface_o_exclusion_contract is not None:
            digest.update(
                json.dumps(
                    p_surface_o_exclusion_contract,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            )
        if metal_coord_active:
            digest.update(
                json.dumps(
                    metal_coordination_contract,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            )
    digest.update(str(tuple(int(axis) for axis in periodic_axes)).encode())
    _hash_update_array(digest, np.asarray(cell, dtype=float))
    for site_id in sorted(domains):
        digest.update(site_id.encode("utf-8", "surrogatepass"))
        for candidate in sorted(domains[site_id], key=lambda item: item["candidate_id"]):
            digest.update(candidate["candidate_id"].encode("utf-8", "surrogatepass"))
            for key in (
                "site_instance_id",
                "site_prototype_id",
                "selection_rank",
                "conformer_sha256",
                "target_donor_labels",
                "oxygen_permutation",
            ):
                value = candidate[key]
                if isinstance(value, (list, tuple)):
                    value = json.dumps(
                        [str(item) for item in value], separators=(",", ":")
                    )
                digest.update(str(value).encode("utf-8", "surrogatepass"))
            _hash_update_array(digest, np.asarray(candidate["symbols"], dtype=str), dtype="<U")
            _hash_update_array(digest, np.asarray(candidate["coordinates"], dtype=float))
            if not legacy_skip:
                if candidate.get("site_kind") == "single_metal_triangle":
                    digest.update(b"single_metal_triangle:")
                    digest.update(
                        json.dumps(
                            sorted(candidate.get("occupied_metal_ids", [])),
                            separators=(",", ":"),
                        ).encode("utf-8")
                    )
                else:
                    bonds = candidate.get("registered_interface_bonds")
                    if not isinstance(bonds, list) or not bonds:
                        raise ValueError(
                            "Strict/check candidate missing registered interface bonds"
                        )
                    digest.update(
                        json.dumps(
                            bonds,
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        ).encode("utf-8")
                    )
                frame = candidate.get("surface_frame")
                if not isinstance(frame, dict):
                    raise ValueError("Strict/check candidate missing Surface Frame")
                surface_frame_coordinates(np.zeros((1, 3)), frame)
                digest.update(
                    json.dumps(
                        frame,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8")
                )
    return digest.hexdigest()


_PROJECTION_ALGORITHM_VERSION = "projection-cache-v1"
_COLLISION_THRESHOLDS_A = {"H-H": 1.2, "H-heavy": 1.4, "heavy-heavy": 1.8}
_COLLISION_ALGORITHM_VERSION = "surface-collision-report-v2-ase-find-mic-partial-pbc"


def _validate_collision_thresholds(thresholds_A) -> dict:
    """Validate and return the exact SAM--SAM collision thresholds."""

    if thresholds_A is None:
        return dict(_COLLISION_THRESHOLDS_A)
    thresholds = {str(key): float(value) for key, value in thresholds_A.items()}
    if sorted(thresholds) != ["H-H", "H-heavy", "heavy-heavy"]:
        raise ValueError(
            "collision thresholds must define H-H, H-heavy, and heavy-heavy"
        )
    if any(not np.isfinite(value) or value <= 0.0 for value in thresholds.values()):
        raise ValueError("collision thresholds must be finite positive numbers")
    return thresholds


def _validated_surface_axes(periodic_axes) -> tuple[int, int]:
    if isinstance(periodic_axes, (str, bytes)):
        raise ValueError("periodic_axes must identify two distinct cell axes")
    try:
        raw_axes = tuple(periodic_axes)
    except TypeError as exc:
        raise ValueError("periodic_axes must identify two distinct cell axes") from exc
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        for value in raw_axes
    ):
        raise ValueError("periodic_axes must identify two distinct integer cell axes")
    axes = tuple(int(value) for value in raw_axes)
    if len(axes) != 2 or len(set(axes)) != 2 or any(axis not in (0, 1, 2) for axis in axes):
        raise ValueError("periodic_axes must identify two distinct cell axes")
    return axes


def resolve_periodic_fractional_axes(
    site_payload: dict, prototypes: dict[str, dict]
) -> tuple[int, int]:
    """Resolve one formal periodic-axis tuple from all referenced prototypes."""

    if not isinstance(site_payload, dict):
        raise ValueError("Site Instance payload must be an object")
    instances = site_payload.get("site_instances")
    if not isinstance(instances, list) or not instances:
        raise ValueError("Site Instance payload is missing or empty")
    used_prototype_ids = []
    for instance in instances:
        if not isinstance(instance, dict):
            raise ValueError("Every Site Instance must be an object")
        prototype_id = instance.get("parent_site_prototype_id")
        if not isinstance(prototype_id, str) or not prototype_id:
            raise ValueError("Every Site Instance must reference a Site Prototype")
        used_prototype_ids.append(prototype_id)

    resolved_by_prototype = {}
    for prototype_id in sorted(set(used_prototype_ids)):
        prototype = prototypes.get(prototype_id)
        if not isinstance(prototype, dict):
            raise ValueError(f"Missing Site Prototype {prototype_id}")
        frame = prototype.get("surface_frame")
        if not isinstance(frame, dict):
            raise ValueError(f"Site Prototype {prototype_id} lacks a Surface Frame")
        surface_frame_coordinates(np.zeros((1, 3)), frame)
        if "periodic_fractional_axes" not in frame:
            raise ValueError(
                f"Site Prototype {prototype_id} Surface Frame lacks "
                "periodic_fractional_axes"
            )
        try:
            resolved_by_prototype[prototype_id] = _validated_surface_axes(
                frame["periodic_fractional_axes"]
            )
        except ValueError as exc:
            raise ValueError(
                f"Site Prototype {prototype_id} has invalid "
                "periodic_fractional_axes"
            ) from exc

    unique_axes = set(resolved_by_prototype.values())
    if len(unique_axes) != 1:
        details = ", ".join(
            f"{prototype_id}={list(axes)}"
            for prototype_id, axes in sorted(resolved_by_prototype.items())
        )
        raise ValueError(
            "Used Site Prototype periodic_fractional_axes are inconsistent: "
            + details
        )
    return next(iter(unique_axes))


def _validated_layer_normal_axis(groups: dict, periodic_axes) -> int:
    axes = _validated_surface_axes(periodic_axes)
    normal_axis = groups.get("normal_axis") if isinstance(groups, dict) else None
    if (
        isinstance(normal_axis, (bool, np.bool_))
        or not isinstance(normal_axis, (int, np.integer))
        or int(normal_axis) not in (0, 1, 2)
    ):
        raise ValueError("Layer groups normal_axis must be an integer chosen from 0, 1, 2")
    normal_axis = int(normal_axis)
    expected = next(axis for axis in range(3) if axis not in axes)
    if normal_axis != expected:
        raise ValueError(
            "Layer groups normal_axis must be the complement of resolved "
            "periodic_fractional_axes"
        )
    return normal_axis


def _resolved_ase_vdw_radii(symbols) -> dict[str, float]:
    """Resolve ASE radii for every actual element without a whitelist."""

    from ase.data import atomic_numbers, vdw_radii

    radii = {}
    for symbol in sorted({str(value) for value in symbols}):
        try:
            value = float(vdw_radii[atomic_numbers[symbol]])
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"ASE has no van-der-Waals radius for {symbol!r}") from exc
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"ASE has no finite positive van-der-Waals radius for {symbol}")
        radii[symbol] = value
    if not radii:
        raise ValueError("At least one actual element is required for vdW screening")
    return radii


def build_substrate_collision_contract(
    *,
    mode,
    periodic_axes,
    mapped_bond_window_A,
    hard_thresholds_A,
    substrate_symbols,
    molecule_symbols=(),
    vdw_radius_scale=0.85,
    height_clearance_A=1.0,
) -> dict:
    """Return the explicit candidate-generation substrate collision contract."""

    if mode not in {"skip", "check", "strict-vdw", "height"}:
        raise ValueError(f"Unsupported substrate collision mode: {mode!r}")
    axes = _validated_surface_axes(periodic_axes)
    minimum, maximum = _validated_registered_bond_window(mapped_bond_window_A)
    common = {
        "schema": "sam-sequential-substrate-collision-contract-v1",
        "mode": mode,
        "periodic_axes": list(axes),
        "normal_axis_wrapped": False,
        "mapped_bond_distance_window_A": [minimum, maximum],
        "mapped_pair_exemption": "exact_atom_pair_only_and_only_while_window_passes",
        "nonmapped_contacts_checked": mode != "skip",
        "element_whitelist_used": False,
    }
    if mode == "skip":
        return {
            **common,
            "algorithm_version": "legacy-unchecked-v1",
            "geometry_checked": False,
            "legacy_domain_fingerprint": True,
        }
    if mode == "check":
        return {
            **common,
            "algorithm_version": "surface-hard-distance-exact-mapped-v2",
            "geometry_checked": True,
            "hard_thresholds_A": _validate_collision_thresholds(
                hard_thresholds_A
            ),
        }
    if mode == "height":
        height_clearance_A = _cohesive_nonnegative(
            height_clearance_A, "height_clearance_A"
        )
        return {
            "schema": "sam-sequential-substrate-collision-contract-v1",
            "mode": "height",
            "algorithm_version": "surface-frame-non-anchor-height-clearance-v2",
            "periodic_axes": list(axes),
            "normal_axis_wrapped": False,
            "geometry_checked": True,
            "height_clearance_A": float(height_clearance_A),
            "surface_height_reference": "maximum_substrate_surface_frame_normal_coordinate",
            "checked_molecule_atoms": "all_except_anchor_group",
            "anchor_group_exclusion": (
                "topology_resolved_anchor_P_plus_bonded_headgroup_O"
            ),
            "mapped_pair_exemption": "none",
            "mapped_bond_distance_window_A": None,
            "pairwise_distance_or_vdw_contacts_checked": False,
            "element_whitelist_used": False,
        }
    if not isinstance(vdw_radius_scale, (int, float)) or not np.isfinite(
        vdw_radius_scale
    ) or float(vdw_radius_scale) <= 0.0:
        raise ValueError("substrate vdW radius scale must be finite and positive")
    resolved_radii = _resolved_ase_vdw_radii(
        list(substrate_symbols) + list(molecule_symbols)
    )
    return {
        **common,
        "algorithm_version": "periodic-vdw-collision-audit-v2-exact-partial-pbc",
        "geometry_checked": True,
        "radii_source": "ASE ase.data.vdw_radii",
        "resolved_vdw_radii_A": resolved_radii,
        "neighbor_search_method": (
            "ase.neighborlist.primitive_neighbor_list_per_atom_vdw_cutoffs"
        ),
        "neighbor_query_contract": "one_vectorized_query_per_candidate_audit",
        "full_pair_matrix_for_minimum_diagnostics": False,
        "minimum_diagnostics_definition": (
            "minimum_over_mapped_bonds_and_vdw_collision_neighbor_evidence_only"
        ),
        "actual_substrate_elements": sorted(
            {str(value) for value in substrate_symbols}
        ),
        "radius_scale": float(vdw_radius_scale),
    }


def _surface_only_pair_vectors(
    molecule_positions, substrate_positions, *, cell, periodic_axes
) -> np.ndarray:
    molecule = np.asarray(molecule_positions, dtype=float)
    substrate = np.asarray(substrate_positions, dtype=float)
    if molecule.ndim != 2 or molecule.shape[1] != 3:
        raise ValueError("Molecule positions must have shape (N, 3)")
    if substrate.ndim != 2 or substrate.shape[1] != 3:
        raise ValueError("Substrate positions must have shape (N, 3)")
    axes = _validated_surface_axes(periodic_axes)
    pbc = np.zeros(3, dtype=bool)
    pbc[list(axes)] = True
    vectors = molecule[:, None, :] - substrate[None, :, :]
    minimum, _ = find_mic(
        vectors.reshape(-1, 3),
        cell=np.asarray(cell, dtype=float),
        pbc=pbc,
    )
    return np.asarray(minimum, dtype=float).reshape(
        len(molecule), len(substrate), 3
    )


def _hard_substrate_collision_audit(
    *,
    molecule_positions,
    molecule_symbols,
    substrate_positions,
    substrate_symbols,
    cell,
    periodic_axes,
    hard_thresholds_A,
    registered_interface_bonds,
) -> dict:
    """Apply hard H/H-heavy/heavy thresholds with exact mapped exemptions."""

    thresholds = _validate_collision_thresholds(hard_thresholds_A)
    molecule_symbols = np.asarray(molecule_symbols, dtype=str)
    substrate_symbols = np.asarray(substrate_symbols, dtype=str)
    vectors = _surface_only_pair_vectors(
        molecule_positions,
        substrate_positions,
        cell=cell,
        periodic_axes=periodic_axes,
    )
    distances = np.linalg.norm(vectors, axis=2)
    molecule_h = molecule_symbols[:, None] == "H"
    substrate_h = substrate_symbols[None, :] == "H"
    pair_thresholds = np.where(
        molecule_h & substrate_h,
        thresholds["H-H"],
        np.where(
            molecule_h | substrate_h,
            thresholds["H-heavy"],
            thresholds["heavy-heavy"],
        ),
    )
    overlap = distances < pair_thresholds
    mapped_bonds = []
    seen_pairs = set()
    for bond in registered_interface_bonds:
        molecule_index = int(bond["molecule_atom_index_0based"])
        substrate_index = int(bond["working_metal_atom_index_0based"])
        pair = (molecule_index, substrate_index)
        if pair in seen_pairs:
            raise ValueError("Registered interface bond pairs must be unique")
        seen_pairs.add(pair)
        minimum, maximum = _validated_registered_bond_window(
            bond["distance_window_A"]
        )
        distance = float(distances[pair])
        comparison_tolerance_A = closed_distance_window_tolerance_A(
            distance,
            minimum,
            maximum,
        )
        passed = distance_within_closed_window(distance, minimum, maximum)
        if passed:
            overlap[pair] = False
        mapped_bonds.append(
            {
                "reason": "mapped_donor_metal_bond",
                "donor_label": bond["donor_label"],
                "molecule_atom_id": molecule_index,
                "molecule_element": str(molecule_symbols[molecule_index]),
                "substrate_atom_id": substrate_index,
                "substrate_element": str(substrate_symbols[substrate_index]),
                "distance_A": distance,
                "accepted_distance_window_A": [minimum, maximum],
                "comparison_tolerance_A": comparison_tolerance_A,
                "passed": passed,
            }
        )
    collisions = []
    for molecule_index, substrate_index in np.argwhere(overlap):
        collisions.append(
            {
                "reason": "hard_distance_overlap",
                "molecule_atom_id": int(molecule_index),
                "molecule_element": str(molecule_symbols[molecule_index]),
                "substrate_atom_id": int(substrate_index),
                "substrate_element": str(substrate_symbols[substrate_index]),
                "distance_A": float(distances[molecule_index, substrate_index]),
                "threshold_A": float(
                    pair_thresholds[molecule_index, substrate_index]
                ),
            }
        )
    collisions.sort(key=lambda record: (record["distance_A"], record["molecule_atom_id"], record["substrate_atom_id"]))
    violations = [record for record in mapped_bonds if not record["passed"]]
    return {
        "mode": "check",
        "passed": not collisions and not violations,
        "geometry_checked": True,
        "collision_count": len(collisions),
        "collisions": collisions,
        "mapped_bond_count": len(mapped_bonds),
        "mapped_bond_violation_count": len(violations),
        "mapped_bonds": mapped_bonds,
        "minimum_distance_A": float(np.min(distances)),
        "minimum_clearance_ratio": float(np.min(distances / pair_thresholds)),
        "minimum_diagnostics_definition": (
            "exact_minimum_over_all_molecule_substrate_pairs"
        ),
        "pair_search": {
            "method": "vectorized_ase_find_mic_full_cross_pair_matrix",
            "full_cartesian_pair_matrix_materialized": True,
        },
        "hard_thresholds_A": thresholds,
        "periodic_axes": list(_validated_surface_axes(periodic_axes)),
    }


def _height_substrate_clearance_audit(
    *,
    molecule_positions,
    molecule_symbols,
    substrate_positions,
    substrate_symbols,
    surface_frame,
    anchor_group_atom_indices_0based,
    height_clearance_A=1.0,
) -> dict:
    """Require every non-anchor SAM atom above the substrate height cutoff."""

    clearance = _cohesive_nonnegative(height_clearance_A, "height_clearance_A")
    molecule_positions = np.asarray(molecule_positions, dtype=float)
    substrate_positions = np.asarray(substrate_positions, dtype=float)
    molecule_symbols = np.asarray(molecule_symbols, dtype=str)
    substrate_symbols = np.asarray(substrate_symbols, dtype=str)
    if molecule_positions.shape != (len(molecule_symbols), 3) or not len(molecule_symbols):
        raise ValueError("Molecule positions and symbols must describe at least one atom")
    if substrate_positions.shape != (len(substrate_symbols), 3) or not len(substrate_symbols):
        raise ValueError("Substrate positions and symbols must describe at least one atom")
    raw_anchor_group = list(anchor_group_atom_indices_0based)
    if (
        not raw_anchor_group
        or any(isinstance(index, bool) or not isinstance(index, numbers.Integral) for index in raw_anchor_group)
    ):
        raise ValueError("Anchor-group atom indices must be non-empty integers")
    anchor_group = [int(index) for index in raw_anchor_group]
    if len(anchor_group) != len(set(anchor_group)):
        raise ValueError("Anchor-group atom indices must be unique")
    if min(anchor_group) < 0 or max(anchor_group) >= len(molecule_symbols):
        raise ValueError("Anchor-group atom index is outside the molecule")
    anchor_symbols = [str(molecule_symbols[index]) for index in anchor_group]
    if anchor_symbols.count("P") != 1 or not any(symbol == "O" for symbol in anchor_symbols):
        raise ValueError("Anchor group must contain one P and bonded headgroup O atoms")
    if any(symbol not in {"P", "O"} for symbol in anchor_symbols):
        raise ValueError("Anchor-group exclusion may contain only P and O atoms")
    anchor_group_set = set(anchor_group)
    checked_indices = np.asarray(
        [index for index in range(len(molecule_symbols)) if index not in anchor_group_set],
        dtype=int,
    )
    if not len(checked_indices):
        raise ValueError("Height clearance requires at least one non-anchor atom")
    molecule_normal = surface_frame_coordinates(
        molecule_positions, surface_frame
    )[:, 2]
    substrate_normal = surface_frame_coordinates(
        substrate_positions, surface_frame
    )[:, 2]
    surface_height = float(np.max(substrate_normal))
    cutoff = surface_height + clearance
    failing = checked_indices[molecule_normal[checked_indices] < cutoff]
    collisions = [
        {
            "reason": "below_surface_height_clearance",
            "molecule_atom_id": int(index),
            "molecule_element": str(molecule_symbols[index]),
            "normal_height_A": float(molecule_normal[index]),
            "surface_height_A": surface_height,
            "required_minimum_height_A": cutoff,
            "clearance_A": float(molecule_normal[index] - surface_height),
        }
        for index in failing
    ]
    return {
        "mode": "height",
        "passed": not collisions,
        "geometry_checked": True,
        "collision_count": len(collisions),
        "collisions": collisions,
        "mapped_bond_count": 0,
        "mapped_bond_violation_count": 0,
        "mapped_bonds": [],
        "height_clearance_A": float(clearance),
        "surface_height_A": surface_height,
        "required_minimum_height_A": cutoff,
        "minimum_molecule_height_A": float(np.min(molecule_normal[checked_indices])),
        "minimum_clearance_A": float(
            np.min(molecule_normal[checked_indices]) - surface_height
        ),
        "minimum_distance_A": None,
        "minimum_clearance_ratio": None,
        "minimum_diagnostics_definition": (
            "minimum_non_anchor_sam_atom_surface_frame_normal_clearance"
        ),
        "checked_molecule_atoms": "all_except_anchor_group",
        "checked_molecule_atom_count": int(len(checked_indices)),
        "excluded_anchor_group_atom_count": len(anchor_group),
        "anchor_group_atom_indices_0based": anchor_group,
        "anchor_group_elements": anchor_symbols,
        "anchor_group_exclusion": (
            "topology_resolved_anchor_P_plus_bonded_headgroup_O"
        ),
        "mapped_pair_exemption": "none",
        "mapped_bond_distance_window_A": None,
        "pairwise_distance_or_vdw_contacts_checked": False,
    }


def audit_candidate_substrate_contacts(
    *,
    mode,
    molecule_positions,
    molecule_symbols,
    substrate_positions,
    substrate_symbols,
    cell,
    periodic_axes,
    surface_frame,
    registered_interface_bonds,
    hard_thresholds_A,
    vdw_radius_scale,
    anchor_group_atom_indices_0based=None,
    height_clearance_A=1.0,
) -> dict:
    """Audit one candidate under skip, corrected hard, or strict-vdW mode."""

    if mode == "skip":
        return {
            "mode": "skip",
            "passed": True,
            "geometry_checked": False,
            "collision_count": 0,
            "collisions": [],
            "mapped_bond_count": len(registered_interface_bonds),
            "mapped_bond_violation_count": 0,
            "mapped_bonds": [],
            "minimum_distance_A": None,
            "minimum_clearance_ratio": None,
            "minimum_diagnostics_definition": "not_computed_geometry_unchecked",
        }
    if mode == "check":
        return _hard_substrate_collision_audit(
            molecule_positions=molecule_positions,
            molecule_symbols=molecule_symbols,
            substrate_positions=substrate_positions,
            substrate_symbols=substrate_symbols,
            cell=cell,
            periodic_axes=periodic_axes,
            hard_thresholds_A=hard_thresholds_A,
            registered_interface_bonds=registered_interface_bonds,
        )
    if mode == "height":
        if anchor_group_atom_indices_0based is None:
            raise ValueError("Height mode requires topology-resolved anchor-group indices")
        return _height_substrate_clearance_audit(
            molecule_positions=molecule_positions,
            molecule_symbols=molecule_symbols,
            substrate_positions=substrate_positions,
            substrate_symbols=substrate_symbols,
            surface_frame=surface_frame,
            anchor_group_atom_indices_0based=anchor_group_atom_indices_0based,
            height_clearance_A=height_clearance_A,
        )
    if mode != "strict-vdw":
        raise ValueError(f"Unsupported substrate collision mode: {mode!r}")
    radii = _resolved_ase_vdw_radii(
        list(molecule_symbols) + list(substrate_symbols)
    )
    mapped_windows = {
        (
            int(bond["molecule_atom_index_0based"]),
            int(bond["working_metal_atom_index_0based"]),
        ): tuple(float(value) for value in bond["distance_window_A"])
        for bond in registered_interface_bonds
    }
    if len(mapped_windows) != len(registered_interface_bonds):
        raise ValueError("Registered interface bond pairs must be unique")
    audit = periodic_vdw_collision_audit(
        molecule_positions=molecule_positions,
        molecule_symbols=molecule_symbols,
        molecule_atom_ids=list(range(len(molecule_symbols))),
        substrate_positions=substrate_positions,
        substrate_symbols=substrate_symbols,
        substrate_atom_ids=list(range(len(substrate_symbols))),
        cell=cell,
        periodic_axes=periodic_axes,
        surface_frame=surface_frame,
        radii_A=radii,
        radius_scale=vdw_radius_scale,
        mapped_bond_windows_A=mapped_windows,
    )
    return {
        **audit,
        "mode": "strict-vdw",
        "geometry_checked": True,
        "radii_source": "ASE ase.data.vdw_radii",
        "resolved_vdw_radii_A": radii,
        "minimum_distance_A": audit["minimum_audited_distance_A"],
        "minimum_clearance_ratio": audit[
            "minimum_audited_clearance_ratio"
        ],
    }


@dataclass
class ProjectionCache:
    """Reusable periodic projection geometry for one candidate domain.

    Templates hold the anchor-relative footprint vertices shared by every
    placement of one prototype/conformer/permutation; placements reconstruct
    the periodic Shapely geometry per candidate and store its coverage bitset
    against the stable sorted Site Instances.
    """

    site_ids: list[str]
    site_index: dict[str, int]
    candidate_ids: list[str]
    coverage_masks: dict[int, int]
    candidate_geometries: dict[int, object]
    template_records: list[dict]
    area_scale_A2: float
    surface_lattice_uv_A: np.ndarray
    contract: dict
    manifest: dict
    relative_vertices: np.ndarray = field(repr=False, default=None)
    template_index: np.ndarray = field(repr=False, default=None)
    anchor_uv: np.ndarray = field(repr=False, default=None)
    site_anchor_fractional: np.ndarray = field(repr=False, default=None)


def _package_versions() -> dict:
    return {
        name: importlib.metadata.version(name)
        for name in ("ase", "numpy", "scipy", "shapely")
    }


def _sha256_of_file(path: Path) -> str:
    return _sha256(path)


def _prototype_evidence_for(
    candidate, *, prototypes: dict[str, dict] | None
) -> str:
    prototype_id = str(candidate["site_prototype_id"])
    if prototypes is None:
        return prototype_id
    prototype = prototypes.get(prototype_id)
    if prototype is None:
        raise ValueError(f"Missing Site Prototype {prototype_id}")
    return str(prototype.get("representative_candidate_id", prototype_id))


def _template_key(
    candidate, *, prototype_evidence, surface_frame, radius_scale, boundary_samples_per_atom
) -> str:
    """Hash the template identity for one prototype/conformer/permutation.

    The key covers prototype plus evidence, conformer structure hash and rank,
    donor permutation, atom-order/symbol hash, frame hash, radius contract, and
    algorithm version; it deliberately excludes the Site Instance so translated
    placements share one relative template.
    """

    digest = hashlib.sha256()
    digest.update(b"projection-template-v1")
    frame_hash = hashlib.sha256(
        json.dumps(surface_frame, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    for field in (
        candidate["site_prototype_id"],
        prototype_evidence,
        candidate["conformer_sha256"],
        str(int(candidate["selection_rank"])),
        json.dumps(
            [str(value) for value in candidate["target_donor_labels"]],
            separators=(",", ":"),
        ),
        json.dumps(
            [int(value) for value in candidate["oxygen_permutation"]],
            separators=(",", ":"),
        ),
        json.dumps(
            [str(value) for value in candidate["symbols"]], separators=(",", ":")
        ),
        frame_hash,
        str(float(radius_scale)),
        str(int(boundary_samples_per_atom)),
        _PROJECTION_ALGORITHM_VERSION,
    ):
        digest.update(str(field).encode("utf-8", "surrogatepass"))
    return digest.hexdigest()


def _projection_contract(
    *,
    domains,
    site_payload,
    cell,
    periodic_axes,
    surface_frame,
    boundary_samples_per_atom,
    radius_scale,
    prototypes,
    substrate_collision_contract=None,
    p_surface_o_exclusion_contract=None,
    metal_coordination_contract=None,
) -> dict:
    """Return the immutable contract that must match before a cache is reused."""

    instance_by_id = {
        instance["site_instance_id"]: instance
        for instance in site_payload["site_instances"]
    }
    anchor_hashes = []
    for site_id in sorted(domains):
        instance = instance_by_id.get(site_id)
        if instance is None:
            raise ValueError(f"Missing Site Instance payload for {site_id}")
        anchor_hashes.append(
            json.dumps(
                [float(value) for value in instance["ideal_site_location"]["cartesian_A"]],
                separators=(",", ":"),
            )
        )
    anchor_digest = hashlib.sha256()
    for value in anchor_hashes:
        anchor_digest.update(value.encode())
    evidence_map = {}
    for site_id in sorted(domains):
        if not domains[site_id]:
            continue
        prototype_id = str(domains[site_id][0]["site_prototype_id"])
        evidence_map[prototype_id] = _prototype_evidence_for(
            domains[site_id][0], prototypes=prototypes
        )
    surface_lattice_uv_A, cell_area_A2 = _surface_lattice_uv(
        cell, periodic_axes, surface_frame
    )
    contract = {
        "schema": "projection-cache-v1",
        "algorithm_version": _PROJECTION_ALGORITHM_VERSION,
        "domain_fingerprint": _candidate_domain_fingerprint(
            domains,
            cell=cell,
            periodic_axes=periodic_axes,
            substrate_collision_contract=substrate_collision_contract,
            p_surface_o_exclusion_contract=p_surface_o_exclusion_contract,
            metal_coordination_contract=metal_coordination_contract,
        ),
        "site_ids": sorted(domains),
        "site_anchor_sha256": anchor_digest.hexdigest(),
        "surface_frame": surface_frame,
        "cell": np.asarray(cell, dtype=float).tolist(),
        "periodic_axes": [int(axis) for axis in periodic_axes],
        "surface_lattice_uv_A": surface_lattice_uv_A.tolist(),
        "cell_area_A2": cell_area_A2,
        "boundary_samples_per_atom": int(boundary_samples_per_atom),
        "radius_scale": float(radius_scale),
        "vdw_radii_source": "ase.data.vdw_radii",
        "prototype_evidence_map": dict(sorted(evidence_map.items())),
    }
    if (
        substrate_collision_contract is not None
        and substrate_collision_contract.get("mode") != "skip"
    ):
        contract["substrate_collision_contract"] = substrate_collision_contract
    if p_surface_o_exclusion_contract is not None:
        contract["p_surface_o_exclusion_contract"] = p_surface_o_exclusion_contract
    if (
        metal_coordination_contract is not None
        and metal_coordination_contract.get("mode") != "off"
    ):
        contract["metal_coordination_contract"] = metal_coordination_contract
    return contract


def _build_projection_cache(
    *,
    domains,
    site_payload,
    cell,
    periodic_axes,
    surface_frame,
    boundary_samples_per_atom,
    radius_scale,
    prototypes,
    contract,
) -> ProjectionCache:
    """Construct the projection cache from a candidate domain."""

    from ase.data import atomic_numbers, vdw_radii

    def radii_A_for(symbols) -> dict:
        radii = {}
        for symbol in set(symbols):
            value = float(vdw_radii[atomic_numbers[str(symbol)]])
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"No registered ASE van-der-Waals radius for {symbol}"
                )
            radii[str(symbol)] = value
        return radii

    _, Point, _, _, _ = _require_shapely()
    lattice, area_scale_A2 = _surface_lattice_uv(cell, periodic_axes, surface_frame)
    inverse = np.linalg.inv(lattice)

    site_ids = sorted(domains)
    site_index = {site_id: index for index, site_id in enumerate(site_ids)}
    instance_by_id = {
        instance["site_instance_id"]: instance
        for instance in site_payload["site_instances"]
    }
    site_anchor_fractional = np.zeros((len(site_ids), 2))
    site_anchor_uv = {}
    for site_id in site_ids:
        instance = instance_by_id.get(site_id)
        if instance is None:
            raise ValueError(f"Missing Site Instance payload for {site_id}")
        cartesian = np.asarray(
            instance["ideal_site_location"]["cartesian_A"], dtype=float
        )
        uv = surface_frame_coordinates(cartesian[None, :], surface_frame)[0, :2]
        site_anchor_uv[site_id] = uv
        site_anchor_fractional[site_index[site_id]] = inverse @ uv

    all_candidates = sorted(
        (candidate for candidates in domains.values() for candidate in candidates),
        key=lambda item: item["candidate_id"],
    )
    candidate_ids = [candidate["candidate_id"] for candidate in all_candidates]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Duplicate candidate IDs in candidate domain")

    templates: dict[str, dict] = {}
    template_index_by_key: dict[str, int] = {}
    candidate_geometries: dict[int, object] = {}
    coverage_masks: dict[int, int] = {}
    anchor_uv_by_candidate = np.zeros((len(all_candidates), 2))
    template_index_by_candidate = np.zeros(len(all_candidates), dtype=np.int64)

    for position, candidate in enumerate(all_candidates):
        prototype_evidence = _prototype_evidence_for(candidate, prototypes=prototypes)
        key = _template_key(
            candidate,
            prototype_evidence=prototype_evidence,
            surface_frame=surface_frame,
            radius_scale=radius_scale,
            boundary_samples_per_atom=boundary_samples_per_atom,
        )
        own_anchor_uv = site_anchor_uv[candidate["site_instance_id"]]
        anchor_uv_by_candidate[position] = own_anchor_uv
        if key in template_index_by_key:
            template_index = template_index_by_key[key]
            templates[key]["candidate_indices"].append(position)
        else:
            footprint = filled_outer_envelope_area(
                positions=candidate["coordinates"],
                symbols=candidate["symbols"],
                surface_frame=surface_frame,
                radii_A=radii_A_for(candidate["symbols"]),
                radius_scale=radius_scale,
                boundary_samples_per_atom=boundary_samples_per_atom,
            )
            relative = (
                np.asarray(footprint["hull_vertices_uv_A"], dtype=float)
                - own_anchor_uv
            )
            template_index = len(templates)
            templates[key] = {
                "template_index": template_index,
                "template_key": key,
                "site_prototype_id": candidate["site_prototype_id"],
                "prototype_evidence": prototype_evidence,
                "conformer_sha256": candidate["conformer_sha256"],
                "selection_rank": int(candidate["selection_rank"]),
                "oxygen_permutation": [
                    int(value) for value in candidate["oxygen_permutation"]
                ],
                "relative_vertices_uv_A": relative.tolist(),
                "candidate_indices": [position],
            }
            template_index_by_key[key] = template_index
        template_index_by_candidate[position] = template_index
        vertices_uv_A = (
            np.asarray(templates[key]["relative_vertices_uv_A"], dtype=float)
            + own_anchor_uv
        )
        geometry, scale = periodic_surface_polygon(
            vertices_uv_A,
            cell=cell,
            periodic_axes=periodic_axes,
            surface_frame=surface_frame,
        )
        if abs(scale - area_scale_A2) > 1.0e-9:
            raise ValueError("Inconsistent surface area scale")
        candidate_geometries[position] = geometry
        coverage_masks[position] = periodic_polygon_coverage_mask(
            geometry, site_anchor_fractional
        )

    template_records = []
    for key in sorted(templates):
        record = templates[key]
        audit_index = min(record["candidate_indices"])
        audit_candidate = all_candidates[audit_index]
        audit_footprint = filled_outer_envelope_area(
            positions=audit_candidate["coordinates"],
            symbols=audit_candidate["symbols"],
            surface_frame=surface_frame,
            radii_A=radii_A_for(audit_candidate["symbols"]),
            radius_scale=radius_scale,
            boundary_samples_per_atom=boundary_samples_per_atom,
        )
        own_uv = site_anchor_uv[audit_candidate["site_instance_id"]]
        audited_relative = (
            np.asarray(audit_footprint["hull_vertices_uv_A"], dtype=float) - own_uv
        )
        geometry_consistent = np.allclose(
            audited_relative,
            np.asarray(record["relative_vertices_uv_A"], dtype=float),
            atol=1.0e-6,
            rtol=0.0,
        )
        geometry = candidate_geometries[audit_index]
        direct = 0
        for site_position, (u, v) in enumerate(site_anchor_fractional):
            if geometry.covers(Point(float(u), float(v))):
                direct |= 1 << site_position
        record["audited_candidate_index"] = audit_index
        record["audit_passed"] = geometry_consistent and (
            direct == coverage_masks[audit_index]
        )
        template_records.append(record)
    template_records.sort(key=lambda item: item["template_index"])
    max_vertices = max(
        len(record["relative_vertices_uv_A"]) for record in template_records
    )
    padded_relative_vertices = np.full(
        (len(template_records), max_vertices, 2), np.nan
    )
    for record in template_records:
        vertices = np.asarray(record["relative_vertices_uv_A"], dtype=float)
        padded_relative_vertices[record["template_index"], : len(vertices)] = vertices

    manifest = {
        "hit": False,
        "rebuild_reason": "first build or contract mismatch",
        "contract": contract,
        "site_ids": site_ids,
        "candidate_ids": candidate_ids,
        "site_count": len(site_ids),
        "candidate_count": len(candidate_ids),
        "template_count": len(template_records),
        "site_anchor_fractional": site_anchor_fractional.tolist(),
        "template_records": [
            {
                key: value
                for key, value in record.items()
                if key != "candidate_indices"
            }
            for record in template_records
        ],
    }
    return ProjectionCache(
        site_ids=site_ids,
        site_index=site_index,
        candidate_ids=candidate_ids,
        coverage_masks=coverage_masks,
        candidate_geometries=candidate_geometries,
        template_records=template_records,
        area_scale_A2=area_scale_A2,
        surface_lattice_uv_A=lattice,
        contract=contract,
        manifest=manifest,
        relative_vertices=padded_relative_vertices,
        template_index=template_index_by_candidate,
        anchor_uv=anchor_uv_by_candidate,
        site_anchor_fractional=site_anchor_fractional,
    )


def _write_projection_cache_atomically(
    cache: ProjectionCache,
    *,
    cache_dir: Path,
    immutable_new_cache: bool = False,
) -> dict:
    """Persist projection data; formal mode never replaces an existing byte."""

    cache_dir = Path(cache_dir)
    if immutable_new_cache:
        if os.path.lexists(os.fspath(cache_dir)):
            raise FileExistsError(
                f"Immutable projection cache directory must be absent: {cache_dir}"
            )
        if cache_dir.parent.is_symlink() or not cache_dir.parent.is_dir():
            raise FileNotFoundError(
                "Immutable projection cache parent must be a real directory: "
                f"{cache_dir.parent}"
            )
        cache_dir.mkdir(exist_ok=False)
        _fsync_directory(cache_dir.parent)
        data_reference = _atomic_install_noclobber_generated_file(
            cache_dir / "projection-cache-data.npz",
            lambda handle: np.savez(
                handle,
                relative_vertices=cache.relative_vertices,
                template_index=cache.template_index,
                anchor_uv=cache.anchor_uv,
                site_anchor_fractional=cache.site_anchor_fractional,
            ),
            temporary_prefix=".projection-data-",
        )
        data_sha256 = data_reference["sha256"]
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)
        fd, temp_data = tempfile.mkstemp(
            dir=cache_dir, prefix=".projection-data-", suffix=".npz"
        )
        os.close(fd)
        try:
            np.savez(
                temp_data,
                relative_vertices=cache.relative_vertices,
                template_index=cache.template_index,
                anchor_uv=cache.anchor_uv,
                site_anchor_fractional=cache.site_anchor_fractional,
            )
            data_sha256 = _sha256_of_file(Path(temp_data))
            os.replace(temp_data, cache_dir / "projection-cache-data.npz")
        finally:
            if os.path.exists(temp_data):
                os.remove(temp_data)

    manifest = {
        **cache.manifest,
        "data_sha256": data_sha256,
        "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "package_versions": _package_versions(),
    }
    if immutable_new_cache:
        manifest["cache_lifecycle"] = {
            "immutable_new_cache": True,
            "cache_hit": False,
            "prior_reuse": False,
            "destination_clobber": False,
            "installation": (
                "same_directory_hard_link_noreplace_after_file_fsync"
            ),
        }
        write_strict_json_noclobber(
            cache_dir / "projection-cache-manifest.json", manifest
        )
    else:
        fd, temp_manifest = tempfile.mkstemp(
            dir=cache_dir, prefix=".projection-manifest-", suffix=".json"
        )
        os.close(fd)
        try:
            Path(temp_manifest).write_text(
                json.dumps(manifest, indent=2, allow_nan=False)
            )
            os.replace(temp_manifest, cache_dir / "projection-cache-manifest.json")
        finally:
            if os.path.exists(temp_manifest):
                os.remove(temp_manifest)
    cache.manifest = manifest
    return manifest


def _load_projection_cache(
    *, cache_dir: Path, contract: dict
) -> tuple[dict | None, str | None]:
    """Load a matching cache manifest or report miss/corruption."""

    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / "projection-cache-manifest.json"
    data_path = cache_dir / "projection-cache-data.npz"
    if not manifest_path.exists() or not data_path.exists():
        return None, "missing cache files"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("contract") != contract:
        return None, "contract mismatch"
    data_sha256 = _sha256_of_file(data_path)
    if manifest.get("data_sha256") != data_sha256:
        return "corrupt", "data hash mismatch"
    return manifest, None


def _load_or_build_projection_cache(
    domains,
    *,
    site_payload,
    cell,
    periodic_axes,
    cache_dir,
    boundary_samples_per_atom=720,
    radius_scale=1.0,
    prototypes=None,
    substrate_collision_contract=None,
    p_surface_o_exclusion_contract=None,
    metal_coordination_contract=None,
    immutable_new_cache: bool = False,
) -> ProjectionCache:
    """Load a matching projection cache or build and persist it atomically.

    A mismatched or missing cache is rebuilt with its reason recorded; a
    corrupt data hash is a hard failure with a ``failure.json`` retained and
    never silently repaired.
    """

    cache_dir = Path(cache_dir)
    if immutable_new_cache and os.path.lexists(os.fspath(cache_dir)):
        raise FileExistsError(
            f"Immutable projection cache directory must be absent: {cache_dir}"
        )
    surface_frame = _common_surface_frame(domains)
    contract = _projection_contract(
        domains=domains,
        site_payload=site_payload,
        cell=cell,
        periodic_axes=periodic_axes,
        surface_frame=surface_frame,
        boundary_samples_per_atom=boundary_samples_per_atom,
        radius_scale=radius_scale,
        prototypes=prototypes,
        substrate_collision_contract=substrate_collision_contract,
        p_surface_o_exclusion_contract=p_surface_o_exclusion_contract,
        metal_coordination_contract=metal_coordination_contract,
    )
    if immutable_new_cache:
        manifest, status = None, "immutable new cache requires no prior reuse"
    else:
        manifest, status = _load_projection_cache(
            cache_dir=cache_dir, contract=contract
        )
    if status == "data hash mismatch":
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "failure.json").write_text(
            json.dumps(
                {
                    "schema": "projection-cache-failure-v1",
                    "reason": "data hash mismatch",
                    "contract": contract,
                    "written_at_utc": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                    ),
                },
                indent=2,
                allow_nan=False,
            )
        )
        raise RuntimeError(
            "projection cache data hash mismatch; refusing partial repair"
        )
    if manifest is not None:
        data = np.load(cache_dir / "projection-cache-data.npz")
        try:
            relative_vertices = data["relative_vertices"]
            template_index = data["template_index"]
            anchor_uv = data["anchor_uv"]
            site_anchor_fractional = data["site_anchor_fractional"]
        finally:
            data.close()
        template_records = manifest.get("template_records", [])
        vertex_counts = [
            len(record["relative_vertices_uv_A"]) for record in template_records
        ]
        candidate_geometries = {}
        coverage_masks = {}
        for position in range(len(template_index)):
            template_position = int(template_index[position])
            vertices_uv_A = (
                relative_vertices[template_position, : vertex_counts[template_position]]
                + anchor_uv[position]
            )
            geometry, scale = periodic_surface_polygon(
                vertices_uv_A,
                cell=cell,
                periodic_axes=periodic_axes,
                surface_frame=surface_frame,
            )
            candidate_geometries[position] = geometry
            coverage_masks[position] = periodic_polygon_coverage_mask(
                geometry, site_anchor_fractional
            )
        surface_lattice_uv_A, area_scale_A2 = _surface_lattice_uv(
            cell, periodic_axes, surface_frame
        )
        cache = ProjectionCache(
            site_ids=manifest["site_ids"],
            site_index={
                site_id: index for index, site_id in enumerate(manifest["site_ids"])
            },
            candidate_ids=manifest["candidate_ids"],
            coverage_masks=coverage_masks,
            candidate_geometries=candidate_geometries,
            template_records=template_records,
            area_scale_A2=area_scale_A2,
            surface_lattice_uv_A=surface_lattice_uv_A,
            contract=contract,
            manifest={**manifest, "hit": True, "rebuild_reason": None},
        )
        return cache

    cache = _build_projection_cache(
        domains=domains,
        site_payload=site_payload,
        cell=cell,
        periodic_axes=periodic_axes,
        surface_frame=surface_frame,
        boundary_samples_per_atom=boundary_samples_per_atom,
        radius_scale=radius_scale,
        prototypes=prototypes,
        contract=contract,
    )
    cache.manifest["rebuild_reason"] = status or "first build"
    _write_projection_cache_atomically(
        cache,
        cache_dir=cache_dir,
        immutable_new_cache=immutable_new_cache,
    )
    return cache



def _reject_flipped_orientation(
    transformed, head_indices, clearance_A=0.8, *, surface_frame=None
) -> bool:
    """Reject placements whose molecular body points through the surface.

    A supplied Site Instance/Prototype Surface Frame defines the outward
    normal.  Omitting it deliberately preserves the historical global-``+z``
    public helper behavior for existing callers.
    """

    transformed = np.asarray(transformed, dtype=float)
    if transformed.ndim != 2 or transformed.shape[1] != 3:
        raise ValueError("transformed coordinates must have shape (N, 3)")
    if not np.all(np.isfinite(transformed)):
        raise ValueError("transformed coordinates must be finite")
    if not np.isfinite(clearance_A) or float(clearance_A) < 0.0:
        raise ValueError("orientation clearance must be finite and non-negative")
    if surface_frame is None:
        normal = np.asarray([0.0, 0.0, 1.0])
    else:
        surface_frame_coordinates(np.zeros((1, 3)), surface_frame)
        normal = np.asarray(
            surface_frame["outward_normal_cartesian_unit"], dtype=float
        )
    ordered_head = tuple(int(index) for index in head_indices)
    if not ordered_head:
        raise ValueError("head-group indices must include an anchor")
    head = set(ordered_head)
    if any(index < 0 or index >= len(transformed) for index in head):
        raise ValueError("head-group index is outside transformed coordinates")
    normal_coordinates = transformed @ normal
    # Index zero is retained only on the legacy no-frame path.  A registered
    # placement uses the first fitted head index, which is the explicit anchor.
    anchor_index = 0 if surface_frame is None else ordered_head[0]
    anchor_normal = float(normal_coordinates[anchor_index])
    body = [
        float(normal_coordinates[index])
        for index in range(len(transformed))
        if index not in head
    ]
    if not body:
        return False
    return min(body) < anchor_normal - float(clearance_A)


def _build_p_surface_o_prefilter(
    *,
    substrate_positions,
    substrate_symbols,
    cell,
    p_surface_o_exclusion_contract: dict,
) -> dict:
    """Build a conservative periodic neighbor prefilter for candidate P atoms."""

    from scipy.spatial import cKDTree

    substrate_positions = np.asarray(substrate_positions, dtype=float)
    substrate_symbols = np.asarray(substrate_symbols, dtype=str)
    axes = _validated_surface_axes(p_surface_o_exclusion_contract["surface_axes"])
    radius = float(p_surface_o_exclusion_contract["released_cutoff_A"])
    inverse_cell = np.linalg.inv(np.asarray(cell, dtype=float))
    oxygen_indices = np.asarray(np.flatnonzero(substrate_symbols == "O"), dtype=int)
    if not len(oxygen_indices):
        return {"tree": None, "oxygen_indices": oxygen_indices, "radius_A": radius}
    bounds = {
        axis: int(np.ceil(radius * np.linalg.norm(inverse_cell[:, axis]))) + 2
        for axis in axes
    }
    image_vectors = []
    image_ids = []
    for first in range(-bounds[axes[0]], bounds[axes[0]] + 1):
        for second in range(-bounds[axes[1]], bounds[axes[1]] + 1):
            image = np.zeros(3, dtype=int)
            image[axes[0]] = first
            image[axes[1]] = second
            image_vectors.append(image @ np.asarray(cell, dtype=float))
            image_ids.append(image)
    image_vectors = np.asarray(image_vectors, dtype=float)
    image_ids = np.asarray(image_ids, dtype=int)
    image_positions = (
        substrate_positions[oxygen_indices, None, :]
        + image_vectors[None, :, :]
    ).reshape((-1, 3))
    repeated_oxygen_indices = np.repeat(oxygen_indices, len(image_vectors))
    return {
        "tree": cKDTree(image_positions),
        "oxygen_indices": oxygen_indices,
        "repeated_oxygen_indices": repeated_oxygen_indices,
        "radius_A": radius,
        "image_count": len(image_vectors),
    }


def audit_p_surface_o_candidate(
    *,
    molecule_positions,
    molecule_symbols,
    p_molecule_local_index_0based: int,
    substrate_positions,
    substrate_symbols,
    full_to_working_atom_map: dict[int, dict],
    cell,
    surface_frame: dict,
    p_surface_o_exclusion_contract: dict,
    p_surface_o_prefilter: dict | None = None,
) -> dict:
    """Audit only P--O contacts for one transformed candidate placement.

    This is intentionally separate from ``audit_candidate_substrate_contacts``:
    in the experiment every non-P--working-O SAM/substrate pair is skipped, not
    accepted by an exemption and not rejected by a broad collision rule.
    """

    contract = p_surface_o_exclusion_contract
    if not isinstance(contract, dict) or contract.get("schema") != _P_SURFACE_O_EXCLUSION_SCHEMA:
        raise ValueError("P--surface-O candidate audit requires a normalized experiment contract")
    axes = _validated_surface_axes(contract.get("surface_axes"))
    if int(contract.get("normal_axis", -1)) != next(axis for axis in range(3) if axis not in axes):
        raise ValueError("P--surface-O candidate audit normal axis is not the complement")
    if contract.get("normal_wrapped") is not False:
        raise ValueError("P--surface-O candidate audit must not wrap the normal axis")
    surface_frame_coordinates(np.zeros((1, 3)), surface_frame)
    frame_axes = surface_frame.get("periodic_fractional_axes")
    if _validated_surface_axes(frame_axes) != axes:
        raise ValueError("P--surface-O candidate Surface Frame axes disagree with the contract")
    cutoff = float(contract["contact_upper_bound_A"])
    if not np.isfinite(cutoff) or cutoff <= 0.0 or contract.get("contact_upper_bound_inclusive") is not True:
        raise ValueError("P--surface-O candidate audit has an invalid inclusive cutoff")

    molecule_positions = np.asarray(molecule_positions, dtype=float)
    substrate_positions = np.asarray(substrate_positions, dtype=float)
    molecule_symbols = np.asarray(molecule_symbols, dtype=str)
    substrate_symbols = np.asarray(substrate_symbols, dtype=str)
    if molecule_positions.shape != (len(molecule_symbols), 3):
        raise ValueError("P--surface-O molecule positions and symbols do not align")
    if substrate_positions.shape != (len(substrate_symbols), 3):
        raise ValueError("P--surface-O substrate positions and symbols do not align")
    p_local = int(p_molecule_local_index_0based)
    if not 0 <= p_local < len(molecule_symbols) or molecule_symbols[p_local] != "P":
        raise ValueError("P--surface-O candidate P identity is not a local P atom")
    p_indices = np.flatnonzero(molecule_symbols == "P")
    if len(p_indices) != 1 or int(p_indices[0]) != p_local:
        raise ValueError("P--surface-O candidate requires one unique topology-validated P anchor")

    inverse_full_map = {}
    for full_id, record in full_to_working_atom_map.items():
        if not isinstance(record, dict):
            raise ValueError("Full-to-working atom map contains a malformed record")
        working_index = int(record["working_atom_index_0based"])
        if working_index in inverse_full_map:
            raise ValueError("Full-to-working atom map has duplicate working indices")
        inverse_full_map[working_index] = {
            "full_atom_id_1based": int(full_id),
            "actual_element": str(record["actual_element"]),
        }
    if p_surface_o_prefilter is None:
        oxygen_indices = [int(value) for value in np.flatnonzero(substrate_symbols == "O")]
    else:
        tree = p_surface_o_prefilter.get("tree")
        if tree is None:
            oxygen_indices = []
        else:
            nearby = tree.query_ball_point(
                molecule_positions[p_local],
                r=float(p_surface_o_prefilter["radius_A"]) + 1.0e-10,
            )
            oxygen_indices = sorted(
                {int(p_surface_o_prefilter["repeated_oxygen_indices"][index]) for index in nearby}
            )
    raw_vectors = molecule_positions[p_local][None, :] - substrate_positions[oxygen_indices]
    pbc = np.zeros(3, dtype=bool)
    pbc[list(axes)] = True
    mic_vectors, distances = find_mic(raw_vectors, cell=np.asarray(cell, dtype=float), pbc=pbc)
    mic_vectors = np.asarray(mic_vectors, dtype=float).reshape((-1, 3))
    distances = np.asarray(distances, dtype=float).reshape(-1)
    if not np.all(np.isfinite(mic_vectors)) or not np.all(np.isfinite(distances)):
        raise RuntimeError("P--surface-O partial MIC returned non-finite evidence")
    inverse_cell = np.linalg.inv(np.asarray(cell, dtype=float))
    image_fractions = (raw_vectors - mic_vectors) @ inverse_cell
    images = np.rint(image_fractions).astype(int)
    if not np.allclose(image_fractions, images, atol=1.0e-8, rtol=0.0):
        raise RuntimeError("P--surface-O MIC image cannot be reconstructed")
    nonperiodic = [axis for axis in range(3) if axis not in axes]
    if np.any(images[:, nonperiodic] != 0):
        raise RuntimeError("P--surface-O MIC wrapped the normal axis")
    if not np.allclose(raw_vectors - images @ np.asarray(cell, dtype=float), mic_vectors, atol=1.0e-8, rtol=0.0):
        raise RuntimeError("P--surface-O recorded periodic image does not reconstruct MIC")

    pair_records = []
    for position, (working_index, vector, distance, image) in enumerate(
        zip(oxygen_indices, mic_vectors, distances, images)
    ):
        mapping = inverse_full_map.get(working_index)
        if mapping is None:
            raise ValueError(f"Working surface-O index {working_index} lacks a full-substrate ID")
        if mapping["actual_element"] != "O":
            raise ValueError("Full-to-working map element disagrees with working surface O")
        local = surface_frame_coordinates(np.asarray(vector, dtype=float)[None, :], surface_frame)[0]
        pair_records.append(
            {
                "p_molecule_local_atom_id_0based": p_local,
                "p_element": "P",
                "p_global_atom_index_0based_after_placement": len(substrate_symbols) + p_local,
                "surface_o_working_local_atom_id_0based": working_index,
                "surface_o_full_atom_id_1based": mapping["full_atom_id_1based"],
                "surface_o_element": "O",
                "partial_mic_vector_A": np.asarray(vector, dtype=float).tolist(),
                "surface_frame_vector_uvn_A": np.asarray(local, dtype=float).tolist(),
                "distance_A": float(distance),
                "cutoff_A": cutoff,
                "inclusive_cutoff": True,
                "periodic_image": [int(value) for value in image],
                "substrate_periodic_image": [int(value) for value in image],
                "surface_axes": list(axes),
                "normal_axis": int(contract["normal_axis"]),
                "normal_wrapped": False,
                "reason": "P-to-any-working-surface-O-distance-at-or-below-contract-upper-bound",
            }
        )
    pair_records.sort(key=lambda record: (record["distance_A"], record["surface_o_working_local_atom_id_0based"]))
    forbidden = [record for record in pair_records if record["distance_A"] <= cutoff]
    minimum_distance = float(np.min(distances)) if len(distances) else None
    sensitivity_counts = {
        f"<={threshold:g}": int(np.count_nonzero(distances <= threshold))
        for threshold in _P_SURFACE_O_SENSITIVITY_THRESHOLDS_A
    }
    return {
        "schema": "sam-sequential-p-surface-o-placement-audit-v1",
        "p_molecule_local_atom_id_0based": p_local,
        "p_element": "P",
        "surface_o_count": len(pair_records),
        "pair_count": len(pair_records),
        "forbidden_pair_count": len(forbidden),
        "forbidden": bool(forbidden),
        "passed": not forbidden,
        "contact_upper_bound_A": cutoff,
        "inclusive_cutoff": True,
        "surface_axes": list(axes),
        "normal_axis": int(contract["normal_axis"]),
        "normal_wrapped": False,
        "minimum_distance_A": minimum_distance,
        "sensitivity_counts": sensitivity_counts,
        "forbidden_pairs": forbidden,
        "all_surface_o_pairs": pair_records,
        "other_sam_substrate_pairs_checked": False,
        "other_sam_substrate_pairs_rejected": False,
    }


def _p_surface_o_stat_record() -> dict:
    return {
        "candidate_placements_before_exclusion": 0,
        "excluded_placement_count": 0,
        "retained_placement_count": 0,
        "excluded_pair_count": 0,
        "sensitivity_counts": {
            f"<={threshold:g}": 0 for threshold in _P_SURFACE_O_SENSITIVITY_THRESHOLDS_A
        },
    }


def _update_p_surface_o_stats(record: dict, audit: dict, excluded: bool) -> None:
    record["candidate_placements_before_exclusion"] += 1
    if excluded:
        record["excluded_placement_count"] += 1
        record["excluded_pair_count"] += int(audit["forbidden_pair_count"])
    else:
        record["retained_placement_count"] += 1
    for key, value in audit["sensitivity_counts"].items():
        record["sensitivity_counts"][key] += int(bool(value))


def _p_surface_o_sensitivity_bins(contract: dict, minimum_distances: list[float]) -> dict:
    upper = float(contract["contact_upper_bound_A"])
    released = float(contract["released_cutoff_A"])
    bins = {
        "<=1.8": 0,
        "(1.8,1.9]": 0,
        "(1.9,contract_upper_bound]": 0,
        "(contract_upper_bound,released_cutoff]": 0,
        ">released_cutoff": 0,
    }
    for value in minimum_distances:
        if value <= 1.8:
            bins["<=1.8"] += 1
        elif value <= 1.9:
            bins["(1.8,1.9]"] += 1
        elif value <= upper:
            bins["(1.9,contract_upper_bound]"] += 1
        elif value <= released:
            bins["(contract_upper_bound,released_cutoff]"] += 1
        else:
            bins[">released_cutoff"] += 1
    return bins


def _build_single_metal_context(
    *,
    single_metal_sites_mode: str,
    substrate_coordinates: np.ndarray,
    substrate_symbols: np.ndarray,
    substrate_cell: np.ndarray,
    periodic_axes: tuple[int, ...] | list[int],
    normal_axis: int,
    retained_site_instances: list[dict],
    full_to_working_atom_map: dict[int, dict] | None,
    surface_frame: dict | None,
    conformers: list[dict],
    metrics_by_rank: dict[int, dict],
    substrate_height_clearance_A: float = 1.0,
    rotation_step_deg: float = 15.0,
    translation_offsets_A: tuple[float, ...] = (0.0, -0.2, 0.2),
    metal_coordination_contract: dict | None = None,
) -> dict | None:
    if single_metal_sites_mode != "dynamic-cn5-uncovered":
        return None
    cutoff_val = float(metal_coordination_contract.get("cutoff_A", 2.7)) if metal_coordination_contract else 2.7
    target_elems = tuple(metal_coordination_contract.get("target_elements", ("In", "Sn"))) if metal_coordination_contract else ("In", "Sn")
    ligand_elems = tuple(metal_coordination_contract.get("ligand_elements", ("O",))) if metal_coordination_contract else ("O",)
    uncovered_metals = _discover_uncovered_cn5_metals(
        substrate_positions=substrate_coordinates,
        substrate_symbols=substrate_symbols,
        cell=substrate_cell,
        periodic_axes=periodic_axes,
        normal_axis=normal_axis,
        retained_site_instances=retained_site_instances,
        full_to_working_atom_map=full_to_working_atom_map,
        cutoff_A=cutoff_val,
        top_surface_depth_A=0.7,
        target_elements=target_elems,
        ligand_elements=ligand_elems,
    )
    return {
        "mode": single_metal_sites_mode,
        "uncovered_metals": uncovered_metals,
        "substrate_positions": np.asarray(substrate_coordinates, dtype=float),
        "substrate_symbols": np.asarray(substrate_symbols, dtype=str),
        "cell": np.asarray(substrate_cell, dtype=float),
        "periodic_axes": tuple(periodic_axes),
        "surface_frame": surface_frame,
        "conformers": conformers,
        "metrics_by_rank": metrics_by_rank,
        "substrate_height_clearance_A": float(substrate_height_clearance_A),
        "rotation_step_deg": float(rotation_step_deg),
        "translation_offsets_A": tuple(float(v) for v in translation_offsets_A),
    }


def _candidate_domains(
    *,
    substrate,
    site_payload: dict,
    prototypes: dict[str, dict],
    conformers: list[dict],
    metrics_by_rank: dict[int, dict],
    spec: MoleculeSpec,
    headgroup_rmsd_max_A: float,
    full_to_working_atom_map: dict[int, dict],
    mapped_bond_window_A=(1.7, 2.8),
    substrate_vdw_radius_scale=0.85,
    substrate_height_clearance_A=1.0,
    periodic_axes=(0, 1),
    collision_thresholds_A=None,
    substrate_collision_mode="check",
    substrate_collision_check=None,
    p_surface_o_exclusion_contract=None,
    metal_coordination_contract=None,
    single_metal_sites_mode="off",
    single_metal_rotation_step_deg=30.0,
    single_metal_translation_offsets_A=(-1.0, -0.5, 0.0, 0.5, 1.0),
) -> tuple[dict[str, list[dict]], dict]:
    """Build candidate domains behind the strict registered-interface gate."""

    if substrate_collision_check is not None:
        compatibility_mode = "check" if substrate_collision_check else "skip"
        if substrate_collision_mode != "check" and substrate_collision_mode != compatibility_mode:
            raise ValueError("Conflicting substrate collision mode/check arguments")
        substrate_collision_mode = compatibility_mode
    if not isinstance(full_to_working_atom_map, dict) or not full_to_working_atom_map:
        raise ValueError("A non-empty full-to-working atom mapping is required")
    axes = _validated_surface_axes(periodic_axes)
    substrate_coordinates = np.asarray(substrate.positions, dtype=float)
    substrate_symbols = np.asarray(substrate.get_chemical_symbols(), dtype=str)
    thresholds_A = _validate_collision_thresholds(collision_thresholds_A)
    instances = site_payload.get("site_instances")
    if not isinstance(instances, list) or not instances:
        raise ValueError("Site Instance payload is missing or empty")
    site_ids = [instance.get("site_instance_id") for instance in instances]
    if any(not isinstance(site_id, str) or not site_id for site_id in site_ids):
        raise ValueError("Every Site Instance requires a non-empty unique ID")
    if len(set(site_ids)) != len(site_ids):
        raise ValueError("Site Instance IDs must be unique")
    if not conformers:
        raise ValueError("At least one conformer is required")
    molecule_elements = sorted(
        {
            str(symbol)
            for conformer in conformers
            for symbol in conformer.get("symbols", [])
        }
    )
    if p_surface_o_exclusion_contract is not None:
        if substrate_collision_mode != "skip":
            raise ValueError("P--surface-O candidate exclusion requires --substrate-collision skip")
        if spec.anchor_element != "P" or int(spec.formula.get("P", 0)) != 1:
            raise ValueError("P--surface-O candidate exclusion requires one P anchor in the SAM formula")
        if p_surface_o_exclusion_contract.get("schema") != _P_SURFACE_O_EXCLUSION_SCHEMA:
            raise ValueError("P--surface-O candidate exclusion contract is not normalized")
        if _validated_surface_axes(p_surface_o_exclusion_contract.get("surface_axes")) != axes:
            raise ValueError("P--surface-O candidate exclusion axes disagree with the Site Prototype")
    substrate_contract = build_substrate_collision_contract(
        mode=substrate_collision_mode,
        periodic_axes=axes,
        mapped_bond_window_A=mapped_bond_window_A,
        hard_thresholds_A=thresholds_A,
        substrate_symbols=substrate_symbols,
        molecule_symbols=molecule_elements,
        vdw_radius_scale=substrate_vdw_radius_scale,
        height_clearance_A=substrate_height_clearance_A,
    )
    p_surface_o_prefilter = (
        _build_p_surface_o_prefilter(
            substrate_positions=substrate_coordinates,
            substrate_symbols=substrate_symbols,
            cell=substrate.cell,
            p_surface_o_exclusion_contract=p_surface_o_exclusion_contract,
        )
        if p_surface_o_exclusion_contract is not None
        else None
    )

    metal_coordination_active = (
        metal_coordination_contract is not None
        and metal_coordination_contract.get("mode") == "exclude-six-coordinated"
    )
    metal_coordination_data = None
    metal_coord_stats = None
    site_instance_stats = None
    if metal_coordination_active:
        m_norm_ax = int(metal_coordination_contract.get("normal_axis", 2))
        m_cutoff = float(metal_coordination_contract.get("cutoff_A", 2.7))
        m_target = metal_coordination_contract.get("target_elements", ("In", "Sn"))
        m_ligand = metal_coordination_contract.get("ligand_elements", ("O",))
        metal_coordination_data = calculate_substrate_metal_coordinations(
            substrate_positions=substrate_coordinates,
            substrate_symbols=substrate_symbols,
            cell=substrate.cell,
            periodic_axes=axes,
            normal_axis=m_norm_ax,
            cutoff_A=m_cutoff,
            target_elements=m_target,
            ligand_elements=m_ligand,
            full_to_working_atom_map=full_to_working_atom_map,
        )
        site_instance_stats = {
            "total_site_instances_before_filter": 0,
            "excluded_site_instances": 0,
            "retained_site_instances": 0,
            "by_element": {
                "In_only": {"before": 0, "excluded": 0, "retained": 0},
                "Sn_only": {"before": 0, "excluded": 0, "retained": 0},
                "mixed_In_Sn": {"before": 0, "excluded": 0, "retained": 0},
            },
            "by_denticity": {
                "2": {"before": 0, "excluded": 0, "retained": 0},
                "3": {"before": 0, "excluded": 0, "retained": 0},
            },
            "by_mapped_max_coordination": defaultdict(lambda: {"before": 0, "excluded": 0, "retained": 0}),
            "by_site_prototype": defaultdict(lambda: {"before": 0, "excluded": 0, "retained": 0}),
            "excluded_site_instance_ids": [],
            "retained_site_instance_ids": [],
            "by_site_instance": {},
        }
        metal_coord_stats = {
            "candidate_count_before_exclusion": 0,
            "excluded_candidate_count": 0,
            "retained_candidate_count": 0,
            "by_element": {
                "In_only": {"before": 0, "excluded": 0, "retained": 0},
                "Sn_only": {"before": 0, "excluded": 0, "retained": 0},
                "mixed_In_Sn": {"before": 0, "excluded": 0, "retained": 0},
            },
            "by_denticity": {
                "2": {"before": 0, "excluded": 0, "retained": 0},
                "3": {"before": 0, "excluded": 0, "retained": 0},
            },
            "by_mapped_max_coordination": defaultdict(lambda: {"before": 0, "excluded": 0, "retained": 0}),
            "by_site_prototype": defaultdict(lambda: {"before": 0, "excluded": 0, "retained": 0}),
            "by_site_instance": defaultdict(lambda: {"before": 0, "excluded": 0, "retained": 0, "denticity": 0, "mapped_metals": [], "element_category": ""}),
            "representative_excluded_placements": [],
        }

    domains = {}
    rejection_counts = Counter()
    per_prototype = defaultdict(lambda: Counter(sites=0, candidates=0))
    representative_limit = 20
    conflict_limit_per_candidate = 5
    representative_rejections = []
    p_surface_o_complete_records = []
    p_surface_o_minimum_distances = []
    p_surface_o_by_rank = defaultdict(_p_surface_o_stat_record)
    p_surface_o_by_site = defaultdict(_p_surface_o_stat_record)
    p_surface_o_pre_count = 0
    p_surface_o_excluded_count = 0
    p_surface_o_excluded_pair_count = 0
    for instance in instances:
        site_id = instance["site_instance_id"]
        prototype_id = instance["parent_site_prototype_id"]
        prototype = prototypes.get(prototype_id)
        if prototype is None:
            raise ValueError(f"Missing Site Prototype {prototype_id}")
        instance_frame = (instance.get("placement_template") or {}).get(
            "surface_frame"
        )
        prototype_frame = prototype.get("surface_frame")
        frame = instance_frame if instance_frame is not None else prototype_frame
        if frame is None:
            raise ValueError(f"Missing Surface Frame for {site_id}")
        surface_frame_coordinates(np.zeros((1, 3)), frame)
        frame_axes = frame.get("periodic_fractional_axes")
        if frame_axes is None:
            raise ValueError(
                f"Site Instance {site_id} Surface Frame lacks "
                "periodic_fractional_axes"
            )
        if _validated_surface_axes(frame_axes) != axes:
            raise ValueError(
                f"Site Instance {site_id} Surface Frame periodic axes disagree with the audit"
            )
        if instance_frame is not None and prototype_frame is not None:
            for key in (
                "u_cartesian_unit",
                "v_cartesian_unit",
                "outward_normal_cartesian_unit",
            ):
                if not np.allclose(
                    np.asarray(instance_frame[key], dtype=float),
                    np.asarray(prototype_frame[key], dtype=float),
                    atol=1.0e-8,
                    rtol=0.0,
                ):
                    raise ValueError(
                        f"Site Instance {site_id} Surface Frame disagrees with prototype"
                    )
            if (
                "periodic_fractional_axes" in instance_frame
                and "periodic_fractional_axes" in prototype_frame
                and tuple(instance_frame["periodic_fractional_axes"])
                != tuple(prototype_frame["periodic_fractional_axes"])
            ):
                raise ValueError(
                    f"Site Instance {site_id} Surface Frame axes disagree with prototype"
                )
        prototype_evidence = str(
            prototype.get("representative_candidate_id", prototype_id)
        )
        target, target_donor_labels = _target_headgroup(instance, prototype)
        donor_count = len(target_donor_labels)

        site_metal_coord_audit = None
        if metal_coordination_active:
            max_allowed = int(metal_coordination_contract.get("max_allowed_coordination", 5))
            site_metal_coord_audit = audit_site_instance_metal_coordinations(
                instance=instance,
                metal_records_by_full_id=metal_coordination_data["metal_records_by_full_id"],
                max_allowed_coordination=max_allowed,
            )
            elem_cat = site_metal_coord_audit["element_category"]
            dent_str = str(site_metal_coord_audit["denticity"])
            max_cn_str = str(site_metal_coord_audit["max_coordination_number"])

            site_instance_stats["total_site_instances_before_filter"] += 1
            if elem_cat not in site_instance_stats["by_element"]:
                site_instance_stats["by_element"][elem_cat] = {"before": 0, "excluded": 0, "retained": 0}
            site_instance_stats["by_element"][elem_cat]["before"] += 1
            if dent_str not in site_instance_stats["by_denticity"]:
                site_instance_stats["by_denticity"][dent_str] = {"before": 0, "excluded": 0, "retained": 0}
            site_instance_stats["by_denticity"][dent_str]["before"] += 1
            site_instance_stats["by_mapped_max_coordination"][max_cn_str]["before"] += 1
            site_instance_stats["by_site_prototype"][str(prototype_id)]["before"] += 1
            site_instance_stats["by_site_instance"][str(site_id)] = site_metal_coord_audit

            if site_metal_coord_audit["forbidden"]:
                site_instance_stats["excluded_site_instances"] += 1
                site_instance_stats["by_element"][elem_cat]["excluded"] += 1
                site_instance_stats["by_denticity"][dent_str]["excluded"] += 1
                site_instance_stats["by_mapped_max_coordination"][max_cn_str]["excluded"] += 1
                site_instance_stats["by_site_prototype"][str(prototype_id)]["excluded"] += 1
                site_instance_stats["excluded_site_instance_ids"].append(str(site_id))
                rejection_counts["six_coordinate_metal_site"] += 1
                per_prototype[prototype_id]["six_coordinate_metal_site_rejected"] += 1
                domains[site_id] = []
                continue
            else:
                site_instance_stats["retained_site_instances"] += 1
                site_instance_stats["by_element"][elem_cat]["retained"] += 1
                site_instance_stats["by_denticity"][dent_str]["retained"] += 1
                site_instance_stats["by_mapped_max_coordination"][max_cn_str]["retained"] += 1
                site_instance_stats["by_site_prototype"][str(prototype_id)]["retained"] += 1
                site_instance_stats["retained_site_instance_ids"].append(str(site_id))

        accepted = []
        per_prototype[prototype_id]["sites"] += 1
        for conformer in conformers:
            rank = int(conformer["source_rank"])
            metrics = metrics_by_rank[rank]
            source_coordinates = np.asarray(conformer["coordinates"], dtype=float)
            conformer_symbols = np.asarray(conformer["symbols"], dtype=str)
            conformer_sha256 = _conformer_geometry_sha256(
                source_coordinates, conformer_symbols
            )
            for permutation in itertools.permutations(
                conformer["o_locals"], donor_count
            ):
                registered_bonds = resolve_candidate_registered_bonds(
                    instance=instance,
                    target_donor_labels=target_donor_labels,
                    oxygen_permutation=permutation,
                    molecule_symbols=conformer_symbols,
                    full_to_working_atom_map=full_to_working_atom_map,
                    distance_window_A=(
                        None
                        if substrate_collision_mode == "height"
                        else mapped_bond_window_A
                    ),
                )
                source_order = (conformer["p_local"],) + tuple(permutation)
                rotation, translation, rmsd = rigid_transform(
                    source_coordinates[list(source_order)], target
                )
                if rmsd > headgroup_rmsd_max_A:
                    rejection_counts["headgroup_rmsd"] += 1
                    per_prototype[prototype_id]["headgroup_rmsd_rejected"] += 1
                    continue
                transformed = source_coordinates @ rotation + translation
                if _reject_flipped_orientation(
                    transformed,
                    source_order,
                    clearance_A=0.8,
                    surface_frame=frame,
                ):
                    rejection_counts["flipped_orientation"] += 1
                    per_prototype[prototype_id]["flipped_orientation_rejected"] += 1
                    continue
                candidate_id = _candidate_id_from_fields(
                    site_instance_id=site_id,
                    site_prototype_id=prototype_id,
                    prototype_evidence=prototype_evidence,
                    selection_rank=rank,
                    conformer_sha256=conformer_sha256,
                    donor_labels=target_donor_labels,
                    oxygen_permutation=permutation,
                    symbols=conformer_symbols,
                )
                substrate_audit = audit_candidate_substrate_contacts(
                    mode=substrate_collision_mode,
                    molecule_positions=transformed,
                    molecule_symbols=conformer_symbols,
                    substrate_positions=substrate_coordinates,
                    substrate_symbols=substrate_symbols,
                    cell=substrate.cell,
                    periodic_axes=axes,
                    surface_frame=frame,
                    registered_interface_bonds=registered_bonds,
                    hard_thresholds_A=thresholds_A,
                    vdw_radius_scale=substrate_vdw_radius_scale,
                    anchor_group_atom_indices_0based=[
                        int(conformer["p_local"]),
                        *[int(value) for value in conformer["o_locals"]],
                    ],
                    height_clearance_A=substrate_height_clearance_A,
                )
                if not substrate_audit["passed"]:
                    if substrate_audit["collision_count"]:
                        rejection_counts["substrate_collision"] += 1
                        per_prototype[prototype_id]["substrate_collision_rejected"] += 1
                    if substrate_audit["mapped_bond_violation_count"]:
                        rejection_counts["mapped_bond_window"] += 1
                        per_prototype[prototype_id]["mapped_bond_window_rejected"] += 1
                    if len(representative_rejections) < representative_limit:
                        representative_rejections.append(
                            {
                                "site_instance_id": site_id,
                                "site_prototype_id": prototype_id,
                                "selection_rank": rank,
                                "oxygen_permutation": [
                                    int(value) for value in permutation
                                ],
                                "candidate_id": candidate_id,
                                "mode": substrate_collision_mode,
                                "registered_interface_bonds": registered_bonds,
                                "conflicts": substrate_audit["collisions"][
                                    :conflict_limit_per_candidate
                                ],
                                "mapped_bond_violations": [
                                    record
                                    for record in substrate_audit["mapped_bonds"]
                                    if not record["passed"]
                                ][:conflict_limit_per_candidate],
                                "total_conflict_count": int(
                                    substrate_audit["collision_count"]
                                ),
                                "total_mapped_bond_violation_count": int(
                                    substrate_audit[
                                        "mapped_bond_violation_count"
                                    ]
                                ),
                            }
                        )
                    continue
                p_surface_o_audit = None
                if p_surface_o_exclusion_contract is not None:
                    p_surface_o_audit = audit_p_surface_o_candidate(
                        molecule_positions=transformed,
                        molecule_symbols=conformer_symbols,
                        p_molecule_local_index_0based=int(conformer["p_local"]),
                        substrate_positions=substrate_coordinates,
                        substrate_symbols=substrate_symbols,
                        full_to_working_atom_map=full_to_working_atom_map,
                        cell=substrate.cell,
                        surface_frame=frame,
                        p_surface_o_exclusion_contract=p_surface_o_exclusion_contract,
                        p_surface_o_prefilter=p_surface_o_prefilter,
                    )
                    p_surface_o_pre_count += 1
                    if p_surface_o_audit["minimum_distance_A"] is not None:
                        p_surface_o_minimum_distances.append(
                            float(p_surface_o_audit["minimum_distance_A"])
                        )
                    _update_p_surface_o_stats(
                        p_surface_o_by_rank[str(rank)],
                        p_surface_o_audit,
                        bool(p_surface_o_audit["forbidden"]),
                    )
                    _update_p_surface_o_stats(
                        p_surface_o_by_site[str(site_id)],
                        p_surface_o_audit,
                        bool(p_surface_o_audit["forbidden"]),
                    )
                    if p_surface_o_audit["forbidden"]:
                        p_surface_o_excluded_count += 1
                        p_surface_o_excluded_pair_count += int(
                            p_surface_o_audit["forbidden_pair_count"]
                        )
                        p_surface_o_complete_records.append(
                            {
                                "site_instance_id": site_id,
                                "site_prototype_id": prototype_id,
                                "selection_rank": rank,
                                "conformer_source": conformer["source"],
                                "conformer_sha256": conformer_sha256,
                                "oxygen_permutation": [int(value) for value in permutation],
                                "candidate_id": candidate_id,
                                "p_molecule_local_atom_id_0based": int(conformer["p_local"]),
                                "excluded_pair_count": int(p_surface_o_audit["forbidden_pair_count"]),
                                "minimum_distance_A": p_surface_o_audit["minimum_distance_A"],
                                "forbidden_pairs": p_surface_o_audit["forbidden_pairs"],
                                "surface_axes": list(axes),
                                "normal_axis": int(p_surface_o_exclusion_contract["normal_axis"]),
                                "normal_wrapped": False,
                                "reason": "candidate_placement_excluded_by_inclusive_P-to-any-working-surface-O_gate",
                            }
                        )
                        rejection_counts["p_surface_o_exclusion"] += 1
                        per_prototype[prototype_id]["p_surface_o_excluded"] += 1
                        continue

                candidate = {
                    "site_instance_id": site_id,
                    "site_prototype_id": prototype_id,
                    "selection_rank": rank,
                    "cluster_id": int(metrics["cluster_id"]),
                    "source": conformer["source"],
                    "target_donor_labels": target_donor_labels,
                    "oxygen_permutation": [int(value) for value in permutation],
                    "registered_interface_bonds": registered_bonds,
                    "headgroup_rmsd_A": float(rmsd),
                    "substrate_minimum_distance_A": substrate_audit[
                        "minimum_distance_A"
                    ],
                    "substrate_clearance_ratio": substrate_audit[
                        "minimum_clearance_ratio"
                    ],
                    "conformer_sha256": conformer_sha256,
                    "p_molecule_local_index_0based": int(conformer["p_local"]),
                    "headgroup_local_indices_0based": [
                        int(conformer["p_local"]),
                        *[int(value) for value in conformer["o_locals"]],
                    ],
                    "surface_frame": frame,
                    "anchor_cartesian_A": target[0].tolist(),
                    "candidate_id": candidate_id,
                    "coordinates": transformed,
                    "symbols": conformer_symbols,
                }
                if "relative_total_energy_eV" in metrics:
                    candidate["conformer_energy_eV"] = float(
                        metrics["relative_total_energy_eV"]
                    )
                if substrate_collision_mode != "skip":
                    candidate["substrate_collision_audit"] = substrate_audit
                if p_surface_o_audit is not None:
                    candidate["p_surface_o_exclusion_audit"] = {
                        key: value
                        for key, value in p_surface_o_audit.items()
                        if key not in {"forbidden_pairs", "all_surface_o_pairs"}
                    }
                if site_metal_coord_audit is not None:
                    candidate["metal_coordination_audit"] = site_metal_coord_audit
                    elem_cat = site_metal_coord_audit["element_category"]
                    dent_str = str(site_metal_coord_audit["denticity"])
                    max_cn_str = str(site_metal_coord_audit["max_coordination_number"])
                    metal_coord_stats["candidate_count_before_exclusion"] += 1
                    metal_coord_stats["retained_candidate_count"] += 1
                    if elem_cat not in metal_coord_stats["by_element"]:
                        metal_coord_stats["by_element"][elem_cat] = {"before": 0, "excluded": 0, "retained": 0}
                    metal_coord_stats["by_element"][elem_cat]["before"] += 1
                    metal_coord_stats["by_element"][elem_cat]["retained"] += 1
                    if dent_str not in metal_coord_stats["by_denticity"]:
                        metal_coord_stats["by_denticity"][dent_str] = {"before": 0, "excluded": 0, "retained": 0}
                    metal_coord_stats["by_denticity"][dent_str]["before"] += 1
                    metal_coord_stats["by_denticity"][dent_str]["retained"] += 1
                    metal_coord_stats["by_mapped_max_coordination"][max_cn_str]["before"] += 1
                    metal_coord_stats["by_mapped_max_coordination"][max_cn_str]["retained"] += 1
                    metal_coord_stats["by_site_prototype"][str(prototype_id)]["before"] += 1
                    metal_coord_stats["by_site_prototype"][str(prototype_id)]["retained"] += 1
                    site_stat = metal_coord_stats["by_site_instance"][str(site_id)]
                    site_stat["before"] += 1
                    site_stat["retained"] += 1
                    site_stat["denticity"] = site_metal_coord_audit["denticity"]
                    site_stat["element_category"] = elem_cat
                    site_stat["mapped_metals"] = site_metal_coord_audit["mapped_metals"]
                accepted.append(candidate)
        domains[site_id] = accepted
        per_prototype[prototype_id]["candidates"] += len(accepted)

    single_metal_diagnostics = {
        "mode": single_metal_sites_mode,
        "enabled": single_metal_sites_mode != "off",
        "uncovered_metal_count": 0,
        "uncovered_metal_atom_ids_1based": [],
        "single_metal_candidate_count": 0,
    }
    if single_metal_sites_mode == "dynamic-cn5-uncovered":
        retained_instances = [
            inst for inst in instances
            if site_instance_stats is None
            or str(inst.get("site_instance_id")) in site_instance_stats["retained_site_instance_ids"]
        ]
        norm_ax = int(metal_coordination_contract.get("normal_axis", 2)) if metal_coordination_contract else 2
        cutoff_val = float(metal_coordination_contract.get("cutoff_A", 2.7)) if metal_coordination_contract else 2.7
        target_elems = tuple(metal_coordination_contract.get("target_elements", ("In", "Sn"))) if metal_coordination_contract else ("In", "Sn")
        ligand_elems = tuple(metal_coordination_contract.get("ligand_elements", ("O",))) if metal_coordination_contract else ("O",)

        uncovered_metals = _discover_uncovered_cn5_metals(
            substrate_positions=substrate_coordinates,
            substrate_symbols=substrate_symbols,
            cell=substrate.cell,
            periodic_axes=axes,
            normal_axis=norm_ax,
            retained_site_instances=retained_instances,
            full_to_working_atom_map=full_to_working_atom_map,
            cutoff_A=cutoff_val,
            top_surface_depth_A=0.7,
            target_elements=target_elems,
            ligand_elements=ligand_elems,
        )

        ref_frame = None
        for inst in instances:
            inst_frame = (inst.get("placement_template") or {}).get("surface_frame") or (prototypes.get(inst.get("parent_site_prototype_id")) or {}).get("surface_frame")
            if inst_frame is not None:
                ref_frame = inst_frame
                break
        if ref_frame is None and prototypes:
            ref_frame = next(iter(prototypes.values())).get("surface_frame")
        if ref_frame is None:
            raise ValueError("Unable to resolve Surface Frame for single-metal candidates")

        single_metal_context = {
            "mode": single_metal_sites_mode,
            "uncovered_metals": uncovered_metals,
            "substrate_positions": substrate_coordinates,
            "substrate_symbols": substrate_symbols,
            "cell": np.asarray(substrate.cell, dtype=float),
            "periodic_axes": axes,
            "surface_frame": ref_frame,
            "conformers": conformers,
            "metrics_by_rank": metrics_by_rank,
            "substrate_height_clearance_A": substrate_height_clearance_A,
            "rotation_step_deg": float(single_metal_rotation_step_deg),
            "translation_offsets_A": tuple(float(v) for v in single_metal_translation_offsets_A),
        }
        single_metal_diagnostics["uncovered_metal_count"] = len(uncovered_metals)
        single_metal_diagnostics["uncovered_metal_atom_ids_1based"] = [
            int(rec["full_metal_atom_id_1based"]) for rec in uncovered_metals
        ]
        single_metal_diagnostics["single_metal_candidate_count"] = "lazy_on_demand"
    else:
        single_metal_context = None

    all_candidate_ids = sorted(
        candidate["candidate_id"]
        for candidates in domains.values()
        for candidate in candidates
    )
    if len(set(all_candidate_ids)) != len(all_candidate_ids):
        raise ValueError("Duplicate candidate IDs in candidate domain")
    index_by_id = {
        candidate_id: index for index, candidate_id in enumerate(all_candidate_ids)
    }
    for candidates in domains.values():
        for candidate in candidates:
            candidate["candidate_index"] = index_by_id[candidate["candidate_id"]]
    fingerprint_contract = (
        None if substrate_collision_mode == "skip" else substrate_contract
    )
    candidate_domain_fingerprint = _candidate_domain_fingerprint(
        domains,
        cell=substrate.cell,
        periodic_axes=axes,
        substrate_collision_contract=fingerprint_contract,
        p_surface_o_exclusion_contract=p_surface_o_exclusion_contract,
        metal_coordination_contract=metal_coordination_contract,
    )
    diagnostics = {
        "site_count": len(domains),
        "candidate_count": sum(len(candidates) for candidates in domains.values()),
        "empty_site_count": sum(not candidates for candidates in domains.values()),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "substrate_collision_contract": substrate_contract,
        "candidate_domain_fingerprint": candidate_domain_fingerprint,
        "candidate_domain_fingerprint_namespace": (
            "metal-coordination-filter-v1"
            if metal_coordination_active
            else (
                "p-surface-o-exclusion-v1"
                if p_surface_o_exclusion_contract is not None
                else ("legacy-v1" if substrate_collision_mode == "skip" else "interface-v2")
            )
        ),
        "representative_rejection_limit": representative_limit,
        "representative_conflict_limit_per_candidate": conflict_limit_per_candidate,
        "representative_substrate_rejections": representative_rejections,
        "per_prototype": {
            key: dict(value) for key, value in sorted(per_prototype.items())
        },
    }
    if p_surface_o_exclusion_contract is not None:
        p_surface_o_by_rank_json = {
            key: dict(value) for key, value in sorted(p_surface_o_by_rank.items())
        }
        p_surface_o_by_site_json = {
            key: dict(value) for key, value in sorted(p_surface_o_by_site.items())
        }
        diagnostics["p_surface_o_exclusion"] = {
            "schema": _P_SURFACE_O_EXCLUSION_SCHEMA,
            "enabled": True,
            "mode": p_surface_o_exclusion_contract["mode"],
            "contact_upper_bound_A": float(p_surface_o_exclusion_contract["contact_upper_bound_A"]),
            "inclusive_cutoff": True,
            "surface_axes": list(axes),
            "normal_axis": int(p_surface_o_exclusion_contract["normal_axis"]),
            "normal_wrapped": False,
            "candidate_count_before_exclusion": int(p_surface_o_pre_count),
            "excluded_placement_count": int(p_surface_o_excluded_count),
            "retained_candidate_count": int(sum(len(candidates) for candidates in domains.values())),
            "excluded_pair_count": int(p_surface_o_excluded_pair_count),
            "by_selection_rank": p_surface_o_by_rank_json,
            "by_site_instance": p_surface_o_by_site_json,
            "sensitivity": {
                "thresholds_A": list(p_surface_o_exclusion_contract["sensitivity"]["thresholds_A"]),
                "placement_counts_le_threshold": {
                    f"<={threshold:g}": int(sum(
                        int(value["sensitivity_counts"][f"<={threshold:g}"])
                        for value in p_surface_o_by_rank_json.values()
                    ))
                    for threshold in _P_SURFACE_O_SENSITIVITY_THRESHOLDS_A
                },
                "minimum_distance_bins": _p_surface_o_sensitivity_bins(
                    p_surface_o_exclusion_contract, p_surface_o_minimum_distances
                ),
                "contract_window": p_surface_o_exclusion_contract["sensitivity"]["contract_window_bins"],
                "runtime_gate_unchanged_at_contract_upper_bound": True,
            },
            "representative_excluded_placements": p_surface_o_complete_records[:20],
            "representative_limit": 20,
            "complete_excluded_placement_audit_count": len(p_surface_o_complete_records),
            "complete_excluded_placement_audit": p_surface_o_complete_records,
            "other_sam_substrate_pairs_checked": False,
            "other_sam_substrate_pairs_rejected": False,
        }
    else:
        diagnostics["p_surface_o_exclusion"] = {
            "schema": _P_SURFACE_O_EXCLUSION_SCHEMA,
            "enabled": False,
            "mode": "off",
            "reason": "explicit_default_off",
        }
    if metal_coordination_active:
        diagnostics["metal_coordination_filter"] = {
            "schema": _METAL_COORDINATION_FILTER_SCHEMA,
            "enabled": True,
            "mode": metal_coordination_contract["mode"],
            "cutoff_A": float(metal_coordination_contract["cutoff_A"]),
            "max_allowed_coordination": int(metal_coordination_contract["max_allowed_coordination"]),
            "target_elements": list(metal_coordination_contract["target_elements"]),
            "ligand_elements": list(metal_coordination_contract["ligand_elements"]),
            "surface_axes": list(metal_coordination_contract["periodic_axes"]),
            "normal_axis": int(metal_coordination_contract["normal_axis"]),
            "normal_wrapped": False,
            "all_mapped_metals_must_pass": True,
            "hypothesis_testing_note": metal_coordination_contract.get("hypothesis_testing_note", ""),
            "substrate_metal_inventory": metal_coordination_data["inventory"],
            "site_instance_summary": {
                "total_site_instances_before_filter": int(site_instance_stats["total_site_instances_before_filter"]),
                "excluded_site_instances": int(site_instance_stats["excluded_site_instances"]),
                "retained_site_instances": int(site_instance_stats["retained_site_instances"]),
                "by_element": site_instance_stats["by_element"],
                "by_denticity": site_instance_stats["by_denticity"],
                "by_mapped_max_coordination": {k: dict(v) for k, v in sorted(site_instance_stats["by_mapped_max_coordination"].items())},
                "by_site_prototype": {k: dict(v) for k, v in sorted(site_instance_stats["by_site_prototype"].items())},
                "excluded_site_instance_ids": list(site_instance_stats["excluded_site_instance_ids"]),
                "retained_site_instance_ids": list(site_instance_stats["retained_site_instance_ids"]),
            },
            "candidate_filter_summary": {
                "candidate_count_before_exclusion": int(metal_coord_stats["candidate_count_before_exclusion"]),
                "excluded_candidate_count": int(metal_coord_stats["excluded_candidate_count"]),
                "retained_candidate_count": int(metal_coord_stats["retained_candidate_count"]),
                "by_element": metal_coord_stats["by_element"],
                "by_denticity": metal_coord_stats["by_denticity"],
                "by_mapped_max_coordination": {k: dict(v) for k, v in sorted(metal_coord_stats["by_mapped_max_coordination"].items())},
                "by_site_prototype": {k: dict(v) for k, v in sorted(metal_coord_stats["by_site_prototype"].items())},
                "by_site_instance": {k: dict(v) for k, v in sorted(metal_coord_stats["by_site_instance"].items())},
                "representative_excluded_placements": metal_coord_stats["representative_excluded_placements"],
            },
        }
    else:
        diagnostics["metal_coordination_filter"] = {
            "schema": _METAL_COORDINATION_FILTER_SCHEMA,
            "enabled": False,
            "mode": "off",
            "reason": "explicit_default_off",
        }
    diagnostics["single_metal_sites"] = single_metal_diagnostics
    return domains, diagnostics


def _conflict_map(site_payload: dict) -> dict[str, set[str]]:
    conflicts = defaultdict(set)
    for left, right in site_payload.get("conflict_edges", []):
        conflicts[str(left)].add(str(right))
        conflicts[str(right)].add(str(left))
    return dict(conflicts)


def _assemble(substrate, placed: list[dict]):
    assembled = substrate.copy()
    for candidate in placed:
        molecule = Atoms(
            symbols=candidate["symbols"].tolist(),
            positions=candidate["coordinates"],
            cell=substrate.cell,
            pbc=substrate.pbc,
        )
        assembled.extend(molecule)
    assembled.set_cell(substrate.cell)
    assembled.set_pbc(substrate.pbc)
    return assembled


def _candidate_actual_metal_ids(candidate, metal_constraints=None) -> set[int]:
    """Resolve actual full-substrate metal identities without element inference."""

    records = candidate.get("registered_interface_bonds") or []
    from_bonds = {
        int(record["full_metal_atom_id_1based"])
        for record in records
        if "full_metal_atom_id_1based" in record
    }
    if from_bonds:
        return from_bonds
    if candidate.get("occupied_metal_ids"):
        return {int(mid) for mid in candidate["occupied_metal_ids"]}
    test_value = candidate.get("actual_metal_ids_for_test")
    if test_value is not None:
        return {int(value) for value in test_value}
    if metal_constraints is not None:
        return {
            int(value)
            for value in metal_constraints.get(
                str(candidate["site_instance_id"]), ()
            )
        }
    return set()


def _repair_objective_summary(objective: dict) -> dict:
    return {
        key: value
        for key, value in objective.items()
        if key not in {"accepted_union", "torus_component_geometries"}
    }


def repair_objective(placed, projection_cache) -> dict:
    """Rebuild the complete accepted footprint union and exact torus objective."""

    accepted_union = None
    candidate_ids = []
    for candidate in placed:
        candidate_id = str(candidate.get("candidate_id", ""))
        candidate_index = candidate.get("candidate_index")
        if not candidate_id or isinstance(candidate_index, bool):
            raise ValueError("Repair objective requires stable candidate identities")
        candidate_index = int(candidate_index)
        if candidate_index not in projection_cache.candidate_geometries:
            raise ValueError(
                f"Repair candidate {candidate_id} is absent from ProjectionCache"
            )
        _, accepted_union = periodic_union_increment_A2(
            accepted_union,
            projection_cache.candidate_geometries[candidate_index],
            projection_cache.area_scale_A2,
        )
        candidate_ids.append(candidate_id)
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Repair accepted candidate IDs must be unique")
    diagnostics = periodic_hole_diagnostics(
        accepted_union,
        projection_cache.area_scale_A2,
        surface_lattice_uv_A=projection_cache.surface_lattice_uv_A,
    )
    components = periodic_uncovered_component_geometries(
        accepted_union,
        projection_cache.area_scale_A2,
        surface_lattice_uv_A=projection_cache.surface_lattice_uv_A,
    )
    maximum_component = components[0]["area_A2"] if components else 0.0
    if maximum_component != diagnostics["maximum_hole_area_A2"]:
        raise RuntimeError(
            "Torus component geometry maximum disagrees with hole diagnostics"
        )
    return {
        "molecule_count_N": len(placed),
        "total_hole_area_A2": float(diagnostics["total_hole_area_A2"]),
        "maximum_hole_area_A2": float(diagnostics["maximum_hole_area_A2"]),
        "total_hole_perimeter_A": float(
            diagnostics["total_hole_perimeter_A"]
        ),
        "hole_count": int(diagnostics["hole_count"]),
        "metric_version": diagnostics["metric_version"],
        "accepted_union": accepted_union,
        "torus_component_geometries": components,
    }


def strict_repair_pareto(before, after, *, area_tolerance_A2=1.0e-6) -> bool:
    """Apply the unweighted Stage2 cardinality/total/max strict Pareto rule."""

    tolerance = float(area_tolerance_A2)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("area_tolerance_A2 must be finite and non-negative")
    before_n = int(before["molecule_count_N"])
    after_n = int(after["molecule_count_N"])
    if after_n < before_n:
        return False
    before_total = float(before["total_hole_area_A2"])
    after_total = float(after["total_hole_area_A2"])
    before_maximum = float(before["maximum_hole_area_A2"])
    after_maximum = float(after["maximum_hole_area_A2"])
    values = [before_total, after_total, before_maximum, after_maximum]
    if not np.isfinite(values).all():
        raise ValueError("Repair Pareto objectives must be finite")
    if after_total > before_total + tolerance:
        return False
    if after_maximum > before_maximum + tolerance:
        return False
    if after_n > before_n:
        return True
    return (
        before_total - after_total > tolerance
        or before_maximum - after_maximum > tolerance
    )


def _repair_action_fingerprint(action_type, remove_ids, add_ids) -> str:
    return _payload_sha256(
        {
            "schema": "sam-local-repair-action-fingerprint-v1",
            "action_type": str(action_type),
            "remove_candidate_ids": sorted(str(value) for value in remove_ids),
            "add_candidate_ids": sorted(str(value) for value in add_ids),
        }
    )


def _candidate_collides_with_any(
    candidate,
    others,
    *,
    cell,
    periodic_axes,
    collision_thresholds_A,
) -> tuple[bool, list[str]]:
    conflicts = []
    for other in others:
        count, _, _ = surface_collision_report(
            candidate["coordinates"],
            candidate["symbols"],
            other["coordinates"],
            other["symbols"],
            cell,
            periodic_axes,
            thresholds_A=collision_thresholds_A,
        )
        if count:
            conflicts.append(str(other["candidate_id"]))
    return bool(conflicts), sorted(conflicts)


def _external_placement_hash(placed) -> str:
    records = []
    for candidate in placed:
        records.append(
            {
                "candidate_id": str(candidate["candidate_id"]),
                "site_instance_id": str(candidate["site_instance_id"]),
                "coordinate_sha256": _coordinate_array_sha256(
                    candidate["coordinates"]
                ),
                "symbol_sequence_sha256": _symbol_sequence_sha256(
                    candidate["symbols"]
                ),
            }
        )
    return _payload_sha256(
        {"schema": "sam-local-repair-fixed-external-v1", "placements": records}
    )


def run_fixed_site_replacements(
    *,
    placed,
    domains,
    projection_cache,
    cell,
    periodic_axes,
    collision_thresholds_A,
    max_replacements=10,
    area_tolerance_A2=1.0e-6,
    metal_constraints=None,
) -> dict:
    """Iteratively select deterministic strict-Pareto same-site replacements."""

    if isinstance(max_replacements, bool) or int(max_replacements) < 0:
        raise ValueError("max_replacements must be a non-negative integer")
    maximum = int(max_replacements)
    thresholds = _validate_collision_thresholds(collision_thresholds_A)
    current = list(placed)
    initial = repair_objective(current, projection_cache)
    rounds = []
    accepted_actions = []
    all_observed = []
    for round_number in range(1, maximum + 1):
        before = repair_objective(current, projection_cache)
        evaluated = []
        externally_compatible = []
        pareto = []
        for placed_index, old in enumerate(current):
            site_id = str(old["site_instance_id"])
            alternatives = sorted(
                (
                    candidate
                    for candidate in domains.get(site_id, ())
                    if str(candidate["candidate_id"])
                    != str(old["candidate_id"])
                ),
                key=lambda candidate: str(candidate["candidate_id"]),
            )
            old_metals = _candidate_actual_metal_ids(
                old, metal_constraints=metal_constraints
            )
            external = current[:placed_index] + current[placed_index + 1 :]
            external_hash = _external_placement_hash(external)
            for candidate in alternatives:
                if str(candidate["site_instance_id"]) != site_id:
                    raise RuntimeError("Same-site replacement changed Site Instance")
                candidate_metals = _candidate_actual_metal_ids(
                    candidate, metal_constraints=metal_constraints
                )
                if candidate_metals != old_metals:
                    raise RuntimeError("Same-site replacement changed actual metals")
                fingerprint = _repair_action_fingerprint(
                    "fixed_N_same_site_replacement",
                    [old["candidate_id"]],
                    [candidate["candidate_id"]],
                )
                base = {
                    "action_type": "fixed_N_same_site_replacement",
                    "site_instance_id": site_id,
                    "remove_candidate_ids": [str(old["candidate_id"])],
                    "add_candidate_ids": [str(candidate["candidate_id"])],
                    "action_fingerprint": fingerprint,
                    "fixed_external_sha256": external_hash,
                    "site_unchanged": True,
                    "actual_metals_unchanged": True,
                    "thresholds_A": thresholds,
                    "weighted_score_used": False,
                }
                evaluated.append(base)
                collides, conflicting_ids = _candidate_collides_with_any(
                    candidate,
                    external,
                    cell=cell,
                    periodic_axes=periodic_axes,
                    collision_thresholds_A=thresholds,
                )
                if collides:
                    base["externally_compatible"] = False
                    base["external_collision_candidate_ids"] = conflicting_ids
                    continue
                base["externally_compatible"] = True
                externally_compatible.append(base)
                trial = list(current)
                trial[placed_index] = candidate
                after = repair_objective(trial, projection_cache)
                summary = {
                    **base,
                    "objective_before": _repair_objective_summary(before),
                    "objective_after": _repair_objective_summary(after),
                    "strict_pareto": strict_repair_pareto(
                        before,
                        after,
                        area_tolerance_A2=area_tolerance_A2,
                    ),
                    "area_tolerance_A2": float(area_tolerance_A2),
                    "_trial": trial,
                }
                if summary["strict_pareto"]:
                    pareto.append(summary)
                    all_observed.append(
                        {key: value for key, value in summary.items() if key != "_trial"}
                    )
        pareto.sort(
            key=lambda action: (
                float(action["objective_after"]["total_hole_area_A2"]),
                float(action["objective_after"]["maximum_hole_area_A2"]),
                float(action["objective_after"]["total_hole_perimeter_A"]),
                str(action["action_fingerprint"]),
            )
        )
        selected = pareto[0] if pareto else None
        round_record = {
            "round": round_number,
            "evaluated_candidate_count": len(evaluated),
            "externally_compatible_candidate_count": len(externally_compatible),
            "pareto_candidate_count": len(pareto),
            "evaluated_actions": evaluated,
            "observed_pareto_actions": [
                {key: value for key, value in action.items() if key != "_trial"}
                for action in pareto
            ],
            "selected_action_fingerprint": (
                None if selected is None else selected["action_fingerprint"]
            ),
            "thresholds_A": thresholds,
            "area_tolerance_A2": float(area_tolerance_A2),
            "weighted_score_used": False,
        }
        rounds.append(round_record)
        if selected is None:
            break
        current = selected.pop("_trial")
        accepted_actions.append(dict(selected))
    final = repair_objective(current, projection_cache)
    return {
        "status": (
            "accepted_fixed_site_replacements"
            if accepted_actions
            else "no_strict_pareto_fixed_site_replacement"
        ),
        "placed": current,
        "objective_before": _repair_objective_summary(initial),
        "objective_after": _repair_objective_summary(final),
        "rounds": rounds,
        "accepted_actions": accepted_actions,
        "all_observed_pareto_actions": all_observed,
        "thresholds_A": thresholds,
        "area_tolerance_A2": float(area_tolerance_A2),
        "weighted_score_used": False,
    }


def certified_static_repair_gate(
    *,
    placed,
    domains,
    metal_constraints,
    site_conflicts,
    time_limit_seconds,
) -> dict:
    """Certify the current-domain static cardinality upper bound before LNS."""

    feasible_sites = sorted(
        str(site_id) for site_id, candidates in domains.items() if candidates
    )
    certificate = solve_static_conflict_capacity(
        feasible_sites,
        metal_constraints,
        site_conflicts=site_conflicts,
        time_limit_seconds=time_limit_seconds,
    )
    molecule_count = len(placed)
    capacity = int(certificate["objective"])
    if capacity < molecule_count:
        raise RuntimeError(
            "Certified static domain capacity is below the accepted molecule count"
        )
    margin = capacity - molecule_count
    skipped = margin == 0
    return {
        "status": (
            "skipped_proven_static_cardinality_bound"
            if skipped
            else "eligible_static_capacity_margin_positive"
        ),
        "molecule_count_N": molecule_count,
        "certified_static_K": capacity,
        "capacity_margin": margin,
        "run_k_to_k_plus_one": not skipped,
        "certificate": certificate,
        "certificate_scope": (
            "entire_current_candidate_domain_site_instances_plus_actual_metals_"
            "and_explicit_static_conflicts"
        ),
        "global_packing_certificate": False,
    }


def _local_collision_edges(candidates, *, cell, periodic_axes, thresholds):
    edges = []
    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1 :]:
            count, _, _ = surface_collision_report(
                left["coordinates"],
                left["symbols"],
                right["coordinates"],
                right["symbols"],
                cell,
                periodic_axes,
                thresholds_A=thresholds,
            )
            if count:
                edges.append(
                    tuple(
                        sorted(
                            (str(left["candidate_id"]), str(right["candidate_id"]))
                        )
                    )
                )
    return sorted(set(edges))


def _repair_milp_record(result, *, purpose, elapsed, time_limit_seconds, objective_sign=-1):
    status = int(getattr(result, "status", -1))
    success = bool(getattr(result, "success", False))
    message = str(getattr(result, "message", "missing solver message"))
    raw_x = getattr(result, "x", None)
    incumbent_present = raw_x is not None and np.asarray(raw_x).size > 0 and np.all(
        np.isfinite(np.asarray(raw_x, dtype=float))
    )
    raw_fun = getattr(result, "fun", None)
    try:
        raw_fun = None if raw_fun is None else float(raw_fun)
    except (TypeError, ValueError, OverflowError):
        raw_fun = None
    if raw_fun is not None and not np.isfinite(raw_fun):
        raw_fun = None
    raw_dual = getattr(result, "mip_dual_bound", None)
    try:
        raw_dual = None if raw_dual is None else float(raw_dual)
    except (TypeError, ValueError, OverflowError):
        raw_dual = None
    if raw_dual is not None and not np.isfinite(raw_dual):
        raw_dual = None
    raw_gap = getattr(result, "mip_gap", None)
    try:
        raw_gap = None if raw_gap is None else float(raw_gap)
    except (TypeError, ValueError, OverflowError):
        raw_gap = None
    if raw_gap is not None and not np.isfinite(raw_gap):
        raw_gap = None
    timeout = status == 1
    if status == 0 and success:
        termination = (
            "certified_local_model_maximum_cardinality"
            if purpose == "maximum_cardinality"
            else "certified_feasible_solution"
        )
    elif status == 2:
        termination = "certified_local_model_infeasible"
    elif timeout and incumbent_present:
        termination = "time_limit_with_incumbent_not_certified"
    elif timeout:
        termination = "time_limit_without_incumbent_not_certified"
    else:
        termination = "solver_termination_not_certified"
    primal = (
        None
        if raw_fun is None
        else float(objective_sign) * raw_fun
    )
    dual = (
        None
        if raw_dual is None
        else float(objective_sign) * raw_dual
    )
    return {
        "purpose": purpose,
        "solver_status_code": status,
        "success": success,
        "message": message,
        "primal_bound": primal,
        "dual_bound": dual,
        "raw_minimization_objective": raw_fun,
        "raw_mip_dual_bound": raw_dual,
        "mip_gap": raw_gap,
        "mip_node_count": (
            None
            if getattr(result, "mip_node_count", None) is None
            or not np.isfinite(float(result.mip_node_count))
            else int(result.mip_node_count)
        ),
        "solve_time_seconds": float(elapsed),
        "time_limit_seconds": float(time_limit_seconds),
        "timeout": timeout,
        "incumbent_present": bool(incumbent_present),
        "termination": termination,
        "local_model_optimum": bool(
            purpose == "maximum_cardinality" and status == 0 and success
        ),
        "global_optimum": False,
        "global_infeasible": False,
        "local_model_infeasible": status == 2,
    }


def _validate_local_binary_milp_solution(
    result,
    *,
    objective_coefficients,
    constraint_matrix,
    constraint_lower,
    constraint_upper,
    label,
) -> tuple[np.ndarray, dict]:
    """Fail closed on a nominally optimal local MILP incumbent/certificate."""

    status = int(getattr(result, "status", -1))
    success = bool(getattr(result, "success", False))
    if status != 0 or not success:
        raise RuntimeError(f"{label} lacks certified optimal solver status")
    coefficients = np.asarray(objective_coefficients, dtype=float).reshape(-1)
    values = np.asarray(getattr(result, "x", []), dtype=float)
    if values.shape != coefficients.shape or not np.all(np.isfinite(values)):
        raise RuntimeError(f"{label} returned a malformed/non-finite x vector")
    rounded = np.rint(values)
    if (
        np.any(values < 0.0)
        or np.any(values > 1.0)
        or not np.allclose(values, rounded, atol=1.0e-7, rtol=0.0)
        or np.any(rounded < 0.0)
        or np.any(rounded > 1.0)
    ):
        raise RuntimeError(f"{label} x must be finite binary values in [0,1]")

    raw_fun = getattr(result, "fun", None)
    try:
        fun = float(raw_fun)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"{label} objective is missing or malformed") from exc
    expected_fun = float(np.dot(coefficients, rounded))
    if not np.isfinite(fun) or abs(fun - expected_fun) > 1.0e-7:
        raise RuntimeError(
            f"{label} objective disagrees with c@x: {fun} != {expected_fun}"
        )

    raw_gap = getattr(result, "mip_gap", None)
    raw_dual = getattr(result, "mip_dual_bound", None)
    try:
        gap = float(raw_gap)
        dual = float(raw_dual)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(
            f"{label} requires present finite gap and dual-bound evidence"
        ) from exc
    if not np.isfinite(gap) or gap != 0.0:
        raise RuntimeError(f"{label} requires an exactly zero present mip_gap")
    if not np.isfinite(dual):
        raise RuntimeError(f"{label} requires a present finite mip_dual_bound")
    if abs(dual - expected_fun) > 1.0e-7:
        raise RuntimeError(
            f"{label} dual bound does not close on the certified objective"
        )

    matrix = constraint_matrix
    lower = np.asarray(constraint_lower, dtype=float).reshape(-1)
    upper = np.asarray(constraint_upper, dtype=float).reshape(-1)
    if matrix.shape != (len(lower), len(values)) or upper.shape != lower.shape:
        raise RuntimeError(f"{label} internal constraint dimensions are malformed")
    loads = np.asarray(matrix @ rounded, dtype=float).reshape(-1)
    if not np.all(np.isfinite(loads)):
        raise RuntimeError(f"{label} produced non-finite constraint-row loads")
    finite_lower = np.isfinite(lower)
    finite_upper = np.isfinite(upper)
    if np.any(loads[finite_lower] < lower[finite_lower] - 1.0e-7) or np.any(
        loads[finite_upper] > upper[finite_upper] + 1.0e-7
    ):
        raise RuntimeError(f"{label} solution violates a constraint row")
    return rounded.astype(np.int8), {
        "x_finite_binary_in_closed_unit_interval": True,
        "raw_minimization_objective": fun,
        "objective_from_solution": expected_fun,
        "mip_gap_present_finite_exactly_zero": True,
        "mip_dual_bound_present_finite_and_closed": True,
        "constraint_row_count": int(len(loads)),
        "maximum_constraint_row_load": (
            float(np.max(loads)) if len(loads) else 0.0
        ),
        "all_constraint_row_bounds_satisfied": True,
    }


def enumerate_local_repair_solutions(
    *,
    candidates,
    target_count,
    metal_constraints,
    site_conflicts,
    cell,
    periodic_axes,
    collision_thresholds_A,
    max_solutions,
    time_limit_seconds,
) -> dict:
    """Certify local cardinality, then enumerate exact target solutions by MILP."""

    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix, vstack

    if isinstance(target_count, bool) or int(target_count) <= 0:
        raise ValueError("target_count must be a positive integer")
    if isinstance(max_solutions, bool) or int(max_solutions) <= 0:
        raise ValueError("max_solutions must be a positive integer")
    target_count = int(target_count)
    max_solutions = int(max_solutions)
    limit = float(time_limit_seconds)
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("time_limit_seconds must be finite and positive")
    thresholds = _validate_collision_thresholds(collision_thresholds_A)
    ordered = sorted(candidates, key=lambda candidate: str(candidate["candidate_id"]))
    candidate_ids = [str(candidate["candidate_id"]) for candidate in ordered]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Local repair candidate IDs must be unique")
    index = {candidate_id: position for position, candidate_id in enumerate(candidate_ids)}
    collision_edges = _local_collision_edges(
        ordered, cell=cell, periodic_axes=periodic_axes, thresholds=thresholds
    )
    site_groups = defaultdict(list)
    metal_groups = defaultdict(list)
    for position, candidate in enumerate(ordered):
        site_id = str(candidate["site_instance_id"])
        site_groups[site_id].append(position)
        for metal in metal_constraints.get(site_id, ()):
            metal_groups[int(metal)].append(position)
    static_edges = set()
    for left_site, neighbors in (site_conflicts or {}).items():
        left_site = str(left_site)
        for right_site in neighbors:
            right_site = str(right_site)
            if left_site == right_site:
                continue
            for left_position in site_groups.get(left_site, ()):
                for right_position in site_groups.get(right_site, ()):
                    static_edges.add(tuple(sorted((left_position, right_position))))

    row_indices = []
    row_kinds = []
    for site_id, positions in sorted(site_groups.items()):
        if len(positions) > 1:
            row_indices.append(positions)
            row_kinds.append(("site_capacity", site_id))
    for metal, positions in sorted(metal_groups.items()):
        if len(positions) > 1:
            row_indices.append(positions)
            row_kinds.append(("actual_metal_capacity", metal))
    for left_id, right_id in collision_edges:
        row_indices.append([index[left_id], index[right_id]])
        row_kinds.append(("exact_sam_collision", [left_id, right_id]))
    for left, right in sorted(static_edges):
        row_indices.append([left, right])
        row_kinds.append(
            ("explicit_static_conflict", [candidate_ids[left], candidate_ids[right]])
        )
    base = lil_matrix((len(row_indices), len(ordered)), dtype=float)
    for row, positions in enumerate(row_indices):
        for position in positions:
            base[row, position] = 1.0
    base = base.tocsr()
    base_lower = np.full(len(row_indices), -np.inf)
    base_upper = np.ones(len(row_indices))
    started = time.perf_counter()
    solve_records = []

    def remaining_limit():
        return max(1.0e-9, limit - (time.perf_counter() - started))

    if not ordered:
        return {
            "candidate_count": 0,
            "target_count": target_count,
            "maximum_cardinality": 0,
            "maximum_cardinality_certified": True,
            "local_infeasible_certified": True,
            "solutions": [],
            "collision_edges": [],
            "static_conflict_edges": [],
            "solve_records": [],
            "search_complete": True,
            "solution_limit_reached": False,
            "timeout_observed": False,
            "global_packing_claim": False,
        }
    maximum_started = time.perf_counter()
    maximum_result = milp(
        c=-np.ones(len(ordered)),
        integrality=np.ones(len(ordered)),
        bounds=Bounds(0.0, 1.0),
        constraints=(
            None
            if not len(row_indices)
            else LinearConstraint(base, base_lower, base_upper)
        ),
        options={"time_limit": remaining_limit(), "mip_rel_gap": 0.0},
    )
    maximum_record = _repair_milp_record(
        maximum_result,
        purpose="maximum_cardinality",
        elapsed=time.perf_counter() - maximum_started,
        time_limit_seconds=limit,
        objective_sign=-1,
    )
    solve_records.append(maximum_record)
    maximum_cardinality = None
    maximum_certified = False
    if maximum_record["termination"] == "certified_local_model_maximum_cardinality":
        rounded, validation = _validate_local_binary_milp_solution(
            maximum_result,
            objective_coefficients=-np.ones(len(ordered)),
            constraint_matrix=base,
            constraint_lower=base_lower,
            constraint_upper=base_upper,
            label="Local maximum-cardinality MILP",
        )
        maximum_cardinality = int(np.sum(rounded))
        if abs(float(maximum_record["dual_bound"]) - maximum_cardinality) > 1.0e-7:
            raise RuntimeError(
                "Local maximum-cardinality MILP converted dual bound does not "
                "close on the maximized integer cardinality"
            )
        validation.update(
            {
                "maximized_integer_cardinality": maximum_cardinality,
                "raw_fun_equals_negative_selected_count": True,
                "converted_dual_bound_closes_on_cardinality": True,
            }
        )
        maximum_record["solution_validation"] = validation
        maximum_certified = True
    if not maximum_certified:
        return {
            "candidate_count": len(ordered),
            "target_count": target_count,
            "maximum_cardinality": maximum_cardinality,
            "maximum_cardinality_certified": False,
            "local_infeasible_certified": False,
            "solutions": [],
            "collision_edges": [list(edge) for edge in collision_edges],
            "static_conflict_edges": [
                [candidate_ids[left], candidate_ids[right]]
                for left, right in sorted(static_edges)
            ],
            "solve_records": solve_records,
            "search_complete": False,
            "solution_limit_reached": False,
            "timeout_observed": any(record["timeout"] for record in solve_records),
            "global_packing_claim": False,
        }
    if maximum_cardinality < target_count:
        return {
            "candidate_count": len(ordered),
            "target_count": target_count,
            "maximum_cardinality": maximum_cardinality,
            "maximum_cardinality_certified": True,
            "local_infeasible_certified": True,
            "solutions": [],
            "collision_edges": [list(edge) for edge in collision_edges],
            "static_conflict_edges": [
                [candidate_ids[left], candidate_ids[right]]
                for left, right in sorted(static_edges)
            ],
            "solve_records": solve_records,
            "search_complete": True,
            "solution_limit_reached": False,
            "timeout_observed": False,
            "global_packing_claim": False,
        }

    equality = lil_matrix((1, len(ordered)), dtype=float)
    equality[0, :] = 1.0
    active_matrix = vstack([base, equality.tocsr()]).tocsr()
    active_lower = np.concatenate([base_lower, [float(target_count)]])
    active_upper = np.concatenate([base_upper, [float(target_count)]])
    solutions = []
    enumeration_exhausted = False
    while len(solutions) < max_solutions and time.perf_counter() - started < limit:
        solve_started = time.perf_counter()
        result = milp(
            c=np.zeros(len(ordered)),
            integrality=np.ones(len(ordered)),
            bounds=Bounds(0.0, 1.0),
            constraints=LinearConstraint(
                active_matrix, active_lower, active_upper
            ),
            options={"time_limit": remaining_limit(), "mip_rel_gap": 0.0},
        )
        record = _repair_milp_record(
            result,
            purpose="target_feasibility_enumeration",
            elapsed=time.perf_counter() - solve_started,
            time_limit_seconds=limit,
            objective_sign=1,
        )
        solve_records.append(record)
        if record["termination"] == "certified_local_model_infeasible":
            enumeration_exhausted = True
            break
        if record["termination"] != "certified_feasible_solution":
            break
        rounded, validation = _validate_local_binary_milp_solution(
            result,
            objective_coefficients=np.zeros(len(ordered)),
            constraint_matrix=active_matrix,
            constraint_lower=active_lower,
            constraint_upper=active_upper,
            label="Local feasibility enumeration MILP",
        )
        selected_positions = np.flatnonzero(rounded > 0.5).tolist()
        selected_ids = [candidate_ids[position] for position in selected_positions]
        if len(selected_ids) != target_count:
            raise RuntimeError(
                "Local feasibility enumeration violates the exact target cardinality"
            )
        prior_fingerprints = {
            solution["solution_fingerprint"] for solution in solutions
        }
        fingerprint = _payload_sha256(selected_ids)
        if fingerprint in prior_fingerprints:
            raise RuntimeError(
                "Local feasibility enumeration returned a duplicate solution "
                "despite active no-good rows"
            )
        validation.update(
            {
                "exact_target_cardinality_satisfied": True,
                "active_no_good_row_count": len(solutions),
                "unique_against_prior_solutions": True,
            }
        )
        record["solution_validation"] = validation
        solutions.append(
            {
                "solution_number": len(solutions) + 1,
                "candidate_ids": selected_ids,
                "solution_fingerprint": fingerprint,
                "exact_target_count": True,
            }
        )
        nogood = lil_matrix((1, len(ordered)), dtype=float)
        for position in selected_positions:
            nogood[0, position] = 1.0
        active_matrix = vstack([active_matrix, nogood.tocsr()]).tocsr()
        active_lower = np.concatenate([active_lower, [-np.inf]])
        active_upper = np.concatenate(
            [active_upper, [float(target_count - 1)]]
        )
    limit_reached = len(solutions) >= max_solutions
    search_complete = bool(enumeration_exhausted and not limit_reached)
    return {
        "candidate_count": len(ordered),
        "target_count": target_count,
        "maximum_cardinality": maximum_cardinality,
        "maximum_cardinality_certified": True,
        "local_infeasible_certified": False,
        "solutions": solutions,
        "collision_edges": [list(edge) for edge in collision_edges],
        "static_conflict_edges": [
            [candidate_ids[left], candidate_ids[right]]
            for left, right in sorted(static_edges)
        ],
        "constraint_counts": {
            "site_capacity": sum(kind[0] == "site_capacity" for kind in row_kinds),
            "actual_metal_capacity": sum(
                kind[0] == "actual_metal_capacity" for kind in row_kinds
            ),
            "exact_sam_collision": len(collision_edges),
            "explicit_static_conflict": len(static_edges),
        },
        "solve_records": solve_records,
        "search_complete": search_complete,
        "solution_limit_reached": limit_reached,
        "timeout_observed": any(record["timeout"] for record in solve_records),
        "global_packing_claim": False,
    }


def _apply_repair_action(placed, remove_ids, additions):
    remove = {str(value) for value in remove_ids}
    if len(remove) != len(remove_ids):
        raise ValueError("Removed repair candidate IDs must be unique")
    indices = [
        index
        for index, candidate in enumerate(placed)
        if str(candidate["candidate_id"]) in remove
    ]
    if len(indices) != len(remove):
        raise ValueError("Removed repair candidate IDs are absent or ambiguous")
    insertion = min(indices)
    external = [
        candidate
        for candidate in placed
        if str(candidate["candidate_id"]) not in remove
    ]
    before = sum(index < insertion for index in range(len(placed)) if str(placed[index]["candidate_id"]) not in remove)
    ordered_additions = sorted(additions, key=lambda candidate: str(candidate["candidate_id"]))
    return external[:before] + ordered_additions + external[before:]


def _validate_accepted_repair_constraints(
    placed,
    *,
    metal_constraints,
    site_conflicts,
    cell,
    periodic_axes,
    thresholds,
):
    site_ids = [str(candidate["site_instance_id"]) for candidate in placed]
    if len(site_ids) != len(set(site_ids)):
        return False
    occupied_metals = set()
    for candidate in placed:
        metals = _candidate_actual_metal_ids(
            candidate, metal_constraints=metal_constraints
        )
        if occupied_metals.intersection(metals):
            return False
        occupied_metals.update(metals)
    occupied_sites = set(site_ids)
    for site_id in occupied_sites:
        if occupied_sites.intersection(
            {str(value) for value in (site_conflicts or {}).get(site_id, ())}
        ):
            return False
    for left_index, left in enumerate(placed):
        collides, _ = _candidate_collides_with_any(
            left,
            placed[left_index + 1 :],
            cell=cell,
            periodic_axes=periodic_axes,
            collision_thresholds_A=thresholds,
        )
        if collides:
            return False
    return True


def evaluate_local_repair_removal_set(
    *,
    placed,
    removed_candidate_ids,
    local_candidates,
    projection_cache,
    metal_constraints,
    site_conflicts,
    cell,
    periodic_axes,
    collision_thresholds_A,
    max_solutions,
    time_limit_seconds,
    area_tolerance_A2=1.0e-6,
) -> dict:
    """Evaluate one fixed-external k->k+1 local repair MILP and Pareto set."""

    thresholds = _validate_collision_thresholds(collision_thresholds_A)
    remove_ids = [str(value) for value in removed_candidate_ids]
    removed = [
        candidate
        for candidate in placed
        if str(candidate["candidate_id"]) in set(remove_ids)
    ]
    if len(removed) != len(remove_ids) or len(set(remove_ids)) != len(remove_ids):
        raise ValueError("Removal set candidate IDs are missing or ambiguous")
    external = [
        candidate
        for candidate in placed
        if str(candidate["candidate_id"]) not in set(remove_ids)
    ]
    external_hash = _external_placement_hash(external)
    external_sites = {str(candidate["site_instance_id"]) for candidate in external}
    external_metals = set().union(
        *(
            _candidate_actual_metal_ids(
                candidate, metal_constraints=metal_constraints
            )
            for candidate in external
        )
    ) if external else set()
    compatible = []
    rejected_site = []
    rejected_metal = []
    rejected_collision = []
    for candidate in sorted(local_candidates, key=lambda item: str(item["candidate_id"])):
        candidate_id = str(candidate["candidate_id"])
        site_id = str(candidate["site_instance_id"])
        if site_id in external_sites or external_sites.intersection(
            {str(value) for value in (site_conflicts or {}).get(site_id, ())}
        ):
            rejected_site.append(candidate_id)
            continue
        metals = _candidate_actual_metal_ids(
            candidate, metal_constraints=metal_constraints
        )
        if metals.intersection(external_metals):
            rejected_metal.append(candidate_id)
            continue
        collides, _ = _candidate_collides_with_any(
            candidate,
            external,
            cell=cell,
            periodic_axes=periodic_axes,
            collision_thresholds_A=thresholds,
        )
        if collides:
            rejected_collision.append(candidate_id)
            continue
        compatible.append(candidate)
    solver = enumerate_local_repair_solutions(
        candidates=compatible,
        target_count=len(remove_ids) + 1,
        metal_constraints=metal_constraints,
        site_conflicts=site_conflicts,
        cell=cell,
        periodic_axes=periodic_axes,
        collision_thresholds_A=thresholds,
        max_solutions=max_solutions,
        time_limit_seconds=time_limit_seconds,
    )
    compatible_by_id = {
        str(candidate["candidate_id"]): candidate for candidate in compatible
    }
    before = repair_objective(placed, projection_cache)
    evaluated = []
    pareto = []
    for solution in solver["solutions"]:
        additions = [
            compatible_by_id[candidate_id]
            for candidate_id in solution["candidate_ids"]
        ]
        trial = _apply_repair_action(placed, remove_ids, additions)
        constraints_passed = _validate_accepted_repair_constraints(
            trial,
            metal_constraints=metal_constraints,
            site_conflicts=site_conflicts,
            cell=cell,
            periodic_axes=periodic_axes,
            thresholds=thresholds,
        )
        if not constraints_passed:
            raise RuntimeError("MILP repair solution failed exact post-solve constraints")
        after = repair_objective(trial, projection_cache)
        fingerprint = _repair_action_fingerprint(
            f"{len(remove_ids)}_to_{len(additions)}_local_repair",
            remove_ids,
            solution["candidate_ids"],
        )
        action = {
            "action_type": f"{len(remove_ids)}_to_{len(additions)}_local_repair",
            "removed_count_k": len(remove_ids),
            "target_add_count": len(additions),
            "remove_candidate_ids": sorted(remove_ids),
            "add_candidate_ids": sorted(solution["candidate_ids"]),
            "action_fingerprint": fingerprint,
            "fixed_external_sha256": external_hash,
            "objective_before": _repair_objective_summary(before),
            "objective_after": _repair_objective_summary(after),
            "strict_pareto": strict_repair_pareto(
                before,
                after,
                area_tolerance_A2=area_tolerance_A2,
            ),
            "exact_constraints_passed": True,
            "thresholds_A": thresholds,
            "area_tolerance_A2": float(area_tolerance_A2),
            "weighted_score_used": False,
            "solver_solution_fingerprint": solution["solution_fingerprint"],
            "_trial": trial,
        }
        evaluated.append(action)
        if action["strict_pareto"]:
            pareto.append(action)
    pareto.sort(
        key=lambda action: (
            -int(action["objective_after"]["molecule_count_N"]),
            float(action["objective_after"]["total_hole_area_A2"]),
            float(action["objective_after"]["maximum_hole_area_A2"]),
            str(action["action_fingerprint"]),
        )
    )
    best = pareto[0] if pareto else None
    return {
        "removed_count_k": len(remove_ids),
        "target_add_count": len(remove_ids) + 1,
        "remove_candidate_ids": sorted(remove_ids),
        "fixed_external_sha256": external_hash,
        "external_compatibility": {
            "raw_candidate_count": len(local_candidates),
            "compatible_candidate_ids": [
                str(candidate["candidate_id"]) for candidate in compatible
            ],
            "rejected_occupied_site_candidate_ids": rejected_site,
            "rejected_external_metal_candidate_ids": rejected_metal,
            "rejected_collision_candidate_ids": rejected_collision,
            "substrate_contract_source": "current_domain_prevalidated",
        },
        "solver": solver,
        "evaluated_solutions": [
            {key: value for key, value in action.items() if key != "_trial"}
            for action in evaluated
        ],
        "all_evaluated_pareto_actions": [
            {key: value for key, value in action.items() if key != "_trial"}
            for action in pareto
        ],
        "best_action": (
            None
            if best is None
            else {key: value for key, value in best.items() if key != "_trial"}
        ),
        "best_placed": None if best is None else best["_trial"],
        "search_complete": bool(solver["search_complete"]),
        "global_packing_claim": False,
    }


def _repair_neighborhood_scope(
    *,
    placed,
    domains,
    projection_cache,
    neighborhood_radius_A,
    max_frontier,
    max_local_candidates,
) -> dict:
    """Select a deterministic bounded physical neighborhood of the largest hole."""

    radius = float(neighborhood_radius_A)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("repair neighborhood radius must be finite and positive")
    for name, value in (
        ("max_frontier", max_frontier),
        ("max_local_candidates", max_local_candidates),
    ):
        if isinstance(value, bool) or int(value) <= 0:
            raise ValueError(f"{name} must be a positive integer")
    objective = repair_objective(placed, projection_cache)
    components = objective["torus_component_geometries"]
    if not components:
        return {
            "status": "no_positive_area_torus_hole",
            "objective": _repair_objective_summary(objective),
            "largest_hole": None,
            "frontier_records": [],
            "frontier_candidates": [],
            "local_candidate_records": [],
            "local_candidates": [],
            "search_complete": True,
            "truncation": {
                "frontier_truncated": False,
                "local_candidates_truncated": False,
            },
        }
    hole = components[0]
    hole_geometry = hole["geometry"]
    hole_boundary = hole_geometry.boundary
    lattice = projection_cache.surface_lattice_uv_A
    frontier_records = []
    for candidate in placed:
        footprint = projection_cache.candidate_geometries[
            int(candidate["candidate_index"])
        ]
        relation = periodic_physical_geometry_metrics(
            footprint.boundary, hole_boundary, lattice
        )
        distance = float(relation["minimum_distance_A"])
        if distance <= radius:
            frontier_records.append(
                {
                    "candidate_id": str(candidate["candidate_id"]),
                    "site_instance_id": str(candidate["site_instance_id"]),
                    "footprint_to_hole_boundary_distance_A": distance,
                    "_candidate": candidate,
                }
            )
    frontier_records.sort(
        key=lambda record: (
            record["footprint_to_hole_boundary_distance_A"],
            record["candidate_id"],
        )
    )
    raw_frontier_count = len(frontier_records)
    selected_frontier = frontier_records[: int(max_frontier)]

    local_records = []
    for candidates in domains.values():
        for candidate in candidates:
            footprint = projection_cache.candidate_geometries[
                int(candidate["candidate_index"])
            ]
            footprint_relation = periodic_physical_geometry_metrics(
                footprint, hole_geometry, lattice
            )
            frame = candidate.get("surface_frame")
            surface_frame_coordinates(np.zeros((1, 3)), frame)
            anchor_uv = surface_frame_coordinates(
                np.asarray(candidate["anchor_cartesian_A"], dtype=float)[None, :],
                frame,
            )[0, :2]
            fractional = np.linalg.solve(lattice, anchor_uv)
            fractional = np.mod(fractional, 1.0)
            _, Point, _, _, _ = _require_shapely()
            anchor_point = Point(float(fractional[0]), float(fractional[1]))
            anchor_relation = periodic_physical_geometry_metrics(
                anchor_point, hole_boundary, lattice
            )
            footprint_distance = float(footprint_relation["minimum_distance_A"])
            anchor_distance = float(anchor_relation["minimum_distance_A"])
            footprint_intersects_hole = bool(
                footprint_relation["intersects"]
                or footprint_relation["intersection_area_A2"] > 0.0
            )
            qualifies = (
                footprint_intersects_hole
                or footprint_distance <= radius
                or anchor_distance <= radius
            )
            if not qualifies:
                continue
            local_records.append(
                {
                    "candidate_id": str(candidate["candidate_id"]),
                    "site_instance_id": str(candidate["site_instance_id"]),
                    "footprint_intersects_largest_hole": footprint_intersects_hole,
                    "footprint_to_largest_hole_distance_A": footprint_distance,
                    "anchor_to_hole_boundary_distance_A": anchor_distance,
                    "selection_distance_A": min(
                        footprint_distance, anchor_distance
                    ),
                    "_candidate": candidate,
                }
            )
    local_records.sort(
        key=lambda record: (
            record["selection_distance_A"],
            record["footprint_to_largest_hole_distance_A"],
            record["anchor_to_hole_boundary_distance_A"],
            record["candidate_id"],
        )
    )
    raw_local_count = len(local_records)
    selected_local = local_records[: int(max_local_candidates)]
    frontier_truncated = raw_frontier_count > len(selected_frontier)
    local_truncated = raw_local_count > len(selected_local)
    return {
        "status": "selected_largest_torus_hole_neighborhood",
        "objective": _repair_objective_summary(objective),
        "largest_hole": {
            "component_id": hole["component_id"],
            "area_A2": float(hole["area_A2"]),
            "perimeter_A": float(hole["perimeter_A"]),
            "planar_fragment_count": int(hole["planar_fragment_count"]),
            "maximum_matches_periodic_hole_diagnostics_exactly": (
                float(hole["area_A2"])
                == float(objective["maximum_hole_area_A2"])
            ),
        },
        "neighborhood_radius_A": radius,
        "frontier_raw_count": raw_frontier_count,
        "frontier_selected_count": len(selected_frontier),
        "frontier_records": [
            {key: value for key, value in record.items() if key != "_candidate"}
            for record in selected_frontier
        ],
        "frontier_candidates": [record["_candidate"] for record in selected_frontier],
        "local_candidate_raw_count": raw_local_count,
        "local_candidate_selected_count": len(selected_local),
        "local_candidate_records": [
            {key: value for key, value in record.items() if key != "_candidate"}
            for record in selected_local
        ],
        "local_candidates": [record["_candidate"] for record in selected_local],
        "truncation": {
            "frontier_truncated": frontier_truncated,
            "local_candidates_truncated": local_truncated,
            "frontier_limit": int(max_frontier),
            "local_candidate_limit": int(max_local_candidates),
        },
        "search_complete": not frontier_truncated and not local_truncated,
        "selected_local_scope_no_action_statement_allowed": (
            not frontier_truncated and not local_truncated
        ),
        "global_no_action_proof": False,
    }


def run_bounded_local_cardinality_repair(
    *,
    placed,
    domains,
    projection_cache,
    metal_constraints,
    site_conflicts,
    cell,
    periodic_axes,
    collision_thresholds_A,
    max_k,
    neighborhood_radius_A,
    max_frontier,
    max_local_candidates,
    max_solutions,
    time_limit_seconds,
    max_fixed_replacements,
    area_tolerance_A2,
) -> dict:
    """Run bounded 1->2 then necessary k->k+1 local repair until termination."""

    if isinstance(max_k, bool) or int(max_k) <= 0:
        raise ValueError("repair max_k must be a positive integer")
    maximum_k = int(max_k)
    limit = float(time_limit_seconds)
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("repair time limit must be finite and positive")
    thresholds = _validate_collision_thresholds(collision_thresholds_A)
    started = time.perf_counter()
    current = list(placed)
    accepted_actions = []
    all_pareto = []
    search_passes = []
    solver_documents = []
    post_insertion_fixed_passes = []
    search_complete = True
    timeout_observed = False
    termination = None
    pass_number = 0
    while True:
        elapsed = time.perf_counter() - started
        if elapsed >= limit:
            timeout_observed = True
            search_complete = False
            termination = "repair_time_limit_reached_not_proof"
            break
        capacity_gate = certified_static_repair_gate(
            placed=current,
            domains=domains,
            metal_constraints=metal_constraints,
            site_conflicts=site_conflicts,
            time_limit_seconds=max(1.0e-9, limit - elapsed),
        )
        if not capacity_gate["run_k_to_k_plus_one"]:
            termination = "skipped_proven_static_cardinality_bound"
            break
        pass_number += 1
        scope = _repair_neighborhood_scope(
            placed=current,
            domains=domains,
            projection_cache=projection_cache,
            neighborhood_radius_A=neighborhood_radius_A,
            max_frontier=max_frontier,
            max_local_candidates=max_local_candidates,
        )
        search_complete &= bool(scope["search_complete"])
        pass_record = {
            "pass": pass_number,
            "capacity_gate": capacity_gate,
            "neighborhood": {
                key: value
                for key, value in scope.items()
                if key not in {"frontier_candidates", "local_candidates"}
            },
            "k_searches": [],
            "accepted_action_fingerprint": None,
        }
        if not scope["frontier_candidates"] or not scope["local_candidates"]:
            termination = (
                "bounded_neighborhood_empty_not_proof"
                if not scope["search_complete"]
                else "complete_selected_neighborhood_has_no_removal_or_candidate"
            )
            search_passes.append(pass_record)
            break
        accepted_result = None
        accepted_action = None
        for k in range(1, maximum_k + 1):
            if len(scope["frontier_candidates"]) < k:
                pass_record["k_searches"].append(
                    {
                        "removed_count_k": k,
                        "status": "skipped_insufficient_selected_frontier",
                        "removal_set_count": 0,
                        "pareto_action_count": 0,
                    }
                )
                continue
            removal_sets = list(
                itertools.combinations(scope["frontier_candidates"], k)
            )
            k_results = []
            k_pareto = []
            k_complete = True
            for removal_set_index, removal_set in enumerate(removal_sets, 1):
                remaining = limit - (time.perf_counter() - started)
                if remaining <= 0.0:
                    timeout_observed = True
                    search_complete = False
                    k_complete = False
                    break
                result = evaluate_local_repair_removal_set(
                    placed=current,
                    removed_candidate_ids=[
                        candidate["candidate_id"] for candidate in removal_set
                    ],
                    local_candidates=scope["local_candidates"],
                    projection_cache=projection_cache,
                    metal_constraints=metal_constraints,
                    site_conflicts=site_conflicts,
                    cell=cell,
                    periodic_axes=periodic_axes,
                    collision_thresholds_A=thresholds,
                    max_solutions=max_solutions,
                    time_limit_seconds=remaining,
                    area_tolerance_A2=area_tolerance_A2,
                )
                solver_document = {
                    "schema": "sam-local-repair-solver-audit-v1",
                    "pass": pass_number,
                    "removed_count_k": k,
                    "removal_set_number": removal_set_index,
                    "remove_candidate_ids": result["remove_candidate_ids"],
                    "fixed_external_sha256": result["fixed_external_sha256"],
                    "external_compatibility": result["external_compatibility"],
                    "solver": result["solver"],
                    "search_scope": {
                        "largest_hole_component_id": scope["largest_hole"][
                            "component_id"
                        ],
                        "frontier_raw_count": scope["frontier_raw_count"],
                        "frontier_selected_count": scope["frontier_selected_count"],
                        "local_candidate_raw_count": scope[
                            "local_candidate_raw_count"
                        ],
                        "local_candidate_selected_count": scope[
                            "local_candidate_selected_count"
                        ],
                        "scope_search_complete": scope["search_complete"],
                    },
                    "global_packing_claim": False,
                }
                solver_documents.append(solver_document)
                k_results.append(result)
                k_complete &= bool(result["search_complete"])
                timeout_observed |= bool(result["solver"]["timeout_observed"])
                for action in result["all_evaluated_pareto_actions"]:
                    k_pareto.append(action)
                    all_pareto.append(action)
            search_complete &= k_complete
            k_pareto.sort(
                key=lambda action: (
                    -int(action["objective_after"]["molecule_count_N"]),
                    float(action["objective_after"]["total_hole_area_A2"]),
                    float(action["objective_after"]["maximum_hole_area_A2"]),
                    str(action["action_fingerprint"]),
                )
            )
            k_record = {
                "removed_count_k": k,
                "target_add_count": k + 1,
                "status": (
                    "strict_pareto_action_found"
                    if k_pareto
                    else (
                        "bounded_no_action_not_proof"
                        if not (scope["search_complete"] and k_complete)
                        else "no_strict_pareto_action_in_complete_selected_local_models"
                    )
                ),
                "removal_set_count": len(removal_sets),
                "evaluated_removal_set_count": len(k_results),
                "pareto_action_count": len(k_pareto),
                "all_evaluated_pareto_actions": k_pareto,
                "search_complete": bool(scope["search_complete"] and k_complete),
            }
            pass_record["k_searches"].append(k_record)
            if k_pareto:
                winner_fingerprint = k_pareto[0]["action_fingerprint"]
                accepted_result = next(
                    result
                    for result in k_results
                    if result["best_action"] is not None
                    and result["best_action"]["action_fingerprint"]
                    == winner_fingerprint
                )
                accepted_action = dict(k_pareto[0])
                # Strict priority: accept 1->2 when available; only search 2->3
                # or larger k when every smaller k has no accepted action.
                break
        if accepted_result is None:
            termination = (
                "bounded_search_no_accepted_action_not_proof"
                if not search_complete
                else "no_strict_pareto_action_in_complete_selected_local_scope"
            )
            search_passes.append(pass_record)
            break
        current = list(accepted_result["best_placed"])
        accepted_actions.append(accepted_action)
        pass_record["accepted_action_fingerprint"] = accepted_action[
            "action_fingerprint"
        ]
        search_passes.append(pass_record)
        fixed = run_fixed_site_replacements(
            placed=current,
            domains=domains,
            projection_cache=projection_cache,
            cell=cell,
            periodic_axes=periodic_axes,
            collision_thresholds_A=thresholds,
            max_replacements=max_fixed_replacements,
            area_tolerance_A2=area_tolerance_A2,
            metal_constraints=metal_constraints,
        )
        current = list(fixed["placed"])
        post_insertion_fixed_passes.append(fixed)
        accepted_actions.extend(fixed["accepted_actions"])
        all_pareto.extend(fixed["all_observed_pareto_actions"])
        # The next while iteration recomputes both certified K and the largest
        # torus hole from the fully rebuilt accepted geometry list.
    final_objective = repair_objective(current, projection_cache)
    return {
        "status": termination,
        "placed": current,
        "objective_after": _repair_objective_summary(final_objective),
        "accepted_actions": accepted_actions,
        "all_evaluated_pareto_actions": all_pareto,
        "search_passes": search_passes,
        "solver_documents": solver_documents,
        "post_insertion_fixed_passes": post_insertion_fixed_passes,
        "search_complete": bool(search_complete and not timeout_observed),
        "timeout_observed": timeout_observed,
        "termination": termination,
        "optimization_scope": (
            "bounded_largest_torus_hole_local_k_to_k_plus_one_milp"
        ),
        "global_packing_claim": False,
        "global_optimality_claim": False,
        "weighted_score_used": False,
        "thresholds_A": thresholds,
        "wall_time_seconds": time.perf_counter() - started,
    }


def repair_output_contract() -> dict:
    return {
        "role": "postprocessed_rearranged_h0",
        "irreversible_sequential_history": False,
        "trajectory_role_allowed": False,
        "global_packing_claim": False,
        "chemistry_scope": "H0_geometry_only_surface_protons_deferred",
        "collision_threshold_adaptation": "forbidden",
        "weighted_score_used": False,
    }


def repair_acceptance_gates(objective, references, *, area_tolerance_A2=1.0e-6):
    """Evaluate original R0 N>=97 baseline and strong corrected torus gates."""

    tolerance = float(area_tolerance_A2)
    definitions = references.get("future_A1_gate_definitions", references)
    result = {
        "R0_original_minimum_molecule_count": 97,
        "metric_version": "periodic-torus-holes-v2",
        "area_tolerance_A2": tolerance,
        "weighted_score_used": False,
    }
    for name in ("primary", "strong"):
        reference = dict(definitions[name]["reference"])
        checks = {
            "molecule_count_N_at_least_97": int(objective["molecule_count_N"]) >= 97,
            "total_hole_area_nonworse": float(objective["total_hole_area_A2"])
            <= float(reference["total_hole_area_A2"]) + tolerance,
            "maximum_hole_area_nonworse": float(objective["maximum_hole_area_A2"])
            <= float(reference["maximum_hole_area_A2"]) + tolerance,
        }
        result[name] = {
            "reference": reference,
            "checks": checks,
            "passed": all(checks.values()),
        }
    return result


def audit_final_registered_interfaces(
    *,
    assembled,
    substrate_atom_count,
    placed,
    periodic_axes=(0, 1),
    radius_scale=0.85,
) -> dict:
    """Independently re-audit accepted interfaces from final assembled coordinates.

    This library seam is intentionally lifecycle-neutral: 1C-b can serialize
    its returned exact global-0-based bond records into ``validation.json``
    without trusting candidate-generation coordinates.
    """

    axes = _validated_surface_axes(periodic_axes)
    if (
        not isinstance(radius_scale, (int, float))
        or not np.isfinite(radius_scale)
        or float(radius_scale) <= 0.0
    ):
        raise ValueError("radius_scale must be finite and positive")
    radius_scale = float(radius_scale)
    if (
        isinstance(substrate_atom_count, (bool, np.bool_))
        or not isinstance(substrate_atom_count, (int, np.integer))
    ):
        raise ValueError("substrate_atom_count must be an integer")
    substrate_atom_count = int(substrate_atom_count)
    if not 0 < substrate_atom_count <= len(assembled):
        raise ValueError("substrate_atom_count is outside the assembled structure")
    if not isinstance(placed, (list, tuple)):
        raise TypeError("placed mappings must be a list or tuple")
    symbols = np.asarray(assembled.get_chemical_symbols(), dtype=str)
    positions = np.asarray(assembled.positions, dtype=float)
    substrate_symbols = symbols[:substrate_atom_count]
    substrate_positions = positions[:substrate_atom_count]
    expected_atom_count = substrate_atom_count + sum(
        len(candidate.get("symbols", ())) for candidate in placed
    )
    if expected_atom_count != len(assembled):
        raise ValueError(
            "Assembled atom count does not match substrate plus placed molecule blocks"
        )

    site_ids = []
    metal_to_placements = defaultdict(list)
    molecule_audits = []
    global_bonds = []
    molecule_offset = substrate_atom_count
    for placement_index, candidate in enumerate(placed):
        if not isinstance(candidate, dict):
            raise TypeError("Each placed mapping must be a candidate dictionary")
        site_id = candidate.get("site_instance_id")
        if not isinstance(site_id, str) or not site_id:
            raise ValueError("Placed candidate lacks a valid Site Instance ID")
        site_ids.append(site_id)
        candidate_symbols = np.asarray(candidate.get("symbols"), dtype=str)
        if candidate_symbols.ndim != 1 or not len(candidate_symbols):
            raise ValueError(f"{site_id}: placed candidate symbols are missing")
        stop = molecule_offset + len(candidate_symbols)
        final_symbols = symbols[molecule_offset:stop]
        if not np.array_equal(final_symbols, candidate_symbols):
            raise ValueError(
                f"{site_id}: final assembled molecule symbols/order changed"
            )
        frame = candidate.get("surface_frame")
        if frame is None:
            raise ValueError(f"{site_id}: placed candidate lacks a Surface Frame")
        surface_frame_coordinates(np.zeros((1, 3)), frame)
        frame_axes = frame.get("periodic_fractional_axes")
        if frame_axes is None:
            raise ValueError(
                f"{site_id}: Surface Frame lacks periodic_fractional_axes"
            )
        if _validated_surface_axes(frame_axes) != axes:
            raise ValueError(f"{site_id}: Surface Frame periodic axes disagree")
        bonds = candidate.get("registered_interface_bonds")
        if not isinstance(bonds, list) or not bonds:
            raise ValueError(f"{site_id}: placed candidate lacks registered interface bonds")
        mapped_windows = {}
        seen_labels = set()
        seen_local_atoms = set()
        placement_metals = set()
        for bond in bonds:
            if not isinstance(bond, dict):
                raise ValueError(f"{site_id}: malformed registered interface bond")
            label = bond.get("donor_label")
            if not isinstance(label, str) or not label or label in seen_labels:
                raise ValueError(f"{site_id}: registered donor labels must be unique")
            seen_labels.add(label)
            molecule_local = bond.get("molecule_atom_index_0based")
            working_metal = bond.get("working_metal_atom_index_0based")
            full_metal = bond.get("full_metal_atom_id_1based")
            for name, value in (
                ("molecule local index", molecule_local),
                ("working metal index", working_metal),
                ("full metal ID", full_metal),
            ):
                if isinstance(value, (bool, np.bool_)) or not isinstance(
                    value, (int, np.integer)
                ):
                    raise ValueError(f"{site_id}: {name} must be an integer")
            molecule_local = int(molecule_local)
            working_metal = int(working_metal)
            full_metal = int(full_metal)
            if not 0 <= molecule_local < len(candidate_symbols):
                raise ValueError(f"{site_id}: molecule local bond index is outside its block")
            if molecule_local in seen_local_atoms:
                raise ValueError(f"{site_id}: a donor atom is registered more than once")
            seen_local_atoms.add(molecule_local)
            if not 0 <= working_metal < substrate_atom_count:
                raise ValueError(f"{site_id}: working metal index is outside substrate")
            molecule_element = str(candidate_symbols[molecule_local])
            substrate_element = str(substrate_symbols[working_metal])
            if molecule_element != bond.get("molecule_element"):
                raise ValueError(f"{site_id}: registered molecule element changed")
            if substrate_element != bond.get("actual_metal_element"):
                raise ValueError(f"{site_id}: registered actual metal element changed")
            window = _validated_registered_bond_window(
                bond.get("distance_window_A")
            )
            pair = (molecule_local, working_metal)
            if pair in mapped_windows:
                raise ValueError(f"{site_id}: duplicate registered atom pair")
            mapped_windows[pair] = window
            placement_metals.add(full_metal)
            global_bonds.append(
                {
                    "substrate_atom_index_0based": working_metal,
                    "sam_atom_index_0based": molecule_offset + molecule_local,
                    "minimum_distance_A": window[0],
                    "maximum_distance_A": window[1],
                    "expected_substrate_element": substrate_element,
                    "expected_sam_element": molecule_element,
                    "site_instance_id": site_id,
                    "donor_label": label,
                    "full_metal_atom_id_1based": full_metal,
                }
            )
        for full_metal in placement_metals:
            metal_to_placements[full_metal].append(
                {
                    "placement_index": placement_index,
                    "site_instance_id": site_id,
                }
            )
        radii = _resolved_ase_vdw_radii(
            list(final_symbols) + list(substrate_symbols)
        )
        audit = periodic_vdw_collision_audit(
            molecule_positions=positions[molecule_offset:stop],
            molecule_symbols=final_symbols,
            molecule_atom_ids=list(range(len(final_symbols))),
            substrate_positions=substrate_positions,
            substrate_symbols=substrate_symbols,
            substrate_atom_ids=list(range(substrate_atom_count)),
            cell=assembled.cell,
            periodic_axes=axes,
            surface_frame=frame,
            radii_A=radii,
            radius_scale=radius_scale,
            mapped_bond_windows_A=mapped_windows,
        )
        molecule_audits.append(
            {
                **audit,
                "molecule_number_1based": placement_index + 1,
                "site_instance_id": site_id,
                "global_atom_range_0based": [molecule_offset, stop - 1],
            }
        )
        molecule_offset = stop

    unique_sites = len(site_ids) == len(set(site_ids))
    metal_capacity_violations = [
        {
            "full_metal_atom_id_1based": full_metal,
            "placements": placements,
            "capacity": 1,
            "occupancy": len(placements),
        }
        for full_metal, placements in sorted(metal_to_placements.items())
        if len(placements) > 1
    ]
    all_audits_passed = all(audit["passed"] for audit in molecule_audits)
    checks = {
        "unique_site_instances": unique_sites,
        "actual_metal_capacity_one": not metal_capacity_violations,
        "all_registered_interface_audits_passed": all_audits_passed,
    }
    return {
        "schema": "sam-sequential-final-registered-interface-audit-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "substrate_atom_count": substrate_atom_count,
        "molecule_count": len(placed),
        "radius_scale": float(radius_scale),
        "radii_source": "ASE ase.data.vdw_radii",
        "periodic_axes": list(axes),
        "normal_axis_wrapped": False,
        "registered_interface_bonds": global_bonds,
        "molecule_audits": molecule_audits,
        "metal_capacity_violations": metal_capacity_violations,
        "duplicate_site_instance_ids": sorted(
            site_id for site_id, count in Counter(site_ids).items() if count > 1
        ),
    }


def audit_final_height_clearance(
    *,
    assembled,
    substrate_atom_count,
    placed,
    height_clearance_A=1.0,
    periodic_axes=(0, 1),
    collision_thresholds_A=None,
    full_to_working_atom_map=None,
    allowed_metal_elements=("In", "Sn"),
) -> dict:
    """Recheck non-anchor SAM height gate, single-metal triangle enclosure, metal capacity, and mixed SAM-SAM collisions from final coordinates."""

    substrate_atom_count = int(substrate_atom_count)
    if not 0 < substrate_atom_count <= len(assembled):
        raise ValueError("substrate_atom_count is outside the assembled structure")
    symbols = np.asarray(assembled.get_chemical_symbols(), dtype=str)
    positions = np.asarray(assembled.positions, dtype=float)
    cell = np.asarray(assembled.cell, dtype=float)
    axes = _validated_surface_axes(periodic_axes)
    pbc = np.zeros(3, dtype=bool)
    pbc[list(axes)] = True
    thresholds = _validate_collision_thresholds(collision_thresholds_A)
    substrate_symbols = symbols[:substrate_atom_count]
    substrate_positions = positions[:substrate_atom_count]
    audits = []
    site_ids = []
    metal_to_sites = defaultdict(list)
    offsets = []
    offset = substrate_atom_count
    for molecule_number, candidate in enumerate(placed, start=1):
        candidate_symbols = np.asarray(candidate.get("symbols"), dtype=str)
        stop = offset + len(candidate_symbols)
        if stop > len(assembled) or not np.array_equal(
            symbols[offset:stop], candidate_symbols
        ):
            raise ValueError("Final assembled molecule symbols/order changed")
        offsets.append((offset, stop))
        site_id = candidate.get("site_instance_id")
        if not isinstance(site_id, str) or not site_id:
            raise ValueError("Placed candidate lacks a valid Site Instance ID")
        site_ids.append(site_id)

        mol_pos = positions[offset:stop]
        mol_sym = symbols[offset:stop]
        frame = candidate["surface_frame"]
        anchor_indices = candidate["headgroup_local_indices_0based"]

        # 1. Height clearance audit on non-anchor atoms
        audit = _height_substrate_clearance_audit(
            molecule_positions=mol_pos,
            molecule_symbols=mol_sym,
            substrate_positions=substrate_positions,
            substrate_symbols=substrate_symbols,
            surface_frame=frame,
            anchor_group_atom_indices_0based=anchor_indices,
            height_clearance_A=height_clearance_A,
        )

        # 2. Single-metal triangle enclosure check if candidate is single_metal_triangle
        single_metal_audit = None
        if candidate.get("site_kind") == "single_metal_triangle":
            registered_bonds = candidate.get("registered_interface_bonds", [])
            registered_bonds_empty = (len(registered_bonds) == 0)

            anchor_metal_fid = candidate.get("anchor_metal_atom_id_1based")
            if anchor_metal_fid is None and candidate.get("occupied_metal_ids"):
                anchor_metal_fid = int(candidate["occupied_metal_ids"][0])

            occupied_metal_ids = list(candidate.get("occupied_metal_ids", []))
            occupied_metal_ids_valid = (
                anchor_metal_fid is not None
                and len(occupied_metal_ids) == 1
                and int(occupied_metal_ids[0]) == int(anchor_metal_fid)
            )

            mapping_valid = True
            element_valid = True
            w_idx = None
            allowed_metals = {str(e) for e in allowed_metal_elements}
            if anchor_metal_fid is None or anchor_metal_fid < 1:
                mapping_valid = False
            else:
                anchor_metal_fid = int(anchor_metal_fid)
                if full_to_working_atom_map is not None:
                    if anchor_metal_fid not in full_to_working_atom_map:
                        mapping_valid = False
                    else:
                        rec = full_to_working_atom_map[anchor_metal_fid]
                        w_idx = int(rec["working_atom_index_0based"])
                        rec_actual = rec.get("actual_element")
                        rec_element = rec.get("element")
                        rec_symbol = rec.get("symbol")
                        rec_elem = rec_actual or rec_element or rec_symbol
                        if not 0 <= w_idx < substrate_atom_count:
                            mapping_valid = False
                        if not rec_elem or str(rec_elem) not in allowed_metals:
                            element_valid = False
                        if rec_actual and rec_element and str(rec_actual) != str(rec_element):
                            element_valid = False
                        if rec_elem and str(substrate_symbols[w_idx]) != str(rec_elem):
                            element_valid = False
                        if (
                            "anchor_metal_working_index_0based" not in candidate
                            or int(candidate["anchor_metal_working_index_0based"]) != w_idx
                        ):
                            mapping_valid = False
                        if "anchor_metal_element" not in candidate:
                            element_valid = False
                        else:
                            cand_elem = str(candidate["anchor_metal_element"])
                            if cand_elem not in allowed_metals:
                                element_valid = False
                            if rec_elem and cand_elem != str(rec_elem):
                                element_valid = False
                            if (
                                w_idx is not None
                                and 0 <= w_idx < substrate_atom_count
                                and cand_elem != str(substrate_symbols[w_idx])
                            ):
                                element_valid = False
                else:
                    if "anchor_metal_working_index_0based" in candidate:
                        w_idx = int(candidate["anchor_metal_working_index_0based"])
                        if not 0 <= w_idx < substrate_atom_count:
                            mapping_valid = False
                        if "anchor_metal_element" not in candidate:
                            element_valid = False
                        else:
                            cand_elem = str(candidate["anchor_metal_element"])
                            if cand_elem not in allowed_metals:
                                element_valid = False
                            if (
                                w_idx is not None
                                and 0 <= w_idx < substrate_atom_count
                                and cand_elem != str(substrate_symbols[w_idx])
                            ):
                                element_valid = False
                    else:
                        mapping_valid = False

                if w_idx is not None and 0 <= w_idx < substrate_atom_count:
                    if str(substrate_symbols[w_idx]) not in allowed_metals:
                        element_valid = False

            in_triangle = False
            triangle_uv = None
            metal_uv = None
            if (
                mapping_valid
                and element_valid
                and w_idx is not None
                and 0 <= w_idx < substrate_atom_count
            ):
                r_metal = substrate_positions[w_idx]
                o_locals = [
                    int(idx)
                    for idx in anchor_indices
                    if idx < len(mol_sym) and str(mol_sym[idx]) == "O"
                ]
                if len(o_locals) != 3:
                    o_locals = [
                        int(idx) for idx in candidate.get("oxygen_permutation", [])
                    ]
                if len(o_locals) == 3:
                    o_trans = mol_pos[o_locals]
                    delta_o = o_trans - o_trans[0]
                    delta_m = r_metal - o_trans[0]
                    mic_o, _ = find_mic(delta_o, cell=cell, pbc=pbc)
                    mic_m, _ = find_mic(delta_m[None, :], cell=cell, pbc=pbc)
                    u_axis = np.asarray(frame["u_cartesian_unit"], dtype=float)
                    v_axis = np.asarray(frame["v_cartesian_unit"], dtype=float)
                    triangle_uv = np.column_stack(
                        (
                            np.asarray(mic_o, dtype=float) @ u_axis,
                            np.asarray(mic_o, dtype=float) @ v_axis,
                        )
                    )
                    metal_uv = np.asarray(
                        [
                            np.dot(np.asarray(mic_m[0], dtype=float), u_axis),
                            np.dot(np.asarray(mic_m[0], dtype=float), v_axis),
                        ]
                    )
                    in_triangle = bool(_point_in_triangle_2d(metal_uv, triangle_uv))

            single_metal_passed = bool(
                in_triangle
                and registered_bonds_empty
                and occupied_metal_ids_valid
                and mapping_valid
                and element_valid
            )
            single_metal_audit = {
                "passed": single_metal_passed,
                "anchor_metal_atom_id_1based": anchor_metal_fid,
                "working_atom_index_0based": w_idx,
                "in_triangle": in_triangle,
                "registered_bonds_empty": registered_bonds_empty,
                "occupied_metal_ids_valid": occupied_metal_ids_valid,
                "mapping_valid": mapping_valid,
                "element_valid": element_valid,
                "metal_uv_A": [float(v) for v in metal_uv] if metal_uv is not None else None,
                "triangle_uv_A": (
                    [[float(v) for v in row] for row in triangle_uv]
                    if triangle_uv is not None
                    else None
                ),
            }
            # Track occupied metal
            if anchor_metal_fid is not None:
                metal_to_sites[anchor_metal_fid].append(site_id)
        else:
            # Legacy / multidentate candidate
            distinct_metals = {
                int(bond["full_metal_atom_id_1based"])
                for bond in candidate.get("registered_interface_bonds", [])
                if "full_metal_atom_id_1based" in bond
            }
            if not distinct_metals and candidate.get("occupied_metal_ids"):
                distinct_metals = {int(mid) for mid in candidate["occupied_metal_ids"]}
            for mid in sorted(distinct_metals):
                metal_to_sites[mid].append(site_id)

        audits.append(
            {
                **audit,
                "molecule_number_1based": molecule_number,
                "site_instance_id": site_id,
                "global_atom_range_0based": [offset, stop - 1],
                "single_metal_triangle": single_metal_audit,
            }
        )
        offset = stop
    if offset != len(assembled):
        raise ValueError("Final assembled molecule blocks do not cover the structure")

    capacity_violations = [
        {
            "full_metal_atom_id_1based": metal,
            "site_instance_ids": sites,
            "occupancy": len(sites),
            "capacity": 1,
        }
        for metal, sites in sorted(metal_to_sites.items())
        if len(sites) > 1
    ]

    # Mixed SAM-SAM collision audit
    sam_sam_collision_records = []
    for left_idx, (left_start, left_stop) in enumerate(offsets):
        left_cand = placed[left_idx]
        left_pos = positions[left_start:left_stop]
        left_sym = symbols[left_start:left_stop]
        for right_idx in range(left_idx + 1, len(offsets)):
            right_cand = placed[right_idx]
            right_start, right_stop = offsets[right_idx]
            right_pos = positions[right_start:right_stop]
            right_sym = symbols[right_start:right_stop]
            is_mixed = (
                left_cand.get("site_kind") == "single_metal_triangle"
                or right_cand.get("site_kind") == "single_metal_triangle"
            )
            if is_mixed:
                c_count, min_dist, min_ratio = periodic_xy_sam_sam_collision(
                    left_pos,
                    left_sym,
                    right_pos,
                    right_sym,
                    cell=cell,
                    periodic_axes=axes,
                    thresholds_A=thresholds,
                    surface_frame=left_cand.get("surface_frame") or right_cand.get("surface_frame"),
                )
                if c_count > 0:
                    sam_sam_collision_records.append(
                        {
                            "left_molecule": left_idx + 1,
                            "right_molecule": right_idx + 1,
                            "collision_type": "periodic_xy_sam_sam_collision",
                            "collision_count": c_count,
                            "min_distance_A": min_dist,
                            "min_ratio": min_ratio,
                        }
                    )
            else:
                count, min_dist, min_ratio = surface_collision_report(
                    left_pos,
                    left_sym,
                    right_pos,
                    right_sym,
                    cell=cell,
                    periodic_axes=axes,
                    thresholds_A=thresholds,
                )
                if count > 0:
                    sam_sam_collision_records.append(
                        {
                            "left_molecule": left_idx + 1,
                            "right_molecule": right_idx + 1,
                            "collision_type": "3d_surface_collision",
                            "collision_count": count,
                            "min_distance_A": min_dist,
                            "min_ratio": min_ratio,
                        }
                    )

    checks = {
        "unique_site_instances": len(site_ids) == len(set(site_ids)),
        "actual_metal_capacity_one": not capacity_violations,
        "all_non_anchor_atoms_pass_height_clearance": all(
            audit["passed"] for audit in audits
        ),
        "all_single_metal_triangles_valid": all(
            (audit["single_metal_triangle"] is None or audit["single_metal_triangle"]["passed"])
            for audit in audits
        ),
        "sam_sam_collisions_passed": not sam_sam_collision_records,
    }
    return {
        "schema": "sam-sequential-final-height-clearance-audit-v2",
        "passed": all(checks.values()),
        "mode": "height",
        "checks": checks,
        "height_clearance_A": float(height_clearance_A),
        "checked_molecule_atoms": "all_except_anchor_group",
        "anchor_group_exclusion": (
            "topology_resolved_anchor_P_plus_bonded_headgroup_O"
        ),
        "mapped_pair_exemption": "none",
        "mapped_bond_distance_window_A": None,
        "validation_scope": "geometric_clearance_and_triangle_anchor_only",
        "physical_interface_pass": False,
        "molecule_audits": audits,
        "metal_capacity_violations": capacity_violations,
        "sam_sam_collision_violations": sam_sam_collision_records,
    }
def _formula_from_symbols(symbols) -> dict[str, int]:
    return {
        str(element): int(count)
        for element, count in sorted(Counter(str(value) for value in symbols).items())
    }


def _expected_interface_formula(
    substrate_formula: dict[str, int], molecule_formula: dict[str, int], molecules: int
) -> dict[str, int]:
    total = Counter({str(key): int(value) for key, value in substrate_formula.items()})
    for element, count in molecule_formula.items():
        total[str(element)] += int(count) * int(molecules)
    return {element: count for element, count in sorted(total.items()) if count}


def _readback_structure_identity(atoms) -> str:
    payload = {
        "schema": "ase-ordered-structure-identity-v1",
        "symbols": [str(value) for value in atoms.get_chemical_symbols()],
        "positions_A": np.asarray(atoms.positions, dtype=float).tolist(),
        "cell_A": np.asarray(atoms.cell, dtype=float).tolist(),
        "pbc": [bool(value) for value in atoms.pbc],
    }
    return _payload_sha256(payload)


def audit_final_sam_sam_collisions(
    *,
    assembled,
    substrate_atom_count: int,
    placed: list[dict],
    periodic_axes=(0, 1),
    collision_thresholds_A=None,
) -> dict:
    """Independently recompute every cross-SAM pair under the SAM--SAM gate."""

    axes = _validated_surface_axes(periodic_axes)
    thresholds = _validate_collision_thresholds(collision_thresholds_A)
    symbols = np.asarray(assembled.get_chemical_symbols(), dtype=str)
    positions = np.asarray(assembled.positions, dtype=float)
    offsets = []
    cursor = int(substrate_atom_count)
    for candidate in placed:
        candidate_symbols = np.asarray(candidate.get("symbols"), dtype=str)
        stop = cursor + len(candidate_symbols)
        if stop > len(assembled) or not np.array_equal(symbols[cursor:stop], candidate_symbols):
            raise ValueError("Final SAM block identity changed before SAM--SAM audit")
        offsets.append((cursor, stop))
        cursor = stop
    if cursor != len(assembled):
        raise ValueError("Final SAM block count does not cover the assembled structure")
    pbc = np.zeros(3, dtype=bool)
    pbc[list(axes)] = True
    inverse_cell = np.linalg.inv(np.asarray(assembled.cell, dtype=float))
    records = []
    pair_block_count = 0
    for left_index, (left_start, left_stop) in enumerate(offsets):
        for right_index in range(left_index + 1, len(offsets)):
            right_start, right_stop = offsets[right_index]
            pair_block_count += 1
            raw = positions[left_start:left_stop, None, :] - positions[right_start:right_stop][None, :, :]
            mic, distances = find_mic(raw.reshape((-1, 3)), cell=np.asarray(assembled.cell, dtype=float), pbc=pbc)
            mic = np.asarray(mic, dtype=float).reshape(raw.shape)
            distances = np.asarray(distances, dtype=float).reshape(raw.shape[:2])
            images = np.rint((raw.reshape((-1, 3)) - mic.reshape((-1, 3))) @ inverse_cell).astype(int).reshape(raw.shape)
            if not np.allclose(raw - images @ np.asarray(assembled.cell, dtype=float), mic, atol=1.0e-8, rtol=0.0):
                raise RuntimeError("Final SAM--SAM MIC image cannot be reconstructed")
            nonperiodic = [axis for axis in range(3) if axis not in axes]
            if np.any(images[..., nonperiodic] != 0):
                raise RuntimeError("Final SAM--SAM MIC wrapped the normal axis")
            left_symbols = symbols[left_start:left_stop]
            right_symbols = symbols[right_start:right_stop]
            left_h = left_symbols[:, None] == "H"
            right_h = right_symbols[None, :] == "H"
            pair_thresholds = np.where(
                left_h & right_h,
                thresholds["H-H"],
                np.where(left_h | right_h, thresholds["H-heavy"], thresholds["heavy-heavy"]),
            )
            for local_left, local_right in np.argwhere(distances < pair_thresholds):
                local_left = int(local_left)
                local_right = int(local_right)
                image = images[local_left, local_right]
                records.append(
                    {
                        "left_molecule_number_1based": left_index + 1,
                        "right_molecule_number_1based": right_index + 1,
                        "left_molecule_local_atom_id_0based": local_left,
                        "right_molecule_local_atom_id_0based": local_right,
                        "left_global_atom_index_0based": left_start + local_left,
                        "right_global_atom_index_0based": right_start + local_right,
                        "left_element": str(left_symbols[local_left]),
                        "right_element": str(right_symbols[local_right]),
                        "distance_A": float(distances[local_left, local_right]),
                        "threshold_A": float(pair_thresholds[local_left, local_right]),
                        "periodic_image": [int(value) for value in image],
                        "surface_axes": list(axes),
                        "normal_wrapped": False,
                        "reason": "SAM-SAM-threshold-violation",
                    }
                )
    records.sort(key=lambda record: (record["distance_A"], record["left_global_atom_index_0based"], record["right_global_atom_index_0based"]))
    type_counts = Counter(
        "-".join(sorted((record["left_element"], record["right_element"])))
        for record in records
    )
    return {
        "schema": "sam-sequential-final-sam-sam-audit-v1",
        "passed": not records,
        "molecule_count": len(offsets),
        "molecule_pair_count": pair_block_count,
        "collision_count": len(records),
        "collision_type_counts": dict(sorted(type_counts.items())),
        "representative_pairs": records[:20],
        "complete_pair_audit": True,
        "thresholds_A": thresholds,
        "surface_axes": list(axes),
        "normal_wrapped": False,
    }


def audit_final_p_surface_o_forbidden(
    *,
    assembled,
    substrate_atom_count: int,
    placed: list[dict],
    periodic_axes,
    p_surface_o_exclusion_contract: dict,
    full_to_working_atom_map: dict[int, dict],
) -> dict:
    """Independently verify that no final P--working-surface-O forbidden pair exists."""

    axes = _validated_surface_axes(periodic_axes)
    if axes != _validated_surface_axes(p_surface_o_exclusion_contract.get("surface_axes")):
        raise ValueError("Final P--surface-O audit axes disagree with the experiment contract")
    symbols = np.asarray(assembled.get_chemical_symbols(), dtype=str)
    positions = np.asarray(assembled.positions, dtype=float)
    substrate_symbols = symbols[:substrate_atom_count]
    substrate_positions = positions[:substrate_atom_count]
    audits = []
    cursor = int(substrate_atom_count)
    for placement_index, candidate in enumerate(placed):
        candidate_symbols = np.asarray(candidate.get("symbols"), dtype=str)
        stop = cursor + len(candidate_symbols)
        p_local = candidate.get("p_molecule_local_index_0based")
        if p_local is None:
            raise ValueError("Final P--surface-O audit requires the sealed candidate P local identity")
        audit = audit_p_surface_o_candidate(
            molecule_positions=positions[cursor:stop],
            molecule_symbols=candidate_symbols,
            p_molecule_local_index_0based=int(p_local),
            substrate_positions=substrate_positions,
            substrate_symbols=substrate_symbols,
            full_to_working_atom_map=full_to_working_atom_map,
            cell=assembled.cell,
            surface_frame=candidate["surface_frame"],
            p_surface_o_exclusion_contract=p_surface_o_exclusion_contract,
        )
        audits.append(
            {
                "molecule_number_1based": placement_index + 1,
                "site_instance_id": candidate["site_instance_id"],
                "global_atom_range_0based": [cursor, stop - 1],
                "forbidden_pair_count": int(audit["forbidden_pair_count"]),
                "minimum_distance_A": audit["minimum_distance_A"],
                "forbidden_pairs": audit["forbidden_pairs"],
                "sensitivity_counts": audit["sensitivity_counts"],
                "surface_axes": list(axes),
                "normal_wrapped": False,
            }
        )
        cursor = stop
    forbidden = [
        pair
        for audit in audits
        for pair in audit["forbidden_pairs"]
    ]
    return {
        "schema": "sam-sequential-final-p-surface-o-forbidden-audit-v1",
        "passed": not forbidden,
        "molecule_count": len(audits),
        "forbidden_pair_count": len(forbidden),
        "forbidden_pairs": forbidden,
        "per_molecule": audits,
        "contact_upper_bound_A": float(p_surface_o_exclusion_contract["contact_upper_bound_A"]),
        "inclusive_cutoff": True,
        "surface_axes": list(axes),
        "normal_axis": int(p_surface_o_exclusion_contract["normal_axis"]),
        "normal_wrapped": False,
        "reason": "algorithmic_count_only_final_forbidden_gate",
    }


def summarize_other_strict_vdw_audit(final_interface_audit: dict) -> dict:
    """Summarize the complete strict-vdW audit without making it a gate."""

    collisions = []
    mapped_violations = []
    for molecule_audit in final_interface_audit.get("molecule_audits", []):
        for record in molecule_audit.get("collisions", []):
            collisions.append(
                {
                    **record,
                    "molecule_number_1based": molecule_audit["molecule_number_1based"],
                    "site_instance_id": molecule_audit["site_instance_id"],
                }
            )
        for record in molecule_audit.get("mapped_bonds", []):
            if not record.get("passed", False):
                mapped_violations.append(
                    {
                        **record,
                        "molecule_number_1based": molecule_audit["molecule_number_1based"],
                        "site_instance_id": molecule_audit["site_instance_id"],
                    }
                )
    collisions.sort(key=lambda record: (record.get("distance_A", float("inf")), record.get("molecule_atom_id", -1), record.get("substrate_atom_id", -1)))
    type_counts = Counter(
        f"{record.get('molecule_element')}-{record.get('substrate_element')}"
        for record in collisions
    )
    return {
        "schema": "sam-sequential-other-substrate-strict-vdw-audit-v1",
        "performed": True,
        "audit_role": "diagnostic_only_intentional_not_a_pass_gate",
        "collision_count": len(collisions),
        "collision_type_counts": dict(sorted(type_counts.items())),
        "representative_pairs": collisions[:20],
        "mapped_bond_violation_count": len(mapped_violations),
        "representative_mapped_bond_violations": mapped_violations[:20],
        "all_molecule_audits_retained": True,
        "strict_interface_pass_not_claimed": True,
        "physical_interface_pass": False,
    }


def independently_validate_final_h0(
    *,
    trajectory_dir: Path,
    final_path: Path,
    substrate,
    placed: list[dict],
    molecule_formula: dict[str, int],
    periodic_axes,
    validation_mode: str,
    collision_thresholds_A: dict,
    substrate_vdw_radius_scale: float,
    mapped_bond_window_A,
    output_role="irreversible_sequential_growth_h0",
    p_surface_o_exclusion_contract: dict | None = None,
    full_to_working_atom_map: dict[int, dict] | None = None,
) -> dict:
    """Read a final H0 artifact back and run both independent CPU geometry gates."""

    if validation_mode not in {"h0", "strict-interface", "algorithmic-count-only"}:
        raise ValueError(
            "Independent H0 validation mode must be h0, strict-interface, or algorithmic-count-only"
        )
    if validation_mode == "algorithmic-count-only":
        if p_surface_o_exclusion_contract is None or full_to_working_atom_map is None:
            raise ValueError(
                "algorithmic-count-only validation requires the normalized P--surface-O contract and full-to-working map"
            )
        verify_p_surface_o_exclusion_contract_sources(p_surface_o_exclusion_contract)
    trajectory_dir = Path(trajectory_dir)
    final_path = Path(final_path)
    if not trajectory_dir.is_dir() or not final_path.is_file():
        raise FileNotFoundError("Independent validation requires a written final structure")
    axes = _validated_surface_axes(periodic_axes)
    thresholds = _validate_collision_thresholds(collision_thresholds_A)
    mapped_window = _validated_registered_bond_window(mapped_bond_window_A)
    molecule_formula = {
        str(element): int(count)
        for element, count in sorted(molecule_formula.items())
        if int(count)
    }
    if not molecule_formula or any(count < 0 for count in molecule_formula.values()):
        raise ValueError("Expected molecule formula must contain positive counts")
    atoms_per_molecule = sum(molecule_formula.values())
    molecule_count = len(placed)
    substrate_atom_count = len(substrate)
    expected_atom_count = substrate_atom_count + molecule_count * atoms_per_molecule
    expected_substrate_formula = _formula_from_symbols(
        substrate.get_chemical_symbols()
    )
    expected_formula = _expected_interface_formula(
        expected_substrate_formula, molecule_formula, molecule_count
    )

    file_sha256_before_readback = _sha256(final_path)
    readback = read_typed_structure(final_path)
    readback_symbols = [str(value) for value in readback.get_chemical_symbols()]
    readback_formula = _formula_from_symbols(readback_symbols)
    readback_cell_matches = bool(
        np.allclose(
            np.asarray(readback.cell, dtype=float),
            np.asarray(substrate.cell, dtype=float),
            atol=1.0e-8,
            rtol=0.0,
        )
    )
    readback_pbc_matches = bool(
        np.array_equal(np.asarray(readback.pbc, dtype=bool), np.asarray(substrate.pbc, dtype=bool))
    )
    readback_substrate_formula = _formula_from_symbols(
        readback_symbols[:substrate_atom_count]
    )
    block_formulas = []
    if len(readback) >= substrate_atom_count:
        for molecule_index in range(molecule_count):
            start = substrate_atom_count + molecule_index * atoms_per_molecule
            stop = start + atoms_per_molecule
            block_formulas.append(_formula_from_symbols(readback_symbols[start:stop]))

    final_interface_audit = audit_final_registered_interfaces(
        assembled=readback,
        substrate_atom_count=substrate_atom_count,
        placed=placed,
        periodic_axes=axes,
        radius_scale=float(substrate_vdw_radius_scale),
    )
    registered_bonds = final_interface_audit["registered_interface_bonds"]
    registered_windows_match_cli = bool(registered_bonds) and all(
        [
            float(record["minimum_distance_A"]),
            float(record["maximum_distance_A"]),
        ]
        == [mapped_window[0], mapped_window[1]]
        for record in registered_bonds
    )
    bonds_document = {
        "schema": (
            "sam-local-repair-registered-interface-bonds-v1"
            if output_role == "postprocessed_rearranged_h0"
            else "sam-sequential-registered-interface-bonds-v1"
        ),
        "schema_version": 1,
        "status": "sealed_exact_global_0based_bonds_from_final_readback",
        "sealed": True,
        "output_role": str(output_role),
        "structure_path": final_path.relative_to(trajectory_dir).as_posix(),
        "structure_sha256": file_sha256_before_readback,
        "structure_identity_sha256": _readback_structure_identity(readback),
        "index_convention": "global_0based_substrate_first_contiguous_sam_blocks",
        "periodic_fractional_axes": list(axes),
        "normal_axis_wrapped": False,
        "registered_interface_bonds": registered_bonds,
    }
    bonds_path = trajectory_dir / "registered-interface-bonds.json"
    bonds_reference = write_strict_json_noclobber(bonds_path, bonds_document)

    validator_config = ValidationConfig(
        substrate_atoms=substrate_atom_count,
        molecules=molecule_count,
        atoms_per_molecule=atoms_per_molecule,
        protons_per_molecule=0,
        anchor_element="P",
        parent_element="O",
        acceptor_element="O",
        hh_min=thresholds["H-H"],
        h_heavy_min=thresholds["H-heavy"],
        heavy_heavy_min=thresholds["heavy-heavy"],
        expected_formula=expected_formula,
        expected_substrate_formula=expected_substrate_formula,
        expected_molecule_formula=molecule_formula,
        surface_h_policy="h0",
        allowed_interface_pairs=(),
        registered_interface_bonds=tuple(registered_bonds),
        surface_periodic_axes=axes,
    )
    validator_report = validate_monolayer_structure(final_path, validator_config)
    file_sha256_after_validation = _sha256(final_path)
    structural_checks = {
        "final_structure_hash_matches_readback": (
            file_sha256_after_validation == file_sha256_before_readback
        ),
        "final_atom_count_matches": len(readback) == expected_atom_count,
        "final_formula_matches": readback_formula == expected_formula,
        "readback_substrate_formula_matches": (
            readback_substrate_formula == expected_substrate_formula
        ),
        "readback_cell_matches_working_substrate": readback_cell_matches,
        "readback_pbc_matches_working_substrate": readback_pbc_matches,
        "every_readback_molecule_formula_matches": (
            len(block_formulas) == molecule_count
            and all(formula == molecule_formula for formula in block_formulas)
        ),
        "registered_bond_windows_match_cli": registered_windows_match_cli,
        "generic_validator_passed": bool(validator_report["passed"]),
        "final_interface_audit_passed": bool(final_interface_audit["passed"]),
    }
    final_audit_required = validation_mode == "strict-interface"
    algorithmic_sam_sam_audit = None
    algorithmic_p_surface_o_audit = None
    algorithmic_other_substrate_audit = None
    if validation_mode == "algorithmic-count-only":
        algorithmic_sam_sam_audit = audit_final_sam_sam_collisions(
            assembled=readback,
            substrate_atom_count=substrate_atom_count,
            placed=placed,
            periodic_axes=axes,
            collision_thresholds_A=thresholds,
        )
        algorithmic_p_surface_o_audit = audit_final_p_surface_o_forbidden(
            assembled=readback,
            substrate_atom_count=substrate_atom_count,
            placed=placed,
            periodic_axes=axes,
            p_surface_o_exclusion_contract=p_surface_o_exclusion_contract,
            full_to_working_atom_map=full_to_working_atom_map,
        )
        algorithmic_other_substrate_audit = summarize_other_strict_vdw_audit(
            final_interface_audit
        )
        validator_checks = validator_report.get("checks", {})
        structural_checks.update(
            {
                "validator_expected_total_atoms": bool(validator_checks.get("expected_total_atoms", False)),
                "validator_expected_element_counts": bool(validator_checks.get("expected_element_counts", False)),
                "validator_integer_molecule_block_size": bool(validator_checks.get("integer_molecule_block_size", False)),
                "validator_uniform_molecule_formula": bool(validator_checks.get("uniform_molecule_formula", False)),
                "validator_each_molecule_formula": bool(validator_checks.get("each_molecule_matches_expected_formula", False)),
                "validator_all_molecule_topologies": bool(validator_checks.get("all_molecule_topologies_valid", False)),
                "sam_sam_thresholds_passed": bool(algorithmic_sam_sam_audit["passed"]),
                "unique_site_instances": bool(final_interface_audit["checks"]["unique_site_instances"]),
                "actual_metal_capacity_one": bool(final_interface_audit["checks"]["actual_metal_capacity_one"]),
                "all_registered_donor_windows_passed": all(
                    int(audit.get("mapped_bond_violation_count", 0)) == 0
                    for audit in final_interface_audit.get("molecule_audits", [])
                ),
                "p_surface_o_forbidden_gate_passed": bool(algorithmic_p_surface_o_audit["passed"]),
                "other_substrate_strict_vdw_audit_performed": bool(algorithmic_other_substrate_audit["performed"]),
                "physical_interface_pass_claim": False,
            }
        )
        required_check_names = [
            "final_structure_hash_matches_readback",
            "final_atom_count_matches",
            "final_formula_matches",
            "readback_substrate_formula_matches",
            "readback_cell_matches_working_substrate",
            "readback_pbc_matches_working_substrate",
            "every_readback_molecule_formula_matches",
            "registered_bond_windows_match_cli",
            "validator_expected_total_atoms",
            "validator_expected_element_counts",
            "validator_integer_molecule_block_size",
            "validator_uniform_molecule_formula",
            "validator_each_molecule_formula",
            "validator_all_molecule_topologies",
            "sam_sam_thresholds_passed",
            "unique_site_instances",
            "actual_metal_capacity_one",
            "all_registered_donor_windows_passed",
            "p_surface_o_forbidden_gate_passed",
            "other_substrate_strict_vdw_audit_performed",
        ]
        passed = all(structural_checks[name] for name in required_check_names)
    else:
        required_check_names = [
            "final_structure_hash_matches_readback",
            "final_atom_count_matches",
            "final_formula_matches",
            "readback_substrate_formula_matches",
            "every_readback_molecule_formula_matches",
            "registered_bond_windows_match_cli",
            "generic_validator_passed",
        ]
        if final_audit_required:
            required_check_names.append("final_interface_audit_passed")
        passed = all(structural_checks[name] for name in required_check_names)
    validation_document = {
        "schema": (
            "sam-local-repair-independent-h0-validation-v1"
            if output_role == "postprocessed_rearranged_h0"
            else (
                "sam-sequential-algorithmic-count-only-validation-v1"
                if validation_mode == "algorithmic-count-only"
                else "sam-sequential-independent-h0-validation-v1"
            )
        ),
        "schema_version": 1,
        "status": (
            "passed_cpu_h0_algorithmic_count_only"
            if validation_mode == "algorithmic-count-only" and passed
            else (
                "failed_cpu_h0_algorithmic_count_only_validation"
                if validation_mode == "algorithmic-count-only"
                else (
                    "passed_independent_h0_validation"
                    if passed
                    else "failed_independent_h0_validation"
                )
            )
        ),
        "sealed": True,
        "output_role": str(output_role),
        "validation_mode": validation_mode,
        "chemistry_scope": "H0_surface_protons_zero",
        "structure": {
            "path": final_path.relative_to(trajectory_dir).as_posix(),
            "sha256": file_sha256_after_validation,
            "identity_sha256": _readback_structure_identity(readback),
            "atom_count": len(readback),
            "formula": readback_formula,
            "expected_atom_count": expected_atom_count,
            "expected_formula": expected_formula,
        },
        "registered_interface_bonds_artifact": {
            "path": bonds_path.relative_to(trajectory_dir).as_posix(),
            "sha256": bonds_reference["sha256"],
            "bytes": bonds_reference["bytes"],
        },
        "validator_report": validator_report,
        "final_interface_audit": final_interface_audit,
        "algorithmic_count_only": (
            None
            if validation_mode != "algorithmic-count-only"
            else {
                "sam_sam_audit": algorithmic_sam_sam_audit,
                "p_surface_o_forbidden_audit": algorithmic_p_surface_o_audit,
                "other_substrate_strict_vdw_audit": algorithmic_other_substrate_audit,
                "physical_interface_pass": False,
                "strict_interface_audit_is_not_a_gate": True,
            }
        ),
        "checks": structural_checks,
        "gate": {
            "required_checks": required_check_names,
            "generic_validator_required": validation_mode != "algorithmic-count-only",
            "generic_structural_topology_checks_required": validation_mode == "algorithmic-count-only",
            "final_interface_audit_required": final_audit_required,
            "final_interface_audit_role": (
                "audit_only_not_physical_pass_gate"
                if validation_mode == "algorithmic-count-only"
                else ("required" if final_audit_required else "diagnostic")
            ),
            "algorithmic_count_only_physical_interface_pass": False,
        },
        "thresholds": {
            "sam_sam_collision_A": thresholds,
            "substrate_vdw_radius_scale": float(substrate_vdw_radius_scale),
            "mapped_bond_distance_window_A": list(mapped_window),
            "surface_periodic_axes": list(axes),
            "normal_axis_wrapped": False,
        },
        "passed": passed,
    }
    validation_path = trajectory_dir / "validation.json"
    validation_reference = write_strict_json_noclobber(
        validation_path, validation_document
    )
    return {
        "passed": passed,
        "mode": validation_mode,
        "output_role": str(output_role),
        "status": validation_document["status"],
        "checks": structural_checks,
        "gate": validation_document["gate"],
        "final_interface_audit": final_interface_audit,
        "validator_report": validator_report,
        "sam_sam_audit": algorithmic_sam_sam_audit,
        "p_surface_o_forbidden_audit": algorithmic_p_surface_o_audit,
        "other_substrate_strict_vdw_audit": algorithmic_other_substrate_audit,
        "physical_interface_pass": False if validation_mode == "algorithmic-count-only" else None,
        "artifacts": {
            "registered_interface_bonds": {
                "path": bonds_path.relative_to(trajectory_dir).as_posix(),
                "sha256": bonds_reference["sha256"],
                "bytes": bonds_reference["bytes"],
            },
            "validation": {
                "path": validation_path.relative_to(trajectory_dir).as_posix(),
                "sha256": validation_reference["sha256"],
                "bytes": validation_reference["bytes"],
            },
        },
    }


def _align_to_cell_bottom(assembled, *, surface_frame=None):
    """Translate the lowest atom to normal-coordinate zero.

    Translation is only along the registered outward normal.  Omitting the
    frame keeps the historical global-z output exactly compatible.
    """

    positions = np.asarray(assembled.positions, dtype=float)
    if surface_frame is None:
        normal = np.asarray([0.0, 0.0, 1.0])
        normal_coordinates = positions[:, 2]
    else:
        normal_coordinates = surface_frame_coordinates(
            positions, surface_frame
        )[:, 2]
        normal = np.asarray(
            surface_frame["outward_normal_cartesian_unit"], dtype=float
        )
    shift = -float(np.min(normal_coordinates))
    if abs(shift) > 1.0e-9:
        assembled.positions = positions.copy() + shift * normal
    return assembled


def _json_safe_trajectory(result: dict) -> dict:
    return {key: value for key, value in result.items() if key != "placed"}


def _relative_artifact_reference(path: Path, *, base: Path) -> dict:
    path = Path(path)
    identity = _file_identity(path)
    return {
        "path": path.relative_to(base).as_posix(),
        "sha256": identity["sha256"],
        "bytes": identity["bytes"],
    }


def _rebase_artifact_references(
    artifacts: dict, *, source_base: Path, destination_base: Path
) -> dict:
    rebased = {}
    for name, reference in artifacts.items():
        source_path = source_base / reference["path"]
        identity = _file_identity(source_path)
        if (
            identity["sha256"] != reference["sha256"]
            or int(identity["bytes"]) != int(reference["bytes"])
        ):
            raise RuntimeError(f"Artifact reference changed before rebasing: {source_path}")
        rebased[name] = {
            **reference,
            "path": source_path.relative_to(destination_base).as_posix(),
            "path_base": "run_root",
        }
    return rebased


def _capacity_K_debt_summary(result: dict, *, selection_policy: str) -> dict:
    if selection_policy != "capacity-aware":
        return {
            "applicable": False,
            "reason": "selection_policy_is_not_capacity_aware",
        }
    records = []
    for step in result.get("steps", []):
        audit = step.get("capacity_audit") or {}
        winner = audit.get("winner_capacity") or {}
        if winner:
            records.append(
                {
                    "step": int(step["step"]),
                    **{
                        name: int(winner[name])
                        for name in (
                            "K_before",
                            "K_after_static",
                            "K_after_collision",
                            "static_debt",
                            "collision_debt",
                            "total_debt",
                        )
                    },
                }
            )
    return {
        "applicable": True,
        "definition": "exact_shared_metal_static_capacity_and_one_step_debt",
        "step_count": len(records),
        "steps": records,
        "debt_totals": {
            name: sum(record[name] for record in records)
            for name in ("static_debt", "collision_debt", "total_debt")
        },
        "all_step_records_present": len(records) == len(result.get("steps", [])),
    }


def _cache_prefix_fingerprint(prefix: Path) -> dict:
    prefix = Path(prefix)
    records = []
    for suffix in (".json", ".npz"):
        path = Path(f"{prefix}{suffix}")
        if path.is_file():
            records.append(_file_identity(path))
    return {
        "schema": "sequential-conformer-cache-fingerprint-v1",
        "prefix": str(prefix),
        "files": records,
        "sha256": _payload_sha256(records if records else {"prefix": str(prefix)}),
    }


def _cache_artifact_records(
    *,
    conformer_prefix: Path,
    projection_directory: Path,
    output_dir: Path,
    formal: bool = False,
    projection_required: bool = False,
) -> dict:
    output_dir = Path(output_dir)
    conformer_prefix = Path(conformer_prefix)
    projection_directory = Path(projection_directory)

    def reference(path: Path) -> dict:
        if formal and path.is_symlink():
            raise RuntimeError(f"Formal cache artifact must not be a symlink: {path}")
        identity = _file_identity(path)
        try:
            relative = path.resolve().relative_to(output_dir.resolve()).as_posix()
            path_base = "run_root"
            rendered_path = relative
        except ValueError:
            path_base = "absolute_external_cache_path"
            rendered_path = identity["path"]
        return {
            "path": rendered_path,
            "path_base": path_base,
            "sha256": identity["sha256"],
            "bytes": identity["bytes"],
        }

    conformer_paths = (
        Path(f"{conformer_prefix}.json"),
        Path(f"{conformer_prefix}.npz"),
    )
    conformer_files = [
        reference(path) for path in conformer_paths if path.is_file()
    ]
    projection_paths = (
        sorted(projection_directory.rglob("*"))
        if projection_directory.is_dir()
        else []
    )
    projection_files = [
        reference(path) for path in projection_paths if path.is_file()
    ]
    if formal:
        expected_conformer = {
            "internal-cache/conformers.json",
            "internal-cache/conformers.npz",
        }
        actual_conformer = {record["path"] for record in conformer_files}
        if actual_conformer != expected_conformer or any(
            record["path_base"] != "run_root" for record in conformer_files
        ):
            raise RuntimeError(
                "Formal conformer cache provenance is incomplete or outside the "
                f"run namespace: {sorted(actual_conformer)}"
            )
        expected_projection = {
            "internal-cache/projection/projection-cache-data.npz",
            "internal-cache/projection/projection-cache-manifest.json",
        }
        actual_projection = {record["path"] for record in projection_files}
        if projection_required:
            if actual_projection != expected_projection or any(
                record["path_base"] != "run_root" for record in projection_files
            ):
                raise RuntimeError(
                    "Formal projection cache provenance is incomplete or outside "
                    f"the run namespace: {sorted(actual_projection)}"
                )
        elif os.path.lexists(os.fspath(projection_directory)):
            raise RuntimeError(
                "Formal projection cache was materialized although the selection "
                "policy does not use projection"
            )

    conformer = {
        "prefix": str(conformer_prefix),
        "path": (
            conformer_prefix.relative_to(output_dir).as_posix()
            if formal
            else str(conformer_prefix)
        ),
        "path_base": "run_root" if formal else "configured_cache_prefix",
        "materialized": bool(conformer_files),
        "cache_hit": False if formal else None,
        "prior_reuse": False if formal else None,
        "destination_clobber": False if formal else None,
        "installation": (
            "same_directory_hard_link_noreplace_after_file_fsync"
            if formal
            else "backward_compatible_cache_semantics"
        ),
        "files": conformer_files,
        "sha256": _payload_sha256(conformer_files),
    }
    projection = {
        "directory": str(projection_directory),
        "path": (
            projection_directory.relative_to(output_dir).as_posix()
            if formal
            else str(projection_directory)
        ),
        "path_base": "run_root" if formal else "configured_cache_directory",
        "required": bool(projection_required),
        "materialized": bool(projection_files),
        "cache_hit": False if formal else None,
        "prior_reuse": False if formal else None,
        "destination_clobber": False if formal else None,
        "installation": (
            "same_directory_hard_link_noreplace_after_file_fsync"
            if formal
            else "backward_compatible_cache_semantics"
        ),
        "files": projection_files,
        "sha256": _payload_sha256(projection_files),
    }
    return {
        "conformer": conformer,
        "projection": projection,
        "sha256": _payload_sha256(
            {"conformer": conformer, "projection": projection}
        ),
        "formal_no_prior_reuse_verified": bool(formal),
    }


def summarize_trajectory_results(results: list[dict], sampling_table: list[dict]) -> dict:
    """Summarize emergent coverage and accepted conformers across independent seeds."""

    if not results:
        raise ValueError("At least one sequential trajectory is required")
    counts = [int(result["emergent_molecule_count"]) for result in results]
    rank_counts = Counter()
    prototype_counts = Counter()
    denticity_counts = Counter()
    for result in results:
        for step in result["steps"]:
            rank_counts[int(step["selection_rank"])] += 1
            prototype_counts[str(step["site_prototype_id"])] += 1
            denticity_counts[len(step.get("target_donor_labels", []))] += 1
    accepted_total = sum(counts)
    base_probabilities = {
        int(record["selection_rank"]): float(record["base_probability"])
        for record in sampling_table
    }
    return {
        "trajectory_count": len(results),
        "emergent_molecule_counts": counts,
        "jamming_count": {
            "minimum": min(counts),
            "maximum": max(counts),
            "mean": float(statistics.mean(counts)),
            "median": float(statistics.median(counts)),
            "sample_standard_deviation": (
                float(statistics.stdev(counts)) if len(counts) > 1 else None
            ),
            "distribution": {
                str(value): frequency
                for value, frequency in sorted(Counter(counts).items())
            },
        },
        "accepted_molecule_total": accepted_total,
        "conformer_rank_distribution": [
            {
                "selection_rank": rank,
                "count": rank_counts[rank],
                "accepted_fraction": rank_counts[rank] / accepted_total,
                "base_probability": base_probabilities[rank],
            }
            for rank in sorted(base_probabilities)
        ],
        "site_prototype_distribution": [
            {
                "site_prototype_id": prototype_id,
                "count": count,
                "accepted_fraction": count / accepted_total,
            }
            for prototype_id, count in sorted(prototype_counts.items())
        ],
        "denticity_distribution": [
            {
                "denticity": denticity,
                "count": count,
                "accepted_fraction": count / accepted_total,
            }
            for denticity, count in sorted(denticity_counts.items())
        ],
    }


# N82 accepted phosphonate growth method; no molecule identity or target count.
# 膦酸生长默认方法；体系路径、化学式及覆盖数量不属于方法默认值。
PVKSAM_GROWTH_DEFAULTS = {
    "selection_policy": "cohesive-frontier",
    "cohesive_frontier_mode": "continuous-snap",
    "max_conformers": 10, "trajectory_count": 1, "seed": 340501,
    "frontier_snap_radius_A": 4.0, "headgroup_rmsd_max_A": 0.50,
    "cohesive_perimeter_relative_tolerance": 0.10,
    "cohesive_contact_relative_tolerance": 0.20,
    "cohesive_area_relative_tolerance": 0.20,
    "cohesive_energy_tolerance_eV": 0.03,
    "cohesive_contact_vdw_radius_scale": 1.10,
    "substrate_collision": "height", "substrate_height_clearance_A": 1.0,
    "hh_min": 1.5, "h_heavy_min": 1.8, "heavy_heavy_min": 2.2,
    "collision_cache": "lazy", "footprint_boundary_samples": 720,
    "projection_radius_scale": 1.0, "independent_validation": "off",
    "metal_coordination_filter": "exclude-six-coordinated",
    "metal_coordination_cutoff_A": 2.7, "metal_coordination_max_allowed": 5,
    "single_metal_sites": "dynamic-cn5-uncovered",
    "single_metal_rotation_step_deg": 30.0,
    "single_metal_translation_offsets_A": [-1.0, -0.5, 0.0, 0.5, 1.0],
    "write_every_step": True,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Grow SAM trajectories to geometric jamming or create a separate "
            "immutable CPU-only H0 local-repair child."
        )
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--operation", choices=("grow", "repair", "stage3-diagnostic"), default="grow",
        help=(
            "grow preserves irreversible sequential adsorption; repair creates a "
            "separate postprocessed/rearranged H0 child of one immutable parent"
        ),
    )
    parser.add_argument("--parent-run", type=Path)
    parser.add_argument("--reference-metrics", type=Path)
    parser.add_argument("--stage1-parent-run", type=Path)
    parser.add_argument("--stage2-parent-run", type=Path)
    parser.add_argument("--corrected-reference-metrics", type=Path)
    parser.add_argument("--stage3-derived-pool-manifest", type=Path)

    def positive_repair_integer(value):
        number = int(value)
        if number <= 0:
            raise argparse.ArgumentTypeError("repair bounds must be positive integers")
        return number

    def nonnegative_repair_integer(value):
        number = int(value)
        if number < 0:
            raise argparse.ArgumentTypeError(
                "--repair-max-fixed-replacements must be non-negative"
            )
        return number

    def positive_repair_float(value):
        number = float(value)
        if not np.isfinite(number) or number <= 0.0:
            raise argparse.ArgumentTypeError(
                "repair radius/time limits must be finite and positive"
            )
        return number

    parser.add_argument("--repair-max-k", type=positive_repair_integer, default=2)
    parser.add_argument(
        "--repair-neighborhood-radius-A", type=positive_repair_float, default=2.2
    )
    parser.add_argument(
        "--repair-max-frontier", type=positive_repair_integer, default=12
    )
    parser.add_argument(
        "--repair-max-local-candidates", type=positive_repair_integer, default=256
    )
    parser.add_argument(
        "--repair-max-solutions", type=positive_repair_integer, default=100
    )
    parser.add_argument(
        "--repair-time-limit-s", type=positive_repair_float, default=600.0
    )
    parser.add_argument(
        "--repair-max-fixed-replacements",
        type=nonnegative_repair_integer,
        default=10,
    )
    parser.add_argument("--substrate", type=Path, required=True)
    parser.add_argument("--layer-groups", type=Path, required=True)
    parser.add_argument("--site-instances", type=Path, required=True)
    parser.add_argument("--site-prototypes", type=Path, required=True)
    parser.add_argument("--conformer-glob", required=True)
    parser.add_argument("--conformer-analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--molecule-formula-json", required=True)
    parser.add_argument("--skeleton-formula-json")
    parser.add_argument("--max-conformers", type=int, default=10)
    parser.add_argument("--trajectory-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=340501)
    parser.add_argument("--energy-scale-eV", type=float, default=1.0)
    parser.add_argument("--area-coefficient", type=float, default=2.0)
    parser.add_argument("--rank-exponent", type=float, default=0.25)

    def finite_positive_headgroup_rmsd(value):
        number = float(value)
        if not np.isfinite(number) or number <= 0.0:
            raise argparse.ArgumentTypeError(
                "--headgroup-rmsd-max-A must be a finite positive number"
            )
        return number

    parser.add_argument(
        "--headgroup-rmsd-max-A",
        type=finite_positive_headgroup_rmsd,
        default=0.35,
    )
    parser.add_argument(
        "--conformer-cache",
        type=Path,
        default=Path(".cache/sequential_conformers"),
        help=(
            "cache prefix; formal independent validation requires exactly "
            "<output-dir>/internal-cache/conformers"
        ),
    )
    parser.add_argument(
        "--selection-policy",
        choices=("projection-aware", "capacity-aware", "cohesive-frontier"),
        default="cohesive-frontier",
    )

    def cohesive_relative_tolerance(value):
        number = float(value)
        if not np.isfinite(number) or not 0.0 <= number < 1.0:
            raise argparse.ArgumentTypeError(
                "cohesive relative tolerance must be finite and in [0, 1)"
            )
        return number

    def cohesive_nonnegative(value):
        number = float(value)
        if not np.isfinite(number) or number < 0.0:
            raise argparse.ArgumentTypeError(
                "--cohesive-energy-tolerance-eV must be finite and non-negative"
            )
        return number

    def cohesive_contact_scale(value):
        number = float(value)
        if not np.isfinite(number) or number < 1.0:
            raise argparse.ArgumentTypeError(
                "--cohesive-contact-vdw-radius-scale must be finite and at least 1"
            )
        return number

    parser.add_argument(
        "--cohesive-relative-tolerance",
        type=cohesive_relative_tolerance,
        default=None,
        help=(
            "Legacy umbrella relative tolerance for perimeter, contact, and "
            "projected-area filters; overridden by layer-specific flags"
        ),
    )
    parser.add_argument(
        "--cohesive-perimeter-relative-tolerance",
        type=cohesive_relative_tolerance,
        default=None,
        help=(
            "Periodic union perimeter increment relative tolerance for "
            "cohesive-frontier selection (default: 0.10)"
        ),
    )
    parser.add_argument(
        "--cohesive-contact-relative-tolerance",
        type=cohesive_relative_tolerance,
        default=None,
        help=(
            "Side contact count relative tolerance for cohesive-frontier "
            "selection (default: 0.20)"
        ),
    )
    parser.add_argument(
        "--cohesive-area-relative-tolerance",
        type=cohesive_relative_tolerance,
        default=None,
        help=(
            "Single-candidate projected area relative tolerance for "
            "cohesive-frontier selection (default: 0.20)"
        ),
    )
    parser.add_argument(
        "--cohesive-energy-tolerance-eV",
        type=cohesive_nonnegative,
        default=0.03,
    )
    parser.add_argument(
        "--cohesive-contact-vdw-radius-scale",
        type=cohesive_contact_scale,
        default=1.10,
    )
    parser.add_argument(
        "--cohesive-frontier-mode",
        choices=("discrete", "continuous-snap"),
        default="continuous-snap",
        help=(
            "Frontier exploration mode: 'discrete' evaluates existing site candidates; "
            "'continuous-snap' generates continuous frontier kissing positions q* "
            "and snaps to matching site candidates within --frontier-snap-radius-A"
        ),
    )
    parser.add_argument(
        "--frontier-snap-radius-A",
        type=positive_repair_float,
        default=4.0,
        help=(
            "Snap search radius in Angstrom for continuous frontier q* proposal to real site "
            "candidates (default: 4.0; grounded in read-only site instance empirical distance "
            "distribution: p90~3.35 A, p95~3.84 A, max~5.17 A)"
        ),
    )

    def positive_integer(value):
        number = int(value)
        if number <= 0:
            raise argparse.ArgumentTypeError(
                "--capacity-beam-width must be a positive integer"
            )
        return number

    parser.add_argument(
        "--capacity-beam-width", type=positive_integer, default=32
    )
    parser.add_argument(
        "--stage3-shape-cluster-threshold",
        type=positive_repair_float,
        default=1.25,
    )
    parser.add_argument(
        "--stage3-max-pool-members",
        type=positive_repair_integer,
        default=56,
    )
    parser.add_argument(
        "--stage3-capacity-time-limit-s",
        type=positive_repair_float,
        default=600.0,
    )
    parser.add_argument(
        "--projection-cache",
        type=Path,
        default=Path(".cache/sequential_projection"),
        help=(
            "cache directory; formal independent validation requires exactly "
            "<output-dir>/internal-cache/projection"
        ),
    )

    def boundary_samples(value):
        number = int(value)
        if number < 12:
            raise argparse.ArgumentTypeError(
                "--footprint-boundary-samples must be an integer of at least 12"
            )
        return number

    parser.add_argument(
        "--footprint-boundary-samples", type=boundary_samples, default=720
    )
    parser.add_argument(
        "--projection-radius-scale",
        type=positive_repair_float,
        default=1.0,
        help="ASE vdW radius scale used by the projected footprint contract",
    )

    def nonnegative_tolerance(value):
        number = float(value)
        if not np.isfinite(number) or number < 0.0:
            raise argparse.ArgumentTypeError(
                "--area-tie-tolerance-A2 must be a finite non-negative number"
            )
        return number

    parser.add_argument(
        "--area-tie-tolerance-A2", type=nonnegative_tolerance, default=1.0e-6
    )

    def positive_perimeter_tolerance(value):
        number = float(value)
        if not np.isfinite(number) or number <= 0.0:
            raise argparse.ArgumentTypeError(
                "--perimeter-tie-tolerance-A must be a finite positive number"
            )
        return number

    parser.add_argument(
        "--perimeter-tie-tolerance-A",
        type=positive_perimeter_tolerance,
        default=1.0e-6,
    )
    parser.add_argument(
        "--collision-cache",
        choices=("off", "lazy"),
        default="off",
    )

    def positive_threshold(value):
        number = float(value)
        if not np.isfinite(number) or number <= 0.0:
            raise argparse.ArgumentTypeError(
                "collision thresholds must be finite positive numbers"
            )
        return number

    parser.add_argument("--hh-min", type=positive_threshold, default=1.20)
    parser.add_argument("--h-heavy-min", type=positive_threshold, default=1.40)
    parser.add_argument("--heavy-heavy-min", type=positive_threshold, default=1.80)
    parser.add_argument(
        "--substrate-collision",
        choices=("check", "skip", "strict-vdw", "height"),
        default="check",
        help=(
            "skip preserves the legacy unchecked candidate algorithm; check uses "
            "the hard H/H-heavy/heavy thresholds with only exact in-window mapped "
            "bonds exempt; strict-vdw uses ASE vdW radii and the Surface Frame"
            "; height requires every non-anchor-group SAM atom to clear the highest "
            "substrate Surface-Frame normal coordinate; the topology-resolved anchor "
            "P plus bonded headgroup O atoms are excluded"
        ),
    )
    parser.add_argument(
        "--substrate-vdw-radius-scale",
        type=positive_threshold,
        default=0.85,
        help="ASE vdW-radius scale for --substrate-collision strict-vdw",
    )
    parser.add_argument(
        "--substrate-height-clearance-A",
        type=positive_threshold,
        default=1.0,
        help=(
            "non-anchor-SAM-atom normal clearance above the highest substrate atom "
            "for --substrate-collision height"
        ),
    )
    parser.add_argument(
        "--p-surface-o-exclusion",
        "--p-surface-o-mode",
        dest="p_surface_o_exclusion",
        choices=("off", "contract"),
        default="off",
        help=(
            "default off; contract enables the hash-locked generic P-to-any-"
            "working-surface-O inclusive initial candidate gate"
        ),
    )
    parser.add_argument(
        "--p-surface-o-contract",
        "--p-surface-o-contract-path",
        dest="p_surface_o_contract",
        type=Path,
    )
    parser.add_argument("--p-surface-o-contract-sha256")
    parser.add_argument(
        "--p-surface-o-source-report",
        "--p-surface-o-source-report-path",
        dest="p_surface_o_source_report",
        type=Path,
    )
    parser.add_argument("--p-surface-o-source-report-sha256")
    parser.add_argument(
        "--p-surface-o-source-config",
        "--p-surface-o-scratch-source",
        dest="p_surface_o_source_config",
        type=Path,
    )
    parser.add_argument("--p-surface-o-source-config-sha256")
    parser.add_argument(
        "--metal-coordination-filter",
        choices=("off", "exclude-six-coordinated"),
        default="exclude-six-coordinated",
        help=(
            "default exclude-six-coordinated; rejects candidates where any mapped "
            "substrate metal (In/Sn) is 6-coordinated by substrate O within 2.7 A prior to adsorption"
        ),
    )
    parser.add_argument(
        "--metal-coordination-cutoff-A",
        type=positive_threshold,
        default=2.7,
        help="exploratory first-neighbor substrate-O cutoff for metal coordination (default: 2.7 A)",
    )
    parser.add_argument(
        "--metal-coordination-max-allowed",
        type=int,
        default=5,
        help="maximum allowed substrate-O coordination number for mapped metals (default: 5)",
    )
    parser.add_argument(
        "--mapped-bond-min-A",
        type=positive_threshold,
        default=1.7,
        help="minimum accepted distance for every exact mapped donor-metal bond",
    )
    parser.add_argument(
        "--mapped-bond-max-A",
        type=positive_threshold,
        default=2.8,
        help="maximum accepted distance for every exact mapped donor-metal bond",
    )
    parser.add_argument(
        "--selection-criterion",
        choices=("uncovered-anchors", "perimeter", "compact"),
        default="uncovered-anchors",
        help="primary projection score: uncovered-anchors (default), "
        "perimeter (minimize periodic union-perimeter increment to fill holes), "
        "or compact (minimize union-area increment to maximize molecule count)",
    )
    parser.add_argument(
        "--independent-validation",
        choices=("off", "h0", "strict-interface", "algorithmic-count-only"),
        default="off",
        help=(
            "off preserves legacy unvalidated output; h0 gates on the generic H0 "
            "validator while reporting the final interface audit diagnostically; "
            "strict-interface requires both independent gates; "
            "algorithmic-count-only is the explicit CPU H0 count/topology gate "
            "with strict-vdW substrate evidence retained as audit-only"
        ),
    )
    parser.add_argument(
        "--single-metal-sites",
        choices=("off", "dynamic-cn5-uncovered"),
        default="off",
        help="phosphonate single-metal growth (default: dynamic-cn5-uncovered)",
    )
    parser.add_argument(
        "--single-metal-rotation-step-deg",
        type=float,
        default=30.0,
        help="rotation search step in degrees for single-metal poses",
    )
    parser.add_argument(
        "--single-metal-translation-offsets-A",
        nargs="+",
        type=float,
        default=[-1.0, -0.5, 0.0, 0.5, 1.0],
        help="discrete translation offsets in Angstrom for single-metal poses",
    )
    parser.add_argument("--write-every-step", action=argparse.BooleanOptionalAction,
                        help="write every accepted growth step; use --no-write-every-step for repair/diagnostic modes")
    # Keep per-tier sentinels so the legacy umbrella tolerance still works.
    # resolve_cohesive_cli_tolerances supplies the same 0.10/0.20/0.20 defaults.
    parser.set_defaults(**{key: value for key, value in PVKSAM_GROWTH_DEFAULTS.items()
                           if key not in {"cohesive_perimeter_relative_tolerance",
                                          "cohesive_contact_relative_tolerance",
                                          "cohesive_area_relative_tolerance"}})
    return parser


def _capacity_policy_manifest(
    *,
    beam_width,
    area_tie_tolerance_A2,
    perimeter_tie_tolerance_A,
    collision_thresholds_A,
) -> dict:
    """Return the durable scientific scope of capacity-aware selection."""

    if isinstance(beam_width, bool) or int(beam_width) != beam_width or int(beam_width) <= 0:
        raise ValueError("capacity beam width must be a positive integer")
    thresholds = _validate_collision_thresholds(collision_thresholds_A)
    return {
        "optimization_scope": "exact_one_step_within_deterministic_beam",
        "global_candidate_optimality": False,
        "global_packing_certificate": False,
        "beam_width": int(beam_width),
        "beam_order": [
            "static_debt_exact_integer_ascending",
            "total_hole_area_after_trial_A2_ascending",
            "candidate_id_ascending",
        ],
        "beam_boundary_policy": (
            "include_every_candidate_with_matching_static_debt_and_total_hole_"
            "area_within_area_tie_tolerance;never_truncate_boundary_tie"
        ),
        "capacity_definitions": {
            "K_before": "K(S_t)",
            "K_after_static": (
                "K(static_live_sites_after_chosen_and_shared_metal_removal)"
            ),
            "K_after_collision": "K(post_preview_live_sites)",
            "static_debt": "K_before - 1 - K_after_static",
            "collision_debt": "K_after_static - K_after_collision",
            "total_debt": "K_before - 1 - K_after_collision",
        },
        "one_plus_K_after_collision": {
            "is_upper_bound": True,
            "scope": "shared_metal_and_individual_feasibility_only",
            "future_candidate_pair_collisions": False,
            "realizable_terminal_state_claim": False,
            "global_packing_claim": False,
        },
        "beam_lexicographic_order": [
            "minimum_total_debt_exact_integer",
            "minimum_total_hole_area_after_trial_A2",
            "minimum_maximum_torus_hole_area_after_trial_A2",
            "minimum_total_physical_hole_perimeter_after_trial_A",
            "approved_conformer_prior_then_seed_only_after_all_metric_ties",
        ],
        "area_tie_tolerance_A2": float(area_tie_tolerance_A2),
        "perimeter_tie_tolerance_A": float(perimeter_tie_tolerance_A),
        "torus_metric_version": "periodic-torus-holes-v2",
        "weighted_objective_used": False,
        "milp_policy": "certified_optimal_zero_gap_only_fail_closed",
        "collision_thresholds_A": thresholds,
        "collision_threshold_adaptation": "disabled",
        "accepted_molecules_moved": False,
        "termination": "no_feasible_site_conformer_mapping_candidate",
        "ranking_artifact": capacity_ranking_artifact_contract(),
        "complexity_evidence_location": (
            "trajectory-XXXX/ranking/step-XXXX.json.gz:"
            "capacity_audit.complexity_bound"
        ),
    }


def _repair_operation_evidence(parent_reference, reference_metrics) -> dict:
    return {
        "schema": "sam-local-repair-operation-evidence-v1",
        "parent": {
            "schema_version": parent_reference["schema_version"],
            "parent_role": parent_reference["parent_role"],
            "parent_top_status": parent_reference["parent_top_status"],
            "trajectory_number": parent_reference["trajectory_number"],
            "trajectory_seed": parent_reference["trajectory_seed"],
            "hash_ledger_before": parent_reference["hash_ledger_before"],
            "parent_tree_ledger_before": parent_reference[
                "parent_tree_ledger_before"
            ],
            "code_evidence_mode": parent_reference["code_evidence_mode"],
            "schema2_git_evidence": parent_reference.get(
                "schema2_git_evidence"
            ),
        },
        "reference_metrics": {
            "manifest": reference_metrics["manifest_identity"],
            "payload": reference_metrics["metrics_identity"],
            "future_A1_gate_definitions": reference_metrics[
                "future_A1_gate_definitions"
            ],
            "validation_proof": reference_metrics["validation_proof"],
            "strict_validation_debt": reference_metrics[
                "strict_validation_debt"
            ],
        },
        "output_contract": repair_output_contract(),
    }


def _validate_repair_parent_contract(
    parent_reference,
    *,
    args,
    substrate,
    molecule_spec,
    periodic_axes,
    selected_source_lock,
    collision_thresholds_A,
    domain_diagnostics=None,
) -> dict:
    top = parent_reference["top_manifest"]
    schema_version = int(parent_reference["schema_version"])
    checks = {}
    parent_metric_evidence = _parent_requested_metric_contract(parent_reference)
    current_metric_contract = _repair_cli_metric_contract(args)
    checks["formal_metric_contract"] = (
        parent_metric_evidence["contract"] == current_metric_contract
    )
    checks["working_substrate_atom_count"] = int(
        top.get("working_substrate_atom_count", -1)
    ) == len(substrate)
    checks["molecule_formula"] = {
        str(key): int(value) for key, value in top.get("molecule_formula", {}).items()
    } == {
        str(key): int(value) for key, value in molecule_spec.formula.items()
    }
    if schema_version == 4:
        parent_axes = top.get("periodic_fractional_axes")
        parent_thresholds = (
            top.get("sam_sam_collision_contract", {}).get("thresholds_A")
        )
        checks["substrate_collision_mode"] = (
            top.get("substrate_collision_mode") == args.substrate_collision
        )
        checks["max_conformers"] = int(top.get("max_conformers", -1)) == int(
            args.max_conformers
        )
        parent_lock = top.get("selected_conformer_source_lock", {}).get(
            "selected_source_files"
        )
        current_lock = selected_source_lock.get("selected_source_files")
        checks["selected_conformer_source_lock"] = (
            isinstance(parent_lock, list)
            and [
                (int(record["selection_rank"]), record["sha256"], int(record["bytes"]))
                for record in parent_lock
            ]
            == [
                (int(record["selection_rank"]), record["sha256"], int(record["bytes"]))
                for record in current_lock
            ]
        )
    else:
        placement = top.get("placement_policy") or {}
        parent_axes = placement.get("periodic_axes")
        parent_thresholds = placement.get("collision_thresholds_A")
        # Historical schema 2 predates the named substrate mode.  A recorded
        # substrate-collision rejection funnel is explicit evidence of its
        # historical hard-distance gate; absence is treated as legacy skip.
        rejection_counts = (top.get("domain_diagnostics") or {}).get(
            "rejection_counts", {}
        )
        inferred_mode = (
            "check" if "substrate_collision" in rejection_counts else "skip"
        )
        checks["historical_substrate_collision_mode"] = (
            inferred_mode == args.substrate_collision
        )
    try:
        checks["periodic_fractional_axes"] = (
            _validated_surface_axes(parent_axes)
            == _validated_surface_axes(periodic_axes)
        )
    except ValueError:
        checks["periodic_fractional_axes"] = False
    try:
        checks["sam_sam_collision_thresholds"] = (
            _validate_collision_thresholds(parent_thresholds)
            == _validate_collision_thresholds(collision_thresholds_A)
        )
    except (TypeError, ValueError):
        checks["sam_sam_collision_thresholds"] = False
    if domain_diagnostics is not None:
        parent_domain = top.get("candidate_domain") or top.get(
            "domain_diagnostics", {}
        )
        if schema_version == 4:
            checks["candidate_domain_fingerprint"] = (
                parent_domain.get("fingerprint")
                == domain_diagnostics.get("candidate_domain_fingerprint")
            )
        else:
            checks["historical_candidate_count"] = int(
                parent_domain.get("candidate_count", -1)
            ) == int(domain_diagnostics.get("candidate_count", -2))
            checks["historical_site_count"] = int(
                parent_domain.get("site_count", -1)
            ) == int(domain_diagnostics.get("site_count", -2))
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(
            "Parent/current scientific domain contract mismatch: "
            + ", ".join(failed)
        )
    return {
        "schema_version": schema_version,
        "checks": checks,
        "all_checks_passed": True,
        "current_domain_required": True,
        "metric_contract": {
            "current_cli": current_metric_contract,
            "parent_evidence": parent_metric_evidence,
            "all_exactly_equal": True,
        },
        "threshold_adaptation": False,
    }


def _serializable_fixed_repair_result(result) -> dict:
    return {key: value for key, value in result.items() if key != "placed"}


def execute_local_repair_package(
    *,
    args,
    output_dir,
    substrate,
    domains,
    domain_diagnostics,
    projection_cache,
    conflicts,
    metal_constraints,
    collision_thresholds_A,
    periodic_axes,
    normal_axis,
    assembly_surface_frame,
    mapped_bond_window_A,
    molecule_spec,
    parent_reference,
    reference_metrics,
    parent_contract,
    run_request,
    run_request_reference,
    implementation,
    code_snapshot,
    primary_inputs,
    selected_source_lock,
    source_verification_checkpoints,
    loaded_conformer_audit,
    cache_reuse_contract,
    conformer_cache_path,
    projection_cache_path,
    domain_fingerprint,
    conformer_pool_fingerprint,
    projection_cache_manifest,
    run_started,
) -> int:
    """Execute and seal one CPU-only postprocessed/rearranged H0 repair child."""

    output_dir = Path(output_dir)
    if not cache_reuse_contract.get("formal"):
        raise ValueError("Local repair requires the formal internal-cache lifecycle")
    if projection_cache_manifest.get("hit") is not False:
        raise RuntimeError(
            "Formal local repair projection cache must be newly built without reuse"
        )
    actions_dir = output_dir / "actions"
    solver_dir = output_dir / "solver"
    actions_dir.mkdir(exist_ok=False)
    solver_dir.mkdir(exist_ok=False)
    _fsync_directory(output_dir)
    reconstruction = reconstruct_parent_placements(
        parent_reference,
        domains=domains,
        substrate=substrate,
        surface_frame=assembly_surface_frame,
    )
    parent_placed = list(reconstruction["placed"])
    parent_objective = repair_objective(parent_placed, projection_cache)
    fixed = run_fixed_site_replacements(
        placed=parent_placed,
        domains=domains,
        projection_cache=projection_cache,
        cell=substrate.cell,
        periodic_axes=periodic_axes,
        collision_thresholds_A=collision_thresholds_A,
        max_replacements=args.repair_max_fixed_replacements,
        area_tolerance_A2=args.area_tie_tolerance_A2,
        metal_constraints=metal_constraints,
    )
    fixed_document = {
        "schema": "sam-local-repair-fixed-site-audit-v1",
        "pass": "initial_before_cardinality_search",
        **_serializable_fixed_repair_result(fixed),
    }
    fixed_reference = write_strict_json_noclobber(
        actions_dir / "fixed-site-initial.json", fixed_document
    )
    current = list(fixed["placed"])
    initial_capacity_gate = certified_static_repair_gate(
        placed=current,
        domains=domains,
        metal_constraints=metal_constraints,
        site_conflicts=conflicts,
        time_limit_seconds=args.repair_time_limit_s,
    )
    capacity_reference = write_strict_json_noclobber(
        solver_dir / "static-capacity-gate.json",
        {
            "schema": "sam-local-repair-static-capacity-gate-v1",
            **initial_capacity_gate,
        },
    )
    if initial_capacity_gate["run_k_to_k_plus_one"]:
        local = run_bounded_local_cardinality_repair(
            placed=current,
            domains=domains,
            projection_cache=projection_cache,
            metal_constraints=metal_constraints,
            site_conflicts=conflicts,
            cell=substrate.cell,
            periodic_axes=periodic_axes,
            collision_thresholds_A=collision_thresholds_A,
            max_k=args.repair_max_k,
            neighborhood_radius_A=args.repair_neighborhood_radius_A,
            max_frontier=args.repair_max_frontier,
            max_local_candidates=args.repair_max_local_candidates,
            max_solutions=args.repair_max_solutions,
            time_limit_seconds=args.repair_time_limit_s,
            max_fixed_replacements=args.repair_max_fixed_replacements,
            area_tolerance_A2=args.area_tie_tolerance_A2,
        )
        current = list(local["placed"])
    else:
        local = {
            "status": "skipped_proven_static_cardinality_bound",
            "termination": "skipped_proven_static_cardinality_bound",
            "placed": current,
            "objective_after": fixed["objective_after"],
            "accepted_actions": [],
            "all_evaluated_pareto_actions": [],
            "search_passes": [],
            "solver_documents": [],
            "post_insertion_fixed_passes": [],
            "search_complete": True,
            "timeout_observed": False,
            "optimization_scope": (
                "static_cardinality_gate_only_no_k_to_k_plus_one_solve"
            ),
            "global_packing_claim": False,
            "global_optimality_claim": False,
            "weighted_score_used": False,
            "thresholds_A": collision_thresholds_A,
            "wall_time_seconds": 0.0,
        }
    solver_artifacts = [
        {
            "role": "complete_static_capacity_gate_certificate",
            "path": Path(capacity_reference["path"])
            .relative_to(output_dir)
            .as_posix(),
            "sha256": capacity_reference["sha256"],
            "bytes": capacity_reference["bytes"],
        }
    ]
    for index, document in enumerate(local["solver_documents"], 1):
        reference = write_strict_json_noclobber(
            solver_dir / f"solver-{index:06d}.json", document
        )
        solver_artifacts.append(
            {
                "path": Path(reference["path"]).relative_to(output_dir).as_posix(),
                "sha256": reference["sha256"],
                "bytes": reference["bytes"],
            }
        )
    local_document = {
        "schema": "sam-local-repair-cardinality-search-audit-v1",
        **{
            key: value
            for key, value in local.items()
            if key not in {"placed", "solver_documents", "post_insertion_fixed_passes"}
        },
        "solver_artifacts": solver_artifacts,
        "post_insertion_fixed_passes": [
            _serializable_fixed_repair_result(result)
            for result in local["post_insertion_fixed_passes"]
        ],
    }
    local_reference = write_strict_json_noclobber(
        actions_dir / "cardinality-search.json", local_document
    )
    accepted_action_artifacts = []
    all_accepted_actions = fixed["accepted_actions"] + local["accepted_actions"]
    for index, action in enumerate(all_accepted_actions, 1):
        reference = write_strict_json_noclobber(
            actions_dir / f"accepted-action-{index:04d}.json",
            {
                "schema": "sam-local-repair-accepted-action-v1",
                "action_number": index,
                "output_role": "postprocessed_rearranged_h0",
                **action,
            },
        )
        accepted_action_artifacts.append(
            {
                "path": Path(reference["path"]).relative_to(output_dir).as_posix(),
                "sha256": reference["sha256"],
                "bytes": reference["bytes"],
            }
        )

    final_objective = repair_objective(current, projection_cache)
    final_path = output_dir / "postprocessed-h0.extxyz"
    final_assembled = _align_to_cell_bottom(
        _assemble(substrate, current), surface_frame=assembly_surface_frame
    )
    if os.path.lexists(os.fspath(final_path)):
        raise FileExistsError(f"Repair final output already exists: {final_path}")
    write(final_path, final_assembled)
    independent_validation = independently_validate_final_h0(
        trajectory_dir=output_dir,
        final_path=final_path,
        substrate=substrate,
        placed=current,
        molecule_formula=molecule_spec.formula,
        periodic_axes=periodic_axes,
        validation_mode=args.independent_validation,
        collision_thresholds_A=collision_thresholds_A,
        substrate_vdw_radius_scale=float(args.substrate_vdw_radius_scale),
        mapped_bond_window_A=mapped_bond_window_A,
        output_role="postprocessed_rearranged_h0",
    )
    parent_unchanged = verify_parent_unchanged(parent_reference)
    references_unchanged = verify_reference_metrics_unchanged(reference_metrics)
    gates = repair_acceptance_gates(
        _repair_objective_summary(final_objective),
        reference_metrics["payload"],
        area_tolerance_A2=args.area_tie_tolerance_A2,
    )
    parent_reference_document = {
        "schema": "sam-local-repair-parent-reference-v1",
        "schema_version": 1,
        "sealed": True,
        "output_role": "postprocessed_rearranged_h0",
        "parent": {
            "parent_role": parent_reference["parent_role"],
            "parent_run": parent_reference["parent_run"],
            "schema_version": parent_reference["schema_version"],
            "parent_top_status": parent_reference["parent_top_status"],
            "trajectory_number": parent_reference["trajectory_number"],
            "trajectory_seed": parent_reference["trajectory_seed"],
            "code_evidence_mode": parent_reference["code_evidence_mode"],
            "hash_ledger_before": parent_reference["hash_ledger_before"],
            "parent_tree_ledger_before": parent_reference[
                "parent_tree_ledger_before"
            ],
            "schema2_git_evidence": parent_reference.get(
                "schema2_git_evidence"
            ),
            "unchanged_ledger": parent_unchanged,
        },
        "reconstruction": {
            key: value for key, value in reconstruction.items() if key != "placed"
        },
        "reference_metrics": {
            "manifest_identity": reference_metrics["manifest_identity"],
            "metrics_identity": reference_metrics["metrics_identity"],
            "validation_proof": reference_metrics["validation_proof"],
            "strict_validation_debt": reference_metrics[
                "strict_validation_debt"
            ],
            "unchanged_ledger": references_unchanged,
        },
        "not_irreversible_sequential_history": True,
        "global_packing_claim": False,
    }
    parent_reference_artifact = write_strict_json_noclobber(
        output_dir / "parent-reference.json", parent_reference_document
    )
    final_capacity_gate = certified_static_repair_gate(
        placed=current,
        domains=domains,
        metal_constraints=metal_constraints,
        site_conflicts=conflicts,
        time_limit_seconds=args.repair_time_limit_s,
    )
    solve_records = [
        record
        for document in local["solver_documents"]
        for record in document["solver"].get("solve_records", [])
    ]
    finite_gaps = [
        float(record["mip_gap"])
        for record in solve_records
        if record.get("mip_gap") is not None
        and np.isfinite(float(record["mip_gap"]))
    ]
    cache_artifacts = _cache_artifact_records(
        conformer_prefix=Path(conformer_cache_path),
        projection_directory=Path(projection_cache_path),
        output_dir=output_dir,
        formal=True,
        projection_required=True,
    )
    # The resolved formal paths are fixed by _cache_reuse_contract; use the
    # recorded artifact paths from the run rather than permitting alternate
    # repair cache namespaces.
    run_status = (
        "passed_cpu_h0_postprocessed_local_repair_and_independent_validation"
        if independent_validation["passed"]
        else "failed_independent_h0_validation"
    )
    manifest = {
        "schema": "sam-local-repair-run-manifest-v1",
        "schema_version": 1,
        "status": run_status,
        "sealed": True,
        "operation": "repair",
        "method": "bounded_postprocessed_local_repair_without_threshold_adaptation",
        "role": "postprocessed_rearranged_h0",
        "output_contract": repair_output_contract(),
        "chemistry_scope": "H0_geometry_only_surface_protons_deferred",
        "execution_scope": "CPU_only_geometry_local_repair_and_independent_validation",
        "parent": {
            "role": parent_reference["parent_role"],
            "top_manifest": parent_reference["hash_ledger_before"]["top_manifest"],
            "trajectory_manifest": parent_reference["hash_ledger_before"][
                "trajectory_manifest"
            ],
            "final_structure": parent_reference["hash_ledger_before"][
                "final_structure"
            ],
            "unchanged_ledger": parent_unchanged,
            "reference_artifact": {
                "path": "parent-reference.json",
                "sha256": parent_reference_artifact["sha256"],
                "bytes": parent_reference_artifact["bytes"],
            },
        },
        "run_request": {
            "path": "run-request.json",
            "sha256": run_request_reference["sha256"],
            "bytes": run_request_reference["bytes"],
            "run_identity_sha256": run_request["run_identity"]["sha256"],
        },
        "implementation": implementation,
        "code_snapshot": code_snapshot,
        "inputs": primary_inputs,
        "parent_contract": parent_contract,
        "selected_conformer_source_lock": selected_source_lock,
        "source_verification_checkpoints": source_verification_checkpoints,
        "loaded_conformer_ensemble_audit": loaded_conformer_audit,
        "candidate_domain": {
            "fingerprint": domain_fingerprint,
            "candidate_count": int(domain_diagnostics["candidate_count"]),
            "site_count": int(domain_diagnostics["site_count"]),
            "empty_site_count": int(domain_diagnostics["empty_site_count"]),
            "substrate_contract_guaranteed": True,
        },
        "fingerprints": {
            "candidate_domain": domain_fingerprint,
            "conformer_pool": conformer_pool_fingerprint,
            "projection": _payload_sha256(projection_cache_manifest["contract"]),
        },
        "periodic_fractional_axes": list(periodic_axes),
        "normal_axis": int(normal_axis),
        "molecule_formula": dict(molecule_spec.formula),
        "thresholds": {
            "sam_sam_collision_A": collision_thresholds_A,
            "mapped_bond_distance_window_A": list(mapped_bond_window_A),
            "substrate_vdw_radius_scale": float(args.substrate_vdw_radius_scale),
            "threshold_adaptation": False,
            "threshold_reduction": False,
        },
        "parent_objective": _repair_objective_summary(parent_objective),
        "fixed_site_repair": {
            "artifact": {
                "path": "actions/fixed-site-initial.json",
                "sha256": fixed_reference["sha256"],
                "bytes": fixed_reference["bytes"],
            },
            "accepted_action_count": len(fixed["accepted_actions"]),
            "all_observed_pareto_actions": fixed[
                "all_observed_pareto_actions"
            ],
        },
        "capacity": {
            "static_gate_artifact": {
                "path": "solver/static-capacity-gate.json",
                "sha256": capacity_reference["sha256"],
                "bytes": capacity_reference["bytes"],
            },
            "initial": initial_capacity_gate,
            "final": final_capacity_gate,
            "capacity_margin": int(final_capacity_gate["capacity_margin"]),
            "certified_static_K": int(
                final_capacity_gate["certified_static_K"]
            ),
        },
        "local_search": {
            "artifact": {
                "path": "actions/cardinality-search.json",
                "sha256": local_reference["sha256"],
                "bytes": local_reference["bytes"],
            },
            "optimization_scope": local["optimization_scope"],
            "configured_scope": {
                "repair_max_k": int(args.repair_max_k),
                "repair_neighborhood_radius_A": float(
                    args.repair_neighborhood_radius_A
                ),
                "repair_max_frontier": int(args.repair_max_frontier),
                "repair_max_local_candidates": int(
                    args.repair_max_local_candidates
                ),
                "repair_max_solutions": int(args.repair_max_solutions),
                "repair_time_limit_s": float(args.repair_time_limit_s),
                "repair_max_fixed_replacements": int(
                    args.repair_max_fixed_replacements
                ),
            },
            "termination": local["termination"],
            "search_complete": local["search_complete"],
            "timeout_observed": local["timeout_observed"],
            "solver_artifacts": solver_artifacts,
            "solver_directory_nonempty": True,
            "solver_artifact_count": len(solver_artifacts),
            "static_capacity_certificate_present_even_when_K_equals_N": True,
            "solver_gap_summary": {
                "solve_count": len(solve_records),
                "finite_gap_count": len(finite_gaps),
                "maximum_finite_gap": max(finite_gaps) if finite_gaps else None,
                "nonzero_gap_count": sum(value != 0.0 for value in finite_gaps),
                "timeout_count": sum(record.get("timeout", False) for record in solve_records),
                "incumbent_timeout_not_labeled_optimal_or_infeasible": True,
            },
            "global_optimality_claim": False,
            "global_packing_claim": False,
        },
        "all_candidate_metrics": {
            "all_observed_fixed_N_pareto_actions": fixed[
                "all_observed_pareto_actions"
            ],
            "all_evaluated_local_pareto_actions": local[
                "all_evaluated_pareto_actions"
            ],
            "accepted_action_artifacts": accepted_action_artifacts,
            "weighted_score_used": False,
        },
        "final_objective": _repair_objective_summary(final_objective),
        "R0_original_gates": gates,
        "final_structure": _relative_artifact_reference(
            final_path, base=output_dir
        ),
        "registered_interface_bonds": independent_validation["artifacts"][
            "registered_interface_bonds"
        ],
        "independent_validation": independent_validation,
        "cache_provenance": {
            "reuse": cache_reuse_contract["reuse"],
            "contract": cache_reuse_contract,
            **cache_artifacts,
        },
        "projection_cache": {
            "hit": projection_cache_manifest["hit"],
            "contract": projection_cache_manifest["contract"],
            "data_sha256": projection_cache_manifest.get("data_sha256"),
        },
        "forbidden_stages_not_run": [
            "MACE",
            "GPU",
            "LAMMPS",
            "protonation",
            "MD",
            "relaxation",
            "dynamics",
        ],
        "not_irreversible_sequential_history": True,
        "global_packing_claim": False,
        "timing": {
            "total_wall_time_seconds": time.perf_counter() - run_started,
            "local_search_wall_time_seconds": local["wall_time_seconds"],
        },
        "cpu_environment": _cpu_environment(),
    }
    manifest["artifact_inventory_scope"] = (
        "every_regular_output_file_before_top_manifest_seal;top_manifest_self_hash_excluded"
    )
    manifest["artifact_inventory"] = _artifact_inventory(
        output_dir, excluded_names={"manifest.json", "failure.json"}
    )
    run_request_verification = _verify_regular_artifact_reference(
        output_dir / "run-request.json",
        run_request_reference,
        label="run-request.json",
    )
    manifest["run_request"]["pre_top_manifest_seal_verification"] = {
        **run_request_verification,
        "path": "run-request.json",
    }
    manifest["code_snapshot"]["pre_top_manifest_seal_verification"] = (
        _verify_code_snapshot(output_dir, code_snapshot)
    )
    write_strict_json_noclobber(output_dir / "manifest.json", manifest)
    return 0 if independent_validation["passed"] else 1



_STAGE3_FIXED_COLLISION_THRESHOLDS_A = {
    "H-H": 1.5,
    "H-heavy": 1.8,
    "heavy-heavy": 2.2,
}


def _stage3_capture_stage2_parent_reference(stage2_run: Path) -> dict:
    root = Path(stage2_run).expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError(f"Stage2 parent is not a real directory: {root}")
    manifest_path = root / "manifest.json"
    top_identity = _hash_check_parent_file(
        manifest_path, {}, label="Stage2 top manifest", require_declared_hash=False
    )
    top = json.loads(manifest_path.read_text())
    if (
        top.get("schema") != "sam-local-repair-run-manifest-v1"
        or int(top.get("schema_version", -1)) != 1
        or top.get("sealed") is not True
        or top.get("role") != "postprocessed_rearranged_h0"
    ):
        raise ValueError("Stage2 parent is not a sealed v5 H0 repair package")
    final_path, final_reference = _resolve_parent_declared_path(
        top.get("final_structure"),
        parent_root=root,
        local_base=root,
        label="Stage2 final H0 structure",
    )
    final_identity = _hash_check_parent_file(
        final_path,
        final_reference,
        label="Stage2 final H0 structure",
        require_declared_hash=True,
    )
    ledger = {
        "manifest": top_identity,
        "final_structure": final_identity,
    }
    request_reference = top.get("run_request")
    if isinstance(request_reference, dict):
        request_path, request_ref = _resolve_parent_declared_path(
            request_reference,
            parent_root=root,
            local_base=root,
            label="Stage2 run request",
        )
        ledger["run_request"] = _hash_check_parent_file(
            request_path,
            request_ref,
            label="Stage2 run request",
            require_declared_hash=True,
        )
    objective = top.get("final_objective")
    if not isinstance(objective, dict):
        raise ValueError("Stage2 parent lacks its sealed final objective")
    if int(objective.get("molecule_count_N", -1)) != 56:
        raise ValueError("Stage2 v5 parent molecule count is not N=56")
    r0 = top.get("R0_original_gates")
    if not isinstance(r0, dict) or r0.get("primary", {}).get("passed") is not False:
        raise ValueError("Stage2 v5 parent does not preserve failed R0 primary evidence")
    if int(r0.get("R0_original_minimum_molecule_count", -1)) != 97:
        raise ValueError("Stage2 v5 parent R0 threshold is not 97")
    return {
        "schema": "sam-stage3-stage2-parent-reference-v1",
        "parent_run": str(root),
        "manifest": top,
        "manifest_identity": top_identity,
        "final_path": str(final_path),
        "final_identity": final_identity,
        "final_objective": objective,
        "R0_original_gates": r0,
        "hash_ledger": {
            label: {
                "path": identity["path"],
                "sha256": identity["sha256"],
                "bytes": int(identity["bytes"]),
            }
            for label, identity in sorted(ledger.items())
        },
        "parent_tree_ledger": _capture_directory_tree_ledger(root),
    }


def _stage3_capture_analysis_provenance(analysis_path: Path) -> dict:
    report_path = Path(analysis_path).expanduser().resolve()
    if report_path.is_symlink() or not report_path.is_file():
        raise FileNotFoundError(f"Stage3 analysis report is not a regular file: {report_path}")
    manifest_path = report_path.parent / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise FileNotFoundError("Stage3 analysis manifest is missing")
    manifest_identity = _file_identity(manifest_path)
    analysis_manifest = json.loads(manifest_path.read_text())
    report_ref = analysis_manifest.get("report")
    declared_report_path, declared_report_ref = _resolve_parent_declared_path(
        report_ref,
        parent_root=manifest_path.parent,
        local_base=manifest_path.parent,
        label="Stage3 analysis report",
    )
    if declared_report_path != report_path:
        raise ValueError("Stage3 analysis report path disagrees with its manifest")
    report_identity = _hash_check_parent_file(
        report_path,
        declared_report_ref,
        label="Stage3 analysis report",
        require_declared_hash=True,
    )
    if int(analysis_manifest.get("geometry_cluster_count", -1)) != 56:
        raise ValueError("Stage3 analysis provenance does not contain 56 clusters")
    report = json.loads(report_path.read_text())
    clustering_section = report.get("geometry_clustering")
    clusters = clustering_section.get("clusters") if isinstance(clustering_section, dict) else None
    if not isinstance(clusters, list) or len(clusters) != 56:
        raise ValueError("Stage3 analysis report must contain all 56 cluster records")
    clusters = sorted(clusters, key=lambda record: int(record.get("cluster_id", -1)))
    if [int(record.get("cluster_id", -1)) for record in clusters] != list(range(1, 57)):
        raise ValueError("Stage3 analysis cluster IDs are not exactly 1..56")
    source_records = []
    for record in clusters:
        structure = record.get("representative_structure")
        path, reference = _resolve_parent_declared_path(
            structure,
            parent_root=report_path.parent,
            local_base=report_path.parent,
            label=f"cluster {record['cluster_id']} representative",
        )
        identity = _hash_check_parent_file(
            path,
            reference,
            label=f"cluster {record['cluster_id']} representative",
            require_declared_hash=True,
        )
        task_index = int(record["representative_task_index"])
        cluster_id = int(record["cluster_id"])
        source_records.append({
            "source_key": f"cluster-{cluster_id:04d}-task-{task_index:04d}",
            "cluster_id": cluster_id,
            "representative_task_index": task_index,
            "path": str(path),
            "identity": identity,
            "relative_total_energy_eV": float(record["representative_relative_total_energy_eV"]),
            "reported_footprint": record.get("footprint"),
        })
    selected = report.get("selection", {}).get("selected")
    if not isinstance(selected, list) or len(selected) < 10:
        raise ValueError("Stage3 analysis report lacks the sealed top-10 selection")
    selected_records = []
    for expected_rank, record in enumerate(selected[:10], 1):
        if int(record.get("selection_rank", -1)) != expected_rank:
            raise ValueError("Stage3 top-10 selection ranks are not exactly 1..10")
        structure = record.get("structure")
        path, reference = _resolve_parent_declared_path(
            structure,
            parent_root=report_path.parent,
            local_base=report_path.parent,
            label=f"top-10 selection rank {expected_rank}",
        )
        identity = _hash_check_parent_file(
            path,
            reference,
            label=f"top-10 selection rank {expected_rank}",
            require_declared_hash=True,
        )
        selected_records.append({
            "selection_rank": expected_rank,
            "cluster_id": int(record["cluster_id"]),
            "representative_task_index": int(record["representative_task_index"]),
            "path": str(path),
            "identity": identity,
            "relative_total_energy_eV": float(record["relative_total_energy_eV"]),
            "footprint_area_A2": float(record["footprint_area_A2"]),
        })
    return {
        "schema": "sam-stage3-analysis-provenance-v1",
        "analysis_manifest": analysis_manifest,
        "analysis_manifest_identity": manifest_identity,
        "analysis_report": report,
        "analysis_report_identity": report_identity,
        "source_records": source_records,
        "selected_top10": selected_records,
    }


def _stage3_capture_reference_metrics_provenance(path: Path) -> dict:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError("Stage3 corrected reference metrics is not a regular file")
    manifest_path = path.parent / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise FileNotFoundError("Stage3 corrected reference metrics sibling manifest is missing")
    manifest_identity = _file_identity(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("sealed") is not True
        or manifest.get("stage") != "Stage1-R0"
        or manifest.get("corrected_reference_metric_gate", {}).get("passed") is not True
    ):
        raise ValueError("Corrected reference metrics sibling manifest is not sealed passed evidence")
    declared_path, declared_reference = _resolve_parent_declared_path(
        manifest.get("reference_metrics"),
        parent_root=manifest_path.parent,
        local_base=manifest_path.parent,
        label="Stage3 corrected reference metrics",
    )
    if declared_path != path:
        raise ValueError("Corrected reference metrics path disagrees with its manifest")
    metrics_identity = _hash_check_parent_file(
        path,
        declared_reference,
        label="Stage3 corrected reference metrics",
        require_declared_hash=True,
    )
    inventory = manifest.get("artifact_inventory") or []
    matches = [record for record in inventory if record.get("path") == "reference-metrics.json"]
    if len(matches) != 1 or {
        "sha256": matches[0].get("sha256"),
        "bytes": matches[0].get("bytes"),
    } != {
        "sha256": metrics_identity["sha256"],
        "bytes": metrics_identity["bytes"],
    }:
        raise ValueError("Corrected reference metrics manifest inventory does not match payload")
    payload = json.loads(path.read_text())
    if (
        payload.get("sealed") is not True
        or payload.get("stage") != "Stage1-R0"
        or payload.get("metric_contract", {}).get("primary_metric") != "periodic-torus-holes-v2"
        or payload.get("corrected_reference_metric_gate", {}).get("passed") is not True
    ):
        raise ValueError("Corrected reference metrics payload is not sealed torus-v2 evidence")
    return {
        "manifest": {"path": str(manifest_path), **manifest_identity},
        "payload": {"path": str(path), **metrics_identity},
        "manifest_schema": manifest.get("schema"),
        "payload_schema": payload.get("schema"),
        "corrected_reference_metric_gate": payload.get("corrected_reference_metric_gate"),
    }


def _stage3_capture_derived_pool_provenance(path: Path) -> dict:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"Stage3 derived-pool manifest is not a regular file: {path}")
    identity = _file_identity(path)
    document = json.loads(path.read_text())
    if document.get("schema") != "sam-stage3-derived-pool-manifest-v1":
        raise ValueError("Explicit Stage3 derived-pool provenance has an unsupported schema")
    members = document.get("members")
    if not isinstance(members, list) or not members:
        raise ValueError("Explicit Stage3 derived-pool provenance has no members")
    verified_members = []
    for member in members:
        if not isinstance(member, dict):
            raise ValueError("Explicit Stage3 derived-pool member evidence is malformed")
        source_path = Path(member.get("source_path", "")).expanduser()
        if not source_path.is_absolute():
            source_path = path.parent / source_path
        source_path = source_path.resolve()
        expected = member.get("source_identity")
        actual = _hash_check_parent_file(
            source_path,
            expected if isinstance(expected, dict) else {},
            label="explicit Stage3 derived-pool source",
            require_declared_hash=True,
        )
        verified_members.append({
            "source_key": str(member.get("source_key", "")),
            "pool_rank": int(member.get("pool_rank", -1)),
            "source_identity": actual,
        })
    return {
        "schema": "sam-stage3-explicit-derived-pool-provenance-v1",
        "manifest": {"path": str(path), **identity},
        "pool_fingerprint": document.get("pool_fingerprint"),
        "pool_index": document.get("pool_index"),
        "member_count": len(verified_members),
        "verified_members": verified_members,
        "intake_role": "explicit_provenance_declaration_only_default_top10_intake_unchanged",
    }


def _stage3_copy_exact_source(source: Path, destination: Path) -> dict:
    source = Path(source)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"Stage3 source is not a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_install_noclobber_bytes(destination, source.read_bytes())
    identity = _file_identity(destination)
    source_identity = _file_identity(source)
    if identity["sha256"] != source_identity["sha256"] or identity["bytes"] != source_identity["bytes"]:
        raise RuntimeError("Stage3 exact source copy failed hash/byte verification")
    return identity


def _stage3_shape_record_from_source(
    source_record: dict,
    *,
    molecule_spec: MoleculeSpec,
    surface_frame: dict,
    boundary_samples_per_atom: int,
    radius_scale: float,
) -> dict:
    atoms = read_typed_structure(Path(source_record["path"]))
    components = molecular_components(atoms, molecule_spec)
    if len(components) != 1:
        raise ValueError(
            f"Stage3 representative {source_record['source_key']} has {len(components)} molecule components"
        )
    coordinates, symbols = unwrap_component(
        atoms, components[0], molecule_spec.hydrogen_parent_elements
    )
    radii = _resolved_ase_vdw_radii(symbols)
    metrics = filled_outer_envelope_metrics(
        positions=coordinates,
        symbols=symbols,
        surface_frame=surface_frame,
        radii_A=radii,
        radius_scale=radius_scale,
        boundary_samples_per_atom=boundary_samples_per_atom,
    )
    return {
        **source_record,
        "shape_metrics": metrics,
        "geometry_hash_basis": "unwrapped_single_molecule_coordinates_and_symbols",
    }


def _stage3_periodic_distance_A(candidate_uv, hole_centroid_fractional, lattice):
    candidate_fractional = np.linalg.solve(lattice, np.asarray(candidate_uv, dtype=float))
    candidate_fractional = candidate_fractional - np.floor(candidate_fractional)
    hole = np.asarray(hole_centroid_fractional, dtype=float)
    distances = []
    for du in (-1, 0, 1):
        for dv in (-1, 0, 1):
            delta = candidate_fractional - hole + np.asarray([du, dv], dtype=float)
            distances.append(float(np.linalg.norm(lattice @ delta)))
    return min(distances)


def _stage3_clone_projection_cache(
    source_cache_dir: Path,
    destination_cache_dir: Path,
    *,
    cell,
    periodic_axes,
    surface_frame,
) -> tuple[ProjectionCache, dict]:
    """Copy a sealed top-10 projection cache into an independent pool namespace."""
    source_cache_dir = Path(source_cache_dir).expanduser().resolve()
    destination_cache_dir = Path(destination_cache_dir)
    if source_cache_dir.is_symlink() or not source_cache_dir.is_dir():
        raise FileNotFoundError(f"Stage3 source projection cache is not a real directory: {source_cache_dir}")
    destination_cache_dir.mkdir(parents=True, exist_ok=False)
    source_manifest_path = source_cache_dir / "projection-cache-manifest.json"
    source_data_path = source_cache_dir / "projection-cache-data.npz"
    source_manifest_identity = _file_identity(source_manifest_path)
    source_data_identity = _file_identity(source_data_path)
    manifest = json.loads(source_manifest_path.read_text())
    if (manifest.get("contract") or {}).get("schema") != "projection-cache-v1":
        raise ValueError("Stage3 source projection cache schema is unsupported")
    manifest_reference = _stage3_copy_exact_source(
        source_manifest_path, destination_cache_dir / source_manifest_path.name
    )
    data_reference = _stage3_copy_exact_source(
        source_data_path, destination_cache_dir / source_data_path.name
    )
    data = np.load(destination_cache_dir / source_data_path.name)
    try:
        relative_vertices = data["relative_vertices"]
        template_index = data["template_index"]
        anchor_uv = data["anchor_uv"]
        site_anchor_fractional = data["site_anchor_fractional"]
    finally:
        data.close()
    template_records = manifest.get("template_records", [])
    vertex_counts = [len(record["relative_vertices_uv_A"]) for record in template_records]
    candidate_geometries, coverage_masks = {}, {}
    for position in range(len(template_index)):
        template_position = int(template_index[position])
        vertices_uv_A = (
            relative_vertices[template_position, : vertex_counts[template_position]]
            + anchor_uv[position]
        )
        geometry, _ = periodic_surface_polygon(
            vertices_uv_A,
            cell=cell,
            periodic_axes=periodic_axes,
            surface_frame=surface_frame,
        )
        candidate_geometries[position] = geometry
        coverage_masks[position] = periodic_polygon_coverage_mask(
            geometry, site_anchor_fractional
        )
    lattice, area_scale = _surface_lattice_uv(cell, periodic_axes, surface_frame)
    cache = ProjectionCache(
        site_ids=manifest["site_ids"],
        site_index={site_id: index for index, site_id in enumerate(manifest["site_ids"])},
        candidate_ids=manifest["candidate_ids"],
        coverage_masks=coverage_masks,
        candidate_geometries=candidate_geometries,
        template_records=template_records,
        area_scale_A2=area_scale,
        surface_lattice_uv_A=lattice,
        contract=manifest["contract"],
        manifest={
            **manifest,
            "hit": False,
            "rebuild_reason": "exact sealed Stage2 top10 cache copied into independent Stage3 pool namespace",
        },
        relative_vertices=relative_vertices,
        template_index=template_index,
        anchor_uv=anchor_uv,
        site_anchor_fractional=site_anchor_fractional,
    )
    return cache, {
        "source_manifest": source_manifest_identity,
        "source_data": source_data_identity,
        "copied_manifest": manifest_reference,
        "copied_data": data_reference,
        "reuse": "exact_parent_cache_copy_bound_to_independent_pool_namespace",
    }


def execute_stage3_diagnostic(
    args,
    *,
    argv: list[str],
    root: Path,
    output_dir: Path,
    resolved_paths: dict[str, Path],
    implementation_identity: dict,
) -> int:
    """Run the CPU-only Stage3 provenance, shape, and pool diagnostic."""

    if (
        args.substrate_collision != "strict-vdw"
        or args.collision_cache != "off"
        or int(args.capacity_beam_width) != 32
        or args.max_conformers != 10
        or args.trajectory_count != 1
    ):
        raise ValueError(
            "Stage3 requires strict-vdw, collision-cache off, beam width 32, "
            "max-conformers 10, and one trajectory"
        )
    if {
        key: float(value)
        for key, value in {
            "H-H": args.hh_min,
            "H-heavy": args.h_heavy_min,
            "heavy-heavy": args.heavy_heavy_min,
        }.items()
    } != _STAGE3_FIXED_COLLISION_THRESHOLDS_A:
        raise ValueError("Stage3 collision thresholds must remain exactly 1.5/1.8/2.2")
    if args.max_conformers <= 0 or args.stage3_max_pool_members < 10:
        raise ValueError("Stage3 pool bounds are invalid")
    if args.stage3_max_pool_members > 56:
        raise ValueError("Stage3 pool cannot exceed the 56 validated representatives")
    def required_path(value, label):
        if value is None:
            raise ValueError(f"Stage3 requires --{label}")
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if path.is_symlink():
            raise FileNotFoundError(f"Stage3 {label} must not be a symlink: {path}")
        return path

    explicit_pool_provenance = None
    if args.stage3_derived_pool_manifest is not None:
        explicit_pool_provenance = _stage3_capture_derived_pool_provenance(
            required_path(args.stage3_derived_pool_manifest, "stage3-derived-pool-manifest")
        )
    stage1_path = required_path(args.stage1_parent_run, "stage1-parent-run")
    stage2_path = required_path(args.stage2_parent_run, "stage2-parent-run")
    reference_path = required_path(
        args.corrected_reference_metrics, "corrected-reference-metrics"
    )
    substrate_path = resolved_paths["substrate"]
    layer_groups_path = resolved_paths["layer_groups"]
    site_instances_path = resolved_paths["site_instances"]
    site_prototypes_path = resolved_paths["site_prototypes"]
    analysis_path = resolved_paths["conformer_analysis"]
    primary_inputs = {
        name: _file_identity(resolved_paths[name])
        for name in _PARENT_PRIMARY_INPUT_NAMES
    }
    stage1_reference = capture_immutable_parent_reference(
        stage1_path, expected_inputs=primary_inputs
    )
    stage2_reference = _stage3_capture_stage2_parent_reference(stage2_path)
    parent_metric_contract = _parent_requested_metric_contract(stage1_reference)["contract"]
    if {
        "footprint_boundary_samples": int(args.footprint_boundary_samples),
        "projection_radius_scale": float(args.projection_radius_scale),
        "area_tie_tolerance_A2": float(args.area_tie_tolerance_A2),
        "perimeter_tie_tolerance_A": float(args.perimeter_tie_tolerance_A),
    } != parent_metric_contract:
        raise ValueError("Stage3 projection metric contract differs from sealed Stage1/Stage2 contract")
    analysis_provenance = _stage3_capture_analysis_provenance(analysis_path)
    reference_provenance = _stage3_capture_reference_metrics_provenance(reference_path)
    reference_identity = reference_provenance["payload"]

    substrate, layer_groups = _load_working_substrate(
        substrate_path, layer_groups_path
    )
    site_payload = json.loads(site_instances_path.read_text())
    prototypes = _prototype_map(site_prototypes_path)
    periodic_axes = resolve_periodic_fractional_axes(site_payload, prototypes)
    normal_axis = _validated_layer_normal_axis(layer_groups, periodic_axes)
    full_to_working_atom_map = build_full_to_working_atom_map(
        working_layer_atom_ids_1based=layer_groups["working_layer_atom_ids_1based"],
        working_symbols=substrate.get_chemical_symbols(),
    )
    spec = MoleculeSpec(
        formula=_json_formula(args.molecule_formula_json) or {},
        skeleton_formula=_json_formula(args.skeleton_formula_json),
    )
    surface_frame = next(iter(prototypes.values())).get("surface_frame")
    if not isinstance(surface_frame, dict):
        raise ValueError("Stage3 lacks a registered prototype Surface Frame")
    shape_records = [
        _stage3_shape_record_from_source(
            record,
            molecule_spec=spec,
            surface_frame=surface_frame,
            boundary_samples_per_atom=int(args.footprint_boundary_samples),
            radius_scale=float(args.substrate_vdw_radius_scale),
        )
        for record in analysis_provenance["source_records"]
    ]
    shape_records.sort(key=lambda record: record["source_key"])
    feature_payload = stage3_standardized_shape_features(shape_records)
    clustering = stage3_complete_link_clusters(
        shape_records,
        feature_payload["standardized_features"],
        threshold=float(args.stage3_shape_cluster_threshold),
    )
    shape_by_cluster = {
        int(record["cluster_id"]): record for record in shape_records
    }
    selected_records = analysis_provenance["selected_top10"]
    selected_keys = []
    for record in selected_records:
        source = shape_by_cluster.get(int(record["cluster_id"]))
        if source is None:
            raise ValueError("Stage3 top-10 cluster is absent from 56 shape records")
        selected_keys.append(source["source_key"])
    if len(selected_keys) != len(set(selected_keys)):
        raise ValueError("Stage3 top-10 selection does not cover unique clusters")

    stage1_manifest = stage1_reference["top_manifest"]
    stage1_domain = stage1_manifest.get("candidate_domain") or {}
    baseline_nonempty = int(stage1_domain.get("site_count", -1)) - int(
        stage1_domain.get("empty_site_count", -1)
    )
    if baseline_nonempty != 82:
        raise ValueError(
            f"Stage1 strict baseline nonempty-site count is {baseline_nonempty}, expected 82"
        )
    stage2_capacity = stage2_reference["manifest"].get("capacity") or {}
    baseline_static_k = int(stage2_capacity.get("final", {}).get("certified_static_K", -1))
    if baseline_static_k != 56:
        raise ValueError(f"Stage2 v5 static K is {baseline_static_k}, expected 56")
    stage1_aggregate = stage1_manifest.get("aggregate") or {}
    baseline_n = int((stage1_aggregate.get("emergent_molecule_counts") or [])[0])
    accepted_ranks = {
        int(record["selection_rank"])
        for record in stage1_aggregate.get("conformer_rank_distribution", [])
        if int(record.get("count", 0)) > 0
    }
    if baseline_n != 56:
        raise ValueError("Stage1 parent N is not 56")

    top10_shape_clusters = {int(clustering["assignments"][key]) for key in selected_keys}
    shape_coverage_gate = {
        "passed": len(top10_shape_clusters) == len(selected_keys),
        "threshold": "top-10 slots must cover 10 distinct deterministic shape clusters",
        "threshold_source": "Stage3 diagnostic contract; no preselected cluster count",
        "top10_unique_shape_cluster_count": len(top10_shape_clusters),
        "top10_count": len(selected_keys),
        "all56_shape_cluster_count": int(clustering["cluster_count"]),
        "coverage_fraction_of_all56_clusters": (
            len(top10_shape_clusters) / int(clustering["cluster_count"])
            if clustering["cluster_count"]
            else 0.0
        ),
        "conclusion": (
            "top10 shape slots are distinct"
            if len(top10_shape_clusters) == len(selected_keys)
            else "top10 contains a deterministic shape-cluster duplicate"
        ),
    }
    top10_gate = {
        "schema": "sam-stage3-auditable-top10-gate-v1",
        "passed": all(
            [
                baseline_n < 97,
                baseline_nonempty == 82,
                baseline_static_k == 56,
                accepted_ranks.issubset({1, 2, 3}),
                shape_coverage_gate["passed"],
            ]
        ),
        "checks": {
            "parent_N_below_97": {
                "passed": baseline_n < 97,
                "observed": baseline_n,
                "threshold": "N < 97",
                "threshold_source": "Stage2 v5 R0_original_gates minimum molecule count",
            },
            "strict_domain_nonempty_sites_82_of_448": {
                "passed": baseline_nonempty == 82,
                "observed_nonempty_sites": baseline_nonempty,
                "observed_site_count": int(stage1_domain.get("site_count", -1)),
                "threshold": "nonempty_sites == 82 and site_count == 448",
                "threshold_source": "sealed Stage1 candidate_domain manifest",
            },
            "static_K_equals_56": {
                "passed": baseline_static_k == 56,
                "observed": baseline_static_k,
                "threshold": "static_K == 56",
                "threshold_source": "sealed Stage2 v5 capacity.final certificate",
            },
            "accepted_ranks_only_1_to_3": {
                "passed": accepted_ranks.issubset({1, 2, 3}),
                "observed_nonzero_ranks": sorted(accepted_ranks),
                "threshold": "set(nonzero accepted ranks) subset of {1,2,3}",
                "threshold_source": "sealed Stage1 aggregate conformer rank distribution",
            },
            "shape_cluster_coverage": shape_coverage_gate,
        },
        "conclusion": "top10 gate is auditable and not an energy/footprint-only decision",
    }

    strict_collision_contract = build_substrate_collision_contract(
        mode="strict-vdw",
        periodic_axes=periodic_axes,
        mapped_bond_window_A=(args.mapped_bond_min_A, args.mapped_bond_max_A),
        hard_thresholds_A=_STAGE3_FIXED_COLLISION_THRESHOLDS_A,
        substrate_symbols=substrate.get_chemical_symbols(),
        molecule_symbols=list(spec.formula),
        vdw_radius_scale=float(args.substrate_vdw_radius_scale),
    )
    conflicts = _conflict_map(site_payload)
    metal_constraints = {
        str(instance["site_instance_id"]): {
            int(atom) for atom in instance.get("actual_metal_atom_ids_1based", [])
        }
        for instance in site_payload["site_instances"]
    }
    metric_by_cluster = {
        int(record["cluster_id"]): {
            "cluster_id": int(record["cluster_id"]),
            "relative_total_energy_eV": float(record["relative_total_energy_eV"]),
            "footprint_area_A2": float(record["shape_metrics"]["area_A2"]),
        }
        for record in shape_records
    }
    selected_metric_by_rank = {
        int(record["selection_rank"]): {
            "cluster_id": int(record["cluster_id"]),
            "relative_total_energy_eV": float(record["relative_total_energy_eV"]),
            "footprint_area_A2": float(record["footprint_area_A2"]),
        }
        for record in selected_records
    }
    common_domain_kwargs = {
        "substrate": substrate,
        "site_payload": site_payload,
        "prototypes": prototypes,
        "spec": spec,
        "headgroup_rmsd_max_A": float(args.headgroup_rmsd_max_A),
        "full_to_working_atom_map": full_to_working_atom_map,
        "mapped_bond_window_A": (args.mapped_bond_min_A, args.mapped_bond_max_A),
        "substrate_vdw_radius_scale": float(args.substrate_vdw_radius_scale),
        "periodic_axes": periodic_axes,
        "collision_thresholds_A": _STAGE3_FIXED_COLLISION_THRESHOLDS_A,
        "substrate_collision_mode": "strict-vdw",
    }

    request_document = {
        "schema": "sam-stage3-diagnostic-run-request-v1",
        "status": "sealed_before_stage3_geometry",
        "operation": "stage3-diagnostic",
        "argv": [str(value) for value in argv],
        "primary_inputs": primary_inputs,
        "stage1_parent_manifest": stage1_reference["hash_ledger_before"]["top_manifest"],
        "stage2_parent_manifest": stage2_reference["manifest_identity"],
        "corrected_reference_metrics": reference_provenance,
        "explicit_derived_pool_provenance": explicit_pool_provenance,
        "contracts": {
            "collision_thresholds_A": _STAGE3_FIXED_COLLISION_THRESHOLDS_A,
            "substrate_collision_mode": "strict-vdw",
            "substrate_vdw_radius_scale": float(args.substrate_vdw_radius_scale),
            "beam_width": 32,
            "collision_cache": "off",
            "footprint_boundary_samples": int(args.footprint_boundary_samples),
            "shape_cluster_threshold": float(args.stage3_shape_cluster_threshold),
        },
        "forbidden_stages_not_run": ["MACE", "GPU", "LAMMPS", "protonation", "MD"],
        "implementation": implementation_identity,
    }
    request_reference = write_strict_json_noclobber(
        output_dir / "run-request.json", request_document
    )

    parent_conformer_cache = output_dir / "parent-domain-cache" / "conformers"
    parent_conformers = source_conformers(
        root,
        args.conformer_glob,
        parent_conformer_cache,
        spec,
        minimum_vertical_angle=0.0,
        maximum_marker_exposure=float("inf"),
        maximum_source_clusters=10,
        immutable_new_cache=True,
    )
    parent_conformer_by_rank = {
        int(conformer["source_rank"]): conformer for conformer in parent_conformers
    }
    parent_step_candidate_ids = []
    for step in stage1_reference["trajectory_manifest"].get("steps", []):
        rank = int(step["selection_rank"])
        conformer = parent_conformer_by_rank.get(rank)
        if conformer is None:
            raise ValueError("Stage1 accepted rank is absent from the sealed top-10 conformers")
        prototype = prototypes.get(str(step["site_prototype_id"]))
        if prototype is None:
            raise ValueError("Stage1 accepted step references an unknown prototype")
        candidate_id = _candidate_id_from_fields(
            site_instance_id=str(step["site_instance_id"]),
            site_prototype_id=str(step["site_prototype_id"]),
            prototype_evidence=str(
                prototype.get("representative_candidate_id", step["site_prototype_id"])
            ),
            selection_rank=rank,
            conformer_sha256=_conformer_geometry_sha256(
                np.asarray(conformer["coordinates"], dtype=float),
                np.asarray(conformer["symbols"], dtype=str),
            ),
            donor_labels=step["target_donor_labels"],
            oxygen_permutation=step["oxygen_permutation"],
            symbols=np.asarray(conformer["symbols"], dtype=str),
        )
        parent_step_candidate_ids.append(candidate_id)

    def evaluate_pool(pool_index, member_keys, member_records, base_pool=None):
        pool_root = output_dir / "derived-pools" / f"pool-{pool_index:04d}"
        pool_root.mkdir(parents=True, exist_ok=False)
        conformer_dir = pool_root / "conformers"
        conformer_dir.mkdir(exist_ok=False)
        copied_members = []
        for rank, (key, member) in enumerate(zip(member_keys, member_records), 1):
            destination = conformer_dir / (
                f"pool-rank-{rank:04d}-{key}.extxyz"
            )
            copied_identity = _stage3_copy_exact_source(
                Path(member["path"]), destination
            )
            copied_members.append({
                "pool_rank": rank,
                "source_key": key,
                "cluster_id": int(member["cluster_id"]),
                "source_path": str(Path(member["path"]).resolve()),
                "source_identity": member["identity"],
                "copied_path": str(destination.relative_to(output_dir)),
                "copied_identity": copied_identity,
                "relative_total_energy_eV": float(member["relative_total_energy_eV"]),
                "shape_cluster_id": int(clustering["assignments"][key]),
            })
        pool_fingerprint = _payload_sha256({
            "schema": "sam-stage3-independent-pool-fingerprint-v1",
            "members": [
                {
                    "pool_rank": item["pool_rank"],
                    "source_key": item["source_key"],
                    "sha256": item["copied_identity"]["sha256"],
                    "bytes": item["copied_identity"]["bytes"],
                }
                for item in copied_members
            ],
            "collision_contract": strict_collision_contract,
        })
        # The sealed Stage2 top-10 projection is copied for pool 0.  This
        # avoids silently reusing its external namespace while preserving the
        # exact parent hole partition.  Only shape-expanded pools regenerate
        # the strict-vdW candidate domain.
        conformer_cache_dir = pool_root / "internal-cache" / "conformers"
        if pool_index == 0:
            pool_conformers = source_conformers(
                pool_root,
                "conformers/*.extxyz",
                conformer_cache_dir,
                spec,
                minimum_vertical_angle=0.0,
                maximum_marker_exposure=float("inf"),
                maximum_source_clusters=len(copied_members),
                immutable_new_cache=True,
            )
            projection_cache, projection_copy_evidence = _stage3_clone_projection_cache(
                stage2_path / "internal-cache" / "projection",
                pool_root / "internal-cache" / "projection",
                cell=substrate.cell,
                periodic_axes=periodic_axes,
                surface_frame=surface_frame,
            )
            domains = None
            domain_diagnostics = {
                "candidate_domain_fingerprint": stage1_domain.get("fingerprint"),
                "site_count": int(stage1_domain["site_count"]),
                "candidate_count": int(stage1_domain.get("candidate_count", -1)),
                "empty_site_count": int(stage1_domain["empty_site_count"]),
                "rejection_counts": {"source": "sealed Stage1 candidate-domain manifest"},
            }
            feasible_site_ids = [
                f"sealed-nonempty-site-{index:04d}"
                for index in range(baseline_nonempty)
            ]
            static_capacity = {
                "status": "sealed_parent_certificate_reused_for_pool0_diagnostic",
                "objective": baseline_static_k,
                "optimality_certified": True,
                "source": stage2_reference["manifest"].get("capacity", {}).get("final", {}),
            }
        else:
            pool_conformers = source_conformers(
                pool_root,
                "conformers/*.extxyz",
                conformer_cache_dir,
                spec,
                minimum_vertical_angle=0.0,
                maximum_marker_exposure=float("inf"),
                maximum_source_clusters=len(copied_members),
                immutable_new_cache=True,
            )
            pool_metric_by_rank = {
                int(item["pool_rank"]): metric_by_cluster[int(item["cluster_id"])]
                for item in copied_members
            }
            projection_copy_evidence = None
            base_domains = base_pool.get("domains") if base_pool is not None else None
            if base_domains is None and empty_site_payload is not None:
                new_rank = len(copied_members)
                cheap_screen_domains, cheap_screen_diagnostics = _candidate_domains(
                    conformers=[pool_conformers[-1]],
                    metrics_by_rank={new_rank: pool_metric_by_rank[new_rank]},
                    **{
                        **common_domain_kwargs,
                        "site_payload": empty_site_payload,
                        "substrate_collision_mode": "skip",
                    },
                )
                cheap_screen_candidate_count = sum(
                    len(candidates) for candidates in cheap_screen_domains.values()
                )
                if cheap_screen_candidate_count == 0:
                    screen_domains, screen_diagnostics = cheap_screen_domains, cheap_screen_diagnostics
                    screen_candidate_count = 0
                    strict_screen_performed = False
                else:
                    screen_domains, screen_diagnostics = _candidate_domains(
                        conformers=[pool_conformers[-1]],
                        metrics_by_rank={new_rank: pool_metric_by_rank[new_rank]},
                        **{**common_domain_kwargs, "site_payload": empty_site_payload},
                    )
                    screen_candidate_count = sum(len(candidates) for candidates in screen_domains.values())
                    strict_screen_performed = True
                if screen_candidate_count == 0:
                    domains = None
                    domain_diagnostics = {
                        "candidate_domain_fingerprint": None,
                        "site_count": int(stage1_domain["site_count"]),
                        "candidate_count": 0,
                        "empty_site_count": int(stage1_domain["empty_site_count"]),
                        "rejection_counts": screen_diagnostics["rejection_counts"],
                        "screening_only": True,
                        "screened_site_count": int(len(empty_site_payload["site_instances"])),
                        "cheap_prerequisite_candidate_count": int(cheap_screen_candidate_count),
                        "strict_vdw_screen_performed": bool(strict_screen_performed),
                        "screen_conclusion": "new conformer added no strict-vdW candidate on a previously empty site",
                    }
                    feasible_site_ids = [
                        f"sealed-nonempty-site-{index:04d}"
                        for index in range(baseline_nonempty)
                    ]
                    static_capacity = {
                        "status": "not_run_screening_negative",
                        "objective": baseline_static_k,
                        "optimality_certified": True,
                    }
                    projection_cache = None
                else:
                    domains, domain_diagnostics = _candidate_domains(
                        conformers=pool_conformers,
                        metrics_by_rank=pool_metric_by_rank,
                        **common_domain_kwargs,
                    )
            elif base_domains is not None:
                new_rank = len(copied_members)
                new_domains, new_diagnostics = _candidate_domains(
                    conformers=[pool_conformers[-1]],
                    metrics_by_rank={new_rank: pool_metric_by_rank[new_rank]},
                    **common_domain_kwargs,
                )
                domains = {
                    str(instance["site_instance_id"]): list(base_domains.get(str(instance["site_instance_id"]), []))
                    + list(new_domains.get(str(instance["site_instance_id"]), []))
                    for instance in site_payload["site_instances"]
                }
                all_candidate_ids = sorted(
                    candidate["candidate_id"]
                    for candidates in domains.values()
                    for candidate in candidates
                )
                if len(all_candidate_ids) != len(set(all_candidate_ids)):
                    raise RuntimeError("Stage3 incremental domain produced duplicate candidate IDs")
                index_by_id = {candidate_id: index for index, candidate_id in enumerate(all_candidate_ids)}
                for candidates in domains.values():
                    for candidate in candidates:
                        candidate["candidate_index"] = index_by_id[candidate["candidate_id"]]
                merged_rejections = Counter(base_pool["domain_diagnostics"]["rejection_counts"])
                merged_rejections.update(new_diagnostics["rejection_counts"])
                domain_diagnostics = {
                    **new_diagnostics,
                    "site_count": len(domains),
                    "candidate_count": sum(len(candidates) for candidates in domains.values()),
                    "empty_site_count": sum(not candidates for candidates in domains.values()),
                    "rejection_counts": dict(sorted(merged_rejections.items())),
                    "candidate_domain_fingerprint": _candidate_domain_fingerprint(
                        domains,
                        cell=substrate.cell,
                        periodic_axes=periodic_axes,
                        substrate_collision_contract=strict_collision_contract,
                    ),
                    "incremental_from_pool_index": int(base_pool["pool_index"]),
                }
            else:
                domains, domain_diagnostics = _candidate_domains(
                    conformers=pool_conformers,
                    metrics_by_rank=pool_metric_by_rank,
                    **common_domain_kwargs,
                )
            if domains is not None:
                projection_cache = _load_or_build_projection_cache(
                    domains,
                    site_payload=site_payload,
                    cell=substrate.cell,
                    periodic_axes=periodic_axes,
                    cache_dir=pool_root / "internal-cache" / "projection",
                    boundary_samples_per_atom=int(args.footprint_boundary_samples),
                    radius_scale=float(args.projection_radius_scale),
                    prototypes=prototypes,
                    substrate_collision_contract=strict_collision_contract,
                    immutable_new_cache=True,
                )
                feasible_site_ids = [site_id for site_id, candidates in domains.items() if candidates]
                static_capacity = solve_static_conflict_capacity(
                    feasible_site_ids,
                    metal_constraints,
                    site_conflicts=conflicts,
                    time_limit_seconds=float(args.stage3_capacity_time_limit_s),
                )
            projection_cache_files = []
        conformer_cache_files = [
            _file_identity(pool_root / "internal-cache" / "conformers.json"),
            _file_identity(pool_root / "internal-cache" / "conformers.npz"),
        ]
        projection_cache_files = (
            [
                _file_identity(pool_root / "internal-cache" / "projection" / name)
                for name in ("projection-cache-data.npz", "projection-cache-manifest.json")
            ]
            if projection_cache is not None
            else []
        )
        pool_manifest = {
            "schema": "sam-stage3-derived-pool-manifest-v1",
            "pool_index": int(pool_index),
            "pool_fingerprint": pool_fingerprint,
            "pool_selection_source": "stage2-top10-then-deterministic-shape-diversity-expansion",
            "members": copied_members,
            "shape_cluster_assignments": {
                key: int(clustering["assignments"][key]) for key in member_keys
            },
            "cache_namespace": {
                "root": str(pool_root.relative_to(output_dir)),
                "conformer_cache": [
                    {
                        **identity,
                        "path": str(Path(identity["path"]).relative_to(output_dir)),
                    }
                    for identity in conformer_cache_files
                ],
                "projection_cache": [
                    {
                        **identity,
                        "path": str(Path(identity["path"]).relative_to(output_dir)),
                    }
                    for identity in projection_cache_files
                ],
                "independent_pool_fingerprint_bound": True,
                "projection_cache_copy_evidence": projection_copy_evidence,
            },
            "candidate_domain": {
                "fingerprint": domain_diagnostics["candidate_domain_fingerprint"],
                "site_count": int(domain_diagnostics["site_count"]),
                "candidate_count": int(domain_diagnostics["candidate_count"]),
                "empty_site_count": int(domain_diagnostics["empty_site_count"]),
                "nonempty_site_count": int(len(feasible_site_ids)),
                "rejection_counts": domain_diagnostics["rejection_counts"],
            },
            "static_capacity": {
                "certified_K": int(static_capacity["objective"]),
                "certificate": static_capacity,
            },
            "contracts": {
                "collision_thresholds_A": _STAGE3_FIXED_COLLISION_THRESHOLDS_A,
                "substrate_collision_mode": "strict-vdw",
                "substrate_vdw_radius_scale": float(args.substrate_vdw_radius_scale),
                "beam_width": 32,
                "collision_cache": "off",
            },
        }
        pool_manifest_reference = write_strict_json_noclobber(
            pool_root / "pool-manifest.json", pool_manifest
        )
        return {
            "pool_index": int(pool_index),
            "pool_root": str(pool_root.relative_to(output_dir)),
            "pool_manifest": {
                "path": str((pool_root / "pool-manifest.json").relative_to(output_dir)),
                **pool_manifest_reference,
            },
            "member_keys": list(member_keys),
            "members": copied_members,
            "domains": domains,
            "domain_diagnostics": domain_diagnostics,
            "projection_cache": projection_cache,
            "static_capacity": static_capacity,
            "nonempty_sites": len(feasible_site_ids),
            "static_K": int(static_capacity["objective"]),
        }

    selected_member_by_key = {
        shape_by_cluster[int(record["cluster_id"])] ["source_key"]: {
            **record,
            "source_key": shape_by_cluster[int(record["cluster_id"])] ["source_key"],
        }
        for record in selected_records
    }
    shape_member_by_key = {record["source_key"]: record for record in shape_records}
    # Pool 0 uses the exact sealed Stage2 top-10 source files, not the cluster
    # representative copies used for the 56-shape diagnostic ensemble.
    pool0_members = [selected_member_by_key[key] for key in selected_keys]
    pool0 = evaluate_pool(0, selected_keys, pool0_members)
    # Projection coverage masks describe footprint overlap, not domain
    # nonemptiness.  Recover the sealed top-10 domain's occupied Site Instance
    # IDs from each candidate anchor coordinate, which is exact for the
    # registered fractional anchor table.
    lattice = np.asarray(pool0["projection_cache"].surface_lattice_uv_A, dtype=float)
    site_anchor_fractional = np.asarray(
        pool0["projection_cache"].site_anchor_fractional, dtype=float
    )
    occupied_indices = set()
    for anchor_uv in np.asarray(pool0["projection_cache"].anchor_uv, dtype=float):
        fractional = np.linalg.solve(lattice, anchor_uv) % 1.0
        delta = np.abs(site_anchor_fractional - fractional)
        delta = np.minimum(delta, 1.0 - delta)
        distances = np.linalg.norm(delta, axis=1)
        nearest = int(np.argmin(distances))
        if float(distances[nearest]) > 1.0e-8:
            raise RuntimeError("Stage3 projection anchor does not match a registered Site Instance")
        occupied_indices.add(nearest)
    if len(occupied_indices) != baseline_nonempty:
        raise RuntimeError(
            "Stage3 projection anchor recovery disagrees with sealed Stage1 nonempty-site count"
        )
    empty_site_ids = {
        site_id
        for site_id, index in pool0["projection_cache"].site_index.items()
        if int(index) not in occupied_indices
    }
    empty_instances = [
        instance
        for instance in site_payload["site_instances"]
        if str(instance["site_instance_id"]) in empty_site_ids
    ]
    empty_site_payload = {
        **site_payload,
        "site_instances": empty_instances,
        "conflict_edges": [
            edge
            for edge in site_payload.get("conflict_edges", [])
            if str(edge[0]) in empty_site_ids and str(edge[1]) in empty_site_ids
        ],
    }
    candidate_index_by_id = {
        str(candidate_id): index
        for index, candidate_id in enumerate(pool0["projection_cache"].candidate_ids)
    }
    if any(candidate_id not in candidate_index_by_id for candidate_id in parent_step_candidate_ids):
        raise RuntimeError("Sealed Stage1 steps are not represented by the Stage2 projection cache")
    parent_placed = [
        {
            "candidate_id": candidate_id,
            "candidate_index": candidate_index_by_id[candidate_id],
        }
        for candidate_id in parent_step_candidate_ids
    ]
    reconstruction = {
        "passed": True,
        "method": "sealed_step_identity_to_sealed_projection_cache_candidate_index",
        "accepted_step_count": len(parent_placed),
        "candidate_ids": parent_step_candidate_ids,
        "source_parent": stage1_reference["parent_run"],
    }
    parent_objective = repair_objective(parent_placed, pool0["projection_cache"])
    stage2_objective = stage2_reference["final_objective"]
    parent_objective_comparison = {
        field: {
            "stage3_reconstructed": float(parent_objective[field]),
            "stage2_v5_declared": float(stage2_objective[field]),
            "absolute_difference": abs(float(parent_objective[field]) - float(stage2_objective[field])),
            "tolerance": 1.0e-6,
            "passed": abs(float(parent_objective[field]) - float(stage2_objective[field])) <= 1.0e-6,
        }
        for field in ("molecule_count_N", "total_hole_area_A2", "maximum_hole_area_A2", "total_hole_perimeter_A")
    }
    if not all(record["passed"] for record in parent_objective_comparison.values()):
        raise RuntimeError("Stage3 reconstructed top-10 parent does not match Stage2 v5 final objective")
    hole_components = parent_objective["torus_component_geometries"]
    if not hole_components:
        raise RuntimeError("Stage3 parent has no largest torus hole")
    largest_hole = hole_components[0]
    lattice = np.asarray(pool0["projection_cache"].surface_lattice_uv_A, dtype=float)
    hole_centroid_fractional = [
        float(value) for value in largest_hole["geometry"].centroid.coords[0]
    ]
    hole_area = float(largest_hole["area_A2"])
    hole_scale = max(float(np.sqrt(hole_area)), 1.0e-12)
    hole_match_scores = {}
    hole_match_records = {}
    for record in shape_records:
        metrics = record["shape_metrics"]
        shape_area = float(metrics["area_A2"])
        area_ratio = shape_area / hole_area
        linear_scale = float(np.sqrt(hole_area / shape_area))
        periodic_distance = _stage3_periodic_distance_A(
            metrics["second_moment_centroid_uv_A"],
            hole_centroid_fractional,
            lattice,
        )
        score = abs(float(np.log(area_ratio))) + 0.25 * periodic_distance / hole_scale
        hole_match_scores[record["source_key"]] = score
        hole_match_records[record["source_key"]] = {
            "largest_parent_hole_area_A2": hole_area,
            "shape_area_A2": shape_area,
            "area_ratio_shape_to_parent_hole": area_ratio,
            "linear_scale_to_parent_hole": linear_scale,
            "periodic_centroid_distance_A": periodic_distance,
            "match_score": score,
            "metric_basis": "absolute_log_area_ratio_plus_scaled_periodic_centroid_distance",
        }

    pool_attempts = [pool0]
    selected_pool_keys = list(selected_keys)
    first_improving_pool = None
    expansion_decisions = []
    while len(selected_pool_keys) < int(args.stage3_max_pool_members):
        decision = stage3_choose_shape_diverse_next(
            shape_records,
            selected_pool_keys,
            clustering["assignments"],
            feature_payload["standardized_features"],
            hole_match_scores,
        )
        next_key = decision["selected_source_key"]
        if next_key in selected_pool_keys:
            raise RuntimeError("Stage3 shape expansion selected a duplicate source")
        next_members = list(selected_pool_keys) + [next_key]
        member_records = [
            selected_member_by_key[key] if key in selected_member_by_key else shape_member_by_key[key]
            for key in next_members
        ]
        pool = evaluate_pool(
            len(pool_attempts),
            next_members,
            member_records,
            base_pool=pool_attempts[-1],
        )
        pool["expansion_decision"] = decision
        pool_attempts.append(pool)
        expansion_decisions.append({
            "pool_index": pool["pool_index"],
            "selected_source_key": next_key,
            "ranking_head": decision["ranking"][:10],
            "nonempty_sites": pool["nonempty_sites"],
            "static_K": pool["static_K"],
            "first_improving_candidate": (
                pool["nonempty_sites"] > baseline_nonempty
                or pool["static_K"] > baseline_static_k
            ),
        })
        selected_pool_keys.append(next_key)
        if (
            pool["nonempty_sites"] > baseline_nonempty
            or pool["static_K"] > baseline_static_k
        ):
            first_improving_pool = pool
            break

    conditional_execution = {
        "ran": first_improving_pool is not None,
        "condition": "first shape-expanded pool must improve nonempty_sites > 82 or static_K > 56",
        "baseline_nonempty_sites": baseline_nonempty,
        "baseline_static_K": baseline_static_k,
        "first_improving_pool_index": (
            None if first_improving_pool is None else first_improving_pool["pool_index"]
        ),
        "grow": None,
        "fixed_site_repair": None,
    }
    if first_improving_pool is not None:
        conditional_dir = output_dir / "conditional-growth-and-repair"
        conditional_dir.mkdir(exist_ok=False)
        growth_result = run_sequential_domains(
            domains=first_improving_pool["domains"],
            site_conflicts=conflicts,
            conformer_weights={int(rank): 1.0 for rank in range(1, len(first_improving_pool["members"]) + 1)},
            cell=substrate.cell,
            periodic_axes=periodic_axes,
            seed=340501,
            selection_policy="capacity-aware",
            projection_cache=first_improving_pool["projection_cache"],
            area_tie_tolerance_A2=1.0e-6,
            perimeter_tie_tolerance_A=1.0e-6,
            collision_profiler=None,
            metal_constraints=metal_constraints,
            collision_cache=None,
            collision_thresholds_A=_STAGE3_FIXED_COLLISION_THRESHOLDS_A,
            selection_criterion="uncovered-anchors",
            capacity_beam_width=32,
            capacity_audit_sink=lambda step, audit: write_capacity_ranking_audit(
                conditional_dir, step=step, audit_document=audit
            ),
            candidate_domain_fingerprint=first_improving_pool["domain_diagnostics"]["candidate_domain_fingerprint"],
        )
        growth_final_path = conditional_dir / "grown-h0.extxyz"
        grown_assembled = _align_to_cell_bottom(
            _assemble(substrate, growth_result["placed"]),
            surface_frame=_common_surface_frame(first_improving_pool["domains"]),
        )
        write(growth_final_path, grown_assembled)
        growth_validation = independently_validate_final_h0(
            trajectory_dir=conditional_dir,
            final_path=growth_final_path,
            substrate=substrate,
            placed=growth_result["placed"],
            molecule_formula=spec.formula,
            periodic_axes=periodic_axes,
            validation_mode="h0",
            collision_thresholds_A=_STAGE3_FIXED_COLLISION_THRESHOLDS_A,
            substrate_vdw_radius_scale=float(args.substrate_vdw_radius_scale),
            mapped_bond_window_A=(args.mapped_bond_min_A, args.mapped_bond_max_A),
        )
        fixed_repair = run_fixed_site_replacements(
            placed=list(growth_result["placed"]),
            domains=first_improving_pool["domains"],
            projection_cache=first_improving_pool["projection_cache"],
            cell=substrate.cell,
            periodic_axes=periodic_axes,
            collision_thresholds_A=_STAGE3_FIXED_COLLISION_THRESHOLDS_A,
            max_replacements=int(args.repair_max_fixed_replacements),
            area_tolerance_A2=1.0e-6,
            metal_constraints=metal_constraints,
        )
        repaired_final_path = conditional_dir / "repaired-h0.extxyz"
        repaired_assembled = _align_to_cell_bottom(
            _assemble(substrate, fixed_repair["placed"]),
            surface_frame=_common_surface_frame(first_improving_pool["domains"]),
        )
        write(repaired_final_path, repaired_assembled)
        conditional_execution["grow"] = {
            "seed": 340501,
            "selection_policy": "capacity-aware",
            "beam_width": 32,
            "collision_cache": "off",
            "result": _json_safe_trajectory(growth_result),
            "final_structure": _relative_artifact_reference(growth_final_path, base=output_dir),
            "independent_validation": growth_validation,
        }
        conditional_execution["fixed_site_repair"] = {
            "result": _serializable_fixed_repair_result(fixed_repair),
            "final_structure": _relative_artifact_reference(repaired_final_path, base=output_dir),
        }
    else:
        conditional_execution["negative_evidence"] = {
            "all_evaluated_pools_failed_improvement": True,
            "evaluated_pool_count": len(pool_attempts),
            "last_pool_nonempty_sites": pool_attempts[-1]["nonempty_sites"],
            "last_pool_static_K": pool_attempts[-1]["static_K"],
            "grow_and_repair_not_run": True,
        }

    parent_tree_after = {
        "stage1": _capture_directory_tree_ledger(stage1_path),
        "stage2": _capture_directory_tree_ledger(stage2_path),
    }
    if parent_tree_after["stage1"] != stage1_reference["parent_tree_ledger_before"]:
        raise RuntimeError("Stage3 changed the immutable Stage1 parent tree")
    if parent_tree_after["stage2"] != stage2_reference["parent_tree_ledger"]:
        raise RuntimeError("Stage3 changed the immutable Stage2 parent tree")

    def pool_summary(pool):
        return {
            key: value
            for key, value in pool.items()
            if key not in {"domains", "projection_cache"}
        }

    diagnostic = {
        "schema": "sam-stage3-diagnostic-v1",
        "status": "passed_cpu_stage3_diagnostic",
        "operation": "stage3-diagnostic",
        "implementation": implementation_identity,
        "request": {"path": "run-request.json", **request_reference},
        "inputs": primary_inputs,
        "stage1_parent": {
            "parent_run": stage1_reference["parent_run"],
            "top_manifest": stage1_reference["hash_ledger_before"]["top_manifest"],
            "trajectory_manifest": stage1_reference["hash_ledger_before"]["trajectory_manifest"],
            "final_structure": stage1_reference["hash_ledger_before"]["final_structure"],
            "tree_sha256": stage1_reference["parent_tree_ledger_before"]["tree_sha256"],
        },
        "stage2_parent": {
            "parent_run": stage2_reference["parent_run"],
            "manifest": stage2_reference["manifest_identity"],
            "final_structure": stage2_reference["final_identity"],
            "tree_sha256": stage2_reference["parent_tree_ledger"]["tree_sha256"],
            "R0_original_gates": stage2_reference["R0_original_gates"],
        },
        "corrected_reference_metrics": reference_provenance,
        "explicit_derived_pool_provenance": explicit_pool_provenance,
        "analysis_provenance": analysis_provenance,
        "shape_feature_standardization": feature_payload,
        "shape_clusters": clustering,
        "shape_records": shape_records,
        "top10_gate": top10_gate,
        "parent_reconstruction": {
            "reconstruction": reconstruction,
            "objective_comparison": parent_objective_comparison,
            "objective": _repair_objective_summary(parent_objective),
            "largest_hole": {
                key: value
                for key, value in largest_hole.items()
                if key not in {"fragment_geometries", "geometry"}
            },
            "hole_centroid_fractional": hole_centroid_fractional,
            "surface_lattice_uv_A": lattice.tolist(),
        },
        "parent_largest_hole_matches": hole_match_records,
        "pool_expansion": {
            "contract": stage3_choose_shape_diverse_next(
                shape_records,
                selected_pool_keys[:-1] if len(selected_pool_keys) > 10 else selected_pool_keys,
                clustering["assignments"],
                feature_payload["standardized_features"],
                hole_match_scores,
            )["contract"] if len(selected_pool_keys) < 56 else {
                "priority": ["uncovered_shape_cluster", "farthest_minimum_distance_to_selected", "parent_largest_hole_area_scale_and_periodic_distance_match", "relative_energy_final_tie_break_only", "source_key_stable_tie_break"],
                "energy_used_as_primary": False,
            },
            "expansion_decisions": expansion_decisions,
            "pools": [pool_summary(pool) for pool in pool_attempts],
            "first_improving_pool": (
                None if first_improving_pool is None else pool_summary(first_improving_pool)
            ),
        },
        "conditional_execution": conditional_execution,
        "parent_tree_after": {
            name: {
                "tree_sha256": ledger["tree_sha256"],
                "path_count": ledger["path_count"],
            }
            for name, ledger in parent_tree_after.items()
        },
        "contracts": {
            "thresholds_A": _STAGE3_FIXED_COLLISION_THRESHOLDS_A,
            "strict_vdw_radius_scale": float(args.substrate_vdw_radius_scale),
            "beam_width": 32,
            "collision_cache": "off",
            "forbidden_stages_not_run": ["MACE", "GPU", "LAMMPS", "protonation", "MD"],
        },
    }
    diagnostic_reference = write_strict_json_noclobber(
        output_dir / "stage3-diagnostic.json", diagnostic
    )
    manifest = {
        "schema": "sam-stage3-diagnostic-run-manifest-v1",
        "schema_version": 1,
        "sealed": True,
        "status": diagnostic["status"],
        "operation": "stage3-diagnostic",
        "run_request": {"path": "run-request.json", **request_reference},
        "diagnostic": {"path": "stage3-diagnostic.json", **diagnostic_reference},
        "artifact_inventory_scope": "every_regular_output_file_before_manifest_seal;manifest_self_hash_excluded",
        "artifact_inventory": _artifact_inventory(
            output_dir, excluded_names={"manifest.json", "failure.json"}
        ),
        "forbidden_stages_not_run": ["MACE", "GPU", "LAMMPS", "protonation", "MD"],
        "cpu_only": True,
    }
    write_strict_json_noclobber(output_dir / "manifest.json", manifest)
    return 0


def execute_sequential_growth(
    args,
    *,
    argv: list[str],
    root: Path,
    output_dir: Path,
    resolved_paths: dict[str, Path],
    cache_reuse_contract: dict,
    implementation_identity: dict,
    p_surface_o_exclusion_contract: dict | None = None,
    metal_coordination_contract: dict | None = None,
) -> int:
    """Execute one already-reserved immutable sequential-growth run."""

    substrate_path = resolved_paths["substrate"]
    layer_groups_path = resolved_paths["layer_groups"]
    site_instances_path = resolved_paths["site_instances"]
    site_prototypes_path = resolved_paths["site_prototypes"]
    analysis_path = resolved_paths["conformer_analysis"]
    if p_surface_o_exclusion_contract is not None:
        verify_p_surface_o_exclusion_contract_sources(p_surface_o_exclusion_contract)
    mapped_bond_window_A = _validated_registered_bond_window(
        (args.mapped_bond_min_A, args.mapped_bond_max_A)
    )
    run_started = time.perf_counter()
    collision_thresholds_A = _validate_collision_thresholds(
        {
            "H-H": float(args.hh_min),
            "H-heavy": float(args.h_heavy_min),
            "heavy-heavy": float(args.heavy_heavy_min),
        }
    )
    policy = SamplingPolicy(
        energy_scale_eV=args.energy_scale_eV,
        area_coefficient=args.area_coefficient,
        rank_exponent=args.rank_exponent,
    )
    policy.validate()
    analysis_records = _selected_analysis_records(analysis_path, args.max_conformers)
    sampling_table = conformer_sampling_table(analysis_records, policy)
    metrics_by_rank = {record["selection_rank"]: record for record in sampling_table}
    conformer_weights = {
        record["selection_rank"]: record["unnormalized_weight"]
        for record in sampling_table
    }
    spec = MoleculeSpec(
        formula=_json_formula(args.molecule_formula_json) or {},
        skeleton_formula=_json_formula(args.skeleton_formula_json),
    )

    # Resolve cheap structure/schema contracts and seal the complete request
    # before conformer parsing, cache materialization, or candidate-domain work.
    substrate, layer_groups = _load_working_substrate(substrate_path, layer_groups_path)
    full_to_working_atom_map = build_full_to_working_atom_map(
        working_layer_atom_ids_1based=layer_groups[
            "working_layer_atom_ids_1based"
        ],
        working_symbols=substrate.get_chemical_symbols(),
    )
    site_payload = json.loads(site_instances_path.read_text())
    prototypes = _prototype_map(site_prototypes_path)
    periodic_axes = resolve_periodic_fractional_axes(site_payload, prototypes)
    normal_axis = _validated_layer_normal_axis(layer_groups, periodic_axes)
    if metal_coordination_contract is None:
        metal_coordination_contract = build_metal_coordination_filter_contract(
            mode=args.metal_coordination_filter,
            cutoff_A=args.metal_coordination_cutoff_A,
            max_allowed_coordination=args.metal_coordination_max_allowed,
            periodic_axes=periodic_axes,
            normal_axis=normal_axis,
        )
    selected_conformer_evidence = _selected_conformer_evidence(
        analysis_records, root=root, analysis_path=analysis_path
    )
    selected_source_lock = _lock_selected_conformer_paths(
        root=root,
        conformer_glob=args.conformer_glob,
        maximum=args.max_conformers,
        selected_evidence=selected_conformer_evidence,
    )
    source_verification_checkpoints = {
        "before_run_request": _verify_locked_conformer_sources(
            selected_source_lock, checkpoint="before_run_request"
        )
    }
    primary_inputs = {
        name: _file_identity(resolved_paths[name])
        for name in (
            "substrate",
            "layer_groups",
            "site_instances",
            "site_prototypes",
            "conformer_analysis",
        )
    }
    implementation = implementation_identity
    code_snapshot_plan = _code_snapshot_plan(implementation)
    parent_reference = None
    reference_metrics = None
    parent_contract = None
    operation_evidence = None
    if args.operation == "repair":
        parent_path = args.parent_run.expanduser()
        if not parent_path.is_absolute():
            parent_path = root / parent_path
        reference_path = args.reference_metrics.expanduser()
        if not reference_path.is_absolute():
            reference_path = root / reference_path
        parent_reference = capture_immutable_parent_reference(
            parent_path,
            expected_inputs=primary_inputs,
        )
        reference_metrics = capture_reference_metrics(
            reference_path,
            parent_reference=parent_reference,
            expected_inputs=primary_inputs,
            selected_source_lock=selected_source_lock,
            repair_metric_contract=_repair_cli_metric_contract(args),
            current_substrate=substrate,
        )
        parent_contract = _validate_repair_parent_contract(
            parent_reference,
            args=args,
            substrate=substrate,
            molecule_spec=spec,
            periodic_axes=periodic_axes,
            selected_source_lock=selected_source_lock,
            collision_thresholds_A=collision_thresholds_A,
        )
        operation_evidence = _repair_operation_evidence(
            parent_reference, reference_metrics
        )
    requested_substrate_contract = build_substrate_collision_contract(
        mode=args.substrate_collision,
        periodic_axes=periodic_axes,
        mapped_bond_window_A=mapped_bond_window_A,
        hard_thresholds_A=collision_thresholds_A,
        substrate_symbols=substrate.get_chemical_symbols(),
        molecule_symbols=[
            element for element, count in spec.formula.items() if int(count) > 0
        ],
        vdw_radius_scale=float(args.substrate_vdw_radius_scale),
        height_clearance_A=float(args.substrate_height_clearance_A),
    )
    run_request = _build_run_request_document(
        argv=argv,
        args=args,
        root=root,
        output_dir=output_dir,
        resolved_paths=resolved_paths,
        implementation=implementation,
        primary_inputs=primary_inputs,
        selected_conformers=selected_conformer_evidence,
        selected_source_lock=selected_source_lock,
        sampling_table=sampling_table,
        periodic_axes=periodic_axes,
        normal_axis=normal_axis,
        collision_thresholds_A=collision_thresholds_A,
        mapped_bond_window_A=mapped_bond_window_A,
        substrate_collision_contract=requested_substrate_contract,
        molecule_spec=spec,
        cache_reuse_contract=cache_reuse_contract,
        code_snapshot=code_snapshot_plan,
        operation_evidence=operation_evidence,
        p_surface_o_exclusion_contract=p_surface_o_exclusion_contract,
        metal_coordination_contract=metal_coordination_contract,
    )
    run_request_reference = write_strict_json_noclobber(
        output_dir / "run-request.json", run_request
    )
    code_snapshot = _write_code_snapshot(output_dir, code_snapshot_plan)
    source_verification_checkpoints["after_run_request"] = (
        _verify_locked_conformer_sources(
            selected_source_lock, checkpoint="after_run_request"
        )
    )
    if p_surface_o_exclusion_contract is not None:
        verify_p_surface_o_exclusion_contract_sources(p_surface_o_exclusion_contract)

    conformers = source_conformers(
        root,
        args.conformer_glob,
        resolved_paths["conformer_cache"],
        spec,
        minimum_vertical_angle=0.0,
        maximum_marker_exposure=float("inf"),
        maximum_source_clusters=args.max_conformers,
        immutable_new_cache=bool(cache_reuse_contract["formal"]),
    )
    source_verification_checkpoints["after_source_conformers"] = (
        _verify_locked_conformer_sources(
            selected_source_lock, checkpoint="after_source_conformers"
        )
    )
    loaded_conformer_audit = _audit_loaded_conformer_ensemble(
        conformers,
        locked_selection=selected_source_lock,
        spec=spec,
        root=root,
    )
    source_verification_checkpoints["after_direct_geometry_audit"] = (
        _verify_locked_conformer_sources(
            selected_source_lock, checkpoint="after_direct_geometry_audit"
        )
    )
    print(
        f"Loaded {len(conformers)} ranked H0 conformers; preparing all Site Instances",
        flush=True,
    )

    domain_started = time.perf_counter()
    domains, domain_diagnostics = _candidate_domains(
        substrate=substrate,
        site_payload=site_payload,
        prototypes=prototypes,
        conformers=conformers,
        metrics_by_rank=metrics_by_rank,
        spec=spec,
        headgroup_rmsd_max_A=float(args.headgroup_rmsd_max_A),
        full_to_working_atom_map=full_to_working_atom_map,
        mapped_bond_window_A=mapped_bond_window_A,
        substrate_vdw_radius_scale=float(args.substrate_vdw_radius_scale),
        substrate_height_clearance_A=float(args.substrate_height_clearance_A),
        periodic_axes=periodic_axes,
        collision_thresholds_A=collision_thresholds_A,
        substrate_collision_mode=args.substrate_collision,
        p_surface_o_exclusion_contract=p_surface_o_exclusion_contract,
        metal_coordination_contract=metal_coordination_contract,
        single_metal_sites_mode=getattr(args, "single_metal_sites", "off"),
        single_metal_rotation_step_deg=float(getattr(args, "single_metal_rotation_step_deg", 30.0)),
        single_metal_translation_offsets_A=tuple(float(v) for v in getattr(args, "single_metal_translation_offsets_A", (-1.0, -0.5, 0.0, 0.5, 1.0))),
    )
    has_domain_candidates = any(domains.values())
    assembly_surface_frame = _common_surface_frame(domains) if has_domain_candidates else None
    substrate_collision_contract = domain_diagnostics[
        "substrate_collision_contract"
    ]
    if substrate_collision_contract != requested_substrate_contract:
        raise RuntimeError(
            "Candidate-domain substrate contract differs from the sealed run request"
        )
    domain_fingerprint_contract = (
        None
        if args.substrate_collision == "skip"
        else substrate_collision_contract
    )
    domain_fingerprint = domain_diagnostics.get(
        "candidate_domain_fingerprint"
    ) or _candidate_domain_fingerprint(
        domains,
        cell=substrate.cell,
        periodic_axes=periodic_axes,
        substrate_collision_contract=domain_fingerprint_contract,
        p_surface_o_exclusion_contract=p_surface_o_exclusion_contract,
        metal_coordination_contract=metal_coordination_contract,
    )
    if args.operation == "repair":
        parent_contract = _validate_repair_parent_contract(
            parent_reference,
            args=args,
            substrate=substrate,
            molecule_spec=spec,
            periodic_axes=periodic_axes,
            selected_source_lock=selected_source_lock,
            collision_thresholds_A=collision_thresholds_A,
            domain_diagnostics=domain_diagnostics,
        )
    domain_seconds = time.perf_counter() - domain_started
    print(
        "Prepared "
        f"{domain_diagnostics['candidate_count']} candidates on "
        f"{domain_diagnostics['site_count'] - domain_diagnostics['empty_site_count']}/"
        f"{domain_diagnostics['site_count']} nonempty sites in {domain_seconds:.2f} s",
        flush=True,
    )
    conflicts = _conflict_map(site_payload)
    metal_constraints = {
        str(instance["site_instance_id"]): {
            int(atom) for atom in instance.get("actual_metal_atom_ids_1based", [])
        }
        for instance in site_payload["site_instances"]
    }
    p_surface_o_audit_reference = None
    p_surface_o_static_capacity = None
    if p_surface_o_exclusion_contract is not None:
        p_surface_o_summary = domain_diagnostics["p_surface_o_exclusion"]
        complete_excluded = p_surface_o_summary.pop(
            "complete_excluded_placement_audit", []
        )
        p_surface_o_static_capacity = solve_static_conflict_capacity(
            [site_id for site_id, candidates in domains.items() if candidates],
            metal_constraints,
            site_conflicts=conflicts,
            time_limit_seconds=30.0,
        )
        p_surface_o_audit_document = {
            "schema": "sam-sequential-p-surface-o-exclusion-audit-v1",
            "sealed": True,
            "experiment_contract": p_surface_o_exclusion_contract,
            "domain": p_surface_o_summary,
            "static_capacity_actual_metal_one": p_surface_o_static_capacity,
            "excluded_placements": complete_excluded,
            "excluded_placement_record_count": len(complete_excluded),
            "complete_pair_audit": True,
            "other_sam_substrate_pairs_checked": False,
            "other_sam_substrate_pairs_rejected": False,
        }
        p_surface_o_audit_path = output_dir / "p-surface-o-exclusion-audit.json"
        p_surface_o_audit_written = write_strict_json_noclobber(
            p_surface_o_audit_path, p_surface_o_audit_document
        )
        p_surface_o_audit_reference = {
            "path": p_surface_o_audit_path.relative_to(output_dir).as_posix(),
            **p_surface_o_audit_written,
        }

    metal_coord_audit_reference = None
    if metal_coordination_contract.get("mode") == "exclude-six-coordinated":
        metal_coord_summary = domain_diagnostics["metal_coordination_filter"]
        metal_coord_audit_document = {
            "schema": "samflow-metal-coordination-filter-audit-v1",
            "sealed": True,
            "experiment_contract": metal_coordination_contract,
            "domain": metal_coord_summary,
            "hypothesis_testing_note": metal_coordination_contract.get("hypothesis_testing_note", ""),
        }
        metal_coord_audit_path = output_dir / "metal-coordination-filter-audit.json"
        metal_coord_audit_written = write_strict_json_noclobber(
            metal_coord_audit_path, metal_coord_audit_document
        )
        metal_coord_audit_reference = {
            "path": metal_coord_audit_path.relative_to(output_dir).as_posix(),
            **metal_coord_audit_written,
        }

    if not has_domain_candidates:
        raise RuntimeError("No candidate remains after candidate filtering")

    projection_cache = None
    projection_cache_manifest = None
    if (
        args.operation == "repair"
        or args.selection_policy in {
            "projection-aware", "capacity-aware", "cohesive-frontier"
        }
    ):
        projection_cache = _load_or_build_projection_cache(
            domains,
            site_payload=site_payload,
            cell=substrate.cell,
            periodic_axes=periodic_axes,
            cache_dir=resolved_paths["projection_cache"],
            boundary_samples_per_atom=args.footprint_boundary_samples,
            radius_scale=float(args.projection_radius_scale),
            prototypes=prototypes,
            substrate_collision_contract=domain_fingerprint_contract,
            p_surface_o_exclusion_contract=p_surface_o_exclusion_contract,
            metal_coordination_contract=metal_coordination_contract,
            immutable_new_cache=bool(cache_reuse_contract["formal"]),
        )
        projection_cache_manifest = projection_cache.manifest
        print(
            f"Projection cache {'hit' if projection_cache.manifest['hit'] else 'built'}: "
            f"{len(projection_cache.template_records)} templates, "
            f"{len(projection_cache.candidate_ids)} candidates",
            flush=True,
        )

    collision_profiler = CollisionQueryProfiler()

    collision_cache = None
    if args.collision_cache == "lazy":
        by_index = {}
        for candidates in domains.values():
            for candidate in candidates:
                by_index[candidate["candidate_index"]] = candidate

        def collision_direct_query(left, right):
            count, _, _ = surface_collision_report(
                by_index[left]["coordinates"],
                by_index[left]["symbols"],
                by_index[right]["coordinates"],
                by_index[right]["symbols"],
                substrate.cell,
                periodic_axes,
                thresholds_A=collision_thresholds_A,
            )
            return count > 0

        collision_cache = LazyCollisionCache(
            candidate_domain_fingerprint=domain_fingerprint,
            collision_contract_fingerprint=_collision_contract_fingerprint(
                periodic_axes=periodic_axes,
                thresholds_A=collision_thresholds_A,
                algorithm_version=_COLLISION_ALGORITHM_VERSION,
            ),
            candidate_count=sum(len(candidates) for candidates in domains.values()),
            direct_query=collision_direct_query,
        )
        print(
            f"Lazy collision cache active for {collision_cache.candidate_count} "
            "candidates",
            flush=True,
        )

    if args.operation == "repair":
        conformer_pool_fingerprint = _payload_sha256(
            {
                "schema": "selected-conformer-pool-v1",
                "molecule_formula": spec.formula,
                "selected": selected_conformer_evidence,
            }
        )
        return execute_local_repair_package(
            args=args,
            output_dir=output_dir,
            substrate=substrate,
            domains=domains,
            domain_diagnostics=domain_diagnostics,
            projection_cache=projection_cache,
            conflicts=conflicts,
            metal_constraints=metal_constraints,
            collision_thresholds_A=collision_thresholds_A,
            periodic_axes=periodic_axes,
            normal_axis=normal_axis,
            assembly_surface_frame=assembly_surface_frame,
            mapped_bond_window_A=mapped_bond_window_A,
            molecule_spec=spec,
            parent_reference=parent_reference,
            reference_metrics=reference_metrics,
            parent_contract=parent_contract,
            run_request=run_request,
            run_request_reference=run_request_reference,
            implementation=implementation,
            code_snapshot=code_snapshot,
            primary_inputs=primary_inputs,
            selected_source_lock=selected_source_lock,
            source_verification_checkpoints=source_verification_checkpoints,
            loaded_conformer_audit=loaded_conformer_audit,
            cache_reuse_contract=cache_reuse_contract,
            conformer_cache_path=resolved_paths["conformer_cache"],
            projection_cache_path=resolved_paths["projection_cache"],
            domain_fingerprint=domain_fingerprint,
            conformer_pool_fingerprint=conformer_pool_fingerprint,
            projection_cache_manifest=projection_cache_manifest,
            run_started=run_started,
        )

    trajectory_summaries = []
    trajectory_results = []
    all_required_validation_passed = True
    cohesive_xy_bounds_by_candidate_id = (
        {
            candidate["candidate_id"]: cohesive_xy_bound(candidate)
            for candidates in domains.values()
            for candidate in candidates
        }
        if args.selection_policy == "cohesive-frontier"
        else None
    )
    for trajectory_index in range(1, args.trajectory_count + 1):
        trajectory_seed = int(args.seed) + trajectory_index - 1
        trajectory_started = time.perf_counter()
        trajectory_dir = output_dir / f"trajectory-{trajectory_index:04d}"
        trajectory_dir.mkdir(exist_ok=False)
        progress_path = trajectory_dir / "progress.json"

        def live_progress_sink(summary: dict) -> None:
            payload = dict(summary)
            payload["trajectory_index"] = trajectory_index
            write_live_progress_atomic(progress_path, payload)
            current_N = payload.get("current_N")
            denticity_phase = payload.get("denticity_phase")
            elapsed_seconds = float(payload.get("elapsed_seconds", 0.0))
            print(
                f"[sequential-progress] trajectory={trajectory_index:04d} N={current_N} phase={denticity_phase} elapsed_s={elapsed_seconds:.1f}",
                flush=True,
            )

        if collision_profiler is not None:
            collision_profiler.begin_trajectory()
        capacity_audit_sink = None
        if args.selection_policy == "capacity-aware":
            def capacity_audit_sink(step, audit_document, *, _directory=trajectory_dir):
                return write_capacity_ranking_audit(
                    _directory,
                    step=step,
                    audit_document=audit_document,
                )
        single_metal_context = _build_single_metal_context(
            single_metal_sites_mode=getattr(args, "single_metal_sites", "off"),
            substrate_coordinates=substrate.positions,
            substrate_symbols=substrate.symbols,
            substrate_cell=substrate.cell,
            periodic_axes=periodic_axes,
            normal_axis=normal_axis,
            retained_site_instances=[
                inst for inst in site_payload["site_instances"]
                if str(inst.get("site_instance_id")) in domain_diagnostics.get("metal_coordination_filter", {}).get("site_instance_summary", {}).get("retained_site_instance_ids", set(domains.keys()))
            ],
            full_to_working_atom_map=full_to_working_atom_map,
            surface_frame=assembly_surface_frame,
            conformers=conformers,
            metrics_by_rank=metrics_by_rank,
            substrate_height_clearance_A=float(args.substrate_height_clearance_A),
            rotation_step_deg=float(getattr(args, "single_metal_rotation_step_deg", 30.0)),
            translation_offsets_A=tuple(float(v) for v in getattr(args, "single_metal_translation_offsets_A", (-1.0, -0.5, 0.0, 0.5, 1.0))),
            metal_coordination_contract=metal_coordination_contract,
        )
        result = run_sequential_domains(
            domains=domains,
            site_conflicts=conflicts,
            conformer_weights=conformer_weights,
            cell=substrate.cell,
            periodic_axes=periodic_axes,
            seed=trajectory_seed,
            selection_policy=args.selection_policy,
            projection_cache=projection_cache,
            area_tie_tolerance_A2=args.area_tie_tolerance_A2,
            perimeter_tie_tolerance_A=args.perimeter_tie_tolerance_A,
            collision_profiler=collision_profiler,
            metal_constraints=metal_constraints,
            collision_cache=collision_cache,
            collision_thresholds_A=collision_thresholds_A,
            selection_criterion=args.selection_criterion,
            capacity_beam_width=args.capacity_beam_width,
            capacity_audit_sink=capacity_audit_sink,
            candidate_domain_fingerprint=domain_fingerprint,
            cohesive_relative_tolerance=args.cohesive_relative_tolerance,
            cohesive_perimeter_relative_tolerance=resolve_cohesive_cli_tolerances(
                args
            )["perimeter"],
            cohesive_contact_relative_tolerance=resolve_cohesive_cli_tolerances(
                args
            )["contact"],
            cohesive_area_relative_tolerance=resolve_cohesive_cli_tolerances(
                args
            )["area"],
            cohesive_energy_tolerance_eV=args.cohesive_energy_tolerance_eV,
            cohesive_contact_vdw_radius_scale=(
                args.cohesive_contact_vdw_radius_scale
            ),
            cohesive_xy_bounds_by_candidate_id=(
                cohesive_xy_bounds_by_candidate_id
            ),
            cohesive_frontier_mode=args.cohesive_frontier_mode,
            frontier_snap_radius_A=args.frontier_snap_radius_A,
            progress_sink=live_progress_sink,
            single_metal_context=single_metal_context,
        )
        final_n = int(result["emergent_molecule_count"])
        write_live_progress_atomic(
            progress_path,
            {
                "schema": "sam-sequential-live-progress-v1",
                "status": "growth_completed",
                "seed": trajectory_seed,
                "trajectory_index": trajectory_index,
                "step": final_n,
                "current_N": final_n,
                "final_N": final_n,
                "emergent_molecule_count": final_n,
                "growth_status": result.get("status"),
                "elapsed_seconds": float(time.perf_counter() - trajectory_started),
            },
        )
        if args.write_every_step:
            step_dir = trajectory_dir / "steps"
            step_dir.mkdir(exist_ok=False)
            for step in range(1, len(result["placed"]) + 1):
                write(
                    step_dir / f"step-{step:04d}.extxyz",
                    _align_to_cell_bottom(
                        _assemble(substrate, result["placed"][:step]),
                        surface_frame=assembly_surface_frame,
                    ),
                )
        final_path = trajectory_dir / "final-h0.extxyz"
        final_assembled = _align_to_cell_bottom(
            _assemble(substrate, result["placed"]),
            surface_frame=assembly_surface_frame,
        )
        write(final_path, final_assembled)

        if args.independent_validation == "off":
            independent_validation = {
                "mode": "off",
                "status": "unvalidated",
                "passed": None,
                "required": False,
                "reason": "independent_validation_explicitly_disabled",
                "artifacts": {},
            }
            if args.substrate_collision in {"strict-vdw", "height"}:
                final_readback = read(final_path)
                if args.substrate_collision == "strict-vdw":
                    final_interface_audit = audit_final_registered_interfaces(
                        assembled=final_readback,
                        substrate_atom_count=len(substrate),
                        placed=result["placed"],
                        periodic_axes=periodic_axes,
                        radius_scale=float(args.substrate_vdw_radius_scale),
                    )
                else:
                    final_interface_audit = audit_final_height_clearance(
                        assembled=final_readback,
                        substrate_atom_count=len(substrate),
                        placed=result["placed"],
                        height_clearance_A=float(
                            args.substrate_height_clearance_A
                        ),
                        periodic_axes=periodic_axes,
                        collision_thresholds_A=collision_thresholds_A,
                        full_to_working_atom_map=full_to_working_atom_map,
                    )
                if not final_interface_audit["passed"]:
                    raise RuntimeError(
                        "Final assembled substrate-interface audit failed"
                    )
                final_interface_audit = {"performed": True, **final_interface_audit}
            else:
                final_interface_audit = {
                    "performed": False,
                    "geometry_checked": False,
                    "reason": "independent_validation_off_and_substrate_mode_has_no_final_hard_gate",
                    "periodic_axes": list(periodic_axes),
                    "normal_axis_wrapped": False,
                }
            trajectory_status = result["status"]
        else:
            independent_validation = independently_validate_final_h0(
                trajectory_dir=trajectory_dir,
                final_path=final_path,
                substrate=substrate,
                placed=result["placed"],
                molecule_formula=spec.formula,
                periodic_axes=periodic_axes,
                validation_mode=args.independent_validation,
                collision_thresholds_A=collision_thresholds_A,
                substrate_vdw_radius_scale=float(args.substrate_vdw_radius_scale),
                mapped_bond_window_A=mapped_bond_window_A,
                p_surface_o_exclusion_contract=p_surface_o_exclusion_contract,
                full_to_working_atom_map=full_to_working_atom_map,
            )
            final_interface_audit = {
                "performed": True,
                **independent_validation["final_interface_audit"],
            }
            trajectory_required_passed = bool(independent_validation["passed"])
            all_required_validation_passed &= trajectory_required_passed
            trajectory_status = (
                "passed_cpu_h0_algorithmic_count_only"
                if args.independent_validation == "algorithmic-count-only" and trajectory_required_passed
                else (
                    "failed_cpu_h0_algorithmic_count_only_validation"
                    if args.independent_validation == "algorithmic-count-only"
                    else (
                        "passed_cpu_h0_sequential_growth_and_independent_validation"
                        if trajectory_required_passed
                        else "failed_independent_h0_validation"
                    )
                )
            )

        trajectory_manifest = _json_safe_trajectory(result)
        if "accepted_union_wkb_hex" in trajectory_manifest:
            wkb_path = trajectory_dir / "accepted-union.wkb"
            wkb_bytes = bytes.fromhex(
                trajectory_manifest.pop("accepted_union_wkb_hex")
            )
            _atomic_install_noclobber_bytes(wkb_path, wkb_bytes)
            trajectory_manifest["accepted_union_wkb"] = _relative_artifact_reference(
                wkb_path, base=trajectory_dir
            )
        trajectory_manifest.update(
            {
                "schema_version": 4,
                "path_base": "trajectory_root",
                "status": trajectory_status,
                "sealed": True,
                "growth_status": result["status"],
                "selection_policy": args.selection_policy,
                "periodic_fractional_axes": list(periodic_axes),
                "normal_axis": normal_axis,
                "molecule_count_N": int(result["emergent_molecule_count"]),
                "final_structure": _relative_artifact_reference(
                    final_path, base=trajectory_dir
                ),
                "independent_validation": independent_validation,
                "final_registered_interface_audit": final_interface_audit,
                "torus_v2_summary": {
                    "applicable": "periodic_hole_diagnostics" in result,
                    "metric_version": "periodic-torus-holes-v2",
                    "diagnostics": result.get("periodic_hole_diagnostics"),
                },
                "capacity_K_debt_summary": _capacity_K_debt_summary(
                    result, selection_policy=args.selection_policy
                ),
                "p_surface_o_exclusion": (
                    None
                    if p_surface_o_exclusion_contract is None
                    else {
                        "shared_domain": True,
                        "contract_upper_bound_A": float(p_surface_o_exclusion_contract["contact_upper_bound_A"]),
                        "candidate_count_before_exclusion": int(domain_diagnostics["p_surface_o_exclusion"]["candidate_count_before_exclusion"]),
                        "excluded_placement_count": int(domain_diagnostics["p_surface_o_exclusion"]["excluded_placement_count"]),
                        "retained_candidate_count": int(domain_diagnostics["p_surface_o_exclusion"]["retained_candidate_count"]),
                        "static_capacity": p_surface_o_static_capacity,
                        "sensitivity": domain_diagnostics["p_surface_o_exclusion"]["sensitivity"],
                    }
                ),
                "algorithmic_validation_summary": (
                    None
                    if args.independent_validation != "algorithmic-count-only"
                    else {
                        "sam_sam_audit": independent_validation["sam_sam_audit"],
                        "p_surface_o_forbidden_audit": independent_validation["p_surface_o_forbidden_audit"],
                        "other_substrate_strict_vdw_audit": independent_validation["other_substrate_strict_vdw_audit"],
                        "physical_interface_pass": False,
                    }
                ),
            }
        )
        if args.selection_policy == "capacity-aware":
            trajectory_manifest["ranking_artifact"] = (
                capacity_ranking_artifact_contract()
            )
            trajectory_manifest["optimization_scope"] = (
                "exact_one_step_within_deterministic_beam"
            )
            trajectory_manifest["global_candidate_optimality"] = False
            trajectory_manifest["global_packing_certificate"] = False
        trajectory_manifest["artifact_inventory_before_manifest_seal"] = (
            _artifact_inventory(trajectory_dir, excluded_names={"manifest.json"})
        )
        trajectory_manifest_path = trajectory_dir / "manifest.json"
        trajectory_manifest_reference = write_strict_json_noclobber(
            trajectory_manifest_path, trajectory_manifest
        )
        trajectory_seconds = time.perf_counter() - trajectory_started
        trajectory_summaries.append(
            {
                "trajectory": trajectory_index,
                "path_base": "run_root",
                "seed": trajectory_seed,
                "status": trajectory_status,
                "growth_status": result["status"],
                "emergent_molecule_count": result["emergent_molecule_count"],
                "initial_candidate_count": int(domain_diagnostics["candidate_count"]),
                "p_surface_o_excluded_placement_count": (
                    None
                    if p_surface_o_exclusion_contract is None
                    else int(domain_diagnostics["p_surface_o_exclusion"]["excluded_placement_count"])
                ),
                "retained_site_count": int(domain_diagnostics["site_count"] - domain_diagnostics["empty_site_count"]),
                "static_capacity": p_surface_o_static_capacity,
                "stop_reason": result["status"],
                "wall_time_seconds": trajectory_seconds,
                "manifest": {
                    "path": trajectory_manifest_path.relative_to(output_dir).as_posix(),
                    "sha256": trajectory_manifest_reference["sha256"],
                    "bytes": trajectory_manifest_reference["bytes"],
                },
                "final_structure": _relative_artifact_reference(
                    final_path, base=output_dir
                ),
                "independent_validation": {
                    "mode": independent_validation["mode"],
                    "status": independent_validation.get("status", trajectory_status),
                    "passed": independent_validation["passed"],
                    "sam_sam_audit": independent_validation.get("sam_sam_audit"),
                    "p_surface_o_forbidden_audit": independent_validation.get("p_surface_o_forbidden_audit"),
                    "other_substrate_strict_vdw_audit": independent_validation.get("other_substrate_strict_vdw_audit"),
                    "physical_interface_pass": independent_validation.get("physical_interface_pass"),
                    "artifacts": _rebase_artifact_references(
                        independent_validation["artifacts"],
                        source_base=trajectory_dir,
                        destination_base=output_dir,
                    ),
                },
                "final_registered_interface_audit_performed": bool(
                    final_interface_audit["performed"]
                ),
                "final_registered_interface_audit_passed": (
                    bool(final_interface_audit["passed"])
                    if final_interface_audit["performed"]
                    else None
                ),
            }
        )
        trajectory_results.append(result)
        print(
            f"Trajectory {trajectory_index}/{args.trajectory_count} seed={trajectory_seed} "
            f"jammed_at={result['emergent_molecule_count']} status={trajectory_status} "
            f"in {trajectory_seconds:.2f} s",
            flush=True,
        )

    if args.independent_validation == "off":
        run_status = "passed_cpu_sequential_growth"
    elif args.independent_validation == "algorithmic-count-only":
        run_status = (
            "passed_cpu_h0_algorithmic_count_only"
            if all_required_validation_passed
            else "failed_cpu_h0_algorithmic_count_only_validation"
        )
    else:
        run_status = (
            "passed_cpu_h0_sequential_growth_and_independent_validation"
            if all_required_validation_passed
            else "failed_independent_h0_validation"
        )
    conformer_pool_fingerprint = _payload_sha256(
        {
            "schema": "selected-conformer-pool-v1",
            "molecule_formula": spec.formula,
            "selected": selected_conformer_evidence,
        }
    )
    conformer_cache_fingerprint = _cache_prefix_fingerprint(
        resolved_paths["conformer_cache"]
    )
    projection_fingerprint = _payload_sha256(
        projection_cache_manifest["contract"]
        if projection_cache_manifest is not None
        else {
            "selection_policy": args.selection_policy,
            "projection_cache": "not_applicable",
        }
    )
    collision_cache_contract_fingerprint = _collision_contract_fingerprint(
        periodic_axes=periodic_axes,
        thresholds_A=collision_thresholds_A,
        algorithm_version=_COLLISION_ALGORITHM_VERSION,
        p_surface_o_exclusion_contract=p_surface_o_exclusion_contract,
    )
    projection_cache_required = args.selection_policy in {
        "projection-aware",
        "capacity-aware",
        "cohesive-frontier",
    }
    if (
        cache_reuse_contract["formal"]
        and projection_cache_required
        and (
            projection_cache_manifest is None
            or projection_cache_manifest.get("hit") is not False
        )
    ):
        raise RuntimeError(
            "Formal projection cache must be newly built without a cache hit"
        )
    cache_artifacts = _cache_artifact_records(
        conformer_prefix=resolved_paths["conformer_cache"],
        projection_directory=resolved_paths["projection_cache"],
        output_dir=output_dir,
        formal=bool(cache_reuse_contract["formal"]),
        projection_required=projection_cache_required,
    )
    if args.selection_policy == "capacity-aware":
        placement_policy_manifest = {
            "candidate_selection": "capacity_aware_complete_candidate_lexicographic_beam",
            "approved_conformer_prior_role": "final_tie_break_only",
            "donor_mapping_draw": "seeded_uniform_only_within_final_selected_rank_tie",
            "headgroup_rmsd_max_A": float(args.headgroup_rmsd_max_A),
            "collision_thresholds_A": collision_thresholds_A,
            "collision_threshold_scope": "SAM-SAM_only",
            "periodic_axes": list(periodic_axes),
            "normal_axis_wrapped": False,
            "accepted_molecules_moved": False,
            "termination": "no_feasible_site_conformer_mapping_candidate",
        }
        capacity_policy_manifest = _capacity_policy_manifest(
            beam_width=args.capacity_beam_width,
            area_tie_tolerance_A2=args.area_tie_tolerance_A2,
            perimeter_tie_tolerance_A=args.perimeter_tie_tolerance_A,
            collision_thresholds_A=collision_thresholds_A,
        )
    elif args.selection_policy == "cohesive-frontier":
        cohesive_manifest_tolerances = resolve_cohesive_cli_tolerances(args)
        cohesive_umbrella_audit = resolve_cohesive_umbrella_audit(args)
        placement_policy_manifest = {
            "candidate_selection": "local_cohesive_frontier_lexicographic_filters",
            "denticity_phase_order": [3, 2, 1],
            "frontier_requirement": "at_least_one_exact_body_atom_contact_to_accepted_SAM",
            "new_nucleus_rule": "allowed_only_when_no_candidate_has_exact_frontier_contact",
            "perimeter_filter": {
                "objective": "minimize_periodic_union_perimeter_increment_A",
                "tolerance_A": (
                    "cohesive_perimeter_relative_tolerance_times_"
                    "minimum_candidate_perimeter_A"
                ),
                "relative_tolerance": float(
                    cohesive_manifest_tolerances["perimeter"]
                ),
                "condensation_gain_role": "diagnostic_only",
            },
            "projected_area_filter": {
                "objective": "minimize_single_candidate_XY_projection_area",
                "relative_tolerance": float(
                    cohesive_manifest_tolerances["area"]
                ),
            },
            "distinct_neighbor_filter": {
                "objective": "maximize_distinct_contacted_adsorbed_neighbors",
                "tolerance_policy": "exact_maximum",
                "allowed_count_deficit": 0,
            },
            "side_contact_filter": {
                "objective": "maximize_side_vdw_contact_pairs",
                "relative_tolerance": float(
                    cohesive_manifest_tolerances["contact"]
                ),
                "minimum_integer_tolerance_pairs": 1,
            },
            "filter_order": [
                "frontier_first",
                "minimize_periodic_union_perimeter_increment_with_"
                "minimum_candidate_perimeter_relative_band",
                "minimize_single_candidate_XY_projection_area",
                "maximize_distinct_contacted_adsorbed_neighbors",
                "maximize_side_vdw_contact_pairs",
                "minimize_conformer_energy",
                "seeded_uniform_exact_tie_break",
            ],
            "perimeter_relative_tolerance": float(
                cohesive_manifest_tolerances["perimeter"]
            ),
            "contact_relative_tolerance": float(
                cohesive_manifest_tolerances["contact"]
            ),
            "area_relative_tolerance": float(
                cohesive_manifest_tolerances["area"]
            ),
            "relative_tolerances": {
                "perimeter": float(
                    cohesive_manifest_tolerances["perimeter"]
                ),
                "contact": float(
                    cohesive_manifest_tolerances["contact"]
                ),
                "area": float(
                    cohesive_manifest_tolerances["area"]
                ),
            },
            "legacy_umbrella_relative_tolerance": cohesive_umbrella_audit[
                "legacy_umbrella_relative_tolerance"
            ],
            "legacy_umbrella_provided": cohesive_umbrella_audit[
                "legacy_umbrella_provided"
            ],
            "legacy_umbrella_applied_layers": list(
                cohesive_umbrella_audit["legacy_umbrella_applied_layers"]
            ),
            "legacy_umbrella_used": cohesive_umbrella_audit[
                "legacy_umbrella_used"
            ],
            "relative_tolerance": cohesive_umbrella_audit[
                "legacy_umbrella_relative_tolerance"
            ],
            "energy_tolerance_eV": float(args.cohesive_energy_tolerance_eV),
            "side_contact_upper_scale_of_vdw_sum": float(
                args.cohesive_contact_vdw_radius_scale
            ),
            "side_contact_excluded_atoms": "candidate_and_neighbor_P_plus_three_headgroup_O",
            "hard_collision_lower_bound": collision_thresholds_A,
            "capacity_evaluated": False,
            "accepted_molecules_moved": False,
            "periodic_axes": list(periodic_axes),
            "normal_axis_wrapped": False,
            "termination": "no_feasible_site_conformer_mapping_candidate",
            "cohesive_frontier_mode": str(args.cohesive_frontier_mode),
            "frontier_snap_radius_A": float(args.frontier_snap_radius_A),
            "algorithm_scope": (
                "frontier-ideal-position real-candidate focusing"
                if args.cohesive_frontier_mode == "continuous-snap"
                else "discrete_site_instances"
            ),
            "continuous_global_optimum_claimed": False,
            "continuous_search_qualification": (
                "Continuous-snap mode operates as a deterministic frontier-ideal-position real-candidate focusing heuristic on discrete boundary vertices and outward normals, bounded by snap_radius and discrete candidate domain; no global geometric continuous optimum is claimed."
                if args.cohesive_frontier_mode == "continuous-snap"
                else None
            ),
        }
        capacity_policy_manifest = None
    else:  # projection-aware: deterministic geometry ranking before tie-breaking.
        placement_policy_manifest = {
            "candidate_selection": "projection_aware_lexicographic_filters",
            "approved_conformer_prior_role": "final_tie_break_only",
            "headgroup_rmsd_max_A": float(args.headgroup_rmsd_max_A),
            "collision_thresholds_A": collision_thresholds_A,
            "collision_threshold_scope": "SAM-SAM_only",
            "periodic_axes": list(periodic_axes),
            "normal_axis_wrapped": False,
            "termination": "no_feasible_site_conformer_mapping_candidate",
        }
        capacity_policy_manifest = None
    manifest = {
        "schema": "sam-sequential-growth-run-manifest-v1",
        "schema_version": 4,
        "status": run_status,
        "sealed": True,
        "method": "irreversible_sequential_adsorption_without_target_coverage",
        "chemistry_scope": "H0_geometry_only_surface_protons_deferred",
        "execution_scope": "CPU_only_geometry_growth_and_requested_independent_validation",
        "selection_policy": args.selection_policy,
        "independent_validation": {
            "mode": args.independent_validation,
            "status": "unvalidated" if args.independent_validation == "off" else run_status,
            "all_required_trajectories_passed": (
                None
                if args.independent_validation == "off"
                else all_required_validation_passed
            ),
            "surface_h_policy": (
                None if args.independent_validation == "off" else "h0"
            ),
            "protons_per_molecule": (
                None if args.independent_validation == "off" else 0
            ),
        },
        "run_request": {
            "path": "run-request.json",
            "sha256": run_request_reference["sha256"],
            "bytes": run_request_reference["bytes"],
            "run_identity_sha256": run_request["run_identity"]["sha256"],
        },
        "implementation": implementation,
        "code_snapshot": code_snapshot,
        "inputs": primary_inputs,
        "selected_conformer_source_lock": selected_source_lock,
        "source_verification_checkpoints": source_verification_checkpoints,
        "loaded_conformer_ensemble_audit": loaded_conformer_audit,
        "cache_provenance": {
            "reuse": cache_reuse_contract["reuse"],
            "contract": cache_reuse_contract,
            "formal_internal_writes_retained_on_failure": bool(
                cache_reuse_contract["formal"]
            ),
            "formal_exact_namespace": bool(cache_reuse_contract["formal"]),
            "prior_reuse": False if cache_reuse_contract["formal"] else None,
            "cache_hit": False if cache_reuse_contract["formal"] else None,
            "destination_clobber": (
                False if cache_reuse_contract["formal"] else None
            ),
            **cache_artifacts,
        },
        "candidate_domain": {
            "fingerprint": domain_fingerprint,
            "fingerprint_namespace": (
                "metal-coordination-filter-v1"
                if metal_coordination_contract.get("mode") == "exclude-six-coordinated"
                else (
                    "p-surface-o-exclusion-v1"
                    if p_surface_o_exclusion_contract is not None
                    else ("legacy-v1" if args.substrate_collision == "skip" else "interface-v2")
                )
            ),
            "candidate_count": int(domain_diagnostics["candidate_count"]),
            "site_count": int(domain_diagnostics["site_count"]),
            "empty_site_count": int(domain_diagnostics["empty_site_count"]),
        },
        "fingerprints": {
            "candidate_domain": domain_fingerprint,
            "conformer_pool": conformer_pool_fingerprint,
            "conformer_cache": conformer_cache_fingerprint,
            "projection": projection_fingerprint,
            "collision_cache_contract": collision_cache_contract_fingerprint,
        },
        "working_substrate_atom_count": len(substrate),
        "archived_restore_atom_count": len(layer_groups["restore_layer_atom_ids_1based"]),
        "periodic_fractional_axes": list(periodic_axes),
        "normal_axis": normal_axis,
        "periodic_axes_source": "all_referenced_site_prototype_surface_frames",
        "molecule_formula": spec.formula,
        "max_conformers": args.max_conformers,
        "sampling_policy": {
            "formula": "rank^(-rank_exponent)*exp(-relative_energy/energy_scale)*exp(-area_coefficient*normalized_area)",
            "energy_scale_eV": policy.energy_scale_eV,
            "area_coefficient": policy.area_coefficient,
            "rank_exponent": policy.rank_exponent,
            "table": sampling_table,
        },
        "placement_policy": placement_policy_manifest,
        "projection_policy": (
            {
                "footprint_boundary_samples": args.footprint_boundary_samples,
                "radius_scale": float(args.projection_radius_scale),
                "area_tie_tolerance_A2": args.area_tie_tolerance_A2,
                "perimeter_tie_tolerance_A": args.perimeter_tie_tolerance_A,
                "scoring_metrics": (
                    {
                        "total_hole_area_after_trial": {
                            "direction": "minimize",
                            "unit": "A2",
                            "definition": "cell_area_minus_trial_union_area",
                            "evaluated_over": "full_candidate_domain_before_beam",
                        },
                        "maximum_torus_hole_area_after_trial": {
                            "direction": "minimize_after_total_debt_and_total_hole",
                            "unit": "A2",
                            "metric_version": "periodic-torus-holes-v2",
                        },
                        "total_physical_hole_perimeter_after_trial": {
                            "direction": "minimize_after_area_objectives",
                            "unit": "A",
                            "metric_version": "physical-flat-torus-perimeter-v1",
                        },
                    }
                    if args.selection_policy == "capacity-aware"
                    else (
                        {
                            "periodic_union_perimeter_increment": {
                                "direction": "minimize_then_keep_within_shared_absolute_tolerance",
                                "relative_tolerance": float(
                                    resolve_cohesive_cli_tolerances(args)[
                                        "perimeter"
                                    ]
                                ),
                                "tolerance_scale": "minimum_candidate_perimeter_A",
                                "unit": "A",
                                "condensation_gain_role": "diagnostic_only",
                            },
                            "side_vdw_contact_pairs": {
                                "direction": "maximize_then_keep_within_relative_tolerance",
                                "relative_tolerance": float(
                                    resolve_cohesive_cli_tolerances(args)[
                                        "contact"
                                    ]
                                ),
                                "minimum_integer_tolerance_pairs": 1,
                            },
                            "single_candidate_projected_area": {
                                "direction": "minimize_then_keep_within_relative_tolerance",
                                "relative_tolerance": float(
                                    resolve_cohesive_cli_tolerances(args)[
                                        "area"
                                    ]
                                ),
                                "unit": "A2",
                            },
                            "conformer_energy": {
                                "direction": "minimize_with_absolute_tolerance",
                                "tolerance_eV": float(args.cohesive_energy_tolerance_eV),
                            },
                        }
                        if args.selection_policy == "cohesive-frontier"
                        else {
                        "uncovered_anchors": {
                            "direction": "maximize",
                            "unit": "count",
                            "comparison": "exact_integer",
                        },
                        "area_increment": {
                            "direction": "minimize",
                            "unit": "A2",
                            "tie_tolerance": args.area_tie_tolerance_A2,
                        },
                        "perimeter_increment": {
                            "direction": "minimize",
                            "unit": "A",
                            "tie_tolerance": args.perimeter_tie_tolerance_A,
                            "metric_version": "physical-flat-torus-perimeter-v1",
                        },
                    })
                ),
                "vdw_radii_source": "ase.data.vdw_radii",
                **(
                    {"role": "capacity_hole_geometry_and_torus_diagnostics"}
                    if args.selection_policy == "capacity-aware"
                    else (
                        {"role": "cohesive_frontier_and_intrinsic_footprint_geometry"}
                        if args.selection_policy == "cohesive-frontier"
                        else {}
                    )
                ),
                "cache": {
                    "hit": projection_cache_manifest["hit"],
                    "rebuild_reason": projection_cache_manifest.get("rebuild_reason"),
                    "contract": projection_cache_manifest["contract"],
                    "template_count": projection_cache_manifest["template_count"],
                    "candidate_count": projection_cache_manifest["candidate_count"],
                    "data_sha256": projection_cache_manifest.get("data_sha256"),
                },
            }
        ),
        "collision_cache": {
            "mode": args.collision_cache,
            "summary": (
                collision_cache.summary() if collision_cache is not None else None
            ),
        },
        "selection_criterion": (
            None
            if args.selection_policy in {"capacity-aware", "cohesive-frontier"}
            else args.selection_criterion
        ),
        "domain_diagnostics": domain_diagnostics,
        "candidate_domain_fingerprint": domain_fingerprint,
        "candidate_domain_fingerprint_namespace": (
            "metal-coordination-filter-v1"
            if metal_coordination_contract.get("mode") == "exclude-six-coordinated"
            else (
                "p-surface-o-exclusion-v1"
                if p_surface_o_exclusion_contract is not None
                else ("legacy-v1" if args.substrate_collision == "skip" else "interface-v2")
            )
        ),
        "sam_sam_collision_contract": {
            "thresholds_A": collision_thresholds_A,
            "periodic_axes": list(periodic_axes),
            "normal_axis_wrapped": False,
            "independent_of_substrate_collision_mode": True,
        },
        "substrate_collision_mode": args.substrate_collision,
        "substrate_collision_contract": substrate_collision_contract,
        "p_surface_o_exclusion": (
            None
            if p_surface_o_exclusion_contract is None
            else {
                "contract": p_surface_o_exclusion_contract,
                "audit_artifact": p_surface_o_audit_reference,
                "static_capacity": p_surface_o_static_capacity,
                "candidate_count_before_exclusion": int(domain_diagnostics["p_surface_o_exclusion"]["candidate_count_before_exclusion"]),
                "excluded_placement_count": int(domain_diagnostics["p_surface_o_exclusion"]["excluded_placement_count"]),
                "retained_candidate_count": int(domain_diagnostics["p_surface_o_exclusion"]["retained_candidate_count"]),
                "other_substrate_pairs_checked": False,
                "other_substrate_pairs_rejected": False,
            }
        ),
        "metal_coordination_filter": (
            None
            if metal_coordination_contract.get("mode") == "off"
            else {
                "contract": metal_coordination_contract,
                "audit_artifact": metal_coord_audit_reference,
                "site_instance_summary": domain_diagnostics["metal_coordination_filter"]["site_instance_summary"],
                "candidate_filter_summary": domain_diagnostics["metal_coordination_filter"]["candidate_filter_summary"],
            }
        ),
        "final_registered_interface_audit_policy": {
            "performed_for_independent_validation_modes": ["h0", "strict-interface", "algorithmic-count-only"],
            "required_for_independent_validation_mode": "strict-interface",
            "diagnostic_for_independent_validation_mode": "h0",
            "algorithmic_count_only_role": "complete_strict_vdw_audit_retained_but_not_a_physical_pass_gate",
            "legacy_off_mode_hard_gate_for_substrate_modes": ["strict-vdw", "height"],
            "height_clearance_A": float(args.substrate_height_clearance_A),
            "radius_scale": float(args.substrate_vdw_radius_scale),
            "radii_source": "ASE ase.data.vdw_radii",
            "periodic_axes": list(periodic_axes),
            "normal_axis_wrapped": False,
            "site_uniqueness_checked": True,
            "actual_metal_capacity_one_checked": True,
            "nonregistered_vdw_contacts_checked": True,
            "mapped_distance_windows_checked": True,
        },
        "timing": {
            "candidate_domain_seconds": domain_seconds,
            "collision_query_profiler": collision_profiler.summary(),
            "total_wall_time_seconds": time.perf_counter() - run_started,
        },
        "cpu_environment": _cpu_environment(),
        "trajectory_count": args.trajectory_count,
        "trajectories": trajectory_summaries,
        "aggregate": summarize_trajectory_results(
            trajectory_results, sampling_table
        ),
    }
    if args.selection_policy == "capacity-aware":
        manifest.update(
            {
                "capacity_policy": capacity_policy_manifest,
                "optimization_scope": "exact_one_step_within_deterministic_beam",
                "global_candidate_optimality": False,
                "global_packing_certificate": False,
            }
        )
    manifest["artifact_inventory_scope"] = (
        "every_regular_output_file_before_top_manifest_seal;top_manifest_self_hash_excluded"
    )
    manifest["artifact_inventory"] = _artifact_inventory(
        output_dir, excluded_names={"manifest.json", "failure.json"}
    )
    run_request_preseal_verification = _verify_regular_artifact_reference(
        output_dir / "run-request.json",
        run_request_reference,
        label="run-request.json",
    )
    code_snapshot_preseal_verification = _verify_code_snapshot(
        output_dir, code_snapshot
    )
    manifest["run_request"]["pre_top_manifest_seal_verification"] = {
        **run_request_preseal_verification,
        "path": "run-request.json",
        "checkpoint": "immediately_before_top_manifest_seal",
    }
    manifest["code_snapshot"]["pre_top_manifest_seal_verification"] = (
        code_snapshot_preseal_verification
    )
    if p_surface_o_exclusion_contract is not None:
        verify_p_surface_o_exclusion_contract_sources(p_surface_o_exclusion_contract)
    write_strict_json_noclobber(output_dir / "manifest.json", manifest)
    return 0 if run_status not in {"failed_independent_h0_validation", "failed_cpu_h0_algorithmic_count_only_validation"} else 1


def _validate_p_surface_o_cli_requirements(args) -> None:
    """Fail closed on the explicit P--surface-O experiment contract."""

    evidence_values = (
        args.p_surface_o_contract,
        args.p_surface_o_contract_sha256,
        args.p_surface_o_source_report,
        args.p_surface_o_source_report_sha256,
        args.p_surface_o_source_config,
        args.p_surface_o_source_config_sha256,
    )
    if args.p_surface_o_exclusion == "off":
        if any(value is not None for value in evidence_values):
            raise ValueError("P--surface-O evidence arguments require --p-surface-o-exclusion contract")
        if args.independent_validation == "algorithmic-count-only":
            raise ValueError("algorithmic-count-only requires --p-surface-o-exclusion contract")
        return
    if args.operation != "grow":
        raise ValueError("P--surface-O initial exclusion is available only for grow")
    if args.substrate_collision != "skip":
        raise ValueError("P--surface-O initial exclusion requires --substrate-collision skip")
    if args.selection_policy != "projection-aware":
        raise ValueError("P--surface-O initial exclusion requires --selection-policy projection-aware")
    if args.selection_criterion != "uncovered-anchors":
        raise ValueError("P--surface-O initial exclusion requires --selection-criterion uncovered-anchors")
    if int(args.max_conformers) != 10:
        raise ValueError("P--surface-O initial exclusion requires --max-conformers 10")
    if args.collision_cache != "off":
        raise ValueError("P--surface-O initial exclusion requires --collision-cache off")
    thresholds = {
        "H-H": float(args.hh_min),
        "H-heavy": float(args.h_heavy_min),
        "heavy-heavy": float(args.heavy_heavy_min),
    }
    if thresholds != {"H-H": 1.5, "H-heavy": 1.8, "heavy-heavy": 2.2}:
        raise ValueError("P--surface-O initial exclusion requires SAM-SAM thresholds exactly 1.5/1.8/2.2")
    if args.independent_validation != "algorithmic-count-only":
        raise ValueError("P--surface-O initial exclusion requires --independent-validation algorithmic-count-only")
    if any(value is None for value in evidence_values):
        raise ValueError(
            "P--surface-O contract mode requires contract/report/scratch paths and explicit SHA-256 values"
        )


def sequential_growth_main(argv=None) -> int:
    """Reserve one immutable output directory, then execute and seal failures."""

    raw_argv = [str(value) for value in (sys.argv[1:] if argv is None else argv)]
    argv_sha256 = _payload_sha256(raw_argv)
    parser = _build_parser()
    args = parser.parse_args(raw_argv)
    if getattr(args, "single_metal_sites", "off") != "off":
        if args.operation != "grow":
            parser.error("--single-metal-sites dynamic-cn5-uncovered requires --operation grow")
        if args.selection_policy != "cohesive-frontier":
            parser.error("--single-metal-sites dynamic-cn5-uncovered requires --selection-policy cohesive-frontier")
        if args.substrate_collision != "height":
            parser.error("--single-metal-sites dynamic-cn5-uncovered requires --substrate-collision height")
        if args.independent_validation != "off":
            parser.error("--single-metal-sites dynamic-cn5-uncovered requires --independent-validation off")
        if getattr(args, "metal_coordination_filter", "off") != "exclude-six-coordinated":
            parser.error("--single-metal-sites dynamic-cn5-uncovered requires --metal-coordination-filter exclude-six-coordinated")
    if (
        args.substrate_collision == "height"
        and args.independent_validation != "off"
    ):
        parser.error(
            "--substrate-collision height currently requires "
            "--independent-validation off; the height gate is independently "
            "re-audited from final coordinates in that mode"
        )
    if args.max_conformers <= 0 or args.trajectory_count <= 0:
        parser.error("--max-conformers and --trajectory-count must be positive")
    if args.operation == "stage3-diagnostic":
        if (
            args.stage1_parent_run is None
            or args.stage2_parent_run is None
            or args.corrected_reference_metrics is None
        ):
            parser.error(
                "--operation stage3-diagnostic requires --stage1-parent-run, "
                "--stage2-parent-run, and --corrected-reference-metrics"
            )
        if args.parent_run is not None or args.reference_metrics is not None:
            parser.error(
                "--parent-run/--reference-metrics are only valid for --operation repair"
            )
        if args.write_every_step:
            parser.error("--write-every-step is invalid for Stage3 diagnostic")
    elif args.operation == "repair":
        if args.parent_run is None or args.reference_metrics is None:
            parser.error(
                "--operation repair requires --parent-run and --reference-metrics"
            )
        if args.trajectory_count != 1:
            parser.error("--operation repair requires --trajectory-count 1")
        if args.independent_validation == "off":
            parser.error(
                "--operation repair requires --independent-validation h0 or strict-interface"
            )
        if args.write_every_step:
            parser.error(
                "--write-every-step is invalid for postprocessed local repair"
            )
    elif args.parent_run is not None or args.reference_metrics is not None:
        parser.error(
            "--parent-run/--reference-metrics require --operation repair"
        )
    try:
        _validate_p_surface_o_cli_requirements(args)
    except ValueError as exc:
        parser.error(str(exc))
    if (
        not np.isfinite(args.headgroup_rmsd_max_A)
        or args.headgroup_rmsd_max_A <= 0.0
    ):
        parser.error("--headgroup-rmsd-max-A must be a finite positive number")
    try:
        _validated_registered_bond_window(
            (args.mapped_bond_min_A, args.mapped_bond_max_A)
        )
    except ValueError as exc:
        parser.error(str(exc))

    root = args.root.expanduser().resolve()

    def lexical(path: Path) -> Path:
        expanded = path.expanduser()
        return expanded if expanded.is_absolute() else root / expanded

    def resolved(path: Path) -> Path:
        return lexical(path).resolve()

    path_arguments = {
        "substrate": args.substrate,
        "layer_groups": args.layer_groups,
        "site_instances": args.site_instances,
        "site_prototypes": args.site_prototypes,
        "conformer_analysis": args.conformer_analysis,
        "conformer_cache": args.conformer_cache,
        "projection_cache": args.projection_cache,
    }
    if args.p_surface_o_exclusion == "contract":
        path_arguments.update(
            {
                "p_surface_o_contract": args.p_surface_o_contract,
                "p_surface_o_source_report": args.p_surface_o_source_report,
                "p_surface_o_source_config": args.p_surface_o_source_config,
            }
        )
    lexical_paths = {
        name: lexical(path) for name, path in path_arguments.items()
    }
    resolved_paths = {
        name: resolved(path) for name, path in path_arguments.items()
    }
    p_surface_o_contract = None
    if args.p_surface_o_exclusion == "contract":
        p_surface_o_contract = load_p_surface_o_exclusion_contract(
            contract_path=resolved_paths["p_surface_o_contract"],
            contract_sha256=args.p_surface_o_contract_sha256,
            source_report_path=resolved_paths["p_surface_o_source_report"],
            source_report_sha256=args.p_surface_o_source_report_sha256,
            source_config_path=resolved_paths["p_surface_o_source_config"],
            source_config_sha256=args.p_surface_o_source_config_sha256,
        )
    output_dir = resolved(args.output_dir)
    if args.operation == "repair":
        _validate_repair_output_evidence_isolation(
            output_dir=output_dir,
            parent_run=resolved(args.parent_run),
            reference_metrics=resolved(args.reference_metrics),
        )
    implementation_capture = _capture_implementation_identity_best_effort()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    # This mkdir is the sole reservation operation.  Any pre-existing path,
    # including an empty directory, loses atomically without a prior check.
    output_dir.mkdir(exist_ok=False)
    try:
        _fsync_directory(output_dir.parent)
        if implementation_capture["status"] != "captured":
            capture_error = implementation_capture[
                "failure_document_identity"
            ]["capture_error"]
            raise RuntimeError(
                "Implementation identity was unavailable before reservation: "
                f"{capture_error['type']}: {capture_error['message']}"
            )
        if args.operation == "stage3-diagnostic":
            return execute_stage3_diagnostic(
                args,
                argv=raw_argv,
                root=root,
                output_dir=output_dir,
                resolved_paths=resolved_paths,
                implementation_identity=implementation_capture["identity"],
            )
        cache_reuse_contract = _cache_reuse_contract(
            args=args,
            output_dir=output_dir,
            resolved_paths=resolved_paths,
            lexical_paths=lexical_paths,
        )
        return execute_sequential_growth(
            args,
            argv=raw_argv,
            root=root,
            output_dir=output_dir,
            resolved_paths=resolved_paths,
            cache_reuse_contract=cache_reuse_contract,
            implementation_identity=implementation_capture["identity"],
            p_surface_o_exclusion_contract=p_surface_o_contract,
        )
    except BaseException as error:
        try:
            _write_failure_document(
                output_dir=output_dir,
                error=error,
                argv_sha256=argv_sha256,
                implementation_identity=implementation_capture[
                    "failure_document_identity"
                ],
                operation=args.operation,
            )
        except BaseException as failure_error:
            print(
                "Sequential-growth failure could not be sealed without clobber: "
                f"{type(failure_error).__name__}: {failure_error}",
                file=sys.stderr,
                flush=True,
            )
        print(
            f"Sequential-growth run failed: {type(error).__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(sequential_growth_main())
