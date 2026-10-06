"""Resolve curated substrate names into auditable construction plans."""

from __future__ import annotations

import argparse
import os
import hashlib
import json
import math
import re
import shutil
import socket
import sys
import tomllib
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from functools import reduce
from itertools import combinations, permutations, product
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from ase import Atoms
from ase.build import make_supercell, surface
from ase.build.supercells import lattice_points_in_supercell
import ase
from ase.data import atomic_numbers, chemical_symbols, covalent_radii, vdw_radii
from ase.geometry import find_mic
from ase.io import read, write

from sam_lammps import write_clean_structure, write_manifest
from sam_structure_tools import (
    DISTANCE_WINDOW_COMPARISON_ULPS,
    closed_distance_window_tolerance_A,
    cluster_1d_by_tolerance,
    distance_within_closed_window,
    filled_outer_envelope_area,
    periodic_vdw_collision_audit,
    rigid_transform,
)
from validate_monolayer import ValidationConfig, validate as validate_monolayer


CATALOG_SCHEMA_VERSION = 1
STRUCTURE_MANIFEST_SCHEMA_VERSION = 1
GPA_PER_EV_A3 = 160.21766208
J_M2_PER_EV_A2 = 16.02176634
MAXIMUM_SIZED_SURFACE_LATTICE_VECTORS = 100_000
DENSE_H0_NORMAL_TRANSLATION_STEP_A = 0.1
DENSE_H0_SITE_ENERGY_QUANTUM_EV = 1.0e-12
COLLISION_AUDIT_IMPLEMENTATION_PATH = Path(
    periodic_vdw_collision_audit.__code__.co_filename
).resolve()
INTERFACE_VALIDATOR_IMPLEMENTATION_PATH = Path(
    validate_monolayer.__code__.co_filename
).resolve()
SURFACE_PROTONATION_IMPLEMENTATION_PATH = Path(__file__).with_name(
    "surface_protonation.py"
).resolve()


def _default_catalog_directory() -> Path:
    external = os.environ.get("SAMFLOW_SUBSTRATE_CATALOG")
    if external:
        return Path(external).expanduser().resolve()
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "substrates"
        if candidate.is_dir():
            return candidate
    return Path(__file__).resolve().parents[1] / "substrates"


def _default_structure_directory() -> Path:
    return _default_catalog_directory() / "structures"


def _default_benchmark_ledger_path() -> Path:
    external = os.environ.get("SAMFLOW_BENCHMARK_LEDGER")
    if external:
        return Path(external).expanduser().resolve()
    relative = Path("configs") / "samflow-benchmark-ledger.json"
    for parent in Path(__file__).resolve().parents:
        candidate = parent / relative
        if candidate.is_file():
            return candidate
    return Path(__file__).resolve().parents[1] / relative


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value) -> str:
    """Hash one JSON-compatible value with the project canonical encoding."""

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _approval_payload_sha256(payload: dict) -> str:
    """Hash the exact canonical JSON payload shown for dense-build approval."""

    if not isinstance(payload, dict):
        raise ValueError("approval payload must be a dictionary")
    return _canonical_json_sha256(payload)


def _dense_h0_execution_plan_identity(plan: dict) -> dict:
    """Return every plan section that the sized-H0 builder may execute from."""

    return {
        "mode": plan.get("mode"),
        "workflow": plan.get("workflow"),
        "inputs": plan.get("inputs"),
        "surface_size_request": plan.get("surface_size_request"),
        "capacity_estimate": plan.get("capacity_estimate"),
        "resource_estimate": plan.get("resource_estimate"),
        "substrate": plan.get("substrate"),
    }


def _formula(symbols) -> dict[str, int]:
    return dict(sorted(Counter(symbols).items()))


def _normalized_name(value: str) -> str:
    return " ".join(value.casefold().replace("_", " ").replace("-", " ").split())


def _table(value, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"Missing or invalid [{name}] table")
    return value


def _positive_int(value, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_number(value, name: str, *, allow_zero: bool = False) -> float:
    valid = isinstance(value, (int, float)) and not isinstance(value, bool)
    minimum_ok = valid and (value >= 0 if allow_zero else value > 0)
    if not valid or not minimum_ok:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} number")
    return float(value)


def _integer_vector(value, name: str, *, length: int = 3) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) != length
        or any(not isinstance(item, int) or isinstance(item, bool) for item in value)
    ):
        raise ValueError(f"{name} must contain {length} integers")
    return list(value)


def _element(value, name: str) -> str:
    if not isinstance(value, str) or value not in atomic_numbers:
        raise ValueError(f"{name} must be a valid element symbol")
    return value


def _load_catalog_file(path: Path, anchor_family: str | None = None) -> dict:
    with path.open("rb") as handle:
        recipe = tomllib.load(handle)
    adsorption_profiles = (recipe.get("adsorption") or {}).get(
        "site_discovery_profiles", {}
    )
    if anchor_family is not None:
        if not isinstance(anchor_family, str) or not anchor_family.strip():
            raise ValueError("anchor_family must be a non-empty string")
        adsorption = dict(recipe.get("adsorption") or {})
        current_discovery = adsorption.get("site_discovery") or {}
        current_family = (
            current_discovery.get("site_prototype_clustering") or {}
        ).get("anchor_family")
        if anchor_family != current_family:
            profile = adsorption_profiles.get(anchor_family)
            if not isinstance(profile, dict):
                available = sorted(
                    {value for value in [current_family] if isinstance(value, str)}
                    | set(adsorption_profiles)
                )
                raise ValueError(
                    f"{path}: no adsorption site-discovery profile for "
                    f"{anchor_family!r}; available: {available}"
                )
            discovery = profile.get("site_discovery")
            if not isinstance(discovery, dict):
                raise ValueError(
                    f"{path}: adsorption profile {anchor_family!r} requires "
                    "a site_discovery table"
                )
            for field in (
                "anchor_element",
                "anchor_neighbor_pattern",
                "released_protons",
                "surface_parent_element",
            ):
                if field not in profile:
                    raise ValueError(
                        f"{path}: adsorption profile {anchor_family!r} is missing {field}"
                    )
                adsorption[field] = profile[field]
            adsorption["site_discovery"] = discovery
        adsorption.pop("site_discovery_profiles", None)
        recipe["adsorption"] = adsorption
    if recipe.get("schema_version") != CATALOG_SCHEMA_VERSION:
        raise ValueError(f"{path}: schema_version must be {CATALOG_SCHEMA_VERSION}")
    key = recipe.get("key")
    aliases = recipe.get("aliases", [])
    if not isinstance(key, str) or not key.strip():
        raise ValueError(f"{path}: key must be a non-empty string")
    if not isinstance(aliases, list) or any(
        not isinstance(alias, str) or not alias.strip() for alias in aliases
    ):
        raise ValueError(f"{path}: aliases must be a string list")
    status = recipe.get("status")
    supported_statuses = {
        "validated_surface_recipe",
        "validated_surface_site_discovery_recipe",
    }
    if status not in supported_statuses:
        raise ValueError(
            f"{path}: status must be one of {sorted(supported_statuses)}"
        )
    site_discovery_only = status == "validated_surface_site_discovery_recipe"

    source = _table(recipe.get("source"), "source")
    source_kind = source.get("kind")
    if source_kind not in {"curated_surface_slab", "bulk_crystal"}:
        raise ValueError(f"{path}: unsupported source.kind {source_kind!r}")
    structure_value = source.get("structure")
    if not isinstance(structure_value, str) or not structure_value.strip():
        raise ValueError(f"{path}: source.structure must be a non-empty path")
    source_path = (path.parent / structure_value).resolve()
    if not source_path.is_file():
        raise ValueError(f"{path}: source structure does not exist: {source_path}")
    expected_sha256 = source.get("sha256")
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ValueError(f"{path}: source.sha256 must be a lowercase SHA-256 digest")
    observed_sha256 = _sha256(source_path)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            f"{path}: source hash mismatch for {source_path}; "
            f"observed {observed_sha256}"
        )
    if source_kind == "curated_surface_slab":
        structure_manifest_value = source.get("structure_manifest")
        if (
            not isinstance(structure_manifest_value, str)
            or not structure_manifest_value.strip()
        ):
            raise ValueError(
                f"{path}: curated surface source requires source.structure_manifest"
            )
        structure_manifest_path = (
            path.parent / structure_manifest_value
        ).resolve()
        if not structure_manifest_path.is_file():
            raise ValueError(
                f"{path}: structure manifest does not exist: "
                f"{structure_manifest_path}"
            )
        structure_manifest = json.loads(
            structure_manifest_path.read_text(encoding="utf-8")
        )
        registered = []
        for precut in structure_manifest.get("precut_slabs", []):
            if not isinstance(precut, dict) or not isinstance(precut.get("path"), str):
                continue
            registered_path = (
                structure_manifest_path.parent / precut["path"]
            ).resolve()
            if registered_path == source_path:
                registered.append(precut)
        if len(registered) != 1:
            raise ValueError(
                f"{path}: curated surface must have one matching precut_slabs record"
            )
        precut = registered[0]
        if (
            precut.get("status")
            not in {
                "accepted_bulk_derived_unrelaxed_precut",
                "accepted_relaxed_maintained_surface",
            }
            or precut.get("sha256") != expected_sha256
        ):
            raise ValueError(
                f"{path}: curated surface registration is not accepted or is stale"
            )
        bulk_parent = structure_manifest.get("bulk_parent") or {}
        if precut.get("parent_bulk_sha256") != bulk_parent.get("sha256"):
            raise ValueError(f"{path}: curated surface parent Bulk Parent is stale")
        records = precut.get("records")
        if not isinstance(records, list):
            raise ValueError(f"{path}: curated surface records must be a list")
        records_by_role = {}
        for record in records:
            if not isinstance(record, dict) or not isinstance(record.get("path"), str):
                raise ValueError(f"{path}: invalid curated surface record")
            role = record.get("role")
            if not isinstance(role, str) or role in records_by_role:
                raise ValueError(f"{path}: invalid or duplicate surface record role")
            record_path = (structure_manifest_path.parent / record["path"]).resolve()
            if not record_path.is_file() or _sha256(record_path) != record.get("sha256"):
                raise ValueError(
                    f"{path}: curated surface record is missing or changed: {record_path}"
                )
            records_by_role[role] = record_path
        if precut["status"] == "accepted_bulk_derived_unrelaxed_precut":
            required_roles = {
                "surface_precut_plan",
                "selection_report",
                "run_manifest",
            }
            if not required_roles.issubset(records_by_role):
                raise ValueError(f"{path}: curated surface evidence chain is incomplete")
            selection_report = json.loads(
                records_by_role["selection_report"].read_text(encoding="utf-8")
            )
            run_manifest = json.loads(
                records_by_role["run_manifest"].read_text(encoding="utf-8")
            )
            selected_artifacts = [
                artifact
                for artifact in run_manifest.get("artifacts", [])
                if artifact.get("role") == "selected_surface_precut_candidate"
            ]
            valid = (
                selection_report.get("status") == "passed_selection"
                and run_manifest.get("status") == "passed_selection"
                and run_manifest.get("fingerprint") == precut.get("run_fingerprint")
                and run_manifest.get("bulk_parent_sha256")
                == precut.get("parent_bulk_sha256")
                and len(selected_artifacts) == 1
                and selected_artifacts[0].get("sha256") == expected_sha256
            )
        else:
            valid = _relaxed_surface_evidence_is_valid(
                precut,
                records_by_role,
                expected_sha256,
            )
        if not valid:
            raise ValueError(f"{path}: curated surface evidence is invalid or stale")
    if not isinstance(source.get("provenance"), str) or not source["provenance"].strip():
        raise ValueError(f"{path}: source.provenance must be a non-empty string")

    surface = _table(recipe.get("surface"), "surface")
    miller = _integer_vector(surface.get("miller"), "surface.miller")
    if miller == [0, 0, 0]:
        raise ValueError("surface.miller cannot be [0, 0, 0]")
    normal_axis = surface.get("normal_axis")
    if normal_axis not in {0, 1, 2}:
        raise ValueError("surface.normal_axis must be 0, 1, or 2")
    if not isinstance(surface.get("termination"), str) or not surface["termination"].strip():
        raise ValueError("surface.termination must be a non-empty string")

    supercell = recipe.get("supercell")
    transform = None
    if site_discovery_only:
        if supercell is not None:
            raise ValueError(
                "A site-discovery-only recipe must leave [supercell] unresolved"
            )
    else:
        supercell = _table(supercell, "supercell")
        transform = supercell.get("transform")
        if not isinstance(transform, list) or len(transform) != 3:
            raise ValueError("supercell.transform must be a 3x3 integer matrix")
        transform = np.asarray(
            [_integer_vector(row, "supercell.transform row") for row in transform],
            dtype=int,
        )
        determinant = int(round(np.linalg.det(transform)))
        if determinant <= 0:
            raise ValueError("supercell.transform must have a positive determinant")
        repeat = _integer_vector(supercell.get("repeat"), "supercell.repeat")
        if any(item < 1 for item in repeat):
            raise ValueError("supercell.repeat values must be positive")
        if repeat[normal_axis] != 1:
            raise ValueError("supercell.repeat must not change slab thickness")
        _positive_int(
            supercell.get("minimum_sam_count"), "supercell.minimum_sam_count"
        )
        supported_workflows = supercell.get("supported_workflows")
        allowed_workflows = {"single_molecule_adsorption", "dense_monolayer"}
        if (
            not isinstance(supported_workflows, list)
            or not supported_workflows
            or any(value not in allowed_workflows for value in supported_workflows)
            or len(set(supported_workflows)) != len(supported_workflows)
        ):
            raise ValueError(
                "supercell.supported_workflows must contain unique supported public "
                "workflow names"
            )
        if "single_molecule_adsorption" not in supported_workflows:
            raise ValueError(
                "Every materializable substrate must support single_molecule_adsorption"
            )

    single_molecule = recipe.get("single_molecule_adsorption")
    if site_discovery_only:
        if single_molecule is not None:
            raise ValueError(
                "A site-discovery-only recipe must leave "
                "[single_molecule_adsorption] unresolved"
            )
    else:
        single_molecule = _table(
            single_molecule, "single_molecule_adsorption"
        )
        if single_molecule.get("target_sam_count") != 1:
            raise ValueError(
                "single_molecule_adsorption.target_sam_count must be 1"
            )
        expected_single_contract = {
            "size_metric": "maximum_atom_pair_distance",
            "supercell_selection": (
                "minimum_source_cell_multiplier_meeting_image_clearance"
            ),
        }
        for field, expected in expected_single_contract.items():
            if single_molecule.get(field) != expected:
                raise ValueError(
                    f"single_molecule_adsorption.{field} must be {expected!r}"
                )
        maximum_repeat = _integer_vector(
            single_molecule.get("maximum_repeat"),
            "single_molecule_adsorption.maximum_repeat",
        )
        if (
            any(value < 1 for value in maximum_repeat)
            or maximum_repeat[normal_axis] != 1
        ):
            raise ValueError(
                "single_molecule_adsorption.maximum_repeat must contain positive "
                "values and must not repeat the slab normal"
            )
        for field in (
            "minimum_lateral_image_clearance_A",
            "minimum_bottom_vacuum_A",
            "minimum_top_image_clearance_A",
        ):
            _positive_number(
                single_molecule.get(field),
                f"single_molecule_adsorption.{field}",
            )

    slab = _table(recipe.get("slab"), "slab")
    source_layers = _positive_int(slab.get("source_layers"), "slab.source_layers")
    target_layers = slab.get("target_layers")
    working_layers = slab.get("working_layers")
    if site_discovery_only:
        if target_layers is not None or working_layers is not None:
            raise ValueError(
                "A site-discovery-only recipe must leave target/working layers unresolved"
            )
    else:
        target_layers = _positive_int(target_layers, "slab.target_layers")
        working_layers = _positive_int(working_layers, "slab.working_layers")
        if working_layers > target_layers:
            raise ValueError("slab.working_layers cannot exceed slab.target_layers")
        if source_kind == "curated_surface_slab" and target_layers != source_layers:
            raise ValueError(
                "A curated_surface_slab currently requires target_layers == source_layers"
            )
    for field in ("layer_gap_A", "minimum_thickness_A", "minimum_vacuum_A"):
        _positive_number(slab.get(field), f"slab.{field}")

    doping = recipe.get("doping")
    if site_discovery_only:
        if doping is not None:
            raise ValueError(
                "A site-discovery-only recipe must leave [doping] unresolved"
            )
    else:
        doping = _table(doping, "doping")
        if doping.get("method") != "substitution":
            raise ValueError("doping.method currently must be substitution")
        host_element = _element(doping.get("host_element"), "doping.host_element")
        dopant_element = _element(doping.get("dopant_element"), "doping.dopant_element")
        if host_element == dopant_element:
            raise ValueError("doping host and dopant elements must differ")
        if doping.get("layer_partition") != (
            "equal_stoichiometric_blocks_by_outward_normal"
        ):
            raise ValueError(
                "doping.layer_partition must use equal stoichiometric blocks by "
                "outward normal"
            )
        if doping.get("distribution") not in {
            "layer_stratified_farthest",
            "layer_stratified_global_farthest",
        }:
            raise ValueError("Unsupported deterministic doping distribution")
        if doping.get("layer_integer_allocation") not in {
            "largest_remainder_bottom_to_top",
            "largest_remainder_maximize_layer_span",
        }:
            raise ValueError("Unsupported doping layer-integer allocation policy")
        eligible_layers = doping.get("eligible_layer_indices_bottom_to_top")
        if (
            not isinstance(eligible_layers, list)
            or not eligible_layers
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or not 1 <= value <= target_layers
                for value in eligible_layers
            )
            or len(set(eligible_layers)) != len(eligible_layers)
            or eligible_layers != sorted(eligible_layers)
        ):
            raise ValueError(
                "doping.eligible_layer_indices_bottom_to_top must contain unique "
                "ascending target-layer indices"
            )
        target_fraction = _positive_number(
            doping.get("target_site_fraction"), "doping.target_site_fraction"
        )
        tolerance = _positive_number(
            doping.get("tolerance"), "doping.tolerance", allow_zero=True
        )
        if target_fraction >= 1 or tolerance >= 1:
            raise ValueError("doping fractions must be less than 1")
        layer_fractions = doping.get("layer_site_fractions")
        if not isinstance(layer_fractions, list) or len(layer_fractions) != target_layers:
            raise ValueError(
                "doping.layer_site_fractions must have one value per target layer"
            )
        for index, fraction in enumerate(layer_fractions):
            fraction = _positive_number(
                fraction, f"doping.layer_site_fractions[{index}]", allow_zero=True
            )
            if fraction >= 1:
                raise ValueError("doping layer fractions must be less than 1")
            if index + 1 not in eligible_layers and fraction != 0:
                raise ValueError(
                    "Ineligible doping layers must have zero layer_site_fraction"
                )
        _positive_number(
            doping.get("minimum_separation_A"), "doping.minimum_separation_A"
        )
        if not isinstance(doping.get("seed"), int) or isinstance(
            doping.get("seed"), bool
        ):
            raise ValueError("doping.seed must be an integer")
        fraction_basis = _table(
            doping.get("target_fraction_basis"),
            "doping.target_fraction_basis",
        )
        allowed_basis_kinds = {
            "project_reference_cation_site_fraction",
            "experimental_material_grade",
            "primary_literature_reference",
            "literature_guided_computational_default",
        }
        if fraction_basis.get("kind") not in allowed_basis_kinds:
            raise ValueError(
                "doping.target_fraction_basis.kind must identify an accepted basis"
            )
        for field in ("status", "quantity", "objective", "selection_method"):
            if not isinstance(fraction_basis.get(field), str) or not fraction_basis[
                field
            ].strip():
                raise ValueError(
                    f"doping.target_fraction_basis.{field} must be non-empty"
                )
        numerator = _positive_int(
            fraction_basis.get("numerator"),
            "doping.target_fraction_basis.numerator",
        )
        denominator = _positive_int(
            fraction_basis.get("denominator"),
            "doping.target_fraction_basis.denominator",
        )
        if numerator >= denominator:
            raise ValueError("Dopant fraction-basis ratio must be less than one")
        if abs(numerator / denominator - target_fraction) > 1.0e-12:
            raise ValueError(
                "doping.target_site_fraction does not match its registered basis ratio"
            )
        physical_target = _positive_number(
            fraction_basis.get("physical_target_fraction"),
            "doping.target_fraction_basis.physical_target_fraction",
        )
        if physical_target >= 1:
            raise ValueError(
                "doping.target_fraction_basis.physical_target_fraction must be less "
                "than 1"
            )
        evidence = fraction_basis.get("evidence")
        if not isinstance(evidence, list) or not evidence or any(
            not isinstance(value, str) or not value.strip() for value in evidence
        ):
            raise ValueError(
                "doping.target_fraction_basis.evidence must be a non-empty string list"
            )
        for field in ("rationale", "limitation", "user_override_policy"):
            if not isinstance(fraction_basis.get(field), str) or not fraction_basis[
                field
            ].strip():
                raise ValueError(
                    f"doping.target_fraction_basis.{field} must be non-empty"
                )
        user_override = _table(doping.get("user_override"), "doping.user_override")
        if user_override.get("enabled") is not True:
            raise ValueError("doping.user_override.enabled must be true")
        for field in (
            "input_quantity",
            "cli_option",
            "outside_range_policy",
            "range_rationale",
        ):
            if not isinstance(user_override.get(field), str) or not user_override[
                field
            ].strip():
                raise ValueError(f"doping.user_override.{field} must be non-empty")
        if user_override.get("cli_option") != "--dopant-site-percent":
            raise ValueError(
                "doping.user_override.cli_option must be '--dopant-site-percent'"
            )
        if user_override.get("layer_profile_policy") != (
            "scale_catalog_layer_site_fractions"
        ):
            raise ValueError(
                "doping.user_override.layer_profile_policy must scale the catalog "
                "layer profile"
            )
        minimum_percent = _positive_number(
            user_override.get("minimum_percent"),
            "doping.user_override.minimum_percent",
            allow_zero=True,
        )
        maximum_percent = _positive_number(
            user_override.get("maximum_percent"),
            "doping.user_override.maximum_percent",
        )
        if maximum_percent >= 100 or minimum_percent >= maximum_percent:
            raise ValueError(
                "doping.user_override percentage range must satisfy "
                "0 <= minimum < maximum < 100"
            )
        if not minimum_percent <= 100.0 * target_fraction <= maximum_percent:
            raise ValueError(
                "The catalog default dopant fraction must lie inside the public "
                "override range"
            )

    adsorption = _table(recipe.get("adsorption"), "adsorption")
    sites_per_source_cell = adsorption.get("sites_per_source_cell")
    if site_discovery_only:
        if sites_per_source_cell is not None:
            raise ValueError(
                "A site-discovery-only recipe must not guess sites_per_source_cell"
            )
    else:
        if sites_per_source_cell is not None:
            raise ValueError(
                "A materializable recipe must resolve capacity from accepted Site "
                "Prototypes, not adsorption.sites_per_source_cell"
            )
        expected_site_contract = {
            "site_source": "accepted_site_prototype_library",
            "site_instance_protocol": "site_prototype_supercell_expansion_v1",
            "coverage_policy": "fixed_catalog_minimum_sam_count",
            "selection_objective": (
                "minimum_total_undoped_parent_adsorption_energy"
            ),
            "conflict_policy": "shared_actual_metal_hard_conflict",
        }
        for field, expected in expected_site_contract.items():
            if adsorption.get(field) != expected:
                raise ValueError(
                    f"adsorption.{field} must be {expected!r} for public "
                    "materialization"
                )
        if adsorption.get("metal_capacity") != 1:
            raise ValueError("adsorption.metal_capacity must be 1")
    _element(adsorption.get("anchor_element"), "adsorption.anchor_element")
    pattern = adsorption.get("anchor_neighbor_pattern")
    if not isinstance(pattern, list) or not pattern:
        raise ValueError("adsorption.anchor_neighbor_pattern must be a non-empty list")
    for index, symbol in enumerate(pattern):
        _element(symbol, f"adsorption.anchor_neighbor_pattern[{index}]")
    _positive_int(adsorption.get("released_protons"), "adsorption.released_protons")
    _element(
        adsorption.get("surface_parent_element"),
        "adsorption.surface_parent_element",
    )
    discovery = adsorption.get("site_discovery")
    probe_manifest_path = None
    if discovery is not None:
        discovery = _table(discovery, "adsorption.site_discovery")
        version = discovery.get("protocol_version")
        if not isinstance(version, str) or not version.strip():
            raise ValueError(
                "adsorption.site_discovery.protocol_version must be non-empty"
            )
        if discovery.get("surface_side") not in {"top", "bottom"}:
            raise ValueError(
                "adsorption.site_discovery.surface_side must be top or bottom"
            )
        metals = discovery.get("surface_metal_elements")
        if not isinstance(metals, list) or not metals:
            raise ValueError(
                "adsorption.site_discovery.surface_metal_elements must be non-empty"
            )
        for index, symbol in enumerate(metals):
            _element(symbol, f"adsorption.site_discovery.surface_metal_elements[{index}]")
        _positive_number(
            discovery.get("surface_metal_depth_A"),
            "adsorption.site_discovery.surface_metal_depth_A",
        )
        denticities = discovery.get("denticities")
        if (
            not isinstance(denticities, list)
            or not denticities
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 2
                for value in denticities
            )
            or len(set(denticities)) != len(denticities)
        ):
            raise ValueError(
                "adsorption.site_discovery.denticities must contain unique integers >= 2"
            )
        donor_labels = discovery.get("probe_donor_labels")
        if (
            not isinstance(donor_labels, list)
            or len(donor_labels) < max(denticities)
            or any(not isinstance(value, str) or not value.strip() for value in donor_labels)
        ):
            raise ValueError(
                "adsorption.site_discovery.probe_donor_labels must cover every denticity"
            )
        if discovery.get("equivalent_probe_donors") is not True:
            raise ValueError(
                "adsorption.site_discovery currently requires equivalent_probe_donors=true"
            )
        distance_window = discovery.get("metal_vertex_distance_window_A")
        if not isinstance(distance_window, list) or len(distance_window) != 2:
            raise ValueError(
                "adsorption.site_discovery.metal_vertex_distance_window_A "
                "must contain [minimum, maximum]"
            )
        minimum_distance = _positive_number(
            distance_window[0],
            "adsorption.site_discovery.metal_vertex_distance_window_A[0]",
        )
        maximum_distance = _positive_number(
            distance_window[1],
            "adsorption.site_discovery.metal_vertex_distance_window_A[1]",
        )
        if minimum_distance >= maximum_distance:
            raise ValueError(
                "adsorption.site_discovery metal distance minimum must be below maximum"
            )
        for field in ("maximum_vertex_normal_span_A", "symmetry_tolerance_A"):
            _positive_number(discovery.get(field), f"adsorption.site_discovery.{field}")
        if discovery.get("require_distinct_periodic_metal_vertices") is not True:
            raise ValueError(
                "adsorption.site_discovery requires distinct periodic metal vertices "
                "for bridge enumeration"
            )
        binding_modes = discovery.get(
            "binding_modes", ["bridging_distinct_metals"]
        )
        allowed_binding_modes = {
            "bridging_distinct_metals",
            "chelating_shared_metal",
        }
        if (
            not isinstance(binding_modes, list)
            or not binding_modes
            or any(value not in allowed_binding_modes for value in binding_modes)
            or len(set(binding_modes)) != len(binding_modes)
        ):
            raise ValueError(
                "adsorption.site_discovery.binding_modes must contain unique "
                f"values from {sorted(allowed_binding_modes)}"
            )
        if (
            "chelating_shared_metal" in binding_modes
            and (2 not in denticities or len(donor_labels) < 2)
        ):
            raise ValueError(
                "chelating_shared_metal requires denticity 2 and two probe donors"
            )
        probe = discovery.get("probe")
        if not isinstance(probe, str) or not probe.strip():
            raise ValueError("adsorption.site_discovery.probe must be non-empty")
        probe_manifest = discovery.get("probe_manifest")
        if not isinstance(probe_manifest, str) or not probe_manifest.strip():
            raise ValueError(
                "adsorption.site_discovery.probe_manifest must be a non-empty path"
            )
        probe_manifest_path = (path.parent / probe_manifest).resolve()
        if not probe_manifest_path.is_file():
            raise ValueError(
                f"{path}: probe manifest does not exist: {probe_manifest_path}"
            )
        if discovery.get("fully_deprotonated") is not True:
            raise ValueError(
                "adsorption.site_discovery requires fully_deprotonated=true"
            )
        preparation = _table(
            discovery.get("probe_preparation"),
            "adsorption.site_discovery.probe_preparation",
        )
        preparation_version = preparation.get("protocol_version")
        if not isinstance(preparation_version, str) or not preparation_version.strip():
            raise ValueError("probe_preparation.protocol_version must be non-empty")
        for field in (
            "minimum_probe_image_separation_A",
            "anchor_height_reference_A",
            "preferred_metal_oxygen_distance_A",
            "minimum_unmapped_donor_metal_distance_A",
            "minimum_probe_substrate_heavy_distance_A",
            "surface_oxygen_depth_A",
            "proton_parent_lateral_radius_A",
            "surface_oxygen_hydrogen_bond_A",
            "minimum_proton_proton_distance_A",
            "minimum_proton_probe_heavy_distance_A",
        ):
            _positive_number(preparation.get(field), f"probe_preparation.{field}")
        height_search = preparation.get("anchor_height_search_A")
        if not isinstance(height_search, list) or len(height_search) != 3:
            raise ValueError(
                "probe_preparation.anchor_height_search_A must contain "
                "[minimum, maximum, step]"
            )
        height_minimum, height_maximum, height_step = (
            _positive_number(value, f"probe_preparation.anchor_height_search_A[{index}]")
            for index, value in enumerate(height_search)
        )
        if height_minimum >= height_maximum or height_step > height_maximum - height_minimum:
            raise ValueError("probe_preparation anchor-height search range is invalid")
        if "chelating_shared_metal" in binding_modes:
            _positive_number(
                preparation.get("chelate_donor_midpoint_height_reference_A"),
                "probe_preparation.chelate_donor_midpoint_height_reference_A",
            )
            chelate_height_search = preparation.get(
                "chelate_donor_midpoint_height_search_A"
            )
            if (
                not isinstance(chelate_height_search, list)
                or len(chelate_height_search) != 3
            ):
                raise ValueError(
                    "probe_preparation.chelate_donor_midpoint_height_search_A "
                    "must contain [minimum, maximum, step]"
                )
            chelate_minimum, chelate_maximum, chelate_step = (
                _positive_number(
                    value,
                    "probe_preparation."
                    f"chelate_donor_midpoint_height_search_A[{index}]",
                )
                for index, value in enumerate(chelate_height_search)
            )
            if (
                chelate_minimum >= chelate_maximum
                or chelate_step > chelate_maximum - chelate_minimum
            ):
                raise ValueError(
                    "probe_preparation chelate midpoint-height search range is invalid"
                )
            azimuth_step = preparation.get("chelate_azimuth_step_degrees")
            if (
                not isinstance(azimuth_step, int)
                or isinstance(azimuth_step, bool)
                or azimuth_step < 1
                or azimuth_step > 180
                or 360 % azimuth_step
            ):
                raise ValueError(
                    "probe_preparation.chelate_azimuth_step_degrees must be a "
                    "positive integer divisor of 360"
                )
        oxygen_window = preparation.get("accepted_metal_oxygen_distance_A")
        if not isinstance(oxygen_window, list) or len(oxygen_window) != 2:
            raise ValueError(
                "probe_preparation.accepted_metal_oxygen_distance_A must contain "
                "[minimum, maximum]"
            )
        oxygen_minimum, oxygen_maximum = (
            _positive_number(
                value,
                f"probe_preparation.accepted_metal_oxygen_distance_A[{index}]",
            )
            for index, value in enumerate(oxygen_window)
        )
        if oxygen_minimum >= oxygen_maximum:
            raise ValueError("probe_preparation metal-O distance window is invalid")
        weight = preparation.get("methyl_outward_alignment_weight")
        if not isinstance(weight, int) or isinstance(weight, bool) or weight < 1:
            raise ValueError(
                "probe_preparation.methyl_outward_alignment_weight must be positive"
            )
        cosine = preparation.get("minimum_methyl_outward_cosine")
        if (
            not isinstance(cosine, (int, float))
            or isinstance(cosine, bool)
            or not 0 < float(cosine) <= 1
        ):
            raise ValueError(
                "probe_preparation.minimum_methyl_outward_cosine must be in (0, 1]"
            )
        relaxation = _table(
            discovery.get("probe_relaxation"),
            "adsorption.site_discovery.probe_relaxation",
        )
        if relaxation.get("protocol_version") != "adsorption_probe_relaxation_v1":
            raise ValueError("Unsupported adsorption probe-relaxation protocol")
        if relaxation.get("relax_cell") is not False:
            raise ValueError("Probe relaxation must keep the complete slab cell fixed")
        if relaxation.get("freeze_lower_substrate_layers") is not True:
            raise ValueError("Probe relaxation must freeze lower substrate layers")
        movable_layers = _positive_int(
            relaxation.get("movable_substrate_layers_from_surface"),
            "probe_relaxation.movable_substrate_layers_from_surface",
        )
        if movable_layers >= int(slab["source_layers"]):
            raise ValueError(
                "Probe relaxation must leave at least one lower substrate layer frozen"
            )
        if relaxation.get("layer_partition") != (
            "equal_stoichiometric_blocks_by_outward_normal"
        ):
            raise ValueError("Unsupported probe-relaxation layer partition policy")
        for field in (
            "probe_atoms_movable",
            "surface_protons_movable",
            "persist_atom_ids_before_relaxation",
            "reference_calculator_reuse",
            "retain_released_protons",
        ):
            if relaxation.get(field) is not True:
                raise ValueError(f"probe_relaxation.{field} must be true")
        if relaxation.get("optimizer") != "LBFGS":
            raise ValueError("Probe relaxation currently requires optimizer=LBFGS")
        for field in (
            "force_tolerance_eV_A",
            "maximum_step_A",
            "isolated_neutral_probe_vacuum_A",
        ):
            _positive_number(relaxation.get(field), f"probe_relaxation.{field}")
        _positive_int(relaxation.get("maximum_steps"), "probe_relaxation.maximum_steps")
        if relaxation.get("clean_slab_reference_policy") != (
            "relax_with_same_frozen_substrate_layers"
        ):
            raise ValueError("Unsupported clean-slab adsorption reference policy")
        if relaxation.get("neutral_probe_reference_policy") != (
            "relax_isolated_neutral_acid"
        ):
            raise ValueError("Unsupported neutral-probe adsorption reference policy")
        validation = _table(
            relaxation.get("validation"), "probe_relaxation.validation"
        )
        coordination_window = validation.get("final_metal_oxygen_distance_A")
        if not isinstance(coordination_window, list) or len(coordination_window) != 2:
            raise ValueError(
                "probe_relaxation.validation.final_metal_oxygen_distance_A must "
                "contain [minimum, maximum]"
            )
        coordination_minimum, coordination_maximum = (
            _positive_number(
                value,
                "probe_relaxation.validation.final_metal_oxygen_distance_A"
                f"[{index}]",
            )
            for index, value in enumerate(coordination_window)
        )
        if coordination_minimum >= coordination_maximum:
            raise ValueError("Final metal-O distance window is invalid")
        for field in (
            "require_final_denticity_in_catalog",
            "require_probe_connectivity",
            "require_substrate_formula_unchanged",
        ):
            if validation.get(field) is not True:
                raise ValueError(f"probe_relaxation.validation.{field} must be true")
        if validation.get("surface_proton_final_policy") != (
            "bound_to_substrate_oxygen"
        ):
            raise ValueError("Unsupported final surface-proton policy")
        for field in (
            "maximum_surface_oxygen_hydrogen_distance_A",
            "maximum_movable_substrate_displacement_A",
        ):
            _positive_number(
                validation.get(field), f"probe_relaxation.validation.{field}"
            )
        clustering = _table(
            discovery.get("site_prototype_clustering"),
            "adsorption.site_discovery.site_prototype_clustering",
        )
        if clustering.get("protocol_version") != (
            "adsorption_site_prototype_clustering_v1"
        ):
            raise ValueError("Unsupported adsorption Site Prototype clustering protocol")
        anchor_family = clustering.get("anchor_family")
        if not isinstance(anchor_family, str) or not anchor_family.strip():
            raise ValueError("site_prototype_clustering.anchor_family must be non-empty")
        if clustering.get("equivalent_probe_donors") is not True:
            raise ValueError(
                "site_prototype_clustering currently requires equivalent probe donors"
            )
        for field in (
            "maximum_contact_distance_rms_difference_A",
            "maximum_metal_geometry_rms_difference_A",
            "maximum_anchor_height_difference_A",
            "maximum_anchor_lateral_offset_difference_A",
        ):
            _positive_number(clustering.get(field), f"site_prototype_clustering.{field}")
        if clustering.get("representative_selection") != (
            "lowest_adsorption_energy_then_candidate_id"
        ):
            raise ValueError("Unsupported Site Prototype representative-selection policy")
        if clustering.get("energy_policy") != (
            "lowest_valid_surface_proton_placement_per_site_combination"
        ):
            raise ValueError("Unsupported Site Prototype energy policy")
        if clustering.get("doped_site_energy_policy") != "reuse_undoped_parent":
            raise ValueError("Unsupported doped Site Instance energy policy")

    resolved = dict(recipe)
    resolved["catalog_path"] = path
    resolved["source_path"] = source_path
    if transform is not None:
        resolved["transform_matrix"] = transform
    if probe_manifest_path is not None:
        resolved["probe_manifest_path"] = probe_manifest_path
    return resolved


def _load_structure_manifest(
    name: str, structure_directory: Path | None = None
) -> dict:
    """Load one maintained bulk-structure manifest by substrate key."""

    if not isinstance(name, str) or not name.strip():
        raise ValueError("substrate name must be a non-empty string")
    directory = Path(structure_directory or _default_structure_directory()).resolve()
    requested = _normalized_name(name)
    matches = []
    for path in sorted(directory.glob("*/manifest.json")):
        with path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("schema_version") != STRUCTURE_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"{path}: schema_version must be {STRUCTURE_MANIFEST_SCHEMA_VERSION}"
            )
        key = manifest.get("substrate_key")
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{path}: substrate_key must be a non-empty string")
        if requested == _normalized_name(key):
            manifest["manifest_path"] = path
            matches.append(manifest)
    if not matches:
        available = sorted(
            path.parent.name for path in directory.glob("*/manifest.json")
        )
        raise ValueError(
            f"Unknown bulk substrate {name!r}; available structure keys: "
            + (", ".join(available) or "none")
        )
    if len(matches) != 1:
        paths = ", ".join(str(item["manifest_path"]) for item in matches)
        raise ValueError(f"Ambiguous bulk substrate {name!r}; matches: {paths}")
    return matches[0]


def _validated_bulk_seed(manifest: dict) -> tuple[Path, object, dict]:
    manifest_path = Path(manifest["manifest_path"])
    seed = _table(manifest.get("bulk_seed"), "bulk_seed")
    if not isinstance(seed.get("status"), str) or not seed["status"].strip():
        raise ValueError(f"{manifest_path}: bulk_seed.status must be non-empty")
    path_value = seed.get("path")
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError(f"{manifest_path}: bulk_seed.path must be a non-empty path")
    seed_path = (manifest_path.parent / path_value).resolve()
    if not seed_path.is_file():
        raise ValueError(f"{manifest_path}: bulk seed does not exist: {seed_path}")
    expected_hash = seed.get("sha256")
    if not isinstance(expected_hash, str) or _sha256(seed_path) != expected_hash:
        raise ValueError(f"{manifest_path}: bulk seed SHA-256 mismatch")

    atoms = read(seed_path)
    if not np.all(np.asarray(atoms.pbc, dtype=bool)):
        raise ValueError(f"{manifest_path}: bulk seed must be periodic in all axes")
    formula = _formula(atoms.get_chemical_symbols())
    expected_formula = seed.get("formula")
    if formula != dict(sorted((expected_formula or {}).items())):
        raise ValueError(
            f"{manifest_path}: bulk seed formula is {formula}; expected {expected_formula}"
        )
    if len(atoms) != seed.get("atom_count"):
        raise ValueError(
            f"{manifest_path}: bulk seed has {len(atoms)} atoms; "
            f"expected {seed.get('atom_count')}"
        )
    return seed_path, atoms, seed


def _validated_registered_bulk_parent(manifest: dict) -> dict | None:
    parent = manifest.get("bulk_parent")
    if parent is None:
        return None
    manifest_path = Path(manifest["manifest_path"])
    parent = _table(parent, "bulk_parent")
    if parent.get("status") != "accepted":
        raise ValueError(
            f"{manifest_path}: unsupported registered bulk_parent status "
            f"{parent.get('status')!r}"
        )
    path_value = parent.get("path")
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError(f"{manifest_path}: bulk_parent.path must be non-empty")
    path = (manifest_path.parent / path_value).resolve()
    if not path.is_file() or _sha256(path) != parent.get("sha256"):
        raise ValueError(f"{manifest_path}: registered Bulk Parent is missing or changed")
    atoms = read(path)
    formula = _formula(atoms.get_chemical_symbols())
    if formula != dict(sorted((parent.get("formula") or {}).items())):
        raise ValueError(f"{manifest_path}: registered Bulk Parent formula changed")
    if len(atoms) != parent.get("atom_count"):
        raise ValueError(f"{manifest_path}: registered Bulk Parent atom count changed")
    records = parent.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{manifest_path}: bulk_parent.records must be non-empty")
    records_by_role = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError(f"{manifest_path}: invalid Bulk Parent record")
        role = record.get("role")
        if not isinstance(role, str) or role in records_by_role:
            raise ValueError(f"{manifest_path}: invalid or duplicate Bulk Parent role")
        record_path = (manifest_path.parent / record["path"]).resolve()
        if not record_path.is_file() or _sha256(record_path) != record.get("sha256"):
            raise ValueError(
                f"{manifest_path}: Bulk Parent record is missing or changed: "
                f"{record_path}"
            )
        records_by_role[role] = record_path
    required_roles = {"acceptance_report", "experimental_position_check"}
    missing_roles = sorted(required_roles - records_by_role.keys())
    if missing_roles:
        raise ValueError(
            f"{manifest_path}: Bulk Parent is missing required records: "
            f"{', '.join(missing_roles)}"
        )
    acceptance_report = json.loads(
        records_by_role["acceptance_report"].read_text(encoding="utf-8")
    )
    if acceptance_report.get("status") != "passed":
        raise ValueError(f"{manifest_path}: Bulk Parent acceptance report did not pass")
    position_report = json.loads(
        records_by_role["experimental_position_check"].read_text(encoding="utf-8")
    )
    relaxation_contract = manifest.get("bulk_relaxation_contract") or {}
    reference = relaxation_contract.get("reference") or {}
    position_acceptance = relaxation_contract.get("acceptance") or {}
    position_checks = position_report.get("checks") or {}
    rms_check = position_checks.get("rms_displacement") or {}
    maximum_check = position_checks.get("maximum_displacement") or {}
    expected_rms_limit = position_acceptance.get(
        "maximum_experimental_position_rmsd_A"
    )
    expected_maximum_limit = position_acceptance.get(
        "maximum_experimental_position_displacement_A"
    )
    if (
        position_report.get("status") != "passed"
        or (position_report.get("bulk_parent") or {}).get("sha256") != parent["sha256"]
        or (position_report.get("experimental_reference") or {}).get("sha256")
        != reference.get("structure_sha256")
        or rms_check.get("passed") is not True
        or maximum_check.get("passed") is not True
        or rms_check.get("maximum_A") != expected_rms_limit
        or maximum_check.get("maximum_A") != expected_maximum_limit
        or rms_check.get("observed_A") != parent.get("experimental_position_rmsd_A")
        or maximum_check.get("observed_A")
        != parent.get("experimental_position_maximum_displacement_A")
    ):
        raise ValueError(
            f"{manifest_path}: experimental atomic-position check is invalid or stale"
        )
    return {
        "path": str(path),
        "sha256": parent["sha256"],
        "status": parent["status"],
        "protocol_version": parent["protocol_version"],
        "formula": formula,
        "atom_count": len(atoms),
        "cell_A": parent["cell_A"],
        "space_group": parent["space_group"],
        "run_fingerprint": parent["run_fingerprint"],
        "potential": dict(parent["potential"]),
        "potential_energy_eV": float(parent["potential_energy_eV"]),
    }


def _relaxed_surface_evidence_is_valid(
    surface: dict,
    records_by_role: dict[str, Path],
    expected_surface_sha256: str,
) -> bool:
    required = {
        "surface_promotion_plan",
        "surface_promotion_report",
        "termination_screen_plan",
        "termination_selection_report",
        "termination_screen_run_manifest",
        "surface_relaxation_plan",
        "surface_relaxation_run_manifest",
        "surface_convergence_report",
        "target_relaxation_report",
        "target_optimizer_log",
        "target_progress",
        "bulk_reference",
    }
    if not required.issubset(records_by_role):
        return False
    promotion_plan = json.loads(
        records_by_role["surface_promotion_plan"].read_text(encoding="utf-8")
    )
    promotion_report = json.loads(
        records_by_role["surface_promotion_report"].read_text(encoding="utf-8")
    )
    selection = json.loads(
        records_by_role["termination_selection_report"].read_text(encoding="utf-8")
    )
    screen_run = json.loads(
        records_by_role["termination_screen_run_manifest"].read_text(encoding="utf-8")
    )
    relaxation_run = json.loads(
        records_by_role["surface_relaxation_run_manifest"].read_text(encoding="utf-8")
    )
    convergence = json.loads(
        records_by_role["surface_convergence_report"].read_text(encoding="utf-8")
    )
    target_report = json.loads(
        records_by_role["target_relaxation_report"].read_text(encoding="utf-8")
    )
    target = promotion_plan.get("target_surface") or {}
    return bool(
        promotion_plan.get("mode") == "surface_promotion_plan"
        and promotion_plan.get("fingerprint") == surface.get("promotion_fingerprint")
        and promotion_report.get("status") == "passed_promotion"
        and promotion_report.get("plan_fingerprint")
        == surface.get("promotion_fingerprint")
        and (promotion_report.get("registered_surface") or {}).get("sha256")
        == expected_surface_sha256
        and target.get("source_sha256") == expected_surface_sha256
        and target.get("oriented_unit_repeats") == surface.get("source_layers")
        and selection.get("status") == "passed_selection"
        and selection.get("selected_candidate_id")
        == surface.get("termination_candidate_id")
        and screen_run.get("status") == "passed_selection"
        and screen_run.get("selected_candidate_id")
        == surface.get("termination_candidate_id")
        and screen_run.get("fingerprint") == surface.get("screen_run_fingerprint")
        and relaxation_run.get("status") == "passed"
        and convergence.get("status") == "passed"
        and (convergence.get("checks") or {})
        .get("two_thickest_surface_energy", {})
        .get("passed")
        is True
        and target_report.get("status") == "passed"
        and target_report.get("oriented_unit_repeats") == surface.get("source_layers")
        and all(
            check.get("passed") is True
            for check in (target_report.get("checks") or {}).values()
        )
        and surface.get("parent_bulk_sha256")
        == (promotion_plan.get("bulk_parent") or {}).get("sha256")
    )


def _validated_registered_surface_precuts(manifest: dict) -> list[dict]:
    manifest_path = Path(manifest["manifest_path"])
    parent = manifest.get("bulk_parent") or {}
    accepted = []
    for precut in manifest.get("precut_slabs", []):
        if not isinstance(precut, dict):
            raise ValueError(f"{manifest_path}: invalid precut_slabs record")
        if precut.get("status") not in {
            "accepted_bulk_derived_unrelaxed_precut",
            "accepted_relaxed_maintained_surface",
        }:
            continue
        path_value = precut.get("path")
        if not isinstance(path_value, str) or not path_value.strip():
            raise ValueError(f"{manifest_path}: accepted Surface Precut path is required")
        path = (manifest_path.parent / path_value).resolve()
        if not path.is_file() or _sha256(path) != precut.get("sha256"):
            raise ValueError(
                f"{manifest_path}: accepted Surface Precut is missing or changed: {path}"
            )
        atoms = read(path)
        if _formula(atoms.get_chemical_symbols()) != dict(
            sorted((precut.get("formula") or {}).items())
        ) or len(atoms) != precut.get("atom_count"):
            raise ValueError(f"{manifest_path}: accepted Surface Precut composition changed")
        if precut.get("parent_bulk_sha256") != parent.get("sha256"):
            raise ValueError(f"{manifest_path}: accepted Surface Precut parent is stale")
        records = precut.get("records")
        if not isinstance(records, list):
            raise ValueError(f"{manifest_path}: Surface Precut records must be a list")
        records_by_role = {}
        for record in records:
            if not isinstance(record, dict) or not isinstance(record.get("path"), str):
                raise ValueError(f"{manifest_path}: invalid Surface Precut record")
            role = record.get("role")
            if not isinstance(role, str) or role in records_by_role:
                raise ValueError(
                    f"{manifest_path}: invalid or duplicate Surface Precut record role"
                )
            record_path = (manifest_path.parent / record["path"]).resolve()
            if not record_path.is_file() or _sha256(record_path) != record.get("sha256"):
                raise ValueError(
                    f"{manifest_path}: Surface Precut record is missing or changed: "
                    f"{record_path}"
                )
            records_by_role[role] = record_path
        if precut["status"] == "accepted_bulk_derived_unrelaxed_precut":
            required = {"surface_precut_plan", "selection_report", "run_manifest"}
            if not required.issubset(records_by_role):
                raise ValueError(
                    f"{manifest_path}: Surface Precut evidence chain is incomplete"
                )
            selection = json.loads(
                records_by_role["selection_report"].read_text(encoding="utf-8")
            )
            run = json.loads(
                records_by_role["run_manifest"].read_text(encoding="utf-8")
            )
            selected_artifacts = [
                artifact
                for artifact in run.get("artifacts", [])
                if artifact.get("role") == "selected_surface_precut_candidate"
            ]
            valid = (
                selection.get("status") == "passed_selection"
                and run.get("status") == "passed_selection"
                and run.get("fingerprint") == precut.get("run_fingerprint")
                and run.get("bulk_parent_sha256")
                == precut.get("parent_bulk_sha256")
                and len(selected_artifacts) == 1
                and selected_artifacts[0].get("sha256") == precut.get("sha256")
            )
        else:
            valid = _relaxed_surface_evidence_is_valid(
                precut,
                records_by_role,
                precut["sha256"],
            )
        if not valid:
            raise ValueError(f"{manifest_path}: Surface Precut evidence is invalid or stale")
        accepted.append(
            {
                "path": str(path),
                "sha256": precut["sha256"],
                "status": precut["status"],
                "miller": list(precut["miller"]),
                "termination": precut["termination"],
                "run_fingerprint": precut.get("run_fingerprint")
                or precut.get("screen_run_fingerprint"),
                "promotion_fingerprint": precut.get("promotion_fingerprint"),
            }
        )
    return accepted


def _validated_bulk_relaxation_contract(manifest: dict) -> dict:
    manifest_path = Path(manifest["manifest_path"])
    contract = _table(
        manifest.get("bulk_relaxation_contract"), "bulk_relaxation_contract"
    )
    version = contract.get("protocol_version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError(f"{manifest_path}: protocol_version must be non-empty")
    if contract.get("cell_constraint") not in {"cubic", "tetragonal"}:
        raise ValueError(
            f"{manifest_path}: cell_constraint must be cubic or tetragonal"
        )
    for field in ("relax_atomic_positions", "relax_cell"):
        if contract.get(field) is not True:
            raise ValueError(f"{manifest_path}: {field} must be true")
    _positive_number(
        contract.get("target_pressure_GPa"),
        "target_pressure_GPa",
        allow_zero=True,
    )
    for field in (
        "force_tolerance_eV_A",
        "stress_tolerance_GPa",
        "maximum_step_A",
        "symmetry_tolerance_A",
        "maximum_steps",
    ):
        if field == "maximum_steps":
            _positive_int(contract.get(field), field)
        else:
            _positive_number(contract.get(field), field)

    reference = _table(contract.get("reference"), "bulk_relaxation_contract.reference")
    reference_cell = _table(reference.get("cell_A"), "reference.cell_A")
    for axis in ("a", "b", "c"):
        _positive_number(reference_cell.get(axis), f"reference.cell_A.{axis}")
    if not isinstance(reference.get("provenance"), str) or not reference[
        "provenance"
    ].strip():
        raise ValueError(f"{manifest_path}: reference.provenance must be non-empty")
    reference_structure = reference.get("structure")
    if not isinstance(reference_structure, str) or not reference_structure.strip():
        raise ValueError(f"{manifest_path}: reference.structure must be non-empty")
    reference_path = (manifest_path.parent / reference_structure).resolve()
    if not reference_path.is_file():
        raise ValueError(
            f"{manifest_path}: experimental reference structure is missing: "
            f"{reference_path}"
        )
    if _sha256(reference_path) != reference.get("structure_sha256"):
        raise ValueError(
            f"{manifest_path}: experimental reference structure SHA-256 mismatch"
        )
    reference_atoms = read(reference_path)
    if not np.all(np.asarray(reference_atoms.pbc, dtype=bool)):
        raise ValueError(
            f"{manifest_path}: experimental reference must be periodic in all axes"
        )
    reference["resolved_structure_path"] = str(reference_path)

    acceptance = _table(contract.get("acceptance"), "acceptance")
    for field in (
        "maximum_relative_length_deviation",
        "maximum_angle_deviation_deg",
        "maximum_experimental_position_rmsd_A",
        "maximum_experimental_position_displacement_A",
    ):
        _positive_number(acceptance.get(field), f"acceptance.{field}")
    if acceptance.get("require_formula_unchanged") is not True:
        raise ValueError("acceptance.require_formula_unchanged must be true")
    if acceptance.get("require_atom_count_unchanged") is not True:
        raise ValueError("acceptance.require_atom_count_unchanged must be true")
    if acceptance.get("require_crystal_system_preserved") is not True:
        raise ValueError("acceptance.require_crystal_system_preserved must be true")
    _positive_int(
        acceptance.get("expected_space_group_number"),
        "acceptance.expected_space_group_number",
    )
    return contract


def _standardized_spglib_cell(atoms, symprec: float):
    try:
        import spglib
    except ImportError as exc:
        raise RuntimeError(
            "Experimental atomic-position comparison requires spglib"
        ) from exc
    standardized = spglib.standardize_cell(
        (
            np.asarray(atoms.cell, dtype=float),
            np.asarray(atoms.get_scaled_positions(wrap=True), dtype=float),
            np.asarray(atoms.numbers, dtype=int),
        ),
        to_primitive=False,
        no_idealize=False,
        symprec=symprec,
    )
    if standardized is None:
        raise ValueError(
            f"Cannot standardize structure for position comparison at {symprec:g} A"
        )
    lattice, positions, numbers = standardized
    return (
        np.asarray(lattice, dtype=float),
        np.asarray(positions, dtype=float) % 1.0,
        np.asarray(numbers, dtype=int),
    )


def compare_experimental_atomic_positions(
    candidate,
    reference,
    *,
    symprec: float,
) -> dict:
    """Compare internal coordinates after symmetry standardization and MIC matching."""

    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:
        raise RuntimeError(
            "Experimental atomic-position comparison requires SciPy"
        ) from exc

    reference_lattice, reference_positions, reference_numbers = (
        _standardized_spglib_cell(reference, symprec)
    )
    _, candidate_positions, candidate_numbers = _standardized_spglib_cell(
        candidate, symprec
    )
    if _formula(chemical_symbols[number] for number in candidate_numbers) != _formula(
        chemical_symbols[number] for number in reference_numbers
    ):
        raise ValueError(
            "Candidate and experimental reference formulas differ after standardization"
        )
    if len(candidate_numbers) != len(reference_numbers):
        raise ValueError(
            "Candidate and experimental reference atom counts differ after standardization"
        )

    element_numbers = sorted(set(int(value) for value in reference_numbers))
    anchor_number = min(
        element_numbers,
        key=lambda number: (int(np.sum(reference_numbers == number)), number),
    )
    anchor_candidate = candidate_positions[candidate_numbers == anchor_number]
    anchor_reference = reference_positions[reference_numbers == anchor_number]
    shifts = {
        tuple(np.round((reference_site - candidate_site + 0.5) % 1.0 - 0.5, 12))
        for candidate_site in anchor_candidate
        for reference_site in anchor_reference
    }

    best = None
    for shift_tuple in sorted(shifts):
        shift = np.asarray(shift_tuple, dtype=float)
        shifted = (candidate_positions + shift) % 1.0
        mappings = []
        squared_sum = 0.0
        for number in element_numbers:
            candidate_indices = np.flatnonzero(candidate_numbers == number)
            reference_indices = np.flatnonzero(reference_numbers == number)
            differences = (
                shifted[candidate_indices, None, :]
                - reference_positions[None, reference_indices, :]
            )
            differences -= np.round(differences)
            vectors = differences @ reference_lattice
            distances = np.linalg.norm(vectors, axis=2)
            rows, columns = linear_sum_assignment(distances)
            for row, column in zip(rows, columns):
                vector = vectors[row, column]
                distance = float(distances[row, column])
                squared_sum += distance * distance
                mappings.append(
                    {
                        "element": chemical_symbols[number],
                        "candidate_standardized_index_0based": int(
                            candidate_indices[row]
                        ),
                        "reference_standardized_index_0based": int(
                            reference_indices[column]
                        ),
                        "candidate_aligned_fractional": shifted[
                            candidate_indices[row]
                        ].tolist(),
                        "reference_fractional": reference_positions[
                            reference_indices[column]
                        ].tolist(),
                        "fractional_difference_minimum_image": differences[
                            row, column
                        ].tolist(),
                        "displacement_vector_A": vector.tolist(),
                        "displacement_A": distance,
                    }
                )
        score = (squared_sum, shift_tuple)
        if best is None or score < best[0]:
            best = (score, shift, mappings)
    if best is None:
        raise ValueError("No valid element-preserving atomic-position mapping found")

    _, origin_shift, mappings = best
    try:
        import spglib

        dataset = spglib.get_symmetry_dataset(
            (reference_lattice, reference_positions, reference_numbers),
            symprec=symprec,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Experimental atomic-position comparison requires spglib"
        ) from exc
    if dataset is None:
        raise ValueError("Cannot resolve reference Wyckoff sites")
    wyckoffs = list(
        dataset.wyckoffs if hasattr(dataset, "wyckoffs") else dataset["wyckoffs"]
    )
    grouped: dict[str, list[dict]] = {}
    by_element: dict[str, list[float]] = {}
    for mapping in mappings:
        element = mapping["element"]
        reference_index = mapping["reference_standardized_index_0based"]
        group = f"{element}:{wyckoffs[reference_index]}"
        grouped.setdefault(group, []).append(mapping)
        by_element.setdefault(element, []).append(mapping["displacement_A"])

    def summarize(values: list[float]) -> dict:
        array = np.asarray(values, dtype=float)
        return {
            "count": len(values),
            "mean_displacement_A": float(np.mean(array)),
            "rms_displacement_A": float(np.sqrt(np.mean(array * array))),
            "maximum_displacement_A": float(np.max(array)),
        }

    all_displacements = [mapping["displacement_A"] for mapping in mappings]
    summary = summarize(all_displacements)

    def summarize_group(group_mappings: list[dict]) -> dict:
        representative = min(
            group_mappings,
            key=lambda item: item["reference_standardized_index_0based"],
        )
        return {
            **summarize(
                [mapping["displacement_A"] for mapping in group_mappings]
            ),
            "representative": {
                "experimental_fractional": representative["reference_fractional"],
                "candidate_fractional": representative[
                    "candidate_aligned_fractional"
                ],
                "fractional_difference_minimum_image": representative[
                    "fractional_difference_minimum_image"
                ],
            },
        }

    return {
        "method": (
            "spglib_standardized_conventional_cells_then_elementwise_"
            "hungarian_minimum_image_matching"
        ),
        "coordinate_basis": (
            "Fractional internal-coordinate differences converted with the "
            "experimental reference lattice; lattice-size error is excluded."
        ),
        "symprec_A": symprec,
        "origin_shift_fractional": origin_shift.tolist(),
        "overall_mean_displacement_A": summary["mean_displacement_A"],
        "overall_rms_displacement_A": summary["rms_displacement_A"],
        "overall_maximum_displacement_A": summary["maximum_displacement_A"],
        "by_element": {
            key: summarize(values) for key, values in sorted(by_element.items())
        },
        "by_wyckoff_group": {
            key: summarize_group(group_mappings)
            for key, group_mappings in sorted(grouped.items())
        },
        "atom_mapping": sorted(
            mappings,
            key=lambda item: item["candidate_standardized_index_0based"],
        ),
    }


def resolve_bulk_relaxation_plan(
    substrate_name: str,
    model_path: Path,
    declared_model_elements: list[str],
    structure_directory: Path | None = None,
) -> dict:
    """Resolve a read-only, potential-consistent Bulk Parent relaxation plan."""

    manifest = _load_structure_manifest(substrate_name, structure_directory)
    seed_path, atoms, seed = _validated_bulk_seed(manifest)
    registered_parent = _validated_registered_bulk_parent(manifest)
    contract = _validated_bulk_relaxation_contract(manifest)

    model_path = Path(model_path).expanduser().resolve()
    if not model_path.is_file():
        raise ValueError(f"Model file does not exist: {model_path}")
    if not declared_model_elements:
        raise ValueError("At least one declared model element is required")
    model_elements = sorted(
        {_element(value, "declared_model_elements") for value in declared_model_elements}
    )
    required_elements = sorted(set(atoms.get_chemical_symbols()))
    missing_elements = sorted(set(required_elements) - set(model_elements))
    if missing_elements:
        raise ValueError(
            "Declared model elements do not cover the bulk seed: "
            + ", ".join(missing_elements)
        )

    lengths = np.asarray(atoms.cell.lengths(), dtype=float)
    angles = np.asarray(atoms.cell.angles(), dtype=float)
    reference_cell = contract["reference"]["cell_A"]
    reference_lengths = np.asarray(
        [reference_cell[axis] for axis in ("a", "b", "c")], dtype=float
    )
    relative_deviations = np.abs(lengths - reference_lengths) / reference_lengths
    cell_gradient_scale = (
        float(contract["stress_tolerance_GPa"])
        / GPA_PER_EV_A3
        * float(atoms.get_volume())
        / float(contract["force_tolerance_eV_A"])
    )
    maximum_angle_deviation = contract["acceptance"][
        "maximum_angle_deviation_deg"
    ]
    if np.max(np.abs(angles - 90.0)) > maximum_angle_deviation:
        raise ValueError(
            "Bulk seed cell angles violate the declared crystal-system constraint"
        )
    equality_tolerance = 1.0e-6
    cell_constraint = contract["cell_constraint"]
    if (
        cell_constraint == "cubic"
        and np.ptp(lengths) / np.mean(lengths) > equality_tolerance
    ):
        raise ValueError("Cubic Bulk Seed must have equal a, b, and c lengths")
    if cell_constraint == "tetragonal" and abs(lengths[0] - lengths[1]) / np.mean(
        lengths[:2]
    ) > equality_tolerance:
        raise ValueError("Tetragonal Bulk Seed must have equal a and b lengths")

    return {
        "schema_version": 1,
        "mode": "bulk_relaxation_plan",
        "substrate": {
            "key": manifest["substrate_key"],
            "host_material": manifest["host_material"],
            "structure_manifest": {
                "path": str(manifest["manifest_path"]),
                "sha256": _sha256(Path(manifest["manifest_path"])),
            },
        },
        "bulk_seed": {
            "path": str(seed_path),
            "sha256": _sha256(seed_path),
            "status": seed["status"],
            "formula": _formula(atoms.get_chemical_symbols()),
            "atom_count": len(atoms),
            "pbc": [bool(value) for value in atoms.pbc],
            "cell_A": {
                "lengths": lengths.tolist(),
                "angles_deg": angles.tolist(),
                "volume_A3": float(atoms.get_volume()),
            },
            "cell_constraint_check": "passed",
        },
        "potential": {
            "model_path": str(model_path),
            "model_sha256": _sha256(model_path),
            "required_elements": required_elements,
            "declared_model_elements": model_elements,
            "coverage": "passed",
            "coverage_basis": "maintainer_declaration_not_model_introspection",
        },
        "relaxation_contract": contract,
        "optimizer_contract": {
            "optimizer": "LBFGS",
            "cell_filter": "FrechetCellFilter",
            "symmetry_constraint": "FixSymmetry",
            "exp_cell_factor": cell_gradient_scale,
            "derivation": (
                "stress_tolerance_eV_A3 * initial_volume_A3 / "
                "force_tolerance_eV_A"
            ),
            "purpose": (
                "Map the declared stress threshold onto the same optimizer fmax "
                "threshold used for atomic forces."
            ),
        },
        "seed_reference_comparison": {
            "reference_lengths_A": reference_lengths.tolist(),
            "relative_length_deviations": relative_deviations.tolist(),
            "within_acceptance_envelope": bool(
                np.all(
                    relative_deviations
                    <= contract["acceptance"]["maximum_relative_length_deviation"]
                )
            ),
            "note": (
                "This comparison characterizes the unrelaxed seed. Acceptance must be "
                "recomputed from the future relaxed candidate."
            ),
        },
        "planned_outputs": {
            "candidate_structure_role": "bulk_parent_candidate",
            "required_records": [
                "relaxed_structure",
                "optimizer_log",
                "final_force_and_stress_metrics",
                "acceptance_report",
                "provenance_manifest",
            ],
            "registration_policy": (
                "Promote a candidate to Bulk Parent only after every acceptance check "
                "passes; never overwrite the Bulk Seed."
            ),
        },
        "bulk_parent_state": (
            "not_registered" if registered_parent is None else "registered"
        ),
        "registered_bulk_parent": registered_parent,
        "execution_contract": {
            "executable": True,
            "read_only": True,
            "execution_command": "bulk-relaxation-run",
            "readiness_issues": [],
        },
    }


def _space_group(atoms, symprec: float) -> dict:
    try:
        import spglib
    except ImportError as exc:
        raise RuntimeError(
            "Bulk relaxation requires spglib to preserve and validate symmetry"
        ) from exc
    cell = (
        np.asarray(atoms.cell, dtype=float),
        np.asarray(atoms.get_scaled_positions(wrap=True), dtype=float),
        np.asarray(atoms.numbers, dtype=int),
    )
    dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)
    if dataset is None:
        raise ValueError(f"Cannot resolve space group at symprec={symprec:g} A")
    number = int(
        dataset.number if hasattr(dataset, "number") else dataset["number"]
    )
    symbol = str(
        dataset.international
        if hasattr(dataset, "international")
        else dataset["international"]
    )
    return {"number": number, "symbol": symbol, "symprec_A": symprec}


def _bulk_metrics(atoms, reference_lengths: np.ndarray) -> dict:
    forces = np.asarray(atoms.get_forces(), dtype=float)
    stress = np.asarray(atoms.get_stress(voigt=False), dtype=float)
    lengths = np.asarray(atoms.cell.lengths(), dtype=float)
    angles = np.asarray(atoms.cell.angles(), dtype=float)
    return {
        "potential_energy_eV": float(atoms.get_potential_energy()),
        "maximum_force_eV_A": float(np.max(np.linalg.norm(forces, axis=1))),
        "maximum_absolute_stress_GPa": float(
            np.max(np.abs(stress)) * GPA_PER_EV_A3
        ),
        "stress_GPa": (stress * GPA_PER_EV_A3).tolist(),
        "cell_A": {
            "lengths": lengths.tolist(),
            "angles_deg": angles.tolist(),
            "volume_A3": float(atoms.get_volume()),
        },
        "relative_length_deviations": (
            np.abs(lengths - reference_lengths) / reference_lengths
        ).tolist(),
    }


def evaluate_bulk_parent_candidate(
    plan: dict,
    atoms,
    *,
    optimizer_converged: bool,
    optimizer_steps: int,
) -> dict:
    """Evaluate a relaxed candidate against every Bulk Parent acceptance gate."""

    if plan.get("mode") != "bulk_relaxation_plan":
        raise ValueError("Candidate acceptance requires a bulk relaxation plan")
    contract = plan["relaxation_contract"]
    acceptance = contract["acceptance"]
    reference_cell = contract["reference"]["cell_A"]
    reference_lengths = np.asarray(
        [reference_cell[axis] for axis in ("a", "b", "c")], dtype=float
    )
    metrics = _bulk_metrics(atoms, reference_lengths)
    symmetry = _space_group(atoms, float(contract["symmetry_tolerance_A"]))
    experimental_reference = read(contract["reference"]["resolved_structure_path"])
    position_comparison = compare_experimental_atomic_positions(
        atoms,
        experimental_reference,
        symprec=float(contract["symmetry_tolerance_A"]),
    )
    formula = _formula(atoms.get_chemical_symbols())
    lengths = np.asarray(metrics["cell_A"]["lengths"], dtype=float)
    angles = np.asarray(metrics["cell_A"]["angles_deg"], dtype=float)
    cell_constraint = contract["cell_constraint"]
    equality_deviation = (
        float(np.ptp(lengths) / np.mean(lengths))
        if cell_constraint == "cubic"
        else float(abs(lengths[0] - lengths[1]) / np.mean(lengths[:2]))
    )

    checks = {
        "optimizer_converged": {
            "passed": bool(optimizer_converged),
            "observed_steps": int(optimizer_steps),
            "maximum_steps": int(contract["maximum_steps"]),
        },
        "force": {
            "passed": metrics["maximum_force_eV_A"]
            <= float(contract["force_tolerance_eV_A"]),
            "observed_eV_A": metrics["maximum_force_eV_A"],
            "maximum_eV_A": float(contract["force_tolerance_eV_A"]),
        },
        "stress": {
            "passed": metrics["maximum_absolute_stress_GPa"]
            <= float(contract["stress_tolerance_GPa"]),
            "observed_GPa": metrics["maximum_absolute_stress_GPa"],
            "maximum_GPa": float(contract["stress_tolerance_GPa"]),
        },
        "reference_lattice": {
            "passed": max(metrics["relative_length_deviations"])
            <= float(acceptance["maximum_relative_length_deviation"]),
            "observed_maximum_relative_deviation": max(
                metrics["relative_length_deviations"]
            ),
            "maximum_relative_deviation": float(
                acceptance["maximum_relative_length_deviation"]
            ),
        },
        "cell_angles": {
            "passed": float(np.max(np.abs(angles - 90.0)))
            <= float(acceptance["maximum_angle_deviation_deg"]),
            "observed_maximum_deviation_deg": float(
                np.max(np.abs(angles - 90.0))
            ),
            "maximum_deviation_deg": float(
                acceptance["maximum_angle_deviation_deg"]
            ),
        },
        "cell_constraint": {
            "passed": equality_deviation <= 1.0e-6,
            "kind": cell_constraint,
            "observed_relative_equality_deviation": equality_deviation,
            "maximum_relative_equality_deviation": 1.0e-6,
        },
        "formula": {
            "passed": formula == plan["bulk_seed"]["formula"],
            "observed": formula,
            "expected": plan["bulk_seed"]["formula"],
        },
        "atom_count": {
            "passed": len(atoms) == plan["bulk_seed"]["atom_count"],
            "observed": len(atoms),
            "expected": plan["bulk_seed"]["atom_count"],
        },
        "space_group": {
            "passed": symmetry["number"]
            == int(acceptance["expected_space_group_number"]),
            "observed": symmetry,
            "expected_number": int(acceptance["expected_space_group_number"]),
        },
        "experimental_atomic_positions": {
            "passed": (
                position_comparison["overall_rms_displacement_A"]
                <= float(acceptance["maximum_experimental_position_rmsd_A"])
                and position_comparison["overall_maximum_displacement_A"]
                <= float(
                    acceptance["maximum_experimental_position_displacement_A"]
                )
            ),
            "observed_rms_displacement_A": position_comparison[
                "overall_rms_displacement_A"
            ],
            "maximum_rms_displacement_A": float(
                acceptance["maximum_experimental_position_rmsd_A"]
            ),
            "observed_maximum_displacement_A": position_comparison[
                "overall_maximum_displacement_A"
            ],
            "maximum_displacement_A": float(
                acceptance["maximum_experimental_position_displacement_A"]
            ),
        },
    }
    return {
        "schema_version": 1,
        "status": (
            "passed" if all(item["passed"] for item in checks.values()) else "failed"
        ),
        "checks": checks,
        "metrics": metrics,
        "symmetry": symmetry,
        "experimental_atomic_positions": position_comparison,
    }


def resolve_bulk_parent_position_check(
    substrate_name: str,
    structure_directory: Path | None = None,
) -> dict:
    """Revalidate a registered Bulk Parent against experimental internal positions."""

    manifest = _load_structure_manifest(substrate_name, structure_directory)
    parent = _validated_registered_bulk_parent(manifest)
    if parent is None:
        raise ValueError(f"No registered Bulk Parent for {substrate_name!r}")
    contract = _validated_bulk_relaxation_contract(manifest)
    candidate = read(parent["path"])
    reference = read(contract["reference"]["resolved_structure_path"])
    comparison = compare_experimental_atomic_positions(
        candidate,
        reference,
        symprec=float(contract["symmetry_tolerance_A"]),
    )
    acceptance = contract["acceptance"]
    rms_passed = comparison["overall_rms_displacement_A"] <= float(
        acceptance["maximum_experimental_position_rmsd_A"]
    )
    maximum_passed = comparison["overall_maximum_displacement_A"] <= float(
        acceptance["maximum_experimental_position_displacement_A"]
    )
    parent_report = dict(parent)
    parent_report["path"] = manifest["bulk_parent"]["path"]
    return {
        "schema_version": 1,
        "mode": "bulk_parent_position_check",
        "status": "passed" if rms_passed and maximum_passed else "failed",
        "substrate": manifest["substrate_key"],
        "bulk_parent": parent_report,
        "experimental_reference": {
            "path": contract["reference"]["structure"],
            "sha256": contract["reference"]["structure_sha256"],
            "citation": contract["reference"]["citation"],
            "doi": contract["reference"]["doi"],
        },
        "checks": {
            "rms_displacement": {
                "passed": rms_passed,
                "observed_A": comparison["overall_rms_displacement_A"],
                "maximum_A": float(
                    acceptance["maximum_experimental_position_rmsd_A"]
                ),
            },
            "maximum_displacement": {
                "passed": maximum_passed,
                "observed_A": comparison["overall_maximum_displacement_A"],
                "maximum_A": float(
                    acceptance["maximum_experimental_position_displacement_A"]
                ),
            },
        },
        "comparison": comparison,
    }


def _validated_surface_precut_contract(manifest: dict) -> dict:
    manifest_path = Path(manifest["manifest_path"])
    contract = _table(
        manifest.get("surface_precut_contract"), "surface_precut_contract"
    )
    version = contract.get("protocol_version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError(f"{manifest_path}: surface precut protocol_version is required")
    miller = _integer_vector(contract.get("miller"), "surface_precut_contract.miller")
    if not any(miller):
        raise ValueError(f"{manifest_path}: surface precut Miller index cannot be zero")
    _positive_int(
        contract.get("oriented_unit_repeats"),
        "surface_precut_contract.oriented_unit_repeats",
    )
    for field in ("vacuum_A", "plane_tolerance_A", "symmetry_tolerance_A"):
        _positive_number(contract.get(field), f"surface_precut_contract.{field}")
    if contract.get("termination_enumeration") != "all_interplanar_gaps":
        raise ValueError(
            f"{manifest_path}: only all_interplanar_gaps termination enumeration "
            "is supported"
        )

    parent_formula = dict(sorted((manifest.get("bulk_parent") or {}).get("formula", {}).items()))
    oxidation_states = _table(
        contract.get("formal_oxidation_states"),
        "surface_precut_contract.formal_oxidation_states",
    )
    if set(oxidation_states) != set(parent_formula):
        raise ValueError(
            f"{manifest_path}: formal oxidation states must cover the Bulk Parent "
            "elements exactly"
        )
    for element, charge in oxidation_states.items():
        _element(element, f"formal_oxidation_states.{element}")
        if not isinstance(charge, (int, float)) or isinstance(charge, bool):
            raise ValueError(f"formal_oxidation_states.{element} must be numeric")
    if not math.isclose(
        sum(parent_formula[element] * oxidation_states[element] for element in parent_formula),
        0.0,
        abs_tol=1.0e-10,
    ):
        raise ValueError(f"{manifest_path}: Bulk Parent formal charge is not neutral")

    selection = _table(contract.get("selection"), "surface_precut_contract.selection")
    if selection.get("require_bulk_stoichiometry") is not True:
        raise ValueError("surface precut selection must require bulk stoichiometry")
    if selection.get("require_formal_charge_neutrality") is not True:
        raise ValueError("surface precut selection must require formal charge neutrality")
    _positive_number(
        selection.get("maximum_absolute_formal_dipole_density_e_per_A"),
        "selection.maximum_absolute_formal_dipole_density_e_per_A",
    )
    preferred = selection.get("preferred_top_elements")
    if not isinstance(preferred, list) or not preferred:
        raise ValueError("selection.preferred_top_elements must be a non-empty list")
    for index, element in enumerate(preferred):
        _element(element, f"selection.preferred_top_elements[{index}]")

    historical = contract.get("historical_comparison")
    if historical is not None:
        historical = _table(historical, "surface_precut_contract.historical_comparison")
        structure = historical.get("structure")
        if not isinstance(structure, str) or not structure.strip():
            raise ValueError("historical_comparison.structure must be a non-empty path")
        path = (manifest_path.parent / structure).resolve()
        if not path.is_file() or _sha256(path) != historical.get("sha256"):
            raise ValueError(
                f"{manifest_path}: historical surface comparison structure is missing "
                "or changed"
            )
        _positive_number(
            historical.get("maximum_scaled_geometry_rmsd_A"),
            "historical_comparison.maximum_scaled_geometry_rmsd_A",
        )
        historical["resolved_structure_path"] = str(path)
    return contract


def _integer_primitive_miller(
    conventional_cell: np.ndarray,
    primitive_cell: np.ndarray,
    conventional_miller: list[int],
) -> list[int]:
    transform = primitive_cell @ np.linalg.inv(conventional_cell)
    transformed = transform @ np.asarray(conventional_miller, dtype=float)
    fractions = [Fraction(float(value)).limit_denominator(96) for value in transformed]
    denominator = math.lcm(*(value.denominator for value in fractions))
    integers = np.asarray(
        [value.numerator * (denominator // value.denominator) for value in fractions],
        dtype=int,
    )
    nonzero = [abs(int(value)) for value in integers if value]
    if not nonzero:
        raise ValueError("Miller-index transformation produced the zero vector")
    integers //= reduce(math.gcd, nonzero)

    conventional_normal = np.linalg.inv(conventional_cell) @ np.asarray(
        conventional_miller, dtype=float
    )
    primitive_normal = np.linalg.inv(primitive_cell) @ integers.astype(float)
    cosine = float(
        np.dot(conventional_normal, primitive_normal)
        / (np.linalg.norm(conventional_normal) * np.linalg.norm(primitive_normal))
    )
    if cosine < 1.0 - 1.0e-7:
        raise ValueError("Cannot preserve the requested surface normal in primitive cell")
    return [int(value) for value in integers]


def _primitive_surface_parent(parent, conventional_miller, symprec: float):
    try:
        import spglib
    except ImportError as exc:
        raise RuntimeError("Surface Precut planning requires spglib") from exc
    standardized = spglib.standardize_cell(
        (
            np.asarray(parent.cell, dtype=float),
            np.asarray(parent.get_scaled_positions(wrap=True), dtype=float),
            np.asarray(parent.numbers, dtype=int),
        ),
        to_primitive=True,
        no_idealize=False,
        symprec=symprec,
    )
    if standardized is None:
        raise ValueError("Cannot standardize the Bulk Parent to a primitive cell")
    lattice, positions, numbers = standardized
    primitive = Atoms(
        numbers=numbers,
        scaled_positions=np.asarray(positions, dtype=float) % 1.0,
        cell=np.asarray(lattice, dtype=float),
        pbc=True,
    )
    primitive_miller = _integer_primitive_miller(
        np.asarray(parent.cell, dtype=float),
        np.asarray(primitive.cell, dtype=float),
        conventional_miller,
    )
    return primitive, primitive_miller


def _interplanar_cut_phases(primitive, primitive_miller) -> list[float]:
    miller = np.asarray(primitive_miller, dtype=int)
    phases = (
        np.asarray(primitive.get_scaled_positions(wrap=True), dtype=float) @ miller
    ) % 1.0
    unique = []
    for value in sorted(float(item) for item in phases):
        if not unique or value - unique[-1] > 1.0e-8:
            unique.append(value)
    cuts = []
    for index, lower in enumerate(unique):
        upper = unique[(index + 1) % len(unique)]
        if index == len(unique) - 1:
            upper += 1.0
        cut = ((lower + upper) / 2.0) % 1.0
        if abs(cut) < 1.0e-10 or abs(cut - 1.0) < 1.0e-10:
            cut = 0.0
        cuts.append(round(cut, 12))
    return sorted(set(cuts))


def _canonicalize_surface_basis(atoms):
    cell = np.asarray(atoms.cell, dtype=float).copy()
    vector_a, vector_b = cell[:2]
    if (
        np.dot(vector_a, vector_b) > 0
        and math.isclose(
            np.linalg.norm(vector_b - vector_a),
            np.linalg.norm(vector_b),
            rel_tol=1.0e-7,
            abs_tol=1.0e-7,
        )
    ):
        cell[1] = vector_b - vector_a
        atoms.set_cell(cell, scale_atoms=False)
        atoms.wrap()
    return atoms


def _build_surface_precut_candidate(primitive, primitive_miller, contract, cut):
    shifted = primitive.copy()
    miller = np.asarray(primitive_miller, dtype=float)
    fractional = np.asarray(shifted.get_scaled_positions(wrap=True), dtype=float)
    fractional -= float(cut) * miller / float(np.dot(miller, miller))
    shifted.set_scaled_positions(fractional % 1.0)
    slab = surface(
        shifted,
        tuple(int(value) for value in primitive_miller),
        int(contract["oriented_unit_repeats"]),
        vacuum=float(contract["vacuum_A"]) / 2.0,
        periodic=True,
    )
    _canonicalize_surface_basis(slab)
    scaled = slab.get_scaled_positions(wrap=True)
    order = np.lexsort(
        (
            scaled[:, 1],
            scaled[:, 0],
            np.asarray(slab.numbers, dtype=int),
            np.asarray(slab.positions[:, 2], dtype=float),
        )
    )
    return slab[order]


def _surface_plane_groups(atoms, tolerance: float) -> list[list[int]]:
    coordinates = np.asarray(atoms.positions[:, 2], dtype=float)
    groups: list[list[int]] = []
    for index in np.argsort(coordinates):
        if not groups or coordinates[index] - coordinates[groups[-1][-1]] > tolerance:
            groups.append([int(index)])
        else:
            groups[-1].append(int(index))
    return groups


def _surface_candidate_summary(atoms, contract: dict, cut: float) -> dict:
    symbols = np.asarray(atoms.get_chemical_symbols())
    z = np.asarray(atoms.positions[:, 2], dtype=float)
    groups = _surface_plane_groups(atoms, float(contract["plane_tolerance_A"]))
    oxidation_states = contract["formal_oxidation_states"]
    charges = np.asarray([float(oxidation_states[symbol]) for symbol in symbols])
    solid_center = (float(np.min(z)) + float(np.max(z))) / 2.0
    formal_dipole = float(np.sum(charges * (z - solid_center)))
    area = float(np.linalg.norm(np.cross(atoms.cell[0], atoms.cell[1])))
    thickness = float(np.ptp(z))
    normal_cell_length = float(np.linalg.norm(atoms.cell[2]))
    top_formula = _formula(symbols[groups[-1]])
    bottom_formula = _formula(symbols[groups[0]])
    selection = contract["selection"]
    preferred = set(selection["preferred_top_elements"])
    formula = _formula(symbols)
    total_charge = float(np.sum(charges))
    dipole_density = formal_dipole / area
    return {
        "cut_fractional_phase": float(cut),
        "formula": formula,
        "atom_count": len(atoms),
        "cell_A": {
            "lengths": np.asarray(atoms.cell.lengths(), dtype=float).tolist(),
            "angles_deg": np.asarray(atoms.cell.angles(), dtype=float).tolist(),
            "surface_area_A2": area,
        },
        "solid_thickness_A": thickness,
        "vacuum_A": normal_cell_length - thickness,
        "plane_count": len(groups),
        "bottom_plane_formula": bottom_formula,
        "top_plane_formula": top_formula,
        "formal_charge_e": total_charge,
        "formal_dipole_eA": formal_dipole,
        "formal_dipole_density_e_per_A": dipole_density,
        "checks": {
            "formal_charge_neutral": math.isclose(total_charge, 0.0, abs_tol=1.0e-8),
            "formal_nonpolar": abs(dipole_density)
            <= float(selection["maximum_absolute_formal_dipole_density_e_per_A"]),
            "preferred_top_elements": set(top_formula).issubset(preferred),
        },
    }


def _surface_geometry_comparison(candidate, reference, maximum_rmsd: float) -> dict:
    candidate_symbols = np.asarray(candidate.get_chemical_symbols())
    reference_symbols = np.asarray(reference.get_chemical_symbols())
    if _formula(candidate_symbols) != _formula(reference_symbols):
        return {
            "status": "not_comparable",
            "reason": "formula_mismatch",
            "candidate_formula": _formula(candidate_symbols),
            "reference_formula": _formula(reference_symbols),
        }
    if len(candidate) != len(reference):
        return {
            "status": "not_comparable",
            "reason": "atom_count_mismatch",
            "candidate_atom_count": len(candidate),
            "reference_atom_count": len(reference),
        }
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:
        raise RuntimeError("Surface geometry comparison requires SciPy") from exc

    candidate_xy = np.asarray(candidate.get_scaled_positions(wrap=True)[:, :2])
    reference_xy = np.asarray(reference.get_scaled_positions(wrap=True)[:, :2])
    candidate_z = np.asarray(candidate.positions[:, 2], dtype=float)
    reference_z = np.asarray(reference.positions[:, 2], dtype=float)
    candidate_z -= (float(np.min(candidate_z)) + float(np.max(candidate_z))) / 2.0
    reference_z -= (float(np.min(reference_z)) + float(np.max(reference_z))) / 2.0
    candidate_area = float(np.linalg.norm(np.cross(candidate.cell[0], candidate.cell[1])))
    reference_area = float(np.linalg.norm(np.cross(reference.cell[0], reference.cell[1])))
    isotropic_scale = math.sqrt(reference_area / candidate_area)
    candidate_z *= isotropic_scale
    anchor = min(
        set(reference_symbols),
        key=lambda element: (int(np.sum(reference_symbols == element)), element),
    )
    shifts = {
        tuple(np.round(reference_site - candidate_site, 8))
        for candidate_site in candidate_xy[candidate_symbols == anchor]
        for reference_site in reference_xy[reference_symbols == anchor]
    }
    reference_basis = np.asarray(reference.cell[:2], dtype=float)
    best = None
    for flip_z in (1, -1):
        for shift_tuple in sorted(shifts):
            shifted_xy = (candidate_xy + np.asarray(shift_tuple)) % 1.0
            mappings = []
            squared_sum = 0.0
            for element in sorted(set(reference_symbols)):
                candidate_indices = np.flatnonzero(candidate_symbols == element)
                reference_indices = np.flatnonzero(reference_symbols == element)
                delta_xy = (
                    shifted_xy[candidate_indices, None, :]
                    - reference_xy[None, reference_indices, :]
                )
                delta_xy -= np.round(delta_xy)
                in_plane = delta_xy @ reference_basis
                delta_z = (
                    flip_z * candidate_z[candidate_indices, None]
                    - reference_z[None, reference_indices]
                )
                distances = np.sqrt(
                    np.sum(in_plane * in_plane, axis=2) + delta_z * delta_z
                )
                rows, columns = linear_sum_assignment(distances)
                for row, column in zip(rows, columns):
                    distance = float(distances[row, column])
                    squared_sum += distance * distance
                    mappings.append((element, distance))
            score = (squared_sum, flip_z, shift_tuple)
            if best is None or score < best[0]:
                best = (score, mappings)
    if best is None:
        raise ValueError("No valid surface atom mapping found")
    (squared_sum, flip_z, shift_tuple), mappings = best
    rmsd = math.sqrt(squared_sum / len(candidate))
    distances = np.asarray([item[1] for item in mappings], dtype=float)
    by_element = {}
    for element in sorted(set(reference_symbols)):
        values = np.asarray([value for key, value in mappings if key == element])
        by_element[element] = {
            "count": len(values),
            "rmsd_A": float(np.sqrt(np.mean(values * values))),
            "maximum_displacement_A": float(np.max(values)),
        }
    candidate_lengths = np.asarray(candidate.cell.lengths()[:2], dtype=float)
    reference_lengths = np.asarray(reference.cell.lengths()[:2], dtype=float)
    return {
        "status": (
            "geometry_equivalent" if rmsd <= maximum_rmsd else "not_geometry_equivalent"
        ),
        "method": (
            "uniform_in_plane_scale_then_elementwise_hungarian_periodic_xy_"
            "and_nonperiodic_z_matching"
        ),
        "maximum_scaled_geometry_rmsd_A": float(maximum_rmsd),
        "scaled_geometry_rmsd_A": rmsd,
        "maximum_displacement_A": float(np.max(distances)),
        "by_element": by_element,
        "candidate_to_reference_isotropic_scale": isotropic_scale,
        "candidate_in_plane_length_relative_deviations": (
            np.abs(candidate_lengths - reference_lengths) / reference_lengths
        ).tolist(),
        "z_flipped": flip_z == -1,
        "origin_shift_fractional_xy": list(shift_tuple),
    }


def _surface_precut_candidates(parent, contract):
    primitive, primitive_miller = _primitive_surface_parent(
        parent,
        contract["miller"],
        float(contract["symmetry_tolerance_A"]),
    )
    expected_formula = {
        element: count * int(contract["oriented_unit_repeats"])
        for element, count in _formula(primitive.get_chemical_symbols()).items()
    }
    candidates = []
    atoms_by_id = {}
    for index, cut in enumerate(
        _interplanar_cut_phases(primitive, primitive_miller), start=1
    ):
        candidate_id = f"termination-{index:03d}"
        atoms = _build_surface_precut_candidate(
            primitive, primitive_miller, contract, cut
        )
        summary = _surface_candidate_summary(atoms, contract, cut)
        summary["candidate_id"] = candidate_id
        summary["checks"]["bulk_stoichiometry"] = summary["formula"] == expected_formula
        summary["selection_eligible"] = all(summary["checks"].values())
        candidates.append(summary)
        atoms_by_id[candidate_id] = atoms
    return primitive, primitive_miller, candidates, atoms_by_id


def resolve_surface_precut_plan(
    substrate_name: str,
    structure_directory: Path | None = None,
) -> dict:
    """Resolve all clean-surface termination candidates without writing files."""

    manifest = _load_structure_manifest(substrate_name, structure_directory)
    parent = _validated_registered_bulk_parent(manifest)
    if parent is None:
        raise ValueError(f"No accepted Bulk Parent for {substrate_name!r}")
    registered_precuts = _validated_registered_surface_precuts(manifest)
    contract = _validated_surface_precut_contract(manifest)
    parent_atoms = read(parent["path"])
    primitive, primitive_miller, candidates, atoms_by_id = _surface_precut_candidates(
        parent_atoms, contract
    )
    eligible = [item for item in candidates if item["selection_eligible"]]
    readiness_issues = []
    if len(eligible) != 1:
        readiness_issues.append(
            {
                "code": "surface_termination_selection_not_unique",
                "message": (
                    f"Expected one eligible termination; found {len(eligible)}. "
                    "Refine the explicit selection contract."
                ),
            }
        )
    historical_report = None
    historical = contract.get("historical_comparison")
    if historical is not None and len(eligible) == 1:
        historical_atoms = read(historical["resolved_structure_path"])
        historical_report = {
            "path": historical["structure"],
            "sha256": historical["sha256"],
            **_surface_geometry_comparison(
                atoms_by_id[eligible[0]["candidate_id"]],
                historical_atoms,
                float(historical["maximum_scaled_geometry_rmsd_A"]),
            ),
        }
    identity = {
        "substrate_key": manifest["substrate_key"],
        "bulk_parent_sha256": parent["sha256"],
        "contract": {
            key: value
            for key, value in contract.items()
            if key != "historical_comparison"
        },
        "historical_comparison": (
            {
                "sha256": historical["sha256"],
                "maximum_scaled_geometry_rmsd_A": historical[
                    "maximum_scaled_geometry_rmsd_A"
                ],
            }
            if historical is not None
            else None
        ),
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema_version": 1,
        "mode": "surface_precut_plan",
        "substrate": {
            "key": manifest["substrate_key"],
            "host_material": manifest["host_material"],
            "structure_manifest": {
                "path": str(manifest["manifest_path"]),
                "sha256": _sha256(Path(manifest["manifest_path"])),
            },
        },
        "bulk_parent": parent,
        "registered_surface_precuts": registered_precuts,
        "surface": {
            "conventional_miller": list(contract["miller"]),
            "primitive_miller": primitive_miller,
            "primitive_formula": _formula(primitive.get_chemical_symbols()),
            "primitive_atom_count": len(primitive),
            "oriented_unit_repeats": int(contract["oriented_unit_repeats"]),
            "vacuum_A": float(contract["vacuum_A"]),
        },
        "surface_precut_contract": contract,
        "termination_candidates": candidates,
        "selected_candidate_id": (
            eligible[0]["candidate_id"] if len(eligible) == 1 else None
        ),
        "historical_comparison": historical_report,
        "fingerprint": fingerprint,
        "execution_contract": {
            "read_only": True,
            "executable": not readiness_issues,
            "readiness_issues": readiness_issues,
            "known_limitations": [
                "Candidates are unrelaxed geometric Surface Precuts.",
                "Formal-charge nonpolarity is not a calculated surface energy.",
            ],
        },
    }


def execute_surface_precut_enumeration(plan: dict, output_root: Path) -> dict:
    """Write one immutable package containing every planned termination candidate."""

    if plan.get("mode") != "surface_precut_plan":
        raise ValueError("Surface Precut enumeration requires a surface precut plan")
    if not plan["execution_contract"]["executable"]:
        raise ValueError("Surface Precut plan has unresolved readiness issues")
    output_root = Path(output_root).expanduser().resolve()
    contract = plan["surface_precut_contract"]
    run_directory = (
        output_root
        / plan["substrate"]["key"].casefold()
        / f"{contract['protocol_version']}-{plan['fingerprint'][:12]}"
    )
    if run_directory.exists():
        raise ValueError(f"Immutable Surface Precut run already exists: {run_directory}")
    run_directory.parent.mkdir(parents=True, exist_ok=True)
    parent_atoms = read(plan["bulk_parent"]["path"])
    _, primitive_miller, candidates, atoms_by_id = _surface_precut_candidates(
        parent_atoms, contract
    )
    if candidates != plan["termination_candidates"]:
        raise ValueError("Surface Precut candidates changed after planning")

    with TemporaryDirectory(
        dir=run_directory.parent, prefix=f".{run_directory.name}.tmp-"
    ) as temporary_name:
        temporary = Path(temporary_name)
        candidates_directory = temporary / "candidates"
        candidates_directory.mkdir()
        plan_path = temporary / "surface-precut-plan.json"
        report_path = temporary / "selection-report.json"
        manifest_path = temporary / "run-manifest.json"
        write_manifest(plan_path, plan)
        artifacts = []
        selected_id = plan["selected_candidate_id"]
        for candidate in candidates:
            candidate_id = candidate["candidate_id"]
            path = candidates_directory / f"{candidate_id}.cif"
            write_clean_structure(atoms_by_id[candidate_id], path)
            artifacts.append(
                {
                    "role": (
                        "selected_surface_precut_candidate"
                        if candidate_id == selected_id
                        else "surface_precut_candidate"
                    ),
                    "candidate_id": candidate_id,
                    "path": str(path.relative_to(temporary)),
                    "sha256": _sha256(path),
                }
            )
        report = {
            "schema_version": 1,
            "status": "passed_selection",
            "selected_candidate_id": selected_id,
            "selection_basis": (
                "bulk stoichiometry, formal charge neutrality, explicit formal-dipole "
                "threshold, and preferred top element"
            ),
            "historical_comparison": plan["historical_comparison"],
            "known_limitations": plan["execution_contract"]["known_limitations"],
        }
        write_manifest(report_path, report)
        artifacts.extend(
            [
                {
                    "role": "surface_precut_plan",
                    "path": plan_path.name,
                    "sha256": _sha256(plan_path),
                },
                {
                    "role": "selection_report",
                    "path": report_path.name,
                    "sha256": _sha256(report_path),
                },
            ]
        )
        manifest = {
            "schema_version": 1,
            "status": "passed_selection",
            "fingerprint": plan["fingerprint"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "output_directory": str(run_directory),
            "host": socket.gethostname(),
            "bulk_parent_sha256": plan["bulk_parent"]["sha256"],
            "conventional_miller": plan["surface"]["conventional_miller"],
            "primitive_miller": primitive_miller,
            "selected_candidate_id": selected_id,
            "artifacts": artifacts,
        }
        write_manifest(manifest_path, manifest)
        temporary.replace(run_directory)
    return manifest


def _reduced_formula(formula: dict[str, int]) -> dict[str, int]:
    counts = [int(value) for value in formula.values()]
    if not counts or any(value < 1 for value in counts):
        raise ValueError("A reduced formula requires positive integer counts")
    divisor = reduce(math.gcd, counts)
    return {element: int(count // divisor) for element, count in formula.items()}


def _formula_units(formula: dict[str, int], unit: dict[str, int]) -> int:
    if set(formula) != set(unit):
        raise ValueError("Formula elements do not match the bulk formula unit")
    ratios = []
    for element in sorted(unit):
        count = formula[element]
        unit_count = unit[element]
        if count % unit_count:
            raise ValueError(
                "Formula is not an integer multiple of the bulk formula unit"
            )
        ratios.append(count // unit_count)
    if not ratios or len(set(ratios)) != 1 or ratios[0] < 1:
        raise ValueError("Formula is not a uniform multiple of the bulk formula unit")
    return int(ratios[0])


def _validated_surface_relaxation_contract(manifest: dict) -> dict:
    manifest_path = Path(manifest["manifest_path"])
    contract = _table(
        manifest.get("surface_relaxation_contract"),
        "surface_relaxation_contract",
    )
    version = contract.get("protocol_version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError(
            f"{manifest_path}: surface relaxation protocol_version is required"
        )
    repeats = contract.get("thickness_repeats")
    if (
        not isinstance(repeats, list)
        or len(repeats) < 2
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in repeats
        )
        or repeats != sorted(set(repeats))
    ):
        raise ValueError(
            "surface_relaxation_contract.thickness_repeats must contain at least "
            "two unique increasing positive integers"
        )
    for field in (
        "vacuum_A",
        "force_tolerance_eV_A",
        "maximum_step_A",
    ):
        _positive_number(contract.get(field), f"surface_relaxation_contract.{field}")
    _positive_int(
        contract.get("maximum_steps"), "surface_relaxation_contract.maximum_steps"
    )
    if contract.get("relax_atomic_positions") is not True:
        raise ValueError("Surface relaxation must relax atomic positions")
    if contract.get("relax_cell") is not False:
        raise ValueError(
            "Surface relaxation must keep the Bulk Parent in-plane cell fixed"
        )
    if contract.get("translation_constraint") != "fix_center_of_mass":
        raise ValueError(
            "Only the fix_center_of_mass translation constraint is supported"
        )
    if contract.get("surface_count") != 2:
        raise ValueError("Only symmetric two-sided slabs are supported")
    convergence = _table(
        contract.get("convergence"), "surface_relaxation_contract.convergence"
    )
    if convergence.get("metric") != "surface_energy_eV_A2":
        raise ValueError("Surface convergence metric must be surface_energy_eV_A2")
    if convergence.get("comparison") != "two_thickest":
        raise ValueError(
            "Surface convergence currently compares the two thickest slabs"
        )
    _positive_number(
        convergence.get("maximum_absolute_difference_eV_A2"),
        "surface_relaxation_contract.convergence.maximum_absolute_difference_eV_A2",
    )
    return contract


def _selected_registered_surface_precut(manifest: dict, contract: dict) -> dict:
    validated = _validated_registered_surface_precuts(manifest)
    matches = [
        entry
        for entry in manifest.get("precut_slabs", [])
        if entry.get("status")
        in {
            "accepted_bulk_derived_unrelaxed_precut",
            "accepted_relaxed_maintained_surface",
        }
        and list(entry.get("miller", [])) == list(contract["miller"])
    ]
    validated_hashes = {entry["sha256"] for entry in validated}
    matches = [entry for entry in matches if entry.get("sha256") in validated_hashes]
    if len(matches) != 1:
        raise ValueError(
            "Surface relaxation requires exactly one accepted geometric precut for "
            "the declared Miller index"
        )
    return matches[0]


def _surface_relaxation_candidates(
    parent_atoms,
    precut_contract: dict,
    relaxation_contract: dict,
    cut_fractional_phase: float,
) -> tuple[list[int], list[dict], dict[int, Atoms]]:
    primitive, primitive_miller = _primitive_surface_parent(
        parent_atoms,
        precut_contract["miller"],
        float(precut_contract["symmetry_tolerance_A"]),
    )
    primitive_formula = _formula(primitive.get_chemical_symbols())
    reduced_bulk_formula = _reduced_formula(primitive_formula)
    candidates = []
    atoms_by_repeat = {}
    for repeat in relaxation_contract["thickness_repeats"]:
        candidate_contract = dict(precut_contract)
        candidate_contract["oriented_unit_repeats"] = int(repeat)
        candidate_contract["vacuum_A"] = float(relaxation_contract["vacuum_A"])
        atoms = _build_surface_precut_candidate(
            primitive,
            primitive_miller,
            candidate_contract,
            float(cut_fractional_phase),
        )
        summary = _surface_candidate_summary(
            atoms,
            candidate_contract,
            float(cut_fractional_phase),
        )
        expected_formula = {
            element: count * int(repeat) for element, count in primitive_formula.items()
        }
        summary["oriented_unit_repeats"] = int(repeat)
        summary["checks"]["bulk_stoichiometry"] = summary["formula"] == expected_formula
        summary["checks"]["symmetric_plane_composition"] = (
            summary["top_plane_formula"] == summary["bottom_plane_formula"]
        )
        summary["selection_eligible"] = all(summary["checks"].values())
        summary["bulk_formula_units"] = _formula_units(
            summary["formula"], reduced_bulk_formula
        )
        candidates.append(summary)
        atoms_by_repeat[int(repeat)] = atoms
    return primitive_miller, candidates, atoms_by_repeat


def resolve_surface_relaxation_plan(
    substrate_name: str,
    model_path: Path,
    declared_model_elements: list[str],
    structure_directory: Path | None = None,
    *,
    default_dtype: str = "float32",
) -> dict:
    """Resolve a read-only symmetric-slab relaxation and convergence plan."""

    manifest = _load_structure_manifest(substrate_name, structure_directory)
    parent = _validated_registered_bulk_parent(manifest)
    if parent is None:
        raise ValueError(f"No accepted Bulk Parent for {substrate_name!r}")
    precut_contract = _validated_surface_precut_contract(manifest)
    relaxation_contract = _validated_surface_relaxation_contract(manifest)
    registered_precut = _selected_registered_surface_precut(manifest, precut_contract)
    if (
        int(registered_precut["source_layers"])
        not in relaxation_contract["thickness_repeats"]
    ):
        raise ValueError(
            "The registered Surface Precut thickness must be included in the convergence series"
        )
    model_path = Path(model_path).expanduser().resolve()
    if not model_path.is_file():
        raise ValueError(f"Model file does not exist: {model_path}")
    model_sha256 = _sha256(model_path)
    parent_potential = parent["potential"]
    if model_sha256 != parent_potential.get("model_sha256"):
        raise ValueError(
            "Surface relaxation model hash must match the accepted Bulk Parent model"
        )
    if default_dtype not in {"float32", "float64"}:
        raise ValueError("default_dtype must be float32 or float64")
    if default_dtype != parent_potential.get("default_dtype"):
        raise ValueError(
            "Surface relaxation dtype must match the accepted Bulk Parent dtype"
        )
    model_elements = sorted(
        {
            _element(value, "declared_model_elements")
            for value in declared_model_elements
        }
    )
    required_elements = sorted(parent["formula"])
    missing = sorted(set(required_elements) - set(model_elements))
    if missing:
        raise ValueError(
            "Declared model elements do not cover the surface: " + ", ".join(missing)
        )

    parent_atoms = read(parent["path"])
    primitive_miller, candidates, _ = _surface_relaxation_candidates(
        parent_atoms,
        precut_contract,
        relaxation_contract,
        float(registered_precut["cut_fractional_phase"]),
    )
    readiness_issues = []
    invalid_repeats = [
        candidate["oriented_unit_repeats"]
        for candidate in candidates
        if not candidate["selection_eligible"]
    ]
    if invalid_repeats:
        readiness_issues.append(
            {
                "code": "invalid_symmetric_thickness_candidate",
                "message": f"Thickness candidates failed geometric gates: {invalid_repeats}",
            }
        )
    reduced_bulk_formula = _reduced_formula(parent["formula"])
    parent_formula_units = _formula_units(parent["formula"], reduced_bulk_formula)
    identity = {
        "substrate_key": manifest["substrate_key"],
        "bulk_parent_sha256": parent["sha256"],
        "surface_precut_sha256": registered_precut["sha256"],
        "model_sha256": model_sha256,
        "default_dtype": default_dtype,
        "surface_relaxation_contract": relaxation_contract,
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema_version": 1,
        "mode": "surface_relaxation_plan",
        "substrate": {
            "key": manifest["substrate_key"],
            "host_material": manifest["host_material"],
            "structure_manifest": {
                "path": str(manifest["manifest_path"]),
                "sha256": _sha256(Path(manifest["manifest_path"])),
            },
        },
        "bulk_parent": parent,
        "bulk_reference": {
            "reduced_formula": reduced_bulk_formula,
            "parent_formula_units": parent_formula_units,
            "energy_policy": "fresh_same_model_single_point_on_accepted_bulk_parent",
        },
        "registered_surface_precut": {
            "path": str(
                (
                    Path(manifest["manifest_path"]).parent / registered_precut["path"]
                ).resolve()
            ),
            "sha256": registered_precut["sha256"],
            "source_layers": int(registered_precut["source_layers"]),
            "miller": list(registered_precut["miller"]),
            "primitive_miller": list(registered_precut["primitive_miller"]),
            "termination": registered_precut["termination"],
            "cut_fractional_phase": float(registered_precut["cut_fractional_phase"]),
        },
        "surface": {
            "conventional_miller": list(precut_contract["miller"]),
            "primitive_miller": primitive_miller,
            "surface_count": 2,
        },
        "potential": {
            "model_path": str(model_path),
            "model_sha256": model_sha256,
            "required_elements": required_elements,
            "declared_model_elements": model_elements,
            "coverage": "passed",
            "coverage_basis": "maintainer_declaration_not_model_introspection",
            "default_dtype": default_dtype,
        },
        "surface_precut_contract": precut_contract,
        "surface_relaxation_contract": relaxation_contract,
        "thickness_candidates": candidates,
        "surface_energy_definition": {
            "equation": "gamma=(E_slab-N_formula*E_bulk_formula)/(2*A)",
            "native_unit": "eV/A^2",
            "reported_units": ["eV/A^2", "J/m^2"],
            "J_m2_per_eV_A2": J_M2_PER_EV_A2,
            "bulk_reference_policy": "same model, dtype, device, and accepted Bulk Parent",
        },
        "fingerprint": fingerprint,
        "execution_contract": {
            "read_only": True,
            "executable": not readiness_issues,
            "execution_command": "surface-relaxation-run",
            "readiness_issues": readiness_issues,
            "registration_policy": (
                "A passed run is only a relaxation/convergence candidate; promotion "
                "to a maintained relaxed surface requires a separate hash-verified record."
            ),
        },
    }


def evaluate_surface_relaxation_convergence(plan: dict, results: list[dict]) -> dict:
    """Evaluate per-thickness relaxation gates and the two-thickest energy delta."""

    if plan.get("mode") != "surface_relaxation_plan":
        raise ValueError("Surface convergence evaluation requires a relaxation plan")
    expected = list(plan["surface_relaxation_contract"]["thickness_repeats"])
    ordered = sorted(results, key=lambda item: item["oriented_unit_repeats"])
    observed = [item["oriented_unit_repeats"] for item in ordered]
    if observed != expected:
        raise ValueError(
            f"Surface relaxation results cover {observed}; expected {expected}"
        )
    thinner, thickest = ordered[-2:]
    delta = abs(
        float(thickest["surface_energy_eV_A2"]) - float(thinner["surface_energy_eV_A2"])
    )
    maximum = float(
        plan["surface_relaxation_contract"]["convergence"][
            "maximum_absolute_difference_eV_A2"
        ]
    )
    checks = {
        "all_relaxations_passed": {
            "passed": all(item.get("status") == "passed" for item in ordered),
            "failed_repeats": [
                item["oriented_unit_repeats"]
                for item in ordered
                if item.get("status") != "passed"
            ],
        },
        "two_thickest_surface_energy": {
            "passed": delta <= maximum,
            "thinner_repeat": thinner["oriented_unit_repeats"],
            "thickest_repeat": thickest["oriented_unit_repeats"],
            "absolute_difference_eV_A2": delta,
            "absolute_difference_J_m2": delta * J_M2_PER_EV_A2,
            "maximum_eV_A2": maximum,
            "maximum_J_m2": maximum * J_M2_PER_EV_A2,
        },
    }
    return {
        "schema_version": 1,
        "status": (
            "passed" if all(item["passed"] for item in checks.values()) else "failed"
        ),
        "checks": checks,
        "thickness_series": ordered,
        "interpretation": (
            "Convergence is established only for this termination, potential, "
            "vacuum, relaxation contract, and tested thickness series."
        ),
    }


def _validated_termination_screening_contract(relaxation_contract: dict) -> dict:
    contract = _table(
        relaxation_contract.get("termination_screening"),
        "surface_relaxation_contract.termination_screening",
    )
    version = contract.get("protocol_version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("termination screening protocol_version is required")
    if contract.get("enumeration") != "all_interplanar_cut_phases":
        raise ValueError("Termination screening must enumerate all interplanar cuts")
    if contract.get("energy_quantity") != "two_surface_pair_average":
        raise ValueError("Termination screening energy must be a two-surface pair average")
    if contract.get("formally_polar_policy") != (
        "classify_without_energy_requires_compensation_protocol"
    ):
        raise ValueError(
            "Uncompensated polar terminations must be classified without an "
            "energy calculation until a compensation protocol is declared"
        )
    selection = _table(contract.get("selection"), "termination_screening.selection")
    if selection.get("require_bulk_stoichiometry") is not True:
        raise ValueError("Termination selection must require bulk stoichiometry")
    if selection.get("require_formal_charge_neutrality") is not True:
        raise ValueError("Termination selection must require formal charge neutrality")
    if selection.get("require_formal_nonpolarity") is not True:
        raise ValueError("Termination selection must exclude uncompensated polar slabs")
    if selection.get("require_equivalent_surface_plane_composition") is not True:
        raise ValueError(
            "Termination selection must require matching top/bottom plane compositions"
        )
    if selection.get("require_force_convergence") is not True:
        raise ValueError("Termination selection must require force convergence")
    if selection.get("require_thickness_convergence") is not True:
        raise ValueError("Termination selection must require thickness convergence")
    if selection.get("ranking") != "lowest_converged_surface_energy":
        raise ValueError("Termination selection ranking must use converged surface energy")
    return contract


def _validated_surface_promotion_contract(relaxation_contract: dict) -> dict:
    contract = _table(
        relaxation_contract.get("promotion"),
        "surface_relaxation_contract.promotion",
    )
    if contract.get("protocol_version") != "relaxed_surface_promotion_v1":
        raise ValueError("Unsupported relaxed-surface promotion protocol")
    if contract.get("preserve_registered_thickness") is not True:
        raise ValueError("Surface promotion must preserve the registered thickness")
    for field in ("destination_directory", "structure_filename"):
        value = contract.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"surface promotion {field} must be non-empty")
        candidate = Path(value)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"surface promotion {field} must be a safe relative path")
    if Path(contract["structure_filename"]).name != contract["structure_filename"]:
        raise ValueError("surface promotion structure_filename must be one filename")
    if Path(contract["structure_filename"]).suffix.casefold() != ".cif":
        raise ValueError("surface promotion structure_filename must end in .cif")
    return contract


def _surface_relaxation_subplan_for_cut(
    base_plan: dict,
    termination: dict,
) -> dict:
    subplan = json.loads(json.dumps(base_plan))
    subplan["registered_surface_precut"]["cut_fractional_phase"] = float(
        termination["cut_fractional_phase"]
    )
    subplan["registered_surface_precut"]["termination"] = (
        "termination_screening_candidate"
    )
    subplan["registered_surface_precut"]["screening_candidate_id"] = termination[
        "candidate_id"
    ]
    subplan["thickness_candidates"] = termination["thickness_candidates"]
    subplan["termination_screening_candidate"] = {
        key: termination[key]
        for key in (
            "candidate_id",
            "cut_fractional_phase",
            "top_plane_formula",
            "bottom_plane_formula",
            "formal_dipole_density_e_per_A",
            "classification",
        )
    }
    identity = {
        "base_plan_fingerprint": base_plan["fingerprint"],
        "termination_candidate_id": termination["candidate_id"],
        "cut_fractional_phase": termination["cut_fractional_phase"],
    }
    subplan["fingerprint"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    subplan["execution_contract"] = {
        **subplan["execution_contract"],
        "executable": True,
        "readiness_issues": [],
        "screening_diagnostic": True,
    }
    return subplan


def resolve_surface_termination_screen_plan(
    substrate_name: str,
    model_path: Path,
    declared_model_elements: list[str],
    structure_directory: Path | None = None,
    *,
    default_dtype: str = "float32",
) -> dict:
    """Resolve every same-face exposure pair for classification and valid ranking."""

    base_plan = resolve_surface_relaxation_plan(
        substrate_name,
        model_path,
        declared_model_elements,
        structure_directory,
        default_dtype=default_dtype,
    )
    screening_contract = _validated_termination_screening_contract(
        base_plan["surface_relaxation_contract"]
    )
    parent_atoms = read(base_plan["bulk_parent"]["path"])
    _, _, terminations, _ = _surface_precut_candidates(
        parent_atoms,
        base_plan["surface_precut_contract"],
    )
    candidates = []
    for termination in terminations:
        _, thickness_candidates, _ = _surface_relaxation_candidates(
            parent_atoms,
            base_plan["surface_precut_contract"],
            base_plan["surface_relaxation_contract"],
            float(termination["cut_fractional_phase"]),
        )
        stoichiometric = all(
            item["checks"]["bulk_stoichiometry"] for item in thickness_candidates
        )
        neutral = all(
            item["checks"]["formal_charge_neutral"] for item in thickness_candidates
        )
        nonpolar = all(
            item["checks"]["formal_nonpolar"] for item in thickness_candidates
        )
        same_plane_composition = all(
            item["top_plane_formula"] == item["bottom_plane_formula"]
            for item in thickness_candidates
        )
        classification = (
            "directly_rankable_nonpolar_pair"
            if stoichiometric
            and neutral
            and nonpolar
            and same_plane_composition
            else "requires_separate_surface_energy_protocol"
        )
        candidates.append(
            {
                "candidate_id": termination["candidate_id"],
                "cut_fractional_phase": termination["cut_fractional_phase"],
                "top_plane_formula": termination["top_plane_formula"],
                "bottom_plane_formula": termination["bottom_plane_formula"],
                "formal_dipole_density_e_per_A": termination[
                    "formal_dipole_density_e_per_A"
                ],
                "classification": classification,
                "checks": {
                    "bulk_stoichiometry": stoichiometric,
                    "formal_charge_neutral": neutral,
                    "formal_nonpolar": nonpolar,
                    "same_top_bottom_plane_composition": same_plane_composition,
                },
                "individual_surface_energy_interpretation": (
                    "allowed_for_symmetric_nonpolar_pair"
                    if nonpolar and same_plane_composition
                    else "pair_average_only"
                ),
                "thickness_candidates": thickness_candidates,
            }
        )
    identity = {
        "base_plan_fingerprint": base_plan["fingerprint"],
        "screening_contract": screening_contract,
        "termination_candidates": [
            {
                "candidate_id": item["candidate_id"],
                "cut_fractional_phase": item["cut_fractional_phase"],
                "classification": item["classification"],
            }
            for item in candidates
        ],
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    rankable = [
        item["candidate_id"]
        for item in candidates
        if item["classification"] == "directly_rankable_nonpolar_pair"
    ]
    readiness_issues = []
    if not rankable:
        readiness_issues.append(
            {
                "code": "no_formally_nonpolar_termination_pair",
                "message": (
                    "No stoichiometric, charge-neutral, formally nonpolar termination "
                    "pair can be ranked without a compensation model."
                ),
            }
        )
    return {
        "schema_version": 1,
        "mode": "surface_termination_screen_plan",
        "substrate": base_plan["substrate"],
        "bulk_parent": base_plan["bulk_parent"],
        "surface": base_plan["surface"],
        "potential": base_plan["potential"],
        "surface_precut_contract": base_plan["surface_precut_contract"],
        "surface_relaxation_contract": base_plan["surface_relaxation_contract"],
        "termination_screening_contract": screening_contract,
        "base_surface_relaxation_plan": base_plan,
        "termination_candidates": candidates,
        "pre_relaxation_rankable_candidate_ids": rankable,
        "energy_interpretation": {
            "reported_quantity": "two_surface_pair_average",
            "equation": "gamma_pair_average=(E_slab-N_formula*E_bulk_formula)/(2*A)",
            "warning": (
                "For an asymmetric or formally polar slab this is only the average "
                "excess energy of its two complementary surfaces, not either individual "
                "surface energy. Uncompensated polar pairs are classification-only "
                "until a compensation protocol is declared."
            ),
        },
        "fingerprint": fingerprint,
        "execution_contract": {
            "read_only": True,
            "executable": not readiness_issues,
            "execution_command": "surface-termination-screen-run",
            "readiness_issues": readiness_issues,
        },
    }


def _make_mace_calculator(model_path: Path, device: str, default_dtype: str):
    try:
        import mace
        import torch
        from mace.calculators import MACECalculator
    except ImportError as exc:
        raise RuntimeError(
            "MACE execution requires MACE, PyTorch, and their ASE calculator"
        ) from exc
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")
    return MACECalculator(
        model_paths=str(model_path),
        device=device,
        default_dtype=default_dtype,
    ), {
        "mace_version": getattr(mace, "__version__", "unknown"),
        "torch_version": getattr(torch, "__version__", "unknown"),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": (
            torch.cuda.get_device_name(torch.device(device))
            if device.startswith("cuda") and torch.cuda.is_available()
            else None
        ),
    }


def execute_bulk_relaxation(
    plan: dict,
    output_root: Path,
    *,
    device: str = "cuda",
    default_dtype: str = "float32",
) -> dict:
    """Run one immutable symmetry-preserving MACE Bulk Parent relaxation."""

    if plan.get("mode") != "bulk_relaxation_plan":
        raise ValueError("Bulk relaxation requires a resolved bulk relaxation plan")
    if not plan["execution_contract"]["executable"]:
        raise ValueError("Bulk relaxation plan is not executable")
    if default_dtype not in {"float32", "float64"}:
        raise ValueError("default_dtype must be float32 or float64")

    try:
        from ase.constraints import FixSymmetry
        from ase.filters import FrechetCellFilter
        from ase.io.trajectory import Trajectory
        from ase.optimize import LBFGS
    except ImportError as exc:
        raise RuntimeError("Bulk relaxation requires a complete ASE installation") from exc

    contract = plan["relaxation_contract"]
    identity = {
        "substrate_key": plan["substrate"]["key"],
        "bulk_seed_sha256": plan["bulk_seed"]["sha256"],
        "model_sha256": plan["potential"]["model_sha256"],
        "relaxation_contract": plan["relaxation_contract"],
        "optimizer_contract": plan["optimizer_contract"],
        "device": device,
        "default_dtype": default_dtype,
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = Path(output_root).expanduser().resolve()
    run_directory = (
        output_root
        / plan["substrate"]["key"].casefold()
        / f"{contract['protocol_version']}-{timestamp}-{fingerprint[:12]}"
    )
    if run_directory.exists():
        raise ValueError(f"Immutable bulk relaxation run already exists: {run_directory}")
    run_directory.mkdir(parents=True)

    plan_path = run_directory / "bulk-relaxation-plan.json"
    progress_path = run_directory / "progress.jsonl"
    optimizer_log_path = run_directory / "optimizer.log"
    trajectory_path = run_directory / "optimization.traj"
    candidate_path = run_directory / "bulk-parent-candidate.cif"
    acceptance_path = run_directory / "acceptance-report.json"
    manifest_path = run_directory / "run-manifest.json"
    write_manifest(plan_path, plan)

    manifest = {
        "schema_version": 1,
        "status": "running",
        "fingerprint": fingerprint,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_directory": str(run_directory),
        "host": socket.gethostname(),
        "python": sys.version.split()[0],
        "device": device,
        "default_dtype": default_dtype,
        "plan": {"path": plan_path.name, "sha256": _sha256(plan_path)},
        "artifacts": [],
    }
    write_manifest(manifest_path, manifest)
    print(f"BULK_RUN_DIRECTORY {run_directory}", flush=True)

    atoms = read(plan["bulk_seed"]["path"])
    initial_formula = _formula(atoms.get_chemical_symbols())
    calculator = None
    trajectory = None
    try:
        symmetry_tolerance = float(contract["symmetry_tolerance_A"])
        initial_symmetry = _space_group(atoms, symmetry_tolerance)
        expected_space_group = int(
            contract["acceptance"]["expected_space_group_number"]
        )
        if initial_symmetry["number"] != expected_space_group:
            raise ValueError(
                f"Bulk Seed space group is {initial_symmetry['number']}; "
                f"expected {expected_space_group}"
            )
        atoms.set_constraint(
            FixSymmetry(atoms, symprec=symmetry_tolerance, verbose=False)
        )
        calculator, runtime = _make_mace_calculator(
            Path(plan["potential"]["model_path"]), device, default_dtype
        )
        atoms.calc = calculator
        pressure_eV_A3 = float(contract["target_pressure_GPa"]) / GPA_PER_EV_A3
        cell_filter = FrechetCellFilter(
            atoms,
            mask=[True, True, True, False, False, False],
            scalar_pressure=pressure_eV_A3,
            exp_cell_factor=float(plan["optimizer_contract"]["exp_cell_factor"]),
        )
        optimizer = LBFGS(
            cell_filter,
            logfile=str(optimizer_log_path),
            maxstep=float(contract["maximum_step_A"]),
        )
        trajectory = Trajectory(str(trajectory_path), "w", atoms)
        optimizer.attach(trajectory.write, interval=1)

        reference_cell = contract["reference"]["cell_A"]
        reference_lengths = np.asarray(
            [reference_cell[axis] for axis in ("a", "b", "c")], dtype=float
        )

        def record_progress():
            metrics = _bulk_metrics(atoms, reference_lengths)
            record = {
                "optimizer_step": int(optimizer.get_number_of_steps()),
                "time_utc": datetime.now(timezone.utc).isoformat(),
                "energy_eV": metrics["potential_energy_eV"],
                "maximum_force_eV_A": metrics["maximum_force_eV_A"],
                "maximum_absolute_stress_GPa": metrics[
                    "maximum_absolute_stress_GPa"
                ],
                "cell_lengths_A": metrics["cell_A"]["lengths"],
            }
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(
                "BULK_PROGRESS "
                f"step={record['optimizer_step']} "
                f"fmax={record['maximum_force_eV_A']:.6f}eV/A "
                f"smax={record['maximum_absolute_stress_GPa']:.6f}GPa "
                f"cell={record['cell_lengths_A']}",
                flush=True,
            )

        optimizer.attach(record_progress, interval=1)
        record_progress()
        converged = bool(
            optimizer.run(
                fmax=float(contract["force_tolerance_eV_A"]),
                steps=int(contract["maximum_steps"]),
            )
        )
        trajectory.close()
        trajectory = None
        atoms.set_constraint()
        write_clean_structure(atoms, candidate_path)
        acceptance_report = evaluate_bulk_parent_candidate(
            plan,
            atoms,
            optimizer_converged=converged,
            optimizer_steps=optimizer.get_number_of_steps(),
        )
        write_manifest(acceptance_path, acceptance_report)
        manifest.update(
            {
                "status": (
                    "passed"
                    if acceptance_report["status"] == "passed"
                    else "failed_acceptance"
                ),
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "runtime": runtime,
                "initial_symmetry": initial_symmetry,
                "initial_formula": initial_formula,
                "acceptance_status": acceptance_report["status"],
                "artifacts": [
                    {
                        "role": "optimizer_log",
                        "path": optimizer_log_path.name,
                        "sha256": _sha256(optimizer_log_path),
                    },
                    {
                        "role": "optimization_trajectory",
                        "path": trajectory_path.name,
                        "sha256": _sha256(trajectory_path),
                    },
                    {
                        "role": "bulk_parent_candidate",
                        "path": candidate_path.name,
                        "sha256": _sha256(candidate_path),
                    },
                    {
                        "role": "acceptance_report",
                        "path": acceptance_path.name,
                        "sha256": _sha256(acceptance_path),
                    },
                    {
                        "role": "progress",
                        "path": progress_path.name,
                        "sha256": _sha256(progress_path),
                    },
                ],
            }
        )
        write_manifest(manifest_path, manifest)
        return manifest
    except Exception as exc:
        if trajectory is not None:
            trajectory.close()
        manifest.update(
            {
                "status": "failed",
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        write_manifest(manifest_path, manifest)
        raise
    finally:
        if calculator is not None:
            atoms.calc = None


def execute_surface_relaxation(
    plan: dict,
    output_root: Path,
    *,
    device: str = "cuda",
) -> dict:
    """Relax one immutable symmetric-slab thickness series with MACE."""

    if plan.get("mode") != "surface_relaxation_plan":
        raise ValueError("Surface relaxation requires a resolved relaxation plan")
    if not plan["execution_contract"]["executable"]:
        raise ValueError("Surface relaxation plan has unresolved readiness issues")
    try:
        from ase.constraints import FixCom
        from ase.io.trajectory import Trajectory
        from ase.optimize import LBFGS
    except ImportError as exc:
        raise RuntimeError(
            "Surface relaxation requires a complete ASE installation"
        ) from exc

    contract = plan["surface_relaxation_contract"]
    default_dtype = plan["potential"]["default_dtype"]
    identity = {
        "plan_fingerprint": plan["fingerprint"],
        "device": device,
        "default_dtype": default_dtype,
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = Path(output_root).expanduser().resolve()
    run_directory = (
        output_root
        / plan["substrate"]["key"].casefold()
        / f"{contract['protocol_version']}-{timestamp}-{fingerprint[:12]}"
    )
    if run_directory.exists():
        raise ValueError(
            f"Immutable surface relaxation run already exists: {run_directory}"
        )
    run_directory.mkdir(parents=True)

    plan_path = run_directory / "surface-relaxation-plan.json"
    bulk_reference_path = run_directory / "bulk-reference.json"
    convergence_path = run_directory / "convergence-report.json"
    manifest_path = run_directory / "run-manifest.json"
    write_manifest(plan_path, plan)
    manifest = {
        "schema_version": 1,
        "status": "running",
        "fingerprint": fingerprint,
        "plan_fingerprint": plan["fingerprint"],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_directory": str(run_directory),
        "host": socket.gethostname(),
        "python": sys.version.split()[0],
        "device": device,
        "default_dtype": default_dtype,
        "plan": {"path": plan_path.name, "sha256": _sha256(plan_path)},
        "artifacts": [],
    }
    write_manifest(manifest_path, manifest)
    print(f"SURFACE_RUN_DIRECTORY {run_directory}", flush=True)

    calculator = None
    active_atoms = None
    active_trajectory = None
    try:
        calculator, runtime = _make_mace_calculator(
            Path(plan["potential"]["model_path"]), device, default_dtype
        )
        bulk_atoms = read(plan["bulk_parent"]["path"])
        bulk_atoms.calc = calculator
        bulk_energy = float(bulk_atoms.get_potential_energy())
        bulk_forces = np.asarray(bulk_atoms.get_forces(), dtype=float)
        bulk_fmax = float(np.max(np.linalg.norm(bulk_forces, axis=1)))
        parent_formula_units = int(plan["bulk_reference"]["parent_formula_units"])
        bulk_energy_per_formula = bulk_energy / parent_formula_units
        bulk_reference = {
            "schema_version": 1,
            "structure": {
                "path": plan["bulk_parent"]["path"],
                "sha256": plan["bulk_parent"]["sha256"],
            },
            "model_sha256": plan["potential"]["model_sha256"],
            "default_dtype": default_dtype,
            "device": device,
            "reduced_formula": plan["bulk_reference"]["reduced_formula"],
            "parent_formula_units": parent_formula_units,
            "potential_energy_eV": bulk_energy,
            "potential_energy_per_formula_unit_eV": bulk_energy_per_formula,
            "maximum_force_eV_A": bulk_fmax,
            "note": (
                "Fresh single-point reference evaluated in this run with the same "
                "calculator instance used for every slab."
            ),
        }
        write_manifest(bulk_reference_path, bulk_reference)
        bulk_atoms.calc = None

        parent_atoms = read(plan["bulk_parent"]["path"])
        _, candidates, atoms_by_repeat = _surface_relaxation_candidates(
            parent_atoms,
            plan["surface_precut_contract"],
            contract,
            float(plan["registered_surface_precut"]["cut_fractional_phase"]),
        )
        if candidates != plan["thickness_candidates"]:
            raise ValueError("Surface thickness candidates changed after planning")

        results = []
        artifacts = [
            {
                "role": "bulk_single_point_reference",
                "path": bulk_reference_path.name,
                "sha256": _sha256(bulk_reference_path),
            }
        ]
        for candidate in candidates:
            repeat = int(candidate["oriented_unit_repeats"])
            repeat_directory = run_directory / f"repeat-{repeat:03d}"
            repeat_directory.mkdir()
            initial_path = repeat_directory / "initial.cif"
            relaxed_path = repeat_directory / "relaxed.cif"
            trajectory_path = repeat_directory / "optimization.traj"
            optimizer_log_path = repeat_directory / "optimizer.log"
            progress_path = repeat_directory / "progress.jsonl"
            report_path = repeat_directory / "relaxation-report.json"
            atoms = atoms_by_repeat[repeat]
            active_atoms = atoms
            initial_cell = np.asarray(atoms.cell, dtype=float).copy()
            initial_positions = np.asarray(atoms.positions, dtype=float).copy()
            write_clean_structure(atoms, initial_path)
            atoms.set_constraint(FixCom())
            atoms.calc = calculator
            initial_energy = float(atoms.get_potential_energy())
            optimizer = LBFGS(
                atoms,
                logfile=str(optimizer_log_path),
                maxstep=float(contract["maximum_step_A"]),
            )
            active_trajectory = Trajectory(str(trajectory_path), "w", atoms)
            optimizer.attach(active_trajectory.write, interval=1)

            def record_progress(
                *,
                _atoms=atoms,
                _optimizer=optimizer,
                _progress_path=progress_path,
                _repeat=repeat,
            ):
                forces = np.asarray(_atoms.get_forces(), dtype=float)
                record = {
                    "optimizer_step": int(_optimizer.get_number_of_steps()),
                    "time_utc": datetime.now(timezone.utc).isoformat(),
                    "energy_eV": float(_atoms.get_potential_energy()),
                    "maximum_force_eV_A": float(np.max(np.linalg.norm(forces, axis=1))),
                }
                with _progress_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                print(
                    "SURFACE_PROGRESS "
                    f"repeat={_repeat} step={record['optimizer_step']} "
                    f"fmax={record['maximum_force_eV_A']:.6f}eV/A",
                    flush=True,
                )

            optimizer.attach(record_progress, interval=1)
            record_progress()
            converged = bool(
                optimizer.run(
                    fmax=float(contract["force_tolerance_eV_A"]),
                    steps=int(contract["maximum_steps"]),
                )
            )
            active_trajectory.close()
            active_trajectory = None
            final_energy = float(atoms.get_potential_energy())
            final_forces = np.asarray(atoms.get_forces(), dtype=float)
            final_fmax = float(np.max(np.linalg.norm(final_forces, axis=1)))
            final_cell = np.asarray(atoms.cell, dtype=float)
            displacement = np.asarray(atoms.positions, dtype=float) - initial_positions
            maximum_displacement = float(np.max(np.linalg.norm(displacement, axis=1)))
            area = float(candidate["cell_A"]["surface_area_A2"])
            formula_units = int(candidate["bulk_formula_units"])
            surface_energy = (
                final_energy - formula_units * bulk_energy_per_formula
            ) / (float(contract["surface_count"]) * area)
            checks = {
                "optimizer_converged": {
                    "passed": converged,
                    "observed_steps": int(optimizer.get_number_of_steps()),
                    "maximum_steps": int(contract["maximum_steps"]),
                },
                "force": {
                    "passed": final_fmax <= float(contract["force_tolerance_eV_A"]),
                    "observed_eV_A": final_fmax,
                    "maximum_eV_A": float(contract["force_tolerance_eV_A"]),
                },
                "formula": {
                    "passed": _formula(atoms.get_chemical_symbols())
                    == candidate["formula"],
                    "observed": _formula(atoms.get_chemical_symbols()),
                    "expected": candidate["formula"],
                },
                "atom_count": {
                    "passed": len(atoms) == candidate["atom_count"],
                    "observed": len(atoms),
                    "expected": candidate["atom_count"],
                },
                "fixed_cell": {
                    "passed": bool(
                        np.allclose(final_cell, initial_cell, atol=1.0e-10, rtol=0.0)
                    ),
                    "maximum_absolute_change_A": float(
                        np.max(np.abs(final_cell - initial_cell))
                    ),
                },
                "periodicity": {
                    "passed": bool(np.all(np.asarray(atoms.pbc, dtype=bool))),
                    "observed": [bool(value) for value in atoms.pbc],
                },
            }
            result = {
                "schema_version": 1,
                "status": (
                    "passed"
                    if all(item["passed"] for item in checks.values())
                    else "failed"
                ),
                "oriented_unit_repeats": repeat,
                "formula": candidate["formula"],
                "atom_count": len(atoms),
                "bulk_formula_units": formula_units,
                "surface_area_A2": area,
                "initial_energy_eV": initial_energy,
                "relaxed_energy_eV": final_energy,
                "relaxation_energy_eV": final_energy - initial_energy,
                "surface_energy_eV_A2": surface_energy,
                "surface_energy_J_m2": surface_energy * J_M2_PER_EV_A2,
                "maximum_force_eV_A": final_fmax,
                "maximum_atomic_displacement_A": maximum_displacement,
                "checks": checks,
            }
            atoms.set_constraint()
            write_clean_structure(atoms, relaxed_path)
            write_manifest(report_path, result)
            atoms.calc = None
            active_atoms = None
            results.append(result)
            for role, path in (
                ("initial_surface_slab", initial_path),
                ("relaxed_surface_slab", relaxed_path),
                ("surface_optimization_trajectory", trajectory_path),
                ("surface_optimizer_log", optimizer_log_path),
                ("surface_progress", progress_path),
                ("surface_relaxation_report", report_path),
            ):
                artifacts.append(
                    {
                        "role": role,
                        "oriented_unit_repeats": repeat,
                        "path": str(path.relative_to(run_directory)),
                        "sha256": _sha256(path),
                    }
                )

        convergence = evaluate_surface_relaxation_convergence(plan, results)
        write_manifest(convergence_path, convergence)
        artifacts.append(
            {
                "role": "surface_convergence_report",
                "path": convergence_path.name,
                "sha256": _sha256(convergence_path),
            }
        )
        manifest.update(
            {
                "status": (
                    "passed"
                    if convergence["status"] == "passed"
                    else "failed_acceptance"
                ),
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "runtime": runtime,
                "bulk_reference": bulk_reference,
                "convergence_status": convergence["status"],
                "artifacts": artifacts,
            }
        )
        write_manifest(manifest_path, manifest)
        return manifest
    except Exception as exc:
        if active_trajectory is not None:
            active_trajectory.close()
        if active_atoms is not None:
            active_atoms.calc = None
        manifest.update(
            {
                "status": "failed",
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        write_manifest(manifest_path, manifest)
        raise


def execute_surface_termination_screen(
    plan: dict,
    output_root: Path,
    *,
    device: str = "cuda",
) -> dict:
    """Classify every cut phase, then relax and rank directly comparable pairs."""

    if plan.get("mode") != "surface_termination_screen_plan":
        raise ValueError("Termination screening requires a resolved screening plan")
    if not plan["execution_contract"]["executable"]:
        raise ValueError("Termination screening plan has unresolved readiness issues")
    identity = {
        "plan_fingerprint": plan["fingerprint"],
        "device": device,
        "default_dtype": plan["potential"]["default_dtype"],
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    contract = plan["termination_screening_contract"]
    output_root = Path(output_root).expanduser().resolve()
    run_directory = (
        output_root
        / plan["substrate"]["key"].casefold()
        / f"{contract['protocol_version']}-{timestamp}-{fingerprint[:12]}"
    )
    if run_directory.exists():
        raise ValueError(f"Immutable termination screening run already exists: {run_directory}")
    run_directory.mkdir(parents=True)
    plan_path = run_directory / "surface-termination-screen-plan.json"
    selection_path = run_directory / "termination-selection-report.json"
    manifest_path = run_directory / "run-manifest.json"
    write_manifest(plan_path, plan)
    manifest = {
        "schema_version": 1,
        "status": "running",
        "fingerprint": fingerprint,
        "plan_fingerprint": plan["fingerprint"],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_directory": str(run_directory),
        "host": socket.gethostname(),
        "device": device,
        "default_dtype": plan["potential"]["default_dtype"],
        "plan": {"path": plan_path.name, "sha256": _sha256(plan_path)},
        "artifacts": [],
    }
    write_manifest(manifest_path, manifest)
    print(f"TERMINATION_SCREEN_RUN_DIRECTORY {run_directory}", flush=True)

    try:
        results = []
        artifacts = []
        child_root = run_directory / "termination-runs"
        for termination in plan["termination_candidates"]:
            candidate_id = termination["candidate_id"]
            print(
                "TERMINATION_SCREEN_START "
                f"candidate={candidate_id} "
                f"cut={termination['cut_fractional_phase']:.12g} "
                f"classification={termination['classification']}",
                flush=True,
            )
            if termination["classification"] != "directly_rankable_nonpolar_pair":
                results.append(
                    {
                        "candidate_id": candidate_id,
                        "cut_fractional_phase": termination[
                            "cut_fractional_phase"
                        ],
                        "top_plane_formula": termination["top_plane_formula"],
                        "bottom_plane_formula": termination[
                            "bottom_plane_formula"
                        ],
                        "formal_dipole_density_e_per_A": termination[
                            "formal_dipole_density_e_per_A"
                        ],
                        "classification": termination["classification"],
                        "individual_surface_energy_interpretation": termination[
                            "individual_surface_energy_interpretation"
                        ],
                        "calculation_status": (
                            "not_run_requires_compensation_or_symmetric_slab_protocol"
                        ),
                        "relaxations_passed": None,
                        "thickness_converged": None,
                        "rankable": False,
                        "thickest_repeat": None,
                        "pair_average_surface_energy_eV_A2": None,
                        "pair_average_surface_energy_J_m2": None,
                        "child_run": None,
                    }
                )
                print(
                    "TERMINATION_SCREEN_SKIP_ENERGY "
                    f"candidate={candidate_id} "
                    "reason=requires_compensation_or_symmetric_slab_protocol",
                    flush=True,
                )
                continue
            subplan = _surface_relaxation_subplan_for_cut(
                plan["base_surface_relaxation_plan"], termination
            )
            child_manifest = execute_surface_relaxation(
                subplan,
                child_root / candidate_id,
                device=device,
            )
            child_directory = Path(child_manifest["output_directory"])
            child_manifest_path = child_directory / "run-manifest.json"
            convergence_path = child_directory / "convergence-report.json"
            if not convergence_path.is_file():
                raise ValueError(
                    f"Termination {candidate_id} did not produce a convergence report"
                )
            convergence = json.loads(convergence_path.read_text(encoding="utf-8"))
            thickest = max(
                convergence["thickness_series"],
                key=lambda item: item["oriented_unit_repeats"],
            )
            relaxations_passed = convergence["checks"]["all_relaxations_passed"][
                "passed"
            ]
            thickness_converged = convergence["checks"][
                "two_thickest_surface_energy"
            ]["passed"]
            pre_rankable = (
                termination["classification"]
                == "directly_rankable_nonpolar_pair"
            )
            result = {
                "candidate_id": candidate_id,
                "cut_fractional_phase": termination["cut_fractional_phase"],
                "top_plane_formula": termination["top_plane_formula"],
                "bottom_plane_formula": termination["bottom_plane_formula"],
                "formal_dipole_density_e_per_A": termination[
                    "formal_dipole_density_e_per_A"
                ],
                "classification": termination["classification"],
                "individual_surface_energy_interpretation": termination[
                    "individual_surface_energy_interpretation"
                ],
                "calculation_status": "completed",
                "relaxations_passed": relaxations_passed,
                "thickness_converged": thickness_converged,
                "rankable": bool(
                    pre_rankable and relaxations_passed and thickness_converged
                ),
                "thickest_repeat": thickest["oriented_unit_repeats"],
                "pair_average_surface_energy_eV_A2": thickest[
                    "surface_energy_eV_A2"
                ],
                "pair_average_surface_energy_J_m2": thickest[
                    "surface_energy_J_m2"
                ],
                "child_run": {
                    "status": child_manifest["status"],
                    "output_directory": str(child_directory),
                    "run_manifest_sha256": _sha256(child_manifest_path),
                    "convergence_report_sha256": _sha256(convergence_path),
                },
            }
            results.append(result)
            artifacts.extend(
                [
                    {
                        "role": "termination_child_run_manifest",
                        "candidate_id": candidate_id,
                        "path": str(child_manifest_path.relative_to(run_directory)),
                        "sha256": _sha256(child_manifest_path),
                    },
                    {
                        "role": "termination_child_convergence_report",
                        "candidate_id": candidate_id,
                        "path": str(convergence_path.relative_to(run_directory)),
                        "sha256": _sha256(convergence_path),
                    },
                ]
            )
        diagnostic_ranking = sorted(
            [
                item
                for item in results
                if item["pair_average_surface_energy_eV_A2"] is not None
            ],
            key=lambda item: (
                item["pair_average_surface_energy_eV_A2"],
                item["candidate_id"],
            ),
        )
        rankable = [item for item in diagnostic_ranking if item["rankable"]]
        selected = rankable[0] if rankable else None
        required_results = [
            item
            for item in results
            if item["classification"] == "directly_rankable_nonpolar_pair"
        ]
        all_required_relaxations_passed = bool(required_results) and all(
            item["relaxations_passed"] for item in required_results
        )
        selection_status = (
            "passed_selection"
            if all_required_relaxations_passed and selected is not None
            else "failed_selection"
        )
        selection_summary = _surface_termination_selection_summary(
            results,
            selected,
        )
        selection = {
            "schema_version": 1,
            "status": selection_status,
            "selected_candidate_id": (
                selected["candidate_id"] if selected is not None else None
            ),
            "selection_rule": (
                "Among stoichiometric, charge-neutral, formally nonpolar, "
                "force-converged and thickness-converged pairs, choose the lowest "
                "thick-limit surface energy."
            ),
            "energy_interpretation": plan["energy_interpretation"],
            "candidate_results": results,
            "diagnostic_pair_average_energy_order": [
                item["candidate_id"] for item in diagnostic_ranking
            ],
            "rankable_energy_order": [item["candidate_id"] for item in rankable],
            "selection_summary": selection_summary,
        }
        write_manifest(selection_path, selection)
        artifacts.append(
            {
                "role": "termination_selection_report",
                "path": selection_path.name,
                "sha256": _sha256(selection_path),
            }
        )
        manifest.update(
            {
                "status": selection_status,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "selected_candidate_id": selection["selected_candidate_id"],
                "all_required_relaxations_passed": (
                    all_required_relaxations_passed
                ),
                "selection_summary": selection_summary,
                "artifacts": artifacts,
            }
        )
        write_manifest(manifest_path, manifest)
        return manifest
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        write_manifest(manifest_path, manifest)
        raise


def _surface_termination_selection_summary(
    results: list[dict],
    selected: dict | None,
) -> dict:
    directly_comparable = [
        item
        for item in results
        if item.get("classification") == "directly_rankable_nonpolar_pair"
    ]
    relaxed = [
        item for item in results if item.get("calculation_status") == "completed"
    ]
    rankable = sorted(
        [item for item in results if item.get("rankable") is True],
        key=lambda item: (
            item["pair_average_surface_energy_eV_A2"],
            item["candidate_id"],
        ),
    )
    if selected is None:
        basis = "no_candidate_passed_all_selection_gates"
        selected_energy = None
    else:
        selected_energy = float(selected["pair_average_surface_energy_eV_A2"])
        if len(directly_comparable) == 1:
            basis = "sole_directly_comparable_candidate_passed_relaxation_and_convergence"
        elif len(rankable) == 1:
            basis = "sole_converged_candidate_among_multiple_directly_comparable_candidates"
        else:
            basis = "lowest_converged_surface_energy_among_multiple_candidates"
    energy_comparison = []
    for item in rankable:
        energy = float(item["pair_average_surface_energy_eV_A2"])
        delta = None if selected_energy is None else energy - selected_energy
        energy_comparison.append(
            {
                "candidate_id": item["candidate_id"],
                "pair_average_surface_energy_eV_A2": energy,
                "pair_average_surface_energy_J_m2": float(
                    item["pair_average_surface_energy_J_m2"]
                ),
                "energy_above_selected_eV_A2": delta,
                "energy_above_selected_J_m2": (
                    None if delta is None else delta * J_M2_PER_EV_A2
                ),
                "selected": bool(
                    selected is not None
                    and item["candidate_id"] == selected["candidate_id"]
                ),
            }
        )
    directly_comparable_status = []
    for item in directly_comparable:
        if item.get("relaxations_passed") is not True:
            outcome = "excluded_relaxation_not_converged"
        elif item.get("thickness_converged") is not True:
            outcome = "excluded_thickness_not_converged"
        elif item.get("rankable") is True:
            outcome = "ranked"
        else:
            outcome = "excluded_other_gate"
        directly_comparable_status.append(
            {
                "candidate_id": item["candidate_id"],
                "outcome": outcome,
                "pair_average_surface_energy_eV_A2": item.get(
                    "pair_average_surface_energy_eV_A2"
                ),
            }
        )
    excluded_counts = Counter(
        item.get("classification", "unknown")
        for item in results
        if item not in directly_comparable
    )
    return {
        "enumerated_candidate_count": len(results),
        "directly_comparable_candidate_count": len(directly_comparable),
        "relaxed_candidate_count": len(relaxed),
        "rankable_candidate_count": len(rankable),
        "excluded_before_relaxation_count": len(results) - len(directly_comparable),
        "excluded_before_relaxation_by_classification": dict(
            sorted(excluded_counts.items())
        ),
        "selected_candidate_id": (
            None if selected is None else selected["candidate_id"]
        ),
        "selection_basis": basis,
        "directly_comparable_candidate_status": directly_comparable_status,
        "rankable_energy_comparison": energy_comparison,
        "reporting_requirement": (
            "Always report enumeration/comparability/relaxation counts. When multiple "
            "candidates are rankable, report every converged energy and its difference "
            "above the selected minimum; when only one is comparable, state that it "
            "was not selected by an energy competition."
        ),
    }


def _verified_run_artifact(
    run_root: Path,
    artifact: dict,
    *,
    role: str | None = None,
) -> Path:
    if role is not None and artifact.get("role") != role:
        raise ValueError(f"Expected artifact role {role!r}")
    path_value = artifact.get("path")
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError("Run artifact path must be non-empty")
    root = run_root.resolve()
    path = (root / path_value).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Run artifact escapes its immutable root: {path}") from exc
    if not path.is_file() or _sha256(path) != artifact.get("sha256"):
        raise ValueError(f"Run artifact is missing or changed: {path}")
    return path


def _one_artifact(
    artifacts: list[dict],
    role: str,
    *,
    candidate_id: str | None = None,
    repeat: int | None = None,
) -> dict:
    matches = [item for item in artifacts if item.get("role") == role]
    if candidate_id is not None:
        matches = [
            item for item in matches if item.get("candidate_id") == candidate_id
        ]
    if repeat is not None:
        matches = [
            item
            for item in matches
            if item.get("oriented_unit_repeats") == repeat
        ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {role!r} artifact; found {len(matches)}"
        )
    return matches[0]


def resolve_surface_promotion_plan(
    substrate_name: str,
    screen_run_directory: Path,
    target_repeat: int,
    structure_directory: Path | None = None,
) -> dict:
    """Validate one termination-screen result for maintained-surface promotion."""

    target_repeat = _positive_int(target_repeat, "target_repeat")
    manifest = _load_structure_manifest(substrate_name, structure_directory)
    manifest_path = Path(manifest["manifest_path"])
    parent = _validated_registered_bulk_parent(manifest)
    if parent is None:
        raise ValueError(f"No accepted Bulk Parent for {substrate_name!r}")
    precut_contract = _validated_surface_precut_contract(manifest)
    relaxation_contract = _validated_surface_relaxation_contract(manifest)
    promotion_contract = _validated_surface_promotion_contract(relaxation_contract)
    registered = _selected_registered_surface_precut(manifest, precut_contract)
    if registered.get("status") != "accepted_bulk_derived_unrelaxed_precut":
        raise ValueError("The maintained surface is already a relaxed promotion")
    if target_repeat != int(registered["source_layers"]):
        raise ValueError(
            "Promotion must preserve the registered public thickness; expected "
            f"{registered['source_layers']} repeats"
        )

    screen_root = Path(screen_run_directory).expanduser().resolve()
    if not screen_root.is_dir():
        raise ValueError(f"Termination screen run directory does not exist: {screen_root}")
    screen_manifest_path = screen_root / "run-manifest.json"
    if not screen_manifest_path.is_file():
        raise ValueError("Termination screen run-manifest.json is missing")
    screen_manifest = json.loads(screen_manifest_path.read_text(encoding="utf-8"))
    if (
        screen_manifest.get("status") != "passed_selection"
        or screen_manifest.get("all_required_relaxations_passed") is not True
    ):
        raise ValueError("Termination screen did not pass selection")
    screen_plan_path = _verified_run_artifact(
        screen_root,
        screen_manifest.get("plan") or {},
    )
    screen_plan = json.loads(screen_plan_path.read_text(encoding="utf-8"))
    if (
        screen_plan.get("mode") != "surface_termination_screen_plan"
        or (screen_plan.get("substrate") or {}).get("key")
        != manifest["substrate_key"]
        or screen_manifest.get("plan_fingerprint") != screen_plan.get("fingerprint")
        or (screen_plan.get("bulk_parent") or {}).get("sha256") != parent["sha256"]
        or (
            (screen_plan.get("base_surface_relaxation_plan") or {}).get(
                "registered_surface_precut"
            )
            or {}
        ).get("sha256")
        != registered["sha256"]
    ):
        raise ValueError("Termination screen plan is stale for the maintained substrate")

    screen_artifacts = screen_manifest.get("artifacts") or []
    selection_artifact = _one_artifact(
        screen_artifacts, "termination_selection_report"
    )
    selection_path = _verified_run_artifact(
        screen_root,
        selection_artifact,
        role="termination_selection_report",
    )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected_id = selection.get("selected_candidate_id")
    selected_results = [
        item
        for item in selection.get("candidate_results", [])
        if item.get("candidate_id") == selected_id
    ]
    if (
        selection.get("status") != "passed_selection"
        or selected_id != screen_manifest.get("selected_candidate_id")
        or len(selected_results) != 1
        or selected_results[0].get("rankable") is not True
        or selected_results[0].get("relaxations_passed") is not True
        or selected_results[0].get("thickness_converged") is not True
    ):
        raise ValueError("Termination selection report is invalid or incomplete")

    child_manifest_artifact = _one_artifact(
        screen_artifacts,
        "termination_child_run_manifest",
        candidate_id=selected_id,
    )
    child_manifest_path = _verified_run_artifact(
        screen_root,
        child_manifest_artifact,
        role="termination_child_run_manifest",
    )
    child_root = child_manifest_path.parent
    child_manifest = json.loads(child_manifest_path.read_text(encoding="utf-8"))
    if (
        child_manifest.get("status") != "passed"
        or child_manifest.get("convergence_status") != "passed"
    ):
        raise ValueError("Selected termination relaxation run did not pass")
    child_plan_path = _verified_run_artifact(
        child_root,
        child_manifest.get("plan") or {},
    )
    child_plan = json.loads(child_plan_path.read_text(encoding="utf-8"))
    if (
        child_manifest.get("plan_fingerprint") != child_plan.get("fingerprint")
        or (child_plan.get("termination_screening_candidate") or {}).get(
            "candidate_id"
        )
        != selected_id
        or (child_plan.get("bulk_parent") or {}).get("sha256") != parent["sha256"]
    ):
        raise ValueError("Selected termination relaxation plan is stale")

    child_artifacts = child_manifest.get("artifacts") or []
    convergence_artifact = _one_artifact(
        child_artifacts, "surface_convergence_report"
    )
    convergence_path = _verified_run_artifact(
        child_root,
        convergence_artifact,
        role="surface_convergence_report",
    )
    convergence = json.loads(convergence_path.read_text(encoding="utf-8"))
    if (
        convergence.get("status") != "passed"
        or (convergence.get("checks") or {})
        .get("all_relaxations_passed", {})
        .get("passed")
        is not True
        or (convergence.get("checks") or {})
        .get("two_thickest_surface_energy", {})
        .get("passed")
        is not True
    ):
        raise ValueError("Selected termination thickness series did not converge")

    evidence_specs = [
        ("termination_screen_plan", screen_plan_path, "termination-screen-plan.json"),
        (
            "termination_selection_report",
            selection_path,
            "termination-selection-report.json",
        ),
        (
            "termination_screen_run_manifest",
            screen_manifest_path,
            "termination-screen-run-manifest.json",
        ),
        ("surface_relaxation_plan", child_plan_path, "surface-relaxation-plan.json"),
        (
            "surface_relaxation_run_manifest",
            child_manifest_path,
            "surface-relaxation-run-manifest.json",
        ),
        (
            "surface_convergence_report",
            convergence_path,
            "surface-convergence-report.json",
        ),
    ]
    artifact_destinations = {
        "bulk_single_point_reference": ("bulk_reference", "bulk-reference.json"),
        "surface_relaxation_report": (
            "target_relaxation_report",
            f"repeat-{target_repeat:03d}-relaxation-report.json",
        ),
        "surface_optimizer_log": (
            "target_optimizer_log",
            f"repeat-{target_repeat:03d}-optimizer.log",
        ),
        "surface_progress": (
            "target_progress",
            f"repeat-{target_repeat:03d}-progress.jsonl",
        ),
    }
    for source_role, (record_role, destination_name) in artifact_destinations.items():
        repeat = None if source_role == "bulk_single_point_reference" else target_repeat
        artifact = _one_artifact(child_artifacts, source_role, repeat=repeat)
        source = _verified_run_artifact(child_root, artifact, role=source_role)
        evidence_specs.append((record_role, source, destination_name))

    initial_artifact = _one_artifact(
        child_artifacts, "initial_surface_slab", repeat=target_repeat
    )
    _verified_run_artifact(child_root, initial_artifact, role="initial_surface_slab")
    if initial_artifact.get("sha256") != registered["sha256"]:
        raise ValueError(
            "Target-thickness initial slab is not the registered public Surface Precut"
        )
    relaxed_artifact = _one_artifact(
        child_artifacts, "relaxed_surface_slab", repeat=target_repeat
    )
    relaxed_path = _verified_run_artifact(
        child_root, relaxed_artifact, role="relaxed_surface_slab"
    )
    target_report_path = next(
        source for role, source, _ in evidence_specs if role == "target_relaxation_report"
    )
    target_report = json.loads(target_report_path.read_text(encoding="utf-8"))
    if (
        target_report.get("status") != "passed"
        or target_report.get("oriented_unit_repeats") != target_repeat
        or not target_report.get("checks")
        or not all(
            check.get("passed") is True
            for check in target_report["checks"].values()
        )
    ):
        raise ValueError("Target-thickness relaxation report did not pass every gate")
    relaxed_atoms = read(relaxed_path)
    target_formula = _formula(relaxed_atoms.get_chemical_symbols())
    if (
        target_formula != dict(sorted(target_report["formula"].items()))
        or len(relaxed_atoms) != target_report["atom_count"]
        or not np.all(np.asarray(relaxed_atoms.pbc, dtype=bool))
    ):
        raise ValueError("Target relaxed surface identity is inconsistent")
    geometry = _surface_candidate_summary(
        relaxed_atoms,
        precut_contract,
        float(registered["cut_fractional_phase"]),
    )
    if (
        geometry["checks"]["formal_charge_neutral"] is not True
        or geometry["checks"]["formal_nonpolar"] is not True
        or geometry["top_plane_formula"] != geometry["bottom_plane_formula"]
    ):
        raise ValueError(
            "Relaxed target no longer satisfies neutral, nonpolar, equivalent-surface gates"
        )
    destination_directory = (
        manifest_path.parent / promotion_contract["destination_directory"]
    ).resolve()
    try:
        destination_directory.relative_to(manifest_path.parent.resolve())
    except ValueError as exc:
        raise ValueError("Promotion destination escapes the structure library") from exc
    destination_structure = (
        destination_directory / promotion_contract["structure_filename"]
    )
    readiness_issues = []
    if destination_directory.exists():
        readiness_issues.append(
            {
                "code": "promotion_destination_exists",
                "message": f"Immutable promotion destination exists: {destination_directory}",
            }
        )
    evidence = [
        {
            "role": role,
            "source_path": str(source),
            "source_sha256": _sha256(source),
            "destination_name": destination_name,
        }
        for role, source, destination_name in evidence_specs
    ]
    identity = {
        "substrate_key": manifest["substrate_key"],
        "bulk_parent_sha256": parent["sha256"],
        "source_precut_sha256": registered["sha256"],
        "screen_run_fingerprint": screen_manifest["fingerprint"],
        "selection_report_sha256": _sha256(selection_path),
        "target_repeat": target_repeat,
        "target_surface_sha256": _sha256(relaxed_path),
        "promotion_contract": promotion_contract,
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema_version": 1,
        "mode": "surface_promotion_plan",
        "fingerprint": fingerprint,
        "substrate": {
            "key": manifest["substrate_key"],
            "host_material": manifest["host_material"],
            "structure_manifest": {
                "path": str(manifest_path),
                "sha256": _sha256(manifest_path),
            },
        },
        "bulk_parent": parent,
        "promotion_contract": promotion_contract,
        "source_surface_precut": {
            "path": str((manifest_path.parent / registered["path"]).resolve()),
            "sha256": registered["sha256"],
            "status": registered["status"],
            "source_layers": int(registered["source_layers"]),
            "miller": list(registered["miller"]),
            "primitive_miller": list(registered["primitive_miller"]),
            "termination": registered["termination"],
            "cut_fractional_phase": float(registered["cut_fractional_phase"]),
        },
        "screen_result": {
            "run_directory": str(screen_root),
            "run_fingerprint": screen_manifest["fingerprint"],
            "selected_candidate_id": selected_id,
            "plan_sha256": _sha256(screen_plan_path),
            "selection_report_sha256": _sha256(selection_path),
            "run_manifest_sha256": _sha256(screen_manifest_path),
        },
        "target_surface": {
            "source_path": str(relaxed_path),
            "source_sha256": _sha256(relaxed_path),
            "oriented_unit_repeats": target_repeat,
            "formula": target_formula,
            "atom_count": len(relaxed_atoms),
            "cell_A": geometry["cell_A"],
            "solid_thickness_A": geometry["solid_thickness_A"],
            "vacuum_A": geometry["vacuum_A"],
            "top_plane_formula": geometry["top_plane_formula"],
            "bottom_plane_formula": geometry["bottom_plane_formula"],
            "formal_dipole_density_e_per_A": geometry[
                "formal_dipole_density_e_per_A"
            ],
            "surface_energy_eV_A2": target_report["surface_energy_eV_A2"],
            "surface_energy_J_m2": target_report["surface_energy_J_m2"],
            "maximum_force_eV_A": target_report["maximum_force_eV_A"],
            "maximum_atomic_displacement_A": target_report[
                "maximum_atomic_displacement_A"
            ],
            "destination_directory": str(destination_directory),
            "destination_path": str(destination_structure),
        },
        "thickness_convergence": convergence["checks"][
            "two_thickest_surface_energy"
        ],
        "evidence": evidence,
        "execution_contract": {
            "read_only": True,
            "executable": not readiness_issues,
            "execution_command": "surface-promotion-run",
            "readiness_issues": readiness_issues,
            "mutation": (
                "Create one immutable maintained-surface package, supersede but do "
                "not delete the geometric precut, and update the structure manifest."
            ),
        },
    }


def execute_surface_promotion(plan: dict) -> dict:
    """Copy verified evidence and atomically register one relaxed maintained surface."""

    if plan.get("mode") != "surface_promotion_plan":
        raise ValueError("Surface promotion requires a resolved promotion plan")
    if not plan["execution_contract"]["executable"]:
        raise ValueError("Surface promotion plan has unresolved readiness issues")
    manifest_path = Path(plan["substrate"]["structure_manifest"]["path"])
    if (
        not manifest_path.is_file()
        or _sha256(manifest_path)
        != plan["substrate"]["structure_manifest"]["sha256"]
    ):
        raise ValueError("Structure manifest changed after promotion planning")
    source_surface = Path(plan["target_surface"]["source_path"])
    if (
        not source_surface.is_file()
        or _sha256(source_surface) != plan["target_surface"]["source_sha256"]
    ):
        raise ValueError("Selected relaxed surface changed after promotion planning")
    for record in plan["evidence"]:
        source = Path(record["source_path"])
        if not source.is_file() or _sha256(source) != record["source_sha256"]:
            raise ValueError(f"Promotion evidence changed after planning: {source}")

    destination_directory = Path(
        plan["target_surface"]["destination_directory"]
    )
    destination_structure = Path(plan["target_surface"]["destination_path"])
    if destination_directory.exists():
        raise ValueError(
            f"Immutable promotion destination already exists: {destination_directory}"
        )
    destination_directory.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    with TemporaryDirectory(
        prefix=f".{destination_directory.name}-", dir=destination_directory.parent
    ) as temporary_value:
        temporary = Path(temporary_value)
        staged_structure = temporary / destination_structure.name
        shutil.copyfile(source_surface, staged_structure)
        records_directory = temporary / "records"
        records_directory.mkdir()
        copied_records = []
        for record in plan["evidence"]:
            copied = records_directory / record["destination_name"]
            shutil.copyfile(Path(record["source_path"]), copied)
            copied_records.append(
                {
                    "role": record["role"],
                    "path": str(copied.relative_to(temporary)),
                    "sha256": _sha256(copied),
                }
            )
        promotion_plan_path = records_directory / "surface-promotion-plan.json"
        write_manifest(promotion_plan_path, plan)
        promotion_report = {
            "schema_version": 1,
            "status": "passed_promotion",
            "protocol_version": plan["promotion_contract"]["protocol_version"],
            "plan_fingerprint": plan["fingerprint"],
            "promoted_at_utc": timestamp,
            "substrate_key": plan["substrate"]["key"],
            "source_surface_precut": plan["source_surface_precut"],
            "screen_result": plan["screen_result"],
            "registered_surface": {
                "path": str(destination_structure),
                "sha256": _sha256(staged_structure),
                "oriented_unit_repeats": plan["target_surface"][
                    "oriented_unit_repeats"
                ],
            },
        }
        promotion_report_path = records_directory / "surface-promotion-report.json"
        write_manifest(promotion_report_path, promotion_report)
        copied_records.extend(
            [
                {
                    "role": "surface_promotion_plan",
                    "path": str(promotion_plan_path.relative_to(temporary)),
                    "sha256": _sha256(promotion_plan_path),
                },
                {
                    "role": "surface_promotion_report",
                    "path": str(promotion_report_path.relative_to(temporary)),
                    "sha256": _sha256(promotion_report_path),
                },
            ]
        )
        temporary.replace(destination_directory)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_matches = [
        item
        for item in manifest.get("precut_slabs", [])
        if item.get("sha256") == plan["source_surface_precut"]["sha256"]
        and item.get("status") == "accepted_bulk_derived_unrelaxed_precut"
    ]
    if len(source_matches) != 1:
        raise ValueError("Registered geometric precut changed before manifest update")
    relative_directory = destination_directory.relative_to(manifest_path.parent)
    relative_structure = relative_directory / destination_structure.name
    records = [
        {
            **record,
            "path": str(relative_directory / record["path"]),
        }
        for record in copied_records
    ]
    target = plan["target_surface"]
    promoted = {
        "path": str(relative_structure),
        "status": "accepted_relaxed_maintained_surface",
        "parent_bulk": manifest["bulk_parent"]["path"],
        "parent_bulk_sha256": plan["bulk_parent"]["sha256"],
        "source_precut_sha256": plan["source_surface_precut"]["sha256"],
        "protocol_version": plan["promotion_contract"]["protocol_version"],
        "promotion_fingerprint": plan["fingerprint"],
        "screen_run_fingerprint": plan["screen_result"]["run_fingerprint"],
        "termination_candidate_id": plan["screen_result"][
            "selected_candidate_id"
        ],
        "miller": plan["source_surface_precut"]["miller"],
        "primitive_miller": plan["source_surface_precut"]["primitive_miller"],
        "termination": plan["source_surface_precut"]["termination"],
        "cut_fractional_phase": plan["source_surface_precut"][
            "cut_fractional_phase"
        ],
        "formula": target["formula"],
        "atom_count": target["atom_count"],
        "source_layers": target["oriented_unit_repeats"],
        "cell_A": target["cell_A"],
        "solid_thickness_A": target["solid_thickness_A"],
        "vacuum_A": target["vacuum_A"],
        "top_plane_formula": target["top_plane_formula"],
        "bottom_plane_formula": target["bottom_plane_formula"],
        "formal_dipole_density_e_per_A": target[
            "formal_dipole_density_e_per_A"
        ],
        "relaxation_status": "force_and_thickness_converged",
        "surface_energy_eV_A2": target["surface_energy_eV_A2"],
        "surface_energy_J_m2": target["surface_energy_J_m2"],
        "maximum_force_eV_A": target["maximum_force_eV_A"],
        "maximum_atomic_displacement_A": target[
            "maximum_atomic_displacement_A"
        ],
        "thickness_convergence": plan["thickness_convergence"],
        "sha256": _sha256(destination_directory / destination_structure.name),
        "records": records,
    }
    source_matches[0]["status"] = "superseded_by_relaxed_maintained_surface"
    source_matches[0]["superseded_by"] = promoted["path"]
    source_matches[0]["superseded_by_sha256"] = promoted["sha256"]
    manifest["precut_slabs"].append(promoted)
    write_manifest(manifest_path, manifest)
    promotion_report.update(
        {
            "structure_manifest": {
                "path": str(manifest_path),
                "sha256": _sha256(manifest_path),
            },
            "registered_surface": {
                **promotion_report["registered_surface"],
                "path": str(destination_directory / destination_structure.name),
            },
        }
    )
    return promotion_report


def available_substrates(catalog_directory: Path | None = None) -> list[str]:
    directory = Path(catalog_directory or _default_catalog_directory())
    keys = []
    for path in sorted(directory.glob("*.toml")):
        keys.append(_load_catalog_file(path)["key"])
    return keys


def load_substrate_recipe(
    name: str,
    catalog_directory: Path | None = None,
    *,
    anchor_family: str | None = None,
) -> dict:
    """Load one unambiguous catalog entry by stable key or alias."""

    if not isinstance(name, str) or not name.strip():
        raise ValueError("substrate name must be a non-empty string")
    directory = Path(catalog_directory or _default_catalog_directory())
    matches = []
    requested = _normalized_name(name)
    for path in sorted(directory.glob("*.toml")):
        recipe = _load_catalog_file(path, anchor_family=anchor_family)
        names = [recipe["key"], *recipe.get("aliases", [])]
        if requested in {_normalized_name(candidate) for candidate in names}:
            matches.append(recipe)
    if not matches:
        available = ", ".join(available_substrates(directory)) or "none"
        raise ValueError(f"Unknown substrate {name!r}; available catalog keys: {available}")
    if len(matches) != 1:
        paths = ", ".join(str(item["catalog_path"]) for item in matches)
        raise ValueError(f"Ambiguous substrate name {name!r}; matches: {paths}")
    return matches[0]


def _site_discovery_surface_frame(atoms, normal_axis: int, side: str) -> dict:
    plane_axes = [axis for axis in range(3) if axis != normal_axis]
    cell = np.asarray(atoms.cell, dtype=float)
    u = cell[plane_axes[0]] / np.linalg.norm(cell[plane_axes[0]])
    trial_v = cell[plane_axes[1]] - np.dot(cell[plane_axes[1]], u) * u
    v = trial_v / np.linalg.norm(trial_v)
    n = np.cross(u, v)
    axis_direction = cell[normal_axis] / np.linalg.norm(cell[normal_axis])
    if np.dot(n, axis_direction) < 0:
        v = -v
        n = -n
    if side == "bottom":
        n = -n
        v = -v
    return {
        "periodic_fractional_axes": plane_axes,
        "u_cartesian_unit": u.tolist(),
        "v_cartesian_unit": v.tolist(),
        "outward_normal_cartesian_unit": n.tolist(),
    }


def _site_translation_key(vertices) -> tuple[tuple[int, int, int], ...]:
    normalized = []
    for _, origin_i, origin_j in vertices:
        normalized.append(
            tuple(
                sorted(
                    (atom_id, image_i - origin_i, image_j - origin_j)
                    for atom_id, image_i, image_j in vertices
                )
            )
        )
    return min(normalized)


def _site_discovery_symmetry_operations(
    atoms,
    metal_indices: list[int],
    normal_axis: int,
    tolerance_A: float,
) -> list[dict]:
    try:
        import spglib
    except ImportError as exc:
        raise RuntimeError("Adsorption-site enumeration requires spglib") from exc

    cell = np.asarray(atoms.cell, dtype=float)
    fractional = np.asarray(atoms.get_scaled_positions(wrap=True), dtype=float)
    numbers = np.asarray(atoms.numbers, dtype=int)
    symmetry = spglib.get_symmetry(
        (cell, fractional, numbers), symprec=float(tolerance_A)
    )
    if symmetry is None:
        raise ValueError("Cannot determine source-surface symmetry operations")
    plane_axes = [axis for axis in range(3) if axis != normal_axis]
    normal_basis = np.zeros(3, dtype=int)
    normal_basis[normal_axis] = 1
    metal_set = set(metal_indices)
    operations = []
    signatures = set()
    for rotation, translation in zip(
        symmetry["rotations"], symmetry["translations"]
    ):
        rotation = np.asarray(rotation, dtype=int)
        if not (
            np.array_equal(rotation @ normal_basis, normal_basis)
            and np.array_equal(normal_basis @ rotation, normal_basis)
        ):
            continue
        atom_map = {}
        valid = True
        for atom_id in metal_indices:
            transformed = rotation @ fractional[atom_id] + translation
            best = None
            for target_id in metal_indices:
                if numbers[target_id] != numbers[atom_id]:
                    continue
                delta = transformed - fractional[target_id]
                shift = np.rint(delta).astype(int)
                residual = (delta - shift) @ cell
                distance = float(np.linalg.norm(residual))
                if best is None or distance < best[0]:
                    best = (distance, target_id, shift)
            if best is None or best[0] > tolerance_A or best[1] not in metal_set:
                valid = False
                break
            atom_map[atom_id] = {
                "target_atom_id": int(best[1]),
                "image_offset": [int(best[2][axis]) for axis in plane_axes],
            }
        if not valid or len({item["target_atom_id"] for item in atom_map.values()}) != len(
            metal_indices
        ):
            continue
        in_plane_rotation = rotation[np.ix_(plane_axes, plane_axes)]
        signature = (
            tuple(int(value) for value in in_plane_rotation.flat),
            tuple(
                (
                    atom_id,
                    atom_map[atom_id]["target_atom_id"],
                    *atom_map[atom_id]["image_offset"],
                )
                for atom_id in sorted(atom_map)
            ),
        )
        if signature in signatures:
            continue
        signatures.add(signature)
        operations.append(
            {
                "rotation": in_plane_rotation,
                "atom_map": atom_map,
            }
        )
    if not operations:
        raise ValueError("No outward-normal-preserving surface symmetry was found")
    return operations


def _site_symmetry_key(vertices, operations) -> tuple[tuple[int, int, int], ...]:
    equivalents = []
    for operation in operations:
        rotation = operation["rotation"]
        transformed = []
        for atom_id, image_i, image_j in vertices:
            mapped = operation["atom_map"][atom_id]
            image = rotation @ np.asarray([image_i, image_j], dtype=int)
            image += np.asarray(mapped["image_offset"], dtype=int)
            transformed.append(
                (mapped["target_atom_id"], int(image[0]), int(image[1]))
            )
        equivalents.append(_site_translation_key(transformed))
    return min(equivalents)


def _enumerate_periodic_site_combinations(atoms, contract: dict, normal_axis: int) -> dict:
    """Enumerate non-monodentate periodic metal combinations and reduce symmetry."""

    side = contract["surface_side"]
    frame = _site_discovery_surface_frame(atoms, normal_axis, side)
    plane_axes = frame["periodic_fractional_axes"]
    normal = np.asarray(frame["outward_normal_cartesian_unit"], dtype=float)
    positions = np.asarray(atoms.positions, dtype=float)
    fractional = np.asarray(atoms.get_scaled_positions(wrap=True), dtype=float)
    symbols = np.asarray(atoms.get_chemical_symbols())
    allowed = set(contract["surface_metal_elements"])
    eligible = np.flatnonzero(np.isin(symbols, list(allowed)))
    if not len(eligible):
        raise ValueError("Source surface contains no configured surface metal elements")
    normal_coordinates = positions @ normal
    exposed_coordinate = float(np.max(normal_coordinates[eligible]))
    depth = float(contract["surface_metal_depth_A"])
    metal_indices = sorted(
        int(index)
        for index in eligible
        if exposed_coordinate - normal_coordinates[index] <= depth + 1.0e-10
    )
    if not metal_indices:
        raise ValueError("No accessible surface metals passed the configured depth gate")

    operations = _site_discovery_symmetry_operations(
        atoms,
        metal_indices,
        normal_axis,
        float(contract["symmetry_tolerance_A"]),
    )
    cell = np.asarray(atoms.cell, dtype=float)
    in_plane_a = cell[plane_axes[0]]
    in_plane_b = cell[plane_axes[1]]
    area = float(np.linalg.norm(np.cross(in_plane_a, in_plane_b)))
    minimum_altitude = min(
        area / np.linalg.norm(in_plane_a),
        area / np.linalg.norm(in_plane_b),
    )
    minimum_distance, maximum_distance = (
        float(value) for value in contract["metal_vertex_distance_window_A"]
    )
    image_radius = max(1, int(math.ceil(maximum_distance / minimum_altitude)) + 1)
    maximum_span = float(contract["maximum_vertex_normal_span_A"])
    binding_modes = contract.get(
        "binding_modes", ["bridging_distinct_metals"]
    )
    by_denticity = {}
    translation_unique_all = {}
    for denticity in sorted(contract["denticities"]):
        translation_unique = {}
        denticity_raw_count = 0
        for anchor_id in metal_indices:
            anchor = (anchor_id, 0, 0)
            anchor_position = positions[anchor_id]
            neighbors = []
            for atom_id in metal_indices:
                for image_i in range(-image_radius, image_radius + 1):
                    for image_j in range(-image_radius, image_radius + 1):
                        vertex = (atom_id, image_i, image_j)
                        if vertex == anchor:
                            continue
                        position = (
                            positions[atom_id]
                            + image_i * in_plane_a
                            + image_j * in_plane_b
                        )
                        distance = float(np.linalg.norm(position - anchor_position))
                        if distance_within_closed_window(
                            distance,
                            minimum_distance,
                            maximum_distance,
                        ):
                            neighbors.append(vertex)
            for selected in combinations(sorted(set(neighbors)), denticity - 1):
                vertices = (anchor, *selected)
                if len(set(vertices)) != denticity:
                    continue
                vertex_positions = np.asarray(
                    [
                        positions[atom_id]
                        + image_i * in_plane_a
                        + image_j * in_plane_b
                        for atom_id, image_i, image_j in vertices
                    ]
                )
                distances = [
                    float(np.linalg.norm(vertex_positions[right] - vertex_positions[left]))
                    for left, right in combinations(range(denticity), 2)
                ]
                span = float(np.ptp(vertex_positions @ normal))
                if (
                    not all(
                        distance_within_closed_window(
                            value,
                            minimum_distance,
                            maximum_distance,
                        )
                        for value in distances
                    )
                    or span > maximum_span
                ):
                    continue
                denticity_raw_count += 1
                key = _site_translation_key(vertices)
                translation_unique[key] = key
        by_denticity[denticity] = {
            "translation_unique": translation_unique,
            "raw_mapping_count": denticity_raw_count,
        }
        translation_unique_all.update(
            {(denticity, key): key for key in translation_unique}
        )

    symmetry_groups = {}
    for (denticity, key), vertices in translation_unique_all.items():
        symmetry_key = _site_symmetry_key(vertices, operations)
        symmetry_groups.setdefault((denticity, symmetry_key), set()).add(key)

    candidates = []
    donor_labels = list(contract["probe_donor_labels"])
    for number, ((denticity, representative), orbit) in enumerate(
        sorted(symmetry_groups.items()), 1
    ):
        vertex_positions = np.asarray(
            [
                positions[atom_id] + image_i * in_plane_a + image_j * in_plane_b
                for atom_id, image_i, image_j in representative
            ]
        )
        centroid = np.mean(vertex_positions, axis=0)
        centroid_fractional = centroid @ np.linalg.inv(cell)
        wrapped_fractional = centroid_fractional.copy()
        wrapped_fractional[plane_axes] %= 1.0
        for axis in plane_axes:
            if math.isclose(
                float(wrapped_fractional[axis]), 1.0, abs_tol=1.0e-8
            ) or math.isclose(
                float(wrapped_fractional[axis]), 0.0, abs_tol=1.0e-8
            ):
                wrapped_fractional[axis] = 0.0
        wrapped_centroid = wrapped_fractional @ cell
        pair_distances = sorted(
            float(np.linalg.norm(vertex_positions[right] - vertex_positions[left]))
            for left, right in combinations(range(denticity), 2)
        )
        vertices = []
        for atom_id, image_i, image_j in representative:
            unwrapped = positions[atom_id] + image_i * in_plane_a + image_j * in_plane_b
            vertices.append(
                {
                    "source_atom_id": int(atom_id),
                    "atom_index_base": 0,
                    "lattice_image": [int(image_i), int(image_j)],
                    "element": str(symbols[atom_id]),
                    "source_fractional_xyz": fractional[atom_id].tolist(),
                    "unwrapped_cartesian_A": unwrapped.tolist(),
                }
            )
        candidates.append(
            {
                "site_combination_id": f"site-combination-{number:04d}",
                "denticity": int(denticity),
                **(
                    {"binding_mode": "bridging_distinct_metals"}
                    if "binding_modes" in contract
                    else {}
                ),
                "probe": contract["probe"],
                "metal_vertices": vertices,
                "donor_metal_mapping": [
                    {
                        "probe_donor_id": donor_labels[index],
                        "metal_vertex_index": index,
                    }
                    for index in range(denticity)
                ],
                "ideal_site_location": {
                    "fractional_uv": [
                        float(wrapped_fractional[axis]) for axis in plane_axes
                    ],
                    "cartesian_A": wrapped_centroid.tolist(),
                    "unwrapped_centroid_cartesian_A": centroid.tolist(),
                    "surface_normal_coordinate_A": float(np.dot(centroid, normal)),
                    "definition": (
                        "centroid of explicitly unwrapped periodic metal vertices, "
                        "wrapped into the source surface cell in u/v"
                    ),
                },
                "metal_pair_distances_A": pair_distances,
                "metal_vertex_normal_span_A": float(
                    np.ptp(vertex_positions @ normal)
                ),
                "canonicalization": {
                    "equivalent_probe_donor_permutations_removed": True,
                    "overall_2d_lattice_translations_removed": True,
                    "surface_symmetry_operations_applied": len(operations),
                    "translation_unique_orbit_size": len(orbit),
                },
            }
        )

    bridge_candidate_count = len(candidates)
    chelate_raw_count = 0
    chelate_translation_unique = {}
    if "chelating_shared_metal" in binding_modes:
        for atom_id in metal_indices:
            vertex = (atom_id, 0, 0)
            key = _site_translation_key((vertex,))
            chelate_raw_count += 1
            chelate_translation_unique[key] = key
        chelate_symmetry_groups = {}
        for key, vertices in chelate_translation_unique.items():
            symmetry_key = _site_symmetry_key(vertices, operations)
            chelate_symmetry_groups.setdefault(symmetry_key, set()).add(key)
        for representative, orbit in sorted(chelate_symmetry_groups.items()):
            atom_id, image_i, image_j = representative[0]
            unwrapped = (
                positions[atom_id]
                + image_i * in_plane_a
                + image_j * in_plane_b
            )
            wrapped_fractional = unwrapped @ np.linalg.inv(cell)
            wrapped_fractional[plane_axes] %= 1.0
            for axis in plane_axes:
                if math.isclose(
                    float(wrapped_fractional[axis]), 1.0, abs_tol=1.0e-8
                ) or math.isclose(
                    float(wrapped_fractional[axis]), 0.0, abs_tol=1.0e-8
                ):
                    wrapped_fractional[axis] = 0.0
            wrapped_position = wrapped_fractional @ cell
            candidates.append(
                {
                    "site_combination_id": (
                        f"site-combination-{len(candidates) + 1:04d}"
                    ),
                    "denticity": 2,
                    "binding_mode": "chelating_shared_metal",
                    "probe": contract["probe"],
                    "metal_vertices": [
                        {
                            "source_atom_id": int(atom_id),
                            "atom_index_base": 0,
                            "lattice_image": [int(image_i), int(image_j)],
                            "element": str(symbols[atom_id]),
                            "source_fractional_xyz": fractional[atom_id].tolist(),
                            "unwrapped_cartesian_A": unwrapped.tolist(),
                        }
                    ],
                    "donor_metal_mapping": [
                        {
                            "probe_donor_id": donor_labels[index],
                            "metal_vertex_index": 0,
                        }
                        for index in range(2)
                    ],
                    "ideal_site_location": {
                        "fractional_uv": [
                            float(wrapped_fractional[axis]) for axis in plane_axes
                        ],
                        "cartesian_A": wrapped_position.tolist(),
                        "unwrapped_centroid_cartesian_A": unwrapped.tolist(),
                        "surface_normal_coordinate_A": float(
                            np.dot(unwrapped, normal)
                        ),
                        "definition": (
                            "shared surface-metal vertex for a two-donor chelate, "
                            "wrapped into the source surface cell in u/v"
                        ),
                    },
                    "metal_pair_distances_A": [],
                    "metal_vertex_normal_span_A": 0.0,
                    "canonicalization": {
                        "equivalent_probe_donor_permutations_removed": True,
                        "overall_2d_lattice_translations_removed": True,
                        "surface_symmetry_operations_applied": len(operations),
                        "translation_unique_orbit_size": len(orbit),
                    },
                }
            )

    counts = {}
    for denticity in sorted(contract["denticities"]):
        counts[str(denticity)] = {
            "raw_anchored_mapping_count": by_denticity[denticity][
                "raw_mapping_count"
            ],
            "translation_unique_count": len(
                by_denticity[denticity]["translation_unique"]
            ),
            "symmetry_unique_count": sum(
                candidate["denticity"] == denticity for candidate in candidates
            ),
        }
    if "chelating_shared_metal" in binding_modes:
        counts["2"]["raw_anchored_mapping_count"] += chelate_raw_count
        counts["2"]["translation_unique_count"] += len(
            chelate_translation_unique
        )
    return {
        "surface_frame": frame,
        "surface_metal_selection": {
            "elements": sorted(allowed),
            "side": side,
            "depth_from_outermost_configured_metal_A": depth,
            "outermost_normal_coordinate_A": exposed_coordinate,
            "selected_atom_count": len(metal_indices),
            "selected_source_atom_ids": metal_indices,
            "atom_index_base": 0,
        },
        "periodic_image_search_radius": image_radius,
        "distance_window_comparison_ulps": DISTANCE_WINDOW_COMPARISON_ULPS,
        "surface_symmetry_operation_count": len(operations),
        "counts_by_denticity": counts,
        "counts_by_binding_mode": {
            "bridging_distinct_metals": bridge_candidate_count,
            "chelating_shared_metal": len(candidates) - bridge_candidate_count,
        },
        "site_combinations": candidates,
    }


def resolve_adsorption_site_enumeration_plan(
    substrate_name: str,
    catalog_directory: Path | None = None,
    *,
    anchor_family: str | None = None,
) -> dict:
    """Resolve a read-only clean-surface multidentate site enumeration plan."""

    recipe = load_substrate_recipe(
        substrate_name, catalog_directory, anchor_family=anchor_family
    )
    contract = recipe["adsorption"].get("site_discovery")
    if not isinstance(contract, dict):
        raise ValueError(
            f"Substrate {recipe['key']!r} has no adsorption.site_discovery contract"
        )
    geometry_contract = {
        key: value
        for key, value in contract.items()
        if key not in {"probe_manifest", "probe_preparation", "probe_relaxation"}
    }
    source_path = Path(recipe["source_path"])
    atoms = read(source_path)
    enumeration = _enumerate_periodic_site_combinations(
        atoms, geometry_contract, int(recipe["surface"]["normal_axis"])
    )
    identity = {
        "substrate_key": recipe["key"],
        "source_sha256": recipe["source"]["sha256"],
        "surface": recipe["surface"],
        "site_discovery_contract": geometry_contract,
        "site_combinations": enumeration["site_combinations"],
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    candidates = enumeration["site_combinations"]
    return {
        "schema_version": 1,
        "mode": "adsorption_site_enumeration_plan",
        "status": "passed_enumeration_pending_manual_review",
        "substrate": {
            "key": recipe["key"],
            "catalog_path": str(recipe["catalog_path"]),
            "catalog_sha256": _sha256(Path(recipe["catalog_path"])),
        },
        "source_surface": {
            "path": str(source_path),
            "sha256": recipe["source"]["sha256"],
            "formula": _formula(atoms.get_chemical_symbols()),
            "atom_count": len(atoms),
            "miller": list(recipe["surface"]["miller"]),
            "termination": recipe["surface"]["termination"],
        },
        "site_discovery_contract": geometry_contract,
        **enumeration,
        "validation": {
            "no_monodentate_candidates": all(
                candidate["denticity"] >= 2 for candidate in candidates
            ),
            "all_configured_denticities_present": set(geometry_contract["denticities"])
            == {candidate["denticity"] for candidate in candidates},
            "candidate_identity_includes_periodic_metal_vertices": all(
                candidate["metal_vertices"]
                and all("lattice_image" in vertex for vertex in candidate["metal_vertices"])
                for candidate in candidates
            ),
            "ideal_site_locations_recorded": all(
                candidate["ideal_site_location"]["fractional_uv"]
                for candidate in candidates
            ),
        },
        "fingerprint": fingerprint,
        "execution_contract": {
            "read_only": True,
            "executable": bool(candidates),
            "starts_optimizer_or_gpu": False,
            "readiness_issues": [] if candidates else [
                {
                    "code": "no_geometrically_reachable_site_combinations",
                    "message": "The configured geometric screen produced no candidates.",
                }
            ],
            "known_limitations": [
                "Candidates are geometric Site Combinations, not relaxed Site Prototypes.",
                "No adsorption energy has been calculated or inferred.",
                "Manual candidate-space review is required before probe generation.",
                "The catalog sites_per_source_cell value is not used as an input or target.",
            ],
        },
    }


def execute_adsorption_site_enumeration(plan: dict, output_root: Path) -> dict:
    """Write one immutable clean-surface site-combination package."""

    if plan.get("mode") != "adsorption_site_enumeration_plan":
        raise ValueError("Site enumeration requires an adsorption-site enumeration plan")
    if not plan["execution_contract"]["executable"]:
        raise ValueError("Adsorption-site enumeration plan has readiness issues")
    source_path = Path(plan["source_surface"]["path"])
    if not source_path.is_file() or _sha256(source_path) != plan["source_surface"]["sha256"]:
        raise ValueError("Source surface changed after adsorption-site planning")
    output_root = Path(output_root).expanduser().resolve()
    run_directory = (
        output_root
        / plan["substrate"]["key"].casefold()
        / f"{plan['site_discovery_contract']['protocol_version']}-{plan['fingerprint'][:12]}"
    )
    if run_directory.exists():
        raise ValueError(f"Immutable adsorption-site run already exists: {run_directory}")
    run_directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        dir=run_directory.parent, prefix=f".{run_directory.name}.tmp-"
    ) as temporary_name:
        temporary = Path(temporary_name)
        plan_path = temporary / "adsorption-site-enumeration-plan.json"
        source_copy = temporary / source_path.name
        combinations_path = temporary / "site-combinations.json"
        manifest_path = temporary / "run-manifest.json"
        shutil.copyfile(source_path, source_copy)
        write_manifest(plan_path, plan)
        combinations_report = {
            "schema_version": 1,
            "status": "passed_enumeration_pending_manual_review",
            "substrate": plan["substrate"],
            "source_surface": plan["source_surface"],
            "site_discovery_contract": plan["site_discovery_contract"],
            "surface_frame": plan["surface_frame"],
            "surface_metal_selection": plan["surface_metal_selection"],
            "periodic_image_search_radius": plan["periodic_image_search_radius"],
            "surface_symmetry_operation_count": plan[
                "surface_symmetry_operation_count"
            ],
            "counts_by_denticity": plan["counts_by_denticity"],
            "site_combinations": plan["site_combinations"],
            "validation": plan["validation"],
            "manual_review": {
                "status": "pending",
                "required_before": "probe_generation_and_adsorption_relaxation",
            },
        }
        write_manifest(combinations_path, combinations_report)
        manifest = {
            "schema_version": 1,
            "status": "passed_enumeration_pending_manual_review",
            "mode": "adsorption_site_enumeration_run",
            "fingerprint": plan["fingerprint"],
            "substrate_key": plan["substrate"]["key"],
            "candidate_count": len(plan["site_combinations"]),
            "counts_by_denticity": plan["counts_by_denticity"],
            "artifacts": [
                {
                    "role": "source_surface_snapshot",
                    "path": source_copy.name,
                    "sha256": _sha256(source_copy),
                },
                {
                    "role": "enumeration_plan",
                    "path": plan_path.name,
                    "sha256": _sha256(plan_path),
                },
                {
                    "role": "site_combinations",
                    "path": combinations_path.name,
                    "sha256": _sha256(combinations_path),
                },
            ],
        }
        write_manifest(manifest_path, manifest)
        temporary.replace(run_directory)
    return {**manifest, "output_directory": str(run_directory)}


def _load_probe_definition(manifest_path: Path) -> tuple[dict, object]:
    manifest_path = Path(manifest_path).resolve()
    with manifest_path.open("rb") as handle:
        manifest = tomllib.load(handle)
    if manifest.get("schema_version") != 1:
        raise ValueError(f"{manifest_path}: probe schema_version must be 1")
    key = manifest.get("key")
    if not isinstance(key, str) or not key.strip():
        raise ValueError(f"{manifest_path}: probe key must be non-empty")
    source_value = manifest.get("source")
    if not isinstance(source_value, str) or not source_value.strip():
        raise ValueError(f"{manifest_path}: probe source must be a non-empty path")
    source_path = (manifest_path.parent / source_value).resolve()
    if not source_path.is_file() or _sha256(source_path) != manifest.get("sha256"):
        raise ValueError(f"{manifest_path}: probe source is missing or changed")
    neutral = read(source_path)
    expected_neutral = dict(
        sorted(_table(manifest.get("expected_neutral_formula"), "expected_neutral_formula").items())
    )
    observed_neutral = _formula(neutral.get_chemical_symbols())
    if observed_neutral != expected_neutral:
        raise ValueError(
            f"{manifest_path}: neutral probe formula {observed_neutral} does not match "
            f"{expected_neutral}"
        )
    topology = _table(manifest.get("topology"), "probe.topology")
    deprotonation = _table(manifest.get("deprotonation"), "probe.deprotonation")
    symbols = neutral.get_chemical_symbols()
    adjacency = _molecular_adjacency(
        symbols,
        np.asarray(neutral.positions, dtype=float),
        np.asarray(neutral.cell, dtype=float),
        np.asarray(neutral.pbc, dtype=bool),
    )
    anchor_element = _element(topology.get("anchor_element"), "probe.anchor_element")
    anchor_candidates = [
        index for index, symbol in enumerate(symbols) if symbol == anchor_element
    ]
    anchors = [
        index
        for index in anchor_candidates
        if sum(symbols[neighbor] == "O" for neighbor in adjacency[index])
        == int(topology["oxygen_neighbor_count"])
        and sum(symbols[neighbor] == "C" for neighbor in adjacency[index])
        == int(topology["carbon_neighbor_count"])
    ]
    if len(anchors) != 1:
        raise ValueError(
            f"{manifest_path}: probe must contain one topologically unique "
            f"{anchor_element} anchor; found {len(anchors)}"
        )
    anchor = anchors[0]
    oxygen_neighbors = sorted(index for index in adjacency[anchor] if symbols[index] == "O")
    carbon_neighbors = sorted(index for index in adjacency[anchor] if symbols[index] == "C")
    if len(oxygen_neighbors) != int(topology["oxygen_neighbor_count"]):
        raise ValueError(f"{manifest_path}: probe anchor oxygen topology changed")
    if len(carbon_neighbors) != int(topology["carbon_neighbor_count"]):
        raise ValueError(f"{manifest_path}: probe anchor carbon topology changed")
    acidic_hydrogens = sorted(
        index
        for oxygen in oxygen_neighbors
        for index in adjacency[oxygen]
        if symbols[index] == "H"
    )
    if len(acidic_hydrogens) != int(topology["acidic_hydrogen_count"]):
        raise ValueError(f"{manifest_path}: acidic-hydrogen topology changed")
    carbon = carbon_neighbors[0]
    methyl_hydrogens = sorted(
        index for index in adjacency[carbon] if symbols[index] == "H"
    )
    if len(methyl_hydrogens) != int(topology["methyl_hydrogen_count"]):
        raise ValueError(f"{manifest_path}: methyl-hydrogen topology changed")
    if deprotonation.get("policy") != (
        "remove_every_hydrogen_covalently_bound_to_anchor_oxygen"
    ):
        raise ValueError(f"{manifest_path}: unsupported probe deprotonation policy")
    if deprotonation.get("retain_released_protons_in_surface_system") is not True:
        raise ValueError(f"{manifest_path}: released probe protons must be retained")
    released = int(deprotonation.get("released_protons", -1))
    if released != len(acidic_hydrogens):
        raise ValueError(f"{manifest_path}: released-proton count is inconsistent")
    kept_source_ids = [
        index for index in range(len(neutral)) if index not in acidic_hydrogens
    ]
    adsorbate = neutral[kept_source_ids]
    expected_adsorbate = dict(
        sorted(
            _table(
                deprotonation.get("fully_deprotonated_formula"),
                "fully_deprotonated_formula",
            ).items()
        )
    )
    observed_adsorbate = _formula(adsorbate.get_chemical_symbols())
    if observed_adsorbate != expected_adsorbate:
        raise ValueError(
            f"{manifest_path}: deprotonated formula {observed_adsorbate} does not "
            f"match {expected_adsorbate}"
        )
    source_to_adsorbate = {
        source_id: adsorbate_id
        for adsorbate_id, source_id in enumerate(kept_source_ids)
    }
    resolved = {
        "schema_version": 1,
        "key": key,
        "display_name": manifest.get("display_name"),
        "anchor_family": manifest.get("anchor_family"),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "source_kind": manifest.get("source_kind"),
        "source_path": str(source_path),
        "source_sha256": manifest["sha256"],
        "source_url": manifest.get("source_url"),
        "pubchem_cid": manifest.get("pubchem_cid"),
        "inchi_key": manifest.get("inchi_key"),
        "neutral_formula": observed_neutral,
        "neutral_atom_count": len(neutral),
        "fully_deprotonated_formula": observed_adsorbate,
        "fully_deprotonated_atom_count": len(adsorbate),
        "released_protons": released,
        "source_atom_ids": {
            "anchor": anchor,
            "carbon": carbon,
            "oxygen_donors": oxygen_neighbors,
            "acidic_hydrogens_removed": acidic_hydrogens,
            "methyl_hydrogens_retained": methyl_hydrogens,
        },
        "adsorbate_atom_ids": {
            "anchor": source_to_adsorbate[anchor],
            "carbon": source_to_adsorbate[carbon],
            "oxygen_donors": [
                source_to_adsorbate[index] for index in oxygen_neighbors
            ],
            "methyl_hydrogens": [
                source_to_adsorbate[index] for index in methyl_hydrogens
            ],
        },
        "adsorbate_to_source_atom_id": kept_source_ids,
        "deprotonation_policy": deprotonation["policy"],
        "alternative_states_enumerated": bool(
            deprotonation.get("alternative_states_enumerated")
        ),
    }
    return resolved, adsorbate


def _probe_chemistry_identity(probe: dict) -> dict:
    """Return the probe identity without host-specific absolute path spellings."""

    if not isinstance(probe, dict):
        raise ValueError("Probe chemistry record must be a table")
    return {
        key: value
        for key, value in probe.items()
        if key not in {"manifest_path", "source_path"}
    }


def _load_site_enumeration_evidence(run_directory: Path) -> dict:
    run_directory = Path(run_directory).expanduser().resolve()
    manifest_path = run_directory / "run-manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing adsorption-site run manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("mode") != "adsorption_site_enumeration_run"
        or manifest.get("status") != "passed_enumeration_pending_manual_review"
    ):
        raise ValueError("Adsorption-site enumeration evidence is not a passed package")
    artifacts = {}
    for artifact in manifest.get("artifacts", []):
        role = artifact.get("role")
        path_value = artifact.get("path")
        if not isinstance(role, str) or role in artifacts or not isinstance(path_value, str):
            raise ValueError("Invalid or duplicate adsorption-site enumeration artifact")
        path = (run_directory / path_value).resolve()
        if not path.is_file() or _sha256(path) != artifact.get("sha256"):
            raise ValueError(f"Adsorption-site enumeration artifact changed: {path}")
        artifacts[role] = path
    required = {"source_surface_snapshot", "enumeration_plan", "site_combinations"}
    if not required.issubset(artifacts):
        raise ValueError("Adsorption-site enumeration evidence chain is incomplete")
    report = json.loads(artifacts["site_combinations"].read_text(encoding="utf-8"))
    plan = json.loads(artifacts["enumeration_plan"].read_text(encoding="utf-8"))
    if (
        report.get("status") != "passed_enumeration_pending_manual_review"
        or plan.get("fingerprint") != manifest.get("fingerprint")
        or report.get("site_combinations") != plan.get("site_combinations")
    ):
        raise ValueError("Adsorption-site enumeration reports are inconsistent")
    return {
        "run_directory": str(run_directory),
        "run_manifest_path": str(manifest_path),
        "run_manifest_sha256": _sha256(manifest_path),
        "fingerprint": manifest["fingerprint"],
        "substrate_key": manifest["substrate_key"],
        "source_surface_path": str(artifacts["source_surface_snapshot"]),
        "source_surface_sha256": _sha256(artifacts["source_surface_snapshot"]),
        "enumeration_plan_path": str(artifacts["enumeration_plan"]),
        "enumeration_plan_sha256": _sha256(artifacts["enumeration_plan"]),
        "site_combinations_path": str(artifacts["site_combinations"]),
        "site_combinations_sha256": _sha256(artifacts["site_combinations"]),
        "surface_frame": report["surface_frame"],
        "site_combinations": report["site_combinations"],
        "candidate_count": len(report["site_combinations"]),
    }


def _load_probe_preparation_evidence(run_directory: Path) -> dict:
    """Load and hash-check an immutable adsorption-probe preparation package."""

    run_directory = Path(run_directory).expanduser().resolve()
    manifest_path = run_directory / "run-manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing adsorption-probe run manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("mode") != "adsorption_probe_preparation_run"
        or manifest.get("status") != "passed_preparation_pending_relaxation"
    ):
        raise ValueError("Adsorption-probe preparation evidence is not a passed package")
    artifacts = {}
    for artifact in manifest.get("artifacts", []):
        role = artifact.get("role")
        path_value = artifact.get("path")
        if not isinstance(role, str) or role in artifacts or not isinstance(path_value, str):
            raise ValueError("Invalid or duplicate adsorption-probe preparation artifact")
        path = (run_directory / path_value).resolve()
        if run_directory not in path.parents or not path.is_file():
            raise ValueError(f"Adsorption-probe artifact path is invalid: {path}")
        if _sha256(path) != artifact.get("sha256"):
            raise ValueError(f"Adsorption-probe preparation artifact changed: {path}")
        artifacts[role] = path
    required = {
        "probe_preparation_plan",
        "source_surface_snapshot",
        "neutral_probe_source_snapshot",
        "starting_structures_report",
    }
    if not required.issubset(artifacts):
        raise ValueError("Adsorption-probe preparation evidence chain is incomplete")
    plan = json.loads(artifacts["probe_preparation_plan"].read_text(encoding="utf-8"))
    report = json.loads(
        artifacts["starting_structures_report"].read_text(encoding="utf-8")
    )
    fingerprint = manifest.get("fingerprint")
    if (
        plan.get("fingerprint") != fingerprint
        or report.get("fingerprint") != fingerprint
        or report.get("status") != "passed_preparation_pending_relaxation"
        or report.get("substrate_key") != manifest.get("substrate_key")
    ):
        raise ValueError("Adsorption-probe preparation reports are inconsistent")
    clean_test_supercell_path = artifacts.get("clean_test_supercell_snapshot")
    if clean_test_supercell_path is None:
        if plan.get("test_supercell", {}).get("repeat") != [1, 1, 1]:
            raise ValueError(
                "Adsorption-probe preparation package lacks the expanded clean "
                "test-supercell reference"
            )
        clean_test_supercell_path = artifacts["source_surface_snapshot"]
    structures = report.get("structures")
    if not isinstance(structures, list) or not structures:
        raise ValueError("Adsorption-probe preparation package contains no structures")
    if len(structures) != int(manifest.get("starting_structure_count", -1)):
        raise ValueError("Adsorption-probe starting-structure count is inconsistent")
    resolved_structures = []
    identities = set()
    for record in structures:
        path_value = record.get("path")
        if not isinstance(path_value, str):
            raise ValueError("Adsorption-probe structure path must be a string")
        path = (run_directory / path_value).resolve()
        if run_directory not in path.parents or not path.is_file():
            raise ValueError(f"Adsorption-probe structure path is invalid: {path}")
        if _sha256(path) != record.get("sha256"):
            raise ValueError(f"Adsorption-probe starting structure changed: {path}")
        identity = (
            record.get("site_combination_id"),
            record.get("proton_placement_id"),
        )
        if any(not isinstance(value, str) or not value for value in identity):
            raise ValueError("Adsorption-probe structure identity is incomplete")
        if identity in identities:
            raise ValueError("Duplicate adsorption-probe starting-structure identity")
        identities.add(identity)
        resolved_structures.append({**record, "absolute_path": str(path)})
    return {
        "run_directory": str(run_directory),
        "run_manifest_path": str(manifest_path),
        "run_manifest_sha256": _sha256(manifest_path),
        "fingerprint": fingerprint,
        "substrate_key": manifest["substrate_key"],
        "site_combination_count": int(manifest["site_combination_count"]),
        "starting_structure_count": len(resolved_structures),
        "preparation_plan_path": str(artifacts["probe_preparation_plan"]),
        "preparation_plan_sha256": _sha256(artifacts["probe_preparation_plan"]),
        "preparation_plan": plan,
        "source_surface_path": str(artifacts["source_surface_snapshot"]),
        "source_surface_sha256": _sha256(artifacts["source_surface_snapshot"]),
        "clean_test_supercell_path": str(clean_test_supercell_path),
        "clean_test_supercell_sha256": _sha256(clean_test_supercell_path),
        "neutral_probe_source_path": str(artifacts["neutral_probe_source_snapshot"]),
        "neutral_probe_source_sha256": _sha256(
            artifacts["neutral_probe_source_snapshot"]
        ),
        "starting_structures_report_path": str(
            artifacts["starting_structures_report"]
        ),
        "starting_structures_report_sha256": _sha256(
            artifacts["starting_structures_report"]
        ),
        "starting_structures": resolved_structures,
    }


def _proper_rotation(source_vectors, target_vectors) -> np.ndarray:
    covariance = np.asarray(source_vectors, dtype=float).T @ np.asarray(
        target_vectors, dtype=float
    )
    left, _, right_transpose = np.linalg.svd(covariance)
    rotation = left @ right_transpose
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right_transpose
    return rotation


def _shortest_in_plane_translation(cell: np.ndarray, plane_axes: list[int]) -> float:
    vectors = [cell[axis] for axis in plane_axes]
    distances = []
    for image_i, image_j in product(range(-3, 4), repeat=2):
        if image_i == 0 and image_j == 0:
            continue
        distances.append(
            float(np.linalg.norm(image_i * vectors[0] + image_j * vectors[1]))
        )
    return min(distances)


def _minimum_periodic_probe_image_distance(
    positions: np.ndarray,
    cell: np.ndarray,
    plane_axes: list[int],
) -> float:
    minimum = math.inf
    for image_i, image_j in product(range(-1, 2), repeat=2):
        if image_i == 0 and image_j == 0:
            continue
        shift = image_i * cell[plane_axes[0]] + image_j * cell[plane_axes[1]]
        shifted = positions + shift
        for position in positions:
            minimum = min(
                minimum,
                float(np.min(np.linalg.norm(shifted - position, axis=1))),
            )
    return minimum


def _map_periodic_vertex_to_supercell(
    supercell,
    position: np.ndarray,
    element: str,
) -> int:
    symbols = np.asarray(supercell.get_chemical_symbols())
    candidates = np.flatnonzero(symbols == element)
    distances = []
    for atom_id in candidates:
        vector = _minimum_image_vector(
            np.asarray(supercell.positions[atom_id], dtype=float) - position,
            np.asarray(supercell.cell, dtype=float),
            np.asarray(supercell.pbc, dtype=bool),
        )
        distances.append(float(np.linalg.norm(vector)))
    nearest = int(np.argmin(distances))
    if distances[nearest] > 1.0e-5:
        raise ValueError("Cannot map a source Site Combination into the test supercell")
    return int(candidates[nearest])


def _probe_substrate_clearance(
    supercell,
    probe_symbols: np.ndarray,
    probe_positions: np.ndarray,
    allowed_contacts: set[tuple[int, int]],
) -> float:
    minimum = math.inf
    heavy_probe = np.flatnonzero(probe_symbols != "H")
    cell = np.asarray(supercell.cell, dtype=float)
    pbc = np.asarray(supercell.pbc, dtype=bool)
    inverse_cell = np.linalg.inv(cell)
    substrate_positions = np.asarray(supercell.positions, dtype=float)
    for probe_id in heavy_probe:
        vectors = substrate_positions - probe_positions[probe_id]
        fractional = vectors @ inverse_cell
        fractional[:, pbc] -= np.round(fractional[:, pbc])
        distances = np.linalg.norm(fractional @ cell, axis=1)
        allowed_substrate_ids = [
            substrate_id
            for allowed_probe_id, substrate_id in allowed_contacts
            if allowed_probe_id == int(probe_id)
        ]
        if allowed_substrate_ids:
            distances[allowed_substrate_ids] = math.inf
        minimum = min(minimum, float(np.min(distances)))
    return minimum


def _unmapped_donor_metal_clearance(
    supercell,
    probe_positions: np.ndarray,
    unmapped_donor_ids: list[int],
    metal_elements: list[str],
) -> float | None:
    if not unmapped_donor_ids:
        return None
    symbols = np.asarray(supercell.get_chemical_symbols())
    metals = np.flatnonzero(np.isin(symbols, metal_elements))
    cell = np.asarray(supercell.cell, dtype=float)
    pbc = np.asarray(supercell.pbc, dtype=bool)
    minimum = math.inf
    for donor_id in unmapped_donor_ids:
        for metal_id in metals:
            vector = _minimum_image_vector(
                np.asarray(supercell.positions[metal_id], dtype=float)
                - probe_positions[donor_id],
                cell,
                pbc,
            )
            minimum = min(minimum, float(np.linalg.norm(vector)))
    return minimum


def _prepare_one_probe_site(
    site: dict,
    supercell,
    source_cell: np.ndarray,
    repeat: int,
    frame: dict,
    probe: dict,
    adsorbate,
    discovery_contract: dict,
    preparation_contract: dict,
) -> dict:
    plane_axes = list(frame["periodic_fractional_axes"])
    normal = np.asarray(frame["outward_normal_cartesian_unit"], dtype=float)
    original_vertices = site["metal_vertices"]
    original_positions = np.asarray(
        [vertex["unwrapped_cartesian_A"] for vertex in original_vertices],
        dtype=float,
    )
    centroid_fractional = np.mean(original_positions, axis=0) @ np.linalg.inv(
        source_cell
    )
    translation_images = [0, 0]
    for local_axis, fractional_axis in enumerate(plane_axes):
        translation_images[local_axis] = int(
            np.floor(repeat / 2.0 - centroid_fractional[fractional_axis] + 0.5)
        )
    translation = (
        translation_images[0] * source_cell[plane_axes[0]]
        + translation_images[1] * source_cell[plane_axes[1]]
    )
    metal_positions = original_positions + translation
    actual_metal_ids = [
        _map_periodic_vertex_to_supercell(
            supercell,
            position,
            vertex["element"],
        )
        for position, vertex in zip(metal_positions, original_vertices)
    ]

    adsorbate_positions = np.asarray(adsorbate.positions, dtype=float)
    adsorbate_symbols = np.asarray(adsorbate.get_chemical_symbols())
    anchor_id = int(probe["adsorbate_atom_ids"]["anchor"])
    carbon_id = int(probe["adsorbate_atom_ids"]["carbon"])
    donor_ids = [int(value) for value in probe["adsorbate_atom_ids"]["oxygen_donors"]]
    anchor_origin = adsorbate_positions[anchor_id]
    donor_directions = [
        (adsorbate_positions[donor_id] - anchor_origin)
        / np.linalg.norm(adsorbate_positions[donor_id] - anchor_origin)
        for donor_id in donor_ids
    ]
    carbon_direction = adsorbate_positions[carbon_id] - anchor_origin
    carbon_direction /= np.linalg.norm(carbon_direction)
    denticity = int(site["denticity"])
    binding_mode = site.get("binding_mode", "bridging_distinct_metals")
    preferred_distance = float(
        preparation_contract["preferred_metal_oxygen_distance_A"]
    )
    accepted_minimum, accepted_maximum = (
        float(value)
        for value in preparation_contract["accepted_metal_oxygen_distance_A"]
    )
    alignment_weight = int(
        preparation_contract["methyl_outward_alignment_weight"]
    )
    feasible = []
    if binding_mode == "bridging_distinct_metals":
        height_minimum, height_maximum, height_step = (
            float(value) for value in preparation_contract["anchor_height_search_A"]
        )
        height_values = np.arange(
            height_minimum,
            height_maximum + 0.5 * height_step,
            height_step,
        )
        placement_trials = []
        for height in height_values:
            anchor_position = np.mean(metal_positions, axis=0) + height * normal
            target_directions = [
                (position - anchor_position)
                / np.linalg.norm(position - anchor_position)
                for position in metal_positions
            ]
            for donor_permutation in permutations(range(len(donor_ids)), denticity):
                rotation = _proper_rotation(
                    [donor_directions[index] for index in donor_permutation]
                    + [carbon_direction] * alignment_weight,
                    target_directions + [normal] * alignment_weight,
                )
                placed = (
                    (adsorbate_positions - anchor_origin) @ rotation
                    + anchor_position
                )
                placement_trials.append(
                    {
                        "anchor_height_A": float(height),
                        "chelate_donor_midpoint_height_A": None,
                        "chelate_azimuth_degrees": None,
                        "rotation": rotation,
                        "placed": placed,
                        "mapped_donor_ids": [
                            donor_ids[index] for index in donor_permutation
                        ],
                        "donor_permutation": list(donor_permutation),
                    }
                )
    elif binding_mode == "chelating_shared_metal":
        if denticity != 2 or len(metal_positions) != 1:
            raise ValueError(
                "chelating_shared_metal requires two donors and one unique metal vertex"
            )
        donor_midpoint = np.mean(adsorbate_positions[donor_ids[:2]], axis=0)
        donor_axis = adsorbate_positions[donor_ids[1]] - adsorbate_positions[donor_ids[0]]
        donor_axis /= np.linalg.norm(donor_axis)
        height_minimum, height_maximum, height_step = (
            float(value)
            for value in preparation_contract[
                "chelate_donor_midpoint_height_search_A"
            ]
        )
        height_values = np.arange(
            height_minimum,
            height_maximum + 0.5 * height_step,
            height_step,
        )
        azimuth_step = int(preparation_contract["chelate_azimuth_step_degrees"])
        u = np.asarray(frame["u_cartesian_unit"], dtype=float)
        v = np.asarray(frame["v_cartesian_unit"], dtype=float)
        placement_trials = []
        for height in height_values:
            target_midpoint = metal_positions[0] + height * normal
            for azimuth in range(0, 360, azimuth_step):
                radians = math.radians(azimuth)
                target_donor_axis = math.cos(radians) * u + math.sin(radians) * v
                rotation = _proper_rotation(
                    [donor_axis] + [carbon_direction] * alignment_weight,
                    [target_donor_axis] + [normal] * alignment_weight,
                )
                placed = (
                    (adsorbate_positions - donor_midpoint) @ rotation
                    + target_midpoint
                )
                anchor_height = float(
                    np.dot(placed[anchor_id] - metal_positions[0], normal)
                )
                placement_trials.append(
                    {
                        "anchor_height_A": anchor_height,
                        "chelate_donor_midpoint_height_A": float(height),
                        "chelate_azimuth_degrees": azimuth,
                        "rotation": rotation,
                        "placed": placed,
                        "mapped_donor_ids": donor_ids[:2],
                        "donor_permutation": [0, 1],
                    }
                )
    else:
        raise ValueError(f"Unsupported binding mode {binding_mode!r}")

    donor_vertex_indices = [
        int(mapping["metal_vertex_index"])
        for mapping in site["donor_metal_mapping"]
    ]
    for trial in placement_trials:
        height = trial["anchor_height_A"]
        rotation = trial["rotation"]
        placed = trial["placed"]
        mapped_donor_ids = trial["mapped_donor_ids"]
        metal_oxygen_distances = [
            float(
                np.linalg.norm(
                    placed[donor_id]
                    - metal_positions[donor_vertex_indices[index]]
                )
            )
            for index, donor_id in enumerate(mapped_donor_ids)
        ]
        outward_cosine = float(carbon_direction @ rotation @ normal)
        mapped_distance_tolerances = [
            closed_distance_window_tolerance_A(
                distance,
                accepted_minimum,
                accepted_maximum,
            )
            for distance in metal_oxygen_distances
        ]
        mapped_distances_pass = all(
            distance_within_closed_window(
                distance,
                accepted_minimum,
                accepted_maximum,
            )
            for distance in metal_oxygen_distances
        )
        outward_pass = outward_cosine >= float(
            preparation_contract["minimum_methyl_outward_cosine"]
        )
        if not (mapped_distances_pass and outward_pass):
            continue
        allowed_contacts = {
            (donor_id, actual_metal_ids[donor_vertex_indices[index]])
            for index, donor_id in enumerate(mapped_donor_ids)
        }
        forbidden_clearance = _probe_substrate_clearance(
            supercell,
            adsorbate_symbols,
            placed,
            allowed_contacts,
        )
        unmapped_donors = [
            donor_id for donor_id in donor_ids if donor_id not in mapped_donor_ids
        ]
        unmapped_clearance = _unmapped_donor_metal_clearance(
            supercell,
            placed,
            unmapped_donors,
            discovery_contract["surface_metal_elements"],
        )
        checks = {
            "mapped_metal_oxygen_distances": mapped_distances_pass,
            "methyl_points_outward": outward_pass,
            "probe_substrate_heavy_clearance": forbidden_clearance
            >= float(
                preparation_contract[
                    "minimum_probe_substrate_heavy_distance_A"
                ]
            ),
            "unmapped_donor_not_precoordinated": (
                unmapped_clearance is None
                or unmapped_clearance
                >= float(
                    preparation_contract[
                        "minimum_unmapped_donor_metal_distance_A"
                    ]
                )
            ),
        }
        if not all(checks.values()):
            continue
        rms_distance_error = float(
            np.sqrt(
                np.mean(
                    (
                        np.asarray(metal_oxygen_distances)
                        - preferred_distance
                    )
                    ** 2
                )
            )
        )
        unmapped_penalty = (
            0.0
            if unmapped_clearance is None
            else 2.0 * max(0.0, 2.8 - unmapped_clearance)
        )
        score = (
            rms_distance_error
            + 0.05
            * abs(
                (
                    trial["chelate_donor_midpoint_height_A"]
                    if binding_mode == "chelating_shared_metal"
                    else height
                )
                - float(
                    preparation_contract[
                        "chelate_donor_midpoint_height_reference_A"
                    ]
                    if binding_mode == "chelating_shared_metal"
                    else preparation_contract["anchor_height_reference_A"]
                )
            )
            + unmapped_penalty
        )
        feasible_item = {
            "score": score,
            "anchor_height_A": float(height),
            "probe_positions_A": placed.tolist(),
            "mapped_adsorbate_donor_ids": mapped_donor_ids,
            "donor_permutation": trial["donor_permutation"],
            "metal_oxygen_distances_A": metal_oxygen_distances,
            "metal_oxygen_comparison_tolerances_A": mapped_distance_tolerances,
            "metal_oxygen_rms_error_from_preferred_A": rms_distance_error,
            "methyl_outward_cosine": outward_cosine,
            "minimum_forbidden_probe_substrate_heavy_distance_A": (
                forbidden_clearance
            ),
            "minimum_unmapped_donor_metal_distance_A": unmapped_clearance,
            "checks": checks,
        }
        if "binding_mode" in site:
            feasible_item["binding_mode"] = binding_mode
        if binding_mode == "chelating_shared_metal":
            feasible_item["chelate_donor_midpoint_height_A"] = trial[
                "chelate_donor_midpoint_height_A"
            ]
            feasible_item["chelate_azimuth_degrees"] = trial[
                "chelate_azimuth_degrees"
            ]
        feasible.append(feasible_item)
    if not feasible:
        raise ValueError(
            f"No valid rigid probe placement for {site['site_combination_id']}"
        )
    selected = min(
        feasible,
        key=lambda item: (
            item["score"],
            item["anchor_height_A"],
            item.get("chelate_azimuth_degrees") or 0,
            item["donor_permutation"],
        ),
    )
    probe_positions = np.asarray(selected["probe_positions_A"], dtype=float)
    periodic_probe_distance = _minimum_periodic_probe_image_distance(
        probe_positions,
        np.asarray(supercell.cell, dtype=float),
        plane_axes,
    )
    if periodic_probe_distance < float(
        preparation_contract["minimum_probe_image_separation_A"]
    ):
        raise ValueError(
            f"Probe image separation failed for {site['site_combination_id']}"
        )

    substrate_symbols = np.asarray(supercell.get_chemical_symbols())
    substrate_positions = np.asarray(supercell.positions, dtype=float)
    surface_parent_element = discovery_contract.get("surface_parent_element", "O")
    oxygen_ids = np.flatnonzero(substrate_symbols == surface_parent_element)
    normal_coordinates = substrate_positions @ normal
    outermost_oxygen = float(np.max(normal_coordinates[oxygen_ids]))
    top_oxygen_ids = [
        int(index)
        for index in oxygen_ids
        if outermost_oxygen - normal_coordinates[index]
        <= float(preparation_contract["surface_oxygen_depth_A"]) + 1.0e-10
    ]
    site_center = np.mean(metal_positions, axis=0)
    lateral_radius = float(preparation_contract["proton_parent_lateral_radius_A"])
    supercell_vectors = [
        np.asarray(supercell.cell[axis], dtype=float) for axis in plane_axes
    ]
    surface_area = float(np.linalg.norm(np.cross(*supercell_vectors)))
    minimum_altitude = min(
        surface_area / np.linalg.norm(supercell_vectors[0]),
        surface_area / np.linalg.norm(supercell_vectors[1]),
    )
    image_radius = max(1, int(math.ceil(lateral_radius / minimum_altitude)) + 1)
    parents = []
    for atom_id in top_oxygen_ids:
        for image_i, image_j in product(
            range(-image_radius, image_radius + 1), repeat=2
        ):
            position = (
                substrate_positions[atom_id]
                + image_i * supercell_vectors[0]
                + image_j * supercell_vectors[1]
            )
            vector = position - site_center
            lateral_vector = vector - np.dot(vector, normal) * normal
            lateral_distance = float(np.linalg.norm(lateral_vector))
            if lateral_distance <= lateral_radius + 1.0e-10:
                parents.append(
                    {
                        "substrate_atom_id": atom_id,
                        "atom_index_base": 0,
                        "lattice_image": [image_i, image_j],
                        "element": str(substrate_symbols[atom_id]),
                        "position_A": position.tolist(),
                        "lateral_distance_to_site_A": lateral_distance,
                    }
                )
    parent_keys = {
        (
            parent["substrate_atom_id"],
            *parent["lattice_image"],
        ): parent
        for parent in parents
    }
    parents = sorted(
        parent_keys.values(),
        key=lambda item: (
            item["lateral_distance_to_site_A"],
            item["substrate_atom_id"],
            item["lattice_image"],
        ),
    )
    proton_placements = []
    oh_distance = float(preparation_contract["surface_oxygen_hydrogen_bond_A"])
    heavy_probe_ids = np.flatnonzero(adsorbate_symbols != "H")
    probe_hydrogen_ids = np.flatnonzero(adsorbate_symbols == "H")
    released_protons = int(probe["released_protons"])
    for selected_parents in combinations(parents, released_protons):
        hydrogen_positions = np.asarray(
            [
                np.asarray(parent["position_A"]) + oh_distance * normal
                for parent in selected_parents
            ]
        )
        hydrogen_hydrogen = (
            min(
                float(np.linalg.norm(hydrogen_positions[right] - hydrogen_positions[left]))
                for left, right in combinations(range(released_protons), 2)
            )
            if released_protons > 1
            else None
        )
        proton_probe_heavy = min(
            float(np.linalg.norm(hydrogen - probe_positions[probe_id]))
            for hydrogen in hydrogen_positions
            for probe_id in heavy_probe_ids
        )
        proton_probe_hydrogen = (
            min(
                float(np.linalg.norm(hydrogen - probe_positions[probe_id]))
                for hydrogen in hydrogen_positions
                for probe_id in probe_hydrogen_ids
            )
            if len(probe_hydrogen_ids)
            else None
        )
        checks = {
            "proton_proton_clearance": hydrogen_hydrogen is None
            or hydrogen_hydrogen
            >= float(preparation_contract["minimum_proton_proton_distance_A"]),
            "proton_probe_heavy_clearance": proton_probe_heavy
            >= float(
                preparation_contract["minimum_proton_probe_heavy_distance_A"]
            ),
            "proton_probe_hydrogen_clearance": proton_probe_hydrogen is None
            or proton_probe_hydrogen
            >= float(preparation_contract["minimum_proton_proton_distance_A"]),
        }
        if not all(checks.values()):
            continue
        proton_placements.append(
            {
                "surface_proton_parent_vertices": list(selected_parents),
                "hydrogen_positions_A": hydrogen_positions.tolist(),
                "hydrogen_hydrogen_distance_A": hydrogen_hydrogen,
                "minimum_proton_probe_heavy_distance_A": proton_probe_heavy,
                "minimum_proton_probe_hydrogen_distance_A": proton_probe_hydrogen,
                "checks": checks,
            }
        )
    proton_placements.sort(
        key=lambda item: (
            sum(
                parent["lateral_distance_to_site_A"]
                for parent in item["surface_proton_parent_vertices"]
            ),
            [
                (
                    parent["substrate_atom_id"],
                    parent["lattice_image"],
                )
                for parent in item["surface_proton_parent_vertices"]
            ],
        )
    )
    for number, placement in enumerate(proton_placements, 1):
        placement["proton_placement_id"] = f"proton-placement-{number:03d}"
    if not proton_placements:
        raise ValueError(
            f"No valid surface-proton placement for {site['site_combination_id']}"
        )
    return {
        "site_combination_id": site["site_combination_id"],
        "denticity": denticity,
        **({"binding_mode": binding_mode} if "binding_mode" in site else {}),
        "source_site": site,
        "source_to_test_supercell_translation": translation_images,
        "actual_test_supercell_metal_atom_ids": actual_metal_ids,
        "actual_metal_positions_A": metal_positions.tolist(),
        "probe_placement": selected,
        "minimum_periodic_probe_image_distance_A": periodic_probe_distance,
        "surface_proton_parent_candidate_count": len(parents),
        "valid_surface_proton_placement_count": len(proton_placements),
        "surface_proton_placements": proton_placements,
    }


def resolve_adsorption_probe_preparation_plan(
    substrate_name: str,
    enumeration_run: Path,
    catalog_directory: Path | None = None,
    *,
    anchor_family: str | None = None,
) -> dict:
    """Plan neutral-stoichiometry probe/H starting structures without optimization."""

    recipe = load_substrate_recipe(
        substrate_name, catalog_directory, anchor_family=anchor_family
    )
    discovery = recipe["adsorption"].get("site_discovery")
    if not isinstance(discovery, dict):
        raise ValueError(f"Substrate {recipe['key']!r} has no site-discovery contract")
    evidence = _load_site_enumeration_evidence(enumeration_run)
    if evidence["substrate_key"] != recipe["key"]:
        raise ValueError("Site-enumeration substrate does not match the requested catalog")
    current_enumeration = resolve_adsorption_site_enumeration_plan(
        substrate_name, catalog_directory, anchor_family=anchor_family
    )
    if current_enumeration["fingerprint"] != evidence["fingerprint"]:
        raise ValueError("Site-enumeration evidence is stale against current geometry")
    if evidence["source_surface_sha256"] != recipe["source"]["sha256"]:
        raise ValueError("Site-enumeration surface is stale against current catalog")
    probe, adsorbate = _load_probe_definition(recipe["probe_manifest_path"])
    if probe["key"] != discovery["probe"]:
        raise ValueError("Probe manifest key does not match site-discovery chemistry")
    if probe["released_protons"] != int(recipe["adsorption"]["released_protons"]):
        raise ValueError("Probe released-proton count does not match adsorption contract")
    source_surface = read(evidence["source_surface_path"])
    normal_axis = int(recipe["surface"]["normal_axis"])
    plane_axes = [axis for axis in range(3) if axis != normal_axis]
    source_cell = np.asarray(source_surface.cell, dtype=float)
    adsorbate_positions = np.asarray(adsorbate.positions, dtype=float)
    probe_diameter = max(
        float(np.linalg.norm(adsorbate_positions[right] - adsorbate_positions[left]))
        for left, right in combinations(range(len(adsorbate)), 2)
    )
    preparation = discovery["probe_preparation"]
    required_translation = (
        probe_diameter + float(preparation["minimum_probe_image_separation_A"])
    )
    source_shortest_translation = _shortest_in_plane_translation(
        source_cell, plane_axes
    )
    repeat = max(1, int(math.ceil(required_translation / source_shortest_translation)))
    repeat_vector = [repeat, repeat, repeat]
    repeat_vector[normal_axis] = 1
    test_supercell = source_surface.repeat(tuple(repeat_vector))
    discovery_for_placement = {
        **discovery,
        "surface_parent_element": recipe["adsorption"]["surface_parent_element"],
    }
    site_preparations = [
        _prepare_one_probe_site(
            site,
            test_supercell,
            source_cell,
            repeat,
            evidence["surface_frame"],
            probe,
            adsorbate,
            discovery_for_placement,
            preparation,
        )
        for site in evidence["site_combinations"]
    ]
    structure_count = sum(
        item["valid_surface_proton_placement_count"] for item in site_preparations
    )
    site_count_by_binding_mode = Counter(
        item["binding_mode"]
        for item in site_preparations
        if "binding_mode" in item
    )
    start_count_by_binding_mode = Counter(
        {
            binding_mode: sum(
                item["valid_surface_proton_placement_count"]
                for item in site_preparations
                if item.get("binding_mode") == binding_mode
            )
            for binding_mode in site_count_by_binding_mode
        }
    )
    combined_formula = Counter(test_supercell.get_chemical_symbols())
    combined_formula.update(adsorbate.get_chemical_symbols())
    combined_formula.update({"H": probe["released_protons"]})
    combined_formula = dict(sorted(combined_formula.items()))
    identity = {
        "package_schema_version": 2,
        "substrate_key": recipe["key"],
        "enumeration_fingerprint": evidence["fingerprint"],
        "probe_source_sha256": probe["source_sha256"],
        "preparation_contract": preparation,
        "test_supercell_repeat": repeat_vector,
        "site_preparations": site_preparations,
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema_version": 1,
        "mode": "adsorption_probe_preparation_plan",
        "package_schema_version": 2,
        "status": "passed_preparation_plan",
        "substrate": {
            "key": recipe["key"],
            "catalog_path": str(recipe["catalog_path"]),
            "catalog_sha256": _sha256(Path(recipe["catalog_path"])),
        },
        "site_enumeration_evidence": evidence,
        "source_surface": {
            "path": evidence["source_surface_path"],
            "sha256": evidence["source_surface_sha256"],
            "formula": _formula(source_surface.get_chemical_symbols()),
            "atom_count": len(source_surface),
        },
        "probe": probe,
        "probe_preparation_contract": preparation,
        "test_supercell": {
            "repeat": repeat_vector,
            "formula": _formula(test_supercell.get_chemical_symbols()),
            "atom_count": len(test_supercell),
            "cell_A": np.asarray(test_supercell.cell, dtype=float).tolist(),
            "source_shortest_in_plane_translation_A": source_shortest_translation,
            "deprotonated_probe_diameter_A": probe_diameter,
            "minimum_requested_probe_image_separation_A": float(
                preparation["minimum_probe_image_separation_A"]
            ),
            "selection_rule": (
                "smallest uniform in-plane integer repeat whose shortest lattice "
                "translation is at least probe diameter plus requested image clearance"
            ),
        },
        "neutral_stoichiometry": {
            "combined_formula": combined_formula,
            "combined_atom_count": len(test_supercell)
            + len(adsorbate)
            + probe["released_protons"],
            "fully_deprotonated_probe_formula": probe[
                "fully_deprotonated_formula"
            ],
            "released_surface_protons": probe["released_protons"],
            "reference_neutral_probe_formula": probe["neutral_formula"],
            "alternative_deprotonation_states_enumerated": False,
        },
        "site_preparations": site_preparations,
        "summary": {
            "site_combination_count": len(site_preparations),
            "bidentate_site_count": sum(
                item["denticity"] == 2 for item in site_preparations
            ),
            "tridentate_site_count": sum(
                item["denticity"] == 3 for item in site_preparations
            ),
            "starting_structure_count": structure_count,
            **(
                {
                    "site_count_by_binding_mode": dict(
                        sorted(site_count_by_binding_mode.items())
                    ),
                    "starting_structure_count_by_binding_mode": dict(
                        sorted(start_count_by_binding_mode.items())
                    ),
                }
                if site_count_by_binding_mode
                else {}
            ),
            "minimum_proton_placements_per_site": min(
                item["valid_surface_proton_placement_count"]
                for item in site_preparations
            ),
            "maximum_proton_placements_per_site": max(
                item["valid_surface_proton_placement_count"]
                for item in site_preparations
            ),
            "minimum_periodic_probe_image_distance_A": min(
                item["minimum_periodic_probe_image_distance_A"]
                for item in site_preparations
            ),
        },
        "fingerprint": fingerprint,
        "execution_contract": {
            "read_only": True,
            "executable": True,
            "starts_optimizer_or_gpu": False,
            "known_limitations": [
                "Rigid placements are starting geometries, not relaxed Site Prototypes.",
                "Surface-proton placements are geometrically screened but not energy ranked.",
                "No adsorption energy is calculated during preparation.",
            ],
        },
    }


def resolve_adsorption_probe_relaxation_atom_constraints(
    atoms,
    *,
    substrate_atom_count: int,
    probe_atom_ids: list[int],
    surface_proton_atom_ids: list[int],
    outward_normal,
    source_layers: int,
    minimum_layer_boundary_gap_A: float,
    contract: dict,
) -> dict:
    """Resolve immutable frozen/movable atom identities before probe relaxation."""

    substrate_atom_count = _positive_int(
        substrate_atom_count, "substrate_atom_count"
    )
    source_layers = _positive_int(source_layers, "source_layers")
    minimum_gap = _positive_number(
        minimum_layer_boundary_gap_A, "minimum_layer_boundary_gap_A"
    )
    if substrate_atom_count > len(atoms):
        raise ValueError("substrate_atom_count exceeds the structure atom count")
    if substrate_atom_count % source_layers:
        raise ValueError(
            "Substrate atom count is not divisible by the declared source-layer count"
        )
    if contract.get("protocol_version") != "adsorption_probe_relaxation_v1":
        raise ValueError("Unsupported adsorption probe-relaxation protocol")
    if contract.get("relax_cell") is not False:
        raise ValueError("Probe relaxation must keep the complete slab cell fixed")
    if contract.get("freeze_lower_substrate_layers") is not True:
        raise ValueError("Probe relaxation must freeze lower substrate layers")
    if contract.get("layer_partition") != (
        "equal_stoichiometric_blocks_by_outward_normal"
    ):
        raise ValueError("Unsupported probe-relaxation layer partition policy")
    movable_layer_count = _positive_int(
        contract.get("movable_substrate_layers_from_surface"),
        "probe_relaxation.movable_substrate_layers_from_surface",
    )
    if movable_layer_count >= source_layers:
        raise ValueError(
            "Probe relaxation must leave at least one lower substrate layer frozen"
        )
    if contract.get("probe_atoms_movable") is not True:
        raise ValueError("Probe atoms must remain movable during probe relaxation")
    if contract.get("surface_protons_movable") is not True:
        raise ValueError("Surface protons must remain movable during probe relaxation")
    if contract.get("persist_atom_ids_before_relaxation") is not True:
        raise ValueError("Probe-relaxation atom identities must be persisted")

    normal = np.asarray(outward_normal, dtype=float)
    if normal.shape != (3,) or not np.all(np.isfinite(normal)):
        raise ValueError("outward_normal must contain three finite components")
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 1.0e-12:
        raise ValueError("outward_normal must be nonzero")
    normal /= normal_norm

    all_ids = set(range(len(atoms)))
    substrate_ids = set(range(substrate_atom_count))
    probe_ids = [int(value) for value in probe_atom_ids]
    proton_ids = [int(value) for value in surface_proton_atom_ids]
    declared_mobile_non_substrate = probe_ids + proton_ids
    if (
        len(set(declared_mobile_non_substrate)) != len(declared_mobile_non_substrate)
        or any(value not in all_ids for value in declared_mobile_non_substrate)
        or substrate_ids.intersection(declared_mobile_non_substrate)
    ):
        raise ValueError("Probe/proton atom IDs must be unique non-substrate atoms")
    undeclared = all_ids - substrate_ids - set(declared_mobile_non_substrate)
    if undeclared:
        raise ValueError(
            "Every non-substrate atom must be declared as probe or surface proton"
        )

    positions = np.asarray(atoms.positions[:substrate_atom_count], dtype=float)
    normal_coordinates = positions @ normal
    order = np.argsort(normal_coordinates, kind="stable")
    atoms_per_layer = substrate_atom_count // source_layers
    symbols = atoms.get_chemical_symbols()
    layer_groups = []
    for layer_index in range(source_layers):
        start = layer_index * atoms_per_layer
        stop = start + atoms_per_layer
        atom_ids = sorted(int(value) for value in order[start:stop])
        coordinates = normal_coordinates[atom_ids]
        layer_groups.append(
            {
                "layer_index_from_bulk_side": layer_index + 1,
                "layer_index_from_surface": source_layers - layer_index,
                "atom_count": len(atom_ids),
                "formula": _formula(symbols[atom_id] for atom_id in atom_ids),
                "atom_ids_0_based": atom_ids,
                "atom_ids_1_based": [atom_id + 1 for atom_id in atom_ids],
                "minimum_normal_coordinate_A": float(np.min(coordinates)),
                "maximum_normal_coordinate_A": float(np.max(coordinates)),
            }
        )
    layer_formulas = [group["formula"] for group in layer_groups]
    if any(formula != layer_formulas[0] for formula in layer_formulas[1:]):
        raise ValueError(
            "Equal normal-coordinate blocks are not stoichiometrically identical layers"
        )
    boundary_gaps = []
    for lower_index in range(source_layers - 1):
        lower = layer_groups[lower_index]
        upper = layer_groups[lower_index + 1]
        gap = (
            upper["minimum_normal_coordinate_A"]
            - lower["maximum_normal_coordinate_A"]
        )
        if gap < minimum_gap:
            raise ValueError(
                "A substrate layer boundary does not pass the configured normal gap"
            )
        boundary_gaps.append(
            {
                "lower_layer_index_from_bulk_side": lower_index + 1,
                "upper_layer_index_from_bulk_side": lower_index + 2,
                "gap_A": float(gap),
            }
        )

    movable_layer_groups = layer_groups[-movable_layer_count:]
    frozen_layer_groups = layer_groups[:-movable_layer_count]
    movable_substrate_ids = sorted(
        atom_id
        for group in movable_layer_groups
        for atom_id in group["atom_ids_0_based"]
    )
    frozen_substrate_ids = sorted(
        atom_id
        for group in frozen_layer_groups
        for atom_id in group["atom_ids_0_based"]
    )
    movable_ids = sorted(movable_substrate_ids + probe_ids + proton_ids)
    frozen_substrate_formula = _formula(
        symbols[atom_id] for atom_id in frozen_substrate_ids
    )
    movable_substrate_formula = _formula(
        symbols[atom_id] for atom_id in movable_substrate_ids
    )
    probe_formula = _formula(symbols[atom_id] for atom_id in probe_ids)
    surface_proton_formula = _formula(symbols[atom_id] for atom_id in proton_ids)
    atom_selection_contract = {
        field: contract[field]
        for field in (
            "protocol_version",
            "relax_cell",
            "freeze_lower_substrate_layers",
            "movable_substrate_layers_from_surface",
            "layer_partition",
            "probe_atoms_movable",
            "surface_protons_movable",
            "persist_atom_ids_before_relaxation",
        )
    }
    selection_identity = {
        "protocol_version": contract["protocol_version"],
        "contract": atom_selection_contract,
        "atom_count": len(atoms),
        "substrate_atom_count": substrate_atom_count,
        "outward_normal_cartesian_unit": normal.tolist(),
        "layer_groups": [
            {
                "layer_index_from_bulk_side": group["layer_index_from_bulk_side"],
                "layer_index_from_surface": group["layer_index_from_surface"],
                "atom_count": group["atom_count"],
                "formula": group["formula"],
                "atom_ids_0_based": group["atom_ids_0_based"],
            }
            for group in layer_groups
        ],
        "frozen_atom_ids_0_based": frozen_substrate_ids,
        "movable_substrate_atom_ids_0_based": movable_substrate_ids,
        "movable_probe_atom_ids_0_based": probe_ids,
        "movable_surface_proton_atom_ids_0_based": proton_ids,
    }
    selection_sha256 = hashlib.sha256(
        json.dumps(selection_identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "protocol_version": contract["protocol_version"],
        "fixed_cell": True,
        "layer_partition": contract["layer_partition"],
        "outward_normal_cartesian_unit": normal.tolist(),
        "source_layer_count": source_layers,
        "movable_substrate_layer_count": movable_layer_count,
        "frozen_substrate_layer_count": source_layers - movable_layer_count,
        "atoms_per_substrate_layer": atoms_per_layer,
        "minimum_layer_boundary_gap_A": minimum_gap,
        "layer_boundary_gaps": boundary_gaps,
        "layer_groups": layer_groups,
        "frozen_substrate_atom_ids_0_based": frozen_substrate_ids,
        "frozen_substrate_atom_ids_1_based": [
            atom_id + 1 for atom_id in frozen_substrate_ids
        ],
        "movable_substrate_atom_ids_0_based": movable_substrate_ids,
        "movable_substrate_atom_ids_1_based": [
            atom_id + 1 for atom_id in movable_substrate_ids
        ],
        "movable_probe_atom_ids_0_based": probe_ids,
        "movable_probe_atom_ids_1_based": [atom_id + 1 for atom_id in probe_ids],
        "movable_surface_proton_atom_ids_0_based": proton_ids,
        "movable_surface_proton_atom_ids_1_based": [
            atom_id + 1 for atom_id in proton_ids
        ],
        "formulas": {
            "frozen_substrate": frozen_substrate_formula,
            "movable_substrate": movable_substrate_formula,
            "movable_probe": probe_formula,
            "movable_surface_protons": surface_proton_formula,
        },
        "frozen_atom_ids_0_based": frozen_substrate_ids,
        "frozen_atom_ids_1_based": [atom_id + 1 for atom_id in frozen_substrate_ids],
        "movable_atom_ids_0_based": movable_ids,
        "movable_atom_ids_1_based": [atom_id + 1 for atom_id in movable_ids],
        "counts": {
            "total_atoms": len(atoms),
            "frozen_atoms": len(frozen_substrate_ids),
            "movable_atoms": len(movable_ids),
            "movable_substrate_atoms": len(movable_substrate_ids),
            "movable_probe_atoms": len(probe_ids),
            "movable_surface_protons": len(proton_ids),
        },
        "selection_sha256": selection_sha256,
        "selection_timing": (
            "resolve once from the unrelaxed starting structure and reuse the exact "
            "persisted atom IDs throughout relaxation"
        ),
    }


def resolve_adsorption_probe_relaxation_plan(
    substrate_name: str,
    preparation_run: Path,
    model_path: Path,
    declared_model_elements: list[str],
    catalog_directory: Path | None = None,
    structure_directory: Path | None = None,
    *,
    default_dtype: str = "float32",
    device: str = "cuda",
    anchor_family: str | None = None,
) -> dict:
    """Resolve all adsorption-reference and candidate relaxations without running MACE."""

    recipe = load_substrate_recipe(
        substrate_name, catalog_directory, anchor_family=anchor_family
    )
    discovery = recipe["adsorption"].get("site_discovery")
    if not isinstance(discovery, dict):
        raise ValueError(f"Substrate {recipe['key']!r} has no site-discovery contract")
    relaxation = discovery["probe_relaxation"]
    evidence = _load_probe_preparation_evidence(preparation_run)
    if evidence["substrate_key"] != recipe["key"]:
        raise ValueError("Probe-preparation substrate does not match the requested catalog")
    if evidence["source_surface_sha256"] != recipe["source"]["sha256"]:
        raise ValueError("Probe-preparation surface is stale against the current catalog")
    preparation_plan = evidence["preparation_plan"]
    if preparation_plan.get("probe_preparation_contract") != discovery["probe_preparation"]:
        raise ValueError("Probe-preparation contract is stale against the current catalog")
    if preparation_plan.get("test_supercell") != (
        json.loads(
            Path(evidence["starting_structures_report_path"]).read_text(encoding="utf-8")
        ).get("test_supercell")
    ):
        raise ValueError("Probe-preparation test-supercell records are inconsistent")

    probe, _ = _load_probe_definition(Path(recipe["probe_manifest_path"]))
    if probe["source_sha256"] != evidence["neutral_probe_source_sha256"]:
        raise ValueError("Neutral probe snapshot is stale against the maintained source")
    if _probe_chemistry_identity(probe) != _probe_chemistry_identity(
        preparation_plan.get("probe")
    ):
        raise ValueError("Probe-preparation chemistry record is stale")

    structure_manifest = _load_structure_manifest(
        substrate_name, structure_directory
    )
    parent = _validated_registered_bulk_parent(structure_manifest)
    if parent is None:
        raise ValueError(f"No accepted Bulk Parent for {substrate_name!r}")
    model_path = Path(model_path).expanduser().resolve()
    if not model_path.is_file():
        raise ValueError(f"Model file does not exist: {model_path}")
    model_sha256 = _sha256(model_path)
    parent_potential = parent["potential"]
    if model_sha256 != parent_potential.get("model_sha256"):
        raise ValueError(
            "Probe-relaxation model hash must match the accepted Bulk Parent model"
        )
    if default_dtype not in {"float32", "float64"}:
        raise ValueError("default_dtype must be float32 or float64")
    if default_dtype != parent_potential.get("default_dtype"):
        raise ValueError(
            "Probe-relaxation dtype must match the accepted Bulk Parent dtype"
        )
    if not isinstance(device, str) or not device.strip():
        raise ValueError("device must be a non-empty MACE device string")
    model_elements = sorted(
        {
            _element(value, "declared_model_elements")
            for value in declared_model_elements
        }
    )
    expected_combined_formula = preparation_plan["neutral_stoichiometry"][
        "combined_formula"
    ]
    required_elements = sorted(expected_combined_formula)
    missing_elements = sorted(set(required_elements) - set(model_elements))
    if missing_elements:
        raise ValueError(
            "Declared model elements do not cover probe relaxation: "
            + ", ".join(missing_elements)
        )

    clean_path = Path(evidence["clean_test_supercell_path"])
    clean_atoms = read(clean_path)
    source_layers = int(recipe["slab"]["source_layers"])
    layer_gap = float(recipe["slab"]["layer_gap_A"])
    surface_frame = preparation_plan["site_enumeration_evidence"]["surface_frame"]
    outward_normal = surface_frame["outward_normal_cartesian_unit"]
    clean_constraints = resolve_adsorption_probe_relaxation_atom_constraints(
        clean_atoms,
        substrate_atom_count=len(clean_atoms),
        probe_atom_ids=[],
        surface_proton_atom_ids=[],
        outward_normal=outward_normal,
        source_layers=source_layers,
        minimum_layer_boundary_gap_A=layer_gap,
        contract=relaxation,
    )
    if _formula(clean_atoms.get_chemical_symbols()) != preparation_plan[
        "test_supercell"
    ]["formula"]:
        raise ValueError("Clean reference formula does not match the prepared test slab")

    neutral_atoms = read(Path(evidence["neutral_probe_source_path"]))
    if _formula(neutral_atoms.get_chemical_symbols()) != probe["neutral_formula"]:
        raise ValueError("Neutral probe reference formula changed")
    vacuum = float(relaxation["isolated_neutral_probe_vacuum_A"])
    neutral_extent = np.ptp(np.asarray(neutral_atoms.positions, dtype=float), axis=0)
    neutral_cell_lengths = (neutral_extent + 2.0 * vacuum).tolist()

    candidates = []
    shared_candidate_constraints = None
    selection_hashes = set()
    by_site = Counter()
    by_denticity = Counter()
    by_binding_mode = Counter()
    for record in evidence["starting_structures"]:
        path = Path(record["absolute_path"])
        atoms = read(path)
        if len(atoms) != int(record["atom_count"]):
            raise ValueError(f"Starting-structure atom count changed: {path}")
        observed_formula = _formula(atoms.get_chemical_symbols())
        if observed_formula != record["formula"] or observed_formula != expected_combined_formula:
            raise ValueError(f"Starting-structure formula changed: {path}")
        substrate_atom_count = int(record["substrate_atom_count"])
        if substrate_atom_count != len(clean_atoms):
            raise ValueError(f"Starting-structure substrate count changed: {path}")
        if not np.allclose(
            np.asarray(atoms.cell, dtype=float),
            np.asarray(clean_atoms.cell, dtype=float),
            atol=1.0e-10,
            rtol=0.0,
        ) or not np.array_equal(
            np.asarray(atoms.pbc, dtype=bool),
            np.asarray(clean_atoms.pbc, dtype=bool),
        ):
            raise ValueError(f"Starting-structure cell or periodicity changed: {path}")
        constraints = resolve_adsorption_probe_relaxation_atom_constraints(
            atoms,
            substrate_atom_count=substrate_atom_count,
            probe_atom_ids=record["probe_atom_ids"],
            surface_proton_atom_ids=record["surface_proton_atom_ids"],
            outward_normal=outward_normal,
            source_layers=source_layers,
            minimum_layer_boundary_gap_A=layer_gap,
            contract=relaxation,
        )
        if shared_candidate_constraints is None:
            shared_candidate_constraints = constraints
        elif constraints != shared_candidate_constraints:
            raise ValueError(
                "Prepared candidates do not share one immutable atom-constraint selection"
            )
        selection_hashes.add(constraints["selection_sha256"])
        site_id = record["site_combination_id"]
        proton_id = record["proton_placement_id"]
        candidate_id = f"{site_id}--{proton_id}"
        denticity = int(record["denticity"])
        binding_mode = record.get("binding_mode")
        by_site[site_id] += 1
        by_denticity[denticity] += 1
        if binding_mode is not None:
            by_binding_mode[binding_mode] += 1
        candidates.append(
            {
                "candidate_id": candidate_id,
                "site_combination_id": site_id,
                "proton_placement_id": proton_id,
                "initial_denticity": denticity,
                **(
                    {"initial_binding_mode": binding_mode}
                    if binding_mode is not None
                    else {}
                ),
                "input_path": str(path),
                "input_sha256": record["sha256"],
                "formula": observed_formula,
                "atom_count": len(atoms),
                "substrate_atom_count": substrate_atom_count,
                "probe_atom_ids_0_based": list(record["probe_atom_ids"]),
                "surface_proton_atom_ids_0_based": list(
                    record["surface_proton_atom_ids"]
                ),
                "surface_proton_parent_vertices": record[
                    "surface_proton_parent_vertices"
                ],
                "atom_constraint_selection_sha256": constraints["selection_sha256"],
                "planned_output_directory": (
                    f"candidates/{site_id}/{proton_id}"
                ),
            }
        )
    if shared_candidate_constraints is None or len(selection_hashes) != 1:
        raise ValueError("No single shared candidate atom-constraint selection was found")
    if len(by_site) != evidence["site_combination_count"]:
        raise ValueError("Probe-relaxation plan does not cover every Site Combination")

    combined_reference_formula = Counter(clean_atoms.get_chemical_symbols())
    combined_reference_formula.update(neutral_atoms.get_chemical_symbols())
    if dict(sorted(combined_reference_formula.items())) != expected_combined_formula:
        raise ValueError("Adsorption-energy reference formulas do not balance")

    identity = {
        "execution_schema_version": 2,
        "substrate_key": recipe["key"],
        "preparation_fingerprint": evidence["fingerprint"],
        "preparation_manifest_sha256": evidence["run_manifest_sha256"],
        "clean_surface_sha256": evidence["source_surface_sha256"],
        "neutral_probe_sha256": evidence["neutral_probe_source_sha256"],
        "candidate_inputs": [
            {
                "candidate_id": candidate["candidate_id"],
                "input_sha256": candidate["input_sha256"],
                "atom_constraint_selection_sha256": candidate[
                    "atom_constraint_selection_sha256"
                ],
                **(
                    {"initial_binding_mode": candidate["initial_binding_mode"]}
                    if "initial_binding_mode" in candidate
                    else {}
                ),
            }
            for candidate in candidates
        ],
        "model_sha256": model_sha256,
        "default_dtype": default_dtype,
        "device": device,
        "probe_relaxation_contract": relaxation,
        "site_selection_policy": (
            "retain_stable_sites_and_report_rejected_geometric_combinations_v1"
        ),
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema_version": 1,
        "mode": "adsorption_probe_relaxation_plan",
        "status": "passed_plan_ready_for_execution",
        "substrate": {
            "key": recipe["key"],
            "catalog_path": str(recipe["catalog_path"]),
            "catalog_sha256": _sha256(Path(recipe["catalog_path"])),
            "structure_manifest_path": str(structure_manifest["manifest_path"]),
            "structure_manifest_sha256": _sha256(
                Path(structure_manifest["manifest_path"])
            ),
        },
        "accepted_bulk_parent": {
            "path": parent["path"],
            "sha256": parent["sha256"],
            "potential": parent_potential,
        },
        "probe_preparation_evidence": {
            key: value
            for key, value in evidence.items()
            if key not in {"preparation_plan", "starting_structures"}
        },
        "potential": {
            "model_path": str(model_path),
            "model_sha256": model_sha256,
            "required_elements": required_elements,
            "declared_model_elements": model_elements,
            "coverage": "passed",
            "coverage_basis": "maintainer_declaration_not_model_introspection",
            "default_dtype": default_dtype,
            "planned_device": device,
            "same_calculator_contract_for_every_task": True,
        },
        "probe_relaxation_contract": relaxation,
        "optimizer_contract": {
            "optimizer": relaxation["optimizer"],
            "force_tolerance_eV_A": float(relaxation["force_tolerance_eV_A"]),
            "maximum_step_A": float(relaxation["maximum_step_A"]),
            "maximum_steps": int(relaxation["maximum_steps"]),
            "force_gate_applies_to": "movable_atoms",
            "frozen_atom_forces": "recorded_diagnostic_not_optimizer_gate",
        },
        "site_selection_contract": {
            "policy": (
                "retain_stable_sites_and_report_rejected_geometric_combinations_v1"
            ),
            "geometric_combinations_without_any_valid_relaxed_candidate": (
                "retain_as_rejected_evidence_not_site_prototypes"
            ),
            "minimum_selected_site_count_for_partial_pass": 1,
        },
        "surface_frame": surface_frame,
        "site_validation_context": {
            "surface_metal_elements": list(discovery["surface_metal_elements"]),
            "allowed_final_denticities": list(discovery["denticities"]),
            "surface_parent_element": recipe["adsorption"]["surface_parent_element"],
            "probe_anchor_element": recipe["adsorption"]["anchor_element"],
            "probe_donor_element": "O",
            "probe": probe,
        },
        "reference_tasks": [
            {
                "task_id": "clean-test-slab-reference",
                "operation": "constrained_relaxation",
                "input_path": str(clean_path),
                "input_sha256": evidence["clean_test_supercell_sha256"],
                "formula": _formula(clean_atoms.get_chemical_symbols()),
                "atom_count": len(clean_atoms),
                "atom_constraints": clean_constraints,
                "policy": relaxation["clean_slab_reference_policy"],
                "planned_output_directory": "references/clean-test-slab",
            },
            {
                "task_id": "isolated-neutral-probe-reference",
                "operation": "unconstrained_internal_relaxation_with_fixed_center_of_mass",
                "input_path": evidence["neutral_probe_source_path"],
                "input_sha256": evidence["neutral_probe_source_sha256"],
                "formula": probe["neutral_formula"],
                "atom_count": probe["neutral_atom_count"],
                "pbc": [False, False, False],
                "vacuum_padding_each_side_A": vacuum,
                "planned_orthorhombic_cell_lengths_A": neutral_cell_lengths,
                "policy": relaxation["neutral_probe_reference_policy"],
                "planned_output_directory": "references/isolated-neutral-probe",
            },
        ],
        "shared_candidate_atom_constraints": shared_candidate_constraints,
        "candidate_relaxations": candidates,
        "summary": {
            "reference_task_count": 2,
            "candidate_relaxation_count": len(candidates),
            "total_calculator_task_count": len(candidates) + 2,
            "site_combination_count": len(by_site),
            "candidate_count_by_initial_denticity": {
                str(key): by_denticity[key] for key in sorted(by_denticity)
            },
            **(
                {
                    "candidate_count_by_initial_binding_mode": {
                        key: by_binding_mode[key] for key in sorted(by_binding_mode)
                    }
                }
                if by_binding_mode
                else {}
            ),
            "minimum_proton_placements_per_site": min(by_site.values()),
            "maximum_proton_placements_per_site": max(by_site.values()),
            "shared_candidate_atom_constraint_selection_sha256": next(
                iter(selection_hashes)
            ),
        },
        "adsorption_energy_definition": {
            "equation": (
                "E_ads=E_relaxed[slab+fully_deprotonated_probe+surface_H]"
                "-E_relaxed[clean_slab]-E_relaxed[isolated_neutral_acid]"
            ),
            "unit": "eV",
            "stoichiometry_balanced": True,
            "released_protons_in_adsorbed_system": probe["released_protons"],
            "per_site_energy_policy": (
                "lowest adsorption energy among independently valid proton placements"
            ),
            "invalid_relaxations_excluded_before_energy_selection": True,
        },
        "validation_contract": relaxation["validation"],
        "planned_outputs": {
            "run_directory_pattern": (
                "runs/adsorption-probe-relaxation/<substrate>/"
                f"{relaxation['protocol_version']}-<UTC>-{fingerprint[:12]}/"
            ),
            "per_relaxation_records": [
                "initial_structure",
                "relaxed_structure",
                "trajectory",
                "optimizer_log",
                "direct_progress",
                "relaxation_and_validation_report",
            ],
            "aggregate_records": [
                "plan_snapshot",
                "reference_energy_report",
                "all_candidate_energy_report",
                "per_site_lowest_valid_energy_report",
                "run_manifest",
            ],
            "immutability": "never reuse or overwrite an existing run directory",
        },
        "fingerprint": fingerprint,
        "execution_contract": {
            "read_only": True,
            "plan_resolved": True,
            "starts_mace_optimizer_or_gpu": False,
            "execution_implemented": True,
            "executable": True,
            "planned_execution_command": "adsorption-probe-relaxation-run",
            "readiness_issues": [],
        },
    }


def _relax_adsorption_task(
    atoms,
    *,
    calculator,
    constraint,
    optimizer_contract: dict,
    task_directory: Path,
    task_id: str,
) -> dict:
    """Run one ASE relaxation while preserving direct task-level evidence."""

    from ase.io.trajectory import Trajectory
    from ase.optimize import LBFGS

    task_directory.mkdir(parents=True)
    initial_path = task_directory / "initial.extxyz"
    relaxed_path = task_directory / "relaxed.extxyz"
    trajectory_path = task_directory / "optimization.traj"
    optimizer_log_path = task_directory / "optimizer.log"
    progress_path = task_directory / "progress.jsonl"
    write(initial_path, atoms, format="extxyz")
    initial_positions = np.asarray(atoms.positions, dtype=float).copy()
    initial_cell = np.asarray(atoms.cell, dtype=float).copy()
    atoms.set_constraint(constraint)
    atoms.calc = calculator
    trajectory = None
    try:
        initial_energy = float(atoms.get_potential_energy())
        optimizer = LBFGS(
            atoms,
            logfile=str(optimizer_log_path),
            maxstep=float(optimizer_contract["maximum_step_A"]),
        )
        trajectory = Trajectory(str(trajectory_path), "w", atoms)
        trajectory.write(atoms)
        optimizer.attach(trajectory.write, interval=1)

        def record_progress():
            forces = np.asarray(atoms.get_forces(), dtype=float)
            record = {
                "task_id": task_id,
                "optimizer_step": int(optimizer.get_number_of_steps()),
                "time_utc": datetime.now(timezone.utc).isoformat(),
                "energy_eV": float(atoms.get_potential_energy()),
                "maximum_constrained_force_eV_A": float(
                    np.max(np.linalg.norm(forces, axis=1))
                ),
            }
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

        optimizer.attach(record_progress, interval=1)
        record_progress()
        converged = bool(
            optimizer.run(
                fmax=float(optimizer_contract["force_tolerance_eV_A"]),
                steps=int(optimizer_contract["maximum_steps"]),
            )
        )
        final_energy = float(atoms.get_potential_energy())
        constrained_forces = np.asarray(atoms.get_forces(), dtype=float)
        raw_forces = np.asarray(atoms.get_forces(apply_constraint=False), dtype=float)
        trajectory.close()
        trajectory = None
        write(relaxed_path, atoms, format="extxyz")
        return {
            "status": "completed",
            "optimizer_converged": converged,
            "optimizer_steps": int(optimizer.get_number_of_steps()),
            "initial_energy_eV": initial_energy,
            "relaxed_energy_eV": final_energy,
            "relaxation_energy_eV": final_energy - initial_energy,
            "initial_positions_A": initial_positions,
            "initial_cell_A": initial_cell,
            "constrained_forces_eV_A": constrained_forces,
            "raw_forces_eV_A": raw_forces,
            "artifacts": {
                "initial_structure": initial_path,
                "relaxed_structure": relaxed_path,
                "trajectory": trajectory_path,
                "optimizer_log": optimizer_log_path,
                "progress": progress_path,
            },
        }
    finally:
        if trajectory is not None:
            trajectory.close()


def _maximum_mic_displacement(initial, final, cell, pbc, atom_ids) -> float:
    if not atom_ids:
        return 0.0
    return max(
        float(
            np.linalg.norm(
                _minimum_image_vector(
                    np.asarray(final[index]) - np.asarray(initial[index]),
                    np.asarray(cell, dtype=float),
                    np.asarray(pbc, dtype=bool),
                )
            )
        )
        for index in atom_ids
    )


def _resolved_probe_anchor_and_donors(
    probe_symbols: list[str], initial_graph: list[list[int]], context: dict
) -> tuple[int, list[int]]:
    """Resolve explicit probe atom identities even when the anchor element repeats."""

    probe_identity = context["probe"]["adsorbate_atom_ids"]
    anchor_local = int(probe_identity["anchor"])
    donor_local_ids = [int(value) for value in probe_identity["oxygen_donors"]]
    if (
        not 0 <= anchor_local < len(probe_symbols)
        or probe_symbols[anchor_local] != context["probe_anchor_element"]
        or any(
            not 0 <= index < len(probe_symbols)
            or probe_symbols[index] != context["probe_donor_element"]
            for index in donor_local_ids
        )
        or any(index not in initial_graph[anchor_local] for index in donor_local_ids)
    ):
        raise ValueError("Configured probe anchor/donor atom identities are inconsistent")
    return anchor_local, donor_local_ids


def evaluate_adsorption_probe_relaxation_candidate(
    plan: dict,
    candidate: dict,
    initial_atoms,
    relaxed_atoms,
    task_result: dict,
) -> dict:
    """Independently validate one relaxed adsorbed probe before energy selection."""

    validation = plan["validation_contract"]
    context = plan["site_validation_context"]
    constraints = plan["shared_candidate_atom_constraints"]
    substrate_count = int(candidate["substrate_atom_count"])
    probe_ids = list(candidate["probe_atom_ids_0_based"])
    proton_ids = list(candidate["surface_proton_atom_ids_0_based"])
    movable_ids = list(constraints["movable_atom_ids_0_based"])
    frozen_ids = list(constraints["frozen_atom_ids_0_based"])
    symbols = relaxed_atoms.get_chemical_symbols()
    initial_positions = np.asarray(initial_atoms.positions, dtype=float)
    final_positions = np.asarray(relaxed_atoms.positions, dtype=float)
    cell = np.asarray(relaxed_atoms.cell, dtype=float)
    pbc = np.asarray(relaxed_atoms.pbc, dtype=bool)

    probe_symbols = [symbols[index] for index in probe_ids]
    initial_probe = initial_atoms[probe_ids]
    relaxed_probe = relaxed_atoms[probe_ids]
    initial_graph = _molecular_adjacency(
        probe_symbols,
        np.asarray(initial_probe.positions, dtype=float),
        np.asarray(initial_atoms.cell, dtype=float),
        np.asarray(initial_atoms.pbc, dtype=bool),
    )
    final_graph = _molecular_adjacency(
        probe_symbols,
        np.asarray(relaxed_probe.positions, dtype=float),
        cell,
        pbc,
    )
    anchor_local, donor_local_ids = _resolved_probe_anchor_and_donors(
        probe_symbols, initial_graph, context
    )
    donor_global_ids = [probe_ids[index] for index in donor_local_ids]
    metal_ids = [
        index
        for index in range(substrate_count)
        if symbols[index] in set(context["surface_metal_elements"])
    ]
    distance_minimum, distance_maximum = (
        float(value) for value in validation["final_metal_oxygen_distance_A"]
    )
    contacts = []
    contacted_donors = set()
    for donor_id in donor_global_ids:
        for metal_id in metal_ids:
            vector = _minimum_image_vector(
                final_positions[donor_id] - final_positions[metal_id], cell, pbc
            )
            distance = float(np.linalg.norm(vector))
            if distance_within_closed_window(
                distance,
                distance_minimum,
                distance_maximum,
            ):
                contacted_donors.add(donor_id)
                contacts.append(
                    {
                        "probe_donor_atom_id_0_based": donor_id,
                        "surface_metal_atom_id_0_based": metal_id,
                        "surface_metal_element": symbols[metal_id],
                        "distance_A": distance,
                        "comparison_tolerance_A": (
                            closed_distance_window_tolerance_A(
                                distance,
                                distance_minimum,
                                distance_maximum,
                            )
                        ),
                    }
                )
    final_denticity = len(contacted_donors)

    oxygen_ids = [index for index, symbol in enumerate(symbols) if symbol == "O"]
    proton_assignments = []
    proton_policy_passed = True
    for proton_id in proton_ids:
        nearest = min(
            (
                float(
                    np.linalg.norm(
                        _minimum_image_vector(
                            final_positions[proton_id] - final_positions[oxygen_id],
                            cell,
                            pbc,
                        )
                    )
                ),
                oxygen_id,
            )
            for oxygen_id in oxygen_ids
        )
        assignment = {
            "surface_proton_atom_id_0_based": proton_id,
            "nearest_oxygen_atom_id_0_based": nearest[1],
            "nearest_oxygen_distance_A": nearest[0],
            "nearest_oxygen_is_substrate": nearest[1] < substrate_count,
        }
        proton_assignments.append(assignment)
        proton_policy_passed &= (
            assignment["nearest_oxygen_is_substrate"]
            and nearest[0]
            <= float(validation["maximum_surface_oxygen_hydrogen_distance_A"])
        )

    constrained_forces = np.asarray(task_result["constrained_forces_eV_A"])
    raw_forces = np.asarray(task_result["raw_forces_eV_A"])
    movable_fmax = max(
        float(np.linalg.norm(constrained_forces[index])) for index in movable_ids
    )
    frozen_raw_fmax = max(
        float(np.linalg.norm(raw_forces[index])) for index in frozen_ids
    )
    fixed_position_change = max(
        float(np.linalg.norm(final_positions[index] - initial_positions[index]))
        for index in frozen_ids
    )
    movable_substrate_displacement = _maximum_mic_displacement(
        initial_positions,
        final_positions,
        cell,
        pbc,
        constraints["movable_substrate_atom_ids_0_based"],
    )
    checks = {
        "optimizer_converged": bool(task_result["optimizer_converged"]),
        "movable_force": movable_fmax
        <= float(plan["optimizer_contract"]["force_tolerance_eV_A"]),
        "formula": _formula(symbols) == candidate["formula"],
        "atom_count": len(relaxed_atoms) == int(candidate["atom_count"]),
        "fixed_cell": bool(
            np.allclose(
                cell, task_result["initial_cell_A"], atol=1.0e-10, rtol=0.0
            )
        ),
        "frozen_positions_unchanged": fixed_position_change <= 1.0e-10,
        "probe_connectivity": initial_graph == final_graph,
        "final_denticity": final_denticity
        in set(context["allowed_final_denticities"]),
        "surface_protons_bound_to_substrate_oxygen": bool(proton_policy_passed),
        "movable_substrate_displacement": movable_substrate_displacement
        <= float(validation["maximum_movable_substrate_displacement_A"]),
    }
    return {
        "schema_version": 1,
        "candidate_id": candidate["candidate_id"],
        "status": "passed" if all(checks.values()) else "failed_validation",
        "valid_for_adsorption_energy_selection": all(checks.values()),
        "optimizer_steps": int(task_result["optimizer_steps"]),
        "relaxed_total_energy_eV": float(task_result["relaxed_energy_eV"]),
        "movable_maximum_force_eV_A": movable_fmax,
        "frozen_raw_maximum_force_eV_A": frozen_raw_fmax,
        "maximum_frozen_position_change_A": fixed_position_change,
        "maximum_movable_substrate_displacement_A": movable_substrate_displacement,
        "final_denticity": final_denticity,
        "final_donor_metal_contacts": contacts,
        "surface_proton_assignments": proton_assignments,
        "checks": checks,
    }


def execute_adsorption_probe_relaxation(
    plan: dict,
    output_root: Path,
    *,
    device: str | None = None,
) -> dict:
    """Run references and every adsorbed probe in one immutable MACE package."""

    if plan.get("mode") != "adsorption_probe_relaxation_plan":
        raise ValueError("Probe relaxation requires a resolved probe-relaxation plan")
    if not plan["execution_contract"]["executable"]:
        raise ValueError("Probe-relaxation plan has unresolved readiness issues")
    from ase.constraints import FixAtoms, FixCom

    requested_device = device or plan["potential"]["planned_device"]
    if requested_device != plan["potential"]["planned_device"]:
        raise ValueError("Execution device must match the planned device identity")
    model_path = Path(plan["potential"]["model_path"])
    if _sha256(model_path) != plan["potential"]["model_sha256"]:
        raise ValueError("Probe-relaxation model changed after planning")
    identity = {
        "plan_fingerprint": plan["fingerprint"],
        "device": requested_device,
        "default_dtype": plan["potential"]["default_dtype"],
    }
    run_fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_directory = (
        Path(output_root).expanduser().resolve()
        / plan["substrate"]["key"].casefold()
        / (
            f"{plan['probe_relaxation_contract']['protocol_version']}-"
            f"{timestamp}-{run_fingerprint[:12]}"
        )
    )
    if run_directory.exists():
        raise ValueError(f"Immutable probe-relaxation run already exists: {run_directory}")
    run_directory.mkdir(parents=True)
    plan_path = run_directory / "adsorption-probe-relaxation-plan.json"
    references_path = run_directory / "reference-energy-report.json"
    candidates_path = run_directory / "candidate-energy-report.json"
    selection_path = run_directory / "site-energy-selection-report.json"
    manifest_path = run_directory / "run-manifest.json"
    write_manifest(plan_path, plan)
    manifest = {
        "schema_version": 1,
        "mode": "adsorption_probe_relaxation_run",
        "status": "running",
        "fingerprint": run_fingerprint,
        "plan_fingerprint": plan["fingerprint"],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_directory": str(run_directory),
        "host": socket.gethostname(),
        "device": requested_device,
        "default_dtype": plan["potential"]["default_dtype"],
        "plan": {"path": plan_path.name, "sha256": _sha256(plan_path)},
        "artifacts": [],
    }
    write_manifest(manifest_path, manifest)
    calculator = None
    try:
        calculator, runtime = _make_mace_calculator(
            model_path,
            requested_device,
            plan["potential"]["default_dtype"],
        )
        reference_results = []
        reference_energies = {}
        for reference in plan["reference_tasks"]:
            reference_input_path = Path(reference["input_path"])
            if _sha256(reference_input_path) != reference["input_sha256"]:
                raise ValueError(
                    "Adsorption-energy reference changed after planning: "
                    f"{reference_input_path}"
                )
            atoms = read(reference_input_path)
            if reference["task_id"] == "clean-test-slab-reference":
                constraint = FixAtoms(
                    indices=reference["atom_constraints"]["frozen_atom_ids_0_based"]
                )
            else:
                atoms.set_cell(reference["planned_orthorhombic_cell_lengths_A"])
                atoms.set_pbc(False)
                atoms.center()
                constraint = FixCom()
            task_directory = run_directory / reference["planned_output_directory"]
            result = _relax_adsorption_task(
                atoms,
                calculator=calculator,
                constraint=constraint,
                optimizer_contract=plan["optimizer_contract"],
                task_directory=task_directory,
                task_id=reference["task_id"],
            )
            forces = np.asarray(result["constrained_forces_eV_A"])
            fmax = float(np.max(np.linalg.norm(forces, axis=1)))
            checks = {
                "optimizer_converged": bool(result["optimizer_converged"]),
                "force": fmax
                <= plan["optimizer_contract"]["force_tolerance_eV_A"],
                "formula": _formula(atoms.get_chemical_symbols())
                == reference["formula"],
            }
            if reference["task_id"] == "clean-test-slab-reference":
                frozen = reference["atom_constraints"]["frozen_atom_ids_0_based"]
                checks["fixed_cell"] = bool(
                    np.allclose(
                        np.asarray(atoms.cell, dtype=float),
                        result["initial_cell_A"],
                        atol=1.0e-10,
                        rtol=0.0,
                    )
                )
                checks["frozen_positions_unchanged"] = bool(
                    np.allclose(
                        np.asarray(atoms.positions, dtype=float)[frozen],
                        result["initial_positions_A"][frozen],
                        atol=1.0e-10,
                        rtol=0.0,
                    )
                )
            else:
                checks["nonperiodic"] = not bool(np.any(atoms.pbc))
            passed = all(checks.values())
            report = {
                "task_id": reference["task_id"],
                "status": "passed" if passed else "failed",
                "relaxed_energy_eV": result["relaxed_energy_eV"],
                "maximum_constrained_force_eV_A": fmax,
                "optimizer_steps": result["optimizer_steps"],
                "checks": checks,
                "artifacts": [
                    {
                        "role": role,
                        "path": str(path.relative_to(run_directory)),
                        "sha256": _sha256(path),
                    }
                    for role, path in result["artifacts"].items()
                ],
            }
            write_manifest(task_directory / "relaxation-report.json", report)
            reference_results.append(report)
            if passed:
                reference_energies[reference["task_id"]] = result["relaxed_energy_eV"]
            atoms.calc = None
        write_manifest(references_path, {"references": reference_results})
        if len(reference_energies) != 2:
            raise RuntimeError("Adsorption-energy reference relaxation failed")

        candidate_reports = []
        for candidate in plan["candidate_relaxations"]:
            input_path = Path(candidate["input_path"])
            if _sha256(input_path) != candidate["input_sha256"]:
                raise ValueError(f"Candidate changed after planning: {input_path}")
            initial_atoms = read(input_path)
            atoms = initial_atoms.copy()
            task_directory = run_directory / candidate["planned_output_directory"]
            result = _relax_adsorption_task(
                atoms,
                calculator=calculator,
                constraint=FixAtoms(
                    indices=plan["shared_candidate_atom_constraints"][
                        "frozen_atom_ids_0_based"
                    ]
                ),
                optimizer_contract=plan["optimizer_contract"],
                task_directory=task_directory,
                task_id=candidate["candidate_id"],
            )
            report = evaluate_adsorption_probe_relaxation_candidate(
                plan, candidate, initial_atoms, atoms, result
            )
            if report["valid_for_adsorption_energy_selection"]:
                report["adsorption_energy_eV"] = (
                    report["relaxed_total_energy_eV"]
                    - reference_energies["clean-test-slab-reference"]
                    - reference_energies["isolated-neutral-probe-reference"]
                )
            else:
                report["adsorption_energy_eV"] = None
            report["site_combination_id"] = candidate["site_combination_id"]
            report["proton_placement_id"] = candidate["proton_placement_id"]
            report["artifacts"] = [
                {
                    "role": role,
                    "path": str(path.relative_to(run_directory)),
                    "sha256": _sha256(path),
                }
                for role, path in result["artifacts"].items()
            ]
            write_manifest(task_directory / "relaxation-report.json", report)
            candidate_reports.append(report)
            atoms.calc = None
        write_manifest(candidates_path, {"candidates": candidate_reports})

        selections = []
        for site_id in sorted({item["site_combination_id"] for item in candidate_reports}):
            site_reports = [
                item
                for item in candidate_reports
                if item["site_combination_id"] == site_id
            ]
            valid = [
                item for item in site_reports if item["adsorption_energy_eV"] is not None
            ]
            valid.sort(
                key=lambda item: (
                    item["adsorption_energy_eV"],
                    item["candidate_id"],
                )
            )
            selected = valid[0] if valid else None
            selections.append(
                {
                    "site_combination_id": site_id,
                    "candidate_count": len(site_reports),
                    "valid_candidate_count": len(valid),
                    "status": "passed" if selected else "failed_no_valid_candidate",
                    "selected_candidate_id": selected["candidate_id"] if selected else None,
                    "adsorption_energy_eV": selected["adsorption_energy_eV"] if selected else None,
                    "valid_candidate_energy_ranking": [
                        {
                            "rank": rank,
                            "candidate_id": item["candidate_id"],
                            "proton_placement_id": item["proton_placement_id"],
                            "final_denticity": item["final_denticity"],
                            "adsorption_energy_eV": item["adsorption_energy_eV"],
                            "energy_above_site_minimum_eV": (
                                item["adsorption_energy_eV"]
                                - selected["adsorption_energy_eV"]
                            ),
                        }
                        for rank, item in enumerate(valid, 1)
                    ],
                }
            )
        selected_site_count = sum(item["status"] == "passed" for item in selections)
        rejected_site_count = len(selections) - selected_site_count
        if rejected_site_count == 0:
            selection_status = "passed"
        elif selected_site_count:
            selection_status = "passed_with_rejected_site_combinations"
        else:
            selection_status = "failed_no_stable_site_combination"
        selection_report = {
            "status": selection_status,
            "site_selections": selections,
        }
        write_manifest(selection_path, selection_report)
        manifest.update(
            {
                "status": (
                    selection_status
                    if selected_site_count
                    else "failed_validation"
                ),
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "runtime": runtime,
                "summary": {
                    "reference_task_count": len(reference_results),
                    "candidate_task_count": len(candidate_reports),
                    "valid_candidate_count": sum(
                        item["valid_for_adsorption_energy_selection"]
                        for item in candidate_reports
                    ),
                    "site_combination_count": len(selections),
                    "selected_site_count": selected_site_count,
                    "rejected_site_combination_count": rejected_site_count,
                },
                "artifacts": [
                    {"role": "reference_energy_report", "path": references_path.name, "sha256": _sha256(references_path)},
                    {"role": "candidate_energy_report", "path": candidates_path.name, "sha256": _sha256(candidates_path)},
                    {"role": "site_energy_selection_report", "path": selection_path.name, "sha256": _sha256(selection_path)},
                ],
            }
        )
        write_manifest(manifest_path, manifest)
        return manifest
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        write_manifest(manifest_path, manifest)
        raise


def _resolve_relocated_run_directory(
    recorded_path: str,
    expected_manifest_sha256: str,
    relative_search_root: Path,
) -> Path:
    """Resolve an immutable run after the same project was mounted elsewhere."""

    candidates = [Path(recorded_path).expanduser()]
    if relative_search_root.is_dir():
        candidates.extend(path.parent for path in relative_search_root.glob("*/run-manifest.json"))
    matches = []
    for candidate in candidates:
        manifest_path = candidate.resolve() / "run-manifest.json"
        if (
            manifest_path.is_file()
            and _sha256(manifest_path) == expected_manifest_sha256
            and manifest_path.parent not in matches
        ):
            matches.append(manifest_path.parent)
    if len(matches) != 1:
        raise ValueError(
            "Cannot resolve one hash-identical immutable upstream run after relocation"
        )
    return matches[0]


def _load_adsorption_probe_relaxation_evidence(run_directory: Path) -> dict:
    """Hash-check the aggregate records of one passed probe-relaxation run."""

    run_directory = Path(run_directory).expanduser().resolve()
    manifest_path = run_directory / "run-manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing adsorption-probe relaxation manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    accepted_statuses = {"passed", "passed_with_rejected_site_combinations"}
    if (
        manifest.get("mode") != "adsorption_probe_relaxation_run"
        or manifest.get("status") not in accepted_statuses
    ):
        raise ValueError("Adsorption-probe relaxation evidence is not a passed package")
    plan_record = manifest.get("plan") or {}
    plan_path = _verified_run_artifact(run_directory, plan_record)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (
        plan.get("mode") != "adsorption_probe_relaxation_plan"
        or plan.get("fingerprint") != manifest.get("plan_fingerprint")
    ):
        raise ValueError("Adsorption-probe relaxation plan is inconsistent")
    artifacts = {}
    for record in manifest.get("artifacts", []):
        role = record.get("role")
        if not isinstance(role, str) or role in artifacts:
            raise ValueError("Invalid or duplicate adsorption-relaxation artifact")
        artifacts[role] = _verified_run_artifact(run_directory, record, role=role)
    required = {
        "reference_energy_report",
        "candidate_energy_report",
        "site_energy_selection_report",
    }
    if not required.issubset(artifacts):
        raise ValueError("Adsorption-relaxation aggregate evidence is incomplete")
    candidates = json.loads(
        artifacts["candidate_energy_report"].read_text(encoding="utf-8")
    ).get("candidates")
    selections_report = json.loads(
        artifacts["site_energy_selection_report"].read_text(encoding="utf-8")
    )
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Adsorption-relaxation candidate report is empty")
    if selections_report.get("status") not in accepted_statuses:
        raise ValueError("Adsorption-relaxation site selection did not pass")
    selections = selections_report.get("site_selections")
    if not isinstance(selections, list) or not selections:
        raise ValueError("Adsorption-relaxation site selection is empty")
    candidate_by_id = {}
    for candidate in candidates:
        candidate_id = candidate.get("candidate_id")
        if not isinstance(candidate_id, str) or candidate_id in candidate_by_id:
            raise ValueError("Duplicate or invalid relaxed candidate identity")
        candidate_by_id[candidate_id] = candidate
    candidate_plan_by_id = {
        candidate["candidate_id"]: candidate
        for candidate in plan.get("candidate_relaxations", [])
    }
    if set(candidate_plan_by_id) != set(candidate_by_id):
        raise ValueError("Relaxed candidate set does not match the resolved plan")
    selected = []
    rejected = []
    for selection in selections:
        if selection.get("status") == "failed_no_valid_candidate":
            if (
                selection.get("selected_candidate_id") is not None
                or selection.get("adsorption_energy_eV") is not None
                or selection.get("valid_candidate_count") != 0
                or selection.get("valid_candidate_energy_ranking") != []
            ):
                raise ValueError("Rejected Site Combination record is inconsistent")
            site_candidates = [
                candidate
                for candidate in candidates
                if candidate.get("site_combination_id")
                == selection.get("site_combination_id")
            ]
            if (
                len(site_candidates) != selection.get("candidate_count")
                or not site_candidates
                or any(
                    candidate.get("valid_for_adsorption_energy_selection") is True
                    or candidate.get("adsorption_energy_eV") is not None
                    for candidate in site_candidates
                )
            ):
                raise ValueError("Rejected Site Combination candidates are inconsistent")
            rejected.append(
                {
                    **selection,
                    "candidate_failures": [
                        {
                            "candidate_id": candidate["candidate_id"],
                            "status": candidate.get("status"),
                            "initial_denticity": candidate_plan_by_id[
                                candidate["candidate_id"]
                            ]["initial_denticity"],
                            "final_denticity": candidate.get("final_denticity"),
                            "optimizer_steps": candidate.get("optimizer_steps"),
                            "failed_checks": sorted(
                                key
                                for key, passed in candidate.get("checks", {}).items()
                                if not passed
                            ),
                        }
                        for candidate in site_candidates
                    ],
                }
            )
            continue
        candidate_id = selection.get("selected_candidate_id")
        candidate = candidate_by_id.get(candidate_id)
        if (
            selection.get("status") != "passed"
            or candidate is None
            or candidate.get("valid_for_adsorption_energy_selection") is not True
            or candidate.get("adsorption_energy_eV") is None
            or candidate.get("site_combination_id")
            != selection.get("site_combination_id")
            or not math.isclose(
                float(candidate["adsorption_energy_eV"]),
                float(selection["adsorption_energy_eV"]),
                abs_tol=1.0e-10,
                rel_tol=0.0,
            )
        ):
            raise ValueError("Selected Site Combination candidate is inconsistent")
        relaxed_records = [
            item for item in candidate.get("artifacts", [])
            if item.get("role") == "relaxed_structure"
        ]
        if len(relaxed_records) != 1:
            raise ValueError("Selected candidate must have one relaxed structure")
        relaxed_path = _verified_run_artifact(
            run_directory, relaxed_records[0], role="relaxed_structure"
        )
        selected.append(
            {
                **candidate,
                "relaxed_structure_path": str(relaxed_path),
                "relaxed_structure_sha256": _sha256(relaxed_path),
            }
        )
    if len(selected) != int((manifest.get("summary") or {}).get("selected_site_count", -1)):
        raise ValueError("Selected relaxed-candidate count is inconsistent")
    if len(rejected) != int(
        (manifest.get("summary") or {}).get("rejected_site_combination_count", -1)
    ):
        raise ValueError("Rejected Site Combination count is inconsistent")
    return {
        "run_directory": str(run_directory),
        "run_manifest_path": str(manifest_path),
        "run_manifest_sha256": _sha256(manifest_path),
        "run_fingerprint": manifest["fingerprint"],
        "plan_path": str(plan_path),
        "plan_sha256": _sha256(plan_path),
        "plan": plan,
        "selected_candidates": selected,
        "rejected_site_combinations": rejected,
        "selection_report_path": str(artifacts["site_energy_selection_report"]),
        "selection_report_sha256": _sha256(artifacts["site_energy_selection_report"]),
        "candidate_report_path": str(artifacts["candidate_energy_report"]),
        "candidate_report_sha256": _sha256(artifacts["candidate_energy_report"]),
    }


def _final_contact_topology_key(
    donor_vertices: dict[str, list[tuple[int, int, int]]],
    operations: list[dict],
    *,
    equivalent_donors: bool,
) -> tuple:
    """Canonicalize a donor-to-periodic-metal graph under surface symmetry."""

    equivalents = []
    for operation in operations:
        transformed = {}
        for donor, vertices in donor_vertices.items():
            mapped_vertices = []
            for atom_id, image_i, image_j in vertices:
                mapped = operation["atom_map"][atom_id]
                image = operation["rotation"] @ np.asarray([image_i, image_j], dtype=int)
                image += np.asarray(mapped["image_offset"], dtype=int)
                mapped_vertices.append(
                    (mapped["target_atom_id"], int(image[0]), int(image[1]))
                )
            transformed[donor] = mapped_vertices
        origins = [vertex for vertices in transformed.values() for vertex in vertices]
        for _, origin_i, origin_j in origins:
            donor_groups = []
            for donor, vertices in sorted(transformed.items()):
                normalized = tuple(
                    sorted(
                        (atom_id, image_i - origin_i, image_j - origin_j)
                        for atom_id, image_i, image_j in vertices
                    )
                )
                donor_groups.append(normalized if equivalent_donors else (donor, normalized))
            if equivalent_donors:
                donor_groups.sort()
            equivalents.append(tuple(donor_groups))
    if not equivalents:
        raise ValueError("No final contact topology could be canonicalized")
    return min(equivalents)


def _rms_sorted_difference(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return math.inf
    if not left:
        return 0.0
    delta = np.asarray(sorted(left), dtype=float) - np.asarray(sorted(right), dtype=float)
    return float(np.sqrt(np.mean(delta * delta)))


def _site_prototype_geometry_compatible(
    member: dict, representative: dict, contract: dict
) -> tuple[bool, dict]:
    geometry = member["geometry_descriptor"]
    reference = representative["geometry_descriptor"]
    differences = {
        "contact_distance_rms_difference_A": _rms_sorted_difference(
            geometry["contact_distances_A"], reference["contact_distances_A"]
        ),
        "metal_geometry_rms_difference_A": _rms_sorted_difference(
            geometry["metal_pair_distances_A"], reference["metal_pair_distances_A"]
        ),
        "anchor_height_difference_A": abs(
            geometry["anchor_height_A"] - reference["anchor_height_A"]
        ),
        "anchor_lateral_offset_difference_A": abs(
            geometry["anchor_lateral_offset_A"]
            - reference["anchor_lateral_offset_A"]
        ),
    }
    checks = {
        "contact_distance": differences["contact_distance_rms_difference_A"]
        <= float(contract["maximum_contact_distance_rms_difference_A"]),
        "metal_geometry": differences["metal_geometry_rms_difference_A"]
        <= float(contract["maximum_metal_geometry_rms_difference_A"]),
        "anchor_height": differences["anchor_height_difference_A"]
        <= float(contract["maximum_anchor_height_difference_A"]),
        "anchor_lateral_offset": differences[
            "anchor_lateral_offset_difference_A"
        ]
        <= float(contract["maximum_anchor_lateral_offset_difference_A"]),
    }
    return all(checks.values()), {**differences, "checks": checks}


def _selected_site_prototype_record(
    selected: dict,
    candidate_plan: dict,
    source_site: dict,
    source_surface,
    test_repeat: list[int],
    surface_frame: dict,
    discovery: dict,
    operations: list[dict],
) -> dict:
    atoms = read(selected["relaxed_structure_path"])
    substrate_count = int(candidate_plan["substrate_atom_count"])
    if substrate_count % int(np.prod(test_repeat)):
        raise ValueError("Test-supercell substrate count is inconsistent with its repeat")
    source_count = substrate_count // int(np.prod(test_repeat))
    if source_count != len(source_surface):
        raise ValueError("Relaxed test slab does not map to the source surface cell")
    probe_ids = list(candidate_plan["probe_atom_ids_0_based"])
    probe = discovery["resolved_probe"]
    donor_local_ids = list(probe["adsorbate_atom_ids"]["oxygen_donors"])
    donor_labels = list(discovery["probe_donor_labels"])
    if len(donor_local_ids) != len(donor_labels):
        raise ValueError("Probe donor IDs and catalog donor labels are inconsistent")
    donor_global_to_label = {
        probe_ids[local_id]: donor_labels[index]
        for index, local_id in enumerate(donor_local_ids)
    }
    anchor_id = probe_ids[int(probe["adsorbate_atom_ids"]["anchor"])]
    cell = np.asarray(atoms.cell, dtype=float)
    source_cell = np.asarray(source_surface.cell, dtype=float)
    plane_axes = list(surface_frame["periodic_fractional_axes"])
    normal = np.asarray(surface_frame["outward_normal_cartesian_unit"], dtype=float)
    u = np.asarray(surface_frame["u_cartesian_unit"], dtype=float)
    v = np.asarray(surface_frame["v_cartesian_unit"], dtype=float)
    inverse_cell = np.linalg.inv(cell)
    contacts = []
    relaxed_vertex_positions = {}
    for contact in selected["final_donor_metal_contacts"]:
        donor_id = int(contact["probe_donor_atom_id_0_based"])
        metal_id = int(contact["surface_metal_atom_id_0_based"])
        if donor_id not in donor_global_to_label or not 0 <= metal_id < substrate_count:
            raise ValueError("Final donor-metal contact atom identity is invalid")
        block = metal_id // source_count
        source_atom_id = metal_id % source_count
        base_image = np.unravel_index(block, tuple(test_repeat))
        fractional_delta = (
            np.asarray(atoms.positions[donor_id], dtype=float)
            - np.asarray(atoms.positions[metal_id], dtype=float)
        ) @ inverse_cell
        test_cell_image = np.rint(fractional_delta).astype(int)
        source_image = [
            int(base_image[axis] + test_cell_image[axis] * test_repeat[axis])
            for axis in range(3)
        ]
        if source_image[int(discovery["normal_axis"])] != 0:
            raise ValueError("Final contact unexpectedly crosses the nonperiodic slab axis")
        vertex = (
            source_atom_id,
            source_image[plane_axes[0]],
            source_image[plane_axes[1]],
        )
        unwrapped_metal = np.asarray(atoms.positions[metal_id], dtype=float).copy()
        unwrapped_metal += sum(
            test_cell_image[axis] * cell[axis] for axis in range(3)
        )
        relaxed_vertex_positions.setdefault(vertex, unwrapped_metal)
        contacts.append(
            {
                "probe_donor_id": donor_global_to_label[donor_id],
                "probe_donor_atom_id_0_based": donor_id,
                "metal_vertex": vertex,
                "surface_metal_atom_id_0_based": metal_id,
                "surface_metal_element": contact["surface_metal_element"],
                "distance_A": float(contact["distance_A"]),
            }
        )
    contacted_donors = sorted({item["probe_donor_id"] for item in contacts})
    if len(contacted_donors) != int(selected["final_denticity"]):
        raise ValueError("Final denticity does not match the final donor-contact graph")
    donor_vertices = {
        donor: [item["metal_vertex"] for item in contacts if item["probe_donor_id"] == donor]
        for donor in contacted_donors
    }
    topology_key = _final_contact_topology_key(
        donor_vertices,
        operations,
        equivalent_donors=bool(discovery["site_prototype_clustering"]["equivalent_probe_donors"]),
    )
    vertices = sorted(relaxed_vertex_positions)
    clean_positions = []
    metal_vertices = []
    for source_atom_id, image_i, image_j in vertices:
        clean_position = np.asarray(source_surface.positions[source_atom_id], dtype=float)
        clean_position = (
            clean_position
            + image_i * source_cell[plane_axes[0]]
            + image_j * source_cell[plane_axes[1]]
        )
        clean_positions.append(clean_position)
        metal_vertices.append(
            {
                "source_atom_id": source_atom_id,
                "atom_index_base": 0,
                "lattice_image": [image_i, image_j],
                "element": source_surface[source_atom_id].symbol,
                "source_fractional_xyz": np.asarray(
                    source_surface.get_scaled_positions(wrap=True)[source_atom_id],
                    dtype=float,
                ).tolist(),
                "unwrapped_clean_cartesian_A": clean_position.tolist(),
            }
        )
    clean_centroid = np.mean(clean_positions, axis=0)
    clean_fractional = clean_centroid @ np.linalg.inv(source_cell)
    wrapped_fractional = clean_fractional.copy()
    wrapped_fractional[plane_axes] %= 1.0
    wrapped_clean = wrapped_fractional @ source_cell
    relaxed_positions = np.asarray([relaxed_vertex_positions[item] for item in vertices])
    relaxed_centroid = np.mean(relaxed_positions, axis=0)
    anchor_position = np.asarray(atoms.positions[anchor_id], dtype=float)
    anchor_delta = anchor_position - relaxed_centroid
    anchor_height = float(np.dot(anchor_delta, normal))
    lateral_vector = anchor_delta - anchor_height * normal
    anchor_projection = anchor_position - anchor_height * normal
    projection_fractional = anchor_projection @ np.linalg.inv(source_cell)
    projection_uv = [float(projection_fractional[axis] % 1.0) for axis in plane_axes]
    binding_oxygen_positions = []
    for donor in contacted_donors:
        donor_id = next(
            atom_id for atom_id, label in donor_global_to_label.items() if label == donor
        )
        position = np.asarray(atoms.positions[donor_id], dtype=float)
        relative = position - relaxed_centroid
        binding_oxygen_positions.append(
            {
                "probe_donor_id": donor,
                "atom_id_0_based": donor_id,
                "cartesian_A": position.tolist(),
                "relative_surface_frame_A": [
                    float(np.dot(relative, u)),
                    float(np.dot(relative, v)),
                    float(np.dot(relative, normal)),
                ],
            }
        )
    metal_pair_distances = sorted(
        float(np.linalg.norm(relaxed_positions[right] - relaxed_positions[left]))
        for left, right in combinations(range(len(relaxed_positions)), 2)
    )
    vertex_index = {vertex: index for index, vertex in enumerate(vertices)}
    donor_mapping = [
        {
            "probe_donor_id": item["probe_donor_id"],
            "metal_vertex_index": vertex_index[item["metal_vertex"]],
            "distance_A": item["distance_A"],
        }
        for item in sorted(
            contacts,
            key=lambda value: (
                value["probe_donor_id"],
                value["metal_vertex"],
                value["distance_A"],
            ),
        )
    ]
    unique_metal_count = len(vertices)
    final_topology = (
        f"{len(contacted_donors)}-donor_{unique_metal_count}-metal_"
        f"{len(contacts)}-contact"
    )
    return {
        "site_combination_id": selected["site_combination_id"],
        "candidate_id": selected["candidate_id"],
        "proton_placement_id": selected["proton_placement_id"],
        "adsorption_energy_eV": float(selected["adsorption_energy_eV"]),
        "initial_denticity": int(candidate_plan["initial_denticity"]),
        **(
            {"initial_binding_mode": candidate_plan["initial_binding_mode"]}
            if "initial_binding_mode" in candidate_plan
            else {}
        ),
        "final_denticity": int(selected["final_denticity"]),
        "final_topology": final_topology,
        "canonical_final_contact_topology": topology_key,
        "metal_vertices": metal_vertices,
        "donor_metal_mapping": donor_mapping,
        "ideal_site_location": {
            "fractional_uv": [float(wrapped_fractional[axis]) for axis in plane_axes],
            "cartesian_A": wrapped_clean.tolist(),
            "unwrapped_centroid_cartesian_A": clean_centroid.tolist(),
            "surface_normal_coordinate_A": float(np.dot(clean_centroid, normal)),
            "definition": (
                "centroid of final contacted metal vertices reconstructed in the "
                "undoped clean source surface cell"
            ),
        },
        "relaxed_adsorption_location": {
            "anchor_atom_id_0_based": anchor_id,
            "anchor_cartesian_A": anchor_position.tolist(),
            "anchor_projection_fractional_uv": projection_uv,
            "anchor_height_A": anchor_height,
            "lateral_offset_A": float(np.linalg.norm(lateral_vector)),
            "binding_oxygen_positions": binding_oxygen_positions,
            "relaxed_contact_metal_centroid_A": relaxed_centroid.tolist(),
        },
        "geometry_descriptor": {
            "contact_distances_A": sorted(item["distance_A"] for item in contacts),
            "metal_pair_distances_A": metal_pair_distances,
            "anchor_height_A": anchor_height,
            "anchor_lateral_offset_A": float(np.linalg.norm(lateral_vector)),
        },
        "surface_proton_assignments": selected["surface_proton_assignments"],
        "relaxed_structure": {
            "source_path": selected["relaxed_structure_path"],
            "source_sha256": selected["relaxed_structure_sha256"],
        },
        "source_site_combination": {
            "site_combination_id": source_site["site_combination_id"],
            **(
                {"initial_binding_mode": source_site["binding_mode"]}
                if "binding_mode" in source_site
                else {}
            ),
            "initial_metal_vertices": source_site["metal_vertices"],
            "initial_ideal_site_location": source_site["ideal_site_location"],
        },
    }


def resolve_adsorption_site_prototype_plan(
    substrate_name: str,
    relaxation_run: Path,
    catalog_directory: Path | None = None,
    *,
    anchor_family: str | None = None,
) -> dict:
    """Cluster selected relaxed probe minima into reusable Site Prototypes."""

    recipe = load_substrate_recipe(
        substrate_name, catalog_directory, anchor_family=anchor_family
    )
    discovery = recipe["adsorption"].get("site_discovery")
    if not isinstance(discovery, dict):
        raise ValueError(f"Substrate {recipe['key']!r} has no site-discovery contract")
    clustering = discovery["site_prototype_clustering"]
    evidence = _load_adsorption_probe_relaxation_evidence(relaxation_run)
    relaxation_plan = evidence["plan"]
    if relaxation_plan["substrate"]["key"] != recipe["key"]:
        raise ValueError("Probe-relaxation substrate does not match the catalog")
    if relaxation_plan["probe_relaxation_contract"] != discovery["probe_relaxation"]:
        raise ValueError("Probe-relaxation contract is stale against the catalog")
    probe, _ = _load_probe_definition(Path(recipe["probe_manifest_path"]))
    if probe["anchor_family"] != clustering["anchor_family"]:
        raise ValueError("Probe anchor family does not match the clustering contract")
    planned_probe = relaxation_plan["site_validation_context"]["probe"]
    if _probe_chemistry_identity(probe) != _probe_chemistry_identity(planned_probe):
        raise ValueError("Relaxed probe chemistry is stale against the maintained probe")
    preparation_record = relaxation_plan["probe_preparation_evidence"]
    preparation_root = _resolve_relocated_run_directory(
        preparation_record["run_directory"],
        preparation_record["run_manifest_sha256"],
        _default_catalog_directory().parent
        / "runs"
        / "adsorption-probe-preparation"
        / recipe["key"].casefold(),
    )
    preparation = _load_probe_preparation_evidence(preparation_root)
    preparation_plan = preparation["preparation_plan"]
    if preparation["run_manifest_sha256"] != preparation_record["run_manifest_sha256"]:
        raise ValueError("Resolved probe-preparation evidence has changed")
    source_surface = read(preparation["source_surface_path"])
    if preparation["source_surface_sha256"] != recipe["source"]["sha256"]:
        raise ValueError("Prototype source surface is stale against the catalog")
    source_sites = {
        item["site_combination_id"]: item
        for item in preparation_plan["site_enumeration_evidence"]["site_combinations"]
    }
    surface_frame = _site_discovery_surface_frame(
        source_surface,
        int(recipe["surface"]["normal_axis"]),
        discovery["surface_side"],
    )
    normal = np.asarray(surface_frame["outward_normal_cartesian_unit"], dtype=float)
    symbols = np.asarray(source_surface.get_chemical_symbols())
    eligible = np.flatnonzero(
        np.isin(symbols, list(discovery["surface_metal_elements"]))
    )
    normal_coordinates = np.asarray(source_surface.positions, dtype=float) @ normal
    exposed_coordinate = float(np.max(normal_coordinates[eligible]))
    selected_source_ids = sorted(
        int(atom_id)
        for atom_id in eligible
        if exposed_coordinate - normal_coordinates[atom_id]
        <= float(discovery["surface_metal_depth_A"]) + 1.0e-10
    )
    operations = _site_discovery_symmetry_operations(
        source_surface,
        selected_source_ids,
        int(recipe["surface"]["normal_axis"]),
        float(discovery["symmetry_tolerance_A"]),
    )
    candidate_plans = {
        item["candidate_id"]: item for item in relaxation_plan["candidate_relaxations"]
    }
    resolved_discovery = {
        **discovery,
        "resolved_probe": probe,
        "normal_axis": int(recipe["surface"]["normal_axis"]),
    }
    records = []
    for selected in evidence["selected_candidates"]:
        candidate_plan = candidate_plans.get(selected["candidate_id"])
        source_site = source_sites.get(selected["site_combination_id"])
        if candidate_plan is None or source_site is None:
            raise ValueError("Selected relaxed candidate lost its preparation provenance")
        records.append(
            _selected_site_prototype_record(
                selected,
                candidate_plan,
                source_site,
                source_surface,
                list(preparation_plan["test_supercell"]["repeat"]),
                relaxation_plan["surface_frame"],
                resolved_discovery,
                operations,
            )
        )
    records.sort(key=lambda item: (item["adsorption_energy_eV"], item["candidate_id"]))
    clusters = []
    for record in records:
        assigned = None
        for cluster in clusters:
            if (
                record["canonical_final_contact_topology"]
                != cluster["representative"]["canonical_final_contact_topology"]
            ):
                continue
            compatible, differences = _site_prototype_geometry_compatible(
                record, cluster["representative"], clustering
            )
            if compatible:
                assigned = (cluster, differences)
                break
        if assigned is None:
            clusters.append(
                {
                    "representative": record,
                    "members": [(record, {
                        "contact_distance_rms_difference_A": 0.0,
                        "metal_geometry_rms_difference_A": 0.0,
                        "anchor_height_difference_A": 0.0,
                        "anchor_lateral_offset_difference_A": 0.0,
                        "checks": {
                            "contact_distance": True,
                            "metal_geometry": True,
                            "anchor_height": True,
                            "anchor_lateral_offset": True,
                        },
                    })],
                }
            )
        else:
            assigned[0]["members"].append((record, assigned[1]))
    prototypes = []
    for number, cluster in enumerate(clusters, 1):
        representative = cluster["representative"]
        prototype_id = (
            f"site-prototype-{clustering['anchor_family'].replace('_', '-')}-{number:04d}"
        )
        members = []
        for member, differences in cluster["members"]:
            members.append(
                {
                    "site_combination_id": member["site_combination_id"],
                    "candidate_id": member["candidate_id"],
                    "proton_placement_id": member["proton_placement_id"],
                    "adsorption_energy_eV": member["adsorption_energy_eV"],
                    **(
                        {"initial_binding_mode": member["initial_binding_mode"]}
                        if "initial_binding_mode" in member
                        else {}
                    ),
                    "energy_above_prototype_minimum_eV": (
                        member["adsorption_energy_eV"]
                        - representative["adsorption_energy_eV"]
                    ),
                    "geometry_difference_from_representative": differences,
                }
            )
        prototypes.append(
            {
                "site_prototype_id": prototype_id,
                "anchor_family": clustering["anchor_family"],
                "probe": probe["key"],
                "representative_candidate_id": representative["candidate_id"],
                "member_count": len(members),
                "members": members,
                "metal_vertices": representative["metal_vertices"],
                "donor_metal_mapping": representative["donor_metal_mapping"],
                "ideal_site_location": representative["ideal_site_location"],
                "relaxed_adsorption_location": representative[
                    "relaxed_adsorption_location"
                ],
                "surface_frame": relaxation_plan["surface_frame"],
                "final_topology": representative["final_topology"],
                "final_denticity": representative["final_denticity"],
                "geometry_descriptor": representative["geometry_descriptor"],
                "adsorption_energy_eV": representative["adsorption_energy_eV"],
                "energy_policy": clustering["energy_policy"],
                "surface_proton_policy": {
                    "released_protons_retained_in_energy": probe["released_protons"],
                    "selected_assignments": representative[
                        "surface_proton_assignments"
                    ],
                    "excluded_from_future_site_integer_program": True,
                },
                "doped_site_energy_policy": clustering["doped_site_energy_policy"],
                "doped_site_energy_source": "undoped_parent_approximation",
                "representative_relaxed_structure": representative["relaxed_structure"],
                "source_site_combination": representative["source_site_combination"],
            }
        )
    prototype_identity = []
    for prototype in prototypes:
        prototype_identity.append(
            {
                **{
                    key: value
                    for key, value in prototype.items()
                    if key != "representative_relaxed_structure"
                },
                "representative_relaxed_structure": {
                    "sha256": prototype["representative_relaxed_structure"][
                        "source_sha256"
                    ]
                },
            }
        )
    identity = {
        "substrate_key": recipe["key"],
        "catalog_sha256": _sha256(Path(recipe["catalog_path"])),
        "relaxation_run_manifest_sha256": evidence["run_manifest_sha256"],
        "selection_report_sha256": evidence["selection_report_sha256"],
        "candidate_report_sha256": evidence["candidate_report_sha256"],
        "preparation_run_manifest_sha256": preparation["run_manifest_sha256"],
        "source_surface_sha256": preparation["source_surface_sha256"],
        "probe_identity": _probe_chemistry_identity(probe),
        "clustering_contract": clustering,
        "prototypes": prototype_identity,
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema_version": 1,
        "mode": "adsorption_site_prototype_plan",
        "status": "passed_clustering_plan",
        "fingerprint": fingerprint,
        "substrate": {
            "key": recipe["key"],
            "catalog_path": str(recipe["catalog_path"]),
            "catalog_sha256": _sha256(Path(recipe["catalog_path"])),
        },
        "anchor_family": clustering["anchor_family"],
        "probe": probe,
        "source_surface": {
            "path": preparation["source_surface_path"],
            "sha256": preparation["source_surface_sha256"],
            "formula": _formula(source_surface.get_chemical_symbols()),
            "atom_count": len(source_surface),
        },
        "relaxation_evidence": {
            key: value for key, value in evidence.items()
            if key not in {"plan", "selected_candidates"}
        },
        "preparation_evidence": {
            "run_directory": preparation["run_directory"],
            "run_manifest_sha256": preparation["run_manifest_sha256"],
            "fingerprint": preparation["fingerprint"],
        },
        "clustering_contract": clustering,
        "summary": {
            "selected_relaxed_site_count": len(records),
            "rejected_geometric_site_combination_count": len(
                evidence["rejected_site_combinations"]
            ),
            "site_prototype_count": len(prototypes),
            "bidentate_prototype_count": sum(
                item["final_denticity"] == 2 for item in prototypes
            ),
            "tridentate_prototype_count": sum(
                item["final_denticity"] == 3 for item in prototypes
            ),
            "collapsed_selected_site_count": len(records) - len(prototypes),
        },
        "rejected_site_combinations": evidence["rejected_site_combinations"],
        "site_prototypes": prototypes,
        "execution_contract": {
            "read_only": True,
            "executable": True,
            "starts_optimizer_or_gpu": False,
            "execution_command": "adsorption-site-prototype-run",
            "output_status": "passed_clustering_pending_promotion",
            "known_limitations": [
                "This library is valid only for the recorded substrate surface and anchor family.",
                *(
                    [
                        "Carboxylic-acid prototypes require an independent acetic-acid calibration."
                    ]
                    if clustering["anchor_family"] != "carboxylic_acid"
                    else []
                ),
                "Doped Site Instances inherit undoped parent energies without a local dopant correction.",
            ],
        },
    }


def execute_adsorption_site_prototype_clustering(plan: dict, output_root: Path) -> dict:
    """Write one immutable Site Prototype candidate library."""

    if plan.get("mode") != "adsorption_site_prototype_plan":
        raise ValueError("Site Prototype clustering requires a resolved plan")
    if not plan["execution_contract"]["executable"]:
        raise ValueError("Site Prototype plan has unresolved readiness issues")
    for prototype in plan["site_prototypes"]:
        source = Path(prototype["representative_relaxed_structure"]["source_path"])
        if (
            not source.is_file()
            or _sha256(source)
            != prototype["representative_relaxed_structure"]["source_sha256"]
        ):
            raise ValueError(f"Representative relaxed structure changed: {source}")
    output_root = Path(output_root).expanduser().resolve()
    run_directory = (
        output_root
        / plan["substrate"]["key"].casefold()
        / (
            f"{plan['clustering_contract']['protocol_version']}-"
            f"{plan['fingerprint'][:12]}"
        )
    )
    if run_directory.exists():
        raise ValueError(f"Immutable Site Prototype run already exists: {run_directory}")
    run_directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        dir=run_directory.parent, prefix=f".{run_directory.name}.tmp-"
    ) as temporary_name:
        temporary = Path(temporary_name)
        structures = temporary / "structures"
        structures.mkdir()
        persisted_prototypes = []
        for prototype in plan["site_prototypes"]:
            source = Path(prototype["representative_relaxed_structure"]["source_path"])
            destination = structures / f"{prototype['site_prototype_id']}.extxyz"
            shutil.copyfile(source, destination)
            persisted_prototypes.append(
                {
                    **prototype,
                    "representative_relaxed_structure": {
                        "path": destination.relative_to(temporary).as_posix(),
                        "sha256": _sha256(destination),
                    },
                }
            )
        plan_path = temporary / "adsorption-site-prototype-plan.json"
        library_path = temporary / "site-prototypes.json"
        manifest_path = temporary / "run-manifest.json"
        write_manifest(plan_path, plan)
        library = {
            "schema_version": 1,
            "status": "passed_clustering_pending_promotion",
            "fingerprint": plan["fingerprint"],
            "substrate_key": plan["substrate"]["key"],
            "anchor_family": plan["anchor_family"],
            "probe": _probe_chemistry_identity(plan["probe"]),
            "source_surface": {
                key: value
                for key, value in plan["source_surface"].items()
                if key != "path"
            },
            "clustering_contract": plan["clustering_contract"],
            "summary": plan["summary"],
            "rejected_site_combinations": plan["rejected_site_combinations"],
            "site_prototypes": persisted_prototypes,
        }
        write_manifest(library_path, library)
        manifest = {
            "schema_version": 1,
            "mode": "adsorption_site_prototype_run",
            "status": "passed_clustering_pending_promotion",
            "fingerprint": plan["fingerprint"],
            "substrate_key": plan["substrate"]["key"],
            "anchor_family": plan["anchor_family"],
            "site_prototype_count": len(persisted_prototypes),
            "artifacts": [
                {
                    "role": "site_prototype_plan",
                    "path": plan_path.name,
                    "sha256": _sha256(plan_path),
                },
                {
                    "role": "site_prototype_library",
                    "path": library_path.name,
                    "sha256": _sha256(library_path),
                },
                *[
                    {
                        "role": "representative_relaxed_structure",
                        "site_prototype_id": prototype["site_prototype_id"],
                        "path": prototype["representative_relaxed_structure"]["path"],
                        "sha256": prototype["representative_relaxed_structure"]["sha256"],
                    }
                    for prototype in persisted_prototypes
                ],
            ],
        }
        write_manifest(manifest_path, manifest)
        temporary.replace(run_directory)
    return {**manifest, "output_directory": str(run_directory)}


def _load_site_prototype_run_evidence(run_directory: Path) -> dict:
    run_directory = Path(run_directory).expanduser().resolve()
    manifest_path = run_directory / "run-manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing Site Prototype run manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("mode") != "adsorption_site_prototype_run"
        or manifest.get("status") != "passed_clustering_pending_promotion"
    ):
        raise ValueError("Site Prototype evidence is not a promotable passed package")
    artifacts = {}
    structures = {}
    for record in manifest.get("artifacts", []):
        role = record.get("role")
        path = _verified_run_artifact(run_directory, record, role=role)
        if role == "representative_relaxed_structure":
            prototype_id = record.get("site_prototype_id")
            if not isinstance(prototype_id, str) or prototype_id in structures:
                raise ValueError("Invalid or duplicate prototype structure identity")
            structures[prototype_id] = {
                "path": path,
                "sha256": _sha256(path),
            }
        elif isinstance(role, str) and role not in artifacts:
            artifacts[role] = path
        else:
            raise ValueError("Invalid or duplicate Site Prototype artifact")
    if not {"site_prototype_plan", "site_prototype_library"}.issubset(artifacts):
        raise ValueError("Site Prototype evidence chain is incomplete")
    plan = json.loads(artifacts["site_prototype_plan"].read_text(encoding="utf-8"))
    library = json.loads(
        artifacts["site_prototype_library"].read_text(encoding="utf-8")
    )
    if (
        plan.get("mode") != "adsorption_site_prototype_plan"
        or plan.get("fingerprint") != manifest.get("fingerprint")
        or library.get("fingerprint") != manifest.get("fingerprint")
        or library.get("status") != "passed_clustering_pending_promotion"
        or library.get("anchor_family") != manifest.get("anchor_family")
    ):
        raise ValueError("Site Prototype plan, library, and manifest are inconsistent")
    prototypes = library.get("site_prototypes")
    if (
        not isinstance(prototypes, list)
        or len(prototypes) != int(manifest.get("site_prototype_count", -1))
    ):
        raise ValueError("Site Prototype count is inconsistent")
    prototype_ids = {item.get("site_prototype_id") for item in prototypes}
    if len(prototype_ids) != len(prototypes) or prototype_ids != set(structures):
        raise ValueError("Site Prototype structures do not cover the library exactly")
    for prototype in prototypes:
        record = prototype.get("representative_relaxed_structure") or {}
        expected = structures[prototype["site_prototype_id"]]
        if record.get("sha256") != expected["sha256"]:
            raise ValueError("Prototype structure hash is inconsistent")
    return {
        "run_directory": run_directory,
        "run_manifest_path": manifest_path,
        "run_manifest_sha256": _sha256(manifest_path),
        "manifest": manifest,
        "plan_path": artifacts["site_prototype_plan"],
        "plan_sha256": _sha256(artifacts["site_prototype_plan"]),
        "plan": plan,
        "library_path": artifacts["site_prototype_library"],
        "library_sha256": _sha256(artifacts["site_prototype_library"]),
        "library": library,
        "structures": structures,
    }


def resolve_adsorption_site_prototype_promotion_plan(
    substrate_name: str,
    prototype_run: Path,
    structure_directory: Path | None = None,
    *,
    anchor_family: str | None = None,
    supersede_existing: bool = False,
) -> dict:
    """Validate one Site Prototype package for maintained-library promotion."""

    recipe = load_substrate_recipe(substrate_name, anchor_family=anchor_family)
    structure_manifest = _load_structure_manifest(substrate_name, structure_directory)
    manifest_path = Path(structure_manifest["manifest_path"])
    evidence = _load_site_prototype_run_evidence(prototype_run)
    plan = evidence["plan"]
    library = evidence["library"]
    discovery = recipe["adsorption"]["site_discovery"]
    clustering = discovery["site_prototype_clustering"]
    if (
        plan["substrate"]["key"] != recipe["key"]
        or library["substrate_key"] != recipe["key"]
    ):
        raise ValueError("Site Prototype substrate does not match the requested catalog")
    if plan["substrate"]["catalog_sha256"] != _sha256(Path(recipe["catalog_path"])):
        raise ValueError("Site Prototype plan is stale against the substrate catalog")
    if library["source_surface"]["sha256"] != recipe["source"]["sha256"]:
        raise ValueError("Site Prototype source surface is stale")
    if library["clustering_contract"] != clustering:
        raise ValueError("Site Prototype clustering contract is stale")
    probe, _ = _load_probe_definition(Path(recipe["probe_manifest_path"]))
    if library["probe"] != _probe_chemistry_identity(probe):
        raise ValueError("Site Prototype probe identity is stale")
    anchor_family = library["anchor_family"]
    destination_directory = (
        manifest_path.parent
        / "adsorption-sites"
        / anchor_family
        / (
            f"{clustering['protocol_version']}-{evidence['manifest']['fingerprint'][:12]}"
        )
    ).resolve()
    try:
        destination_directory.relative_to(manifest_path.parent.resolve())
    except ValueError as exc:
        raise ValueError("Site Prototype destination escapes the structure library") from exc
    accepted_for_family = [
        item
        for item in structure_manifest.get("adsorption_site_libraries", [])
        if item.get("status") == "accepted_site_prototype_library"
        and item.get("anchor_family") == anchor_family
        and item.get("source_surface_sha256") == recipe["source"]["sha256"]
    ]
    readiness_issues = []
    if destination_directory.exists():
        readiness_issues.append(
            {
                "code": "promotion_destination_exists",
                "message": f"Immutable promotion destination exists: {destination_directory}",
            }
        )
    if accepted_for_family and not supersede_existing:
        readiness_issues.append(
            {
                "code": "accepted_library_already_registered",
                "message": (
                    "An accepted Site Prototype library already exists for this "
                    "surface and anchor family"
                ),
            }
        )
    relaxation = plan["relaxation_evidence"]
    evidence_records = [
        {
            "role": "site_prototype_plan",
            "source_path": str(evidence["plan_path"]),
            "source_sha256": evidence["plan_sha256"],
            "destination_name": "site-prototype-plan.json",
        },
        {
            "role": "site_prototype_run_manifest",
            "source_path": str(evidence["run_manifest_path"]),
            "source_sha256": evidence["run_manifest_sha256"],
            "destination_name": "site-prototype-run-manifest.json",
        },
        {
            "role": "probe_relaxation_plan",
            "source_path": relaxation["plan_path"],
            "source_sha256": relaxation["plan_sha256"],
            "destination_name": "probe-relaxation-plan.json",
        },
        {
            "role": "probe_relaxation_run_manifest",
            "source_path": relaxation["run_manifest_path"],
            "source_sha256": relaxation["run_manifest_sha256"],
            "destination_name": "probe-relaxation-run-manifest.json",
        },
        {
            "role": "candidate_energy_report",
            "source_path": relaxation["candidate_report_path"],
            "source_sha256": relaxation["candidate_report_sha256"],
            "destination_name": "candidate-energy-report.json",
        },
        {
            "role": "site_energy_selection_report",
            "source_path": relaxation["selection_report_path"],
            "source_sha256": relaxation["selection_report_sha256"],
            "destination_name": "site-energy-selection-report.json",
        },
    ]
    for record in evidence_records:
        source = Path(record["source_path"])
        if not source.is_file() or _sha256(source) != record["source_sha256"]:
            raise ValueError(f"Site Prototype promotion evidence changed: {source}")
    identity = {
        "substrate_key": recipe["key"],
        "structure_manifest_sha256": _sha256(manifest_path),
        "prototype_run_manifest_sha256": evidence["run_manifest_sha256"],
        "prototype_library_sha256": evidence["library_sha256"],
        "destination_directory": str(destination_directory),
        "supersede_existing": supersede_existing,
        "accepted_registrations_to_supersede": accepted_for_family,
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema_version": 1,
        "mode": "adsorption_site_prototype_promotion_plan",
        "status": "passed_promotion_plan" if not readiness_issues else "blocked",
        "fingerprint": fingerprint,
        "substrate": {
            "key": recipe["key"],
            "structure_manifest_path": str(manifest_path),
            "structure_manifest_sha256": _sha256(manifest_path),
        },
        "source_surface_sha256": recipe["source"]["sha256"],
        "anchor_family": anchor_family,
        "probe": library["probe"],
        "clustering_contract": clustering,
        "prototype_run": {
            "directory": str(evidence["run_directory"]),
            "manifest_sha256": evidence["run_manifest_sha256"],
            "fingerprint": evidence["manifest"]["fingerprint"],
        },
        "prototype_library": {
            "source_path": str(evidence["library_path"]),
            "source_sha256": evidence["library_sha256"],
            "summary": library["summary"],
            "destination_directory": str(destination_directory),
            "destination_path": str(destination_directory / "site-prototypes.json"),
        },
        "supersession": {
            "requested": supersede_existing,
            "accepted_registrations_to_supersede": accepted_for_family,
        },
        "representative_structures": [
            {
                "site_prototype_id": prototype_id,
                "source_path": str(record["path"]),
                "source_sha256": record["sha256"],
                "destination_name": f"{prototype_id}.extxyz",
            }
            for prototype_id, record in sorted(evidence["structures"].items())
        ],
        "evidence": evidence_records,
        "execution_contract": {
            "read_only": True,
            "executable": not readiness_issues,
            "execution_command": "adsorption-site-prototype-promotion-run",
            "readiness_issues": readiness_issues,
            "mutation": (
                "Create one immutable maintained Site Prototype package and append "
                "its hash-verified registration to the substrate structure manifest"
                + (
                    ", while retaining and marking the prior accepted registration "
                    "as superseded."
                    if accepted_for_family and supersede_existing
                    else "."
                )
            ),
        },
    }


def execute_adsorption_site_prototype_promotion(plan: dict) -> dict:
    """Copy and atomically register one accepted Site Prototype library."""

    if plan.get("mode") != "adsorption_site_prototype_promotion_plan":
        raise ValueError("Site Prototype promotion requires a resolved promotion plan")
    if not plan["execution_contract"]["executable"]:
        raise ValueError("Site Prototype promotion plan has readiness issues")
    manifest_path = Path(plan["substrate"]["structure_manifest_path"])
    if (
        not manifest_path.is_file()
        or _sha256(manifest_path) != plan["substrate"]["structure_manifest_sha256"]
    ):
        raise ValueError("Structure manifest changed after Site Prototype planning")
    source_library = Path(plan["prototype_library"]["source_path"])
    if (
        not source_library.is_file()
        or _sha256(source_library) != plan["prototype_library"]["source_sha256"]
    ):
        raise ValueError("Site Prototype library changed after promotion planning")
    all_sources = [*plan["representative_structures"], *plan["evidence"]]
    for record in all_sources:
        source = Path(record["source_path"])
        if not source.is_file() or _sha256(source) != record["source_sha256"]:
            raise ValueError(f"Site Prototype promotion source changed: {source}")
    destination_directory = Path(
        plan["prototype_library"]["destination_directory"]
    )
    if destination_directory.exists():
        raise ValueError(
            f"Immutable Site Prototype destination exists: {destination_directory}"
        )
    destination_directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        dir=destination_directory.parent,
        prefix=f".{destination_directory.name}.tmp-",
    ) as temporary_name:
        temporary = Path(temporary_name)
        structures_directory = temporary / "structures"
        records_directory = temporary / "records"
        structures_directory.mkdir()
        records_directory.mkdir()
        shutil.copyfile(source_library, temporary / "site-prototypes.json")
        copied_structures = []
        for record in plan["representative_structures"]:
            destination = structures_directory / record["destination_name"]
            shutil.copyfile(Path(record["source_path"]), destination)
            copied_structures.append(
                {
                    "site_prototype_id": record["site_prototype_id"],
                    "path": destination.relative_to(temporary).as_posix(),
                    "sha256": _sha256(destination),
                }
            )
        copied_evidence = []
        for record in plan["evidence"]:
            destination = records_directory / record["destination_name"]
            shutil.copyfile(Path(record["source_path"]), destination)
            copied_evidence.append(
                {
                    "role": record["role"],
                    "path": destination.relative_to(temporary).as_posix(),
                    "sha256": _sha256(destination),
                }
            )
        promotion_plan_path = records_directory / "site-prototype-promotion-plan.json"
        write_manifest(promotion_plan_path, plan)
        promotion_report = {
            "schema_version": 1,
            "mode": "adsorption_site_prototype_promotion_run",
            "status": "passed_promotion",
            "fingerprint": plan["fingerprint"],
            "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
            "substrate_key": plan["substrate"]["key"],
            "anchor_family": plan["anchor_family"],
            "source_surface_sha256": plan["source_surface_sha256"],
            "prototype_run": plan["prototype_run"],
            "prototype_library_sha256": _sha256(temporary / "site-prototypes.json"),
        }
        promotion_report_path = records_directory / "site-prototype-promotion-report.json"
        write_manifest(promotion_report_path, promotion_report)
        copied_evidence.extend(
            [
                {
                    "role": "site_prototype_promotion_plan",
                    "path": promotion_plan_path.relative_to(temporary).as_posix(),
                    "sha256": _sha256(promotion_plan_path),
                },
                {
                    "role": "site_prototype_promotion_report",
                    "path": promotion_report_path.relative_to(temporary).as_posix(),
                    "sha256": _sha256(promotion_report_path),
                },
            ]
        )
        temporary.replace(destination_directory)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    relative_directory = destination_directory.relative_to(manifest_path.parent)
    superseded_registrations = []
    expected_superseded = plan.get("supersession", {}).get(
        "accepted_registrations_to_supersede", []
    )
    libraries = manifest.setdefault("adsorption_site_libraries", [])
    for expected in expected_superseded:
        matches = [
            item
            for item in libraries
            if item.get("status") == "accepted_site_prototype_library"
            and item.get("promotion_fingerprint")
            == expected.get("promotion_fingerprint")
            and item.get("path") == expected.get("path")
            and item.get("sha256") == expected.get("sha256")
        ]
        if len(matches) != 1 or matches[0] != expected:
            raise ValueError(
                "Accepted Site Prototype registration changed after supersession planning"
            )
        superseded_registrations.append(matches[0])
    registration = {
        "status": "accepted_site_prototype_library",
        "anchor_family": plan["anchor_family"],
        "probe": plan["probe"]["key"],
        "source_surface_sha256": plan["source_surface_sha256"],
        "protocol_version": plan["clustering_contract"]["protocol_version"],
        "promotion_fingerprint": plan["fingerprint"],
        "prototype_run_fingerprint": plan["prototype_run"]["fingerprint"],
        "path": (relative_directory / "site-prototypes.json").as_posix(),
        "sha256": _sha256(destination_directory / "site-prototypes.json"),
        "summary": plan["prototype_library"]["summary"],
        "doped_site_energy_policy": plan["clustering_contract"]
        ["doped_site_energy_policy"],
        "representative_structures": [
            {
                **record,
                "path": (relative_directory / Path(record["path"])).as_posix(),
            }
            for record in copied_structures
        ],
        "records": [
            {
                **record,
                "path": (relative_directory / Path(record["path"])).as_posix(),
            }
            for record in copied_evidence
        ],
    }
    superseded_at_utc = datetime.now(timezone.utc).isoformat()
    for old_registration in superseded_registrations:
        old_registration["status"] = "superseded_site_prototype_library"
        old_registration["superseded_at_utc"] = superseded_at_utc
        old_registration["superseded_by"] = registration["path"]
        old_registration["superseded_by_sha256"] = registration["sha256"]
        old_registration["superseded_by_promotion_fingerprint"] = registration[
            "promotion_fingerprint"
        ]
    libraries.append(registration)
    write_manifest(manifest_path, manifest)
    promotion_report.update(
        {
            "registered_library": {
                **registration,
                "path": str(destination_directory / "site-prototypes.json"),
            },
            "superseded_library_count": len(superseded_registrations),
            "structure_manifest": {
                "path": str(manifest_path),
                "sha256": _sha256(manifest_path),
            },
        }
    )
    return promotion_report


def execute_adsorption_probe_preparation(plan: dict, output_root: Path) -> dict:
    """Write all validated probe/surface-H starting structures immutably."""

    if plan.get("mode") != "adsorption_probe_preparation_plan":
        raise ValueError("Probe preparation requires an adsorption probe plan")
    if not plan["execution_contract"]["executable"]:
        raise ValueError("Adsorption probe preparation plan has readiness issues")
    source_path = Path(plan["source_surface"]["path"])
    probe_source = Path(plan["probe"]["source_path"])
    if _sha256(source_path) != plan["source_surface"]["sha256"]:
        raise ValueError("Probe-preparation source surface changed after planning")
    if _sha256(probe_source) != plan["probe"]["source_sha256"]:
        raise ValueError("Probe source changed after planning")
    _, adsorbate = _load_probe_definition(Path(plan["probe"]["manifest_path"]))
    source_surface = read(source_path)
    test_supercell = source_surface.repeat(tuple(plan["test_supercell"]["repeat"]))
    output_root = Path(output_root).expanduser().resolve()
    run_directory = (
        output_root
        / plan["substrate"]["key"].casefold()
        / f"{plan['probe_preparation_contract']['protocol_version']}-{plan['fingerprint'][:12]}"
    )
    if run_directory.exists():
        raise ValueError(f"Immutable probe-preparation run already exists: {run_directory}")
    run_directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        dir=run_directory.parent, prefix=f".{run_directory.name}.tmp-"
    ) as temporary_name:
        temporary = Path(temporary_name)
        structures_directory = temporary / "structures"
        structures_directory.mkdir()
        plan_path = temporary / "adsorption-probe-preparation-plan.json"
        structures_report_path = temporary / "starting-structures.json"
        source_copy = temporary / source_path.name
        clean_test_supercell_path = temporary / "clean-test-supercell.extxyz"
        probe_copy = temporary / probe_source.name
        manifest_path = temporary / "run-manifest.json"
        write_manifest(plan_path, plan)
        shutil.copyfile(source_path, source_copy)
        write(clean_test_supercell_path, test_supercell, format="extxyz")
        shutil.copyfile(probe_source, probe_copy)
        structure_records = []
        for site in plan["site_preparations"]:
            site_directory = structures_directory / site["site_combination_id"]
            site_directory.mkdir()
            probe_atoms = adsorbate.copy()
            probe_atoms.set_positions(
                np.asarray(site["probe_placement"]["probe_positions_A"], dtype=float)
            )
            for proton_placement in site["surface_proton_placements"]:
                combined = test_supercell.copy()
                combined.extend(probe_atoms)
                combined.extend(
                    Atoms(
                        ["H"] * int(plan["probe"]["released_protons"]),
                        positions=np.asarray(
                            proton_placement["hydrogen_positions_A"], dtype=float
                        ),
                    )
                )
                combined.set_cell(test_supercell.cell)
                combined.set_pbc(test_supercell.pbc)
                structure_path = site_directory / (
                    f"{site['site_combination_id']}-"
                    f"{proton_placement['proton_placement_id']}.extxyz"
                )
                write(structure_path, combined, format="extxyz")
                observed_formula = _formula(combined.get_chemical_symbols())
                if observed_formula != plan["neutral_stoichiometry"]["combined_formula"]:
                    raise ValueError("Prepared probe structure formula changed")
                structure_records.append(
                    {
                        "site_combination_id": site["site_combination_id"],
                        "denticity": site["denticity"],
                        **(
                            {"binding_mode": site["binding_mode"]}
                            if "binding_mode" in site
                            else {}
                        ),
                        "proton_placement_id": proton_placement[
                            "proton_placement_id"
                        ],
                        "path": str(structure_path.relative_to(temporary)),
                        "sha256": _sha256(structure_path),
                        "formula": observed_formula,
                        "atom_count": len(combined),
                        "substrate_atom_count": len(test_supercell),
                        "probe_atom_ids": list(
                            range(len(test_supercell), len(test_supercell) + len(adsorbate))
                        ),
                        "surface_proton_atom_ids": list(
                            range(
                                len(test_supercell) + len(adsorbate),
                                len(test_supercell)
                                + len(adsorbate)
                                + int(plan["probe"]["released_protons"]),
                            )
                        ),
                        "actual_test_supercell_metal_atom_ids": site[
                            "actual_test_supercell_metal_atom_ids"
                        ],
                        "mapped_adsorbate_donor_ids": site["probe_placement"][
                            "mapped_adsorbate_donor_ids"
                        ],
                        "surface_proton_parent_vertices": proton_placement[
                            "surface_proton_parent_vertices"
                        ],
                        "validation": {
                            **site["probe_placement"]["checks"],
                            **proton_placement["checks"],
                            "neutral_stoichiometry": True,
                        },
                    }
                )
        structures_report = {
            "schema_version": 1,
            "status": "passed_preparation_pending_relaxation",
            "fingerprint": plan["fingerprint"],
            "substrate_key": plan["substrate"]["key"],
            "probe": plan["probe"],
            "test_supercell": plan["test_supercell"],
            "neutral_stoichiometry": plan["neutral_stoichiometry"],
            "summary": plan["summary"],
            "structures": structure_records,
        }
        write_manifest(structures_report_path, structures_report)
        manifest = {
            "schema_version": 1,
            "mode": "adsorption_probe_preparation_run",
            "status": "passed_preparation_pending_relaxation",
            "fingerprint": plan["fingerprint"],
            "substrate_key": plan["substrate"]["key"],
            "site_combination_count": plan["summary"]["site_combination_count"],
            "starting_structure_count": len(structure_records),
            "artifacts": [
                {
                    "role": "probe_preparation_plan",
                    "path": plan_path.name,
                    "sha256": _sha256(plan_path),
                },
                {
                    "role": "source_surface_snapshot",
                    "path": source_copy.name,
                    "sha256": _sha256(source_copy),
                },
                {
                    "role": "clean_test_supercell_snapshot",
                    "path": clean_test_supercell_path.name,
                    "sha256": _sha256(clean_test_supercell_path),
                },
                {
                    "role": "neutral_probe_source_snapshot",
                    "path": probe_copy.name,
                    "sha256": _sha256(probe_copy),
                },
                {
                    "role": "starting_structures_report",
                    "path": structures_report_path.name,
                    "sha256": _sha256(structures_report_path),
                },
            ],
        }
        write_manifest(manifest_path, manifest)
        temporary.replace(run_directory)
    return {**manifest, "output_directory": str(run_directory)}


def _minimum_image_vector(
    vector: np.ndarray, cell: np.ndarray, pbc: np.ndarray
) -> np.ndarray:
    if not np.any(pbc):
        return vector
    fractional = np.asarray(vector, dtype=float) @ np.linalg.inv(cell)
    fractional[pbc] -= np.round(fractional[pbc])
    return fractional @ cell


def _molecular_adjacency(
    symbols: list[str],
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
) -> list[set[int]]:
    adjacency = [set() for _ in symbols]
    for left in range(len(symbols)):
        for right in range(left + 1, len(symbols)):
            cutoff = 1.25 * (
                covalent_radii[atomic_numbers[symbols[left]]]
                + covalent_radii[atomic_numbers[symbols[right]]]
            )
            vector = _minimum_image_vector(
                positions[right] - positions[left], cell, pbc
            )
            distance = float(np.linalg.norm(vector))
            if distance > 0.2 and distance_within_closed_window(
                distance,
                0.2,
                cutoff,
            ):
                adjacency[left].add(right)
                adjacency[right].add(left)
    return adjacency


def _unwrap_molecule(
    positions: np.ndarray,
    adjacency: list[set[int]],
    cell: np.ndarray,
    pbc: np.ndarray,
    start: int,
) -> np.ndarray:
    unwrapped = np.empty_like(positions, dtype=float)
    unwrapped[start] = positions[start]
    visited = {start}
    stack = [start]
    while stack:
        current = stack.pop()
        for neighbor in sorted(adjacency[current]):
            if neighbor in visited:
                continue
            vector = _minimum_image_vector(
                positions[neighbor] - positions[current], cell, pbc
            )
            unwrapped[neighbor] = unwrapped[current] + vector
            visited.add(neighbor)
            stack.append(neighbor)
    if len(visited) != len(positions):
        raise ValueError("Cannot unwrap a disconnected SAM structure")
    return unwrapped


def _connected_components(adjacency: list[set[int]]) -> list[list[int]]:
    unseen = set(range(len(adjacency)))
    components = []
    while unseen:
        start = min(unseen)
        stack = [start]
        unseen.remove(start)
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(adjacency[current], reverse=True):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


def inspect_sam_submission(
    path: Path,
    adsorption: dict,
    *,
    allow_deprotonated_h0: bool = False,
) -> dict:
    """Validate one complete isolated SAM and infer its supported anchor."""

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"SAM structure does not exist: {path}")
    atoms = read(path)
    if len(atoms) < 2:
        raise ValueError("SAM structure must contain a complete molecule")
    symbols = atoms.get_chemical_symbols()
    unknown = sorted(set(symbols) - set(atomic_numbers))
    if unknown:
        raise ValueError("SAM structure contains unknown elements: " + ", ".join(unknown))
    positions = np.asarray(atoms.positions, dtype=float)
    cell = np.asarray(atoms.cell, dtype=float)
    pbc = np.asarray(atoms.pbc, dtype=bool)
    if np.any(pbc) and abs(float(np.linalg.det(cell))) < 1.0e-8:
        raise ValueError("Periodic SAM input must have a non-singular cell")
    adjacency = _molecular_adjacency(symbols, positions, cell, pbc)
    components = _connected_components(adjacency)
    if len(components) != 1:
        sizes = [len(component) for component in components]
        raise ValueError(
            "SAM input must contain exactly one covalently connected molecule; "
            f"found {len(components)} components with sizes {sizes}"
        )

    anchor_element = adsorption["anchor_element"]
    expected_pattern = sorted(adsorption["anchor_neighbor_pattern"])
    candidates = []
    for index, symbol in enumerate(symbols):
        if symbol != anchor_element:
            continue
        observed = sorted(symbols[neighbor] for neighbor in adjacency[index])
        if observed == expected_pattern:
            candidates.append(index)
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one {anchor_element} anchor with neighbors {expected_pattern}; "
            f"found {len(candidates)}"
        )
    anchor_index = candidates[0]
    donor_indices = sorted(
        neighbor
        for neighbor in adjacency[anchor_index]
        if symbols[neighbor] == adsorption["surface_parent_element"]
    )
    donor_labels = list(adsorption["site_discovery"]["probe_donor_labels"])
    if len(donor_indices) != len(donor_labels):
        raise ValueError(
            f"Anchor has {len(donor_indices)} donor atoms; maintained profile "
            f"requires {len(donor_labels)}"
        )
    released_hydrogen_indices = sorted(
        {
            neighbor
            for donor_index in donor_indices
            for neighbor in adjacency[donor_index]
            if symbols[neighbor] == "H"
        }
    )
    expected_released = int(adsorption["released_protons"])
    observed_released = len(released_hydrogen_indices)
    if observed_released == expected_released:
        protonation = {
            "input_state": "neutral_acid",
            "observed_acidic_hydrogen_count": observed_released,
            "catalog_released_protons_per_sam": expected_released,
            "inventory_source": "input_topology",
            "allowed_scope": "public_plan_and_substrate_materialization",
        }
    elif observed_released == 0 and allow_deprotonated_h0:
        protonation = {
            "input_state": "already_deprotonated_h0",
            "observed_acidic_hydrogen_count": 0,
            "catalog_released_protons_per_sam": expected_released,
            "inventory_source": "chemistry_profile",
            "allowed_scope": "public_plan_and_materialization",
        }
    else:
        raise ValueError(
            f"SAM anchor has {observed_released} acidic O-H hydrogens; maintained "
            f"profile requires either {expected_released} for a neutral acid or "
            "zero for a chemistry-profile-backed deprotonated H0"
        )
    unwrapped = _unwrap_molecule(positions, adjacency, cell, pbc, anchor_index)
    all_differences = unwrapped[:, None, :] - unwrapped[None, :, :]
    maximum_atom_pair_distance = float(
        np.max(np.linalg.norm(all_differences, axis=2))
    )
    heavy_indices = np.flatnonzero(np.asarray(symbols) != "H")
    heavy_positions = unwrapped[heavy_indices]
    heavy_differences = (
        heavy_positions[:, None, :] - heavy_positions[None, :, :]
    )
    maximum_heavy_atom_pair_distance = float(
        np.max(np.linalg.norm(heavy_differences, axis=2))
    )
    maximum_anchor_distance = float(
        np.max(np.linalg.norm(unwrapped - unwrapped[anchor_index], axis=1))
    )

    non_headgroup = [
        neighbor
        for neighbor in adjacency[anchor_index]
        if symbols[neighbor] != adsorption["surface_parent_element"]
    ]
    footprint = None
    if len(non_headgroup) == 1:
        axis = unwrapped[non_headgroup[0]] - unwrapped[anchor_index]
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm > 1.0e-8:
            axis /= axis_norm
            heavy = np.asarray(
                [position for symbol, position in zip(symbols, unwrapped) if symbol != "H"]
            )
            relative = heavy - unwrapped[anchor_index]
            lateral = relative - np.outer(relative @ axis, axis)
            differences = lateral[:, None, :] - lateral[None, :, :]
            footprint = float(np.max(np.linalg.norm(differences, axis=2)))

    return {
        "path": str(path),
        "sha256": _sha256(path),
        "atom_count": len(atoms),
        "formula": _formula(symbols),
        "connected_components": 1,
        "anchor": {
            "element": anchor_element,
            "atom_id_1based": anchor_index + 1,
            "neighbor_pattern": expected_pattern,
            "donor_labels": donor_labels,
            "donor_atom_ids_1based": [index + 1 for index in donor_indices],
        },
        "released_hydrogen_atom_ids_1based": [
            index + 1 for index in released_hydrogen_indices
        ],
        "protonation": protonation,
        "h0_atom_count": len(atoms) - len(released_hydrogen_indices),
        "size_metrics_A": {
            "maximum_atom_pair_distance": maximum_atom_pair_distance,
            "maximum_heavy_atom_pair_distance": (
                maximum_heavy_atom_pair_distance
            ),
            "maximum_anchor_to_atom_distance": maximum_anchor_distance,
            "definition": (
                "whole unwrapped molecule; orientation-independent conservative "
                "diameters"
            ),
        },
        "estimated_lateral_footprint_diameter_A": footprint,
    }


def _layer_counts(values: np.ndarray, gap: float) -> list[int]:
    ordered = np.sort(np.asarray(values, dtype=float))
    groups: list[list[float]] = []
    for value in ordered:
        if not groups or value - groups[-1][-1] > gap:
            groups.append([float(value)])
        else:
            groups[-1].append(float(value))
    return [len(group) for group in groups]


def _layer_index_groups(indices: np.ndarray, coordinates: np.ndarray, gap: float):
    ordered = sorted((float(coordinates[index]), int(index)) for index in indices)
    groups: list[list[int]] = []
    previous_coordinate = None
    for coordinate, index in ordered:
        if previous_coordinate is None or coordinate - previous_coordinate > gap:
            groups.append([index])
        else:
            groups[-1].append(index)
        previous_coordinate = coordinate
    return groups


def _doping_layer_index_groups(
    atoms,
    doping: dict,
    slab: dict,
    normal_axis: int,
) -> tuple[list[list[int]], list[list[int]]]:
    """Return complete structural layers and substitution-host members per layer."""

    if doping.get("layer_partition") != (
        "equal_stoichiometric_blocks_by_outward_normal"
    ):
        raise ValueError("Unsupported doping layer-partition policy")
    layer_count = int(slab["source_layers"])
    if len(atoms) % layer_count:
        raise ValueError(
            "Substrate atom count is not divisible by the declared source-layer count"
        )
    normal = np.array(atoms.cell[normal_axis], dtype=float, copy=True)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 1.0e-12:
        raise ValueError("Substrate outward-normal cell vector must be nonzero")
    normal /= normal_norm
    coordinates = np.asarray(atoms.positions, dtype=float) @ normal
    order = np.argsort(coordinates, kind="stable")
    atoms_per_layer = len(atoms) // layer_count
    symbols = np.asarray(atoms.get_chemical_symbols())
    all_layers: list[list[int]] = []
    for layer_index in range(layer_count):
        start = layer_index * atoms_per_layer
        stop = start + atoms_per_layer
        all_layers.append(sorted(int(value) for value in order[start:stop]))
    layer_formulas = [
        _formula(symbols[index] for index in layer) for layer in all_layers
    ]
    if any(formula != layer_formulas[0] for formula in layer_formulas[1:]):
        raise ValueError(
            "Equal normal-coordinate blocks are not stoichiometrically identical layers"
        )
    minimum_gap = float(slab["layer_gap_A"])
    for lower, upper in zip(all_layers, all_layers[1:]):
        boundary_gap = float(
            np.min(coordinates[upper]) - np.max(coordinates[lower])
        )
        if boundary_gap + 1.0e-8 < minimum_gap:
            raise ValueError(
                "A substrate doping-layer boundary does not pass the configured "
                "normal gap"
            )
    host_element = doping["host_element"]
    host_layers = [
        [index for index in layer if symbols[index] == host_element]
        for layer in all_layers
    ]
    if any(not layer for layer in host_layers):
        raise ValueError(
            f"A registered structural layer contains no {host_element} host sites"
        )
    return all_layers, host_layers


def _periodic_distances(
    positions: np.ndarray, reference: np.ndarray, cell: np.ndarray
) -> np.ndarray:
    differences = positions - reference
    fractional = differences @ np.linalg.inv(cell)
    fractional[:, :2] -= np.round(fractional[:, :2])
    return np.linalg.norm(fractional @ cell, axis=1)


def _stable_site_rank(seed: int, layer: int, atom_index: int) -> str:
    return hashlib.sha256(f"{seed}:{layer}:{atom_index}".encode()).hexdigest()


def _select_farthest_sites(
    atoms,
    candidates: list[int],
    count: int,
    seed: int,
    layer: int,
    minimum_separation: float,
    existing_selected: list[int] | None = None,
) -> tuple[list[int], float | None]:
    if count == 0:
        return [], None
    if count > len(candidates):
        raise ValueError("Requested more dopants than eligible host sites")
    positions = np.asarray(atoms.positions, dtype=float)
    cell = np.asarray(atoms.cell, dtype=float)
    prior = list(existing_selected or [])
    ordered = sorted(candidates, key=lambda index: _stable_site_rank(seed, layer, index))
    if prior:
        initial_distances = {
            index: float(
                np.min(_periodic_distances(positions[prior], positions[index], cell))
            )
            for index in ordered
        }
        best_initial = max(initial_distances.values())
        initial_ties = [
            index
            for index in ordered
            if abs(initial_distances[index] - best_initial) <= 1.0e-10
        ]
        first = min(
            initial_ties,
            key=lambda index: _stable_site_rank(seed, layer, index),
        )
    else:
        initial_distances = {index: math.inf for index in ordered}
        first = ordered[0]
    selected = [first]
    available = set(ordered) - {first}
    minimum_distances = {
        index: initial_distances[index] for index in available
    }
    while len(selected) < count:
        latest = selected[-1]
        candidate_list = sorted(available)
        distances = _periodic_distances(
            positions[candidate_list], positions[latest], cell
        )
        for index, distance in zip(candidate_list, distances):
            minimum_distances[index] = min(minimum_distances[index], float(distance))
        best_distance = max(minimum_distances[index] for index in candidate_list)
        tied = [
            index
            for index in candidate_list
            if abs(minimum_distances[index] - best_distance) <= 1.0e-10
        ]
        chosen = min(tied, key=lambda index: _stable_site_rank(seed, layer, index))
        selected.append(chosen)
        available.remove(chosen)
        minimum_distances.pop(chosen)

    combined = prior + selected
    if len(combined) == 1:
        realized_minimum = None
    else:
        realized_minimum = math.inf
        for offset, index in enumerate(combined[:-1]):
            distances = _periodic_distances(
                positions[combined[offset + 1 :]], positions[index], cell
            )
            realized_minimum = min(realized_minimum, float(np.min(distances)))
        if realized_minimum + 1.0e-8 < minimum_separation:
            raise ValueError(
                f"Dopant minimum separation after layer {layer} is "
                f"{realized_minimum:.6g} A; required {minimum_separation:.6g} A"
            )
    return sorted(selected), realized_minimum


def _allocate_layer_dopants(
    layer_host_counts: list[int],
    fractions: list[float],
    expected_total: int,
    policy: str,
) -> list[int]:
    raw = [count * fraction for count, fraction in zip(layer_host_counts, fractions)]
    result = [int(math.floor(value)) for value in raw]
    residual = expected_total - sum(result)
    if policy == "largest_remainder_bottom_to_top":
        layer_priority = list(range(len(raw)))
    elif policy == "largest_remainder_maximize_layer_span":
        remaining = set(range(len(raw)))
        layer_priority = []
        while remaining:
            if not layer_priority:
                selected = min(remaining)
            else:
                selected = max(
                    remaining,
                    key=lambda index: (
                        min(abs(index - prior) for prior in layer_priority),
                        -index,
                    ),
                )
            layer_priority.append(selected)
            remaining.remove(selected)
    else:
        raise ValueError("Unsupported doping layer-integer allocation policy")
    priority_rank = {
        layer_index: rank for rank, layer_index in enumerate(layer_priority)
    }
    order = sorted(
        range(len(raw)),
        key=lambda index: (
            -(raw[index] - result[index]),
            priority_rank[index],
        ),
    )
    if residual < 0 or residual > len(order):
        raise ValueError("Layer doping fractions are inconsistent with target concentration")
    for index in order[:residual]:
        result[index] += 1
    if any(dopants > sites for dopants, sites in zip(result, layer_host_counts)):
        raise ValueError("Layer doping request exceeds available host sites")
    return result


def _resolve_public_doping_request(
    recipe: dict,
    requested_site_fraction: float | None,
) -> dict:
    """Resolve the catalog recommendation or an explicit sublattice-site override."""

    registered = recipe["doping"]
    default_fraction = float(registered["target_site_fraction"])
    default_basis = dict(registered["target_fraction_basis"])
    override_contract = dict(registered["user_override"])
    recommendation = {
        "site_fraction": default_fraction,
        "site_fraction_percent": 100.0 * default_fraction,
        "physical_target_fraction": float(default_basis["physical_target_fraction"]),
        "physical_target_fraction_percent": (
            100.0 * float(default_basis["physical_target_fraction"])
        ),
        "quantity": default_basis["quantity"],
        "objective": default_basis["objective"],
        "selection_method": default_basis["selection_method"],
        "rationale": default_basis["rationale"],
        "evidence": list(default_basis["evidence"]),
        "limitation": default_basis["limitation"],
    }
    resolved = dict(registered)
    resolved["layer_site_fractions"] = [
        float(value) for value in registered["layer_site_fractions"]
    ]
    resolved["catalog_recommended_default"] = recommendation
    resolved["user_override_contract"] = override_contract

    if requested_site_fraction is None:
        resolved["selection"] = {
            "origin": "catalog_recommended_default",
            "selected_site_fraction": default_fraction,
            "selected_site_fraction_percent": 100.0 * default_fraction,
            "difference_from_catalog_default_percentage_points": 0.0,
            "reason": default_basis["rationale"],
            "warning": default_basis["limitation"],
        }
        resolved["layer_profile_scale_from_catalog"] = 1.0
        return resolved

    if isinstance(requested_site_fraction, bool):
        raise ValueError("User dopant site fraction must be a number")
    requested = float(requested_site_fraction)
    if not math.isfinite(requested):
        raise ValueError("User dopant site fraction must be finite")
    requested_percent = 100.0 * requested
    minimum_percent = float(override_contract["minimum_percent"])
    maximum_percent = float(override_contract["maximum_percent"])
    if not minimum_percent <= requested_percent <= maximum_percent:
        raise ValueError(
            f"User dopant site percentage {requested_percent:.12g}% is outside the "
            f"registered range {minimum_percent:.12g}-{maximum_percent:.12g}%; "
            f"{override_contract['range_rationale']}"
        )
    profile_scale = requested / default_fraction
    layer_fractions = [
        float(value) * profile_scale
        for value in registered["layer_site_fractions"]
    ]
    if any(value >= 1.0 for value in layer_fractions):
        raise ValueError(
            "The requested dopant fraction makes the scaled layer profile occupy an "
            "entire host layer; register a separately reviewed layer policy"
        )
    selected_basis = {
        "kind": "user_override",
        "status": "user_supplied_not_catalog_recommended",
        "quantity": default_basis["quantity"],
        "requested_site_fraction": requested,
        "requested_site_fraction_percent": requested_percent,
        "input_contract": override_contract["input_quantity"],
        "input_option": override_contract["cli_option"],
        "rationale": (
            "Use the explicit user-requested substitutional-site percentage instead "
            "of the catalog recommendation."
        ),
        "evidence": ["user_input:--dopant-site-percent_or_API"],
        "limitation": (
            "A user override is an input assumption, not a newly calculated optimum; "
            "the finite supercell may realize a nearby integer percentage."
        ),
    }
    resolved["target_site_fraction"] = requested
    resolved["layer_site_fractions"] = layer_fractions
    resolved["target_fraction_basis"] = selected_basis
    resolved["layer_profile_scale_from_catalog"] = profile_scale
    resolved["selection"] = {
        "origin": "user_override",
        "selected_site_fraction": requested,
        "selected_site_fraction_percent": requested_percent,
        "difference_from_catalog_default_percentage_points": (
            requested_percent - 100.0 * default_fraction
        ),
        "reason": selected_basis["rationale"],
        "warning": selected_basis["limitation"],
    }
    return resolved


def _resolve_public_supercell(
    recipe: dict,
    sam: dict,
    source_atoms,
    workflow: str,
    doping: dict,
    target_lateral_size_nm: tuple[float, float] | None = None,
) -> dict:
    """Choose either the maintained dense cell or a size-adaptive monomer cell."""

    globally_supported = {"single_molecule_adsorption", "dense_monolayer"}
    if workflow not in globally_supported:
        raise ValueError(
            f"Unsupported public workflow {workflow!r}; expected one of "
            f"{sorted(globally_supported)}"
        )
    recipe_supported = set(recipe["supercell"]["supported_workflows"])
    sized_estimate = target_lateral_size_nm is not None
    if workflow not in recipe_supported and not (
        sized_estimate and workflow == "dense_monolayer"
    ):
        raise ValueError(
            f"Substrate {recipe['key']} does not support public workflow {workflow!r}; "
            f"registered workflows are {sorted(recipe_supported)}"
        )
    transform = np.asarray(recipe["transform_matrix"], dtype=int)
    source_cell = np.asarray(source_atoms.cell, dtype=float)
    normal_axis = int(recipe["surface"]["normal_axis"])
    plane_axes = [axis for axis in range(3) if axis != normal_axis]
    source_host_count = sum(
        symbol == doping["host_element"]
        for symbol in source_atoms.get_chemical_symbols()
    )

    def describe_matrix_candidate(
        matrix: np.ndarray, repeat: list[int] | None = None
    ) -> dict:
        matrix = np.asarray(matrix, dtype=int)
        multiplier = int(round(np.linalg.det(matrix)))
        if multiplier <= 0:
            raise ValueError("surface supercell matrix must have positive determinant")
        cell = matrix @ source_cell
        shortest = _shortest_in_plane_translation(cell, plane_axes)
        total_hosts = source_host_count * multiplier
        dopants = int(round(total_hosts * float(doping["target_site_fraction"])))
        realized_fraction = dopants / total_hosts
        fraction_error = abs(
            realized_fraction - float(doping["target_site_fraction"])
        )
        area = float(
            np.linalg.norm(np.cross(cell[plane_axes[0]], cell[plane_axes[1]]))
        )
        return {
            "repeat": repeat,
            "matrix": matrix.tolist(),
            "source_cell_multiplier": multiplier,
            "surface_area_A2": area,
            "shortest_in_plane_translation_A": shortest,
            "host_site_count": total_hosts,
            "dopant_count": dopants,
            "realized_dopant_fraction": realized_fraction,
            "dopant_fraction_absolute_error": fraction_error,
            "dopant_fraction_within_tolerance": (
                fraction_error <= float(doping["tolerance"])
            ),
        }

    def describe_candidate(repeat: list[int]) -> dict:
        matrix = np.diag(np.asarray(repeat, dtype=int)) @ transform
        return describe_matrix_candidate(matrix, repeat)

    if sized_estimate:
        if workflow != "dense_monolayer":
            raise ValueError(
                "target lateral size applies only to the dense-monolayer workflow"
            )
        requested_A = np.asarray(
            [10.0 * value for value in target_lateral_size_nm], dtype=float
        )
        surface_basis = source_cell[plane_axes]
        gram = surface_basis @ surface_basis.T
        schur_complement = float(
            gram[0, 0] - gram[0, 1] * gram[0, 1] / gram[1, 1]
        )
        if gram[1, 1] <= 1.0e-24 or schur_complement <= 1.0e-24:
            raise ValueError("source surface lattice vectors are linearly dependent")

        target_fraction = float(doping["target_site_fraction"])
        doping_tolerance = float(doping["tolerance"])
        if source_host_count <= 0 or doping_tolerance <= 0.0:
            raise ValueError("surface recipe must provide host sites and positive tolerance")
        guaranteed_determinant_bound = max(
            1, int(math.ceil(0.5 / (source_host_count * doping_tolerance)))
        )
        seed_determinant = None
        for determinant in range(1, guaranteed_determinant_bound + 1):
            total_hosts = source_host_count * determinant
            realized_fraction = round(total_hosts * target_fraction) / total_hosts
            if abs(realized_fraction - target_fraction) <= doping_tolerance:
                seed_determinant = determinant
                break
        if seed_determinant is None:
            raise ValueError("cannot construct a dopant-compatible target surface cell")

        seed_root = int(math.ceil(math.sqrt(seed_determinant)))
        seed_coefficients = np.asarray(
            [
                [seed_root, 1],
                [seed_root * seed_root - seed_determinant, seed_root],
            ],
            dtype=int,
        )
        seed_matrix = np.eye(3, dtype=int)
        seed_matrix[plane_axes[0], :] = 0
        seed_matrix[plane_axes[1], :] = 0
        seed_matrix[plane_axes[0], plane_axes] = seed_coefficients[0]
        seed_matrix[plane_axes[1], plane_axes] = seed_coefficients[1]
        target_area = float(np.prod(requested_A))

        def annotate_sized_candidate(candidate: dict) -> dict:
            cell = np.asarray(candidate["matrix"], dtype=int) @ source_cell
            realized_A = np.asarray(
                [float(np.linalg.norm(cell[axis])) for axis in plane_axes]
            )
            relative_errors = np.abs(realized_A / requested_A - 1.0)
            candidate["realized_lateral_size_A"] = realized_A.tolist()
            candidate["maximum_relative_lateral_size_error"] = float(
                np.max(relative_errors)
            )
            candidate["sum_squared_relative_lateral_size_error"] = float(
                np.sum(relative_errors**2)
            )
            candidate["relative_surface_area_error"] = abs(
                candidate["surface_area_A2"] / target_area - 1.0
            )
            return candidate

        def sized_candidate_score(candidate: dict) -> tuple:
            return (
                candidate["maximum_relative_lateral_size_error"],
                candidate["sum_squared_relative_lateral_size_error"],
                candidate["relative_surface_area_error"],
                candidate["source_cell_multiplier"],
                tuple(value for row in candidate["matrix"] for value in row),
            )

        selected = annotate_sized_candidate(describe_matrix_candidate(seed_matrix))
        if not selected["dopant_fraction_within_tolerance"]:
            raise ValueError("internal dopant-compatible seed construction failed")
        initial_error_bound = selected["maximum_relative_lateral_size_error"]
        maximum_vector_length = float(
            np.max(requested_A * (1.0 + initial_error_bound))
        )
        first_coefficient_limit = int(
            math.ceil(maximum_vector_length / math.sqrt(schur_complement))
        ) + 1
        lattice_vectors = []
        quadratic_a = float(gram[1, 1])
        for first in range(-first_coefficient_limit, first_coefficient_limit + 1):
            quadratic_b = 2.0 * float(gram[0, 1]) * first
            quadratic_c = float(gram[0, 0]) * first * first - (
                maximum_vector_length + 1.0e-10
            ) ** 2
            discriminant = quadratic_b * quadratic_b - 4.0 * quadratic_a * quadratic_c
            if discriminant < 0.0:
                continue
            square_root = math.sqrt(max(0.0, discriminant))
            second_minimum = int(
                math.ceil((-quadratic_b - square_root) / (2.0 * quadratic_a) - 1.0e-10)
            )
            second_maximum = int(
                math.floor((-quadratic_b + square_root) / (2.0 * quadratic_a) + 1.0e-10)
            )
            for second in range(second_minimum, second_maximum + 1):
                if first == 0 and second == 0:
                    continue
                if first < 0 or (first == 0 and second < 0):
                    continue
                if len(lattice_vectors) >= MAXIMUM_SIZED_SURFACE_LATTICE_VECTORS:
                    raise ValueError(
                        "target lateral size generates too many fixed-lattice vectors; "
                        "reduce the requested size or register a reviewed supercell"
                    )
                coefficients = np.asarray([first, second], dtype=int)
                vector = coefficients @ surface_basis
                lattice_vectors.append(
                    {
                        "coefficients": coefficients,
                        "length_A": float(np.linalg.norm(vector)),
                    }
                )
        axis_vector_candidates = []
        for requested in requested_A:
            axis_vector_candidates.append(
                sorted(
                    (
                        {
                            **item,
                            "relative_error": abs(item["length_A"] - requested)
                            / requested,
                        }
                        for item in lattice_vectors
                    ),
                    key=lambda item: (
                        item["relative_error"],
                        int(np.dot(item["coefficients"], item["coefficients"])),
                        tuple(int(value) for value in item["coefficients"]),
                    ),
                )
            )
        considered_candidate_count = 1
        eligible_candidate_count = 1
        for first_vector in axis_vector_candidates[0]:
            if (
                first_vector["relative_error"]
                > selected["maximum_relative_lateral_size_error"] + 1.0e-12
            ):
                break
            for second_vector in axis_vector_candidates[1]:
                if (
                    max(
                        first_vector["relative_error"],
                        second_vector["relative_error"],
                    )
                    > selected["maximum_relative_lateral_size_error"] + 1.0e-12
                ):
                    break
                first_coefficients = first_vector["coefficients"]
                second_coefficients = second_vector["coefficients"].copy()
                determinant = int(
                    first_coefficients[0] * second_coefficients[1]
                    - first_coefficients[1] * second_coefficients[0]
                )
                if determinant == 0:
                    continue
                if determinant < 0:
                    second_coefficients *= -1
                matrix = np.eye(3, dtype=int)
                matrix[plane_axes[0], :] = 0
                matrix[plane_axes[1], :] = 0
                matrix[plane_axes[0], plane_axes] = first_coefficients
                matrix[plane_axes[1], plane_axes] = second_coefficients
                candidate = describe_matrix_candidate(matrix)
                considered_candidate_count += 1
                if not candidate["dopant_fraction_within_tolerance"]:
                    continue
                eligible_candidate_count += 1
                candidate = annotate_sized_candidate(candidate)
                if sized_candidate_score(candidate) < sized_candidate_score(selected):
                    selected = candidate
        cell = np.asarray(selected["matrix"], dtype=int) @ source_cell
        realized_A = [float(np.linalg.norm(cell[axis])) for axis in plane_axes]
        selected.update(
            {
                "workflow": workflow,
                "selection_mode": "user_nearest_lateral_size",
                "target_sam_count": None,
                "molecular_size_metric": None,
                "molecular_diameter_A": None,
                "requested_lateral_image_clearance_A": None,
                "required_shortest_in_plane_translation_A": None,
                "realized_lateral_image_clearance_A": None,
                "considered_candidate_count": considered_candidate_count,
                "eligible_candidate_count": eligible_candidate_count,
                "candidate_search_complete_within_proven_bound": True,
                "minimum_multiplier_eligible_candidates": [],
                "requested_lateral_size_nm": (requested_A / 10.0).tolist(),
                "realized_lateral_size_nm": [value / 10.0 for value in realized_A],
                "selection_rationale": (
                    "Choose the dopant-compatible fixed-lattice integer supercell whose "
                    "two periodic surface-vector lengths are closest to the requested "
                    "lengths; the realized lengths may be smaller or larger."
                ),
            }
        )
        return selected

    if workflow == "dense_monolayer":
        repeat = list(recipe["supercell"]["repeat"])
        selected = describe_candidate(repeat)
        selected.update(
            {
                "workflow": workflow,
                "selection_mode": "catalog_fixed_dense_monolayer_cell",
                "target_sam_count": int(
                    recipe["supercell"]["minimum_sam_count"]
                ),
                "molecular_size_metric": None,
                "required_shortest_in_plane_translation_A": None,
                "realized_lateral_image_clearance_A": None,
                "considered_candidate_count": 1,
                "eligible_candidate_count": 1,
                "minimum_multiplier_eligible_candidates": [repeat],
                "selection_rationale": (
                    "Use the separately validated dense-monolayer catalog cell."
                ),
            }
        )
        return selected

    contract = recipe["single_molecule_adsorption"]
    size_metric = contract["size_metric"]
    diameter = float(sam["size_metrics_A"][size_metric])
    requested_clearance = float(contract["minimum_lateral_image_clearance_A"])
    required_translation = diameter + requested_clearance
    maximum_repeat = list(contract["maximum_repeat"])
    candidates = []
    ranges = [
        range(1, maximum_repeat[axis] + 1) if axis in plane_axes else [1]
        for axis in range(3)
    ]
    for repeat_tuple in product(*ranges):
        candidate = describe_candidate(list(repeat_tuple))
        candidate["image_clearance_passed"] = (
            candidate["shortest_in_plane_translation_A"] + 1.0e-8
            >= required_translation
        )
        candidate["eligible"] = (
            candidate["image_clearance_passed"]
            and candidate["dopant_fraction_within_tolerance"]
        )
        candidates.append(candidate)
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    if not eligible:
        best_translation = max(
            candidate["shortest_in_plane_translation_A"] for candidate in candidates
        )
        raise ValueError(
            "No allowed single-molecule substrate supercell satisfies both the "
            f"required {required_translation:.6g} A image translation and dopant "
            f"tolerance; maximum tested translation was {best_translation:.6g} A"
        )
    selected = min(
        eligible,
        key=lambda candidate: (
            candidate["source_cell_multiplier"],
            candidate["surface_area_A2"],
            candidate["shortest_in_plane_translation_A"] - required_translation,
            candidate["repeat"],
        ),
    )
    minimum_multiplier = selected["source_cell_multiplier"]
    minimum_multiplier_eligible = [
        candidate
        for candidate in eligible
        if candidate["source_cell_multiplier"] == minimum_multiplier
    ]
    selected.update(
        {
            "workflow": workflow,
            "selection_mode": contract["supercell_selection"],
            "target_sam_count": int(contract["target_sam_count"]),
            "molecular_size_metric": size_metric,
            "molecular_diameter_A": diameter,
            "requested_lateral_image_clearance_A": requested_clearance,
            "required_shortest_in_plane_translation_A": required_translation,
            "realized_lateral_image_clearance_A": (
                selected["shortest_in_plane_translation_A"] - diameter
            ),
            "considered_candidate_count": len(candidates),
            "eligible_candidate_count": len(eligible),
            "minimum_multiplier_eligible_candidates": [
                {
                    "repeat": candidate["repeat"],
                    "source_cell_multiplier": candidate[
                        "source_cell_multiplier"
                    ],
                    "surface_area_A2": candidate["surface_area_A2"],
                    "shortest_in_plane_translation_A": candidate[
                        "shortest_in_plane_translation_A"
                    ],
                    "realized_lateral_image_clearance_A": candidate[
                        "shortest_in_plane_translation_A"
                    ]
                    - diameter,
                    "realized_dopant_fraction": candidate[
                        "realized_dopant_fraction"
                    ],
                }
                for candidate in sorted(
                    minimum_multiplier_eligible,
                    key=lambda item: (item["surface_area_A2"], item["repeat"]),
                )
            ],
            "selection_rationale": (
                "Choose the fewest substrate source cells that simultaneously give "
                "the whole unwrapped molecule the catalog image clearance and realize "
                "the registered dopant fraction within tolerance; break remaining "
                "ties by area, excess clearance, and repeat vector."
            ),
        }
    )
    return selected


def _resolve_adsorption_vacuum(
    recipe: dict,
    sam: dict,
    source_atoms,
    workflow: str,
) -> dict:
    """Reserve conservative top-side space for one adsorbed molecule."""

    normal_axis = int(recipe["surface"]["normal_axis"])
    normal_vector = np.asarray(source_atoms.cell[normal_axis], dtype=float)
    normal_length = float(np.linalg.norm(normal_vector))
    unit_normal = normal_vector / normal_length
    projected = np.asarray(source_atoms.positions, dtype=float) @ unit_normal
    source_bottom_vacuum = float(np.min(projected))
    source_top_vacuum = normal_length - float(np.max(projected))
    if workflow == "dense_monolayer":
        return {
            "workflow": workflow,
            "source_normal_cell_length_A": normal_length,
            "target_normal_cell_length_A": normal_length,
            "source_bottom_vacuum_A": source_bottom_vacuum,
            "source_top_vacuum_A": source_top_vacuum,
            "target_bottom_vacuum_A": source_bottom_vacuum,
            "target_top_vacuum_A": source_top_vacuum,
            "substrate_shift_along_normal_A": 0.0,
            "normal_cell_extension_A": 0.0,
            "molecular_height_envelope_A": None,
            "selection_rationale": "Preserve the validated dense-monolayer cell.",
        }
    contract = recipe["single_molecule_adsorption"]
    diameter = float(sam["size_metrics_A"][contract["size_metric"]])
    required_bottom = float(contract["minimum_bottom_vacuum_A"])
    required_top = diameter + float(contract["minimum_top_image_clearance_A"])
    bottom_shift = max(0.0, required_bottom - source_bottom_vacuum)
    top_extension = max(0.0, required_top - source_top_vacuum)
    target_length = normal_length + bottom_shift + top_extension
    return {
        "workflow": workflow,
        "source_normal_cell_length_A": normal_length,
        "target_normal_cell_length_A": target_length,
        "source_bottom_vacuum_A": source_bottom_vacuum,
        "source_top_vacuum_A": source_top_vacuum,
        "minimum_bottom_vacuum_A": required_bottom,
        "minimum_top_image_clearance_A": float(
            contract["minimum_top_image_clearance_A"]
        ),
        "molecular_height_envelope_A": diameter,
        "required_top_vacuum_A": required_top,
        "target_bottom_vacuum_A": source_bottom_vacuum + bottom_shift,
        "target_top_vacuum_A": source_top_vacuum + top_extension,
        "substrate_shift_along_normal_A": bottom_shift,
        "normal_cell_extension_A": bottom_shift + top_extension,
        "selection_rationale": (
            "Use the whole-molecule diameter as a conservative unknown-orientation "
            "height envelope, preserve bottom clearance, and add the catalog top "
            "image clearance."
        ),
    }


def _resolve_submission_recipe_and_sam(
    sam_path: Path,
    substrate_name: str,
    catalog_directory: Path | None,
    *,
    allow_deprotonated_h0: bool = False,
) -> tuple[dict, dict]:
    """Infer one maintained anchor family from the submitted molecular topology."""

    default_recipe = load_substrate_recipe(substrate_name, catalog_directory)
    if default_recipe.get("status") != "validated_surface_recipe":
        raise ValueError(
            f"Substrate {default_recipe['key']!r} currently supports maintainer "
            "adsorption-site discovery only; target supercell, slab working layers, "
            "dopant inventory, and adsorption capacity must be registered before "
            "public materialization"
        )
    adsorption = default_recipe["adsorption"]
    default_family = (
        (adsorption.get("site_discovery") or {})
        .get("site_prototype_clustering", {})
        .get("anchor_family")
    )
    profiles = adsorption.get("site_discovery_profiles") or {}
    families = sorted(
        {family for family in [default_family] if isinstance(family, str)}
        | set(profiles)
    )
    matches = []
    failures = {}
    for family in families:
        recipe = load_substrate_recipe(
            substrate_name,
            catalog_directory,
            anchor_family=family,
        )
        try:
            sam = inspect_sam_submission(
                sam_path,
                recipe["adsorption"],
                allow_deprotonated_h0=allow_deprotonated_h0,
            )
        except ValueError as exc:
            failures[family] = str(exc)
            continue
        sam["anchor_family"] = family
        matches.append((recipe, sam))
    if len(matches) != 1:
        if not matches:
            details = "; ".join(
                f"{family}: {message}" for family, message in sorted(failures.items())
            )
            raise ValueError(
                "SAM anchor topology does not match exactly one maintained family; "
                f"checked {families}. {details}"
            )
        raise ValueError(
            "SAM anchor topology is ambiguous across maintained families: "
            + ", ".join(recipe["adsorption"]["site_discovery"]
                        ["site_prototype_clustering"]["anchor_family"]
                        for recipe, _ in matches)
        )
    return matches[0]


def _load_accepted_site_prototype_library(recipe: dict) -> tuple[dict, dict, Path]:
    """Load the one hash-registered accepted library for this surface and family."""

    discovery = recipe["adsorption"]["site_discovery"]
    anchor_family = discovery["site_prototype_clustering"]["anchor_family"]
    manifest_path = (
        recipe["catalog_path"].parent / recipe["source"]["structure_manifest"]
    ).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    matches = [
        record
        for record in manifest.get("adsorption_site_libraries", [])
        if record.get("status") == "accepted_site_prototype_library"
        and record.get("anchor_family") == anchor_family
        and record.get("source_surface_sha256") == recipe["source"]["sha256"]
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one accepted {anchor_family!r} Site Prototype library for "
            f"surface {recipe['source']['sha256']}; found {len(matches)}"
        )
    record = matches[0]
    library_path = (manifest_path.parent / record["path"]).resolve()
    if not library_path.is_file() or _sha256(library_path) != record.get("sha256"):
        raise ValueError(f"Accepted Site Prototype library is missing or stale: {library_path}")
    library = json.loads(library_path.read_text(encoding="utf-8"))
    prototypes = library.get("site_prototypes")
    valid = (
        library.get("schema_version") == 1
        and library.get("anchor_family") == anchor_family
        and (library.get("source_surface") or {}).get("sha256")
        == recipe["source"]["sha256"]
        and isinstance(prototypes, list)
        and len(prototypes) == record.get("summary", {}).get("site_prototype_count")
    )
    if not valid:
        raise ValueError(f"Accepted Site Prototype library is internally inconsistent: {library_path}")
    return record, library, library_path


def _expand_site_prototypes_to_supercell(
    source_atoms,
    supercell,
    supercell_matrix: np.ndarray,
    library: dict,
) -> list[dict]:
    """Map every validated source-cell prototype into every target-cell coset."""

    source_cell = np.asarray(source_atoms.cell, dtype=float)
    output_cell = np.asarray(supercell.cell, dtype=float)
    inverse_output_cell = np.linalg.inv(output_cell)
    pbc = np.asarray(supercell.pbc, dtype=bool)
    translations = np.rint(
        lattice_points_in_supercell(supercell_matrix) @ supercell_matrix
    ).astype(int)
    if len(translations) != int(round(np.linalg.det(supercell_matrix))):
        raise ValueError("Cannot resolve source-cell translations in target supercell")
    translations = sorted(
        translations.tolist(),
        key=lambda value: tuple(
            np.round((np.asarray(value) @ source_cell) @ inverse_output_cell, 12)
        ),
    )
    source_symbols = source_atoms.get_chemical_symbols()
    source_fractional = np.asarray(source_atoms.get_scaled_positions(wrap=True))
    supercell_symbols = supercell.get_chemical_symbols()
    supercell_fractional = np.asarray(supercell.get_scaled_positions(wrap=True))

    def position_key(element: str, fractional_xyz: np.ndarray) -> tuple:
        values = np.asarray(fractional_xyz, dtype=float).copy()
        values[pbc] -= np.floor(values[pbc])
        values[np.isclose(values, 1.0, atol=1.0e-8)] = 0.0
        return (element, *np.round(values, 7).tolist())

    atom_lookup = {}
    for atom_id, (element, fractional_xyz) in enumerate(
        zip(supercell_symbols, supercell_fractional)
    ):
        key = position_key(element, fractional_xyz)
        if key in atom_lookup:
            raise ValueError("Target supercell contains duplicate atomic coordinates")
        atom_lookup[key] = atom_id
    instances = []
    seen = set()
    for prototype in sorted(
        library["site_prototypes"], key=lambda item: item["site_prototype_id"]
    ):
        prototype_id = prototype.get("site_prototype_id")
        energy = prototype.get("adsorption_energy_eV")
        vertices = prototype.get("metal_vertices")
        donor_mapping = prototype.get("donor_metal_mapping")
        frame = prototype.get("surface_frame") or {}
        plane_axes = frame.get("periodic_fractional_axes")
        if (
            not isinstance(prototype_id, str)
            or not isinstance(energy, (int, float))
            or not isinstance(vertices, list)
            or not vertices
            or not isinstance(donor_mapping, list)
            or not isinstance(plane_axes, list)
            or len(plane_axes) != 2
        ):
            raise ValueError(f"Invalid Site Prototype record: {prototype_id!r}")
        for translation_number, translation in enumerate(translations, 1):
            translation = np.asarray(translation, dtype=int)
            actual_metals = []
            for vertex in vertices:
                source_id = vertex.get("source_atom_id")
                if (
                    vertex.get("atom_index_base") != 0
                    or not isinstance(source_id, int)
                    or not 0 <= source_id < len(source_atoms)
                    or source_symbols[source_id] != vertex.get("element")
                ):
                    raise ValueError(f"{prototype_id}: invalid source metal identity")
                registered_fractional = np.asarray(
                    vertex.get("source_fractional_xyz"), dtype=float
                )
                difference = registered_fractional - source_fractional[source_id]
                difference[np.asarray(source_atoms.pbc, dtype=bool)] -= np.round(
                    difference[np.asarray(source_atoms.pbc, dtype=bool)]
                )
                if np.linalg.norm(difference @ source_cell) > 1.0e-5:
                    raise ValueError(f"{prototype_id}: registered source metal moved")
                image = vertex.get("lattice_image")
                if (
                    not isinstance(image, list)
                    or len(image) != 2
                    or any(not isinstance(value, int) for value in image)
                ):
                    raise ValueError(f"{prototype_id}: invalid metal lattice image")
                image_shift = np.zeros(3, dtype=int)
                image_shift[plane_axes] = image
                position = (
                    registered_fractional + translation + image_shift
                ) @ source_cell
                target_fractional = position @ inverse_output_cell
                actual_id = atom_lookup.get(
                    position_key(vertex["element"], target_fractional)
                )
                if actual_id is None:
                    actual_id = _map_periodic_vertex_to_supercell(
                        supercell, position, vertex["element"]
                    )
                actual_metals.append(
                    {
                        "atom_id_1based": actual_id + 1,
                        "element": vertex["element"],
                        "source_atom_id_0based": source_id,
                        "source_lattice_image": image,
                    }
                )
            unique_metal_ids = sorted(
                {metal["atom_id_1based"] for metal in actual_metals}
            )
            mapped_donors = []
            for mapping in donor_mapping:
                vertex_index = mapping.get("metal_vertex_index")
                if not isinstance(vertex_index, int) or not 0 <= vertex_index < len(actual_metals):
                    raise ValueError(f"{prototype_id}: invalid donor-to-metal mapping")
                mapped_donors.append(
                    {
                        "probe_donor_id": mapping["probe_donor_id"],
                        "metal_atom_id_1based": actual_metals[vertex_index][
                            "atom_id_1based"
                        ],
                        "parent_contact_distance_A": mapping["distance_A"],
                    }
                )
            location = np.asarray(
                prototype["ideal_site_location"]["cartesian_A"], dtype=float
            ) + translation @ source_cell
            fractional = location @ inverse_output_cell
            fractional[pbc] -= np.floor(fractional[pbc])
            location = fractional @ output_cell
            relaxed_location = prototype.get("relaxed_adsorption_location") or {}
            parent_centroid = np.asarray(
                relaxed_location.get("relaxed_contact_metal_centroid_A"),
                dtype=float,
            )
            parent_anchor = np.asarray(
                relaxed_location.get("anchor_cartesian_A"), dtype=float
            )
            binding_oxygens = relaxed_location.get("binding_oxygen_positions")
            frame_basis = np.asarray(
                [
                    frame.get("u_cartesian_unit"),
                    frame.get("v_cartesian_unit"),
                    frame.get("outward_normal_cartesian_unit"),
                ],
                dtype=float,
            )
            if (
                parent_centroid.shape != (3,)
                or parent_anchor.shape != (3,)
                or not isinstance(binding_oxygens, list)
                or not binding_oxygens
                or frame_basis.shape != (3, 3)
                or not np.all(np.isfinite(frame_basis))
            ):
                raise ValueError(f"{prototype_id}: incomplete relaxed placement template")
            translated_centroid = parent_centroid + translation @ source_cell
            centroid_fractional = translated_centroid @ inverse_output_cell
            centroid_fractional[pbc] -= np.floor(centroid_fractional[pbc])
            wrapped_centroid = centroid_fractional @ output_cell
            target_anchor = wrapped_centroid + parent_anchor - parent_centroid
            target_donors = []
            donor_position_by_label = {
                item["probe_donor_id"]: item for item in binding_oxygens
            }
            if any(
                mapped["probe_donor_id"] not in donor_position_by_label
                for mapped in mapped_donors
            ):
                raise ValueError(f"{prototype_id}: mapped donor lacks relaxed coordinates")
            for label, donor in sorted(donor_position_by_label.items()):
                relative = np.asarray(donor["relative_surface_frame_A"], dtype=float)
                if relative.shape != (3,) or not np.all(np.isfinite(relative)):
                    raise ValueError(f"{prototype_id}: invalid relaxed donor {label}")
                target_donors.append(
                    {
                        "probe_donor_id": label,
                        "cartesian_A": (
                            wrapped_centroid + relative @ frame_basis
                        ).tolist(),
                    }
                )
            instance_id = f"{prototype_id}--cell-{translation_number:04d}"
            key = (prototype_id, tuple(unique_metal_ids))
            if key in seen:
                raise ValueError(f"Duplicate expanded Site Instance: {instance_id}")
            seen.add(key)
            instances.append(
                {
                    "site_instance_id": instance_id,
                    "parent_site_prototype_id": prototype_id,
                    "anchor_family": prototype["anchor_family"],
                    "source_cell_translation": translation.tolist(),
                    "ideal_site_location": {
                        "fractional_xyz": fractional.tolist(),
                        "cartesian_A": location.tolist(),
                        "definition": "translated ideal parent Site Prototype location",
                    },
                    "actual_metal_atom_ids_1based": unique_metal_ids,
                    "actual_metals": actual_metals,
                    "donor_metal_mapping": mapped_donors,
                    "placement_template": {
                        "source": "registered_relaxed_adsorption_motif",
                        "surface_frame": frame,
                        "contact_metal_centroid_A": wrapped_centroid.tolist(),
                        "anchor_cartesian_A": target_anchor.tolist(),
                        "binding_donor_positions": target_donors,
                    },
                    "final_denticity": prototype["final_denticity"],
                    "final_topology": prototype["final_topology"],
                    "site_energy_eV": float(energy),
                    "energy_source": "undoped_parent_approximation",
                    "doped_site_energy_policy": "reuse_undoped_parent",
                }
            )
    return sorted(instances, key=lambda item: item["site_instance_id"])


def _site_instance_conflicts(instances: list[dict]) -> tuple[list[dict], list[list[str]]]:
    by_metal = {}
    for instance in instances:
        for atom_id in instance["actual_metal_atom_ids_1based"]:
            by_metal.setdefault(atom_id, []).append(instance["site_instance_id"])
    edges = set()
    for site_ids in by_metal.values():
        edges.update(combinations(sorted(site_ids), 2))
    constraints = [
        {
            "metal_atom_id_1based": atom_id,
            "capacity": 1,
            "site_instance_ids": sorted(site_ids),
        }
        for atom_id, site_ids in sorted(by_metal.items())
    ]
    return constraints, [list(edge) for edge in sorted(edges)]


def _solve_site_instance_selection(
    instances: list[dict], metal_constraints: list[dict], target_count: int
) -> dict:
    """Solve cardinality and fixed-coverage Site Instance MILPs globally."""

    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_matrix, vstack
    except ImportError as exc:
        raise RuntimeError("Site Instance selection requires scipy.optimize.milp") from exc
    index_by_id = {
        instance["site_instance_id"]: index
        for index, instance in enumerate(instances)
    }
    rows = []
    columns = []
    for row, constraint in enumerate(metal_constraints):
        for site_id in constraint["site_instance_ids"]:
            rows.append(row)
            columns.append(index_by_id[site_id])
    matrix = coo_matrix(
        (np.ones(len(rows)), (rows, columns)),
        shape=(len(metal_constraints), len(instances)),
    ).tocsr()
    bounds = Bounds(np.zeros(len(instances)), np.ones(len(instances)))
    integrality = np.ones(len(instances), dtype=int)
    capacity = LinearConstraint(
        matrix,
        np.zeros(len(metal_constraints)),
        np.ones(len(metal_constraints)),
    )
    cardinality_result = milp(
        -np.ones(len(instances)),
        integrality=integrality,
        bounds=bounds,
        constraints=capacity,
    )
    if not cardinality_result.success:
        raise ValueError(
            "Maximum-cardinality Site Instance optimization failed: "
            + cardinality_result.message
        )
    maximum_count = int(round(float(np.sum(cardinality_result.x))))
    if target_count > maximum_count:
        raise ValueError(
            f"Target SAM count {target_count} exceeds shared-metal capacity "
            f"{maximum_count}"
        )
    occupancy_matrix = vstack(
        [matrix, coo_matrix(np.ones((1, len(instances))))]
    ).tocsr()
    lower = np.r_[np.zeros(len(metal_constraints)), target_count]
    upper = np.r_[np.ones(len(metal_constraints)), target_count]
    energies = np.asarray(
        [instance["site_energy_eV"] for instance in instances], dtype=float
    )
    fixed_result = milp(
        energies,
        integrality=integrality,
        bounds=bounds,
        constraints=LinearConstraint(occupancy_matrix, lower, upper),
    )
    if not fixed_result.success:
        raise ValueError(
            "Fixed-coverage Site Instance optimization failed: "
            + fixed_result.message
        )
    selected = [
        instance
        for instance, value in zip(instances, fixed_result.x)
        if value > 0.5
    ]
    selected_ids = {item["site_instance_id"] for item in selected}
    for constraint in metal_constraints:
        if sum(site_id in selected_ids for site_id in constraint["site_instance_ids"]) > 1:
            raise ValueError("Site Instance optimizer returned a shared-metal conflict")
    prototype_counts = Counter(
        item["parent_site_prototype_id"] for item in selected
    )
    return {
        "solver": "scipy.optimize.milp_highs",
        "metal_capacity": 1,
        "maximum_cardinality": maximum_count,
        "target_count": target_count,
        "selected_count": len(selected),
        "objective": "minimum_total_undoped_parent_adsorption_energy",
        "total_inherited_site_energy_eV": float(np.sum(
            [item["site_energy_eV"] for item in selected]
        )),
        "selected_site_instance_ids": sorted(selected_ids),
        "selected_parent_prototype_counts": dict(sorted(prototype_counts.items())),
    }


def _resolve_site_instance_selection(
    recipe: dict,
    source_atoms,
    supercell_matrix: np.ndarray,
    target_count: int,
) -> dict:
    record, library, library_path = _load_accepted_site_prototype_library(recipe)
    clean = make_supercell(source_atoms, supercell_matrix, wrap=True)
    instances = _expand_site_prototypes_to_supercell(
        source_atoms, clean, supercell_matrix, library
    )
    constraints, edges = _site_instance_conflicts(instances)
    selection = _solve_site_instance_selection(instances, constraints, target_count)
    return {
        "protocol_version": recipe["adsorption"]["site_instance_protocol"],
        "anchor_family": library["anchor_family"],
        "site_library": {
            "path": str(library_path),
            "sha256": record["sha256"],
            "promotion_fingerprint": record["promotion_fingerprint"],
            "prototype_count": len(library["site_prototypes"]),
        },
        "candidate_instance_count": len(instances),
        "metal_constraint_count": len(constraints),
        "conflict_edge_count": len(edges),
        "conflict_policy": recipe["adsorption"]["conflict_policy"],
        "selection": selection,
        "site_instances": instances,
        "metal_constraints": constraints,
        "conflict_edges": edges,
    }


def _build_doped_submission_supercell(
    source_atoms,
    supercell_matrix: np.ndarray,
    doping: dict,
    slab: dict,
    normal_axis: int,
    layer_dopant_counts: list[int],
) -> dict:
    """Build the deterministic doped slab in memory without writing artifacts."""

    clean = make_supercell(source_atoms, supercell_matrix, wrap=True)
    structural_layers, host_layers = _doping_layer_index_groups(
        clean, doping, slab, normal_axis
    )
    selected_by_layer = []
    layer_minimum_distances = []
    globally_selected: list[int] = []
    distribution = doping.get("distribution", doping.get("selection_rule"))
    use_global_separation = distribution == (
        "layer_stratified_global_farthest"
    )
    for layer_number, (indices, requested) in enumerate(
        zip(host_layers, layer_dopant_counts), 1
    ):
        selected, realized_minimum = _select_farthest_sites(
            clean,
            indices,
            requested,
            int(doping["seed"]),
            layer_number,
            float(doping["minimum_separation_A"]),
            globally_selected if use_global_separation else None,
        )
        selected_by_layer.append(selected)
        layer_minimum_distances.append(realized_minimum)
        globally_selected.extend(selected)
    selected_indices = [index for layer in selected_by_layer for index in layer]
    doped = clean.copy()
    doped_symbols = doped.get_chemical_symbols()
    for index in selected_indices:
        doped_symbols[index] = doping["dopant_element"]
    doped.set_chemical_symbols(doped_symbols)
    return {
        "clean": clean,
        "doped": doped,
        "structural_layers": structural_layers,
        "host_layers": host_layers,
        "selected_by_layer": selected_by_layer,
        "selected_indices": selected_indices,
        "layer_minimum_distances_A": layer_minimum_distances,
    }


def _registered_motif_geometry_dry_run(
    *,
    sam_path: Path,
    sam: dict,
    recipe: dict,
    doped_substrate,
    site_instances: list[dict],
    target_cell: np.ndarray,
    substrate_shift: np.ndarray,
    formal_collision_radius_scale: float | None,
) -> dict:
    """Audit the unrelaxed registered-motif candidate distribution on CPU."""

    sam_atoms = read(sam_path)
    sam_symbols = sam_atoms.get_chemical_symbols()
    sam_positions = np.asarray(sam_atoms.positions, dtype=float)
    sam_cell = np.asarray(sam_atoms.cell, dtype=float)
    sam_pbc = np.asarray(sam_atoms.pbc, dtype=bool)
    adjacency = _molecular_adjacency(sam_symbols, sam_positions, sam_cell, sam_pbc)
    anchor_index = int(sam["anchor"]["atom_id_1based"]) - 1
    unwrapped = _unwrap_molecule(
        sam_positions, adjacency, sam_cell, sam_pbc, anchor_index
    )
    donor_labels = list(sam["anchor"]["donor_labels"])
    donor_indices = [
        int(atom_id) - 1
        for atom_id in sam["anchor"]["donor_atom_ids_1based"]
    ]
    released = {
        int(atom_id) - 1 for atom_id in sam["released_hydrogen_atom_ids_1based"]
    }
    h0_indices = [index for index in range(len(sam_atoms)) if index not in released]
    h0_symbols = [sam_symbols[index] for index in h0_indices]
    substrate_symbols = doped_substrate.get_chemical_symbols()
    required_elements = sorted(set(h0_symbols + substrate_symbols))
    radii = {
        symbol: float(vdw_radii[atomic_numbers[symbol]])
        for symbol in required_elements
    }
    if any(not math.isfinite(value) or value <= 0.0 for value in radii.values()):
        raise ValueError("ASE does not provide finite positive vdW radii for all atoms")

    validation = recipe["adsorption"]["site_discovery"]["probe_relaxation"][
        "validation"
    ]
    bond_window = tuple(
        float(value) for value in validation["final_metal_oxygen_distance_A"]
    )
    collision_scales = [0.8, 0.85, 0.9, 0.95, 1.0]
    if (
        formal_collision_radius_scale is not None
        and formal_collision_radius_scale not in collision_scales
    ):
        collision_scales.append(formal_collision_radius_scale)
        collision_scales.sort()
    collision_aggregates = {
        scale: {
            "candidate_count": 0,
            "passed_candidate_count": 0,
            "rejected_candidate_count": 0,
            "vdw_collision_count": 0,
            "mapped_bond_violation_count": 0,
            "passed_candidate_ids": [],
        }
        for scale in collision_scales
    }
    footprint_areas = []
    candidate_ids = []
    candidate_site_energies = []
    candidate_site_instance_ids = []
    candidate_donor_assignments = []
    alignment_rmsd = []
    substrate_positions = np.asarray(doped_substrate.positions, dtype=float)
    substrate_atom_ids = list(range(1, len(doped_substrate) + 1))

    for instance in site_instances:
        template = instance["placement_template"]
        frame = template["surface_frame"]
        target_by_label = {
            item["probe_donor_id"]: np.asarray(item["cartesian_A"], dtype=float)
            + substrate_shift
            for item in template["binding_donor_positions"]
        }
        template_labels = [
            label for label in donor_labels if label in target_by_label
        ]
        target_points = np.vstack(
            [
                np.asarray(template["anchor_cartesian_A"], dtype=float)
                + substrate_shift,
                *[target_by_label[label] for label in template_labels],
            ]
        )
        metal_by_label = {
            item["probe_donor_id"]: int(item["metal_atom_id_1based"])
            for item in instance["donor_metal_mapping"]
        }
        for donor_permutation in permutations(donor_indices):
            source_donor_by_label = dict(zip(donor_labels, donor_permutation))
            donor_assignment = "--".join(
                f"{label}-atom-{source_donor_by_label[label] + 1:04d}"
                for label in donor_labels
            )
            candidate_id = (
                f"{instance['site_instance_id']}--donor-map--{donor_assignment}"
            )
            candidate_ids.append(candidate_id)
            candidate_site_instance_ids.append(instance["site_instance_id"])
            candidate_donor_assignments.append(
                {
                    label: source_donor_by_label[label] + 1
                    for label in donor_labels
                }
            )
            source_points = np.vstack(
                [
                    unwrapped[anchor_index],
                    *[
                        unwrapped[source_donor_by_label[label]]
                        for label in template_labels
                    ],
                ]
            )
            rotation, translation, rmsd = rigid_transform(source_points, target_points)
            placed = unwrapped @ rotation + translation
            h0_positions = placed[h0_indices]
            alignment_rmsd.append(rmsd)
            mapped_windows = {
                (
                    source_donor_by_label[label] + 1,
                    metal_by_label[label],
                ): bond_window
                for label in template_labels
            }
            for scale in collision_scales:
                audit = periodic_vdw_collision_audit(
                    molecule_positions=h0_positions,
                    molecule_symbols=h0_symbols,
                    molecule_atom_ids=[index + 1 for index in h0_indices],
                    substrate_positions=substrate_positions,
                    substrate_symbols=substrate_symbols,
                    substrate_atom_ids=substrate_atom_ids,
                    cell=target_cell,
                    periodic_axes=frame["periodic_fractional_axes"],
                    surface_frame=frame,
                    radii_A=radii,
                    radius_scale=scale,
                    mapped_bond_windows_A=mapped_windows,
                )
                aggregate = collision_aggregates[scale]
                aggregate["candidate_count"] += 1
                if audit["passed"]:
                    aggregate["passed_candidate_count"] += 1
                    aggregate["passed_candidate_ids"].append(candidate_id)
                else:
                    aggregate["rejected_candidate_count"] += 1
                aggregate["vdw_collision_count"] += audit["collision_count"]
                aggregate["mapped_bond_violation_count"] += audit[
                    "mapped_bond_violation_count"
                ]
            footprint = filled_outer_envelope_area(
                positions=h0_positions,
                symbols=h0_symbols,
                surface_frame=frame,
                radii_A=radii,
                radius_scale=1.0,
                boundary_samples_per_atom=72,
            )
            footprint_areas.append(footprint["area_A2"])
            candidate_site_energies.append(float(instance["site_energy_eV"]))

    footprint_values = np.asarray(footprint_areas, dtype=float)

    def cluster_rows_for_candidate_indices(indices: list[int]) -> list[dict]:
        selected_values = footprint_values[indices]
        rows = []
        for tolerance in [0.25, 0.5, 1.0]:
            clusters = cluster_1d_by_tolerance(
                selected_values, tolerance=tolerance
            )
            cluster_records = [
                {
                    key: value
                    for key, value in cluster.items()
                    if key != "member_indices"
                }
                | {
                    "member_candidate_ids": [
                        candidate_ids[indices[index]]
                        for index in cluster["member_indices"]
                    ]
                }
                for cluster in clusters
            ]
            rows.append(
                {
                    "tolerance_A2": tolerance,
                    "cluster_count": len(clusters),
                    "member_count": sum(
                        cluster["member_count"] for cluster in clusters
                    ),
                    "cluster_sizes": [
                        cluster["member_count"] for cluster in clusters
                    ],
                    "clusters": cluster_records,
                }
            )
        return rows

    all_candidate_cluster_rows = cluster_rows_for_candidate_indices(
        list(range(len(candidate_ids)))
    )
    formal_result = None
    screened_cluster_rows = None
    screening_funnel = {
        "conformer_count": 1,
        "placement_candidate_count": len(candidate_ids),
        "formal_collision_radius_scale": formal_collision_radius_scale,
        "collision_survivor_count": None,
        "collision_rejected_count": None,
        "footprint_clustering_scope": "formal_collision_survivors",
        "formal_footprint_cluster_tolerance_A2": None,
        "footprint_cluster_count_by_tolerance": None,
    }
    if formal_collision_radius_scale is not None:
        aggregate = collision_aggregates[formal_collision_radius_scale]
        formal_result = {
            "radius_scale": formal_collision_radius_scale,
            **aggregate,
        }
        passed_ids = set(aggregate["passed_candidate_ids"])
        passed_indices = [
            index
            for index, candidate_id in enumerate(candidate_ids)
            if candidate_id in passed_ids
        ]
        screened_cluster_rows = cluster_rows_for_candidate_indices(passed_indices)
        screening_funnel.update(
            {
                "collision_survivor_count": aggregate["passed_candidate_count"],
                "collision_rejected_count": aggregate["rejected_candidate_count"],
                "footprint_cluster_count_by_tolerance": [
                    {
                        "tolerance_A2": row["tolerance_A2"],
                        "cluster_count": row["cluster_count"],
                    }
                    for row in screened_cluster_rows
                ],
            }
        )
    formal_passed_ids = (
        set(formal_result["passed_candidate_ids"])
        if formal_result is not None
        else set()
    )
    candidate_records = [
        {
            "candidate_id": candidate_id,
            "projected_footprint_area_A2": float(footprint_area),
            "registered_motif_alignment_rmsd_A": float(rmsd),
            "parent_probe_adsorption_energy_eV": float(site_energy),
            "parent_probe_energy_role": "site_ranking_prior_not_submitted_sam_energy",
            "formal_collision_passed": (
                candidate_id in formal_passed_ids if formal_result is not None else None
            ),
        }
        | {
            "site_instance_id": site_instance_id,
            "donor_assignment_1based": donor_assignment,
        }
        for candidate_id, footprint_area, rmsd, site_energy, site_instance_id, donor_assignment in zip(
            candidate_ids,
            footprint_areas,
            alignment_rmsd,
            candidate_site_energies,
            candidate_site_instance_ids,
            candidate_donor_assignments,
        )
    ]
    return {
        "alignment_audit": {
            "method": "Kabsch_rigid_fit_anchor_plus_registered_binding_donors",
            "candidate_count": len(alignment_rmsd),
            "minimum_rmsd_A": float(np.min(alignment_rmsd)),
            "mean_rmsd_A": float(np.mean(alignment_rmsd)),
            "maximum_rmsd_A": float(np.max(alignment_rmsd)),
            "interpretation": (
                "Diagnostic fit to a registered relaxed motif; not an energy or "
                "relaxation result."
            ),
        },
        "collision_sensitivity": {
            "radii_source": "ASE ase.data.vdw_radii",
            "radii_source_version": ase.__version__,
            "resolved_radii_A": radii,
            "mapped_donor_metal_distance_window_A": list(bond_window),
            "formal_radius_scale": formal_collision_radius_scale,
            "formal_radius_scale_origin": (
                "user_input" if formal_collision_radius_scale is not None else None
            ),
            "formal_result": formal_result,
            "exploratory_results": [
                {"radius_scale": scale, **collision_aggregates[scale]}
                for scale in collision_scales
            ],
        },
        "initial_footprint": {
            "candidate_count": len(footprint_areas),
            "definition": "convex_hull_of_projected_vdw_disks",
            "radii_source": "ASE ase.data.vdw_radii",
            "radii_source_version": ase.__version__,
            "radius_scale": 1.0,
            "boundary_samples_per_atom": 72,
            "released_acidic_hydrogens_excluded": True,
            "minimum_A2": float(np.min(footprint_values)),
            "mean_A2": float(np.mean(footprint_values)),
            "maximum_A2": float(np.max(footprint_values)),
        },
        "all_candidate_footprint_cluster_sensitivity": {
            "method": "deterministic_complete_span_1d_clustering",
            "input_scope": "all_unrelaxed_placement_candidates",
            "formal_tolerance_A2": None,
            "exploratory_results": all_candidate_cluster_rows,
        },
        "screened_footprint_cluster_sensitivity": {
            "method": "deterministic_complete_span_1d_clustering",
            "input_scope": "formal_collision_survivors",
            "formal_collision_radius_scale": formal_collision_radius_scale,
            "formal_tolerance_A2": None,
            "exploratory_results": screened_cluster_rows,
        },
        "screening_funnel": screening_funnel,
        "candidate_records": candidate_records,
    }


def _validated_target_lateral_size_nm(value) -> tuple[float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(
            "target lateral size must contain two finite positive lengths in nm"
        )
    try:
        left, right = (float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "target lateral size must contain two finite positive lengths in nm"
        ) from exc
    if any(not math.isfinite(item) or item <= 0.0 for item in (left, right)):
        raise ValueError(
            "target lateral size must contain two finite positive lengths in nm"
        )
    return left, right


def _estimate_sized_monolayer(
    recipe: dict,
    sam: dict,
    source_atoms,
    supercell_resolution: dict,
    packing_fraction: float,
) -> tuple[dict, dict, dict]:
    """Estimate capacity cheaply; do not expand the requested large supercell."""

    transform = np.asarray(recipe["transform_matrix"], dtype=int)
    normal_axis = int(recipe["surface"]["normal_axis"])
    plane_axes = [axis for axis in range(3) if axis != normal_axis]
    calibration_candidates = []
    for left_repeat, right_repeat in product(range(1, 5), repeat=2):
        repeat = [1, 1, 1]
        repeat[plane_axes[0]] = left_repeat
        repeat[plane_axes[1]] = right_repeat
        matrix = np.diag(repeat) @ transform
        try:
            resolution = _resolve_site_instance_selection(
                recipe, source_atoms, matrix, 1
            )
        except ValueError:
            continue
        calibration_candidates.append((matrix, repeat, resolution))
    if not calibration_candidates:
        raise ValueError(
            "no bounded periodic calibration cell can resolve unique adsorption sites"
        )
    calibration_matrix, calibration_repeat, calibration = min(
        calibration_candidates,
        key=lambda item: (
            int(round(np.linalg.det(item[0]))),
            item[1],
        ),
    )
    calibration_multiplier = int(round(np.linalg.det(calibration_matrix)))
    target_multiplier = int(supercell_resolution["source_cell_multiplier"])
    calibration_capacity = int(calibration["selection"]["maximum_cardinality"])
    site_capacity = int(
        math.floor(
            calibration_capacity * target_multiplier / calibration_multiplier
        )
    )

    molecular_diameter = sam.get("estimated_lateral_footprint_diameter_A")
    footprint_method = "anchor_tail_axis_heavy_atom_envelope"
    if molecular_diameter is None:
        molecular_diameter = sam["size_metrics_A"][
            "maximum_heavy_atom_pair_distance"
        ]
        footprint_method = "whole_molecule_heavy_atom_diameter_fallback"
    sam_atoms = read(Path(sam["path"]))
    heavy_radii = [
        float(vdw_radii[atomic_numbers[symbol]])
        for symbol in sam_atoms.get_chemical_symbols()
        if symbol != "H" and np.isfinite(vdw_radii[atomic_numbers[symbol]])
    ]
    effective_diameter = float(molecular_diameter) + 2.0 * max(heavy_radii)
    footprint_area = math.pi * (0.5 * effective_diameter) ** 2
    surface_area = float(supercell_resolution["surface_area_A2"])

    def footprint_count(fraction: float) -> int:
        return int(math.floor(surface_area * fraction / footprint_area))

    lower_fraction = max(0.0, packing_fraction - 0.10)
    upper_fraction = min(1.0, packing_fraction + 0.10)
    estimated_count = min(site_capacity, footprint_count(packing_fraction))
    lower_count = min(site_capacity, footprint_count(lower_fraction))
    upper_count = min(site_capacity, footprint_count(upper_fraction))
    if estimated_count < 1:
        raise ValueError(
            "target surface is too small for one submitted SAM under the selected "
            "footprint estimate"
        )
    candidate_density = (
        calibration["candidate_instance_count"] / calibration_multiplier
    )
    estimated_candidate_instances = int(round(candidate_density * target_multiplier))
    summary = {
        **calibration,
        "candidate_instance_count": estimated_candidate_instances,
        "metal_constraint_count": None,
        "conflict_edge_count": None,
        "selection": None,
        "site_instances": [],
        "capacity_calibration": {
            "matrix": calibration_matrix.tolist(),
            "repeat": calibration_repeat,
            "source_cell_multiplier": calibration_multiplier,
            "maximum_cardinality": calibration_capacity,
        },
    }
    capacity = {
        "method": "fast_geometry_and_site_density_estimate_v1",
        "packing_fraction": packing_fraction,
        "packing_fraction_sensitivity": [lower_fraction, upper_fraction],
        "surface_area_A2": surface_area,
        "effective_footprint_diameter_A": effective_diameter,
        "effective_footprint_area_A2": footprint_area,
        "footprint_method": footprint_method,
        "site_capacity_upper_bound": site_capacity,
        "footprint_limited_count_at_selected_fraction": footprint_count(
            packing_fraction
        ),
        "estimated_sam_count": estimated_count,
        "estimated_sam_count_range": [lower_count, upper_count],
        "doping_handling": {
            "policy": "ignore_substitutional_doping_for_capacity_estimate",
            "dopant_element": recipe["doping"]["dopant_element"],
            "treated_as_host_element": recipe["doping"]["host_element"],
            "applies_to": "capacity_estimate_only",
        },
        "not_a_packing_or_free_energy_certificate": True,
        "limitations": [
            "Site capacity is scaled from one periodic calibration cell.",
            "Substitutional dopants are counted as their parent host species for capacity estimation.",
            "Footprint uses the submitted conformation and a circular vdW envelope.",
            "The estimate does not solve collision-aware packing or adsorption free energy.",
        ],
    }
    atoms_per_sam = int(sam["h0_atom_count"]) + int(
        recipe["adsorption"]["released_protons"]
    )
    substrate_atom_count = len(source_atoms) * target_multiplier
    total_atoms = substrate_atom_count + estimated_count * atoms_per_sam
    benchmark_path = _default_benchmark_ledger_path()
    with benchmark_path.open(encoding="utf-8") as handle:
        benchmark_ledger = json.load(handle)
    if benchmark_ledger.get("schema_version") != 1 or len(
        benchmark_ledger.get("entries", [])
    ) != 1:
        raise ValueError("SAMFlow benchmark ledger schema is unsupported")
    benchmark = benchmark_ledger["entries"][0]
    reference_atoms = int(benchmark["atom_count"])
    ratio = total_atoms / reference_atoms
    observations = benchmark["observations"]
    reference_minutes = (
        float(observations["stage0_walltime_window_seconds"])
        + float(observations["three_annealing_cycles_walltime_window_seconds"])
    ) / 60.0
    walltime_candidates = [
        reference_minutes * ratio**0.9 * 0.75,
        reference_minutes * ratio**1.3 * 1.5,
    ]
    memory_reference = float(observations["peak_gpu_memory_GiB_approximate"])
    mace_reference_estimate = {
        "status": "scaled_from_project_benchmark",
        "benchmark_id": benchmark["benchmark_id"],
        "benchmark_ledger": {
            "path": str(benchmark_path),
            "sha256": _sha256(benchmark_path),
        },
        "reference_atom_count": reference_atoms,
        "hardware": benchmark["hardware"],
        "model": benchmark["model"],
        "precision": benchmark["precision"],
        "predicted_peak_gpu_memory_GiB_range": [
            round(memory_reference * ratio * 0.8, 2),
            round(memory_reference * ratio * 1.3, 2),
        ],
        "predicted_stage0_plus_three_annealing_cycles_minutes_range": [
            round(min(walltime_candidates), 1),
            round(max(walltime_candidates), 1),
        ],
        "scaling_contract": benchmark["scaling_contract"],
        "warning": (
            "This range is valid only as a rough same-hardware, same-model, "
            "same-protocol reference. Precision was not recorded in the baseline."
        ),
        "provenance": benchmark["provenance"],
    }
    resources = {
        "method": "atom_and_force_call_accounting_v1",
        "substrate_atom_count": substrate_atom_count,
        "estimated_sam_count": estimated_count,
        "atoms_added_per_adsorbed_sam_including_released_protons": atoms_per_sam,
        "estimated_total_atom_count": total_atoms,
        "estimated_site_instance_count": estimated_candidate_instances,
        "atom_force_evaluations_per_force_call": total_atoms,
        "atom_force_evaluations_per_1000_force_calls": 1000 * total_atoms,
        "walltime_estimate": mace_reference_estimate[
            "predicted_stage0_plus_three_annealing_cycles_minutes_range"
        ],
        "walltime_status": "reference_scaled_hardware_model_specific_range",
        "mace_reference_estimate": mace_reference_estimate,
    }
    return summary, capacity, resources


def _resolve_single_adsorption_selection(
    geometry: dict,
    objective: str,
    maximum_output_structures: int,
) -> dict:
    supported = {"minimum_energy", "minimum_projected_area", "multiple"}
    if objective not in supported:
        raise ValueError(
            f"unsupported single-structure objective {objective!r}; expected one of "
            f"{sorted(supported)}"
        )
    if (
        isinstance(maximum_output_structures, bool)
        or not isinstance(maximum_output_structures, int)
        or maximum_output_structures < 1
    ):
        raise ValueError("maximum output structures must be a positive integer")
    formal = geometry["collision_sensitivity"]["formal_result"]
    if formal is None:
        raise ValueError(
            "single-structure selection requires a formal collision radius scale"
        )
    survivors = sorted(
        (
            record
            for record in geometry["candidate_records"]
            if record["formal_collision_passed"]
        ),
        key=lambda record: (
            record["projected_footprint_area_A2"],
            record["parent_probe_adsorption_energy_eV"],
            record["candidate_id"],
        ),
    )
    if not survivors:
        raise ValueError("no formal collision survivor is available for selection")
    base = {
        "objective": objective,
        "uses_large_language_model": False,
        "candidate_source": "formal_collision_survivors",
        "formal_collision_radius_scale": formal["radius_scale"],
        "candidate_count": len(survivors),
        "maximum_output_structures": maximum_output_structures,
    }
    if objective == "minimum_energy":
        return {
            **base,
            "status": "requires_relaxation",
            "requires_relaxation": True,
            "ranking_quantity": (
                "relaxed_adsorption_energy_of_the_submitted_sam_eV"
            ),
            "selected_candidate_ids": [],
            "candidates_scheduled_for_relaxation": [
                record["candidate_id"] for record in survivors
            ],
            "energy_warning": (
                "Parent probe energies are site-ranking priors and are not energies "
                "of the submitted SAM."
            ),
        }
    if objective == "minimum_projected_area":
        return {
            **base,
            "status": "selected_unrelaxed_candidate",
            "requires_relaxation": False,
            "ranking_quantity": "unrelaxed_projected_vdw_footprint_area_A2",
            "selected_candidate_ids": [survivors[0]["candidate_id"]],
            "candidates_scheduled_for_relaxation": [],
        }
    count = min(maximum_output_structures, len(survivors))
    indices = sorted(
        set(int(round(value)) for value in np.linspace(0, len(survivors) - 1, count))
    )
    selected_ids = [survivors[index]["candidate_id"] for index in indices]
    return {
        **base,
        "status": "selected_multiple_unrelaxed_candidates",
        "requires_relaxation": False,
        "ranking_quantity": "deterministic_footprint_span",
        "selected_candidate_ids": selected_ids,
        "candidates_scheduled_for_relaxation": [],
    }


def resolve_submission_plan(
    sam_path: Path,
    substrate_name: str,
    catalog_directory: Path | None = None,
    *,
    workflow: str = "single_molecule_adsorption",
    dopant_site_fraction: float | None = None,
    formal_collision_radius_scale: float | None = None,
    target_lateral_size_nm: tuple[float, float] | None = None,
    packing_fraction: float = 0.70,
    single_structure_objective: str | None = None,
    maximum_output_structures: int = 8,
) -> dict:
    """Resolve the two-input public contract without writing files or launching work."""

    target_lateral_size_nm = _validated_target_lateral_size_nm(
        target_lateral_size_nm
    )
    sized_estimate = target_lateral_size_nm is not None
    if sized_estimate:
        if workflow != "dense_monolayer":
            raise ValueError(
                "target lateral size applies only to the dense-monolayer workflow"
            )
        if (
            isinstance(packing_fraction, bool)
            or not math.isfinite(float(packing_fraction))
            or not 0.0 < float(packing_fraction) <= 1.0
        ):
            raise ValueError("packing fraction must be in the interval (0, 1]")
        packing_fraction = float(packing_fraction)
    if single_structure_objective is not None and workflow != "single_molecule_adsorption":
        raise ValueError(
            "single-structure objective applies only to single-molecule adsorption"
        )

    if formal_collision_radius_scale is not None:
        if (
            isinstance(formal_collision_radius_scale, bool)
            or not math.isfinite(float(formal_collision_radius_scale))
            or float(formal_collision_radius_scale) <= 0.0
        ):
            raise ValueError("formal collision radius scale must be a positive number")
        formal_collision_radius_scale = float(formal_collision_radius_scale)

    recipe, sam = _resolve_submission_recipe_and_sam(
        sam_path,
        substrate_name,
        catalog_directory,
        allow_deprotonated_h0=True,
    )
    source_path = recipe["source_path"]
    source_atoms = read(source_path)
    source_symbols = source_atoms.get_chemical_symbols()
    observed_formula = _formula(source_symbols)
    expected_formula = dict(sorted(recipe["source"]["expected_formula"].items()))
    if observed_formula != expected_formula:
        raise ValueError(
            f"Substrate source formula is {observed_formula}; expected {expected_formula}"
        )

    surface = recipe["surface"]
    slab = recipe["slab"]
    doping = _resolve_public_doping_request(recipe, dopant_site_fraction)
    normal_axis = surface["normal_axis"]
    host_indices = np.flatnonzero(np.asarray(source_symbols) == doping["host_element"])
    coordinate = np.asarray(source_atoms.positions)[:, normal_axis]
    structural_layers, host_layer_indices = _doping_layer_index_groups(
        source_atoms, doping, slab, normal_axis
    )
    if len(host_layer_indices) != slab["source_layers"]:
        raise ValueError(
            f"Substrate source has {len(host_layer_indices)} host layers; "
            f"catalog declares {slab['source_layers']}"
        )
    slab_thickness = float(np.ptp(coordinate))
    normal_cell_length = float(np.linalg.norm(np.asarray(source_atoms.cell)[normal_axis]))
    vacuum = normal_cell_length - slab_thickness
    if slab_thickness + 1.0e-6 < slab["minimum_thickness_A"]:
        raise ValueError(
            f"Substrate source thickness {slab_thickness:.6g} A is below the catalog minimum"
        )
    if vacuum + 1.0e-6 < slab["minimum_vacuum_A"]:
        raise ValueError(
            f"Substrate source vacuum {vacuum:.6g} A is below the catalog minimum"
        )

    supercell_resolution = _resolve_public_supercell(
        recipe,
        sam,
        source_atoms,
        workflow,
        doping,
        target_lateral_size_nm=target_lateral_size_nm,
    )
    supercell_matrix = np.asarray(supercell_resolution["matrix"], dtype=int)
    surface_cell_multiplier = int(round(np.linalg.det(supercell_matrix)))
    capacity_estimate = None
    resource_estimate = None
    if sized_estimate:
        site_resolution, capacity_estimate, resource_estimate = (
            _estimate_sized_monolayer(
                recipe,
                sam,
                source_atoms,
                supercell_resolution,
                packing_fraction,
            )
        )
        planned_sam_count = int(capacity_estimate["estimated_sam_count"])
        site_capacity = int(capacity_estimate["site_capacity_upper_bound"])
        supercell_resolution["target_sam_count"] = planned_sam_count
    else:
        planned_sam_count = int(supercell_resolution["target_sam_count"])
        site_resolution = _resolve_site_instance_selection(
            recipe, source_atoms, supercell_matrix, planned_sam_count
        )
        site_capacity = site_resolution["selection"]["maximum_cardinality"]
    dense_h0_pose_search = (
        _validated_dense_h0_pose_search(
            recipe,
            site_resolution["anchor_family"],
            required=False,
        )
        if sized_estimate
        else None
    )

    source_cell = np.asarray(source_atoms.cell, dtype=float)
    output_cell = supercell_matrix @ source_cell
    vacuum_resolution = _resolve_adsorption_vacuum(
        recipe, sam, source_atoms, workflow
    )
    output_normal = np.asarray(output_cell[normal_axis], dtype=float)
    output_cell[normal_axis] = (
        output_normal
        / np.linalg.norm(output_normal)
        * float(vacuum_resolution["target_normal_cell_length_A"])
    )
    total_host_sites = len(host_indices) * surface_cell_multiplier
    target_fraction = float(doping["target_site_fraction"])
    target_dopants = int(round(total_host_sites * target_fraction))
    realized_fraction = target_dopants / total_host_sites
    if abs(realized_fraction - target_fraction) > doping["tolerance"]:
        raise ValueError(
            "Selected supercell cannot realize the requested dopant fraction within tolerance"
        )
    layer_host_counts = [
        len(layer) * surface_cell_multiplier for layer in host_layer_indices
    ]
    layer_atom_counts = [
        len(layer) * surface_cell_multiplier for layer in structural_layers
    ]
    layer_dopants = _allocate_layer_dopants(
        layer_host_counts,
        [float(value) for value in doping["layer_site_fractions"]],
        target_dopants,
        doping["layer_integer_allocation"],
    )

    source_kind = recipe["source"]["kind"]
    fraction_basis = dict(doping["target_fraction_basis"])
    readiness_issues = [
        {
            "code": "end_to_end_execution_not_wired",
            "message": (
                "Conformer search, adsorption assembly, protonation, dynamics, and "
                "analysis are not yet connected to the two-input public entry point."
            ),
        },
    ]
    if sized_estimate and dense_h0_pose_search is not None:
        readiness_issues.append(
            {
                "code": "explicit_plan_approval_required",
                "message": (
                    "The deterministic CPU H0 builder is available only after the user "
                    "approves the exact approval_contract.plan_sha256."
                ),
            }
        )
    elif sized_estimate:
        readiness_issues.append(
            {
                "code": "dense_h0_pose_search_not_registered",
                "message": (
                    "The substrate and anchor family do not register a dense-H0 "
                    "pose-search policy; planning is read-only and execution is blocked."
                ),
            }
        )

    geometry_candidate_plan = None
    single_adsorption_selection = None
    if workflow == "single_molecule_adsorption":
        donor_count = len(sam["anchor"]["donor_atom_ids_1based"])
        donor_permutations_per_site = math.factorial(donor_count)
        site_instance_count = site_resolution["candidate_instance_count"]
        geometry_candidate_plan = {
            "status": "read_only_dry_run_pending_calibration",
            "conformer_count": 1,
            "site_instance_count": site_instance_count,
            "donor_permutations_per_site": donor_permutations_per_site,
            "registered_motif_alignment_candidate_count": (
                site_instance_count * donor_permutations_per_site
            ),
            "orientation_enumeration": {
                "status": "pending_calibration",
                "generated_orientations_per_alignment": 1,
                "generated_orientation_role": (
                    "registered_motif_alignment_only"
                ),
                "azimuth_grid_degrees": None,
                "tilt_grid_degrees": None,
            },
            "readiness_issues": [
                {
                    "code": "isolated_conformer_generation_not_connected",
                    "message": (
                        "The supplied isolated structure is the only conformer in "
                        "this read-only dry run."
                    ),
                },
                {
                    "code": "azimuth_tilt_grid_not_calibrated",
                    "message": (
                        "Only the registered adsorption-motif alignment is generated; "
                        "azimuth and tilt grids remain unset."
                    ),
                },
                *(
                    []
                    if formal_collision_radius_scale is not None
                    else [
                        {
                            "code": "vdw_collision_scale_not_selected",
                            "message": (
                                "Collision-scale sensitivity may be reported, but no "
                                "formal vdW scale has been selected."
                            ),
                        }
                    ]
                ),
                {
                    "code": "footprint_cluster_tolerance_not_selected",
                    "message": (
                        "Footprint-cluster sensitivity may be reported, but no formal "
                        "area tolerance has been selected."
                    ),
                },
            ],
        }
        geometry_substrate = _build_doped_submission_supercell(
            source_atoms,
            supercell_matrix,
            doping,
            slab,
            normal_axis,
            layer_dopants,
        )
        if len(geometry_substrate["selected_indices"]) != target_dopants:
            raise ValueError("Geometry dry run selected the wrong dopant count")
        doped_geometry = geometry_substrate["doped"]
        original_normal = np.asarray(doped_geometry.cell[normal_axis], dtype=float)
        unit_normal = original_normal / np.linalg.norm(original_normal)
        substrate_shift = (
            float(vacuum_resolution["substrate_shift_along_normal_A"])
            * unit_normal
        )
        if np.linalg.norm(substrate_shift) > 0.0:
            doped_geometry.positions += substrate_shift
        doped_geometry.set_cell(output_cell, scale_atoms=False)
        geometry_candidate_plan.update(
            _registered_motif_geometry_dry_run(
                sam_path=Path(sam["path"]),
                sam=sam,
                recipe=recipe,
                doped_substrate=doped_geometry,
                site_instances=site_resolution["site_instances"],
                target_cell=output_cell,
                substrate_shift=substrate_shift,
                formal_collision_radius_scale=formal_collision_radius_scale,
            )
        )
        if single_structure_objective is not None:
            single_adsorption_selection = _resolve_single_adsorption_selection(
                geometry_candidate_plan,
                single_structure_objective,
                maximum_output_structures,
            )
    surface_size_request = None
    if sized_estimate:
        surface_size_request = {
            "requested_lateral_size_nm": list(target_lateral_size_nm),
            "realized_lateral_size_nm": supercell_resolution[
                "realized_lateral_size_nm"
            ],
            "interpretation": "closest_reachable_lengths_from_fixed_surface_lattice",
        }
    structure_manifest_path = (
        recipe["catalog_path"].parent / recipe["source"]["structure_manifest"]
    ).resolve()
    approval_payload = {
        "planner_implementation_sha256": _sha256(Path(__file__).resolve()),
        "collision_audit_implementation_sha256": _sha256(
            COLLISION_AUDIT_IMPLEMENTATION_PATH
        ),
        "interface_validator_implementation_sha256": _sha256(
            INTERFACE_VALIDATOR_IMPLEMENTATION_PATH
        ),
        "surface_protonation_implementation_sha256": _sha256(
            SURFACE_PROTONATION_IMPLEMENTATION_PATH
        ),
        "sam_sha256": sam["sha256"],
        "substrate_catalog_sha256": _sha256(recipe["catalog_path"]),
        "substrate_source_sha256": _sha256(source_path),
        "substrate_structure_manifest_sha256": _sha256(structure_manifest_path),
        "site_library_sha256": site_resolution["site_library"]["sha256"],
        "site_library_promotion_fingerprint": site_resolution["site_library"][
            "promotion_fingerprint"
        ],
        "anchor_family": site_resolution["anchor_family"],
        "dense_h0_pose_search": dense_h0_pose_search,
        "workflow": workflow,
        "supercell_matrix": supercell_matrix.tolist(),
        "target_lateral_size_nm": (
            list(target_lateral_size_nm) if target_lateral_size_nm is not None else None
        ),
        "planned_sam_count": planned_sam_count,
        "packing_fraction": packing_fraction if sized_estimate else None,
        "formal_collision_radius_scale": formal_collision_radius_scale,
        "dopant_site_fraction": target_fraction,
        "single_structure_objective": single_structure_objective,
        "maximum_output_structures": maximum_output_structures,
    }
    approval_contract = {
        "plan_sha256": _approval_payload_sha256(approval_payload),
        "hash_scope": "approval_payload_canonical_json",
        "approval_payload": approval_payload,
        "required_before_expensive_or_dense_execution": True,
    }
    plan = {
        "schema_version": 1,
        "mode": "public_submission_plan",
        "workflow": workflow,
        "inputs": {
            "sam_structure": sam,
            "substrate_request": substrate_name,
        },
        "geometry_candidate_plan": geometry_candidate_plan,
        "single_adsorption_selection": single_adsorption_selection,
        "surface_size_request": surface_size_request,
        "capacity_estimate": capacity_estimate,
        "resource_estimate": resource_estimate,
        "approval_contract": approval_contract,
        "substrate": {
            "key": recipe["key"],
            "display_name": recipe.get("display_name", recipe["key"]),
            "status": recipe.get("status"),
            "catalog": {
                "path": str(recipe["catalog_path"]),
                "sha256": _sha256(recipe["catalog_path"]),
            },
            "source": {
                "kind": source_kind,
                "path": str(source_path),
                "sha256": _sha256(source_path),
                "structure_manifest": {
                    "path": str(structure_manifest_path),
                    "sha256": _sha256(structure_manifest_path),
                },
                "provenance": recipe["source"]["provenance"],
                "formula": observed_formula,
                "atom_count": len(source_atoms),
            },
            "surface": {
                "miller": list(surface["miller"]),
                "termination": surface["termination"],
                "normal_axis": normal_axis,
            },
            "supercell": {
                "shape": recipe["supercell"]["shape"],
                "repeat": supercell_resolution["repeat"],
                "matrix": supercell_matrix.tolist(),
                "source_cell_multiplier": surface_cell_multiplier,
                "cell_A": output_cell.tolist(),
                "adsorption_site_capacity": site_capacity,
                "adsorption_site_capacity_definition": (
                    "maximum_shared_metal_feasible_occupancy"
                ),
                "planned_sam_count": planned_sam_count,
                "selection_mode": supercell_resolution["selection_mode"],
                "molecular_size_metric": supercell_resolution[
                    "molecular_size_metric"
                ],
                "molecular_diameter_A": supercell_resolution.get(
                    "molecular_diameter_A"
                ),
                "requested_lateral_image_clearance_A": supercell_resolution.get(
                    "requested_lateral_image_clearance_A"
                ),
                "required_shortest_in_plane_translation_A": (
                    supercell_resolution[
                        "required_shortest_in_plane_translation_A"
                    ]
                ),
                "shortest_in_plane_translation_A": supercell_resolution[
                    "shortest_in_plane_translation_A"
                ],
                "realized_lateral_image_clearance_A": supercell_resolution[
                    "realized_lateral_image_clearance_A"
                ],
                "surface_area_A2": supercell_resolution["surface_area_A2"],
                "considered_candidate_count": supercell_resolution[
                    "considered_candidate_count"
                ],
                "eligible_candidate_count": supercell_resolution[
                    "eligible_candidate_count"
                ],
                "candidate_search_complete_within_proven_bound": (
                    supercell_resolution.get(
                        "candidate_search_complete_within_proven_bound"
                    )
                ),
                "minimum_multiplier_eligible_candidates": supercell_resolution[
                    "minimum_multiplier_eligible_candidates"
                ],
                "selection_rationale": supercell_resolution[
                    "selection_rationale"
                ],
            },
            "slab": {
                "source_layers": slab["source_layers"],
                "target_layers": slab["target_layers"],
                "working_layers": slab["working_layers"],
                "layer_gap_A": slab["layer_gap_A"],
                "source_thickness_A": slab_thickness,
                "vacuum_A": vacuum,
                "adsorption_vacuum": vacuum_resolution,
                "full_supercell_atom_count_before_doping": (
                    len(source_atoms) * surface_cell_multiplier
                ),
            },
            "doping": {
                "method": doping["method"],
                "host_element": doping["host_element"],
                "dopant_element": doping["dopant_element"],
                "target_site_fraction": target_fraction,
                "target_site_fraction_percent": 100.0 * target_fraction,
                "realized_site_fraction": realized_fraction,
                "realized_site_fraction_percent": 100.0 * realized_fraction,
                "absolute_fraction_error": abs(
                    realized_fraction - target_fraction
                ),
                "integer_realization_rule": (
                    "nearest_integer_host_count_times_target_fraction"
                ),
                "unrounded_dopant_count": total_host_sites * target_fraction,
                "dopant_count": target_dopants,
                "host_site_count_before_substitution": total_host_sites,
                "layer_partition": doping["layer_partition"],
                "eligible_layer_indices_bottom_to_top": list(
                    doping["eligible_layer_indices_bottom_to_top"]
                ),
                "layer_atom_counts_bottom_to_top": layer_atom_counts,
                "layer_host_site_counts_bottom_to_top": layer_host_counts,
                "layer_dopant_counts_bottom_to_top": layer_dopants,
                "selection_rule": doping["distribution"],
                "layer_integer_allocation": doping[
                    "layer_integer_allocation"
                ],
                "minimum_separation_A": doping["minimum_separation_A"],
                "seed": doping["seed"],
                "charge_compensation": doping["charge_compensation"],
                "target_fraction_basis": fraction_basis,
                "selection": doping["selection"],
                "catalog_recommended_default": doping[
                    "catalog_recommended_default"
                ],
                "user_override_contract": doping["user_override_contract"],
                "layer_profile_policy": doping["user_override_contract"][
                    "layer_profile_policy"
                ],
                "layer_profile_scale_from_catalog": doping[
                    "layer_profile_scale_from_catalog"
                ],
                "user_notice": (
                    f"Selected {100.0 * target_fraction:.6g}% means "
                    f"{fraction_basis['quantity']}; selection origin is "
                    f"{doping['selection']['origin']} and basis status is "
                    f"{fraction_basis['status']}. {fraction_basis['limitation']} "
                    f"Catalog recommendation: "
                    f"{doping['catalog_recommended_default']['site_fraction_percent']:.6g}% "
                    f"for {doping['catalog_recommended_default']['objective']}"
                ),
            },
            "adsorption": {
                "site_source": recipe["adsorption"]["site_source"],
                "site_instance_protocol": site_resolution["protocol_version"],
                "anchor_family": site_resolution["anchor_family"],
                "anchor": sam["anchor"],
                "released_protons_per_sam": recipe["adsorption"]["released_protons"],
                "surface_parent_element": recipe["adsorption"]["surface_parent_element"],
                "dense_h0_pose_search": dense_h0_pose_search,
                "site_library": site_resolution["site_library"],
                "candidate_instance_count": site_resolution[
                    "candidate_instance_count"
                ],
                "metal_constraint_count": site_resolution[
                    "metal_constraint_count"
                ],
                "conflict_edge_count": site_resolution["conflict_edge_count"],
                "conflict_policy": site_resolution["conflict_policy"],
                "surface_protons_in_integer_program": False,
                "selection": site_resolution["selection"],
                "single_molecule_candidate_policy": (
                    "retain_all_instances_for_real_sam_adsorption; the selected "
                    "instance is only the inherited-energy default"
                    if workflow == "single_molecule_adsorption"
                    else None
                ),
            },
        },
        "stages": [
            "sam_intake",
            "substrate_supercell_and_slab",
            "deterministic_doping",
            "adsorption_site_resolution",
            "conformer_search",
            (
                "single_molecule_site_adsorption_screen"
                if workflow == "single_molecule_adsorption"
                else "dense_monolayer_assembly"
            ),
            "post_sam_surface_protonation",
            "validation",
            "stage0",
            "annealing",
            "production",
            "layer_restoration",
            "structure_trajectory_dipole_analysis",
            "final_report",
        ],
        "execution_contract": {
            "input_level": "isolated_sam_structure_and_substrate_catalog_key",
            "executable": False,
            "approved_cpu_h0_builder_available": bool(
                sized_estimate and dense_h0_pose_search is not None
            ),
            "approved_cpu_h0_builder_scope": (
                "deterministic_collision_screened_unrelaxed_H0_only"
                if sized_estimate and dense_h0_pose_search is not None
                else None
            ),
            "ready_stages": [
                "sam_intake",
                "substrate_recipe_resolution",
                "substrate_build_planning",
                "substrate_supercell_and_slab",
                "deterministic_doping",
                "layer_group_resolution",
                "site_prototype_library_resolution",
                "site_instance_expansion",
                "shared_metal_conflict_resolution",
                (
                    "single_molecule_default_site_ranking"
                    if workflow == "single_molecule_adsorption"
                    else "fixed_coverage_site_selection"
                ),
            ],
            "readiness_issues": readiness_issues,
        },
    }
    approval_payload["dense_h0_execution_plan_sha256"] = _canonical_json_sha256(
        _dense_h0_execution_plan_identity(plan)
    )
    approval_contract["plan_sha256"] = _approval_payload_sha256(approval_payload)
    return plan


def _path_component(value: str) -> str:
    component = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-.")
    if not component:
        raise ValueError("Cannot derive an output path component")
    return component


def materialize_submission_substrate(plan: dict, output_root: Path) -> dict:
    """Write one immutable clean/doped substrate package from a resolved plan."""

    if plan.get("mode") != "public_submission_plan":
        raise ValueError("Substrate materialization requires a public submission plan")
    protonation = plan.get("inputs", {}).get("sam_structure", {}).get(
        "protonation", {}
    )
    if protonation.get("input_state") == "already_deprotonated_h0":
        raise ValueError(
            "An already-deprotonated H0 input is limited to the read-only CPU "
            "geometry dry run and cannot be materialized"
        )
    substrate = plan["substrate"]
    source = substrate["source"]
    source_path = Path(source["path"])
    if _sha256(source_path) != source["sha256"]:
        raise ValueError("Substrate source changed after planning")

    identity_payload = {
        "workflow": plan.get("workflow"),
        "sam": plan["inputs"]["sam_structure"]["sha256"],
        "catalog": substrate["catalog"]["sha256"],
        "source": source["sha256"],
        "surface": substrate["surface"],
        "supercell": substrate["supercell"],
        "slab": substrate["slab"],
        "doping": substrate["doping"],
        "adsorption": {
            "site_instance_protocol": substrate["adsorption"][
                "site_instance_protocol"
            ],
            "anchor_family": substrate["adsorption"]["anchor_family"],
            "site_library_sha256": substrate["adsorption"]["site_library"][
                "sha256"
            ],
            "site_library_promotion_fingerprint": substrate["adsorption"]
            ["site_library"]["promotion_fingerprint"],
            "conflict_policy": substrate["adsorption"]["conflict_policy"],
            "selection": substrate["adsorption"]["selection"],
        },
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    sam_name = _path_component(Path(plan["inputs"]["sam_structure"]["path"]).stem)
    substrate_name = _path_component(substrate["key"].casefold())
    output_root = Path(output_root).expanduser().resolve()
    package_directory = (
        output_root
        / f"{sam_name}-{substrate_name}"
        / f"substrate-{fingerprint[:12]}"
    )
    if package_directory.exists():
        raise ValueError(f"Immutable substrate package already exists: {package_directory}")

    substrate_build = _build_doped_submission_supercell(
        read(source_path),
        np.asarray(substrate["supercell"]["matrix"], dtype=int),
        substrate["doping"],
        substrate["slab"],
        substrate["surface"]["normal_axis"],
        substrate["doping"]["layer_dopant_counts_bottom_to_top"],
    )
    clean = substrate_build["clean"]
    doped = substrate_build["doped"]
    expected_atoms = substrate["slab"]["full_supercell_atom_count_before_doping"]
    if len(clean) != expected_atoms:
        raise ValueError(f"Constructed substrate has {len(clean)} atoms; expected {expected_atoms}")
    host_element = substrate["doping"]["host_element"]
    normal_axis = substrate["surface"]["normal_axis"]
    structural_layers = substrate_build["structural_layers"]
    host_layers = substrate_build["host_layers"]
    expected_host_counts = substrate["doping"][
        "layer_host_site_counts_bottom_to_top"
    ]
    if [len(layer) for layer in host_layers] != expected_host_counts:
        raise ValueError("Constructed substrate host-layer counts do not match the plan")
    expected_layer_counts = substrate["doping"][
        "layer_atom_counts_bottom_to_top"
    ]
    if [len(layer) for layer in structural_layers] != expected_layer_counts:
        raise ValueError("Constructed substrate structural-layer counts do not match the plan")

    selected_by_layer = substrate_build["selected_by_layer"]
    layer_minimum_distances = substrate_build["layer_minimum_distances_A"]
    selected_indices = substrate_build["selected_indices"]
    if len(selected_indices) != substrate["doping"]["dopant_count"]:
        raise ValueError("Selected dopant count does not match the plan")
    doped_symbols = doped.get_chemical_symbols()

    catalog_path = Path(substrate["catalog"]["path"])
    if _sha256(catalog_path) != substrate["catalog"]["sha256"]:
        raise ValueError("Substrate catalog changed after planning")
    recipe = _load_catalog_file(
        catalog_path,
        anchor_family=substrate["adsorption"]["anchor_family"],
    )
    site_record, site_library, site_library_path = (
        _load_accepted_site_prototype_library(recipe)
    )
    site_instances = _expand_site_prototypes_to_supercell(
        read(source_path),
        clean,
        np.asarray(substrate["supercell"]["matrix"], dtype=int),
        site_library,
    )
    metal_constraints, conflict_edges = _site_instance_conflicts(site_instances)
    site_selection = _solve_site_instance_selection(
        site_instances,
        metal_constraints,
        substrate["supercell"]["planned_sam_count"],
    )
    planned_selection = substrate["adsorption"]["selection"]
    if site_selection != planned_selection:
        raise ValueError("Site Instance selection changed after planning")
    selected_site_ids = set(site_selection["selected_site_instance_ids"])
    for instance in site_instances:
        instance["selected"] = instance["site_instance_id"] in selected_site_ids
        for metal in instance["actual_metals"]:
            atom_index = metal["atom_id_1based"] - 1
            metal["actual_element_after_doping"] = doped_symbols[atom_index]

    original_cell = np.asarray(clean.cell, dtype=float)
    normal_vector = original_cell[normal_axis]
    unit_normal = normal_vector / np.linalg.norm(normal_vector)
    normal_shift = float(
        substrate["slab"]["adsorption_vacuum"][
            "substrate_shift_along_normal_A"
        ]
    )
    target_cell = np.asarray(substrate["supercell"]["cell_A"], dtype=float)
    translation = normal_shift * unit_normal
    if normal_shift:
        clean.positions += translation
        doped.positions += translation
    clean.set_cell(target_cell, scale_atoms=False)
    doped.set_cell(target_cell, scale_atoms=False)
    inverse_target_cell = np.linalg.inv(target_cell)
    target_pbc = np.asarray(clean.pbc, dtype=bool)
    for instance in site_instances:
        cartesian = np.asarray(
            instance["ideal_site_location"]["cartesian_A"], dtype=float
        ) + translation
        fractional = cartesian @ inverse_target_cell
        fractional[target_pbc] -= np.floor(fractional[target_pbc])
        instance["ideal_site_location"]["cartesian_A"] = cartesian.tolist()
        instance["ideal_site_location"]["fractional_xyz"] = fractional.tolist()

    coordinates = np.asarray(clean.positions)[:, normal_axis]
    layer_centers = [float(np.mean(coordinates[layer])) for layer in host_layers]
    layer_atom_ids = [
        [atom_id + 1 for atom_id in layer] for layer in structural_layers
    ]
    restore_layer_count = (
        substrate["slab"]["target_layers"] - substrate["slab"]["working_layers"]
    )
    restore_atom_ids = [
        atom_id
        for layer in layer_atom_ids[:restore_layer_count]
        for atom_id in layer
    ]
    working_atom_ids = [
        atom_id
        for layer in layer_atom_ids[restore_layer_count:]
        for atom_id in layer
    ]
    top_coordinate = float(np.max(coordinates))
    substitutions = []
    for layer_number, indices in enumerate(selected_by_layer, 1):
        for index in indices:
            substitutions.append(
                {
                    "atom_id_1based": index + 1,
                    "from": host_element,
                    "to": substrate["doping"]["dopant_element"],
                    "layer_bottom_to_top": layer_number,
                    "coordinate_A": float(coordinates[index]),
                    "depth_from_top_A": top_coordinate - float(coordinates[index]),
                    "position_A": np.asarray(clean.positions[index]).tolist(),
                }
            )

    layer_groups = {
        "schema_version": 1,
        "normal_axis": normal_axis,
        "partition": substrate["doping"]["layer_partition"],
        "layers_bottom_to_top": [
            {
                "layer": layer + 1,
                "host_center_A": layer_centers[layer],
                "atom_count": len(atom_ids),
                "atom_ids_1based": atom_ids,
            }
            for layer, atom_ids in enumerate(layer_atom_ids)
        ],
        "restore_layer_atom_ids_1based": restore_atom_ids,
        "working_layer_atom_ids_1based": working_atom_ids,
    }
    adsorption_sites = {
        "schema_version": 1,
        "status": (
            "passed_single_molecule_default_site_ranking"
            if plan.get("workflow") == "single_molecule_adsorption"
            else "passed_fixed_coverage_site_selection"
        ),
        "workflow": plan.get("workflow"),
        "site_source": substrate["adsorption"]["site_source"],
        "site_instance_protocol": substrate["adsorption"][
            "site_instance_protocol"
        ],
        "anchor_family": substrate["adsorption"]["anchor_family"],
        "site_library": {
            "path": str(site_library_path),
            "sha256": site_record["sha256"],
            "promotion_fingerprint": site_record["promotion_fingerprint"],
            "prototype_count": len(site_library["site_prototypes"]),
        },
        "energy_policy": {
            "doped_site_energy_policy": "reuse_undoped_parent",
            "energy_source": "undoped_parent_approximation",
            "surface_protons_retained_in_parent_adsorption_energy": True,
            "surface_protons_in_integer_program": False,
        },
        "conflict_policy": substrate["adsorption"]["conflict_policy"],
        "candidate_instance_count": len(site_instances),
        "metal_constraint_count": len(metal_constraints),
        "conflict_edge_count": len(conflict_edges),
        "selection": site_selection,
        "site_instances": site_instances,
        "metal_constraints": metal_constraints,
        "conflict_edges": conflict_edges,
    }

    package_directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        dir=package_directory.parent, prefix=f".{package_directory.name}-"
    ) as temporary_directory:
        temporary = Path(temporary_directory)
        source_copy = temporary / f"source-structure{source_path.suffix.lower()}"
        source_copy.write_bytes(source_path.read_bytes())
        clean_path = temporary / "clean-surface-slab.cif"
        doped_path = temporary / "doped-surface-slab.cif"
        layer_path = temporary / "layer-groups.json"
        adsorption_path = temporary / "adsorption-sites.json"
        manifest_path = temporary / "substrate-build-manifest.json"
        write_clean_structure(clean, clean_path)
        write_clean_structure(doped, doped_path)
        write_manifest(layer_path, layer_groups)
        write_manifest(adsorption_path, adsorption_sites)
        manifest = {
            "schema_version": 1,
            "status": "passed_substrate_and_site_materialization",
            "workflow": plan.get("workflow"),
            "fingerprint": fingerprint,
            "output_directory": str(package_directory),
            "source": source,
            "surface": substrate["surface"],
            "supercell": substrate["supercell"],
            "slab": substrate["slab"],
            "doping": {
                **substrate["doping"],
                "layer_realized_minimum_separation_A": layer_minimum_distances,
                "substitutions": substitutions,
            },
            "composition": {
                "clean": _formula(clean.get_chemical_symbols()),
                "doped": _formula(doped.get_chemical_symbols()),
            },
            "artifacts": [
                {
                    "role": "source_structure",
                    "path": source_copy.name,
                    "sha256": _sha256(source_copy),
                },
                {
                    "role": "clean_surface_slab",
                    "path": clean_path.name,
                    "sha256": _sha256(clean_path),
                },
                {
                    "role": "doped_surface_slab",
                    "path": doped_path.name,
                    "sha256": _sha256(doped_path),
                },
                {
                    "role": "layer_groups",
                    "path": layer_path.name,
                    "sha256": _sha256(layer_path),
                },
                {
                    "role": "adsorption_site_instances_and_selection",
                    "path": adsorption_path.name,
                    "sha256": _sha256(adsorption_path),
                },
            ],
        }
        write_manifest(manifest_path, manifest)
        temporary.replace(package_directory)
    return manifest


def _hard_added_atom_collision_count(
    new_positions,
    new_symbols,
    existing_positions,
    existing_symbols,
    cell,
    periodic_axes,
) -> int:
    if not len(existing_positions):
        return 0
    new_positions = np.asarray(new_positions, dtype=float)
    existing_positions = np.asarray(existing_positions, dtype=float)
    delta = new_positions[:, None, :] - existing_positions[None, :, :]
    partial_pbc = np.zeros(3, dtype=bool)
    partial_pbc[list(periodic_axes)] = True
    _, distances = find_mic(
        delta.reshape((-1, 3)), cell=np.asarray(cell, dtype=float), pbc=partial_pbc
    )
    distances = np.asarray(distances, dtype=float).reshape(delta.shape[:2])
    new_h = np.asarray(new_symbols, dtype=str)[:, None] == "H"
    existing_h = np.asarray(existing_symbols, dtype=str)[None, :] == "H"
    thresholds = np.where(
        new_h & existing_h,
        1.2,
        np.where(new_h | existing_h, 1.4, 1.8),
    )
    return int(np.count_nonzero(distances < thresholds))


def _outward_normal_translation_values(
    mapped_bond_window_A: tuple[float, float],
    step_A: float = DENSE_H0_NORMAL_TRANSLATION_STEP_A,
) -> list[float]:
    """Enumerate deterministic outward clearance poses within the bond window span."""

    minimum, maximum = (float(value) for value in mapped_bond_window_A)
    step_A = float(step_A)
    if not 0.0 < minimum <= maximum or not math.isfinite(minimum + maximum):
        raise ValueError("mapped bond window must be finite, positive, and ordered")
    if not math.isfinite(step_A) or step_A <= 0.0:
        raise ValueError("normal translation step must be finite and positive")
    maximum_translation = maximum - minimum
    step_count = int(math.floor(maximum_translation / step_A + 1.0e-10))
    return [round(index * step_A, 10) for index in range(step_count + 1)]


def _validated_dense_h0_pose_search(
    recipe: dict,
    anchor_family: str,
    *,
    required: bool,
) -> dict | None:
    """Return one family-authorized dense-H0 pose policy or fail closed."""

    policy = recipe["adsorption"].get("dense_h0_pose_search")
    if policy is None:
        if required:
            raise ValueError(
                f"dense H0 pose search is not registered for substrate {recipe['key']}"
            )
        return None
    if not isinstance(policy, dict):
        raise ValueError("dense H0 pose search registration must be a table")
    expected = {
        "protocol_version": "outward_normal_clearance_scan_v1",
        "translation_direction": "outward_surface_normal",
        "maximum_translation_origin": (
            "final_metal_oxygen_distance_window_width"
        ),
        "candidate_retention": (
            "all_compatible_translations_per_donor_mapping"
        ),
    }
    for name, expected_value in expected.items():
        if policy.get(name) != expected_value:
            raise ValueError(
                f"unsupported dense H0 pose search {name}: {policy.get(name)!r}"
            )
    families = policy.get("anchor_families")
    if (
        not isinstance(families, list)
        or not families
        or any(not isinstance(value, str) or not value for value in families)
        or len(set(families)) != len(families)
    ):
        raise ValueError(
            "dense H0 pose search anchor_families must be a non-empty unique list"
        )
    if anchor_family not in families:
        raise ValueError(
            f"dense H0 pose search is not registered for anchor family {anchor_family!r}"
        )
    step_A = policy.get("normal_translation_step_A")
    if (
        isinstance(step_A, bool)
        or not isinstance(step_A, (int, float))
        or not math.isfinite(float(step_A))
        or float(step_A) <= 0.0
    ):
        raise ValueError(
            "dense H0 pose search normal_translation_step_A must be finite and positive"
        )
    return {
        **expected,
        "anchor_families": list(families),
        "normal_translation_step_A": float(step_A),
    }


def _validated_pose_outward_normal(
    surface_frame: dict,
    *,
    reference_outward_normal,
) -> np.ndarray:
    """Use the registered Surface Frame normal after rejecting reversed frames."""

    registered = np.asarray(
        surface_frame.get("outward_normal_cartesian_unit"), dtype=float
    )
    reference = np.asarray(reference_outward_normal, dtype=float)
    if registered.shape != (3,) or reference.shape != (3,):
        raise ValueError("pose-search outward normals must have three components")
    registered_norm = float(np.linalg.norm(registered))
    reference_norm = float(np.linalg.norm(reference))
    if (
        not np.all(np.isfinite(registered))
        or not np.all(np.isfinite(reference))
        or registered_norm <= 1.0e-12
        or reference_norm <= 1.0e-12
    ):
        raise ValueError("pose-search outward normals must be finite and nonzero")
    registered /= registered_norm
    reference /= reference_norm
    if float(np.dot(registered, reference)) <= 0.0:
        raise ValueError("registered Surface Frame outward normal opposes slab outward normal")
    return registered


def _solve_pose_candidate_selection(
    candidates: list[dict],
    *,
    target_count: int,
    cell,
) -> tuple[list[dict], dict]:
    """Globally maximize collision-free pose count, then preserve motif geometry."""

    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_matrix, vstack
    except ImportError as exc:
        raise RuntimeError("Dense H0 pose selection requires scipy.optimize.milp") from exc
    if not candidates:
        raise ValueError("dense H0 pose search found no substrate-compatible candidate")
    if isinstance(target_count, bool) or int(target_count) != target_count or target_count <= 0:
        raise ValueError("dense H0 target count must be a positive integer")
    target_count = int(target_count)

    groups: list[list[int]] = []
    candidates_by_site: dict[str, list[int]] = {}
    candidates_by_metal: dict[int, list[int]] = {}
    for index, candidate in enumerate(candidates):
        candidates_by_site.setdefault(candidate["site_instance_id"], []).append(index)
        for metal_id in candidate["actual_metal_atom_ids_1based"]:
            candidates_by_metal.setdefault(int(metal_id), []).append(index)
    groups.extend(
        indices for indices in candidates_by_site.values() if len(indices) > 1
    )
    groups.extend(
        sorted(set(indices))
        for indices in candidates_by_metal.values()
        if len(set(indices)) > 1
    )

    collision_edges = []
    tested_nonstatic_candidate_pairs = 0
    for left, right in combinations(range(len(candidates)), 2):
        first = candidates[left]
        second = candidates[right]
        if first["site_instance_id"] == second["site_instance_id"]:
            continue
        if set(first["actual_metal_atom_ids_1based"]) & set(
            second["actual_metal_atom_ids_1based"]
        ):
            continue
        tested_nonstatic_candidate_pairs += 1
        if _hard_added_atom_collision_count(
            first["h0_positions"],
            first["h0_symbols"],
            second["h0_positions"],
            second["h0_symbols"],
            cell,
            first["periodic_fractional_axes"],
        ):
            collision_edges.append((left, right))
    groups.extend([left, right] for left, right in collision_edges)

    rows = []
    columns = []
    for row, group in enumerate(groups):
        for column in group:
            rows.append(row)
            columns.append(column)
    matrix = coo_matrix(
        (np.ones(len(rows)), (rows, columns)),
        shape=(len(groups), len(candidates)),
    ).tocsr()
    bounds = Bounds(np.zeros(len(candidates)), np.ones(len(candidates)))
    integrality = np.ones(len(candidates), dtype=int)
    capacity = LinearConstraint(
        matrix,
        np.full(len(groups), -np.inf),
        np.ones(len(groups)),
    )
    cardinality_result = milp(
        c=-np.ones(len(candidates)),
        integrality=integrality,
        bounds=bounds,
        constraints=capacity,
        options={"mip_rel_gap": 0.0},
    )
    if not cardinality_result.success or cardinality_result.x is None:
        raise ValueError(
            "Maximum-cardinality dense H0 pose optimization failed: "
            + cardinality_result.message
        )
    maximum_count = int(round(float(np.sum(cardinality_result.x))))
    selected_count = min(target_count, maximum_count)

    cardinality_row = coo_matrix(np.ones((1, len(candidates))))
    fixed_matrix = vstack([matrix, cardinality_row]).tocsr()
    lower = np.r_[np.full(len(groups), -np.inf), selected_count]
    upper = np.r_[np.ones(len(groups)), selected_count]
    shift_steps = np.asarray(
        [int(candidate["normal_translation_step_index"]) for candidate in candidates],
        dtype=float,
    )
    shift_result = milp(
        c=shift_steps,
        integrality=integrality,
        bounds=bounds,
        constraints=LinearConstraint(fixed_matrix, lower, upper),
        options={"mip_rel_gap": 0.0},
    )
    if not shift_result.success or shift_result.x is None:
        raise ValueError(
            "Minimum-translation dense H0 pose optimization failed: "
            + shift_result.message
        )
    minimum_shift_steps = int(round(float(shift_steps @ shift_result.x)))

    shift_row = coo_matrix(shift_steps.reshape((1, -1)))
    energy_matrix = vstack([fixed_matrix, shift_row]).tocsr()
    energy_lower = np.r_[lower, minimum_shift_steps]
    energy_upper = np.r_[upper, minimum_shift_steps]
    site_energies = np.asarray(
        [float(candidate["site_energy_eV"]) for candidate in candidates]
    )
    if not np.all(np.isfinite(site_energies)):
        raise ValueError("dense H0 site energies must be finite")
    scaled_site_energies = site_energies / DENSE_H0_SITE_ENERGY_QUANTUM_EV
    if np.any(np.abs(scaled_site_energies) > np.iinfo(np.int64).max):
        raise ValueError("dense H0 site energies exceed the integer objective range")
    energy_units = np.rint(scaled_site_energies).astype(np.int64)
    energy_offset_units = energy_units - int(np.min(energy_units))
    nonzero_units = [
        abs(int(value)) for value in energy_offset_units if int(value) != 0
    ]
    energy_unit_gcd = math.gcd(*nonzero_units) if nonzero_units else 1
    energy_objective = energy_offset_units.astype(float) / energy_unit_gcd
    energy_result = milp(
        c=energy_objective,
        integrality=integrality,
        bounds=bounds,
        constraints=LinearConstraint(energy_matrix, energy_lower, energy_upper),
        options={"mip_rel_gap": 0.0},
    )
    if not energy_result.success or energy_result.x is None:
        raise ValueError(
            "Minimum-energy dense H0 pose optimization failed: "
            + energy_result.message
        )
    selected = [
        candidate
        for candidate, value in zip(candidates, energy_result.x)
        if value > 0.5
    ]
    selected.sort(
        key=lambda candidate: (
            candidate["site_energy_eV"],
            candidate["site_instance_id"],
            candidate["normal_translation_A"],
            candidate["candidate_id"],
        )
    )
    return selected, {
        "solver": "scipy.optimize.milp_highs",
        "objective": (
            "maximum_cardinality_then_minimum_total_outward_translation_steps_"
            "then_minimum_inherited_site_energy"
        ),
        "global_optimality_scope": "enumerated_substrate_compatible_pose_candidates",
        "candidate_count": len(candidates),
        "surviving_site_instance_count": len(candidates_by_site),
        "tested_nonstatic_candidate_pair_count": tested_nonstatic_candidate_pairs,
        "sam_sam_collision_edge_count": len(collision_edges),
        "maximum_cardinality": maximum_count,
        "target_count": target_count,
        "selected_count": len(selected),
        "target_reached": len(selected) == target_count,
        "minimum_total_normal_translation_steps": minimum_shift_steps,
        "minimum_total_site_energy_eV": float(site_energies @ energy_result.x),
        "site_energy_quantum_eV": DENSE_H0_SITE_ENERGY_QUANTUM_EV,
        "minimum_total_site_energy_units": int(
            round(float(energy_units.astype(float) @ energy_result.x))
        ),
        "cardinality_mip_gap": float(getattr(cardinality_result, "mip_gap", 0.0)),
        "translation_mip_gap": float(getattr(shift_result, "mip_gap", 0.0)),
        "energy_mip_gap": float(getattr(energy_result, "mip_gap", 0.0)),
    }


def materialize_sized_monolayer(
    plan: dict,
    output_root: Path,
    approved_plan_sha256: str,
) -> dict:
    """Build a deterministic collision-screened unrelaxed sized H0 monolayer."""

    if (
        plan.get("mode") != "public_submission_plan"
        or plan.get("workflow") != "dense_monolayer"
        or plan.get("surface_size_request") is None
    ):
        raise ValueError("sized monolayer execution requires a user-sized dense plan")
    approval_contract = plan.get("approval_contract")
    if not isinstance(approval_contract, dict):
        raise ValueError("sized monolayer plan has no approval contract")
    if approval_contract.get("hash_scope") != "approval_payload_canonical_json":
        raise ValueError("unsupported sized monolayer approval hash scope")
    approval_payload = approval_contract.get("approval_payload")
    actual_payload_sha256 = _approval_payload_sha256(approval_payload)
    expected_approval = approval_contract.get("plan_sha256")
    if actual_payload_sha256 != expected_approval:
        raise ValueError("approval payload hash does not match the resolved plan")
    if approved_plan_sha256 != expected_approval:
        raise ValueError("approval hash does not match the resolved plan")
    actual_execution_plan_sha256 = _canonical_json_sha256(
        _dense_h0_execution_plan_identity(plan)
    )
    if approval_payload.get(
        "dense_h0_execution_plan_sha256"
    ) != actual_execution_plan_sha256:
        raise ValueError(
            "approved plan field 'dense_h0_execution_plan_sha256' does not match "
            "the executable plan"
        )

    sam_record = plan["inputs"]["sam_structure"]
    substrate = plan["substrate"]
    approved_plan_fields = {
        "sam_sha256": sam_record["sha256"],
        "substrate_catalog_sha256": substrate["catalog"]["sha256"],
        "substrate_source_sha256": substrate["source"]["sha256"],
        "substrate_structure_manifest_sha256": substrate["source"][
            "structure_manifest"
        ]["sha256"],
        "workflow": plan["workflow"],
        "supercell_matrix": substrate["supercell"]["matrix"],
        "target_lateral_size_nm": plan["surface_size_request"][
            "requested_lateral_size_nm"
        ],
        "planned_sam_count": plan["capacity_estimate"]["estimated_sam_count"],
        "packing_fraction": plan["capacity_estimate"]["packing_fraction"],
        "dopant_site_fraction": substrate["doping"]["target_site_fraction"],
        "anchor_family": substrate["adsorption"]["anchor_family"],
        "dense_h0_pose_search": substrate["adsorption"]["dense_h0_pose_search"],
        "site_library_sha256": substrate["adsorption"]["site_library"][
            "sha256"
        ],
        "site_library_promotion_fingerprint": substrate["adsorption"][
            "site_library"
        ]["promotion_fingerprint"],
    }
    for field, plan_value in approved_plan_fields.items():
        if approval_payload.get(field) != plan_value:
            raise ValueError(
                f"approved plan field {field!r} does not match approval payload"
            )
    if (
        int(substrate["supercell"]["planned_sam_count"])
        != approval_payload["planned_sam_count"]
    ):
        raise ValueError(
            "approved plan field 'planned_sam_count' conflicts with supercell plan"
        )
    if approval_payload.get("planner_implementation_sha256") != _sha256(
        Path(__file__).resolve()
    ):
        raise ValueError(
            "planner implementation changed after the sized monolayer plan was resolved"
        )
    if approval_payload.get("collision_audit_implementation_sha256") != _sha256(
        COLLISION_AUDIT_IMPLEMENTATION_PATH
    ):
        raise ValueError(
            "collision audit implementation changed after the sized monolayer plan "
            "was resolved"
        )
    if approval_payload.get("interface_validator_implementation_sha256") != _sha256(
        INTERFACE_VALIDATOR_IMPLEMENTATION_PATH
    ):
        raise ValueError(
            "interface validator implementation changed after the sized monolayer "
            "plan was resolved"
        )
    if approval_payload.get("surface_protonation_implementation_sha256") != _sha256(
        SURFACE_PROTONATION_IMPLEMENTATION_PATH
    ):
        raise ValueError(
            "surface protonation implementation changed after the sized monolayer "
            "plan was resolved"
        )
    collision_scale = approval_payload.get("formal_collision_radius_scale")
    if collision_scale is None:
        raise ValueError(
            "sized monolayer execution requires a formal collision radius scale"
        )
    if sam_record["protonation"]["input_state"] not in {
        "neutral_acid",
        "already_deprotonated_h0",
    }:
        raise ValueError("sized monolayer execution requires a supported canonical H0 input")
    sam_path = Path(sam_record["path"])
    if _sha256(sam_path) != sam_record["sha256"]:
        raise ValueError("SAM structure changed after planning")

    catalog_path = Path(substrate["catalog"]["path"])
    if _sha256(catalog_path) != substrate["catalog"]["sha256"]:
        raise ValueError("substrate catalog changed after planning")
    recipe = _load_catalog_file(
        catalog_path,
        anchor_family=substrate["adsorption"]["anchor_family"],
    )
    pose_search_policy = _validated_dense_h0_pose_search(
        recipe,
        substrate["adsorption"]["anchor_family"],
        required=True,
    )
    if pose_search_policy != approval_payload["dense_h0_pose_search"]:
        raise ValueError("dense H0 pose search registration changed after planning")
    structure_manifest_path = (
        catalog_path.parent / recipe["source"]["structure_manifest"]
    ).resolve()
    if _sha256(structure_manifest_path) != approval_payload.get(
        "substrate_structure_manifest_sha256"
    ):
        raise ValueError("substrate structure manifest changed after planning")
    source_path = Path(substrate["source"]["path"])
    if _sha256(source_path) != substrate["source"]["sha256"]:
        raise ValueError("substrate source changed after planning")
    source_atoms = read(source_path)
    matrix = np.asarray(substrate["supercell"]["matrix"], dtype=int)
    doping = _resolve_public_doping_request(
        recipe, float(substrate["doping"]["target_site_fraction"])
    )
    substrate_build = _build_doped_submission_supercell(
        source_atoms,
        matrix,
        doping,
        recipe["slab"],
        int(recipe["surface"]["normal_axis"]),
        list(substrate["doping"]["layer_dopant_counts_bottom_to_top"]),
    )
    clean = substrate_build["clean"]
    doped = substrate_build["doped"]
    target_cell = np.asarray(substrate["supercell"]["cell_A"], dtype=float)
    normal_axis = int(substrate["surface"]["normal_axis"])
    normal = np.asarray(source_atoms.cell[normal_axis], dtype=float)
    unit_normal = normal / np.linalg.norm(normal)
    substrate_shift = float(
        substrate["slab"]["adsorption_vacuum"][
            "substrate_shift_along_normal_A"
        ]
    ) * unit_normal
    clean.positions += substrate_shift
    doped.positions += substrate_shift
    clean.set_cell(target_cell, scale_atoms=False)
    doped.set_cell(target_cell, scale_atoms=False)

    site_record, site_library, site_library_path = (
        _load_accepted_site_prototype_library(recipe)
    )
    if (
        site_record["sha256"] != approval_payload["site_library_sha256"]
        or site_record["promotion_fingerprint"]
        != approval_payload["site_library_promotion_fingerprint"]
    ):
        raise ValueError("accepted site library changed after planning")
    unshifted_clean = make_supercell(source_atoms, matrix, wrap=True)
    instances = _expand_site_prototypes_to_supercell(
        source_atoms, unshifted_clean, matrix, site_library
    )
    instances.sort(key=lambda item: (item["site_energy_eV"], item["site_instance_id"]))
    prototype_by_id = {
        prototype["site_prototype_id"]: prototype
        for prototype in site_library["site_prototypes"]
    }

    sam_atoms = read(sam_path)
    sam_symbols = sam_atoms.get_chemical_symbols()
    sam_positions = np.asarray(sam_atoms.positions, dtype=float)
    sam_cell = np.asarray(sam_atoms.cell, dtype=float)
    sam_pbc = np.asarray(sam_atoms.pbc, dtype=bool)
    adjacency = _molecular_adjacency(sam_symbols, sam_positions, sam_cell, sam_pbc)
    anchor_index = int(sam_record["anchor"]["atom_id_1based"]) - 1
    unwrapped = _unwrap_molecule(
        sam_positions, adjacency, sam_cell, sam_pbc, anchor_index
    )
    donor_labels = list(sam_record["anchor"]["donor_labels"])
    donor_indices = [
        int(atom_id) - 1
        for atom_id in sam_record["anchor"]["donor_atom_ids_1based"]
    ]
    released = {
        int(atom_id) - 1
        for atom_id in sam_record["released_hydrogen_atom_ids_1based"]
    }
    h0_indices = [index for index in range(len(sam_atoms)) if index not in released]
    h0_symbols = [sam_symbols[index] for index in h0_indices]
    h0_local_id_by_source_index = {
        source_index: local_index + 1
        for local_index, source_index in enumerate(h0_indices)
    }
    substrate_symbols = doped.get_chemical_symbols()
    radii = {
        symbol: float(vdw_radii[atomic_numbers[symbol]])
        for symbol in sorted(set(substrate_symbols + h0_symbols))
    }
    bond_window = tuple(
        float(value)
        for value in recipe["adsorption"]["site_discovery"]["probe_relaxation"]
        ["validation"]["final_metal_oxygen_distance_A"]
    )
    target_count = int(approval_payload["planned_sam_count"])
    normal_translation_step_A = pose_search_policy["normal_translation_step_A"]
    normal_translations = _outward_normal_translation_values(
        bond_window,
        normal_translation_step_A,
    )
    pose_search_method = (
        "registered_motif_fit_plus_all_compatible_outward_"
        "normal_clearance_scan"
    )
    rejection_counts = Counter()
    pose_candidates = []
    donor_mapping_count = 0
    evaluated_pose_candidate_count = 0
    for instance in instances:
        template = instance["placement_template"]
        frame = template["surface_frame"]
        pose_unit_normal = _validated_pose_outward_normal(
            frame,
            reference_outward_normal=unit_normal,
        )
        template_donor_labels = [
            item["probe_donor_id"]
            for item in template["binding_donor_positions"]
        ]
        target_by_label = {
            item["probe_donor_id"]: np.asarray(item["cartesian_A"], dtype=float)
            + substrate_shift
            for item in template["binding_donor_positions"]
        }
        metal_by_label = {
            item["probe_donor_id"]: int(item["metal_atom_id_1based"])
            for item in instance["donor_metal_mapping"]
        }
        for donor_permutation in permutations(
            donor_indices, len(template_donor_labels)
        ):
            donor_mapping_count += 1
            donor_by_label = dict(
                zip(template_donor_labels, donor_permutation)
            )
            source_points = np.vstack(
                [
                    unwrapped[anchor_index],
                    *[
                        unwrapped[donor_by_label[label]]
                        for label in template_donor_labels
                    ],
                ]
            )
            target_points = np.vstack(
                [
                    np.asarray(template["anchor_cartesian_A"], dtype=float)
                    + substrate_shift,
                    *[
                        target_by_label[label]
                        for label in template_donor_labels
                    ],
                ]
            )
            rotation, translation, rmsd = rigid_transform(
                source_points, target_points
            )
            placed = unwrapped @ rotation + translation
            mapped_windows = {
                (donor_by_label[label] + 1, metal_by_label[label]): bond_window
                for label in template_donor_labels
            }
            donor_assignment = {
                label: donor_by_label[label] + 1
                for label in template_donor_labels
            }
            mapping_id = "--".join(
                f"{label}-atom-{donor_assignment[label]:04d}"
                for label in template_donor_labels
            )
            mapping_passed = False
            for translation_step, normal_translation_A in enumerate(
                normal_translations
            ):
                evaluated_pose_candidate_count += 1
                h0_positions = (
                    placed[h0_indices]
                    + float(normal_translation_A) * pose_unit_normal
                )
                substrate_audit = periodic_vdw_collision_audit(
                    molecule_positions=h0_positions,
                    molecule_symbols=h0_symbols,
                    molecule_atom_ids=[index + 1 for index in h0_indices],
                    substrate_positions=doped.positions,
                    substrate_symbols=substrate_symbols,
                    substrate_atom_ids=list(range(1, len(doped) + 1)),
                    cell=target_cell,
                    periodic_axes=frame["periodic_fractional_axes"],
                    surface_frame=frame,
                    radii_A=radii,
                    radius_scale=float(collision_scale),
                    mapped_bond_windows_A=mapped_windows,
                )
                if not substrate_audit["passed"]:
                    rejection_counts["pose_candidate_sam_substrate_collision"] += 1
                    continue
                mapping_passed = True
                mapped_distances = [
                    float(record["distance_A"])
                    for record in substrate_audit["mapped_bonds"]
                ]
                pose_candidates.append(
                    {
                        "candidate_id": (
                            f"{instance['site_instance_id']}--donor-map--{mapping_id}"
                            f"--normal-step-{translation_step:03d}"
                        ),
                        "site_instance_id": instance["site_instance_id"],
                        "parent_site_prototype_id": instance[
                            "parent_site_prototype_id"
                        ],
                        "site_energy_eV": float(instance["site_energy_eV"]),
                        "actual_metal_atom_ids_1based": sorted(
                            set(instance["actual_metal_atom_ids_1based"])
                        ),
                        "periodic_fractional_axes": tuple(
                            frame["periodic_fractional_axes"]
                        ),
                        "h0_positions": h0_positions,
                        "h0_symbols": h0_symbols,
                        "donor_assignment_1based": donor_assignment,
                        "alignment_rmsd_A": float(rmsd),
                        "normal_translation_A": float(normal_translation_A),
                        "normal_translation_step_index": translation_step,
                        "mapped_metal_oxygen_distance_range_A": [
                            min(mapped_distances),
                            max(mapped_distances),
                        ],
                        "mapped_bonds": [
                            {
                                "sam_atom_id_1based": donor_by_label[label] + 1,
                                "sam_h0_atom_id_1based": (
                                    h0_local_id_by_source_index[
                                        donor_by_label[label]
                                    ]
                                ),
                                "substrate_metal_atom_id_1based": metal_by_label[
                                    label
                                ],
                            }
                            for label in template_donor_labels
                        ],
                    }
                )
            if not mapping_passed:
                rejection_counts[
                    "donor_mapping_without_substrate_compatible_pose"
                ] += 1

    if not pose_candidates:
        raise ValueError(
            "coverage-first dense H0 search found no collision-free placement; "
            f"rejections={dict(sorted(rejection_counts.items()))}"
        )
    selected_candidates, pose_selection = _solve_pose_candidate_selection(
        pose_candidates,
        target_count=target_count,
        cell=target_cell,
    )
    accepted_h0 = []
    placements = []
    used_metals = set()
    for candidate in selected_candidates:
        h0 = sam_atoms[h0_indices]
        h0.positions = candidate["h0_positions"]
        h0.set_cell(target_cell)
        h0.set_pbc(doped.pbc)
        accepted_h0.append(h0)
        used_metals.update(candidate["actual_metal_atom_ids_1based"])
        placements.append(
            {
                "placement_number": len(placements) + 1,
                "candidate_id": candidate["candidate_id"],
                "site_instance_id": candidate["site_instance_id"],
                "parent_site_prototype_id": candidate[
                    "parent_site_prototype_id"
                ],
                "site_energy_eV": candidate["site_energy_eV"],
                "donor_assignment_1based": candidate[
                    "donor_assignment_1based"
                ],
                "alignment_rmsd_A": candidate["alignment_rmsd_A"],
                "normal_translation_A": candidate["normal_translation_A"],
                "mapped_metal_oxygen_distance_range_A": candidate[
                    "mapped_metal_oxygen_distance_range_A"
                ],
                "mapped_bonds": candidate["mapped_bonds"],
            }
        )
    if not placements:
        raise ValueError(
            "coverage-first dense H0 search found no collision-free placement; "
            f"rejections={dict(sorted(rejection_counts.items()))}"
        )

    sam_layer = sum(accepted_h0[1:], accepted_h0[0].copy())
    final_atoms = doped + sam_layer
    final_atoms.set_cell(target_cell)
    final_atoms.set_pbc(doped.pbc)
    released_protons_per_sam = int(
        substrate["adsorption"]["released_protons_per_sam"]
    )
    protonation_requirements = {
        "schema": "samflow-surface-proton-requirements-v1",
        "schema_version": 1,
        "status": "pending_global_assignment_after_physical_interface_promotion",
        "chemistry_state": "H0_surface_protons_zero",
        "assignment_policy": "global_solver_after_physical_interface_promotion",
        "released_protons_per_sam": released_protons_per_sam,
        "sam_count": len(placements),
        "required_surface_proton_count": (
            len(placements) * released_protons_per_sam
        ),
        "fixed_parent_assignments": [],
        "site_prototype_proton_assignments_are_binding": False,
    }
    registered_interface_bonds = [
        {
            "placement_number": int(placement["placement_number"]),
            "site_instance_id": placement["site_instance_id"],
            "substrate_atom_index_0based": int(
                bond["substrate_metal_atom_id_1based"]
            )
            - 1,
            "sam_atom_index_0based": len(doped)
            + (int(placement["placement_number"]) - 1) * len(h0_indices)
            + int(bond["sam_h0_atom_id_1based"])
            - 1,
            "minimum_distance_A": bond_window[0],
            "maximum_distance_A": bond_window[1],
            "expected_substrate_element": substrate_symbols[
                int(bond["substrate_metal_atom_id_1based"]) - 1
            ],
            "expected_sam_element": h0_symbols[
                int(bond["sam_h0_atom_id_1based"]) - 1
            ],
        }
        for placement in placements
        for bond in placement["mapped_bonds"]
    ]
    molecule_formula = _formula(h0_symbols)
    substrate_formula = _formula(substrate_symbols)
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "plan_sha256": expected_approval,
                "algorithm": "coverage_first_global_pose_selection_v1",
                "site_library_sha256": site_record["sha256"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    sam_name = _path_component(sam_path.stem)
    substrate_name = _path_component(substrate["key"].casefold())
    output_root = Path(output_root).expanduser().resolve()
    package_directory = (
        output_root
        / f"{sam_name}-{substrate_name}"
        / f"sized-monolayer-{fingerprint[:12]}"
    )
    if package_directory.exists():
        raise ValueError(f"Immutable sized monolayer already exists: {package_directory}")
    package_directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        dir=package_directory.parent, prefix=f".{package_directory.name}-"
    ) as temporary_directory:
        temporary = Path(temporary_directory)
        final_name = "sized-monolayer-h0.extxyz"
        write(temporary / final_name, final_atoms)
        write(temporary / "doped-substrate.cif", doped)
        write_manifest(temporary / "placements.json", placements)
        protonation_requirements_name = "surface-proton-requirements.json"
        write_manifest(
            temporary / protonation_requirements_name, protonation_requirements
        )
        manifest = {
            "schema_version": 2,
            "status": (
                "built_requested_estimated_count"
                if len(placements) == target_count
                else "jammed_below_estimated_count"
            ),
            "algorithm": "coverage_first_global_pose_selection_v1",
            "algorithm_scope": (
                "global_maximum_cardinality_over_enumerated_pose_candidates"
            ),
            "approval": {
                "plan_sha256": expected_approval,
                "planner_implementation_sha256": approval_payload[
                    "planner_implementation_sha256"
                ],
                "collision_audit_implementation_sha256": approval_payload[
                    "collision_audit_implementation_sha256"
                ],
                "interface_validator_implementation_sha256": approval_payload[
                    "interface_validator_implementation_sha256"
                ],
                "surface_protonation_implementation_sha256": approval_payload[
                    "surface_protonation_implementation_sha256"
                ],
            },
            "output_directory": str(package_directory),
            "requested_lateral_size_nm": plan["surface_size_request"][
                "requested_lateral_size_nm"
            ],
            "realized_lateral_size_nm": plan["surface_size_request"][
                "realized_lateral_size_nm"
            ],
            "estimated_sam_count": target_count,
            "actual_sam_count": len(placements),
            "substrate_atom_count": len(doped),
            "atoms_per_sam_h0": len(h0_indices),
            "molecule_formula_h0": molecule_formula,
            "substrate_formula": substrate_formula,
            "chemistry_state": "H0_surface_protons_zero",
            "surface_proton_count": 0,
            "planned_surface_proton_count": protonation_requirements[
                "required_surface_proton_count"
            ],
            "sam_h0_atom_count": len(sam_layer),
            "final_atom_count": len(final_atoms),
            "final_structure": final_name,
            "final_structure_sha256": _sha256(temporary / final_name),
            "rejection_counts": dict(sorted(rejection_counts.items())),
            "pose_search": {
                "method": pose_search_method,
                "protocol_version": pose_search_policy["protocol_version"],
                "authorized_anchor_families": pose_search_policy[
                    "anchor_families"
                ],
                "normal_translation_step_A": normal_translation_step_A,
                "normal_translation_values_A": normal_translations,
                "maximum_translation_origin": pose_search_policy[
                    "maximum_translation_origin"
                ],
                "candidate_retention": pose_search_policy["candidate_retention"],
                "donor_mapping_count": donor_mapping_count,
                "evaluated_pose_candidate_count": evaluated_pose_candidate_count,
                "substrate_compatible_pose_candidate_count": len(pose_candidates),
                "mapped_bond_window_A": list(bond_window),
                "formal_collision_radius_scale": float(collision_scale),
                "collision_thresholds_unchanged": True,
            },
            "selection": pose_selection,
            "protonation_requirements": {
                "path": protonation_requirements_name,
                "sha256": _sha256(temporary / protonation_requirements_name),
                "status": protonation_requirements["status"],
            },
            "interface_validation_contract": {
                "schema": "samflow-interface-validation-contract-v1",
                "schema_version": 1,
                "chemistry_state": "H0_surface_protons_zero",
                "anchor_family": substrate["adsorption"]["anchor_family"],
                "anchor_element": sam_symbols[anchor_index],
                "anchor_atom_index_within_sam_h0_0based": (
                    h0_local_id_by_source_index[anchor_index] - 1
                ),
                "donor_element": "O",
                "headgroup_oxygen_count": len(donor_indices),
                "interface_pair_policy": "exact",
                "surface_h_policy": "h0",
                "surface_periodic_axes": list(
                    instances[0]["placement_template"]["surface_frame"][
                        "periodic_fractional_axes"
                    ]
                ),
                "surface_normal_cartesian_unit": list(
                    instances[0]["placement_template"]["surface_frame"][
                        "outward_normal_cartesian_unit"
                    ]
                ),
                "collision_thresholds_A": {
                    "H-H": 1.2,
                    "H-heavy": 1.4,
                    "heavy-heavy": 1.8,
                },
                "registered_interface_bonds": registered_interface_bonds,
                "minimum_registered_contacts_per_sam": 1,
            },
            "validation": {
                "shared_metal_capacity_passed": len(used_metals)
                == sum(
                    len(set(instance_by_bond["substrate_metal_atom_id_1based"] for instance_by_bond in placement["mapped_bonds"]))
                    for placement in placements
                ),
                "sam_sam_collision_passed": True,
                "formal_sam_substrate_collision_passed": True,
                "proton_inventory_declared": protonation_requirements[
                    "required_surface_proton_count"
                ]
                == len(placements) * released_protons_per_sam,
            },
            "promotion_eligible": False,
            "promotion_blocker": (
                "Unrelaxed globally selected H0 requires independent physical-interface validation, "
                "protonation review, and force-based relaxation before promotion."
            ),
        }
        write_manifest(temporary / "manifest.json", manifest)
        temporary.replace(package_directory)
    return manifest


def promote_sized_monolayer_h0(
    parent_manifest_path: Path,
    output_root: Path,
    *,
    approved_plan_sha256: str,
) -> dict:
    """Independently validate and seal physical-interface promotion for a sized H0."""

    parent_manifest_path = Path(parent_manifest_path).expanduser().resolve()
    try:
        parent = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("cannot read sized monolayer parent manifest") from exc
    if parent.get("schema_version") != 2:
        raise ValueError("physical promotion requires sized monolayer manifest schema 2")
    if parent.get("chemistry_state") != "H0_surface_protons_zero":
        raise ValueError("physical promotion requires a true H0 parent")
    if parent.get("surface_proton_count") != 0:
        raise ValueError("physical promotion rejects a parent containing surface protons")
    if parent.get("promotion_eligible") is not False:
        raise ValueError("physical promotion requires an unpromoted sized H0 parent")
    parent_approval_record = parent.get("approval", {})
    parent_approval = parent_approval_record.get("plan_sha256")
    if parent_approval != approved_plan_sha256:
        raise ValueError("approval hash does not match the sized H0 parent")
    implementation_identity_keys = (
        "planner_implementation_sha256",
        "collision_audit_implementation_sha256",
        "interface_validator_implementation_sha256",
        "surface_protonation_implementation_sha256",
    )
    if any(
        not isinstance(parent_approval_record.get(key), str)
        or re.fullmatch(r"[0-9a-f]{64}", parent_approval_record[key]) is None
        for key in implementation_identity_keys
    ):
        raise ValueError("sized H0 parent has incomplete implementation identities")
    if parent_approval_record.get(
        "interface_validator_implementation_sha256"
    ) != _sha256(INTERFACE_VALIDATOR_IMPLEMENTATION_PATH):
        raise ValueError(
            "interface validator implementation changed after the sized monolayer "
            "plan was resolved"
        )

    parent_directory = parent_manifest_path.parent
    parent_structure = parent_directory / str(parent.get("final_structure", ""))
    if not parent_structure.is_file():
        raise ValueError("sized H0 parent structure is missing")
    if _sha256(parent_structure) != parent.get("final_structure_sha256"):
        raise ValueError("sized H0 parent structure hash mismatch")
    requirements = parent_directory / str(
        parent.get("protonation_requirements", {}).get("path", "")
    )
    if not requirements.is_file() or _sha256(requirements) != parent.get(
        "protonation_requirements", {}
    ).get("sha256"):
        raise ValueError("sized H0 protonation requirements hash mismatch")
    requirements_record = json.loads(requirements.read_text(encoding="utf-8"))
    if (
        requirements_record.get("schema")
        != "samflow-surface-proton-requirements-v1"
        or requirements_record.get("fixed_parent_assignments") != []
    ):
        raise ValueError("sized H0 protonation requirements are not globally assigned")

    contract = parent.get("interface_validation_contract")
    if not isinstance(contract, dict) or contract.get("schema") != (
        "samflow-interface-validation-contract-v1"
    ):
        raise ValueError("sized H0 parent lacks a supported interface validation contract")
    registered_bonds = contract.get("registered_interface_bonds")
    if not isinstance(registered_bonds, list) or not registered_bonds:
        raise ValueError("interface validation contract has no registered bonds")
    thresholds = contract.get("collision_thresholds_A", {})
    if set(thresholds) != {"H-H", "H-heavy", "heavy-heavy"}:
        raise ValueError("interface validation contract collision thresholds are incomplete")

    parent_manifest_sha256 = _sha256(parent_manifest_path)
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "parent_manifest_sha256": parent_manifest_sha256,
                "parent_structure_sha256": parent["final_structure_sha256"],
                "contract": contract,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    output_root = Path(output_root).expanduser().resolve()
    package_directory = output_root / f"physical-interface-promotion-{fingerprint[:12]}"
    if package_directory.exists():
        raise ValueError(f"Immutable physical promotion already exists: {package_directory}")

    atoms = read(parent_structure)
    substrate_atom_count = int(parent["substrate_atom_count"])
    molecule_count = int(parent["actual_sam_count"])
    atoms_per_molecule = int(parent["atoms_per_sam_h0"])
    if len(atoms) != substrate_atom_count + molecule_count * atoms_per_molecule:
        raise ValueError("sized H0 parent atom layout does not match its manifest")
    validation_config = ValidationConfig(
        substrate_atoms=substrate_atom_count,
        molecules=molecule_count,
        atoms_per_molecule=atoms_per_molecule,
        protons_per_molecule=0,
        anchor_element=str(contract["anchor_element"]),
        anchor_atom_index_within_molecule_0based=int(
            contract["anchor_atom_index_within_sam_h0_0based"]
        ),
        parent_element="O",
        acceptor_element="O",
        hh_min=float(thresholds["H-H"]),
        h_heavy_min=float(thresholds["H-heavy"]),
        heavy_heavy_min=float(thresholds["heavy-heavy"]),
        expected_substrate_formula=dict(parent["substrate_formula"]),
        expected_molecule_formula=dict(parent["molecule_formula_h0"]),
        surface_h_policy="h0",
        interface_pair_policy="exact",
        allowed_interface_pairs=(),
        registered_interface_bonds=tuple(registered_bonds),
        surface_periodic_axes=tuple(contract["surface_periodic_axes"]),
    )
    validation = validate_monolayer(parent_structure, validation_config)
    exact_windows_pass = all(
        record.get("window_passed") is True
        for record in validation.get("registered_interface_bonds", [])
    )
    per_sam_counts = Counter(
        (int(record["sam_atom_index_0based"]) - substrate_atom_count)
        // atoms_per_molecule
        + 1
        for record in registered_bonds
    )
    minimum_contacts = int(contract["minimum_registered_contacts_per_sam"])
    every_sam_registered = (
        len(per_sam_counts) == molecule_count
        and min(per_sam_counts.values(), default=0) >= minimum_contacts
    )
    physical_interface_pass = bool(
        validation["passed"] and exact_windows_pass and every_sam_registered
    )
    validation["physical_interface_gate"] = {
        "required": True,
        "generic_validator_passed": bool(validation["passed"]),
        "all_registered_windows_passed": exact_windows_pass,
        "every_sam_has_minimum_registered_contacts": every_sam_registered,
        "minimum_registered_contacts_per_sam": minimum_contacts,
        "physical_interface_pass": physical_interface_pass,
    }

    package_directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        dir=package_directory.parent, prefix=f".{package_directory.name}-"
    ) as temporary_directory:
        temporary = Path(temporary_directory)
        promoted_structure_name = "promoted-h0.extxyz"
        shutil.copy2(parent_structure, temporary / promoted_structure_name)
        validation_name = "physical-interface-validation.json"
        write_manifest(temporary / validation_name, validation)
        manifest = {
            "schema": "samflow-sized-h0-physical-interface-promotion-v1",
            "schema_version": 1,
            "status": (
                "passed_physical_interface_promotion"
                if physical_interface_pass
                else "failed_physical_interface_promotion"
            ),
            "output_directory": str(package_directory),
            "chemistry_state": "H0_surface_protons_zero",
            "physical_interface_pass": physical_interface_pass,
            "promotion_eligible": physical_interface_pass,
            "parent": {
                "manifest_path": str(parent_manifest_path),
                "manifest_sha256": parent_manifest_sha256,
                "structure_path": str(parent_structure),
                "structure_sha256": parent["final_structure_sha256"],
                "approval_plan_sha256": parent_approval,
                "approval_implementation_identities": {
                    key: parent_approval_record[key]
                    for key in implementation_identity_keys
                },
            },
            "structure": {
                "path": promoted_structure_name,
                "sha256": _sha256(temporary / promoted_structure_name),
                "atom_count": len(atoms),
            },
            "interface_validation_contract": contract,
            "validation": {
                "path": validation_name,
                "sha256": _sha256(temporary / validation_name),
                "implementation_sha256": parent_approval_record[
                    "interface_validator_implementation_sha256"
                ],
            },
            "next_stage": (
                "surface_protonation"
                if physical_interface_pass
                else "blocked_failed_physical_interface_gate"
            ),
        }
        write_manifest(temporary / "manifest.json", manifest)
        temporary.replace(package_directory)
    return manifest


def promote_geometric_h0(
    input_structure_path: Path,
    output_root: Path,
    *,
    substrate_atoms: int,
    molecules: int,
    atoms_per_sam_h0: int,
    released_protons_per_sam: int = 2,
    surface_metal_elements: tuple[str, ...] = ("In", "Sn"),
    surface_metal_depth_A: float = 0.7,
    interface_height_clearance_A: float = 1.0,
    surface_normal: tuple[float, float, float] = (0.0, 0.0, 1.0),
    surface_periodic_axes: tuple[int, int] = (0, 1),
    anchor_element: str = "P",
    triangle_anchor_molecule_numbers: tuple[int, ...] | None = None,
) -> dict:
    """Independently validate H0 using the phosphonate-triangle/height contract.

    This is the promotion path for complete H0 structures whose interface was
    generated geometrically and therefore has no pre-registered O--metal pair map.
    It does not modify the input structure or waive SAM--SAM collision checks.
    """

    input_structure_path = Path(input_structure_path).expanduser().resolve()
    if not input_structure_path.is_file():
        raise ValueError("geometric H0 input structure is missing")
    integer_inputs = {
        "substrate_atoms": substrate_atoms,
        "molecules": molecules,
        "atoms_per_sam_h0": atoms_per_sam_h0,
        "released_protons_per_sam": released_protons_per_sam,
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in integer_inputs.values()
    ):
        raise ValueError("geometric H0 layout and proton counts must be positive integers")
    surface_metal_elements = tuple(str(item) for item in surface_metal_elements)
    if not surface_metal_elements or any(
        item not in atomic_numbers for item in surface_metal_elements
    ):
        raise ValueError("surface_metal_elements must contain known chemical elements")
    atoms = read(input_structure_path)
    symbols = np.asarray(atoms.get_chemical_symbols())
    expected_atom_count = substrate_atoms + molecules * atoms_per_sam_h0
    if len(atoms) != expected_atom_count:
        raise ValueError(
            "geometric H0 atom layout mismatch: "
            f"found {len(atoms)}, expected {expected_atom_count}"
        )
    substrate_symbols = symbols[:substrate_atoms]
    if np.any(substrate_symbols == "H"):
        raise ValueError("geometric H0 input contains hydrogen in the substrate block")
    anchor_local_ids = []
    for molecule_number in range(molecules):
        start = substrate_atoms + molecule_number * atoms_per_sam_h0
        block_symbols = symbols[start : start + atoms_per_sam_h0]
        local = np.flatnonzero(block_symbols == anchor_element)
        if len(local) != 1:
            raise ValueError(
                f"SAM {molecule_number + 1} has {len(local)} {anchor_element} anchors"
            )
        anchor_local_ids.append(int(local[0]))
    if len(set(anchor_local_ids)) != 1:
        raise ValueError("geometric H0 SAM blocks do not share one anchor atom offset")
    anchor_local_index = anchor_local_ids[0]

    substrate_formula = dict(sorted(Counter(substrate_symbols).items()))
    molecule_formula = dict(
        sorted(
            Counter(
                symbols[substrate_atoms : substrate_atoms + atoms_per_sam_h0]
            ).items()
        )
    )
    validation_config = ValidationConfig(
        substrate_atoms=substrate_atoms,
        molecules=molecules,
        atoms_per_molecule=atoms_per_sam_h0,
        protons_per_molecule=0,
        anchor_element=anchor_element,
        anchor_atom_index_within_molecule_0based=anchor_local_index,
        expected_substrate_formula=substrate_formula,
        expected_molecule_formula=molecule_formula,
        surface_h_policy="h0",
        interface_pair_policy="phosphonate_triangle_height",
        mobile_interface_metal_elements=surface_metal_elements,
        surface_metal_depth_A=float(surface_metal_depth_A),
        interface_height_clearance_A=float(interface_height_clearance_A),
        surface_normal=tuple(float(value) for value in surface_normal),
        surface_periodic_axes=tuple(surface_periodic_axes),
        triangle_anchor_molecule_numbers=(
            tuple(int(value) for value in triangle_anchor_molecule_numbers)
            if triangle_anchor_molecule_numbers is not None
            else None
        ),
    )
    validation = validate_monolayer(input_structure_path, validation_config)
    physical_interface_pass = bool(validation["passed"])
    interface_contract = {
        "schema": "samflow-phosphonate-o-triangle-height-contract-v1",
        "anchor_element": anchor_element,
        "anchor_atom_index_within_sam_h0_0based": anchor_local_index,
        "anchor_oxygen_count": 3,
        "surface_metal_elements": list(surface_metal_elements),
        "surface_metal_depth_A": float(surface_metal_depth_A),
        "minimum_non_anchor_height_A": float(interface_height_clearance_A),
        "surface_normal_cartesian_unit": list(validation[
            "phosphonate_triangle_height_interface"
        ]["surface_normal_cartesian_unit"]),
        "surface_periodic_axes": list(surface_periodic_axes),
        "triangle_anchor_molecule_numbers": (
            list(triangle_anchor_molecule_numbers)
            if triangle_anchor_molecule_numbers is not None
            else list(range(1, molecules + 1))
        ),
        "exact_oxygen_metal_pair_required": False,
    }
    requirements = {
        "schema": "samflow-surface-proton-requirements-v1",
        "assignment_policy": "global_solver_after_physical_interface_promotion",
        "released_protons_per_sam": released_protons_per_sam,
        "required_surface_proton_count": molecules * released_protons_per_sam,
        "fixed_parent_assignments": [],
    }
    geometry_record = {
        "substrate_atom_count": substrate_atoms,
        "molecule_count": molecules,
        "atoms_per_sam_h0": atoms_per_sam_h0,
        "substrate_formula": substrate_formula,
        "molecule_formula_h0": molecule_formula,
        "surface_proton_count": 0,
    }
    validator_sha256 = _sha256(INTERFACE_VALIDATOR_IMPLEMENTATION_PATH)
    protonation_sha256 = _sha256(SURFACE_PROTONATION_IMPLEMENTATION_PATH)
    input_sha256 = _sha256(input_structure_path)
    basis_payload = {
        "source_structure_sha256": input_sha256,
        "interface_contract": interface_contract,
        "geometry": geometry_record,
        "protonation_requirements": requirements,
        "validator_implementation_sha256": validator_sha256,
        "surface_protonation_implementation_sha256": protonation_sha256,
    }
    promotion_basis_sha256 = hashlib.sha256(
        json.dumps(basis_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output_root = Path(output_root).expanduser().resolve()
    package_directory = output_root / (
        f"geometric-h0-physical-interface-promotion-{promotion_basis_sha256[:12]}"
    )
    if package_directory.exists():
        raise ValueError(f"Immutable geometric H0 promotion already exists: {package_directory}")

    package_directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        dir=package_directory.parent, prefix=f".{package_directory.name}-"
    ) as temporary_directory:
        temporary = Path(temporary_directory)
        structure_name = "promoted-h0.extxyz"
        shutil.copy2(input_structure_path, temporary / structure_name)
        requirements_name = "protonation-requirements.json"
        write_manifest(temporary / requirements_name, requirements)
        validation_name = "physical-interface-validation.json"
        validation["physical_interface_gate"] = {
            "required": True,
            "interface_policy": "phosphonate_o_triangle_plus_non_anchor_height",
            "physical_interface_pass": physical_interface_pass,
        }
        write_manifest(temporary / validation_name, validation)
        manifest = {
            "schema": "samflow-geometric-h0-physical-interface-promotion-v1",
            "schema_version": 1,
            "status": (
                "passed_physical_interface_promotion"
                if physical_interface_pass
                else "failed_physical_interface_promotion"
            ),
            "output_directory": str(package_directory),
            "chemistry_state": "H0_surface_protons_zero",
            "physical_interface_pass": physical_interface_pass,
            "promotion_eligible": physical_interface_pass,
            "promotion_basis_sha256": promotion_basis_sha256,
            "parent": {
                "source_structure_path": str(input_structure_path),
                "source_structure_sha256": input_sha256,
                "implementation_identities": {
                    "interface_validator_implementation_sha256": validator_sha256,
                    "surface_protonation_implementation_sha256": protonation_sha256,
                },
            },
            "geometry": geometry_record,
            "structure": {
                "path": structure_name,
                "sha256": _sha256(temporary / structure_name),
                "atom_count": len(atoms),
            },
            "interface_validation_contract": interface_contract,
            "protonation_requirements": {
                "path": requirements_name,
                "sha256": _sha256(temporary / requirements_name),
            },
            "validation": {
                "path": validation_name,
                "sha256": _sha256(temporary / validation_name),
                "implementation_sha256": validator_sha256,
            },
            "next_stage": (
                "surface_protonation"
                if physical_interface_pass
                else "blocked_failed_physical_interface_gate"
            ),
        }
        write_manifest(temporary / "manifest.json", manifest)
        temporary.replace(package_directory)
    return manifest


def materialize_submission_candidates(plan: dict, output_root: Path) -> dict:
    """Write balanced unrelaxed single-SAM starting structures selected by a plan."""

    if plan.get("mode") != "public_submission_plan" or plan.get("workflow") != (
        "single_molecule_adsorption"
    ):
        raise ValueError(
            "candidate materialization requires a single-molecule public submission plan"
        )
    selection = plan.get("single_adsorption_selection")
    if not selection:
        raise ValueError("candidate materialization requires a single-adsorption objective")
    candidate_ids = list(selection["selected_candidate_ids"])
    if selection["requires_relaxation"]:
        candidate_ids = list(selection["candidates_scheduled_for_relaxation"])
    if not candidate_ids:
        raise ValueError("candidate selection produced no structures to materialize")
    sam_record = plan["inputs"]["sam_structure"]
    if sam_record["protonation"]["input_state"] not in {
        "neutral_acid",
        "already_deprotonated_h0",
    }:
        raise ValueError("candidate materialization requires a supported canonical H0 input")
    sam_path = Path(sam_record["path"])
    if _sha256(sam_path) != sam_record["sha256"]:
        raise ValueError("SAM structure changed after planning")

    substrate_manifest = materialize_submission_substrate(plan, output_root)
    substrate_directory = Path(substrate_manifest["output_directory"])
    doped_artifact = next(
        artifact
        for artifact in substrate_manifest["artifacts"]
        if artifact["role"] == "doped_surface_slab"
    )
    doped_path = substrate_directory / doped_artifact["path"]
    if _sha256(doped_path) != doped_artifact["sha256"]:
        raise ValueError("materialized doped substrate hash mismatch")
    doped = read(doped_path)

    substrate = plan["substrate"]
    catalog_path = Path(substrate["catalog"]["path"])
    if _sha256(catalog_path) != substrate["catalog"]["sha256"]:
        raise ValueError("substrate catalog changed after planning")
    recipe = _load_catalog_file(
        catalog_path,
        anchor_family=substrate["adsorption"]["anchor_family"],
    )
    source_path = Path(substrate["source"]["path"])
    if _sha256(source_path) != substrate["source"]["sha256"]:
        raise ValueError("substrate source changed after planning")
    source_atoms = read(source_path)
    matrix = np.asarray(substrate["supercell"]["matrix"], dtype=int)
    clean_unshifted = make_supercell(source_atoms, matrix, wrap=True)
    site_record, site_library, site_library_path = (
        _load_accepted_site_prototype_library(recipe)
    )
    instances = _expand_site_prototypes_to_supercell(
        source_atoms, clean_unshifted, matrix, site_library
    )
    instance_by_id = {instance["site_instance_id"]: instance for instance in instances}
    prototype_by_id = {
        prototype["site_prototype_id"]: prototype
        for prototype in site_library["site_prototypes"]
    }
    record_by_id = {
        record["candidate_id"]: record
        for record in plan["geometry_candidate_plan"]["candidate_records"]
    }
    if any(candidate_id not in record_by_id for candidate_id in candidate_ids):
        raise ValueError("selected candidate is absent from the planned geometry audit")

    sam_atoms = read(sam_path)
    sam_symbols = sam_atoms.get_chemical_symbols()
    sam_positions = np.asarray(sam_atoms.positions, dtype=float)
    sam_cell = np.asarray(sam_atoms.cell, dtype=float)
    sam_pbc = np.asarray(sam_atoms.pbc, dtype=bool)
    adjacency = _molecular_adjacency(sam_symbols, sam_positions, sam_cell, sam_pbc)
    anchor_index = int(sam_record["anchor"]["atom_id_1based"]) - 1
    unwrapped = _unwrap_molecule(
        sam_positions, adjacency, sam_cell, sam_pbc, anchor_index
    )
    released = {
        int(atom_id) - 1
        for atom_id in sam_record["released_hydrogen_atom_ids_1based"]
    }
    h0_indices = [index for index in range(len(sam_atoms)) if index not in released]
    donor_labels = list(sam_record["anchor"]["donor_labels"])
    target_cell = np.asarray(substrate["supercell"]["cell_A"], dtype=float)
    normal_axis = int(substrate["surface"]["normal_axis"])
    normal = np.asarray(source_atoms.cell[normal_axis], dtype=float)
    unit_normal = normal / np.linalg.norm(normal)
    substrate_shift = float(
        substrate["slab"]["adsorption_vacuum"][
            "substrate_shift_along_normal_A"
        ]
    ) * unit_normal
    source_cell = np.asarray(source_atoms.cell, dtype=float)
    source_fractional = np.asarray(source_atoms.get_scaled_positions(wrap=True))

    identity_payload = {
        "sam_sha256": sam_record["sha256"],
        "substrate_fingerprint": substrate_manifest["fingerprint"],
        "site_library_sha256": site_record["sha256"],
        "selection": selection,
        "candidate_ids": candidate_ids,
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    sam_name = _path_component(sam_path.stem)
    substrate_name = _path_component(substrate["key"].casefold())
    output_root = Path(output_root).expanduser().resolve()
    package_directory = (
        output_root
        / f"{sam_name}-{substrate_name}"
        / f"adsorption-candidates-{fingerprint[:12]}"
    )
    if package_directory.exists():
        raise ValueError(f"Immutable candidate package already exists: {package_directory}")

    candidate_artifacts = []
    package_directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        dir=package_directory.parent, prefix=f".{package_directory.name}-"
    ) as temporary_directory:
        temporary = Path(temporary_directory)
        for candidate_number, candidate_id in enumerate(candidate_ids, 1):
            record = record_by_id[candidate_id]
            if record["formal_collision_passed"] is not True:
                raise ValueError("refusing to materialize a failed collision candidate")
            instance = instance_by_id[record["site_instance_id"]]
            template = instance["placement_template"]
            target_by_label = {
                item["probe_donor_id"]: np.asarray(item["cartesian_A"], dtype=float)
                + substrate_shift
                for item in template["binding_donor_positions"]
            }
            donor_assignment = {
                label: int(atom_id) - 1
                for label, atom_id in record["donor_assignment_1based"].items()
            }
            source_points = np.vstack(
                [
                    unwrapped[anchor_index],
                    *[unwrapped[donor_assignment[label]] for label in donor_labels],
                ]
            )
            target_points = np.vstack(
                [
                    np.asarray(template["anchor_cartesian_A"], dtype=float)
                    + substrate_shift,
                    *[target_by_label[label] for label in donor_labels],
                ]
            )
            rotation, translation, alignment_rmsd = rigid_transform(
                source_points, target_points
            )
            placed_positions = unwrapped @ rotation + translation
            placed_h0 = sam_atoms[h0_indices]
            placed_h0.positions = placed_positions[h0_indices]
            placed_h0.set_cell(target_cell)
            placed_h0.set_pbc(doped.pbc)

            prototype = prototype_by_id[instance["parent_site_prototype_id"]]
            representative_record = prototype["representative_relaxed_structure"]
            representative_path = (
                site_library_path.parent / representative_record["path"]
            ).resolve()
            if _sha256(representative_path) != representative_record["sha256"]:
                raise ValueError("Site Prototype representative structure hash changed")
            representative = read(representative_path)
            proton_positions = []
            proton_parent_atom_ids = []
            for assignment in prototype["surface_proton_policy"][
                "selected_assignments"
            ]:
                parent_id = int(assignment["nearest_oxygen_atom_id_0_based"])
                hydrogen_id = int(assignment["surface_proton_atom_id_0_based"])
                vector = (
                    representative.positions[hydrogen_id]
                    - representative.positions[parent_id]
                )
                vector, _ = find_mic(
                    vector, representative.cell, representative.pbc
                )
                source_position = (
                    source_fractional[parent_id]
                    + np.asarray(instance["source_cell_translation"], dtype=int)
                ) @ source_cell
                target_parent = _map_periodic_vertex_to_supercell(
                    clean_unshifted, source_position, "O"
                )
                proton_positions.append(doped.positions[target_parent] + vector)
                proton_parent_atom_ids.append(target_parent + 1)
            expected_protons = int(
                substrate["adsorption"]["released_protons_per_sam"]
            )
            if len(proton_positions) != expected_protons:
                raise ValueError("Site Prototype proton inventory does not match the plan")
            surface_h = Atoms("H" * len(proton_positions), positions=proton_positions)
            surface_h.set_cell(target_cell)
            surface_h.set_pbc(doped.pbc)
            combined = doped + surface_h + placed_h0
            combined.set_cell(target_cell)
            combined.set_pbc(doped.pbc)
            structure_name = f"candidate-{candidate_number:04d}.extxyz"
            structure_path = temporary / structure_name
            write(structure_path, combined)
            candidate_artifacts.append(
                {
                    "candidate_id": candidate_id,
                    "structure": structure_name,
                    "sha256": _sha256(structure_path),
                    "atom_count": len(combined),
                    "substrate_atom_count": len(doped),
                    "surface_proton_count": len(surface_h),
                    "surface_proton_parent_atom_ids_1based": proton_parent_atom_ids,
                    "sam_h0_atom_count": len(placed_h0),
                    "formal_collision_passed": record["formal_collision_passed"],
                    "projected_footprint_area_A2": record[
                        "projected_footprint_area_A2"
                    ],
                    "alignment_rmsd_A": float(alignment_rmsd),
                }
            )
        manifest = {
            "schema_version": 1,
            "status": "passed_unrelaxed_adsorption_candidate_materialization",
            "objective": selection["objective"],
            "energy_selection_pending_relaxation": selection["requires_relaxation"],
            "uses_large_language_model": False,
            "fingerprint": fingerprint,
            "output_directory": str(package_directory),
            "substrate_package": str(substrate_directory),
            "candidate_count": len(candidate_artifacts),
            "candidates": candidate_artifacts,
        }
        write_manifest(temporary / "manifest.json", manifest)
        temporary.replace(package_directory)
    return manifest


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect maintained substrate structures without launching work."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    bulk_plan = subparsers.add_parser(
        "bulk-relaxation-plan",
        help="validate and print a read-only Bulk Parent relaxation plan",
    )
    bulk_run = subparsers.add_parser(
        "bulk-relaxation-run",
        help="run one immutable symmetry-preserving MACE bulk relaxation",
    )
    position_check = subparsers.add_parser(
        "bulk-parent-position-check",
        help="compare a registered Bulk Parent with experimental atomic positions",
    )
    position_check.add_argument(
        "--substrate", required=True, help="maintained substrate key"
    )
    position_check.add_argument(
        "--structure-directory",
        type=Path,
        help="override substrates/structures for testing or maintenance",
    )
    surface_plan = subparsers.add_parser(
        "surface-precut-plan",
        help="enumerate clean-surface terminations without writing structures",
    )
    surface_run = subparsers.add_parser(
        "surface-precut-run",
        help="write an immutable package of clean-surface termination candidates",
    )
    for surface_command in (surface_plan, surface_run):
        surface_command.add_argument(
            "--substrate", required=True, help="maintained substrate key"
        )
        surface_command.add_argument(
            "--structure-directory",
            type=Path,
            help="override substrates/structures for testing or maintenance",
        )
    surface_run.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="root for immutable Surface Precut enumeration packages",
    )
    relaxation_plan = subparsers.add_parser(
        "surface-relaxation-plan",
        help="validate a read-only symmetric-slab relaxation and convergence plan",
    )
    relaxation_run = subparsers.add_parser(
        "surface-relaxation-run",
        help="run one immutable MACE surface relaxation thickness series",
    )
    termination_plan = subparsers.add_parser(
        "surface-termination-screen-plan",
        help="plan all same-face exposed-layer relaxation and energy comparisons",
    )
    termination_run = subparsers.add_parser(
        "surface-termination-screen-run",
        help="run all same-face exposed-layer relaxation and energy comparisons",
    )
    for relaxation in (
        relaxation_plan,
        relaxation_run,
        termination_plan,
        termination_run,
    ):
        relaxation.add_argument(
            "--substrate", required=True, help="maintained substrate key"
        )
        relaxation.add_argument(
            "--model", required=True, type=Path, help="local MACE model file"
        )
        relaxation.add_argument(
            "--model-element",
            action="append",
            required=True,
            dest="model_elements",
            help="element explicitly declared as supported; repeat for each element",
        )
        relaxation.add_argument(
            "--default-dtype",
            choices=("float32", "float64"),
            default="float32",
            help="MACE calculator precision; must match the accepted Bulk Parent",
        )
        relaxation.add_argument(
            "--structure-directory",
            type=Path,
            help="override substrates/structures for testing or maintenance",
        )
    relaxation_run.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="root for immutable surface relaxation run directories",
    )
    relaxation_run.add_argument("--device", default="cuda", help="MACE device")
    termination_run.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="root for immutable surface termination screening runs",
    )
    termination_run.add_argument("--device", default="cuda", help="MACE device")
    promotion_plan = subparsers.add_parser(
        "surface-promotion-plan",
        help="validate a passed termination screen for maintained-surface promotion",
    )
    promotion_run = subparsers.add_parser(
        "surface-promotion-run",
        help="copy verified evidence and register a relaxed maintained surface",
    )
    for promotion in (promotion_plan, promotion_run):
        promotion.add_argument(
            "--substrate", required=True, help="maintained substrate key"
        )
        promotion.add_argument(
            "--screen-run",
            required=True,
            type=Path,
            help="passed immutable surface-termination-screen run directory",
        )
        promotion.add_argument(
            "--target-repeat",
            required=True,
            type=int,
            help="registered public thickness to promote",
        )
        promotion.add_argument(
            "--structure-directory",
            type=Path,
            help="override substrates/structures for testing or maintenance",
        )
    site_plan = subparsers.add_parser(
        "adsorption-site-enumeration-plan",
        help="enumerate clean-surface multidentate metal combinations without writing",
    )
    site_run = subparsers.add_parser(
        "adsorption-site-enumeration-run",
        help="write an immutable clean-surface site-combination package",
    )
    for site_command in (site_plan, site_run):
        site_command.add_argument(
            "--substrate", required=True, help="maintained substrate catalog key"
        )
        site_command.add_argument(
            "--catalog-directory",
            type=Path,
            help="override the substrate catalog directory",
        )
        site_command.add_argument(
            "--anchor-family",
            help="explicit adsorption anchor family (for example carboxylic_acid)",
        )
    site_run.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="root for immutable adsorption-site enumeration packages",
    )
    probe_plan = subparsers.add_parser(
        "adsorption-probe-preparation-plan",
        help="plan fully deprotonated probe and retained surface-H starting structures",
    )
    probe_run = subparsers.add_parser(
        "adsorption-probe-preparation-run",
        help="write immutable probe and retained surface-H starting structures",
    )
    for probe_command in (probe_plan, probe_run):
        probe_command.add_argument(
            "--substrate", required=True, help="maintained substrate catalog key"
        )
        probe_command.add_argument(
            "--enumeration-run",
            required=True,
            type=Path,
            help="reviewed immutable adsorption-site enumeration package",
        )
        probe_command.add_argument(
            "--catalog-directory",
            type=Path,
            help="override the substrate catalog directory",
        )
        probe_command.add_argument(
            "--anchor-family",
            help="explicit adsorption anchor family used by the enumeration package",
        )
    probe_run.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="root for immutable adsorption-probe preparation packages",
    )
    adsorption_relaxation_plan = subparsers.add_parser(
        "adsorption-probe-relaxation-plan",
        help="validate all probe/reference relaxations without starting MACE",
    )
    adsorption_relaxation_run = subparsers.add_parser(
        "adsorption-probe-relaxation-run",
        help="run immutable MACE probe/reference relaxations and validation",
    )
    for adsorption_relaxation in (
        adsorption_relaxation_plan,
        adsorption_relaxation_run,
    ):
        adsorption_relaxation.add_argument(
            "--substrate", required=True, help="maintained substrate catalog key"
        )
        adsorption_relaxation.add_argument(
            "--preparation-run",
            required=True,
            type=Path,
            help="passed immutable adsorption-probe preparation package",
        )
        adsorption_relaxation.add_argument(
            "--model", required=True, type=Path, help="local MACE model file"
        )
        adsorption_relaxation.add_argument(
            "--model-element",
            action="append",
            required=True,
            dest="model_elements",
            help="element explicitly declared as supported; repeat for each element",
        )
        adsorption_relaxation.add_argument(
            "--default-dtype",
            choices=("float32", "float64"),
            default="float32",
            help="MACE calculator precision; must match the accepted Bulk Parent",
        )
        adsorption_relaxation.add_argument(
            "--device", default="cuda", help="planned and executed MACE device"
        )
        adsorption_relaxation.add_argument(
            "--catalog-directory",
            type=Path,
            help="override the substrate catalog directory",
        )
        adsorption_relaxation.add_argument(
            "--structure-directory",
            type=Path,
            help="override substrates/structures for testing or maintenance",
        )
        adsorption_relaxation.add_argument(
            "--anchor-family",
            help="explicit adsorption anchor family used by the preparation package",
        )
    adsorption_relaxation_run.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="root for immutable adsorption-probe relaxation packages",
    )
    prototype_plan = subparsers.add_parser(
        "adsorption-site-prototype-plan",
        help="cluster selected relaxed probe minima into Site Prototypes without writing",
    )
    prototype_run = subparsers.add_parser(
        "adsorption-site-prototype-run",
        help="write an immutable Site Prototype candidate library",
    )
    for prototype_command in (prototype_plan, prototype_run):
        prototype_command.add_argument(
            "--substrate", required=True, help="maintained substrate catalog key"
        )
        prototype_command.add_argument(
            "--relaxation-run",
            required=True,
            type=Path,
            help="passed immutable adsorption-probe relaxation package",
        )
        prototype_command.add_argument(
            "--catalog-directory",
            type=Path,
            help="override the substrate catalog directory",
        )
        prototype_command.add_argument(
            "--anchor-family",
            help="explicit adsorption anchor family used by the relaxation package",
        )
    prototype_run.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="root for immutable Site Prototype candidate libraries",
    )
    prototype_promotion_plan = subparsers.add_parser(
        "adsorption-site-prototype-promotion-plan",
        help="validate a Site Prototype package for maintained-library promotion",
    )
    prototype_promotion_run = subparsers.add_parser(
        "adsorption-site-prototype-promotion-run",
        help="copy and register an accepted maintained Site Prototype library",
    )
    for promotion_command in (
        prototype_promotion_plan,
        prototype_promotion_run,
    ):
        promotion_command.add_argument(
            "--substrate", required=True, help="maintained substrate catalog key"
        )
        promotion_command.add_argument(
            "--prototype-run",
            required=True,
            type=Path,
            help="passed immutable Site Prototype clustering package",
        )
        promotion_command.add_argument(
            "--structure-directory",
            type=Path,
            help="override substrates/structures for testing or maintenance",
        )
        promotion_command.add_argument(
            "--anchor-family",
            help="explicit adsorption anchor family to promote",
        )
        promotion_command.add_argument(
            "--supersede-existing",
            action="store_true",
            help=(
                "retain and mark the current accepted library as superseded when "
                "registering a replacement"
            ),
        )
    for bulk in (bulk_plan, bulk_run):
        bulk.add_argument("--substrate", required=True, help="maintained substrate key")
        bulk.add_argument(
            "--model", required=True, type=Path, help="local MACE model file"
        )
        bulk.add_argument(
            "--model-element",
            action="append",
            required=True,
            dest="model_elements",
            help="element explicitly declared as supported; repeat for each element",
        )
        bulk.add_argument(
            "--structure-directory",
            type=Path,
            help="override substrates/structures for testing or maintenance",
        )
    bulk_run.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="root for immutable candidate run directories",
    )
    bulk_run.add_argument("--device", default="cuda", help="MACE device")
    bulk_run.add_argument(
        "--default-dtype",
        choices=("float32", "float64"),
        default="float32",
        help="MACE calculator precision",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_cli_parser()
    args = parser.parse_args(argv)
    if args.command in {"bulk-relaxation-plan", "bulk-relaxation-run"}:
        plan = resolve_bulk_relaxation_plan(
            args.substrate,
            args.model,
            args.model_elements,
            args.structure_directory,
        )
        if args.command == "bulk-relaxation-plan":
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0
        manifest = execute_bulk_relaxation(
            plan,
            args.output_root,
            device=args.device,
            default_dtype=args.default_dtype,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0 if manifest["status"] == "passed" else 1
    if args.command == "bulk-parent-position-check":
        report = resolve_bulk_parent_position_check(
            args.substrate,
            args.structure_directory,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["status"] == "passed" else 1
    if args.command in {"surface-precut-plan", "surface-precut-run"}:
        plan = resolve_surface_precut_plan(
            args.substrate,
            args.structure_directory,
        )
        if args.command == "surface-precut-plan":
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0 if plan["execution_contract"]["executable"] else 1
        manifest = execute_surface_precut_enumeration(plan, args.output_root)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0 if manifest["status"] == "passed_selection" else 1
    if args.command in {"surface-relaxation-plan", "surface-relaxation-run"}:
        plan = resolve_surface_relaxation_plan(
            args.substrate,
            args.model,
            args.model_elements,
            args.structure_directory,
            default_dtype=args.default_dtype,
        )
        if args.command == "surface-relaxation-plan":
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0 if plan["execution_contract"]["executable"] else 1
        manifest = execute_surface_relaxation(
            plan,
            args.output_root,
            device=args.device,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0 if manifest["status"] == "passed" else 1
    if args.command in {
        "surface-termination-screen-plan",
        "surface-termination-screen-run",
    }:
        plan = resolve_surface_termination_screen_plan(
            args.substrate,
            args.model,
            args.model_elements,
            args.structure_directory,
            default_dtype=args.default_dtype,
        )
        if args.command == "surface-termination-screen-plan":
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0 if plan["execution_contract"]["executable"] else 1
        manifest = execute_surface_termination_screen(
            plan,
            args.output_root,
            device=args.device,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0 if manifest["status"] == "passed_selection" else 1
    if args.command in {"surface-promotion-plan", "surface-promotion-run"}:
        plan = resolve_surface_promotion_plan(
            args.substrate,
            args.screen_run,
            args.target_repeat,
            args.structure_directory,
        )
        if args.command == "surface-promotion-plan":
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0 if plan["execution_contract"]["executable"] else 1
        report = execute_surface_promotion(plan)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["status"] == "passed_promotion" else 1
    if args.command in {
        "adsorption-site-enumeration-plan",
        "adsorption-site-enumeration-run",
    }:
        plan = resolve_adsorption_site_enumeration_plan(
            args.substrate,
            args.catalog_directory,
            anchor_family=args.anchor_family,
        )
        if args.command == "adsorption-site-enumeration-plan":
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0 if plan["execution_contract"]["executable"] else 1
        manifest = execute_adsorption_site_enumeration(plan, args.output_root)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0 if manifest["status"].startswith("passed_enumeration") else 1
    if args.command in {
        "adsorption-probe-preparation-plan",
        "adsorption-probe-preparation-run",
    }:
        plan = resolve_adsorption_probe_preparation_plan(
            args.substrate,
            args.enumeration_run,
            args.catalog_directory,
            anchor_family=args.anchor_family,
        )
        if args.command == "adsorption-probe-preparation-plan":
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0 if plan["execution_contract"]["executable"] else 1
        manifest = execute_adsorption_probe_preparation(plan, args.output_root)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0 if manifest["status"].startswith("passed_preparation") else 1
    if args.command in {
        "adsorption-probe-relaxation-plan",
        "adsorption-probe-relaxation-run",
    }:
        plan = resolve_adsorption_probe_relaxation_plan(
            args.substrate,
            args.preparation_run,
            args.model,
            args.model_elements,
            args.catalog_directory,
            args.structure_directory,
            default_dtype=args.default_dtype,
            device=args.device,
            anchor_family=args.anchor_family,
        )
        if args.command == "adsorption-probe-relaxation-plan":
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0 if plan["execution_contract"]["plan_resolved"] else 1
        manifest = execute_adsorption_probe_relaxation(
            plan, args.output_root, device=args.device
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0 if manifest["status"].startswith("passed") else 1
    if args.command in {
        "adsorption-site-prototype-plan",
        "adsorption-site-prototype-run",
    }:
        plan = resolve_adsorption_site_prototype_plan(
            args.substrate,
            args.relaxation_run,
            args.catalog_directory,
            anchor_family=args.anchor_family,
        )
        if args.command == "adsorption-site-prototype-plan":
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0 if plan["execution_contract"]["executable"] else 1
        manifest = execute_adsorption_site_prototype_clustering(
            plan, args.output_root
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0 if manifest["status"].startswith("passed_clustering") else 1
    if args.command in {
        "adsorption-site-prototype-promotion-plan",
        "adsorption-site-prototype-promotion-run",
    }:
        plan = resolve_adsorption_site_prototype_promotion_plan(
            args.substrate,
            args.prototype_run,
            args.structure_directory,
            anchor_family=args.anchor_family,
            supersede_existing=args.supersede_existing,
        )
        if args.command == "adsorption-site-prototype-promotion-plan":
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0 if plan["execution_contract"]["executable"] else 1
        report = execute_adsorption_site_prototype_promotion(plan)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["status"] == "passed_promotion" else 1
    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
