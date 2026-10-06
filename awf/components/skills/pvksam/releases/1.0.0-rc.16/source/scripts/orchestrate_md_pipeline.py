import os
import argparse
import time
import subprocess
import sys
import json
import hashlib
import tomllib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ase.data import atomic_numbers, covalent_radii

from sam_lammps import (
    frozen_atom_ids,
    minimization_summary,
    parse_species,
    read_frozen_atom_ids,
    read_typed_structure,
    run_lammps,
    validate_layout,
    write_manifest,
)
from stage0_relaxation import load_protonated_parent
from surface_protonation import protonate_promoted_h0
from substrate_library import (
    materialize_submission_candidates,
    materialize_sized_monolayer,
    materialize_submission_substrate,
    promote_geometric_h0,
    promote_sized_monolayer_h0,
    resolve_submission_plan,
)


BASE_PLAN_STAGES = (
    "input_validation",
    "stage0",
    "annealing",
    "production",
    "analysis",
)

RUN_STAGE_DIRECTORIES = {
    "input_validation": "30-validate",
    "stage0": "40-stage0",
    "annealing": "50-anneal",
    "production": "60-production",
    "layer_restoration": "70-restore",
}

EXECUTABLE_PROTOCOL = "preassembled_phosphonate_oxide_md_v1"


def _section(config: dict, name: str) -> dict:
    value = config.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Missing or invalid [{name}] section")
    return value


def _reject_unknown(section: dict, allowed: set[str], name: str) -> None:
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ValueError(f"Unknown {name} keys: {', '.join(unknown)}")


def load_run_description(path: Path) -> tuple[dict, Path]:
    path = Path(path).expanduser().resolve()
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    if config.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    _reject_unknown(
        config,
        {
            "schema_version",
            "system",
            "sam",
            "substrate",
            "interface",
            "assembly",
            "protocol",
            "executor",
            "run",
        },
        "top-level",
    )
    system = _section(config, "system")
    sam = _section(config, "sam")
    substrate = _section(config, "substrate")
    interface = _section(config, "interface")
    assembly = _section(config, "assembly")
    protocol = _section(config, "protocol")
    executor = _section(config, "executor")
    anchor = _section(sam, "anchor")
    marker = sam.get("marker")
    if marker is not None and not isinstance(marker, dict):
        raise ValueError("[sam.marker] must be a table")

    _reject_unknown(system, {"id", "structure"}, "system")
    _reject_unknown(sam, {"anchor", "marker"}, "sam")
    _reject_unknown(anchor, {"element", "neighbor_pattern"}, "sam.anchor")
    if marker is not None:
        _reject_unknown(marker, {"element", "neighbor_pattern"}, "sam.marker")
    _reject_unknown(
        substrate,
        {"surface_normal", "restore_layer", "expected_restore_atoms"},
        "substrate",
    )
    _reject_unknown(
        interface,
        {
            "protons_per_sam",
            "preproduction_surface_h_policy",
            "production_surface_h_policy",
            "allowed_contact_pairs",
            "protonation_manifest",
        },
        "interface",
    )
    _reject_unknown(
        assembly,
        {"molecule_count", "conformer_pool", "repeat"},
        "assembly",
    )
    _reject_unknown(
        protocol,
        {"name", "stage0", "annealing", "production"},
        "protocol",
    )
    _reject_unknown(
        executor,
        {"name", "species", "engine", "model", "lammps", "device"},
        "executor",
    )
    if "run" in config:
        _reject_unknown(
            config["run"],
            {"id", "output_root", "seed", "smoke", "frozen_atom_ids_file"},
            "run",
        )
    return config, path


def _required_string(section: dict, key: str, name: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name}.{key} must be a non-empty string")
    return value.strip()


def _positive_int(section: dict, key: str, name: str, allow_zero=False) -> int:
    value = section.get(key)
    minimum = 0 if allow_zero else 1
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name}.{key} must be a {qualifier} integer")
    return value


def _number(section: dict, key: str, name: str, allow_zero=False) -> float:
    value = section.get(key)
    valid_type = isinstance(value, (int, float)) and not isinstance(value, bool)
    valid_range = valid_type and (value >= 0 if allow_zero else value > 0)
    if not valid_range:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name}.{key} must be a {qualifier} number")
    return float(value)


def _boolean(section: dict, key: str, name: str) -> bool:
    value = section.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{name}.{key} must be true or false")
    return value


def _subsection(section: dict, key: str, name: str) -> dict:
    value = section.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing or invalid [{name}] section")
    return value


def _element_pattern(section: dict, name: str) -> tuple[str, tuple[str, ...]]:
    element = _required_string(section, "element", name)
    pattern = section.get("neighbor_pattern")
    if element not in atomic_numbers:
        raise ValueError(f"Unknown element in {name}: {element}")
    if not isinstance(pattern, list) or not pattern or not all(
        isinstance(item, str) and item in atomic_numbers for item in pattern
    ):
        raise ValueError(f"{name}.neighbor_pattern must be a non-empty element list")
    return element, tuple(sorted(pattern))


def _formula(symbols) -> dict[str, int]:
    return dict(sorted(Counter(symbols).items()))


def _infer_contiguous_layout(
    symbols: np.ndarray,
    molecule_count: int,
    anchor_element: str,
) -> tuple[int, int]:
    candidates = []
    for block_size in range(2, len(symbols) // molecule_count + 1):
        substrate_atoms = len(symbols) - molecule_count * block_size
        if substrate_atoms < 1:
            continue
        formulas = []
        valid = True
        for molecule in range(molecule_count):
            start = substrate_atoms + molecule * block_size
            block = symbols[start : start + block_size]
            if int(np.count_nonzero(block == anchor_element)) != 1:
                valid = False
                break
            formulas.append(_formula(block))
        if valid and all(formula == formulas[0] for formula in formulas[1:]):
            candidates.append((substrate_atoms, block_size))
    if len(candidates) != 1:
        raise ValueError(
            "Could not infer a unique substrate-first contiguous SAM layout; "
            f"candidate_count={len(candidates)}"
        )
    return candidates[0]


def _bonded_neighbor_symbols(atoms, block: np.ndarray, center: int) -> tuple[str, ...]:
    center_number = atomic_numbers[atoms[center].symbol]
    neighbors = []
    for index in block:
        index = int(index)
        if index == center:
            continue
        number = atomic_numbers[atoms[index].symbol]
        cutoff = 1.20 * (covalent_radii[center_number] + covalent_radii[number])
        if atoms.get_distance(center, index, mic=True) <= cutoff:
            neighbors.append(atoms[index].symbol)
    return tuple(sorted(neighbors))


def _validate_local_pattern(
    atoms,
    substrate_atoms: int,
    molecule_count: int,
    block_size: int,
    element: str,
    expected: tuple[str, ...],
    label: str,
) -> None:
    symbols = np.asarray(atoms.get_chemical_symbols())
    for molecule in range(molecule_count):
        start = substrate_atoms + molecule * block_size
        block = np.arange(start, start + block_size)
        matches = block[symbols[block] == element]
        if len(matches) != 1:
            raise ValueError(
                f"Molecule {molecule + 1} has {len(matches)} {label} element matches"
            )
        observed = _bonded_neighbor_symbols(atoms, block, int(matches[0]))
        if observed != expected:
            raise ValueError(
                f"Molecule {molecule + 1} {label} neighbor pattern is "
                f"{list(observed)}; expected {list(expected)}"
            )


def _infer_surface_frame(atoms, substrate_atoms: int) -> dict:
    cell = np.asarray(atoms.cell, dtype=float)
    positions = np.asarray(atoms.positions, dtype=float)
    candidates = []
    for axis in range(3):
        first = cell[(axis + 1) % 3]
        second = cell[(axis + 2) % 3]
        normal = np.cross(first, second)
        length = float(np.linalg.norm(normal))
        if length == 0.0:
            raise ValueError("Structure cell contains linearly dependent vectors")
        normal /= length
        spacing = abs(float(np.dot(cell[axis], normal)))
        projections = positions @ normal
        span = float(np.ptp(projections))
        candidates.append((spacing - span, axis, spacing, normal))
    vacuum_gap, axis, spacing, normal = max(candidates, key=lambda item: item[0])
    substrate_mean = float(np.mean(positions[:substrate_atoms] @ normal))
    sam_mean = float(np.mean(positions[substrate_atoms:] @ normal))
    if sam_mean < substrate_mean:
        normal = -normal
    separation = abs(sam_mean - substrate_mean)
    if separation <= 1.0e-6:
        raise ValueError("Cannot orient the surface normal from substrate and SAM positions")
    return {
        "method": "largest_cell_vacuum_gap",
        "cell_axis": axis,
        "outward_normal": [float(value) for value in normal],
        "cell_spacing_A": spacing,
        "estimated_vacuum_gap_A": vacuum_gap,
        "sam_substrate_centroid_separation_A": separation,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _path_component(value: str, name: str) -> str:
    value = value.strip()
    reserved = '<>:"/\\|?*'
    if (
        not value
        or Path(value).name != value
        or value in {".", ".."}
        or any(character in value for character in reserved)
    ):
        raise ValueError(f"{name} must be a single safe path component")
    return value


def _run_fingerprint(plan: dict) -> str:
    restore = plan["substrate"].get("restore_layer")
    identity = {
        "config": plan["config"]["sha256"],
        "structure": plan["structure"]["sha256"],
        "model": plan["executor"]["model"]["sha256"],
        "restore_layer": restore["sha256"] if restore else None,
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _execution_contract(
    protocol_name: str,
    engine: str,
    anchor_element: str,
    anchor_pattern: tuple[str, ...],
    marker: dict | None,
    surface_frame: dict,
) -> dict:
    issues = []
    if protocol_name != EXECUTABLE_PROTOCOL:
        issues.append(
            {
                "code": "protocol_not_implemented",
                "message": (
                    f"Requested protocol {protocol_name!r}; the only implemented "
                    f"config execution protocol is {EXECUTABLE_PROTOCOL!r}."
                ),
            }
        )
    if engine != "lammps-mliap":
        issues.append(
            {
                "code": "engine_not_lammps_mliap",
                "message": (
                    "Current config execution is implemented only for the "
                    "lammps-mliap engine."
                ),
            }
        )
    if anchor_element != "P" or anchor_pattern != ("C", "O", "O", "O"):
        issues.append(
            {
                "code": "anchor_not_current_phosphonate",
                "message": (
                    "Current execution requires one P anchor bonded locally to "
                    "one C and three O atoms."
                ),
            }
        )
    if marker is None or not marker["neighbor_pattern"] or set(
        marker["neighbor_pattern"]
    ) != {"C"}:
        issues.append(
            {
                "code": "marker_not_carbon_neighbor_countable",
                "message": (
                    "Current execution validation requires a marker whose local "
                    "neighbors are all carbon."
                ),
            }
        )
    normal = np.asarray(surface_frame["outward_normal"], dtype=float)
    if float(np.dot(normal, [0.0, 0.0, 1.0])) < 0.999:
        issues.append(
            {
                "code": "surface_frame_not_global_positive_z",
                "message": (
                    "Current dynamics selects the frozen substrate region along "
                    "global +z; the resolved outward normal is not aligned with it."
                ),
            }
        )
    return {
        "implemented_protocol": EXECUTABLE_PROTOCOL,
        "requested_protocol": protocol_name,
        "input_level": "preassembled_dense_interface",
        "status": "ready_with_known_limits" if not issues else "not_executable",
        "executable": not issues,
        "readiness_issues": issues,
        "limitations": [
            {
                "code": "preassembled_input_only",
                "classification": "protocol_constraint",
                "message": (
                    "Construction and protonation are not part of config execution; "
                    "the input must already contain the complete dense interface."
                ),
            },
            {
                "code": "contiguous_substrate_first_layout",
                "classification": "protocol_constraint",
                "message": (
                    "Atoms must be substrate-first followed by equal, contiguous "
                    "SAM molecule blocks."
                ),
            },
            {
                "code": "all_substrate_h_are_interface_protons",
                "classification": "protocol_constraint",
                "message": (
                    "Every H atom in the substrate block is counted as an interface "
                    "proton and is expected to have an O parent."
                ),
            },
            {
                "code": "global_z_frozen_region",
                "classification": "missing_generalization",
                "message": (
                    "The resolved Surface Frame is not yet propagated to frozen-layer "
                    "selection in the dynamics stages."
                ),
            },
            {
                "code": "carbon_neighbor_marker_validation",
                "classification": "missing_generalization",
                "message": (
                    "Execution validation represents marker topology only as a count "
                    "of carbon neighbors."
                ),
            },
            {
                "code": "analysis_not_orchestrated",
                "classification": "missing_generalization",
                "message": (
                    "Trajectory, structural, and dipole analysis are not yet stages "
                    "of config execution."
                ),
            },
        ],
    }


def resolve_system_plan(config_path: Path) -> dict:
    config, config_path = load_run_description(config_path)
    system = config["system"]
    anchor = config["sam"]["anchor"]
    marker = config["sam"].get("marker")
    interface = config["interface"]
    assembly = config["assembly"]
    executor = config["executor"]

    system_id = _required_string(system, "id", "system")
    structure = (config_path.parent / _required_string(system, "structure", "system")).resolve()
    if not structure.is_file():
        raise ValueError(f"System structure does not exist: {structure}")
    molecule_count = _positive_int(assembly, "molecule_count", "assembly")
    protons_per_sam = _positive_int(
        interface, "protons_per_sam", "interface", allow_zero=True
    )
    protocol_name = _required_string(config["protocol"], "name", "protocol")
    executor_name = _required_string(executor, "name", "executor")
    engine = _required_string(executor, "engine", "executor")
    if engine not in {"ase", "lammps-mliap"}:
        raise ValueError("executor.engine must be ase or lammps-mliap")
    model = (config_path.parent / _required_string(executor, "model", "executor")).resolve()
    if not model.is_file():
        raise ValueError(f"Executor model does not exist: {model}")
    lammps = executor.get("lammps")
    if engine == "lammps-mliap" and (not isinstance(lammps, str) or not lammps.strip()):
        raise ValueError("executor.lammps is required for lammps-mliap")
    species = parse_species(executor.get("species", ()))
    anchor_element, anchor_pattern = _element_pattern(anchor, "sam.anchor")

    atoms = read_typed_structure(structure, species)
    symbols = np.asarray(atoms.get_chemical_symbols())
    missing_species = sorted(set(symbols) - set(species))
    if missing_species:
        raise ValueError(
            "Executor species do not cover the structure: " + ", ".join(missing_species)
        )
    substrate_atoms, block_size = _infer_contiguous_layout(
        symbols, molecule_count, anchor_element
    )
    _validate_local_pattern(
        atoms,
        substrate_atoms,
        molecule_count,
        block_size,
        anchor_element,
        anchor_pattern,
        "anchor",
    )
    marker_result = None
    if marker is not None:
        marker_element, marker_pattern = _element_pattern(marker, "sam.marker")
        _validate_local_pattern(
            atoms,
            substrate_atoms,
            molecule_count,
            block_size,
            marker_element,
            marker_pattern,
            "marker",
        )
        marker_result = {
            "element": marker_element,
            "neighbor_pattern": list(marker_pattern),
        }
    surface_normal = config["substrate"].get("surface_normal")
    if surface_normal != "infer":
        raise ValueError('substrate.surface_normal currently must be "infer"')
    surface_frame = _infer_surface_frame(atoms, substrate_atoms)

    substrate_result = {"formula": _formula(symbols[:substrate_atoms])}
    restore_layer = config["substrate"].get("restore_layer")
    stages = list(BASE_PLAN_STAGES)
    if restore_layer is not None:
        if not isinstance(restore_layer, str) or not restore_layer.strip():
            raise ValueError("substrate.restore_layer must be a non-empty path")
        restore_path = (config_path.parent / restore_layer).resolve()
        if not restore_path.is_file():
            raise ValueError(f"Restore layer does not exist: {restore_path}")
        expected_restore_atoms = _positive_int(
            config["substrate"], "expected_restore_atoms", "substrate"
        )
        restore_atoms = read_typed_structure(restore_path, species)
        if len(restore_atoms) != expected_restore_atoms:
            raise ValueError(
                f"Restore layer has {len(restore_atoms)} atoms; "
                f"expected {expected_restore_atoms}"
            )
        substrate_result["restore_layer"] = {
            "path": str(restore_path),
            "sha256": _sha256(restore_path),
            "atom_count": len(restore_atoms),
        }
        stages.insert(stages.index("analysis"), "layer_restoration")
    elif "expected_restore_atoms" in config["substrate"]:
        raise ValueError("substrate.expected_restore_atoms requires restore_layer")

    observed_protons = int(np.count_nonzero(symbols[:substrate_atoms] == "H"))
    expected_protons = molecule_count * protons_per_sam
    if observed_protons != expected_protons:
        raise ValueError(
            f"Interface proton inventory mismatch: observed={observed_protons}, "
            f"expected={expected_protons}"
        )
    first_block = symbols[substrate_atoms : substrate_atoms + block_size]
    allowed_policies = {"network", "retained", "mobile"}
    preproduction_policy = _required_string(
        interface, "preproduction_surface_h_policy", "interface"
    )
    production_policy = _required_string(
        interface, "production_surface_h_policy", "interface"
    )
    if preproduction_policy not in allowed_policies or production_policy not in allowed_policies:
        raise ValueError("Interface surface-H policies must be network, retained, or mobile")
    allowed_contacts = interface.get("allowed_contact_pairs", [])
    if not isinstance(allowed_contacts, list) or not all(
        isinstance(pair, str) and pair.strip() for pair in allowed_contacts
    ):
        raise ValueError("interface.allowed_contact_pairs must be a string list")
    execution_contract = _execution_contract(
        protocol_name,
        engine,
        anchor_element,
        anchor_pattern,
        marker_result,
        surface_frame,
    )
    protonation_manifest_record = None
    protonation_manifest_value = interface.get("protonation_manifest")
    if protonation_manifest_value is None:
        execution_contract["readiness_issues"].append(
            {
                "code": "physical_promotion_ancestry_required",
                "message": (
                    "interface.protonation_manifest must seal the exact input structure "
                    "and a passed physical-interface promotion"
                ),
            }
        )
        execution_contract["executable"] = False
    elif not isinstance(protonation_manifest_value, str) or not (
        protonation_manifest_value.strip()
    ):
        raise ValueError("interface.protonation_manifest must be a non-empty path")
    else:
        protonation_manifest_path = (
            config_path.parent / protonation_manifest_value
        ).resolve()
        load_protonated_parent(structure, protonation_manifest_path)
        protonation_manifest_record = {
            "path": str(protonation_manifest_path),
            "sha256": _sha256(protonation_manifest_path),
        }
    return {
        "schema_version": 1,
        "mode": "plan",
        "system_id": system_id,
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "structure": {"path": str(structure), "sha256": _sha256(structure)},
        "atom_count": len(atoms),
        "composition": _formula(symbols),
        "layout": {
            "substrate_atoms": substrate_atoms,
            "molecules": molecule_count,
            "atoms_per_molecule": block_size,
        },
        "sam": {
            "formula": _formula(first_block),
            "anchor": {
                "element": anchor_element,
                "neighbor_pattern": list(anchor_pattern),
            },
            "marker": marker_result,
        },
        "substrate": substrate_result,
        "surface_frame": surface_frame,
        "interface_protons": {
            "expected": expected_protons,
            "observed": observed_protons,
        },
        "interface_policy": {
            "preproduction_surface_h_policy": preproduction_policy,
            "production_surface_h_policy": production_policy,
            "allowed_contact_pairs": allowed_contacts,
            "protonation_manifest": protonation_manifest_record,
        },
        "protocol": protocol_name,
        "execution_contract": execution_contract,
        "executor": {
            "name": executor_name,
            "engine": engine,
            "model": {"path": str(model), "sha256": _sha256(model)},
            "lammps": lammps,
            "device": executor.get("device"),
            "species": list(species),
        },
        "stages": stages,
        "warnings": [
            "Execution is limited to the explicit protocol contract; "
            "see execution_contract."
        ],
    }


def resolve_execution_args(config_path: Path, plan: dict | None = None) -> argparse.Namespace:
    """Map one validated run description onto the existing formal LAMMPS pipeline."""
    config, config_path = load_run_description(config_path)
    if plan is None:
        plan = resolve_system_plan(config_path)
    elif Path(plan["config"]["path"]) != config_path:
        raise ValueError("Resolved plan does not belong to the requested config")

    executor = config["executor"]
    contract = plan["execution_contract"]
    if not contract["executable"]:
        details = "; ".join(
            f"{issue['code']}: {issue['message']}"
            for issue in contract["readiness_issues"]
        )
        raise ValueError(f"Execution contract is not satisfied: {details}")
    if executor["engine"] != "lammps-mliap":
        raise ValueError("Config execution currently supports only lammps-mliap")

    protocol = config["protocol"]
    stage0 = _subsection(protocol, "stage0", "protocol.stage0")
    annealing = _subsection(protocol, "annealing", "protocol.annealing")
    production = _subsection(protocol, "production", "protocol.production")
    _reject_unknown(
        stage0,
        {"enabled", "freeze_depth_A", "iterations", "evaluations", "relative_etol"},
        "protocol.stage0",
    )
    _reject_unknown(
        annealing,
        {
            "cycles",
            "steps",
            "timestep_fs",
            "min_steps",
            "min_evaluations",
            "min_relative_etol",
            "max_temperature_K",
        },
        "protocol.annealing",
    )
    _reject_unknown(
        production,
        {
            "steps",
            "timestep_fs",
            "restrain_molecular_bonds",
            "heavy_bond_k_eV_per_A2",
            "h_bond_k_eV_per_A2",
        },
        "protocol.production",
    )

    marker = plan["sam"]["marker"]
    marker_pattern = marker["neighbor_pattern"]

    run = config.get("run")
    if not isinstance(run, dict):
        raise ValueError("Missing or invalid [run] section")
    if "seed" in run:
        raise ValueError("run.seed is not supported by the current formal pipeline")
    output_root = (
        config_path.parent / _required_string(run, "output_root", "run")
    ).resolve()
    run_label = run.get("id", "formal")
    if not isinstance(run_label, str):
        raise ValueError("run.id must be a non-empty string")
    run_label = _path_component(run_label, "run.id")
    system_id = _path_component(plan["system_id"], "system.id")
    run_fingerprint = _run_fingerprint(plan)
    run_id = f"{run_label}-{run_fingerprint[:12]}"
    outdir = output_root / system_id / run_id
    smoke = run.get("smoke", False)
    if not isinstance(smoke, bool):
        raise ValueError("run.smoke must be true or false")
    frozen_atom_ids_file = run.get("frozen_atom_ids_file")
    if frozen_atom_ids_file is not None:
        if not isinstance(frozen_atom_ids_file, str) or not frozen_atom_ids_file.strip():
            raise ValueError("run.frozen_atom_ids_file must be a non-empty path")
        frozen_atom_ids_file = (config_path.parent / frozen_atom_ids_file).resolve()
        if not frozen_atom_ids_file.is_file():
            raise ValueError(
                f"Frozen atom ID file does not exist: {frozen_atom_ids_file}"
            )

    restore = plan["substrate"].get("restore_layer")
    interface = config["interface"]
    return argparse.Namespace(
        engine="lammps-mliap",
        input_data=Path(plan["structure"]["path"]),
        model=Path(plan["executor"]["model"]["path"]),
        mliap_model=Path(plan["executor"]["model"]["path"]),
        lammps=plan["executor"]["lammps"],
        outdir=str(outdir),
        prefix=system_id,
        run_id=run_id,
        run_label=run_label,
        run_fingerprint=run_fingerprint,
        substrate_atoms=plan["layout"]["substrate_atoms"],
        molecules=plan["layout"]["molecules"],
        atoms_per_molecule=plan["layout"]["atoms_per_molecule"],
        protons_per_molecule=interface["protons_per_sam"],
        parent_protonation_manifest=Path(
            plan["interface_policy"]["protonation_manifest"]["path"]
        ),
        marker_element=marker["element"],
        marker_carbon_neighbors=len(marker_pattern),
        surface_h_policy=plan["interface_policy"][
            "preproduction_surface_h_policy"
        ],
        production_surface_h_policy=plan["interface_policy"][
            "production_surface_h_policy"
        ],
        allowed_interface_pair=list(
            plan["interface_policy"]["allowed_contact_pairs"]
        ),
        skip_stage0=not _boolean(stage0, "enabled", "protocol.stage0"),
        freeze_depth=_number(stage0, "freeze_depth_A", "protocol.stage0"),
        frozen_atom_ids_file=frozen_atom_ids_file,
        stage0_iterations=_positive_int(
            stage0, "iterations", "protocol.stage0"
        ),
        stage0_evaluations=_positive_int(
            stage0, "evaluations", "protocol.stage0"
        ),
        stage0_etol=_number(
            stage0, "relative_etol", "protocol.stage0", allow_zero=True
        ),
        anneal_cycles=_positive_int(
            annealing, "cycles", "protocol.annealing"
        ),
        anneal_steps=_positive_int(annealing, "steps", "protocol.annealing"),
        dt_anneal=_number(annealing, "timestep_fs", "protocol.annealing"),
        anneal_min_steps=_positive_int(
            annealing, "min_steps", "protocol.annealing"
        ),
        anneal_min_evaluations=_positive_int(
            annealing, "min_evaluations", "protocol.annealing"
        ),
        anneal_min_etol=_number(
            annealing,
            "min_relative_etol",
            "protocol.annealing",
            allow_zero=True,
        ),
        max_temp=_number(
            annealing, "max_temperature_K", "protocol.annealing"
        ),
        production_steps=_positive_int(
            production, "steps", "protocol.production"
        ),
        dt_production=_number(
            production, "timestep_fs", "protocol.production"
        ),
        production_restrain_molecular_bonds=_boolean(
            production, "restrain_molecular_bonds", "protocol.production"
        ),
        production_heavy_bond_k=_number(
            production,
            "heavy_bond_k_eV_per_A2",
            "protocol.production",
            allow_zero=True,
        ),
        production_h_bond_k=_number(
            production,
            "h_bond_k_eV_per_A2",
            "protocol.production",
            allow_zero=True,
        ),
        restore_layer=Path(restore["path"]) if restore else None,
        expected_restore_atoms=restore["atom_count"] if restore else None,
        smoke=smoke,
        resolved_plan=plan,
    )


def _artifact_record(args, path: Path, role: str, sha256: str | None = None) -> dict:
    path = Path(path).resolve()
    try:
        recorded_path = str(path.relative_to(args.outdir_path))
        external = False
    except ValueError:
        recorded_path = str(path)
        external = True
    return {
        "role": role,
        "path": recorded_path,
        "sha256": sha256 or _sha256(path),
        "external": external,
    }


def prepare_run_workspace(args) -> Path:
    """Create one immutable, fingerprinted workspace for config-driven execution."""
    if getattr(args, "resolved_plan", None) is None:
        raise ValueError("A managed run requires a resolved plan")
    outdir = Path(args.outdir).resolve()
    if outdir.exists():
        raise ValueError(
            f"Run directory already exists and will not be overwritten: {outdir}"
        )
    outdir.mkdir(parents=True)
    args.outdir_path = outdir
    args.stage_dirs = {
        name: outdir / "stages" / directory
        for name, directory in RUN_STAGE_DIRECTORIES.items()
    }
    for directory in (
        outdir / "inputs",
        *args.stage_dirs.values(),
        outdir / "reports",
        outdir / "work",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    plan = args.resolved_plan
    config_snapshot = outdir / "inputs" / "run-config.toml"
    config_snapshot.write_bytes(Path(plan["config"]["path"]).read_bytes())
    resolved_plan = outdir / "inputs" / "resolved-plan.json"
    write_manifest(resolved_plan, plan)
    inputs = [
        _artifact_record(args, config_snapshot, "run_config"),
        _artifact_record(args, resolved_plan, "resolved_plan"),
        _artifact_record(
            args,
            Path(plan["structure"]["path"]),
            "system_structure",
            plan["structure"]["sha256"],
        ),
        _artifact_record(
            args,
            Path(plan["executor"]["model"]["path"]),
            "mliap_model",
            plan["executor"]["model"]["sha256"],
        ),
    ]
    restore = plan["substrate"].get("restore_layer")
    if restore:
        inputs.append(
            _artifact_record(
                args,
                Path(restore["path"]),
                "substrate_restore_layer",
                restore["sha256"],
            )
        )
    stage_names = [
        name
        for name in RUN_STAGE_DIRECTORIES
        if name != "layer_restoration" or restore is not None
    ]
    args.run_manifest_path = outdir / "run-manifest.json"
    args.current_stage = None
    write_manifest(
        args.run_manifest_path,
        {
            "schema_version": 1,
            "run_id": args.run_id,
            "run_label": args.run_label,
            "run_fingerprint": args.run_fingerprint,
            "system_id": plan["system_id"],
            "status": "running",
            "started_at": _timestamp(),
            "finished_at": None,
            "inputs": inputs,
            "stages": [
                {"name": name, "status": "pending", "outputs": []}
                for name in stage_names
            ],
            "error": None,
        },
    )
    return outdir


def _read_run_manifest(args) -> dict:
    return json.loads(Path(args.run_manifest_path).read_text())


def _set_run_stage(args, name: str, status: str, outputs=()) -> None:
    if not hasattr(args, "run_manifest_path"):
        return
    allowed = {"pending", "running", "passed", "failed", "skipped"}
    if status not in allowed:
        raise ValueError(f"Invalid run stage status: {status}")
    manifest = _read_run_manifest(args)
    try:
        stage = next(item for item in manifest["stages"] if item["name"] == name)
    except StopIteration as error:
        raise ValueError(f"Unknown managed run stage: {name}") from error
    stage["status"] = status
    if status == "running":
        stage["started_at"] = _timestamp()
        args.current_stage = name
    elif status in {"passed", "failed", "skipped"}:
        stage["finished_at"] = _timestamp()
        if outputs:
            stage["outputs"] = [
                _artifact_record(args, Path(path), role) for path, role in outputs
            ]
        if getattr(args, "current_stage", None) == name:
            args.current_stage = None
    write_manifest(args.run_manifest_path, manifest)


def _set_run_status(args, status: str, error: BaseException | None = None) -> None:
    if not hasattr(args, "run_manifest_path"):
        return
    manifest = _read_run_manifest(args)
    if status == "passed":
        unfinished = [
            stage["name"]
            for stage in manifest["stages"]
            if stage["status"] not in {"passed", "skipped"}
        ]
        if unfinished:
            raise ValueError(
                "Cannot pass a run with unfinished stages: " + ", ".join(unfinished)
            )
    manifest["status"] = status
    if status in {"passed", "failed", "interrupted"}:
        manifest["finished_at"] = _timestamp()
    manifest["error"] = (
        {"type": type(error).__name__, "message": str(error)} if error else None
    )
    write_manifest(args.run_manifest_path, manifest)


def _run_validation(args, structure: Path, label: str) -> Path:
    if hasattr(args, "stage_dirs"):
        stage = "input_validation" if label == "input" else "stage0"
        output = args.stage_dirs[stage] / f"{args.prefix}_{label}_validation.json"
    else:
        output = args.outdir_path / f"{args.prefix}_{label}_validation.json"
    command = [
        sys.executable,
        str(Path(__file__).with_name("validate_monolayer.py")),
        str(structure),
        "--substrate-atoms",
        str(args.substrate_atoms),
        "--molecules",
        str(args.molecules),
        "--atoms-per-molecule",
        str(args.atoms_per_molecule),
        "--protons-per-molecule",
        str(args.protons_per_molecule),
        "--marker-element",
        args.marker_element,
        "--marker-carbon-neighbors",
        str(args.marker_carbon_neighbors),
        "--forbid-marker-h",
        "--surface-h-policy",
        args.surface_h_policy,
        "--json-output",
        str(output),
    ]
    for pair in args.allowed_interface_pair:
        command.extend(("--allowed-interface-pair", pair))
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    return output


def run_lammps_pipeline(args) -> int:
    # MACE is a mandatory dependency even though force evaluation is delegated
    # to LAMMPS's embedded Python/ML-IAP interface.
    import mace  # noqa: F401

    managed = getattr(args, "resolved_plan", None) is not None
    if managed:
        prepare_run_workspace(args)
    try:
        result = _execute_lammps_pipeline(args)
        if managed:
            _set_run_status(args, "passed")
        return result
    except KeyboardInterrupt as error:
        if managed:
            if args.current_stage:
                _set_run_stage(args, args.current_stage, "failed")
            _set_run_status(args, "interrupted", error)
        raise
    except Exception as error:
        if managed:
            if args.current_stage:
                _set_run_stage(args, args.current_stage, "failed")
            _set_run_status(args, "failed", error)
        raise


def _execute_lammps_pipeline(args) -> int:
    required = {
        "input_data": args.input_data,
        "mliap_model": args.mliap_model,
        "parent_protonation_manifest": getattr(
            args, "parent_protonation_manifest", None
        ),
        "substrate_atoms": args.substrate_atoms,
        "molecules": args.molecules,
        "atoms_per_molecule": args.atoms_per_molecule,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"LAMMPS pipeline missing parameters: {', '.join(missing)}")
    args.outdir_path = Path(args.outdir).resolve()
    args.outdir_path.mkdir(parents=True, exist_ok=True)
    workdir = (args.outdir_path / "work").resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    progress = args.outdir_path / "progress.txt"
    scripts = Path(__file__).resolve().parent
    input_data = Path(args.input_data).resolve()
    mliap_model = Path(args.mliap_model).resolve()
    initial_atoms = read_typed_structure(input_data)
    initial_symbols = validate_layout(
        initial_atoms,
        args.substrate_atoms,
        args.molecules,
        args.atoms_per_molecule,
    )
    if args.frozen_atom_ids_file:
        frozen_ids_path = Path(args.frozen_atom_ids_file).resolve()
        frozen_ids = read_frozen_atom_ids(frozen_ids_path, args.substrate_atoms)
        initial_freeze_z = float(
            max(initial_atoms.positions[atom_id - 1, 2] for atom_id in frozen_ids)
        )
    else:
        initial_freeze_z, frozen_ids = frozen_atom_ids(
            initial_atoms,
            initial_symbols,
            args.substrate_atoms,
            args.freeze_depth,
        )
        frozen_ids_path = workdir / f"{args.prefix}_frozen_atom_ids.txt"
        frozen_ids_path.write_text(
            "# Stable 1-based atom IDs selected once from the pipeline input.\n"
            + "\n".join(
                " ".join(map(str, frozen_ids[start : start + 32]))
                for start in range(0, len(frozen_ids), 32)
            )
            + "\n"
        )
    progress.write_text("Pipeline preflight: validating input\n")
    _set_run_stage(args, "input_validation", "running")
    baseline_validation = _run_validation(args, input_data, "input")
    _set_run_stage(
        args,
        "input_validation",
        "passed",
        ((baseline_validation, "input_validation"),),
    )

    if args.skip_stage0:
        stage0_data = input_data
        stage0_validation = baseline_validation
        stage0_minimization = None
        _set_run_stage(args, "stage0", "skipped")
    else:
        _set_run_stage(args, "stage0", "running")
        stage0_dir = (
            args.stage_dirs["stage0"]
            if hasattr(args, "stage_dirs")
            else args.outdir_path
        )
        stage0_input = workdir / f"{args.prefix}_stage0.in"
        stage0_data = stage0_dir / f"{args.prefix}_stage0.data"
        stage0_log = stage0_dir / f"{args.prefix}_stage0.log"
        generate = [
            sys.executable,
            str(scripts / "stage0_relaxation.py"),
            "generate",
            str(input_data),
            str(stage0_input),
            "--data",
            str(input_data),
            "--result",
            str(stage0_data),
            "--model",
            str(mliap_model),
            "--parent-protonation-manifest",
            str(args.parent_protonation_manifest),
            "--substrate-atoms",
            str(args.substrate_atoms),
            "--molecules",
            str(args.molecules),
            "--atoms-per-molecule",
            str(args.atoms_per_molecule),
            "--protons-per-molecule",
            str(args.protons_per_molecule),
            "--freeze-depth",
            str(args.freeze_depth),
            "--frozen-atom-ids-file",
            str(frozen_ids_path),
            "--maxiter",
            str(args.stage0_iterations),
            "--maxeval",
            str(args.stage0_evaluations),
            "--etol",
            str(args.stage0_etol),
        ]
        subprocess.run(generate, check=True)
        progress.write_text("Pipeline Stage 0: running\n")
        run_lammps(args.lammps, stage0_input, stage0_log, progress, None, workdir)
        stage0_minimization = minimization_summary(stage0_log)
        stage0_validation = _run_validation(args, stage0_data, "stage0")
        _set_run_stage(
            args,
            "stage0",
            "passed",
            (
                (stage0_data, "stage0_structure"),
                (stage0_log, "stage0_log"),
                (stage0_validation, "stage0_validation"),
            ),
        )

    _set_run_stage(args, "annealing", "running")
    anneal_dir = (
        args.stage_dirs["annealing"]
        if hasattr(args, "stage_dirs")
        else args.outdir_path / "annealing"
    )
    anneal_manifest = anneal_dir / f"{args.prefix}_annealing_manifest.json"
    # A smoke run must remain short without compressing a 300->500->300 K
    # schedule into only a handful of timesteps; that creates nonphysical
    # proton-network failures unrelated to pipeline correctness.
    anneal_steps = 60 if args.smoke else args.anneal_steps
    anneal_cycles = 1 if args.smoke else args.anneal_cycles
    anneal_min_steps = 50 if args.smoke else args.anneal_min_steps
    anneal_min_evals = 500 if args.smoke else args.anneal_min_evaluations
    anneal_command = [
        sys.executable,
        str(scripts / "run_annealing.py"),
        "--engine",
        "lammps-mliap",
        "--input",
        str(stage0_data),
        "--model",
        str(args.model or mliap_model),
        "--mliap-model",
        str(mliap_model),
        "--lammps",
        args.lammps,
        "--outdir",
        str(anneal_dir),
        "--workdir",
        str(workdir / "anneal"),
        "--manifest",
        str(anneal_manifest),
        "--progress",
        str(progress),
        "--prefix",
        args.prefix,
        "--substrate-atoms",
        str(args.substrate_atoms),
        "--molecules",
        str(args.molecules),
        "--atoms-per-molecule",
        str(args.atoms_per_molecule),
        "--protons-per-molecule",
        str(args.protons_per_molecule),
        "--cycles",
        str(anneal_cycles),
        "--steps",
        str(anneal_steps),
        "--dt",
        str(args.dt_anneal),
        "--max-temp",
        str(args.max_temp),
        "--min-steps",
        str(anneal_min_steps),
        "--min-evaluations",
        str(anneal_min_evals),
        "--min-etol",
        str(args.anneal_min_etol),
        "--freeze-depth",
        str(args.freeze_depth),
        "--frozen-atom-ids-file",
        str(frozen_ids_path),
        "--validate-script",
        str(scripts / "validate_monolayer.py"),
        "--validation-dir",
        str(anneal_dir / "validation"),
        "--marker-element",
        args.marker_element,
        "--marker-carbon-neighbors",
        str(args.marker_carbon_neighbors),
        "--forbid-marker-h",
        "--surface-h-policy",
        args.surface_h_policy,
    ]
    for pair in args.allowed_interface_pair:
        anneal_command.extend(("--allowed-interface-pair", pair))
    progress.write_text("Pipeline annealing: starting\n")
    subprocess.run(anneal_command, check=True)
    anneal = json.loads(anneal_manifest.read_text())
    best_data = Path(anneal["best_data"])
    _set_run_stage(
        args,
        "annealing",
        "passed",
        (
            (anneal_manifest, "annealing_manifest"),
            (best_data, "annealing_best_structure"),
        ),
    )

    _set_run_stage(args, "production", "running")
    production_dir = (
        args.stage_dirs["production"]
        if hasattr(args, "stage_dirs")
        else args.outdir_path / "production"
    )
    production_manifest = production_dir / f"{args.prefix}_production_manifest.json"
    production_steps = 12 if args.smoke else args.production_steps
    production_command = [
        sys.executable,
        str(scripts / "run_langevin_md_hmr.py"),
        "--engine",
        "lammps-mliap",
        "--input",
        str(best_data),
        "--model",
        str(args.model or mliap_model),
        "--mliap-model",
        str(mliap_model),
        "--lammps",
        args.lammps,
        "--outdir",
        str(production_dir),
        "--workdir",
        str(workdir / "production"),
        "--manifest",
        str(production_manifest),
        "--progress",
        str(progress),
        "--prefix",
        args.prefix,
        "--substrate-atoms",
        str(args.substrate_atoms),
        "--molecules",
        str(args.molecules),
        "--atoms-per-molecule",
        str(args.atoms_per_molecule),
        "--protons-per-molecule",
        str(args.protons_per_molecule),
        "--steps",
        str(production_steps),
        "--dt",
        str(args.dt_production),
        "--freeze-depth",
        str(args.freeze_depth),
        "--frozen-atom-ids-file",
        str(frozen_ids_path),
        "--validate-script",
        str(scripts / "validate_monolayer.py"),
        "--validation-output",
        str(production_dir / f"{args.prefix}_production_validation.json"),
        "--marker-element",
        args.marker_element,
        "--marker-carbon-neighbors",
        str(args.marker_carbon_neighbors),
        "--forbid-marker-h",
        "--surface-h-policy",
        args.production_surface_h_policy,
    ]
    if args.production_restrain_molecular_bonds:
        production_command.extend(
            (
                "--restrain-molecular-bonds",
                "--heavy-bond-k",
                str(args.production_heavy_bond_k),
                "--h-bond-k",
                str(args.production_h_bond_k),
            )
        )
    for pair in args.allowed_interface_pair:
        production_command.extend(("--allowed-interface-pair", pair))
    progress.write_text("Pipeline production MD: starting\n")
    subprocess.run(production_command, check=True)
    production = json.loads(production_manifest.read_text())
    _set_run_stage(
        args,
        "production",
        "passed",
        (
            (production_manifest, "production_manifest"),
            (Path(production["final_data"]), "production_final_data"),
            (Path(production["final_cif"]), "production_final_cif"),
        ),
    )

    restored = None
    restore_manifest = None
    if args.restore_layer:
        _set_run_stage(args, "layer_restoration", "running")
        restore_dir = (
            args.stage_dirs["layer_restoration"]
            if hasattr(args, "stage_dirs")
            else args.outdir_path
        )
        restored = restore_dir / f"{args.prefix}_production_final_full3layer.cif"
        restore_manifest = restore_dir / f"{args.prefix}_layer_restore_manifest.json"
        restore_command = [
            sys.executable,
            str(scripts / "monolayer_builder.py"),
            "--method",
            "layer-restore",
            "--input",
            production["final_cif"],
            "--layer",
            str(Path(args.restore_layer).resolve()),
            "--output",
            str(restored),
            "--manifest",
            str(restore_manifest),
            "--substrate-atoms",
            str(args.substrate_atoms),
            "--molecules",
            str(args.molecules),
            "--atoms-per-molecule",
            str(args.atoms_per_molecule),
        ]
        if args.expected_restore_atoms:
            restore_command.extend(
                ("--expected-layer-atoms", str(args.expected_restore_atoms))
            )
        subprocess.run(restore_command, check=True)
        _set_run_stage(
            args,
            "layer_restoration",
            "passed",
            (
                (restored, "restored_full_snapshot"),
                (restore_manifest, "layer_restore_manifest"),
            ),
        )

    manifest = {
        "engine": "lammps-mliap",
        "preproduction_surface_h_policy": args.surface_h_policy,
        "production_surface_h_policy": args.production_surface_h_policy,
        "production_restrain_molecular_bonds": args.production_restrain_molecular_bonds,
        "stage0_relative_etol": args.stage0_etol,
        "anneal_min_relative_etol": args.anneal_min_etol,
        "frozen_selection_mode": "stable_atom_ids",
        "frozen_atom_ids_file": str(frozen_ids_path),
        "frozen_atom_count": len(frozen_ids),
        "initial_frozen_z_max_A": initial_freeze_z,
        "smoke_test": args.smoke,
        "input": str(input_data),
        "input_validation": str(baseline_validation),
        "stage0_data": str(stage0_data),
        "stage0_validation": str(stage0_validation),
        "stage0_minimization": stage0_minimization,
        "annealing_manifest": str(anneal_manifest),
        "annealing_best_data": str(best_data),
        "production_manifest": str(production_manifest),
        "production_final_data": production["final_data"],
        "production_final_cif": production["final_cif"],
        "restored_full_snapshot": str(restored) if restored else None,
        "restore_manifest": str(restore_manifest) if restore_manifest else None,
        "passed": True,
    }
    pipeline_manifest = (
        args.outdir_path / "reports" / f"{args.prefix}_pipeline_manifest.json"
        if hasattr(args, "stage_dirs")
        else args.outdir_path / f"{args.prefix}_pipeline_manifest.json"
    )
    write_manifest(pipeline_manifest, manifest)
    progress.write_text("Pipeline completed with every validation gate passed\n")
    print(json.dumps(manifest, indent=2))
    return 0

PVKSAM_RELEASE = '1.0.0-rc.12'


def _pvksam_file_lock(paths):
    """Content identities, never mtime / 按内容锁定输入及阶段产物。"""
    return {str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in sorted(set(map(str, paths)))}


def _pvksam_check_lock(files):
    for path, expected in files.items():
        p = Path(path)
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != expected:
            raise ValueError(f'PVKSAM locked file changed or missing: {p}')


def _pvksam_runtime_identity(python):
    code = ('import sys,json,importlib.metadata as m; '
            'names=["ase","numpy","scipy","torch","mace-torch"]; '
            'installed={d.metadata["Name"].lower():d.version for d in m.distributions()}; '
            'print(json.dumps({"python":sys.version,"packages":'
            '{n:installed.get(n) for n in names}},sort_keys=True))')
    return json.loads(subprocess.check_output([python, '-c', code], text=True))


def _pvksam_front_digest(plan):
    return hashlib.sha256(json.dumps({k:v for k,v in plan.items() if k != 'plan_sha256'},
        sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def resolve_pvksam_front_plan(config_path):
    """Seal the fixed-head phosphonate systematic-skeleton scan.

    The maintained slab/site package is external scientific input, not regenerated
    from molecule size. No candidate rank, atom count or output N is invented.
    """
    from substrate_library import _resolve_submission_recipe_and_sam
    source = Path(config_path).resolve()
    c = json.loads(source.read_text())
    required = {'sam', 'substrate_key', 'catalog', 'growth_substrate', 'layer_groups',
                'site_instances', 'site_prototypes', 'model', 'output',
                'site_prototype_id', 'site_cell'}
    if not isinstance(c, dict):
        raise ValueError('PVKSAM front configuration must be a JSON object')
    removed_options = {'confsearch', 'confsearch_awf_root', 'scan_angles_deg',
                       'z_height_prefilter'} & set(c)
    if removed_options:
        raise ValueError('Only phosphonate-skeleton-grid is supported; removed options: '
                         + ', '.join(sorted(removed_options)))
    optional = {'donor_assignment', 'torsion_atom_ids', 'python', 'fmax',
                'max_steps', 'maxstep', 'candidate_source', 'skeleton_scan'}
    if required-set(c) or set(c)-required-optional:
        raise ValueError(f'PVKSAM front requires {sorted(required)}; optional {sorted(optional)}')
    c = dict(c)
    c.setdefault('candidate_source', 'phosphonate-skeleton-grid')
    if c['candidate_source'] != 'phosphonate-skeleton-grid':
        raise ValueError('Only phosphonate-skeleton-grid is supported for PVKSAM initial conformers')
    if 'torsion_atom_ids' not in c:
        raise ValueError('phosphonate-skeleton-grid requires explicit torsion_atom_ids')
    settings = c.get('skeleton_scan', {})
    if not isinstance(settings, dict) or set(settings) - {'step_deg','batch_size','max_candidates','limit','isomer_name'}:
        raise ValueError('invalid skeleton_scan settings')
    settings = dict(settings)
    settings.setdefault('step_deg', 30.0)
    settings.setdefault('batch_size', 256)
    settings.setdefault('max_candidates', 100000)
    if (isinstance(settings['step_deg'], bool) or not np.isfinite(settings['step_deg'])
            or settings['step_deg'] <= 0 or not np.isclose(round(360/settings['step_deg'])*settings['step_deg'],360)):
        raise ValueError('skeleton_scan.step_deg must divide 360 exactly')
    for key in ('batch_size','max_candidates'):
        if type(settings[key]) is not int or settings[key] <= 0:
            raise ValueError('skeleton_scan '+key+' must be a positive integer')
    if settings.get('limit') is not None and (type(settings['limit']) is not int or settings['limit'] <= 0):
        raise ValueError('skeleton_scan.limit must be a positive integer')
    c['skeleton_scan'] = settings
    for key in required-{'substrate_key', 'site_prototype_id', 'site_cell'}:
        p = Path(c[key]).expanduser()
        c[key] = str((source.parent/p).resolve() if not p.is_absolute() else p.resolve())
    c.setdefault('python', sys.executable)
    c['python'] = str(Path(c['python']).expanduser().resolve())
    for key, default in [('fmax', .03), ('maxstep', .05), ('max_steps', 500)]:
        c.setdefault(key, default)
        if isinstance(c[key], bool) or not np.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f'Invalid positive optimizer parameter {key}')
    if type(c['max_steps']) is not int or type(c['site_cell']) is not int or c['site_cell'] < 1:
        raise ValueError('max_steps and site_cell must be positive integers')
    recipe, sam = _resolve_submission_recipe_and_sam(Path(c['sam']), c['substrate_key'],
        Path(c['catalog']), allow_deprotonated_h0=True)
    if sam['anchor']['element'] != 'P':
        raise ValueError('PVKSAM front growth currently requires phosphonate; carboxylate adapter is pending')
    released = set(sam['released_hydrogen_atom_ids_1based'])
    kept = [i for i in range(1, sam['atom_count']+1) if i not in released]
    id_map = {str(old): new for new, old in enumerate(kept, 1)}
    assignment = c.get('donor_assignment', dict(zip(sam['anchor']['donor_labels'],
                                                  sam['anchor']['donor_atom_ids_1based'])))
    if (not isinstance(assignment, dict) or set(assignment) != set(sam['anchor']['donor_labels'])
            or sorted(assignment.values()) != sorted(sam['anchor']['donor_atom_ids_1based'])):
        raise ValueError('donor_assignment must map every anchor O exactly once (original SAM 1-based IDs)')
    c['donor_assignment'] = assignment
    mapped = {label: id_map[str(i)] for label, i in assignment.items()}
    torsions = c['torsion_atom_ids']
    if (not isinstance(torsions, list) or not torsions or
            any(not isinstance(row, list) or len(row) != 4 or any(
                type(i) is not int or str(i) not in id_map for i in row) for row in torsions)):
        raise ValueError('torsion_atom_ids must contain explicit four-atom paths in the retained SAM')
    torsions = [[id_map[str(i)] for i in row] for row in torsions]
    catalog = Path(c['catalog'])
    catalog_files = [p for p in catalog.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    if not catalog_files:
        raise ValueError('Empty substrate catalog')
    # Include external paths referenced by the recipe, even outside the catalog.
    from substrate_library import _load_accepted_site_prototype_library
    _, library, library_path = _load_accepted_site_prototype_library(recipe)
    representatives = [(library_path.parent/item['representative_relaxed_structure']['path']).resolve()
                       for item in library['site_prototypes']]
    inputs = [c[k] for k in ['sam', 'growth_substrate', 'layer_groups', 'site_instances',
                             'site_prototypes', 'model', 'python']]
    inputs += [source, recipe['source_path'], library_path, *representatives, *catalog_files]
    owners = list(Path(__file__).parent.glob('*.py'))
    plan = {'schema': 'pvksam-front-plan-v1', 'workflow_release': PVKSAM_RELEASE,
        'config': c, 'input_lock': _pvksam_file_lock(inputs),
        'implementation_lock': _pvksam_file_lock(owners),
        'runtime': _pvksam_runtime_identity(c['python']),
        'controller_runtime': _pvksam_runtime_identity(sys.executable),
        'worker_environment': {'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                               'OPENBLAS_NUM_THREADS': '1'},
        'catalog_inventory': sorted(str(p.resolve()) for p in catalog_files),
        'intake': sam, 'retained_original_ids_1based': kept, 'original_to_h0_1based': id_map,
        'h0_donor_assignment': mapped, 'h0_torsions': torsions,
        'stages': ['canonicalize_h0', 'skeleton_coarse_cpu'],
        'scope': ('Only the fixed-head phosphonate systematic torsion grid is enabled. '
                  'MMFF94s optimization, common P-C phase recovery, symmetry-aware dedup, '
                  'site-roll screening and MACE are explicit subsequent stages.')}
    plan['plan_sha256'] = _pvksam_front_digest(plan)
    return plan


def execute_pvksam_front_plan(plan, *, through='cpu', resume=False):
    """Execute and seal the only supported initial-conformer route's CPU scan."""
    import contextlib
    from types import SimpleNamespace
    from ase.io import read, write
    import mace_conformation_scan as scan
    if plan.get('plan_sha256') != _pvksam_front_digest(plan):
        raise ValueError('Front plan content changed after sealing')
    c = plan['config']; out = Path(c['output'])
    if c.get('candidate_source') != 'phosphonate-skeleton-grid':
        raise ValueError('Only phosphonate-skeleton-grid is supported for PVKSAM initial conformers')
    if through != 'cpu':
        raise ValueError('The only supported front stage is the systematic skeleton scan; use --front-through cpu')
    def check():
        _pvksam_check_lock(plan['input_lock'])
        _pvksam_check_lock(plan['implementation_lock'])
        current = sorted(str(p.resolve()) for p in Path(c['catalog']).rglob('*')
                         if p.is_file() and '__pycache__' not in p.parts)
        if current != plan['catalog_inventory']:
            raise ValueError('Catalog inventory changed after planning')
    check()
    if (_pvksam_runtime_identity(c['python']) != plan['runtime'] or
            _pvksam_runtime_identity(sys.executable) != plan['controller_runtime']):
        raise ValueError('PVKSAM runtime changed after planning')
    if resume:
        saved = json.loads((out/'front-plan.json').read_text())
        if saved != plan:
            raise ValueError('Resume requires the exact sealed front plan')
        state = json.loads((out/'front-manifest.json').read_text())
        if state['status'] == 'failed':
            raise ValueError('Failed stage retained; use a new output directory for repair/retry')
    else:
        out.mkdir(parents=True, exist_ok=False)
        write_manifest(out/'front-plan.json', plan)
        state = {'schema': 'pvksam-front-run-v1', 'status': 'running', 'stages': {}}
    def checkpoint():
        write_manifest(out/'front-manifest.json', state)
    def stage(name, action):
        check()
        for item in state['stages'].values():
            _pvksam_check_lock(item['artifacts'])
        if name in state['stages']:
            return state['stages'][name]['result']
        state.update(status='running', current_stage=name); checkpoint()
        with (out/(name+'.log')).open('w') as log, contextlib.redirect_stdout(log):
            result, files = action()
        state['stages'][name] = {'result': result, 'artifacts': _pvksam_file_lock(files)}
        checkpoint()
        print(json.dumps({'stage': name, 'status': 'completed', 'result': result}), flush=True)
        return result
    def canonicalize():
        atoms = read(c['sam'])[[i-1 for i in plan['retained_original_ids_1based']]]
        atoms.set_constraint()
        path = out/'sam-h0.extxyz'; write(path, atoms)
        return {'path': str(path), 'formula': dict(Counter(atoms.get_chemical_symbols()))}, [path]
    try:
        h0 = stage('canonicalize_h0', canonicalize)
        def skeleton_cpu():
            s = c['skeleton_scan']
            scan.run_phosphonate_skeleton_cpu_screen(SimpleNamespace(
                sam=Path(h0['path']), torsion_atom_ids=plan['h0_torsions'],
                scan_step_deg=s['step_deg'], full_grid_batch_size=s['batch_size'],
                scan_max_candidates=s['max_candidates'], skeleton_limit=s.get('limit'),
                skeleton_isomer_name=s.get('isomer_name', Path(c['sam']).stem),
                output_root=out/'01-skeleton-scan'))
            paths = list((out/'01-skeleton-scan').glob('*/manifest.json'))
            if len(paths) != 1:
                raise ValueError('Expected exactly one skeleton CPU manifest')
            manifest_path = paths[0]
            manifest = json.loads(manifest_path.read_text())
            files = [manifest_path, Path(manifest['candidate_outcomes']),
                     Path(manifest['system']['sam_source']['path']),
                     *manifest_path.parent.glob('*_monomer_unoptimized_conformations/*.extxyz')]
            return {'manifest': str(manifest_path), 'summary': manifest['summary'],
                    'grid_complete': manifest['grid_complete']}, files
        result = stage('skeleton_coarse_cpu', skeleton_cpu)
        state.update(status='prepared', current_stage='awaiting_fast_skeleton_optimization',
                     scope='isolated_skeleton_coarse_screen', grid_complete=result['grid_complete'])
        checkpoint()
        return state
    except BaseException as error:
        state.update(status='failed', error={'type': type(error).__name__, 'message': str(error)})
        checkpoint()
        raise

PVKSAM_DEFAULTS = {
    'steps':2000, 'fmax':0.01, 'maxstep':0.02, 'threads':4,
    'cycles':20, 'convergence_patience':5, 'energy_per_sam':0.005,
    'cycle_steps':1500, 'dt':2.0, 'h_mass':4.0, 'seed':340827,
    'parents_per_sam':24, 'directions_per_parent':4, 'groups_per_sam':276,
    'metal_elements':['In','Sn'],
}


def resolve_pvksam_post_plan(config_path):
    """Read-only plan for H0 -> constrained relax -> H -> relax -> cyclic anneal."""
    from relax_monolayer import read_pvksam_structure, build_pvksam_restraints, sha
    path=Path(config_path).resolve();user=json.loads(path.read_text())
    required={'input','interface_bonds','model','output','substrate_atoms','molecules',
              'atoms_per_molecule','fixed_atoms'}
    if not isinstance(user,dict) or required-set(user):
        raise ValueError(f'PVKSAM requires {sorted(required)}')
    unknown=set(user)-required-set(PVKSAM_DEFAULTS)
    if unknown:raise ValueError(f'Unknown PVKSAM keys: {sorted(unknown)}')
    config={**PVKSAM_DEFAULTS,**user}
    for key in ('input','interface_bonds','model','output'):
        candidate=Path(config[key]).expanduser()
        config[key]=str((path.parent/candidate).resolve() if not candidate.is_absolute() else candidate.resolve())
    for key in ('substrate_atoms','molecules','atoms_per_molecule','steps','threads','cycles',
                'convergence_patience','cycle_steps','parents_per_sam','directions_per_parent','groups_per_sam'):
        if type(config[key]) is not int or config[key]<=0:raise ValueError(f'{key} must be a positive integer')
    if type(config['fixed_atoms']) is not int or not 0<=config['fixed_atoms']<=config['substrate_atoms']:
        raise ValueError('fixed_atoms must describe the leading fixed substrate block')
    if config['cycle_steps']<6:raise ValueError('cycle_steps must be >=6')
    for key in ('fmax','maxstep','energy_per_sam','dt','h_mass'):
        if isinstance(config[key],bool) or not np.isfinite(config[key]) or config[key]<=0:
            raise ValueError(f'{key} must be finite and positive')
    if not isinstance(config['metal_elements'],list) or not config['metal_elements']:
        raise ValueError('metal_elements must be a nonempty list')
    if type(config['seed']) is not int or config['seed']<0:raise ValueError('seed must be a nonnegative integer')
    atoms=read_pvksam_structure(config['input'])
    interface=json.loads(Path(config['interface_bonds']).read_text())
    spec=build_pvksam_restraints(atoms,config['substrate_atoms'],config['molecules'],
        config['atoms_per_molecule'],interface,metal_elements=config['metal_elements'])
    from validate_monolayer import audit_pvksam_adsorption_metals
    eligibility=audit_pvksam_adsorption_metals(atoms,config['substrate_atoms'],
        interface['bonds'],config['metal_elements'])
    if not eligibility['passed']:
        raise ValueError('PVKSAM excludes pre-adsorption CN>=6 metals; forbidden 1-based IDs: '
                         + str(eligibility['forbidden_metal_ids_1based']))
    hashes={k:sha(config[k]) for k in ('input','interface_bonds','model')}
    owners=['orchestrate_md_pipeline.py','relax_monolayer.py','surface_protonation.py',
            'validate_monolayer.py','sam_lammps.py','sam_structure_tools.py']
    implementations={name:sha(Path(__file__).parent/name) for name in owners}
    return {'schema':'pvksam-post-adsorption-plan-v1','workflow':'pvksam','workflow_release':PVKSAM_RELEASE,'config':config,
        'input_sha256':hashes,'implementation_sha256':implementations,'sealed_spec':spec,
        'adsorption_metal_eligibility':eligibility,
        'family':spec['family'],'surface_H_count':config['molecules']*spec['protons_per_molecule'],
        'stages':['seal_intact_h0','relax_h0','global_protonation','relax_post_h','cyclic_annealing'],
        'surface_OH_restraints':0,'physical_interface_pass':None,
        'not_evaluated':['O3_projected_triangle','non_anchor_height','post_relaxation_full_topology'],
        'verification_scope':'Engineering-tested; two-family end-to-end scientific validation deferred by user',
        'cycle_checks':'every cycle; cycle cap is a review checkpoint, not convergence'}


def execute_pvksam_post_plan(plan):
    """Execute the sealed stage chain; keep failures and refuse unconverged promotion."""
    from relax_monolayer import save, sha, remap_pvksam_restraints
    from surface_protonation import protonate_restrained_h0
    c=plan['config'];out=Path(c['output']);spec=plan['sealed_spec']
    for key,value in plan['input_sha256'].items():
        if sha(c[key])!=value:raise ValueError(f'Stale PVKSAM input: {key}')
    for name,value in plan['implementation_sha256'].items():
        if sha(Path(__file__).parent/name)!=value:raise ValueError(f'Stale PVKSAM implementation: {name}')
    from validate_monolayer import audit_pvksam_adsorption_metals
    from relax_monolayer import read_pvksam_structure
    eligibility=audit_pvksam_adsorption_metals(read_pvksam_structure(c['input']),
        c['substrate_atoms'],json.loads(Path(c['interface_bonds']).read_text())['bonds'],c['metal_elements'])
    if not eligibility['passed'] or eligibility != plan.get('adsorption_metal_eligibility'):
        raise ValueError('PVKSAM adsorption metal eligibility failed or plan is stale')
    out.mkdir(parents=True,exist_ok=False)
    status={'schema':'pvksam-post-adsorption-run-v1','workflow':'pvksam','workflow_release':PVKSAM_RELEASE,'status':'running',
            'stages':[],'physical_interface_pass':None,'scientific_end_to_end_validation':'deferred'}
    save(out/'plan.json',plan)
    def record(stage,state,**extra):
        item={'stage':stage,'status':state,**extra};status['stages'].append(item)
        save(out/'manifest.json',status);save(out/'progress.json',item)
        print(json.dumps(item,ensure_ascii=False),flush=True)
    def write_spec(name,current):
        directory=out/name;directory.mkdir()
        save(directory/'bonds.json',{'bonds':current['bonds']})
        save(directory/'interface.json',{'bonds':[{'metal':b['i']+1,'donor':b['j']+1,'target_A':b['r0_A']}
            for b in current['bonds'] if b['kind']=='interface']})
        if current['tetra']:save(directory/'tetra.json',current['tetra'])
        return directory
    def worker(stage,source,identities,nh,mode='relax'):
        dest=out/stage
        argv=[sys.executable,str(Path(__file__).with_name('relax_monolayer.py')),'--restrained',
            '--input',str(source),'--interface-bonds',str(identities/'interface.json'),
            '--restraints-json',str(identities/'bonds.json'),'--model',c['model'],
            '--output',str(dest),'--output-prefix','relaxed','--mode',mode,'--surface-h',str(nh)]
        for key in ('substrate_atoms','molecules','atoms_per_molecule','fixed_atoms','steps','fmax','maxstep','threads'):
            argv.extend(['--'+key.replace('_','-'),str(c[key])])
        if (identities/'tetra.json').exists():argv.extend(['--tetra-json',str(identities/'tetra.json')])
        if mode=='anneal':
            for key in ('cycles','convergence_patience','energy_per_sam','cycle_steps','dt','h_mass','seed'):
                argv.extend(['--'+key.replace('_','-'),str(c[key])])
        record(stage,'running',argv=argv)
        environment=os.environ.copy()
        environment.update(OMP_NUM_THREADS=str(c['threads']),MKL_NUM_THREADS=str(c['threads']),OPENBLAS_NUM_THREADS='1')
        with (out/(stage+'.log')).open('w',buffering=1) as log:
            subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,check=True,env=environment)
        result=json.loads((dest/'result.json').read_text())
        passed=(result.get('converged') is True if mode=='relax' else result.get('search_converged') is True)
        record(stage,'passed' if passed else 'needs_review',result=result)
        return passed,dest
    try:
        h0=write_spec('identities-h0',spec)
        post=write_spec('identities-post-h',remap_pvksam_restraints(spec,c['substrate_atoms'],plan['surface_H_count']))
        record('seal_intact_h0','passed',family=spec['family'],bond_count=len(spec['bonds']),
               tetrahedral_P_count=len(spec['tetra']['P_indices']) if spec['tetra'] else 0)
        ok,relaxed=worker('01-relax-h0',c['input'],h0,0)
        if not ok:
            status.update(status='needs_review',stop_reason='H0 force not converged');return status
        record('02-protonate','running')
        result=protonate_restrained_h0(relaxed/'relaxed.extxyz',out/'02-protonate',
            substrate_atoms=c['substrate_atoms'],molecules=c['molecules'],atoms_per_molecule=c['atoms_per_molecule'],
            sealed_spec=spec,**{k:c[k] for k in ('parents_per_sam','directions_per_parent','groups_per_sam')})
        record('02-protonate','passed',result=result)
        ok,relaxed=worker('03-relax-post-h',out/'02-protonate/protonated-interface.extxyz',post,plan['surface_H_count'])
        if not ok:
            status.update(status='needs_review',stop_reason='Post-H force not converged');return status
        ok,annealed=worker('04-anneal',relaxed/'relaxed.extxyz',post,plan['surface_H_count'],mode='anneal')
        status.update(status='completed' if ok else 'needs_review',
            search_converged=ok,best_structure=str(annealed/'best-relaxed.extxyz'),
            best_structure_sha256=sha(annealed/'best-relaxed.extxyz'),
            stop_reason=json.loads((annealed/'result.json').read_text())['stop_reason'])
        return status
    except BaseException as exc:
        status.update(status='failed',error={'type':type(exc).__name__,'message':str(exc)})
        raise
    finally:
        save(out/'manifest.json',status);save(out/'progress.json',status)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Orchestrated SAM/substrate MD pipeline")
    parser.add_argument('--pvksam-front', type=Path, help='Sealed fixed-head phosphonate skeleton-scan JSON')
    parser.add_argument('--front-through', choices=('cpu',), default='cpu')
    parser.add_argument('--front-resume', action='store_true', help='Resume only a verified prepared front run')
    parser.add_argument("--pvksam-post", type=Path, help="PVKSAM H0-to-anneal JSON; combine with --plan or --execute")
    parser.add_argument("--substrate-catalog", type=Path, help="External maintained substrate catalog for portable AWF execution")
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--sam",
        type=Path,
        help="Complete isolated SAM structure for the two-input public workflow",
    )
    parser.add_argument(
        "--substrate",
        help="Maintained substrate catalog key or alias for the public workflow",
    )
    parser.add_argument(
        "--workflow",
        choices=("single-adsorption", "dense-monolayer"),
        default="single-adsorption",
        help=(
            "public workflow target; single-adsorption is the default and sizes the "
            "substrate from the submitted molecule"
        ),
    )
    parser.add_argument(
        "--dopant-site-percent",
        type=float,
        help=(
            "override the substrate catalog recommendation with the percentage of "
            "registered host-sublattice sites replaced by dopant atoms; for ITO this "
            "is 100*Sn/(In+Sn), not wt%% SnO2"
        ),
    )
    parser.add_argument(
        "--vdw-radius-scale",
        type=float,
        help=(
            "formal ASE vdW-radius scale for the single-adsorption CPU collision "
            "gate; the selected value is recorded as user input"
        ),
    )
    parser.add_argument(
        "--target-size-nm",
        type=float,
        nargs=2,
        metavar=("U_NM", "V_NM"),
        help=(
            "target lengths of the two periodic surface-cell vectors; the nearest "
            "dopant-compatible fixed-lattice integer supercell may be smaller or "
            "larger; requires "
            "--workflow dense-monolayer"
        ),
    )
    parser.add_argument(
        "--packing-fraction",
        type=float,
        default=0.70,
        help=(
            "heuristic area packing fraction for a user-sized capacity estimate "
            "(default: 0.70)"
        ),
    )
    parser.add_argument(
        "--single-objective",
        choices=("minimum-energy", "minimum-projected-area", "multiple"),
        help="deterministic selection contract for single-molecule adsorption",
    )
    parser.add_argument(
        "--maximum-output-structures",
        type=int,
        default=8,
        help="maximum number of structures selected by --single-objective multiple",
    )
    config_mode = parser.add_mutually_exclusive_group()
    config_mode.add_argument("--plan", action="store_true")
    config_mode.add_argument(
        "--execute",
        action="store_true",
        help="Execute a validated config using the formal LAMMPS pipeline",
    )
    config_mode.add_argument(
        "--prepare-substrate",
        action="store_true",
        help="Materialize the resolved clean and doped substrate package",
    )
    config_mode.add_argument(
        "--prepare-candidates",
        action="store_true",
        help="Materialize balanced unrelaxed single-SAM adsorption candidates",
    )
    config_mode.add_argument(
        "--build-sized-monolayer",
        action="store_true",
        help="Build an approved collision-screened user-sized unrelaxed H0 monolayer",
    )
    config_mode.add_argument(
        "--promote-sized-monolayer",
        type=Path,
        metavar="PARENT_MANIFEST",
        help="Independently validate and promote a hash-sealed sized H0 parent",
    )
    config_mode.add_argument(
        "--promote-geometric-h0",
        type=Path,
        metavar="H0_STRUCTURE",
        help=(
            "Independently validate a geometric phosphonate H0 structure using "
            "the three-O projected-triangle and non-anchor-height contract"
        ),
    )
    config_mode.add_argument(
        "--protonate-promoted-h0",
        type=Path,
        metavar="PROMOTION_MANIFEST",
        help="Globally assign surface protons to a hash-sealed promoted H0 parent",
    )
    parser.add_argument(
        "--approve-plan-hash",
        help="exact approval_contract.plan_sha256 required by --build-sized-monolayer",
    )
    parser.add_argument(
        "--submission-root",
        type=Path,
        default=Path("runs"),
        help="Output root for two-input preparation artifacts (default: runs)",
    )
    parser.add_argument("--engine", choices=("ase", "lammps-mliap"), default="ase")
    parser.add_argument("--isomer", type=str, help="Legacy SAM isomer/project name")
    parser.add_argument("--peeled-cif", type=str, help="Path to starting peeled CIF supercell")
    parser.add_argument("--model", type=str, help="Path to Python MACE model file")
    parser.add_argument("--outdir", type=str, default="./pvksam-output", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="cuda/cpu")
    parser.add_argument("--restrain-pc", action="store_true", help="Apply Hookean harmonic restraints to P-C1 bonds during initial adsorption relaxation.")
    parser.add_argument("--restrain-all-bonds", action="store_true", help="Apply Hookean harmonic restraints to ALL SAM covalent bonds during simulated annealing.")
    parser.add_argument("--dt-anneal", type=float, default=2.0, help="Timestep in fs for simulated annealing.")
    parser.add_argument("--input-data", type=Path)
    parser.add_argument("--mliap-model", type=Path)
    parser.add_argument("--parent-protonation-manifest", type=Path)
    parser.add_argument("--lammps", default="lmp")
    parser.add_argument("--substrate-atoms", type=int)
    parser.add_argument("--molecules", type=int)
    parser.add_argument("--atoms-per-molecule", type=int)
    parser.add_argument("--protons-per-molecule", type=int, default=2)
    parser.add_argument(
        "--surface-metal-element",
        action="append",
        default=None,
        help="Accessible substrate metal element for geometric promotion; repeat as needed",
    )
    parser.add_argument("--surface-metal-depth", type=float, default=0.7)
    parser.add_argument("--interface-height-clearance", type=float, default=1.0)
    parser.add_argument(
        "--triangle-anchor-molecule",
        action="append",
        type=int,
        default=None,
        help=(
            "1-based SAM block required to satisfy the projected O-triangle rule; "
            "repeatable; default: every SAM block"
        ),
    )
    parser.add_argument(
        "--surface-normal",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 1.0),
        metavar=("NX", "NY", "NZ"),
    )
    parser.add_argument(
        "--surface-periodic-axes",
        nargs=2,
        type=int,
        default=(0, 1),
        metavar=("AXIS_U", "AXIS_V"),
    )
    parser.add_argument("--freeze-depth", type=float, default=1.10)
    parser.add_argument("--frozen-atom-ids-file", type=Path)
    parser.add_argument("--marker-element", default="S")
    parser.add_argument("--marker-carbon-neighbors", type=int, default=2)
    parser.add_argument(
        "--surface-h-policy",
        choices=("network", "retained", "mobile"),
        default="retained",
    )
    parser.add_argument(
        "--production-surface-h-policy",
        choices=("network", "retained", "mobile"),
        default="mobile",
    )
    parser.add_argument("--allowed-interface-pair", action="append", default=[])
    parser.add_argument("--prefix", default="sam")
    parser.add_argument("--skip-stage0", action="store_true")
    parser.add_argument("--stage0-iterations", type=int, default=240)
    parser.add_argument("--stage0-evaluations", type=int, default=2400)
    parser.add_argument("--stage0-etol", type=float, default=0.0)
    parser.add_argument("--anneal-cycles", type=int, default=3)
    parser.add_argument("--anneal-steps", type=int, default=1500)
    parser.add_argument("--anneal-min-steps", type=int, default=200)
    parser.add_argument("--anneal-min-evaluations", type=int, default=2000)
    parser.add_argument("--anneal-min-etol", type=float, default=0.0)
    parser.add_argument("--max-temp", type=float, default=500.0)
    parser.add_argument("--dt-production", type=float, default=2.0)
    parser.add_argument("--production-steps", type=int, default=3000)
    parser.add_argument(
        "--production-restrain-molecular-bonds",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--production-heavy-bond-k", type=float, default=20.0)
    parser.add_argument("--production-h-bond-k", type=float, default=10.0)
    parser.add_argument("--restore-layer", type=Path)
    parser.add_argument("--expected-restore-atoms", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.pvksam_front:
        if args.pvksam_post or args.config or args.sam or args.substrate or not (args.plan or args.execute):
            parser.error('--pvksam-front requires --plan or --execute without other workflow inputs')
        try:
            document = json.loads(args.pvksam_front.read_text())
            plan = document if document.get('schema') == 'pvksam-front-plan-v1' else resolve_pvksam_front_plan(args.pvksam_front)
            result = execute_pvksam_front_plan(plan, through=args.front_through,
                resume=args.front_resume) if args.execute else plan
        except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
            parser.error(str(error))
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.pvksam_post:
        if args.config or args.sam or args.substrate or not (args.plan or args.execute):
            parser.error('--pvksam-post requires --plan or --execute without --config/--sam/--substrate')
        try:
            plan = resolve_pvksam_post_plan(args.pvksam_post)
            result = execute_pvksam_post_plan(plan) if args.execute else plan
        except (OSError, ValueError, RuntimeError) as error:
            parser.error(str(error))
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result.get('status', 'completed') == 'completed' else 2
    if args.protonate_promoted_h0:
        if args.config or args.sam or args.substrate:
            parser.error(
                "--protonate-promoted-h0 consumes a prior manifest and cannot be "
                "combined with --config, --sam, or --substrate"
            )
        try:
            result = protonate_promoted_h0(
                args.protonate_promoted_h0,
                args.submission_root,
            )
        except (OSError, ValueError, RuntimeError) as error:
            parser.error(str(error))
        print(json.dumps(result, indent=2))
        return 0
    if args.promote_sized_monolayer:
        if args.config or args.sam or args.substrate:
            parser.error(
                "--promote-sized-monolayer consumes a prior manifest and cannot be "
                "combined with --config, --sam, or --substrate"
            )
        if not args.approve_plan_hash:
            parser.error("--promote-sized-monolayer requires --approve-plan-hash")
        try:
            result = promote_sized_monolayer_h0(
                args.promote_sized_monolayer,
                args.submission_root,
                approved_plan_sha256=args.approve_plan_hash,
            )
        except (OSError, ValueError) as error:
            parser.error(str(error))
        print(json.dumps(result, indent=2))
        return 0
    if args.promote_geometric_h0:
        if args.config or args.sam or args.substrate:
            parser.error(
                "--promote-geometric-h0 consumes a prior structure and cannot be "
                "combined with --config, --sam, or --substrate"
            )
        if any(
            value is None
            for value in (
                args.substrate_atoms,
                args.molecules,
                args.atoms_per_molecule,
            )
        ):
            parser.error(
                "--promote-geometric-h0 requires --substrate-atoms, --molecules, "
                "and --atoms-per-molecule"
            )
        try:
            result = promote_geometric_h0(
                args.promote_geometric_h0,
                args.submission_root,
                substrate_atoms=args.substrate_atoms,
                molecules=args.molecules,
                atoms_per_sam_h0=args.atoms_per_molecule,
                released_protons_per_sam=args.protons_per_molecule,
                surface_metal_elements=tuple(
                    args.surface_metal_element or ("In", "Sn")
                ),
                surface_metal_depth_A=args.surface_metal_depth,
                interface_height_clearance_A=args.interface_height_clearance,
                surface_normal=tuple(args.surface_normal),
                surface_periodic_axes=tuple(args.surface_periodic_axes),
                triangle_anchor_molecule_numbers=(
                    tuple(args.triangle_anchor_molecule)
                    if args.triangle_anchor_molecule is not None
                    else None
                ),
            )
        except (OSError, ValueError) as error:
            parser.error(str(error))
        print(json.dumps(result, indent=2))
        return 0
    if args.config:
        if args.workflow != "single-adsorption":
            parser.error("--workflow applies only to --sam/--substrate public runs")
        if args.dopant_site_percent is not None:
            parser.error(
                "--dopant-site-percent applies only to --sam/--substrate public runs"
            )
        if args.vdw_radius_scale is not None:
            parser.error(
                "--vdw-radius-scale applies only to --sam/--substrate public runs"
            )
        if args.sam or args.substrate:
            parser.error("--config cannot be combined with --sam or --substrate")
        if args.prepare_substrate:
            parser.error("--prepare-substrate requires --sam and --substrate")
        if args.prepare_candidates:
            parser.error("--prepare-candidates requires --sam and --substrate")
        if args.build_sized_monolayer or args.approve_plan_hash:
            parser.error(
                "--build-sized-monolayer and --approve-plan-hash require --sam and --substrate"
            )
        if not args.plan and not args.execute:
            parser.error("--config requires exactly one of --plan or --execute")
        try:
            plan = resolve_system_plan(args.config)
        except (OSError, ValueError, tomllib.TOMLDecodeError) as error:
            parser.error(str(error))
        if args.plan:
            print(json.dumps(plan, indent=2))
            return 0
        try:
            execution_args = resolve_execution_args(args.config, plan)
        except (OSError, ValueError, tomllib.TOMLDecodeError) as error:
            parser.error(str(error))
        return run_lammps_pipeline(execution_args)
    if args.sam or args.substrate:
        if not args.sam or not args.substrate:
            parser.error("The public workflow requires both --sam and --substrate")
        if args.execute:
            parser.error(
                "Two-input --execute is not available yet; use --plan or "
                "--prepare-substrate"
            )
        if (
            not args.plan
            and not args.prepare_substrate
            and not args.prepare_candidates
            and not args.build_sized_monolayer
        ):
            parser.error(
                "--sam and --substrate require --plan, --prepare-substrate, or "
                "--prepare-candidates, or --build-sized-monolayer"
            )
        if args.build_sized_monolayer and not args.approve_plan_hash:
            parser.error(
                "--build-sized-monolayer requires --approve-plan-hash"
            )
        try:
            plan = resolve_submission_plan(
                args.sam,
                args.substrate,
                catalog_directory=args.substrate_catalog,
                workflow={
                    "single-adsorption": "single_molecule_adsorption",
                    "dense-monolayer": "dense_monolayer",
                }[args.workflow],
                dopant_site_fraction=(
                    None
                    if args.dopant_site_percent is None
                    else args.dopant_site_percent / 100.0
                ),
                formal_collision_radius_scale=args.vdw_radius_scale,
                target_lateral_size_nm=args.target_size_nm,
                packing_fraction=args.packing_fraction,
                single_structure_objective=(
                    None
                    if args.single_objective is None
                    else args.single_objective.replace("-", "_")
                ),
                maximum_output_structures=args.maximum_output_structures,
            )
        except (OSError, ValueError, tomllib.TOMLDecodeError) as error:
            parser.error(str(error))
        if args.plan:
            print(json.dumps(plan, indent=2))
            return 0
        try:
            if args.build_sized_monolayer:
                result = materialize_sized_monolayer(
                    plan, args.submission_root, args.approve_plan_hash
                )
            elif args.prepare_candidates:
                result = materialize_submission_candidates(plan, args.submission_root)
            else:
                result = materialize_submission_substrate(plan, args.submission_root)
        except (OSError, ValueError) as error:
            parser.error(str(error))
        print(json.dumps(result, indent=2))
        return 0
    if (
        args.plan
        or args.execute
        or args.prepare_substrate
        or args.prepare_candidates
        or args.build_sized_monolayer
        or args.approve_plan_hash
    ):
        parser.error(
            "public/config actions require --config or both --sam and --substrate"
        )
    if args.workflow != "single-adsorption":
        parser.error("--workflow applies only to --sam/--substrate public runs")
    if args.dopant_site_percent is not None:
        parser.error(
            "--dopant-site-percent applies only to --sam/--substrate public runs"
        )
    if args.vdw_radius_scale is not None:
        parser.error(
            "--vdw-radius-scale applies only to --sam/--substrate public runs"
        )
    if not args.isomer:
        parser.error("--isomer is required when --config is absent")
    if args.engine == "lammps-mliap":
        return run_lammps_pipeline(args)
    if not args.peeled_cif or not args.model:
        parser.error("ASE engine requires --peeled-cif and --model")
    
    t_start = time.time()
    progress_path = os.path.join(args.outdir, "progress.txt")
    
    # 0. STAGE 0: Adsorption Geometry Relaxation
    print("\n==========================================")
    print("STAGE 0: Running LBFGS Adsorption Geometry Relaxation...")
    print("==========================================")
    relaxed_cif = os.path.join(args.outdir, f"{args.isomer}_4x4_md_peeled_relaxed.cif")
    with open(progress_path, 'w') as f:
        f.write("Pipeline stage: Adsorption relaxation...\n")
        
    relax_cmd = [
        "/home/software/anaconda3/envs/ase/bin/python",
        os.path.join(os.path.dirname(__file__), "relax_monolayer.py"),
        "--input", args.peeled_cif,
        "--model", args.model,
        "--output", relaxed_cif,
        "--device", args.device
    ]
    if args.restrain_pc:
        relax_cmd.append("--restrain-pc")
    if args.restrain_all_bonds:
        relax_cmd.append("--restrain-all-bonds")
        
    subprocess.check_call(relax_cmd)
    
    # Validation step: check for broken bonds immediately after relaxation
    print("\n==========================================")
    print("VALIDATION: Verifying covalent bond integrity...")
    print("==========================================")
    with open(progress_path, 'w') as f:
        f.write("Pipeline stage: Validation check...\n")
        
    validation_cmd = [
        "/home/software/anaconda3/envs/ase/bin/python",
        os.path.join(os.path.dirname(__file__), "check_relaxed_bonds.py"),
        "--raw", args.peeled_cif,
        "--relaxed", relaxed_cif
    ]
    subprocess.check_call(validation_cmd)
    
    # 1. STAGE 1: Simulated Annealing (max 500 K)
    print("\n==========================================")
    print("STAGE 1: Running Langevin Simulated Annealing (max 500 K)...")
    print("==========================================")
    with open(progress_path, 'w') as f:
        f.write("Pipeline stage: Annealing...\n")
        
    anneal_cmd = [
        "/home/software/anaconda3/envs/ase/bin/python",
        os.path.join(os.path.dirname(__file__), "run_annealing.py"),
        "--input", relaxed_cif,
        "--model", args.model,
        "--outdir", args.outdir,
        "--cycles", "3",
        "--steps", "1500",
        "--dt", str(args.dt_anneal)
    ]
    if args.restrain_pc:
        anneal_cmd.append("--restrain-pc")
    if args.restrain_all_bonds:
        anneal_cmd.append("--restrain-all-bonds")
        
    subprocess.check_call(anneal_cmd)
    
    best_cif = os.path.join(args.outdir, f"{args.isomer}_4x4_md_peeled_relaxed_anneal_relaxed_cycle_3.cif")
    if not os.path.exists(best_cif):
        best_cif = os.path.join(args.outdir, f"{args.isomer}_4x4_md_peeled_anneal_relaxed_cycle_3.cif")
        
    # Validation step 2: check for broken bonds immediately after simulated annealing
    print("\n==========================================")
    print("VALIDATION 2: Verifying covalent bond integrity after simulated annealing...")
    print("==========================================")
    with open(progress_path, 'w') as f:
        f.write("Pipeline stage: Validation check 2...\n")
        
    validation2_cmd = [
        "/home/software/anaconda3/envs/ase/bin/python",
        os.path.join(os.path.dirname(__file__), "check_relaxed_bonds.py"),
        "--raw", args.peeled_cif,
        "--relaxed", best_cif
    ]
    subprocess.check_call(validation2_cmd)
        
    # 2. STAGE 2: Production Langevin NVT MD
    print("\n==========================================")
    print("STAGE 2: Running Production Langevin NVT MD (3000 steps)...")
    print("==========================================")
    with open(progress_path, 'w') as f:
        f.write("Pipeline stage: Production MD...\n")
        
    md_cmd = [
        "/home/software/anaconda3/envs/ase/bin/python",
        os.path.join(os.path.dirname(__file__), "run_langevin_md_hmr.py"),
        "--input", best_cif,
        "--model", args.model,
        "--outdir", args.outdir,
        "--temp", "300.0",
        "--steps", "3000"
    ]
    if args.restrain_pc:
        md_cmd.append("--restrain-pc")
        
    subprocess.check_call(md_cmd)
    
    # 3. STAGE 3: Trajectory Post-processing Analysis
    traj_path = f"{os.path.splitext(best_cif)[0]}_production_nvt.traj"
    print("\n==========================================")
    print("STAGE 3: Running Trajectory Analysis...")
    print("==========================================")
    with open(progress_path, 'w') as f:
        f.write("Pipeline stage: Trajectory Analysis...\n")
        
    analysis_cmd = [
        "/home/software/anaconda3/envs/ase/bin/python",
        os.path.join(os.path.dirname(__file__), "analyze_production_traj.py"),
        "--prefix", f"{args.isomer}_4x4_md",
        "--traj", traj_path
    ]
    subprocess.check_call(analysis_cmd)
    
    # 4. STAGE 4: Trajectory Plotting
    print("\n==========================================")
    print("STAGE 4: Running Trajectory Plotting...")
    print("==========================================")
    
    plot_cmd = [
        "/home/software/anaconda3/envs/ase/bin/python",
        os.path.join(os.path.dirname(__file__), "plot_production_results.py"),
        "--prefix", f"{args.isomer}_4x4_md",
        "--dir", args.outdir
    ]
    subprocess.check_call(plot_cmd)
    
    t_end = time.time()
    print("\n==========================================")
    print(f"ORCHESTRATED PIPELINE COMPLETED IN {t_end - t_start:.2f} SECONDS!")
    print("==========================================")
    
    with open(progress_path, 'w') as f:
        f.write(f"Completed {args.isomer} 4x4 MD Pipeline successfully!\n")

if __name__ == "__main__":
    main()
