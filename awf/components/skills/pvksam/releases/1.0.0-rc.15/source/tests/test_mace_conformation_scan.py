from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT_DIRECTORY = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIRECTORY))

import mace_conformation_scan as scan  # noqa: E402

from mace_conformation_scan import (  # noqa: E402
    FIXED_SITE_CLUSTER_SENSITIVITY_A,
    FIXED_SITE_CLUSTER_TOLERANCE_A,
    TARGETED_CALIBRATION_MOLECULE_FORMULA,
    TARGETED_CALIBRATION_SITE_INSTANCE_ID,
    _fixed_site_surface_frame,
    _parse_donor_assignment,
    _targeted_calibration_residual_nontarget_audit,
    _targeted_file_record,
    _targeted_identity_value,
    _targeted_interface_covalent_contacts,
    _targeted_source_surface_contract,
    _targeted_surface_frame_from_evidence,
    build_targeted_calibration_initial_structure,
    classify_targeted_calibration,
    map_targeted_surface_protons,
    resolve_targeted_calibration_plan,
    complete_link_clusters,
    pareto_front_indices,
    resolve_fixed_site_layer_constraints,
    resolve_targeted_donor_mapping,
    targeted_p_o_coordination,
    validate_targeted_calibration_final_geometry,
)


def test_fixed_site_donor_assignment_is_explicit_and_one_to_one():
    assert _parse_donor_assignment("O1:45,O2:46,O3:44") == {
        "O1": 45,
        "O2": 46,
        "O3": 44,
    }
    with pytest.raises(ValueError, match="unique"):
        _parse_donor_assignment("O1:44,O2:44,O3:46")


def test_complete_link_uses_user_selected_two_angstrom_threshold():
    assert FIXED_SITE_CLUSTER_TOLERANCE_A == pytest.approx(2.0)
    assert FIXED_SITE_CLUSTER_SENSITIVITY_A == (0.5, 0.75, 1.0, 2.0)
    distances = np.array(
        [
            [0.0, 1.9, 2.1, 5.0],
            [1.9, 0.0, 1.0, 5.0],
            [2.1, 1.0, 0.0, 4.5],
            [5.0, 5.0, 4.5, 0.0],
        ]
    )
    # Complete-link blocks the 0-1-2 chaining merge because d(0, 2) > 2 A.
    assert complete_link_clusters(distances, 2.0) == [[0, 1], [2], [3]]


def test_pareto_front_keeps_area_and_energy_endpoints_without_weighted_score():
    areas = [10.0, 12.0, 11.0, 13.0]
    energies = [0.5, 0.0, 0.2, 0.8]
    assert pareto_front_indices(areas, energies) == [0, 2, 1]


def test_fixed_site_layer_constraints_allow_dopant_formula_variation():
    from ase import Atoms

    atoms = Atoms(
        "InOSnInOSn",
        positions=[
            [0, 0, 0.0], [0, 0, 0.1],
            [0, 0, 1.0], [0, 0, 1.1],
            [0, 0, 2.0], [0, 0, 2.1],
        ],
        cell=[5, 5, 5],
        pbc=True,
    )
    resolved = resolve_fixed_site_layer_constraints(
        atoms,
        substrate_atom_count=6,
        structural_layer_count=3,
        movable_layers_from_surface=1,
        minimum_layer_gap_A=0.5,
    )
    assert resolved["frozen_atom_ids_1based"] == [1, 2, 3, 4]
    assert resolved["movable_substrate_atom_ids_1based"] == [5, 6]
    assert [layer["composition"] for layer in resolved["layers_bottom_to_top"]] == [
        {"In": 1, "O": 1},
        {"In": 1, "Sn": 1},
        {"O": 1, "Sn": 1},
    ]


def test_fixed_site_surface_frame_is_right_handed_and_uses_cell_axes():
    plan = {
        "system": {"cell_A": [[10, 0, 0], [0, 8, 0], [0, 0, 20]]},
        "constraints": {"outward_normal": [0, 0, 1]},
    }
    frame = _fixed_site_surface_frame(plan)
    assert frame["u_cartesian_unit"] == pytest.approx([1, 0, 0])
    assert frame["v_cartesian_unit"] == pytest.approx([0, 1, 0])
    assert frame["outward_normal_cartesian_unit"] == pytest.approx([0, 0, 1])
    assert frame["periodic_fractional_axes"] == [0, 1]


def test_targeted_calibration_classification_has_three_reachable_branches():
    hard_gates = {"complete_sam_graph_unchanged": True, "calibration_residual_nontarget_audit_pass": True}
    initial = {
        "intramolecular_coordination": 4,
        "surface_o_contacts": 1,
        "total_coordination": 5,
    }
    stable = classify_targeted_calibration(True, hard_gates, 1.95, initial, initial)
    released = classify_targeted_calibration(
        True,
        hard_gates,
        2.21,
        initial,
        {"intramolecular_coordination": 4, "surface_o_contacts": 0, "total_coordination": 4},
    )
    failed = classify_targeted_calibration(False, hard_gates, 1.8, initial, initial)
    unknown_interval = classify_targeted_calibration(True, hard_gates, 2.1, initial, initial)
    hard_failed = classify_targeted_calibration(
        True, {**hard_gates, "complete_sam_graph_unchanged": False}, 1.8, initial, initial
    )
    assert stable["classification"] == "stable_extra_contact"
    assert released["classification"] == "released_repulsion"
    assert failed["classification"] == "failed_or_unknown"
    assert unknown_interval["classification"] == "failed_or_unknown"
    assert hard_failed["classification"] == "failed_or_unknown"


def test_targeted_plan_resolver_fails_closed_before_reserving_invalid_request(tmp_path):
    output = tmp_path / "must-not-be-created"
    with pytest.raises(ValueError, match="restricted to molecule index 62"):
        resolve_targeted_calibration_plan(
            tmp_path / "missing-trajectory",
            61,
            tmp_path / "missing-model",
            output,
        )
    assert not output.exists()


def test_targeted_hash_lock_rejects_mutated_bytes(tmp_path):
    path = tmp_path / "input.json"
    path.write_text("{}")
    record = _targeted_file_record(path, label="test input")
    path.write_text("{\"changed\":true}")
    with pytest.raises(ValueError, match="hash mismatch"):
        _targeted_file_record(path, expected_sha256=record["sha256"], label="test input")


def test_targeted_donor_mapping_resolves_full_to_working_atom_ids_and_elements():
    working_ids = list(range(1, 3841))[:2560]
    symbols = ["O"] * len(working_ids)
    for index in (572, 578, 583):
        symbols[index] = "In"
    step = {
        "target_donor_labels": ["O1", "O2", "O3"],
        "oxygen_permutation": [44, 45, 43],
        "registered_interface_bonds": [
            {"donor_label": "O1", "molecule_atom_index_0based": 44, "full_metal_atom_id_1based": 579, "actual_metal_element": "In"},
            {"donor_label": "O2", "molecule_atom_index_0based": 45, "full_metal_atom_id_1based": 584, "actual_metal_element": "In"},
            {"donor_label": "O3", "molecule_atom_index_0based": 43, "full_metal_atom_id_1based": 573, "actual_metal_element": "In"},
        ],
    }
    instance = {
        "site_instance_id": TARGETED_CALIBRATION_SITE_INSTANCE_ID,
        "parent_site_prototype_id": "site-prototype-phosphonic-acid-0014",
        "actual_metal_atom_ids_1based": [573, 579, 584],
        "actual_metals": [
            {"atom_id_1based": 573, "actual_element_after_doping": "In"},
            {"atom_id_1based": 579, "actual_element_after_doping": "In"},
            {"atom_id_1based": 584, "actual_element_after_doping": "In"},
        ],
        "donor_metal_mapping": [
            {"probe_donor_id": "O1", "metal_atom_id_1based": 579},
            {"probe_donor_id": "O2", "metal_atom_id_1based": 584},
            {"probe_donor_id": "O3", "metal_atom_id_1based": 573},
        ],
    }
    result = resolve_targeted_donor_mapping(step, instance, working_ids, symbols)
    assert [item["working_metal_atom_index_0based"] for item in result] == [578, 583, 572]
    instance["donor_metal_mapping"][0]["metal_atom_id_1based"] = 584
    with pytest.raises(ValueError, match="donor mapping mismatch"):
        resolve_targeted_donor_mapping(step, instance, working_ids, symbols)


def test_targeted_donor_element_is_instance_derived_and_mapping_fails_closed():
    working_ids = list(range(1, 3841))[:2560]
    symbols = ["O"] * len(working_ids)
    for index in (572, 578, 583):
        symbols[index] = "In"
    step = {
        "target_donor_labels": ["O1", "O2", "O3"],
        "oxygen_permutation": [44, 45, 43],
        "registered_interface_bonds": [
            {"donor_label": "O1", "molecule_atom_index_0based": 44, "full_metal_atom_id_1based": 579, "actual_metal_element": "Sn"},
            {"donor_label": "O2", "molecule_atom_index_0based": 45, "full_metal_atom_id_1based": 584, "actual_metal_element": "In"},
            {"donor_label": "O3", "molecule_atom_index_0based": 43, "full_metal_atom_id_1based": 573, "actual_metal_element": "In"},
        ],
    }
    instance = {
        "site_instance_id": TARGETED_CALIBRATION_SITE_INSTANCE_ID,
        "parent_site_prototype_id": "site-prototype-phosphonic-acid-0014",
        "actual_metal_atom_ids_1based": [573, 579, 584],
        "actual_metals": [
            {"atom_id_1based": 573, "actual_element_after_doping": "In"},
            {"atom_id_1based": 579, "actual_element_after_doping": "Sn"},
            {"atom_id_1based": 584, "actual_element_after_doping": "In"},
        ],
        "donor_metal_mapping": [
            {"probe_donor_id": "O1", "metal_atom_id_1based": 579},
            {"probe_donor_id": "O2", "metal_atom_id_1based": 584},
            {"probe_donor_id": "O3", "metal_atom_id_1based": 573},
        ],
    }
    symbols[578] = "Sn"
    result = resolve_targeted_donor_mapping(step, instance, working_ids, symbols)
    assert result[0]["metal_element"] == "Sn"
    step["registered_interface_bonds"][0]["actual_metal_element"] = "In"
    with pytest.raises(ValueError, match="element mismatch"):
        resolve_targeted_donor_mapping(step, instance, working_ids, symbols)


def test_targeted_proton_parent_identity_fails_closed():
    import json
    from ase.build import make_supercell
    from ase.io import read

    root = next(
        (parent for parent in Path(__file__).resolve().parents
         if (parent / "substrates").is_dir()),
        None,
    )
    if root is None:
        pytest.skip("project substrate fixtures are external to the AWF snapshot")
    prototype_path = root / "substrates/structures/ito/adsorption-sites/phosphonic_acid/adsorption_site_prototype_clustering_v1-10d5c9fed77c/site-prototypes.json"
    prototype_library = json.loads(prototype_path.read_text())
    prototype = next(item for item in prototype_library["site_prototypes"] if item["site_prototype_id"] == "site-prototype-phosphonic-acid-0014")
    prototype["surface_proton_policy"]["selected_assignments"][0]["nearest_oxygen_is_substrate"] = False
    representative = read(prototype_path.parent / "structures/site-prototype-phosphonic-acid-0014.extxyz")
    source = read(root / "substrates/structures/ito/slabs/111/stoichiometric/maintained/relaxed-surface-v1/In2O3-Ia-3-111-three-repeat-O-terminated-relaxed-mace-mpa-0-medium-v1.cif")
    build_manifest = json.loads((root / "runs/site-instance-materialization-validation/pubchem-cid-13818-3d-ito/substrate-2f8decd47195/substrate-build-manifest.json").read_text())
    full = make_supercell(source, np.asarray(build_manifest["supercell"]["matrix"], dtype=int), wrap=True)
    layer_groups = json.loads((root / "runs/site-instance-materialization-validation/pubchem-cid-13818-3d-ito/substrate-2f8decd47195/layer-groups.json").read_text())
    with pytest.raises(ValueError, match="not declared as substrate O"):
        map_targeted_surface_protons(
            representative,
            prototype,
            source,
            full,
            [2, 0, 0],
            layer_groups["working_layer_atom_ids_1based"],
            prototype_library["source_surface"],
            prototype["surface_frame"],
        )


def test_targeted_input_keeps_only_selected_block_and_two_h_atoms():
    from ase import Atoms

    substrate = Atoms(
        symbols=["O"] * 2560,
        positions=np.zeros((2560, 3)),
        cell=[100, 100, 40],
        pbc=True,
    )
    molecule_symbols = list("PC" + "C" * 7 + "N" + "C" * 14 + "S" + "H" * 18 + "O" * 3)
    assert len(molecule_symbols) == 46
    blocks = Atoms(
        symbols=molecule_symbols * 68,
        positions=np.zeros((46 * 68, 3)),
        cell=substrate.cell,
        pbc=True,
    )
    final_h0 = substrate + blocks
    records = [
        {"working_parent_atom_index_0based": 0, "parent_to_h_vector_A": [0, 0, 1]},
        {"working_parent_atom_index_0based": 1, "parent_to_h_vector_A": [0, 0, 1]},
    ]
    target = build_targeted_calibration_initial_structure(final_h0, 62, records)
    assert len(target) == 2608
    assert target[2560:2606].get_chemical_symbols() == molecule_symbols
    assert target[2606:].get_chemical_symbols() == ["H", "H"]
    assert target[0:2560].get_chemical_symbols() == ["O"] * 2560


def test_targeted_p_o_coordination_reports_total_p_c_plus_three_p_o_plus_surface_o():
    from ase import Atoms

    atoms = Atoms(
        symbols=["P", "C", "O", "O", "O", "O"],
        positions=[
            [0, 0, 2], [1.5, 0, 2], [0, 1.5, 2], [0, -1.5, 2],
            [0, 0, 0.5], [0, 0, 0.228],
        ],
        cell=[30, 30, 30],
        pbc=True,
    )
    coordination = targeted_p_o_coordination(
        atoms,
        0,
        periodic_axes=(0, 1),
        molecule_indices=[0, 1, 2, 3, 4],
        substrate_indices=[5],
    )
    assert coordination["intramolecular_coordination"] == 4
    assert len(coordination["intramolecular_p_o_contacts"]) == 3
    assert coordination["surface_o_contacts"] == 1
    assert coordination["total_coordination"] == 5
    assert coordination["cutoff_A"] == pytest.approx(2.0)
    contacts = _targeted_interface_covalent_contacts(
        atoms, substrate_indices=[5], molecule_indices=[0], periodic_axes=(0, 1)
    )
    assert contacts
    # The helper's formal interface uses two surface axes; an empty tuple is
    # intentionally rejected rather than silently wrapping the slab normal.
    with pytest.raises(ValueError):
        _targeted_interface_covalent_contacts(
            atoms, substrate_indices=[5], molecule_indices=[0], periodic_axes=()
        )


def test_raw_production_failure_target_only_is_diagnostic_but_residual_contact_is_hard():
    target = {
        "molecule_atom_id": 0,
        "substrate_atom_id": 1477,
        "molecule_element": "P",
        "substrate_element": "O",
        "distance_A": 1.772,
    }
    extra = {
        "molecule_atom_id": 43,
        "substrate_atom_id": 1477,
        "molecule_element": "O",
        "substrate_element": "O",
        "distance_A": 1.9,
    }
    raw_only_target = {"passed": False, "collisions": [target], "mapped_bonds": []}
    only_target = _targeted_calibration_residual_nontarget_audit(raw_only_target, 0, 1477)
    assert raw_only_target["passed"] is False
    assert only_target["passed"] is True
    assert only_target["target_pair_short_contacts"] == [target]
    stable = classify_targeted_calibration(
        True,
        {"complete_sam_graph_unchanged": True, "calibration_residual_nontarget_audit_pass": only_target["passed"]},
        1.95,
        {"intramolecular_coordination": 4, "surface_o_contacts": 1, "total_coordination": 5},
        {"intramolecular_coordination": 4, "surface_o_contacts": 1, "total_coordination": 5},
    )
    assert stable["classification"] == "stable_extra_contact"
    assert raw_only_target["passed"] is False  # production_pass remains false
    residual = _targeted_calibration_residual_nontarget_audit(
        {"passed": False, "collisions": [target, extra], "mapped_bonds": []}, 0, 1477
    )
    assert residual["passed"] is False
    assert residual["non_target_short_contact_count"] == 1


def test_surface_frame_is_resolved_from_rotated_sealed_evidence():
    frame = _targeted_surface_frame_from_evidence(
        prototype={
            "surface_frame": {
                "periodic_fractional_axes": [0, 2],
                "u_cartesian_unit": [0, 0, 1],
                "v_cartesian_unit": [1, 0, 0],
                "outward_normal_cartesian_unit": [0, 1, 0],
            }
        },
        layer_groups={"normal_axis": 1},
    )
    assert frame["periodic_fractional_axes"] == [0, 2]
    assert frame["normal_axis"] == 1
    assert frame["u_cartesian_unit"] == pytest.approx([0, 0, 1])
    with pytest.raises(ValueError, match="Surface Frame"):
        _targeted_surface_frame_from_evidence(
            prototype={"surface_frame": {"periodic_fractional_axes": [0, 1]}},
            layer_groups={"normal_axis": 2},
        )


def _validation_fixture_plan_and_atoms():
    from ase import Atoms

    substrate_count = 2560
    molecule_symbols = list("PC" + "C" * 7 + "N" + "C" * 14 + "S" + "H" * 18 + "O" * 3)
    symbols = ["O"] * (substrate_count + len(molecule_symbols) + 2)
    positions = np.zeros((len(symbols), 3), dtype=float)
    for index in range(substrate_count):
        positions[index] = [(index % 20) * 4.0, ((index // 20) % 20) * 4.0, 10.0 + ((index // 400) % 5) * 4.0]
    positions[0] = [0.0, 0.0, 0.0]
    positions[1] = [10.0, 0.0, 0.0]
    metal_indices = [572, 578, 583]
    for index in metal_indices:
        symbols[index] = "In"
    molecule_start = substrate_count
    symbols[molecule_start:molecule_start + len(molecule_symbols)] = molecule_symbols
    positions[molecule_start + 0] = [50.0, 50.0, 50.0]
    positions[molecule_start + 1] = [51.5, 50.0, 50.0]
    positions[molecule_start + 43] = [50.0, 51.5, 50.0]
    positions[molecule_start + 44] = [48.5, 50.0, 50.0]
    positions[molecule_start + 45] = [50.0, 48.5, 50.0]
    for local in list(range(2, 43)) + list(range(10, 25)):
        if local in {43, 44, 45}:
            continue
        positions[molecule_start + local] = [70.0 + 0.3 * local, 70.0, 70.0]
    positions[572] = [50.0, 51.5, 52.0]
    positions[578] = [48.5, 50.0, 52.0]
    positions[583] = [50.0, 48.5, 52.0]
    positions[1477] = [50.0, 50.0, 48.228]
    positions[-2] = [0.0, 0.0, 1.0]
    positions[-1] = [10.0, 0.0, 1.0]
    atoms = Atoms(symbols=symbols, positions=positions, cell=[100, 100, 100], pbc=True)
    edges = scan._sam_edge_set(
        atoms,
        range(molecule_start, molecule_start + len(molecule_symbols)),
        scale=1.25,
    )
    frame = {
        "periodic_fractional_axes": [0, 1],
        "normal_axis": 2,
        "u_cartesian_unit": [1.0, 0.0, 0.0],
        "v_cartesian_unit": [0.0, 1.0, 0.0],
        "outward_normal_cartesian_unit": [0.0, 0.0, 1.0],
    }
    mapping = [
        {"donor_label": "O1", "molecule_atom_index_0based": 44, "working_metal_atom_index_0based": 578, "expected_metal_element_from_instance": "In", "metal_element": "In", "distance_window_A": [1.7, 2.8]},
        {"donor_label": "O2", "molecule_atom_index_0based": 45, "working_metal_atom_index_0based": 583, "expected_metal_element_from_instance": "In", "metal_element": "In", "distance_window_A": [1.7, 2.8]},
        {"donor_label": "O3", "molecule_atom_index_0based": 43, "working_metal_atom_index_0based": 572, "expected_metal_element_from_instance": "In", "metal_element": "In", "distance_window_A": [1.7, 2.8]},
    ]
    plan = {
        "system": {
            "substrate_atom_count": substrate_count,
            "molecule_atom_ids_0based": list(range(molecule_start, molecule_start + len(molecule_symbols))),
            "surface_h_atom_ids_0based": [2606, 2607],
            "expected_atom_count": 2608,
            "surface_proton_records": [
                {"working_parent_atom_index_0based": 0},
                {"working_parent_atom_index_0based": 1},
            ],
            "donor_mapping": mapping,
            "target_p_molecule_local_index_0based": 0,
            "target_surface_oxygen_atom_index_0based": 1477,
            "initial_sam_covalent_edges_global_1based": edges,
            "surface_frame": frame,
            "normal_axis": 2,
        },
        "constraints": {
            "frozen_substrate_indices_0based": list(range(1280)),
            "movable_substrate_indices_0based": list(range(1280, substrate_count)),
            "movable_atom_indices_0based": list(range(1280, 2608)),
        },
        "parameters": {
            "cell_max_abs_change_A": 1.0e-8,
            "frozen_max_displacement_A": 1.0e-8,
            "covalent_radius_scale": 1.25,
            "surface_h_parent_max_distance_A": 1.25,
            "p_o_coordination_cutoff_A": 2.0,
            "max_steps": 500,
            "fmax_eV_per_A": 0.03,
        },
    }
    return atoms, plan


def test_validation_keeps_raw_production_boundary_separate_from_calibration(monkeypatch):
    atoms, plan = _validation_fixture_plan_and_atoms()
    target = {"molecule_atom_id": 0, "substrate_atom_id": 1477, "distance_A": 1.772}
    raw_target_only = {"passed": False, "collisions": [target], "mapped_bonds": [], "periodic_axes": [0, 1]}
    monkeypatch.setattr(scan, "_targeted_strict_vdw_audit", lambda *args: dict(raw_target_only))
    fake_contacts = {
        (572, 2600 + 43): {"distance_A": 2.0, "cutoff_A": 2.2},
        (578, 2600 + 44): {"distance_A": 2.0, "cutoff_A": 2.2},
        (583, 2600 + 45): {"distance_A": 2.0, "cutoff_A": 2.2},
    }
    monkeypatch.setattr(scan, "_targeted_interface_covalent_contacts", lambda *args, **kwargs: fake_contacts)
    stable = validate_targeted_calibration_final_geometry(
        atoms, atoms.copy(), plan, np.zeros((len(atoms), 3)), 0, True
    )
    assert stable["classification"] == "stable_extra_contact"
    assert stable["production_pass"] is False
    assert "initial_strict_vdw_gate" not in stable["hard_gates"]
    extra = {**target, "molecule_atom_id": 1}
    monkeypatch.setattr(
        scan,
        "_targeted_strict_vdw_audit",
        lambda *args: {"passed": False, "collisions": [target, extra], "mapped_bonds": [], "periodic_axes": [0, 1]},
    )
    unknown = validate_targeted_calibration_final_geometry(
        atoms, atoms.copy(), plan, np.zeros((len(atoms), 3)), 0, True
    )
    assert unknown["classification"] == "failed_or_unknown"
    assert unknown["hard_gates"]["calibration_residual_nontarget_audit_pass"] is False


def test_targeted_validator_uses_plan_constraint_key_and_rejects_legacy_key():
    atoms, plan = _validation_fixture_plan_and_atoms()
    assert plan["constraints"]["frozen_substrate_indices_0based"] == list(range(1280))
    assert "frozen_atom_indices_0based" not in plan["constraints"]

    legacy_constraints = dict(plan["constraints"])
    legacy_constraints["frozen_atom_indices_0based"] = legacy_constraints.pop(
        "frozen_substrate_indices_0based"
    )
    legacy_plan = dict(plan)
    legacy_plan["constraints"] = legacy_constraints
    with pytest.raises(KeyError, match="frozen_substrate_indices_0based"):
        validate_targeted_calibration_final_geometry(
            atoms,
            atoms.copy(),
            legacy_plan,
            np.zeros((len(atoms), 3)),
            0,
            True,
            initial_strict_vdw={"passed": True, "collisions": [], "mapped_bonds": []},
            final_strict_vdw={"passed": True, "collisions": [], "mapped_bonds": []},
        )


def test_targeted_identity_material_uses_project_relative_or_content_hash_paths(tmp_path):
    root = tmp_path / "project"
    artifact = root / "evidence" / "record.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("sealed")
    normalized = _targeted_identity_value(
        {"path": str(artifact), "sha256": "locked", "bytes": 6}, root
    )
    assert "path" not in normalized
    assert normalized["project_relative_path"] == "evidence/record.json"
    assert str(root) not in repr(normalized)


def test_source_surface_contract_comes_from_hash_locked_evidence():
    contract = _targeted_source_surface_contract(
        {"formula": {"X": 2, "Y": 1}, "atom_count": 3, "sha256": "content-lock"}
    )
    assert contract == {"atom_count": 3, "formula": {"X": 2, "Y": 1}}
    with pytest.raises(ValueError, match="does not match"):
        _targeted_source_surface_contract({"formula": {"X": 2}, "atom_count": 3})


def test_target_surface_oxygen_reports_the_sealed_audit_id_convention():
    target = {
        "molecule_atom_id": 0,
        "substrate_atom_id": 1477,
        "molecule_element": "P",
        "substrate_element": "O",
        "distance_A": 1.772,
    }
    validation = {
        "final_interface_audit": {
            "schema": "sam-sequential-final-registered-interface-audit-v1",
            "substrate_atom_count": 2560,
            "molecule_audits": [
                {
                    "molecule_number_1based": 62,
                    "site_instance_id": TARGETED_CALIBRATION_SITE_INSTANCE_ID,
                    "global_atom_range_0based": [5366, 5411],
                    "collisions": [target],
                    "mapped_bonds": [],
                }
            ],
        }
    }
    target_index, evidence = scan._targeted_resolve_target_surface_oxygen(
        validation, TARGETED_CALIBRATION_SITE_INSTANCE_ID, 62, 0
    )
    assert target_index == 1477
    assert evidence["sealed_audit_id_convention"] == scan.TARGETED_CALIBRATION_SEALED_AUDIT_ID_CONVENTION


def test_target_surface_oxygen_fails_closed_on_global_audit_id_mismatch():
    target = {
        "molecule_atom_id": 0,
        "substrate_atom_id": 2560 + 1477,
        "molecule_element": "P",
        "substrate_element": "O",
        "distance_A": 1.772,
    }
    validation = {
        "final_interface_audit": {
            "schema": "sam-sequential-final-registered-interface-audit-v1",
            "substrate_atom_count": 2560,
            "molecule_audits": [
                {
                    "molecule_number_1based": 62,
                    "site_instance_id": TARGETED_CALIBRATION_SITE_INSTANCE_ID,
                    "global_atom_range_0based": [5366, 5411],
                    "collisions": [target],
                    "mapped_bonds": [],
                }
            ],
        }
    }
    with pytest.raises(ValueError, match="sealed audit ID convention mismatch"):
        scan._targeted_resolve_target_surface_oxygen(
            validation, TARGETED_CALIBRATION_SITE_INSTANCE_ID, 62, 0
        )


def _pso_sealed_plan_fixture():
    from ase import Atoms

    symbols = ["O", "O", "O", "O", "In", "In", "In", "O", "P", "C", "O", "O", "O", "H", "H"]
    positions = np.array([
        [4.0, 4.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0],
        [0.0, 0.0, 1.5], [5.0, 0.0, 0.0], [5.0, 2.0, 0.0],
        [5.0, 4.0, 0.0], [8.0, 8.0, 0.0], [0.0, 0.0, 4.0],
        [1.5, 0.0, 4.0], [0.0, 1.5, 4.0], [0.0, -1.5, 4.0],
        [0.0, 0.0, 2.5], [0.0, 0.0, 1.0], [2.0, 0.0, 1.0],
    ])
    baseline = Atoms(symbols=symbols, positions=positions, cell=[20.0, 20.0, 20.0], pbc=True)
    contract = {
        "schema": "sam-p-surface-o-experimental-contract-v3",
        "schema_version": 3,
        "model": {"sha256": "a" * 64, "declared_elements": ["C", "H", "In", "O", "P"]},
        "surface_frame": {
            "periodic_fractional_axes": [0, 1], "normal_axis": 2,
            "u_cartesian_unit": [1.0, 0.0, 0.0], "v_cartesian_unit": [0.0, 1.0, 0.0],
            "outward_normal_cartesian_unit": [0.0, 0.0, 1.0],
        },
        "system": {
            "substrate_atom_count": 8,
            "molecule_indices_0based": [8, 9, 10, 11, 12],
            "surface_h_indices_0based": [13, 14],
            "surface_h_parent_indices_0based": [1, 2],
            "total_atom_count": 15,
            "molecule_formula": {"C": 1, "O": 3, "P": 1},
            "cell_A": np.asarray(baseline.cell).tolist(), "pbc": [True, True, True],
            "frozen_indices_0based": [0, 1, 2, 3],
            "movable_indices_0based": list(range(4, 15)),
            "target_p_global_index_0based": 8,
            "target_p_molecule_local_index_0based": 0,
            "target_surface_oxygen_working_index_0based": 3,
            "target_surface_oxygen_metal_indices_0based": [4, 5, 6],
            "registered_donor_mappings": [
                {"donor_label": "O1", "molecule_local_index_0based": 2, "working_metal_index_0based": 4, "expected_metal_element": "In", "distance_window_A": [1.0, 8.0]},
                {"donor_label": "O2", "molecule_local_index_0based": 3, "working_metal_index_0based": 5, "expected_metal_element": "In", "distance_window_A": [1.0, 8.0]},
                {"donor_label": "O3", "molecule_local_index_0based": 4, "working_metal_index_0based": 6, "expected_metal_element": "In", "distance_window_A": [1.0, 8.0]},
            ],
        },
        "contract": {
            "required_model_sha256": "a" * 64, "device": "cuda", "default_dtype": "float32",
            "covalent_radius_scale": 1.25, "strict_vdw_radius_scale": 0.85,
            "strict_vdw_experimental_exemption": {"global_molecule_atom_index_0based": 8, "working_substrate_atom_index_0based": 3},
            "p_o_coordination_cutoff_A": 2.0, "target_stable_cutoff_A": 2.0, "target_released_cutoff_A": 2.2,
            "minimum_total_p_coordination_for_stable": 5, "normal_intramolecular_coordination_for_released": 4,
            "surface_h_parent_max_distance_A": 1.25, "fmax_eV_per_A": 0.03, "maxstep_A": 0.05,
            "max_steps": 500, "frozen_max_displacement_A": 1.0e-8, "rigidity_tolerance_A": 1.0e-8,
            "design_distance_tolerance_A": 1.0e-8,
        },
        "candidates": [
            {"candidate_id": "P1", "type": "rigid_translation", "translation_A": [0.0, 0.0, 0.10]},
            {"candidate_id": "P2", "type": "rigid_rotation", "axis_cartesian": [0.0, 0.0, 1.0], "angle_deg": 5.0},
            {"candidate_id": "C1", "type": "rigid_translation_to_target_distance", "axis_cartesian": [0.0, 0.0, 1.0], "target_distance_A": 2.50},
        ],
    }
    return None, {"contract": contract}, baseline


def test_pso_pure_transforms_move_the_complete_molecule_and_preserve_rigidity():
    package, plan, baseline = _pso_sealed_plan_fixture()
    contract = plan["contract"]
    molecule = np.asarray(contract["system"]["molecule_indices_0based"], dtype=int)
    p_index = int(contract["system"]["target_p_global_index_0based"])
    target_o = int(contract["system"]["target_surface_oxygen_working_index_0based"])

    p1, p1_record, _ = scan._pso_apply_candidate_transform(
        baseline, contract, plan["contract"]["candidates"][0]
    )
    translation = p1.positions[molecule] - baseline.positions[molecule]
    assert np.allclose(translation, [0.0, 0.0, 0.10], atol=1.0e-10)
    assert p1_record["whole_molecule_transformed"] is True
    assert p1_record["single_atom_P_move"] is False
    assert p1_record["rigidity_max_pair_distance_error_A"] < 1.0e-8

    p2, p2_record, _ = scan._pso_apply_candidate_transform(
        baseline, contract, plan["contract"]["candidates"][1]
    )
    assert np.allclose(p2.positions[p_index], baseline.positions[p_index], atol=1.0e-10)
    assert p2_record["design"]["angle_deg"] == pytest.approx(5.0)
    for left in molecule:
        for right in molecule:
            assert p2.get_distance(int(left), int(right), mic=False) == pytest.approx(
                baseline.get_distance(int(left), int(right), mic=False), abs=1.0e-8
            )

    c1, c1_record, _ = scan._pso_apply_candidate_transform(
        baseline, contract, plan["contract"]["candidates"][2]
    )
    assert scan._targeted_distance(
        c1, p_index, target_o, periodic_axes=(0, 1)
    ) == pytest.approx(2.50, abs=1.0e-8)
    assert c1_record["rigidity_max_pair_distance_error_A"] < 1.0e-8
    assert np.allclose(
        c1.positions[molecule] - baseline.positions[molecule],
        c1.positions[molecule[0]] - baseline.positions[molecule[0]],
        atol=1.0e-8,
    )


def test_pso_unique_strict_vdw_exemption_filters_only_local_target_pair():
    package, plan, baseline = _pso_sealed_plan_fixture()
    raw = {
        "passed": False,
        "collisions": [
            {"molecule_atom_id": 0, "substrate_atom_id": 3, "distance_A": 1.8},
            {"molecule_atom_id": 2, "substrate_atom_id": 3, "distance_A": 1.9},
        ],
        "mapped_bonds": [],
    }
    residual = scan._pso_target_residual_audit(raw, plan["contract"])
    assert residual["target_pair_filter_count"] == 1
    assert residual["nonregistered_short_contact_count"] == 1
    assert residual["nonregistered_short_contacts"][0]["molecule_atom_id"] == 2
    assert residual["only_experimental_strict_vdw_filter_exemption"] is True
    assert residual["production_raw_audit_modified"] is False


def test_pso_hash_fail_close_and_contract_exemption_validation(tmp_path):
    artifact = tmp_path / "locked.json"
    artifact.write_text("original")
    record = scan._targeted_file_record(artifact, label="v3 test artifact")
    artifact.write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        scan._targeted_file_record(
            artifact, expected_sha256=record["sha256"], label="v3 test artifact"
        )

    package, plan, baseline = _pso_sealed_plan_fixture()
    bad = json.loads(json.dumps(plan["contract"]))
    bad["contract"]["strict_vdw_experimental_exemption"]["working_substrate_atom_index_0based"] = 1478
    with pytest.raises(ValueError, match="strict-vdW exemption"):
        scan._pso_validate_config_and_baseline(bad, baseline)


def test_pso_audit_dependency_lock_rejects_mutated_snapshot(tmp_path):
    lock = scan._pso_resolve_audit_dependency()
    assert lock["loaded_path"].endswith("scripts/sam_structure_tools.py")
    assert lock["whole_file_sha256"] == lock["sha256"]
    assert lock["function"]["name"] == "periodic_vdw_collision_audit"
    assert lock["function"]["source_sha256"]

    package = tmp_path / "package"
    snapshot_dir = package / "code-snapshot"
    snapshot_dir.mkdir(parents=True)
    snapshot = snapshot_dir / "sam_structure_tools.py"
    snapshot.write_bytes(Path(lock["loaded_path"]).read_bytes())
    locked = {**lock, "snapshot_path": "code-snapshot/sam_structure_tools.py"}
    locked["snapshot_sha256"] = scan._sha256(snapshot)
    locked["snapshot_bytes"] = snapshot.stat().st_size
    scan._pso_verify_audit_dependency(locked, package_root=package)

    snapshot.write_bytes(snapshot.read_bytes() + b"\\n# dependency mutation\\n")
    with pytest.raises(ValueError, match="snapshot hash mismatch"):
        scan._pso_verify_audit_dependency(locked, package_root=package)


def test_pso_new_contract_explicitly_forbids_prototype0001():
    contract = {
        "prototype0001_deferred": True,
        "forbidden_candidates": scan._pso_forbidden_candidates(),
        "candidates": [{"candidate_id": "P1"}, {"candidate_id": "P2"}, {"candidate_id": "C1"}],
    }
    scan._pso_require_new_contract_declarations(contract)
    bad = json.loads(json.dumps(contract))
    bad["candidates"].append({"candidate_id": "prototype0001"})
    with pytest.raises(ValueError, match="forbidden"):
        scan._pso_require_new_contract_declarations(bad)


def test_pso_analysis_reports_only_strictly_comparable_relative_total_energies():
    root = next(
        (parent for parent in Path(__file__).resolve().parents if (parent / "runs").is_dir()),
        None,
    )
    if root is None:
        pytest.skip("project run fixtures are external to the AWF snapshot")
    package = root / "runs/targeted-fixed-site-calibration-p-o-contract-f6c405a4"
    if not (package / "analysis/analysis.json").is_file():
        pytest.skip("sealed v3 execution package is not present in this workspace")
    analysis = json.loads((package / "analysis/analysis.json").read_text())
    assert analysis["comparability"]["relative_energy_eligible"] is True
    assert analysis["comparability"]["strict_same_composition"] is True
    assert analysis["comparability"]["same_cell"] is True
    assert analysis["comparability"]["same_model"] is True
    assert analysis["comparability"]["same_dtype"] is True
    assert analysis["comparability"]["same_frozen_set"] is True
    assert analysis["comparability"]["same_two_surface_H_inventory"] is True
    rows = analysis["candidate_results"]
    assert [row["candidate_id"] for row in rows] == ["P1", "P2", "C1"]
    assert all(row["relative_total_energy_eV"] is not None for row in rows)
    assert analysis["comparability"]["adsorption_energy"] is False
    assert analysis["comparability"]["mpa_reference_subtraction"] is False


def test_precomputed_headgroup_alignment_preserves_identity(tmp_path, monkeypatch):
    """Synthetic distant substrate tests handoff/alignment, not adsorption physics."""
    import json
    from types import SimpleNamespace
    from ase import Atoms
    from ase.io import read
    import mace_conformation_scan as owner
    import sam_structure_tools as geometry
    from substrate_library import _molecular_adjacency
    import numpy as np
    mol=Atoms('POOOC',positions=[[0,0,0],[1.45,0,-.5],[-.72,1.25,-.5],[-.72,-1.25,-.5],[0,0,1.8]])
    mol.set_cell([30,30,30]);mol.set_pbc([True,True,False])
    substrate=Atoms('In',positions=[[15,15,0]],cell=mol.cell,pbc=mol.pbc)
    graph=_molecular_adjacency(mol.get_chemical_symbols(),mol.positions,np.asarray(mol.cell),np.asarray(mol.pbc))
    prepared={'sam':mol,'headgroup_atom_ids_0based':[0,1,2,3],
        'headgroup_target_positions_A':mol.positions[:4].copy(),'substrate_with_h':substrate,
        'molecular_adjacency':graph,'mapped_windows':{},'radii_A':{},'surface_frame':{'periodic_fractional_axes':[0,1]},
        'initial_collision_audit':{'passed':True},'metadata':{'test':'synthetic'}}
    monkeypatch.setattr(owner,'_prepare_fixed_site_system',lambda args:prepared)
    monkeypatch.setattr(geometry,'periodic_vdw_collision_audit',lambda **kw:{'passed':True})
    rotation=np.array([[0,-1,0],[1,0,0],[0,0,1]])
    ensemble=tmp_path/'seeds.json'
    transformed=mol.positions@rotation+[4,7,2]
    bad=transformed.copy();bad[4]+=20
    selected_records = [
        {'candidate_id':'skeleton-1__pc-roll-0deg','positions_A':transformed.tolist()},
        {'candidate_id':'skeleton-2__pc-roll-30deg','positions_A':bad.tolist()},
    ]
    selection_path=tmp_path/'selection.json'
    ensemble.write_text(json.dumps({
        'schema':'pvksam-fixed-site-candidate-ensemble-v1',
        'candidate_source':'one lowest-MMFF94s-single-point-energy surface-safe P-C roll per skeleton',
        'symbols':mol.get_chemical_symbols(),'records':selected_records,
        'provenance':{
            'selection_manifest':selection_path.name,
            'selection_method':'minimum MMFF94s single-point energy among CPU-screen-passing rolls, independently per skeleton',
        },
    }))
    selection_path.write_text(json.dumps({
        'schema':'pvksam-single-pc-roll-selection-v1','status':'passed',
        'outputs':{'selected_candidate_ensemble':str(ensemble.resolve())},
        'selected_candidate_ensemble_sha256':owner._sha256(ensemble),
        'selected':[{'candidate_id':row['candidate_id']} for row in selected_records],
        'summary':{'selected_roll_count':len(selected_records)},
    }))
    args=SimpleNamespace(sam=None,candidate_ensemble=ensemble,output_root=tmp_path/'scan',
        donor_assignment='O1:2,O2:3,O3:4',site_prototype_id='synthetic',site_cell=1,
        vdw_radius_scale=.85,headgroup_alignment_max_rmsd_A=.5,
        organic_z_floor_at_anchor_oxygen_mean=True)
    assert owner.run_fixed_site_cpu_screen(args)==0
    manifest=next(args.output_root.glob('*/01_cpu_screen/manifest.json'))
    data=json.loads(manifest.read_text());assert data['summary']['collision_safe_count']==1
    assert data['summary']['generation_failed_count']==1
    assert data['parameters']['candidate_ensemble']['candidate_source']=='one lowest-MMFF94s-single-point-energy surface-safe P-C roll per skeleton'
    row=data['records'][0];a=read(manifest.parent/row['structure'])
    assert a[1:].get_chemical_symbols()==mol.get_chemical_symbols()
    assert np.allclose(a.positions[1:],mol.positions,atol=1e-7)
    assert data['records'][1]['reason']=='candidate_alignment_or_geometry_failed'


def test_fixed_site_cpu_rejects_direct_torsion_grid_without_roll_ensemble():
    from types import SimpleNamespace
    import mace_conformation_scan as owner

    args = SimpleNamespace(candidate_ensemble=None)
    with pytest.raises(ValueError, match='direct fixed-site torsion-grid generation has been removed'):
        owner.run_fixed_site_cpu_screen(args)


def test_legacy_216_start_scan_mode_is_removed(monkeypatch):
    import sys
    import mace_conformation_scan as owner

    monkeypatch.setattr(sys, 'argv', ['mace_conformation_scan.py', '--mode', 'legacy'])
    with pytest.raises(SystemExit) as exc:
        owner.main()
    assert exc.value.code == 2


def test_batched_dihedral_rotation_matches_ase_set_dihedral():
    """The skeleton-grid rotation helper preserves ASE candidate geometry."""
    import numpy as np
    from ase import Atoms
    import mace_conformation_scan as owner

    molecule = Atoms(
        "C6",
        positions=[
            [0.0, 0.0, 0.0],
            [1.5, 0.1, 0.2],
            [2.8, 1.0, -0.1],
            [4.1, 1.4, 1.1],
            [5.3, 0.4, 1.6],
            [6.7, -0.2, 0.8],
        ],
    )
    torsions = [(0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5)]
    moving = [[2, 3, 4, 5], [3, 4, 5], [4, 5]]
    targets = np.asarray([[30.0, 60.0, -90.0], [150.0, -30.0, 120.0]])
    batched = np.repeat(molecule.positions[None, :, :], len(targets), axis=0)
    for column, (torsion, indices) in enumerate(zip(torsions, moving)):
        current = owner._batch_dihedral_degrees(batched, torsion)
        owner._batch_rotate_about_bond(
            batched,
            torsion[1],
            torsion[2],
            indices,
            np.deg2rad(targets[:, column] - current),
        )
    for row_index, row in enumerate(targets):
        reference = molecule.copy()
        for torsion, angle, indices in zip(torsions, row, moving):
            reference.set_dihedral(*torsion, float(angle), indices=indices)
        assert np.max(np.abs(batched[row_index] - reference.positions)) < 1.0e-10


def _write_synthetic_phosphonate_skeleton(tmp_path, *, clashing_hydrogens=False):
    """A noncollinear PO3 head and acyclic C4-H body, without real-run inputs."""
    from ase import Atoms
    from ase.io import write

    molecule = Atoms(
        "POOOCCCCH",
        positions=[
            [0.0, 0.0, 0.5],
            [1.45, 0.0, 0.0],
            [-0.725, 1.2557, 0.0],
            [-0.725, -1.2557, 0.0],
            [0.0, 0.0, 2.3],
            [1.42, 0.0, 2.88],
            [1.9, 1.4, 2.9],
            [3.0, 1.8, 3.8],
            [3.72, 2.18, 3.14],
        ],
    )
    if clashing_hydrogens:
        # Each H is bonded to C4, but this H-H separation is 1.174 A:
        # below the 1.2 A clash gate and above covalent-bond inference range.
        molecule += Atoms("H", positions=[[3.72, 1.02, 3.32]])
    # The input is deliberately outside the standard frame; one proper rigid
    # transform must recover the plane while preserving molecular distances.
    molecule.rotate(37.0, [1.0, 2.0, 3.0])
    molecule.translate([7.0, -4.0, 11.0])
    source = tmp_path / "synthetic-h0.extxyz"
    write(source, molecule)
    return source, molecule


def _forbid_skeleton_optimization(monkeypatch):
    """Guard the stage boundary, including an accidental lazy GPU import."""
    import builtins
    from ase.optimize.optimize import Optimizer

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"torch", "mace"}:
            raise AssertionError("CPU skeleton screening imported a GPU/model runtime")
        return original_import(name, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU skeleton screening entered optimization or site preparation")

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(Optimizer, "__init__", forbidden)
    monkeypatch.setattr(scan, "_load_torch_for_mace_worker", forbidden)
    monkeypatch.setattr(scan, "_prepare_fixed_site_system", forbidden)


def _run_synthetic_skeleton_screen(tmp_path, monkeypatch, **overrides):
    from types import SimpleNamespace

    clashing_hydrogens = overrides.pop("clashing_hydrogens", False)
    source, molecule = _write_synthetic_phosphonate_skeleton(
        tmp_path, clashing_hydrogens=clashing_hydrogens
    )
    _forbid_skeleton_optimization(monkeypatch)
    args = SimpleNamespace(
        sam=source,
        output_root=tmp_path / "screen",
        skeleton_isomer_name="synthetic",
        # O-P-C1-C2 changes only axial phase and must not add a grid dimension.
        torsion_atom_ids=[[2, 1, 5, 6], [1, 5, 6, 7], [5, 6, 7, 8]],
        scan_step_deg=30.0,
        full_grid_batch_size=7,
        scan_max_candidates=144,
        skeleton_limit=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    assert scan.run_phosphonate_skeleton_cpu_screen(args) == 0
    manifest_path = next(args.output_root.glob("*/manifest.json"))
    manifest = json.loads(manifest_path.read_text())
    outcomes = [
        json.loads(line)
        for line in Path(manifest["candidate_outcomes"]).read_text().splitlines()
    ]
    return manifest_path, manifest, outcomes, molecule


def test_phosphonate_skeleton_screen_complete_grid_preserves_plane_and_all_atom_safety(
    tmp_path, monkeypatch
):
    from ase.io import read

    manifest_path, manifest, outcomes, original = _run_synthetic_skeleton_screen(
        tmp_path, monkeypatch
    )
    parameters = manifest["parameters"]
    summary = manifest["summary"]
    assert parameters["scan_angles_deg"] == list(range(-180, 180, 30))
    assert len(set(value % 360 for value in parameters["scan_angles_deg"])) == 12
    assert parameters["torsion_atom_ids_1based"] == [[1, 5, 6, 7], [5, 6, 7, 8]]
    assert parameters["omitted_pc_phase_torsions_1based"] == [[2, 1, 5, 6]]
    assert parameters["height_scope"] == "all_C_side_atoms_including_H_excluding_PO3"
    assert manifest["status"] == "screen_completed"
    assert manifest["grid_complete"] is True
    assert summary["grid_candidate_count"] == 12**2
    assert summary["evaluated_candidate_count"] == len(outcomes) == 12**2
    assert summary["optimization_started_count"] == 0
    assert summary["generation_failed_count"] == 0
    assert summary["screen_safe_count"] == len(manifest["records"]) > 0
    assert summary["z_rejected_count"] > 0
    assert [row["grid_index"] for row in outcomes] == list(range(1, 145))
    assert len({tuple(row["start_dihedrals_deg"]) for row in outcomes}) == 144
    assert all(len(row["start_dihedrals_deg"]) == 2 for row in outcomes)
    rejected = [row for row in outcomes if row["status"] == "rejected"]
    assert len(rejected) + summary["screen_safe_count"] == 144
    assert all("structure" not in row for row in rejected)
    # This body's P-C axis is normal to the O plane, so axial phase cannot
    # change any height. ASE independently reconstructs each rejected grid
    # member: its heavy body lies above the plane and only H crosses it.
    standardized = read(manifest_path.parent / "sam-h0.extxyz")
    # EXTXYZ writes eight decimal places, so successive coordinate writes
    # allow a 2e-8 A transverse residual for the originally vertical axis.
    assert np.linalg.norm(standardized.positions[4, :2] - standardized.positions[0, :2]) < 2e-8
    for row in rejected:
        assert row["reason"] == "height_no_feasible_pc_phase"
        candidate = standardized.copy()
        for torsion, target, moving in zip(
            [[0, 4, 5, 6], [4, 5, 6, 7]],
            row["start_dihedrals_deg"],
            [[5, 6, 7, 8], [6, 7, 8]],
        ):
            candidate.set_dihedral(*torsion, target, indices=moving)
        assert np.min(candidate.positions[4:8, 2]) > 0.0
        assert candidate.positions[8, 2] < 0.0

    # Expected bonds are independent of the scanner's distance-based topology.
    expected_bonds = {
        (0, 1), (0, 2), (0, 3), (0, 4), (4, 5), (5, 6), (6, 7), (7, 8)
    }
    original_distances = original.get_all_distances()
    symbols = original.get_chemical_symbols()
    expected_files = set()
    for rank, row in enumerate(manifest["records"], 1):
        assert row["status"] == "passed_skeleton_cpu_screen"
        assert row["safe_rank"] == rank
        path = manifest_path.parent / row["structure"]
        expected_files.add(path)
        atoms = read(path)
        assert atoms.get_chemical_symbols() == symbols
        assert atoms.positions[1:4, 2] == pytest.approx(np.zeros(3), abs=1e-8)
        assert atoms.positions[0, 2] > 0
        assert np.min(atoms.positions[4:, 2]) >= -1.1e-8
        assert atoms.positions[8, 2] >= -1.1e-8  # The H belongs to the height gate.
        assert row["minimum_organic_z_A"] == pytest.approx(
            np.min(atoms.positions[4:, 2]), abs=1e-8
        )
        distances = atoms.get_all_distances()
        for left, right in expected_bonds:
            assert distances[left, right] == pytest.approx(
                original_distances[left, right], abs=4e-8
            )
        for left in range(len(atoms)):
            for right in range(left + 1, len(atoms)):
                if (left, right) in expected_bonds:
                    continue
                threshold = 1.4 if "H" in (symbols[left], symbols[right]) else 1.8
                assert distances[left, right] >= threshold - 2e-8
        for torsion, target in zip([[0, 4, 5, 6], [4, 5, 6, 7]], row["start_dihedrals_deg"]):
            difference = (atoms.get_dihedral(*torsion) - target + 180.0) % 360.0 - 180.0
            assert difference == pytest.approx(0.0, abs=2e-6)
    assert set(manifest_path.parent.glob("synthetic_monomer_unoptimized_conformations/*.extxyz")) == expected_files


def test_phosphonate_skeleton_screen_discards_nonbonded_hydrogen_clashes_before_saving(
    tmp_path, monkeypatch
):
    manifest_path, manifest, outcomes, _ = _run_synthetic_skeleton_screen(
        tmp_path, monkeypatch, clashing_hydrogens=True
    )
    assert manifest["status"] == "screen_completed"
    assert len(outcomes) == manifest["summary"]["evaluated_candidate_count"] == 144
    assert manifest["summary"]["intramolecular_collision_rejected_count"] == 144
    assert manifest["summary"]["screen_safe_count"] == 0
    assert manifest["summary"]["optimization_started_count"] == 0
    assert manifest["records"] == []
    assert not list(manifest_path.parent.glob("synthetic_monomer_unoptimized_conformations/*.extxyz"))
    for row in outcomes:
        assert row["status"] == "rejected"
        assert row["reason"] == "intramolecular_collision"
        assert row["first_collision"]["atom_ids_1based"] == [9, 10]
        assert row["first_collision"]["distance_A"] < 1.2
        assert row["first_collision"]["threshold_A"] == 1.2
        assert "structure" not in row


def test_phosphonate_skeleton_screen_partial_limit_retains_every_evaluated_outcome(
    tmp_path, monkeypatch
):
    manifest_path, manifest, outcomes, _ = _run_synthetic_skeleton_screen(
        tmp_path, monkeypatch, skeleton_limit=5
    )
    assert manifest["status"] == "partial_screen_completed"
    assert manifest["grid_complete"] is False
    assert manifest["summary"]["grid_candidate_count"] == 144
    assert manifest["summary"]["evaluated_candidate_count"] == len(outcomes) == 5
    assert manifest["summary"]["optimization_started_count"] == 0
    assert [row["grid_index"] for row in outcomes] == [1, 2, 3, 4, 5]
    assert len(manifest["records"]) == manifest["summary"]["screen_safe_count"]
    progress = json.loads((manifest_path.parent / "progress.json").read_text())
    assert progress["status"] == "partial_screen_completed"
    assert progress["summary"] == manifest["summary"]


def test_phosphonate_skeleton_screen_rejects_nonperiodic_step_before_output(
    tmp_path, monkeypatch
):
    with pytest.raises(ValueError, match="divide 360 degrees exactly"):
        _run_synthetic_skeleton_screen(tmp_path, monkeypatch, scan_step_deg=31.0)
    assert not (tmp_path / "screen").exists()


def test_fixed_site_mace_plan_rejects_isolated_skeleton_screen(tmp_path, monkeypatch):
    from types import SimpleNamespace

    manifest_path, _, _, _ = _run_synthetic_skeleton_screen(
        tmp_path, monkeypatch, skeleton_limit=1
    )
    model = tmp_path / "placeholder-model"
    model.write_bytes(b"The rejection must precede loading a model.")
    output = tmp_path / "must-not-plan-mace"
    with pytest.raises(ValueError, match="an ITO-site audit before a MACE plan"):
        scan.run_fixed_site_optimization_plan(
            SimpleNamespace(
                cpu_manifest=manifest_path,
                model=model,
                formal_output_root=output,
                generalized=True,
            )
        )
    assert not output.exists()
