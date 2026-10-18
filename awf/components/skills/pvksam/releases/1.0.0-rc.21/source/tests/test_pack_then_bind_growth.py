"""Behavioral checks for selection before binding, using real geometry caches."""

import copy
import itertools
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import monolayer_sequential_growth as growth
from monolayer_headgroup_fit import rotate_phosphonate_about_pc, sample_phosphonate_pc_rotations


FRAME = {
    "u_cartesian_unit": [1.0, 0.0, 0.0],
    "v_cartesian_unit": [0.0, 1.0, 0.0],
    "outward_normal_cartesian_unit": [0.0, 0.0, 1.0],
}


def config(**changes):
    values = {
        "footprint_boundary_samples": 72,
        "projection_radius_scale": 1.0,
        "hh_min": 1.5, "h_heavy_min": 1.8, "heavy_heavy_min": 2.2,
        "cohesive_contact_vdw_radius_scale": 1.10,
        "cohesive_frontier_mode": "continuous-snap", "frontier_snap_radius_A": 4.0,
        "cohesive_energy_tolerance_eV": 0.03,
        "cohesive_perimeter_relative_tolerance": 0.10,
        "cohesive_contact_relative_tolerance": 0.20,
        "cohesive_area_relative_tolerance": 0.20,
        "headgroup_rmsd_max_A": 0.50,
        "mapped_bond_min_A": 1.7, "mapped_bond_max_A": 2.8,
        "substrate_vdw_radius_scale": 1.0,
        "write_every_step": False,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def conformer():
    # An exact tetrahedron plus an expanded organic footprint. This is a
    # synthetic geometry fixture, not a chemically optimized source molecule.
    oxygens = np.array([
        [np.sqrt(8 / 9) * np.cos(phi), np.sqrt(8 / 9) * np.sin(phi), -1 / 3]
        for phi in np.deg2rad([0.0, 120.0, 240.0])
    ]) * 1.54
    return {
        "coordinates": np.vstack(([0.0, 0.0, 0.0], oxygens, [0.0, 0.0, 1.84], [1.3, 0.0, 4.0], [-1.3, 0.0, 4.0])),
        "symbols": np.array(["P", "O", "O", "O", "C", "C", "C"]),
        "p_local": 0, "o_locals": [1, 2, 3], "source_rank": 1,
        "source": "synthetic-geometry-only.extxyz",
    }


def candidate(site, position, phi=0.0, energy=0.0):
    value = growth._pack_candidate(conformer(), position, site, "same-prototype", FRAME, phi, {"cluster_id": 1})
    value["target_donor_labels"] = ["O1", "O2", "O3"]
    value["conformer_energy_eV"] = energy
    return value


def substrate():
    return Atoms("In", positions=[[0.0, 0.0, -5.0]], cell=np.diag([25.0, 25.0, 30.0]), pbc=[True, True, False])


def projection_cache(domains, base, args):
    payload = {"site_instances": [
        {"site_instance_id": site, "ideal_site_location": {"cartesian_A": pool[0]["anchor_cartesian_A"]}}
        for site, pool in domains.items()
    ]}
    return growth._pack_projection_cache(domains, payload, base, (0, 1), FRAME, args)


def accepting_fit(chosen, placed):
    # The geometry core makes a real torsion; this callback deliberately omits
    # surface binding so growth-state behavior can be isolated from site input.
    coordinates = rotate_phosphonate_about_pc(chosen["coordinates"], 0, [1, 2, 3], 4, 37.5)
    trial = {
        **chosen, "coordinates": coordinates, "headgroup_pc_rotation_deg": 37.5,
        "registered_interface_bonds": [], "headgroup_rmsd_A": 0.0,
        "binding_fit_audit": {"organic_and_P_fixed": True},
    }
    return trial, {"passed": True}


def test_same_azimuth_shares_template_across_translations_and_different_azimuth_does_not():
    first = candidate("first", [1.25, 7.33, 2.71], phi=30.0)
    translated = candidate("translated", [193.125, -27.913, 2.71], phi=30.0)
    rotated = candidate("rotated", [1.25, 7.33, 2.71], phi=60.0)
    assert first["conformer_sha256"] == translated["conformer_sha256"]
    assert first["conformer_sha256"] != rotated["conformer_sha256"]
    assert first["headgroup_binding_pending"] is True
    expected = np.array(translated["anchor_cartesian_A"]) - first["anchor_cartesian_A"]
    np.testing.assert_allclose(translated["coordinates"] - first["coordinates"], np.broadcast_to(expected, (7, 3)), atol=3e-14)


def test_binding_failure_backtracks_without_consuming_site_metal_or_union(tmp_path, monkeypatch):
    args, base = config(), substrate()
    bad = candidate("site", [4.0, 4.0, 3.0], phi=0, energy=-1)
    good = candidate("site", [4.0, 4.0, 3.0], phi=180, energy=0)
    competitor = candidate("other-site-same-metal", [17.0, 17.0, 3.0], phi=0, energy=10)
    domains = {"site": [bad, good], "other-site-same-metal": [competitor]}
    cache = projection_cache(domains, base, args)
    events, union_at_selection = [], []
    real_selector = growth.choose_cohesive_frontier_candidate

    def observed_selector(**kwargs):
        events.append("select")
        union_at_selection.append(kwargs["accepted_union"])
        return real_selector(**kwargs)

    def fit(chosen, placed):
        events.append(("fit", chosen["candidate_id"], len(placed)))
        if chosen["candidate_id"] == bad["candidate_id"]:
            return None, {"passed": False, "reason": "deliberately_unreachable_headgroup"}
        return accepting_fit(chosen, placed)

    monkeypatch.setattr(growth, "choose_cohesive_frontier_candidate", observed_selector)
    result = growth.run_pack_then_bind_domains(
        domains=domains, projection_cache=cache, substrate=base, axes=(0, 1), args=args,
        seed=12, fit_candidate=fit, metal_constraints={"site": {1}, "other-site-same-metal": {1}},
        conflicts={}, output_dir=tmp_path,
    )
    assert events == ["select", ("fit", bad["candidate_id"], 0), "select", ("fit", good["candidate_id"], 0)]
    assert union_at_selection == [None, None]
    assert result["attempt_count"] == 2
    assert [c["candidate_id"] for c in result["placed"]] == [good["candidate_id"]]
    assert len(result["steps"]) == 1
    step = result["steps"][0]
    assert step["global_azimuth_deg"] == 180.0
    assert step["headgroup_pc_rotation_deg"] == 37.5
    assert step["policy_audit"]["nucleus_reason"] == "initial_molecule"
    np.testing.assert_array_equal(result["placed"][0]["coordinates"][[0, 4, 5, 6]], good["coordinates"][[0, 4, 5, 6]])
    attempts = [json.loads(line) for line in (tmp_path / "binding-attempts.jsonl").read_text().splitlines()]
    assert [record["before_step"] for record in attempts] == [1, 1]
    assert attempts[0]["binding"]["passed"] is False
    assert attempts[1]["binding"]["passed"] is True


def test_next_molecule_uses_real_cohesive_frontier_after_first_acceptance(tmp_path):
    args, base = config(), substrate()
    left = candidate("left", [4.0, 4.0, 3.0], energy=-1)
    right = candidate("right", [9.0, 4.0, 3.0], energy=0)
    domains = {"left": [left], "right": [right]}
    cache = projection_cache(domains, base, args)
    result = growth.run_pack_then_bind_domains(
        domains=domains, projection_cache=cache, substrate=base, axes=(0, 1), args=args,
        seed=12, fit_candidate=accepting_fit, metal_constraints={"left": {1}, "right": {2}},
        conflicts={}, output_dir=tmp_path,
    )
    assert len(result["placed"]) == 2
    assert result["steps"][0]["policy_audit"]["growth_mode"] == "new-nucleus"
    second = result["steps"][1]["policy_audit"]
    assert second["new_nucleus_event"] is False
    assert second["frontier_candidate_ids"] == [right["candidate_id"]]
    assert second["projection_prefilter"]["maximum_local_neighbor_count"] == 1


def test_single_phase_continuation_keeps_previous_cache_footprint_and_contact(tmp_path):
    args, base = config(), substrate()
    left = candidate("left", [4.0, 4.0, 3.0])
    initial_domains = {"left": [left]}
    initial_cache = projection_cache(initial_domains, base, args)
    first_dir = tmp_path / "first"
    first_dir.mkdir()
    first = growth.run_pack_then_bind_domains(
        domains=initial_domains, projection_cache=initial_cache, substrate=base, axes=(0, 1), args=args,
        seed=12, fit_candidate=accepting_fit, metal_constraints={"left": {1}},
        conflicts={}, output_dir=first_dir,
    )
    original_geometry = first["placed"][0]["packing_footprint_geometry"]
    right = candidate("single-right", [9.0, 4.0, 3.0])
    right.update({"phase_priority": 1, "target_donor_labels": ["O1"]})
    next_domains = {"single-right": [right]}
    next_cache = projection_cache(next_domains, base, args)
    # Both independent caches initially use index zero, but for different sites.
    assert first["placed"][0]["candidate_index"] == right["candidate_index"] == 0
    next_dir = tmp_path / "next"
    next_dir.mkdir()
    result = growth.run_pack_then_bind_domains(
        domains=next_domains, projection_cache=next_cache, substrate=base, axes=(0, 1), args=args,
        seed=13, fit_candidate=accepting_fit, metal_constraints={"single-right": {2}},
        conflicts={}, output_dir=next_dir, placed=first["placed"],
    )
    assert len(result["placed"]) == 2
    assert len(result["steps"]) == 1
    assert result["steps"][0]["denticity_phase"] == 1
    assert result["steps"][0]["policy_audit"]["new_nucleus_event"] is False
    old = result["placed"][0]
    assert old["candidate_index"] != right["candidate_index"]
    assert next_cache.candidate_geometries[old["candidate_index"]].equals_exact(original_geometry, 1e-14)


def test_full_collision_check_includes_rotatable_o_and_periodic_images():
    left = {
        "coordinates": np.array([[5, 5, 2], [0.2, 2, 2], [4, 4, 2], [4, 6, 2], [10, 10, 7]], dtype=float),
        "symbols": np.array(["P", "O", "O", "O", "C"]),
        "organic_local_indices_0based": [4],
    }
    right = {
        "coordinates": np.array([[15, 5, 2], [19.9, 2, 2], [14, 4, 2], [14, 6, 2], [15, 15, 7]], dtype=float),
        "symbols": np.array(["P", "O", "O", "O", "C"]),
        "organic_local_indices_0based": [4],
    }
    thresholds = {"H-H": 1.5, "H-heavy": 1.8, "heavy-heavy": 2.2}
    cell = np.diag([20.0, 20.0, 30.0])
    assert not growth._pack_collision(left, right, cell, (0, 1), thresholds, pending_left=True)
    assert growth._pack_collision(left, right, cell, (0, 1), thresholds, pending_left=False)


def test_actual_head_fit_keeps_body_and_checks_true_mapped_metal_distances():
    args = config()
    packed = candidate("registered", [5.0, 5.0, 3.0], phi=30.0)
    target_coordinates = rotate_phosphonate_about_pc(packed["coordinates"], 0, [1, 2, 3], 4, 23.0)
    metal_positions = target_coordinates[1:4] - [0.0, 0.0, 2.2]
    base = Atoms("In3", positions=metal_positions, cell=np.diag([25.0, 25.0, 30.0]), pbc=[True, True, False])
    instance = {
        "site_instance_id": "registered", "actual_metal_atom_ids_1based": [1, 2, 3],
        "actual_metals": [{"atom_id_1based": i, "actual_element_after_doping": "In"} for i in [1, 2, 3]],
        "donor_metal_mapping": [{"probe_donor_id": f"O{i}", "metal_atom_id_1based": i} for i in [1, 2, 3]],
        "final_denticity": 3,
    }
    packed.update({"headgroup_target_A": target_coordinates[:4].tolist(), "site_instance_record": instance})
    full_map = growth.build_full_to_working_atom_map(working_layer_atom_ids_1based=[1, 2, 3], working_symbols=["In"] * 3)
    before = packed["coordinates"].copy()
    trial, audit = growth._pack_fit_candidate(packed, [], substrate=base, axes=(0, 1), full_map=full_map, args=args)
    assert trial is not None and audit["passed"]
    assert trial["headgroup_binding_pending"] is False
    assert trial["headgroup_pc_rotation_deg"] == pytest.approx(23.0, abs=1e-12)
    np.testing.assert_array_equal(trial["coordinates"][[0, 4, 5, 6]], before[[0, 4, 5, 6]])
    np.testing.assert_array_equal(packed["coordinates"], before)
    assert trial["substrate_collision_audit"]["passed"] is True
    assert len(trial["registered_interface_bonds"]) == 3
    for bond in trial["registered_interface_bonds"]:
        donor = bond["molecule_atom_index_0based"]
        metal = bond["working_metal_atom_index_0based"]
        assert np.linalg.norm(trial["coordinates"][donor] - base.positions[metal]) == pytest.approx(2.2, abs=1e-12)
        assert bond["full_metal_atom_id_1based"] == metal + 1


def test_single_metal_binding_records_actual_donor_and_never_moves_body():
    args = config()
    packed = candidate("single", [5.0, 5.0, 3.0], phi=0.0)
    metal_position = packed["coordinates"][1] - [0.0, 0.0, 2.2]
    base = Atoms("Sn", positions=[metal_position], cell=np.diag([25.0, 25.0, 30.0]), pbc=[True, True, False])
    packed.update({
        "phase_priority": 1, "target_donor_labels": ["O1"],
        "metal_position_A": metal_position.tolist(), "anchor_metal_atom_id_1based": 7,
        "anchor_metal_working_index_0based": 0, "anchor_metal_element": "Sn",
    })
    full_map = growth.build_full_to_working_atom_map(working_layer_atom_ids_1based=[7], working_symbols=["Sn"])
    before = copy.deepcopy(packed)
    trial, audit = growth._pack_fit_candidate(packed, [], substrate=base, axes=(0, 1), full_map=full_map, args=args)
    assert trial is not None and audit["passed"]
    assert trial["headgroup_binding_pending"] is False
    np.testing.assert_array_equal(trial["coordinates"][[0, 4, 5, 6]], before["coordinates"][[0, 4, 5, 6]])
    assert len(trial["registered_interface_bonds"]) == 1
    bond = trial["registered_interface_bonds"][0]
    assert bond["full_metal_atom_id_1based"] == 7
    assert bond["working_metal_atom_index_0based"] == 0
    assert bond["molecule_atom_index_0based"] in [1, 2, 3]
    assert bond["donor_label"] == "O1"
    assert np.linalg.norm(trial["coordinates"][bond["molecule_atom_index_0based"]] - metal_position) == pytest.approx(2.2, abs=1e-12)


@pytest.mark.parametrize("blocking_atom", [None, "fixed-body", "rotating-head"])
def test_split_body_head_binding_matches_full_pair_reference(blocking_atom):
    """Compare actual accepted/rejected fits against unsplit all-atom audits."""
    args = config()
    packed = candidate("equivalence", [5.0, 5.0, 3.0], phi=30.0)
    wanted = rotate_phosphonate_about_pc(packed["coordinates"], 0, [1, 2, 3], 4, 23.0)
    metal_positions = wanted[1:4] - [0.0, 0.0, 2.2]
    if blocking_atom == "fixed-body":
        metal_positions = np.vstack((metal_positions, packed["coordinates"][4] + [0.1, 0.0, 0.0]))
    elif blocking_atom == "rotating-head":
        metal_positions = np.vstack((metal_positions, wanted[1] - [0.0, 0.0, 1.5]))
    base = Atoms("In" * len(metal_positions), positions=metal_positions, cell=np.diag([25.0, 25.0, 30.0]), pbc=[True, True, False])
    packed.update({
        "headgroup_target_A": wanted[:4].tolist(),
        "site_instance_record": {
            "site_instance_id": "equivalence", "actual_metal_atom_ids_1based": [1, 2, 3],
            "actual_metals": [{"atom_id_1based": i, "actual_element_after_doping": "In"} for i in [1, 2, 3]],
            "donor_metal_mapping": [{"probe_donor_id": f"O{i}", "metal_atom_id_1based": i} for i in [1, 2, 3]],
            "final_denticity": 3,
        },
    })
    full_map = growth.build_full_to_working_atom_map(
        working_layer_atom_ids_1based=list(range(1, len(base) + 1)), working_symbols=base.get_chemical_symbols(),
    )
    proposals = []
    for order in itertools.permutations([1, 2, 3]):
        for fit in sample_phosphonate_pc_rotations(
            packed["coordinates"], 0, [1, 2, 3], 4, wanted[1:4], donor_order=order,
            target_p_coordinate=wanted[0], max_headgroup_rmsd_A=args.headgroup_rmsd_max_A, angle_step_deg=5.0,
        ):
            proposals.append((fit["headgroup_rmsd_A"], order, fit["pc_rotation_deg"], fit))
    proposals.sort(key=lambda row: (row[0], row[1], row[2]))
    reference = None
    for _, order, _, fit in proposals:
        bonds = growth.resolve_candidate_registered_bonds(
            instance=packed["site_instance_record"], target_donor_labels=packed["target_donor_labels"],
            oxygen_permutation=order, molecule_symbols=packed["symbols"], full_to_working_atom_map=full_map,
            distance_window_A=(args.mapped_bond_min_A, args.mapped_bond_max_A),
        )
        # Verify this fixture has no independent O/body rejection, then apply
        # the old complete pair matrix with original local atom indices.
        distances = np.linalg.norm(fit["coordinates"][[1, 2, 3], None] - fit["coordinates"][[4, 5, 6]][None], axis=2)
        assert np.min(distances) > 1.8
        audit = growth.audit_candidate_substrate_contacts(
            mode="check", molecule_positions=fit["coordinates"], molecule_symbols=packed["symbols"],
            substrate_positions=base.positions, substrate_symbols=base.get_chemical_symbols(),
            cell=base.cell, periodic_axes=(0, 1), surface_frame=FRAME,
            registered_interface_bonds=bonds,
            hard_thresholds_A={"H-H": 1.5, "H-heavy": 1.8, "heavy-heavy": 2.2}, vdw_radius_scale=1.0,
        )
        if audit["passed"]:
            reference = fit
            break
    trial, split_audit = growth._pack_fit_candidate(packed, [], substrate=base, axes=(0, 1), full_map=full_map, args=args)
    assert (trial is None) == (reference is None)
    assert (trial is not None) == (blocking_atom is None)
    if reference is not None:
        np.testing.assert_allclose(trial["coordinates"], reference["coordinates"], atol=1e-14)
        assert trial["headgroup_pc_rotation_deg"] == pytest.approx(reference["pc_rotation_deg"], abs=1e-12)
        assert sorted(row["molecule_atom_id"] for row in trial["substrate_collision_audit"]["mapped_bonds"]) == [1, 2, 3]
    elif blocking_atom == "fixed-body":
        assert split_audit["reason"] == "fixed_body_substrate_collision"
    else:
        assert split_audit["reason"] == "no_feasible_headgroup_rotation"


def pending_head_overlap_pair():
    """Synthetic head overlap across the x seam, with valid body contact."""
    pending = {
        "candidate_id": "pending", "headgroup_binding_pending": True,
        "coordinates": np.array([[1, 1, 3], [0.2, 2, 2], [2, 1, 2], [1, 2, 2], [5, 5, 6]], dtype=float),
        "symbols": np.array(["P", "O", "O", "O", "C"]),
        "headgroup_local_indices_0based": [0, 1, 2, 3],
    }
    accepted = {
        "candidate_id": "accepted", "headgroup_binding_pending": False,
        "coordinates": np.array([[15, 5, 2], [19.9, 2, 2], [15, 6, 2], [15, 4, 2], [8, 5, 6]], dtype=float),
        "symbols": np.array(["P", "O", "O", "O", "C"]),
        "headgroup_local_indices_0based": [0, 1, 2, 3],
    }
    return pending, accepted


def contact_report(pending, accepted):
    return growth.cohesive_side_contact_report(
        pending, [accepted], cell=np.diag([20.0, 20.0, 30.0]), periodic_axes=(0, 1),
        collision_thresholds_A={"H-H": 1.5, "H-heavy": 1.8, "heavy-heavy": 2.2},
    )


def test_pending_head_overlap_does_not_abort_organic_packing_contact():
    pending, accepted = pending_head_overlap_pair()
    report = contact_report(pending, accepted)
    assert report["has_contact"] is True
    assert report["contact_count"] == 1
    assert report["contacted_neighbor_ids"] == ["accepted"]


@pytest.mark.parametrize("accepted_index", [1, 4])
def test_pending_head_does_not_exempt_organic_overlap_with_any_accepted_atom(accepted_index):
    pending, accepted = pending_head_overlap_pair()
    pending["coordinates"][4] = accepted["coordinates"][accepted_index] + [1.0, 0.0, 0.0]
    with pytest.raises(RuntimeError, match="prior SAM-SAM collision gate"):
        contact_report(pending, accepted)


def test_fitted_head_overlap_still_aborts_contact_scoring():
    pending, accepted = pending_head_overlap_pair()
    pending["headgroup_binding_pending"] = False
    with pytest.raises(RuntimeError, match="prior SAM-SAM collision gate"):
        contact_report(pending, accepted)


def replay_fixture(tmp_path):
    """A sealed failed parent with one rejected pose then one accepted pose."""
    args = config(write_every_step=True, branch_rank=1, seed=12,
                  pack_replay_prefix=None, pack_replay_verify_only=False)
    base = substrate()
    bad = candidate("site", [4., 4., 3.], phi=0, energy=-1)
    good = candidate("site", [4., 4., 3.], phi=180, energy=0)
    rival = candidate("rival", [17., 17., 3.], energy=10)
    domains = {"site": [bad, good], "rival": [rival]}
    constraints = {"site": {1}, "rival": {1}}
    parent = tmp_path / "parent"; parent.mkdir()
    trajectory = parent / "trajectory-0001"; trajectory.mkdir()
    source = tmp_path / "source.xyz"; source.write_text("synthetic identity fixture\n")
    inputs = {"source": source}
    selected = [{"source": str(source)}]

    def fit(chosen, placed):
        if chosen["candidate_id"] == bad["candidate_id"]:
            return None, {"passed": False, "reason": "deliberately_unreachable"}
        trial, binding = accepting_fit(chosen, placed)
        return trial, {**binding, "headgroup_pc_rotation_deg": 37.5}

    original = growth.run_pack_then_bind_domains(
        domains=domains, projection_cache=projection_cache(domains, base, args), substrate=base,
        axes=(0, 1), args=args, seed=12, fit_candidate=fit, metal_constraints=constraints,
        conflicts={}, output_dir=trajectory,
    )
    write(parent / "bare-substrate.extxyz", base)
    request = {"schema": "pvksam-pack-then-bind-request-v1", "parameters": vars(args),
               "inputs": {"source": growth._file_identity(source)},
               "conformers": [growth._file_identity(source)]}
    (parent / "run-request.json").write_text(json.dumps(request))
    ledger = [{**growth._file_identity(p), "path": p.relative_to(parent).as_posix()}
              for p in parent.rglob("*") if p.is_file()]
    (parent / "failure.json").write_text(json.dumps({
        "schema": "sam-sequential-growth-failure-v1", "schema_version": 1, "sealed": True,
        "artifact_inventory_capture": {"status": "captured"}, "artifacts": ledger}))
    child = tmp_path / "child"; child.mkdir()
    return SimpleNamespace(args=args, base=base, domains=domains, constraints=constraints,
                           parent=parent, child=child, inputs=inputs, selected=selected,
                           fit=fit, original=original)


def freeze_fixture(f):
    return growth._pack_freeze_replay_prefix(f.parent, output_dir=f.child, args=f.args,
                                           resolved_paths=f.inputs, conformers=f.selected)


def replay_fixture_run(f, replay, domains=None, fit=None):
    domains = copy.deepcopy(f.domains if domains is None else domains)
    return growth.run_pack_then_bind_domains(
        domains=domains, projection_cache=projection_cache(domains, f.base, f.args),
        substrate=f.base, axes=(0, 1), args=f.args, seed=12, fit_candidate=fit or f.fit,
        metal_constraints={**f.constraints, "extra": {2}}, conflicts={}, output_dir=f.child,
        replay_prefix=replay,
    )


def test_prefix_replay_reproduces_bytes_keeps_float64_and_never_scores(tmp_path, monkeypatch):
    f = replay_fixture(tmp_path)
    replay = freeze_fixture(f)
    monkeypatch.setattr(growth, "choose_cohesive_frontier_candidate",
                        lambda **kwargs: pytest.fail("Replay must not recalculate selection scores"))
    result = replay_fixture_run(f, replay)
    np.testing.assert_array_equal(result["placed"][0]["coordinates"], f.original["placed"][0]["coordinates"])
    rounded = read(replay["steps"][1]).positions[-7:]
    assert not np.array_equal(result["placed"][0]["coordinates"], rounded)
    assert (f.child / "steps/step-0001.extxyz").read_bytes() == replay["steps"][1].read_bytes()
    assert result["steps"][0]["policy_audit"]["selection_recomputed"] is False
    assert result["steps"][0]["policy_audit"]["parent_policy_audit"] is None
    assert result["attempt_count"] == 2


@pytest.mark.parametrize("field,value", [("seed", 99), ("headgroup_rmsd_max_A", .25), ("hh_min", 1.4)])
def test_replay_rejects_changed_scientific_or_selection_parameters(tmp_path, field, value):
    f = replay_fixture(tmp_path)
    setattr(f.args, field, value)
    with pytest.raises(ValueError, match="parameters differ"):
        freeze_fixture(f)


def test_replay_rejects_changed_source_identity(tmp_path):
    f = replay_fixture(tmp_path)
    f.inputs["source"].write_text("tampered source\n")
    with pytest.raises(ValueError, match="input identity mismatch"):
        freeze_fixture(f)


def test_replay_rejects_sealed_step_tampering(tmp_path):
    f = replay_fixture(tmp_path)
    step = f.parent / "trajectory-0001/steps/step-0001.extxyz"
    atoms = read(step); atoms.positions[-1, 0] += .01; write(step, atoms)
    with pytest.raises(ValueError, match="Sealed parent artifact changed"):
        freeze_fixture(f)


def test_replay_rejects_missing_step(tmp_path):
    f = replay_fixture(tmp_path)
    (f.parent / "trajectory-0001/steps/step-0001.extxyz").unlink()
    with pytest.raises(FileNotFoundError):
        freeze_fixture(f)


def test_replay_rejects_unexhausted_prefix(tmp_path):
    f = replay_fixture(tmp_path); replay = freeze_fixture(f)
    domains = {**f.domains, "extra": [candidate("extra", [15., 4., 3.], energy=100)]}
    with pytest.raises(ValueError, match="not exhausted"):
        replay_fixture_run(f, replay, domains=domains)


def test_replay_rejects_changed_failure_result(tmp_path):
    f = replay_fixture(tmp_path); replay = freeze_fixture(f)
    with pytest.raises(ValueError, match="headgroup fit differs"):
        replay_fixture_run(f, replay, fit=lambda chosen, placed: accepting_fit(chosen, placed))


def test_replay_rejects_changed_coordinates_even_if_binding_receipt_matches(tmp_path):
    f = replay_fixture(tmp_path); replay = freeze_fixture(f)
    def moved_fit(chosen, placed):
        trial, binding = f.fit(chosen, placed)
        if trial is not None:
            trial["coordinates"] = trial["coordinates"].copy()
            trial["coordinates"][-1, 0] += .01
        return trial, binding
    with pytest.raises(ValueError, match="structure bytes"):
        replay_fixture_run(f, replay, fit=moved_fit)


def test_replay_rejects_changed_psi_receipt(tmp_path):
    f = replay_fixture(tmp_path); replay = freeze_fixture(f)
    replay["attempts"][-1]["binding"]["headgroup_pc_rotation_deg"] += 30
    with pytest.raises(ValueError, match="headgroup fit differs"):
        replay_fixture_run(f, replay)


def test_active_parent_snapshot_ignores_later_CN5_append(tmp_path):
    f = replay_fixture(tmp_path)
    (f.parent / "failure.json").unlink()
    log = f.parent / "trajectory-0001/binding-attempts.jsonl"
    tail = {"phase": 1, "before_step": 2, "candidate_id": "single-tail", "binding": {"passed": False}}
    with log.open("a") as stream:
        stream.write(json.dumps(tail)+"\n")
    replay = freeze_fixture(f)
    assert replay["evidence"]["parent_state_at_snapshot"] == "active_stage_snapshot"
    assert replay["evidence"]["parent_run_completed_claim"] is False
    with log.open("a") as stream:
        stream.write(json.dumps({**tail, "candidate_id": "later-single-tail"})+"\n")
    result = replay_fixture_run(f, replay)
    assert result["attempt_count"] == 2
    assert len(result["placed"]) == 1


def test_active_parent_without_completed_multidentate_stage_is_rejected(tmp_path):
    f = replay_fixture(tmp_path); (f.parent / "failure.json").unlink()
    with pytest.raises(ValueError, match="entry into CN5"):
        freeze_fixture(f)


@pytest.mark.parametrize("status,expected_completed,expected_state", [
    ("completed", True, "completed"),
    ("validated_multidentate_prefix_only", False, "validated_prefix_only"),
])
def test_replay_parent_manifest_status_is_explicit(tmp_path, status, expected_completed, expected_state):
    f = replay_fixture(tmp_path); (f.parent / "failure.json").unlink()
    (f.parent / "manifest.json").write_text(json.dumps({"schema": "pvksam-pack-then-bind-run-v1", "status": status}))
    replay = freeze_fixture(f)
    assert replay["evidence"]["parent_run_completed_claim"] is expected_completed
    assert replay["evidence"]["parent_state_at_snapshot"] == expected_state
    assert (f.child / "parent-evidence/manifest.json").read_bytes() == (f.parent / "manifest.json").read_bytes()
    assert len(replay_fixture_run(f, replay)["placed"]) == 1


def test_arbitrary_root_manifest_is_not_completed_parent(tmp_path):
    f = replay_fixture(tmp_path); (f.parent / "failure.json").unlink()
    (f.parent / "manifest.json").write_text(json.dumps({"schema": "pvksam-pack-then-bind-run-v1", "status": "still_running"}))
    with pytest.raises(ValueError, match="entry into CN5"):
        freeze_fixture(f)


def test_unknown_root_manifest_schema_is_rejected(tmp_path):
    f = replay_fixture(tmp_path); (f.parent / "failure.json").unlink()
    (f.parent / "manifest.json").write_text(json.dumps({"schema": "unknown", "status": "completed"}))
    with pytest.raises(ValueError, match="manifest schema"):
        freeze_fixture(f)


def test_replay_output_isolation_happens_before_reservation(tmp_path):
    f = replay_fixture(tmp_path)
    output = f.parent / "must-not-be-created"
    argv = ["--root", str(tmp_path), "--substrate", str(f.inputs["source"]),
            "--layer-groups", str(f.inputs["source"]), "--site-instances", str(f.inputs["source"]),
            "--site-prototypes", str(f.inputs["source"]), "--conformer-analysis", str(f.inputs["source"]),
            "--conformer-glob", "source.xyz", "--molecule-formula-json", '{"C":1}',
            "--pose-mode", "pack-then-bind", "--branch-rank", "1", "--output-dir", str(output),
            "--pack-replay-prefix", str(f.parent)]
    with pytest.raises(ValueError, match="separate directory trees"):
        growth.sequential_growth_main(argv)
    assert not output.exists()


def test_replayed_full_precision_state_gives_same_CN5_continuation(tmp_path):
    f = replay_fixture(tmp_path); replay = freeze_fixture(f)
    rebuilt = replay_fixture_run(f, replay)
    results = []
    for name, initial in [("old", f.original["placed"]), ("replayed", rebuilt["placed"])]:
        out = tmp_path / name; out.mkdir()
        one = candidate("single-a", [9., 4., 3.])
        two = candidate("single-b", [4., 9., 3.])
        for value in [one, two]:
            value.update({"target_donor_labels": ["O1"], "phase_priority": 1})
        domains = {"single-a": [one], "single-b": [two]}
        results.append(growth.run_pack_then_bind_domains(
            domains=domains, projection_cache=projection_cache(domains, f.base, f.args),
            substrate=f.base, axes=(0,1), args=f.args, seed=13, fit_candidate=f.fit,
            metal_constraints={"single-a": {2}, "single-b": {3}}, conflicts={}, output_dir=out,
            placed=copy.deepcopy(initial)))
    assert [s["candidate_id"] for s in results[0]["steps"]] == [s["candidate_id"] for s in results[1]["steps"]]
    for old, new in zip(results[0]["placed"], results[1]["placed"]):
        np.testing.assert_array_equal(old["coordinates"], new["coordinates"])


@pytest.mark.parametrize("field,value", [("sealed", False), ("schema", "other"), ("schema_version", 2)])
def test_replay_rejects_unsealed_or_unknown_failure_receipt(tmp_path, field, value):
    f = replay_fixture(tmp_path); path = f.parent / "failure.json"
    value_dict = json.loads(path.read_text()); value_dict[field] = value; path.write_text(json.dumps(value_dict))
    with pytest.raises(ValueError, match="recognized sealed failure"):
        freeze_fixture(f)


def test_multi_dentate_attempt_after_CN5_is_rejected():
    data = b'{"phase":3}\n{"phase":1}\n{"phase":2}\n'
    with pytest.raises(ValueError, match="stage order"):
        growth._pack_prefix_attempt_bytes(data)


def test_joined_accepted_reference_matches_sequential_OR_at_periodic_boundary():
    cell = np.diag([25., 25., 30.])
    positions = [[.2, 3., 4.], [12., 12., 4.], [24.8, 12., 4.], [6., 6., 4.]]
    domains = {"site": [{"candidate_id": str(i), "coordinates": np.array([p]),
                          "symbols": np.array(["H"]), "organic_local_indices_0based": [0]}
                         for i,p in enumerate(positions)]}
    accepted = [{"coordinates": np.array([[24.9, 3., 4.]]), "symbols": np.array(["H"])},
                {"coordinates": np.array([[.2, 12., 4.]]), "symbols": np.array(["O"])}]
    thresholds = {"H-H": 1.5, "H-heavy": 1.8, "heavy-heavy": 2.2}
    sequential, joined = copy.deepcopy(domains), copy.deepcopy(domains)
    old_count = sum(growth._pack_prune_collisions(sequential, old, cell, (0,1), thresholds) for old in accepted)
    new_count = growth._pack_prune_against_placed(joined, accepted, cell, (0,1), thresholds)
    assert old_count == new_count == 2
    assert [c["candidate_id"] for c in sequential["site"]] == [c["candidate_id"] for c in joined["site"]] == ["1", "3"]


def mixed_source(rank, *, spread=1.3, head_scale=1.0):
    source = conformer()
    source["source_rank"] = rank
    source["source"] = f"synthetic-source-{rank}.extxyz"
    source["coordinates"][1:4] *= head_scale
    source["coordinates"][5:, 0] = [spread, -spread]
    return source


def mixed_pose(source, site, position, *, energy=0.0, phi=0.0):
    pose = growth._pack_candidate(source, position, site, "mixed-prototype", FRAME,
                                  phi, {"cluster_id": source["source_rank"],
                                        "relative_total_energy_eV": energy})
    pose["target_donor_labels"] = ["O1", "O2", "O3"]
    return pose


def bound_mixed_fit(chosen, placed):
    trial, receipt = accepting_fit(chosen, placed)
    trial["headgroup_binding_pending"] = False
    return trial, receipt


@pytest.mark.parametrize("wide_rank,compact_rank", [(1, 2), (2, 1)])
def test_mixed_conformer_selector_uses_geometry_and_contacts_not_rank_order(tmp_path, wide_rank, compact_rank):
    """Small footprint wins nucleation despite energy; wider body then reaches frontier."""
    args, base = config(), substrate()
    wide = mixed_source(wide_rank, spread=2.0)
    compact = mixed_source(compact_rank, spread=0.3)
    domains = {
        site: [mixed_pose(wide, site, position, energy=-1.0),
               mixed_pose(compact, site, position, energy=1.0)]
        for site, position in [("left", [4., 4., 3.]), ("right", [9.5, 4., 3.])]
    }
    cache = projection_cache(domains, base, args)
    result = growth.run_pack_then_bind_domains(
        domains=domains, projection_cache=cache, substrate=base, axes=(0, 1), args=args,
        seed=12, fit_candidate=bound_mixed_fit, metal_constraints={"left": {1}, "right": {2}},
        conflicts={}, output_dir=tmp_path)
    assert [p["selection_rank"] for p in result["placed"]] == [compact_rank, wide_rank]
    assert len({p["source"] for p in result["placed"]}) == 2
    first, second = result["steps"]
    assert first["policy_audit"]["growth_mode"] == "new-nucleus"
    assert second["policy_audit"]["growth_mode"] == "frontier"
    assert second["policy_audit"]["frontier_candidate_ids"] == [second["candidate_id"]]
    # Same site/phi, different source geometries must not alias projection templates.
    left = domains["left"]
    assert cache.template_index[left[0]["candidate_index"]] != cache.template_index[left[1]["candidate_index"]]


def test_mixed_conformer_failed_binding_preserves_other_source_on_same_site(tmp_path, monkeypatch):
    args, base = config(), substrate()
    bad = mixed_pose(mixed_source(1), "site", [4., 4., 3.], energy=-1.)
    good = mixed_pose(mixed_source(2), "site", [4., 4., 3.], energy=0.)
    shared_metal = mixed_pose(mixed_source(3), "other-site", [17., 17., 3.], energy=10.)
    domains = {"site": [bad, good], "other-site": [shared_metal]}
    unions, fitted = [], []
    original = growth.choose_cohesive_frontier_candidate
    def observe(**kwargs):
        unions.append(kwargs["accepted_union"])
        return original(**kwargs)
    def fit(chosen, placed):
        fitted.append((chosen["selection_rank"], len(placed)))
        if chosen["selection_rank"] == 1:
            return None, {"passed": False, "reason": "synthetic_unreachable_head"}
        return bound_mixed_fit(chosen, placed)
    monkeypatch.setattr(growth, "choose_cohesive_frontier_candidate", observe)
    result = growth.run_pack_then_bind_domains(
        domains=domains, projection_cache=projection_cache(domains, base, args),
        substrate=base, axes=(0, 1), args=args, seed=12, fit_candidate=fit,
        metal_constraints={"site": {1}, "other-site": {1}}, conflicts={}, output_dir=tmp_path)
    assert fitted == [(1, 0), (2, 0)]
    assert unions == [None, None]
    assert [c["candidate_id"] for c in result["placed"]] == [good["candidate_id"]]
    assert result["attempt_count"] == 2


def test_mixed_conformer_acceptance_consumes_shared_site_metals_and_collisions(tmp_path):
    args, base = config(), substrate()
    first = mixed_pose(mixed_source(1), "selected", [4., 4., 3.], energy=-1.)
    same_site = mixed_pose(mixed_source(2), "selected", [4., 4., 3.], energy=0.)
    same_metal = mixed_pose(mixed_source(3), "shared-metal", [17., 17., 3.], energy=1.)
    colliding = mixed_pose(mixed_source(2), "other-metal-overlap", [4.5, 4., 3.], energy=1.)
    free = mixed_pose(mixed_source(2), "free", [12., 12., 3.], energy=1.)
    domains = {"selected": [first, same_site], "shared-metal": [same_metal],
               "other-metal-overlap": [colliding], "free": [free]}
    result = growth.run_pack_then_bind_domains(
        domains=domains, projection_cache=projection_cache(domains, base, args),
        substrate=base, axes=(0, 1), args=args, seed=12, fit_candidate=bound_mixed_fit,
        metal_constraints={"selected": {1, 2, 3}, "shared-metal": {3, 4, 5},
                           "other-metal-overlap": {6, 7, 8}, "free": {9, 10, 11}},
        conflicts={}, output_dir=tmp_path)
    assert [c["candidate_id"] for c in result["placed"]] == [first["candidate_id"], free["candidate_id"]]
    assert result["attempt_count"] == 2
    assert result["steps"][0]["collision_pruned"] == 1


def mixed_registered_fixture(tmp_path):
    """Two source heads with different P-O lengths, one real donor/metal mapping."""
    sources = [mixed_source(1), mixed_source(2, head_scale=1.03)]
    for i, source in enumerate(sources):
        source["coordinates"] += [0., 0., 0.4*i]
        path = tmp_path / f"source-{i+1}.extxyz"
        write(path, Atoms(["In", *source["symbols"]],
                         positions=np.vstack(([0., 0., -5.], source["coordinates"]))))
        source["source"] = str(path)
        source["source_sha256"] = growth._sha256_of_file(path)
    anchor = np.array([5., 5., 3.])
    target = sources[0]["coordinates"][:4] + anchor
    base = Atoms("In3", positions=target[1:]-[0., 0., 2.2],
                 cell=np.diag([25., 25., 30.]), pbc=[True, True, False])
    prototype = {"site_prototype_id": "prototype", "final_denticity": 3,
                 "ideal_site_location": {"cartesian_A": [5., 5., 0.]},
                 "surface_frame": FRAME,
                 "relaxed_adsorption_location": {
                     "anchor_cartesian_A": anchor.tolist(),
                     "binding_oxygen_positions": [
                         {"probe_donor_id": f"O{i}", "cartesian_A": target[i].tolist()}
                         for i in [1, 2, 3]]}}
    instance = {"site_instance_id": "site", "parent_site_prototype_id": "prototype",
                "ideal_site_location": {"cartesian_A": [5., 5., 0.]},
                "actual_metal_atom_ids_1based": [1, 2, 3],
                "actual_metals": [{"atom_id_1based": i, "actual_element_after_doping": "In"} for i in [1, 2, 3]],
                "donor_metal_mapping": [{"probe_donor_id": f"O{i}", "metal_atom_id_1based": i} for i in [1, 2, 3]],
                "placement_template": {"surface_frame": FRAME}}
    coordination = {"metal_records_by_full_id": {
        i: {"coordination_number": 5, "element": "In", "working_atom_index_0based": i-1}
        for i in [1, 2, 3]}}
    metrics = {i: {"cluster_id": i, "relative_total_energy_eV": float(i-1)} for i in [1, 2]}
    return sources, base, {"site_instances": [instance]}, {"prototype": prototype}, coordination, metrics


def test_mixed_registered_site_pool_contains_all_sources_and_azimuths(tmp_path):
    sources, base, payload, prototypes, coordination, metrics = mixed_registered_fixture(tmp_path)
    args = config(azimuth_step_deg=180., substrate_height_clearance_A=1.)
    domains, effective, metals, audits, frame = growth._pack_registered_domains(
        sources, metrics, payload, prototypes, base, coordination, args)
    assert set(domains) == {"site"}
    assert {(c["selection_rank"], c["global_azimuth_deg"]) for c in domains["site"]} == {
        (1, 0.), (1, 180.), (2, 0.), (2, 180.)}
    assert len({c["candidate_id"] for c in domains["site"]}) == 4
    assert metals == {"site": {1, 2, 3}}
    assert len(effective["site_instances"]) == len(audits) == 1
    for c in domains["site"]:
        assert c["source_sha256"] == sources[c["selection_rank"]-1]["source_sha256"]
        np.testing.assert_array_equal(c["coordinates"][0], [5., 5., 3.])


def test_mixed_CN5_pool_preserves_each_source_height_and_shared_metal_identity(tmp_path):
    sources, base, _, _, _, metrics = mixed_registered_fixture(tmp_path)
    args = config(azimuth_step_deg=180., single_metal_translation_offsets_A=[0.],
                  substrate_height_clearance_A=1.)
    metal = {"full_metal_atom_id_1based": 1, "working_atom_index_0based": 0,
             "element": "In", "coordination_number": 5}
    domains, payload, metals = growth._pack_single_metal_pool(
        sources, metrics, [metal], base, FRAME, args)
    pool = [c for group in domains.values() for c in group]
    assert len(domains) == len(payload["site_instances"]) == 1
    assert len(pool) == 4
    assert {(c["selection_rank"], c["global_azimuth_deg"]) for c in pool} == {
        (1, 0.), (1, 180.), (2, 0.), (2, 180.)}
    assert list(metals.values()) == [{1}]
    for c in pool:
        assert c["phase_priority"] == 1 and c["target_donor_labels"] == ["O1"]
        assert c["anchor_metal_atom_id_1based"] == 1
        expected_height = 5.0 if c["selection_rank"] == 1 else 5.4
        assert c["coordinates"][0, 2]-base.positions[0, 2] == pytest.approx(expected_height)


def test_mixed_CN5_pool_prunes_every_source_against_the_same_accepted_layer(tmp_path):
    sources, base, _, _, _, metrics = mixed_registered_fixture(tmp_path)
    args = config(azimuth_step_deg=180., single_metal_translation_offsets_A=[0.],
                  substrate_height_clearance_A=1.)
    metals = [{"full_metal_atom_id_1based": 1, "working_atom_index_0based": 0,
               "element": "In", "coordination_number": 5}]
    reference_domains, _, _ = growth._pack_single_metal_pool(sources, metrics, metals, base, FRAME, args)
    candidate = next(iter(reference_domains.values()))[0]
    accepted = [{"coordinates": candidate["coordinates"][[4]].copy(), "symbols": np.array(["C"])}]
    pooled, _, _ = growth._pack_single_metal_pool(sources, metrics, metals, base, FRAME, args, placed=accepted)
    # Both sources' P-bound C collide; checking just one source would leave two poses.
    assert sum(map(len, pooled.values())) == 0


def test_mixed_source_shape_audit_uses_the_accepted_source_not_last_pool_source(tmp_path):
    sources, _, _, _, _, _ = mixed_registered_fixture(tmp_path)
    for source in sources:
        packed = mixed_pose(source, "site", [5., 5., 3.], phi=60.)
        trial, _ = bound_mixed_fit(packed, [])
        assert growth._pack_source_shape_audit(trial, source)["passed"]
        wrong = sources[1] if source["source_rank"] == 1 else sources[0]
        with pytest.raises(ValueError, match="source"):
            growth._pack_source_shape_audit(trial, wrong)
        wrong_geometry_same_identity = {**wrong, "source_rank": source["source_rank"],
                                        "source": source["source"], "source_sha256": source["source_sha256"]}
        assert not growth._pack_source_shape_audit(trial, wrong_geometry_same_identity)["passed"]
        damaged = {**trial, "coordinates": trial["coordinates"].copy()}
        damaged["coordinates"][1] += [0.1, 0., 0.]
        assert not growth._pack_source_shape_audit(damaged, source)["passed"]
        damaged_body = {**trial, "coordinates": trial["coordinates"].copy()}
        damaged_body["coordinates"][6] += [0.15, 0., 0.]
        assert not growth._pack_source_shape_audit(damaged_body, source)["passed"]
        reflected = {**trial, "coordinates": trial["coordinates"].copy()}
        reflected["coordinates"][[1, 2]] = reflected["coordinates"][[2, 1]]
        reflection_audit = growth._pack_source_shape_audit(reflected, source)
        assert not reflection_audit["passed"]
        assert not reflection_audit["chirality_preserved"]


def test_mixed_pool_topology_guard_rejects_atom_order_or_connectivity_mismatch(tmp_path):
    sources, _, _, _, _, _ = mixed_registered_fixture(tmp_path)
    growth._pack_validate_conformer_pool(sources)
    reordered = copy.deepcopy(sources)
    reordered[1]["symbols"][[1, 4]] = reordered[1]["symbols"][[4, 1]]
    with pytest.raises(ValueError):
        growth._pack_validate_conformer_pool(reordered)
    broken = copy.deepcopy(sources)
    broken[1]["coordinates"][1] += [8., 0., 0.]
    with pytest.raises(ValueError):
        growth._pack_validate_conformer_pool(broken)


def test_mixed_pool_preserves_source_oxygen_enumeration_but_requires_same_anchor_set(tmp_path):
    sources, _, _, _, _, _ = mixed_registered_fixture(tmp_path)
    sources[1]["o_locals"] = [3, 1, 2]
    audit = growth._pack_validate_conformer_pool(sources)
    assert audit["passed"]
    assert audit["sources"][0]["o_locals"] == [1, 2, 3]
    assert audit["sources"][1]["o_locals"] == [3, 1, 2]
    assert sources[1]["o_locals"] == [3, 1, 2]
    # Same full atom order/graph, but a different O atom is incorrectly assigned
    # to the anchor in the second source. Distinguish identity from enumeration.
    for source in sources:
        source["symbols"] = np.append(source["symbols"], "O")
        source["coordinates"] = np.vstack((source["coordinates"], [9., 9., 9.]))
    assert growth._pack_validate_conformer_pool(sources)["passed"]
    sources[1]["o_locals"] = [1, 2, 7]
    with pytest.raises(ValueError, match="anchor mapping"):
        growth._pack_validate_conformer_pool(sources)


def minimal_pack_cli(tmp_path):
    arguments = ["--pose-mode", "pack-then-bind", "--molecule-formula-json", '{"P":1,"O":3,"C":3}']
    for name in ["substrate", "layer-groups", "site-instances", "site-prototypes",
                 "conformer-glob", "conformer-analysis", "output-dir"]:
        arguments.extend([f"--{name}", str(tmp_path/name)])
    return arguments


def test_mixed_mode_rejects_independent_prefix_without_explicit_branch_rank(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        growth.sequential_growth_main([*minimal_pack_cli(tmp_path), "--pack-replay-prefix", str(tmp_path/"old")])
    assert error.value.code == 2
    assert "explicit --branch-rank" in capsys.readouterr().err


@pytest.mark.parametrize("branch_rank,expected_ranks", [(None, {1, 2}), (2, {2})])
def test_driver_default_hands_joint_pool_to_one_growth_state_and_explicit_rank_filters_it(
    tmp_path, monkeypatch, branch_rank, expected_ranks
):
    """Exercise actual entry and domain builder; isolate only front-end I/O and stop before growth."""
    sources, base, payload, prototypes, coordination, metrics = mixed_registered_fixture(tmp_path)
    args = growth._build_parser().parse_args(minimal_pack_cli(tmp_path))
    for key, value in vars(config()).items():
        setattr(args, key, value)
    args.branch_rank = branch_rank
    args.azimuth_step_deg = 180.
    args.substrate_height_clearance_A = 1.
    args.single_metal_sites = "off"
    args.molecule_formula_json = '{"P":1,"O":3,"C":3}'
    args.max_conformers = 2
    paths = {name: tmp_path/f"{name}.json" for name in [
        "substrate", "layer_groups", "site_instances", "site_prototypes", "conformer_analysis"]}
    for path in paths.values():
        path.write_text('{}')
    paths["site_instances"].write_text(json.dumps(payload))
    paths["site_prototypes"].write_text(json.dumps({"site_prototypes": list(prototypes.values())}))
    records = [{"selection_rank": rank, **metric, "footprint_area_A2": 12.,
                "structure": {"path": sources[rank-1]["source"]}}
               for rank, metric in metrics.items()]
    monkeypatch.setattr(growth, "_load_working_substrate", lambda *a: (base.copy(), {"working_layer_atom_ids_1based": [1,2,3]}))
    monkeypatch.setattr(growth, "resolve_periodic_fractional_axes", lambda *a: (0, 1))
    monkeypatch.setattr(growth, "_validated_layer_normal_axis", lambda *a: 2)
    monkeypatch.setattr(growth, "calculate_substrate_metal_coordinations", lambda **k: coordination)
    monkeypatch.setattr(growth, "_selected_analysis_records", lambda *a: records)
    monkeypatch.setattr(growth, "_selected_conformer_evidence", lambda *a, **k: {})
    monkeypatch.setattr(growth, "_lock_selected_conformer_paths", lambda **k: {})
    monkeypatch.setattr(growth, "_verify_locked_conformer_sources", lambda *a, **k: None)
    monkeypatch.setattr(growth, "_audit_loaded_conformer_ensemble", lambda *a, **k: [
        {"selection_rank": source["source_rank"], "sam_atom_indices_0based": list(range(1, 8))}
        for source in sources])
    monkeypatch.setattr(growth, "source_conformers", lambda *a, **k: copy.deepcopy(sources))
    monkeypatch.setattr(growth, "_code_snapshot_plan", lambda *a: {})
    monkeypatch.setattr(growth, "_write_code_snapshot", lambda *a: None)
    calls = []
    class CapturedGrowthBoundary(Exception):
        pass
    def observe(**kwargs):
        calls.append(kwargs)
        raise CapturedGrowthBoundary()
    monkeypatch.setattr(growth, "run_pack_then_bind_domains", observe)
    output = tmp_path/"output"
    output.mkdir()
    with pytest.raises(CapturedGrowthBoundary):
        growth.execute_pack_then_bind_growth(args, argv=[], root=tmp_path, output_dir=output,
                                            resolved_paths=paths, implementation_identity={})
    assert len(calls) == 1
    assert set(calls[0]["domains"]) == {"site"}
    pool = calls[0]["domains"]["site"]
    assert {candidate["selection_rank"] for candidate in pool} == expected_ranks
    assert len(pool) == 2*len(expected_ranks)
    assert calls[0].get("placed") in (None, [])
    request = json.loads((output/"run-request.json").read_text())
    assert len(request["conformers"]) == len(expected_ranks)
    assert request["conformer_selection_mode"] == ("joint" if branch_rank is None else "single_conformer_diagnostic")
