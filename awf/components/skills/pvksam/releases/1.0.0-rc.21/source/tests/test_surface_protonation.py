from pathlib import Path
import hashlib
import json
import sys

import numpy as np
from ase import Atoms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pytest
import surface_protonation as surface_protonation_module

from surface_protonation import (
    ProtonationConfig,
    audit_surface_parent_sites,
    evaluate_position,
    headgroup_indices,
    load_promoted_h0_parent,
    mic_vectors,
    single_candidates_for_sam,
)


def _config():
    return ProtonationConfig(
        substrate_atoms=1,
        molecule_count=1,
        atoms_per_molecule=1,
        protons_per_molecule=1,
    )


def test_evaluate_position_rejects_h_below_ito_surface_plane():
    atoms = Atoms(
        ["O", "O"],
        positions=[[5.0, 5.0, 4.0], [5.0, 5.0, 6.70]],
        cell=[20.0, 20.0, 20.0],
        pbc=True,
    )
    h_position = np.array([5.0, 5.0, 4.98])

    result = evaluate_position(
        atoms,
        parent_o=0,
        h_position=h_position,
        all_acceptors=np.array([0, 1]),
        head_acceptors=set(),
        minimum_h_z=5.10,
        config=_config(),
    )

    assert result is None


def test_evaluate_position_records_height_for_h_above_ito_surface_plane():
    atoms = Atoms(
        ["O", "O"],
        positions=[[5.0, 5.0, 5.20], [5.0, 5.0, 7.90]],
        cell=[20.0, 20.0, 20.0],
        pbc=True,
    )
    h_position = np.array([5.0, 5.0, 6.18])

    result = evaluate_position(
        atoms,
        parent_o=0,
        h_position=h_position,
        all_acceptors=np.array([0, 1]),
        head_acceptors=set(),
        minimum_h_z=5.10,
        config=_config(),
    )

    assert result is not None
    assert result["height_above_surface_plane_A"] == 1.08


def test_headgroup_uses_local_anchor_identity_when_anchor_element_is_not_unique():
    atoms = Atoms(
        ["O", "C", "O", "C"],
        positions=[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 0.0, 0.0]],
        cell=[20.0, 20.0, 20.0],
        pbc=True,
    )
    config = ProtonationConfig(
        substrate_atoms=1,
        molecule_count=1,
        atoms_per_molecule=3,
        protons_per_molecule=1,
        anchor_element="C",
        anchor_atom_index_within_molecule_0based=2,
        headgroup_count=1,
    )

    anchor, acceptors, block = headgroup_indices(atoms, 0, config)

    assert anchor == 3
    assert acceptors.tolist() == [2]
    assert block.tolist() == [1, 2, 3]


def test_surface_frame_height_uses_registered_normal_instead_of_global_z():
    atoms = Atoms(
        ["O", "O"],
        positions=[[5.0, 5.2, 5.0], [5.0, 7.9, 5.0]],
        cell=[20.0, 20.0, 20.0],
        pbc=True,
    )
    config = ProtonationConfig(
        substrate_atoms=1,
        molecule_count=1,
        atoms_per_molecule=1,
        protons_per_molecule=1,
        surface_normal=(0.0, 1.0, 0.0),
        surface_periodic_axes=(0, 2),
    )

    result = evaluate_position(
        atoms,
        parent_o=0,
        h_position=np.array([5.0, 6.18, 5.0]),
        all_acceptors=np.array([0, 1]),
        head_acceptors=set(),
        minimum_h_z=5.10,
        config=config,
    )

    assert result is not None
    assert result["height_above_surface_plane_A"] == pytest.approx(1.08)


def test_surface_mic_does_not_wrap_the_registered_slab_normal():
    vectors = mic_vectors(
        np.zeros(3),
        np.asarray([[0.0, 9.6, 0.0]]),
        np.diag([10.0, 10.0, 10.0]),
        periodic_axes=(0, 2),
    )

    assert vectors[0] == pytest.approx([0.0, 9.6, 0.0])


def test_candidate_direction_prefilter_keeps_closed_lower_roundoff_boundary(
    monkeypatch,
):
    def accepted_direction_count(distance_A):
        atoms = Atoms(
            ["O", "O", "P"],
            positions=[
                [0.0, 0.0, 0.0],
                [10.0 - distance_A, 0.0, 0.0],
                [0.0, 0.0, 3.0],
            ],
            cell=[10.0, 10.0, 20.0],
            pbc=True,
        )
        config = ProtonationConfig(
            substrate_atoms=2,
            molecule_count=1,
            atoms_per_molecule=1,
            protons_per_molecule=1,
            anchor_element="P",
            headgroup_count=0,
            acceptor_parent_min=1.7,
            acceptor_parent_max=2.8,
        )
        _, diagnostics = single_candidates_for_sam(
            atoms,
            molecule_number=0,
            substrate_parent_atoms=np.array([0]),
            all_acceptors=np.array([0, 1]),
            parents_per_sam=1,
            directions_per_parent=10,
            minimum_h_z=0.0,
            config=config,
        )
        return diagnostics["parent_diagnostics"][0]["generated_direction_count"]

    monkeypatch.setattr(
        surface_protonation_module,
        "evaluate_position",
        lambda *_args, **_kwargs: {
            "coordination_count": 1,
            "interaction_score": 0.0,
            "minimum_clearance_ratio": 1.0,
            "interactions": [],
        },
    )

    exact_count = accepted_direction_count(1.7)
    below_count = accepted_direction_count(1.7 - 1.0e-12)

    assert exact_count > below_count


def _write_promotion_fixture(tmp_path, *, promoted=True):
    tmp_path.mkdir(parents=True, exist_ok=True)
    structure = tmp_path / "promoted-h0.extxyz"
    structure.write_text("sealed H0 fixture\n")
    validation = tmp_path / "physical-interface-validation.json"
    validation.write_text(
        json.dumps(
            {
                "physical_interface_gate": {
                    "physical_interface_pass": promoted,
                }
            }
        )
        + "\n"
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "samflow-sized-h0-physical-interface-promotion-v1",
                "schema_version": 1,
                "chemistry_state": "H0_surface_protons_zero",
                "physical_interface_pass": promoted,
                "promotion_eligible": promoted,
                "parent": {
                    "approval_plan_sha256": "a" * 64,
                    "approval_implementation_identities": {
                        "surface_protonation_implementation_sha256": hashlib.sha256(
                            Path(surface_protonation_module.__file__).read_bytes()
                        ).hexdigest(),
                    },
                },
                "structure": {
                    "path": structure.name,
                    "sha256": hashlib.sha256(structure.read_bytes()).hexdigest(),
                },
                "validation": {
                    "path": validation.name,
                    "sha256": hashlib.sha256(validation.read_bytes()).hexdigest(),
                },
            }
        )
        + "\n"
    )
    return structure, manifest


def test_protonation_accepts_only_hash_verified_physical_promotion(tmp_path):
    structure, manifest = _write_promotion_fixture(tmp_path)

    parent = load_promoted_h0_parent(structure, manifest)

    assert parent["promotion_eligible"] is True
    assert parent["physical_interface_pass"] is True
    assert parent["parent"]["approval_plan_sha256"] == "a" * 64


def test_protonation_rejects_failed_or_tampered_physical_promotion(tmp_path):
    failed_structure, failed_manifest = _write_promotion_fixture(
        tmp_path / "failed", promoted=False
    )
    with pytest.raises(ValueError, match="not promotion eligible"):
        load_promoted_h0_parent(failed_structure, failed_manifest)

    structure, manifest = _write_promotion_fixture(tmp_path / "tampered")
    structure.write_text("tampered\n")
    with pytest.raises(ValueError, match="structure hash mismatch"):
        load_promoted_h0_parent(structure, manifest)


def test_parent_site_default_rejects_recessed_o_and_keeps_high_o():
    atoms = Atoms("OOIn", positions=[[1, 1, 2], [2, 1, 3], [1, 1, 1]],
                  cell=[20, 20, 20])
    parents, audit = audit_surface_parent_sites(atoms, 3)
    assert parents.tolist() == [1]
    assert audit["counts"] == {"substrate_atoms": 3, "parent_O_total": 2,
                               "eligible": 1, "excluded": 1, "geometrically_recessed": 1}
    assert audit["rows"][0]["depth_A"] == 1.0
    assert audit["rows"][0]["local_top_atom_1based"] == 2
    assert audit["policy"]["mode"] == "local-upper-envelope"
    assert audit["policy"]["height_tolerance_A"] == 1.0e-7


def test_parent_site_envelope_includes_covering_metal_not_only_oxygen():
    atoms = Atoms("OInSn", positions=[[1, 1, 2], [2, 1, 2.5], [3, 1, 3]],
                  cell=[20, 20, 20])
    parents, audit = audit_surface_parent_sites(atoms, 3)
    assert parents.tolist() == []
    assert audit["rows"][0]["local_top_element"] == "Sn"
    assert audit["rows"][0]["depth_A"] == 1.0


def test_parent_site_xy_boundary_uses_periodic_image_without_z_wrapping():
    atoms = Atoms("OOIn", positions=[[0.1, 0.1, 2], [9.9, 9.9, 3], [0.1, 0.1, -8]],
                  cell=[10, 10, 10], pbc=True)
    parents, audit = audit_surface_parent_sites(atoms, 3, radius_A=0.5)
    assert parents.tolist() == [1]
    assert audit["rows"][0]["local_top_horizontal_distance_A"] == pytest.approx(np.sqrt(0.08))
    assert audit["rows"][1]["depth_A"] == 0.0


@pytest.mark.parametrize("lateral_distance,eligible", [(3.0, False), (3.0 + 1.0e-8, True)])
def test_parent_site_lateral_cutoff_preserves_closed_roundoff_boundary(lateral_distance, eligible):
    atoms = Atoms("OO", positions=[[8.5, 10, 2], [8.5 + lateral_distance, 10, 3]],
                  cell=[40, 30, 25])
    parents, audit = audit_surface_parent_sites(atoms, 2)
    assert (0 in parents) is eligible
    assert audit["policy"]["lateral_numerical_tolerance_A"] == 1.0e-10


@pytest.mark.parametrize("height_difference,eligible", [(0.5e-7, True), (2e-7, False)])
def test_parent_site_height_tolerance_is_numerical(height_difference, eligible):
    atoms = Atoms("OIn", positions=[[1, 1, 2], [2, 1, 2 + height_difference]],
                  cell=[20, 20, 20])
    parents, audit = audit_surface_parent_sites(atoms, 2)
    assert bool(len(parents)) is eligible
    assert audit["rows"][0]["geometry_eligible"] is eligible


@pytest.mark.parametrize("radius", [0.0, -1.0, np.nan, np.inf, True, "3.0"])
def test_parent_site_rejects_invalid_radius(radius):
    atoms = Atoms("O", positions=[[1, 1, 2]], cell=[20, 20, 20])
    with pytest.raises(ValueError, match="radius"):
        audit_surface_parent_sites(atoms, 1, radius_A=radius)


@pytest.mark.parametrize("normal", [(0, 0, 0), (0, 0, np.nan), (1, 0, 0), (0, 0)])
def test_parent_site_rejects_invalid_or_incompatible_normal(normal):
    atoms = Atoms("O", positions=[[1, 1, 2]], cell=[20, 20, 20])
    with pytest.raises(ValueError, match="normal"):
        audit_surface_parent_sites(atoms, 1, normal=normal)


def test_parent_site_rejects_nonfinite_positions_singular_cell_and_unknown_policy():
    atoms = Atoms("O", positions=[[1, 1, np.nan]], cell=[20, 20, 20])
    with pytest.raises(ValueError, match="finite"):
        audit_surface_parent_sites(atoms, 1)
    atoms.positions[0, 2] = 2
    atoms.cell[2] = 0
    with pytest.raises(ValueError, match="nonsingular"):
        audit_surface_parent_sites(atoms, 1)
    atoms.set_cell([20, 20, 20])
    with pytest.raises(ValueError, match="Unknown"):
        audit_surface_parent_sites(atoms, 1, policy="disable")


def test_parent_site_rejects_unregistered_substrate_element():
    atoms = Atoms("OZn", positions=[[1, 1, 2], [2, 1, 3]], cell=[20, 20, 20])
    with pytest.raises(ValueError, match="In/Sn/O"):
        audit_surface_parent_sites(atoms, 2)


def test_parent_site_legacy_pool_requires_explicit_historical_policy():
    atoms = Atoms("OOIn", positions=[[1, 1, 2], [2, 1, 3], [1, 1, 1]],
                  cell=[20, 20, 20])
    parents, audit = audit_surface_parent_sites(atoms, 3, policy="legacy-unfiltered")
    assert parents.tolist() == [0, 1]
    assert audit["rows"][0]["geometry_eligible"] is False
    assert audit["rows"][0]["eligible"] is True
    assert audit["policy"]["legacy_reproduction"] is True


def test_candidate_fallback_cannot_reintroduce_excluded_parent(monkeypatch):
    atoms = Atoms("OOInP", positions=[[0, 0, 2], [10, 0, 4], [0, 0, 3], [20, 20, 10]],
                  cell=[40, 40, 30])
    parents, _ = audit_surface_parent_sites(atoms, 3)
    assert parents.tolist() == [1]
    config = ProtonationConfig(substrate_atoms=3, molecule_count=1,
                              atoms_per_molecule=1, protons_per_molecule=1,
                              headgroup_count=0)
    evaluated_parents = []
    def evaluate(_atoms, parent, *_args):
        evaluated_parents.append(parent)
        return {"coordination_count": 0, "interaction_score": 0.0,
                "minimum_clearance_ratio": 1.0, "interactions": []}
    monkeypatch.setattr(surface_protonation_module, "evaluate_position", evaluate)
    singles, diagnostic = single_candidates_for_sam(
        atoms, 0, parents, np.array([0, 1]), 24, 4, -100.0, config)
    assert singles and diagnostic["local_parent_count"] == 1
    assert set(evaluated_parents) == {1}
    assert {h["parent_atom"] for h in singles} == {1}


def _small_pvksam_source(tmp_path, *, recessed=False, family="phosphonate"):
    from ase.io import write
    from relax_monolayer import build_pvksam_restraints
    atoms = Atoms("InOO", positions=[[10, 10, 2], [8.5, 10, 2 if recessed else 3],
                                     [11.5, 10, 3]], cell=[40, 30, 25], pbc=[True, True, False])
    if family == "phosphonate":
        directions = np.array([[np.sqrt(8/9) * np.cos(x), np.sqrt(8/9) * np.sin(x), -1/3]
                               for x in [0, 2*np.pi/3, 4*np.pi/3]])
        molecule = Atoms("POOOC", positions=np.vstack([[10, 10, 6],
                         np.array([10, 10, 6]) + 1.5*directions, [10, 10, 7.82]]))
    else:
        molecule = Atoms("COOC", positions=[[10, 10, 6], [8.9, 10, 5.35],
                                            [11.1, 10, 5.35], [10, 10, 7.5]])
    atoms += molecule
    spec = build_pvksam_restraints(atoms, 3, 1, len(molecule),
                                  {"bonds": [{"metal": 1, "donor": 5, "target_A": 2.29}]})
    source = tmp_path / "h0.extxyz"
    write(source, atoms)
    return source, spec, len(molecule)


def test_pvksam_insufficient_eligible_parents_fails_before_writing_structure(tmp_path):
    source, spec, size = _small_pvksam_source(tmp_path, recessed=True)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="Insufficient eligible"):
        surface_protonation_module.protonate_restrained_h0(source, output,
            substrate_atoms=3, molecules=1, atoms_per_molecule=size, sealed_spec=spec)
    assert (output / "failure.json").exists()
    audit = json.loads((output / "parent-site-audit.json").read_text())
    assert audit["counts"]["eligible"] == 1
    assert not (output / "manifest.json").exists()
    assert not list(output.glob("protonated-interface.*"))


def test_pvksam_selected_parent_independently_rechecked_after_solver(tmp_path, monkeypatch):
    source, spec, size = _small_pvksam_source(tmp_path, recessed=True, family="carboxylate")
    output = tmp_path / "output"
    bad_group = {"protons": [{"parent_atom": 1, "position": [8.5, 10, 2.98]}]}
    monkeypatch.setattr(surface_protonation_module, "proton_group_candidates_for_sam",
                        lambda *_args: [bad_group])
    monkeypatch.setattr(surface_protonation_module, "solve_global_groups",
                        lambda *_args: ([bad_group], {}))
    with pytest.raises(ValueError, match="independent parent-site policy"):
        surface_protonation_module.protonate_restrained_h0(source, output,
            substrate_atoms=3, molecules=1, atoms_per_molecule=size, sealed_spec=spec)
    assert (output / "failure.json").exists()
    assert not (output / "manifest.json").exists()
    assert not list(output.glob("protonated-interface.*"))


def test_pvksam_default_records_audit_conserves_protons_and_preserves_h0(tmp_path):
    from ase.io import read
    source, spec, size = _small_pvksam_source(tmp_path)
    output = tmp_path / "output"
    result = surface_protonation_module.protonate_restrained_h0(source, output,
        substrate_atoms=3, molecules=1, atoms_per_molecule=size, sealed_spec=spec)
    original, post = read(source), read(output / "protonated-interface.extxyz")
    assert result["surface_H_count"] == 2 and len(post) == len(original) + 2
    assert np.array_equal(post.positions[:3], original.positions[:3])
    assert np.array_equal(post.positions[5:], original.positions[3:])
    assert result["surface_OH_restraints"] == 0 and result["H_height_policy"] == "none"
    assert result["parent_site_policy"]["mode"] == "local-upper-envelope"
    audit = json.loads((output / "parent-site-audit.json").read_text())
    assert audit["counts"]["selected"] == 2
    assert audit["independent_selected_site_check"]["coordinate_recomputed"] is True
    assert all(row["eligible"] for row in audit["rows"] if row["selected"])
    assert result["parent_site_audit"]["sha256"] == hashlib.sha256(
        (output / "parent-site-audit.json").read_bytes()).hexdigest()


def test_pvksam_explicit_legacy_mode_retains_unfiltered_parent_inventory(tmp_path):
    source, spec, size = _small_pvksam_source(tmp_path)
    output = tmp_path / "output"
    result = surface_protonation_module.protonate_restrained_h0(source, output,
        substrate_atoms=3, molecules=1, atoms_per_molecule=size, sealed_spec=spec,
        parent_site_policy="legacy-unfiltered")
    assert result["surface_H_count"] == 2
    assert result["parent_site_policy"]["legacy_reproduction"] is True
    assert result["parent_site_audit"]["counts"]["eligible"] == 2
