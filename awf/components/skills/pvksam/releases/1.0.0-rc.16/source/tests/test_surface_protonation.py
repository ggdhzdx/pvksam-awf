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
