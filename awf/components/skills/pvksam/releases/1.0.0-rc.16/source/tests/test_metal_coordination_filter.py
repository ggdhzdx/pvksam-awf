"""Focused tests for pre-adsorption 6-coordinate In/Sn candidate and site filtering."""

from __future__ import annotations

import json
from pathlib import Path
import sys

_project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "scripts"))

import numpy as np
import pytest
from ase import Atoms
from ase.geometry import find_mic
from ase.io import read as ase_read

from sam_structure_tools import calculate_substrate_metal_coordinations
from monolayer_sequential_growth import (
    audit_site_instance_metal_coordinations,
    build_metal_coordination_filter_contract,
    build_full_to_working_atom_map,
    _candidate_domains,
    _candidate_domain_fingerprint,
    _prototype_map,
    MoleculeSpec,
)


@pytest.fixture
def working_substrate_and_metadata():
    substrate_path = _project_root / "runs/single-metal-triangle-hole-insertion/dbf34-n88-preserve-largest-residual-hole-20260923T080155Z/preserve-largest-hole-added-to-N88.extxyz"
    groups_path = _project_root / "runs/site-instance-materialization-validation/pubchem-cid-13818-3d-ito/substrate-2f8decd47195/layer-groups.json"
    sites_path = _project_root / "runs/site-instance-materialization-validation/pubchem-cid-13818-3d-ito/substrate-2f8decd47195/adsorption-sites.json"
    proto_path = _project_root / "substrates/structures/ito/adsorption-sites/phosphonic_acid/adsorption_site_prototype_clustering_v1-10d5c9fed77c/site-prototypes.json"

    if not substrate_path.exists() or not groups_path.exists() or not sites_path.exists() or not proto_path.exists():
        pytest.skip("Required reference substrate/site fixture files not found")

    full_atoms = ase_read(str(substrate_path), format="extxyz")
    substrate = full_atoms[:2560]
    groups = json.loads(groups_path.read_text())
    sites_doc = json.loads(sites_path.read_text())
    prototypes = _prototype_map(proto_path)

    full_to_working = build_full_to_working_atom_map(
        working_layer_atom_ids_1based=groups["working_layer_atom_ids_1based"],
        working_symbols=substrate.get_chemical_symbols(),
    )

    return {
        "substrate": substrate,
        "groups": groups,
        "sites_doc": sites_doc,
        "prototypes": prototypes,
        "full_to_working": full_to_working,
    }


def test_2d_mic_oblique_cell():
    """Verify 2D minimum-image convention on an oblique cell with non-wrapping normal axis."""
    # Oblique in-plane lattice vectors with z-normal
    cell = np.array([
        [10.0, 0.0, 0.0],
        [5.0, 8.660254, 0.0],  # 60-degree angle with a
        [0.0, 0.0, 30.0],      # non-periodic slab normal
    ])
    pbc = np.array([True, True, False])

    # Point 1 at origin, Point 2 across the oblique periodic boundary
    pos1 = np.array([0.0, 0.0, 10.0])
    pos2 = np.array([9.5, 0.1, 10.0])

    diff = pos2 - pos1
    _, d_wrapped = find_mic(diff, cell, pbc=pbc)
    # The shortest distance across PBC is ~0.51 A (pos2 is near the [10, 0, 0] image of pos1)
    assert float(d_wrapped) < 1.0

    # Along the z axis (normal_axis=2), PBC must NOT wrap
    pos_z_far = np.array([0.0, 0.0, 25.0])
    diff_z = pos_z_far - pos1
    _, d_z = find_mic(diff_z, cell, pbc=pbc)
    assert np.isclose(float(d_z), 15.0)


def test_calculate_substrate_metal_coordinations(working_substrate_and_metadata):
    """Verify calculate_substrate_metal_coordinations correctly audits doped ITO working substrate."""
    data = working_substrate_and_metadata
    substrate = data["substrate"]
    full_to_working = data["full_to_working"]

    audit = calculate_substrate_metal_coordinations(
        substrate_positions=substrate.positions,
        substrate_symbols=substrate.get_chemical_symbols(),
        cell=substrate.cell,
        periodic_axes=(0, 1),
        normal_axis=2,
        cutoff_A=2.7,
        target_elements=("In", "Sn"),
        ligand_elements=("O",),
        full_to_working_atom_map=full_to_working,
    )

    inv = audit["inventory"]
    assert inv["total_metal_count"] > 0
    assert inv["undercoordinated_count"] + inv["six_coordinated_count"] == inv["total_metal_count"]
    assert "In" in inv["by_element"]
    assert len(audit["metal_records_by_full_id"]) == inv["total_metal_count"]


def test_site_instance_coordination_filter_rejection_and_retention(working_substrate_and_metadata):
    """Verify site instance pre-filtering accurately identifies undercoordinated vs 6-coordinate sites."""
    data = working_substrate_and_metadata
    substrate = data["substrate"]
    sites_doc = data["sites_doc"]
    instances = sites_doc["site_instances"]
    full_to_working = data["full_to_working"]

    coord_data = calculate_substrate_metal_coordinations(
        substrate_positions=substrate.positions,
        substrate_symbols=substrate.get_chemical_symbols(),
        cell=substrate.cell,
        periodic_axes=(0, 1),
        normal_axis=2,
        cutoff_A=2.7,
        target_elements=("In", "Sn"),
        ligand_elements=("O",),
        full_to_working_atom_map=full_to_working,
    )

    retained = 0
    excluded = 0
    by_denticity = {2: {"retained": 0, "excluded": 0}, 3: {"retained": 0, "excluded": 0}}

    for inst in instances:
        res = audit_site_instance_metal_coordinations(
            instance=inst,
            metal_records_by_full_id=coord_data["metal_records_by_full_id"],
            max_allowed_coordination=5,
        )
        dent = res["denticity"]
        if res["passed"]:
            retained += 1
            by_denticity[dent]["retained"] += 1
        else:
            excluded += 1
            by_denticity[dent]["excluded"] += 1

    total = len(instances)
    assert total == 448
    # Dynamic verification on the reference ITO structure:
    assert retained == 224
    assert excluded == 224
    assert by_denticity[3]["retained"] == 192
    assert by_denticity[3]["excluded"] == 192
    assert by_denticity[2]["retained"] == 32
    assert by_denticity[2]["excluded"] == 32


def test_site_instance_audit_rejects_mismatched_metal_ids():
    """Verify audit_site_instance_metal_coordinations rejects mismatch between donor mapping and actual IDs."""
    instance = {
        "site_instance_id": "test-site-001",
        "donor_metal_mapping": [
            {"donor_label": "O1", "metal_atom_id_1based": 10},
            {"donor_label": "O2", "metal_atom_id_1based": 11},
        ],
        "actual_metal_atom_ids_1based": [10, 12],  # mismatch: 12 vs 11
        "final_denticity": 2,
    }
    dummy_records = {
        10: {"working_atom_index_0based": 0, "element": "In", "coordination_number": 5},
        11: {"working_atom_index_0based": 1, "element": "In", "coordination_number": 5},
        12: {"working_atom_index_0based": 2, "element": "In", "coordination_number": 5},
    }
    with pytest.raises(ValueError, match="disagree with"):
        audit_site_instance_metal_coordinations(
            instance=instance,
            metal_records_by_full_id=dummy_records,
            max_allowed_coordination=5,
        )


def test_candidate_domain_filtering_off_vs_active(working_substrate_and_metadata):
    """Verify candidate domain generation cleanly skips excluded sites in active mode while preserving off mode."""
    data = working_substrate_and_metadata
    substrate = data["substrate"]
    sites_doc = data["sites_doc"]
    prototypes = data["prototypes"]
    full_to_working = data["full_to_working"]

    # Minimal dummy conformer for domain building
    spec = MoleculeSpec(
        formula={"P": 1, "O": 4, "C": 1, "H": 3},
        skeleton_formula={"C": 1, "H": 3},
        anchor_element="P",
    )
    # 1 P atom, 3 donor O atoms, 1 terminal O, 1 C, 3 H
    conf_coords = np.array([
        [0.0, 0.0, 0.0],     # P (0)
        [1.5, 0.0, -0.5],    # O1 (1)
        [-0.75, 1.3, -0.5],  # O2 (2)
        [-0.75, -1.3, -0.5], # O3 (3)
        [0.0, 0.0, 1.5],     # O4 (4)
        [0.0, 0.0, 3.0],     # C (5)
        [1.0, 0.0, 3.5],     # H (6)
        [-0.5, 0.8, 3.5],    # H (7)
        [-0.5, -0.8, 3.5],   # H (8)
    ])
    conf_symbols = ["P", "O", "O", "O", "O", "C", "H", "H", "H"]
    conformers = [{
        "source": "dummy-conf-01",
        "source_rank": 0,
        "coordinates": conf_coords,
        "symbols": conf_symbols,
        "p_local": 0,
        "o_locals": [1, 2, 3],
    }]
    metrics_by_rank = {0: {"cluster_id": 0, "relative_total_energy_eV": 0.0}}

    contract_off = build_metal_coordination_filter_contract(mode="off")
    contract_active = build_metal_coordination_filter_contract(
        mode="exclude-six-coordinated",
        cutoff_A=2.7,
        max_allowed_coordination=5,
    )

    domains_off, diag_off = _candidate_domains(
        substrate=substrate,
        site_payload=sites_doc,
        prototypes=prototypes,
        conformers=conformers,
        metrics_by_rank=metrics_by_rank,
        spec=spec,
        headgroup_rmsd_max_A=0.8,
        full_to_working_atom_map=full_to_working,
        substrate_collision_mode="skip",
        metal_coordination_contract=contract_off,
    )

    domains_active, diag_active = _candidate_domains(
        substrate=substrate,
        site_payload=sites_doc,
        prototypes=prototypes,
        conformers=conformers,
        metrics_by_rank=metrics_by_rank,
        spec=spec,
        headgroup_rmsd_max_A=0.8,
        full_to_working_atom_map=full_to_working,
        substrate_collision_mode="skip",
        metal_coordination_contract=contract_active,
    )

    assert diag_off["metal_coordination_filter"]["enabled"] is False
    assert diag_active["metal_coordination_filter"]["enabled"] is True

    site_summary = diag_active["metal_coordination_filter"]["site_instance_summary"]
    assert site_summary["total_site_instances_before_filter"] == 448
    assert site_summary["retained_site_instances"] == 224
    assert site_summary["excluded_site_instances"] == 224

    # The 224 excluded sites must have empty candidate domain lists
    empty_sites = [sid for sid, cands in domains_active.items() if len(cands) == 0]
    assert len(empty_sites) == 224

    # Distinct fingerprints and namespaces
    assert diag_off["candidate_domain_fingerprint"] != diag_active["candidate_domain_fingerprint"]
    assert diag_active["candidate_domain_fingerprint_namespace"] == "metal-coordination-filter-v1"
