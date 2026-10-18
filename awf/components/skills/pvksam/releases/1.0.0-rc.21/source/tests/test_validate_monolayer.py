from pathlib import Path
import json
import sys

import numpy as np
import pytest
from ase import Atoms
from ase.io import write


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from validate_monolayer import (
    ValidationConfig,
    _normalize_registered_interface_bonds,
    classify_surface_proton,
    is_allowed_interface_pair,
    mic_vectors,
    molecule_component_size,
    parse_element_pair,
    validate,
)


def test_allowed_pair_is_unordered_and_interface_only():
    allowed = {frozenset(parse_element_pair("O-P"))}
    assert is_allowed_interface_pair("O", "P", 0, 11, allowed)
    assert is_allowed_interface_pair("P", "O", 8, 0, allowed)
    assert not is_allowed_interface_pair("O", "P", 8, 11, allowed)
    assert not is_allowed_interface_pair("C", "O", 0, 11, allowed)


def test_surface_proton_location_classification():
    assert classify_surface_proton(0.98, 2.0, 1.25) == "substrate"
    assert classify_surface_proton(1.05, 1.10, 1.25) == "shared"
    assert classify_surface_proton(1.40, 1.02, 1.25) == "sam"
    assert classify_surface_proton(1.40, 1.50, 1.25) == "unbound"


def test_h0_policy_rejects_nonzero_proton_inventory_before_reading(tmp_path):
    config = ValidationConfig(
        substrate_atoms=1,
        molecules=1,
        atoms_per_molecule=1,
        protons_per_molecule=1,
        surface_h_policy="h0",
    )

    try:
        validate(tmp_path / "absent.extxyz", config)
    except ValueError as exc:
        assert str(exc) == "surface_h_policy=h0 requires protons_per_molecule=0"
    else:
        raise AssertionError("h0 policy accepted a nonzero proton inventory")


def _validate_two_heavy_atoms(
    tmp_path,
    *,
    separation_vector,
    heavy_heavy_min=1.8,
    surface_periodic_axes=(0, 1),
    name="candidate.extxyz",
):
    atoms = Atoms(
        symbols=["C", "P"],
        positions=[[0.0, 0.0, 0.0], list(separation_vector)],
        cell=np.diag([10.0, 10.0, 10.0]),
        pbc=True,
    )
    path = tmp_path / name
    write(path, atoms)
    return validate(
        path,
        ValidationConfig(
            substrate_atoms=1,
            molecules=1,
            atoms_per_molecule=1,
            protons_per_molecule=0,
            surface_h_policy="h0",
            heavy_heavy_min=heavy_heavy_min,
            surface_periodic_axes=surface_periodic_axes,
        ),
    )


def _triangle_interface_fixture(*, metal_xy=(5.0, 4.8), low_non_anchor=False):
    non_anchor_tip_z = 0.9 if low_non_anchor else 1.2
    substrate = Atoms(
        ["In", "O"],
        positions=[[metal_xy[0], metal_xy[1], 0.0], [0.0, 0.0, 0.2]],
        cell=[20.0, 20.0, 30.0],
        pbc=True,
    )
    sam = Atoms(
        ["P", "O", "O", "O", "C", "C", "C"],
        positions=[
            [5.0, 5.0, 3.9],
            [4.2, 4.2, 2.6],
            [5.8, 4.2, 2.6],
            [5.0, 5.8, 2.6],
            [7.0, 5.0, 3.9],
            [7.0, 5.0, 2.5],
            [7.0, 5.0, non_anchor_tip_z],
        ],
        cell=substrate.cell,
        pbc=True,
    )
    return substrate + sam


def _validate_triangle_interface(
    tmp_path,
    atoms,
    *,
    molecules=1,
    triangle_anchor_molecule_numbers=None,
    name="triangle.extxyz",
):
    path = tmp_path / name
    write(path, atoms)
    return validate(
        path,
        ValidationConfig(
            substrate_atoms=2,
            molecules=molecules,
            atoms_per_molecule=7,
            protons_per_molecule=0,
            anchor_element="P",
            anchor_atom_index_within_molecule_0based=0,
            surface_h_policy="h0",
            interface_pair_policy="phosphonate_triangle_height",
            mobile_interface_metal_elements=("In", "Sn"),
            surface_metal_depth_A=0.7,
            interface_height_clearance_A=1.0,
            triangle_anchor_molecule_numbers=triangle_anchor_molecule_numbers,
            expected_substrate_formula={"In": 1, "O": 1},
            expected_molecule_formula={"C": 3, "O": 3, "P": 1},
        ),
    )


def test_triangle_height_policy_passes_without_exact_oxygen_metal_pairs(tmp_path):
    result = _validate_triangle_interface(
        tmp_path, _triangle_interface_fixture()
    )

    assert result["passed"] is True
    assert result["checks"][
        "configured_triangle_anchor_sams_have_surface_metal_inside_anchor_o_triangle"
    ]
    assert result["checks"]["all_non_anchor_atoms_clear_surface_height"]
    assert result["registered_interface_bonds"] == []
    assert result["validation_policy"]["triangle_height_interface_contract"][
        "exact_oxygen_metal_pair_required"
    ] is False


def test_triangle_height_policy_rejects_metal_outside_triangle(tmp_path):
    result = _validate_triangle_interface(
        tmp_path,
        _triangle_interface_fixture(metal_xy=(8.0, 8.0)),
    )

    assert result["checks"][
        "configured_triangle_anchor_sams_have_surface_metal_inside_anchor_o_triangle"
    ] is False
    assert result["passed"] is False


def test_triangle_height_policy_rejects_non_anchor_atom_below_clearance(tmp_path):
    result = _validate_triangle_interface(
        tmp_path,
        _triangle_interface_fixture(low_non_anchor=True),
    )

    assert result["checks"]["all_non_anchor_atoms_clear_surface_height"] is False
    assert result["passed"] is False


def test_triangle_height_policy_keeps_sam_sam_collision_gate(tmp_path):
    one = _triangle_interface_fixture()
    substrate = one[:2]
    sam = one[2:]
    two = substrate + sam + sam.copy()
    result = _validate_triangle_interface(
        tmp_path, two, molecules=2, name="overlapping-sams.extxyz"
    )

    assert result["checks"]["no_sam_sam_collisions"] is False
    assert result["passed"] is False


def test_triangle_height_policy_allows_legacy_sams_with_mobile_contacts(tmp_path):
    first = _triangle_interface_fixture(metal_xy=(4.0, 4.1))
    substrate = first[:2] + Atoms(
        ["In"], positions=[[15.0, 4.8, 0.0]], cell=first.cell, pbc=True
    )
    first_sam = first[2:]
    second_sam = first_sam.copy()
    second_sam.translate([10.0, 0.0, 0.0])
    atoms = substrate + first_sam + second_sam
    path = tmp_path / "mixed-legacy-and-triangle.extxyz"
    write(path, atoms)

    result = validate(
        path,
        ValidationConfig(
            substrate_atoms=3,
            molecules=2,
            atoms_per_molecule=7,
            protons_per_molecule=0,
            anchor_element="P",
            anchor_atom_index_within_molecule_0based=0,
            surface_h_policy="h0",
            interface_pair_policy="phosphonate_triangle_height",
            mobile_interface_metal_elements=("In", "Sn"),
            surface_metal_depth_A=0.7,
            interface_height_clearance_A=1.0,
            triangle_anchor_molecule_numbers=(2,),
            surface_periodic_axes=(0, 1),
        ),
    )

    geometry = result["phosphonate_triangle_height_interface"]
    assert geometry["molecules"][0]["metal_inside_anchor_o_triangle"] is False
    assert geometry["molecules"][1]["metal_inside_anchor_o_triangle"] is True
    assert result["checks"][
        "configured_triangle_anchor_sams_have_surface_metal_inside_anchor_o_triangle"
    ] is True
    assert result["checks"]["all_other_sams_have_mobile_anchor_metal_contact"] is True
    assert result["checks"]["no_other_sam_mobile_interface_contacts_below_minimum"]
    assert result["checks"]["no_sam_sam_collisions"] is True
    assert result["passed"] is True


def test_validate_uses_maximum_configured_threshold_as_neighbor_cutoff(tmp_path):
    result = _validate_two_heavy_atoms(
        tmp_path,
        separation_vector=[2.0, 0.0, 0.0],
        heavy_heavy_min=2.2,
    )

    assert result["required_cross_component_collision_count"] == 1
    assert result["shortest_required_collisions"][0]["distance_A"] == pytest.approx(2.0)
    assert result["shortest_required_collisions"][0]["threshold_A"] == pytest.approx(2.2)
    contract = result["validation_policy"]["collision_contract"]
    assert contract["neighbor_search_cutoff_A"] == pytest.approx(2.2)
    assert contract["thresholds_A"] == {
        "H-H": pytest.approx(1.2),
        "H-heavy": pytest.approx(1.4),
        "heavy-heavy": pytest.approx(2.2),
    }


def test_validate_preserves_old_1_8_angstrom_collision_behavior(tmp_path):
    outside = _validate_two_heavy_atoms(
        tmp_path,
        separation_vector=[2.0, 0.0, 0.0],
        name="outside.extxyz",
    )
    inside = _validate_two_heavy_atoms(
        tmp_path,
        separation_vector=[1.7, 0.0, 0.0],
        name="inside.extxyz",
    )

    assert outside["required_cross_component_collision_count"] == 0
    assert inside["required_cross_component_collision_count"] == 1
    assert inside["shortest_required_collisions"][0]["threshold_A"] == pytest.approx(1.8)


def test_validate_wraps_only_explicit_surface_periodic_axes(tmp_path):
    default_surface_axes = _validate_two_heavy_atoms(
        tmp_path,
        separation_vector=[0.0, 0.0, 9.6],
        name="normal-not-wrapped.extxyz",
    )
    explicitly_wrapped_axis = _validate_two_heavy_atoms(
        tmp_path,
        separation_vector=[0.0, 0.0, 9.6],
        surface_periodic_axes=(0, 2),
        name="explicit-axis-wrapped.extxyz",
    )

    assert default_surface_axes["required_cross_component_collision_count"] == 0
    assert explicitly_wrapped_axis["required_cross_component_collision_count"] == 1
    contract = default_surface_axes["validation_policy"]["collision_contract"]
    assert contract["surface_periodic_axes"] == [0, 1]
    assert contract["nonperiodic_cell_axes"] == [2]
    assert contract["slab_normal_axis"] == 2
    assert contract["slab_normal_wrapped"] is False
    explicit_contract = explicitly_wrapped_axis["validation_policy"][
        "collision_contract"
    ]
    assert explicit_contract["surface_periodic_axes"] == [0, 2]
    assert explicit_contract["slab_normal_axis"] == 1
    assert explicit_contract["slab_normal_wrapped"] is False

    skew_cell = np.asarray(
        [[10.0, 0.0, 0.0], [9.0, 1.0, 0.0], [0.0, 0.0, 10.0]]
    )
    fractional_displacement = np.asarray([0.49, 0.49, 0.90])
    surface_vector = mic_vectors(
        np.zeros(3),
        np.asarray([fractional_displacement @ skew_cell]),
        skew_cell,
        (0, 1),
    )[0]
    assert surface_vector[:2] == pytest.approx([0.31, -0.51])
    assert np.linalg.norm(surface_vector[:2]) == pytest.approx(
        np.hypot(0.31, 0.51)
    )
    assert surface_vector[2] == pytest.approx(9.0)

    # Existing three-argument callers retain the historical full 3-D MIC.
    legacy_vector = mic_vectors(
        np.zeros(3), np.asarray([[0.0, 0.0, 9.6]]), np.diag([10.0] * 3)
    )[0]
    assert legacy_vector == pytest.approx([0.0, 0.0, -0.4])
    boundary_molecule = Atoms(
        symbols=["P", "H"],
        positions=[[0.0, 0.0, 0.2], [0.0, 0.0, 9.8]],
        cell=np.diag([10.0] * 3),
        pbc=True,
    )
    assert molecule_component_size(
        boundary_molecule, np.asarray([0, 1]), "P"
    ) == 2


def _registered_bond_record(
    *, substrate_index=0, sam_index=2, minimum=1.7, maximum=2.8
):
    return {
        "substrate_atom_index_0based": substrate_index,
        "sam_atom_index_0based": sam_index,
        "minimum_distance_A": minimum,
        "maximum_distance_A": maximum,
        "expected_substrate_element": "In",
        "expected_sam_element": "O",
    }


def _validate_registered_fixture(
    tmp_path,
    *,
    donor_distance_A=1.75,
    maximum_distance_A=2.8,
    name="registered.extxyz",
    cell=None,
    positions=None,
    registered_bonds=None,
    expected_molecule_formula=None,
    allowed_interface_pairs=(),
):
    if cell is None:
        cell = np.diag([10.0, 10.0, 20.0])
    if positions is None:
        positions = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, donor_distance_A + 1.45],
            [0.0, 0.0, donor_distance_A],
        ]
    atoms = Atoms(
        ["In", "P", "O"], positions=positions, cell=cell, pbc=True
    )
    path = tmp_path / name
    write(path, atoms)
    if registered_bonds is None:
        registered_bonds = (
            _registered_bond_record(maximum=maximum_distance_A),
        )
    return validate(
        path,
        ValidationConfig(
            substrate_atoms=1,
            molecules=1,
            atoms_per_molecule=2,
            protons_per_molecule=0,
            surface_h_policy="h0",
            registered_interface_bonds=tuple(registered_bonds),
            expected_molecule_formula=(
                {"O": 1, "P": 1}
                if expected_molecule_formula is None
                else expected_molecule_formula
            ),
            allowed_interface_pairs=tuple(allowed_interface_pairs),
        ),
    )


def test_exact_registered_interface_pair_passes_only_inside_window(tmp_path):
    inside = _validate_registered_fixture(tmp_path, donor_distance_A=1.75)
    outside = _validate_registered_fixture(
        tmp_path, donor_distance_A=1.60, name="outside-window.extxyz"
    )

    assert inside["passed"] is True
    assert inside["checks"]["registered_interface_bonds_in_window"] is True
    assert inside["required_cross_component_collision_count"] == 0
    assert len(inside["registered_interface_contacts"]) == 1
    assert inside["registered_interface_contacts"][0]["distance_A"] == pytest.approx(1.75)
    assert inside["registered_interface_contacts"][0]["window_passed"] is True
    assert inside["legacy_allowed_interface_contacts"] == []

    assert outside["passed"] is False
    assert outside["checks"]["registered_interface_bonds_in_window"] is False
    assert outside["registered_interface_bonds"][0]["window_passed"] is False
    assert outside["required_cross_component_collision_count"] >= 1


def test_exact_registered_interface_pair_uses_closed_window_at_machine_precision(
):
    def normalized_bond(distance_A):
        atoms = Atoms(
            ["In", "P", "O"],
            positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 4.0], [distance_A, 0.0, 0.0]],
            cell=np.diag([10.0, 10.0, 20.0]),
            pbc=True,
        )
        normalized, _, _ = _normalize_registered_interface_bonds(
            (_registered_bond_record(),),
            symbols=np.asarray(atoms.get_chemical_symbols()),
            atom_count=len(atoms),
            substrate_atoms=1,
            atoms=atoms,
            periodic_axes=(0, 1),
        )
        return normalized[0]

    exact_lower = normalized_bond(1.7)
    below_lower = normalized_bond(1.7 - 1.0e-12)
    exact_upper = normalized_bond(2.8)
    above_upper = normalized_bond(2.8 + 1.0e-12)

    assert exact_lower["distance_A"] < 1.7
    assert exact_lower["window_passed"] is True
    assert exact_upper["window_passed"] is True
    assert below_lower["window_passed"] is False
    assert above_upper["window_passed"] is False


def test_mobile_interface_policy_accepts_new_metal_partner(tmp_path):
    atoms = Atoms(
        ["In", "Sn", "P", "O"],
        positions=[
            [4.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 3.35],
            [0.0, 0.0, 1.90],
        ],
        cell=np.diag([12.0, 12.0, 20.0]),
        pbc=True,
    )
    path = tmp_path / "mobile-partner-exchange.extxyz"
    write(path, atoms)
    registered = (
        {
            **_registered_bond_record(substrate_index=0, sam_index=3),
            "maximum_distance_A": 2.8,
        },
    )

    exact = validate(
        path,
        ValidationConfig(
            substrate_atoms=2,
            molecules=1,
            atoms_per_molecule=2,
            protons_per_molecule=0,
            surface_h_policy="h0",
            registered_interface_bonds=registered,
            expected_molecule_formula={"O": 1, "P": 1},
        ),
    )
    mobile = validate(
        path,
        ValidationConfig(
            substrate_atoms=2,
            molecules=1,
            atoms_per_molecule=2,
            protons_per_molecule=0,
            surface_h_policy="h0",
            interface_pair_policy="mobile",
            registered_interface_bonds=registered,
            expected_molecule_formula={"O": 1, "P": 1},
        ),
    )

    assert exact["passed"] is False
    assert mobile["checks"]["registered_interface_bonds_in_window"] is False
    assert "registered_interface_bonds_in_window" in mobile["validation_policy"][
        "diagnostic_checks"
    ]
    assert mobile["checks"]["all_SAM_molecules_have_mobile_interface_contact"] is True
    assert mobile["passed"] is True
    assert mobile["mobile_interface"]["fully_detached_molecules"] == []
    assert mobile["mobile_interface"]["contacts"] == [
        {
            "molecule": 1,
            "metal_atom_index_0based": 1,
            "metal_element": "Sn",
            "donor_atom_index_0based": 3,
            "donor_element": "O",
            "distance_A": pytest.approx(1.9),
        }
    ]


def test_mobile_interface_policy_keeps_closed_upper_roundoff_boundary(tmp_path):
    atoms = Atoms(
        ["In", "P", "O"],
        positions=[[0.0, 0.0, 0.0], [8.3, 0.0, 1.5], [8.3, 0.0, 0.0]],
        cell=np.diag([10.0, 10.0, 20.0]),
        pbc=True,
    )
    path = tmp_path / "mobile-upper-boundary.extxyz"
    write(path, atoms)

    result = validate(
        path,
        ValidationConfig(
            substrate_atoms=1,
            molecules=1,
            atoms_per_molecule=2,
            protons_per_molecule=0,
            surface_h_policy="h0",
            interface_pair_policy="mobile",
            mobile_interface_min=1.0,
            mobile_interface_max=1.7,
            expected_molecule_formula={"O": 1, "P": 1},
        ),
    )

    assert atoms.get_distance(0, 2, mic=True) < 1.7
    assert result["mobile_interface"]["contacts"][0]["distance_A"] == pytest.approx(
        1.7
    )
    assert result["checks"]["all_SAM_molecules_have_mobile_interface_contact"] is True
    assert result["mobile_interface"]["fully_detached_molecules"] == []


def test_mobile_interface_policy_keeps_closed_lower_roundoff_boundary(tmp_path):
    def validate_distance(distance_A, name):
        atoms = Atoms(
            ["In", "P", "O"],
            positions=[
                [0.0, 0.0, 0.0],
                [10.0 - distance_A, 0.0, 1.5],
                [10.0 - distance_A, 0.0, 0.0],
            ],
            cell=np.diag([10.0, 10.0, 20.0]),
            pbc=True,
        )
        path = tmp_path / f"{name}.traj"
        write(path, atoms)
        return atoms, validate(
            path,
            ValidationConfig(
                substrate_atoms=1,
                molecules=1,
                atoms_per_molecule=2,
                protons_per_molecule=0,
                surface_h_policy="h0",
                interface_pair_policy="mobile",
                mobile_interface_min=1.7,
                mobile_interface_max=2.8,
                expected_molecule_formula={"O": 1, "P": 1},
            ),
        )

    exact_atoms, exact = validate_distance(1.7, "mobile-exact-lower")
    _, below = validate_distance(1.7 - 1.0e-12, "mobile-below-lower")

    assert exact_atoms.get_distance(0, 2, mic=True) < 1.7
    assert len(exact["mobile_interface"]["contacts"]) == 1
    assert exact["mobile_interface"]["too_short_contacts"] == []
    assert exact["checks"]["all_SAM_molecules_have_mobile_interface_contact"] is True
    assert below["mobile_interface"]["contacts"] == []
    assert len(below["mobile_interface"]["too_short_contacts"]) == 1
    assert below["checks"]["no_mobile_interface_contacts_below_minimum"] is False


def test_mobile_interface_policy_rejects_fully_detached_molecule(tmp_path):
    atoms = Atoms(
        ["In", "Sn", "P", "O"],
        positions=[
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [2.5, 0.0, 5.0],
            [2.5, 0.0, 3.55],
        ],
        cell=np.diag([12.0, 12.0, 20.0]),
        pbc=True,
    )
    path = tmp_path / "mobile-detached.extxyz"
    write(path, atoms)

    result = validate(
        path,
        ValidationConfig(
            substrate_atoms=2,
            molecules=1,
            atoms_per_molecule=2,
            protons_per_molecule=0,
            surface_h_policy="h0",
            interface_pair_policy="mobile",
            registered_interface_bonds=(
                _registered_bond_record(substrate_index=0, sam_index=3),
            ),
            expected_molecule_formula={"O": 1, "P": 1},
        ),
    )

    assert result["checks"]["all_SAM_molecules_have_mobile_interface_contact"] is False
    assert result["mobile_interface"]["fully_detached_molecules"] == [1]
    assert result["passed"] is False


def test_chelate_registered_bonds_allow_two_unique_sam_donors_on_one_substrate_atom(
    tmp_path,
):
    atoms = Atoms(
        ["In", "P", "O", "O"],
        positions=[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 3.2],
            [0.8, 0.0, 1.85],
            [-0.8, 0.0, 1.85],
        ],
        cell=np.diag([10.0, 10.0, 20.0]),
        pbc=True,
    )
    path = tmp_path / "chelate.extxyz"
    write(path, atoms)
    bonds = (
        _registered_bond_record(sam_index=2),
        _registered_bond_record(sam_index=3),
    )

    result = validate(
        path,
        ValidationConfig(
            substrate_atoms=1,
            molecules=1,
            atoms_per_molecule=3,
            protons_per_molecule=0,
            surface_h_policy="h0",
            registered_interface_bonds=bonds,
            expected_molecule_formula={"O": 2, "P": 1},
        ),
    )

    assert result["passed"] is True
    assert len(result["registered_interface_bonds"]) == 2
    assert {
        record["substrate_atom_index_0based"]
        for record in result["registered_interface_bonds"]
    } == {0}
    assert {
        record["sam_atom_index_0based"]
        for record in result["registered_interface_bonds"]
    } == {2, 3}


def test_registered_bond_is_checked_beyond_collision_neighbor_cutoff(tmp_path):
    result = _validate_registered_fixture(
        tmp_path,
        donor_distance_A=2.5,
        maximum_distance_A=2.4,
        name="beyond-neighbor-cutoff.extxyz",
    )

    assert result["validation_policy"]["collision_contract"]["neighbor_search_cutoff_A"] == pytest.approx(1.8)
    assert result["cross_component_collision_count"] == 0
    assert result["checks"]["registered_interface_bonds_in_window"] is False
    assert result["registered_interface_bonds"][0]["distance_A"] == pytest.approx(2.5)
    assert result["passed"] is False


def test_same_element_nonmapped_interface_contact_is_not_exempt(tmp_path):
    atoms = Atoms(
        ["In", "In", "P", "O"],
        positions=[
            [0.0, 0.0, 0.0],
            [0.10, 0.0, 0.0],
            [0.0, 0.0, 3.20],
            [0.0, 0.0, 1.75],
        ],
        cell=np.diag([10.0, 10.0, 20.0]),
        pbc=True,
    )
    path = tmp_path / "same-element-nonmapped.extxyz"
    write(path, atoms)
    result = validate(
        path,
        ValidationConfig(
            substrate_atoms=2,
            molecules=1,
            atoms_per_molecule=2,
            protons_per_molecule=0,
            surface_h_policy="h0",
            registered_interface_bonds=(
                _registered_bond_record(sam_index=3),
            ),
            expected_molecule_formula={"O": 1, "P": 1},
        ),
    )

    assert result["checks"]["registered_interface_bonds_in_window"] is True
    assert len(result["registered_interface_contacts"]) == 1
    assert result["required_cross_component_collision_count"] == 1
    collision = result["shortest_required_collisions"][0]
    assert {collision["i"], collision["j"]} == {1, 3}
    assert result["passed"] is False


def test_registered_interface_never_exempts_sam_sam_collision(tmp_path):
    atoms = Atoms(
        ["In", "P", "O", "P", "O"],
        positions=[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 3.20],
            [0.0, 0.0, 1.75],
            [0.50, 0.0, 3.20],
            [0.50, 0.0, 1.75],
        ],
        cell=np.diag([10.0, 10.0, 20.0]),
        pbc=True,
    )
    path = tmp_path / "sam-sam.extxyz"
    write(path, atoms)
    result = validate(
        path,
        ValidationConfig(
            substrate_atoms=1,
            molecules=2,
            atoms_per_molecule=2,
            protons_per_molecule=0,
            surface_h_policy="h0",
            registered_interface_bonds=(_registered_bond_record(),),
            expected_molecule_formula={"O": 1, "P": 1},
        ),
    )

    assert result["registered_interface_contacts"]
    assert any(
        all(component != 0 for component in record["components"])
        for record in result["shortest_required_collisions"]
    )
    assert result["passed"] is False


def test_registered_interface_never_exempts_sam_sam_hydrogen_collision(tmp_path):
    atoms = Atoms(
        ["In", "P", "H", "P", "H"],
        positions=[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 3.20],
            [0.0, 0.0, 1.75],
            [0.50, 0.0, 3.20],
            [0.50, 0.0, 1.75],
        ],
        cell=np.diag([10.0, 10.0, 20.0]),
        pbc=True,
    )
    path = tmp_path / "sam-sam-h.extxyz"
    write(path, atoms)
    bond = {
        **_registered_bond_record(),
        "expected_sam_element": "H",
    }
    result = validate(
        path,
        ValidationConfig(
            substrate_atoms=1,
            molecules=2,
            atoms_per_molecule=2,
            protons_per_molecule=0,
            surface_h_policy="h0",
            registered_interface_bonds=(bond,),
            expected_molecule_formula={"H": 1, "P": 1},
        ),
    )

    assert result["checks"]["registered_interface_bonds_in_window"] is True
    assert any(
        record["elements"] == "H-H"
        and all(component != 0 for component in record["components"])
        for record in result["shortest_required_collisions"]
    )
    assert result["passed"] is False


def test_registered_interface_uses_true_surface_only_mic_in_skew_cell(tmp_path):
    cell = np.asarray(
        [[10.0, 0.0, 0.0], [4.0, 8.0, 0.0], [0.0, 0.0, 20.0]]
    )
    substrate = np.asarray([0.98, 0.98, 0.0]) @ cell
    donor = np.asarray([0.02, 0.02, 0.08]) @ cell
    anchor = donor + np.asarray([0.0, 0.0, 1.45])
    result = _validate_registered_fixture(
        tmp_path,
        name="skew.extxyz",
        cell=cell,
        positions=[substrate, anchor, donor],
    )
    expected = np.linalg.norm(np.asarray([0.04, 0.04, 0.08]) @ cell)

    assert result["registered_interface_bonds"][0]["distance_A"] == pytest.approx(expected)
    assert result["registered_interface_bonds"][0]["window_passed"] is True
    assert result["passed"] is True


@pytest.mark.parametrize(
    ("records", "message"),
    [
        (
            [_registered_bond_record(substrate_index=1)],
            "substrate atom index",
        ),
        (
            [_registered_bond_record(sam_index=0)],
            "SAM atom index",
        ),
        (
            [_registered_bond_record(sam_index=99)],
            "outside",
        ),
        (
            [_registered_bond_record(), _registered_bond_record()],
            "duplicate",
        ),
        (
            [
                {
                    **_registered_bond_record(),
                    "expected_substrate_element": "Sn",
                }
            ],
            "expected substrate element",
        ),
    ],
)
def test_registered_interface_schema_and_elements_fail_closed(
    tmp_path, records, message
):
    with pytest.raises(ValueError, match=message):
        _validate_registered_fixture(tmp_path, registered_bonds=records)


def test_expected_molecule_formula_is_checked_for_every_block(tmp_path):
    passed = _validate_registered_fixture(tmp_path)
    failed = _validate_registered_fixture(
        tmp_path,
        expected_molecule_formula={"O": 2, "P": 1},
        name="wrong-molecule-formula.extxyz",
    )

    assert passed["checks"]["each_molecule_matches_expected_formula"] is True
    assert failed["checks"]["each_molecule_matches_expected_formula"] is False
    assert failed["molecule_formula_mismatches"][0]["actual_formula"] == {
        "O": 1,
        "P": 1,
    }
    assert failed["passed"] is False


def test_legacy_allowed_interface_pair_remains_distinct_in_report(tmp_path):
    result = _validate_registered_fixture(
        tmp_path,
        registered_bonds=(),
        expected_molecule_formula={"O": 1, "P": 1},
        allowed_interface_pairs=(("In", "O"),),
        name="legacy-whitelist.extxyz",
    )

    assert result["registered_interface_contacts"] == []
    assert len(result["legacy_allowed_interface_contacts"]) == 1
    assert result["legacy_allowed_interface_contacts"][0]["classification"] == (
        "legacy_element_pair_whitelist"
    )
    assert len(result["allowed_interface_contacts"]) == 1
    assert "classification" not in result["allowed_interface_contacts"][0]
    assert result["validation_policy"]["legacy_allowed_interface_pairs"] == [
        "In-O"
    ]


def test_registered_interface_cli_json_and_expected_molecule_formula(tmp_path):
    from validate_monolayer import build_parser, load_registered_interface_bonds

    bond_path = tmp_path / "registered-bonds.json"
    bond_path.write_text(
        json.dumps({"registered_interface_bonds": [_registered_bond_record()]})
    )
    parser = build_parser()
    args = parser.parse_args(
        _required_cli_arguments()
        + [
            "--registered-interface-bonds-json",
            str(bond_path),
            "--expected-molecule-formula-json",
            '{"P":1}',
        ]
    )

    assert args.registered_interface_bonds_json == bond_path
    assert json.loads(args.expected_molecule_formula_json) == {"P": 1}
    assert load_registered_interface_bonds(bond_path) == (
        _registered_bond_record(),
    )


def test_registered_interface_cli_executes_exact_seam(tmp_path, capsys):
    from validate_monolayer import main

    structure = tmp_path / "cli-registered.extxyz"
    write(
        structure,
        Atoms(
            ["In", "P", "O"],
            positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 3.2], [0.0, 0.0, 1.75]],
            cell=np.diag([10.0, 10.0, 20.0]),
            pbc=True,
        ),
    )
    bonds = tmp_path / "cli-registered-bonds.json"
    bonds.write_text(json.dumps([_registered_bond_record()]))
    output = tmp_path / "validation.json"

    status = main(
        [
            str(structure),
            "--substrate-atoms",
            "1",
            "--molecules",
            "1",
            "--atoms-per-molecule",
            "2",
            "--protons-per-molecule",
            "0",
            "--surface-h-policy",
            "h0",
            "--registered-interface-bonds-json",
            str(bonds),
            "--expected-molecule-formula-json",
            '{"O":1,"P":1}',
            "--json-output",
            str(output),
        ]
    )

    assert status == 0
    report = json.loads(output.read_text())
    assert report["passed"] is True
    assert report["registered_interface_bonds"][0]["window_passed"] is True
    assert json.loads(capsys.readouterr().out)["passed"] is True


def test_validator_main_rejects_nonfinite_stdout_and_json_before_writing(
    tmp_path, monkeypatch, capsys
):
    import validate_monolayer as validator

    monkeypatch.setattr(
        validator,
        "validate",
        lambda *args, **kwargs: {"passed": True, "nonfinite": float("nan")},
    )
    output = tmp_path / "validation.json"

    with pytest.raises(ValueError, match="Out of range float values"):
        validator.main(
            [
                "unused.extxyz",
                "--substrate-atoms",
                "1",
                "--molecules",
                "1",
                "--atoms-per-molecule",
                "1",
                "--json-output",
                str(output),
            ]
        )

    assert capsys.readouterr().out == ""
    assert not output.exists()


def _required_cli_arguments():
    return [
        "candidate.extxyz",
        "--substrate-atoms",
        "1",
        "--molecules",
        "1",
        "--atoms-per-molecule",
        "1",
    ]


def test_validate_cli_exposes_auditable_collision_contract():
    from validate_monolayer import build_parser

    parser = build_parser()
    defaults = parser.parse_args(_required_cli_arguments())
    assert defaults.hh_min == pytest.approx(1.2)
    assert defaults.h_heavy_min == pytest.approx(1.4)
    assert defaults.heavy_heavy_min == pytest.approx(1.8)
    assert tuple(defaults.surface_periodic_axes) == (0, 1)

    custom = parser.parse_args(
        _required_cli_arguments()
        + [
            "--heavy-heavy-min",
            "2.2",
            "--surface-periodic-axes",
            "0",
            "2",
        ]
    )
    assert custom.heavy_heavy_min == pytest.approx(2.2)
    assert tuple(custom.surface_periodic_axes) == (0, 2)
    with pytest.raises(SystemExit):
        parser.parse_args(_required_cli_arguments() + ["--hh-min", "0"])
    with pytest.raises(SystemExit):
        parser.parse_args(
            _required_cli_arguments()
            + ["--surface-periodic-axes", "1", "1"]
        )


if __name__ == "__main__":
    test_allowed_pair_is_unordered_and_interface_only()
    test_surface_proton_location_classification()
    print("validate_monolayer interface-pair regression tests passed")
