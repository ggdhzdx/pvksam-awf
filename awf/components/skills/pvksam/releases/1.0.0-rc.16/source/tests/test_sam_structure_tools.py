from pathlib import Path
import sys

import numpy as np
import pytest
from ase.geometry import find_mic


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import sam_structure_tools as structure_tools
from sam_structure_tools import (
    canonicalize_pc_axis,
    cluster_1d_by_tolerance,
    filled_outer_envelope_area,
    filled_outer_envelope_metrics,
    periodic_polygon_coverage_mask,
    periodic_hole_diagnostics,
    periodic_physical_geometry_metrics,
    periodic_surface_polygon,
    periodic_uncovered_component_geometries,
    periodic_union_increment_A2,
    periodic_union_increment_perimeter,
    periodic_vdw_collision_audit,
    phosphonate_axial_height_intervals,
    pc_axial_rmsd_matrix,
    standardize_phosphonate_head_plane,
    surface_frame_coordinates,
)


def _phosphonate_geometry():
    return np.asarray([
        [0.0, 0.0, 1.0],  # P
        [-1.0, 0.0, 0.0], [0.5, 0.8, 0.0], [0.5, -0.8, 0.0],  # PO3
        [0.0, 0.0, 2.0], [1.0, 0.4, 1.7], [-0.3, 0.6, 2.2],  # C-side, incl. H
    ])


def _pair_distances(positions):
    return np.linalg.norm(positions[:, None] - positions[None, :], axis=2)


def _signed_tetrahedron_volume(positions):
    return np.linalg.det(positions[1:4] - positions[0]) / 6.0


def _rotate_test_fragment(positions, p_index, c_index, moving_indices, angle_deg):
    """Independent Rodrigues reference used to inspect geometry after rotation."""
    result = positions.copy()
    origin = result[p_index].copy()
    axis = result[c_index] - origin
    axis /= np.linalg.norm(axis)
    angle = np.deg2rad(angle_deg)
    for index in moving_indices:
        vector = positions[index] - origin
        result[index] = (
            origin + np.cos(angle) * vector
            + np.sin(angle) * np.cross(axis, vector)
            + (1.0 - np.cos(angle)) * np.dot(axis, vector) * axis
        )
    return result


def _tilted_height_geometry(constant, phases_deg, amplitude=1.0):
    """Atoms on a tilted axis with known h(theta)=constant+amplitude*cos(theta-phase)."""
    unit = np.sqrt(0.5)
    axis = np.asarray([unit, 0.0, unit])
    transverse_cosine = np.asarray([-unit, 0.0, unit])
    transverse_sine = np.asarray([0.0, -1.0, 0.0])
    p = np.asarray([0.0, 0.0, 1.0])
    points = [p, p + axis]
    for phase in np.deg2rad(phases_deg):
        points.append(
            p + ((constant - p[2]) / unit) * axis
            + (amplitude * np.cos(phase) / unit) * transverse_cosine
            - (amplitude * np.sin(phase) / unit) * transverse_sine
        )
    return np.asarray(points)


def _pc_shape_features(positions):
    aligned, _ = canonicalize_pc_axis(positions, 0, 4)
    return aligned[[4, 5, 6]]


def test_pc_axial_rmsd_removes_one_common_continuous_pc_phase():
    original = _phosphonate_geometry()
    rotated = _rotate_test_fragment(original, 0, 4, [4, 5, 6], 73.218)
    rmsd, angle = pc_axial_rmsd_matrix(
        _pc_shape_features(original), _pc_shape_features(rotated), return_angles=True
    )

    assert rmsd.shape == (1, 1)
    assert rmsd[0, 0] == pytest.approx(0.0, abs=1.0e-14)
    assert angle[0, 0] == pytest.approx(360.0 - 73.218, abs=1.0e-12)
    assert np.array_equal(rotated[:4], original[:4])


def test_pc_feature_frame_is_proper_and_global_rigid_motion_invariant():
    original = _phosphonate_geometry()
    proper_rotation = np.asarray([[0.36, -0.48, 0.8], [0.8, 0.6, 0.0], [-0.48, 0.64, 0.6]])
    transformed = original @ proper_rotation + [13.0, -8.0, 4.0]
    aligned, metadata = canonicalize_pc_axis(transformed, 0, 4)
    rotation = np.asarray(metadata["rotation_matrix"])

    assert aligned[0] == pytest.approx([0.0, 0.0, 0.0], abs=1.0e-14)
    assert aligned[4] == pytest.approx([0.0, 0.0, 1.0], abs=1.0e-14)
    assert aligned == pytest.approx(
        transformed @ rotation + metadata["translation_A"], abs=1.0e-14
    )
    assert rotation.T @ rotation == pytest.approx(np.eye(3), abs=1.0e-14)
    assert metadata["rotation_determinant"] == pytest.approx(1.0, abs=1.0e-14)
    assert _pair_distances(aligned) == pytest.approx(_pair_distances(transformed), abs=1.0e-14)
    assert pc_axial_rmsd_matrix(_pc_shape_features(original), aligned[[4, 5, 6]])[0, 0] == pytest.approx(
        0.0, abs=1.0e-13
    )
    assert np.array_equal(original, _phosphonate_geometry())
    assert metadata["physical_output_coordinates_replaced"] is False


@pytest.mark.parametrize("axis", [[0.0, 0.0, -1.0], [1.0e-12, -2.0e-12, -1.0], [1.0, 0.0, 0.0]])
def test_pc_axis_opposite_or_nearly_parallel_is_not_reflected(axis):
    positions = _phosphonate_geometry()
    positions[4] = positions[0] + axis
    aligned, metadata = canonicalize_pc_axis(positions, 0, 4)
    rotation = np.asarray(metadata["rotation_matrix"])

    assert aligned[4] == pytest.approx([0.0, 0.0, np.linalg.norm(axis)], abs=1.0e-14)
    assert rotation.T @ rotation == pytest.approx(np.eye(3), abs=1.0e-14)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1.0e-14)
    assert _signed_tetrahedron_volume(aligned) == pytest.approx(
        _signed_tetrahedron_volume(positions), abs=1.0e-14
    )


def test_pc_axial_rmsd_preserves_internal_bend_and_uses_3d_atom_norm():
    reference = np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 2.0]])
    bent = reference.copy()
    bent[1, 2] += 3.0

    assert pc_axial_rmsd_matrix(reference, bent)[0, 0] == pytest.approx(3.0 / np.sqrt(2.0))
    # A change in P-C bond length is not translated away by centering the body.
    stretched = reference.copy()
    stretched[0, 2] += 0.2
    assert pc_axial_rmsd_matrix(reference, stretched)[0, 0] == pytest.approx(0.2 / np.sqrt(2.0))


def test_pc_axial_rmsd_does_not_mirror_or_rotate_each_atom_separately():
    reference = np.asarray([[0.0, 0.0, 1.0], [1.0, 0.2, 1.5], [-0.4, 0.9, 2.3], [0.2, -0.7, 3.0]])
    mirrored = reference * [-1.0, 1.0, 1.0]
    different_phases = reference.copy()
    different_phases[1] = [-reference[1, 1], reference[1, 0], reference[1, 2]]

    assert pc_axial_rmsd_matrix(reference, mirrored)[0, 0] > 0.5
    assert pc_axial_rmsd_matrix(reference, different_phases)[0, 0] > 0.3
    assert pc_axial_rmsd_matrix(reference, reference[[0, 2, 1, 3]])[0, 0] > 0.5


def test_pc_axial_rmsd_batch_is_symmetric_and_matches_reference_subset():
    original = _phosphonate_geometry()
    phased = _rotate_test_fragment(original, 0, 4, [4, 5, 6], 31.7)
    bent = original.copy()
    bent[5, 2] += 0.17
    features = np.stack([_pc_shape_features(item) for item in (original, phased, bent)])
    matrix, angles = pc_axial_rmsd_matrix(features, return_angles=True)

    assert matrix.shape == (3, 3)
    assert matrix == pytest.approx(matrix.T, abs=1.0e-14)
    assert np.diag(matrix) == pytest.approx(np.zeros(3), abs=1.0e-14)
    assert matrix[0, 1] == pytest.approx(0.0, abs=1.0e-14)
    assert matrix[0, 2] > 0.0
    assert pc_axial_rmsd_matrix(features[[2]], features[[0, 1]]) == pytest.approx(matrix[[2], :2], abs=1.0e-14)
    assert np.all(angles >= 0.0) and np.all(angles < 360.0)


def test_pc_axial_rmsd_recovers_tiny_physical_difference_after_norm_cancellation():
    reference = np.asarray([[0.0, 0.0, 1.0], [11.0, 4.0, 7.0]])
    perturbed = reference.copy()
    perturbed[1, 2] += 1.0e-8

    assert pc_axial_rmsd_matrix(reference, perturbed)[0, 0] == pytest.approx(
        (perturbed[1, 2] - reference[1, 2]) / np.sqrt(2.0), rel=1.0e-12
    )
    assert pc_axial_rmsd_matrix(np.zeros((2, 3)), np.zeros((2, 3)), return_angles=True)[1][0, 0] == 0.0


def test_pc_axial_rmsd_matches_direct_common_rotation_for_general_3d_shapes():
    rng = np.random.default_rng(341270)
    references = rng.normal(size=(4, 7, 3))
    candidates = rng.normal(size=(5, 7, 3))
    references[:, 0] = [0.0, 0.0, 1.0]
    candidates[:, 0] = [0.0, 0.0, 1.2]
    matrix, angles = pc_axial_rmsd_matrix(references, candidates, return_angles=True)

    for row in range(len(references)):
        for column in range(len(candidates)):
            phase = np.deg2rad(angles[row, column])
            rotation = np.asarray([
                [np.cos(phase), -np.sin(phase), 0.0],
                [np.sin(phase), np.cos(phase), 0.0],
                [0.0, 0.0, 1.0],
            ])
            residual = references[row] - candidates[column] @ rotation.T
            direct_rmsd = np.sqrt(np.mean(np.sum(residual * residual, axis=1)))
            assert matrix[row, column] == pytest.approx(direct_rmsd, abs=1.0e-14)
            for angle_offset in [-0.013, 0.017]:
                trial = phase + angle_offset
                trial_rotation = np.asarray([
                    [np.cos(trial), -np.sin(trial), 0.0],
                    [np.sin(trial), np.cos(trial), 0.0],
                    [0.0, 0.0, 1.0],
                ])
                trial_residual = references[row] - candidates[column] @ trial_rotation.T
                assert np.sum(trial_residual * trial_residual) >= np.sum(residual * residual)


@pytest.mark.parametrize("invalid", [np.zeros((0, 3)), np.zeros((2, 2)), [[0.0, 0.0, np.nan]]])
def test_pc_feature_helpers_reject_invalid_coordinates(invalid):
    with pytest.raises(ValueError, match="finite|nonempty"):
        pc_axial_rmsd_matrix(invalid)
    with pytest.raises(ValueError, match="finite|nonempty"):
        canonicalize_pc_axis(invalid, 0, 1)


def test_pc_feature_helpers_reject_axis_and_mapping_errors():
    positions = _phosphonate_geometry()
    positions[4] = positions[0]
    with pytest.raises(ValueError, match="nonzero"):
        canonicalize_pc_axis(positions, 0, 4)
    with pytest.raises(ValueError, match="distinct"):
        canonicalize_pc_axis(positions, 0, 0)
    with pytest.raises(ValueError, match="integer"):
        canonicalize_pc_axis(positions, False, 4)
    with pytest.raises(ValueError, match="same atom mapping"):
        pc_axial_rmsd_matrix(np.zeros((2, 3)), np.zeros((3, 3)))
    with pytest.raises(ValueError, match="overflow"):
        pc_axial_rmsd_matrix(np.full((2, 3), 1.0e200))


def test_po3_standardization_is_one_distance_preserving_proper_transform():
    original = _phosphonate_geometry()
    standardized, metadata = standardize_phosphonate_head_plane(original, 0, [1, 2, 3])
    rotation = np.asarray(metadata["rotation_matrix"])

    assert standardized[[1, 2, 3], 2] == pytest.approx([0.0] * 3, abs=1.0e-14)
    assert standardized[[1, 2, 3]].mean(axis=0) == pytest.approx([0.0] * 3, abs=1.0e-14)
    assert standardized[0, 2] > 0.0
    assert rotation.T @ rotation == pytest.approx(np.eye(3), abs=1.0e-14)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1.0e-14)
    assert standardized == pytest.approx(
        original @ rotation + metadata["translation_A"], abs=1.0e-14
    )
    assert _pair_distances(standardized) == pytest.approx(_pair_distances(original), abs=1.0e-14)
    assert _signed_tetrahedron_volume(standardized) == pytest.approx(
        _signed_tetrahedron_volume(original), abs=1.0e-14
    )
    assert np.array_equal(original, _phosphonate_geometry())
    assert metadata["reflection_used"] is False


def test_po3_standardization_is_invariant_to_initial_rigid_orientation():
    original = _phosphonate_geometry()
    proper_rotation = np.asarray([[0.36, -0.48, 0.8], [0.8, 0.6, 0.0], [-0.48, 0.64, 0.6]])
    reference, _ = standardize_phosphonate_head_plane(original, 0, [1, 2, 3])
    transformed, _ = standardize_phosphonate_head_plane(
        original @ proper_rotation + [13.0, -8.0, 4.0], 0, [1, 2, 3]
    )

    assert transformed == pytest.approx(reference, abs=1.0e-13)


def test_po3_standardization_preserves_mirror_chirality_and_p_positive_side():
    original = _phosphonate_geometry()
    mirrored = original * [-1.0, 1.0, 1.0]
    standard, _ = standardize_phosphonate_head_plane(original, 0, [1, 2, 3])
    mirrored_standard, metadata = standardize_phosphonate_head_plane(mirrored, 0, [1, 2, 3])

    assert metadata["rotation_determinant"] == pytest.approx(1.0)
    assert mirrored_standard[0, 2] > 0.0
    assert _signed_tetrahedron_volume(mirrored_standard) == pytest.approx(
        _signed_tetrahedron_volume(mirrored)
    )
    assert _signed_tetrahedron_volume(mirrored_standard) == pytest.approx(
        -_signed_tetrahedron_volume(standard)
    )
    # Reordering the O labels may rotate the frame, but cannot reflect the SAM.
    reordered, metadata = standardize_phosphonate_head_plane(original, 0, [2, 1, 3])
    assert metadata["rotation_determinant"] == pytest.approx(1.0)
    assert _signed_tetrahedron_volume(reordered) == pytest.approx(
        _signed_tetrahedron_volume(original)
    )


def test_po3_standardization_rejects_collinear_oxygen_and_coplanar_p():
    collinear = _phosphonate_geometry()
    collinear[[1, 2, 3]] = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    with pytest.raises(ValueError, match="noncollinear"):
        standardize_phosphonate_head_plane(collinear, 0, [1, 2, 3])
    coplanar = _phosphonate_geometry()
    coplanar[0, 2] = 0.0
    with pytest.raises(ValueError, match="outside.*plane"):
        standardize_phosphonate_head_plane(coplanar, 0, [1, 2, 3])


def test_axial_height_vertical_axis_is_constant_and_includes_hydrogen():
    positions = _phosphonate_geometry()
    organic = [4, 5, 6]
    result = phosphonate_axial_height_intervals(positions, 0, 4, organic)

    assert result["feasible"] is True
    assert result["feasible_intervals_deg"] == [[0.0, 360.0]]
    assert result["representative_angle_deg"] == 0.0
    rotated = _rotate_test_fragment(positions, 0, 4, organic, 127.0)
    assert rotated[organic, 2] == pytest.approx(positions[organic, 2], abs=1.0e-14)
    assert np.array_equal(rotated[:4], positions[:4])
    assert _pair_distances(rotated[organic]) == pytest.approx(
        _pair_distances(positions[organic]), abs=1.0e-14
    )
    positions[6, 2] = -1.0e-5  # An organic H below the plane is sufficient to reject.
    rejected = phosphonate_axial_height_intervals(positions, 0, 4, organic)
    assert rejected["feasible"] is False
    assert rejected["individually_impossible_atom_indices"] == [6]
    assert rejected["representative_angle_deg"] is None
    assert phosphonate_axial_height_intervals(positions, 0, 4, [4, 5])["feasible"] is True


def test_axial_height_finds_narrow_tilted_axis_arc_missed_by_30_degree_grid():
    positions = _tilted_height_geometry(-np.cos(np.deg2rad(3.0)), [15.0])
    result = phosphonate_axial_height_intervals(positions, 0, 1, [1, 2], tolerance=0.0)

    assert result["feasible"] is True
    assert result["angular_grid_used"] is False
    assert np.asarray(result["feasible_intervals_deg"]) == pytest.approx(
        np.asarray([[12.0, 18.0]]), abs=1.0e-10
    )
    assert result["representative_angle_deg"] == pytest.approx(15.0)
    assert result["feasible_angle_measure_deg"] == pytest.approx(6.0, abs=1.0e-10)
    assert all(
        _rotate_test_fragment(positions, 0, 1, [1, 2], angle)[2, 2] < 0.0
        for angle in range(0, 360, 30)
    )
    for angle in [12.0, 15.0, 18.0]:
        rotated = _rotate_test_fragment(positions, 0, 1, [1, 2], angle)
        assert np.min(rotated[[1, 2], 2]) >= -1.0e-14


def test_axial_height_intersects_atoms_instead_of_separate_maximum_heights():
    positions = _tilted_height_geometry(-0.5, [0.0, 180.0])
    result = phosphonate_axial_height_intervals(positions, 0, 1, [1, 2, 3], tolerance=0.0)

    assert all(record["maximum_height_A"] > 0.0 for record in result["atom_height_coefficients"])
    assert all(record["feasible_intervals_deg"] for record in result["atom_height_coefficients"])
    assert result["individually_impossible_atom_indices"] == []
    assert result["feasible"] is False
    assert result["feasible_intervals_deg"] == []


def test_axial_height_wraparound_uses_circular_midpoint_and_keeps_tangent_point():
    positions = _tilted_height_geometry(-0.5, [0.0])
    result = phosphonate_axial_height_intervals(positions, 0, 1, [1, 2], tolerance=0.0)
    assert np.asarray(result["feasible_intervals_deg"]) == pytest.approx(
        np.asarray([[0.0, 60.0], [300.0, 360.0]]), abs=1.0e-12
    )
    assert result["representative_angle_deg"] == pytest.approx(0.0)
    assert result["feasible_angle_measure_deg"] == pytest.approx(120.0)

    tangent = _tilted_height_geometry(-1.0, [0.0])
    result = phosphonate_axial_height_intervals(tangent, 0, 1, [1, 2], tolerance=0.0)
    assert result["feasible"] is True
    assert np.asarray(result["feasible_intervals_deg"]) == pytest.approx(
        np.asarray([[0.0, 0.0]]), abs=1.0e-12
    )
    assert result["representative_minimum_height_A"] == pytest.approx(0.0, abs=1.0e-14)


def test_axial_height_and_head_collision_use_the_same_feasible_angle():
    positions = np.asarray([
        [0.0, 0.0, 1.0],  # P
        [1.0, 0.0, 0.0], [-0.5, 0.8, 0.0], [-0.5, -0.8, 0.0],  # PO3
        [0.0, 0.0, 2.0], [1.0, 0.0, 1.0],  # C-side
    ])
    height_only = phosphonate_axial_height_intervals(positions, 0, 4, [4, 5], tolerance=0.0)
    joint = phosphonate_axial_height_intervals(
        positions, 0, 4, [4, 5], tolerance=0.0,
        collision_pairs_0based=[[1, 5]], collision_min_distances_A=[1.4],
    )
    half_exclusion = np.rad2deg(np.arccos((3.0 - 1.4**2) / 2.0))

    assert height_only["representative_angle_deg"] == 0.0
    assert np.linalg.norm(positions[1] - positions[5]) < 1.4
    assert joint["height_only_feasible"] is True
    assert joint["collision_constraints_count"] == 1
    assert joint["feasible"] is True
    assert np.asarray(joint["feasible_intervals_deg"]) == pytest.approx(
        np.asarray([[half_exclusion, 360.0 - half_exclusion]]), abs=1.0e-12
    )
    assert joint["representative_angle_deg"] == pytest.approx(180.0)
    rotated = _rotate_test_fragment(positions, 0, 4, [4, 5], joint["representative_angle_deg"])
    assert np.array_equal(rotated[:4], positions[:4])
    assert np.min(rotated[[4, 5], 2]) >= 0.0
    assert np.linalg.norm(rotated[1] - rotated[5]) >= 1.4
    # Pair order is irrelevant to the same fixed/moving distance constraint.
    reverse = phosphonate_axial_height_intervals(
        positions, 0, 4, [4, 5], tolerance=0.0,
        collision_pairs_0based=[[5, 1]], collision_min_distances_A=[1.4],
    )
    assert reverse["feasible_intervals_deg"] == joint["feasible_intervals_deg"]


def test_axial_height_and_collision_can_be_separately_feasible_but_jointly_empty():
    positions = _tilted_height_geometry(-np.cos(np.deg2rad(3.0)), [15.0])
    at_peak = _rotate_test_fragment(positions, 0, 1, [1, 2], 15.0)[2]
    fixed = at_peak.copy()
    fixed[2] = 0.0
    positions = np.vstack([positions, fixed])
    joint = phosphonate_axial_height_intervals(
        positions, 0, 1, [1, 2], tolerance=0.0,
        collision_pairs_0based=[[2, 3]], collision_min_distances_A=[0.2],
    )

    assert joint["height_only_feasible"] is True
    assert joint["collision_squared_distance_coefficients"][0]["feasible_intervals_deg"]
    assert joint["feasible"] is False
    assert joint["representative_angle_deg"] is None


def test_axial_joint_intervals_match_direct_tilted_rotation_with_sine_distance_term():
    positions = _tilted_height_geometry(0.4, [35.0, 115.0], amplitude=0.8)
    positions = np.vstack([positions, [0.8, 0.6, 0.0], [-0.7, -0.4, 0.0]])
    organic = [1, 2, 3]
    pairs = [[4, 2], [3, 5]]
    minimum_distances = [1.3, 1.0]
    result = phosphonate_axial_height_intervals(
        positions, 0, 1, organic, tolerance=0.0,
        collision_pairs_0based=pairs, collision_min_distances_A=minimum_distances,
    )

    assert all(
        abs(record["sine_squared_distance_A2"]) > 0.1
        for record in result["collision_squared_distance_coefficients"]
    )
    samples = []
    for angle in np.arange(0.25, 360.0, 7.5):
        rotated = _rotate_test_fragment(positions, 0, 1, organic, angle)
        physically_feasible = bool(
            np.min(rotated[organic, 2]) >= 0.0
            and all(
                np.linalg.norm(rotated[left] - rotated[right]) >= threshold
                for (left, right), threshold in zip(pairs, minimum_distances)
            )
        )
        inside_analytic_intervals = any(
            start <= angle <= end for start, end in result["feasible_intervals_deg"]
        )
        assert inside_analytic_intervals is physically_feasible
        samples.append(physically_feasible)
    assert any(samples) and not all(samples)


def test_axial_height_rejects_incomplete_indices_and_invariant_collision_pairs():
    positions = _phosphonate_geometry()
    with pytest.raises(ValueError, match="include C"):
        phosphonate_axial_height_intervals(positions, 0, 4, [5, 6])
    with pytest.raises(ValueError, match="moving and one fixed"):
        phosphonate_axial_height_intervals(
            positions, 0, 4, [4, 5, 6],
            collision_pairs_0based=[[4, 5]], collision_min_distances_A=[1.0],
        )
    with pytest.raises(ValueError, match="provided together"):
        phosphonate_axial_height_intervals(
            positions, 0, 4, [4, 5, 6], collision_pairs_0based=[[1, 5]],
        )


def identity_frame():
    return {
        "u_cartesian_unit": [1.0, 0.0, 0.0],
        "v_cartesian_unit": [0.0, 1.0, 0.0],
        "outward_normal_cartesian_unit": [0.0, 0.0, 1.0],
    }


def envelope_for(positions):
    return filled_outer_envelope_area(
        positions=positions,
        symbols=["H"],
        surface_frame=identity_frame(),
        radii_A={"H": 1.0},
        boundary_samples_per_atom=72,
    )


def test_filled_outer_envelope_metrics_are_finite_and_translation_invariant():
    positions = np.asarray(
        [[0.0, 0.0, 2.0], [4.0, 0.0, 2.0], [4.0, 1.0, 2.0], [0.0, 1.0, 2.0]]
    )
    first = filled_outer_envelope_metrics(
        positions=positions,
        symbols=["H"] * len(positions),
        surface_frame=identity_frame(),
        radii_A={"H": 0.2},
        boundary_samples_per_atom=72,
    )
    translated = filled_outer_envelope_metrics(
        positions=positions + [11.0, -7.0, 3.0],
        symbols=["H"] * len(positions),
        surface_frame=identity_frame(),
        radii_A={"H": 0.2},
        boundary_samples_per_atom=72,
    )

    for key in (
        "area_A2",
        "perimeter_A",
        "compactness",
        "anisotropy",
        "aspect_ratio",
        "principal_orientation_rad_mod_pi",
    ):
        assert np.isfinite(first[key])
        assert translated[key] == pytest.approx(first[key], abs=1.0e-10)
    assert first["principal_orientation_cos2"] == pytest.approx(
        translated["principal_orientation_cos2"], abs=1.0e-10
    )
    assert first["principal_orientation_sin2"] == pytest.approx(
        translated["principal_orientation_sin2"], abs=1.0e-10
    )


def test_surface_frame_coordinates_use_registered_local_axes():
    frame = {
        "u_cartesian_unit": [0.0, 1.0, 0.0],
        "v_cartesian_unit": [-1.0, 0.0, 0.0],
        "outward_normal_cartesian_unit": [0.0, 0.0, 1.0],
    }

    local = surface_frame_coordinates([[2.0, 3.0, 4.0]], frame)

    assert np.allclose(local, [[3.0, -2.0, 4.0]])
    with pytest.raises(ValueError, match="orthonormal right-handed"):
        surface_frame_coordinates(
            [[2.0, 3.0, 4.0]],
            {**frame, "v_cartesian_unit": [1.0, 0.0, 0.0]},
        )


def test_periodic_vdw_gate_reports_surface_frame_overlap_and_image():
    frame = {
        "u_cartesian_unit": [1.0, 0.0, 0.0],
        "v_cartesian_unit": [0.0, 1.0, 0.0],
        "outward_normal_cartesian_unit": [0.0, 0.0, 1.0],
    }

    audit = periodic_vdw_collision_audit(
        molecule_positions=[[9.8, 0.0, 1.0]],
        molecule_symbols=["H"],
        molecule_atom_ids=[7],
        substrate_positions=[[0.2, 0.0, 0.0]],
        substrate_symbols=["O"],
        substrate_atom_ids=[11],
        cell=np.diag([10.0, 10.0, 20.0]),
        periodic_axes=(0, 1),
        surface_frame=frame,
        radii_A={"H": 1.2, "O": 1.52},
        radius_scale=0.5,
    )

    assert audit["passed"] is False
    assert audit["collision_count"] == 1
    collision = audit["collisions"][0]
    assert collision["reason"] == "vdw_overlap"
    assert collision["molecule_atom_id"] == 7
    assert collision["substrate_atom_id"] == 11
    assert collision["substrate_periodic_image"] == [1, 0, 0]
    assert collision["lateral_separation_A"] == pytest.approx(0.4)
    assert collision["normal_separation_A"] == pytest.approx(1.0)


def test_periodic_vdw_gate_exempts_only_registered_mapped_bond():
    frame = {
        "u_cartesian_unit": [1.0, 0.0, 0.0],
        "v_cartesian_unit": [0.0, 1.0, 0.0],
        "outward_normal_cartesian_unit": [0.0, 0.0, 1.0],
    }

    audit = periodic_vdw_collision_audit(
        molecule_positions=[[9.8, 0.0, 1.0]],
        molecule_symbols=["O"],
        molecule_atom_ids=[7],
        substrate_positions=[[0.2, 0.0, 0.0]],
        substrate_symbols=["In"],
        substrate_atom_ids=[11],
        cell=np.diag([10.0, 10.0, 20.0]),
        periodic_axes=(0, 1),
        surface_frame=frame,
        radii_A={"O": 1.52, "In": 1.93},
        radius_scale=0.85,
        mapped_bond_windows_A={(7, 11): (1.0, 1.2)},
    )

    assert audit["passed"] is True
    assert audit["collision_count"] == 0
    assert audit["mapped_bond_count"] == 1
    assert audit["mapped_bond_violation_count"] == 0
    assert audit["mapped_bonds"][0]["distance_A"] == pytest.approx(
        np.sqrt(0.4**2 + 1.0**2)
    )


def test_periodic_vdw_gate_uses_closed_bond_window_at_machine_precision():
    def audit_at_x(x):
        return periodic_vdw_collision_audit(
            molecule_positions=[[x, 0.0, 0.0]],
            molecule_symbols=["O"],
            molecule_atom_ids=[7],
            substrate_positions=[[0.0, 0.0, 0.0]],
            substrate_symbols=["In"],
            substrate_atom_ids=[11],
            cell=np.diag([10.0, 10.0, 20.0]),
            periodic_axes=(0, 1),
            surface_frame=identity_frame(),
            radii_A={"O": 1.52, "In": 1.93},
            radius_scale=0.85,
            mapped_bond_windows_A={(7, 11): (1.7, 2.8)},
        )

    exact_lower = audit_at_x(8.3)
    below_lower = audit_at_x(1.7 - 1.0e-12)
    exact_upper = audit_at_x(2.8)
    above_upper = audit_at_x(2.8 + 1.0e-12)

    assert exact_lower["mapped_bonds"][0]["distance_A"] < 1.7
    assert exact_lower["passed"] is True
    assert exact_upper["passed"] is True
    assert below_lower["passed"] is False
    assert above_upper["passed"] is False


def test_periodic_vdw_gate_finds_review_counterexample_beyond_local_3x3_images():
    cell = np.asarray(
        [
            [2.199, 0.0, 0.0],
            [-21.055, 6.430, 0.0],
            [0.0, 0.0, 20.0],
        ]
    )
    displacement = np.asarray([-0.50, -0.45, 0.0]) @ cell

    audit = periodic_vdw_collision_audit(
        molecule_positions=[displacement],
        molecule_symbols=["H"],
        molecule_atom_ids=[7],
        substrate_positions=[[0.0, 0.0, 0.0]],
        substrate_symbols=["O"],
        substrate_atom_ids=[11],
        cell=cell,
        periodic_axes=(0, 1),
        surface_frame=identity_frame(),
        radii_A={"H": 1.6, "O": 1.6},
        radius_scale=1.0,
    )

    expected_vector, expected_distance = find_mic(
        displacement, cell=cell, pbc=[True, True, False]
    )
    assert expected_distance < 3.2
    assert audit["collision_count"] == 1
    collision = audit["collisions"][0]
    image = np.asarray(collision["substrate_periodic_image"], dtype=int)
    reconstructed = displacement - image @ cell
    assert np.max(np.abs(image[:2])) > 1
    assert reconstructed == pytest.approx(expected_vector)
    assert collision["distance_A"] == pytest.approx(expected_distance)
    assert collision["lateral_separation_A"] == pytest.approx(
        np.linalg.norm(expected_vector[:2])
    )
    assert collision["normal_separation_A"] == pytest.approx(0.0)


def test_periodic_vdw_gate_matches_independent_find_mic_for_skew_partial_pbc_property():
    rng = np.random.default_rng(20260809)
    frame = {
        "u_cartesian_unit": [1.0, 0.0, 0.0],
        "v_cartesian_unit": [0.0, 0.0, 1.0],
        "outward_normal_cartesian_unit": [0.0, -1.0, 0.0],
    }
    for case in range(32):
        short = rng.uniform(1.8, 3.2)
        cell = np.asarray(
            [
                [short, 0.0, 0.0],
                [0.0, 30.0, 0.0],
                [-rng.uniform(7.0, 12.0) * short, 0.0, rng.uniform(4.0, 8.0)],
            ]
        )
        fractional = rng.uniform(-0.5, 0.5, 3)
        displacement = fractional @ cell
        displacement[1] = rng.uniform(-4.0, 4.0)
        expected_vector, expected_distance = find_mic(
            displacement, cell=cell, pbc=[True, False, True]
        )

        audit = periodic_vdw_collision_audit(
            molecule_positions=[displacement],
            molecule_symbols=["O"],
            molecule_atom_ids=[100 + case],
            substrate_positions=[[0.0, 0.0, 0.0]],
            substrate_symbols=["In"],
            substrate_atom_ids=[500 + case],
            cell=cell,
            periodic_axes=(0, 2),
            surface_frame=frame,
            radii_A={"O": 0.5, "In": 0.5},
            radius_scale=0.1,
            mapped_bond_windows_A={(100 + case, 500 + case): (1.0e-6, 100.0)},
        )

        mapped = audit["mapped_bonds"][0]
        image = np.asarray(mapped["substrate_periodic_image"], dtype=int)
        reconstructed = displacement - image @ cell
        expected_local = surface_frame_coordinates(expected_vector, frame)[0]
        assert image[1] == 0
        assert reconstructed == pytest.approx(expected_vector, abs=1.0e-9)
        assert mapped["distance_A"] == pytest.approx(expected_distance, abs=1.0e-9)
        assert mapped["lateral_separation_A"] == pytest.approx(
            np.linalg.norm(expected_local[:2]), abs=1.0e-9
        )
        assert mapped["signed_normal_separation_A"] == pytest.approx(
            expected_local[2], abs=1.0e-9
        )


def test_periodic_vdw_gate_requires_unique_atom_id_arrays_and_existing_mapped_ids():
    kwargs = {
        "molecule_positions": [[0.0, 0.0, 2.0], [4.0, 0.0, 2.0]],
        "molecule_symbols": ["O", "O"],
        "molecule_atom_ids": [7, 8],
        "substrate_positions": [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        "substrate_symbols": ["In", "In"],
        "substrate_atom_ids": [11, 12],
        "cell": np.diag([10.0, 10.0, 20.0]),
        "periodic_axes": (0, 1),
        "surface_frame": identity_frame(),
        "radii_A": {"O": 1.52, "In": 1.93},
        "radius_scale": 0.85,
    }
    with pytest.raises(ValueError, match="Molecule atom IDs must be unique"):
        periodic_vdw_collision_audit(
            **{**kwargs, "molecule_atom_ids": [7, 7]}
        )
    with pytest.raises(ValueError, match="Substrate atom IDs must be unique"):
        periodic_vdw_collision_audit(
            **{**kwargs, "substrate_atom_ids": [11, 11]}
        )
    with pytest.raises(ValueError, match="absent"):
        periodic_vdw_collision_audit(
            **kwargs, mapped_bond_windows_A={(7, 99): (1.7, 2.8)}
        )


def test_mapped_pair_outside_window_remains_a_vdw_collision_and_violation():
    audit = periodic_vdw_collision_audit(
        molecule_positions=[[0.0, 0.0, 1.5]],
        molecule_symbols=["O"],
        molecule_atom_ids=[7],
        substrate_positions=[[0.0, 0.0, 0.0]],
        substrate_symbols=["In"],
        substrate_atom_ids=[11],
        cell=np.diag([10.0, 10.0, 20.0]),
        periodic_axes=(0, 1),
        surface_frame=identity_frame(),
        radii_A={"O": 1.52, "In": 1.93},
        radius_scale=0.85,
        mapped_bond_windows_A={(7, 11): (1.7, 2.8)},
    )

    assert audit["passed"] is False
    assert audit["mapped_bond_violation_count"] == 1
    assert audit["collision_count"] == 1
    assert audit["mapped_bonds"][0]["passed"] is False
    assert audit["collisions"][0]["molecule_atom_id"] == 7
    assert audit["collisions"][0]["substrate_atom_id"] == 11


def test_periodic_vdw_gate_uses_one_vectorized_neighbor_query_for_many_cross_pairs(
    monkeypatch,
):
    calls = []
    original = structure_tools.primitive_neighbor_list

    def counted(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(structure_tools, "primitive_neighbor_list", counted)
    molecule_count = 48
    substrate_count = 128
    molecule_positions = np.column_stack(
        (
            np.linspace(0.0, 47.0, molecule_count),
            np.zeros(molecule_count),
            np.full(molecule_count, 10.0),
        )
    )
    substrate_positions = np.column_stack(
        (
            np.linspace(0.0, 47.0, substrate_count),
            np.full(substrate_count, 40.0),
            np.zeros(substrate_count),
        )
    )

    audit = periodic_vdw_collision_audit(
        molecule_positions=molecule_positions,
        molecule_symbols=["H"] * molecule_count,
        molecule_atom_ids=list(range(molecule_count)),
        substrate_positions=substrate_positions,
        substrate_symbols=["O"] * substrate_count,
        substrate_atom_ids=list(range(1000, 1000 + substrate_count)),
        cell=np.diag([100.0, 100.0, 50.0]),
        periodic_axes=(0, 1),
        surface_frame=identity_frame(),
        radii_A={"H": 1.2, "O": 1.52},
        radius_scale=0.85,
    )

    assert len(calls) == 1
    search = audit["neighbor_search"]
    assert search["primitive_neighbor_list_query_count"] == 1
    assert search["cross_pair_count"] == molecule_count * substrate_count
    assert search["full_cartesian_pair_matrix_materialized"] is False
    assert search["python_cross_pair_loop"] is False


def test_filled_outer_envelope_counts_the_gap_between_projected_disks():
    frame = {
        "u_cartesian_unit": [1.0, 0.0, 0.0],
        "v_cartesian_unit": [0.0, 1.0, 0.0],
        "outward_normal_cartesian_unit": [0.0, 0.0, 1.0],
    }

    footprint = filled_outer_envelope_area(
        positions=[[0.0, 0.0, 0.0], [2.0, 0.0, 3.0]],
        symbols=["H", "H"],
        surface_frame=frame,
        radii_A={"H": 1.0},
        radius_scale=1.0,
        boundary_samples_per_atom=720,
    )

    assert footprint["area_A2"] == pytest.approx(np.pi + 4.0, abs=2.0e-4)
    assert footprint["definition"] == "convex_hull_of_projected_vdw_disks"


def test_filled_outer_envelope_returns_vertices_and_bounds():
    footprint = filled_outer_envelope_area(
        positions=[[0, 0, 0], [2, 0, 3]],
        symbols=["H", "H"],
        surface_frame=identity_frame(),
        radii_A={"H": 1.0},
        boundary_samples_per_atom=72,
    )
    vertices = np.asarray(footprint["hull_vertices_uv_A"])
    assert vertices.ndim == 2 and vertices.shape[1] == 2
    assert np.allclose(
        footprint["bounds_uv_A"],
        [
            [vertices[:, 0].min(), vertices[:, 1].min()],
            [vertices[:, 0].max(), vertices[:, 1].max()],
        ],
    )


def test_envelope_vertices_translate_in_surface_frame():
    first = envelope_for([[0, 0, 0]])
    moved = envelope_for([[3, -2, 7]])
    assert np.allclose(
        np.asarray(moved["hull_vertices_uv_A"]),
        np.asarray(first["hull_vertices_uv_A"]) + [3.0, -2.0],
    )


def _surface_lattice_for_test(cell, frame):
    a_uv = surface_frame_coordinates(cell[0][None, :], frame)[0, :2]
    b_uv = surface_frame_coordinates(cell[1][None, :], frame)[0, :2]
    return np.column_stack([a_uv, b_uv])


def _fractional_rect_to_uv(frac_rect, lattice):
    xmin, ymin, xmax, ymax = frac_rect
    corners_frac = np.asarray(
        [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=float
    )
    return corners_frac @ lattice.T


def _diagonal_cell_lattice_frame():
    cell = np.diag([10.0, 10.0, 20.0])
    frame = identity_frame()
    return cell, _surface_lattice_for_test(cell, frame), frame


def test_periodic_surface_polygon_preserves_area_across_boundaries():
    cell, lattice, frame = _diagonal_cell_lattice_frame()
    for frac_rect in (
        (0.3, 0.3, 0.5, 0.7),  # 内部
        (-0.2, 0.3, 0.2, 0.5),  # U 边界
        (0.3, -0.2, 0.5, 0.2),  # V 边界
        (-0.1, -0.2, 0.1, 0.2),  # 角穿越（跨 U 与 V 两条边界）
    ):
        vertices = _fractional_rect_to_uv(frac_rect, lattice)
        geometry, area_scale_A2 = periodic_surface_polygon(
            vertices, cell=cell, periodic_axes=(0, 1), surface_frame=frame,
        )
        assert area_scale_A2 == pytest.approx(100.0)
        assert geometry.area * area_scale_A2 == pytest.approx(8.0, abs=1.0e-9)


def test_periodic_surface_polygon_handles_skew_cell_and_rotated_frame():
    cell = np.array([[10.0, 3.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 20.0]])
    frame = identity_frame()
    lattice = _surface_lattice_for_test(cell, frame)
    assert np.linalg.det(lattice) == pytest.approx(100.0)
    vertices = _fractional_rect_to_uv((0.2, 0.2, 0.4, 0.4), lattice)
    geometry, area_scale_A2 = periodic_surface_polygon(
        vertices, cell=cell, periodic_axes=(0, 1), surface_frame=frame,
    )
    assert area_scale_A2 == pytest.approx(100.0)
    assert geometry.area * area_scale_A2 == pytest.approx(4.0, abs=1.0e-9)

    rotated = {
        "u_cartesian_unit": [0.0, 1.0, 0.0],
        "v_cartesian_unit": [-1.0, 0.0, 0.0],
        "outward_normal_cartesian_unit": [0.0, 0.0, 1.0],
    }
    rotated_lattice = _surface_lattice_for_test(cell, rotated)
    assert np.linalg.det(rotated_lattice) == pytest.approx(100.0)
    rotated_vertices = _fractional_rect_to_uv((0.2, 0.2, 0.4, 0.4), rotated_lattice)
    rotated_geometry, rotated_scale = periodic_surface_polygon(
        rotated_vertices, cell=cell, periodic_axes=(0, 1), surface_frame=rotated,
    )
    assert rotated_scale == pytest.approx(100.0)
    assert rotated_geometry.area * rotated_scale == pytest.approx(4.0, abs=1.0e-9)


def test_periodic_union_increment_reports_overlap_and_disjoint():
    cell, lattice, frame = _diagonal_cell_lattice_frame()
    rect_a = _fractional_rect_to_uv((0.0, 0.0, 0.2, 0.2), lattice)
    rect_overlap = _fractional_rect_to_uv((0.1, 0.0, 0.3, 0.2), lattice)
    rect_disjoint = _fractional_rect_to_uv((0.5, 0.5, 0.7, 0.7), lattice)
    geom_a, scale = periodic_surface_polygon(
        rect_a, cell=cell, periodic_axes=(0, 1), surface_frame=frame,
    )
    geom_overlap, _ = periodic_surface_polygon(
        rect_overlap, cell=cell, periodic_axes=(0, 1), surface_frame=frame,
    )
    geom_disjoint, _ = periodic_surface_polygon(
        rect_disjoint, cell=cell, periodic_axes=(0, 1), surface_frame=frame,
    )

    increment, _ = periodic_union_increment_A2(geom_a, geom_overlap, scale)
    assert increment == pytest.approx(2.0, abs=1.0e-9)
    increment, _ = periodic_union_increment_A2(geom_a, geom_disjoint, scale)
    assert increment == pytest.approx(4.0, abs=1.0e-9)


def test_periodic_polygon_coverage_mask_wraps_anchors_into_unit_square():
    cell, lattice, frame = _diagonal_cell_lattice_frame()
    vertices = _fractional_rect_to_uv((0.1, 0.1, 0.3, 0.3), lattice)
    geometry, _ = periodic_surface_polygon(
        vertices, cell=cell, periodic_axes=(0, 1), surface_frame=frame,
    )
    anchors = np.array(
        [
            [0.2, 0.2],  # 内部 → 覆盖
            [0.5, 0.5],  # 外部 → 不覆盖
            [0.1, 0.1],  # 边界 → covers 判定覆盖
            [1.2, 0.2],  # 周期图像 → 覆盖
            [-0.8, 0.2],  # 周期图像 → 覆盖
        ]
    )
    mask = periodic_polygon_coverage_mask(geometry, anchors)
    assert mask == 0b11101


def _periodic_union_for_fractional_rectangles(rectangles, *, shift=(0.0, 0.0)):
    cell, lattice, frame = _diagonal_cell_lattice_frame()
    union = None
    for rectangle in rectangles:
        xmin, ymin, xmax, ymax = rectangle
        du, dv = shift
        vertices = _fractional_rect_to_uv(
            (xmin + du, ymin + dv, xmax + du, ymax + dv), lattice
        )
        geometry, scale = periodic_surface_polygon(
            vertices, cell=cell, periodic_axes=(0, 1), surface_frame=frame,
        )
        _, union = periodic_union_increment_A2(union, geometry, scale)
    return union, scale


def test_periodic_hole_diagnostics_merges_only_positive_length_u_seams():
    occupied, scale = _periodic_union_for_fractional_rectangles(
        [(0.2, 0.0, 0.3, 1.0), (0.7, 0.0, 0.8, 1.0)]
    )

    diagnostics = periodic_hole_diagnostics(occupied, scale)

    # The planar cell cut produces three fragments.  On the torus, only the
    # first and last intervals share a positive-length U seam, so there are two.
    assert diagnostics["hole_count"] == 2
    assert diagnostics["hole_areas_A2"] == pytest.approx([40.0, 40.0])
    assert diagnostics["total_hole_area_A2"] == pytest.approx(80.0)
    assert diagnostics["maximum_hole_area_A2"] == pytest.approx(40.0)
    assert diagnostics["seam_merge_diagnostics"]["planar_fragment_count"] == 3
    assert diagnostics["seam_merge_diagnostics"]["accepted_merge_count"] == 1


def test_periodic_hole_diagnostics_handles_v_and_corner_wrapping():
    horizontal_bands, scale = _periodic_union_for_fractional_rectangles(
        [(0.0, 0.2, 1.0, 0.3), (0.0, 0.7, 1.0, 0.8)]
    )
    v_diagnostics = periodic_hole_diagnostics(horizontal_bands, scale)
    assert v_diagnostics["hole_count"] == 2
    assert v_diagnostics["hole_areas_A2"] == pytest.approx([40.0, 40.0])

    from shapely.geometry import box
    from shapely.ops import unary_union

    uncovered_corner_fragments = unary_union(
        [
            box(0.0, 0.0, 0.1, 0.2),
            box(0.9, 0.0, 1.0, 0.2),
            box(0.0, 0.8, 0.1, 1.0),
            box(0.9, 0.8, 1.0, 1.0),
        ]
    )
    occupied = box(0.0, 0.0, 1.0, 1.0).difference(
        uncovered_corner_fragments
    )
    corner_diagnostics = periodic_hole_diagnostics(occupied, scale)
    assert corner_diagnostics["hole_count"] == 1
    assert corner_diagnostics["hole_areas_A2"] == pytest.approx([8.0])
    assert (
        corner_diagnostics["seam_merge_diagnostics"]["planar_fragment_count"]
        == 4
    )


def test_periodic_hole_diagnostics_does_not_merge_disjoint_or_point_seams():
    from shapely.geometry import box
    from shapely.ops import unary_union

    cell_box = box(0.0, 0.0, 1.0, 1.0)
    disjoint = unary_union(
        [box(0.0, 0.1, 0.1, 0.3), box(0.9, 0.6, 1.0, 0.8)]
    )
    disjoint_result = periodic_hole_diagnostics(
        cell_box.difference(disjoint), 100.0
    )
    assert disjoint_result["hole_count"] == 2
    assert disjoint_result["seam_merge_diagnostics"]["accepted_merge_count"] == 0

    point_touching = unary_union(
        [box(0.0, 0.1, 0.1, 0.3), box(0.9, 0.3, 1.0, 0.5)]
    )
    point_result = periodic_hole_diagnostics(
        cell_box.difference(point_touching), 100.0
    )
    assert point_result["hole_count"] == 2
    seam_audit = point_result["seam_merge_diagnostics"]
    assert seam_audit["point_contact_semantics"] == "does_not_merge"
    assert seam_audit["point_contact_count"] >= 1
    assert seam_audit["buffer_used"] is False


def test_periodic_hole_diagnostics_is_invariant_to_common_fractional_translation():
    from shapely import affinity

    rectangles = [(0.2, 0.0, 0.3, 1.0), (0.7, 0.0, 0.8, 1.0)]
    original, scale = _periodic_union_for_fractional_rectangles(rectangles)
    common_shift = (0.173, 0.287)
    translated = affinity.translate(
        original, xoff=common_shift[0], yoff=common_shift[1]
    )

    original_result = periodic_hole_diagnostics(original, scale)
    translated_result = periodic_hole_diagnostics(
        translated, scale, cell_origin_fractional=common_shift
    )

    assert translated_result["hole_count"] == original_result["hole_count"]
    assert translated_result["hole_areas_A2"] == pytest.approx(
        original_result["hole_areas_A2"], abs=1.0e-9
    )
    assert translated_result["total_hole_area_A2"] == pytest.approx(
        original_result["total_hole_area_A2"], abs=1.0e-9
    )


def test_periodic_hole_diagnostics_empty_full_and_cell_contained_regressions():
    from shapely.geometry import box

    empty_coverage = periodic_hole_diagnostics(None, 100.0)
    assert empty_coverage["hole_count"] == 1
    assert empty_coverage["hole_areas_A2"] == pytest.approx([100.0])
    assert empty_coverage["cell_area_A2"] == pytest.approx(100.0)
    assert empty_coverage["uncovered_area_fraction"] == pytest.approx(1.0)

    full_coverage = periodic_hole_diagnostics(box(0.0, 0.0, 1.0, 1.0), 100.0)
    assert full_coverage["hole_count"] == 0
    assert full_coverage["hole_areas_A2"] == []
    assert full_coverage["total_hole_area_A2"] == pytest.approx(0.0)

    interior_island = periodic_hole_diagnostics(
        box(0.4, 0.4, 0.6, 0.6), 100.0
    )
    assert interior_island["hole_count"] == 1
    assert interior_island["hole_areas_A2"] == pytest.approx([96.0])
    assert interior_island["metric_version"] == "periodic-torus-holes-v2"
    conservation = interior_island["area_conservation"]
    assert conservation["passed"] is True
    assert conservation["covered_area_A2"] == pytest.approx(4.0)
    assert conservation["uncovered_area_A2"] == pytest.approx(96.0)
    assert conservation["residual_A2"] == pytest.approx(0.0, abs=1.0e-12)
    assert conservation["tolerance_A2"] == pytest.approx(1.0e-8)


def _physical_torus_perimeter(geometry, lattice, **kwargs):
    from sam_structure_tools import periodic_torus_perimeter_A

    return periodic_torus_perimeter_A(geometry, lattice, **kwargs)


def test_physical_torus_perimeter_has_known_10_by_20_value():
    from shapely.geometry import box

    lattice = np.diag([10.0, 20.0])
    rectangle = box(0.2, 0.3, 0.4, 0.6)  # 2 A by 6 A

    assert _physical_torus_perimeter(rectangle, lattice) == pytest.approx(16.0)


def test_physical_torus_perimeter_uses_skew_lattice_metric():
    from shapely.geometry import box

    lattice = np.column_stack(([10.0, 0.0], [3.0, 8.0]))
    rectangle = box(0.1, 0.2, 0.3, 0.45)
    expected = 2.0 * (
        0.2 * np.linalg.norm(lattice[:, 0])
        + 0.25 * np.linalg.norm(lattice[:, 1])
    )

    assert _physical_torus_perimeter(rectangle, lattice) == pytest.approx(
        expected, abs=1.0e-12
    )


def test_physical_torus_perimeter_is_seam_and_origin_invariant():
    from shapely import affinity

    cell = np.diag([10.0, 20.0, 30.0])
    frame = identity_frame()
    lattice = _surface_lattice_for_test(cell, frame)
    interior, _ = periodic_surface_polygon(
        _fractional_rect_to_uv((0.3, 0.2, 0.5, 0.5), lattice),
        cell=cell,
        periodic_axes=(0, 1),
        surface_frame=frame,
    )
    across_u, _ = periodic_surface_polygon(
        _fractional_rect_to_uv((-0.1, 0.2, 0.1, 0.5), lattice),
        cell=cell,
        periodic_axes=(0, 1),
        surface_frame=frame,
    )

    interior_value = _physical_torus_perimeter(interior, lattice)
    assert interior_value == pytest.approx(16.0)
    assert _physical_torus_perimeter(across_u, lattice) == pytest.approx(
        interior_value, abs=1.0e-12
    )

    common_shift = (0.173, 0.287)
    translated = affinity.translate(
        across_u, xoff=common_shift[0], yoff=common_shift[1]
    )
    assert _physical_torus_perimeter(
        translated, lattice, cell_origin_fractional=common_shift
    ) == pytest.approx(interior_value, abs=1.0e-12)


def test_physical_torus_perimeter_full_cell_is_zero_and_true_seam_counts_once():
    from shapely.geometry import box
    from shapely.ops import unary_union

    lattice = np.diag([10.0, 20.0])
    assert _physical_torus_perimeter(
        box(0.0, 0.0, 1.0, 1.0), lattice
    ) == pytest.approx(0.0, abs=1.0e-12)

    # Occupied on only one side of the U cut: x=0 is a real torus interface.
    # It and x=0.4 each contribute one 20 A segment, never a 0.5 correction.
    seam_collinear_band = box(0.0, 0.0, 0.4, 1.0)
    assert _physical_torus_perimeter(
        seam_collinear_band, lattice
    ) == pytest.approx(40.0, abs=1.0e-12)

    tolerance = 1.0e-6
    near_delta = 0.5 * tolerance
    near_paired_seams = unary_union(
        [
            box(0.0, 0.2, 0.1, 0.4),
            box(0.9, 0.2 + near_delta, 1.0, 0.4 - near_delta),
        ]
    )
    near_audit = _physical_torus_perimeter(
        near_paired_seams,
        lattice,
        seam_coordinate_tolerance_fractional=tolerance,
        return_diagnostics=True,
    )
    assert near_audit["true_interface_on_paired_seams_A"]["u"] == pytest.approx(
        0.0, abs=1.0e-15
    )

    far_delta = 2.0 * tolerance
    far_paired_seams = unary_union(
        [
            box(0.0, 0.2, 0.1, 0.4),
            box(0.9, 0.2 + far_delta, 1.0, 0.4 - far_delta),
        ]
    )
    far_audit = _physical_torus_perimeter(
        far_paired_seams,
        lattice,
        seam_coordinate_tolerance_fractional=tolerance,
        return_diagnostics=True,
    )
    assert far_audit["true_interface_on_paired_seams_A"]["u"] == pytest.approx(
        2.0 * far_delta * np.linalg.norm(lattice[:, 1]), abs=1.0e-15
    )


def test_physical_perimeter_increment_and_hole_diagnostics_use_angstroms():
    from shapely.geometry import box

    lattice = np.diag([10.0, 20.0])
    candidate = box(0.2, 0.3, 0.4, 0.6)
    increment_A, merged = periodic_union_increment_perimeter(
        None, candidate, lattice
    )
    assert increment_A == pytest.approx(16.0)
    assert merged.equals(candidate)

    diagnostics = periodic_hole_diagnostics(
        candidate,
        200.0,
        surface_lattice_uv_A=lattice,
    )
    assert diagnostics["hole_perimeters_A"] == pytest.approx([16.0])
    assert diagnostics["total_hole_perimeter_A"] == pytest.approx(16.0)
    assert diagnostics["maximum_hole_perimeter_A"] == pytest.approx(16.0)
    assert diagnostics["perimeter_metric"]["unit"] == "A"
    assert diagnostics["perimeter_metric"]["artificial_cell_seams_counted"] is False


def test_periodic_union_increment_perimeter_with_precomputed_base_perimeter(monkeypatch):
    from shapely.geometry import box
    import sam_structure_tools as tools

    lattice = np.diag([10.0, 20.0])
    base = box(0.1, 0.1, 0.4, 0.4)
    candidate = box(0.3, 0.3, 0.6, 0.6)

    # Baseline without precomputed base perimeter
    inc_baseline, merged_baseline = periodic_union_increment_perimeter(
        base, candidate, lattice
    )

    base_perimeter = tools.periodic_torus_perimeter_A(base, lattice)

    # With precomputed base perimeter
    inc_fast, merged_fast = periodic_union_increment_perimeter(
        base, candidate, lattice, current_union_perimeter_A=base_perimeter
    )

    assert inc_fast == pytest.approx(inc_baseline, abs=1.0e-12)
    assert merged_fast.equals(merged_baseline)

    # Verify call count: only 1 call for merged geometry, 0 for base
    call_count = 0
    orig_fn = tools.periodic_torus_perimeter_A

    def counted_fn(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return orig_fn(*args, **kwargs)

    monkeypatch.setattr(tools, "periodic_torus_perimeter_A", counted_fn)
    inc_counted, _ = periodic_union_increment_perimeter(
        base, candidate, lattice, current_union_perimeter_A=base_perimeter
    )
    assert inc_counted == pytest.approx(inc_baseline, abs=1.0e-12)
    assert call_count == 1


def test_periodic_union_increment_perimeter_rewards_compact_fill():
    cell, lattice, frame = _diagonal_cell_lattice_frame()

    def geom(frac_rect):
        vertices = _fractional_rect_to_uv(frac_rect, lattice)
        geometry, _ = periodic_surface_polygon(
            vertices, cell=cell, periodic_axes=(0, 1), surface_frame=frame,
        )
        return geometry

    base = geom((0.1, 0.1, 0.4, 0.4))
    adjacent = geom((0.4, 0.1, 0.7, 0.4))  # 与 base 共边
    isolated = geom((0.7, 0.7, 0.9, 0.9))  # 远离

    adjacent_increment, merged = periodic_union_increment_perimeter(base, adjacent)
    isolated_increment, _ = periodic_union_increment_perimeter(base, isolated)
    assert adjacent_increment < isolated_increment

    # 填孔：四块矩形围成带孔环，填孔应降低总周长（负增量）
    ring_pieces = [
        geom((0.2, 0.7, 0.8, 0.8)),  # 上
        geom((0.2, 0.2, 0.8, 0.3)),  # 下
        geom((0.2, 0.3, 0.3, 0.7)),  # 左
        geom((0.7, 0.3, 0.8, 0.7)),  # 右
    ]
    ring_union = None
    for piece in ring_pieces:
        _, ring_union = periodic_union_increment_perimeter(ring_union, piece)
    filler = geom((0.3, 0.3, 0.7, 0.7))  # 精确贴合中心孔壁（填实孔）
    increment, _ = periodic_union_increment_perimeter(ring_union, filler)
    assert increment < 0.0
    # 而一个放不进孔的岛状分子仍是正增量
    island, _ = periodic_union_increment_perimeter(
        ring_union, geom((0.55, 0.55, 0.6, 0.6))
    )
    assert island > 0.0


def test_torus_component_geometries_share_the_exact_diagnostics_merge_rule():
    from shapely.geometry import box
    from shapely.ops import unary_union

    # One uncovered strip is split by the U seam; a second smaller hole is planar.
    occupied = unary_union(
        [
            box(0.10, 0.0, 0.90, 1.0),
            box(0.0, 0.30, 0.10, 0.70),
            box(0.90, 0.30, 1.0, 0.70),
            box(0.42, 0.42, 0.58, 0.58),
        ]
    )
    lattice = np.asarray([[10.0, 2.0], [0.0, 20.0]])
    components = periodic_uncovered_component_geometries(
        occupied,
        abs(np.linalg.det(lattice)),
        surface_lattice_uv_A=lattice,
    )
    diagnostics = periodic_hole_diagnostics(
        occupied,
        abs(np.linalg.det(lattice)),
        surface_lattice_uv_A=lattice,
    )

    assert components
    assert all("fragment_geometries" in component for component in components)
    assert len(components[0]["fragment_geometries"]) > 1
    assert [component["area_A2"] for component in components] == pytest.approx(
        diagnostics["hole_areas_A2"], abs=1.0e-12
    )
    assert components[0]["area_A2"] == pytest.approx(
        diagnostics["maximum_hole_area_A2"], abs=1.0e-12
    )
    assert components[0]["component_id"] == diagnostics["holes"][0]["component_id"]


def test_periodic_physical_geometry_metrics_rectangular_seam_and_area():
    from shapely.geometry import box

    lattice = np.asarray([[10.0, 0.0], [0.0, 20.0]])
    across_left = box(0.90, 0.20, 0.95, 0.40)
    across_right = box(0.05, 0.20, 0.10, 0.40)
    seam = periodic_physical_geometry_metrics(
        across_left, across_right, lattice
    )
    overlap = periodic_physical_geometry_metrics(
        box(0.10, 0.10, 0.40, 0.40),
        box(0.20, 0.20, 0.50, 0.50),
        lattice,
    )

    assert seam["minimum_distance_A"] == pytest.approx(1.0, abs=1.0e-12)
    assert seam["intersection_area_A2"] == pytest.approx(0.0, abs=1.0e-12)
    assert seam["intersects"] is False
    assert overlap["minimum_distance_A"] == pytest.approx(0.0, abs=1.0e-12)
    assert overlap["intersection_area_A2"] == pytest.approx(8.0, abs=1.0e-12)
    assert overlap["left_area_A2"] == pytest.approx(18.0, abs=1.0e-12)
    assert overlap["right_area_A2"] == pytest.approx(18.0, abs=1.0e-12)


def test_periodic_physical_geometry_metrics_skew_and_origin_translation_invariant():
    from shapely import affinity
    from shapely.geometry import box

    lattice = np.asarray([[10.0, 4.0], [0.0, 20.0]])
    left = box(0.90, 0.20, 0.95, 0.40)
    right = box(0.05, 0.20, 0.10, 0.40)
    original = periodic_physical_geometry_metrics(left, right, lattice)
    translated = periodic_physical_geometry_metrics(
        affinity.translate(left, xoff=0.37, yoff=-0.23),
        affinity.translate(right, xoff=0.37, yoff=-0.23),
        lattice,
        cell_origin_fractional=(0.37, -0.23),
    )

    # For a skew cell the shortest separation is the component of 0.1*a
    # perpendicular to b, not a fractional/unit-square distance.
    expected = 0.1 * abs(np.linalg.det(lattice)) / np.linalg.norm(lattice[:, 1])
    assert original["minimum_distance_A"] == pytest.approx(expected, abs=1.0e-12)
    assert translated["minimum_distance_A"] == pytest.approx(
        original["minimum_distance_A"], abs=1.0e-12
    )
    assert translated["intersection_area_A2"] == pytest.approx(
        original["intersection_area_A2"], abs=1.0e-12
    )
    assert translated["left_area_A2"] == pytest.approx(
        original["left_area_A2"], abs=1.0e-12
    )


def test_periodic_physical_geometry_metrics_adaptively_enumerates_nonreduced_images():
    from shapely.geometry import Point

    # b = 20*a + (0, 0.1 A).  The closest image differs by ten copies of a,
    # which a fixed 3x3 image stencil cannot see.
    lattice = np.asarray([[1.0, 20.0], [0.0, 0.1]])
    result = periodic_physical_geometry_metrics(
        Point(0.25, 0.75), Point(0.25, 0.25), lattice
    )

    assert result["minimum_distance_A"] == pytest.approx(0.05, abs=1.0e-12)
    audit = result["image_enumeration_audit"]
    assert audit["complete"] is True
    assert audit["fixed_stencil_used"] is False
    assert audit["method"] == "gauss_reduced_adaptive_complete_cvp_enumeration"
    assert audit["maximum_absolute_original_image_index"] >= 10
    assert result["periodic_images_evaluated"] == audit["image_count"]
    assert abs(round(np.linalg.det(audit["unimodular_transform"]))) == 1


def test_area_clustering_is_deterministic_and_does_not_require_cluster_count():
    clusters = cluster_1d_by_tolerance([2.0, 1.1, 1.0, 1.25], tolerance=0.2)

    assert [cluster["member_indices"] for cluster in clusters] == [
        [2, 1],
        [3],
        [0],
    ]
    assert [cluster["cluster_id"] for cluster in clusters] == [
        "cluster-0001",
        "cluster-0002",
        "cluster-0003",
    ]
