"""Geometric invariants of packing first, then binding a rigid PO3 headgroup."""

from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from monolayer_headgroup_fit import (
    fit_phosphonate_pc_rotation,
    rotate_phosphonate_about_pc,
    rotate_sam_about_surface_normal,
    sample_phosphonate_pc_rotations,
    single_metal_pc_rotation_angles,
)


@pytest.fixture
def tilted_phosphonate():
    """A tetrahedral P whose P–C axis is tilted relative to all Cartesian axes."""

    p = np.array([3.0, -4.0, 2.0])
    axis = np.array([1.0, 2.0, 3.0])
    axis /= np.linalg.norm(axis)
    transverse = np.cross(axis, [0.0, 0.0, 1.0])
    transverse /= np.linalg.norm(transverse)
    second = np.cross(axis, transverse)
    oxygens = [
        p + 1.54 * (-axis / 3.0 + np.sqrt(8.0 / 9.0) * (
            np.cos(phi) * transverse + np.sin(phi) * second
        ))
        for phi in np.deg2rad([0.0, 120.0, 240.0])
    ]
    c = p + 1.84 * axis
    return np.array([p, *oxygens, c, c + [1.0, 0.2, 0.5], c + [-0.4, 0.9, 0.1]])


def independently_rotate_head(points, angle):
    """SciPy column-vector convention supplies an independent Rodrigues oracle."""

    result = points.copy()
    axis = points[4] - points[0]
    axis /= np.linalg.norm(axis)
    result[1:4] = points[0] + Rotation.from_rotvec(np.deg2rad(angle) * axis).apply(points[1:4] - points[0])
    return result


def assert_tetrahedral_invariants(before, after):
    before_vectors = before[[1, 2, 3, 4]] - before[0]
    after_vectors = after[[1, 2, 3, 4]] - after[0]
    # The complete Gram matrix verifies all P–O/P–C lengths and all six angles.
    np.testing.assert_allclose(after_vectors @ after_vectors.T, before_vectors @ before_vectors.T, atol=4e-14)
    # Oriented volume additionally rejects reflection or relabelling the O atoms.
    assert np.linalg.det(after_vectors[:3]) == pytest.approx(np.linalg.det(before_vectors[:3]), abs=2e-14)
    np.testing.assert_array_equal(after[[0, 4, 5, 6]], before[[0, 4, 5, 6]])


def test_whole_sam_azimuth_is_right_handed_and_keeps_p_fixed(tilted_phosphonate):
    before = tilted_phosphonate.copy()
    after = rotate_sam_about_surface_normal(before, 0, 90.0)
    expected = before[0] + Rotation.from_euler("z", 90, degrees=True).apply(before - before[0])
    np.testing.assert_allclose(after, expected, atol=2e-15)
    np.testing.assert_array_equal(after[0], before[0])
    np.testing.assert_array_equal(before, tilted_phosphonate)
    np.testing.assert_allclose(after[:, 2], before[:, 2], atol=2e-15)
    np.testing.assert_allclose(np.linalg.norm(after[:, None] - after, axis=2), np.linalg.norm(before[:, None] - before, axis=2), atol=2e-15)


def test_whole_sam_rotation_supports_non_z_surface_normal(tilted_phosphonate):
    normal = np.array([3.0, 4.0, -2.0])
    angle = -41.5
    before = tilted_phosphonate
    actual = rotate_sam_about_surface_normal(before, 0, angle, normal)
    oracle = before[0] + Rotation.from_rotvec(np.deg2rad(angle) * normal / np.linalg.norm(normal)).apply(before - before[0])
    np.testing.assert_allclose(actual, oracle, atol=3e-15)


@pytest.mark.parametrize("angle", [0.0, 37.25, 179.0, -103.5, 540.0])
def test_head_rotation_preserves_organic_tetrahedron_and_handedness(tilted_phosphonate, angle):
    before = tilted_phosphonate.copy()
    actual = rotate_phosphonate_about_pc(before, 0, [1, 2, 3], 4, angle)
    np.testing.assert_allclose(actual, independently_rotate_head(before, angle), atol=2e-15)
    assert_tetrahedral_invariants(before, actual)
    np.testing.assert_array_equal(before, tilted_phosphonate)


def test_fit_recovers_known_torsion_after_independent_global_azimuth(tilted_phosphonate):
    packed = rotate_sam_about_surface_normal(tilted_phosphonate, 0, 67.0)
    expected = independently_rotate_head(packed, 137.2)
    fit = fit_phosphonate_pc_rotation(packed, 0, [1, 2, 3], 4, expected[1:4])
    assert fit["pc_rotation_deg"] == pytest.approx(137.2, abs=1e-12)
    assert fit["oxygen_rmsd_A"] < 1e-14
    assert fit["headgroup_rmsd_A"] < 1e-14
    np.testing.assert_allclose(fit["coordinates"], expected, atol=2e-15)
    assert_tetrahedral_invariants(packed, fit["coordinates"])


@pytest.mark.parametrize("donors", [(3, 1, 2), (2, 1), (3,)])
def test_fit_maps_actual_donor_identity_and_rotates_all_three_o(tilted_phosphonate, donors):
    expected = independently_rotate_head(tilted_phosphonate, -71.4)
    fit = fit_phosphonate_pc_rotation(
        tilted_phosphonate, 0, [1, 2, 3], 4, expected[list(donors)], donor_order=donors,
    )
    assert fit["pc_rotation_deg"] == pytest.approx(288.6, abs=1e-12)
    assert fit["donor_order"] == donors
    np.testing.assert_allclose(fit["coordinates"], expected, atol=2e-15)
    assert_tetrahedral_invariants(tilted_phosphonate, fit["coordinates"])


def test_unreachable_target_reports_residual_without_tilting_or_translating_body(tilted_phosphonate):
    before = tilted_phosphonate
    target = independently_rotate_head(before, 42.3)
    axis = (before[4] - before[0]) / np.linalg.norm(before[4] - before[0])
    target[1:4] += 0.8 * axis
    target_p = before[0] + [0.4, 0.0, 0.0]
    fit = fit_phosphonate_pc_rotation(before, 0, [1, 2, 3], 4, target[1:4], target_p_coordinate=target_p)
    assert fit["pc_rotation_deg"] == pytest.approx(42.3, abs=1e-12)
    assert fit["oxygen_rmsd_A"] == pytest.approx(0.8, abs=1e-14)
    assert fit["headgroup_rmsd_A"] == pytest.approx(np.sqrt((3 * 0.8**2 + 0.4**2) / 4), abs=1e-14)
    assert_tetrahedral_invariants(before, fit["coordinates"])


def test_analytic_fit_is_global_minimum_against_dense_independent_scan(tilted_phosphonate):
    target = independently_rotate_head(tilted_phosphonate, 173.31)[1:4]
    target += [[0.2, -0.4, 0.1], [-0.1, 0.25, 0.37], [0.4, 0.1, -0.3]]
    fit = fit_phosphonate_pc_rotation(tilted_phosphonate, 0, [1, 2, 3], 4, target)
    axis = (tilted_phosphonate[4] - tilted_phosphonate[0])
    axis /= np.linalg.norm(axis)
    rotations = Rotation.from_rotvec(np.deg2rad(np.arange(0, 360, 0.1))[:, None] * axis)
    source = tilted_phosphonate[1:4] - tilted_phosphonate[0]
    target_vectors = target - tilted_phosphonate[0]
    sampled_sse = sum(np.sum((rotations.apply(vector) - wanted) ** 2, axis=1) for vector, wanted in zip(source, target_vectors))
    assert fit["oxygen_rmsd_A"] ** 2 * 3 <= float(np.min(sampled_sse)) + 1e-13


def test_feasible_arc_includes_exact_boundaries_and_maximum_step(tilted_phosphonate):
    before = tilted_phosphonate
    desired_angle = 43.5
    half_width = 17.3
    target = independently_rotate_head(before, desired_angle)[1:4]
    boundary = independently_rotate_head(before, desired_angle + half_width)
    limit = np.sqrt(np.sum((boundary[1:4] - target) ** 2) / 4)
    candidates = sample_phosphonate_pc_rotations(
        before, 0, [1, 2, 3], 4, target, max_headgroup_rmsd_A=limit, angle_step_deg=5.0,
    )
    assert candidates[0]["is_analytic_minimum"] is True
    offsets = sorted(item["offset_from_optimum_deg"] for item in candidates)
    assert offsets[0] == pytest.approx(-half_width, abs=1e-12)
    assert offsets[-1] == pytest.approx(half_width, abs=1e-12)
    assert max(np.diff(offsets)) <= 5.0 + 1e-12
    assert all(item["headgroup_rmsd_A"] <= limit + 1e-13 for item in candidates)
    assert candidates[-1]["headgroup_rmsd_A"] == pytest.approx(limit, abs=1e-13)
    for candidate in candidates:
        assert_tetrahedral_invariants(before, candidate["coordinates"])


def test_unreachable_rmsd_bound_gives_no_candidates(tilted_phosphonate):
    target = tilted_phosphonate[1:4] + [30.0, 0.0, 0.0]
    candidates = sample_phosphonate_pc_rotations(
        tilted_phosphonate, 0, [1, 2, 3], 4, target, max_headgroup_rmsd_A=0.5, angle_step_deg=5.0,
    )
    assert candidates == []


def test_zero_rmsd_bound_retains_numerically_exact_solution(tilted_phosphonate):
    target = independently_rotate_head(tilted_phosphonate, 16.2)[1:4]
    candidates = sample_phosphonate_pc_rotations(
        tilted_phosphonate, 0, [1, 2, 3], 4, target, max_headgroup_rmsd_A=0.0, angle_step_deg=5.0,
    )
    assert len(candidates) == 1
    assert candidates[0]["pc_rotation_deg"] == pytest.approx(16.2, abs=1e-12)


def test_angle_independent_objective_samples_full_circle_without_duplicates(tilted_phosphonate):
    before = tilted_phosphonate
    axis = (before[4] - before[0]) / np.linalg.norm(before[4] - before[0])
    target = before[0] + np.array([1.0, 2.0, 3.0])[:, None] * axis
    fit = fit_phosphonate_pc_rotation(before, 0, [1, 2, 3], 4, target)
    assert fit["objective_is_angle_independent"] is True
    assert fit["pc_rotation_deg"] == 0.0
    candidates = sample_phosphonate_pc_rotations(
        before, 0, [1, 2, 3], 4, target, max_headgroup_rmsd_A=10, angle_step_deg=5.0,
    )
    angles = sorted(item["pc_rotation_deg"] for item in candidates)
    assert len(angles) == len(set(angles)) == 72
    np.testing.assert_allclose(angles, np.arange(0.0, 360.0, 5.0), atol=1e-13)


def test_single_metal_roots_hit_target_and_keep_all_other_geometry(tilted_phosphonate):
    before = tilted_phosphonate
    metal = before[0] + 2.0 * (before[1] - before[0])
    candidates = single_metal_pc_rotation_angles(before, 0, [1, 2, 3], 4, metal, target_distance_A=2.2)
    assert len(candidates) == 6
    assert {item["donor_local"] for item in candidates} == {1, 2, 3}
    for candidate in candidates:
        assert candidate["target_reachable"] is True
        assert candidate["metal_distance_A"] == pytest.approx(2.2, abs=3e-14)
        assert candidate["target_distance_error_A"] < 3e-14
        assert_tetrahedral_invariants(before, candidate["coordinates"])
        oracle = independently_rotate_head(before, candidate["pc_rotation_deg"])
        np.testing.assert_allclose(candidate["coordinates"], oracle, atol=3e-15)


@pytest.mark.parametrize("target_distance,closest_index", [(0.2, 0), (100.0, 1)])
def test_single_metal_unreachable_target_returns_nearest_possible_distance(tilted_phosphonate, target_distance, closest_index):
    before = tilted_phosphonate
    metal = before[0] + 2.0 * (before[1] - before[0])
    candidates = single_metal_pc_rotation_angles(
        before, 0, [1, 2, 3], 4, metal, target_distance_A=target_distance, donor_local=2,
    )
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["target_reachable"] is False
    assert candidate["metal_distance_A"] == pytest.approx(candidate["reachable_distance_range_A"][closest_index], abs=1e-14)
    assert candidate["donor_local"] == 2
    assert_tetrahedral_invariants(before, candidate["coordinates"])


def test_single_metal_on_axis_has_angle_independent_distance(tilted_phosphonate):
    before = tilted_phosphonate
    metal = before[4]
    target = float(np.linalg.norm(before[1] - metal))
    candidates = single_metal_pc_rotation_angles(before, 0, [1, 2, 3], 4, metal, target_distance_A=target)
    assert len(candidates) == 3
    assert all(item["pc_rotation_deg"] == 0.0 and item["target_reachable"] for item in candidates)
    assert all(item["metal_distance_A"] == pytest.approx(target, abs=1e-14) for item in candidates)


def test_single_metal_window_samples_both_arcs_and_retains_target_roots(tilted_phosphonate):
    before = tilted_phosphonate
    metal = before[0] + 2.0 * (before[1] - before[0])
    roots = single_metal_pc_rotation_angles(before, 0, [1, 2, 3], 4, metal, donor_local=1)
    candidates = single_metal_pc_rotation_angles(
        before, 0, [1, 2, 3], 4, metal, donor_local=1,
        distance_window_A=(1.7, 2.8), angle_step_deg=5.0,
    )
    angles = np.array([item["pc_rotation_deg"] for item in candidates])
    assert len(angles) > len(roots)
    assert len(angles) == len(set(angles))
    for root in roots:
        assert np.min(np.abs((angles - root["pc_rotation_deg"] + 180.0) % 360.0 - 180.0)) < 1e-11
    distances = [item["metal_distance_A"] for item in candidates]
    assert min(distances) == pytest.approx(1.7, abs=1e-13)
    assert max(distances) == pytest.approx(2.8, abs=1e-13)
    assert all(1.7 - 1e-13 <= distance <= 2.8 + 1e-13 for distance in distances)
    # Independently rotate a dense circle: every feasible angle is no farther
    # than half the maximum sample gap from one returned endpoint/grid/root.
    dense_angles = np.arange(0.0, 360.0, 0.05)
    axis = (before[4] - before[0]) / np.linalg.norm(before[4] - before[0])
    rotated = before[0] + Rotation.from_rotvec(np.deg2rad(dense_angles)[:, None] * axis).apply(before[1] - before[0])
    dense_distances = np.linalg.norm(rotated - metal, axis=1)
    feasible = dense_angles[(dense_distances >= 1.7) & (dense_distances <= 2.8)]
    gaps = np.min(np.abs((feasible[:, None] - angles[None] + 180.0) % 360.0 - 180.0), axis=1)
    assert max(gaps) <= 2.5 + 1e-10
    for candidate in candidates:
        assert_tetrahedral_invariants(before, candidate["coordinates"])


def test_single_metal_zero_width_window_has_only_exact_roots(tilted_phosphonate):
    before = tilted_phosphonate
    metal = before[0] + 2.0 * (before[1] - before[0])
    candidates = single_metal_pc_rotation_angles(
        before, 0, [1, 2, 3], 4, metal, donor_local=1,
        distance_window_A=(2.2, 2.2), angle_step_deg=5.0,
    )
    assert len(candidates) == 2
    assert all(item["metal_distance_A"] == pytest.approx(2.2, abs=1e-13) for item in candidates)


def test_single_metal_window_does_not_require_exact_target_to_be_reachable(tilted_phosphonate):
    before = tilted_phosphonate
    metal = before[0] + 2.0 * (before[1] - before[0])
    candidates = single_metal_pc_rotation_angles(
        before, 0, [1, 2, 3], 4, metal, donor_local=1, target_distance_A=0.2,
        distance_window_A=(1.7, 2.8), angle_step_deg=5.0,
    )
    assert candidates
    assert all(not item["target_reachable"] for item in candidates)
    assert all(1.7 - 1e-13 <= item["metal_distance_A"] <= 2.8 + 1e-13 for item in candidates)
    assert single_metal_pc_rotation_angles(
        before, 0, [1, 2, 3], 4, metal, donor_local=1,
        distance_window_A=(10.0, 11.0), angle_step_deg=5.0,
    ) == []


def test_single_metal_angle_independent_window_samples_full_circle(tilted_phosphonate):
    before = tilted_phosphonate
    distance = float(np.linalg.norm(before[1] - before[4]))
    candidates = single_metal_pc_rotation_angles(
        before, 0, [1, 2, 3], 4, before[4], donor_local=1,
        distance_window_A=(distance - 0.1, distance + 0.1), angle_step_deg=5.0,
    )
    assert len(candidates) == 72
    np.testing.assert_allclose(sorted(item["pc_rotation_deg"] for item in candidates), np.arange(0.0, 360.0, 5.0), atol=1e-13)


@pytest.mark.parametrize("options", [
    {"distance_window_A": (1.7, 2.8)},
    {"angle_step_deg": 5.0},
    {"distance_window_A": (2.8, 1.7), "angle_step_deg": 5.0},
    {"distance_window_A": (1.7, np.inf), "angle_step_deg": 5.0},
    {"distance_window_A": (1.7, 2.8), "angle_step_deg": 0.0},
])
def test_single_metal_rejects_invalid_window_options(tilted_phosphonate, options):
    with pytest.raises(ValueError):
        single_metal_pc_rotation_angles(tilted_phosphonate, 0, [1, 2, 3], 4, [0, 0, 0], **options)


@pytest.mark.parametrize("kwargs,match", [
    ({"o_locals": [1, 1, 2]}, "distinct"),
    ({"o_locals": [1, 2]}, "three"),
    ({"c_local": 0}, "distinct"),
    ({"p_local": -1}, "outside"),
    ({"p_local": True}, "integer"),
    ({"donor_order": [1, 2, 4]}, "headgroup O"),
    ({"target_p_coordinate": [0, np.nan, 1]}, "finite"),
])
def test_fit_rejects_invalid_geometry_contract(tilted_phosphonate, kwargs, match):
    arguments = dict(coordinates=tilted_phosphonate, p_local=0, o_locals=[1, 2, 3], c_local=4, target_oxygen_coordinates=tilted_phosphonate[1:4])
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=match):
        fit_phosphonate_pc_rotation(**arguments)


def test_fit_requires_explicit_partial_donor_mapping_and_nondegenerate_axis(tilted_phosphonate):
    with pytest.raises(ValueError, match="explicit"):
        fit_phosphonate_pc_rotation(tilted_phosphonate, 0, [1, 2, 3], 4, tilted_phosphonate[1:2])
    broken = tilted_phosphonate.copy()
    broken[4] = broken[0]
    with pytest.raises(ValueError, match="axis must be nonzero"):
        fit_phosphonate_pc_rotation(broken, 0, [1, 2, 3], 4, broken[1:4])


def test_rotation_and_sampling_reject_nonfinite_or_invalid_parameters(tilted_phosphonate):
    with pytest.raises(ValueError, match="angle must be finite"):
        rotate_sam_about_surface_normal(tilted_phosphonate, 0, np.nan)
    with pytest.raises(ValueError, match="nonzero"):
        rotate_sam_about_surface_normal(tilted_phosphonate, 0, 0, [0, 0, 0])
    with pytest.raises(ValueError, match="angle_step_deg"):
        sample_phosphonate_pc_rotations(tilted_phosphonate, 0, [1, 2, 3], 4, tilted_phosphonate[1:4], max_headgroup_rmsd_A=0.5, angle_step_deg=0)
    with pytest.raises(ValueError, match="nonnegative"):
        sample_phosphonate_pc_rotations(tilted_phosphonate, 0, [1, 2, 3], 4, tilted_phosphonate[1:4], max_headgroup_rmsd_A=-0.5, angle_step_deg=5)
    with pytest.raises(ValueError, match="headgroup O"):
        single_metal_pc_rotation_angles(tilted_phosphonate, 0, [1, 2, 3], 4, [0, 0, 0], donor_local=4)
