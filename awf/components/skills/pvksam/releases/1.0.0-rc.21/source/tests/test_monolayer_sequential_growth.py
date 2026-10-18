import copy
import gc
import gzip
import hashlib
import itertools
import json
import os
import random
import stat
import subprocess
import time
import weakref
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.io import write as ase_write


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from monolayer_sequential_growth import (
    CAPACITY_RANKING_AUDIT_SCHEMA,
    _COLLISION_ALGORITHM_VERSION,
    CollisionQueryProfiler,
    LazyCollisionCache,
    ProjectionCache,
    SamplingPolicy,
    _assemble,
    _build_parser,
    _candidate_domain_fingerprint,
    _candidate_domains,
    _capacity_beam,
    _capacity_policy_manifest,
    _candidate_id_from_fields,
    _collision_contract_fingerprint,
    _common_surface_frame,
    _conformer_geometry_sha256,
    _load_or_build_projection_cache,
    _load_working_substrate,
    _build_single_metal_context,
    _discover_uncovered_cn5_metals,
    _generate_single_metal_triangle_candidates,
    _point_in_triangle_2d,
    _resolve_conformer_source_anchor_height,
    _hard_substrate_collision_audit,
    _height_substrate_clearance_audit,
    _repair_neighborhood_scope,
    _select_capacity_finalist,
    _static_conflict_compatible_capacity,
    _target_headgroup,
    audit_final_registered_interfaces,
    audit_final_height_clearance,
    build_full_to_working_atom_map,
    build_substrate_collision_contract,
    capacity_debt_metrics,
    capacity_ranking_artifact_contract,
    capture_immutable_parent_reference,
    _parent_requested_metric_contract,
    capture_reference_metrics,
    certified_static_repair_gate,
    cohesive_filter_conformer_energy,
    cohesive_active_denticity,
    cohesive_filter_distinct_neighbors,
    cohesive_filter_perimeter_increment,
    cohesive_filter_projected_area,
    cohesive_filter_side_contacts,
    cohesive_lazy_side_contacts,
    cohesive_perimeter_condensation_gain,
    cohesive_side_contact_report,
    cohesive_xy_bound,
    cohesive_xy_may_contact,
    CohesiveProjectionNeighborIndex,
    CandidateCollisionProjectionIndex,
    choose_cohesive_frontier_candidate,
    choose_capacity_candidate,
    conformer_sampling_table,
    enumerate_local_repair_solutions,
    evaluate_local_repair_removal_set,
    execute_local_repair_package,
    reconstruct_parent_placements,
    repair_acceptance_gates,
    repair_objective,
    repair_output_contract,
    resolve_candidate_registered_bonds,
    resolve_cohesive_cli_tolerances,
    resolve_cohesive_relative_tolerances,
    resolve_cohesive_umbrella_audit,
    run_bounded_local_cardinality_repair,
    run_fixed_site_replacements,
    run_sequential_domains,
    simulate_candidate_transition,
    strict_repair_pareto,
    sequential_growth_main,
    solve_static_conflict_capacity,
    stage3_choose_shape_diverse_next,
    stage3_complete_link_clusters,
    stage3_standardized_shape_features,
    summarize_trajectory_results,
    surface_collision_report,
    verify_parent_unchanged,
    write_capacity_ranking_audit,
)

from sam_structure_tools import MoleculeSpec, _surface_lattice_uv, surface_frame_coordinates
from shapely.geometry import Point, box as shapely_box


def _pack_memo_selection_fixture():
    """Real periodic contact/perimeter geometry, with several exact RNG ties."""
    import monolayer_sequential_growth as growth
    lattice = np.diag([12.0, 12.0])
    candidates = []
    for i, angle in enumerate(np.arange(0.0, 360.0, 45.0)):
        xy = np.array([1.0, 1.0]) + 3.5 * np.array([
            np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
        candidate = _cohesive_policy_candidate(f"pending-{i}", i, 1, 0.0)
        candidate.update({"coordinates": np.array([[*xy, 0.0], [*xy, 5.0]]),
                          "pose_mode": "pack-then-bind", "headgroup_binding_pending": True,
                          "template_key": "square", "anchor_uv_A": xy})
        candidates.append(candidate)
    seed = _cohesive_policy_candidate("accepted-seed", len(candidates), 1, 0.0)
    seed.update({"coordinates": np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 5.0]]),
                 "pose_mode": "pack-then-bind", "headgroup_binding_pending": False})
    all_candidates = candidates + [seed]
    geometries = {}
    for c in all_candidates:
        xy = c["coordinates"][0, :2] / 12.0
        geometries[c["candidate_index"]] = growth._wrap_to_unit_cell(
            shapely_box(xy[0]-1.7/12, xy[1]-1.7/12,
                        xy[0]+1.7/12, xy[1]+1.7/12))
    cache = _fake_cache({}, geometries, [], area_scale_A2=144.0,
                        surface_lattice_uv_A=lattice)
    cache.template_records = [{"template_key": "square", "selection_rank": 1,
        "relative_vertices_uv_A": [[-1.7, -1.7], [1.7, -1.7], [1.7, 1.7], [-1.7, 1.7]]}]
    thresholds = {"H-H": 1.5, "H-heavy": 1.8, "heavy-heavy": 2.2}
    index = growth.CohesiveProjectionNeighborIndex(
        projection_cache=cache, candidates=all_candidates,
        collision_thresholds_A=thresholds)
    index.add(seed)
    return candidates, seed, cache, index, thresholds


@pytest.mark.parametrize("mode", ["discrete", "continuous-snap"])
def test_pack_epoch_memo_exact_reject_accept_sequence(mode, monkeypatch):
    """Memo changes work only: every selected pose, audit and RNG state match."""
    import monolayer_sequential_growth as growth
    calls = {"perimeter": 0, "contact": 0, "boundary": 0}
    for name, key in [("periodic_union_increment_perimeter", "perimeter"),
                      ("cohesive_side_contact_report", "contact"),
                      ("extract_frontier_boundary_rings", "boundary")]:
        original = getattr(growth, name)
        def count(*args, _original=original, _key=key, **kwargs):
            calls[_key] += 1
            return _original(*args, **kwargs)
        monkeypatch.setattr(growth, name, count)

    def run(enabled):
        live, seed, cache, index, thresholds = _pack_memo_selection_fixture()
        index.pack_then_bind_memo_enabled = enabled
        placed = [seed]
        union = cache.candidate_geometries[seed["candidate_index"]]
        rng = random.Random(672)
        records, work_per_step = [], []
        for accepted in [False, True, False, False, True, False]:
            previous = dict(calls)
            winner, audit = growth.choose_cohesive_frontier_candidate(
                candidates=live, placed=placed, accepted_union=union,
                projection_cache=cache, projection_neighbor_index=index,
                cell=np.diag([12.0, 12.0, 30.0]), periodic_axes=(0, 1),
                collision_thresholds_A=thresholds, rng=rng,
                cohesive_frontier_mode=mode, frontier_snap_radius_A=4.0)
            records.append((winner["candidate_id"], winner["coordinates"].tolist(),
                            copy.deepcopy(audit), rng.getstate()))
            work_per_step.append({key: calls[key]-previous[key] for key in calls})
            live = [c for c in live if c["candidate_id"] != winner["candidate_id"]]
            if accepted:
                winner["headgroup_binding_pending"] = False
                placed.append(winner)
                index.candidate_by_id[winner["candidate_id"]] = winner
                index.add(winner)
                assert index._pack_scoring_memo == {}
                _, union = growth.periodic_union_increment_A2(
                    union, cache.candidate_geometries[winner["candidate_index"]], cache.area_scale_A2)
        return records, work_per_step

    reference, reference_work = run(False)
    memoized, memo_work = run(True)
    assert memoized == reference
    for i in (1, 3):  # Only rejected poses have been removed since previous call.
        assert memo_work[i]["perimeter"] == 0
        assert memo_work[i]["boundary"] == 0
        assert reference_work[i]["perimeter"] > 0
    assert memo_work[2]["perimeter"] > 0  # Successful acceptance invalidated memo.
    assert sum(c["contact"] for c in memo_work) < sum(c["contact"] for c in reference_work)


def test_pack_epoch_snap_memo_active_template_counts_and_exact_ties():
    import monolayer_sequential_growth as growth
    lattice = np.array([[12.0, 2.0], [0.0, 10.0]])
    union = shapely_box(0.25, 0.25, 0.50, 0.50)
    templates = {key: {"template_key": key, "relative_vertices_uv_A": vertices}
                 for key, vertices in {
                     "a": [[-1, -1], [1, -1], [1, 1], [-1, 1]],
                     "b": [[-1, -1], [1, -1], [0, 1]],
                 }.items()}
    live = [{"candidate_id": f"{tpl}-{i}", "template_key": tpl,
             "anchor_uv_A": np.array(xy), "candidate_index": i}
            for tpl in templates for i, xy in enumerate([[3., 3.], [4., 5.], [9., 9.]])]
    memo = {}
    def calculate(pool, active, use_memo):
        return growth.focus_continuous_frontier_candidates_streaming(
            accepted_union=union, candidate_templates=active,
            spatial_index=growth.SameTemplatePeriodicSpatialIndex(pool, lattice),
            surface_lattice_uv_A=lattice, snap_radius_A=3.0,
            geometry_memo=memo if use_memo else None)
    calculate(live, templates, True)
    for pool in (live[1:], [c for c in live if c["template_key"] == "b"]):
        active = {key: val for key, val in templates.items()
                  if key in {c["template_key"] for c in pool}}
        reference = calculate(pool, active, False)
        actual = calculate(pool, active, True)
        assert actual[:4] == reference[:4]
        for cid, proposal in actual[4].items():
            for key, value in proposal.items():
                np.testing.assert_equal(value, reference[4][cid][key])


def test_pack_epoch_memo_is_disabled_for_legacy_and_changed_contact_parameters():
    live, seed, cache, index, thresholds = _pack_memo_selection_fixture()
    kwargs = dict(candidates=live, placed=[seed],
                  accepted_union=cache.candidate_geometries[seed["candidate_index"]],
                  cell=np.diag([12., 12., 30.]), periodic_axes=(0, 1),
                  collision_thresholds_A=thresholds, contact_vdw_radius_scale=1.1,
                  frontier_snap_radius_A=4.0)
    memo = index.pack_scoring_memo(**kwargs)
    memo["marker"] = True
    assert index.pack_scoring_memo(**kwargs) is memo
    assert index.pack_scoring_memo(**{**kwargs, "contact_vdw_radius_scale": 1.2}) == {}
    live[0]["pose_mode"] = "legacy"
    assert index.pack_scoring_memo(**kwargs) is None


def _identity_frame():
    return {
        "u_cartesian_unit": [1.0, 0.0, 0.0],
        "v_cartesian_unit": [0.0, 1.0, 0.0],
        "outward_normal_cartesian_unit": [0.0, 0.0, 1.0],
    }


def _axis_02_frame():
    return {
        "u_cartesian_unit": [1.0, 0.0, 0.0],
        "v_cartesian_unit": [0.0, 0.0, 1.0],
        "outward_normal_cartesian_unit": [0.0, -1.0, 0.0],
        "periodic_fractional_axes": [0, 2],
    }


def _make_candidate(
    site, rank, permutation, coordinates, symbols, frame, conformer_sha256=None,
    prototype_id=None,
):
    prototype_id = prototype_id or f"proto-{site}"
    conformer_sha256 = conformer_sha256 or _conformer_geometry_sha256(
        coordinates, symbols
    )
    return {
        "site_instance_id": site,
        "site_prototype_id": prototype_id,
        "selection_rank": rank,
        "cluster_id": rank,
        "source": f"rank-{rank}",
        "target_donor_labels": ["O1", "O2", "O3"],
        "oxygen_permutation": permutation,
        "headgroup_rmsd_A": 0.1,
        "substrate_minimum_distance_A": 2.5,
        "substrate_clearance_ratio": 1.3,
        "surface_frame": frame,
        "conformer_sha256": conformer_sha256,
        "anchor_cartesian_A": list(coordinates[0]),
        "coordinates": np.asarray(coordinates, dtype=float),
        "symbols": np.asarray(symbols, dtype=str),
        "candidate_id": _candidate_id_from_fields(
            site_instance_id=site,
            site_prototype_id=prototype_id,
            prototype_evidence=f"{prototype_id}-evidence",
            selection_rank=rank,
            conformer_sha256=conformer_sha256,
            donor_labels=["O1", "O2", "O3"],
            oxygen_permutation=permutation,
            symbols=np.asarray(symbols, dtype=str),
        ),
    }


def _domain_fixture(site_order=("site-a", "site-b")):
    frame = _identity_frame()
    domains = {}
    for site in site_order:
        domains[site] = [
            _make_candidate(
                site, 1, [1, 2, 3],
                [[0.0, 0.0, 5.0], [1.0, 0.0, 4.0], [0.0, 1.0, 4.0], [-1.0, 0.0, 4.0]],
                ["C", "O", "O", "O"], frame,
            ),
            _make_candidate(
                site, 2, [3, 1, 2],
                [[0.5, 0.5, 5.0], [1.0, 0.5, 4.0], [0.5, 1.5, 4.0], [-0.5, 0.5, 4.0]],
                ["C", "O", "O", "O"], frame,
            ),
        ]
    return domains


def _fingerprint_kwargs():
    return {"cell": np.diag([10.0, 10.0, 20.0]), "periodic_axes": (0, 1)}


def _sorted_candidate_ids(domains):
    return sorted(
        candidate["candidate_id"]
        for candidates in domains.values()
        for candidate in candidates
    )


def _selection_records():
    values = [
        (1, 15, 0.9921875, 50.78297025078126),
        (2, 51, 0.0, 125.3567621625038),
        (3, 32, 0.799560546875, 57.71411875197035),
        (4, 48, 0.730712890625, 60.47346762895133),
        (5, 9, 0.668212890625, 67.3240186667038),
        (6, 37, 0.60205078125, 76.29564041156169),
        (7, 43, 0.564208984375, 79.93061507062119),
        (8, 17, 0.22412109375, 97.58346268999853),
        (9, 3, 0.185791015625, 104.45681399812842),
        (10, 6, 0.163818359375, 109.96365084219617),
    ]
    return [
        {
            "selection_rank": rank,
            "cluster_id": cluster,
            "relative_total_energy_eV": energy,
            "footprint_area_A2": area,
            "structure": {"path": f"selection-{rank:04d}.extxyz", "sha256": "0" * 64},
        }
        for rank, cluster, energy, area in values
    ]










def _cohesive_cache_for_simple_domains(domains, cell):
    from shapely.geometry import box
    geometries = {}
    for index, candidate in enumerate(c for rows in domains.values() for c in rows):
        candidate.update(candidate_id=f"simple-{index}", candidate_index=index,
            denticity=1, target_donor_labels=["O1"], surface_frame=_identity_frame(),
            conformer_energy_eV=0.0, projected_area_A2=1.0,
            headgroup_local_indices_0based=[],
            anchor_cartesian_A=candidate['coordinates'][0].tolist())
        x, y = candidate['coordinates'][0][:2] / np.diag(cell)[:2]
        geometries[index] = box(x, y, x+.01, y+.01)
    return _fake_cache(masks={i:0 for i in geometries}, geometries=geometries,
        site_ids=sorted(domains), area_scale_A2=float(cell[0,0]*cell[1,1]),
        surface_lattice_uv_A=np.asarray(cell)[:2,:2])










def _real_site_instance_schema_fixture():
    """Small worked example using the materializer's real Site Instance fields."""

    return {
        "site_instance_id": "site-prototype-phosphonic-acid-0001--cell-0001",
        "parent_site_prototype_id": "site-prototype-phosphonic-acid-0001",
        "final_denticity": 2,
        "actual_metal_atom_ids_1based": [2, 4],
        "actual_metals": [
            {
                "atom_id_1based": 2,
                "element": "In",
                "actual_element_after_doping": "In",
            },
            {
                "atom_id_1based": 4,
                "element": "In",
                "actual_element_after_doping": "Sn",
            },
        ],
        "donor_metal_mapping": [
            {
                "probe_donor_id": "O1",
                "metal_atom_id_1based": 4,
                "parent_contact_distance_A": 2.14,
            },
            {
                "probe_donor_id": "O3",
                "metal_atom_id_1based": 2,
                "parent_contact_distance_A": 2.18,
            },
        ],
    }


def _real_full_to_working_fixture():
    full = Atoms(
        ["O", "In", "O", "Sn", "O"],
        positions=np.zeros((5, 3)),
        cell=np.diag([10.0, 10.0, 20.0]),
        pbc=True,
    )
    working_ids = [2, 4, 5]
    working = full[[atom_id - 1 for atom_id in working_ids]]
    return build_full_to_working_atom_map(
        working_layer_atom_ids_1based=working_ids,
        working_symbols=working.get_chemical_symbols(),
    )






def _write_layer_partition_fixture(tmp_path, groups):
    substrate_path = tmp_path / "full-substrate.extxyz"
    ase_write(
        substrate_path,
        Atoms(
            ["O", "In", "Sn", "O"],
            positions=np.zeros((4, 3)),
            cell=np.diag([10.0, 10.0, 20.0]),
            pbc=True,
        ),
    )
    groups_path = tmp_path / "layer-groups.json"
    groups_path.write_text(json.dumps(groups))
    return substrate_path, groups_path










def _strict_interface_candidate_fixture(
    *, donor_height_A=2.0, add_nonmapped_collision=False
):
    frame = {
        **_identity_frame(),
        "periodic_fractional_axes": [0, 1],
    }
    substrate_symbols = ["In", "Sn"]
    # Real donor map: O1 -> full ID 4 (working Sn index 1), O3 -> full ID 2.
    substrate_positions = [[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]]
    working_ids = [2, 4]
    if add_nonmapped_collision:
        substrate_symbols.append("In")
        substrate_positions.append([0.0, 0.0, 3.0])
        working_ids.append(5)
    substrate = Atoms(
        substrate_symbols,
        positions=substrate_positions,
        cell=np.diag([20.0, 20.0, 30.0]),
        pbc=True,
    )
    source_coordinates = np.asarray(
        [
            [0.0, 0.0, 3.0],
            [-2.0, 0.0, donor_height_A],
            [2.0, 0.0, donor_height_A],
            [0.0, 0.0, 6.0],
        ]
    )
    symbols = np.asarray(["P", "O", "O", "C"])
    prototype = {
        "site_prototype_id": "site-prototype-phosphonic-acid-0001",
        "representative_candidate_id": "representative-001",
        "final_denticity": 2,
        "ideal_site_location": {"cartesian_A": [0.0, 0.0, 0.0]},
        "relaxed_adsorption_location": {
            "anchor_cartesian_A": source_coordinates[0].tolist(),
            "binding_oxygen_positions": [
                {
                    "probe_donor_id": "O1",
                    "cartesian_A": source_coordinates[1].tolist(),
                },
                {
                    "probe_donor_id": "O3",
                    "cartesian_A": source_coordinates[2].tolist(),
                },
            ],
        },
        "surface_frame": frame,
    }
    instance = {
        **_real_site_instance_schema_fixture(),
        "ideal_site_location": {"cartesian_A": [0.0, 0.0, 0.0]},
        "placement_template": {"surface_frame": frame},
    }
    conformer = {
        "source_rank": 1,
        "source": "worked-example.extxyz",
        "coordinates": source_coordinates,
        "symbols": symbols,
        "p_local": 0,
        "o_locals": [1, 2],
    }
    return {
        "substrate": substrate,
        "site_payload": {"site_instances": [instance]},
        "prototypes": {prototype["site_prototype_id"]: prototype},
        "conformers": [conformer],
        "metrics_by_rank": {1: {"cluster_id": 1}},
        "spec": MoleculeSpec(formula={"C": 1, "O": 2, "P": 1}),
        "headgroup_rmsd_max_A": 1.0e-8,
        "periodic_axes": (0, 1),
        "collision_thresholds_A": {
            "H-H": 1.5,
            "H-heavy": 1.8,
            "heavy-heavy": 2.2,
        },
        "full_to_working_atom_map": build_full_to_working_atom_map(
            working_layer_atom_ids_1based=working_ids,
            working_symbols=substrate.get_chemical_symbols(),
        ),
        "mapped_bond_window_A": (1.7, 2.8),
        "substrate_vdw_radius_scale": 0.85,
    }


def _build_interface_domains(mode, **fixture_options):
    return _candidate_domains(
        **_strict_interface_candidate_fixture(**fixture_options),
        substrate_collision_mode=mode,
    )


def _rotated_axis_02_interface_fixture():
    fixture = _strict_interface_candidate_fixture()

    def rotate(values):
        values = np.asarray(values, dtype=float)
        return np.column_stack((values[:, 0], -values[:, 2], values[:, 1]))

    fixture["substrate"].positions = rotate(fixture["substrate"].positions)
    fixture["substrate"].set_cell(np.diag([20.0, 30.0, 20.0]))
    fixture["periodic_axes"] = (0, 2)
    frame = _axis_02_frame()
    instance = fixture["site_payload"]["site_instances"][0]
    instance["placement_template"]["surface_frame"] = copy.deepcopy(frame)
    instance["ideal_site_location"]["cartesian_A"] = rotate(
        [instance["ideal_site_location"]["cartesian_A"]]
    )[0].tolist()
    prototype = next(iter(fixture["prototypes"].values()))
    prototype["surface_frame"] = copy.deepcopy(frame)
    prototype["ideal_site_location"]["cartesian_A"] = rotate(
        [prototype["ideal_site_location"]["cartesian_A"]]
    )[0].tolist()
    relaxed = prototype["relaxed_adsorption_location"]
    relaxed["anchor_cartesian_A"] = rotate([relaxed["anchor_cartesian_A"]])[0].tolist()
    for donor in relaxed["binding_oxygen_positions"]:
        donor["cartesian_A"] = rotate([donor["cartesian_A"]])[0].tolist()
    fixture["conformers"][0]["coordinates"] = rotate(
        fixture["conformers"][0]["coordinates"]
    )
    return fixture












































def _site_payload_for(domains):
    instances = []
    for site_id in sorted(domains):
        candidates = domains[site_id]
        ideal = list(candidates[0]["anchor_cartesian_A"])
        instances.append(
            {
                "site_instance_id": site_id,
                "parent_site_prototype_id": candidates[0]["site_prototype_id"],
                "ideal_site_location": {"cartesian_A": ideal},
            }
        )
    return {"site_instances": instances}


def _shared_prototype_domains(site_order=("site-a", "site-b"), prototype_id="proto-shared"):
    frame = _identity_frame()
    domains = {}
    for site in site_order:
        domains[site] = [
            _make_candidate(
                site, 1, [1, 2, 3],
                [[0.0, 0.0, 5.0], [1.0, 0.0, 4.0], [0.0, 1.0, 4.0], [-1.0, 0.0, 4.0]],
                ["C", "O", "O", "O"], frame, prototype_id=prototype_id,
            ),
            _make_candidate(
                site, 2, [3, 1, 2],
                [[0.5, 0.5, 5.0], [1.0, 0.5, 4.0], [0.5, 1.5, 4.0], [-0.5, 0.5, 4.0]],
                ["C", "O", "O", "O"], frame, prototype_id=prototype_id,
            ),
        ]
    return domains


def _cache_call(domains, tmp_path, site_payload=None, **overrides):
    kwargs = {
        "domains": domains,
        "site_payload": site_payload or _site_payload_for(domains),
        "cell": np.diag([20.0, 20.0, 30.0]),
        "periodic_axes": (0, 1),
        "cache_dir": tmp_path,
        "boundary_samples_per_atom": 72,
        "radius_scale": 1.0,
    }
    kwargs.update(overrides)
    return _load_or_build_projection_cache(**kwargs)








def _assert_rebuild(rebuilt, first_manifest):
    assert rebuilt.manifest["hit"] is False
    assert rebuilt.manifest["rebuild_reason"]
    assert rebuilt.manifest["contract"] != first_manifest["contract"]












def _unconflicted_domain_fixture():
    cell = np.diag([20.0, 20.0, 30.0])
    domains = {
        site: [
            {
                "site_instance_id": site,
                "site_prototype_id": "prototype-1",
                "selection_rank": rank,
                "cluster_id": rank,
                "source": f"rank-{rank}",
                "oxygen_permutation": perm,
                "headgroup_rmsd_A": 0.1,
                "coordinates": np.asarray([[position, 0.0, 5.0]]),
                "symbols": np.asarray(["C"]),
            }
            for rank, perm in ((1, [1, 2, 3]), (2, [3, 2, 1]))
        ]
        for site, position in (("site-a", 0.0), ("site-b", 5.0), ("site-c", 10.0))
    }
    return domains, cell


def _steps_without_timing(steps):
    return [
        {
            key: value
            for key, value in step.items()
            if key != "static_capacity_seconds"
        }
        for step in steps
    ]


def _shape_record(source_key, theta, area=10.0, perimeter=14.0):
    return {
        "source_key": source_key,
        "cluster_id": int(source_key.split("-")[0]),
        "source_task_index": int(source_key.split("-")[1]),
        "relative_total_energy_eV": float(source_key.split("-")[1]),
        "shape_metrics": {
            "area_A2": area,
            "perimeter_A": perimeter,
            "compactness": 4.0 * np.pi * area / perimeter**2,
            "anisotropy": 0.4,
            "aspect_ratio": 1.5,
            "principal_orientation_cos2": float(np.cos(2.0 * theta)),
            "principal_orientation_sin2": float(np.sin(2.0 * theta)),
        },
    }










def _fake_cache(
    masks,
    geometries,
    site_ids,
    area_scale_A2=100.0,
    surface_lattice_uv_A=None,
):
    if surface_lattice_uv_A is None:
        side = float(np.sqrt(area_scale_A2))
        surface_lattice_uv_A = np.diag([side, side])
    return SimpleNamespace(
        coverage_masks=masks,
        candidate_geometries=geometries,
        area_scale_A2=area_scale_A2,
        surface_lattice_uv_A=np.asarray(surface_lattice_uv_A, dtype=float),
        site_ids=site_ids,
    )


def _policy_candidate(candidate_id, index, site, rank):
    return {
        "candidate_id": candidate_id,
        "candidate_index": index,
        "site_instance_id": site,
        "selection_rank": rank,
        "oxygen_permutation": [1, 2, 3],
    }


def _choose_projection(
    candidates,
    *,
    cache,
    conflicts=None,
    feasible_mask=0b111,
    accepted_mask=0,
    accepted_union=None,
    weights=None,
    seed=7,
    criterion="uncovered-anchors",
    area_tolerance_A2=1.0e-6,
    perimeter_tolerance_A=1.0e-6,
):
    from monolayer_sequential_growth import choose_projection_candidate

    return choose_projection_candidate(
        candidates=candidates,
        site_conflicts=conflicts or {},
        feasible_site_mask=feasible_mask,
        accepted_coverage_mask=accepted_mask,
        accepted_union=accepted_union,
        area_scale_A2=cache.area_scale_A2,
        conformer_weights=weights or {1: 1.0},
        rng=random.Random(seed),
        area_tie_tolerance_A2=area_tolerance_A2,
        perimeter_tie_tolerance_A=perimeter_tolerance_A,
        projection_cache=cache,
        selection_criterion=criterion,
    )


def test_cohesive_perimeter_condensation_gain_formula_and_validation():
    assert cohesive_perimeter_condensation_gain(10.0, 4.0, 8.0) == pytest.approx(1.5)
    assert cohesive_perimeter_condensation_gain(10.0, 4.0, 14.0) == pytest.approx(0.0)
    assert cohesive_perimeter_condensation_gain(10.0, 4.0, 15.0) == pytest.approx(-0.25)
    with pytest.raises(ValueError, match="positive"):
        cohesive_perimeter_condensation_gain(10.0, 0.0, 8.0)
    with pytest.raises(TypeError, match="not bool"):
        cohesive_perimeter_condensation_gain(True, 4.0, 8.0)


def test_cohesive_perimeter_filter_uses_10_percent_default_and_supports_override():
    candidates = [
        {
            "candidate_id": "best",
            "increment_A": 1.0,
            "candidate_perimeter_A": 10.0,
            "perimeter_condensation_gain": 0.9,
        },
        {
            "candidate_id": "large-does-not-set-scale",
            "increment_A": 3.000001,
            "candidate_perimeter_A": 100.0,
            "perimeter_condensation_gain": 0.97,
        },
        {
            "candidate_id": "boundary_10pct",
            "increment_A": 2.0,
            "candidate_perimeter_A": 20.0,
            "perimeter_condensation_gain": 0.85,
        },
        {
            "candidate_id": "middle_10pct",
            "increment_A": 1.5,
            "candidate_perimeter_A": 30.0,
            "perimeter_condensation_gain": 0.93,
        },
        {
            "candidate_id": "legacy_20pct_only",
            "increment_A": 2.8,
            "candidate_perimeter_A": 20.0,
            "perimeter_condensation_gain": 0.80,
        },
    ]
    # Formal default: 10% tolerance (allowed excess = 10.0 * 0.10 = 1.0; cutoff = 2.0)
    finalists, audit = cohesive_filter_perimeter_increment(candidates)
    assert [record["candidate_id"] for record in finalists] == [
        "best",
        "boundary_10pct",
        "middle_10pct",
    ]
    assert audit["relative_tolerance"] == pytest.approx(0.10)
    assert audit["minimum_increment_A"] == pytest.approx(1.0)
    assert audit["minimum_candidate_perimeter_A"] == pytest.approx(10.0)
    assert audit["allowed_increment_excess_A"] == pytest.approx(1.0)
    assert audit["cutoff_increment_A"] == pytest.approx(2.0)
    assert audit["comparison_operator"] == "<="

    # Explicit override: 20% tolerance (allowed excess = 10.0 * 0.20 = 2.0; cutoff = 3.0)
    finalists_20, audit_20 = cohesive_filter_perimeter_increment(
        candidates, relative_tolerance=0.20
    )
    assert [record["candidate_id"] for record in finalists_20] == [
        "best",
        "boundary_10pct",
        "legacy_20pct_only",
        "middle_10pct",
    ]
    assert audit_20["relative_tolerance"] == pytest.approx(0.20)
    assert audit_20["allowed_increment_excess_A"] == pytest.approx(2.0)
    assert audit_20["cutoff_increment_A"] == pytest.approx(3.0)


@pytest.mark.parametrize(
    ("best_increment", "boundary_increment", "rejected_increment"),
    [(0.0, 1.0, 1.000001), (-2.0, -1.0, -0.999999)],
)
def test_cohesive_perimeter_filter_handles_zero_and_negative_best_increment(
    best_increment, boundary_increment, rejected_increment
):
    finalists, audit = cohesive_filter_perimeter_increment(
        [
            {
                "candidate_id": "best",
                "increment_A": best_increment,
                "candidate_perimeter_A": 10.0,
                "perimeter_condensation_gain": 1.0,
            },
            {
                "candidate_id": "boundary",
                "increment_A": boundary_increment,
                "candidate_perimeter_A": 20.0,
                "perimeter_condensation_gain": 0.8,
            },
            {
                "candidate_id": "rejected",
                "increment_A": rejected_increment,
                "candidate_perimeter_A": 30.0,
                "perimeter_condensation_gain": 0.7,
            },
        ]
    )
    assert [record["candidate_id"] for record in finalists] == [
        "best",
        "boundary",
    ]
    assert audit["cutoff_increment_A"] == pytest.approx(boundary_increment)
    assert audit["comparison_operator"] == "<="


def test_cohesive_perimeter_filter_rejects_invalid_perimeter_and_invariant():
    with pytest.raises(ValueError, match="positive"):
        cohesive_filter_perimeter_increment(
            [
                {
                    "candidate_id": "bad-perimeter",
                    "increment_A": 0.0,
                    "candidate_perimeter_A": 0.0,
                    "perimeter_condensation_gain": 0.0,
                }
            ]
        )
    with pytest.raises(ValueError, match="invariant violated"):
        cohesive_filter_perimeter_increment(
            [
                {
                    "candidate_id": "bad-gain",
                    "increment_A": 1.0,
                    "candidate_perimeter_A": 10.0,
                    "perimeter_condensation_gain": -2.0e-9,
                }
            ]
        )


@pytest.mark.parametrize(
    ("maximum", "boundary", "rejected", "expected_deficit"),
    [(0, 0, None, 1), (8, 6, 5, 2), (20, 16, 15, 4)],
)
def test_cohesive_side_contact_integer_tolerance(
    maximum, boundary, rejected, expected_deficit
):
    records = [
        {"candidate_id": "maximum", "side_contact_count": maximum},
        {"candidate_id": "boundary", "side_contact_count": boundary},
    ]
    if rejected is not None:
        records.append({"candidate_id": "rejected", "side_contact_count": rejected})
    finalists, audit = cohesive_filter_side_contacts(records)
    assert "maximum" in {record["candidate_id"] for record in finalists}
    assert "boundary" in {record["candidate_id"] for record in finalists}
    assert "rejected" not in {record["candidate_id"] for record in finalists}
    assert audit["allowed_count_deficit"] == expected_deficit


def test_cohesive_filter_distinct_neighbors_exact_maximum_and_validation():
    records = [
        {"candidate_id": "two_neighbors", "distinct_neighbor_count": 2},
        {"candidate_id": "three_neighbors_a", "distinct_neighbor_count": 3},
        {"candidate_id": "three_neighbors_b", "distinct_neighbor_count": 3},
        {"candidate_id": "one_neighbor", "distinct_neighbor_count": 1},
    ]
    finalists, audit = cohesive_filter_distinct_neighbors(records)
    assert [r["candidate_id"] for r in finalists] == [
        "three_neighbors_a",
        "three_neighbors_b",
    ]
    assert audit["maximum_distinct_neighbor_count"] == 3
    assert audit["cutoff_distinct_neighbor_count"] == 3
    assert audit["allowed_count_deficit"] == 0
    assert audit["retained_candidate_ids"] == [
        "three_neighbors_a",
        "three_neighbors_b",
    ]

    # Validation: negative or non-integer count raises error
    with pytest.raises(TypeError, match="must be an integer"):
        cohesive_filter_distinct_neighbors(
            [{"candidate_id": "bad", "distinct_neighbor_count": "two"}]
        )
    with pytest.raises(TypeError, match="must be an integer"):
        cohesive_filter_distinct_neighbors(
            [{"candidate_id": "bad", "distinct_neighbor_count": True}]
        )
    with pytest.raises(ValueError, match="must be non-negative"):
        cohesive_filter_distinct_neighbors(
            [{"candidate_id": "bad", "distinct_neighbor_count": -1}]
        )


def test_cohesive_lazy_contact_skips_perimeter_rejects():
    perimeter_finalists, _ = cohesive_filter_perimeter_increment(
        [
            {
                "candidate_id": "keep-a",
                "increment_A": 1.0,
                "candidate_perimeter_A": 10.0,
                "perimeter_condensation_gain": 0.9,
            },
            {
                "candidate_id": "keep-b",
                "increment_A": 2.0,
                "candidate_perimeter_A": 20.0,
                "perimeter_condensation_gain": 0.8,
            },
            {
                "candidate_id": "reject",
                "increment_A": 2.1,
                "candidate_perimeter_A": 30.0,
                "perimeter_condensation_gain": 0.7,
            },
        ]
    )
    called = []

    def contact_count(record):
        called.append(record["candidate_id"])
        return {"keep-a": 8, "keep-b": 6}[record["candidate_id"]]

    finalists, audit = cohesive_lazy_side_contacts(
        perimeter_finalists, contact_count
    )
    assert called == ["keep-a", "keep-b"]
    assert audit["evaluated_candidate_ids"] == ["keep-a", "keep-b"]
    assert {record["candidate_id"] for record in finalists} == {"keep-a", "keep-b"}


def test_cohesive_projected_area_uses_120_percent_boundary():
    finalists, audit = cohesive_filter_projected_area(
        [
            {"candidate_id": "minimum", "projected_area_A2": 50.0},
            {"candidate_id": "boundary", "projected_area_A2": 60.0},
            {"candidate_id": "above", "projected_area_A2": 60.0001},
        ]
    )
    assert [record["candidate_id"] for record in finalists] == [
        "boundary",
        "minimum",
    ]
    assert audit["cutoff_area_A2"] == pytest.approx(60.0)
    with pytest.raises(ValueError, match="positive"):
        cohesive_filter_projected_area(
            [{"candidate_id": "zero", "projected_area_A2": 0.0}]
        )


def test_cohesive_conformer_energy_uses_003_eV_boundary():
    finalists, audit = cohesive_filter_conformer_energy(
        [
            {"candidate_id": "minimum", "conformer_energy_eV": -2.50},
            {"candidate_id": "boundary", "conformer_energy_eV": -2.47},
            {"candidate_id": "above", "conformer_energy_eV": -2.4699},
        ]
    )
    assert [record["candidate_id"] for record in finalists] == [
        "boundary",
        "minimum",
    ]
    assert audit["cutoff_energy_eV"] == pytest.approx(-2.47)


def test_cohesive_filters_reject_duplicate_ids_and_nonfinite_values():
    with pytest.raises(ValueError, match="duplicate candidate_id"):
        cohesive_filter_projected_area(
            [
                {"candidate_id": "same", "projected_area_A2": 1.0},
                {"candidate_id": "same", "projected_area_A2": 2.0},
            ]
        )
    with pytest.raises(ValueError, match="finite"):
        cohesive_filter_conformer_energy(
            [{"candidate_id": "nan", "conformer_energy_eV": float("nan")}]
        )


def _contact_candidate(candidate_id, body_coordinate, *, head_coordinate=None):
    if head_coordinate is None:
        head_coordinate = (body_coordinate[0], body_coordinate[1], 0.0)
    return {
        "candidate_id": candidate_id,
        "coordinates": np.asarray([head_coordinate, body_coordinate], dtype=float),
        "symbols": np.asarray(["P", "C"]),
        "headgroup_local_indices_0based": [0],
    }


def _contact_report(new, accepted, **kwargs):
    return cohesive_side_contact_report(
        new,
        accepted,
        cell=np.diag([10.0, 10.0, 30.0]),
        periodic_axes=(0, 1),
        collision_thresholds_A={
            "H-H": 1.0,
            "H-heavy": 1.2,
            "heavy-heavy": 1.5,
        },
        **kwargs,
    )


def test_cohesive_side_contact_wraps_surface_axis_but_not_normal():
    accepted = _contact_candidate("accepted", (1.0, 5.0, 10.0))
    across_x = _contact_candidate("across-x", (9.0, 5.0, 10.0))
    report = _contact_report(across_x, [accepted])
    assert report["contact_count"] == 1
    assert report["periodic_axes"] == [0, 1]
    assert report["normal_wrapped"] is False

    across_z = _contact_candidate(
        "across-z", (1.0, 5.0, 29.0), head_coordinate=(1.0, 5.0, 29.0)
    )
    low_z = _contact_candidate("low-z", (1.0, 5.0, 1.0))
    assert _contact_report(across_z, [low_z])["has_contact"] is False


def test_cohesive_side_contact_excludes_headgroup_but_counts_body():
    accepted = {
        "candidate_id": "accepted",
        "coordinates": np.asarray([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]]),
        "symbols": np.asarray(["P", "C"]),
        "headgroup_local_indices_0based": [0],
    }
    head_only = {
        "candidate_id": "head-only",
        "coordinates": np.asarray([[2.0, 0.0, 0.0], [9.0, 9.0, 9.0]]),
        "symbols": np.asarray(["O", "C"]),
        "headgroup_local_indices_0based": [0],
    }
    assert _contact_report(head_only, [accepted])["contact_count"] == 0

    body_contact = dict(head_only)
    body_contact["candidate_id"] = "body-contact"
    body_contact["coordinates"] = np.asarray(
        [[2.0, 0.0, 0.0], [7.5, 5.0, 5.0]]
    )
    assert _contact_report(body_contact, [accepted])["contact_count"] == 1


def test_cohesive_side_contact_upper_boundary_is_inclusive():
    from ase.data import atomic_numbers, vdw_radii

    carbon_radius = float(vdw_radii[atomic_numbers["C"]])
    upper = 1.10 * 2.0 * carbon_radius
    accepted = _contact_candidate("accepted", (0.0, 0.0, 10.0))
    boundary = _contact_candidate("boundary", (upper, 0.0, 10.0))
    outside = _contact_candidate("outside", (upper + 1.0e-4, 0.0, 10.0))
    assert _contact_report(boundary, [accepted])["contact_count"] == 1
    assert _contact_report(outside, [accepted])["contact_count"] == 0


def test_cohesive_side_contact_rechecks_full_molecule_collision_gate():
    accepted = _contact_candidate("accepted", (5.0, 5.0, 10.0))
    collision = _contact_candidate("collision", (6.0, 5.0, 10.0))
    with pytest.raises(RuntimeError, match="collision gate"):
        _contact_report(collision, [accepted])


def test_cohesive_side_contact_early_stop_is_audited():
    new = _contact_candidate("new", (0.0, 0.0, 10.0))
    accepted_a = _contact_candidate("a", (2.0, 0.0, 10.0))
    accepted_b = _contact_candidate("b", (0.0, 2.0, 10.0))
    report = _contact_report(
        new, [accepted_b, accepted_a], stop_after_first=True
    )
    assert report["has_contact"] is True
    assert report["contact_count_exact"] is False
    assert report["distinct_contacted_neighbor_count_exact"] is False
    assert report["contacted_neighbor_count_exact"] is False
    assert report["distinct_contacted_neighbor_count"] == 1
    assert report["contacted_neighbor_ids"] == ["a"]
    assert report["evaluated_existing_candidate_ids"] == ["a"]


def test_cohesive_side_contact_report_distinct_neighbors_and_exactness():
    new = _contact_candidate("new", (0.0, 0.0, 10.0))
    accepted_a = _contact_candidate("a", (2.0, 0.0, 10.0))
    accepted_b = _contact_candidate("b", (0.0, 2.0, 10.0))
    accepted_far = _contact_candidate("far", (5.0, 5.0, 10.0))

    # Full evaluation (stop_after_first=False)
    report_full = _contact_report(
        new, [accepted_far, accepted_b, accepted_a], stop_after_first=False
    )
    assert report_full["has_contact"] is True
    assert report_full["contact_count"] == 2
    assert report_full["contact_count_exact"] is True
    assert report_full["contacted_neighbor_ids"] == ["a", "b"]
    assert report_full["distinct_contacted_neighbor_count"] == 2
    assert report_full["distinct_contacted_neighbor_count_exact"] is True
    assert report_full["contacted_neighbor_count"] == 2
    assert report_full["contacted_neighbor_count_exact"] is True
    assert report_full["contact_pair_counts_by_neighbor_id"] == {"a": 1, "b": 1}
    assert report_full["stopped_after_first"] is False
    assert report_full["evaluated_existing_candidate_ids"] == ["a", "b", "far"]

    # Early stop evaluation (stop_after_first=True)
    report_early = _contact_report(
        new, [accepted_far, accepted_b, accepted_a], stop_after_first=True
    )
    assert report_early["has_contact"] is True
    assert report_early["stopped_after_first"] is True
    assert report_early["contact_count"] == 1
    assert report_early["contact_count_exact"] is False
    assert report_early["distinct_contacted_neighbor_count"] == 1
    assert report_early["distinct_contacted_neighbor_count_exact"] is False
    assert report_early["contacted_neighbor_count_exact"] is False
    assert report_early["contacted_neighbor_ids"] == ["a"]
    assert report_early["evaluated_existing_candidate_ids"] == ["a"]

    # No contact evaluation (stop_after_first=True)
    report_no_contact = _contact_report(
        new, [accepted_far], stop_after_first=True
    )
    assert report_no_contact["has_contact"] is False
    assert report_no_contact["stopped_after_first"] is False
    assert report_no_contact["contact_count"] == 0
    assert report_no_contact["contact_count_exact"] is True
    assert report_no_contact["distinct_contacted_neighbor_count"] == 0
    assert report_no_contact["distinct_contacted_neighbor_count_exact"] is True
    assert report_no_contact["contacted_neighbor_ids"] == []


def test_cohesive_xy_prefilter_is_periodic_and_has_no_false_negative():
    frame = _identity_frame()
    accepted = _contact_candidate("accepted", (0.9, 5.0, 10.0))
    across_seam = _contact_candidate("across-seam", (9.1, 5.0, 10.0))
    far = _contact_candidate("far", (5.0, 5.0, 10.0))
    for candidate in (accepted, across_seam, far):
        candidate["anchor_cartesian_A"] = candidate["coordinates"][0].tolist()
        candidate["surface_frame"] = frame
    bounds = {
        candidate["candidate_id"]: cohesive_xy_bound(candidate)
        for candidate in (accepted, across_seam, far)
    }
    seam_report = cohesive_xy_may_contact(
        across_seam,
        [accepted],
        cell=np.diag([10.0, 10.0, 30.0]),
        periodic_axes=(0, 1),
        bounds_by_candidate_id=bounds,
    )
    assert seam_report["may_contact"] is True
    assert _contact_report(across_seam, [accepted])["has_contact"] is True
    assert cohesive_xy_may_contact(
        far,
        [accepted],
        cell=np.diag([10.0, 10.0, 30.0]),
        periodic_axes=(0, 1),
        bounds_by_candidate_id=bounds,
    )["may_contact"] is False


def test_cohesive_projection_index_finds_only_local_periodic_neighbors():
    import monolayer_sequential_growth as growth
    from shapely.ops import unary_union

    candidates = [
        _cohesive_policy_candidate("seam-left", 0, 3, 0.0),
        _cohesive_policy_candidate("seam-right", 1, 3, 0.0),
        _cohesive_policy_candidate("far", 2, 3, 0.0),
        _cohesive_policy_candidate("query", 3, 3, 0.0),
    ]
    cache = _fake_cache(
        masks={index: 0 for index in range(4)},
        geometries={
            0: shapely_box(0.00, 0.45, 0.04, 0.55),
            1: shapely_box(0.96, 0.45, 1.00, 0.55),
            2: shapely_box(0.45, 0.45, 0.55, 0.55),
            3: shapely_box(0.98, 0.46, 1.00, 0.54),
        },
        site_ids=[candidate["site_instance_id"] for candidate in candidates],
        surface_lattice_uv_A=np.diag([20.0, 20.0]),
    )
    index = CohesiveProjectionNeighborIndex(
        projection_cache=cache,
        candidates=candidates,
        collision_thresholds_A={"H-H": 1.5, "H-heavy": 1.8, "heavy-heavy": 2.2},
        contact_vdw_radius_scale=1.10,
    )
    index.add(candidates[0])
    index.add(candidates[1])
    index.add(candidates[2])
    neighbors, audit = index.query(candidates[3])
    assert [candidate["candidate_id"] for candidate in neighbors] == [
        "seam-left", "seam-right"
    ]
    assert audit["global_accepted_count"] == 3
    assert audit["local_neighbor_count"] == 2
    local_trial = index.perimeter_trial(candidates[3], neighbors)
    global_union = unary_union([cache.candidate_geometries[i] for i in (0, 1, 2)])
    before_A, _ = growth.periodic_union_increment_perimeter(
        None, global_union, cache.surface_lattice_uv_A
    )
    candidate_A, _ = growth.periodic_union_increment_perimeter(
        None, cache.candidate_geometries[3], cache.surface_lattice_uv_A
    )
    increment_A, _ = growth.periodic_union_increment_perimeter(
        global_union, cache.candidate_geometries[3], cache.surface_lattice_uv_A
    )
    global_gain = cohesive_perimeter_condensation_gain(
        before_A, candidate_A, before_A + increment_A
    )
    assert local_trial["perimeter_condensation_gain"] == pytest.approx(global_gain)


def test_cohesive_projection_index_oblique_shell_is_conservative():
    candidate = _cohesive_policy_candidate("accepted", 0, 3, 0.0)
    query = _cohesive_policy_candidate("query", 1, 3, 0.0)
    lattice = np.column_stack(([10.0, 0.0], [9.0, 2.0]))
    cache = _fake_cache(
        masks={0: 0, 1: 0},
        geometries={
            0: shapely_box(0.48, 0.48, 0.50, 0.50),
            1: shapely_box(0.50, 0.50, 0.52, 0.52),
        },
        site_ids=[candidate["site_instance_id"], query["site_instance_id"]],
        surface_lattice_uv_A=lattice,
    )
    index = CohesiveProjectionNeighborIndex(
        projection_cache=cache,
        candidates=[candidate, query],
        collision_thresholds_A={"H-H": 1.5, "H-heavy": 1.8, "heavy-heavy": 2.2},
        contact_vdw_radius_scale=1.10,
    )
    inverse = np.linalg.inv(lattice)
    assert index.fractional_shell_halfwidths == pytest.approx(
        index.physical_shell_A * np.linalg.norm(inverse, axis=1)
    )
    index.add(candidate)
    assert [item["candidate_id"] for item in index.query(query)[0]] == ["accepted"]


def test_cohesive_side_contact_rejects_malformed_headgroup_and_unknown_radius():
    accepted = _contact_candidate("accepted", (5.0, 5.0, 10.0))
    malformed = _contact_candidate("malformed", (7.5, 5.0, 10.0))
    malformed["headgroup_local_indices_0based"] = [0, 0]
    with pytest.raises(ValueError, match="unique"):
        _contact_report(malformed, [accepted])

    unknown = _contact_candidate("unknown", (7.5, 5.0, 10.0))
    unknown["symbols"] = np.asarray(["P", "Xx"])
    with pytest.raises(ValueError, match="van-der-Waals radius"):
        _contact_report(unknown, [accepted])


def _cohesive_policy_candidate(candidate_id, index, denticity, energy):
    coordinates = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
    return {
        "candidate_id": candidate_id,
        "candidate_index": index,
        "site_instance_id": f"site-{candidate_id}",
        "site_prototype_id": f"prototype-{candidate_id}",
        "selection_rank": 1,
        "cluster_id": 1,
        "source": "synthetic-cohesive-test",
        "target_donor_labels": [f"O{value}" for value in range(denticity)],
        "oxygen_permutation": list(range(1, denticity + 1)),
        "headgroup_rmsd_A": 0.0,
        "conformer_energy_eV": energy,
        "coordinates": coordinates,
        "symbols": np.asarray(["P", "C"]),
        "headgroup_local_indices_0based": [0],
        "anchor_cartesian_A": coordinates[0].tolist(),
        "surface_frame": _identity_frame(),
    }










def test_cohesive_active_denticity_is_strictly_3_then_2_then_1():
    candidates = [
        _cohesive_policy_candidate("mono", 0, 1, 0.0),
        _cohesive_policy_candidate("bi", 1, 2, 0.0),
        _cohesive_policy_candidate("tri-a", 2, 3, 0.0),
        _cohesive_policy_candidate("tri-b", 3, 3, 0.0),
    ]
    assert cohesive_active_denticity(candidates) == 3
    assert cohesive_active_denticity(candidates[:2]) == 2
    assert cohesive_active_denticity(candidates[:1]) == 1
    assert cohesive_active_denticity([]) is None


def test_cohesive_selector_nucleus_uses_candidate_area_then_energy():
    candidates = [
        _cohesive_policy_candidate("large-low-energy", 0, 3, 0.0),
        _cohesive_policy_candidate("small-high-energy", 1, 3, 0.10),
        _cohesive_policy_candidate("small-low-energy", 2, 3, 0.0),
    ]
    cache = _fake_cache(
        masks={0: 0, 1: 0, 2: 0},
        geometries={
            0: shapely_box(0.0, 0.0, 0.3, 0.3),
            1: shapely_box(0.0, 0.0, 0.2, 0.2),
            2: shapely_box(0.0, 0.0, 0.2, 0.2),
        },
        site_ids=[candidate["site_instance_id"] for candidate in candidates],
    )
    winner, audit = choose_cohesive_frontier_candidate(
        candidates=candidates,
        placed=[],
        accepted_union=None,
        projection_cache=cache,
        cell=np.diag([10.0, 10.0, 30.0]),
        periodic_axes=(0, 1),
        collision_thresholds_A={"H-H": 1.5, "H-heavy": 1.8, "heavy-heavy": 2.2},
        rng=random.Random(7),
    )
    assert winner["candidate_id"] == "small-low-energy"
    assert audit["growth_mode"] == "new-nucleus"
    assert audit["new_nucleus_event"] is True
    assert audit["capacity_evaluated"] is False
    assert audit["perimeter_filter"] is None
    assert audit["distinct_neighbor_filter"] is None
    assert audit["full_contact_filter"] is None
    assert audit["contact_pair_filter"] is None
    assert audit["area_filter"] is not None
    assert audit["energy_filter"] is not None


def test_cohesive_selector_nucleus_when_no_candidates_have_contact(monkeypatch):
    import monolayer_sequential_growth as growth

    candidates = [
        _cohesive_policy_candidate("large-low-energy", 0, 3, 0.0),
        _cohesive_policy_candidate("small-high-energy", 1, 3, 0.10),
        _cohesive_policy_candidate("small-low-energy", 2, 3, 0.0),
    ]
    cache = _fake_cache(
        masks={0: 0, 1: 0, 2: 0},
        geometries={
            0: shapely_box(0.0, 0.0, 0.3, 0.3),
            1: shapely_box(0.0, 0.0, 0.2, 0.2),
            2: shapely_box(0.0, 0.0, 0.2, 0.2),
        },
        site_ids=[candidate["site_instance_id"] for candidate in candidates],
    )
    placed = [_cohesive_policy_candidate("placed", 9, 3, 0.0)]

    def fake_no_contact(candidate, accepted, **kwargs):
        return {"has_contact": False, "contact_count": 0}

    monkeypatch.setattr(growth, "cohesive_side_contact_report", fake_no_contact)
    winner, audit = growth.choose_cohesive_frontier_candidate(
        candidates=candidates,
        placed=placed,
        accepted_union=shapely_box(0.8, 0.8, 0.9, 0.9),
        projection_cache=cache,
        cell=np.diag([10.0, 10.0, 30.0]),
        periodic_axes=(0, 1),
        collision_thresholds_A={"H-H": 1.5, "H-heavy": 1.8, "heavy-heavy": 2.2},
        rng=random.Random(7),
    )
    assert winner["candidate_id"] == "small-low-energy"
    assert audit["growth_mode"] == "new-nucleus"
    assert audit["new_nucleus_event"] is True
    assert audit["nucleus_reason"] == "no_exact_frontier_contact"
    assert audit["perimeter_filter"] is None
    assert audit["distinct_neighbor_filter"] is None
    assert audit["full_contact_filter"] is None
    assert audit["contact_pair_filter"] is None
    assert audit["area_filter"]["retained_candidate_ids"] == [
        "small-high-energy",
        "small-low-energy",
    ]


def test_cohesive_selector_filters_perimeter_before_full_contact(monkeypatch):
    import monolayer_sequential_growth as growth

    candidates = [
        _cohesive_policy_candidate("best", 0, 3, 0.0),
        _cohesive_policy_candidate("boundary", 1, 3, 0.0),
        _cohesive_policy_candidate("perimeter-reject", 2, 3, 0.0),
    ]
    cache = _fake_cache(
        masks={0: 0, 1: 0, 2: 0},
        geometries={
            0: shapely_box(0.25, 0.1, 0.45, 0.3),
            1: shapely_box(0.28, 0.1, 0.48, 0.3),
            2: shapely_box(0.31, 0.1, 0.51, 0.3),
        },
        site_ids=[candidate["site_instance_id"] for candidate in candidates],
    )
    placed = [_cohesive_policy_candidate("placed", 9, 3, 0.0)]
    early = []
    full = []

    def fake_contact(candidate, accepted, **kwargs):
        if kwargs["stop_after_first"]:
            early.append(candidate["candidate_id"])
            return {"has_contact": True, "contact_count": 1}
        full.append(candidate["candidate_id"])
        return {
            "has_contact": True,
            "contact_count": {"best": 10, "boundary": 8}[candidate["candidate_id"]],
        }

    monkeypatch.setattr(growth, "cohesive_side_contact_report", fake_contact)
    winner, audit = choose_cohesive_frontier_candidate(
        candidates=candidates,
        placed=placed,
        accepted_union=shapely_box(0.1, 0.1, 0.3, 0.3),
        projection_cache=cache,
        cell=np.diag([10.0, 10.0, 30.0]),
        periodic_axes=(0, 1),
        collision_thresholds_A={"H-H": 1.5, "H-heavy": 1.8, "heavy-heavy": 2.2},
        rng=random.Random(7),
    )
    assert winner["candidate_id"] in {"best", "boundary"}
    assert early == ["best", "boundary", "perimeter-reject"]
    assert full == ["best", "boundary"]
    assert audit["perimeter_filter"]["retained_candidate_ids"] == [
        "best",
        "boundary",
    ]
    assert audit["capacity_evaluated"] is False


def test_cohesive_selector_area_rejects_do_not_trigger_full_contact(monkeypatch):
    import monolayer_sequential_growth as growth

    candidates = [
        _cohesive_policy_candidate("best-perimeter-small-area", 0, 3, 0.0),
        _cohesive_policy_candidate("surviving-perimeter-large-area", 1, 3, 0.0),
        _cohesive_policy_candidate("surviving-perimeter-small-area", 2, 3, 0.0),
    ]
    # Geometries:
    # 0: area = (0.45 - 0.25) * (0.3 - 0.1) = 0.04
    # 1: area = (0.55 - 0.25) * (0.4 - 0.1) = 0.09 -> eliminated by 20% area tolerance (0.09 > 0.04 * 1.20 = 0.048)
    # 2: area = (0.46 - 0.26) * (0.3 - 0.1) = 0.04 -> survives area filter
    cache = _fake_cache(
        masks={0: 0, 1: 0, 2: 0},
        geometries={
            0: shapely_box(0.25, 0.1, 0.45, 0.3),
            1: shapely_box(0.25, 0.1, 0.55, 0.4),
            2: shapely_box(0.26, 0.1, 0.46, 0.3),
        },
        site_ids=[candidate["site_instance_id"] for candidate in candidates],
    )
    placed = [_cohesive_policy_candidate("placed", 9, 3, 0.0)]
    early = []
    full = []

    def fake_contact(candidate, accepted, **kwargs):
        if kwargs["stop_after_first"]:
            early.append(candidate["candidate_id"])
            return {"has_contact": True, "contact_count": 1}
        full.append(candidate["candidate_id"])
        return {
            "has_contact": True,
            "contact_count": 5,
            "distinct_contacted_neighbor_count": 1,
            "contacted_neighbor_ids": ["placed"],
        }

    monkeypatch.setattr(growth, "cohesive_side_contact_report", fake_contact)
    winner, audit = growth.choose_cohesive_frontier_candidate(
        candidates=candidates,
        placed=placed,
        accepted_union=shapely_box(0.1, 0.1, 0.3, 0.3),
        projection_cache=cache,
        cell=np.diag([10.0, 10.0, 30.0]),
        periodic_axes=(0, 1),
        collision_thresholds_A={"H-H": 1.5, "H-heavy": 1.8, "heavy-heavy": 2.2},
        rng=random.Random(42),
    )
    # Early contact is called for all 3 candidates to determine frontier
    assert sorted(early) == [
        "best-perimeter-small-area",
        "surviving-perimeter-large-area",
        "surviving-perimeter-small-area",
    ]
    # But full contact MUST NOT be called for the area-eliminated candidate!
    assert "surviving-perimeter-large-area" not in full
    assert sorted(full) == [
        "best-perimeter-small-area",
        "surviving-perimeter-small-area",
    ]
    assert audit["area_filter"]["retained_candidate_ids"] == [
        "best-perimeter-small-area",
        "surviving-perimeter-small-area",
    ]


def test_cohesive_selector_neighbor_count_takes_precedence_over_pair_count(monkeypatch):
    import monolayer_sequential_growth as growth

    candidates = [
        _cohesive_policy_candidate("many-neighbors-few-pairs", 0, 3, 0.0),
        _cohesive_policy_candidate("one-neighbor-many-pairs", 1, 3, 0.0),
    ]
    cache = _fake_cache(
        masks={0: 0, 1: 0},
        geometries={
            0: shapely_box(0.25, 0.1, 0.45, 0.3),
            1: shapely_box(0.25, 0.1, 0.45, 0.3),
        },
        site_ids=[candidate["site_instance_id"] for candidate in candidates],
    )
    placed = [
        _cohesive_policy_candidate("placed-1", 8, 3, 0.0),
        _cohesive_policy_candidate("placed-2", 9, 3, 0.0),
    ]

    def fake_contact(candidate, accepted, **kwargs):
        if kwargs["stop_after_first"]:
            return {"has_contact": True, "contact_count": 1}
        cid = candidate["candidate_id"]
        if cid == "many-neighbors-few-pairs":
            return {
                "has_contact": True,
                "contact_count": 2,
                "distinct_contacted_neighbor_count": 2,
                "contacted_neighbor_ids": ["placed-1", "placed-2"],
                "contact_pair_counts_by_neighbor_id": {"placed-1": 1, "placed-2": 1},
            }
        else:
            return {
                "has_contact": True,
                "contact_count": 100,
                "distinct_contacted_neighbor_count": 1,
                "contacted_neighbor_ids": ["placed-1"],
                "contact_pair_counts_by_neighbor_id": {"placed-1": 100},
            }

    monkeypatch.setattr(growth, "cohesive_side_contact_report", fake_contact)
    winner, audit = growth.choose_cohesive_frontier_candidate(
        candidates=candidates,
        placed=placed,
        accepted_union=shapely_box(0.1, 0.1, 0.3, 0.3),
        projection_cache=cache,
        cell=np.diag([10.0, 10.0, 30.0]),
        periodic_axes=(0, 1),
        collision_thresholds_A={"H-H": 1.5, "H-heavy": 1.8, "heavy-heavy": 2.2},
        rng=random.Random(42),
    )
    # Distinct neighbor count is criterion 3a; it eliminates the 1-neighbor candidate before pair count (3b)
    assert winner["candidate_id"] == "many-neighbors-few-pairs"
    assert audit["distinct_neighbor_filter"]["maximum_distinct_neighbor_count"] == 2
    assert audit["distinct_neighbor_filter"]["retained_candidate_ids"] == [
        "many-neighbors-few-pairs"
    ]
    assert audit["full_contact_filter"]["retained_candidate_ids"] == [
        "many-neighbors-few-pairs"
    ]


def test_cohesive_selector_pair_count_effective_after_tied_neighbor_count(monkeypatch):
    import monolayer_sequential_growth as growth

    candidates = [
        _cohesive_policy_candidate("two-neighbors-many-pairs", 0, 3, 0.0),
        _cohesive_policy_candidate("two-neighbors-boundary-pairs", 1, 3, 0.0),
        _cohesive_policy_candidate("two-neighbors-few-pairs", 2, 3, 0.0),
    ]
    cache = _fake_cache(
        masks={0: 0, 1: 0, 2: 0},
        geometries={
            0: shapely_box(0.25, 0.1, 0.45, 0.3),
            1: shapely_box(0.25, 0.1, 0.45, 0.3),
            2: shapely_box(0.25, 0.1, 0.45, 0.3),
        },
        site_ids=[candidate["site_instance_id"] for candidate in candidates],
    )
    placed = [
        _cohesive_policy_candidate("placed-1", 8, 3, 0.0),
        _cohesive_policy_candidate("placed-2", 9, 3, 0.0),
    ]

    def fake_contact(candidate, accepted, **kwargs):
        if kwargs["stop_after_first"]:
            return {"has_contact": True, "contact_count": 1}
        cid = candidate["candidate_id"]
        # All three candidates have 2 distinct neighbors.
        # Max pair count = 20. Allowed deficit = max(1, ceil(0.20 * 20)) = 4. Cutoff = 16.
        # many-pairs: 20 pairs -> retained
        # boundary-pairs: 16 pairs -> retained
        # few-pairs: 10 pairs -> eliminated
        pairs_map = {
            "two-neighbors-many-pairs": 20,
            "two-neighbors-boundary-pairs": 16,
            "two-neighbors-few-pairs": 10,
        }
        return {
            "has_contact": True,
            "contact_count": pairs_map[cid],
            "distinct_contacted_neighbor_count": 2,
            "contacted_neighbor_ids": ["placed-1", "placed-2"],
        }

    monkeypatch.setattr(growth, "cohesive_side_contact_report", fake_contact)
    winner, audit = growth.choose_cohesive_frontier_candidate(
        candidates=candidates,
        placed=placed,
        accepted_union=shapely_box(0.1, 0.1, 0.3, 0.3),
        projection_cache=cache,
        cell=np.diag([10.0, 10.0, 30.0]),
        periodic_axes=(0, 1),
        collision_thresholds_A={"H-H": 1.5, "H-heavy": 1.8, "heavy-heavy": 2.2},
        rng=random.Random(42),
    )
    # All 3 survive distinct neighbor filter (3a)
    assert audit["distinct_neighbor_filter"]["retained_candidate_ids"] == [
        "two-neighbors-boundary-pairs",
        "two-neighbors-few-pairs",
        "two-neighbors-many-pairs",
    ]
    # In pair count filter (3b), 20% tolerance retains 20 and 16, eliminating 10
    assert audit["full_contact_filter"]["cutoff_contact_count"] == 16
    assert audit["full_contact_filter"]["retained_candidate_ids"] == [
        "two-neighbors-boundary-pairs",
        "two-neighbors-many-pairs",
    ]
    assert winner["candidate_id"] in {
        "two-neighbors-many-pairs",
        "two-neighbors-boundary-pairs",
    }


def test_cohesive_run_uses_strict_denticity_stages_and_bypasses_capacity():
    candidates = [
        _cohesive_policy_candidate("tri-central-o", 0, 3, 0.0),
        _cohesive_policy_candidate("tri-no-central-o", 1, 3, 0.0),
        _cohesive_policy_candidate("bi", 2, 2, 0.0),
        _cohesive_policy_candidate("mono", 3, 1, 0.0),
    ]
    for candidate, x in zip(candidates, (1.0, 5.0, 9.0, 13.0)):
        candidate["coordinates"] = np.asarray(
            [[x, 1.0, 0.0], [x, 1.0, 5.0]], dtype=float
        )
        candidate["anchor_cartesian_A"] = candidate["coordinates"][0].tolist()
        candidate["surface_frame"] = _identity_frame()
        candidate["conformer_sha256"] = f"sha-{candidate['candidate_id']}"
    domains = {
        candidate["site_instance_id"]: [candidate] for candidate in candidates
    }
    cache = _fake_cache(
        masks={index: 0 for index in range(4)},
        geometries={
            index: shapely_box(0.05 + 0.20 * index, 0.05, 0.15 + 0.20 * index, 0.15)
            for index in range(4)
        },
        site_ids=sorted(domains),
        area_scale_A2=400.0,
        surface_lattice_uv_A=np.diag([20.0, 20.0]),
    )
    result = run_sequential_domains(
        domains=domains,
        site_conflicts={},
        conformer_weights={1: 1.0},
        cell=np.diag([20.0, 20.0, 30.0]),
        periodic_axes=(0, 1),
        seed=7,
        selection_policy="cohesive-frontier",
        projection_cache=cache,
        collision_thresholds_A={
            "H-H": 1.5,
            "H-heavy": 1.8,
            "heavy-heavy": 2.2,
        },
    )
    assert [step["denticity_phase"] for step in result["steps"]] == [3, 3, 2, 1]
    assert all(step["static_capacity"] is None for step in result["steps"])
    assert all(
        step["capacity_not_computed_reason"]
        == "cohesive-frontier_has_no_future_capacity_objective"
        for step in result["steps"]
    )
    assert result["cohesive_frontier_summary"]["new_nucleus_event_count"] == 4












def _assign_candidate_indices(domains):
    all_ids = sorted(
        candidate["candidate_id"]
        for candidates in domains.values()
        for candidate in candidates
    )
    index_by_id = {candidate_id: index for index, candidate_id in enumerate(all_ids)}
    for candidates in domains.values():
        for candidate in candidates:
            candidate["candidate_index"] = index_by_id[candidate["candidate_id"]]
    return domains












def _capacity_audit_sink_fixture(site_count=3):
    frame = _identity_frame()
    cell = np.diag([40.0, 20.0, 30.0])
    domains = {}
    geometries = {}
    metal_constraints = {}
    for index in range(site_count):
        site_id = f"site-{index:03d}"
        candidate = _make_candidate(
            site_id,
            1,
            [1, 2, 3],
            [[3.0 + 10.0 * index, 5.0, 5.0]],
            ["C"],
            frame,
        )
        domains[site_id] = [candidate]
        metal_constraints[site_id] = {f"metal-{index:03d}"}
    _assign_candidate_indices(domains)
    for index, site_id in enumerate(sorted(domains)):
        candidate_index = domains[site_id][0]["candidate_index"]
        left = 0.05 + 0.20 * index
        geometries[candidate_index] = shapely_box(left, 0.20, left + 0.08, 0.30)
    cache = _fake_cache(
        masks={
            candidate["candidate_index"]: 0
            for candidates in domains.values()
            for candidate in candidates
        },
        geometries=geometries,
        site_ids=sorted(domains),
        area_scale_A2=800.0,
        surface_lattice_uv_A=np.diag([40.0, 20.0]),
    )
    return domains, cache, cell, metal_constraints


def _strict_json_loads(data):
    def reject_constant(value):
        raise ValueError(f"non-standard JSON constant {value}")

    return json.loads(data, parse_constant=reject_constant)










def _minimal_capacity_audit_document(step=1):
    return {
        "schema": CAPACITY_RANKING_AUDIT_SCHEMA,
        "schema_version": 1,
        "step": step,
        "capacity_audit": {},
    }
























def _bruteforce_capacity(site_ids, metal_constraints, site_conflicts):
    sites = sorted(site_ids)
    best = 0
    for count in range(len(sites) + 1):
        for selected in itertools.combinations(sites, count):
            selected_set = set(selected)
            metals = [
                metal
                for site_id in selected
                for metal in metal_constraints.get(site_id, set())
            ]
            if len(metals) != len(set(metals)):
                continue
            if any(
                right in selected_set
                for left in selected
                for right in site_conflicts.get(left, set())
            ):
                continue
            best = max(best, count)
    return best












def _capacity_record(
    candidate_id,
    *,
    total_debt,
    total_hole,
    maximum_hole,
    total_perimeter,
    rank=1,
):
    return {
        "candidate_id": candidate_id,
        "selection_rank": rank,
        "total_debt": total_debt,
        "total_hole_area_after_trial_A2": total_hole,
        "maximum_torus_hole_area_after_trial_A2": maximum_hole,
        "total_physical_hole_perimeter_after_trial_A": total_perimeter,
    }














def _frozen_value(value):
    if isinstance(value, np.ndarray):
        return ("ndarray", str(value.dtype), value.shape, value.tobytes())
    if isinstance(value, dict):
        return tuple((key, _frozen_value(item)) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return tuple(_frozen_value(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_frozen_value(item) for item in value))
    return value


def _transition_fixture():
    frame = _identity_frame()
    chosen = _make_candidate(
        "site-a", 1, [1, 2, 3], [[0.0, 0.0, 5.0]], ["C"], frame
    )
    domains = {
        "site-a": [chosen],
        "site-b": [
            _make_candidate(
                "site-b", 1, [1, 2, 3], [[4.0, 0.0, 5.0]], ["C"], frame
            ),
            _make_candidate(
                "site-b", 2, [3, 2, 1], [[1.0, 0.0, 5.0]], ["C"], frame
            ),
        ],
        # This 2.0 A heavy-heavy contact is safe under the legacy 1.8 A
        # default but must be pruned under this task's explicit 2.2 A contract.
        "site-c": [
            _make_candidate(
                "site-c", 1, [1, 2, 3], [[2.0, 0.0, 5.0]], ["C"], frame
            )
        ],
        "site-d": [
            _make_candidate(
                "site-d", 1, [1, 2, 3], [[6.0, 0.0, 5.0]], ["C"], frame
            )
        ],
        "site-e": [
            _make_candidate(
                "site-e", 1, [1, 2, 3], [[8.0, 0.0, 5.0]], ["C"], frame
            )
        ],
    }
    _assign_candidate_indices(domains)
    return domains, chosen




def _write_preflight_failure_fixture(tmp_path, *, output_dir="formal/output"):
    """Write valid cheap preflight inputs and one deliberately unreadable conformer."""

    frame = {
        **_identity_frame(),
        "periodic_fractional_axes": [0, 1],
    }
    substrate_path = tmp_path / "substrate.extxyz"
    ase_write(
        substrate_path,
        Atoms(
            ["O", "In", "Sn"],
            positions=[[0.0, 0.0, -5.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            cell=np.diag([20.0, 20.0, 30.0]),
            pbc=True,
        ),
    )
    (tmp_path / "layer-groups.json").write_text(
        json.dumps(
            {
                "normal_axis": 2,
                "restore_layer_atom_ids_1based": [1],
                "working_layer_atom_ids_1based": [2, 3],
            }
        )
    )
    prototype_id = "prototype-a"
    (tmp_path / "site-prototypes.json").write_text(
        json.dumps(
            {
                "site_prototypes": [
                    {
                        "site_prototype_id": prototype_id,
                        "representative_candidate_id": "representative-a",
                        "final_denticity": 2,
                        "ideal_site_location": {"cartesian_A": [0.0, 0.0, 0.0]},
                        "relaxed_adsorption_location": {
                            "anchor_cartesian_A": [0.0, 0.0, 3.4],
                            "binding_oxygen_positions": [
                                {
                                    "probe_donor_id": "O1",
                                    "cartesian_A": [-1.0, 0.0, 2.0],
                                },
                                {
                                    "probe_donor_id": "O2",
                                    "cartesian_A": [1.0, 0.0, 2.0],
                                },
                            ],
                        },
                        "surface_frame": frame,
                    }
                ]
            }
        )
    )
    (tmp_path / "site-instances.json").write_text(
        json.dumps(
            {
                "site_instances": [
                    {
                        "site_instance_id": "site-a",
                        "parent_site_prototype_id": prototype_id,
                        "final_denticity": 2,
                        "ideal_site_location": {"cartesian_A": [0.0, 0.0, 0.0]},
                        "placement_template": {"surface_frame": frame},
                        "actual_metal_atom_ids_1based": [2, 3],
                        "actual_metals": [
                            {
                                "atom_id_1based": 2,
                                "element": "In",
                                "actual_element_after_doping": "In",
                            },
                            {
                                "atom_id_1based": 3,
                                "element": "Sn",
                                "actual_element_after_doping": "Sn",
                            },
                        ],
                        "donor_metal_mapping": [
                            {
                                "probe_donor_id": "O1",
                                "metal_atom_id_1based": 2,
                                "parent_contact_distance_A": 2.0,
                            },
                            {
                                "probe_donor_id": "O2",
                                "metal_atom_id_1based": 3,
                                "parent_contact_distance_A": 2.0,
                            },
                        ],
                    }
                ],
                "conflict_edges": [],
            }
        )
    )
    conformer_path = tmp_path / "conformer-0001.extxyz"
    conformer_path.write_bytes(b"deliberately not an ASE structure\n")
    conformer_sha256 = hashlib.sha256(conformer_path.read_bytes()).hexdigest()
    (tmp_path / "analysis.json").write_text(
        json.dumps(
            {
                "selection": {
                    "selected": [
                        {
                            "selection_rank": 1,
                            "cluster_id": 1,
                            "relative_total_energy_eV": 0.0,
                            "footprint_area_A2": 50.0,
                            "structure": {
                                "path": conformer_path.name,
                                "sha256": conformer_sha256,
                            },
                        }
                    ]
                }
            }
        )
    )
    return [
        "--root",
        str(tmp_path),
        "--substrate",
        substrate_path.name,
        "--layer-groups",
        "layer-groups.json",
        "--site-instances",
        "site-instances.json",
        "--site-prototypes",
        "site-prototypes.json",
        "--conformer-glob",
        "conformer-*.extxyz",
        "--conformer-analysis",
        "analysis.json",
        "--output-dir",
        output_dir,
        "--molecule-formula-json",
        '{"C":1,"O":2,"P":1}',
        "--max-conformers",
        "1",
        "--conformer-cache",
        f"{output_dir}/internal-cache/conformers",
        "--projection-cache",
        f"{output_dir}/internal-cache/projection",
        "--substrate-collision",
        "skip",
    ]












def _write_public_subprocess_fixture(
    tmp_path, *, output_dir="public-output", vdw_radius_scale=0.85
):
    """Write a tiny real CPU-only H0 growth fixture for public subprocess tests."""

    frame = {
        **_identity_frame(),
        "periodic_fractional_axes": [0, 1],
    }
    substrate_path = tmp_path / "substrate.extxyz"
    ase_write(
        substrate_path,
        Atoms(
            ["O", "In", "Sn"],
            positions=[[0.0, 0.0, -5.0], [-1.4, 0.0, 0.0], [1.4, 0.0, 0.0]],
            cell=np.diag([20.0, 20.0, 30.0]),
            pbc=True,
        ),
    )
    (tmp_path / "layer-groups.json").write_text(
        json.dumps(
            {
                "normal_axis": 2,
                "restore_layer_atom_ids_1based": [1],
                "working_layer_atom_ids_1based": [2, 3],
            }
        )
    )
    prototype_id = "prototype-a"
    (tmp_path / "site-prototypes.json").write_text(
        json.dumps(
            {
                "site_prototypes": [
                    {
                        "site_prototype_id": prototype_id,
                        "representative_candidate_id": "representative-a",
                        "final_denticity": 2,
                        "ideal_site_location": {"cartesian_A": [0.0, 0.0, 0.0]},
                        "relaxed_adsorption_location": {
                            "anchor_cartesian_A": [0.0, 0.0, 3.4],
                            "binding_oxygen_positions": [
                                {
                                    "probe_donor_id": "O1",
                                    "cartesian_A": [-1.4, 0.0, 2.4],
                                },
                                {
                                    "probe_donor_id": "O2",
                                    "cartesian_A": [1.4, 0.0, 2.4],
                                },
                            ],
                        },
                        "surface_frame": frame,
                    }
                ]
            }
        )
    )
    (tmp_path / "site-instances.json").write_text(
        json.dumps(
            {
                "site_instances": [
                    {
                        "site_instance_id": "site-a",
                        "parent_site_prototype_id": prototype_id,
                        "final_denticity": 2,
                        "ideal_site_location": {"cartesian_A": [0.0, 0.0, 0.0]},
                        "placement_template": {"surface_frame": frame},
                        "actual_metal_atom_ids_1based": [2, 3],
                        "actual_metals": [
                            {
                                "atom_id_1based": 2,
                                "element": "In",
                                "actual_element_after_doping": "In",
                            },
                            {
                                "atom_id_1based": 3,
                                "element": "Sn",
                                "actual_element_after_doping": "Sn",
                            },
                        ],
                        "donor_metal_mapping": [
                            {
                                "probe_donor_id": "O1",
                                "metal_atom_id_1based": 2,
                                "parent_contact_distance_A": 2.4,
                            },
                            {
                                "probe_donor_id": "O2",
                                "metal_atom_id_1based": 3,
                                "parent_contact_distance_A": 2.4,
                            },
                        ],
                    }
                ],
                "conflict_edges": [],
            }
        )
    )
    conformer_path = tmp_path / "selection-0001.extxyz"
    ase_write(
        conformer_path,
        Atoms(
            ["P", "O", "O", "O", "C", "C", "C", "C", "C"],
            positions=[
                [0.0, 0.0, 0.0],
                [-1.4, 0.0, -1.0],
                [1.4, 0.0, -1.0],
                [0.0, 1.4, 0.0],
                [0.0, 0.0, 1.5],
                [0.0, 0.0, 2.9],
                [0.0, 0.0, 4.3],
                [0.0, 0.0, 5.7],
                [1.4, 0.0, 5.7],
            ],
            cell=np.diag([20.0, 20.0, 30.0]),
            pbc=True,
        ),
    )
    conformer_sha256 = hashlib.sha256(conformer_path.read_bytes()).hexdigest()
    (tmp_path / "analysis.json").write_text(
        json.dumps(
            {
                "selection": {
                    "selected": [
                        {
                            "selection_rank": 1,
                            "cluster_id": 1,
                            "relative_total_energy_eV": 0.0,
                            "footprint_area_A2": 50.0,
                            "structure": {
                                "path": conformer_path.name,
                                "sha256": conformer_sha256,
                            },
                        }
                    ]
                }
            }
        )
    )
    return [
        "--root",
        str(tmp_path),
        "--substrate",
        substrate_path.name,
        "--layer-groups",
        "layer-groups.json",
        "--site-instances",
        "site-instances.json",
        "--site-prototypes",
        "site-prototypes.json",
        "--conformer-glob",
        "selection-*.extxyz",
        "--conformer-analysis",
        "analysis.json",
        "--output-dir",
        output_dir,
        "--molecule-formula-json",
        '{"C":5,"O":3,"P":1}',
        "--max-conformers",
        "1",
        "--conformer-cache",
        f"{output_dir}/internal-cache/conformers",
        "--projection-cache",
        f"{output_dir}/internal-cache/projection",
        "--selection-policy",
        "projection-aware",
        "--footprint-boundary-samples",
        "12",
        "--substrate-collision",
        "skip",
        "--substrate-vdw-radius-scale",
        str(vdw_radius_scale),
        "--independent-validation",
        "strict-interface",
    ]


def _run_public_builder_subprocess(argv):
    builder = Path(__file__).resolve().parents[1] / "scripts" / "monolayer_builder.py"
    return subprocess.run(
        [sys.executable, str(builder), "--method", "sequential-growth", *argv],
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )


def _assert_recursive_artifact_references_resolve(payload, *, base):
    references = []

    def visit(value):
        if isinstance(value, dict):
            if {"path", "sha256"}.issubset(value):
                references.append(value)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payload)
    assert references
    for reference in references:
        artifact = base / reference["path"]
        assert artifact.is_file(), reference
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == reference["sha256"]












def _replace_cli_argument(argv, option, value):
    updated = list(argv)
    updated[updated.index(option) + 1] = str(value)
    return updated
































def _immutable_conformer_cache_call(tmp_path, cache_prefix, **overrides):
    import monolayer_headgroup_fit as headgroup_fit

    _write_public_subprocess_fixture(tmp_path)
    kwargs = {
        "root": tmp_path,
        "conformer_glob": "selection-*.extxyz",
        "cache_prefix": cache_prefix,
        "spec": MoleculeSpec(formula={"C": 5, "O": 3, "P": 1}),
        "minimum_vertical_angle": 0.0,
        "maximum_marker_exposure": float("inf"),
        "maximum_source_clusters": 1,
        "immutable_new_cache": True,
    }
    kwargs.update(overrides)
    return headgroup_fit.source_conformers(**kwargs)
















def _two_source_loaded_ensemble_fixture(tmp_path):
    import monolayer_sequential_growth as growth

    _write_public_subprocess_fixture(tmp_path)
    first = tmp_path / "selection-0001.extxyz"
    second = tmp_path / "selection-0002.extxyz"
    second.write_bytes(first.read_bytes())
    selected = []
    for rank, path in enumerate((first, second), 1):
        selected.append(
            {
                "selection_rank": rank,
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
        )
    lock = {"selected_source_files": selected}
    spec = MoleculeSpec(formula={"C": 5, "O": 3, "P": 1})
    conformers = growth.source_conformers(
        tmp_path,
        "selection-*.extxyz",
        tmp_path / "audit-cache" / "conformers",
        spec,
        minimum_vertical_angle=0.0,
        maximum_marker_exposure=float("inf"),
        maximum_source_clusters=2,
    )
    return growth, lock, spec, conformers


def test_loaded_conformer_ensemble_audits_unique_rank_source_symbols_and_coordinates(
    tmp_path
):
    growth, lock, spec, conformers = _two_source_loaded_ensemble_fixture(tmp_path)

    audit = growth._audit_loaded_conformer_ensemble(
        conformers, locked_selection=lock, spec=spec, root=tmp_path
    )

    assert [record["selection_rank"] for record in audit] == [1, 2]
    assert all(record["geometry_hashes_match"] for record in audit)
    assert all(
        record["loaded"]["geometry_sha256"]
        == record["direct_source"]["geometry_sha256"]
        for record in audit
    )


@pytest.mark.parametrize("mutation", ["duplicate-rank", "source", "coordinates", "symbols"])
def test_loaded_conformer_ensemble_corruption_fails_closed(tmp_path, mutation):
    growth, lock, spec, conformers = _two_source_loaded_ensemble_fixture(tmp_path)
    corrupted = copy.deepcopy(conformers)
    if mutation == "duplicate-rank":
        corrupted[1]["source_rank"] = 1
    elif mutation == "source":
        corrupted[1]["source"] = corrupted[0]["source"]
    elif mutation == "coordinates":
        corrupted[1]["coordinates"][0, 0] += 0.25
    else:
        corrupted[1]["symbols"][0] = "C"

    with pytest.raises(ValueError):
        growth._audit_loaded_conformer_ensemble(
            corrupted, locked_selection=lock, spec=spec, root=tmp_path
        )






def _independent_validation_fixture(tmp_path):
    frame = {
        **_identity_frame(),
        "periodic_fractional_axes": [0, 1],
    }
    substrate = Atoms(
        ["In", "Sn"],
        positions=[[-1.5, 0.0, 0.0], [1.5, 0.0, 0.0]],
        cell=np.diag([20.0, 20.0, 30.0]),
        pbc=True,
    )
    candidate = {
        "site_instance_id": "site-a",
        "site_prototype_id": "prototype-a",
        "symbols": np.asarray(["P", "O", "O", "C"]),
        "coordinates": np.asarray(
            [
                [0.0, 0.0, 3.3],
                [-1.5, 0.0, 2.3],
                [1.5, 0.0, 2.3],
                [0.0, 0.0, 4.8],
            ]
        ),
        "surface_frame": frame,
        "registered_interface_bonds": [
            {
                "donor_label": "O1",
                "molecule_atom_index_0based": 1,
                "molecule_element": "O",
                "full_metal_atom_id_1based": 2,
                "working_metal_atom_index_0based": 0,
                "actual_metal_element": "In",
                "distance_window_A": [1.7, 2.8],
            },
            {
                "donor_label": "O2",
                "molecule_atom_index_0based": 2,
                "molecule_element": "O",
                "full_metal_atom_id_1based": 3,
                "working_metal_atom_index_0based": 1,
                "actual_metal_element": "Sn",
                "distance_window_A": [1.7, 2.8],
            },
        ],
    }
    trajectory_dir = tmp_path / "trajectory-0001"
    trajectory_dir.mkdir()
    final_path = trajectory_dir / "final-h0.extxyz"
    assembled = _assemble(substrate, [candidate])
    ase_write(final_path, assembled)
    return substrate, candidate, assembled, trajectory_dir, final_path










def _required_cli_arguments():
    return [
        "--substrate", "substrate.cif",
        "--layer-groups", "layer-groups.json",
        "--site-instances", "site-instances.json",
        "--site-prototypes", "site-prototypes.json",
        "--conformer-glob", "selection-*.extxyz",
        "--conformer-analysis", "analysis.json",
        "--output-dir", "output",
        "--molecule-formula-json", '{"C":1}',
    ]






def test_cohesive_tolerance_resolver_precedence_and_overrides():
    # 1. Pure defaults: perimeter=0.10, contact=0.20, area=0.20
    resolved = resolve_cohesive_relative_tolerances()
    assert resolved == {"perimeter": 0.10, "contact": 0.20, "area": 0.20}

    # 2. Legacy umbrella alone sets all three
    resolved_umbrella = resolve_cohesive_relative_tolerances(relative_tolerance=0.15)
    assert resolved_umbrella == {"perimeter": 0.15, "contact": 0.15, "area": 0.15}

    # 3. Layer overrides take precedence over umbrella
    resolved_override = resolve_cohesive_relative_tolerances(
        perimeter=0.08,
        area=0.12,
        relative_tolerance=0.25,
    )
    assert resolved_override == {"perimeter": 0.08, "contact": 0.25, "area": 0.12}

    # 4. Layer override without umbrella leaves other layers at formal defaults
    resolved_layer_only = resolve_cohesive_relative_tolerances(perimeter=0.05)
    assert resolved_layer_only == {"perimeter": 0.05, "contact": 0.20, "area": 0.20}

    # 5. Validation errors
    for bad in (-0.01, 1.0, 1.5, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            resolve_cohesive_relative_tolerances(perimeter=bad)


def test_cohesive_umbrella_audit_tracking():
    # 1. Umbrella provided, no layers overridden -> all applied, used=True
    audit_all = resolve_cohesive_umbrella_audit(relative_tolerance=0.15)
    assert audit_all["legacy_umbrella_relative_tolerance"] == pytest.approx(0.15)
    assert audit_all["legacy_umbrella_provided"] is True
    assert audit_all["legacy_umbrella_applied_layers"] == ["perimeter", "contact", "area"]
    assert audit_all["legacy_umbrella_used"] is True

    # 2. Umbrella provided, some layers overridden -> only un-overridden applied, used=True
    audit_partial = resolve_cohesive_umbrella_audit(
        relative_tolerance=0.15,
        perimeter=0.08,
    )
    assert audit_partial["legacy_umbrella_provided"] is True
    assert audit_partial["legacy_umbrella_applied_layers"] == ["contact", "area"]
    assert audit_partial["legacy_umbrella_used"] is True

    # 3. Umbrella provided, but ALL THREE layers overridden -> applied=[], used=False
    audit_all_overridden = resolve_cohesive_umbrella_audit(
        relative_tolerance=0.15,
        perimeter=0.08,
        contact=0.18,
        area=0.22,
    )
    assert audit_all_overridden["legacy_umbrella_relative_tolerance"] == pytest.approx(0.15)
    assert audit_all_overridden["legacy_umbrella_provided"] is True
    assert audit_all_overridden["legacy_umbrella_applied_layers"] == []
    assert audit_all_overridden["legacy_umbrella_used"] is False

    # 4. Umbrella omitted -> provided=False, applied=[], used=False, tolerance=None
    audit_none = resolve_cohesive_umbrella_audit()
    assert audit_none["legacy_umbrella_relative_tolerance"] is None
    assert audit_none["legacy_umbrella_provided"] is False
    assert audit_none["legacy_umbrella_applied_layers"] == []
    assert audit_none["legacy_umbrella_used"] is False

    # 5. CLI args namespace test
    parser = _build_parser()
    args_all_overridden = parser.parse_args(
        _required_cli_arguments()
        + [
            "--cohesive-relative-tolerance", "0.25",
            "--cohesive-perimeter-relative-tolerance", "0.08",
            "--cohesive-contact-relative-tolerance", "0.15",
            "--cohesive-area-relative-tolerance", "0.18",
        ]
    )
    audit_cli = resolve_cohesive_umbrella_audit(args_all_overridden)
    assert audit_cli["legacy_umbrella_provided"] is True
    assert audit_cli["legacy_umbrella_applied_layers"] == []
    assert audit_cli["legacy_umbrella_used"] is False
    assert audit_cli["legacy_umbrella_relative_tolerance"] == pytest.approx(0.25)


















def _collision_cache_for(domains, cell, direct_query, candidate_count=None, domain_fp=None, contract_fp=None):
    return LazyCollisionCache(
        candidate_domain_fingerprint=(
            domain_fp
            or _candidate_domain_fingerprint(domains, cell=cell, periodic_axes=(0, 1))
        ),
        collision_contract_fingerprint=(
            contract_fp
            or _collision_contract_fingerprint(
                periodic_axes=(0, 1),
                thresholds_A={"H-H": 1.2, "H-heavy": 1.4, "heavy-heavy": 1.8},
                algorithm_version=_COLLISION_ALGORITHM_VERSION,
            )
        ),
        candidate_count=candidate_count or sum(
            len(candidates) for candidates in domains.values()
        ),
        direct_query=direct_query,
    )


































def _repair_candidate(site, candidate_id, x, *, rank=1, metal=1):
    return {
        "candidate_id": candidate_id,
        "candidate_index": -1,
        "site_instance_id": site,
        "site_prototype_id": f"prototype-{site}",
        "selection_rank": rank,
        "cluster_id": rank,
        "source": f"source-{rank}",
        "target_donor_labels": ["O1"],
        "oxygen_permutation": [0],
        "headgroup_rmsd_A": 0.01,
        "surface_frame": {**_identity_frame(), "periodic_fractional_axes": [0, 1]},
        "anchor_cartesian_A": [float(x), 0.0, 2.0],
        "coordinates": np.asarray([[float(x), 0.0, 2.0]]),
        "symbols": np.asarray(["C"]),
        "registered_interface_bonds": [],
        "actual_metal_ids_for_test": {metal},
    }


def _repair_projection(candidates, geometries, *, site_ids=None, cell_size=10.0):
    ordered = list(candidates)
    for index, candidate in enumerate(ordered):
        candidate["candidate_index"] = index
    site_ids = sorted(site_ids or {candidate["site_instance_id"] for candidate in ordered})
    lattice = np.asarray([[cell_size, 0.0], [0.0, cell_size]])
    return ProjectionCache(
        site_ids=site_ids,
        site_index={site: index for index, site in enumerate(site_ids)},
        candidate_ids=[candidate["candidate_id"] for candidate in ordered],
        coverage_masks={index: 0 for index in range(len(ordered))},
        candidate_geometries={
            candidate["candidate_index"]: geometry
            for candidate, geometry in zip(ordered, geometries)
        },
        template_records=[],
        area_scale_A2=float(abs(np.linalg.det(lattice))),
        surface_lattice_uv_A=lattice,
        contract={},
        manifest={},
    )
























def _evaluate_repair_fixture(*, removal_count):
    from shapely.geometry import box

    removed = [
        _repair_candidate(f"old-{index}", f"old-{index}", 2.0 + 4.0 * index, metal=10 + index)
        for index in range(removal_count)
    ]
    added = [
        _repair_candidate(f"new-{index}", f"new-{index}", 1.0 + 4.0 * index, metal=20 + index)
        for index in range(removal_count + 1)
    ]
    all_candidates = removed + added
    geometries = [
        box(0.10 + 0.12 * index, 0.10, 0.18 + 0.12 * index, 0.18)
        for index in range(removal_count)
    ] + [
        box(0.08 + 0.12 * index, 0.08, 0.19 + 0.12 * index, 0.19)
        for index in range(removal_count + 1)
    ]
    projection = _repair_projection(all_candidates, geometries)
    metal_constraints = {
        candidate["site_instance_id"]: set(candidate["actual_metal_ids_for_test"])
        for candidate in all_candidates
    }
    result = evaluate_local_repair_removal_set(
        placed=removed,
        removed_candidate_ids=[candidate["candidate_id"] for candidate in removed],
        local_candidates=added,
        projection_cache=projection,
        metal_constraints=metal_constraints,
        site_conflicts={},
        cell=np.diag([30.0, 30.0, 30.0]),
        periodic_axes=(0, 1),
        collision_thresholds_A={"H-H": 1.5, "H-heavy": 1.8, "heavy-heavy": 2.2},
        max_solutions=20,
        time_limit_seconds=10.0,
        area_tolerance_A2=1.0e-9,
    )
    return result, removed, added








def _sha_record(path):
    payload = Path(path).read_bytes()
    return {
        "path": str(Path(path).resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def _write_synthetic_parent(tmp_path, *, schema_version):
    parent = tmp_path / f"parent-schema-{schema_version}"
    trajectory = parent / "trajectory-0001"
    code_dir = parent / "code-snapshot"
    trajectory.mkdir(parents=True)
    code_dir.mkdir()
    input_records = {}
    for name in ("substrate", "layer_groups", "site_instances", "site_prototypes", "conformer_analysis"):
        path = tmp_path / f"{schema_version}-{name}.dat"
        path.write_text(f"{name}-evidence\n")
        input_records[name] = _sha_record(path)
    code_path = code_dir / "monolayer_sequential_growth.py"
    code_path.write_text("# immutable parent code\n")
    code_record = _sha_record(code_path)

    substrate = Atoms("In", positions=[[0.0, 0.0, 1.0]], cell=[10.0, 10.0, 20.0], pbc=[True, True, False])
    candidate = _repair_candidate("site-a", "candidate-a", 4.0, metal=1)
    candidate["coordinates"] = np.asarray([[4.0, 0.0, 2.0]])
    assembled = _assemble(substrate, [candidate])
    assembled.positions[:, 2] -= 1.0
    final_path = trajectory / "final-h0.extxyz"
    ase_write(final_path, assembled)
    final_record = _sha_record(final_path)
    final_record["path"] = "final-h0.extxyz" if schema_version == 4 else str(final_path.resolve())
    step = {
        "step": 1,
        "site_instance_id": "site-a",
        "site_prototype_id": "prototype-site-a",
        "selection_rank": 1,
        "cluster_id": 1,
        "source": "source-1",
        "target_donor_labels": ["O1"],
        "oxygen_permutation": [0],
        "headgroup_rmsd_A": 0.01,
    }
    if schema_version == 4:
        step["capacity_audit"] = {"winner_candidate_id": "candidate-a"}
    trajectory_manifest = {
        "schema_version": schema_version,
        "sealed": True if schema_version == 4 else None,
        "status": "passed_cpu_h0_sequential_growth_and_independent_validation" if schema_version == 4 else "jammed_no_feasible_candidate",
        "seed": 7,
        "selection_policy": "capacity-aware" if schema_version == 4 else "uniform",
        "emergent_molecule_count": 1,
        "molecule_count_N": 1 if schema_version == 4 else None,
        "periodic_fractional_axes": [0, 1] if schema_version == 4 else None,
        "final_structure": {k: v for k, v in final_record.items() if k != "bytes"} if schema_version == 2 else {"path": "final-h0.extxyz", "sha256": final_record["sha256"], "bytes": final_record["bytes"]},
        "steps": [step],
    }
    trajectory_manifest_path = trajectory / "manifest.json"
    trajectory_manifest_path.write_text(json.dumps(trajectory_manifest, indent=2) + "\n")
    trajectory_record = _sha_record(trajectory_manifest_path)

    if schema_version == 4:
        request_path = parent / "run-request.json"
        request_path.write_text(json.dumps({"primary_inputs": input_records}, indent=2) + "\n")
        request_record = _sha_record(request_path)
        top = {
            "schema": "sam-sequential-growth-run-manifest-v1",
            "schema_version": 4,
            "sealed": True,
            "status": "passed_cpu_h0_sequential_growth_and_independent_validation",
            "method": "irreversible_sequential_adsorption_without_target_coverage",
            "inputs": input_records,
            "run_request": {"path": "run-request.json", "sha256": request_record["sha256"], "bytes": request_record["bytes"]},
            "implementation": {"implementation": code_record, "dependencies": {}},
            "code_snapshot": {
                "path": "code-snapshot",
                "files": [{"module": code_path.name, "path": f"code-snapshot/{code_path.name}", "sha256": code_record["sha256"], "bytes": code_record["bytes"]}],
            },
            "working_substrate_atom_count": 1,
            "molecule_formula": {"C": 1},
            "periodic_fractional_axes": [0, 1],
            "normal_axis": 2,
            "substrate_collision_mode": "strict-vdw",
            "sam_sam_collision_contract": {"thresholds_A": {"H-H": 1.5, "H-heavy": 1.8, "heavy-heavy": 2.2}},
            "trajectories": [{
                "trajectory": 1,
                "seed": 7,
                "manifest": {"path": "trajectory-0001/manifest.json", "sha256": trajectory_record["sha256"], "bytes": trajectory_record["bytes"]},
                "final_structure": {"path": "trajectory-0001/final-h0.extxyz", "sha256": final_record["sha256"], "bytes": final_record["bytes"]},
            }],
        }
    else:
        legacy_code = tmp_path / "legacy-owner.py"
        legacy_code.write_bytes(code_path.read_bytes())
        legacy_code_record = _sha_record(legacy_code)
        top = {
            "schema_version": 2,
            "status": "passed_cpu_sequential_growth",
            "method": "irreversible_sequential_adsorption_without_target_coverage",
            "inputs": {**input_records, "implementation": legacy_code_record},
            "working_substrate_atom_count": 1,
            "molecule_formula": {"C": 1},
            "placement_policy": {
                "collision_thresholds_A": {"H-H": 1.5, "H-heavy": 1.8, "heavy-heavy": 2.2},
                "periodic_axes": [0, 1],
            },
            "trajectories": [{
                "trajectory": 1,
                "seed": 7,
                "manifest": str(trajectory_manifest_path.resolve()),
                "final_structure": str(final_path.resolve()),
            }],
        }
    (parent / "manifest.json").write_text(json.dumps(top, indent=2) + "\n")
    return parent, input_records, substrate, candidate


@pytest.fixture(scope="module")
def real_schema2_adapter_evidence(tmp_path_factory):
    import monolayer_sequential_growth as growth

    root = next(
        (
            candidate
            for candidate in Path(__file__).resolve().parents
            if (candidate / "runs/sequential-adsorption").is_dir()
        ),
        Path(__file__).resolve().parents[1],
    )
    parent_root = (
        root
        / "runs/sequential-adsorption/thr-nosub-aligned-1.5-1.8-2.2"
    )
    r0_root = (
        root
        / "runs/sequential-adsorption/"
        "dbf34-h0-corrected-reference-metrics-torus-v2-20260808"
    )
    top = json.loads((parent_root / "manifest.json").read_text())
    inputs = {
        name: _sha_record(Path(top["inputs"][name]["path"]))
        for name in (
            "substrate",
            "layer_groups",
            "site_instances",
            "site_prototypes",
            "conformer_analysis",
        )
    }
    parent = capture_immutable_parent_reference(
        parent_root, expected_inputs=inputs
    )
    selected = []
    for record in top["sampling_policy"]["table"]:
        selected.append(
            {
                "selection_rank": int(record["selection_rank"]),
                **_sha_record(Path(record["structure"]["path"])),
            }
        )
    selected_lock = {
        "conformer_glob": (
            "runs/monomer-conformation-optimization/"
            "fixed-site-mace-4403f5058979/03_analysis/selected/"
            "selection-*.extxyz"
        ),
        "selected_source_files": selected,
    }
    substrate, layer_groups = growth._load_working_substrate(
        Path(top["inputs"]["substrate"]["path"]),
        Path(top["inputs"]["layer_groups"]["path"]),
    )
    r0 = capture_reference_metrics(
        r0_root,
        parent_reference=parent,
        expected_inputs=inputs,
        selected_source_lock=selected_lock,
        repair_metric_contract={
            "footprint_boundary_samples": 720,
            "projection_radius_scale": 1.0,
            "area_tie_tolerance_A2": 1.0e-6,
            "perimeter_tie_tolerance_A": 1.0e-6,
        },
        current_substrate=substrate,
    )

    cache_root = tmp_path_factory.mktemp("schema2-readonly-adapter-cache")
    spec = MoleculeSpec(
        formula={key: int(value) for key, value in top["molecule_formula"].items()}
    )
    conformers = growth.source_conformers(
        root,
        selected_lock["conformer_glob"],
        cache_root / "conformers",
        spec,
        minimum_vertical_angle=0.0,
        maximum_marker_exposure=float("inf"),
        maximum_source_clusters=10,
    )
    site_payload = json.loads(
        Path(top["inputs"]["site_instances"]["path"]).read_text()
    )
    prototypes = growth._prototype_map(
        Path(top["inputs"]["site_prototypes"]["path"])
    )
    full_to_working = build_full_to_working_atom_map(
        working_layer_atom_ids_1based=layer_groups[
            "working_layer_atom_ids_1based"
        ],
        working_symbols=substrate.get_chemical_symbols(),
    )
    domains, diagnostics = _candidate_domains(
        substrate=substrate,
        site_payload=site_payload,
        prototypes=prototypes,
        conformers=conformers,
        metrics_by_rank={
            int(record["selection_rank"]): record
            for record in top["sampling_policy"]["table"]
        },
        spec=spec,
        headgroup_rmsd_max_A=0.35,
        full_to_working_atom_map=full_to_working,
        mapped_bond_window_A=(1.7, 2.8),
        substrate_vdw_radius_scale=0.85,
        periodic_axes=(0, 1),
        collision_thresholds_A={
            "H-H": 1.5,
            "H-heavy": 1.8,
            "heavy-heavy": 2.2,
        },
        substrate_collision_mode="skip",
    )
    return {
        "root": root,
        "parent_root": parent_root,
        "r0_root": r0_root,
        "top": top,
        "inputs": inputs,
        "parent": parent,
        "r0": r0,
        "selected_lock": selected_lock,
        "substrate": substrate,
        "domains": domains,
        "diagnostics": diagnostics,
    }




































def test_continuous_snap_preserves_nucleus_rule_for_first_molecule():
    import monolayer_sequential_growth as growth

    candidates = [
        _cohesive_policy_candidate("large-low-energy", 0, 3, 0.0),
        _cohesive_policy_candidate("small-high-energy", 1, 3, 0.10),
        _cohesive_policy_candidate("small-low-energy", 2, 3, 0.0),
    ]
    for cand in candidates:
        cand["template_key"] = "tpl-std"
        cand["anchor_uv_A"] = np.array([0.0, 0.0])
    cache = _fake_cache(
        masks={0: 0, 1: 0, 2: 0},
        geometries={
            0: shapely_box(0.0, 0.0, 0.3, 0.3),
            1: shapely_box(0.0, 0.0, 0.2, 0.2),
            2: shapely_box(0.0, 0.0, 0.2, 0.2),
        },
        site_ids=[candidate["site_instance_id"] for candidate in candidates],
    )
    cache.template_records = [
        {
            "template_key": "tpl-std",
            "relative_vertices_uv_A": [(-0.1, -0.1), (0.1, -0.1), (0.1, 0.1), (-0.1, 0.1)],
            "selection_rank": 1,
            "conformer_energy_eV": 0.0,
            "projected_area_A2": 0.04,
        }
    ]
    winner, audit = growth.choose_cohesive_frontier_candidate(
        candidates=candidates,
        placed=[],
        accepted_union=None,
        projection_cache=cache,
        cell=np.diag([10.0, 10.0, 30.0]),
        periodic_axes=(0, 1),
        collision_thresholds_A={"H-H": 1.5, "H-heavy": 1.8, "heavy-heavy": 2.2},
        rng=random.Random(7),
        cohesive_frontier_mode="continuous-snap",
        frontier_snap_radius_A=4.0,
    )
    assert winner["candidate_id"] == "small-low-energy"
    assert audit["growth_mode"] == "new-nucleus"
    assert audit["new_nucleus_event"] is True


def test_continuous_snap_executes_post_snap_tiered_filtering(monkeypatch):
    import monolayer_sequential_growth as growth

    candidates = [
        _cohesive_policy_candidate("best", 0, 3, 0.0),
        _cohesive_policy_candidate("boundary", 1, 3, 0.0),
        _cohesive_policy_candidate("perimeter-reject", 2, 3, 0.0),
    ]
    for cand in candidates:
        cand["template_key"] = "tpl-std"
        cand["anchor_uv_A"] = np.array([0.25, 0.1])
    cache = _fake_cache(
        masks={0: 0, 1: 0, 2: 0},
        geometries={
            0: shapely_box(0.25, 0.1, 0.45, 0.3),
            1: shapely_box(0.28, 0.1, 0.48, 0.3),
            2: shapely_box(0.31, 0.1, 0.51, 0.3),
        },
        site_ids=[candidate["site_instance_id"] for candidate in candidates],
    )
    cache.template_records = [
        {
            "template_key": "tpl-std",
            "relative_vertices_uv_A": [(-0.1, -0.1), (0.1, -0.1), (0.1, 0.1), (-0.1, 0.1)],
            "selection_rank": 1,
            "conformer_energy_eV": 0.0,
            "projected_area_A2": 0.04,
        }
    ]
    placed = [_cohesive_policy_candidate("placed", 9, 3, 0.0)]
    placed[0]["coordinates"] = np.array([[0.1, 0.1, 0.0], [0.1, 0.1, 5.0]])

    def fake_contact(candidate, accepted, **kwargs):
        return {
            "has_contact": True,
            "contact_count": {"best": 10, "boundary": 8, "perimeter-reject": 5}[candidate["candidate_id"]],
        }

    monkeypatch.setattr(growth, "cohesive_side_contact_report", fake_contact)
    winner, audit = growth.choose_cohesive_frontier_candidate(
        candidates=candidates,
        placed=placed,
        accepted_union=shapely_box(0.1, 0.1, 0.3, 0.3),
        projection_cache=cache,
        cell=np.diag([10.0, 10.0, 30.0]),
        periodic_axes=(0, 1),
        collision_thresholds_A={"H-H": 1.5, "H-heavy": 1.8, "heavy-heavy": 2.2},
        rng=random.Random(7),
        cohesive_frontier_mode="continuous-snap",
        frontier_snap_radius_A=4.0,
    )
    assert winner["candidate_id"] in {"best", "boundary"}
    assert audit["cohesive_frontier_mode"] == "continuous-snap"
    assert audit["snap_radius_A"] == 4.0
    assert "omission_audit" in audit


def test_continuous_snap_fallback_to_global_when_local_infeasible(monkeypatch):
    import monolayer_sequential_growth as growth

    candidates = [
        _cohesive_policy_candidate("cand-local", 0, 3, 0.0),
        _cohesive_policy_candidate("cand-global", 1, 3, 0.0),
    ]
    candidates[0]["template_key"] = "tpl-std"
    candidates[0]["anchor_uv_A"] = np.array([0.25, 0.1])
    candidates[1]["template_key"] = "tpl-std"
    candidates[1]["anchor_uv_A"] = np.array([8.0, 8.0])

    cache = _fake_cache(
        masks={0: 0, 1: 0},
        geometries={
            0: shapely_box(0.25, 0.1, 0.45, 0.3),
            1: shapely_box(0.80, 0.80, 0.82, 0.82),
        },
        site_ids=[candidate["site_instance_id"] for candidate in candidates],
    )
    cache.template_records = [
        {
            "template_key": "tpl-std",
            "relative_vertices_uv_A": [(-0.1, -0.1), (0.1, -0.1), (0.1, 0.1), (-0.1, 0.1)],
            "selection_rank": 1,
            "conformer_energy_eV": 0.0,
            "projected_area_A2": 0.04,
        }
    ]
    placed = [_cohesive_policy_candidate("placed", 9, 3, 0.0)]

    def fake_contact(candidate, accepted, **kwargs):
        has_contact = candidate["candidate_id"] == "cand-global"
        return {"has_contact": has_contact, "contact_count": 5 if has_contact else 0}

    monkeypatch.setattr(growth, "cohesive_side_contact_report", fake_contact)
    winner, audit = growth.choose_cohesive_frontier_candidate(
        candidates=candidates,
        placed=placed,
        accepted_union=shapely_box(0.1, 0.1, 0.3, 0.3),
        projection_cache=cache,
        cell=np.diag([10.0, 10.0, 30.0]),
        periodic_axes=(0, 1),
        collision_thresholds_A={"H-H": 1.5, "H-heavy": 1.8, "heavy-heavy": 2.2},
        rng=random.Random(7),
        cohesive_frontier_mode="continuous-snap",
        frontier_snap_radius_A=4.0,
    )
    assert winner["candidate_id"] == "cand-global"
    assert audit["fallback_to_global"] is True


def test_continuous_snap_cli_parser_defaults_and_explicit_options():
    import monolayer_sequential_growth as growth

    parser = growth._build_parser()
    defaults = parser.parse_args(_required_cli_arguments())
    assert defaults.cohesive_frontier_mode == "continuous-snap"
    assert defaults.headgroup_rmsd_max_A == pytest.approx(0.50)
    assert defaults.frontier_snap_radius_A == pytest.approx(4.0)

    action = [a for a in parser._actions if a.dest == "frontier_snap_radius_A"][0]
    assert "p90" in action.help and "3.35" in action.help
    assert "p95" in action.help and "3.84" in action.help
    assert "5.17" in action.help

    args = parser.parse_args(_required_cli_arguments() + [
        "--selection-policy", "cohesive-frontier",
        "--cohesive-frontier-mode", "continuous-snap",
        "--headgroup-rmsd-max-A", "0.50",
        "--frontier-snap-radius-A", "4.0",
    ])
    assert args.cohesive_frontier_mode == "continuous-snap"
    assert args.headgroup_rmsd_max_A == pytest.approx(0.50)
    assert args.frontier_snap_radius_A == pytest.approx(4.0)


def test_continuous_snap_derives_area_from_polygon_and_energy_from_candidates(monkeypatch):
    import monolayer_sequential_growth as growth

    candidates = [
        _cohesive_policy_candidate("cand-1", 0, 3, 0.02),
        _cohesive_policy_candidate("cand-2", 1, 3, 0.02),
    ]
    for cand in candidates:
        cand["template_key"] = "tpl-poly"
        cand["anchor_uv_A"] = np.array([0.25, 0.1])
    cache = _fake_cache(
        masks={0: 0, 1: 0},
        geometries={
            0: shapely_box(0.25, 0.1, 0.45, 0.3),
            1: shapely_box(0.28, 0.1, 0.48, 0.3),
        },
        site_ids=[c["site_instance_id"] for c in candidates],
    )
    # Real template record without projected_area_A2 or conformer_energy_eV
    cache.template_records = [
        {
            "template_key": "tpl-poly",
            "relative_vertices_uv_A": [(-0.1, -0.1), (0.1, -0.1), (0.1, 0.1), (-0.1, 0.1)],
            "selection_rank": 1,
        }
    ]
    placed = [_cohesive_policy_candidate("placed", 9, 3, 0.0)]

    def fake_contact(candidate, accepted, **kwargs):
        return {"has_contact": True, "contact_count": 5}

    monkeypatch.setattr(growth, "cohesive_side_contact_report", fake_contact)

    winner, audit = growth.choose_cohesive_frontier_candidate(
        candidates=candidates,
        placed=placed,
        accepted_union=shapely_box(0.1, 0.1, 0.3, 0.3),
        projection_cache=cache,
        cell=np.diag([10.0, 10.0, 30.0]),
        periodic_axes=(0, 1),
        collision_thresholds_A={"H-H": 1.5, "H-heavy": 1.8, "heavy-heavy": 2.2},
        rng=random.Random(42),
        cohesive_frontier_mode="continuous-snap",
        frontier_snap_radius_A=4.0,
    )
    assert winner["candidate_id"] in {"cand-1", "cand-2"}
    assert audit["cohesive_frontier_mode"] == "continuous-snap"

    # Inconsistent conformer energy for the same template must fail closed
    cand_inconsistent = [
        _cohesive_policy_candidate("c-a", 0, 3, 0.01),
        _cohesive_policy_candidate("c-b", 1, 3, 0.09),
    ]
    for c in cand_inconsistent:
        c["template_key"] = "tpl-poly"
        c["anchor_uv_A"] = np.array([0.25, 0.1])
    with pytest.raises(ValueError, match="Inconsistent conformer_energy_eV"):
        growth.choose_cohesive_frontier_candidate(
            candidates=cand_inconsistent,
            placed=placed,
            accepted_union=shapely_box(0.1, 0.1, 0.3, 0.3),
            projection_cache=cache,
            cell=np.diag([10.0, 10.0, 30.0]),
            periodic_axes=(0, 1),
            collision_thresholds_A={"H-H": 1.5, "H-heavy": 1.8, "heavy-heavy": 2.2},
            rng=random.Random(42),
            cohesive_frontier_mode="continuous-snap",
            frontier_snap_radius_A=4.0,
        )


def test_continuous_snap_omission_count_deduplicated_and_non_negative(monkeypatch):
    import monolayer_sequential_growth as growth

    candidates = [
        _cohesive_policy_candidate(f"cand-{i}", i, 3, 0.0)
        for i in range(5)
    ]
    for cand in candidates:
        cand["template_key"] = "tpl-std"
        cand["anchor_uv_A"] = np.array([0.25, 0.1])
    cache = _fake_cache(
        masks={i: 0 for i in range(5)},
        geometries={i: shapely_box(0.25, 0.1, 0.45, 0.3) for i in range(5)},
        site_ids=[c["site_instance_id"] for c in candidates],
    )
    cache.template_records = [
        {
            "template_key": "tpl-std",
            "relative_vertices_uv_A": [(-0.1, -0.1), (0.1, -0.1), (0.1, 0.1), (-0.1, 0.1)],
            "selection_rank": 1,
        }
    ]
    placed = [_cohesive_policy_candidate("placed", 9, 3, 0.0)]

    def fake_contact(candidate, accepted, **kwargs):
        return {"has_contact": True, "contact_count": 5}

    monkeypatch.setattr(growth, "cohesive_side_contact_report", fake_contact)

    winner, audit = growth.choose_cohesive_frontier_candidate(
        candidates=candidates,
        placed=placed,
        accepted_union=shapely_box(0.1, 0.1, 0.3, 0.3),
        projection_cache=cache,
        cell=np.diag([10.0, 10.0, 30.0]),
        periodic_axes=(0, 1),
        collision_thresholds_A={"H-H": 1.5, "H-heavy": 1.8, "heavy-heavy": 2.2},
        rng=random.Random(7),
        cohesive_frontier_mode="continuous-snap",
        frontier_snap_radius_A=4.0,
    )
    omission = audit["omission_audit"]
    assert omission["global_candidate_count"] == 5
    assert omission["focused_candidate_count"] <= 5
    assert omission["omitted_candidate_count"] >= 0
    assert omission["focused_candidate_count"] + omission["omitted_candidate_count"] == 5


def test_continuous_snap_audit_metadata_and_global_winner_comparison(monkeypatch):
    import monolayer_sequential_growth as growth

    candidates = [
        _cohesive_policy_candidate("best", 0, 3, 0.0),
        _cohesive_policy_candidate("suboptimal", 1, 3, 0.0),
    ]
    for cand in candidates:
        cand["template_key"] = "tpl-std"
        cand["anchor_uv_A"] = np.array([0.25, 0.1])
    cache = _fake_cache(
        masks={0: 0, 1: 0},
        geometries={
            0: shapely_box(0.25, 0.1, 0.45, 0.3),
            1: shapely_box(0.26, 0.1, 0.46, 0.3),
        },
        site_ids=[c["site_instance_id"] for c in candidates],
    )
    cache.template_records = [
        {
            "template_key": "tpl-std",
            "relative_vertices_uv_A": [(-0.1, -0.1), (0.1, -0.1), (0.1, 0.1), (-0.1, 0.1)],
            "selection_rank": 1,
        }
    ]
    placed = [_cohesive_policy_candidate("placed", 9, 3, 0.0)]

    def fake_contact(candidate, accepted, **kwargs):
        return {"has_contact": True, "contact_count": 5}

    monkeypatch.setattr(growth, "cohesive_side_contact_report", fake_contact)

    winner, audit = growth.choose_cohesive_frontier_candidate(
        candidates=candidates,
        placed=placed,
        accepted_union=shapely_box(0.1, 0.1, 0.3, 0.3),
        projection_cache=cache,
        cell=np.diag([10.0, 10.0, 30.0]),
        periodic_axes=(0, 1),
        collision_thresholds_A={"H-H": 1.5, "H-heavy": 1.8, "heavy-heavy": 2.2},
        rng=random.Random(7),
        cohesive_frontier_mode="continuous-snap",
        frontier_snap_radius_A=4.0,
    )
    assert audit["algorithm_scope"] == "frontier-ideal-position real-candidate focusing"
    assert audit["continuous_global_optimum_claimed"] is False
    assert audit["pre_snap_side_contact"] == "not_evaluated_no_3d_pose"
    assert audit["pre_snap_perimeter"] == "not_evaluated_virtual_geometry"
    assert audit["virtual_polygon_constructed"] is False
    assert audit["ideal_proposal_count"] > 0
    assert audit["snap_match_pair_count"] > 0
    assert audit["focused_frontier_candidate_count"] > 0

    omission = audit["omission_audit"]
    assert omission["benchmark_policy"] == "global_discrete_cohesive_frontier"
    assert "benchmark_semantics" in omission
    assert omission["pre_snap_perimeter"] == "not_evaluated_virtual_geometry"
    assert omission["virtual_polygon_constructed"] is False
    assert omission["ideal_proposal_count"] == audit["ideal_proposal_count"]
    assert omission["snap_match_pair_count"] == audit["snap_match_pair_count"]
    assert omission["focused_frontier_candidate_count"] == audit["focused_frontier_candidate_count"]
    assert omission["global_discrete_winner_candidate_id"] in {"best", "suboptimal"}
    assert isinstance(omission["global_discrete_winner_in_focused"], bool)
    assert isinstance(omission["delta_L_excess_A"], float)
    assert isinstance(omission["within_global_perimeter_band"], bool)
    assert isinstance(omission["global_min_delta_L_A"], float)


def test_continuous_snap_surface_lattice_fail_closed(monkeypatch):
    import monolayer_sequential_growth as growth

    candidates = [_cohesive_policy_candidate("cand-1", 0, 3, 0.0)]
    candidates[0]["template_key"] = "tpl-std"
    candidates[0]["anchor_uv_A"] = np.array([0.25, 0.1])
    cache = _fake_cache(
        masks={0: 0},
        geometries={0: shapely_box(0.25, 0.1, 0.45, 0.3)},
        site_ids=["site-cand-1"],
    )
    cache.template_records = [
        {
            "template_key": "tpl-std",
            "relative_vertices_uv_A": [(-0.1, -0.1), (0.1, -0.1), (0.1, 0.1), (-0.1, 0.1)],
            "selection_rank": 1,
        }
    ]
    placed = [_cohesive_policy_candidate("placed", 9, 3, 0.0)]

    def fake_contact(candidate, accepted, **kwargs):
        return {"has_contact": True, "contact_count": 5}

    monkeypatch.setattr(growth, "cohesive_side_contact_report", fake_contact)

    # Missing surface lattice
    cache.surface_lattice_uv_A = None
    with pytest.raises(KeyError, match="surface_lattice_uv_A"):
        growth.choose_cohesive_frontier_candidate(
            candidates=candidates,
            placed=placed,
            accepted_union=shapely_box(0.1, 0.1, 0.3, 0.3),
            projection_cache=cache,
            cell=np.diag([10.0, 10.0, 30.0]),
            periodic_axes=(0, 1),
            collision_thresholds_A={"H-H": 1.5, "H-heavy": 1.8, "heavy-heavy": 2.2},
            rng=random.Random(7),
            cohesive_frontier_mode="continuous-snap",
            frontier_snap_radius_A=4.0,
        )

    # Singular surface lattice (zero determinant)
    cache.surface_lattice_uv_A = np.zeros((2, 2))
    with pytest.raises(ValueError, match="invertible"):
        growth.choose_cohesive_frontier_candidate(
            candidates=candidates,
            placed=placed,
            accepted_union=shapely_box(0.1, 0.1, 0.3, 0.3),
            projection_cache=cache,
            cell=np.diag([10.0, 10.0, 30.0]),
            periodic_axes=(0, 1),
            collision_thresholds_A={"H-H": 1.5, "H-heavy": 1.8, "heavy-heavy": 2.2},
            rng=random.Random(7),
            cohesive_frontier_mode="continuous-snap",
            frontier_snap_radius_A=4.0,
        )








def test_continuous_snap_zero_virtual_perimeter_calls(monkeypatch):
    """(a) Monkeypatch proving new choose path calls virtual periodic perimeter 0 times for N ideal q*."""
    import monolayer_sequential_growth as growth

    candidates = [
        _cohesive_policy_candidate(f"cand-{i}", i, 3, 0.0)
        for i in range(4)
    ]
    for cand in candidates:
        cand["template_key"] = "tpl-std"
        cand["anchor_uv_A"] = np.array([2.5, 2.5])

    cache = _fake_cache(
        masks={i: 0 for i in range(4)},
        geometries={i: shapely_box(0.24, 0.24, 0.26, 0.26) for i in range(4)},
        site_ids=[c["site_instance_id"] for c in candidates],
    )
    cache.template_records = [
        {
            "template_key": "tpl-std",
            "relative_vertices_uv_A": [(-0.02, -0.02), (0.02, -0.02), (0.02, 0.02), (-0.02, 0.02)],
            "selection_rank": 1,
            "conformer_energy_eV": 0.0,
            "projected_area_A2": 0.0016,
        }
    ]
    placed = [_cohesive_policy_candidate("placed", 99, 3, 0.0)]
    accepted_union = shapely_box(0.2, 0.2, 0.3, 0.3)

    def fake_contact(candidate, accepted, **kwargs):
        return {"has_contact": True, "contact_count": 5}

    monkeypatch.setattr(growth, "cohesive_side_contact_report", fake_contact)

    virtual_perimeter_calls = []
    real_candidate_geometries = list(cache.candidate_geometries.values())
    orig_inc = growth.periodic_union_increment_perimeter

    def tracked_increment(union, cand_geom, lattice, **kwargs):
        if cand_geom is not None:
            is_expected = cand_geom.equals(accepted_union) or any(
                cand_geom.equals(cg) for cg in real_candidate_geometries
            )
            if not is_expected:
                virtual_perimeter_calls.append(cand_geom)
        return orig_inc(union, cand_geom, lattice, **kwargs)

    monkeypatch.setattr(growth, "periodic_union_increment_perimeter", tracked_increment)

    winner, audit = growth.choose_cohesive_frontier_candidate(
        candidates=candidates,
        placed=placed,
        accepted_union=accepted_union,
        projection_cache=cache,
        cell=np.diag([10.0, 10.0, 30.0]),
        periodic_axes=(0, 1),
        collision_thresholds_A={"H-H": 1.5, "H-heavy": 1.8, "heavy-heavy": 2.2},
        rng=random.Random(42),
        cohesive_frontier_mode="continuous-snap",
        frontier_snap_radius_A=4.0,
    )

    assert audit["ideal_proposal_count"] > 0
    assert audit["pre_snap_perimeter"] == "not_evaluated_virtual_geometry"
    assert audit["virtual_polygon_constructed"] is False
    assert len(virtual_perimeter_calls) == 0


def test_continuous_snap_synthetic_boundary_with_hole_recall_matches_brute_force():
    """(b) Synthetic periodic boundary + hole: vectorized/indexed recall matches brute-force reference ID set exactly."""
    import monolayer_sequential_growth as growth
    from shapely.geometry import Polygon

    surface_lattice = np.array([[12.0, 1.0], [0.0, 10.0]])
    lat_inv = np.linalg.inv(surface_lattice)

    exterior = [(1.0, 1.0), (11.0, 1.0), (11.0, 9.0), (1.0, 9.0), (1.0, 1.0)]
    hole = [(4.0, 4.0), (8.0, 4.0), (8.0, 7.0), (4.0, 7.0), (4.0, 4.0)]
    accepted_union = Polygon(exterior, [hole])

    templates = {
        "tpl-1": {
            "template_key": "tpl-1",
            "relative_vertices_uv_A": [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)],
        },
        "tpl-2": {
            "template_key": "tpl-2",
            "relative_vertices_uv_A": [(-0.3, -0.6), (0.3, -0.6), (0.0, 0.6)],
        },
    }

    ideal_proposals = growth.generate_continuous_frontier_ideal_positions(
        accepted_union=accepted_union,
        candidate_templates=templates,
        surface_lattice_uv_A=surface_lattice,
    )

    target_types = {p["target_type"] for p in ideal_proposals}
    assert "exterior" in target_types
    assert "hole" in target_types
    assert len(ideal_proposals) >= 16

    candidates = []
    cand_idx = 0
    for u in np.linspace(0.5, 11.5, 12):
        for v in np.linspace(0.5, 9.5, 10):
            tpl_k = f"tpl-{(cand_idx % 2) + 1}"
            candidates.append({
                "candidate_id": f"cand-{cand_idx}",
                "candidate_index": cand_idx,
                "template_key": tpl_k,
                "anchor_uv_A": np.array([u, v]),
            })
            cand_idx += 1

    snap_radius_A = 4.0
    ref_matches = 0
    ref_recalled_ids = set()
    for prop in ideal_proposals:
        p_q = prop["q_uv_A"]
        p_tpl = prop["template_key"]
        for cand in candidates:
            if cand["template_key"] != p_tpl:
                continue
            diff = cand["anchor_uv_A"] - p_q
            diff_frac = diff @ lat_inv
            mic_frac = diff_frac - np.round(diff_frac)
            dist = float(np.linalg.norm(mic_frac @ surface_lattice))
            if dist <= snap_radius_A + 1e-8:
                ref_matches += 1
                ref_recalled_ids.add(cand["candidate_id"])

    spatial_index = growth.SameTemplatePeriodicSpatialIndex(candidates, surface_lattice)
    snap_match_pair_count, focused_ids, distances, recalling_props = spatial_index.query_proposals(
        ideal_proposals, snap_radius_A=snap_radius_A
    )

    assert snap_match_pair_count == ref_matches
    assert focused_ids == ref_recalled_ids
    assert len(focused_ids) > 0


def test_continuous_snap_overlapping_proposals_single_focused_evaluation(monkeypatch):
    """(c) Two different q* recalling overlapping candidates are evaluated once; full focused set calls tier pipeline once."""
    import monolayer_sequential_growth as growth

    candidates = [
        _cohesive_policy_candidate("cand-A", 0, 3, 0.0),
        _cohesive_policy_candidate("cand-B", 1, 3, 0.0),
        _cohesive_policy_candidate("cand-C", 2, 3, 0.0),
    ]
    candidates[0]["template_key"] = "tpl-std"
    candidates[0]["anchor_uv_A"] = np.array([1.0, 1.0])
    candidates[1]["template_key"] = "tpl-std"
    candidates[1]["anchor_uv_A"] = np.array([2.0, 1.0])
    candidates[2]["template_key"] = "tpl-std"
    candidates[2]["anchor_uv_A"] = np.array([3.0, 1.0])

    cache = _fake_cache(
        masks={0: 0, 1: 0, 2: 0},
        geometries={
            0: shapely_box(0.10, 0.10, 0.12, 0.12),
            1: shapely_box(0.20, 0.10, 0.22, 0.12),
            2: shapely_box(0.30, 0.10, 0.32, 0.12),
        },
        site_ids=[c["site_instance_id"] for c in candidates],
    )
    cache.template_records = [
        {
            "template_key": "tpl-std",
            "relative_vertices_uv_A": [(-0.01, -0.01), (0.01, -0.01), (0.01, 0.01), (-0.01, 0.01)],
            "selection_rank": 1,
            "conformer_energy_eV": 0.0,
            "projected_area_A2": 0.0004,
        }
    ]
    placed = [_cohesive_policy_candidate("placed", 9, 3, 0.0)]

    def fake_contact(candidate, accepted, **kwargs):
        return {"has_contact": True, "contact_count": 5}

    monkeypatch.setattr(growth, "cohesive_side_contact_report", fake_contact)

    def fake_focus(accepted_union, candidate_templates, spatial_index, surface_lattice_uv_A, snap_radius_A=4.0):
        props = [
            {"q_uv_A": np.array([1.5, 1.0]), "template_key": "tpl-std", "target_type": "exterior", "hole_area_A2": 0.0},
            {"q_uv_A": np.array([2.5, 1.0]), "template_key": "tpl-std", "target_type": "exterior", "hole_area_A2": 0.0},
        ]
        match_count, focused_ids, dists, recalling = spatial_index.query_proposals(props, snap_radius_A=snap_radius_A)
        return len(props), match_count, focused_ids, dists, recalling

    monkeypatch.setattr(growth, "focus_continuous_frontier_candidates_streaming", fake_focus)

    eval_calls = []
    orig_eval = growth._evaluate_cohesive_candidates

    def tracked_eval(eval_cands, *args, **kwargs):
        eval_calls.append([c["candidate_id"] for c in eval_cands])
        return orig_eval(eval_cands, *args, **kwargs)

    monkeypatch.setattr(growth, "_evaluate_cohesive_candidates", tracked_eval)

    winner, audit = growth.choose_cohesive_frontier_candidate(
        candidates=candidates,
        placed=placed,
        accepted_union=shapely_box(0.05, 0.05, 0.08, 0.08),
        projection_cache=cache,
        cell=np.diag([10.0, 10.0, 30.0]),
        periodic_axes=(0, 1),
        collision_thresholds_A={"H-H": 1.5, "H-heavy": 1.8, "heavy-heavy": 2.2},
        rng=random.Random(7),
        cohesive_frontier_mode="continuous-snap",
        frontier_snap_radius_A=0.8,
    )

    assert len(eval_calls) == 2
    assert sorted(eval_calls[0]) == ["cand-A", "cand-B", "cand-C"]
    assert sorted(eval_calls[1]) == ["cand-A", "cand-B", "cand-C"]
    assert len(eval_calls[1]) == 3

    # With snap_radius 0.8: Prop 1 (1.5) recalls A (1.0) and B (2.0) -> 2 matches
    # Prop 2 (2.5) recalls B (2.0) and C (3.0) -> 2 matches
    # Total match pairs = 4; unique candidates = 3; evaluated once!
    assert audit["snap_match_pair_count"] == 4
    assert audit["focused_candidate_count"] == 3
    assert audit["focused_frontier_candidate_count"] == 3
    assert audit["omitted_candidate_count"] == 0


def test_continuous_snap_merging_real_candidates_prefers_better_delta_l_over_first_hit(monkeypatch):
    """(d) Old first-hit would choose A from first proposal; Scheme B merges real candidates and chooses B with better real delta L."""
    import monolayer_sequential_growth as growth

    candidates = [
        _cohesive_policy_candidate("cand-A", 0, 3, 0.0),
        _cohesive_policy_candidate("cand-B", 1, 3, 0.0),
    ]
    candidates[0]["template_key"] = "tpl-A"
    candidates[0]["anchor_uv_A"] = np.array([1.0, 1.0])
    candidates[1]["template_key"] = "tpl-B"
    candidates[1]["anchor_uv_A"] = np.array([3.0, 1.0])

    cache = _fake_cache(
        masks={0: 0, 1: 0},
        geometries={
            0: shapely_box(0.10, 0.10, 0.12, 0.12),
            1: shapely_box(0.30, 0.10, 0.32, 0.12),
        },
        site_ids=[c["site_instance_id"] for c in candidates],
    )
    cache.template_records = [
        {
            "template_key": "tpl-A",
            "relative_vertices_uv_A": [(-0.01, -0.01), (0.01, -0.01), (0.01, 0.01), (-0.01, 0.01)],
            "selection_rank": 1,
            "conformer_energy_eV": 0.0,
            "projected_area_A2": 0.0004,
        },
        {
            "template_key": "tpl-B",
            "relative_vertices_uv_A": [(-0.01, -0.01), (0.01, -0.01), (0.01, 0.01), (-0.01, 0.01)],
            "selection_rank": 1,
            "conformer_energy_eV": 0.0,
            "projected_area_A2": 0.0004,
        },
    ]
    placed = [_cohesive_policy_candidate("placed", 9, 3, 0.0)]

    def fake_contact(candidate, accepted, **kwargs):
        return {"has_contact": True, "contact_count": 5}

    monkeypatch.setattr(growth, "cohesive_side_contact_report", fake_contact)

    def fake_focus(accepted_union, candidate_templates, spatial_index, surface_lattice_uv_A, snap_radius_A=4.0):
        props = [
            {"q_uv_A": np.array([1.0, 1.0]), "template_key": "tpl-A", "target_type": "exterior", "hole_area_A2": 0.0},
            {"q_uv_A": np.array([3.0, 1.0]), "template_key": "tpl-B", "target_type": "exterior", "hole_area_A2": 0.0},
        ]
        match_count, focused_ids, dists, recalling = spatial_index.query_proposals(props, snap_radius_A=snap_radius_A)
        return len(props), match_count, focused_ids, dists, recalling

    monkeypatch.setattr(growth, "focus_continuous_frontier_candidates_streaming", fake_focus)

    def fake_increment(union, cand_geom, lattice, **kwargs):
        if union is None:
            return 4.0, None
        if cand_geom.equals(cache.candidate_geometries[0]):
            return 3.0, None  # cand-A has delta_L = 3.0 (gain = 0.25 >= 0)
        elif cand_geom.equals(cache.candidate_geometries[1]):
            return 1.0, None  # cand-B has delta_L = 1.0 (gain = 0.75 >= 0)
        return 4.0, None

    monkeypatch.setattr(growth, "periodic_union_increment_perimeter", fake_increment)

    winner, audit = growth.choose_cohesive_frontier_candidate(
        candidates=candidates,
        placed=placed,
        accepted_union=shapely_box(0.05, 0.05, 0.08, 0.08),
        projection_cache=cache,
        cell=np.diag([10.0, 10.0, 30.0]),
        periodic_axes=(0, 1),
        collision_thresholds_A={"H-H": 1.5, "H-heavy": 1.8, "heavy-heavy": 2.2},
        rng=random.Random(7),
        cohesive_frontier_mode="continuous-snap",
        frontier_snap_radius_A=1.0,
    )

    assert winner["candidate_id"] == "cand-B"
    assert audit["winner_candidate_id"] == "cand-B"
    assert audit["fallback_to_global"] is False


def test_continuous_snap_focused_frontier_empty_falls_back_to_global(monkeypatch):
    """(e) When focused frontier is empty, falls back to same-denticity global."""
    import monolayer_sequential_growth as growth

    candidates = [
        _cohesive_policy_candidate("cand-near-no-contact", 0, 3, 0.0),
        _cohesive_policy_candidate("cand-far-has-contact", 1, 3, 0.0),
    ]
    candidates[0]["template_key"] = "tpl-std"
    candidates[0]["anchor_uv_A"] = np.array([1.0, 1.0])
    candidates[1]["template_key"] = "tpl-std"
    candidates[1]["anchor_uv_A"] = np.array([8.0, 8.0])

    cache = _fake_cache(
        masks={0: 0, 1: 0},
        geometries={
            0: shapely_box(0.10, 0.10, 0.12, 0.12),
            1: shapely_box(0.79, 0.79, 0.81, 0.81),
        },
        site_ids=[c["site_instance_id"] for c in candidates],
    )
    cache.template_records = [
        {
            "template_key": "tpl-std",
            "relative_vertices_uv_A": [(-0.01, -0.01), (0.01, -0.01), (0.01, 0.01), (-0.01, 0.01)],
            "selection_rank": 1,
            "conformer_energy_eV": 0.0,
            "projected_area_A2": 0.0004,
        }
    ]
    placed = [_cohesive_policy_candidate("placed", 9, 3, 0.0)]

    def fake_contact(candidate, accepted, **kwargs):
        has_c = candidate["candidate_id"] == "cand-far-has-contact"
        return {"has_contact": has_c, "contact_count": 5 if has_c else 0}

    monkeypatch.setattr(growth, "cohesive_side_contact_report", fake_contact)

    def fake_focus(accepted_union, candidate_templates, spatial_index, surface_lattice_uv_A, snap_radius_A=4.0):
        props = [
            {"q_uv_A": np.array([1.0, 1.0]), "template_key": "tpl-std", "target_type": "exterior", "hole_area_A2": 0.0}
        ]
        match_count, focused_ids, dists, recalling = spatial_index.query_proposals(props, snap_radius_A=snap_radius_A)
        return len(props), match_count, focused_ids, dists, recalling

    monkeypatch.setattr(growth, "focus_continuous_frontier_candidates_streaming", fake_focus)

    winner, audit = growth.choose_cohesive_frontier_candidate(
        candidates=candidates,
        placed=placed,
        accepted_union=shapely_box(0.05, 0.05, 0.08, 0.08),
        projection_cache=cache,
        cell=np.diag([10.0, 10.0, 30.0]),
        periodic_axes=(0, 1),
        collision_thresholds_A={"H-H": 1.5, "H-heavy": 1.8, "heavy-heavy": 2.2},
        rng=random.Random(7),
        cohesive_frontier_mode="continuous-snap",
        frontier_snap_radius_A=4.0,
    )

    assert winner["candidate_id"] == "cand-far-has-contact"
    assert audit["fallback_to_global"] is True
    assert audit["omission_audit"]["fallback_to_global"] is True
    assert audit["focused_candidate_count"] == 1
    assert audit["focused_frontier_candidate_count"] == 0


def test_continuous_snap_rng_branch_determinism(monkeypatch):
    """(f) RNG branch determinism: frozen RNG state ensures independent benchmark and committed branch determinism."""
    import monolayer_sequential_growth as growth

    candidates = [
        _cohesive_policy_candidate(f"cand-{i}", i, 3, 0.0)
        for i in range(3)
    ]
    for cand in candidates:
        cand["template_key"] = "tpl-std"
        cand["anchor_uv_A"] = np.array([1.0, 1.0])

    cache = _fake_cache(
        masks={i: 0 for i in range(3)},
        geometries={i: shapely_box(0.10, 0.10, 0.12, 0.12) for i in range(3)},
        site_ids=[c["site_instance_id"] for c in candidates],
    )
    cache.template_records = [
        {
            "template_key": "tpl-std",
            "relative_vertices_uv_A": [(-0.01, -0.01), (0.01, -0.01), (0.01, 0.01), (-0.01, 0.01)],
            "selection_rank": 1,
            "conformer_energy_eV": 0.0,
            "projected_area_A2": 0.0004,
        }
    ]
    placed = [_cohesive_policy_candidate("placed", 9, 3, 0.0)]

    def fake_contact(candidate, accepted, **kwargs):
        return {"has_contact": True, "contact_count": 5}

    monkeypatch.setattr(growth, "cohesive_side_contact_report", fake_contact)

    rng1 = random.Random(12345)
    winner1, audit1 = growth.choose_cohesive_frontier_candidate(
        candidates=candidates,
        placed=placed,
        accepted_union=shapely_box(0.05, 0.05, 0.08, 0.08),
        projection_cache=cache,
        cell=np.diag([10.0, 10.0, 30.0]),
        periodic_axes=(0, 1),
        collision_thresholds_A={"H-H": 1.5, "H-heavy": 1.8, "heavy-heavy": 2.2},
        rng=rng1,
        cohesive_frontier_mode="continuous-snap",
        frontier_snap_radius_A=4.0,
    )

    rng2 = random.Random(12345)
    winner2, audit2 = growth.choose_cohesive_frontier_candidate(
        candidates=candidates,
        placed=placed,
        accepted_union=shapely_box(0.05, 0.05, 0.08, 0.08),
        projection_cache=cache,
        cell=np.diag([10.0, 10.0, 30.0]),
        periodic_axes=(0, 1),
        collision_thresholds_A={"H-H": 1.5, "H-heavy": 1.8, "heavy-heavy": 2.2},
        rng=rng2,
        cohesive_frontier_mode="continuous-snap",
        frontier_snap_radius_A=4.0,
    )

    assert winner1["candidate_id"] == winner2["candidate_id"]
    assert rng1.getstate() == rng2.getstate()


def test_continuous_snap_strict_denticity_and_4tier_pipeline_regression(monkeypatch):
    """(g) Strict denticity 3->2->1 and 4-tier lexicographic ordering (perimeter -> contact -> area -> energy -> seed)."""
    import monolayer_sequential_growth as growth

    candidates = [
        _cohesive_policy_candidate("cand-perimeter-best", 0, 3, 0.05),
        _cohesive_policy_candidate("cand-perimeter-worst", 1, 3, 0.00),
    ]
    candidates[0]["template_key"] = "tpl-A"
    candidates[0]["anchor_uv_A"] = np.array([1.0, 1.0])
    candidates[1]["template_key"] = "tpl-B"
    candidates[1]["anchor_uv_A"] = np.array([1.0, 1.0])

    cache = _fake_cache(
        masks={0: 0, 1: 0},
        geometries={
            0: shapely_box(0.10, 0.10, 0.12, 0.12),
            1: shapely_box(0.20, 0.10, 0.22, 0.12),
        },
        site_ids=[c["site_instance_id"] for c in candidates],
    )
    cache.template_records = [
        {
            "template_key": "tpl-A",
            "relative_vertices_uv_A": [(-0.01, -0.01), (0.01, -0.01), (0.01, 0.01), (-0.01, 0.01)],
            "selection_rank": 1,
            "conformer_energy_eV": 0.05,
            "projected_area_A2": 0.0004,
        },
        {
            "template_key": "tpl-B",
            "relative_vertices_uv_A": [(-0.01, -0.01), (0.01, -0.01), (0.01, 0.01), (-0.01, 0.01)],
            "selection_rank": 1,
            "conformer_energy_eV": 0.00,
            "projected_area_A2": 0.0004,
        },
    ]
    placed = [_cohesive_policy_candidate("placed", 9, 3, 0.0)]

    def fake_contact(candidate, accepted, **kwargs):
        return {"has_contact": True, "contact_count": 5}

    monkeypatch.setattr(growth, "cohesive_side_contact_report", fake_contact)

    def fake_increment(union, cand_geom, lattice, **kwargs):
        if union is None:
            return 4.0, None
        if cand_geom.equals(cache.candidate_geometries[0]):
            return 1.0, None
        return 3.0, None

    monkeypatch.setattr(growth, "periodic_union_increment_perimeter", fake_increment)

    winner, audit = growth.choose_cohesive_frontier_candidate(
        candidates=candidates,
        placed=placed,
        accepted_union=shapely_box(0.05, 0.05, 0.08, 0.08),
        projection_cache=cache,
        cell=np.diag([10.0, 10.0, 30.0]),
        periodic_axes=(0, 1),
        collision_thresholds_A={"H-H": 1.5, "H-heavy": 1.8, "heavy-heavy": 2.2},
        rng=random.Random(7),
        cohesive_frontier_mode="continuous-snap",
        frontier_snap_radius_A=4.0,
    )

    assert winner["candidate_id"] == "cand-perimeter-best"
    assert len(winner["target_donor_labels"]) == 3


def test_continuous_snap_lightweight_performance_count_no_virtual_torus_perimeter(monkeypatch):
    """(h) Lightweight performance count proving O(V*T) torus perimeter calls are no longer generated."""
    import monolayer_sequential_growth as growth

    # 4 boundary vertices and 4 templates = 16 (V*T) potential virtual evaluations
    candidates = [
        _cohesive_policy_candidate(f"cand-{i}", i, 3, 0.0)
        for i in range(2)
    ]
    for i, cand in enumerate(candidates):
        cand["template_key"] = f"tpl-{i}"
        cand["anchor_uv_A"] = np.array([2.0, 2.0])

    cache = _fake_cache(
        masks={i: 0 for i in range(2)},
        geometries={i: shapely_box(0.19, 0.19, 0.21, 0.21) for i in range(2)},
        site_ids=[c["site_instance_id"] for c in candidates],
    )
    cache.template_records = [
        {
            "template_key": f"tpl-{i}",
            "relative_vertices_uv_A": [(-0.01, -0.01), (0.01, -0.01), (0.01, 0.01), (-0.01, 0.01)],
            "selection_rank": 1,
            "conformer_energy_eV": 0.0,
            "projected_area_A2": 0.0004,
        }
        for i in range(4)
    ]
    placed = [_cohesive_policy_candidate("placed", 9, 3, 0.0)]
    accepted_union = shapely_box(0.1, 0.1, 0.3, 0.3)

    def fake_contact(candidate, accepted, **kwargs):
        return {"has_contact": True, "contact_count": 5}

    monkeypatch.setattr(growth, "cohesive_side_contact_report", fake_contact)

    perimeter_call_count = 0
    orig_inc = growth.periodic_union_increment_perimeter

    def counted_increment(*args, **kwargs):
        nonlocal perimeter_call_count
        perimeter_call_count += 1
        return orig_inc(*args, **kwargs)

    monkeypatch.setattr(growth, "periodic_union_increment_perimeter", counted_increment)

    winner, audit = growth.choose_cohesive_frontier_candidate(
        candidates=candidates,
        placed=placed,
        accepted_union=accepted_union,
        projection_cache=cache,
        cell=np.diag([10.0, 10.0, 30.0]),
        periodic_axes=(0, 1),
        collision_thresholds_A={"H-H": 1.5, "H-heavy": 1.8, "heavy-heavy": 2.2},
        rng=random.Random(7),
        cohesive_frontier_mode="continuous-snap",
        frontier_snap_radius_A=4.0,
    )

    # In Scheme B:
    # 0 calls during ideal position generation!
    # Real candidate evaluations:
    # Global benchmark (2 candidates) + focused set (2 candidates) = <= 10 calls total!
    assert perimeter_call_count <= 10
    assert audit["ideal_proposal_count"] >= 4
    assert audit["pre_snap_perimeter"] == "not_evaluated_virtual_geometry"
    assert audit["virtual_polygon_constructed"] is False


def test_continuous_snap_streaming_matches_materialized_reference():
    """Streaming focus exactly matches materialized reference across periodic boundary, hole, skew lattice, and multi-template."""
    import monolayer_sequential_growth as growth
    from shapely.geometry import Polygon

    # Skew non-orthogonal 2D lattice
    skew_lattice = np.array([
        [12.0, 3.5],
        [2.0, 10.0],
    ])

    # Footprint with exterior that crosses periodic boundaries and has an interior hole
    exterior_A = [
        (1.0, 1.0),
        (13.0, 1.0),
        (13.0, 11.0),
        (1.0, 11.0),
        (1.0, 1.0),
    ]
    hole_A = [
        (5.0, 4.0),
        (8.0, 4.0),
        (8.0, 7.0),
        (5.0, 7.0),
        (5.0, 4.0),
    ]
    accepted_union = Polygon(shell=exterior_A, holes=[hole_A])

    templates = {
        "tpl-rect": {
            "template_key": "tpl-rect",
            "relative_vertices_uv_A": [(-0.6, -0.4), (0.6, -0.4), (0.6, 0.4), (-0.6, 0.4)],
            "selection_rank": 1,
            "conformer_energy_eV": 0.01,
            "projected_area_A2": 0.96,
        },
        "tpl-tri": {
            "template_key": "tpl-tri",
            "relative_vertices_uv_A": [(-0.5, -0.5), (0.5, -0.5), (0.0, 0.7)],
            "selection_rank": 2,
            "conformer_energy_eV": 0.03,
            "projected_area_A2": 0.60,
        },
    }

    # Grid of candidate anchors covering the periodic cell
    candidates = []
    c_idx = 0
    for u in np.linspace(0.0, 12.0, 8):
        for v in np.linspace(0.0, 10.0, 8):
            tpl_k = "tpl-rect" if (c_idx % 2 == 0) else "tpl-tri"
            candidates.append({
                "candidate_id": f"cand-{c_idx}",
                "candidate_index": c_idx,
                "template_key": tpl_k,
                "anchor_uv_A": np.array([u, v]),
            })
            c_idx += 1

    spatial_index = growth.SameTemplatePeriodicSpatialIndex(candidates, skew_lattice)

    # 1. Materialized reference
    ref_props = growth.generate_continuous_frontier_ideal_positions(
        accepted_union=accepted_union,
        candidate_templates=templates,
        surface_lattice_uv_A=skew_lattice,
    )
    ref_pairs, ref_fids, ref_dists, ref_rec = spatial_index.query_proposals(
        ref_props, snap_radius_A=3.5
    )

    # 2. Streaming pure function
    s_ideal, s_pairs, s_fids, s_dists, s_rec = growth.focus_continuous_frontier_candidates_streaming(
        accepted_union=accepted_union,
        candidate_templates=templates,
        spatial_index=spatial_index,
        surface_lattice_uv_A=skew_lattice,
        snap_radius_A=3.5,
    )

    assert s_ideal == len(ref_props)
    assert s_pairs == ref_pairs
    assert s_fids == ref_fids
    assert len(s_dists) == len(ref_dists)
    for cid in s_dists:
        assert s_dists[cid] == pytest.approx(ref_dists[cid], abs=1e-12)
        assert s_rec[cid]["template_key"] == ref_rec[cid]["template_key"]
        assert s_rec[cid]["target_type"] == ref_rec[cid]["target_type"]
        assert np.allclose(s_rec[cid]["q_uv_A"], ref_rec[cid]["q_uv_A"])


def test_continuous_snap_streaming_equidistant_tie_determinism():
    """Equidistant candidate proposals produce deterministic tie-breaking matching reference."""
    import monolayer_sequential_growth as growth
    from shapely.geometry import Polygon

    lattice = np.diag([10.0, 10.0])

    # A square footprint centered at (5.0, 5.0) of size 2.0 x 2.0
    accepted_union = Polygon([(4.0, 4.0), (6.0, 4.0), (6.0, 6.0), (4.0, 6.0), (4.0, 4.0)])

    # Symmetric template
    templates = {
        "tpl-sym": {
            "template_key": "tpl-sym",
            "relative_vertices_uv_A": [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)],
            "selection_rank": 1,
            "conformer_energy_eV": 0.0,
            "projected_area_A2": 1.0,
        }
    }

    # Candidate placed symmetrically along center line
    candidate = {
        "candidate_id": "cand-center",
        "candidate_index": 0,
        "template_key": "tpl-sym",
        "anchor_uv_A": np.array([5.0, 5.0]),
    }

    spatial_index = growth.SameTemplatePeriodicSpatialIndex([candidate], lattice)

    # Reference
    ref_props = growth.generate_continuous_frontier_ideal_positions(
        accepted_union=accepted_union,
        candidate_templates=templates,
        surface_lattice_uv_A=lattice,
    )
    ref_pairs, ref_fids, ref_dists, ref_rec = spatial_index.query_proposals(
        ref_props, snap_radius_A=4.0
    )

    # Streaming run 1
    s_ideal1, s_pairs1, s_fids1, s_dists1, s_rec1 = growth.focus_continuous_frontier_candidates_streaming(
        accepted_union=accepted_union,
        candidate_templates=templates,
        spatial_index=spatial_index,
        surface_lattice_uv_A=lattice,
        snap_radius_A=4.0,
    )

    # Streaming run 2
    s_ideal2, s_pairs2, s_fids2, s_dists2, s_rec2 = growth.focus_continuous_frontier_candidates_streaming(
        accepted_union=accepted_union,
        candidate_templates=templates,
        spatial_index=spatial_index,
        surface_lattice_uv_A=lattice,
        snap_radius_A=4.0,
    )

    # Assert exact match with reference
    assert s_rec1["cand-center"]["template_key"] == ref_rec["cand-center"]["template_key"]
    assert np.allclose(s_rec1["cand-center"]["q_uv_A"], ref_rec["cand-center"]["q_uv_A"])
    assert s_dists1["cand-center"] == pytest.approx(ref_dists["cand-center"], abs=1e-12)

    # Assert determinism across runs
    assert np.allclose(s_rec1["cand-center"]["q_uv_A"], s_rec2["cand-center"]["q_uv_A"])


def test_continuous_snap_streaming_does_not_call_materialized_helper(monkeypatch):
    """Streaming function operates independently and does not call generate_continuous_frontier_ideal_positions."""
    import monolayer_sequential_growth as growth
    from shapely.geometry import box as shapely_box

    def boom(*args, **kwargs):
        raise RuntimeError("Materialized helper generate_continuous_frontier_ideal_positions should never be called by streaming!")

    monkeypatch.setattr(growth, "generate_continuous_frontier_ideal_positions", boom)

    lattice = np.diag([10.0, 10.0])
    accepted_union = shapely_box(2.0, 2.0, 4.0, 4.0)

    templates = {
        "tpl-test": {
            "template_key": "tpl-test",
            "relative_vertices_uv_A": [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)],
            "selection_rank": 1,
            "conformer_energy_eV": 0.0,
            "projected_area_A2": 1.0,
        }
    }
    candidates = [
        {
            "candidate_id": "c-1",
            "candidate_index": 0,
            "template_key": "tpl-test",
            "anchor_uv_A": np.array([2.5, 2.5]),
        }
    ]
    spatial_index = growth.SameTemplatePeriodicSpatialIndex(candidates, lattice)

    # Must execute and return valid results without calling the monkeypatched function
    s_ideal, s_pairs, s_fids, s_dists, s_rec = growth.focus_continuous_frontier_candidates_streaming(
        accepted_union=accepted_union,
        candidate_templates=templates,
        spatial_index=spatial_index,
        surface_lattice_uv_A=lattice,
        snap_radius_A=4.0,
    )

    assert s_ideal > 0
    assert s_pairs > 0
    assert "c-1" in s_fids


def test_continuous_snap_production_wiring_uses_streaming_without_materialized_helpers(monkeypatch):
    """Production choose path calls focus_continuous_frontier_candidates_streaming exactly once and does not call materialized helpers."""
    import monolayer_sequential_growth as growth
    from shapely.geometry import box as shapely_box

    # 1. Monkeypatch generate_continuous_frontier_ideal_positions to raise
    def raise_ideal(*args, **kwargs):
        raise AssertionError("generate_continuous_frontier_ideal_positions should not be called by production choose!")
    monkeypatch.setattr(growth, "generate_continuous_frontier_ideal_positions", raise_ideal)

    # 2. Monkeypatch spatial_index.query_proposals to raise
    def raise_query(*args, **kwargs):
        raise AssertionError("spatial_index.query_proposals should not be called by production choose!")
    monkeypatch.setattr(growth.SameTemplatePeriodicSpatialIndex, "query_proposals", raise_query)

    # 3. Track calls to focus_continuous_frontier_candidates_streaming
    streaming_call_count = 0
    orig_streaming = growth.focus_continuous_frontier_candidates_streaming
    def tracked_streaming(*args, **kwargs):
        nonlocal streaming_call_count
        streaming_call_count += 1
        return orig_streaming(*args, **kwargs)
    monkeypatch.setattr(growth, "focus_continuous_frontier_candidates_streaming", tracked_streaming)

    candidates = [
        _cohesive_policy_candidate("cand-A", 0, 3, 0.0),
        _cohesive_policy_candidate("cand-B", 1, 3, 0.0),
    ]
    candidates[0]["template_key"] = "tpl-std"
    candidates[0]["anchor_uv_A"] = np.array([1.0, 1.0])
    candidates[1]["template_key"] = "tpl-std"
    candidates[1]["anchor_uv_A"] = np.array([2.0, 1.0])

    cache = _fake_cache(
        masks={0: 0, 1: 0},
        geometries={
            0: shapely_box(0.10, 0.10, 0.12, 0.12),
            1: shapely_box(0.20, 0.10, 0.22, 0.12),
        },
        site_ids=[c["site_instance_id"] for c in candidates],
    )
    cache.template_records = [
        {
            "template_key": "tpl-std",
            "relative_vertices_uv_A": [(-0.01, -0.01), (0.01, -0.01), (0.01, 0.01), (-0.01, 0.01)],
            "selection_rank": 1,
            "conformer_energy_eV": 0.0,
            "projected_area_A2": 0.0004,
        }
    ]
    placed = [_cohesive_policy_candidate("placed", 9, 3, 0.0)]

    def fake_contact(candidate, accepted, **kwargs):
        return {"has_contact": True, "contact_count": 5}
    monkeypatch.setattr(growth, "cohesive_side_contact_report", fake_contact)

    winner, audit = growth.choose_cohesive_frontier_candidate(
        candidates=candidates,
        placed=placed,
        accepted_union=shapely_box(0.05, 0.05, 0.08, 0.08),
        projection_cache=cache,
        cell=np.diag([10.0, 10.0, 30.0]),
        periodic_axes=(0, 1),
        collision_thresholds_A={"H-H": 1.5, "H-heavy": 1.8, "heavy-heavy": 2.2},
        rng=random.Random(7),
        cohesive_frontier_mode="continuous-snap",
        frontier_snap_radius_A=4.0,
    )

    assert winner is not None
    assert audit["cohesive_frontier_mode"] == "continuous-snap"
    assert streaming_call_count == 1












def _setup_mock_cli_run(tmp_path, monkeypatch):
    import monolayer_sequential_growth as growth

    substrate, candidate, _, _, _ = _independent_validation_fixture(tmp_path)
    candidate["candidate_index"] = 0
    candidate["selection_rank"] = 1
    candidate["cluster_id"] = 1
    candidate["source"] = "selection-0001.extxyz"
    candidate["target_donor_labels"] = ["O1", "O2"]
    candidate["oxygen_permutation"] = [1, 2]
    candidate["headgroup_rmsd_A"] = 0.0
    candidate["substrate_minimum_distance_A"] = None
    candidate["substrate_clearance_ratio"] = None
    candidate["conformer_sha256"] = _conformer_geometry_sha256(
        candidate["coordinates"], candidate["symbols"]
    )
    candidate["anchor_cartesian_A"] = candidate["coordinates"][0].tolist()
    candidate["candidate_id"] = "candidate-a"
    candidate["registered_interface_bonds"][0]["full_metal_atom_id_1based"] = 1
    candidate["registered_interface_bonds"][1]["full_metal_atom_id_1based"] = 2

    for name, payload in {
        "substrate.extxyz": b"substrate input\n",
        "layer-groups.json": b"{}\n",
        "site-prototypes.json": b"{}\n",
    }.items():
        (tmp_path / name).write_bytes(payload)
    (tmp_path / "site-instances.json").write_text(
        json.dumps(
            {
                "site_instances": [
                    {
                        "site_instance_id": "site-a",
                        "parent_site_prototype_id": "prototype-a",
                        "actual_metal_atom_ids_1based": [1, 2],
                    }
                ],
                "conflict_edges": [],
            }
        )
    )
    selected_path = tmp_path / "selection-0001.extxyz"
    selected_path.write_bytes(b"hash-locked conformer bytes\n")
    selected_sha256 = hashlib.sha256(selected_path.read_bytes()).hexdigest()
    analysis_record = {
        "selection_rank": 1,
        "cluster_id": 1,
        "relative_total_energy_eV": 0.0,
        "footprint_area_A2": 50.0,
        "structure": {"path": selected_path.name, "sha256": selected_sha256},
    }
    (tmp_path / "analysis.json").write_text(
        json.dumps({"selection": {"selected": [analysis_record]}})
    )
    monkeypatch.setattr(
        growth, "_selected_analysis_records", lambda *args: [analysis_record]
    )

    def fake_source_conformers(*args, **kwargs):
        assert kwargs["immutable_new_cache"] is True
        cache_prefix = Path(args[2])
        cache_prefix.parent.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(
            Path(f"{cache_prefix}.npz"),
            coordinates=np.zeros((1, 1, 3)),
            symbols=np.asarray(["C"]),
        )
        Path(f"{cache_prefix}.json").write_text(
            json.dumps({"schema_version": 2}, allow_nan=False)
        )
        return [{"source_rank": 1}]

    monkeypatch.setattr(growth, "source_conformers", fake_source_conformers)
    monkeypatch.setattr(
        growth,
        "_audit_loaded_conformer_ensemble",
        lambda *args, **kwargs: [{"selection_rank": 1, "geometry_hashes_match": True}],
    )
    monkeypatch.setattr(
        growth,
        "_load_working_substrate",
        lambda *args: (
            substrate,
            {
                "normal_axis": 2,
                "restore_layer_atom_ids_1based": [3],
                "working_layer_atom_ids_1based": [1, 2],
            },
        ),
    )
    monkeypatch.setattr(
        growth,
        "_prototype_map",
        lambda *args: {"prototype-a": {"surface_frame": candidate["surface_frame"]}},
    )
    skip_contract = build_substrate_collision_contract(
        mode="skip",
        periodic_axes=(0, 1),
        mapped_bond_window_A=(1.7, 2.8),
        hard_thresholds_A={
            "H-H": 1.5,
            "H-heavy": 1.8,
            "heavy-heavy": 2.2,
        },
        substrate_symbols=substrate.get_chemical_symbols(),
    )

    def fake_candidate_domains(**kwargs):
        request_path = tmp_path / "formal-output" / "run-request.json"
        assert request_path.is_file()
        _strict_json_loads(request_path.read_bytes())
        return (
            {"site-a": [candidate]},
            {
                "candidate_count": 1,
                "site_count": 1,
                "empty_site_count": 0,
                "substrate_collision_contract": skip_contract,
                "candidate_domain_fingerprint": "d" * 64,
                "candidate_domain_fingerprint_namespace": "legacy-v1",
            },
        )

    monkeypatch.setattr(growth, "_candidate_domains", fake_candidate_domains)

    argv = [
        "--root",
        str(tmp_path),
        "--substrate",
        "substrate.extxyz",
        "--layer-groups",
        "layer-groups.json",
        "--site-instances",
        "site-instances.json",
        "--site-prototypes",
        "site-prototypes.json",
        "--conformer-glob",
        "selection-*.extxyz",
        "--conformer-analysis",
        "analysis.json",
        "--output-dir",
        "formal-output",
        "--conformer-cache",
        "formal-output/internal-cache/conformers",
        "--projection-cache",
        "formal-output/internal-cache/projection",
        "--molecule-formula-json",
        '{"C":1,"O":2,"P":1}',
        "--max-conformers",
        "1",
        "--seed",
        "19",
        "--hh-min",
        "1.5",
        "--h-heavy-min",
        "1.8",
        "--heavy-heavy-min",
        "2.2",
        "--substrate-collision",
        "skip",
        "--independent-validation",
        "strict-interface",
    ]
    return candidate, argv


























def test_single_metal_continuous_snap_discrete_fallback_audit():
    """Verify that cohesive_frontier_mode='continuous-snap' performs explicit discrete fallback on single-metal candidates with audit tracking."""
    from monolayer_sequential_growth import run_sequential_domains
    from shapely.geometry import box

    cell = np.diag([20.0, 20.0, 30.0])
    periodic_axes = (0, 1)
    thresholds = {"H-H": 1.2, "H-heavy": 1.4, "heavy-heavy": 1.8}

    domains = {}
    cache = _fake_cache(
        masks={},
        geometries={},
        site_ids=[],
        area_scale_A2=400.0,
        surface_lattice_uv_A=np.diag([20.0, 20.0]),
    )
    cache.candidate_ids = []

    conformer_coords = np.array([
        [0.0, 0.0, 0.5],
        [0.0, 0.5, 0.0],
        [-0.5, -0.5, 0.0],
        [0.5, -0.5, 0.0],
        [0.0, 0.0, 7.5],
    ])
    conformers = [
        {
            "source_rank": 1,
            "source": "dummy_source",
            "p_local": 0,
            "o_locals": [1, 2, 3],
            "coordinates": conformer_coords,
            "symbols": np.array(["P", "O", "O", "O", "C"]),
            "source_anchor_height_A": 1.85,
        }
    ]

    single_metal_context = {
        "mode": "dynamic-cn5-uncovered",
        "uncovered_metals": [
            {"full_metal_atom_id_1based": 104, "working_atom_index_0based": 0, "element": "In", "coordination_number": 5},
        ],
        "substrate_positions": np.array([[10.0, 10.0, 4.0]]),
        "substrate_symbols": np.array(["In"]),
        "cell": cell,
        "periodic_axes": periodic_axes,
        "surface_frame": _identity_frame(),
        "conformers": conformers,
        "metrics_by_rank": {1: {"cluster_id": 1, "relative_total_energy_eV": 0.015}},
        "substrate_height_clearance_A": 1.0,
        "rotation_step_deg": 60.0,
        "translation_offsets_A": (0.0,),
    }

    result = run_sequential_domains(
        domains=domains,
        site_conflicts={},
        conformer_weights={1: 1.0},
        cell=cell,
        periodic_axes=periodic_axes,
        seed=42,
        selection_policy="cohesive-frontier",
        cohesive_frontier_mode="continuous-snap",
        projection_cache=cache,
        collision_thresholds_A=thresholds,
        single_metal_context=single_metal_context,
    )

    assert result["emergent_molecule_count"] == 1
    step = result["steps"][0]
    assert step["site_kind"] == "single_metal_triangle"
    audit = step["policy_audit"]
    assert audit.get("continuous_snap_bypass_reason") == "single_metal_triangle_uses_discrete_uncovered_metal_anchors"























