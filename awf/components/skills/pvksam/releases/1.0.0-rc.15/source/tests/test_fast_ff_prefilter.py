"""Scientific regressions for the isolated, fixed-head skeleton FF stage.

RDKit is intentionally kept in the project's separate chemistry environment.
These tests exercise its real minimizer through that interpreter rather than
replacing force-field behavior with a mock or requiring RDKit in the ASE env.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
OWNER = ROOT / "scripts" / "fast_ff_prefilter.py"
RDKIT_PYTHON = Path("/home/software/anaconda3/envs/rdkit/bin/python")


@pytest.fixture(scope="module")
def rdkit_python() -> str:
    if not RDKIT_PYTHON.is_file():
        pytest.skip("the separate project RDKit environment is unavailable")
    check = subprocess.run(
        [str(RDKIT_PYTHON), "-c", "from rdkit.Chem import AllChem"],
        capture_output=True,
        text=True,
        check=False,
    )
    if check.returncode:
        pytest.skip("the separate project interpreter cannot import RDKit")
    return str(RDKIT_PYTHON)


def _rdkit_json(interpreter: str, body: str, payload: dict) -> dict:
    program = (
        "import json, sys\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        f"sys.path.insert(0, {str(ROOT / 'scripts')!r})\n"
        "import fast_ff_prefilter as ff\n"
        "payload = json.loads(sys.argv[1])\n"
        + body
        + "\nprint('TEST_RESULT=' + json.dumps(result, sort_keys=True))\n"
    )
    completed = subprocess.run(
        [interpreter, "-c", program, json.dumps(payload)],
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    line = next(
        item for item in reversed(completed.stdout.splitlines())
        if item.startswith("TEST_RESULT=")
    )
    return json.loads(line.removeprefix("TEST_RESULT="))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def skeleton_ensemble(tmp_path: Path, rdkit_python: str) -> dict:
    """Build a small acid with explicit H and a sealed two-candidate H0 set."""

    source_sdf = tmp_path / "butylphosphonic-acid.sdf"
    chemistry = _rdkit_json(
        rdkit_python,
        """
from rdkit import Chem
from rdkit.Chem import AllChem
acid = Chem.AddHs(Chem.MolFromSmiles('CCCCP(=O)(O)O'))
assert AllChem.EmbedMolecule(acid, randomSeed=4917) == 0
Chem.MolToMolFile(acid, payload['source_sdf'])
p = next(atom.GetIdx() for atom in acid.GetAtoms() if atom.GetSymbol() == 'P')
oxygen = sorted(atom.GetIdx() for atom in acid.GetAtomWithIdx(p).GetNeighbors()
                if atom.GetSymbol() == 'O')
removed = sorted(neighbor.GetIdx() for index in oxygen
                 for neighbor in acid.GetAtomWithIdx(index).GetNeighbors()
                 if neighbor.GetSymbol() == 'H')
retained = [index for index in range(acid.GetNumAtoms()) if index not in removed]
symbols = [acid.GetAtomWithIdx(index).GetSymbol() for index in retained]
positions = np.asarray(acid.GetConformer().GetPositions())[retained]
mapped = {old: new for new, old in enumerate(retained)}
head = sorted([mapped[p]] + [mapped[index] for index in oxygen])
o = [mapped[index] for index in oxygen]
origin = positions[o].mean(axis=0)
x = positions[o[1]] - positions[o[0]]
x /= np.linalg.norm(x)
normal = np.cross(positions[o[1]] - positions[o[0]],
                  positions[o[2]] - positions[o[0]])
normal /= np.linalg.norm(normal)
if np.dot(positions[mapped[p]] - origin, normal) < 0:
    normal = -normal
y = np.cross(normal, x)
positions = (positions - origin) @ np.column_stack((x, y, normal))
molecule, actual_head = ff._h0_rdkit_template(Path(payload['source_sdf']), symbols)
result = {
    'source_atom_count': acid.GetNumAtoms(), 'removed_h': removed,
    'retained': retained, 'symbols': symbols, 'positions': positions.tolist(),
    'head': head, 'actual_head': actual_head,
    'formal_charge': sum(atom.GetFormalCharge() for atom in molecule.GetAtoms()),
    'expected_bonds': sorted([sorted((mapped[bond.GetBeginAtomIdx()],
                                      mapped[bond.GetEndAtomIdx()]))
                             for bond in acid.GetBonds()
                             if bond.GetBeginAtomIdx() in mapped
                             and bond.GetEndAtomIdx() in mapped]),
    'actual_bonds': sorted([sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
                           for bond in molecule.GetBonds()]),
}
""",
        {"source_sdf": str(source_sdf)},
    )
    sys.path.insert(0, str(ROOT / "scripts"))
    import fast_ff_prefilter as ff

    symbols = chemistry["symbols"]
    positions = np.asarray(chemistry["positions"])
    h0_path = tmp_path / "sam-h0.extxyz"
    ff._write_extxyz(h0_path, symbols, positions, 'Properties=species:S:1:pos:R:3')
    candidates = tmp_path / "ButylPA_monomer_unoptimized_conformations"
    candidates.mkdir()
    non_head = [index for index in range(len(symbols)) if index not in chemistry["head"]]
    records = []
    for rank in (1, 2):
        candidate = positions.copy()
        # A real geometric perturbation provides a nonzero force to minimize.
        candidate[non_head[-rank]] += [0.35, -0.12, 0.08]
        path = candidates / f"candidate-{rank:06d}.extxyz"
        ff._write_extxyz(path, symbols, candidate, 'Properties=species:S:1:pos:R:3')
        records.append({
            "grid_index": rank,
            "safe_rank": rank,
            "status": "passed_skeleton_cpu_screen",
            "start_dihedrals_deg": [-180.0 + 30.0 * rank],
            "structure": str(path.relative_to(tmp_path)),
            "structure_sha256": _sha256(path),
        })
    manifest = {
        "schema": "sam-phosphonate-skeleton-cpu-v1",
        "status": "screen_completed",
        "grid_complete": True,
        "scope": "isolated_geometry_only_not_ITO_collision_audit",
        "records_scope": "isolated_skeleton_screen_safe_only",
        "system": {
            "sam_atom_count": len(symbols),
            "sam_source": {"path": str(h0_path), "sha256": _sha256(h0_path)},
        },
        "parameters": {"head_atom_ids_1based": [index + 1 for index in chemistry["head"]]},
        "summary": {"screen_safe_count": len(records)},
        "records": records,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return {
        "source_sdf": source_sdf,
        "h0": h0_path,
        "manifest": manifest_path,
        "chemistry": chemistry,
        "records": records,
        "root": tmp_path,
    }


def _run_stage(interpreter: str, ensemble: dict, *, max_iterations: int = 1000,
               limit: int | None = None, optimization_only: bool = True) -> subprocess.CompletedProcess:
    arguments = [
        interpreter, str(OWNER),
        "--cpu-manifest", str(ensemble["manifest"]),
        "--template-sdf", str(ensemble["source_sdf"]),
        "--output-root", str(ensemble["root"] / "optimized"),
        "--force-field", "mmff94s", "--workers", "1", "--chunksize", "1",
        "--max-iterations", str(max_iterations),
        "--movable-fmax-eV-A", "0.03", "--isomer-name", "ButylPA",
    ]
    if optimization_only:
        arguments.append("--optimization-only")
    if limit is not None:
        arguments.extend(("--limit", str(limit)))
    return subprocess.run(arguments, capture_output=True, text=True, check=False, timeout=90)


def _output(ensemble: dict) -> tuple[dict, list[dict]]:
    manifests = list((ensemble["root"] / "optimized").rglob("fast-ff-manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    records = [json.loads(line) for line in Path(manifest["artifacts"]["optimized_records"]).read_text().splitlines()]
    return manifest, records


def test_neutral_acid_to_h0_preserves_atom_order_and_bond_mapping(skeleton_ensemble: dict) -> None:
    chemistry = skeleton_ensemble["chemistry"]
    assert len(chemistry["removed_h"]) == 2
    assert len(chemistry["symbols"]) == chemistry["source_atom_count"] - 2
    assert chemistry["actual_head"] == chemistry["head"]
    assert chemistry["formal_charge"] == -2
    assert chemistry["actual_bonds"] == chemistry["expected_bonds"]


def test_skeleton_optimization_fixes_four_head_atoms_and_reports_real_forces(
    rdkit_python: str, skeleton_ensemble: dict,
) -> None:
    completed = _run_stage(rdkit_python, skeleton_ensemble)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    manifest, records = _output(skeleton_ensemble)
    assert manifest["schema"] == "sam-phosphonate-skeleton-ff-v1"
    assert manifest["parameters"]["force_field_actual_variant"] == "MMFF94s"
    assert len(records) == 2
    assert all(record["status"] == "optimized_converged" for record in records)
    assert manifest["summary"]["all_input_candidates_processed"] is True
    assert manifest["summary"]["force_field_finite_result_count"] == 2
    assert manifest["summary"]["force_field_converged_count"] == 2
    assert manifest["summary"]["substrate_collision_audit_count"] == 0
    assert manifest["summary"]["clustering_performed"] is False
    assert manifest["system"]["fixed_head_atom_ids_1based"] == [
        index + 1 for index in skeleton_ensemble["chemistry"]["head"]
    ]
    assert "no_ITO_audit" in manifest["scope"]
    assert not list((skeleton_ensemble["root"] / "optimized").rglob("clusters.json"))
    head = skeleton_ensemble["chemistry"]["head"]
    non_head = [index for index in range(len(skeleton_ensemble["chemistry"]["symbols"])) if index not in head]
    for record in records:
        assert record["force_field_actual_variant"] == "MMFF94s"
        observed = _rdkit_json(
            rdkit_python,
            """
from rdkit import Chem
from rdkit.Chem import AllChem
symbols, before, _ = ff._read_extxyz_bytes(Path(payload['source']).read_bytes())
_, after, _ = ff._read_extxyz_bytes(Path(payload['optimized']).read_bytes())
molecule, head = ff._h0_rdkit_template(Path(payload['sdf']), symbols)
conf = molecule.GetConformer()
for index, xyz in enumerate(after):
    conf.SetAtomPosition(index, tuple(float(value) for value in xyz))
properties = AllChem.MMFFGetMoleculeProperties(molecule, mmffVariant='MMFF94s')
force_field = AllChem.MMFFGetMoleculeForceField(molecule, properties)
for index in head:
    force_field.AddFixedPoint(index)
force_field.Initialize()
gradient = np.asarray(force_field.CalcGrad()).reshape((-1, 3))
movable = [index for index in range(len(symbols)) if index not in head]
result = {'before': before.tolist(), 'after': after.tolist(),
          'movable_fmax_eV_A': float(np.max(np.linalg.norm(gradient[movable], axis=1)))
                              * 0.0433641153087705}
""",
            {"source": record["structure"], "optimized": record["optimized_structure"],
             "sdf": str(skeleton_ensemble["source_sdf"])},
        )
        before, after = np.asarray(observed["before"]), np.asarray(observed["after"])
        np.testing.assert_allclose(after[head], before[head], atol=1.0e-12, rtol=0)
        assert np.max(np.linalg.norm(after[non_head] - before[non_head], axis=1)) > 0.05
        assert observed["movable_fmax_eV_A"] <= 0.03
        assert record["force_field_max_gradient_non_head_eV_A"] == pytest.approx(
            observed["movable_fmax_eV_A"], abs=1.0e-7,
        )


def test_iteration_limit_retains_geometry_without_claiming_convergence(
    rdkit_python: str, skeleton_ensemble: dict,
) -> None:
    completed = _run_stage(rdkit_python, skeleton_ensemble, max_iterations=1)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    _, records = _output(skeleton_ensemble)
    assert len(records) == 2
    assert all(record["status"] == "optimized_not_converged" for record in records)
    assert all(record["force_field_not_converged"] for record in records)
    assert all(record["force_field_termination_code"] != 0 for record in records)
    assert all(Path(record["optimized_structure"]).is_file() for record in records)


def test_partial_limit_remains_explicit_and_does_not_cluster(
    rdkit_python: str, skeleton_ensemble: dict,
) -> None:
    completed = _run_stage(rdkit_python, skeleton_ensemble, limit=1)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    manifest, records = _output(skeleton_ensemble)
    assert len(records) == 1
    assert manifest["summary"]["input_skeleton_safe_count"] == 2
    assert manifest["summary"]["unique_candidates_submitted"] == 1
    assert manifest["summary"]["all_input_candidates_processed"] is False
    assert manifest["summary"]["input_grid_complete"] is True
    assert manifest["summary"]["limit"] == 1
    assert not list((skeleton_ensemble["root"] / "optimized").rglob("clusters.json"))


def test_skeleton_contract_requires_optimization_only(
    rdkit_python: str, skeleton_ensemble: dict,
) -> None:
    completed = _run_stage(rdkit_python, skeleton_ensemble, optimization_only=False)
    assert completed.returncode != 0
    assert "optimization-only" in completed.stderr
    assert not list((skeleton_ensemble["root"] / "optimized").rglob("fast-ff-manifest.json"))


def test_changed_source_hash_stops_before_optimization(
    rdkit_python: str, skeleton_ensemble: dict,
) -> None:
    with skeleton_ensemble["h0"].open("a") as handle:
        handle.write("\n")
    completed = _run_stage(rdkit_python, skeleton_ensemble)
    assert completed.returncode != 0
    assert "hash" in completed.stderr.lower()
    assert not list((skeleton_ensemble["root"] / "optimized").rglob("fast-ff-manifest.json"))


def test_changed_candidate_hash_stops_before_optimization(
    rdkit_python: str, skeleton_ensemble: dict,
) -> None:
    source = skeleton_ensemble["root"] / skeleton_ensemble["records"][0]["structure"]
    with source.open("a") as handle:
        handle.write("\n")
    completed = _run_stage(rdkit_python, skeleton_ensemble)
    assert completed.returncode != 0
    assert "candidate hash changed" in completed.stderr
    assert not list((skeleton_ensemble["root"] / "optimized").rglob("fast-ff-manifest.json"))


def test_worker_rechecks_hash_before_reading_candidate_geometry(
    rdkit_python: str, skeleton_ensemble: dict,
) -> None:
    record = skeleton_ensemble["records"][0]
    source = skeleton_ensemble["root"] / record["structure"]
    observed = _rdkit_json(
        rdkit_python,
        """
ff._worker_init(payload['sdf'], payload['symbols'], 0, len(payload['symbols']),
                1000, 'mmff94s', 0.03)
source = Path(payload['source'])
with source.open('a') as handle:
    handle.write('\\n')
result = ff._optimize_one({
    'structure': str(source), 'structure_sha256': payload['sha256'],
    'grid_index': 1, 'safe_rank': 1, 'candidate_key': [-150.0],
    'start_dihedrals_deg': [-150.0],
})
""",
        {"source": str(source), "sha256": record["structure_sha256"],
         "sdf": str(skeleton_ensemble["source_sdf"]),
         "symbols": skeleton_ensemble["chemistry"]["symbols"]},
    )
    assert observed["status"] == "failed"
    assert "hash changed" in observed["error"]
    assert "positions" not in observed


def test_legacy_fixed_site_stage_keeps_single_minimize_and_clustering_contract(
    rdkit_python: str, skeleton_ensemble: dict,
) -> None:
    """The legacy finite-result 'passed' status must not change meaning."""

    legacy_directory = skeleton_ensemble["root"] / "legacy" / "02-fixed-site" / "screen-id"
    legacy_directory.mkdir(parents=True)
    records = []
    for original in skeleton_ensemble["records"]:
        record = dict(original)
        record["status"] = "passed_cpu_collision_screen"
        record["structure"] = str(skeleton_ensemble["root"] / original["structure"])
        records.append(record)
    legacy_manifest = {
        "records_scope": "collision_safe_only",
        "system": {
            "substrate_atom_count": 0,
            "surface_h_count": 0,
            "sam_atom_count": len(skeleton_ensemble["chemistry"]["symbols"]),
        },
        "summary": {"collision_safe_count": len(records)},
        "records": records,
    }
    legacy_path = legacy_directory / "manifest.json"
    legacy_path.write_text(json.dumps(legacy_manifest))
    completed = subprocess.run(
        [rdkit_python, str(OWNER),
         "--cpu-manifest", str(legacy_path),
         "--template-sdf", str(skeleton_ensemble["source_sdf"]),
         "--output-root", str(skeleton_ensemble["root"] / "optimized"),
         "--force-field", "mmff94s", "--workers", "1", "--chunksize", "1",
         "--max-iterations", "1"],
        capture_output=True, text=True, check=False, timeout=90,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    manifest, optimized = _output(skeleton_ensemble)
    assert manifest["status"] == "passed_ff_pending_substrate_audit"
    assert manifest["schema_version"] == 2
    assert "schema" not in manifest
    assert manifest["summary"]["force_field_passed_count"] == 2
    assert manifest["summary"]["cluster_count"] >= 1
    assert Path(manifest["artifacts"]["clusters"]).is_file()
    assert all(record["status"] == "passed" for record in optimized)
    assert all(record["force_field_not_converged"] for record in optimized)
    assert all(record["force_field_minimize_block_count"] == 1 for record in optimized)
    assert all(record["force_field_requested_iteration_budget"] == 1 for record in optimized)
    head = skeleton_ensemble["chemistry"]["head"]
    sys.path.insert(0, str(ROOT / "scripts"))
    import fast_ff_prefilter as ff
    for record in optimized:
        _, before, _ = ff._read_extxyz_bytes(Path(record["structure"]).read_bytes())
        _, after, _ = ff._read_extxyz_bytes(Path(record["optimized_structure"]).read_bytes())
        np.testing.assert_allclose(after[head], before[head], atol=1.0e-12, rtol=0)


def test_mmff94s_uses_the_official_variant_for_a_sensitive_amine(
    rdkit_python: str, tmp_path: Path,
) -> None:
    """An out-of-plane N distinguishes MMFF94s from RDKit's MMFF94 fallback.

    Alkyl phosphonates alone do not reliably distinguish these variants.  A
    diphenylamine tail displaced out of its local plane supplies a diagnostic
    MMFF94s term; reference energies are calculated at identical coordinates.
    """

    observed = _rdkit_json(
        rdkit_python,
        """
from rdkit import Chem
from rdkit.Chem import AllChem
acid = Chem.AddHs(Chem.MolFromSmiles('O=P(O)(O)CCN(c1ccccc1)c1ccccc1'))
assert AllChem.EmbedMolecule(acid, randomSeed=4917) == 0
sdf = Path(payload['directory']) / 'diphenylamine-phosphonic-acid.sdf'
Chem.MolToMolFile(acid, str(sdf))
p = next(atom.GetIdx() for atom in acid.GetAtoms() if atom.GetSymbol() == 'P')
oxygen = [atom.GetIdx() for atom in acid.GetAtomWithIdx(p).GetNeighbors()
          if atom.GetSymbol() == 'O']
removed = {neighbor.GetIdx() for index in oxygen
           for neighbor in acid.GetAtomWithIdx(index).GetNeighbors()
           if neighbor.GetSymbol() == 'H'}
retained = [index for index in range(acid.GetNumAtoms()) if index not in removed]
symbols = [acid.GetAtomWithIdx(index).GetSymbol() for index in retained]
positions = np.asarray(acid.GetConformer().GetPositions())[retained]
n = next(index for index, symbol in enumerate(symbols) if symbol == 'N')
mapped = {old: new for new, old in enumerate(retained)}
neighbors = [mapped[atom.GetIdx()]
             for atom in acid.GetAtomWithIdx(retained[n]).GetNeighbors()]
assert len(neighbors) == 3
neighbor_positions = positions[neighbors]
origin = neighbor_positions.mean(axis=0)
normal = np.cross(neighbor_positions[1] - neighbor_positions[0],
                  neighbor_positions[2] - neighbor_positions[0])
normal /= np.linalg.norm(normal)
positions[n] -= np.dot(positions[n] - origin, normal) * normal
positions[n] += 0.4 * normal
candidate = Path(payload['directory']) / 'candidate.extxyz'
ff._write_extxyz(candidate, symbols, positions, 'Properties=species:S:1:pos:R:3')
ff._worker_init(str(sdf), symbols, 0, len(symbols), 1, 'mmff94s', 0.03)
record = ff._optimize_one({
    'structure': str(candidate), 'structure_sha256': ff._sha256_path(candidate),
    'grid_index': 1, 'safe_rank': 1, 'candidate_key': [0.0],
    'start_dihedrals_deg': [0.0],
})
assert record['status'] == 'passed', record
molecule, head = ff._h0_rdkit_template(sdf, symbols)
conf = molecule.GetConformer()
for index, xyz in enumerate(record['positions']):
    conf.SetAtomPosition(index, tuple(xyz))
energies = {}
for variant in ('MMFF94', 'MMFF94s'):
    properties = AllChem.MMFFGetMoleculeProperties(molecule, mmffVariant=variant)
    force_field = AllChem.MMFFGetMoleculeForceField(molecule, properties)
    for index in head:
        force_field.AddFixedPoint(index)
    force_field.Initialize()
    energies[variant] = float(force_field.CalcEnergy())
result = {'actual_variant': record['force_field_actual_variant'],
          'recorded_energy': record['force_field_energy_kcal_mol'],
          'reference_energies': energies}
""",
        {"directory": str(tmp_path)},
    )
    energy_94 = observed["reference_energies"]["MMFF94"]
    energy_94s = observed["reference_energies"]["MMFF94s"]
    assert abs(energy_94s - energy_94) > 1.0e-3
    assert observed["actual_variant"] == "MMFF94s"
    assert observed["recorded_energy"] == pytest.approx(energy_94s, abs=1.0e-8)
    assert abs(observed["recorded_energy"] - energy_94) > 1.0e-3


def _postprocess_input(ensemble: dict) -> tuple[Path, list[dict]]:
    """Seal artificial geometry-audit cases with the real reviewed SDF graph.

    These intentionally stretched coordinates test the geometric contract,
    not force-field energetics. Convergence metadata is a synthetic gate input.
    """

    sys.path.insert(0, str(ROOT / "scripts"))
    import fast_ff_prefilter as ff

    chemistry = ensemble["chemistry"]
    symbols = chemistry["symbols"]
    head = chemistry["head"]
    organic = sorted(set(range(len(symbols))) - set(head))
    p = symbols.index("P")
    c = next(right if left == p else left
             for left, right in chemistry["expected_bonds"]
             if p in (left, right) and symbols[right if left == p else left] == "C")
    coordinates = np.asarray(chemistry["positions"]).copy()
    axis = np.array([0.6, 0.0, 0.8])
    coordinates[c] = coordinates[p] + 1.8 * axis
    for serial, index in enumerate(index for index in organic if index != c):
        coordinates[index] = coordinates[p] + (10.0 + 4.0 * serial) * axis
    hydrogen_indices = [index for index in organic if symbols[index] == "H"]
    perpendicular = np.array([-0.8, 0.0, 0.6])
    coordinates[hydrogen_indices[-1]] = coordinates[p] + 20.0 * axis - 80.0 * perpendicular
    assert coordinates[hydrogen_indices[-1], 2] < 0.0
    records = []
    for rank, condition in enumerate(("rescuable", "collision", "not_converged", "height_impossible"), 1):
        candidate = coordinates.copy()
        if condition == "collision":
            candidate[hydrogen_indices[0]] = candidate[hydrogen_indices[1]]
        if condition == "height_impossible":
            candidate[c] = candidate[p] + [3.0, 0.0, -2.0]
        path = ensemble["root"] / f"audit-source-{rank}.extxyz"
        ff._write_extxyz(path, symbols, candidate, 'Properties=species:S:1:pos:R:3')
        records.append({
            "safe_rank": rank, "grid_index": rank + 10, "candidate_key": [rank * 30.0],
            "status": "optimized_not_converged" if condition == "not_converged" else "optimized_converged",
            "force_field_converged": condition != "not_converged",
            "force_field_max_gradient_non_head_eV_A": 1.0 if condition == "not_converged" else 0.001,
            "force_field_energy_kcal_mol": 999.0, "force_field": "mmff94s",
            "force_field_actual_variant": "MMFF94s",
            "structure": str(path), "structure_sha256": _sha256(path),
            "optimized_structure": str(path), "optimized_structure_sha256": _sha256(path),
        })
    records_path = ensemble["root"] / "ff-records.jsonl"
    records_path.write_text("".join(json.dumps(record) + "\n" for record in records))
    source_paths = {"cpu_manifest": ensemble["manifest"], "template_sdf": ensemble["source_sdf"],
                    "h0_source": ensemble["h0"]}
    sources = {key: str(path) for key, path in source_paths.items()}
    sources.update({f"{key}_sha256": _sha256(path) for key, path in source_paths.items()})
    manifest = {
        "schema": "sam-phosphonate-skeleton-ff-v1", "records_scope": "optimized_skeleton_all_outcomes",
        "sources": sources, "parameters": {"isomer_name": "ButylPA", "movable_fmax_eV_A": 0.03,
                                              "force_field": "mmff94s", "force_field_actual_variant": "MMFF94s"},
        "system": {"sam_h0_formula_symbols": symbols, "sam_atom_count": len(symbols),
                   "fixed_head_atom_ids_1based": [index + 1 for index in head]},
        "summary": {"unique_candidates_submitted": len(records), "all_input_candidates_processed": True,
                    "input_grid_complete": True},
        "artifacts": {"optimized_records": str(records_path)},
    }
    manifest_path = ensemble["root"] / "ff-manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path, records


def _run_postprocess(manifest: Path, rdkit_python: str, output: Path, *,
                     limit: int | None = None, tolerance_A: float = 1.0,
                     method: str | None = None) -> subprocess.CompletedProcess:
    arguments = [sys.executable, str(OWNER), "--skeleton-postprocess-manifest", str(manifest),
                 "--topology-python", rdkit_python, "--output-root", str(output),
                 "--rmsd-tolerance-A", str(tolerance_A)]
    if limit is not None:
        arguments.extend(["--limit", str(limit)])
    if method is not None:
        arguments.extend(["--skeleton-dedup-method", method])
    return subprocess.run(arguments, capture_output=True, text=True, check=False, timeout=90)


def test_postprocess_rescues_negative_z_and_rejects_actual_geometry_failures(
    rdkit_python: str, skeleton_ensemble: dict,
) -> None:
    manifest_path, source_records = _postprocess_input(skeleton_ensemble)
    output = skeleton_ensemble["root"] / "postprocess"
    completed = _run_postprocess(manifest_path, rdkit_python, output)
    assert completed.returncode == 0, completed.stderr + completed.stdout
    manifest = json.loads(next(output.rglob("skeleton-postprocess-manifest.json")).read_text())
    assert manifest["parameters"]["skeleton_dedup_method"] == "energy-radius-symmetry"
    assert manifest["parameters"]["symmetry_automorphism_count"] >= 1
    assert manifest["system"]["topology"]["rmsd_atom_automorphisms"]
    outcomes = [json.loads(line) for line in Path(manifest["artifacts"]["outcomes"]).read_text().splitlines()]
    assert [record["status"] for record in outcomes] == [
        "passed_shared_pc_phase_geometry", "rejected_invariant_intramolecular_collision",
        "rejected_source_force_field_not_converged", "rejected_height_no_common_pc_phase"]
    assert manifest["summary"]["raw_negative_z_rescued_count"] == 1
    assert manifest["summary"]["optimization_started_count"] == 0
    assert manifest["summary"]["ITO_collision_audit_count"] == 0
    assert manifest["summary"]["phase_safe_count"] == manifest["summary"]["cluster_count"] == 1
    safe = outcomes[0]
    assert not safe["pose_optimized_convergence_claim"]
    assert not safe["pose_force_rechecked"]
    assert "force_field_energy_kcal_mol" not in safe
    assert "force_field_converged" not in safe
    assert safe["source_optimized_result"]["force_field_energy_kcal_mol"] == 999.0
    assert safe["pose_organic_z_min_A"] >= -1.1e-8
    assert safe["pose_minimum_nonbonded_clearance_A"] >= -1.1e-8
    assert safe["shared_pc_phase"]["feasible_intervals_deg"]
    assert safe["pc_tilt_to_head_normal_deg"] == pytest.approx(np.degrees(np.arccos(0.8)))
    assert safe["pc_length_A"] == pytest.approx(1.8)
    sys.path.insert(0, str(ROOT / "scripts"))
    import fast_ff_prefilter as ff

    _, before, _ = ff._read_extxyz_bytes(Path(source_records[0]["optimized_structure"]).read_bytes())
    _, after, _ = ff._read_extxyz_bytes(Path(safe["pose_structure"]).read_bytes())
    assert np.array_equal(after[skeleton_ensemble["chemistry"]["head"]],
                          before[skeleton_ensemble["chemistry"]["head"]])
    organic = np.asarray(manifest["system"]["organic_atom_ids_1based"]) - 1
    before_distances = np.linalg.norm(before[organic, None] - before[None, organic], axis=2)
    after_distances = np.linalg.norm(after[organic, None] - after[None, organic], axis=2)
    assert np.allclose(before_distances, after_distances, atol=1.0e-9)


def test_postprocess_partial_limit_and_source_hash_integrity(
    rdkit_python: str, skeleton_ensemble: dict,
) -> None:
    manifest_path, records = _postprocess_input(skeleton_ensemble)
    output = skeleton_ensemble["root"] / "partial"
    completed = _run_postprocess(manifest_path, rdkit_python, output, limit=1, tolerance_A=0.5)
    assert completed.returncode == 0, completed.stderr + completed.stdout
    manifest = json.loads(next(output.rglob("skeleton-postprocess-manifest.json")).read_text())
    assert manifest["status"] == "partial_completed"
    assert not manifest["grid_complete"]
    assert not manifest["summary"]["all_parent_records_processed"]
    assert manifest["summary"]["processed_count"] == 1
    assert "0.5-A geometry families" in manifest["cluster_identity_scope"]
    changed = Path(records[-1]["optimized_structure"])
    changed.write_text(changed.read_text() + "\n# changed\n")
    failed_output = skeleton_ensemble["root"] / "hash-failure"
    failed = _run_postprocess(manifest_path, rdkit_python, failed_output, limit=1)
    assert failed.returncode == 2
    assert "hash changed" in failed.stderr
    assert not failed_output.exists()


def test_axial_complete_link_uses_true_atom_rmsd_and_blocks_chaining() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import fast_ff_prefilter as ff

    reference = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 2.0]])
    features = np.stack([reference, reference + [0.0, 0.0, 0.75],
                         reference + [0.0, 0.0, 1.5]])
    clusters, assignments, diameters = ff._complete_link_pc_axis_clusters(features, 1.0, batch_size=1)
    assert clusters == [[0, 1], [2]]
    assert assignments == [0, 0, 1]
    assert diameters == pytest.approx([0.75, 0.0])
    assert ff._complete_link_pc_axis_clusters(features, 1.0, batch_size=3) == (clusters, assignments, diameters)
    # A pure 1.2 A atom displacement is above 1 A in true 3D RMSD. Averaging
    # Cartesian components would incorrectly report 1.2/sqrt(3) and merge it.
    separated = np.stack([reference, reference + [0.0, 0.0, 1.2]])
    assert ff._complete_link_pc_axis_clusters(separated, 1.0)[0] == [[0], [1]]


def test_axial_complete_link_removes_only_a_common_pc_phase() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import fast_ff_prefilter as ff

    reference = np.array([[0.0, 0.0, 1.0], [1.0, 0.2, 2.0], [-0.2, 1.0, 3.0]])
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    phase_rotated = reference @ rotation.T
    internal_change = phase_rotated.copy()
    internal_change[-1, 2] += 3.0
    clusters, _, _ = ff._complete_link_pc_axis_clusters(
        np.stack([reference, phase_rotated, internal_change]), 1.0)
    assert clusters == [[0, 1], [2]]


def test_postprocess_uses_exact_original_sdf_mapping_and_rejects_changed_fixed_head(
    rdkit_python: str, skeleton_ensemble: dict,
) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import fast_ff_prefilter as ff

    topology = ff._h0_topology_contract(skeleton_ensemble["source_sdf"],
                                       skeleton_ensemble["chemistry"]["symbols"], rdkit_python)
    assert topology["bond_pairs"] == skeleton_ensemble["chemistry"]["actual_bonds"]
    assert topology["original_atom_ids_1based"] == [index + 1 for index in skeleton_ensemble["chemistry"]["retained"]]
    assert topology["source_atom_count"] - len(topology["original_atom_ids_1based"]) == 2
    assert topology["source_formal_charge"] == 0
    assert topology["h0_formal_charge"] == -2
    manifest_path, records = _postprocess_input(skeleton_ensemble)
    first_path = Path(records[0]["optimized_structure"])
    symbols, coordinates, comment = ff._read_extxyz_bytes(first_path.read_bytes())
    coordinates[topology["p_index"], 0] += 0.01
    ff._write_extxyz(first_path, symbols, coordinates, comment)
    records[0]["optimized_structure_sha256"] = records[0]["structure_sha256"] = _sha256(first_path)
    records_path = Path(json.loads(manifest_path.read_text())["artifacts"]["optimized_records"])
    records_path.write_text("".join(json.dumps(record) + "\n" for record in records))
    output = skeleton_ensemble["root"] / "head-invalid"
    failed = _run_postprocess(manifest_path, rdkit_python, output)
    assert failed.returncode == 2
    assert "fixed P/O coordinates changed" in failed.stderr
    assert not list(output.rglob("skeleton-postprocess-manifest.json"))


def _axial_shift_features(shifts: list[float]) -> np.ndarray:
    reference = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 2.0]])
    return np.stack([reference + [0.0, 0.0, shift] for shift in shifts])


def test_energy_radius_orders_before_selection_and_does_not_replace_representatives() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import fast_ff_prefilter as ff

    features = _axial_shift_features([-0.9, 0.0, 0.9])
    clusters, assignments, representatives, distances = ff._energy_ordered_pc_radius_clusters(
        features, [1.0, 0.0, 2.0], [10, 11, 12], [1, 2, 3], 1.0, batch_size=1)
    assert clusters == [[1, 0, 2]]
    assert representatives == [1]
    assert assignments == [0, 0, 0]
    assert distances == pytest.approx([0.9, 0.0, 0.9])
    # Members can be 1.8 A apart; this is a representative radius, not a
    # complete-link maximum pair distance.
    from sam_structure_tools import pc_axial_rmsd_matrix
    assert pc_axial_rmsd_matrix(features[0], features[2])[0, 0] == pytest.approx(1.8)
    assert ff._energy_ordered_pc_radius_clusters(features, [1.0, 0.0, 2.0],
                                                [10, 11, 12], [1, 2, 3], 1.0, batch_size=3) == (
                                                    clusters, assignments, representatives, distances)


def test_energy_radius_symmetry_merges_exact_graph_equivalent_atom_orderings() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import fast_ff_prefilter as ff
    from sam_structure_tools import pc_axial_rmsd_matrix

    reference = np.array([[2.0, 0.0, 1.0], [0.0, 3.0, -1.0]])
    candidate = reference[[1, 0]]
    features = np.stack([reference, candidate])
    original = features.copy()
    energies, grids, ranks = [0.0, 1.0], [1, 2], [1, 2]

    fixed = ff._energy_ordered_pc_radius_clusters(features, energies, grids, ranks, 1.0)
    symmetry = ff._energy_ordered_pc_radius_clusters(
        features, energies, grids, ranks, 1.0,
        atom_permutations=[[0, 1], [1, 0]],
    )

    assert pc_axial_rmsd_matrix(reference, candidate)[0, 0] > 1.0
    assert fixed[2] == [0, 1]
    assert symmetry[0] == [[0, 1]]
    assert symmetry[1] == [0, 0]
    assert symmetry[2] == [0]
    assert symmetry[3] == pytest.approx([0.0, 0.0])
    assert np.array_equal(features, original)


def test_energy_radius_never_remaps_members_to_later_higher_energy_representatives() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import fast_ff_prefilter as ff

    features = _axial_shift_features([0.0, 0.9, 1.5])
    clusters, _, representatives, distances = ff._energy_ordered_pc_radius_clusters(
        features, [0.0, 1.0, 2.0], [1, 2, 3], [1, 2, 3], 1.0, batch_size=3)
    assert clusters == [[0, 1], [2]]
    assert representatives == [0, 2]
    assert distances[1] == pytest.approx(0.9)
    # Candidate 1 is closer to the later representative (0.6 A), whose
    # energy exceeds its own. A final nearest-center reassignment is invalid.
    from sam_structure_tools import pc_axial_rmsd_matrix
    assert pc_axial_rmsd_matrix(features[1], features[2])[0, 0] == pytest.approx(0.6)


def test_energy_radius_uses_nearest_existing_leader_and_deterministic_energy_ties() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import fast_ff_prefilter as ff

    clusters, _, representatives, _ = ff._energy_ordered_pc_radius_clusters(
        _axial_shift_features([0.0, 1.8, 1.0]), [0.0, 1.0, 2.0], [1, 2, 3], [1, 2, 3], 1.0)
    assert representatives == [0, 1]
    assert clusters == [[0], [1, 2]]
    equal_distance = ff._energy_ordered_pc_radius_clusters(
        _axial_shift_features([0.0, 2.0, 1.0]), [0.0, 1.0, 2.0], [1, 2, 3], [1, 2, 3], 1.0)
    assert equal_distance[0] == [[0, 2], [1]]
    tied_energy = ff._energy_ordered_pc_radius_clusters(
        _axial_shift_features([0.0, 0.0, 0.0]), [1.0, 1.0, 1.0], [7, 8, 7], [3, 1, 2], 1.0)
    assert tied_energy[2] == [2]
    assert tied_energy[0] == [[2, 0, 1]]


def test_energy_radius_guarantees_cover_separation_and_representative_energy() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import fast_ff_prefilter as ff
    from sam_structure_tools import pc_axial_rmsd_matrix

    features = _axial_shift_features([0.0, 0.9, 1.5, 2.4, 3.7])
    energies = np.array([3.0, 2.0, 1.0, 4.0, 0.0])
    clusters, assignments, representatives, distances = ff._energy_ordered_pc_radius_clusters(
        features, energies, [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 1.0, batch_size=2)
    assert sorted(index for members in clusters for index in members) == list(range(5))
    matrix = pc_axial_rmsd_matrix(features)
    for index, assignment in enumerate(assignments):
        representative = representatives[assignment]
        assert matrix[index, representative] <= 1.0 + 1.0e-12
        assert distances[index] == pytest.approx(matrix[index, representative])
        assert energies[representative] <= energies[index]
    representative_matrix = matrix[np.ix_(representatives, representatives)]
    assert np.all(representative_matrix[np.triu_indices(len(representatives), 1)] > 1.0)


def test_energy_radius_empty_and_nonfinite_energy_contract() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import fast_ff_prefilter as ff

    assert ff._energy_ordered_pc_radius_clusters(np.empty((0, 2, 3)), [], [], [], 1.0) == ([], [], [], [])
    for energy in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="finite"):
            ff._energy_ordered_pc_radius_clusters(_axial_shift_features([0.0]), [energy], [1], [1], 1.0)
    with pytest.raises(ValueError, match="match candidates"):
        ff._energy_ordered_pc_radius_clusters(_axial_shift_features([0.0]), [], [1], [1], 1.0)


def _completed_multi_member_postprocess(ensemble: dict, rdkit_python: str) -> Path:
    """Three eligible synthetic poses, one historical high-energy representative."""

    ff_path, records = _postprocess_input(ensemble)
    for rank, energy in ((5, -10.0), (6, 0.0)):
        duplicate = dict(records[0])
        duplicate.update({"safe_rank": rank, "grid_index": rank + 10,
                          "candidate_key": [rank * 30.0], "force_field_energy_kcal_mol": energy})
        records.append(duplicate)
    ff_manifest = json.loads(ff_path.read_text())
    ff_manifest["summary"]["unique_candidates_submitted"] = len(records)
    Path(ff_manifest["artifacts"]["optimized_records"]).write_text(
        "".join(json.dumps(record) + "\n" for record in records))
    ff_path.write_text(json.dumps(ff_manifest))
    output = ensemble["root"] / "historical-complete-link"
    completed = _run_postprocess(ff_path, rdkit_python, output, method="complete-link")
    assert completed.returncode == 0, completed.stderr + completed.stdout
    post_path = next(output.rglob("skeleton-postprocess-manifest.json"))
    post = json.loads(post_path.read_text())
    assert post["parameters"]["skeleton_dedup_method"] == "complete-link"
    assert post["summary"]["phase_safe_count"] == 3
    assert post["summary"]["cluster_count"] == 1
    assert post["representatives"][0]["representative_safe_rank"] == 1
    assert post["representatives"][0]["representative_source_force_field_energy_kcal_mol"] == 999.0
    return post_path


def _run_rededup(parent: Path, rdkit_python: str, output: Path, *, limit: int | None = None) -> subprocess.CompletedProcess:
    arguments = [sys.executable, str(OWNER), "--skeleton-rededup-manifest", str(parent),
                 "--topology-python", rdkit_python, "--output-root", str(output)]
    if limit is not None:
        arguments.extend(["--limit", str(limit)])
    return subprocess.run(arguments, capture_output=True, text=True, check=False, timeout=90)


def test_rededup_cli_reuses_all_safe_members_and_original_energies_without_phase_search(
    rdkit_python: str, skeleton_ensemble: dict,
) -> None:
    parent_path = _completed_multi_member_postprocess(skeleton_ensemble, rdkit_python)
    output = skeleton_ensemble["root"] / "rededup"
    completed = _run_rededup(parent_path, rdkit_python, output)
    assert completed.returncode == 0, completed.stderr + completed.stdout
    manifest = json.loads(next(output.rglob("skeleton-dedup-manifest.json")).read_text())
    assert manifest["schema"] == "sam-phosphonate-skeleton-dedup-v1"
    assert manifest["parameters"]["skeleton_dedup_method"] == "energy-radius-symmetry"
    assert manifest["parameters"]["symmetry_automorphism_count"] >= 1
    assert manifest["summary"]["eligible_phase_safe_count"] == manifest["summary"]["processed_phase_safe_count"] == 3
    assert manifest["summary"]["parent_representative_count"] == 1
    assert manifest["summary"]["reused_source_outcome_count"] == 6
    assert manifest["summary"]["phase_recomputed_count"] == manifest["summary"]["optimization_started_count"] == 0
    assert manifest["representatives"][0]["representative_safe_rank"] == 5
    assert manifest["representatives"][0]["representative_source_force_field_energy_kcal_mol"] == -10.0
    assert "maximum_pairwise_axial_rmsd_A" not in manifest["representatives"][0]
    safe = [json.loads(line) for line in Path(manifest["artifacts"]["phase_safe_records"]).read_text().splitlines()]
    assert {record["safe_rank"] for record in safe} == {1, 5, 6}
    assert all(record["representative_safe_rank"] == 5 for record in safe)
    assert all(record["assigned_representative_rmsd_A"] <= 1.0 + 1.0e-12 for record in safe)
    assert all(not record["pose_force_rechecked"] for record in safe)
    assert all(Path(record["pose_structure"]).is_relative_to(parent_path.parent) for record in safe)
    assert len(list(output.rglob("*.extxyz"))) == 1  # Only the new representative is copied.


def test_rededup_cli_partial_limit_and_changed_pose_hash_rejection(
    rdkit_python: str, skeleton_ensemble: dict,
) -> None:
    parent_path = _completed_multi_member_postprocess(skeleton_ensemble, rdkit_python)
    output = skeleton_ensemble["root"] / "rededup-partial"
    completed = _run_rededup(parent_path, rdkit_python, output, limit=1)
    assert completed.returncode == 0, completed.stderr + completed.stdout
    manifest = json.loads(next(output.rglob("skeleton-dedup-manifest.json")).read_text())
    assert manifest["status"] == "partial_completed" and not manifest["grid_complete"]
    assert manifest["summary"]["eligible_phase_safe_count"] == 3
    assert manifest["summary"]["processed_phase_safe_count"] == 1
    parent = json.loads(parent_path.read_text())
    safe = [json.loads(line) for line in Path(parent["artifacts"]["phase_safe_records"]).read_text().splitlines()]
    changed = Path(safe[-1]["pose_structure"])
    changed.write_text(changed.read_text() + "\n# changed\n")
    failed_output = skeleton_ensemble["root"] / "rededup-hash-invalid"
    failed = _run_rededup(parent_path, rdkit_python, failed_output, limit=1)
    assert failed.returncode == 2 and "hash changed" in failed.stderr
    assert not failed_output.exists()


def test_rededup_cli_rejects_different_atom_mapping_and_mixed_force_field_variants(
    rdkit_python: str, skeleton_ensemble: dict,
) -> None:
    parent_path = _completed_multi_member_postprocess(skeleton_ensemble, rdkit_python)
    parent = json.loads(parent_path.read_text())
    parent["system"]["rmsd_atom_ids_1based"] = list(reversed(parent["system"]["rmsd_atom_ids_1based"]))
    parent_path.write_text(json.dumps(parent))
    output = skeleton_ensemble["root"] / "rededup-mapping-invalid"
    failed = _run_rededup(parent_path, rdkit_python, output)
    assert failed.returncode == 2 and "atom mapping changed" in failed.stderr
    assert not output.exists()
    sys.path.insert(0, str(ROOT / "scripts"))
    import fast_ff_prefilter as ff

    record = {"force_field": "mmff94s", "force_field_actual_variant": "MMFF94",
              "force_field_energy_kcal_mol": 10.0}
    manifest = {"parameters": {"force_field": "mmff94s", "force_field_actual_variant": "MMFF94s"}}
    with pytest.raises(ValueError, match="same parent force field"):
        ff._validated_source_force_field_energy(record, manifest)
