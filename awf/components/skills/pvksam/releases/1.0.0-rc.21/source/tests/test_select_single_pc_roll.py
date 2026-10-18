from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest


SOURCE = Path(__file__).resolve().parents[1]
SCRIPT = SOURCE / "scripts" / "select_single_pc_roll.py"
RDKIT_PYTHON = Path("/home/software/anaconda3/envs/rdkit/bin/python")


@pytest.fixture(scope="module")
def rdkit_python() -> str:
    if not RDKIT_PYTHON.is_file():
        pytest.skip("the project RDKit interpreter is unavailable")
    check = subprocess.run(
        [str(RDKIT_PYTHON), "-c", "from rdkit.Chem import AllChem"],
        capture_output=True,
        text=True,
        check=False,
    )
    if check.returncode:
        pytest.skip("the project RDKit interpreter cannot import RDKit")
    return str(RDKIT_PYTHON)


def _input_case(tmp_path: Path, interpreter: str) -> tuple[Path, Path, Path]:
    template = tmp_path / "phosphonate-acid.sdf"
    ensemble = tmp_path / "candidates.json"
    cpu_manifest = tmp_path / "cpu-manifest.json"
    builder = r'''
import hashlib, json, math, sys
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
sys.path.insert(0, sys.argv[1])
import fast_ff_prefilter as ff
root = Path(sys.argv[2])
acid = Chem.AddHs(Chem.MolFromSmiles("CCP(=O)(O)O"))
assert AllChem.EmbedMolecule(acid, randomSeed=20417) == 0
Chem.MolToMolFile(acid, str(root / "phosphonate-acid.sdf"))
symbols = [atom.GetSymbol() for atom in acid.GetAtoms()]
h0_symbols = [symbol for i, symbol in enumerate(symbols)
              if not (symbol == "H" and acid.GetAtomWithIdx(i).GetNeighbors()[0].GetSymbol() == "O"
                      and acid.GetAtomWithIdx(i).GetNeighbors()[0].GetNeighbors()[0].GetSymbol() == "P")]
mol, head = ff._h0_rdkit_template(root / "phosphonate-acid.sdf", h0_symbols)
xyz = np.asarray(mol.GetConformer().GetPositions(), dtype=float)
p = next(atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == "P")
c = next(n.GetIdx() for n in mol.GetAtomWithIdx(p).GetNeighbors() if n.GetSymbol() == "C")
axis = xyz[c] - xyz[p]
axis /= np.linalg.norm(axis)
organic = [i for i, atom in enumerate(mol.GetAtoms()) if atom.GetSymbol() not in {"P", "O"}]
rows = []
cpu = []
for skeleton, angles in (("family-a", [0, 90, 180]), ("family-b", [30, 150])):
    for angle in angles:
        theta = math.radians(angle)
        cross = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
        rotation = np.eye(3) * math.cos(theta) + (1 - math.cos(theta)) * np.outer(axis, axis) + math.sin(theta) * cross
        positions = xyz.copy()
        positions[organic] = (xyz[organic] - xyz[p]) @ rotation.T + xyz[p]
        candidate_id = f"{skeleton}__pc-roll-{angle:03d}deg"
        rows.append({"candidate_id": candidate_id, "positions_A": positions.tolist()})
        cpu.append({"candidate_id": candidate_id,
                    "status": "rejected_surface_collision" if (skeleton == "family-a" and angle == 180)
                              or (skeleton == "family-b" and angle == 150)
                              else "passed_cpu_collision_screen"})
(root / "candidates.json").write_text(json.dumps({"symbols": h0_symbols, "records": rows}))
candidate_path = root / "candidates.json"
candidate_hash = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
(root / "cpu-manifest.json").write_text(json.dumps({
    "parameters": {"candidate_ensemble": {"path": str(candidate_path), "sha256": candidate_hash}},
    "records": cpu}))
'''
    completed = subprocess.run(
        [interpreter, "-c", builder, str(SOURCE / "scripts"), str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    return template, ensemble, cpu_manifest


def _run(interpreter: str, template: Path, ensemble: Path, cpu: Path, output: Path):
    return subprocess.run(
        [
            interpreter,
            str(SCRIPT),
            "--template-sdf", str(template),
            "--candidate-ensemble", str(ensemble),
            "--cpu-manifest", str(cpu),
            "--output-dir", str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
    )


def test_selects_one_lowest_energy_surface_safe_roll_per_skeleton(
    tmp_path: Path, rdkit_python: str, monkeypatch
) -> None:
    template, ensemble, cpu = _input_case(tmp_path, rdkit_python)
    output = tmp_path / "selected"
    completed = _run(rdkit_python, template, ensemble, cpu, output)
    assert completed.returncode == 0, completed.stderr + completed.stdout

    manifest = json.loads((output / "single-pc-roll-selection.json").read_text())
    selected = json.loads((output / "selected-candidate-ensemble.json").read_text())
    assert manifest["summary"] == {
        "skeleton_count": 2,
        "input_roll_count": 5,
        "surface_safe_roll_count": 3,
        "screen_rejected_roll_count": 2,
        "selected_roll_count": 2,
        "surface_safe_roll_count_min": 1,
        "surface_safe_roll_count_max": 2,
    }
    assert len(selected["records"]) == 2
    chosen = {row["skeleton_id"]: row for row in manifest["selected"]}
    for skeleton_id in ("family-a", "family-b"):
        safe = [
            row for row in manifest["all_surface_safe_roll_energies"]
            if row["skeleton_id"] == skeleton_id
        ]
        assert chosen[skeleton_id]["mmff94s_single_point_energy_kcal_mol"] == min(
            row["mmff94s_single_point_energy_kcal_mol"] for row in safe
        )
    assert manifest["method"]["coordinates_optimized_in_this_stage"] is False
    assert manifest["sources"]["candidate_ensemble_sha256"]
    # Consume the actual selector artifact, rather than a handcrafted receipt.
    # Geometry gates are covered elsewhere; this isolates inter-stage contracts.
    import numpy as np
    from ase import Atoms
    from types import SimpleNamespace
    import mace_conformation_scan as scan
    import sam_structure_tools as geometry
    from substrate_library import _molecular_adjacency
    ep=output/'selected-candidate-ensemble.json'
    scan._validate_single_pc_roll_selection(ep,selected)
    mol=Atoms(selected['symbols'],positions=selected['records'][0]['positions_A'],cell=[100,100,100],pbc=[True,True,False])
    head=manifest['method']['fixed_head_atom_indices_0based']
    graph=_molecular_adjacency(mol.get_chemical_symbols(),mol.positions,np.asarray(mol.cell),np.asarray(mol.pbc))
    substrate=Atoms('In',positions=[[50,50,0]],cell=mol.cell,pbc=mol.pbc)
    prepared={'sam':mol,'headgroup_atom_ids_0based':head,'headgroup_target_positions_A':mol.positions[head].copy(),
        'substrate_with_h':substrate,'molecular_adjacency':graph,'mapped_windows':{},'radii_A':{},
        'surface_frame':{'periodic_fractional_axes':[0,1]},'initial_collision_audit':{'passed':True},
        'metadata':{'substrate_atom_count':1,'surface_h_count':0,'sam_atom_count':len(mol),'surface_protons':[]}}
    monkeypatch.setattr(scan,'_prepare_fixed_site_system',lambda args:prepared)
    monkeypatch.setattr(scan,'_candidate_anchor_height_audit',lambda *a,**kw:{'passed':True})
    monkeypatch.setattr(scan,'_intramolecular_collision_audit',lambda *a,**kw:{'passed':True})
    monkeypatch.setattr(geometry,'periodic_vdw_collision_audit',lambda **kw:{'passed':True})
    args=SimpleNamespace(candidate_ensemble=ep,output_root=tmp_path/'rescreen',donor_assignment='O1:1,O2:2,O3:3',
        site_prototype_id='synthetic-contract',site_cell=1,vdw_radius_scale=.85,headgroup_alignment_max_rmsd_A=.5,
        organic_z_floor_at_anchor_oxygen_mean=True)
    assert scan.run_fixed_site_cpu_screen(args)==0
    cp=next(args.output_root.glob('*/01_cpu_screen/manifest.json'))
    assert json.loads(cp.read_text())['summary']['collision_safe_count']==2
    monkeypatch.setattr(scan,'resolve_fixed_site_layer_constraints',lambda *a,**kw:{'synthetic':True})
    monkeypatch.setattr(scan,'_resolve_fixed_site_anchor_mapping',lambda *a,**kw:[])
    model=tmp_path/'placeholder.model';model.write_bytes(b'no calculator in planning')
    plan_args=SimpleNamespace(cpu_manifest=cp,model=model,generalized=True,formal_output_root=tmp_path/'mace',fmax=.03,max_steps=500,maxstep=.05)
    assert scan.run_fixed_site_optimization_plan(plan_args)==0
    plan=json.loads(next(plan_args.formal_output_root.glob('*/plan.json')).read_text())
    assert [r['candidate_id'] for r in plan['tasks']]==[r['candidate_id'] for r in selected['records']]
    manifest['selected_candidate_ensemble_sha256']='contradictory-legacy-hash'
    (output/'single-pc-roll-selection.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='conflicting'):
        scan._validate_single_pc_roll_selection(ep,selected)


def test_refuses_unmatched_cpu_manifest_and_preserves_no_result(
    tmp_path: Path, rdkit_python: str
) -> None:
    template, ensemble, cpu = _input_case(tmp_path, rdkit_python)
    manifest = json.loads(cpu.read_text())
    manifest["records"].pop()
    cpu.write_text(json.dumps(manifest))
    output = tmp_path / "bad-selection"
    completed = _run(rdkit_python, template, ensemble, cpu, output)
    assert completed.returncode == 2
    assert "candidate set differs" in completed.stderr
    assert not (output / "single-pc-roll-selection.json").exists()


def test_refuses_skeleton_with_no_surface_safe_roll(
    tmp_path: Path, rdkit_python: str
) -> None:
    template, ensemble, cpu = _input_case(tmp_path, rdkit_python)
    manifest = json.loads(cpu.read_text())
    for row in manifest["records"]:
        if row["candidate_id"].startswith("family-b__"):
            row["status"] = "rejected_height"
    cpu.write_text(json.dumps(manifest))
    output = tmp_path / "no-safe-roll"
    completed = _run(rdkit_python, template, ensemble, cpu, output)
    assert completed.returncode == 2
    assert "no surface-safe P-C roll remains" in completed.stderr
    assert not (output / "single-pc-roll-selection.json").exists()


def test_refuses_candidate_coordinates_changed_after_cpu_screen(
    tmp_path: Path, rdkit_python: str
) -> None:
    template, ensemble, cpu = _input_case(tmp_path, rdkit_python)
    payload = json.loads(ensemble.read_text())
    payload["records"][0]["positions_A"][0][0] += 0.01
    ensemble.write_text(json.dumps(payload))
    output = tmp_path / "changed-candidate"
    completed = _run(rdkit_python, template, ensemble, cpu, output)
    assert completed.returncode == 2
    assert "does not hash-bind this exact candidate ensemble" in completed.stderr
    assert not (output / "single-pc-roll-selection.json").exists()
