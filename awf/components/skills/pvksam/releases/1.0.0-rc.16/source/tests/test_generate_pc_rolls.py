from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

from ase import Atoms
from ase.io import read, write
import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1]
SCRIPT = SOURCE / "scripts" / "generate_pc_rolls.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _postprocess_case(tmp_path: Path) -> Path:
    representatives = []
    for cluster_index, shift in ((1, 0.0), (2, 2.0)):
        path = tmp_path / f"cluster-{cluster_index:06d}.extxyz"
        atoms = Atoms(
            symbols=["P", "O", "O", "O", "C", "H"],
            positions=[
                [0.0, 0.0, 1.0 + shift],
                [1.0, 0.0, 0.0 + shift],
                [-0.5, 0.8, 0.0 + shift],
                [-0.5, -0.8, 0.0 + shift],
                [0.0, 0.0, 2.0 + shift],
                [1.0, 0.0, 2.0 + shift],
            ],
        )
        write(path, atoms, format="extxyz")
        representatives.append(
            {
                "cluster_index": cluster_index,
                "representative_structure": str(path),
                "representative_structure_sha256": _sha256(path),
            }
        )
    manifest = {
        "schema": "sam-phosphonate-skeleton-postprocess-v1",
        "run_identity": "synthetic-parent-identity",
        "status": "completed",
        "grid_complete": True,
        "records_scope": "post_optimization_skeleton_geometry_all_outcomes",
        "system": {
            "p_atom_id_1based": 1,
            "c_atom_id_1based": 5,
            "organic_atom_ids_1based": [5, 6],
        },
        "representatives": representatives,
    }
    path = tmp_path / "skeleton-postprocess-manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def _run(manifest: Path, output: Path, step: int = 90) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--skeleton-postprocess-manifest", str(manifest),
            "--output-dir", str(output),
            "--step-deg", str(step),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def test_generates_complete_periodic_roll_grid_and_keeps_head_fixed(tmp_path: Path) -> None:
    source = _postprocess_case(tmp_path)
    output = tmp_path / "rolls"
    completed = _run(source, output)
    assert completed.returncode == 0, completed.stderr + completed.stdout
    manifest = json.loads((output / "pc-roll-generation.json").read_text())
    ensemble = json.loads((output / "pc-roll-candidate-ensemble.json").read_text())
    assert manifest["summary"] == {
        "skeleton_count": 2,
        "phase_count_per_skeleton": 4,
        "candidate_count": 8,
    }
    assert manifest["method"]["angles_deg"] == [0.0, 90.0, 180.0, 270.0]
    assert len(ensemble["records"]) == 8
    by_skeleton: dict[str, list[dict]] = {}
    for row in ensemble["records"]:
        by_skeleton.setdefault(row["skeleton_id"], []).append(row)
    for rows in by_skeleton.values():
        positions = [np.asarray(row["positions_A"]) for row in rows]
        assert all(np.array_equal(value[:4], positions[0][:4]) for value in positions)
        assert len({tuple(np.round(value[5], 8)) for value in positions}) == 4
    assert manifest["output"]["candidate_ensemble_sha256"] == _sha256(
        output / "pc-roll-candidate-ensemble.json"
    )


def test_rejects_nonperiodic_grid_step_and_changed_representative(tmp_path: Path) -> None:
    source = _postprocess_case(tmp_path)
    invalid_step = _run(source, tmp_path / "bad-step", step=100)
    assert invalid_step.returncode == 2
    assert "must divide the 360-degree circle" in invalid_step.stderr

    manifest = json.loads(source.read_text())
    structure = Path(manifest["representatives"][0]["representative_structure"])
    structure.write_text(structure.read_text() + "# changed\n")
    changed = _run(source, tmp_path / "bad-hash")
    assert changed.returncode == 2
    assert "missing or changed" in changed.stderr
