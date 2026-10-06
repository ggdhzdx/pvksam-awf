"""Shared LAMMPS ML-IAP helpers for generic SAM/substrate workflows."""

from __future__ import annotations

import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from ase.data import atomic_masses, atomic_numbers, covalent_radii
from ase.io import read, write


DEFAULT_SPECIES = ("H", "C", "N", "O", "P", "S", "In", "Sn")


def resolved_path(value: str | Path | None) -> Path | None:
    """Normalize a required or optional CLI path before changing cwd."""

    if value is None:
        return None
    return Path(value).expanduser().resolve()


def parse_species(value: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(value, str):
        items = tuple(item.strip() for item in value.replace(",", " ").split() if item.strip())
    else:
        items = tuple(value)
    if not items or any(item not in atomic_numbers for item in items):
        raise ValueError(f"Invalid species order: {items}")
    return items


def read_typed_structure(path: Path, species=DEFAULT_SPECIES):
    path = Path(path)
    species = parse_species(species)
    if path.suffix.lower() == ".data":
        z_of_type = {index: atomic_numbers[element] for index, element in enumerate(species, 1)}
        atoms = read(
            path,
            format="lammps-data",
            atom_style="atomic",
            Z_of_type=z_of_type,
        )
    elif path.suffix.lower() == ".cif":
        lines = path.read_text().splitlines(keepends=True)
        if any(line.startswith("_chemical_formula_structural") for line in lines):
            with TemporaryDirectory(prefix="sam_cif_") as directory:
                cleaned = Path(directory) / path.name
                cleaned.write_text(
                    "".join(
                        line
                        for line in lines
                        if not line.startswith("_chemical_formula_structural")
                    )
                )
                atoms = read(cleaned)
        else:
            atoms = read(path)
    else:
        atoms = read(path)
    atoms.set_pbc(True)
    return atoms


def write_lammps_data(atoms, path: Path, species=DEFAULT_SPECIES) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write(path, atoms, format="lammps-data", atom_style="atomic", specorder=list(parse_species(species)))


def write_clean_structure(atoms, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write(path, atoms)
    if path.suffix.lower() == ".cif":
        lines = path.read_text().splitlines(keepends=True)
        path.write_text(
            "".join(line for line in lines if not line.startswith("_chemical_formula_structural"))
        )


def validate_layout(atoms, substrate_atoms: int, molecules: int, atoms_per_molecule: int) -> np.ndarray:
    expected = substrate_atoms + molecules * atoms_per_molecule
    if len(atoms) != expected:
        raise ValueError(f"Structure has {len(atoms)} atoms; expected {expected}")
    return np.asarray(atoms.get_chemical_symbols())


def frozen_region(atoms, symbols: np.ndarray, substrate_atoms: int, depth: float) -> tuple[float, int]:
    z_max, atom_ids = frozen_atom_ids(atoms, symbols, substrate_atoms, depth)
    return z_max, len(atom_ids)


def frozen_atom_ids(
    atoms,
    symbols: np.ndarray,
    substrate_atoms: int,
    depth: float,
) -> tuple[float, list[int]]:
    """Select the frozen substrate atoms once and return 1-based IDs."""

    indices = np.arange(substrate_atoms)
    heavy = indices[symbols[indices] != "H"]
    if not len(heavy):
        raise ValueError("Substrate contains no heavy atoms")
    z_max = float(np.min(atoms.positions[heavy, 2]) + depth)
    selected = indices[atoms.positions[indices, 2] <= z_max]
    if not len(selected):
        raise ValueError("Dynamic frozen region contains no atoms")
    return z_max, [int(index) + 1 for index in selected]


def read_frozen_atom_ids(path: Path, substrate_atoms: int) -> list[int]:
    """Read and validate a stable, 1-based substrate atom-ID selection."""

    tokens = []
    for line in path.read_text().splitlines():
        tokens.extend(line.split("#", 1)[0].replace(",", " ").split())
    try:
        atom_ids = [int(token) for token in tokens]
    except ValueError as error:
        raise ValueError(f"Invalid atom ID in {path}") from error
    if not atom_ids:
        raise ValueError(f"Frozen atom-ID file is empty: {path}")
    if len(atom_ids) != len(set(atom_ids)):
        raise ValueError(f"Frozen atom-ID file contains duplicates: {path}")
    invalid = [atom_id for atom_id in atom_ids if not 1 <= atom_id <= substrate_atoms]
    if invalid:
        raise ValueError(
            f"Frozen atom IDs must lie in 1..{substrate_atoms}; invalid={invalid[:10]}"
        )
    return sorted(atom_ids)


def group_id_block(group_name: str, atom_ids: list[int], chunk_size: int = 32) -> list[str]:
    """Create a continued LAMMPS ``group ... id`` command."""

    if not atom_ids:
        raise ValueError("A frozen atom-ID group cannot be empty")
    lines = [f"group {group_name} id &"]
    for start in range(0, len(atom_ids), chunk_size):
        chunk = atom_ids[start : start + chunk_size]
        continuation = " &" if start + chunk_size < len(atom_ids) else ""
        lines.append("    " + " ".join(map(str, chunk)) + continuation)
    return lines


def anchor_bonds(
    atoms,
    symbols: np.ndarray,
    substrate_atoms: int,
    molecules: int,
    atoms_per_molecule: int,
    anchor_element: str = "P",
    neighbor_element: str = "C",
    maximum: float = 2.15,
) -> list[tuple[int, int, float]]:
    result = []
    for molecule in range(molecules):
        start = substrate_atoms + molecule * atoms_per_molecule
        indices = np.arange(start, start + atoms_per_molecule)
        anchors = indices[symbols[indices] == anchor_element]
        neighbors = indices[symbols[indices] == neighbor_element]
        if len(anchors) != 1 or not len(neighbors):
            raise ValueError(
                f"Molecule {molecule + 1}: {anchor_element}={len(anchors)}, "
                f"{neighbor_element}={len(neighbors)}"
            )
        anchor = int(anchors[0])
        distances = atoms.get_distances(anchor, neighbors, mic=True)
        position = int(np.argmin(distances))
        neighbor = int(neighbors[position])
        distance = float(distances[position])
        if distance >= maximum:
            raise ValueError(
                f"Molecule {molecule + 1} anchor bond is {distance:.4f} A"
            )
        result.append((anchor + 1, neighbor + 1, distance))
    return result


def parent_h_bonds(
    atoms,
    symbols: np.ndarray,
    substrate_atoms: int,
    expected: int,
    parent_element: str = "O",
    maximum: float = 1.25,
) -> list[tuple[int, int, float]]:
    indices = np.arange(substrate_atoms)
    hydrogens = indices[symbols[indices] == "H"]
    parents = indices[symbols[indices] == parent_element]
    result = []
    for hydrogen_raw in hydrogens:
        hydrogen = int(hydrogen_raw)
        distances = atoms.get_distances(hydrogen, parents, mic=True)
        position = int(np.argmin(distances))
        parent = int(parents[position])
        distance = float(distances[position])
        if distance >= maximum:
            raise ValueError(f"Substrate H {hydrogen + 1} has no valid parent")
        result.append((parent + 1, hydrogen + 1, distance))
    if len(result) != expected or len({item[0] for item in result}) != expected:
        raise ValueError(
            f"Expected {expected} unique parent-H bonds; got {len(result)} bonds and "
            f"{len({item[0] for item in result})} unique parents"
        )
    return result


def covalent_bond_cutoff(element_a: str, element_b: str) -> float:
    overrides = {
        frozenset(("C", "H")): 1.25,
        frozenset(("N", "H")): 1.25,
        frozenset(("O", "H")): 1.25,
        frozenset(("C", "C")): 1.75,
        frozenset(("C", "N")): 1.70,
        frozenset(("C", "S")): 2.05,
        frozenset(("C", "P")): 2.15,
        frozenset(("P", "O")): 1.85,
    }
    pair = frozenset((element_a, element_b))
    if pair in overrides:
        return overrides[pair]
    return min(
        2.20,
        1.20
        * float(
            covalent_radii[atomic_numbers[element_a]]
            + covalent_radii[atomic_numbers[element_b]]
        ),
    )


# Backward-compatible private alias for older workspace callers.
_bond_cutoff = covalent_bond_cutoff


def molecular_bonds(
    atoms,
    symbols: np.ndarray,
    substrate_atoms: int,
    molecules: int,
    atoms_per_molecule: int,
) -> list[tuple[int, int, float, bool]]:
    """Return within-block covalent bonds as 1-based IDs and an H-bond flag."""

    result = []
    for molecule in range(molecules):
        start = substrate_atoms + molecule * atoms_per_molecule
        indices = np.arange(start, start + atoms_per_molecule)
        for offset, atom_i_raw in enumerate(indices):
            atom_i = int(atom_i_raw)
            later = indices[offset + 1 :]
            if not len(later):
                continue
            distances = atoms.get_distances(atom_i, later, mic=True)
            for atom_j_raw, distance_raw in zip(later, distances):
                atom_j = int(atom_j_raw)
                distance = float(distance_raw)
                if distance < covalent_bond_cutoff(symbols[atom_i], symbols[atom_j]):
                    result.append(
                        (
                            atom_i + 1,
                            atom_j + 1,
                            distance,
                            "H" in (symbols[atom_i], symbols[atom_j]),
                        )
                    )
    if not result:
        raise ValueError("No molecular covalent bonds were detected")
    return result


def mass_lines(species=DEFAULT_SPECIES, hmr_mass: float | None = None) -> list[str]:
    lines = []
    for index, element in enumerate(parse_species(species), 1):
        mass = hmr_mass if element == "H" and hmr_mass is not None else atomic_masses[atomic_numbers[element]]
        lines.append(f"mass {index} {float(mass):.8f}")
    return lines


def restrain_block(
    fix_name: str,
    group: str,
    bonds: list[tuple[int, int, float]],
    spring: float | None = None,
) -> list[str]:
    if not bonds:
        return []
    lines = [f"fix {fix_name} {group} restrain &"]
    for position, (atom_a, atom_b, target) in enumerate(bonds):
        if spring is None:
            raise ValueError("A spring constant is required")
        continuation = " &" if position < len(bonds) - 1 else ""
        lines.append(
            f"    bond {atom_a} {atom_b} {spring:.8g} {spring:.8g} "
            f"{target:.8g} {target:.8g}{continuation}"
        )
    return lines


def mixed_restrain_block(
    fix_name: str,
    group: str,
    bonds: list[tuple[int, int, float, float]],
) -> list[str]:
    if not bonds:
        return []
    lines = [f"fix {fix_name} {group} restrain &"]
    for position, (atom_a, atom_b, spring, target) in enumerate(bonds):
        continuation = " &" if position < len(bonds) - 1 else ""
        lines.append(
            f"    bond {atom_a} {atom_b} {spring:.8g} {spring:.8g} "
            f"{target:.8g} {target:.8g}{continuation}"
        )
    return lines


def physical_write_data_lines(path: Path, species=DEFAULT_SPECIES) -> list[str]:
    """Restore physical element masses before persisting a LAMMPS data file.

    Hydrogen mass repartitioning is an integration aid, not a change of
    chemical identity.  Leaving the HMR mass in a data file can make readers
    infer helium from the approximately 4 u mass.
    """

    return [
        "# Restore physical masses so HMR does not leak into structural output",
        *mass_lines(species),
        f"write_data {path}",
    ]


class ThermoStepParser:
    """Extract timesteps only from LAMMPS thermo tables.

    LAMMPS writes many unrelated lines that begin with integers (for example,
    ``5632 atoms`` and neighbor-list summaries).  A numeric token is therefore
    accepted only after a ``Step ...`` header and only when the complete row is
    numeric and has the same number of columns as that header.
    """

    def __init__(self) -> None:
        self.column_count: int | None = None

    def feed(self, line: str) -> int | None:
        fields = line.split()
        if fields and fields[0] == "Step":
            self.column_count = len(fields)
            return None
        if self.column_count is None or len(fields) != self.column_count:
            return None
        try:
            step = int(fields[0])
            for value in fields[1:]:
                float(value)
        except ValueError:
            return None
        return step


def run_lammps(
    executable: str,
    input_path: Path,
    output_log: Path,
    progress_path: Path | None = None,
    expected_steps: int | None = None,
    cwd: Path | None = None,
) -> None:
    output_log.parent.mkdir(parents=True, exist_ok=True)
    thermo_steps = ThermoStepParser()
    process = subprocess.Popen(
        [executable, "-in", str(input_path)],
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    with output_log.open("w") as handle, process.stdout:
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            print(line, end="", flush=True)
            if progress_path and expected_steps:
                step = thermo_steps.feed(line)
                if step is not None and 0 <= step <= expected_steps:
                    progress_path.parent.mkdir(parents=True, exist_ok=True)
                    progress_path.write_text(
                        f"LAMMPS step {step}/{expected_steps} "
                        f"({100.0 * step / expected_steps:.1f}%)\n"
                    )
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"LAMMPS exited with code {return_code}; see {output_log}")


def final_energy(log_path: Path) -> float:
    pattern = re.compile(r"FINAL_PE\s+([-+0-9.eE]+)")
    for line in reversed(log_path.read_text(errors="replace").splitlines()):
        match = pattern.search(line)
        if match:
            return float(match.group(1))
    raise ValueError(f"No FINAL_PE record in {log_path}")


def minimization_summary(log_path: Path) -> dict:
    """Extract the final LAMMPS minimization outcome for audit records."""

    patterns = {
        "stopping_criterion": re.compile(r"Stopping criterion\s*=\s*(.+?)\s*$"),
        "force_two_norm": re.compile(
            r"Force two-norm initial, final\s*=\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)"
        ),
        "force_max_component": re.compile(
            r"Force max component initial, final\s*=\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)"
        ),
        "iterations": re.compile(
            r"Iterations, force evaluations\s*=\s*(\d+)\s+(\d+)"
        ),
    }
    summary: dict[str, object] = {}
    for line in log_path.read_text(errors="replace").splitlines():
        if match := patterns["stopping_criterion"].search(line):
            summary["stopping_criterion"] = match.group(1)
        elif match := patterns["force_two_norm"].search(line):
            summary["initial_force_two_norm_eV_per_A"] = float(match.group(1))
            summary["final_force_two_norm_eV_per_A"] = float(match.group(2))
        elif match := patterns["force_max_component"].search(line):
            summary["initial_force_max_component_eV_per_A"] = float(match.group(1))
            summary["final_force_max_component_eV_per_A"] = float(match.group(2))
        elif match := patterns["iterations"].search(line):
            summary["iterations"] = int(match.group(1))
            summary["force_evaluations"] = int(match.group(2))
    if "stopping_criterion" not in summary:
        raise ValueError(f"No minimization stopping criterion in {log_path}")
    summary["force_converged"] = str(summary["stopping_criterion"]).startswith(
        "force tolerance"
    )
    return summary


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def composition(atoms) -> dict[str, int]:
    return dict(sorted(Counter(atoms.get_chemical_symbols()).items()))
