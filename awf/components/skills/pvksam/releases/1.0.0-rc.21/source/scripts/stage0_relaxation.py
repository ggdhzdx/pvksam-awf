#!/usr/bin/env python3
"""Prepare restrained LAMMPS Stage 0 inputs and convert their outputs.

The structure contract is substrate-first followed by contiguous equal-size SAM
blocks.  Counts, chemistry labels, restraint parameters, MACE model, species order,
and bottom-layer depth are explicit parameters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from ase.data import atomic_masses, atomic_numbers
from ase.io import read, write

from sam_lammps import group_id_block, read_frozen_atom_ids


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_protonated_parent(structure: Path, protonation_manifest_path: Path) -> dict:
    """Verify the protonated structure and its physical-promotion ancestry."""

    structure = Path(structure).expanduser().resolve()
    protonation_manifest_path = Path(protonation_manifest_path).expanduser().resolve()
    try:
        manifest = json.loads(protonation_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("cannot read parent protonation manifest") from exc
    is_legacy = manifest.get("algorithm") == (
        "post-SAM global proton-group multi-acceptor optimization"
    )
    is_global_v1 = (
        manifest.get("schema") == "samflow-global-surface-protonation-v1"
        and manifest.get("schema_version") == 1
    )
    if not is_legacy and not is_global_v1:
        raise ValueError("unsupported parent protonation manifest")
    if is_global_v1 and manifest.get("status") != "passed_global_surface_protonation":
        raise ValueError("global surface protonation did not pass")
    promotion = manifest.get("parent_physical_interface_promotion")
    if not isinstance(promotion, dict) or promotion.get(
        "physical_interface_pass"
    ) is not True:
        raise ValueError("protonation parent lacks a passed physical-interface ancestry")
    basis_name = (
        "promotion_basis_sha256"
        if "promotion_basis_sha256" in promotion
        else "approval_plan_sha256"
    )
    for name in ("manifest_sha256", basis_name):
        value = promotion.get(name)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"protonation parent has an invalid {name}")
    if is_global_v1:
        promotion_path = Path(str(promotion.get("manifest_path", ""))).expanduser()
        if not promotion_path.is_absolute():
            promotion_path = protonation_manifest_path.parent / promotion_path
        promotion_path = promotion_path.resolve()
        if not promotion_path.is_file() or _sha256(promotion_path) != promotion[
            "manifest_sha256"
        ]:
            raise ValueError("physical-interface promotion manifest hash mismatch")
        try:
            promotion_manifest = json.loads(
                promotion_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("cannot read physical-interface promotion manifest") from exc
        if (
            promotion_manifest.get("schema")
            not in {
                "samflow-sized-h0-physical-interface-promotion-v1",
                "samflow-geometric-h0-physical-interface-promotion-v1",
            }
            or promotion_manifest.get("physical_interface_pass") is not True
            or promotion_manifest.get("promotion_eligible") is not True
        ):
            raise ValueError("physical-interface promotion ancestry did not pass")
        requirements_record = manifest.get("protonation_requirements")
        if not isinstance(requirements_record, dict):
            raise ValueError("global protonation manifest lacks proton requirements")
        requirements_path = Path(
            str(requirements_record.get("path", ""))
        ).expanduser()
        if not requirements_path.is_absolute():
            requirements_path = protonation_manifest_path.parent / requirements_path
        requirements_path = requirements_path.resolve()
        if not requirements_path.is_file() or _sha256(requirements_path) != (
            requirements_record.get("sha256")
        ):
            raise ValueError("surface-proton requirements hash mismatch")
        try:
            requirements = json.loads(requirements_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("cannot read surface-proton requirements") from exc
        if requirements.get("schema") != "samflow-surface-proton-requirements-v1":
            raise ValueError("unsupported surface-proton requirements")
        inventory = {
            "input_substrate_atom_count": manifest.get("input_substrate_atom_count"),
            "protonated_substrate_atom_count": manifest.get(
                "protonated_substrate_atom_count"
            ),
            "SAM_count": manifest.get("SAM_count"),
            "atoms_per_SAM": manifest.get("atoms_per_SAM"),
            "surface_H_count": manifest.get("surface_H_count"),
            "surface_H_per_SAM": manifest.get("surface_H_per_SAM"),
            "required_surface_proton_count": requirements.get(
                "required_surface_proton_count"
            ),
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in inventory.values()
        ) or not (
            inventory["surface_H_count"]
            == inventory["SAM_count"] * inventory["surface_H_per_SAM"]
            == inventory["required_surface_proton_count"]
            and inventory["protonated_substrate_atom_count"]
            == inventory["input_substrate_atom_count"]
            + inventory["surface_H_count"]
        ):
            raise ValueError("global surface-proton inventory is inconsistent")

    if is_global_v1:
        structure_record = manifest.get("structure")
        if not isinstance(structure_record, dict):
            raise ValueError("global protonation manifest lacks its structure record")
        registered_path = Path(str(structure_record.get("path", ""))).expanduser()
        if not registered_path.is_absolute():
            registered_path = protonation_manifest_path.parent / registered_path
        candidate_records = [
            (str(registered_path.resolve()), structure_record.get("sha256"))
        ]
    else:
        candidate_records = [
            (manifest.get("output_cif"), manifest.get("output_cif_sha256")),
            (manifest.get("output_data"), manifest.get("output_data_sha256")),
        ]
    matching_records = [
        (Path(path).expanduser().resolve(), sha256)
        for path, sha256 in candidate_records
        if isinstance(path, str) and isinstance(sha256, str)
        and Path(path).expanduser().resolve() == structure
    ]
    if len(matching_records) != 1:
        raise ValueError("Stage 0 structure is not uniquely registered by protonation")
    if not structure.is_file() or _sha256(structure) != matching_records[0][1]:
        raise ValueError("protonated Stage 0 structure hash mismatch")
    return manifest

def read_structure(path: Path):
    if path.suffix.lower() == ".data":
        return read(path, format="lammps-data", atom_style="atomic")
    return read(path)


def species_list(value: str) -> list[str]:
    species = [item.strip() for item in value.split(",") if item.strip()]
    if not species:
        raise argparse.ArgumentTypeError("species list cannot be empty")
    unknown = [item for item in species if item not in atomic_numbers]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown elements: {unknown}")
    return species


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return number


def generate_input(args: argparse.Namespace) -> int:
    protonation_parent = load_protonated_parent(
        args.structure, args.parent_protonation_manifest
    )
    if protonation_parent.get("schema") == "samflow-global-surface-protonation-v1":
        layout_contract = {
            "substrate-atoms": (
                args.substrate_atoms,
                protonation_parent.get("protonated_substrate_atom_count"),
            ),
            "molecules": (args.molecules, protonation_parent.get("SAM_count")),
            "atoms-per-molecule": (
                args.atoms_per_molecule,
                protonation_parent.get("atoms_per_SAM"),
            ),
            "protons-per-molecule": (
                args.protons_per_molecule,
                protonation_parent.get("surface_H_per_SAM"),
            ),
        }
        for name, (cli_value, manifest_value) in layout_contract.items():
            if cli_value != manifest_value:
                raise ValueError(
                    f"Stage 0 --{name} disagrees with global protonation manifest: "
                    f"CLI={cli_value}, manifest={manifest_value}"
                )
    atoms = read_structure(args.structure)
    symbols = np.asarray(atoms.get_chemical_symbols())
    expected_atoms = args.substrate_atoms + args.molecules * args.atoms_per_molecule
    if len(atoms) != expected_atoms:
        raise ValueError(f"Structure has {len(atoms)} atoms; expected {expected_atoms}")

    anchor_bonds = []
    for molecule in range(args.molecules):
        start = args.substrate_atoms + molecule * args.atoms_per_molecule
        stop = start + args.atoms_per_molecule
        indices = np.arange(start, stop)
        anchors = indices[symbols[indices] == args.anchor_element]
        neighbors = indices[symbols[indices] == args.anchor_neighbor_element]
        if len(anchors) != 1 or not len(neighbors):
            raise ValueError(
                f"Invalid molecule block {molecule + 1}: "
                f"{args.anchor_element}={len(anchors)}, "
                f"{args.anchor_neighbor_element}={len(neighbors)}"
            )
        anchor_index = int(anchors[0])
        distances = atoms.get_distances(anchor_index, neighbors, mic=True)
        nearest = int(np.argmin(distances))
        neighbor_index = int(neighbors[nearest])
        distance = float(distances[nearest])
        if distance >= args.anchor_neighbor_max:
            raise ValueError(
                f"Molecule {molecule + 1} {args.anchor_element}-"
                f"{args.anchor_neighbor_element} bond is {distance:.4f} A"
            )
        anchor_bonds.append((anchor_index + 1, neighbor_index + 1, distance))

    substrate_indices = np.arange(args.substrate_atoms)
    substrate_h = substrate_indices[symbols[substrate_indices] == "H"]
    substrate_parents = substrate_indices[
        symbols[substrate_indices] == args.parent_element
    ]
    parent_h_bonds = []
    for h_index in substrate_h:
        distances = atoms.get_distances(int(h_index), substrate_parents, mic=True)
        nearest = int(np.argmin(distances))
        parent_index = int(substrate_parents[nearest])
        distance = float(distances[nearest])
        if distance >= args.parent_h_max:
            raise ValueError(
                f"Substrate H {h_index + 1} lacks a valid "
                f"{args.parent_element} parent"
            )
        parent_h_bonds.append((parent_index + 1, int(h_index) + 1, distance))
    expected_h = args.molecules * args.protons_per_molecule
    if (
        len(parent_h_bonds) != expected_h
        or len({parent_id for parent_id, _, _ in parent_h_bonds}) != expected_h
    ):
        raise ValueError(
            f"Expected {expected_h} unique substrate parent-H bonds; "
            f"found {len(parent_h_bonds)} bonds and "
            f"{len({item[0] for item in parent_h_bonds})} unique parents"
        )

    if args.frozen_atom_ids_file:
        frozen_ids = read_frozen_atom_ids(
            args.frozen_atom_ids_file, args.substrate_atoms
        )
        frozen_z_max = float(np.max(atoms.positions[np.asarray(frozen_ids) - 1, 2]))
        frozen_group_lines = group_id_block("frozen_atoms", frozen_ids)
        frozen_mode = "atom_ids"
    else:
        substrate_heavy_z = atoms.positions[
            substrate_indices[symbols[substrate_indices] != "H"], 2
        ]
        frozen_z_max = float(np.min(substrate_heavy_z) + args.freeze_depth)
        frozen_ids = [
            int(index) + 1
            for index in substrate_indices[
                atoms.positions[substrate_indices, 2] <= frozen_z_max
            ]
        ]
        if not frozen_ids:
            raise ValueError("Automatic bottom-layer region contains no substrate atoms")
        frozen_group_lines = [
            f"region frozen_bottom block INF INF INF INF INF {frozen_z_max:.6f} units box",
            "group frozen_atoms region frozen_bottom",
        ]
        frozen_mode = "coordinate_region"
    frozen_count = len(frozen_ids)

    prevent_pairs = []
    for specification in args.prevent_transfer:
        h_text, atom_text = specification.split(":", 1)
        h_id, atom_id = int(h_text), int(atom_text)
        distance = float(atoms.get_distance(h_id - 1, atom_id - 1, mic=True))
        prevent_pairs.append((h_id, atom_id, distance))

    mass_lines = [
        f"mass {index} {atomic_masses[atomic_numbers[element]]:.8f}"
        for index, element in enumerate(args.species, 1)
    ]
    lines = [
        "# Generic restrained SAM/substrate Stage 0.",
        (
            "# parent_protonation_manifest_sha256="
            f"{_sha256(args.parent_protonation_manifest)}"
        ),
        (
            "# "
            + (
                "promotion_basis_sha256="
                if "promotion_basis_sha256"
                in protonation_parent["parent_physical_interface_promotion"]
                else "approval_plan_sha256="
            )
            + str(
                protonation_parent["parent_physical_interface_promotion"].get(
                    "promotion_basis_sha256",
                    protonation_parent["parent_physical_interface_promotion"].get(
                        "approval_plan_sha256"
                    ),
                )
            )
        ),
        "units metal",
        "atom_style atomic",
        "atom_modify map yes",
        "dimension 3",
        "boundary p p p",
        "",
        f"read_data {args.data}",
        "",
        *mass_lines,
        "",
        f"pair_style mliap unified {args.model} 0",
        f"pair_coeff * * {' '.join(args.species)}",
        "",
        *frozen_group_lines,
        "group mobile_atoms subtract all frozen_atoms",
        "velocity frozen_atoms set 0.0 0.0 0.0",
        "",
        "fix protect_anchor mobile_atoms restrain &",
    ]
    for position, (anchor_id, neighbor_id, _) in enumerate(anchor_bonds):
        continuation = " &" if position < len(anchor_bonds) - 1 else ""
        lines.append(
            f"    bond {anchor_id} {neighbor_id} {args.anchor_k:.8g} "
            f"{args.anchor_k:.8g} {args.anchor_target:.8g} "
            f"{args.anchor_target:.8g}{continuation}"
        )
    lines.extend(["", "fix_modify protect_anchor energy yes", "", "fix preserve_parent_h mobile_atoms restrain &"])
    for position, (parent_id, h_id, _) in enumerate(parent_h_bonds):
        continuation = " &" if position < len(parent_h_bonds) - 1 else ""
        lines.append(
            f"    bond {parent_id} {h_id} {args.parent_h_k:.8g} "
            f"{args.parent_h_k:.8g} {args.parent_h_target:.8g} "
            f"{args.parent_h_target:.8g}{continuation}"
        )
    lines.extend(["", "fix_modify preserve_parent_h energy yes"])
    if prevent_pairs:
        lines.extend(["", "fix prevent_transfer mobile_atoms restrain &"])
        for position, (h_id, atom_id, _) in enumerate(prevent_pairs):
            continuation = " &" if position < len(prevent_pairs) - 1 else ""
            lines.append(
                f"    bond {h_id} {atom_id} {args.prevent_k:.8g} "
                f"{args.prevent_k:.8g} {args.prevent_target:.8g} "
                f"{args.prevent_target:.8g}{continuation}"
            )
        lines.extend(["", "fix_modify prevent_transfer energy yes"])
    # fix restrain can apply force to both atoms in each listed pair, including
    # frozen substrate endpoints. Keep setforce last so those forces are zeroed
    # after every restraint while the mobile endpoints retain their forces.
    lines.extend(
        [
            "",
            "fix hold_bottom frozen_atoms setforce 0.0 0.0 0.0",
        ]
    )
    lines.extend(
        [
            "",
            f"neighbor {args.neighbor_skin:.8g} bin",
            "neigh_modify delay 0 every 1 check yes",
            f"thermo {args.thermo_every}",
            "thermo_style custom step pe fnorm",
            "run 0",
            "",
        ]
    )
    if args.dump_every:
        dump_path = Path(args.result).with_suffix(".dump")
        lines.extend(
            [
                f"dump stage0_relax all custom {args.dump_every} {dump_path} "
                "id type x y z fx fy fz",
                "dump_modify stage0_relax sort id",
                "",
            ]
        )
    lines.extend(
        [
            f"min_style {args.min_style}",
            f"min_modify dmax {args.dmax:.8g}",
            f"minimize {args.etol:.8g} {args.ftol:.8g} "
            f"{args.maxiter} {args.maxeval}",
            "",
            *(["undump stage0_relax"] if args.dump_every else []),
            "unfix protect_anchor",
            "unfix preserve_parent_h",
            *(["unfix prevent_transfer"] if prevent_pairs else []),
            f"write_data {args.result}",
            "",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    bond_distances = np.asarray([item[2] for item in anchor_bonds])
    print(
        f"Wrote {args.output} with {len(anchor_bonds)} anchor bonds, "
        f"{len(parent_h_bonds)} parent-H bonds, and {len(prevent_pairs)} "
        f"anti-transfer restraints; initial anchor bond="
        f"{bond_distances.min():.4f}-{bond_distances.max():.4f} A; "
        f"freeze z<={frozen_z_max:.4f} A ({frozen_count} atoms)"
        f" using {frozen_mode}"
    )
    return 0


def convert_output(args: argparse.Namespace) -> int:
    z_of_type = {
        index: atomic_numbers[element]
        for index, element in enumerate(args.species, 1)
    }
    atoms = read(
        args.input,
        format="lammps-data",
        atom_style="atomic",
        Z_of_type=z_of_type,
    )
    if args.expected_atoms is not None and len(atoms) != args.expected_atoms:
        raise ValueError(
            f"Stage-0 output has {len(atoms)} atoms, expected {args.expected_atoms}"
        )
    atoms.set_pbc(True)
    write(args.output, atoms)
    if args.output.suffix.lower() == ".cif":
        lines = args.output.read_text().splitlines(keepends=True)
        args.output.write_text(
            "".join(
                line
                for line in lines
                if not line.startswith("_chemical_formula_structural")
            )
        )
    print(f"Wrote {len(atoms)} atoms to {args.output}")
    return 0


def add_generate_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("structure", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--parent-protonation-manifest",
        type=Path,
        required=True,
        help="hash-sealed protonation manifest with passed physical-interface ancestry",
    )
    parser.add_argument("--data", required=True, help="LAMMPS-visible input data path")
    parser.add_argument("--result", required=True, help="LAMMPS-visible result data path")
    parser.add_argument("--model", required=True, help="LAMMPS ML-IAP model path")
    parser.add_argument("--species", type=species_list, default=species_list("H,C,N,O,P,S,In,Sn"))
    parser.add_argument("--substrate-atoms", type=int, required=True)
    parser.add_argument("--molecules", type=int, required=True)
    parser.add_argument("--atoms-per-molecule", type=int, required=True)
    parser.add_argument("--protons-per-molecule", type=int, default=2)
    parser.add_argument("--anchor-element", default="P")
    parser.add_argument("--anchor-neighbor-element", default="C")
    parser.add_argument("--parent-element", default="O")
    parser.add_argument("--anchor-neighbor-max", type=float, default=2.15)
    parser.add_argument("--parent-h-max", type=float, default=1.25)
    parser.add_argument("--freeze-depth", type=float, default=1.10)
    parser.add_argument("--frozen-atom-ids-file", type=Path)
    parser.add_argument("--anchor-k", type=float, default=100.0)
    parser.add_argument("--anchor-target", type=float, default=1.82)
    parser.add_argument("--parent-h-k", type=float, default=100.0)
    parser.add_argument("--parent-h-target", type=float, default=0.98)
    parser.add_argument("--prevent-transfer", action="append", default=[], metavar="H_ID:ATOM_ID")
    parser.add_argument("--prevent-k", type=float, default=100.0)
    parser.add_argument("--prevent-target", type=float, default=1.80)
    parser.add_argument("--neighbor-skin", type=float, default=2.0)
    parser.add_argument("--thermo-every", type=int, default=5)
    parser.add_argument(
        "--dump-every",
        type=nonnegative_int,
        default=0,
        help="write custom id/type/position/force frames during minimization; 0 disables",
    )
    parser.add_argument("--min-style", default="fire")
    parser.add_argument("--dmax", type=float, default=0.01)
    parser.add_argument(
        "--etol",
        type=float,
        default=0.0,
        help="LAMMPS relative energy tolerance; 0 disables energy-based stopping",
    )
    parser.add_argument("--ftol", type=float, default=1.0e-6)
    parser.add_argument("--maxiter", type=int, default=120)
    parser.add_argument("--maxeval", type=int, default=1200)


def add_convert_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--species", type=species_list, default=species_list("H,C,N,O,P,S,In,Sn"))
    parser.add_argument("--expected-atoms", type=int)


def generate_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a restrained LAMMPS Stage-0 input")
    add_generate_arguments(parser)
    return generate_input(parser.parse_args(argv))


def convert_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert a typed LAMMPS Stage-0 output")
    add_convert_arguments(parser)
    return convert_output(parser.parse_args(argv))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser("generate")
    add_generate_arguments(generate_parser)
    convert_parser = subparsers.add_parser("convert")
    add_convert_arguments(convert_parser)
    args = parser.parse_args()
    return generate_input(args) if args.command == "generate" else convert_output(args)


if __name__ == "__main__":
    raise SystemExit(main())
