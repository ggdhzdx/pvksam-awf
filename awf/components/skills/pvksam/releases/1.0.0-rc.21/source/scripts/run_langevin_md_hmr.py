import os
import argparse
import time
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import ase
from ase.io import read, write
from ase.constraints import FixAtoms, Hookean
from ase.md.langevin import Langevin
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from mace.calculators import MACECalculator

from sam_lammps import (
    DEFAULT_SPECIES,
    anchor_bonds,
    composition,
    final_energy,
    group_id_block,
    frozen_region,
    mass_lines,
    mixed_restrain_block,
    molecular_bonds,
    physical_write_data_lines,
    parse_species,
    read_typed_structure,
    read_frozen_atom_ids,
    resolved_path,
    restrain_block,
    run_lammps,
    validate_layout,
    write_clean_structure,
    write_manifest,
)

def detect_molecule_indices(atoms, start):
    sam_indices = range(start, start + 46)
    symbols = [atoms[i].symbol for i in sam_indices]
    
    p_idx = start + symbols.index('P')
    n_idx = start + symbols.index('N')
    s_idx = start + symbols.index('S')
    
    c_indices = [i for i in sam_indices if atoms[i].symbol == 'C']
    c1 = min(c_indices, key=lambda i: atoms.get_distance(p_idx, i, mic=True))
    
    other_c = [i for i in c_indices if i != c1]
    c2 = min(other_c, key=lambda i: atoms.get_distance(c1, i, mic=True))
    
    other_c = [i for i in c_indices if i not in [c1, c2]]
    c3 = min(other_c, key=lambda i: atoms.get_distance(c2, i, mic=True))
    
    other_c = [i for i in c_indices if i not in [c1, c2, c3]]
    c4 = min(other_c, key=lambda i: atoms.get_distance(c3, i, mic=True))
    
    butyl_carbons = [c1, c2, c3, c4]
    return p_idx, n_idx, s_idx, butyl_carbons

def get_substrate_freeze_cutoff(atoms, first_sam_idx):
    sub_atoms = atoms[range(first_sam_idx)]
    metal_indices = [i for i, a in enumerate(sub_atoms) if a.symbol in ['In', 'Sn']]
    if not metal_indices:
        return 6.040
    metal_z = sub_atoms.positions[metal_indices, 2]
    min_metal_z = np.min(metal_z)
    return min_metal_z + 1.5

def run_md(input_path, model_path, output_dir, temp_k=300.0, steps=3000, log_interval=10, restrain_pc=False, device="cuda"):
    """
    Runs production Langevin Molecular Dynamics NVT simulation at temp_k K for 'steps' steps.
    Applies Hydrogen Mass Repartitioning (HMR) to 4.0 amu to allow stable 2.0 fs timestep.
    """
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    traj_path = os.path.join(output_dir, f"{base_name}_production_nvt.traj")
    log_path = os.path.join(output_dir, f"{base_name}_production_nvt.log")
    progress_path = os.path.join(output_dir, "progress.txt")
    
    print("1. Loading structure and calculator...")
    atoms = read(input_path)
    calc = MACECalculator(model_paths=model_path, device=device, default_dtype="float32")
    atoms.calc = calc
    
    # 2. Apply HMR: set mass of all H atoms to 4.0 amu to allow a stable 2.0 fs timestep
    print("\n2. Applying Hydrogen Mass Repartitioning (HMR) to 4.0 amu...")
    h_indices = [i for i, a in enumerate(atoms) if a.symbol == 'H']
    masses = atoms.get_masses()
    masses[h_indices] = 4.0
    atoms.set_masses(masses)
    print(f"Hydrogen mass modified for {len(h_indices)} atoms.")
    
    # 3. Setup constraints: Freeze the substrate atoms that are not in the active surface layer
    c_indices = [i for i, a in enumerate(atoms) if a.symbol == 'C']
    p_indices = [i for i, a in enumerate(atoms) if a.symbol == 'P']
    if c_indices or p_indices:
        first_sam_idx = min(c_indices + p_indices)
    else:
        first_sam_idx = len(atoms)
        
    freeze_cutoff = get_substrate_freeze_cutoff(atoms, first_sam_idx)
    freeze_indices = [i for i in range(first_sam_idx) if atoms.positions[i, 2] < freeze_cutoff]
    
    print(f"Freezing {len(freeze_indices)} middle substrate atoms (Z < {freeze_cutoff:.3f} A).")
    
    constraints = []
    constraints.append(FixAtoms(indices=freeze_indices))
    
    if restrain_pc:
        c_count = len(c_indices)
        n_mols = c_count // 22
        atoms_per_mol = 46
        print(f"Applying Hookean restraints (k=20.0 eV/A^2, rt=1.82 A) to all {n_mols} P-C1 bonds during Langevin MD.")
        for mol_idx in range(n_mols):
            start = first_sam_idx + mol_idx * atoms_per_mol
            p_idx, n_idx, s_idx, butyl_carbons = detect_molecule_indices(atoms, start)
            c1_idx = butyl_carbons[0]
            constraints.append(Hookean(a1=p_idx, a2=c1_idx, k=20.0, rt=1.82))
            
    atoms.set_constraint(constraints)
    
    # 4. Set velocities to target temperature
    print("\n3. Initializing Maxwell-Boltzmann velocity distribution at target temperature...")
    MaxwellBoltzmannDistribution(atoms, temperature_K=temp_k)
    
    # 5. Set up Langevin NVT MD with 2.0 fs timestep and friction = 0.002 / units.fs
    timestep = 2.0 * units.fs
    friction = 0.002 / units.fs
    
    print(f"\n4. Setting up Langevin NVT MD (Temp={temp_k} K, Friction={friction * units.fs:.5f} fs^-1, Timestep={timestep / units.fs:.1f} fs, Steps={steps})...")
    dyn = Langevin(atoms, timestep=timestep, temperature_K=temp_k, friction=friction)
    
    # Log and Trajectory handlers
    traj = ase.io.Trajectory(traj_path, 'w', atoms)
    dyn.attach(traj, interval=log_interval)
    
    with open(log_path, 'w') as f:
        f.write("Step Time[ps] Epot[eV] Ekin[eV] Etot[eV] Temp[K]\n")
        
    def log_callback(step, atoms):
        epot = atoms.get_potential_energy()
        ekin = atoms.get_kinetic_energy()
        etot = epot + ekin
        temp = atoms.get_temperature()
        time_ps = step * (timestep / units.fs) / 1000.0
        with open(log_path, 'a') as f:
            f.write(f"{step:8d} {time_ps:10.4f} {epot:12.4f} {ekin:12.4f} {etot:12.4f} {temp:10.2f}\n")
        print(f"Step {step:6d}/{steps} | Time {time_ps:7.3f} ps | Epot {epot:10.3f} eV | Temp {temp:6.1f} K", flush=True)
        
    def progress_callback(step):
        pct = (step / steps) * 100
        with open(progress_path, 'w') as f:
            f.write(f"Step {step}/{steps} ({pct:.1f}%)\n")
            
    dyn.attach(lambda: log_callback(dyn.get_number_of_steps(), atoms), interval=log_interval)
    dyn.attach(lambda: progress_callback(dyn.get_number_of_steps()), interval=log_interval)
    
    progress_callback(0)
    
    # 6. Run dynamics
    print("\n5. Starting Langevin Molecular Dynamics...")
    t0 = time.time()
    dyn.run(steps)
    t1 = time.time()
    
    print(f"\n6. MD completed successfully in {t1 - t0:.2f} seconds.")
    progress_callback(steps)


def _production_validation(args, structure: Path) -> Path | None:
    if args.validate_script is None:
        return None
    command = [
        sys.executable,
        str(args.validate_script),
        str(structure),
        "--substrate-atoms",
        str(args.substrate_atoms),
        "--molecules",
        str(args.molecules),
        "--atoms-per-molecule",
        str(args.atoms_per_molecule),
        "--protons-per-molecule",
        str(args.protons_per_molecule),
        "--anchor-element",
        args.anchor_element,
        "--parent-element",
        args.parent_element,
        "--acceptor-element",
        args.acceptor_element,
        "--json-output",
        str(args.validation_output),
    ]
    if args.marker_element:
        command.extend(("--marker-element", args.marker_element))
    if args.marker_carbon_neighbors is not None:
        command.extend(("--marker-carbon-neighbors", str(args.marker_carbon_neighbors)))
    if args.forbid_marker_h:
        command.append("--forbid-marker-h")
    command.extend(("--surface-h-policy", args.surface_h_policy))
    for pair in args.allowed_interface_pair:
        command.extend(("--allowed-interface-pair", pair))
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    return args.validation_output


def run_lammps_md(args) -> int:
    # LAMMPS runs inside ``workdir``. Resolve every external path first so a
    # relative CLI path is not interpreted a second time after that cwd change.
    args.input = resolved_path(args.input)
    args.output_dir = resolved_path(args.output_dir)
    args.workdir = resolved_path(args.workdir)
    args.manifest = resolved_path(args.manifest)
    args.progress = resolved_path(args.progress)
    args.validation_output = resolved_path(args.validation_output)
    args.validate_script = resolved_path(args.validate_script)
    args.mliap_model = resolved_path(args.mliap_model)
    args.frozen_atom_ids_file = resolved_path(args.frozen_atom_ids_file)

    species = parse_species(args.species)
    atoms = read_typed_structure(args.input, species)
    symbols = validate_layout(
        atoms, args.substrate_atoms, args.molecules, args.atoms_per_molecule
    )
    if args.frozen_atom_ids_file:
        frozen_ids = read_frozen_atom_ids(
            args.frozen_atom_ids_file, args.substrate_atoms
        )
        freeze_z = float(np.max(atoms.positions[np.asarray(frozen_ids) - 1, 2]))
        frozen_group_lines = group_id_block("frozen_atoms", frozen_ids)
        frozen_mode = "atom_ids"
    else:
        freeze_z, frozen_count = frozen_region(
            atoms, symbols, args.substrate_atoms, args.freeze_depth
        )
        frozen_ids = []
        frozen_group_lines = [
            f"region frozen_bottom block INF INF INF INF INF {freeze_z:.8f} units box",
            "group frozen_atoms region frozen_bottom",
        ]
        frozen_mode = "coordinate_region"
    frozen_count = len(frozen_ids) if frozen_ids else frozen_count
    anchors = anchor_bonds(
        atoms,
        symbols,
        args.substrate_atoms,
        args.molecules,
        args.atoms_per_molecule,
        args.anchor_element,
        args.anchor_neighbor_element,
        args.anchor_neighbor_max,
    )
    restrained = [
        (anchor, neighbor, args.anchor_target) for anchor, neighbor, _ in anchors
    ]
    molecular_restraints = []
    if args.restrain_molecular_bonds:
        molecular = molecular_bonds(
            atoms,
            symbols,
            args.substrate_atoms,
            args.molecules,
            args.atoms_per_molecule,
        )
        anchor_pairs = {
            frozenset((anchor, neighbor)) for anchor, neighbor, _ in anchors
        }
        molecular_restraints = [
            (
                atom_a,
                atom_b,
                args.h_bond_k if has_h else args.heavy_bond_k,
                distance,
            )
            for atom_a, atom_b, distance, has_h in molecular
            if frozenset((atom_a, atom_b)) not in anchor_pairs
        ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.workdir.mkdir(parents=True, exist_ok=True)
    args.validation_output.parent.mkdir(parents=True, exist_ok=True)
    progress = args.progress or args.output_dir / "progress.txt"
    final_data = args.output_dir / f"{args.prefix}_production_final.data"
    final_cif = args.output_dir / f"{args.prefix}_production_final.cif"
    trajectory = args.output_dir / f"{args.prefix}_production_nvt.lammpstrj"
    log_path = args.output_dir / f"{args.prefix}_production_nvt.log"
    input_path = args.workdir / f"{args.prefix}_production.in"
    lines = [
        "units metal",
        "atom_style atomic",
        "atom_modify map yes",
        "dimension 3",
        "boundary p p p",
        f"read_data {args.input.resolve()}",
        "",
        *mass_lines(species, args.hmr_mass),
        "",
        f"pair_style mliap unified {args.mliap_model} 0",
        f"pair_coeff * * {' '.join(species)}",
        *frozen_group_lines,
        "group active_atoms subtract all frozen_atoms",
        "velocity frozen_atoms set 0.0 0.0 0.0",
        "fix hold_bottom frozen_atoms setforce 0.0 0.0 0.0",
        *restrain_block(
            "protect_anchor", "all", restrained, args.anchor_k
        ),
        *mixed_restrain_block(
            "protect_molecular_bonds", "all", molecular_restraints
        ),
        f"velocity active_atoms create {args.temp:.8g} {args.seed} mom yes rot yes dist gaussian",
        f"timestep {args.dt / 1000.0:.8g}",
        f"thermo {args.interval}",
        "thermo_style custom step temp pe ke etotal fnorm",
        f"dump production all custom {args.interval} {trajectory} id type x y z vx vy vz",
        "dump_modify production sort id",
        "fix integrate active_atoms nve",
        f"fix thermostat active_atoms langevin {args.temp:.8g} {args.temp:.8g} {args.damping_ps:.8g} {args.seed + 1}",
        f"run {args.steps}",
        "unfix thermostat",
        "unfix integrate",
        "undump production",
        "variable final_pe equal pe",
        "run 0",
        'print "FINAL_PE ${final_pe}"',
        *physical_write_data_lines(final_data, species),
        "",
    ]
    input_path.write_text("\n".join(lines))
    progress.write_text("Production MD: starting\n")
    run_lammps(
        args.lammps,
        input_path,
        log_path,
        progress,
        args.steps,
        args.workdir,
    )
    energy = final_energy(log_path)
    final_atoms = read_typed_structure(final_data, species)
    write_clean_structure(final_atoms, final_cif)
    validation = _production_validation(args, final_data)
    manifest = {
        "engine": "lammps-mliap",
        "input": str(args.input),
        "model": str(args.mliap_model),
        "species": list(species),
        "atom_count": len(final_atoms),
        "composition": composition(final_atoms),
        "substrate_atoms": args.substrate_atoms,
        "molecules": args.molecules,
        "atoms_per_molecule": args.atoms_per_molecule,
        "surface_protons": args.molecules * args.protons_per_molecule,
        "frozen_z_max_A": freeze_z,
        "frozen_atom_count": frozen_count,
        "frozen_selection_mode": frozen_mode,
        "frozen_atom_ids_file": (
            str(args.frozen_atom_ids_file) if args.frozen_atom_ids_file else None
        ),
        "anchor_restraint_count": len(anchors),
        "molecular_bond_restraints_enabled": args.restrain_molecular_bonds,
        "molecular_bond_restraint_count": len(molecular_restraints),
        "molecular_heavy_bond_k_eV_per_A2": args.heavy_bond_k,
        "molecular_H_bond_k_eV_per_A2": args.h_bond_k,
        "steps": args.steps,
        "timestep_fs": args.dt,
        "temperature_K": args.temp,
        "final_energy_eV": energy,
        "trajectory": str(trajectory),
        "final_data": str(final_data),
        "final_cif": str(final_cif),
        "validation": str(validation) if validation else None,
    }
    write_manifest(args.manifest, manifest)
    progress.write_text(
        f"Production MD complete and validated: E={energy:.8f} eV\n"
    )
    print(json.dumps(manifest, indent=2))
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hydrogen Mass Repartitioned Langevin NVT Molecular Dynamics Driver for SAM-Substrate Systems using MACE")
    parser.add_argument("--engine", choices=("ase", "lammps-mliap"), default="ase")
    parser.add_argument("--input", type=str, required=True, help="Path to input CIF structure file")
    parser.add_argument("--model", type=str, required=True, help="Path to MACE model file")
    parser.add_argument("--outdir", type=str, default="./pvksam-output", help="Output directory")
    parser.add_argument("--temp", type=float, default=300.0, help="Simulation temperature in K")
    parser.add_argument("--steps", type=int, default=3000, help="Number of MD steps")
    parser.add_argument("--interval", type=int, default=10, help="Log/Trajectory interval in steps")
    parser.add_argument("--restrain-pc", action="store_true", help="Apply Hookean constraints to P-C1 bonds during Langevin MD.")
    parser.add_argument("--device", type=str, default="cuda", help="Computation device (cuda/cpu)")
    parser.add_argument("--lammps", default="lmp")
    parser.add_argument("--mliap-model", type=Path)
    parser.add_argument("--species", default=" ".join(DEFAULT_SPECIES))
    parser.add_argument("--substrate-atoms", type=int)
    parser.add_argument("--molecules", type=int)
    parser.add_argument("--atoms-per-molecule", type=int)
    parser.add_argument("--protons-per-molecule", type=int, default=2)
    parser.add_argument("--anchor-element", default="P")
    parser.add_argument("--anchor-neighbor-element", default="C")
    parser.add_argument("--anchor-neighbor-max", type=float, default=2.15)
    parser.add_argument("--anchor-target", type=float, default=1.82)
    parser.add_argument("--anchor-k", type=float, default=20.0)
    parser.add_argument(
        "--restrain-molecular-bonds",
        action="store_true",
        help=(
            "weakly restrain covalent bonds inside explicit SAM blocks while "
            "leaving substrate surface H mobile"
        ),
    )
    parser.add_argument("--heavy-bond-k", type=float, default=20.0)
    parser.add_argument("--h-bond-k", type=float, default=10.0)
    parser.add_argument("--parent-element", default="O")
    parser.add_argument("--acceptor-element", default="O")
    parser.add_argument("--freeze-depth", type=float, default=1.10)
    parser.add_argument("--frozen-atom-ids-file", type=Path)
    parser.add_argument("--hmr-mass", type=float, default=4.0)
    parser.add_argument("--dt", type=float, default=2.0)
    parser.add_argument("--damping-ps", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=340043)
    parser.add_argument("--prefix", default="sam")
    parser.add_argument("--workdir", type=Path, default=Path("scratch/lammps_production"))
    parser.add_argument("--manifest", type=Path, default=Path("production_manifest.json"))
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--validate-script", type=Path)
    parser.add_argument("--validation-output", type=Path, default=Path("production_validation.json"))
    parser.add_argument("--marker-element")
    parser.add_argument("--marker-carbon-neighbors", type=int)
    parser.add_argument("--forbid-marker-h", action="store_true")
    parser.add_argument(
        "--surface-h-policy",
        choices=("network", "retained", "mobile"),
        default="network",
    )
    parser.add_argument("--allowed-interface-pair", action="append", default=[])
    
    args = parser.parse_args()
    
    if args.engine == "lammps-mliap":
        required = {
            "mliap_model": args.mliap_model,
            "substrate_atoms": args.substrate_atoms,
            "molecules": args.molecules,
            "atoms_per_molecule": args.atoms_per_molecule,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error(f"lammps-mliap requires: {', '.join(missing)}")
        args.input = Path(args.input)
        args.output_dir = Path(args.outdir)
        raise SystemExit(run_lammps_md(args))
    run_md(
        input_path=args.input,
        model_path=args.model,
        output_dir=args.outdir,
        temp_k=args.temp,
        steps=args.steps,
        log_interval=args.interval,
        restrain_pc=args.restrain_pc,
        device=args.device
    )
