import os
import argparse
import time
import json
import hashlib
import math
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import ase
from ase.io import read, write
from ase.constraints import FixAtoms
from ase.optimize import LBFGS
from ase.md.langevin import Langevin
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.constraints import FixAtoms, Hookean
from mace.calculators import MACECalculator

from sam_lammps import (
    DEFAULT_SPECIES,
    composition,
    final_energy,
    group_id_block,
    frozen_region,
    mass_lines,
    minimization_summary,
    mixed_restrain_block,
    physical_write_data_lines,
    molecular_bonds,
    parent_h_bonds,
    parse_species,
    read_typed_structure,
    read_frozen_atom_ids,
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

def run_annealing_protocol(input_path, model_path, output_dir, n_cycles=3, steps_per_cycle=1500, dt=2.0, restrain_pc=False, restrain_all_bonds=False, max_temp=500.0, device="cuda"):
    """
    Runs multi-cycle MD Simulated Annealing with Hydrogen Mass Repartitioning (HMR) for a configurable timestep.
    Optimizes the structure at the end of each cycle and tracks the global minimum energy structure.
    """
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    progress_path = os.path.join(output_dir, "progress.txt")
    
    print("1. Loading structure and MACE potential...")
    initial_atoms = read(input_path)
    calc = MACECalculator(model_paths=model_path, device=device, default_dtype="float32")
    
    # Apply HMR: set mass of all H atoms to 4.0 amu to allow a stable timestep
    print("\n2. Applying Hydrogen Mass Repartitioning (HMR) to 4.0 amu...")
    h_indices = [i for i, a in enumerate(initial_atoms) if a.symbol == 'H']
    masses = initial_atoms.get_masses()
    masses[h_indices] = 4.0
    initial_atoms.set_masses(masses)
    print(f"Hydrogen mass modified for {len(h_indices)} atoms.")
    
    c_count = len([a for a in initial_atoms if a.symbol == 'C'])
    n_mols = c_count // 22
    atoms_per_mol = 46
    first_sam_idx = len(initial_atoms) - n_mols * atoms_per_mol
    
    # Determine substrate freezing index cutoff
    freeze_cutoff = get_substrate_freeze_cutoff(initial_atoms, first_sam_idx)
    freeze_indices = [i for i in range(first_sam_idx) if initial_atoms.positions[i, 2] < freeze_cutoff]
    
    print(f"Freezing {len(freeze_indices)} middle substrate atoms (Z < {freeze_cutoff:.3f} A).")
    
    constraints = []
    constraints.append(FixAtoms(indices=freeze_indices))
    
    if restrain_pc and not restrain_all_bonds:
        print(f"Applying Hookean restraints (k=20.0 eV/A^2, rt=1.82 A) to all {n_mols} P-C1 bonds during simulated annealing.")
        for mol_idx in range(n_mols):
            start = first_sam_idx + mol_idx * atoms_per_mol
            p_idx, n_idx, s_idx, butyl_carbons = detect_molecule_indices(initial_atoms, start)
            c1_idx = butyl_carbons[0]
            constraints.append(Hookean(a1=p_idx, a2=c1_idx, k=20.0, rt=1.82))
            
    if restrain_all_bonds:
        print(f"Applying weak Hookean restraints (k=20/10 eV/A^2) to ALL internal covalent bonds of {n_mols} SAMs during simulated annealing.")
        for mol_idx in range(n_mols):
            start = first_sam_idx + mol_idx * atoms_per_mol
            sam_indices = list(range(start, start + atoms_per_mol))
            
            for i in range(atoms_per_mol):
                idx_i = sam_indices[i]
                sym_i = initial_atoms[idx_i].symbol
                for j in range(i + 1, atoms_per_mol):
                    idx_j = sam_indices[j]
                    sym_j = initial_atoms[idx_j].symbol
                    
                    d_raw = initial_atoms.get_distance(idx_i, idx_j, mic=True)
                    if d_raw < 1.95:  # covering all covalent bonds
                        # Soft restraints for H bonds, stronger for heavy-heavy bonds
                        k_val = 10.0 if ('H' in (sym_i, sym_j)) else 20.0
                        constraints.append(Hookean(a1=idx_i, a2=idx_j, k=k_val, rt=d_raw))
            
    current_atoms = initial_atoms.copy()
    current_atoms.calc = calc
    current_atoms.set_constraint(constraints)
    
    best_pot_energy = 999999.0
    best_structure_path = None
    
    timestep = dt * units.fs
    friction = 0.002 / units.fs
    
    # Set progress.txt initial state
    with open(progress_path, 'w') as f:
        f.write("Annealing starting...\n")
        
    for cycle in range(1, n_cycles + 1):
        print(f"\n==========================================")
        print(f"STARTING ANNEALING CYCLE {cycle} / {n_cycles}")
        print(f"==========================================")
        
        # Initialize velocities for new cycle
        print("Initializing Maxwell-Boltzmann distribution at 300 K...")
        MaxwellBoltzmannDistribution(current_atoms, temperature_K=300.0)
        
        # Setup Langevin MD
        traj_path = os.path.join(output_dir, f"{base_name}_anneal_cycle_{cycle}.traj")
        log_path = os.path.join(output_dir, f"{base_name}_anneal_cycle_{cycle}.log")
        
        dyn = Langevin(current_atoms, timestep=timestep, temperature_K=300.0, friction=friction)
        traj = ase.io.Trajectory(traj_path, 'w', current_atoms)
        dyn.attach(traj, interval=10)
        
        with open(log_path, 'w') as f:
            f.write("Step Temp[K] Epot[eV]\n")
            
        def log_callback(step, atoms_obj, dyn_obj, log_file):
            # Annealing schedule:
            # 1. Heating (0 to 1/6 steps): 300 K -> max_temp K
            # 2. Holding (1/6 to 3/6 steps): max_temp K
            # 3. Cooling (3/6 to 6/6 steps): max_temp K -> 300 K
            n_heat = steps_per_cycle // 6
            n_hold = steps_per_cycle // 3
            
            if step < n_heat:
                temp = 300.0 + (max_temp - 300.0) * (step / n_heat)
            elif step < n_heat + n_hold:
                temp = max_temp
            else:
                temp = max_temp - (max_temp - 300.0) * ((step - n_heat - n_hold) / (steps_per_cycle - n_heat - n_hold))
                
            dyn_obj.set_temperature(temperature_K=temp)
            
            # Print state to console
            real_temp = atoms_obj.get_temperature()
            epot = atoms_obj.get_potential_energy()
            print(f"Cycle {cycle} | Step {step:4d}/{steps_per_cycle} | Target Temp {temp:5.1f} K | Real Temp {real_temp:5.1f} K | Epot {epot:.2f} eV", flush=True)
            
            # Write to file
            with open(log_file, 'a') as lf:
                lf.write(f"{step} {real_temp:.2f} {epot:.4f}\n")
                
        # Attach callback (using step count since we initialize anew)
        dyn.attach(lambda: log_callback(dyn.get_number_of_steps(), current_atoms, dyn, log_path), interval=10)
        dyn.run(steps_per_cycle)
        
        # Geometry Optimization at end of cycle (frozen substrate)
        print(f"\nRunning final relaxation for Cycle {cycle}...")
        opt = LBFGS(current_atoms, logfile=None, maxstep=0.2)
        opt.run(fmax=0.08, steps=100)
        
        cycle_epot = current_atoms.get_potential_energy()
        print(f"Cycle {cycle} final potential energy: {cycle_epot:.4f} eV")
        
        cycle_cif = os.path.join(output_dir, f"{base_name}_anneal_relaxed_cycle_{cycle}.cif")
        write(cycle_cif, current_atoms)
        print(f"Saved Cycle {cycle} structure to: {cycle_cif}")
        
        if cycle_epot < best_pot_energy:
            best_pot_energy = cycle_epot
            best_structure_path = cycle_cif
            print(f"New global minimum structure found at Cycle {cycle}!")
            
    print(f"\nAnnealing finished. Best structure is {best_structure_path} with energy {best_pot_energy:.4f} eV.")
    with open(progress_path, 'w') as f:
        f.write("Annealing completed successfully.\n")


def _annealing_validation(args, structure: Path, cycle: int) -> Path | None:
    if args.validate_script is None:
        return None
    output = args.validation_dir / f"anneal_cycle_{cycle}_validation.json"
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
        str(output),
    ]
    if args.marker_element:
        command.extend(("--marker-element", args.marker_element))
    if args.marker_carbon_neighbors is not None:
        command.extend(("--marker-carbon-neighbors", str(args.marker_carbon_neighbors)))
    if args.forbid_marker_h:
        command.append("--forbid-marker-h")
    command.extend(("--surface-h-policy", args.surface_h_policy))
    command.extend(("--interface-pair-policy", args.interface_pair_policy))
    if args.registered_interface_bonds_json:
        command.extend(
            (
                "--registered-interface-bonds-json",
                str(args.registered_interface_bonds_json),
            )
        )
    for pair in args.allowed_interface_pair:
        command.extend(("--allowed-interface-pair", pair))
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    return output


def _cycle_numbers(cycle_start: int, cycles: int) -> range:
    if cycle_start < 1:
        raise ValueError("cycle_start must be at least 1")
    if cycles < 1:
        raise ValueError("cycles must be at least 1")
    return range(cycle_start, cycle_start + cycles)


def _coordination_bonds(path, atoms, symbols, substrate_atoms, molecules,
                        atoms_per_molecule, spring, metal_elements,
                        donor_element, minimum, maximum):
    """Read explicit initial donor-metal pairs; never infer new bonds in MD."""
    if path is None:
        return []
    if not math.isfinite(spring) or spring <= 0:
        raise ValueError("coordination spring must be finite and positive")
    if not metal_elements:
        raise ValueError("coordination metal elements must be explicit")
    if not (math.isfinite(minimum) and math.isfinite(maximum) and 0 < minimum < maximum):
        raise ValueError("invalid coordination distance window")
    payload = json.loads(Path(path).read_text())
    if payload.get("schema") != "sam-ito-explicit-initial-coordination-restraints-v1":
        raise ValueError("unsupported coordination restraint schema")
    records = payload.get("bonds")
    if not isinstance(records, list) or not records:
        raise ValueError("coordination restraint list is empty")
    seen = set()
    per_molecule = [0] * molecules
    bonds = []
    for record in records:
        metal = int(record["metal_atom_id_1based"])
        donor = int(record["donor_atom_id_1based"])
        target = float(record["distance_A"])
        if not (1 <= metal <= substrate_atoms < donor <= len(atoms)):
            raise ValueError(f"invalid coordination pair {metal}-{donor}")
        if symbols[metal - 1] not in metal_elements or symbols[donor - 1] != donor_element:
            raise ValueError(f"coordination pair has incorrect elements: {metal}-{donor}")
        if (metal, donor) in seen:
            raise ValueError(f"duplicate coordination pair {metal}-{donor}")
        seen.add((metal, donor))
        molecule = (donor - 1 - substrate_atoms) // atoms_per_molecule
        if not 0 <= molecule < molecules:
            raise ValueError(f"coordination donor is outside SAM blocks: {donor}")
        start = substrate_atoms + molecule * atoms_per_molecule
        anchors = [i for i in range(start, start + atoms_per_molecule)
                   if symbols[i] == "P"]
        if len(anchors) != 1 or atoms.get_distance(anchors[0], donor - 1, mic=True) >= 1.9:
            raise ValueError(f"coordination donor {donor} is not a P-bound O")
        measured = float(atoms.get_distance(metal - 1, donor - 1, mic=True))
        if not (math.isfinite(target) and minimum <= target <= maximum
                and abs(measured - target) <= 0.005):
            raise ValueError(f"coordination target mismatch {metal}-{donor}: {measured:.6f} vs {target:.6f} A")
        per_molecule[molecule] += 1
        bonds.append((metal, donor, spring, target))
    if any(count == 0 for count in per_molecule):
        raise ValueError("every SAM must have at least one coordination restraint")
    return bonds


def run_lammps_annealing(args) -> int:
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
    organic = molecular_bonds(
        atoms,
        symbols,
        args.substrate_atoms,
        args.molecules,
        args.atoms_per_molecule,
    )
    expected_h = args.molecules * args.protons_per_molecule
    parent_h = parent_h_bonds(
        atoms,
        symbols,
        args.substrate_atoms,
        expected_h,
        args.parent_element,
        args.parent_h_max,
    )
    restraints = [
        (atom_a, atom_b, args.h_bond_k if has_h else args.heavy_bond_k, distance)
        for atom_a, atom_b, distance, has_h in organic
    ]
    if args.parent_h_k > 0:
        restraints.extend(
            (parent, hydrogen, args.parent_h_k, distance)
            for parent, hydrogen, distance in parent_h
        )
    metal_elements = parse_species(args.coordination_metal_elements) if args.coordination_metal_elements else ()
    coordination = _coordination_bonds(
        args.coordination_bonds_json, atoms, symbols, args.substrate_atoms,
        args.molecules, args.atoms_per_molecule, args.coordination_k,
        metal_elements, args.coordination_donor_element,
        args.coordination_min_A, args.coordination_max_A,
    )
    restraints.extend(coordination)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.workdir.mkdir(parents=True, exist_ok=True)
    args.validation_dir.mkdir(parents=True, exist_ok=True)
    progress = args.progress or args.output_dir / "progress.txt"
    current_data = args.input.resolve()
    heat = max(1, args.steps // 6)
    hold = max(1, args.steps // 3)
    cool = args.steps - heat - hold
    if cool <= 0:
        raise ValueError("Annealing requires at least 4 steps")

    cycles = []
    best = None
    cycle_numbers = _cycle_numbers(args.cycle_start, args.cycles)
    final_cycle = cycle_numbers.stop - 1
    for cycle in cycle_numbers:
        cycle_data = args.output_dir / f"{args.prefix}_anneal_cycle_{cycle}.data"
        cycle_cif = args.output_dir / f"{args.prefix}_anneal_cycle_{cycle}.cif"
        input_path = args.workdir / f"{args.prefix}_anneal_cycle_{cycle}.in"
        log_path = args.output_dir / f"{args.prefix}_anneal_cycle_{cycle}.log"
        seed_velocity = args.seed + 1009 * cycle
        seed_langevin = args.seed + 2003 * cycle
        lines = [
            "units metal",
            "atom_style atomic",
            "atom_modify map yes",
            "dimension 3",
            "boundary p p p",
            f"read_data {current_data}",
            "",
            *mass_lines(species, args.hmr_mass),
            "",
            f"pair_style mliap unified {args.mliap_model} 0",
            f"pair_coeff * * {' '.join(species)}",
            *frozen_group_lines,
            "group active_atoms subtract all frozen_atoms",
            "velocity frozen_atoms set 0.0 0.0 0.0",
            *mixed_restrain_block("protect_covalent", "all", restraints),
            "fix_modify protect_covalent energy yes",
            "fix hold_bottom frozen_atoms setforce 0.0 0.0 0.0",
            f"velocity active_atoms create 300.0 {seed_velocity} mom yes rot yes dist gaussian",
            f"timestep {args.dt / 1000.0:.8g}",
            f"thermo {args.thermo_every}",
            "thermo_style custom step temp pe ke etotal fnorm",
            "fix integrate active_atoms nve",
            f"fix thermostat active_atoms langevin 300.0 {args.max_temp:.8g} {args.damping_ps:.8g} {seed_langevin}",
            f"run {heat}",
            "unfix thermostat",
            f"fix thermostat active_atoms langevin {args.max_temp:.8g} {args.max_temp:.8g} {args.damping_ps:.8g} {seed_langevin + 1}",
            f"run {hold}",
            "unfix thermostat",
            f"fix thermostat active_atoms langevin {args.max_temp:.8g} 300.0 {args.damping_ps:.8g} {seed_langevin + 2}",
            f"run {cool}",
            "unfix thermostat",
            "unfix integrate",
            f"min_style {args.min_style}",
            f"minimize {args.min_etol:.8g} {args.min_ftol:.8g} {args.min_steps} {args.min_evaluations}",
            "variable final_pe equal pe",
            "run 0",
            'print "FINAL_PE ${final_pe}"',
            *physical_write_data_lines(cycle_data, species),
            "",
        ]
        input_path.write_text("\n".join(lines))
        progress.write_text(f"Annealing cycle {cycle}/{final_cycle}: starting\n")
        run_lammps(
            args.lammps,
            input_path,
            log_path,
            progress,
            args.steps,
            args.workdir,
        )
        energy = final_energy(log_path)
        minimization = minimization_summary(log_path)
        cycle_atoms = read_typed_structure(cycle_data, species)
        write_clean_structure(cycle_atoms, cycle_cif)
        validation = _annealing_validation(args, cycle_data, cycle)
        record = {
            "cycle": cycle,
            "energy_eV": energy,
            "data": str(cycle_data),
            "cif": str(cycle_cif),
            "validation": str(validation) if validation else None,
            "minimization": minimization,
        }
        cycles.append(record)
        if best is None or energy < best[0]:
            best = (energy, cycle_data, cycle_cif, cycle)
        current_data = cycle_data.resolve()
        progress.write_text(
            f"Annealing cycle {cycle}/{args.cycles}: validated, E={energy:.8f} eV\n"
        )

    assert best is not None
    best_data = args.output_dir / f"{args.prefix}_annealed_best.data"
    best_cif = args.output_dir / f"{args.prefix}_annealed_best.cif"
    shutil.copy2(best[1], best_data)
    shutil.copy2(best[2], best_cif)
    manifest = {
        "engine": "lammps-mliap",
        "input": str(args.input),
        "model": str(args.mliap_model),
        "species": list(species),
        "atom_count": len(atoms),
        "composition": composition(atoms),
        "substrate_atoms": args.substrate_atoms,
        "molecules": args.molecules,
        "atoms_per_molecule": args.atoms_per_molecule,
        "surface_protons": expected_h,
        "frozen_z_max_A": freeze_z,
        "frozen_atom_count": frozen_count,
        "frozen_selection_mode": frozen_mode,
        "frozen_atom_ids_file": (
            str(args.frozen_atom_ids_file) if args.frozen_atom_ids_file else None
        ),
        "molecular_restraint_count": len(organic),
        "parent_h_restraint_count": len(parent_h) if args.parent_h_k > 0 else 0,
        "coordination_restraint_count": len(coordination),
        "coordination_bonds_json": str(args.coordination_bonds_json) if coordination else None,
        "coordination_bonds_sha256": hashlib.sha256(args.coordination_bonds_json.read_bytes()).hexdigest() if coordination else None,
        "coordination_k_eV_per_A2": args.coordination_k if coordination else None,
        "min_relative_etol": args.min_etol,
        "cycle_start": args.cycle_start,
        "cycle_count": args.cycles,
        "cycles": cycles,
        "best_cycle": best[3],
        "best_energy_eV": best[0],
        "best_data": str(best_data),
        "best_cif": str(best_cif),
    }
    write_manifest(args.manifest, manifest)
    progress.write_text(
        f"Annealing complete: best cycle {best[3]}, E={best[0]:.8f} eV\n"
    )
    print(json.dumps(manifest, indent=2))
    return 0

def main():
    parser = argparse.ArgumentParser(description="Multi-cycle MD Simulated Annealing for SAM-ITO.")
    parser.add_argument("--engine", choices=("ase", "lammps-mliap"), default="ase")
    parser.add_argument("--input", type=str, required=True, help="Input relaxed CIF file.")
    parser.add_argument("--model", type=str, required=True, help="MACE model file path.")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory.")
    parser.add_argument("--cycles", type=int, default=3, help="Number of annealing cycles.")
    parser.add_argument(
        "--cycle-start",
        type=int,
        default=1,
        help="Global cycle number assigned to the first cycle in this run.",
    )
    parser.add_argument("--steps", type=int, default=1500, help="Steps per cycle.")
    parser.add_argument("--dt", type=float, default=2.0, help="Timestep in fs.")
    parser.add_argument("--restrain-pc", action="store_true", help="Apply Hookean harmonic restraints to P-C1 bonds.")
    parser.add_argument("--restrain-all-bonds", action="store_true", help="Apply Hookean harmonic restraints to ALL SAM covalent bonds.")
    parser.add_argument("--max-temp", type=float, default=500.0, help="Maximum annealing temperature in K.")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu.")
    parser.add_argument("--lammps", default="lmp")
    parser.add_argument("--mliap-model", type=Path)
    parser.add_argument("--species", default=" ".join(DEFAULT_SPECIES))
    parser.add_argument("--substrate-atoms", type=int)
    parser.add_argument("--molecules", type=int)
    parser.add_argument("--atoms-per-molecule", type=int)
    parser.add_argument("--protons-per-molecule", type=int, default=2)
    parser.add_argument("--anchor-element", default="P")
    parser.add_argument("--parent-element", default="O")
    parser.add_argument("--acceptor-element", default="O")
    parser.add_argument("--parent-h-max", type=float, default=1.25)
    parser.add_argument("--freeze-depth", type=float, default=1.10)
    parser.add_argument("--frozen-atom-ids-file", type=Path)
    parser.add_argument("--heavy-bond-k", type=float, default=20.0)
    parser.add_argument("--h-bond-k", type=float, default=10.0)
    parser.add_argument("--parent-h-k", type=float, default=10.0)
    parser.add_argument("--coordination-bonds-json", type=Path)
    parser.add_argument("--coordination-k", type=float, default=20.0)
    parser.add_argument("--coordination-metal-elements", help="Explicit substrate metal elements, e.g. 'In Sn'.")
    parser.add_argument("--coordination-donor-element", default="O")
    parser.add_argument("--coordination-min-A", type=float, default=1.7)
    parser.add_argument("--coordination-max-A", type=float, default=2.8)
    parser.add_argument("--hmr-mass", type=float, default=4.0)
    parser.add_argument("--damping-ps", type=float, default=0.5)
    parser.add_argument("--thermo-every", type=int, default=10)
    parser.add_argument("--min-style", default="cg")
    parser.add_argument(
        "--min-etol",
        type=float,
        default=0.0,
        help="LAMMPS relative energy tolerance; 0 disables energy-based stopping",
    )
    parser.add_argument("--min-ftol", type=float, default=1.0e-6)
    parser.add_argument("--min-steps", type=int, default=200)
    parser.add_argument("--min-evaluations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=340043)
    parser.add_argument("--prefix", default="sam")
    parser.add_argument("--workdir", type=Path, default=Path("scratch/lammps_anneal"))
    parser.add_argument("--manifest", type=Path, default=Path("annealing_manifest.json"))
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--validate-script", type=Path)
    parser.add_argument("--validation-dir", type=Path, default=Path("validation"))
    parser.add_argument("--marker-element")
    parser.add_argument("--marker-carbon-neighbors", type=int)
    parser.add_argument("--forbid-marker-h", action="store_true")
    parser.add_argument(
        "--surface-h-policy",
        choices=("network", "retained", "mobile"),
        default="network",
    )
    parser.add_argument("--allowed-interface-pair", action="append", default=[])
    parser.add_argument("--registered-interface-bonds-json", type=Path)
    parser.add_argument(
        "--interface-pair-policy",
        choices=("exact", "mobile"),
        default="exact",
    )
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
        return run_lammps_annealing(args)
    return run_annealing_protocol(
        input_path=args.input,
        model_path=args.model,
        output_dir=args.outdir,
        n_cycles=args.cycles,
        steps_per_cycle=args.steps,
        dt=args.dt,
        restrain_pc=args.restrain_pc,
        restrain_all_bonds=args.restrain_all_bonds,
        max_temp=args.max_temp,
        device=args.device
    )

if __name__ == "__main__":
    main()
