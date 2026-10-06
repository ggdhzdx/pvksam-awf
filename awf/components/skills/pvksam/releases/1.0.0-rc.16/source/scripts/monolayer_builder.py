"""Build SAM monolayers using sequential, RSA, or headgroup-fit workflows."""

import os
import sys
import argparse
import json
import random
from collections import Counter
from pathlib import Path
import numpy as np
import ase
from ase.io import write
from ase.constraints import FixAtoms
from scipy.optimize import linear_sum_assignment
import time

from sam_lammps import read_typed_structure


def _metal_layers(atoms, metal_elements=("In", "Sn"), gap=0.8):
    symbols = np.asarray(atoms.get_chemical_symbols())
    indices = np.flatnonzero(np.isin(symbols, metal_elements))
    ordered = indices[np.argsort(atoms.positions[indices, 2])]
    layers = []
    for index_raw in ordered:
        index = int(index_raw)
        if not layers or atoms.positions[index, 2] - atoms.positions[layers[-1][-1], 2] > gap:
            layers.append([index])
        else:
            layers[-1].append(index)
    return layers


def _mic_distances(delta, cell):
    inverse = np.linalg.inv(cell)
    fractional = delta @ inverse
    fractional -= np.round(fractional)
    return np.linalg.norm(fractional @ cell, axis=-1)


def prepare_layer_package(argv):
    parser = argparse.ArgumentParser(
        description="Recover a removed bottom substrate layer and map it to a peeled reference."
    )
    parser.add_argument("--full-reference", type=Path, required=True)
    parser.add_argument("--peeled-reference", type=Path, required=True)
    parser.add_argument("--layer-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metal-elements", nargs="+", default=["In", "Sn"])
    parser.add_argument("--layer-gap", type=float, default=0.8)
    parser.add_argument("--full-reference-repeat", nargs=3, type=int, default=(1, 1, 1))
    parser.add_argument("--peeled-reference-repeat", nargs=3, type=int, default=(1, 1, 1))
    parser.add_argument("--repeat", nargs=3, type=int, default=(1, 1, 1))
    parser.add_argument("--orthogonalize", action="store_true")
    parser.add_argument("--alignment-max", type=float, default=1.0e-5)
    args = parser.parse_args(argv)

    full = read_typed_structure(args.full_reference)
    peeled = read_typed_structure(args.peeled_reference)
    full_reference_repeat = tuple(args.full_reference_repeat)
    peeled_reference_repeat = tuple(args.peeled_reference_repeat)
    if full_reference_repeat != (1, 1, 1):
        full = full * full_reference_repeat
    if peeled_reference_repeat != (1, 1, 1):
        peeled = peeled * peeled_reference_repeat
    if not np.allclose(full.cell, peeled.cell, atol=1.0e-6):
        raise ValueError(
            "Full and peeled reference cells differ after reference repeats: "
            f"full={np.asarray(full.cell).tolist()}, "
            f"peeled={np.asarray(peeled.cell).tolist()}"
        )
    full_layers = _metal_layers(full, tuple(args.metal_elements), args.layer_gap)
    peeled_layers = _metal_layers(peeled, tuple(args.metal_elements), args.layer_gap)
    if len(full_layers) != len(peeled_layers) + 1 or len(full_layers) < 3:
        raise ValueError(
            f"Expected one removed metal layer; full={len(full_layers)}, "
            f"peeled={len(peeled_layers)}"
        )
    full_middle = full_layers[1]
    peeled_middle = peeled_layers[0]
    full_symbols = np.asarray(full.get_chemical_symbols())
    peeled_symbols = np.asarray(peeled.get_chemical_symbols())
    if Counter(full_symbols[full_middle]) != Counter(peeled_symbols[peeled_middle]):
        raise ValueError("Middle-layer metal compositions do not match")

    full_positions = full.positions[full_middle]
    peeled_positions = peeled.positions[peeled_middle]
    full_middle_symbols = full_symbols[full_middle]
    peeled_middle_symbols = peeled_symbols[peeled_middle]
    cell = np.asarray(peeled.cell)
    candidate_element = full_middle_symbols[0]
    peeled_candidates = np.flatnonzero(peeled_middle_symbols == candidate_element)
    best = None
    for candidate in peeled_candidates:
        shift = peeled_positions[int(candidate)] - full_positions[0]
        assigned_distances = []
        for element in sorted(set(full_middle_symbols)):
            source = full_positions[full_middle_symbols == element] + shift
            target = peeled_positions[peeled_middle_symbols == element]
            distances = _mic_distances(
                source[:, None, :] - target[None, :, :], cell
            )
            rows, columns = linear_sum_assignment(distances)
            assigned_distances.extend(distances[rows, columns].tolist())
        rmsd = float(np.sqrt(np.mean(np.square(assigned_distances))))
        maximum = float(max(assigned_distances))
        if best is None or rmsd < best[0]:
            best = (rmsd, maximum, shift)
    assert best is not None
    if best[1] > args.alignment_max:
        raise ValueError(
            f"Middle-layer mapping is not exact: RMSD={best[0]:.6g}, "
            f"max={best[1]:.6g} A"
        )

    bottom_middle_cutoff = 0.5 * (
        float(np.mean(full.positions[full_layers[0], 2]))
        + float(np.mean(full.positions[full_layers[1], 2]))
    )
    bottom_indices = np.flatnonzero(full.positions[:, 2] < bottom_middle_cutoff)
    molecular_elements = {"C", "N", "P", "S"}
    if any(full_symbols[index] in molecular_elements for index in bottom_indices):
        raise ValueError("Bottom-layer selection contains molecular atoms")
    layer = full[bottom_indices].copy()
    layer.translate(best[2])
    layer.set_cell(peeled.cell)
    layer.set_pbc(peeled.pbc)
    repeat = tuple(args.repeat)
    if repeat != (1, 1, 1):
        layer = layer * repeat
    if args.orthogonalize:
        repeated_cell = np.asarray(layer.cell)
        orthogonal_cell = np.asarray(
            [repeated_cell[0], repeated_cell[0] + repeated_cell[1], repeated_cell[2]]
        )
        off_diagonal = orthogonal_cell.copy()
        np.fill_diagonal(off_diagonal, 0.0)
        if np.max(np.abs(off_diagonal)) > 1.0e-5:
            raise ValueError(f"Repeated cell is not orthogonalizable: {orthogonal_cell}")
        layer.set_cell(orthogonal_cell, scale_atoms=False)
        layer.wrap()

    args.layer_output.parent.mkdir(parents=True, exist_ok=True)
    write(args.layer_output, layer)
    payload = {
        "method": "middle-layer exact translation mapping",
        "full_reference": str(args.full_reference),
        "peeled_reference": str(args.peeled_reference),
        "layer_output": str(args.layer_output),
        "full_metal_layer_count": len(full_layers),
        "peeled_metal_layer_count": len(peeled_layers),
        "bottom_middle_cutoff_A": bottom_middle_cutoff,
        "translation_A": best[2].tolist(),
        "alignment_rmsd_A": best[0],
        "alignment_max_A": best[1],
        "full_reference_repeat": list(full_reference_repeat),
        "peeled_reference_repeat": list(peeled_reference_repeat),
        "repeat": list(repeat),
        "orthogonalized": args.orthogonalize,
        "layer_atom_count": len(layer),
        "layer_formula": dict(sorted(Counter(layer.get_chemical_symbols()).items())),
        "cell_A": np.asarray(layer.cell).tolist(),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


def restore_layer(argv):
    parser = argparse.ArgumentParser(
        description="Insert a saved bottom layer before contiguous SAM blocks."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--layer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--substrate-atoms", type=int, required=True)
    parser.add_argument("--molecules", type=int, required=True)
    parser.add_argument("--atoms-per-molecule", type=int, required=True)
    parser.add_argument("--expected-layer-atoms", type=int)
    parser.add_argument("--minimum-cross-distance", type=float, default=0.80)
    args = parser.parse_args(argv)

    current = read_typed_structure(args.input)
    layer = read_typed_structure(args.layer)
    expected_current = args.substrate_atoms + args.molecules * args.atoms_per_molecule
    if len(current) != expected_current:
        raise ValueError(f"Current structure has {len(current)} atoms; expected {expected_current}")
    if args.expected_layer_atoms is not None and len(layer) != args.expected_layer_atoms:
        raise ValueError(f"Saved layer has {len(layer)} atoms; expected {args.expected_layer_atoms}")
    if not np.allclose(current.cell, layer.cell, atol=1.0e-5):
        raise ValueError("Current structure and saved layer cells differ")
    distances = _mic_distances(
        layer.positions[:, None, :] - current.positions[: args.substrate_atoms][None, :, :],
        np.asarray(current.cell),
    )
    minimum = float(np.min(distances))
    if minimum < args.minimum_cross_distance:
        raise ValueError(
            f"Restored layer overlaps retained substrate: minimum={minimum:.6f} A"
        )
    restored = current[: args.substrate_atoms].copy()
    restored.extend(layer)
    restored.extend(current[args.substrate_atoms :])
    restored.set_cell(current.cell)
    restored.set_pbc(current.pbc)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, restored)
    if args.output.suffix.lower() == ".cif":
        lines = args.output.read_text().splitlines(keepends=True)
        args.output.write_text(
            "".join(line for line in lines if not line.startswith("_chemical_formula_structural"))
        )
    payload = {
        "input": str(args.input),
        "layer": str(args.layer),
        "output": str(args.output),
        "working_atom_count": len(current),
        "restored_layer_atom_count": len(layer),
        "final_atom_count": len(restored),
        "final_formula": dict(sorted(Counter(restored.get_chemical_symbols()).items())),
        "minimum_layer_retained_distance_A": minimum,
        "molecular_blocks_preserved_at_end": True,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0

def unwrap_substructure(atoms, indices, ref_idx):
    """
    Unwraps coordinates of atoms in `indices` relative to `ref_idx` using MIC.
    Modifies positions in-place on the `atoms` object.
    """
    ref_pos = atoms.positions[ref_idx].copy()
    cell = atoms.get_cell()
    inv_cell = np.linalg.inv(cell)
    
    for idx in indices:
        if idx == ref_idx:
            continue
        diff = atoms.positions[idx] - ref_pos
        frac_diff = np.dot(diff, inv_cell)
        frac_diff = frac_diff - np.round(frac_diff)
        atoms.positions[idx] = ref_pos + np.dot(frac_diff, cell)

def get_sam_component_no_o_neighbors(atoms, start_idx):
    visited = {start_idx}
    frontier = [start_idx]
    allowed_symbols = {'P', 'C', 'N', 'S', 'O', 'H'}
    
    while frontier:
        curr = frontier.pop(0)
        sym_curr = atoms[curr].symbol
        
        # Treat oxygen as leaf node to prevent bleeding into substrate or hydrogen bonds
        if sym_curr == 'O':
            continue
            
        for neighbor in range(len(atoms)):
            if neighbor in visited:
                continue
            sym_neigh = atoms[neighbor].symbol
            if sym_neigh not in allowed_symbols:
                continue
                
            d = atoms.get_distance(curr, neighbor, mic=True)
            
            # Covalent thresholds
            if 'H' in (sym_curr, sym_neigh):
                thresh = 1.35
            elif 'O' in (sym_curr, sym_neigh):
                thresh = 1.75
            else:
                thresh = 2.1
                
            if d < thresh:
                visited.add(neighbor)
                frontier.append(neighbor)
    return sorted(list(visited))

def build_monolayer(base_dir, isomer, supercell_size=(4,4), n_sam=32, dimer_ids=[1,2,3,4], peel_substrate=True, collision_threshold=1.8):
    """
    Constructs a randomized, collision-free high-density SAM monolayer supercell on a crystalline substrate.
    Applies rigid-body centering, correct stoichiometry (no duplicate surface protons), and optional substrate peeling.
    """
    # 1. Load structures
    dimer_dir = os.path.join(base_dir, f"{isomer}_dimer_optimized_conformations")
    input_1x1 = os.path.join(base_dir, f"{isomer}-ITO.cif")
    output_cif = os.path.join(base_dir, f"{isomer}_{supercell_size[0]}x{supercell_size[1]}_md_peeled.cif" if peel_substrate else f"{isomer}_{supercell_size[0]}x{supercell_size[1]}_md.cif")
    
    print("1. Reading 1x1 structure and dimer conformations...")
    atoms_1x1 = read_typed_structure(input_1x1)
    cell_1x1 = atoms_1x1.get_cell()
    inv_cell_1x1 = np.linalg.inv(cell_1x1)
    a_vec = cell_1x1[0]
    b_vec = cell_1x1[1]
    
    # 2. Extract standing monomer conformations from optimized dimer structures
    standing_confs = []
    print(f"Extracting monomer conformations from dimer clusters: {dimer_ids}...")
    for k in dimer_ids:
        cif_path = os.path.join(dimer_dir, f"{isomer}_dimer_cluster_{k}.cif")
        if os.path.exists(cif_path):
            dimer = read_typed_structure(cif_path)
            # Find the starting index of SAM in dimer dynamically
            p_indices = [i for i, a in enumerate(dimer) if a.symbol == 'P']
            assert len(p_indices) == 2, f"Expected 2 P atoms in dimer, got {len(p_indices)}"
            
            mol1_indices = get_sam_component_no_o_neighbors(dimer, p_indices[0])
            unwrap_substructure(dimer, mol1_indices, p_indices[0])
            mol1 = dimer[mol1_indices].copy()
            
            mol2_indices = get_sam_component_no_o_neighbors(dimer, p_indices[1])
            unwrap_substructure(dimer, mol2_indices, p_indices[1])
            mol2 = dimer[mol2_indices].copy()
            
            standing_confs.append(mol1)
            standing_confs.append(mol2)
            
    print(f"Total extracted standing monomer conformations: {len(standing_confs)}")
    if len(standing_confs) == 0:
        raise ValueError("No monomer conformations were successfully extracted! Please check --dimers input.")
        
    # 3. Setup supercell substrate
    metals = ['In', 'Sn', 'Ti', 'Zn', 'Al', 'Si', 'Fe', 'Cu', 'Ni', 'Ag', 'Au']
    p_idx_1x1 = [i for i, atom in enumerate(atoms_1x1) if atom.symbol == 'P'][0]
    p_o_indices_1x1 = [i for i, atom in enumerate(atoms_1x1) if atom.symbol == 'O' and atoms_1x1.get_distance(p_idx_1x1, i, mic=True) < 2.0]
    
    sam_indices_no_surf_h = {p_idx_1x1}
    from collections import deque
    queue = deque([p_idx_1x1])
    while queue:
        curr = queue.popleft()
        for n in range(len(atoms_1x1)):
            if n in sam_indices_no_surf_h:
                continue
            if atoms_1x1[n].symbol in metals:
                continue
            if atoms_1x1[n].symbol == 'O' and n not in p_o_indices_1x1:
                continue
            d = atoms_1x1.get_distance(curr, n, mic=True)
            if d < 1.95:
                sam_indices_no_surf_h.add(n)
                queue.append(n)
                
    sam_indices = sorted(list(sam_indices_no_surf_h))
    substrate_indices = [i for i in range(len(atoms_1x1)) if i not in sam_indices]
    
    substrate_1x1 = atoms_1x1[substrate_indices]
    in_indices_1x1 = [i for i, atom in enumerate(substrate_1x1) if atom.symbol == 'In' and atom.position[2] > 7.0]
    
    print(f"Searching for triads in 1x1 substrate...")
    candidate_centroids_1x1 = []
    for i, j, k in itertools_combinations(range(len(in_indices_1x1)), 3):
        idx_i, idx_j, idx_k = in_indices_1x1[i], in_indices_1x1[j], in_indices_1x1[k]
        
        d_ij = substrate_1x1.get_distance(idx_i, idx_j, mic=True)
        if not (3.2 < d_ij < 4.2): continue
        
        d_ik = substrate_1x1.get_distance(idx_i, idx_k, mic=True)
        if not (3.2 < d_ik < 4.2): continue
        
        d_jk = substrate_1x1.get_distance(idx_j, idx_k, mic=True)
        if not (3.2 < d_jk < 4.2): continue
        
        pos_i = substrate_1x1.positions[idx_i]
        
        pos_j_raw = substrate_1x1.positions[idx_j]
        d_vec_j = pos_j_raw - pos_i
        frac_j = np.dot(d_vec_j, inv_cell_1x1)
        frac_j = frac_j - np.round(frac_j)
        pos_j = pos_i + np.dot(frac_j, cell_1x1)
        
        pos_k_raw = substrate_1x1.positions[idx_k]
        d_vec_k = pos_k_raw - pos_i
        frac_k = np.dot(d_vec_k, inv_cell_1x1)
        frac_k = frac_k - np.round(frac_k)
        pos_k = pos_i + np.dot(frac_k, cell_1x1)
        
        centroid = (pos_i + pos_j + pos_k) / 3.0
        candidate_centroids_1x1.append(centroid)
        
    print(f"Found {len(candidate_centroids_1x1)} candidate triad centroids in 1x1 unit cell.")
    
    # Tile substrate to supercell
    N_a, N_b = supercell_size
    substrate_super = ase.Atoms()
    for x in range(N_a):
        for y in range(N_b):
            block = substrate_1x1.copy()
            block.translate(x * a_vec + y * b_vec)
            substrate_super.extend(block)
            
    # Tile centroids to supercell
    triad_centroids = []
    for x in range(N_a):
        for y in range(N_b):
            for c in candidate_centroids_1x1:
                triad_centroids.append(c + x * a_vec + y * b_vec)
                
    print(f"Total tiled triad centroids in {N_a}x{N_b} supercell: {len(triad_centroids)}")
    
    # Define cell vectors
    cell_super = cell_1x1.copy()
    cell_super[0] *= N_a
    cell_super[1] *= N_b
    
    # 4. Random Sequential Adsorption (RSA) Placement
    placed_sam = ase.Atoms()
    placed_sam.set_cell(cell_super)
    placed_sam.set_pbc([True, True, True])
    
    shuffled_centroids = triad_centroids.copy()
    random.shuffle(shuffled_centroids)
    
    t_start = time.time()
    placed_count = 0
    
    # Helper to check collision with periodic boundary
    # Helper to check collision with periodic boundary using vectorized MIC
    def check_collision(new_mol, placed_mols, heavy_thresh, any_thresh):
        if len(placed_mols) == 0:
            return False
        cell = cell_super
        inv_cell = np.linalg.inv(cell)
        
        pos_new = new_mol.positions
        pos_old = placed_mols.positions
        
        diff = pos_new[:, None, :] - pos_old[None, :, :]
        frac_diff = np.dot(diff, inv_cell)
        frac_diff = frac_diff - np.round(frac_diff)
        d_mic = np.dot(frac_diff, cell)
        dists = np.linalg.norm(d_mic, axis=2)
        
        sym_new = np.array([a.symbol for a in new_mol])
        sym_old = np.array([a.symbol for a in placed_mols])
        
        heavy_new = (sym_new != 'H')
        heavy_old = (sym_old != 'H')
        
        # Heavy-heavy collision check
        heavy_dists = dists[np.ix_(heavy_new, heavy_old)]
        if np.any(heavy_dists < heavy_thresh):
            return True
            
        # Any-any collision check
        if np.any(dists < any_thresh):
            return True
            
        return False
        
    current_threshold = collision_threshold
    max_retries = 50
    placed_sam = None
    placed_count = 0
    
    for attempt in range(max_retries):
        # Slowly decrease heavy-heavy threshold:
        # keep at collision_threshold for first 10 attempts, then decrease by 0.1 A every 10 attempts.
        current_threshold = collision_threshold - (attempt // 10) * 0.1
        if current_threshold < 1.3:
            current_threshold = 1.3
            
        placed_sam = ase.Atoms()
        placed_sam.set_cell(cell_super)
        placed_sam.set_pbc([True, True, True])
        
        shuffled_centroids = triad_centroids.copy()
        random.shuffle(shuffled_centroids)
        placed_count = 0
        
        t_start = time.time()
        for c_pos in shuffled_centroids:
            if placed_count >= n_sam:
                break
                
            mol = random.choice(standing_confs).copy()
            mol.set_cell(cell_super)
            mol.set_pbc([True, True, True])
            
            p_rel_idx = [i for i, a in enumerate(mol) if a.symbol == 'P'][0]
            p_offset = mol.positions[p_rel_idx].copy()
            xy_translation = np.array([c_pos[0] - p_offset[0], c_pos[1] - p_offset[1], 0.0])
            mol.translate(xy_translation)
            mol.rotate(random.uniform(0, 360), 'z', center=c_pos)
            
            # Check collision with substrate (soft 1.0 A heavy, 1.0 A any-any)
            if check_collision(mol, substrate_super, 1.0, 1.0):
                continue
            # Check collision with already placed molecules (current_threshold heavy, 1.1 A any-any)
            if check_collision(mol, placed_sam, current_threshold, 1.1):
                continue
                
            placed_sam.extend(mol)
            placed_count += 1
            
        t_end = time.time()
        print(f"RSA Monolayer placement (attempt {attempt+1}, threshold={current_threshold:.2f} A): placed {placed_count}/{n_sam} in {t_end - t_start:.2f} seconds.", flush=True)
        if placed_count >= n_sam:
            break
            
    if placed_sam is None or placed_count < n_sam:
        raise RuntimeError(f"Failed to place all {n_sam} molecules after {max_retries} attempts.")
        
    # Combine substrate and SAM
    combined = substrate_super + placed_sam
    combined.set_cell(cell_super)
    combined.set_pbc([True, True, True])
    
    # 6. Center structure
    sam_z = combined.positions[len(substrate_super):, 2]
    sam_z_mid = (np.max(sam_z) + np.min(sam_z)) / 2.0
    shift_cart = np.array([0.0, 0.0, cell_super[2, 2]/2.0 - sam_z_mid])
    combined.translate(shift_cart)
    combined.wrap()
    
    # 7. Substrate Peeling
    if peel_substrate:
        print("\n3. Applying Substrate Peeling...")
        sub_super_indices = range(len(substrate_super))
        sub_super_atoms = combined[sub_super_indices]
        
        metal_indices = [i for i, a in enumerate(sub_super_atoms) if a.symbol in ['In', 'Sn']]
        metal_z = sub_super_atoms.positions[metal_indices, 2]
        
        metal_layers = []
        for z in sorted(metal_z):
            if not metal_layers or abs(z - metal_layers[-1]) > 1.0:
                metal_layers.append(z)
                
        print("Substrate metal layers detected at Z coordinates:", metal_layers)
        cutoff_bottom_mid = (metal_layers[0] + metal_layers[1]) / 2.0
        cutoff_mid_top = (metal_layers[1] + metal_layers[2]) / 2.0
        
        sub_z = sub_super_atoms.positions[:, 2]
        bottom_indices = np.where(sub_z < cutoff_bottom_mid)[0]
        mid_indices = np.where((sub_z >= cutoff_bottom_mid) & (sub_z < cutoff_mid_top))[0]
        top_indices = np.where(sub_z >= cutoff_mid_top)[0]
        
        print(f"Substrate Atoms: Bottom={len(bottom_indices)} (deleted), Middle={len(mid_indices)} (frozen), Top={len(top_indices)} (active).")
        
        keep_indices = [i for i in range(len(combined)) if i not in bottom_indices]
        peeled_atoms = combined[keep_indices]
        
        write(output_cif, peeled_atoms)
        print(f"Peeled structure written successfully to: {output_cif}. Total atoms: {len(peeled_atoms)}.")
        return peeled_atoms
    else:
        write(output_cif, combined)
        print(f"Structure written successfully to: {output_cif}. Total atoms: {len(combined)}.")
        return combined

def itertools_combinations(iterable, r):
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        yield tuple(pool[i] for i in indices)

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    method_parser = argparse.ArgumentParser(add_help=False)
    method_parser.add_argument(
        "--method",
        choices=(
            "rsa",
            "sequential-growth",
            "headgroup-fit",
            "layer-prepare",
            "layer-restore",
        ),
        default="rsa",
    )
    method_args, remaining = method_parser.parse_known_args(argv)
    if method_args.method == "headgroup-fit":
        from monolayer_headgroup_fit import headgroup_fit_main

        return headgroup_fit_main(remaining)
    if method_args.method == "sequential-growth":
        from monolayer_sequential_growth import sequential_growth_main

        return sequential_growth_main(remaining)
    if method_args.method == "layer-prepare":
        return prepare_layer_package(remaining)
    if method_args.method == "layer-restore":
        return restore_layer(remaining)

    parser = argparse.ArgumentParser(description="Build customized randomized high-density SAM monolayer supercell.")
    parser.add_argument(
        "--method",
        choices=(
            "rsa",
            "sequential-growth",
            "headgroup-fit",
            "layer-prepare",
            "layer-restore",
        ),
        default="rsa",
    )
    parser.add_argument("--base-dir", type=str, default=".", help="Workspace base directory.")
    parser.add_argument("--isomer", type=str, required=True, help="SAM isomer/project name.")
    parser.add_argument("--supercell", type=str, default="4,4", help="Supercell size Nx,Ny (e.g. 4,4).")
    parser.add_argument("--n-sam", type=int, default=32, help="Number of SAM molecules to place.")
    parser.add_argument("--dimers", type=str, default="1,2,3,4", help="Comma-separated dimer cluster IDs to use (e.g., 1,2,7,8,9).")
    parser.add_argument("--peel", action="store_true", default=True, help="Apply substrate peeling.")
    parser.add_argument("--no-peel", action="store_false", dest="peel", help="Disable substrate peeling.")
    parser.add_argument("--collision-threshold", type=float, default=1.8, help="Heavy-heavy collision check threshold in Angstroms.")
    args = parser.parse_args(argv)
    
    supercell_size = tuple(map(int, args.supercell.split(',')))
    dimer_ids = list(map(int, args.dimers.split(',')))
    
    build_monolayer(
        base_dir=args.base_dir,
        isomer=args.isomer,
        supercell_size=supercell_size,
        n_sam=args.n_sam,
        dimer_ids=dimer_ids,
        peel_substrate=args.peel,
        collision_threshold=args.collision_threshold
    )

if __name__ == '__main__':
    raise SystemExit(main())
