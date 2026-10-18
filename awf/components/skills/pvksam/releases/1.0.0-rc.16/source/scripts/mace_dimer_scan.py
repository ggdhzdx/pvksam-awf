import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import time
import numpy as np
import torch
import ase
from ase.io import read, write

from ase.constraints import FixAtoms
import argparse
import concurrent.futures
from pathlib import Path

# Define worker initialization and worker functions at the module level for multiprocessing
calc = None

def init_worker(model_path, device):
    global calc
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from mace.calculators import MACECalculator
    calc = MACECalculator(model_paths=model_path, device=device, default_dtype="float64")

def single_point_worker(args):
    idx, atoms, active_indices, constraint = args
    global calc
    atoms_active = atoms[active_indices]
    atoms_active.set_constraint(constraint)
    atoms_active.calc = calc
    try:
        energy = atoms_active.get_potential_energy()
        return {"idx": idx, "energy": energy}
    except Exception as e:
        return {"idx": idx, "energy": 9999.0, "error": str(e)}
    finally:
        pass

def relax_worker(args):
    idx, atoms, active_indices, constraint, max_steps, cap_step = args
    global calc
    from ase.optimize import LBFGS
    
    atoms_active = atoms[active_indices]
    atoms_active.set_constraint(constraint)
    atoms_active.calc = calc
    
    try:
        opt = LBFGS(atoms_active, logfile=None, maxstep=cap_step)
        opt.run(fmax=0.08, steps=max_steps)
        
        # Only return the positions of active_indices and energy to keep serialization light
        positions_active = atoms_active.positions.copy()
        energy = atoms_active.get_potential_energy()
        return {
            "idx": idx,
            "positions_active": positions_active,
            "energy": energy
        }
    except Exception as e:
        return {"idx": idx, "energy": 9999.0, "error": str(e)}
    finally:
        pass

def stage4_worker(args):
    idx, atoms, active_indices, constraint, local_freeze_indices = args
    global calc
    from ase.optimize import LBFGS
    import numpy as np
    
    atoms_active = atoms[active_indices]
    atoms_active.set_constraint(constraint)
    atoms_active.calc = calc
    
    try:
        opt = LBFGS(atoms_active, logfile=None)
        opt.run(fmax=0.08, steps=300)
        
        # Write back relaxed coordinates inside worker to evaluate full cell energy
        atoms.positions[active_indices] = atoms_active.positions
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        
        forces = atoms_active.get_forces()
        free_forces = np.delete(forces, local_freeze_indices, axis=0)
        max_force = np.max(np.linalg.norm(free_forces, axis=1))
        
        return {
            "idx": idx,
            "positions_active": atoms_active.positions.copy(),
            "energy": energy,
            "max_force": max_force
        }
    except Exception as e:
        return {"idx": idx, "energy": 9999.0, "error": str(e)}
    finally:
        pass


def main():
    # 1. Parse Arguments
    parser = argparse.ArgumentParser(description="MACE Dimer Conformation Funnel Scan for SAM Isomers")
    parser.add_argument("--isomer", type=str, required=True, help="Isomer name, e.g., dbf21id, dbf32id, dbf43id")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel worker processes (default: 8)")
    parser.add_argument("--model", type=Path, required=True, help="Path to the MACE model file")
    parser.add_argument("--workdir", type=Path, default=Path("."), help="Workspace containing dimer input and outputs")
    args = parser.parse_args()
    isomer = args.isomer
    num_workers = args.workers
    workspace = args.workdir.resolve()

    # 2. Get hardware and configuration info
    model_path = str(args.model.resolve())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"CUDA Available: {torch.cuda.is_available()} (using {device})", flush=True)

    # 3. Read unoptimized dimer structure
    dimer_cif_path = workspace / f"{isomer}_dimer_unopt.cif"
    atoms_dimer = read(dimer_cif_path)
    cell = atoms_dimer.get_cell()

    # Define indices
    substrate_indices = list(range(480))
    sam1_indices = list(range(480, 526))
    sam2_indices = list(range(526, 572))
    surf_h_indices = list(range(572, 576))
    sam_dimer_indices = sam1_indices + sam2_indices + surf_h_indices

    # Find coordinating O atoms of both SAMs
    p1_idx = [i for i in sam1_indices if atoms_dimer[i].symbol == 'P'][0]
    sam1_o_indices = [i for i in sam1_indices if atoms_dimer[i].symbol == 'O' and atoms_dimer.get_distance(p1_idx, i, mic=True) < 2.0]
    p2_idx = [i for i in sam2_indices if atoms_dimer[i].symbol == 'P'][0]
    sam2_o_indices = [i for i in sam2_indices if atoms_dimer[i].symbol == 'O' and atoms_dimer.get_distance(p2_idx, i, mic=True) < 2.0]

    # Find path P -> S for both SAMs
    def build_adjacency_sam(atoms, indices):
        adj = {i: [] for i in indices}
        for i in indices:
            for j in indices:
                if i >= j:
                    continue
                d = atoms.get_distance(i, j, mic=True)
                elem_i, elem_j = atoms[i].symbol, atoms[j].symbol
                cutoff = 1.95 if (elem_i == 'P' or elem_j == 'P' or elem_i == 'S' or elem_j == 'S') else 1.65
                if elem_i == 'H' or elem_j == 'H':
                    cutoff = 1.35
                if d < cutoff:
                    adj[i].append(j)
                    adj[j].append(i)
        return adj

    adj1 = build_adjacency_sam(atoms_dimer, sam1_indices)
    adj2 = build_adjacency_sam(atoms_dimer, sam2_indices)

    # BFS paths
    s1_idx = [i for i in sam1_indices if atoms_dimer[i].symbol == 'S'][0]
    bfs_q1 = [[p1_idx]]
    vis1 = {p1_idx}
    path_sam1 = None
    while bfs_q1:
        path = bfs_q1.pop(0)
        curr = path[-1]
        if curr == s1_idx:
            path_sam1 = path
            break
        for nbr in adj1[curr]:
            if nbr not in vis1:
                vis1.add(nbr)
                bfs_q1.append(path + [nbr])

    s2_idx = [i for i in sam2_indices if atoms_dimer[i].symbol == 'S'][0]
    bfs_q2 = [[p2_idx]]
    vis2 = {p2_idx}
    path_sam2 = None
    while bfs_q2:
        path = bfs_q2.pop(0)
        curr = path[-1]
        if curr == s2_idx:
            path_sam2 = path
            break
        for nbr in adj2[curr]:
            if nbr not in vis2:
                vis2.add(nbr)
                bfs_q2.append(path + [nbr])

    C1_1, C2_1, C3_1, C4_1 = path_sam1[1:5]
    C1_2, C2_2, C3_2, C4_2 = path_sam2[1:5]

    def get_downstream_indices(adj, bond_start, bond_end):
        vis = {bond_start, bond_end}
        q = [bond_end]
        downstream = [bond_end]
        while q:
            curr = q.pop(0)
            for nbr in adj[curr]:
                if nbr not in vis:
                    vis.add(nbr)
                    q.append(nbr)
                    downstream.append(nbr)
        return downstream

    downstream1_sam1 = get_downstream_indices(adj1, C1_1, C2_1)
    downstream2_sam1 = get_downstream_indices(adj1, C2_1, C3_1)
    downstream3_sam1 = get_downstream_indices(adj1, C3_1, C4_1)

    downstream1_sam2 = get_downstream_indices(adj2, C1_2, C2_2)
    downstream2_sam2 = get_downstream_indices(adj2, C2_2, C3_2)
    downstream3_sam2 = get_downstream_indices(adj2, C3_2, C4_2)

    # --- DYNAMIC SLAB TRUNCATION ---
    metals = ['In', 'Sn']
    all_metals = [i for i in substrate_indices if atoms_dimer[i].symbol in metals]
    metal_z = atoms_dimer.positions[all_metals, 2]
    metal_layers = []
    for z in sorted(metal_z):
        if not metal_layers or abs(z - metal_layers[-1]) > 1.0:
            metal_layers.append(z)

    if len(metal_layers) >= 2:
        z_cut = (metal_layers[-2] + metal_layers[-1]) / 2.0
    else:
        z_cut = -999.0

    sub_top_indices = [i for i in substrate_indices if atoms_dimer.positions[i, 2] >= z_cut]
    active_indices = sam_dimer_indices + sub_top_indices

    # Freeze constraints
    freeze_original = sub_top_indices + sam1_o_indices + sam2_o_indices + surf_h_indices
    local_freeze_indices = [k for k, idx in enumerate(active_indices) if idx in freeze_original]
    constraint = FixAtoms(indices=local_freeze_indices)

    # --- SINGLE-CONFORMATION DATABASE (GENERATED ON THE FLY FROM INITIAL CIF) ---
    print("\nPre-screening single conformations from initial CIF...", flush=True)
    angles = [180, 120, 60, 0, -60, -120]
    valid_single_dihedrals = []
    
    # We take the initial monomer unit from the template dimer
    sam_mol_unit = atoms_dimer[sam1_indices]
    sam_mol_symbols = [a.symbol for a in sam_mol_unit]
    substrate_unit = atoms_dimer[substrate_indices]
    
    # Setup single collision check against substrate
    n_sam1 = len(sam1_indices)
    n_sub = len(substrate_indices)
    thresholds_sub = np.zeros((n_sam1, n_sub))
    for i in range(n_sam1):
        for j in range(n_sub):
            elem_i, elem_j = sam_mol_symbols[i], substrate_unit[j].symbol
            if elem_i == 'H' and elem_j == 'H':
                thresholds_sub[i, j] = 1.2
            elif elem_i == 'H' or elem_j == 'H':
                thresholds_sub[i, j] = 1.4
            else:
                thresholds_sub[i, j] = 1.8

    # BFS local indices for rotation on sam_mol_unit
    p_idx_local = p1_idx - 480
    c1_l, c2_l, c3_l, c4_l = path_sam1[1:5]
    c1_l -= 480
    c2_l -= 480
    c3_l -= 480
    c4_l -= 480
    n1_idx_local = s1_idx - 480
    
    # downstream local indices
    downstream1_1 = [idx - 480 for idx in downstream1_sam1]
    downstream2_1 = [idx - 480 for idx in downstream2_sam1]
    downstream3_1 = [idx - 480 for idx in downstream3_sam1]

    for a1 in angles:
        for a2 in angles:
            for a3 in angles:
                sam_temp = sam_mol_unit.copy()
                try:
                    sam_temp.set_dihedral(p_idx_local, c1_l, c2_l, c3_l, a1, indices=downstream1_1)
                    sam_temp.set_dihedral(c1_l, c2_l, c3_l, c4_l, a2, indices=downstream2_1)
                    sam_temp.set_dihedral(c2_l, c3_l, c4_l, n1_idx_local, a3, indices=downstream3_1)
                except Exception:
                    continue
                dists = np.linalg.norm(sam_temp.positions[:, None, :] - substrate_unit.positions[None, :, :], axis=2)
                if np.any(dists < thresholds_sub):
                    continue
                
                valid_single_dihedrals.append({
                    "dihedrals": (a1, a2, a3),
                    "atoms": sam_temp
                })

    print(f"Pre-screening found {len(valid_single_dihedrals)} substrate-safe single conformations.", flush=True)

    # --- DIMER SCREENING ---
    # Setup dimer collision screening thresholds
    n_sam_dimer = len(sam_dimer_indices)
    thresholds_dimer = np.zeros((n_sam_dimer, n_sam_dimer))
    symbols_dimer = np.array(atoms_dimer.get_chemical_symbols())[sam_dimer_indices]
    for i in range(n_sam_dimer):
        elem_i = symbols_dimer[i]
        for j in range(n_sam_dimer):
            elem_j = symbols_dimer[j]
            if elem_i == 'H' and elem_j == 'H':
                thresholds_dimer[i, j] = 1.2
            elif elem_i == 'H' or elem_j == 'H':
                thresholds_dimer[i, j] = 1.4
            else:
                thresholds_dimer[i, j] = 1.8  # Calibrated packing cutoff for intermolecular heavy atoms

    # Exclude intramolecular contacts (SAM1 self-distances, SAM2 self-distances, surface protons)
    adj_mask_dimer = np.zeros((n_sam_dimer, n_sam_dimer), dtype=bool)
    for i in range(n_sam_dimer):
        idx_i = sam_dimer_indices[i]
        for j in range(n_sam_dimer):
            idx_j = sam_dimer_indices[j]
            if (idx_i in sam1_indices and idx_j in sam1_indices) or \
               (idx_i in sam2_indices and idx_j in sam2_indices) or \
               (idx_i in surf_h_indices) or (idx_j in surf_h_indices):
                adj_mask_dimer[i, j] = True

    # Exclude frozen anchor atoms and coordinating oxygens from collision detection
    # local indices of anchor atoms: P is index 0-3, O is index 4-7
    # P: 480, sam1_o: [480+x], P: 526, sam2_o: [526+x]
    sam1_anchor_and_o = [p1_idx] + sam1_o_indices
    sam2_anchor_and_o = [p2_idx] + sam2_o_indices
    for i in range(n_sam_dimer):
        idx_i = sam_dimer_indices[i]
        if idx_i in sam1_anchor_and_o or idx_i in sam2_anchor_and_o:
            adj_mask_dimer[i, :] = True
            adj_mask_dimer[:, i] = True

    print("Screening dimer combinations for intermolecular collisions...", flush=True)
    t0 = time.time()
    valid_dimers_meta = []
    
    # Pre-initialize temporary atoms object outside the loop to avoid deepcopy overhead
    temp_sam_dimer = atoms_dimer[sam_dimer_indices].copy()

    # Pre-calculate translation vector from Site 1 to Site 2
    translation_vector = atoms_dimer.positions[p2_idx] - atoms_dimer.positions[p1_idx]
    
    # Loop over all combinations of single conformations
    for i_idx, s1 in enumerate(valid_single_dihedrals):
        d1 = s1["dihedrals"]
        pos1 = s1["atoms"].positions
        for j_idx, s2 in enumerate(valid_single_dihedrals):
            d2 = s2["dihedrals"]
            pos2 = s2["atoms"].positions
            
            # Combine positions into temporary coordinates
            coord = np.zeros((n_sam_dimer, 3))
            # sam1 coordinates (organic)
            coord[:46] = pos1
            # sam2 coordinates (organic)
            coord[46:92] = pos2 + translation_vector
            # surface protons (keep original template values for screening)
            coord[92:] = atoms_dimer.positions[surf_h_indices]
            
            # Distance matrix for SAM dimer
            # Calculate distance matrix with cell (under MIC)
            temp_sam_dimer.positions = coord
            dists = temp_sam_dimer.get_all_distances(mic=True)
            
            # Apply masks
            dists[adj_mask_dimer] = 999.0
            
            if np.any(dists < thresholds_dimer):
                continue
                
            valid_dimers_meta.append({
                "dihedrals": (d1, d2),
                "pos1": pos1,
                "pos2": pos2
            })

    t1 = time.time()
    n_dimers = len(valid_dimers_meta)
    print(f"Dimer collision screening completed in {t1 - t0:.2f} seconds. Valid dimers: {n_dimers} / {len(valid_single_dihedrals)**2}", flush=True)

    # Assemble initial dimer Atoms list
    dimer_list = []
    for idx, meta in enumerate(valid_dimers_meta):
        atoms_temp = atoms_dimer.copy()
        shift_vector = atoms_dimer.positions[sam1_indices[0]] - sam_mol_unit.positions[0]
        
        atoms_temp.positions[sam1_indices] = meta["pos1"] + shift_vector
        atoms_temp.positions[sam2_indices] = meta["pos2"] + shift_vector + translation_vector
        
        dimer_list.append({
            "idx": idx,
            "dihedrals": meta["dihedrals"],
            "atoms": atoms_temp,
            "energy": 0.0,
            "max_force": 999.0
        })

    # --- ADAPTIVE SINGLE-POINT PRE-SCREENING ---
    if len(dimer_list) > 8000:
        print(f"\n>>> Large search space detected ({len(dimer_list)} dimers). Running parallel MACE single-point pre-screening to prune to 8,000 structures...", flush=True)
        t_sp_start = time.time()
        
        sp_args_list = []
        for idx, s in enumerate(dimer_list):
            sp_args_list.append((idx, s["atoms"], active_indices, constraint))
            
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker, initargs=(model_path, device)) as executor:
            futures = {executor.submit(single_point_worker, args): args for args in sp_args_list}
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                completed += 1
                if res is not None:
                    dimer_list[res["idx"]]["energy"] = res["energy"]
                if completed % 5000 == 0 or completed == len(dimer_list):
                    msg = f"  Single-point Progress: {completed}/{len(dimer_list)} evaluated."
                    print(msg, flush=True)
                    try:
                        with open(workspace / "progress.txt", "w") as f:
                            f.write(msg + "\n")
                    except Exception:
                        pass
        
        # Sort and retain top 8,000 structures
        dimer_list.sort(key=lambda x: x["energy"])
        dimer_list = dimer_list[:8000]
        # Re-index remaining structures to match positions in dimer_list
        for i, s in enumerate(dimer_list):
            s["idx"] = i
        n_dimers = len(dimer_list)
        print(f"Pre-screening completed in {time.time() - t_sp_start:.2f} seconds. Retained top {n_dimers} structures.", flush=True)

    # --- FUNNEL PROGRESSIVE SCREENING ---
    print("\n" + "="*80, flush=True)
    print("STARTING 4-STAGE FUNNEL CONFORMATION SCREENING", flush=True)
    print("="*80, flush=True)

    # Parallelized/Sequential relaxation runner
    def run_stage_relaxation_parallel(structures, max_steps, description, cap_step=0.15):
        print(f"\n>>> Running {description} (LBFGS steps={max_steps})...", flush=True)
        t_start = time.time()
        n_structs = len(structures)
        
        if num_workers == 1:
            from mace.calculators import MACECalculator
            from ase.optimize import LBFGS
            calc_local = MACECalculator(model_paths=model_path, device=device, default_dtype="float64")
            completed = 0
            for idx, s in enumerate(structures):
                atoms_active = s["atoms"][active_indices]
                atoms_active.set_constraint(constraint)
                atoms_active.calc = calc_local
                try:
                    opt = LBFGS(atoms_active, logfile=None, maxstep=cap_step)
                    opt.run(fmax=0.08, steps=max_steps)
                    s["atoms"].positions[active_indices] = atoms_active.positions
                    s["energy"] = atoms_active.get_potential_energy()
                except Exception as e:
                    s["energy"] = 9999.0
                completed += 1
                if completed % 100 == 0 or completed == n_structs:
                    percent = completed * 100.0 / n_structs
                    msg = f"  Progress {description}: {completed}/{n_structs} structures relaxed ({percent:.1f}%)."
                    print(msg, flush=True)
                    try:
                        with open(workspace / "progress.txt", "w") as f:
                            f.write(msg + "\n")
                    except Exception:
                        pass
        else:
            # Prepare args
            args_list = []
            for idx, s in enumerate(structures):
                args_list.append((idx, s["atoms"], active_indices, constraint, max_steps, cap_step))
                
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker, initargs=(model_path, device)) as executor:
                completed = 0
                batch_size = 200
                for i in range(0, n_structs, batch_size):
                    batch_args = args_list[i : i + batch_size]
                    futures = {executor.submit(relax_worker, args): args for args in batch_args}
                    for future in concurrent.futures.as_completed(futures):
                        res = future.result()
                        completed += 1
                        if res is not None:
                            if "error" in res:
                                structures[res["idx"]]["energy"] = 9999.0
                            else:
                                idx_local = res["idx"]
                                structures[idx_local]["atoms"].positions[active_indices] = res["positions_active"]
                                structures[idx_local]["energy"] = res["energy"]
                                
                        if completed % 100 == 0 or completed == n_structs:
                            percent = completed * 100.0 / n_structs
                            msg = f"  Progress {description}: {completed}/{n_structs} structures relaxed ({percent:.1f}%)."
                            print(msg, flush=True)
                            try:
                                with open(workspace / "progress.txt", "w") as f:
                                    f.write(msg + "\n")
                            except Exception:
                                pass
                    
        t_end = time.time()
        print(f"Completed in {t_end - t_start:.2f} seconds.", flush=True)

    # STAGE 1: 3-step LBFGS on all structures -> Keep 2,000
    print(f"\n--- STAGE 1: 3-step relaxation on all {n_dimers} structures ---", flush=True)
    run_stage_relaxation_parallel(dimer_list, max_steps=3, description="Stage 1", cap_step=0.20)

    # Sort and filter
    dimer_list.sort(key=lambda x: x["energy"])
    stage1_retained = dimer_list[:2000]
    stage1_discarded = dimer_list[2000:]
    # Re-index for the next stage
    for i, s in enumerate(stage1_retained):
        s["idx"] = i
    e_cut_1 = stage1_retained[-1]["energy"]
    print(f"Stage 1 Cutoff Energy: {e_cut_1:.4f} eV", flush=True)

    # STAGE 2: 5-step LBFGS on 2,000 -> Keep 1,000
    print(f"\n--- STAGE 2: 5-step relaxation on top 2,000 structures ---", flush=True)
    run_stage_relaxation_parallel(stage1_retained, max_steps=5, description="Stage 2", cap_step=0.25)

    stage1_retained.sort(key=lambda x: x["energy"])
    stage2_retained = stage1_retained[:1000]
    stage2_discarded = stage1_retained[1000:]
    for i, s in enumerate(stage2_retained):
        s["idx"] = i
    e_cut_2 = stage2_retained[-1]["energy"]
    print(f"Stage 2 Cutoff Energy: {e_cut_2:.4f} eV", flush=True)

    # STAGE 3: 10-step LBFGS on 1,000 -> Keep 150
    print(f"\n--- STAGE 3: 10-step relaxation on top 1,000 structures ---", flush=True)
    run_stage_relaxation_parallel(stage2_retained, max_steps=10, description="Stage 3", cap_step=0.30)

    stage2_retained.sort(key=lambda x: x["energy"])
    stage3_retained = stage2_retained[:150]
    stage3_discarded = stage2_retained[150:]
    for i, s in enumerate(stage3_retained):
        s["idx"] = i
    e_cut_3 = stage3_retained[-1]["energy"]
    print(f"Stage 3 Cutoff Energy: {e_cut_3:.4f} eV", flush=True)

    # STAGE 4: Full Optimization on 150
    print(f"\n--- STAGE 4: Full Optimization (fmax < 0.08 eV/A) on top 150 structures ---", flush=True)
    t_start_full = time.time()
    
    if num_workers == 1:
        from mace.calculators import MACECalculator
        from ase.optimize import LBFGS
        calc_local = MACECalculator(model_paths=model_path, device=device, default_dtype="float64")
        completed = 0
        for s in stage3_retained:
            atoms_active = s["atoms"][active_indices]
            atoms_active.set_constraint(constraint)
            atoms_active.calc = calc_local
            try:
                opt = LBFGS(atoms_active, logfile=None)
                opt.run(fmax=0.08, steps=300)
                s["atoms"].positions[active_indices] = atoms_active.positions
                s["atoms"].calc = calc_local
                s["energy"] = s["atoms"].get_potential_energy()
                forces = atoms_active.get_forces()
                free_forces = np.delete(forces, local_freeze_indices, axis=0)
                s["max_force"] = np.max(np.linalg.norm(free_forces, axis=1))
            except Exception as e:
                s["energy"] = 9999.0
            completed += 1
            if completed % 5 == 0 or completed == len(stage3_retained):
                percent = completed * 100.0 / len(stage3_retained)
                msg = f"  Progress Stage 4: {completed}/{len(stage3_retained)} structures fully optimized ({percent:.1f}%)."
                print(msg, flush=True)
                try:
                    with open(workspace / "progress.txt", "w") as f:
                        f.write(msg + "\n")
                except Exception:
                    pass
    else:
        stage4_args_list = []
        for idx, s in enumerate(stage3_retained):
            stage4_args_list.append((idx, s["atoms"], active_indices, constraint, local_freeze_indices))
            
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker, initargs=(model_path, device)) as executor:
            futures = {executor.submit(stage4_worker, args): args for args in stage4_args_list}
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                completed += 1
                if res is not None:
                    if "error" in res:
                        stage3_retained[res["idx"]]["energy"] = 9999.0
                    else:
                        idx_local = res["idx"]
                        stage3_retained[idx_local]["atoms"].positions[active_indices] = res["positions_active"]
                        stage3_retained[idx_local]["energy"] = res["energy"]
                        stage3_retained[idx_local]["max_force"] = res["max_force"]
                        
                if completed % 5 == 0 or completed == len(stage3_retained):
                    percent = completed * 100.0 / len(stage3_retained)
                    msg = f"  Progress Stage 4: {completed}/{len(stage3_retained)} structures fully optimized ({percent:.1f}%)."
                    print(msg, flush=True)
                    try:
                        with open(workspace / "progress.txt", "w") as f:
                            f.write(msg + "\n")
                    except Exception:
                        pass
                
    t_end_full = time.time()
    print(f"Stage 4 completed in {t_end_full - t_start_full:.2f} seconds.", flush=True)

    # Final sort
    stage3_retained = [s for s in stage3_retained if s["energy"] < 999.0]
    stage3_retained.sort(key=lambda x: x["energy"])

    # --- FUNNEL LEAK DIAGNOSTICS & TUNING ENGINE ---
    print("\n" + "="*80, flush=True)
    print("FUNNEL LEAK DIAGNOSTIC REPORT", flush=True)
    print("="*80, flush=True)

    # Approximate rank shifts
    ranks_stage1 = {id(s): r for r, s in enumerate(dimer_list)}
    ranks_stage2_new = {id(s): r for r, s in enumerate(stage1_retained)}
    rank_shifts_1to2 = [abs(ranks_stage1[id(s)] - ranks_stage2_new[id(s)]) for s in stage1_retained if id(s) in ranks_stage1]
    avg_shift_1to2 = np.mean(rank_shifts_1to2) if rank_shifts_1to2 else 0.0

    ranks_stage3_new = {id(s): r for r, s in enumerate(stage2_retained)}
    rank_shifts_2to3 = [abs(ranks_stage2_new[id(s)] - ranks_stage3_new[id(s)]) for s in stage2_retained if id(s) in ranks_stage2_new]
    avg_shift_2to3 = np.mean(rank_shifts_2to3) if rank_shifts_2to3 else 0.0

    print(f"Funnel Stability Analysis:", flush=True)
    print(f"  - Stage 1 -> Stage 2 average rank shift: {avg_shift_1to2:.2f} (lower is more stable)", flush=True)
    print(f"  - Stage 2 -> Stage 3 average rank shift: {avg_shift_2to3:.2f} (lower is more stable)", flush=True)

    # Diagnostic leak risk on CUDA
    print("\nEvaluating boundary structures for leak diagnostics...", flush=True)
    # We do a fast evaluation by submitting to a temporary small thread pool or just local GPU evaluations
    # since it's only a few boundaries. Let's do it in a small loop using the local CUDA context.
    # Note: we can initialize calc locally here in the parent process since multiprocessing has finished!
    # Yes! The parent process can now safely initialize its own MACECalculator on CUDA since the multiprocessing pool has been closed!
    from mace.calculators import MACECalculator
    calc_parent = MACECalculator(model_paths=model_path, device=device, default_dtype="float64")
    
    stage1_leakers = []
    # stage1_discarded boundary check
    for s in stage1_discarded[:150]:
        if s["energy"] - e_cut_1 > 0.15:
            break
        try:
            atoms_active = s["atoms"][active_indices]
            atoms_active.set_constraint(constraint)
            atoms_active.calc = calc_parent
            forces = atoms_active.get_forces()
            free_forces = np.delete(forces, local_freeze_indices, axis=0)
            max_f = np.max(np.linalg.norm(free_forces, axis=1))
            if max_f > 1.5:
                stage1_leakers.append(s)
        except Exception:
            pass

    stage2_leakers = []
    for s in stage2_discarded[:100]:
        if s["energy"] - e_cut_2 > 0.1:
            break
        try:
            atoms_active = s["atoms"][active_indices]
            atoms_active.set_constraint(constraint)
            atoms_active.calc = calc_parent
            forces = atoms_active.get_forces()
            free_forces = np.delete(forces, local_freeze_indices, axis=0)
            max_f = np.max(np.linalg.norm(free_forces, axis=1))
            if max_f > 1.0:
                stage2_leakers.append(s)
        except Exception:
            pass

    stage3_leakers = []
    for s in stage3_discarded[:50]:
        if s["energy"] - e_cut_3 > 0.05:
            break
        try:
            atoms_active = s["atoms"][active_indices]
            atoms_active.set_constraint(constraint)
            atoms_active.calc = calc_parent
            forces = atoms_active.get_forces()
            free_forces = np.delete(forces, local_freeze_indices, axis=0)
            max_f = np.max(np.linalg.norm(free_forces, axis=1))
            if max_f > 0.5:
                stage3_leakers.append(s)
        except Exception:
            pass

    print(f"\nLeak Risk Assessment:", flush=True)
    print(f"  - Stage 1 Discarded: found {len(stage1_leakers)} leakers (Leak Risk: {'HIGH' if len(stage1_leakers) > 10 else 'LOW'})", flush=True)
    print(f"  - Stage 2 Discarded: found {len(stage2_leakers)} leakers (Leak Risk: {'HIGH' if len(stage2_leakers) > 5 else 'LOW'})", flush=True)
    print(f"  - Stage 3 Discarded: found {len(stage3_leakers)} leakers (Leak Risk: {'HIGH' if len(stage3_leakers) > 2 else 'LOW'})", flush=True)

    print(f"\nParameter Tuning Recommendations:", flush=True)
    advice_given = False
    if len(stage1_leakers) > 10:
        print(f"  [!] Stage 1 steps (3 steps) may be too short. Action: Increase Stage 1 steps to 5, or increase Stage 1 retention from 2000 to 3000.", flush=True)
        advice_given = True
    if len(stage2_leakers) > 5:
        print(f"  [!] Stage 2 retention (1000) may be too tight. Action: Increase Stage 2 retention to 1200, or increase Stage 2 steps to 8.", flush=True)
        advice_given = True
    if len(stage3_leakers) > 2:
        print(f"  [!] Stage 3 retention (150) may be too tight. Action: Increase Stage 3 retention to 250.", flush=True)
        advice_given = True
    if not advice_given:
        print(f"  [✓] Funnel parameters are highly optimal! No significant leak risks detected.", flush=True)

    # --- CLUSTERING AND OUTPUT ---
    print("\nPerforming clustering on fully optimized dimers...", flush=True)
    clusters = []
    s_idx_local = [k for k, i in enumerate(sam1_indices) if atoms_dimer[i].symbol == 'S'][0]

    for item in stage3_retained:
        found_cluster = False
        for cluster in clusters:
            rep = cluster["representative"]
            d_energy = abs(item["energy"] - rep["energy"])
            
            z_s1 = item["atoms"].positions[sam1_indices[s_idx_local], 2]
            z_s2 = item["atoms"].positions[sam2_indices[s_idx_local], 2]
            rep_z_s1 = rep["atoms"].positions[sam1_indices[s_idx_local], 2]
            rep_z_s2 = rep["atoms"].positions[sam2_indices[s_idx_local], 2]
            d_height = max(abs(z_s1 - rep_z_s1), abs(z_s2 - rep_z_s2))
            
            if d_energy < 0.05 and d_height < 0.4:
                cluster["members"].append(item)
                found_cluster = True
                break
        if not found_cluster:
            clusters.append({
                "representative": item,
                "members": [item]
            })

    clusters.sort(key=lambda x: x["representative"]["energy"])

    # Output folder
    opt_dir = workspace / f"{isomer}_dimer_optimized_conformations"
    os.makedirs(opt_dir, exist_ok=True)
    for f in os.listdir(opt_dir):
        if f.endswith(".cif"):
            os.remove(os.path.join(opt_dir, f))

    print("\n" + "="*80, flush=True)
    print("DIMER CONFORMATION CLUSTERING RESULTS (SORTED BY ENERGY)", flush=True)
    print("="*80, flush=True)
    print(f"{'ID':<4} | {'Energy (eV)':<12} | {'Dihedrals SAM1':<20} | {'Dihedrals SAM2':<20} | {'Population':<10}", flush=True)
    print("-"*80, flush=True)
    if clusters:
        min_energy = clusters[0]["representative"]["energy"]
        for c_idx, cluster in enumerate(clusters):
            rep = cluster["representative"]
            rel_energy = rep["energy"] - min_energy
            d1, d2 = rep["dihedrals"]
            pop = len(cluster["members"])
            print(f"C{c_idx+1:<3} | {rel_energy:10.4f} | {str(d1):<20} | {str(d2):<20} | {pop:<10}", flush=True)
            
            # Save representative to CIF
            rep_atoms = rep["atoms"].copy()
            rep_atoms.calc = None
            rep_atoms.set_constraint()
            filename = f"{isomer}_dimer_cluster_{c_idx+1}.cif"
            write(os.path.join(opt_dir, filename), rep_atoms)
            
        global_min = clusters[0]["representative"]["atoms"].copy()
        global_min.calc = None
        global_min.set_constraint()
        write(workspace / f"{isomer}_dimer_global_min.cif", global_min)
        print(f"\nSaved global minimum dimer to {isomer}_dimer_global_min.cif", flush=True)
    else:
        print("No dimer conformations optimized.", flush=True)
    print("="*80, flush=True)
    print(f"Dimer search run completed. Representative structures of all {len(clusters)} clusters saved to {opt_dir}", flush=True)


if __name__ == '__main__':
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()

