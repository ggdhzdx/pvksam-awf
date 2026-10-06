#!/usr/bin/env python3
"""SAM relaxation: legacy CLI plus PVKSAM restrained backend / SAM受约束优化.

Use --restrained followed by --help for the parameterized ASE backend. Existing
legacy flags retain their historical behavior. PVKSAM never uses legacy inference.
"""
import os
import argparse
import numpy as np
from ase.io import read, write
from ase.constraints import FixAtoms, Hookean
from ase.optimize import LBFGS

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

def legacy_main():
    from mace.calculators import MACECalculator
    parser = argparse.ArgumentParser(description="Zero-temperature LBFGS geometry relaxation for SAM-ITO.")
    parser.add_argument("--input", type=str, required=True, help="Input peeled CIF file.")
    parser.add_argument("--model", type=str, required=True, help="MACE model file path.")
    parser.add_argument("--output", type=str, required=True, help="Output relaxed CIF file.")
    parser.add_argument("--freeze-z", type=float, default=None, help="Cutoff Z-coordinate below which substrate atoms are frozen (defaults to dynamic detection).")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu.")
    parser.add_argument("--fmax", type=float, default=0.08, help="Maximum force convergence threshold.")
    parser.add_argument("--restrain-pc", action="store_true", help="Apply Hookean harmonic restraints to P-C1 bonds during optimization.")
    parser.add_argument("--restrain-all-bonds", action="store_true", help="Apply Hookean restraints to all covalent bonds during initial relaxation to prevent bond snapping.")
    args = parser.parse_args()
    
    print(f"Reading structure from {args.input}...")
    atoms = read(args.input)
    
    print(f"Loading MACE model from {args.model} on {args.device}...")
    calc = MACECalculator(model_paths=args.model, device=args.device, default_dtype="float32")
    atoms.calc = calc
    
    # Dynamically detect SAM molecules and substrate boundaries
    c_count = len([a for a in atoms if a.symbol == 'C'])
    n_mols = c_count // 22
    atoms_per_mol = 46
    first_sam_idx = len(atoms) - n_mols * atoms_per_mol
    print(f"Detected {n_mols} SAM molecules (each with {atoms_per_mol} atoms). SAM starts at index {first_sam_idx}.")
    
    if args.freeze_z is not None:
        freeze_cutoff = args.freeze_z
    else:
        freeze_cutoff = get_substrate_freeze_cutoff(atoms, first_sam_idx)
        
    freeze_indices = [i for i in range(first_sam_idx) if atoms.positions[i, 2] < freeze_cutoff]
    print(f"Freezing {len(freeze_indices)} bottom substrate atoms (Z < {freeze_cutoff:.4f} A) out of {len(atoms)} total atoms.")
    
    constraints = []
    constraints.append(FixAtoms(indices=freeze_indices))
    
    if args.restrain_pc:
        print(f"Applying Hookean restraints (k=100.0 eV/A^2, rt=1.82 A) to all {n_mols} P-C1 bonds during relaxation.")
        for mol_idx in range(n_mols):
            start = first_sam_idx + mol_idx * atoms_per_mol
            p_idx, n_idx, s_idx, butyl_carbons = detect_molecule_indices(atoms, start)
            c1_idx = butyl_carbons[0]
            constraints.append(Hookean(a1=p_idx, a2=c1_idx, k=100.0, rt=1.82))
            
    all_bond_constraints = []
    if args.restrain_all_bonds:
        print(f"Generating all-covalent-bond Hookean restraints for {n_mols} molecules to prevent bond snapping...")
        for mol_idx in range(n_mols):
            start = first_sam_idx + mol_idx * atoms_per_mol
            sam_indices = list(range(start, start + atoms_per_mol))
            for i in range(atoms_per_mol):
                idx_i = sam_indices[i]
                sym_i = atoms[idx_i].symbol
                for j in range(i + 1, atoms_per_mol):
                    idx_j = sam_indices[j]
                    sym_j = atoms[idx_j].symbol
                    d_raw = atoms.get_distance(idx_i, idx_j, mic=True)
                    if d_raw < 1.95:  # covalent bond
                        k_val = 10.0 if ('H' in (sym_i, sym_j)) else 20.0
                        all_bond_constraints.append(Hookean(a1=idx_i, a2=idx_j, k=k_val, rt=d_raw))
                        
    def opt_callback():
        fmax_val = np.sqrt((atoms.get_forces() ** 2).sum(axis=1)).max()
        print(f"LBFGS: fmax = {fmax_val:.4f} eV/A | Epot = {atoms.get_potential_energy():.4f} eV", flush=True)

    if args.restrain_all_bonds:
        print("Applying all-covalent-bond Hookean restraints active during the entire relaxation...")
        atoms.set_constraint(constraints + all_bond_constraints)
    else:
        atoms.set_constraint(constraints)
        
    print("Starting LBFGS optimization...")
    opt = LBFGS(atoms, logfile=None, maxstep=0.2)
    opt.attach(opt_callback)
    opt.run(fmax=args.fmax, steps=300)
    
    # Remove restraints before saving structure (leave only FixAtoms constraint)
    atoms.set_constraint(FixAtoms(indices=freeze_indices))
    
    write(args.output, atoms)
    print(f"Relaxed structure saved to: {args.output}")


# Parameterized backend consolidated from the verified task adapter.
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms, Hookean
from ase.geometry import find_mic
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.optimize import LBFGS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from sam_lammps import molecular_bonds


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def restraint_terms(atoms, bonds):
    if not bonds: return 0., np.zeros((len(atoms), 3))
    ij = np.asarray([[b['i'], b['j']] for b in bonds], dtype=int)
    v, r = find_mic(atoms.positions[ij[:, 1]] - atoms.positions[ij[:, 0]],
                    atoms.cell, atoms.pbc)
    if np.any(r <= 1e-12):
        raise ValueError('Zero-length restrained pair')
    delta = r - np.asarray([b['r0_A'] for b in bonds])
    unilateral = np.asarray([b['kind'] == 'interface' for b in bonds])
    delta[unilateral] = np.maximum(delta[unilateral], 0.)
    k = np.asarray([b['k_eV_A2'] for b in bonds])
    fpair = (k * delta / r)[:, None] * v
    forces = np.zeros((len(atoms), 3))
    np.add.at(forces, ij[:, 0], fpair)
    np.add.at(forces, ij[:, 1], -fpair)
    return float(np.sum(.5 * k * delta**2)), forces


def tetra_terms(atoms, spec):
    """Harmonic angles (radians) + unilateral P/substrate-O repulsion.

    角度 U=1/2*k*(theta-theta0)^2；排斥 U=1/2*k*min(r-rmin,0)^2.
    Reference ligand IDs stay sealed; current distances never redefine them.
    """
    if spec is None or not spec['angles']: return 0., np.zeros((len(atoms),3))
    ar=np.asarray(spec['angles'],float)
    ii,pp,jj=ar[:,:3].astype(int).T
    u,ru=find_mic(atoms.positions[ii]-atoms.positions[pp],atoms.cell,[True,True,False])
    v,rv=find_mic(atoms.positions[jj]-atoms.positions[pp],atoms.cell,[True,True,False])
    u=u/ru[:,None];v=v/rv[:,None]
    cos=np.clip(np.sum(u*v,axis=1),-1.,1.)
    theta=np.arccos(cos);sine=np.sqrt(np.maximum(1.-cos*cos,1e-20))
    delta=theta-ar[:,3];k=spec['angle_k_eV_rad2']
    fi=(k*delta/(sine*ru))[:,None]*(v-cos[:,None]*u)
    fj=(k*delta/(sine*rv))[:,None]*(u-cos[:,None]*v)
    f=np.zeros((len(atoms),3))
    np.add.at(f,ii,fi);np.add.at(f,jj,fj);np.add.at(f,pp,-fi-fj)
    pi=np.asarray(spec['P_indices'],int);oi=np.asarray(spec['substrate_O_indices'],int)
    vv,rr=find_mic((atoms.positions[oi][None,:,:]-atoms.positions[pi][:,None,:]).reshape(-1,3),atoms.cell,[True,True,False])
    dr=np.minimum(rr-spec['repulsion_min_A'],0.)
    kr=spec['repulsion_k_eV_A2'];ff=(kr*dr/rr)[:,None]*vv
    ff=ff.reshape(len(pi),len(oi),3)
    np.add.at(f,pi,ff.sum(axis=1));np.add.at(f,oi,-ff.sum(axis=0))
    return float(.5*k*np.sum(delta**2)+.5*kr*np.sum(dr**2)),f


def tetra_audit(atoms,spec):
    """Predeclared P-only gate; no global molecular topology/height gate."""
    if spec is None:return None
    from validate_monolayer import covalent_cutoff
    sy=atoms.get_chemical_symbols();records=[]
    for p,lig in zip(spec['P_indices'],spec['ligand_indices']):
        v,r=find_mic(atoms.positions-atoms.positions[p],atoms.cell,[True,True,False])
        near=[j for j in range(len(atoms)) if j!=p and sy[j] not in spec.get('metal_elements', ['In','Sn']) and r[j]<covalent_cutoff('P',sy[j])]
        u=v[lig]/r[lig,None]
        weights=np.linalg.solve(np.vstack([u.T,np.ones(4)]),[0,0,0,1])
        ar=[x for x in spec['angles'] if x[1]==p]
        angles=[float(np.arccos(np.clip(v[i]@v[j]/(r[i]*r[j]),-1,1))) for i,_,j,_ in ar]
        deviation=float(np.max(np.abs(np.array(angles)-np.array(ar)[:,3]))*180/np.pi)
        passed=(set(near)==set(lig) and bool(np.all(weights>0)) and deviation<=spec['max_angle_deviation_deg'])
        records.append({'P_1based':p+1,'neighbor_count':len(near),'extra_ids_1based':[j+1 for j in near if j not in lig],
                        'missing_ids_1based':[j+1 for j in lig if j not in near],
                        'angle_min_deg':float(min(angles)*180/np.pi),'angle_max_deg':float(max(angles)*180/np.pi),
                        'max_reference_deviation_deg':deviation,'inside':bool(np.all(weights>0)),'passed':bool(passed)})
    return {'all_passed':all(x['passed'] for x in records),'failed_P_ids_1based':[x['P_1based'] for x in records if not x['passed']],
            'max_reference_deviation_deg':max((x['max_reference_deviation_deg'] for x in records), default=0.),'records':records}


class RestrainedCalculator(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, base, bonds, tetra=None):
        super().__init__()
        self.base, self.bonds, self.tetra = base, bonds, tetra

    def calculate(self, atoms=None, properties=('energy', 'forces'), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.base.calculate(atoms, properties, system_changes)
        e, f = restraint_terms(atoms, self.bonds)
        te,tf=tetra_terms(atoms,self.tetra)
        e+=te;f+=tf
        self.results = {'energy': float(self.base.results['energy']) + e,
                        'forces': np.asarray(self.base.results['forces']) + f}
        self.restraint_energy = e


def check_restraint_implementation(atoms, bonds):
    # Analytic force vs energy gradient; interface overlay vs ASE Hookean.
    probe = atoms.copy()
    probe.positions[bonds[-1]['j'], 0] += .05
    _, f = restraint_terms(probe, bonds)
    i = bonds[-1]['j']
    h = 1e-6
    probe.positions[i, 0] += h
    plus = restraint_terms(probe, bonds)[0]
    probe.positions[i, 0] -= 2*h
    minus = restraint_terms(probe, bonds)[0]
    error = abs(f[i, 0] + (plus-minus)/(2*h))
    if error > 1e-5:
        raise AssertionError(('restraint gradient', error))
    interface = [b for b in bonds if b['kind'] == 'interface']
    probe = atoms.copy()
    probe.positions[sorted({b['j'] for b in interface})] += [.013, .017, .021]
    e_ref, f_ref = 0., np.zeros((len(probe), 3))
    for b in interface:
        c = Hookean(b['i'], b['j'], k=b['k_eV_A2'], rt=b['r0_A'])
        e_ref += c.adjust_potential_energy(probe)
        c.adjust_forces(probe, f_ref)
    e_vec, f_vec = restraint_terms(probe, interface)
    if not np.allclose(f_ref, f_vec, atol=1e-9) or abs(e_ref-e_vec) > 1e-9:
        raise AssertionError('Vectorized interface differs from original ASE Hookean')
    return {'gradient_error_eV_A': error, 'interface_Hookean_equivalence': True}


def replay_lbfgs(opt, path):
    """ASE 3.27 update expects flattened arrays, unlike its replay helper.

    Reconstruct only the history that survives the LBFGS memory bound, leaving
    the final frame for the first new step, just as the native replay intends.
    """
    with Trajectory(str(path), 'r') as traj:
        r0=f0=None
        for i in range(max(0,len(traj)-opt.memory-2),len(traj)-1):
            frame=traj[i]
            pos=frame.get_positions().ravel()
            forces=frame.get_forces().ravel()
            opt.update(pos,forces,r0,f0)
            r0,f0=pos.copy(),forces.copy()
            opt.iteration+=1
        opt.r0,opt.f0=r0,f0
        if not all(np.isfinite(x).all() for x in opt.s+opt.y):
            raise ValueError('Non-finite restored LBFGS history')


def search_plateau(best_history, patience, epsilon_per_sam, molecules):
    """Five-cycle cumulative improvement / 连续窗口累计改善，非逐轮阈值."""
    if patience <= 0 or len(best_history) <= patience:
        return False
    values=np.asarray(best_history,float)
    if not np.isfinite(values).all() or np.any(np.diff(values)>1e-10):
        raise ValueError('Best-energy history must be finite and nonincreasing')
    improvement=values[-patience-1]-values[-1]
    threshold=epsilon_per_sam*molecules
    tolerance=8*np.finfo(float).eps*max(1.,float(np.max(np.abs(values))))
    return bool(improvement < threshold-tolerance)


def anneal_contacts(atoms,a):
    from types import SimpleNamespace
    from validate_monolayer import audit_short_contacts as analyze
    # The diagnostic reads the immutable candidate just exported by the caller.
    settings=SimpleNamespace(substrate_atoms=a.substrate_atoms,surface_h=a.surface_h,
        molecules=a.molecules,mol_size=a.atoms_per_molecule,parent_max=1.25,
        hh=1.2,h_heavy=1.4,heavy=1.8)
    registered={tuple(sorted((b['i'],b['j']))) for b in atoms.calc.bonds if b['kind']=='interface'}
    return analyze(a.output/'candidate-check.extxyz',settings,registered)


def run_anneal(a, atoms, initial):
    """Task adapter for existing ASE Langevin cycle / 复用升温保温冷却流程.

    H mass scaling is the historical acceleration convention (not mass-conserving
    repartition); physical masses are restored in all final structure exports.
    """
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    physical_masses = atoms.get_masses().copy()
    masses = physical_masses.copy()
    masses[atoms.numbers == 1] = a.h_mass
    atoms.set_masses(masses)
    rng = np.random.default_rng(a.seed)
    start = time.monotonic()
    records = []
    best_energy = float('inf')
    best_cycle=None
    best_history=[]
    search_converged=False
    search_enabled=bool(a.convergence_patience and a.mode=='anneal')
    stop_reason='cycle_limit'
    completed_cycles=0
    cycles = 1 if a.mode == 'md-smoke' else a.cycles
    nsteps = 100 if a.mode == 'md-smoke' else a.cycle_steps
    def snapshot(prefix):
        out = atoms.copy()
        out.set_momenta(np.zeros((len(out), 3)))
        out.set_masses(physical_masses)
        write(a.output/(prefix+'.extxyz'), out)
        write(a.output/(prefix+'.cif'), out)
        write(a.output/(prefix+'.data'), out, format='lammps-data', atom_style='atomic',
              specorder=sorted(set(atoms.get_chemical_symbols())), masses=True)
    if search_enabled:
        if np.linalg.norm(atoms.get_forces(),axis=1).max()>=a.fmax:
            raise ValueError('Convergence search requires force-converged input; relax input first')
        best_energy=float(atoms.get_potential_energy());best_cycle=0
        best_history=[best_energy];snapshot('best-relaxed')
        snapshot('candidate-check')
        contact=anneal_contacts(atoms,a);save(a.output/'entry-contact-audit.json',contact)
        if contact['count']:raise ValueError('Resolve entry short contacts before convergence search')
    for cycle in range(1, cycles+1):
        MaxwellBoltzmannDistribution(atoms, temperature_K=500. if a.mode=='md-smoke' else 300., rng=rng)
        dyn = Langevin(atoms, timestep=a.dt*units.fs, temperature_K=300.,
                       friction=.002/units.fs, rng=rng, fixcm=False)
        temperatures = []
        with Trajectory(str(a.output/f'cycle-{cycle:03d}.traj'), 'w', atoms) as traj, \
             (a.output/f'cycle-{cycle:03d}.jsonl').open('w', buffering=1) as log:
            for step in range(1, nsteps+1):
                heat, hold = nsteps//6, nsteps//3
                target = (300.+200.*step/heat if step<=heat else
                          500. if step<=heat+hold else
                          500.-200.*(step-heat-hold)/(nsteps-heat-hold))
                if a.mode=='md-smoke': target=500.
                dyn.set_temperature(temperature_K=target)
                dyn.run(1)
                f = atoms.get_forces()
                if not np.isfinite(atoms.positions).all() or not np.isfinite(f).all():
                    raise ValueError('Non-finite dynamics')
                if not np.array_equal(initial[:a.fixed_atoms], atoms.positions[:a.fixed_atoms]):
                    raise ValueError('Frozen substrate moved during MD')
                if step==1 or step%10==0:
                    rec = {'status':'running', 'stage':'MD', 'cycle':cycle, 'step':step,
                           'time_ps':step*a.dt/1000, 'target_K':target,
                           'temperature_K':atoms.get_temperature(),
                           'energy_eV':atoms.get_potential_energy(),
                           'fmax_eV_A':float(np.linalg.norm(f,axis=1).max()),
                           'elapsed_s':time.monotonic()-start}
                    temperatures.append(rec['temperature_K'])
                    log.write(json.dumps(rec, allow_nan=False)+'\n')
                    save(a.output/'progress.json',rec)
                    traj.write()
                    if step%100==0:
                        if atoms.calc.tetra:
                            gate=tetra_audit(atoms,atoms.calc.tetra)
                            save(a.output/f'cycle-{cycle:03d}-step-{step:04d}-tetra.json',gate)
                            rec['P_tetra_all_passed']=gate['all_passed']
                            rec['P_max_angle_deviation_deg']=gate['max_reference_deviation_deg']
                            if not gate['all_passed']:
                                snapshot(f'cycle-{cycle:03d}-step-{step:04d}-failed-tetra')
                                raise ValueError('Sampled MD P tetrahedral gate failed')
                        print(json.dumps(rec),flush=True)
        if a.mode=='md-smoke':
            if atoms.calc.tetra:
                gate=tetra_audit(atoms,atoms.calc.tetra);save(a.output/'tetra-smoke-audit.json',gate)
                if not gate['all_passed']:raise ValueError('MD smoke P tetrahedral gate failed')
            snapshot('smoke-final')
            save(a.output/'result.json', {'status':'completed','steps':nsteps,
                 'elapsed_s':time.monotonic()-start,'temperature_range_K':[min(temperatures),max(temperatures)],
                 'finite_dynamics':True,'fixed_max_displacement_A':0.,
                 'scope':'Short numerical smoke only; no convergence, chemical or timestep accuracy certification'})
            return
        atoms.set_momenta(np.zeros((len(atoms),3)))
        opt = LBFGS(atoms, logfile=str(a.output/f'cycle-{cycle:03d}-relax.log'),
                    trajectory=str(a.output/f'cycle-{cycle:03d}-relax.traj'),maxstep=a.maxstep)
        def report_opt():
            rec={'status':'running','stage':'post_cycle_relax','cycle':cycle,'step':opt.nsteps,
                 'energy_eV':atoms.get_potential_energy(),
                 'fmax_eV_A':float(np.linalg.norm(atoms.get_forces(),axis=1).max()),
                 'elapsed_s':time.monotonic()-start}
            save(a.output/'progress.json',rec)
            if opt.nsteps%100==0: print(json.dumps(rec),flush=True)
        opt.attach(report_opt,interval=10)
        converged=opt.run(fmax=a.fmax,steps=a.steps)
        if not np.array_equal(initial[:a.fixed_atoms], atoms.positions[:a.fixed_atoms]):
            raise ValueError('Frozen substrate moved during minimization')
        if atoms.calc.tetra:
            gate=tetra_audit(atoms,atoms.calc.tetra);save(a.output/f'cycle-{cycle:03d}-tetra-audit.json',gate)
            if not gate['all_passed']:
                snapshot(f'cycle-{cycle:03d}-failed-tetra')
                raise ValueError('Post-cycle P tetrahedral gate failed')
        snapshot(f'cycle-{cycle:03d}-relaxed')
        energy=atoms.get_potential_energy()
        rec={'cycle':cycle,'energy_eV':energy,'relax_converged':bool(converged),
             'relax_steps':opt.nsteps,'fmax_eV_A':float(np.linalg.norm(atoms.get_forces(),axis=1).max()),
             'temperature_range_K':[min(temperatures),max(temperatures)]}
        if search_enabled:
            snapshot('candidate-check')
            contact=anneal_contacts(atoms,a);save(a.output/f'cycle-{cycle:03d}-contact-audit.json',contact)
            rec['contact_count']=contact['count']
        records.append(rec)
        completed_cycles=cycle
        if search_enabled and (not converged or rec['contact_count']):
            stop_reason='local_force_not_converged' if not converged else 'unresolved_short_contacts'
            save(a.output/'cycles.json',records)
            break
        if energy<best_energy:
            best_energy=energy
            best_cycle=cycle
            snapshot('best-relaxed')
        save(a.output/'cycles.json',records)
        print(json.dumps(rec),flush=True)
        if search_enabled:
            best_history.append(float(best_energy))
            search_converged=search_plateau(best_history,a.convergence_patience,a.energy_per_sam,a.molecules)
            if search_converged:
                stop_reason='best_energy_plateau_and_force_convergence'
                break
    snapshot('last-relaxed')
    if search_enabled:
        save(a.output/'search-checkpoint.json',{'best_history_eV':best_history,'best_cycle':best_cycle,
            'completed_cycles':completed_cycles,'rng_state':rng.bit_generator.state,
            'last_structure_sha256':sha(a.output/'last-relaxed.extxyz'),
            'best_structure_sha256':sha(a.output/'best-relaxed.extxyz'),
            'patience':a.convergence_patience,'epsilon_eV_per_SAM':a.energy_per_sam,
            'molecules':a.molecules,'fmax_eV_A':a.fmax,'stop_reason':stop_reason,
            'continuation_note':'Preserve RNG, full history and prior best on continuation; do not reset plateau window.'})
    save(a.output/'result.json',{'status':'completed' if not search_enabled or search_converged else 'needs_review','cycles':records,
         'search_converged':search_converged if search_enabled else None,'stop_reason':stop_reason,
         'best_cycle':best_cycle,'best_energy_eV':best_energy,
         'MD_time_ps':completed_cycles*nsteps*a.dt/1000,'elapsed_s':time.monotonic()-start,
         'surface_OH_restraints':0,'fixed_max_displacement_A':0.,
         'disabled_structure_checks':'not_run_by_user_instruction'})
    save(a.output/'progress.json',json.loads((a.output/'result.json').read_text()))


def restrained_main(argv=None):
    p = argparse.ArgumentParser(description="PVKSAM restrained ASE relaxation and cyclic annealing / 约束优化和循环退火")
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--interface-bonds', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['benchmark', 'relax', 'md-smoke', 'anneal', 'audit'], required=True)
    p.add_argument('--substrate-atoms', type=int, required=True)
    p.add_argument('--surface-h', type=int, default=0, help='Unrestrained H block after substrate / 表面自由 H 数')
    p.add_argument('--output-prefix', default='relaxed-h0')
    p.add_argument('--molecules', type=int, required=True)
    p.add_argument('--atoms-per-molecule', type=int, required=True)
    p.add_argument('--fixed-atoms', type=int, required=True)
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--fmax', type=float, default=.01)
    p.add_argument('--maxstep', type=float, default=.02)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--restraints-json', type=Path, help='Reuse exact sealed initial restraints on continuation; no bond redetection')
    p.add_argument('--audit-stage', choices=['intake','strict'], default='strict', help='Intake reports extra neighbors; strict requires exactly original four')
    p.add_argument('--repair-sam', type=int, help='Local repair: only this 1-based SAM mobile, all other atoms fixed')
    p.add_argument('--tetra-json', type=Path, help='Sealed P angle and substrate-O exclusion settings / P四面体约束')
    p.add_argument('--replay-trajectory', type=Path, help='Restore LBFGS history from previous segment')
    p.add_argument('--cycles', type=int, default=3, help='Cycle count, or review checkpoint cap in convergence mode')
    p.add_argument('--convergence-patience', type=int, default=0, help='0 keeps fixed-cycle behavior; e.g. 5 enables cumulative plateau stopping')
    p.add_argument('--energy-per-sam', type=float, default=.005, help='Cumulative best-energy improvement tolerance, eV/SAM')
    p.add_argument('--cycle-steps', type=int, default=1500)
    p.add_argument('--dt', type=float, default=2., help='MD time step in fs')
    p.add_argument('--h-mass', type=float, default=4., help='Integration H mass in u; physical mass in exports')
    p.add_argument('--seed', type=int, default=340827)
    a = p.parse_args(argv)
    if a.cycles<1 or a.cycle_steps<6 or a.convergence_patience<0 or a.energy_per_sam<=0:
        p.error('Invalid annealing or convergence parameters')
    if a.convergence_patience and a.mode!='anneal':
        p.error('Convergence search requires anneal mode')
    a.output.mkdir(parents=True, exist_ok=False)
    try:
        if a.mode=='audit':
            if not a.tetra_json:raise ValueError('audit requires --tetra-json')
            spec=json.loads(a.tetra_json.read_text())
            atoms=read_pvksam_structure(a.input)
            report=tetra_audit(atoms,spec)
            if a.audit_stage=='intake':
                report['strict_all_passed']=report['all_passed']
                report['intake_extra_neighbor_P_ids_1based']=[r['P_1based'] for r in report['records'] if r['extra_ids_1based']]
                for r in report['records']:
                    r['strict_passed']=r['passed']
                    r['passed']=bool(not r['missing_ids_1based'] and r['inside'] and r['max_reference_deviation_deg']<=spec['max_angle_deviation_deg'])
                report['all_passed']=all(r['passed'] for r in report['records'])
                report['failed_P_ids_1based']=[r['P_1based'] for r in report['records'] if not r['passed']]
            report['audit_stage']=a.audit_stage
            report.update(source=str(a.input.resolve()),source_sha256=sha(a.input),
                          tetra_sha256=sha(a.tetra_json),worker_sha256=sha(__file__))
            save(a.output/'tetra-audit.json',report)
            print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)
            if not report['all_passed']:raise ValueError('P tetrahedral audit failed')
        else:
            run_restrained(a)
    except BaseException as exc:
        save(a.output / 'failure.json', {'type': type(exc).__name__, 'message': str(exc)})
        raise


def run_restrained(a):
    import torch
    import ase
    import importlib.metadata as md
    from mace.calculators import MACECalculator
    atoms = read_pvksam_structure(a.input)
    if len(atoms) != a.substrate_atoms + a.surface_h + a.molecules*a.atoms_per_molecule:
        raise ValueError('Atom layout mismatch')
    if np.any(atoms.numbers[:a.substrate_atoms] == 1):
        raise ValueError('Input substrate contains H')
    atoms.set_pbc([True, True, False])
    sam_start = a.substrate_atoms + a.surface_h
    if a.surface_h and not a.restraints_json:
        raise ValueError('Post-H input requires sealed restraints')
    if not np.all(atoms.numbers[a.substrate_atoms:sam_start] == 1):
        raise ValueError('Surface H block identity mismatch')
    symbols = np.asarray(atoms.get_chemical_symbols())
    blocks = symbols[sam_start:].reshape(a.molecules, a.atoms_per_molecule)
    if not np.all(blocks == blocks[0]):
        raise ValueError('Molecular atom identity order differs')
    bonds = [{'i': b['metal']-1, 'j': b['donor']-1, 'r0_A': b['target_A'],
              'k_eV_A2': 100., 'kind': 'interface'}
             for b in json.loads(a.interface_bonds.read_text())['bonds']]
    if a.restraints_json:
        sealed = json.loads(a.restraints_json.read_text())['bonds']
        if [b for b in sealed if b['kind']=='interface'] != bonds:
            raise ValueError('Continuation interface restraint identity mismatch')
        bonds=sealed
        internal=[b for b in bonds if b['kind']=='SAM_internal']
        if any(not 0<=b['i']<len(atoms) or not 0<=b['j']<len(atoms) for b in bonds):
            raise ValueError('Invalid restraint indices')
    else:
        internal = molecular_bonds(atoms, symbols, a.substrate_atoms, a.molecules, a.atoms_per_molecule)
        bonds += [{'i': i-1, 'j': j-1, 'r0_A': r, 'k_eV_A2': 10. if has_h else 20.,
                   'kind': 'SAM_internal'} for i,j,r,has_h in internal]
    if any(a.substrate_atoms <= b[key] < sam_start for b in bonds for key in ('i','j')):
        raise ValueError('Surface H must have no restraint')
    save(a.output/'restraints.json', {'formula_internal': 'U=0.5*k*(r-r0)^2',
        'formula_interface': 'U=0.5*100*max(r-r0,0)^2; original ASE Hookean', 'bonds': bonds})
    verification = check_restraint_implementation(atoms, bonds)
    tetra=json.loads(a.tetra_json.read_text()) if a.tetra_json else None
    if tetra:
        save(a.output/'tetra-restraints.json',tetra)
        if any(atoms[i].symbol!='P' for i in tetra['P_indices']):raise ValueError('P identity mismatch')
        if a.mode in ('anneal','md-smoke'):
            gate=tetra_audit(atoms,tetra);save(a.output/'tetra-entry-audit.json',gate)
            if not gate['all_passed']:raise ValueError('P tetrahedral entry gate failed')
    initial = atoms.positions.copy()
    fixed_indices=np.arange(a.fixed_atoms)
    if a.repair_sam:
        lo=sam_start+(a.repair_sam-1)*a.atoms_per_molecule
        fixed_indices=np.r_[np.arange(lo),np.arange(lo+a.atoms_per_molecule,len(atoms))]
    atoms.set_constraint(FixAtoms(indices=fixed_indices))
    write(a.output/'start.extxyz', atoms)
    shutil.copy2(__file__, a.output/'worker.py')
    save(a.output/'request.json', {
        'argv': sys.argv, 'source': str(a.input.resolve()), 'source_sha256': sha(a.input),
        'interface_source': str(a.interface_bonds.resolve()), 'interface_sha256': sha(a.interface_bonds),
        'model_sha256': sha(a.model), 'worker_sha256': sha(__file__),
        'bond_helper_sha256': sha(ROOT/'scripts/sam_lammps.py'),
        'restraints_parent_sha256': sha(a.restraints_json) if a.restraints_json else None,
        'replay_trajectory_sha256': sha(a.replay_trajectory) if a.replay_trajectory else None,
        'surface_H_count': a.surface_h, 'surface_OH_restraint_count': 0,
        'fixed_indices_1based':(fixed_indices+1).tolist(), 'local_repair_SAM':a.repair_sam,
        'atom_count': len(atoms), 'fixed_ids_1based': [1,a.fixed_atoms],
        'mobile_substrate_ids_1based': [a.fixed_atoms+1,a.substrate_atoms],
        'molecular_bond_count': len(internal), 'interface_bond_count': len(bonds)-len(internal),
        'disabled_checks_by_user': ['interface_contacts','O3_projected_triangle',
                                   'non_anchor_height','molecular_topology'],
        'tetra_source_sha256':sha(a.tetra_json) if a.tetra_json else None,
        'P_local_checks_enabled':bool(tetra),'extra_PO_repulsion':bool(tetra), 'restraint_implementation_test': verification,
        'fmax_eV_A': a.fmax, 'maxstep_A': a.maxstep, 'max_steps': a.steps,
        'versions': {'torch':torch.__version__, 'ase':ase.__version__, 'mace':md.version('mace-torch'),
                     'cuequivariance':md.version('cuequivariance-torch')},
        'MD_parameters': {'dt_fs': a.dt, 'H_mass_u': a.h_mass, 'cycles': a.cycles,
                          'steps_per_cycle': a.cycle_steps, 'schedule_K': [300,500,300],
                          'friction_per_fs': .002, 'seed': a.seed, 'fixcm': False,
                          'convergence_patience':a.convergence_patience,'epsilon_eV_per_SAM':a.energy_per_sam,
                          'cycle_cap_is_review_checkpoint':bool(a.convergence_patience)},
        'gpu': torch.cuda.get_device_name(0), 'dtype':'float32', 'mode':a.mode})
    torch.set_num_threads(a.threads)
    base = MACECalculator(model_paths=str(a.model.resolve()), device='cuda', default_dtype='float32', enable_cueq=True)
    # Same no-stress force path as the successful fixed-cell ASE baseline.
    base.use_compile = True
    atoms.calc = RestrainedCalculator(base, bonds, tetra)
    if a.mode == 'benchmark':
        records, reference = [], None
        for threads in [1,4]:
            torch.set_num_threads(threads)
            times=[]
            for repeat in range(4):
                atoms.calc.reset()
                torch.cuda.synchronize()
                t=time.monotonic()
                f=atoms.get_forces()
                e=atoms.get_potential_energy()
                torch.cuda.synchronize()
                dt=time.monotonic()-t
                if repeat: times.append(dt)
            if reference is None: reference=(e,f.copy())
            error=float(np.max(np.abs(f-reference[1])))
            if error > 1e-3 or abs(e-reference[0]) > .05:
                raise AssertionError('Thread layout force/energy equivalence failed')
            records.append({'threads':threads, 'seconds':times, 'median_seconds':float(np.median(times)),
                'max_force_component_difference_eV_A':error, 'energy_difference_eV':e-reference[0],
                'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30})
            print(json.dumps(records[-1]),flush=True)
        save(a.output/'benchmark.json',{'records':records,'selected_threads':min(records,key=lambda r:r['median_seconds'])['threads'],
            'acceptance_predefined': {'max_force_difference_eV_A':.001,'energy_difference_eV':.05},
            'scope':f'{len(atoms)} atoms, same force kernel and restraints; not a universal optimum'})
        return
    if a.mode in ('anneal','md-smoke'):
        run_anneal(a, atoms, initial)
        return
    t0=time.monotonic()
    with (a.output/'optimization.log').open('w', buffering=1) as log:
        opt=LBFGS(atoms, logfile=log, maxstep=a.maxstep, trajectory=str(a.output/'optimization.traj'))
        if a.replay_trajectory:
            replay_lbfgs(opt,a.replay_trajectory)
        def progress():
            forces=atoms.get_forces()
            record={'status':'running','step':opt.nsteps,'energy_eV':atoms.get_potential_energy(),
                    'fmax_eV_A':float(np.linalg.norm(forces,axis=1).max()),
                    'elapsed_s':time.monotonic()-t0,'restraint_energy_eV':atoms.calc.restraint_energy}
            if not np.isfinite(forces).all(): raise ValueError('Non-finite force')
            save(a.output/'progress.json',record)
            if opt.nsteps%10==0: print(json.dumps(record),flush=True)
            if opt.nsteps%100==0: write(a.output/f'checkpoint-{opt.nsteps:04d}.extxyz',atoms)
        opt.attach(progress,interval=1)
        converged=opt.run(fmax=a.fmax,steps=a.steps)
    if tetra:
        gate=tetra_audit(atoms,tetra)
        save(a.output/'tetra-final-audit.json',gate)
        if not gate['all_passed']:
            write(a.output/'failed-tetra.extxyz', atoms)
            raise ValueError('Post-optimization P tetrahedral gate failed')
    write(a.output/(a.output_prefix+'.extxyz'),atoms)
    write(a.output/(a.output_prefix+'.cif'),atoms)
    write(a.output/(a.output_prefix+'.data'),atoms,format='lammps-data',atom_style='atomic',
          specorder=sorted(set(atoms.get_chemical_symbols())),masses=True)
    if not np.array_equal(initial[fixed_indices],atoms.positions[fixed_indices]):
        raise AssertionError('Frozen substrate moved')
    record=json.loads((a.output/'progress.json').read_text())
    record.update(status='completed',converged=bool(converged),steps=opt.nsteps,
        stop_reason='force_converged' if converged else 'iteration_limit',
        fixed_max_displacement_A=0.,atom_count=len(atoms),
        structure_sha256=sha(a.output/(a.output_prefix+'.extxyz')),
        interface_height_triangle_topology_checks='not_run_by_user_instruction')
    save(a.output/'result.json',record)
    save(a.output/'progress.json',record)
    print(json.dumps(record),flush=True)




def read_pvksam_structure(path):
    """Read simulation P1 CIF without symmetry expansion; otherwise use ASE."""
    path = Path(path)
    if path.suffix.lower() == '.cif':
        from ase.io.cif import parse_cif
        block = next(parse_cif(str(path)))
        if block.get_spacegroup(subtrans_included=True).no == 1:
            atoms = block.get_unsymmetrized_structure()
            atoms.set_pbc(True)
            return atoms
    return read(path)


def build_pvksam_restraints(atoms, substrate_atoms, molecules, atoms_per_molecule,
                            interface, *, metal_elements=('In', 'Sn')):
    """Seal H0 bonds/headgroups from intact geometry / 从完整H0封存约束.

    Supported: one phosphonate or carboxylate anchor per contiguous SAM block.
    Input validation here is not a post-relaxation full-topology gate.
    Interface records use 1-based metal/donor IDs and target_A; bonds use 0-based IDs.
    """
    from itertools import combinations
    from validate_monolayer import covalent_cutoff
    n, nm, size = substrate_atoms, molecules, atoms_per_molecule
    if min(n, nm, size) <= 0 or len(atoms) != n + nm * size:
        raise ValueError('Invalid H0 block layout')
    if not np.isfinite(atoms.positions).all() or np.any(atoms.numbers[:n] == 1):
        raise ValueError('H0 requires finite positions and no substrate H')
    # The current proton candidate implementation assumes an orthogonal +z frame.
    if (not np.allclose(atoms.cell[:], np.diag(atoms.cell.diagonal()), atol=1e-10)
            or np.any(atoms.cell.diagonal() <= 0)):
        raise ValueError('PVKSAM rc.1 supports positive diagonal cells, surface periodic a,b and outward +z only')
    atoms = atoms.copy(); atoms.set_pbc([True, True, False])
    sy = np.asarray(atoms.get_chemical_symbols())
    if not np.all(sy[n:].reshape(nm, size) == sy[n:n+size]):
        raise ValueError('SAM blocks must have identical element ordering')
    raw = molecular_bonds(atoms, sy, n, nm, size)
    bonds = [{'i':i-1, 'j':j-1, 'r0_A':r, 'k_eV_A2':10. if h else 20.,
              'kind':'SAM_internal'} for i,j,r,h in raw]
    graph = {i:set() for i in range(n, len(atoms))}
    for b in bonds:
        graph[b['i']].add(b['j']); graph[b['j']].add(b['i'])
    heads=[]; all_p=[]; ligands=[]; angles=[]
    for m in range(nm):
        lo=n+m*size; block=list(range(lo,lo+size))
        reached={lo}; stack=[lo]
        while stack:
            for j in graph[stack.pop()] - reached:
                reached.add(j); stack.append(j)
        if len(reached)!=size: raise ValueError(f'SAM {m+1}: disconnected intact reference')
        candidates=[]
        for i in block:
            near=sorted(j for j in block if j!=i and
                        atoms.get_distance(i,j,mic=True)<covalent_cutoff(sy[i],sy[j]))
            ns=sorted(sy[j] for j in near)
            if sy[i]=='P':
                if ns!=['C','O','O','O']:
                    raise ValueError(f'P {i+1}: expected intact 3O+1C tetrahedral connectivity')
                all_p.append(i); ligands.append(near)
                vectors=atoms.get_distances(i,near,mic=True,vector=True)
                unit=vectors/np.linalg.norm(vectors,axis=1)[:,None]
                weights=np.linalg.solve(np.vstack([unit.T,np.ones(4)]),[0,0,0,1])
                if not np.all(weights>0): raise ValueError(f'P {i+1}: reference is not tetrahedral')
                for a,b in combinations(near,2):
                    theta=np.deg2rad(atoms.get_angle(a,i,b,mic=True))
                    angles.append([a,i,b,float(theta)])
                candidates.append(('phosphonate',i,[j for j in near if sy[j]=='O'],2))
            elif sy[i]=='C' and ns==['C','O','O']:
                candidates.append(('carboxylate',i,[j for j in near if sy[j]=='O'],1))
        if len(candidates)!=1:
            raise ValueError(f'SAM {m+1}: ambiguous/unsupported anchor family ({len(candidates)} matches)')
        family,center,oxygen,protons=candidates[0]
        # Deprotonated H0, not a neutral acid block with an unreleased acidic H.
        if any(sy[j]=='H' for o in oxygen for j in graph[o]):
            raise ValueError('H0 headgroup still contains acidic H; canonicalize the submitted monomer first')
        heads.append({'family':family,'center':center,'oxygen':oxygen,'protons_per_sam':protons,
                      'local_center':center-lo,'local_oxygen':[j-lo for j in oxygen]})
    signature=lambda h:(h['family'],h['local_center'],h['local_oxygen'])
    if any(signature(h)!=signature(heads[0]) for h in heads):
        raise ValueError('SAM blocks have inconsistent anchor identities')
    interface_bonds=[]; covered=set(); pairs=set()
    for b in interface.get('bonds',[]):
        i,j=int(b['metal'])-1,int(b['donor'])-1
        r=float(b['target_A'])
        if not (0<=i<n and n<=j<len(atoms)) or sy[i] not in metal_elements or sy[j]!='O':
            raise ValueError('Interface pair must be substrate metal to SAM donor O')
        m=(j-n)//size
        if j not in heads[m]['oxygen'] or not np.isfinite(r) or r<=0 or (i,j) in pairs:
            raise ValueError('Invalid/duplicate registered interface pair')
        pairs.add((i,j)); covered.add(m)
        interface_bonds.append({'i':i,'j':j,'r0_A':r,'k_eV_A2':100.,'kind':'interface'})
    if covered!=set(range(nm)):raise ValueError('Every SAM needs at least one registered metal-O pair')
    tetra={'schema':'P-tetrahedral-angle-exclusion-v1','P_indices':all_p,'ligand_indices':ligands,
           'angles':angles,'angle_k_eV_rad2':20.,'substrate_O_indices':np.flatnonzero(sy[:n]=='O').tolist(),
           'repulsion_min_A':2.20,'repulsion_k_eV_A2':100.,'max_angle_deviation_deg':20.,
           'metal_elements':list(metal_elements)} if all_p else None
    return {'bonds':interface_bonds+bonds,'tetra':tetra,'headgroups':heads,
            'family':heads[0]['family'],'protons_per_molecule':heads[0]['protons_per_sam']}


def remap_pvksam_restraints(spec, substrate_atoms, surface_h):
    """Insert free H between substrate and SAM; preserve all reference values."""
    import copy
    out=copy.deepcopy(spec)
    shift=lambda i:i+surface_h if i>=substrate_atoms else i
    for b in out['bonds']:
        b['i'],b['j']=shift(b['i']),shift(b['j'])
    if out['tetra']:
        t=out['tetra'];t['P_indices']=[shift(i) for i in t['P_indices']]
        t['ligand_indices']=[[shift(i) for i in row] for row in t['ligand_indices']]
        t['angles']=[[shift(i),shift(p),shift(j),angle] for i,p,j,angle in t['angles']]
    for h in out['headgroups']:
        h['center']=shift(h['center']);h['oxygen']=[shift(i) for i in h['oxygen']]
    return out


if __name__ == '__main__':
    if '--restrained' in sys.argv:
        restrained_main([x for x in sys.argv[1:] if x != '--restrained'])
    else:
        legacy_main()
