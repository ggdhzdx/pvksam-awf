"""PVKSAM engineering regressions; synthetic geometry is not scientific validation."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from ase import Atoms
from ase.io import write, read
from ase.constraints import FixAtoms
from ase.calculators.calculator import Calculator, all_changes
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import relax_monolayer as r
import orchestrate_md_pipeline as op
import surface_protonation as sp
from validate_monolayer import audit_short_contacts


def fixture(family='phosphonate', count=1):
    symbols=['In','O','O']; positions=[[10,10,2],[8.5,10,3],[11.5,10,3]]
    if family=='phosphonate':
        directions=np.array([[np.sqrt(8/9)*np.cos(x),np.sqrt(8/9)*np.sin(x),-1/3]
                             for x in [0,2*np.pi/3,4*np.pi/3]])
        mol=Atoms('POOOC',positions=np.vstack([[10,10,6],np.array([10,10,6])+1.5*directions,[10,10,7.82]]))
    else:
        mol=Atoms('COOC',positions=[[10,10,6],[8.9,10,5.35],[11.1,10,5.35],[10,10,7.5]])
    atoms=Atoms(symbols,positions=positions,cell=[40,30,25],pbc=[True,True,False])
    for i in range(count):
        block=mol.copy();block.positions[:,0]+=i*10;atoms+=block
    interface={'bonds':[{'metal':1,'donor':3+i*len(mol)+2,'target_A':2.29} for i in range(count)]}
    return atoms,interface,len(mol)


@pytest.mark.parametrize('family,nh,np_count',[('phosphonate',2,1),('carboxylate',1,0)])
def test_identity_and_remapping(family,nh,np_count):
    atoms,interface,size=fixture(family,2)
    spec=r.build_pvksam_restraints(atoms,3,2,size,interface)
    assert spec['protons_per_molecule']==nh
    assert (len(spec['tetra']['P_indices']) if spec['tetra'] else 0) == np_count*2
    post=r.remap_pvksam_restraints(spec,3,nh*2)
    for before,after in zip(spec['bonds'],post['bonds']):
        for key in ['r0_A','k_eV_A2','kind']:assert before[key]==after[key]
        assert after['i']==before['i']+(nh*2 if before['i']>=3 else 0)
        assert after['j']==before['j']+(nh*2 if before['j']>=3 else 0)
        assert not 3<=after['i']<3+nh*2 and not 3<=after['j']<3+nh*2
    if spec['tetra']:
        assert len(spec['tetra']['angles'])==12
        assert r.tetra_audit(atoms,spec['tetra'])['all_passed']


def test_reject_missing_interface_and_nonorthogonal():
    a,i,size=fixture()
    with pytest.raises(ValueError,match='Every SAM'):r.build_pvksam_restraints(a,3,1,size,{'bonds':[]})
    a.cell[1,0]=1
    with pytest.raises(ValueError,match='diagonal'):r.build_pvksam_restraints(a,3,1,size,i)


def test_reject_damaged_reference():
    a,i,size=fixture();a.positions[4]=a.positions[3]+[0,0,3]
    with pytest.raises(ValueError):r.build_pvksam_restraints(a,3,1,size,i)


def test_force_gradient_and_one_sided_interface():
    a,i,size=fixture();spec=r.build_pvksam_restraints(a,3,1,size,i)
    assert r.check_restraint_implementation(a,spec['bonds'])['interface_Hookean_equivalence']
    a.positions[4,0]+=.07
    e,f=r.tetra_terms(a,spec['tetra']);h=1e-6
    for index,axis in [(3,0),(4,0),(5,2)]:
        a.positions[index,axis]+=h;ep=r.tetra_terms(a,spec['tetra'])[0]
        a.positions[index,axis]-=2*h;em=r.tetra_terms(a,spec['tetra'])[0]
        a.positions[index,axis]+=h
        assert abs(f[index,axis]+(ep-em)/(2*h))<1e-5
    bond={'i':0,'j':1,'r0_A':3.,'k_eV_A2':100.,'kind':'interface'}
    b=Atoms('InO',positions=[[0,0,0],[2,0,0]])
    assert r.restraint_terms(b,[bond])[0]==0


@pytest.mark.parametrize('family', ['phosphonate','carboxylate'])
def test_global_protonation_without_mace(tmp_path,family):
    a,i,size=fixture(family);spec=r.build_pvksam_restraints(a,3,1,size,i)
    source=tmp_path/'h0.extxyz';write(source,a)
    result=sp.protonate_restrained_h0(source,tmp_path/'protonated',substrate_atoms=3,
        molecules=1,atoms_per_molecule=size,sealed_spec=spec)
    assert result['surface_H_count']==spec['protons_per_molecule']
    assert result['surface_OH_restraints']==0 and result['physical_interface_pass'] is None
    post=read(tmp_path/'protonated/protonated-interface.extxyz');nh=result['surface_H_count']
    assert np.array_equal(post.positions[3+nh:],read(source).positions[3:])
    assert result['added_H_audit']['unique_O_parents']==nh
    # No process-wide monkey-patching of headgroup identification.
    assert sp.headgroup_indices.__module__=='surface_protonation'


def config_fixture(tmp_path):
    a,i,size=fixture();write(tmp_path/'h0.extxyz',a)
    (tmp_path/'pairs.json').write_text(json.dumps(i));(tmp_path/'model').write_text('synthetic mock model; never loaded')
    config={'input':'h0.extxyz','interface_bonds':'pairs.json','model':'model','output':'run',
            'substrate_atoms':3,'molecules':1,'atoms_per_molecule':size,'fixed_atoms':1}
    path=tmp_path/'config.json';path.write_text(json.dumps(config));return path


def test_public_plan_read_only_and_strict_keys(tmp_path):
    path=config_fixture(tmp_path);before=set(tmp_path.iterdir())
    assert op.main(['--pvksam-post',str(path),'--plan'])==0
    assert set(tmp_path.iterdir())==before
    plan=op.resolve_pvksam_post_plan(path)
    assert plan['surface_H_count']==2 and plan['config']['convergence_patience']==5
    value=json.loads(path.read_text());value['unknown']=1;path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='Unknown'):op.resolve_pvksam_post_plan(path)


def test_stage_chain_with_mock_optimizer(tmp_path,monkeypatch):
    path=config_fixture(tmp_path);plan=op.resolve_pvksam_post_plan(path);calls=[]
    def worker(argv,**kwargs):
        get=lambda key:argv[argv.index(key)+1]
        dest=Path(get('--output'));dest.mkdir();calls.append(get('--mode'))
        assert '--tetra-json' in argv
        if get('--mode')=='relax':
            write(dest/'relaxed.extxyz',read(get('--input')))
            (dest/'result.json').write_text(json.dumps({'converged':True}))
        else:
            write(dest/'best-relaxed.extxyz',read(get('--input')))
            assert get('--convergence-patience')=='5' and get('--cycles')=='20'
            (dest/'result.json').write_text(json.dumps({'search_converged':True,'stop_reason':'synthetic plateau'}))
    monkeypatch.setattr(op.subprocess,'run',worker)
    result=op.execute_pvksam_post_plan(plan)
    assert calls==['relax','relax','anneal'] and result['status']=='completed'
    assert result['physical_interface_pass'] is None
    with pytest.raises(FileExistsError):op.execute_pvksam_post_plan(plan)


def test_unconverged_stage_stops_before_protonation(tmp_path,monkeypatch):
    plan=op.resolve_pvksam_post_plan(config_fixture(tmp_path))
    def worker(argv,**kwargs):
        out=Path(argv[argv.index('--output')+1]);out.mkdir()
        (out/'result.json').write_text(json.dumps({'converged':False}))
    monkeypatch.setattr(op.subprocess,'run',worker)
    result=op.execute_pvksam_post_plan(plan)
    assert result['status']=='needs_review' and not (tmp_path/'run/02-protonate').exists()


def test_stale_input_refused_before_output(tmp_path):
    plan=op.resolve_pvksam_post_plan(config_fixture(tmp_path));(tmp_path/'model').write_text('changed')
    with pytest.raises(ValueError,match='Stale'):op.execute_pvksam_post_plan(plan)
    assert not (tmp_path/'run').exists()


def test_contact_detection_and_free_oh(tmp_path):
    a=Atoms('OHC',positions=[[5,5,5],[5,5,5.98],[8,8,8]],cell=[20]*3,pbc=True)
    p=tmp_path/'a.extxyz';write(p,a)
    settings=SimpleNamespace(substrate_atoms=1,surface_h=1,molecules=1,mol_size=1,
        hh=1.2,h_heavy=1.4,heavy=1.8,parent_max=1.25)
    assert audit_short_contacts(p,settings,set())['count']==0
    a.positions[2]=[5,5,6.2];write(p,a)
    assert audit_short_contacts(p,settings,set())['count']>0


class Harmonic(Calculator):
    implemented_properties=['energy','forces'];tetra=None;bonds=[]
    def calculate(self,atoms=None,properties=('energy',),system_changes=all_changes):
        super().calculate(atoms,properties,system_changes)
        d=atoms.positions-self.origin;self.results={'energy':float((d*d).sum()/2),'forces':-d}


@pytest.mark.parametrize('patience,cycles,steps,status,reason',[
    (0,2,0,'completed','cycle_limit'),
    (5,8,1000,'completed','best_energy_plateau_and_force_convergence'),
    (5,2,1000,'needs_review','cycle_limit'),
    (5,8,0,'needs_review','local_force_not_converged')])
def test_cycle_checks(tmp_path,monkeypatch,patience,cycles,steps,status,reason):
    assert not r.search_plateau([0,-.1,-.2,-.3,-.4,-.5],5,.005,82)
    assert not r.search_plateau([0,0,0,0,0,-.41],5,.005,82)
    a=Atoms('OH',positions=[[0,0,0],[0,0,1]],cell=[10]*3);a.set_constraint(FixAtoms(indices=[0]))
    a.calc=Harmonic();a.calc.origin=a.positions.copy();checks=[]
    def contacts(atoms,args):checks.append(1);return {'count':0,'scope':'synthetic test stub'}
    monkeypatch.setattr(r,'anneal_contacts',contacts)
    args=SimpleNamespace(output=tmp_path,h_mass=4.,seed=1,mode='anneal',cycles=cycles,
        cycle_steps=6,dt=2.,fixed_atoms=1,maxstep=.02,fmax=.01,steps=steps,
        convergence_patience=patience,energy_per_sam=.005,molecules=82)
    r.run_anneal(args,a,a.positions.copy());result=json.loads((tmp_path/'result.json').read_text())
    assert result['status']==status and result['stop_reason']==reason
    if patience:assert len(checks)==len(result['cycles'])+1
    if result['search_converged']:assert len(result['cycles'])==5
    assert read(tmp_path/'best-relaxed.extxyz').get_masses()[1]==pytest.approx(1.008)


def test_external_catalog_cli_forwarding(tmp_path,monkeypatch):
    seen={}
    def resolve(sam,substrate,**kwargs):
        seen.update(kwargs);return {'planning_only':True}
    monkeypatch.setattr(op,'resolve_submission_plan',resolve)
    catalog=tmp_path/'external-catalog'
    assert op.main(['--sam','monomer.xyz','--substrate','ITO','--substrate-catalog',str(catalog),'--plan'])==0
    assert seen['catalog_directory']==catalog


def test_external_project_data_paths(tmp_path,monkeypatch):
    import substrate_library as sl
    monkeypatch.setenv('SAMFLOW_SUBSTRATE_CATALOG',str(tmp_path/'catalog'))
    monkeypatch.setenv('SAMFLOW_BENCHMARK_LEDGER',str(tmp_path/'bench.json'))
    assert sl._default_catalog_directory()==tmp_path/'catalog'
    assert sl._default_benchmark_ledger_path()==tmp_path/'bench.json'


def test_coordination_filter_is_enabled_by_default():
    from monolayer_sequential_growth import _build_parser, build_metal_coordination_filter_contract
    parser=_build_parser()
    assert parser.get_default('metal_coordination_filter')=='exclude-six-coordinated'
    assert parser.get_default('metal_coordination_cutoff_A')==2.7
    assert parser.get_default('metal_coordination_max_allowed')==5
    assert build_metal_coordination_filter_contract()['mode']=='exclude-six-coordinated'
    # Explicit legacy comparisons remain possible, never PVKSAM's implicit default.
    assert build_metal_coordination_filter_contract(mode='off')['mode']=='off'


@pytest.mark.parametrize('cn',[5,6,7])
def test_adsorption_metal_eligibility_counts_only_substrate_oxygen(cn):
    from validate_monolayer import audit_pvksam_adsorption_metals
    positions=[[10,10,10]]+[[10+2*np.cos(t),10+2*np.sin(t),10]
                           for t in np.linspace(0,2*np.pi,cn,endpoint=False)]
    # Additional close SAM O must not increase pre-adsorption substrate CN.
    a=Atoms(['In']+['O']*cn+['O'],positions=positions+[[10,10,11.5]],cell=[30]*3)
    result=audit_pvksam_adsorption_metals(a,1+cn,[{'metal':1,'donor':cn+2}])
    assert result['passed']==(cn<=5)
    assert result['mapped_metal_records'][0]['coordination_number']==cn
    assert len(a)==cn+2  # No slab atoms are deleted by excluding adsorption sites.


def test_public_post_entry_rejects_six_coordinated_metal(tmp_path):
    path=config_fixture(tmp_path)
    a=read(tmp_path/'h0.extxyz')
    extra=Atoms('OOOO',positions=[[10,12,2],[10,8,2],[10,10,0],[10,10,4]])
    a=a[:3]+extra+a[3:];write(tmp_path/'h0.extxyz',a)
    value=json.loads(path.read_text());value['substrate_atoms']=7;path.write_text(json.dumps(value))
    (tmp_path/'pairs.json').write_text(json.dumps({'bonds':[{'metal':1,'donor':9,'target_A':2.29}]}))
    with pytest.raises(ValueError,match='CN>=6'):
        op.resolve_pvksam_post_plan(path)
    assert not (tmp_path/'run').exists()


def test_random_site_policy_removed_and_frontier_cli_defaults():
    import monolayer_sequential_growth as growth
    parser=growth._build_parser()
    actions={action.dest:action for action in parser._actions}
    assert actions['selection_policy'].default == 'cohesive-frontier'
    assert actions['cohesive_frontier_mode'].default == 'continuous-snap'
    assert 'uniform' not in actions['selection_policy'].choices
    assert 'discrete' in actions['cohesive_frontier_mode'].choices
    with pytest.raises(ValueError,match='Unknown selection policy'):
        growth.run_sequential_domains(domains={},site_conflicts={},
            conformer_weights={},cell=np.diag([20.,20.,20.]),seed=1,
            selection_policy='uniform')


def test_complete_n82_growth_defaults_and_explicit_overrides():
    import monolayer_sequential_growth as g
    parser=g._build_parser()
    argv=['--substrate','sub.extxyz','--layer-groups','layers.json',
        '--site-instances','sites.json','--site-prototypes','protos.json',
        '--conformer-glob','*.extxyz','--conformer-analysis','analysis.json',
        '--output-dir','new-run','--molecule-formula-json','{"C":1,"P":1,"O":3}']
    args=parser.parse_args(argv)
    assert args.single_metal_sites == 'dynamic-cn5-uncovered'
    assert args.substrate_collision == 'height'
    assert args.headgroup_rmsd_max_A == .50
    assert (args.hh_min,args.h_heavy_min,args.heavy_heavy_min)==(1.5,1.8,2.2)
    assert args.collision_cache == 'lazy' and args.write_every_step
    tiers=g.resolve_cohesive_cli_tolerances(args)
    for key,value in g.PVKSAM_GROWTH_DEFAULTS.items():
        actual=getattr(args,key)
        if key.startswith('cohesive_') and key.endswith('_relative_tolerance'):
            actual=tiers[key.split('_')[1]]
        assert actual==value,key
    # Alternative/repair callers can explicitly disable grow-only behavior.
    other=parser.parse_args(argv+['--single-metal-sites','off',
        '--substrate-collision','check','--no-write-every-step'])
    assert other.single_metal_sites=='off' and not other.write_every_step
    assert other.substrate_collision=='check'


@pytest.fixture
def front_case(tmp_path, monkeypatch):
    import substrate_library as lib
    import mace_conformation_scan as scan
    catalog=tmp_path/'catalog';catalog.mkdir()
    source=catalog/'source.xyz';source.write_text('source bytes')
    sites=catalog/'sites.json';sites.write_text('{}')
    sam=tmp_path/'sam.extxyz'
    write(sam,Atoms('POOOCHH',positions=[[0,0,0],[1,0,0],[0,1,0],[0,0,1],[-1,-1,-1],[2,0,0],[0,2,0]]))
    model=tmp_path/'model';model.write_text('synthetic model identity only')
    cfg={'candidate_source':'phosphonate-skeleton-grid','torsion_atom_ids':[[2,1,5,3]],
        'sam':str(sam),'substrate_key':'test','catalog':str(catalog),
        'growth_substrate':str(source),'layer_groups':str(sites),
        'site_instances':str(sites),'site_prototypes':str(sites),'model':str(model),
        'output':str(tmp_path/'run'),'site_prototype_id':'site-1','site_cell':1}
    config=tmp_path/'front.json';config.write_text(json.dumps(cfg))
    record={'atom_count':7,'anchor':{'element':'P','donor_labels':['O1','O2','O3'],
        'donor_atom_ids_1based':[2,3,4]},'released_hydrogen_atom_ids_1based':[6,7]}
    monkeypatch.setattr(lib,'_resolve_submission_recipe_and_sam',lambda *a,**k:({'source_path':source},record))
    monkeypatch.setattr(lib,'_load_accepted_site_prototype_library',lambda *a:({}, {'site_prototypes':[]}, sites))
    monkeypatch.setattr(op,'_pvksam_runtime_identity',lambda p:{'runtime':'synthetic'})
    calls=[]
    def skeleton(a):
        calls.append('skeleton')
        assert a.torsion_atom_ids == [[2,1,5,3]]
        assert a.scan_step_deg == 30 and a.full_grid_batch_size == 256
        assert len(read(a.sam)) == 5
        directory = a.output_root / 'synthetic'
        directory.mkdir(parents=True)
        h0 = directory / 'sam-h0.extxyz'
        write(h0, read(a.sam))
        outcomes = directory / 'candidate-outcomes.jsonl'
        outcomes.write_text('{}\n')
        payload = {'grid_complete': True,
                   'summary': {'screen_safe_count': 1, 'optimization_started_count': 0},
                   'candidate_outcomes': str(outcomes),
                   'system': {'sam_source': {'path': str(h0)}}}
        (directory / 'manifest.json').write_text(json.dumps(payload))
        return 0
    monkeypatch.setattr(scan,'run_phosphonate_skeleton_cpu_screen',skeleton)
    return config, calls


def test_front_uses_only_systematic_skeleton_scan_and_seals_handoff(front_case):
    config,calls=front_case;plan=op.resolve_pvksam_front_plan(config)
    assert not Path(plan['config']['output']).exists()
    prepared=op.execute_pvksam_front_plan(plan,through='cpu')
    assert prepared['status']=='prepared'
    assert prepared['current_stage']=='awaiting_fast_skeleton_optimization'
    assert prepared['grid_complete'] and calls==['skeleton']
    op.execute_pvksam_front_plan(plan,resume=True)
    assert calls==['skeleton']


def test_front_chain_rejects_changed_input_before_output(front_case):
    config,calls=front_case;plan=op.resolve_pvksam_front_plan(config)
    Path(plan['config']['model']).write_text('changed')
    with pytest.raises(ValueError,match='locked file changed'):
        op.execute_pvksam_front_plan(plan)
    assert calls==[] and not Path(plan['config']['output']).exists()


def test_front_chain_rejects_changed_parameters(front_case):
    config,calls=front_case;plan=op.resolve_pvksam_front_plan(config)
    plan['config']['skeleton_scan']['step_deg']+=1
    with pytest.raises(ValueError,match='content changed'):
        op.execute_pvksam_front_plan(plan)
    # The plan must not share mutable defaults with the owner.


def test_front_chain_rejects_changed_stage_artifact(front_case):
    config,calls=front_case;plan=op.resolve_pvksam_front_plan(config)
    op.execute_pvksam_front_plan(plan,through='cpu')
    (Path(plan['config']['output'])/'sam-h0.extxyz').write_text('changed')
    with pytest.raises(ValueError,match='locked file changed'):
        op.execute_pvksam_front_plan(plan,resume=True)
    assert calls==['skeleton']


def test_front_runtime_change_rejected(front_case,monkeypatch):
    config,calls=front_case;plan=op.resolve_pvksam_front_plan(config)
    monkeypatch.setattr(op,'_pvksam_runtime_identity',lambda p:{'runtime':'changed'})
    with pytest.raises(ValueError,match='runtime changed'):
        op.execute_pvksam_front_plan(plan)
    assert calls==[]


def test_front_skeleton_grid_defaults_and_cpu_only_handoff(front_case, monkeypatch):
    import mace_conformation_scan as scan
    config, calls = front_case
    cfg = json.loads(config.read_text())
    plan = op.resolve_pvksam_front_plan(config)
    assert plan['stages'] == ['canonicalize_h0', 'skeleton_coarse_cpu']
    assert plan['config']['candidate_source'] == 'phosphonate-skeleton-grid'
    assert plan['h0_torsions'] == [[2, 1, 5, 3]]
    assert plan['config']['skeleton_scan'] == {
        'step_deg': 30.0, 'batch_size': 256, 'max_candidates': 100000}
    out = Path(plan['config']['output'])
    with pytest.raises(ValueError, match='front-through cpu'):
        op.execute_pvksam_front_plan(plan, through='mace')
    assert not out.exists() and calls == []

    def skeleton(a):
        calls.append('skeleton')
        assert a.torsion_atom_ids == [[2, 1, 5, 3]]
        assert a.scan_step_deg == 30 and a.full_grid_batch_size == 256
        assert len(read(a.sam)) == 5
        directory = a.output_root / 'synthetic'
        directory.mkdir(parents=True)
        write(directory / 'sam-h0.extxyz', read(a.sam))
        outcomes = directory / 'candidate-outcomes.jsonl'
        outcomes.write_text('{}\n')
        payload = {'grid_complete': True,
                   'summary': {'screen_safe_count': 0, 'optimization_started_count': 0},
                   'candidate_outcomes': str(outcomes),
                   'system': {'sam_source': {'path': str(directory / 'sam-h0.extxyz')}}}
        (directory / 'manifest.json').write_text(json.dumps(payload))
        return 0

    monkeypatch.setattr(scan, 'run_phosphonate_skeleton_cpu_screen', skeleton)
    prepared = op.execute_pvksam_front_plan(plan, through='cpu')
    assert prepared['current_stage'] == 'awaiting_fast_skeleton_optimization'
    assert prepared['grid_complete'] and calls == ['skeleton']
    op.execute_pvksam_front_plan(plan, through='cpu', resume=True)
    assert calls == ['skeleton']
    outcomes = out / '01-skeleton-scan/synthetic/candidate-outcomes.jsonl'
    outcomes.write_text('changed')
    with pytest.raises(ValueError, match='locked file changed'):
        op.execute_pvksam_front_plan(plan, through='cpu', resume=True)


@pytest.mark.parametrize('extra', [
    {'scan_angles_deg': [0]}, {'z_height_prefilter': {'enabled': True}},
    {'skeleton_scan': {'step_deg': 31}}, {'skeleton_scan': {'batch_size': 0}},
    {'skeleton_scan': {'limit': True}}, {'skeleton_scan': {'unexpected': 1}},
])
def test_front_skeleton_grid_rejects_ambiguous_or_invalid_settings(front_case, extra):
    config, calls = front_case
    cfg = json.loads(config.read_text())
    cfg.update(candidate_source='phosphonate-skeleton-grid',
               torsion_atom_ids=[[2, 1, 5, 3]], **extra)
    config.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        op.resolve_pvksam_front_plan(config)
    assert calls == [] and not Path(cfg['output']).exists()


def test_front_worker_failure_stops_before_analysis(front_case,monkeypatch):
    config,calls=front_case;plan=op.resolve_pvksam_front_plan(config)
    def fail(*a,**k):raise RuntimeError('synthetic skeleton scan failure')
    monkeypatch.setattr('mace_conformation_scan.run_phosphonate_skeleton_cpu_screen',fail)
    with pytest.raises(RuntimeError,match='synthetic skeleton'):
        op.execute_pvksam_front_plan(plan,through='cpu')
    state=json.loads((Path(plan['config']['output'])/'front-manifest.json').read_text())
    assert state['status']=='failed' and calls==[]
    assert 'skeleton_coarse_cpu' not in state['stages']


def test_front_config_no_shared_mutable_defaults(front_case):
    config,_=front_case;plan=op.resolve_pvksam_front_plan(config)
    plan['config']['skeleton_scan']['step_deg'] = 15
    fresh=op.resolve_pvksam_front_plan(config)
    assert fresh['config']['skeleton_scan']['step_deg'] == 30


@pytest.mark.parametrize('route',['confsearch-etkdgv3','legacy-torsion-grid'])
def test_front_rejects_removed_initial_conformer_routes(front_case,route):
    config,calls=front_case
    cfg=json.loads(config.read_text());cfg['candidate_source']=route
    config.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='Only phosphonate-skeleton-grid'):
        op.resolve_pvksam_front_plan(config)
    assert calls==[] and not Path(cfg['output']).exists()
