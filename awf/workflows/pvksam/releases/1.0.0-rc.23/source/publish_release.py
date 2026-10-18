#!/usr/bin/env python3
"""Assemble, validate and publish the joint pack-then-bind successor / 共同生长发布。

Refactors PVKSAM rc.19's publication owner; retains immutable predecessors,
verified backups, CAS metadata writes and project pin isolation. No scientific
production jobs, downloads or remote publication. Units: Angstrom, degree, eV.
"""
from __future__ import annotations
import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
sys.dont_write_bytecode = True
import verify

WF='pvksam'; VERSION='1.0.0-rc.23'; BASE='1.0.0-rc.22'
SKILL='1.0.0-rc.21'; OLD_SKILL='1.0.0-rc.20'
WF_REL=f'workflows/{WF}/releases/{VERSION}'
COMP_DESC='components/skills/pvksam/component.yaml'
SOURCE_REL=f'components/skills/pvksam/releases/{SKILL}/source'
ATTEST='attestations/release-integrity-20261010-pvksam-h0-relaxation.yaml'
IMPORTS=['scripts/relax_monolayer.py']
GROWTH={'pose_mode':'pack-then-bind','conformer_selection_mode':'joint',
        'azimuth_step_deg':30.0,'headgroup_feasible_arc_step_deg':5.0,
        'branch_rank':None,'organic_and_P_fixed_during_head_fit':True,
        'headgroup_fit_scope':'three O rigid rotation about source P-C axis',
        'candidate_energy':'source conformer relative total energy; not incremental adsorption energy',
        'failure_commit':'remove failed pose only; no site/metal/placed-state consumption',
        'default_single_metal_azimuth_source':'azimuth_step_deg',
        'scientific_scope':'finite geometry search; no global packing or energy minimum guarantee'}

def load(p): return json.loads(Path(p).read_text())
def sha(p): return verify.file_sha256(Path(p))
def data(v): return (json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode()
def write_new(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:
        f.write(v if isinstance(v,bytes) else data(v));f.flush();os.fsync(f.fileno())
def unregistered(root):
    reg=load(root/'registry.yaml');entry=next(x for x in reg['workflows'] if x['id']==WF)
    if VERSION in entry['releases'] or SKILL in load(root/COMP_DESC)['releases']:
        raise ValueError('Registered releases are immutable')
    if load(root/entry['descriptor'])['release']!=BASE:raise ValueError('Current pointer changed')
    return reg,entry

def prepare(root,project,evidence):
    _,entry=unregistered(root);source=root/SOURCE_REL
    rows=[]
    for rel in ['registry.yaml',COMP_DESC,'workflows/pvksam/workflow.yaml','workflows/pvksam/versions.lock']:
        p=root/rel;q=evidence/'metadata-before'/rel;write_new(q,p.read_bytes());assert sha(q)==sha(p)
        rows.append({'path':rel,'sha256':sha(p),'backup':str(q),'bytes':p.stat().st_size})
    write_new(evidence/'metadata-before/receipt.json',{'verified_metadata_backup':rows,'project_declaration_sha256':sha(project/'.codex/workflows.yaml'),'project_adoption_changed':False,'prior_workflow':BASE,'prior_skill':OLD_SKILL})
    assert sha(source/'scripts/relax_monolayer.py')==sha(project/'scripts/relax_monolayer.py')
    for name in ['scripts/orchestrate_md_pipeline.py','tests/test_pvksam.py']:
        p=source/name;p.write_text(p.read_text().replace(repr(BASE),repr(VERSION)))
    protocol=(source/'references/pvksam.zh-CN.md').read_text()
    if 'fmax=0.05 eV/Å' not in protocol or '最多 5000 步' not in protocol:
        raise ValueError('The requested relaxation defaults are missing from the PVKSAM protocol')
    wf=load(root/WF_REL/'workflow.yaml');wf['release']=VERSION
    wf['release_notes']={'authorization':'User explicitly requested a PVKSAM relaxation-default upgrade after the 4Nap-Ac Stage 13 iteration-limit result','change':'Raise post-adsorption restrained-relaxation defaults from 2000 steps/fmax 0.01 eV/A to 5000 steps/fmax 0.05 eV/A, preserve maxstep 0.02 A, synchronize planner and worker defaults, and add regression/evidence records','prior_release_preserved':BASE,'scientific_method_or_parameter_changes':True,'project_pin_migration':False,'downloads':0}
    lock=load(root/entry['lock']);lock.update(workflow_release=VERSION,release=VERSION,locked_at='2026-10-10');lock['components']['skill.pvksam']=SKILL
    (root/WF_REL/'workflow.yaml').write_bytes(data(wf));(root/WF_REL/'versions.lock').write_bytes(data(lock))
    write_new(evidence/'preparation.json',{'status':'PREPARED_NOT_REGISTERED','source':str(source),'imports':IMPORTS,'failure_evidence':str(project/'runs/pvksam-front/3Nap-Ac-20261009-01/management/site-screen-pilot.log'),'project_pin_unchanged':True})
    return {'status':'PREPARED_NOT_REGISTERED'}

def finalize(root,project,evidence):
    unregistered(root);source=root/SOURCE_REL
    for rel in IMPORTS:
        if sha(source/rel)!=sha(project/rel):raise ValueError('Project owner differs')
    commands=[(sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider',str(source/'tests')),
              (sys.executable,'-B',str(source/'scripts/mace_conformation_scan.py'),'--help'),
              (sys.executable,'-B',str(source/'scripts/relax_monolayer.py'),'--restrained','--help')]
    rows=[];env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    for i,cmd in enumerate(commands):
        p=subprocess.run(cmd,cwd=source,env=env,capture_output=True,text=True)
        log=evidence/f'component-check-{i}.log';write_new(log,(p.stdout+p.stderr).encode())
        rows.append({'argv':list(cmd),'exit_code':p.returncode,'log':str(log),'log_sha256':sha(log)})
        if p.returncode:
            write_new(evidence/'failed-validation.json',rows);raise ValueError(f'Component check failed: {log}')
    prior=load(source/'provenance.json');prior.pop('source_files_sha256',None)
    evidence_record=load(source/'evidence/h0-relaxation-parameter-transfer.json')
    expected_change={'steps_cap':5000,'fmax_eV_A':0.05,'maxstep_A':0.02,'scope':'PVKSAM post-adsorption restrained optimization: H0, post-protonation, and post-cycle relaxation'}
    if evidence_record.get('change')!=expected_change:
        raise ValueError('Parameter-transfer evidence does not match the requested defaults')
    provenance={'schema_version':1,'base_release':f'skill.pvksam@{OLD_SKILL}','date':'2026-10-10','change':'User-authorized post-adsorption relaxation defaults: 5000-step cap and fmax=0.05 eV/A; prior related case retained as limited evidence, no other scientific parameters changed','prior_route_provenance':prior,'project_imports_sha256':{r:sha(project/r) for r in IMPORTS},'engineering_validation':rows,'parameter_transfer_evidence':'evidence/h0-relaxation-parameter-transfer.json','parameter_transfer_evidence_sha256':sha(source/'evidence/h0-relaxation-parameter-transfer.json'),'source_files_sha256':{str(p.relative_to(source)):sha(p) for p in verify.regular_files(source) if p.name!='provenance.json'}}
    (source/'provenance.json').write_bytes(data(provenance))
    write_new(evidence/'ready.json',{'component_tests_passed':True,'cli_checks_passed':True,'case_audit_passed':True,'case_audit_scope':'Reviewed transfer record comparing the prior 2Nap-Ac post-H case with the nonconverged 4Nap-Ac H0 run; limited to one similar-size prior case','component_source_tree_sha256':verify.tree_sha256(source),'parameter_transfer_evidence_sha256':sha(source/'evidence/h0-relaxation-parameter-transfer.json'),'checks':rows})
    return {'status':'ENGINEERING_CHECKS_PASSED','test_summary':Path(rows[0]['log']).read_text().strip().splitlines()[-1]}

def publication_plan(root,project,evidence):
    registry,live=unregistered(root);backup=evidence/'metadata-before';receipt=load(backup/'receipt.json')
    for row in receipt['verified_metadata_backup']:
        if sha(row['backup'])!=row['sha256']:raise ValueError('Backup changed')
        if row['path'].startswith('workflows/pvksam/') and sha(root/row['path'])!=row['sha256']:raise ValueError('Alias changed')
    if sha(project/'.codex/workflows.yaml')!=receipt['project_declaration_sha256']:raise ValueError('Project pin changed')
    ready=load(evidence/'ready.json');source=root/SOURCE_REL
    if not all(ready.get(k) is True for k in ['component_tests_passed','cli_checks_passed','case_audit_passed']):raise ValueError('Readiness missing')
    if verify.tree_sha256(source)!=ready['component_source_tree_sha256']:raise ValueError('Validated source changed')
    for c in ready['checks']:
        if c['exit_code']!=0 or sha(c['log'])!=c['log_sha256']:raise ValueError('Test evidence changed')
    import package_closure
    package_closure.verify_frozen_manifest(root)
    prior=next(x for x in load(backup/'registry.yaml')['workflows'] if x['id']==WF)
    if live!=prior:raise ValueError('Workflow catalog changed')
    desc=load(root/COMP_DESC)
    if desc!=load(backup/COMP_DESC):raise ValueError('Component catalog changed')
    meta=copy.deepcopy(desc['releases'][OLD_SKILL])
    meta.update(status='candidate',observed_at='2026-10-10',source_snapshot=SOURCE_REL,
        tree_sha256=verify.tree_sha256(source),predecessor=f'skill.pvksam@{OLD_SKILL}',
        provenance='User-authorized post-adsorption relaxation update: 5000-step cap and fmax=0.05 eV/A, with matching planner/worker defaults, a regression, and limited prior-case evidence.')
    meta['capabilities']+=['post-adsorption-relaxation-step-cap-5000','post-adsorption-relaxation-fmax-0.05-eV-A']
    meta['limitations'].append('The 5000-step/fmax=0.05 eV/A relaxation defaults have one similar-size 2Nap-Ac/ITO precedent and are not universal cross-molecule calibration.')
    meta['limitations']=[x for x in meta['limitations'] if not x.startswith('fixed-site-cpu accepts only')]
    meta['limitations'].append('Generated grid is CPU-prescreen-only; MACE plan requires the selected ensemble rescreen.')
    meta['limitations']+=['Finite sampled greedy packing; no global maximum coverage or adsorption-energy minimum guarantee.']
    desc['releases'][SKILL]=meta
    paths=verify.tree_v2_paths(source,'tree')
    att={'schema_version':1,'digest_contract':'awf-tree-v2','observed_at':'2026-10-10',
         'owner':'codex-pvksam-h0-relaxation-defaults','provenance':'User-authorized post-adsorption relaxation parameter update with component regression and limited prior-case evidence',
         'releases':{f'skill.pvksam@{SKILL}':{'coverage':'tree','descriptor':COMP_DESC,
          'descriptor_file_sha256':hashlib.sha256(data(desc)).hexdigest(),'metadata_sha256':verify.canonical_object_sha256(meta),
          'root':SOURCE_REL,'legacy_tree_sha256':meta['tree_sha256'],'tree_sha256':verify.tree_sha256_v2(source,'tree'),
          'entry_count':len(paths),'regular_file_bytes':verify.tree_file_bytes(source),'symlink_count':sum(p.is_symlink() for p in paths)}}}
    wf=root/WF_REL/'workflow.yaml';lock=root/WF_REL/'versions.lock'
    if load(lock)['components']!={**load(root/prior['lock'])['components'],'skill.pvksam':SKILL}:raise ValueError('Dependency drift')
    planned=copy.deepcopy(registry);entry=next(x for x in planned['workflows'] if x['id']==WF)
    catalog={'descriptor':str(wf.relative_to(root)),'descriptor_sha256':sha(wf),'lock':str(lock.relative_to(root)),'lock_sha256':sha(lock),'status':'candidate'}
    entry['releases'][VERSION]=catalog;entry.update({k:v for k,v in catalog.items() if k!='status'})
    planned['integrity_attestations'].append(ATTEST);planned['integrity_policy']['required_release_attestations'].append(f'skill.pvksam@{SKILL}')
    targets={root/COMP_DESC:data(desc),root/ATTEST:data(att),root/'workflows/pvksam/workflow.yaml':wf.read_bytes(),root/'workflows/pvksam/versions.lock':lock.read_bytes(),root/'registry.yaml':data(planned)}
    if (root/ATTEST).exists():raise ValueError('Attestation exists')
    plan={'workflow':f'{WF}@{VERSION}','skill':f'skill.pvksam@{SKILL}',
          'targets':[{'path':str(p.relative_to(root)),'before_sha256':sha(p) if p.exists() else None,'after_sha256':hashlib.sha256(b).hexdigest()} for p,b in targets.items()],
          'project_pin_changed':False,'other_workflows_changed':False,'old_release_metadata_preserved':True,
          'rollback':'Preserve release history; restore only PVKSAM rolling pointer/alias from verified predecessor metadata. Interrupted transactions require live hash comparison first.'}
    return plan,targets

def publish(root,project,evidence,execute):
    plan,targets=publication_plan(root,project,evidence)
    if execute:
        before=evidence/('cas-before-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
        for row in plan['targets']:
            p=root/row['path']
            if (sha(p) if p.exists() else None)!=row['before_sha256']:raise ValueError('Concurrent mutation')
            if p.exists():
                write_new(before/row['path'],p.read_bytes())
                if sha(before/row['path'])!=row['before_sha256']:raise ValueError('CAS backup mismatch')
        write_new(before/'receipt.json',plan)
        for p,b in targets.items():
            row=next(r for r in plan['targets'] if r['path']==str(p.relative_to(root)))
            if (sha(p) if p.exists() else None)!=row['before_sha256']:raise ValueError('Immediate CAS failed')
            if p.exists():
                tmp=p.with_name(p.name+'.pvksam23-transaction');write_new(tmp,b);os.replace(tmp,p)
            else:write_new(p,b)
            if sha(p)!=row['after_sha256']:raise ValueError('Readback failed')
        plan['status']='REGISTERED_VERIFIED_TARGET_BYTES';plan['latest_verified_backup']=str(before)
    else:plan['status']='DRY_RUN'
    write_new(evidence/('publication.json' if execute else 'publication-plan.json'),plan)
    return plan

def accept(root,project,evidence):
    """Read-only release audits and fresh verified packages; no runtime restore."""
    src=root/WF_REL/'source';py=sys.executable
    commands=[
        ('release-integrity',[py,'-B',str(src/'verify.py'),'--workflow',WF,'--release',VERSION,'--repository-only','--json']),
        ('historical-pin-integrity',[py,'-B',str(src/'verify.py'),'--workflow',WF,'--release','1.0.0-rc.14','--repository-only','--json']),
        ('restore-both-plan',[py,'-B',str(src/'restore_skills.py'),'--workflow',WF,'--release',VERSION,'--harness','both','--allow-candidate','--json']),
        ('restore-hermes-plan',[py,'-B',str(src/'restore_skills.py'),'--workflow',WF,'--release',VERSION,'--harness','hermes','--allow-candidate','--json']),
        ('historical-exact-release-plan',[py,'-B',str(src/'restore_skills.py'),'--workflow',WF,'--release','1.0.0-rc.14','--harness','codex','--allow-candidate','--json']),
        ('exact-closure',[py,'-B',str(src/'package_closure.py'),'--awf-root',str(root),'--output-dir',str(evidence/'exact-closure'),'--json']),
        ('core-backup',[py,'-B',str(src/'create_core_bundle.py'),'--bundle-dir',str(evidence/'core-backup')]),
        ('core-backup-verification',[py,'-B',str(src/'verify_core_bundle.py'),str(evidence/'core-backup'),'--json'])]
    rows=[]
    for name,cmd in commands:
        print('Checking '+name,flush=True)
        p=subprocess.run(cmd,cwd=root,env={**os.environ,'AWF_ROOT':str(root),'PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True)
        log=evidence/(name+'.log');write_new(log,(p.stdout+p.stderr).encode())
        rows.append({'name':name,'argv':cmd,'exit_code':p.returncode,'log':str(log),'log_sha256':sha(log)})
        if p.returncode:
            write_new(evidence/'failed-acceptance.json',rows);raise ValueError('Acceptance failed: '+name)
    import package_closure
    package_closure.verify_frozen_manifest(root)
    if sha(project/'.codex/workflows.yaml')!=load(evidence/'metadata-before/receipt.json')['project_declaration_sha256']:
        raise ValueError('Historical project pin changed')
    result={'status':'PASS','workflow':f'{WF}@{VERSION}','skill':f'skill.pvksam@{SKILL}',
            'checks':rows,'project_pin_unchanged':True,'downloads':0,'scientific_jobs_launched':0}
    write_new(evidence/'acceptance.json',result)
    return {'status':'PASS','checks':len(rows),'receipt':str(evidence/'acceptance.json')}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--awf-root',type=Path,default=Path(os.environ.get('AWF_ROOT','/home/zc/agent_workflows')))
    p.add_argument('--project',type=Path,default=Path('/home/zc/sam-ito'))
    p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--phase',choices=['prepare','finalize','publish','accept'],required=True)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args();root=a.awf_root.resolve();verify.ROOT=root
    try:
        result=publish(root,a.project.resolve(),a.evidence.resolve(),a.execute) if a.phase=='publish' else globals()[a.phase](root,a.project.resolve(),a.evidence.resolve())
        print(json.dumps(result,ensure_ascii=False,indent=2));return 0
    except (OSError,ValueError,KeyError) as e:
        print(json.dumps({'status':'FAIL','error':str(e)},ensure_ascii=False));return 1
if __name__=='__main__':raise SystemExit(main())
