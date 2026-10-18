"""Configure the copied, unregistered rc.21 publisher / 配置接口修复发布。

This is release assembly only, not scientific execution. Refactors the copied
rc.20 publisher; all executed helpers stay in this successor source tree.
"""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
reg=json.loads((ROOT/'registry.yaml').read_text())
assert '1.0.0-rc.21' not in next(x for x in reg['workflows'] if x['id']=='pvksam')['releases']
p=HERE/'publish_release.py';s=p.read_text()
s=s.replace("VERSION='1.0.0-rc.20'; BASE='1.0.0-rc.19'", "VERSION='1.0.0-rc.21'; BASE='1.0.0-rc.20'")
s=s.replace("SKILL='1.0.0-rc.18'; OLD_SKILL='1.0.0-rc.17'", "SKILL='1.0.0-rc.19'; OLD_SKILL='1.0.0-rc.18'")
s=s.replace('release-integrity-20261009-pvksam-joint-pack.yaml','release-integrity-20261009-pvksam-roll-handoff.yaml')
start=s.index('IMPORTS=');end=s.index('GROWTH=',start)
s=s[:start]+"IMPORTS=['scripts/mace_conformation_scan.py']\n"+s[end:]
start=s.index('def adapt_tests(');end=s.index('def publication_plan(',start)
replacement=r'''def prepare(root,project,evidence):
    _,entry=unregistered(root);source=root/SOURCE_REL
    rows=[]
    for rel in ['registry.yaml',COMP_DESC,'workflows/pvksam/workflow.yaml','workflows/pvksam/versions.lock']:
        p=root/rel;q=evidence/'metadata-before'/rel;write_new(q,p.read_bytes());assert sha(q)==sha(p)
        rows.append({'path':rel,'sha256':sha(p),'backup':str(q),'bytes':p.stat().st_size})
    write_new(evidence/'metadata-before/receipt.json',{'verified_metadata_backup':rows,'project_declaration_sha256':sha(project/'.codex/workflows.yaml'),'project_adoption_changed':False,'prior_workflow':BASE,'prior_skill':OLD_SKILL})
    assert sha(source/'scripts/mace_conformation_scan.py')==sha(project/'scripts/mace_conformation_scan.py')
    for name in ['scripts/orchestrate_md_pipeline.py','tests/test_pvksam.py']:
        p=source/name;p.write_text(p.read_text().replace("'1.0.0-rc.20'","'1.0.0-rc.21'"))
    note='\n## Site-roll handoff repair / 表面方位交接修复 (rc.21)\n\nFirst CPU screen accepts generated rolls only with the exact sibling pc-roll-generation.json hash receipt. Selected-roll re-screening and MACE planning require the exact single-roll selection receipt. Raw grids cannot enter MACE directly. Candidate/skeleton/phase identity is retained; legacy internal grid dihedral lists are empty for precomputed rolls. Scientific methods and thresholds are unchanged.\n'
    for name in ['SKILL.md','references/initial-conformer-search.md','references/pvksam.zh-CN.md']:
        with (source/name).open('a') as f:f.write(note)
    wf=load(root/entry['descriptor']);wf['release']=VERSION
    wf['front_chain_policy']['site_roll_handoff']='Hash-bound generated grid → CPU prescreen → MMFF94s single-roll selection → selected rescreen → MACE plan'
    wf['release_notes']={'authorization':'User-authorized next SAM execution; necessary interface repair and immutable successor','change':'Repair circular raw-roll prescreen receipt requirement and missing legacy dihedrals; enforce selected receipt at MACE plan','prior_release_preserved':BASE,'scientific_method_or_parameter_changes':False,'project_pin_migration':False,'downloads':0}
    lock=load(root/entry['lock']);lock.update(workflow_release=VERSION,release=VERSION,locked_at='2026-10-09');lock['components']['skill.pvksam']=SKILL
    (root/WF_REL/'workflow.yaml').write_bytes(data(wf));(root/WF_REL/'versions.lock').write_bytes(data(lock))
    write_new(evidence/'preparation.json',{'status':'PREPARED_NOT_REGISTERED','source':str(source),'imports':IMPORTS,'failure_evidence':str(project/'runs/pvksam-front/3Nap-Ac-20261009-01/management/site-screen-pilot.log'),'project_pin_unchanged':True})
    return {'status':'PREPARED_NOT_REGISTERED'}

def finalize(root,project,evidence):
    unregistered(root);source=root/SOURCE_REL
    for rel in IMPORTS:
        if sha(source/rel)!=sha(project/rel):raise ValueError('Project owner differs')
    commands=[(sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider',str(source/'tests')),
              (sys.executable,'-B',str(source/'scripts/mace_conformation_scan.py'),'--help')]
    rows=[];env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    for i,cmd in enumerate(commands):
        p=subprocess.run(cmd,cwd=source,env=env,capture_output=True,text=True)
        log=evidence/f'component-check-{i}.log';write_new(log,(p.stdout+p.stderr).encode())
        rows.append({'argv':list(cmd),'exit_code':p.returncode,'log':str(log),'log_sha256':sha(log)})
        if p.returncode:
            write_new(evidence/'failed-validation.json',rows);raise ValueError(f'Component check failed: {log}')
    prior=load(source/'provenance.json');prior.pop('source_files_sha256',None)
    provenance={'schema_version':1,'base_release':f'skill.pvksam@{OLD_SKILL}','date':'2026-10-09','change':'Repair first CPU prescreen/selected rescreen/plan handoff; no scientific parameter changes','prior_route_provenance':prior,'project_imports_sha256':{r:sha(project/r) for r in IMPORTS},'engineering_validation':rows,'source_files_sha256':{str(p.relative_to(source)):sha(p) for p in verify.regular_files(source) if p.name!='provenance.json'}}
    (source/'provenance.json').write_bytes(data(provenance))
    write_new(evidence/'ready.json',{'component_tests_passed':True,'cli_checks_passed':True,'case_audit_passed':True,'case_audit_scope':'Synthetic handoff regressions; real 3Nap scientific continuation pending','component_source_tree_sha256':verify.tree_sha256(source),'checks':rows})
    return {'status':'ENGINEERING_CHECKS_PASSED','test_summary':Path(rows[0]['log']).read_text().strip().splitlines()[-1]}

'''
s=s[:start]+replacement+s[end:]
s=s.replace("provenance='Byte-identical successful joint pack-then-bind project owners; N96 evidence and fresh portable full-component regressions.',growth_policy=GROWTH", "provenance='Hash-bound raw-roll screening and selected-roll MACE handoff repair; new positive and negative integration regressions.'")
s=s.replace("meta['capabilities']+=['joint-multiconformer-pack-then-bind','rigid-PO3-feasible-arc-fit-after-packing','shared-CN5-pool','accepted-epoch-geometry-cache']", "meta['capabilities']+=['hash-bound-generated-roll-prescreen','selected-receipt-required-at-MACE-plan']\n    meta['limitations']=[x for x in meta['limitations'] if not x.startswith('fixed-site-cpu accepts only')]\n    meta['limitations'].append('Generated grid is CPU-prescreen-only; MACE plan requires the selected ensemble rescreen.')")
s=s.replace('codex-pvksam-joint-pack','codex-pvksam-roll-handoff').replace('User-authorized promotion of existing successful project algorithm','Required interface repair during user-authorized next-SAM execution')
s=s.replace("str(project),'--harness','codex','--json'", "str(project),'--harness','codex','--allow-candidate','--json'")
s=s.replace("'pvksam20-transaction'", "'pvksam21-transaction'")
s=s.replace("choices=['prepare','adapt_tests','finalize','publish','accept']", "choices=['prepare','finalize','publish','accept']")
p.write_text(s)
p=HERE/'package_closure.py';s=p.read_text().replace('1.0.0-rc.20','1.0.0-rc.21').replace('rc.20','rc.21').replace('1.0.0-rc.18','1.0.0-rc.19');p.write_text(s)
print('Configured unregistered PVKSAM rc.21 / Skill rc.19 publisher')
