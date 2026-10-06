#!/usr/bin/env python3
import ast, hashlib, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
manifest=json.loads((ROOT/'MANIFEST.json').read_text(encoding='utf-8')); errors=[]
for rel,meta in manifest['files'].items():
 p=ROOT/rel
 if not p.is_file(): errors.append(f'missing: {rel}'); continue
 if p.stat().st_size!=meta['bytes']: errors.append(f'size: {rel}')
 if hashlib.sha256(p.read_bytes()).hexdigest()!=meta['sha256']: errors.append(f'hash: {rel}')
known=set(manifest['files'])|{'MANIFEST.json'}
actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.parts and 'dist' not in p.parts}
for rel in sorted(actual-known): errors.append(f'unlisted: {rel}')
lock_path=ROOT/'awf/workflows/pvksam/releases/1.0.0-rc.17/versions.lock'
if lock_path.is_file():
 lock=json.loads(lock_path.read_text())
 if lock.get('workflow_release')!='1.0.0-rc.17': errors.append('wrong workflow release')
 expected={'skill.pvksam':'1.0.0-rc.15','recipe.pvksam-ase-cueq':'1.0.0-rc.2','artifact.mace-mpa-0-medium':'1.0.0-rc.2'}
 if lock.get('components')!=expected: errors.append('wrong component lock')
else: errors.append('missing workflow lock')
for p in ROOT.rglob('*'):
 if p.is_dir() and p.name=='__pycache__': errors.append(f'cache: {p.relative_to(ROOT)}')
 if p.is_file() and p.suffix.lower() in {'.pyc','.model','.pt','.pth','.ckpt'}: errors.append(f'excluded file type: {p.relative_to(ROOT)}')
 if p.is_file() and p.suffix=='.py':
  try: ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
  except SyntaxError as exc: errors.append(f'python syntax: {p.relative_to(ROOT)}: {exc}')
model_path=ROOT/'awf/components/artifacts/mace-mpa-0-medium/releases/1.0.0-rc.2/source/model.json'
if model_path.is_file():
 model=json.loads(model_path.read_text())
 identity=(model.get('file_bytes'),model.get('sha256'),model.get('path_env'),model.get('bundled'))
 if identity!=(79462305,'75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638','PVKSAM_MACE_MODEL',False): errors.append('wrong external model identity')
else: errors.append('missing external model identity')
if errors:
 print('FAIL'); print(chr(10).join(errors)); sys.exit(1)
print(f"PASS: {len(manifest['files'])} files; PVKSAM {manifest['workflow_release']}; external model identity verified")
