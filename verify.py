#!/usr/bin/env python3
import ast,hashlib,json,re,sys,tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parent; errors=[]
manifest=json.loads((ROOT/'MANIFEST.json').read_text(encoding='utf-8'))
for rel,meta in manifest['files'].items():
 p=ROOT/rel
 if not p.is_file(): errors.append(f'missing: {rel}'); continue
 if p.stat().st_size!=meta['bytes']: errors.append(f'size: {rel}')
 if hashlib.sha256(p.read_bytes()).hexdigest()!=meta['sha256']: errors.append(f'hash: {rel}')
known=set(manifest['files'])|{'MANIFEST.json'}
actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.parts and 'dist' not in p.parts}
for rel in sorted(actual-known): errors.append(f'unlisted: {rel}')
lock_path=ROOT/'awf/workflows/pvksam/releases/1.0.0-rc.18/versions.lock'
expected={'skill.pvksam':'1.0.0-rc.16','recipe.pvksam-ase-cueq':'1.0.0-rc.3','artifact.mace-mpa-0-medium':'1.0.0-rc.2','artifact.pvksam-reference-models':'1.0.0-rc.1'}
if lock_path.is_file():
 lock=json.loads(lock_path.read_text());
 if lock.get('workflow_release')!='1.0.0-rc.18': errors.append('wrong workflow release')
 if lock.get('components')!=expected: errors.append('wrong component lock')
else: errors.append('missing workflow lock')
source=ROOT/'awf/components/artifacts/pvksam-reference-models/releases/1.0.0-rc.1/source'
catalog_path=source/'MODEL-CATALOG.json'
if catalog_path.is_file():
 catalog=json.loads(catalog_path.read_text())
 if len(catalog.get('sam_models',[]))!=9: errors.append('wrong SAM model count')
 if catalog.get('substrates',{}).get('ids')!=['fto','ito']: errors.append('wrong substrate ids')
else: errors.append('missing model catalog')
sums=source/'SHA256SUMS'
if sums.is_file():
 for line in sums.read_text().splitlines():
  digest,rel=line.split('  ',1); p=source/rel
  if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest: errors.append(f'model checksum: {rel}')
else: errors.append('missing model checksums')
for p in ROOT.rglob('*'):
 if '.git' in p.parts: continue
 if p.is_dir() and p.name=='__pycache__': errors.append(f'cache: {p.relative_to(ROOT)}')
 if not p.is_file(): continue
 if p.suffix.lower() in {'.pyc','.model','.pt','.pth','.ckpt'}: errors.append(f'excluded file type: {p.relative_to(ROOT)}')
 if p.suffix=='.py':
  try: ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
  except SyntaxError as exc: errors.append(f'python syntax: {p.relative_to(ROOT)}: {exc}')
 if p.suffix=='.toml':
  try: tomllib.loads(p.read_text(encoding='utf-8'))
  except Exception as exc: errors.append(f'toml: {p.relative_to(ROOT)}: {exc}')
 if p.suffix.lower() in {'.json','.yaml'}:
  try: json.loads(p.read_text(encoding='utf-8'))
  except Exception as exc: errors.append(f'json: {p.relative_to(ROOT)}: {exc}')
 if p.name!='verify.py' and p.stat().st_size<5_000_000:
  try: text=p.read_text(encoding='utf-8')
  except UnicodeDecodeError: text=''
  if re.search(r'(/home/zc|/mnt/z|\\\\192\.168|yangchen-proj|qq\.com|Zc@zst)',text,re.I): errors.append(f'private path/content: {p.relative_to(ROOT)}')
model_path=ROOT/'awf/components/artifacts/mace-mpa-0-medium/releases/1.0.0-rc.2/source/model.json'
if model_path.is_file():
 model=json.loads(model_path.read_text()); identity=(model.get('file_bytes'),model.get('sha256'),model.get('path_env'),model.get('bundled'))
 if identity!=(79462305,'75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638','PVKSAM_MACE_MODEL',False): errors.append('wrong external model identity')
else: errors.append('missing external model identity')
if errors: print('FAIL'); print(chr(10).join(errors)); sys.exit(1)
print(f"PASS: {len(manifest['files'])} files; PVKSAM {manifest['workflow_release']}; ITO/FTO and 9 SAM models verified")
