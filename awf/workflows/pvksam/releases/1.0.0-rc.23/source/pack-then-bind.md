# Joint pack-then-bind / 多构象共同生长

PVKSAM rc.20 uses organic packing followed by deferred rigid PO3 binding for
reviewed same-topology phosphonate sources. This extends the growth stage only.
The preceding source preparation and following nonrecessed-ITO protonation,
restraints, force thresholds and annealing defaults remain those of rc.19.

## Physical meaning and scope / 原理与适用范围

At each step, all selected source conformers compete in the same surface layer.
The candidate is a source × registered site × azimuth combination. Translate P
to the site's target and rotate the whole molecule about the surface normal
through P. This preserves the source internal shape and P-C tilt. Pending PO3
coordinates do not veto organic packing before the head has been fitted.
Keep previously accepted molecules fixed during construction.

The default azimuth grid is 30 degrees (12 directions). It controls BOTH the
registered-site pool and dynamic CN5 pool in this mode. A 15-degree override
doubles directions but requires a new runtime estimate; it does not modify the
5-degree head-fit grid or the earlier internal-torsion/site-roll grids.

Rank by the existing cohesive-frontier + continuous-snap policy: prefer current
frontier contact, minimize periodic projected-union perimeter increment, then
single-candidate projected area, maximize distinct contacted neighbors and side
contact pairs, then minimize source conformer relative energy, then seeded tie
break. Preserve existing tolerance bands. With no frontier, select a new nucleus
by projected area, source energy and seeded tie break. Source energy is the
previous source calculation, not an incremental adsorption energy for each pose.

After selection, fix P and all organic atoms. Rotate only the three O as a rigid
body about P-C. Solve feasible head-fit arcs for donor permutations, then sample
5 degrees plus analytic optima/endpoints within RMSD <=0.50 Angstrom. Require
actual mapped metal-O distances in 1.7–2.8 Angstrom. Check O-organic clearance
(O-H 1.4, O-heavy 1.8 Angstrom), all SAM-SAM and unmapped SAM-substrate hard
contacts (H-H 1.5, H-heavy 1.8, heavy-heavy 2.2 Angstrom). Source tetrahedral
shape and chirality are preserved, not replaced by an ideal tetrahedron.

A rejected fit removes only that pose, without consuming its site, metals or
accepted projection. Analytically impossible head-fit templates can share their
proof across translation-equivalent sites; environment-specific collisions may
not. Only accepted molecules consume shared state. Tridentate then bidentate
stages are followed by dynamic uncovered-CN5 filling, using all selected sources
and each source's P height. Per-source generation/pruning before pool merge is
an equivalent memory optimization, not seven independent growth trajectories.

The organic height prefilter includes H, excludes P/O3, and requires >=1.0
Angstrom above the substrate's maximum normal height. Initial metal coordination
counts substrate O only, cutoff 2.7 Angstrom, excludes CN>=6. These construction
rules are distinct from the post-H0 relaxation checks. Preserve periodic axes,
actual donor mappings, no metal reuse, and the declared formula/atom ordering.

Caches reuse geometry only while the accepted state is unchanged; acceptance
invalidates the epoch. No energy calculator, new tilt search, internal torsion
relaxation or MD is performed here. The algorithm is finite and greedy; it does
not establish globally maximum coverage or minimum adsorption energy. The N96
case validates one seven-source 2Nap-Ac/ITO construction, not new chemical families.

## Invocation / 调用

Use the locked Skill source `scripts/monolayer_sequential_growth.py`. The generic
CLI retains historical `registered-headgroup` as its default pose mode for compatibility;
AWF growth MUST explicitly select the following mode. Do not specify
`--branch-rank`: that option is solely a single-source diagnostic. Independent
layer prefixes cannot initialize a joint result. `--max-conformers` must cover
all task-selected sources; 7 and the formula below are case data, not universals.

```bash
python "$PVKSAM_SKILL_SOURCE/scripts/monolayer_sequential_growth.py" \
  --root "$PROJECT" --operation grow \
  --substrate "$SUBSTRATE" --layer-groups "$LAYER_GROUPS" \
  --site-instances "$SITE_INSTANCES" --site-prototypes "$SITE_PROTOTYPES" \
  --conformer-glob "$CONFORMER_GLOB" --conformer-analysis "$ANALYSIS_REPORT" \
  --molecule-formula-json '{"C":39,"H":34,"N":1,"O":3,"P":1}' \
  --max-conformers 7 --pose-mode pack-then-bind --azimuth-step-deg 30 \
  --single-metal-sites dynamic-cn5-uncovered --output-dir "$FRESH_OUTPUT"
```

Read `--help` and seal the full effective arguments and all input hashes. Verify
source identity against the analysis receipt and topology compatibility across
sources. Record every accepted source rank/hash, phi/psi, head fit, actual bond
mapping and rejection. Audit final source-specific geometry, periodic contacts,
atom counts, old-coordinate prefixes and actual achieved count. Query the
program-run guide before a new formal job. Validated CPU backend: one shared
process, OMP/OPENBLAS/MKL native threads=1 on node02; no MPI/GPU growth backend.

## Evidence and known failure / 证据与负向案例

`evidence/joint-pack-then-bind-case.json` records the original case and hashes:
392 sites × 7 sources × 12 angles = 32928 initial poses; N96, all tridentate;
source counts 42/0/12/26/12/2/2. Final CN5 generation planned 806400 poses,
196 survived pruning, all actually tried, none accepted. Full run wall time was
8433.47 seconds. No complete per-module profile or universal speedup is claimed.
The historical local selector profile found periodic-perimeter work expensive;
it is not an end-to-end timing breakdown.

The original seven-independent-layer orchestration was wrong for joint growth.
Tests preserve shared candidate competition, failed-fit source alternatives,
shared metal/collision consumption, per-source CN5 heights, source-specific
shape checks and rejection of incompatible topology or independent prefixes.
The packaged tests include real geometry synthetic fixtures and legacy selector
regressions. Historical scientific input/output hashes refer to project data;
large trajectories and source datasets are intentionally external.
