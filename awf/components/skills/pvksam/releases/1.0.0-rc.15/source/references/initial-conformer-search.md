# PVKSAM's single initial-conformer and placement route

PVKSAM currently accepts one initial-conformer route for an original SAM
molecule: a deterministic 30-degree internal phosphonate skeleton grid,
followed by fixed-head MMFF94s optimization and symmetry-aware deduplication.
The P-C axial roll is omitted from the isolated-skeleton grid and handled later
at the selected ITO site. Other initial-conformer sources have been removed
from the PVKSAM interface; the front planner rejects them.

Keep separate the isolated skeleton shape count, safe site-roll count, MACE
surface minima and grown SAM count. MMFF94s gas-phase energy is neither an
adsorption energy nor a substitute for MACE surface relaxation.

## 1. Reviewed H0 and isolated skeleton grid

Start from an explicit-H source SDF with reviewed bond orders, stereochemistry,
charge and original atom IDs. Remove only the declared acidic protons and keep
the full source-to-H0 atom/proton map. The grid implementation currently derives
rotatable adjacency from the reviewed unoptimized H0 geometry, so compare its
adjacency with the authoritative source SDF before each new molecule. Do not
guess graph data from optimized coordinates.

Rigidly transform the full molecule with a proper rotation so the three
phosphonate anchor O atoms lie in the `z=0` plane and P is on positive z. Do not
move the O atoms independently or flatten the headgroup. Fix P and all three
anchor O atoms for the isolated force-field stage. Review every four-atom
internal torsion using that molecule's original 1-based IDs. Sample the unique
periodic angles `[-180, -150, ..., 150]` degrees. Omit the P-C central bond from
the grid because its axial phase is searched at the ITO site.

For `k` reviewed internal torsions the complete grid contains `12^k` points.
Stream bounded batches and apply gates before optimizing any point:

1. Check every nonbonded intramolecular pair (exclude only direct bonds):
   H-H `>=1.2 A`, H-heavy `>=1.4 A`, heavy-heavy `>=1.8 A`.
2. Solve the full continuous P-C phase intervals for all carbon-side atoms,
   including H, to satisfy `z>=0` and clear the fixed PO3 head at the same
   phase. A few sampled angles failing is not proof that the whole phase circle
   is impossible.
3. Reject unsafe grid points immediately; they do not enter MMFF94s
   optimization. These early checks are molecular-internal, not ITO surface
   checks.

The cited 2Nap-Ac skeleton used four internal torsions, hence `12^4=20,736`
grid points. Its initial screen retained 3,655 points. Those counts are a case
record, not a general fixed expectation.

## 2. Fixed-head MMFF94s optimization and deduplication

Optimize only early-screen survivors with RDKit's exact `MMFF94s` variant. Fix
P and the three anchor O coordinates; let all other atoms move. The cited run
checked the maximum movable-atom force against `0.03 eV/A` every 250 requested
iterations, with a 5,000-iteration request ceiling. These are recorded case
parameters, not a cross-system calibration. Preserve failures and nonconverged
outputs, but do not allow them into the usable set.

After optimization, solve the common continuous P-C phase again. Rotate the
full carbon-side fragment to the midpoint of the widest feasible connected
circular interval, then independently check written atom heights, all
headgroup-organic distances and fixed-head coordinates. Do not reject a
candidate solely because its unrotated FF endpoint had negative z. Do not
re-optimize the phase-rotated coordinates or call them separately converged.

Deduplicate the safe set using organic heavy-atom shape, excluding P and the
three anchor O. For comparison only, align the P-to-first-carbon axis with a
proper rotation and quotient the common P-C axial phase; do not perform an
unrestricted Kabsch fit. Enumerate equivalent atom mappings from the exact H0
heavy-atom graph and minimize the per-atom three-dimensional RMSD across those
mappings. Process by the original MMFF94s energy, then grid index and safe rank.
Assign a candidate to its nearest existing representative if its RMSD is at
most `1.0 A`; otherwise create a representative. Keep all safe family members
and original coordinates. The threshold is a shape-family convention, not
proof of a shared energy basin or adsorption pose.

In the cited run, 2,765 safe members passed the post-optimization common-phase
checks and yielded 121 symmetry-aware representatives at the 1.0-A threshold.

## 3. Registered ITO-site P-C roll screening

For each skeleton representative, preserve the selected anchor-O/metal donor
mapping and rigidly fit P plus the three donor O atoms to the registered site.
Reject sites whose mapped metal has six or more pre-adsorption substrate-O
neighbors using the PVKSAM default cutoff `2.7 A` (maximum allowed CN 5).
Rotate only the carbon-side fragment around P-C, at `0, 30, ..., 330` degrees.
For every pose, check:

1. PO3 fit RMSD `<=0.50 A` (reuse gate, not yet cross-system calibrated).
2. Every non-anchor SAM atom, including H, at or above the mean height of the
   three mapped anchor O atoms (`1e-8 A` numerical tolerance).
3. Intramolecular minimum distances listed above.
4. Periodic SAM-ITO and SAM-surface-H collisions using the recorded vdW scale
   and exact site package. The cited run used `0.85`.

For each skeleton with safe poses, calculate MMFF94s single-point energies on
the same H0 topology and retain exactly one minimum-energy safe roll for MACE.
The coordinates are not optimized during this comparison. Re-screen the exact
selected candidate IDs and coordinates before creating the MACE plan. If a
skeleton has no safe phase, preserve that failure and do not weaken the gates.

The selector `scripts/select_single_pc_roll.py` takes the candidate ensemble,
the one-to-one CPU screen manifest and the reviewed source SDF. It validates
the `<skeleton-id>__pc-roll-<angle>deg` IDs, requires a matching audit row for
every pose, and writes the one selected pose per family plus an energy receipt.

## 4. Command handoffs

Replace all paths, torsion IDs, site IDs and assignments with values reviewed
for the current SAM and maintained substrate package. Never reuse another
molecule's atom numbering.

```bash
# Generate the isolated H0 skeleton grid and online geometry screen.
.venv/bin/python scripts/mace_conformation_scan.py \
  --mode phosphonate-skeleton-cpu \
  --sam /absolute/path/reviewed-h0.extxyz \
  --torsion-atom-ids '[[a,b,c,d],[e,f,g,h],[i,j,k,l],[m,n,o,p]]' \
  --scan-step-deg 30 \
  --output-root /absolute/path/new-skeleton-scan

# Optimize only safe members with P and the three anchor O fixed.
.venv/bin/python scripts/fast_ff_prefilter.py \
  --cpu-manifest /absolute/path/skeleton-grid/manifest.json \
  --template-sdf /absolute/path/reviewed-explicit-h-source.sdf \
  --optimization-only --force-field mmff94s \
  --movable-fmax-eV-A 0.03 --max-iterations 5000 \
  --workers 16 --chunksize 1 \
  --output-root /absolute/path/new-mmff-run

# Restore common P-C phase, audit geometry and symmetry-deduplicate.
.venv/bin/python scripts/fast_ff_prefilter.py \
  --skeleton-postprocess-manifest /absolute/path/mmff/fast-ff-manifest.json \
  --topology-python /home/software/anaconda3/envs/rdkit/bin/python \
  --rmsd-tolerance-A 1.0 --skeleton-dedup-method energy-radius-symmetry \
  --output-root /absolute/path/new-skeleton-postprocess

# Generate site rolls and run the site-bound periodic CPU screen.
.venv/bin/python scripts/generate_pc_rolls.py \
  --skeleton-postprocess-manifest /absolute/path/postprocess/skeleton-postprocess-manifest.json \
  --step-deg 30 --output-dir /absolute/path/new-pc-rolls
.venv/bin/python scripts/mace_conformation_scan.py \
  --mode fixed-site-cpu --sam /absolute/path/reviewed-h0.extxyz \
  --candidate-ensemble /absolute/path/new-pc-rolls/pc-roll-candidate-ensemble.json \
  --substrate <registered-substrate-key> --substrate-catalog /absolute/path/catalog \
  --site-prototype-id <registered-site-prototype> --site-cell <cell-index> \
  --donor-assignment '<registered-donor-map>' \
  --organic-z-floor-at-anchor-oxygen-mean --vdw-radius-scale 0.85 \
  --output-root /absolute/path/new-fixed-site-screen

# Select one safe MMFF94s single-point minimum per skeleton and re-screen it.
/home/software/anaconda3/envs/rdkit/bin/python scripts/select_single_pc_roll.py \
  --candidate-ensemble /absolute/path/new-pc-rolls/pc-roll-candidate-ensemble.json \
  --cpu-manifest /absolute/path/fixed-site/manifest.json \
  --template-sdf /absolute/path/reviewed-explicit-h-source.sdf \
  --output-dir /absolute/path/new-single-roll-selection
.venv/bin/python scripts/mace_conformation_scan.py \
  --mode fixed-site-cpu --sam /absolute/path/reviewed-h0.extxyz \
  --candidate-ensemble /absolute/path/new-single-roll-selection/selected-candidate-ensemble.json \
  --substrate <registered-substrate-key> --substrate-catalog /absolute/path/catalog \
  --site-prototype-id <registered-site-prototype> --site-cell <cell-index> \
  --donor-assignment '<registered-donor-map>' \
  --organic-z-floor-at-anchor-oxygen-mean --vdw-radius-scale 0.85 \
  --output-root /absolute/path/new-selected-pose-screen
.venv/bin/python scripts/mace_conformation_scan.py \
  --mode fixed-site-optimize-plan \
  --cpu-manifest /absolute/path/new-selected-pose-screen/manifest.json \
  --model /absolute/path/registered-model.model --generalized \
  --formal-output-root /absolute/path/new-mace-plan \
  --plan /absolute/path/new-mace-plan.json
```

The selected ensemble and its exact rescreen manifest must be sealed together.
Never reuse a screen receipt after changing candidate IDs or coordinates. Use
the locked snapshot's `--help` when an exact release has different CLI argument
names; do not patch an immutable release.

## Evidence and limits

In the existing 2Nap-Ac phosphonate case, 20,736 grid points produced 3,655
early-safe starts; 2,765 passed post-optimization common-phase checks and
formed 121 symmetry-aware representatives. At one registered ITO site, 1,292
of 1,452 P-C rolls passed, one pose per skeleton went to MACE, 121 surface
structures were valid, 81 geometry clusters remained, and seven energy/area
Pareto or endpoint candidates were selected. Only one site prototype was
searched. These are recorded counts for that molecule and package, not targets
or guarantees for future SAMs.

The source-ingestion receipt's derived-SDF byte hash differed from the actual
search SDF hash, although formula and canonical graph identity were checked.
The downstream runs bind to the actual search input; the byte-level receipt
discrepancy remains unresolved and must not be rewritten. Before a new SAM run,
verify its source graph, H0 map, torsion adjacency, symmetry maps and site
package. The route does not claim global coverage of every continuous
conformation, every registered ITO site or every possible grown-monolayer
minimum.
