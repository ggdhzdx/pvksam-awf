---
name: pvksam
description: Build phosphonate SAM adsorption structures on registered oxide substrates through a single systematic conformer route, MACE surface optimization, monolayer growth, restrained relaxation, global proton placement, and convergence-checked cyclic annealing. Use when the user invokes AWF PVKSAM or requests this complete SAM workflow.
---

# AWF PVKSAM

Resolve `pvksam` from the AWF Registry once per task and record the exact
Release. Use the locked scripts shipped with that Release. Keep molecule
inputs, substrate catalogs, model weights, run directories and scientific
results external to the shared component.

## Scope

From an original molecule, the current route supports one reviewed phosphonic
acid/phosphonate anchor on a registered oxide surface. It preserves atom
identity, released-proton counts and adsorption-site mappings. Carboxylate
post-H0 handling exists, but its original-molecule conformer/placement adapter
is not connected to this route. Other anchor types fail with a diagnostic.

PVKSAM has one initial-conformer route: the 30-degree isolated phosphonate
skeleton grid, followed by fixed-head MMFF94s optimization, continuous P-C
phase recovery, symmetry-aware skeleton deduplication, site-bound P-C rotation
screening and one-pose-per-skeleton MMFF94s single-point selection. ConfSearch
and the historical fixed-site 216-start grid are removed from the PVKSAM
configuration and current route. The independent ConfSearch Workflow remains
available for other tasks, but it is not an alternate PVKSAM branch. Unsupported
route keys are rejected rather than silently substituted.

Read `references/initial-conformer-search.md` before running the molecular or
site-bound conformer stages. Read `references/pvksam.zh-CN.md` for the full
stage contract, parameters, growth defaults, post-H0 restraints and evidence
limits. The component bundles current reusable scripts and tests; a prepared
CPU skeleton scan is only the first handoff, not a completed adsorption layer.

## Execute the route

1. Review the explicit-H source SDF, charge, bond orders, stereochemistry and
   original atom IDs. If the source is an image/PDF/PPT depiction, use the
   project `structure_source_ingest.py` adapter first and preserve its receipt.
   Remove only the declared acidic protons to produce H0; do not infer topology
   from optimized coordinates.
2. Define and review every internal four-atom torsion using this SAM's original
   1-based atom IDs. Normalize the three PO3 O atoms to `z=0` with one proper
   rigid-body transform and place P at positive z. Scan the unique 30-degree
   grid for internal torsions, excluding the P-C axial roll.
3. Before any force-field optimization, reject severe intramolecular contacts
   and solve the full shared P-C phase intervals for all organic atoms to be at
   `z >= 0` while clearing the fixed PO3 head. Unsafe grid points never enter
   optimization. Optimize only survivors using RDKit MMFF94s with P and the
   three anchor O atoms fixed. Recover the shared phase again, then deduplicate
   using the reviewed H0 graph's exact symmetry mappings and the energy-ordered
   1.0-A representative-radius rule. Keep every safe family member.
4. Map each representative to the selected registered ITO site while
   preserving donor identities. Enforce the default-on exclusion of any site
   metal with six or more pre-adsorption substrate-O neighbors (2.7-A cutoff,
   maximum allowed CN 5). Rotate the carbon-side fragment about P-C in 30-degree
   increments. Apply head-fit, mean anchor-O height, intramolecular and periodic
   surface collision gates before energy calculation.
5. For each skeleton with at least one safe roll, compute MMFF94s single-point
   energies on the same H0 graph and retain exactly one minimum-energy safe
   roll. This selects an initial pose; it does not optimize coordinates or
   claim an adsorption-energy minimum. Re-screen the selected ensemble and
   produce an exact MACE plan. Run the locked MACE backend, inspect convergence
   and geometry receipts, then perform the existing fixed-site clustering and
   area/energy Pareto selection.
6. Build the H0 layer using the registered growth defaults:
   `cohesive-frontier + continuous-snap`, followed by the enabled dynamic CN5
   single-metal fill stage. Do not treat any past target count such as 82 as a
   required default. Preserve the actual metal-O mappings from growth.
7. Construct `post.json` from the completed H0 manifest, with actual atom
   counts, fixed-layer boundary, interface bonds, model and output paths. Run
   `orchestrate_md_pipeline.py --pvksam-post post.json --plan`, review the
   sealed restraints, then execute the exact plan. Stop when a force, P
   tetrahedron, contact, protonation or other required stage check fails.

The `--pvksam-front` entry validates and seals the H0 input and performs the
initial skeleton CPU scan only. It returns
`awaiting_fast_skeleton_optimization`; the later force-field, deduplication,
site-roll, MACE and growth stages are explicit handoffs described in the route
reference. Never report the CPU handoff as a completed adsorption layer.

## Adsorption and growth defaults

Before placement, use `--metal-coordination-filter exclude-six-coordinated
--metal-coordination-cutoff-A 2.7 --metal-coordination-max-allowed 5` and
verify the audit manifest. The H0 entry independently verifies every mapped
metal. Keep the chosen metal identity from the site/growth manifest; never
replace it with a fresh nearest-metal search.

Use the locked growth policy: three-coordinate candidates, then two-coordinate
candidates, then enabled `dynamic-cn5-uncovered` single-metal filling. Preserve
`cohesive-frontier + continuous-snap`, the registered seed and all selection,
collision and projection settings in the run manifest. The substrate/site
package and molecule determine the achieved count; no count is promised.

## H0, global protonation and annealing contract

- Derive and seal original SAM bonds, metal-O interface pairs and every
  supported P tetrahedron from the intact H0 structure before the first
  optimization. Heavy-atom and H-containing SAM bond constants are respectively
  20 and 10 eV/A^2. Keep actual mono-, bi- and tridentate metal-O mappings with
  one-sided 100 eV/A^2 stretching restraints and their existing target lengths.
- Every P must have the reviewed intact 3O+1C tetrahedral reference and all six
  ligand-angle terms from the first relaxation. Do not invent P restraints for
  carboxylates.
- Place surface H globally using the proton ledger: two H per phosphonate and
  one per carboxylate. Newly created surface OH is free in relaxation and MD;
  do not constrain its bond, angle or position.
- Relax to `fmax=0.01 eV/A`, then anneal in repeating 300→500→300 K cycles,
  1500 steps at 2 fs per cycle (3 ps), with the locked 4 amu H mass scaling.
  Check force convergence, every applicable P tetrahedron and short contacts
  after each cycle. Stop only when the best-energy improvement over the latest
  five cycles is below 0.005 eV/SAM and all gates pass. Check every cycle from
  cycle five. A 20-cycle review cap is not a convergence result.
- Do not apply the obsolete projected O3 triangle rule, non-anchor-height gate
  or post-relaxation full-topology gate. Keep the specified contact and local P
  checks. Report `physical_interface_pass: null` for this protocol.

## Runtime, provenance and upgrades

Before formal calculations, use a run guide matching program build, backend,
hardware, task and system scale. Lock molecule/source hashes, site package,
model, code, runtime, parameters and every stage handoff. Do not download models,
install software or switch backends implicitly. If the run fails, preserve the
evidence, fix the existing module, add a focused regression and resume only into
a new output directory with a new plan.

The cited full initial-conformer evidence is the existing 2Nap-Ac phosphonate
case at one registered ITO site. It does not establish universal coverage,
carboxylate support from original inputs or success on unsearched site
prototypes. The user has deferred two-family end-to-end validation; do not
launch it merely to use or publish this Workflow. If a new molecule exposes a
failure, retain it as a regression and publish a new immutable Release; never
edit an existing Release in place.
