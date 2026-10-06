# Maintained adsorption probes

This directory stores immutable, hash-verified neutral molecular geometry seeds used
by the substrate-maintenance adsorption-site workflow. A probe source is reference
geometry data, not a relaxed adsorbate, calculated gas-phase reference energy, or Site
Prototype.

Each probe has its own directory containing:

- `sources/`: downloaded or supplied source bytes, preserved without modification;
- `manifest.toml`: source provenance and hash, expected neutral formula and topology,
  and the explicit deprotonation/retained-proton policy.

Preparation code infers the declared topology from coordinates and fails if it cannot
identify the expected acid H atoms. It never chooses a deprotonation state from a
filename. New probe chemistry requires a new maintained manifest and validation record.

Current entries:

- `methylphosphonic_acid`: PubChem CID 13818 neutral CH5O3P 3D seed; preparation removes
  the two O-bound H atoms to make CH3PO3 and places both released H atoms on the surface.
- `acetic_acid`: PubChem CID 176 neutral C2H4O2 3D seed; preparation identifies the
  carboxyl carbon topologically (rather than assuming carbon is unique), removes the
  one O-bound acidic H to make acetate C2H3O2, and places the released H on the surface.
  Independent `carboxylic_acid` Site Prototype libraries have passed and are maintained
  separately from phosphonic-acid libraries for both undoped In2O3(111) and SnO2(110).
