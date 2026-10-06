# PVKSAM AWF public distribution

Portable public distribution of `pvksam@1.0.0-rc.18` (`AWF PVKSAM`). It implements the selected phosphonate SAM route and now includes the reference scientific structures needed to inspect and reuse that route.

The exact Release is recorded because AWF uses a rolling current pointer. The historical `candidate` label is retained for compatibility and audit history; it does not block task-scoped use.

## Included

- exact Workflow descriptor and immutable version lock;
- `skill.pvksam@1.0.0-rc.16` and `recipe.pvksam-ase-cueq@1.0.0-rc.3`;
- maintained ITO(111) and FTO(110) catalogs with bulk/source structures, relaxed slabs and phosphonic-acid/carboxylic-acid site packages;
- six authoritative explicit-H SAM source SDFs: 1Br2PADCB, 1Br4PADCB, 2Br2PADCB, 2Nap-Ac, 3Nap-Ac and 4Nap-Ac;
- three coordinate-only historical used-SAM models: DBF21ID, DBF34ID and DBF43ID;
- lowest-energy optimized reference SDFs for the three brominated molecules;
- tests, provenance records and scientific evidence boundaries.

Select the SAM explicitly from `awf/components/artifacts/pvksam-reference-models/releases/1.0.0-rc.1/source/MODEL-CATALOG.json`. The Skill discovers the bundled substrate catalog automatically. The historical DBF files do not preserve authoritative bond orders and require topology review before use as source molecules.

## External MACE model

The MACE weights remain external. Set `PVKSAM_MACE_MODEL` to a legitimately acquired payload with 79,462,305 bytes and SHA-256 `75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638`.

## Verify

```bash
python verify.py
```

The verifier checks every distributed file, the exact lock, the bundled model catalog and checksums, path privacy, Python syntax and the external MACE identity declaration.

## Use with AWF

`awf/registry.fragment.json` is merge input, not a replacement registry. Publish the listed immutable Releases through the recipient AWF installation, merge the fragment and run that installation's verifier.

Engineering verification and bundled file identity do not establish universal conformer coverage, adsorption minima, force-field transferability or device performance.
