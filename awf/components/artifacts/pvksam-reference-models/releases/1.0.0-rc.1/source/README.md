# PVKSAM reference models

This immutable Artifact bundles the reusable structure inputs requested for the
PVKSAM Workflow:

- maintained ITO and FTO catalogs, including source/bulk structures, relaxed
  slabs and phosphonic-acid/carboxylic-acid adsorption-site packages;
- six authoritative explicit-H SAM source SDFs: 1Br2PADCB, 1Br4PADCB,
  2Br2PADCB, 2Nap-Ac, 3Nap-Ac and 4Nap-Ac;
- lowest-energy optimized reference SDFs for the three brominated molecules;
- coordinate-only molecular extractions for the historically used DBF21ID,
  DBF34ID and DBF43ID systems.

Use `MODEL-CATALOG.json` to select an input explicitly. The Artifact records
availability and provenance; it does not assert that a model is chemically
appropriate for a new calculation. The DBF extxyz files do not preserve
authoritative bond orders and must not be substituted for a reviewed source SDF
without topology review.

The bundled substrate catalog is discovered automatically by the locked Skill.
For a custom catalog, set `SAMFLOW_SUBSTRATE_CATALOG`.
