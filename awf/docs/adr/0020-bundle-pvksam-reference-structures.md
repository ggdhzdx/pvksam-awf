# ADR 0020: Bundle PVKSAM reference structures

Date: 2026-10-06

## Decision

Publish `artifact.pvksam-reference-models@1.0.0-rc.1` and lock it into
`pvksam@1.0.0-rc.18`. The Artifact contains the maintained ITO and FTO catalogs,
including relaxed slabs and adsorption-site packages, plus nine SAM models used
in the source project.

Six SAM entries are authoritative explicit-H source SDFs. Three historical DBF
entries are coordinate-only molecular extractions from recorded adsorbed
global-minimum CIFs; their bond orders are not authoritative. The Skill selects
no default molecule and requires explicit user/project selection.

The locked Skill discovers the bundled substrate catalog in a normal AWF
checkout. `SAMFLOW_SUBSTRATE_CATALOG` remains an explicit override. MACE weights
remain external and hash-identified.

## Rationale

The earlier public Release contained the workflow code but required project
ITO/SAM files, which prevented a recipient from inspecting or reusing the same
reference structures. A separate immutable Artifact makes the scientific inputs
visible, hash-bound and independently replaceable in a successor Release.

## Evidence boundary

Bundling proves file identity and availability. It does not prove that a surface,
site, conformer or force field is suitable for a new molecule or device. The
structure data keeps its provenance and separate data notice; the repository MIT
license does not relicense third-party scientific records.
