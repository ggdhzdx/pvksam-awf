# PVKSAM AWF public distribution

Current public release: **`pvksam@1.0.0-rc.23`**, with **`skill.pvksam@1.0.0-rc.21`**. This repository publishes the exact registered AWF closure, including the Workflow's release-engineering scripts and the four locked Components. Older versioned source trees and older archives are retained for history; the selected `awf/registry.yaml` and the current archive manifest identify the current public closure.

AWF uses a rolling current pointer. Historical `candidate` labels do not block use. Existing Releases remain immutable; fixes are published as successors.

## Included and updated

- Exact Workflow descriptor, version lock, source manifest, selected Registry, Component descriptors and authoritative integrity-attestation records.
- `skill.pvksam@1.0.0-rc.21`, `recipe.pvksam-ase-cueq@1.0.0-rc.3`, `artifact.mace-mpa-0-medium@1.0.0-rc.2` and `artifact.pvksam-reference-models@1.0.0-rc.1`.
- Current H0 geometry/restraint checks, surface-protonation and annealing scripts, with the post-adsorption restrained-relaxation defaults recorded by this Release: 5000-step cap, `fmax=0.05 eV/A`, and `maxstep=0.02 A`. Geometry, restraint and protonation audits remain separate requirements. The included limited-case evidence does not establish universal transferability.
- Maintained ITO(111) and FTO(110) catalogs, bulk/source structures, relaxed slabs, and phosphonic-acid/carboxylic-acid site packages.
- Six explicit-H SAM source SDFs: 1Br2PADCB, 1Br4PADCB, 2Br2PADCB, 2Nap-Ac, 3Nap-Ac and 4Nap-Ac; three historical coordinate-only DBF models; and optimized reference SDFs for the three brominated molecules.
- Complete locked source, tests, provenance and release-engineering helpers, including `package_closure.py` and the frozen `verify.py`.

Select SAM inputs explicitly from `components/artifacts/pvksam-reference-models/releases/1.0.0-rc.1/source/MODEL-CATALOG.json` under the extracted AWF root. The Skill discovers the bundled substrate catalog. DBF21ID, DBF34ID and DBF43ID do not preserve authoritative bond orders and require topology review before use as source molecules.

## Exact archives and receipts

The current distribution is in `dist/pvksam-1.0.0-rc.23/`:

- `pvksam-awf-1.0.0-rc.23.tar.gz` and `.zip` contain the identical exact closure, with a top-level `agent_workflows/` directory.
- `manifest.json` records member hashes, bytes, types and Unix modes. `SHA256SUMS` identifies the archives.
- `original-manifest.json`, `original-SHA256SUMS` and `PUBLICATION-RECEIPT.json` retain the original receipt and its SHA-256. The inherited `pvksam-rc20-closure` archive names were changed for publication; archive bytes and member records were not changed. The original receipt already identifies Workflow rc.23.

Git does not preserve every directory permission or empty directory. Use the TAR archive with permission preservation for integrity checks and deployment. The `awf/` tree is also available for source browsing. Its `registry.fragment.json` contains the current selection and preserved historical rc16 integrity metadata as merge input for another AWF installation; it must not replace that installation's global Registry. Mutable Component indexes retain previously published release records alongside the current selection.

## Verify on Linux

Run from the repository root, using a new extraction directory. These checks do not start a scientific calculation or download software or model weights.

```bash
REPO="$PWD"
(cd "$REPO/dist/pvksam-1.0.0-rc.23" && sha256sum -c SHA256SUMS)
mkdir pvksam-rc23-check
# --same-permissions preserves modes used by the AWF tree integrity contract.
tar --same-permissions --no-same-owner -xzf \
  "$REPO/dist/pvksam-1.0.0-rc.23/pvksam-awf-1.0.0-rc.23.tar.gz" \
  -C pvksam-rc23-check
export AWF_ROOT="$PWD/pvksam-rc23-check/agent_workflows"
RELEASE_SOURCE="$AWF_ROOT/workflows/pvksam/releases/1.0.0-rc.23/source"
python3 -B "$RELEASE_SOURCE/verify.py" \
  --workflow pvksam --release 1.0.0-rc.23 --repository-only --json
python3 -B "$RELEASE_SOURCE/package_closure.py" --awf-root "$AWF_ROOT" \
  --verify-output-dir "$REPO/dist/pvksam-1.0.0-rc.23" --json
```

To reproduce the package from the extracted, unchanged closure, use a fresh output directory:

```bash
python3 -B "$RELEASE_SOURCE/package_closure.py" --awf-root "$AWF_ROOT" \
  --output-dir "$REPO/pvksam-rc23-repacked" --json
```

The frozen helper keeps its historical output basenames; inspect its manifest for the exact Workflow identity. Ordinary packaging verifies the frozen Workflow source manifest first. Do not run `--prepare-manifest` on a registered Release. The `publish_release.py` and its original README document historical release registration; they are retained engineering records and are not installation steps for this public package. Historical machine paths in provenance and limited-case evidence are audit references; set `AWF_ROOT` and explicit command arguments for a new installation.

## External runtime and model

Python runtime dependencies and MACE weights are external. The external MACE payload identity is 79,462,305 bytes with SHA-256 `75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638`. Obtain it through a legitimate source and set `PVKSAM_MACE_MODEL`; weights are not redistributed here. See the frozen Recipe and `MODEL-NOTICE.md` before preparing a runtime.

Scientific structure records retain their source terms and provenance. See `DATA-NOTICE.md` and `THIRD-PARTY-NOTICES.md`; the repository code license does not relicense third-party scientific data. Engineering verification and file identity do not establish universal conformer coverage, adsorption minima, force-field transferability or device performance.
