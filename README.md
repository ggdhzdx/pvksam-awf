# PVKSAM AWF public distribution

Portable public distribution of `pvksam@1.0.0-rc.17` (`AWF PVKSAM`). It implements the selected phosphonate SAM route for registered oxide substrates: systematic skeleton sampling, fixed-head optimization, site roll screening, MACE surface optimization, monolayer growth, proton placement and cyclic annealing.

The exact Release is recorded because AWF uses a rolling current pointer. The historical `candidate` label is retained for compatibility and audit history; it does not block task-scoped use.

## Included

- exact Workflow descriptor and immutable version lock;
- `skill.pvksam@1.0.0-rc.15`;
- `recipe.pvksam-ase-cueq@1.0.0-rc.2`;
- portable identity for `artifact.mace-mpa-0-medium@1.0.0-rc.2`;
- tests, small synthetic fixtures and scientific evidence boundaries.

Private workspace paths, project ITO/SAM structures, substrate/site packages, trajectories, results, credentials and model weights are excluded.

## External MACE model

Obtain the model through a legitimate source and set `PVKSAM_MACE_MODEL` to its local path. Required identity:

- bytes: `79,462,305`
- SHA-256: `75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638`

This repository does not authorize downloading or redistributing the model weights.

## Verify

```bash
python verify.py
```

The standalone verifier checks the repository manifest, exact Workflow lock, public-content exclusions, Python syntax and external-model identity declaration. Scientific execution additionally requires the environment and project inputs described by the Recipe.

## Use with AWF

`awf/registry.fragment.json` is merge input, not a replacement registry. Publish the listed immutable Releases through the recipient AWF installation's normal process, merge the fragment, and run that installation's verifier before use.

Engineering verification does not establish universal conformer coverage, an ITO-bound global minimum, force-field transferability or device performance.
