# ADR 0019: Publish a portable PVKSAM workflow closure

Status: accepted for `pvksam@1.0.0-rc.17` (2026-10-06).

## Context

PVKSAM rc.14 locked the selected phosphonate skeleton-grid route and preserved
its scientific evidence boundaries, but the Skill still contained private
workspace defaults. The external MACE Artifact also named a user-specific
payload path. Model provenance did not establish redistribution permission.

## Decision

`skill.pvksam@1.0.0-rc.14` replaces private workspace defaults with explicit
arguments or relative output locations. The historical dimer utility now
requires `--model` and accepts `--workdir`.

`artifact.mace-mpa-0-medium@1.0.0-rc.2` publishes only the model identity:
79,462,305 bytes and SHA-256
`75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638`.
The payload remains external and is located with `PVKSAM_MACE_MODEL`. No model
download or redistribution is authorized by the Artifact.

Recipe rc.2 preserves the rc.1 instructions and updates its declared Artifact
dependency to the portable model identity rc.2. Skill rc.15 preserves the
path-neutral rc.14 source and updates its declared Recipe dependency to rc.2.
Workflow rc.17 locks the three consistent public successors. Workflows rc.15
and rc.16 are preserved as audit records of rejected mixed dependencies.
Project structures,
substrate/site packages, trajectories, results,
credentials and model weights remain outside the public closure.

## Consequences

The public repository can distribute the complete executable method without
revealing internal paths or copying the external model. A recipient must
legitimately acquire the exact model, set `PVKSAM_MACE_MODEL`, verify its size
and hash, and provide reviewed project inputs. Scientific claims remain limited
to the recorded one-molecule, one-site evidence.
