# PVKSAM rc.23 release engineering

This successor keeps the rc.22 route and locked Recipe/Artifacts. It updates the
post-adsorption restrained-relaxation defaults to a 5000-step cap,
`fmax=0.05 eV/A`, and the existing `maxstep=0.02 A` for H0, post-protonation,
and post-annealing-cycle relaxations. Geometry, restraint and protonation audits
remain separate required gates. The target has one similar-size 2Nap-Ac/ITO
precedent and is not a universal cross-molecule calibration.

The lock is exact: `skill.pvksam@1.0.0-rc.21`,
`recipe.pvksam-ase-cueq@1.0.0-rc.3`,
`artifact.mace-mpa-0-medium@1.0.0-rc.2`, and
`artifact.pvksam-reference-models@1.0.0-rc.1`. No model or software download,
production calculation, annealing job, or project-lock migration is part of
publication. Historical workflow releases and the project's rc.14 declaration
are preserved.

Every release-engineering script used here is included under this Release's
`source/` or the new Skill Release's `source/engineering/`. The exact Skill Draft
was created from rc.20 with the co-located `manage_drafts.py` on node01 because
node02's NFS mount rejects `renameat2(RENAME_NOREPLACE)`. The failed NFS attempt
is retained in the project's release evidence. The Draft itself remains an
unregistered review input; only the frozen rc.21 source is locked by this
Workflow.

```bash
export AWF_ROOT=/home/zc/agent_workflows
export PVKSAM_EVIDENCE=/home/zc/sam-ito/analysis/awf-pvksam-h0-relaxation-20261010/attempt02
export PVKSAM_RELEASE_SRC="$AWF_ROOT/workflows/pvksam/releases/1.0.0-rc.23/source"
export PVKSAM_SKILL_SRC="$AWF_ROOT/components/skills/pvksam/releases/1.0.0-rc.21/source"

python -B "$PVKSAM_RELEASE_SRC/publish_release.py" --evidence "$PVKSAM_EVIDENCE" --phase prepare
/home/zc/sam-ito/.venv/bin/python -B "$PVKSAM_RELEASE_SRC/publish_release.py" --evidence "$PVKSAM_EVIDENCE" --phase finalize
/home/zc/sam-ito/.venv/bin/python -B "$PVKSAM_RELEASE_SRC/package_closure.py" --prepare-manifest --json
/home/zc/sam-ito/.venv/bin/python -B "$PVKSAM_RELEASE_SRC/publish_release.py" --evidence "$PVKSAM_EVIDENCE" --phase publish
/home/zc/sam-ito/.venv/bin/python -B "$PVKSAM_RELEASE_SRC/publish_release.py" --evidence "$PVKSAM_EVIDENCE" --phase publish --execute
/home/zc/sam-ito/.venv/bin/python -B "$PVKSAM_RELEASE_SRC/publish_release.py" --evidence "$PVKSAM_EVIDENCE" --phase accept
```

`prepare` verifies and records pre-change metadata backups. `finalize` runs the
complete packaged Skill tests and CLI help checks, records commands, exit codes
and log hashes, and writes source provenance. `package_closure.py
--prepare-manifest` freezes this Workflow's complete engineering source before
publication. The first `publish` call is a dry-run; `--execute` applies the
preconditioned per-file writes with Registry last. `accept` checks the exact
current release, historical rc.14 integrity, restore plans, exact package
closure, and a verified full-core backup. All evidence paths are recorded under
the project evidence directory.

The restore and package checks are validation only. They do not install a
runtime skill, alter the historical project pin, or start a scientific job.
