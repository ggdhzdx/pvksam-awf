# PVKSAM ASE/CuEq runtime Recipe

This release reuses the established fixed-cell ASE MACE/CuEq force path. It does
not introduce LAMMPS ML-IAP or silently substitute a model. The reference runtime
is Python 3.11, ASE 3.27.0, MACE 0.3.14, Torch 2.9.1+cu128, CuEq 0.10.0 on node02
RTX5090. Model: the exact external artifact.mace-mpa-0-medium payload in the lock.

Before a real run, consult `~/.codex/program-run-guides/index.yaml` and the guide
`mace-cueq-ase-node02-sam6332-restrained`. Check installed versions, model digest,
GPU occupancy and library availability. Four Torch/OMP/MKL threads, one GPU,
OPENBLAS_NUM_THREADS=1 is the historical recommendation for 6332–6496 atoms;
geometry/method/backend/hardware/size changes need a bounded recalibration, not
an assumption that this layout is universally fastest.

Historical reference: 6496 atoms with P angle/exclusion overlay, median force
call 0.196153 s at 4 threads versus 0.559784 s at 1 thread, peak 8.55 GiB. The
predeclared equivalence limits were 0.001 eV/A force components and 0.05 eV energy.
These values describe the historical adapter, not a new benchmark of this release.
The released backend preserves its analytic potentials; CPU identity/gradient
regressions cover the refactor. No new MACE/GPU or full-chain runs were made for
publication, as requested by the user.

On the reference host:

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH=/home/software/anaconda3/envs/ase/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:/home/software/anaconda3/envs/ase/lib/python3.11/site-packages/nvidia/cublas/lib:/home/software/anaconda3/envs/ase/lib/python3.11/site-packages/torch/lib:/home/software/anaconda3/envs/ase/lib/python3.11/site-packages/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
/home/software/anaconda3/envs/ase/bin/python -u <locked-source>/scripts/orchestrate_md_pipeline.py --pvksam-post /absolute/project/post.json --execute
```

Treat paths as deployment parameters on other hosts. Read-only plans and
engineering tests require Python, NumPy, SciPy and ASE; pytest is test-only.
Publication tests used the project's `.venv`, not the GPU environment (which
has no pytest). No package install, download, resource change, or full-size
calculation is part of publication. Expected external download for this task: 0.
Missing runtimes/model artifacts are reported, never downloaded automatically.

The rc.1 proton candidate geometry supports diagonal positive cells with a/b
periodicity and outward +z. General tilted-cell handling remains deferred.
H mass 4 u is integration mass scaling; exported structures use physical masses.
Energy plateau uses the same restrained potential, fixed reference and atom count
throughout one run. It is a search stopping criterion, not a global-minimum proof.


## Bundled reference structures

The locked `artifact.pvksam-reference-models@1.0.0-rc.1` supplies the ITO/FTO
substrate catalogs and the recorded SAM molecule models. The Skill discovers
the bundled substrate catalog automatically. Select the SAM input explicitly
from `MODEL-CATALOG.json`; do not infer a molecule from a filename or substitute
the coordinate-only historical DBF entries for an authoritative bond-order
source. Model weights remain external through `PVKSAM_MACE_MODEL`.
