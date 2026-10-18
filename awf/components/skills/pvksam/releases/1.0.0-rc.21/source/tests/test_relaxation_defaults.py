from pathlib import Path
import sys

import pytest

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / "scripts"))

import orchestrate_md_pipeline as pipeline
import relax_monolayer as relaxation


def test_locked_post_adsorption_relaxation_defaults_match_worker_defaults():
    assert pipeline.PVKSAM_DEFAULTS["steps"] == 5000
    assert pipeline.PVKSAM_DEFAULTS["fmax"] == pytest.approx(0.05)
    assert pipeline.PVKSAM_DEFAULTS["maxstep"] == pytest.approx(0.02)

    args = relaxation.build_restrained_parser().parse_args(
        [
            "--input", "h0.extxyz",
            "--interface-bonds", "interface.json",
            "--model", "model.model",
            "--output", "output",
            "--mode", "relax",
            "--substrate-atoms", "10",
            "--molecules", "1",
            "--atoms-per-molecule", "10",
            "--fixed-atoms", "5",
        ]
    )
    assert args.steps == 5000
    assert args.fmax == pytest.approx(0.05)
    assert args.maxstep == pytest.approx(0.02)
