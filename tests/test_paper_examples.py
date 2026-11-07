"""The arXiv:2511.22987 example configs parse, carry the paper's settings and run."""

from pathlib import Path

import pytest
import yaml

from neural_priors_gym.cli import main
from neural_priors_gym.config.parser import load_config
from neural_priors_gym.config.schemas.flow import FlowjaxConfig
from neural_priors_gym.flows.flowjax_prior import load_flowjax_prior

from .test_eos_consistency import _write_jester_h5

EXAMPLES = Path(__file__).parent.parent / "examples" / "paper_2511.22987"
CASES = [
    f"{st}_{mm}"
    for st in ("bns", "nsbh")
    for mm in ("uniform", "gaussian", "double_gaussian")
]
PAPER_MASSES = {
    "uniform": {"m_min": 1.0},
    "gaussian": {"m_min": 1.0, "mean": 1.33, "std": 0.09},
    "double_gaussian": {
        "m_min": 1.0,
        "mean_1": 1.34,
        "std_1": 0.07,
        "weight": 0.65,
        "mean_2": 1.80,
        "std_2": 0.21,
    },
}


@pytest.mark.parametrize("case", CASES)
def test_paper_config_settings(case: str) -> None:
    cfg = load_config(EXAMPLES / case / "config.yaml")
    st, mm = case.split("_", 1)
    assert cfg.source_type == st
    assert cfg.masses is not None and cfg.masses.type == mm
    for k, v in PAPER_MASSES[mm].items():
        assert getattr(cfg.masses, k) == v
    if st == "nsbh":
        assert cfg.m_max_bh == 5.0
        assert cfg.training.parameter_names == [
            "chirp_mass_source",
            "mass_ratio",
            "lambda_2",
        ]
    else:
        assert cfg.training.parameter_names == [
            "chirp_mass_source",
            "mass_ratio",
            "lambda_1",
            "lambda_2",
        ]
    t = cfg.training
    assert (t.n_samples, t.validation_split, t.num_epochs) == (200_000, 0.2, 2000)
    assert (t.learning_rate, t.batch_size, t.max_patience) == (1e-4, 1024, 250)
    assert t.scale_input and not t.log_lambda
    f = cfg.flow
    assert isinstance(f, FlowjaxConfig)
    assert (f.flow_type, f.n_transforms, f.nn_width, f.nn_depth) == (
        "coupling_nsf",
        4,
        256,
        3,
    )


@pytest.mark.parametrize("case", ["bns_double_gaussian", "nsbh_gaussian"])
def test_paper_config_smoke_run(case: str, tmp_path: Path) -> None:
    raw = yaml.safe_load((EXAMPLES / case / "config.yaml").read_text())
    h5 = tmp_path / "results.h5"
    _write_jester_h5(h5)
    raw["lambdas"]["jester_path"] = "results.h5"  # relative to the config file
    raw["output_dir"] = "out"
    raw["flow"].update({"nn_width": 16, "nn_depth": 1, "n_transforms": 2})
    raw["training"].update({"n_samples": 2000, "num_epochs": 2, "batch_size": 256})
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    main(load_config(path))
    prior = load_flowjax_prior(tmp_path / "out" / "model")
    assert list(prior.names) == raw["training"]["parameter_names"]
