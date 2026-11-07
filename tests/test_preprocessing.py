"""Tests for the backend-independent preprocessing record and its NumPy helpers.

The target is a known distribution: a correlated Gaussian in
``(chirp_mass_source, ln(1 - q), ln lambda_1)``, so the physical density is
analytic. Flows are trained on it with the standard scaler, ``log_lambda`` and
``log_one_minus_q``.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import lognorm, multivariate_normal, norm

from neural_priors_gym.config.schema import TrainingConfig
from neural_priors_gym.flows.base import FlowBase
from neural_priors_gym.preprocessing import (
    flow_to_physical,
    in_support,
    load_preprocessing,
    make_preprocessing_record,
    physical_log_prob,
    physical_to_flow,
)
from neural_priors_gym.training.trainer import FlowTrainer

NAMES = ["chirp_mass_source", "mass_ratio", "lambda_1"]
MEAN = np.array([1.19, np.log(0.2), 6.0])
STD = np.array([0.01, 0.3, 0.4])
CORR = np.array([[1.0, 0.5, -0.6], [0.5, 1.0, -0.3], [-0.6, -0.3, 1.0]])
COV = CORR * np.outer(STD, STD)


def _to_latent(x: np.ndarray) -> np.ndarray:
    return np.column_stack([x[:, 0], np.log(1.0 - x[:, 1]), np.log(x[:, 2])])


def _draw(n: int, seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).multivariate_normal(MEAN, COV, size=n)
    return np.column_stack([v[:, 0], 1.0 - np.exp(v[:, 1]), np.exp(v[:, 2])])


def _true_log_prob(x: np.ndarray) -> np.ndarray:
    # Change of variables written out by hand: |dq/da| = 1 - q, |dL/db| = L.
    return (
        multivariate_normal(MEAN, COV).logpdf(_to_latent(x))
        - np.log(1.0 - x[:, 1])
        - np.log(x[:, 2])
    )


def _record(scale=(2.0, 0.5, -3.0), shift=(0.1, -0.2, 0.3)) -> dict:
    return make_preprocessing_record(
        parameter_names=NAMES,
        scaler={
            "type": "test",
            "scale": list(scale),
            "min": list(shift),
            "data_min": [0] * 3,
            "data_max": [1] * 3,
        },
        log_transform=["lambda_1"],
        log1m_transform=["mass_ratio"],
        bounds_min=[1.1, 0.0, 1.0],
        bounds_max=[1.3, 1.0, 5000.0],
    )


# ---------------------------------------------------------------------------
# Helpers against exact answers
# ---------------------------------------------------------------------------


def test_round_trip_physical_flow_physical() -> None:
    meta = _record()
    x = _draw(1000, seed=1)
    y, _ = physical_to_flow(x, meta)
    np.testing.assert_allclose(flow_to_physical(y, meta), x, rtol=1e-13, atol=0)
    # Explicit forward map for one point.
    y0 = np.array(
        [
            x[0, 0] * 2.0 + 0.1,
            np.log(1 - x[0, 1]) * 0.5 - 0.2,
            np.log(x[0, 2]) * -3.0 + 0.3,
        ]
    )
    np.testing.assert_allclose(y[0], y0, rtol=1e-14)


def test_log_det_matches_finite_difference_jacobian() -> None:
    meta = _record()
    x = _draw(30, seed=2)
    _, log_det = physical_to_flow(x, meta)
    for xi, ld in zip(x, log_det):
        h = 1e-6 * np.maximum(np.abs(xi), 1e-3)
        jac = np.empty((3, 3))
        for j in range(3):
            e = np.zeros(3)
            e[j] = h[j]
            jac[:, j] = (
                physical_to_flow(xi + e, meta)[0] - physical_to_flow(xi - e, meta)[0]
            ) / (2 * h[j])
        np.testing.assert_allclose(ld, np.linalg.slogdet(jac)[1], atol=1e-7)


def test_physical_log_prob_exact_with_known_flow_density() -> None:
    """With a known flow-space density, the physical density matches scipy's log-normals.

    Independent dimensions: Mc ~ N, 1 - q ~ LogNormal, Lambda ~ LogNormal. The
    flow-space density of y = u * s + m is the Gaussian of u pushed through the
    affine map; the expected physical density uses scipy's lognorm directly.
    """
    meta = _record()
    s, m = np.array([2.0, 0.5, -3.0]), np.array([0.1, -0.2, 0.3])

    def flow_log_prob(y):
        return np.sum(norm(MEAN * s + m, STD * np.abs(s)).logpdf(y), axis=-1)

    x = _draw(500, seed=3)
    expected = (
        norm(MEAN[0], STD[0]).logpdf(x[:, 0])
        + lognorm(STD[1], scale=np.exp(MEAN[1])).logpdf(1.0 - x[:, 1])
        + lognorm(STD[2], scale=np.exp(MEAN[2])).logpdf(x[:, 2])
    )
    np.testing.assert_allclose(
        physical_log_prob(flow_log_prob, x, meta), expected, rtol=0, atol=1e-10
    )


def test_support_and_minus_inf_outside() -> None:
    meta = _record()
    x = np.array(
        [
            [1.2, 0.5, 100.0],
            [1.2, 1.0, 100.0],
            [1.2, 1.5, 100.0],
            [1.2, 0.5, 0.0],
            [1.2, 0.5, -1.0],
            [1.2, np.nan, 1.0],
        ]
    )
    np.testing.assert_array_equal(
        in_support(x, meta), [True, False, False, False, False, False]
    )
    calls = []

    def flow_log_prob(y):
        calls.append(y.copy())
        return np.zeros(len(y))

    lp = physical_log_prob(flow_log_prob, x, meta)
    assert np.isfinite(lp[0]) and np.all(lp[1:] == -np.inf)
    assert (
        len(calls) == 1 and calls[0].shape == (1, 3) and np.all(np.isfinite(calls[0]))
    )
    # flow_to_physical never leaves the support: any finite y maps inside it.
    y = np.random.default_rng(0).normal(scale=5.0, size=(10_000, 3))
    xb = flow_to_physical(y, meta)
    assert np.all(xb[:, 1] < 1) and np.all(xb[:, 2] > 0)


def test_no_scaler_and_no_transforms_is_identity() -> None:
    meta = make_preprocessing_record(NAMES, None, [], [], [0] * 3, [1] * 3)
    x = _draw(10, seed=4)
    y, log_det = physical_to_flow(x, meta)
    np.testing.assert_array_equal(y, x)
    np.testing.assert_array_equal(log_det, 0.0)


def test_record_consistent_with_flowjax_prior() -> None:
    """The NumPy helpers agree with FlowjaxPrior's own JAX maps."""
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    from neural_priors_gym.flows.flowjax_prior import build_flowjax_flow, make_prior

    meta = {
        **_record(scale=(2.0, 0.5, 3.0)),
        "bounds": {"min": [0.0] * 3, "max": [1.0] * 3},
    }
    prior = make_prior(
        build_flowjax_flow(n_inputs=3, nn_width=8, nn_depth=1, dtype="float64"),
        meta,
        dtype="float64",
    )
    x = _draw(200, seed=5)
    y, log_det = physical_to_flow(x, meta)
    yj, log_det_j = prior.physical_to_flow(x)
    np.testing.assert_allclose(y, np.asarray(yj), rtol=1e-12)
    np.testing.assert_allclose(log_det, np.asarray(log_det_j), rtol=1e-12)
    np.testing.assert_allclose(
        flow_to_physical(y, meta), np.asarray(prior.flow_to_physical(y)), rtol=1e-12
    )
    lp = physical_log_prob(lambda yy: np.asarray(prior.flow.log_prob(yy)), x, meta)
    np.testing.assert_allclose(lp, np.asarray(prior.log_prob(x)), rtol=1e-12)


# ---------------------------------------------------------------------------
# Trained glasflow model
# ---------------------------------------------------------------------------


def _config(tmp_path: Path, flow: dict) -> TrainingConfig:
    return TrainingConfig(
        output_dir=str(tmp_path),
        flow=flow,  # type: ignore[arg-type]
        training={  # type: ignore[arg-type]
            "parameter_names": NAMES,
            "data_path": str(tmp_path / "unused.npz"),
            "log_lambda": True,
            "log_one_minus_q": True,
            "scaler": "standard",
            "num_epochs": 150,
            "learning_rate": 3e-3,
            "batch_size": 256,
            "max_patience": 30,
            "validation_split": 0.2,
            "scale_input": True,
        },
    )


GLASFLOW = {
    "backend": "glasflow",
    "n_transforms": 4,
    "n_neurons": 32,
    "n_blocks_per_transform": 1,
    "num_bins": 8,
}


@pytest.fixture(scope="module")
def glasflow_run(tmp_path_factory):
    import torch

    torch.manual_seed(0)
    tmp_path = tmp_path_factory.mktemp("glasflow_pre")
    x = _draw(10_000, seed=0)
    data = {name: x[:, i] for i, name in enumerate(NAMES)}
    model_dir = tmp_path / "model"
    flow, *_ = FlowTrainer(_config(tmp_path, GLASFLOW)).train(
        data, NAMES, output_dir=model_dir
    )
    flow.save(model_dir)
    return {"x": x, "flow": flow, "model_dir": model_dir}


def test_trained_glasflow_record_contents(glasflow_run) -> None:
    meta = load_preprocessing(glasflow_run["model_dir"])
    assert meta is not None
    assert meta["parameter_names"] == NAMES
    assert meta["log_transform"] == ["lambda_1"]
    assert meta["log1m_transform"] == ["mass_ratio"]
    assert meta["scaler"]["type"] == "StandardScaler"
    x = glasflow_run["x"]
    np.testing.assert_allclose(meta["bounds"]["min"], x.min(0))
    np.testing.assert_allclose(meta["bounds"]["max"], x.max(0))
    # Standard scaler: y = (u - mean) / sd of the transformed training data.
    u = _to_latent(x)
    np.testing.assert_allclose(meta["scaler"]["scale"], 1 / u.std(0), rtol=1e-10)
    np.testing.assert_allclose(
        meta["scaler"]["min"], -u.mean(0) / u.std(0), rtol=1e-8, atol=1e-12
    )
    # The pickled scaler is still written for torch backends.
    assert (glasflow_run["model_dir"] / "scaler.gz").exists()


def test_trained_glasflow_physical_density_matches_analytic(glasflow_run) -> None:
    """Short fit, so a loose tolerance (measured: mean -0.018, median |diff| 0.084 nat)."""
    flow = glasflow_run["flow"]
    meta = flow.preprocessing
    x = _draw(4000, seed=123)
    d = _to_latent(x) - MEAN
    core = np.einsum("ij,jk,ik->i", d, np.linalg.inv(COV), d) < 2.5**2
    x = x[core]
    lp = physical_log_prob(flow.log_prob, x, meta)
    diff = lp - _true_log_prob(x)
    # Mean of (truth - flow) over the core is a KL-like discrepancy.
    assert abs(np.mean(diff)) < 0.05, np.mean(diff)
    assert np.median(np.abs(diff)) < 0.15, np.median(np.abs(diff))
    # Dropping the log/log1m Jacobian terms would be off by ln(1-q) + ln(Lambda) ~ 4.4 nat.
    y, _ = physical_to_flow(x, meta)
    assert (
        np.median(
            np.abs(
                flow.log_prob(y)
                + np.sum(np.log(np.abs(meta["scaler"]["scale"])))
                - _true_log_prob(x)
            )
        )
        > 2
    )


def test_trained_glasflow_samples_in_support_and_match_truth(glasflow_run) -> None:
    from scipy.stats import ks_2samp

    flow = glasflow_run["flow"]
    xs = flow_to_physical(flow.sample(5000), flow.preprocessing)
    assert np.all(xs[:, 1] < 1) and np.all(xs[:, 2] > 0)
    truth = _draw(5000, seed=7)
    for i in range(3):
        assert ks_2samp(xs[:, i], truth[:, i]).pvalue > 1e-3, NAMES[i]


def test_glasflow_save_load_round_trip_identical(glasflow_run) -> None:
    flow = glasflow_run["flow"]
    loaded = FlowBase.load_from_dir(glasflow_run["model_dir"])
    assert loaded.preprocessing == flow.preprocessing
    assert json.loads(json.dumps(flow.preprocessing)) == load_preprocessing(
        glasflow_run["model_dir"]
    )
    x = _draw(500, seed=9)
    np.testing.assert_array_equal(
        physical_log_prob(loaded.log_prob, x, loaded.preprocessing),
        physical_log_prob(flow.log_prob, x, flow.preprocessing),
    )
    z = np.random.default_rng(0).normal(size=(200, 3))
    np.testing.assert_array_equal(loaded.latent_to_data(z), flow.latent_to_data(z))


def test_old_format_model_loads_unchanged(tmp_path: Path) -> None:
    """A glasflow model without the record (as saved before it existed) loads as before."""
    import joblib
    import torch
    from sklearn.preprocessing import MinMaxScaler

    from neural_priors_gym.config.schemas.flow import GlasflowConfig
    from neural_priors_gym.flows.glasflow import GlasflowNSF

    torch.manual_seed(1)
    flow = GlasflowNSF.from_config(
        GlasflowConfig(
            n_transforms=2, n_neurons=16, n_blocks_per_transform=1, num_bins=4
        ),
        3,
    )
    assert flow.preprocessing is None
    flow.save(tmp_path)
    kwargs = json.loads((tmp_path / "model_kwargs.json").read_text())
    assert "preprocessing" not in kwargs
    x = _draw(300, seed=11)
    scaler = MinMaxScaler().fit(x)
    joblib.dump(scaler, tmp_path / "scaler.gz")

    loaded = FlowBase.load_from_dir(tmp_path)
    assert loaded.preprocessing is None
    assert load_preprocessing(tmp_path) is None
    y = joblib.load(tmp_path / "scaler.gz").transform(x)
    old = flow.log_prob(y) + FlowBase.compute_log_jacobian_correction(scaler)
    new = loaded.log_prob(y) + FlowBase.compute_log_jacobian_correction(
        joblib.load(tmp_path / "scaler.gz")
    )
    np.testing.assert_array_equal(new, old)


def test_flowjax_writes_same_record_as_glasflow(tmp_path: Path) -> None:
    """Same data and training options: the flowjax record equals the glasflow one."""
    pytest.importorskip("flowjax")
    x = _draw(2000, seed=0)
    data = {name: x[:, i] for i, name in enumerate(NAMES)}
    records = []
    for name, flow_cfg in [
        ("glasflow", {**GLASFLOW, "n_transforms": 1}),
        (
            "flowjax",
            {"backend": "flowjax", "n_transforms": 1, "nn_width": 8, "nn_depth": 1},
        ),
    ]:
        cfg = _config(tmp_path / name, flow_cfg)
        cfg.training.num_epochs = 1
        flow, *_ = FlowTrainer(cfg).train(data, NAMES, output_dir=tmp_path / name)
        flow.save(tmp_path / name)
        records.append(load_preprocessing(tmp_path / name))
        # Loading and re-saving keeps the record, including log1m_transform.
        reloaded = FlowBase.load_from_dir(tmp_path / name)
        reloaded.save(tmp_path / f"{name}_resaved")
        assert load_preprocessing(tmp_path / f"{name}_resaved") == records[-1]
    assert records[0] == records[1]
