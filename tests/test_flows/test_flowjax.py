"""Tests for the flowjax backend and the standalone JAX prior loader.

The flows are trained for a few hundred optimiser steps on distributions whose
physical density is known in closed form, so the physical-unit log density of
the loaded prior can be checked against the truth. The scaler maps the data to
``[0, 1]``; its log-Jacobian here is several nats per dimension, so a missing or
wrong Jacobian fails these tests by far more than their tolerance.
"""

import json
import subprocess
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import multivariate_normal

from neural_priors_gym.config.schema import TrainingConfig
from neural_priors_gym.config.schemas.flow import FlowjaxConfig
from neural_priors_gym.flows.base import FlowBase
from neural_priors_gym.flows.flowjax_flow import FlowjaxFlow
from neural_priors_gym.flows.flowjax_prior import (
    FlowjaxPrior,
    load_flowjax_prior,
    WEIGHTS_FILE,
)
from neural_priors_gym.training.trainer import FlowTrainer


NAMES = ["chirp_mass_source", "mass_ratio", "lambda_2"]

# Latent Gaussian: mean and covariance in the space the analytic density is
# defined in (physical units, or log(lambda_2) for the log-normal case).
MEAN = np.array([1.2, 0.8, 6.0])
STD = np.array([0.01, 0.05, 0.4])
CORR = np.array(
    [
        [1.0, 0.7, -0.4],
        [0.7, 1.0, -0.3],
        [-0.4, -0.3, 1.0],
    ]
)
COV = CORR * np.outer(STD, STD)

# In the Gaussian case lambda_2 is an affine rescaling of the third latent
# coordinate, lambda_2 = LAM_SHIFT + LAM_SCALE * u_2 (mean 400, sd 100); in the
# log-normal case lambda_2 = exp(u_2).
LAM_SCALE = 250.0
LAM_SHIFT = 400.0 - LAM_SCALE * 6.0

N_SAMPLES = 5000


def _make_config(tmp_path: Path, log_lambda: bool, **flow_kwargs) -> TrainingConfig:
    flow = {
        "backend": "flowjax",
        "flow_type": "coupling_nsf",
        "n_transforms": 3,
        "nn_width": 32,
        "nn_depth": 1,
        "num_bins": 8,
        "seed": 1,
    }
    flow.update(flow_kwargs)
    return TrainingConfig(
        output_dir=str(tmp_path),
        flow=flow,  # type: ignore[arg-type]
        training={  # type: ignore[arg-type]
            "parameter_names": NAMES,
            "data_path": str(tmp_path / "unused.npz"),
            "log_lambda": log_lambda,
            "num_epochs": 40,
            "learning_rate": 3e-3,
            "batch_size": 256,
            "max_patience": 40,
            "validation_split": 0.2,
            "scale_input": True,
        },
    )


def _to_latent(x: np.ndarray, log_normal: bool) -> np.ndarray:
    u = np.array(x, dtype=np.float64)
    if log_normal:
        u[:, 2] = np.log(u[:, 2])
    else:
        u[:, 2] = (u[:, 2] - LAM_SHIFT) / LAM_SCALE
    return u


def _draw(log_normal: bool, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.multivariate_normal(MEAN, COV, size=n)
    if log_normal:
        x[:, 2] = np.exp(x[:, 2])
    else:
        x[:, 2] = LAM_SHIFT + LAM_SCALE * x[:, 2]
    return x


def _true_log_prob(x: np.ndarray, log_normal: bool) -> np.ndarray:
    u = _to_latent(x, log_normal)
    log_jac = -np.log(x[:, 2]) if log_normal else -np.log(LAM_SCALE)
    return multivariate_normal(MEAN, COV).logpdf(u) + log_jac  # type: ignore[arg-type]


def _train(tmp_path: Path, log_normal: bool, **flow_kwargs):
    x = _draw(log_normal, N_SAMPLES, seed=0)
    data = {name: x[:, i] for i, name in enumerate(NAMES)}
    config = _make_config(tmp_path, log_lambda=log_normal, **flow_kwargs)
    trainer = FlowTrainer(config)
    model_dir = tmp_path / "model"
    flow, train_losses, val_losses, scaler = trainer.train(
        data, NAMES, output_dir=model_dir
    )
    flow.save(model_dir)
    return {
        "x": x,
        "flow": flow,
        "scaler": scaler,
        "model_dir": model_dir,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "log_normal": log_normal,
    }


@pytest.fixture(scope="module")
def gaussian_run(tmp_path_factory):
    return _train(tmp_path_factory.mktemp("flowjax_gauss"), log_normal=False)


@pytest.fixture(scope="module")
def lognormal_run(tmp_path_factory):
    return _train(tmp_path_factory.mktemp("flowjax_lognormal"), log_normal=True)


def _core_test_points(log_normal: bool, n: int = 2000) -> np.ndarray:
    """Fresh draws from the truth, restricted to Mahalanobis radius < 2.5."""
    x = _draw(log_normal, n, seed=123)
    d = _to_latent(x, log_normal) - MEAN
    r2 = np.einsum("ij,jk,ik->i", d, np.linalg.inv(COV), d)
    return x[r2 < 2.5**2]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_flowjax_config_defaults_match_paper() -> None:
    cfg = FlowjaxConfig()
    assert cfg.backend == "flowjax"
    assert cfg.flow_type == "coupling_nsf"
    assert cfg.n_transforms == 4
    assert cfg.nn_width == 256
    assert cfg.nn_depth == 3
    assert cfg.num_bins == 8
    assert cfg.tail_bound == 5.0


def test_flowjax_config_selected_by_backend(tmp_path: Path) -> None:
    config = _make_config(tmp_path, log_lambda=False, flow_type="maf_nsf")
    assert isinstance(config.flow, FlowjaxConfig)
    assert config.flow.flow_type == "maf_nsf"


# ---------------------------------------------------------------------------
# Physical-unit density against the analytic truth
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("run_name", ["gaussian_run", "lognormal_run"])
def test_physical_log_prob_matches_analytic(run_name: str, request) -> None:
    run = request.getfixturevalue(run_name)
    prior = load_flowjax_prior(run["model_dir"])
    x = _core_test_points(run["log_normal"])
    lp_flow = np.asarray(prior.log_prob(jnp.asarray(x)))
    lp_true = _true_log_prob(x, run["log_normal"])
    diff = lp_flow - lp_true
    # The Jacobian terms are several nats here (scaler: sum of -log(range),
    # about -2.8 nat for the Gaussian case; log transform: -log(lambda_2),
    # about -6 nat), while a short fit is accurate to ~0.1 nat.
    assert abs(np.median(diff)) < 0.1, np.median(diff)
    assert np.median(np.abs(diff)) < 0.15, np.median(np.abs(diff))
    assert np.percentile(np.abs(diff), 90) < 0.35


@pytest.mark.parametrize("run_name", ["gaussian_run", "lognormal_run"])
def test_samples_reweight_to_truth(run_name: str, request) -> None:
    """Samples and log_prob describe the same density, in physical units.

    Importance weights p_true / q with q = prior.log_prob must turn
    prior.sample draws into the true distribution. This checks the samples'
    units and their consistency with the density, independently of how well
    the short fit matches the tails (the raw sample standard deviations are
    10-20% too large after this short training).
    """
    run = request.getfixturevalue(run_name)
    log_normal = run["log_normal"]
    prior = load_flowjax_prior(run["model_dir"])
    n = 20_000
    s = np.asarray(prior.sample(jax.random.key(0), n), dtype=np.float64)
    assert s.shape == (n, 3)
    if log_normal:
        assert np.all(s[:, 2] > 0)
    log_w = _true_log_prob(s, log_normal) - np.asarray(
        prior.log_prob(jnp.asarray(s, dtype=jnp.float32)), dtype=np.float64
    )
    w = np.exp(log_w - log_w.max())
    w /= w.sum()
    assert 1.0 / np.sum(w**2) > 0.5 * n

    u = _to_latent(s, log_normal)
    mean = w @ u
    std = np.sqrt(w @ (u - mean) ** 2)
    assert np.all(np.abs(mean - MEAN) < 0.03 * STD), (mean - MEAN) / STD
    np.testing.assert_allclose(std, STD, rtol=0.03)


def test_log_prob_outside_log_support_is_minus_inf(lognormal_run) -> None:
    prior = load_flowjax_prior(lognormal_run["model_dir"])
    x = jnp.array([[1.2, 0.8, -1.0], [1.2, 0.8, 0.0]])
    lp = prior.log_prob(x)
    assert np.all(np.isneginf(np.asarray(lp)))


# ---------------------------------------------------------------------------
# Scaler and its Jacobian
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("run_name", ["gaussian_run", "lognormal_run"])
def test_saved_scaler_matches_sklearn_and_no_pickle(run_name: str, request) -> None:
    run = request.getfixturevalue(run_name)
    model_dir = run["model_dir"]
    assert not (model_dir / "scaler.gz").exists()
    assert (model_dir / WEIGHTS_FILE).exists()

    meta = json.loads((model_dir / "model_kwargs.json").read_text())
    assert meta["backend"] == "flowjax"
    assert meta["parameter_names"] == NAMES
    scaler = run["scaler"]
    np.testing.assert_array_equal(meta["scaler"]["min"], scaler.min_)
    np.testing.assert_array_equal(meta["scaler"]["scale"], scaler.scale_)
    np.testing.assert_array_equal(meta["scaler"]["data_min"], scaler.data_min_)
    np.testing.assert_array_equal(meta["scaler"]["data_max"], scaler.data_max_)
    expected_log = ["lambda_2"] if run["log_normal"] else []
    assert meta["log_transform"] == expected_log

    prior = load_flowjax_prior(model_dir)
    assert prior.names == tuple(NAMES)
    np.testing.assert_array_equal(meta["bounds"]["min"], run["x"].min(0))
    np.testing.assert_array_equal(meta["bounds"]["max"], run["x"].max(0))
    # The prior holds them in JAX's default float dtype (float32 here).
    np.testing.assert_allclose(prior.bounds[:, 0], run["x"].min(0), rtol=1e-7)
    np.testing.assert_allclose(prior.bounds[:, 1], run["x"].max(0), rtol=1e-7)


@pytest.mark.parametrize("run_name", ["gaussian_run", "lognormal_run"])
def test_jacobian_explicit_change_of_variables(run_name: str, request) -> None:
    r"""log p_X(x) = log p_Y(T(x)) + log|det dT/dx| with T built from the JSON."""
    run = request.getfixturevalue(run_name)
    model_dir = run["model_dir"]
    meta = json.loads((model_dir / "model_kwargs.json").read_text())
    scale = jnp.asarray(meta["scaler"]["scale"], dtype=jnp.float32)
    shift = jnp.asarray(meta["scaler"]["min"], dtype=jnp.float32)
    log_dims = [NAMES.index(n) for n in meta["log_transform"]]

    def T(x):
        if log_dims:
            x = x.at[jnp.array(log_dims)].set(jnp.log(x[jnp.array(log_dims)]))
        return x * scale + shift

    prior = load_flowjax_prior(model_dir)
    flow: FlowjaxFlow = run["flow"]
    x = _core_test_points(run["log_normal"], n=50).astype(np.float32)
    for xi in x:
        xi = jnp.asarray(xi)
        _, logdet = jnp.linalg.slogdet(jax.jacfwd(T)(xi))
        lp_scaled = flow.log_prob(np.asarray(T(xi))[None, :])[0]
        expected = lp_scaled + float(logdet)
        np.testing.assert_allclose(float(prior.log_prob(xi)), expected, atol=2e-4)

    # Cross-check with the package's own Jacobian helper used by the other
    # backends (only valid without a log transform).
    if not log_dims:
        corr = FlowBase.compute_log_jacobian_correction(run["scaler"])
        y = run["scaler"].transform(x).astype(np.float32)
        np.testing.assert_allclose(
            np.asarray(prior.log_prob(jnp.asarray(x))),
            flow.log_prob(y) + corr,
            atol=2e-4,
        )


# ---------------------------------------------------------------------------
# Save / load, jit, vmap, dispatch
# ---------------------------------------------------------------------------


def test_round_trip_identical(gaussian_run) -> None:
    flow: FlowjaxFlow = gaussian_run["flow"]
    in_memory = flow.as_prior()
    loaded = load_flowjax_prior(gaussian_run["model_dir"])
    x = jnp.asarray(_core_test_points(False, n=200))
    np.testing.assert_array_equal(
        np.asarray(in_memory.log_prob(x)), np.asarray(loaded.log_prob(x))
    )
    key = jax.random.key(42)
    np.testing.assert_array_equal(
        np.asarray(in_memory.sample(key, 100)), np.asarray(loaded.sample(key, 100))
    )
    z = jax.random.normal(key, (10, 3))
    np.testing.assert_array_equal(
        np.asarray(in_memory.latent_to_physical(z)),
        np.asarray(loaded.latent_to_physical(z)),
    )


def test_jit_and_vmap(gaussian_run) -> None:
    prior = load_flowjax_prior(gaussian_run["model_dir"])
    x = jnp.asarray(_core_test_points(False, n=100))
    eager = np.asarray(prior.log_prob(x))
    # float32: jit fuses operations differently, so agreement is ~1e-5 nat.
    single = np.array([float(prior.log_prob(xi)) for xi in x[:5]])
    np.testing.assert_allclose(single, eager[:5], atol=1e-4)

    np.testing.assert_allclose(jax.jit(prior.log_prob)(x), eager, atol=1e-4)
    np.testing.assert_allclose(jax.vmap(prior.log_prob)(x), eager, atol=1e-4)
    np.testing.assert_allclose(jax.jit(jax.vmap(prior.log_prob))(x), eager, atol=1e-4)

    # The prior is a pytree and can be passed through eqx.filter_jit as an
    # argument (flowjax modules hold functions, so plain jax.jit needs a closure).
    @eqx.filter_jit
    def f(p: FlowjaxPrior, xx):
        return jax.vmap(p.log_prob)(xx)

    np.testing.assert_allclose(f(prior, x), eager, atol=1e-4)

    # Gradients are finite (needed by gradient-based samplers).
    g = jax.grad(lambda xx: prior.log_prob(xx))(x[0])
    assert np.all(np.isfinite(np.asarray(g)))

    s = jax.jit(lambda k: prior.sample(k, 10))(jax.random.key(0))
    assert s.shape == (10, 3)


def test_load_from_dir_dispatches_to_flowjax(gaussian_run) -> None:
    loaded = FlowBase.load_from_dir(gaussian_run["model_dir"])
    assert isinstance(loaded, FlowjaxFlow)
    y = np.random.default_rng(0).uniform(0.2, 0.8, size=(20, 3)).astype(np.float32)
    np.testing.assert_array_equal(loaded.log_prob(y), gaussian_run["flow"].log_prob(y))
    assert loaded.sample(5).shape == (5, 3)
    assert loaded.latent_to_data(np.zeros((4, 3), np.float32)).shape == (4, 3)


def test_training_losses_decrease(gaussian_run) -> None:
    tl, vl = gaussian_run["train_losses"], gaussian_run["val_losses"]
    assert len(tl) == len(vl) > 0
    assert np.all(np.isfinite(tl)) and np.all(np.isfinite(vl))
    assert min(vl) < vl[0] - 0.5


def test_maf_variant_trains_and_round_trips(tmp_path: Path) -> None:
    run = _train(tmp_path, log_normal=False, flow_type="maf_nsf")
    prior = load_flowjax_prior(run["model_dir"])
    x = jnp.asarray(_core_test_points(False, n=200))
    np.testing.assert_array_equal(
        np.asarray(run["flow"].as_prior().log_prob(x)), np.asarray(prior.log_prob(x))
    )
    diff = np.asarray(prior.log_prob(x)) - _true_log_prob(np.asarray(x), False)
    assert np.median(np.abs(diff)) < 0.3


def test_early_stopping_flowjax(tmp_path: Path) -> None:
    x = _draw(False, 1000, seed=0)
    data = {name: x[:, i] for i, name in enumerate(NAMES)}
    config = _make_config(tmp_path, log_lambda=False)
    config.training.num_epochs = 500
    config.training.max_patience = 1
    config.training.learning_rate = 1.0  # diverges quickly, so val loss stalls
    _, train_losses, _, _ = FlowTrainer(config).train(
        data, NAMES, output_dir=tmp_path / "m"
    )
    assert len(train_losses) < 500


# ---------------------------------------------------------------------------
# Subprocess tests: float64 and JAX-only imports
# ---------------------------------------------------------------------------


def test_float64_load(gaussian_run, tmp_path: Path) -> None:
    prior32 = load_flowjax_prior(gaussian_run["model_dir"])
    x = _core_test_points(False, n=50)
    lp32 = np.asarray(prior32.log_prob(jnp.asarray(x, dtype=jnp.float32)))
    np.save(tmp_path / "x.npy", x)
    np.save(tmp_path / "lp32.npy", lp32)
    code = f"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from neural_priors_gym.flows.flowjax_prior import load_flowjax_prior
prior = load_flowjax_prior({str(gaussian_run["model_dir"])!r})
x = jnp.asarray(np.load({str(tmp_path / "x.npy")!r}))
assert x.dtype == jnp.float64
lp = jax.jit(jax.vmap(prior.log_prob))(x)
assert lp.dtype == jnp.float64, lp.dtype
s = prior.sample(jax.random.key(0), 10)
assert s.dtype == jnp.float64, s.dtype
leaves = jax.tree_util.tree_leaves(prior)
leaves = [l for l in leaves if hasattr(l, "dtype")]
leaves = [l for l in leaves if jnp.issubdtype(l.dtype, jnp.floating)]
assert all(l.dtype == jnp.float64 for l in leaves)
lp32 = np.load({str(tmp_path / "lp32.npy")!r})
np.testing.assert_allclose(np.asarray(lp), lp32, atol=5e-4)
# float64 resolves differences far below float32 resolution in chirp mass.
dx = jnp.zeros(3).at[0].set(1e-9)
assert prior.log_prob(x[0] + dx) != prior.log_prob(x[0])
print("OK")
"""
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    assert "OK" in out.stdout


def test_loader_imports_only_jax_stack() -> None:
    code = """
import sys
import neural_priors_gym.flows.flowjax_prior as m
heavy = ["torch", "sklearn", "bilby", "glasflow", "zuko", "jesterTOV",
         "pydantic", "yaml", "matplotlib", "corner", "joblib"]
loaded = [h for h in heavy if h in sys.modules]
assert not loaded, loaded
assert hasattr(m, "load_flowjax_prior")
print("OK")
"""
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    assert "OK" in out.stdout


def test_flowjax_major_version_mismatch_raises(gaussian_run, tmp_path: Path) -> None:
    import shutil

    model_dir = tmp_path / "model"
    shutil.copytree(gaussian_run["model_dir"], model_dir)
    meta = json.loads((model_dir / "model_kwargs.json").read_text())
    meta["versions"]["flowjax"] = "999.0.0"
    (model_dir / "model_kwargs.json").write_text(json.dumps(meta))
    with pytest.raises(RuntimeError, match="flowjax 999.0.0"):
        load_flowjax_prior(model_dir)


def test_affine_metadata_matches_both_scalers():
    """The JSON affine map y = x * scale + min reproduces sklearn's transform for the min-max and the standard scaler."""
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    from neural_priors_gym.training.trainer import _affine

    rng = np.random.default_rng(0)
    x = rng.normal([1.2, 0.9, 500.0, 900.0], [0.05, 0.05, 200.0, 400.0], size=(5000, 4))
    for scaler in (MinMaxScaler(), StandardScaler()):
        scaler.fit(x)
        a = _affine(scaler, x)
        np.testing.assert_allclose(x * np.asarray(a["scale"]) + np.asarray(a["min"]), scaler.transform(x), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(a["data_min"], x.min(axis=0))
        np.testing.assert_allclose(a["data_max"], x.max(axis=0))
