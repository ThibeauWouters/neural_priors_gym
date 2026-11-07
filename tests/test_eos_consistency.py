"""Tests that each training sample uses a single EOS, read from npz or jester h5.

arXiv:2511.22987 draws one EOS per training sample, which fixes both M_TOV
(the upper bound of the NS masses, and the lower bound of the BH mass for
NSBH) and Lambda(m). The synthetic EOS set below makes the EOS of a sample
identifiable from (m, Lambda): Lambda_k(m) = c_k exp(-5 (m - 1)) with a
distinct c_k per EOS and a distinct M_TOV per EOS.
"""

import time
from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy import stats

from neural_priors_gym.config.schema import TrainingConfig
from neural_priors_gym.config.schemas.lambdas import LambdaConfig
from neural_priors_gym.config.schemas.masses import (
    DoubleGaussianMassConfig,
    GaussianMassConfig,
)
from neural_priors_gym.convert import convert
from neural_priors_gym.data.generator import TrainingDataGenerator
from neural_priors_gym.data.lambdas.interpolator import EOSLambdaInterpolator
from neural_priors_gym.data.masses import (
    DoubleGaussianMassGenerator,
    GaussianMassGenerator,
)

MTOV = np.array([1.9, 2.1, 2.3, 2.5])
C_K = np.array([100.0, 200.0, 400.0, 800.0])
N_GRID = 100


def _eos_arrays() -> tuple[np.ndarray, np.ndarray]:
    """Stable-branch grids as written by jester: linspace(m_min_grid, M_TOV)."""
    masses = np.stack([np.linspace(0.3, m, N_GRID) for m in MTOV])
    lambdas = C_K[:, None] * np.exp(-5.0 * (masses - 1.0))
    return masses, lambdas


def _write_jester_h5(path: Path, add_invalid: bool = True) -> None:
    """Write a file in the layout of jester's InferenceResult.save (branch opt)."""
    masses, lambdas = _eos_arrays()
    radii = 12.0 - 0.5 * masses
    if add_invalid:
        # jester writes zeros (nan_to_num) for EOS that fail the TOV solve.
        masses = np.vstack([masses, np.zeros(N_GRID)])
        lambdas = np.vstack([lambdas, np.zeros(N_GRID)])
        radii = np.vstack([radii, np.zeros(N_GRID)])
    n_eos = len(masses)
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        post = f.create_group("posterior")
        derived = post.create_group("derived_eos")
        derived.create_dataset("masses_EOS", data=masses)
        derived.create_dataset("radii_EOS", data=radii)
        derived.create_dataset("Lambdas_EOS", data=lambdas)
        for key in ("n", "p", "e", "cs2"):
            derived.create_dataset(key, data=rng.uniform(0.1, 1.0, (n_eos, 200)))
        derived.create_dataset("n_TOV", data=rng.uniform(0.5, 1.2, n_eos))
        params = post.create_group("parameters")
        for key in ("E_sym", "L_sym", "K_sat"):
            params.create_dataset(key, data=rng.normal(size=n_eos))
        post.create_group("sampler_specific")
        post.create_dataset("log_prob", data=rng.normal(size=n_eos))
        meta = f.create_group("metadata")
        meta.create_dataset("config", data="{}")


@pytest.fixture
def eos_npz(tmp_path: Path) -> Path:
    masses, lambdas = _eos_arrays()
    path = tmp_path / "eos.npz"
    np.savez(path, masses_EOS=masses, Lambdas_EOS=lambdas)
    return path


@pytest.fixture
def jester_h5(tmp_path: Path) -> Path:
    path = tmp_path / "results.h5"
    _write_jester_h5(path)
    return path


def _identify_eos(m: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Return the EOS index k with Lambda = c_k exp(-5 (m - 1)), or -1.

    Linear interpolation of the 100-point grid is accurate to ~1.5e-3, while
    the c_k differ by factors of 2.
    """
    c = lam / np.exp(-5.0 * (m - 1.0))
    k = np.argmin(np.abs(np.log(c[:, None] / C_K[None, :])), axis=1)
    ok = np.isclose(c, C_K[k], rtol=3e-3)
    return np.where(ok, k, -1)


MASS_MODELS = {
    "uniform": {"type": "uniform", "m_min": 1.0},
    "gaussian": {"type": "gaussian", "m_min": 1.0, "mean": 1.33, "std": 0.09},
    "double_gaussian": {
        "type": "double_gaussian",
        "m_min": 1.0,
        "mean_1": 1.34,
        "std_1": 0.07,
        "mean_2": 1.80,
        "std_2": 0.21,
        "weight": 0.65,
    },
}


def _generate(source_type: str, masses: dict, lambdas: dict, n: int = 4000) -> dict:
    config = TrainingConfig(
        source_type=source_type,  # type: ignore[arg-type]
        m_max_bh=5.0,
        masses=masses,  # type: ignore[arg-type]
        lambdas=lambdas,  # type: ignore[arg-type]
        flow={"backend": "flowjax"},  # type: ignore[arg-type]
        training={  # type: ignore[arg-type]
            "parameter_names": ["chirp_mass_source", "mass_ratio", "lambda_2"],
            "n_samples": n,
        },
    )
    np.random.seed(0)
    return TrainingDataGenerator.from_config(config).generate()


@pytest.mark.parametrize("model", list(MASS_MODELS))
def test_bns_sample_uses_one_eos(model: str, eos_npz: Path) -> None:
    data = _generate("bns", MASS_MODELS[model], {"eos_path": str(eos_npz)})
    k1 = _identify_eos(data["mass_1_source"], data["lambda_1"])
    k2 = _identify_eos(data["mass_2_source"], data["lambda_2"])
    assert np.all(k1 >= 0) and np.all(k2 >= 0)
    # Both stars use the same EOS ...
    np.testing.assert_array_equal(k1, k2)
    # ... and that EOS's M_TOV bounds the masses (no clamped Lambda values).
    assert np.all(data["mass_1_source"] <= MTOV[k1])
    assert np.all(data["mass_2_source"] >= 1.0)
    # Every EOS is used.
    assert set(np.unique(k1)) == set(range(len(MTOV)))


@pytest.mark.parametrize("model", list(MASS_MODELS))
def test_nsbh_sample_uses_one_eos(model: str, eos_npz: Path) -> None:
    data = _generate("nsbh", MASS_MODELS[model], {"eos_path": str(eos_npz)})
    k = _identify_eos(data["mass_2_source"], data["lambda_2"])
    assert np.all(k >= 0)
    assert np.all(data["lambda_1"] == 0.0)
    assert np.all(data["mass_2_source"] <= MTOV[k])
    # BH mass uniform in [M_TOV, 5] of the same EOS.
    assert np.all(data["mass_1_source"] >= MTOV[k])
    assert np.all(data["mass_1_source"] <= 5.0)
    for j in range(len(MTOV)):
        m1 = data["mass_1_source"][k == j]
        assert stats.kstest(m1, stats.uniform(MTOV[j], 5.0 - MTOV[j]).cdf).pvalue > 1e-3


# ---------------------------------------------------------------------------
# jester h5 layout
# ---------------------------------------------------------------------------


def test_convert_reads_jester_layout(jester_h5: Path, tmp_path: Path) -> None:
    out = tmp_path / "eos_samples.npz"
    convert(jester_h5, out)
    npz = np.load(out)
    masses, lambdas = _eos_arrays()
    # convert() copies the arrays; invalid rows are removed by the interpolator.
    np.testing.assert_array_equal(npz["masses_EOS"][:-1], masses)
    np.testing.assert_array_equal(npz["Lambdas_EOS"][:-1], lambdas)
    assert npz["masses_EOS"].shape == (len(MTOV) + 1, N_GRID)


def test_interpolator_from_jester_h5_and_npz_agree(
    jester_h5: Path, tmp_path: Path
) -> None:
    out = tmp_path / "eos_samples.npz"
    convert(jester_h5, out)
    from_h5 = EOSLambdaInterpolator(LambdaConfig(jester_path=str(jester_h5)))
    from_npz = EOSLambdaInterpolator(LambdaConfig(eos_path=str(out)))
    for interp in (from_h5, from_npz):
        # The all-zero (failed) EOS is dropped; M_TOV is the end of the
        # stable branch.
        np.testing.assert_array_equal(interp.mtov_array, MTOV)
        m = np.array([[1.4], [1.0]])
        lam = interp.interpolate(np.repeat(m, 1, axis=1), eos_indices=np.array([2, 0]))
        np.testing.assert_allclose(
            lam[:, 0],
            [C_K[2] * np.exp(-5 * 0.4), C_K[0]],
            rtol=2e-3,  # linear interpolation error on this 100-point grid: 1.5e-3
        )


def test_interpolator_drops_unphysical_eos(tmp_path: Path) -> None:
    masses, lambdas = _eos_arrays()
    bad_nan = masses[0].copy()
    bad_nan[5] = np.nan
    bad_nonmono = masses[1][::-1].copy()  # decreasing masses
    masses = np.vstack([masses, bad_nan, bad_nonmono])
    lambdas = np.vstack([lambdas, lambdas[0], lambdas[1]])
    path = tmp_path / "eos.npz"
    np.savez(path, masses_EOS=masses, Lambdas_EOS=lambdas)
    interp = EOSLambdaInterpolator(LambdaConfig(eos_path=str(path)))
    np.testing.assert_array_equal(interp.mtov_array, MTOV)


def test_generation_from_jester_h5(jester_h5: Path) -> None:
    data = _generate("bns", MASS_MODELS["uniform"], {"jester_path": str(jester_h5)})
    k1 = _identify_eos(data["mass_1_source"], data["lambda_1"])
    assert np.all(k1 >= 0)
    assert np.all(data["mass_1_source"] <= MTOV[k1])


# ---------------------------------------------------------------------------
# Vectorised truncated-Gaussian mass draws
# ---------------------------------------------------------------------------


def _tn(mean: float, std: float, lo: float, hi: float):
    return stats.truncnorm((lo - mean) / std, (hi - mean) / std, loc=mean, scale=std)


def _single_cdf(model: str, m_min: float, mtov: float):
    if model == "gaussian":
        return _tn(1.33, 0.09, m_min, mtov).cdf
    # Mixture of two components, each truncated to [m_min, M_TOV].
    c1, c2 = _tn(1.34, 0.07, m_min, mtov), _tn(1.80, 0.21, m_min, mtov)
    return lambda m: 0.65 * c1.cdf(m) + 0.35 * c2.cdf(m)


def _generator(model: str):
    if model == "gaussian":
        return GaussianMassGenerator(GaussianMassConfig(m_min=1.0, mean=1.33, std=0.09))
    return DoubleGaussianMassGenerator(
        DoubleGaussianMassConfig(
            m_min=1.0, mean_1=1.34, std_1=0.07, mean_2=1.80, std_2=0.21, weight=0.65
        )
    )


@pytest.mark.parametrize("model", ["gaussian", "double_gaussian"])
def test_truncated_gaussian_draws_match_exact_cdf(model: str) -> None:
    gen = _generator(model)
    np.random.seed(1)
    n = 20_000
    for k, mtov in enumerate(MTOV):
        idx = np.full(n, k)
        F = _single_cdf(model, 1.0, mtov)
        ns = gen.generate_ns_mass(n, MTOV, eos_indices=idx)
        assert np.all((ns >= 1.0) & (ns <= mtov))
        assert stats.kstest(ns, F).pvalue > 1e-3

        pair = gen.generate(n, MTOV, eos_indices=idx)
        m1, m2 = pair[:, 0], pair[:, 1]
        assert np.all(m1 >= m2)
        assert np.all((m2 >= 1.0) & (m1 <= mtov))
        # Random pairing: m1 = max, m2 = min of two independent draws.
        assert stats.kstest(m1, lambda m: F(m) ** 2).pvalue > 1e-3
        assert stats.kstest(m2, lambda m: 1 - (1 - F(m)) ** 2).pvalue > 1e-3


@pytest.mark.parametrize("model", ["gaussian", "double_gaussian"])
def test_truncated_gaussian_draws_are_fast(model: str) -> None:
    gen = _generator(model)
    t0 = time.perf_counter()
    out = gen.generate(200_000, MTOV)
    assert out.shape == (200_000, 2)
    assert time.perf_counter() - t0 < 5.0
