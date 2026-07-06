"""Tests for mass generator classes."""

import numpy as np
import pytest
from pathlib import Path

from neural_priors_gym.config.schemas.masses import (
    BilbyMassConfig,
    ChirpMassRatioGaussianRedshiftMassConfig,
    ChirpMassRatioRedshiftMassConfig,
    DoubleGaussianMassConfig,
    GaussianMassConfig,
    UniformMassConfig,
)
from neural_priors_gym.data.masses import (
    BilbyPriorMassGenerator,
    ChirpMassRatioGaussianRedshiftMassGenerator,
    ChirpMassRatioRedshiftMassGenerator,
    DoubleGaussianMassGenerator,
    GaussianMassGenerator,
    UniformMassGenerator,
)


@pytest.fixture
def masses_EOS() -> np.ndarray:
    """Small synthetic EOS mass grid."""
    rng = np.random.default_rng(1)
    n_eos, n_pts = 10, 50
    arr = np.zeros((n_eos, n_pts))
    for i in range(n_eos):
        arr[i] = np.linspace(0.5, 2.0 + rng.uniform(0, 0.3), n_pts)
    return arr


@pytest.fixture
def uniform_config() -> UniformMassConfig:
    return UniformMassConfig(m_min=1.0)


@pytest.fixture
def gaussian_config() -> GaussianMassConfig:
    return GaussianMassConfig(m_min=1.0, mean=1.33, std=0.09)


@pytest.fixture
def double_gaussian_config() -> DoubleGaussianMassConfig:
    return DoubleGaussianMassConfig(
        m_min=1.0,
        mean_1=1.34,
        std_1=0.07,
        mean_2=1.80,
        std_2=0.21,
        weight=0.65,
    )


def test_uniform_generator_shape(
    uniform_config: UniformMassConfig, masses_EOS: np.ndarray
) -> None:
    gen = UniformMassGenerator(uniform_config)
    result = gen.generate(100, masses_EOS.max(axis=1))
    assert result.shape == (100, 2)


def test_uniform_generator_m1_ge_m2(
    uniform_config: UniformMassConfig, masses_EOS: np.ndarray
) -> None:
    gen = UniformMassGenerator(uniform_config)
    result = gen.generate(200, masses_EOS.max(axis=1))
    assert np.all(result[:, 0] >= result[:, 1])


def test_uniform_generator_within_bounds(
    uniform_config: UniformMassConfig, masses_EOS: np.ndarray
) -> None:
    gen = UniformMassGenerator(uniform_config)
    mtov = masses_EOS.max(axis=1)
    result = gen.generate(200, mtov)
    assert np.all(result[:, 1] >= 1.0)
    assert np.all(result[:, 0] <= mtov.max() + 1e-9)


def test_gaussian_generator_shape(
    gaussian_config: GaussianMassConfig, masses_EOS: np.ndarray
) -> None:
    gen = GaussianMassGenerator(gaussian_config)
    result = gen.generate(50, masses_EOS.max(axis=1))
    assert result.shape == (50, 2)


def test_gaussian_generator_m1_ge_m2(
    gaussian_config: GaussianMassConfig, masses_EOS: np.ndarray
) -> None:
    gen = GaussianMassGenerator(gaussian_config)
    result = gen.generate(50, masses_EOS.max(axis=1))
    assert np.all(result[:, 0] >= result[:, 1])


def test_double_gaussian_generator_shape(
    double_gaussian_config: DoubleGaussianMassConfig, masses_EOS: np.ndarray
) -> None:
    gen = DoubleGaussianMassGenerator(double_gaussian_config)
    result = gen.generate(50, masses_EOS.max(axis=1))
    assert result.shape == (50, 2)


def test_double_gaussian_generator_m1_ge_m2(
    double_gaussian_config: DoubleGaussianMassConfig, masses_EOS: np.ndarray
) -> None:
    gen = DoubleGaussianMassGenerator(double_gaussian_config)
    result = gen.generate(50, masses_EOS.max(axis=1))
    assert np.all(result[:, 0] >= result[:, 1])


def test_bilby_generator(bilby_prior_file: Path, masses_EOS: np.ndarray) -> None:
    config = BilbyMassConfig(prior_file=str(bilby_prior_file))
    gen = BilbyPriorMassGenerator(config)
    result = gen.generate(50, masses_EOS.max(axis=1))
    assert result.shape == (50, 2)
    assert np.all(np.isfinite(result))


def test_bilby_generator_missing_file(masses_EOS: np.ndarray) -> None:
    with pytest.raises(FileNotFoundError):
        config = BilbyMassConfig(prior_file="/nonexistent/path.prior")
        BilbyPriorMassGenerator(config)


@pytest.fixture
def chirp_mass_ratio_redshift_config() -> ChirpMassRatioRedshiftMassConfig:
    return ChirpMassRatioRedshiftMassConfig(
        mc_min=1.0,
        mc_max=2.0,
        q_min=0.5,
        q_max=1.0,
        z_min=0.001,
        z_max=0.1,
    )


def test_chirp_mass_ratio_redshift_generator_shape(
    chirp_mass_ratio_redshift_config: ChirpMassRatioRedshiftMassConfig,
    masses_EOS: np.ndarray,
) -> None:
    gen = ChirpMassRatioRedshiftMassGenerator(chirp_mass_ratio_redshift_config)
    result = gen.generate(200, masses_EOS.max(axis=1))
    assert result.shape == (200, 2)
    assert np.all(np.isfinite(result))


def test_chirp_mass_ratio_redshift_generator_m1_ge_m2(
    chirp_mass_ratio_redshift_config: ChirpMassRatioRedshiftMassConfig,
    masses_EOS: np.ndarray,
) -> None:
    gen = ChirpMassRatioRedshiftMassGenerator(chirp_mass_ratio_redshift_config)
    result = gen.generate(200, masses_EOS.max(axis=1))
    assert np.all(result[:, 0] >= result[:, 1])


def test_chirp_mass_ratio_redshift_generator_extra_quantities(
    chirp_mass_ratio_redshift_config: ChirpMassRatioRedshiftMassConfig,
    masses_EOS: np.ndarray,
) -> None:
    gen = ChirpMassRatioRedshiftMassGenerator(chirp_mass_ratio_redshift_config)
    assert gen.extra_quantities() == {}

    gen.generate(200, masses_EOS.max(axis=1))
    extras = gen.extra_quantities()
    assert set(extras) == {"redshift", "chirp_mass"}
    redshift = extras["redshift"]
    assert redshift.shape == (200,)
    assert np.all(redshift >= chirp_mass_ratio_redshift_config.z_min)
    assert np.all(redshift <= chirp_mass_ratio_redshift_config.z_max)
    chirp_mass = extras["chirp_mass"]
    assert chirp_mass.shape == (200,)
    assert np.all(chirp_mass >= chirp_mass_ratio_redshift_config.mc_min)
    assert np.all(chirp_mass <= chirp_mass_ratio_redshift_config.mc_max)


def test_chirp_mass_ratio_redshift_generator_source_frame_below_detector_frame(
    chirp_mass_ratio_redshift_config: ChirpMassRatioRedshiftMassConfig,
    masses_EOS: np.ndarray,
) -> None:
    """Source-frame masses must be smaller than detector-frame masses (z > 0)."""
    gen = ChirpMassRatioRedshiftMassGenerator(chirp_mass_ratio_redshift_config)
    result = gen.generate(200, masses_EOS.max(axis=1))
    redshift = gen.extra_quantities()["redshift"]
    total_mass_source = result[:, 0] + result[:, 1]
    total_mass_detector = total_mass_source * (1.0 + redshift)
    assert np.all(total_mass_detector > total_mass_source)


def test_chirp_mass_ratio_redshift_generator_ns_mass(
    chirp_mass_ratio_redshift_config: ChirpMassRatioRedshiftMassConfig,
    masses_EOS: np.ndarray,
) -> None:
    gen = ChirpMassRatioRedshiftMassGenerator(chirp_mass_ratio_redshift_config)
    result = gen.generate_ns_mass(100, masses_EOS.max(axis=1))
    assert result.shape == (100,)
    assert np.all(np.isfinite(result))


@pytest.fixture
def chirp_mass_ratio_gaussian_redshift_config() -> (
    ChirpMassRatioGaussianRedshiftMassConfig
):
    return ChirpMassRatioGaussianRedshiftMassConfig(
        mc_min=1.0,
        mc_max=2.0,
        q_min=0.5,
        q_max=1.0,
        z_mean=0.02,
        z_std=0.005,
    )


def test_chirp_mass_ratio_gaussian_redshift_generator_shape(
    chirp_mass_ratio_gaussian_redshift_config: ChirpMassRatioGaussianRedshiftMassConfig,
    masses_EOS: np.ndarray,
) -> None:
    gen = ChirpMassRatioGaussianRedshiftMassGenerator(
        chirp_mass_ratio_gaussian_redshift_config
    )
    result = gen.generate(200, masses_EOS.max(axis=1))
    assert result.shape == (200, 2)
    assert np.all(np.isfinite(result))


def test_chirp_mass_ratio_gaussian_redshift_generator_m1_ge_m2(
    chirp_mass_ratio_gaussian_redshift_config: ChirpMassRatioGaussianRedshiftMassConfig,
    masses_EOS: np.ndarray,
) -> None:
    gen = ChirpMassRatioGaussianRedshiftMassGenerator(
        chirp_mass_ratio_gaussian_redshift_config
    )
    result = gen.generate(200, masses_EOS.max(axis=1))
    assert np.all(result[:, 0] >= result[:, 1])


def test_chirp_mass_ratio_gaussian_redshift_generator_extra_quantities(
    chirp_mass_ratio_gaussian_redshift_config: ChirpMassRatioGaussianRedshiftMassConfig,
    masses_EOS: np.ndarray,
) -> None:
    gen = ChirpMassRatioGaussianRedshiftMassGenerator(
        chirp_mass_ratio_gaussian_redshift_config
    )
    assert gen.extra_quantities() == {}

    gen.generate(2000, masses_EOS.max(axis=1))
    extras = gen.extra_quantities()
    assert set(extras) == {"redshift", "chirp_mass"}
    redshift = extras["redshift"]
    assert redshift.shape == (2000,)
    assert np.all(redshift >= 0.0)
    chirp_mass = extras["chirp_mass"]
    assert chirp_mass.shape == (2000,)
    assert np.all(chirp_mass >= chirp_mass_ratio_gaussian_redshift_config.mc_min)
    assert np.all(chirp_mass <= chirp_mass_ratio_gaussian_redshift_config.mc_max)
    # Sanity check the sampled redshift is centered near z_mean, within a
    # generous multiple of z_std given the small sample size.
    assert abs(redshift.mean() - chirp_mass_ratio_gaussian_redshift_config.z_mean) < (
        5 * chirp_mass_ratio_gaussian_redshift_config.z_std
    )


def test_chirp_mass_ratio_gaussian_redshift_generator_ns_mass(
    chirp_mass_ratio_gaussian_redshift_config: ChirpMassRatioGaussianRedshiftMassConfig,
    masses_EOS: np.ndarray,
) -> None:
    gen = ChirpMassRatioGaussianRedshiftMassGenerator(
        chirp_mass_ratio_gaussian_redshift_config
    )
    result = gen.generate_ns_mass(100, masses_EOS.max(axis=1))
    assert result.shape == (100,)
    assert np.all(np.isfinite(result))
