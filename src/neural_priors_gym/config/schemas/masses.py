"""Configuration schemas for mass generator types."""

from typing import Annotated, Literal, Union

from pydantic import Discriminator

from ._base import NeuralPriorBaseModel


class UniformMassConfig(NeuralPriorBaseModel):
    """Uniform mass distribution between m_min and MTOV (per EOS sample)."""

    type: Literal["uniform"] = "uniform"
    m_min: float


class GaussianMassConfig(NeuralPriorBaseModel):
    """Single Gaussian mass distribution truncated at m_min and MTOV."""

    type: Literal["gaussian"] = "gaussian"
    m_min: float
    mean: float
    std: float
    max_attempts: int = 1_000_000


class DoubleGaussianMassConfig(NeuralPriorBaseModel):
    """Mixture of two Gaussians for the mass distribution."""

    type: Literal["double_gaussian"] = "double_gaussian"
    m_min: float
    mean_1: float
    std_1: float
    mean_2: float
    std_2: float
    weight: float
    max_attempts: int = 1_000_000


class BilbyMassConfig(NeuralPriorBaseModel):
    """Mass distribution sampled from a bilby prior file.

    The prior file must define parameters from which bilby's
    generate_all_bns_parameters can derive mass_1_source and mass_2_source
    (e.g. chirp_mass, mass_ratio, luminosity_distance).
    """

    type: Literal["bilby"] = "bilby"
    prior_file: str


class ChirpMassRatioRedshiftMassConfig(NeuralPriorBaseModel):
    """Detector-frame chirp mass and mass ratio with an explicit redshift prior.

    Chirp mass and mass ratio are drawn from bilby's
    ``UniformInComponentsChirpMass`` and ``UniformInComponentsMassRatio``
    priors, which together give a joint distribution uniform in the
    detector-frame component masses. Redshift is drawn independently from a
    bilby cosmological prior and used to convert the detector-frame
    component masses to source frame via mass_source = mass / (1 + z).
    """

    type: Literal["chirp_mass_q_redshift"] = "chirp_mass_q_redshift"
    mc_min: float
    mc_max: float
    q_min: float
    q_max: float
    z_min: float
    z_max: float
    redshift_prior: Literal["uniform_source_frame", "uniform_comoving_volume"] = (
        "uniform_source_frame"
    )
    cosmology: str | None = None


class ChirpMassRatioGaussianRedshiftMassConfig(NeuralPriorBaseModel):
    """Detector-frame chirp mass and mass ratio with a fixed Gaussian redshift.

    Chirp mass and mass ratio are drawn from bilby's
    ``UniformInComponentsChirpMass`` and ``UniformInComponentsMassRatio``
    priors, which together give a joint distribution uniform in the
    detector-frame component masses. Redshift is drawn from a Gaussian
    (truncated at zero) centered on ``z_mean`` with width ``z_std``, and
    used to convert the detector-frame component masses to source frame via
    mass_source = mass / (1 + z).

    This represents a BNS with an electromagnetic counterpart (e.g. a
    kilonova/GRB host galaxy), where the redshift is known from the
    counterpart to good precision rather than sampled from a broad
    population prior.
    """

    type: Literal["chirp_mass_q_gaussian_redshift"] = "chirp_mass_q_gaussian_redshift"
    mc_min: float
    mc_max: float
    q_min: float
    q_max: float
    z_mean: float
    z_std: float


MassConfig = Annotated[
    Union[
        UniformMassConfig,
        GaussianMassConfig,
        DoubleGaussianMassConfig,
        BilbyMassConfig,
        ChirpMassRatioRedshiftMassConfig,
        ChirpMassRatioGaussianRedshiftMassConfig,
    ],
    Discriminator("type"),
]
