"""Orchestrates mass and Lambda generation to produce training data."""

from pathlib import Path

import numpy as np
from bilby.gw.conversion import (
    component_masses_to_chirp_mass,
    component_masses_to_mass_ratio,
    lambda_1_lambda_2_to_delta_lambda_tilde,
    lambda_1_lambda_2_to_lambda_tilde,
)

from neural_priors_gym.config.schema import TrainingConfig
from neural_priors_gym.config.schemas.masses import (
    BilbyMassConfig,
    ChirpMassRatioGaussianRedshiftMassConfig,
    ChirpMassRatioRedshiftMassConfig,
    DoubleGaussianMassConfig,
    GaussianMassConfig,
    UniformMassConfig,
)
from neural_priors_gym.data.lambdas.interpolator import EOSLambdaInterpolator
from neural_priors_gym.data.masses.base import MassGenerator
from neural_priors_gym.data.masses.bilby_prior import BilbyPriorMassGenerator
from neural_priors_gym.data.masses.chirp_mass_ratio_gaussian_redshift import (
    ChirpMassRatioGaussianRedshiftMassGenerator,
)
from neural_priors_gym.data.masses.chirp_mass_ratio_redshift import (
    ChirpMassRatioRedshiftMassGenerator,
)
from neural_priors_gym.data.masses.double_gaussian import DoubleGaussianMassGenerator
from neural_priors_gym.data.masses.gaussian import GaussianMassGenerator
from neural_priors_gym.data.masses.nsbh import NSBHMassGenerator
from neural_priors_gym.data.masses.uniform import UniformMassGenerator
from neural_priors_gym.logging_config import get_logger

logger = get_logger("neural_priors_gym.data")


def _build_mass_quantities(
    mass_1_source: np.ndarray, mass_2_source: np.ndarray
) -> dict[str, np.ndarray]:
    """Compute all supported mass quantities from component masses."""
    return {
        "mass_1_source": mass_1_source,
        "mass_2_source": mass_2_source,
        "chirp_mass_source": component_masses_to_chirp_mass(
            mass_1_source, mass_2_source
        ),
        "mass_ratio": component_masses_to_mass_ratio(mass_1_source, mass_2_source),
    }


def _build_lambda_quantities(
    mass_1_source: np.ndarray,
    mass_2_source: np.ndarray,
    lambda_1: np.ndarray,
    lambda_2: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute all supported lambda quantities for BNS systems."""
    return {
        "lambda_1": lambda_1,
        "lambda_2": lambda_2,
        "lambda_tilde": lambda_1_lambda_2_to_lambda_tilde(
            lambda_1, lambda_2, mass_1_source, mass_2_source
        ),
        "delta_lambda_tilde": lambda_1_lambda_2_to_delta_lambda_tilde(
            lambda_1, lambda_2, mass_1_source, mass_2_source
        ),
    }


def build_mass_generator(config: TrainingConfig) -> MassGenerator:
    """Construct the appropriate MassGenerator from the training config.

    For NSBH sources, the base NS mass generator is wrapped in
    NSBHMassGenerator so that m1 (BH) is sampled above MTOV and m2 (NS)
    is sampled from the chosen NS population model.
    """
    mass_config = config.masses
    if isinstance(mass_config, UniformMassConfig):
        ns_gen: MassGenerator = UniformMassGenerator(mass_config)
    elif isinstance(mass_config, GaussianMassConfig):
        ns_gen = GaussianMassGenerator(mass_config)
    elif isinstance(mass_config, DoubleGaussianMassConfig):
        ns_gen = DoubleGaussianMassGenerator(mass_config)
    elif isinstance(mass_config, BilbyMassConfig):
        ns_gen = BilbyPriorMassGenerator(mass_config)
    elif isinstance(mass_config, ChirpMassRatioRedshiftMassConfig):
        ns_gen = ChirpMassRatioRedshiftMassGenerator(mass_config)
    elif isinstance(mass_config, ChirpMassRatioGaussianRedshiftMassConfig):
        ns_gen = ChirpMassRatioGaussianRedshiftMassGenerator(mass_config)
    else:
        raise ValueError(f"Unknown mass config type: {type(mass_config)}")

    if config.source_type == "nsbh":
        return NSBHMassGenerator(ns_gen, config.m_max_bh)

    return ns_gen


class TrainingDataGenerator:
    """Generates training data by pairing a mass generator with an EOS interpolator.

    Parameters
    ----------
    mass_generator:
        Produces (m1, m2) pairs according to the chosen population model.
    lambda_interpolator:
        Produces Lambda values from EOS samples.
    n_samples:
        Number of training samples to generate.
    source_type:
        ``"bns"`` or ``"nsbh"``. For NSBH, only m2 (the NS) is passed to the
        lambda interpolator; m1 (the BH) has no tidal deformability.
    """

    def __init__(
        self,
        mass_generator: MassGenerator,
        lambda_interpolator: EOSLambdaInterpolator,
        n_samples: int,
        source_type: str = "bns",
    ) -> None:
        self.mass_generator = mass_generator
        self.lambda_interpolator = lambda_interpolator
        self.n_samples = n_samples
        self.source_type = source_type

    @classmethod
    def from_config(cls, config: TrainingConfig) -> "TrainingDataGenerator":
        """Build a TrainingDataGenerator from a TrainingConfig."""
        if config.masses is None or config.lambdas is None:
            raise ValueError(
                "TrainingDataGenerator.from_config requires 'masses' and 'lambdas' "
                "to be set in the config. Use 'training.data_path' to load a "
                "pre-generated npz file instead."
            )
        mass_gen = build_mass_generator(config)
        lambda_interp = EOSLambdaInterpolator(config.lambdas)
        return cls(
            mass_generator=mass_gen,
            lambda_interpolator=lambda_interp,
            n_samples=config.training.n_samples,
            source_type=config.source_type,
        )

    def generate(self) -> dict[str, np.ndarray]:
        """Generate the full training dataset.

        Always returns all computable mass and lambda quantities. The caller
        selects which columns to use for training via
        ``training.parameter_names``.

        For NSBH sources, only mass_2_source (the NS) is passed to the lambda
        interpolator. mass_1_source (BH) gets lambda_1=0, so lambda_tilde and
        delta_lambda_tilde are still well-defined.

        Returns
        -------
        dict[str, np.ndarray]
            All supported quantities as 1-D arrays of length ``n_samples``:
            ``mass_1_source``, ``mass_2_source``, ``chirp_mass_source``,
            ``mass_ratio``, ``lambda_1``, ``lambda_2``, ``lambda_tilde``,
            ``delta_lambda_tilde``.
        """
        logger.info(f"Generating {self.n_samples} training samples ...")

        masses = self.mass_generator.generate(
            self.n_samples, self.lambda_interpolator.mtov_array
        )
        mass_1_source, mass_2_source = masses[:, 0], masses[:, 1]

        mass_quantities = _build_mass_quantities(mass_1_source, mass_2_source)
        mass_quantities.update(self.mass_generator.extra_quantities())

        # For NSBH, only pass mass_2_source (NS) to the interpolator.
        if self.source_type == "nsbh":
            lambda_masses = mass_2_source[:, np.newaxis]
        else:
            lambda_masses = masses

        lambdas = self.lambda_interpolator.interpolate(lambda_masses)

        if self.source_type == "bns":
            lambda_quantities: dict[str, np.ndarray] = _build_lambda_quantities(
                mass_1_source, mass_2_source, lambdas[:, 0], lambdas[:, 1]
            )
        else:
            # NSBH: BH has lambda_1=0; NS lambda_2 drives all tidal combinations.
            lambda_1_nsbh = np.zeros_like(lambdas[:, 0])
            lambda_2_nsbh = lambdas[:, 0]
            lambda_quantities = _build_lambda_quantities(
                mass_1_source, mass_2_source, lambda_1_nsbh, lambda_2_nsbh
            )

        data = {**mass_quantities, **lambda_quantities}
        logger.info("Training data generation complete.")
        return data

    def save(self, path: Path) -> None:
        """Generate and save training data to an npz file."""
        data = self.generate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, allow_pickle=True, **data)
        logger.info(f"Training data saved to {path}")
