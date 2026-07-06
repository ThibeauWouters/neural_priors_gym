"""Mass generator from chirp mass, mass ratio, and a fixed Gaussian redshift."""

import numpy as np

from neural_priors_gym.config.schemas.masses import (
    ChirpMassRatioGaussianRedshiftMassConfig,
)
from neural_priors_gym.logging_config import get_logger

from ._chirp_mass_ratio import sample_component_masses
from .base import MassGenerator

logger = get_logger("neural_priors_gym.data.masses")


class ChirpMassRatioGaussianRedshiftMassGenerator(MassGenerator):
    """Samples detector-frame chirp mass and mass ratio uniformly in
    component mass, plus a fixed Gaussian redshift, then converts to
    source-frame component masses.

    Redshift is drawn from a Gaussian truncated at zero (bilby's
    ``TruncatedGaussian``) centered on ``z_mean`` with width ``z_std``,
    representing a BNS with a known electromagnetic counterpart (the
    redshift is measured from the host galaxy rather than sampled from a
    broad population prior).
    """

    def __init__(self, config: ChirpMassRatioGaussianRedshiftMassConfig) -> None:
        self.mc_min = config.mc_min
        self.mc_max = config.mc_max
        self.q_min = config.q_min
        self.q_max = config.q_max
        self.z_mean = config.z_mean
        self.z_std = config.z_std
        self._redshift: np.ndarray | None = None
        self._chirp_mass: np.ndarray | None = None

    def _sample_redshift(self, n_samples: int) -> np.ndarray:
        from bilby.core.prior import TruncatedGaussian

        redshift_prior = TruncatedGaussian(
            mu=self.z_mean,
            sigma=self.z_std,
            minimum=0.0,
            maximum=np.inf,
            name="redshift",
        )
        return np.asarray(redshift_prior.sample(n_samples))

    def _sample_source_frame_masses(self, n_samples: int) -> np.ndarray:
        mass_1, mass_2, chirp_mass, _ = sample_component_masses(
            self.mc_min, self.mc_max, self.q_min, self.q_max, n_samples
        )
        self._chirp_mass = chirp_mass

        redshift = self._sample_redshift(n_samples)
        self._redshift = redshift

        mass_1_source = mass_1 / (1.0 + redshift)
        mass_2_source = mass_2 / (1.0 + redshift)

        m1_out = np.maximum(mass_1_source, mass_2_source)
        m2_out = np.minimum(mass_1_source, mass_2_source)
        return np.column_stack([m1_out, m2_out])

    def generate(
        self, n_samples: int, mtov_array: np.ndarray
    ) -> np.ndarray:  # noqa: ARG002
        result = self._sample_source_frame_masses(n_samples)
        logger.debug(
            f"Generated {n_samples} mass samples from chirp mass / mass ratio "
            "/ Gaussian redshift prior"
        )
        return result

    def generate_ns_mass(
        self, n_samples: int, mtov_array: np.ndarray
    ) -> np.ndarray:  # noqa: ARG002
        return self.generate(n_samples, mtov_array)[:, 1]

    def extra_quantities(self) -> dict[str, np.ndarray]:
        if self._redshift is None or self._chirp_mass is None:
            return {}
        return {"redshift": self._redshift, "chirp_mass": self._chirp_mass}
