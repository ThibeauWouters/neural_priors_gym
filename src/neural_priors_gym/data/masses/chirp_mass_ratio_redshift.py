"""Mass generator from detector-frame chirp mass, mass ratio, and redshift."""

import numpy as np

from neural_priors_gym.config.schemas.masses import ChirpMassRatioRedshiftMassConfig
from neural_priors_gym.logging_config import get_logger

from ._chirp_mass_ratio import sample_component_masses
from .base import MassGenerator

logger = get_logger("neural_priors_gym.data.masses")


class ChirpMassRatioRedshiftMassGenerator(MassGenerator):
    """Samples detector-frame chirp mass and mass ratio uniformly in
    component mass, plus an independent redshift prior, then converts to
    source-frame component masses.

    Chirp mass and mass ratio are drawn from bilby's
    ``UniformInComponentsChirpMass`` and ``UniformInComponentsMassRatio``
    priors, which together are equivalent to a uniform-in-component-mass
    prior. Redshift is drawn from a bilby cosmological prior
    (``UniformSourceFrame`` or ``UniformComovingVolume``) and used to
    convert the resulting detector-frame component masses to source frame
    via mass_source = mass / (1 + z).
    """

    def __init__(self, config: ChirpMassRatioRedshiftMassConfig) -> None:
        self.mc_min = config.mc_min
        self.mc_max = config.mc_max
        self.q_min = config.q_min
        self.q_max = config.q_max
        self.z_min = config.z_min
        self.z_max = config.z_max
        self.redshift_prior_type = config.redshift_prior
        self.cosmology = config.cosmology
        self._redshift: np.ndarray | None = None
        self._chirp_mass: np.ndarray | None = None

    def _sample_redshift(self, n_samples: int) -> np.ndarray:
        from bilby.gw.prior import UniformComovingVolume, UniformSourceFrame

        redshift_cls = (
            UniformSourceFrame
            if self.redshift_prior_type == "uniform_source_frame"
            else UniformComovingVolume
        )
        redshift_prior = redshift_cls(
            minimum=self.z_min,
            maximum=self.z_max,
            cosmology=self.cosmology,
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
            "/ redshift prior"
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
