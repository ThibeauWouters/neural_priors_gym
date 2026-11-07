"""Mass generator that samples from a bilby prior file."""

from pathlib import Path

import numpy as np

from neural_priors_gym.config.schemas.masses import BilbyMassConfig
from neural_priors_gym.logging_config import get_logger

from .base import MassGenerator

logger = get_logger("neural_priors_gym.data.masses")


class BilbyPriorMassGenerator(MassGenerator):
    """Generates masses by sampling from a bilby prior file.

    The prior file is loaded with bilby.core.prior.PriorDict. Samples are
    passed through bilby.gw.conversion.generate_all_bns_parameters, and
    mass_1_source and mass_2_source are extracted. The prior file must define
    all parameters required for this conversion (typically chirp_mass,
    mass_ratio, and luminosity_distance).
    """

    def __init__(self, config: BilbyMassConfig) -> None:
        prior_file = Path(config.prior_file)
        if not prior_file.exists():
            raise FileNotFoundError(f"Prior file not found: {prior_file}")
        self.prior_file = str(prior_file)

    def generate(
        self,
        n_samples: int,
        mtov_array: np.ndarray,  # noqa: ARG002
        eos_indices: np.ndarray | None = None,  # noqa: ARG002
    ) -> np.ndarray:
        """Draw masses from the prior file; MTOV and the EOS are not used."""
        import bilby
        from bilby.gw.conversion import generate_all_bns_parameters

        prior = bilby.core.prior.PriorDict(self.prior_file)
        samples = prior.sample(n_samples)
        samples = generate_all_bns_parameters(samples)

        name_1, name_2 = "mass_1_source", "mass_2_source"
        if name_1 not in samples or name_2 not in samples:
            raise ValueError(
                f"generate_all_bns_parameters did not produce '{name_1}' and "
                f"'{name_2}'. Check that the prior file defines the necessary "
                f"parameters (chirp_mass, mass_ratio, luminosity_distance)."
            )

        m1 = np.asarray(samples[name_1])
        m2 = np.asarray(samples[name_2])

        # Ensure m1 >= m2
        m1_out = np.maximum(m1, m2)
        m2_out = np.minimum(m1, m2)

        logger.debug(
            f"Generated {n_samples} mass samples from bilby prior {self.prior_file}"
        )
        return np.column_stack([m1_out, m2_out])

    def generate_ns_mass(
        self,
        n_samples: int,
        mtov_array: np.ndarray,
        eos_indices: np.ndarray | None = None,  # noqa: ARG002
    ) -> np.ndarray:
        """Return the secondary (NS) masses from the bilby prior for NSBH use."""
        return self.generate(n_samples, mtov_array)[:, 1]
