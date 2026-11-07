"""Uniform mass generator."""

import numpy as np

from neural_priors_gym.config.schemas.masses import UniformMassConfig

from .base import MassGenerator, mtov_per_sample


class UniformMassGenerator(MassGenerator):
    """Samples masses uniformly in [m_min, MTOV] for each EOS draw.

    For each sample, the upper bound is the maximum mass (MTOV) of that
    sample's EOS. Both masses are drawn from the
    same uniform distribution and then sorted so that mass_1_source >= mass_2_source.
    """

    def __init__(self, config: UniformMassConfig) -> None:
        self.m_min = config.m_min

    def generate(
        self,
        n_samples: int,
        mtov_array: np.ndarray,
        eos_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        mtov = mtov_per_sample(n_samples, mtov_array, eos_indices)
        u1 = np.random.uniform(0.0, 1.0, n_samples)
        u2 = np.random.uniform(0.0, 1.0, n_samples)
        m_a = self.m_min + u1 * (mtov - self.m_min)
        m_b = self.m_min + u2 * (mtov - self.m_min)

        m1 = np.maximum(m_a, m_b)
        m2 = np.minimum(m_a, m_b)
        return np.column_stack([m1, m2])

    def generate_ns_mass(
        self,
        n_samples: int,
        mtov_array: np.ndarray,
        eos_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        mtov = mtov_per_sample(n_samples, mtov_array, eos_indices)
        u = np.random.uniform(0.0, 1.0, n_samples)
        return self.m_min + u * (mtov - self.m_min)
