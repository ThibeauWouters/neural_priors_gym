"""Gaussian mass generator."""

import numpy as np

from neural_priors_gym.config.schemas.masses import GaussianMassConfig

from .base import MassGenerator, mtov_per_sample, sample_truncated_normal


class GaussianMassGenerator(MassGenerator):
    """Samples masses from a Gaussian distribution truncated to [m_min, MTOV].

    The upper bound is the MTOV of each sample's EOS. arXiv:2511.22987 uses
    mean 1.33 and standard deviation 0.09.
    """

    def __init__(self, config: GaussianMassConfig) -> None:
        self.m_min = config.m_min
        self.mean = config.mean
        self.std = config.std
        # Kept for config compatibility; the inverse-CDF sampler needs no retries.
        self.max_attempts = config.max_attempts

    def _draw(self, mtov: np.ndarray) -> np.ndarray:
        return sample_truncated_normal(self.mean, self.std, self.m_min, mtov)

    def generate(
        self,
        n_samples: int,
        mtov_array: np.ndarray,
        eos_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        mtov = mtov_per_sample(n_samples, mtov_array, eos_indices)
        m_a = self._draw(mtov)
        m_b = self._draw(mtov)
        return np.column_stack([np.maximum(m_a, m_b), np.minimum(m_a, m_b)])

    def generate_ns_mass(
        self,
        n_samples: int,
        mtov_array: np.ndarray,
        eos_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        return self._draw(mtov_per_sample(n_samples, mtov_array, eos_indices))
