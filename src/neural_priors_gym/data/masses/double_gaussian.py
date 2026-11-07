"""Double Gaussian mass generator."""

import numpy as np

from neural_priors_gym.config.schemas.masses import DoubleGaussianMassConfig

from .base import MassGenerator, mtov_per_sample, sample_truncated_normal


class DoubleGaussianMassGenerator(MassGenerator):
    """Samples masses from a mixture of two Gaussians truncated to [m_min, MTOV].

    For each draw a component is chosen with probability ``weight`` (first) or
    ``1 - weight`` (second), and the mass is drawn from that component
    truncated to [m_min, MTOV]. The mixture weights are therefore not
    renormalised for the mass each component loses to the truncation.
    arXiv:2511.22987 uses (1.34, 0.07) with weight 0.65 and (1.80, 0.21).
    """

    def __init__(self, config: DoubleGaussianMassConfig) -> None:
        self.m_min = config.m_min
        self.mean_1 = config.mean_1
        self.std_1 = config.std_1
        self.mean_2 = config.mean_2
        self.std_2 = config.std_2
        self.weight = config.weight
        # Kept for config compatibility; the inverse-CDF sampler needs no retries.
        self.max_attempts = config.max_attempts

    def _draw(self, mtov: np.ndarray) -> np.ndarray:
        first = np.random.uniform(size=mtov.shape) < self.weight
        mean = np.where(first, self.mean_1, self.mean_2)
        std = np.where(first, self.std_1, self.std_2)
        return sample_truncated_normal(mean, std, self.m_min, mtov)

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
