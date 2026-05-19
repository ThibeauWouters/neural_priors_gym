"""Gaussian mass generator."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import truncnorm

from neural_priors_gym.config.schemas.masses import GaussianMassConfig

from .base import MassGenerator
from .br2016 import max_spin, mmax_from_spin

if TYPE_CHECKING:
    from neural_priors_gym.config.schemas.spin import SpinConfig


class GaussianMassGenerator(MassGenerator):
    """Samples masses from a Gaussian distribution truncated to [m_min, MTOV].

    Default parameters (mean=1.33, std=0.09) are taken from the galactic
    NS population study in arXiv:2407.16669.
    """

    def __init__(self, config: GaussianMassConfig) -> None:
        self.m_min = config.m_min
        self.mean = config.mean
        self.std = config.std
        self.max_attempts = config.max_attempts

    def _sample_truncated(self, mtov: float) -> float:
        a = (self.m_min - self.mean) / self.std
        b = (mtov - self.mean) / self.std
        m = float(truncnorm.rvs(a, b, loc=self.mean, scale=self.std))
        if not (self.m_min <= m <= mtov):
            raise RuntimeError(
                f"truncnorm sample {m:.4f} out of [{self.m_min}, {mtov:.4f}] "
                f"after {self.max_attempts} max_attempts (numerical edge case)."
            )
        return m

    def _sample_truncated_upper(self, upper: float) -> float:
        """Sample from Gaussian truncated to [m_min, upper]."""
        a = (self.m_min - self.mean) / self.std
        b = (upper - self.mean) / self.std
        m = float(truncnorm.rvs(a, b, loc=self.mean, scale=self.std))
        if not (self.m_min <= m <= upper):
            raise RuntimeError(
                f"truncnorm sample {m:.4f} out of [{self.m_min}, {upper:.4f}] "
                "(numerical edge case)."
            )
        return m

    def generate(self, n_samples: int, mtov_array: np.ndarray) -> np.ndarray:
        eos_indices = np.random.randint(0, len(mtov_array), size=n_samples)
        mtov_per_sample = mtov_array[eos_indices]

        masses = np.empty((n_samples, 2))
        for idx in range(n_samples):
            mtov = mtov_per_sample[idx]
            masses[idx, 0] = self._sample_truncated(mtov)
            masses[idx, 1] = self._sample_truncated(mtov)

        m1 = np.maximum(masses[:, 0], masses[:, 1])
        m2 = np.minimum(masses[:, 0], masses[:, 1])
        return np.column_stack([m1, m2])

    def generate_ns_mass(self, n_samples: int, mtov_array: np.ndarray) -> np.ndarray:
        eos_indices = np.random.randint(0, len(mtov_array), size=n_samples)
        mtov_per_sample = mtov_array[eos_indices]
        return np.array(
            [self._sample_truncated(mtov_per_sample[i]) for i in range(n_samples)]
        )

    def generate_with_spin(
        self,
        n_samples: int,
        mtov_array: np.ndarray,
        rtov_array: np.ndarray,
        spin_config: SpinConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        """BNS Gaussian masses with spin on mass_1_source (a_1).

        m_a is drawn from the Gaussian truncated to [m_min, M_max(a_1)].
        m_b is drawn from the Gaussian truncated to [m_min, MTOV].  After
        sorting, samples where mass_2_source > MTOV are rejected.
        """
        spins = np.empty(n_samples)
        m1_out = np.empty(n_samples)
        m2_out = np.empty(n_samples)

        remaining = list(range(n_samples))
        while remaining:
            r = len(remaining)
            eos_idx = np.random.randint(0, len(mtov_array), size=r)
            mtov = mtov_array[eos_idx]
            rtov = rtov_array[eos_idx]

            spin_max_phys = max_spin(mtov, rtov)
            eff_spin_max = np.minimum(spin_config.spin_max, spin_max_phys)
            a1 = np.random.uniform(spin_config.spin_min, eff_spin_max)
            mmax_1 = mmax_from_spin(a1, mtov, rtov)

            new_remaining = []
            for j, i in enumerate(remaining):
                m_a = self._sample_truncated_upper(float(mmax_1[j]))
                m_b = self._sample_truncated(float(mtov[j]))
                cand_m1 = max(m_a, m_b)
                cand_m2 = min(m_a, m_b)
                if cand_m2 <= float(mtov[j]):
                    m1_out[i] = cand_m1
                    m2_out[i] = cand_m2
                    spins[i] = float(a1[j])
                else:
                    new_remaining.append(i)
            remaining = new_remaining

        return np.column_stack([m1_out, m2_out]), spins

    def generate_ns_mass_with_spin(
        self,
        n_samples: int,
        mtov_array: np.ndarray,
        rtov_array: np.ndarray,
        spin_config: SpinConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        """NS mass with spin-corrected upper bound for use in NSBH (a_2)."""
        eos_idx = np.random.randint(0, len(mtov_array), size=n_samples)
        mtov = mtov_array[eos_idx]
        rtov = rtov_array[eos_idx]

        spin_max_phys = max_spin(mtov, rtov)
        eff_spin_max = np.minimum(spin_config.spin_max, spin_max_phys)
        a2 = np.random.uniform(spin_config.spin_min, eff_spin_max)
        mmax_ns = mmax_from_spin(a2, mtov, rtov)

        masses = np.array(
            [self._sample_truncated_upper(float(mmax_ns[i])) for i in range(n_samples)]
        )
        return masses, a2
