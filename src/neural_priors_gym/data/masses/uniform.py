"""Uniform mass generator."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from neural_priors_gym.config.schemas.masses import UniformMassConfig

from .base import MassGenerator
from .br2016 import max_spin, mmax_from_spin

if TYPE_CHECKING:
    from neural_priors_gym.config.schemas.spin import SpinConfig


class UniformMassGenerator(MassGenerator):
    """Samples masses uniformly in [m_min, MTOV] for each EOS draw.

    For each sample, an EOS is chosen at random and the upper bound is set
    to the maximum mass of that EOS (MTOV). Both masses are drawn from the
    same uniform distribution and then sorted so that mass_1_source >= mass_2_source.
    """

    def __init__(self, config: UniformMassConfig) -> None:
        self.m_min = config.m_min

    def generate(self, n_samples: int, mtov_array: np.ndarray) -> np.ndarray:
        eos_indices = np.random.randint(0, len(mtov_array), size=n_samples)
        mtov_per_sample = mtov_array[eos_indices]

        u1 = np.random.uniform(0.0, 1.0, n_samples)
        u2 = np.random.uniform(0.0, 1.0, n_samples)
        m_a = self.m_min + u1 * (mtov_per_sample - self.m_min)
        m_b = self.m_min + u2 * (mtov_per_sample - self.m_min)

        m1 = np.maximum(m_a, m_b)
        m2 = np.minimum(m_a, m_b)
        return np.column_stack([m1, m2])

    def generate_ns_mass(self, n_samples: int, mtov_array: np.ndarray) -> np.ndarray:
        eos_indices = np.random.randint(0, len(mtov_array), size=n_samples)
        mtov_per_sample = mtov_array[eos_indices]
        u = np.random.uniform(0.0, 1.0, n_samples)
        return self.m_min + u * (mtov_per_sample - self.m_min)

    def generate_with_spin(
        self,
        n_samples: int,
        mtov_array: np.ndarray,
        rtov_array: np.ndarray,
        spin_config: SpinConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        """BNS uniform masses with spin on mass_1_source (a_1).

        mass_1_source is sampled from [m_min, M_max(a_1)] where M_max is
        spin-corrected.  mass_2_source is bounded by bare MTOV.  Rejection
        sampling removes the rare cases where the min of the two draws exceeds
        MTOV (which would place mass_2_source above the non-spinning limit).
        """
        spins = np.empty(n_samples)
        m1_out = np.empty(n_samples)
        m2_out = np.empty(n_samples)

        remaining = np.arange(n_samples)
        while len(remaining) > 0:
            r = len(remaining)
            eos_idx = np.random.randint(0, len(mtov_array), size=r)
            mtov = mtov_array[eos_idx]
            rtov = rtov_array[eos_idx]

            spin_max_phys = max_spin(mtov, rtov)
            eff_spin_max = np.minimum(spin_config.spin_max, spin_max_phys)
            a1 = np.random.uniform(spin_config.spin_min, eff_spin_max)

            mmax_1 = mmax_from_spin(a1, mtov, rtov)

            u1 = np.random.uniform(0.0, 1.0, r)
            u2 = np.random.uniform(0.0, 1.0, r)
            m_a = self.m_min + u1 * (mmax_1 - self.m_min)
            m_b = self.m_min + u2 * (mtov - self.m_min)

            cand_m1 = np.maximum(m_a, m_b)
            cand_m2 = np.minimum(m_a, m_b)

            # Reject when the lighter draw exceeds bare MTOV.  This happens
            # only when m_a > MTOV and m_b < m_a, i.e. when the spin-boosted
            # draw would become the heavier component but is above MTOV, making
            # the non-spinning draw (mass_2_source) exceed MTOV.
            valid = cand_m2 <= mtov

            idx = remaining[valid]
            m1_out[idx] = cand_m1[valid]
            m2_out[idx] = cand_m2[valid]
            spins[idx] = a1[valid]
            remaining = remaining[~valid]

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
        u = np.random.uniform(0.0, 1.0, n_samples)
        masses = self.m_min + u * (mmax_ns - self.m_min)
        return masses, a2
