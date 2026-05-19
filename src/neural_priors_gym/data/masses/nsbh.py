"""NSBH mass generator: BH mass above MTOV, NS mass from any NS generator."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import MassGenerator

if TYPE_CHECKING:
    from neural_priors_gym.config.schemas.spin import SpinConfig


class NSBHMassGenerator(MassGenerator):
    """Generates mass pairs for neutron star-black hole systems.

    The primary (mass_1_source) is the black hole, sampled uniformly in
    [MTOV, m_max_bh]. The secondary (mass_2_source) is the neutron star,
    sampled from a wrapped NS mass generator. By construction
    mass_1_source > MTOV >= mass_2_source, so the ordering is guaranteed.

    Parameters
    ----------
    ns_generator:
        Any MassGenerator used to produce the NS (secondary) mass.
    m_max_bh:
        Upper bound on the BH mass in solar masses.
    """

    def __init__(self, ns_generator: MassGenerator, m_max_bh: float) -> None:
        self.ns_generator = ns_generator
        self.m_max_bh = m_max_bh

    def generate(self, n_samples: int, mtov_array: np.ndarray) -> np.ndarray:
        eos_indices = np.random.randint(0, len(mtov_array), size=n_samples)
        mtov_per_sample = mtov_array[eos_indices]

        if np.any(mtov_per_sample >= self.m_max_bh):
            raise ValueError(
                f"m_max_bh={self.m_max_bh} is not above MTOV for all EOS samples "
                f"(max MTOV found: {mtov_per_sample.max():.2f} Msun). Increase m_max_bh."
            )

        u = np.random.uniform(0.0, 1.0, n_samples)
        m1 = mtov_per_sample + u * (self.m_max_bh - mtov_per_sample)

        m2 = self.ns_generator.generate_ns_mass(n_samples, mtov_array)

        return np.column_stack([m1, m2])

    def generate_ns_mass(self, n_samples: int, mtov_array: np.ndarray) -> np.ndarray:
        return self.ns_generator.generate_ns_mass(n_samples, mtov_array)

    def generate_with_spin(
        self,
        n_samples: int,
        mtov_array: np.ndarray,
        rtov_array: np.ndarray,
        spin_config: SpinConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        """NSBH masses with spin on the NS secondary (a_2).

        The NS mass upper bound is extended by the spin correction.  The BH
        mass is still sampled uniformly in [MTOV, m_max_bh].  Rejection
        sampling removes rare cases where the spin-boosted NS mass exceeds
        the BH mass.
        """
        m2_arr, spins = self.ns_generator.generate_ns_mass_with_spin(
            n_samples, mtov_array, rtov_array, spin_config
        )

        m1_arr = np.empty(n_samples)
        remaining = np.where(np.ones(n_samples, dtype=bool))[0]

        while len(remaining) > 0:
            r = len(remaining)
            eos_idx = np.random.randint(0, len(mtov_array), size=r)
            mtov = mtov_array[eos_idx]

            if np.any(mtov >= self.m_max_bh):
                raise ValueError(
                    f"m_max_bh={self.m_max_bh} is not above MTOV for all EOS samples "
                    f"(max MTOV found: {mtov.max():.2f} Msun). Increase m_max_bh."
                )

            u = np.random.uniform(0.0, 1.0, r)
            cand_m1 = mtov + u * (self.m_max_bh - mtov)

            # Reject if BH (m1) is lighter than the spinning NS (m2)
            valid = cand_m1 >= m2_arr[remaining]
            idx = remaining[valid]
            m1_arr[idx] = cand_m1[valid]
            remaining = remaining[~valid]

        return np.column_stack([m1_arr, m2_arr]), spins

    def generate_ns_mass_with_spin(
        self,
        n_samples: int,
        mtov_array: np.ndarray,
        rtov_array: np.ndarray,
        spin_config: SpinConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.ns_generator.generate_ns_mass_with_spin(
            n_samples, mtov_array, rtov_array, spin_config
        )
