"""Abstract base class for mass generators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from neural_priors_gym.config.schemas.spin import SpinConfig


class MassGenerator(ABC):
    """Base class for all mass distribution generators.

    Subclasses implement generate() to produce pairs of neutron star masses
    consistent with the chosen population model and the available EOS samples.
    """

    @abstractmethod
    def generate(self, n_samples: int, mtov_array: np.ndarray) -> np.ndarray:
        """Generate mass pairs.

        Parameters
        ----------
        n_samples:
            Number of (mass_1_source, mass_2_source) pairs to generate.
        mtov_array:
            Array of shape (n_eos,) with the maximum TOV mass for each EOS.
            A random EOS is drawn per sample and its MTOV used as the upper
            bound on the NS mass.

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, 2) where each row is
            (mass_1_source, mass_2_source) with mass_1_source >= mass_2_source.
        """
        ...

    @abstractmethod
    def generate_ns_mass(self, n_samples: int, mtov_array: np.ndarray) -> np.ndarray:
        """Generate single NS masses for use as the secondary in NSBH systems.

        Parameters
        ----------
        n_samples:
            Number of NS masses to generate.
        mtov_array:
            Array of shape (n_eos,) with MTOV per EOS, used as the upper bound.

        Returns
        -------
        np.ndarray
            Array of shape (n_samples,) containing NS masses below MTOV.
        """
        ...

    def generate_with_spin(
        self,
        n_samples: int,
        mtov_array: np.ndarray,
        rtov_array: np.ndarray,
        spin_config: SpinConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate mass pairs with spin-corrected upper mass limit.

        The spin is sampled uniformly in ``[spin_min, min(spin_max, chi_max)]``
        where ``chi_max`` is the physical maximum spin for the drawn EOS.  The
        effective maximum mass is computed via the universal relation and used
        as the upper bound for the spinning NS.

        Parameters
        ----------
        n_samples:
            Number of samples to generate.
        mtov_array:
            Maximum TOV mass per EOS, shape (n_eos,).
        rtov_array:
            Radius at the TOV mass per EOS in km, shape (n_eos,).
        spin_config:
            Spin prior configuration (``parameter_name``, ``spin_min``,
            ``spin_max``).

        Returns
        -------
        masses : np.ndarray
            Shape (n_samples, 2).
        spins : np.ndarray
            Shape (n_samples,).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement generate_with_spin()."
        )

    def generate_ns_mass_with_spin(
        self,
        n_samples: int,
        mtov_array: np.ndarray,
        rtov_array: np.ndarray,
        spin_config: SpinConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate single NS masses with spin-corrected upper bound for NSBH.

        Parameters
        ----------
        n_samples:
            Number of NS masses to generate.
        mtov_array:
            Maximum TOV mass per EOS, shape (n_eos,).
        rtov_array:
            Radius at the TOV mass per EOS in km, shape (n_eos,).
        spin_config:
            Spin prior configuration.

        Returns
        -------
        masses : np.ndarray
            Shape (n_samples,).
        spins : np.ndarray
            Shape (n_samples,).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement generate_ns_mass_with_spin()."
        )
