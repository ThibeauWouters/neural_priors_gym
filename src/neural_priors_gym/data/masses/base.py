"""Abstract base class for mass generators."""

from abc import ABC, abstractmethod

import numpy as np
from scipy.special import ndtr, ndtri


class MassGenerator(ABC):
    """Base class for all mass distribution generators.

    Subclasses implement generate() to produce pairs of neutron star masses
    consistent with the chosen population model and the available EOS samples.

    Each sample is tied to one EOS, whose maximum TOV mass bounds the NS
    masses. :class:`~neural_priors_gym.data.generator.TrainingDataGenerator`
    passes the same ``eos_indices`` to the mass generator and to the Lambda
    interpolator, so that masses and tidal deformabilities of a sample come from
    the same EOS, as in arXiv:2511.22987. If ``eos_indices`` is omitted, an EOS
    is drawn at random per sample.
    """

    @abstractmethod
    def generate(
        self,
        n_samples: int,
        mtov_array: np.ndarray,
        eos_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        """Generate mass pairs.

        Parameters
        ----------
        n_samples:
            Number of (mass_1_source, mass_2_source) pairs to generate.
        mtov_array:
            Array of shape (n_eos,) with the maximum TOV mass for each EOS.
        eos_indices:
            Optional array of shape (n_samples,) with the EOS used by each
            sample. Drawn uniformly at random if omitted.

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, 2) where each row is
            (mass_1_source, mass_2_source) with mass_1_source >= mass_2_source.
        """
        ...

    @abstractmethod
    def generate_ns_mass(
        self,
        n_samples: int,
        mtov_array: np.ndarray,
        eos_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        """Generate single NS masses for use as the secondary in NSBH systems.

        Parameters
        ----------
        n_samples:
            Number of NS masses to generate.
        mtov_array:
            Array of shape (n_eos,) with MTOV per EOS, used as the upper bound.
        eos_indices:
            Optional array of shape (n_samples,) with the EOS used by each
            sample. Drawn uniformly at random if omitted.

        Returns
        -------
        np.ndarray
            Array of shape (n_samples,) containing NS masses below MTOV.
        """
        ...


def mtov_per_sample(
    n_samples: int, mtov_array: np.ndarray, eos_indices: np.ndarray | None
) -> np.ndarray:
    """Return the M_TOV of each sample's EOS, drawing EOS indices if needed."""
    mtov_array = np.asarray(mtov_array)
    if eos_indices is None:
        eos_indices = np.random.randint(0, len(mtov_array), size=n_samples)
    eos_indices = np.asarray(eos_indices)
    if eos_indices.shape != (n_samples,):
        raise ValueError(
            f"eos_indices must have shape ({n_samples},), got {eos_indices.shape}"
        )
    return mtov_array[eos_indices]


def sample_truncated_normal(
    mean: np.ndarray | float,
    std: np.ndarray | float,
    low: np.ndarray | float,
    high: np.ndarray | float,
) -> np.ndarray:
    r"""Vectorised inverse-CDF draws from a truncated normal distribution.

    Draws from :math:`N(\mu, \sigma)` truncated to [low, high].

    All arguments broadcast against each other. To keep precision when the
    interval lies in the upper tail, draws with a positive standardised lower
    bound are made from the mirrored distribution.
    """
    mean, std, low, high = np.broadcast_arrays(
        np.asarray(mean, float),
        np.asarray(std, float),
        np.asarray(low, float),
        np.asarray(high, float),
    )
    if np.any(low >= high):
        raise ValueError("Truncation interval is empty (low >= high).")
    a = (low - mean) / std
    b = (high - mean) / std
    flip = a > 0
    lo = np.where(flip, -b, a)
    hi = np.where(flip, -a, b)
    u = np.random.uniform(size=mean.shape)
    z = ndtri(ndtr(lo) + u * (ndtr(hi) - ndtr(lo)))
    z = np.clip(z, lo, hi)
    z = np.where(flip, -z, z)
    return np.clip(mean + std * z, low, high)
