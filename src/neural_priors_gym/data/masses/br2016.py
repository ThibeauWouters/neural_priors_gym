"""Universal relations for the maximum mass of a spinning neutron star."""

import numpy as np

_CGS_G = 6.674e-8
_CGS_MSUN = 1.989e33
_CGS_C = 2.998e10

_B4 = 5.543e-1
_C2 = 4.283e-1
_C4 = 7.533e-1


def _compactness(mtov: np.ndarray, rtov: np.ndarray) -> np.ndarray:
    """Dimensionless compactness at the TOV mass. mtov in Msun, rtov in km."""
    return (_CGS_G * _CGS_MSUN * mtov) / (_CGS_C**2 * rtov * 1e5)


def max_spin(mtov: np.ndarray, rtov: np.ndarray) -> np.ndarray:
    r"""Physical maximum dimensionless spin magnitude.

    Parameters
    ----------
    mtov:
        TOV mass in solar masses.
    rtov:
        Radius at the TOV mass in km.

    Returns
    -------
    np.ndarray
        Maximum dimensionless spin, same shape as inputs.
    """
    return _B4 * _compactness(mtov, rtov) ** -0.5


def mmax_from_spin(
    spin: np.ndarray,
    mtov: np.ndarray,
    rtov: np.ndarray,
) -> np.ndarray:
    r"""Maximum mass of a spinning neutron star.

    Applies the universal relation

    .. math::

        M_\mathrm{max} = M_\mathrm{TOV}
            \bigl(1 + c_2 \mathcal{C}\,\chi^2 + c_4 \mathcal{C}^2\,\chi^4\bigr)

    where :math:`\mathcal{C}` is the compactness at the TOV mass and
    :math:`\chi` is the dimensionless spin.

    Parameters
    ----------
    spin:
        Dimensionless spin magnitude.
    mtov:
        TOV mass in solar masses.
    rtov:
        Radius at the TOV mass in km.

    Returns
    -------
    np.ndarray
        Maximum mass in solar masses, same shape as inputs.
    """
    c = _compactness(mtov, rtov)
    return mtov * (1.0 + _C2 * c * spin**2 + _C4 * c**2 * spin**4)
