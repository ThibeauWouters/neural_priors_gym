"""Configuration schema for NS spin."""

from typing import Literal

from pydantic import model_validator

from ._base import NeuralPriorBaseModel


class SpinConfig(NeuralPriorBaseModel):
    r"""Uniform prior on the dimensionless spin magnitude of one neutron star.

    When set, the universal relation from Breu & Rezzolla (2016) is used to
    compute an effective maximum mass for the spinning neutron star.  The spin
    is drawn from a uniform distribution in ``[spin_min, spin_max]``, clipped
    from above by the physical maximum spin for each EOS sample.  The spin is
    then included as an additional training parameter.

    The ``parameter_name`` field selects which component carries the spin:

    * ``"a_1"`` — the heavier neutron star (BNS sources only, ``source_type="bns"``).
    * ``"a_2"`` — the lighter neutron star (BNS) or the neutron star in an
      NSBH system (``source_type="nsbh"``).

    .. note::
        ``a_2`` for BNS is not currently supported because the mass ordering
        :math:`m_1 \ge m_2` becomes ill-defined when the spinning secondary
        can be heavier than the non-spinning primary.

    Radii at the TOV mass (``radii_EOS``) must be available in the EOS data
    source.  For ``jester_path``, these are always present.  For ``eos_path``
    npz files, the archive must include a ``radii_EOS`` array.

    Parameters
    ----------
    parameter_name:
        Which component carries the spin; ``"a_1"`` or ``"a_2"``.
    spin_min:
        Lower bound of the uniform spin prior (default 0.0).
    spin_max:
        Upper bound of the uniform spin prior (must be in :math:`(0, 1]`).
    """

    parameter_name: Literal["a_1", "a_2"] = "a_1"
    spin_min: float = 0.0
    spin_max: float

    @model_validator(mode="after")
    def _check_range(self) -> "SpinConfig":
        if self.spin_min < 0:
            raise ValueError("spin_min must be >= 0.")
        if self.spin_max <= self.spin_min:
            raise ValueError("spin_max must be > spin_min.")
        if self.spin_max > 1.0:
            raise ValueError(
                "spin_max must be <= 1.0 (dimensionless spin cannot exceed 1)."
            )
        return self
