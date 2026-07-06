"""Shared helper for sampling detector-frame chirp mass and mass ratio."""

import numpy as np


def sample_component_masses(
    mc_min: float, mc_max: float, q_min: float, q_max: float, n_samples: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample detector-frame component masses uniform in component mass.

    Draws chirp mass and mass ratio from bilby's
    ``UniformInComponentsChirpMass`` and ``UniformInComponentsMassRatio``
    priors, which together are equivalent to a uniform-in-component-mass
    prior, then converts to component masses.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (mass_1, mass_2, chirp_mass, mass_ratio), all detector-frame,
        with mass_1 >= mass_2.
    """
    from bilby.gw.conversion import chirp_mass_and_mass_ratio_to_component_masses
    from bilby.gw.prior import (
        UniformInComponentsChirpMass,
        UniformInComponentsMassRatio,
    )

    chirp_mass_prior = UniformInComponentsChirpMass(minimum=mc_min, maximum=mc_max)
    mass_ratio_prior = UniformInComponentsMassRatio(minimum=q_min, maximum=q_max)
    chirp_mass = np.asarray(chirp_mass_prior.sample(n_samples))
    mass_ratio = np.asarray(mass_ratio_prior.sample(n_samples))

    mass_1, mass_2 = chirp_mass_and_mass_ratio_to_component_masses(
        chirp_mass, mass_ratio
    )
    return np.asarray(mass_1), np.asarray(mass_2), chirp_mass, mass_ratio
