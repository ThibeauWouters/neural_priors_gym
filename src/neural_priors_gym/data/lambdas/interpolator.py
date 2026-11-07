"""EOS-based tidal deformability interpolator."""

from pathlib import Path

import numpy as np

from neural_priors_gym.config.schemas.lambdas import LambdaConfig
from neural_priors_gym.logging_config import get_logger

logger = get_logger("neural_priors_gym.data.lambdas")

LAMBDA_CLIP_MIN = 1e-4

#: Location of the M-Lambda curves in a jester ``results.h5`` file
#: (``InferenceResult.save``): each array has shape (n_eos, n_points).
JESTER_DERIVED_GROUP = "posterior/derived_eos"


def load_jester_eos(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read ``masses_EOS`` and ``Lambdas_EOS`` from a jester ``results.h5`` file.

    The file is read directly with h5py, so jester itself is not needed. In
    jester, each row holds the stable branch on a mass grid from the lightest
    star up to MTOV; EOS whose TOV solve failed are stored as rows of zeros.
    """
    import h5py

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Jester result file not found: {path}")
    with h5py.File(path, "r") as f:
        if JESTER_DERIVED_GROUP not in f:
            raise KeyError(
                f"{path} has no '{JESTER_DERIVED_GROUP}' group; is it a jester "
                "inference result with derived EOS quantities?"
            )
        group = f[JESTER_DERIVED_GROUP]
        missing = [k for k in ("masses_EOS", "Lambdas_EOS") if k not in group]  # type: ignore[operator]
        if missing:
            raise KeyError(f"{path}: '{JESTER_DERIVED_GROUP}' lacks {missing}")
        masses = np.asarray(group["masses_EOS"][:])  # type: ignore[index]
        lambdas = np.asarray(group["Lambdas_EOS"][:])  # type: ignore[index]
    return masses, lambdas


def _load_eos_arrays(config: LambdaConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return (masses_EOS, Lambdas_EOS) from either an npz or jester HDF5 file."""
    if config.eos_path is not None:
        path = Path(config.eos_path)
        if not path.exists():
            raise FileNotFoundError(f"EOS npz file not found: {path}")
        logger.info(f"Loading EOS samples from npz: {path}")
        data = np.load(path)
        return data["masses_EOS"], data["Lambdas_EOS"]

    logger.info(f"Loading EOS samples from jester result: {config.jester_path}")
    return load_jester_eos(config.jester_path)  # type: ignore[arg-type]


def _stable_branch(
    masses: np.ndarray, lambdas: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return the curve up to MTOV, or None if the EOS is unusable.

    Rows with non-finite values, non-positive Lambdas (including the all-zero
    rows jester writes for failed EOS) or a mass grid that is not strictly
    increasing up to its maximum are rejected. Points past the maximum mass
    (an unstable branch) are cut off and the stable part is resampled onto the
    original number of points.
    """
    if not (np.all(np.isfinite(masses)) and np.all(np.isfinite(lambdas))):
        return None
    if np.any(lambdas <= 0.0) or masses.max() <= 0.0:
        return None
    i_max = int(np.argmax(masses))
    m, lam = masses[: i_max + 1], lambdas[: i_max + 1]
    if len(m) < 2 or np.any(np.diff(m) <= 0.0):
        return None
    if i_max == len(masses) - 1:
        return masses, lambdas
    grid = np.linspace(m[0], m[-1], len(masses))
    return grid, np.interp(grid, m, lam)


class EOSLambdaInterpolator:
    """Interpolates tidal deformabilities from a set of EOS samples.

    Lambda(m) is obtained by linear interpolation of an EOS's Lambda(M) curve.
    Masses outside the curve get the value at its nearest end.

    Parameters
    ----------
    config:
        Lambda configuration. Either ``eos_path`` (path to a ``.npz`` file)
        or ``jester_path`` (path to a jester ``.h5`` result file) must be set.
    """

    def __init__(self, config: LambdaConfig) -> None:
        masses_raw, lambdas_raw = _load_eos_arrays(config)

        n_total = len(masses_raw)
        kept_m, kept_l = [], []
        for m, lam in zip(masses_raw, lambdas_raw):
            curve = _stable_branch(np.asarray(m, float), np.asarray(lam, float))
            if curve is not None:
                kept_m.append(curve[0])
                kept_l.append(curve[1])

        n_removed = n_total - len(kept_m)
        if n_removed > 0:
            logger.info(
                f"Removed {n_removed}/{n_total} EOS samples with non-finite or "
                "non-positive Lambdas or an invalid mass grid."
            )
        if not kept_m:
            raise ValueError("No valid EOS samples left after filtering.")

        self.masses_EOS: np.ndarray = np.array(kept_m)
        self.lambdas_EOS: np.ndarray = np.array(kept_l)

        # Warn about EOS samples with MTOV < 2.0 Msun
        n_low_mtov = sum(m.max() < 2.0 for m in self.masses_EOS)
        if n_low_mtov > 0:
            logger.warning(
                f"{n_low_mtov}/{len(self.masses_EOS)} EOS samples have MTOV < 2.0 Msun. "
                "These are kept but may produce unphysical mass samples."
            )

        logger.info(f"Loaded {len(self.masses_EOS)} valid EOS samples.")

    @property
    def mtov_array(self) -> np.ndarray:
        """Maximum TOV mass per EOS, shape (n_eos,)."""
        return np.array([m.max() for m in self.masses_EOS])

    @property
    def n_eos(self) -> int:
        """Number of valid EOS samples."""
        return len(self.masses_EOS)

    def interpolate(
        self, masses: np.ndarray, eos_indices: np.ndarray | None = None
    ) -> np.ndarray:
        """Compute Lambda values for an array of masses.

        All masses in a row use the same EOS, ``eos_indices[row]`` (drawn at
        random if omitted). Values are clipped to a minimum of
        LAMBDA_CLIP_MIN for numerical stability.

        Parameters
        ----------
        masses:
            Array of shape (n_samples, k) where k is the number of neutron
            star masses to interpolate (k=2 for BNS, k=1 for NSBH where only
            the NS secondary is passed).
        eos_indices:
            Optional array of shape (n_samples,) with the EOS of each row.

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, k) containing tidal deformabilities.
        """
        n_samples, n_stars = masses.shape
        if eos_indices is None:
            eos_indices = np.random.randint(0, self.n_eos, size=n_samples)
        eos_indices = np.asarray(eos_indices)
        if eos_indices.shape != (n_samples,):
            raise ValueError(
                f"eos_indices must have shape ({n_samples},), got {eos_indices.shape}"
            )

        lambdas = np.empty((n_samples, n_stars))
        for idx in np.unique(eos_indices):
            rows = eos_indices == idx
            lambdas[rows] = np.interp(
                masses[rows], self.masses_EOS[idx], self.lambdas_EOS[idx]
            )

        return np.clip(lambdas, a_min=LAMBDA_CLIP_MIN, a_max=None)
