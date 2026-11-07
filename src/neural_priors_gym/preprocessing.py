r"""Backend-independent preprocessing of flow inputs, in NumPy.

Every model trained with :class:`~neural_priors_gym.training.trainer.FlowTrainer`
stores a *preprocessing record* under the key ``"preprocessing"`` of its
``model_kwargs.json``, whatever the backend. The flow is trained on
:math:`y = T(x)`, where :math:`x` is the physical point and :math:`T` acts
per dimension in two steps:

1. :math:`u_i = \ln x_i` for the dimensions in ``log_transform``,
   :math:`u_i = \ln(1 - x_i)` for those in ``log1m_transform``, and
   :math:`u_i = x_i` otherwise;
2. :math:`y_i = u_i s_i + m_i`, with :math:`s_i` and :math:`m_i` the
   ``scale`` and ``min`` entries of ``scaler`` (identity if ``scaler`` is
   ``None``).

The physical density is then

.. math::

    \log p_X(x) = \log q(T(x)) + \sum_i \log |s_i|
        - \sum_{i \in \mathrm{log}} u_i - \sum_{i \in \mathrm{log1m}} u_i ,

since :math:`\mathrm{d}\ln x/\mathrm{d}x = 1/x = e^{-u}` and
:math:`|\mathrm{d}\ln(1 - x)/\mathrm{d}x| = 1/(1 - x) = e^{-u}`. The support
is :math:`x_i > 0` for log dimensions and :math:`x_i < 1` for log1m
dimensions.

The record has the keys ``format_version``, ``parameter_names``, ``scaler``
(``None`` or a dict with ``type``, ``scale``, ``min``, ``data_min``,
``data_max``; the last two are the range of the transformed data :math:`u`),
``log_transform``, ``log1m_transform`` and ``bounds`` (``{"min": [...],
"max": [...]}``, the range of the raw training data in physical units).

This module only needs NumPy, so it can be imported without torch or JAX
(for example by the bilby prior).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

PREPROCESSING_KEY = "preprocessing"
FORMAT_VERSION = 1
MODEL_KWARGS_FILE = "model_kwargs.json"

#: Keys that flowjax models before format version 1 stored at the top level.
_RECORD_KEYS = (
    "parameter_names",
    "scaler",
    "log_transform",
    "log1m_transform",
    "bounds",
)


def make_preprocessing_record(
    parameter_names: list[str],
    scaler: dict[str, Any] | None,
    log_transform: list[str],
    log1m_transform: list[str],
    bounds_min: list[float],
    bounds_max: list[float],
) -> dict[str, Any]:
    """Assemble a JSON-serialisable preprocessing record."""
    return {
        "format_version": FORMAT_VERSION,
        "parameter_names": list(parameter_names),
        "scaler": scaler,
        "log_transform": list(log_transform),
        "log1m_transform": list(log1m_transform),
        "bounds": {"min": list(bounds_min), "max": list(bounds_max)},
    }


def load_preprocessing(model_dir: str | Path) -> dict[str, Any] | None:
    """Read the preprocessing record of a model directory.

    Returns the ``"preprocessing"`` entry of ``model_kwargs.json``. flowjax
    models saved before that entry existed kept the same keys at the top
    level; those are returned as a record too. Older torch models (glasflow,
    zuko), whose only preprocessing is a pickled ``scaler.gz``, give ``None``.
    """
    path = Path(model_dir) / MODEL_KWARGS_FILE
    with open(path) as f:
        kwargs = json.load(f)
    record = kwargs.get(PREPROCESSING_KEY)
    if record is not None:
        return record
    if kwargs.get("backend") == "flowjax" and "parameter_names" in kwargs:
        return {"format_version": 0, **{k: kwargs.get(k) for k in _RECORD_KEYS}}
    return None


def _masks(
    meta: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    names = list(meta["parameter_names"])
    n = len(names)
    scaler = meta.get("scaler")
    if scaler is None:
        scale, shift = np.ones(n), np.zeros(n)
    else:
        scale = np.asarray(scaler["scale"], dtype=np.float64)
        shift = np.asarray(scaler["min"], dtype=np.float64)
    log_names = set(meta.get("log_transform") or [])
    log1m_names = set(meta.get("log1m_transform") or [])
    if log_names & log1m_names:
        raise ValueError(
            f"Parameters in both log_transform and log1m_transform: {log_names & log1m_names}"
        )
    unknown = (log_names | log1m_names) - set(names)
    if unknown:
        raise ValueError(
            f"Transformed parameters {unknown} are not in parameter_names {names}"
        )
    log_mask = np.array([name in log_names for name in names])
    log1m_mask = np.array([name in log1m_names for name in names])
    return scale, shift, log_mask, log1m_mask


def in_support(x: np.ndarray, meta: dict[str, Any]) -> np.ndarray:
    """Whether each point lies where the transforms are defined.

    ``x`` has shape ``(..., n_dim)``; the result has shape ``(...)``. NaN
    entries count as outside the support.
    """
    x = np.asarray(x, dtype=np.float64)
    _, _, log_mask, log1m_mask = _masks(meta)
    with np.errstate(invalid="ignore"):
        ok = np.where(log_mask, x > 0, True) & np.where(log1m_mask, x < 1, True)
    return np.all(ok & ~np.isnan(x), axis=-1)


def physical_to_flow(
    x: np.ndarray, meta: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    r"""Map physical points to the flow's space.

    Parameters
    ----------
    x:
        Physical points, shape ``(..., n_dim)``.
    meta:
        The preprocessing record.

    Returns
    -------
    y, log_det:
        The flow-space points and :math:`\log|\det \partial y / \partial x|`,
        computed in float64. Both are NaN at points outside the support
        (see :func:`in_support`).
    """
    x = np.asarray(x, dtype=np.float64)
    scale, shift, log_mask, log1m_mask = _masks(meta)
    with np.errstate(invalid="ignore", divide="ignore"):
        u = np.where(log_mask, np.log(np.where(log_mask, x, 1.0)), x)
        u = np.where(log1m_mask, np.log1p(-np.where(log1m_mask, x, 0.0)), u)
    y = u * scale + shift
    log_det = np.sum(np.log(np.abs(scale))) - np.sum(
        np.where(log_mask | log1m_mask, u, 0.0), axis=-1
    )
    bad = ~in_support(x, meta)
    if np.any(bad):
        y = np.where(bad[..., None], np.nan, y)
        log_det = np.where(bad, np.nan, log_det)
    return y, log_det


def flow_to_physical(y: np.ndarray, meta: dict[str, Any]) -> np.ndarray:
    r"""Map flow-space points back to physical units (float64).

    The inverse of :func:`physical_to_flow`:
    :math:`x = e^u` for log dimensions and :math:`x = 1 - e^u` (computed as
    ``-expm1(u)``) for log1m dimensions.

    In float64, :math:`1 - e^u` rounds to exactly 1 for :math:`u \lesssim -37`,
    and :math:`e^u` underflows to 0 for :math:`u \lesssim -745`. Those values
    are clamped to the largest double below 1 and the smallest positive normal
    double, so every finite ``y`` maps into the support and
    :func:`physical_log_prob` stays finite at every point this function
    returns. For any trained prior these regions carry negligible mass.
    """
    y = np.asarray(y, dtype=np.float64)
    scale, shift, log_mask, log1m_mask = _masks(meta)
    u = (y - shift) / scale
    with np.errstate(over="ignore"):
        x_log = np.maximum(np.exp(u), np.finfo(np.float64).tiny)
        x_log1m = np.minimum(-np.expm1(u), np.nextafter(1.0, 0.0))
    return np.where(log_mask, x_log, np.where(log1m_mask, x_log1m, u))


def physical_log_prob(
    flow_log_prob: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    meta: dict[str, Any],
) -> np.ndarray:
    """Log density in physical units, including all Jacobian terms.

    Parameters
    ----------
    flow_log_prob:
        The flow's log density in flow space, called on an array of shape
        ``(n, n_dim)`` (e.g. :meth:`FlowBase.log_prob`). It is only called on
        points inside the support.
    x:
        Physical points, shape ``(n, n_dim)`` or ``(n_dim,)``.
    meta:
        The preprocessing record.

    Returns
    -------
    np.ndarray
        Shape ``(n,)``; ``-inf`` outside the support. The density is not
        truncated to the training-data ``bounds``.
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    out = np.full(x.shape[0], -np.inf)
    ok = in_support(x, meta)
    if np.any(ok):
        y, log_det = physical_to_flow(x[ok], meta)
        out[ok] = np.asarray(flow_log_prob(y), dtype=np.float64) + log_det
    return out


__all__ = [
    "FORMAT_VERSION",
    "PREPROCESSING_KEY",
    "flow_to_physical",
    "in_support",
    "load_preprocessing",
    "make_preprocessing_record",
    "physical_log_prob",
    "physical_to_flow",
]
