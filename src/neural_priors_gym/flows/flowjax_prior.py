r"""Standalone loader for flowjax neural priors, usable with only the JAX stack.

This module depends only on ``numpy``, ``jax``, ``equinox`` and ``flowjax``. It
does not import torch, scikit-learn, bilby, glasflow, zuko or jester, so a
trained prior can be used inside a pure-JAX sampler:

.. code-block:: python

    import jax
    jax.config.update("jax_enable_x64", True)
    from neural_priors_gym.flows.flowjax_prior import load_flowjax_prior

    prior = load_flowjax_prior("outdir/model")
    prior.names            # ("chirp_mass_source", "mass_ratio", "lambda_1", "lambda_2")
    prior.bounds           # (n_dim, 2) training-data min and max, physical units
    log_p = jax.jit(jax.vmap(prior.log_prob))(x)   # x: (n, n_dim), physical units
    samples = prior.sample(jax.random.key(0), 1000)

The flow is trained on preprocessed data :math:`y = T(x)`, where :math:`T` first
takes the logarithm of selected dimensions (``log_transform``) and then applies
the min-max scaling :math:`y_i = u_i s_i + m_i` with the scikit-learn
``MinMaxScaler`` parameters ``scale_`` (:math:`s_i`) and ``min_``
(:math:`m_i`). The physical density is

.. math::

    \log p_X(x) = \log q(T(x)) + \sum_i \log s_i - \sum_{i \in \mathrm{log}} \log x_i .

The model directory holds ``model_kwargs.json`` (hyperparameters, parameter
names, preprocessing and bounds) and ``flowjax_weights.eqx`` (the
``equinox.tree_serialise_leaves`` output). The flow skeleton is rebuilt from the
hyperparameters with :func:`build_flowjax_flow`, so the same flowjax major
version should be used for training and loading.
"""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import paramax
from flowjax.bijections import RationalQuadraticSpline
from flowjax.distributions import AbstractDistribution, StandardNormal, Transformed
from flowjax.flows import coupling_flow, masked_autoregressive_flow


WEIGHTS_FILE = "flowjax_weights.eqx"
MODEL_KWARGS_FILE = "model_kwargs.json"

FLOW_TYPES = ("coupling_nsf", "maf_nsf")

_ACTIVATIONS = {
    "relu": jnn.relu,
    "gelu": jnn.gelu,
    "silu": jnn.silu,
    "tanh": jnp.tanh,
}

#: Keys of ``model_kwargs.json`` that define the flow architecture.
ARCHITECTURE_KEYS = (
    "flow_type",
    "n_inputs",
    "n_transforms",
    "nn_width",
    "nn_depth",
    "num_bins",
    "tail_bound",
    "activation",
    "invert",
    "seed",
    "dtype",
)


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _resolve_dtype(dtype: str | None) -> Any:
    """Return the requested float dtype, defaulting to JAX's default float."""
    if dtype is None:
        return jnp.zeros(()).dtype  # float64 if jax_enable_x64, else float32
    if dtype == "float64" and not jax.config.read("jax_enable_x64"):
        raise ValueError(
            "dtype='float64' requires jax_enable_x64; call "
            "jax.config.update('jax_enable_x64', True) first."
        )
    return jnp.dtype(dtype)


def cast_floats(tree: Any, dtype: Any) -> Any:
    """Cast every floating-point array leaf of a pytree to ``dtype``."""

    def _cast(leaf):
        if eqx.is_array(leaf) and jnp.issubdtype(leaf.dtype, jnp.floating):
            return jnp.asarray(leaf, dtype=dtype)
        return leaf

    return jax.tree_util.tree_map(_cast, tree)


def build_flowjax_flow(
    *,
    n_inputs: int,
    flow_type: str = "coupling_nsf",
    n_transforms: int = 4,
    nn_width: int = 256,
    nn_depth: int = 3,
    num_bins: int = 8,
    tail_bound: float = 5.0,
    activation: str = "relu",
    invert: bool = True,
    seed: int = 0,
    dtype: str = "float32",
) -> Transformed:
    """Construct an (untrained) flowjax rational-quadratic-spline flow.

    The same function builds the flow for training and the skeleton into which
    saved weights are deserialised, so both always have the same structure.

    Parameters
    ----------
    n_inputs:
        Dimensionality of the data.
    flow_type:
        ``"coupling_nsf"`` (coupling layers, as glasflow's ``CouplingNSF``) or
        ``"maf_nsf"`` (masked autoregressive layers).
    n_transforms:
        Number of flow layers.
    nn_width, nn_depth:
        Width and number of hidden layers of each conditioner MLP.
    num_bins:
        Number of spline bins (knots) of the rational quadratic spline.
    tail_bound:
        The spline acts on :math:`[-B, B]` and is the identity outside.
    activation:
        Conditioner activation: ``"relu"``, ``"gelu"``, ``"silu"`` or ``"tanh"``.
    invert:
        flowjax's ``invert`` flag. ``True`` makes ``log_prob`` a single pass,
        which is what a sampler evaluating the prior density needs.
    seed:
        Seed for the weight initialisation and the permutations.
    dtype:
        ``"float32"`` or ``"float64"`` (the latter needs ``jax_enable_x64``).
    """
    if flow_type not in FLOW_TYPES:
        raise ValueError(f"Unknown flow_type {flow_type!r}; choose from {FLOW_TYPES}")
    if activation not in _ACTIVATIONS:
        raise ValueError(
            f"Unknown activation {activation!r}; choose from {list(_ACTIVATIONS)}"
        )
    _resolve_dtype(dtype)
    builder = (
        coupling_flow if flow_type == "coupling_nsf" else masked_autoregressive_flow
    )
    flow = builder(
        jr.key(seed),
        base_dist=StandardNormal((n_inputs,)),
        transformer=RationalQuadraticSpline(knots=num_bins, interval=tail_bound),
        flow_layers=n_transforms,
        nn_width=nn_width,
        nn_depth=nn_depth,
        nn_activation=_ACTIVATIONS[activation],
        invert=invert,
    )
    return cast_floats(flow, jnp.dtype(dtype))


class FlowjaxPrior(eqx.Module):
    r"""A trained flowjax flow with its preprocessing, in physical units.

    The object is an ``equinox.Module`` (a pytree), so its methods can be
    jitted and vmapped, and the prior itself can be passed as an argument to a
    jitted function.

    Attributes
    ----------
    flow:
        The unwrapped flowjax distribution over the preprocessed data.
    scale, shift:
        Per-dimension affine scaler, :math:`y = u \cdot \mathrm{scale} +
        \mathrm{shift}`, with :math:`u` the (partly log-transformed) data.
    bounds:
        Array of shape ``(n_dim, 2)`` with the minimum and maximum of the
        training data in physical units.
    names:
        Parameter names, in column order.
    log_mask:
        Per dimension, whether the logarithm is taken before scaling.
    """

    flow: Transformed
    scale: jax.Array
    shift: jax.Array
    bounds: jax.Array
    names: tuple[str, ...] = eqx.field(static=True)
    log_mask: tuple[bool, ...] = eqx.field(static=True)

    def __hash__(self) -> int:
        # Equinox hashes modules by their fields, which fails for array
        # fields. Hashing by identity lets ``jax.jit(prior.log_prob)`` work
        # (jit hashes the bound method); equal priors then only miss the cache.
        return id(self)

    @property
    def n_dim(self) -> int:
        """Number of parameters."""
        return len(self.names)

    def _log_mask_array(self) -> jax.Array:
        return jnp.asarray(self.log_mask, dtype=bool)

    def physical_to_flow(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        r"""Map physical ``x`` (shape ``(..., n_dim)``) to the flow's space.

        Returns
        -------
        y, log_det:
            The preprocessed point and :math:`\log|\det \partial y/\partial x|`.
        """
        x = jnp.asarray(x, dtype=self.scale.dtype)
        mask = self._log_mask_array()
        safe = jnp.where(mask, x, 1.0)
        u = jnp.where(mask, jnp.log(safe), x)
        y = u * self.scale + self.shift
        log_det = jnp.sum(jnp.log(self.scale)) - jnp.sum(
            jnp.where(mask, u, 0.0), axis=-1
        )
        return y, log_det

    def flow_to_physical(self, y: jax.Array) -> jax.Array:
        """Map points from the flow's space back to physical units."""
        u = (y - self.shift) / self.scale
        return jnp.where(self._log_mask_array(), jnp.exp(u), u)

    def log_prob(self, x: jax.Array) -> jax.Array:
        """Log density in physical units, including all Jacobian terms.

        ``x`` has shape ``(n_dim,)`` or ``(..., n_dim)``. Points with a
        non-positive value in a log-transformed dimension get ``-inf``. The
        density is not truncated to :attr:`bounds`.
        """
        y, log_det = self.physical_to_flow(x)
        lp = self.flow.log_prob(y) + log_det
        x = jnp.asarray(x, dtype=self.scale.dtype)
        valid = jnp.all(jnp.where(self._log_mask_array(), x > 0, True), axis=-1)
        return jnp.where(valid, lp, -jnp.inf)

    def sample(self, key: jax.Array, n: int) -> jax.Array:
        """Draw ``n`` samples in physical units, shape ``(n, n_dim)``."""
        return self.flow_to_physical(self.flow.sample(key, (n,)))

    def latent_to_physical(self, z: jax.Array) -> jax.Array:
        """Map standard-normal latent points ``z`` to physical units."""
        z = jnp.asarray(z, dtype=self.scale.dtype)
        transform = jnp.vectorize(self.flow.bijection.transform, signature="(d)->(d)")
        return self.flow_to_physical(transform(z))


def make_prior(
    flow: AbstractDistribution,
    metadata: dict[str, Any],
    dtype: Any = None,
) -> FlowjaxPrior:
    """Assemble a :class:`FlowjaxPrior` from a flow and its saved metadata.

    Parameters
    ----------
    flow:
        The (possibly paramax-wrapped) flowjax distribution.
    metadata:
        The contents of ``model_kwargs.json``.
    dtype:
        Float dtype for all arrays. Defaults to JAX's default float dtype
        (float64 when ``jax_enable_x64`` is on).
    """
    dtype = _resolve_dtype(None) if dtype is None else jnp.dtype(dtype)
    names = tuple(metadata["parameter_names"])
    n = len(names)
    scaler = metadata.get("scaler")
    if scaler is None:
        scale, shift = np.ones(n), np.zeros(n)
    else:
        scale = np.asarray(scaler["scale"], dtype=np.float64)
        shift = np.asarray(scaler["min"], dtype=np.float64)
    log_names = set(metadata.get("log_transform", []))
    bounds = np.column_stack(
        [np.asarray(metadata["bounds"]["min"]), np.asarray(metadata["bounds"]["max"])]
    )
    return FlowjaxPrior(
        flow=cast_floats(paramax.unwrap(flow), dtype),
        scale=jnp.asarray(scale, dtype=dtype),
        shift=jnp.asarray(shift, dtype=dtype),
        bounds=jnp.asarray(bounds, dtype=dtype),
        names=names,
        log_mask=tuple(name in log_names for name in names),
    )


def save_flowjax_model(
    directory: str | Path,
    flow: AbstractDistribution,
    metadata: dict[str, Any],
) -> None:
    """Write ``model_kwargs.json`` and the serialised weights to ``directory``.

    ``metadata`` must contain the architecture keys (:data:`ARCHITECTURE_KEYS`),
    ``parameter_names``, ``scaler`` (or ``None``), ``log_transform`` and
    ``bounds``. The backend name and library versions are added here.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    missing = [
        k
        for k in (*ARCHITECTURE_KEYS, "parameter_names", "bounds")
        if k not in metadata
    ]
    if missing:
        raise ValueError(f"metadata is missing keys: {missing}")
    meta = {
        "backend": "flowjax",
        **metadata,
        "versions": {
            "flowjax": _package_version("flowjax"),
            "equinox": _package_version("equinox"),
            "jax": _package_version("jax"),
        },
    }
    with open(directory / MODEL_KWARGS_FILE, "w") as f:
        json.dump(meta, f, indent=4)
    eqx.tree_serialise_leaves(directory / WEIGHTS_FILE, flow)


def load_flowjax_model(
    directory: str | Path,
) -> tuple[AbstractDistribution, dict[str, Any]]:
    """Load the (wrapped, as trained) flow and its metadata from ``directory``."""
    directory = Path(directory)
    kwargs_path = directory / MODEL_KWARGS_FILE
    if not kwargs_path.exists():
        raise FileNotFoundError(f"{MODEL_KWARGS_FILE} not found in {directory}")
    with open(kwargs_path) as f:
        meta = json.load(f)
    if meta.get("backend") != "flowjax":
        raise ValueError(
            f"{kwargs_path} has backend {meta.get('backend')!r}, expected 'flowjax'"
        )
    _check_flowjax_version(meta, kwargs_path)
    arch = {k: meta[k] for k in ARCHITECTURE_KEYS}
    # The skeleton is built in float32 if x64 is off; leaves are cast later.
    if arch["dtype"] == "float64" and not jax.config.read("jax_enable_x64"):
        arch["dtype"] = "float32"
    skeleton = build_flowjax_flow(**arch)
    flow = eqx.tree_deserialise_leaves(
        directory / WEIGHTS_FILE, skeleton, filter_spec=_deserialise_as_skeleton
    )
    return flow, meta


def _check_flowjax_version(meta: dict[str, Any], kwargs_path: Path) -> None:
    """Refuse to load weights saved with a different flowjax major version.

    The spline parametrisation changes between flowjax majors (e.g. 17 and 19
    use different numbers of spline parameters per dimension), so the weights
    of one cannot be loaded into the flow skeleton of the other.
    """
    saved = (meta.get("versions") or {}).get("flowjax")
    installed = _package_version("flowjax")
    if saved is None or installed is None:
        return
    if saved.split(".")[0] != installed.split(".")[0]:
        raise RuntimeError(
            f"{kwargs_path} was saved with flowjax {saved}, but flowjax "
            f"{installed} is installed. Install a matching major version, e.g. "
            f"pip install 'flowjax=={saved}'."
        )


def _deserialise_as_skeleton(f: Any, like: Any) -> Any:
    """Load a leaf and cast it to the skeleton's dtype.

    Integer leaves (permutations) are saved as int32 or int64 depending on
    ``jax_enable_x64`` at save time, so the on-disk dtype can differ from the
    skeleton built at load time. Float leaves are cast to the requested dtype
    later, in :func:`make_prior`.
    """
    if isinstance(like, jax.Array):
        return jnp.asarray(np.load(f), dtype=like.dtype)
    return eqx.default_deserialise_filter_spec(f, like)


def load_flowjax_prior(directory: str | Path, dtype: Any = None) -> FlowjaxPrior:
    """Load a trained flowjax prior that evaluates in physical units.

    Parameters
    ----------
    directory:
        Model directory written by the flowjax backend (``outdir/model``).
    dtype:
        Float dtype of the returned prior. Defaults to JAX's default float
        dtype, i.e. float64 when ``jax_enable_x64`` is on.
    """
    flow, meta = load_flowjax_model(directory)
    return make_prior(flow, meta, dtype=dtype)


__all__ = [
    "FlowjaxPrior",
    "build_flowjax_flow",
    "load_flowjax_model",
    "load_flowjax_prior",
    "make_prior",
    "save_flowjax_model",
]
