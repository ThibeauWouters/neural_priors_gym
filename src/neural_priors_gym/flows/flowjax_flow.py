"""flowjax rational-quadratic-spline backend (coupling or masked autoregressive)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import paramax

from neural_priors_gym.config.schemas.flow import FlowjaxConfig
from neural_priors_gym.logging_config import get_logger

from .base import FlowBase
from .flowjax_prior import (
    build_flowjax_flow,
    FlowjaxPrior,
    load_flowjax_model,
    make_prior,
    save_flowjax_model,
)


logger = get_logger("neural_priors_gym.flows")


class ArrayBatchLoader:
    """Minimal replacement for a torch ``DataLoader`` over one array.

    Iterating yields batches of ``batch_size`` rows (the last one may be
    smaller, as with ``DataLoader(drop_last=False)``); with ``shuffle=True`` the
    rows are permuted at the start of every pass.
    """

    def __init__(
        self, x: np.ndarray, batch_size: int, shuffle: bool, seed: int = 0
    ) -> None:
        self.x = jnp.asarray(x)
        self.batch_size = min(batch_size, len(x))
        self.shuffle = shuffle
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return int(-(-len(self.x) // self.batch_size))

    def epoch_arrays(self) -> tuple[jax.Array, jax.Array]:
        """Return ``(full_batches, remainder)`` for one pass over the data.

        ``full_batches`` has shape ``(n_full, batch_size, n_dim)``; the
        remainder has between 0 and ``batch_size - 1`` rows.
        """
        x = self.x
        if self.shuffle:
            x = x[self._rng.permutation(len(x))]
        n_full = len(x) // self.batch_size
        cut = n_full * self.batch_size
        full = x[:cut].reshape(n_full, self.batch_size, x.shape[-1])
        return full, x[cut:]

    def __iter__(self) -> Iterator[jax.Array]:
        full, rest = self.epoch_arrays()
        yield from full
        if len(rest):
            yield rest


def _nll(params, static, batch):
    dist = paramax.unwrap(eqx.combine(params, static))
    return -dist.log_prob(batch).mean()


class FlowjaxFlow(FlowBase):
    """Wrapper around a flowjax rational-quadratic-spline flow.

    Like the other backends, :meth:`log_prob`, :meth:`sample` and
    :meth:`latent_to_data` work in the space the flow was trained in (i.e.
    scaled data). Use :meth:`as_prior` or
    :func:`~neural_priors_gym.flows.flowjax_prior.load_flowjax_prior` for a
    pure-JAX prior in physical units.

    Parameters
    ----------
    flow:
        flowjax distribution (as built by :func:`build_flowjax_flow`).
    n_inputs:
        Dimensionality of the data.
    config:
        Flow hyperparameter config.
    preprocessing:
        Parameter names, scaler, log transform and bounds, set by
        :class:`~neural_priors_gym.training.trainer.FlowTrainer`.
    """

    uses_pickled_scaler = False

    def __init__(
        self,
        flow: Any,
        n_inputs: int,
        config: FlowjaxConfig,
        preprocessing: dict[str, Any] | None = None,
    ) -> None:
        self._flow = flow
        self.n_inputs = n_inputs
        self.config = config
        self.preprocessing = preprocessing
        self._key = jr.key(config.seed + 1)
        self._optimizer: optax.GradientTransformation | None = None
        self._opt_state: Any = None
        self._compile()

    # ------------------------------------------------------------------ setup

    def _architecture(self) -> dict[str, Any]:
        c = self.config
        return {
            "flow_type": c.flow_type,
            "n_inputs": self.n_inputs,
            "n_transforms": c.n_transforms,
            "nn_width": c.nn_width,
            "nn_depth": c.nn_depth,
            "num_bins": c.num_bins,
            "tail_bound": c.tail_bound,
            "activation": c.activation,
            "invert": c.invert,
            "seed": c.seed,
            "dtype": c.dtype,
        }

    @classmethod
    def from_config(cls, config: FlowjaxConfig, n_inputs: int) -> FlowjaxFlow:
        """Construct a FlowjaxFlow from a FlowjaxConfig."""
        if config.dtype == "float64" and not jax.config.read("jax_enable_x64"):
            logger.info("Enabling jax_enable_x64 for float64 flowjax training")
            jax.config.update("jax_enable_x64", True)
        instance = cls(None, n_inputs, config)
        instance._flow = build_flowjax_flow(**instance._architecture())
        instance._compile()
        logger.info(
            f"Created flowjax {config.flow_type} with {n_inputs} inputs on "
            f"{jax.default_backend()}"
        )
        return instance

    def _dtype(self) -> Any:
        """Dtype of the training data (the configured training dtype)."""
        return jnp.dtype(self.config.dtype)

    @staticmethod
    def _input_dtype() -> Any:
        """Dtype for inference inputs: JAX's default float dtype.

        With ``jax_enable_x64`` on (jester enables it on import), flowjax
        creates float64 constants internally, so float32 inputs to the
        sampling direction would make ``lax.scan`` carries change dtype.
        """
        return jnp.zeros(()).dtype

    def _compile(self) -> None:
        """(Re)partition the flow into trainable params and static parts."""
        if self._flow is None:
            return
        self._params, self._static = eqx.partition(
            self._flow,
            eqx.is_inexact_array,
            is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
        )

    def _current_flow(self) -> Any:
        return eqx.combine(self._params, self._static)

    def _unwrapped(self) -> Any:
        return paramax.unwrap(self._current_flow())

    # --------------------------------------------------------------- training

    def set_optimizer(self, learning_rate: float) -> None:
        """Initialise the Adam optimizer (called by FlowTrainer before training)."""
        self._optimizer = optax.adam(learning_rate)
        self._opt_state = self._optimizer.init(self._params)
        optimizer, static = self._optimizer, self._static

        @eqx.filter_jit
        def step(params, opt_state, batch):
            loss, grads = eqx.filter_value_and_grad(_nll)(params, static, batch)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            return eqx.apply_updates(params, updates), opt_state, loss

        @eqx.filter_jit
        def scan_epoch(params, opt_state, batches):
            def body(carry, batch):
                p, s = carry
                p, s, loss = step(p, s, batch)
                return (p, s), loss

            (params, opt_state), losses = jax.lax.scan(
                body, (params, opt_state), batches
            )
            return params, opt_state, losses

        self._step = step
        self._scan_epoch = scan_epoch
        self._batch_nll = eqx.filter_jit(lambda p, b: _nll(p, static, b))

    def make_data_loaders(
        self, x_train: np.ndarray, x_val: np.ndarray, batch_size: int
    ) -> tuple[ArrayBatchLoader, ArrayBatchLoader]:
        """Array-backed loaders; training batches are reshuffled every epoch."""
        dtype = self._dtype()
        return (
            ArrayBatchLoader(
                x_train.astype(dtype), batch_size, shuffle=True, seed=self.config.seed
            ),
            ArrayBatchLoader(x_val.astype(dtype), batch_size, shuffle=False),
        )

    def train_epoch(self, data_loader: ArrayBatchLoader) -> float:  # type: ignore[override]
        """One pass over the training data; returns the mean batch NLL."""
        if self._optimizer is None:
            raise RuntimeError("Call set_optimizer before train_epoch")
        full, rest = data_loader.epoch_arrays()
        losses = []
        params, opt_state = self._params, self._opt_state
        if len(full):
            params, opt_state, batch_losses = self._scan_epoch(params, opt_state, full)
            losses.append(np.asarray(batch_losses))
        if len(rest):
            params, opt_state, loss = self._step(params, opt_state, rest)
            losses.append(np.atleast_1d(np.asarray(loss)))
        self._params, self._opt_state = params, opt_state
        return float(np.concatenate(losses).mean())

    def eval_epoch(self, data_loader: ArrayBatchLoader) -> float:  # type: ignore[override]
        """Mean over batches of the batch-mean NLL (as the torch backends)."""
        losses = [float(self._batch_nll(self._params, b)) for b in data_loader]
        return float(np.mean(losses))

    def get_state(self) -> Any:
        """Trainable parameters (immutable JAX arrays, so no copy is needed)."""
        return self._params

    def set_state(self, state: Any) -> None:
        self._params = state
        self._flow = self._current_flow()

    # ---------------------------------------------------------------- inference

    def _next_key(self) -> jax.Array:
        self._key, sub = jr.split(self._key)
        return sub

    def sample(self, n: int) -> np.ndarray:
        return np.asarray(self._unwrapped().sample(self._next_key(), (n,)))

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        xj = jnp.asarray(x, dtype=self._input_dtype())
        return np.asarray(self._unwrapped().log_prob(xj))

    def latent_to_data(self, z: np.ndarray) -> np.ndarray:
        zj = jnp.asarray(z, dtype=self._input_dtype())
        transform = jax.vmap(self._unwrapped().bijection.transform)
        return np.asarray(transform(zj))

    def _metadata(self) -> dict[str, Any]:
        pre = self.preprocessing
        if pre is None:
            pre = {
                "parameter_names": [f"x{i}" for i in range(self.n_inputs)],
                "scaler": None,
                "log_transform": [],
                "bounds": {
                    "min": [-np.inf] * self.n_inputs,
                    "max": [np.inf] * self.n_inputs,
                },
            }
        return {**self._architecture(), **pre}

    def as_prior(self, dtype: Any = None) -> FlowjaxPrior:
        """The current flow as a physical-unit :class:`FlowjaxPrior`."""
        return make_prior(self._current_flow(), self._metadata(), dtype=dtype)

    def save(self, directory: Path) -> None:
        """Save weights (``tree_serialise_leaves``) and JSON metadata."""
        save_flowjax_model(directory, self._current_flow(), self._metadata())
        logger.info(f"Flow saved to {directory}")

    @classmethod
    def load(cls, directory: Path) -> FlowjaxFlow:
        """Load a saved flowjax flow from directory."""
        flow, meta = load_flowjax_model(directory)
        config = FlowjaxConfig(
            **{
                k: meta[k]
                for k in FlowjaxConfig.model_fields
                if k in meta and k != "backend"
            }
        )
        pre_keys = ("parameter_names", "scaler", "log_transform", "bounds")
        preprocessing = {k: meta.get(k) for k in pre_keys}
        logger.info(f"Flow loaded from {directory}")
        return cls(flow, meta["n_inputs"], config, preprocessing)
