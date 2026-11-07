"""Abstract base class for normalizing flow backends."""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


class FlowBase(ABC):
    """Common interface for all normalizing flow backends.

    Each backend (glasflow, zuko, etc.) provides a concrete subclass that
    wraps the backend-specific flow object and implements this API.

    Attributes
    ----------
    _flow:
        The backend-specific flow model object (e.g. a glasflow
        ``CouplingNSF`` or a zuko ``MAF``).  Set by each subclass's
        ``__init__``.
    n_inputs:
        Dimensionality of the data the flow was built for.  Set by each
        subclass's ``__init__``.
    config:
        The hyperparameter config object used to construct the flow.  Set by
        each subclass's ``__init__``.
    """

    _flow: Any
    n_inputs: int
    config: Any

    #: Whether the trainer should pickle the fitted ``MinMaxScaler`` next to
    #: the model (``scaler.gz``). Backends that store the scaler themselves
    #: (flowjax) set this to ``False``.
    uses_pickled_scaler: bool = True

    def make_data_loaders(
        self, x_train: np.ndarray, x_val: np.ndarray, batch_size: int
    ) -> tuple[Any, Any]:
        """Build the training and validation loaders passed to the epoch methods.

        The default returns torch ``DataLoader`` objects (shuffled for training).
        """
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        train = TensorDataset(torch.tensor(x_train, dtype=torch.float32))
        val = TensorDataset(torch.tensor(x_val, dtype=torch.float32))
        return (
            DataLoader(train, batch_size=batch_size, shuffle=True),
            DataLoader(val, batch_size=batch_size, shuffle=False),
        )

    def get_state(self) -> Any:
        """Return a copy of the trainable state, used to keep the best epoch."""
        import copy

        return copy.deepcopy(self._flow.state_dict())

    def set_state(self, state: Any) -> None:
        """Restore a state returned by :meth:`get_state`."""
        self._flow.load_state_dict(state)

    @abstractmethod
    def set_optimizer(self, learning_rate: float) -> None:
        """Initialise the optimizer (called by FlowTrainer before training)."""
        ...

    @abstractmethod
    def train_epoch(self, data_loader: "DataLoader") -> float:
        """Run one training epoch and return the mean negative log-likelihood."""
        ...

    @abstractmethod
    def eval_epoch(self, data_loader: "DataLoader") -> float:
        """Evaluate the flow on a data loader and return the mean NLL."""
        ...

    @abstractmethod
    def sample(self, n: int) -> np.ndarray:
        """Draw n samples from the flow. Returns array of shape (n, n_inputs)."""
        ...

    @abstractmethod
    def log_prob(self, x: np.ndarray) -> np.ndarray:
        """Compute log probabilities for x. Returns array of shape (len(x),)."""
        ...

    @abstractmethod
    def latent_to_data(self, z: np.ndarray) -> np.ndarray:
        """Map samples from the standard normal latent space to data space.

        This is the generative direction of the flow (inverse of the normalising
        transform).  It is the operation needed by nested samplers such as
        dynesty to rescale from a unit hypercube through a standard normal into
        the parameter space.

        Parameters
        ----------
        z:
            Samples from the standard normal latent space, shape
            ``(n, n_inputs)``.

        Returns
        -------
        np.ndarray
            Corresponding samples in data space, shape ``(n, n_inputs)``.
        """
        ...

    @abstractmethod
    def save(self, directory: Path) -> None:
        """Save the flow weights and metadata to directory."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, directory: Path) -> "FlowBase":
        """Load a saved flow from directory."""
        ...

    @classmethod
    def load_from_dir(cls, directory: Path) -> "FlowBase":
        r"""Load a flow from a model directory by dispatching on the backend field.

        Reads ``model_kwargs.json``, determines the ``backend`` value, and calls
        the appropriate subclass ``load()`` method.

        Parameters
        ----------
        directory:
            Path to the model directory produced by :class:`FlowTrainer`.

        Returns
        -------
        FlowBase
            The loaded flow instance.

        Raises
        ------
        FileNotFoundError
            If ``model_kwargs.json`` is absent from *directory*.
        ValueError
            If ``model_kwargs.json`` does not contain a ``backend`` key.
        NotImplementedError
            If the backend value is not registered.

        Notes
        -----
        **Adding a new backend**: register the new subclass in the
        ``_BACKEND_MODULES`` dict inside this method.  Failing to do so will
        cause ``load_from_dir`` to raise :exc:`NotImplementedError` for any
        model saved with the new backend.
        """
        directory = Path(directory)
        kwargs_path = directory / "model_kwargs.json"
        if not kwargs_path.exists():
            raise FileNotFoundError(f"model_kwargs.json not found in {directory}")

        with open(kwargs_path) as f:
            model_kwargs = json.load(f)

        backend = model_kwargs.get("backend")
        if backend is None:
            raise ValueError(
                f"'backend' key missing from model_kwargs.json in {directory}"
            )

        # Imported lazily to avoid circular imports at module load time.
        # IMPORTANT: register every new backend class here.
        _BACKEND_MODULES: dict[str, tuple[str, str]] = {
            "glasflow": ("neural_priors_gym.flows.glasflow", "GlasflowNSF"),
            "zuko": ("neural_priors_gym.flows.zuko_maf", "ZukoMAF"),
            "flowjax": ("neural_priors_gym.flows.flowjax_flow", "FlowjaxFlow"),
        }

        if backend not in _BACKEND_MODULES:
            raise NotImplementedError(
                f"Backend '{backend}' is not registered in FlowBase.load_from_dir. "
                f"Supported backends: {list(_BACKEND_MODULES.keys())}. "
                "To add a new backend, register its class in the _BACKEND_MODULES "
                "dict inside FlowBase.load_from_dir in "
                "neural_priors_gym/flows/base.py."
            )

        # Import only the backend that is needed, so e.g. loading a flowjax
        # model does not require torch.
        import importlib

        module_name, class_name = _BACKEND_MODULES[backend]
        flow_cls: type[FlowBase] = getattr(
            importlib.import_module(module_name), class_name
        )
        return flow_cls.load(directory)

    @staticmethod
    def compute_log_jacobian_correction(scaler) -> float:
        r"""Compute the log-determinant Jacobian correction for a MinMaxScaler.

        When data is scaled via :math:`x'_i = (x_i - x_{\min,i}) /
        (x_{\max,i} - x_{\min,i})`, the per-dimension Jacobian factor is
        :math:`1 / (x_{\max,i} - x_{\min,i})`, giving

        .. math::

            \log |\det J| = -\sum_i \log(x_{\max,i} - x_{\min,i}).

        This correction must be added to the flow log-probability when the flow
        was trained on scaled data and the log-prob is evaluated in the original
        (unscaled) parameter space.

        Parameters
        ----------
        scaler:
            A fitted ``sklearn.preprocessing.MinMaxScaler`` instance with
            ``data_max_`` and ``data_min_`` attributes populated.

        Returns
        -------
        float
            The log-determinant Jacobian correction.
        """
        param_ranges = scaler.data_max_ - scaler.data_min_
        return float(-np.sum(np.log(param_ranges)))
