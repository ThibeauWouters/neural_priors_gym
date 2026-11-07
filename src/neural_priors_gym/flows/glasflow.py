"""Glasflow CouplingNSF backend."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from glasflow.flows.nsf import CouplingNSF
from torch.utils.data import DataLoader

from neural_priors_gym.config.schemas.flow import GlasflowConfig
from neural_priors_gym.logging_config import get_logger

from .base import FlowBase

logger = get_logger("neural_priors_gym.flows")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_ACTIVATION_MAP = {
    "relu": F.relu,
    "gelu": F.gelu,
    "silu": F.silu,
    "tanh": torch.tanh,
}

MODEL_WEIGHTS_FILE = "model.pt"
MODEL_KWARGS_FILE = "model_kwargs.json"


class GlasflowNSF(FlowBase):
    """Thin wrapper around glasflow CouplingNSF.

    Parameters
    ----------
    flow:
        Underlying glasflow CouplingNSF instance.
    n_inputs:
        Dimensionality of the data.
    config:
        Flow hyperparameter config (saved with the model for reloading).
    """

    def __init__(
        self, flow: CouplingNSF, n_inputs: int, config: GlasflowConfig
    ) -> None:
        self._flow = flow
        self.n_inputs = n_inputs
        self.config = config

    @classmethod
    def from_config(cls, config: GlasflowConfig, n_inputs: int) -> "GlasflowNSF":
        """Construct a GlasflowNSF from a GlasflowConfig."""
        activation = _ACTIVATION_MAP.get(config.activation, F.relu)
        linear_transform = (
            None if config.linear_transform == "none" else config.linear_transform
        )

        flow = CouplingNSF(
            n_inputs=n_inputs,
            n_transforms=config.n_transforms,
            n_neurons=config.n_neurons,
            n_blocks_per_transform=config.n_blocks_per_transform,
            num_bins=config.num_bins,
            tail_bound=config.tail_bound,
            activation=activation,
            linear_transform=linear_transform,
            dropout_probability=config.dropout_probability,
            batch_norm_within_blocks=config.batch_norm_within_blocks,
            batch_norm_between_transforms=config.batch_norm_between_transforms,
        )
        flow.to(DEVICE)
        logger.info(f"Created CouplingNSF with {n_inputs} inputs on {DEVICE}")
        return cls(flow, n_inputs, config)

    def train_epoch(self, data_loader: DataLoader) -> float:
        self._flow.train()
        optimizer = self._optimizer
        total_loss = 0.0
        for (batch,) in data_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            loss = -self._flow.log_prob(inputs=batch).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(data_loader)

    def eval_epoch(self, data_loader: DataLoader) -> float:
        self._flow.eval()
        total_loss = 0.0
        with torch.no_grad():
            for (batch,) in data_loader:
                batch = batch.to(DEVICE)
                loss = -self._flow.log_prob(inputs=batch).mean()
                total_loss += loss.item()
        return total_loss / len(data_loader)

    def set_optimizer(self, learning_rate: float) -> None:
        """Initialise the Adam optimizer (called by FlowTrainer before training)."""
        self._optimizer = torch.optim.Adam(self._flow.parameters(), lr=learning_rate)

    def sample(self, n: int) -> np.ndarray:
        self._flow.eval()
        with torch.no_grad():
            samples = self._flow.sample(n)
        return samples.cpu().numpy()

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        self._flow.eval()
        x_tensor = torch.tensor(x, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            lp = self._flow.log_prob(inputs=x_tensor)
        return lp.cpu().numpy()

    def latent_to_data(self, z: np.ndarray) -> np.ndarray:
        self._flow.eval()
        z_tensor = torch.tensor(z, dtype=torch.float32).to(DEVICE)
        with torch.inference_mode():
            x, _ = self._flow.inverse(z_tensor)
        return x.cpu().numpy()

    def save(self, directory: Path) -> None:
        """Save model weights and hyperparameters to directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        weights_path = directory / MODEL_WEIGHTS_FILE
        torch.save(self._flow.state_dict(), weights_path)

        kwargs = {
            "backend": "glasflow",
            "flow_type": self.config.flow_type,
            "n_inputs": self.n_inputs,
            "n_transforms": self.config.n_transforms,
            "n_neurons": self.config.n_neurons,
            "n_blocks_per_transform": self.config.n_blocks_per_transform,
            "num_bins": self.config.num_bins,
            "tail_bound": self.config.tail_bound,
            "activation": self.config.activation,
            "linear_transform": self.config.linear_transform,
            "dropout_probability": self.config.dropout_probability,
            "batch_norm_within_blocks": self.config.batch_norm_within_blocks,
            "batch_norm_between_transforms": self.config.batch_norm_between_transforms,
        }
        with open(directory / MODEL_KWARGS_FILE, "w") as f:
            json.dump(self._add_preprocessing(kwargs), f, indent=4)

        logger.info(f"Flow saved to {directory}")

    @classmethod
    def load(cls, directory: Path) -> "GlasflowNSF":
        """Load a saved flow from directory."""
        directory = Path(directory)
        kwargs_path = directory / MODEL_KWARGS_FILE
        if not kwargs_path.exists():
            raise FileNotFoundError(f"Model kwargs not found: {kwargs_path}")

        with open(kwargs_path) as f:
            kwargs = json.load(f)

        config = GlasflowConfig(
            backend="glasflow",
            flow_type=kwargs.get("flow_type", "CouplingNSF"),
            n_transforms=kwargs["n_transforms"],
            n_neurons=kwargs["n_neurons"],
            n_blocks_per_transform=kwargs["n_blocks_per_transform"],
            num_bins=kwargs["num_bins"],
            tail_bound=kwargs.get("tail_bound", 5.0),
            activation=kwargs.get("activation", "relu"),
            linear_transform=kwargs.get("linear_transform", "none"),
            dropout_probability=kwargs.get("dropout_probability", 0.0),
            batch_norm_within_blocks=kwargs.get("batch_norm_within_blocks", False),
            batch_norm_between_transforms=kwargs.get(
                "batch_norm_between_transforms", False
            ),
        )
        n_inputs = kwargs["n_inputs"]

        instance = cls.from_config(config, n_inputs)
        weights_path = directory / MODEL_WEIGHTS_FILE
        instance._flow.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        instance._flow.eval()
        instance.preprocessing = kwargs.get("preprocessing")
        logger.info(f"Flow loaded from {directory}")
        return instance
