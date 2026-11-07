"""Zuko Masked Autoregressive Flow (MAF) backend."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from zuko.flows import MAF

from neural_priors_gym.config.schemas.flow import ZukoMAFConfig
from neural_priors_gym.logging_config import get_logger

from .base import FlowBase

logger = get_logger("neural_priors_gym.flows")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_WEIGHTS_FILE = "model.pt"
MODEL_KWARGS_FILE = "model_kwargs.json"


class ZukoMAF(FlowBase):
    """Thin wrapper around zuko's Masked Autoregressive Flow.

    Parameters
    ----------
    flow:
        Underlying zuko MAF instance.
    n_inputs:
        Dimensionality of the data.
    config:
        Flow hyperparameter config (saved with the model for reloading).
    """

    def __init__(self, flow: MAF, n_inputs: int, config: ZukoMAFConfig) -> None:
        self._flow = flow
        self.n_inputs = n_inputs
        self.config = config

    @classmethod
    def from_config(cls, config: ZukoMAFConfig, n_inputs: int) -> "ZukoMAF":
        """Construct a ZukoMAF from a ZukoMAFConfig."""
        flow = MAF(
            features=n_inputs,
            context=0,
            transforms=config.transforms,
            randperm=config.randperm,
            hidden_features=config.hidden_features,
        )
        flow.to(DEVICE)
        logger.info(f"Created ZukoMAF with {n_inputs} inputs on {DEVICE}")
        return cls(flow, n_inputs, config)

    def set_optimizer(self, learning_rate: float) -> None:
        """Initialise the Adam optimizer (called by FlowTrainer before training)."""
        self._optimizer = torch.optim.Adam(self._flow.parameters(), lr=learning_rate)

    def train_epoch(self, data_loader: DataLoader) -> float:
        self._flow.train()
        optimizer = self._optimizer
        total_loss = 0.0
        for (batch,) in data_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            loss = -self._flow().log_prob(batch).mean()
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
                loss = -self._flow().log_prob(batch).mean()
                total_loss += loss.item()
        return total_loss / len(data_loader)

    def sample(self, n: int) -> np.ndarray:
        self._flow.eval()
        with torch.no_grad():
            samples = self._flow().sample((n,))
        return samples.cpu().numpy()

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        self._flow.eval()
        x_tensor = torch.tensor(x, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            lp = self._flow().log_prob(x_tensor)
        return lp.cpu().numpy()

    def latent_to_data(self, z: np.ndarray) -> np.ndarray:
        self._flow.eval()
        z_tensor = torch.tensor(z, dtype=torch.float32).to(DEVICE)
        with torch.inference_mode():
            x = self._flow().transform.inv(z_tensor)
        return x.cpu().numpy()

    def save(self, directory: Path) -> None:
        """Save model weights and hyperparameters to directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        torch.save(self._flow.state_dict(), directory / MODEL_WEIGHTS_FILE)

        kwargs = {
            "backend": "zuko",
            "flow_type": "maf",
            "n_inputs": self.n_inputs,
            "transforms": self.config.transforms,
            "randperm": self.config.randperm,
            "hidden_features": self.config.hidden_features,
        }
        with open(directory / MODEL_KWARGS_FILE, "w") as f:
            json.dump(self._add_preprocessing(kwargs), f, indent=4)

        logger.info(f"Flow saved to {directory}")

    @classmethod
    def load(cls, directory: Path) -> "ZukoMAF":
        """Load a saved flow from directory."""
        directory = Path(directory)
        kwargs_path = directory / MODEL_KWARGS_FILE
        if not kwargs_path.exists():
            raise FileNotFoundError(f"Model kwargs not found: {kwargs_path}")

        with open(kwargs_path) as f:
            kwargs = json.load(f)

        config = ZukoMAFConfig(
            transforms=kwargs["transforms"],
            randperm=kwargs.get("randperm", False),
            hidden_features=kwargs["hidden_features"],
        )
        n_inputs = kwargs["n_inputs"]

        instance = cls.from_config(config, n_inputs)
        instance._flow.load_state_dict(
            torch.load(directory / MODEL_WEIGHTS_FILE, map_location=DEVICE)
        )
        instance._flow.eval()
        instance.preprocessing = kwargs.get("preprocessing")
        logger.info(f"Flow loaded from {directory}")
        return instance
