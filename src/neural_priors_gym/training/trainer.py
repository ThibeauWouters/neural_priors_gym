"""Flow trainer: handles the training loop, early stopping, and scaling."""

from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

from neural_priors_gym.config.schema import TrainingConfig
from neural_priors_gym.config.schemas.flow import (
    FlowjaxConfig,
    GlasflowConfig,
    ZukoMAFConfig,
)
from neural_priors_gym.flows.base import FlowBase
from neural_priors_gym.logging_config import get_logger

logger = get_logger("neural_priors_gym.training")

SCALER_FILE = "scaler.gz"


class FlowTrainer:
    """Trains a normalizing flow on tabular training data.

    Parameters
    ----------
    config:
        Top-level training configuration.
    """

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.training_cfg = config.training
        self.output_dir = Path(config.output_dir)

    def build_flow(self, n_inputs: int) -> FlowBase:
        """Construct a flow from the config."""
        flow_config = self.config.flow
        if isinstance(flow_config, GlasflowConfig):
            from neural_priors_gym.flows.glasflow import GlasflowNSF

            return GlasflowNSF.from_config(flow_config, n_inputs)
        if isinstance(flow_config, ZukoMAFConfig):
            from neural_priors_gym.flows.zuko_maf import ZukoMAF

            return ZukoMAF.from_config(flow_config, n_inputs)
        if isinstance(flow_config, FlowjaxConfig):
            from neural_priors_gym.flows.flowjax_flow import FlowjaxFlow

            return FlowjaxFlow.from_config(flow_config, n_inputs)
        raise ValueError(f"Unknown flow config type: {type(flow_config)}")

    def train(
        self,
        data: dict[str, np.ndarray],
        parameter_names: list[str],
        output_dir: Optional[Path] = None,
    ) -> tuple[FlowBase, list[float], list[float], Optional[MinMaxScaler]]:
        """Train a flow on the provided data.

        Parameters
        ----------
        data:
            Dictionary mapping parameter names to arrays of shape (n_samples,).
        parameter_names:
            Ordered list of keys to extract from data (defines column order).
        output_dir:
            Directory to save the scaler. Defaults to self.output_dir.

        Returns
        -------
        flow:
            Trained flow (best checkpoint by validation loss).
        train_losses:
            Mean NLL per epoch for the training set.
        val_losses:
            Mean NLL per epoch for the validation set.
        scaler:
            Fitted MinMaxScaler if scale_input is True, else None.
        """
        cfg = self.training_cfg
        save_dir = output_dir or self.output_dir
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Apply log transform to lambda columns in-memory before training.
        # The saved training_data.npz retains raw values; this transform is
        # only visible to the flow and scaler.
        _LAMBDA_NAMES = {"lambda_1", "lambda_2", "lambda_tilde", "delta_lambda_tilde"}
        raw_data = data
        lambda_cols: list[str] = []
        if cfg.log_lambda:
            data = dict(data)
            lambda_cols = [n for n in parameter_names if n in _LAMBDA_NAMES]
            for name in lambda_cols:
                data[name] = np.log(data[name])
            logger.info(f"Applied log transform to lambda parameters: {lambda_cols}")
        log1m_cols: list[str] = []
        if cfg.log_one_minus_q and "mass_ratio" in parameter_names:
            data = dict(data)
            q = np.asarray(data["mass_ratio"], dtype=np.float64)
            if np.any(q >= 1):
                raise ValueError("log_one_minus_q needs mass_ratio < 1 in the training data")
            data["mass_ratio"] = np.log1p(-q)
            log1m_cols = ["mass_ratio"]
            logger.info("Applied ln(1 - q) transform to mass_ratio")

        x_full = np.column_stack([data[name] for name in parameter_names])
        n_inputs = x_full.shape[1]
        flow = self.build_flow(n_inputs)

        x_train, x_val = train_test_split(
            x_full,
            test_size=cfg.validation_split,
            random_state=42,
        )

        scaler: Optional[MinMaxScaler | StandardScaler] = None
        x_combined = np.vstack([x_train, x_val])
        if cfg.scale_input:
            scaler = StandardScaler() if cfg.scaler == "standard" else MinMaxScaler()
            scaler.fit(x_combined)
            x_train = scaler.transform(x_train)
            x_val = scaler.transform(x_val)
            if flow.uses_pickled_scaler:
                joblib.dump(scaler, save_dir / SCALER_FILE)
                logger.info(f"{type(scaler).__name__} saved to {save_dir / SCALER_FILE}")

        if hasattr(flow, "preprocessing"):
            # Backends that store their own preprocessing (flowjax) get the
            # scaler's affine parameters, the log transform and the bounds
            # of the raw training data, so that saved models need no pickle.
            raw = np.column_stack([raw_data[name] for name in parameter_names])
            flow.preprocessing = {  # type: ignore[attr-defined]
                "parameter_names": list(parameter_names),
                "scaler": None
                if scaler is None
                else _affine(scaler, x_combined),
                "log_transform": list(lambda_cols) if cfg.log_lambda else [],
                "log1m_transform": log1m_cols,
                "bounds": {
                    "min": raw.min(axis=0).tolist(),
                    "max": raw.max(axis=0).tolist(),
                },
            }

        flow.set_optimizer(cfg.learning_rate)
        train_loader, val_loader = flow.make_data_loaders(
            np.asarray(x_train), np.asarray(x_val), cfg.batch_size
        )

        best_val_loss = float("inf")
        patience_counter = 0
        best_state: Any = None
        train_losses: list[float] = []
        val_losses: list[float] = []

        for epoch in range(cfg.num_epochs):
            train_loss = flow.train_epoch(train_loader)
            val_loss = flow.eval_epoch(val_loader)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = flow.get_state()
            else:
                patience_counter += 1

            if (epoch + 1) % cfg.log_every_n_epochs == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{cfg.num_epochs}  "
                    f"train={train_loss:.4f}  val={val_loss:.4f}"
                )

            if patience_counter >= cfg.max_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        if best_state is not None:
            flow.set_state(best_state)
        logger.info(f"Training finished. Best validation loss: {best_val_loss:.4f}")
        return flow, train_losses, val_losses, scaler


def _affine(scaler, x_combined: np.ndarray) -> dict:
    """The scaler as y = x * scale + min, plus the range of the data it was fitted on (MinMaxScaler or StandardScaler)."""
    if isinstance(scaler, StandardScaler):
        scale = 1.0 / scaler.scale_
        shift = -scaler.mean_ * scale
    else:
        scale, shift = scaler.scale_, scaler.min_
    return {"min": np.asarray(shift).tolist(), "scale": np.asarray(scale).tolist(), "data_min": x_combined.min(axis=0).tolist(),
            "data_max": x_combined.max(axis=0).tolist(), "type": type(scaler).__name__}
