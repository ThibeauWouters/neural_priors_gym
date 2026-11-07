"""YAML configuration parser for neural_priors_gym."""

from pathlib import Path
from typing import Union

import yaml
from pydantic import ValidationError

from neural_priors_gym.logging_config import get_logger

from .schema import TrainingConfig

logger = get_logger("neural_priors_gym.config")


def load_config(config_path: Union[str, Path]) -> TrainingConfig:
    """Load and validate a training configuration from a YAML file.

    Relative paths within the config (e.g. eos_path, prior_file) are
    resolved relative to the directory containing the config file.

    Parameters
    ----------
    config_path:
        Path to the YAML configuration file.

    Returns
    -------
    TrainingConfig
        Validated configuration object.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ValueError
        If YAML parsing or Pydantic validation fails.
    """
    config_path = Path(config_path).resolve()
    logger.debug(f"Loading configuration from: {config_path}")

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path) as f:
        try:
            config_dict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML in {config_path}: {e}") from e

    if config_dict is None:
        raise ValueError(f"Configuration file is empty: {config_path}")

    config_dir = config_path.parent

    # Resolve relative paths relative to the config file directory
    if "lambdas" in config_dict and "eos_path" in config_dict["lambdas"]:
        eos_path = Path(config_dict["lambdas"]["eos_path"])
        if not eos_path.is_absolute():
            config_dict["lambdas"]["eos_path"] = str((config_dir / eos_path).resolve())

    if (
        "lambdas" in config_dict
        and isinstance(config_dict["lambdas"], dict)
        and config_dict["lambdas"].get("jester_path") is not None
    ):
        jester_path = Path(config_dict["lambdas"]["jester_path"])
        if not jester_path.is_absolute():
            config_dict["lambdas"]["jester_path"] = str(
                (config_dir / jester_path).resolve()
            )

    if "masses" in config_dict and "prior_file" in config_dict["masses"]:
        prior_file = Path(config_dict["masses"]["prior_file"])
        if not prior_file.is_absolute():
            config_dict["masses"]["prior_file"] = str(
                (config_dir / prior_file).resolve()
            )

    if (
        "training" in config_dict
        and isinstance(config_dict["training"], dict)
        and "data_path" in config_dict["training"]
        and config_dict["training"]["data_path"] is not None
    ):
        data_path = Path(config_dict["training"]["data_path"])
        if not data_path.is_absolute():
            config_dict["training"]["data_path"] = str(
                (config_dir / data_path).resolve()
            )

    if "output_dir" in config_dict:
        output_dir = Path(config_dict["output_dir"])
        if not output_dir.is_absolute():
            config_dict["output_dir"] = str((config_dir / output_dir).resolve())

    try:
        config = TrainingConfig(**config_dict)
        logger.debug("Configuration validation successful")
        return config
    except ValidationError as e:
        raise ValueError(_format_validation_error(e, config_path)) from e
    except Exception as e:
        raise ValueError(
            f"Error validating configuration from {config_path}: {e}"
        ) from e


def _format_validation_error(e: ValidationError, config_path: Path) -> str:
    blocks: list[str] = [f"Configuration error in {config_path}:"]
    for error in e.errors():
        loc_parts = [str(p) for p in error["loc"] if p != "__root__"]
        loc = " -> ".join(loc_parts) if loc_parts else "(top level)"
        msg: str = error["msg"]
        if msg.startswith("Value error, "):
            msg = msg[len("Value error, ") :]
        blocks.append(f"\n[{loc}]\n{msg}")
    return "\n".join(blocks)
