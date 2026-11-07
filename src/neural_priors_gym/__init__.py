"""neural_priors_gym - Train normalizing flow priors for neutron star inference."""

from typing import TYPE_CHECKING, Any

__version__ = "0.0.1"
__author__ = "ThibeauWouters"

if TYPE_CHECKING:
    from neural_priors_gym.config import load_config
    from neural_priors_gym.config.schema import TrainingConfig

__all__ = ["load_config", "TrainingConfig"]


def __getattr__(name: str) -> Any:
    # Imported lazily so that JAX-only submodules (e.g.
    # ``neural_priors_gym.flows.flowjax_prior``) can be used without pydantic.
    if name == "load_config":
        from neural_priors_gym.config import load_config

        return load_config
    if name == "TrainingConfig":
        from neural_priors_gym.config.schema import TrainingConfig

        return TrainingConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
