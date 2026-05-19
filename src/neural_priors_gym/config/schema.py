"""Top-level training configuration schema."""

from typing import Literal

from pydantic import model_validator

from .schemas._base import NeuralPriorBaseModel
from .schemas.flow import FlowConfig, GlasflowConfig
from .schemas.lambdas import LambdaConfig
from .schemas.masses import (
    BilbyMassConfig,
    DoubleGaussianMassConfig,
    GaussianMassConfig,
    MassConfig,
    UniformMassConfig,
)
from .schemas.spin import SpinConfig
from .schemas.training import TrainingHyperparamsConfig


class TrainingConfig(NeuralPriorBaseModel):
    """Top-level configuration for training a neural prior.

    Parameters
    ----------
    output_dir:
        Directory where all outputs (model, plots, training data) are saved.
    generate_only:
        If True, generate and save training data and its corner plot, then exit
        without training the flow. Useful for inspecting samples before training.
    source_type:
        "bns" for binary neutron star (both masses below MTOV, lambda_1 and
        lambda_2 interpolated) or "nsbh" for neutron star-black hole (m1
        sampled uniformly above MTOV, only lambda_2 interpolated for the NS).
    m_max_bh:
        Upper bound on the black-hole mass for NSBH sources (solar masses).
        Only used when source_type is "nsbh".
    masses:
        Configuration for the mass distribution generator. Must be provided
        together with ``lambdas`` unless ``training.data_path`` is set.
    lambdas:
        Configuration for the tidal deformability generator. Must be provided
        together with ``masses`` unless ``training.data_path`` is set.
    spin:
        Optional uniform prior on the spin magnitude of one neutron star.
        When set, the maximum mass is spin-corrected and the spin is included
        as a training parameter.  Requires ``radii_EOS`` to be available in
        the EOS data source (always present for ``jester_path``; must be
        included in the ``eos_path`` npz archive when using that option).
        For ``source_type="bns"``, only ``parameter_name="a_1"`` is supported.
        For ``source_type="nsbh"``, only ``parameter_name="a_2"`` is supported.
    flow:
        Configuration for the normalizing flow backend and architecture.
    training:
        Hyperparameters for the training loop, including ``parameter_names``
        (the columns the flow trains on) and optionally ``data_path`` to skip
        generation and load a pre-existing ``.npz`` file instead.
    """

    output_dir: str = "./outdir"
    source_type: Literal["bns", "nsbh"] = "bns"
    m_max_bh: float = 5.0
    generate_only: bool = False
    masses: MassConfig | None = None
    lambdas: LambdaConfig | None = None
    spin: SpinConfig | None = None
    flow: FlowConfig
    training: TrainingHyperparamsConfig

    @model_validator(mode="after")
    def _check_data_source(self) -> "TrainingConfig":
        has_generation = self.masses is not None and self.lambdas is not None
        has_data_path = self.training.data_path is not None
        has_partial = (self.masses is None) != (self.lambdas is None)

        if has_partial:
            raise ValueError(
                "'masses' and 'lambdas' must both be provided or both omitted. "
                "If you supply your own data, set 'training.data_path' and omit both."
            )
        if not has_generation and not has_data_path:
            raise ValueError(
                "Either provide both 'masses' and 'lambdas' for data generation, "
                "or set 'training.data_path' to load a pre-generated npz file."
            )
        if has_generation and has_data_path:
            raise ValueError(
                "Provide either ('masses' + 'lambdas') or 'training.data_path', "
                "not both."
            )
        if self.spin is not None:
            if self.source_type == "bns" and self.spin.parameter_name != "a_1":
                raise ValueError(
                    "For BNS sources, only spin.parameter_name='a_1' is supported. "
                    "Spinning mass_2 with mass ordering is currently not implemented."
                )
            if self.source_type == "nsbh" and self.spin.parameter_name != "a_2":
                raise ValueError(
                    "For NSBH sources, only spin.parameter_name='a_2' is supported "
                    "(the black hole spin is not modelled here)."
                )
        return self


__all__ = [
    "TrainingConfig",
    "MassConfig",
    "UniformMassConfig",
    "GaussianMassConfig",
    "DoubleGaussianMassConfig",
    "BilbyMassConfig",
    "LambdaConfig",
    "FlowConfig",
    "GlasflowConfig",
    "SpinConfig",
    "TrainingHyperparamsConfig",
    "NeuralPriorBaseModel",
]
