"""Configuration schemas for normalizing flow backends."""

from typing import Annotated, Literal, Union

from pydantic import Discriminator

from ._base import NeuralPriorBaseModel


class GlasflowConfig(NeuralPriorBaseModel):
    """Configuration for the glasflow CouplingNSF backend."""

    backend: Literal["glasflow"] = "glasflow"
    flow_type: Literal["CouplingNSF"] = (
        "CouplingNSF"  # TODO: extend with more flow types
    )
    n_transforms: int = 4
    n_neurons: int = 256
    n_blocks_per_transform: int = 3
    num_bins: int = 8
    tail_bound: float = 5.0
    activation: Literal["relu", "gelu", "silu", "tanh"] = "relu"
    linear_transform: Literal["permutation", "lu", "svd", "none"] = "none"
    dropout_probability: float = 0.0
    batch_norm_within_blocks: bool = False
    batch_norm_between_transforms: bool = False


class ZukoMAFConfig(NeuralPriorBaseModel):
    """Configuration for the zuko Masked Autoregressive Flow (MAF) backend."""

    backend: Literal["zuko"] = "zuko"
    flow_type: Literal["maf"] = "maf"
    transforms: int = 3
    randperm: bool = False
    hidden_features: list[int] = [64, 64]


class FlowjaxConfig(NeuralPriorBaseModel):
    """Configuration for the flowjax rational-quadratic-spline backend.

    The defaults follow the glasflow ``CouplingNSF`` setup of arXiv:2511.22987
    as closely as flowjax allows. The conditioner is a plain MLP with
    ``nn_depth`` hidden layers of ``nn_width`` units, whereas glasflow uses a
    residual network with ``n_blocks_per_transform`` residual blocks.
    """

    backend: Literal["flowjax"] = "flowjax"
    flow_type: Literal["coupling_nsf", "maf_nsf"] = "coupling_nsf"
    n_transforms: int = 4
    nn_width: int = 256
    nn_depth: int = 3
    num_bins: int = 8
    tail_bound: float = 5.0
    activation: Literal["relu", "gelu", "silu", "tanh"] = "relu"
    invert: bool = True
    dtype: Literal["float32", "float64"] = "float32"
    seed: int = 0


FlowConfig = Annotated[
    Union[GlasflowConfig, ZukoMAFConfig, FlowjaxConfig],
    Discriminator("backend"),
]
