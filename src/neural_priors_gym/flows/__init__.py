"""Normalizing flow backends.

The backends are imported lazily, so that importing one of them (for example
the JAX-only :mod:`neural_priors_gym.flows.flowjax_prior`) does not pull in the
dependencies of the others (torch, glasflow, zuko).
"""

import importlib
from typing import Any, TYPE_CHECKING

from .base import FlowBase


if TYPE_CHECKING:
    from .flowjax_flow import FlowjaxFlow
    from .flowjax_prior import FlowjaxPrior, load_flowjax_prior
    from .glasflow import GlasflowNSF
    from .zuko_maf import ZukoMAF

_LAZY = {
    "GlasflowNSF": ".glasflow",
    "ZukoMAF": ".zuko_maf",
    "FlowjaxFlow": ".flowjax_flow",
    "FlowjaxPrior": ".flowjax_prior",
    "load_flowjax_prior": ".flowjax_prior",
}

__all__ = [
    "FlowBase",
    "GlasflowNSF",
    "ZukoMAF",
    "FlowjaxFlow",
    "FlowjaxPrior",
    "load_flowjax_prior",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
