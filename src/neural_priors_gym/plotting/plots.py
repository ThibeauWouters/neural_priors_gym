"""Plotting utilities for training diagnostics."""

from pathlib import Path
from typing import Optional

import corner
import matplotlib.pyplot as plt
import numpy as np

from neural_priors_gym.logging_config import get_logger
from neural_priors_gym.plotting.labels import get_tex_labels

logger = get_logger("neural_priors_gym.plotting")


def _setup_tex() -> None:
    """Enable TeX rendering if a TeX installation is found, else disable it.

    Set ``NEURAL_PRIORS_GYM_USETEX=0`` to disable it anyway, e.g. on a cluster where ``latex`` exists but lacks packages.
    """
    import os
    import shutil

    if os.environ.get("NEURAL_PRIORS_GYM_USETEX", "1") != "0" and shutil.which("latex") is not None:
        plt.rcParams.update({"text.usetex": True})
    else:
        plt.rcParams.update({"text.usetex": False})
        logger.debug("LaTeX not found on PATH; falling back to matplotlib mathtext")


_setup_tex()

_CORNER_LABEL_FONTSIZE: int = 18
_CORNER_TICK_FONTSIZE: int = 14


def plot_losses(
    train_losses: list[float],
    val_losses: list[float],
    output_dir: Path,
    filename: str = "training_loss.pdf",
) -> None:
    """Plot training and validation loss curves and save to output_dir.

    Uses a log scale on the y-axis when all loss values are positive.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label="Training loss")
    ax.plot(val_losses, label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Negative log-likelihood")
    ax.set_title("Training and validation loss")
    ax.legend()

    all_positive = all(v > 0 for v in train_losses) and all(v > 0 for v in val_losses)
    if all_positive:
        ax.set_yscale("log")

    save_path = output_dir / filename
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Loss plot saved to {save_path}")


def plot_training_data(
    samples: np.ndarray,
    parameter_names: list[str],
    output_dir: Path,
    filename: str = "training_data_corner.pdf",
    n_plot: Optional[int] = 10_000,
) -> None:
    """Corner plot of the generated training data samples.

    Saves a single-dataset corner plot so users can inspect the prior samples
    before committing to a full training run.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    if n_plot is not None and len(samples) > n_plot:
        idx = rng.choice(len(samples), size=n_plot, replace=False)
        samples = samples[idx]

    fig = corner.corner(  # type: ignore[call-arg]
        samples,
        labels=get_tex_labels(parameter_names),
        color="steelblue",
        hist_kwargs={"density": True},
        plot_datapoints=False,
        fill_contours=True,
        smooth=1.0,
        label_kwargs={"fontsize": _CORNER_LABEL_FONTSIZE},
    )
    for ax in fig.get_axes():
        ax.tick_params(labelsize=_CORNER_TICK_FONTSIZE)

    save_path = output_dir / filename
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Training data corner plot saved to {save_path}")


def plot_corner(
    train_samples: np.ndarray,
    flow_samples: np.ndarray,
    jsd_millibits: list[float],
    parameter_names: list[str],
    output_dir: Path,
    filename: str = "corner_plot.pdf",
    n_plot: Optional[int] = 10_000,
    oob_results: Optional[dict[str, dict[str, float]]] = None,
) -> None:
    """Overlay corner plots of training data and flow samples.

    A random subset of n_plot samples is used from each set to keep the plot
    readable. The JSD in millibits for each 1D marginal is shown above the
    corresponding diagonal panel. If *oob_results* is provided, the
    out-of-bounds percentage is appended to the title for any flagged parameter.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    def subsample(arr: np.ndarray) -> np.ndarray:
        if n_plot is not None and len(arr) > n_plot:
            idx = rng.choice(len(arr), size=n_plot, replace=False)
            return arr[idx]
        return arr

    train_plot = subsample(train_samples)
    flow_plot = subsample(flow_samples)

    tex_labels = get_tex_labels(parameter_names)
    fig = corner.corner(  # type: ignore[call-arg]
        train_plot,
        labels=tex_labels,
        color="blue",
        hist_kwargs={"density": True},
        plot_datapoints=False,
        fill_contours=True,
        smooth=1.0,
        label_kwargs={"fontsize": _CORNER_LABEL_FONTSIZE},
    )
    corner.corner(  # type: ignore[call-arg]
        flow_plot,
        labels=tex_labels,
        color="orange",
        hist_kwargs={"density": True},
        plot_datapoints=False,
        fill_contours=True,
        smooth=1.0,
        fig=fig,
        label_kwargs={"fontsize": _CORNER_LABEL_FONTSIZE},
    )
    axes = fig.get_axes()
    n = len(parameter_names)
    for ax in axes:
        ax.tick_params(labelsize=_CORNER_TICK_FONTSIZE)
    for i, (name, jsd) in enumerate(zip(parameter_names, jsd_millibits)):
        title = f"{jsd:.1f} mb"
        if oob_results is not None and name in oob_results:
            pct = oob_results[name]["pct_oob"]
            title += f" | OOB: {pct:.1f}%"
        axes[i * n + i].set_title(title, fontsize=_CORNER_TICK_FONTSIZE)

    save_path = output_dir / filename
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Corner plot saved to {save_path}")
