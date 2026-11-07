"""Command-line entry point for train_neural_prior."""

import sys
from pathlib import Path

import numpy as np

from neural_priors_gym.config.parser import load_config
from neural_priors_gym.config.schema import TrainingConfig
from neural_priors_gym.data.generator import TrainingDataGenerator
from neural_priors_gym.evaluation.bounds import check_out_of_bounds
from neural_priors_gym.evaluation.metrics import compute_jsd
from neural_priors_gym.preprocessing import flow_to_physical
from neural_priors_gym.logging_config import get_logger
from neural_priors_gym.plotting.plots import (
    plot_corner,
    plot_losses,
    plot_training_data,
)
from neural_priors_gym.training.trainer import FlowTrainer

logger = get_logger("neural_priors_gym.cli")


def main(config: TrainingConfig) -> None:
    """Run the full training pipeline for a neural prior.

    Steps:
    1. Generate training data (or load from ``training.data_path``).
    2. Validate that all ``training.parameter_names`` exist in the data.
    3. Save training data as an npz file (generation mode only).
    4. Train the normalizing flow.
    5. Save the flow and MinMaxScaler.
    6. Generate flow samples for evaluation.
    7. Compute JSD between training data and flow samples.
    8. Save loss plot and corner plot.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parameter_names = config.training.parameter_names
    logger.info(f"Training neural prior for parameters: {parameter_names}")

    if config.training.data_path is not None:
        # User-supplied npz — skip generation entirely.
        data_path = Path(config.training.data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"data_path not found: {data_path}")
        logger.info(f"Loading training data from {data_path}")
        raw = np.load(data_path)
        training_data: dict[str, np.ndarray] = {k: raw[k] for k in raw.files}
        missing = [n for n in parameter_names if n not in training_data]
        if missing:
            raise ValueError(
                f"The following parameter_names are not present in the npz file "
                f"{data_path}: {missing}. Available keys: {sorted(training_data)}"
            )
    else:
        # Generate training data from masses + lambdas config.
        data_gen = TrainingDataGenerator.from_config(config)
        training_data = data_gen.generate()

        missing = [n for n in parameter_names if n not in training_data]
        if missing:
            raise ValueError(
                f"The following parameter_names were not produced by the generator: "
                f"{missing}. Available keys: {sorted(training_data)}"
            )

        training_data_path = output_dir / "training_data.npz"
        np.savez(training_data_path, allow_pickle=True, **training_data)
        logger.info(f"Training data saved to {training_data_path}")

    # Plot training data samples so users can inspect the generated prior
    x_data = np.column_stack([training_data[name] for name in parameter_names])
    plot_training_data(x_data, parameter_names, output_dir)

    if config.generate_only:
        logger.info(
            "generate_only=True: skipping training. "
            f"Inspect training_data_corner.pdf in {output_dir}, "
            "then rerun with generate_only: false to train."
        )
        return

    # Train flow
    model_dir = output_dir / "model"
    logger.info(f"Will report every {config.training.log_every_n_epochs} epochs")
    trainer = FlowTrainer(config)
    flow, train_losses, val_losses, scaler = trainer.train(
        training_data, parameter_names, output_dir=model_dir
    )

    # Save flow
    flow.save(model_dir)

    # Plots and evaluation
    plot_losses(train_losses, val_losses, output_dir)

    n_eval = min(10_000, len(x_data))
    flow_samples = flow.sample(n_eval)

    # The trainer always sets the record, so this undoes the scaler and any log transforms.
    pre = flow.preprocessing
    flow_samples_original = (
        flow_samples if pre is None else flow_to_physical(flow_samples, pre)
    )

    jsd_results = compute_jsd(x_data[:n_eval], flow_samples_original)
    jsd_per_dim = [
        jsd_results[f"dim_{i}_millibits"] for i in range(len(parameter_names))
    ]
    logger.info(f"Mean JSD = {jsd_results['mean_millibits']:.2f} millibits")
    for name, jsd in zip(parameter_names, jsd_per_dim):
        logger.info(f"  {name}: {jsd:.2f} millibits")

    oob_results = check_out_of_bounds(flow_samples_original, parameter_names)
    if oob_results:
        logger.info("Out-of-bounds flow samples:")
        for name, stats in oob_results.items():
            logger.info(
                f"  {name}: {stats['pct_oob']:.2f}% OOB "
                f"({stats['pct_below']:.2f}% below, {stats['pct_above']:.2f}% above)"
            )
    else:
        logger.info("No out-of-bounds parameters detected in flow samples.")

    plot_corner(
        x_data[:n_eval],
        flow_samples_original,
        jsd_per_dim,
        parameter_names,
        output_dir,
        oob_results=oob_results if oob_results else None,
    )

    logger.info(f"Training complete. All outputs saved to {output_dir}")


def cli_entry_point() -> None:
    """Console script entry point: train_neural_prior config.yaml"""
    if len(sys.argv) != 2:
        print("Usage: train_neural_prior <config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]
    config = load_config(config_path)
    main(config)


if __name__ == "__main__":
    cli_entry_point()
