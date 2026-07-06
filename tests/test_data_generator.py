"""Tests for TrainingDataGenerator."""

import numpy as np
import yaml
from pathlib import Path

from neural_priors_gym.config.parser import load_config
from neural_priors_gym.data.generator import TrainingDataGenerator


def test_generate_returns_correct_keys(small_config_yaml: Path) -> None:
    config = load_config(small_config_yaml)
    gen = TrainingDataGenerator.from_config(config)
    data = gen.generate()
    assert "mass_1_source" in data
    assert "mass_2_source" in data
    assert "lambda_1" in data
    assert "lambda_2" in data


def test_generate_correct_length(small_config_yaml: Path) -> None:
    config = load_config(small_config_yaml)
    gen = TrainingDataGenerator.from_config(config)
    data = gen.generate()
    n = config.training.n_samples
    for arr in data.values():
        assert len(arr) == n


def test_generate_lambdas_positive(small_config_yaml: Path) -> None:
    config = load_config(small_config_yaml)
    gen = TrainingDataGenerator.from_config(config)
    data = gen.generate()
    assert np.all(data["lambda_1"] > 0)
    assert np.all(data["lambda_2"] > 0)


def test_generate_m1_ge_m2(small_config_yaml: Path) -> None:
    config = load_config(small_config_yaml)
    gen = TrainingDataGenerator.from_config(config)
    data = gen.generate()
    assert np.all(data["mass_1_source"] >= data["mass_2_source"])


def test_save_creates_npz(small_config_yaml: Path, tmp_path: Path) -> None:
    config = load_config(small_config_yaml)
    gen = TrainingDataGenerator.from_config(config)
    output_path = tmp_path / "training_data.npz"
    gen.save(output_path)
    assert output_path.exists()
    loaded = np.load(output_path)
    assert "mass_1_source" in loaded


def test_generate_includes_redshift_for_chirp_mass_ratio_redshift_config(
    tmp_path: Path, small_eos_npz: Path
) -> None:
    """The chirp_mass_q_redshift mass config should expose redshift as a
    trainable quantity, giving a 5D BNS parameter space instead of 4D."""
    config_dict = {
        "output_dir": str(tmp_path / "outdir"),
        "masses": {
            "type": "chirp_mass_q_redshift",
            "mc_min": 1.0,
            "mc_max": 2.0,
            "q_min": 0.5,
            "q_max": 1.0,
            "z_min": 0.001,
            "z_max": 0.1,
        },
        "lambdas": {
            "eos_path": str(small_eos_npz),
        },
        "flow": {
            "backend": "glasflow",
            "n_transforms": 2,
            "n_neurons": 32,
            "n_blocks_per_transform": 1,
            "num_bins": 4,
        },
        "training": {
            "parameter_names": [
                "mass_1_source",
                "mass_2_source",
                "redshift",
                "lambda_1",
                "lambda_2",
            ],
            "num_epochs": 5,
            "learning_rate": 1e-3,
            "batch_size": 64,
            "max_patience": 5,
            "validation_split": 0.2,
            "scale_input": True,
            "n_samples": 500,
        },
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f)

    config = load_config(config_path)
    gen = TrainingDataGenerator.from_config(config)
    data = gen.generate()

    assert "redshift" in data
    assert len(data["redshift"]) == config.training.n_samples
    assert np.all(data["redshift"] >= 0.001)
    assert np.all(data["redshift"] <= 0.1)
    assert np.all(data["mass_1_source"] >= data["mass_2_source"])
