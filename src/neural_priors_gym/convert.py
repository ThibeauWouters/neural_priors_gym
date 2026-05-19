"""Convert a jester inference result HDF5 file to the npz format expected by neural_priors_gym."""

import sys
from pathlib import Path

import argparse
import numpy as np

from jesterTOV.inference.result import InferenceResult


def convert(input_path: Path, output_path: Path) -> None:
    """Convert a jester HDF5 result file to an EOS npz file.

    Extracts ``masses_EOS``, ``Lambdas_EOS``, and ``radii_EOS`` (when
    available) from the posterior and saves them as a NumPy archive.
    Including ``radii_EOS`` enables spin-aware mass generation via the
    ``spin`` config option.

    Args:
        input_path: Path to the jester ``results.h5`` file.
        output_path: Destination path for the output ``.npz`` file.
    """
    print(f"Loading jester result from {input_path} ...")
    result = InferenceResult.load(str(input_path))

    posterior = result.posterior
    masses_EOS: np.ndarray = posterior["masses_EOS"]
    Lambdas_EOS: np.ndarray = posterior["Lambdas_EOS"]

    arrays: dict[str, np.ndarray] = {
        "masses_EOS": masses_EOS,
        "Lambdas_EOS": Lambdas_EOS,
    }

    if "radii_EOS" in posterior:
        arrays["radii_EOS"] = posterior["radii_EOS"]
        print("Including radii_EOS (enables spin-aware mass generation).")
    else:
        print(
            "No radii_EOS found in posterior — spin correction will not be available."
        )

    print(f"Found {len(masses_EOS)} EOS samples.")
    if "radii_EOS" in arrays:
        np.savez(
            output_path,
            masses_EOS=arrays["masses_EOS"],
            Lambdas_EOS=arrays["Lambdas_EOS"],
            radii_EOS=arrays["radii_EOS"],
        )
    else:
        np.savez(
            output_path,
            masses_EOS=arrays["masses_EOS"],
            Lambdas_EOS=arrays["Lambdas_EOS"],
        )
    print(f"Saved to {output_path}")


def cli_entry_point() -> None:
    """Console script entry point: jester_hdf5_to_npz results.h5 [eos_samples.npz]"""
    parser = argparse.ArgumentParser(
        description="Convert a jester HDF5 inference result to an EOS npz file."
    )
    parser.add_argument("input", type=Path, help="Path to the jester results.h5 file.")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=Path("eos_samples.npz"),
        help="Output npz path (default: eos_samples.npz).",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    convert(args.input, args.output)


if __name__ == "__main__":
    cli_entry_point()
