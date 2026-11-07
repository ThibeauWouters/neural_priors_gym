"""Convert a jester inference result HDF5 file to the npz format expected by neural_priors_gym."""

import sys
from pathlib import Path

import argparse
import numpy as np

from neural_priors_gym.data.lambdas.interpolator import load_jester_eos


def convert(input_path: Path, output_path: Path) -> None:
    """Convert a jester HDF5 result file to an EOS npz file.

    Extracts ``masses_EOS`` and ``Lambdas_EOS`` from the
    ``posterior/derived_eos`` group (read with h5py, so jester is not needed)
    and saves them unchanged as a NumPy archive. Invalid EOS (e.g. the
    all-zero rows jester writes for failed TOV solves) are kept here and
    removed when the file is loaded by the interpolator. The output can be used
    with :class:`~neural_priors_gym.data.lambdas.interpolator.EOSLambdaInterpolator`.

    Args:
        input_path: Path to the jester ``results.h5`` file.
        output_path: Destination path for the output ``.npz`` file.
    """
    print(f"Loading jester result from {input_path} ...")
    masses_EOS, Lambdas_EOS = load_jester_eos(input_path)

    print(f"Found {len(masses_EOS)} EOS samples.")
    np.savez(output_path, masses_EOS=masses_EOS, Lambdas_EOS=Lambdas_EOS)
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
