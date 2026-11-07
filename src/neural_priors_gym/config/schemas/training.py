"""Configuration schema for training hyperparameters."""

from typing import Literal

from ._base import NeuralPriorBaseModel


class TrainingHyperparamsConfig(NeuralPriorBaseModel):
    """Hyperparameters for the normalizing flow training loop.

    Parameters
    ----------
    parameter_names:
        Ordered list of parameter names to train the normalizing flow on.
        These are the columns extracted from the training data (whether
        generated or loaded from ``data_path``). All names must exist as
        keys in the training data.
    data_path:
        Path to a pre-generated ``.npz`` file to use as training data.
        When set, the ``masses`` and ``lambdas`` generation steps are
        skipped entirely. The file must contain all keys listed in
        ``parameter_names``.
    log_lambda:
        If ``True``, apply a natural-log transform to all parameters whose
        names appear in the known lambda parameter set (``lambda_1``,
        ``lambda_2``, ``lambda_tilde``, ``delta_lambda_tilde``) before
        the flow sees the data. The saved ``training_data.npz`` always
        retains raw values.
    log_one_minus_q:
        If ``True``, the flow sees :math:`\\ln(1 - q)` instead of the mass ratio :math:`q`, so the sharp edge of the training data at
        :math:`q = 1` becomes a smooth tail and the flow cannot put mass at :math:`q > 1`. Works for every backend; the
        transform is recorded in the ``preprocessing`` entry of ``model_kwargs.json`` (see :mod:`neural_priors_gym.preprocessing`).
    scaler:
        ``"minmax"`` (default) rescales each parameter to [0, 1]; ``"standard"`` to zero mean and unit variance. With spline flows
        whose spline interval is ``[-tail_bound, tail_bound]`` (5 by default), min-max data occupy about one of the 8 bins, so the spline
        cannot resolve the distribution and the flow can put much of its mass outside the data; ``"standard"`` spreads the data over
        several bins.
    """

    parameter_names: list[str]
    data_path: str | None = None
    log_lambda: bool = False
    num_epochs: int = 2000
    learning_rate: float = 1e-3
    batch_size: int = 1024
    max_patience: int = 250
    validation_split: float = 0.2
    scale_input: bool = True
    scaler: Literal["minmax", "standard"] = "minmax"
    log_one_minus_q: bool = False
    n_samples: int = 20_000
    log_every_n_epochs: int = 100
