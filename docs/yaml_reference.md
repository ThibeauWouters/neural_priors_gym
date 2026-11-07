(yaml-reference)=
# YAML configuration reference

This page is the authoritative reference for all supported YAML configuration
options in ``neural_priors_gym``. When you add or change configuration fields,
update this file by hand to keep it accurate.

## Overview

``neural_priors_gym`` uses YAML configuration files validated by Pydantic v2
models (see {class}`~neural_priors_gym.config.schema.TrainingConfig`). This
reference documents every supported field, its type, default value, and purpose.
For a worked example from start to finish, see the {doc}`quickstart` guide.

---

## Top-level options

Control the output directory, source type, and whether to run the full training
pipeline or only generate training data.

::::{dropdown} **Top-level configuration options**
:open:

```yaml
output_dir: ./outdir   # Directory for all outputs (training data, model, plots)
source_type: bns       # "bns" or "nsbh"
m_max_bh: 5.0          # Upper bound on BH mass in solar masses (NSBH only)
generate_only: false   # If true, generate data and corner plot only, then exit
```

**Field details:**

- **`output_dir`** (`str`, default: `"./outdir"`) — Directory where all outputs
  are written. Created automatically if it does not exist.
- **`source_type`** (`str`, default: `"bns"`) — Compact binary system type.
  Use `"bns"` for binary neutron star systems (both components below $M_\mathrm{TOV}$)
  and `"nsbh"` for neutron star–black hole systems (primary sampled above $M_\mathrm{TOV}$,
  secondary is the neutron star).
- **`m_max_bh`** (`float`, default: `5.0`) — Upper bound on the black-hole mass
  in solar masses, used when `source_type: "nsbh"`. Must exceed the maximum
  $M_\mathrm{TOV}$ across all EOS samples in the provided file.
- **`generate_only`** (`bool`, default: `false`) — When `true`, the pipeline
  generates and saves the training data and a corner plot, then exits without
  training the flow. Useful for inspecting the prior before committing to a
  training run. Rerun with `generate_only: false` to proceed to training.

::::

---

## Mass distribution

(yaml-reference-masses)=

The `masses` section specifies the population model used to sample compact-object
masses. The `type` field selects the distribution, and each type has its own
set of parameters.

### Uniform

Masses are drawn uniformly between `m_min` and the maximum TOV mass ($M_\mathrm{TOV}$)
of a randomly selected EOS sample. The mass convention is $m_1 \geq m_2$ and both
must lie below $M_\mathrm{TOV}$ for BNS systems.
The Pydantic config schema is {class}`~neural_priors_gym.config.schemas.masses.UniformMassConfig`.

::::{dropdown} **Uniform mass configuration**
:open:

```yaml
masses:
  type: uniform                # Required: selects uniform distribution
  parameter_names:             # Names of mass quantities written to training data
    - mass_1_source
    - mass_2_source
  m_min: 1.0                   # Minimum NS mass in solar masses
```

**Field details:**

- **`type`** (`str`) — Must be `"uniform"` for this distribution.
- **`parameter_names`** (`list[str]`, **required**) — Ordered list of mass
  quantities to include in the training data and flow. Supported names:
  `mass_1_source`, `mass_2_source`, `chirp_mass_source`, `mass_ratio`.
- **`m_min`** (`float`, **required**) — Minimum neutron star mass in solar masses.
  Masses are drawn uniformly in `[m_min, M_TOV]`.

::::

### Gaussian

Masses are drawn from a single Gaussian distribution truncated at `m_min` (below)
and $M_\mathrm{TOV}$ (above). Samples that fall outside these bounds are rejected
and redrawn up to `max_attempts` times.
The Pydantic config schema is {class}`~neural_priors_gym.config.schemas.masses.GaussianMassConfig`.

::::{dropdown} **Gaussian mass configuration**

```yaml
masses:
  type: gaussian               # Required: selects single Gaussian distribution
  parameter_names:
    - chirp_mass_source
    - mass_ratio
  m_min: 1.0                   # Minimum mass (lower truncation bound)
  mean: 1.33                   # Gaussian mean in solar masses
  std: 0.07                    # Gaussian standard deviation in solar masses
  max_attempts: 1000000        # Maximum rejection-sampling attempts (optional)
```

**Field details:**

- **`type`** (`str`) — Must be `"gaussian"`.
- **`parameter_names`** (`list[str]`, **required**) — Supported names: `mass_1_source`,
  `mass_2_source`, `chirp_mass_source`, `mass_ratio`.
- **`m_min`** (`float`, **required**) — Lower truncation bound in solar masses.
- **`mean`** (`float`, **required**) — Mean of the Gaussian in solar masses.
- **`std`** (`float`, **required**) — Standard deviation of the Gaussian in solar masses.
- **`max_attempts`** (`int`, default: `1_000_000`) — Maximum number of rejection-sampling
  attempts before raising an error. Increase if the acceptance rate is very low
  (e.g. when `m_min` or $M_\mathrm{TOV}$ cuts away a large fraction of the Gaussian).

::::

### Double Gaussian

Masses are drawn from a mixture of two Gaussians, each independently truncated
at `m_min` (below) and $M_\mathrm{TOV}$ (above). The `weight` parameter controls
the mixing fraction of the first Gaussian.
The Pydantic config schema is {class}`~neural_priors_gym.config.schemas.masses.DoubleGaussianMassConfig`.

::::{dropdown} **Double Gaussian mass configuration**

```yaml
masses:
  type: double_gaussian        # Required: selects double Gaussian distribution
  parameter_names:
    - chirp_mass_source
    - mass_ratio
  m_min: 1.0                   # Minimum mass (lower truncation bound)
  mean_1: 1.34                 # Mean of the first Gaussian in solar masses
  std_1: 0.07                  # Standard deviation of the first Gaussian
  mean_2: 1.80                 # Mean of the second Gaussian in solar masses
  std_2: 0.21                  # Standard deviation of the second Gaussian
  weight: 0.65                 # Mixing weight of the first Gaussian (0 to 1)
  max_attempts: 1000000        # Maximum rejection-sampling attempts (optional)
```

**Field details:**

- **`type`** (`str`) — Must be `"double_gaussian"`.
- **`parameter_names`** (`list[str]`, **required**) — Same supported names as for
  the uniform and Gaussian distributions.
- **`m_min`** (`float`, **required**) — Lower truncation bound in solar masses.
- **`mean_1`** (`float`, **required**) — Mean of the first component in solar masses.
- **`std_1`** (`float`, **required**) — Standard deviation of the first component.
- **`mean_2`** (`float`, **required**) — Mean of the second component in solar masses.
- **`std_2`** (`float`, **required**) — Standard deviation of the second component.
- **`weight`** (`float`, **required**) — Probability of drawing from the first
  component. Must be in `[0, 1]`; the second component receives weight `1 - weight`.
- **`max_attempts`** (`int`, default: `1_000_000`) — Maximum rejection-sampling
  attempts.

::::

### Bilby prior

```{warning}
The bilby prior mass generator is experimental. It has seen limited testing
compared to the analytic distributions above. Verify that the generated
`mass_1_source` and `mass_2_source` samples look as expected (e.g. using
`generate_only: true`) before committing to a full training run.
```

Masses are generated by sampling from a ``bilby`` prior file and then converting
the sampled parameters to source-frame component masses using ``bilby``'s
built-in conversion utilities. The prior file should define parameters from which
``bilby.gw.conversion.generate_all_bns_parameters`` can derive
`mass_1_source` and `mass_2_source` (for example `chirp_mass` and `mass_ratio`).
The Pydantic config schema is {class}`~neural_priors_gym.config.schemas.masses.BilbyMassConfig`.

::::{dropdown} **Bilby prior mass configuration**

```yaml
masses:
  type: bilby                  # Required: selects bilby prior file
  parameter_names:
    - mass_1_source
    - mass_2_source
  prior_file: prior.prior      # Path to a bilby .prior file
```

**Field details:**

- **`type`** (`str`) — Must be `"bilby"`.
- **`parameter_names`** (`list[str]`, **required**) — Mass quantity names to
  include in the training data.
- **`prior_file`** (`str`, **required**) — Path to a ``bilby`` prior specification
  file (`.prior` extension). The file must define parameters from which
  `mass_1_source` and `mass_2_source` can be derived.

::::

---

## Lambda (tidal deformability) configuration

The `lambdas` section configures how tidal deformabilities are interpolated from
a set of EOS samples. For each training sample, a random EOS is drawn from the
provided file and $\Lambda(M)$ is evaluated at the sampled mass by linear
interpolation.

Exactly one of `eos_path` or `jester_path` must be set; they are mutually
exclusive.

::::{dropdown} **Lambda configuration — npz file**
:open:

```yaml
lambdas:
  parameter_names:             # Lambda quantities to include in the training data
    - lambda_1
    - lambda_2
  eos_path: /path/to/eos_samples.npz   # Path to a NumPy npz EOS file
  log_lambda: false            # Apply log transform to Lambda before training
```

**Field details:**

- **`parameter_names`** (`list[str]`, default: `["lambda_1", "lambda_2"]`) —
  Lambda quantities to include. Supported names for BNS systems:
  `lambda_1`, `lambda_2`, `lambda_tilde`, `delta_lambda_tilde`.
  For NSBH systems (`source_type: "nsbh"`), only `lambda_2` is supported,
  since the black hole contributes no tidal deformability.
- **`eos_path`** (`str | null`, default: `null`) — Path to a NumPy `.npz` archive
  containing the arrays `masses_EOS` and `Lambdas_EOS`, each of shape
  `(n_eos, n_mass_points)`. EOS samples with any negative $\Lambda$ values are
  automatically removed at load time. Mutually exclusive with `jester_path`.
- **`jester_path`** (`str | null`, default: `null`) — Path to a
  [jester](https://github.com/nuclear-multimessenger-astronomy/jester) inference
  result `.h5` file. The posterior must contain `masses_EOS` and `Lambdas_EOS`
  arrays. Mutually exclusive with `eos_path`.
- **`log_lambda`** (`bool`, default: `false`) — When `true`, a natural-log
  transform is applied to all Lambda columns before the scaler and flow see the
  data. The raw (untransformed) values are always written to `training_data.npz`;
  only the flow and scaler operate on the log-transformed values.

::::

::::{dropdown} **Lambda configuration — jester HDF5 file**

```yaml
lambdas:
  parameter_names:
    - lambda_1
    - lambda_2
  jester_path: /path/to/results.h5   # Path to a jester inference result file
  log_lambda: false
```


::::

---

(yaml-reference-flow)=
## Flow configuration

The `flow` section configures the normalizing flow backend and architecture.
The `backend` field selects the backend; currently `"glasflow"`, `"zuko"` and `"flowjax"` are supported.

### Glasflow CouplingNSF

The glasflow backend wraps the ``glasflow`` library's ``CouplingNSF`` (Neural
Spline Flow with coupling layers). Input features are first projected by a linear
transform and then processed by a series of spline coupling transforms.
The Pydantic config schema is {class}`~neural_priors_gym.config.schemas.flow.GlasflowConfig`.

::::{dropdown} **Glasflow CouplingNSF configuration**
:open:

```yaml
flow:
  backend: glasflow              # Required: selects the glasflow backend
  flow_type: CouplingNSF         # Flow type (only CouplingNSF is currently supported)
  n_transforms: 4                # Number of coupling transform layers
  n_neurons: 256                 # Hidden units per layer in the conditioner networks
  n_blocks_per_transform: 3      # Residual blocks per coupling transform
  num_bins: 8                    # Number of spline bins
  tail_bound: 5.0                # Spline tail bound (outside tails are linear)
  activation: relu               # Activation function: "relu", "gelu", "silu", "tanh"
  linear_transform: none         # Pre-transform on inputs: "permutation", "lu", "svd", "none"
  dropout_probability: 0.0       # Dropout probability in conditioner networks
  batch_norm_within_blocks: false        # Batch norm inside each residual block
  batch_norm_between_transforms: false   # Batch norm between coupling transforms
```

**Field details:**

- **`backend`** (`str`) — Must be `"glasflow"` for this configuration.
- **`flow_type`** (`str`, default: `"CouplingNSF"`) — Flow architecture within
  the glasflow backend. Currently only `"CouplingNSF"` is supported.
- **`n_transforms`** (`int`, default: `4`) — Number of coupling transform layers.
  More transforms increase expressiveness at the cost of additional parameters
  and slower inference.
- **`n_neurons`** (`int`, default: `256`) — Number of hidden units in each layer
  of the conditioner neural networks inside the coupling transforms.
- **`n_blocks_per_transform`** (`int`, default: `3`) — Number of residual blocks
  in the conditioner network of each coupling transform.
- **`num_bins`** (`int`, default: `8`) — Number of bins used by the rational
  quadratic spline. More bins allow sharper features but require more parameters.
- **`tail_bound`** (`float`, default: `5.0`) — The spline is defined on the
  interval `[-tail_bound, tail_bound]`; outside this range the transform is
  linear. When `scale_input: true` (recommended), all training data is mapped
  to `[0, 1]` by the ``MinMaxScaler``, which is well within the spline support.
- **`activation`** (`str`, default: `"relu"`) — Activation function for the
  conditioner networks. Choices: `"relu"`, `"gelu"`, `"silu"`, `"tanh"`.
- **`linear_transform`** (`str`, default: `"none"`) — Optional linear
  pre-transform applied to the inputs before the coupling stack. Choices:
  `"permutation"`, `"lu"`, `"svd"`, `"none"`.
- **`dropout_probability`** (`float`, default: `0.0`) — Dropout rate inside the
  conditioner networks. A small value (e.g. `0.1`) can improve generalization on
  smaller datasets.
- **`batch_norm_within_blocks`** (`bool`, default: `false`) — Apply batch
  normalization inside each residual block of the conditioner networks.
- **`batch_norm_between_transforms`** (`bool`, default: `false`) — Apply batch
  normalization between consecutive coupling transform layers.

::::

### Zuko MAF

The zuko MAF backend wraps the ``zuko`` library's ``MAF`` (Masked Autoregressive
Flow). Each transform is a masked autoregressive layer whose parameters are
predicted by a masked MLP, ensuring the autoregressive property.
Compared to coupling-based flows, MAF can be more expressive for
low-dimensional data, but sampling requires one sequential forward pass per
feature rather than a single parallel pass.
The Pydantic config schema is {class}`~neural_priors_gym.config.schemas.flow.ZukoMAFConfig`.

::::{dropdown} **Zuko MAF configuration**

```yaml
flow:
  backend: zuko                  # Required: selects the zuko backend
  flow_type: maf                 # Required: type of zuko flow (maf = Masked Autoregressive Flow)
  transforms: 3                  # Number of autoregressive transform layers
  randperm: false                # Randomly permute features between transforms
  hidden_features:               # Hidden layer sizes in each masked MLP
    - 64
    - 64
```

**Field details:**

- **`backend`** (`str`) — Must be `"zuko"` for this configuration.
- **`flow_type`** (`str`) — Must be `"maf"` for the Masked Autoregressive Flow. This field selects which zuko flow type to use, allowing other zuko flows to be added in the future.
- **`transforms`** (`int`, default: `3`) — Number of autoregressive transform
  layers. More transforms increase the expressiveness of the flow. A value of
  3–6 is a good starting point for the 4-dimensional (mass, Lambda) space.
- **`randperm`** (`bool`, default: `false`) — When `true`, features are randomly
  permuted between consecutive autoregressive transforms. When `false`, the
  ordering alternates between forward and reverse. Random permutation can
  improve mixing but makes results non-reproducible without a fixed seed.
- **`hidden_features`** (`list[int]`, default: `[64, 64]`) — Width of each hidden
  layer in the masked MLP that parameterises each autoregressive transform.
  Increasing depth or width raises model capacity at the cost of more parameters
  and slower training.

::::

### flowjax neural spline flow

The flowjax backend trains a rational-quadratic-spline flow in JAX, either with
coupling layers (`coupling_nsf`, the default) or with masked autoregressive
layers (`maf_nsf`). It is meant for priors that will be evaluated inside a JAX
sampler. The defaults follow the glasflow `CouplingNSF` setup of
arXiv:2511.22987 as closely as flowjax allows; the main difference is that the
conditioner is a plain MLP with `nn_depth` hidden layers rather than a residual
network. Training uses the same data split, scaler and early stopping as the
other backends.
The Pydantic config schema is {class}`~neural_priors_gym.config.schemas.flow.FlowjaxConfig`.

::::{dropdown} **flowjax configuration**

```yaml
flow:
  backend: flowjax               # Required: selects the flowjax backend
  flow_type: coupling_nsf        # "coupling_nsf" or "maf_nsf"
  n_transforms: 4                # Number of flow layers
  nn_width: 256                  # Hidden units per conditioner layer
  nn_depth: 3                    # Hidden layers per conditioner MLP
  num_bins: 8                    # Number of spline bins
  tail_bound: 5.0                # Spline acts on [-tail_bound, tail_bound], identity outside
  activation: relu               # "relu", "gelu", "silu", "tanh"
  invert: true                   # flowjax `invert` flag; true makes log_prob a single pass
  dtype: float32                 # "float32" or "float64" (training precision)
  seed: 0                        # Seed for initialisation, permutations and shuffling
```

The model directory contains `model_kwargs.json` and `flowjax_weights.eqx`.
The JSON file holds the hyperparameters, the parameter names, the affine
parameters of the scaler, the list of log-transformed parameters, the bounds of
the training data and the library versions, so no pickled scikit-learn object
is written. A model can be loaded with only numpy, JAX, equinox and flowjax
installed, and evaluated in physical units, scaler Jacobian included:

```python
import jax
jax.config.update("jax_enable_x64", True)
from neural_priors_gym.flows.flowjax_prior import load_flowjax_prior

prior = load_flowjax_prior("outdir/model")
log_p = jax.jit(jax.vmap(prior.log_prob))(x)  # x in physical units, shape (n, n_dim)
samples = prior.sample(jax.random.key(0), 1000)
```

The spline parametrisation differs between flowjax major versions, so the
model must be loaded with the flowjax major version it was trained with; the
loader raises an error otherwise.

::::

---

## Training hyperparameters

The `training` section controls the training loop, early stopping, and data
preprocessing. All fields have sensible defaults and can be omitted entirely
if the defaults are acceptable.
The Pydantic config schema is {class}`~neural_priors_gym.config.schemas.training.TrainingHyperparamsConfig`.

::::{dropdown} **Training hyperparameter configuration**
:open:

```yaml
training:
  n_samples: 20000           # Number of training samples to generate
  num_epochs: 2000           # Maximum number of training epochs
  learning_rate: 0.001       # Adam optimizer learning rate
  batch_size: 1024           # Mini-batch size for training
  max_patience: 250          # Early stopping patience (epochs without improvement)
  validation_split: 0.2      # Fraction of data held out for validation
  scale_input: true          # Fit a MinMaxScaler on the training data
  log_every_n_epochs: 100    # Log training progress every N epochs
```

**Field details:**

- **`n_samples`** (`int`, default: `20_000`) — Number of (mass, Lambda) training
  samples to generate. Increase for smoother priors or when using more flow
  parameters; typical values range from 20 000 to 200 000.
- **`num_epochs`** (`int`, default: `2000`) — Maximum number of training epochs.
  Training terminates earlier if the validation loss does not improve for
  `max_patience` consecutive epochs.
- **`learning_rate`** (`float`, default: `1e-3`) — Initial learning rate for the
  Adam optimizer.
- **`batch_size`** (`int`, default: `1024`) — Number of samples per mini-batch.
- **`max_patience`** (`int`, default: `250`) — Early stopping patience: if the
  validation loss does not improve for this many consecutive epochs, training stops
  and the best checkpoint is restored.
- **`validation_split`** (`float`, default: `0.2`) — Fraction of samples reserved
  for validation. The scaler (when `scale_input: true`) is fitted on the combined
  training and validation sets to avoid data leakage.
- **`scale_input`** (`bool`, default: `true`) — When `true`, a ``MinMaxScaler``
  is fitted on the data and applied before training, mapping each column to
  `[0, 1]`. The scaler is saved to ``outdir/scaler.gz`` so that flow samples can
  be back-transformed at evaluation time.
- **`log_every_n_epochs`** (`int`, default: `100`) — Interval (in epochs) at
  which the training and validation NLL are logged.

::::

---

## Complete examples

### BNS with uniform masses

A minimal configuration for training a prior on BNS source-frame masses with a
uniform mass distribution.

::::{dropdown} **BNS uniform configuration**
:open:

```yaml
output_dir: ./outdir
source_type: bns

masses:
  type: uniform
  parameter_names:
    - mass_1_source
    - mass_2_source
  m_min: 1.0

lambdas:
  parameter_names:
    - lambda_1
    - lambda_2
  eos_path: /path/to/eos_samples.npz

flow:
  backend: glasflow
  n_transforms: 4
  n_neurons: 256
  n_blocks_per_transform: 3
  num_bins: 8

training:
  n_samples: 20000
  num_epochs: 500
  learning_rate: 0.0001
  batch_size: 1024
  max_patience: 100
```

::::

### BNS with double Gaussian masses

A configuration for training a prior in the $(\mathcal{M}_\mathrm{c}, q)$ space
using a double Gaussian mass distribution, which is a more realistic population
model for BNS systems.

::::{dropdown} **BNS double Gaussian configuration**

```yaml
output_dir: ./outdir
source_type: bns

masses:
  type: double_gaussian
  parameter_names:
    - chirp_mass_source
    - mass_ratio
  m_min: 1.0
  mean_1: 1.34
  std_1: 0.07
  mean_2: 1.80
  std_2: 0.21
  weight: 0.65

lambdas:
  parameter_names:
    - lambda_1
    - lambda_2
  eos_path: /path/to/eos_samples.npz

flow:
  backend: glasflow
  n_transforms: 4
  n_neurons: 256
  n_blocks_per_transform: 3
  num_bins: 8

training:
  n_samples: 200000
  num_epochs: 500
  learning_rate: 0.0001
  batch_size: 1024
  max_patience: 100
```

::::

### NSBH with uniform masses

A configuration for neutron star–black hole systems. The black hole (primary)
is sampled uniformly above $M_\mathrm{TOV}$ up to `m_max_bh`; the neutron star
(secondary) is sampled from the chosen NS distribution. Only `lambda_2` (the NS
tidal deformability) is included.

::::{dropdown} **NSBH uniform configuration**

```yaml
output_dir: ./outdir
source_type: nsbh
m_max_bh: 5.0             # Must exceed the MTOV of all EOS samples

masses:
  type: uniform
  parameter_names:
    - mass_1_source
    - mass_2_source
  m_min: 1.0

lambdas:
  parameter_names:
    - lambda_2             # Only the NS tidal deformability
  eos_path: /path/to/eos_samples.npz

flow:
  backend: glasflow
  n_transforms: 4
  n_neurons: 256
  n_blocks_per_transform: 3
  num_bins: 8

training:
  n_samples: 200000
  num_epochs: 500
  learning_rate: 0.0001
  batch_size: 1024
  max_patience: 100
```

::::

### BNS with uniform masses and zuko MAF

The same BNS uniform prior as above, but using the zuko Masked Autoregressive
Flow backend. This is a good starting point for comparing the two backends on
the same dataset.

::::{dropdown} **BNS uniform with zuko MAF**

```yaml
output_dir: ./outdir
source_type: bns

masses:
  type: uniform
  parameter_names:
    - mass_1_source
    - mass_2_source
  m_min: 1.0

lambdas:
  parameter_names:
    - lambda_1
    - lambda_2
  eos_path: /path/to/eos_samples.npz

flow:
  backend: zuko
  flow_type: maf
  transforms: 4
  hidden_features:
    - 64
    - 64

training:
  n_samples: 20000
  num_epochs: 500
  learning_rate: 0.0001
  batch_size: 1024
  max_patience: 100
```

::::

### Generate training data only

Run the pipeline up to data generation and plotting only, without training the
flow. Useful for quick inspection of the prior before committing to a full
training run.

::::{dropdown} **Generate-only configuration**

```yaml
output_dir: ./outdir
source_type: bns
generate_only: true

masses:
  type: uniform
  parameter_names:
    - mass_1_source
    - mass_2_source
  m_min: 1.0

lambdas:
  parameter_names:
    - lambda_1
    - lambda_2
  eos_path: /path/to/eos_samples.npz

flow:
  backend: glasflow

training:
  n_samples: 20000
```

::::
