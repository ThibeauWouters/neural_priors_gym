# Adding a new mass model

This guide walks through the steps required to add a new mass distribution model
to ``neural_priors_gym``. Mass models are interchangeable: each one is a thin
class that implements the common
{class}`~neural_priors_gym.data.masses.base.MassGenerator` interface and is
selected at runtime through the Pydantic configuration layer.

## Overview

Adding a new mass model requires changes in three places:

1. **A new config schema** in `config/schemas/masses.py` — defines the
   hyperparameters specific to the new distribution.
2. **A new generator class** in `data/masses/<model>.py` — implements the
   sampling logic.
3. **A dispatch entry** in `data/generator.py` — wires the config type to the
   generator class.

The guide covers the BNS and NSBH cases separately because their roles are
slightly different: in a BNS system both components are neutron stars drawn from
the same population model, while in an NSBH system only the secondary (NS) mass
comes from your model — the primary (BH) mass has its own sampling logic.

## BNS systems

### Step 1 — Add a config schema

Open `src/neural_priors_gym/config/schemas/masses.py` and add a new Pydantic
model. The `type` field must be a `Literal` with a unique string that users will
write in their YAML file.

```python
class PowerLawMassConfig(NeuralPriorBaseModel):
    """Power-law mass distribution between m_min and MTOV."""

    type: Literal["power_law"] = "power_law"
    parameter_names: list[str]
    m_min: float
    alpha: float  # spectral index
```

Then add the new class to the `MassConfig` discriminated union at the bottom of
the same file:

```python
MassConfig = Annotated[
    Union[
        UniformMassConfig,
        GaussianMassConfig,
        DoubleGaussianMassConfig,
        BilbyMassConfig,
        PowerLawMassConfig,  # <-- new entry
    ],
    Discriminator("type"),
]
```

### Step 2 — Implement the generator class

Create `src/neural_priors_gym/data/masses/power_law.py`. Your class must
subclass {class}`~neural_priors_gym.data.masses.base.MassGenerator` and implement
two abstract methods:

- `generate(n_samples, mtov_array, eos_indices=None)` — produces `(m1, m2)` pairs for a BNS
  system. Both masses come from your distribution. By convention, the returned
  array has shape `(n_samples, 2)` with `mass_1_source >= mass_2_source`.
- `generate_ns_mass(n_samples, mtov_array, eos_indices=None)` — produces a single NS mass per
  sample. This is called by `NSBHMassGenerator` when your model is used as the
  NS population in an NSBH run (see below).

In both methods, `mtov_array` is a 1-D array of length `n_eos` containing the
maximum TOV mass for each EOS sample in the dataset, and `eos_indices` gives
the EOS of each sample. The data generator draws these indices once and passes
the same ones to the Lambda interpolator, so that the masses and tidal
deformabilities of a sample come from the same EOS. Use the MTOV of each
sample's EOS as the upper bound on the NS mass; the helper
`neural_priors_gym.data.masses.base.mtov_per_sample` returns it and draws
random indices when `eos_indices` is `None`.

```python
"""Power-law mass generator."""

import numpy as np

from neural_priors_gym.config.schemas.masses import PowerLawMassConfig

from .base import MassGenerator, mtov_per_sample as get_mtov


class PowerLawMassGenerator(MassGenerator):
    """Samples NS masses from a power-law distribution in [m_min, MTOV]."""

    def __init__(self, config: PowerLawMassConfig) -> None:
        self.parameter_names = config.parameter_names
        self.m_min = config.m_min
        self.alpha = config.alpha

    def _sample_power_law(self, n: int, mtov: np.ndarray) -> np.ndarray:
        """Draw n samples from a power-law in [m_min, mtov_i] for each i."""
        u = np.random.uniform(0.0, 1.0, n)
        a, b, alpha = self.m_min, mtov, self.alpha
        if alpha == -1.0:
            return a * (b / a) ** u
        exp = alpha + 1.0
        return (a**exp + u * (b**exp - a**exp)) ** (1.0 / exp)

    def generate(
        self, n_samples: int, mtov_array: np.ndarray, eos_indices=None
    ) -> np.ndarray:
        mtov_per_sample = get_mtov(n_samples, mtov_array, eos_indices)

        m_a = self._sample_power_law(n_samples, mtov_per_sample)
        m_b = self._sample_power_law(n_samples, mtov_per_sample)

        m1 = np.maximum(m_a, m_b)
        m2 = np.minimum(m_a, m_b)
        return np.column_stack([m1, m2])

    def generate_ns_mass(
        self, n_samples: int, mtov_array: np.ndarray, eos_indices=None
    ) -> np.ndarray:
        mtov_per_sample = get_mtov(n_samples, mtov_array, eos_indices)
        return self._sample_power_law(n_samples, mtov_per_sample)
```

Export the new class from `data/masses/__init__.py`:

```python
from .power_law import PowerLawMassGenerator
```

### Step 3 — Add a dispatch entry in the generator

Open `src/neural_priors_gym/data/generator.py` and import your new config and
generator classes at the top, then add an `elif` branch to `build_mass_generator`:

```python
from neural_priors_gym.config.schemas.masses import (
    BilbyMassConfig,
    DoubleGaussianMassConfig,
    GaussianMassConfig,
    PowerLawMassConfig,   # <-- new import
    UniformMassConfig,
)
from neural_priors_gym.data.masses.power_law import PowerLawMassGenerator  # <-- new import

def build_mass_generator(config: TrainingConfig) -> MassGenerator:
    mass_config = config.masses
    if isinstance(mass_config, UniformMassConfig):
        ns_gen: MassGenerator = UniformMassGenerator(mass_config)
    elif isinstance(mass_config, GaussianMassConfig):
        ns_gen = GaussianMassGenerator(mass_config)
    elif isinstance(mass_config, DoubleGaussianMassConfig):
        ns_gen = DoubleGaussianMassGenerator(mass_config)
    elif isinstance(mass_config, BilbyMassConfig):
        ns_gen = BilbyPriorMassGenerator(mass_config)
    elif isinstance(mass_config, PowerLawMassConfig):   # <-- new branch
        ns_gen = PowerLawMassGenerator(mass_config)
    else:
        raise ValueError(f"Unknown mass config type: {type(mass_config)}")

    if config.source_type == "nsbh":
        return NSBHMassGenerator(ns_gen, config.m_max_bh)

    return ns_gen
```

Notice that the same dispatch function is used for both BNS and NSBH. When
`source_type = "nsbh"` the NS generator you just wrote is automatically wrapped
in `NSBHMassGenerator`, which handles the BH component separately.

## NSBH systems

NSBH mass generation has two independent components: the NS (secondary) mass and
the BH (primary) mass. Which path you take depends on what you want to change.

### Adding a new NS population model (common case)

If you only want to change how the neutron star mass is distributed — for example
to use a power-law or a peaked Gaussian — follow the BNS steps above exactly.
The `build_mass_generator` function already wraps any NS generator in
`NSBHMassGenerator` when `source_type = "nsbh"` is set in the config, so your
new generator will work for NSBH runs without any additional changes.

Your YAML file then looks like:

```yaml
source_type: nsbh
masses:
  type: power_law       # your new model
  parameter_names: [mass_1_source, mass_2_source]
  m_min: 1.0
  alpha: -2.5
  # m_max_bh is a top-level field, not inside masses
m_max_bh: 10.0
```

The `m_max_bh` field is read by `build_mass_generator` from the top-level
`TrainingConfig` and passed to `NSBHMassGenerator` as the upper bound for
uniform BH mass sampling in :math:`[\mathrm{MTOV}, m_{\max,\mathrm{BH}}]`.

### Custom BH mass distribution (advanced case)

The default `NSBHMassGenerator` draws the BH mass uniformly between MTOV and
`m_max_bh`. If you need a different BH mass distribution — for example a
power-law or a peaked distribution motivated by X-ray binary observations — you
should write a fully custom NSBH generator rather than trying to extend
`NSBHMassGenerator`.

Create `src/neural_priors_gym/data/masses/nsbh_power_law.py`. This class
implements `MassGenerator` directly and handles both components internally:

```python
"""NSBH generator with a power-law BH mass distribution."""

import numpy as np

from neural_priors_gym.config.schemas.masses import NSBHPowerLawConfig

from .base import MassGenerator, mtov_per_sample as get_mtov


class NSBHPowerLawMassGenerator(MassGenerator):
    """NSBH system where the BH mass follows a power law above MTOV.

    The NS (secondary) is drawn uniformly in [m_min_ns, MTOV].
    The BH (primary) is drawn from a power law in [MTOV, m_max_bh].
    """

    def __init__(self, config: NSBHPowerLawConfig) -> None:
        self.parameter_names = config.parameter_names
        self.m_min_ns = config.m_min_ns
        self.m_max_bh = config.m_max_bh
        self.alpha_bh = config.alpha_bh

    def _sample_power_law(
        self, n: int, lo: np.ndarray, hi: float, alpha: float
    ) -> np.ndarray:
        u = np.random.uniform(0.0, 1.0, n)
        if alpha == -1.0:
            return lo * (hi / lo) ** u
        exp = alpha + 1.0
        return (lo**exp + u * (hi**exp - lo**exp)) ** (1.0 / exp)

    def generate(
        self, n_samples: int, mtov_array: np.ndarray, eos_indices=None
    ) -> np.ndarray:
        mtov_per_sample = get_mtov(n_samples, mtov_array, eos_indices)

        if np.any(mtov_per_sample >= self.m_max_bh):
            raise ValueError(
                f"m_max_bh={self.m_max_bh} must be above MTOV for all EOS samples "
                f"(max MTOV found: {mtov_per_sample.max():.2f} Msun)."
            )

        m1 = self._sample_power_law(
            n_samples, mtov_per_sample, self.m_max_bh, self.alpha_bh
        )
        u_ns = np.random.uniform(0.0, 1.0, n_samples)
        m2 = self.m_min_ns + u_ns * (mtov_per_sample - self.m_min_ns)

        return np.column_stack([m1, m2])

    def generate_ns_mass(
        self, n_samples: int, mtov_array: np.ndarray, eos_indices=None
    ) -> np.ndarray:
        mtov_per_sample = get_mtov(n_samples, mtov_array, eos_indices)
        u = np.random.uniform(0.0, 1.0, n_samples)
        return self.m_min_ns + u * (mtov_per_sample - self.m_min_ns)
```

Add a matching config class in `config/schemas/masses.py`:

```python
class NSBHPowerLawConfig(NeuralPriorBaseModel):
    """NSBH system: power-law BH mass, uniform NS mass."""

    type: Literal["nsbh_power_law"] = "nsbh_power_law"
    parameter_names: list[str]
    m_min_ns: float
    m_max_bh: float
    alpha_bh: float
```

Include `NSBHPowerLawConfig` in the `MassConfig` union and add an `elif` branch
in `build_mass_generator` that returns an `NSBHPowerLawMassGenerator` directly
(without wrapping it in the generic `NSBHMassGenerator`):

```python
elif isinstance(mass_config, NSBHPowerLawConfig):
    return NSBHPowerLawMassGenerator(mass_config)
```

Because this generator handles both components itself, it does **not** go through
the `if config.source_type == "nsbh"` wrapping step. Set `source_type: nsbh` in
your config so that the data generator only passes mass_2_source (the NS) to
the Lambda interpolator.

## Step 4 — Add tests

Add a test file, for example `tests/test_masses/test_power_law.py`, that
verifies at minimum:

- The generator can be constructed from a config.
- `generate` returns an array of shape `(n_samples, 2)` with
  `mass_1_source >= mass_2_source` and all values in `[m_min, m_max_bh]` (or MTOV).
- `generate_ns_mass` returns a 1-D array of length `n_samples` with all values
  below MTOV.
- All returned masses are finite.

Run the tests with:

```bash
uv run pytest -v tests/test_masses/test_power_law.py
```

## Step 5 — Update the documentation

Add your new config class to the API reference in
`docs/api/neural_priors_gym.config.schemas.masses.rst` and document the new YAML
options in `docs/yaml_reference.md` following the existing dropdown pattern.
Rebuild the docs in strict mode to check for warnings:

```bash
uv run sphinx-build -W --keep-going docs docs/_build/html
```
