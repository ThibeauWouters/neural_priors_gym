``neural_priors_gym.config`` module
====================================

.. currentmodule:: neural_priors_gym.config

The configuration module provides Pydantic v2 models that validate and parse the
YAML configuration file used to drive ``train_neural_prior``. The top-level
entry point is :func:`~neural_priors_gym.config.parser.load_config`, which reads
a YAML file and returns a validated :class:`~neural_priors_gym.config.schema.TrainingConfig`.
For a complete reference of all supported configuration fields, see the
:doc:`/yaml_reference` guide.

Parser
------

.. currentmodule:: neural_priors_gym.config.parser

.. autosummary::
   :nosignatures:
   :toctree: _autosummary/

   load_config

Top-level schema
----------------

.. currentmodule:: neural_priors_gym.config.schema

.. autosummary::
   :nosignatures:
   :toctree: _autosummary/

   TrainingConfig

Mass configuration schemas
--------------------------

.. currentmodule:: neural_priors_gym.config.schemas.masses

.. autosummary::
   :nosignatures:
   :toctree: _autosummary/

   UniformMassConfig
   GaussianMassConfig
   DoubleGaussianMassConfig
   BilbyMassConfig

Lambda configuration schema
----------------------------

.. currentmodule:: neural_priors_gym.config.schemas.lambdas

.. autosummary::
   :nosignatures:
   :toctree: _autosummary/

   LambdaConfig

Flow configuration schemas
---------------------------

.. currentmodule:: neural_priors_gym.config.schemas.flow

.. autosummary::
   :nosignatures:
   :toctree: _autosummary/

   GlasflowConfig
   ZukoMAFConfig
   FlowjaxConfig

Training hyperparameter schema
-------------------------------

.. currentmodule:: neural_priors_gym.config.schemas.training

.. autosummary::
   :nosignatures:
   :toctree: _autosummary/

   TrainingHyperparamsConfig
