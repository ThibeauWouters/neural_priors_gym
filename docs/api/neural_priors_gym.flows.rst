``neural_priors_gym.flows`` module
===================================

.. currentmodule:: neural_priors_gym.flows

The flows module provides the abstract base class and concrete implementations
for normalizing flow backends. All backends implement the
:class:`~neural_priors_gym.flows.base.FlowBase` interface, which exposes a
common API for training, sampling, density evaluation, and saving/loading.
New flow backends can be added by subclassing :class:`~neural_priors_gym.flows.base.FlowBase`;
see :doc:`/developer_guide/adding_new_flow` for instructions.

Base class
----------

.. currentmodule:: neural_priors_gym.flows.base

.. autosummary::
   :nosignatures:
   :toctree: _autosummary/

   FlowBase

Glasflow backend
----------------

.. currentmodule:: neural_priors_gym.flows.glasflow

.. autosummary::
   :nosignatures:
   :toctree: _autosummary/

   GlasflowNSF

Zuko MAF backend
----------------

.. currentmodule:: neural_priors_gym.flows.zuko_maf

.. autosummary::
   :nosignatures:
   :toctree: _autosummary/

   ZukoMAF

flowjax backend
---------------

.. currentmodule:: neural_priors_gym.flows.flowjax_flow

.. autosummary::
   :nosignatures:
   :toctree: _autosummary/

   FlowjaxFlow

.. currentmodule:: neural_priors_gym.flows.flowjax_prior

.. autosummary::
   :nosignatures:
   :toctree: _autosummary/

   FlowjaxPrior
   load_flowjax_prior
   build_flowjax_flow
