``neural_priors_gym.preprocessing`` module
==========================================

.. currentmodule:: neural_priors_gym.preprocessing

Every trained model stores its preprocessing (scaler as an affine map, the
:math:`\ln x` and :math:`\ln(1 - x)` transforms, and the range of the training
data) under the ``"preprocessing"`` key of ``model_kwargs.json``, whatever the
backend. The functions below evaluate a flow in physical units from that
record with NumPy only, Jacobian included; they are what the bilby prior uses.

.. autosummary::
   :nosignatures:
   :toctree: _autosummary/

   load_preprocessing
   make_preprocessing_record
   physical_to_flow
   flow_to_physical
   physical_log_prob
   in_support
