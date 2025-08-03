.. SHERLOCK PIPEline documentation master file, created by
   sphinx-quickstart on Thu Jul  8 08:43:51 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=============================================
Fit Module
=============================================

The ``fit`` module in SHERLOCK allows you to perform Bayesian parameter estimation for exoplanet transit models. This module integrates with ``allesfitter`` to provide a comprehensive fitting environment.

Overview
--------

The fit module is responsible for:

- Preparing light curve data for model fitting
- Setting up priors and model configurations
- Running MCMC or Nested Sampling algorithms
- Generating posterior distributions
- Creating diagnostic plots and fit reports

Usage
-----

The fit module can be run directly from the command line:

.. code-block:: bash

   # Basic usage with a specific candidate
   python3 -m sherlockpipe.fit --candidate 1
   

The input configuration file (when used) specifies parameters like:

- Transit model type
- Sampling method and parameters
- Prior distributions
- Data sources

API Reference
------------

.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:
   
   sherlockpipe.fit
   sherlockpipe.bayesian_fit

Integration with External Packages
---------------------------------

The fit module uses ``allesfitter`` as its main backend for parameter estimation. This integration allows SHERLOCK to leverage:

- Multiple sampling algorithms (emcee, dynesty)
- Comprehensive model capabilities
- Publication-ready plotting functions
