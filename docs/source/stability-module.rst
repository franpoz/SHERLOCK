.. SHERLOCK PIPEline documentation master file, created by
   sphinx-quickstart on Thu Jul  8 08:43:51 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=============================================
Stability Module
=============================================

The ``stability`` module in SHERLOCK provides tools for assessing the dynamical stability of multi-planet systems. This module helps determine whether a detected planetary system configuration is physically plausible over long timescales.

Overview
--------

The stability module offers capabilities for:

- N-body simulations of planetary systems
- Assessment of orbital stability over different timescales
- Identification of unstable orbital configurations
- Determination of stable regions for additional planets
- Stability maps and visualizations

Usage
-----

The stability module can be run directly from the command line:

.. code-block:: bash

   # Using a properties YAML configuration file (main usage method)
   python3 -m sherlockpipe.stability --properties stability.yaml
   
   # See example configuration file at examples/properties/stability.yaml
   # `<https://github.com/franpoz/SHERLOCK/blob/master/examples/properties/stability.yaml>`_
   
   # Additional optional parameters
   python3 -m sherlockpipe.stability --properties stability.yaml --years 1000 --dt 0.05 --cpus 4

The input configuration file (when used) typically includes:

- Planet masses, radii, and orbital parameters
- Simulation duration and timestep
- Output preferences for stability metrics
- Integration method settings

API Reference
------------

.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:
   
   sherlockpipe.stability

Integration with External Packages
---------------------------------

The stability module integrates with specialized N-body simulation packages to perform stability analysis, providing a user-friendly interface to configure and run these simulations within the SHERLOCK environment.
