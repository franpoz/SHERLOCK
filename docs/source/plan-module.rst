.. SHERLOCK PIPEline documentation master file, created by
   sphinx-quickstart on Thu Jul  8 08:43:51 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=============================================
Plan Module
=============================================

The ``plan`` module in SHERLOCK provides tools for planning follow-up observations of exoplanet candidates. This module helps astronomers determine optimal observation windows and configurations for confirming candidate planets.

Overview
--------

The plan module offers capabilities for:

- Transit event prediction for follow-up observations
- Visibility calculations for specific observatories
- Weather and seeing condition integration
- Prioritization of targets based on scientific criteria
- Generation of observation plans and schedules

Usage
-----

The observation plan module can be run directly from the command line (within the target directory):

.. code-block:: bash

   # Basic usage for planning observations of a specific candidate
   # Note: Must specify either observatories file OR coordinates
   python3 -m sherlockpipe.plan --candidate 1 --observatories observatories.csv
   
   # Alternative: specify observatory coordinates directly
   python3 -m sherlockpipe.plan --candidate 1 --lat 28.3 --lon -16.5 --alt 2400
   
   # For example configuration format, see examples/properties/plan.yaml
   # `<https://github.com/franpoz/SHERLOCK/blob/master/examples/properties/plan.yaml>`_
   # Note: This module requires Bayesian fit results to be present in the target directory
   
   # Example observatories.csv file format:
   # `<https://github.com/franpoz/SHERLOCK/blob/master/examples/observatories.csv>`_
   

The input configuration file typically includes:

- Target information (coordinates, magnitudes)
- Observatory details (location, capabilities)
- Observation constraints (airmass, moon separation)
- Time period for planning

API Reference
------------

.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:
   
   sherlockpipe.observation_plan
   sherlockpipe.plan

Integration with External Packages
---------------------------------

The plan module integrates with astronomical packages for:

- Ephemeris calculations
- Observatory visibility computations
- Atmospheric condition predictions
- Telescope scheduling optimization
