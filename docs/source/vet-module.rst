.. SHERLOCK PIPEline documentation master file, created by
   sphinx-quickstart on Thu Jul  8 08:43:51 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=============================================
Vet Module
=============================================

The ``vet`` module in SHERLOCK provides tools for vetting exoplanet candidates. It helps assess whether a detected transit signal is likely to be a genuine exoplanet or a false positive.

Overview
--------

The vet module implements several tests and metrics to evaluate candidate signals:

- Transit shape analysis
- Odd-even transit depth comparison
- Secondary eclipse detection
- Centroid shift analysis
- Statistical validation
- Integration with TESS and Kepler validation tools

This module leverages ``dearwatson`` , ``triceratops`` and other packages to provide comprehensive vetting capabilities.

Usage
-----

The vet module can be run directly from the command line (being inside the target directory):

.. code-block:: bash

   # Basic usage with a specific candidate
   python3 -m sherlockpipe.vet --candidate 1
   
   # For example configuration file format, see examples/properties/vet.yaml
   # `<https://github.com/franpoz/SHERLOCK/blob/master/examples/properties/vet.yaml>`_

API Reference
------------

.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:
   
   sherlockpipe.vet

Integration with External Packages
---------------------------------

The vet module uses several external tools and libraries:

- ``dearwatson`` for candidate validation
- ``triceratops`` for false positive assessment
- Visualization libraries for diagnostic plots
