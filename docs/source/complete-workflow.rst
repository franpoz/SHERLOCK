.. SHERLOCK PIPEline documentation master file, created by
   sphinx-quickstart on Thu Jul  8 08:43:51 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=============================================
Complete SHERLOCK Workflow
=============================================

This tutorial demonstrates a complete workflow using SHERLOCK's modules for exoplanet detection, characterization, and follow-up planning.

Overview
--------

The SHERLOCK pipeline provides a comprehensive suite of tools for the complete exoplanet discovery and characterization workflow:

1. **Search** - Detect transit signals in light curves
2. **Fit** - Perform detailed parameter estimation 
3. **Vet** - Validate candidates and rule out false positives
4. **Stability** - Assess dynamical stability of multi-planet systems
5. **Plan** - Create follow-up observation plans

This tutorial demonstrates how these modules can be used together in a coordinated workflow.

Step 1: Candidate Detection
--------------------------

First, we use the search module to identify transit candidates:

.. code-block:: bash

   # Run SHERLOCK search with a properties file
   python3 -m sherlockpipe --properties properties.yaml
   
   # For example properties files, see examples/properties directory
   # `<https://github.com/franpoz/SHERLOCK/tree/master/examples/properties>`_

Step 2: Parameter Fitting
------------------------

Next, move into the target directory and use the fit module to estimate physical parameters for the detected candidate:

.. code-block:: bash

   # Run Bayesian fitting on the first detected candidate
   python3 -m sherlockpipe.fit --candidate 1
   
   # Alternative: use a custom configuration file
   # python3 -m sherlockpipe.fit --input fit_config.yaml

Step 3: Candidate Vetting
------------------------

Validate that the candidate is likely a real planet using the vet module:

.. code-block:: bash

   # Run standard vetting tests on the first candidate
   python3 -m sherlockpipe.vet --candidate 1
   
   # Alternative: use a custom vetting configuration file
   # python3 -m sherlockpipe.vet --input vet_config.yaml

Step 4: Stability Analysis
-------------------------

For multi-planet systems, assess orbital stability using the stability module:

.. code-block:: bash

   # Run stability analysis using a properties file
   python3 -m sherlockpipe.stability --properties stability.yaml
   
   # For example stability configuration file format:
   # `<https://github.com/franpoz/SHERLOCK/blob/master/examples/properties/stability.yaml>`_

Step 5: Observation Planning
--------------------------

Finally, plan follow-up observations using the observation plan module:

.. code-block:: bash

   # Plan observations for the first candidate (requires observatories file)
   python3 -m sherlockpipe.plan --candidate 1 --observatories observatories.csv
   
   # For example observatories.csv file format:
   # `<https://github.com/franpoz/SHERLOCK/blob/master/examples/observatories.csv>`_
   
   # Alternative: use a custom planning configuration file
   # python3 -m sherlockpipe.plan --input plan_config.yaml

Complete Pipeline Integration
---------------------------

All these steps can be integrated into a single workflow script or shell script that executes the entire process from candidate detection to follow-up planning. Here's an example of a simple shell script that would run the complete workflow:

.. code-block:: bash

   #!/bin/bash
   # Complete SHERLOCK workflow example
   
   # 1. Run the search
   python3 -m sherlockpipe --properties properties.yaml
   
   # 2. Perform Bayesian fitting on the first candidate
   python3 -m sherlockpipe.fit --candidate 1
   
   # 3. Vet the candidate
   python3 -m sherlockpipe.vet --candidate 1
   
   # 4. Run stability analysis (if it's a multi-planet system)
   python3 -m sherlockpipe.stability --system 1
   
   # 5. Plan follow-up observations
   python3 -m sherlockpipe.plan --candidate 1 --observatories LCO-SSO,NOT
