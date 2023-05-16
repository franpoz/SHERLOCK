.. SHERLOCK PIPEline documentation master file, created by
   sphinx-quickstart on Thu Jul  8 08:43:51 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=============================================
Installation
=============================================

Firstly, you must know that *SHERLOCK* is a package that lies on many different dependencies whose versions are fixed.
Hence, we strongly encourage all the users to use a *Python* virtual environment, preferently built with a conda distribution.
If anaconda is available in your distribution, please try using it. You can install it following the next instructions
https://docs.anaconda.com/free/anaconda/install/linux/

Once you installed it, you'd have to create a new environment for *SHERLOCK*:
`conda create -n sherlockenv python=3.10 anaconda`


Some of the *SHERLOCK* dependencies need ``numpy`` and ``Cython``  before being installed and therefore you should
install them before trying the SHERLOCK installation. Take also into account that the dependencies brought by SHERLOCK
also need additional operating system tools that need to be installed separately (C++ compiler, Fortran compiler, etc).
So, be aware of counting with the next OS packages installed in your environment (e.g. for Python3 in a Linux
distribution):

.. code-block::

   build-essential
   libssl-dev
   python3.10
   python3-pip
   libbz2-dev
   libssl-dev
   libreadline-dev
   libffi-dev
   libsqlite3-dev
   tk-dev
   libpng-dev
   libfreetype6-dev
   llvm-9
   llvm-9-dev
   gfortran
   gcc
   locales
   python3-tk
   python3.10-dev

In case you are running a non-Linux distribution you will need to guess your OS packages matching the ones we mention for Linux.

Therefore, once you have got the OS tools, the *SHERLOCK* package can be installed in several ways. The cleanest one is by running::

   conda activate sherlockenv
   wget https://raw.githubusercontent.com/PlanetHunters/SHERLOCK/master/requirements.txt
   python3 -m pip install -r requirements.txt

An additional installation method is running the next commands::

   conda activate sherlockenv
   python3 -m pip install numpy
   python3 -m pip install Cython
   python3 -m pip install sherlockpipe

You can also use our Docker image from `DockerHub <https://hub.docker.com/repository/docker/sherlockpipe/sherlockpipe>`_
or build it from our `Dockerfile <https://github.com/PlanetHunters/SHERLOCK/blob/master/docker/Dockerfile>`_. Therefore, you
can also use as a Singularity container meanwhile they support Docker.

-------------
Dependencies
-------------

All the needed dependencies should be included by your `pip` installation of *SHERLOCK*. If you are
interested you can inspect the requirements list under
`setup.py <https://github.com/PlanetHunters/SHERLOCK/blob/master/setup.py>`_.

--------
Testing
--------

**SHERLOCK** comes with a light automated tests suite that can be executed with:

``tox``

This suite tests several points from the pipeline for the supported *Python* versions:

* The construction of the ``SHERLOCK`` *Python* object.
* The parameters setup of the ``SHERLOCK`` *Python* object.
* The provisioning of objects of interest files.
* Load and filtering of objects of interest.
* Different kind of short **SHERLOCK** executions.

In case you want to test the entire *SHERLOCK* functionality we encourage you to
run some (or all) the `manual examples <https://github.com/PlanetHunters/SHERLOCK/tree/master/examples>`_.
If so, please read the instructions provided there to execute them.