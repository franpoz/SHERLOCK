.. SHERLOCK PIPEline documentation master file, created by
   sphinx-quickstart on Thu Jul  8 08:43:51 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=============================================
Installation
=============================================

Some of the SHERLOCK dependencies need ``numpy`` and ``Cython``  before being installed and therefore you should install them before trying the SHERLOCK installation. Take also into account that the dependencies brough by SHERLOCK also need additional operating system tools that need to be installed separately (C++ compiler, Fortran compiler, etc). So, be aware of counting with the next OS packages installed in your environment (e.g. for Python3.8 in a Linux distribution):

``build-essential libssl-dev python3.8 python3-pip libbz2-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev tk-dev libpng-dev libfreetype6-dev llvm-9 llvm-9-dev gfortran gcc locales python3-tk libpython3.8-dev``

In case you are running a non-Linux distribution you will need to guess your OS packages matching the ones we mention for Linux.

Therefore, once you have got the OS tools, the SHERLOCK package can be installed from the PyPi repositories (after installing `numpy` and `Cython`):

``python3 -m pip install numpy``

``python3 -m pip install Cython``

``python3 -m pip install sherlockpipe``

You can also use our Docker image from `DockerHub <https://hub.docker.com/repository/docker/sherlockpipe/sherlockpipe>`_
or build it from our `Dockerfile <https://github.com/franpoz/SHERLOCK/blob/master/docker/Dockerfile>`_. Therefore, you
can also use as a Singularity container meanwhile they support Docker.