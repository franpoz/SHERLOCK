FROM python:3.10.10

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential libssl-dev \
    libbz2-dev libssl-dev libreadline-dev \
    libffi-dev libsqlite3-dev tk-dev libpng-dev libfreetype6-dev llvm-9 llvm-9-dev \
    gfortran gcc locales python3-tk \
    poppler-utils \
    libhdf5-dev\
    nano
RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && locale-gen
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
ENV LLVM_CONFIG=/usr/bin/llvm-config-9
RUN ls /usr/bin
RUN python3 --version
RUN python3 -m pip install pip -U
RUN python3 -m pip install setuptools -U
RUN python3 -m pip install extension-helpers -U
RUN python3 -m pip install wheel -U
RUN python3 -m pip install Cython
RUN python3 -m pip install numpy==1.23.5
RUN python3 -m pip install sherlockpipe
CMD ["/bin/bash"]
