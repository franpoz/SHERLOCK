FROM ubuntu:18.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential libssl-dev python3.8 \
    python3-pip libbz2-dev libssl-dev libreadline-dev \
    libffi-dev libsqlite3-dev tk-dev libpng-dev libfreetype6-dev llvm-9 llvm-9-dev \
    gfortran gcc locales python3-tk libpython3.8-dev
RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && locale-gen
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
ENV LLVM_CONFIG=/usr/bin/llvm-config-9
RUN python3.8 -m pip install Cython
RUN python3.8 -m pip install numpy
RUN python3.8 -m pip install sherlockpipe --pre
CMD ["/bin/bash"]
