FROM ubuntu:18.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential libssl-dev python3.6 \
    python3-pip libbz2-dev libssl-dev libreadline-dev \
    libffi-dev libsqlite3-dev tk-dev libpng-dev libfreetype6-dev llvm-9 llvm-9-dev
ENV LLVM_CONFIG=/usr/bin/llvm-config-9
RUN python3.6 -m pip install numpy
RUN python3.6 -m pip install sherlockpipe
CMD ["/bin/bash"]