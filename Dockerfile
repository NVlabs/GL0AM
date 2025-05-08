FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

WORKDIR /src
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y gnupg2 ca-certificates

RUN echo "deb [trusted=yes] https://downloads.skewed.de/apt jammy main" >> /etc/apt/sources.list
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-key 612DEFB798507F25
RUN apt-get update

RUN apt-get install -y git
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:ubuntu-toolchain-r/test -y
RUN apt-get update
RUN apt-get install -y gcc-11 g++-11
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100

RUN apt-get install -y libpython-all-dev
RUN apt-get install -y libboost-all-dev
RUN apt-get install -y libcairo2
RUN apt-get install -y libcairo2-dev
RUN apt-get install -y python3-matplotlib
RUN apt-get update
RUN apt-get install -y python3-graph-tool

RUN apt install -y curl

RUN apt-get install -y vim
RUN apt-get install -y python3-pip

RUN pip install torch==2.1.0
RUN pip install dgl -f https://data.dgl.ai/wheels/torch-2.1/cu121/repo.html
RUN pip install pycairo==1.20.1
RUN pip install pandas==2.2.2
RUN pip install scikit-learn
RUN pip install numpy==1.24.4
							
RUN pip install pydantic
RUN pip install pyyaml
RUN pip install torchdata==0.7.1
RUN pip install cupy-cuda12x
RUN pip install tqdm==4.53.0

ENV DGLBACKEND="pytorch"

WORKDIR /app
RUN git clone --recursive https://github.com/ASU-VDA-Lab/MLCAD25-Contest-Scripts-Benchmarks.git

WORKDIR /app/MLCAD25-Contest-Scripts-Benchmarks/OpenROAD/
RUN ./etc/DependencyInstaller.sh -all
RUN mkdir build
WORKDIR /app/MLCAD25-Contest-Scripts-Benchmarks/OpenROAD/build
RUN cmake ..
RUN make -j 4

WORKDIR /src
RUN curl  --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
RUN source $HOME/.cargo/env

WORKDIR /app
