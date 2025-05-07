FROM nvcr.io/nvidia/dgl:23.11-py3

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

RUN apt-get -y autoremove

RUN apt-get install -y vim
RUN apt-get install -y python3-pip

RUN apt update
RUN apt install -y libqt5charts5 libqt5charts5-dev
RUN pip install pandas
RUN pip install networkx==2.6.3
RUN pip install scikit-learn
RUN pip install numpy==1.24.4
RUN pip install tqdm==4.53.0
RUN pip install pydantic
RUN pip install cupy-cuda12x

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
source $HOME/.cargo/env

WORKDIR /app
