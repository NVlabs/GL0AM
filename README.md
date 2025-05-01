# MLCAD2025_Contest_Sim

## Introduction

This is a branch of GL0AM that is serving the purpose of a logic cone simulator/verifier for the 2025 MLCAD Design Contest: https://asu-vda-lab.github.io/MLCAD25-Contest/ . As such, we can view it as a 'subset' form of GL0AM that _only_ simulates the combinational logic cones for a golden netlist, vs. a resynthesized netlist. The stimuli for the logic cones are created randomly, and at the end of their respective simulations, the results for the logic cone endpoints for both netlists are compared. 

## Prerequisites, packages, installation

We will need the following to use the GPU simulator:

### 1. Hardware Platform
  * Developed on NVIDIA GV100 GPU and Intel Xeon Platinum 8174 CPU. (But, most any GPUs should work)

### 2. Software Platform
  * OS: Ubuntu 20.04.5
  * CUDA: nvcc-11.8
  * CUDA driver: 550.90.07
  * C/C++: gcc-9.4.0
  * Rust: 1.85.1 (though 1.82 and/or above should work)
  * hMetis: 1.5
  * Python: Python-3.8.10, with the following packages:
    * PyTorch: 2.2.0+cu121
    * DGL: 2.0.0+cu118
    * NumPy: 1.22.2
    * Networkx: 2.6.3
    * SciPy: 1.10.1
    * CuPy: 11.0.0b2 (https://github.com/leofang/cupy)
    * Scikit-learn: 0.24.2
    * Pandas: 1.5.2
    * librosa: 0.9.2
    * Jupyter-notebook: 7.4.8
    * Pillow: 9.2.0
    * xgboost: 1.6.2
  
Licenses for the 3rd party software can be found in [LICENSES.txt](LICENSES.txt).

## Setup and Trial Run
### 0. Install Rustc and Python packages
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
#and follow prompt, can check installation correct if you can run 'rustc -V'
#We also want to add a submodule to the repo:
git submodule add  https://github.com/gzz2000/eda-infra-rs.git
#build the rustc executable that translates Verilog into simulation graph
cd <TOP_DIR>
cargo build
#if you see "Finished `dev` ..." then rustc portion is all set to go!
```
We used a docker container to manage our software platform, an example installation script can be found in [packages.sh](packages.sh) . In any case, successful installation may be subject to the OS, but the key is that [PyTorch](https://pytorch.org/), a GPU enabled version of [DGL](https://www.dgl.ai/pages/start.html), and [CuPy](https://cupy.dev/) should be installed. 

### 1. Convert Verilog Netlist to Combinational Logic Cones in CSR format
There are a few sample netlists in <TOP_DIR>/build_gatspi_graph/tests for this trial.
```
cargo run --bin build_gatspi_graph ./build_gatspi_graph/tests/adder.v ./gatspi/adder.pkl
cargo run --bin build_gatspi_graph ./build_gatspi_graph/tests/adder_altCorrect.v ./gatspi/adder_altCorrect.pkl
cargo run --bin build_gatspi_graph ./build_gatspi_graph/tests/adder_altIncorrect.v ./gatspi/adder_altIncorrect.pkl
#"cargo run --bin build_gatspi_graph" will print the usage
```
### 2. Compile the CSR Graphs to Simulation Graphs, Simulate Golden and Resynthesized Logic Cones, Compare the Simulation Results
```
cd <TOP_DIR>/gatspi
python3 runGatspi.py --top_name adder --graph0FilePath ./adder.pkl --graph1FilePath ./adder_altCorrect.pkl --dumpDGLGraph 1 --createStdCellLibLUT 1
python3 runGatspi.py --top_name adder --graph0FilePath ./adder.pkl --graph1FilePath ./adder_altIncorrect.pkl --dumpDGLGraph 1
#"python3 runGatspi.py --help" will print the usage.
#The results of the simulation comparison is printed to STDOUT
#Generally, we'll want --dumpDGLGraph 1 if either netlist changes
#Generally, only need to run with --createStdCellLibLUT 1 once
```
