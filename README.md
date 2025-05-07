# MLCAD2025_Contest_Sim

## Introduction

This is a branch of GL0AM that is serving the purpose of a logic cone simulator/verifier for the 2025 MLCAD Design Contest: https://asu-vda-lab.github.io/MLCAD25-Contest/ . As such, we can view it as a 'subset' form of GL0AM that _only_ simulates the combinational logic cones for a golden netlist, vs. a resynthesized netlist. The stimuli for the logic cones are created randomly, and at the end of their respective simulations, the results for the logic cone endpoints for both netlists are compared. 

## Prerequisites, packages, installation

We used a [Dockerfile](Dockerfile) to build a docker for the environment to run this simulator. Since this simulator is tied to the [2025 MLCAD Design Contest](https://asu-vda-lab.github.io/MLCAD25-Contest/), the Dockerfile is very similar to the one for the design contest. For the most part, the following is needed:

### 1. Hardware Platform
  * Developed on NVIDIA GV100 GPU and Intel Xeon Platinum 8174 CPU. (But, most any GPUs should work)

### 2. Software Platform
  * OS: Ubuntu 20.04.5
  * CUDA: nvcc-12.3
  * CUDA driver: 550.90.07 or similar
  * Rust: 1.85.1 (though 1.82 and/or above should work)
  * Python: Python-3.8.10, with the following packages:
    * PyTorch: 2.2.0+cu121
    * DGL: 1.2
    * CuPy: 12.2.0 or similar
  
Licenses for the 3rd party software can be found in [LICENSES.txt](LICENSES.txt).

## Setup and Trial Run
### 0. Install Rustc and Python packages (if not using docker), Add a Git Submodule)
```
git submodule add  https://github.com/gzz2000/eda-infra-rs.git
#build the rustc executable that translates Verilog into simulation graph
cd <TOP_DIR>
cargo build
#if you see "Finished `dev` ..." then rustc portion is all set to go!
```

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
