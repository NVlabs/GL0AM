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
```

### 1. Convert precompiled Python lilmatrix graph to DGL graph
```
python3 GL0AM.generateGraph.py --block qadd_pipe1000 --graphPrecompile qadd_pipe1000_GraphPrecompile.pkl.gz
python3 GL0AM.splitLargeFOs.py --combo_graph qadd_pipe1000_fullSDF_DGLGraph --target_combo_graph qadd_pipe1000_combo_DGLGraph
```
### 2. Partition the combinational logic graph
```
python3 GL0AM.partitioningSetup.py --block qadd_pipe1000 --graph4 qadd_pipe1000_combo_DGLGraph
hmetis qadd_pipe1000.hgr 160 5 10 4 1 1 0 0
python3 GL0AM.partitioningPostprocessing.py --block qadd_pipe1000 --graph2 qadd_pipe1000_update_DGLGraph \
--graph3 qadd_pipe1000_sram_DGLGraph --graph4 qadd_pipe1000_combo_DGLGraph --hMetisResults qadd_pipe1000.hgr.part.160 --partitions 160 \
--graphPrecompile qadd_pipe1000_GraphPrecompile.pkl.gz --graph qadd_pipe1000_fullSDF_DGLGraph
```
### 3. Compile input waveforms, compile graph information to simulation array information, run GL0AM simulation, output 4-value SAIF file.
```
python3 GL0AM.runSim.py --block qadd_pipe1000 --instance_name whatever  --testname random \
--cycles 3000 --period 1000 --input_trace_file qadd_pipe1000.random.waveforms_part0.gz \
--graphPrecompile qadd_pipe1000_GraphPrecompile.pkl.gz --graph qadd_pipe1000_fullSDF_DGLGraph \
--graph2 qadd_pipe1000_update_DGLGraph --graph3 qadd_pipe1000_sram_DGLGraph --graph4 qadd_pipe1000_combo_DGLGraph \
--timescale 1 --pin_net_file qadd_pipe1000_GraphPrecompile.CellNetTranslation --num_splits 32 --duration 3000000 --clk clk \
--first_edge 500 --num_of_sub_chunks 1 --zeroDelayReportSignals UI_0__add0_GL0AMs_a_pipe[31:0] UI_0__add0_GL0AMs_b_pipe[31:0] c[31:0] \
--clkTreeFile qadd_pipe1000_GraphPrecompile.clkTreePins --partitionsFile qadd_pipe1000.160.partitions
```
