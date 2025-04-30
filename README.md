# MLCAD2025_Contest_Sim

## Introduction

This is a branch of GL0AM that is serving the purpose of a logic cone simulator/verifier for the 2025 MLCAD Design Contest: https://asu-vda-lab.github.io/MLCAD25-Contest/ . 

## Prerequisites, packages, installation, and dataset

GL0AM was developed using the following platform:

### 1. Hardware Platform
  * Developed on NVIDIA GV100 GPU and Intel Xeon Platinum 8174 CPU
  * Performance metrics gathered on NVIDIA H100 GPU and 80GB Intel Xeon Gold 6136 CPU

### 2. Software Platform
  * OS: Ubuntu 20.04.5
  * CUDA: nvcc-11.8
  * CUDA driver: 550.90.07
  * C/C++: gcc-9.4.0
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
  
  We used a docker container to manage our software platform, an example installation script can be found in [install/packages.sh](install/packages.sh). Licenses for the 3rd party software can be found in [LICENSES.txt](LICENSES.txt).

   
### 3. Dataset
Input datasets are from open sources, and can be found at [this link](https://drive.google.com/drive/folders/1VIeTu6O_yIVv1qkEpi-qSUaYuhhC4ovK?usp=sharing) . Due to some of the compilation process still in progress of transitioning to fully open source, some precompiled graph data format is also included, for now.

## Trial Run
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
