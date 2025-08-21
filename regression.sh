#!/bin/bash
#export CUDA_LIBRARY_PATH to your CUDA library tool kit path
#add path to nvcc, cargo to your $PATH environment variable
#add CUDA_LIBRARY_PATH to LD_LIBRARY_PATH

#usage: ./regression.sh [--skip-graph-build]

# Check if skip flags are specified.
SKIP_GRAPH_BUILD=false

for arg in "$@"; do
 case $arg in
  --skip-graph-build)
   SKIP_GRAPH_BUILD=true
   shift
   ;;
 esac
done


if [ "$SKIP_GRAPH_BUILD" = false ]; then
 #build the graphs for the regression
 echo "building qadd_pipe graph..."
 ./target/release/build_gatspi_graph ../GATSPIDataset/qadd_pipe/qadd_pipe.GEN.gv qadd_pipe.pkl qadd_pipe ../GATSPIDataset/qadd_pipe/qadd_pipe.GEN.sdf  > regression.log 2>&1
 if grep -q "panic" regression.log; then
  echo "ERROR: Panic detected in qadd_pipe graph build. Exiting."
  exit 1
 fi

 echo "building jpeg graph..."
 ./target/release/build_gatspi_graph ../GATSPIDataset/jpeg/jpeg.GEN.gv jpeg.pkl jpeg_encoder ../GATSPIDataset/jpeg/jpeg.GEN.sdf  > regression.log 2>&1
 if grep -q "panic" regression.log; then
  echo "ERROR: Panic detected in jpeg graph build. Exiting."
  exit 1
 fi

 echo "building NVDLA_m graph..."
 ./target/release/build_gatspi_graph ../GATSPIDataset/NVDLA_m/NVDLA_m.GEN.gv NVDLA_m.pkl NV_NVDLA_partition_m ../GATSPIDataset/NVDLA_m/NVDLA_m.GEN.sdf  > regression.log 2>&1
 if grep -q "panic" regression.log; then
  echo "ERROR: Panic detected in NVDLA_m graph build. Exiting."
  exit 1
 fi

 echo "building qadd_pipe1000 graph..."
 ./target/release/build_gatspi_graph ../GATSPIDataset/qadd_pipe1000/qadd_pipe1000.GEN.gv qadd_pipe1000.pkl qadd_pipe1000 ../GATSPIDataset/qadd_pipe1000/qadd_pipe1000.GEN.sdf  > regression.log 2>&1
 if grep -q "panic" regression.log; then
  echo "ERROR: Panic detected in qadd_pipe1000 graph build. Exiting."
  exit 1
 fi

 echo "building NVDLA graph..."
 ./target/release/build_gatspi_graph ../GATSPIDataset/NVDLA/NV_nvdla.GEN.gv NV_nvdla.pkl NV_nvdla ../GATSPIDataset/NVDLA/NV_nvdla.GEN.sdf  > regression.log 2>&1
 if grep -q "panic" regression.log; then
  echo "ERROR: Panic detected in NVDLA graph build. Exiting."
  exit 1
 fi

 echo "All graph builds completed successfully!"
else
 echo "Skipping graph building...assuming .pkl graphs already built..."
fi
