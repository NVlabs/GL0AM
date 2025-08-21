#!/bin/bash
#export CUDA_LIBRARY_PATH to your CUDA library tool kit path
#add path to nvcc, cargo to your $PATH environment variable
#add CUDA_LIBRARY_PATH to LD_LIBRARY_PATH

#usage: ./regression.sh [--skip-graph-build] [--gpu-arch V100|custom <NVDLAM> <QADD1000> <NVDLA>]
#  V100: sets SUBCHUNKS_NVDLAM=10, SUBCHUNKS_QADD1000=1, SUBCHUNKS_NVDLA=12
#  custom: requires 3 values for all three subchunks variables

# Set subchunks variables
SUBCHUNKS_NVDLAM=10
SUBCHUNKS_QADD1000=1
SUBCHUNKS_NVDLA=12

# Check if skip flags are specified.
SKIP_GRAPH_BUILD=false
GPU_ARCH=""

for arg in "$@"; do
 case $arg in
  --skip-graph-build)
   SKIP_GRAPH_BUILD=true
   shift
   ;;
  --gpu-arch)
   shift
   GPU_ARCH="$1"
   if [ "$GPU_ARCH" = "V100" ]; then
    SUBCHUNKS_NVDLAM=10
    SUBCHUNKS_QADD1000=1
    SUBCHUNKS_NVDLA=12
    echo "GPU architecture set to V100:"
    echo "  SUBCHUNKS_NVDLAM=10"
    echo "  SUBCHUNKS_QADD1000=1"
    echo "  SUBCHUNKS_NVDLA=12"
   elif [ "$GPU_ARCH" = "custom" ]; then
    shift
    if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
     echo "ERROR: --gpu-arch custom requires 3 values after 'custom'"
     echo "Usage: --gpu-arch custom <NVDLAM_value> <QADD1000_value> <NVDLA_value>"
     exit 1
    fi
    SUBCHUNKS_NVDLAM="$1"
    SUBCHUNKS_QADD1000="$2"
    SUBCHUNKS_NVDLA="$3"
    echo "GPU architecture set to custom:"
    echo "  SUBCHUNKS_NVDLAM=$SUBCHUNKS_NVDLAM"
    echo "  SUBCHUNKS_QADD1000=$SUBCHUNKS_QADD1000"
    echo "  SUBCHUNKS_NVDLA=$SUBCHUNKS_NVDLA"
   else
    echo "ERROR: Invalid GPU architecture '$GPU_ARCH'"
    echo "Valid options: V100 or custom <value>"
    exit 1
   fi
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

cd gatspi; 
python3 runGatspi.py --topName qadd_pipe --testname regression --graphFilePath ../qadd_pipe.pkl \
--inputTraceFile ../../GATSPIDataset/Waveforms/qadd_pipe.waveforms_part0 --duration 6000000 --period 500 --numOfSubchunks 1 \
--dumpDGLGraph 1 --createStdCellLibLUT 1
../target/release/saif_dumper ../../GATSPIDataset/qadd_pipe/qadd_pipe.golden.saif qadd_pipe_regression_6000000ps.saif 0 0 > regression.log 2>&1
 if grep -q "panic" regression.log; then
  echo "ERROR: Panic detected in qadd_pipe simulation. Exiting."
  exit 1
 fi

python3 runGatspi.py --topName jpeg --testname regression --graphFilePath ../jpeg.pkl \
--inputTraceFile ../../GATSPIDataset/Waveforms/jpeg_encoder.waveforms_part0 --duration 40000000 --period 2000 --numOfSubchunks 1 \
--dumpDGLGraph 1
../target/release/saif_dumper ../../GATSPIDataset/jpeg/jpeg.golden.saif jpeg_regression_40000000ps.saif 0 0 > regression.log 2>&1
 if grep -q "panic" regression.log; then
  echo "ERROR: Panic detected in jpeg simulation. Exiting."
  exit 1
 fi

python3 runGatspi.py --topName NVDLA_m --testname regression --graphFilePath ../NVDLA_m.pkl \
--inputTraceFile ../../GATSPIDataset/Waveforms/NV_NVDLA_partition_m.waveforms_part0 --duration 1199984000 --period 2000 --numOfSubchunks $SUBCHUNKS_NVDLAM \
--dumpDGLGraph 1
../target/release/saif_dumper ../../GATSPIDataset/NVDLA_m/NVDLA_m.golden.saif NVDLA_m_regression_1199984000ps.saif 0 0 > regression.log 2>&1
 if grep -q "panic" regression.log; then
  echo "ERROR: Panic detected in NVDLA_m simulation. Exiting."
  exit 1
 fi
 
python3 runGatspi.py --topName qadd_pipe1000 --testname regression --graphFilePath ../qadd_pipe1000.pkl \
--inputTraceFile ../../GATSPIDataset/Waveforms/qadd_pipe1000.waveforms_part0 --duration 3000000 --period 500 --numOfSubchunks $SUBCHUNKS_QADD1000 \
--dumpDGLGraph 1
../target/release/saif_dumper ../../GATSPIDataset/qadd_pipe1000/qadd_pipe1000.golden.saif qadd_pipe1000_regression_3000000ps.saif 0 0 > regression.log 2>&1
 if grep -q "panic" regression.log; then
  echo "ERROR: Panic detected in qadd_pipe1000 simulation. Exiting."
  exit 1
 fi
 
python3 runGatspi.py --topName NV_nvdla --testname regression --graphFilePath ../NV_nvdla.pkl \
--inputTraceFile ../../GATSPIDataset/Waveforms/NV_nvdla.waveforms_part0 --duration 1299976000 --period 2000 --numOfSubchunks $SUBCHUNKS_NVDLA \
--dumpDGLGraph 1
../target/release/saif_dumper ../../GATSPIDataset/NVDLA/NV_nvdla.golden.saif NV_nvdla_regression_1299976000ps.saif 0 0 > regression.log 2>&1
 if grep -q "panic" regression.log; then
  echo "ERROR: Panic detected in NV_nvdla simulation. Exiting."
  exit 1
 fi

echo "Regression complete! ALL PASSED!"
cd ../
