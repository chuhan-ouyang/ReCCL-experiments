#!/bin/bash

# Load required modules
module load PrgEnv-gnu
module load cudatoolkit
module load cray-mpich

# Set accelerator target for GPU-aware MPI
export CRAY_ACCEL_TARGET=nvidia80

# Define paths for NCCL and CUDA
NCCL_PATH=/global/common/software/nersc9/nccl/2.21.5
CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4
MPI_PATH=/opt/cray/pe/mpich/8.1.30/ofi/gnu/12.3

# Compile the CUDA code separately (include NCCL headers)
nvcc -c nccl_test.cu -o nccl_test.o \
    -I${NCCL_PATH}/include -I${CUDA_PATH}/include -I${MPI_PATH}/include

# Link using Cray compiler wrapper (ensuring NCCL and CUDA libraries are included)
CC -o nccl_test nccl_test.o \
    -I${NCCL_PATH}/include -I${CUDA_PATH}/include -I${MPI_PATH}/include \
    -L${MPI_PATH}/lib -lmpi_gnu_123 \
    -L${NCCL_PATH}/lib -lnccl \
    -L${CUDA_PATH}/lib64 -lcudart \
    -target-accel=nvidia80
