#!/bin/bash

# Load necessary modules
module load PrgEnv-gnu
module load cudatoolkit
module load cray-mpich

# Set environment variables for GPU-aware MPI
export CRAY_ACCEL_TARGET=nvidia80
export MPICH_GPU_SUPPORT_ENABLED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Run with Slurm (use `srun` instead of `mpirun`)
srun --ntasks=4 --gpus-per-task=1 --gpu-bind=closest ./nccl_test
