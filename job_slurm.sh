#!/bin/bash

#SBATCH --job-name=cuda-compile-run       # Job name
#SBATCH --nodes=1                         # Run on a single node
#SBATCH --partition=standard
#SBATCH --gpus=1
#SBATCH --time=00:05:00                   # Time limit hrs:min:sec
#SBATCH --output=slurm-%j.out             # Standard output and error log

# module load cuda/10.0                     # Load the CUDA module (adjust version as needed)
cd ~/github/CuEVM                     # Change to your repository's directory
make                                      # Assuming you have a Makefile to build your project
./cuEVM --bytecode 0x1234 --input 0x1234                     # Run your program


