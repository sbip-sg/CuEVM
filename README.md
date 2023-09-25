# CuEVM
Cuda implementation of EVM bytecode executor

## Prerequisites

- CUDA Toolkit (tested with version 10.1)
- A CUDA-capable GPU (cuda compute capability x.x TODO)
- A C++ compiler compatible with the CUDA Toolkit

## Compile
* `make`
Or with custom nvcc path
* `make NVCC=/usr/local/cuda-10.0/bin/nvcc`

## Usage 

* `./cuEVM --bytecode [hex string of byte code] --input [hex string of input]`

TODO: change options, configs, and how we use the tool in the future.

## Code structure
TODO
