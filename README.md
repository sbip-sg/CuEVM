# CuEVM
Cuda implementation of EVM bytecode executor

## Prerequisites

- CUDA Toolkit (tested with version 10.1)
- A CUDA-capable GPU (cuda compute capability x.x TODO)
- A C++ compiler compatible with the CUDA Toolkit

## Compile
* `make interpreter`
Or for running with cpu
* `make cpu_interpreter`

## Usage 

* `compute-sanitizer ./out/interpreter --input [inpot_json_file] --output [output_json_file]`
Or for running with cpu
* `./out/cpu_interpreter --input [inpot_json_file] --output [output_json_file]`
Easy test:
* `./out/cpu_interpreter --input ./input/evm_arith.json --output ./output/evm_test.json`

TODO: change options, configs, and how we use the tool in the future.

## Code structure
TODO
