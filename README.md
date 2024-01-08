# CuEVM
Cuda implementation of EVM bytecode executor

## Prerequisites
- CUDA Toolkit (Version 12.0+, because we use `--std c++20`)
- A CUDA-capable GPU (CUDA compute capabilily 7+ other older GPUs compability are not tested fully)
- A C++ compiler compatible with the CUDA Toolkit (gcc/g++ version 10+)
- For docker image, you dont need the above but the system with docker installed

## Compile and Run
There are two methods, one requires installing all prequisited in the system, the second one use docker image:

#### On your own system

Note : In Makefile, there is one `ENABLE_TRACING` flag that is required for now to compare the results against reference REVM and to demonstrate bug detection. It will slowdown the execution.

* Example : `make ENABLE_TRACING=1 interpreter`
* `mkdir out` folder if it does not exist

Building on Ubuntu (with sudo):
* Setup required libraries: `sudo apt install libgmp-dev`
* Setup cJSON: `git clone https://github.com/DaveGamble/cJSON ; cd cJSON ; make ; sudo make install`
* `make interpreter` or for running with cpu :`make cpu_interpreter`

Building without sudo is also possible with extra configuration and modification on the Makefile or evironment variables, please refer to online tutorials on how to build and use libraries in custom directory

Building using docker image:
* Build the docker image first: `docker build -f .devcontainer/Dockerfile -t cuevm`
* Run and mount the current code folder `docker run -it -v $(pwd):/CuEVM cuvm`
* `cd CuEVM` then `make interpreter` or for running with cpu: `make cpu_interpreter`

## Usage

#### Demo of functionality for testing transaction sequences:
Please refer to subfolder `samples/README.md` for testing and demo how to use this CuEVM.
#### Testing of the EVM using ethtest:
Please refer to `REVMI/` for usage.

For example, test all json files in VMTest:

`cd REVMI ; ./evm-interpreter compare --executable ../out/interpreter --test-json dev-resources/ethtest/GeneralStateTests/VMTests/ --limit 10000`



## Tool usage [TODO after completion]
* `clear && compute-sanitizer --tool memcheck --leak-check=full ./out/interpreter --input [inpot_json_file] --output [output_json_file]`
* `clear && valgrind --leak-check=full --show-leak-kinds=all ./out/*`
Or for running with cpu
* `./out/cpu_interpreter --input [inpot_json_file] --output [output_json_file]`
Easy test:
* `./out/cpu_interpreter --input ./input/evm_arith.json --output ./output/evm_test.json`



## Code structure
TODO

## Documentation
TODO
docygen+sphinx+breathe+exhale