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

Sample testing bytecode :

```
0x6006 PUSH1 0x06
0x6007 PUSH1 0x07
0x02 MUL 
0x50 POP // Return 42
0x600660070250 => POP 42 from the stack
```
Usage :
```
./cuEVM --bytecode 0x600660070250 --input 0x1234
Bytecode: 60 06 60 07 02 50 
Input: 12 34 
PUSH1 OPCODE: 
push_val: 6 0 0 0 0 0 0 0 
***************
PUSH1 OPCODE: 
push_val: 7 0 0 0 0 0 0 0 
***************
MUL OPCODE: 
op1: 7 0 0 0 0 0 0 0 op2: 6 0 0 0 0 0 0 0 result: 42 0 0 0 0 0 0 0 
***************
Popped Stack value: 42 0 0 0 0 0 0 0 
***************
```

TODO: change options, configs, and how we use the tool in the future.

## Code structure
TODO
