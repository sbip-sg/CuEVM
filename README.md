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

Loop with jumpi :
```
LOGIC:
1. Perform 6 * 7 = res;
2. Loop: while(res!=0): res = res - 14 (loop 3 times) => STOP

PC 0 : 0x6006 PUSH1 0x06
PC 2 : 0x6007 PUSH1 0x07
PC 4 : 0x02 MUL  // TOP STACK 42
PC 5 : 0x5b JUMPDEST  // TAG1_JUMP
PC 6 : 0x600e PUSH 0x0e
PC 8 : 0x90 SWAP1
PC 9 : 0x03 SUB  // 42 - 14 // condition != 1 jump;
PC 10 : 0x80 DUP1  // DUP the result because JUMPI will remove it
PC 11 : 0x6005 PUSH1 TAG1_JUMP // destination
PC 13 : 0x57 JUMP I
PC 14 : 0x50 POP // for testing
PC 15 : 0xf3 RETURN // for testing
Bytecode:
0x60066007025b600e90038060055750f3
```
Run
```
./cuEVM --bytecode 0x60066007025b600e90038060055750f3 --input 0x1234
```

Reference tools (www.evm.codes) for testing bytecode sequence : [Simulate test bytecode sequence](https://www.evm.codes/playground?fork=shanghai&unit=Wei&codeType=Bytecode&code=%27%7E6%7E7025b%7Ee900380%7E55750f3%27%7E600%01%7E_)

TODO: change options, configs, and how we use the tool in the future.

## Code structure
TODO
