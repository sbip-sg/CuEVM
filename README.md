# CuEVM
Cuda implementation of EVM bytecode executor

## Prerequisites
- CUDA Toolkit (Version 12.0+, because we use `--std c++20`)
- A CUDA-capable GPU (CUDA compute capabilily 7+ other older GPUs compability are not tested fully)
- A C++ compiler compatible with the CUDA Toolkit (gcc/g++ version 10+)
- For docker image, you dont need the above but the system with docker installed

## Compile and Run
There are two methods, one requires installing all prequisited in the system, the second one use docker image:

### On your own system

Note : In Makefile, there is one `ENABLE_TRACING` flag that is required for now to compare the results against reference REVM and to demonstrate bug detection. It will slowdown the execution.

* Example : `make ENABLE_TRACING=1 interpreter`
* `mkdir out` folder if it does not exist

Building on Ubuntu (with sudo):
* Setup required libraries: `sudo apt install libgmp-dev`
* Setup cJSON: `sudo apt install libcjson-dev`
* `make interpreter` or for running with cpu :`make cpu_interpreter`


Building without sudo is also possible with extra configuration and modification on the Makefile or evironment variables, please refer to online tutorials on how to build and use libraries in custom directory

#### Building using docker image:
* Build the docker image first: `docker build -f .devcontainer/Dockerfile -t cuevm`
* Run and mount the current code folder `docker run -it -v $(pwd):/CuEVM cuevm`
* `cd CuEVM` then `make interpreter` or for running with cpu: `make cpu_interpreter`

#### CMake

* `mkdir build`
* `cmake -S . -B build ` (to build only CPU version : `-DONLY_CPU=ON for debug -DCMAKE_BUILD_TYPE=Debug)
* `cmake --build build`


## Usage
`{interpreter}` is `build/cuevm` for CMake build and one of the `out/cpu_interpreter` or `out/interpreter`
#### Demo of functionality for testing transaction sequences:
Please refer to subfolder `samples/README.md` for testing and demo how to use this CuEVM.

#### Testing of the EVM using ethtest:
Please refer to `scripts/run-ethtest-by-fork`.

`python scripts/run-ethtest-by-fork.py -i ./tests/GeneralStateTests -t ./tmp --runtest-bin runtest --geth gethvm --cuevm ./build/cuevm --ignore-errors`



## Tool usage [TODO after completion]
* `clear && compute-sanitizer --tool memcheck --leak-check=full {gpu_interpreter} --input [input_json_file] --output [output_json_file]`
* `clear && valgrind --leak-check=full --show-leak-kinds=all {cpu_interpreter}  --input [input_json_file] --output [output_json_file]`
* `clear && valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./build/cuevm  --input [input_json_file] --output [output_json_file]`
Or for running with cpu
* `{interpreter} --input [input_json_file] --output [output_json_file]`
Easy test:
* `{interpreter} --input ./input/evm_arith.json --output ./output/evm_test.json`



## Code structure
TODO

## Documentation
TODO
docygen+sphinx+breathe+exhale


# Test results



## GeneralStateTests without stateRoot comparison

We use the test files in
[ethereum/](https://github.com/ethereum/tests/tree/develop/GeneralStateTests)
to test whether we can get the same results with the go-ethereum. To run the tests,

- we take the test json files from `tests/GeneralStateTests`
- extract the tests which targeted at the Shanghai fork
- use [cassc/goevmlab](https://github.com/cassc/goevmlab) `runtest` to compare the results between `geth` and `cuevm`

These tests are ignored, they contain some stress tests which could crash the EVM:

- stCreateTest
- stQuadraticComplexityTest
- stStaticCall
- stTimeConsuming


### Test results by comparing the traces without stateRoot comparison

> Note that there can be multiple tests in one input json, the number of tests
> shown below can be larger than number of input files.


``` text
Cancun.log:Test result, Passed: 0, Failed: 0, Skipped: 37
Pyspecs.log:Test result, Passed: 201, Failed: 2, Skipped: 1233
Python.log:Test result, Passed: 1, Failed: 0, Skipped: 4
Shanghai.log:Test result, Passed: 20, Failed: 3, Skipped: 0
stArgsZeroOneBalance.log:Test result, Passed: 95, Failed: 1, Skipped: 0
stAttackTest.log:Test result, Passed: 0, Failed: 2, Skipped: 0
stBadOpcode.log:Test result, Passed: 3542, Failed: 118, Skipped: 1
stBugs.log:Test result, Passed: 9, Failed: 0, Skipped: 0
stCallCodes.log:Test result, Passed: 75, Failed: 10, Skipped: 1
stCallCreateCallCodeTest.log:Test result, Passed: 39, Failed: 9, Skipped: 0
stCallDelegateCodesCallCodeHomestead.log:Test result, Passed: 51, Failed: 7, Skipped: 0
stCallDelegateCodesHomestead.log:Test result, Passed: 51, Failed: 7, Skipped: 0
stChainId.log:Test result, Passed: 2, Failed: 0, Skipped: 0
stCodeCopyTest.log:Test result, Passed: 1, Failed: 1, Skipped: 1
stCodeSizeLimit.log:Test result, Passed: 7, Failed: 0, Skipped: 0
stCreate2.log:Test result, Passed: 124, Failed: 16, Skipped: 3
stDelegatecallTestHomestead.log:Test result, Passed: 23, Failed: 6, Skipped: 0
stEIP150singleCodeGasPrices.log:Test result, Passed: 115, Failed: 5, Skipped: 1
stEIP150Specific.log:Test result, Passed: 24, Failed: 1, Skipped: 0
stEIP158Specific.log:Test result, Passed: 6, Failed: 2, Skipped: 3
stEIP1559.log:Test result, Passed: 1817, Failed: 2, Skipped: 1
stEIP2930.log:Test result, Passed: 15, Failed: 5, Skipped: 0
stEIP3607.log:Test result, Passed: 12, Failed: 0, Skipped: 2
stExample.log:Test result, Passed: 34, Failed: 4, Skipped: 0
stExtCodeHash.log:Test result, Passed: 23, Failed: 21, Skipped: 3
stHomesteadSpecific.log:Test result, Passed: 5, Failed: 0, Skipped: 0
stInitCodeTest.log:Test result, Passed: 20, Failed: 2, Skipped: 0
stLogTests.log:Test result, Passed: 46, Failed: 0, Skipped: 0
stMemExpandingEIP150Calls.log:Test result, Passed: 9, Failed: 1, Skipped: 0
stMemoryStressTest.log:Test result, Passed: 82, Failed: 0, Skipped: 0
stMemoryTest.log:Test result, Passed: 386, Failed: 7, Skipped: 0
stNonZeroCallsTest.log:Test result, Passed: 23, Failed: 1, Skipped: 12
stPreCompiledContracts.log:Test result, Passed: 564, Failed: 1, Skipped: 1
stPreCompiledContracts2.log:Test result, Passed: 189, Failed: 17, Skipped: 0
stRandom.log:Test result, Passed: 305, Failed: 9, Skipped: 0
stRandom2.log:Test result, Passed: 221, Failed: 5, Skipped: 0
stRecursiveCreate.log:Test result, Passed: 2, Failed: 0, Skipped: 0
stRefundTest.log:Test result, Passed: 10, Failed: 12, Skipped: 1
stReturnDataTest.log:Test result, Passed: 57, Failed: 10, Skipped: 0
stRevertTest.log:Test result, Passed: 185, Failed: 10, Skipped: 11
stSelfBalance.log:Test result, Passed: 26, Failed: 2, Skipped: 0
stShift.log:Test result, Passed: 42, Failed: 0, Skipped: 0
stSLoadTest.log:Test result, Passed: 1, Failed: 0, Skipped: 0
stSolidityTest.log:Test result, Passed: 23, Failed: 0, Skipped: 0
stSpecialTest.log:Test result, Passed: 20, Failed: 2, Skipped: 3
stSStoreTest.log:Test result, Passed: 205, Failed: 17, Skipped: 1
stStackTests.log:Test result, Passed: 185, Failed: 3, Skipped: 0
stStaticFlagEnabled.log:Test result, Passed: 1, Failed: 12, Skipped: 0
stSystemOperationsTest.log:Test result, Passed: 71, Failed: 8, Skipped: 1
stTransactionTest.log:Test result, Passed: 159, Failed: 7, Skipped: 2
stTransitionTest.log:Test result, Passed: 6, Failed: 0, Skipped: 0
stWalletTest.log:Test result, Passed: 20, Failed: 26, Skipped: 0
stZeroCallsRevert.log:Test result, Passed: 16, Failed: 0, Skipped: 8
stZeroCallsTest.log:Test result, Passed: 24, Failed: 0, Skipped: 12
stZeroKnowledge.log:Test result, Passed: 681, Failed: 29, Skipped: 1
stZeroKnowledge2.log:Test result, Passed: 485, Failed: 10, Skipped: 3
VMTests.log:Test result, Passed: 567, Failed: 10, Skipped: 0
```


### Test results by comparing both the traces and the stateRoot

``` text
Cancun.log:Test result, Passed: 0, Failed: 0, Skipped: 37
Pyspecs.log:Test result, Passed: 201, Failed: 2, Skipped: 1233
Python.log:Test result, Passed: 1, Failed: 0, Skipped: 4
Shanghai.log:Test result, Passed: 1, Failed: 8, Skipped: 0
stArgsZeroOneBalance.log:Test result, Passed: 72, Failed: 16, Skipped: 0
stAttackTest.log:Test result, Passed: 0, Failed: 2, Skipped: 0
stBadOpcode.log:Test result, Passed: 27, Failed: 119, Skipped: 1
stBugs.log:Test result, Passed: 1, Failed: 4, Skipped: 0
stCallCodes.log:Test result, Passed: 3, Failed: 77, Skipped: 1
stCallCreateCallCodeTest.log:Test result, Passed: 17, Failed: 29, Skipped: 0
stCallDelegateCodesCallCodeHomestead.log:Test result, Passed: 0, Failed: 58, Skipped: 0
stCallDelegateCodesHomestead.log:Test result, Passed: 0, Failed: 58, Skipped: 0
stChainId.log:Test result, Passed: 0, Failed: 2, Skipped: 0
stCodeCopyTest.log:Test result, Passed: 0, Failed: 2, Skipped: 1
stCodeSizeLimit.log:Test result, Passed: 4, Failed: 3, Skipped: 0
stCreate2.log:Test result, Passed: 13, Failed: 46, Skipped: 3
stDelegatecallTestHomestead.log:Test result, Passed: 6, Failed: 22, Skipped: 0
stEIP150singleCodeGasPrices.log:Test result, Passed: 0, Failed: 39, Skipped: 1
stEIP150Specific.log:Test result, Passed: 1, Failed: 13, Skipped: 0
stEIP158Specific.log:Test result, Passed: 1, Failed: 7, Skipped: 3
stEIP1559.log:Test result, Passed: 1751, Failed: 8, Skipped: 1
stEIP2930.log:Test result, Passed: 12, Failed: 6, Skipped: 0
stEIP3607.log:Test result, Passed: 12, Failed: 0, Skipped: 2
stExample.log:Test result, Passed: 29, Failed: 9, Skipped: 0
stExtCodeHash.log:Test result, Passed: 0, Failed: 36, Skipped: 3
stHomesteadSpecific.log:Test result, Passed: 3, Failed: 2, Skipped: 0
stInitCodeTest.log:Test result, Passed: 6, Failed: 12, Skipped: 0
stLogTests.log:Test result, Passed: 11, Failed: 35, Skipped: 0
stMemExpandingEIP150Calls.log:Test result, Passed: 1, Failed: 8, Skipped: 0
stMemoryStressTest.log:Test result, Passed: 78, Failed: 2, Skipped: 0
stMemoryTest.log:Test result, Passed: 24, Failed: 47, Skipped: 0
stNonZeroCallsTest.log:Test result, Passed: 6, Failed: 18, Skipped: 12
stPreCompiledContracts.log:Test result, Passed: 13, Failed: 9, Skipped: 1
stPreCompiledContracts2.log:Test result, Passed: 76, Failed: 84, Skipped: 0
stRandom.log:Test result, Passed: 121, Failed: 193, Skipped: 0
stRandom2.log:Test result, Passed: 89, Failed: 137, Skipped: 0
stRecursiveCreate.log:Test result, Passed: 0, Failed: 2, Skipped: 0
stRefundTest.log:Test result, Passed: 0, Failed: 22, Skipped: 1
stReturnDataTest.log:Test result, Passed: 3, Failed: 38, Skipped: 0
stRevertTest.log:Test result, Passed: 47, Failed: 37, Skipped: 11
stSelfBalance.log:Test result, Passed: 0, Failed: 6, Skipped: 0
stShift.log:Test result, Passed: 0, Failed: 42, Skipped: 0
stSLoadTest.log:Test result, Passed: 0, Failed: 1, Skipped: 0
stSolidityTest.log:Test result, Passed: 4, Failed: 14, Skipped: 0
stSpecialTest.log:Test result, Passed: 5, Failed: 10, Skipped: 3
stSStoreTest.log:Test result, Passed: 0, Failed: 28, Skipped: 1
stStackTests.log:Test result, Passed: 185, Failed: 3, Skipped: 0
stStaticFlagEnabled.log:Test result, Passed: 0, Failed: 13, Skipped: 0
stSystemOperationsTest.log:Test result, Passed: 26, Failed: 43, Skipped: 1
stTransactionTest.log:Test result, Passed: 110, Failed: 21, Skipped: 2
stTransitionTest.log:Test result, Passed: 0, Failed: 6, Skipped: 0
stWalletTest.log:Test result, Passed: 4, Failed: 40, Skipped: 0
stZeroCallsRevert.log:Test result, Passed: 12, Failed: 4, Skipped: 8
stZeroCallsTest.log:Test result, Passed: 6, Failed: 18, Skipped: 12
stZeroKnowledge.log:Test result, Passed: 410, Failed: 33, Skipped: 1
stZeroKnowledge2.log:Test result, Passed: 485, Failed: 10, Skipped: 3
VMTests.log:Test result, Passed: 232, Failed: 58, Skipped: 0
```
