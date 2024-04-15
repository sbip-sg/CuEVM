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
[ethereum/GeneralStateTests](https://github.com/ethereum/tests/tree/develop/GeneralStateTests)
to test whether we can get the same results with the go-ethereum. To run the tests,

- build the binary with the debug flag on: `cpu_debug_interpreter`
- either download or build the geth binary, in this test the version `v1.13.14` is used, later version will probably produce the same result
- we take the test json files from `tests/GeneralStateTests`
- extract the tests which targeted at the Shanghai fork
- build `runtest` binary from [cassc/goevmlab](https://github.com/cassc/goevmlab) which contains the CuEVM driver, to compare the results between `geth` and `cuevm`

These tests are ignored, they contain some stress tests which could crash the EVM as well as the test script itself:

- stCreateTest
- stQuadraticComplexityTest
- stStaticCall
- stTimeConsuming


### Test results by comparing the traces between geth and cuevm without stateRoot comparison

The tests results are collected by running the [Python script](https://gist.github.com/cassc/b300005b38d7c01461b443ef67169659) from the [ethereum](https://github.com/ethereum/tests) root folder:

``` bash
python ./run-ethtest-without-stateroot-comparison.py -t /tmp/out --runtest-bin runtest --geth geth --cuevm cuevm  --ignore-errors
```

> Note that there can be multiple tests in one input json, the number of tests shown below can be larger than number of input files.

| Test folder                          | Passed       | Failed      | Skipped       |
|--------------------------------------|--------------|-------------|---------------|
| Cancun                               | Passed: 0    | Failed: 0   | Skipped: 37   |
| Pyspecs                              | Passed: 201  | Failed: 2   | Skipped: 1233 |
| Python                               | Passed: 1    | Failed: 0   | Skipped: 4    |
| Shanghai                             | Passed: 20   | Failed: 3   | Skipped: 0    |
| stArgsZeroOneBalance                 | Passed: 95   | Failed: 1   | Skipped: 0    |
| stAttackTest                         | Passed: 0    | Failed: 2   | Skipped: 0    |
| stBadOpcode                          | Passed: 3542 | Failed: 118 | Skipped: 1    |
| stBugs                               | Passed: 9    | Failed: 0   | Skipped: 0    |
| stCallCodes                          | Passed: 75   | Failed: 10  | Skipped: 1    |
| stCallCreateCallCodeTest             | Passed: 39   | Failed: 9   | Skipped: 0    |
| stCallDelegateCodesCallCodeHomestead | Passed: 51   | Failed: 7   | Skipped: 0    |
| stCallDelegateCodesHomestead         | Passed: 51   | Failed: 7   | Skipped: 0    |
| stChainId                            | Passed: 2    | Failed: 0   | Skipped: 0    |
| stCodeCopyTest                       | Passed: 1    | Failed: 1   | Skipped: 1    |
| stCodeSizeLimit                      | Passed: 7    | Failed: 0   | Skipped: 0    |
| stCreate2                            | Passed: 124  | Failed: 16  | Skipped: 3    |
| stDelegatecallTestHomestead          | Passed: 23   | Failed: 6   | Skipped: 0    |
| stEIP150singleCodeGasPrices          | Passed: 115  | Failed: 5   | Skipped: 1    |
| stEIP150Specific                     | Passed: 24   | Failed: 1   | Skipped: 0    |
| stEIP158Specific                     | Passed: 6    | Failed: 2   | Skipped: 3    |
| stEIP1559                            | Passed: 1817 | Failed: 2   | Skipped: 1    |
| stEIP2930                            | Passed: 15   | Failed: 5   | Skipped: 0    |
| stEIP3607                            | Passed: 12   | Failed: 0   | Skipped: 2    |
| stExample                            | Passed: 34   | Failed: 4   | Skipped: 0    |
| stExtCodeHash                        | Passed: 23   | Failed: 21  | Skipped: 3    |
| stHomesteadSpecific                  | Passed: 5    | Failed: 0   | Skipped: 0    |
| stInitCodeTest                       | Passed: 20   | Failed: 2   | Skipped: 0    |
| stLogTests                           | Passed: 46   | Failed: 0   | Skipped: 0    |
| stMemExpandingEIP150Calls            | Passed: 9    | Failed: 1   | Skipped: 0    |
| stMemoryStressTest                   | Passed: 82   | Failed: 0   | Skipped: 0    |
| stMemoryTest                         | Passed: 386  | Failed: 7   | Skipped: 0    |
| stNonZeroCallsTest                   | Passed: 23   | Failed: 1   | Skipped: 12   |
| stPreCompiledContracts               | Passed: 564  | Failed: 1   | Skipped: 1    |
| stPreCompiledContracts2              | Passed: 189  | Failed: 17  | Skipped: 0    |
| stRandom                             | Passed: 305  | Failed: 9   | Skipped: 0    |
| stRandom2                            | Passed: 221  | Failed: 5   | Skipped: 0    |
| stRecursiveCreate                    | Passed: 2    | Failed: 0   | Skipped: 0    |
| stRefundTest                         | Passed: 10   | Failed: 12  | Skipped: 1    |
| stReturnDataTest                     | Passed: 57   | Failed: 10  | Skipped: 0    |
| stRevertTest                         | Passed: 185  | Failed: 10  | Skipped: 11   |
| stSelfBalance                        | Passed: 26   | Failed: 2   | Skipped: 0    |
| stShift                              | Passed: 42   | Failed: 0   | Skipped: 0    |
| stSLoadTest                          | Passed: 1    | Failed: 0   | Skipped: 0    |
| stSolidityTest                       | Passed: 23   | Failed: 0   | Skipped: 0    |
| stSpecialTest                        | Passed: 20   | Failed: 2   | Skipped: 3    |
| stSStoreTest                         | Passed: 205  | Failed: 17  | Skipped: 1    |
| stStackTests                         | Passed: 185  | Failed: 3   | Skipped: 0    |
| stStaticFlagEnabled                  | Passed: 1    | Failed: 12  | Skipped: 0    |
| stSystemOperationsTest               | Passed: 71   | Failed: 8   | Skipped: 1    |
| stTransactionTest                    | Passed: 159  | Failed: 7   | Skipped: 2    |
| stTransitionTest                     | Passed: 6    | Failed: 0   | Skipped: 0    |
| stWalletTest                         | Passed: 20   | Failed: 26  | Skipped: 0    |
| stZeroCallsRevert                    | Passed: 16   | Failed: 0   | Skipped: 8    |
| stZeroCallsTest                      | Passed: 24   | Failed: 0   | Skipped: 12   |
| stZeroKnowledge                      | Passed: 681  | Failed: 29  | Skipped: 1    |
| stZeroKnowledge2                     | Passed: 485  | Failed: 10  | Skipped: 3    |
| VMTests                              | Passed: 567  | Failed: 10  | Skipped: 0    |



### Test results by comparing the traces between geth and cuevm without stateRoot comparison

The tests results are collected by running the [Python script](https://gist.github.com/cassc/a161177bf850dbee4b7f3b2614250108) from the [ethereum](https://github.com/ethereum/tests) root folder:

``` bash
python ./run-ethtest-with-stateroot-comparison.py -t /tmp/out --runtest-bin runtest --geth geth --cuevm cuevm  --ignore-errors
```


| Test folder                          | Passed       | Failed      | Skipped       |
|--------------------------------------|--------------|-------------|---------------|
| Cancun                               | Passed: 0    | Failed: 0   | Skipped: 37   |
| Pyspecs                              | Passed: 201  | Failed: 2   | Skipped: 1233 |
| Python                               | Passed: 1    | Failed: 0   | Skipped: 4    |
| Shanghai                             | Passed: 1    | Failed: 8   | Skipped: 0    |
| stArgsZeroOneBalance                 | Passed: 72   | Failed: 16  | Skipped: 0    |
| stAttackTest                         | Passed: 0    | Failed: 2   | Skipped: 0    |
| stBadOpcode                          | Passed: 27   | Failed: 119 | Skipped: 1    |
| stBugs                               | Passed: 1    | Failed: 4   | Skipped: 0    |
| stCallCodes                          | Passed: 3    | Failed: 77  | Skipped: 1    |
| stCallCreateCallCodeTest             | Passed: 17   | Failed: 29  | Skipped: 0    |
| stCallDelegateCodesCallCodeHomestead | Passed: 0    | Failed: 58  | Skipped: 0    |
| stCallDelegateCodesHomestead         | Passed: 0    | Failed: 58  | Skipped: 0    |
| stChainId                            | Passed: 0    | Failed: 2   | Skipped: 0    |
| stCodeCopyTest                       | Passed: 0    | Failed: 2   | Skipped: 1    |
| stCodeSizeLimit                      | Passed: 4    | Failed: 3   | Skipped: 0    |
| stCreate2                            | Passed: 13   | Failed: 46  | Skipped: 3    |
| stDelegatecallTestHomestead          | Passed: 6    | Failed: 22  | Skipped: 0    |
| stEIP150singleCodeGasPrices          | Passed: 0    | Failed: 39  | Skipped: 1    |
| stEIP150Specific                     | Passed: 1    | Failed: 13  | Skipped: 0    |
| stEIP158Specific                     | Passed: 1    | Failed: 7   | Skipped: 3    |
| stEIP1559                            | Passed: 1751 | Failed: 8   | Skipped: 1    |
| stEIP2930                            | Passed: 12   | Failed: 6   | Skipped: 0    |
| stEIP3607                            | Passed: 12   | Failed: 0   | Skipped: 2    |
| stExample                            | Passed: 29   | Failed: 9   | Skipped: 0    |
| stExtCodeHash                        | Passed: 0    | Failed: 36  | Skipped: 3    |
| stHomesteadSpecific                  | Passed: 3    | Failed: 2   | Skipped: 0    |
| stInitCodeTest                       | Passed: 6    | Failed: 12  | Skipped: 0    |
| stLogTests                           | Passed: 11   | Failed: 35  | Skipped: 0    |
| stMemExpandingEIP150Calls            | Passed: 1    | Failed: 8   | Skipped: 0    |
| stMemoryStressTest                   | Passed: 78   | Failed: 2   | Skipped: 0    |
| stMemoryTest                         | Passed: 24   | Failed: 47  | Skipped: 0    |
| stNonZeroCallsTest                   | Passed: 6    | Failed: 18  | Skipped: 12   |
| stPreCompiledContracts               | Passed: 13   | Failed: 9   | Skipped: 1    |
| stPreCompiledContracts2              | Passed: 76   | Failed: 84  | Skipped: 0    |
| stRandom                             | Passed: 121  | Failed: 193 | Skipped: 0    |
| stRandom2                            | Passed: 89   | Failed: 137 | Skipped: 0    |
| stRecursiveCreate                    | Passed: 0    | Failed: 2   | Skipped: 0    |
| stRefundTest                         | Passed: 0    | Failed: 22  | Skipped: 1    |
| stReturnDataTest                     | Passed: 3    | Failed: 38  | Skipped: 0    |
| stRevertTest                         | Passed: 47   | Failed: 37  | Skipped: 11   |
| stSelfBalance                        | Passed: 0    | Failed: 6   | Skipped: 0    |
| stShift                              | Passed: 0    | Failed: 42  | Skipped: 0    |
| stSLoadTest                          | Passed: 0    | Failed: 1   | Skipped: 0    |
| stSolidityTest                       | Passed: 4    | Failed: 14  | Skipped: 0    |
| stSpecialTest                        | Passed: 5    | Failed: 10  | Skipped: 3    |
| stSStoreTest                         | Passed: 0    | Failed: 28  | Skipped: 1    |
| stStackTests                         | Passed: 185  | Failed: 3   | Skipped: 0    |
| stStaticFlagEnabled                  | Passed: 0    | Failed: 13  | Skipped: 0    |
| stSystemOperationsTest               | Passed: 26   | Failed: 43  | Skipped: 1    |
| stTransactionTest                    | Passed: 110  | Failed: 21  | Skipped: 2    |
| stTransitionTest                     | Passed: 0    | Failed: 6   | Skipped: 0    |
| stWalletTest                         | Passed: 4    | Failed: 40  | Skipped: 0    |
| stZeroCallsRevert                    | Passed: 12   | Failed: 4   | Skipped: 8    |
| stZeroCallsTest                      | Passed: 6    | Failed: 18  | Skipped: 12   |
| stZeroKnowledge                      | Passed: 410  | Failed: 33  | Skipped: 1    |
| stZeroKnowledge2                     | Passed: 485  | Failed: 10  | Skipped: 3    |
| VMTests                              | Passed: 232  | Failed: 58  | Skipped: 0    |
