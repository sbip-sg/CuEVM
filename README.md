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

`python3 scripts/run-ethtest-by-fork.py -i ./tests/GeneralStateTests -t ./tmp --runtest-bin runtest --geth gethvm --cuevm ./build/cuevm --ignore-errors --microtests`

Single test:
`python3 scripts/run-ethtest-by-fork.py -i ./tmp/{input_test_file_dir} -t ./dtmp --runtest-bin runtest --geth gethvm --cuevm ./build/cuevm --ignore-errors --microtests`

To see the results of gethvm
`gethvm --json --noreturndata --dump statetest {input_test_file} &> geth.out`
To see the result of cuevm
`clear && valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./build/cuevm  --input {input_test_file} &> cuevm.out`

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
[ethereum/tests/GeneralStateTests](https://github.com/ethereum/tests/tree/develop/GeneralStateTests)
to test whether we can get the same results with the go-ethereum. To run the tests,

- build the binary with the debug flag on: `make cpu_debug_interpreter`
- either download or build the geth binary, in this test the version [v1.13.14](https://github.com/ethereum/go-ethereum/releases/tag/v1.13.14) is used, later version will probably produce the same result
- get the test json files from [ethereum/tests](https://github.com/ethereum/tests)
- extract and keep the tests which targeted at the Shanghai fork
- build the `runtest` binary from [cassc/goevmlab](https://github.com/cassc/goevmlab) which adds support for CuEVM
- run `goevmlab/runtest` to compare the results between `geth` and `cuevm`

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

| Test folder                          | Passed | Failed | Skipped |
|--------------------------------------|--------|--------|---------|
| Cancun                               | 0      | 0      | 37      |
| Pyspecs                              | 201    | 2      | 1233    |
| Python                               | 1      | 0      | 4       |
| Shanghai                             | 20     | 3      | 0       |
| stArgsZeroOneBalance                 | 95     | 1      | 0       |
| stAttackTest                         | 0      | 2      | 0       |
| stBadOpcode                          | 3542   | 118    | 1       |
| stBugs                               | 9      | 0      | 0       |
| stCallCodes                          | 75     | 10     | 1       |
| stCallCreateCallCodeTest             | 39     | 9      | 0       |
| stCallDelegateCodesCallCodeHomestead | 51     | 7      | 0       |
| stCallDelegateCodesHomestead         | 51     | 7      | 0       |
| stChainId                            | 2      | 0      | 0       |
| stCodeCopyTest                       | 1      | 1      | 1       |
| stCodeSizeLimit                      | 7      | 0      | 0       |
| stCreate2                            | 124    | 16     | 3       |
| stDelegatecallTestHomestead          | 23     | 6      | 0       |
| stEIP150singleCodeGasPrices          | 115    | 5      | 1       |
| stEIP150Specific                     | 24     | 1      | 0       |
| stEIP158Specific                     | 6      | 2      | 3       |
| stEIP1559                            | 1817   | 2      | 1       |
| stEIP2930                            | 15     | 5      | 0       |
| stEIP3607                            | 12     | 0      | 2       |
| stExample                            | 34     | 4      | 0       |
| stExtCodeHash                        | 23     | 21     | 3       |
| stHomesteadSpecific                  | 5      | 0      | 0       |
| stInitCodeTest                       | 20     | 2      | 0       |
| stLogTests                           | 46     | 0      | 0       |
| stMemExpandingEIP150Calls            | 9      | 1      | 0       |
| stMemoryStressTest                   | 82     | 0      | 0       |
| stMemoryTest                         | 386    | 7      | 0       |
| stNonZeroCallsTest                   | 23     | 1      | 12      |
| stPreCompiledContracts               | 564    | 1      | 1       |
| stPreCompiledContracts2              | 189    | 17     | 0       |
| stRandom                             | 305    | 9      | 0       |
| stRandom2                            | 221    | 5      | 0       |
| stRecursiveCreate                    | 2      | 0      | 0       |
| stRefundTest                         | 10     | 12     | 1       |
| stReturnDataTest                     | 57     | 10     | 0       |
| stRevertTest                         | 185    | 10     | 11      |
| stSelfBalance                        | 26     | 2      | 0       |
| stShift                              | 42     | 0      | 0       |
| stSLoadTest                          | 1      | 0      | 0       |
| stSolidityTest                       | 23     | 0      | 0       |
| stSpecialTest                        | 20     | 2      | 3       |
| stSStoreTest                         | 205    | 17     | 1       |
| stStackTests                         | 185    | 3      | 0       |
| stStaticFlagEnabled                  | 1      | 12     | 0       |
| stSystemOperationsTest               | 71     | 8      | 1       |
| stTransactionTest                    | 159    | 7      | 2       |
| stTransitionTest                     | 6      | 0      | 0       |
| stWalletTest                         | 20     | 26     | 0       |
| stZeroCallsRevert                    | 16     | 0      | 8       |
| stZeroCallsTest                      | 24     | 0      | 12      |
| stZeroKnowledge                      | 681    | 29     | 1       |
| stZeroKnowledge2                     | 485    | 10     | 3       |
| VMTests                              | 567    | 10     | 0       |



### Test results by comparing the traces between geth and cuevm with stateRoot comparison

The tests results are collected by running the [Python script](https://gist.github.com/cassc/a161177bf850dbee4b7f3b2614250108) from the [ethereum](https://github.com/ethereum/tests) root folder:

``` bash
python ./run-ethtest-with-stateroot-comparison.py -t /tmp/out --runtest-bin runtest --geth geth --cuevm cuevm  --ignore-errors
```


| Test folder                          | Passed | Failed | Skipped |
|--------------------------------------|--------|--------|---------|
| Cancun                               | 0      | 0      | 37      |
| Pyspecs                              | 201    | 2      | 1233    |
| Python                               | 1      | 0      | 4       |
| Shanghai                             | 1      | 8      | 0       |
| stArgsZeroOneBalance                 | 72     | 16     | 0       |
| stAttackTest                         | 0      | 2      | 0       |
| stBadOpcode                          | 27     | 119    | 1       |
| stBugs                               | 3      | 6      | 0       |
| stCallCodes                          | 76     | 10     | 1       |
| stCallCreateCallCodeTest             | 41     | 14     | 0       |
| stCallDelegateCodesCallCodeHomestead | 51     | 7      | 0       |
| stCallDelegateCodesHomestead         | 51     | 7      | 0       |
| stChainId                            | 2      | 0      | 0       |
| stCodeCopyTest                       | 1      | 1      | 1       |
| stCodeSizeLimit                      | 7      | 0      | 0       |
| stCreate2                            | 102    | 88     | 3       |
| stDelegatecallTestHomestead          | 22     | 9      | 0       |
| stEIP150singleCodeGasPrices          | 337    | 3      | 1       |
| stEIP150Specific                     | 24     | 1      | 0       |
| stEIP158Specific                     | 7      | 1      | 3       |
| stEIP1559                            | 1843   | 2      | 1       |
| stEIP2930                            | 15     | 125    | 0       |
| stEIP3607                            | 12     | 0      | 2       |
| stExample                            | 29     | 9      | 0       |
| stExtCodeHash                        | 39     | 30     | 3       |
| stHomesteadSpecific                  | 5      | 0      | 0       |
| stInitCodeTest                       | 16     | 6      | 0       |
| stLogTests                           | 46     | 0      | 0       |
| stMemExpandingEIP150Calls            | 8      | 2      | 0       |
| stMemoryStressTest                   | 82     | 0      | 0       |
| stMemoryTest                         | 564    | 14     | 0       |
| stNonZeroCallsTest                   | 24     | 0      | 12      |
| stPreCompiledContracts               | 929    | 31     | 1       |
| stPreCompiledContracts2              | 223    | 25     | 0       |
| stRandom                             | 303    | 7      | 0       |
| stRandom2                            | 215    | 6      | 0       |
| stRecursiveCreate                    | 0      | 2      | 0       |
| stRefundTest                         | 26     | 0      | 1       |
| stReturnDataTest                     | 230    | 43     | 0       |
| stRevertTest                         | 231    | 41     | 11      |
| stSelfBalance                        | 41     | 1      | 0       |
| stShift                              | 40     | 2      | 0       |
| stSLoadTest                          | 1      | 0      | 0       |
| stSolidityTest                       | 21     | 2      | 0       |
| stSpecialTest                        | 19     | 3      | 3       |
| stSStoreTest                         | 189    | 286    | 1       |
| stStackTests                         | 371    | 4      | 0       |
| stStaticFlagEnabled                  | 34     | 0      | 0       |
| stSystemOperationsTest               | 70     | 13     | 1       |
| stTransactionTest                    | 253    | 7      | 2       |
| stTransitionTest                     | 6      | 0      | 0       |
| stWalletTest                         | 46     | 0      | 0       |
| stZeroCallsRevert                    | 16     | 0      | 8       |
| stZeroCallsTest                      | 24     | 0      | 12      |
| stZeroKnowledge                      | 800    | 0      | 1       |
| stZeroKnowledge2                     | 519    | 0      | 3       |
| VMTests                              | 641    | 10     | 0       |
| VMTests/vmArithmeticTest/            | 219    | 0      | 0       |
| VMTests/vmIOandFlowOperations/       | 170    | 0      | 0       |
| VMTests/vmBitwiseLogicOperation/     | 57     | 0      | 0       |
| VMTests/vmLogTest/                   | 46     | 0      | 0       |
| VMTests/vmTests/                     | 136    | 0      | 0       |
| VMTests/vmPerformance/               | 13     | 10     | 0       |
