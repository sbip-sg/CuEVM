# CuEVM
Cuda implementation of EVM bytecode executor

## Prerequisites
- CUDA Toolkit, Version 12.4 or above
- A CUDA-capable GPU (CUDA compute capabilily 7+ other older GPUs compability are not tested fully)
- A C++ compiler compatible with the CUDA Toolkit (gcc/g++ version 10+)


## Build

### Build standalone binary

This builds the standalone executable binary `cuevm_GPU` inside the `build` folder:

``` bash
# From the project root folder
rm -rf build
cmake -DBUILD_GO_LIBRARY=OFF  -DENABLE_EIP_3155=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCUDA_COMPUTE_CAPABILITY=86 cmake  -S . -B build
cmake --build build -j $(nproc)
```


### Build dynamic library

This builds the dynamic library `libcuevm_go.so` inside the `build` folder:

``` bash
# From the project root folder
rm -rf build
cmake -DBUILD_GO_LIBRARY=ON  -DENABLE_EIP_3155=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCUDA_COMPUTE_CAPABILITY=86 cmake  -S . -B build
cmake --build build -j $(nproc)
```


### Build in Docker

``` bash
# Inside the CuEVM project folder
docker run --rm -it -v ./:/workspace/cuevm -w /workspace/cuevm augustus/goevmlab-cuevm:20241216 /bin/bash
# You can compile in the docker container with the same commands as above
```

## Usage

### Using the standalone binary executor

The executor takes an input json file and after executing it, outputs in the standard output. The input format follows the [ethereum/tests](https://github.com/ethereum/tests/) format with minor difference that only one test case is supported by CuEVM at the moment.

``` bash
./build/cuevm_GPU --input fuzzing/eth-tests/erc20_mint.json
```

You should see the EVM execution traces in the output.

### Using the dynamic library

Please refer to [medusa](https://github.com/minhhn2910/medusa-backup) for usage.

# Test

## Test method

We use goevmlab to compare the execution traces from running the
[ethereum/tests](https://github.com/ethereum/tests/tree/shanghai) the go-ethereum
VM executor and CuEVM.

1. Install go-etheruem https://github.com/ethereum/go-ethereum  Tested with geth version 1.14.12.
2. Install goevmlab
   ```bash
   git clone --depth=1 -b add-cuevm https://github.com/cassc/goevmlab
   go install ./cmd/runtest/
   ```
2. Clone [ethereum/tests](https://github.com/ethereum/tests/tree/shanghai)
   ```bash
   git clone --depth 1 -b shanghai git@github.com:ethereum/tests.git ethereum-tests
   ```
3. Run all the tests in `GeneralStateTests` and compare the traces from go-ethereum and CuEVM:
   ``` bash
python3 scripts/run-ethtest-by-fork.py --ignore-errors --microtests --without-state-root \
  -i ethereum-tests/GeneralStateTests \
  -t ./tmp --runtest-bin runtest \
  --geth geth \
  --cuevm ./build/cuevm_GPU \
   ```


## Test results

We use the test files in
[ethereum/tests/GeneralStateTests](https://github.com/ethereum/tests/tree/develop/GeneralStateTests)
to test whether we can get the same results with the go-ethereum. To run the tests,

These tests are ignored, they contain some stress tests which could crash the EVM as well as the test script itself:

- stCreateTest
- stQuadraticComplexityTest
- stStaticCall
- stTimeConsuming


### Test results by comparing the traces between geth and cuevm without stateRoot comparison

The tests results are collected by running the [Python script](https://gist.github.com/cassc/b300005b38d7c01461b443ef67169659) from the [ethereum](https://github.com/ethereum/tests) root folder:

``` bash
python run-ethtest-without-stateroot-comparison.py --runtest-bin runtest --geth geth --cuevm /home/garfield/tmp/CuEVM-internal/build/cuevm_GPU --ignore-errors -t /tmp/ethtest/
```

> Note that there can be multiple tests in one input json, the number of tests shown below can be larger than number of input files.



| Test folder                          | Passed | Failed | Skipped | Timeout |
|--------------------------------------|--------|--------|---------|---------|
| stNonZeroCallsTest                   | 24     | 0      | 0       | 0       |
| stEIP3607                            | 7      | 5      | 0       | 0       |
| stEIP150singleCodeGasPrices          | 330    | 10     | 1       | 0       |
| stCallDelegateCodesCallCodeHomestead | 51     | 7      | 0       | 0       |
| stArgsZeroOneBalance                 | 96     | 0      | 0       | 0       |
| stStaticFlagEnabled                  | 25     | 0      | 0       | 9       |
| stShift                              | 40     | 1      | 0       | 1       |
| stEIP158Specific                     | 6      | 1      | 0       | 0       |
| stMemoryTest                         | 522    | 56     | 0       | 0       |
| stZeroKnowledge2                     | 519    | 0      | 0       | 0       |
| stEIP1559                            | 1643   | 200    | 0       | 2       |
| stReturnDataTest                     | 269    | 4      | 0       | 0       |
| stCodeCopyTest                       | 2      | 0      | 0       | 0       |
| stMemoryStressTest                   | 75     | 7      | 0       | 0       |
| stInitCodeTest                       | 21     | 1      | 0       | 0       |
| stMemExpandingEIP150Calls            | 10     | 0      | 0       | 0       |
| stWalletTest                         | 46     | 0      | 0       | 0       |
| stSpecialTest                        | 18     | 3      | 0       | 1       |
| stExtCodeHash                        | 59     | 6      | 0       | 0       |
| stTimeConsuming                      | 3807   | 1380   | 0       | 3       |
| stCreateTest                         | 153    | 47     | 0       | 3       |
| stRecursiveCreate                    | 1      | 0      | 0       | 1       |
| stCallDelegateCodesHomestead         | 51     | 7      | 0       | 0       |
| stZeroKnowledge                      | 745    | 55     | 0       | 0       |
| stTransitionTest                     | 6      | 0      | 0       | 0       |
| stCallCodes                          | 78     | 9      | 0       | 0       |
| stHomesteadSpecific                  | 5      | 0      | 0       | 0       |
| stCallCreateCallCodeTest             | 39     | 6      | 0       | 10      |
| stSolidityTest                       | 21     | 1      | 0       | 1       |
| stExample                            | 33     | 6      | 0       | 0       |
| stSStoreTest                         | 471    | 4      | 0       | 0       |
| stZeroCallsTest                      | 24     | 0      | 0       | 0       |
| stSelfBalance                        | 41     | 0      | 0       | 1       |
| stDelegatecallTestHomestead          | 20     | 3      | 0       | 8       |
| stQuadraticComplexityTest            | 14     | 1      | 0       | 17      |
| stEIP150Specific                     | 25     | 0      | 0       | 0       |
| stStackTests                         | 247    | 128    | 0       | 0       |
| stChainId                            | 2      | 0      | 0       | 0       |
| stAttackTest                         | 0      | 1      | 0       | 1       |
| stBugs                               | 9      | 0      | 0       | 0       |
| stBadOpcode                          | 4094   | 5      | 1       | 116     |
| stTransactionTest                    | 156    | 8      | 0       | 0       |
| stCreate2                            | 156    | 29     | 0       | 5       |
| stPreCompiledContracts2              | 233    | 15     | 0       | 0       |
| stRevertTest                         | 257    | 9      | 0       | 5       |
| stLogTests                           | 46     | 0      | 0       | 0       |
| stRandom                             | 297    | 11     | 0       | 6       |
| stRefundTest                         | 26     | 0      | 1       | 0       |
| stStaticCall                         | 421    | 9      | 0       | 48      |
| stRandom2                            | 212    | 9      | 0       | 5       |
| Shanghai                             | 12     | 15     | 0       | 0       |
| stCodeSizeLimit                      | 6      | 1      | 0       | 0       |
| stZeroCallsRevert                    | 16     | 0      | 0       | 0       |
| stPreCompiledContracts               | 897    | 31     | 0       | 32      |
| stSystemOperationsTest               | 76     | 1      | 0       | 6       |
| stEIP2930                            | 12     | 128    | 0       | 0       |
| VMTests                              | 625    | 3      | 0       | 0       |
| stSLoadTest                          | 1      | 0      | 0       | 0       |
