# CuEVM
CUDA implementation of an EVM bytecode executor

## Prerequisites
- CUDA Toolkit, Version 12.4 or above
- A CUDA-capable GPU (CUDA compute capability 8+; older GPUs' compatibility has not been fully tested)
- A C++ compiler compatible with the CUDA Toolkit (gcc/g++ version 10+)

## Build

### Build standalone binary

This builds the standalone executable binary `cuevm_GPU` inside the `build` folder:

```bash
# From the project root folder
rm -rf build
cmake -DBUILD_GO_LIBRARY=OFF -DENABLE_EIP_3155=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCUDA_COMPUTE_CAPABILITY=86 -S . -B build
cmake --build build -j $(nproc)
```

### Build dynamic library

This builds the dynamic library `libcuevm_go.so` inside the `build` folder:

```bash
# From the project root folder
rm -rf build
cmake -DBUILD_GO_LIBRARY=ON -DENABLE_EIP_3155=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCUDA_COMPUTE_CAPABILITY=86 -S . -B build
cmake --build build -j $(nproc)
```

### Build in Docker

```bash
# Inside the CuEVM project folder
docker run --rm -it -v ./:/workspace/cuevm -w /workspace/cuevm augustus/goevmlab-cuevm:20241216 /bin/bash
# You can compile in the docker container with the same commands as above
```

## Usage

### Using the standalone binary executor

The executor takes an input JSON file and outputs the result to standard output after execution. The input format follows the [ethereum/tests](https://github.com/ethereum/tests/) format, with the minor difference that CuEVM currently supports only one test transaction in each test json.

```bash
./build/cuevm_GPU --input fuzzing/eth-tests/erc20_mint.json
```

You should see the EVM execution traces in the output.

### Using the dynamic library

Please refer to [medusa](https://github.com/minhhn2910/medusa-backup) for example usage.

### Multi-GPU mode

If your system has multiple GPUs, CuEVM can automatically distribute the workload (N transactions) evenly across all available GPUs for improved performance.

By default, CuEVM detects and utilizes all available GPUs without additional configuration. If you want to limit or specify which GPUs to use, set the `CUDA_VISIBLE_DEVICES` environment variable.

For example, to use only GPU 0 and GPU 2 on a system with 4 GPUs:

  * `CUDA_VISIBLE_DEVICES=0,2 ./build/cuevm_GPU  --input fuzzing/eth-tests/erc20_mint.json `
  * `CUDA_VISIBLE_DEVICES=0,2 medusa fuzz --config medusa.json`
# Testing

## Testing Methodology

We use goevmlab to compare execution traces between the [ethereum/tests](https://github.com/ethereum/tests/tree/shanghai) run on the go-ethereum VM executor and CuEVM.

1. Install go-ethereum: https://github.com/ethereum/go-ethereum (Tested with geth version 1.14.12)
2. Install goevmlab:
   ```bash
   git clone --depth=1 -b add-cuevm https://github.com/cassc/goevmlab
   go install ./cmd/runtest/
   ```
3. Clone [ethereum/tests](https://github.com/ethereum/tests/tree/shanghai):
   ```bash
   git clone --depth 1 -b shanghai git@github.com:ethereum/tests.git ethereum-tests
   ```
4. Run all tests in `GeneralStateTests` and compare traces between go-ethereum and CuEVM. The test script `scripts/run-ethtest-by-fork.py` expands the input JSON file into multiple test cases, each containing a single transaction. The script then runs the test cases on both the go-ethereum and CuEVM executors, comparing the execution traces.
   ```bash
   python3 scripts/run-ethtest-by-fork.py --ignore-errors --microtests --without-state-root \
     -i ethereum-tests/GeneralStateTests \
     -t ./tmp --runtest-bin runtest \
     --geth geth \
     --cuevm ./build/cuevm_GPU
   ```

## Test Results

We use test files from [ethereum/tests/GeneralStateTests](https://github.com/ethereum/tests/tree/develop/GeneralStateTests) to verify consistency with go-ethereum results.

The following tests are ignored as they contain stress tests that may crash the EVM or the test script itself:
- stCreateTest
- stQuadraticComplexityTest
- stStaticCall
- stTimeConsuming

### Test Results (trace comparison between geth and cuevm without stateRoot comparison)

Test results were collected using a [Python script](https://gist.github.com/cassc/b300005b38d7c01461b443ef67169659) run from the [ethereum/tests](https://github.com/ethereum/tests) root folder:

```bash
python3 run-ethtest-without-stateroot-comparison.py --runtest-bin runtest --geth geth --cuevm ./build/cuevm_GPU --ignore-errors -t /tmp/ethtest/
```

> Note: A single input JSON file may contain multiple tests, so the number of tests shown below may exceed the number of input files.

| Test folder                          | Passed     | Failed  | Skipped/Timeout |
|--------------------------------------|------------|---------|-----------------|
| stNonZeroCallsTest                   | 24         | 0       | 0               |
| stEIP3607                            | 7          | 5       | 0               |
| stEIP150singleCodeGasPrices          | 330        | 10      | 1               |
| stCallDelegateCodesCallCodeHomestead | 51         | 7       | 0               |
| stArgsZeroOneBalance                 | 96         | 0       | 0               |
| stStaticFlagEnabled                  | 25         | 0       | 9               |
| stShift                              | 40         | 1       | 1               |
| stEIP158Specific                     | 6          | 1       | 0               |
| stMemoryTest                         | 522        | 56      | 0               |
| stZeroKnowledge2                     | 519        | 0       | 0               |
| stEIP1559                            | 1643       | 200     | 2               |
| stReturnDataTest                     | 269        | 4       | 0               |
| stCodeCopyTest                       | 2          | 0       | 0               |
| stMemoryStressTest                   | 75         | 7       | 0               |
| stInitCodeTest                       | 21         | 1       | 0               |
| stMemExpandingEIP150Calls            | 10         | 0       | 0               |
| stWalletTest                         | 46         | 0       | 0               |
| stSpecialTest                        | 18         | 3       | 1               |
| stExtCodeHash                        | 59         | 6       | 0               |
| stRecursiveCreate                    | 1          | 0       | 1               |
| stCallDelegateCodesHomestead         | 51         | 7       | 0               |
| stZeroKnowledge                      | 745        | 55      | 0               |
| stTransitionTest                     | 6          | 0       | 0               |
| stCallCodes                          | 78         | 9       | 0               |
| stHomesteadSpecific                  | 5          | 0       | 0               |
| stCallCreateCallCodeTest             | 39         | 6       | 10              |
| stSolidityTest                       | 21         | 1       | 1               |
| stExample                            | 33         | 6       | 0               |
| stSStoreTest                         | 471        | 4       | 0               |
| stZeroCallsTest                      | 24         | 0       | 0               |
| stSelfBalance                        | 41         | 0       | 1               |
| stDelegatecallTestHomestead          | 20         | 3       | 8               |
| stEIP150Specific                     | 25         | 0       | 0               |
| stStackTests                         | 247        | 128     | 0               |
| stChainId                            | 2          | 0       | 0               |
| stAttackTest                         | 0          | 1       | 1               |
| stBugs                               | 9          | 0       | 0               |
| stBadOpcode                          | 4094       | 5       | 117             |
| stTransactionTest                    | 156        | 8       | 0               |
| stCreate2                            | 156        | 29      | 5               |
| stPreCompiledContracts2              | 233        | 15      | 0               |
| stRevertTest                         | 257        | 9       | 5               |
| stLogTests                           | 46         | 0       | 0               |
| stRandom                             | 297        | 11      | 6               |
| stRefundTest                         | 26         | 0       | 1               |
| stRandom2                            | 212        | 9       | 5               |
| Shanghai                             | 12         | 15      | 0               |
| stCodeSizeLimit                      | 6          | 1       | 0               |
| stZeroCallsRevert                    | 16         | 0       | 0               |
| stPreCompiledContracts               | 897        | 31      | 32              |
| stSystemOperationsTest               | 76         | 1       | 6               |
| stEIP2930                            | 12         | 128     | 0               |
| VMTests                              | 625        | 3       | 0               |
| stSLoadTest                          | 1          | 0       | 0               |
| **Total**                            | **12,116** | **789** | **214**         |




## Contributors

<div>
  <span style="text-align: center; margin-right: 12px;">
    <a href="https://github.com/minhhn2910">
      <img src="https://github.com/minhhn2910.png" width="50px;" alt="minhhn2910" class="avatar circle" style="margin-right:6px;"/>
    </a>
    <span>Nhut-Minh Ho</span>
  </span>
  <span style="text-align: center; margin-right: 12px;">
    <a href="https://github.com/sdcioc">
      <img src="https://github.com/sdcioc.png" width="50px;" alt="sdcioc" class="avatar circle" style="margin-right:6px;"/>
    </a>
    <span>Stefan-Dan Ciocirlan</span>
  </span>
  <span style="text-align: center; margin-right: 12px;">
    <a href="https://github.com/cassc">
      <img src="https://github.com/cassc.png" width="50px;" alt="cassc" class="avatar circle" style="margin-right:6px;"/>
    </a>
    <span>Chen Li</span>
  </span>
</div>

This project is part of the [Singapore Blockchain Innovation Programme (SBIP)](https://sbip.sg/). We extend our gratitude to the programme and its team members for their expertise and dedication to the development of this project.


## Documentation

An auto generated source code documentation is available at [https://sbip-sg.github.io/CuEVM/files.html](https://sbip-sg.github.io/CuEVM/files.html)
