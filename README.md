# CuEVM
CUDA implementation of an EVM bytecode executor for fuzzing and beyond.

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

### Docker image

To produce the prebuilt image used in our releases:

```bash
docker build -t cuevm:latest .
```

The Dockerfile currently compiles both the shared library and the standalone binary with `-DCUDA_COMPUTE_CAPABILITY="86;89;90"`. Adjust these flags in the Dockerfile before building if you need support for other GPU architectures. 

Run the trace-comparison test suite directly in the container (mount a host directory for the temporary artifacts):

```bash
docker run --rm --gpus all \
  -v /tmp/ethtest:/tmp/ethtest \
  -v .:/app \
  cuevm:latest \
  run-ethtest-without-stateroot-comparison.py \
    --input /ethereum-tests-shanghai/ \
    --temporary-path /tmp/ethtest \
    --runtest-bin runtest \
    --geth go-evm \
    --cuevm cuevm \
    --ignore-errors
```

Run the sample Medusa fuzz campaign from the image:

```bash
docker run --rm --gpus all cuevm:latest \
  medusa fuzz --config /opt/cuevm/medusa_sample_config/medusa.json
```

## Usage

### Using the standalone binary executor

The executor takes an input JSON file and outputs the result to standard output after execution. The input format follows the [ethereum/tests](https://github.com/ethereum/tests/) format, with the minor difference that CuEVM currently supports only one test transaction in each test json.

```bash
./build/cuevm_GPU --input fuzzing/eth-tests/erc20_mint.json
```

### Using the dynamic library

We developed a python library in `fuzzing/` to showcase interfacing with `libcuevm_go.so` for bug detection in solidity smart contracts.

For a performant and ready-to-use fuzzer, please refer to [medusa-cuevm](https://github.com/minhhn2910/medusa-cuevm) for the official fuzzing tool built from [Medusa v1.2.1](https://github.com/crytic/medusa) utilizing cuevm library.
### Multi-GPU mode

If your system has multiple GPUs, CuEVM can automatically distribute the workload (N transactions) evenly across all available GPUs for improved performance.

By default, CuEVM detects and utilizes all available GPUs without additional configuration. If you want to limit or specify which GPUs to use, set the `CUDA_VISIBLE_DEVICES` environment variable.

For example, to use only GPU 0 and GPU 2 on a system with 4 GPUs:

  * `CUDA_VISIBLE_DEVICES=0,2 ./build/cuevm_GPU  --input fuzzing/eth-tests/erc20_mint.json `
  * `CUDA_VISIBLE_DEVICES=0,2 medusa fuzz --config medusa.json`

## Correctness Testing

### Testing Methodology

We use goevmlab to compare execution traces between the [ethereum/tests](https://github.com/ethereum/tests/tree/shanghai) run on the go-ethereum VM executor and CuEVM.

1. Install go-ethereum: https://github.com/ethereum/go-ethereum (Tested with geth version 1.14.13)
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

### Run trace comparison between geth and cuevm

We use test files from [ethereum/tests/GeneralStateTests](https://github.com/ethereum/tests/tree/develop/GeneralStateTests) to verify consistency with go-ethereum results. Test results were collected using a [Python script](https://gist.github.com/cassc/b300005b38d7c01461b443ef67169659) run from the [ethereum/tests](https://github.com/ethereum/tests) root folder:

```bash
python3 run-ethtest-without-stateroot-comparison.py --runtest-bin runtest --geth geth --cuevm ./build/cuevm_GPU --ignore-errors -t /tmp/ethtest/
```

> <small>
>  Note: A single input JSON file may contain multiple tests, so the number of tests shown below may exceed the number of input files. The tests passes means all lines are matched line-by-line in every intruction, except the final stateroot value (we skipped for simplicity). The following tests are ignored as they contain stress tests that largely result in timeout in printing log or crash the test script itself: [stQuadraticComplexityTest, stTimeConsuming, vmPerformance]
> </small>
### Test Results Summary

| Test Folder | Total Tests | Passed (%) |
| --- | --- | --- |
| **TOTAL** | **14380** | <div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: #8ac48d; height: 100%; width: 96.19610570236439%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">96.2%</div></div> |
| VMTests | 628 | <div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: #8ac48d; height: 100%; width: 100.0%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">100.0%</div></div> |


<details>
<summary><strong>📊 Click to view detailed results for all test folders</strong></summary>

| Test Folder | Total Tests | Passed (%) |
| --- | --- | --- |
| VMTests | 628 | <div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: #8ac48d; height: 100%; width: 100.0%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">100.0%</div></div> |
| stZeroKnowledge2 | 519 | <div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: #8ac48d; height: 100%; width: 100.0%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">100.0%</div></div> |
| stStackTests | 375 | <div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: #8ac48d; height: 100%; width: 100.0%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">100.0%</div></div> |
| stEIP150singleCodeGasPrices | 340 | <div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: #8ac48d; height: 100%; width: 100.0%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">100.0%</div></div> |
| stReturnDataTest | 273 | <div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: #8ac48d; height: 100%; width: 100.0%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">100.0%</div></div> |
| stArgsZeroOneBalance | 96 | <div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: #8ac48d; height: 100%; width: 100.0%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">100.0%</div></div> |
| stLogTests | 46 | <div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: #8ac48d; height: 100%; width: 100.0%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">100.0%</div></div> |
| stWalletTest | 46 | <div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: #8ac48d; height: 100%; width: 100.0%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">100.0%</div></div> |
| Shanghai | 27 | <div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: #8ac48d; height: 100%; width: 100.0%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">100.0%</div></div> |
| stRefundTest | 26 | <div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: #8ac48d; height: 100%; width: 100.0%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">100.0%</div></div> |
| stEIP150Specific | 25 | <div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: #8ac48d; height: 100%; width: 100.0%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">100.0%</div></div> |
| stNonZeroCallsTest | 24 | <div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: #8ac48d; height: 100%; width: 100.0%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">100.0%</div></div> |
| stZeroCallsTest | 24 | <div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: #8ac48d; height: 100%; width: 100.0%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">100.0%</div></div> |
| stInitCodeTest | 22 | <div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: #8ac48d; height: 100%; width: 100.0%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">100.0%</div></div> |
| stZeroCallsRevert | 16 | <div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: #8ac48d; height: 100%; width: 100.0%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">100.0%</div></div> |
| stEIP3607 | 12 | <div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: #8ac48d; height: 100%; width: 100.0%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">100.0%</div></div> |
| stMemExpandingEIP150Calls | 10 | <div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: #8ac48d; height: 100%; width: 100.0%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">100.0%</div></div> |
| stBugs | 9 | <div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: #8ac48d; height: 100%; width: 100.0%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">100.0%</div></div> |
| stCodeSizeLimit | 7 | <div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: #8ac48d; height: 100%; width: 100.0%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">100.0%</div></div> |
| stEIP158Specific | 7 | <div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: #8ac48d; height: 100%; width: 100.0%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">100.0%</div></div> |
| stTransitionTest | 6 | <div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: #8ac48d; height: 100%; width: 100.0%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">100.0%</div></div> |
| stHomesteadSpecific | 5 | <div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: #8ac48d; height: 100%; width: 100.0%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">100.0%</div></div> |
| stCodeCopyTest | 2 | <div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: #8ac48d; height: 100%; width: 100.0%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">100.0%</div></div> |
| stChainId | 2 | <div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: #8ac48d; height: 100%; width: 100.0%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">100.0%</div></div> |
| stSLoadTest | 1 | <div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: #8ac48d; height: 100%; width: 100.0%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">100.0%</div></div> |
| stEIP1559 | 1845 | <div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: #8ac48d; height: 100%; width: 99.56639566395664%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">99.6%</div></div> |
| stSStoreTest | 475 | <div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: #8ac48d; height: 100%; width: 99.1578947368421%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">99.2%</div></div> |
| stSelfBalance | 42 | <div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: #8ac48d; height: 100%; width: 97.61904761904762%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">97.6%</div></div> |
| stRandom | 314 | <div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: #8ac48d; height: 100%; width: 97.13375796178345%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">97.1%</div></div> |
| stBadOpcode | 4215 | <div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: #8ac48d; height: 100%; width: 97.12930011862396%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">97.1%</div></div> |
| stRevertTest | 271 | <div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: #8ac48d; height: 100%; width: 97.04797047970479%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">97.0%</div></div> |
| stTransactionTest | 164 | <div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: #8ac48d; height: 100%; width: 96.95121951219512%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">97.0%</div></div> |
| stRandom2 | 226 | <div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: #8ac48d; height: 100%; width: 96.46017699115043%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">96.5%</div></div> |
| stSolidityTest | 23 | <div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: #8ac48d; height: 100%; width: 95.65217391304348%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">95.7%</div></div> |
| stExtCodeHash | 65 | <div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: #8ac48d; height: 100%; width: 95.38461538461539%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">95.4%</div></div> |
| stShift | 42 | <div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: #8ac48d; height: 100%; width: 95.23809523809523%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">95.2%</div></div> |
| stExample | 39 | <div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: #8ac48d; height: 100%; width: 94.87179487179486%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">94.9%</div></div> |
| stPreCompiledContracts2 | 248 | <div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: #8ac48d; height: 100%; width: 94.75806451612904%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">94.8%</div></div> |
| stPreCompiledContracts | 960 | <div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: #8ac48d; height: 100%; width: 93.4375%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">93.4%</div></div> |
| stZeroKnowledge | 800 | <div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: #8ac48d; height: 100%; width: 93.125%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">93.1%</div></div> |
| stMemoryTest | 578 | <div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: #8ac48d; height: 100%; width: 91.86851211072664%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">91.9%</div></div> |
| stCreateTest | 203 | <div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: #8ac48d; height: 100%; width: 91.62561576354679%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">91.6%</div></div> |
| stSystemOperationsTest | 83 | <div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: #8ac48d; height: 100%; width: 91.56626506024097%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">91.6%</div></div> |
| stMemoryStressTest | 82 | <div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: #8ac48d; height: 100%; width: 91.46341463414635%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">91.5%</div></div> |
| stCallCodes | 87 | <div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: #b8d88e; height: 100%; width: 89.65517241379311%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">89.7%</div></div> |
| stStaticCall | 478 | <div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: #b8d88e; height: 100%; width: 88.91213389121339%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">88.9%</div></div> |
| stCallDelegateCodesCallCodeHomestead | 58 | <div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: #b8d88e; height: 100%; width: 87.93103448275862%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">87.9%</div></div> |
| stCallDelegateCodesHomestead | 58 | <div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: #b8d88e; height: 100%; width: 87.93103448275862%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">87.9%</div></div> |
| stSpecialTest | 22 | <div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: #b8d88e; height: 100%; width: 86.36363636363636%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">86.4%</div></div> |
| stEIP2930 | 140 | <div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: #b8d88e; height: 100%; width: 83.57142857142857%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">83.6%</div></div> |
| stCreate2 | 190 | <div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: #b8d88e; height: 100%; width: 82.10526315789474%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">82.1%</div></div> |
| stCallCreateCallCodeTest | 55 | <div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: #b8d88e; height: 100%; width: 80.0%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">80.0%</div></div> |
| stDelegatecallTestHomestead | 31 | <div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: #f4d06f; height: 100%; width: 74.19354838709677%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">74.2%</div></div> |
| stStaticFlagEnabled | 34 | <div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: #f4d06f; height: 100%; width: 73.52941176470588%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">73.5%</div></div> |
| stRecursiveCreate | 2 | <div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: #f4d06f; height: 100%; width: 50.0%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">50.0%</div></div> |
| stAttackTest | 2 | <div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: #f4d06f; height: 100%; width: 50.0%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">50.0%</div></div> |

</details>


**Summary Statistics:**
- Total Tests Run: 14,380
- Passed: 13,833 (96.2%)
- Failed: 295 ; Timeout: 252 ; Skipped: 37

## Contributors

- [**Nhut-Minh Ho**](https://github.com/minhhn2910) — *National University of Singapore*
- [**Stefan-Dan Ciocirlan**](https://github.com/sdcioc) — *University Politehnica of Bucharest*
- [**Chen Li**](https://github.com/cassc) — *National University of Singapore*

We also acknowledge leadership and contribution from:

- [**Prof. Ooi Beng Chin**](https://github.com/ooibc) — *National University of Singapore and Zhejiang University*
- [**Prof. Xiao Xiaokui**](https://github.com/xkxiao) — *National University of Singapore*
- [**Prof. Anh Dinh**](https://github.com/ug93tad) — *Deakin University*
- [**Ta Quang Trung**](https://github.com/taquangtrung) — *National University of Singapore*
- [**Fredrik Svantes**](https://github.com/fredrik0x) — *Ethereum Foundation*

This project is funded by the Ethereum Foundation.
## Documentation

An auto generated source code documentation is available at [https://sbip-sg.github.io/CuEVM/files.html](https://sbip-sg.github.io/CuEVM/files.html)
