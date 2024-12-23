# CuEVM
Cuda implementation of EVM bytecode executor


## Prerequisites
- CUDA Toolkit (Version 12.0+, because we use `--std c++20`)
- A CUDA-capable GPU (CUDA compute capabilily 7+ other older GPUs compability are not tested fully)
- A C++ compiler compatible with the CUDA Toolkit (gcc/g++ version 10+)
- For docker image, you don't need the above but the system with docker installed

## Compile and Build binary
There are two methods, one requires installing all prequisited in the system, the second one use docker image:

### On your own system


Building on Ubuntu (with sudo):
* Setup required libraries: `sudo apt install libgmp-dev`
* Setup cJSON: `sudo apt install libcjson-dev`
* Use cmake to build the binary (Adjust `-DCUDA_COMPUTE_CAPABILITY=86` according to your GPU compute capability number):

``` bash
cmake -S . -B build -DTESTS=OFF -DGPU=ON -DCPU=OFF \
    -DCUDA_COMPUTE_CAPABILITY=86
    -DENABLE_EIP_3155_OPTIONAL=OFF \
    -DENABLE_EIP_3155=ON \
    -DENABLE_PAIRING_CODE=ON

cmake --build build
```

Building without sudo is also possible with extra configuration and modification on the Makefile or evironment variables, please refer to online tutorials on how to build and use libraries in custom directory

#### Building using docker image

* Pull the docker image first: `docker pull augustus/goevmlab-cuevm:20241008`
* Run and mount the current code folder `docker run -it -v $(pwd):/workspaces/CuEVM augustus/goevmlab-cuevm:20241008`
* Inside the docker container, you can build the code using the same commands as above (Adjust `-DCUDA_COMPUTE_CAPABILITY=86` according to your GPU compute capability number):
``` bash
cmake -S . -B build -DTESTS=OFF -DGPU=ON -DCPU=OFF \
    -DCUDA_COMPUTE_CAPABILITY=86 \
    -DENABLE_EIP_3155_OPTIONAL=OFF \
    -DENABLE_EIP_3155=ON \
    -DENABLE_PAIRING_CODE=ON

cmake --build build
```



## Usage

After building the binary, a binary named `cuevm_GPU` will be created in the `build`
folder. You can run the binary using the following command:

``` bash
./build/cuevmGPU --input [input_json_file]
```

For example:

``` bash
./build/cuevm_GPU --input samples/underflowTest.json
```

The execution trace and output state will be printed to the stdout, you can use
`jq` to keep only the Json output:

``` bash
./build/cuevm_GPU --input samples/underflowTest.json \
    | jq -R 'select(try fromjson? | . != null) | fromjson'
{
  "pc": 0,
  "op": 97,
  "gas": "0x79bf20",
  "gasCost": "0x3",
  "memSize": 0,
  "stack": [],
  "depth": 1,
  "refund": 0
}
...
```


## Demo of the GPU-accelerated fuzzer and CuEVM library mode

[Run Google Colab demo using free GPU](https://colab.research.google.com/drive/1W_3zKOJR2Jpv_6SoM0cmOFgVHP2b7rny?usp=sharing)

## Testing using ethtest

The script `scripts/run-ethtest-by-fork` can be used to run the tests from the
[ethereum/tests](https://github.com/ethereum/tests/tree/shanghai/GeneralStateTests). It
compares the traces from the outputs of CuEVM and `geth` without stateRoot.


Requirements:
- Shanghai branch of [ethereum/tests](https://github.com/ethereum/tests/tree/shanghai/GeneralStateTests)
-  [goevmlab with CuEVM driver](https://github.com/cassc/goevmlab/tree/add-cuevm)

The following will run all the tests in `ethereum/tests/GeneralStateTests`, note that this may take a few hours:

``` bash
git clone --depth=1 --branch shanghai https://github.com/ethereum/tests.git ethereum/tests
python scripts/run-ethtest-by-fork.py --without-state-root --ignore-errors --microtests \
  --input ethereum/tests/GeneralStateTests/ \
  -t /tmp/ \
  --runtest-bin runtest \
  --geth geth \
  --cuevm build/cuevm_GPU
```

To run a single json test case in `ethereum/tests`, create a new folder and copy the json file to the new folder, for example,

``` bash
mkdir my-test-folder
cp ethereum/tests/GeneralStateTests/Shanghai/stEIP3651-warmcoinbase/coinbaseWarmAccountCallGas.json my-test-folder/
python scripts/run-ethtest-by-fork.py --without-state-root --ignore-errors --microtests \
  --input my-test-folder/ \
  -t /tmp/ \
  --runtest-bin runtest \
  --geth geth \
  --cuevm build/cuevm_GPU
```

To inpsect the output from each EVM exeuctor, specify the path of the generated test file:
- for go-ethereum:

``` bash
geth --json --noreturndata --dump statetest {input_test_file}
# eg., geth --json --noreturndata --dump statetest /tmp/./coinbaseWarmAccountCallGas-0-0-0/coinbaseWarmAccountCallGas.json
```
- for CuEVM

``` text
./build/cuevm_GPU --input {input_test_file}
# eg., ./build/cuevm_GPU --input /tmp/./coinbaseWarmAccountCallGas-0-0-0/coinbaseWarmAccountCallGas.json
```
- to compare results between CuEVM and go-ethereum:

``` text
runtest --outdir=./ --geth=geth --cuevm=./build/cuevm_GPU /tmp/./coinbaseWarmAccountCallGas-0-0-0/coinbaseWarmAccountCallGas.json
```


## Test results

### Test results by comparing the traces between geth and cuevm without stateRoot comparison

We use the test files in Shanghai branch of
[ethereum/tests/GeneralStateTests](https://github.com/ethereum/tests/tree/shanghai/GeneralStateTests)
to test whether we can get the same traces with the go-ethereum. To run the tests,

- Build the `cuevm_GPU` binary
- Either download or build the geth binary, in this test the version [v1.13.14](https://github.com/ethereum/go-ethereum/releases/tag/v1.13.14) is used, later version will probably produce the same result
- Get the test json files from [ethereum/tests](https://github.com/ethereum/tests): `git clone --depth=1 --branch shanghai https://github.com/ethereum/tests.git ethereum/tests`
- Build the `runtest` binary from [cassc/goevmlab](https://github.com/cassc/goevmlab) which adds support for CuEVM
- Run `goevmlab/runtest` to compare the results between `geth` and `cuevm`
- The tests results are collected by running the [Python script](https://gist.github.com/cassc/b300005b38d7c01461b443ef67169659) from the [ethereum/tests](https://github.com/ethereum/tests/tree/shanghai) root folder:

``` bash
python ./run-ethtest-without-stateroot-comparison.py -t /tmp/out --runtest-bin runtest --geth geth --cuevm cuevm  --ignore-errors
```

> - Note that there can be multiple tests in one input json, the number of tests shown below can be larger than number of input json files.
> - Some tests are skipped if they do not support Shanghai VM.



| Test folder                          | Passed | Failed | Skipped | Timeout | Time taken (seconds) |
|--------------------------------------|--------|--------|---------|---------|----------------------|
| stNonZeroCallsTest                   | 24     | 0      | 0       | 0       | 11.77                |
| stEIP3607                            | 12     | 0      | 0       | 0       | 5.95                 |
| stEIP150singleCodeGasPrices          | 340    | 0      | 1       | 0       | 607.93               |
| stCallDelegateCodesCallCodeHomestead | 57     | 1      | 0       | 0       | 411.56               |
| stArgsZeroOneBalance                 | 96     | 0      | 0       | 0       | 36.50                |
| stStaticFlagEnabled                  | 25     | 9      | 0       | 0       | 59.68                |
| stShift                              | 40     | 0      | 0       | 2       | 193.98               |
| stEIP158Specific                     | 7      | 0      | 0       | 0       | 3.65                 |
| stMemoryTest                         | 569    | 0      | 0       | 9       | 1333.89              |
| stZeroKnowledge2                     | 519    | 0      | 0       | 0       | 170.10               |
| stEIP1559                            | 1841   | 4      | 0       | 0       | 594.56               |
| stReturnDataTest                     | 260    | 13     | 0       | 0       | 292.33               |
| stCodeCopyTest                       | 2      | 0      | 0       | 0       | 1.11                 |
| stMemoryStressTest                   | 80     | 2      | 0       | 0       | 24.99                |
| stInitCodeTest                       | 22     | 0      | 0       | 0       | 10.42                |
| stMemExpandingEIP150Calls            | 9      | 1      | 0       | 0       | 5.38                 |
| stWalletTest                         | 20     | 26     | 0       | 0       | 95.15                |
| stSpecialTest                        | 18     | 1      | 0       | 3       | 344.69               |
| stExtCodeHash                        | 63     | 2      | 0       | 0       | 37.55                |
| stTimeConsuming                      | 5187   | 0      | 0       | 3       | 4884.31              |
| stCreateTest                         | 168    | 31     | 0       | 4       | 531.46               |
| stRecursiveCreate                    | 1      | 1      | 0       | 0       | 7.79                 |
| stCallDelegateCodesHomestead         | 58     | 0      | 0       | 0       | 437.24               |
| stZeroKnowledge                      | 751    | 49     | 0       | 0       | 301.65               |
| stTransitionTest                     | 6      | 0      | 0       | 0       | 3.17                 |
| stCallCodes                          | 87     | 0      | 0       | 0       | 538.96               |
| stHomesteadSpecific                  | 5      | 0      | 0       | 0       | 2.32                 |
| stCallCreateCallCodeTest             | 41     | 2      | 0       | 12      | 1159.56              |
| stSolidityTest                       | 14     | 7      | 0       | 2       | 199.26               |
| stExample                            | 38     | 1      | 0       | 0       | 13.64                |
| stSStoreTest                         | 468    | 7      | 0       | 0       | 247.11               |
| stZeroCallsTest                      | 24     | 0      | 0       | 0       | 11.06                |
| stSelfBalance                        | 35     | 7      | 0       | 0       | 51.03                |
| stDelegatecallTestHomestead          | 23     | 0      | 0       | 8       | 783.35               |
| stQuadraticComplexityTest            | 12     | 2      | 0       | 18      | 1639.77              |
| stEIP150Specific                     | 25     | 0      | 0       | 0       | 10.64                |
| stStackTests                         | 248    | 117    | 0       | 10      | 10326.64             |
| stChainId                            | 2      | 0      | 0       | 0       | 0.53                 |
| stAttackTest                         | 1      | 0      | 0       | 1       | 177.33               |
| stBugs                               | 9      | 0      | 0       | 0       | 3.86                 |
| stBadOpcode                          | 3404   | 808    | 1       | 3       | 5784.11              |
| stTransactionTest                    | 164    | 0      | 0       | 0       | 87.66                |
| stCreate2                            | 182    | 3      | 0       | 5       | 550.60               |
| stPreCompiledContracts2              | 247    | 0      | 0       | 1       | 219.27               |
| stRevertTest                         | 257    | 4      | 0       | 10      | 1031.87              |
| stLogTests                           | 46     | 0      | 0       | 0       | 19.47                |
| stRandom                             | 302    | 12     | 0       | 0       | 124.95               |
| stRefundTest                         | 26     | 0      | 1       | 0       | 13.80                |
| stStaticCall                         | 390    | 7      | 0       | 81      | 9530.82              |
| stRandom2                            | 214    | 11     | 0       | 1       | 194.87               |
| Shanghai                             | 25     | 1      | 0       | 1       | 175.50               |
| stCodeSizeLimit                      | 7      | 0      | 0       | 0       | 3.28                 |
| stZeroCallsRevert                    | 16     | 0      | 0       | 0       | 6.15                 |
| stPreCompiledContracts               | 935    | 25     | 0       | 0       | 627.29               |
| stSystemOperationsTest               | 73     | 1      | 0       | 9       | 872.60               |
| stEIP2930                            | 140    | 0      | 0       | 0       | 115.53               |
| VMTests                              | 570    | 55     | 3       | 3       | 584.94               |
| stSLoadTest                          | 1      | 0      | 0       | 0       | 0.54                 |


## Citation

If you use our code in your research, please kindly cite:

```
To be updated.
```


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
