# CuEVM GPU Fuzzer

Demo interfacing with the GPU-accelerated smart contract fuzzing library for finding vulnerabilities in Ethereum contracts. 

## Quick Start

```bash
# Basic usage
python fuzzer.py --input contracts/overflow.sol --contract_name TestOverflow

# With custom parameters  
python fuzzer.py --input contracts/overflow.sol --contract_name TestOverflow \
                 --num_instances 32 --num_iterations 5 --sequence_length 2
```

## Features

- **GPU Acceleration**: Uses `libcuevm_go.so` for high-performance execution
- **Bug Detection**: Finds overflows, underflows, and other vulnerabilities  
- **Source Mapping**: Shows exact line numbers and code where bugs occur
- **Sequence Testing**: Tests complex transaction sequences

## Sample Output

```bash
Bug found: PC=200, Type=1, Contract=144
Bug location: PC=200, Lines=(8, 8), Source: a * factor
Found 1 new bugs!

Bug 200_1_144: Integer overflow at PC 200
Function: multiply  
Location: Line 8
Source: a * factor
```

## Options

```
--input SOURCE              Solidity source file
--contract_name NAME         Contract name to test
--num_instances N           Parallel instances (default: 32)
--num_iterations N          Fuzzing iterations (default: 100)  
--sequence_length N         Transaction sequence length (default: 1)
--config PATH               Configuration file
```

## Test Contracts

- `contracts/overflow.sol` - Integer overflow bugs
- `contracts/erc20.sol` - ERC20 token with underflow
- `contracts/state_change.sol` - State persistence testing

## Setup

```bash
pip install -r requirements.txt
# Ensure libcuevm_go.so is built in ../build/
```

That's it! The fuzzer will automatically detect bugs and map them to your source code.