## Simple demonstration on how to Run CuEVM to detect bug based on traces

To run the demo in this folder, first you need to install the required packages:
* `pip install -r requirements.txt`


### Simple testing of the 3rd milestone: state persistents across different txs

We allow the test of a sequence of transactions on top of the same initial state using the format in `configurations/state_change.json`.
The sequence of transaction is a list of call to function with values and input arguments (currently only one sender is supported).

* `python test_evm.py --config configurations/state_change.json --source contracts/state_change.sol --output-path /tmp/test_evm --evm-executable ../out/interpreter --detect-bug`

### Simple testing of the 4th milestone: multiple contracts in the state and cross-contract call supported.

* `python test_evm.py --config configurations/cross_contract.json --source contracts/cross_contract.sol --output-path /tmp/test_evm --evm-executable ../out/interpreter --detect-bug`

Sample buggy code:
```solidity
function underflow(address tokenA, address tokenB) public payable{
        // just simply mint and check the total supply demontraing 4 external calls
        ERC20Token tokenAContract = ERC20Token(tokenA);
        ERC20Token tokenBContract = ERC20Token(tokenB);
        // mint and check total supply :
        tokenAContract.mint{value: msg.value/3}();
        // minted tokens = call value = msg.value/3
        tokenBContract.mint{value: msg.value/2}();
        // minted tokens = call value = msg.value/2

        uint total_balance_a = tokenAContract.totalSupply();
        uint total_balance_b = tokenBContract.totalSupply();

        // check the diff in balance :
        unchecked {
            uint diff = total_balance_a - total_balance_b; // overflow here
            // For example, in the file configurations/cross_contract, we send msg.value = 300 wei.
            // So the token A supply = 100 and token B supply = 150, leading to underflow
        }
    }
```

Expected output:
```
...
--------------------------------------------------------------------------------
found operation 100 - 150 = 115792089237316195423570985008687907853269984665640564039457584007913129639886
underflow detected at program counter 526
Line: 46 : Source total_balance_a - total_balance_b
--------------------------------------------------------------------------------

```

### Usage

```
usage: test_evm.py [-h] [--source SOURCE] [--config CONFIG] [--evm-executable EVM_EXECUTABLE] [--output-path OUTPUT_PATH] [--detect-bug]

Run EVM test cases

optional arguments:
  -h, --help            show this help message and exit
  --source SOURCE       source file (solidity source file)
  --config CONFIG       config file: file contains the sequence of transactions to run (for reproducing bugs, can generate this file with a fuzzer)
  --evm-executable EVM_EXECUTABLE
                        path to the compiled evm executable (it must be compiled with "tracing-enabled" for bug detection example to work)
  --output-path OUTPUT_PATH
                        output path
  --detect-bug          enable example for bug detection

```

### Detect overflow bug in contracts/overflow.sol

Very simple code for causing overflow bug over two transactions.

* `python test_evm.py --config configurations/overflow.json --source contracts/overflow.sol --output-path /tmp/test_evm --evm-executable ../out/interpreter --detect-bug`

Sample output:

```
...
--------------------------------------------------------------------------------
found operation 32 + 4 = 36
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
found operation 57896044618658097711785492504343953926634992332820282019728792003956564819969 * 4 = 4
overflow detected at program counter 200
Line: 8 : Source a * factor

```

### Detect underflow bug in contracts/erc20.sol

Besides bug detection, this test also can show the EVM works on ERC20 sample code, including reading message.value and minting token.

* `python test_evm.py --config configurations/erc20.json --source contracts/erc20.sol --output-path /tmp/test_evm --evm-executable ../out/interpreter --detect-bug`

Sample output:

```
...
--------------------------------------------------------------------------------
found operation 32 + 32 = 64
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
found operation 0 - 10 = 115792089237316195423570985008687907853269984665640564039457584007913129639926
underflow detected at program counter 2146
Line: 37 : Source balanceOf[msg.sender] -= amount
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
found operation 32 + 0 = 32
--------------------------------------------------------------------------------

```