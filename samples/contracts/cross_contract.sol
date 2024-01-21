pragma solidity ^0.8.0;
// SPDX-License-Identifier: MIT
// Modified from : https://solidity-by-example.org/app/erc20/
// Note : this one is to simulate a token ERC20 interface
// It's not a token or secured to use in production
// The bugs and everyone can mint are allowed on purpose
interface ERC20Token { //extended with arbitrary minting and burning for testing purpose
    function totalSupply() external view returns (uint);

    function balanceOf(address account) external view returns (uint);

    function transfer(address recipient, uint amount) external returns (bool);

    function allowance(address owner, address spender) external view returns (uint);

    function approve(address spender, uint amount) external returns (bool);

    function transferFrom(
        address sender,
        address recipient,
        uint amount
    ) external returns (bool);

    function mint() external payable;
    function burn(uint amount) external;

    event Transfer(address indexed from, address indexed to, uint value);
    event Approval(address indexed owner, address indexed spender, uint value);
}

contract TestCrossContract{
    function underflow(address tokenA, address tokenB) public payable{

        // just simply mint and check the total supply demontraing 4 external calls
        ERC20Token tokenAContract = ERC20Token(tokenA);
        ERC20Token tokenBContract = ERC20Token(tokenB);
        // mint and check total supply :
        tokenAContract.mint{value: msg.value/3}(); // minted tokens = call value = msg.value/3
        tokenBContract.mint{value: msg.value/2}(); // minted tokens = call value = msg.value/2

        uint total_balance_a = tokenAContract.totalSupply();
        uint total_balance_b = tokenBContract.totalSupply();

        // check the diff in balance :
        unchecked {
            uint diff = total_balance_a - total_balance_b; // overflow here
            // For example, in the configurations of test, we send msg.value = 300 wei. So the token A supply = 100 and token B supply = 150, leading to underflow
        }
    }
}