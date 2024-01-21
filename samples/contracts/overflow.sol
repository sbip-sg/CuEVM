pragma solidity ^0.7.0;
contract TestOverflow{
    uint public a = 0;
    function set_state(uint256 val) public {
        a = val;
    }
    function multiply(uint256 factor) public {
        a = a * factor;
    }
}