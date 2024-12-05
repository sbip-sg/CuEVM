pragma solidity ^0.8.0;
contract TestStateChange{
    uint public a = 1;
    function increase() public {
        a = a + 1;
    }
}