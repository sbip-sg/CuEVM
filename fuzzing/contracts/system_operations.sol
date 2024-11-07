pragma solidity ^0.7.0;
// SPDX-License-Identifier: MIT
// for demonstration purpose

contract TestCrossContract {
    function test_call(address addr, uint condition) public payable {
        bool res;
        if (condition == 0x1111) {
            (res, ) = addr.call{value: 1 ether}(
                abi.encodeWithSignature("increase()")
            );
        }
        if (condition == 0x2222) {
            (res, ) = addr.delegatecall(abi.encodeWithSignature("increase()"));
        }
        if (condition == 0x3333) {
            //selfdestruct address addr
            selfdestruct(payable(addr));
        }
        //call address addr
        (res, ) = addr.call(abi.encodeWithSignature("increase()"));
        //call address addr with value 1 ether

        //delegatecall address addr
        (res, ) = addr.delegatecall(abi.encodeWithSignature("increase()"));
    }
}
