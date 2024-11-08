// SPDX-License-Identifier: MIT
pragma solidity ^0.7.0;
contract TestBug {
    address test;
    uint test_value;
    function bug_selfdestruct() public {
        test_value = 1;
        selfdestruct(msg.sender);
    }

    function bug_unauthorized_send() public {
        test_value = 2;
        msg.sender.transfer(1 ether);
    }

    function bug_txorigin() public {
        test_value = 3;
        test = tx.origin;
    }

    function bug_overflow(uint input) public returns (uint) {
        test_value = 4;
        uint res = input + (2 ** 256 - 1000); // overflow if input > 1000
        return res;
    }

    function bug_underflow(uint input) public returns (uint) {
        test_value = 5;
        uint res = input - 10 ** 18;
        return res;
    }

    function bug_combined() public {
        bug_overflow(10001);
        bug_underflow(10001);
        bug_txorigin();
        bug_unauthorized_send();
        // bug_selfdestruct();
    }
}
