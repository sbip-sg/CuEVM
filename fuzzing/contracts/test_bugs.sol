pragma solidity ^0.7.0;
contract TestBug {
    uint public branch = 0;
    address test;
    function bug_selfdestruct(uint input) public {
        require(input == 4567);
        branch = 2;
        selfdestruct(msg.sender);
    }

    function bug_unauthorized_send(uint input) public {
        // msg.sender.transfer(1 ether);
        // input = 40000 triggers the bug
        if (input + 5678 == 45678) msg.sender.transfer(1 ether);
    }

    function bug_txorigin(uint input) public {
        if (input > 20000) test = tx.origin;
    }

    function bug_overflow(uint input1, uint input2) public returns (uint) {
        // overflow if input1%2 == 0 and input2%2 == 0 and input1 + input2 > 10000
        if (input1 % 2 == 0 && input2 % 2 == 0)
            return input1 + input2 + (2 ** 256 - 10000);
    }
    function bug_underflow(uint input) public {
        if (input < 100) branch = 0 - input;
    }
}
