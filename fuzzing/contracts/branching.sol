pragma solidity ^0.7.0;
contract TestBranching {
    uint public branch = 0;
    function test_branch(uint input) public {
        if (input < 10) branch = 1;
        else {
            if (input > 500) {
                if (input == 2345) branch = 3;
                else branch = 4;
            } else branch = 2;
        }
    }
    // function test_branch_1() public {
    //     branch = 1;
    // }

    // function test_branch_2(uint input) public {
    //     if (input == 4567){
    //         branch = 2;
    //         selfdestruct(msg.sender);
    //     }
    // }
    function test_branch_3(uint input) public {
        if (input == 45678) {
            branch = 2;
            msg.sender.send(1 ether);
        }
    }
}
