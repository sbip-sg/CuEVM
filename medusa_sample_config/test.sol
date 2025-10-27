pragma solidity ^0.7.0;

contract Test{
    uint256 public counter1 = 0;
    uint256 public counter2 = 0;
    uint256 public counter3 = 0;
    uint256 counter = 0;
    function set1(uint input) public {
        if (input % 5 == 1)
            counter1 ++;

        // counter2 = 0;
    }
    function set2(uint input) public {
        if (input % 20 == 3)
            counter2 ++;
        // counter1 = 0;
    }

    function unreachable() public {
        if (false) {
            counter ++;
        }
    }

    function bug1() public {
        assert(counter1 == 0);
    }

    function bug2() public {
        assert(counter2 == 0);
    }

}