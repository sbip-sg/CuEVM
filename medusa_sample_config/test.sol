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
/*
contract Test {

    // uint256 public counter = 2 ** 256/4;
    uint256 public counter1 = 0;
    uint256 public counter2 = 0;
    uint256 public counter3 = 0;
    function dummy1(uint256 val) public returns (uint256) {
    }
    function dummy2(uint256 val) public returns (uint256) {
    }
    function set1(uint input) public {
        if (input % 5 == 3 && input < 1000000000000000000)
        counter1 ++;
    }
    function set2() public {
        counter2 ++;
    }
    function set3() public {
        counter3 ++;
    }
    
    function bug1() public {
        // assert (counter1 != 1);

        counter1 = 0;
    }

    function bug2() public {
        // assert (counter1 != 1 || counter2 != 1);

        counter1 = 0;
        counter2 = 0;
    }

    function bug3() public {
        // assert (counter1 != 1 || counter2 != 1 || counter3 != 1);

        counter1 = 0;
        counter2 = 0;
        counter3 = 0;
    }
}

contract Test{
    uint256 public counter1 = 0;
    uint256 public counter2 = 0;
    uint256 public counter3 = 0;
    uint256 counter = 0;
    function set1(uint input) public {
        if (input % 10 == 1)
            counter1 ++;

        counter2 = 0;
    }
    function set2(uint input) public {
        if (input % 10 == 2)
            counter2 ++;
        counter1 = 0;
    }
    function set3(uint input) public {
        if (input % 3000 == 1004)
            counter3 ++;
        counter1 = 0;
        counter2 = 0;
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


contract Test{
    uint256 public counter1 = 0;
    uint256 public counter2 = 0;
    uint256 public counter3 = 0;
    uint256 counter = 0;
    function set1(uint input) public {
        if (input % 5 == 2)
            counter1 ++;

        counter ++;
    }
    function set2(uint input) public {
        if (input % 20 == 3)
            counter2 ++;
        counter ++;
    }
    function set3(uint input) public {
        if (input % 3000 == 1004)
            counter3 ++;
        counter ++;
    }
    function unreachable() public {
        if (false) {
            counter ++;
        }
    }

    function bug1() public {
        assert(counter == 0);
    }

    function bug2() public {
        assert(counter1 == 0);
    }

}
*/