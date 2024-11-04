pragma solidity ^0.8.0;
// SPDX-License-Identifier: MIT
// Modified from : https://solidity-by-example.org/app/erc20/
// Note : this one is to simulate a token ERC20 interface
// It's not a token or secured to use in production
// The bugs and everyone can mint are allowed on purpose
interface IERC20 {
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

    event Transfer(address indexed from, address indexed to, uint value);
    event Approval(address indexed owner, address indexed spender, uint value);
}
contract ERC20 is IERC20 {
    uint public totalSupply;
    mapping(address => uint) public balanceOf;
    mapping(address => mapping(address => uint)) public allowance;
    string public name = "Solidity by Example";
    string public symbol = "SOLBYEX";
    uint8 public decimals = 18;

    function transfer(address recipient, uint amount) external returns (bool) {
        unchecked {
            balanceOf[msg.sender] -= amount;
            balanceOf[recipient] += amount;
        }
        emit Transfer(msg.sender, recipient, amount);
        return true;
    }

    function approve(address spender, uint amount) public returns (bool) {
        allowance[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
        return true;
    }
    function approve_from(address owner, address spender, uint amount) internal {
        allowance[owner][spender] = amount;
    }

    function transferFrom(
        address sender,
        address recipient,
        uint amount
    ) external returns (bool) {
        unchecked {
            allowance[sender][msg.sender] -= amount;
            balanceOf[sender] -= amount;
            balanceOf[recipient] += amount;
        }
        emit Transfer(sender, recipient, amount);
        return true;
    }

    function mint() external payable{
        // simple mint for testing purpose
        unchecked {
            balanceOf[msg.sender] += msg.value;
            totalSupply += msg.value;
        }
        emit Transfer(address(0), msg.sender, msg.value);
    }

    function _mint(address receiver, uint amount) internal {
        unchecked {
            balanceOf[receiver] += amount;
            totalSupply += amount;
        }
        emit Transfer(address(0), msg.sender, amount);
    }

    function burn(uint amount) external {
        unchecked {
            balanceOf[msg.sender] -= amount;
            totalSupply -= amount;
        }
        emit Transfer(msg.sender, address(0), amount);
    }
    receive() external payable{}
}