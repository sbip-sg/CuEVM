#ifndef _EVM_OPCODES_H_
#define _EVM_OPCODES_H_

// artithmetic
#define OP_STOP 0x00
#define OP_ADD 0x01
#define OP_MUL 0x02
#define OP_SUB 0x03
#define OP_DIV 0x04
#define OP_SDIV 0x05
#define OP_MOD 0x06
#define OP_SMOD 0x07
#define OP_ADDMOD 0x08
#define OP_MULMOD 0x09
#define OP_EXP 0x0A
#define OP_SIGNEXTEND 0x0B

// comparison & bitwise logic
#define OP_LT 0x10
#define OP_GT 0x11
#define OP_SLT 0x12
#define OP_SGT 0x13
#define OP_EQ 0x14
#define OP_ISZERO 0x15
#define OP_AND 0x16
#define OP_OR 0x17
#define OP_XOR 0x18
#define OP_NOT 0x19
#define OP_BYTE 0x1A
#define OP_SHL 0x1B
#define OP_SHR 0x1C
#define OP_SAR 0x1D

// SHA3
#define OP_SHA3 0x20

// environmental information
#define OP_ADDRESS 0x30
#define OP_BALANCE 0x31
#define OP_ORIGIN 0x32
#define OP_CALLER 0x33
#define OP_CALLVALUE 0x34
#define OP_CALLDATALOAD 0x35
#define OP_CALLDATASIZE 0x36
#define OP_CALLDATACOPY 0x37
#define OP_CODESIZE 0x38
#define OP_CODECOPY 0x39
#define OP_GASPRICE 0x3A
#define OP_EXTCODESIZE 0x3B
#define OP_EXTCODECOPY 0x3C
#define OP_RETURNDATASIZE 0x3D
#define OP_RETURNDATACOPY 0x3E
#define OP_EXTCODEHASH 0x3F

// block information
#define OP_BLOCKHASH 0x40
#define OP_COINBASE 0x41
#define OP_TIMESTAMP 0x42
#define OP_NUMBER 0x43
#define OP_DIFFICULTY 0x44
#define OP_GASLIMIT 0x45
#define OP_CHAINID 0x46
#define OP_SELFBALANCE 0x47
#define OP_BASEFEE 0x48


// stack, memory, storage and flow operations
#define OP_POP 0x50
#define OP_MLOAD 0x51
#define OP_MSTORE 0x52
#define OP_MSTORE8 0x53
#define OP_SLOAD 0x54
#define OP_SSTORE 0x55
#define OP_JUMP 0x56
#define OP_JUMPI 0x57
#define OP_PC 0x58
#define OP_MSIZE 0x59
#define OP_GAS 0x5A
#define OP_JUMPDEST 0x5B

// push
#define OP_PUSH1 0x60
#define OP_PUSH2 0x61
#define OP_PUSH3 0x62
#define OP_PUSH4 0x63
#define OP_PUSH5 0x64
#define OP_PUSH6 0x65
#define OP_PUSH7 0x66
#define OP_PUSH8 0x67
#define OP_PUSH9 0x68
#define OP_PUSH10 0x69
#define OP_PUSH11 0x6A
#define OP_PUSH12 0x6B
#define OP_PUSH13 0x6C
#define OP_PUSH14 0x6D
#define OP_PUSH15 0x6E
#define OP_PUSH16 0x6F
#define OP_PUSH17 0x70
#define OP_PUSH18 0x71
#define OP_PUSH19 0x72
#define OP_PUSH20 0x73
#define OP_PUSH21 0x74
#define OP_PUSH22 0x75
#define OP_PUSH23 0x76
#define OP_PUSH24 0x77
#define OP_PUSH25 0x78
#define OP_PUSH26 0x79
#define OP_PUSH27 0x7A
#define OP_PUSH28 0x7B
#define OP_PUSH29 0x7C
#define OP_PUSH30 0x7D
#define OP_PUSH31 0x7E
#define OP_PUSH32 0x7F

// dup
#define OP_DUP1 0x80
#define OP_DUP2 0x81
#define OP_DUP3 0x82
#define OP_DUP4 0x83
#define OP_DUP5 0x84
#define OP_DUP6 0x85
#define OP_DUP7 0x86
#define OP_DUP8 0x87
#define OP_DUP9 0x88
#define OP_DUP10 0x89
#define OP_DUP11 0x8A
#define OP_DUP12 0x8B
#define OP_DUP13 0x8C
#define OP_DUP14 0x8D
#define OP_DUP15 0x8E
#define OP_DUP16 0x8F

// swap
#define OP_SWAP1 0x90
#define OP_SWAP2 0x91
#define OP_SWAP3 0x92
#define OP_SWAP4 0x93
#define OP_SWAP5 0x94
#define OP_SWAP6 0x95
#define OP_SWAP7 0x96
#define OP_SWAP8 0x97
#define OP_SWAP9 0x98
#define OP_SWAP10 0x99
#define OP_SWAP11 0x9A
#define OP_SWAP12 0x9B
#define OP_SWAP13 0x9C
#define OP_SWAP14 0x9D
#define OP_SWAP15 0x9E
#define OP_SWAP16 0x9F

// log
#define OP_LOG0 0xA0
#define OP_LOG1 0xA1
#define OP_LOG2 0xA2
#define OP_LOG3 0xA3
#define OP_LOG4 0xA4

// system operations
#define OP_CREATE 0xF0
#define OP_CALL 0xF1
#define OP_CALLCODE 0xF2
#define OP_RETURN 0xF3
#define OP_DELEGATECALL 0xF4
#define OP_CREATE2 0xF5
#define OP_STATICCALL 0xFA
#define OP_REVERT 0xFD
#define OP_SELFDESTRUCT 0xFF

#endif