#ifndef _EVM_ERRORCODES_H_
#define _EVM_ERRORCODES_H_

// error codes
#define ERR_NONE 0x00
#define ERR_SUCCESS 0x00
#define ERR_RETURN 0x01
#define ERR_REVERT 0x02

// stack codes
#define ERR_STACK_UNDERFLOW 0x03
#define ERR_STACK_OVERFLOW 0x04
#define ERR_STACK_INVALID_SIZE 0x05
#define ERR_STACK_INVALID_INDEX 0x06

// arithmetic codes
#define ERR_INVALID_ZERO_DIVIDE 0x07
#define ERR_INVALID_MODULUS 0x08


// global storage error codes
#define ERR_STATE_INVALID_ADDRESS 0x09
#define ERR_STATE_INVALID_KEY 0x0A


// operation not implemented
#define ERR_NOT_IMPLEMENTED 0x0B

// EVM error codes
#define ERR_MAX_DEPTH_EXCEEDED 0x0C

// block error codes
#define ERR_BLOCK_INVALID_NUMBER 0x0D

// message error codes
#define ERR_MESSAGE_INVALID_INDEX 0x0E

// memory error codes
#define ERR_MEMEORY_INVALID_INDEX 0x0F
#define ERR_MEMORY_INVALID_ALLOCATION 0x10
#define ERR_MEMORY_INVALID_OFFSET 0x16

// execution codes
#define ERR_INVALID_JUMP_DESTINATION 0x11


// static call error
#define ERR_STATIC_CALL_CONTEXT 0x14

// gas
#define ERR_OUT_OF_GAS 0x15

// transaction
#define ERROR_TRANSACTION_TYPE 0x17
#define ERROR_TRANSACTION_SENDER_EMPTY 0x18
#define ERROR_TRANSACTION_SENDER_CODE 0x19
#define ERROR_TRANSACTION_NONCE 0x1A
#define ERROR_TRANSACTION_GAS 0x1B
#define ERROR_TRANSACTION_SENDER_BALANCE 0x1C
#define ERROR_TRANSACTION_GAS_PRICE 0x1D
#define ERROR_TRANSACTION_GAS_PRIORITY 0x1E
#define ERROR_TRANSACTION_BLOCK_GAS_LIMIT 0x1F


// message calls
#define ERROR_MESSAGE_CALL_SENDER_BALANCE 0x20
#define ERROR_MESSAGE_CALL_CREATE_CONTRACT_EXISTS 0x31
#define ERROR_MESSAGE_CALL_DEPTH_EXCEEDED 0x32

// gas
#define ERROR_GAS_LIMIT_EXCEEDED 0x21

// stack
#define ERROR_STACK_INVALID_PUSHX_X 0x22
#define ERROR_STACK_INVALID_DUPX_X 0x23


// return data errors
#define ERROR_RETURN_DATA_INVALID_SIZE 0x24
#define ERROR_RETURN_DATA_OVERFLOW 0x25

// static call erorr
#define ERROR_STATIC_CALL_CONTEXT_CALL_VALUE 0x26
#define ERROR_STATIC_CALL_CONTEXT_SSTORE 0x27
#define ERROR_STATIC_CALL_CONTEXT_CREATE 0x28
#define ERROR_STATIC_CALL_CONTEXT_LOG 0x29
#define ERROR_STATIC_CALL_CONTEXT_SELFDESTRUCT 0x2A
#define ERROR_STATIC_CALL_CONTEXT_CREATE2 0x2B


// EVM error codes
#define ERROR_MAX_DEPTH_EXCEEDED    0x0C


// CREATE errors
#define ERROR_CREATE2_ADDRESS_ALREADY_EXISTS 0x30
#define ERROR_CREATE_INIT_CODE_SIZE_EXCEEDED 0x33
#define ERROR_CREATE_CODE_SIZE_EXCEEDED 0x34
#define ERROR_CREATE_CODE_FIRST_BYTE_INVALID 0x35

#endif