#ifndef _CUEVM_ERRORCODES_H_
#define _CUEVM_ERRORCODES_H_

// error codes
#define ERROR_SUCCESS 0x00
#define ERROR_FAILED 0xFF
#define ERROR_RETURN 0xFE
#define ERROR_REVERT 0xFD

// stack codes
#define ERROR_STACK_UNDERFLOW 0x03
#define ERROR_STACK_OVERFLOW 0x04
#define ERROR_STACK_INVALID_SIZE 0x05
#define ERROR_STACK_INVALID_INDEX 0x06

// arithmetic codes
#define ERROR_INVALID_ZERO_DIVIDE 0x07
#define ERROR_INVALID_MODULUS 0x08


// global storage error codes
#define ERROR_STATE_ADDRESS_NOT_FOUND 0x09
#define ERROR_STORAGE_KEY_NOT_FOUND 0x0A


// operation not implemented
#define ERROR_NOT_IMPLEMENTED 0x0B

// EVM error codes
#define ERROR_MAX_DEPTH_EXCEEDED 0x0C

// block error codes
#define ERROR_BLOCK_INVALID_NUMBER 0x0D

// message error codes
#define ERROR_MESSAGE_INVALID_INDEX 0x0E

// memory error codes
#define ERR_MEMEORY_INVALID_INDEX 0x0F
#define ERR_MEMORY_INVALID_ALLOCATION 0x10
#define ERR_MEMORY_INVALID_OFFSET 0x16
#define ERR_MEMORY_INVALID_SIZE 0x1A

// execution codes
#define ERROR_INVALID_JUMP_DESTINATION 0x11


// static call error
#define ERROR_STATIC_CALL_CONTEXT 0x14

// gas
#define ERROR_OUT_OF_GAS 0x15

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
#define ERROR_TRANSACTION_FEES 0x20


// message calls
#define ERROR_MESSAGE_CALL_SENDER_BALANCE 0x20
#define ERROR_MESSAGE_CALL_CREATE_CONTRACT_EXISTS 0x31
#define ERROR_MESSAGE_CALL_DEPTH_EXCEEDED 0x32
#define ERROR_MESSAGE_CALL_CREATE_NONCE_EXCEEDED 0x33

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

// Precompile execution error
#define ERROR_PRECOMPILE_INVALID_ROUNDS 0x40
#define ERROR_PRECOMPILE_UNEXPECTED_INPUT_LENGTH 0x41
#define ERROR_PRECOMPILE_EXECUTION_FAILURE 0x42
#define ERROR_PRECOMPILE_UNEXPECTED_INPUT 0x43

// precompile errors
#define ERROR_PRECOMPILE_MODEXP_OVERFLOW 0x36

// memory allocation
#define ERROR_MEMORY_ALLOCATION_FAILED 0x37

// wrong evm word size
#define ERROR_INVALID_WORD_SIZE 0x38

// byte array errors
#define ERROR_BYTE_ARRAY_INVALID_SIZE 0x39
#define ERROR_BYTE_ARRAY_OVERFLOW_VALUES 0x3A

// addded errors
#define ERROR_INSUFFICIENT_FUNDS 0x3B


// core errors
#define ERROR_INVALID_HEX_STRING 0x3C


#endif