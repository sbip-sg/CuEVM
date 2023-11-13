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

// execution codes

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

#endif