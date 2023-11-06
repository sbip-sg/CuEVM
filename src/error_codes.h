#ifndef _EVM_ERRORCODES_H_
#define _EVM_ERRORCODES_H_

// error codes
#define ERR_NONE 0x00
#define ERR_SUCCESS 0x00

// stack codes
#define ERR_STACK_UNDERFLOW 0x01
#define ERR_STACK_OVERFLOW 0x02
#define ERR_STACK_INVALID_SIZE 0x03
#define ERR_STACK_INVALID_INDEX 0x04

// arithmetic codes
#define ERR_INVALID_ZERO_DIVIDE 0x05
#define ERR_INVALID_MODULUS 0x06

// execution codes

// global storage error codes
#define ERR_STATE_INVALID_ADDRESS 0x07
#define ERR_STATE_INVALID_KEY 0x08


// operation not implemented
#define ERR_NOT_IMPLEMENTED 0x09

// EVM error codes
#define ERR_MAX_DEPTH_EXCEEDED 0x0A
#endif