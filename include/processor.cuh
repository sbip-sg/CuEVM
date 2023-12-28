#ifndef PROCESSOR_CUH
#define PROCESSOR_CUH

#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include "uint256.cuh"
#include "stack.cuh"
#define EVM_VERSION "petersburg" // hardcode to only one version !

typedef struct {
    base_uint origin;
    base_uint block_numer;
    base_uint block_difficulty;
    // other fields
} environment;

// Struct to represent a call frame
// Not implemented yet (cross contract)
// typedef struct {
//     base_uint caller;
//     base_uint callValue;
//     base_uint gasLimit; // not supported
//     uint8_t* inputData;
//     size_t inputDataSize;
// } CallFrame;

// Struct for the EVM processor
typedef struct {
    // other fields as new functionality is added
    // base_uint gasRemaining; not implemented
    // CallFrame* callStack; // not implemented yet
    base_uint_stack stack;
    base_uint caller;
    uint32_t programCounter;
    // uint8_t* bytecode; temporarily not needed in single contract
} processor;
// util functions

#endif // PROCESSOR_CUH
