#ifndef _EVM_DEFINES_H_
#define _EVM_DEFINES_H_

#define EIP_170
#define EIP_3860

#define BITS_PER_BYTE 8

#define EVM_WORD_SIZE 32

#define EVM_WORD_BITS (EVM_WORD_SIZE * BITS_PER_BYTE)

#define EVM_MAX_STACK_SIZE 1024

#define EVM_ADDRESS_SIZE 20

#define EVM_ADDRESS_BITS (EVM_ADDRESS_SIZE * BITS_PER_BYTE)

#define EVM_MAX_DEPTH 1024

#define EVM_HASH_SIZE 32

#ifdef EIP_170
#define EVM_MAX_CODE_SIZE 24576
#else
#error "EIP_170 is not defined"
#endif

#ifdef EIP_3860
#define EVM_MAX_INITCODE_SIZE (2 * EVM_MAX_CODE_SIZE)
#else
#error "EIP_3860 is not defined"
#endif


// CGBN parameters
#define CGBN_TPI 32

#endif