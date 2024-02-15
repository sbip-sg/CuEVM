#include <cuda_runtime.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuevm_test.h"
#include "opcode.h"
#include "processor.cuh"
#include "stack.cuh"
#include "uint256.cuh"
#define NUMTHREAD 4096
#define DEBUG 1
// simple draft kernel for place holder
// simple testing opcodes and return the popped top of stack value
__global__ void cuEVM(unsigned char *bytecode, unsigned char *input, size_t bytecode_len, size_t input_len,
                      size_t num_threads) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num_threads) {
        if (idx == 0) {
            printf("Bytecode: ");
            for (size_t i = 0; i < bytecode_len; i++) {
                printf("%02x ", bytecode[i]);
            }
            printf("\n");

            printf("Input: ");
            for (size_t i = 0; i < input_len; i++) {
                printf("%02x ", input[i]);
            }
            printf("\n");

            // todo refactor processor.execute to a function outside of this kernel
            processor evm;
            evm.programCounter = 0;  // redundant but better safe.
            init_stack(&evm.stack);

            // push(&stack, a);
            // pop(&stack, &b);
            // debugging : print_stack(&stack);
            // define 3 reusable temp uints for binary op
            base_uint op1, op2, result;
            uint safe_counter = 0;  // safety counter prevent infinite loop
            while (evm.programCounter < bytecode_len) {
                unsigned char opcode = bytecode[evm.programCounter];
                safe_counter++;
                if (safe_counter > 100) {
                    printf("Safety counter exceeded, return from execution\n");
                    return;
                }
                switch (opcode) {
                    case ADD:  // ADD
                        // TODO: check stack size
                        // future optimization : can override push pop ops and modify the stack directly
                        pop(&evm.stack, &op1);
                        pop(&evm.stack, &op2);
                        base_uint_add(&op1, &op2, &result);

#if DEBUG
                        printf("ADD OPCODE: \n");
                        printf("op1: ");
                        print_base_uint(&op1);
                        printf("op2: ");
                        print_base_uint(&op2);
                        printf("result: ");
                        print_base_uint(&result);

#endif

                        push(&evm.stack, result);
                        break;
                    case SUB:
                        pop(&evm.stack, &op1);
                        pop(&evm.stack, &op2);
                        base_uint_sub(&op1, &op2, &result);

#if DEBUG
                        printf("SUB OPCODE: \n");
                        printf("op1: ");
                        print_base_uint(&op1);
                        printf("op2: ");
                        print_base_uint(&op2);
                        printf("result: ");
                        print_base_uint(&result);

#endif

                        push(&evm.stack, result);
                        break;

                    case MUL:  // MUL
                        // TODO: check stack size
                        pop(&evm.stack, &op1);
                        pop(&evm.stack, &op2);
                        base_uint_mul(&op1, &op2, &result);

#if DEBUG
                        printf("MUL OPCODE: \n");
                        printf("op1: ");
                        print_base_uint(&op1);
                        printf("op2: ");
                        print_base_uint(&op2);
                        printf("result: ");
                        print_base_uint(&result);

#endif

                        push(&evm.stack, result);
                        break;
                    case PUSH1:
                        unsigned char push_val = bytecode[++evm.programCounter];
                        result = {{push_val, 0, 0, 0, 0, 0, 0, 0}};
                        push(&evm.stack, result);

#if DEBUG
                        printf("PUSH1 OPCODE: \n");
                        printf("push_val: ");
                        print_base_uint(&result);

#endif

                        break;
                    case PUSH2:
                        // Increment the program counter to point to the first byte of data
                        evm.programCounter++;

                        // Read the two bytes from the bytecode
                        unsigned char byte1 = bytecode[evm.programCounter];
                        unsigned char byte2 = bytecode[++evm.programCounter];

                        // Combine the two bytes into a single 16-bit value
                        uint16_t push_val_16 = (byte1 << 8) | byte2;

                        // Convert the 16-bit value into your base_uint format
                        result = {{push_val_16, 0, 0, 0, 0, 0, 0, 0}};

                        // Push the value onto the stack
                        push(&evm.stack, result);

#if DEBUG
                        printf("PUSH2 OPCODE: \n");
                        printf("push_val: ");
                        print_base_uint(&result);

#endif

                        break;

                    case POP:
                        pop(&evm.stack, &result);
                        printf("Popped Stack value: ");
                        print_base_uint(&result);
                        break;

                    case SWAP1:
                        pop(&evm.stack, &op1);
                        pop(&evm.stack, &op2);
                        push(&evm.stack, op1);
                        push(&evm.stack, op2);
                        break;

                    case DUP1:
                        printf("DUP1 OPCODE: \n");
                        push(&evm.stack, evm.stack.items[evm.stack.top]);
                        break;
                    case JUMPI:
                        pop(&evm.stack, &op1);
                        pop(&evm.stack, &op2);
#if DEBUG
                        printf("JUMPI OPCODE: \n");
                        printf("Condition:\n");
                        print_base_uint(&op2);
                        printf("Destination:\n");
                        print_base_uint(&op1);
                        printf("is ZEROP op1: %d\n", is_zero(&op1));
#endif
                        if (!is_zero(&op2)) {
                            evm.programCounter = op1.pn[0];
                            // TODO: check JUMPDEST in destiation ?
                        }
                        break;

                    case JUMP:
                        pop(&evm.stack, &op1);
                        evm.programCounter = op1.pn[0];
                        // TODO: check JUMPDEST in destiation ?
                        break;

                    case JUMPDEST:
                        // do nothing
                        break;
                    case RETURN:
                        printf("RETURN OPCODE\n");
                        evm.programCounter = bytecode_len;
                        break;
                    default:
                        printf("Unknown opcode %d at position %d\n", opcode, evm.programCounter);
                        printf("Return from execution\n");
                        return;
                }

                evm.programCounter++;
#if DEBUG
                printf("Program counter: %d\n", evm.programCounter);
                printf("Stack size: %d\n", evm.stack.top + 1);
                print_stack(&evm.stack);
                printf("\n***************\n");
#endif
            }
        }
    }
}
int adjustedLength(char **hexString) {
    if (strncmp(*hexString, "0x", 2) == 0 || strncmp(*hexString, "0X", 2) == 0) {
        *hexString += 2;  // Skip the "0x" prefix
        return (strlen(*hexString) / 2);
    }
    return (strlen(*hexString) / 2);
}

void hexStringToByteArray(const char *hexString, unsigned char *byteArray, int length) {
    for (int i = 0; i < length; i += 2) {
        sscanf(&hexString[i], "%2hhx", &byteArray[i / 2]);
    }
}

int main(int argc, char *argv[]) {
    char *byte_code_hex = NULL;
    char *input_hex = NULL;

    static struct option long_options[] = {{"bytecode", required_argument, 0, 'b'},
                                           {"input", required_argument, 0, 'i'},
                                           {"test", no_argument, 0, 't'},
                                           {0, 0, 0, 0}};

    int opt;
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, "b:i:", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'b':
                byte_code_hex = optarg;
                break;
            case 'i':
                input_hex = optarg;
                break;
            case 't':
                test_arithmetic_operations();
                test_stack();
                exit(0);
            default:
                fprintf(stderr, "Usage: %s --bytecode <hexstring> --input <hexstring>\n", argv[0]);
                exit(EXIT_FAILURE);
        }
    }

    if (!byte_code_hex || !input_hex) {
        fprintf(stderr, "Both --bytecode and --input flags are required\n");
        exit(EXIT_FAILURE);
    }

    int bytecode_len = adjustedLength(&byte_code_hex);
    int input_len = adjustedLength(&input_hex);

    unsigned char *byte_code = (unsigned char *)malloc(bytecode_len);
    unsigned char *input = (unsigned char *)malloc(input_len);

    hexStringToByteArray(byte_code_hex, byte_code, bytecode_len * 2);
    hexStringToByteArray(input_hex, input, input_len * 2);

    unsigned char *d_bytecode, *d_input;
    cudaMalloc((void **)&d_bytecode, bytecode_len);
    cudaMalloc((void **)&d_input, input_len);

    cudaMemcpy(d_bytecode, byte_code, bytecode_len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, input, input_len, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (NUMTHREAD + blockSize - 1) / blockSize;
    cuEVM<<<numBlocks, blockSize>>>(d_bytecode, d_input, bytecode_len, input_len, NUMTHREAD);

    cudaDeviceSynchronize();

    cudaFree(d_bytecode);
    cudaFree(d_input);
    free(byte_code);
    free(input);

    return 0;
}
