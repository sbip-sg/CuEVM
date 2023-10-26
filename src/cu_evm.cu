#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <cuda_runtime.h>

#define NUMTHREAD 4096

// simple draft kernel for place holder
__global__ void cuEVM(unsigned char *bytecode, unsigned char *input, size_t bytecode_len, size_t input_len, size_t num_threads)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num_threads)
    {
        if (idx == 0)
        {
            printf("Bytecode: ");
            for (size_t i = 0; i < bytecode_len; i++)
            {
                printf("%02x ", bytecode[i]);
            }
            printf("\n");

            printf("Input: ");
            for (size_t i = 0; i < input_len; i++)
            {
                printf("%02x ", input[i]);
            }
            printf("\n");
        }
    }
}
int adjustedLength(char** hexString) {
    if (strncmp(*hexString, "0x", 2) == 0 || strncmp(*hexString, "0X", 2) == 0) {
        *hexString += 2;  // Skip the "0x" prefix
        return (strlen(*hexString) / 2);
    }
    return (strlen(*hexString) / 2);
}

void hexStringToByteArray(const char *hexString, unsigned char *byteArray, int length)
{
    for (int i = 0; i < length; i += 2)
    {
        sscanf(&hexString[i], "%2hhx", &byteArray[i / 2]);
    }
}

int main(int argc, char *argv[])
{
    char *byte_code_hex = NULL;
    char *input_hex = NULL;

    static struct option long_options[] = {
        {"bytecode", required_argument, 0, 'b'},
        {"input", required_argument, 0, 'i'},
        {0, 0, 0, 0}};

    int opt;
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, "b:i:", long_options, &option_index)) != -1)
    {
        switch (opt)
        {
        case 'b':
            byte_code_hex = optarg;
            break;
        case 'i':
            input_hex = optarg;
            break;
        default:
            fprintf(stderr, "Usage: %s --bytecode <hexstring> --input <hexstring>\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    if (!byte_code_hex || !input_hex)
    {
        fprintf(stderr, "Both --bytecode and --input flags are required\n");
        exit(EXIT_FAILURE);
    }

    int bytecode_len = adjustedLength(&byte_code_hex);
    int input_len = adjustedLength(&input_hex);

    unsigned char *byte_code = (unsigned char *)malloc(bytecode_len);
    unsigned char *input = (unsigned char *)malloc(input_len);

    hexStringToByteArray(byte_code_hex, byte_code, bytecode_len * 2);
    hexStringToByteArray(input_hex, input, input_len * 2);

    printf("Pass conversion\n");
    unsigned char *d_bytecode, *d_input;
    cudaMalloc((void **)&d_bytecode, bytecode_len);
    cudaMalloc((void **)&d_input, input_len);

    cudaMemcpy(d_bytecode, byte_code, bytecode_len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, input, input_len, cudaMemcpyHostToDevice);
    printf("Pass allocation and memcpy\n");

    int blockSize = 256;
    int numBlocks = (NUMTHREAD + blockSize - 1) / blockSize;
    cuEVM<<<numBlocks, blockSize>>>(d_bytecode, d_input, bytecode_len, input_len, NUMTHREAD);
    printf("RUN\n");

    cudaDeviceSynchronize();
    printf("Syncrhronize\n");

    cudaFree(d_bytecode);
    cudaFree(d_input);
    free(byte_code);
    free(input);

    return 0;
}
