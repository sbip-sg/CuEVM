#ifndef STACK_CUH
#define STACK_CUH

#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include "uint256.cuh"
#define STACK_SIZE 100  // For example, temporarily set the stack size of 100

typedef struct {
    base_uint items[STACK_SIZE];
    int top;
} base_uint_stack;

__host__ __device__ void init_stack(base_uint_stack* stack);

__host__ __device__ bool push(base_uint_stack* stack, base_uint item);

__host__ __device__ bool pop(base_uint_stack* stack, base_uint* item);

__host__ __device__ bool swap_with_top(base_uint_stack* stack, int i);

__host__ __device__ void print_stack(base_uint_stack* stack);


#endif // STACK_CUH
