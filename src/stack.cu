#include "stack.cuh"

__host__ __device__ void init_stack(base_uint_stack* stack) { stack->top = -1; }

__host__ __device__ bool push(base_uint_stack* stack, base_uint item) {
    if (stack->top >= STACK_SIZE - 1) {
        return false;  // Stack is full
    }
    stack->top++;
    stack->items[stack->top] = item;
    return true;
}

__host__ __device__ bool pop(base_uint_stack* stack, base_uint* item) {
    if (stack->top < 0) {
        return false;  // Stack is empty
    }
    *item = stack->items[stack->top];
    stack->top--;
    return true;
}

__host__ __device__ bool swap_with_top(base_uint_stack* stack, int i) {
    if (stack->top < 0 || i > stack->top || i < 0) {
        return false;  // Stack is empty or index out of bounds
    }
    base_uint temp = stack->items[i];
    stack->items[i] = stack->items[stack->top];
    stack->items[stack->top] = temp;
    return true;
}

__host__ __device__ void print_stack(base_uint_stack* stack) {
    printf("Stack: ");
    for (int i = 0; i <= stack->top; i++) {
        printf("[");
        for (int j = 0; j < WIDTH; j++) {
            printf("%u", stack->items[i].pn[j]);
            if (j < WIDTH - 1) printf(",");
        }
        printf("] ");
    }
    printf("\n");
}
