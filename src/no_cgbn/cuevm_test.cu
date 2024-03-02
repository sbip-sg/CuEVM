#include "cuevm_test.h"

void test_arithmetic_operations() {
    base_uint a, b, c, d;

    // Test addition
    base_uint_set_hex(&a, "11111111111111111111111111111111");
    base_uint_set_hex(&b, "22222222222222222222222222222222");
    base_uint_add(&a, &b, &c);
    printf("Addition Result: ");

    char buffer[BITS / 4 + 1] = {0};
    base_uint_get_hex(&c, buffer);

    printf("%s\n", buffer);

    if (strcmp(buffer, "0000000000000000000000000000000033333333333333333333333333333333") != 0) {
        printf("Addition failed!\n");
    }
    // Test addition with carry
    base_uint_set_hex(&a, "1");
    base_uint_set_hex(&b, "ffffffffffffffffffffffffffffffff");
    base_uint_add(&a, &b, &c);
    printf("Addition Result: ");

    base_uint_get_hex(&c, buffer);

    printf("%s\n", buffer);

    if (strcmp(buffer, "0000000000000000000000000000000100000000000000000000000000000000") != 0) {
        printf("Addition failed!\n");
    }
    // Test addition overflow carry
    base_uint_set_hex(&a, "1234");
    base_uint_set_hex(&b, "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff");
    base_uint_add(&a, &b, &c);
    printf("Addition Result: ");

    base_uint_get_hex(&c, buffer);

    printf("%s\n", buffer);

    if (strcmp(buffer, "0000000000000000000000000000000000000000000000000000000000001233") != 0) {
        printf("Addition failed!\n");
    }

    // Test subtraction
    base_uint_set_hex(&a, "ffffffffffffffffffffffffffffffff");
    base_uint_set_hex(&b, "fe");
    base_uint_sub(&a, &b, &c);
    printf("Subtraction Result: ");
    base_uint_get_hex(&c, buffer);
    printf("%s\n", buffer);
    if (strcmp(buffer, "00000000000000000000000000000000ffffffffffffffffffffffffffffff01") != 0) {
        printf("Subtraction failed!\n");
    }

    // Test subtraction underflow
    base_uint_set_hex(&a, "01");
    base_uint_set_hex(&b, "ff");
    base_uint_sub(&a, &b, &c);
    printf("Subtraction Result: ");
    base_uint_get_hex(&c, buffer);
    printf("%s\n", buffer);
    if (strcmp(buffer, "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff02") != 0) {
        printf("Subtraction failed!\n");
    }

    // Test multiplication
    base_uint_set_hex(&a, "ffffffffffffffffffffffffffffffff");
    base_uint_set_hex(&b, "ffffffffffffffffffffffffffffff");
    base_uint_mul(&a, &b, &c);
    printf("Multiplication Result: ");
    base_uint_get_hex(&c, buffer);
    printf("%s\n", buffer);
    if (strcmp(buffer, "00fffffffffffffffffffffffffffffeff000000000000000000000000000001") != 0) {
        printf("Multiplication failed!\n");
    }
    // Test multiplication overflow
    base_uint_set_hex(&a, "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff");
    base_uint_set_hex(&b, "2");
    base_uint_mul(&a, &b, &c);
    printf("Multiplication Result: ");
    base_uint_get_hex(&c, buffer);
    printf("%s\n", buffer);
    if (strcmp(buffer, "0000000000000000000000000000000000000000000000000000000000000000") != 0) {
        printf("Multiplication overflow failed!\n");
    }
}

void test_stack() {
    base_uint_stack stack;
    init_stack(&stack);

    // Test push and print
    base_uint a = {{1, 2, 3, 4}};
    printf("Pushing: ");
    for (int i = 0; i < WIDTH; i++) printf("%u ", a.pn[i]);
    printf("\n");
    push(&stack, a);
    print_stack(&stack);

    // Test pop
    base_uint b;
    if (pop(&stack, &b)) {
        printf("Popped: ");
        for (int i = 0; i < WIDTH; i++) printf("%u ", b.pn[i]);
        printf("\n");
    }
    print_stack(&stack);

    // Test swap with top
    push(&stack, a);
    base_uint c = {{5, 6, 7, 8}};
    push(&stack, c);
    printf("Before swap with top:\n");
    print_stack(&stack);
    swap_with_top(&stack, 0);
    printf("After swap with top:\n");
    print_stack(&stack);
}
