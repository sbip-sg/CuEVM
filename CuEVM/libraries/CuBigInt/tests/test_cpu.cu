#include <gtest/gtest.h>
#include <CuBigInt/bigint.cuh>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

class BigIntTest : public ::testing::Test {
protected:
    void SetUp() override {
        bigint_init(a);
        bigint_init(b);
        bigint_init(c);
        bigint_init(d);
        for (int i = 0; i < 20; i++) bigint_init(e + i);
    }

    void TearDown() override {
        bigint_free(a);
        bigint_free(b);
        bigint_free(c);
        bigint_free(d);
        for (int i = 0; i < 20; i++) bigint_free(e + i);
    }

    bigint a[1], b[1], c[1], d[1], e[20];
};

TEST_F(BigIntTest, TestBigIntMulSubAdd) {
    bigint_from_str(a, "1");
    bigint_from_word(b, 1);
    bigint_shift_left(a, a, 1 << 10);

    bigint_mul(c, a, a);
    bigint_sub(c, c, a);
    bigint_sub(c, c, a);
    bigint_add(c, c, b);

    bigint_sub(a, a, b);
    bigint_mul(d, a, a);

    EXPECT_EQ(bigint_cmp(c, d), 0);
}

TEST_F(BigIntTest, TestBigIntFromStrBase) {
    char buf[65536];
    const char *text = "123456790123456790120987654320987654321";
    bigint_from_str_base(a, text, 10);

    int i = strcmp(text, bigint_write(buf, sizeof(buf), a));
    EXPECT_EQ(i, 0);
}

TEST_F(BigIntTest, TestBigIntFromInt) {
    bigint_from_int(a, INT_MIN);
    EXPECT_EQ(bigint_double(a), INT_MIN);
}

TEST_F(BigIntTest, TestBigIntFromStr) {
    bigint_from_str(a, "-9007199254740992");
    EXPECT_EQ(bigint_double(a), -9007199254740992.0);
}

TEST_F(BigIntTest, BasicCmpOperations) {
    bigint_from_int(a, 0);
    EXPECT_EQ(bigint_double(a), 0.0);
    bigint_from_int(a, +1);
    EXPECT_EQ(bigint_double(a), +1);
    bigint_from_int(a, -1);
    EXPECT_EQ(bigint_double(a), -1);

    bigint_from_str(a, "");
    bigint_from_str(b, "0");
    bigint_from_str(c, "-0");
    EXPECT_EQ(bigint_cmp(a, b), 0);
    EXPECT_EQ(bigint_cmp(a, c), 0);
}

TEST_F(BigIntTest, RandomizedOperations) {
    for (int i = 0; i < 12345; i++) {
        int x = rand() % 12345;
        int y = rand() % 12345;
        int shift = rand() % 1234;
        if (rand() & 1) x = -x;
        if (rand() & 1) y = -y;

        bigint_from_int(a, x);
        bigint_from_int(b, y);
        bigint_from_int(e + 0, x + y);
        bigint_from_int(e + 1, x - y);
        bigint_from_int(e + 2, x * y);

        if (y != 0) {
            bigint_from_int(e + 3, x / y);
            bigint_from_int(e + 4, x % y);
        }

        bigint_from_int(e + 5, x);
        bigint_from_int(e + 6, bigint_int_gcd(x, y));

        bigint_cpy(c, a);
        bigint_shift_left(a, a, shift);
        bigint_shift_right(a, a, shift);

        EXPECT_EQ(bigint_cmp(a, c), 0);

        bigint_add(e + 10, a, b);
        bigint_sub(e + 11, a, b);
        bigint_mul(e + 12, a, b);
        bigint_div(e + 13, a, b);
        bigint_mod(e + 14, a, b);
        bigint_from_int(e + 15, x);
        bigint_gcd(e + 16, a, b);

        for (int j = 0; j < 7; j++) {
            if (y == 0 && (j == 3 || j == 4)) continue;
            if (bigint_cmp(e + j, e + j + 10) != 0) {
                printf("i %i, j %i failed for bigints %i, %i\n", i, j, x, y);
            }
            EXPECT_EQ(bigint_cmp(e + j, e + j + 10), 0);
        }
    }
}