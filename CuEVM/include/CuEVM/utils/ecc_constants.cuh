#pragma once

#include <CuEVM/core/evm_word.cuh>

namespace CuEVM {
/**
 * The arithmetic environment class is a wrapper around the CGBN library.
 * It provides a context, environment, and instance for the CGBN library.
 * It also provides some utility functions for converting between CGBN and other
 * types.
 */

/// The secp256k1 field prime number (P) and order
inline constexpr const char *secp256k1_FieldPrime_hex =
    "0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f";
inline constexpr const char *secp256k1_Order_hex = "0xfffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141";

inline constexpr const char *secp256k1_GX_hex = "0x79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798";
inline constexpr const char *secp256k1_GY_hex = "0x483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8";

/// The alt_BN128 field prime number (P) and order
inline constexpr const char *alt_BN128_FieldPrime_hex =
    "0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47";
inline constexpr const char *alt_BN128_Order_hex = "0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001";

inline constexpr const char *alt_BN128_G2_B_0_hex =
    "0x2b149d40ceb8aaae81be18991be06ac3b5b4c5e559dbefa33267e6dc24a138e5";
inline constexpr const char *alt_BN128_G2_B_1_hex = "0x9713b03af0fed4cd2cafadeed8fdf4a74fa084e52d1852e4a2bd0685c315d2";

inline constexpr const char *alt_BN128_GX_hex = "0x01";
inline constexpr const char *alt_BN128_GY_hex = "0x02";

inline constexpr const char *alt_BN128_G2X1_hex = "0x1800deef121f1e76426a00665e5c4479674322d4f75edadd46debd5cd992f6ed";
inline constexpr const char *alt_BN128_G2X2_hex = "0x198e9393920d483a7260bfb731fb5d25f1aa493335a9e71297e485b7aef312c2";
inline constexpr const char *alt_BN128_G2Y1_hex = "0x12c85ea5db8c6deb4aab71808dcb408fe3d1e7690c43d37b4ce6cc0166fa7daa";
inline constexpr const char *alt_BN128_G2Y2_hex = "0x90689d0585ff075ec9e99ad690c3395bc4b313370b38ef355acdadcd122975b";
inline constexpr const char *ate_loop_count_hex = "0x19d797039be763ba8";
constexpr size_t log_ate_loop_count = 63;

struct Curve {
    evm_word_t FieldPrime;
    evm_word_t Order;
    evm_word_t GX;
    evm_word_t GY;
    uint32_t B = 3;
};

struct EccConstants {
    Curve secp256k1;
    Curve alt_BN128;
    evm_word_t alt_BN128_G2_B_0;
    evm_word_t alt_BN128_G2_B_1;
    evm_word_t alt_BN128_G2X1;
    evm_word_t alt_BN128_G2X2;
    evm_word_t alt_BN128_G2Y1;
    evm_word_t alt_BN128_G2Y2;
    evm_word_t ate_loop_count;
    evm_word_t final_exp[11];

    __host__ EccConstants() {
        // Initialize secp256k1 curve
        secp256k1.FieldPrime.from_hex(secp256k1_FieldPrime_hex);
        secp256k1.Order.from_hex(secp256k1_Order_hex);
        secp256k1.GX.from_hex(secp256k1_GX_hex);
        secp256k1.GY.from_hex(secp256k1_GY_hex);
        secp256k1.B = 7;

        // Initialize alt_BN128 curve
        alt_BN128.FieldPrime.from_hex(alt_BN128_FieldPrime_hex);
        alt_BN128.Order.from_hex(alt_BN128_Order_hex);
        alt_BN128.GX.from_hex(alt_BN128_GX_hex);
        alt_BN128.GY.from_hex(alt_BN128_GY_hex);
        alt_BN128.B = 3;

        // Initialize other alt_BN128 constants
        alt_BN128_G2_B_0.from_hex(alt_BN128_G2_B_0_hex);
        alt_BN128_G2_B_1.from_hex(alt_BN128_G2_B_1_hex);
        alt_BN128_G2X1.from_hex(alt_BN128_G2X1_hex);
        alt_BN128_G2X2.from_hex(alt_BN128_G2X2_hex);
        alt_BN128_G2Y1.from_hex(alt_BN128_G2Y1_hex);
        alt_BN128_G2Y2.from_hex(alt_BN128_G2Y2_hex);
        ate_loop_count.from_hex(ate_loop_count_hex);
        const char *final_exp_const[11] = {"0000002f4b6dc97020fddadf107d20bc842d43bf6369b1ff6a1c71015f3f7be2",
                                           "e1e30a73bb94fec0daf15466b2383a5d3ec3d15ad524d8f70c54efee1bd8c3b2",
                                           "1377e563a09a1b705887e72eceaddea3790364a61f676baaf977870e88d5c6c8",
                                           "fef0781361e443ae77f5b63a2a2264487f2940a8b1ddb3d15062cd0fb2015dfc",
                                           "6668449aed3cc48a82d0d602d268c7daab6a41294c0cc4ebe5664568dfc50e16",
                                           "48a45a4a1e3a5195846a3ed011a337a02088ec80e0ebae8755cfe107acf3aafb",
                                           "40494e406f804216bb10cf430b0f37856b42db8dc5514724ee93dfb10826f0dd",
                                           "4a0364b9580291d2cd65664814fde37ca80bb4ea44eacc5e641bbadf423f9a2c",
                                           "bf813b8d145da90029baee7ddadda71c7f3811c4105262945bba1668c3be69a3",
                                           "c230974d83561841d766f9c9d570bb7fbe04c7e8a6c3c760c0de81def35692da",
                                           "361102b6b9b2b918837fa97896e84abb40a4efb7e54523a486964b64ca86f120"};
        for (int i = 0; i < 11; i++) {
            final_exp[i].from_hex(final_exp_const[i]);
        }
        // printf("EccConstants initialized\n");
    }
};
}  // namespace CuEVM