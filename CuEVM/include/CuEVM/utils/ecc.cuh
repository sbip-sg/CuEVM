// Elliptic curve utilities using CGBN
#pragma once
#include <CuCrypto/keccak.cuh>
#include <CuEVM/core/evm_word.cuh>
#include <CuEVM/utils/evm_utils.cuh>
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
namespace ecc {
using namespace CuEVM;

typedef struct {
    evm_word_t r;
    evm_word_t s;
    uint32_t v;
    evm_word_t msg_hash;
    evm_word_t res;
    evm_word_t signer;
} signature_t;

// Reuse Curve struct definition from CuEVM namespace
using Curve = CuEVM::Curve;

template <size_t Degree>
struct FQ {
    evm_word_t coeffs[Degree];
};

template <size_t Degree>
__device__ void print_fqp(FQ<Degree> &P, const char *name);

__device__ void uint256_modular_inverse(uint256 *res, uint256 *a, uint256 *mod);

__device__ void uint256_mul_mod(uint256 *res, uint256 *a, uint256 *b, uint256 *mod);

__device__ void uint256_add_mod(uint256 *res, uint256 *a, uint256 *b, uint256 *mod);

__device__ void uint256_sub_mod(uint256 *res, uint256 *a, uint256 *b, uint256 *mod);

// __device__ void uint256_div_mod(uint256 *res, uint256 *a, uint256 *b, uint256 *mod);

__device__ bool is_on_cuve_simple(uint256 *Px, uint256 *Py, uint256 *mod, uint32_t B);

template <size_t Degree>
__device__ bool FQP_equals(FQ<Degree> *P1, FQ<Degree> *P2);

__device__ int ec_add(Curve curve, evm_word_t *ResX, evm_word_t *ResY, evm_word_t *Px, evm_word_t *Py, evm_word_t *Qx,
                      evm_word_t *Qy, bool check_curve = true);

__device__ int ec_mul(Curve curve, evm_word_t *ResX, evm_word_t *ResY, evm_word_t *Gx, evm_word_t *Gy, evm_word_t *n);

__device__ void convert_point_to_address(evm_word_t *address, evm_word_t *X, evm_word_t *Y);

__device__ int ec_recover(CuEVM::EccConstants *ecc_constants_ptr, signature_t *sig, evm_word_t *signer);

// template <size_t Degree>
// __device__ void getFQ12_from_cgbn_t(ArithEnv &arith, FQ<Degree> &res, bn_t (&coeffs)[Degree]);

template <size_t Degree>
__device__ void FQP_add(FQ<Degree> *Res, FQ<Degree> *P1, FQ<Degree> *P2, uint256 *mod);

template <size_t Degree>
__device__ void FQP_sub(FQ<Degree> *Res, FQ<Degree> *P1, FQ<Degree> *P2, uint256 *mod);

template <size_t Degree>
__device__ uint deg(FQ<Degree> *P);

template <size_t Degree>
__device__ FQ<Degree> get_one();

template <size_t Degree>
__device__ void poly_rounded_div(FQ<Degree> *Res, FQ<Degree> *A, FQ<Degree> *B, uint256 *mod);

template <size_t Degree>
__device__ void FQP_copy(FQ<Degree> *Res, FQ<Degree> *P);

template <size_t Degree>
__device__ void FQP_mul(FQ<Degree> *Res, FQ<Degree> *P1, FQ<Degree> *P2, uint256 *mod);

template <size_t Degree>
__device__ void FQP_inv(FQ<Degree> *Res, FQ<Degree> *P, uint256 *mod);

template <size_t Degree>
__device__ void FQP_div(FQ<Degree> *Res, FQ<Degree> *P1, FQ<Degree> *P2, uint256 *mod);

template <size_t Degree>
__device__ void FQP_neg(FQ<Degree> *Res, FQ<Degree> *P, uint256 *mod);

template <size_t Degree>
__device__ void FQP_pow(FQ<Degree> *Res, FQ<Degree> *P, uint256 *n, uint256 *mod);

template <size_t Degree>
__device__ void FQP_mul_scalar(FQ<Degree> *Res, FQ<Degree> *P, uint256 *n, uint256 *mod);

template <size_t Degree>
__device__ bool FQP_is_on_curve(FQ<Degree> *Px, FQ<Degree> *Py, uint256 *mod, FQ<Degree> *B);

template <size_t Degree>
__device__ bool FQP_is_valid(FQ<Degree> *P, uint256 *mod);

template <size_t Degree>
__device__ bool FQP_is_inf(FQ<Degree> *Px, FQ<Degree> *Py);

template <size_t Degree>
__device__ void FQP_ec_add(FQ<Degree> *ResX, FQ<Degree> *ResY, FQ<Degree> *Px, FQ<Degree> *Py, FQ<Degree> *Qx,
                           FQ<Degree> *Qy, uint256 *mod_fp);

template <size_t Degree>
__device__ void FQP_ec_mul(FQ<Degree> *ResX, FQ<Degree> *ResY, FQ<Degree> *Gx, FQ<Degree> *Gy, uint256 *n,
                           uint256 *mod_fp);

template <size_t Degree>
__device__ void FQP_linefunc(FQ<Degree> *Res, FQ<Degree> *P1x, FQ<Degree> *P1y, FQ<Degree> *P2x, FQ<Degree> *P2y,
                             FQ<Degree> *Tx, FQ<Degree> *Ty, uint256 *mod);

__device__ void FQP_twist(FQ<12> *Rx, FQ<12> *Ry, FQ<2> *Px, FQ<2> *Py, uint256 *mod_fp);

template <size_t Degree>
__device__ void FQP_final_exponentiation(FQ<Degree> *res, FQ<Degree> *p, uint256 *mod);

template <size_t Degree>
__device__ void miller_loop(FQ<Degree> *Result, FQ<Degree> *Qx, FQ<Degree> *Qy, FQ<Degree> *Px, FQ<Degree> *Py,
                            uint256 *mod_fp, uint256 *curve_order, uint256 *ate_loop_count, bool final_exp = true);

__device__ void pairing(FQ<12> *Res, FQ<2> *Qx, FQ<2> *Qy, FQ<1> *Px, FQ<1> *Py, uint256 *mod_fp, uint256 *curve_order,
                        uint256 *ate_loop_count, bool final_exp = true);

__device__ int pairing_multiple(EccConstants *ecc_constants_ptr, uint8_t *points_data, size_t data_len);

}  // namespace ecc

// #include "ecc_impl.cuh"
