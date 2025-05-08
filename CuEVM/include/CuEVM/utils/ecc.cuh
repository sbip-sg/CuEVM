// Elliptic curve utilities using CGBN
#pragma once
#include <CuCrypto/keccak.cuh>
#include <CuEVM/core/evm_word.cuh>
#include <CuEVM/utils/evm_utils.cuh>

/**
 * The elliptic curve cryptography utilities.
 * Contains implementations for elliptic curve operations needed for EVM precompiled contracts:
 * - ecrecover: Recovers a public key from a signature
 * - ecadd: Adds two points on an elliptic curve
 * - ecmul: Multiplies a point on an elliptic curve by a scalar
 * - ecpairing: Performs pairing checks on the alt_bn128 curve
 */
namespace CuEVM {

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

/**
 * Curve parameters for elliptic curve operations.
 */
struct Curve {
    evm_word_t FieldPrime; /**< The field prime for the curve */
    evm_word_t Order;      /**< The order of the curve */
    evm_word_t GX;         /**< The x-coordinate of the generator point */
    evm_word_t GY;         /**< The y-coordinate of the generator point */
    uint32_t B = 3;        /**< The B parameter in the curve equation y^2 = x^3 + B */
};

/**
 * Constants for elliptic curve operations.
 */
struct EccConstants {
    Curve secp256k1;             /**< Constants for the secp256k1 curve */
    Curve alt_BN128;             /**< Constants for the alt_bn128 curve */
    evm_word_t alt_BN128_G2_B_0; /**< First coefficient of B for G2 in alt_bn128 */
    evm_word_t alt_BN128_G2_B_1; /**< Second coefficient of B for G2 in alt_bn128 */
    evm_word_t alt_BN128_G2X1;   /**< First x-coordinate coefficient for G2 */
    evm_word_t alt_BN128_G2X2;   /**< Second x-coordinate coefficient for G2 */
    evm_word_t alt_BN128_G2Y1;   /**< First y-coordinate coefficient for G2 */
    evm_word_t alt_BN128_G2Y2;   /**< Second y-coordinate coefficient for G2 */
    evm_word_t ate_loop_count;   /**< Loop count for the ate pairing */
    evm_word_t final_exp[11];    /**< Constants for the final exponentiation in pairing */

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

/**
 * Structure to hold a signature for the ecrecover operation.
 */
typedef struct {
    evm_word_t r;        /**< The r component of the signature */
    evm_word_t s;        /**< The s component of the signature */
    uint32_t v;          /**< The v component of the signature */
    evm_word_t msg_hash; /**< The hash of the message that was signed */
    evm_word_t res;      /**< The result of the operation */
    evm_word_t signer;   /**< The recovered signer address */
} signature_t;

// Reuse Curve struct definition from CuEVM namespace
using Curve = CuEVM::Curve;

/**
 * Structure for finite field elements of degree n.
 * Used for representing elements in extension fields GF(p^n).
 *
 * @tparam Degree The degree of the extension field
 */
template <size_t Degree>
struct FQ {
    evm_word_t coeffs[Degree]; /**< Coefficients of the finite field element */
};

/**
 * Print a finite field element for debugging.
 *
 * @tparam Degree The degree of the extension field
 * @param[in] P The finite field element to print
 * @param[in] name The name to display when printing
 */
template <size_t Degree>
__device__ void print_fqp(FQ<Degree> &P, const char *name);

/**
 * Compute the modular inverse of a 256-bit integer.
 *
 * @param[out] res The result of the modular inverse
 * @param[in] a The number to compute the inverse of
 * @param[in] mod The modulus
 */
__device__ void uint256_modular_inverse(uint256 *res, uint256 *a, uint256 *mod);

/**
 * Compute modular multiplication of two 256-bit integers.
 *
 * @param[out] res The result of the modular multiplication
 * @param[in] a The first operand
 * @param[in] b The second operand
 * @param[in] mod The modulus
 */
__device__ void uint256_mul_mod(uint256 *res, uint256 *a, uint256 *b, uint256 *mod);

/**
 * Compute modular addition of two 256-bit integers.
 *
 * @param[out] res The result of the modular addition
 * @param[in] a The first operand
 * @param[in] b The second operand
 * @param[in] mod The modulus
 */
__device__ void uint256_add_mod(uint256 *res, uint256 *a, uint256 *b, uint256 *mod);

/**
 * Compute modular subtraction of two 256-bit integers.
 *
 * @param[out] res The result of the modular subtraction
 * @param[in] a The first operand
 * @param[in] b The second operand
 * @param[in] mod The modulus
 */
__device__ void uint256_sub_mod(uint256 *res, uint256 *a, uint256 *b, uint256 *mod);

/**
 * Check if a point is on the curve with equation y^2 = x^3 + B.
 *
 * @param[in] Px The x-coordinate of the point
 * @param[in] Py The y-coordinate of the point
 * @param[in] mod The field modulus
 * @param[in] B The B parameter in the curve equation
 * @return true if the point is on the curve, false otherwise
 */
__device__ bool is_on_cuve_simple(uint256 *Px, uint256 *Py, uint256 *mod, uint32_t B);

/**
 * Check if two finite field elements are equal.
 *
 * @tparam Degree The degree of the extension field
 * @param[in] P1 The first finite field element
 * @param[in] P2 The second finite field element
 * @return true if equal, false otherwise
 */
template <size_t Degree>
__device__ bool FQP_equals(FQ<Degree> *P1, FQ<Degree> *P2);

/**
 * Add two points on an elliptic curve.
 *
 * @param[in] curve The curve parameters
 * @param[out] ResX The x-coordinate of the result
 * @param[out] ResY The y-coordinate of the result
 * @param[in] Px The x-coordinate of the first point
 * @param[in] Py The y-coordinate of the first point
 * @param[in] Qx The x-coordinate of the second point
 * @param[in] Qy The y-coordinate of the second point
 * @param[in] check_curve Whether to check if the points are on the curve
 * @return 0 for success, negative value for failure
 */
__device__ int ec_add(Curve curve, evm_word_t *ResX, evm_word_t *ResY, evm_word_t *Px, evm_word_t *Py, evm_word_t *Qx,
                      evm_word_t *Qy, bool check_curve = true);

/**
 * Multiply a point on an elliptic curve by a scalar.
 *
 * @param[in] curve The curve parameters
 * @param[out] ResX The x-coordinate of the result
 * @param[out] ResY The y-coordinate of the result
 * @param[in] Gx The x-coordinate of the point to multiply
 * @param[in] Gy The y-coordinate of the point to multiply
 * @param[in] n The scalar multiplier
 * @return 0 for success, negative value for failure
 */
__device__ int ec_mul(Curve curve, evm_word_t *ResX, evm_word_t *ResY, evm_word_t *Gx, evm_word_t *Gy, evm_word_t *n);

/**
 * Convert an elliptic curve point to an Ethereum address.
 *
 * @param[out] address The resulting Ethereum address
 * @param[in] X The x-coordinate of the point
 * @param[in] Y The y-coordinate of the point
 */
__device__ void convert_point_to_address(evm_word_t *address, evm_word_t *X, evm_word_t *Y);

/**
 * Recover the signer's public key from a signature.
 *
 * @param[in] ecc_constants_ptr Pointer to elliptic curve constants
 * @param[in] sig The signature structure containing r, s, v and message hash
 * @param[out] signer The recovered address of the signer
 * @return 0 for success, negative value for failure
 */
__device__ int ec_recover(CuEVM::EccConstants *ecc_constants_ptr, signature_t *sig, evm_word_t *signer);

/**
 * Add two finite field elements.
 *
 * @tparam Degree The degree of the extension field
 * @param[out] Res The result of addition
 * @param[in] P1 The first finite field element
 * @param[in] P2 The second finite field element
 * @param[in] mod The modulus
 */
template <size_t Degree>
__device__ void FQP_add(FQ<Degree> *Res, FQ<Degree> *P1, FQ<Degree> *P2, uint256 *mod);

/**
 * Subtract two finite field elements.
 *
 * @tparam Degree The degree of the extension field
 * @param[out] Res The result of subtraction
 * @param[in] P1 The first finite field element
 * @param[in] P2 The second finite field element
 * @param[in] mod The modulus
 */
template <size_t Degree>
__device__ void FQP_sub(FQ<Degree> *Res, FQ<Degree> *P1, FQ<Degree> *P2, uint256 *mod);

/**
 * Get the degree of a polynomial in a finite field.
 *
 * @tparam Degree The maximum degree of the polynomial
 * @param[in] P The polynomial as a finite field element
 * @return The highest non-zero coefficient index
 */
template <size_t Degree>
__device__ uint deg(FQ<Degree> *P);

/**
 * Get the multiplicative identity element (1) in a finite field.
 *
 * @tparam Degree The degree of the extension field
 * @return The identity element
 */
template <size_t Degree>
__device__ FQ<Degree> get_one();

/**
 * Polynomial division with rounding.
 *
 * @tparam Degree The degree of the extension field
 * @param[out] Res The result of division
 * @param[in] A The dividend
 * @param[in] B The divisor
 * @param[in] mod The modulus
 */
template <size_t Degree>
__device__ void poly_rounded_div(FQ<Degree> *Res, FQ<Degree> *A, FQ<Degree> *B, uint256 *mod);

/**
 * Copy a finite field element.
 *
 * @tparam Degree The degree of the extension field
 * @param[out] Res The destination
 * @param[in] P The source
 */
template <size_t Degree>
__device__ void FQP_copy(FQ<Degree> *Res, FQ<Degree> *P);

/**
 * Multiply two finite field elements.
 *
 * @tparam Degree The degree of the extension field
 * @param[out] Res The result of multiplication
 * @param[in] P1 The first finite field element
 * @param[in] P2 The second finite field element
 * @param[in] mod The modulus
 */
template <size_t Degree>
__device__ void FQP_mul(FQ<Degree> *Res, FQ<Degree> *P1, FQ<Degree> *P2, uint256 *mod);

/**
 * Compute the multiplicative inverse of a finite field element.
 *
 * @tparam Degree The degree of the extension field
 * @param[out] Res The inverse
 * @param[in] P The finite field element to invert
 * @param[in] mod The modulus
 */
template <size_t Degree>
__device__ void FQP_inv(FQ<Degree> *Res, FQ<Degree> *P, uint256 *mod);

/**
 * Divide two finite field elements.
 *
 * @tparam Degree The degree of the extension field
 * @param[out] Res The result of division
 * @param[in] P1 The dividend
 * @param[in] P2 The divisor
 * @param[in] mod The modulus
 */
template <size_t Degree>
__device__ void FQP_div(FQ<Degree> *Res, FQ<Degree> *P1, FQ<Degree> *P2, uint256 *mod);

/**
 * Negate a finite field element.
 *
 * @tparam Degree The degree of the extension field
 * @param[out] Res The negated element
 * @param[in] P The element to negate
 * @param[in] mod The modulus
 */
template <size_t Degree>
__device__ void FQP_neg(FQ<Degree> *Res, FQ<Degree> *P, uint256 *mod);

/**
 * Compute the modular exponentiation of a finite field element.
 *
 * @tparam Degree The degree of the extension field
 * @param[out] Res The result of exponentiation
 * @param[in] P The base
 * @param[in] n The exponent
 * @param[in] mod The modulus
 */
template <size_t Degree>
__device__ void FQP_pow(FQ<Degree> *Res, FQ<Degree> *P, uint256 *n, uint256 *mod);

/**
 * Multiply a finite field element by a scalar.
 *
 * @tparam Degree The degree of the extension field
 * @param[out] Res The result of multiplication
 * @param[in] P The finite field element
 * @param[in] n The scalar
 * @param[in] mod The modulus
 */
template <size_t Degree>
__device__ void FQP_mul_scalar(FQ<Degree> *Res, FQ<Degree> *P, uint256 *n, uint256 *mod);

/**
 * Check if a point is on an elliptic curve in a finite field.
 *
 * @tparam Degree The degree of the extension field
 * @param[in] Px The x-coordinate of the point
 * @param[in] Py The y-coordinate of the point
 * @param[in] mod The modulus
 * @param[in] B The B parameter of the curve
 * @return true if on curve, false otherwise
 */
template <size_t Degree>
__device__ bool FQP_is_on_curve(FQ<Degree> *Px, FQ<Degree> *Py, uint256 *mod, FQ<Degree> *B);

/**
 * Check if a finite field element is valid (within the field).
 *
 * @tparam Degree The degree of the extension field
 * @param[in] P The finite field element
 * @param[in] mod The modulus
 * @return true if valid, false otherwise
 */
template <size_t Degree>
__device__ bool FQP_is_valid(FQ<Degree> *P, uint256 *mod);

/**
 * Check if a point is the infinity point.
 *
 * @tparam Degree The degree of the extension field
 * @param[in] Px The x-coordinate of the point
 * @param[in] Py The y-coordinate of the point
 * @return true if infinity, false otherwise
 */
template <size_t Degree>
__device__ bool FQP_is_inf(FQ<Degree> *Px, FQ<Degree> *Py);

/**
 * Add two points on an elliptic curve in a finite field.
 *
 * @tparam Degree The degree of the extension field
 * @param[out] ResX The x-coordinate of the result
 * @param[out] ResY The y-coordinate of the result
 * @param[in] Px The x-coordinate of the first point
 * @param[in] Py The y-coordinate of the first point
 * @param[in] Qx The x-coordinate of the second point
 * @param[in] Qy The y-coordinate of the second point
 * @param[in] mod_fp The field modulus
 */
template <size_t Degree>
__device__ void FQP_ec_add(FQ<Degree> *ResX, FQ<Degree> *ResY, FQ<Degree> *Px, FQ<Degree> *Py, FQ<Degree> *Qx,
                           FQ<Degree> *Qy, uint256 *mod_fp);

/**
 * Multiply a point on an elliptic curve by a scalar in a finite field.
 *
 * @tparam Degree The degree of the extension field
 * @param[out] ResX The x-coordinate of the result
 * @param[out] ResY The y-coordinate of the result
 * @param[in] Gx The x-coordinate of the point
 * @param[in] Gy The y-coordinate of the point
 * @param[in] n The scalar
 * @param[in] mod_fp The field modulus
 */
template <size_t Degree>
__device__ void FQP_ec_mul(FQ<Degree> *ResX, FQ<Degree> *ResY, FQ<Degree> *Gx, FQ<Degree> *Gy, uint256 *n,
                           uint256 *mod_fp);

/**
 * Compute the line function for elliptic curve pairing.
 *
 * @tparam Degree The degree of the extension field
 * @param[out] Res The result of the line function
 * @param[in] P1x The x-coordinate of the first point
 * @param[in] P1y The y-coordinate of the first point
 * @param[in] P2x The x-coordinate of the second point
 * @param[in] P2y The y-coordinate of the second point
 * @param[in] Tx The x-coordinate of the evaluation point
 * @param[in] Ty The y-coordinate of the evaluation point
 * @param[in] mod The field modulus
 */
template <size_t Degree>
__device__ void FQP_linefunc(FQ<Degree> *Res, FQ<Degree> *P1x, FQ<Degree> *P1y, FQ<Degree> *P2x, FQ<Degree> *P2y,
                             FQ<Degree> *Tx, FQ<Degree> *Ty, uint256 *mod);

/**
 * Convert a point from the base field to a twisted curve representation.
 *
 * @param[out] Rx The twisted x-coordinate
 * @param[out] Ry The twisted y-coordinate
 * @param[in] Px The original x-coordinate
 * @param[in] Py The original y-coordinate
 * @param[in] mod_fp The field modulus
 */
__device__ void FQP_twist(FQ<12> *Rx, FQ<12> *Ry, FQ<2> *Px, FQ<2> *Py, uint256 *mod_fp);

/**
 * Perform the final exponentiation step in pairing computation.
 *
 * @tparam Degree The degree of the extension field
 * @param[out] res The result of the final exponentiation
 * @param[in] p The input value
 * @param[in] mod The field modulus
 */
template <size_t Degree>
__device__ void FQP_final_exponentiation(FQ<Degree> *res, FQ<Degree> *p, uint256 *mod);

/**
 * Perform the Miller loop algorithm for pairing computation.
 *
 * @tparam Degree The degree of the extension field
 * @param[out] Result The result of the Miller loop
 * @param[in] Qx The x-coordinate of the first point
 * @param[in] Qy The y-coordinate of the first point
 * @param[in] Px The x-coordinate of the second point
 * @param[in] Py The y-coordinate of the second point
 * @param[in] mod_fp The field modulus
 * @param[in] curve_order The curve order
 * @param[in] ate_loop_count The loop count for ate pairing
 * @param[in] final_exp Whether to perform final exponentiation
 */
template <size_t Degree>
__device__ void miller_loop(FQ<Degree> *Result, FQ<Degree> *Qx, FQ<Degree> *Qy, FQ<Degree> *Px, FQ<Degree> *Py,
                            uint256 *mod_fp, uint256 *curve_order, uint256 *ate_loop_count, bool final_exp = true);

/**
 * Perform a bilinear pairing of two points.
 *
 * @param[out] Res The result of the pairing
 * @param[in] Qx The x-coordinate of the first point in the extension field
 * @param[in] Qy The y-coordinate of the first point in the extension field
 * @param[in] Px The x-coordinate of the second point in the base field
 * @param[in] Py The y-coordinate of the second point in the base field
 * @param[in] mod_fp The field modulus
 * @param[in] curve_order The curve order
 * @param[in] ate_loop_count The loop count for ate pairing
 * @param[in] final_exp Whether to perform final exponentiation
 */
__device__ void pairing(FQ<12> *Res, FQ<2> *Qx, FQ<2> *Qy, FQ<1> *Px, FQ<1> *Py, uint256 *mod_fp, uint256 *curve_order,
                        uint256 *ate_loop_count, bool final_exp = true);

/**
 * Perform pairing checks on multiple points on the alt_bn128 curve.
 * Used for the ecpairing precompiled contract.
 *
 * @param[in] ecc_constants_ptr Pointer to elliptic curve constants
 * @param[in] points_data Array of points data in the format of (Px, Py, Qx, Qy)
 * @param[in] data_len Length of the points_data in bytes
 * @return 1 if the pairing succeeds, 0 if it fails, -1 if inputs are invalid
 */
__host__ __device__ int pairing_multiple(EccConstants *ecc_constants_ptr, uint8_t *points_data, size_t data_len);

}  // namespace ecc

// #include "ecc_impl.cuh"
