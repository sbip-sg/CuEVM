// Elliptic curve utilities using CGBN
#pragma once
#include <CuCrypto/keccak.cuh>
#include <CuEVM/core/evm_word.cuh>
#include <CuEVM/utils/ecc_constants.cuh>
#include <CuEVM/utils/evm_utils.cuh>
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
                      evm_word_t *Qy);

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
