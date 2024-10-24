// Elliptic curve utilities using CGBN
#pragma once
#include <CuCrypto/keccak.cuh>
#include <CuEVM/core/evm_word.cuh>
#include <CuEVM/utils/arith.cuh>
#include <CuEVM/utils/ecc_constants.cuh>

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
    bn_t coeffs[Degree];
};

template <size_t Degree>
void print_fqp(env_t env, FQ<Degree> &P, const char *name);

__host__ __device__ void cgbn_mul_mod(env_t env, bn_t &res, bn_t &a, bn_t &b, bn_t &mod);

__host__ __device__ void cgbn_add_mod(env_t env, bn_t &res, bn_t &a, bn_t &b, bn_t &mod);

__host__ __device__ void cgbn_sub_mod(env_t env, bn_t &res, bn_t &a, bn_t &b, bn_t &mod);

__host__ __device__ void cgbn_div_mod(env_t env, bn_t &res, bn_t &a, bn_t &b, bn_t &mod);

__host__ __device__ bool is_on_cuve_simple(env_t env, bn_t &Px, bn_t &Py, bn_t &mod, uint32_t B);

template <size_t Degree>
__host__ __device__ bool FQP_equals(ArithEnv &arith, FQ<Degree> &P1, FQ<Degree> &P2);

__host__ __device__ int ec_add(ArithEnv &arith, Curve curve, bn_t &ResX, bn_t &ResY, bn_t &Px, bn_t &Py, bn_t &Qx,
                               bn_t &Qy);

__host__ __device__ int ec_mul(ArithEnv &arith, Curve curve, bn_t &ResX, bn_t &ResY, bn_t &Gx, bn_t &Gy, bn_t &n);

__host__ __device__ void convert_point_to_address(ArithEnv &arith, bn_t &address, bn_t &X, bn_t &Y);

__host__ __device__ int ec_recover(ArithEnv &arith, CuEVM::EccConstants *ecc_constants_ptr, signature_t &sig,
                                   bn_t &signer);

template <size_t Degree>
__host__ __device__ void getFQ12_from_cgbn_t(ArithEnv &arith, FQ<Degree> &res, bn_t (&coeffs)[Degree]);

template <size_t Degree>
__host__ __device__ void FQP_add(ArithEnv &arith, FQ<Degree> &Res, FQ<Degree> &P1, FQ<Degree> &P2, bn_t &mod);

template <size_t Degree>
__host__ __device__ void FQP_sub(ArithEnv &arith, FQ<Degree> &Res, FQ<Degree> &P1, FQ<Degree> &P2, bn_t &mod);

template <size_t Degree>
__host__ __device__ uint deg(ArithEnv &arith, const FQ<Degree> &P);

template <size_t Degree>
__host__ __device__ FQ<Degree> get_one(ArithEnv &arith);

template <size_t Degree>
__host__ __device__ void poly_rounded_div(ArithEnv &arith, FQ<Degree> &Res, FQ<Degree> &A, FQ<Degree> &B, bn_t &mod);

template <size_t Degree>
__host__ __device__ void FQP_copy(ArithEnv &arith, FQ<Degree> &Res, FQ<Degree> &P);

template <size_t Degree>
__host__ __device__ void FQP_mul(ArithEnv &arith, FQ<Degree> &Res, FQ<Degree> &P1, FQ<Degree> &P2, bn_t &mod);

template <size_t Degree>
__host__ __device__ void FQP_inv(ArithEnv &arith, FQ<Degree> &Res, FQ<Degree> &P, bn_t &mod);

template <size_t Degree>
__host__ __device__ void FQP_div(ArithEnv &arith, FQ<Degree> &Res, FQ<Degree> &P1, FQ<Degree> &P2, bn_t &mod);

template <size_t Degree>
__host__ __device__ void FQP_neg(ArithEnv &arith, FQ<Degree> &Res, FQ<Degree> &P, bn_t &mod);

template <size_t Degree>
__host__ __device__ void FQP_pow(ArithEnv &arith, FQ<Degree> &Res, FQ<Degree> &P, bn_t &n, bn_t &mod);

template <size_t Degree>
__host__ __device__ void FQP_mul_scalar(ArithEnv &arith, FQ<Degree> &Res, FQ<Degree> &P, bn_t &n, bn_t &mod);

template <size_t Degree>
__host__ __device__ bool FQP_is_on_curve(ArithEnv &arith, FQ<Degree> &Px, FQ<Degree> &Py, bn_t &mod, FQ<Degree> &B);

template <size_t Degree>
__host__ __device__ bool FQP_is_valid(ArithEnv &arith, FQ<Degree> &P, bn_t &mod);

template <size_t Degree>
__host__ __device__ bool FQP_is_inf(ArithEnv &arith, FQ<Degree> &Px, FQ<Degree> &Py);

template <size_t Degree>
__host__ __device__ void FQP_ec_add(ArithEnv &arith, FQ<Degree> &ResX, FQ<Degree> &ResY, FQ<Degree> &Px, FQ<Degree> &Py,
                                    FQ<Degree> &Qx, FQ<Degree> &Qy, bn_t &mod_fp);

template <size_t Degree>
__host__ __device__ void FQP_ec_mul(ArithEnv &arith, FQ<Degree> &ResX, FQ<Degree> &ResY, FQ<Degree> &Gx, FQ<Degree> &Gy,
                                    bn_t &n, bn_t &mod_fp);

template <size_t Degree>
__host__ __device__ void FQP_linefunc(ArithEnv &arith, FQ<Degree> &Res, FQ<Degree> &P1x, FQ<Degree> &P1y,
                                      FQ<Degree> &P2x, FQ<Degree> &P2y, FQ<Degree> &Tx, FQ<Degree> &Ty, bn_t &mod);

__host__ __device__ void FQP_twist(ArithEnv &arith, FQ<12> &Rx, FQ<12> &Ry, FQ<2> &Px, FQ<2> &Py, bn_t &mod_fp);

template <size_t Degree>
__host__ __device__ void FQP_final_exponentiation(ArithEnv &arith, EccConstants *constants, FQ<Degree> &res,
                                                  FQ<Degree> &p, bn_t &mod);

template <size_t Degree>
__host__ __device__ void miller_loop(ArithEnv &arith, FQ<Degree> &Result, FQ<Degree> &Qx, FQ<Degree> &Qy,
                                     FQ<Degree> &Px, FQ<Degree> &Py, bn_t &mod_fp, bn_t &curve_order,
                                     bn_t &ate_loop_count, bool final_exp = true);

__host__ __device__ void pairing(ArithEnv &arith, FQ<12> &Res, FQ<2> &Qx, FQ<2> &Qy, FQ<1> &Px, FQ<1> &Py, bn_t &mod_fp,
                                 bn_t &curve_order, bn_t &ate_loop_count, bool final_exp = true);

__host__ __device__ int pairing_multiple(ArithEnv &arith, EccConstants *ecc_constants_ptr, uint8_t *points_data,
                                         size_t data_len);

}  // namespace ecc

// #include "ecc_impl.cuh"
