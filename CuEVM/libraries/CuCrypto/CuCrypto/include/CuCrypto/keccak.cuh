// CuCrypto - CUDA based Cryptography Library
// Copyright 2024 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2024-06-21
// SPDX-License-Identifier: MIT

#ifndef _KECCAK_H_
#define _KECCAK_H_

#include <CuCrypto/utils.cuh>
#include <cuda.h>
#include <stdio.h>
#include <stdint.h>

namespace CuCrypto {

    namespace keccak
    {
        CONSTANT_EXPR uint64_t RNDC[24] = {
            0x0000000000000001, 0x0000000000008082, 0x800000000000808a,
            0x8000000080008000, 0x000000000000808b, 0x0000000080000001,
            0x8000000080008081, 0x8000000000008009, 0x000000000000008a,
            0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
            0x000000008000808b, 0x800000000000008b, 0x8000000000008089,
            0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
            0x000000000000800a, 0x800000008000000a, 0x8000000080008081,
            0x8000000000008080, 0x0000000080000001, 0x8000000080008008};
        CONSTANT_EXPR int32_t ROTC[24] = {
            1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14,
            27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44};
        CONSTANT_EXPR int32_t PILN[24] = {
            10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4,
            15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1};
        CONSTANT uint32_t ROUNDS = 24;


        typedef struct alignas(32)
        {
            union
            {                   // state:
                uint8_t b[200]; // 8-bit bytes
                uint64_t q[25]; // 64-bit words
            } st;
            int32_t pt, rsiz, mdlen; // these don't overflow
        } keccak_ctx_t;

        __host__ __device__ void sha3_keccakf(
            uint64_t st[25]);

        __host__ __device__ void sha3_init(
            keccak_ctx_t &state,
            int32_t mdlen);

        __host__ __device__ void sha3_update(
            keccak_ctx_t &state,
            const uint8_t *data,
            size_t len);

        __host__ __device__ void sha3_final(
            keccak_ctx_t &state,
            uint8_t *md);

        __host__ __device__ void nist_sha3_final(
            keccak_ctx_t &state,
            uint8_t *md);

        __host__ __device__ void sha3(
            const uint8_t *in,
            size_t inlen,
            uint8_t *md,
            int32_t mdlen);

        __host__ __device__ void nist_sha3(
            const uint8_t *in,
            size_t inlen,
            uint8_t *md,
            int32_t mdlen);

        // SHAKE128 and SHAKE256 extensible-output functions
        __host__ __device__ void shake_xof(
            keccak_ctx_t &state,
            uint8_t *md,
            int32_t len);

        __host__ __device__ void shake_out(
            keccak_ctx_t &state,
            uint8_t *out,
            size_t len);

        __host__ __device__ void shae128_init(
            keccak_ctx_t &state);

        __host__ __device__ void shake256_init(
            keccak_ctx_t &state);

        __host__ __device__ void shake_update(
            keccak_ctx_t &state,
            const uint8_t *in,
            size_t inlen);

        __host__ __device__ void sha3_256(
            const uint8_t *in,
            size_t inlen,
            uint8_t *md);

        __host__ __device__ void sha3_512(
            const uint8_t *in,
            size_t inlen,
            uint8_t *md);
    } // namespace keccak
} // namespace CuCrypto

#endif