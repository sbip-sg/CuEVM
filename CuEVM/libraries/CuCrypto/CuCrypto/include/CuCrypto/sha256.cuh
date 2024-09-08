// CuCrypto SHA-256
// Copyright 2024 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2024-06-21
// SPDX-License-Identifier: MIT
// Credits: https://github.com/amosnier/sha-2

#ifndef _SHA256_H_
#define _SHA256_H_

#include <CuCrypto/utils.cuh>
#include <cuda.h>
#include <stdio.h>
#include <stdint.h>

namespace CuCrypto {
    namespace sha256 {
        /*
        * @brief Size of the SHA-256 sum. This times eight is 256 bits.
        */
        #define SIZE_OF_SHA_256_HASH 32

        /*
        * @brief Size of the chunks used for the calculations.
        *
        * @note This should mostly be ignored by the user, although when using the streaming API, it has an impact for
        * performance. Add chunks whose size is a multiple of this, and you will avoid a lot of superfluous copying in RAM!
        */
        #define SIZE_OF_SHA_256_CHUNK 64
        #define SHA256_TOTAL_LEN_LEN 8
        /**
         * Initialize array of round constants:
         * (first 32 bits of the fractional parts of the cube roots
         * of the first 64 primes 2..311):
         */
        CONSTANT uint32_t SHA256_K[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
            0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
            0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
            0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
            0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
            0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
            0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
            0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
        
        CONSTANT uint32_t SHA256_INITIAL_HASH[8] = {
            0x6a09e667,
            0xbb67ae85,
            0x3c6ef372,
            0xa54ff53a,
            0x510e527f,
            0x9b05688c,
            0x1f83d9ab,
            0x5be0cd19};
        
        CONSTANT uint32_t ROUNDS = 64;

        typedef struct
        {
            uint8_t chunk[SIZE_OF_SHA_256_CHUNK];
            uint8_t *chunk_pos;
            size_t space_left;
            uint64_t total_len;
            uint32_t h[8];
        } sha256_ctx_t;

        __host__ __device__ __forceinline__ void sha256_consume_chunk(
            uint32_t *h,
            const uint8_t *p);

        __host__ __device__ void sha256_init(
            sha256_ctx_t &state);

        __host__ __device__ void sha256_update(
            sha256_ctx_t &state,
            const uint8_t *data,
            size_t len);

        __host__ __device__ void sha256_final(
            sha256_ctx_t &state,
            uint8_t *md);

        __host__ __device__ void sha(
            const uint8_t *in,
            size_t inlen,
            uint8_t *md);
    } // namespace sha256
} // namespace CuCrypto


#endif