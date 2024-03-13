// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-03-13
// SPDX-License-Identifier: MIT
// Credits: https://github.com/amosnier/sha-2

#ifndef _SHA256_H_
#define _SHA256_H_

#include "include/utils.h"


#ifndef ROTR32
#define ROTR32(x, y) (((x) >> (y)) | ((x) << (32 - (y))))
#endif

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
#define TOTAL_LEN_LEN 8


namespace sha256
{
    /**
     * Initialize array of round constants:
     * (first 32 bits of the fractional parts of the cube roots
     * of the first 64 primes 2..311):
     */
    const uint32_t sha256_k[64] = {
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
    
    const uint32_t sha256_initial_hash[8] = {
        0x6a09e667,
        0xbb67ae85,
        0x3c6ef372,
        0xa54ff53a,
        0x510e527f,
        0x9b05688c,
        0x1f83d9ab,
        0x5be0cd19};

    class sha256_t
    {
    public:
        static const uint32_t SHA256_ROUNDS = 64;

        typedef struct alignas(32)
        {
            uint8_t chunk[SIZE_OF_SHA_256_CHUNK];
            uint8_t *chunk_pos;
            size_t space_left;
            uint64_t total_len;
            uint32_t h[8];
        } sha256_ctx_t;

        typedef struct
        {
            uint32_t *k;
            uint32_t *init_h;
        } sha256_parameters_t;

        sha256_parameters_t *_parameters;
        sha256_ctx_t *_content;

        __host__ __device__ __forceinline__ sha256_t(
            sha256_parameters_t *parameters) : _parameters(parameters)
        {
            SHARED_MEMORY sha256_ctx_t *content;
            ONE_THREAD_PER_INSTANCE(
                content = new sha256_ctx_t;)
            _content = content;
        }

        __host__ sha256_t()
        {
#ifndef ONLY_CPU
            CUDA_CHECK(cudaMallocManaged(
                (void **)&(_parameters),
                sizeof(sha256_parameters_t)));
            CUDA_CHECK(cudaMallocManaged(
                (void **)&(_parameters->k),
                sizeof(uint32_t) * 64));
            CUDA_CHECK(cudaMallocManaged(
                (void **)&(_parameters->init_h),
                sizeof(uint32_t) * 8));
#else
            _parameters = new sha256_parameters_t;
            _parameters->k = new uint32_t[64];
            _parameters->init_h = new uint32_t[8];
#endif
            memcpy(_parameters->k, &(sha256_k[0]), sizeof(uint32_t) * 64);
            memcpy(_parameters->init_h, &(sha256_initial_hash[0]), sizeof(uint32_t) * 8);

            _content = new sha256_ctx_t;
        }

        __host__ __device__ __forceinline__ ~sha256_t()
        {
            ONE_THREAD_PER_INSTANCE(
                delete _content;)
            _content = NULL;
            _parameters = NULL;
        }

        __host__ void free_parameters()
        {
#ifndef ONLY_CPU
            CUDA_CHECK(cudaFree(_parameters->k));
            CUDA_CHECK(cudaFree(_parameters->init_h));
            CUDA_CHECK(cudaFree(_parameters));
#else
            delete[] _parameters->k;
            delete[] _parameters->init_h;
            delete _parameters;
#endif
        }

        __host__ __device__ __forceinline__ void sha256_consume_chunk(
            uint32_t *h,
            const uint8_t *p)
        {
            unsigned i, j;
            uint32_t ah[8];

            /* Initialize working variables to current hash value: */
            for (i = 0; i < 8; i++)
                ah[i] = h[i];

            /*
            * The w-array is really w[64], but since we only need 16 of them at a time, we save stack by
            * calculating 16 at a time.
            *
            * This optimization was not there initially and the rest of the comments about w[64] are kept in their
            * initial state.
            */

            /*
            * create a 64-entry message schedule array w[0..63] of 32-bit words (The initial values in w[0..63]
            * don't matter, so many implementations zero them here) copy chunk into first 16 words w[0..15] of the
            * message schedule array
            */
            uint32_t w[16];

            /* Compression function main loop: */
            for (i = 0; i < 4; i++) {
                for (j = 0; j < 16; j++) {
                    if (i == 0) {
                        w[j] =
                            (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | (uint32_t)p[3];
                        p += 4;
                    } else {
                        /* Extend the first 16 words into the remaining 48 words w[16..63] of the
                        * message schedule array: */
                        const uint32_t s0 = ROTR32(w[(j + 1) & 0xf], 7) ^ ROTR32(w[(j + 1) & 0xf], 18) ^
                                    (w[(j + 1) & 0xf] >> 3);
                        const uint32_t s1 = ROTR32(w[(j + 14) & 0xf], 17) ^
                                    ROTR32(w[(j + 14) & 0xf], 19) ^ (w[(j + 14) & 0xf] >> 10);
                        w[j] = w[j] + s0 + w[(j + 9) & 0xf] + s1;
                    }
                    const uint32_t s1 = ROTR32(ah[4], 6) ^ ROTR32(ah[4], 11) ^ ROTR32(ah[4], 25);
                    const uint32_t ch = (ah[4] & ah[5]) ^ (~ah[4] & ah[6]);

                    const uint32_t temp1 = ah[7] + s1 + ch + _parameters->k[i << 4 | j] + w[j];
                    const uint32_t s0 = ROTR32(ah[0], 2) ^ ROTR32(ah[0], 13) ^ ROTR32(ah[0], 22);
                    const uint32_t maj = (ah[0] & ah[1]) ^ (ah[0] & ah[2]) ^ (ah[1] & ah[2]);
                    const uint32_t temp2 = s0 + maj;

                    ah[7] = ah[6];
                    ah[6] = ah[5];
                    ah[5] = ah[4];
                    ah[4] = ah[3] + temp1;
                    ah[3] = ah[2];
                    ah[2] = ah[1];
                    ah[1] = ah[0];
                    ah[0] = temp1 + temp2;
                }
            }

            /* Add the compressed chunk to the current hash value: */
            for (i = 0; i < 8; i++)
                h[i] += ah[i];
        }

        __host__ __device__ void sha256_init()
        {
            uint32_t idx;
            for (idx = 0; idx < SIZE_OF_SHA_256_CHUNK; idx++)
                _content->chunk[idx] = 0;
            _content->chunk_pos = _content->chunk;
            _content->space_left = SIZE_OF_SHA_256_CHUNK;
            _content->total_len = 0;
            for (idx = 0; idx < 8; idx++)
                _content->h[idx] = _parameters->init_h[idx];
        }

        __host__ __device__ void sha256_update(const uint8_t *data, size_t len)
        {
            _content->total_len += len;
            const uint8_t *p = data;

            while (len > 0) {
                if (
                    (_content->space_left == SIZE_OF_SHA_256_CHUNK) && 
                    (len >= SIZE_OF_SHA_256_CHUNK)
                ) {
                    sha256_consume_chunk(_content->h, p);
                    len -= SIZE_OF_SHA_256_CHUNK;
                    p += SIZE_OF_SHA_256_CHUNK;
                } else {
                    const size_t consumed_len = len < _content->space_left ? len : _content->space_left;
                    ONE_THREAD_PER_INSTANCE(
                        memcpy(_content->chunk_pos, p, consumed_len);)
                    _content->space_left -= consumed_len;
                    len -= consumed_len;
                    p += consumed_len;
                    if (_content->space_left == 0) {
                        sha256_consume_chunk(_content->h, _content->chunk);
                        _content->chunk_pos = _content->chunk;
                        _content->space_left = SIZE_OF_SHA_256_CHUNK;
                    } else {
                        _content->chunk_pos += consumed_len;
                    }

                }
            }
        }

        __host__ __device__ void sha256_final(uint8_t *md)
        {
            uint8_t *pos = _content->chunk_pos;
            size_t space_left = _content->space_left;
            uint32_t *const h = _content->h;

            *pos++ = 0x80;
            space_left--;

            if (space_left < TOTAL_LEN_LEN) {
                ONE_THREAD_PER_INSTANCE(
                    memset(pos, 0, space_left);)
                sha256_consume_chunk(h, _content->chunk);
                pos = _content->chunk;
                space_left = SIZE_OF_SHA_256_CHUNK;
            }

            const size_t left = space_left - TOTAL_LEN_LEN;
            ONE_THREAD_PER_INSTANCE(
                memset(pos, 0, left);)
            pos += left;
            uint64_t len = _content->total_len;
            pos[7] = (uint8_t)(len << 3);
            len >>= 5;
            int i;
            for (i = 6; i >= 0; i--) {
                pos[i] = (uint8_t)len;
                len >>= 8;
            }
            sha256_consume_chunk(h, _content->chunk);

            int j;
            for(i = 0, j = 0; i < 8; i++) {
                md[j++] = (uint8_t)(h[i] >> 24);
                md[j++] = (uint8_t)(h[i] >> 16);
                md[j++] = (uint8_t)(h[i] >> 8);
                md[j++] = (uint8_t)h[i];
            }
        }

        __host__ __device__ void sha(
            const uint8_t *in,
            size_t inlen,
            uint8_t *md)
        {
            sha256_init();
            sha256_update(in, inlen);
            sha256_final(md);
        }
    };
}

#endif