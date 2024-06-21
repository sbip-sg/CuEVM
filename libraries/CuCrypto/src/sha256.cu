// CuCrypto SHA-256 implementation
// Copyright 2024 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2024-06-21
// SPDX-License-Identifier: MIT
// Credits: https://github.com/amosnier/sha-2
#include <CuCrypto/sha256.cuh>

namespace CuCrypto {
    namespace sha256 {
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

                    const uint32_t temp1 = ah[7] + s1 + ch + SHA256_K[i << 4 | j] + w[j];
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

        __host__ __device__ void sha256_init(
            sha256_ctx_t &state)
        {
            uint32_t idx;
            for (idx = 0; idx < SIZE_OF_SHA_256_CHUNK; idx++)
                state.chunk[idx] = 0;
            state.chunk_pos = state.chunk;
            state.space_left = SIZE_OF_SHA_256_CHUNK;
            state.total_len = 0;
            for (idx = 0; idx < 8; idx++)
                state.h[idx] = SHA256_INITIAL_HASH[idx];
        }

        __host__ __device__ void sha256_update(
            sha256_ctx_t &state,
            const uint8_t *data,
            size_t len)
        {
            state.total_len += len;
            const uint8_t *p = data;

            while (len > 0) {
                if (
                    (state.space_left == SIZE_OF_SHA_256_CHUNK) && 
                    (len >= SIZE_OF_SHA_256_CHUNK)
                ) {
                    sha256_consume_chunk(state.h, p);
                    len -= SIZE_OF_SHA_256_CHUNK;
                    p += SIZE_OF_SHA_256_CHUNK;
                } else {
                    const size_t consumed_len = len < state.space_left ? len : state.space_left;
                    memcpy(state.chunk_pos, p, consumed_len);
                    state.space_left -= consumed_len;
                    len -= consumed_len;
                    p += consumed_len;
                    if (state.space_left == 0) {
                        sha256_consume_chunk(state.h, state.chunk);
                        state.chunk_pos = state.chunk;
                        state.space_left = SIZE_OF_SHA_256_CHUNK;
                    } else {
                        state.chunk_pos += consumed_len;
                    }

                }
            }
        }

        __host__ __device__ void sha256_final(
            sha256_ctx_t &state,
            uint8_t *md)
        {
            uint8_t *pos = state.chunk_pos;
            size_t space_left = state.space_left;
            uint32_t *const h = state.h;

            *pos++ = 0x80;
            space_left--;

            if (space_left < SHA256_TOTAL_LEN_LEN) {
                memset(pos, 0, space_left);
                sha256_consume_chunk(h, state.chunk);
                pos = state.chunk;
                space_left = SIZE_OF_SHA_256_CHUNK;
            }

            const size_t left = space_left - SHA256_TOTAL_LEN_LEN;
            memset(pos, 0, left);
            pos += left;
            uint64_t len = state.total_len;
            pos[7] = (uint8_t)(len << 3);
            len >>= 5;
            int i;
            for (i = 6; i >= 0; i--) {
                pos[i] = (uint8_t)len;
                len >>= 8;
            }
            sha256_consume_chunk(h, state.chunk);

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
            sha256_ctx_t state;
            sha256_init(state);
            sha256_update(state, in, inlen);
            sha256_final(state, md);
        }
    } // namespace sha256
} // namespace CuCrypto
