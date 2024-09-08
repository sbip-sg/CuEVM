// CuCrypto - CUDA based Cryptography Library
// Copyright 2024 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2024-06-21
// SPDX-License-Identifier: MIT

#include <CuCrypto/keccak.cuh>

namespace CuCrypto {

    namespace keccak
    {
        __host__ __device__ void sha3_keccakf(
            uint64_t st[25])
        {
            // variables
            int32_t i, j, r;
            uint64_t t, bc[5];

#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
            uint8_t *v;

            // endianess conversion. this is redundant on little-endian targets
            for (i = 0; i < 25; i++)
            {
                v = (uint8_t *)&st[i];
                st[i] = ((uint64_t)v[0]) | (((uint64_t)v[1]) << 8) |
                        (((uint64_t)v[2]) << 16) | (((uint64_t)v[3]) << 24) |
                        (((uint64_t)v[4]) << 32) | (((uint64_t)v[5]) << 40) |
                        (((uint64_t)v[6]) << 48) | (((uint64_t)v[7]) << 56);
            }
#endif

            // actual iteration
            for (r = 0; r < ROUNDS; r++)
            {

                // Theta
                for (i = 0; i < 5; i++)
                    bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];

                for (i = 0; i < 5; i++)
                {
                    t = bc[(i + 4) % 5] ^ ROTL64(bc[(i + 1) % 5], 1);
                    for (j = 0; j < 25; j += 5)
                        st[j + i] ^= t;
                }

                // Rho Pi
                t = st[1];
                for (i = 0; i < 24; i++)
                {
                    j = PILN[i];
                    bc[0] = st[j];
                    st[j] = ROTL64(t, ROTC[i]);
                    t = bc[0];
                }

                //  Chi
                for (j = 0; j < 25; j += 5)
                {
                    for (i = 0; i < 5; i++)
                        bc[i] = st[j + i];
                    for (i = 0; i < 5; i++)
                        st[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
                }

                //  Iota
                st[0] ^= RNDC[r];
            }

#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
            // endianess conversion. this is redundant on little-endian targets
            for (i = 0; i < 25; i++)
            {
                v = (uint8_t *)&st[i];
                t = st[i];
                v[0] = t & 0xFF;
                v[1] = (t >> 8) & 0xFF;
                v[2] = (t >> 16) & 0xFF;
                v[3] = (t >> 24) & 0xFF;
                v[4] = (t >> 32) & 0xFF;
                v[5] = (t >> 40) & 0xFF;
                v[6] = (t >> 48) & 0xFF;
                v[7] = (t >> 56) & 0xFF;
            }
#endif
        }

        __host__ __device__ void sha3_init(
            keccak_ctx_t &state,
            int32_t mdlen)
        {
            uint32_t idx;
            for (idx = 0; idx < 25; idx++)
                state.st.q[idx] = 0;
            state.mdlen = mdlen;
            state.rsiz = 200 - 2 * mdlen;
            state.pt = 0;
        }

        __host__ __device__ void sha3_update(
            keccak_ctx_t &state,
            const uint8_t *data,
            size_t len)
        {
            size_t idx;
            int32_t j;
            j = state.pt;
            for (idx = 0; idx < len; idx++)
            {
                state.st.b[j++] ^= data[idx];
                if (j >= state.rsiz)
                {
                    sha3_keccakf(state.st.q);
                    j = 0;
                }
            }
            state.pt = j;
        }

        __host__ __device__ void sha3_final(
            keccak_ctx_t &state,
            uint8_t *md)
        {
            int32_t idx;
            // why not RNDC[0]? 0x06 for sha3, 0x1F for shake, 0x01 for keccak
            state.st.b[state.pt] ^= 0x01;
            state.st.b[state.rsiz - 1] ^= 0x80;
            sha3_keccakf(state.st.q);
            for (idx = 0; idx < state.mdlen; idx++)
                md[idx] = state.st.b[idx];
        }

        __host__ __device__ void nist_sha3_final(
            keccak_ctx_t &state,
            uint8_t *md)
        {
            int32_t idx;
            state.st.b[state.pt] ^= 0x06;
            state.st.b[state.rsiz - 1] ^= 0x80;
            sha3_keccakf(state.st.q);
            for (idx = 0; idx < state.mdlen; idx++)
                md[idx] = state.st.b[idx];
        }

        __host__ __device__ void sha3(
            const uint8_t *in,
            size_t inlen,
            uint8_t *md,
            int32_t mdlen)
        {
            keccak_ctx_t state;
            sha3_init(state, mdlen);
            sha3_update(state, in, inlen);
            sha3_final(state, md);
        }

        __host__ __device__ void nist_sha3(
            const uint8_t *in,
            size_t inlen,
            uint8_t *md,
            int32_t mdlen)
        {
            keccak_ctx_t state;
            sha3_init(state, mdlen);
            sha3_update(state, in, inlen);
            nist_sha3_final(state, md);
        }
        
        // SHAKE128 and SHAKE256 extensible-output functions
        __host__ __device__ void shake_xof(
            keccak_ctx_t &state,
            uint8_t *md,
            int32_t len)
        {
            state.st.b[state.pt] ^= 0x1F;
            state.st.b[state.rsiz - 1] ^= 0x80;
            sha3_keccakf(state.st.q);
            state.pt = 0;
        }

        __host__ __device__ void shake_out(
            keccak_ctx_t &state,
            uint8_t *out,
            size_t len)
        {
            size_t idx;
            int j;
            j = state.pt;
            for (idx = 0; idx < len; idx++)
            {
                if (j >= state.rsiz)
                {
                    sha3_keccakf(state.st.q);
                    j = 0;
                }
                ((uint8_t *)out)[idx] = state.st.b[j++];
            }
            state.pt = j;
        }

        __host__ __device__ void shae128_init(
            keccak_ctx_t &state)
        {
            sha3_init(state, 16);
        }

        __host__ __device__ void shake256_init(
            keccak_ctx_t &state)
        {
            sha3_init(state, 32);
        }

        __host__ __device__ void shake_update(
            keccak_ctx_t &state,
            const uint8_t *in,
            size_t inlen)
        {
            sha3_update(state, in, inlen);
        }

        __host__ __device__ void sha3_256(
            const uint8_t *in,
            size_t inlen,
            uint8_t *md)
        {
            sha3(in, inlen, md, 32);
        }

        __host__ __device__ void sha3_512(
            const uint8_t *in,
            size_t inlen,
            uint8_t *md)
        {
            sha3(in, inlen, md, 64);
        }
    } // namespace keccak
} // namespace CuCrypto
