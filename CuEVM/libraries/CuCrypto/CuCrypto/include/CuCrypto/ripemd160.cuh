#ifndef CUCRYPTO_RIPEMD160_H
#define CUCRYPTO_RIPEMD160_H

#include <CuCrypto/utils.cuh>
#include <cuda.h>
#include <stdio.h>
#include <stdint.h>


namespace CuCrypto {
    namespace ripemd160 {

        #define RIPEMD160_F(x, y, z) ((x) ^ (y) ^ (z))
        #define RIPEMD160_G(x, y, z) (((x) & (y)) | (~(x) & (z)))
        #define RIPEMD160_H(x, y, z) (((x) | ~(y)) ^ (z))
        #define RIPEMD160_IQ(x, y, z) (((x) & (z)) | ((y) & ~(z)))
        #define RIPEMD160_J(x, y, z) ((x) ^ ((y) | ~(z)))

        #define RIPEMD160_FF(a, b, c, d, e, x, s)        \
            {                                  \
                (a) += RIPEMD160_F((b), (c), (d)) + (x); \
                (a) = ROTL32((a), (s)) + (e);     \
                (c) = ROTL32((c), 10);            \
            }
        #define RIPEMD160_GG(a, b, c, d, e, x, s)                       \
            {                                                 \
                (a) += RIPEMD160_G((b), (c), (d)) + (x) + 0x5a827999UL; \
                (a) = ROTL32((a), (s)) + (e);                    \
                (c) = ROTL32((c), 10);                           \
            }
        #define RIPEMD160_HH(a, b, c, d, e, x, s)                       \
            {                                                 \
                (a) += RIPEMD160_H((b), (c), (d)) + (x) + 0x6ed9eba1UL; \
                (a) = ROTL32((a), (s)) + (e);                    \
                (c) = ROTL32((c), 10);                           \
            }
        #define RIPEMD160_II(a, b, c, d, e, x, s)                        \
            {                                                  \
                (a) += RIPEMD160_IQ((b), (c), (d)) + (x) + 0x8f1bbcdcUL; \
                (a) = ROTL32((a), (s)) + (e);                     \
                (c) = ROTL32((c), 10);                            \
            }
        #define RIPEMD160_JJ(a, b, c, d, e, x, s)                       \
            {                                                 \
                (a) += RIPEMD160_J((b), (c), (d)) + (x) + 0xa953fd4eUL; \
                (a) = ROTL32((a), (s)) + (e);                    \
                (c) = ROTL32((c), 10);                           \
            }
        #define RIPEMD160_FFF(a, b, c, d, e, x, s)       \
            {                                  \
                (a) += RIPEMD160_F((b), (c), (d)) + (x); \
                (a) = ROTL32((a), (s)) + (e);     \
                (c) = ROTL32((c), 10);            \
            }
        #define RIPEMD160_GGG(a, b, c, d, e, x, s)                      \
            {                                                 \
                (a) += RIPEMD160_G((b), (c), (d)) + (x) + 0x7a6d76e9UL; \
                (a) = ROTL32((a), (s)) + (e);                    \
                (c) = ROTL32((c), 10);                           \
            }
        #define RIPEMD160_HHH(a, b, c, d, e, x, s)                      \
            {                                                 \
                (a) += RIPEMD160_H((b), (c), (d)) + (x) + 0x6d703ef3UL; \
                (a) = ROTL32((a), (s)) + (e);                    \
                (c) = ROTL32((c), 10);                           \
            }
        #define RIPEMD160_III(a, b, c, d, e, x, s)                       \
            {                                                  \
                (a) += RIPEMD160_IQ((b), (c), (d)) + (x) + 0x5c4dd124UL; \
                (a) = ROTL32((a), (s)) + (e);                     \
                (c) = ROTL32((c), 10);                            \
            }
        #define RIPEMD160_JJJ(a, b, c, d, e, x, s)                      \
            {                                                 \
                (a) += RIPEMD160_J((b), (c), (d)) + (x) + 0x50a28be6UL; \
                (a) = ROTL32((a), (s)) + (e);                    \
                (c) = ROTL32((c), 10);                           \
            }

        __host__ __device__ void compress(
            uint32_t* MDbuf,
            uint32_t* X);

        __host__ __device__ void ripemd160(
            const uint8_t* msg,
            uint32_t msg_len,
            uint8_t* hash);

    } // namespace ripemd160
} // namespace CuCrypto


#endif