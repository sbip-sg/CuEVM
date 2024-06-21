
#ifndef CUCRYPTO_BLAKE2_H
#define CUCRYPTO_BLAKE2_H

#include <CuCrypto/utils.cuh>
#include <cuda.h>
#include <stdio.h>
#include <stdint.h>

namespace CuCrypto {
    namespace blake2 {
        CONSTANT uint32_t BLAKE2B_BLOCKBYTES = 128;
        CONSTANT uint32_t BLAKE2B_OUTBYTES = 64;
        CONSTANT uint32_t BLAKE2B_KEYBYTES = 64;
        CONSTANT uint32_t BLAKE2B_SALTBYTES = 16;
        CONSTANT uint32_t BLAKE2B_PERSONALBYTES = 16;

        CONSTANT uint64_t BLAKE2B_IV[8] =
        {
            0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
            0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
            0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
            0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
        };

        CONSTANT uint8_t BLAKE2B_SIGMA[10][16] =
        {
            {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 } ,
            { 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 } ,
            { 11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4 } ,
            {  7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8 } ,
            {  9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13 } ,
            {  2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9 } ,
            { 12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11 } ,
            { 13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10 } ,
            {  6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5 } ,
            { 10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13 , 0 }
        };

        typedef struct blake2b_state__
        {
            uint64_t h[8];
            uint64_t t[2];
            uint64_t f[2];
            uint8_t  buf[BLAKE2B_BLOCKBYTES];
            size_t   buflen;
            size_t   outlen;
            uint8_t  last_node;
        } blake2b_state;

        #define BLAKE2_G(i,a,b,c,d,x,y)                   \
        do {                                      \
            a = a + b + x; \
            d = ROTR64((d ^ a), 32);                  \
            c = c + d;                              \
            b = ROTR64((b ^ c), 24);                  \
            a = a + b + y; \
            d = ROTR64((d ^ a), 16);                  \
            c = c + d;                              \
            b = ROTR64((b ^ c), 63);                  \
        } while(0)

        #define BLAKE2_ROUND(i, m)                   \
        do {                              \
            BLAKE2_G(0,v[ 0],v[ 4],v[ 8],v[12],m[BLAKE2B_SIGMA[i][0]], m[BLAKE2B_SIGMA[i][1]]); \
            BLAKE2_G(1,v[ 1],v[ 5],v[ 9],v[13],m[BLAKE2B_SIGMA[i][2]], m[BLAKE2B_SIGMA[i][3]]); \
            BLAKE2_G(2,v[ 2],v[ 6],v[10],v[14],m[BLAKE2B_SIGMA[i][4]], m[BLAKE2B_SIGMA[i][5]]); \
            BLAKE2_G(3,v[ 3],v[ 7],v[11],v[15],m[BLAKE2B_SIGMA[i][6]], m[BLAKE2B_SIGMA[i][7]]); \
            BLAKE2_G(4,v[ 0],v[ 5],v[10],v[15],m[BLAKE2B_SIGMA[i][8]], m[BLAKE2B_SIGMA[i][9]]); \
            BLAKE2_G(5,v[ 1],v[ 6],v[11],v[12],m[BLAKE2B_SIGMA[i][10]],m[BLAKE2B_SIGMA[i][11]]); \
            BLAKE2_G(6,v[ 2],v[ 7],v[ 8],v[13],m[BLAKE2B_SIGMA[i][12]],m[BLAKE2B_SIGMA[i][13]]); \
            BLAKE2_G(7,v[ 3],v[ 4],v[ 9],v[14],m[BLAKE2B_SIGMA[i][14]],m[BLAKE2B_SIGMA[i][15]]); \
        } while(0)

        __host__ __device__ void blake2_compress(
            uint64_t rounds,
            blake2b_state *S,
            const uint8_t block[BLAKE2B_BLOCKBYTES]);


        __host__ __device__ void blake2f(
            uint64_t rounds,
            uint64_t h[8],
            const uint64_t m[16],
            uint64_t t[2],
            int32_t f);

    } // namespace blake2
} // namespace CuCrypto


#endif