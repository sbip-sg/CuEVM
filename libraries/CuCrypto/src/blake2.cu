
#include <CuCrypto/blake2.cuh>

namespace CuCrypto {
    namespace blake2 {
        __host__ __device__ inline uint64_t load64(
            const void *src)
        {
            #if defined(NATIVE_LITTLE_ENDIAN)
            uint64_t w;
            memcpy(&w, src, sizeof w);
            return w;
            #else
            const uint8_t *p = ( const uint8_t * )src;
            return (( uint64_t )( p[0] ) <<  0) |
                    (( uint64_t )( p[1] ) <<  8) |
                    (( uint64_t )( p[2] ) << 16) |
                    (( uint64_t )( p[3] ) << 24) |
                    (( uint64_t )( p[4] ) << 32) |
                    (( uint64_t )( p[5] ) << 40) |
                    (( uint64_t )( p[6] ) << 48) |
                    (( uint64_t )( p[7] ) << 56) ;
            #endif
        }

        __host__ __device__ inline void store64(
            void *dst,
            uint64_t w)
        {
            memcpy(dst, &w, sizeof w);
        }

        __host__ __device__ void blake2_compress(
            uint64_t rounds,
            blake2b_state *S,
            const uint8_t block[BLAKE2B_BLOCKBYTES])
        {
            uint64_t m[16];
            uint64_t v[16];
            size_t i;

            for( i = 0; i < 16; ++i ) {
                m[i] = load64( block + i * sizeof( m[i] ) );
            }

            for( i = 0; i < 8; ++i ) {
                v[i] = S->h[i];
            }

            v[ 8] = BLAKE2B_IV[0];
            v[ 9] = BLAKE2B_IV[1];
            v[10] = BLAKE2B_IV[2];
            v[11] = BLAKE2B_IV[3];
            v[12] = BLAKE2B_IV[4] ^ S->t[0];
            v[13] = BLAKE2B_IV[5] ^ S->t[1];
            v[14] = BLAKE2B_IV[6] ^ S->f[0];
            v[15] = BLAKE2B_IV[7] ^ S->f[1];

            for (uint64_t i = 0; i < rounds; i++){
                uint64_t r = (i % 10);
                BLAKE2_ROUND( r, m );
            }

            for( i = 0; i < 8; ++i ) {
                S->h[i] = S->h[i] ^ v[i] ^ v[i + 8];
            }
        }


        __host__ __device__ void blake2f(
            uint64_t rounds,
            uint64_t h[8],
            const uint64_t m[16],
            uint64_t t[2],
            int32_t f)
        {
            blake2b_state S;
            memset(&S, 0, sizeof(S));

            // Manual state initialization
            for (int32_t i = 0; i < 8; i++) {
                S.h[i] = h[i];
            }
            S.t[0] = t[0];
            S.t[1] = t[1];
            S.f[0] = f ? (uint64_t)-1 : 0; // Final block flag

            uint8_t block[BLAKE2B_BLOCKBYTES];
            for (int32_t i = 0; i < 16; i++) {
                store64(block + sizeof(uint64_t) * i, m[i]);
            }

            // Perform the compression
            blake2_compress(rounds, &S, block);

            // Update the input state with the new state
            for (int32_t i = 0; i < 8; i++) {
                h[i] = S.h[i];
            }
        }

    } // namespace blake2
} // namespace CuCrypto