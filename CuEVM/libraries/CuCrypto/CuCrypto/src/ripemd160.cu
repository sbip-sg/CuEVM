#include <CuCrypto/ripemd160.cuh>

namespace CuCrypto {
    namespace ripemd160 {
        __host__ __device__ void compress(
            uint32_t* MDbuf,
            uint32_t* X)
        {
            uint32_t aa = MDbuf[0], bb = MDbuf[1], cc = MDbuf[2], dd = MDbuf[3], ee = MDbuf[4];
            uint32_t aaa = MDbuf[0], bbb = MDbuf[1], ccc = MDbuf[2], ddd = MDbuf[3], eee = MDbuf[4];

            /* round 1 */
            RIPEMD160_FF(aa, bb, cc, dd, ee, X[0], 11);
            RIPEMD160_FF(ee, aa, bb, cc, dd, X[1], 14);
            RIPEMD160_FF(dd, ee, aa, bb, cc, X[2], 15);
            RIPEMD160_FF(cc, dd, ee, aa, bb, X[3], 12);
            RIPEMD160_FF(bb, cc, dd, ee, aa, X[4], 5);
            RIPEMD160_FF(aa, bb, cc, dd, ee, X[5], 8);
            RIPEMD160_FF(ee, aa, bb, cc, dd, X[6], 7);
            RIPEMD160_FF(dd, ee, aa, bb, cc, X[7], 9);
            RIPEMD160_FF(cc, dd, ee, aa, bb, X[8], 11);
            RIPEMD160_FF(bb, cc, dd, ee, aa, X[9], 13);
            RIPEMD160_FF(aa, bb, cc, dd, ee, X[10], 14);
            RIPEMD160_FF(ee, aa, bb, cc, dd, X[11], 15);
            RIPEMD160_FF(dd, ee, aa, bb, cc, X[12], 6);
            RIPEMD160_FF(cc, dd, ee, aa, bb, X[13], 7);
            RIPEMD160_FF(bb, cc, dd, ee, aa, X[14], 9);
            RIPEMD160_FF(aa, bb, cc, dd, ee, X[15], 8);

            /* round 2 */
            RIPEMD160_GG(ee, aa, bb, cc, dd, X[7], 7);
            RIPEMD160_GG(dd, ee, aa, bb, cc, X[4], 6);
            RIPEMD160_GG(cc, dd, ee, aa, bb, X[13], 8);
            RIPEMD160_GG(bb, cc, dd, ee, aa, X[1], 13);
            RIPEMD160_GG(aa, bb, cc, dd, ee, X[10], 11);
            RIPEMD160_GG(ee, aa, bb, cc, dd, X[6], 9);
            RIPEMD160_GG(dd, ee, aa, bb, cc, X[15], 7);
            RIPEMD160_GG(cc, dd, ee, aa, bb, X[3], 15);
            RIPEMD160_GG(bb, cc, dd, ee, aa, X[12], 7);
            RIPEMD160_GG(aa, bb, cc, dd, ee, X[0], 12);
            RIPEMD160_GG(ee, aa, bb, cc, dd, X[9], 15);
            RIPEMD160_GG(dd, ee, aa, bb, cc, X[5], 9);
            RIPEMD160_GG(cc, dd, ee, aa, bb, X[2], 11);
            RIPEMD160_GG(bb, cc, dd, ee, aa, X[14], 7);
            RIPEMD160_GG(aa, bb, cc, dd, ee, X[11], 13);
            RIPEMD160_GG(ee, aa, bb, cc, dd, X[8], 12);

            /* round 3 */
            RIPEMD160_HH(dd, ee, aa, bb, cc, X[3], 11);
            RIPEMD160_HH(cc, dd, ee, aa, bb, X[10], 13);
            RIPEMD160_HH(bb, cc, dd, ee, aa, X[14], 6);
            RIPEMD160_HH(aa, bb, cc, dd, ee, X[4], 7);
            RIPEMD160_HH(ee, aa, bb, cc, dd, X[9], 14);
            RIPEMD160_HH(dd, ee, aa, bb, cc, X[15], 9);
            RIPEMD160_HH(cc, dd, ee, aa, bb, X[8], 13);
            RIPEMD160_HH(bb, cc, dd, ee, aa, X[1], 15);
            RIPEMD160_HH(aa, bb, cc, dd, ee, X[2], 14);
            RIPEMD160_HH(ee, aa, bb, cc, dd, X[7], 8);
            RIPEMD160_HH(dd, ee, aa, bb, cc, X[0], 13);
            RIPEMD160_HH(cc, dd, ee, aa, bb, X[6], 6);
            RIPEMD160_HH(bb, cc, dd, ee, aa, X[13], 5);
            RIPEMD160_HH(aa, bb, cc, dd, ee, X[11], 12);
            RIPEMD160_HH(ee, aa, bb, cc, dd, X[5], 7);
            RIPEMD160_HH(dd, ee, aa, bb, cc, X[12], 5);

            /* round 4 */
            RIPEMD160_II(cc, dd, ee, aa, bb, X[1], 11);
            RIPEMD160_II(bb, cc, dd, ee, aa, X[9], 12);
            RIPEMD160_II(aa, bb, cc, dd, ee, X[11], 14);
            RIPEMD160_II(ee, aa, bb, cc, dd, X[10], 15);
            RIPEMD160_II(dd, ee, aa, bb, cc, X[0], 14);
            RIPEMD160_II(cc, dd, ee, aa, bb, X[8], 15);
            RIPEMD160_II(bb, cc, dd, ee, aa, X[12], 9);
            RIPEMD160_II(aa, bb, cc, dd, ee, X[4], 8);
            RIPEMD160_II(ee, aa, bb, cc, dd, X[13], 9);
            RIPEMD160_II(dd, ee, aa, bb, cc, X[3], 14);
            RIPEMD160_II(cc, dd, ee, aa, bb, X[7], 5);
            RIPEMD160_II(bb, cc, dd, ee, aa, X[15], 6);
            RIPEMD160_II(aa, bb, cc, dd, ee, X[14], 8);
            RIPEMD160_II(ee, aa, bb, cc, dd, X[5], 6);
            RIPEMD160_II(dd, ee, aa, bb, cc, X[6], 5);
            RIPEMD160_II(cc, dd, ee, aa, bb, X[2], 12);

            /* round 5 */
            RIPEMD160_JJ(bb, cc, dd, ee, aa, X[4], 9);
            RIPEMD160_JJ(aa, bb, cc, dd, ee, X[0], 15);
            RIPEMD160_JJ(ee, aa, bb, cc, dd, X[5], 5);
            RIPEMD160_JJ(dd, ee, aa, bb, cc, X[9], 11);
            RIPEMD160_JJ(cc, dd, ee, aa, bb, X[7], 6);
            RIPEMD160_JJ(bb, cc, dd, ee, aa, X[12], 8);
            RIPEMD160_JJ(aa, bb, cc, dd, ee, X[2], 13);
            RIPEMD160_JJ(ee, aa, bb, cc, dd, X[10], 12);
            RIPEMD160_JJ(dd, ee, aa, bb, cc, X[14], 5);
            RIPEMD160_JJ(cc, dd, ee, aa, bb, X[1], 12);
            RIPEMD160_JJ(bb, cc, dd, ee, aa, X[3], 13);
            RIPEMD160_JJ(aa, bb, cc, dd, ee, X[8], 14);
            RIPEMD160_JJ(ee, aa, bb, cc, dd, X[11], 11);
            RIPEMD160_JJ(dd, ee, aa, bb, cc, X[6], 8);
            RIPEMD160_JJ(cc, dd, ee, aa, bb, X[15], 5);
            RIPEMD160_JJ(bb, cc, dd, ee, aa, X[13], 6);

            /* parallel round 1 */
            RIPEMD160_JJJ(aaa, bbb, ccc, ddd, eee, X[5], 8);
            RIPEMD160_JJJ(eee, aaa, bbb, ccc, ddd, X[14], 9);
            RIPEMD160_JJJ(ddd, eee, aaa, bbb, ccc, X[7], 9);
            RIPEMD160_JJJ(ccc, ddd, eee, aaa, bbb, X[0], 11);
            RIPEMD160_JJJ(bbb, ccc, ddd, eee, aaa, X[9], 13);
            RIPEMD160_JJJ(aaa, bbb, ccc, ddd, eee, X[2], 15);
            RIPEMD160_JJJ(eee, aaa, bbb, ccc, ddd, X[11], 15);
            RIPEMD160_JJJ(ddd, eee, aaa, bbb, ccc, X[4], 5);
            RIPEMD160_JJJ(ccc, ddd, eee, aaa, bbb, X[13], 7);
            RIPEMD160_JJJ(bbb, ccc, ddd, eee, aaa, X[6], 7);
            RIPEMD160_JJJ(aaa, bbb, ccc, ddd, eee, X[15], 8);
            RIPEMD160_JJJ(eee, aaa, bbb, ccc, ddd, X[8], 11);
            RIPEMD160_JJJ(ddd, eee, aaa, bbb, ccc, X[1], 14);
            RIPEMD160_JJJ(ccc, ddd, eee, aaa, bbb, X[10], 14);
            RIPEMD160_JJJ(bbb, ccc, ddd, eee, aaa, X[3], 12);
            RIPEMD160_JJJ(aaa, bbb, ccc, ddd, eee, X[12], 6);

            /* parallel round 2 */
            RIPEMD160_III(eee, aaa, bbb, ccc, ddd, X[6], 9);
            RIPEMD160_III(ddd, eee, aaa, bbb, ccc, X[11], 13);
            RIPEMD160_III(ccc, ddd, eee, aaa, bbb, X[3], 15);
            RIPEMD160_III(bbb, ccc, ddd, eee, aaa, X[7], 7);
            RIPEMD160_III(aaa, bbb, ccc, ddd, eee, X[0], 12);
            RIPEMD160_III(eee, aaa, bbb, ccc, ddd, X[13], 8);
            RIPEMD160_III(ddd, eee, aaa, bbb, ccc, X[5], 9);
            RIPEMD160_III(ccc, ddd, eee, aaa, bbb, X[10], 11);
            RIPEMD160_III(bbb, ccc, ddd, eee, aaa, X[14], 7);
            RIPEMD160_III(aaa, bbb, ccc, ddd, eee, X[15], 7);
            RIPEMD160_III(eee, aaa, bbb, ccc, ddd, X[8], 12);
            RIPEMD160_III(ddd, eee, aaa, bbb, ccc, X[12], 7);
            RIPEMD160_III(ccc, ddd, eee, aaa, bbb, X[4], 6);
            RIPEMD160_III(bbb, ccc, ddd, eee, aaa, X[9], 15);
            RIPEMD160_III(aaa, bbb, ccc, ddd, eee, X[1], 13);
            RIPEMD160_III(eee, aaa, bbb, ccc, ddd, X[2], 11);

            /* parallel round 3 */
            RIPEMD160_HHH(ddd, eee, aaa, bbb, ccc, X[15], 9);
            RIPEMD160_HHH(ccc, ddd, eee, aaa, bbb, X[5], 7);
            RIPEMD160_HHH(bbb, ccc, ddd, eee, aaa, X[1], 15);
            RIPEMD160_HHH(aaa, bbb, ccc, ddd, eee, X[3], 11);
            RIPEMD160_HHH(eee, aaa, bbb, ccc, ddd, X[7], 8);
            RIPEMD160_HHH(ddd, eee, aaa, bbb, ccc, X[14], 6);
            RIPEMD160_HHH(ccc, ddd, eee, aaa, bbb, X[6], 6);
            RIPEMD160_HHH(bbb, ccc, ddd, eee, aaa, X[9], 14);
            RIPEMD160_HHH(aaa, bbb, ccc, ddd, eee, X[11], 12);
            RIPEMD160_HHH(eee, aaa, bbb, ccc, ddd, X[8], 13);
            RIPEMD160_HHH(ddd, eee, aaa, bbb, ccc, X[12], 5);
            RIPEMD160_HHH(ccc, ddd, eee, aaa, bbb, X[2], 14);
            RIPEMD160_HHH(bbb, ccc, ddd, eee, aaa, X[10], 13);
            RIPEMD160_HHH(aaa, bbb, ccc, ddd, eee, X[0], 13);
            RIPEMD160_HHH(eee, aaa, bbb, ccc, ddd, X[4], 7);
            RIPEMD160_HHH(ddd, eee, aaa, bbb, ccc, X[13], 5);

            /* parallel round 4 */
            RIPEMD160_GGG(ccc, ddd, eee, aaa, bbb, X[8], 15);
            RIPEMD160_GGG(bbb, ccc, ddd, eee, aaa, X[6], 5);
            RIPEMD160_GGG(aaa, bbb, ccc, ddd, eee, X[4], 8);
            RIPEMD160_GGG(eee, aaa, bbb, ccc, ddd, X[1], 11);
            RIPEMD160_GGG(ddd, eee, aaa, bbb, ccc, X[3], 14);
            RIPEMD160_GGG(ccc, ddd, eee, aaa, bbb, X[11], 14);
            RIPEMD160_GGG(bbb, ccc, ddd, eee, aaa, X[15], 6);
            RIPEMD160_GGG(aaa, bbb, ccc, ddd, eee, X[0], 14);
            RIPEMD160_GGG(eee, aaa, bbb, ccc, ddd, X[5], 6);
            RIPEMD160_GGG(ddd, eee, aaa, bbb, ccc, X[12], 9);
            RIPEMD160_GGG(ccc, ddd, eee, aaa, bbb, X[2], 12);
            RIPEMD160_GGG(bbb, ccc, ddd, eee, aaa, X[13], 9);
            RIPEMD160_GGG(aaa, bbb, ccc, ddd, eee, X[9], 12);
            RIPEMD160_GGG(eee, aaa, bbb, ccc, ddd, X[7], 5);
            RIPEMD160_GGG(ddd, eee, aaa, bbb, ccc, X[10], 15);
            RIPEMD160_GGG(ccc, ddd, eee, aaa, bbb, X[14], 8);

            /* parallel round 5 */
            RIPEMD160_FFF(bbb, ccc, ddd, eee, aaa, X[12], 8);
            RIPEMD160_FFF(aaa, bbb, ccc, ddd, eee, X[15], 5);
            RIPEMD160_FFF(eee, aaa, bbb, ccc, ddd, X[10], 12);
            RIPEMD160_FFF(ddd, eee, aaa, bbb, ccc, X[4], 9);
            RIPEMD160_FFF(ccc, ddd, eee, aaa, bbb, X[1], 12);
            RIPEMD160_FFF(bbb, ccc, ddd, eee, aaa, X[5], 5);
            RIPEMD160_FFF(aaa, bbb, ccc, ddd, eee, X[8], 14);
            RIPEMD160_FFF(eee, aaa, bbb, ccc, ddd, X[7], 6);
            RIPEMD160_FFF(ddd, eee, aaa, bbb, ccc, X[6], 8);
            RIPEMD160_FFF(ccc, ddd, eee, aaa, bbb, X[2], 13);
            RIPEMD160_FFF(bbb, ccc, ddd, eee, aaa, X[13], 6);
            RIPEMD160_FFF(aaa, bbb, ccc, ddd, eee, X[14], 5);
            RIPEMD160_FFF(eee, aaa, bbb, ccc, ddd, X[0], 15);
            RIPEMD160_FFF(ddd, eee, aaa, bbb, ccc, X[3], 13);
            RIPEMD160_FFF(ccc, ddd, eee, aaa, bbb, X[9], 11);
            RIPEMD160_FFF(bbb, ccc, ddd, eee, aaa, X[11], 11);

            /* combine results */
            ddd += cc + MDbuf[1];
            MDbuf[1] = MDbuf[2] + dd + eee;
            MDbuf[2] = MDbuf[3] + ee + aaa;
            MDbuf[3] = MDbuf[4] + aa + bbb;
            MDbuf[4] = MDbuf[0] + bb + ccc;
            MDbuf[0] = ddd;
        }

        __host__ __device__ void ripemd160(
            const uint8_t* msg,
            uint32_t msg_len,
            uint8_t* hash)
        {
            uint32_t i;
            int j;
            uint32_t digest[5] = {0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476, 0xc3d2e1f0UL};

            for (i = 0; i < (msg_len >> 6); ++i) {
                uint32_t chunk[16];

                for (j = 0; j < 16; ++j) {
                    chunk[j] = (uint32_t)(*(msg++));
                    chunk[j] |= (uint32_t)(*(msg++)) << 8;
                    chunk[j] |= (uint32_t)(*(msg++)) << 16;
                    chunk[j] |= (uint32_t)(*(msg++)) << 24;
                }

                compress(digest, chunk);
            }

            // Last chunk
            {
                uint32_t chunk[16] = {0};

                for (i = 0; i < (msg_len & 63); ++i) {
                    chunk[i >> 2] ^= (uint32_t)*msg++ << ((i & 3) << 3);
                }

                chunk[(msg_len >> 2) & 15] ^= (uint32_t)1 << (8 * (msg_len & 3) + 7);

                if ((msg_len & 63) > 55) {
                    compress(digest, chunk);
                    memset(chunk, 0, 64);
                }

                chunk[14] = msg_len << 3;
                chunk[15] = (msg_len >> 29);
                compress(digest, chunk);
            }

            for (i = 0; i < 5; ++i) {
                *(hash++) = digest[i];
                *(hash++) = digest[i] >> 8;
                *(hash++) = digest[i] >> 16;
                *(hash++) = digest[i] >> 24;
            }
        }
    } // namespace ripemd160
} // namespace CuCrypto
