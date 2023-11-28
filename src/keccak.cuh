#ifndef _KECCAK_H_
#define _KECCAK_H_

#include "utils.h"


#ifndef ROTL64
#define ROTL64(x, y) (((x) << (y)) | ((x) >> (64 - (y))))
#endif

namespace keccak {
    const uint64_t keccakf_rndc[24] = {
        0x0000000000000001, 0x0000000000008082, 0x800000000000808a,
        0x8000000080008000, 0x000000000000808b, 0x0000000080000001,
        0x8000000080008081, 0x8000000000008009, 0x000000000000008a,
        0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
        0x000000008000808b, 0x800000000000008b, 0x8000000000008089,
        0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
        0x000000000000800a, 0x800000008000000a, 0x8000000080008081,
        0x8000000000008080, 0x0000000080000001, 0x8000000080008008
    };
    const int keccakf_rotc[24] = {
        1,  3,  6,  10, 15, 21, 28, 36, 45, 55, 2,  14,
        27, 41, 56, 8,  25, 43, 62, 18, 39, 61, 20, 44
    };
    const int keccakf_piln[24] = {
        10, 7,  11, 17, 18, 3, 5,  16, 8,  21, 24, 4,
        15, 23, 19, 13, 12, 2, 20, 14, 22, 9,  6,  1
    };



    class keccak_t {
        public:

        static const uint32_t KECCAK_ROUNDS = 24;

        typedef struct alignas(32)
        {
            union
            {                   // state:
                uint8_t b[200]; // 8-bit bytes
                uint64_t q[25]; // 64-bit words
            } st;
            int pt, rsiz, mdlen; // these don't overflow
        } sha3_ctx_t;

        typedef struct {
            uint64_t *rndc; // shared
            int *rotc; //shared
            int *piln; //shared
            sha3_ctx_t *state; //specific to every instance
        } sha3_parameters_t;

        uint64_t *_rndc;
        int *_rotc;
        int *_piln;
        sha3_ctx_t *_content;

        __host__ __device__ keccak_t(uint64_t *rndc, int *rotc, int *piln, sha3_ctx_t *content) : _rndc(rndc), _rotc(rotc), _piln(piln), _content(content) {

        }

        __host__ __device__ void sha3_keccakf(uint64_t st[25]) {
            // variables
            int i, j, r;
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
            for (r = 0; r < KECCAK_ROUNDS; r++)
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
                    j = _piln[i];
                    bc[0] = st[j];
                    st[j] = ROTL64(t, _rotc[i]);
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
                st[0] ^= _rndc[r];
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
        
        __host__ __device__ void sha3_init(int mdlen) {
            uint32_t idx;
            for (idx = 0; idx < 25; idx++)
                _content->st.q[idx] = 0;
            _content->mdlen = mdlen;
            _content->rsiz = 200 - 2 * mdlen;
            _content->pt = 0;
        }

        __host__ __device__ void sha3_update(const uint8_t *data, size_t len) {
            size_t  idx;
            int j;
            j=_content->pt;
            for (idx = 0; idx < len; idx++) {
                _content->st.b[j++] ^= data[idx];
                if (j >= _content->rsiz) {
                    sha3_keccakf(_content->st.q);
                    j = 0;
                }
            }
            _content->pt = j;
        }

        __host__ __device__ void sha3_final(uint8_t *md) {
            int idx;
            _content->st.b[_content->pt] ^= 0x01; //why not _rndc[0]? 0x06 for sha3, 0x1F for shake, 0x01 for keccak
            _content->st.b[_content->rsiz - 1] ^= 0x80;
            sha3_keccakf(_content->st.q);
            for (idx = 0; idx < _content->mdlen; idx++)
                md[idx] = _content->st.b[idx];
        }

        __host__ __device__ void sha3(const uint8_t *in, size_t inlen, uint8_t *md, int mdlen) {
            sha3_init(mdlen);
            sha3_update(in, inlen);
            sha3_final(md);
        }
        // SHAKE128 and SHAKE256 extensible-output functions
        __host__ __device__ void shake_xof(uint8_t *md, int len) {
            _content->st.b[_content->pt] ^= 0x1F;
            _content->st.b[_content->rsiz - 1] ^= 0x80;
            sha3_keccakf(_content->st.q);
            _content->pt = 0;
        }

        __host__ __device__ void shake_out(uint8_t *out, size_t len) {
            size_t idx;
            int j;
            j = _content->pt;
            for (idx = 0; idx < len; idx++) {
                if (j >= _content->rsiz) {
                    sha3_keccakf(_content->st.q);
                    j = 0;
                }
                ((uint8_t *)out)[idx] = _content->st.b[j++];
            }
            _content->pt = j;
        }

        __host__ __device__ void shae128_init() {
            sha3_init(16);
        }

        __host__ __device__ void shake256_init() {
            sha3_init(32);
        }

        __host__ __device__ void shake_update(const uint8_t *in, size_t inlen) {
            sha3_update(in, inlen);
        }

        __host__ __device__ void sha3_256(const uint8_t *in, size_t inlen, uint8_t *md) {
            sha3(in, inlen, md, 32);
        }

        __host__ __device__ void sha3_512(const uint8_t *in, size_t inlen, uint8_t *md) {
            sha3(in, inlen, md, 64);
        }

        __host__ static sha3_parameters_t *get_cpu_instances(uint32_t count) {
            sha3_parameters_t *cpu_instances= (sha3_parameters_t *)malloc(sizeof(sha3_parameters_t) * count);
            uint64_t *rndc = (uint64_t *)malloc(sizeof(uint64_t) * 24);
            int *rotc = (int *)malloc(sizeof(int) * 24);
            int *piln = (int *)malloc(sizeof(int) * 24);
            memcpy(rndc, keccakf_rndc, sizeof(uint64_t) * 24);
            memcpy(rotc, keccakf_rotc, sizeof(int) * 24);
            memcpy(piln, keccakf_piln, sizeof(int) * 24);
            for (uint32_t idx = 0; idx < count; idx++) {
                cpu_instances[idx].rndc = rndc;
                cpu_instances[idx].rotc = rotc;
                cpu_instances[idx].piln = piln;
                cpu_instances[idx].state = (sha3_ctx_t *)malloc(sizeof(sha3_ctx_t));
            }
            return cpu_instances;
        }

        __host__ static void free_cpu_instances(sha3_parameters_t *cpu_instances, uint32_t count) {
            free(cpu_instances[0].rndc);
            free(cpu_instances[0].rotc);
            free(cpu_instances[0].piln);
            for (uint32_t idx = 0; idx < count; idx++) {
                free(cpu_instances[idx].state);
            }
            free(cpu_instances);
        }

        __host__ static sha3_parameters_t *get_gpu_instances(sha3_parameters_t *cpu_instances, uint32_t count) {
            sha3_parameters_t *gpu_instances, *tmp_cpu_instances;
            cudaMalloc((void **)&gpu_instances, sizeof(sha3_parameters_t) * count);
            tmp_cpu_instances = (sha3_parameters_t *)malloc(sizeof(sha3_parameters_t) * count);
            uint64_t *rndc;
            int *rotc;
            int *piln;
            sha3_ctx_t *state;
            cudaMalloc((void **)&rndc, sizeof(uint64_t) * 24);
            cudaMalloc((void **)&rotc, sizeof(int) * 24);
            cudaMalloc((void **)&piln, sizeof(int) * 24);
            cudaMemcpy(rndc, cpu_instances[0].rndc, sizeof(uint64_t) * 24, cudaMemcpyHostToDevice);
            cudaMemcpy(rotc, cpu_instances[0].rotc, sizeof(int) * 24, cudaMemcpyHostToDevice);
            cudaMemcpy(piln, cpu_instances[0].piln, sizeof(int) * 24, cudaMemcpyHostToDevice);
            for (uint32_t idx = 0; idx < count; idx++) {
                tmp_cpu_instances[idx].rndc = rndc;
                tmp_cpu_instances[idx].rotc = rotc;
                tmp_cpu_instances[idx].piln = piln;
                cudaMalloc((void **)&(state), sizeof(sha3_ctx_t));
                tmp_cpu_instances[idx].state = state;
                printf("allocating %d pointer:%p\n", idx, tmp_cpu_instances[idx].state);
            }
            // print the instances
            for (uint32_t idx = 0; idx < count; idx++) {
                printf("allocating %d pointer:%p\n", idx, tmp_cpu_instances[idx].state);
                printf("sha3_parameters->piln: %p\n", tmp_cpu_instances[idx].piln);
                printf("sha3_parameters->rotc: %p\n", tmp_cpu_instances[idx].rotc);
                printf("sha3_parameters->rndc: %p\n", tmp_cpu_instances[idx].rndc);

            }
            cudaMemcpy(gpu_instances, tmp_cpu_instances, sizeof(sha3_parameters_t) * count, cudaMemcpyHostToDevice);
            free(tmp_cpu_instances);
            return gpu_instances;
        }

        __host__ static void free_gpu_instances(sha3_parameters_t *gpu_instance, uint32_t count) {
            sha3_parameters_t *tmp_cpu_instances;
            tmp_cpu_instances = (sha3_parameters_t *)malloc(sizeof(sha3_parameters_t) * count);
            cudaMemcpy(tmp_cpu_instances, gpu_instance, sizeof(sha3_parameters_t) * count, cudaMemcpyDeviceToHost);
            cudaFree(tmp_cpu_instances[0].rndc);
            cudaFree(tmp_cpu_instances[0].rotc);
            cudaFree(tmp_cpu_instances[0].piln);
            for (uint32_t idx = 0; idx < count; idx++) {
                printf("freeing %d pointer:%p\n", idx, tmp_cpu_instances[idx].state);
                cudaFree(tmp_cpu_instances[idx].state);
            }
            cudaFree(gpu_instance);
            free(tmp_cpu_instances);
        }
    };
}
#endif