#ifndef CUCRYPTO_UTILS_H
#define CUCRYPTO_UTILS_H

#ifndef ROTL64
#define ROTL64(x, y) (((x) << (y)) | ((x) >> (64 - (y))))
#endif

#ifndef ROTR64
#define ROTR64(x, y) (((x) >> (y)) | ((x) << (64 - (y))))
#endif

#ifndef ROTL32
#define ROTL32(x, y) (((x) << (y)) | ((x) >> (32 - (y))))
#endif

#ifndef ROTR32
#define ROTR32(x, y) (((x) >> (y)) | ((x) << (32 - (y))))
#endif

#ifdef __CUDA_ARCH__
#define CONSTANT __device__ __constant__ const
#else
#define CONSTANT const
#endif

#endif // CUCRYPTO_UTILS_H