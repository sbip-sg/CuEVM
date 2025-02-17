#pragma once

#define INSTANCES_PER_BLOCK 128
// #define SHARED_STACK_SIZE 12

#include <cuda.h>

#include <CuEVM/utils/uint256.cuh>

#define __SHARED_MEMORY__ __shared__

#ifdef DEBUG
#define DEBUG_PRINT(fmt, args...) fprintf(stderr, "DEBUG: %s:%d:%s(): " fmt, __FILE__, __LINE__, __func__, ##args)
#else
#define DEBUG_PRINT(fmt, args...) /* Don't do anything in release builds */
#endif

#ifndef __CUDA_ARCH__
#undef CONSTANT
#define CONSTANT const
#else
#undef CONSTANT
#define CONSTANT __device__ __constant__ const
#endif

#define CUDA_CHECK(action) cuda_check(action, #action, __FILE__, __LINE__)
#define CGBN_CHECK(report) cgbn_check(report, __FILE__, __LINE__)

#ifdef __CUDA_ARCH__
#define THREADIDX threadIdx.x
#define INSTANCE_BLK_IDX threadIdx.x
#define INSTANCE_GLOBAL_IDX (threadIdx.x + blockIdx.x * blockDim.x)
#else
#define INSTANCE_BLK_IDX 0
#define INSTANCE_GLOBAL_IDX 0
#define THREADIDX 0
#endif

void cuda_check(cudaError_t status, const char *action = NULL, const char *file = NULL, int32_t line = 0);
