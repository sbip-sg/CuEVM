// CuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation
// Programme) Author: Stefan-Dan Ciocirlan Date: 2024-09-15
// SPDX-License-Identifier: MIT

#pragma once


// CGBN parameters
#ifndef CGBN_TPI
#define CGBN_TPI 32
#endif

#define CGBN_IBP 2
#define SHARED_STACK_SIZE 128

#include <CGBN/cgbn.h>
#include <cuda.h>

#ifdef __CUDA_ARCH__
#ifndef MULTIPLE_THREADS_PER_INSTANCE
#define MULTIPLE_THREADS_PER_INSTANCE
#endif
#define THREADIDX threadIdx.x
#else
#define THREADIDX 0
#endif
#ifdef MULTIPLE_THREADS_PER_INSTANCE
#define __ONE_THREAD_PER_INSTANCE(X)   \
    if (threadIdx.x % CGBN_TPI == 0) { \
        X                              \
    }
#define __ONE_GPU_THREAD_BEGIN__ \
    __syncthreads();             \
    if (threadIdx.x % CGBN_TPI == 0) {
#define __ONE_GPU_THREAD_END__ \
    }                          \
    __syncthreads();
#define __ONE_GPU_THREAD_WOSYNC_BEGIN__ if (threadIdx.x % CGBN_TPI == 0) {
#define __ONE_GPU_THREAD_WOSYNC_END__ }
#define __SYNC_THREADS__ __syncthreads();
#define __SHARED_MEMORY__ __shared__
#else
#define __ONE_THREAD_PER_INSTANCE(X) X
#define __ONE_GPU_THREAD_BEGIN__
#define __ONE_GPU_THREAD_END__
#define __ONE_GPU_THREAD_WOSYNC_BEGIN__
#define __ONE_GPU_THREAD_WOSYNC_END__
#define __SYNC_THREADS__
#define __SHARED_MEMORY__
#endif

#ifdef DEBUG
#define DEBUG_PRINT(fmt, args...) fprintf(stderr, "DEBUG: %s:%d:%s(): " fmt, __FILE__, __LINE__, __func__, ##args)
#else
#define DEBUG_PRINT(fmt, args...) /* Don't do anything in release builds */
#endif

#ifdef __CUDA_ARCH__
#define CONSTANT __device__ __constant__ const
#else
#define CONSTANT const
#endif

#define CUDA_CHECK(action) cuda_check(action, #action, __FILE__, __LINE__)
#define CGBN_CHECK(report) cgbn_check(report, __FILE__, __LINE__)

#ifdef __CUDA_ARCH__
#define INSTANCE_BLK_IDX threadIdx.x / CGBN_TPI
#define INSTANCE_GLOBAL_IDX (threadIdx.x + blockIdx.x * blockDim.x) / CGBN_TPI
#define THREAD_IDX_PER_INSTANCE threadIdx.x % CGBN_TPI
#define INSTANCE_IDX_PER_BLOCK threadIdx.x / CGBN_TPI
#else
#define INSTANCE_BLK_IDX 0
#define INSTANCE_GLOBAL_IDX 0
#define THREAD_IDX_PER_INSTANCE 0
#define INSTANCE_IDX_PER_BLOCK 0
#endif

void cuda_check(cudaError_t status, const char *action = NULL, const char *file = NULL, int32_t line = 0);
void cgbn_check(cgbn_error_report_t *report, const char *file = NULL, int32_t line = 0);
