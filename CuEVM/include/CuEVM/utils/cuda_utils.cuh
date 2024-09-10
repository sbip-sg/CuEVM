
#ifndef _CUDA_UTILS_H_
#define _CUDA_UTILS_H_

#include <cuda.h>
#include <CGBN/cgbn.h>

#ifdef __CUDA_ARCH__
#ifndef MULTIPLE_THREADS_PER_INSTANCE
#define MULTIPLE_THREADS_PER_INSTANCE
#endif
#endif
#ifdef MULTIPLE_THREADS_PER_INSTANCE
#define __ONE_THREAD_PER_INSTANCE(X)__ __syncthreads(); if (threadIdx.x == 0) { X } __syncthreads();
#define __ONE_GPU_THREAD_BEGIN__ __syncthreads(); if (threadIdx.x == 0) {
#define __ONE_GPU_THREAD_END__ } __syncthreads();
#define __SYNC_THREADS__ __syncthreads();
#define __SHARED_MEMORY__ __shared__
#else
#define __ONE_THREAD_PER_INSTANCE(X)__ X
#define __ONE_GPU_THREAD_BEGIN__
#define __ONE_GPU_THREAD_END__
#define __SYNC_THREADS__
#define __SHARED_MEMORY__
#endif

#ifdef __CUDA_ARCH__
#define CUEVM_MALLOC(ptr, type, count) ptr=(type *)std::malloc(count, sizeof(type))
#define CUEVM_FREE(ptr) std::free(ptr)
#else
#ifdef ONLY_CPU
#define CUEVM_MALLOC(ptr, type, count) ptr=(type *)std::malloc(count, sizeof(type))
#define CUEVM_FREE(ptr) std::free(ptr)
#else
#define CUEVM_MALLOC(ptr, type, count) cuda_check(cudaMallocManaged((void **)&ptr, count * sizeof(type)), "cudaMallocManaged", __FILE__, __LINE__)
#define CUEVM_FREE(ptr) cuda_check(cudaFree(ptr), "cudaFree", __FILE__, __LINE__)
#endif
#endif

#ifdef DEBUG
#define DEBUG_PRINT(fmt, args...) fprintf(stderr, "DEBUG: %s:%d:%s(): " fmt, \
  __FILE__, __LINE__, __func__, ##args)
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

void cuda_check(cudaError_t status, const char *action=NULL, const char *file=NULL, int32_t line=0);
void cgbn_check(cgbn_error_report_t *report, const char *file=NULL, int32_t line=0);

#endif