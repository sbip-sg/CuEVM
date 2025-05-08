#include <CuEVM/utils/cuda_utils.cuh>

// support routines
void cuda_check(cudaError_t status, const char *action, const char *file, int32_t line) {
    // check for cuda errors

    if (status != cudaSuccess) {
        printf("CUDA error occurred: %s\n", cudaGetErrorString(status));
        if (action != NULL) printf("While running %s   (file %s, line %d)\n", action, file, line);
        exit(1);
    }
}
