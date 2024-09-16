#include <CuEVM/utils/cuda_utils.cuh>

// support routines
void cuda_check(cudaError_t status, const char *action, const char *file,
                int32_t line) {
    // check for cuda errors

    if (status != cudaSuccess) {
        printf("CUDA error occurred: %s\n", cudaGetErrorString(status));
        if (action != NULL)
            printf("While running %s   (file %s, line %d)\n", action, file,
                   line);
        exit(1);
    }
}

void cgbn_check(cgbn_error_report_t *report, const char *file, int32_t line) {
    // check for cgbn errors

    if (cgbn_error_report_check(report)) {
        printf("\n");
        printf("CGBN error occurred: %s\n", cgbn_error_string(report));

        if (report->_instance != 0xFFFFFFFF) {
            printf("Error reported by instance %d", report->_instance);
            if (report->_blockIdx.x != 0xFFFFFFFF ||
                report->_threadIdx.x != 0xFFFFFFFF)
                printf(", ");
            if (report->_blockIdx.x != 0xFFFFFFFF)
                printf("blockIdx=(%d, %d, %d) ", report->_blockIdx.x,
                       report->_blockIdx.y, report->_blockIdx.z);
            if (report->_threadIdx.x != 0xFFFFFFFF)
                printf("threadIdx=(%d, %d, %d)", report->_threadIdx.x,
                       report->_threadIdx.y, report->_threadIdx.z);
            printf("\n");
        } else {
            printf("Error reported by blockIdx=(%d %d %d)", report->_blockIdx.x,
                   report->_blockIdx.y, report->_blockIdx.z);
            printf("threadIdx=(%d %d %d)\n", report->_threadIdx.x,
                   report->_threadIdx.y, report->_threadIdx.z);
        }
        if (file != NULL) printf("file %s, line %d\n", file, line);
        exit(1);
    }
}