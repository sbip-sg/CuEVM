#ifndef CGBN_UTILS_CUH
#define CGBN_UTILS_CUH

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

/* basic types */
typedef enum {
  cgbn_no_error=0,
  cgbn_unsupported_threads_per_instance=1,
  cgbn_unsupported_size=2,
  cgbn_unsupported_limbs_per_thread=3,
  cgbn_unsupported_operation=4,
  cgbn_threads_per_block_mismatch=5,
  cgbn_threads_per_instance_mismatch=6,
  cgbn_division_by_zero_error=7,
  cgbn_division_overflow_error=8,
  cgbn_invalid_montgomery_modulus_error=9,
  cgbn_modulus_not_odd_error=10,
  cgbn_inverse_does_not_exist_error=11,
} cgbn_error_t;

typedef struct {
  volatile cgbn_error_t _error;
  uint32_t              _instance;
  dim3                  _threadIdx;
  dim3                  _blockIdx;
} cgbn_error_report_t;

typedef enum {
  cgbn_no_checks,       /* disable error checking - improves performance */
  cgbn_report_monitor,  /* writes errors to the reporter object, no other actions */
  cgbn_print_monitor,   /* writes errors to the reporter and prints the error to stdout */
  cgbn_halt_monitor,    /* writes errors to the reporter and halts */
} cgbn_monitor_t;

// default tpi is 32
const uint32_t cgbn_default_tpi=32;
// default bits is 256
const uint32_t cgbn_default_bits=256;

template<uint32_t bits>
struct cgbn_mem_t {
  public:
  uint32_t _limbs[(bits+31)/32];
};

cudaError_t cgbn_error_report_alloc(cgbn_error_report_t **report);
cudaError_t cgbn_error_report_free(cgbn_error_report_t *report);
bool        cgbn_error_report_check(cgbn_error_report_t *report);
void        cgbn_error_report_reset(cgbn_error_report_t *report);
const char *cgbn_error_string(cgbn_error_report_t *report);


#endif