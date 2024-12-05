#include <CuEVM/state/account_flags.cuh>

namespace CuEVM {

__host__ __device__ void account_flags_t::print() const {
    __ONE_GPU_THREAD_WOSYNC_BEGIN__
    printf("Account flags: %08x\n", flags);
    __ONE_GPU_THREAD_WOSYNC_END__
}

__host__ char *account_flags_t::to_hex(char *hex) const {
    sprintf(hex, "%08x", flags);
    return hex;
}
}  // namespace CuEVM