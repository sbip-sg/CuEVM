#ifndef _CUEVM_SYSTEMS_OP_H_
#define _CUEVM_SYSTEMS_OP_H_

#include <CuEVM/core/evm_call_context.cuh>
#include <CuEVM/core/memory.cuh>
#include <CuEVM/core/memory_pool.cuh>
#include <CuEVM/core/stack.cuh>
#include <CuEVM/state/state_db.cuh>

/**
 * The system operations class.
 * It contains the implementation of the system operations.
 * - 00: STOP - Halts execution
 * - f0-ff: System operations:
 *      - CREATE(f0) - Creates a new account with associated code
 *      - CALL(f1) - Message-call into an account
 *      - CALLCODE(f2) - Message-call into this account with an alternative account's code
 *      - RETURN(f3) - Halts execution returning output data
 *      - DELEGATECALL(f4) - Message-call into this account with an alternative account's code, but persisting the
 * current values for sender and value
 *      - CREATE2(f5) - Creates a new account with associated code at a predictable address
 *      - STATICCALL(fa) - Static message-call into an account
 *      - REVERT(fd) - Halt execution reverting state changes but returning data and remaining gas
 *      - INVALID(fe) - Designated invalid instruction
 *      - SELFDESTRUCT(ff) - Halt execution and register account for later deletion
 */
namespace CuEVM::operations {
/**
 * The STOP operation.
 * Halts execution.
 *
 * @param[out] call_state_ptr The call state pointer.
 * @return The error code. 0 if no error.
 */
__device__ int32_t STOP(CuEVM::evm_call_context_t *call_state_ptr);

/**
 * The CREATE operation.
 * Creates a new account with associated code and gives the new evm call state.
 * Takes value, memory offset, and memory length from the stack.
 * Value is transferred from the current account to the new account.
 * The code is the data in memory from the given offset and length.
 * The created account's address is computed based on the sender's address and nonce.
 *
 * @param[in] current_context The current call context.
 * @param[out] new_context_ptr The new call context pointer.
 * @param[in] cached_state The cached call context.
 * @return The error code. 0 if the operation is successful.
 */
__device__ int32_t CREATE(CuEVM::evm_call_context_t *current_context, CuEVM::evm_call_context_t *&new_context_ptr,
                          CuEVM::cached_evm_call_context &cached_state);

/**
 * The CALL operation.
 * Message-call into an account and gives the new evm call state.
 * Takes gas, address, value, args offset, args size, ret offset, ret size from the stack.
 * Value is transferred from the current account to the called account.
 * The arguments are taken from memory at the given offset and size.
 * The return data is written to memory at the given return offset and size.
 *
 * @param[in] current_context The current call context.
 * @param[out] new_context_ptr The new call context pointer.
 * @param[in] cached_state The cached call context.
 * @return The error code. 0 if the operation is successful.
 */
__device__ int32_t CALL(CuEVM::evm_call_context_t *current_context, CuEVM::evm_call_context_t *&new_context_ptr,
                        CuEVM::cached_evm_call_context &cached_state);

/**
 * The CALLCODE operation.
 * Message-call into this account with an alternative account's code and gives the new evm call state.
 * Takes gas, address, value, args offset, args size, ret offset, ret size from the stack.
 * Similar to CALL but the message sender remains the current contract and value is not transferred.
 * Only the code is taken from the called account.
 *
 * @param[in] current_context The current call context.
 * @param[out] new_context_ptr The new call context pointer.
 * @param[in] cached_state The cached call context.
 * @return The error code. 0 if the operation is successful.
 */
__device__ int32_t CALLCODE(CuEVM::evm_call_context_t *current_context, CuEVM::evm_call_context_t *&new_context_ptr,
                            CuEVM::cached_evm_call_context &cached_state);

/**
 * The RETURN operation.
 * Halts execution returning output data.
 * Takes offset and length from the stack and copies the data from memory
 * as the return data.
 *
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[in] stack The stack.
 * @param[in] call_context The call context containing memory and other execution state.
 * @return The error code. 0 if the operation is successful.
 */
__device__ int32_t RETURN(const gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_stack_t &stack,
                          CuEVM::evm_call_context_t *call_context);
/**
 * The DELEGATECALL operation.
 * Message-call into this account with an alternative account's code but persisting
 * the current values for sender and value. Gives the new evm call state.
 * Takes gas, address, args offset, args size, ret offset, ret size from the stack.
 * Similar to CALLCODE but keeps the original values of sender and value.
 *
 * @param[in] current_context The current call context.
 * @param[out] new_context_ptr The new call context pointer.
 * @param[in] cached_state The cached call context.
 * @return The error code. 0 if the operation is successful.
 */
__device__ int32_t DELEGATECALL(CuEVM::evm_call_context_t *current_context, CuEVM::evm_call_context_t *&new_context_ptr,
                                CuEVM::cached_evm_call_context &cached_state);

/**
 * The CREATE2 operation.
 * Creates a new account with associated code at a predictable address and gives the new evm call state.
 * Takes value, memory offset, memory length, and salt from the stack.
 * Similar to CREATE but the address is computed using a deterministic formula that includes the salt.
 *
 * @param[in] current_context The current call context.
 * @param[out] new_context_ptr The new call context pointer.
 * @param[in] cached_state The cached call context.
 * @return The error code. 0 if the operation is successful.
 */
__device__ int32_t CREATE2(CuEVM::evm_call_context_t *current_context, CuEVM::evm_call_context_t *&new_context_ptr,
                           CuEVM::cached_evm_call_context &cached_state);

/**
 * The STATICCALL operation.
 * Static message-call into an account and gives the new evm call state.
 * Takes gas, address, args offset, args size, ret offset, ret size from the stack.
 * Similar to CALL but does not allow state modifications and no value transfer.
 *
 * @param[in] current_context The current call context.
 * @param[out] new_context_ptr The new call context pointer.
 * @param[in] cached_state The cached call context.
 * @return The error code. 0 if the operation is successful.
 */
__device__ int32_t STATICCALL(CuEVM::evm_call_context_t *current_context, CuEVM::evm_call_context_t *&new_context_ptr,
                              CuEVM::cached_evm_call_context &cached_state);

/**
 * The REVERT operation.
 * Halt execution reverting state changes but returning data and remaining gas.
 * Takes offset and length from the stack and copies the data from memory
 * as the return data. All state changes in the current call frame are reverted.
 *
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[in] stack The stack.
 * @param[in] call_state_ptr The call state pointer.
 * @return The error code.
 */
__device__ int32_t REVERT(const gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_stack_t &stack,
                          CuEVM::evm_call_context_t *call_state_ptr);

/**
 * The INVALID operation.
 * Designated invalid instruction that consumes all gas and reverts all state changes.
 *
 * @return The error code.
 */
__device__ int32_t INVALID();

/**
 * The SELFDESTRUCT operation.
 * Halt execution and register account for later deletion.
 * Takes the beneficiary address from the stack, to which all remaining Ether is transferred.
 * The current account is registered for deletion after the transaction completes.
 *
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[in] stack The stack.
 * @param[in] call_state_ptr The call state pointer.
 * @return The error code. 0 if the operation is successful.
 */
__device__ int32_t SELFDESTRUCT(const gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_stack_t &stack,
                                CuEVM::evm_call_context_t *call_state_ptr);
}  // namespace CuEVM::operations

#endif