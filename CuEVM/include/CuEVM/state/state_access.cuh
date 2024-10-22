// CuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation
// Programme) Author: Stefan-Dan Ciocirlan Date: 2024-09-15
// SPDX-License-Identifier: MIT

#pragma once

#include <CuEVM/state/account.cuh>
#include <CuEVM/state/account_flags.cuh>
#include <CuEVM/state/state.cuh>
#include <CuEVM/utils/arith.cuh>

namespace CuEVM {

/**
 * The state access struct.
 * It extends the state struct with the flags for each account.
 */
struct state_access_t : state_t {
    CuEVM::account_flags_t *flags; /**< The flags fro each account */

    /**
     * The default constructor.
     */
    __host__ __device__ state_access_t();

    /**
     * THe copy constructor.
     * @param[in] other The other state access.
     */
    __host__ __device__ state_access_t(const state_access_t &other);

    /**
     * The destructor.
     */
    __host__ __device__ ~state_access_t();

    /**
     * The free function.
     */
    __host__ __device__ void free();

    /**
     * The free managed function.
     */
    __host__ void free_managed();

    /**
     * The clear function.
     */
    __host__ __device__ void clear();
    /**
     * The assignment operator.
     * @param[in] other The other state access.
     * @return The current state access.
     */
    __host__ __device__ state_access_t &operator=(const state_access_t &other);

    /**
     * The duplicate function.
     * @param[in] other The other state access.
     */
    __host__ __device__ void duplicate(const state_access_t &other);

    /**
     * The get account function.
     * @param[in] arith The arithmetic environment.
     * @param[in] address The address.
     * @param[out] account The account.
     * @param[in] flag The flag access.
     * @return If found 0. otherwise error.
     */
    __host__ __device__ int32_t get_account(ArithEnv &arith, const bn_t &address, CuEVM::account_t &account,
                                            const CuEVM::account_flags_t flag = CuEVM::ACCOUNT_NONE_FLAG);

    /**
     * The get account function.
     * @param[in] arith The arithmetic environment.
     * @param[in] address The address.
     * @param[out] account_ptr The account pointer.
     * @param[in] flag The flag access.
     * @return If found 0. otherwise error.
     */
    __host__ __device__ int32_t get_account(ArithEnv &arith, const bn_t &address, CuEVM::account_t *&account_ptr,
                                            const CuEVM::account_flags_t flag = CuEVM::ACCOUNT_NONE_FLAG);

    /**
     * The get account index function.
     * @param[in] address The address.
     * @param[out] index The index.
     * @return If found 0. otherwise error.
     */
    __host__ int32_t get_account_index_evm(const evm_word_t &address, uint32_t &index) const;

    /**
     * The add account function.
     * @param[in] account The account.
     * @param[in] flag The flag access.
     * @return If added 0. otherwise error.
     */
    __host__ __device__ int32_t add_account(const CuEVM::account_t &account,
                                            const CuEVM::account_flags_t flag = CuEVM::ACCOUNT_NONE_FLAG);

    /**
     * The add duplicate account function.
     * @param[out] account_ptr The account pointer.
     * @param[in] src_account_ptr The source account pointer.
     * @param[in] flag The flag access.
     * @return If added 0. otherwise error.
     */
    __host__ __device__ int32_t add_duplicate_account(ArithEnv &arith, CuEVM::account_t *&account_ptr,
                                                      CuEVM::account_t *&src_account_ptr,
                                                      const CuEVM::account_flags_t flag = CuEVM::ACCOUNT_NONE_FLAG);

    /**
     * The add new account function.
     * @param[in] arith The arithmetic environment.
     * @param[in] address The address.
     * @param[out] account_ptr The account pointer.
     * @param[in] flag The flag access.
     * @return If added 0. otherwise error.
     */
    __host__ __device__ int32_t add_new_account(ArithEnv &arith, const bn_t &address, CuEVM::account_t *&account_ptr,
                                                const CuEVM::account_flags_t flag = CuEVM::ACCOUNT_NONE_FLAG);

    /**
     * The set account function.
     * @param[in] arith The arithmetic environment.
     * @param[in] account The account.
     * @param[in] flag The flag access.
     * @return If added 0. otherwise error.
     */
    __host__ __device__ int32_t set_account(ArithEnv &arith, const CuEVM::account_t &account,
                                            const CuEVM::account_flags_t flag = CuEVM::ACCOUNT_ALL_FLAG);

    /**
     * The update account function.
     * @param[in] arith The arithmetic environment.
     * @param[in] account The account.
     * @param[in] flag The flag access.
     * @return If added 0. otherwise error.
     */
    __host__ __device__ int32_t update_account(ArithEnv &arith, const CuEVM::account_t &account,
                                               const CuEVM::account_flags_t flag = CuEVM::ACCOUNT_ALL_FLAG);

    /**
     * The update state function.
     * @param[in] arith The arithmetic environment.
     * @param[in] other The other state access.
     * @return If added 0. otherwise error.
     */
    __host__ __device__ int32_t update(ArithEnv &arith, const state_access_t &other);

    /**
     * The print function.
     */
    __host__ __device__ void print();

    /**
     * The to JSON function.
     * @return The JSON.
     */
    __host__ cJSON *to_json();

    // STATIC FUNCTIONS
    /**
     * The merge of two states, the first state is the base state.
     * @param[in] state1 The first state.
     * @param[in] state2 The second state.
     * @return The merged state in JSON.
     */
    __host__ static cJSON *merge_json(const state_t &state1, const state_access_t &state2);
    /**
     * Get the cpu states.
     * @param[in] count The count.
     * @return The cpu states.
     */
    __host__ static state_access_t *get_cpu(uint32_t count);
    /**
     * Free the cpu states.
     * @param[in] cpu_states The cpu states.
     * @param[in] count The count.
     */
    __host__ static void cpu_free(state_access_t *cpu_states, uint32_t count);
    /**
     * Get the gpu states from the cpu.
     * @param[in] cpu_states The cpu states.
     * @param[in] count The count.
     * @return The gpu states.
     */
    __host__ static state_access_t *get_gpu_from_cpu(const state_access_t *cpu_states, uint32_t count);
    /**
     * Free the gpu states.
     * @param[in] gpu_states The gpu states.
     * @param[in] count The count.
     */
    __host__ static void gpu_free(state_access_t *gpu_states, uint32_t count);
    /**
     * Get the cpu states from the gpu.
     * @param[in] gpu_states The gpu states.
     * @param[in] count The count.
     * @return The cpu states.
     */
    __host__ static state_access_t *get_cpu_from_gpu(state_access_t *gpu_states, uint32_t count);
};

/**
 * The state_access_t transfer kernel.
 * @param[out] dst_instances The destination instances.
 * @param[in] src_instances The source instances.
 * @param[in] count The count.
 */
__global__ void state_access_t_transfer_kernel(state_access_t *dst_instances, state_access_t *src_instances,
                                               uint32_t count);
}  // namespace CuEVM