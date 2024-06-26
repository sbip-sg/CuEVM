#ifndef _CUEVM_ACCOUNT_H_
#define _CUEVM_ACCOUNT_H_

#include "arith.cuh"

namespace cuEVM
{
  namespace account
  {
    /**
     * The storage entry type.
    */
    typedef struct
    {
        evm_word_t key; /**< The key of the storage */
        evm_word_t value; /**< The value of the storage for the given key */
    } contract_storage_t;
    /**
     * The account type.
    */
    typedef struct
    {
        evm_word_t address; /**< The address of the account (YP: \f$a\f$) */
        evm_word_t balance; /**< The balance of the account (YP: \f$\sigma[a]_{b}\f$) */
        evm_word_t nonce; /**< The nonce of the account (YP: \f$\sigma[a]_{n}\f$) */
        byte_array_t byte_code; /**< The bytecode of the account (YP: \f$b\f$) */
        uint32_t storage_size; /**< The number of storage entries (YP: \f$|\sigma[a]_{s}|\f$) */
        contract_storage_t *storage; /**< The storage of the account (YP: \f$\sigma[a]_{s}\f$) */
    } account_t;

    /**
     * The kernel to copy the account data structures.
     * @param[out] dst The destination account data structure
     * @param[in] src The source account data structure
     * @param[in] count The number of instances
     */
    __global__ void transfer_kernel(
        account_t *dst_instances,
        account_t *src_instances,
        uint32_t count);
   
    /**
     * Free the memory of the internal byte_code and storage
     * of the account.
     * @param[inout] account The account data structure
     */
    __host__ __device__ void free_internals_account(
        account_t &account);

    /**
     * Generate the cpu data structures for the account.
     * @param[in] count The number of instances
     * @return The cpu data structures
     */
    __host__ account_t *get_cpu_instances(
        uint32_t count);
    
    /**
     * Free the cpu data structures for the account.
     * @param[inout] cpu_instances The cpu data structures
     * @param[in] count The number of instances
     */
    __host__ void free_cpu_instances(
        account_t *cpu_instances,
        uint32_t count);
    
    /**
     * Generate the gpu data structures for the account.
     * @param[in] cpu_instances The cpu data structures
     * @param[in] count The number of instances
     */
    __host__ account_t *get_gpu_instances_from_cpu_instances(
        account_t *cpu_instances,
        uint32_t count);
    
    /**
     * Free the gpu data structures for the account.
     * @param[inout] gpu_instances The gpu data structures
     * @param[in] count The number of instances
     */
    __host__ void free_gpu_instances(
        account_t *gpu_instances,
        uint32_t count);
    
    /**
     * Generate the cpu data structures for the account from the gpu data structures.
     * @param[in] gpu_instances The gpu data structures
     * @param[in] count The number of instances
     * @return The cpu data structures
     */
    __host__ account_t *get_cpu_from_gpu_instances(
        account_t *gpu_instances,
        uint32_t count);
    
    /**
     * Get the managed instances.
     * @param[in] count The number of instances
     * @return The managed instances
     */
    __host__ account_t *get_managed_instances(
        uint32_t count);
    /**
     * Free the internal for managed instance.
     * @param[inout] managed_instance The managed instance
     */
    __host__ void free_internals_managed_instance(
        account_t &managed_instance);
    /**
     * Free the managed instances.
     * @param[inout] managed_instances The managed instances
     * @param[in] count The number of instances
     */
    __host__ void free_managed_instances(
        account_t *managed_instances,
        uint32_t count);
    
    /**
     * Fill the account data structure from the json.
     * @param[out] account The account data structure
     * @param[in] json The json object
     * @param[in] managed The flag to indicate if the memory is managed
     */
    __host__ void from_json(
        account_t &account,
        const cJSON *account_json,
        bool managed = false);
    
    /**
     * Generate a JSON object from the account data structure.
     * @param[in] account The account data structure
     * @return The JSON object
     */
    __host__ cJSON* json(
        account_t &account);
    
    /**
     * Print the account data structure.
     * @param[in] account The account data structure
     */
    __host__ __device__ void print(
        account_t &account);
    
    /**
     * Get the storage index for the given key.
     * @param[out] index The index of the storage
     * @param[in] arith The arithmetical environment
     * @param[in] account The account data structure
     * @param[in] key The key
     * @return If found 1, otherwise 0
     */
    __host__ __device__ int32_t get_storage_index(
        int32_t &index,
        ArithEnv &arith,
        const account_t &account,
        bn_t &key);
    
    /**
     * Make the account empty.
     * @param[inout] account The account data structure
     */
    __host__ __device__ void empty(
        account_t &account);
    
    /**
     * Duplicate the account data structure.
     * @param[out] dst The destination account data structure
     * @param[in] src The source account data structure
     * @param[in] with_storage The flag to indicate if the storage should be duplicated
     */
    __host__ __device__ void duplicate(
        account_t &dst,
        const account_t &src,
        bool with_storage = false);
    
    /**
     * Check if the account is empty.
     * @param[in] arith The arithmetical environment
     * @param[in] account The account data structure
     * @return 1 if the account is empty, 0 otherwise
     */
    __host__ __device__ int32_t is_empty(
        ArithEnv &arith,
        account_t &account);
  }
}
#endif