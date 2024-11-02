// CuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2024 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2024-07-12
// SPDX-License-Identifier: MIT

#include <CuEVM/core/message.cuh>

namespace CuEVM {
// __host__ __device__ evm_message_call_t::evm_message_call_t(
//     ArithEnv &arith, const bn_t &sender, const bn_t &recipient, const bn_t &contract_address, const bn_t &gas_limit,
//     const bn_t &value, const uint32_t depth, const uint32_t call_type, const bn_t &storage_address,
//     const CuEVM::byte_array_t &data, const CuEVM::byte_array_t &byte_code, const bn_t &return_data_offset,
//     const bn_t &return_data_size, const uint32_t static_env) {
//     cgbn_store(arith.env, &this->sender, sender);
//     cgbn_store(arith.env, &this->recipient, recipient);
//     // printf("evm_message_call_t constructor contract_address: ");
//     // print_bnt(arith, contract_address);

//     cgbn_store(arith.env, &this->contract_address, contract_address);
//     this->contract_address.print();
//     cgbn_store(arith.env, &this->gas_limit, gas_limit);
//     cgbn_store(arith.env, &this->value, value);
//     this->depth = depth;
//     this->call_type = call_type;
//     cgbn_store(arith.env, &this->storage_address, storage_address);
//     this->data = new byte_array_t(data);
//     this->byte_code = new byte_array_t(byte_code);
//     cgbn_store(arith.env, &this->return_data_offset, return_data_offset);
//     cgbn_store(arith.env, &this->return_data_size, return_data_size);
//     this->static_env = static_env;
//     // create the jump destinations
//     this->jump_destinations = new CuEVM::jump_destinations_t(*this->byte_code);
// }

__host__ __device__ evm_message_call_t_shadow::evm_message_call_t_shadow(
    ArithEnv &arith, const evm_word_t *sender, const evm_word_t *recipient, const evm_word_t *contract_address,
    const evm_word_t *gas_limit, const evm_word_t *value, const uint32_t depth, const uint32_t call_type,
    const evm_word_t *storage_address, const CuEVM::byte_array_t &data, const CuEVM::byte_array_t &byte_code,
    const bn_t &return_data_offset, const bn_t &return_data_size, const uint32_t static_env) {
    __SHARED_MEMORY__ evm_word_t *new_params_data;
    __ONE_GPU_THREAD_WOSYNC_BEGIN__
    new_params_data = new evm_word_t[8];
    __ONE_GPU_THREAD_END__
    this->params_data = new_params_data;
    /*
    evm_word_t *sender;
    evm_word_t *recipient;
    evm_word_t *contract_address;
    evm_word_t *gas_limit;
    evm_word_t *value;
    evm_word_t *storage_address;
    evm_word_t *return_data_offset;
    evm_word_t *return_data_size;
*/
    this->params_data[0] = *sender;
    this->params_data[1] = *recipient;
    this->params_data[2] = *contract_address;
    this->params_data[3] = *gas_limit;
    this->params_data[4] = *value;
    this->params_data[5] = *storage_address;
    // this->params_data[6] = *return_data_offset;
    // this->params_data[7] = *return_data_size;
    cgbn_store(arith.env, &this->params_data[6], return_data_offset);
    cgbn_store(arith.env, &this->params_data[7], return_data_size);
    this->params_data[2].print();

    this->depth = depth;
    this->call_type = call_type;

    this->data = new byte_array_t(data);
    this->byte_code = new byte_array_t(byte_code);

    // printf("evm message call constructor return data offset: ");
    // print_bnt(arith, return_data_offset);
    // this->params_data[6].print();
    // printf("evm message call constructor return data size: ");
    // print_bnt(arith, return_data_size);
    // this->params_data[7].print();

    this->static_env = static_env;
    // create the jump destinations
    // printf("evm create jump destinations %p\n", this->jump_destinations);
    this->jump_destinations = new CuEVM::jump_destinations_t(*this->byte_code);
}

// Copy function , only words global -> shared mem
__host__ __device__ void evm_message_call_t::copy_from(const evm_message_call_t_shadow *other) {
    // printf("evm_message_call_t copy_from other\n ");

    // sender = other.sender;
    // recipient = other.recipient;
    // contract_address = other.contract_address;
    // gas_limit = other.gas_limit;
    // value = other.value;
    // storage_address = other.storage_address;
    // return_data_offset = other.return_data_offset;
    // return_data_size = other.return_data_size;
    __ONE_GPU_THREAD_WOSYNC_BEGIN__
    // memcpy(sender._limbs, other->sender._limbs, CuEVM::cgbn_limbs * sizeof(uint32_t));
    // memcpy(recipient._limbs, other->recipient._limbs, CuEVM::cgbn_limbs * sizeof(uint32_t));
    // memcpy(contract_address._limbs, other->contract_address._limbs, CuEVM::cgbn_limbs * sizeof(uint32_t));
    // memcpy(gas_limit._limbs, other->gas_limit._limbs, CuEVM::cgbn_limbs * sizeof(uint32_t));
    // memcpy(value._limbs, other->value._limbs, CuEVM::cgbn_limbs * sizeof(uint32_t));
    // memcpy(storage_address._limbs, other->storage_address._limbs, CuEVM::cgbn_limbs * sizeof(uint32_t));
    // memcpy(return_data_offset._limbs, other->return_data_offset._limbs, CuEVM::cgbn_limbs * sizeof(uint32_t));
    // memcpy(return_data_size._limbs, other->return_data_size._limbs, CuEVM::cgbn_limbs * sizeof(uint32_t));
    // may not work:
    memcpy(sender._limbs, other->params_data[0]._limbs, CuEVM::cgbn_limbs * sizeof(uint32_t) * 8);
    // printf("evm_message_call_t copy_from other after memcpy\n ");
    data = other->data;
    byte_code = other->byte_code;
    static_env = other->static_env;
    depth = other->depth;
    // printf("evm_message_call_t copy_from other->depth\n ");
    call_type = other->call_type;
    jump_destinations = other->jump_destinations;
    __ONE_GPU_THREAD_END__
}

__host__ __device__ evm_message_call_t::~evm_message_call_t() {
    // todo maybe to delete the inside vectors who knows
    delete jump_destinations;
    jump_destinations = nullptr;
    delete data;
    data = nullptr;
    delete byte_code;
    byte_code = nullptr;
}

/**
 * Get the sender address.
 * @param[in] arith The arithmetical environment.
 * @param[out] sender The sender address YP: \f$s\f$.
 */
__host__ __device__ void evm_message_call_t::get_sender(ArithEnv &arith, bn_t &sender) const {
    cgbn_load(arith.env, sender, (cgbn_evm_word_t_ptr) & this->sender);
}

/**
 * Get the recipient address.
 * @param[in] arith The arithmetical environment.
 * @param[out] recipient The recipient address YP: \f$r\f$.
 */
__host__ __device__ void evm_message_call_t::get_recipient(ArithEnv &arith, bn_t &recipient) const {
    cgbn_load(arith.env, recipient, (cgbn_evm_word_t_ptr) & this->recipient);
}

/**
 * Get the contract address.
 * @param[in] arith The arithmetical environment.
 * @param[out] contract_address The contract address YP: \f$c\f$.
 */
__host__ __device__ void evm_message_call_t::get_contract_address(ArithEnv &arith, bn_t &contract_address) const {
    cgbn_load(arith.env, contract_address, (cgbn_evm_word_t_ptr) & this->contract_address);
}

/**
 * Get the gas limit.
 * @param[in] arith The arithmetical environment.
 * @param[out] gas_limit The gas limit YP: \f$g\f$.
 */
__host__ __device__ void evm_message_call_t::get_gas_limit(ArithEnv &arith, bn_t &gas_limit) const {
    cgbn_load(arith.env, gas_limit, (cgbn_evm_word_t_ptr) & this->gas_limit);
}

/**
 * Get the value.
 * @param[in] arith The arithmetical environment.
 * @param[out] value The value YP: \f$v\f$ or \f$v^{'}\f$ for DelegateCALL.
 */
__host__ __device__ void evm_message_call_t::get_value(ArithEnv &arith, bn_t &value) const {
    cgbn_load(arith.env, value, (cgbn_evm_word_t_ptr) & this->value);
}

/**
 * Get the depth.
 * @return The depth YP: \f$e\f$.
 */
__host__ __device__ uint32_t evm_message_call_t::get_depth() const { return this->depth; }

/**
 * Get the call type.
 * @return The call type internal has the opcode YP: \f$w\f$.
 */
__host__ __device__ uint32_t evm_message_call_t::get_call_type() const { return this->call_type; }

/**
 * Get the storage address.
 * @param[in] arith The arithmetical environment.
 * @param[out] storage_address The storage address YP: \f$a\f$.
 */
__host__ __device__ void evm_message_call_t::get_storage_address(ArithEnv &arith, bn_t &storage_address) const {
    cgbn_load(arith.env, storage_address, (cgbn_evm_word_t_ptr) & this->storage_address);
}

/**
 * Get the call/init data.
 * @return The data YP: \f$d\f$.
 */
__host__ __device__ CuEVM::byte_array_t evm_message_call_t::get_data() const { return *this->data; }

/**
 * Get the byte code.
 * @return The byte code YP: \f$b\f$.
 */
__host__ __device__ CuEVM::byte_array_t evm_message_call_t::get_byte_code() const { return *this->byte_code; }

/**
 * Get the return data offset.
 * @param[in] arith The arithmetical environment.
 * @param[out] return_data_offset The return data offset in memory.
 */
__host__ __device__ void evm_message_call_t::get_return_data_offset(ArithEnv &arith, bn_t &return_data_offset) const {
    cgbn_load(arith.env, return_data_offset, (cgbn_evm_word_t_ptr) & this->return_data_offset);
}

/**
 * Get the return data size.
 * @param[in] arith The arithmetical environment.
 * @param[out] return_data_size The return data size in memory.
 */
__host__ __device__ void evm_message_call_t::get_return_data_size(ArithEnv &arith, bn_t &return_data_size) const {
    cgbn_load(arith.env, return_data_size, (cgbn_evm_word_t_ptr) & this->return_data_size);
}

/**
 * Get the static flag.
 * @return The static flag (STATICCALL) YP: \f$w\f$.
 */
__host__ __device__ uint32_t evm_message_call_t::get_static_env() const { return this->static_env; }

/**
 * Set the gas limit.
 * @param[in] arith The arithmetical environment.
 * @param[in] gas_limit The gas limit YP: \f$g\f$.
 */
__host__ __device__ void evm_message_call_t::set_gas_limit(ArithEnv &arith, bn_t &gas_limit) {
    cgbn_store(arith.env, &this->gas_limit, gas_limit);
}

/**
 * Set the call data.
 * @param[in] data The data YP: \f$d\f$.
 */
__host__ __device__ void evm_message_call_t::set_data(CuEVM::byte_array_t &data) { *this->data = data; }

/**
 * Set the byte code.
 * @param[in] byte_code The byte code YP: \f$b\f$.
 */
__host__ __device__ void evm_message_call_t::set_byte_code(CuEVM::byte_array_t &byte_code) {
    *this->byte_code = byte_code;
    // printf("*this->byte_code = byte_code; \n");
    // this->byte_code->print();
    // printf("other byte_code; \n");
    // byte_code.print();
#ifdef __CUDA_ARCH__
    // printf("*this->byte_code = byte_code;  idx %d \n", threadIdx.x);
#endif
    // __ONE_GPU_THREAD_WOSYNC_BEGIN__
    if (jump_destinations == nullptr) {
        jump_destinations = new CuEVM::jump_destinations_t(*this->byte_code);
        // delete jump_destinations;
        //  jump_destinations = nullptr;
    }
    // __ONE_GPU_THREAD_END__
    jump_destinations->set_bytecode(byte_code);
}

/**
 * Set the return data offset.
 * @param[in] arith The arithmetical environment.
 * @param[in] return_data_offset The return data offset in memory.
 */
__host__ __device__ void evm_message_call_t::set_return_data_offset(ArithEnv &arith, bn_t &return_data_offset) {
    cgbn_store(arith.env, (cgbn_evm_word_t_ptr) & this->return_data_offset, return_data_offset);
}

/**
 * Set the return data size.
 * @param[in] arith The arithmetical environment.
 * @param[in] return_data_size The return data size in memory.
 */
__host__ __device__ void evm_message_call_t::set_return_data_size(ArithEnv &arith, bn_t &return_data_size) {
    cgbn_store(arith.env, (cgbn_evm_word_t_ptr) & this->return_data_size, return_data_size);
}

/**
 * Get the jump destinations.
 * @return The jump destinations.
 */
// __host__ __device__ CuEVM::jump_destinations_t *evm_message_call_t::get_jump_destinations() const {
//     return jump_destinations;
// }

/**
 * Print the message.
 */
__host__ __device__ void evm_message_call_t::print() const {
    printf("sender: ");
    sender.print();
    printf("\nrecipient: ");
    recipient.print();
    printf("\ncontract_address: ");
    contract_address.print();
    printf("\ngas_limit: ");
    gas_limit.print();
    printf("\nvalue: ");
    value.print();
    printf("\ndepth: %d", depth);
    printf("\ncall_type: %d", call_type);
    printf("\nstorage_address: ");
    storage_address.print();
    printf("\ndata: ");
    data->print();
    printf("\nbyte_code: ");
    byte_code->print();
    printf("\nreturn_data_offset: ");
    return_data_offset.print();
    printf("\nreturn_data_size: ");
    return_data_size.print();
    printf("\nstatic_env: %d\n", static_env);
}
}  // namespace CuEVM
