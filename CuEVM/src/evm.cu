#include <CuEVM/evm.cuh>
#include <CuEVM/gas_cost.cuh>
#include <CuEVM/operations/arithmetic.cuh>
#include <CuEVM/operations/bitwise.cuh>
#include <CuEVM/operations/block.cuh>
#include <CuEVM/operations/compare.cuh>
#include <CuEVM/operations/environmental.cuh>
#include <CuEVM/operations/flow.cuh>
#include <CuEVM/operations/log.cuh>
#include <CuEVM/operations/memory.cuh>
#include <CuEVM/operations/stack.cuh>
#include <CuEVM/operations/storage.cuh>
#include <CuEVM/operations/system.cuh>
#include <CuEVM/precompile.cuh>
#include <CuEVM/utils/arith.cuh>
#include <CuEVM/utils/ecc_constants.cuh>
#include <CuEVM/utils/error_codes.cuh>
#include <CuEVM/utils/opcodes.cuh>

namespace CuEVM {

// define the kernel function
__global__ void kernel_evm_multiple_instances(cgbn_error_report_t *report, CuEVM::evm_instance_t *instances,
                                              uint32_t count) {
    int32_t instance = (blockIdx.x * blockDim.x + threadIdx.x) / CuEVM::cgbn_tpi;
    if (instance >= count) return;
    CuEVM::ArithEnv arith(cgbn_no_checks, report, instance);
    // CuEVM::bn_t test;
    // printf("print simplified trace data device\n");
    // instances[instance].simplified_trace_data_ptr->print();
// printf("new instance %d\n", instance);
#ifdef EIP_3155
    // if (instance == 0) {
    //     __ONE_GPU_THREAD_WOSYNC_BEGIN__
    //     printf("instance %d\n", instance);
    //     printf("world state\n");
    //     instances[instance].world_state_data_ptr->print();
    //     printf("touch state\n");
    //     instances[instance].touch_state_data_ptr->print();
    //     printf("instance %d\n", instance);
    //     printf("transaction\n");
    //     instances[instance].transaction_ptr->print();
    //     __ONE_GPU_THREAD_WOSYNC_END__
    // }
#endif
    // __SHARED_MEMORY__ CuEVM::evm_message_call_t shared_message_call;
    // __SHARED_MEMORY__ CuEVM::evm_word_t shared_stack[CuEVM::shared_stack_size];
    // CuEVM::evm_t *evm = new CuEVM::evm_t(arith, instances[instance], &shared_message_call, shared_stack);
    // printf("evm allocated %p, threadid %d\n", evm, THREADIDX);
    // printf("evm->call_state_ptr %p, threadid %d\n", evm->call_state_ptr, THREADIDX);
    // CuEVM::cached_evm_call_state cached_state(arith, evm->call_state_ptr);
    __SHARED_MEMORY__ CuEVM::evm_message_call_t shared_message_call[CGBN_IBP];
    __SHARED_MEMORY__ CuEVM::evm_word_t shared_stack[CGBN_IBP][CuEVM::shared_stack_size];
    CuEVM::evm_t *evm = new CuEVM::evm_t(arith, instances[instance], &shared_message_call[INSTANCE_IDX_PER_BLOCK],
                                         shared_stack[INSTANCE_IDX_PER_BLOCK]);
    CuEVM::cached_evm_call_state cached_state(arith, evm->call_state_ptr);
    // printf("\nevm->run(arith) instance %d\n", instance);
    // printf("print simplified trace data device inside evm\n");
    // evm->simplified_trace_data_ptr->print();
    // printf("gas limit %d thread %d\n", THREADIDX);
    // print_bnt(arith, evm->call_state_ptr->gas_limit);
    // print_bnt(arith, cached_state.gas_limit);
    __SYNC_THREADS__
    evm->run(arith, cached_state);

#ifdef EIP_3155
    if (instance == 0) {
        __ONE_GPU_THREAD_BEGIN__
        // instances[0].tracer_ptr->print(arith);
        instances[0].tracer_ptr->print_err();
        __ONE_GPU_THREAD_WOSYNC_END__
    }
#endif
    // print the final world state
    // __ONE_GPU_THREAD_WOSYNC_BEGIN__
    // instances[instance].world_state_data_ptr->print();
    // printf("simplified trace data\n");
    // instances[instance].simplified_trace_data_ptr->print();
    // __ONE_GPU_THREAD_WOSYNC_END__
    // delete evm;
    // evm = nullptr;
}

__host__ __device__ evm_t::evm_t(ArithEnv &arith, CuEVM::state_t *world_state_data_ptr,
                                 CuEVM::block_info_t *block_info_ptr, CuEVM::evm_transaction_t *transaction_ptr,
                                 CuEVM::state_access_t *touch_state_data_ptr, CuEVM::log_state_data_t *log_state_ptr,
                                 CuEVM::evm_return_data_t *return_data_ptr, CuEVM::EccConstants *ecc_constants_ptr,
                                 CuEVM::evm_message_call_t *shared_message_call_ptr, CuEVM::evm_word_t *shared_stack_ptr
#ifdef EIP_3155
                                 ,
                                 CuEVM::utils::tracer_t *tracer_ptr
#endif

                                 ,
                                 CuEVM::serialized_worldstate_data *serialized_worldstate_data_ptr,
                                 CuEVM::utils::simplified_trace_data *simplified_trace_data_ptr)
    : world_state(world_state_data_ptr),
      block_info_ptr(block_info_ptr),
      transaction_ptr(transaction_ptr),

      serialized_worldstate_data_ptr(serialized_worldstate_data_ptr),
      simplified_trace_data_ptr(simplified_trace_data_ptr),
      ecc_constants_ptr(ecc_constants_ptr) {
    // TODO: store in local/shared memory
    call_state_ptr = new CuEVM::evm_call_state_t(arith, &world_state, nullptr, nullptr, log_state_ptr,
                                                 touch_state_data_ptr, return_data_ptr);
    // call_state_local = evm_call_steate_t(arith, call_state_ptr);
    // #ifndef __CUDA_ARCH__
    //     shared_message_call_ptr = new evm_message_call_t();
    // #endif

    int32_t error_code = transaction_ptr->validate(arith, call_state_ptr->touch_state_ptr, *block_info_ptr,
                                                   call_state_ptr->gas_used, gas_price, gas_priority_fee);
    // printf("transaction validated error code %d\n", error_code);
    // #ifdef __CUDA_ARCH__
    //     printf("error  code %d idx %d \n", error_code, threadIdx.x);
    // #endif
    // __SHARED_MEMORY__ evm_message_call_t shared_message_call;
    if (error_code == ERROR_SUCCESS) {
        CuEVM::evm_message_call_t_shadow *transaction_call_message_ptr = nullptr;
        error_code =
            transaction_ptr->get_message_call(arith, call_state_ptr->touch_state_ptr, transaction_call_message_ptr);

        shared_message_call_ptr->copy_from(transaction_call_message_ptr);

        // printf("\n\n message call copied %d\n", THREADIDX);
        // shared_message_call_ptr->print();

        CuEVM::evm_call_state_t *child_call_state_ptr = new CuEVM::evm_call_state_t(
            arith, call_state_ptr, shared_message_call_ptr, transaction_call_message_ptr, shared_stack_ptr);
        // printf("child_call_state_ptr allocated %p, threadid %d\n", child_call_state_ptr, THREADIDX);
        // print_bnt(arith, child_call_state_ptr->gas_limit);
        // subtract the gas used by the transaction initialization from the gas
        // limit
        cgbn_sub(arith.env, child_call_state_ptr->gas_limit, child_call_state_ptr->gas_limit, call_state_ptr->gas_used);
        call_state_ptr = child_call_state_ptr;
        // cached_call_state = cached_evm_call_state(arith, call_state_ptr);
        // call_state_local = evm_call_state_t(arith, call_state_ptr);
        // #ifdef EIP_3155
        //         // printing debug when enabling tracer.
        //         printf("call_state_ptr allocated %p, threadid %d\n", call_state_ptr, THREADIDX);
        //         // transaction_ptr->print();
        //         call_state_ptr->print(arith);
        //         printf("call_state_local %d\n", THREADIDX);
        //         call_state_local.print(arith);
        // #endif
    }
#ifdef EIP_3155
    this->tracer_ptr = tracer_ptr;
#endif
    status = error_code;
}

__host__ __device__ evm_t::evm_t(ArithEnv &arith, CuEVM::evm_instance_t &evm_instance,
                                 CuEVM::evm_message_call_t *message_call, CuEVM::evm_word_t *shared_stack_ptr)
    : evm_t(arith, evm_instance.world_state_data_ptr, evm_instance.block_info_ptr, evm_instance.transaction_ptr,
            evm_instance.touch_state_data_ptr, evm_instance.log_state_ptr, evm_instance.return_data_ptr,
            evm_instance.ecc_constants_ptr, message_call, shared_stack_ptr
#ifdef EIP_3155
            ,
            evm_instance.tracer_ptr
#endif
            ,
            evm_instance.serialized_worldstate_data_ptr, evm_instance.simplified_trace_data_ptr) {
}

__host__ __device__ evm_t::~evm_t() {
    // if (call_state_ptr != nullptr) {
    //     call_state_ptr->touch_state.clear();
    //     delete call_state_ptr;
    // }
    /// Todo double check touch_state_ptr
    call_state_ptr = nullptr;
    block_info_ptr = nullptr;
    transaction_ptr = nullptr;
#ifdef EIP_3155
    tracer_ptr = nullptr;
#endif
}

__host__ __device__ int32_t evm_t::start_CALL(ArithEnv &arith, cached_evm_call_state &cached_call_state) {
    // printf("start_CALL\n");
    // bn_t sender, recipient, value;
    // call_state_ptr->message_ptr->get_sender(arith, sender);
    // call_state_ptr->message_ptr->get_recipient(arith, recipient);
    // #ifdef EIP_3155
    // printf("start_CALL %d\n", THREADIDX);
    // __SYNC_THREADS__
    // #endif

    bn_t value;
    const evm_word_t *sender = &call_state_ptr->message_ptr->sender;
    const evm_word_t *recipient = &call_state_ptr->message_ptr->recipient;
    call_state_ptr->message_ptr->get_value(arith, value);

    int32_t error_code = (((cgbn_compare_ui32(arith.env, value, 0) > 0) &&
                           // (cgbn_compare(arith.env, sender, recipient) != 0) &&
                           (call_state_ptr->message_ptr->call_type != OP_DELEGATECALL))
                              ? call_state_ptr->touch_state_ptr->transfer(arith, sender, recipient, value)
                              : ERROR_SUCCESS);
    if (error_code != ERROR_SUCCESS) {
        // avoid complication in the subsequent code
        // call failed = account never warmed up
        return error_code;
    }
#ifdef EIP_3155
// #ifdef __CUDA_ARCH__
//     printf("start_CALL transfer error code %d idx %d \n", error_code, threadIdx.x);
//     __ONE_THREAD_PER_INSTANCE(printf("value ");)
//     print_bnt(arith, value);
// #endif
#endif

    if (call_state_ptr->message_ptr->call_type == OP_CALL || call_state_ptr->message_ptr->call_type == OP_CALLCODE ||
        call_state_ptr->message_ptr->call_type == OP_DELEGATECALL ||
        call_state_ptr->message_ptr->call_type == OP_STATICCALL) {
        CuEVM::account_t *contract = nullptr;
        const evm_word_t *contract_address = &call_state_ptr->message_ptr->contract_address;
        // call_state_ptr->message_ptr->get_contract_address(arith, contract_address);
        // error_code |=
        // #ifdef __CUDA_ARCH__
        //         printf("get byte code idx %d contract_address %p \n", threadIdx.x, contract_address);
        //         contract_address->print();
        // #endif
        call_state_ptr->touch_state_ptr->get_account(arith, contract_address, contract, ACCOUNT_BYTE_CODE_FLAG);
        // #ifdef __CUDA_ARCH__
        //         printf("call_state_ptr->touch_state.get_account idx %d  \n", threadIdx.x);
        //         contract_address->print();
        //         contract->byte_code.print();
        // #endif
        call_state_ptr->message_ptr->set_byte_code(contract->byte_code);
        // #ifdef __CUDA_ARCH__
        //         printf("call_state_ptr->message_ptr->set_byte_code %d  \n", threadIdx.x);
        //         contract_address->print();
        // #endif
    }
    // #ifdef __CUDA_ARCH__
    //     printf("before warmup sender %p recipient %p  idx %d\n", sender, recipient, threadIdx.x);
    // #endif
    // warmup the accounts
    CuEVM::account_t *account_ptr = nullptr;
    call_state_ptr->touch_state_ptr->set_warm_account(arith, sender);
    // #ifdef __CUDA_ARCH__
    //     printf("after  set_warm_account first %d\n", threadIdx.x);
    // #endif
    call_state_ptr->touch_state_ptr->set_warm_account(arith, recipient);
    // #ifdef __CUDA_ARCH__
    //     printf("after  set_warm_account %d\n", threadIdx.x);
    // #endif
    // error_code |=
    call_state_ptr->touch_state_ptr->get_account(arith, recipient, account_ptr, ACCOUNT_NONE_FLAG);

    // #ifdef EIP_3155
    // #ifdef __CUDA_ARCH__
    //     printf("call_state_ptr->touch_state.get_account error code %d,  idx %d  pointer %p, balance\n", error_code,
    //            threadIdx.x, account_ptr);
    //     account_ptr->balance.print();
    // #endif
    // #endif

    cached_call_state.set_byte_code(call_state_ptr->message_ptr->byte_code);

    if ((call_state_ptr->message_ptr->call_type == OP_CREATE) ||
        (call_state_ptr->message_ptr->call_type == OP_CREATE2)) {
        error_code |= account_ptr->is_empty_create() ? ERROR_SUCCESS : ERROR_MESSAGE_CALL_CREATE_CONTRACT_EXISTS;
        // printf("start_CALL contract ERROR_MESSAGE_CALL_CREATE_CONTRACT_EXISTS\n");
        bn_t contract_nonce;
        cgbn_set_ui32(arith.env, contract_nonce, 1);
        error_code |= call_state_ptr->touch_state_ptr->set_nonce(arith, recipient, contract_nonce);
        bn_t sender_nonce;
        error_code |= call_state_ptr->touch_state_ptr->get_nonce(arith, sender, sender_nonce);

        uint64_t nonce;
        error_code |= cgbn_get_uint64_t(arith.env, nonce, sender_nonce) == ERROR_VALUE_OVERFLOW
                          ? ERROR_MESSAGE_CALL_CREATE_NONCE_EXCEEDED
                          : ERROR_SUCCESS;
        cgbn_add_ui32(arith.env, sender_nonce, sender_nonce, 1);

    } else {
        // Go-ethereum: check depth > 1024 before increase
        // -> depth > 1025 after increase
        // test: stSelfBalance/diffPlaces.json
        error_code |= call_state_ptr->depth > CuEVM::max_depth + 1 ? ERROR_MESSAGE_CALL_DEPTH_EXCEEDED : ERROR_SUCCESS;
        // #ifdef __CUDA_ARCH__
        //         printf("else code size 0 code %d idx %d \n", error_code, threadIdx.x);
        // #endif
        // Dont use account ptr here, byte_code already set
        if (call_state_ptr->message_ptr->byte_code->size == 0) {
            bn_t contract_address;
            call_state_ptr->message_ptr->get_contract_address(arith, contract_address);
            if (cgbn_compare_ui32(arith.env, contract_address, CuEVM::no_precompile_contracts) == -1) {
                switch (cgbn_get_ui32(arith.env, contract_address)) {
                    case 0x01:

                        return CuEVM::precompile_operations::operation_ecRecover(
                            arith, this->ecc_constants_ptr, cached_call_state.gas_limit, cached_call_state.gas_used,
                            call_state_ptr->parent->last_return_data_ptr, call_state_ptr->message_ptr);
                        break;
                    case 0x02:
                        return CuEVM::precompile_operations::operation_SHA256(
                            arith, cached_call_state.gas_limit, cached_call_state.gas_used,
                            call_state_ptr->parent->last_return_data_ptr, call_state_ptr->message_ptr);
                    case 0x03:
                        return CuEVM::precompile_operations::operation_RIPEMD160(
                            arith, cached_call_state.gas_limit, cached_call_state.gas_used,
                            call_state_ptr->parent->last_return_data_ptr, call_state_ptr->message_ptr);
                    case 0x04:
                        return CuEVM::precompile_operations::operation_IDENTITY(
                            arith, cached_call_state.gas_limit, cached_call_state.gas_used,
                            call_state_ptr->parent->last_return_data_ptr, call_state_ptr->message_ptr);
                    case 0x05:
                        return CuEVM::precompile_operations::operation_MODEXP(
                            arith, cached_call_state.gas_limit, cached_call_state.gas_used,
                            call_state_ptr->parent->last_return_data_ptr, call_state_ptr->message_ptr);
                    case 0x06:
                        return CuEVM::precompile_operations::operation_ecAdd(
                            arith, this->ecc_constants_ptr, cached_call_state.gas_limit, cached_call_state.gas_used,
                            call_state_ptr->parent->last_return_data_ptr, call_state_ptr->message_ptr);
                    case 0x07:
                        return CuEVM::precompile_operations::operation_ecMul(
                            arith, this->ecc_constants_ptr, cached_call_state.gas_limit, cached_call_state.gas_used,
                            call_state_ptr->parent->last_return_data_ptr, call_state_ptr->message_ptr);
                    case 0x08:
                        return CuEVM::precompile_operations::operation_ecPairing(
                            arith, this->ecc_constants_ptr, cached_call_state.gas_limit, cached_call_state.gas_used,
                            call_state_ptr->parent->last_return_data_ptr, call_state_ptr->message_ptr);
                    case 0x09:
                        return CuEVM::precompile_operations::operation_BLAKE2(
                            arith, cached_call_state.gas_limit, cached_call_state.gas_used,
                            call_state_ptr->parent->last_return_data_ptr, call_state_ptr->message_ptr);
                    case 0x0a:
                        return ERROR_RETURN;
                    default:
                        return ERROR_RETURN;
                        break;
                }
            } else {
                // operation stop
                // clear return data
                // call_state_ptr->parent->last_return_data_ptr->free();
                // call_state_ptr->parent->last_return_data_ptr = new CuEVM::evm_return_data_t();
                CuEVM::byte_array_t::reset_return_data(call_state_ptr->parent->last_return_data_ptr);
                return ERROR_RETURN;
            }
        }
    }
    // #ifdef __CUDA_ARCH__
    //     printf("start_CALL end error code %d idx %d\n", error_code, threadIdx.x);
    // #endif
    return error_code;
}
__host__ __device__ void evm_t::run(ArithEnv &arith) {
    cached_evm_call_state cached_call_state(arith, call_state_ptr);
    run(arith, cached_call_state);
}

__host__ __device__ void evm_t::run(ArithEnv &arith, cached_evm_call_state &cached_call_state) {
    if (status != ERROR_SUCCESS) {
        return;  // finish transaction
    }
#ifdef BUILD_LIBRARY
    simplified_trace_data_ptr->start_call(0, call_state_ptr->message_ptr);
#endif
    int32_t error_code = start_CALL(arith, cached_call_state);
    if (error_code != ERROR_SUCCESS) {
#ifdef BUILD_LIBRARY
        simplified_trace_data_ptr->finish_call(0);
#endif
        return;  // finish call
    }
    uint8_t opcode;
#ifdef BUILD_LIBRARY
    uint32_t pc_src;
#endif
    CuEVM::evm_call_state_t *child_call_state_ptr = nullptr;
    while (true) {
        // uint32_t current_pc = call_state_ptr->pc;  // TODO: store in local/shared memory
        // opcode = ((current_pc < ((call_state_ptr->message_ptr)->byte_code)->size)
        //               ? (call_state_ptr->message_ptr)->byte_code->data[current_pc]
        //               : OP_STOP);
        opcode = ((cached_call_state.pc < cached_call_state.byte_code_size)
                      ? cached_call_state.byte_code_data[cached_call_state.pc]
                      : OP_STOP);
        __SYNC_THREADS__
#ifdef EIP_3155
        // #ifdef __CUDA_ARCH__
        //         __ONE_GPU_THREAD_WOSYNC_BEGIN__
        //         printf("idx %d before start_operation opcode %d tracer_idx %d , tracer ptr %p, tracer size %d\n",
        //         threadIdx.x,
        //                opcode, call_state_ptr->trace_idx, tracer_ptr, tracer_ptr->size);
        //         __ONE_GPU_THREAD_WOSYNC_END__
        // #endif
        // printf("before start_operation mem ptr %p memsize %d threadidx %d\n", call_state_ptr->memory_ptr,
        //        call_state_ptr->memory_ptr->size, THREADIDX);
        uint32_t trace_idx = tracer_ptr->start_operation(arith, cached_call_state.pc, opcode,
                                                         *call_state_ptr->memory_ptr, *cached_call_state.stack_ptr,
                                                         call_state_ptr->depth, *call_state_ptr->last_return_data_ptr,
                                                         cached_call_state.gas_limit, cached_call_state.gas_used);
        call_state_ptr->trace_idx = trace_idx;

        // __ONE_GPU_THREAD_WOSYNC_BEGIN__
        // printf("\npc: %d opcode: %d, depth %d, thread %d \n", cached_call_state.pc, opcode, call_state_ptr->depth,
        //        THREADIDX);
        // print_bnt(arith, cached_call_state.gas_limit);
        // print_bnt(arith, cached_call_state.gas_used);
        // __ONE_GPU_THREAD_WOSYNC_END__

#endif

        // __ONE_GPU_THREAD_WOSYNC_BEGIN__
        // printf("\npc: %d opcode: %d, depth %d, thread %d \n", cached_call_state.pc, opcode, call_state_ptr->depth,
        //        THREADIDX);
        // __ONE_GPU_THREAD_WOSYNC_END__

#ifdef BUILD_LIBRARY
        // comparison, arithmetic, revert/invalid
        if ((opcode <= OP_EXP || opcode >= OP_REVERT || opcode == OP_SSTORE) && opcode != 0) {
            simplified_trace_data_ptr->start_operation(cached_call_state.pc, opcode, *cached_call_state.stack_ptr);
        }
        if (opcode == OP_JUMPI) pc_src = cached_call_state.pc;
#endif
        // DEBUG PRINT

        // __ONE_THREAD_PER_INSTANCE(printf("\npc: %d opcode: %d\n", call_state_ptr->pc, opcode););
        // printf("touch state BEGIN BEGIN BEGIN\n");
        // call_state_ptr->touch_state.print();
        // printf("touch state END END END\n");
        if (((opcode & 0xF0) == 0x60) || ((opcode & 0xF0) == 0x70)) {
            error_code = CuEVM::operations::PUSHX(arith, cached_call_state.gas_limit, cached_call_state.gas_used,
                                                  cached_call_state.pc, *cached_call_state.stack_ptr,
                                                  *((call_state_ptr->message_ptr)->byte_code), opcode);
        } else if ((opcode & 0xF0) == 0x80)  // DUPX
        {
            error_code = CuEVM::operations::DUPX(arith, cached_call_state.gas_limit, cached_call_state.gas_used,
                                                 *cached_call_state.stack_ptr, opcode);
        } else if ((opcode & 0xF0) == 0x90)  // SWAPX
        {
            error_code = CuEVM::operations::SWAPX(arith, cached_call_state.gas_limit, cached_call_state.gas_used,
                                                  *cached_call_state.stack_ptr, opcode);
        } else {
            switch (opcode) {
                case OP_STOP:
                    error_code = CuEVM::operations::STOP(*call_state_ptr->parent->last_return_data_ptr);
                    break;
                case OP_ADD:
                    error_code = CuEVM::operations::ADD(arith, cached_call_state.gas_limit, cached_call_state.gas_used,
                                                        *cached_call_state.stack_ptr);
                    break;
                case OP_MUL:
                    error_code = CuEVM::operations::MUL(arith, cached_call_state.gas_limit, cached_call_state.gas_used,
                                                        *cached_call_state.stack_ptr);
                    break;
                case OP_SUB:
                    error_code = CuEVM::operations::SUB(arith, cached_call_state.gas_limit, cached_call_state.gas_used,
                                                        *cached_call_state.stack_ptr);
                    break;
                case OP_DIV:
                    error_code = CuEVM::operations::DIV(arith, cached_call_state.gas_limit, cached_call_state.gas_used,
                                                        *cached_call_state.stack_ptr);
                    break;
                case OP_SDIV:
                    error_code = CuEVM::operations::SDIV(arith, cached_call_state.gas_limit, cached_call_state.gas_used,
                                                         *cached_call_state.stack_ptr);
                    break;
                case OP_MOD:
                    error_code = CuEVM::operations::MOD(arith, cached_call_state.gas_limit, cached_call_state.gas_used,
                                                        *cached_call_state.stack_ptr);
                    break;
                case OP_SMOD:
                    error_code = CuEVM::operations::SMOD(arith, cached_call_state.gas_limit, cached_call_state.gas_used,
                                                         *cached_call_state.stack_ptr);
                    break;
                case OP_ADDMOD:
                    error_code = CuEVM::operations::ADDMOD(arith, cached_call_state.gas_limit,
                                                           cached_call_state.gas_used, *cached_call_state.stack_ptr);
                    break;
                case OP_MULMOD:
                    error_code = CuEVM::operations::MULMOD(arith, cached_call_state.gas_limit,
                                                           cached_call_state.gas_used, *cached_call_state.stack_ptr);
                    break;
                case OP_EXP:
                    error_code = CuEVM::operations::EXP(arith, cached_call_state.gas_limit, cached_call_state.gas_used,
                                                        *cached_call_state.stack_ptr);
                    break;
                case OP_SIGNEXTEND:
                    error_code = CuEVM::operations::SIGNEXTEND(
                        arith, cached_call_state.gas_limit, cached_call_state.gas_used, *cached_call_state.stack_ptr);
                    break;
                case OP_LT:
#ifdef BUILD_LIBRARY
                    simplified_trace_data_ptr->record_distance(arith, opcode, *cached_call_state.stack_ptr);
#endif
                    error_code = CuEVM::operations::LT(arith, cached_call_state.gas_limit, cached_call_state.gas_used,
                                                       *cached_call_state.stack_ptr);
                    break;
                case OP_GT:
#ifdef BUILD_LIBRARY
                    simplified_trace_data_ptr->record_distance(arith, opcode, *cached_call_state.stack_ptr);
#endif
                    error_code = CuEVM::operations::GT(arith, cached_call_state.gas_limit, cached_call_state.gas_used,
                                                       *cached_call_state.stack_ptr);
                    break;
                case OP_SLT:
#ifdef BUILD_LIBRARY
                    simplified_trace_data_ptr->record_distance(arith, opcode, *cached_call_state.stack_ptr);
#endif
                    error_code = CuEVM::operations::SLT(arith, cached_call_state.gas_limit, cached_call_state.gas_used,
                                                        *cached_call_state.stack_ptr);
                    break;
                case OP_SGT:
#ifdef BUILD_LIBRARY
                    simplified_trace_data_ptr->record_distance(arith, opcode, *cached_call_state.stack_ptr);
#endif
                    error_code = CuEVM::operations::SGT(arith, cached_call_state.gas_limit, cached_call_state.gas_used,
                                                        *cached_call_state.stack_ptr);
                    break;
                case OP_EQ:
#ifdef BUILD_LIBRARY
                    simplified_trace_data_ptr->record_distance(arith, opcode, *cached_call_state.stack_ptr);
#endif
                    error_code = CuEVM::operations::EQ(arith, cached_call_state.gas_limit, cached_call_state.gas_used,
                                                       *cached_call_state.stack_ptr);
                    break;
                case OP_ISZERO:
                    error_code = CuEVM::operations::ISZERO(arith, cached_call_state.gas_limit,
                                                           cached_call_state.gas_used, *cached_call_state.stack_ptr);
                    break;
                case OP_AND:
                    error_code = CuEVM::operations::AND(arith, cached_call_state.gas_limit, cached_call_state.gas_used,
                                                        *cached_call_state.stack_ptr);
                    break;
                case OP_OR:
                    error_code = CuEVM::operations::OR(arith, cached_call_state.gas_limit, cached_call_state.gas_used,
                                                       *cached_call_state.stack_ptr);
                    break;
                case OP_XOR:
                    error_code = CuEVM::operations::XOR(arith, cached_call_state.gas_limit, cached_call_state.gas_used,
                                                        *cached_call_state.stack_ptr);
                    break;
                case OP_NOT:
                    error_code = CuEVM::operations::NOT(arith, cached_call_state.gas_limit, cached_call_state.gas_used,
                                                        *cached_call_state.stack_ptr);
                    break;
                case OP_BYTE:
                    error_code = CuEVM::operations::BYTE(arith, cached_call_state.gas_limit, cached_call_state.gas_used,
                                                         *cached_call_state.stack_ptr);
                    break;
                case OP_SHL:
                    error_code = CuEVM::operations::SHL(arith, cached_call_state.gas_limit, cached_call_state.gas_used,
                                                        *cached_call_state.stack_ptr);
                    break;
                case OP_SHR:
                    error_code = CuEVM::operations::SHR(arith, cached_call_state.gas_limit, cached_call_state.gas_used,
                                                        *cached_call_state.stack_ptr);
                    break;
                case OP_SAR:
                    error_code = CuEVM::operations::SAR(arith, cached_call_state.gas_limit, cached_call_state.gas_used,
                                                        *cached_call_state.stack_ptr);
                    break;
                case OP_SHA3:
                    error_code = CuEVM::operations::SHA3(arith, cached_call_state.gas_limit, cached_call_state.gas_used,
                                                         *cached_call_state.stack_ptr, *call_state_ptr->memory_ptr);
                    break;
                case OP_ADDRESS:
                    error_code =
                        CuEVM::operations::ADDRESS(arith, cached_call_state.gas_limit, cached_call_state.gas_used,
                                                   *cached_call_state.stack_ptr, *call_state_ptr->message_ptr);
                    break;
                case OP_BALANCE:
                    error_code =
                        CuEVM::operations::BALANCE(arith, cached_call_state.gas_limit, cached_call_state.gas_used,
                                                   *cached_call_state.stack_ptr, *call_state_ptr->touch_state_ptr);
                    break;
                case OP_ORIGIN:
                    error_code =
                        CuEVM::operations::ORIGIN(arith, cached_call_state.gas_limit, cached_call_state.gas_used,
                                                  *cached_call_state.stack_ptr, *transaction_ptr);
                    break;
                case OP_CALLER:
                    error_code =
                        CuEVM::operations::CALLER(arith, cached_call_state.gas_limit, cached_call_state.gas_used,
                                                  *cached_call_state.stack_ptr, *call_state_ptr->message_ptr);
                    break;
                case OP_CALLVALUE:
                    error_code =
                        CuEVM::operations::CALLVALUE(arith, cached_call_state.gas_limit, cached_call_state.gas_used,
                                                     *cached_call_state.stack_ptr, *call_state_ptr->message_ptr);
                    break;
                case OP_CALLDATALOAD:
                    error_code =
                        CuEVM::operations::CALLDATALOAD(arith, cached_call_state.gas_limit, cached_call_state.gas_used,
                                                        *cached_call_state.stack_ptr, *call_state_ptr->message_ptr);
                    break;
                case OP_CALLDATASIZE:
                    error_code =
                        CuEVM::operations::CALLDATASIZE(arith, cached_call_state.gas_limit, cached_call_state.gas_used,
                                                        *cached_call_state.stack_ptr, *call_state_ptr->message_ptr);
                    break;
                case OP_CALLDATACOPY:
                    error_code = CuEVM::operations::CALLDATACOPY(
                        arith, cached_call_state.gas_limit, cached_call_state.gas_used, *cached_call_state.stack_ptr,
                        *call_state_ptr->message_ptr, *call_state_ptr->memory_ptr);
                    break;
                case OP_CODESIZE:
                    error_code =
                        CuEVM::operations::CODESIZE(arith, cached_call_state.gas_limit, cached_call_state.gas_used,
                                                    *cached_call_state.stack_ptr, *call_state_ptr->message_ptr);
                    break;
                case OP_CODECOPY:
                    error_code = CuEVM::operations::CODECOPY(arith, cached_call_state.gas_limit,
                                                             cached_call_state.gas_used, *cached_call_state.stack_ptr,
                                                             *call_state_ptr->message_ptr, *call_state_ptr->memory_ptr);
                    break;
                case OP_GASPRICE:
                    error_code =
                        CuEVM::operations::GASPRICE(arith, cached_call_state.gas_limit, cached_call_state.gas_used,
                                                    *cached_call_state.stack_ptr, *block_info_ptr, *transaction_ptr);
                    break;
                case OP_EXTCODESIZE:
                    error_code =
                        CuEVM::operations::EXTCODESIZE(arith, cached_call_state.gas_limit, cached_call_state.gas_used,
                                                       *cached_call_state.stack_ptr, *call_state_ptr->touch_state_ptr);
                    break;
                case OP_EXTCODECOPY:
                    error_code = CuEVM::operations::EXTCODECOPY(
                        arith, cached_call_state.gas_limit, cached_call_state.gas_used, *cached_call_state.stack_ptr,
                        *call_state_ptr->touch_state_ptr, *call_state_ptr->memory_ptr);
                    break;
                case OP_RETURNDATASIZE:
                    error_code = CuEVM::operations::RETURNDATASIZE(
                        arith, cached_call_state.gas_limit, cached_call_state.gas_used, *cached_call_state.stack_ptr,
                        *call_state_ptr->last_return_data_ptr);
                    break;
                case OP_RETURNDATACOPY:
                    error_code = CuEVM::operations::RETURNDATACOPY(
                        arith, cached_call_state.gas_limit, cached_call_state.gas_used, *cached_call_state.stack_ptr,
                        *call_state_ptr->memory_ptr, *call_state_ptr->last_return_data_ptr);
                    break;
                case OP_EXTCODEHASH:
                    error_code =
                        CuEVM::operations::EXTCODEHASH(arith, cached_call_state.gas_limit, cached_call_state.gas_used,
                                                       *cached_call_state.stack_ptr, *call_state_ptr->touch_state_ptr);
                    break;
                case OP_BLOCKHASH:
                    error_code =
                        CuEVM::operations::BLOCKHASH(arith, cached_call_state.gas_limit, cached_call_state.gas_used,
                                                     *cached_call_state.stack_ptr, *block_info_ptr);
                    break;
                case OP_COINBASE:
                    error_code =
                        CuEVM::operations::COINBASE(arith, cached_call_state.gas_limit, cached_call_state.gas_used,
                                                    *cached_call_state.stack_ptr, *block_info_ptr);
                    break;
                case OP_TIMESTAMP:
                    error_code =
                        CuEVM::operations::TIMESTAMP(arith, cached_call_state.gas_limit, cached_call_state.gas_used,
                                                     *cached_call_state.stack_ptr, *block_info_ptr);
                    break;
                case OP_NUMBER:
                    error_code =
                        CuEVM::operations::NUMBER(arith, cached_call_state.gas_limit, cached_call_state.gas_used,
                                                  *cached_call_state.stack_ptr, *block_info_ptr);
                    break;
                case OP_DIFFICULTY:
                    error_code =
                        CuEVM::operations::PREVRANDAO(arith, cached_call_state.gas_limit, cached_call_state.gas_used,
                                                      *cached_call_state.stack_ptr, *block_info_ptr);
                    break;
                case OP_GASLIMIT:
                    error_code =
                        CuEVM::operations::GASLIMIT(arith, cached_call_state.gas_limit, cached_call_state.gas_used,
                                                    *cached_call_state.stack_ptr, *block_info_ptr);
                    break;
                case OP_CHAINID:
                    error_code =
                        CuEVM::operations::CHAINID(arith, cached_call_state.gas_limit, cached_call_state.gas_used,
                                                   *cached_call_state.stack_ptr, *block_info_ptr);
                    break;
                case OP_SELFBALANCE:
                    error_code = CuEVM::operations::SELFBALANCE(
                        arith, cached_call_state.gas_limit, cached_call_state.gas_used, *cached_call_state.stack_ptr,
                        *call_state_ptr->touch_state_ptr, *call_state_ptr->message_ptr);
                    break;
                case OP_BASEFEE:
                    error_code =
                        CuEVM::operations::BASEFEE(arith, cached_call_state.gas_limit, cached_call_state.gas_used,
                                                   *cached_call_state.stack_ptr, *block_info_ptr);
                    break;
                case OP_POP:
                    error_code = CuEVM::operations::POP(arith, cached_call_state.gas_limit, cached_call_state.gas_used,
                                                        *cached_call_state.stack_ptr);
                    break;
                case OP_MLOAD:
                    error_code =
                        CuEVM::operations::MLOAD(arith, cached_call_state.gas_limit, cached_call_state.gas_used,
                                                 *cached_call_state.stack_ptr, *call_state_ptr->memory_ptr);
                    break;
                case OP_MSTORE:
                    error_code =
                        CuEVM::operations::MSTORE(arith, cached_call_state.gas_limit, cached_call_state.gas_used,
                                                  *cached_call_state.stack_ptr, *call_state_ptr->memory_ptr);
                    break;
                case OP_MSTORE8:
                    error_code =
                        CuEVM::operations::MSTORE8(arith, cached_call_state.gas_limit, cached_call_state.gas_used,
                                                   *cached_call_state.stack_ptr, *call_state_ptr->memory_ptr);
                    break;
                case OP_SLOAD:
                    error_code = CuEVM::operations::SLOAD(
                        arith, cached_call_state.gas_limit, cached_call_state.gas_used, *cached_call_state.stack_ptr,
                        *call_state_ptr->touch_state_ptr, *call_state_ptr->message_ptr);
                    break;
                case OP_SSTORE:
                    error_code = CuEVM::operations::SSTORE(
                        arith, cached_call_state.gas_limit, cached_call_state.gas_used, call_state_ptr->gas_refund,
                        *cached_call_state.stack_ptr, *call_state_ptr->touch_state_ptr, *call_state_ptr->message_ptr);
                    break;
                case OP_JUMP:
                    error_code = CuEVM::operations::JUMP(arith, cached_call_state.gas_limit, cached_call_state.gas_used,
                                                         cached_call_state.pc, *cached_call_state.stack_ptr,
                                                         *call_state_ptr->message_ptr);
                    break;
                case OP_JUMPI:

                    error_code = CuEVM::operations::JUMPI(arith, cached_call_state.gas_limit,
                                                          cached_call_state.gas_used, cached_call_state.pc,
                                                          *cached_call_state.stack_ptr, *call_state_ptr->message_ptr
#ifdef BUILD_LIBRARY
                                                          ,
                                                          simplified_trace_data_ptr
#endif
                    );

                    break;

                case OP_PC:
                    error_code = CuEVM::operations::PC(arith, cached_call_state.gas_limit, cached_call_state.gas_used,
                                                       cached_call_state.pc, *cached_call_state.stack_ptr);
                    break;

                case OP_MSIZE:
                    error_code =
                        CuEVM::operations::MSIZE(arith, cached_call_state.gas_limit, cached_call_state.gas_used,
                                                 *cached_call_state.stack_ptr, *call_state_ptr->memory_ptr);
                    break;

                case OP_GAS:
                    error_code = CuEVM::operations::GAS(arith, cached_call_state.gas_limit, cached_call_state.gas_used,
                                                        *cached_call_state.stack_ptr);
                    break;

                case OP_JUMPDEST:
                    error_code =
                        CuEVM::operations::JUMPDEST(arith, cached_call_state.gas_limit, cached_call_state.gas_used);
                    break;

                case OP_PUSH0:
                    error_code = CuEVM::operations::PUSH0(arith, cached_call_state.gas_limit,
                                                          cached_call_state.gas_used, *cached_call_state.stack_ptr);
                    break;

                case OP_CREATE:
                    child_call_state_ptr = nullptr;
                    error_code =
                        CuEVM::operations::CREATE(arith, *call_state_ptr, child_call_state_ptr, cached_call_state);
                    break;

                case OP_CALL:
                    child_call_state_ptr = nullptr;
                    error_code =
                        CuEVM::operations::CALL(arith, *call_state_ptr, child_call_state_ptr, cached_call_state);
                    break;

                case OP_CALLCODE:
                    child_call_state_ptr = nullptr;
                    error_code =
                        CuEVM::operations::CALLCODE(arith, *call_state_ptr, child_call_state_ptr, cached_call_state);
                    break;

                case OP_RETURN:
                    error_code = CuEVM::operations::RETURN(
                        arith, cached_call_state.gas_limit, cached_call_state.gas_used, *cached_call_state.stack_ptr,
                        *call_state_ptr->memory_ptr, *call_state_ptr->parent->last_return_data_ptr);
                    break;

                case OP_DELEGATECALL:
                    child_call_state_ptr = nullptr;
                    error_code = CuEVM::operations::DELEGATECALL(arith, *call_state_ptr, child_call_state_ptr,
                                                                 cached_call_state);
                    break;

                case OP_CREATE2:
                    child_call_state_ptr = nullptr;
                    error_code =
                        CuEVM::operations::CREATE2(arith, *call_state_ptr, child_call_state_ptr, cached_call_state);
                    break;

                case OP_STATICCALL:
                    child_call_state_ptr = nullptr;
                    error_code =
                        CuEVM::operations::STATICCALL(arith, *call_state_ptr, child_call_state_ptr, cached_call_state);
                    break;

                case OP_REVERT:
                    error_code = CuEVM::operations::REVERT(
                        arith, cached_call_state.gas_limit, cached_call_state.gas_used, *cached_call_state.stack_ptr,
                        *call_state_ptr->memory_ptr, *call_state_ptr->parent->last_return_data_ptr);
                    break;

                case OP_SELFDESTRUCT:
                    error_code = CuEVM::operations::SELFDESTRUCT(
                        arith, cached_call_state.gas_limit, cached_call_state.gas_used, *cached_call_state.stack_ptr,
                        *call_state_ptr->message_ptr, *call_state_ptr->touch_state_ptr,
                        *call_state_ptr->parent->last_return_data_ptr);
                    break;

                default:
                    if ((opcode >= 0xA0) && (opcode <= 0xA4))  // LOGX // not common
                    {
                        error_code = CuEVM::operations::LOGX(arith, cached_call_state.gas_limit,
                                                             cached_call_state.gas_used, *cached_call_state.stack_ptr,
                                                             *call_state_ptr->memory_ptr, *call_state_ptr->message_ptr,
                                                             *call_state_ptr->log_state_ptr, opcode);
                    } else
                        error_code = CuEVM::operations::INVALID();
                    break;
            }
        }

        // TODO: to see after calls
        // increase program counter
        cached_call_state.pc++;
        // call_state_ptr->pc = ++current_pc;  // current pc may be changed by pushx
        // call_state_ptr->pc++;
        // __SYNC_THREADS__

// #ifdef __CUDA_ARCH__
//         __SYNC_THREADS__
//         printf("idx %d after increase pc %d , call_state_ptr %p, call_state_ptr->trace_idx %d , depth %d\n",
//                threadIdx.x, call_state_ptr->pc, call_state_ptr, call_state_ptr->trace_idx, call_state_ptr->depth);
// #endif
#ifdef EIP_3155
        if (call_state_ptr->trace_idx > 0 || (call_state_ptr->trace_idx == 0 && call_state_ptr->depth == 1)) {
            tracer_ptr->finish_operation(arith, call_state_ptr->trace_idx, cached_call_state.gas_used,
                                         call_state_ptr->gas_refund
#ifdef EIP_3155_OPTIONAL
                                         ,
                                         error_code
#endif
            );
        }
#endif
#ifdef BUILD_LIBRARY
        if ((opcode <= OP_EXP || opcode >= OP_REVERT || opcode == OP_SSTORE) && opcode != 0) {
            simplified_trace_data_ptr->finish_operation(*cached_call_state.stack_ptr, error_code);
        }
#endif
        // #ifdef __CUDA_ARCH__
        //         printf("after finish_operation idx %d opcode %d error code %d\n", threadIdx.x, opcode, error_code);
        // #endif
        // all calls  + create
        if (opcode >= OP_CREATE && opcode <= OP_STATICCALL && opcode != OP_RETURN) {
            // #ifdef __CUDA_ARCH__
            //             printf("opcode %d error code %d\n", opcode, error_code);
            // #endif
            if (error_code == ERROR_SUCCESS) {
                cached_call_state.write_cache_to_state(arith, call_state_ptr);
                call_state_ptr = child_call_state_ptr;
                cached_call_state = cached_evm_call_state(arith, call_state_ptr);
                error_code = start_CALL(arith, cached_call_state);
#ifdef BUILD_LIBRARY
                simplified_trace_data_ptr->start_call(call_state_ptr->parent->pc, call_state_ptr->message_ptr);
#endif
            } else if (opcode == OP_CREATE || opcode == OP_CREATE2) {
                // Logic: when op_create or create2 does not succeed,
                // there is no start_CALL but do not revert parent contract:
                //   + A contract already exists at the destination address.
                //   + other (inside start_CALL)
                if (error_code == ERROR_MESSAGE_CALL_CREATE_CONTRACT_EXISTS) {
                    // bypass the below by setting error_code == ERROR_SUCCESS
                    printf("ERROR_MESSAGE_CALL_CREATE_CONTRACT_EXISTS\n");
                    error_code = ERROR_SUCCESS;
                    // setting address = 0 to the stack
                    bn_t create_output;
                    cgbn_set_ui32(arith.env, create_output, 0);
                    call_state_ptr->stack_ptr->push(arith, create_output);
                    call_state_ptr->message_ptr->copy_from(call_state_ptr->message_ptr_copy);
                    CuEVM::byte_array_t::reset_return_data(call_state_ptr->last_return_data_ptr);
                }
            }
        }
        // #ifdef __CUDA_ARCH__
        //         printf("after checking elseif create CREATE %d error_code: %d , depth %d\n", threadIdx.x, error_code,
        //                call_state_ptr->depth);
        // #endif
        if (error_code != ERROR_SUCCESS) {
            if ((error_code == ERROR_RETURN) && (call_state_ptr->message_ptr->call_type == OP_CREATE ||
                                                 call_state_ptr->message_ptr->call_type == OP_CREATE2)) {
                // TODO: finish create call add the contract to the state
                // printf("Create call\n");
                error_code = finish_CREATE(arith, cached_call_state);
            }

            if (call_state_ptr->depth == 1) {
                // TODO: finish transaction
                // printf("Finish transaction\n");
                // #ifdef __CUDA_ARCH__
                //                 printf(" call_state_ptr->depth == 1 finish call %d error_code: %d\n", threadIdx.x,
                //                 error_code);
                // #endif
                cached_call_state.write_cache_to_state(arith, call_state_ptr);
                finish_CALL(arith, error_code);
                finish_TRANSACTION(arith, error_code);
                return;
            } else {
                // TODO: finish call
                // printf("Finish call\n");
                cached_call_state.write_cache_to_state(arith, call_state_ptr);
                error_code |= finish_CALL(arith, error_code);
                cached_call_state = cached_evm_call_state(arith, call_state_ptr);
            }
        }
    }
}

__host__ __device__ int32_t evm_t::finish_TRANSACTION(ArithEnv &arith, int32_t error_code) {
    // sent the gas value to the block beneficiary
    bn_t gas_value;
    const evm_word_t *beneficiary = &(block_info_ptr->coin_base);
    // block_info_ptr->get_coin_base(arith, beneficiary);
#ifdef EIP_3155
    __ONE_THREAD_PER_INSTANCE(printf("finish_TRANSACTION %d error_code: %d\n", THREADIDX, error_code););
#endif
    if ((error_code == ERROR_RETURN) || (error_code == ERROR_REVERT)) {
        bn_t gas_left;
        // \f$T_{g} - g\f$
        cgbn_sub(arith.env, gas_left, call_state_ptr->gas_limit, call_state_ptr->gas_used);

        // if return add the refund gas
        if (error_code == ERROR_RETURN) {
            bn_t capped_refund_gas;
            // \f$g/5\f$
            cgbn_div_ui32(arith.env, capped_refund_gas, call_state_ptr->gas_used, 5);
            // min ( \f$g/5\f$, \f$R_{g}\f$)

            if (cgbn_compare(arith.env, capped_refund_gas, call_state_ptr->gas_refund) > 0) {
                cgbn_set(arith.env, capped_refund_gas, call_state_ptr->gas_refund);
            }
            // g^{*} = \f$T_{g} - g + min ( \f$g/5\f$, \f$R_{g}\f$)\f$
            cgbn_add(arith.env, gas_value, gas_left, capped_refund_gas);
        } else {
            cgbn_set(arith.env, gas_value, gas_left);
        }
        bn_t send_back_gas;
        cgbn_mul(arith.env, send_back_gas, gas_value, gas_price);
        // add to sender balance g^{*}
        bn_t sender_balance;
        // bn_t sender_address;
        // send back the gas left and gas refund to the sender
        // transaction_ptr->get_sender(arith, sender_address);
        // deduct transaction value; TODO this probably should be done at some
        // other place _transaction->get_value(tx_value); cgbn_sub(arith.env,
        // sender_balance, sender_balance, tx_value); the gas value for the
        // beneficiary is \f$T_{g} - g^{*}\f$
        cgbn_sub(arith.env, gas_value, call_state_ptr->gas_limit, gas_value);
        // TODO: to see if true
        // gas used by the entire transaction save in the parent
        cgbn_set(arith.env, call_state_ptr->parent->gas_used, gas_value);
        cgbn_mul(arith.env, gas_value, gas_value, gas_priority_fee);

        // update the transaction state
        if (error_code == ERROR_RETURN) {
            call_state_ptr->parent->update(arith, *call_state_ptr);
        }
        // sent the value of unused gas to the sender
        call_state_ptr->parent->touch_state_ptr->get_balance(arith, &transaction_ptr->sender, sender_balance);
        cgbn_add(arith.env, sender_balance, sender_balance, send_back_gas);
        call_state_ptr->parent->touch_state_ptr->set_balance(arith, &transaction_ptr->sender, sender_balance);

        // set the eror code for a succesfull transaction
        status = error_code;
    } else {  // TODO: do we consider gas_priority_fee in revert?
        cgbn_set(arith.env, call_state_ptr->parent->gas_used, call_state_ptr->gas_limit);
        // cgbn_mul(arith.env, gas_value, cached_call_state.gas_limit, gas_priority_fee);
        // set z to the given error or 1 TODO: 1 in YP
#ifdef EIP_3155
        __ONE_THREAD_PER_INSTANCE(printf("finish_TRANSACTION %d error_code: %d\n", THREADIDX, error_code););
        print_bnt(arith, gas_value);
        print_bnt(arith, call_state_ptr->gas_limit);
        print_bnt(arith, call_state_ptr->gas_used);
        print_bnt(arith, gas_priority_fee);

#endif
        status = error_code;
    }
    // send the gas value to the beneficiary
    if (cgbn_compare_ui32(arith.env, gas_value, 0) > 0) {
        bn_t beneficiary_balance;
        call_state_ptr->parent->touch_state_ptr->get_balance(arith, beneficiary, beneficiary_balance);
        cgbn_add(arith.env, beneficiary_balance, beneficiary_balance, gas_value);
        call_state_ptr->parent->touch_state_ptr->set_balance(arith, beneficiary, beneficiary_balance);
    }

    CuEVM::evm_call_state_t *parent_call_state_ptr = call_state_ptr->parent;
    delete call_state_ptr;
    call_state_ptr = parent_call_state_ptr;
    // call_state_local = evm_call_state_t(arith, call_state_ptr);
// #ifdef EIP_3155
//     // printing debug when enabling tracer.
//     printf("call_state_ptr allocated %p, threadid %d\n", call_state_ptr, THREADIDX);
//     // transaction_ptr->print();
//     call_state_ptr->print(arith);
//     printf("call_state_local %d\n", THREADIDX);
//     call_state_local.print(arith);
// #endif
#ifdef EIP_3155
    tracer_ptr->finish_transaction(arith, *call_state_ptr->last_return_data_ptr, call_state_ptr->gas_used, status);
#endif
    // update the final world state : TODO combine both
    // __SYNC_THREADS__
    this->world_state.update(arith, call_state_ptr->touch_state_ptr->get_state());
    this->world_state.serialize_data(arith, serialized_worldstate_data_ptr);
    // printf("updated final world state\n");

    return status;
}

__host__ __device__ int32_t evm_t::finish_CALL(ArithEnv &arith, int32_t error_code) {
    bn_t child_success;
    // set the child call to failure
    cgbn_set_ui32(arith.env, child_success, 0);

    // #ifdef __CUDA_ARCH__
    //     printf("evm_t::finish_CALL %d error_code: %d\n", threadIdx.x, error_code);
    // #endif
    // if the child call return from normal halting
    // no errors
    // #ifdef EIP_3155
    // printf(" finish_CALL error_code: %d %d\n", error_code, THREADIDX);
    // #endif
    if ((error_code == ERROR_RETURN) || (error_code == ERROR_REVERT) || (error_code == ERROR_INSUFFICIENT_FUNDS) ||
        (error_code == ERROR_MESSAGE_CALL_CREATE_NONCE_EXCEEDED) || error_code == ERROR_MESSAGE_CALL_DEPTH_EXCEEDED) {
        // give back the gas left from the child computation
        bn_t gas_left;
        cgbn_sub(arith.env, gas_left, call_state_ptr->gas_limit, call_state_ptr->gas_used);

        cgbn_sub(arith.env, call_state_ptr->parent->gas_used, call_state_ptr->parent->gas_used, gas_left);
        // #ifdef __CUDA_ARCH__
        //         printf("finish_CALL %d gas_left: ", threadIdx.x);
        //         print_bnt(arith, gas_left);
        // #endif
        // if is a succesfull call
        if (error_code == ERROR_RETURN) {
            // update the parent state with the states of the child
            call_state_ptr->parent->update(arith, *call_state_ptr);
            // sum the refund gas
            cgbn_add(arith.env, call_state_ptr->parent->gas_refund, call_state_ptr->parent->gas_refund,
                     call_state_ptr->gas_refund);
            // for CALL operations set the child success to 1
            cgbn_set_ui32(arith.env, child_success, 1);
            // if CREATEX operation, set the address of the contract
            if ((call_state_ptr->message_ptr->get_call_type() == OP_CREATE) ||
                (call_state_ptr->message_ptr->get_call_type() == OP_CREATE2)) {
                call_state_ptr->message_ptr->get_recipient(arith, child_success);
            }
        }
    }
#ifdef BUILD_LIBRARY
    simplified_trace_data_ptr->finish_call((error_code == ERROR_RETURN));
#endif
    // printf("end finish_CALL before set warm account %d idx %d\n", error_code, THREADIDX);
    // #ifdef __CUDA_ARCH__
    //     printf("evm_t::finish_CALL %d before setting warm accounts error_code: %d\n", threadIdx.x, error_code);
    // #endif
    // warm the sender and receiver regardless of revert
    // bn_t sender, receiver;
    // call_state_ptr->message_ptr->get_sender(arith, sender);
    call_state_ptr->parent->touch_state_ptr->set_warm_account(arith, &call_state_ptr->message_ptr->sender);

    // call_state_ptr->message_ptr->get_recipient(arith, receiver);
    call_state_ptr->parent->touch_state_ptr->set_warm_account(arith, &call_state_ptr->message_ptr->recipient);
    // #ifdef __CUDA_ARCH__
    //     printf("evm_t::finish_CALL %d after setting warm accounts error_code: %d\n", threadIdx.x, error_code);
    // #endif
    if (call_state_ptr->depth > 1 && error_code != ERROR_RETURN && error_code != ERROR_REVERT) {
        // abnormal halting where return data ptr is not handled, need to reset
        // it
        // if (call_state_ptr->parent->last_return_data_ptr != nullptr)
        //     delete call_state_ptr->parent->last_return_data_ptr;
        // call_state_ptr->parent->last_return_data_ptr = new CuEVM::evm_return_data_t();
        CuEVM::byte_array_t::reset_return_data(call_state_ptr->parent->last_return_data_ptr);
    }
    // printf("end finish_CALL after checking and allocating return data: %d idx %d\n", error_code, THREADIDX);
    // get the memory offset and size of the return data
    // in the parent memory
    bn_t ret_offset, ret_size;
    call_state_ptr->message_ptr->get_return_data_offset(arith, ret_offset);
    call_state_ptr->message_ptr->get_return_data_size(arith, ret_size);
    // reset the error code for the parent
    error_code = ERROR_SUCCESS;
    // #ifdef __CUDA_ARCH__
    //     printf("evm_t::finish_CALL %d after getting return data error_code: %d\n", threadIdx.x, error_code);
    // #endif
    if (call_state_ptr->depth > 1) {
        // push the result in the parent stack
        error_code |= call_state_ptr->parent->stack_ptr->push(arith, child_success);

        // set the parent memory with the return data

        // write the return data in the memory
        error_code |= call_state_ptr->parent->memory_ptr->set(arith, *call_state_ptr->parent->last_return_data_ptr,
                                                              ret_offset, ret_size);

        // change the call state to the parent
        CuEVM::evm_call_state_t *parent_call_state_ptr = call_state_ptr->parent;
        delete call_state_ptr;

        call_state_ptr = parent_call_state_ptr;
        //         call_state_local = evm_call_state_t(arith, call_state_ptr);
        // #ifdef EIP_3155
        //         // printing debug when enabling tracer.
        //         printf("call_state_ptr allocated %p, threadid %d\n", call_state_ptr, THREADIDX);
        //         // transaction_ptr->print();
        //         call_state_ptr->print(arith);
        //         printf("call_state_local %d\n", THREADIDX);
        //         call_state_local.print(arith);
        // #endif
        // copy back the shadow message_call t to shared memory
        call_state_ptr->message_ptr->copy_from(call_state_ptr->message_ptr_copy);
        // cached_call_state = cached_evm_call_state(arith, call_state_ptr);
    }
    // #ifdef __CUDA_ARCH__
    //     printf("evm_t::finish_CALL %d before return error_code: %d\n", threadIdx.x, error_code);
    // #endif
    // printf("end finish_CALL error_code: %d idx %d\n", error_code, THREADIDX);
    return error_code;
}

__host__ __device__ int32_t evm_t::finish_CREATE(ArithEnv &arith, cached_evm_call_state &cached_call_state) {
    // TODO: increase sender nonce if the sender is a contract
    // to see if the contract is a contract
    // bn_t sender_address;
    // call_state_ptr->message_ptr->get_sender(arith, sender_address);
    CuEVM::account_t *sender_account = nullptr;
    call_state_ptr->parent->touch_state_ptr->get_account(arith, &call_state_ptr->message_ptr->sender, sender_account,
                                                         ACCOUNT_BYTE_CODE_FLAG);
    // if (sender_account->is_contract()) {
    //     printf("\naccount is contract\n");
    //     bn_t sender_nonce;
    //     call_state_ptr->parent->touch_state.get_nonce(arith, sender_address,
    //                                                   sender_nonce);
    //     cgbn_add_ui32(arith.env, sender_nonce, sender_nonce, 1);
    //     call_state_ptr->parent->touch_state.set_nonce(arith, sender_address,
    //                                                   sender_nonce);
    // }

    // bn_t sender_nonce;
    // call_state_ptr->parent->touch_state.get_nonce(arith, sender_address,
    // sender_nonce); cgbn_add_ui32(arith.env, sender_nonce, sender_nonce, 1);
    // call_state_ptr->parent->touch_state.set_nonce(arith, sender_address,
    // sender_nonce); compute the gas to deposit the contract

    bn_t code_size;
    cgbn_set_ui32(arith.env, code_size, call_state_ptr->parent->last_return_data_ptr->size);
    CuEVM::gas_cost::code_cost(arith, cached_call_state.gas_used, code_size);
    int32_t error_code = ERROR_SUCCESS;
    error_code |= CuEVM::gas_cost::has_gas(arith, cached_call_state.gas_limit, cached_call_state.gas_used);
    if (error_code == ERROR_SUCCESS) {
        // compute the address of the contract
        // bn_t contract_address;
        // call_state_ptr->message_ptr->get_recipient(arith, contract_address);
#ifdef EIP_3541
        uint8_t *code = call_state_ptr->parent->last_return_data_ptr->data;
#endif
        uint32_t code_size = call_state_ptr->parent->last_return_data_ptr->size;

        if (code_size <= CuEVM::max_code_size) {
#ifdef EIP_3541
            if ((code_size > 0) && (code[0] == 0xef)) {
                error_code = ERROR_CREATE_CODE_FIRST_BYTE_INVALID;
            }
#endif
            call_state_ptr->touch_state_ptr->set_code(arith, &call_state_ptr->message_ptr->recipient,
                                                      *call_state_ptr->parent->last_return_data_ptr);
        } else {
            error_code = ERROR_CREATE_CODE_SIZE_EXCEEDED;
        }
        // reset last return data after CREATE
        // if (call_state_ptr->parent->last_return_data_ptr != nullptr)
        //     delete call_state_ptr->parent->last_return_data_ptr;
        // call_state_ptr->parent->last_return_data_ptr = new CuEVM::evm_return_data_t();
        CuEVM::byte_array_t::reset_return_data(call_state_ptr->parent->last_return_data_ptr);
    }
    // if success, return ERROR_RETURN to continue finish call
    return error_code ? error_code : ERROR_RETURN;
}

__host__ int32_t get_evm_instances(ArithEnv &arith, evm_instance_t *&evm_instances, const cJSON *test_json,
                                   uint32_t &num_instances, int32_t managed) {
    // get the world state
    CuEVM::state_t *world_state_data_ptr = nullptr;
    const cJSON *world_state_json = NULL;  // the json for the world state
    // get the world state json
    if (cJSON_IsObject(test_json))
        world_state_json = cJSON_GetObjectItemCaseSensitive(test_json, "pre");
    else if (cJSON_IsArray(test_json))
        world_state_json = test_json;
    else
        return 1;
    if (managed)
        CUDA_CHECK(cudaMallocManaged(&world_state_data_ptr, sizeof(CuEVM::state_t)));
    else
        world_state_data_ptr = new CuEVM::state_t();

    world_state_data_ptr->from_json(world_state_json, managed);

    // get the block info
    CuEVM::block_info_t *block_info_ptr = nullptr;
    CuEVM::get_block_info(block_info_ptr, test_json, managed);

    // get the transaction
    CuEVM::evm_transaction_t *transactions_ptr = nullptr;
    uint32_t num_transactions = 0;
    CuEVM::transaction::get_transactions(arith, transactions_ptr, test_json, num_transactions, managed,
                                         world_state_data_ptr);

    // generate the evm instances

    if (managed)
        CUDA_CHECK(cudaMallocManaged(&evm_instances, num_transactions * sizeof(evm_instance_t)));
    else
        evm_instances = new evm_instance_t[num_transactions];
    for (uint32_t index = 0; index < num_transactions; index++) {
        evm_instances[index].world_state_data_ptr = world_state_data_ptr;
        evm_instances[index].block_info_ptr = block_info_ptr;
        evm_instances[index].transaction_ptr = &transactions_ptr[index];
        if (managed == 0) {
            evm_instances[index].touch_state_data_ptr = new CuEVM::state_access_t();
            evm_instances[index].log_state_ptr = new CuEVM::log_state_data_t();
            evm_instances[index].return_data_ptr = new CuEVM::evm_return_data_t();
#ifdef EIP_3155
            evm_instances[index].tracer_ptr = new CuEVM::utils::tracer_t();
#endif
        } else {
            CuEVM::state_access_t *access_state = new CuEVM::state_access_t();
            CUDA_CHECK(cudaMallocManaged(&evm_instances[index].touch_state_data_ptr, sizeof(CuEVM::state_access_t)));
            memcpy(evm_instances[index].touch_state_data_ptr, access_state, sizeof(CuEVM::state_access_t));
            delete access_state;
            CuEVM::log_state_data_t *log_state = new CuEVM::log_state_data_t();
            CUDA_CHECK(cudaMallocManaged(&evm_instances[index].log_state_ptr, sizeof(CuEVM::log_state_data_t)));
            memcpy(evm_instances[index].log_state_ptr, log_state, sizeof(CuEVM::log_state_data_t));
            delete log_state;
            CuEVM::evm_return_data_t *return_data = new CuEVM::evm_return_data_t();
            CUDA_CHECK(cudaMallocManaged(&evm_instances[index].return_data_ptr, sizeof(CuEVM::evm_return_data_t)));
            memcpy(evm_instances[index].return_data_ptr, return_data, sizeof(CuEVM::evm_return_data_t));
            delete return_data;

            CuEVM::EccConstants *ecc_constants_ptr = new CuEVM::EccConstants();
            CUDA_CHECK(cudaMallocManaged(&evm_instances[index].ecc_constants_ptr, sizeof(CuEVM::EccConstants)));
            memcpy(evm_instances[index].ecc_constants_ptr, ecc_constants_ptr, sizeof(CuEVM::EccConstants));
            delete ecc_constants_ptr;
#ifdef EIP_3155
            CuEVM::utils::tracer_t *tracer = new CuEVM::utils::tracer_t();
            CUDA_CHECK(cudaMallocManaged(&evm_instances[index].tracer_ptr, sizeof(CuEVM::utils::tracer_t)));
            memcpy(evm_instances[index].tracer_ptr, tracer, sizeof(CuEVM::utils::tracer_t));
            delete tracer;
#endif

            CuEVM::serialized_worldstate_data *serialized_worldstate_data = new CuEVM::serialized_worldstate_data();
            CUDA_CHECK(cudaMallocManaged(&evm_instances[index].serialized_worldstate_data_ptr,
                                         sizeof(CuEVM::serialized_worldstate_data)));
            memcpy(evm_instances[index].serialized_worldstate_data_ptr, serialized_worldstate_data,
                   sizeof(CuEVM::serialized_worldstate_data));
            delete serialized_worldstate_data;

            CuEVM::utils::simplified_trace_data *simplified_trace_data = new CuEVM::utils::simplified_trace_data();
            CUDA_CHECK(cudaMallocManaged(&evm_instances[index].simplified_trace_data_ptr,
                                         sizeof(CuEVM::utils::simplified_trace_data)));
            memcpy(evm_instances[index].simplified_trace_data_ptr, simplified_trace_data,
                   sizeof(CuEVM::utils::simplified_trace_data));
            delete simplified_trace_data;
        }
    }
    num_instances = num_transactions;
    return ERROR_SUCCESS;
}

__host__ void free_evm_instances(evm_instance_t *&evm_instances, uint32_t num_instances, int32_t managed) {
    if (managed == 0) {
        delete evm_instances[0].world_state_data_ptr;
        delete evm_instances[0].block_info_ptr;
        for (uint32_t index = 0; index < num_instances; index++) {
            delete evm_instances[index].touch_state_data_ptr;
            delete evm_instances[index].log_state_ptr;
            delete evm_instances[index].return_data_ptr;
#ifdef EIP_3155
            delete evm_instances[index].tracer_ptr;
#endif
        }
        delete[] evm_instances[0].transaction_ptr;
        delete[] evm_instances;
    } else {
        CUDA_CHECK(cudaFree(evm_instances[0].world_state_data_ptr));
        CUDA_CHECK(cudaFree(evm_instances[0].block_info_ptr));
        for (uint32_t index = 0; index < num_instances; index++) {
            // CUDA_CHECK(cudaFree(evm_instances[index].access_state_data_ptr));
            CUDA_CHECK(cudaFree(evm_instances[index].touch_state_data_ptr));
            CUDA_CHECK(cudaFree(evm_instances[index].log_state_ptr));
            CUDA_CHECK(cudaFree(evm_instances[index].return_data_ptr));
#ifdef EIP_3155
            CUDA_CHECK(cudaFree(evm_instances[index].tracer_ptr));
#endif
        }
        CUDA_CHECK(cudaFree(evm_instances[0].transaction_ptr));
        CUDA_CHECK(cudaFree(evm_instances));
    }
}

}  // namespace CuEVM
// todo|: make a vector o functions global constants so you can call them
