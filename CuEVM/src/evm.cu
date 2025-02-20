#include <CuEVM/evm.cuh>
#include <cassert>

namespace CuEVM {

// define the kernel function
__global__ void kernel_evm_multiple_instances(CuEVM::transaction::TransactionList *transaction_list_ptr, uint32_t count,
                                              bool copy_state_data) {
    int32_t instance = blockIdx.x * blockDim.x + threadIdx.x;
    if (instance >= count) return;

    CuEVM::evm_t evm = CuEVM::evm_t(transaction_list_ptr);
#ifdef BUILD_LIBRARY
    // printf("global_simplified_trace: %p\n", global_simplified_trace);
    // assert(global_simplified_trace != nullptr);
    // printf("transaction_list_ptr: %p\n", transaction_list_ptr);
    // if (THREADIDX == 1) transaction_list_ptr->print();
#endif
    cached_evm_call_context cached_call_state(evm.call_state_ptr);

    evm.run(cached_call_state, copy_state_data);

#ifdef EIP_3155
    // if (instance == 0) {
    //     evm.tracer_ptr->print_err();
    // }
    __syncthreads();
    if (instance == 1) {
        printf("\n\ninstance 10\n\n");
        evm.tracer_ptr->print_err();
    }
#endif
}

__device__ evm_t::evm_t(CuEVM::transaction::TransactionList *transaction_list_ptr)
    : transaction_list_ptr(transaction_list_ptr) {
    // printf("evm_t constructor\n");
    call_state_ptr = memory_pool::get_call_context(0);
    // CuEVM::evm_call_context_t *root_call_state_ptr = memory_pool::get_call_context(INSTANCE_GLOBAL_IDX);
    uint8_t *byte_code = nullptr;
    uint32_t byte_code_size = 0;
    uint8_t *call_data = &transaction_list_ptr->call_data[transaction_list_ptr->call_data_offset[INSTANCE_GLOBAL_IDX]];
    uint32_t call_data_size = transaction_list_ptr->call_data_size[INSTANCE_GLOBAL_IDX];

    CuEVM::evm_stack_t *stack_ptr = memory_pool::get_stack(0);
    // new CuEVM::evm_stack_t(CuEVM::memory_pool::global_memory_pool->stack_base);

    CuEVM::evm_memory_t *memory_ptr = memory_pool::get_memory(0);

    // new CuEVM::evm_memory_t();  // memory_pool::global_memory_pool->get_memory(threadIdx.x);
    if (transaction_list_ptr->type == SPECIAL_CREATE_TRANSACTION_TYPE) {
        uint32_t sender_nonce_uint = CuEVM::global_state_db_ptr->get_nonce(&transaction_list_ptr->sender);
        evm_word_t sender_nonce(sender_nonce_uint);

        CuEVM::utils::get_contract_address_create(&transaction_list_ptr->to, &transaction_list_ptr->sender,
                                                  &sender_nonce);
        // special case ? Tests allow create to acc with storage
        // TODO: simplify this

        transaction_list_ptr->to.print();
        if (!CuEVM::global_state_db_ptr->is_empty_create(&transaction_list_ptr->to)) {
            // todo: return error code
            return;
        }

        call_state_ptr->initiate_values(1, transaction_list_ptr->gas_limit[INSTANCE_GLOBAL_IDX], stack_ptr, memory_ptr,
                                        transaction_list_ptr->sender, transaction_list_ptr->to,
                                        transaction_list_ptr->to, transaction_list_ptr->value[INSTANCE_GLOBAL_IDX],
                                        OP_CREATE, call_data, call_data_size, call_data, call_data_size);
    } else {
        byte_code = global_state_db_ptr->get_code(byte_code_size, &transaction_list_ptr->to);

        call_state_ptr->initiate_values(1, transaction_list_ptr->gas_limit[INSTANCE_GLOBAL_IDX], stack_ptr, memory_ptr,
                                        transaction_list_ptr->sender, transaction_list_ptr->to,
                                        transaction_list_ptr->to, transaction_list_ptr->value[INSTANCE_GLOBAL_IDX],
                                        OP_CALL, call_data, call_data_size, byte_code, byte_code_size);
    }
    // charge gas and validate balance
    CuEVM::gas_t gas_intrinsic;
    CuEVM::gas_cost::transaction_intrinsic_gas(transaction_list_ptr, gas_intrinsic);
    call_state_ptr->gas_used = gas_intrinsic;
    // deduct upfront cost
    evm_word_t upfront_cost = transaction_list_ptr->gas_limit[INSTANCE_GLOBAL_IDX];

    uint256_mul(&upfront_cost, &upfront_cost, &transaction_list_ptr->gas_price);
    global_state_db_ptr->deduct_balance_sender(&transaction_list_ptr->sender, &upfront_cost);
    // global_state_db_ptr->set_warm_account(&global_block_info->coin_base);
    // printf("\n\ncall state ptr created %p\n\n", call_state_ptr);
    // call_state_ptr->print();
#ifdef EIP_3155
    this->tracer_ptr = new CuEVM::utils::tracer_t();
#endif
}

__device__ evm_t::~evm_t() {
    call_state_ptr = nullptr;
    transaction_list_ptr = nullptr;
#ifdef EIP_3155
    tracer_ptr = nullptr;
#endif
}

__device__ int32_t evm_t::start_CALL(cached_evm_call_context &cached_call_state) {
    // printf("Start call sender receipient %d code size %d\n", THREADIDX, call_state_ptr->byte_code_size);
    // call_state_ptr->from.print();
    // call_state_ptr->to.print();

    const evm_word_t *sender = &call_state_ptr->from;
    const evm_word_t *recipient = &call_state_ptr->to;

    int32_t error_code =
        (((uint256_cmp_word(&call_state_ptr->value, 0) > 0) && (call_state_ptr->call_type != OP_DELEGATECALL))
             ? global_state_db_ptr->transfer(sender, recipient, &call_state_ptr->value, call_state_ptr->snapshot_state)
             : ERROR_SUCCESS);

    if (error_code != ERROR_SUCCESS) return error_code;
    // printf("After transfer\n");
    // warmup the accounts

    // Go-ethereum: check depth > 1024 before increase -> depth > 1025 after increase
    // test: stSelfBalance/diffPlaces.json
    error_code |= call_state_ptr->depth > CuEVM::max_depth + 1 ? ERROR_MESSAGE_CALL_DEPTH_EXCEEDED : ERROR_SUCCESS;
    // Dont use account ptr here, byte_code already set

    if (call_state_ptr->byte_code_size == 0) {
        if (call_state_ptr->to.is_precompile()) {
            printf("precompile %d \n", call_state_ptr->to.words[0]);
            switch (call_state_ptr->to.words[0]) {
                case 0x01:
                    return CuEVM::precompile_operations::operation_ecRecover(
                        CuEVM::memory_pool::ecc_constants_ptr, cached_call_state.gas_limit, cached_call_state.gas_used,
                        call_state_ptr);
                    break;
                case 0x02:
                    return CuEVM::precompile_operations::operation_SHA256(cached_call_state.gas_limit,
                                                                          cached_call_state.gas_used, call_state_ptr);
                case 0x03:
                    return CuEVM::precompile_operations::operation_RIPEMD160(
                        cached_call_state.gas_limit, cached_call_state.gas_used, call_state_ptr);
                case 0x04:
                    return CuEVM::precompile_operations::operation_IDENTITY(cached_call_state.gas_limit,
                                                                            cached_call_state.gas_used, call_state_ptr);
                case 0x05:
                    return CuEVM::precompile_operations::operation_MODEXP(cached_call_state.gas_limit,
                                                                          cached_call_state.gas_used, call_state_ptr);
                case 0x06:
                    return CuEVM::precompile_operations::operation_ecAdd(CuEVM::memory_pool::ecc_constants_ptr,
                                                                         cached_call_state.gas_limit,
                                                                         cached_call_state.gas_used, call_state_ptr);
                case 0x07:
                    return CuEVM::precompile_operations::operation_ecMul(CuEVM::memory_pool::ecc_constants_ptr,
                                                                         cached_call_state.gas_limit,
                                                                         cached_call_state.gas_used, call_state_ptr);
                case 0x08:
                    return CuEVM::precompile_operations::operation_ecPairing(
                        CuEVM::memory_pool::ecc_constants_ptr, cached_call_state.gas_limit, cached_call_state.gas_used,
                        call_state_ptr);
                case 0x09:
                    return CuEVM::precompile_operations::operation_BLAKE2(cached_call_state.gas_limit,
                                                                          cached_call_state.gas_used, call_state_ptr);
                case 0x0a:
                    return ERROR_RETURN;
                default:
                    return ERROR_RETURN;
                    break;
            }
        } else {
            // operation stop
            // TODO: fix this
            // CuEVM::byte_array_t::reset_return_data(call_state_ptr->return_data_ptr);
            call_state_ptr->dynamic_ret_size = 0;

            return ERROR_RETURN;
        }
    }

    //    }

    return error_code;
}
// __device__ void evm_t::run() {
//     cached_evm_call_context cached_call_state(call_state_ptr);
//     run(cached_call_state);
// }

__device__ void evm_t::run(cached_evm_call_context &cached_call_state, bool copy_state_data) {
#ifdef BUILD_LIBRARY
    global_simplified_trace[INSTANCE_GLOBAL_IDX].start_call(0, call_state_ptr);  // pc is 0?
#endif

    int32_t error_code = start_CALL(cached_call_state);
    // printf("error_code start_call: %d\n", error_code);
    if (error_code != ERROR_SUCCESS) {
#ifdef BUILD_LIBRARY
        global_simplified_trace[INSTANCE_GLOBAL_IDX].finish_call(0);
#endif
        return;  // finish call
    }
    uint8_t opcode;
#ifdef BUILD_LIBRARY
    uint32_t pc_src;
#endif
    CuEVM::evm_call_context_t *child_call_state_ptr = nullptr;
    while (true) {
        opcode = ((cached_call_state.pc < cached_call_state.byte_code_size)
                      ? cached_call_state.byte_code_data[cached_call_state.pc]
                      : OP_STOP);

#ifdef EIP_3155

        tracer_ptr->start_operation(cached_call_state.pc, opcode, call_state_ptr->memory_ptr,
                                    cached_call_state.stack_ptr, call_state_ptr->depth, nullptr,
                                    cached_call_state.gas_limit, cached_call_state.gas_used);

#endif
        // if (INSTANCE_GLOBAL_IDX == 0) {
        //     printf("\nInstance %d, pc: %d opcode: %d, depth %d, memsize %d stacksize %d gas_limit %lu gas_used
        //     %lu\n",
        //            INSTANCE_GLOBAL_IDX, cached_call_state.pc, opcode, call_state_ptr->depth,
        //            call_state_ptr->memory_ptr->size, cached_call_state.stack_ptr->stack_offset,
        //            cached_call_state.gas_limit, cached_call_state.gas_used);

        //     // printf("\n\n");
        //     // cached_call_state.stack_ptr->print();
        // }
        // if (INSTANCE_GLOBAL_IDX == 1) {
        //     printf("\nIdx %d, pc: %d op: %d, depth %d, msize %d stksze %d gs_lmit %lu g_used %lu\n ",
        //            INSTANCE_GLOBAL_IDX, cached_call_state.pc, opcode, call_state_ptr->depth,
        //            call_state_ptr->memory_ptr->size, cached_call_state.stack_ptr->stack_offset,
        //            cached_call_state.gas_limit, cached_call_state.gas_used);
        //     // call_state_ptr->memory_ptr->print();
        //     // printf("\n\n");
        //     // cached_call_state.stack_ptr->print();
        //     // if (call_state_ptr->memory_ptr->size <= 64) {
        //     //     call_state_ptr->memory_ptr->print();
        //     // }
        // }

#ifdef BUILD_LIBRARY
        // comparison, arithmetic, revert/invalid
        if ((opcode <= OP_EXP || opcode >= OP_REVERT || opcode == OP_SSTORE) && opcode != 0) {
            global_simplified_trace[INSTANCE_GLOBAL_IDX].start_operation(cached_call_state.pc, opcode,
                                                                         *cached_call_state.stack_ptr);
        }
        if (opcode == OP_JUMPI) pc_src = cached_call_state.pc;
#endif

        if (((opcode & 0xF0) == 0x60) || ((opcode & 0xF0) == 0x70)) {
            evm_stack_t *stack_ptr = cached_call_state.stack_ptr;
            error_code =
                CuEVM::operations::PUSHX(cached_call_state.gas_limit, cached_call_state.gas_used, cached_call_state.pc,
                                         stack_ptr, call_state_ptr->byte_code, call_state_ptr->byte_code_size, opcode);
        } else if ((opcode & 0xF0) == 0x80)  // DUPX
        {
            error_code = CuEVM::operations::DUPX(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                 *cached_call_state.stack_ptr, opcode);
        } else if ((opcode & 0xF0) == 0x90)  // SWAPX
        {
            error_code = CuEVM::operations::SWAPX(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                  *cached_call_state.stack_ptr, opcode);
        } else {
            switch (opcode) {
                case OP_STOP:
                    error_code = CuEVM::operations::STOP(call_state_ptr);
                    break;
                case OP_ADD:
                    error_code = CuEVM::operations::ADD(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                        cached_call_state.stack_ptr);
                    break;
                case OP_MUL:
                    error_code = CuEVM::operations::MUL(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                        cached_call_state.stack_ptr);
                    break;
                case OP_SUB:
                    error_code = CuEVM::operations::SUB(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                        cached_call_state.stack_ptr);
                    break;
                case OP_DIV:
                    error_code = CuEVM::operations::DIV(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                        cached_call_state.stack_ptr);
                    break;
                case OP_SDIV:
                    error_code = CuEVM::operations::SDIV(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                         cached_call_state.stack_ptr);
                    break;
                case OP_MOD:
                    error_code = CuEVM::operations::MOD(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                        cached_call_state.stack_ptr);
                    break;
                case OP_SMOD:
                    error_code = CuEVM::operations::SMOD(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                         cached_call_state.stack_ptr);
                    break;
                case OP_ADDMOD:
                    error_code = CuEVM::operations::ADDMOD(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                           cached_call_state.stack_ptr);
                    break;
                case OP_MULMOD:
                    error_code = CuEVM::operations::MULMOD(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                           cached_call_state.stack_ptr);
                    break;
                case OP_EXP:
                    error_code = CuEVM::operations::EXP(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                        cached_call_state.stack_ptr);
                    break;
                case OP_SIGNEXTEND:
                    error_code = CuEVM::operations::SIGNEXTEND(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                               cached_call_state.stack_ptr);
                    break;
                case OP_LT:
#ifdef BUILD_LIBRARY
                    global_simplified_trace[INSTANCE_GLOBAL_IDX].record_distance(opcode, *cached_call_state.stack_ptr);
#endif
                    error_code = CuEVM::operations::LT(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                       cached_call_state.stack_ptr);
                    break;
                case OP_GT:
#ifdef BUILD_LIBRARY
                    global_simplified_trace[INSTANCE_GLOBAL_IDX].record_distance(opcode, *cached_call_state.stack_ptr);
#endif
                    error_code = CuEVM::operations::GT(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                       cached_call_state.stack_ptr);
                    break;
                case OP_SLT:
#ifdef BUILD_LIBRARY
                    global_simplified_trace[INSTANCE_GLOBAL_IDX].record_distance(opcode, *cached_call_state.stack_ptr);
#endif
                    error_code = CuEVM::operations::SLT(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                        cached_call_state.stack_ptr);
                    break;
                case OP_SGT:
#ifdef BUILD_LIBRARY
                    global_simplified_trace[INSTANCE_GLOBAL_IDX].record_distance(opcode, *cached_call_state.stack_ptr);
#endif
                    error_code = CuEVM::operations::SGT(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                        cached_call_state.stack_ptr);
                    break;
                case OP_EQ:
#ifdef BUILD_LIBRARY
                    global_simplified_trace[INSTANCE_GLOBAL_IDX].record_distance(opcode, *cached_call_state.stack_ptr);
#endif
                    error_code = CuEVM::operations::EQ(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                       cached_call_state.stack_ptr);
                    break;
                case OP_ISZERO:
                    error_code = CuEVM::operations::ISZERO(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                           cached_call_state.stack_ptr);
                    break;
                case OP_AND:
                    error_code = CuEVM::operations::AND(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                        cached_call_state.stack_ptr);
                    break;
                case OP_OR:
                    error_code = CuEVM::operations::OR(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                       cached_call_state.stack_ptr);
                    break;
                case OP_XOR:
                    error_code = CuEVM::operations::XOR(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                        cached_call_state.stack_ptr);
                    break;
                case OP_NOT:
                    error_code = CuEVM::operations::NOT(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                        cached_call_state.stack_ptr);
                    break;
                case OP_BYTE:
                    error_code = CuEVM::operations::BYTE(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                         cached_call_state.stack_ptr);
                    break;
                case OP_SHL:
                    error_code = CuEVM::operations::SHL(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                        cached_call_state.stack_ptr);
                    break;
                case OP_SHR:
                    error_code = CuEVM::operations::SHR(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                        cached_call_state.stack_ptr);
                    break;
                case OP_SAR:
                    error_code = CuEVM::operations::SAR(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                        cached_call_state.stack_ptr);
                    break;
                case OP_SHA3:
                    error_code = CuEVM::operations::SHA3(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                         *cached_call_state.stack_ptr, *call_state_ptr->memory_ptr);
                    break;
                case OP_ADDRESS:
                    error_code = CuEVM::operations::ADDRESS(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                            call_state_ptr);
                    break;
                case OP_BALANCE:
                    error_code = CuEVM::operations::BALANCE(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                            call_state_ptr);
                    break;
                case OP_ORIGIN:
                    error_code = CuEVM::operations::ORIGIN(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                           *cached_call_state.stack_ptr, transaction_list_ptr);
                    break;
                case OP_CALLER:
                    error_code = CuEVM::operations::CALLER(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                           call_state_ptr);
                    break;
                case OP_CALLVALUE:
                    error_code = CuEVM::operations::CALLVALUE(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                              call_state_ptr);
                    break;
                case OP_CALLDATALOAD:
                    error_code = CuEVM::operations::CALLDATALOAD(cached_call_state.gas_limit,
                                                                 cached_call_state.gas_used, call_state_ptr);
                    break;
                case OP_CALLDATASIZE:
                    error_code = CuEVM::operations::CALLDATASIZE(cached_call_state.gas_limit,
                                                                 cached_call_state.gas_used, call_state_ptr);
                    break;
                case OP_CALLDATACOPY:
                    error_code = CuEVM::operations::CALLDATACOPY(cached_call_state.gas_limit,
                                                                 cached_call_state.gas_used, call_state_ptr);
                    break;
                case OP_CODESIZE:
                    error_code = CuEVM::operations::CODESIZE(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                             call_state_ptr);
                    break;
                case OP_CODECOPY:
                    error_code = CuEVM::operations::CODECOPY(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                             call_state_ptr);
                    break;
                case OP_GASPRICE:
                    error_code = CuEVM::operations::GASPRICE(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                             *cached_call_state.stack_ptr, *global_block_info,
                                                             transaction_list_ptr);
                    break;
                case OP_EXTCODESIZE:
                    error_code = CuEVM::operations::EXTCODESIZE(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                                call_state_ptr);
                    break;
                case OP_EXTCODECOPY:
                    error_code = CuEVM::operations::EXTCODECOPY(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                                call_state_ptr);
                    break;
                case OP_RETURNDATASIZE:
                    error_code = CuEVM::operations::RETURNDATASIZE(cached_call_state.gas_limit,
                                                                   cached_call_state.gas_used, call_state_ptr);
                    break;
                case OP_RETURNDATACOPY:
                    error_code = CuEVM::operations::RETURNDATACOPY(cached_call_state.gas_limit,
                                                                   cached_call_state.gas_used, call_state_ptr);
                    break;
                case OP_EXTCODEHASH:
                    error_code = CuEVM::operations::EXTCODEHASH(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                                call_state_ptr);
                    break;
                case OP_BLOCKHASH:
                    error_code = CuEVM::operations::BLOCKHASH(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                              *cached_call_state.stack_ptr);
                    break;
                case OP_COINBASE:
                    error_code = CuEVM::operations::COINBASE(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                             *cached_call_state.stack_ptr);
                    break;
                case OP_TIMESTAMP:
                    error_code = CuEVM::operations::TIMESTAMP(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                              *cached_call_state.stack_ptr);
                    break;
                case OP_NUMBER:
                    error_code = CuEVM::operations::NUMBER(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                           *cached_call_state.stack_ptr);
                    break;
                case OP_DIFFICULTY:
                    error_code = CuEVM::operations::PREVRANDAO(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                               *cached_call_state.stack_ptr);
                    break;
                case OP_GASLIMIT:
                    error_code = CuEVM::operations::GASLIMIT(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                             *cached_call_state.stack_ptr);
                    break;
                case OP_CHAINID:
                    error_code = CuEVM::operations::CHAINID(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                            *cached_call_state.stack_ptr);
                    break;
                case OP_SELFBALANCE:
                    error_code = CuEVM::operations::SELFBALANCE(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                                call_state_ptr);
                    break;
                case OP_BASEFEE:
                    error_code = CuEVM::operations::BASEFEE(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                            *cached_call_state.stack_ptr);
                    break;
                case OP_POP:
                    error_code = CuEVM::operations::POP(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                        *cached_call_state.stack_ptr);
                    break;
                case OP_MLOAD:
                    error_code = CuEVM::operations::MLOAD(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                          *cached_call_state.stack_ptr, *call_state_ptr->memory_ptr);
                    break;
                case OP_MSTORE:
                    error_code = CuEVM::operations::MSTORE(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                           *cached_call_state.stack_ptr, *call_state_ptr->memory_ptr);
                    break;
                case OP_MSTORE8:
                    error_code = CuEVM::operations::MSTORE8(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                            *cached_call_state.stack_ptr, *call_state_ptr->memory_ptr);
                    break;
                case OP_SLOAD:
                    error_code =
                        CuEVM::operations::SLOAD(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                 *cached_call_state.stack_ptr, global_state_db_ptr, call_state_ptr);
                    break;
                case OP_SSTORE:
                    error_code = CuEVM::operations::SSTORE(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                           call_state_ptr->gas_refund, *cached_call_state.stack_ptr,
                                                           global_state_db_ptr, call_state_ptr);
                    break;
                case OP_JUMP:
                    error_code =
                        CuEVM::operations::JUMP(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                cached_call_state.pc, *cached_call_state.stack_ptr, call_state_ptr);
                    break;
                case OP_JUMPI:
                    error_code =
                        CuEVM::operations::JUMPI(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                 cached_call_state.pc, *cached_call_state.stack_ptr, call_state_ptr
#ifdef BUILD_LIBRARY
                                                 ,
                                                 &global_simplified_trace[INSTANCE_GLOBAL_IDX]
#endif
                        );

                    break;

                case OP_PC:
                    error_code = CuEVM::operations::PC(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                       cached_call_state.pc, *cached_call_state.stack_ptr);
                    break;

                case OP_MSIZE:
                    error_code = CuEVM::operations::MSIZE(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                          *cached_call_state.stack_ptr, *call_state_ptr->memory_ptr);
                    break;

                case OP_GAS:
                    error_code = CuEVM::operations::GAS(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                        *cached_call_state.stack_ptr);
                    break;

                case OP_JUMPDEST:
                    error_code = CuEVM::operations::JUMPDEST(cached_call_state.gas_limit, cached_call_state.gas_used);
                    break;

                case OP_PUSH0:
                    error_code = CuEVM::operations::PUSH0(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                          *cached_call_state.stack_ptr);
                    break;

                case OP_CREATE:
                    child_call_state_ptr = nullptr;
                    error_code = CuEVM::operations::CREATE(call_state_ptr, child_call_state_ptr, cached_call_state);
                    break;

                case OP_CALL:
                    child_call_state_ptr = nullptr;
                    error_code = CuEVM::operations::CALL(call_state_ptr, child_call_state_ptr, cached_call_state);
                    break;

                case OP_CALLCODE:
                    child_call_state_ptr = nullptr;
                    error_code = CuEVM::operations::CALLCODE(call_state_ptr, child_call_state_ptr, cached_call_state);
                    break;

                case OP_RETURN:
                    // TODO: fix this
                    error_code = CuEVM::operations::RETURN(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                           *cached_call_state.stack_ptr, call_state_ptr);

                    break;

                case OP_DELEGATECALL:
                    child_call_state_ptr = nullptr;
                    error_code =
                        CuEVM::operations::DELEGATECALL(call_state_ptr, child_call_state_ptr, cached_call_state);
                    break;

                case OP_CREATE2:
                    child_call_state_ptr = nullptr;
                    error_code = CuEVM::operations::CREATE2(call_state_ptr, child_call_state_ptr, cached_call_state);
                    break;

                case OP_STATICCALL:
                    child_call_state_ptr = nullptr;
                    error_code = CuEVM::operations::STATICCALL(call_state_ptr, child_call_state_ptr, cached_call_state);
                    break;

                case OP_REVERT:
                    // TODO: fix this
                    error_code = CuEVM::operations::REVERT(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                           *cached_call_state.stack_ptr, call_state_ptr);
                    break;

                case OP_SELFDESTRUCT:
                    // TODO: fix this
                    error_code =
                        CuEVM::operations::SELFDESTRUCT(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                        *cached_call_state.stack_ptr, call_state_ptr);
                    break;

                default:
                    if ((opcode >= 0xA0) && (opcode <= 0xA4))  // LOGX // not common
                    {
                        error_code = CuEVM::operations::LOGX(cached_call_state.gas_limit, cached_call_state.gas_used,
                                                             *cached_call_state.stack_ptr, call_state_ptr, opcode);
                    } else
                        error_code = CuEVM::operations::INVALID();
                    break;
            }
        }

        // TODO: to see after calls
        // increase program counter
        cached_call_state.pc++;
#ifdef EIP_3155
        // printf("finish operation, gas used %lu error_code %d\n", cached_call_state.gas_used, error_code);
        tracer_ptr->finish_operation(cached_call_state.gas_used, call_state_ptr->gas_refund);

#endif
#ifdef BUILD_LIBRARY
        if ((opcode <= OP_EXP || opcode >= OP_REVERT || opcode == OP_SSTORE) && opcode != 0) {
            global_simplified_trace[INSTANCE_GLOBAL_IDX].finish_operation(*cached_call_state.stack_ptr, error_code);
        }
#endif

        // all calls  + create
        if (opcode >= OP_CREATE && opcode <= OP_STATICCALL && opcode != OP_RETURN) {
            if (error_code == ERROR_SUCCESS) {
                cached_call_state.write_cache_to_state(call_state_ptr);

                call_state_ptr = child_call_state_ptr;
                cached_call_state = cached_evm_call_context(call_state_ptr);
                error_code = start_CALL(cached_call_state);
#ifdef BUILD_LIBRARY
                global_simplified_trace[INSTANCE_GLOBAL_IDX].start_call(call_state_ptr->parent->pc, call_state_ptr);
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
                    evm_word_t create_output = 0;
                    call_state_ptr->stack_ptr->push(create_output);
                    // TODO: fix this
                    // call_state_ptr->message_ptr->copy_from(call_state_ptr->message_ptr_copy);
                    // CuEVM::byte_array_t::reset_return_data(call_state_ptr->last_return_data_ptr);
                }
            }
        }

        if (error_code != ERROR_SUCCESS) {
            if ((error_code == ERROR_RETURN) &&
                (call_state_ptr->call_type == OP_CREATE || call_state_ptr->call_type == OP_CREATE2)) {
                // TODO: finish create call add the contract to the state
                // printf("Create call\n");
                error_code = finish_CREATE(cached_call_state);
            }
            // TODO: remove this, read cached state inside finish_CALL
            cached_call_state.write_cache_to_state(call_state_ptr);
            if (call_state_ptr->depth == 1) {
                finish_CALL(error_code);
                finish_TRANSACTION(error_code, copy_state_data);
                return;
            } else {
                // TODO: finish call
                // printf("Finish call\n");
                error_code |= finish_CALL(error_code);
                cached_call_state = cached_evm_call_context(call_state_ptr);
            }
        }
    }
}

__device__ int32_t evm_t::finish_TRANSACTION(int32_t error_code, bool copy_state_data) {
    // sent the gas value to the block beneficiary
    gas_t gas_value;
    const evm_word_t *beneficiary = &(global_block_info->coin_base);
    // block_info_ptr->get_coin_base(beneficiary);

    if ((error_code == ERROR_RETURN) || (error_code == ERROR_REVERT)) {
        gas_t gas_left;
        // \f$T_{g} - g\f$
        gas_left = call_state_ptr->gas_limit - call_state_ptr->gas_used;

        // if return add the refund gas
        if (error_code == ERROR_RETURN) {
            gas_t capped_refund_gas;
            // \f$g/5\f$
            capped_refund_gas = call_state_ptr->gas_used / 5;
            // min ( \f$g/5\f$, \f$R_{g}\f$)

            if (capped_refund_gas > call_state_ptr->gas_refund) {
                capped_refund_gas = call_state_ptr->gas_refund;
            }
            // g^{*} = \f$T_{g} - g + min ( \f$g/5\f$, \f$R_{g}\f$)\f$
            gas_value = gas_left + capped_refund_gas;
        } else {
            gas_value = gas_left;
        }
        gas_t send_back_gas;
        send_back_gas = gas_value * gas_price;
        // add to sender balance g^{*}
        evm_word_t *sender_balance;
        // bn_t sender_address;
        // send back the gas left and gas refund to the sender
        // transaction_ptr->get_sender(sender_address);
        // deduct transaction value; TODO this probably should be done at some
        // other place _transaction->get_value(tx_value); cgbn_sub(arith.env,
        // sender_balance, sender_balance, tx_value); the gas value for the
        // beneficiary is \f$T_{g} - g^{*}\f$
        gas_value = call_state_ptr->gas_limit - gas_value;
        // TODO: to see if true
        // gas used by the entire transaction save in the parent
        // TODO: fix this
        // call_state_ptr->parent->gas_used = gas_value;
        gas_value = gas_value * gas_priority_fee;

        // update the transaction state
        if (error_code == ERROR_RETURN) {
            // call_state_ptr->parent->update(*call_state_ptr);
        }
        // sent the value of unused gas to the sender
        // TODO: fix this
        // sender_balance = call_state_ptr->parent->state_db_ptr->get_balance(&transaction_list_ptr->sender);
        // uint256_add_word(sender_balance, sender_balance, send_back_gas);
        // call_state_ptr->parent->state_db_ptr->update_balance(&transaction_list_ptr->sender, sender_balance);

        // set the eror code for a succesfull transaction
        status = error_code;
    } else {
        // call_state_ptr->parent->gas_used = call_state_ptr->gas_limit;
        // cgbn_mul(arith.env, gas_value, cached_call_state.gas_limit, gas_priority_fee);
        // set z to the given error or 1 TODO: 1 in YP
        status = error_code;
    }
    // send the gas value to the beneficiary
    if (gas_value > 0) {
        // TODO: fix this
        // evm_word_t *beneficiary_balance = call_state_ptr->parent->state_db_ptr->get_balance(beneficiary);
        // uint256_add_word(beneficiary_balance, beneficiary_balance, gas_value);
        // call_state_ptr->parent->state_db_ptr->update_balance(beneficiary, beneficiary_balance);
    }

    // CuEVM::evm_call_context_t *parent_call_state_ptr = call_state_ptr->parent;

    // TODO: fix this
    // delete call_state_ptr;

    //  call_state_ptr = parent_call_state_ptr;

#ifdef EIP_3155
    tracer_ptr->finish_transaction(nullptr, call_state_ptr->gas_used, status);
#endif

#ifdef BUILD_LIBRARY
    // serialize data
    if (copy_state_data) {
        python_utils::serialize_state_data(&global_serialized_worldstate[INSTANCE_GLOBAL_IDX]);
    }
#endif
    // this->state_db_ptr->serialize_data(serialized_worldstate_data_ptr);
    // printf("updated final world state\n");

    return status;
}

__device__ int32_t evm_t::finish_CALL(int32_t error_code) {
    evm_word_t child_success = 0;
    // printf("finish_CALL thread %d, call_state_ptr %p\n", INSTANCE_GLOBAL_IDX, call_state_ptr);
    if ((error_code == ERROR_RETURN) || (error_code == ERROR_REVERT) || (error_code == ERROR_INSUFFICIENT_FUNDS) ||
        (error_code == ERROR_MESSAGE_CALL_CREATE_NONCE_EXCEEDED) || error_code == ERROR_MESSAGE_CALL_DEPTH_EXCEEDED) {
        // give back the gas left from the child computation
        gas_t gas_left = call_state_ptr->gas_limit - call_state_ptr->gas_used;
        // printf("gas left %lu\n", gas_left);
        // printf("gas used %lu\n", call_state_ptr->gas_used);
        // printf("gas limit %lu\n", call_state_ptr->gas_limit);
        // if (call_state_ptr->parent != nullptr) printf("parent gas used %lu\n", call_state_ptr->parent->gas_used);
        if (call_state_ptr->parent != nullptr) {
            call_state_ptr->parent->gas_used -= gas_left;
        }

        // if is a succesfull call
        if (error_code == ERROR_RETURN) {
            // update the parent state with the states of the child
            // call_state_ptr->parent->update(*call_state_ptr);
            // sum the refund gas
            // call_state_ptr->parent->gas_refund += call_state_ptr->gas_refund;
            // for CALL operations set the child success to 1
            child_success = 1;
            // if CREATEX operation, set the address of the contract
            if ((call_state_ptr->call_type == OP_CREATE) || (call_state_ptr->call_type == OP_CREATE2)) {
                // TODO: fix this
                child_success = call_state_ptr->to;
            }
        } else {
            // perform revert mechanism
        }
    }
#ifdef BUILD_LIBRARY
    global_simplified_trace[INSTANCE_GLOBAL_IDX].finish_call((error_code == ERROR_RETURN));
#endif

    uint32_t ret_dynamic_size = call_state_ptr->dynamic_ret_size;

    if (call_state_ptr->depth > 1 && error_code != ERROR_RETURN && error_code != ERROR_REVERT) {
        call_state_ptr->parent->dynamic_ret_size = 0;
        call_state_ptr->fixed_ret_size = 0;
    }

    if (error_code != ERROR_RETURN && call_state_ptr->snapshot_state != nullptr) {
        // printf("\n\nRevert to previous depth %d \n\n", call_state_ptr->depth - 1);
        call_state_ptr->revert();
    }

    if (call_state_ptr->depth > 1) {
        uint32_t ret_offset = call_state_ptr->fixed_ret_offset;
        uint32_t ret_size = call_state_ptr->fixed_ret_size;
        // change the call state to the parent
        CuEVM::evm_call_context_t *parent_call_state_ptr = call_state_ptr->parent;
        // printf("finish_CALL thread %d, call state ptr %p, parent call state ptr %p\n", INSTANCE_GLOBAL_IDX,
        //        call_state_ptr, parent_call_state_ptr);

        SnapshotState *snapshot_state = call_state_ptr->snapshot_state;
        // printf("Finish call, set parent snapshot state, current state %p\n", snapshot_state);
        if (snapshot_state != nullptr) {
            while (snapshot_state->next_state != nullptr) {
                // printf("current snapshot chain %p\n", snapshot_state);
                snapshot_state = snapshot_state->next_state;
            }
            snapshot_state->next_state = parent_call_state_ptr->snapshot_state;

            // Revert chain : new_parent -> chain -> old_parent
        }
        SnapshotState *new_parent_snapshot_state = CuEVM::memory_pool::get_snapshot_state();
        new_parent_snapshot_state->address = parent_call_state_ptr->to;
        new_parent_snapshot_state->storage_size = 0;
        new_parent_snapshot_state->touched_account_counts = 0;
        new_parent_snapshot_state->diff_account_counts = 0;
        new_parent_snapshot_state->preallocated_offset =
            CuEVM::memory_pool::global_memory_pool->snapshot_slot_counts[INSTANCE_GLOBAL_IDX];

        new_parent_snapshot_state->next_state =
            call_state_ptr->snapshot_state ? call_state_ptr->snapshot_state : parent_call_state_ptr->snapshot_state;
        parent_call_state_ptr->snapshot_state = new_parent_snapshot_state;

        // free memory
        if (call_state_ptr->depth > memory_pool_call_context_preallocate) {
            delete call_state_ptr;
        } else {
            call_state_ptr->clear();
        }

        // push the result in the parent stack
        error_code |= parent_call_state_ptr->stack_ptr->push(child_success);
        // Free previous memory first before copy return data to memory
        // write the return data in the memory
        parent_call_state_ptr->memory_ptr->grow(ret_offset + ret_size);
        // printf("return data size %d\n", ret_size);
        // printf("return data offset %d\n", ret_offset);
        // have to clear memory first before copy return data to memory // due to shared memory between depths
        parent_call_state_ptr->copy_return_data_to_memory(ret_offset, 0, ret_size);

        call_state_ptr = parent_call_state_ptr;
    }

    // printf("end finish_CALL error_code: %d idx %d\n", error_code, THREADIDX);
    return error_code;
}

__device__ int32_t evm_t::finish_CREATE(cached_evm_call_context &cached_call_state) {
    // TODO: increase sender nonce if the sender is a contract
    // to see if the contract is a contract
    // bn_t sender_address;
    // call_state_ptr->message_ptr->get_sender(sender_address);

    // TODO: fix this
    printf("finish_CREATE thread %d, call_state_ptr %p\n", INSTANCE_GLOBAL_IDX, call_state_ptr);

    CuEVM::gas_cost::code_cost(cached_call_state.gas_used, call_state_ptr->dynamic_ret_size);
    int32_t error_code = ERROR_SUCCESS;
    error_code |= CuEVM::gas_cost::has_gas(cached_call_state.gas_limit, cached_call_state.gas_used);
    uint32_t code_size = call_state_ptr->dynamic_ret_size;
    if (error_code == ERROR_SUCCESS && code_size > 0) {
#ifdef EIP_3541
        uint8_t *code = new uint8_t[code_size];
        call_state_ptr->parent->copy_return_data(code, 0, code_size);
#endif

        if (code_size <= CuEVM::max_code_size) {
#ifdef EIP_3541
            if ((code_size > 0) && (code[0] == 0xef)) {
                error_code = ERROR_CREATE_CODE_FIRST_BYTE_INVALID;
            }
#endif
            global_state_db_ptr->update_code(&call_state_ptr->to, code_size, code);
        } else {
            error_code = ERROR_CREATE_CODE_SIZE_EXCEEDED;
        }

        delete[] code;
    }
    // TODO check if neccessary
    call_state_ptr->dynamic_ret_size = 0;
    if (call_state_ptr->parent != nullptr) {
        call_state_ptr->parent->dynamic_ret_size = 0;
    }

    // if success, return ERROR_RETURN to continue finish call
    return error_code ? error_code : ERROR_RETURN;

    return ERROR_SUCCESS;
}

__host__ CuEVM::transaction::TransactionList *get_evm_instances(const cJSON *test_json, uint32_t &num_instances,
                                                                uint32_t &num_accounts, uint32_t clones) {
    // get the world state

    CuEVM::StateDb *state_db_ptr = nullptr;
    const cJSON *world_state_json = NULL;  // the json for the world state
    // get the world state json
    if (cJSON_IsObject(test_json))
        world_state_json = cJSON_GetObjectItemCaseSensitive(test_json, "pre");
    else if (cJSON_IsArray(test_json))
        world_state_json = test_json;
    else
        return nullptr;

    CuEVM::get_block_info(test_json);

    // get the transaction
    CuEVM::transaction::TransactionList *transaction_list_ptr = nullptr;
    uint32_t num_transactions = 0;
    uint32_t num_original_transactions = 0;

    CuEVM::transaction::get_transactions(transaction_list_ptr, test_json, num_transactions, clones);
    // num_original_transactions = num_transactions;
    // num_transactions *= clones;
    // generate the evm instances

    // CUDA_CHECK(cudaMallocManaged(&evm_instances, num_transactions * sizeof(evm_instance_t)));
    printf("num_transactions %d\n", num_transactions);

    // evm_instance_t *evm_instances = new evm_instance_t[num_transactions];

    CuEVM::StateDb::GPUfromJson(state_db_ptr, world_state_json, num_transactions, num_accounts);

    num_instances = num_transactions;
    return transaction_list_ptr;
}

__host__ void free_evm_instances(evm_instance_t *&evm_instances, uint32_t num_instances) {
    /*
    for (uint32_t index = 0; index < num_instances; index++) {
        delete evm_instances[index].log_state_ptr;
        delete evm_instances[index].log_state_ptr;
        delete evm_instances[index].return_data_ptr;
#ifdef EIP_3155
        delete evm_instances[index].tracer_ptr;
#endif
    }
    delete[] evm_instances[0].transaction_ptr;
    delete[] evm_instances;

         CUDA_CHECK(cudaFree(evm_instances[0].state_db_ptr));
         CUDA_CHECK(cudaFree(evm_instances[0].block_info_ptr));
         for (uint32_t index = 0; index < num_instances; index++) {
             // CUDA_CHECK(cudaFree(evm_instances[index].access_state_data_ptr));

             CUDA_CHECK(cudaFree(evm_instances[index].log_state_ptr));
             CUDA_CHECK(cudaFree(evm_instances[index].return_data_ptr));
 #ifdef EIP_3155
             CUDA_CHECK(cudaFree(evm_instances[index].tracer_ptr));
 #endif
         }
         CUDA_CHECK(cudaFree(evm_instances[0].transaction_ptr));
         CUDA_CHECK(cudaFree(evm_instances));
   */
}

}  // namespace CuEVM
// todo|: make a vector o functions global constants so you can call them
