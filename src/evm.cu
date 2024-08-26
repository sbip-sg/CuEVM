#include "include/evm.cuh"

#include "include/operations/stack.cuh"
#include "include/operations/arithmetic.cuh"
#include "include/operations/bitwise.cuh"
#include "include/operations/block.cuh"
#include "include/operations/environmental.cuh"
#include "include/operations/log.cuh"
#include "include/operations/memory.cuh"
#include "include/operations/storage.cuh"
#include "include/operations/system.cuh"
#include "include/operations/compare.cuh"
#include "include/operations/flow.cuh"
#include "include/gas_cost.cuh"
#include "include/utils/arith.cuh"
#include "include/utils/error_codes.cuh"

#include "include/utils/opcodes.cuh"

namespace cuEVM {
    __host__ __device__  evm_t::evm_t(
        ArithEnv &arith,
        cuEVM::state::state_t *world_state_data_ptr,
        cuEVM::block_info_t* block_info_ptr,
        cuEVM::evm_transaction_t* transaction_ptr,
        cuEVM::state::state_access_t *access_state_data_ptr,
        cuEVM::state::state_access_t *touch_state_data_ptr,
        cuEVM::state::log_state_data_t* log_state_ptr,
        cuEVM::evm_return_data_t* return_data_ptr
        #ifdef EIP_3155
        , cuEVM::utils::tracer_t* tracer_ptr
        #endif
    ) : world_state(world_state_data_ptr), block_info_ptr(block_info_ptr), transaction_ptr(transaction_ptr), access_state(access_state_data_ptr, &world_state) {
        call_state_ptr = new cuEVM::evm_call_state_t(
            arith,
            &access_state,
            nullptr,
            nullptr,
            log_state_ptr,
            touch_state_data_ptr,
            return_data_ptr
        );
        int32_t error_code = transaction_ptr->validate(
            arith,
            access_state,
            call_state_ptr->touch_state,
            *block_info_ptr,
            call_state_ptr->gas_used,
            gas_price,
            gas_priority_fee
        );
        if (error_code == ERROR_SUCCESS) {
            cuEVM::evm_message_call_t *transaction_call_message_ptr = nullptr;
            error_code = transaction_ptr->get_message_call(
                arith,
                access_state,
                transaction_call_message_ptr
            );
            cuEVM::evm_call_state_t* child_call_state_ptr = new cuEVM::evm_call_state_t(
                arith,
                call_state_ptr,
                transaction_call_message_ptr
            );
            call_state_ptr = child_call_state_ptr;
        }
        #ifdef EIP_3155
        this->tracer_ptr = tracer_ptr;
        #endif
        status = error_code;
    }

    __host__ __device__  evm_t::evm_t(
        ArithEnv &arith,
        cuEVM::evm_instance_t &evm_instance
    ) {
        evm_t(
            arith,
            evm_instance.world_state_data_ptr,
            evm_instance.block_info_ptr,
            evm_instance.transaction_ptr,
            evm_instance.access_state_data_ptr,
            evm_instance.touch_state_data_ptr,
            evm_instance.log_state_ptr,
            evm_instance.return_data_ptr
            #ifdef EIP_3155
            , evm_instance.tracer_ptr
            #endif
        );
    }

    __host__ __device__ evm_t::~evm_t() {
        if (call_state_ptr != nullptr) {
            delete call_state_ptr;
        }
        call_state_ptr = nullptr;
        block_info_ptr = nullptr;
        transaction_ptr = nullptr;
        #ifdef EIP_3155
        tracer_ptr = nullptr;
        #endif
    }


    __host__ __device__ int32_t evm_t::start_CALL(ArithEnv &arith) {
        bn_t sender, recipient, value, sender_balance, recipient_balance;
        call_state_ptr->message_ptr->get_sender(arith, sender);
        call_state_ptr->message_ptr->get_recipient(arith, recipient);
        call_state_ptr->message_ptr->get_value(arith, value);
        int32_t error_code = (
            (
                (cgbn_compare_ui32(arith.env, value, 0) > 0) &&
                (cgbn_compare(arith.env, sender, recipient) != 0) &&
                (call_state_ptr->message_ptr->call_type != OP_DELEGATECALL)
            ) ?
            call_state_ptr->touch_state.transfer(
                arith,
                sender,
                recipient,
                value
            ) :
            ERROR_SUCCESS
        );
        // warmup the accounts
        cuEVM::account::account_t* account_ptr;
        error_code |= call_state_ptr->touch_state.get_account(arith, sender, account_ptr, ACCOUNT_NONE_FLAG);
        error_code |= call_state_ptr->touch_state.get_account(arith, recipient, account_ptr, ACCOUNT_BYTE_CODE_FLAG);

        if (
            (call_state_ptr->message_ptr->call_type == OP_CREATE) ||
            (call_state_ptr->message_ptr->call_type == OP_CREATE2)
        ) {
            error_code |= account_ptr->is_contract() ? ERROR_MESSAGE_CALL_CREATE_CONTRACT_EXISTS : ERROR_SUCCESS;
            bn_t contract_nonce;
            cgbn_set_ui32(arith.env, contract_nonce, 1);
            error_code |= call_state_ptr->touch_state.set_nonce(arith, recipient, contract_nonce);
            bn_t sender_nonce;
            error_code |= call_state_ptr->touch_state.get_nonce(arith, sender, sender_nonce);
            cgbn_add_ui32(arith.env, sender_nonce, sender_nonce, 1);
            uint64_t nonce;
            error_code |= arith.uint64_t_from_cgbn(nonce, sender_nonce) == 1 ?  ERROR_MESSAGE_CALL_CREATE_NONCE_EXCEEDED : ERROR_SUCCESS;
        } else {
            error_code |= call_state_ptr->depth >= cuEVM::max_depth ? ERROR_MESSAGE_CALL_DEPTH_EXCEEDED : ERROR_SUCCESS;
            if (account_ptr->byte_code.size == 0) {
                bn_t contract_address;
                call_state_ptr->message_ptr->get_contract_address(arith, contract_address);
                if (cgbn_compare_ui32(arith.env, contract_address, cuEVM::no_precompile_contracts) == -1) {

                    switch (cgbn_get_ui32(arith.env, contract_address))
                    {
                        case 1:
                            /* code */
                            return 1;
                            break;
                        
                        default:
                            break;
                    }
                } else {
                    // operation stop
                    return 1;
                }
            }
        }
        return error_code;
    }

    __host__ __device__ void evm_t::run(ArithEnv &arith) {
        if (status != ERROR_SUCCESS) {
            return; //finish transaction
        }
        int32_t error_code = start_CALL(arith);
        if (error_code != ERROR_SUCCESS) {
            return; //finish call
        }
        uint8_t opcode;
        bn_t gas_limit;
        bn_t gas_left;
        call_state_ptr->message_ptr->get_gas_limit(arith, gas_limit);
        cuEVM::evm_call_state_t* child_call_state_ptr = nullptr;
        while (
            status == ERROR_SUCCESS 
        ) {
            opcode = (
                (call_state_ptr->pc <
                ((call_state_ptr->message_ptr)->byte_code).size) ? 
                (call_state_ptr->message_ptr)->byte_code.data[call_state_ptr->pc] :
                OP_STOP
            );
            #ifdef EIP_3155
            cgbn_sub(arith.env, gas_left, call_state_ptr->gas_limit, call_state_ptr->gas_used);
            call_state_ptr->trace_idx = tracer_ptr->push_init(
                arith,
                call_state_ptr->pc,
                opcode,
                *call_state_ptr->memory_ptr,
                *call_state_ptr->stack_ptr,
                call_state_ptr->depth,
                *call_state_ptr->last_return_data_ptr,
                gas_left
            );
            #endif
            if (((opcode & 0xF0) == 0x60) || ((opcode & 0xF0) == 0x70))
            {
                error_code = cuEVM::operations::PUSHX(
                    arith,
                    gas_limit,
                    call_state_ptr->gas_used,
                    call_state_ptr->pc,
                    *call_state_ptr->stack_ptr,
                    ((call_state_ptr->message_ptr)->byte_code),
                    opcode
                );
            }
            else if ((opcode & 0xF0) == 0x80) // DUPX
            {
                error_code = cuEVM::operations::DUPX(
                    arith,
                    gas_limit,
                    call_state_ptr->gas_used,
                    *call_state_ptr->stack_ptr,
                    opcode
                );
            }
            else if ((opcode & 0xF0) == 0x90) // SWAPX
            {
                error_code = cuEVM::operations::SWAPX(
                    arith,
                    gas_limit,
                    call_state_ptr->gas_used,
                    *call_state_ptr->stack_ptr,
                    opcode
                );
            }
            else if ((opcode >= 0xA0) && (opcode <= 0xA4)) // LOGX
            {
                error_code = cuEVM::operations::LOGX(
                    arith,
                    gas_limit,
                    call_state_ptr->gas_used,
                    *call_state_ptr->stack_ptr,
                    *call_state_ptr->memory_ptr,
                    *call_state_ptr->message_ptr,
                    *call_state_ptr->log_state_ptr,
                    opcode
                );
            }
            else
            {

                switch (opcode)
                {
                case OP_STOP:
                    error_code = cuEVM::operations::STOP(
                        *call_state_ptr->parent->last_return_data_ptr
                    );
                    break;
                case OP_ADD:
                    error_code = cuEVM::operations::ADD(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr
                    );
                    break;
                case OP_MUL:
                    error_code = cuEVM::operations::MUL(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr
                    );
                    break;
                case OP_SUB:
                    error_code = cuEVM::operations::SUB(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr
                    );
                    break;
                case OP_DIV:
                    error_code = cuEVM::operations::DIV(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr
                    );
                    break;
                case OP_SDIV:
                    error_code = cuEVM::operations::SDIV(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr
                    );
                    break;
                case OP_MOD:
                    error_code = cuEVM::operations::MOD(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr
                    );
                    break;
                case OP_SMOD:
                    error_code = cuEVM::operations::SMOD(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr
                    );
                    break;
                case OP_ADDMOD:
                    error_code = cuEVM::operations::ADDMOD(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr
                    );
                    break;
                case OP_MULMOD:
                    error_code = cuEVM::operations::MULMOD(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr
                    );
                    break;
                case OP_EXP:
                    error_code = cuEVM::operations::EXP(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr
                    );
                    break;
                case OP_SIGNEXTEND:
                    error_code = cuEVM::operations::SIGNEXTEND(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr
                    );
                    break;
                case OP_LT:
                    error_code = cuEVM::operations::LT(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr
                    );
                    break;
                case OP_GT:
                    error_code = cuEVM::operations::GT(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr
                    );
                    break;
                case OP_SLT:
                    error_code = cuEVM::operations::SLT(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr
                    );
                    break;
                case OP_SGT:
                    error_code = cuEVM::operations::SGT(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr
                    );
                    break;
                case OP_EQ:
                    error_code = cuEVM::operations::EQ(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr
                    );
                    break;
                case OP_ISZERO:
                    error_code = cuEVM::operations::ISZERO(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr
                    );
                    break;
                case OP_AND:
                    error_code = cuEVM::operations::AND(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr
                    );
                    break;
                case OP_OR:
                    error_code = cuEVM::operations::OR(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr
                    );
                    break;
                case OP_XOR:
                    error_code = cuEVM::operations::XOR(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr
                    );
                    break;
                case OP_NOT:
                    error_code = cuEVM::operations::NOT(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr
                    );
                    break;
                case OP_BYTE:
                    error_code = cuEVM::operations::BYTE(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr
                    );
                    break;
                case OP_SHL:
                    error_code = cuEVM::operations::SHL(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr
                    );
                    break;
                case OP_SHR:
                    error_code = cuEVM::operations::SHR(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr
                    );
                    break;
                case OP_SAR:
                    error_code = cuEVM::operations::SAR(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr
                    );
                    break;
                case OP_SHA3:
                    error_code = cuEVM::operations::SHA3(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr,
                        *call_state_ptr->memory_ptr
                    );
                    break;
                case OP_ADDRESS:
                    error_code = cuEVM::operations::ADDRESS(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr,
                        *call_state_ptr->message_ptr
                    );
                    break;
                case OP_BALANCE:
                    error_code = cuEVM::operations::BALANCE(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr,
                        access_state,
                        call_state_ptr->touch_state
                    );
                    break;
                case OP_ORIGIN:
                    error_code = cuEVM::operations::ORIGIN(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr,
                        *transaction_ptr
                    );
                    break;
                case OP_CALLER:
                    error_code = cuEVM::operations::CALLER(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr,
                        *call_state_ptr->message_ptr
                    );
                    break;
                case OP_CALLVALUE:
                    error_code = cuEVM::operations::CALLVALUE(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr,
                        *call_state_ptr->message_ptr
                    );
                    break;
                case OP_CALLDATALOAD:
                    error_code = cuEVM::operations::CALLDATALOAD(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr,
                        *call_state_ptr->message_ptr
                    );
                    break;
                case OP_CALLDATASIZE:
                    error_code = cuEVM::operations::CALLDATASIZE(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr,
                        *call_state_ptr->message_ptr
                    );
                    break;
                case OP_CALLDATACOPY:
                    error_code = cuEVM::operations::CALLDATACOPY(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr,
                        *call_state_ptr->message_ptr,
                        *call_state_ptr->memory_ptr
                    );
                    break;
                case OP_CODESIZE:
                    error_code = cuEVM::operations::CODESIZE(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr,
                        *call_state_ptr->message_ptr
                    );
                    break;
                case OP_CODECOPY:
                    error_code = cuEVM::operations::CODECOPY(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr,
                        *call_state_ptr->message_ptr,
                        *call_state_ptr->memory_ptr
                    );
                    break;
                case OP_GASPRICE:
                    error_code = cuEVM::operations::GASPRICE(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr,
                        *block_info_ptr,
                        *transaction_ptr
                    );
                    break;
                case OP_EXTCODESIZE:
                    error_code = cuEVM::operations::EXTCODESIZE(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr,
                        access_state,
                        call_state_ptr->touch_state
                    );
                    break;
                case OP_EXTCODECOPY:
                    error_code = cuEVM::operations::EXTCODECOPY(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr,
                        access_state,
                        call_state_ptr->touch_state,
                        *call_state_ptr->memory_ptr
                    );
                    break;
                case OP_RETURNDATASIZE:
                    error_code = cuEVM::operations::RETURNDATASIZE(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr,
                        *call_state_ptr->last_return_data_ptr
                    );
                    break;
                case OP_RETURNDATACOPY:
                    error_code = cuEVM::operations::RETURNDATACOPY(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr,
                        *call_state_ptr->memory_ptr,
                        *call_state_ptr->last_return_data_ptr
                    );
                    break;
                case OP_EXTCODEHASH:
                    error_code = cuEVM::operations::EXTCODEHASH(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr,
                        access_state,
                        call_state_ptr->touch_state
                    );
                    break;
                case OP_BLOCKHASH:
                    error_code = cuEVM::operations::BLOCKHASH(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr,
                        *block_info_ptr
                    );
                    break;
                case OP_COINBASE:
                    error_code = cuEVM::operations::COINBASE(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr,
                        *block_info_ptr
                    );
                    break;
                case OP_TIMESTAMP:
                    error_code = cuEVM::operations::TIMESTAMP(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr,
                        *block_info_ptr
                    );
                    break;
                case OP_NUMBER:
                    error_code = cuEVM::operations::NUMBER(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr,
                        *block_info_ptr
                    );
                    break;
                case OP_DIFFICULTY:
                    error_code = cuEVM::operations::PREVRANDAO(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr,
                        *block_info_ptr
                    );
                    break;
                case OP_GASLIMIT:
                    error_code = cuEVM::operations::GASLIMIT(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr,
                        *block_info_ptr
                    );
                    break;
                case OP_CHAINID:
                    error_code = cuEVM::operations::CHAINID(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr,
                        *block_info_ptr
                    );
                    break;
                case OP_SELFBALANCE:
                    error_code = cuEVM::operations::SELFBALANCE(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr,
                        call_state_ptr->touch_state,
                        *call_state_ptr->message_ptr
                    );
                    break;
                case OP_BASEFEE:
                    error_code = cuEVM::operations::BASEFEE(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr,
                        *block_info_ptr
                    );
                    break;
                case OP_POP:
                    error_code = cuEVM::operations::POP(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr
                    );
                    break;
                case OP_MLOAD:
                    error_code = cuEVM::operations::MLOAD(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr,
                        *call_state_ptr->memory_ptr
                    );
                    break;
                case OP_MSTORE:
                    error_code = cuEVM::operations::MSTORE(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr,
                        *call_state_ptr->memory_ptr
                    );
                    break;
                case OP_MSTORE8:
                    error_code = cuEVM::operations::MSTORE8(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr,
                        *call_state_ptr->memory_ptr
                    );
                    break;
                case OP_SLOAD:
                    error_code = cuEVM::operations::SLOAD(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr,
                        access_state,
                        call_state_ptr->touch_state,
                        *call_state_ptr->message_ptr
                    );
                    break;
                case OP_SSTORE:
                    error_code = cuEVM::operations::SSTORE(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        call_state_ptr->gas_refund,
                        *call_state_ptr->stack_ptr,
                        access_state,
                        call_state_ptr->touch_state,
                        *call_state_ptr->message_ptr
                    );
                    break;
                case OP_JUMP:
                    error_code = cuEVM::operations::JUMP(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        call_state_ptr->pc,
                        *call_state_ptr->stack_ptr,
                        *call_state_ptr->message_ptr
                    );
                    break;
                case OP_JUMPI:
                    error_code = cuEVM::operations::JUMPI(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        call_state_ptr->pc,
                        *call_state_ptr->stack_ptr,
                        *call_state_ptr->message_ptr
                    );
                    break;
                
                case OP_PC:
                    error_code = cuEVM::operations::PC(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        call_state_ptr->pc,
                        *call_state_ptr->stack_ptr
                    );
                    break;
                
                case OP_MSIZE:
                    error_code = cuEVM::operations::MSIZE(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr,
                        *call_state_ptr->memory_ptr
                    );
                    break;
                
                case OP_GAS:
                    error_code = cuEVM::operations::GAS(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr
                    );
                    break;

                case OP_JUMPDEST:
                    error_code = cuEVM::operations::JUMPDEST(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used
                    );
                    break;
                
                case OP_PUSH0:
                    error_code = cuEVM::operations::PUSH0(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr
                    );
                    break;
                
                case OP_CREATE:
                    child_call_state_ptr = nullptr;
                    error_code = cuEVM::operations::CREATE(
                        arith,
                        access_state,
                        *call_state_ptr,
                        child_call_state_ptr
                    );

                    if (error_code == ERROR_SUCCESS) {
                        call_state_ptr = child_call_state_ptr;
                        start_CALL(arith);
                    }
                    break;
                
                case OP_CALL:
                    child_call_state_ptr = nullptr;
                    error_code = cuEVM::operations::CALL(
                        arith,
                        access_state,
                        *call_state_ptr,
                        child_call_state_ptr
                    );

                    if (error_code == ERROR_SUCCESS) {
                        call_state_ptr = child_call_state_ptr;
                        start_CALL(arith);
                    }
                    break;
                
                case OP_CALLCODE:
                    child_call_state_ptr = nullptr;
                    error_code = cuEVM::operations::CALLCODE(
                        arith,
                        access_state,
                        *call_state_ptr,
                        child_call_state_ptr
                    );

                    if (error_code == ERROR_SUCCESS) {
                        call_state_ptr = child_call_state_ptr;
                        start_CALL(arith);
                    }
                    break;
                
                case OP_RETURN:
                    error_code = cuEVM::operations::RETURN(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr,
                        *call_state_ptr->memory_ptr,
                        *call_state_ptr->parent->last_return_data_ptr
                    );
                    break;
                
                case OP_DELEGATECALL:
                    child_call_state_ptr = nullptr;
                    error_code = cuEVM::operations::DELEGATECALL(
                        arith,
                        access_state,
                        *call_state_ptr,
                        child_call_state_ptr
                    );

                    if (error_code == ERROR_SUCCESS) {
                        call_state_ptr = child_call_state_ptr;
                        start_CALL(arith);
                    }
                    break;
                
                case OP_CREATE2:
                    child_call_state_ptr = nullptr;
                    error_code = cuEVM::operations::CREATE2(
                        arith,
                        access_state,
                        *call_state_ptr,
                        child_call_state_ptr
                    );

                    if (error_code == ERROR_SUCCESS) {
                        call_state_ptr = child_call_state_ptr;
                        start_CALL(arith);
                    }
                    break;
                
                case OP_STATICCALL:
                    child_call_state_ptr = nullptr;
                    error_code = cuEVM::operations::STATICCALL(
                        arith,
                        access_state,
                        *call_state_ptr,
                        child_call_state_ptr
                    );

                    if (error_code == ERROR_SUCCESS) {
                        call_state_ptr = child_call_state_ptr;
                        start_CALL(arith);
                    }
                    break;

                case OP_REVERT:
                    error_code = cuEVM::operations::REVERT(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr,
                        *call_state_ptr->memory_ptr,
                        *call_state_ptr->parent->last_return_data_ptr
                    );
                    break;
                
                case OP_SELFDESTRUCT:
                    error_code = cuEVM::operations::SELFDESTRUCT(
                        arith,
                        gas_limit,
                        call_state_ptr->gas_used,
                        *call_state_ptr->stack_ptr,
                        *call_state_ptr->message_ptr,
                        call_state_ptr->touch_state,
                        *call_state_ptr->parent->last_return_data_ptr
                    );
                    break;
                
                
                default:
                    error_code = cuEVM::operations::INVALID();
                    break;
                }
            }

            #ifdef EIP_3155
            if (call_state_ptr->trace_idx > 0 ||
                (call_state_ptr->trace_idx == 0 && call_state_ptr->depth == 0) ) {
                bn_t gas_left_2;
                cgbn_sub(arith.env, gas_left_2, call_state_ptr->gas_limit, call_state_ptr->gas_used);
                cgbn_sub(arith.env, gas_left, gas_left, gas_left_2);
                tracer_ptr->push_final(
                    arith,
                    call_state_ptr->trace_idx,
                    gas_left,
                    call_state_ptr->gas_refund
                    #ifdef EIP_3155_OPTIONAL
                    , error_code,
                    call_state_ptr->touch_state
                    #endif
                );
            }
            #endif
            // TODO: to see after calls

            if (error_code != ERROR_SUCCESS) {
                if (
                    (error_code == ERROR_RETURN) && 
                    (call_state_ptr->message_ptr->call_type == OP_CREATE ||
                    call_state_ptr->message_ptr->call_type == OP_CREATE2)
                ) {
                    // TODO: finish create call add the contract to the state
                    printf("Create call\n");
                    error_code |= finish_CREATE(arith);
                }

                if (call_state_ptr->depth == 0) {
                    // TODO: finish transaction
                    printf("Finish transaction\n");
                    error_code |= finish_CALL(arith, error_code);
                    error_code |= finish_TRANSACTION(arith, error_code);
                } else {
                    // TODO: finish call
                    printf("Finish call\n");
                    error_code |= finish_CALL(arith, error_code);
                }
            }
        }
    }

    __host__ __device__ int32_t evm_t::finish_TRANSACTION(
        ArithEnv &arith,
        int32_t error_code) {
        // sent the gas value to the block beneficiary
        bn_t gas_value;
        bn_t beneficiary;
        block_info_ptr->get_coin_base(arith, beneficiary);

        if ( (error_code == ERROR_RETURN) || (error_code == ERROR_REVERT) )
        {
            bn_t gas_left;
            // \f$T_{g} - g\f$
            cgbn_sub(arith.env, gas_left, call_state_ptr->gas_limit, call_state_ptr->gas_used);

            // if return add the refund gas
            if (error_code == ERROR_RETURN) {
                bn_t capped_refund_gas;
                // \f$g/5\f$
                cgbn_div_ui32(arith.env, capped_refund_gas, call_state_ptr->gas_used, 5);
                // min ( \f$g/5\f$, \f$R_{g}\f$)

                if (cgbn_compare(arith.env, capped_refund_gas, call_state_ptr->gas_refund) > 0)
                {
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
            bn_t sender_address;
            // send back the gas left and gas refund to the sender
            transaction_ptr->get_sender(arith, sender_address);
            // deduct transaction value; TODO this probably should be done at some other place
            // _transaction->get_value(tx_value);
            // cgbn_sub(arith.env, sender_balance, sender_balance, tx_value);
            // the gas value for the beneficiary is \f$T_{g} - g^{*}\f$
            cgbn_sub(arith.env, gas_value, call_state_ptr->gas_limit, gas_value);
            cgbn_mul(arith.env, gas_value, gas_value, gas_priority_fee);


            // update the transaction state
            if (error_code == ERROR_RETURN)
            {
                call_state_ptr->update(arith, *call_state_ptr->parent);
            }
            // sent the value of unused gas to the sender
            call_state_ptr->parent->touch_state.get_balance(arith, sender_address, sender_balance);
            cgbn_add(arith.env, sender_balance, sender_balance, send_back_gas);
            call_state_ptr->parent->touch_state.set_balance(arith, sender_address, sender_balance);

            // set the eror code for a succesfull transaction
            status = error_code;
        }
        else
        {
            cgbn_mul(arith.env, gas_value, call_state_ptr->gas_limit, gas_priority_fee);
            // set z to the given error or 1 TODO: 1 in YP
            status = error_code;
        }
        // send the gas value to the beneficiary
        if (cgbn_compare_ui32(arith.env, gas_value, 0) > 0 ) {
            bn_t beneficiary_balance;
            call_state_ptr->parent->touch_state.get_balance(arith, beneficiary, beneficiary_balance);
            cgbn_add(arith.env, beneficiary_balance, beneficiary_balance, gas_value);
            call_state_ptr->parent->touch_state.set_balance(arith, beneficiary, beneficiary_balance);
        }

        call_state_ptr = call_state_ptr->parent;
        return status;

    }

    
    __host__ __device__ int32_t evm_t::finish_CALL(ArithEnv &arith, int32_t error_code) {
        
        bn_t child_success;
        // set the child call to failure
        cgbn_set_ui32(arith.env, child_success, 0);
        // if the child call return from normal halting
        // no errors
        if ( (error_code == ERROR_RETURN) || (error_code == ERROR_REVERT) )
        {
            // give back the gas left from the child computation
            bn_t gas_left;
            cgbn_sub(arith.env, gas_left, call_state_ptr->gas_limit, call_state_ptr->gas_used);
            cgbn_sub(arith.env, call_state_ptr->parent->gas_used, call_state_ptr->parent->gas_used, gas_left);

            // if is a succesfull call
            if (error_code == ERROR_RETURN)
            {
                // update the parent state with the states of the child
                call_state_ptr->parent->update(arith, *call_state_ptr);
                // sum the refund gas
                cgbn_add(
                    arith.env,
                    call_state_ptr->parent->gas_refund,
                    call_state_ptr->parent->gas_refund,
                    call_state_ptr->gas_refund);
                // for CALL operations set the child success to 1
                cgbn_set_ui32(arith.env, child_success, 1);
                // if CREATEX operation, set the address of the contract
                if (
                    (call_state_ptr->message_ptr->get_call_type() == OP_CREATE) ||
                    (call_state_ptr->message_ptr->get_call_type() == OP_CREATE2))
                {
                    call_state_ptr->message_ptr->get_recipient(arith, child_success);
                }
            }
        }
        // get the memory offset and size of the return data
        // in the parent memory
        bn_t ret_offset, ret_size;
        call_state_ptr->message_ptr->get_return_data_offset(arith, ret_offset);
        call_state_ptr->message_ptr->get_return_data_size(arith, ret_size);
        // reset the error code for the parent
        error_code = ERROR_SUCCESS;

        // push the result in the parent stack
        error_code |= call_state_ptr->parent->stack_ptr->push(arith, child_success);
        // set the parent memory with the return data
        
        // write the return data in the memory
        error_code |= call_state_ptr->parent->memory_ptr->set(
            arith,
            *call_state_ptr->parent->last_return_data_ptr,
            ret_offset,
            ret_size);
        
        // trace the call
        #ifdef EIP_3155
        if (call_state_ptr->depth > 0) {
            bn_t gas_left;
            cgbn_sub(arith.env, gas_left, call_state_ptr->gas_limit, call_state_ptr->gas_used);
            tracer_ptr->push_final(
                arith,
                call_state_ptr->trace_idx,
                gas_left,
                call_state_ptr->gas_refund
                #ifdef EIP_3155_OPTIONAL
                , error_code,
                call_state_ptr->touch_state
                #endif
            );
        }
        #endif
        
        return error_code;
    }

    
    __host__ __device__ int32_t evm_t::finish_CREATE(ArithEnv &arith) {
        // compute the gas to deposit the contract
        bn_t code_size;
        cgbn_set_ui32(arith.env, code_size, call_state_ptr->last_return_data_ptr->size);
        cuEVM::gas_cost::code_cost(arith, call_state_ptr->gas_used, code_size);
        int32_t error_code = ERROR_SUCCESS;
        error_code |= cuEVM::gas_cost::has_gas(
            arith,
            call_state_ptr->gas_limit,
            call_state_ptr->gas_used
        );
        if (error_code == ERROR_SUCCESS) {
            // compute the address of the contract
            bn_t contract_address;
            call_state_ptr->message_ptr->get_recipient(arith, contract_address);
            uint8_t *code = call_state_ptr->last_return_data_ptr->data;
            uint32_t code_size = call_state_ptr->last_return_data_ptr->size;
            if (code_size <= cuEVM::max_code_size) {
                #ifdef EIP_3541
                if ((code_size > 0) && (code[0] == 0xef)) {
                    error_code = ERROR_CREATE_CODE_FIRST_BYTE_INVALID;
                }
                #endif
                call_state_ptr->touch_state.set_code(
                    arith,
                    contract_address,
                    *call_state_ptr->last_return_data_ptr
                );
            } else {
                error_code = ERROR_CREATE_CODE_SIZE_EXCEEDED;
            }
        }
        return error_code;
    }

    __host__ int32_t get_cpu_evm_instances(
        ArithEnv &arith,
        evm_instance_t* &evm_instances,
        const cJSON* test_json,
        uint32_t &num_instances
    ) {
        // get the world state
        cuEVM::state::state_t *world_state_data_ptr = nullptr;
        const cJSON *world_state_json = NULL; // the json for the world state
        // get the world state json
        if (cJSON_IsObject(test_json))
            world_state_json = cJSON_GetObjectItemCaseSensitive(test_json, "pre");
        else if (cJSON_IsArray(test_json))
            world_state_json = test_json;
        else
            return 1;
        
        world_state_data_ptr = new cuEVM::state::state_t(world_state_json);

        // get the block info
        cuEVM::block_info_t* block_info_ptr = new cuEVM::block_info_t(test_json);

        // get the transaction
        cuEVM::evm_transaction_t* transactions_ptr = nullptr;
        uint32_t num_transactions = 0;
        cuEVM::transaction::get_transactios(
            arith,
            transactions_ptr,
            test_json,
            num_transactions
        );

        // generate the evm instances
        evm_instances = new evm_instance_t[num_transactions];
        uint32_t index = 0;
        for (uint32_t index = 0; index < num_transactions; index++) {
            evm_instances[index].world_state_data_ptr = world_state_data_ptr;
            evm_instances[index].block_info_ptr = block_info_ptr;
            evm_instances[index].transaction_ptr = &transactions_ptr[index];
            evm_instances[index].access_state_data_ptr = new cuEVM::state::state_access_t();
            evm_instances[index].touch_state_data_ptr = new cuEVM::state::state_access_t();
            evm_instances[index].log_state_ptr = new cuEVM::state::log_state_data_t();
            evm_instances[index].return_data_ptr = new cuEVM::evm_return_data_t();
            #ifdef EIP_3155
            evm_instances[index].tracer_ptr = new cuEVM::utils::tracer_t();
            #endif
        }
        num_instances = num_transactions;
    }

    __host__ void free_cpu_evm_instances(
        evm_instance_t* &evm_instances,
        uint32_t num_instances
    ) {
        delete evm_instances[0].world_state_data_ptr;
        delete evm_instances[0].block_info_ptr;
        for (uint32_t index = 0; index < num_instances; index++) {
            delete evm_instances[index].access_state_data_ptr;
            delete evm_instances[index].touch_state_data_ptr;
            delete evm_instances[index].log_state_ptr;
            delete evm_instances[index].return_data_ptr;
            #ifdef EIP_3155
            delete evm_instances[index].tracer_ptr;
            #endif
        }
        delete[] evm_instances[0].transaction_ptr;
        delete[] evm_instances;
    }
}
// todo|: make a vector o functions global constants so you can call them