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
        status = error_code;
    }

    __host__ __device__ evm_t::~evm_t() {
        if (call_state_ptr != nullptr) {
            delete call_state_ptr;
        }
        call_state_ptr = nullptr;
        block_info_ptr = nullptr;
        transaction_ptr = nullptr;
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
        call_state_ptr->message_ptr->get_gas_limit(arith, gas_limit);
        while (
            status == ERROR_SUCCESS 
        ) {
            opcode = (
                (call_state_ptr->pc <
                ((call_state_ptr->message_ptr)->byte_code).size) ? 
                (call_state_ptr->message_ptr)->byte_code.data[call_state_ptr->pc] :
                OP_STOP
            );
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
                        call_state_ptr->gas_used,
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
                    cuEVM::evm_call_state_t* child_call_state_ptr = nullptr;
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
                    cuEVM::evm_call_state_t* child_call_state_ptr = nullptr;
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
                    cuEVM::evm_call_state_t* child_call_state_ptr = nullptr;
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
        }
    }
}

// todo|: make a vector o functions global constants so you can call them