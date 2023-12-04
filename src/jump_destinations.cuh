#ifndef _JUMP_DESTINATION_H_
#define _JUMP_DESTINATION_H_

#include "utils.h"

class jump_destinations_t {
    private:
    size_t _size;
    size_t *_destinations;

    public:
    __host__ __device__ jump_destinations_t(uint8_t *byte_code, size_t code_size) {
        _size = 0;
        _destinations = NULL;
        uint8_t opcode;
        uint8_t push_size;
        size_t pc;
        for (pc=0; pc < code_size; pc++) {
            opcode=byte_code[pc];
            // if a push x 
            if ( ((opcode&0xF0)==0x60) || ((opcode&0xF0)==0x70) ) {
                push_size=(opcode&0x1F)+1;
                pc=pc+push_size;
            }
            if (opcode==OP_JUMPDEST) {
                _size = _size + 1;
            }
        }
        if (_size > 0) {
            _destinations = new size_t[_size];
            size_t index = 0;
            for (pc=0; pc < code_size; pc++) {
                opcode=byte_code[pc];
                // if a push x 
                if ( ((opcode&0xF0)==0x60) || ((opcode&0xF0)==0x70) ) {
                    push_size=(opcode&0x1F)+1;
                    pc=pc+push_size;
                }
                if (opcode==OP_JUMPDEST) {
                    _destinations[index] = pc;
                    index = index + 1;
                }
            }
        }
    }

    __host__ __device__ ~jump_destinations_t() {
        if ( (_destinations != NULL) && (_size > 0)) {
            delete[] _destinations;
            _destinations = NULL;
            _size = 0;
        }
    }

    __host__ __device__ uint32_t has(size_t pc) {
        size_t index;
        for (index=0; index < _size; index++) {
            if (_destinations[index] == pc) {
                return 1;
            }
        }
        return 0;
    }

};

#endif