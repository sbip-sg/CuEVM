#ifndef OPCODE_H
#define OPCODE_H

#define ADD 0x01
#define MUL 0x02
#define SUB 0x03


#define POP 0x50
#define PUSH1 0x60
#define PUSH2 0x61

#define SWAP1 0x90


#define JUMP 0x56
#define JUMPI 0x57
#define JUMPDEST 0x5b


#define RETURN 0xf3
// Add other opcode definitions here as your VM expands

#endif // OPCODE_H
