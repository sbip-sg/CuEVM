# Object files
objects = cuevm.o cuevm_test.o stack.o uint256.o

# Compiler
NVCC = nvcc

# Compiler Flags
NVCC_FLAGS = -I./include -lstdc++ -dc

all: $(objects)
	$(NVCC) $(objects) -o cuEVM

%.o: src/%.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -f *.o cuEVM
