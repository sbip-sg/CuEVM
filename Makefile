NVCC = nvcc
NVCC_FLAGS = -I./CGBN/include -lstdc++ -lm -lgmp
GCC = gcc
GCC_FLAGS = -lgmp
OUT_DIRECTORY = ./out

all: cuEVM

cuEVM:
	$(NVCC) $(NVCC_FLAGS) -o $(OUT_DIRECTORY)/cuEVM src/cu_evm.cu

test_gmp: src/test/test_gmp.c
	$(GCC) -o $(OUT_DIRECTORY)/test_gmp src/test/test_gmp.c $(GCC_FLAGS) 

test_cgbn: src/test/test_cgbn.cu
	$(NVCC) $(NVCC_FLAGS) -lineinfo -o $(OUT_DIRECTORY)/test_cgbn src/test/test_cgbn.cu

test_stack: src/test/test_stack.cu
	$(NVCC) $(NVCC_FLAGS) -rdc=true --std c++20 -lcudadevrt -lineinfo -o $(OUT_DIRECTORY)/test_stack src/test/test_stack.cu

test_memory: src/test/test_memory.cu
	$(NVCC) $(NVCC_FLAGS) -rdc=true --std c++20 -lcudadevrt -lineinfo -o $(OUT_DIRECTORY)/test_memory src/test/test_memory.cu

test_fixed_memory: src/test/test_fixed_memory.cu
	$(NVCC) $(NVCC_FLAGS) -rdc=true --std c++20 -lcudadevrt -lineinfo -o $(OUT_DIRECTORY)/test_fixed_memory src/test/test_fixed_memory.cu
clean:
	rm -f $(OUT_DIRECTORY)/cuEVM
	rm -f $(OUT_DIRECTORY)/test_gmp
	rm -f $(OUT_DIRECTORY)/test_cgbn
	rm -f $(OUT_DIRECTORY)/test_stack
	rm -f $(OUT_DIRECTORY)/test_memory
	rm -f $(OUT_DIRECTORY)/test_fixed_memory