NVCC = nvcc
NVCC_FLAGS = -I./CGBN/include -lstdc++ -lm -lgmp -lcjson
GCC = gcc
GCC_FLAGS = -lgmp -lcjson
OUT_DIRECTORY = ./out

all: cuEVM

cuEVM:
	$(NVCC) $(NVCC_FLAGS) -o $(OUT_DIRECTORY)/cuEVM src/cu_evm.cu

test_gmp: src/test/test_gmp.c
	$(GCC) -o $(OUT_DIRECTORY)/test_gmp src/test/test_gmp.c $(GCC_FLAGS) 
	
test_cjson: src/test/test_cjson.c
	$(GCC) -o $(OUT_DIRECTORY)/test_cjson src/test/test_cjson.c $(GCC_FLAGS) 
	
test_cjson_evm: src/test/test_cjson_evm.c
	$(GCC) -o $(OUT_DIRECTORY)/test_cjson_evm src/test/test_cjson_evm.c $(GCC_FLAGS) 


test_cgbn: src/test/test_cgbn.cu
	$(NVCC) $(NVCC_FLAGS) -lineinfo -o $(OUT_DIRECTORY)/test_cgbn src/test/test_cgbn.cu

test_stack: src/test/test_stack.cu
	$(NVCC) $(NVCC_FLAGS) -rdc=true --std c++20 -lcudadevrt -lineinfo -o $(OUT_DIRECTORY)/test_stack src/test/test_stack.cu

test_memory: src/test/test_memory.cu
	$(NVCC) $(NVCC_FLAGS) -rdc=true --std c++20 -lcudadevrt -lineinfo -o $(OUT_DIRECTORY)/test_memory src/test/test_memory.cu

test_fixed_memory: src/test/test_fixed_memory.cu
	$(NVCC) $(NVCC_FLAGS) -rdc=true --std c++20 -lcudadevrt -lineinfo -o $(OUT_DIRECTORY)/test_fixed_memory src/test/

test_contract: src/test/test_contract.cu
	$(NVCC) $(NVCC_FLAGS) -rdc=true --std c++20 -lcudadevrt -lineinfo -o $(OUT_DIRECTORY)/test_contract src/test/test_contract.cu

test_block: src/test/test_block.cu
	$(NVCC) $(NVCC_FLAGS) -rdc=true --std c++20 -lcudadevrt -lineinfo -o $(OUT_DIRECTORY)/test_block src/test/test_block.cu

test_message: src/test/test_message.cu
	$(NVCC) $(NVCC_FLAGS) -rdc=true --std c++20 -lcudadevrt -lineinfo -o $(OUT_DIRECTORY)/test_message src/test/test_message.cu

test_storage: src/test/test_local_storage.cu
	$(NVCC) $(NVCC_FLAGS) -rdc=true --std c++20 -lcudadevrt -lineinfo -o $(OUT_DIRECTORY)/test_storage src/test/test_local_storage.cu


cu_evm_interpreter: src/cu_evm_interpreter.cu
	$(NVCC) $(NVCC_FLAGS) -rdc=true --std c++20 -lcudadevrt -lineinfo -o $(OUT_DIRECTORY)/cu_evm_interpreter src/cu_evm_interpreter.cu


clean:
	rm -f $(OUT_DIRECTORY)/cuEVM
	rm -f $(OUT_DIRECTORY)/test_gmp
	rm -f $(OUT_DIRECTORY)/test_cgbn
	rm -f $(OUT_DIRECTORY)/test_stack
	rm -f $(OUT_DIRECTORY)/test_memory
	rm -f $(OUT_DIRECTORY)/test_fixed_memory
	rm -f $(OUT_DIRECTORY)/test_contract
	rm -f $(OUT_DIRECTORY)/test_block
	rm -f $(OUT_DIRECTORY)/test_message
	rm -f $(OUT_DIRECTORY)/test_storage
	rm -f $(OUT_DIRECTORY)/cu_evm_interpreter