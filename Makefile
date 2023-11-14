NVCC = nvcc
NVCC_FLAGS = -I./CGBN/include -lstdc++ -lm -lgmp -lcjson -rdc=true --std c++20 -lcudadevrt -lineinfo
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

% :: src/test/%.cu
	$(NVCC) $(NVCC_FLAGS) -o $(OUT_DIRECTORY)/$@ $<

cu_evm_interpreter: src/cu_evm_interpreter.cu
	$(NVCC) $(NVCC_FLAGS) -o $(OUT_DIRECTORY)/cu_evm_interpreter src/cu_evm_interpreter.cu

interpreter: src/interpreter.cu
	$(NVCC) -D TRACER $(NVCC_FLAGS) -o $(OUT_DIRECTORY)/interpreter src/interpreter.cu

clean:
	rm -f $(OUT_DIRECTORY)/*