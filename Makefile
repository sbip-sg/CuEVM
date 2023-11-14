NVCC = nvcc
NVCC_FLAGS = -I./CGBN/include -lstdc++ -lm -lgmp -lcjson -rdc=true --std c++20 -lcudadevrt -lineinfo
GCC = gcc
GCC_FLAGS = -lm -lgmp -lcjson
GPP = g++
GPP_FLAGS = -I./CGBN/include -lm -lgmp -lcjson 
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

interpreter: src/interpreter.cu
	$(NVCC) -D TRACER $(NVCC_FLAGS) -o $(OUT_DIRECTORY)/$@ $<

cpu_interpreter: src/interpreter.cu
	$(NVCC) -D TRACER -D ONLY_CPU $(NVCC_FLAGS) -o $(OUT_DIRECTORY)/$@ $<

% :: src/test/%.cu
	$(NVCC) $(NVCC_FLAGS) -o $(OUT_DIRECTORY)/$@ $<

clean:
	rm -f $(OUT_DIRECTORY)/*