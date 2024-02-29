# Compiler
NVCC = nvcc
NVCC_FLAGS = -I./CGBN/include -lstdc++ -lm -lgmp -lcjson -rdc=true --std c++20 -lcudadevrt -lineinfo
GCC = gcc
GCC_FLAGS = -lm -lgmp -lcjson
GPP = g++
GPP_FLAGS = -I./CGBN/include -lm -lgmp -lcjson
OUT_DIRECTORY = ./out

ENABLE_TRACING ?= 0
SM_ARCH ?= sm_89
ifeq ($(ENABLE_TRACING),1)
    TRACER_FLAG = -D TRACER
else
    TRACER_FLAG =
endif


test_gmp: src/test/test_gmp.c
	$(GCC) -o $(OUT_DIRECTORY)/test_gmp src/test/test_gmp.c $(GCC_FLAGS)

test_cjson: src/test/test_cjson.c
	$(GCC) -o $(OUT_DIRECTORY)/test_cjson src/test/test_cjson.c $(GCC_FLAGS)

test_cjson_evm: src/test/test_cjson_evm.c
	$(GCC) -o $(OUT_DIRECTORY)/test_cjson_evm src/test/test_cjson_evm.c $(GCC_FLAGS)

test_cgbn: src/test/test_cgbn.cu
	$(NVCC) $(NVCC_FLAGS) -o $(OUT_DIRECTORY)/test_cgbn src/test/test_cgbn.cu

interpreter:
	$(NVCC) $(TRACER_FLAG) $(NVCC_FLAGS) -arch=$(SM_ARCH) -o $(OUT_DIRECTORY)/$@ src/interpreter.cu

debug_interpreter:
	$(NVCC) -D TRACER -D COMPLEX_TRACER -D GAS $(NVCC_FLAGS) -arch=$(SM_ARCH) -g -lineinfo -o $(OUT_DIRECTORY)/$@ src/interpreter.cu

cpu_interpreter:
	$(NVCC) -D ONLY_CPU $(TRACER_FLAG) $(NVCC_FLAGS) -o $(OUT_DIRECTORY)/$@ src/interpreter.cu

cpu_debug_interpreter:
	$(NVCC) -D TRACER -D COMPLEX_TRACER -D ONLY_CPU -D GAS $(NVCC_FLAGS) -g -G -o $(OUT_DIRECTORY)/$@ src/interpreter.cu

% :: src/test/%.cu
	$(NVCC) $(NVCC_FLAGS) -g -G -o $(OUT_DIRECTORY)/$@ $<

clean:
	rm -f $(OUT_DIRECTORY)/*
