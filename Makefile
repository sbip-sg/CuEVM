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
	$(NVCC) $(NVCC_FLAGS) -o $(OUT_DIRECTORY)/test_cgbn src/test/test_cgbn.cu

clean:
	rm -f $(OUT_DIRECTORY)/cuEVM
	rm -f $(OUT_DIRECTORY)/test_gmp
	rm -f $(OUT_DIRECTORY)/test_cgbn