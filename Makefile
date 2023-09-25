NVCC = nvcc
NVCC_FLAGS = -I./include -lstdc++

all: cuEVM

cuEVM:
	$(NVCC) $(NVCC_FLAGS) -o cuEVM src/cu_evm.cu

clean:
	rm -f cuEVM