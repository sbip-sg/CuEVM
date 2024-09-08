#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <CGBN/cgbn.h>


// Define the CGBN context and environment
const uint32_t INSTANCES = 100000; // Number of instances

typedef cgbn_context_t<cgbn_default_tpi> context_t;
typedef cgbn_env_t<context_t, cgbn_default_bits> env_t;
typedef typename env_t::cgbn_t bn_t;
typedef typename env_t::cgbn_wide_t bn_wide_t;
typedef cgbn_mem_t<cgbn_default_bits> bn_mem_t;

// Declare the instance type
typedef struct {
  bn_mem_t a;
  bn_mem_t b;
  bn_mem_t sum;
} instance_t;


uint32_t random_word() {
  uint32_t random;

  random=rand() & 0xFFFF;
  random=(random<<16) + (rand() & 0xFFFF);
  return random;
}

void random_words(uint32_t *x, uint32_t count) {
  int index;

  for(index=0;index<count;index++)
    x[index]=random_word();
}


// support routine to generate random instances
instance_t *generate_instances(uint32_t count) {
  instance_t *instances=(instance_t *)malloc(sizeof(instance_t)*count);

  for(int index=0;index<count;index++) {
    random_words(instances[index].a._limbs, cgbn_default_bits/32);
    random_words(instances[index].b._limbs, cgbn_default_bits/32);
  }
  return instances;
}

void add_words(uint32_t *r, uint32_t *x, uint32_t *y, uint32_t count) {
  int     index;
  int64_t sum=0;

  for(index=0;index<count;index++) {
    sum=sum+x[index]+y[index];
    r[index]=sum;
    sum=sum>>32;
  }
}

void sub_words(uint32_t *r, uint32_t *x, uint32_t *y, uint32_t count) {
  int     index;
  int64_t sum=0;

  for(index=0;index<count;index++) {
    sum=sum+x[index]-y[index];
    r[index]=sum;
    sum=sum>>32;
  }
}

void compare_words(uint32_t *x, uint32_t *y, uint32_t count) {
  for(int index=0;index<count;index++) {
    EXPECT_EQ(x[index], y[index]);
  }
}

// support routine to verify the GPU results using the CPU
void verify_results(instance_t *instances, uint32_t count) {
  uint32_t correct[cgbn_default_bits/32];
  
  for(int index=0;index<count;index++) {
    add_words(correct, instances[index].a._limbs, instances[index].b._limbs, cgbn_default_bits/32);
    compare_words(correct, instances[index].sum._limbs, cgbn_default_bits/32);
  }
}

// the actual kernel
__global__ void kernel_add(cgbn_error_report_t *report, instance_t *instances, uint32_t count) {
  int32_t instance;
  
  // decode an instance number from the blockIdx and threadIdx
  instance=(blockIdx.x*blockDim.x + threadIdx.x)/cgbn_default_tpi;
  if(instance>=count)
    return;

  context_t      bn_context(cgbn_report_monitor, report, instance);   // construct a context
  env_t          bn_env(bn_context.env<env_t>());                     // construct an environment for 1024-bit math
  env_t::cgbn_t  a, b, r;                                             // define a, b, r as 1024-bit bignums

  cgbn_load(bn_env, a, &(instances[instance].a));      // load my instance's a value
  cgbn_load(bn_env, b, &(instances[instance].b));      // load my instance's b value
  cgbn_add(bn_env, r, a, b);                           // r=a+b
  cgbn_store(bn_env, &(instances[instance].sum), r);   // store r into sum
}

TEST(CGBNTests, CGBNADD) {
  instance_t          *instances, *gpuInstances;
  cgbn_error_report_t *report;
  
  instances=generate_instances(INSTANCES);
  
  cudaSetDevice(0);
cudaDeviceReset();
  cudaMalloc((void **)&gpuInstances, sizeof(instance_t)*INSTANCES);
  cudaMemcpy(gpuInstances, instances, sizeof(instance_t)*INSTANCES, cudaMemcpyHostToDevice);
  cgbn_error_report_alloc(&report);
  
  // launch with 32 threads per instance, 128 threads (4 instances) per block
  kernel_add<<<(INSTANCES+3)/4, 128>>>(report, gpuInstances, INSTANCES);

  // error report uses managed memory, so we sync the device (or stream) and check for cgbn errors
  cudaDeviceSynchronize();
    
  // copy the instances back from gpuMemory
  cudaMemcpy(instances, gpuInstances, sizeof(instance_t)*INSTANCES, cudaMemcpyDeviceToHost);
  
  verify_results(instances, INSTANCES);
  
  // clean up
  free(instances);
  cudaFree(gpuInstances);
cudaDeviceReset();
  cgbn_error_report_free(report);
}
