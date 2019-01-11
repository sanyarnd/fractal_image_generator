#include "cuda_utils.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

cudaDeviceProp cudaDevicePropertries(int device) {
  cudaDeviceProp prop{};
  cudaCheckRet(cudaGetDeviceProperties(&prop, device));
  return prop;
}
