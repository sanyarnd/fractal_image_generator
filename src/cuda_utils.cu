#include "cuda_utils.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

namespace cuda_info {
cudaDeviceProp devicePropertries(int device) {
  cudaDeviceProp prop{};
  cudaCheckRet(cudaGetDeviceProperties(&prop, device));
  return prop;
}

bool deviceIsPresent() {
  int num_devices{};
  cudaGetDeviceCount(&num_devices);
  cudaError code = cudaGetLastError();
  return (code == cudaSuccess) && (num_devices > 0);
}
} // namespace cuda_info

namespace cuda_gl {
cudaGraphicsResource *registerBuffer(GLuint buf) {
  cudaGraphicsResource *res = nullptr;
  cudaCheckRet(
      cudaGraphicsGLRegisterBuffer(&res, buf, cudaGraphicsRegisterFlagsNone));
  return res;
}

void unregisterBuffer(cudaGraphicsResource *res) {
  cudaCheckRet(cudaGraphicsUnregisterResource(res));
}

void *map(cudaGraphicsResource *res) {
  if (cudaCheckRet(cudaGraphicsMapResources(1, &res)) != cudaSuccess) {
    return nullptr;
  }

  void *devPtr = nullptr;
  size_t size;
  if (cudaCheckRet(cudaGraphicsResourceGetMappedPointer(&devPtr, &size, res)) != cudaSuccess) {
    return nullptr;
  }
  return devPtr;
}

void unmap(cudaGraphicsResource *res) {
  cudaCheckRet(cudaGraphicsUnmapResources(1, (cudaGraphicsResource **) &res));
}
} // namespace cuda_gl
