#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <GL/freeglut.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#define cudaCheckRet(ans) _cuda_check_ret((ans), __FILE__, __LINE__)
inline cudaError_t _cuda_check_ret(cudaError_t code, const char *file, int line,
                                   bool abort = false) {
  if (code != cudaSuccess) {
    std::cerr << "cudaCheckRet() failed at: " << file << ":" << line << ": "
              << cudaGetErrorString(code) << std::endl;
    if (abort) {
      exit(code);
    }
  }
  return code;
}
#define cudaCheckError() _cuda_check_error(__FILE__, __LINE__)
inline cudaError_t _cuda_check_error(const char *file, const int line,
                                     bool abort = false) {
  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    std::cerr << "cudaCheckError() failed at: " << file << ":" << line << ": "
              << cudaGetErrorString(code) << std::endl;
    if (abort) {
      exit(code);
    }
  }
  return code;
}

namespace cuda_info {
cudaDeviceProp devicePropertries(int device = 0);
bool deviceIsPresent();
} // namespace cuda_info

namespace cuda_gl {
cudaGraphicsResource *registerBuffer(GLuint buf);
void unregisterBuffer(cudaGraphicsResource *res);
void *map(cudaGraphicsResource *res);
void unmap(cudaGraphicsResource *res);
} // namespace cuda_gl

#endif // CUDA_UTILS_CUH
