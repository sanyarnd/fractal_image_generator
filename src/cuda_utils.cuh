#ifndef CUDA_UTILS_HPP
#define CUDA_UTILS_HPP

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#define cudaCheckRet(ans)                                                      \
  { _cuda_check_ret((ans), __FILE__, __LINE__); }
inline void _cuda_check_ret(cudaError_t code, const char *file, int line,
                            bool abort = true) {
  if (code != cudaSuccess) {
    std::cerr << "cudaCheckRet() failed at: " << file << ":" << line << ": "
              << cudaGetErrorString(code) << "\n";
    if (abort) {
      exit(code);
    }
  }
}
#define cudaCheckError()                                                       \
  { _cuda_check_error(__FILE__, __LINE__); }
inline void _cuda_check_error(const char *file, const int line) {
  cudaError code = cudaGetLastError();
  if (code != cudaSuccess) {
    std::cerr << "cudaCheckError() failed at: " << file << ":" << line << ": "
              << cudaGetErrorString(code) << "\n";
    exit(-1);
  }
}

cudaDeviceProp cudaDevicePropertries(int device = 0);

#endif // CUDA_UTILS_HPP
