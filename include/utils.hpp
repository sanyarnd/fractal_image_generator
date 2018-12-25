#pragma once
#ifndef UTILS_HPP
#define UTILS_HPP

#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <vector>

namespace fractal {

#define cudaCheckRet(ans) { _cuda_check_ret((ans), __FILE__, __LINE__); }
inline void _cuda_check_ret(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort)
      exit(code);
  }
}
#define cudaCheckError() { _cuda_check_error( __FILE__, __LINE__ ); }
inline void _cuda_check_error(const char *file, const int line) {
  cudaError err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n",
            file, line, cudaGetErrorString(err));
    exit(-1);
  }
}

// math constants
__host__ __device__ constexpr float pi() { return std::acos(-1.0f); }
__host__ __device__ constexpr float pi_two() { return 2 * pi(); }

// safe equivalent of % operator for doubles
__host__ __device__ inline float nonnan_mod(float a, float b) {
  return a - (static_cast<int>(a / b) * b);
}

// range s in [a, b] to t in [c, d]
__host__ __device__ inline float map_range(float a1, float a2, float b1, float b2, float s) {
  return b1 + (s - a1) * (b2 - b1) / (a2 - a1);
}

template<typename T, typename U>
__host__ __device__ constexpr T *mat_offset_elem(T *base, U row, U col, U row_size) {
  return base + row * row_size + col;
}

template<typename T, typename U>
constexpr T *mat_offset_elem(std::vector<T> &base, U row, U col, U row_size) {
  return &base[row * row_size + col];
}

}

#endif // UTILS_HPP
