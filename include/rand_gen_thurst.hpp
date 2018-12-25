#pragma once
#ifndef RAND_GEN_THURST_HPP
#define RAND_GEN_THURST_HPP

#include <cuda_runtime.h>
#include <thrust/random.h>

namespace fractal {

class rand_gen_thurst {
public:
  __host__ __device__ explicit rand_gen_thurst(unsigned int seed = 1u);

  __host__ __device__ rand_gen_thurst(rand_gen_thurst const &) = delete;
  __host__ __device__ rand_gen_thurst(rand_gen_thurst &&) = delete;
  __host__ __device__ rand_gen_thurst &operator=(rand_gen_thurst const &) = delete;
  __host__ __device__ rand_gen_thurst &operator=(rand_gen_thurst &&) = delete;
  ~rand_gen_thurst() = default;

  // range: [a, b)
  __host__ __device__ double rand_double(double low = 0.0, double high = 1.0);
  // range: [a, b)
  __host__ __device__ float rand_float(float low = 0.0, float high = 1.0);
  // range: [a, b]
  __host__ __device__ int rand_int(int low, int high);
  // range: [a, b]
  __host__ __device__ unsigned int rand_uint(unsigned int low, unsigned int high);
  // range: [a, b]
  __host__ __device__ size_t rand_size_t(size_t low, size_t high);
  // range: [a, b]
  __host__ __device__ unsigned char rand_byte(unsigned char low = 0, unsigned char high = 255);
  // range: [a, b]
  __host__ __device__ bool random_bool();

//  // range: [a, b]
//  int rand_int(int low, int high);
//  // range: [a, b]
//  size_t rand_size_t(size_t low, size_t high);
//  // range: [a, b]
//  unsigned char rand_byte(unsigned char low = 0, unsigned char high = 255);
//  // range: [a, b]
//  bool random_bool() { return static_cast<bool>(rand_int(0, 1)); }
//

private:
  thrust::random::default_random_engine re;
};

}
#endif //RAND_GEN_THURST_HPP
