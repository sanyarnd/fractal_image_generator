#include "rand_gen_thurst.hpp"

namespace fractal {

__host__ __device__ rand_gen_thurst::rand_gen_thurst(unsigned int seed) : re(seed) {}

__host__ __device__ double rand_gen_thurst::rand_double(double low, double high) {
  using Dist = thrust::random::uniform_real_distribution<double>;
  Dist uid{};
  re.discard(1);
  return uid(re, Dist::param_type{low, high});
}

float rand_gen_thurst::rand_float(float low, float high) {
  using Dist = thrust::random::uniform_real_distribution<float>;
  Dist uid{};
  re.discard(1);
  return uid(re, Dist::param_type{low, high});
}

__host__ __device__ int rand_gen_thurst::rand_int(int low, int high) {
  return static_cast<int>(ceilf(rand_float(low, high)));
}

__host__ __device__ unsigned int rand_gen_thurst::rand_uint(unsigned int low, unsigned int high) {
  return static_cast<unsigned int>(ceil(rand_double(low, high)));
}

__host__ __device__ size_t rand_gen_thurst::rand_size_t(size_t low, size_t high) {
  return static_cast<size_t>(ceil(rand_double(low, high)));
}

__host__ __device__ unsigned char rand_gen_thurst::rand_byte(unsigned char low, unsigned char high) {
  return static_cast<unsigned char>(ceilf(rand_float(low, high)));
}

__host__ __device__ bool rand_gen_thurst::random_bool() {
  return static_cast<bool>(rand_int(0, 1));
}

}
