#include "rand_gen_std.hpp"

namespace fractal {
rand_gen_std::rand_gen_std(unsigned long int seed) : re(seed) {}

double rand_gen_std::rand_double(double low, double high) {
  using Dist = std::uniform_real_distribution<double>;
  static Dist uid{};
  return uid(re, Dist::param_type{low, high});
}

float rand_gen_std::rand_float(float low, float high) {
  using Dist = std::uniform_real_distribution<float>;
  static Dist uid{};
  return uid(re, Dist::param_type{low, high});
}

int rand_gen_std::rand_int(int low, int high) {
  using Dist = std::uniform_int_distribution<int>;
  static Dist uid{};
  return uid(re, Dist::param_type{low, high});
}

unsigned int rand_gen_std::rand_uint(unsigned int low, unsigned int high) {
  using Dist = std::uniform_int_distribution<unsigned int>;
  static Dist uid{};
  return uid(re, Dist::param_type{low, high});
}

size_t rand_gen_std::rand_size_t(size_t low, size_t high) {
  using Dist = std::uniform_int_distribution<size_t>;
  static Dist uid{};
  return uid(re, Dist::param_type{low, high});
}

unsigned char rand_gen_std::rand_byte(unsigned char low, unsigned char high) {
  return static_cast<unsigned char>(rand_int(low, high));
}

bool rand_gen_std::random_bool() {
  return static_cast<bool>(rand_int(0, 1));
}

}
