#pragma once
#ifndef RAND_GEN_HPP
#define RAND_GEN_HPP

#include <random>

namespace fractal {

class rand_gen_std {
public:
  explicit rand_gen_std(unsigned long int seed = 1u);

  rand_gen_std(rand_gen_std const &) = delete;
  rand_gen_std(rand_gen_std &&) = delete;
  rand_gen_std &operator=(rand_gen_std const &) = delete;
  rand_gen_std &operator=(rand_gen_std &&) = delete;
  ~rand_gen_std() = default;

  // range: [a, b)
  double rand_double(double low = 0.0, double high = 1.0);
  // range: [a, b)
  float rand_float(float low = 0.0, float high = 1.0);
  // range: [a, b]
  int rand_int(int low, int high);
  // range: [a, b]
  unsigned int rand_uint(unsigned int low, unsigned int high);
  // range: [a, b]
  size_t rand_size_t(size_t low, size_t high);
  // range: [a, b]
  unsigned char rand_byte(unsigned char low = 0, unsigned char high = 255);
  // range: [a, b]
  bool random_bool();

private:
  std::default_random_engine re;
};

}

#endif //RAND_GEN_HPP
