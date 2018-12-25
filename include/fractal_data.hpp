#pragma once
#ifndef FRACTAL_DATA_HPP
#define FRACTAL_DATA_HPP

#include <memory>

#include "rand_gen_std.hpp"

namespace fractal {

struct coeff {
  float ac, bc, cc, dc, ec, fc;
  float pa1, pa2, pa3, pa4;
  unsigned char r, g, b;
};

struct pix {
  unsigned char r, g, b;
  unsigned int inc_cnt;
};

struct rect {
  float x, y, w, h;
  constexpr bool contains(float px, float py) const { return px >= x && px <= x + w && py >= y && py <= y + h; }
  constexpr float x_end() const { return x + w; }
  constexpr float y_end() const { return y + h; }
};

class flame_coeffs {
public:
  explicit flame_coeffs(unsigned int coeff_count, unsigned long int seed, pix color, bool randomize = true);
  explicit flame_coeffs(unsigned int coeff_count, std::string const &color_file, std::string const &coeff_file);

  constexpr unsigned int size() const { return _coeff_count; }
  coeff operator[](unsigned int index) const;
  coeff *data() const;

  void print_coeffs() const;

private:
  unsigned int _coeff_count;
  std::unique_ptr<coeff[]> _cf;

  /* from Paul Bourke */
  void contraction_mapping(coeff &cf, rand_gen_std &rg);
};

}

#endif //FRACTAL_DATA_HPP
