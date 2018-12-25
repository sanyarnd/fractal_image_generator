#pragma once
#ifndef PARAMS_HPP
#define PARAMS_HPP

#include <cxxopts.hpp>
#include <memory>
#include <string>

#include "fractal_data.hpp"
#include "rand_gen_std.hpp"

namespace fractal {

enum class renderer_type { THREADS, CUDA };

class params {
public:
  explicit params(int argc, char **argv);

  constexpr unsigned int height() const { return _res_y; }
  constexpr unsigned int width() const { return _res_x; }
  constexpr rect view() const { return _view; }
  constexpr pix color() const { return _rgb; }
  constexpr unsigned short fractal_equations_count() const { return _fractal_equations_count; }
  constexpr unsigned short supersampling() const { return _supersample; }
  constexpr unsigned int samples() const { return _samples; }
  constexpr unsigned short iter_per_sample() const { return _iterations; }
  constexpr bool invert_color() const { return _invert; }
  constexpr unsigned short symmetry() const { return _symmetry; }
  constexpr unsigned long int seed() const { return _seed; }
  constexpr unsigned short num_threads() const { return _num_threads; }
  constexpr float gamma() const { return _gamma; }
  constexpr renderer_type image_renderer() const { return _renderer; }
  constexpr bool randomize_color() const { return _rand_color; }
  constexpr bool dump_coefficients() const { return _dump_coeffs; }

  std::string output_file() const;
  std::vector<unsigned short> transformations() const;
  std::string color_file() const;
  std::string coeff_file() const;

private:
  unsigned short _supersample;
  unsigned int _res_y;
  unsigned int _res_x;
  rect _view;
  pix _rgb;
  unsigned short _fractal_equations_count;
  unsigned int _samples;
  unsigned short _iterations;
  bool _invert;
  unsigned short _symmetry;
  unsigned long int _seed;
  unsigned short _num_threads;
  float _gamma;
  std::string _file;
  std::vector<unsigned short> _transforms;
  std::string _color_file;
  std::string _coeff_file;
  renderer_type _renderer;
  bool _rand_color;
  bool _dump_coeffs;

  explicit params(cxxopts::ParseResult const &p);
  static cxxopts::ParseResult parse(int argc, char **argv);
  static void print_help(cxxopts::Options const &options);
};

// cxxopts parsers
void parse_value(const std::string &text, renderer_type &value);
void parse_value(const std::string &text, rect &value);
void parse_value(const std::string &text, pix &value);
}

#endif //PARAMS_HPP
