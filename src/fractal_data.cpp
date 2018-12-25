#include "fractal_data.hpp"

#include <fstream>
#include <iostream>
#include <iomanip>
#include <memory>
#include <sstream>

namespace fractal {

flame_coeffs::flame_coeffs(unsigned int coeff_count, unsigned long int seed, pix color, bool randomize) :
    _coeff_count(coeff_count), _cf(std::make_unique<coeff[]>(coeff_count)) {
  std::cout << "Using random fractal coefficients with the seed `" << seed << "`\n\n";
  rand_gen_std rg(seed);

  for (unsigned int i = 0; i < _coeff_count; ++i) {
    if (rg.random_bool()) {
      contraction_mapping(_cf[i], rg);
    } else {
      _cf[i].ac = rg.rand_float(-1.5f, 1.5f);
      _cf[i].bc = rg.rand_float(-1.5f, 1.5f);
      _cf[i].cc = rg.rand_float(-1.5f, 1.5f);
      _cf[i].dc = rg.rand_float(-1.5f, 1.5f);
      _cf[i].ec = rg.rand_float(-1.5f, 1.5f);
      _cf[i].fc = rg.rand_float(-1.5f, 1.5f);
    }
    _cf[i].pa1 = rg.rand_float(-2.f, 2.f);
    _cf[i].pa2 = rg.rand_float(-2.f, 2.f);
    _cf[i].pa3 = rg.rand_float(-2.f, 2.f);
    _cf[i].pa4 = rg.rand_float(-2.f, 2.f);

    _cf[i].r = color.r;
    _cf[i].g = color.g;
    _cf[i].b = color.b;

    if (randomize) {
      _cf[i].r += rg.rand_byte(64, 255);
      _cf[i].g += rg.rand_byte(64, 255);
      _cf[i].b += rg.rand_byte(64, 255);
    }
  }
}

flame_coeffs::flame_coeffs(unsigned int coeff_count, std::string const &color_file, std::string const &coeff_file) :
    _coeff_count(coeff_count), _cf(std::make_unique<coeff[]>(coeff_count)) {
  std::cout << " Using fractal coeffs from `" << color_file << "` and `" << coeff_file << "`\n";
  std::vector<int> colors;
  std::ifstream clrfile(color_file);
  for (std::string line; std::getline(clrfile, line);) {
    std::istringstream iss(line);
    int r, g, b;
    if (!(iss >> r >> g >> b)) {
      break;
    }
    colors.push_back(r);
    colors.push_back(g);
    colors.push_back(b);
  }

  std::vector<float> coeffs;
  std::ifstream cffile(coeff_file);
  for (std::string line; std::getline(cffile, line);) {
    std::istringstream iss(line);
    float fa, fb, fc, fd, fe, ff, pa1, pa2, pa3, pa4;
    if (!(iss >> fa >> fb >> fc >> fd >> fe >> ff >> pa1 >> pa2 >> pa3 >> pa4)) {
      break;
    }
    coeffs.push_back(fa);
    coeffs.push_back(fb);
    coeffs.push_back(fc);
    coeffs.push_back(fd);
    coeffs.push_back(fe);
    coeffs.push_back(ff);
    coeffs.push_back(pa1);
    coeffs.push_back(pa2);
    coeffs.push_back(pa3);
    coeffs.push_back(pa4);
  }

  const int clr_sz = 3;
  const int cf_sz = 10;

  if (colors.size() / clr_sz < _coeff_count || coeffs.size() / cf_sz < _coeff_count) {
    std::cout << " There are less coefficients than specified in `eq_number`!\n";
    exit(EXIT_FAILURE);
  }

  // copy data
  for (size_t i = 0; i < _coeff_count; ++i) {
    _cf[i].ac = coeffs[i * cf_sz];
    _cf[i].bc = coeffs[i * cf_sz + 1];
    _cf[i].cc = coeffs[i * cf_sz + 2];
    _cf[i].dc = coeffs[i * cf_sz + 3];
    _cf[i].ec = coeffs[i * cf_sz + 4];
    _cf[i].fc = coeffs[i * cf_sz + 5];
    _cf[i].pa1 = coeffs[i * cf_sz + 6];
    _cf[i].pa2 = coeffs[i * cf_sz + 7];
    _cf[i].pa3 = coeffs[i * cf_sz + 8];
    _cf[i].pa4 = coeffs[i * cf_sz + 9];
  }

  for (size_t i = 0; i < _coeff_count; ++i) {
    _cf[i].r = static_cast<unsigned char>(colors[i * clr_sz]);
    _cf[i].g = static_cast<unsigned char>(colors[i * clr_sz + 1]);
    _cf[i].b = static_cast<unsigned char>(colors[i * clr_sz + 2]);
  }

}

coeff flame_coeffs::operator[](unsigned int index) const { return _cf[index]; }
coeff *flame_coeffs::data() const { return _cf.get(); }

void flame_coeffs::print_coeffs() const {
  std::cout << "Equations' coefficients:\n";
  int o1 = 10;
  for (unsigned int i = 0; i < _coeff_count; ++i) {
    std::cout << std::fixed << std::setprecision(2) << std::right <<
              std::setw(o1) << _cf[i].ac << std::setw(o1) << _cf[i].bc << std::setw(o1) << _cf[i].cc <<
              std::setw(o1) << _cf[i].dc << std::setw(o1) << _cf[i].ec << std::setw(o1) << _cf[i].fc <<
              std::setw(o1) << _cf[i].pa1 << std::setw(o1) << _cf[i].pa2 <<
              std::setw(o1) << _cf[i].pa3 << std::setw(o1) << _cf[i].pa4 << "\n";
  }

  int o2 = 5;
  std::cout << "Color values:\n";
  for (unsigned int i = 0; i < _coeff_count; ++i) {
    std::cout << std::fixed << std::setprecision(2) << std::right <<
              std::setw(o2) << static_cast<int>(_cf[i].r) <<
              std::setw(o2) << static_cast<int>(_cf[i].g) <<
              std::setw(o2) << static_cast<int>(_cf[i].b) << "\n";
  }
  std::cout << "\n";
}

void flame_coeffs::contraction_mapping(coeff &cf, rand_gen_std &rg) {
  float a, b, d, e;

  do {
    do {
      a = rg.rand_float();
      d = rg.rand_float(a * a, 1);
      if (rg.random_bool())
        d = -d;
    } while ((a * a + d * d) > 1);
    do {
      b = rg.rand_float();
      e = rg.rand_float(b * b, 1);
      if (rg.random_bool())
        e = -e;
    } while ((b * b + e * e) > 1);
  } while ((a * a + b * b + d * d + e * e) >
      (1 + (a * e - d * b) * (a * e - d * b)));

  cf.ac = a;
  cf.bc = b;
  cf.cc = rg.rand_float(-2.f, 2.f);
  cf.dc = d;
  cf.ec = e;
  cf.fc = rg.rand_float(-2.f, 2.f);
}

}
