#pragma once
#ifndef FRACTAL_HPP
#define FRACTAL_HPP

#include <memory>
#include <string>
#include <cstring>
#include <iostream>

#include "fractal_data.hpp"

namespace fractal {

class image {
public:
  explicit image(unsigned int width, unsigned int height);
  explicit image(unsigned int width, unsigned int height, pix const*pixels);

  pix *data() const;
  pix *pixel(unsigned int row, unsigned int col);
  pix *pixel(unsigned int row, unsigned int col) const;

  ~image() = default;
  image(image const &other);
  image(image &&other) noexcept;
  image &operator=(image const &other);
  image &operator=(image &&other) noexcept;

  constexpr bool contains(size_t x, size_t y) const { return x < _width && y < _height; }
  static constexpr bool contains(unsigned int width,
                                 unsigned int height,
                                 unsigned int x,
                                 unsigned int y) { return x < width && y < height; }
  constexpr unsigned int width() const { return _width; }
  constexpr unsigned int height() const { return _height; }

  void gamma_log(float gamma);
  void save(std::string const &filename, unsigned short reduce_coeff = 1, bool invert_color = false);

private:
  std::unique_ptr<pix[]> _pixels;

  unsigned int _width;
  unsigned int _height;

  image anti_aliasing(unsigned short sample);
  void write_to_tiff(bool invert, std::string const &filename);
};

}

#endif // FRACTAL_HPP
