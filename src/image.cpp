#include "image.hpp"

#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>
#include <iostream>
#include <tiffio.h>
#include <mutex>

#include "rand_gen_std.hpp"
#include "utils.hpp"

namespace fractal {

image::image(unsigned int width, unsigned int height) : _width(width), _height(height) {
  std::cout << "Creating an empty image " << _width << "x" << _height << '\n';
  std::cout << "Pix structure size is " << sizeof(pix) << "b\n";
  size_t sz = ((_height * _width * sizeof(pix))) / (1024 * 1024);
  std::cout << "Allocating " << sz << " MiB of RAM\n\n";
  _pixels = std::make_unique<pix[]>(_width * _height);
}

image::image(unsigned int width, unsigned int height, pix const*pixels) : _width(width), _height(height) {
  std::cout << "Creating an image " << _width << "x" << _height << " from existing buffer\n";
  std::cout << "Pix structure size is " << sizeof(pix) << "b\n";
  size_t sz = ((_height * _width * sizeof(pix))) / (1024 * 1024);
  std::cout << "Allocating " << sz << " MiB of RAM\n";
  _pixels = std::make_unique<pix[]>(_width * _height);
  std::cout << "Copying buffer...\n\n";
  std::copy(pixels, pixels + _width * _height, _pixels.get());
}

image::image(image const &other) : image(other._width, other._height) {
  std::cout << "Filling image data from existing...\n\n";
  std::copy(other.data(), other.data() + other.width() * other.height(), _pixels.get());
}

image::image(image &&other) noexcept
    : _pixels(std::move(other._pixels)), _width(other.width()), _height(other.height()) {}

image &image::operator=(image const &other) {
  if (&other != this) {
    *this = image(other);
  }
  return *this;
}

image &image::operator=(image &&other) noexcept {
  _pixels = std::move(other._pixels);
  _width = other.width();
  _height = other.height();

  return *this;
}

pix *image::data() const { return _pixels.get(); }
pix *image::pixel(unsigned int row, unsigned int col) { return mat_offset_elem(_pixels.get(), row, col, _width); }
pix *image::pixel(unsigned int row, unsigned int col) const { return mat_offset_elem(_pixels.get(), row, col, _width); }


void image::save(std::string const &filename, unsigned short reduce_coeff, bool invert_color) {
  if (reduce_coeff > 1) {
    image reduced = anti_aliasing(reduce_coeff);
    reduced.write_to_tiff(invert_color, filename);
  } else {
    write_to_tiff(invert_color, filename);
  }
}

void image::write_to_tiff(bool invert, std::string const &filename) {
  std::cout << "Saving file as `" << filename << "`\n\n";

  // tiff strip buffer
  auto raster = std::make_unique<char[]>(_width * 3);

  // opening file
  TIFF *output = TIFFOpen(filename.c_str(), "w");
  if (output == nullptr) {
    std::cerr << "Cannot open '" << filename << "' to write result!\n";
    exit(EXIT_FAILURE);
  }

  // write tiff
  TIFFSetField(output, TIFFTAG_IMAGEWIDTH, _width);
  TIFFSetField(output, TIFFTAG_IMAGELENGTH, _height);
  TIFFSetField(output, TIFFTAG_COMPRESSION, COMPRESSION_DEFLATE);
  TIFFSetField(output, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
  TIFFSetField(output, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
  TIFFSetField(output, TIFFTAG_BITSPERSAMPLE, 8);
  TIFFSetField(output, TIFFTAG_SAMPLESPERPIXEL, 3);
  for (unsigned short row = 0; row < _height; ++row) {
    // populate strip
    for (unsigned short col = 0; col < _width; ++col) {
      pix *px = pixel(row, col);
      raster[(col * 3)] = invert ? ~(px->r) : px->r;
      raster[(col * 3) + 1] = invert ? ~(px->g) : px->g;
      raster[(col * 3) + 2] = invert ? ~(px->b) : px->b;
    }

    // write strip to file
    if (TIFFWriteScanline(output, raster.get(),
                          static_cast<uint32>(row),
                          static_cast<uint16>(_width * 3)) != 1) {
      std::cerr << "Error while writing to the image!\n";
      TIFFClose(output);
      exit(EXIT_FAILURE);
    }
  }
  // finally closing file
  TIFFClose(output);
}

void image::gamma_log(float gamma) {
  std::cout << "Fixing image gamma...\n\n";
  auto normals = std::make_unique<float[]>(_width * _height);
  float max = 0.0;

  for (unsigned int row = 0; row < _height; ++row) {
    for (unsigned int col = 0; col < _width; ++col) {
      pix *px = pixel(row, col);

      if (px->inc_cnt != 0) {
        float *normal = mat_offset_elem(normals.get(), row, col, _width);
        *normal = log10f(px->inc_cnt);

        max = std::max(*normal, max);
      }
    }
  }

  for (unsigned int row = 0; row < _height; ++row) {
    for (unsigned int col = 0; col < _width; ++col) {
      pix *px = pixel(row, col);
      float *normal = mat_offset_elem(normals.get(), row, col, _width);
      // clamp?
      *normal /= max;
      float d = powf(*normal, (1.0f / gamma));
      px->r = (unsigned char) (px->r * d);
      px->g = (unsigned char) (px->g * d);
      px->b = (unsigned char) (px->b * d);
    }
  }
}

image image::anti_aliasing(unsigned short sample) {
  std::cout << "Reducing image in `" << sample << "` times with anti-aliasing algorithm...\n\n";

  unsigned int res_x = _width / sample;
  unsigned int res_y = _height / sample;
  image reduced(res_x, res_y);

  // simple grid algorithm anti-aliasing
  // numerical average of colors in a square region
  for (unsigned int y = 0; y < res_y; ++y) {
    for (unsigned int x = 0; x < res_x; ++x) {
      unsigned int R = 0;
      unsigned int G = 0;
      unsigned int B = 0;
      uint32_t count = 0;
      for (unsigned int sy = 0; sy < sample; ++sy) {
        for (unsigned int sx = 0; sx < sample; ++sx) {
          pix *px = pixel(y * sample + sy, x * sample + sx);
          R += px->r;
          G += px->g;
          B += px->b;
          count += px->inc_cnt;
        }
      }
      pix *px = reduced.pixel(y, x);
      px->r = (unsigned char) (R / (sample * sample));
      px->g = (unsigned char) (G / (sample * sample));
      px->b = (unsigned char) (B / (sample * sample));
      px->inc_cnt = count;
    }
  }

  return reduced;
}
}
