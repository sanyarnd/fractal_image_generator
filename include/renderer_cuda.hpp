#pragma once
#ifndef RENDERER_CUDA_HPP
#define RENDERER_CUDA_HPP

#include <cuda_runtime.h>
#include <vector>

#include "fractal_data.hpp"
#include "image.hpp"

namespace fractal {

image render_cuda(unsigned int width,
                  unsigned int height,
                  flame_coeffs const &fc,
                  unsigned long int seed,
                  int samples,
                  short iter_per_sample,
                  std::vector<unsigned short> const &trans,
                  rect const &view,
                  short symmetry);
}

#endif //RENDERER_CUDA_HPP
