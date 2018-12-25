#pragma once
#ifndef RENDERER_HPP
#define RENDERER_HPP

#include <cuda_runtime.h>
#include <mutex>
#include <vector>

#include "fractal_data.hpp"
#include "image.hpp"

namespace fractal {

image render_threads(unsigned int width,
                     unsigned int height,
                     unsigned short num_threads,
                     flame_coeffs const &fc,
                     unsigned long int seed,
                     unsigned int samples,
                     unsigned short iter_per_sample,
                     std::vector<unsigned short> const &transformations,
                     rect const &view,
                     unsigned short symmetry);
}

#endif //RENDERER_HPP
