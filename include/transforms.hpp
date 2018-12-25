#pragma once
#ifndef TRANSFORMS_HPP
#define TRANSFORMS_HPP

#include <cuda_runtime.h>

#include "fractal_data.hpp"

namespace fractal {
__host__ __device__ void transform_xy(float *new_x,
                                      float *new_y,
                                      unsigned short transform,
                                      coeff const *cf,
                                      float rand_d1_01,
                                      float rand_d2_01,
                                      bool rand_bool);
}

#endif // TRANSFORMS_HPP
